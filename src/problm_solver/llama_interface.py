"""llama.cpp python interface for running local models."""

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
from llama_cpp import Llama

from problm_solver.analysis.probabilities import sample_from_logprobs
from problm_solver.analysis.tokenizer import LlamaTokenizer, TokenSequence
from problm_solver.data import LLMNextTokenData, LLMOutputData, LLMTokenData

# Callable that receives a top-M {token: log_prob} dict and returns a modified
# log-prob dict. Values need not be normalised; renormalisation is applied by
# sample_from_logprobs before sampling.
AdjustFn = Callable[[dict[str, float]], dict[str, float]]


class ModelInstance:
    """Model class."""

    def __init__(self, fname: str, context: str, logits_all: bool = False) -> None:
        """Init method."""
        self._llm = Llama(model_path=fname, n_ctx=2048, logits_all=logits_all)
        self.context = context


    def query_n_times(self, n: int) -> npt.NDArray[Any]:
        """Query the LLM with the same context N times, return the output."""
        return np.array([self.query() for _ in range(n)], dtype=str)


    def query(self) -> str:
        """Query the LLM once."""
        output = self._llm.create_chat_completion(
            messages=[{'role': 'user', 'content': self.context}],
            max_tokens=512,
        )
        return output['choices'][0]['message']['content']


    def generate_data(self, n_samples: int) -> LLMOutputData:
        """Generate data by querying the LLM `n_samples` times."""
        data = self.query_n_times(n_samples)
        return LLMOutputData(prompt=self.context, data=data)


    def query_log_probs(self) -> LLMTokenData:
        """Query the model and return the response as tokens with probabilities.

        Calls the model with ``logprobs=True``, which causes the API to return
        the model's own BPE tokenization of the response alongside the
        log-probability of each token at its position. Log-probabilities are
        converted to probabilities via ``exp``.
        """
        output = self._llm.create_chat_completion(
            messages=[{'role': 'user', 'content': self.context}],
            max_tokens=512,
            logprobs=True,
            top_logprobs=1,
        )
        logprob_content = output['choices'][0]['logprobs']['content']
        tokens: TokenSequence = [entry['token'] for entry in logprob_content]
        probs: list[float] = [math.exp(entry['logprob']) for entry in logprob_content]
        return LLMTokenData(prompt=self.context, tokens=tokens, probs=probs)


    def query_log_probs_next_token(
        self,
        context_tokens: list[int],
        n_tokens: int,
    ) -> LLMNextTokenData:
        """Query next M most likely tokens in current context.

        Query the model for a single token, output the top M most likely tokens
        and their logarithmic probabilities.

        :param context_tokens: The current context as a list of integer token IDs.
        :param n_tokens: Number of top candidate tokens (M) to return.
        :returns: ``LLMNextTokenData`` containing the top-M token → log-prob mapping.
        """
        output = self._llm.create_completion(
                context_tokens,
                max_tokens=1,
                logprobs=True,
                top_logprobs=n_tokens
        )
        top_logprobs: dict = output['choices'][0]['logprobs']['top_logprobs'][0]
        return LLMNextTokenData(
            prompt=self.context,
            output_vec=context_tokens,
            top_m_tokens=top_logprobs
        )


    def get_tokenizer(self) -> LlamaTokenizer:
        """Return a LlamaTokenizer backed by this model's vocabulary."""
        return LlamaTokenizer(self._llm)


    def _format_chat_prompt(self) -> list[int]:
        """Apply the model's chat template to ``self.context`` and return token IDs.

        Calls the same internal chat handler as ``create_chat_completion()`` to
        ensure identical prompt formatting, then tokenises the result to a list
        of integer token IDs. This list is the initial context passed to
        ``generate_adjusted()``.

        .. note::
            Accesses ``self._llm._chat_handler``, a private attribute of
            ``llama_cpp.Llama``. This is intentional: it is the only way to
            apply the model's chat template without triggering inference.
        """
        result = self._llm._chat_handler(
            llama=self._llm,
            messages=[{'role': 'user', 'content': self.context}],
        )
        return self._llm.tokenize(
            result.prompt.encode('utf-8'),
            add_bos=False,
            special=True,
        )


    def generate_adjusted(
        self,
        n_tokens: int,
        adjust_fn: AdjustFn,
        max_tokens: int,
    ) -> LLMOutputData:
        """Generate a response token-by-token with adjusted next-token probabilities.

        At each step the top ``n_tokens`` candidate next tokens are retrieved
        and passed to ``adjust_fn``, which may modify the log-probability
        distribution. A single token is then sampled from the adjusted
        distribution and appended to the context before the next step.

        :param n_tokens: Number of top candidate tokens to retrieve at each
            step (M).
        :param adjust_fn: Callable that receives a ``dict[str, float]`` mapping
            token strings to log-probabilities and returns a modified mapping.
            Values do not need to be normalised.
        :param max_tokens: Maximum number of tokens to generate.
        :returns: ``LLMOutputData`` containing the prompt and the generated
            response string.
        """
        context = self._format_chat_prompt()
        prompt_length = len(context)
        eos_id = self._llm.token_eos()

        for _ in range(max_tokens):
            next_token_data = self.query_log_probs_next_token(context, n_tokens)
            adjusted = adjust_fn(next_token_data.top_m_tokens)
            token_str = sample_from_logprobs(adjusted)
            token_ids = self._llm.tokenize(
                token_str.encode('utf-8'), add_bos=False, special=False,
            )
            if not token_ids or eos_id in token_ids:
                break
            context.extend(token_ids)

        generated_ids = context[prompt_length:]
        response = self._llm.detokenize(generated_ids).decode('utf-8', errors='replace')
        return LLMOutputData(prompt=self.context, data=np.array([response], dtype=str))
