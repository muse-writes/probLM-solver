"""llama.cpp python interface for running local models."""

import math
from typing import Any

import numpy as np
import numpy.typing as npt
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Jinja2ChatFormatter

from problm_solver.adjust_probs import AdjustFn
from problm_solver.analysis.probabilities import prob_of_token, sample_from_logprobs
from problm_solver.analysis.tokenizer import LlamaTokenizer, TokenSequence
from problm_solver.data import LLMNextTokenData, LLMOutputData, LLMTokenData


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
    ) -> LLMNextTokenData | None:
        """Query next M most likely tokens in current context.

        Query the model for a single token, output the top M most likely tokens
        and their logarithmic probabilities. Returns ``None`` when the model
        generates an EOS token, indicated by an empty ``top_logprobs`` list in
        the response (llama_cpp does not include EOS in logprob output).

        :param context_tokens: The current context as a list of integer token IDs.
        :param n_tokens: Number of top candidate tokens (M) to return.
        :returns: ``LLMNextTokenData`` containing the top-M token → log-prob
            mapping, or ``None`` if EOS was generated.
        """
        output = self._llm.create_completion(
                context_tokens,
                max_tokens=1,
                logprobs=n_tokens,
        )
        top_logprobs_list: list = output['choices'][0]['logprobs']['top_logprobs']
        if not top_logprobs_list:
            return None
        return LLMNextTokenData(
            prompt=self.context,
            output_vec=context_tokens,
            top_m_tokens=top_logprobs_list[0]
        )


    def get_tokenizer(self) -> LlamaTokenizer:
        """Return a LlamaTokenizer backed by this model's vocabulary."""
        return LlamaTokenizer(self._llm)


    def _format_chat_prompt(self) -> list[int]:
        """Apply the model's chat template to ``self.context`` and return token IDs.

        Constructs a :class:`Jinja2ChatFormatter` from the chat template embedded
        in the model's GGUF metadata, applies it to ``self.context`` as a
        user-role message, and tokenises the resulting prompt string to a list
        of integer token IDs. This list is the initial context passed to
        ``generate_adjusted()``.
        """
        chat_template = self._llm.metadata['tokenizer.chat_template']
        eos_token = self._llm.detokenize([self._llm.token_eos()]).decode('utf-8', errors='ignore')
        bos_token = self._llm.detokenize([self._llm.token_bos()]).decode('utf-8', errors='ignore')
        formatter = Jinja2ChatFormatter(
            template=chat_template,
            eos_token=eos_token,
            bos_token=bos_token,
        )
        result = formatter(messages=[{'role': 'user', 'content': self.context}])
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
        prev_probs: list[float] = []

        for _ in range(max_tokens):
            next_token_data = self.query_log_probs_next_token(context, n_tokens)
            if next_token_data is None:
                break
            adjusted = adjust_fn(next_token_data.top_m_tokens, prev_probs)
            token_str = sample_from_logprobs(adjusted)
            token_ids = self._llm.tokenize(
                token_str.encode('utf-8'), add_bos=False, special=False,
            )
            if not token_ids or eos_id in token_ids:
                break
            prev_probs.append(prob_of_token(token_str, adjusted))
            context.extend(token_ids)

        generated_ids = context[prompt_length:]
        response = self._llm.detokenize(generated_ids).decode('utf-8', errors='replace')
        return LLMOutputData(prompt=self.context, data=np.array([response], dtype=str))
