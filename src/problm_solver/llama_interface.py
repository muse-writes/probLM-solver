"""llama.cpp python interface for running local models."""

import math
from typing import Any

import numpy as np
import numpy.typing as npt
from llama_cpp import Llama, LlamaRAMCache, LlamaState
from llama_cpp.llama_chat_format import Jinja2ChatFormatter

from problm_solver.adjust_probs import AdjustFn, GenerationContext
from problm_solver.analysis.probabilities import prob_of_token, sample_from_logprobs
from problm_solver.analysis.tokenizer import LlamaTokenizer, TokenSequence
from problm_solver.data import LLMNextTokenData, LLMOutputData, LLMTokenData


class ModelInstance:
    """Keeps a model instance and its context, with methods for querying the Llama instance."""

    def __init__(self, fname: str, context: str, logits_all: bool = False) -> None:
        """Initialize Llama instance and store context.

        The RAM cache capacity is derived from the model's own metadata so
        that it can hold four full KV-cache states. One state covers the
        entire context window for all layers and KV heads at fp16 precision:

            bytes_per_state = n_ctx × 2 × n_layers × n_kv_heads × head_dim × 2

        Four states comfortably accommodates the save/restore pattern used by
        :class:`~problm_solver.adjust_probs.SamplePowerDist`: the saved
        pre-branch snapshot, the current working state, and spare capacity
        for shared prefix entries.

        :param fname: absolute path of the model .gguf file.
        :param context: query that the model is initialised with.
        :param logits_all: whether or not probability logging is necessary in the Llama instance.
        """
        self._llm = Llama(model_path=fname, n_ctx=2048, logits_all=logits_all)

        arch = self._llm.metadata['general.architecture']
        n_layers = int(self._llm.metadata[f'{arch}.block_count'])
        n_kv_heads = int(self._llm.metadata[f'{arch}.attention.head_count_kv'])
        n_heads = int(self._llm.metadata[f'{arch}.attention.head_count'])
        head_dim = int(self._llm.metadata[f'{arch}.embedding_length']) // n_heads
        bytes_per_state = self._llm.n_ctx() * 2 * n_layers * n_kv_heads * head_dim * 2

        self._cache = LlamaRAMCache(capacity_bytes=4 * bytes_per_state)
        self._llm.set_cache(self._cache)
        self.context = context


## -- Methods for querying the LLM. -- ##

    def query_n_times(self, n: int) -> npt.NDArray[Any]:
        """Query the LLM with the same context N times, return the output.

        :param n: number of times N to query the Llama instance.
        :returns: an array of response strings.
        """
        return np.array([self.query() for _ in range(n)], dtype=str)


    def query(self) -> str:
        """Query the LLM once.

        :returns: the response string.
        """
        output = self._llm.create_chat_completion(
            messages=[{'role': 'user', 'content': self.context}],
            max_tokens=512,
        )
        return output['choices'][0]['message']['content']


    def generate_data(self, n_samples: int) -> LLMOutputData:
        """Generate data by querying the LLM `n_samples` times.

        :param n_samples: the number of times to query the Llama instance.
        :returns: A data container with all responses and the prompt.
        """
        data = self.query_n_times(n_samples)
        return LLMOutputData(prompt=self.context, data=data)


    def query_log_probs(self) -> LLMTokenData:
        """Query the model and return the response as tokens with probabilities.

        Calls the model with ``logprobs=True``, which causes the API to return
        the model's own BPE tokenization of the response alongside the
        log-probability of each token at its position. Log-probabilities are
        converted to probabilities via ``exp``.

        :returns: A data container holding the prompt, alongside the tokens and their
            probabilities.
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


    def query_branch(self, context_tokens: list[int], max_tokens: int) -> float:
        """Generate a branch of up to max_tokens and return its total log-probability.

        Makes a single ``create_completion`` call with ``logprobs=1`` so that
        llama_cpp populates ``token_logprobs`` — the log-probability of each
        actually-sampled token under the full model vocabulary. The values are
        summed to give the branch log-probability.

        If EOS is reached before ``max_tokens``, the shorter response is
        returned with no penalty, matching the early-EOS behaviour of the
        previous token-by-token loop.

        :param context_tokens: The current context as a list of integer token IDs.
        :param max_tokens: Maximum number of tokens to generate in the branch.
        :returns: Sum of per-token log-probabilities for all generated tokens,
            or ``0.0`` if the model generates EOS immediately.
        """
        output = self._llm.create_completion(
            context_tokens,
            max_tokens=max_tokens,
            logprobs=1,
        )
        token_logprobs: list[float | None] = (
            output['choices'][0]['logprobs']['token_logprobs']
        )
        return float(sum(lp for lp in token_logprobs if lp is not None))


## -- Miscellaneous -- ##

    def reset_state(self) -> None:
        """Reset model state. Low-level Llama API."""
        self._llm.reset()


    def eval_tokens(self, tokens: list[int]) -> None:
        """Evaluate tokens. Low-level Llama API."""
        self._llm.eval(tokens)


    def save_live_state(self) -> LlamaState:
        """Return LlamaState object."""
        self._llm.save_state()


    def load_live_state(self, state: LlamaState) -> None:
        """Restore LlamaState object."""
        self._llm.load_state(state)


    def get_tokenizer(self) -> LlamaTokenizer:
        """Exposes a LlamaTokenizer backed by this model's vocabulary.

        :returns: The tokenizer instance, defined in ``problm_solver.analysis.tokenizer``.
        """
        return LlamaTokenizer(self._llm)


    def _format_chat_prompt(self) -> list[int]:
        """Apply the model's chat template to ``self.context`` and return token IDs.

        Constructs a :class:`Jinja2ChatFormatter` from the chat template embedded
        in the model's GGUF metadata, applies it to ``self.context`` as a
        user-role message, and tokenises the resulting prompt string to a list
        of integer token IDs. This list is the initial context passed to
        ``generate_adjusted()``.

        :returns: A tokenized prompt.
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

## -- Adjusting probabilities -- ##

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
        :param adjust_fn: Callable that receives a ``GenerationContext`` and
            returns a modified ``dict[str, float]`` of token log-probabilities.
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
            ctx = GenerationContext(
                token_probs=next_token_data.top_m_tokens,
                prev_probs=list(prev_probs),
                context_tokens=list(context),
                query_next=lambda ctx_ids: (
                    r.top_m_tokens
                    if (r := self.query_log_probs_next_token(ctx_ids, n_tokens)) is not None
                    else None
                ),
                query_branch=lambda ctx_ids, depth: self.query_branch(ctx_ids, depth),
                tokenize_token=lambda s: self._llm.tokenize(
                    s.encode('utf-8'), add_bos=False, special=False,
                ),
            )
            adjusted = adjust_fn(ctx)
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
