"""llama.cpp python interface for running local models."""

import copy
import logging
from inspect import isclass
from typing import Any

import numpy as np
import numpy.typing as npt
from llama_cpp import Llama, LlamaRAMCache, LlamaState
from llama_cpp.llama_chat_format import Jinja2ChatFormatter
from tqdm import tqdm

from problm_solver.adjust_probs import AdjustFn, GenerationContext
from problm_solver.analysis.probabilities import prob_of_token, sample_from_logprobs
from problm_solver.data import (
    Hyperparams,
    LLMNextTokenData,
    LLMOutputData,
    LLMOutputDataFull,
    LLMTokenData,
)
from problm_solver.utils import _as_rng

# -- Module-wide setup -- #

_logger = logging.getLogger(__name__)

ADEQUATE_TOPK = 30
ADEQUATE_TOPP = 0.8


# -- Main model instance -- #

class ModelInstance:
    """Keeps a model instance and its context, with methods for querying the Llama instance."""

    def __init__(self, fname: str, context: str, n_ctx: int = 4096, logits_all: bool = False) -> None:
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
        :param n_ctx: context window size in tokens. Must be large enough to
            hold the formatted prompt plus ``max_tokens`` of generated output.
            Defaults to 4096, which comfortably fits MATH500 problems
            (~300 prompt tokens) and up to 2048 generated tokens.
        :param logits_all: whether or not probability logging is necessary in the Llama instance.
        """
        self._llm = Llama(model_path=fname, n_ctx=n_ctx, logits_all=logits_all, verbose=False)
        _logger.info('Model %r loaded.', fname)

        arch = self._llm.metadata['general.architecture']
        n_layers = int(self._llm.metadata[f'{arch}.block_count'])
        n_kv_heads = int(self._llm.metadata[f'{arch}.attention.head_count_kv'])
        n_heads = int(self._llm.metadata[f'{arch}.attention.head_count'])
        head_dim = int(self._llm.metadata[f'{arch}.embedding_length']) // n_heads
        bytes_per_state = self._llm.n_ctx() * 2 * n_layers * n_kv_heads * head_dim * 2

        self._cache = LlamaRAMCache(capacity_bytes=4 * bytes_per_state)
        self._llm.set_cache(self._cache)
        self.context = context
        self._initial_context_length: int = len(self.context)


## -- Methods for querying the LLM. -- ##

    def query_n_times(self, n: int) -> npt.NDArray[Any]:
        """Query the LLM with the same context N times, return the output.

        :param n: number of times N to query the Llama instance.
        :returns: an array of response strings.
        """
        return np.array([self.query() for _ in range(n)], dtype=str)


    def query(self, rng: np.random.Generator | int | None = None) -> str:
        """Query the LLM once.

        :returns: the response string.
        """
        max_tokens = 512 # TODO(Clio): Remove hard-coded maximum here.
        prompt_tokens = self._format_chat_prompt()
        self._llm.reset()
        self._llm.eval(prompt_tokens)
        tokens = []
        rng = _as_rng(rng)
        for _ in range(max_tokens):
            logprobs = self._log_softmax(self._llm.scores[self._llm.n_tokens - 1])
            next_id = int(np.argmax(logprobs + rng.gumbel(size=len(logprobs))))
            if next_id == self._llm.token_eos():
                break
            tokens.append(next_id)
            self._llm.eval([next_id])
        return self._llm.detokenize(tokens).decode('utf-8')


    def generate_data(self, n_samples: int) -> LLMOutputData:
        """Generate data by querying the LLM `n_samples` times.

        :param n_samples: the number of times to query the Llama instance.
        :returns: A data container with all responses and the prompt.
        """
        data = self.query_n_times(n_samples)
        return LLMOutputData(prompt=self.context, data=data)


    def query_log_probs(self, rng: np.random.Generator | int | None = None) -> LLMTokenData:
        """Query the model and return the response as tokens with probabilities.

        Evaluates the formatted prompt in a single forward pass, then samples
        tokens autoregressively using the Gumbel-max trick until EOS or
        ``max_tokens`` steps are reached.  Each sampled token is decoded to a
        string via :meth:`_tokens_as_strings` and its probability is computed
        as ``exp(log_softmax(logits)[token_id])``.

        :returns: A data container holding the prompt, alongside the tokens and
            their probabilities.
        """
        max_tokens = 512  # TODO(Clio): Remove hard-coded maximum here.
        prompt_tokens = self._format_chat_prompt()
        self._llm.reset()
        self._llm.eval(prompt_tokens)
        tokens: TokenSequence = []
        probs: list[float] = []
        eos_id = self._llm.token_eos()
        rng = _as_rng(rng)
        for _ in range(max_tokens):
            logprobs = self._log_softmax(self._llm.scores[self._llm.n_tokens - 1])
            next_id = int(np.argmax(logprobs + rng.gumbel(size=len(logprobs))))
            if next_id == eos_id:
                break
            tokens.append(self._tokens_as_strings([next_id])[0])
            probs.append(float(np.exp(logprobs[next_id])))
            self._llm.eval([next_id])
        return LLMTokenData(prompt=self.context, tokens=tokens, probs=probs)


    def query_log_probs_next_token(
        self,
        context_tokens: list[int],
        n_tokens: int,
    ) -> LLMNextTokenData:
        """Return the top-M most likely next tokens and their log-probabilities.

        Resets the model state, evaluates ``context_tokens`` in a single
        forward pass, applies log-softmax to the last-position logits, and
        returns the top ``n_tokens`` candidates via :meth:`_top_k_from_logprobs`.

        EOS detection is the caller's responsibility: the EOS token will
        appear naturally in the returned distribution when the model prefers
        it, and :meth:`generate_adjusted` stops the loop after the EOS token
        ID is sampled.

        :param context_tokens: The current context as a list of integer token IDs.
        :param n_tokens: Number of top candidate tokens to return.
        :returns: ``LLMNextTokenData`` containing the top-K token → log-prob
            mapping.
        """
        self._llm.reset()
        self._llm.eval(context_tokens)
        logprobs = self._log_softmax(self._llm.scores[self._llm.n_tokens - 1])
        top_k = self._top_k_from_logprobs(logprobs, n_tokens)
        return LLMNextTokenData(
            prompt=self.context,
            output_vec=context_tokens,
            top_k_tokens=top_k
        )


    def query_branch(
        self,
        context_tokens: list[int],
        max_tokens: int,
        rng: np.random.Generator | int | None = None
    ) -> float:
        """Generate a branch of up to max_tokens and return its total log-probability.

        Evaluates ``context_tokens`` in a single forward pass, then immediately
        snapshots the resulting KV cache and logit state via
        :meth:`save_live_state`.  That snapshot is restored via
        :meth:`load_live_state` before the generation loop begins, which
        guarantees a clean branch start regardless of any side-effects from
        the save itself, and lays the groundwork for future multi-branch calls
        where the context prefix need only be evaluated once.

        At each step ``scores[n_tokens - 1]`` is the logit row for the most
        recently decoded position (the ``[n_past : n_past + n_tokens]`` slice
        written by :meth:`eval` with ``logits_all=True``).  A token is sampled
        via the Gumbel-max trick — equivalent to ancestral sampling from the
        full-vocabulary categorical distribution — and its log-probability is
        accumulated.  Generation stops at EOS or after ``max_tokens`` steps.

        :param context_tokens: The current context as a list of integer token IDs.
        :param max_tokens: Maximum number of tokens to generate in the branch.
        :returns: Sum of per-token log-probabilities for all generated tokens,
            or ``0.0`` if EOS is sampled on the first step.
        """
        self._llm.reset()
        self._llm.eval(context_tokens)

        # Snapshot the KV cache and logits immediately after evaluating the
        # context.  Restoring this state before generation ensures the branch
        # always starts from the clean post-context position and is not
        # affected by any internal bookkeeping inside save_live_state.
        pre_branch_state = self.save_live_state()
        self.load_live_state(pre_branch_state)

        eos_id = self._llm.token_eos()
        total_log_prob = 0.0
        rng = _as_rng(rng)

        for _ in range(max_tokens):
            # scores[n_tokens - 1] is the most recently decoded logit row,
            # valid for logits_all=True (filled by eval's n_past slice).
            logprobs = self._log_softmax(self._llm.scores[self._llm.n_tokens - 1])

            # Gumbel-max trick: argmax(log p + Gumbel(0,1)) is equivalent to
            # drawing from categorical(softmax(log p)) without materialising
            # the full probability vector.
            next_id = int(np.argmax(logprobs + rng.gumbel(size=len(logprobs))))

            if next_id == eos_id:
                break

            total_log_prob += float(logprobs[next_id])
            self._llm.eval([next_id])

        return total_log_prob


## -- Miscellaneous -- ##

### -- Expose lower level Llama API -- ###

    def save_live_state(self) -> LlamaState:
        """Return LlamaState object."""
        return self._llm.save_state()


    def load_live_state(self, state: LlamaState) -> None:
        """Restore LlamaState object."""
        self._llm.load_state(state)


### -- Tokens, formatting, and changing context -- ###

    def change_context(self, ctx: str) -> None:
        """Update the provided context.

        Making this change requires resetting the Llama instance state.

        :param ctx: New context string provided to the model.
        """
        self.context = ctx
        self._llm.reset()
        self._initial_context_length = len(self.context)


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


    def _tokens_as_strings(self, token_ids: list[int]) -> list[str]:
        """Use repeated calls to Llama.detokenize to return a list of token strings."""
        return [
            self._llm.detokenize([tid], special=True).decode('utf-8', errors='replace')
            for tid in token_ids
        ]


    @staticmethod
    def _log_softmax(logits: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
        """Apply numerically stable log-softmax to a 1-D logits vector.

        Subtracts the maximum logit before exponentiation to prevent float
        overflow (common with raw LLM logits which can exceed ±300), then
        uses the log-sum-exp identity:

        .. code-block:: text

            log_softmax(x_i) = (x_i − max x) − log Σ_j exp(x_j − max x)

        The result satisfies ``exp(result).sum() ≈ 1`` and all values are ≤ 0.

        :param logits: 1-D array of raw model logits for the full vocabulary.
        :returns: 1-D float64 array of log-probabilities.
        """
        x = logits.astype(np.float64)
        shifted = x - x.max()
        return shifted - np.log(np.exp(shifted).sum())


    @staticmethod
    def _top_k_ids_from_logprobs(
        logprobs: npt.NDArray[np.float64],
        n: int,
    ) -> list[tuple[int, float]]:
        """Return top-n ``(token_id, logprob)`` pairs ordered descending by logprob."""
        n = min(n, len(logprobs))
        top_indices = np.argpartition(logprobs, -n)[-n:]
        top_indices = top_indices[np.argsort(logprobs[top_indices])[::-1]]
        return [(int(idx), float(logprobs[idx])) for idx in top_indices]

    def _top_k_from_logprobs(
        self,
        logprobs: npt.NDArray[np.float64],
        n: int,
    ) -> dict[str, float]:
        """Return the top-n tokens and their log-probabilities from a full-vocabulary array.

        Uses ``numpy.argpartition`` for O(V) selection, then sorts the
        selected indices so the returned dict is ordered from highest to
        lowest log-probability (Python 3.7+ dict insertion order).
        Token IDs are converted to strings via :meth:`_tokens_as_strings`.

        If ``n`` exceeds the vocabulary size it is silently clamped so that
        all tokens are returned.

        :param logprobs: 1-D log-probability array over the full vocabulary,
            as returned by :meth:`_log_softmax`.
        :param n: Number of top candidates to return.
        :returns: ``{token_string: log_prob}`` for the *n* most probable
            tokens, ordered from highest to lowest log-probability.
        """
        n = min(n, len(logprobs))
        top_indices = np.argpartition(logprobs, -n)[-n:]
        top_indices = top_indices[np.argsort(logprobs[top_indices])[::-1]]
        token_strings = self._tokens_as_strings(top_indices.tolist())
        return {s: float(logprobs[idx]) for s, idx in zip(token_strings, top_indices, strict=True)}


    def _candidates_from_logprobs(
        self,
        logprobs: npt.NDArray[np.float64],
        top_k: int,
        top_p: float
    ) -> dict[str, float]:
        """Return the most probable tokens using top-k and top-p params from the vocabulary.

        Similar to ``_top_k_from_logprobs`` but after acquiring the top-k log-probabilities, it
        only returns the smallest number of tokens whose probabilities sum to more than top-p.
        If there are more than ``top_k`` tokens required to make up the ``top_p`` probabilities, it
        just returns all top-k tokens.

        :param logprobs: 1-D log-probability array over the full vocabulary,
            as returned by :meth:`_log_softmax`.
        :param top_k: Number of top candidates to return.
        :param top_p: Total probability of candidate tokens.
        :returns: ``{token_string: log_prob}`` for the *n* most probable
            tokens, ordered from highest to lowest log-probability.
        """
        top_k = min(top_k, len(logprobs))
        top_indices = np.argpartition(logprobs, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(logprobs[top_indices])[::-1]]
        token_strings = self._tokens_as_strings(top_indices.tolist())
        tokens_out = {}
        total_prob = 0
        for s, idx in zip(token_strings, top_indices, strict=True):
            tokens_out[s] = float(logprobs[idx])
            total_prob += np.exp(logprobs[idx])
            if total_prob > top_p:
                break
        return tokens_out


## -- Adjusting probabilities -- ##

    def generate_adjusted(
        self,
        top_k: int,
        top_p: float,
        adjust_fn: AdjustFn,
        max_tokens: int,
        *,
        alpha: float = 1.0,
        sampling_method: str | None = None,
        branch_sampler: str | None = None,
    ) -> LLMOutputDataFull:
        """Generate a response token-by-token with adjusted next-token probabilities.

        At each step the top ``top_k`` candidate next tokens are retrieved
        and passed to ``adjust_fn``, which may modify the log-probability
        distribution. A single token is then sampled from the adjusted
        distribution and appended to the context before the next step.

        :param top_k: Number of top candidate tokens to retrieve at each
            step.
        :param top_p: Threshold total probability of retrieved tokens.
        :param adjust_fn: Callable that receives a ``GenerationContext`` and
            returns a modified ``dict[str, float]`` of token log-probabilities.
            Values do not need to be normalised.
        :param max_tokens: Maximum number of tokens to generate.
        :returns: ``LLMOutputDataFull`` containing the model's response,
            candidate tokens at each step, and logprobs.
        """
# Hyperparam and sampling setup.
        if sampling_method is None:
            if isclass(adjust_fn):
                sampling_method = adjust_fn.__class__.__name__
            else:
                sampling_method = getattr(adjust_fn, '__name__', type(adjust_fn).__name__)
        if top_p >= 1.:
            candidate_generator = self._top_k_from_logprobs
        elif top_p > 0.:
            def candidate_generator(logprobs: npt.NDArray[np.float64], top_k: int) -> dict[str, float]:
                return self._candidates_from_logprobs(logprobs, top_k, top_p)
        else:
            msg = f'top_p must be in (0, 1], got {top_p}'
            raise ValueError(msg)

# Parameter warnings.
        _logger.info('Generation with adjusted probabilities started')
        if top_k < ADEQUATE_TOPK:
            _logger.warning(
                'top-k set to %d < 30. Model may struggle to sample rare vocab.', top_k
            )
        if top_p < ADEQUATE_TOPP:
            _logger.warning(
                'top-p set to %.4f < 0.8. Model may act excessively greedy.', top_p
            )


# Data storage variables setup.
        prev_probs: list[float] = []
        response_prob_tokens: list[str] = []
        response_prob_values: list[float] = []
        response_topk_dists: list[dict[str, float]] = []

# LLM state setup.
        eos_id = self._llm.token_eos()
        context = self._format_chat_prompt()
        self._llm.reset()
        self._llm.eval(context)

# Main generation loop.
        for step in tqdm(range(max_tokens), desc='generate_adjusted', unit='tok'):
            logprobs = self._log_softmax(self._llm.scores[self._llm.n_tokens - 1])
            top_k_lp = candidate_generator(logprobs, top_k)
            pre_adjust_state = self.save_live_state()
            ctx = GenerationContext(
                token_probs=top_k_lp,
                prev_probs=list(prev_probs),
                context_tokens=list(context),
                query_next=lambda ctx_ids: (
                    self.query_log_probs_next_token(ctx_ids, top_k).top_k_tokens
                ),
                query_branch=self.query_branch,
                tokenize_token=lambda s: self._llm.tokenize(
                    s.encode('utf-8'), add_bos=False, special=False,
                ),
                base_live_state=pre_adjust_state,
                query_next_ids_from_live=lambda n: self._top_k_ids_from_logprobs(
                    self._log_softmax(self._llm.scores[self._llm.n_tokens - 1]),
                    n,
                ),
                save_live_state=self.save_live_state,
                load_live_state=self.load_live_state,
                eval_tokens=self._llm.eval,
            )
            adjusted = adjust_fn(ctx)
            self.load_live_state(pre_adjust_state)
            token_str = sample_from_logprobs(adjusted)
            token_ids = self._llm.tokenize(
                token_str.encode('utf-8'), add_bos=False, special=True,
            )

# Check for end, call eval() if continuing.
            if not token_ids or eos_id in token_ids:
                break
            self._llm.eval(token_ids)

# Assign various data variables for safekeeping.
            token_prob = float(prob_of_token(token_str, adjusted))
            _logger.debug(
                'step %d/%d -- Sampled %r (p=%.4f)', step + 1, max_tokens, token_str, token_prob
            )
            prev_probs.append(token_prob)
            response_prob_tokens.append(token_str)
            response_prob_values.append(token_prob)
            response_topk_dists.append({k: float(v) for k, v in adjusted.items()})
            context.extend(token_ids)

        _logger.info('Generation with adjusted probabilities complete.')

# Construct dataclass output.
        return LLMOutputDataFull(
            context=self._tokens_as_strings(context[:self._initial_context_length]),
            hyperparams=Hyperparams(
                alpha=alpha,
                top_k=top_k,
                top_p=top_p,
                max_tokens=max_tokens,
            ),
            response_probabilities=(response_prob_tokens, response_prob_values),
            response_topk=(copy.copy(response_prob_tokens), response_topk_dists),
            sampling_method=sampling_method,
            branch_sampler=branch_sampler
        )


## -- Testing on datasets -- ##

    def test_dataset_adjusted(
        self,
        dataset: list[str],
        top_k: int,
        top_p: float,
        adjust_fn: AdjustFn,
        max_tokens: int
    ) -> list[str]:
        """Generate answers to a series of questions in a provided dataset."""
        answers = []
        n_problems = len(dataset)
        for ii in tqdm(range(n_problems), desc='dataset_progress', unit='problem'):
            problem = dataset[ii]
            self.change_context(problem)
            out = self.generate_adjusted(top_k, top_p, adjust_fn, max_tokens)
            answers.append(''.join(out.response_probabilities[0]))
            _logger.info('Completed problem: %d/%d', ii + 1, n_problems)
        return answers
