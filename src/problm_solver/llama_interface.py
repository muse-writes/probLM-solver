"""llama.cpp python interface for running local models."""

import copy
import logging
from collections.abc import Callable
from inspect import isclass
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from llama_cpp import Llama, LlamaRAMCache, LlamaState
from llama_cpp.llama_chat_format import Jinja2ChatFormatter
from tqdm import tqdm

from problm_solver.adjust_probs import AdjustFn, GenerationContext
from problm_solver.analysis.probabilities import prob_of_token, sample_from_logprobs  # noqa: F401
from problm_solver.candidates import CandidateGeneratorFactory, CandidateTokens
from problm_solver.data import (
    Hyperparams,
    LLMNextTokenData,
    LLMOutputData,
    LLMOutputDataFull,
    LLMTokenData,
)
from problm_solver.llama_lowlevel import ModelBackendGeneric, ModelCBackend, ModelLlamaBackend
from problm_solver.random import RNGLike, resolve_rng

# -- Module-wide setup -- #

_logger = logging.getLogger(__name__)

ADEQUATE_TOPK = 30
ADEQUATE_TOPP = 0.8


# -- Main model instance -- #

class ModelInstance:
    """Keeps a model instance and its context, with methods for querying the Llama instance."""

    def __init__( # noqa: PLR0913
        self,
        fname: str,
        context: str,
        n_ctx: int = 4096,
        logits_all: bool = False, # noqa: FBT001 FBT002
        n_gpu_layers: int = 0,
        use_c_api: bool = True, # noqa: FBT001 FBT002
        *,
        rng: RNGLike = None,
    ) -> None:
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
        :param n_gpu_layers: number of GPU layers to pass to Llama. Required for GPU accelerated
            jobs, set to 0 otherwise.
        :param rng: Optional random source. May be a ``numpy.random.Generator``,
            integer seed, or ``RandomManager``.
        """
        self._llm = Llama(
            model_path=fname,
            n_ctx=n_ctx,
            logits_all=logits_all,
            verbose=False,
            n_gpu_layers=n_gpu_layers
        )
        self._logits_all = logits_all
        _logger.info('Model %r loaded.', fname)
        self._llm_backend: ModelBackendGeneric
        if use_c_api:
            self._llm_backend = ModelCBackend(self._llm)
        else:
            self._llm_backend = ModelLlamaBackend(self._llm)

# Calculate number of bytes needed in Llama cache.
        arch = self._llm_backend.metadata()['general.architecture']
        n_layers = int(self._llm_backend.metadata()[f'{arch}.block_count'])
        n_kv_heads = int(self._llm_backend.metadata()[f'{arch}.attention.head_count_kv'])
        n_heads = int(self._llm_backend.metadata()[f'{arch}.attention.head_count'])
        head_dim = int(self._llm_backend.metadata()[f'{arch}.embedding_length']) // n_heads
        bytes_per_state = self._llm_backend.n_ctx() * 2 * n_layers * n_kv_heads * head_dim * 2

# Initialise and set cache and context.
        self._cache = LlamaRAMCache(capacity_bytes=4 * bytes_per_state)
        self._llm.set_cache(self._cache)
        self.context = context
        self._initial_context_length: int = len(self.context)

# Set up RNG handling.
        self._rng = resolve_rng(rng, stream='global')

# Vocabulary pruning stuff (important for computationally intensive probability adjustments).
        self._candidate_factory = CandidateGeneratorFactory()


## -- Methods for querying the LLM. -- ##

    def query_n_times(self, n: int) -> npt.NDArray[Any]:
        """Query the LLM with the same context N times, return the output.

        :param n: number of times N to query the Llama instance.
        :returns: an array of response strings.
        """
        return np.array([self.query() for _ in range(n)], dtype=str)


    def query(
        self,
        max_tokens: int = 512,
        rng: RNGLike = None
    ) -> str:
        """Query the LLM once.

        :returns: the response string.
        """
        prompt_tokens = self._format_chat_prompt()
        self._llm_backend.reset()
        self._llm_backend.decode(prompt_tokens)
        tokens = []
        method_rng = resolve_rng(
            self._rng if rng is None else rng,
            stream='llama.query',
        )
        for _ in range(max_tokens):
            logprobs = self._log_softmax(self._llm.scores[self._llm.n_tokens - 1])
            next_id = int(np.argmax(logprobs + method_rng.gumbel(size=len(logprobs))))
            if next_id == self._llm.token_eos():
                break
            tokens.append(next_id)
            self._llm_backend.decode([next_id])
        return self._llm.detokenize(tokens).decode('utf-8')


    def generate_data(self, n_samples: int) -> LLMOutputData:
        """Generate data by querying the LLM `n_samples` times.

        :param n_samples: the number of times to query the Llama instance.
        :returns: A data container with all responses and the prompt.
        """
        data = self.query_n_times(n_samples)
        return LLMOutputData(prompt=self.context, data=data)


    def query_log_probs(self, rng: RNGLike = None) -> LLMTokenData:
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
        self._llm_backend.reset()
        self._llm_backend.decode(prompt_tokens)
        tokens: list[str] = []
        probs: list[float] = []
        eos_id = self._llm_backend.token_eos()
        method_rng = resolve_rng(
            self._rng if rng is None else rng,
            stream='llama.query_log_probs',
        )
        for _ in range(max_tokens):
            logprobs = self._log_softmax(self._llm.scores[self._llm.n_tokens - 1])
            next_id = int(np.argmax(logprobs + method_rng.gumbel(size=len(logprobs))))
            if next_id == eos_id:
                break
            tokens.append(self._tokens_as_strings([next_id])[0])
            probs.append(float(np.exp(logprobs[next_id])))
            self._llm.eval([next_id])
        return LLMTokenData(prompt=self.context, tokens=tokens, probs=probs)


    def query_log_probs_next_token_ids(
        self,
        context_tokens: list[int],
        n_tokens: int,
    ) -> CandidateTokens:
        """Return top-k next-token candidates in token-ID space."""
        self._llm_backend.reset()
        self._llm_backend.decode(context_tokens)
        logprobs = self._log_softmax(self._llm_backend.last_logits())
        generator = self._candidate_factory.get_candidate_generator(top_k=n_tokens, top_p=1.0)
        return generator(logprobs)

    def query_log_probs_next_token(
        self,
        context_tokens: list[int],
        n_tokens: int,
    ) -> LLMNextTokenData:
        """Return the top-k most likely next tokens and their log-probabilities.

        Resets the model state, evaluates ``context_tokens`` in a single
        forward pass, applies log-softmax to the last-position logits, selects
        candidate IDs via :class:`CandidateGeneratorFactory`, and converts only
        those selected IDs to token strings.

        EOS detection is the caller's responsibility: the EOS token will
        appear naturally in the returned distribution when the model prefers
        it, and :meth:`generate_adjusted` stops the loop after the EOS token
        ID is sampled.

        :param context_tokens: The current context as a list of integer token IDs.
        :param n_tokens: Number of top candidate tokens to return.
        :returns: ``LLMNextTokenData`` containing the top-K token → log-prob
            mapping.
        """
        candidates = self.query_log_probs_next_token_ids(context_tokens, n_tokens)
        top_k = self._candidate_tokens_to_logprob_map(candidates)
        return LLMNextTokenData(
            prompt=self.context,
            output_vec=context_tokens,
            top_k_tokens=top_k
        )


    def query_branch(
        self,
        context_tokens: list[int],
        max_tokens: int,
        rng: RNGLike = None
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
        self._llm_backend.reset()
        self._llm_backend.decode(context_tokens)

# Snapshot the KV cache and logits immediately after evaluating the
# context.  Restoring this state before generation ensures the branch
# always starts from the clean post-context position and is not
# affected by any internal bookkeeping inside save_live_state.
        pre_branch_state = self.save_live_state()
        self.load_live_state(pre_branch_state)

        eos_id = self._llm_backend.token_eos()
        total_log_prob = 0.0
        method_rng = resolve_rng(
            self._rng if rng is None else rng,
            stream='llama.query_branch',
        )

        for _ in range(max_tokens):
# scores[n_tokens - 1] is the most recently decoded logit row,
# valid for logits_all=True (filled by eval's n_past slice).
            logprobs = self._log_softmax(self._llm_backend.last_logits())

# Gumbel-max trick: argmax(log p + Gumbel(0,1)) is equivalent to
# drawing from categorical(softmax(log p)) without materialising
# the full probability vector.
            next_id = int(np.argmax(logprobs + method_rng.gumbel(size=len(logprobs))))

            if next_id == eos_id:
                break

            total_log_prob += float(logprobs[next_id])
            self._llm_backend.decode([next_id])

        return total_log_prob


    def query_branch_from_live(self, max_tokens: int, rng: RNGLike = None) -> float:
        """Generate a branch from the currently loaded live state.

        Unlike :meth:`query_branch`, this method does not call ``reset()`` and
        does not evaluate any prefix/context tokens. It assumes the live model
        state is already positioned at the branch root.

        :param max_tokens: Maximum number of tokens to generate in the branch.
        :returns: Sum of per-token log-probabilities for all generated tokens,
            or ``0.0`` if EOS is sampled on the first step.
        """
        eos_id = self._llm_backend.token_eos()
        total_log_prob = 0.0
        method_rng = resolve_rng(
            self._rng if rng is None else rng,
            stream='llama.query_branch_from_live',
        )

        for _ in range(max_tokens):
            logprobs = self._log_softmax(self._llm_backend.last_logits())
            next_id = int(np.argmax(logprobs + method_rng.gumbel(size=len(logprobs))))

            if next_id == eos_id:
                break

            total_log_prob += float(logprobs[next_id])
            self._llm_backend.decode([next_id])

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

    def _candidate_tokens_to_logprob_map(
        self,
        candidates: CandidateTokens,
    ) -> dict[str, float]:
        """Convert candidate IDs/logprobs to a string-keyed log-probability map."""
        ids = candidates.candidate_ids.tolist()
        toks = self._tokens_as_strings(ids)
        return {
            tok: float(lp)
            for tok, lp in zip(toks, candidates.candidate_logprobs, strict=True)
        }

    @staticmethod
    def _normalise_adjusted_to_candidates(adjusted_raw: CandidateTokens) -> CandidateTokens:
        """Validate adjust_fn output is CandidateTokens."""
        if isinstance(adjusted_raw, CandidateTokens):
            return adjusted_raw
        msg = 'adjust_fn must return CandidateTokens.'
        raise TypeError(msg)

    @staticmethod
    def _sample_token_from_candidates(
        candidates: CandidateTokens,
        rng: np.random.Generator,
    ) -> tuple[int, float, float]:
        """Sample one token ID from candidate logprobs and return id/prob/logprob."""
        lp = candidates.candidate_logprobs
        ids = candidates.candidate_ids
        shifted = lp - lp.max()
        probs = np.exp(shifted)
        probs /= probs.sum()
        sampled_idx = int(rng.choice(len(ids), p=probs))
        return int(ids[sampled_idx]), float(probs[sampled_idx]), float(lp[sampled_idx])


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


    def _top_k_ids_from_logprobs(
        self,
        logprobs: npt.NDArray[np.float64],
        n: int,
    ) -> list[tuple[int, float]]:
        """Return top-k ``(token_id, logprob)`` pairs ordered descending by logprob.

        :param logprobs: Full vocabulary array of log-probabilities as
            returned by :meth:`_log_softmax`
        :param n: number of ``(token_id, logprob)`` pairs to return.
        """
        generator = self._candidate_factory.get_candidate_generator(top_k=n, top_p=1.0)
        candidates = generator(logprobs)
        return [
            (int(idx), float(lp))
            for idx, lp in zip(
                candidates.candidate_ids,
                candidates.candidate_logprobs,
                strict=True,
            )
        ]


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
        rng: RNGLike = None,
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
            returns adjusted candidate token IDs/log-probabilities as
            ``CandidateTokens``. Values do not need to be normalised.
        :param max_tokens: Maximum number of tokens to generate.
        :returns: ``LLMOutputDataFull`` containing the model's response,
            candidate tokens at each step, and logprobs.
        """
# Hyperparam and sampling setup.
        if not self._logits_all:
            msg = (
                'generate_adjusted() requires logits_all=True when constructing ModelInstance '
                'so per-token logits are available.'
            )
            raise ValueError(msg)

        if sampling_method is None:
            if isclass(adjust_fn):
                sampling_method = adjust_fn.__class__.__name__
            else:
                sampling_method = getattr(adjust_fn, '__name__', type(adjust_fn).__name__)
        candidate_generator = self._candidate_factory.get_candidate_generator(top_k, top_p)

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
        eos_id = self._llm_backend.token_eos()
        context = self._format_chat_prompt()
        self._llm_backend.reset()
        self._llm_backend.decode(context)

# Main generation loop.
        method_rng = resolve_rng(
            self._rng if rng is None else rng,
            stream='llama.generate_adjusted',
        )
        reset_fn = getattr(adjust_fn, 'reset', None)
        if callable(reset_fn):
            cast(Callable[[], None], reset_fn)()
        for step in tqdm(range(max_tokens), desc='generate_adjusted', unit='tok'):

# Determine logprobs and sample intersection of top-k and top-p.
            logprobs = self._log_softmax(self._llm_backend.last_logits())
            candidates = candidate_generator(logprobs)

# Skip adjustment and sampling if only one logprob.
            if len(candidates.candidate_ids) == 1:
                sampled_id = int(candidates.candidate_ids[0])
                token_prob = 1.0
                adjusted_candidates = candidates
            else:
                pre_adjust_state = self.save_live_state()
                ctx = GenerationContext(
                    token_id_probs=candidates,
                    prev_probs=list(prev_probs),
                    context_tokens=list(context),
                    query_next_id=lambda ctx_ids: self.query_log_probs_next_token_ids(
                        ctx_ids,
                        top_k,
                    ),
                    query_branch=lambda ctx_ids, depth: self.query_branch(
                        ctx_ids,
                        depth,
                        rng=method_rng,
                    ),
                    query_branch_from_live=lambda depth: self.query_branch_from_live(
                        depth,
                        rng=method_rng,
                    ),
                    base_live_state=pre_adjust_state,
                    query_next_ids_from_live=lambda n: self._top_k_ids_from_logprobs(
                        self._log_softmax(self._llm_backend.last_logits()),
                        n,
                    ),
                    save_live_state=self.save_live_state,
                    load_live_state=self.load_live_state,
                    eval_tokens=self._llm_backend.decode,
                )
                adjusted_raw = adjust_fn(ctx)
                self.load_live_state(pre_adjust_state)
                adjusted_candidates = self._normalise_adjusted_to_candidates(adjusted_raw)
                if len(adjusted_candidates.candidate_ids) == 0:
                    msg = 'adjust_fn returned an empty token distribution.'
                    raise ValueError(msg)
                sampled_id, token_prob, _ = self._sample_token_from_candidates(
                    adjusted_candidates,
                    method_rng,
                )

# Resolve sampled token IDs for EOS checks and optional commit.
            token_ids = [sampled_id]

# Check for end, call decode() if continuing.
            if not token_ids or eos_id in token_ids:
                break
            self._llm_backend.decode(token_ids)

# Assign various data variables for safekeeping.
            token_str = self._tokens_as_strings([sampled_id])[0]
            adjusted = self._candidate_tokens_to_logprob_map(adjusted_candidates)
            _logger.debug(
                'step %d/%d -- Sampled %r (p=%.4f)', step + 1, max_tokens, token_str, token_prob
            )
            prev_probs.append(token_prob)
            response_prob_tokens.append(token_str)
            response_prob_values.append(token_prob)
            response_topk_dists.append(adjusted)
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


    def sample_token_adjusted(
        self,
        top_k: int,
        top_p: float,
        adjust_fn: AdjustFn,
        *,
        use_live_state: bool = True,
        context_tokens: list[int] | None = None,
        prev_probs: list[float] | None = None,
        commit_token: bool = True,
        rng: RNGLike = None,
    ) -> dict[str, Any]:
        """Sample exactly one token from an adjusted next-token distribution.

        This method is optimised for iterative decoding: when ``use_live_state``
        is ``True`` and the model already has decoded tokens
        (``self._llm.n_tokens > 0``), it *does not* rebuild prompt/KV state and
        samples directly from the current live logits.

        Fallback behaviour:
        - if ``use_live_state=False``: always rebuild state from
          ``context_tokens`` (if provided) or the formatted prompt.
        - if ``use_live_state=True`` but no live state exists: rebuild from
          ``context_tokens`` or the formatted prompt.

        :param top_k: Number of top-k tokens to consider.
        :param top_p: Nucleus threshold in ``(0, 1]``.
        :param adjust_fn: Function that adjusts candidate token log-probabilities.
        :param use_live_state: Prefer sampling from current live model state.
        :param context_tokens: Optional explicit context token IDs to evaluate
            when rebuilding state.
        :param prev_probs: Optional previously sampled token probabilities,
            passed through to ``GenerationContext``.
        :param commit_token: Whether to append the sampled token to the live
            model state via ``eval(token_ids)`` when non-terminal.
        :returns: A dictionary containing candidate distributions before/after
            adjustment and details of the sampled token.
        """
        if not self._logits_all:
            msg = (
                'sample_token_adjusted() requires logits_all=True when constructing '
                'ModelInstance so per-token logits are available.'
            )
            raise ValueError(msg)

        candidate_generator = self._candidate_factory.get_candidate_generator(top_k, top_p)

        method_rng = resolve_rng(
            self._rng if rng is None else rng,
            stream='llama.sample_token_adjusted',
        )

        state_source: str
        effective_context_tokens: list[int] | None
        if use_live_state and self._llm_backend.n_tokens > 0:
            state_source = 'live'
            effective_context_tokens = None
        else:
            self._llm_backend.reset()
            if context_tokens is None:
                effective_context_tokens = self._format_chat_prompt()
                state_source = 'prompt'
            else:
                effective_context_tokens = list(context_tokens)
                state_source = 'context_tokens'
            self._llm_backend.decode(effective_context_tokens)

        logprobs = self._log_softmax(self._llm_backend.last_logits())
        candidates = candidate_generator(logprobs)

        prev_prob_values = list(prev_probs) if prev_probs is not None else []

        if len(candidates.candidate_ids) == 1:
            adjusted_candidates = candidates
            sampled_id = int(candidates.candidate_ids[0])
            token_prob = 1.0
            token_logprob = float(candidates.candidate_logprobs[0])
        else:
            pre_adjust_state = self.save_live_state()
            ctx = GenerationContext(
                token_id_probs=candidates,
                prev_probs=prev_prob_values,
                context_tokens=(
                    list(effective_context_tokens)
                    if effective_context_tokens is not None
                    else []
                ),
                query_next_id=lambda ctx_ids: self.query_log_probs_next_token_ids(
                    ctx_ids,
                    top_k,
                ),
                query_branch=lambda ctx_ids, depth: self.query_branch(
                    ctx_ids,
                    depth,
                    rng=method_rng,
                ),
                query_branch_from_live=lambda depth: self.query_branch_from_live(
                    depth,
                    rng=method_rng,
                ),
                base_live_state=pre_adjust_state,
                query_next_ids_from_live=lambda n: self._top_k_ids_from_logprobs(
                    self._log_softmax(self._llm_backend.last_logits()),
                    n,
                ),
                save_live_state=self.save_live_state,
                load_live_state=self.load_live_state,
                eval_tokens=self._llm_backend.decode,
            )
            adjusted_raw = adjust_fn(ctx)
            self.load_live_state(pre_adjust_state)
            adjusted_candidates = self._normalise_adjusted_to_candidates(adjusted_raw)

            if len(adjusted_candidates.candidate_ids) == 0:
                msg = 'adjust_fn returned an empty token distribution.'
                raise ValueError(msg)

            sampled_id, token_prob, token_logprob = self._sample_token_from_candidates(
                adjusted_candidates,
                method_rng,
            )

        token_ids = [sampled_id]
        eos_id = self._llm_backend.token_eos()
        sampled_is_terminal = sampled_id == eos_id

        if commit_token and not sampled_is_terminal:
            self._llm_backend.decode(token_ids)

        top_k_lp = self._candidate_tokens_to_logprob_map(candidates)
        adjusted = self._candidate_tokens_to_logprob_map(adjusted_candidates)
        token_str = self._tokens_as_strings([sampled_id])[0]

        sampled_token: dict[str, Any] | None
        if sampled_is_terminal:
            sampled_token = None
        else:
            sampled_token = {
                'token': token_str,
                'token_ids': token_ids,
                'logprob': token_logprob,
                'prob': token_prob,
            }

        return {
            'state_source': state_source,
            'used_live_state': state_source == 'live',
            'top_k': int(top_k),
            'top_p': float(top_p),
            'candidates_before_adjustment': _serialise_candidates(top_k_lp),
            'candidates_after_adjustment': _serialise_candidates(adjusted),
            'sampled_token': sampled_token,
            'sampled_token_is_terminal': sampled_is_terminal,
            'context_tokens_used_for_eval': effective_context_tokens,
        }


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


## -- Module Helpers -- ##

def _serialise_candidates(lp_map: dict[str, float]) -> list[dict[str, float | str]]:
    return [
        {
            'token': tok,
            'logprob': float(lp),
            'prob': float(prob_of_token(tok, lp_map)),
        }
        for tok, lp in lp_map.items()
    ]
