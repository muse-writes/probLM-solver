"""Implement several sampling functions for generate_with_sampler and sample_token."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from problm_solver.candidates import CandidateTokens
from problm_solver.random import RNGLike, resolve_rng

# -- Module-wide setup -- #

_logger = logging.getLogger(__name__)


@dataclass
class GenerationContext:
    """All information available to an :data:`AdjustFn` at each generation step.

    Injected by ``generate_with_sampler()`` so that adjustment functions can
    access model-querying capabilities without a direct dependency on
    ``Model``. All mutable fields are defensive copies.

    :param token_id_probs: Current top-k candidate token IDs and log-probabilities.
    :param prev_probs: Normalized probabilities of all previously selected
        tokens in this generation. Empty on the first step.
    :param context_tokens: The current token ID sequence (prompt + generated
        tokens so far).
    :param query_next_id: Queries the model for top-k next-token candidates in
        token-ID space given a context token ID list.
    :param query_branch: Generates a complete branch of up to ``depth`` tokens
        from the given context in a single model call and returns the sum of
        per-token log-probabilities. Returns ``0.0`` on immediate EOS.
    :param query_branch_from_live: Generates a complete branch of up to
        ``depth`` tokens from the model's currently loaded live state and
        returns the sum of per-token log-probabilities. Optional.
    :param query_branches_from_live_batch: Generates ``n_branches`` independent
        branches of up to ``depth`` tokens in a single batched pass from the
        live state, returning an ``(n_branches,)`` array of per-branch total
        log-probabilities. Optional; when present, :class:`SamplePowerDist`
        uses it to vectorise proposal generation.
    """

    token_id_probs: CandidateTokens
    prev_probs: list[float]
    context_tokens: list[int]
    query_next_id: Callable[[list[int]], CandidateTokens]
    query_branch: Callable[[list[int], int], float]
    query_branch_from_live: Callable[[int], float] | None = None
    base_live_state: Any | None = None
    query_next_ids_from_live: Callable[[int], list[tuple[int, float]]] | None = None
    save_live_state: Callable[[], Any] | None = None
    load_live_state: Callable[[Any], None] | None = None
    eval_tokens: Callable[[list[int]], None] | None = None
    query_branches_from_live_batch: Callable[[int, int], npt.NDArray[np.float64]] | None = None


# Callable that receives a GenerationContext and returns adjusted
# token-ID candidates with log-probabilities.
type AdjustFn = Callable[[GenerationContext], CandidateTokens]


def candidate_tokens_to_id_logprobs(candidates: CandidateTokens) -> dict[int, float]:
    """Convert CandidateTokens to an insertion-ordered token-id logprob map."""
    return {
        int(tid): float(lp)
        for tid, lp in zip(
            candidates.candidate_ids,
            candidates.candidate_logprobs,
            strict=True,
        )
    }


def id_logprobs_to_candidate_tokens(id_logprobs: dict[int, float]) -> CandidateTokens:
    """Convert token-id logprob map to CandidateTokens preserving insertion order."""
    items = list(id_logprobs.items())
    if not items:
        return CandidateTokens(
            candidate_ids=np.empty(0, dtype=np.int32),
            candidate_logprobs=np.empty(0, dtype=np.float64),
        )

    ids = np.array([tid for tid, _ in items], dtype=np.int32)
    lps = np.array([lp for _, lp in items], dtype=np.float64)
    return CandidateTokens(candidate_ids=ids, candidate_logprobs=lps)


def adjust_identity(context: GenerationContext) -> CandidateTokens:
    """Return token log-probabilities unchanged in token-ID space."""
    return context.token_id_probs


class SampleLowTemp:
    """Adjust token log-probabilities by per-step power-scaling (low-temperature sampling).

    At each generation step the current token log-probabilities are multiplied
    by ``alpha`` (equivalently, the probabilities are raised to ``alpha``),
    which sharpens the distribution for ``alpha > 1`` and flattens it for
    ``0 < alpha < 1``. The result is returned as log-probabilities for
    downstream renormalisation and sampling.

    The selection history (``prev_probs``) is deliberately *not* folded into
    the output. An earlier version added ``alpha * sum(log(prev_probs))`` to
    every candidate's log-probability, modelling a joint-sequence probability.
    Because that term is identical across candidates at a given step, it is
    shift-invariant under the softmax used for sampling and therefore has no
    effect on which token is selected or on the stored per-token
    probabilities. Its only observable effect was to inject a steadily growing
    negative offset into the top-k log-probability map stored in
    ``LLMOutputDataFull.response_topk`` (drifting to ~``-alpha * n * mean_logprob``
    by the end of a long generation). Omitting it keeps the stored log-probabilities
    on a stable, interpretable scale without altering the sampling rigidity,
    which is governed solely by the per-step ``alpha * lp`` scaling.

    :param alpha: Scaling exponent. Values greater than 1 sharpen the
        distribution (favoring already-likely tokens); values between 0
        and 1 flatten it.

    Example usage::

        adjust_fn = SampleLowTemp(alpha=2)
        result = adjust_fn(context)
    """

    def __init__(self, alpha: float) -> None:
        """Initialize with scaling exponent.

        :param alpha: Exponent applied to the current token probabilities
            when computing the adjustment.
        """
        self.alpha = alpha

    def __call__(self, context: GenerationContext) -> CandidateTokens:
        """Apply per-step power-scaling adjustment to the current token-ID distribution."""
        candidate_ids = context.token_id_probs.candidate_ids
        lp = context.token_id_probs.candidate_logprobs.astype(np.float64, copy=True)
        lp -= lp.max()

        new_logprobs: npt.NDArray[np.float64] = self.alpha * lp
        return CandidateTokens(
            candidate_ids=candidate_ids.astype(np.int32, copy=False),
            candidate_logprobs=new_logprobs,
        )


class BranchSampler(ABC):
    """Abstract base class for branch-level sampling strategies.

    A ``BranchSampler`` typically runs over complete branch proposals. Some
    subclasses may additionally provide token-by-token beam expansion via
    :meth:`future_logprob_from_context`.
    """

    supports_token_beam = False

    @property
    def max_proposals(self) -> int:
        """Max branch proposals a batched path should generate upfront.

        Returns ``0`` by default to signal that this sampler does not support
        batched (vectorised) proposal generation. Samplers that do (e.g.
        :class:`MetropolisSampler`) override this so :class:`SamplePowerDist`
        knows how many proposals to request in a single batched call.
        """
        return 0

    def reset(self) -> None: # noqa: B027
        """Reset internal state at the start of each candidate-token chain.

        No-op for stateless samplers. Stateful samplers (e.g.
        :class:`MetropolisSampler`) should override this.
        """

    @abstractmethod
    def step(
        self,
        proposed_log_prob: float,
        alpha: float = 1.0,
        forward_log_q: float = 0.0,
        reverse_log_q: float = 0.0,
    ) -> float:
        """Process one proposed branch and return the accepted chain state."""

    @abstractmethod
    def should_continue(self, branch_log_probs: npt.NDArray[np.float64]) -> bool:
        """Return ``True`` if more branch proposals should be sampled."""

    @abstractmethod
    def future_logprob(self, alpha: float, branch_log_probs: npt.NDArray[np.float64]) -> np.float64:
        """Calculate weighting to token probability from sampled branches."""

    def future_logprob_from_context(
        self,
        alpha: float,
        base_live_state: Any,
        branch_token_ids: list[int],
        lookahead_depth: int,
        query_next_ids_from_live: Callable[[int], list[tuple[int, float]]],
        save_live_state: Callable[[], Any],
        load_live_state: Callable[[Any], None],
        eval_tokens: Callable[[list[int]], None],
    ) -> np.float64:
        """Optional token-by-token beam expansion hook.

        Subclasses that implement token-level beam search should override this
        method and set ``supports_token_beam = True``.
        """
        raise NotImplementedError


class MetropolisSampler(BranchSampler):
    """Metropolis-Hastings sampler over complete branch proposals.

    Given a sequence of proposed branch log-probabilities, maintains an MCMC
    chain and accepts a proposal with probability

    ``min(1, exp(log p(x') - log p(x) + log q(x|x') - log q(x'|x)))``.

    Convergence across accepted branch samples is assessed via the standard
    error of the mean (SEM): ``SEM = std(branch_log_probs) / sqrt(n)``.
    Sampling continues until ``SEM < tolerance``, after at least
    ``equil_branches`` samples, and always stops at ``max_branches``.

    :param equil_branches: Number of accepted samples treated as burn-in
        (equilibration); discarded before checking SEM.
    :param max_branches: Hard upper limit on accepted samples.
    :param tolerance: SEM threshold below which sampling is considered
        converged.
    """

    def __init__(
        self,
        equil_branches: int = 5,
        max_branches: int = 30,
        tolerance: float = 1e-1,
        rng: RNGLike = None
    ) -> None:
        """Initialize with convergence parameters."""
        self._current_log_prob: float | None = None
        self._equil_branches = equil_branches
        self._max_branches = max_branches
        self._tolerance = tolerance
        self._rng = resolve_rng(rng, stream='adjust.metropolis')

    @property
    def max_proposals(self) -> int:
        """Number of proposals to generate in one batched pass (= max_branches)."""
        return self._max_branches

    def reset(self) -> None:
        """Clear chain state before starting a new candidate-token chain."""
        self._current_log_prob = None

    def step(
        self,
        proposed_log_prob: float,
        alpha: float = 1.0,
        forward_log_q: float = 0.0,
        reverse_log_q: float = 0.0,
    ) -> float:
        """Apply one Metropolis-Hastings accept/reject step targeting ``p^α``.

        The log acceptance ratio is
        ``(α-1) * (log p(x') - log p(x)) + log q(x|x') - log q(x'|x)``.
        When the proposal ``q`` is the base model ``p`` the proposal terms
        cancel (``forward_log_q = proposed_log_prob``,
        ``reverse_log_q = current_log_prob``), reducing to
        ``(α-1) * (proposed - current)``.

        :param proposed_log_prob: Proposed branch log-probability under ``p``.
        :param alpha: Power-distribution exponent.
        :param forward_log_q: ``log q(x'|x)`` for the proposal.
        :param reverse_log_q: ``log q(x|x')`` for the reverse proposal.
        :returns: Accepted chain state's log-probability.
        """
        if self._current_log_prob is None:
            self._current_log_prob = proposed_log_prob
            return self._current_log_prob

        log_accept_ratio = (
            (alpha - 1) * (proposed_log_prob - self._current_log_prob)
            + reverse_log_q
            - forward_log_q
        )
        if np.log(self._rng.random()) < min(0.0, log_accept_ratio):
            self._current_log_prob = proposed_log_prob

        return self._current_log_prob

    def should_continue(self, branch_log_probs: npt.NDArray[np.float64]) -> bool:
        """Return ``True`` if more proposals should be sampled.

        Uses SEM-based convergence after ``equil_branches`` and before
        ``max_branches``.

        :param: branch logarithmic probabilities to date.
        """
        return len(branch_log_probs) < self._max_branches

    def future_logprob(self, alpha: float, branch_log_probs: npt.NDArray[np.float64]) -> np.float64:
        """Monte Carlo mean weight."""
        post_eq = branch_log_probs[self._equil_branches:]
        scaled = alpha * post_eq
        max_lp = np.float64(scaled.max())
        return np.log(np.mean(np.exp(scaled - max_lp))) + max_lp


class BeamSampler(BranchSampler):
    """Token-by-token beam expansion for future-branch scoring.

    This sampler performs deterministic beam search over lookahead tokens.
    For each candidate token, it repeatedly expands active beams using
    ``query_next_ids_from_live`` and keeps only the top ``beam_width`` cumulative
    log-probability branches at every depth.

    :param beam_width: Number of active beams retained per depth.
    :param branch_top_k: Number of next-token candidates considered for each
        active beam during expansion.
    """

    supports_token_beam = True

    def __init__(self, beam_width: int = 3, branch_top_k: int = 5) -> None:
        """Initialize beam-search width and per-beam expansion width."""
        if beam_width < 1:
            msg = f'beam_width must be >= 1, got {beam_width}'
            raise ValueError(msg)
        if branch_top_k < 1:
            msg = f'branch_top_k must be >= 1, got {branch_top_k}'
            raise ValueError(msg)

        self.beam_width = beam_width
        self.branch_top_k = branch_top_k

    def reset(self) -> None:
        """No-op: beam expansion is stateless across candidates."""

    def step(
        self,
        proposed_log_prob: float,
        alpha: float = 1.0, #noqa:ARG002
        forward_log_q: float = 0.0, #noqa:ARG002
        reverse_log_q: float = 0.0, #noqa:ARG002
    ) -> float:
        """Compatibility no-op; token-beam mode does not use MH transitions."""
        return proposed_log_prob

    def should_continue(self, branch_log_probs: npt.NDArray[np.float64]) -> bool:
        """Compatibility no-op; token-beam mode controls depth directly."""
        return False

    def future_logprob(self, alpha: float, branch_log_probs: npt.NDArray[np.float64]) -> np.float64:
        """Return log-mean-exp over supplied branch scores.

        This method is retained for compatibility, but token-beam mode
        normally uses :meth:`future_logprob_from_context`.
        """
        if len(branch_log_probs) == 0:
            msg = 'branch_log_probs cannot be empty'
            raise ValueError(msg)

        scaled = alpha * branch_log_probs
        max_lp = np.float64(scaled.max())
        return np.log(np.mean(np.exp(scaled - max_lp))) + max_lp

    def future_logprob_from_context(
        self,
        alpha: float,
        base_live_state: Any,
        branch_token_ids: list[int],
        lookahead_depth: int,
        query_next_ids_from_live: Callable[[int], list[tuple[int, float]]],
        save_live_state: Callable[[], Any],
        load_live_state: Callable[[Any], None],
        eval_tokens: Callable[[list[int]], None],
    ) -> np.float64:
        """Run token-level beam expansion with KV-cache state reuse."""
        load_live_state(base_live_state)
        if branch_token_ids:
            eval_tokens(branch_token_ids)
        root_state = save_live_state()

        beams: list[tuple[Any, float]] = [(root_state, 0.0)]

        for _ in range(lookahead_depth):
            expanded: list[tuple[Any, float]] = []

            for beam_state, cum_lp in beams:
                load_live_state(beam_state)
                top_next = query_next_ids_from_live(self.branch_top_k)

                for token_id, token_lp in top_next:
                    load_live_state(beam_state)
                    eval_tokens([token_id])
                    child_state = save_live_state()
                    expanded.append((child_state, cum_lp + float(token_lp)))

            if not expanded:
                break

            expanded.sort(key=lambda item: item[1], reverse=True)
            beams = expanded[: self.beam_width]

        if not beams:
            return np.float64(-np.inf)

        beam_log_probs = np.array([lp for _, lp in beams], dtype=np.float64)
        scaled = alpha * beam_log_probs
        max_lp = np.float64(scaled.max())
        return np.log(np.mean(np.exp(scaled - max_lp))) + max_lp


class SamplePowerDist:
    """Adjust token log-probabilities using future-branch power-distribution sampling.

    For each candidate next token, repeatedly proposes complete future
    branches of length ``lookahead_depth`` and updates a Markov chain using the
    injected :class:`BranchSampler` (e.g. Metropolis-Hastings), continuing
    until :meth:`~BranchSampler.should_continue` signals convergence. Each
    branch is evaluated in a single model call via
    :attr:`~GenerationContext.query_branch`, rather than token-by-token. The
    accepted branch log-probabilities are kept as a ``numpy`` array and
    combined with the current token's log-probability via log-sum-exp to
    produce the adjusted distribution.

    Branches that reach EOS before ``lookahead_depth`` are terminated early
    with no penalty — their partial log-probability is used as-is.

    :param alpha: Scaling exponent applied to the current token log-probability.
    :param lookahead_depth: Maximum number of steps to sample in each branch.
    :param branch_sampler: Strategy used to sample tokens within each branch
        and to determine when enough branches have been collected.
        Must be a :class:`BranchSampler` instance; its
        :meth:`~BranchSampler.reset` method is called at the start of every
        branch.

    Example usage::

        sampler = SamplePowerDist(
            alpha=2.0,
            lookahead_depth=3,
            branch_sampler=MetropolisSampler(),
        )
        result = sampler(context)
    """

    def __init__(
        self,
        alpha: float,
        lookahead_depth: int,
        branch_sampler: BranchSampler,
    ) -> None:
        """Initialize with lookahead parameters and a branch sampler.

        :param alpha: Scaling exponent for the current token log-probability.
        :param lookahead_depth: Maximum depth of each branch.
        :param branch_sampler: The :class:`BranchSampler` to use within
            branches and for convergence decisions.
        """
        self.alpha = alpha
        self.lookahead_depth = lookahead_depth
        self.branch_sampler = branch_sampler

    def _run_mh_chain(self, proposals: npt.NDArray[np.float64]) -> np.float64:
        """Run the MH accept/reject loop over a batch of branch log-probabilities.

        Resets the branch sampler, feeds each proposal through :meth:`step`,
        stops when :meth:`should_continue` returns ``False``, and returns the
        converged weighting via :meth:`future_logprob`. Shared between the
        batched and sequential proposal paths so both produce identical
        chain output for identical proposals.

        :param proposals: 1-D array of branch log-probabilities.
        :returns: The sampler's ``future_logprob`` weighting for the chain.
        """
        self.branch_sampler.reset()
        accepted: list[float] = []
        for proposed in proposals:
            accepted.append(self.branch_sampler.step(
                proposed_log_prob=float(proposed), alpha=self.alpha,
            ))
            if not self.branch_sampler.should_continue(
                np.array(accepted, dtype=np.float64)
            ):
                break
        return self.branch_sampler.future_logprob(
            self.alpha, np.array(accepted, dtype=np.float64)
        )

    def _run_sequential_chain(self, propose_branch: Callable[[], float]) -> np.float64:
        """Run the MH accept/reject loop, pulling proposals one at a time.

        Mirrors :meth:`_run_mh_chain` but for proposal paths whose branch
        log-probabilities cannot be materialised upfront (each proposal
        mutates shared live state and must be generated on demand). Shared by
        the live-state and token-ID-context paths so both produce identical
        chain output for identical proposals.

        :param propose_branch: Zero-arg callable returning one proposed
            branch log-probability under ``p``.
        :returns: The sampler's ``future_logprob`` weighting for the chain.
        """
        self.branch_sampler.reset()
        accepted: list[float] = []
        while True:
            accepted.append(
                self.branch_sampler.step(
                    proposed_log_prob=propose_branch(), alpha=self.alpha,
                )
            )
            if not self.branch_sampler.should_continue(
                np.array(accepted, dtype=np.float64)
            ):
                break
        return self.branch_sampler.future_logprob(
            self.alpha, np.array(accepted, dtype=np.float64)
        )

    @staticmethod
    def _require_live_callables(context: GenerationContext, what: str) -> None:
        """Raise ``ValueError`` if any shared live-state callable is missing.

        :param context: The generation context to inspect.
        :param what: Human-readable label for the calling path, used in the
            error message.
        """
        if (
            context.base_live_state is None
            or context.save_live_state is None
            or context.load_live_state is None
            or context.eval_tokens is None
        ):
            msg = f'{what} requires live-state callables in GenerationContext'
            raise ValueError(msg)

    def _make_beam_scorer(
        self, context: GenerationContext
    ) -> Callable[[int], np.float64]:
        """Build a per-candidate scorer using token-level beam expansion.

        :raises ValueError: if the token-beam live-state callables are missing.
        """
        if context.query_next_ids_from_live is None:
            msg = 'Token-beam sampler requires live-state callables in GenerationContext'
            raise ValueError(msg)
        self._require_live_callables(context, 'Token-beam sampler')

        def score_future_from_candidate_id(candidate_id: int) -> np.float64:
            return self.branch_sampler.future_logprob_from_context(
                alpha=self.alpha,
                base_live_state=context.base_live_state,
                branch_token_ids=[candidate_id],
                lookahead_depth=self.lookahead_depth,
                query_next_ids_from_live=context.query_next_ids_from_live,
                save_live_state=context.save_live_state,
                load_live_state=context.load_live_state,
                eval_tokens=context.eval_tokens,
            )

        return score_future_from_candidate_id

    def _make_batched_branch_scorer(
        self, context: GenerationContext
    ) -> Callable[[int], np.float64] | None:
        """Build a per-candidate scorer using one batched proposal pass per candidate.

        All ``max_proposals`` branch proposals are drawn from the base model in
        a single batched call; because proposals are independent of chain
        state, batching does not change the MH target distribution. Returns
        ``None`` if the batched live-state path is unavailable.
        """
        if (
            context.query_branches_from_live_batch is None
            or context.base_live_state is None
            or context.save_live_state is None
            or context.load_live_state is None
            or context.eval_tokens is None
            or self.branch_sampler.max_proposals <= 0
        ):
            return None

        base_live_state = context.base_live_state
        query_branches_from_live_batch = context.query_branches_from_live_batch
        load_live_state = context.load_live_state
        eval_tokens = context.eval_tokens
        lookahead_depth = self.lookahead_depth
        n_proposals = self.branch_sampler.max_proposals

        def score_future_from_candidate_id(candidate_id: int) -> np.float64:
            # Position the live state at the candidate root on seq 0.
            load_live_state(base_live_state)
            eval_tokens([candidate_id])
            proposals = query_branches_from_live_batch(lookahead_depth, n_proposals)
            return self._run_mh_chain(proposals)

        return score_future_from_candidate_id

    def _make_live_branch_scorer(
        self, context: GenerationContext
    ) -> Callable[[int], np.float64] | None:
        """Build a per-candidate scorer that proposes branches one at a time from live state.

        Returns ``None`` if the sequential live-state path is unavailable.
        """
        if (
            context.query_branch_from_live is None
            or context.base_live_state is None
            or context.save_live_state is None
            or context.load_live_state is None
            or context.eval_tokens is None
        ):
            return None

        base_live_state = context.base_live_state
        query_branch_from_live = context.query_branch_from_live
        save_live_state = context.save_live_state
        load_live_state = context.load_live_state
        eval_tokens = context.eval_tokens
        lookahead_depth = self.lookahead_depth

        def score_future_from_candidate_id(candidate_id: int) -> np.float64:
            load_live_state(base_live_state)
            eval_tokens([candidate_id])
            candidate_root_state = save_live_state()

            def propose_branch() -> float:
                # Reload the candidate root before each proposal so the branch
                # grows from the same base context every iteration.
                load_live_state(candidate_root_state)
                return query_branch_from_live(lookahead_depth)

            return self._run_sequential_chain(propose_branch)

        return score_future_from_candidate_id

    def _make_context_branch_scorer(
        self, context: GenerationContext
    ) -> Callable[[int], np.float64]:
        """Build a per-candidate scorer using token-ID context branches (no live state)."""
        context_tokens = context.context_tokens
        query_branch = context.query_branch
        lookahead_depth = self.lookahead_depth

        def score_future_from_candidate_id(candidate_id: int) -> np.float64:
            branch_ctx = [*list(context_tokens), candidate_id]
            return self._run_sequential_chain(
                lambda: query_branch(branch_ctx, lookahead_depth)
            )

        return score_future_from_candidate_id

    def _make_branch_scorer(
        self, context: GenerationContext
    ) -> Callable[[int], np.float64]:
        """Select the best available branch-based per-candidate scorer.

        Preference order: batched live-state, sequential live-state,
        token-ID-context fallback.
        """
        return (
            self._make_batched_branch_scorer(context)
            or self._make_live_branch_scorer(context)
            or self._make_context_branch_scorer(context)
        )

    def __call__(self, context: GenerationContext) -> CandidateTokens:
        """Apply power-distribution adjustment using lookahead branch sampling.

        :param context: The current generation context in token-ID space.
        :returns: Adjusted candidate token IDs with log-probabilities.
        """
        if self.branch_sampler.supports_token_beam:
            score_future_from_candidate_id = self._make_beam_scorer(context)
        else:
            score_future_from_candidate_id = self._make_branch_scorer(context)

        result: dict[int, float] = {}
        candidate_ids = context.token_id_probs.candidate_ids
        candidate_logprobs = context.token_id_probs.candidate_logprobs

        candidate_bar = tqdm(
            zip(candidate_ids, candidate_logprobs, strict=True),
            desc='candidates',
            total=len(candidate_ids),
            unit='tok',
            leave=False,
        )
        for token_id, log_prob in candidate_bar:
            tid = int(token_id)
            future_lp = score_future_from_candidate_id(tid)
            result[tid] = self.alpha * float(log_prob) + float(future_lp)

        return id_logprobs_to_candidate_tokens(result)
