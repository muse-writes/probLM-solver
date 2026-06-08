"""Implement several adjustment functions for generate_adjusted."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from problm_solver.utils import _as_rng

# -- Module-wide setup -- #

_logger = logging.getLogger(__name__)


@dataclass
class GenerationContext:
    """All information available to an :data:`AdjustFn` at each generation step.

    Injected by ``generate_adjusted()`` so that adjustment functions can
    access model-querying capabilities without a direct dependency on
    ``ModelInstance``. All mutable fields are defensive copies.

    :param token_probs: Current top-M mapping of token string to log-probability.
    :param prev_probs: Normalised probabilities of all previously selected
        tokens in this generation. Empty on the first step.
    :param context_tokens: The current token ID sequence (prompt + generated
        tokens so far).
    :param query_next: Queries the model for the top-M next-token log-prob
        dict given a context token ID list. Pre-bound to the current
        ``n_tokens``.
    :param query_branch: Generates a complete branch of up to ``depth`` tokens
        from the given context in a single model call and returns the sum of
        per-token log-probabilities. Returns ``0.0`` on immediate EOS.
    :param tokenize_token: Converts a single token string to its token ID(s).
    """

    token_probs: dict[str, float]
    prev_probs: list[float]
    context_tokens: list[int]
    query_next: Callable[[list[int]], dict[str, float]]
    query_branch: Callable[[list[int], int], float]
    tokenize_token: Callable[[str], list[int]]


# Callable that receives a GenerationContext and returns a modified log-prob
# dict. Values need not be normalised; renormalisation is applied by
# sample_from_logprobs before sampling.
AdjustFn = Callable[[GenerationContext], dict[str, float]]


def adjust_identity(context: GenerationContext) -> dict[str, float]:
    """Return token log-probabilities unchanged.

    Satisfies the :data:`AdjustFn` interface without modifying the
    distribution. Useful as a baseline and for testing.

    :param context: The current generation context.
    :returns: ``context.token_probs`` unmodified.
    """
    return context.token_probs


class SampleLowTemp:
    """Adjust token log-probabilities by power-scaling with selection history.

    At each generation step, the current token probabilities are raised to
    ``alpha`` and multiplied by the product of all previously selected token
    probabilities each also raised to ``alpha``. The result is returned as
    log-probabilities for downstream renormalisation and sampling.

    :param alpha: Scaling exponent. Values greater than 1 sharpen the
        distribution (favouring already-likely tokens); values between 0
        and 1 flatten it.

    Example usage::

        adjust_fn = SampleLowTemp(alpha=2)
        result = adjust_fn(context)
    """

    def __init__(self, alpha: float) -> None:
        """Initialise with scaling exponent.

        :param alpha: Exponent applied to current and previous token
            probabilities when computing the adjustment.
        """
        self.alpha = alpha

    def __call__(self, context: GenerationContext) -> dict[str, float]:
        """Apply power-scaling adjustment to the current token distribution.

        :param context: The current generation context. Uses
            ``context.token_probs`` and ``context.prev_probs``.
        :returns: Adjusted log-probability mapping. Note that tokens with very
            low probability may produce ``-inf`` log-probabilities after
            scaling.
        """
        tokens = list(context.token_probs.keys())
        lp = np.array([context.token_probs[t] for t in tokens], dtype=float)
        lp -= lp.max()
        p: npt.NDArray = np.exp(lp)
        prev_alpha = float(
            np.prod(np.array(context.prev_probs, dtype=float) ** self.alpha)
        )
        new_logprobs: npt.NDArray = np.log(p ** self.alpha * prev_alpha)
        return dict(zip(tokens, list(new_logprobs), strict=True))


class BranchSampler(ABC):
    """Abstract base class for branch-level sampling strategies.

    A ``BranchSampler`` typically runs over complete branch proposals. Some
    subclasses may additionally provide token-by-token beam expansion via
    :meth:`future_logprob_from_context`.
    """

    supports_token_beam = False

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
        branch_ctx: list[int],
        lookahead_depth: int,
        query_next: Callable[[list[int]], dict[str, float]],
        tokenize_token: Callable[[str], list[int]],
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
        rng: np.random.Generator | int | None = None
    ) -> None:
        """Initialise with convergence parameters."""
        self._current_log_prob: float | None = None
        self._equil_branches = equil_branches
        self._max_branches = max_branches
        self._tolerance = tolerance
        self._rng = _as_rng(rng)

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
    ``query_next`` and keeps only the top ``beam_width`` cumulative
    log-probability branches at every depth.

    :param beam_width: Number of active beams retained per depth.
    :param branch_top_k: Number of next-token candidates considered for each
        active beam during expansion.
    """

    supports_token_beam = True

    def __init__(self, beam_width: int = 3, branch_top_k: int = 5) -> None:
        """Initialise beam-search width and per-beam expansion width."""
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
        branch_ctx: list[int],
        lookahead_depth: int,
        query_next: Callable[[list[int]], dict[str, float]],
        tokenize_token: Callable[[str], list[int]],
    ) -> np.float64:
        """Run token-level beam expansion and return the future log-weight."""
        beams: list[tuple[list[int], float]] = [(list(branch_ctx), 0.0)]

        for _ in range(lookahead_depth):
            expanded: list[tuple[list[int], float]] = []

            for beam_ctx, cum_lp in beams:
                next_token_lps = query_next(beam_ctx)
                top_items = sorted(
                    next_token_lps.items(),
                    key=lambda kv: kv[1],
                    reverse=True,
                )[: self.branch_top_k]

                for token_str, token_lp in top_items:
                    token_ids = tokenize_token(token_str)
                    if not token_ids:
                        continue
                    expanded.append((beam_ctx + token_ids, cum_lp + float(token_lp)))

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
        """Initialise with lookahead parameters and a branch sampler.

        :param alpha: Scaling exponent for the current token log-probability.
        :param lookahead_depth: Maximum depth of each branch.
        :param branch_sampler: The :class:`BranchSampler` to use within
            branches and for convergence decisions.
        """
        self.alpha = alpha
        self.lookahead_depth = lookahead_depth
        self.branch_sampler = branch_sampler

    def __call__(self, context: GenerationContext) -> dict[str, float]:
        """Apply power-distribution adjustment using lookahead branch sampling.

        :param context: The current generation context. Uses all fields:
            ``token_probs``, ``context_tokens``, ``query_next``, and
            ``tokenize_token``.
        :returns: Adjusted log-probability mapping combining the current
            token distribution with estimated future log-probabilities.
        """
        result: dict[str, float] = {}

        if self.branch_sampler.supports_token_beam:
            def score_future(branch_ctx: list[int]) -> np.float64:
                return self.branch_sampler.future_logprob_from_context(
                    alpha=self.alpha,
                    branch_ctx=branch_ctx,
                    lookahead_depth=self.lookahead_depth,
                    query_next=context.query_next,
                    tokenize_token=context.tokenize_token,
                )
        else:
            def score_future(branch_ctx: list[int]) -> np.float64:
                branch_log_probs_list: list[float] = []
                self.branch_sampler.reset()

                while True:
                    proposed_branch_log_prob = context.query_branch(
                        branch_ctx, self.lookahead_depth
                    )

                    accepted_log_prob = self.branch_sampler.step(
                        proposed_log_prob=proposed_branch_log_prob,
                        alpha=self.alpha,
                    )
                    branch_log_probs_list.append(accepted_log_prob)

                    if not self.branch_sampler.should_continue(
                        np.array(branch_log_probs_list, dtype=np.float64)
                    ):
                        break

                branch_log_probs = np.array(branch_log_probs_list, dtype=np.float64)
                return self.branch_sampler.future_logprob(self.alpha, branch_log_probs)

        candidate_bar = tqdm(
            context.token_probs.items(),
            desc='candidates',
            total=len(context.token_probs),
            unit='tok',
            leave=False,
        )
        for token, log_prob in candidate_bar:
            token_ids = context.tokenize_token(token)
            branch_ctx = list(context.context_tokens) + token_ids
            future_lp = score_future(branch_ctx)
            result[token] = self.alpha * log_prob + future_lp

        return result
