"""Implement several adjustment functions for generate_adjusted."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from problm_solver.utils import _as_rng


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
        ``n_tokens``. Returns ``None`` on EOS.
    :param query_branch: Generates a complete branch of up to ``depth`` tokens
        from the given context in a single model call and returns the sum of
        per-token log-probabilities. Returns ``0.0`` on immediate EOS.
    :param tokenize_token: Converts a single token string to its token ID(s).
    """

    token_probs: dict[str, float]
    prev_probs: list[float]
    context_tokens: list[int]
    query_next: Callable[[list[int]], dict[str, float] | None]
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

    A ``BranchSampler`` runs a Markov chain over complete branch proposals.
    :meth:`step` receives a proposed branch log-probability and returns the
    chain state after accept/reject. :meth:`should_continue` decides whether
    to keep proposing additional branches.
    """

    def reset(self) -> None:
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
        """Process one proposed branch and return the accepted chain state.

        :param proposed_log_prob: Log-probability of the proposed branch under
            the base model.
        :param alpha: Power-distribution exponent. The acceptance ratio targets
            ``π(x) ∝ p(x)^α``, so the ratio is computed as
            ``(α-1) * (log p(x') - log p(x)) + log q(x|x') - log q(x'|x)``.
        :param forward_log_q: Log proposal probability ``log q(x'|x)``.
        :param reverse_log_q: Log reverse proposal probability ``log q(x|x')``.
        :returns: The current chain state's branch log-probability after
            accept/reject.
        """

    @abstractmethod
    def should_continue(self, branch_log_probs: npt.NDArray[np.float64]) -> bool:
        """Return ``True`` if more branch proposals should be sampled.

        Called after each completed MCMC step with all accepted branch
        log-probabilities so far.

        :param branch_log_probs: 1-D array of accepted branch log-probabilities
            from all completed steps.
        :returns: ``True`` to sample another proposal, ``False`` to stop.
        """


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
# TODO(Clio): Investigate viable convergence tolerances.
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
        n = len(branch_log_probs)
        if n < self._equil_branches:
            return True
        if n >= self._max_branches:
            return False
# TODO(Clio): Write proper equilibration algorithm?? Allow input control?
# Discard equilibration probabilities in SEM calculation.
# set to > equil_branches a.t.m.
# Noise in probabilities is likely to be quite high, think more about this.
        post_eq = branch_log_probs[self._equil_branches:]
        if len(post_eq) <= 1:
            return True
        sem = float(np.std(post_eq) / np.sqrt(len(post_eq)))
        return sem >= self._tolerance


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

        for token, log_prob in context.token_probs.items():
            token_ids = context.tokenize_token(token)
            branch_ctx = list(context.context_tokens) + token_ids
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

# Combine via log-sum-exp over alpha-scaled branch log-probs
            scaled = self.alpha * branch_log_probs
            max_lp = scaled.max()
            future_lp = float(
                np.log(np.sum(np.exp(scaled - max_lp))) + max_lp
            )
            result[token] = self.alpha * log_prob + future_lp

        return result
