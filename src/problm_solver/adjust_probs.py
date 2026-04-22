"""Implement several adjustment functions for generate_adjusted."""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

# Callable that receives a top-M {token: log_prob} dict and the list of
# normalised probabilities of all previously selected tokens in this
# generation, and returns a modified log-prob dict. Values need not be
# normalised; renormalisation is applied by sample_from_logprobs before
# sampling.
AdjustFn = Callable[[dict[str, float], list[float]], dict[str, float]]


def adjust_identity(
    token_probs: dict[str, float],
    prev_probs: list[float],  # noqa: ARG001
) -> dict[str, float]:
    """Return token log-probabilities unchanged.

    Satisfies the :data:`AdjustFn` interface without modifying the
    distribution. Useful as a baseline and for testing.

    :param token_probs: Top-M mapping of token string to log-probability.
    :param prev_probs: Unused; included to satisfy the :data:`AdjustFn`
        interface.
    :returns: ``token_probs`` unmodified.
    """
    return token_probs


class AdjustProbPower:
    """Adjust token log-probabilities by power-scaling with selection history.

    At each generation step, the current token probabilities are raised to
    ``alpha`` and multiplied by the product of all previously selected token
    probabilities each also raised to ``alpha``. The result is returned as
    log-probabilities for downstream renormalisation and sampling.

    :param alpha: Scaling exponent. Values greater than 1 sharpen the
        distribution (favouring already-likely tokens); values between 0
        and 1 flatten it.

    Example usage::

        adjust_fn = AdjustProbPower(alpha=2)
        result = adjust_fn(token_probs, prev_probs)
    """

    def __init__(self, alpha: int) -> None:
        """Initialise with scaling exponent.

        :param alpha: Exponent applied to current and previous token
            probabilities when computing the adjustment.
        """
        self.alpha = alpha

    def __call__(
        self,
        token_probs: dict[str, float],
        prev_probs: list[float],
    ) -> dict[str, float]:
        """Apply power-scaling adjustment to the current token distribution.

        :param token_probs: Top-M mapping of token string to log-probability
            at the current generation step.
        :param prev_probs: Normalised probabilities of all previously selected
            tokens in this generation. Empty on the first step.
        :returns: Adjusted log-probability mapping. Note that tokens with very
            low probability may produce ``-inf`` log-probabilities after
            scaling.
        """
        tokens = list(token_probs.keys())
        lp = np.array([token_probs[t] for t in tokens], dtype=float)
        lp -= lp.max()
        p: npt.NDArray = np.exp(lp)
        prev_alpha: float = np.prod(np.array(prev_probs, dtype=float) ** self.alpha)
        new_logprobs: npt.NDArray = np.log(p ** self.alpha * prev_alpha)
        return dict(zip(tokens, list(new_logprobs), strict=True))
