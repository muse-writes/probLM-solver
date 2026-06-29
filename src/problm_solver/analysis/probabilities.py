"""Utilities for sampling tokens and getting token probabilities."""


import numpy as np

from problm_solver.random import RNGLike, resolve_rng


def prob_of_token(token: str, log_probs: dict[str, float]) -> float:
    """Return the normalised probability of a specific token from a log-prob dict.

    Applies the same shift-exp-normalise procedure as
    :func:`sample_from_logprobs`, then returns the scalar probability for
    the named token rather than sampling.

    :param token: The token string to look up. Must be a key in ``log_probs``.
    :param log_probs: Mapping of token string to log-probability.
    :returns: The normalised probability of ``token`` in the distribution,
        in the range (0, 1].
    :raises KeyError: If ``token`` is not present in ``log_probs``.
    """
    tokens = list(log_probs.keys())
    lp = np.array([log_probs[t] for t in tokens], dtype=np.float64)
    lp -= lp.max()
    probs = np.exp(lp)
    probs /= probs.sum()
    return float(probs[tokens.index(token)])


def sample_from_logprobs(
    log_probs: dict[str, float],
    rng: RNGLike = None
) -> str:
    """Sample a token from a log-probability distribution.

    Converts log-probabilities to probabilities via ``exp()``, renormalises,
    and returns a single sampled token string.

    :param log_probs: Mapping of token string to log-probability. Values do
        not need to correspond to a normalised distribution — renormalisation
        is applied before sampling.
    :returns: A single sampled token string drawn from the distribution.
    """
    tokens = list(log_probs.keys())
    lp = np.array([log_probs[t] for t in tokens], dtype=np.float64)
    lp -= lp.max()  # shift for numerical stability before exp
    probs = np.exp(lp)
    probs /= probs.sum()
    method_rng = resolve_rng(rng, stream='analysis.sample_from_logprobs')
    idx: int = int(method_rng.choice(len(tokens), p=probs))
    return tokens[idx]
