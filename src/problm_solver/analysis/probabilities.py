"""Evaluates token probabilities from a probLM-solver dataset."""

from typing import Any

import numpy as np
import numpy.typing as npt

from problm_solver.data import LLMOutputData


def sample_from_logprobs(log_probs: dict[str, float]) -> str:
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
    idx: int = int(np.random.choice(len(tokens), p=probs))
    return tokens[idx]


class Probabilities:
    """Probabilities associated with a dataset."""

    def __init__(self, data: LLMOutputData, entry: str) -> None:
        """Initialize."""
        self.data: npt.NDArray[Any] = data.data
        self.probs = None

# The response for which probabilities are calculated.
        self.entry_tokens: List[str] = entry.split()


    def evaluate(self) -> None:
        """Calculate probabilities."""
        probabilities = np.empty(len(self.entry_tokens), dtype=np.float64)
        for ii in range(len(self.entry_tokens)):
            for data_entry in data:
                pass
