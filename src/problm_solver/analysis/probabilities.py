"""Evaluates token probabilities from a probLM-solver dataset."""

from typing import Any

import numpy as np
import numpy.typing as npt

from problm_solver.data import LLMOutputData


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
