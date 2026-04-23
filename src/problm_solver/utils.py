"""Internal helper functions."""

import numpy as np


def _as_rng(rng: np.random.Generator | int | None) -> np.random.Generator:
    """Normalise a seed, None, or pass forward an existing generator."""
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)
