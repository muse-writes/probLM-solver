"""Internal helper functions."""

import logging

import numpy as np
from tqdm import tqdm


def _as_rng(rng: np.random.Generator | int | None) -> np.random.Generator:
    """Normalise a seed, None, or pass forward an existing generator."""
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


class TqdmHandler(logging.StreamHandler):
    """A logging handler that writes via tqdm.write(), preserving the progress bar."""

    def emit(self, record: logging.LogRecord) -> None:
        """Emit logging record via tqdm.write().

        :param record: Logging record to be written.
        """
        try:
            tqdm.write(self.format(record))
            self.flush()
        except Exception: # noqa: BLE001 - Logging must *not* crash the app under any circumstances.
            self.handleError(record)
