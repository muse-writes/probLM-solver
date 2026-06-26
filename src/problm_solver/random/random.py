"""Module containing RNG utilities."""

from __future__ import annotations

import hashlib

import numpy as np

## -- RNG management class -- ##

class RandomManager:
    """Manage deterministic pseudo-random streams from a single user seed.

    A ``RandomManager`` provides named RNG streams so the caller can set one
    seed and still keep randomness in different subsystems independent
    (e.g. token sampling vs. Metropolis acceptance draws).
    """

    def __init__(self, seed: int = 314159) -> None:
        """Initialise random manager.

        :param seed: Root integer seed for all derived streams.
        """
        self._seed = int(seed)
        self._streams: dict[str, np.random.Generator] = {}
        self._spawn_counts: dict[str, int] = {}

    @property
    def seed(self) -> int:
        """Return root seed value."""
        return self._seed

    def get_rng(self, stream: str = 'global') -> np.random.Generator:
        """Return a cached RNG for a named stream.

        Repeated calls with the same ``stream`` return the same generator
        instance so state advances naturally across calls.

        :param stream: Name of the deterministic stream.
        :returns: ``numpy.random.Generator`` bound to ``stream``.
        """
        if stream not in self._streams:
            self._streams[stream] = np.random.default_rng(self._seed_sequence(stream, index=0))
        return self._streams[stream]

    def spawn_rng(self, stream: str = 'global') -> np.random.Generator:
        """Return a fresh RNG in a named stream family.

        Unlike :meth:`get_rng`, each call returns a *new* generator that is
        deterministic for ``(seed, stream, spawn_index)``.

        :param stream: Name of stream family.
        :returns: Fresh deterministic ``numpy.random.Generator`` instance.
        """
        index = self._spawn_counts.get(stream, 0)
        self._spawn_counts[stream] = index + 1
        return np.random.default_rng(self._seed_sequence(stream, index=index))

    def _seed_sequence(self, stream: str, index: int) -> np.random.SeedSequence:
        """Create stable SeedSequence for a stream name and index.

        :param stream: Stream name.
        :param index: Stream-local spawn index.
        :returns: Stable seed sequence for deterministic RNG initialisation.
        """
        digest = hashlib.blake2b(stream.encode('utf-8'), digest_size=16).digest()
        stream_words = np.frombuffer(digest, dtype=np.uint32).tolist()
        entropy = [self._seed, int(index), *stream_words]
        return np.random.SeedSequence(entropy)

