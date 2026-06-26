"""Tests for random.py."""

import numpy as np
import pytest

from problm_solver.random import RandomManager, resolve_rng


class TestRandomManager:
    """Tests for RandomManager."""

    def test_same_seed_same_stream_is_reproducible(self) -> None:
        """Two managers with same seed produce same sequence per stream name."""
        rm1 = RandomManager(seed=1234)
        rm2 = RandomManager(seed=1234)

        a1 = rm1.get_rng('metropolis').integers(0, 1000, size=20)
        a2 = rm2.get_rng('metropolis').integers(0, 1000, size=20)

        assert np.array_equal(a1, a2)

    def test_different_streams_have_different_sequences(self) -> None:
        """Distinct stream names produce different deterministic sequences."""
        rm = RandomManager(seed=1234)

        a = rm.get_rng('query').integers(0, 1_000_000, size=20)
        b = rm.get_rng('metropolis').integers(0, 1_000_000, size=20)

        assert not np.array_equal(a, b)

    def test_get_rng_returns_cached_generator(self) -> None:
        """get_rng returns the same generator instance for a stream name."""
        rm = RandomManager(seed=1234)

        g1 = rm.get_rng('query')
        g2 = rm.get_rng('query')

        assert g1 is g2

    def test_spawn_rng_returns_fresh_generators_reproducibly(self) -> None:
        """spawn_rng returns new generators but order is reproducible by seed."""
        rm1 = RandomManager(seed=1234)
        rm2 = RandomManager(seed=1234)

        s11 = rm1.spawn_rng('branch').integers(0, 1000, size=10)
        s12 = rm1.spawn_rng('branch').integers(0, 1000, size=10)

        s21 = rm2.spawn_rng('branch').integers(0, 1000, size=10)
        s22 = rm2.spawn_rng('branch').integers(0, 1000, size=10)

        assert np.array_equal(s11, s21)
        assert np.array_equal(s12, s22)
        assert not np.array_equal(s11, s12)

    def test_resolve_rng_raises_for_generator_with_fresh_true(self) -> None:
        """resolve_rng rejects fresh=True when rng is already a Generator."""
        gen = np.random.default_rng(1234)

        with pytest.raises(ValueError, match='fresh=True'):
            resolve_rng(gen, stream='query', fresh=True)
