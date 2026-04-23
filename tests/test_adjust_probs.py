"""Tests for adjust_probs.py."""

import numpy as np
import pytest

from problm_solver.adjust_probs import AdjustFn, SampleLowTemp, adjust_identity


class TestAdjustIdentity:
    """Tests for adjust_identity."""

    def test_returns_dict(self) -> None:
        """adjust_identity() returns a dict."""
        result = adjust_identity({' hello': -0.5, ' world': -1.2}, [])
        assert isinstance(result, dict)

    def test_returns_input_unchanged(self) -> None:
        """adjust_identity() returns the token_probs dict unmodified."""
        token_probs = {' hello': -0.5, ' world': -1.2}
        result = adjust_identity(token_probs, [])
        assert result == token_probs

    def test_ignores_prev_probs(self) -> None:
        """adjust_identity() produces the same result regardless of prev_probs."""
        token_probs = {' a': -0.1, ' b': -0.5}
        assert adjust_identity(token_probs, []) == adjust_identity(token_probs, [0.9, 0.3])

    def test_satisfies_adjust_fn_signature(self) -> None:
        """adjust_identity is a valid AdjustFn (callable with correct signature)."""
        assert callable(adjust_identity)
        result = adjust_identity({' x': -1.0}, [0.5])
        assert isinstance(result, dict)


class TestSampleLowTempInit:
    """Tests for SampleLowTemp.__init__."""

    def test_stores_alpha(self) -> None:
        """alpha is stored as an instance attribute."""
        adj = SampleLowTemp(alpha=2)
        assert adj.alpha == 2

    def test_different_alpha_values_stored(self) -> None:
        """Different alpha values are stored independently."""
        adj1 = SampleLowTemp(alpha=1)
        adj2 = SampleLowTemp(alpha=3)
        assert adj1.alpha == 1
        assert adj2.alpha == 3

    def test_is_callable(self) -> None:
        """SampleLowTemp instances are callable."""
        assert callable(SampleLowTemp(alpha=2))

    def test_satisfies_adjust_fn_signature(self) -> None:
        """SampleLowTemp instances satisfy the AdjustFn interface."""
        adj = SampleLowTemp(alpha=2)
        result = adj({' hello': -0.5}, [])
        assert isinstance(result, dict)


class TestSampleLowTempCall:
    """Tests for SampleLowTemp.__call__."""

    @pytest.fixture()
    def adj(self) -> SampleLowTemp:
        """Return a SampleLowTemp instance with alpha=2."""
        return SampleLowTemp(alpha=2)

    def test_returns_dict(self, adj) -> None:
        """__call__() returns a dict."""
        result = adj({' hello': -0.5, ' world': -1.2}, [])
        assert isinstance(result, dict)

    def test_output_keys_match_input_keys(self, adj) -> None:
        """The returned dict has the same keys as the input."""
        token_probs = {' hello': -0.5, ' world': -1.2, ' foo': -2.0}
        result = adj(token_probs, [])
        assert set(result.keys()) == set(token_probs.keys())

    def test_output_values_are_floats(self, adj) -> None:
        """All values in the returned dict are floats."""
        result = adj({' hello': -0.5, ' world': -1.2}, [])
        assert all(isinstance(v, float) for v in result.values())

    def test_empty_prev_probs_gives_finite_output(self, adj) -> None:
        """With no previous tokens, output log-probs are finite for non-zero probs."""
        result = adj({' a': -0.1, ' b': -0.5}, [])
        assert all(np.isfinite(v) for v in result.values())

    def test_with_prev_probs_changes_output(self, adj) -> None:
        """Providing non-empty prev_probs produces a different result than empty."""
        token_probs = {' a': -0.1, ' b': -0.5}
        result_no_prev = adj(token_probs, [])
        result_with_prev = adj(token_probs, [0.8])
        # The absolute values differ; relative order may be preserved
        assert result_no_prev != result_with_prev

    def test_relative_order_preserved_with_empty_prev_probs(self, adj) -> None:
        """With no previous tokens, the most likely token remains most likely after adjustment."""
        token_probs = {' likely': -0.1, ' unlikely': -5.0}
        result = adj(token_probs, [])
        assert result[' likely'] > result[' unlikely']

    def test_alpha_one_preserves_relative_order(self) -> None:
        """With alpha=1 and no prev_probs, relative token ordering is unchanged."""
        adj1 = SampleLowTemp(alpha=1)
        token_probs = {' a': -0.2, ' b': -0.8, ' c': -2.0}
        result = adj1(token_probs, [])
        assert result[' a'] > result[' b'] > result[' c']

    def test_different_alpha_produces_different_output(self) -> None:
        """Different alpha values produce different adjusted distributions."""
        token_probs = {' a': -0.2, ' b': -1.0}
        result_a1 = SampleLowTemp(alpha=1)(token_probs, [])
        result_a3 = SampleLowTemp(alpha=3)(token_probs, [])
        assert result_a1 != result_a3


class TestAdjustFnTypeAlias:
    """Tests for the AdjustFn type alias."""

    def test_adjust_identity_is_valid_adjust_fn(self) -> None:
        """adjust_identity satisfies the AdjustFn calling convention."""
        fn: AdjustFn = adjust_identity
        assert fn({' a': -0.5}, []) == {' a': -0.5}

    def test_adjust_prob_power_instance_is_valid_adjust_fn(self) -> None:
        """A SampleLowTemp instance satisfies the AdjustFn calling convention."""
        fn: AdjustFn = SampleLowTemp(alpha=2)
        result = fn({' a': -0.5, ' b': -1.0}, [0.7])
        assert isinstance(result, dict)
