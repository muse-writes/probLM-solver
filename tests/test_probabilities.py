"""Tests for analysis/probabilities.py."""

import numpy as np
import pytest

from problm_solver.analysis.probabilities import prob_of_token, sample_from_logprobs


class TestProbOfToken:
    """Tests for prob_of_token."""

    def test_returns_float(self) -> None:
        """prob_of_token() returns a float."""
        result = prob_of_token(' hello', {' hello': -0.5, ' world': -1.2})
        assert isinstance(result, float)

    def test_result_is_between_zero_and_one(self) -> None:
        """The returned probability is in the range (0, 1]."""
        result = prob_of_token(' hello', {' hello': -0.5, ' world': -1.2})
        assert 0.0 < result <= 1.0

    def test_single_token_returns_one(self) -> None:
        """A single-entry dict gives a probability of 1.0."""
        result = prob_of_token(' only', {' only': -99.9})
        assert result == pytest.approx(1.0)

    def test_dominant_token_has_high_probability(self) -> None:
        """A token with a much higher log-prob gets a probability close to 1."""
        result = prob_of_token(' yes', {' yes': 0.0, ' no': -1000.0})
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_equal_logprobs_give_equal_probabilities(self) -> None:
        """Tokens with equal log-probs each receive probability 1/N."""
        lp = {' a': -1.0, ' b': -1.0, ' c': -1.0}
        for token in lp:
            assert prob_of_token(token, lp) == pytest.approx(1 / 3)

    def test_probabilities_sum_to_one(self) -> None:
        """Probabilities for all tokens in the dict sum to 1."""
        log_probs = {' a': -0.1, ' b': -0.5, ' c': -1.0}
        total = sum(prob_of_token(t, log_probs) for t in log_probs)
        assert total == pytest.approx(1.0)

    def test_raises_for_missing_token(self) -> None:
        """prob_of_token raises when the requested token is not in the dict."""
        with pytest.raises((KeyError, ValueError)):
            prob_of_token(' missing', {' hello': -0.5})

    def test_consistent_with_sample_from_logprobs_normalisation(self) -> None:
        """prob_of_token and sample_from_logprobs use the same normalisation."""
        log_probs = {' a': -0.2, ' b': -0.8}
        prob_a = prob_of_token(' a', log_probs)
        prob_b = prob_of_token(' b', log_probs)
        assert prob_a + prob_b == pytest.approx(1.0)
        assert prob_a > prob_b  # ' a' has higher log-prob so higher probability



class TestSampleFromLogprobs:
    """Tests for sample_from_logprobs."""

    def test_returns_string(self) -> None:
        """sample_from_logprobs() returns a plain string."""
        result = sample_from_logprobs({' hello': -0.5, ' world': -1.2})
        assert isinstance(result, str)

    def test_returned_token_is_in_input(self) -> None:
        """The returned token is always one of the keys in the input dict."""
        log_probs = {' hello': -0.5, ' world': -1.2, ' foo': -2.0}
        result = sample_from_logprobs(log_probs)
        assert result in log_probs

    def test_single_token_always_returned(self) -> None:
        """A single-entry dict always returns that entry regardless of its value."""
        result = sample_from_logprobs({' only': -99.9})
        assert result == ' only'

    def test_dominant_token_sampled_with_fixed_seed(self) -> None:
        """With a heavily dominant token, it is returned under a fixed seed."""
        np.random.seed(0)
        # ' yes' has log_prob 0, others are -1000 — effectively probability 1
        result = sample_from_logprobs({' yes': 0.0, ' no': -1000.0, ' maybe': -1000.0})
        assert result == ' yes'

    def test_numerically_stable_with_very_negative_logprobs(self) -> None:
        """No underflow/overflow when all log-probs are very negative."""
        # Without the max-subtraction trick, exp(-1000) underflows to 0
        # and normalisation would produce NaN. This must not raise or return NaN.
        result = sample_from_logprobs({' a': -1000.0, ' b': -1000.5})
        assert result in {' a', ' b'}

    def test_unnormalised_input_produces_valid_output(self) -> None:
        """Input log-probs that do not sum-to-one still yield a valid token."""
        # These do not correspond to a normalised distribution — that is fine.
        result = sample_from_logprobs({' x': 5.0, ' y': 3.0, ' z': 1.0})
        assert result in {' x', ' y', ' z'}

    def test_all_samples_are_valid_over_many_draws(self) -> None:
        """Every token sampled over many draws is a key in the input dict."""
        log_probs = {' a': -0.1, ' b': -0.5, ' c': -1.0}
        for _ in range(50):
            assert sample_from_logprobs(log_probs) in log_probs
