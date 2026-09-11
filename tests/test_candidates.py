"""Tests for CandidateGeneratorFactory routing and temperature scaling."""

import numpy as np
import pytest

from problm_solver.candidates import CandidateGeneratorFactory, CandidateTokens, log_softmax


def _logits() -> np.ndarray:
    """Return a representative raw-logits vector (descending-ish, with spread)."""
    return np.array([2.0, 5.0, 3.0, 0.5, -1.0, 1.0, 0.2], dtype=np.float32)


# Validation of the generator factory.
class TestGeneratorValidation:
    """Tests for get_candidate_generator parameter validation."""

    @pytest.mark.parametrize('alpha', [0.0, -1.0, -0.5])
    def test_non_positive_alpha_raises(self, alpha: float) -> None:
        """Alpha must be strictly positive (HF: temperature > 0)."""
        with pytest.raises(ValueError, match='alpha > 0'):
            CandidateGeneratorFactory().get_candidate_generator(top_k=3, top_p=1.0, alpha=alpha)

    @pytest.mark.parametrize('alpha', [float('nan'), float('inf')])
    def test_non_finite_alpha_raises(self, alpha: float) -> None:
        """Infinite or NaN temperature is impermissible."""
        with pytest.raises(ValueError, match='alpha > 0'):
            CandidateGeneratorFactory().get_candidate_generator(top_k=3, top_p=1.0, alpha=alpha)

    def test_top_k_below_one_raises(self) -> None:
        """top_k must be at least 1 when provided."""
        with pytest.raises(ValueError, match='top_k'):
            CandidateGeneratorFactory().get_candidate_generator(top_k=0, top_p=1.0)

    @pytest.mark.parametrize('top_p', [0.0, -0.5, 1.5])
    def test_top_p_out_of_range_raises(self, top_p: float) -> None:
        """top_p must lie in (0, 1] when provided."""
        with pytest.raises(ValueError, match='top_p'):
            CandidateGeneratorFactory().get_candidate_generator(top_k=3, top_p=top_p)

    def test_optional_parameters_default_to_disabled(self) -> None:
        """Omitting top_k and top_p is valid and yields a generator."""
        generator = CandidateGeneratorFactory().get_candidate_generator()
        result = generator(_logits())
        assert isinstance(result, CandidateTokens)


# Routing of the generator factory.
class TestGeneratorRouting:
    """Tests for the truncation routing table."""

    def test_both_disabled_returns_full_vocabulary(self) -> None:
        """top_k=None, top_p=None returns every token with log-softmax values."""
        logits = _logits()
        result = CandidateGeneratorFactory().get_candidate_generator(top_k=None, top_p=None)(logits)
        assert result.candidate_ids.tolist() == list(range(len(logits)))
        assert result.candidate_logprobs == pytest.approx(log_softmax(logits).tolist())

    def test_top_p_of_one_disables_top_p(self) -> None:
        """top_p=1.0 behaves identically to top_p=None for the same top_k."""
        logits = _logits()
        factory = CandidateGeneratorFactory()
        a = factory.get_candidate_generator(top_k=3, top_p=None)(logits)
        b = factory.get_candidate_generator(top_k=3, top_p=1.0)(logits)
        assert a.candidate_ids.tolist() == b.candidate_ids.tolist()
        assert a.candidate_logprobs == pytest.approx(b.candidate_logprobs)

    def test_both_disabled_values_return_full_vocabulary(self) -> None:
        """top_k=None with top_p=1.0 disables both stages (HF guard parity)."""
        logits = _logits()
        result = CandidateGeneratorFactory().get_candidate_generator(top_k=None, top_p=1.0)(logits)
        assert result.candidate_ids.tolist() == list(range(len(logits)))

    def test_top_k_of_one_returns_argmax(self) -> None:
        """top_k=1 pins the argmax token regardless of top_p."""
        logits = _logits()
        expected = int(np.argmax(log_softmax(logits)))
        for top_p in (None, 1.0, 0.9):
            result = CandidateGeneratorFactory().get_candidate_generator(top_k=1, top_p=top_p)(logits)
            assert result.candidate_ids.tolist() == [expected]
            assert len(result.candidate_ids) == 1

    def test_top_k_selects_highest_tokens(self) -> None:
        """Pure top-k keeps exactly the k highest-logprob tokens."""
        logits = _logits()
        lp = log_softmax(logits)
        expected = [int(i) for i in np.argsort(lp)[::-1][:3]]
        result = CandidateGeneratorFactory().get_candidate_generator(top_k=3, top_p=1.0)(logits)
        assert result.candidate_ids.tolist() == expected

    def test_top_k_only_with_none_top_p(self) -> None:
        """top_k with top_p=None is pure top-k."""
        logits = _logits()
        factory = CandidateGeneratorFactory()
        a = factory.get_candidate_generator(top_k=3, top_p=None)(logits)
        b = factory.get_candidate_generator(top_k=3, top_p=1.0)(logits)
        assert a.candidate_ids.tolist() == b.candidate_ids.tolist()

    def test_top_p_only_when_top_k_disabled(self) -> None:
        """top_k=None with top_p<1 performs pure nucleus truncation."""
        logits = _logits()
        result = CandidateGeneratorFactory().get_candidate_generator(top_k=None, top_p=0.6)(logits)
        lp = log_softmax(logits)
        order = np.argsort(lp)[::-1]
        probs = np.exp(lp[order])
        nucleus = int(np.searchsorted(np.cumsum(probs), 0.6, side='left') + 1)
        assert len(result.candidate_ids) == nucleus
        assert result.candidate_ids.tolist() == [int(i) for i in order[:nucleus]]

    def test_combined_routes_to_top_k_p(self) -> None:
        """Both enabled routes to the combined top-k/top-p truncation."""
        logits = _logits()
        result = CandidateGeneratorFactory().get_candidate_generator(top_k=4, top_p=0.6)(logits)
        lp = log_softmax(logits)
        order = np.argsort(lp)[::-1][:4]
        renorm = lp[order] - np.log(np.exp(lp[order]).sum())
        cumulative = np.cumsum(np.exp(renorm))
        expected = min(int(np.searchsorted(cumulative, 0.6, side='left')) + 1, 4)
        assert len(result.candidate_ids) == expected


# ---------------------------------------------------------------------------
# Tie rule (HuggingFace TopKLogitsWarper parity)
# ---------------------------------------------------------------------------
class TestGeneratorTieRule:
    """Tests for the boundary-tie rule (tokens tied at the k-th value are kept)."""

    @staticmethod
    def _tied_logits() -> np.ndarray:
        """Return logits whose 3rd/4th/5th tokens are exactly tied at prob 0.1."""
        return np.log(np.array([0.4, 0.3, 0.1, 0.1, 0.1, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9]))

    def test_top_k_keeps_all_boundary_ties(self) -> None:
        """top_k=3 over a tie group at the boundary keeps all tied tokens (HF parity)."""
        n_expected = 5
        logits = self._tied_logits().astype(np.float32)
        result = CandidateGeneratorFactory().get_candidate_generator(top_k=3, top_p=1.0)(logits)
        assert set(result.candidate_ids.tolist()) == {0, 1, 2, 3, 4}
        assert len(result.candidate_ids) == n_expected

    def test_top_k_keeps_exactly_k_without_ties(self) -> None:
        """Distinct boundary values still yield exactly k candidates."""
        n_expected = 3
        logits = np.log(np.array([0.4, 0.3, 0.2, 0.05, 0.02, 0.03]), dtype=np.float32)
        result = CandidateGeneratorFactory().get_candidate_generator(top_k=3, top_p=1.0)(logits)
        assert len(result.candidate_ids) == n_expected
        assert set(result.candidate_ids.tolist()) == {0, 1, 2}

    def test_top_k_keeps_ties_above_the_boundary(self) -> None:
        """Tokens tied at the max survive even when they already exceed k."""
        logits = np.log(np.array([0.4, 0.4, 0.1, 0.1]), dtype=np.float32)
        result = CandidateGeneratorFactory().get_candidate_generator(top_k=2, top_p=1.0)(logits)
        assert set(result.candidate_ids.tolist()) == {0, 1}

    def test_tie_order_is_descending_value_then_ascending_id(self) -> None:
        """Output is ordered descending by value, with ties in ascending id order."""
        logits = self._tied_logits().astype(np.float32)
        result = CandidateGeneratorFactory().get_candidate_generator(top_k=3, top_p=1.0)(logits)
        assert result.candidate_ids.tolist() == [0, 1, 2, 3, 4]
        lp = result.candidate_logprobs
        assert np.all(lp[:-1] >= lp[1:])
        # Equal-valued neighbours keep ascending token ids.
        for i in range(len(lp) - 1):
            if lp[i] == lp[i + 1]:
                assert result.candidate_ids[i] < result.candidate_ids[i + 1]

    def test_combined_truncates_kept_tie_set_with_top_p(self) -> None:
        """The top-p stage truncates the (possibly enlarged) tied kept set."""
        logits = self._tied_logits().astype(np.float32)
        # Kept top-k set: {0,1,2,3,4} (ties kept); un-renormalised cumulative:
        # [0.4, 0.7, 0.8, 0.9, 1.0] -> first > 0.75 is 0.8 (index 2) -> keep 3.
        result = CandidateGeneratorFactory().get_candidate_generator(top_k=3, top_p=0.75)(logits)
        assert result.candidate_ids.tolist() == [0, 1, 2]

    def test_repeated_calls_are_deterministic(self) -> None:
        """Repeated calls with tied values produce identical output."""
        logits = self._tied_logits().astype(np.float32)
        generator = CandidateGeneratorFactory().get_candidate_generator(top_k=3, top_p=1.0)
        a = generator(logits)
        b = generator(logits)
        assert np.array_equal(a.candidate_ids, b.candidate_ids)
        assert np.array_equal(a.candidate_logprobs, b.candidate_logprobs)


# ---------------------------------------------------------------------------
# Renormalisation and threshold strictness
# ---------------------------------------------------------------------------
class TestCombinedRenormalisation:
    """Tests for Issue 4 (renormalised top-k mass) and Issue 3 (non-strict threshold)."""

    def test_combined_renormalises_top_k_mass(self) -> None:
        """top-p is a fraction of the renormalised top-k mass (HF parity)."""
        # Top-5 mass is 0.85 < top_p=0.9; renormalised, the nucleus is 4 tokens.
        probs = np.array([0.30, 0.20, 0.10, 0.08, 0.05, 0.04, 0.03, 0.02, 0.01, 0.17])
        logits = np.log(probs, dtype=np.float32)
        result = CandidateGeneratorFactory().get_candidate_generator(top_k=5, top_p=0.9)(logits)
        assert set(result.candidate_ids.tolist()) == {0, 1, 2, 9}
        assert len(result.candidate_ids) == 4

    def test_top_p_stops_at_exact_threshold(self) -> None:
        """A cumulative exactly equal to top_p stops the nucleus (non-strict rule)."""
        logits = np.log(np.array([0.5, 0.5]), dtype=np.float32)
        result = CandidateGeneratorFactory().get_candidate_generator(top_k=None, top_p=0.5)(logits)
        # cum = [0.5, 1.0]; first cum >= 0.5 is the top token itself.
        assert len(result.candidate_ids) == 1

    def test_combined_returns_original_logprobs(self) -> None:
        """The renormalisation only affects selection; stored logprobs are unscaled."""
        probs = np.array([0.30, 0.20, 0.10, 0.08, 0.05, 0.04, 0.03, 0.02, 0.01, 0.17])
        logits = np.log(probs, dtype=np.float32)
        result = CandidateGeneratorFactory().get_candidate_generator(top_k=5, top_p=0.9)(logits)
        expected = log_softmax(logits)
        for token_id, value in zip(result.candidate_ids, result.candidate_logprobs, strict=True):
            assert float(value) == pytest.approx(float(expected[token_id]))


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------
class TestGeneratorTemperature:
    """Tests for alpha (inverse temperature) handling."""

    def test_alpha_one_matches_default(self) -> None:
        """alpha=1.0 produces identical output to the default."""
        logits = _logits()
        factory = CandidateGeneratorFactory()
        a = factory.get_candidate_generator(top_k=3, top_p=1.0, alpha=1.0)(logits)
        b = factory.get_candidate_generator(top_k=3, top_p=1.0)(logits)
        assert a.candidate_ids.tolist() == b.candidate_ids.tolist()
        assert a.candidate_logprobs == pytest.approx(b.candidate_logprobs)

    def test_temp_variant_scales_logits_before_softmax(self) -> None:
        """Scaled log-probs equal log_softmax(alpha * logits) on the kept tokens."""
        logits = _logits()
        alpha = 2.0
        result = CandidateGeneratorFactory().get_candidate_generator(
            top_k=3, top_p=1.0, alpha=alpha
        )(logits)
        expected_full = log_softmax(logits * np.float64(alpha))
        for token_id, value in zip(result.candidate_ids, result.candidate_logprobs, strict=True):
            assert float(value) == pytest.approx(float(expected_full[token_id]))

    def test_no_op_temp_returns_full_vocabulary_scaled(self) -> None:
        """The no-truncation temp variant spans the vocabulary with scaled logprobs."""
        logits = _logits()
        result = CandidateGeneratorFactory().get_candidate_generator(
            top_k=None, top_p=None, alpha=2.0
        )(logits)
        assert result.candidate_ids.tolist() == list(range(len(logits)))
        assert result.candidate_logprobs == pytest.approx(log_softmax(logits * np.float64(2.0)))

    def test_sharpening_shrinks_nucleus(self) -> None:
        """Alpha > 1 sharpens the distribution, so the top-p nucleus shrinks."""
        logits = _logits()
        factory = CandidateGeneratorFactory()
        plain = factory.get_candidate_generator(top_k=None, top_p=0.6, alpha=1.0)(logits)
        sharp = factory.get_candidate_generator(top_k=None, top_p=0.6, alpha=4.0)(logits)
        assert len(sharp.candidate_ids) <= len(plain.candidate_ids)

    def test_flattening_grows_nucleus(self) -> None:
        """Alpha < 1 flattens the distribution, so the top-p nucleus grows."""
        logits = _logits()
        factory = CandidateGeneratorFactory()
        plain = factory.get_candidate_generator(top_k=None, top_p=0.6, alpha=1.0)(logits)
        flat = factory.get_candidate_generator(top_k=None, top_p=0.6, alpha=0.5)(logits)
        assert len(flat.candidate_ids) >= len(plain.candidate_ids)

    def test_selected_tokens_identical_to_unscaled_for_extreme_alpha(self) -> None:
        """Temperature never changes the argmax for positive alpha (monotone scaling)."""
        logits = _logits()
        factory = CandidateGeneratorFactory()
        plain = factory.get_candidate_generator(top_k=1, top_p=1.0, alpha=1.0)(logits)
        sharp = factory.get_candidate_generator(top_k=1, top_p=1.0, alpha=8.0)(logits)
        assert plain.candidate_ids.tolist() == sharp.candidate_ids.tolist()

    def test_kept_set_matches_reference_math(self) -> None:
        """The temp combined path equals the reference nucleus on the scaled softmax."""
        logits = _logits()
        alpha, top_k, top_p = 0.5, 5, 0.7
        lp = log_softmax(logits * np.float64(alpha))
        order = [int(i) for i in np.argsort(lp)[::-1][:top_k]]
        renorm = lp[order] - np.log(np.exp(lp[order]).sum())
        cumulative = np.cumsum(np.exp(renorm))
        keep_n = min(int(np.searchsorted(cumulative, top_p, side='left')) + 1, top_k)
        result = CandidateGeneratorFactory().get_candidate_generator(
            top_k=top_k, top_p=top_p, alpha=alpha
        )(logits)
        assert result.candidate_ids.tolist() == order[:keep_n]
