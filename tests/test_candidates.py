"""Tests for CandidateGeneratorFactory routing and temperature scaling."""

import numpy as np
import pytest

from problm_solver.candidates import CandidateGeneratorFactory, CandidateTokens, log_softmax


def _logits() -> np.ndarray:
    """Return a representative raw-logits vector (descending-ish, with spread)."""
    return np.array([2.0, 5.0, 3.0, 0.5, -1.0, 1.0, 0.2], dtype=np.float32)


# ---------------------------------------------------------------------------
# get_candidate_generator: validation
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# get_candidate_generator: routing
# ---------------------------------------------------------------------------
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
        nucleus = int(np.searchsorted(np.cumsum(probs), 0.6, side='right') + 1)
        assert len(result.candidate_ids) == nucleus
        assert result.candidate_ids.tolist() == [int(i) for i in order[:nucleus]]

    def test_combined_routes_to_top_k_p(self) -> None:
        """Both enabled routes to the combined top-k/top-p truncation."""
        logits = _logits()
        result = CandidateGeneratorFactory().get_candidate_generator(top_k=4, top_p=0.6)(logits)
        lp = log_softmax(logits)
        order = np.argsort(lp)[::-1][:4]
        cumulative = np.cumsum(np.exp(lp[order]))
        expected = min(int(np.searchsorted(cumulative, 0.6, side='right')) + 1, 4)
        assert len(result.candidate_ids) == expected


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
        cumulative = np.cumsum(np.exp(lp[order]))
        keep_n = min(int(np.searchsorted(cumulative, top_p, side='right')) + 1, top_k)
        result = CandidateGeneratorFactory().get_candidate_generator(
            top_k=top_k, top_p=top_p, alpha=alpha
        )(logits)
        assert result.candidate_ids.tolist() == order[:keep_n]
