"""Tests for backend adapters in llama_lowlevel.py."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from problm_solver.llama_lowlevel import ModelCBackend, ModelLlamaBackend


def _make_llm_mock() -> MagicMock:
    """Create a minimal Llama-like mock for backend adapter tests."""
    mock_llm = MagicMock()
    mock_llm.n_tokens = 7
    mock_llm.scores = np.array([[0.1, 0.2], [1.0, -1.0]], dtype=np.float32)
    mock_llm.token_eos.return_value = 2
    mock_llm.token_bos.return_value = 1
    mock_llm.tokenize.return_value = [11, 12]
    mock_llm.detokenize.return_value = b'x'
    mock_llm.metadata = {'key': 'value'}
    mock_llm.n_ctx.return_value = 4096
    return mock_llm


@pytest.mark.parametrize('backend_cls', [ModelCBackend, ModelLlamaBackend])
class TestModelBackends:
    """Shared delegation tests for both backend adapter implementations."""

    def test_n_tokens_property(self, backend_cls) -> None:
        """n_tokens exposes the wrapped model's token count as int."""
        llm = _make_llm_mock()
        backend = backend_cls(llm)
        assert backend.n_tokens == 7

    def test_reset_delegates(self, backend_cls) -> None:
        """reset() delegates to the wrapped model."""
        llm = _make_llm_mock()
        backend = backend_cls(llm)
        backend.reset()
        llm.reset.assert_called_once_with()

    def test_decode_delegates_to_eval(self, backend_cls) -> None:
        """decode() forwards tokens to eval()."""
        llm = _make_llm_mock()
        backend = backend_cls(llm)
        backend.decode([4, 5])
        llm.eval.assert_called_once_with([4, 5])

    def test_last_logits_uses_latest_row(self, backend_cls) -> None:
        """last_logits() returns scores[n_tokens - 1]."""
        llm = _make_llm_mock()
        llm.n_tokens = 2
        backend = backend_cls(llm)
        result = backend.last_logits()
        assert np.array_equal(result, np.array([1.0, -1.0], dtype=np.float32))

    def test_token_helpers_delegate(self, backend_cls) -> None:
        """token_eos/token_bos delegate and return ints."""
        llm = _make_llm_mock()
        backend = backend_cls(llm)
        assert backend.token_eos() == 2
        assert backend.token_bos() == 1

    def test_tokenize_and_detokenize_delegate(self, backend_cls) -> None:
        """tokenize/detokenize calls are passed through."""
        llm = _make_llm_mock()
        backend = backend_cls(llm)
        assert backend.tokenize(b'hi', add_bos=False, special=True) == [11, 12]
        assert backend.detokenize([11], special=True) == b'x'
        llm.tokenize.assert_called_once_with(b'hi', add_bos=False, special=True)
        llm.detokenize.assert_called_once_with([11], special=True)

    def test_metadata_returns_dict_copy(self, backend_cls) -> None:
        """metadata() returns a dict copy, not the original object."""
        llm = _make_llm_mock()
        backend = backend_cls(llm)
        meta = backend.metadata()
        assert meta == {'key': 'value'}
        assert meta is not llm.metadata

    def test_n_ctx_cached_after_first_call(self, backend_cls) -> None:
        """n_ctx() reads model once and reuses cached value."""
        llm = _make_llm_mock()
        backend = backend_cls(llm)
        assert backend.n_ctx() == 4096
        assert backend.n_ctx() == 4096
        llm.n_ctx.assert_called_once_with()
