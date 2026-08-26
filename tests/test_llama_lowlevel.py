"""Tests for backend adapters in llama_lowlevel.py."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from problm_solver.llama_lowlevel import ModelCBackend


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


@pytest.mark.parametrize('backend_cls', [ModelCBackend])
class TestModelBackends:
    """Shared delegation tests for the backend adapter implementation."""

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


def _make_batched_llm_mock(n_vocab: int = 4) -> MagicMock:
    """Create a Llama-like mock with a wider vocab for batched-decode tests."""
    mock_llm = _make_llm_mock()
    mock_llm.scores = np.zeros((8, n_vocab), dtype=np.float32)
    return mock_llm


class TestDecodeBatch:
    """Tests for ModelCBackend.decode_batch (multi-sequence batched decoding)."""

    def test_empty_input_returns_zero_rows(self) -> None:
        """decode_batch with no active sequences returns a (0, vocab) array."""
        llm = _make_batched_llm_mock(n_vocab=4)
        backend = ModelCBackend(llm)
        result = backend.decode_batch([], [], [])
        assert result.shape == (0, 4)
        assert result.dtype == np.float32

    def test_length_mismatch_raises_value_error(self) -> None:
        """Mismatched token/seq/position lengths raise ValueError."""
        llm = _make_batched_llm_mock()
        backend = ModelCBackend(llm)
        with pytest.raises(ValueError, match='equal length'):
            backend.decode_batch([1, 2], [0], [0, 1])
        with pytest.raises(ValueError, match='equal length'):
            backend.decode_batch([1, 2], [0, 1], [0])

    def test_oversize_batch_raises_value_error(self) -> None:
        """More tokens than batch capacity raises ValueError."""
        llm = _make_batched_llm_mock()
        llm.n_ctx.return_value = 2
        backend = ModelCBackend(llm)
        with pytest.raises(ValueError, match='batch capacity'):
            backend.decode_batch([1, 2, 3], [0, 1, 2], [0, 1, 2])

    def test_fills_batch_fields_per_sequence(self) -> None:
        """token/pos/seq_id/n_seq_id/logits are written for each active row."""
        llm = _make_batched_llm_mock(n_vocab=4)
        backend = ModelCBackend(llm)
        backend.decode_batch([10, 20, 30], [0, 1, 2], [5, 6, 7])
        batch = backend._batch  # noqa: SLF001
        assert batch.n_tokens == 3
        assert [batch.token[i] for i in range(3)] == [10, 20, 30]
        assert [batch.pos[i] for i in range(3)] == [5, 6, 7]
        assert [batch.n_seq_id[i] for i in range(3)] == [1, 1, 1]
        assert [batch.seq_id[i][0] for i in range(3)] == [0, 1, 2]
        assert [batch.logits[i] for i in range(3)] == [1, 1, 1]

    def test_returns_scores_rows_indexed_by_position(self) -> None:
        """Returned logits are the ``scores`` rows at each active position."""
        llm = _make_batched_llm_mock(n_vocab=4)
        llm.scores = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [0, 0, 0, 0]],
            dtype=np.float32,
        )
        backend = ModelCBackend(llm)
        result = backend.decode_batch([10, 20, 30], [0, 1, 2], [0, 1, 2])
        assert result.shape == (3, 4)
        assert result.dtype == np.float32
        assert np.array_equal(result, llm.scores[[0, 1, 2]])
        # Returns a copy, not a view into the scores array.
        assert result.base is None or result.base is not llm.scores

    def test_returns_active_by_vocab_logits(self) -> None:
        """Returned array shape is (n_active, vocab) for arbitrary positions."""
        llm = _make_batched_llm_mock(n_vocab=4)
        backend = ModelCBackend(llm)
        result = backend.decode_batch([10, 20], [0, 1], [3, 3])
        assert result.shape == (2, 4)
        assert result.dtype == np.float32
        assert np.array_equal(result, llm.scores[[3, 3]])

    def test_delegates_to_eval_in_mock_fallback(self) -> None:
        """Mock fallback path forwards token ids to llm.eval (like decode())."""
        llm = _make_batched_llm_mock(n_vocab=4)
        backend = ModelCBackend(llm)
        backend.decode_batch([10, 20], [0, 1], [3, 3])
        llm.eval.assert_called_once_with([10, 20])

    def test_increments_decode_call_stats(self) -> None:
        """decode_batch counts as one decode call in backend stats."""
        llm = _make_batched_llm_mock(n_vocab=4)
        backend = ModelCBackend(llm)
        before = backend.stats().decode_calls
        backend.decode_batch([10, 20], [0, 1], [0, 0])
        assert backend.stats().decode_calls == before + 1

    def test_does_not_mutate_model_n_tokens(self) -> None:
        """Batched decode leaves single-sequence n_tokens bookkeeping to caller."""
        llm = _make_batched_llm_mock(n_vocab=4)
        backend = ModelCBackend(llm)
        original_n_tokens = backend.n_tokens
        backend.decode_batch([10, 20], [0, 1], [0, 0])
        assert backend.n_tokens == original_n_tokens


class TestKVCacheSeqHelpers:
    """Tests for per-sequence KV-cache management helpers."""

    def test_kv_cache_seq_rm_delegates(self) -> None:
        """kv_cache_seq_rm forwards to the wrapped context."""
        llm = _make_batched_llm_mock()
        backend = ModelCBackend(llm)
        backend.kv_cache_seq_rm(0, 5, 10)
        llm._ctx.kv_cache_seq_rm.assert_called_once_with(0, 5, 10)  # noqa: SLF001

    def test_kv_cache_seq_cp_delegates(self) -> None:
        """kv_cache_seq_cp forwards to the wrapped context."""
        llm = _make_batched_llm_mock()
        backend = ModelCBackend(llm)
        backend.kv_cache_seq_cp(0, 1, 0, -1)
        llm._ctx.kv_cache_seq_cp.assert_called_once_with(0, 1, 0, -1)  # noqa: SLF001

    def test_kv_cache_seq_keep_delegates(self) -> None:
        """kv_cache_seq_keep forwards to the wrapped context."""
        llm = _make_batched_llm_mock()
        backend = ModelCBackend(llm)
        backend.kv_cache_seq_keep(2)
        llm._ctx.kv_cache_seq_keep.assert_called_once_with(2)  # noqa: SLF001
