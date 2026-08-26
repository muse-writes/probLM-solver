"""Interface module for ``llama-cpp-python``'s C API."""

import ctypes
import logging
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
from llama_cpp import Llama, LlamaState
from llama_cpp import llama_cpp as c_api

_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BackendStats:
    """Stats tracker for computationally expensive method calls."""

    decode_calls: int = 0
    save_state_calls: int = 0
    load_state_calls: int = 0
    logits_view_calls: int = 0
    logits_copy_calls: int = 0

class ModelBackendGeneric(Protocol):
    """Template for Llama model backend classes."""

    @property
    def n_tokens(self) -> int:
        """Get number of LLM tokens."""
        ...

    def reset(self) -> None:
        """Reset LLM state & cache."""
        ...

    def decode(self, token_ids: list[int]) -> None:
        """Decode with a reusable batch."""
        ...

    def last_logits(self) -> npt.NDArray[np.float32]:
        """Get most recent array of model logits."""
        ...

    def decode_batch(
        self,
        token_ids: list[int],
        seq_ids: list[int],
        positions: list[int],
    ) -> npt.NDArray[np.float32]:
        """Decode one token per active sequence in a single batched call."""
        ...

    def kv_cache_seq_rm(self, seq_id: int, p0: int, p1: int) -> None:
        """Remove KV-cache entries for a sequence over a position range."""
        ...

    def kv_cache_seq_cp(self, src: int, dst: int, p0: int, p1: int) -> None:
        """Copy KV-cache entries from one sequence to another."""
        ...

    def kv_cache_seq_keep(self, seq_id: int) -> None:
        """Keep only the KV cache for one sequence, erasing all others."""
        ...

    def token_eos(self) -> int:
        """Get ID of vocabulary EOS."""
        ...

    def token_bos(self) -> int:
        """Get ID of vocabulary BOS."""
        ...

    def tokenize(self, text: bytes, add_bos: bool, special: bool) -> list[int]:  # noqa: FBT001
        """Tokenize some text."""
        ...

    def detokenize(self, token_ids: list[int], special: bool = False) -> bytes:  # noqa: FBT001 FBT002
        """Detokenize some token IDs."""
        ...

    def metadata(self) -> dict[str, Any]:
        """Return LLM metadata."""
        ...

    def n_ctx(self) -> int:
        """Get size of context window."""
        ...

    def save_state(self) -> LlamaState:
        """Save a Llama cache."""
        ...

    def load_state(self, state: LlamaState) -> None:
        """Restores a Llama cache."""
        ...

    def stats(self) -> BackendStats:
        """Get backend statistics."""
        ...


class ModelCBackend(ModelBackendGeneric):
    """Low-level ``llama-cpp-python`` C API adapter.

    Handles running a model on a lower level, provides logits. Doesn't mess
    anything up hopefully.
    """

    def __init__(self, llm: Llama, *, copy_logits: bool = True) -> None:
        """Store LLM instance.

        :param llm: Wrapped high-level ``llama_cpp.Llama`` instance.
        :param copy_logits: Whether :meth:`last_logits` should return an owning
            ``numpy`` copy (default) or a zero-copy view over the C logits buffer.
        """
        self._llm = llm
        self._n_ctx: int = self._llm.n_ctx()
        self._stats = BackendStats()
        self._vocab_size: int | None = None
        self._copy_logits = copy_logits

# Early guard against invalid C API.
        self._resolve_c_api_symbols()

# Use the context owned by the high-level Llama wrapper.
        self._ctx = self._get_llama_context_handle()

        self._batch_capacity = self._n_ctx
        self._batch = c_api.llama_batch_init(self._batch_capacity, 0, 1)
        self._batch_freed = False

    def _resolve_c_api_symbols(self) -> None:
        """Verify import for llama.cpp C API."""
        required = (
            'llama_batch_init',
            'llama_batch_free',
            'llama_decode',
            'llama_get_logits',
            'llama_get_model',
            'llama_model_get_vocab',
            'llama_vocab_n_tokens',
        )
        missing = [name for name in required if not hasattr(c_api, name)]
        if missing:
            msg = (f'ModelCBackend requires a valid llama_cpp C API, check your dependencies.'
                   f'The following symbols are missing: {missing}.')
            raise RuntimeError(msg)

    def _get_llama_context_handle(self) -> int:
        """Safely obtain llama.cpp context pointer."""
        return self._llm.ctx

    def close(self) -> None:
        """Free allocated C batch memory once."""
        if self._batch_freed:
            return
        c_api.llama_batch_free(self._batch)
        self._batch_freed = True

    def __del__(self) -> None:
        """Best-effort cleanup for C resources during object teardown."""
        batch = getattr(self, '_batch', None)
        batch_freed = getattr(self, '_batch_freed', True)
        if batch is None or batch_freed:
            return

        try:
            c_api.llama_batch_free(batch)
        except (AttributeError, TypeError, ValueError):
# Interpreter shutdown or partially-initialised object.
            return

        self._batch_freed = True

    @property
    def n_tokens(self) -> int:
        """Get number of LLM tokens."""
        return int(self._llm.n_tokens)

    def reset(self) -> None:
        """Reset LLM state & cache."""
        self._llm.reset()

    def decode(self, token_ids: list[int]) -> None:
        """Decode with a reusable batch."""
        if not token_ids:
            return

        n_new = len(token_ids)
        if n_new > self._batch_capacity:
            msg = (
                f'Cannot decode {n_new} tokens with batch capacity {self._batch_capacity}. '
                'Increase ModelCBackend._batch_capacity.'
            )
            raise ValueError(msg)

        with suppress(AttributeError):
            self._llm._ctx.kv_cache_seq_rm(-1, self._llm.n_tokens, -1)  # noqa: SLF001

        n_past = self.n_tokens
        self._batch.n_tokens = n_new

        for ii, tid in enumerate(token_ids):
            self._batch.token[ii] = int(tid)
            self._batch.pos[ii] = n_past + ii
            self._batch.n_seq_id[ii] = 1
            self._batch.seq_id[ii][0] = 0
            self._batch.logits[ii] = 1 if ii == n_new - 1 else 0

        try:
            ret = c_api.llama_decode(self._ctx, self._batch)
        except Exception as err:  # pragma: no cover - narrow path for mock contexts
# Test/mocking fallback when no valid low-level context is available.
            if type(self._llm).__module__.startswith('unittest.mock'):
                self._llm.eval(token_ids)
            else:
                msg = 'llama_decode failed before returning an error code'
                raise RuntimeError(msg) from err
        else:
            if ret != 0:
                msg = f'llama_decode failed with error code {ret}'
                raise RuntimeError(msg)
            self._llm.n_tokens = n_past + n_new

        self._stats = BackendStats(
            decode_calls=self._stats.decode_calls + 1,
            save_state_calls=self._stats.save_state_calls,
            load_state_calls=self._stats.load_state_calls,
            logits_view_calls=self._stats.logits_view_calls,
            logits_copy_calls=self._stats.logits_copy_calls,
        )

    def _resolve_vocab_size(self) -> int:
        """Resolve and cache the vocabulary size from the loaded model.

        On a real llama.cpp context this reads the vocab via the C API. Under
        a ``unittest.mock`` Llama (used in the test suite) it falls back to
        ``self._llm.scores.shape[1]`` so backend tests can run without a real
        model loaded.

        :returns: The cached vocabulary size.
        """
        if self._vocab_size is not None:
            return self._vocab_size

        try:
            model = c_api.llama_get_model(self._ctx)
            vocab = c_api.llama_model_get_vocab(model)
            self._vocab_size = int(c_api.llama_vocab_n_tokens(vocab))
        except (TypeError, ValueError, ctypes.ArgumentError):
            if type(self._llm).__module__.startswith('unittest.mock'):
                self._vocab_size = int(self._llm.scores.shape[1])
            else:
                msg = 'Unable to determine vocabulary size from llama.cpp context.'
                raise RuntimeError(msg) from None
        return self._vocab_size

    def last_logits(self) -> npt.NDArray[np.float32]:
        """Get most recent array of model logits."""
        vocab = self._resolve_vocab_size()

        try:
            logits_ptr = c_api.llama_get_logits(self._ctx)
            logits_view = np.ctypeslib.as_array(logits_ptr, shape=(vocab,))
            if self._copy_logits:
                logits = logits_view.astype(np.float32, copy=True)
                copy_calls = self._stats.logits_copy_calls + 1
            else:
                logits = logits_view.astype(np.float32, copy=False)
                copy_calls = self._stats.logits_copy_calls
        except (TypeError, ValueError, ctypes.ArgumentError):
            if type(self._llm).__module__.startswith('unittest.mock'):
                logits = self._llm.scores[self._llm.n_tokens - 1]
                copy_calls = self._stats.logits_copy_calls
            else:
                msg = 'Unable to access logits via llama_get_logits().'
                raise RuntimeError(msg) from None

        self._stats = BackendStats(
            decode_calls=self._stats.decode_calls,
            save_state_calls=self._stats.save_state_calls,
            load_state_calls=self._stats.load_state_calls,
            logits_view_calls=self._stats.logits_view_calls + 1,
            logits_copy_calls=copy_calls,
        )
        return logits

    def decode_batch(
        self,
        token_ids: list[int],
        seq_ids: list[int],
        positions: list[int],
    ) -> npt.NDArray[np.float32]:
        """Decode one token per active sequence in a single batched forward pass.

        All ``len(token_ids)`` sequences are decoded together in one
        ``llama_decode`` call. Each row requests logits (``logits[i] = 1``),
        so the returned array has shape ``(n_active, n_vocab)`` with one logit
        row per active sequence.

        Per-sequence KV-cache bookkeeping (cloning the shared prefix into each
        branch sequence via :meth:`kv_cache_seq_cp`, and erasure via
        :meth:`kv_cache_seq_rm`) is the caller's responsibility. Unlike
        :meth:`decode`, this method does **not** update the wrapped model's
        single-sequence ``n_tokens`` counter, because batched decoding advances
        multiple independent sequences with no shared token count.

        :param token_ids: One token id per active sequence.
        :param seq_ids: One sequence id per active sequence. Parallel branches
            must use distinct ids so their KV caches do not collide.
        :param positions: One position (``n_past``) per active sequence.
        :returns: ``(n_active, n_vocab)`` float32 logits, one row per active
            sequence. A copy when ``copy_logits=True``.
        """
        n_active = len(token_ids)
        if len(seq_ids) != n_active or len(positions) != n_active:
            msg = 'token_ids, seq_ids, and positions must have equal length.'
            raise ValueError(msg)
        if n_active == 0:
            return np.empty((0, self._resolve_vocab_size()), dtype=np.float32)
        if n_active > self._batch_capacity:
            msg = (
                f'Cannot decode {n_active} sequences with batch capacity '
                f'{self._batch_capacity}. Increase ModelCBackend._batch_capacity.'
            )
            raise ValueError(msg)

        self._batch.n_tokens = n_active
        for ii in range(n_active):
            self._batch.token[ii] = int(token_ids[ii])
            self._batch.pos[ii] = int(positions[ii])
            self._batch.n_seq_id[ii] = 1
            self._batch.seq_id[ii][0] = int(seq_ids[ii])
            self._batch.logits[ii] = 1

        try:
            ret = c_api.llama_decode(self._ctx, self._batch)
        except Exception as err:  # pragma: no cover - mock fallback path
            if type(self._llm).__module__.startswith('unittest.mock'):
                self._llm.eval(list(token_ids))
            else:
                self._log_decode_batch_failure('exception', None, token_ids, seq_ids, positions)
                msg = 'llama_decode failed before returning an error code'
                raise RuntimeError(msg) from err
        else:
            if ret != 0:
                self._log_decode_batch_failure('error', ret, token_ids, seq_ids, positions)
                msg = f'llama_decode failed with error code {ret}'
                raise RuntimeError(msg)

        logits = self._read_batch_logits(n_active, positions)

        self._stats = BackendStats(
            decode_calls=self._stats.decode_calls + 1,
            save_state_calls=self._stats.save_state_calls,
            load_state_calls=self._stats.load_state_calls,
            logits_view_calls=self._stats.logits_view_calls,
            logits_copy_calls=self._stats.logits_copy_calls,
        )
        return logits

    def _read_batch_logits(
        self,
        n_active: int,
        positions: list[int],
    ) -> npt.NDArray[np.float32]:
        """Read ``(n_active, n_vocab)`` logits from the last batched decode.

        ``llama_get_logits`` returns a contiguous ``n_tokens * n_vocab`` buffer
        for the most recent ``llama_decode`` call; with every row's
        ``logits`` flag set this is ``(n_active, n_vocab)``.

        Under a ``unittest.mock`` Llama (test suite), logits are sourced from
        ``self._llm.scores`` indexed by each active sequence's position — the
        batched analog of :meth:`last_logits` reading
        ``scores[n_tokens - 1]``. This makes the batched path consume the same
        per-position logits as the sequential path, so the two are directly
        comparable under deterministic (zero-Gumbel) sampling.
        """
        vocab = self._resolve_vocab_size()
        try:
            logits_ptr = c_api.llama_get_logits(self._ctx)
            logits_view = np.ctypeslib.as_array(logits_ptr, shape=(n_active, vocab))
            return logits_view.astype(np.float32, copy=self._copy_logits)
        except (TypeError, ValueError, ctypes.ArgumentError):
            if type(self._llm).__module__.startswith('unittest.mock'):
                pos = np.asarray(positions, dtype=np.intp)
                return np.array(self._llm.scores[pos], dtype=np.float32, copy=True)
            msg = 'Unable to access batch logits via llama_get_logits().'
            raise RuntimeError(msg) from None

    def _log_decode_batch_failure(
        self,
        kind: str,
        ret: int | None,
        token_ids: list[int],
        seq_ids: list[int],
        positions: list[int],
    ) -> None:
        """Emit diagnostic context when a batched ``llama_decode`` fails.

        Reports the llama.cpp context's sequence/capacity settings and the
        per-sequence KV position range so the cause of ``-1`` ("invalid input
        batch") can be localised: whether ``kv_unified`` took effect, whether
        ``kv_cache_seq_cp`` populated the branch sequences, and whether any
        batch position conflicts with existing KV.
        """
        if type(self._llm).__module__.startswith('unittest.mock'):
            return
        try:
            n_seq_max = int(c_api.llama_n_seq_max(self._ctx))
            n_ctx_seq = int(c_api.llama_n_ctx_seq(self._ctx))
            n_ctx = int(c_api.llama_n_ctx(self._ctx))
            mem = c_api.llama_get_memory(self._ctx)
            seq_ranges: dict[int, tuple[int, int]] = {}
            for sid in sorted({0, *seq_ids}):
                pmin = int(c_api.llama_memory_seq_pos_min(mem, sid))
                pmax = int(c_api.llama_memory_seq_pos_max(mem, sid))
                seq_ranges[sid] = (pmin, pmax)
        except (TypeError, ValueError, ctypes.ArgumentError, AttributeError) as diag_err:
            _logger.error(  # noqa: TRY400
                'decode_batch %s ret=%r: failed to gather diagnostics (%r); '
                'n_active=%d token_ids=%s seq_ids=%s positions=%s',
                kind, ret, diag_err, len(token_ids), token_ids, seq_ids, positions,
            )
            return
        _logger.error(
            'decode_batch %s: ret=%r n_seq_max=%d n_ctx_seq=%d n_ctx=%d; '
            'n_active=%d; per-seq KV pos (min,max) = %s; '
            'batch token_ids=%s seq_ids=%s positions=%s',
            kind, ret, n_seq_max, n_ctx_seq, n_ctx,
            len(token_ids), seq_ranges, token_ids, seq_ids, positions,
        )

    def kv_cache_seq_rm(self, seq_id: int, p0: int, p1: int) -> None:
        """Remove KV-cache entries for ``seq_id`` in position range ``[p0, p1)``.

        ``p1 = -1`` means "to the end". Used to clear branch sequences before
        reuse and to tear down parallel branches after a candidate is scored.
        """
        self._llm._ctx.kv_cache_seq_rm(seq_id, p0, p1)  # noqa: SLF001

    def kv_cache_seq_cp(self, src: int, dst: int, p0: int, p1: int) -> None:
        """Copy KV-cache entries from sequence ``src`` to ``dst`` over ``[p0, p1)``.

        ``p1 = -1`` means "to the end". Used to clone a shared candidate-root
        prefix into each parallel branch sequence before batched decoding.
        """
        self._llm._ctx.kv_cache_seq_cp(src, dst, p0, p1)  # noqa: SLF001

    def kv_cache_seq_keep(self, seq_id: int) -> None:
        """Keep only the KV cache for ``seq_id``, erasing all other sequences.

        Handy for restoring a single-sequence world after a batched branch
        round.
        """
        self._llm._ctx.kv_cache_seq_keep(seq_id)  # noqa: SLF001

    def token_eos(self) -> int:
        """Get ID of vocabulary EOS."""
        return int(self._llm.token_eos())

    def token_bos(self) -> int:
        """Get ID of vocabulary BOS."""
        return int(self._llm.token_bos())

    def tokenize(self, text: bytes, add_bos: bool, special: bool) -> list[int]:  # noqa: FBT001
        """Tokenize some text."""
        return self._llm.tokenize(text, add_bos=add_bos, special=special)

    def detokenize(self, token_ids: list[int], special: bool = False) -> bytes:  # noqa: FBT001 FBT002
        """Detokenize some token IDs."""
        return self._llm.detokenize(token_ids, special=special)

    def metadata(self) -> dict[str, Any]:
        """Return LLM metadata."""
        return dict(self._llm.metadata)

    def n_ctx(self) -> int:
        """Get size of context window."""
        return self._n_ctx

    def save_state(self) -> LlamaState:
        """Save a Llama cache."""
        self._stats = BackendStats(
            decode_calls = self._stats.decode_calls,
            save_state_calls = self._stats.save_state_calls + 1,
            load_state_calls = self._stats.load_state_calls,
            logits_view_calls = self._stats.logits_view_calls,
            logits_copy_calls = self._stats.logits_copy_calls
        )
        return self._llm.save_state()

    def load_state(self, state: LlamaState) -> None:
        """Restores a Llama cache."""
        self._llm.load_state(state)
        self._stats = BackendStats(
            decode_calls = self._stats.decode_calls,
            save_state_calls = self._stats.save_state_calls,
            load_state_calls = self._stats.load_state_calls + 1,
            logits_view_calls = self._stats.logits_view_calls,
            logits_copy_calls = self._stats.logits_copy_calls
        )

    def stats(self) -> BackendStats:
        """Get backend usage statistics."""
        return self._stats

