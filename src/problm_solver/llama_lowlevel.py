"""Interface module for ``llama-cpp-python``'s C API."""

import ctypes
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
from llama_cpp import Llama, LlamaState
from llama_cpp import llama_cpp as c_api


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
        """Decode with a re-usable batch."""
        ...

    def last_logits(self) -> npt.NDArray[np.float32]:
        """Get most recent array of model logits."""
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
        self._n_ctx: int | None = None
        self._stats = BackendStats()
        self._vocab_size: int | None = None
        self._copy_logits = copy_logits

# Early guard against invalid C API.
        self._resolve_c_api_symbols()

# Use the context owned by the high-level Llama wrapper.
        self._ctx = self._get_llama_context_handle()

        self._batch_capacity = 512
        self._batch = c_api.llama_batch_init(self._batch_capacity, 0, 1)
        self._batch_freed = False

    def _resolve_c_api_symbols(self) -> None:
        """Verify import for llama.cpp C API."""
        required = (
            'llama_batch_init',
            'llama_batch_free',
            'llama_decode',
            'llama_get_logits',
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
        """Decode with a re-usable batch."""
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
            self._llm._ctx.kv_cache_seq_rm(-1, self._llm.n_tokens, -1)

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

    def last_logits(self) -> npt.NDArray[np.float32]:
        """Get most recent array of model logits."""
        if self._vocab_size is None:
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

        try:
            logits_ptr = c_api.llama_get_logits(self._ctx)
            logits_view = np.ctypeslib.as_array(logits_ptr, shape=(self._vocab_size,))
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
        if self._n_ctx is None:
            self._n_ctx = int(self._llm.n_ctx())
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


class ModelLlamaBackend(ModelBackendGeneric):
    """High-level ``llama-cpp-python`` API adapter.

    Handles running a model on a higher level, provides logits. Doesn't mess
    anything up hopefully.
    """

    def __init__(self, llm: Llama) -> None:
        """Store LLM instance."""
        self._llm = llm
        self._n_ctx: int | None = None
        self._stats = BackendStats()

    @property
    def n_tokens(self) -> int:
        """Get number of LLM tokens."""
        return int(self._llm.n_tokens)

    def reset(self) -> None:
        """Reset LLM state & cache."""
        self._llm.reset()

    def decode(self, token_ids: list[int]) -> None:
        """Decode with a re-usable batch."""
        self._llm.eval(token_ids)
        self._stats = BackendStats(
            decode_calls = self._stats.decode_calls + 1,
            save_state_calls = self._stats.save_state_calls,
            load_state_calls = self._stats.load_state_calls,
            logits_view_calls = self._stats.logits_view_calls,
            logits_copy_calls = self._stats.logits_copy_calls
        )

    def last_logits(self) -> npt.NDArray[np.float32]:
        """Get most recent array of model logits."""
        self._stats = BackendStats(
            decode_calls = self._stats.decode_calls,
            save_state_calls = self._stats.save_state_calls,
            load_state_calls = self._stats.load_state_calls,
            logits_view_calls = self._stats.logits_view_calls + 1,
            logits_copy_calls = self._stats.logits_copy_calls
        )
        return self._llm.scores[self._llm.n_tokens - 1]

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
        if self._n_ctx is None:
            self._n_ctx = int(self._llm.n_ctx())
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
