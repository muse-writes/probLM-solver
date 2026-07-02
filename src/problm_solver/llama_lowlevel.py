"""Interface module for ``llama-cpp-python``'s C API."""

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
from llama_cpp import Llama, LlamaState


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
        # TODO(Clio): Replace fallback behaviour with call to llama_decode().
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
        # TODO(Clio): Replace with pointer to logits.
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
