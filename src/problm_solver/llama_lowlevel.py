"""Interface module for ``llama-cpp-python``'s C API."""


import llama_cpp
import numpy as np
import numpy.typing as npt
from llama_cpp import Llama


class ModelBackend:
    """Low-level ``llama-cpp-python`` C API adapter.

    Handles running a model on a lower level, provides logits. Doesn't mess
    anything up hopefully.
    """

    def __init__(self, llm: Llama) -> None:
        """Store LLM instance."""
        self._llm = llm

    @property
    def n_tokens(self) -> int:
        """Get number of LLM tokens."""
        return int(self._llm.n_tokens)

    def reset(self) -> None:
        """Reset LLM state & cache."""
        self._llm.reset()

    def decode(self, token_ids: list[int]) -> None:
        """Decode with a re-usable batch."""
#TODO(Clio): Replace fallback behaviour with call to llama_decode().
        self._llm.eval(token_ids)

    def last_logits(self) -> npt.NDArray[np.float32]:
        """Get most recent array of model logits."""
#TODO(Clio): Replace with pointer to logits.
        return self._llm.scores[self._llm.n_tokens - 1]

    def token_eos(self) -> int:
        """Get ID of vocabulary EOS."""
        return int(self._llm.token_eos())

    def token_bos(self) -> int:
        """Get ID of vocabulary BOS."""
        return int(self._llm.token_bos())

    def tokenize(self, text: bytes, add_bos: bool, special: bool) -> list[int]:
        """Tokenize some text."""
        return self._llm.tokenize(text, add_bos=add_bos, special=special)

    def detokenize(self, token_ids: list[int], special: bool = False) -> bytes:
        """Detokenize some token IDs."""
        return self._llm.detokenize(token_ids, special=special)
