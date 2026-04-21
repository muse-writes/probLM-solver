"""Tokenizer implementations for statistical analysis of LLM responses."""

import re
from abc import ABC, abstractmethod
from typing import Protocol

# A token is a plain string — the decoded surface form of a model vocabulary piece.
Token = str

# An ordered sequence of tokens produced by splitting a single response string.
TokenSequence = list[Token]


class Tokenizer(ABC):
    """Abstract base class for tokenizers.

    All tokenizers convert a text string to an ordered list of Token strings.
    The splitting strategy is implementation-defined, but must be deterministic
    and consistent: the same input always produces the same output.

    Subclass and implement :meth:`tokenize` to define a new strategy.
    """

    @abstractmethod
    def tokenize(self, text: str) -> TokenSequence:
        """Split text into a sequence of tokens.

        :param text: The input string to tokenize.
        :returns: An ordered list of token strings covering the input.
        """


class WordTokenizer(Tokenizer):
    r"""Regex-based word tokenizer requiring no external dependencies.

    Splits text into word tokens and single-character punctuation tokens.
    Whitespace acts as a separator and is not itself returned as a token.
    Contiguous alphanumeric runs (including underscores, per ``\\w``) are
    kept as single tokens; every non-word, non-space character becomes its
    own token.

    :param lowercase: If ``True``, all tokens are lowercased before returning.
        Default is ``False``.

    >>> t = WordTokenizer()
    >>> t.tokenize('Hello, world!')
    ['Hello', ',', 'world', '!']

    >>> t = WordTokenizer(lowercase=True)
    >>> t.tokenize("It's alive!")
    ["it's", 'alive', '!']

    >>> WordTokenizer().tokenize('')
    []
    """

    _PATTERN: re.Pattern[str] = re.compile(r"\w+(?:'\w+)*|[^\w\s]")

    def __init__(self, *, lowercase: bool = False) -> None:
        """Initialise the tokenizer."""
        self.lowercase = lowercase

    def tokenize(self, text: str) -> TokenSequence:
        """Split text into word and punctuation tokens.

        :param text: The input string to tokenize.
        :returns: Words (including contractions) and punctuation characters as
            separate tokens. Whitespace is consumed as a separator and is
            not included in the output.
        """
        tokens: TokenSequence = self._PATTERN.findall(text)
        if self.lowercase:
            return [t.lower() for t in tokens]
        return tokens


class _LlamaInterface(Protocol):
    """Structural protocol for the llama_cpp.Llama methods used here.

    Allows :class:`LlamaTokenizer` to accept any object with these two
    methods, avoiding a hard import dependency on ``llama_cpp`` in this
    module.
    """

    def tokenize(
        self,
        text: bytes,
        add_bos: bool = ...,
        special: bool = ...,
    ) -> list[int]:
        """Encode bytes to a list of integer token IDs."""
        ...

    def detokenize(
        self,
        tokens: list[int],
        prev_tokens: list[int] | None = ...,
        special: bool = ...,
    ) -> bytes:
        """Decode a list of integer token IDs back to bytes."""
        ...


class LlamaTokenizer(Tokenizer):
    """Tokenizer backed by a loaded llama.cpp model.

    Uses the model's own byte-pair encoding (BPE) vocabulary, so the tokens
    produced are exactly the sub-word pieces the model operates on internally.
    This is the most faithful strategy for analysing outputs from that model.

    Each token ID is decoded individually so that the token boundaries are
    preserved. Leading spaces are kept as part of each piece (e.g. ``' world'``),
    reflecting the space-prefixed convention used in SentencePiece vocabularies.

    :param llama: A loaded Llama model instance. Concretely, any object that
        provides ``tokenize(bytes, add_bos, special) -> list[int]`` and
        ``detokenize(list[int]) -> bytes`` is accepted.

    >>> from unittest.mock import MagicMock
    >>> mock = MagicMock()
    >>> mock.tokenize.return_value = [1, 2, 3]
    >>> mock.detokenize.side_effect = lambda ids, **_: {1: b'Hello', 2: b',', 3: b' world'}[ids[0]]
    >>> t = LlamaTokenizer(mock)
    >>> t.tokenize('Hello, world')
    ['Hello', ',', ' world']
    """

    def __init__(self, llama: _LlamaInterface) -> None:
        """Initialise with a ``llama_cpp.Llama`` instance."""
        self._llama = llama

    def tokenize(self, text: str) -> TokenSequence:
        """Tokenize text using the model's BPE vocabulary.

        Calls the underlying ``Llama.tokenize`` to obtain a sequence of integer
        token IDs, then converts each ID individually to a string via
        ``Llama.detokenize``.

        :param text: The input string to tokenize.
        :returns: Sub-word pieces as decoded UTF-8 strings. Any byte sequences
            that are not valid UTF-8 are replaced with the Unicode replacement
            character (``U+FFFD``).
        """
        token_ids = self._llama.tokenize(text.encode(), add_bos=False)
        return [
            self._llama.detokenize([tid]).decode('utf-8', errors='replace')
            for tid in token_ids
        ]
