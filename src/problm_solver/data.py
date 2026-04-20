"""Data container for LLM output."""

import json
from typing import Any

import numpy as np
import numpy.typing as npt

from problm_solver.analysis import TokenSequence


class LLMOutputData:
    """Stores LLM output data and handles serialization."""

    def __init__(self, prompt: str, data: npt.NDArray[Any]) -> None:
        """Initialize.

        Args:
            prompt: The prompt used to generate the data.
            data: Array of string responses from the LLM.
        """
        self.prompt = prompt
        self.data = data
        self.written = False

    def write(self, fname: str) -> None:
        """Save data to a file in JSONL format.

        Default filename is `[model]_[timestamp].jsonl`.
        """
        with open(fname, 'w', encoding='utf-8') as writer:
            for i, response in enumerate(self.data, start=1):
                record = {
                    'id': i,
                    'prompt': self.prompt,
                    'response': response
                }
                writer.write(json.dumps(record) + '\n')
        self.written = True


    def read(self, fname: str) -> None:
        """Read data to this object from a JSONL file."""
        responses = []
        with open(fname, encoding='utf-8') as reader:
            for line in reader:
                record = json.loads(line)
                self.prompt = record['prompt']
                responses.append(record['response'])
        self.data = np.array(responses, dtype=str)
        self.written = True


class LLMTokenData:
    """Store a single tokenized LLM response paired with per-token probabilities."""

    def __init__(self, prompt: str, tokens: TokenSequence, probs: list[float]) -> None:
        """Initialize.

        Parameters
        ----------
        prompt : str
            The prompt used to generate the response.
        tokens : TokenSequence
            The response as an ordered list of token strings.
        probs : list[float]
            The probability of each token at its position in the response.
            Must be the same length as ``tokens``.
        """
        self.prompt = prompt
        self.tokens = tokens
        self.probs = probs
