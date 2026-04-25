"""Data container for LLM output."""

import json
from typing import Any

import numpy as np
import numpy.typing as npt


class LLMOutputData:
    """Stores LLM output data and handles serialization."""

    def __init__(self, prompt: str, data: npt.NDArray[Any]) -> None:
        """Initialize.

        :param prompt: The prompt used to generate the data.
        :param data: Array of string responses from the LLM.
        """
        self.prompt = prompt
        self.data = data
        self.written = False

    def write(self, fname: str) -> None:
        """Save data to a file in JSONL format.

        :param fname: File name to write to, absolute path.
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
        """Read data to this object from a JSONL file.

        :param fname: File name to read from, absolute path.
        """
        responses = []
        with open(fname, encoding='utf-8') as reader:
            for line in reader:
                record = json.loads(line)
                self.prompt = record['prompt']
                responses.append(record['response'])
        self.data = np.array(responses, dtype=str)
        self.written = True


class TokenProbError(ValueError):
    """Raise an error when number of tokens in data isn't equal to probs."""

    def __init__(self, data_class: type) -> None:
        """Initialize error message.

        Error is intended for dataclasses and other containers that hold tokens and their
        probabilities.

        :param data_class: Class type to be passed to error message.
        """
        super().__init__(
            f'Number of probabilities in `{data_class}` does not match the '
            f'number of tokens.'
        )


class LLMTokenData:
    """Store a single tokenized LLM response paired with per-token probabilities."""

    def __init__(self, prompt: str, tokens: list[str], probs: list[float]) -> None:
        """Initialize.

        :param prompt: The prompt used to generate the response.
        :param tokens: The response as an ordered list of token strings.
        :param probs: The probability of each token at its position in the response.
            Must be the same length as ``tokens``.
        """
        if len(tokens) != len(probs):
            raise TokenProbError(type(self))
        self.prompt = prompt
        self.tokens = tokens
        self.probs = probs
        self.written = False


    def write(self, fname: str) -> None:
        """Save data to a file in JSON format.

        :param fname: File name to write to, absolute path.
        """
        with open(fname, 'w', encoding='utf-8') as writer:
            record = {
                'prompt': self.prompt,
                'tokens': self.tokens,
                'probs': self.probs,
            }
            writer.write(json.dumps(record))
        self.written = True


    def read(self, fname: str) -> None:
        """Read data from JSON file.

        :param fname: File name to read from, absolute path.
        """
        with open(fname, encoding='utf-8') as reader:
            data = json.load(reader)
            self.prompt = data['prompt']
            self.tokens = data['tokens']
            self.probs = data['probs']
        self.written = True

class LLMNextTokenData:
    """Store M most likely next tokens given context and output vector."""

    def __init__(
        self,
        prompt: str,
        output_vec: list[int],
        top_m_tokens: dict[str, float],
    ) -> None:
        """Initialize context and next token data.

        :param prompt: The original user prompt.
        :param output_vec: The current token ID sequence (prompt + generated
            tokens so far).
        :param top_m_tokens: Mapping of token string to log-probability for
            the top M candidate next tokens at this position.
        """
        self.prompt = prompt
        self.output_vec = output_vec
        self.top_m_tokens = top_m_tokens
        self.written = False


    def write(self, fname: str) -> None:
        """Save data to a file in JSON format.

        :param fname: File name to write to, absolute path.
        """
        with open(fname, 'w', encoding='utf-8') as writer:
            record = {
                'prompt': self.prompt,
                'output_vec': self.output_vec,
                'top_m_tokens': self.top_m_tokens,
            }
            writer.write(json.dumps(record))
        self.written = True


    def read(self, fname: str) -> None:
        """Read from JSON file.

        :param fname: File name to read from, absolute path.
        """
        with open(fname, encoding='utf-8') as reader:
            data = json.load(reader)
            self.prompt = data['prompt']
            self.output_vec = data['output_vec']
            self.top_m_tokens = data['top_m_tokens']
        self.written = True
