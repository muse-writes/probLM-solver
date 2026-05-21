"""Data container for LLM output."""

import json
from dataclasses import asdict, dataclass, field
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
        top_k_tokens: dict[str, float],
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
        self.top_k_tokens = top_k_tokens
        self.written = False


    def write(self, fname: str) -> None:
        """Save data to a file in JSON format.

        :param fname: File name to write to, absolute path.
        """
        with open(fname, 'w', encoding='utf-8') as writer:
            record = {
                'prompt': self.prompt,
                'output_vec': self.output_vec,
                'top_k_tokens': self.top_k_tokens,
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
            self.top_k_tokens = data['top_k_tokens']
        self.written = True


# Types for easier parsing of LLMOutputDataFull.
Tokens = list[str]
TopKProbs = dict[str, float]
ResponseProbs = tuple[Tokens, list[float]]
TopKResponse = tuple[Tokens, list[TopKProbs]]


@dataclass
class Hyperparams:
    """An assistant dataclass to `LLMOutputDataFull` with hyperparameter data included."""

    alpha: float
    top_k: int | None
    top_p: float | None
    max_tokens: int


@dataclass
class LLMOutputDataFull:
    """A fuller dataclass than `LLMOutputData`.

    Includes various parameters as strings, as well as top-k finalised token probabilities.

    :param context: Tokenised context provided to the model.
    :param hyperparams: Nested dataclass for hyperparameter storage.
    :param response_probabilities: response tokens and their context-dependent selection
        probabilities after adjustment.
    :param response_topk: For each token sampled, supplies the top K tokens and their probabilities.
    :param sampling_method: User label for sampling method used.
    :param branch_sampler: User label for branch sampler used, if sampling from the power
        distribution.
    """

    context: Tokens
    hyperparams: Hyperparams
    response_probabilities: ResponseProbs
    response_topk: TopKResponse
    sampling_method: str
    branch_sampler: str | None

# Unsaved data state tracking variable.
    _written: bool = field(default=False, repr=False)

    def write(self, fname: str) -> None:
        """Write data to JSON file."""
        with open(fname, 'w', encoding='utf-8') as writer:
            record = asdict(self, dict_factory=self._dict_factory)
            record['hyperparams'] = asdict(self.hyperparams)
            writer.write(json.dumps(record))
        self._written = True

    def read(self, fname: str) -> None:
        """Read data from JSON file."""
        with open(fname, encoding='utf-8') as reader:
            data = json.load(reader)
            self.context = data['context']
            self.hyperparams = Hyperparams(**data['hyperparams'])
            self.response_probabilities = (
                data['response_probabilities'][0],
                data['response_probabilities'][1],
            )
            self.response_topk = (
                data['response_topk'][0],
                data['response_topk'][1],
            )
            self.sampling_method = data['sampling_method']
            self.branch_sampler = data['branch_sampler']
        self._written = True


    @staticmethod
    def _dict_factory(x: 'LLMOutputDataFull') -> dict:
        """Create dict but exclude state variables."""
        exclude = ('_written', )
        return {k: v for (k, v) in x if k not in exclude}

