"""llama.cpp python interface for running local models."""

from typing import Any

import numpy as np
import numpy.typing as npt
from llama_cpp import Llama

from problm_solver.data import LLMOutputData


class ModelInstance:
    """Model class."""

    def __init__(self, fname: str, context: str) -> None:
        """Init method."""
        self._llm = Llama(model_path=fname, n_ctx=2048)
        self.context = context


    def query_n_times(self, n: int) -> npt.NDArray[Any]:
        """Query the LLM with the same context N times, return the output."""
        return np.array([self.query() for _ in range(n)], dtype=str)


    def query(self) -> str:
        """Query the LLM once."""
        output = self._llm.create_chat_completion(
            messages=[
                {'role': 'user', 'content': self.context},
            ],
            max_tokens=512,
        )
        return output['choices'][0]['message']['content']


    def generate_data(self, n_samples: int) -> LLMOutputData:
        """Generate data by querying the LLM `n_samples` times."""
        data = self.query_n_times(n_samples)
        return LLMOutputData(prompt=self.context, data=data)
