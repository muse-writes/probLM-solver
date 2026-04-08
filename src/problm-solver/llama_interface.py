"""llama.cpp python interface for running local models."""

import numpy as np
from llama_cpp import Llama


class ModelInstance():
    """Model class.
    """

    def __init__(self, fname: str, context: str) -> None:
        """Init method.
        """
        self.llm = Llama(model_path=fname, n_ctx=2048)
        self.context = context


    def query_n_times(self, n: int) -> list[str]:
        """Queries the LLM with the same context N times, returns the output.
        """
        return np.array([self.query() for _ in range(n)], dtype=str)


    def query(self) -> str:
        """Queries the LLM once.
        """
        output = self.llm.create_chat_completion(
            messages=[
                {'role': 'user', 'content': self.context},
            ],
            max_tokens=256,
        )
        return output['choices'][0]['message']['content']
