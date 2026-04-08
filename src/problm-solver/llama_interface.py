"""llama.cpp python interface for running local models."""

import numpy as np
from llama_cpp import Llama

MODEL_PATH = ('/home/clio/Documents/Code/Prob_AI_RSE/llama/models/'
              'NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf')

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


# TESTING VIA SCRIPTS
def test_query_sky() -> str:
    """Load the local GGUF model and ask why the sky is blue."""
    llm = Llama(model_path=MODEL_PATH, n_ctx=2048)
    output = llm.create_chat_completion(
        messages=[
            {'role': 'user', 'content': 'Why is the sky blue?'},
        ],
        max_tokens=256,
    )
    return output['choices'][0]['message']['content']


if __name__ == '__main__':
    print(test_query_sky())
