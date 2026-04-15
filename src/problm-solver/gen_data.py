"""Code for generating data from llama interface
"""

import json
import numpy as np
import numpy.typing as npt

from llama_interface import ModelInstance


class LLMOutputData:
    """Handles the generation, storage, and reading of LLM output for a
    specific model.
    """

    def __init__(self, model: ModelInstance) -> None:
        """Initialization.
        """
        self.model = model

# Data is None until generated.
        self.data = None
        self.written = False


    def generate(self, n_samples: int) -> None:
        """Generates data from query.
        """
# Guard against overwriting data in class instance.
        if self.data is not None and not self.written:
            raise UserWarning(
                'Should not overwrite data variable in '
                '`LLMOutputData` without first calling `write()`'
            )
        self.data: npt.NDArray[str] = self.model.query_n_times(n_samples)
        self.written = False


    def write(self, fname=None) -> str:
        """Saves data to a file in JSONL format.

        Default filename is `[model]_[timestamp].jsonl`.
        """
        with open(fname, 'w', encoding='utf-8') as writer:
            for i, response in enumerate(self.data, start=1):
                record = {
                    'id': i,
                    'prompt': self.model.context,
                    'response': response
                }
                writer.write(json.dumps(record) + '\n')
        self.written = True
