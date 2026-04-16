"""Data container for LLM output."""

import json
import numpy as np
import numpy.typing as npt


class LLMOutputData:
    """Stores LLM output data and handles serialization."""

    def __init__(self, prompt: str, data: npt.NDArray[str]) -> None:
        """Initialization.

        Args:
            prompt: The prompt used to generate the data.
            data: Array of string responses from the LLM.
        """
        self.prompt = prompt
        self.data = data
        self.written = False

    def write(self, fname: str) -> None:
        """Saves data to a file in JSONL format.

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
        """Reads data to this object from a JSONL file.
        """
        responses = []
        with open(fname, 'r', encoding='utf-8') as reader:
            for line in reader:
                record = json.loads(line)
                self.prompt = record['prompt']
                responses.append(record['response'])
        self.data = np.array(responses, dtype=str)
        self.written = True
