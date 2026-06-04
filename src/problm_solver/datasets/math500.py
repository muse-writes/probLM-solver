"""Gets MATH500 dataset from `huggingface.co`."""


import logging

import pandas as pd

_logger = logging.getLogger(__name__)

def get_data(fname: str | None = None) -> pd.DataFrame:
    """Download the MATH500 dataset and store it as a pandas :class:`DataFrame`.

    Basic function for getting MATH500. Logs the download and returns the data.
    Data is organised into six columns:
    - ``problem``: str
    - ``solution``: str
    - ``answer``: str
    - ``subject``: str
    - ``level``: int
    - ``unique_id``: str

    :returns: pandas :class:`DataFrame` containing the dataset.
    """
    if fname is not None:
        data = pd.read_json(fname, lines=True)
    else:
        data = pd.read_json('hf://datasets/HuggingFaceH4/MATH-500/test.jsonl', lines=True)
    _logger.info('MATH500 dataset successfully loaded.')
    return data


def get_problems(fname: str | None = None) -> list[str]:
    """Return only the problem statements from the MATH500 dataset.

    Convenience wrapper around :func:`get_data` that extracts the first
    column (``problem``) and returns it as a plain list of strings, stripping
    away the DataFrame overhead for callers that only need the questions.

    :returns: List of problem statement strings, one per dataset row.
    """
    if fname is not None:
        return get_data(fname).iloc[:,0].tolist()
    return get_data().iloc[:, 0].tolist()
