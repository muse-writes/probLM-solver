"""Module for downloading various common testing datasets."""

import logging

from problm_solver.datasets import math500

logging.getLogger('httpcore').setLevel(logging.WARNING)

get_math500 = math500.get_data
get_problems_math500 = math500.get_problems

__all__ = [
    'get_math500',
    'get_problems_math500,'
]
