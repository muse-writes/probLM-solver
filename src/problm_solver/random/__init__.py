"""Module interface for random."""

from problm_solver.random.random import RandomManager
from problm_solver.random.resolve import resolve_rng
from problm_solver.random.types import RNGLike

__all__ = [
    'RNGLike',
    'RandomManager',
    'resolve_rng',
]
