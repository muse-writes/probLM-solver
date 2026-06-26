"""Resolve RNG handling."""

import numpy as np

from problm_solver.random.random import RandomManager
from problm_solver.random.types import RNGLike


def resolve_rng(
    rng: RNGLike,
    *,
    stream: str,
    default_manager: RandomManager | None = None,
    fresh: bool = False,
) -> np.random.Generator:
    """Take an :type:`RNGLike` and returns a generator.

    :param rng: :type:`RNGLike` object, to be sorted.
    :param stream: Named stream for :class:`RandomManager` objects.
    :param default_manager: Fallback manager, used when calling
        :meth:`resolve_rng` in classes that manage their own RNG state.
    :param fresh: Whether or not to reinitialise a generator (should be
        ``False`` in most cases).
    :returns: A numpy generator exhibiting specified behaviour.
    """
    if isinstance(rng, np.random.Generator):
        if fresh:
            msg = (
                'resolve_rng(..., fresh=True) cannot be used with an existing '
                'numpy.random.Generator. Pass a RandomManager to request a '
                'fresh derived stream, or set fresh=False.'
            )
            raise ValueError(msg)
        gen = rng
    elif isinstance(rng, (int, np.integer)):
        gen = np.random.default_rng(rng)
    elif isinstance(rng, RandomManager):
        gen = rng.spawn_rng(stream) if fresh else rng.get_rng(stream)
    elif rng is None and default_manager is not None:
        gen = default_manager.spawn_rng(stream) if fresh else default_manager.get_rng(stream)
    else:
        gen = np.random.default_rng()
    return gen

