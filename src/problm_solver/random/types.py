"""RNG types."""

import numpy as np

from problm_solver.random.random import RandomManager

type RNGLike = RandomManager | np.random.Generator | int | None
