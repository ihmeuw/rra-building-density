from typing import Any

import numpy as np
import numpy.typing as npt


def safe_divide(
    x: npt.NDArray[np.floating[Any]],
    y: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.floating[Any]]:
    """Divide two arrays, returning 0 where both are 0 and NaN where only the denominator is 0."""
    z = x / y
    z[(x == 0) & (y == 0)] = 0
    z[np.isinf(z)] = np.nan
    return z
