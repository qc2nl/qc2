"""Module docstring"""
import numpy as np


def reshape_2(arr, dim, dim_2=None):
    """Docstring."""
    return np.asarray(arr).reshape((dim, dim_2 if dim_2 is not None else dim))


def reshape_4(arr, dim):
    """Docstring."""
    return np.asarray(arr).reshape((dim,) * 4)