"""Module containing general helper functions."""
from typing import Union, List, Optional
import numpy as np


def vector_to_skew_symmetric(
    vector: Union[List[float], np.ndarray]    
) -> np.ndarray:
    r"""
    Map a vector to an anti-symmetric matrix with np.tril_indices.

    Args:
        vector (Union[List[float], np.ndarray]): 1d tensor

    Returns:
        np.ndarray:
            A skew-symmetric matrix corresponding to the input vector.
        
    **Example**

    The resulting matrix for `np.array([1,2,3,4,5,6])` is:

    .. math::
        \begin{pmatrix}
            0 & -1 & -2 & -4\\
            1 &  0 & -3 & -5\\
            2 &  3 &  0 & -6\\
            4 &  5 &  6 &  0
        \end{pmatrix}
    """
    size = int(np.sqrt(8 * np.shape(vector)[0] + 1) + 1) // 2
    matrix = np.zeros((size, size))
    tril_indices = np.tril_indices(size, k=-1)
    matrix[tril_indices[0], tril_indices[1]] = vector
    matrix[tril_indices[1], tril_indices[0]] = -vector
    return matrix


def skew_symmetric_to_vector(kappa_matrix: np.ndarray) -> np.ndarray:
    """Converts a skew-symmetric matrix to a 1D vector.

    This function extracts the lower triangular part of a skew-symmetric
    matrix and flattens it to a 1D vector.

    Args:
        kappa_matrix (np.ndarray): A skew-symmetric matrix.

    Returns:
        np.ndarray:
            A 1-dimensional vector containing the elements of the lower
            triangular part of the input matrix.
    """
    size = np.shape(kappa_matrix)[0]
    tril_indices = np.tril_indices(size, k=-1)
    return kappa_matrix[tril_indices[0], tril_indices[1]]


def reshape_2(
    arr: Union[List[float], np.ndarray],
    dim: int,
    dim_2: Optional[int] = None
) -> np.ndarray:
    """
    Reshapes a flattened 2D array into a 2D array with specified dimensions.

    Args:
        arr (Union[List[float], np.ndarray]): A flattened array or list to be
            reshaped.
        dim (int): The first dimension of the reshaped array.
        dim_2 (int, optional): The second dimension of the reshaped array.
            If None, it is set equal to dim.

    Returns:
        np.ndarray:
            A 2-dimensional array reshaped according to the specified
            dimensions.
    """
    return np.asarray(arr).reshape((dim, dim_2 if dim_2 is not None else dim))


def get_non_redundant_indices(
    occ_idx: np.ndarray,
    act_idx: np.ndarray,
    virt_idx: np.ndarray,
    freeze_active: bool
) -> np.ndarray:
    """Calculates the non-redundant indices for orbital parameters.

    Args:
        occ_idx (np.ndarray): Indices of occupied orbitals.
        act_idx (np.ndarray): Indices of active orbitals.
        virt_idx (np.ndarray): Indices of virtual orbitals.
        freeze_active (bool): If True, active orbitals are frozen.

    Returns:
        np.ndarray:
            Indices of non-redundant orbital rotation parameters.
    """
    no, na, nv = len(occ_idx), len(act_idx), len(virt_idx)
    nao = no + na + nv
    rotation_sizes = [no * na, na * nv, no * nv]
    if not freeze_active:
        rotation_sizes.append(na * (na - 1) // 2)
    n_kappa = sum(rotation_sizes)
    params_idx = np.array([], dtype=int)
    num = 0
    for l_idx, r_idx in zip(*np.tril_indices(nao, -1)):
        if not (
            ((l_idx in act_idx and r_idx in act_idx) and freeze_active)
                or (l_idx in occ_idx and r_idx in occ_idx)
                or (l_idx in virt_idx and r_idx in virt_idx)):
            params_idx = np.append(params_idx, [num])
        num += 1
    assert n_kappa == len(params_idx)
    return params_idx

