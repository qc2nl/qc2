"""Module docstring"""
import numpy as np


def vector_to_skew_symmetric(vector):
    r"""
    Map a vector to an anti-symmetric matrix with np.tril_indices.

    For example, the resulting matrix for `np.array([1,2,3,4,5,6])` is:

    .. math::
        \begin{pmatrix}
            0 & -1 & -2 & -4\\
            1 &  0 & -3 & -5\\
            2 &  3 &  0 & -6\\
            4 &  5 &  6 &  0
        \end{pmatrix}

    Args:
        vector (torch.Tensor): 1d tensor
    """
    size = int(np.sqrt(8 * np.shape(vector)[0] + 1) + 1) // 2
    matrix = np.zeros((size, size))
    tril_indices = np.tril_indices(size, k=-1)
    matrix[tril_indices[0], tril_indices[1]] = vector
    matrix[tril_indices[1], tril_indices[0]] = -vector
    return matrix


def get_active_space_idx(
        nao, nelectron,
        n_active_orbitals,
        n_active_electrons
):
    """Calculates active space indices given active orbitals and electrons."""
    # Set active space parameters
    nelecore = sum(nelectron) - sum(n_active_electrons)
    if nelecore % 2 == 1:
        raise ValueError('odd number of core electrons')

    occ_idx = np.arange(nelecore // 2)
    act_idx = (occ_idx[-1] + 1 + np.arange(n_active_orbitals)
               if len(occ_idx) > 0
               else np.arange(n_active_orbitals))
    virt_idx = np.arange(act_idx[-1]+1, nao)

    return occ_idx, act_idx, virt_idx


def get_non_redundant_indices(occ_idx, act_idx, virt_idx, freeze_active):
    """Calculate non-redundant indices for indexing kappa vectors
    for a given active space"""
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

