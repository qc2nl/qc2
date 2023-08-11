""""Module docstring."""

from typing import Tuple
import numpy as np
from scipy.linalg import expm

from qiskit_nature.second_q.formats.qcschema import QCSchema

from .utils import (reshape_2, reshape_4)


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
    # matrix = np.convert_like(np.zeros((size, size)), vector)
    tril_indices = np.tril_indices(size, k=-1)
    matrix[tril_indices[0], tril_indices[1]] = vector
    matrix[tril_indices[1], tril_indices[0]] = -vector
    #matrix = np.set_index(
    #    matrix, (tril_indices[0], tril_indices[1]), vector)
    #matrix = np.set_index(
    #    matrix, (tril_indices[1], tril_indices[0]), -vector)
    return matrix


def _get_active_space_idx(nao, nelectron,
                          n_active_orbitals,
                          n_active_electrons):
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


def _non_redundant_indices(occ_idx, act_idx, virt_idx, freeze_active):
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


class oo_energy:
    """Docstring."""

    def __init__(
            self, qcschema: QCSchema,
            n_active_electrons: Tuple,
            n_active_orbitals: int,
            freeze_active: bool = False
            ) -> None:
        """Module Docstrings."""

        # qcschema dataclass object
        self.qcschema = qcschema

        # molecular integrals
        self.int1e_mo_a, self.int1e_mo_b = self._get_int1e_mo()
        (self.int2e_mo_aa, self.int2e_mo_bb,
         self.int2e_mo_ba) = self._get_int2e_mo()

        # active space parameters
        self.n_active_orbitals = n_active_orbitals
        self.n_active_electrons = n_active_electrons

        self.n_electrons = (
            self.qcschema.properties.calcinfo_nalpha,
            self.qcschema.properties.calcinfo_nbeta
            )
        self.nao = self.qcschema.properties.calcinfo_nmo

        (self.occ_idx, self.act_idx,
         self.virt_idx) = _get_active_space_idx(self.nao, self.n_electrons,
                                                self.n_active_orbitals,
                                                self.n_active_electrons)

        #print(self.occ_idx, self.act_idx, self.virt_idx)

        # Calculate non-redundant orbital rotations
        self.params_idx = _non_redundant_indices(
            self.occ_idx, self.act_idx, self.virt_idx, freeze_active
        )

        #print(self.params_idx)

        self.n_kappa = len(self.params_idx)

    def rotate_int1e_mo(self, kappa) -> tuple[np.ndarray, np.ndarray]:
        """Docstring."""
        rot_mat_a = self.get_rotation_matrix(kappa)
        rot_mat_b = rot_mat_a
        hij_rot = np.einsum('ij,jk,kl->il',
                            rot_mat_a.T,
                            self.int1e_mo_a,
                            rot_mat_a)
        hij_b_rot = None

        if self.int1e_mo_b is not None:
            hij_b_rot = np.einsum('ij,jk,kl->il',
                                  rot_mat_b.T,
                                  self.int1e_mo_b,
                                  rot_mat_b)
        # hij_rot = rot_mat_a.T @ self.int1e_mo_a @ rot_mat_a.T
        # hij_b_rot = rot_mat_b.T @ self.int1e_mo_b @ rot_mat_b.T
        return hij_rot, hij_b_rot

    def rotate_int2e_mo(self, kappa) -> tuple[np.ndarray, np.ndarray,
                                              np.ndarray, np.ndarray]:
        """Docstring."""
        pass   

    def get_rotation_matrix(self, kappa):
        """Creates rotationa matrix from kappa parameters."""
        kappa_matrix = self.kappa_vector_to_matrix(kappa)
        return expm(-kappa_matrix)

    def kappa_vector_to_matrix(self, kappa):
        """Generates skew-symm. matrix from orbital rotation parameters."""
        kappa_total_vector = np.zeros(self.nao * (self.nao - 1) // 2)
        kappa_total_vector[np.array(self.params_idx)] = kappa
        return vector_to_skew_symmetric(kappa_total_vector)

    def _get_int1e_mo(self) -> tuple[np.ndarray, np.ndarray]:
        """Docstring."""
        norb = self.qcschema.properties.calcinfo_nmo
        hij = reshape_2(self.qcschema.wavefunction.scf_fock_mo_a, norb)
        hij_b = None

        if self.qcschema.wavefunction.scf_fock_mo_b is not None:
            hij_b = reshape_2(self.qcschema.wavefunction.scf_fock_mo_b, norb)

        return hij, hij_b

    def _get_int2e_mo(self) -> tuple[np.ndarray, np.ndarray,
                                     np.ndarray, np.ndarray]:
        """Docstring."""
        norb = self.qcschema.properties.calcinfo_nmo
        hijkl = reshape_4(self.qcschema.wavefunction.scf_eri_mo_aa, norb)
        hijkl_bb = None
        hijkl_ba = None

        if self.qcschema.wavefunction.scf_eri_mo_bb is not None:
            hijkl_bb = reshape_4(
                self.qcschema.wavefunction.scf_eri_mo_bb, norb
                )
        if self.qcschema.wavefunction.scf_eri_mo_ba is not None:
            hijkl_ba = reshape_4(
                self.qcschema.wavefunction.scf_eri_mo_ba, norb
                )

        return hijkl, hijkl_bb, hijkl_ba
