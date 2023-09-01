""""Module docstring."""
from typing import Tuple
import numpy as np
from scipy.linalg import expm

from qiskit_nature.second_q.formats.qcschema import QCSchema
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy

from .utils import (
    reshape_2, reshape_4,
    vector_to_skew_symmetric,
    get_active_space_idx,
    non_redundant_indices
)


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
        self.int1e_mo_a, self.int1e_mo_b = self._get_onebody_integrals()
        (self.int2e_mo_aa, self.int2e_mo_bb,
         self.int2e_mo_ba) = self._get_twobody_integrals()

        # active space parameters
        self.n_active_orbitals = n_active_orbitals
        self.n_active_electrons = n_active_electrons

        self.n_electrons = (
            self.qcschema.properties.calcinfo_nalpha,
            self.qcschema.properties.calcinfo_nbeta
        )
        self.nao = self.qcschema.properties.calcinfo_nmo

        (self.occ_idx, self.act_idx,
         self.virt_idx) = get_active_space_idx(
            self.nao, self.n_electrons,
            self.n_active_orbitals,
            self.n_active_electrons
         )

        # Calculate non-redundant orbital rotations
        self.params_idx = non_redundant_indices(
            self.occ_idx, self.act_idx,
            self.virt_idx, freeze_active
        )

        self.n_kappa = len(self.params_idx)

    def get_rotated_hamiltonian(self, kappa) -> ElectronicEnergy:
        """Docstring."""
        (h1_a, h1_b) = self.get_rotated_onebody_integrals(kappa)
        (h2_aa, h2_bb, h2_ba) = self.get_rotated_twobody_integrals(kappa)
        return ElectronicEnergy.from_raw_integrals(
            h1_a, h2_aa,
            h1_b, h2_bb, h2_ba,
            auto_index_order=False
        )

    def get_rotated_onebody_integrals(self, kappa) -> tuple[np.ndarray,
                                                            np.ndarray]:
        """Docstring."""
        rot_mat_a = self.get_rotation_matrix(kappa)

        # restricted case constraint
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

        return hij_rot, hij_b_rot

    def get_rotated_twobody_integrals(self, kappa) -> tuple[np.ndarray,
                                                            np.ndarray,
                                                            np.ndarray]:
        """Docstring."""
        rot_mat_a = self.get_rotation_matrix(kappa)

        # restricted case constraint
        rot_mat_b = rot_mat_a
        print(rot_mat_a)
        einsum_subscripts = 'pi,qj,pqrs,rk,sl->pqrs'
        hijkl_rot = np.einsum(einsum_subscripts,
                              rot_mat_a.T,
                              rot_mat_a.T,
                              self.int2e_mo_aa,
                              rot_mat_a,
                              rot_mat_a)

        hijkl_bb_rot = None
        hijkl_ba_rot = None

        if self.int2e_mo_bb is not None:
            hijkl_bb_rot = np.einsum(einsum_subscripts,
                                     rot_mat_b.T,
                                     rot_mat_b.T,
                                     self.int2e_mo_bb,
                                     rot_mat_b,
                                     rot_mat_b)

            hijkl_ba_rot = np.einsum(einsum_subscripts,
                                     rot_mat_b.T,
                                     rot_mat_a.T,
                                     self.int2e_mo_ba,
                                     rot_mat_b,
                                     rot_mat_a)

        return hijkl_rot, hijkl_bb_rot, hijkl_ba_rot

    def get_rotation_matrix(self, kappa):
        """Creates rotationa matrix from kappa parameters."""
        kappa_matrix = self.kappa_vector_to_matrix(kappa)
        return expm(-kappa_matrix)

    def kappa_vector_to_matrix(self, kappa):
        """Generates skew-symm. matrix from orbital rotation parameters."""
        kappa_total_vector = np.zeros(self.nao * (self.nao - 1) // 2)
        kappa_total_vector[np.array(self.params_idx)] = kappa
        return vector_to_skew_symmetric(kappa_total_vector)

    def _get_onebody_integrals(self) -> tuple[np.ndarray, np.ndarray]:
        """Docstring."""
        norb = self.qcschema.properties.calcinfo_nmo
        hij = reshape_2(self.qcschema.wavefunction.scf_fock_mo_a, norb)
        hij_b = None

        if self.qcschema.wavefunction.scf_fock_mo_b is not None:
            hij_b = reshape_2(self.qcschema.wavefunction.scf_fock_mo_b, norb)

        return hij, hij_b

    def _get_twobody_integrals(self) -> tuple[np.ndarray, np.ndarray,
                                              np.ndarray]:
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
