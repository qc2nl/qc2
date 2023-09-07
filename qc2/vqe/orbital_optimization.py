""""Module docstring."""
from typing import Tuple, Optional
import numpy as np
from scipy.linalg import expm

# provisory imports
from qiskit_nature.second_q.formats.qcschema import QCSchema
from qiskit_nature.second_q.formats import qcschema_to_problem
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.problems import ElectronicBasis
from qiskit_nature.second_q.operators import ElectronicIntegrals
from qiskit_nature.second_q.transformers import BasisTransformer
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer

from .utils import (
    vector_to_skew_symmetric,
    get_active_space_idx,
    get_non_redundant_indices
)


class OrbitalOptimization:
    """Docstring."""

    def __init__(
            self, qcschema: QCSchema,
            n_active_electrons: Tuple,
            n_active_orbitals: int,
            freeze_active: bool = False,
            es_problem: Optional[ElectronicStructureProblem] = None
    ) -> None:
        """Module Docstrings."""

        # qcschema dataclass object
        self.qcschema = qcschema

        # instance of `ElectronicStructureProblem`
        self.es_problem = es_problem

        # active space parameters
        self.n_active_orbitals = n_active_orbitals
        self.n_active_electrons = n_active_electrons

        self.n_electrons = (
            self.qcschema.properties.calcinfo_nalpha,
            self.qcschema.properties.calcinfo_nbeta
        )
        self.nao = self.qcschema.properties.calcinfo_nmo

        (self.occ_idx,
         self.act_idx,
         self.virt_idx) = get_active_space_idx(
            self.nao, self.n_electrons,
            self.n_active_orbitals,
            self.n_active_electrons
         )

        # calculate non-redundant orbital rotations
        self.params_idx = get_non_redundant_indices(
            self.occ_idx, self.act_idx,
            self.virt_idx, freeze_active
        )

        # set dimension of the kappa vector
        self.n_kappa = len(self.params_idx)

    def get_transformed_hamiltonian(self, kappa
    ) -> Tuple[float, ElectronicEnergy]:
        """Docstring."""
        # create a instance of `ElectronicStructureProblem`
        self.es_problem = qcschema_to_problem(
            self.qcschema, include_dipole=False)

        ## calculate active space `ElectronicEnergy` hamiltonian
        #(core_energy, hamiltonian) = self.get_active_space_hamiltonian(
        #    self.es_problem)

        # calculate rotation matrix given the kappa parameters
        coeffs_a = self.get_rotation_matrix(kappa)
        # restricted case constraint
        coeffs_b = coeffs_a

        # create an instance of `BasisTransformer`
        # see qiskit_nature/second_q/transformers/basis_transformer.py
        transformer = BasisTransformer(
            ElectronicBasis.MO,
            ElectronicBasis.MO,
            ElectronicIntegrals.from_raw_integrals(coeffs_a, h1_b=coeffs_b),
        )

        # transform the original `ElectronicStructureProblem`
        # and `ElectronicEnergy` into the new rotated basis
        #transformed_hamiltonian = transformer.transform_hamiltonian(hamiltonian)
        #return core_energy, transformed_hamiltonian
        transformed_es_problem = transformer.transform(self.es_problem)
        # calculate active space `ElectronicEnergy` hamiltonian
        return self.get_active_space_hamiltonian(transformed_es_problem)

    def get_active_space_hamiltonian(
            self, es_problem: ElectronicStructureProblem
    ) -> Tuple[float, ElectronicEnergy]:
        """Docstring"""
        # convert `ElectronicStructureProblem`` into an instance of
        # `ElectronicEnergy` hamiltonian in second quantization;
        # see qiskit_nature/second_q/problems/electronic_structure_problem.py
        hamiltonian = es_problem.hamiltonian

        # in case of space selection, reduce the space extent of the
        # fermionic Hamiltonian based on the number of active electrons
        # and orbitals
        transformer = ActiveSpaceTransformer(
            self.n_active_electrons, self.n_active_orbitals)

        transformer.prepare_active_space(
            es_problem.num_particles, es_problem.num_spatial_orbitals)

        # after preparation, transform hamiltonian
        active_space_hamiltonian = transformer.transform_hamiltonian(
            hamiltonian)

        # set up core energy after transformation
        nuclear_repulsion_energy = active_space_hamiltonian.constants[
            'nuclear_repulsion_energy']
        inactive_space_energy = active_space_hamiltonian.constants[
            'ActiveSpaceTransformer']
        core_energy = nuclear_repulsion_energy + inactive_space_energy

        return core_energy, active_space_hamiltonian

    def get_rotation_matrix(self, kappa):
        """Creates rotationa matrix from kappa parameters."""
        kappa_matrix = self.kappa_vector_to_matrix(kappa)
        return expm(-kappa_matrix)

    def kappa_vector_to_matrix(self, kappa):
        """Generates skew-symm. matrix from orbital rotation parameters."""
        kappa_total_vector = np.zeros(self.nao * (self.nao - 1) // 2)
        kappa_total_vector[np.array(self.params_idx)] = kappa
        return vector_to_skew_symmetric(kappa_total_vector)
