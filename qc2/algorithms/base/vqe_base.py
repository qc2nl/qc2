"""This module defines the base class for VQE algorithms."""
from qc2.algorithms.base.base_algorithm import BaseAlgorithm


class VQEBASE(BaseAlgorithm):
    """Base class for VQE"""

    def __init__(self, qc2data=None, format=""):
        """Initiate the class

        Args:
            qc2data (data container): qc2 data container with scf information.
            active_space (ActiveSpace, optional): Description of the active
                space. Defaults to ActiveSpace((2, 2), 2).
            mapper (qiskit_nature.second_q.mappers, optional): Method used to
                map the qubits. Defaults to JordanWignerMapper().
            format (str, optional): Which quantum backend we want to use.
                Defaults to "qiskit".
        """
        self.qc2data = qc2data
        self.format = format

    def _init_qubit_hamiltonian(self):
        if self.qc2data is None:
            raise ValueError("qc2data attribute set incorrectly in VQE.")

        self.e_core, self.qubit_op = self.qc2data.get_qubit_hamiltonian(
            self.active_space.num_active_electrons,
            self.active_space.num_active_spatial_orbitals,
            self.mapper,
            format=self.format,
        )
