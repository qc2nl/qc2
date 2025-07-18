"""Module defining base class for all algorithms."""
from abc import ABC


class BaseAlgorithm(ABC):
    """Base class for all qc2 algos."""

    def set_qc2data(self, qc2data):
        """set the data"""
        self.qc2data = qc2data

    def run(self, *args, **kwargs):
        """run it"""
        raise NotImplementedError("BaseAlgorithm doens't have a .run() implemented ")


    def _init_qubit_hamiltonian(self):
        if self.qc2data is None:
            raise ValueError(f"qc2data attribute set incorrectly in {self.__class__.__name__}")


        self.e_core, self.second_q_op = self.qc2data.get_fermionic_hamiltonian(
            self.active_space.num_active_electrons,
            self.active_space.num_active_spatial_orbitals
        )

        self.qubit_op = self.mapper.map(self.second_q_op)
