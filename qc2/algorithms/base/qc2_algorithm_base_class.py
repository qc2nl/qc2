"""Module defining base class for all algorithms."""
from abc import ABC
from typing import Union
from typing import Any
import h5py


from ...data.qcschema.qc_qcircuit import QCQCircuit
from ...data.qcschema.qc_second_q import QCSecondQ
from ..second_q.second_quantizer import SecondQuantizer

# from ...qc2_driver import QC2

class QC2BaseAlgorithm(ABC):
    """Base class for all qc2 algos."""

    second_quantizer: SecondQuantizer
    active_space: Any
    mapper: Any
    qc2data: Any

    def set_qc2data(self, qc2data):
        """set the data"""
        self.qc2data = qc2data
        self.second_quantizer = SecondQuantizer(qc2data)

    def run(self, *args, **kwargs):
        """run it"""
        raise NotImplementedError("QC2BaseAlgorithm doens't have a .run() implemented ")
    
    def save(self, datafile: Union[h5py.File, str]) -> None:
        """Dumps qchem data to a datafile using QCSchema or FCIDump formats."""
    
        qcsecondq = QCSecondQ(
            active_electrons = self.active_space.num_active_electrons,
            active_spatial_orbitals = self.active_space.num_active_spatial_orbitals,
            inactive_energy = self.e_core,
            creation_anhiliton_operators = list(self.second_q_op.keys()),
            creation_anhiliton_coefficients = list(self.second_q_op.values()))
        
        coeffs, paulis = self.mapper.get_representation(self.qubit_op)
        qcqcircuit = QCQCircuit(
            backend = self.mapper.backend,
            hamiltonian_pauli_strings = paulis,
            hamiltonian_coefficients = coeffs
        )

        with h5py.File(datafile, 'a') as h5file:
            qcsecondq.to_hdf5(h5file)    
            qcqcircuit.to_hdf5(h5file)

    def _init_qubit_hamiltonian(self):

        if self.qc2data is None:
            raise ValueError(f"qc2data attribute set incorrectly in {self.__class__.__name__}")

        # compute the second q optimized
        self.e_core, self.second_q_op = self.second_quantizer.get_fermionic_hamiltonian(
            self.active_space.num_active_electrons,
            self.active_space.num_active_spatial_orbitals
        )

        # map the secondq operator to qubits
        self.qubit_op = self.mapper.map(self.second_q_op)

        self.save(self.qc2data._filename)

        
