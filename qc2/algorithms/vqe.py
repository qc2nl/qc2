""""Module docstring."""
from typing import Tuple, List

from qc2.data import qc2Data


def _import_sdk(package):
    """Import required modules specific to a chosen SDK."""
    try:
        if package == 'qiskit':
            from qiskit_nature.second_q.circuit.library import HartreeFock as hf
            from qiskit_nature.second_q.circuit.library import UCCSD as ucc
            from qiskit_nature.second_q.mappers import JordanWignerMapper as mapper
            from qiskit.primitives import Estimator as estimator
            # import also later the backend
            return mapper, hf, ucc, estimator
    except ImportError as error:
        raise ImportError(
            "Unable to import the requested package. "
        ) from error


class VQE:
    """Docstring."""

    def __init__(
            self,
            qc2data: qc2Data,
            n_active_electrons: Tuple[int, int],
            n_active_orbitals: int,
            quantum_computing_sdk: str = 'qiskit'
    ) -> None:
        """Module Docstrings."""
        self.qc2data = qc2data

        # active space parameters
        self.n_active_orbitals = n_active_orbitals
        self.n_active_electrons = n_active_electrons

        # get electronic structure data
        self._get_qchem_data()

        # circuit data
        self.sdk = quantum_computing_sdk
        self.mapper = None
        self.reference_state = None
        self.ansatz = None
        self.estimator = None

    def _get_qchem_data(self) -> None:
        """Run ASE calculator"""
        self.qc2data.run()

    def _build_ansatz(self) -> None:
        """Set up VQE circuit."""
        # set up reference quantum circuit
        mapper, hf, ucc, estimator = _import_sdk(self.sdk)  # => provisory solution
        self.estimator = estimator()
        self.mapper = mapper()
        self.reference_state = hf(
            self.n_active_orbitals,
            self.n_active_electrons,
            self.mapper,
        )
        # set up the ansatz using unitary CC as variational form
        self.ansatz = ucc(
            self.n_active_orbitals,
            self.n_active_electrons,
            self.mapper,
            initial_state=self.reference_state
        )

    def get_rdms(self, var_params: List):
        """Get 1- and 2-RDMs."""
        # set up VQE a ansatz
        self._build_ansatz()
        if len(var_params) != self.ansatz.num_parameters:
            raise ValueError("Incorrect dimension for amplitude list.")

        # get the fermionic hamiltonian
        _, core_energy, fermionic_op = self.qc2data.get_fermionic_hamiltonian(
            self.n_active_electrons, self.n_active_orbitals
        )

        # run over the terms of the hamiltonian
        for key, coeff in fermionic_op.terms():
            # assign indices depending on one- or two-body term
            length = len(key)
            if length == 2:
                print(key, len(key))
            elif length == 4:
                print(key, len(key))
            # then calculate expectation values for each of those terms
            # and save into matrices...these are the RDMs!

    def energy_from_rdms(self):
        """Calculate energy based on 1- and 2-RDMs."""
        raise NotImplementedError

    def run(self) -> None:
        """Run VQE."""
        raise NotImplementedError
