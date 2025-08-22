from typing import Iterable
import numpy as np
import ffsim
import pyscf
from qiskit.circuit.library.blueprintcircuit import BlueprintCircuit
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.exceptions import QiskitError
from typing import Optional


class LUCJ(BlueprintCircuit):
    """LUCJ ansatz using Unitary Coupled-Cluster with Jastrow factor."""

    def __init__(
        self,
        mol: pyscf.gto.Mole,
        active_space: Iterable | None = None,
        n_reps: int = 2,
        initial_state: Optional[QuantumCircuit] = None,
        name: str | None = "LUCJ",
    ):
        """
        Args:
            scf: Self-consistent field calculation object
            num_spatial_orbitals: Number of spatial orbitals
            num_particles: Tuple of (alpha, beta) electrons
            n_reps: Number of repetitions in UCJ operator
            initial_state: Initial state circuit (Hartree-Fock if None)
        """
        
        super().__init__(name=name)
        self.mol = mol
        self.active_space = active_space
        self.scf, self.mol_data = self._run_scf()
        self._num_spatial_orbitals = self.mol_data.norb
        self._num_particles = self.mol_data.nelec
        self._num_qubits = 2 * self._num_spatial_orbitals
        self.n_reps = n_reps
        self._set_initial_state(initial_state)

        if self.num_qubits == 0:
            self.qregs = []
        else:
            self.qregs = [QuantumRegister(self.num_qubits, name="q")]
        self.ccsd = None 

    @property
    def num_qubits(self) -> int:
        """The number of qubits."""
        return self._num_qubits

    @num_qubits.setter
    def num_qubits(self, n: int) -> None:
        """Sets the number of qubits."""
        self._num_qubits = n

    @property
    def num_spatial_orbitals(self) -> int:
        """The number of spatial orbitals."""
        return self._num_spatial_orbitals

    @num_spatial_orbitals.setter
    def num_spatial_orbitals(self, n: int) -> None:
        """Sets the number of spatial orbitals."""
        self._num_spatial_orbitals = n

    @property
    def num_particles(self) -> tuple[int, int]:
        """The number of particles."""
        return self._num_particles

    @num_particles.setter
    def num_particles(self, n: tuple[int, int]) -> None:
        """Sets the number of particles."""
        self._num_particles = n

    def _run_scf(self):
        """Runs a self-consistent field calculation."""
        scf = pyscf.scf.RHF(self.mol).run()
        mol_data = ffsim.MolecularData.from_scf(scf, active_space=self.active_space)
        return scf, mol_data

    def _set_initial_state(self, initial_state: QuantumCircuit | None) -> None:
        """
        Sets the initial state circuit for the LUCJ ansatz.

        Args:
            initial_state: Initial state circuit (Hartree-Fock if None)
        """
        if initial_state is None:
            self._initial_state = ffsim.qiskit.PrepareHartreeFockJW(
                self._num_spatial_orbitals, self._num_particles
            )
        else:
            self._initial_state = initial_state

    def _check_configuration(self, raise_on_failure: bool = True) -> bool:

        """Check if the configuration of the NLocal class is valid.

        Args:
            raise_on_failure: Whether to raise on failure.

        Returns:
            True, if the configuration is valid and the circuit can be constructed. Otherwise
            an ValueError is raised.

        Raises:
            ValueError: If the numbr fo qubit is not set.
            ValueError: If the number of spatial orbitals is lower than the number of particles
        """
        valid = True
        if self.num_qubits is None:
            valid = False
            if raise_on_failure:
                raise ValueError("No number of qubits specified.")

        # check no needed parameters are None
        if self._num_spatial_orbitals < self._num_particles[0] or self._num_spatial_orbitals < self._num_particles[1]:
            valid = False
            if raise_on_failure :
                raise ValueError("Number of spatial orbitals inferior to number of particles.")

        return valid

    def _build(self):
        if self._is_built:
            return
        super()._build()

        # create the circuit
        circuit = QuantumCircuit(self.num_qubits, name=self.name) 

        # Apply initial state
        circuit.compose(self._initial_state.copy(), inplace=True, copy=False)

        # get teh lucj op
        operator = self.get_operator()

        # Apply UCJ operator to reference state
        circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(operator), 
                       range(self.num_qubits), 
                       copy=False)

        # append to self
        try:
            block = circuit.to_gate()
        except QiskitError:
            block = circuit.to_instruction()
        self.append(block, range(self.num_qubits), copy=False)

    def get_operator(self) -> ffsim.UCJOpSpinBalanced:
        """"
        Returns the LUCJ operator.
        """
        # Get CCSD t2 amplitudes for initializing ansatz
        if self.ccsd is None:
            self.ccsd = pyscf.cc.CCSD(self.scf, frozen=[i for i in range(self.mol.nao_nr()) if i not in self.active_space]).run()
        operator = ffsim.UCJOpSpinBalanced.from_t_amplitudes(self.ccsd.t2, n_reps=self.n_reps)
        return operator

    def get_state(self, initial_state: np.ndarray | None = None) -> np.ndarray:
        """Constructs and returns the LUCJ ansatz state."""
        operator = self.get_operator()
        if initial_state is None:
            initial_state = ffsim.hartree_fock_state(
                    self._num_spatial_orbitals, self._num_particles
                )
        # Apply UCJ operator to reference state
        return ffsim.apply_unitary(initial_state, 
                                   operator, 
                                   norb=self.num_spatial_orbitals, 
                                   nelec=self.num_particles)
