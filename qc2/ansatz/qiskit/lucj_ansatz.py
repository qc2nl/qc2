import numpy as np
import ffsim
from pyscf import cc
from qiskit.circuit.library.blueprintcircuit import BlueprintCircuit
from qiskit.circuit import QuantumCircuit
from typing import Optional


class LUCJ(BlueprintCircuit):
    """LUCJ ansatz using Unitary Coupled-Cluster with Jastrow factor."""

    def __init__(
        self,
        mol_data,
        scf,
        num_spatial_orbitals: int,
        num_particles: tuple,
        n_reps: int = 2,
        initial_state: Optional[QuantumCircuit] = None,
    ):
        """
        Args:
            mol_data: Molecular data from ffsim
            scf: Self-consistent field calculation object
            num_spatial_orbitals: Number of spatial orbitals
            num_particles: Tuple of (alpha, beta) electrons
            n_reps: Number of repetitions in UCJ operator
            initial_state: Initial state circuit (Hartree-Fock if None)
        """
        
        self.mol_data = mol_data
        self.scf = scf
        self._num_spatial_orbitals = num_spatial_orbitals
        self._num_particles = num_particles
        self._num_qubits = 2 * num_spatial_orbitals
        self.n_reps = n_reps
        self._set_initial_state(initial_state)
        super().__init__("LUCJ")

    @property
    def num_qubits(self) -> int:
        """The number of qubits."""
        return self._num_qubits

    @num_qubits.setter
    def num_qubits(self, n: int) -> None:
        """Sets the number of qubits."""
        self._invalidate()
        self._num_qubits = n

    @property
    def num_spatial_orbitals(self) -> int:
        """The number of spatial orbitals."""
        return self._num_spatial_orbitals

    @num_spatial_orbitals.setter
    def num_spatial_orbitals(self, n: int) -> None:
        """Sets the number of spatial orbitals."""
        self._invalidate()
        self._num_spatial_orbitals = n

    @property
    def num_particles(self) -> tuple[int, int]:
        """The number of particles."""
        return self._num_particles

    @num_particles.setter
    def num_particles(self, n: tuple[int, int]) -> None:
        """Sets the number of particles."""
        self._invalidate()
        self._num_particles = n

    def _set_initial_state(self, initial_state: QuantumCircuit) -> None:
        """
        Sets the initial state circuit for the LUCJ ansatz.

        Args:
            initial_state: Initial state circuit (Hartree-Fock if None)
        """
        if initial_state is None:
            self.initial_state = ffsim.hartree_fock_state(
                self._num_spatial_orbitals, self._num_particles
        )
        else:
            self.initial_state = initial_state
            
    def _check_configuration(self, raise_on_failure: bool = True) -> bool:

        """Check if the configuration of the NLocal class is valid.

        Args:
            raise_on_failure: Whether to raise on failure.

        Returns:
            True, if the configuration is valid and the circuit can be constructed. Otherwise
            an ValueError is raised.

        Raises:
            ValueError: If the numbr fo qubit is not se.
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

    def _build(self) -> None:
        """Builds the Gate Fabric circuit
        """
        if self._is_built:
            return
        self._check_configuration()
        self._build_circuit()
        self._is_built = True

    def _build_circuit(self) -> None:
        """Constructs and returns the LUCJ ansatz state."""
        # Get CCSD t2 amplitudes for initializing ansatz
        ccsd = cc.CCSD(self.scf).run()
        operator = ffsim.UCJOpSpinBalanced.from_t_amplitudes(ccsd.t2, n_reps=self.n_reps)

        # Apply UCJ operator to reference state
        self.compose(ffsim.apply_unitary(self.initial_state, operator, norb=self.num_spatial_orbitals, nelec=self.num_particles), inplace=True)

