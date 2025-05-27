from typing import Iterable
import numpy as np
import ffsim
import pyscf
from qiskit.circuit import QuantumCircuit
from typing import Optional


class LUCJ():
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
        
       
        self.mol = mol
        self.active_space = active_space
        self._run_scf()
        self._num_qubits = 2 * self._num_spatial_orbitals
        self.n_reps = n_reps
        self._set_initial_state(initial_state)
        self._name = name

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
        self.scf = pyscf.scf.RHF(self.mol).run()
        self.mol_data = ffsim.MolecularData.from_scf(self.scf, active_space=self.active_space)
        self._num_spatial_orbitals = self.mol_data.norb
        self._num_particles = self.mol_data.nelec

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
        

    def get_state(self) -> np.ndarray:
        """Constructs and returns the LUCJ ansatz state."""
        # Get CCSD t2 amplitudes for initializing ansatz
        ccsd = pyscf.cc.CCSD(self.scf, frozen=[i for i in range(self.mol.nao_nr()) if i not in self.active_space]).run()
        operator = ffsim.UCJOpSpinBalanced.from_t_amplitudes(ccsd.t2, n_reps=self.n_reps)

        # Apply UCJ operator to reference state
        return ffsim.apply_unitary(self.initial_state, operator, norb=self.num_spatial_orbitals, nelec=self.num_particles)
