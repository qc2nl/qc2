import numpy as np
import ffsim
from pyscf import cc
from qiskit.circuit import QuantumCircuit
from qiskit_nature.second_q.mappers import QubitMapper
from qiskit_nature.second_q.circuit.library import HartreeFock
from typing import Optional


class LUCJ:
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
        self.num_spatial_orbitals = num_spatial_orbitals
        self.num_particles = num_particles
        self.n_reps = n_reps

#        if initial_state is None:
#            self.initial_state = HartreeFock(
#                num_spatial_orbitals, num_particles, QubitMapper()
#            )
#        else:
#            self.initial_state = initial_state

    def build(self) -> np.ndarray:
        """Constructs and returns the LUCJ ansatz state."""
        # Get CCSD t2 amplitudes for initializing ansatz
        ccsd = cc.CCSD(self.scf).run()
        operator = ffsim.UCJOpSpinBalanced.from_t_amplitudes(ccsd.t2, n_reps=self.n_reps)
#        print(operator)

        # Create reference state (Hartree-Fock)
        reference_state = ffsim.hartree_fock_state(
            self.num_spatial_orbitals, self.num_particles
        )
        print(reference_state, self.num_spatial_orbitals, self.num_particles )
#        print(reference_state.shape)
#        print(operator.shape)
        # Apply UCJ operator to reference state
        return ffsim.apply_unitary(reference_state, operator, norb=self.num_spatial_orbitals, nelec=self.num_particles)

