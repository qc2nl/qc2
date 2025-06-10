"""Module defining the QPE algorithm for qiskit"""
import numpy as np
from scipy.linalg import expm
from qiskit_algorithms import PhaseEstimation, IterativePhaseEstimation
from qiskit import QuantumCircuit
from qiskit.primitives import Sampler
from qiskit.circuit.library import UnitaryGate
from qiskit_nature.second_q.circuit.library import HartreeFock
from qc2.algorithms.base.base_algorithm import BaseAlgorithm
from qc2.algorithms.utils.mappers import FermionicToQubitMapper
from qc2.algorithms.utils.active_space import ActiveSpace
from qiskit_nature.second_q.mappers import QubitMapper
from qc2.algorithms.algorithms_results import QPEResults

class QPEBase(BaseAlgorithm):
    def __init__(self, 
                 qc2data=None, 
                 active_space=None, 
                 mapper=None, 
                 sampler=None, 
                 reference_state=None,  
                 verbose=0):
        
        self.qc2data = qc2data
        self.format = "qiskit"
        self.verbose = verbose
        self.solver = None 

        # init active space and mapper
        self.active_space = (
            ActiveSpace((2, 2), 2) if active_space is None else active_space
        )

        self.mapper = (
            FermionicToQubitMapper.from_string('parity')()
            if mapper is None
            else FermionicToQubitMapper.from_string(mapper)()
        )

        self.reference_state = (
            self._get_default_reference(self.active_space, self.mapper)
            if reference_state is None
            else reference_state
        )

        self.sampler = Sampler() if sampler is None else sampler


    @staticmethod
    def _get_default_reference(
        active_space: ActiveSpace, mapper: QubitMapper
    ) -> QuantumCircuit:
        """Set up the default reference state circuit based on Hartree Fock.

        Args:
            active_space (ActiveSpace): description of the active space.
            mapper (mapper): mapper class instance.

        Returns:
            QuantumCircuit: Hartree-Fock circuit as the reference state.
        """
        return HartreeFock(
            active_space.num_active_spatial_orbitals,
            active_space.num_active_electrons,
            mapper,
        )
    
    def _init_qubit_hamiltonian(self):
        if self.qc2data is None:
            raise ValueError("qc2data attribute set incorrectly in VQE.")

        self.e_core, self.qubit_op = self.qc2data.get_qubit_hamiltonian(
            self.active_space.num_active_electrons,
            self.active_space.num_active_spatial_orbitals,
            self.mapper,
            format=self.format,
        )
    @staticmethod
    def _phase_to_energy(phase: float) -> float:
        """
        Convert a phase from 0 to 1 to an energy from -pi to pi.

        Args:
            phase (float): The phase to convert.

        Returns:
            float: The energy corresponding to the given phase.
        """
        return (phase - 1) * 2*np.pi

    def run(self) -> QPEResults: 
        """
        Executes the Quantum Phase Estimation (QPE) algorithm to estimate the energy
        of the electronic ground state of a molecule.

        Initializes the qubit Hamiltonian, constructs the unitary matrix, runs the
        QPE algorithm, and calculates the phase and energy. The results are
        encapsulated in a `QPEResults` object.

        Returns:
            QPEResults: An instance of the `QPEResults` class containing the optimal
            energy, eigenvalue, and phase obtained from the QPE algorithm.

        **Example**

        >>> from ase.build import molecule
        >>> from qc2.ase import PySCF
        >>> from qc2.data import qc2Data
        >>> from qc2.algorithms.qiskit import QPE
        >>> from qc2.algorithms.utils import ActiveSpace
        >>>
        >>> mol = molecule('H2O')
        >>>
        >>> hdf5_file = 'h2o.hdf5'
        >>> qc2data = qc2Data(hdf5_file, mol, schema='qcschema')
        >>> qc2data.molecule.calc = PySCF()
        >>> qc2data.run()
        >>> qc2data.algorithm = QPE(
        ...     active_space=ActiveSpace(
        ...         num_active_electrons=(2, 2),
        ...         num_active_spatial_orbitals=4
        ...     ),
        ...     mapper='parity',
        ...     num_evaluation_qubits=39
        ... )
        >>> results = qc2data.algorithm.run()

        """
         # create Hamiltonian
        self._init_qubit_hamiltonian()

        # create the unitary matrix from the qubit operator
        unitary = UnitaryGate(expm(1j*self.qubit_op.to_matrix()))

        # run QPE algorithm  
        qiskit_res = self.solver.estimate(unitary, self.reference_state)

        # get the energy
        energy = self._phase_to_energy(qiskit_res.phase)

        # instantiate VQEResults
        results = QPEResults()
        results.optimal_energy = energy + self.e_core
        results.eigenvalue = energy
        results.phase = qiskit_res.phase

        print(f"=== QISKIT {self.__class__.__name__} RESULTS ===")
        print("* Electronic ground state "
              f"energy (Hartree): {results.eigenvalue}")
        print(f"* Inactive core energy (Hartree): {self.e_core}")
        print(">>> Total ground state "
              f"energy (Hartree): {results.optimal_energy}\n")

        return results

class QPE(QPEBase):
    def __init__(self, 
                 qc2data=None, 
                 num_evaluation_qubits=None,
                 active_space=None, 
                 mapper=None, 
                 sampler=None, 
                 reference_state=None,  
                 verbose=0):
        super().__init__(qc2data, active_space, mapper, sampler, reference_state, verbose)
        self.num_evaluation_qubits = num_evaluation_qubits
        self.solver = PhaseEstimation(self.num_evaluation_qubits, self.sampler)

class IQPE(QPEBase):
    def __init__(self, 
                 qc2data=None, 
                 num_iterations=None,
                 active_space=None, 
                 mapper=None, 
                 sampler=None, 
                 reference_state=None,  
                 verbose=0):
        super().__init__(qc2data, active_space, mapper, sampler, reference_state, verbose)
        self.num_iterations = num_iterations
        self.solver = IterativePhaseEstimation(self.num_iterations, self.sampler)
