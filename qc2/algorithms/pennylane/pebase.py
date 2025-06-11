"""Module defining QPE algorithm for PennyLane."""
from scipy.linalg import expm
import pennylane as qml
from pennylane import numpy as np
from pennylane import QNode
from pennylane.operation import Operator
from qc2.algorithms.utils.active_space import ActiveSpace
from qc2.algorithms.utils.mappers import FermionicToQubitMapper
from qc2.algorithms.algorithms_results import QPEResults
from qc2.algorithms.base.base_algorithm import BaseAlgorithm

class PEBase(BaseAlgorithm):
    def __init__(
        self,
        qc2data=None,
        active_space=None,
        mapper=None,
        device=None,
        reference_state=None,
        verbose=0
    ):
        
        self.qc2data = qc2data
        self.format = "pennylane"
        self.verbose = verbose
        self.circuit = None
        self.num_evaluation_qubits = None

        # init active space and mapper
        self.active_space = (
            ActiveSpace((2, 2), 2) if active_space is None else active_space
        )

        self.device = "default.qubit" if device is None else device
        self.mapper = (
            FermionicToQubitMapper.from_string('jw')()
            if mapper is None
            else FermionicToQubitMapper.from_string(mapper)()
        )

        self.qubits = 2 * self.active_space.num_active_spatial_orbitals
        self.electrons = sum(self.active_space.num_active_electrons)

        self.reference_state = (
            self._get_default_reference(self.qubits, self.electrons)
            if reference_state is None
            else reference_state
        )

    @staticmethod
    def _get_default_reference(qubits: int, electrons: int) -> np.ndarray:
        """Generate the default reference state for the ansatz.

        Args:
            qubits (int): Number of qubits in the circuit.
            electrons (int): Number of electrons in the system.

        Returns:
            np.ndarray: Reference state vector.
        """
        return qml.qchem.hf_state(electrons, qubits)


    def _init_qubit_hamiltonian(self):
        """
        Initializes the qubit Hamiltonian for the quantum phase estimation algorithm.

        This method retrieves the qubit Hamiltonian representation of the target 
        molecule from the `qc2data` object. It requires prior initialization of 
        `qc2data` with the molecular data.

        Raises:
            ValueError: If `qc2data` is not set correctly.

        Attributes:
            e_core (float): The core energy of the system.
            qubit_op (Operator): The qubit operator representing the Hamiltonian.
        """

        if self.qc2data is None:
            raise ValueError("qc2data attribute set incorrectly in QPE.")

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

    @staticmethod
    def _build_circuit(
        dev: str,
        qubits: int,
        num_estimation_wires: int,
        reference_state: np.ndarray,
        unitary_op: Operator,
        device_args=None,
        device_kwargs=None,
        qnode_args=None,
        qnode_kwargs=None
    ) -> QNode:
        """
        Constructs and returns a PennyLane QNode for quantum phase estimation.

        This method sets up a quantum circuit on a specified quantum device using 
        the provided parameters. It initializes the qubits, applies necessary quantum 
        operations including the unitary operator, and prepares the reference state.

        Args:
            dev (str): Identifier for the PennyLane quantum device to be used.
            qubits (int): Total number of qubits in the circuit.
            num_estimation_wires (int): Number of qubits designated for phase estimation.
            reference_state (np.ndarray): Initial state of the qubits, typically representing 
                the Hartree-Fock state.
            unitary_op (Operator): Operator that defines the unitary evolution in the circuit.
            device_args (list, optional): Additional positional arguments for the quantum 
                device. Defaults to None.
            device_kwargs (dict, optional): Additional keyword arguments for the quantum 
                device. Defaults to None.
            qnode_args (list, optional): Additional positional arguments for the QNode. 
                Defaults to None.
            qnode_kwargs (dict, optional): Additional keyword arguments for the QNode. 
                Defaults to None.

        Returns:
            QNode: The constructed QNode with the specified ansatz for phase estimation.
        """
        raise NotImplementedError('Implement a _build_circuit method.')

    def get_phase(self):
        """
        Retrieves the estimated phase after the QPE algorithm has been run.

        Returns:
            float: The estimated phase.
        """
        raise NotImplementedError('Implement a get_phase method.')

    def run(self, *args, **kwargs) -> QPEResults:
        """Executes QPE algorithm.

        Args:
            *args:
                - device_args (optional): ``qml.device`` arguments.
                - qnode_args (optional): ``qml.qnode`` arguments.
            **kwargs:
                - device_kwargs (optional): ``qml.device`` keyword arguments.
                - qnode_kwargs (optional): ``qml.qnode`` keyword arguments.

        Returns:
            VQEResults:
                An instance of :class:`qc2.algorithms.pennylane.vqe.VQEResults`
                class with all VQE info.

        **Example**

        >>> from ase.build import molecule
        >>> from qc2.ase import PySCF
        >>> from qc2.data import qc2Data
        >>> from qc2.algorithms.pennylane import QPE
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
        ...     num_evaluation_qubits=3, 
        ...     device="default.qubit"
        ... )
        >>> results = qc2data.algorithm.run()
        """
        print(">>> Optimizing circuit parameters...")

        # create Hamiltonian
        self._init_qubit_hamiltonian()

        # create the unitary operator
        unitary_op = qml.exp(self.qubit_op, coeff=1j)

        # build circuit after building the qubit hamiltonian
        self.circuit = self._build_circuit(
            self.device,
            self.qubits,
            self.num_evaluation_qubits,
            self.reference_state,
            unitary_op,
            *args, **kwargs
        )
       
        # extract the phase
        phase = self.get_phase()

        # get the energy
        energy = self._phase_to_energy(phase)

        # instantiate VQEResults
        results = QPEResults()
        results.optimal_energy = energy + self.e_core
        results.eigenvalue = energy
        results.phase = phase

        print(f"=== PENNYLANE {self.__class__.__name__} RESULTS ===")
        print("* Electronic ground state "
              f"energy (Hartree): {results.eigenvalue}")
        print(f"* Inactive core energy (Hartree): {self.e_core}")
        print(">>> Total ground state "
              f"energy (Hartree): {results.optimal_energy}\n")
        
        return results