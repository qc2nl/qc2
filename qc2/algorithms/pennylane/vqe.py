"""Module defining VQE algorithm for PennyLane."""
from typing import Callable
import pennylane as qml
from pennylane import numpy as np
from pennylane import QNode
from pennylane.operation import Operator
from qc2.algorithms.utils.active_space import ActiveSpace
from qc2.algorithms.utils.mappers import FermionicToQubitMapper
from qc2.algorithms.base.vqe_base import VQEBASE
from qc2.algorithms.algorithms_results import VQEResults


class VQE(VQEBASE):
    """
    Main class for the VQE algorithm with PennyLane.

    This class initializes and executes the VQE algorithm using specified
    quantum components like ansatz, optimizer, and estimator.

    Attributes:
        ansatz (Callable): The ansatz for the VQE algorithm.
            Defaults to ``qml.UCCSD``.
        active_space (ActiveSpace): Instance of
            :class:`~qc2.algorithm.utils.activate_space.ActiveSpace`.
            Defaults to ``ActiveSpace((2, 2), 2)``.
        mapper (QubitMapper): Strategy for fermionic-to-qubit mapping.
            Defaults to ``JordanWignerMapper``.
        device (qml.device): Device for estimating the expectation value.
            Defaults to ``default.qubit``.
        optimizer (qml.optimizer): Optimization routine for circuit
            variational parameters. Defaults
            to ``qml.GradientDescentOptimizer``.
        reference_state (qml.ref_state): Reference state for the VQE
            algorithm. Defaults to ``qml.qchem.hf_state``.
        params (List): List of initial VQE circuit parameters.
            Defaults to a list with entries of zero.
        max_iterations (int): Maximum number of iterations for the combined
            circuit-orbitals parameters optimization. Defaults to 50.
        conv_tol (float): Convergence tolerance for the optimization.
            Defaults to 1e-7.
        verbose (int): Verbosity level. Defaults to 0.
        circuit (QNode): Quantum circuit generated for the VQE algorithm.
    """

    def __init__(
        self,
        qc2data=None,
        ansatz=None,
        active_space=None,
        mapper=None,
        device=None,
        optimizer=None,
        reference_state=None,
        init_params=None,
        max_iterations=50,
        conv_tol=1e-7,
        verbose=0
    ):
        """Initializes the VQE class.

        Args:
            qc2data (qc2Data): An instance of :class:`~qc2.data.data.qc2Data`.
            ansatz (Callable): The ansatz for the VQE algorithm.
                Defaults to ``qml.UCCSD``.
            active_space (ActiveSpace): Instance of
                :class:`~qc2.algorithm.utils.active_space.ActiveSpace`.
                Defaults to ``ActiveSpace((2, 2), 2)``.
            mapper (str): Strategy for fermionic-to-qubit mapping.
                Common options are ``jw`` for ``JordanWignerMapper``
                or "bk" for ``BravyiKitaevMapper``. Defaults to ``jw``.
            device (qml.device): Device for estimating the expectation value.
                Defaults to ``default.qubit``.
            optimizer (qml.optimizer): Optimization routine for circuit
                variational parameters. Defaults
                to ``qml.GradientDescentOptimizer``.
            reference_state (qml.ref_state): Reference state for the VQE
                algorithm. Defaults to ``qml.qchem.hf_state``.
            init_params (List): List of VQE circuit parameters.
                Defaults to a list with entries of zero.
            max_iterations (int): Maximum number of iterations for the combined
                circuit-orbitals parameters optimization. Defaults to 50.
            conv_tol (float): Convergence tolerance for the optimization.
                Defaults to 1e-7.
            verbose (int): Verbosity level. Defaults to 0.

        **Example**

        >>> from ase.build import molecule
        >>> from qc2.ase import PySCF
        >>> from qc2.data import qc2Data
        >>> from qc2.algorithms.pennylane import VQE
        >>> from qc2.algorithms.utils import ActiveSpace
        >>>
        >>> mol = molecule('H2O')
        >>>
        >>> hdf5_file = 'h2o.hdf5'
        >>> qc2data = qc2Data(hdf5_file, mol, schema='qcschema')
        >>> qc2data.molecule.calc = PySCF()
        >>> qc2data.run()
        >>> qc2data.algorithm = VQE(
        ...     active_space=ActiveSpace(
        ...         num_active_electrons=(2, 2),
        ...         num_active_spatial_orbitals=4
        ...     ),
        ...     mapper="jw",
        ...     optimizer=qml.GradientDescentOptimizer(stepsize=0.5),
        ...     device="default.qubit"
        ... )
        >>> results = qc2data.algorithm.run()
        """
        super().__init__(qc2data, "pennylane")

        # init active space and mapper
        self.active_space = (
            ActiveSpace((2, 2), 2) if active_space is None else active_space
        )

        # init circuit
        self.device = "default.qubit" if device is None else device
        self.mapper = (
            FermionicToQubitMapper.from_string('jw')()
            if mapper is None
            else FermionicToQubitMapper.from_string(mapper)()
        )
        self.qubits = 2 * self.active_space.num_active_spatial_orbitals
        self.electrons = sum(self.active_space.num_active_electrons)
        self.optimizer = (
            qml.GradientDescentOptimizer(stepsize=0.5)
            if optimizer is None
            else optimizer
        )
        self.reference_state = (
            self._get_default_reference(self.qubits, self.electrons)
            if reference_state is None
            else reference_state
        )
        self.ansatz = (
            self._get_default_ansatz(
                self.qubits, self.electrons, self.reference_state
            )
            if ansatz is None
            else ansatz
        )
        self.params = (
            self._get_default_init_params(self.qubits, self.electrons)
            if init_params is None
            else init_params
        )

        # init algorithm-specific attributes
        self.max_iterations = max_iterations
        self.conv_tol = conv_tol
        self.verbose = verbose
        self.circuit = None

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

    @staticmethod
    def _get_default_ansatz(
        qubits: int, electrons: int, reference_state: np.ndarray
    ) -> Callable:
        """Create the default ansatz function for the VQE circuit.

        Args:
            qubits (int): Number of qubits in the circuit.
            electrons (int): Number of electrons in the system.
            reference_state (np.ndarray): Reference state for the ansatz.

        Returns:
            Callable: Function that applies the UCCSD ansatz.
        """
        # Generate single and double excitations
        singles, doubles = qml.qchem.excitations(electrons, qubits)

        # Map excitations to the wires the UCCSD circuit will act on
        s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)

        # Return a function that applies the UCCSD ansatz
        def ansatz(params):
            qml.UCCSD(
                params, wires=range(qubits), s_wires=s_wires,
                d_wires=d_wires, init_state=reference_state
            )
        return ansatz

    @staticmethod
    def _get_default_init_params(qubits: int, electrons: int) -> np.ndarray:
        """Generate default initial parameters for the ansatz.

        Args:
            qubits (int): Number of qubits in the circuit.
            electrons (int): Number of electrons in the system.

        Returns:
            np.ndarray: Array of initial parameter values.
        """
        # Generate single and double excitations
        singles, doubles = qml.qchem.excitations(electrons, qubits)
        return np.zeros(len(singles) + len(doubles))

    @staticmethod
    def _build_circuit(
        dev: str,
        qubits: int,
        ansatz: Callable,
        qubit_op: Operator,
        device_args=None,
        device_kwargs=None,
        qnode_args=None,
        qnode_kwargs=None
    ) -> QNode:
        """Builds and return PennyLane QNode.

        Args:
            dev (str): PennyLane quantum device.
            qubits (int): Number of qubits in the circuit.
            ansatz (Callable): Ansatz function for the circuit.
            qubit_op (Operator): Qubit operator for the Hamiltonian.
            device_args (list, optional): Additional arguments for the quantum
                device. Defaults to None.
            device_kwargs (dict, optional): Additional keyword arguments for
                the quantum device. Defaults to None.
            qnode_args (list, optional): Additional arguments for the QNode.
                Defaults to None.
            qnode_kwargs (dict, optional): Additional keyword arguments for
                the QNode. Defaults to None.

        Returns:
            QNode: PennyLane qnode with built-in ansatz.
        """
        # Set default values if None
        device_args = device_args if device_args is not None else []
        device_kwargs = device_kwargs if device_kwargs is not None else {}
        qnode_args = qnode_args if qnode_args is not None else []
        qnode_kwargs = qnode_kwargs if qnode_kwargs is not None else {}

        # Define the device
        device = qml.device(dev, wires=qubits, *device_args, **device_kwargs)

        # Define the QNode and call the ansatz function within it
        @qml.qnode(device, *qnode_args, **qnode_kwargs)
        def circuit(params):
            ansatz(params)
            return qml.expval(qubit_op)

        return circuit

    def run(self, *args, **kwargs) -> VQEResults:
        """Executes VQE algorithm.

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
        >>> from qc2.algorithms.pennylane import VQE
        >>> from qc2.algorithms.utils import ActiveSpace
        >>>
        >>> mol = molecule('H2O')
        >>>
        >>> hdf5_file = 'h2o.hdf5'
        >>> qc2data = qc2Data(hdf5_file, mol, schema='qcschema')
        >>> qc2data.molecule.calc = PySCF()
        >>> qc2data.run()
        >>> qc2data.algorithm = VQE(
        ...     active_space=ActiveSpace(
        ...         num_active_electrons=(2, 2),
        ...         num_active_spatial_orbitals=4
        ...     ),
        ...     optimizer=qml.GradientDescentOptimizer(stepsize=0.5),
        ...     device="default.qubit"
        ... )
        >>> results = qc2data.algorithm.run()
        """
        print(">>> Optimizing circuit parameters...")

        # create Hamiltonian
        self._init_qubit_hamiltonian()

        # build circuit after building the qubit hamiltonian
        self.circuit = self._build_circuit(
            self.device,
            self.qubits,
            self.ansatz,
            self.qubit_op,
            *args, **kwargs
        )

        # set initial theta parameters
        theta = self.params

        # create lists to save intermediate energy and circuit params
        energy_l = []
        theta_l = []

        # optimize the circuit parameters and compute the energy
        for n in range(self.max_iterations):
            theta, corr_energy = self.optimizer.step_and_cost(
                self.circuit, theta
            )

            # update lists with intermediate data
            energy = corr_energy + self.e_core
            energy_l.append(energy)
            theta_l.append(theta.numpy().tolist())

            if self.verbose is not None:
                if n % 2 == 0:
                    print(f"iter = {n:03}, energy = {energy_l[-1]:.12f} Ha")

            if n > 1:
                if abs(energy_l[-1] - energy_l[-2]) < self.conv_tol:
                    # instantiate VQEResults
                    results = VQEResults()
                    results.optimizer_evals = n
                    results.optimal_energy = energy_l[-1]
                    results.optimal_params = theta_l[-1]
                    results.energy = energy_l
                    results.parameters = theta_l

                    if self.verbose is not None:
                        print("optimization finished.\n")
                        print("=== PENNYLANE VQE RESULTS ===")
                        print("* Electronic ground state "
                              f"energy (Hartree): {corr_energy:.12f}")
                        print("* Inactive core "
                              f"energy (Hartree): {self.e_core:.12f}")
                        print(">>> Total ground state "
                              "energy (Hartree): "
                              f"{results.optimal_energy:.12f}\n")
                    break
        # in case of non-convergence
        else:
            raise RuntimeError(
                "Optimization did not converge within the maximum iterations."
                " Consider increasing 'max_iterations' attribute or"
                " setting a different 'optimizer'."
            )

        return results
