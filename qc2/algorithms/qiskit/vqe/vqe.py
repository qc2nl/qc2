"""Module defining VQE algorithm for Qiskit-Nature."""
from typing import List, Dict, Union, Any
import numpy as np
from qiskit_algorithms.minimum_eigensolvers import VQE as vqe_solver
from qiskit_algorithms.optimizers import SLSQP
from qiskit.primitives import Estimator
from qiskit.circuit import QuantumCircuit
from qc2.algorithms.base.qc2_algorithm_base_class import QC2BaseAlgorithm
from qc2.algorithms.results import VQEResults
from qc2.algorithms.second_q.active_space import ActiveSpace
from qc2.algorithms.qiskit.ansatz.generate_ansatz import generate_ansatz
from qc2.algorithms.qiskit.qubit_mappers.jordan_wigner import JordanWigner
from qc2.qc2_driver import QC2
from qc2.algorithms.qiskit.qubit_mappers.base_mapper import QiskitBaseMapper


class VQE(QC2BaseAlgorithm):
    """
    Main class for the VQE algorithm with Qiskit-Nature.

    This class initializes and executes the VQE algorithm using specified
    quantum components like ansatz, optimizer, and estimator.

    Attributes:
        active_space (ActiveSpace): Describes the active space for quantum
            simulation. Defaults to ``ActiveSpace((2, 2), 2)``.
        mapper (QubitMapper): Strategy for fermionic-to-qubit mapping.
            Defaults to :class:`qiskit.JordanWignerMapper`.
        estimator (BaseEstimator): Method for estimating the
            expectation value. Defaults to :class:`qiskit.Estimator`
        optimizer (qiskit.Optmizer): Optimization routine for circuit
            variational parameters. Defaults to
            :class:`qiskit_algorithms.SLSQP`.
        ansatz (UCC): The ansatz for the VQE algorithm.
            Defaults to :class:`qiskit.UCCSD`.
        params (List): List of initial VQE circuit parameters.
            Defaults to a list with entries of zero.
        verbose (int): Verbosity level. Defaults to 0.
    """

    def __init__(
        self,
        qc2data: QC2 | None = None,
        ansatz: QuantumCircuit | None = None,
        active_space: ActiveSpace | None = None,
        mapper: QiskitBaseMapper | None = None,
        estimator: Estimator | None = None,
        optimizer: Any = None,
        init_params: List | None = None,
        verbose: int = 0
    ):
        """Initializes the VQE class.

        Args:
            qc2data (qc2Data): An instance of :class:`~qc2.qc2_driver.QC2`.
            ansatz (None, str, QuantmumCircuit): The ansatz for the VQE algorithm.
                Defaults to :class:`qiskit.UCCSD`.
            active_space (ActiveSpace): Describes the active space for quantum
                simulation. Defaults to ``ActiveSpace((2, 2), 2)``.
            mapper (str): Strategy for fermionic-to-qubit mapping.
                Common options are ``jw`` for ``JordanWignerMapper``
                or "bk" for ``BravyiKitaevMapper``. Defaults to ``jw``.
            estimator (BaseEstimator): Method for estimating the
                expectation value. Defaults to :class:`qiskit.Estimator`
            optimizer (qiskit.Optmizer): Optimization routine for circuit
                variational parameters. Defaults to
                :class:`qiskit_algorithms.SLSQP`.
            init_params (List): List of VQE circuit parameters.
                Defaults to a list with entries of zero.
            verbose (int): Verbosity level. Defaults to 0.

        **Example**

        >>> from ase.build import molecule
        >>> from qc2.ase import PySCF
        >>> from qc2.qc2_driver import QC2 as qc2Data
        >>> from qc2.algorithms.qiskit import VQE
        >>> from qc2.second_q.active_space import ActiveSpace
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
        ...     optimizer=SLSQP(),
        ...     estimator=Estimator(),
        ... )
        >>> results = qc2data.algorithm.run()
        """

        self.qc2data = qc2data
        self.format = "qiskit"

        # init active space and mapper
        self.active_space = (
            ActiveSpace((2, 2), 2) if active_space is None else active_space
        )

        self.mapper = (
            JordanWigner()
            if mapper is None
            else mapper
        )

        # init circuit
        self.estimator = Estimator() if estimator is None else estimator
        self.optimizer = SLSQP() if optimizer is None else optimizer
        
        self.ansatz = (
            self._get_default_ansatz(
                ansatz, self.active_space, self.mapper
            )
            if not isinstance(ansatz, QuantumCircuit)
            else ansatz
        )
        self.params = (
            np.array(self._get_default_init_params(self.ansatz.num_parameters))
            if init_params is None
            else init_params
        )

        # init algorithm-specific attributes
        self.verbose = verbose

    @staticmethod
    def _get_default_ansatz(
        ansatz: Union[str, None],
        active_space: ActiveSpace,
        mapper: QiskitBaseMapper,
    ) -> QuantumCircuit:
        """Set up the default UCC ansatz from a Hartree Fock reference state.

        Args:
            active_space (ActiveSpace): Description of the active space.
            mapper (QubitMapper): Mapper class instance.

        Returns:
            QuantumCircuit: the ansatz quantum circuit.
        """

        return generate_ansatz(
            num_spatial_orbitals=active_space.num_active_spatial_orbitals,
            num_particles=active_space.num_active_electrons,
            mapper=mapper,
            ansatz_type=ansatz,
            mol=None,
        )

    @staticmethod
    def _get_default_init_params(nparams: List) -> List:
        """Generates a list of initial circuit parameters for the ansatz.

        Args:
            nparams (int): Number of parameters in the ansatz.

        Returns:
            List[float]: List of initial parameter values (all zeros).
        """
        return [0.0] * nparams

    def run(self, *args, **kwargs) -> VQEResults:
        """Executes the VQE algorithm.

        Args:
            *args: Variable length argument list to be passed to
                the :class:`qiskit_algorithm.VQE` class.
            **kwargs: Arbitrary keyword arguments to be passed to
                the :class:`qiskit_algorithm.VQE` class.

        Returns:
            VQEResults:
                An instance of :class:`qc2.algorithms.qiskit.vqe.VQEResults`
                class with all VQE info.

        **Example**

        >>> from ase.build import molecule
        >>> from qc2.ase import PySCF
        >>> from qc2.qc2_driver import QC2 as qc2Data
        >>> from qc2.algorithms.qiskit import VQE
        >>> from qc2.second_q.active_space import ActiveSpace
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
        ...     optimizer=SLSQP(),
        ...     estimator=Estimator(),
        ... )
        >>> results = qc2data.algorithm.run()
        """
        # create Hamiltonian
        self._init_qubit_hamiltonian()

        # create a simple callback to print intermediate results
        intermediate_info = {
            "nfev": [],
            "parameters": [],
            "energy": [],
            "metadata": []
        }

        def callback(
            nfev: int,
            parameters: List,
            energy: float,
            metadata: Dict
        ) -> None:
            intermediate_info["nfev"].append(nfev)
            intermediate_info["parameters"].append(parameters)
            intermediate_info["energy"].append(energy + self.e_core)
            intermediate_info["metadata"].append(metadata)
            if self.verbose is not None:
                if nfev % 2 == 0:
                    print(
                        f"iter = {intermediate_info['nfev'][-1]:03}, "
                        f"energy = {intermediate_info['energy'][-1]:.12f} Ha"
                    )

        # instantiate the solver
        solver = vqe_solver(
            self.estimator,
            self.ansatz,
            self.optimizer,
            callback=callback,
            *args,
            **kwargs
        )
        solver.initial_point = self.params

        # call the minimizer and save final results
        qiskit_res = solver.compute_minimum_eigenvalue(self.qubit_op)

        # instantiate VQEResults
        results = VQEResults()
        results.optimizer_evals = intermediate_info["nfev"][-1]
        results.optimal_energy = intermediate_info["energy"][-1]
        results.optimal_params = intermediate_info["parameters"][-1]
        results.energy = intermediate_info["energy"]
        results.parameters = intermediate_info["parameters"]
        results.metadata = intermediate_info["metadata"]

        print("=== QISKIT VQE RESULTS ===")
        print("* Electronic ground state "
              f"energy (Hartree): {qiskit_res.eigenvalue}")
        print(f"* Inactive core energy (Hartree): {self.e_core}")
        print(">>> Total ground state "
              f"energy (Hartree): {results.optimal_energy}\n")

        return results

