"""Module defining oo-VQE algorithm for Qiskit-Nature."""
from typing import List, Union
from qiskit.circuit import QuantumCircuit
from qiskit_nature.second_q.mappers import QubitMapper
from qc2.algorithms.qiskit.vqe.sa_oo_vqe import SA_OO_VQE
from qc2.algorithms.utils.active_space import ActiveSpace
from qc2.ansatz.qiskit.generate_ansatz import generate_ansatz

class OO_VQE(SA_OO_VQE):
    """Main class for orbital-optimized VQE with Qiskit-Nature.

    This class extends the VQE class to include orbital optimization. It
    supports customized ansatzes, active space definitions, qubit mapping
    strategies, estimation methods, and optimization routines. Orbital
    optimization is performed alongside VQE parameter optimization using
    analytic first and second derivatives

    Attributes:
        freeze_active (bool): If True, freezes the active
            space during optimization.
        orbital_params (List): List of orbital optimization parameters.
            Defaults to a list with entries of zero.
        circuit_params (List): List of VQE circuit parameters.
            Defaults to a list with entries of zero.
        oo_problem (OrbitalOptimization): An instance of
            :class:`~qc2.algorithms.utils.orbital_optimization.OrbitalOptimization`
            problem class. Defaults to None.
        max_iterations (int): Maximum number of iterations for the combined
            circuit-orbitals parameters optimization. Defaults to 50.
        conv_tol (float): Convergence tolerance for the optimization.
            Defaults to 1e-7
        verbose (int): Verbosity level. Defaults to 0.
    """
    def __init__(
        self,
        qc2data=None,
        ansatz=None,
        active_space=None,
        mapper=None,
        estimator=None,
        optimizer=None,
        init_circuit_params=None,
        init_orbital_params=None,
        freeze_active=False,
        max_iterations=50,
        conv_tol=1e-7,
        verbose=0
    ):
        """Initializes the oo-VQE class.

        Args:
            qc2data (qc2Data): An instance of :class:`~qc2.data.data.qc2Data`.
            ansatz (UCC): The ansatz for the VQE algorithm.
                Defaults to :class:`qiskit.UCCSD`.
            active_space (ActiveSpace): Instance of
                :class:`~qc2.algorithm.utils.active_space.ActiveSpace`.
                Defaults to ``ActiveSpace((2, 2), 2)``.
            mapper (str): Strategy for fermionic-to-qubit mapping.
                Common options are ``jw`` for ``JordanWignerMapper``
                or "bk" for ``BravyiKitaevMapper``. Defaults to ``jw``.
            estimator (BaseEstimator): Method for estimating the
                expectation value. Defaults to :class:`qiskit.Estimator`
            optimizer (qiskit.Optimizer): Optimization routine for circuit
                variational parameters. Defaults to
                :class:`qiskit_algorithms.SLSQP`.
            init_circuit_params (List): List of VQE circuit parameters.
                Defaults to a list with entries of zero.
            init_orbital_params (List): List of orbital optimization
                parameters. Defaults to a list with entries of zero.
            freeze_active (bool): If True, freezes the active
                space during optimization.
            max_iterations (int): Maximum number of iterations for the combined
                circuit-orbitals parameters optimization. Defaults to 50.
            conv_tol (float): Convergence tolerance for the optimization.
                Defaults to 1e-7.
            verbose (int): Verbosity level. Defaults to 0.

        **Example**

        >>> from ase.build import molecule
        >>> from qc2.ase import PySCF
        >>> from qc2.data import qc2Data
        >>> from qc2.algorithms.qiskit import OO_VQE
        >>> from qc2.algorithms.utils import ActiveSpace
        >>>
        >>> mol = molecule('H2O')
        >>>
        >>> hdf5_file = 'h2o.hdf5'
        >>> qc2data = qc2Data(hdf5_file, mol, schema='qcschema')
        >>> qc2data.molecule.calc = PySCF()
        >>> qc2data.run()
        >>> qc2data.algorithm = OO_VQE(
        ...     active_space=ActiveSpace(
        ...         num_active_electrons=(2, 2),
        ...         num_active_spatial_orbitals=4
        ...     ),
        ...     optimizer=SLSQP(),
        ...     estimator=Estimator(),
        ... )
        >>> results = qc2data.algorithm.run()
        """
        super().__init__(
            qc2data,
            ansatz,
            active_space,
            mapper,
            estimator,
            optimizer,
            [1],
            init_circuit_params,
            init_orbital_params,
            freeze_active,
            False,
            max_iterations,
            conv_tol,
            verbose
        )
    
    @staticmethod
    def _get_default_ansatzes(
        ansatz: Union[str, None],
        active_space: ActiveSpace,
        mapper: QubitMapper
    ) -> List[QuantumCircuit]:
        """Set up the default UCC ansatz from a Hartree Fock reference state.

        Args:
            active_space (ActiveSpace): Description of the active space.
            mapper (QubitMapper): Mapper class instance.
            reference_state (QuantumCircuit): Reference state circuit.

        Returns:
            QuantumCircuit: the ansatz quantum circuit.
        """

        return [generate_ansatz(
            num_spatial_orbitals=active_space.num_active_spatial_orbitals,
            num_particles=active_space.num_active_electrons,
            mapper=mapper,
            ansatz_type=ansatz,
            mol_data=None,
            scf=None
        )]
