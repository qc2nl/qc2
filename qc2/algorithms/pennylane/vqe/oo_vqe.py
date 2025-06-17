"""Module defining oo-VQE algorithm for PennyLane."""
from typing import List, Tuple
from typing import Callable
import pennylane as qml
import itertools as itt
from pennylane import numpy as np
from qiskit_nature.second_q.operators import FermionicOp
from qc2.algorithms.pennylane.vqe.sa_oo_vqe import SA_OO_VQE
from qc2.algorithms.utils.orbital_optimization import OrbitalOptimization
from qc2.pennylane.convert import _qiskit_nature_to_pennylane
from qc2.ansatz.pennylane.generate_ansatz import generate_ansatz

class OO_VQE(SA_OO_VQE):
    """Main class for orbital-optimized VQE with PennyLane.

    This class is responsible for optimizing both circuit and orbital
    parameters of simple molecules. Analytic first and second derivatives are
    considered in the orbital optimization part.

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
            Defaults to 1e-7.
        verbose (int): Verbosity level. Defaults to 0.
    """
    def __init__(
        self,
        qc2data=None,
        ansatz=None,
        active_space=None,
        mapper=None,
        device=None,
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
        >>> from qc2.algorithms.pennylane import OO_VQE
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
        ...     mapper="jw",
        ...     optimizer=qml.GradientDescentOptimizer(stepsize=0.5),
        ...     device="default.qubit"
        ... )
        >>> results = qc2data.algorithm.run()
        """
        super().__init__(
            qc2data,
            ansatz,
            active_space,
            mapper,
            device,
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
    def _get_default_ansatzes(ansatz: str| None,
        qubits: int, electrons: int
    ) -> Callable:
        """Create the default ansatz function for the VQE circuit.

        Args:
            ansatz (str| None): Type of ansatz to use.
            qubits (int): Number of qubits in the circuit.
            electrons (int): Number of electrons in the system.

        Returns:
            Callable: Function that applies the UCCSD ansatz.
        """
        return [generate_ansatz(qubits, electrons, ansatz)]
