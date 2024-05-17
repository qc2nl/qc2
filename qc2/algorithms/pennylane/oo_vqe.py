"""Module defining oo-VQE algorithm for PennyLane."""
from typing import List, Tuple
import itertools as itt
from pennylane import numpy as np
from qiskit_nature.second_q.operators import FermionicOp
from qc2.algorithms.pennylane.vqe import VQE
from qc2.algorithms.algorithms_results import OOVQEResults
from qc2.algorithms.utils.orbital_optimization import OrbitalOptimization
from qc2.pennylane.convert import _qiskit_nature_to_pennylane


class oo_VQE(VQE):
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
        reference_state=None,
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
        >>> from qc2.algorithms.pennylane import oo_VQE
        >>> from qc2.algorithms.utils import ActiveSpace
        >>>
        >>> mol = molecule('H2O')
        >>>
        >>> hdf5_file = 'h2o.hdf5'
        >>> qc2data = qc2Data(hdf5_file, mol, schema='qcschema')
        >>> qc2data.molecule.calc = PySCF()
        >>> qc2data.run()
        >>> qc2data.algorithm = oo_VQE(
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
            reference_state,
            init_circuit_params,
            max_iterations,
            conv_tol,
            verbose
        )
        self.freeze_active = freeze_active
        self.orbital_params = init_orbital_params
        self.circuit_params = self.params
        self.oo_problem = None

    @staticmethod
    def _get_default_init_orbital_params(n_kappa: List) -> List:
        """Set up the init orbital rotation parameters.

        Args:
            n_kappa (List): number of orbital rotation parameters.

        Returns:
            List : List of params values
        """
        return [0.0] * n_kappa

    def run(self, *args, **kwargs) -> OOVQEResults:
        """Optimizes both the circuit and orbital parameters.

        Args:
            *args:
                - device_args (optional): ``qml.device`` arguments.
                - qnode_args (optional): ``qml.qnode`` arguments.
            **kwargs:
                - device_kwargs (optional): ``qml.device`` keyword arguments.
                - qnode_kwargs (optional): ``qml.qnode`` keyword arguments.

        Returns:
            OOVQEResults:
                An instance of :class:`qc2.algorithms.pennylane.vqe.OOVQEResults`
                class with all oo-VQE info.

        **Example**

        >>> from ase.build import molecule
        >>> from qc2.ase import PySCF
        >>> from qc2.data import qc2Data
        >>> from qc2.algorithms.pennylane import oo_VQE
        >>> from qc2.algorithms.utils import ActiveSpace
        >>>
        >>> mol = molecule('H2O')
        >>>
        >>> hdf5_file = 'h2o.hdf5'
        >>> qc2data = qc2Data(hdf5_file, mol, schema='qcschema')
        >>> qc2data.molecule.calc = PySCF()
        >>> qc2data.run()
        >>> qc2data.algorithm = oo_VQE(
        ...     active_space=ActiveSpace(
        ...         num_active_electrons=(2, 2),
        ...         num_active_spatial_orbitals=4
        ...     ),
        ...     optimizer=qml.GradientDescentOptimizer(stepsize=0.5),
        ...     device="default.qubit"
        ... )
        >>> results = qc2data.algorithm.run()
        """
        print(">>> Optimizing circuit and orbital parameters...")

        # instantiate oo class
        self.oo_problem = OrbitalOptimization(
            self.qc2data,
            self.active_space,
            self.freeze_active,
            self.mapper,
            "pennylane"
        )

        # set initial parameters
        self.orbital_params = (
            self._get_default_init_orbital_params(self.oo_problem.n_kappa)
            if self.orbital_params is None
            else self.orbital_params
        )

        # set initial circuit (theta) and orbital rotation (kappa) parameters
        theta = self.circuit_params
        kappa = self.orbital_params

        # create lists to save intermediate energy, circuit and orbital params
        energy_l = []
        theta_l = []
        kappa_l = []

        # get initial energy from initial circuit params
        energy_init = self._get_energy_from_parameters(
            theta, kappa, *args, **kwargs
        )
        if self.verbose is not None:
            print(f"iter = 000, energy = {energy_init:.12f} Ha")
            energy_l.append(energy_init)

        for n in range(self.max_iterations):
            # optimize circuit parameters with fixed kappa
            theta, _ = self._circuit_optimization(
                theta, kappa, *args, **kwargs
            )

            # optimize orbital parameters with fixed theta from previous run
            rdm1, rdm2 = self._get_rdms(theta, *args, **kwargs)
            kappa, _ = self.oo_problem.orbital_optimization(rdm1, rdm2, kappa)

            # calculate final energy with all optimized parameters
            energy = self._get_energy_from_parameters(
                theta, kappa, *args, **kwargs
            )

            # update lists with intermediate data
            theta_l.append(theta)
            kappa_l.append(kappa)
            energy_l.append(energy)

            if self.verbose is not None:
                print(f"iter = {n+1:03}, energy = {energy:.12f} Ha")
            if n > 1:
                if abs(energy_l[-1] - energy_l[-2]) < self.conv_tol:
                    # instantiate OOVQEResults
                    results = OOVQEResults()
                    results.optimizer_evals = n
                    results.optimal_energy = energy_l[-1]
                    results.optimal_circuit_params = theta_l[-1]
                    results.optimal_orbital_params = kappa_l[-1]
                    results.energy = energy_l
                    results.circuit_parameters = theta_l
                    results.orbital_parameters = kappa_l

                    if self.verbose is not None:
                        print("optimization finished.\n")
                        print("=== PENNYLANE oo-VQE RESULTS ===")
                        print("* Total ground state "
                              f"energy (Hartree): {results.optimal_energy:.12f}")
                    break
        # in case of non-convergence
        else:
            raise RuntimeError(
                "Optimization did not converge within the maximum iterations."
                " Consider increasing 'max_iterations' attribute or"
                " setting a different 'optimizer'."
            )

        return results

    def _circuit_optimization(
            self,
            theta: List,
            kappa: List,
            *args, **kwargs
    ) -> Tuple[List, float]:
        """Get total energy and best circuit parameters for a given kappa.

        Args:
            theta (List): List with circuit variational parameters.
            kappa (List): List with orbital rotation parameters.
            *args:
                - device_args (optional): ``qml.device`` arguments.
                - qnode_args (optional): ``qml.qnode`` arguments.
            **kwargs:
                - device_kwargs (optional): ``qml.device`` keyword arguments.
                - qnode_kwargs (optional): ``qml.qnode`` keyword arguments.

        Returns:
            Tuple[List, float]:
                Optimized circuit parameters and associated energy.
        """
        energy_l = []
        theta_l = []

        # get qubit Hamiltonian for given orbital rotation parameters
        (core_energy,
         qubit_op) = self.oo_problem.get_transformed_qubit_hamiltonian(kappa)

        # build up the pennylane circuit
        circuit = VQE._build_circuit(
            self.device,
            self.qubits,
            self.ansatz,
            qubit_op,
            *args, **kwargs
        )

        # Optimize the circuit parameters and compute the energy
        circ_params = theta
        for n in range(self.max_iterations):
            circ_params, corr_energy = self.optimizer.step_and_cost(
                circuit, circ_params
            )
            energy = corr_energy + core_energy
            energy_l.append(energy)
            theta_l.append(circ_params)

            if n > 1:
                if abs(energy_l[-1] - energy_l[-2]) < self.conv_tol:
                    theta_optimized = theta_l[-1]
                    energy_optimized = energy_l[-1]
                    break
        # in case of non-convergence
        else:
            raise RuntimeError(
                "Circuit optimization step did not converge."
                " Consider increasing 'max_iterations' attribute or"
                " setting a different 'optimizer'."
            )

        return theta_optimized, energy_optimized

    def _get_energy_from_parameters(
            self,
            theta: List,
            kappa: List,
            *args, **kwargs
    ) -> float:
        """Calculates total energy given circuit and orbital parameters.

        Args:
            theta (List): List with circuit variational parameters.
            kappa (List): List with orbital rotation parameters.
            *args:
                - device_args (optional): ``qml.device`` arguments.
                - qnode_args (optional): ``qml.qnode`` arguments.
            **kwargs:
                - device_kwargs (optional): ``qml.device`` keyword arguments.
                - qnode_kwargs (optional): ``qml.qnode`` keyword arguments.

        Returns:
            float:
                Total ground-state energy for a given circuit
                and orbital parameters.
        """
        mo_coeff_a, mo_coeff_b = self.oo_problem.get_transformed_mos(kappa)
        one_rdm, two_rdm = self._get_rdms(theta, *args, **kwargs)
        return self.oo_problem.get_energy_from_mo_coeffs(
            mo_coeff_a, mo_coeff_b, one_rdm, two_rdm
        )

    def _get_rdms(
            self,
            theta: List,
            sum_spin=True,
            *args, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates 1- and 2-electron reduced density matrices (RDMs).

        Args:
            theta (List): circuit parameters at which
                RDMs are calculated.
            sum_spin (bool): If True, the spin-summed 1-RDM and 2-RDM will be
                returned. If False, the full 1-RDM and 2-RDM will be returned.
                Defaults to True.
            *args:
                - device_args (optional): ``qml.device`` arguments.
                - qnode_args (optional): ``qml.qnode`` arguments.
            **kwargs:
                - device_kwargs (optional): ``qml.device`` keyword arguments.
                - qnode_kwargs (optional): ``qml.qnode`` keyword arguments.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                1- and 2-RDMs.
        """
        if len(theta) != len(self.params):
            raise ValueError("Incorrect dimension for amplitude list.")

        # initialize the RDM arrays
        n_mol_orbitals = self.active_space.num_active_spatial_orbitals
        n_spin_orbitals = self.active_space.num_active_spatial_orbitals * 2
        rdm1_spin = np.zeros((n_spin_orbitals,) * 2, dtype=complex)
        rdm2_spin = np.zeros((n_spin_orbitals,) * 4, dtype=complex)

        # get the fermionic hamiltonian
        _, _, fermionic_op = self.qc2data.get_fermionic_hamiltonian(
            self.active_space.num_active_electrons,
            self.active_space.num_active_spatial_orbitals
        )

        # run over the hamiltonian terms and calculate expectation values
        for key, _ in fermionic_op.terms():
            # assign indices depending on one- or two-body term
            length = len(key)
            if length == 2:
                iele, jele = (int(ele[1]) for ele in tuple(key[0:2]))
            elif length == 4:
                iele, jele, kele, lele = (int(ele[1]) for ele in tuple(key[0:4]))

            # get fermionic and qubit representation of each term
            fermionic_ham_temp = FermionicOp.from_terms([(key, 1.0)])
            qubit_ham_temp_qiskit = self.mapper.map(
                fermionic_ham_temp, register_length=n_spin_orbitals
            )

            # convert qiskit `SparsePauliOp` to pennylane `Operator`
            coefficients, operators = _qiskit_nature_to_pennylane(qubit_ham_temp_qiskit)
            qubit_ham_temp = sum(c * op for c, op in zip(coefficients, operators))

            # calculate expectation values
            circuit = VQE._build_circuit(
                self.device,
                self.qubits,
                self.ansatz,
                qubit_ham_temp,
                *args, **kwargs
            )
            energy_temp = circuit(theta)

            # put the values in np arrays (differentiate 1- and 2-RDM)
            if length == 2:
                rdm1_spin[iele, jele] = energy_temp
            elif length == 4:
                rdm2_spin[iele, lele, jele, kele] = energy_temp

        if sum_spin:
            # get spin-free RDMs
            rdm1_np = np.zeros((n_mol_orbitals,) * 2, dtype=np.complex128)
            rdm2_np = np.zeros((n_mol_orbitals,) * 4, dtype=np.complex128)

            # construct spin-summed 1-RDM
            mod = n_spin_orbitals // 2
            for i, j in itt.product(range(n_spin_orbitals), repeat=2):
                rdm1_np[i % mod, j % mod] += rdm1_spin[i, j]

            # construct spin-summed 2-RDM
            for i, j, k, l in itt.product(range(n_spin_orbitals), repeat=4):
                rdm2_np[
                    i % mod, j % mod, k % mod, l % mod
                ] += rdm2_spin[i, j, k, l]

            return rdm1_np, rdm2_np

        return rdm1_spin, rdm2_spin
