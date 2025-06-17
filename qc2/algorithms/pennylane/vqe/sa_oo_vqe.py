"""Module defining SA_OO-VQE algorithm for PennyLane."""
from typing import List,Tuple, Callable
import itertools as itt
from pennylane import numpy as np
import pennylane as qml
from qc2.algorithms.pennylane.vqe.vqe import VQE
from qc2.algorithms.utils.orbital_optimization import OrbitalOptimization
from qc2.algorithms.algorithms_results import SAOOVQEResults
from qc2.ansatz.pennylane.state_resolution import state_resolution_initializer
from qc2.pennylane.convert import _qiskit_nature_to_pennylane
from qiskit_nature.second_q.operators import FermionicOp
from qc2.ansatz.pennylane.generate_ansatz import generate_state_resolution_ansatz

class SA_OO_VQE(VQE):
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
        state_weights=None,
        init_circuit_params=None,
        init_orbital_params=None,
        freeze_active=False,
        state_resolution=True,
        max_iterations=50,
        conv_tol=1e-7,
        verbose=0
    ):
        """Initializes the SA-OO-VQE class.

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
            state_weights (List): List of state weights. Defaults to [0.5, 0.5].
            init_circuit_params (List): List of VQE circuit parameters.
                Defaults to a list with entries of zero.
            init_orbital_params (List): List of orbital optimization
                parameters. Defaults to a list with entries of zero.
            freeze_active (bool): If True, freezes the active
                space during optimization.
            state_resolution (bool): If True, uses state resolution.
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
        >>> qc2data.algorithm = SA_OO_VQE(
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
            init_circuit_params,
            max_iterations,
            conv_tol,
            verbose
        )
        self.freeze_active = freeze_active
        self.orbital_params = init_orbital_params
        self.circuit_params = self.params
        self.oo_problem = None
        self.max_iterations = max_iterations
        self.conv_tol = conv_tol
        self.state_resolution = state_resolution

        # create the ansatz
        self.ansatz = (self._get_default_ansatzes(self.qubits, self.electrons) 
            if ansatz is None 
            else ansatz 
        )

        # create the weights
        self.state_weights = ([0.5, 0.5] 
                              if state_weights is None 
                              else state_weights
        )

        # sanity check
        assert len(self.state_weights) == len(self.ansatz), (
            "Number of ansatzes and state weights must be equal."	
        )
        
    def run(self, *args, **kwargs) -> SAOOVQEResults:
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

        # init the result class
        results = SAOOVQEResults(state_weights=self.state_weights,
                                 energy=[],
                                 circuit_parameters=[],
                                 orbital_parameters=[])

        # get initial energy from initial circuit params
        energy_init = self._get_energy_from_parameters(
            theta, kappa, *args, **kwargs
        )

        # update lists with intermediate data
        results.update(theta, kappa, energy_init)

        # initial values of the optimization
        self._print_iteration_information(0, energy_init, self.verbose)

        for n in range(self.max_iterations):
            # optimize circuit parameters with fixed kappa
            theta, _ = self._circuit_optimization(
                theta, kappa, *args, **kwargs
            )

            # optimize orbital parameters with fixed theta from previous run
            kappa, _ = self._orbital_optimization(
                theta, kappa, *args, **kwargs
            )

            # calculate final energy with all optimized parameters
            energy = self._get_energy_from_parameters(
                theta, kappa, *args, **kwargs
            )

            # update lists with intermediate data
            results.update(theta, kappa, energy)

            # print opt status
            self._print_iteration_information(n, energy, self.verbose)
                
            if n > 1:
                if self._converged(results.energy):
                    self._print_converged_iteration_information(results)
                    break

        # in case of non-convergence
        else:
            raise RuntimeError(
                "Optimization did not converge within the maximum iterations."
                " Consider increasing 'max_iterations' attribute or"
                " setting a different 'optimizer'."
            )

        if self.state_resolution:
            print(">>> Optimizing phase parameter...")
            initial_phase  = 0.0
            phase, energy = self._phase_optimization(initial_phase, theta, kappa, *args, **kwargs)
            results.optimal_phase = phase
            results.update(theta, kappa, [energy, None])
            self._print_converged_iteration_information(results)

        return results
    
    @staticmethod
    def _get_default_init_orbital_params(n_kappa: List) -> List:
        """Set up the init orbital rotation parameters.

        Args:
            n_kappa (List): number of orbital rotation parameters.

        Returns:
            List : List of params values
        """
        return [0.0] * n_kappa

    @staticmethod
    def _get_default_ansatzes(
        ansatz: str| None,
        qubits: int, 
        electrons: int, 
    ) -> List[Callable]:
        """Create the default ansatz function for the VQE circuit.

        Args:
            qubits (int): Number of qubits in the circuit.
            electrons (int): Number of electrons in the system.

        Returns:
            Callable: Function that applies the UCCSD ansatz.
        """
        return [generate_state_resolution_ansatz(qubits, electrons, ansatz, 0.0), 
                generate_state_resolution_ansatz(qubits, electrons, ansatz, np.pi/2)]
    
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
        energy = []
        mo_coeff_a, mo_coeff_b = self.oo_problem.get_transformed_mos(kappa)
        for qc in self.ansatz:
            one_rdm, two_rdm = self._get_rdms(qc, theta, *args, **kwargs)
            energy.append(self.oo_problem.get_energy_from_mo_coeffs(
                mo_coeff_a, mo_coeff_b, one_rdm, two_rdm)
                )
        return energy

    @staticmethod
    def _print_iteration_information(n: int , energy: List[float], verbose) -> None:
        """
        Prints the iteration number and corresponding energy in Hartree.

        Args:
            n (int): The current iteration number.
            energy (float): The energy value associated with the current iteration.
        """
        if verbose is not None:
            print(f"iter = {n+1:03}")
            for ie, e in enumerate(energy):
                print(f"\t energy_{ie} = {e:.12f} Ha")

    def _print_converged_iteration_information(self, results) -> None:
        """
        Prints the final optimization results, including the total ground state energy.

        Args:
            results: An object containing the optimization results, including the
                    optimal energy.

        """
        if self.verbose is not None:
            print("optimization finished.\n")
            print(f"=== PENNYLANE {self.__class__.__name__} RESULTS ===")
            print("* Total ground state "
                    f"energy (Hartree): {results.optimal_energy:.12f}")

    def _converged(self, energy: List) -> bool:
        """
        Checks if the difference between the current and previous energy is smaller
        than the set convergence tolerance.

        Args:
            energy (List): The list of energy values.

        Returns:
            bool: True if the difference between the current and previous energy is
            smaller than the set convergence tolerance, False otherwise.
        """
        return abs(energy[-1][0] - energy[-2][0]) < self.conv_tol

    def _orbital_optimization(self, theta, kappa, *args, **kwargs):
        """Optimize orbital parameters with fixed circuit parameters.

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
                Optimized orbital parameters and associated energy.
        """
        nparam = len(kappa)
        out = [0.0] * nparam
        total_cost = 0.0
        for ansatz, weight in zip(self.ansatz, self.state_weights):
            rdm1, rdm2 = self._get_rdms(ansatz, theta, *args, **kwargs)
            new_kappas, cost = self.oo_problem.orbital_optimization(rdm1, rdm2, kappa)
            total_cost += weight * cost
            out = [out[i] + weight * new_kappas[i] for i in range(nparam)]
        return out, total_cost
    
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
        circuits = [VQE._build_circuit(
            self.device,
            self.qubits,
            ansatz,
            qubit_op,
            *args, **kwargs
        ) for ansatz in self.ansatz]

        # Optimize the circuit parameters and compute the energy
        circ_params = theta
        for n in range(self.max_iterations):
            corr_energy = 0
            for qc, weight in zip(circuits, self.state_weights):
                circ_params, state_corr_energy = self.optimizer.step_and_cost(
                    qc, circ_params
                )
                corr_energy += weight * state_corr_energy
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

    def _phase_optimization(self, phi: float, theta: List, kappa: List, *args, **kwargs) -> Tuple[float, float]:
        """Optimize phase parameters with fixed circuit and orbital parameters.

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
            Tuple[float, float]:
                Optimized phase parameter and associated energy.
        """

        # Generate single and double excitations
        singles, doubles = qml.qchem.excitations(self.electrons, self.qubits)
        s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)
        (core_energy, qubit_op) = self.oo_problem.get_transformed_qubit_hamiltonian(kappa)
        device = qml.device(self.device, wires=self.qubits)

        @qml.qnode(device, *args, **kwargs)
        def phase_optimization_circuit(phase):
            state_resolution_initializer(self.electrons//2, self.electrons//2, phase)
            qml.UCCSD(
                theta, wires=range(self.qubits), 
                s_wires=s_wires, d_wires=d_wires, 
                init_state=np.zeros(self.qubits).astype(int)
            )
            return qml.expval(qubit_op)
        
        # init the container
        energy_l = []
        phase_l = []

        # Optimize the circuit parameters and compute the energy
        circ_params = phi
        for n in range(self.max_iterations):
            circ_params, state_corr_energy = self.optimizer.step_and_cost(
                phase_optimization_circuit, circ_params
            )
            energy = state_corr_energy + core_energy
            energy_l.append(energy)
            phase_l.append(circ_params)

            if n > 1:
                if abs(energy_l[-1] - energy_l[-2]) < self.conv_tol:
                    phase_optimized = phase_l[-1]
                    energy_optimized = energy_l[-1]
                    break
        # in case of non-convergence
        else:
            raise RuntimeError(
                "Circuit optimization step did not converge."
                " Consider increasing 'max_iterations' attribute or"
                " setting a different 'optimizer'."
            )

        return phase_optimized, energy_optimized

    def _get_rdms(
            self,
            ansatz, 
            theta: List,
            sum_spin=True,
            *args, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates 1- and 2-electron reduced density matrices (RDMs).

        Args:
            ansatz (): ansatz circuit.
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
                ansatz,
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