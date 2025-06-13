"""Module defining SA-OO-VQE algorithm for Qiskit-Nature."""
from typing import List, Union, Tuple
import numpy as np
import itertools as itt
from qiskit.circuit import QuantumCircuit
from functools import partial
from qiskit_nature.second_q.circuit.library import HartreeFock, UCC
from qiskit_nature.second_q.mappers import QubitMapper
from qiskit_nature.second_q.operators import FermionicOp

from qc2.algorithms.qiskit.vqe.vqe import VQE
from qc2.algorithms.algorithms_results import SAOOVQEResults
from qc2.algorithms.utils.active_space import ActiveSpace
from qc2.algorithms.utils.orbital_optimization import OrbitalOptimization
from qc2.ansatz.qiskit.state_resolution import StateResolution

class SA_OO_VQE(VQE):
    """Main class for state-averaged orbital-optimized VQE with Qiskit-Nature.

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
        state_weights=None,
        init_circuit_params=None,
        init_orbital_params=None,
        freeze_active=False,
        state_resolution=True,
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
            init_circuit_params,
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
        self.ansatz = (self._get_default_ansatzes(
            active_space=self.active_space,
            mapper=self.mapper) 
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

    def run(self) -> SAOOVQEResults:
        """Optimizes both the circuit and orbital parameters.

        Returns:
            SAOOVQEResults:
                An instance of :class:`qc2.algorithms.algorithms_results.SAOOVQEResults`
                class with all SA-OO-VQE info.

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
        >>> qc2data.algorithm = SA_OO_VQE(
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
        print(">>> Optimizing circuit and orbital parameters...")

        # instantiate oo class
        self.oo_problem = OrbitalOptimization(
            self.qc2data,
            self.active_space,
            self.freeze_active,
            self.mapper,
            "qiskit"
        )

        # set initial parameters
        self.orbital_params = (
            self._get_default_init_params(self.oo_problem.n_kappa)
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
        energy_init = self._get_energy_from_parameters(theta, kappa)

        # update lists with intermediate data
        results.update(theta, kappa, energy_init)

        # initial values of the optimization
        self._print_iteration_information(0, energy_init, self.verbose)
        
        for n in range(self.max_iterations):
            # optimize circuit parameters with fixed kappa
            theta, _ = self._circuit_optimization(theta, kappa)

            # optimize orbital parameters with fixed theta from previous run
            kappa, _ = self._orbital_opimization(theta, kappa)

            # calculate final energy with all optimized parameters
            energy = self._get_energy_from_parameters(theta, kappa)

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
            phase, energy = self._phase_optimization(theta, kappa)
            results.optimal_phase = phase
            results.update(theta, kappa, [energy, None])
            self._print_converged_iteration_information(results)

        return results

    @staticmethod
    def _get_default_ansatzes(
        active_space: ActiveSpace,
        mapper: QubitMapper
    ) -> List[QuantumCircuit]:
        """Set up the default UCC ansatz from a Hartree Fock reference state.

        Args:
            active_space (ActiveSpace): Description of the active space.
            mapper (QubitMapper): Mapper class instance.
            reference_state (QuantumCircuit): Reference state circuit.

        Returns:
            UCC: UCC ansatz quantum circuit.
        """

        # create reference state
        reference_state = StateResolution(active_space)

        return [UCC(
            num_spatial_orbitals=active_space.num_active_spatial_orbitals,
            num_particles=active_space.num_active_electrons,
            qubit_mapper=mapper,
            initial_state=reference_state.assign_parameters([phase], inplace=False),
            excitations="sd",
        )
        for phase in [0, np.pi/2]]


    def _get_energy_from_parameters(
            self,
            theta: List,
            kappa: List
    ) -> float:
        """Calculates total energy given circuit and orbital parameters.

        Args:
            theta (List): List with circuit variational parameters.
            kappa (List): List with orbital rotation parameters.

        Returns:
            List:
                Total ground-state energy for the differemnt ansatzes.
        """
        energy = []
        mo_coeff_a, mo_coeff_b = self.oo_problem.get_transformed_mos(kappa)
        for qc in self.ansatz:
            one_rdm, two_rdm = self._get_rdms(qc, theta)
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
            print(f"=== QISKIT {self.__class__.__name__} RESULTS ===")
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


    def _orbital_opimization(self, theta, kappa):
        """
        Optimize orbital parameters with fixed circuit parameters.

        Args:
            theta (List): List with circuit variational parameters.
            kappa (List): List with orbital rotation parameters.

        Returns:
            Tuple[List, float]:
                Optimized orbital parameters and associated energy.
        """
        nparam = len(kappa)
        out = [0.0] * nparam
        total_cost = 0.0
        for qc, weight in zip(self.ansatz, self.state_weights):
            rdm1, rdm2 = self._get_rdms(qc, theta)
            new_kappas, cost = self.oo_problem.orbital_optimization(rdm1, rdm2, kappa)
            total_cost += weight * cost
            out = [out[i] + weight * new_kappas[i] for i in range(nparam)]
        return out, total_cost

    def _circuit_optimization_objective_function(self, theta, kappa):
        """
        Calculate the cost function for circuit optimization.

        Args:
            theta (List): List of circuit variational parameters.
            kappa (List): List of orbital rotation parameters.

        Returns:
            float: The calculated cost function value consisting of the 
                qubit operation energy and core energy.
        """
        (core_energy, qubit_op) = self.oo_problem.get_transformed_qubit_hamiltonian(
                kappa
            )
        
        cost = 0.0        
        for qc , weight in zip(self.ansatz, self.state_weights):
            e = self.estimator.run(
                circuits=qc,
                observables=qubit_op,
                parameter_values=theta
                ).result().values + core_energy
            cost += weight * e
        return cost

    def _circuit_optimization(
            self,
            theta: List,
            kappa: List
        ) -> Tuple[List, float]:
        """Get total energy and best circuit parameters for a given kappa.

        Args:
            theta (List): List with circuit variational parameters.
            kappa (List): List with orbital rotation parameters.

        Returns:
            Tuple[List, float]:
                Optimized circuit parameters and associated energy.
        """
        # optimize theta with kappa fixed
        circuit_optimization_result = self.optimizer.minimize(
            fun=partial(self._circuit_optimization_objective_function, 
                        kappa=kappa), 
            x0=theta
        )
        theta_optimized = circuit_optimization_result.x

        return (theta_optimized, 
                self._circuit_optimization_objective_function(
                    theta_optimized, kappa)
        )


    def _phase_optimization(self, theta, kappa):
        """
        Optimize phase parameters with fixed circuit and orbital parameters.

        Args:
            theta (List): List with circuit variational parameters.
            kappa (List): List with orbital rotation parameters.

        Returns:
            Tuple[List, float]:
                Optimized phase parameters and associated energy.
        """
        phase_optimization_result = self.optimizer.minimize(
            fun=partial(self._phase_optimization_objective_function, 
                    theta=theta, kappa=kappa),
                        x0 = 0.0)
        
        phase_optimized = phase_optimization_result.x

        return (phase_optimized, 
                self._phase_optimization_objective_function(phase_optimized, theta, kappa)
        )    

    def _phase_optimization_objective_function(self, x, theta, kappa):
        """
        Calculate the cost function for phase optimization.

        Args:
            x (float): Phase parameter.
            theta (List): List with circuit variational parameters.
            kappa (List): List with orbital rotation parameters.

        Returns:
            float: The calculated cost function value for the phase optimization.
        """
        mo_coeff_a, mo_coeff_b = self.oo_problem.get_transformed_mos(kappa)
        initial_state = StateResolution(self.active_space)

        qc = UCC(
            num_spatial_orbitals=self.active_space.num_active_spatial_orbitals,
            num_particles=self.active_space.num_active_electrons,
            qubit_mapper=self.mapper,
            initial_state=initial_state.assign_parameters(x, inplace=False),
            excitations="sd")
        
        one_rdm, two_rdm = self._get_rdms(qc, theta)
        return self.oo_problem.get_energy_from_mo_coeffs(
                mo_coeff_a, mo_coeff_b, one_rdm, two_rdm
            )
    
    def _get_rdms(
            self,
            ansatz: QuantumCircuit,
            theta: List,
            sum_spin=True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates 1- and 2-electron reduced density matrices (RDMs).

        Args:
            ansatz (QuantumCircuit): ansatz circuit.
            theta (List): circuit parameters at which
                RDMs are calculated.
            sum_spin (bool): If True, the spin-summed 1-RDM and 2-RDM will be
                returned. If False, the full 1-RDM and 2-RDM will be returned.
                Defaults to True.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                1- and 2-RDMs.
        """
        if len(theta) != ansatz.num_parameters:
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
            qubit_ham_temp = self.mapper.map(
                fermionic_ham_temp, register_length=n_spin_orbitals
            )
            # calculate expectation values
            energy_temp = self.estimator.run(
                circuits=ansatz,
                observables=qubit_ham_temp,
                parameter_values=theta
            ).result().values

            # put the values in np arrays (differentiate 1- and 2-RDM)
            if length == 2:
                rdm1_spin[iele, jele] = energy_temp[0]
            elif length == 4:
                rdm2_spin[iele, lele, jele, kele] = energy_temp[0]

        if sum_spin:
            # get spin-free RDMs
            rdm1_np = np.zeros((n_mol_orbitals,) * 2, dtype=np.complex128)
            rdm2_np = np.zeros((n_mol_orbitals,) * 4, dtype=np.complex128)

            # construct spin-summed 1-RDM
            mod = n_spin_orbitals // 2
            for i, j in itt.product(range(n_spin_orbitals), repeat=2):
                # use i//2 if electrons are organized as a,b,..a,b (pennylane)
                rdm1_np[i % mod, j % mod] += rdm1_spin[i, j]

            # construct spin-summed 2-RDM
            for i, j, k, l in itt.product(range(n_spin_orbitals), repeat=4):
                rdm2_np[
                    i % mod, j % mod, k % mod, l % mod
                ] += rdm2_spin[i, j, k, l]

            return rdm1_np, rdm2_np

        return rdm1_spin, rdm2_spin