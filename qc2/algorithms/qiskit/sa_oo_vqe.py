"""Module defining SA-OO-VQE algorithm for Qiskit-Nature."""
from typing import List, Union
import numpy as np
from qiskit.circuit import QuantumCircuit
from functools import partial
from qiskit_nature.second_q.circuit.library import HartreeFock, UCC
from qiskit_nature.second_q.mappers import QubitMapper

from qc2.algorithms.qiskit.oo_vqe import OO_VQE
from qc2.algorithms.algorithms_results import SAOOVQEResults
from qc2.algorithms.utils.active_space import ActiveSpace

class SA_OO_VQE(OO_VQE):
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
        reference_state=None,
        state_weights=None,
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
            reference_state (QuantumCircuit): Reference state for the VQE
                algorithm. Defaults to :class:`qiskit.HartreeFock`.
            state_weights (List): List of state weights. Defaults to [0.5, 0.5].
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
            reference_state,
            init_circuit_params,
            init_orbital_params,
            freeze_active,
            max_iterations,
            conv_tol,
            verbose
        )

        # create the reference states 
        # self.reference_state = self._get_default_list_reference_state()
        self.reference_state = self._get_default_list_reference_state_rotation()

        # create the ansatz
        self.ansatz = self._get_default_list_ansatz(
            active_space=self.active_space,
            mapper=self.mapper,
            reference_state=self.reference_state
        )

        # create the weights
        self.state_weights = [0.5, 0.5] if state_weights is None else state_weights


    def _get_default_list_reference_state(
        self
    ) -> List[QuantumCircuit]:
        """Set up the default reference state circuit based on Hartree Fock and singlet excitation.

        Returns:
            List[QuantumCircuit]: Hartree-Fock circuit as the reference state.
        """
        hf = self._get_default_reference(self.active_space, self.mapper)
        return[
            hf,
            self._get_excited_state_circuit(hf.copy(), self.active_space)
        ]
    @staticmethod
    def _get_excited_state_circuit(
            circuit: QuantumCircuit,
            active_space: ActiveSpace,
            excitation: List[List[int]] | List[int] | None = None
    ) -> QuantumCircuit:
        """
        Set up the default excited state circuit based on Hartree Fock and single excitation.

        Parameters:
        excitation (List[List[int, int], List[int, int]] | List[int, int] | None): 
            The excitation to be applied to the Hartree-Fock state. If None, the default
            excitation is to excite the highest occupied molecular orbital (HOMO) to the lowest
            unoccupied molecular orbital (LUMO) for both alpha and beta spin orbitals.

        Returns:
            QuantumCircuit: The excited state circuit.
        """
        norb = active_space.num_active_spatial_orbitals
        nalpha, nbeta = active_space.num_active_electrons

        # excitation
        if excitation is None:
            alpha_xt = [nalpha - 1, nalpha]
            beta_xt = [nbeta - 1, nbeta]
        elif isinstance(excitation[0], int):
            alpha_xt = excitation
            beta_xt = excitation
        elif isinstance(excitation[0], tuple):
            alpha_xt, beta_xt = excitation
        else:
            raise ValueError("excitation must be a List of Lists or a List of ints")

        circuit.barrier()
        circuit.h(alpha_xt[1])
        circuit.cx(alpha_xt[1],  beta_xt[1]+norb)
        circuit.x(alpha_xt[1])
        circuit.barrier()
        circuit.cx(alpha_xt[1],alpha_xt[0])
        circuit.cx(beta_xt[1]+norb,beta_xt[0]+norb)
        circuit.z(beta_xt[1]+norb)
        return circuit
    

    def _get_default_list_reference_state_rotation(
                self
    ) -> List[QuantumCircuit]:
        """Set up the default reference state circuit based on Hartree Fock and singlet excitation.

        Returns:
            List[QuantumCircuit]: Hartree-Fock circuit as the reference state.
        """
        return[self._get_state_rotation(self.active_space, rotation=0.0),
               self._get_state_rotation(self.active_space, rotation=np.pi/2)]

    @staticmethod
    def _get_state_rotation(
        active_space,
        rotation: float
    ) -> QuantumCircuit:
        """
        Set up the default reference state circuit based on Hartree Fock and singlet excitation.

        Parameters:
        active_space (ActiveSpace): Description of the active space.
        rotation (float): The rotation angle in radians.

        Returns:
            QuantumCircuit: The reference state circuit.
        """
        norb = active_space.num_active_spatial_orbitals
        nalpha, nbeta = active_space.num_active_electrons

        idx_alpha_homo = nalpha - 1
        idx_beta_homo  = norb + nbeta - 1
        idx_alpha_lumo = nalpha
        idx_beta_lumo  = norb + nbeta

        qc = QuantumCircuit(2*norb)

        # create the fixed electrons 
        for i in range(idx_alpha_homo):
            qc.x(i)    

        for i in range(norb, idx_beta_homo):
            qc.x(i)    

        # rotation
        qc.ry(2*rotation, idx_alpha_homo)
        qc.x(idx_beta_homo)
        qc.ch(idx_alpha_homo, idx_beta_lumo)

        # cnots
        qc.cx(idx_beta_lumo, idx_beta_homo)
        qc.cx(idx_beta_lumo, idx_alpha_homo)
        qc.cx(idx_alpha_homo, idx_alpha_lumo)

        # last steps
        qc.x(idx_alpha_homo)
        qc.z(idx_beta_lumo)

        return qc

    @staticmethod
    def _get_default_list_ansatz(
        active_space: ActiveSpace,
        mapper: QubitMapper,
        reference_state: List[QuantumCircuit]
    ) -> List[QuantumCircuit]:
        """Set up the default UCC ansatz from a Hartree Fock reference state.

        Args:
            active_space (ActiveSpace): Description of the active space.
            mapper (QubitMapper): Mapper class instance.
            reference_state (QuantumCircuit): Reference state circuit.

        Returns:
            UCC: UCC ansatz quantum circuit.
        """

        return [UCC(
            num_spatial_orbitals=active_space.num_active_spatial_orbitals,
            num_particles=active_space.num_active_electrons,
            qubit_mapper=mapper,
            initial_state=ref,
            excitations="sd",
        )
        for ref in reference_state]


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
            float:
                Total ground-state energy for a given circuit
                and orbital parameters.
        """
        energy = []
        mo_coeff_a, mo_coeff_b = self.oo_problem.get_transformed_mos(kappa)
        for qc in self.ansatz:
            one_rdm, two_rdm = self._get_rdms(qc, theta)
            energy.append(self.oo_problem.get_energy_from_mo_coeffs(
                mo_coeff_a, mo_coeff_b, one_rdm, two_rdm
            ))
        return energy

    @staticmethod
    def _print_iteration_information(n: int , energy: List[float]) -> None:
        """
        Prints the iteration number and corresponding energy in Hartree.

        Args:
            n (int): The current iteration number.
            energy (float): The energy value associated with the current iteration.
        """
        print(f"iter = {n+1:03}, energy_0  = {energy[0]:.12f} Ha, energy_1 = {energy[1]:.12f} Ha")

    def convegence_criterion(self, energy: List, prev_energy: List) -> bool:
        """
        Checks if the difference between the current and previous energy is smaller
        than the set convergence tolerance.

        Args:
            energy (float): The current energy value.
            prev_energy (float): The previous energy value.

        Returns:
            bool: True if the difference between the current and previous energy is
            smaller than the set convergence tolerance, False otherwise.
        """
        return abs(energy[0] - prev_energy[0]) < self.conv_tol


    def _rotation_opimization(self, theta, kappa):
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


    def _estimate_states_energies(self, theta, kappa):
        """
        Calculate the energies of the different states

        Args:
            theta (List): List of circuit variational parameters.
            kappa (List): List of orbital rotation parameters.

        Returns:
            List: The calculated state energies
        """
        (core_energy, qubit_op) = self.oo_problem.get_transformed_qubit_hamiltonian(
                kappa
            )
        return [
            self.estimator.run(
                circuits=qc,
                observables=qubit_op,
                parameter_values=theta
            ).result().values + core_energy
            for qc in self.ansatz
            ]

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
        cost = 0.0
        energies = self._estimate_states_energies(theta, kappa)
        for e , weight in zip(energies, self.state_weights):
            cost += weight * e
        return cost
     
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
        return phase_optimized, phase_optimization_result(phase_optimized) 

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
        
        initial_state = self._get_state_rotation(self.active_space, rotation=x)

        qc =  UCC(
            num_spatial_orbitals=self.active_space.num_active_spatial_orbitals,
            num_particles=self.active_space.num_active_electrons,
            qubit_mapper=self.mapper,
            initial_state=initial_state,
            excitations="sd")
        
        one_rdm, two_rdm = self._get_rdms(qc, theta)
        return self.oo_problem.get_energy_from_mo_coeffs(
                mo_coeff_a, mo_coeff_b, one_rdm, two_rdm
            )

    def _store_results(self, energy_l, theta_l, kappa_l, n):
        """
        Stores the results of the optimization process in an OOVQEResults instance.

        Args:
            energy_l (List[float]): List of energies at each iteration.
            theta_l (List[List[float]]): List of circuit parameters at each iteration.
            kappa_l (List[List[float]]): List of orbital parameters at each iteration.
            n (int): Number of optimizer evaluations performed.

        Returns:
            OOVQEResults: An instance containing the results of the optimization, 
            including optimal parameters and energies.
        """

         # instantiate OOVQEResults
        results = SAOOVQEResults()
        results.optimizer_evals = n
        results.optimal_energy = energy_l[-1][0]
        results.optimal_circuit_params = theta_l[-1]
        results.optimal_orbital_params = kappa_l[-1]
        results.optimal_state_weights = self.state_weights
        results.energy = energy_l
        results.circuit_parameters = theta_l
        results.orbital_parameters = kappa_l

        return results