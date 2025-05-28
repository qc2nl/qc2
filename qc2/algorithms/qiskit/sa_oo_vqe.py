"""Module defining SA-OO-VQE algorithm for Qiskit-Nature."""
from typing import List, Union
from qiskit.circuit import QuantumCircuit

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
        self.reference_state = (
            self._get_default_excited_state_reference()
            if reference_state is None
            else reference_state
        )

        # create the ansatz
        self.ansatz = self._get_default_excited_state_ansatz(
            active_space=self.active_space,
            mapper=self.mapper,
            reference_state=self.reference_state
        )

        # create the weights
        self.state_weights = self._get_default_state_weights(state_weights)


    def _get_default_excited_state_reference(
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
        return circuit
    
    @staticmethod
    def _get_default_excited_state_ansatz(
        active_space: ActiveSpace,
        mapper: QubitMapper,
        reference_state: QuantumCircuit
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

    @staticmethod
    def _get_default_state_weights(
        state_weights: Union[None, List[float]]
    ) -> List[float]:
        """Set up the default state weights.

        Args:
            state_weights (List[float]): List of state weights.

        Returns:
            List[float]: List of state weights.
        """
        if state_weights is None:
            return [0.5, 0.5]
        return state_weights

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
        energy = 0.0
        mo_coeff_a, mo_coeff_b = self.oo_problem.get_transformed_mos(kappa)
        for qc, weight in zip(self.ansatz, self.state_weights):
            one_rdm, two_rdm = self._get_rdms(qc, theta)
            energy +=  weight * self.oo_problem.get_energy_from_mo_coeffs(
                mo_coeff_a, mo_coeff_b, one_rdm, two_rdm
            )
        return energy
    
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
        for qc, weight in zip(self.ansatz, self.state_weights):
            rdm1, rdm2 = self._get_rdms(qc, theta)
            new_kappas, _ = self.oo_problem.orbital_optimization(rdm1, rdm2, kappa)
            out = [out[i] + weight * new_kappas[i] for i in range(nparam)]
        return out

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
        (core_energy,
            qubit_op) = self.oo_problem.get_transformed_qubit_hamiltonian(
                kappa
            )
        cost = 0.0
        for qc , weight in zip(self.ansatz, self.state_weights):
            job = self.estimator.run(
                circuits=qc,
                observables=qubit_op,
                parameter_values=theta
            )
            cost += weight * job.result().values + core_energy
        return cost
     
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
        results.optimal_energy = energy_l[-1]
        results.optimal_circuit_params = theta_l[-1]
        results.optimal_orbital_params = kappa_l[-1]
        results.optimal_state_weights = self.state_weights
        results.energy = energy_l
        results.circuit_parameters = theta_l
        results.orbital_parameters = kappa_l

        return results