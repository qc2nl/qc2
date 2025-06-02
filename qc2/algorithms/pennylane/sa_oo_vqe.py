"""Module defining oo-VQE algorithm for PennyLane."""
from typing import List, Union, Tuple, Callable
import itertools as itt
from pennylane import numpy as np
import pennylane as qml
from qiskit_nature.second_q.operators import FermionicOp
from qc2.algorithms.pennylane.vqe import VQE
from qc2.algorithms.pennylane.oo_vqe import OO_VQE
from qc2.algorithms.algorithms_results import SAOOVQEResults
from qc2.algorithms.utils.orbital_optimization import OrbitalOptimization
from qc2.pennylane.convert import _qiskit_nature_to_pennylane
from qc2.algorithms.utils.active_space import ActiveSpace

class SA_OO_VQE(OO_VQE):
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
        state_weights=None,
        init_circuit_params=None,
        init_orbital_params=None,
        freeze_active=False,
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
            reference_state,
            init_circuit_params,
            init_orbital_params,
            freeze_active,
            max_iterations,
            conv_tol,
            verbose
        )

        # create the reference states 
        self.reference_state = self._get_default_list_reference_state(self.qubits, self.electrons)

        # create the ansatz
        self.ansatz  = self._get_default_list_ansatz(
            self.qubits,
            self.electrons,
            reference_state=self.reference_state
        )

        # create the state weights
        self.state_weights = self._get_default_state_weights(state_weights)

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
        

    def _get_default_list_reference_state(self, qubits: int, electrons: int) -> List[np.ndarray]:
        """Generate the default reference state for the ansatz.

        Args:
            qubits (int): Number of qubits in the circuit.
            electrons (int): Number of electrons in the system.

        Returns:
            np.ndarray: Reference state vector.
        """
        hf = qml.qchem.hf_state(electrons, qubits)
        return [hf, 
                self._get_excited_state(hf, self.active_space)
            ]

    @staticmethod
    def _get_excited_state(
        reference_state: np.ndarray,
        active_space: ActiveSpace,
        excitation: List[List[int]] | List[int] | None = None
    ) -> np.ndarray:
        """Generate the excited state for the ansatz.

        Args:
            reference_state (np.ndarray): Reference state vector.
            active_space (ActiveSpace): Instance of
                :class:`~qc2.algorithm.utils.active_space.ActiveSpace`.
            excitation (List[List[int]] | List[int] | None): Excitation
                operator. Defaults to None.

        Returns:
            np.ndarray: Excited state vector.
        """
        nalpha, nbeta = active_space.num_active_electrons

        if excitation is None:
            alpha_xt = [nalpha + nbeta - 2, nalpha + nbeta]
            beta_xt = [nbeta + nbeta - 1, nbeta + nbeta + 1]
        elif isinstance(excitation[0], int):
            alpha_xt = excitation
            beta_xt = excitation
        elif isinstance(excitation[0], tuple):
            alpha_xt, beta_xt = excitation
        else:
            raise ValueError("excitation must be a List of Lists or a List of ints")
        
        reference_state[alpha_xt[0]] = 0
        reference_state[alpha_xt[1]] = 1

        # reference_state[beta_xt[0]] = 0
        # reference_state[beta_xt[1]] = 1

        return reference_state
        

    @staticmethod
    def _get_default_list_ansatz(
        qubits: int, 
        electrons: int, 
        reference_state: List[np.ndarray]
    ) -> List[Callable]:
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
        # on the HF
        def ref_ansatz(params):
            qml.UCCSD(
                params, wires=range(qubits), s_wires=s_wires,
                d_wires=d_wires, init_state=reference_state[0]
            )
            
        # Return a function that applies the UCCSD ansatz
        # on the excited state
        def excited_ansatz(params):

            #create the HF state
            for iq in range(electrons//2):
                qml.X(2*iq)
                qml.X(2*iq+1)

            #create the excited state
            alpha_xt = [electrons - 2, electrons]
            beta_xt = [electrons - 1, electrons + 1]
            qml.H(alpha_xt[1])
            qml.CNOT(alpha_xt[1], beta_xt[1])
            qml.H(alpha_xt[1])
            qml.CNOT(alpha_xt[1], beta_xt[0])
            qml.CNOT(beta_xt[1], beta_xt[0])

            # create the ansatz
            qml.UCCSD(
                params, wires=range(qubits), s_wires=s_wires,
                d_wires=d_wires, init_state=np.zeros_like(reference_state[0])
            )

        return [ref_ansatz, excited_ansatz]
    
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
        energy = 0.0
        mo_coeff_a, mo_coeff_b = self.oo_problem.get_transformed_mos(kappa)
        for ansatz, weight in zip(self.ansatz, self.state_weights):
            one_rdm, two_rdm = self._get_rdms(ansatz, theta, *args, **kwargs)
            energy +=  weight * self.oo_problem.get_energy_from_mo_coeffs(
                mo_coeff_a, mo_coeff_b, one_rdm, two_rdm
            )
        return energy
    
    def _rotation_optimization(self, theta, kappa, *args, **kwargs):
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