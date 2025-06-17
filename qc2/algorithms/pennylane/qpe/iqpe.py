"""Module defining QPE algorithm for PennyLane."""
import pennylane as qml
from pennylane import numpy as np
from pennylane import QNode
from pennylane.operation import Operator
from .pebase import PEBase

class IQPE(PEBase):
    def __init__(
        self,
        qc2data=None,
        active_space=None,
        mapper=None,
        device=None,
        reference_state=None,
        num_iterations=None,
        verbose=0
    ):
        
        super().__init__(qc2data, active_space, mapper, device, reference_state, verbose)
        self.num_iterations = 3 if num_iterations is None else num_iterations
        self.num_evaluation_qubits = 1
        
    def get_phase(self):
        """Estimate the phase from the quantum circuit.

        The method executes the quantum circuit and determines the most probable 
        measurement outcome's index. This index is then divided by 2 raised to the 
        power of the number of evaluation qubits to estimate the phase.

        Returns:
            float: The estimated phase.
        """
        omega = 0.0
        for k in range(self.num_iterations,0, -1):
            omega /= 2 
            probs = self.circuit(k, omega)
            x = 1 if probs[1] > probs[0] else 0
            omega = omega + x / 2
        return omega
    
    @staticmethod
    def _build_circuit(
        dev: str,
        qubits: int,
        num_estimation_wires: int,
        reference_state: np.ndarray,
        unitary_op: Operator,
        device_args=None,
        device_kwargs=None,
        qnode_args=None,
        qnode_kwargs=None
    ) -> QNode:
        """Builds and return PennyLane QNode.

        Args:
            dev (str): PennyLane quantum device.
            qubits (int): Number of qubits in the circuit.
            num_estimation_wires (int): number of qubits for estimation
            reference_state (np.ndarray): Reference state for the circuit.
            unitary_op (Operator): Qubit operator for the exp(iH).
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
        if num_estimation_wires != 1:
            raise ValueError('Number of evaluation wires must be 1 for IQPE')
        
        # Set default values if None
        device_args = device_args if device_args is not None else []
        device_kwargs = device_kwargs if device_kwargs is not None else {}
        qnode_args = qnode_args if qnode_args is not None else []
        qnode_kwargs = qnode_kwargs if qnode_kwargs is not None else {}

        # Define the device
        device = qml.device(dev, wires=(qubits+num_estimation_wires), *device_args, **device_kwargs)

        # range of estimation qubits
        estimation_wires = range(qubits, qubits + num_estimation_wires)

        # Define the QNode and call the ansatz function within it
        @qml.qnode(device, *qnode_args, **qnode_kwargs)
        def circuit(k, omega):
            
            # HF state
            qml.BasisState(reference_state, wires=range(qubits))
            
            # Hadamard on estimation wires
            qml.Hadamard(wires=estimation_wires[0])

            # controlled unitary
            qml.ctrl(qml.pow(unitary_op,2**(k-1)), control=estimation_wires[0])
                
            # phase on eval
            qml.PhaseShift(omega, wires=estimation_wires[0])

            # hadamard on eval
            qml.Hadamard(wires=estimation_wires[0])

            # measure
            return qml.probs(estimation_wires)

        return circuit