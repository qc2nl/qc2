"""Module defining QPE algorithm for PennyLane."""
import pennylane as qml
from pennylane import numpy as np
from pennylane import QNode
from pennylane.operation import Operator
from .pebase import PEBase

class QPE(PEBase):
    def __init__(
        self,
        qc2data=None,
        active_space=None,
        mapper=None,
        device=None,
        reference_state=None,
        num_evaluation_qubits=None,
        verbose=0
    ):
        
        super().__init__(qc2data, active_space, mapper, device, reference_state, verbose)
        self.num_evaluation_qubits = 3 if num_evaluation_qubits is None else num_evaluation_qubits
        
    def get_phase(self):
        """Estimate the phase from the quantum circuit.

        The method executes the quantum circuit and determines the most probable 
        measurement outcome's index. This index is then divided by 2 raised to the 
        power of the number of evaluation qubits to estimate the phase.

        Returns:
            float: The estimated phase.
        """
        return np.argmax(self.circuit()) / 2**self.num_evaluation_qubits
    
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
        def circuit():
            
            # HF state
            qml.BasisState(reference_state, wires=range(qubits))
            
            # Hadamard on estimation wires
            for q in estimation_wires:
                qml.Hadamard(wires=q)

            # cotrolled unitary
            for q in range(num_estimation_wires):
                qml.ctrl(qml.pow(unitary_op,2**q), 
                         control=qubits+num_estimation_wires-q-1)
                
            # QFT
            qml.adjoint(qml.templates.QFT(wires=estimation_wires))

            # measure
            return qml.probs(estimation_wires)

        return circuit