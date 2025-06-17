import pennylane as qml
import numpy as np 
from .state_resolution import state_resolution_initializer
def generate_ansatz(
    qubits: int, 
    electrons: int,
    ansatz_type: str, 
):
    """
    Generate an ansatz function based on the given type.

    Args:
        qubits (int): Number of qubits in the circuit.
        electrons (int): Number of electrons in the system.
        ansatz_type (str): The type of ansatz ('UCCSD').

    Returns:
        Callable: Function that applies the chosen ansatz.
    """
    reference_state = qml.qchem.hf_state(electrons, qubits)

    if ansatz_type is None:
        ansatz_type = "UCCSD"

    if ansatz_type.upper() == "UCCSD":

         # Generate single and double excitations
        singles, doubles = qml.qchem.excitations(electrons, qubits)

        # Map excitations to the wires the UCCSD circuit will act on
        s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)

        # number of parameters
        parameters = np.zeros(len(singles) + len(doubles))

        # Return a function that applies the UCCSD ansatz
        def ansatz(params):
            qml.UCCSD(
                params, wires=range(qubits), s_wires=s_wires,
                d_wires=d_wires, init_state=reference_state
            )
        
        return ansatz, parameters
    
    else:
        raise ValueError("Unsupported ansatz type. Choose from 'UCCSD'.")


def generate_state_resolution_ansatz(    
    qubits: int, 
    electrons: int,
    ansatz_type: str, 
    phase: float
):
    
    """
    Generate an ansatz function that applies the UCCSD ansatz after a state resolution operation.

    Args:
        qubits (int): Number of qubits in the circuit.
        electrons (int): Number of electrons in the system.
        ansatz_type (str): The type of ansatz ('UCCSD').
        phase (float): Phase to be used in the state resolution operation.

    Returns:
        Callable: Function that applies the UCCSD ansatz after the state resolution operation.
    """
    
    if ansatz_type is None:
        ansatz_type = "UCCSD"

    if ansatz_type.upper() == "UCCSD":

         # Generate single and double excitations
        singles, doubles = qml.qchem.excitations(electrons, qubits)

        # Map excitations to the wires the UCCSD circuit will act on
        s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)

        # number of parameters
        parameters = np.zeros(len(singles) + len(doubles))

        def ansatz(params):

            state_resolution_initializer(electrons//2, electrons//2, phase)

            qml.UCCSD(
                params, wires=range(qubits), 
                s_wires=s_wires, d_wires=d_wires, 
                init_state=np.zeros(qubits).astype(int)
            )
        return ansatz, parameters
    
    else:
        raise ValueError("Unsupported ansatz type. Choose from 'UCCSD'.")