import pennylane as qml
from pennylane import QNode


# class StateResolution(Operation):
#     """State Resolution initial state for SA OO VQE."""

#     def __init__(self, active_space, rotation=0.0, wires=None, id=None):
#         self.active_space = active_space
#         self.rotation = rotation
#         self.wires = range(2 * active_space.num_active_spatial_orbitals) if wires is None else wires

#         super().__init__(wires=self.wires, id=id)


def state_resolution_initializer(
    active_space,
    rotation: float,
    dev: str,
    device_args=None,
    device_kwargs=None,
    qnode_args=None,
    qnode_kwargs=None
) -> QNode:
    """
    Args:
        dev (str): PennyLane quantum device.
        qubits (int): Number of qubits in the circuit.
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

    norb = active_space.num_active_spatial_orbitals
    nalpha, nbeta = active_space.num_active_electrons
    qubits = 2 * norb

    idx_alpha_homo = nalpha + nbeta - 2
    idx_beta_homo  = nalpha + nbeta - 1
    idx_alpha_lumo = nalpha + nbeta
    idx_beta_lumo  = nalpha + nbeta + 1

    # Set default values if None
    device_args = device_args if device_args is not None else []
    device_kwargs = device_kwargs if device_kwargs is not None else {}
    qnode_args = qnode_args if qnode_args is not None else []
    qnode_kwargs = qnode_kwargs if qnode_kwargs is not None else {}

    # Define the device
    device = qml.device(dev, wires=(qubits), *device_args, **device_kwargs)

    # Define the QNode and call the ansatz function within it
    @qml.qnode(device, *qnode_args, **qnode_kwargs)
    def circuit():
        

        # create the fixed electrons 
        for i in range(idx_alpha_homo):
            qml.X(2*i)    

        for i in range(idx_beta_homo):
            qml.X(2*i+1)    

        # rotation
        qml.RY(2*rotation, idx_alpha_homo)
        qml.X(idx_beta_homo)
        qml.CH([idx_alpha_homo, idx_beta_lumo])

        # cnots
        qml.CNOT([idx_beta_lumo, idx_beta_homo])
        qml.CNOT([idx_beta_lumo, idx_alpha_homo])
        qml.CNOT([idx_alpha_homo, idx_alpha_lumo])

        # last steps
        qml.X(idx_alpha_homo)
        qml.Z(idx_beta_lumo)

    return circuit   
        