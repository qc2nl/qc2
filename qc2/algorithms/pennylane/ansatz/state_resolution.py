import pennylane as qml


def state_resolution_initializer(
    nalpha, nbeta,
    rotation: float,
) -> None:
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

    idx_alpha_homo = nalpha + nbeta - 2
    idx_beta_homo  = nalpha + nbeta - 1
    idx_alpha_lumo = nalpha + nbeta
    idx_beta_lumo  = nalpha + nbeta + 1
  
    # create the fixed electrons 
    for i in range(nalpha-1):
        qml.X(2*i)    

    for i in range(nbeta-1):
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