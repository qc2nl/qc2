from typing import List, Union
from qiskit.circuit.library.blueprintcircuit import BlueprintCircuit
from qiskit.exceptions import QiskitError
from qiskit.circuit import QuantumCircuit, Parameter, QuantumRegister, QuantumCircuit
from qiskit_nature.second_q.mappers import QubitMapper
from typing import Optional
import numpy as np
from qc2.algorithms.utils.active_space import ActiveSpace

def get_excited_state_circuit(
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

        circuit = QuantumCircuit(2*norb) 
        circuit.barrier()
        circuit.h(alpha_xt[1])
        circuit.cx(alpha_xt[1],  beta_xt[1]+norb)
        circuit.x(alpha_xt[1])
        circuit.barrier()
        circuit.cx(alpha_xt[1],alpha_xt[0])
        circuit.cx(beta_xt[1]+norb,beta_xt[0]+norb)
        circuit.z(beta_xt[1]+norb)
        return circuit


def get_excited_state_rotation(
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