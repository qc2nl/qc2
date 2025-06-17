from typing import Tuple
from qiskit.circuit import QuantumCircuit
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD, SUCCD, PUCCSD, PUCCD
from qiskit_nature.second_q.mappers import QubitMapper
from .gate_fabric import GateFabric  # Import Gate Fabric ansatz
from .lucj import LUCJ  # Import LUCJ ansatz

"""
- UCCSD - Unitary Coupled-Cluster ansatz with single and double excitations, 
		commonly used for quantum chemistry simulations.
- SUCCD - A simplified Unitary Coupled-Cluster ansatz that includes only double excitations.
- PUCCSD -  A spin-adapted paired-UCC ansatz that preserves both particle number and 
		total spin by including paired single and double excitations.
- PUCCD -  A paired-UCC ansatz that contains only double excitations, 
		enforcing parallel excitations in both alpha and beta spin sectors.
- QNP_GFA - A quantum number preserving Gate Fabric ansatz that ensures particle number and 
		spin are conserved in the ansatz construction.
- LUCJ - A Unitary Coupled-Cluster ansatz with a Jastrow factor, initialized using 
		CCSD amplitudes and implemented with ffsim for enhanced expressibility.
	
	# Example usage
	ansatz_type = "LUCJ"
	ansatz = create_ansatz(num_spatial_orbitals, num_particles, mapper, ansatz_type, mol_data, scf)
	print(ansatz)  # Prints LUCJ ansatz state as a numpy array
"""



def generate_ansatz(
    num_spatial_orbitals: int, 
    num_particles: Tuple[int, int], 
    mapper: QubitMapper, 
    ansatz_type: str, 
    reference_state: QuantumCircuit | None = None, 
    mol_data=None, 
    scf=None
) -> QuantumCircuit:
    """
    Creates an ansatz based on the given type.
    Args:
        num_spatial_orbitals (int): Number of spatial orbitals
        num_particles (tuple): Tuple of (alpha, beta) electrons
        mapper (QubitMapper): The fermion-to-qubit mapper
        ansatz_type (str): The type of ansatz ('UCCSD', 'SUCCD', 'PUCCSD', 'PUCCD', 'GateFabric', 'LUCJ')
        reference_state (QuantumCircuit|None): Reference state circuit
        mol_data: Molecular data (needed only for LUCJ)
        scf: SCF object (needed only for LUCJ)
    Returns:
        QuantumCircuit or np.ndarray: The constructed ansatz circuit/state
    """
    if reference_state is None:
        reference_state = HartreeFock(num_spatial_orbitals, num_particles, mapper)

    if ansatz_type is None:
        ansatz_type = "UCCSD"

    if ansatz_type.upper() == "UCCSD":
        return UCCSD(num_spatial_orbitals, num_particles, mapper, initial_state=reference_state)

    elif ansatz_type.upper() == "SUCCD":
        return SUCCD(num_spatial_orbitals, num_particles, mapper, initial_state=reference_state)

    elif ansatz_type.upper() == "PUCCSD":
        return PUCCSD(num_spatial_orbitals, num_particles, mapper, initial_state=reference_state)

    elif ansatz_type.upper() == "PUCCD":
        return PUCCD(num_spatial_orbitals, num_particles, mapper, initial_state=reference_state)

    elif ansatz_type.upper() == "GateFabric":
        return GateFabric(num_spatial_orbitals, num_particles, mapper, initial_state=reference_state)

    elif ansatz_type.upper() == "LUCJ":
        if mol_data is None or scf is None:
            raise ValueError("LUCJ requires 'mol_data' and 'scf' objects.")
        return LUCJ(mol_data, scf, num_spatial_orbitals, num_particles)

    else:
        raise ValueError("Unsupported ansatz type. Choose from 'UCCSD', 'SUCCD', 'PUCCSD', 'PUCCD', 'GateFabric', or 'LUCJ'.")