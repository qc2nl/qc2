"""Tests for the qiskit ansatz"""
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qc2.ansatz.qiskit import LUCJ, GateFabric, create_ansatz

def test_gate_fabric():
    num_spatial_orbitals = 4
    num_particles = (2, 2)
    mapper = JordanWignerMapper()
    reference_state = HartreeFock(num_spatial_orbitals, num_particles, mapper)
    gate_fabric = GateFabric(num_spatial_orbitals, num_particles, mapper, initial_state=reference_state)
    assert gate_fabric.num_qubits == 8
