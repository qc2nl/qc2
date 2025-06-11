"""Tests for the qiskit ansatz"""
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qc2.ansatz.qiskit import LUCJ, GateFabric
import pyscf

def test_gate_fabric():
    num_spatial_orbitals = 4
    num_particles = (2, 2)
    mapper = JordanWignerMapper()
    reference_state = HartreeFock(num_spatial_orbitals, num_particles, mapper)
    gate_fabric = GateFabric(num_spatial_orbitals, num_particles, mapper, initial_state=reference_state)
    gate_fabric._build()
    assert gate_fabric.num_qubits == 8
    

def test_lucj():

    # Build an N2 molecule
    mol = pyscf.gto.Mole()
    mol.build(atom=[["N", (0, 0, 0)], ["N", (1.0, 0, 0)]], basis="6-31g", symmetry="Dooh")
    active_space = range(4, mol.nao_nr())
    lucj = LUCJ(mol, active_space)
    lucj.get_state()
    lucj._build()

if __name__ == "__main__":
    test_gate_fabric()
    test_lucj()