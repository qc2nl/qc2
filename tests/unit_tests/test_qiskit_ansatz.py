import unittest
from pyscf.gto import Mole 
from qc2.ansatz.qiskit.generate_ansatz import generate_ansatz
from qc2.ansatz.qiskit.hatree_fock import HartreeFock
from qc2.qubit_mappers.qiskit.jordan_wigner import JordanWigner
from qc2.ansatz.qiskit import LUCJ, GateFabric, UCCSD, PUCCSD


class TestGenerateAnsatz(unittest.TestCase):
    def test_uccsd_ansatz(self):
        num_spatial_orbitals = 4
        num_particles = (2, 2)
        mapper = JordanWigner()
        ansatz_type = "UCCSD"
        ansatz = generate_ansatz(num_spatial_orbitals, num_particles, mapper, ansatz_type)
        self.assertIsInstance(ansatz, UCCSD)

    def test_puccsd_ansatz(self):
        num_spatial_orbitals = 4
        num_particles = (2, 2)
        mapper = JordanWigner()
        ansatz_type = "PUCCSD"
        ansatz = generate_ansatz(num_spatial_orbitals, num_particles, mapper, ansatz_type)
        self.assertIsInstance(ansatz, PUCCSD)

    def test_gate_fabric_ansatz(self):
        num_spatial_orbitals = 4
        num_particles = (2, 2)
        mapper = JordanWigner()
        ansatz_type = "GateFabric"
        ansatz = generate_ansatz(num_spatial_orbitals, num_particles, mapper, ansatz_type)
        self.assertIsInstance(ansatz, GateFabric)

    def test_lucj_ansatz(self):
        num_spatial_orbitals = 4
        num_particles = (2, 2)
        mapper = JordanWigner()
        mol = Mole()
        mol.atom = '''O 0 0 0; H  0 1 0; H 0 0 1'''
        mol.basis = 'sto-3g'
        mol.build()
        ansatz_type = "LUCJ"
        ansatz = generate_ansatz(num_spatial_orbitals, num_particles, mapper, ansatz_type, mol=mol)
        self.assertIsInstance(ansatz, LUCJ)

    def test_unsupported_ansatz_type(self):
        num_spatial_orbitals = 4
        num_particles = (2, 2)
        mapper = JordanWigner()
        ansatz_type = "Unsupported"
        with self.assertRaises(ValueError):
            generate_ansatz(num_spatial_orbitals, num_particles, mapper, ansatz_type)

    def test_lucj_without_mol(self):
        num_spatial_orbitals = 4
        num_particles = (2, 2)
        mapper = JordanWigner()
        ansatz_type = "LUCJ"
        with self.assertRaises(ValueError):
            generate_ansatz(num_spatial_orbitals, num_particles, mapper, ansatz_type)

    def test_default_reference_state(self):
        num_spatial_orbitals = 4
        num_particles = (2, 2)
        mapper = JordanWigner()
        ansatz_type = "UCCSD"
        ansatz = generate_ansatz(num_spatial_orbitals, num_particles, mapper, ansatz_type)
        self.assertIsInstance(ansatz.initial_state, HartreeFock)

    def test_provided_reference_state(self):
        num_spatial_orbitals = 4
        num_particles = (2, 2)
        mapper = JordanWigner()
        ansatz_type = "UCCSD"
        reference_state = HartreeFock(num_spatial_orbitals, num_particles, mapper)
        ansatz = generate_ansatz(num_spatial_orbitals, num_particles, mapper, ansatz_type, reference_state=reference_state)
        self.assertEqual(ansatz.initial_state, reference_state)

if __name__ == '__main__':
    unittest.main() 