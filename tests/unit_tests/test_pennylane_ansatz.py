from typing import Callable
import unittest
import pennylane as qml
import numpy as np
from qc2.ansatz.pennylane.generate_ansatz import generate_ansatz

class TestGenerateAnsatz(unittest.TestCase):
    def test_uccsd_ansatz_generation(self):
        qubits = 4
        electrons = 2
        ansatz_type = "UCCSD"
        ansatz, parameters = generate_ansatz(qubits, electrons, ansatz_type)
        self.assertIsInstance(ansatz, Callable)
        self.assertEqual(len(parameters), 3) 

    def test_default_ansatz_type(self):
        qubits = 4
        electrons = 2
        ansatz, parameters = generate_ansatz(qubits, electrons, None)
        self.assertIsInstance(ansatz, Callable)
        self.assertEqual(len(parameters), 3)

    def test_invalid_ansatz_type(self):
        qubits = 4
        electrons = 2
        ansatz_type = "Invalid"
        with self.assertRaises(ValueError):
            generate_ansatz(qubits, electrons, ansatz_type)

    def test_ansatz_function_application(self):
        qubits = 4
        electrons = 2
        ansatz_type = "UCCSD"
        ansatz, parameters = generate_ansatz(qubits, electrons, ansatz_type)
        params = np.random.rand(len(parameters))
        ansatz(params)

    def test_ansatz_function_application_invalid_params(self):
        qubits = 4
        electrons = 2
        ansatz_type = "UCCSD"
        ansatz, parameters = generate_ansatz(qubits, electrons, ansatz_type)
        params = np.random.rand(len(parameters) + 1)  # invalid length
        with self.assertRaises(ValueError):
            ansatz(params)
            
if __name__ == "__main__":
    unittest.main()