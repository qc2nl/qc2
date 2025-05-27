from qiskit.circuit import QuantumCircuit, Parameter
from qiskit_nature.second_q.mappers import QubitMapper
from typing import Optional
import numpy as np

class GateFabric:
    """Gate Fabric ansatz implementation for Qiskit."""

    def __init__(
        self,
        num_spatial_orbitals: int,
        num_particles: tuple,
        qubit_mapper: QubitMapper,
        initial_state: Optional[QuantumCircuit] = None,
        num_layers: int = 1,
        include_pi: bool = False,
    ) -> None:
        """
        Args:
            num_spatial_orbitals: Number of spatial orbitals
            num_particles: Tuple of (alpha, beta) electrons
            qubit_mapper: Mapping of fermionic to qubit operators
            initial_state: Initial state circuit (should be provided externally)
            num_layers: Number of Gate Fabric layers
            include_pi: Whether to include pi rotations in the ansatz
        """
        self._num_spatial_orbitals = num_spatial_orbitals
        self._num_particles = num_particles
        self._qubit_mapper = qubit_mapper
        self._num_layers = num_layers
        self._include_pi = include_pi

        # Number of qubits needed for the problem
        self._num_qubits = 2 * num_spatial_orbitals

        # Use provided initial state, default to an empty circuit
        self._initial_state = initial_state if initial_state is not None else QuantumCircuit(self._num_qubits)

    def _orbital_rotation(self, circuit: QuantumCircuit, theta: Parameter, qubits: list) -> None:
        """Implements the orbital rotation gate"""
        circuit.ry(theta / 2, qubits[1])
        circuit.cx(qubits[1], qubits[0])
        circuit.ry(-theta / 2, qubits[1])
        circuit.cx(qubits[1], qubits[0])

        circuit.ry(theta / 2, qubits[3])
        circuit.cx(qubits[3], qubits[2])
        circuit.ry(-theta / 2, qubits[3])
        circuit.cx(qubits[3], qubits[2])

    def _double_excitation(self, circuit: QuantumCircuit, phi: Parameter, qubits: list) -> None:
        """Implements the double excitation gate"""
        circuit.cx(qubits[3], qubits[2])
        circuit.cx(qubits[2], qubits[1])
        circuit.cx(qubits[1], qubits[0])

        circuit.rz(phi, qubits[0])

        circuit.cx(qubits[1], qubits[0])
        circuit.cx(qubits[2], qubits[1])
        circuit.cx(qubits[3], qubits[2])

    def _gate_fabric_layer(self, circuit: QuantumCircuit, parameters: list, wires: list) -> None:
        """Implements a single layer of Gate Fabric"""
        n_gates = len(wires) // 2 - 1

        for i in range(n_gates):
            qubits = wires[i : i + 4]

            if self._include_pi:
                self._orbital_rotation(circuit, np.pi, qubits)

            self._double_excitation(circuit, parameters[i][0], qubits)
            self._orbital_rotation(circuit, parameters[i][1], qubits)

    def build(self) -> QuantumCircuit:
        """Builds the Gate Fabric circuit
        
        Returns:
            QuantumCircuit: The constructed circuit
        """
        circuit = QuantumCircuit(self._num_qubits)

        # Add initial state preparation
        circuit.compose(self._initial_state, inplace=True)

        # Create parameters
        parameters = []
        for l in range(self._num_layers):
            layer_params = []
            for g in range(self._num_qubits // 2 - 1):
                theta = Parameter(f'θ_{l}_{g}')
                phi = Parameter(f'φ_{l}_{g}')
                layer_params.append([theta, phi])
            parameters.append(layer_params)

        # Apply Gate Fabric layers
        for layer in range(self._num_layers):
            self._gate_fabric_layer(circuit, parameters[layer], range(self._num_qubits))

        return circuit

