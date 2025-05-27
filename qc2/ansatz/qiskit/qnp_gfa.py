from qiskit.circuit.library.blueprintcircuit import BlueprintCircuit
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit_nature.second_q.mappers import QubitMapper
from typing import Optional
import numpy as np

class GateFabric(BlueprintCircuit):
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
        super().__init__(2 * num_spatial_orbitals, "GateFabric")
        self._num_spatial_orbitals = num_spatial_orbitals
        self._num_particles = num_particles
        self._qubit_mapper = qubit_mapper
        self._num_layers = num_layers
        self._include_pi = include_pi


        # Use provided initial state, default to an empty circuit
        self._initial_state = initial_state if initial_state is not None else QuantumCircuit(self.num_qubits)

    def _check_configuration(self, raise_on_failure: bool = True) -> bool:

        """Check if the configuration of the NLocal class is valid.

        Args:
            raise_on_failure: Whether to raise on failure.

        Returns:
            True, if the configuration is valid and the circuit can be constructed. Otherwise
            an ValueError is raised.

        Raises:
            ValueError: If the numbr fo qubit is not se.
            ValueError: If the number of spatial orbitals is lower than the number of particles
        """
        valid = True
        if self.num_qubits is None:
            valid = False
            if raise_on_failure:
                raise ValueError("No number of qubits specified.")

        # check no needed parameters are None
        if self._num_spatial_orbitals < self._num_particles[0] or self._num_spatial_orbitals < self._num_particles[1]:
            valid = False
            if raise_on_failure :
                raise ValueError("Number of spatial orbitals inferior to number of particles.")

        return valid

    def _build(self) -> None:
        """Builds the Gate Fabric circuit
        """
        if self._is_built:
            return
        self._check_configuration()
        self._build_circuit()
        self._is_built = True

    def _build_circuit(self):
        # Add initial state preparation
        self.compose(self._initial_state, inplace=True)

        # Create parameters
        parameters = []
        for l in range(self._num_layers):
            layer_params = []
            for g in range(self.num_qubits // 2 - 1):
                theta = Parameter(f'θ_{l}_{g}')
                phi = Parameter(f'φ_{l}_{g}')
                layer_params.append([theta, phi])
            parameters.append(layer_params)

        # Apply Gate Fabric layers
        for layer in range(self._num_layers):
            self._gate_fabric_layer(parameters[layer], range(self.num_qubits))


    def _orbital_rotation(self, theta: Parameter, qubits: list) -> None:
        """Implements the orbital rotation gate"""
        self.ry(theta / 2, qubits[1])
        self.cx(qubits[1], qubits[0])
        self.ry(-theta / 2, qubits[1])
        self.cx(qubits[1], qubits[0])

        self.ry(theta / 2, qubits[3])
        self.cx(qubits[3], qubits[2])
        self.ry(-theta / 2, qubits[3])
        self.cx(qubits[3], qubits[2])

    def _double_excitation(self,  phi: Parameter, qubits: list) -> None:
        """Implements the double excitation gate"""
        self.cx(qubits[3], qubits[2])
        self.cx(qubits[2], qubits[1])
        self.cx(qubits[1], qubits[0])

        self.rz(phi, qubits[0])

        self.cx(qubits[1], qubits[0])
        self.cx(qubits[2], qubits[1])
        self.cx(qubits[3], qubits[2])

    def _gate_fabric_layer(self, parameters: list, wires: list) -> None:
        """Implements a single layer of Gate Fabric"""
        n_gates = len(wires) // 2 - 1

        for i in range(n_gates):
            qubits = wires[i : i + 4]

            if self._include_pi:
                self._orbital_rotation( np.pi, qubits)

            self._double_excitation(parameters[i][0], qubits)
            self._orbital_rotation(parameters[i][1], qubits)


