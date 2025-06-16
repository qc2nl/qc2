"""Module defining the QPE algorithm for qiskit"""
import numpy as np
from qiskit_algorithms import IterativePhaseEstimation 
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.primitives import BaseSampler
from .pebase import PEBase

class QC2IterativePhaseEstimation(IterativePhaseEstimation):
    """Run the Iterative quantum phase estimation (QPE) algorithm.

    Rewrote here to control the qubit ordering and harmonize with pennlyane implementation

    """
    def __init__(
        self,
        num_iterations: int,
        sampler: BaseSampler | None = None,
    ) -> None:
        super().__init__(num_iterations, sampler)

    def construct_circuit(
        self,
        unitary: QuantumCircuit,
        state_preparation: QuantumCircuit,
        k: int,
        omega: float = 0.0,
        measurement: bool = False,
    ) -> QuantumCircuit:
        """Construct the kth iteration Quantum Phase Estimation circuit.

        For details of parameters, see Fig. 2 in https://arxiv.org/pdf/quant-ph/0610214.pdf.

        Args:
            unitary: The circuit representing the unitary operator whose eigenvalue (via phase)
                 will be measured.
            state_preparation: The circuit that prepares the state whose eigenphase will be
                 measured.  If this parameter is omitted, no preparation circuit
                 will be run and input state will be the all-zero state in the
                 computational basis.
            k: the iteration idx.
            omega: the feedback angle.
            measurement: Boolean flag to indicate if measurement should
                be included in the circuit.

        Returns:
            QuantumCircuit: the quantum circuit per iteration
        """

        qr_eval = QuantumRegister(1, 'eval')
        qr_state = QuantumRegister(unitary.num_qubits, 'q')
        creg = ClassicalRegister(1, name="meas")
        circuit = QuantumCircuit(qr_state, qr_eval, creg)

        self._num_state_qubits = unitary.num_qubits
        k = self._num_iterations if k is None else k

        # initial state
        circuit.compose(state_preparation, qubits=qr_state, inplace=True)

        # hadamard on eval qubit
        circuit.h(qr_eval)

        # power of unitary
        circuit.compose(unitary.power(2 ** (k - 1)).control(), 
                        qubits=[qr_eval[0]] + qr_state[:], 
                        inplace=True)

        # phase on eval
        circuit.p(omega, qr_eval[0])

        # hadamard on eval register
        circuit.h(qr_eval[0])

        # measure
        if measurement:
            circuit.barrier()
            circuit.measure(qr_eval, creg)

        return circuit

class IQPE(PEBase):
    def __init__(self, 
                 qc2data=None, 
                 num_iterations=None,
                 active_space=None, 
                 mapper=None, 
                 sampler=None, 
                 reference_state=None,  
                 verbose=0):
        super().__init__(qc2data, active_space, mapper, sampler, reference_state, verbose)
        self.num_iterations = 3 if num_iterations is None else num_iterations
        self.solver = QC2IterativePhaseEstimation(self.num_iterations, self.sampler)
