"""Module defining the QPE algorithm for qiskit"""
from qiskit_algorithms import PhaseEstimation
from qiskit_algorithms import PhaseEstimationResult
from qiskit_algorithms.exceptions import AlgorithmError
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import QFT
from qiskit.primitives import BaseSampler
from .pebase import PEBase


class QC2PhaseEstimation(PhaseEstimation):
    r"""Run the Quantum Phase Estimation (QPE) algorithm.

    Rewrote here to control the qubit ordering 

    """

    def __init__(
        self,
        num_evaluation_qubits: int,
        sampler: BaseSampler | None = None,
    ) -> None:
        r"""
        Args:
            num_evaluation_qubits: The number of qubits used in estimating the phase. The phase will
                be estimated as a binary string with this many bits.
            sampler: The sampler primitive on which the circuit will be sampled.

        Raises:
            AlgorithmError: If a sampler is not provided
        """
        super().__init__(num_evaluation_qubits, sampler)

    def construct_circuit(
        self, unitary: QuantumCircuit, 
        state_preparation: QuantumCircuit | None = None,
        name:str = "QPE"
    ) -> QuantumCircuit:
        """Return the circuit to be executed to estimate phases.

        This circuit includes as sub-circuits the core phase estimation circuit,
        with the addition of the state-preparation circuit and possibly measurement instructions.
        """

        qr_eval = QuantumRegister(self._num_evaluation_qubits, "eval")
        qr_state = QuantumRegister(unitary.num_qubits, "q")
        self._num_state_qubits = unitary.num_qubits
        circuit = QuantumCircuit(qr_state, qr_eval, name=name)
        iqft = QFT(self._num_evaluation_qubits, inverse=True, do_swaps=False)

        # initial state
        circuit.compose(state_preparation, qubits=qr_state, inplace=True) 

        # hadamard on evaluation qubits
        for q in qr_eval:
            circuit.h(q)

        # power of unitary
        for q in range(self._num_evaluation_qubits):
            circuit.compose(unitary.power(2**q).control(), 
                            qubits=[qr_eval[-(q+1)]] + qr_state[:], 
                            inplace=True)

        # qft
        circuit.compose(iqft, qubits=qr_eval, inplace=True)

        return circuit

    @staticmethod
    def _get_bitstring(length: int, number: int) -> str:
        return f"{number:b}".zfill(length)

    def _add_measurement_if_required(self, pe_circuit):

        # Measure only the evaluation qubits.
        regname = "meas"
        creg = ClassicalRegister(self._num_evaluation_qubits, regname)
        pe_circuit.add_register(creg)
        pe_circuit.barrier()
        idx = range(self._num_state_qubits, self._num_state_qubits + self._num_evaluation_qubits)
        pe_circuit.measure(idx, range(self._num_evaluation_qubits))

    def estimate_from_pe_circuit(self, pe_circuit: QuantumCircuit) -> PhaseEstimationResult:
        """Run the phase estimation algorithm on a phase estimation circuit

        Args:
            pe_circuit: The phase estimation circuit.

        Returns:
            A phase estimation result.

        Raises:
            AlgorithmError: Primitive job failed.
        """

        self._add_measurement_if_required(pe_circuit)

        try:
            circuit_job = self._sampler.run([pe_circuit])
            circuit_result = circuit_job.result()
        except Exception as exc:
            raise AlgorithmError("The primitive job failed!") from exc
        phases = circuit_result.quasi_dists[0]
        phases_bitstrings = {}
        for key, phase in phases.items():
            bitstring_key = self._get_bitstring(self._num_evaluation_qubits, key)
            phases_bitstrings[bitstring_key] = phase
        phases = phases_bitstrings

        return PhaseEstimationResult(
            self._num_evaluation_qubits, circuit_result=circuit_result, phases=phases
        )


class QPE(PEBase):
    def __init__(self, 
                 qc2data=None, 
                 num_evaluation_qubits=None,
                 active_space=None, 
                 mapper=None, 
                 sampler=None, 
                 reference_state=None,  
                 verbose=0,
                 debug=True):
        super().__init__(qc2data, active_space, mapper, sampler, reference_state, verbose)
        self.num_evaluation_qubits = num_evaluation_qubits
        if debug:
            self.solver = PhaseEstimation(self.num_evaluation_qubits, self.sampler)
        else:
            self.solver = QC2PhaseEstimation(self.num_evaluation_qubits, self.sampler)

