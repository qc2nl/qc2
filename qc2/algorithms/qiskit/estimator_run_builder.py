"""This module defines the estimator run builder class."""
from typing import Union, List
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.primitives import Estimator, StatevectorEstimator
from qiskit_aer.primitives import Estimator as aer_Estimator
from qiskit_aer.primitives import EstimatorV2 as aer_EstimatorV2
from qiskit_ibm_runtime import Estimator as ibm_runtime_Estimator
from qiskit_ibm_runtime import EstimatorV2 as ibm_runtime_EstimatorV2
from .base_run_builder import BasePrimitiveRunBuilder


EstimatorValidType = Union[
    Estimator,
    StatevectorEstimator,
    aer_Estimator,
    aer_EstimatorV2,
    ibm_runtime_Estimator,
    ibm_runtime_EstimatorV2,
]


class EstimatorRunBuilder(BasePrimitiveRunBuilder):
    """
    A class to build and configure estimator runs based on their provenance.

    Attributes:
        estimator (EstimatorValidType): The quantum estimator instance.
        circuits (List[QuantumCircuit]): List of quantum circuits.
        observables (List[SparsePauliOp]): List of observables.
        parameter_sets (List[List[float]]): List of parameter sets.
    """
    def __init__(
        self,
        estimator: EstimatorValidType,
        circuits: List[QuantumCircuit],
        observables: List[SparsePauliOp],
        parameter_sets: List[List[float]],
    ):
        """
        Initializes the EstimatorRunBuilder with the given arguments.

        Args:
            estimator (EstimatorValidType): The estimator to use for runs.
            circuits (List[QuantumCircuit]): The quantum circuits to run.
            observables (List[SparsePauliOp]): The observables to measure.
            parameter_sets (List[List[float]]): The parameters of the circuits.
        """
        super().__init__(estimator, circuits, parameter_sets)
        self.observables = observables

    def _select_run_builder(self):
        builders = {
            ("qiskit", "Estimator", "BaseEstimatorV1"): self._build_v1_run,
            ("qiskit", "StatevectorEstimator", "BaseEstimatorV2"):
                self._build_native_qiskit_run,
            ("qiskit_aer", "Estimator", "BaseEstimatorV1"): self._build_v1_run,
            ("qiskit_aer", "EstimatorV2", "BaseEstimatorV2"):
                self._build_v2_run,
            ("qiskit_ibm_runtime", "Estimator", "BaseEstimatorV1"):
                self._build_v1_run,
            ("qiskit_ibm_runtime", "EstimatorV2", "BaseEstimatorV2"):
                self._build_v2_run,
        }
        try:
            return builders[self.provenance]
        except KeyError as err:
            raise NotImplementedError(
                f"{
                    self.__class__.__name__
                } not compatible with {self.provenance}."
            ) from err

    def _build_native_qiskit_run(self):
        """Builds a run function for a standard qiskit Estimator."""
        pubs = []
        for qc, obs, param in zip(
            self.circuits, self.observables, self.parameter_sets
        ):
            pubs.append((qc, obs, param))
        return self.primitive.run(pubs)

    def _build_v2_run(self):
        """Builds a run function for aer and ibm-runtime EstimatorV2."""
        backend = self.primitive._backend
        optimization_level = 3
        pm = generate_preset_pass_manager(optimization_level, backend)
        pubs = []
        for qc, obs, param in zip(
            self.circuits, self.observables, self.parameter_sets
        ):
            isa_circuit = pm.run(qc)
            isa_obs = obs.apply_layout(isa_circuit.layout)
            pubs.append((isa_circuit, isa_obs, param))
        return self.primitive.run(pubs)
