"""This module defines a base class for primitive run builders."""
from typing import List, Tuple
from qiskit import QuantumCircuit
from qiskit.primitives.base import BaseEstimatorV1, BaseEstimatorV2


class BasePrimitiveRunBuilder:
    """Base class for building primitives."""
    def __init__(
        self,
        primitive,
        circuits: List[QuantumCircuit],
        parameter_sets: List[List[float]],
    ):
        """
        Initializes BasePrimitiveRunBuilder for given attributes.

        Args:
            primitive (Union[SamplerValidType, EstimatorValidType]):
                The primitive to use for runs.
            circuits (List[QuantumCircuit]): The quantum circuits to run.
            parameter_sets (List[List[float]]): The parameters of the circuits.
        """
        self.primitive = primitive
        self.circuits = circuits
        self.parameter_sets = parameter_sets
        self.provenance = self.find_provenance()

    def find_provenance(self) -> Tuple[str, str, str]:
        """Gets the primitive's provenance based on its class and module."""
        for base_class in (BaseEstimatorV2, BaseEstimatorV1):
            if issubclass(self.primitive.__class__, base_class):
                base_estimator = base_class
                break
        else:
            raise ValueError(
                f"Primitive {
                    self.primitive.__class__.__name__} not recognized."
            )
        return (
            self.primitive.__class__.__module__.split(".")[0],
            self.primitive.__class__.__name__,
            base_estimator.__name__
        )

    def build_run(self):
        """
        Configures and returns primitive runs based on its provenance.

        Raises:
            NotImplementedError: If the primitive's is not supported.

        Returns:
            Union[PrimitiveJob, RuntimeJobV2]: A primitive job.
        """
        primitive_job = self._select_run_builder()
        return primitive_job()

    def _select_run_builder(self):
        """Selects the builder function based on the primitive's provenance."""
        raise NotImplementedError(
            "This method should be implemented by subclasses."
        )

    def _build_native_qiskit_run(self):
        """Builds a run function for a standard qiskit primitive."""
        raise NotImplementedError(
            "This method should be implemented by subclasses."
        )

    def _build_v2_run(self):
        """Builds a run function for aer and ibm-runtime V2 primitives."""
        raise NotImplementedError(
            "This method should be implemented by subclasses."
        )

    def _build_v1_run(self):
        """
        Primitives V1, which will be soon deprecated.

        Raises:
            NotImplementedError: Indicates that V1 will be soon deprecated.
        """
        raise NotImplementedError(
            "Primitives V1 are deprecated. Please, use V2 implementation."
        )
