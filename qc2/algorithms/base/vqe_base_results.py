"""This module defines the base class for VQE algorithms results."""
from typing import Optional, List, Dict
from qc2.algorithms.base.base_algorithm_results import BaseAlgorithmResults


class VQEBASEResults(BaseAlgorithmResults):
    """VQE result base class."""

    def __init__(self) -> None:
        super().__init__()
        self._optimizer_evals: Optional[int] = None
        self._optimal_params: Optional[List] = None
        self._optimal_energy: Optional[float] = None
        self._energy: Optional[List] = None
        self._parameters: Optional[List] = None
        self._metadata: Optional[Dict] = None

    @property
    def optimizer_evals(self) -> Optional[int]:
        """Returns number of optimizer evaluations."""
        return self._optimizer_evals

    @optimizer_evals.setter
    def optimizer_evals(self, value: int) -> None:
        """Sets number of optimizer evaluations."""
        self._optimizer_evals = value

    @property
    def optimal_params(self) -> Optional[List]:
        """Returns optimal parameters."""
        return self._optimal_params

    @optimal_params.setter
    def optimal_params(self, value: List) -> None:
        """Sets optimal parameters."""
        self._optimal_params = value

    @property
    def optimal_energy(self) -> Optional[float]:
        """Returns optimal energy."""
        return self._optimal_energy

    @optimal_energy.setter
    def optimal_energy(self, value: float) -> None:
        """Sets optimal energy."""
        self._optimal_energy = value

    @property
    def energy(self) -> Optional[List]:
        """Returns list of energies of all iterations."""
        return self._energy

    @energy.setter
    def energy(self, value: List) -> None:
        """Sets list of energies of all iterations."""
        self._energy = value

    @property
    def parameters(self) -> Optional[List]:
        """Returns list with parameters of all iteration."""
        return self._parameters

    @parameters.setter
    def parameters(self, value: List) -> None:
        """Sets list with parameters of all iteration."""
        self._parameters = value

    @property
    def metadata(self) -> Optional[Dict]:
        """Returns dict with metadata of all iterations."""
        return self._metadata

    @metadata.setter
    def metadata(self, value: Dict) -> None:
        """Sets dict with metadata of all iterations."""
        self._metadata = value
