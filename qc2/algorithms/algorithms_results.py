"""This modules contains child classes to build algorithm results."""
from typing import Optional, List
from qc2.algorithms.base.vqe_base_results import VQEBASEResults


class VQEResults(VQEBASEResults):
    """Extends VQEBASEResults to provide specialized data container for VQE."""
    def __init__(self) -> None:
        super().__init__()


class OOVQEResults(VQEResults):
    """Extends VQEResults to provide a results data container for oo-VQE."""
    def __init__(self) -> None:
        super().__init__()
        self._optimal_circuit_params: Optional[List] = None
        self._optimal_orbital_params: Optional[List] = None
        self._circuit_parameters: Optional[List] = None
        self._orbital_parameters: Optional[List] = None

    @property
    def optimal_circuit_params(self) -> Optional[List]:
        """Returns the optimal circuit parameters."""
        return self._optimal_circuit_params

    @optimal_circuit_params.setter
    def optimal_circuit_params(self, value: List) -> None:
        """Sets optimal circuit parameters."""
        self._optimal_circuit_params = value

    @property
    def optimal_orbital_params(self) -> Optional[List]:
        """Returns optimal orbital parameters."""
        return self._optimal_orbital_params

    @optimal_orbital_params.setter
    def optimal_orbital_params(self, value: List) -> None:
        """Sets optimal orbital parameters."""
        self._optimal_orbital_params = value

    @property
    def circuit_parameters(self) -> Optional[List]:
        """Returns list with circuit parameters of all iteration."""
        return self._circuit_parameters

    @circuit_parameters.setter
    def circuit_parameters(self, value: List) -> None:
        """Sets list with circuit parameters of all iteration."""
        self._circuit_parameters = value

    @property
    def orbital_parameters(self) -> Optional[List]:
        """Returns list with orbital parameters of all iteration."""
        return self._orbital_parameters

    @orbital_parameters.setter
    def orbital_parameters(self, value: List) -> None:
        """Sets list with parameters of all iteration."""
        self._orbital_parameters = value
