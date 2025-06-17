"""This modules contains child classes to build algorithm results."""
from typing import Optional, List
from qc2.algorithms.base.vqe_base_results import VQEBASEResults
from qc2.algorithms.base.base_algorithm_results import BaseAlgorithmResults

class QPEResults(BaseAlgorithmResults):
    """Extends VQEBASEResults to provide specialized data container for QPE."""
    def __init__(self) -> None:
        super().__init__()
        self._phase: Optional[float] = None
        self._eigenvalue: Optional[float] = None
        self._optimal_energy: Optional[float] = None

    @property
    def phase(self) -> float:
        """Returns the phase."""
        return self._phase

    @phase.setter
    def phase(self, value: float) -> None:
        """Sets the phase."""
        self._phase = value

    @property
    def eigenvalue(self) -> float:
        """Returns the eigenvalue."""
        return self._eigenvalue

    @eigenvalue.setter
    def eigenvalue(self, value: float) -> None:
        """Sets eigenvalue."""
        self._eigenvalue = value

    @property
    def optimal_energy(self) -> Optional[float]:
        """Returns optimal energy."""
        return self._optimal_energy

    @optimal_energy.setter
    def optimal_energy(self, value: float) -> None:
        """Sets optimal energy."""
        self._optimal_energy = value

class VQEResults(VQEBASEResults):
    """Extends VQEBASEResults to provide specialized data container for VQE."""
    def __init__(self) -> None:
        super().__init__()


class SAOOVQEResults():
    """Extends VQEResults to provide a results data container for oo-VQE."""
    def __init__(self, 
                 state_weights=None, 
                 energy=None,
                 circuit_parameters=None, 
                 orbital_parameters=None) -> None:
        super().__init__()
        self._energy: Optional[List] = energy
        self._circuit_parameters: Optional[List] = circuit_parameters
        self._orbital_parameters: Optional[List] = orbital_parameters
        self._state_weights: Optional[List] = state_weights
        self._optimal_phase: Optional[float] = None


    @property
    def optimal_energy(self) -> Optional[float]:
        """Returns optimal energy."""
        return self._energy[-1][0]

    @property
    def energy(self) -> Optional[List]:
        """Returns list of energies of all iterations."""
        return self._energy

    @energy.setter
    def energy(self, value: List) -> None:
        """Sets list of energies of all iterations."""
        self._energy = value

    @property
    def optimal_circuit_params(self) -> Optional[List]:
        """Returns the optimal circuit parameters."""
        return self._circuit_parameters[-1]

    @property
    def optimal_orbital_params(self) -> Optional[List]:
        """Returns optimal orbital parameters."""
        return self._orbital_parameters[-1]


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

    @property
    def state_weights(self) -> Optional[List]:
        """Returns optimal state weights."""
        return self._state_weights

    @state_weights.setter
    def state_weights(self, value: List) -> None:
        """Sets optimal state weights."""
        self._state_weights = value

    @property
    def optimal_phase(self) -> Optional[float]:
        """Returns optimal phase."""
        return self._optimal_phase

    @optimal_phase.setter
    def optimal_phase(self, value: float) -> None:
        """Sets optimal phase."""
        self._optimal_phase = value

    def update(self, theta: List, kappa: List, energy: List) -> None:
        """
        Updates the internal state with new circuit parameters, orbital parameters, 
        and energy values.

        Args:
            theta (List): New set of circuit parameters to be appended.
            kappa (List): New set of orbital parameters to be appended.
            energy (List): New energy values to be appended.
        """
        self._circuit_parameters.append(theta)
        self._orbital_parameters.append(kappa)
        self._energy.append(energy)	