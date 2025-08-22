"""This modules contains child classes to build algorithm results."""
from typing import Optional

class QPEResults:
    """Provide specialized data container for QPE."""
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