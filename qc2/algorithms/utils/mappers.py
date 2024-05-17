"""This module creates a wrapper around qiskit-nature mappers."""
from qiskit_nature.second_q.mappers import (
    QubitMapper,
    JordanWignerMapper,
    BravyiKitaevMapper
)


class FermionicToQubitMapper:
    _mappers = {
        'JW': JordanWignerMapper,
        'BK': BravyiKitaevMapper
    }

    @classmethod
    def register_mapper(cls, key: str, mapper: QubitMapper):
        """
        Register a new mapper class with a given key.

        Args:
            key (str): The key (name) for the mapper.
            mapper (QubitMapper): The mapper class to be registered.

        Raises:
            ValueError: If the key is already registered.

        **Example**

        >>> from qiskit_nature.second_q.mappers import TaperedQubitMapper
        >>> from qc2.algorithms.utils import FermionicToQubitMapper
        >>>
        >>> FermionicToQubitMapper.register_mapper('TQM', TaperedQubitMapper)
        >>> mapper = FermionicToQubitMapper.from_string('tqm')
        """
        if key.upper() in cls._mappers:
            raise ValueError(f"Mapper '{key}' is already registered.")
        cls._mappers[key.upper()] = mapper

    @classmethod
    def from_string(cls, s):
        """
        Retrieve the mapper class corresponding to the provided string.

        Args:
            s (str): The string identifier of the mapper. Case-insensitive.

        Returns:
            QubitMapper: The corresponding mapper class.

        Raises:
            ValueError: If the provided string does not correspond
                to any known mapper.

        **Example**

        >>> from qc2.algorithms.utils import FermionicToQubitMapper
        >>> mapper = FermionicToQubitMapper.from_string('jw'))
        """
        try:
            return cls._mappers[s.upper()]
        except KeyError as err:
            raise ValueError(f"No mapper found for '{s}'") from err

