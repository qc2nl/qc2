"""
This module implements the abstract base class for qc2 ase calculators.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Any, Union
import os
import h5py

from qiskit_nature.second_q.formats.qcschema import QCSchema
from qiskit_nature.second_q.formats.fcidump import FCIDump


class BaseQc2ASECalculator(ABC):
    """Abstract base class for the qc2 ASE calculators."""
    def __init__(self) -> None:

        self._implemented_formats = ["qcschema", "fcidump"]
        self._format = None
        self.format = "qcschema"

    @property
    def format(self) -> str:
        """Returs the format attribute."""
        return self._format

    @format.setter
    def format(self, schema_format) -> None:
        """Sets the format attribute."""
        if schema_format not in self._implemented_formats:
            raise ValueError(
                f"Attribute {schema_format} not recognized. "
                "Valid option are `qcschema` or `fcidump`"
            )
        self._format = schema_format

    @abstractmethod
    def save(self, datafile: str) -> None:
        """Dumps qchem data to a file using QCSchema or FCIDump formats."""

    def load(self, datafile: str) -> Union[QCSchema, FCIDump]:
        """Loads qchem data from a QCSchema- or FCIDump-formatted files."""
        # first check if the file exists
        if not os.path.exists(datafile):
            raise FileNotFoundError(f"{datafile} file not found!")

        # check if the file has a valid format
        if (self._format == "qcschema" and not h5py.is_hdf5(datafile)):
            raise ValueError(f"{datafile} is not an hdf5 file")
        
        # TODO: add checks for fcidump
        #if (self._format == "fcidump" and ....):
        #    raise ValueError(f"{datafile} is not an fcidump-formated file")

        # populating QCSchema or FCIDump dataclasses
        if (self._format == "qcschema"):
            with h5py.File(datafile, 'r') as file:
                return QCSchema._from_hdf5_group(file)
        elif (self._format == "fcidump"):
            return FCIDump.from_file(datafile)
        

    @abstractmethod
    def get_integrals_mo_basis(self) -> Tuple[Any, ...]:
        """Calculates core energy, one- and two-body integrals in MO basis."""
