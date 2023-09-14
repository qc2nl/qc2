"""
This module implements the abstract base class for qc2 ase calculators.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Any, Union
import os
import h5py

from qiskit_nature.second_q.formats.qcschema import (
    QCSchema, QCTopology, QCProperties,
    QCModel, QCProvenance, QCWavefunction
)
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
                f"Format {schema_format} not recognized. "
                "Valid options are `qcschema` or `fcidump`"
            )
        self._format = schema_format

    def save(self, datafile: Union[h5py.File, str]) -> None:
        """Dumps qchem data to a file using QCSchema or FCIDump formats."""

    def load(self, datafile: str) -> Union[QCSchema, FCIDump]:
        """Loads qchem data from a QCSchema- or FCIDump-formatted files."""
        # first check if the file exists
        if not os.path.exists(datafile):
            raise FileNotFoundError(f"{datafile} file not found!")

        # check if the file has a valid format
        if (self._format == "qcschema" and not h5py.is_hdf5(datafile)):
            raise ValueError(f"{datafile} is not an hdf5 file")

        # add here checks for fcidump...
        # if (self._format == "fcidump" and ....):
        #     raise ValueError(f"{datafile} is not an fcidump-formated file")

        # populating QCSchema or FCIDump dataclasses
        if self._format == "fcidump":
            return FCIDump.from_file(datafile)

        with h5py.File(datafile, 'r') as file:
            return QCSchema._from_hdf5_group(file)

    def get_integrals_mo_basis(self) -> Tuple[Any, ...]:
        """Calculates core energy, one- and two-body integrals in MO basis."""
        return NotImplementedError

    def get_integrals_ao_basis(self) -> Tuple[Any, ...]:
        """Calculates one- and two-electron integrals in AO basis."""
        return NotImplementedError

    def get_molecular_orbitals_coefficients(self) -> Tuple[Any, ...]:
        """Reads alpha and beta molecular orbital coefficients."""
        return NotImplementedError

    def get_molecular_orbitals_energies(self) -> Tuple[Any, ...]:
        """Reads alpha and beta orbital energies."""
        return NotImplementedError

    def get_overlap_matrix(self) -> Tuple[Any, ...]:
        """Reads overlap matrix."""
        return NotImplementedError

    def instantiate_qcschema(self, *args, **kwargs) -> QCSchema:
        """Creates an instance of QCSchema dataclass"""
        return QCSchema(*args, **kwargs)

    def instantiate_qctopology(self, *args, **kwargs) -> QCTopology:
        """Creates an instance of QCTopology dataclass"""
        return QCTopology(*args, **kwargs)

    def instantiate_qcproperties(self, *args, **kwargs) -> QCProperties:
        """Creates an instance of QCProperties dataclass"""
        return QCProperties(*args, **kwargs)

    def instantiate_qcmodel(self, *args, **kwargs) -> QCModel:
        """Creates an instance of QCModel dataclass"""
        return QCModel(*args, **kwargs)

    def instantiate_qcprovenance(self, *args, **kwargs) -> QCProvenance:
        """Creates an instance of QCProvenance dataclass"""
        return QCProvenance(*args, **kwargs)

    def instantiate_qcwavefunction(self, *args, **kwargs) -> QCWavefunction:
        """Creates an instance of QCProvenance dataclass"""
        return QCWavefunction(*args, **kwargs)
