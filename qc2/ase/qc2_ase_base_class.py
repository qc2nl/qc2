"""This module implements the abstract base class for qc2 ase calculators."""
from abc import ABC
from typing import Tuple, Any, Union
import os
import h5py

from ..qc2schema.qcschema import (
    QCSchema, QCTopology, QCProperties,
    QCModel, QCProvenance, QCWavefunction
)

class BaseQc2ASECalculator(ABC):
    """Abstract base class for all qc2 ASE calculators."""
    def __init__(self) -> None:
        """:class:`BaseQc2ASECalculator` class constructor.

        Astract base class for all qc2 ASE calculators.
        """
        # format in which to read/write qchem data
        self._implemented_formats = ["qcschema", "fcidump"]
        self._schema_format = None
        self.schema_format = "qcschema"

    @property
    def schema_format(self) -> str:
        """Returs the format attribute."""
        return self._schema_format

    @schema_format.setter
    def schema_format(self, format) -> None:
        """Sets the format attribute."""
        if format not in self._implemented_formats:
            raise ValueError(
                f"Format {format} not recognized. "
                "Valid options are `qcschema` or `fcidump`"
            )
        self._schema_format = format

    def save(self, datafile: Union[h5py.File, str]) -> None:
        """Dumps qchem data to a datafile using QCSchema or FCIDump formats."""
        raise NotImplementedError("Subclasses should implement this method.")

    def load(self, datafile: Union[h5py.File, str]) -> QCSchema:
        """Loads qchem data from a QCSchema- or FCIDump-formatted datafile."""
        # first check if the file exists
        if not os.path.exists(datafile):
            raise FileNotFoundError(f"{datafile} file not found!")

        # check if the file has a valid format
        if (self._schema_format == "qcschema" and not h5py.is_hdf5(datafile)):
            raise ValueError(f"{datafile} is not an hdf5 file.")

        with h5py.File(datafile, 'r') as file:
            return QCSchema._from_hdf5_group(file)

    def get_integrals_mo_basis(self) -> Tuple[Any, ...]:
        """Calculates one- and two-body integrals in MO basis."""
        raise NotImplementedError("Subclasses should implement this method.")

    def get_integrals_ao_basis(self) -> Tuple[Any, ...]:
        """Calculates one- and two-electron integrals in AO basis."""
        raise NotImplementedError("Subclasses should implement this method.")

    def get_molecular_orbitals_coefficients(self) -> Tuple[Any, ...]:
        """Reads alpha and beta molecular orbital coefficients."""
        raise NotImplementedError("Subclasses should implement this method.")

    def get_molecular_orbitals_energies(self) -> Tuple[Any, ...]:
        """Reads alpha and beta orbital energies."""
        raise NotImplementedError("Subclasses should implement this method.")

    def get_overlap_matrix(self) -> Tuple[Any, ...]:
        r"""Reads overlap matrix.

        Notes:
            Can also be evaluated on-the-fly as:

            .. math::
                S = (C^{T})^{-1} \times C^{-1}

            where, :math:`C` is the matrix of mo coefficients.

        """
        raise NotImplementedError("Subclasses should implement this method.")

    def instantiate_qcschema(self, *args, **kwargs) -> QCSchema:
        """Creates an instance of QCSchema dataclass."""
        return QCSchema(*args, **kwargs)

    def instantiate_qctopology(self, *args, **kwargs) -> QCTopology:
        """Creates an instance of QCTopology dataclass."""
        return QCTopology(*args, **kwargs)

    def instantiate_qcproperties(self, *args, **kwargs) -> QCProperties:
        """Creates an instance of QCProperties dataclass."""
        return QCProperties(*args, **kwargs)

    def instantiate_qcmodel(self, *args, **kwargs) -> QCModel:
        """Creates an instance of QCModel dataclass."""
        return QCModel(*args, **kwargs)

    def instantiate_qcprovenance(self, *args, **kwargs) -> QCProvenance:
        """Creates an instance of QCProvenance dataclass."""
        return QCProvenance(*args, **kwargs)

    def instantiate_qcwavefunction(self, *args, **kwargs) -> QCWavefunction:
        """Creates an instance of QCProvenance dataclass."""
        return QCWavefunction(*args, **kwargs)
