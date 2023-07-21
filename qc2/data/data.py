"""This module defines the main qc2 data class."""
from typing import Optional
import os

from ase import Atoms
from .schema import generate_empty_h5


class qc2Data:
    """Main qc2 class.

    It orchestrates classical qchem programs and
    python libraries for quantum computing.
    """
    def __init__(self,
                 filename: str,
                 molecule: Optional[str],
                 ):
        """_summary_

        Args:
            molecule (Optional[str]): _description_
        """
        # this version uses the JSON schema for quantum chemistry (QCSchema)
        # for more details, see https://molssi.org/software/qcschema-2/
        # 'qc_schema_output.schema' taken from
        # https://github.com/MolSSI/QCSchema/tree/master/qcschema/data/v2
        json_file = os.path.join(
            os.path.dirname(__file__), 'qc_schema_output.schema'
            )

        # define attributes
        self._schema = json_file
        self._filename = filename
        self._init_data_file()

        self._molecule = None
        self.molecule = molecule

    def _init_data_file(self):
        """Initialize empty hdf5 file following the QCSchema format."""
        generate_empty_h5(self._schema, self._filename)

    @property
    def molecule(self) -> Atoms:
        """Return the molecule.

        Returns:
            Molecule as an ASE Atoms object.
        """
        return self._molecule

    @molecule.setter
    def molecule(self, *args, **kwargs) -> None:
        """Set the molecule."""
        self._molecule = Atoms(*args, **kwargs)
