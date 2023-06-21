from typing import Optional
import os

from ase import Atoms
from .schema import generate_empty_h5
# from .schema import generate_json_schema_file


class qc2Data:
    """Class docstring."""
    def __init__(self,
                 filename: str,
                 molecule: Optional[str],
                 ):
        """_summary_

        Args:
            molecule (Optional[str]): _description_
        """
        # generate JSON schema file from QC2schema plain text
        # json_file = os.path.join(os.path.dirname(__file__), 'qc2_schema.json')
        # generate_json_schema_file(json_file)

        json_file = os.path.join(
            os.path.dirname(__file__), 'qc_schema_output.schema'
            )

        # define attributes
        self._schema = json_file  # QCSchema
        self._filename = filename
        self._init_data_file()

        self._molecule = None
        self.molecule = molecule

    def _init_data_file(self):
        """initialize the hdf5 file containing the data."""
        generate_empty_h5(self._schema, self._filename)

    @property
    def molecule(self) -> Atoms:
        """Return the molecule.

        Returns:
            Molecule: molecular data
        """
        return self._molecule

    @molecule.setter
    def molecule(self, *args, **kwargs) -> None:
        """Set the molecule."""
        self._molecule = Atoms(*args, **kwargs)