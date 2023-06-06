from ase import Atoms
from typing import Optional 
from .schema import generate_empty_h5
import os

class qc2Data:

    def __init__(self,
                filename: str,  
                molecule: Optional[str],
                ):
        """_summary_

        Args:
            molecule (Optional[str]): _description_
        """
        self._schema = os.path.join(
            os.path.dirname(__file__), 'qc2_schema.json')
        self._filename = filename 
        self._init_data_file()

        self._molecule = None
        self.molecule = molecule

    def _init_data_file(self):
        """initialize the hdf5 file containing the data
        """
        generate_empty_h5(self._schema, self._filename)

    @property
    def molecule(self) -> Atoms:
        """Return the molecule 

        Returns:
            Molecule: molecular data
        """
        return self._molecule
    
    @molecule.setter
    def molecule(self, *args, **kwargs) -> None:
        """Set the molecule
        """
        self._molecule = Atoms(*args, **kwargs)