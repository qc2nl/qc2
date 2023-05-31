"""This module defines an ASE interface to DIRAC.

Official website:
https://www.diracprogram.org/

GitLab repo:
https://gitlab.com/dirac/dirac
"""
from typing import Optional, List, Dict, Any

from ase import Atoms
from ase.calculators.calculator import FileIOCalculator
from .dirac_io import write_dirac_in, read_dirac_out
from ase.io import write


class DIRAC(FileIOCalculator):
    """A general ASE calculator for the relativistic qchem DIRAC code.

    Args:
        FileIOCalculator (FileIOCalculator): Base class for calculators
            that write/read input/output files.
    """
    implemented_properties: List[str] = ['energy']
    label: str = 'dirac'
    command: str = "pam --inp=PREFIX.inp --mol=PREFIX.xyz --silent"

    def __init__(self,
                 restart: Optional[bool] = None,
                 ignore_bad_restart_file:
                 Optional[bool] = FileIOCalculator._deprecated,
                 label: Optional[str] = None,
                 atoms: Optional[Atoms] = None,
                 command: Optional[str] = None,
                 **kwargs) -> None:
        """ASE-DIRAC Class Constructor to initialize the object."""
        super().__init__(restart, ignore_bad_restart_file,
                         label, atoms, command, **kwargs)
        """Transforms **kwargs into a dictionary with calculation parameters.

        Starting with (attr1=value1, attr2=value2, ...)
            it creates self.parameters['attr1']=value1, and so on.
        """
        self.prefix: str = 'DIRAC'

    def calculate(self, *args, **kwargs) -> None:
        """Executes DIRAC workflow."""
        super().calculate(*args, **kwargs)

    def write_input(
            self,
            atoms: Optional[Atoms] = None,
            properties: Optional[List[str]] = None,
            system_changes: Optional[List[str]] = None
            ) -> None:
        """Generates all necessary inputs for DIRAC."""
        super().write_input(atoms, properties, system_changes)

        # generate xyz geometry file
        xyz_file = self.prefix + ".xyz"
        write(xyz_file, atoms)

        # generate DIRAC inp file
        inp_file = self.prefix + ".inp"
        write_dirac_in(inp_file, **self.parameters)

    def read_results(self):
        """Read energy, forces, ... from output file."""
        out_file = self.prefix + "_" + self.prefix + ".out"
        output = read_dirac_out(out_file)
        self.results = output

    def dump_data_for_qc2() -> None:
        """Dumps molecular data to a HDF5 format file for qc2."""
        # Format to be specified....
        pass

    def dump_ibos_from_rose_to_chkfile(self,
                                       input_file: str = "ibo.pyscf",
                                       output_file: str = "ibo.chk") -> None:
        """Saves calculated ROSE IBOs to a checkpoint file."""
        pass

    def dump_mo_input_file_for_rose(self, output_file: str) -> None:
        """Writes molecular orbitals input file for ROSE."""
        pass

if __name__ == '__main__':
    
    h2_molecule = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.7]])
    
    h2_molecule.calc = DIRAC(dirac={'.wave function': ''},
                             wave_function={'.scf': ''},
                             molecule={'*basis': {'.default': 'sto-3g'}}
                             )

    print(h2_molecule.get_potential_energy())
