"""This module defines an ASE interface to DIRAC.

Official website:
https://www.diracprogram.org/

GitLab repo:
https://gitlab.com/dirac/dirac
"""
import h5py
from typing import Optional, List, Dict, Any

from ase import Atoms
from ase.calculators.calculator import FileIOCalculator
from .dirac_io import write_dirac_in, read_dirac_out, _update_dict
from ase.io import write




class DIRAC(FileIOCalculator):
    """A general ASE calculator for the relativistic qchem DIRAC code.

    Args:
        FileIOCalculator (FileIOCalculator): Base class for calculators
            that write/read input/output files.

    Example of a typical ASE-DIRAC input:

    >>> from ase import Atoms
    >>> from ase.build import molecule
    >>> from qc2.ase.dirac import DIRAC
    >>>
    >>> molecule = Atoms(...) or molecule = molecule('...')
    >>> molecule.calc = DIRAC(dirac={}, wave_function={}...)
    >>> energy = molecule.get_potential_energy()
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
        """ASE-DIRAC Class Constructor to initialize the object.
        
        Args:
            restart (bool, optional): Prefix for restart file.
                May contain a directory. Defaults to None: don't restart.
            ignore_bad_restart (bool, optional): Deprecated and will
                stop working in the future. Defaults to False.
            label (str, optional): Calculator name. Defaults to 'dirac'.
            atoms (Atoms, optional): Atoms object to which the calculator
                will be attached. When restarting, atoms will get its
                positions and unit-cell updated from file. Defaults to None.
            command (str, optional): Command used to start calculation.
                Defaults to None.
            directory (str, optional): Working directory in which
                to perform calculations. Defaults to '.'.
        """
        # initializing base class Calculator.
        # see ase/ase/calculators/calculator.py.
        super().__init__(restart, ignore_bad_restart_file,
                         label, atoms, command, **kwargs)
        """Transforms **kwargs into a dictionary with calculation parameters.

        Starting with (attr1=value1, attr2=value2, ...)
            it creates self.parameters['attr1']=value1, and so on.
        """
        self.prefix = self.label

        # Check self.parameters input keys and values
        self.check_dirac_attributes()

    def check_dirac_attributes(self) -> None:
        """Checks for any missing and/or mispelling DIRAC input attribute.

        Notes:
            it can also be used to eventually set specific
            options in the near future.
        """
        # here, we could check any mispelling attribute |
        #                                               |
        #                                            <--|

        # setting up default parameters
        if 'dirac' not in self.parameters:
            key = 'dirac'
            value = {'.title': 'DIRAC-ASE calculation',
                     '.wave function': ''}
            # **DIRAC heading must always come first in the dict/input
            self.parameters = _update_dict(self.parameters, key, value)

        if 'hamiltonian' not in self.parameters:
            self.parameters.update(hamiltonian={'.levy-leblond': ''})

        if 'wave_function' not in self.parameters:
            self.parameters.update(wave_function={'.scf': ''})

        if 'molecule' not in self.parameters:
            self.parameters.update(molecule={'*basis': {'.default': 'sto-3g'}})

        if '*charge' not in self.parameters:
            self.parameters['molecule']['*charge']={'.charge': '0'}

    def calculate(self, *args, **kwargs) -> None:
        """Execute DIRAC workflow."""
        super().calculate(*args, **kwargs)

    def write_input(
            self,
            atoms: Optional[Atoms] = None,
            properties: Optional[List[str]] = None,
            system_changes: Optional[List[str]] = None
            ) -> None:
        """Generate all necessary inputs for DIRAC."""
        super().write_input(atoms, properties, system_changes)

        # generate xyz geometry file
        xyz_file = self.prefix + ".xyz"
        write(xyz_file, atoms)

        # generate DIRAC inp file
        inp_file = self.prefix + ".inp"
        write_dirac_in(inp_file, **self.parameters)

    def read_results(self):
        """Read energy from DIRAC output file."""
        out_file = self.prefix + "_" + self.prefix + ".out"
        output = read_dirac_out(out_file)
        self.results = output

    def save(self, filename: str) -> None:
        """Dump molecular data into a HDF5 file."""
        # Format to be specified

        print(self.atoms.numbers)

        with h5py.File("{}".format(filename), "w") as f:
            # Save geometry:
            d_geom = f.create_group("geometry")

            #data = file["geometry/atoms"]
            #print(data.keys())


    def load(self) -> None:
        pass


if __name__ == '__main__':
    
    h2_molecule = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.7]])
    
    h2_molecule.calc = DIRAC(dirac={'.wave function': ''},
                             wave_function={'.scf': ''},
                             molecule={'*basis': {'.default': 'sto-3g'}}
                             )

    print(h2_molecule.get_potential_energy())

    print(h2_molecule.calc.parameters.molecule)
