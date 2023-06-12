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
from ase.calculators.calculator import InputError
from .dirac_io import write_dirac_in, read_dirac_out, _update_dict
from ase.io import write
from ase.units import Bohr

from qc2.data import read_hdf5, write_hdf5, generate_dict_for_qc2_schema


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
    command: str = "pam --inp=PREFIX.inp --mol=PREFIX.xyz --silent --get='MRCONEE MDCINT'"

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
        recognized_key_sections: List[str] = [
            'dirac', 'general', 'molecule',
            'hamiltonian', 'wave_function', 'analyse', 'properties',
            'visual', 'integrals', 'grid', 'moltra'
            ]

        # check any mispelling
        for key, value in self.parameters.items():
            if key not in recognized_key_sections:
                raise InputError('Keyword', key,
                                 ' not recognized. Please check input.')

        # set default parameters
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

        if 'integrals' not in self.parameters:
            # useful to compare with nonrel calc done with other programs
            self.parameters.update(integrals={'.nucmod': '1'})

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
        """Dumps molecular data to a HDF5 file using qc2 format."""
        # read dirac info
        dirac_hdf5_out_file = self.prefix + "_" + self.prefix + ".h5"
        data = read_hdf5(dirac_hdf5_out_file)

        # generate qc2 schema
        qc2_data = generate_dict_for_qc2_schema()

        qc2_data['/input/aobasis/1/descriptor']['value']  = 'DIRAC'
        qc2_data['/result/mobasis/1/descriptor']['value'] = 'DIRAC_scf'

        for label, item in qc2_data.items():
            if 'mobasis' in label:
                diraclabel = label.replace('mobasis/1','wavefunctions/scf/mobasis')
            else:
                diraclabel = label
            if item['use'] == 'required' and item['type'] != 'composite': 
                try:
                    qc2_data[label]['value'] = data[diraclabel]['value']
                except:
                    print('required data {} not found'.format(label))

        write_hdf5(filename, qc2_data)

    def load(self) -> None:
        pass

# functions below still under developments

    def get_mol_data(self) -> Dict[str, Any]:
        """Pulls out molecular data from dirac HDF5 output file."""

        #charge = self.mol.charge
        natom  = self._get_from_dirac_hdf5_file(
            '/input/molecule/n_atoms')[0]
        #nelec  = self.mol.nelectron
        #nalpha = None
        #nbeta  = None
        nao    = self._get_from_dirac_hdf5_file(
            '/input/aobasis/1/n_ao')[0]
        nshells= self._get_from_dirac_hdf5_file(
            '/input/aobasis/1/n_shells')
        #multiplicity = self.mol.spin + 1

        cart_coord = self._get_from_dirac_hdf5_file(
            '/input/molecule/geometry') / Bohr  #=> in a.u

        print(nao, nshells)
    
    def _get_from_dirac_hdf5_file(self, property_name):
        """Helper routine to open dirac HDF5 output file and take property."""
        out_hdf5_file = self.prefix + "_" + self.prefix + ".h5"
        try:
            with h5py.File("{}".format(out_hdf5_file), "r") as f:
                data = f[property_name][...]
        except (KeyError, IOError):
            data = None
        return data
