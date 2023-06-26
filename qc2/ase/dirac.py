"""This module defines an ASE interface to DIRAC.

Official website:
https://www.diracprogram.org/

GitLab repo:
https://gitlab.com/dirac/dirac
"""

import subprocess
import os
from typing import Optional, List, Dict, Any
import h5py

from ase import Atoms
from ase.calculators.calculator import FileIOCalculator
from ase.calculators.calculator import InputError, CalculationFailed
from ase.units import Bohr
from ase.io import write
from .dirac_io import write_dirac_in, read_dirac_out, _update_dict


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
    command: str = 'pam --inp=PREFIX.inp --mol=PREFIX.xyz --silent --get="AOMOMAT MRCONEE MDCINT"'

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

        #if '.4index' not in self.parameters['dirac']:
        #    self.parameters['dirac'].update({'.4index': ''})

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
        """Dumps molecular data to an HDF5 file.

        Notes:
            HDF5 files are written following the QCSchema.
        """

        # Open the HDF5 file in read/write mode
        file = h5py.File(filename, "w")

        # molecule group
        molecule = file.create_group("molecule")

        symbols = self._get_from_dirac_hdf5_file(
           '/input/molecule/symbols'
           )
        molecule.attrs['symbols'] = symbols

        geometry = self._get_from_dirac_hdf5_file(
           '/input/molecule/geometry') / Bohr  # => in a.u
        molecule.attrs['geometry'] = geometry
        
        # include here charge/multiplicity ?

        molecule.attrs['schema_name'] = "qcschema_molecule"
        molecule.attrs['schema_version'] = 2

        # properties group
        properties = file.create_group("properties")

        nbasis = self._get_from_dirac_hdf5_file(
            '/input/aobasis/1/n_ao')
        properties.attrs['calcinfo_nbasis'] = nbasis

        nmo = self._get_from_dirac_hdf5_file(
           '/result/wavefunctions/scf/mobasis/n_mo')
        properties.attrs['calcinfo_nmo'] = nmo

        natom = self._get_from_dirac_hdf5_file(
           '/input/molecule/n_atoms')
        properties.attrs['calcinfo_natom'] = natom

        energy = self._get_from_dirac_hdf5_file(
           '/result/wavefunctions/scf/energy')
        properties.attrs['return_energy'] = energy

        # model group
        model = file.create_group("model")

        basis = self.parameters['molecule']['*basis']['.default']
        model.attrs['basis'] = basis

        method = list(self.parameters['wave_function'].keys())[-1].strip('.')
        model.attrs['method'] = method

        # provenance group
        provenance = file.create_group("provenance")
        provenance.attrs['creator'] = self.name
        provenance.attrs['version'] = "2"
        provenance.attrs['routine'] = f"ASE-{self.__class__.__name__}.save()"

        # keywords group
        provenance = file.create_group("keywords")

        # driver group --- still not working
        qc_input = file.create_group("qcinput")
        qc_input.attrs['driver'] = "energy"

        file.close()

    def load(self) -> None:
        pass

    def get_integrals_data(self) -> Dict[str, Any]:
        """Retrieve one- and two-electron integrals from DIRAC fcidump file.
        
        Notes:
            Requires MRCONEE MDCINT files obtained using
            **DIRAC .4INDEX and 'pam ... --get="MRCONEE MDCINT"' options
        """
        command = "dirac_mointegral_export.x fcidump"
        try:
            proc = subprocess.Popen(command, shell=True, cwd=self.directory)
        except OSError as err:
            msg = 'Failed to execute "{}"'.format(command)
            raise EnvironmentError(msg) from err

        errorcode = proc.wait()

        if errorcode:
            path = os.path.abspath(self.directory)
            msg = ('Calculator "{}" failed with command "{}" failed in '
                   '{} with error code {}'.format(self.name, command,
                                                  path, errorcode))
            raise CalculationFailed(msg)
    
        if os.path.exists("FCIDUMP"):
             E_core = 0
             spinor = {}
             one_body_int = {}
             two_body_int = {}
             num_lines = sum(1 for line in open("FCIDUMP"))
             with open('FCIDUMP') as f:
               start_reading=0
               for line in f:
                 start_reading+=1
                 if "&END" in line:
                   break
               listed_values = [[token for token in line.split()] for line in f.readlines()] 
               complex_int = False
               if (len(listed_values[0]) == 6) : complex_int = True
               if not complex_int:
                  for row in range(num_lines-start_reading):
                    a_1 = int(listed_values[row][1])
                    a_2 = int(listed_values[row][2])
                    a_3 = int(listed_values[row][3])
                    a_4 = int(listed_values[row][4])
                    if a_4 == 0 and a_3 == 0:
                      if a_2 == 0:
                        if a_1 == 0:
                          E_core = float(listed_values[row][0])
                        else:
                          spinor[a_1] = float(listed_values[row][0])
                      else:
                        one_body_int[a_1,a_2] = float(listed_values[row][0])
                    else:
                      two_body_int[a_1,a_2,a_3,a_4] = float(listed_values[row][0])
               if complex_int:
                  for row in range(num_lines-start_reading):
                    a_1 = int(listed_values[row][2])
                    a_2 = int(listed_values[row][3])
                    a_3 = int(listed_values[row][4])
                    a_4 = int(listed_values[row][5])
                    if a_4 == 0 and a_3 == 0:
                      if a_2 == 0:
                        if a_1 == 0:
                          E_core = complex(
                             float(listed_values[row][0]),float(listed_values[row][1]))
                        else:
                          spinor[a_1] = complex(
                             float(listed_values[row][0]),float(listed_values[row][1]))
                      else:
                        one_body_int[a_1,a_2] = complex(
                           float(listed_values[row][0]),float(listed_values[row][1]))
                    else:
                      two_body_int[a_1,a_2,a_3,a_4] = complex(
                         float(listed_values[row][0]),float(listed_values[row][1]))
             # self.n_orbitals = len(self.spinor)
             # self.n_qubits = len(self.spinor)
        else:
             print('FCIDUMP not found.')
        
        ei_data = {}
        ei_data = {'E_core': E_core,
                   'spinor': spinor,
                   'one_body_int': one_body_int,
                   'two_body_int': two_body_int}

        return ei_data

# functions below still under developments
#
    #def old_save(self, filename: str) -> None:
    #    """Dumps molecular data to a HDF5 file using qc2 format."""
    #    # read dirac info
    #    dirac_hdf5_out_file = self.prefix + "_" + self.prefix + ".h5"
    #    data = read_hdf5(dirac_hdf5_out_file)
#
    #    # generate qc2 schema
    #    qc2_data = generate_dict_from_text_schema()
#
    #    qc2_data['/input/aobasis/1/descriptor']['value']  = 'DIRAC'
    #    qc2_data['/result/mobasis/1/descriptor']['value'] = 'DIRAC_scf'
#
    #    for label, item in qc2_data.items():
    #        if 'mobasis' in label:
    #            diraclabel = label.replace('mobasis/1','wavefunctions/scf/mobasis')
    #        else:
    #            diraclabel = label
    #        if item['use'] == 'required' and item['type'] != 'composite': 
    #            try:
    #                qc2_data[label]['value'] = data[diraclabel]['value']
    #            except:
    #                print('required data {} not found'.format(label))
#
    #    write_hdf5(filename, qc2_data)
#
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
        """Helper routine to open dirac HDF5 output and extract desired property."""
        out_hdf5_file = self.prefix + "_" + self.prefix + ".h5"
        try:
            with h5py.File("{}".format(out_hdf5_file), "r") as f:
                data = f[property_name][...]
        except (KeyError, IOError):
            data = None
        return data
