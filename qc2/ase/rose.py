"""This module defines an ASE interface to Rose.

Original paper & official release:
https://pubs.acs.org/doi/10.1021/acs.jctc.0c00964

GitLab page:
https://gitlab.com/quantum_rose/rose

Note: see also https://pubs.acs.org/doi/10.1021/ct400687b.
"""
from typing import Optional, List, Union, Sequence
import os

from ase import Atoms
from ase.calculators.calculator import FileIOCalculator
from ase.io import write

from .rose_dataclass import RoseInputDataClass, RoseCalcType
from .rose_io import write_rose_in


class Rose(RoseInputDataClass, FileIOCalculator):
    """A general ASE calculator for ROSE (Reduction of Orbital Space Extent).

    Args:
        RoseInputDataClass (RoseInputDataClass): Custom-made base dataclass
            for Rose; see rose_dataclass.py.
        FileIOCalculator (FileIOCalculator): Base class for ASE calculators
            that write/read input/output files.
    """
    implemented_properties: List[str] = []
    command: str = ''
    restart: bool = None
    ignore_bad_restart_file: bool = FileIOCalculator._deprecated
    label: str = 'rose'
    atoms: Atoms = None
    _directory = '.'

    def __init__(self, *args, **kwargs) -> None:
        """ASE-Rose Class Constructor to initialize the object.

        Give an example here on how to use....
        """
        # initializing base classes
        RoseInputDataClass.__init__(self, *args, **kwargs)
        FileIOCalculator.__init__(self, *args, **kwargs)

        # define a list of filenames; to be used later
        self.mol_frags_filenames: List[str] = []

    def calculate(self, *args, **kwargs) -> None:
        """Executes Rose workflow."""
        FileIOCalculator.calculate(self, *args, **kwargs)

        self.generate_input_genibo()
        self.generate_mol_frags_xyz()
        self.generate_input_mo_files()
        self.run_rose()

        if self.save:
            self.save_ibos()

    def generate_input_genibo(self) -> None:
        """Generates fortran input file for Rose.

        This method generates a file named "INPUT_GENIBO"
            containing Rose input options.
        """
        input_genibo_filename = "INPUT_GENIBO"
        write_rose_in(input_genibo_filename,
                      self, **self.parameters)

    def generate_mol_frags_xyz(self) -> None:
        """Generates molecule and fragment xyz files for Rose.

        Notes:
            Cartesian coordinates in angstrom.
        """
        # generate supermolecule xyz file using ASE write()
        mol_filename = "MOLECULE"
        write(mol_filename + ".XYZ", self.rose_target)
        self.mol_frags_filenames.append(mol_filename)

        # generate fragments xyz files
        for i, frag in enumerate(self.rose_frags):

            # comply with the required Rose files format
            if self.rose_calc_type == RoseCalcType.ATOM_FRAG.value:
                # use the Z number of the atom for filename
                frag_Z = frag.numbers[0]
                frag_filename = f"{frag_Z:03d}"
            else:
                # use the index of the fragment for filename
                frag_filename = f"frag{i}"

            # use ASE write() to generate files
            write(frag_filename + ".xyz", frag)
            self.mol_frags_filenames.append(frag_filename)

    def generate_input_mo_files(self) -> None:
        """Generates orbitals input files for Rose."""
        # create a set of calculator file extensions for molecule and fragments
        calculator_file_extensions = [
            self.rose_target.calc.name.lower(),
            *[frag.calc.name.lower() for frag in self.rose_frags]
            ]

        # generate a set of expected MO file names
        mo_file_names = [
            f"{file}.{ext}" for file, ext in zip(self.mol_frags_filenames,
                                                 calculator_file_extensions)]

        # if the expected MO files already exist, check if they can be found
        if self.target_mo_file_exists:
            if os.path.exists(mo_file_names[0]):
                print(f"{mo_file_names[0]} file found. Proceeding to run 'genibo.x' ...")
                return
            else:
                raise FileNotFoundError(f"{mo_file_names[0]} file not found")

        if self.frags_mo_files_exist:
            if all(os.path.exists(file) for file in mo_file_names[1:]):
                print(f"{mo_file_names[1:]} files found. Proceeding to run 'genibo.x' ...")
                return
            else:
                raise FileNotFoundError(f"{mo_file_names[1:]} files not found")

        # generate a dictionary of filenames and corresponding ASE Atoms objects
        filename_atoms_dict = dict(zip(mo_file_names,
                                       [self.rose_target, *self.rose_frags]))

        # generate ROSE MO file for each ASE Atoms object
        for filename, atoms in filename_atoms_dict.items():
            try:
                # check if this method is implemented
                atoms.calc.dump_mo_input_file_for_rose(filename)
            except AttributeError as e:
                print(f"Method 'dump_mo_input_file_for_rose'"
                      f" not implemented in {atoms.calc.__class__.__name__} calculator.")
                raise

    def run_rose(self) -> None:
        """Runs Rose executable 'genibo.x'."""
        self.rose_output_filename = "OUTPUT_ROSE"
        self.command = "genibo.x > " + self.rose_output_filename
        FileIOCalculator.calculate(self)

    def save_ibos(self) -> None:
        """Generates a checkpoint file ('ibo.chk') with the final IBOs."""
        ibo_input_filename = "ibo" + "." + str(self.rose_target.calc.name.lower())
        ibo_output_filename = "ibo.chk"
        atoms = self.rose_target
        # call a specific method implemented inside each calculator
        try:
            atoms.calc.dump_ibos_from_rose_to_chkfile(ibo_input_filename,
                                                      ibo_output_filename)
        except AttributeError as e:
            print(f"Method 'dump_ibos_from_rose_to_chkfile"
                  f" not implemented in {atoms.calc.__class__.__name__} calculator.")
            raise

    def dump_data_for_qc2() -> None:
        """Dumps molecular data to a HDF5 format file for qc2."""
        # Format to be specified....
        pass
