"""This module defines an ASE interface to Rose.

Original paper & official release:
https://pubs.acs.org/doi/10.1021/acs.jctc.0c00964

GitLab page:
https://gitlab.com/quantum_rose/rose

Note: see also https://pubs.acs.org/doi/10.1021/ct400687b.
"""
from typing import Optional, List, Union, Sequence

from ase import Atoms
from ase.calculators.calculator import FileIOCalculator
from ase.io import write
from ase.units import Bohr
from .rose_dataclass import RoseInputDataClass, RoseCalcType
from .rose_io import write_rose_in

import os
import re
import numpy as np
import copy


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
        #self.generate_mol_frags_xyz()
        #self.generate_mo_files()
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
        """Generates Molecule and Fragment xyz files for Rose.

        Note: Cartesian coordinates in angstrom.
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

    def generate_mo_files(self) -> None:
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
            atoms.calc.dump_mo_coeff_file_for_rose(filename)

    def run_rose(self) -> None:
        """Runs Rose executable 'genibo.x'."""
        self.rose_output_filename = "OUTPUT_ROSE"
        self.command = "genibo.x > " + self.rose_output_filename
        FileIOCalculator.calculate(self)

    def save_ibos(self) -> None:
        """Generates a checkpoint file ('ibo.chk') with the final IBOs."""
        ibo_input_filename = "ibo" + "." + str(self.rose_target.calc.name.lower())
        ibo_output_filename = "ibo.chk"

        try:
            with open(ibo_input_filename, "r") as f:
                nao = read_int("Number of basis functions", f)
                alpha_energies = read_real_list("Alpha Orbital Energies", f)
                alpha_IBO = read_real_list("Alpha MO coefficients", f)
                beta_energies = read_real_list("Beta Orbital Energies", f)
                beta_IBO = read_real_list("Beta MO coefficients", f)
        except FileNotFoundError:
            print("Cannot open", ibo_input_filename)

        # => pyscf specific variables # begin

        from pyscf.scf.chkfile import dump_scf

        mol = self.rose_target.calc.mol
        mf = self.rose_target.calc.mf

        ibo_wfn = copy.copy(mf)

        nmo = len(alpha_energies)
        if self.restricted:
            alpha_IBO_coeff = np.zeros((len(mf.mo_coeff), nao), dtype=float)
            beta_IBO_coeff = np.zeros((len(mf.mo_coeff), nao), dtype=float)
        else:
            alpha_IBO_coeff = np.zeros((len(mf.mo_coeff[0]), nao), dtype=float)
            beta_IBO_coeff = np.zeros((len(mf.mo_coeff[1]), nao), dtype=float)   

        ij = 0
        for i in range(nmo):
            for j in range(nao):
                alpha_IBO_coeff[j, i] = alpha_IBO[ij]
                if not self.restricted:
                    beta_IBO_coeff[j, i] = beta_IBO[ij]
                ij += 1

        if self.restricted:
            alpha_energy = np.zeros(len(mf.mo_energy), dtype=float)
            alpha_energy[:len(alpha_energies)] = alpha_energies
            ibo_wfn.mo_energy = alpha_energy
            ibo_wfn.mo_coeff = alpha_IBO_coeff
        else:
            alpha_energy = np.zeros(len(mf.mo_energy[0]), dtype=float)
            alpha_energy[:len(alpha_energies)] = alpha_energies
            beta_energy = np.zeros(len(mf.mo_energy[1]), dtype=float)
            beta_energy[:len(beta_energies)] = beta_energies
            ibo_wfn.mo_energy[0] = alpha_energy
            ibo_wfn.mo_energy[1] = beta_energy
            ibo_wfn.mo_coeff[0] = alpha_IBO_coeff
            ibo_wfn.mo_coeff[1] = beta_IBO_coeff

        e_tot = 0.0
        dump_scf(mol, ibo_output_filename,
                 e_tot, ibo_wfn.mo_energy, ibo_wfn.mo_coeff, ibo_wfn.mo_occ)

        # => pyscf specific variables # end


def write_int(f, text, var):
    """Writes an integer value to a file in a specific format.

    Args:
        f (file object): The file object to write to.
        text (str): A string of text to precede the integer value.
        var (int): The integer value to be written to the file.

    Returns:
        None
    """
    f.write("{:43}I{:17d}\n".format(text, var))


def write_int_list(f, text, var):
    """Writes a list of integers to a file in a specific format.

    Args:
        f (file object): The file object to write to.
        text (str): A string of text to precede the list of integers.
        var (list): The list of integers to be written to the file.

    Returns:
        None
    """
    f.write("{:43}{:3} N={:12d}\n".format(text, "I", len(var)))
    dim = 0
    buff = 6
    if (len(var) < 6):
        buff = len(var)
    for i in range((len(var)-1)//6+1):
        for j in range(buff):
            f.write("{:12d}".format(var[dim+j]))
        f.write("\n")
        dim = dim + 6
        if (len(var) - dim) < 6:
            buff = len(var) - dim


def write_singlep_list(f, text, var):
    """Writes a list of single-precision floating-point values to a file object.

    Args:
        f: A file object to write to.
        text (str): The text to be written before the list.
        var (list): The list of single-precision floating-point
            values to write.

    Returns:
        None
    """
    f.write("{:43}{:3} N={:12d}\n".format(text, "R", len(var)))
    dim = 0
    buff = 5
    if (len(var) < 5):
        buff = len(var)
    for i in range((len(var)-1)//5+1):
        for j in range(buff):
            f.write("{:16.8e}".format(var[dim+j]))
        f.write("\n")
        dim = dim + 5
        if (len(var) - dim) < 5:
            buff = len(var) - dim


def write_doublep_list(f, text, var):
    """Writes a list of double precision floating point numbers to a file.

    Args:
        f (file object): the file to write the data to
        text (str): a label or description for the data
        var (list): a list of double precision floating point
            numbers to write to file

    Returns:
        None
    """
    f.write("{:43}{:3} N={:12d}\n".format(text, "R", len(var)))
    dim = 0
    buff = 5
    if (len(var) < 5):
        buff = len(var)
    for i in range((len(var)-1)//5+1):
        for j in range(buff):
            f.write("{:24.16e}".format(var[dim+j]))
        f.write("\n")
        dim = dim + 5
        if (len(var) - dim) < 5:
            buff = len(var) - dim


def read_int(text, f):
    """Reads an integer value from a text file.

    Args:
        text (str): The text to search for in the file.
        f (file): The file object to read from.

    Returns:
        int: The integer value found in the file.
    """
    for line in f:
        if re.search(text, line):
            var = int(line.rsplit(None, 1)[-1])
            return var


def read_real(text, f):
    """Reads a floating-point value from a text file.

    Args:
        text (str): The text to search for in the file.
        f (file): The file object to read from.

    Returns:
        float: The floating-point value found in the file.
    """
    for line in f:
        if re.search(text, line):
            var = float(line.rsplit(None, 1)[-1])
            return var


def read_real_list(text, f):
    """Reads a list of floating-point values from a text file.

    Args:
        text (str): The text to search for in the file.
        f (file): The file object to read from.

    Returns:
        list: A list of floating-point values found in the file.
    """
    for line in f:
        if re.search(text, line):
            n = int(line.rsplit(None, 1)[-1])
            var = []
            for i in range((n-1)//5+1):
                line = next(f)
                for j in line.split():
                    var += [float(j)]
            return var

