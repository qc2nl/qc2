"""This module defines an ASE interface to Rose.

Original paper & official release:
https://pubs.acs.org/doi/10.1021/acs.jctc.0c00964

GitLab page:
https://gitlab.com/quantum_rose/rose

Note: see also https://pubs.acs.org/doi/10.1021/ct400687b.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Union, Sequence

from ase import Atoms
from ase.calculators.calculator import FileIOCalculator
from ase.io import write
from ase.units import Bohr

import os
import re
import numpy as np
import copy


class RoseCalcType(Enum):
    """Enumerator class defining the type of calculation done by Rose."""
    ATOM_FRAG = 'atom_frag'  # calc IAOs from atom-free AOs.
    MOL_FRAG = 'mol_frag'    # calc IFOs from free molecular fragment MOs.


class RoseInterfaceMOCalculators(Enum):
    """Enumerator class defining the program to which Rose is interfaced."""
    PYSCF = 'pyscf'
    PSI4 = 'psi4'
    DIRAC = 'dirac'
    ADF = 'adf'
    GAUSSIAN = 'gaussian'


class RoseIFOVersion(Enum):
    """Enumerator class defining the 'recipe' to obtain IFOs."""
    STNDRD_2013 = "Stndrd_2013"
    SIMPLE_2013 = "Simple_2013"
    SIMPLE_2014 = "Simple_2014"


class RoseILMOExponent(Enum):
    """Enumerator class defining the exponent used to obtain ILMOs."""
    TWO = 2
    THREE = 3
    FOUR = 4


@dataclass
class RoseInputDataClass:
    """A dataclass representing input options for Rose."""
    rose_target: Atoms
    rose_frags: Union[Atoms, Sequence[Atoms]]

    # calculate_mo: bool = True => made inactive
    rose_calc_type: RoseCalcType = RoseCalcType.ATOM_FRAG.value

    # run_postscf: bool = False => made inactive
    restricted: bool = True
    # openshell: bool = False => made inactive
    relativistic: bool = False
    spatial_orbitals: bool = True
    include_core: bool = False

    # Rose intrinsic options => expected to be fixed at the default values?
    version: RoseIFOVersion = RoseIFOVersion.STNDRD_2013.value
    exponent: RoseILMOExponent = RoseILMOExponent.FOUR.value
    spherical: bool = False
    uncontract: bool = True
    test: bool = False
    # wf_restart: bool = False => made inactive
    get_oeint: bool = False
    save: bool = True
    # avas_frag: Optional[List[int]] = field(default_factory=list) => made inactive
    # nmo_avas: Optional[List[int]] = field(default_factory=list) => made inactive

    # options for virtual orbitals localization.
    additional_virtuals_cutoff: Optional[float] = None  # 2.0  # Eh
    frag_threshold: Optional[float] = None  # 10.0  # Eh
    frag_valence: Optional[List[List[int]]] = field(default_factory=list)
    frag_core: Optional[List[List[int]]] = field(default_factory=list)
    frag_bias: Optional[List[List[int]]] = field(default_factory=list)

    def __post_init__(self):
        """This method is called after the instance is initialized."""
        if self.test:
            self.get_oeint = True

    """Description of the attributes to be added below."""


class Rose(RoseInputDataClass, FileIOCalculator):
    """A general ASE calculator for ROSE (Reduction of Orbital Space Extent).

    Args:
        RoseInputDataClass (RoseInputDataClass): Tailor-made base dataclass
            for Rose; see above.
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

        # define a list of filenames
        self.mol_frags_filenames: List[str] = []

    def calculate(self, *args, **kwargs) -> None:
        """Executes Rose workflow."""
        FileIOCalculator.calculate(self, *args, **kwargs)

        self.generate_input_genibo()
        self.generate_mol_frags_xyz()
        self.generate_mo_files()
        self.run_rose()

        if self.save:
            self.save_ibos()

    def generate_input_genibo(self) -> None:
        """Generates fortran input file for Rose.

        This method generates a file named "INPUT_GENIBO"
            containing Rose input options.
        """
        print(self.parameters)
        input_genibo_filename = "INPUT_GENIBO"

        with open(input_genibo_filename, "w") as f:
            f.write("**ROSE\n")
            f.write(".VERSION\n")
            f.write(self.version + "\n")
            f.write(".CHARGE\n")
            f.write(str(self.rose_target.calc
                        .parameters.charge) + "\n")
            f.write(".EXPONENT\n")
            f.write(str(self.exponent) + "\n")
            f.write(".FILE_FORMAT\n")
            f.write(self.rose_target.calc
                    .name.lower() + "\n")
            if self.test:
                f.write(".TEST\n")
            if not self.restricted:
                f.write(".UNRESTRICTED\n")
            if not self.spatial_orbitals:
                f.write(".SPINORS\n")
            if self.include_core:
                f.write(".INCLUDE_CORE\n")
            if self.rose_calc_type == RoseCalcType.MOL_FRAG.value:
                f.write(".NFRAGMENTS\n")
                f.write(str(len(self.rose_frags)) + "\n")
                if self.additional_virtuals_cutoff:
                    f.write(".ADDITIONAL_VIRTUALS_CUTOFF\n")
                    f.write(str(self.additional_virtuals_cutoff) + "\n")
                if self.frag_threshold:
                    f.write(".FRAG_THRESHOLD\n")
                    f.write(str(self.frag_threshold) + "\n")
                if self.frag_valence:
                    for item in self.frag_valence:
                        f.write(".FRAG_VALENCE\n")
                        f.write(str(item[0]) + "\n")
                        f.write(str(item[1]) + "\n")
                if self.frag_core:
                    for item in self.frag_core:
                        f.write(".FRAG_CORE\n")
                        f.write(str(item[0]) + "\n")
                        f.write(str(item[1]) + "\n")
                if self.frag_bias:
                    for item in self.frag_bias:
                        f.write(".FRAG_BIAS\n")
                        f.write(str(item[0]) + "\n")
                        f.write(str(item[1]) + "\n")
            f.write("\n*END OF INPUT\n")

    def generate_mol_frags_xyz(self) -> None:
        """Generates Molecule and Fragment xyz files for Rose.

        Note: cartesian coordinates in angstrom.
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
        # Create a set of calculator file extensions for molecule and fragments
        calculator_file_extensions = [
            self.rose_target.calc.name.lower(),
            *[frag.calc.name.lower() for frag in self.rose_frags]
            ]

        # Generate a set of expected MO file names
        mo_file_names = [
            f"{file}.{ext}" for file, ext in zip(self.mol_frags_filenames,
                                                 calculator_file_extensions)]
        
        # Check if all expected MO files already exist
        # if all(os.path.exists(file) for file in mo_file_names):
        #     print("MO files", mo_file_names, "exist.")
        #     print("Proceeding to the next step.")
        #     return

        # Generate a dictionary of filenames and corresponding ASE Atoms objects
        filename_atoms_dict = dict(zip(mo_file_names,
                                       [self.rose_target, *self.rose_frags]))
        
        for filename, atoms in filename_atoms_dict.items():
            # Generate MO file for each ASE Atoms object

            # Extract necessary information from the ASE Atoms object
            natom = len(atoms.symbols)
            nelec = (sum(atoms.numbers) - atoms.calc.parameters.charge)
            spin = atoms.calc.parameters.multiplicity - 1
            nalpha = (nelec + spin) // 2
            nbeta = (nelec - spin) // 2
            cart_coord = [pos / Bohr for pos in atoms.positions.flatten()]

            # Check Rose restrictions on spherical vs cartesian basis functions
            if self.spherical:
                raise ValueError("ROSE does not work with spherical basis functions.")
            pureamd = 1
            pureamf = 1

            # Extract necessary information from the ASE Atoms object
            # and associated PySCF objects

            # => pyscf specific variables # begin

            mol = atoms.calc.mol
            mf = atoms.calc.mf

            nao = mol.nao_cart()
            nshells = mol.nbas

            shell_atom_map = []
            orb_momentum = []
            contract_coeff = []

            for i in range(len(mol._bas)):
                shell_atom_map.append(mol._bas[i][0] + 1)
                orb_momentum.append(mol._bas[i][1])
                contract_coeff.append(mol._bas[i][3])

            nprim_shell = []
            coord_shell = []
            for i in range(nshells):
                nprim_shell.append(1)
                for j in range(3):
                    coord_shell.append(atoms.positions[
                            shell_atom_map[i] - 1][j]/Bohr)

            prim_exp = []
            for i in range(natom):
                atom_type = atoms.symbols[i]
                primitives_exp = mol._basis[atom_type]
                for j in range(len(primitives_exp)):
                    for k in range(1, len(primitives_exp[0])):
                        prim_exp.append(primitives_exp[j][k][0])

            scf_e = mf.e_tot

            alpha_MO = []
            beta_MO = []

            if self.restricted:
                alpha_coeff = mf.mo_coeff.copy()
                alpha_energies = mf.mo_energy.copy()
            if not self.restricted:
                alpha_coeff = mf.mo_coeff[0].copy()
                beta_coeff = mf.mo_coeff[1].copy()
                alpha_energies = mf.mo_energy[0].copy()
                beta_energies = mf.mo_energy[1].copy()

            for i in range(alpha_coeff.shape[1]):
                for j in range(alpha_coeff.shape[0]):
                    alpha_MO.append(alpha_coeff[j][i])

            if not self.restricted:
                for i in range(beta_coeff.shape[1]):
                    for j in range(beta_coeff.shape[0]):
                        beta_MO.append(beta_coeff[j][i])

            if self.get_oeint:
                E_core = scf_e - mf.energy_elec()[0]
                one_body_int = []
                if self.restricted:
                    h1e = alpha_coeff.T.dot(
                        mf.get_hcore()).dot(alpha_coeff)
                    h1e = mf.mo_coeff.T.dot(
                        mf.get_hcore()).dot(mf.mo_coeff)
                    for i in range(1, h1e.shape[0]+1):
                        for j in range(1, i+1):
                            one_body_int.append(h1e[i-1, j-1])
                else:
                    h1e_alpha = alpha_coeff.T.dot(
                        mf.get_hcore()).dot(alpha_coeff)
                    h1e_beta = beta_coeff.T.dot(
                        mf.get_hcore()).dot(beta_coeff)
                    h1e_alpha = mf.mo_coeff[0].T.dot(
                        mf.get_hcore()).dot(mf.mo_coeff[0])
                    h1e_beta = mf.mo_coeff[1].T.dot(
                        mf.get_hcore()).dot(mf.mo_coeff[1])
                    for i in range(1, h1e_alpha.shape[0]+1):
                        for j in range(1, i+1):
                            one_body_int.append(h1e_alpha[i-1, j-1])
                            one_body_int.append(h1e_beta[i-1, j-1])

            # => pyscf specific variables # end

            # start writing Rose input mo files
            with open(filename, "w") as f:
                f.write("{:13}{:10}\n"
                        .format("Generated by",
                                atoms.calc.name.upper()))
                write_int(f, "Number of atoms", natom)
                write_int(f, "Charge", atoms.calc.parameters.charge)
                write_int(f, "Multiplicity",
                          atoms.calc.parameters.multiplicity)
                write_int(f, "Number of electrons", nelec)
                write_int(f, "Number of alpha electrons", nalpha)
                write_int(f, "Number of beta electrons", nbeta)
                write_int(f, "Number of basis functions", nao)
                write_int_list(f, "Atomic numbers", atoms.numbers)
                write_singlep_list(f, "Nuclear charges", atoms.numbers)
                write_doublep_list(f, "Current cartesian coordinates",
                                   cart_coord)
                write_int(f, "Number of primitive shells", nshells)
                write_int(f, "Pure/Cartesian d shells", pureamd)
                write_int(f, "Pure/Cartesian f shells", pureamf)
                write_int_list(f, "Shell types", orb_momentum)
                write_int_list(f, "Number of primitives per shell",
                               nprim_shell)
                write_int_list(f, "Shell to atom map", shell_atom_map)
                write_singlep_list(f, "Primitive exponents", prim_exp)
                write_singlep_list(f, "Contraction coefficients",
                                   [1]*len(prim_exp))
                write_doublep_list(f, "Coordinates of each shell",
                                   coord_shell)
                f.write("{:43}R{:27.15e}\n".format("Total Energy", scf_e))
                write_doublep_list(f, "Alpha Orbital Energies",
                                   alpha_energies)
                write_doublep_list(f, "Alpha MO coefficients", alpha_MO)
                if not self.restricted:
                    write_doublep_list(f, "Beta Orbital Energies",
                                       beta_energies)
                    write_doublep_list(f, "Beta MO coefficients", beta_MO)
                if self.get_oeint:
                    f.write("{:43}R{:27.15e}\n".format(
                        "Core Energy", E_core))
                    write_doublep_list(f, "One electron integrals",
                                       one_body_int)

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

