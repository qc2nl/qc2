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
    avas_frag: Optional[List[int]] = field(default_factory=list)
    nmo_avas: Optional[List[int]] = field(default_factory=list)

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

        if self.avas_frag:
            self.generate_input_avas()

        self.generate_mol_frags_xyz()
        self.generate_mo_files()
        self.run_rose()

        if self.save:
            self.save_ibos()

        if self.avas_frag:
            self.run_avas()

        # if self.run_postscf:
        #     self.run_post_hf()

    def generate_input_genibo(self) -> None:
        """Generates fortran input file for Rose.

        This method generates a file named "INPUT_GENIBO"
            containing Rose input options.
        """
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
            if self.avas_frag:
                f.write(".AVAS\n")
                f.write(str(len(self.avas_frag)) + "\n")
                f.writelines("{:3d}".format(item) for item in self.avas_frag)
            f.write("\n*END OF INPUT\n")

    def generate_input_avas(self) -> None:
        """Generates fortran AVAS input for Rose.

        This method generates a file named "INPUT_AVAS"
            containing input options to be read by Rose.
        """
        input_avas_filename = "INPUT_AVAS"

        # create a vector containing the calculator of each fragment
        # Could we use different calculators for them ?
        fragments_mo_calculator = []
        for frag in enumerate(self.rose_frags):
            fragments_mo_calculator.append(self.rose_frags[frag[0]]
                                           .calc.name.lower())

        with open(input_avas_filename, "w") as f:
            f.write(str(len(self.rose_target.symbols))
                    + " # natoms\n")
            f.write(str(
                self.rose_target.calc.parameters.charge) + " # charge\n")
            if self.restricted:
                f.write("1 # restricted\n")
            else:
                f.write("0 # unrestricted\n")
            f.write("1 # spatial orbs\n")
            f.write(str(self.rose_target.calc
                    .name.lower())
                    + " # MO file for the full molecule\n")
            f.write(" ".join(map(str, fragments_mo_calculator))
                    + " # MO file for the fragments\n")
            f.write(str(len(self.nmo_avas))
                    + " # number of valence MOs in B2\n")
            f.writelines("{:3d}".format(item) for item in self.nmo_avas)

    def generate_mol_frags_xyz(self) -> None:
        """Generates Molecule and Fragment xyz files for Rose.

        Note: cartesian coordinates in angstrom.
        """
        # generate supermolecule xyz file using ASE write()
        mol_filename = "MOLECULE"
        write(mol_filename + ".XYZ", self.rose_target)
        self.mol_frags_filenames.append(mol_filename)

        # generate fragments xyz files
        for frag in enumerate(self.rose_frags):

            # comply with the required Rose files format
            if self.rose_calc_type == RoseCalcType.ATOM_FRAG.value:
                # defining atomic fragment's Z number
                frag_Z = self.rose_frags[frag[0]].numbers[0]

                frag_file_name = "{:03d}".format(frag_Z)
                write(frag_file_name + ".xyz", self.rose_frags[frag[0]])
                self.mol_frags_filenames.append(frag_file_name)

            else:
                frag_file_name = "frag{:d}".format(frag[0])
                write(frag_file_name + ".xyz", self.rose_frags[frag[0]])
                self.mol_frags_filenames.append(frag_file_name)

    def generate_mo_files(self) -> None:
        """Generates orbitals input files for Rose."""
        # First, check whether the required mo files already exist

        # create a vector with selected calculators for molecule and frags
        # put here no restrictions that they must be the same
        calculator_file_extensions = []
        calculator_file_extensions.append(
            self.rose_target.calc.name.lower())
        for frag in enumerate(self.rose_frags):
            calculator_file_extensions.append(
                self.rose_frags[frag[0]].calc.name.lower())

        # vector with all expected mo files
        mo_files_with_extensions = []
        for file in enumerate(self.mol_frags_filenames):
            mo_files_with_extensions.append(
                str(file[1]) + "." + calculator_file_extensions[file[0]])

        # check if such mo files with the required extensions exist
        all_files_exist = True
        for file in mo_files_with_extensions:
            if not os.path.exists(file):
                all_files_exist = False

        # if they do not exist, then generate them
        if not all_files_exist:

            # create a list with all ASE Atoms objects
            # for the supermolecule and fragments
            list_of_ase_Atoms = []
            list_of_ase_Atoms.append(self.rose_target)
            for frag in enumerate(self.rose_frags):
                list_of_ase_Atoms.append(self.rose_frags[frag[0]])

            # create dictionary with filename as key and ASE Atoms as values
            filename_Atoms_dict = dict(
                zip(mo_files_with_extensions, list_of_ase_Atoms))

            for filename_key in filename_Atoms_dict:

                ase_Atom = filename_Atoms_dict[filename_key]

                natom = len(ase_Atom.symbols) 

                # nelec = atomic number minus electronic charge
                nelec = (sum(ase_Atom.numbers)
                         - ase_Atom.calc.parameters.charge)
                spin = ase_Atom.calc.parameters.multiplicity - 1
                nalpha = (nelec + spin)//2
                nbeta = (nelec - spin)//2

                # cartesian coordinates in a.u.
                cart_coord = []
                for i in range(natom):
                    for j in range(3):
                        cart_coord.append(
                            ase_Atom.positions[i][j]/Bohr)

                # Rose restrictions on spherical x cartesian basis functions
                if self.spherical:
                    pureamd = 0
                    pureamf = 0
                    print("ROSE does not work with spherical basis functions.")
                else:
                    pureamd = 1
                    pureamf = 1

                # => pyscf specific variables # begin
                # self.wf = scf.addons.frac_occ(self.wf)
                # assign fractional occupancy for degenerated occupied HOMOs.
                # important to reproduce the results

                mol = ase_Atom.calc.mol
                mf = ase_Atom.calc.wf

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
                        coord_shell.append(ase_Atom.positions[
                                shell_atom_map[i] - 1][j]/Bohr)

                prim_exp = []
                for i in range(natom):
                    atom_type = ase_Atom.symbols[i]
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
                with open(filename_key, "w") as f:
                    f.write("{:13}{:10}\n"
                            .format("Generated by",
                                    ase_Atom.calc.name.upper()))
                    write_int(f, "Number of atoms", natom)
                    write_int(f, "Charge", ase_Atom.calc.parameters.charge)
                    write_int(f, "Multiplicity",
                              ase_Atom.calc.parameters.multiplicity)
                    write_int(f, "Number of electrons", nelec)
                    write_int(f, "Number of alpha electrons", nalpha)
                    write_int(f, "Number of beta electrons", nbeta)
                    write_int(f, "Number of basis functions", nao)
                    write_int_list(f, "Atomic numbers", ase_Atom.numbers)
                    write_singlep_list(f, "Nuclear charges", ase_Atom.numbers)
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
        else:
            print("MO files", mo_files_with_extensions, "exist.")
            print("Proceeding to the next step.")

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
        mf = self.rose_target.calc.wf

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

    def run_avas(self) -> None:
        """Runs 'avas.x' executable."""
        self.avas_output_filename = "OUTPUT_AVAS"
        self.command = "avas.x > " + self.avas_output_filename
        FileIOCalculator.calculate(self)

    def run_post_hf(self) -> None:
        """Performs CASCI and/or CASSCF calculations."""
        pass

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
