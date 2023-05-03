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
    include_core: bool = False

    # Rose intrinsic options => expected to be fixed at the default values?
    version: RoseIFOVersion = RoseIFOVersion.STNDRD_2013.value
    exponent: RoseILMOExponent = RoseILMOExponent.TWO.value
    spherical: bool = False
    uncontract: bool = True
    test: bool = False
    wf_restart: bool = False
    get_oeint: bool = True
    save: bool = True
    avas_frag: Optional[List[int]] = field(default_factory=list)
    nmo_avas: Optional[List[int]] = field(default_factory=list)

    # options for virtual orbitals localization.
    additional_virtuals_cutoff: float = 2.0  # Eh
    frag_threshold: float = 10.0  # Eh
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
    command: str = 'echo "Executing Rose...done"'
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
        # RoseInputDataClass.__init__(self, *args, **kwargs)
        # FileIOCalculator.__init__(self, *args, **kwargs)
        super().__init__(*args, **kwargs)

        # print(self.parameters)

    def calculate(self, *args, **kwargs) -> None:
        """Executes Rose workflow."""
        super().calculate(*args, **kwargs)

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
        with open("INPUT_GENIBO", "w") as f:
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
            if self.relativistic:
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
        # create a vector containing the calculator of each fragment
        # Could we use different calculators for them ?
        fragments_mo_calculator = []
        for frag in enumerate(self.rose_frags):
            fragments_mo_calculator.append(self.rose_frags[frag[0]]
                                           .calc.name.lower())

        with open("INPUT_AVAS", "w") as f:
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
        # define a list of filenames
        self.mol_frags_filenames = []

        # generate supramolecule xyz file using ASE write()
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
        """Generates atomic and molecular orbitals files for Rose."""
        # First, check whether the required mo files already exist

        # create a vector with selected calculators for molecule and frags
        # putting here no restrictions that they must be the same
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

            # create a list with all ASE Atoms object
            list_of_Atoms = []
            list_of_Atoms.append(self.rose_target)
            for frag in enumerate(self.rose_frags):
                list_of_Atoms.append(self.rose_frags[frag[0]])

            # create dictionary with filename as key and ASE Atoms as values
            dict_of_tasks = dict(zip(mo_files_with_extensions, list_of_Atoms))

            for filename_key in dict_of_tasks:

                natom = len(dict_of_tasks[filename_key].symbols)

                # nelec = atomic number minus electronic charge
                nelec = (sum(dict_of_tasks[filename_key].numbers)
                         - dict_of_tasks[filename_key].calc.parameters.charge)
                spin = dict_of_tasks[
                    filename_key].calc.parameters.multiplicity - 1
                nalpha = (nelec + spin)//2
                nbeta = (nelec - spin)//2

                # cartesian coordinates in a.u.
                cart_coord = []
                for i in range(natom):
                    for j in range(3):
                        cart_coord.append(
                            dict_of_tasks[filename_key].positions[i][j]/Bohr)

                # Rose restrictions on spherical x cartesian basis functions
                if self.spherical:
                    pureamd = 0
                    pureamf = 0
                    print("ROSE does not work with spherical basis functions.")
                else:
                    pureamd = 1
                    pureamf = 1

                # => pyscf specific variables # begin

                nao = dict_of_tasks[
                    filename_key].calc.mol.nao_cart()

                nshells = dict_of_tasks[
                    filename_key].calc.mol.nbas

                shell_atom_map = []
                orb_momentum = []
                contract_coeff = []
                mol_bas = dict_of_tasks[
                    filename_key].calc.mol._bas

                for i in range(len(mol_bas)):
                    shell_atom_map.append(mol_bas[i][0] + 1)
                    orb_momentum.append(mol_bas[i][1])
                    contract_coeff.append(mol_bas[i][3])

                nprim_shell = []
                coord_shell = []
                for i in range(nshells):
                    nprim_shell.append(1)
                    for j in range(3):
                        coord_shell.append(dict_of_tasks[
                            filename_key].positions[
                                shell_atom_map[i] - 1][j])

                prim_exp = []
                for i in range(natom):
                # atom_type =
                    print(i)

                # => pyscf specific variables # end

                # start writing Rose input mo files
                with open(filename_key, "w") as f:
                    f.write("{:13}{:10}\n"
                            .format("Generated by",
                                    dict_of_tasks[filename_key]
                                    .calc.name.upper()))
                    write_int(f, "Number of atoms", natom)
                    write_int(f, "Charge", dict_of_tasks[filename_key]
                              .calc.parameters.charge)
                    write_int(f, "Multiplicity", dict_of_tasks[filename_key]
                              .calc.parameters.multiplicity)
                    write_int(f, "Number of electrons", nelec)
                    write_int(f, "Number of alpha electrons", nalpha)
                    write_int(f, "Number of beta electrons", nbeta)
                    write_int(f, "Number of basis functions", nao)
                    write_int_list(f, "Atomic numbers", dict_of_tasks[
                        filename_key].numbers)
                    write_singlep_list(f, "Nuclear charges", dict_of_tasks[
                        filename_key].numbers)
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

                pass
        else:
            print("MO files", mo_files_with_extensions, "exist.")
            print("Proceeding to the next step.")

    def run_rose(self) -> None:
        """Runs Rose executable 'genibo.x'."""
        print('Executing Rose....done')

    def save_ibos(self) -> None:
        """Generates a checkpoint file ('ibo.chk') with the final IBOs."""
        print('Saving ibos....done')

    def run_avas(self) -> None:
        """Runs 'avas.x' executable."""
        print('Executing avas....done')

    def run_post_hf(self) -> None:
        """Performs CASCI and/or CASSCF calculations."""
        print('CASSCF?....done')

def write_int(f,text,var):
    """_summary.

    Args:
        f (_type_): _description_
        text (_type_): _description_
        var (_type_): _description_
    """
    f.write("{:43}I{:17d}\n".format(text,var))


def write_int_list(f,text,var):
    """_summary.

    Args:
        f (_type_): _description_
        text (_type_): _description_
        var (_type_): _description_
    """
    f.write("{:43}{:3} N={:12d}\n".format(text,"I",len(var)))
    dim = 0
    buff = 6
    if (len(var) < 6): buff = len(var)
    for i in range((len(var)-1)//6+1):
        for j in range(buff):
            f.write("{:12d}".format(var[dim+j]))
        f.write("\n")
        dim = dim + 6
        if (len(var) - dim) < 6 : buff = len(var) - dim


def write_singlep_list(f,text,var):
    """_summary.

    Args:
        f (_type_): _description_
        text (_type_): _description_
        var (_type_): _description_
    """
    f.write("{:43}{:3} N={:12d}\n".format(text,"R",len(var)))
    dim = 0
    buff = 5
    if (len(var) < 5): buff = len(var)
    for i in range((len(var)-1)//5+1):
        for j in range(buff):
            f.write("{:16.8e}".format(var[dim+j]))
        f.write("\n")
        dim = dim + 5
        if (len(var) - dim) < 5 : buff = len(var) - dim


def write_doublep_list(f,text,var):
    """_summary.

    Args:
        f (_type_): _description_
        text (_type_): _description_
        var (_type_): _description_
    """
    f.write("{:43}{:3} N={:12d}\n".format(text,"R",len(var)))
    dim = 0
    buff = 5
    if (len(var) < 5): buff = len(var)
    for i in range((len(var)-1)//5+1):
        for j in range(buff):
            f.write("{:24.16e}".format(var[dim+j]))
        f.write("\n")
        dim = dim + 5
        if (len(var) - dim) < 5 : buff = len(var) - dim
