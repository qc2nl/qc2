"""This module defines an ASE interface to Rose.

Original paper & official release:
https://pubs.acs.org/doi/10.1021/acs.jctc.0c00964

GitLab page:
https://gitlab.com/quantum_rose/rose

Note: see also https://pubs.acs.org/doi/10.1021/ct400687b.
"""
from dataclasses import dataclass, field
from enum import Enum
from ase import Atoms
from ase.calculators.calculator import FileIOCalculator
from typing import Optional, List, Union, Sequence


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

    calculate_mo: bool = True
    rose_calc_type: RoseCalcType = RoseCalcType.ATOM_FRAG.value

    run_postscf: bool = False
    restricted: bool = True
    openshell: bool = False
    relativistic: bool = False
    include_core: bool = False

    # Rose intrinsic options => expected to be fixed at the default values?
    version: RoseIFOVersion = RoseIFOVersion.STNDRD_2013.value
    exponent: RoseILMOExponent = RoseILMOExponent.FOUR.value
    spherical: bool = False
    uncontract: bool = True
    test: bool = True
    wf_restart: bool = True
    get_oeint: bool = True
    save: bool = True
    avas_frag: Optional[List[int]] = field(default_factory=list)
    nmo_avas: Optional[List[int]] = field(default_factory=list)
    additional_virtuals_cutoff: float = 2.0
    frag_threshold: float = 10.0
    frag_valence: Optional[List[List[int]]] = field(default_factory=list)
    frag_core: Optional[List[List[int]]] = field(default_factory=list)

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

        # print()

    def calculate(self, *args, **kwargs) -> None:
        """Executes Rose workflow."""
        super().calculate(*args, **kwargs)
        self.generate_input_genibo_avas()
        self.generate_input_xyz()

        if self.calculate_mo:
            self.generate_mo_files()

        self.run_rose()

        if self.save:
            self.save_ibos()

        if self.avas_frag:
            self.run_avas()

        if self.run_postscf:
            self.run_post_hf()

    def generate_input_genibo_avas(self) -> None:
        """Generates INPUT_GENIBO & INPUT_AVAS fortran files for Rose."""
        write_fortran_genibo_avas_input(self,
                                        genibo_filename="INPUT_GENIBO",
                                        avas_filename="INPUT_AVAS")

    def generate_input_xyz(self) -> None:
        """Generates Molecule and Fragment xyz files for Rose."""
        print("Creating Molecule and Frags inputs....done")

    def generate_mo_files(self) -> None:
        """Generates atomic and molecular orbitals files for Rose."""
        print('Calculating MO files....done')

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


def write_fortran_genibo_avas_input(
        input_data: RoseInputDataClass,
        genibo_filename: str = "INPUT_GENIBO",
        avas_filename: str = "INPUT_AVAS") -> None:
    """Writes Rose fortran inputs.

    Args:
        input_data (RoseInputDataClass): dataclass defining input
            options for Rose.
        genibo_filename (str): file to be generated containing
            Rose input options.
        avas_filename (str): name of the file to be generated with
            AVAS input options.
    """
    print("Creating genibo and avas input....done")

    rose_calc_type = input_data.rose_calc_type

    molecule_charge = input_data.rose_target.calc.parameters.charge
    molecule_mo_calculator = input_data.rose_target.calc.name.lower()
    molecule_natom = len(input_data.rose_target.symbols)

    fragments_mo_calculator = []
    for n in range(len(input_data.rose_frags)):
        frag_mo_program = input_data.rose_frags[n].calc.name.lower()
        fragments_mo_calculator.append(frag_mo_program)

    version = input_data.version
    exponent = input_data.exponent
    restricted = input_data.restricted
    test = input_data.test

    avas_frag = input_data.avas_frag
    nmo_avas = input_data.nmo_avas

    # creating INPUT_GENIBO file
    with open(genibo_filename, "w") as f:
        f.write("**ROSE\n")
        f.write(".VERSION\n")
        f.write(version + "\n")
        f.write(".CHARGE\n")
        f.write(str(molecule_charge) + "\n")
        f.write(".EXPONENT\n")
        f.write(str(exponent) + "\n")
        if not restricted:
            f.write(".UNRESTRICTED\n")
        f.write(".FILE_FORMAT\n")
        f.write(molecule_mo_calculator + "\n")
        if test:
            f.write(".TEST\n")
        if (rose_calc_type == RoseCalcType.MOL_FRAG.value):
            f.write(".NFRAGMENTS\n")
        if avas_frag:
            f.write(".AVAS\n")
        f.write(str(len(avas_frag)) + "\n")
        f.writelines("{:3d}".format(item) for item in avas_frag)
        f.write("\n*END OF INPUT\n")

    # creating INPUT_AVAS file
    if avas_frag:
        with open(avas_filename, "w") as f:
            f.write(str(molecule_natom) + " # natoms\n")
            f.write(str(molecule_charge) + " # charge\n")
            if restricted:
                f.write("1 # restricted\n")
            else:
                f.write("0 # unrestricted\n")
            f.write("1 # spatial orbs\n")
            f.write(str(molecule_mo_calculator)
                    + " # MO file for the full molecule\n")
            f.write(" ".join(map(str, fragments_mo_calculator))
                    + " # MO file for the fragments\n")
            f.write(str(len(nmo_avas))
                    + " # number of valence MOs in B2\n")
            f.writelines("{:3d}".format(item) for item in nmo_avas)

    # print(molecule_natom)
