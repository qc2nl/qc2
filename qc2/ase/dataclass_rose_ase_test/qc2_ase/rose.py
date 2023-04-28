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

    # Rose intrinsic options => expected to be fixed at the default values?
    version: str = ['Stndrd_2013', 'Simple_2013', 'Simple_2014'][0]
    exponent: int = [2, 3, 4][2]
    spherical: bool = False
    uncontract: bool = True
    test: bool = True
    wf_restart: bool = True
    get_oeint: bool = True
    save: bool = True
    avas_frag: Optional[List[int]] = field(default_factory=list)
    nmo_avas: Optional[List[int]] = field(default_factory=list)

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

    def __init__(self, *args, **kwargs) -> None:
        """ASE-Rose Class Constructor to initialize the object.

        Give an example here on how to use....
        """
        # initializing base classes
        RoseInputDataClass.__init__(self, *args, **kwargs)
        FileIOCalculator.__init__(self, *args, **kwargs)

    def calculate(self) -> None:
        """Executes Rose workflow."""
        self.get_input_genibo_avas()
        self.get_input_xyz()

        if self.calculate_mo:
            self.get_mo_files()

        self.run_rose()

        if self.save:
            self.save_ibos()

        if len(self.avas_frag) != 0:
            self.run_avas()

        if self.run_postscf:
            self.run_post_hf()

    def get_input_genibo_avas(self) -> None:
        """Generates INPUT_GENIBO & INPUT_AVAS fortran files for Rose."""
        write_input_genibo_avas(self)

    def get_input_xyz(self) -> None:
        """Generates Molecule and Fragment xyz files for Rose."""
        print("Creating Molecule and Frags inputs....done")

    def get_mo_files(self) -> None:
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


def write_input_genibo_avas(input_data: RoseInputDataClass) -> None:
    """Summary.

    Args:
        input_data (RoseInputDataClass): description
    """
    print("Creating genibo and avas input....done")

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
    with open("INPUT_GENIBO", "w") as f:
        f.write("**ROSE\n")
        f.write(".VERSION\n")
        f.write(version+"\n")
        f.write(".CHARGE\n")
        f.write(str(molecule_charge)+"\n")
        f.write(".EXPONENT\n")
        f.write(str(exponent)+"\n")
        if not restricted:
            f.write(".UNRESTRICTED\n")
        f.write(".FILE_FORMAT\n")
        f.write(molecule_mo_calculator+"\n")
        if test == 1:
            f.write(".TEST  \n")
        f.write(".AVAS  \n")
        f.write(str(len(avas_frag))+"\n")
        f.writelines("{:3d}".format(item) for item in avas_frag)
        f.write("\n*END OF INPUT\n")

    # creating INPUT_AVAS file
    with open("INPUT_AVAS", "w") as f:
        f.write(str(molecule_natom) + " # natoms\n")
        f.write(str(molecule_charge) + " # charge\n")
        if restricted:
            f.write("1 # restricted\n")
        else:
            f.write("0 # restricted\n")
        f.write("1 # spatial orbs\n")
        f.write(molecule_mo_calculator + " # MO file for the full molecule\n")
        f.write(fragments_mo_calculator + " # MO file for the fragments\n")
        f.write(str(len(nmo_avas)) + " # number of valence MOs in B2\n")
        f.writelines("{:3d}".format(item) for item in nmo_avas)

    # print(molecule_natom)
