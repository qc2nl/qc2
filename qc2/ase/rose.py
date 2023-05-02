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
from ase.io import write


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
    # openshell: bool = False => made inactive
    relativistic: bool = False
    include_core: bool = False

    # Rose intrinsic options => expected to be fixed at the default values?
    version: RoseIFOVersion = RoseIFOVersion.STNDRD_2013.value
    exponent: RoseILMOExponent = RoseILMOExponent.TWO.value
    spherical: bool = False
    uncontract: bool = True
    test: bool = False
    wf_restart: bool = True
    get_oeint: bool = True
    save: bool = True
    avas_frag: Optional[List[int]] = field(default_factory=list)
    nmo_avas: Optional[List[int]] = field(default_factory=list)
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

        if self.calculate_mo:
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
                        f.write(str(item[0])+"\n")
                        f.write(str(item[1])+"\n")
                if self.frag_core:
                    for item in self.frag_core:
                        f.write(".FRAG_CORE\n")
                        f.write(str(item[0])+"\n")
                        f.write(str(item[1])+"\n")
                if self.frag_bias:
                    for item in self.frag_bias:
                        f.write(".FRAG_BIAS\n")
                        f.write(str(item[0])+"\n")
                        f.write(str(item[1])+"\n")
            if self.avas_frag:
                f.write(".AVAS\n")
                f.write(str(len(self.avas_frag)) + "\n")
                f.writelines("{:3d}".format(item) for item in self.avas_frag)
            f.write("\n*END OF INPUT\n")

    def generate_input_avas(self) -> None:
        """Generates fortran AVAS input for Rose.

        This method generates a file named "INPUT_AVAS"
            containing input options to be used by Rose.
        """
        fragments_mo_calculator = []
        for n_frag in range(len(self.rose_frags)):
            frag_mo_program = self.rose_frags[n_frag].calc.name.lower()
            fragments_mo_calculator.append(frag_mo_program)

        with open("INPUT_AVAS", "w") as f:
            f.write(str(len(self.rose_target.symbols))
                    + " # natoms\n")
            f.write(str(self.rose_target.calc
                        .parameters.charge)
                        + " # charge\n")
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
        """Generates Molecule and Fragment xyz files for Rose."""
        # generate supramolecule xyz file using ASE write()
        write("MOLECULE.XYZ", self.rose_target)

        # generate fragments xyz files
        for n_frag in range(len(self.rose_frags)):

            if self.rose_calc_type == RoseCalcType.ATOM_FRAG.value:
                # defining atomic fragment's Z number
                n_frag_Z = self.rose_frags[n_frag].numbers[0]

                frag_file_name = "{:03d}.xyz".format(n_frag_Z)
                write(frag_file_name, self.rose_frags[n_frag])

            else:
                frag_file_name = "frag{:d}.xyz".format(n_frag)
                write(frag_file_name, self.rose_frags[n_frag])

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

def name_molecule(geometry,
                  spin,
                  charge,
                  description):
    """Function to name molecules.

    Args:
        geometry: A list of tuples giving the coordinates of each atom.
            example is [('H', (0, 0, 0)), ('H', (0, 0, 0.7414))].
            Distances in angstrom. Use atomic symbols to specify atoms.
        spin: An integer giving the spin 2S (difference between alpha and beta electrons)
        charge: An integer giving the total molecular charge.
        description: A string giving a description. As an example,
            for dimers a likely description is the bond length (e.g. 0.7414).

    Returns:
        name: A string giving the name of the instance.
    """
    if not isinstance(geometry, basestring):
        # Get sorted atom vector.
        atoms = [item[0] for item in geometry]
        atom_charge_info = [(atom, atoms.count(atom)) for atom in set(atoms)]
        sorted_info = sorted(atom_charge_info,
                             key=lambda atom: periodic_hash_table[atom[0]])

        # Name molecule.
        name = '{}{}'.format(sorted_info[0][0], sorted_info[0][1])
        for info in sorted_info[1::]:
            name += '-{}{}'.format(info[0], info[1])
    else:
        name = geometry

    # Ass spin
    name += '_{}'.format(spin)

    # Add charge.
    if charge > 0:
        name += '_{}+'.format(charge)
    elif charge < 0:
        name += '_{}-'.format(charge)

    # Optionally add descriptive tag and return.
    if description:
        name += '_{}'.format(description)
    return name
