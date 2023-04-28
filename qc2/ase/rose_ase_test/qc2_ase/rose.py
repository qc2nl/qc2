"""This module defines an ASE interface to ROSE.

Original paper & official release:
https://pubs.acs.org/doi/10.1021/acs.jctc.0c00964

GitLab page:
https://gitlab.com/quantum_rose/rose

Note: see also https://pubs.acs.org/doi/10.1021/ct400687b.
"""
from ase import Atoms
from ase.calculators.calculator import FileIOCalculator
from typing import Optional, List, Union, TypedDict, Sequence  # , Tuple,
from qc2_ase.rose_io import *
# from qc2_ase.pyscf import PySCF


class RoseInputDataTypes(TypedDict):
    """Rose input data type definition."""
    rose_target: Atoms
    rose_frags: Union[Atoms, Sequence[Atoms]]
    rose_calc_type: str
    calculate_mo: bool
    uncontract: bool
    version: str
    exponent: int
    relativistic: bool
    spherical: bool
    restricted: bool
    openshell: bool
    test: bool
    avas_frag: List[int]
    nmo_avas: List[int]
    get_oeint: bool
    save: bool
    run_postscf: bool


class ROSE(FileIOCalculator):
    """A general ASE calculator for ROSE (Reduction of Orbital Space Extent).

    Args:
        FileIOCalculator (FileIOCalculator): Base class for calculators
            that write/read input/output files.
    """
    implemented_properties = []
    command = 'echo "Executing Rose...done"'  # => test

    default_parameters: RoseInputDataTypes = {
        'rose_target': None,
        'rose_frags': [],
        'rose_calc_type': ['atom_frag', 'mol_frag'][0],
        'calculate_mo': True,
        'uncontract': True,
        'version': ['Stndrd_2013', 'Simple_2013', 'Simple_2014'][0],
        'exponent': [2, 3, 4][2],
        'relativistic': False,
        'spherical': False,
        'restricted': True,
        'openshell': False,
        'test': True,
        'avas_frag': [],
        'nmo_avas': [],
        'get_oeint': True,
        'save': True,
        'run_postscf': False
        }

    def __init__(self,
                 restart: Optional[bool] = None,
                 ignore_bad_restart_file:
                 Optional[bool] = FileIOCalculator._deprecated,
                 label: Optional[str] = 'rose',
                 atoms: Optional[Atoms] = None,
                 command: Optional[str] = None,
                 **kwargs) -> None:
        """ASE-Rose Class Constructor to initialize the object."""
        super().__init__(restart, ignore_bad_restart_file,
                         label, atoms, command, **kwargs)
        """Transforms **kwargs into a dictionary with calculation parameters.

        Starting with (attr1=value1, attr2=value2, ...)
            it creates self.parameters['attr1']=value1, and so on.
        """

        #print(self.parameters)
        #print(self.parameters.rose_target.calc.parameters.method)
        # print(self.parameters.rose_target.calc)
        # print(self.parameters.rose_frags[0].calc)
        # print(self.parameters.rose_frags[1].calc)

    def calculate(self, *args, **kwargs) -> None:
        """Executes Rose workflow."""
        super().calculate(*args, **kwargs)
        # calls:
        # 1) write_input()
        # 2) execute()

    def write_input(
            self,
            atoms: Optional[Atoms] = None,
            properties: Optional[List[str]] = None,
            system_changes: Optional[List[str]] = None
            ) -> None:
        """Generates all inputs necessary for Rose."""
        super().write_input(atoms, properties, system_changes)

        # write INPUT_GENIBO and/or INPUT_AVAS
        write_input_genibo_avas(self.parameters)

        # generate Molecule.xyz and Frags.xyz
        write_input_mol_frags_xyz(self.parameters)

        if self.parameters['calculate_mo']:
            # calculate "on-the-fly" the orbitals files
            # for the system and all its fragments
            write_mo_files(self.parameters)

    def execute(self):
        """_summary_."""
        super().execute()
        #
        # TODO
        #
        # 1) Execute genibo.x.
        # 2) Store IAOs & IBOs.
        # 3) Execute avas.x, if required.
        # 4) Post-HF calculations, if required.
