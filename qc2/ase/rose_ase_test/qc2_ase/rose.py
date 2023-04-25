"""This module defines an ASE interface to ROSE.

Original paper & official release:
https://pubs.acs.org/doi/10.1021/acs.jctc.0c00964

GitLab page:
https://gitlab.com/quantum_rose/rose

Note: see also https://pubs.acs.org/doi/10.1021/ct400687b.
"""
from ase import Atoms
from ase.calculators.calculator import FileIOCalculator
from ase.units import Ha  # => needed only for testing; remove latter.
from typing import Optional  # List, Tuple, Union
# from qc2_ase.rose_io import *
# from qc2_ase.pyscf import PySCF


class ROSE(FileIOCalculator):
    """A general ASE calculator for ROSE (Reduction of Orbital Space Extent).

    Args:
        FileIOCalculator (FileIOCalculator): Base class for calculators
            that write/read input/output files.
    """
    #implemented_properties = ['orbitals']
    command = 'echo "Executing Rose...done"'  # => test

    default_parameters = {
        'rose_calc_type': ['atom_frag', 'mol_frag'][1],
        'calculate_mo': True,
        'uncontract': True,
        'version': ['Stndrd_2013', 'Simple_2013', 'Simple_2014'][0],
        'exponent': [2, 3, 4][2],
        'relativistic': False,
        'spherical': False
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

        print(self.parameters)
        # print(self.parameters.rose_target.calc)
        # print(self.parameters.rose_frags[0].calc)
        # print(self.parameters.rose_frags[1].calc)

    def calculate(self, *args, **kwargs):
        """Executes Rose workflow."""
        super().calculate(*args, **kwargs)
        # calls:
        # 1) write_input()
        # 2) execute()
        # 3) read_results()

    def write_input(self, atoms, properties=None, system_changes=None):
        """Generates all inputs necessary for Rose."""
        super().write_input(atoms, properties, system_changes)
        #
        # TODO
        #
        # 1) Write INPUT_GENIBO.
        # 2) Generate Molecule.xyz and Frags.xyz.
        # 3) If requested, calculate "on-the-fly" the orbitals file
        #       for the system and all its fragments

        print("Writing INPUT_GENIBO and INPUT_AVAS....done\n")

        print("Generating Molecule.xyz and Frags.xyz....done\n")

        print("Generating orbitals file....done")

        # test
        mol = self.parameters.rose_target
        print('H2O energy/Eh =', mol.get_potential_energy()/Ha)
        print('H2O orbitals =', mol.calc.wf.mo_coeff)

        for fragment in self.parameters.rose_frags:
            print(fragment.symbols, 'energy/Eh =',
                  fragment.get_potential_energy()/Ha)
            print(fragment.calc.wf.mo_coeff)
        print(" ")

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

    # def read_results(self):
    #    """_summary_"""
    #    #
    #    # TODO
    #    #
    #    # 1) Print SCF and/or post-HF energies
    #    self.results['energy'] = 1.0
    #    #print(self.results)
    #    print("Reading/Printing results...done")
