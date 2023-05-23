"""Rose water input test."""
from ase import Atoms
from ase.units import Ha
# from ase.build import molecule

import sys
sys.path.append('..')

from qc2.ase.rose import Rose
from qc2.ase.pyscf import PySCF

# define target molecule
mol = Atoms('OH2',
            positions=[[0.,  0.00000,  0.59372],
                       [0.,  0.76544, -0.00836],
                       [0., -0.76544, -0.00836]
                       ],
               calculator=PySCF(method='scf.RHF',
                                basis='unc-sto-3g',
                                multiplicity=1,
                                cart=True,
                                scf_addons='frac_occ',
                                output='water'
                                )
               )
mol.get_potential_energy()
mol.calc.dump_mo_input_file_for_rose(output_file='MOLECULE.pyscf')

# define atomic fragments
frag1 = Atoms('O',
              calculator=PySCF(method='scf.RHF',
                               basis='unc-sto-3g',
                               multiplicity=1,
                               cart=True,
                               scf_addons='frac_occ',
                               output='O'
                               )
              )
frag1.get_potential_energy()
frag1.calc.dump_mo_input_file_for_rose(output_file='008.pyscf')

frag2 = Atoms('H',
              calculator=PySCF(method='scf.ROHF',
                               basis='unc-sto-3g',
                               multiplicity=2,
                               cart=True,
                               scf_addons='frac_occ',
                               output='H'
                               )
              )
frag2.get_potential_energy()
frag2.calc.dump_mo_input_file_for_rose(output_file='001.pyscf')

# Rose-ASE calculator
rose_calc = Rose(rose_calc_type='atom_frag',
                 exponent=4,
                 rose_target=mol,
                 rose_frags=[frag1, frag2],
                 test = True
                 )

rose_calc.calculate()