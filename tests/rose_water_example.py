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
                                )
               )

# define atomic fragments
frag1 = Atoms('O',
              calculator=PySCF(method='scf.RHF',
                               basis='unc-sto-3g',
                               multiplicity=1,
                               )
              )

frag2 = Atoms('H',
              calculator=PySCF(method='scf.ROHF',
                               basis='unc-sto-3g',
                               multiplicity=2
                               )
              )

print(mol.get_potential_energy() / Ha)
print(frag1.get_potential_energy() / Ha)
print(frag2.get_potential_energy() / Ha)

# Rose-ASE calculator
rose_calc = Rose(rose_calc_type='atom_frag',
                 rose_target=mol,
                 rose_frags=[frag1, frag2],
                 test = True,
                 avas_frag=[0], nmo_avas=[3, 4, 5]
                 )

rose_calc.calculate()

# print(rose_calc.mol_frags_filenames)
