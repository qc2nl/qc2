"""Rose water input test."""
from ase import Atoms
from ase.units import Ha

import sys
sys.path.append('..')

from qc2.ase.rose import Rose
from qc2.ase.pyscf import PySCF

# define target molecule
mol = Atoms('NH3OH2',
            positions=[[-1.39559, -0.02156,  0.00004],
                       [-1.62981,  0.96110, -0.10622],
                       [-1.86277, -0.51254, -0.75597],
                       [-1.83355, -0.33077,  0.86231],
                       [ 1.56850,  0.10589,  0.00001],
                       [ 0.60674, -0.03396, -0.00063],
                       [ 1.94052, -0.78000,  0.00022]
                       ],
            calculator=PySCF(method='scf.RHF',
                             basis='unc-sto-3g',
                             multiplicity=1)
            )

# define molecular fragments
frag1 = Atoms('NH3',
              positions=[[-1.39559, -0.02156,  0.00004],
                         [-1.62981,  0.96110, -0.10622],
                         [-1.86277, -0.51254, -0.75597],
                         [-1.83355, -0.33077,  0.86231]
                         ],
              calculator=PySCF(method='scf.RHF',
                               basis='unc-sto-3g',
                               multiplicity=1
                               )
              )

frag2 = Atoms('OH2',
              positions=[[1.56850,  0.10589,  0.00001],
                         [0.60674, -0.03396, -0.00063],
                         [1.94052, -0.78000,  0.00022]
                         ],
              calculator=PySCF(method='scf.RHF',
                               basis='unc-sto-3g',
                               multiplicity=1
                               )
              )

print(mol.get_potential_energy() / Ha)
print(frag1.get_potential_energy() / Ha)
print(frag2.get_potential_energy() / Ha)

# Rose-ASE calculator
rose_calc = Rose(rose_calc_type='mol_frag',
                 exponent=4,
                 rose_target=mol,
                 rose_frags=[frag1, frag2],
                 test = True,
                 frag_valence=[[1,7],[2,6]],
                 frag_core=[[1,1],[2,1]]
                 )

# avas_frag=[0], nmo_avas=[3, 4, 5],

rose_calc.calculate()
