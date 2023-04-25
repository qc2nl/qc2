"""Rose input test."""
from ase import Atoms
from ase.build import molecule
# from ase.units import Ha
# import rose calculator module
from qc2_ase.rose import ROSE
from qc2_ase.pyscf import PySCF

# define target molecule
mol = molecule('H2O',
               calculator=PySCF(method='scf.RHF',
                                basis='sto-3g',
                                multiplicity=1
                                )
               )

# define atomic fragments
frag1 = Atoms('O',
              calculator=PySCF(method='scf.RHF',
                               basis='sto-3g',
                               multiplicity=1
                               )
              )

frag2 = Atoms('H',
              calculator=PySCF(method='scf.ROHF',
                               basis='sto-3g',
                               multiplicity=2
                               )
              )

# print(mol.get_potential_energy() / Ha)
# print(frag1.get_potential_energy() / Ha)
# print(frag2.get_potential_energy() / Ha)

# Rose-ASE calculator
rose_calc = ROSE(rose_calc_type='atom_frag',
                 rose_target=mol,
                 rose_frags=[frag1, frag2])

rose_calc.calculate()
