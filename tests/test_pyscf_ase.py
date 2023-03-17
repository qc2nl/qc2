"""Tests for the ASE-PySCF interface"""

import pytest

import numpy as np
from ase import Atoms
from ase.build import molecule
from ase.optimize import BFGS
from ase.units import Ha
from qc2.ase.pyscf import PySCF

def test_PySCF_energy_rks():
    """Test case # 1 - RKS H2 molecule"""

    h2_molecule = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.7]])

    h2_molecule.calc = PySCF(method='dft.RKS', xc='b3lyp', basis='6-31g*', charge=0, multiplicity=1, verbose=0)
    energy_Eh = h2_molecule.get_potential_energy() / Ha

    assert energy_Eh == pytest.approx(-1.1673385906283245, rel=1e-5)


def test_PySCF_energy_uhf():
    """Test case # 2 - UHF H atom"""
    
    h_atom = Atoms('H')

    h_atom.calc = PySCF(method='scf.UHF', basis='6-31g*', charge=0, multiplicity=2, verbose=0)
    energy_Eh = h_atom.get_potential_energy() / Ha

    assert energy_Eh == pytest.approx(-0.5, rel=0.01)

def test_PySCF_energy_rohf():
    """Test case # 3 - ROHF O2 molecule"""

    o2_molecule = molecule('O2')
    
    o2_molecule.calc = PySCF()
    o2_molecule.calc.method = 'scf.ROHF'
    o2_molecule.calc.basis = '6-31g*'
    o2_molecule.calc.charge = 0 
    o2_molecule.calc.multiplicity = 3
    o2_molecule.calc.verbose = 0
    energy_Eh = o2_molecule.get_potential_energy() / Ha

    assert energy_Eh == pytest.approx(-149.58311940650503, rel=1e-5)

def test_PySCF_forces():
    """Test case # 4 - RKS forces H2 molecule"""

    h2_molecule = molecule('H2')
    h2_molecule.calc = PySCF(method='dft.RKS', xc='b3lyp', basis='6-31g*', charge=0, multiplicity=1, verbose=0)
    opt = BFGS(h2_molecule)
    opt.run(fmax=1e-4)
    energy_opt = h2_molecule.get_potential_energy()
    gradient_opt = h2_molecule.get_forces()

    assert gradient_opt == pytest.approx(np.zeros((2, 3)), abs=1e-6) 
