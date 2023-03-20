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

    # define molecule geometry
    h2_molecule = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.7]])

    # run calc and convert electronic energy into atomic units 
    h2_molecule.calc = PySCF(method='dft.RKS', xc='pbe', basis='sto-3g', charge=0, multiplicity=1, verbose=0)
    energy_Eh = h2_molecule.get_potential_energy() / Ha

    # compare with the energy obtained using pyscf alone - assuming convergence up to ~1e-6
    assert energy_Eh == pytest.approx(-1.15101082903195, 1e-6)


def test_PySCF_energy_uhf():
    """Test case # 2 - UHF H atom"""
    
    h_atom = Atoms('H')

    h_atom.calc = PySCF(method='scf.UHF', basis='sto-3g', charge=0, multiplicity=2, verbose=0)
    energy_Eh = h_atom.get_potential_energy() / Ha

    assert energy_Eh == pytest.approx(-0.466581849557275, 1e-6)

def test_PySCF_energy_rohf():
    """Test case # 3 - ROHF O2 molecule"""

    # now using the built-in molecule dataset to define the geometry 
    o2_molecule = molecule('O2')
    
    # define the calculator
    o2_molecule.calc = PySCF()

    # as an alternative to #1, define PySCF wave function using its specific attributes  
    o2_molecule.calc.method = 'scf.ROHF'
    o2_molecule.calc.basis = 'sto-3g'
    o2_molecule.calc.charge = 0 
    o2_molecule.calc.multiplicity = 3
    o2_molecule.calc.verbose = 0
    energy_Eh = o2_molecule.get_potential_energy() / Ha

    assert energy_Eh == pytest.approx(-147.630640704849, 1e-6)

def test_PySCF_energy_opt():
    """Test case # 4 - RKS geometry optimization H2 molecule"""

    # define geometry and wave function
    h2_molecule = molecule('H2')
    h2_molecule.calc = PySCF(method='dft.RKS', xc='pbe', basis='sto-3g', charge=0, multiplicity=1, verbose=0)

    # run optimization and get the energy at minium
    opt = BFGS(h2_molecule)
    opt.run(fmax=1e-4)
    energy_opt = h2_molecule.get_potential_energy() / Ha

    # Just an example, you can also calculate the gradient
    gradient_opt = h2_molecule.get_forces()

    # now compare with the energy obtained using pyscf pyberny optimization alone
    assert energy_opt == pytest.approx(-1.15209884495, 1e-6)
