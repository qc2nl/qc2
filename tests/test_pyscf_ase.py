"""Tests for the ASE-PySCF interface"""

import pytest

from ase import Atoms
from ase.build import molecule
from ase.optimize import BFGS
from ase.units import Ha
from ase.calculators.calculator import InputError
from ase.calculators.calculator import CalculatorSetupError
from qc2.ase.pyscf import PySCF

def test_PySCF_energy_rks():
    """Test case # 1 - RKS H2 molecule"""

    # define molecule geometry
    h2_molecule = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.7]])

    # run calc and convert electronic energy into atomic units 
    h2_molecule.calc = PySCF(method='dft.RKS', xc='pbe', basis='sto-3g', charge=0, multiplicity=1, verbose=0)
    energy_Eh = h2_molecule.get_potential_energy() / Ha

    # compare with the energy obtained using pyscf alone => assuming convergence up to ~1e-6
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
    
    o2_molecule.calc = PySCF(method='scf.ROHF', basis='sto-3g', charge=0, multiplicity=3, verbose=0)
    energy_Eh = o2_molecule.get_potential_energy() / Ha

    assert energy_Eh == pytest.approx(-147.630640704849, 1e-6)


def test_PySCF_energy_opt():
    """Test case # 4 - RKS geometry optimization H2 molecule"""

    # define geometry and wave function
    h2_molecule = molecule('H2')
    h2_molecule.calc = PySCF(method='dft.RKS', xc='pbe', basis='sto-3g', charge=0, multiplicity=1, verbose=0)

    # run optimization and get the energy at the minium
    opt = BFGS(h2_molecule)
    opt.run(fmax=0.1)
    energy_opt = h2_molecule.get_potential_energy() / Ha

    # Calculate the gradient if needed
    gradient_opt = h2_molecule.get_forces()

    # compare with the energy obtained using pyscf alone via pyberny optimization 
    assert energy_opt == pytest.approx(-1.15209884495, 1e-6)


def test_PySCF_with_attribute_error():
    """Test case # 5 - Initializing attribute with unrecognized name"""

    with pytest.raises(InputError) as excinfo:

        # setting the basis attribute with an unrecognized name
        mol = Atoms()
        mol.calc = PySCF(bas='sto-3g')
        energy = mol.get_potential_energy()
         
    assert 'not recognized' in str(excinfo.value)


def test_PySCF_with_wf_error():
    """Test case # 6 - Setting wave function not yet implemented"""

    with pytest.raises(CalculatorSetupError) as excinfo:

        mol = Atoms()
        mol.calc = PySCF(method='mp.MP2')
        energy = mol.get_potential_energy()
   
    assert 'Method not yet implemented' in str(excinfo.value)    
