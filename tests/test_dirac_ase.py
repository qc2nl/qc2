"""Tests for the ASE-PySCF interface"""

import pytest

from ase import Atoms
from ase.build import molecule
from ase.optimize import BFGS
from ase.units import Ha
from ase.calculators.calculator import InputError
from ase.calculators.calculator import CalculatorSetupError
from qc2.ase.dirac import DIRAC

import subprocess


@pytest.fixture(scope="session", autouse=True)
def clean_up_files():
    """Remove DIRAC calculation outputs at the end of each test."""
    yield
    command = ("rm *.xyz* *.inp* *.out* *.h5* *.tgz*")
    subprocess.run(command, shell=True, capture_output=True)


def test_DIRAC_energy_rks():
    """Test case # 1 - RKS/B3LYP H2 molecule."""

    # define molecule geometry
    h2_molecule = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.7284]])

    # run calc and convert electronic energy into atomic units 
    h2_molecule.calc = DIRAC(dirac={'.wave function':''},
                             hamiltonian={'.nonrel': '', '.dft': 'b3lyp'},
                             molecule={'*basis': {'.default': 'sto-3g'}}
                             )
    energy_Eh = h2_molecule.get_potential_energy() / Ha

    # compare with the energy obtained using dirac alone => assuming convergence up to ~1e-6
    assert energy_Eh == pytest.approx(-1.1587324754260060, 1e-6)


def test_DIRAC_energy_hf():
    """Test case # 2 - Testing ASE-DIRAC default parameters - HF/sto-3g."""
    
    h_atom = Atoms('H')

    h_atom.calc = DIRAC()
    energy_Eh = h_atom.get_potential_energy() / Ha

    # compare with the energy obtained using dirac alone => assuming convergence up to ~1e-6
    assert energy_Eh == pytest.approx(-0.466581849557275, 1e-6)

def test_DIRAC_energy_mp2():
    """Test case # 3 - MP2/STO-3G Relativistic H2O.
    
    Notes:
        Adapted from the original DIRAC test set.    
    """

    h2o_molecule = molecule('H2O')

    h2o_molecule.calc = DIRAC(hamiltonian={'.lvcorr':''},
                              wave_function={'.scf':'', '.mp2':'',
                                             '*scf': {'.itrint': '5 50', '.maxitr': '25'},
                                             '*mp2cal': {'.occup': '2..5', '.virtual': 'all', '.virthr': '2.0D00'}},
                              molecule={'*basis': {'.default': 'sto-3g'}}
                                             )
    energy_Eh = h2o_molecule.get_potential_energy() / Ha

    # compare with the energy retrieved from DIRAC test set results.
    assert energy_Eh == pytest.approx(-75.043050906542774, 1e-6)


def test_DIRAC_energy_ccsdt():
    """Test case # 4 - CCSD(T)/STO-3G Relativistic H2O.
    
    Notes:
        Adapted from the original DIRAC test set.
        Uses integral transformation option **MOLTRA.
    """

    h2o_molecule = molecule('H2O')

    h2o_molecule.calc = DIRAC(dirac={'.wave function': '', '.4index': ''},
                              hamiltonian={'.lvcorr':''},
                              wave_function={'.scf':'', '.relccsd':'',
                                             '*scf': {'.evccnv': '1.0E-8'}},
                              moltra={'.active': 'all'}
                              )
    energy_Eh = h2o_molecule.get_potential_energy() / Ha

    # compare with the energy retrieved from DIRAC test set results.
    assert energy_Eh == pytest.approx(-75.05762412870262, 1e-6)