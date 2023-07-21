"""Tests for the ASE-PySCF interface"""

import os
import subprocess
import pytest

from ase import Atoms
from ase.build import molecule
from ase.units import Ha
import numpy as np
import h5py
from qc2.ase.dirac import DIRAC


def create_test_atoms():
    """Create a test ASE Atoms object."""
    return Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])


@pytest.fixture
def dirac_calculator():
    """Fixture to set up a test instance of the DIRAC calculator."""
    atoms = create_test_atoms()
    calc = DIRAC(molecule={'*basis': {'.default': '6-31g'}})
    atoms.calc = calc
    return atoms


@pytest.fixture(scope="session", autouse=True)
def clean_up_files():
    """Remove DIRAC calculation outputs at the end of each test."""
    yield
    command = "rm *.xyz* *.inp* *.out* *.h5* *.tgz* MDCINT* " \
        "MRCONEE* FCIDUMP* AOMOMAT* FCI*"
    subprocess.run(command, shell=True, capture_output=True)


def test_DIRAC_energy_rks():
    """Test case # 1 - RKS/B3LYP H2 molecule."""

    # define molecule geometry
    h2_molecule = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.7284]])

    # run calc and convert electronic energy into atomic units
    h2_molecule.calc = DIRAC(dirac={'.wave function': ''},
                             hamiltonian={'.nonrel': '', '.dft': 'b3lyp'},
                             molecule={'*basis': {'.default': 'sto-3g'}}
                             )
    energy_Eh = h2_molecule.get_potential_energy() / Ha

    # compare with the energy obtained using dirac alone
    # => assuming convergence up to ~1e-6
    assert energy_Eh == pytest.approx(-1.1587324754260060, 1e-6)


def test_DIRAC_energy_hf():
    """Test case # 2 - Testing ASE-DIRAC default parameters - HF/sto-3g."""

    h_atom = Atoms('H')

    h_atom.calc = DIRAC()
    energy_Eh = h_atom.get_potential_energy() / Ha

    # compare with the energy obtained using dirac alone
    # => assuming convergence up to ~1e-6
    assert energy_Eh == pytest.approx(-0.466581849557275, 1e-6)


def test_DIRAC_energy_mp2():
    """Test case # 3 - MP2/STO-3G Relativistic H2O.

    Notes:
        Adapted from the original DIRAC test set.
    """

    h2o_molecule = molecule('H2O')

    h2o_molecule.calc = DIRAC(hamiltonian={'.lvcorr': ''},
                              wave_function={'.scf': '', '.mp2': '',
                                             '*scf': {'.itrint': '5 50',
                                                      '.maxitr': '25'},
                                             '*mp2cal': {'.occup': '2..5',
                                                         '.virtual': 'all',
                                                         '.virthr': '2.0D00'}},
                              molecule={'*basis': {'.default': 'sto-3g'}})
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
                              hamiltonian={'.lvcorr': ''},
                              wave_function={'.scf': '', '.relccsd': '',
                                             '*scf': {'.evccnv': '1.0E-8'}},
                              moltra={'.active': 'all'}
                              )
    energy_Eh = h2o_molecule.get_potential_energy() / Ha

    # compare with the energy retrieved from DIRAC test set results.
    assert energy_Eh == pytest.approx(-75.05762412870262, 1e-6)


def test_DIRAC_energy_open_shell():
    """Test case # 5 - open shell HF/STO-3G Relativistic C.

    Notes:
        Adapted from the original DIRAC open-shell test set.
    """

    c_atom = Atoms('C')

    c_atom.calc = DIRAC(hamiltonian={'.x2c': ''},
                        integrals={'*readin': {'.uncontracted': "#"}},
                        molecule={'*basis': {'.default': 'sto-3g'},
                                  '*charge': {'.charge': '0'}},
                                  # '*symmetry': {'.nosym': '#'}},
                        wave_function={'.scf': '',
                                       '*scf': {'.closed shell': '4 0',
                                                '.open shell': '2\n1/0,2\n1/0,4',
                                                '.kpsele': '3\n-1 1 -2\n4 0 0\n0 2 0\n0 0 4'}}
    )
    energy_Eh = c_atom.get_potential_energy() / Ha

    # compare with the energy obtained using dirac alone
    # => assuming convergence up to ~1e-6
    assert energy_Eh == pytest.approx(-37.253756513429018, 1e-6)


def test_DIRAC_save_function(dirac_calculator):
    """Test case # 6 - tesing the save method of the DIRAC calculator."""

    # Perform calculation to generate results
    energy = dirac_calculator.get_potential_energy()/Ha

    # Save results to HDF5 file
    hdf5_filename = str('test_save.h5')
    dirac_calculator.calc.save(hdf5_filename)

    # Check if the HDF5 file exists
    assert os.path.isfile(hdf5_filename)

    # Verify the content of the HDF5 file
    with h5py.File(hdf5_filename, 'r') as f:
        # Check if required datasets exist
        assert 'wavefunction/scf_fock_mo_a' in f
        assert 'wavefunction/scf_fock_mo_b' in f
        assert 'wavefunction/scf_eri_mo_aa' in f
        assert 'wavefunction/scf_eri_mo_bb' in f
        assert 'wavefunction/scf_eri_mo_ba' in f
        assert 'wavefunction/scf_eri_mo_ab' in f
        # Check if energy is stored correctly
        assert np.isclose(f.attrs['return_result'], energy)


def test_DIRAC_load_function(dirac_calculator):
    """Test case # 7 - testing the load method of the DIRAC calculator."""

    # Perform calculation to generate results
    energy = dirac_calculator.get_potential_energy()/Ha

    # Save results to HDF5 file
    hdf5_filename = str('test_load.h5')
    dirac_calculator.calc.save(hdf5_filename)

    # Create a new atoms object
    atoms_new = create_test_atoms()

    # Load results from the HDF5 file
    atoms_new.calc = DIRAC()
    atoms_new.calc.load(hdf5_filename)

    # Check if the energy kept in 'return_energy'
    # is equal to the expected energy
    energy_new = atoms_new.calc.return_energy
    assert np.isclose(energy_new, energy)


def test_DIRAC_get_integrals_function(dirac_calculator):
    """Test case # 8 - testing the get_integrals method of the ASE-DIRAC."""

    # Perform calculation to generate results
    dirac_calculator.get_potential_energy()/Ha

    # Calculate integrals
    (e_core, spinor,
     one_body_int, two_body_int) = dirac_calculator.calc.get_integrals()

    # Check the type and content of the integrals
    assert isinstance(e_core, (float, complex))
    assert isinstance(spinor, dict)
    assert isinstance(one_body_int, dict)
    assert isinstance(two_body_int, dict)
    assert len(spinor) > 0
    assert len(one_body_int) > 0
    assert len(two_body_int) > 0
