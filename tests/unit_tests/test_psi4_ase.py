"""Tests for the ASE-Psi4 interface"""
import os
import glob
import pytest
import h5py
import numpy as np

from ase import Atoms
from ase.units import Ha
from qc2.ase import Psi4


@pytest.fixture(scope="session", autouse=True)
def clean_up_files():
    """Runs at the end of all tests."""
    yield
    # Define the patterns for files to delete
    file_pattern = "*.h5 *.dat"
    # Get a list of files that match the patterns
    matching_files = []
    for pattern in file_pattern.split():
        matching_files.extend(glob.glob(pattern))
    # Loop through the matching files and delete each one
    for file_path in matching_files:
        os.remove(file_path)


def create_test_atoms():
    """Create a test ASE Atoms object."""
    return Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])


@pytest.fixture
def psi4_calculator():
    """Fixture to set up a test instance of the Psi4 calculator."""
    atoms = create_test_atoms()
    calc = Psi4(method='hf', basis='sto-3g')
    atoms.calc = calc
    return atoms


def test_Psi4_save_function(psi4_calculator):
    """Testing the save method of the Psi4 calculator."""
    # Perform calculation to generate results
    energy = psi4_calculator.get_potential_energy()/Ha

    # Save results to HDF5 file
    hdf5_filename = str('test_save.h5')
    psi4_calculator.calc.save(hdf5_filename)

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


def test_Psi4_load_function(psi4_calculator):
    """Testing the load method of the Psi4 calculator."""
    # Perform calculation to generate results
    energy = psi4_calculator.get_potential_energy()/Ha

    # Save results to HDF5 file
    hdf5_filename = str('test_load.h5')
    psi4_calculator.calc.save(hdf5_filename)

    # Create a new atoms object
    atoms_new = create_test_atoms()

    # Load results from the HDF5 file
    atoms_new.calc = Psi4(method='hf', basis='sto-3g')
    qcschema = atoms_new.calc.load(hdf5_filename)

    # Check if the energy kept in 'return_result'
    # is equal to the expected energy
    energy_new = qcschema.return_result
    assert np.isclose(energy_new, energy)


def test_Psi4_get_mo_integrals_function(psi4_calculator):
    """Testing get_mo_integrals method of the ASE-Psi4."""
    # Perform calculation to generate results
    psi4_calculator.get_potential_energy()

    # Calculate integrals
    (one_body_int_a, one_body_int_b,
     two_body_int_aa, two_body_int_bb,
     two_body_int_ab,
     two_body_int_ba) = psi4_calculator.calc.get_integrals_mo_basis()

    # Check the type and content of the integrals
    assert isinstance(one_body_int_a, np.ndarray)
    assert isinstance(one_body_int_b, np.ndarray)
    assert isinstance(two_body_int_aa, np.ndarray)
    assert isinstance(two_body_int_bb, np.ndarray)
    assert isinstance(two_body_int_ab, np.ndarray)
    assert isinstance(two_body_int_ba, np.ndarray)
    assert len(one_body_int_a) > 0
    assert len(one_body_int_b) > 0
    assert len(two_body_int_aa) > 0
    assert len(two_body_int_bb) > 0
    assert len(two_body_int_ab) > 0
    assert len(two_body_int_ba) > 0


def test_Psi4_get_ao_integrals_function(psi4_calculator):
    """Testing get_ao_integrals method of the ASE-Psi4."""
    # Perform calculation to generate results
    psi4_calculator.get_potential_energy()
    one_e_int, two_e_int = psi4_calculator.calc.get_integrals_ao_basis()
    # Add assertions to check the correctness of the returned values.
    assert isinstance(one_e_int, np.ndarray)
    assert isinstance(two_e_int, np.ndarray)


def test_Psi4_get_molecular_orbitals_coefficients(psi4_calculator):
    """Testing get_molecular_orbitals_coefficients of the ASE-Psi4."""
    # Perform calculation to generate results
    psi4_calculator.get_potential_energy()
    (alpha_coeffs,
     beta_coeffs) = psi4_calculator.calc.get_molecular_orbitals_coefficients()
    # Add assertions to check the correctness of the returned coefficients.
    assert isinstance(alpha_coeffs, np.ndarray)
    # for restricted case
    assert isinstance(beta_coeffs, np.ndarray)


def test_Psi4_get_molecular_orbitals_energies(psi4_calculator):
    """Testing get_molecular_orbitals_energies of the ASE-Psi4."""
    # Perform calculation to generate results
    psi4_calculator.get_potential_energy()
    (alpha_energies,
     beta_energies) = psi4_calculator.calc.get_molecular_orbitals_energies()
    print(alpha_energies, beta_energies)
    # Add assertions to check the correctness of the returned energies.
    assert isinstance(alpha_energies, np.ndarray)
    # for restricted case
    assert isinstance(beta_energies, np.ndarray)


def test_get_overlap_matrix(psi4_calculator):
    """Testing get_overlap_matrix of the ASE-Psi4."""
    # Perform calculation to generate results
    psi4_calculator.get_potential_energy()
    overlap_matrix = psi4_calculator.calc.get_overlap_matrix()
    # Add assertions to check the correctness of the returned overlap matrix.
    assert isinstance(overlap_matrix, np.ndarray)
