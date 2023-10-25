"""Tests for the ASE-PySCF interface"""
import os
import pytest
import h5py
import numpy as np

from ase import Atoms
from ase.build import molecule
from ase.optimize import BFGS
from ase.units import Ha
from ase.calculators.calculator import InputError
from ase.calculators.calculator import CalculatorSetupError
from qc2.ase.pyscf import PySCF


def create_test_atoms():
    """Create a test ASE Atoms object."""
    return Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])


@pytest.fixture
def pyscf_calculator():
    """Fixture to set up a test instance of the PySCF calculator."""
    atoms = create_test_atoms()
    calc = PySCF()
    atoms.calc = calc
    return atoms


def test_PySCF_energy_rks():
    """Test case # 1 - RKS H2 molecule."""

    # define molecule geometry
    h2_molecule = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.7]])

    # run calc and convert electronic energy into atomic units 
    h2_molecule.calc = PySCF(method='dft.RKS', xc='pbe', basis='sto-3g',
                             charge=0, multiplicity=1, verbose=0)
    energy_Eh = h2_molecule.get_potential_energy() / Ha

    # compare with the energy obtained using pyscf alone
    # => assuming convergence up to ~1e-6
    assert energy_Eh == pytest.approx(-1.15101082903195, 1e-6)


def test_PySCF_energy_uhf():
    """Test case # 2 - UHF H atom."""

    h_atom = Atoms('H')

    h_atom.calc = PySCF(method='scf.UHF', basis='sto-3g', charge=0,
                        multiplicity=2, verbose=0)
    energy_Eh = h_atom.get_potential_energy() / Ha

    assert energy_Eh == pytest.approx(-0.466581849557275, 1e-6)


def test_PySCF_energy_rohf():
    """Test case # 3 - ROHF O2 molecule."""

    # now using the built-in molecule dataset to define the geometry
    o2_molecule = molecule('O2')

    o2_molecule.calc = PySCF(method='scf.ROHF', basis='sto-3g', charge=0,
                             multiplicity=3, verbose=0)
    energy_Eh = o2_molecule.get_potential_energy() / Ha

    assert energy_Eh == pytest.approx(-147.630640704849, 1e-6)


def test_PySCF_energy_opt():
    """Test case # 4 - RKS geometry optimization H2 molecule."""

    # define geometry and wave function
    h2_molecule = molecule('H2')
    h2_molecule.calc = PySCF(method='dft.RKS', xc='pbe', basis='sto-3g',
                             charge=0, multiplicity=1, verbose=0)

    # run optimization and get the energy at the minium
    opt = BFGS(h2_molecule)
    opt.run(fmax=0.1)
    energy_opt = h2_molecule.get_potential_energy() / Ha

    # Calculate the gradient if needed
    h2_molecule.get_forces()

    # compare with the energy obtained using pyscf
    # alone via pyberny optimization
    assert energy_opt == pytest.approx(-1.15209884495, 1e-6)


def test_PySCF_energy_rel():
    """Test case # 5 - C atom with relativistic and cartesian basis sets."""

    c_atom = Atoms('C')

    c_atom.calc = PySCF(method='scf.UHF', basis='cc-pVDZ', multiplicity=3,
                        relativistic=True, cart=True)
    energy_Eh = c_atom.get_potential_energy() / Ha

    assert energy_Eh == pytest.approx(-37.70098106211602, 1e-6)


def test_PySCF_energy_scf_addons():
    """Test case # 6 - C atom with frac occupancy for degenerated HOMOs."""

    c_atom = Atoms('C')

    c_atom.calc = PySCF(method='scf.UHF', multiplicity=3,
                        scf_addons='frac_occ')
    energy_Eh = c_atom.get_potential_energy() / Ha

    assert energy_Eh == pytest.approx(-37.01038363165726, 1e-6)


def test_PySCF_with_attribute_error():
    """Test case # 7 - Initializing attribute with unrecognized name."""

    with pytest.raises(InputError) as excinfo:

        # setting the basis attribute with an unrecognized name
        mol = Atoms()
        mol.calc = PySCF(bas='sto-3g')
        mol.get_potential_energy()

    assert 'not recognized' in str(excinfo.value)


def test_PySCF_with_wf_error():
    """Test case # 8 - Setting wave function not yet implemented."""

    with pytest.raises(CalculatorSetupError) as excinfo:

        mol = Atoms()
        mol.calc = PySCF(method='mp.MP2')
        mol.get_potential_energy()

    assert 'Method not yet implemented' in str(excinfo.value)


def test_PySCF_save_function(pyscf_calculator):
    """Test case # 9 - tesing the save method of the PySCF calculator."""

    # Perform calculation to generate results
    energy = pyscf_calculator.get_potential_energy()/Ha

    # Save results to HDF5 file
    hdf5_filename = str('test_save.h5')
    pyscf_calculator.calc.save(hdf5_filename)

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


def test_PySCF_load_function(pyscf_calculator):
    """Test case # 10 - testing the load method of the PySCF calculator."""

    # Perform calculation to generate results
    energy = pyscf_calculator.get_potential_energy()/Ha

    # Save results to HDF5 file
    hdf5_filename = str('test_load.h5')
    pyscf_calculator.calc.save(hdf5_filename)

    # Create a new atoms object
    atoms_new = create_test_atoms()

    # Load results from the HDF5 file
    atoms_new.calc = PySCF()
    qcschema = atoms_new.calc.load(hdf5_filename)

    # Check if the energy kept in 'return_result'
    # is equal to the expected energy
    energy_new = qcschema.return_result
    assert np.isclose(energy_new, energy)


def test_PySCF_get_integrals_function(pyscf_calculator):
    """Test case # 11 - testing get_integrals method of the ASE-PySCF."""

    # Perform calculation to generate results
    pyscf_calculator.get_potential_energy()/Ha

    # Calculate integrals
    (one_body_int_a, one_body_int_b,
     two_body_int_aa, two_body_int_bb,
     two_body_int_ab, two_body_int_ba) = pyscf_calculator.calc.get_integrals_mo_basis()

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
