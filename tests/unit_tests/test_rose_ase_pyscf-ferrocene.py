import os
import shutil
import pytest
from qc2.ase.rose import Rose
from qc2.ase.pyscf import PySCF
from ase.io import read

from .rose_test_functions import clean_up, extract_number, read_output


# Check first if the `genibo.x` and `avas.x` executables are available
if not shutil.which("gen.x") or not shutil.which("avas.x"):
    pytest.skip("ROSE executables not found or not in your path. "
                "Skipping tests.", allow_module_level=True)


def run_ferrocene_rose_no_avas():
    """Ferrocene mol_frag/relativistic example calculation.

    Notes:
        Input Atoms read from files.
    """
    # define target molecule
    mol = read("data/Ferrocene.xyz")
    mol.calc = PySCF(method='scf.RHF',
                     basis='unc-sto-3g',
                     multiplicity=1,
                     relativistic=True,
                     cart=True,
                     scf_addons='frac_occ'
                     )

    # define atomic/molecular fragments
    frag1 = read("data/Ferrocene_frag1.xyz")
    frag1.calc = PySCF(method='scf.RHF',
                       basis='unc-sto-3g',
                       multiplicity=1,
                       cart=True,
                       relativistic=True,
                       charge=2,
                       scf_addons='frac_occ')

    frag2 = read("data/Ferrocene_frag2.xyz")
    frag2.calc = PySCF(method='scf.RHF',
                       basis='unc-sto-3g',
                       multiplicity=1,
                       cart=True,
                       relativistic=True,
                       charge=-1,
                       scf_addons='frac_occ')

    frag3 = read("data/Ferrocene_frag3.xyz")
    frag3.calc = PySCF(method='scf.RHF',
                       basis='unc-sto-3g',
                       multiplicity=1,
                       cart=True,
                       relativistic=True,
                       charge=-1,
                       scf_addons='frac_occ')

    mol.get_potential_energy()
    frag1.get_potential_energy()
    frag2.get_potential_energy()
    frag3.get_potential_energy()

    # Rose-ASE calculator
    rose_calc = Rose(rose_calc_type='mol_frag',
                     exponent=4,
                     rose_target=mol,
                     rose_frags=[frag1, frag2, frag3],
                     test=True,
                     relativistic=True
                     )

    # run the calculator
    rose_calc.calculate()


EXPECTED_OUTPUT_FILE = 'test_rose_ase_pyscf-ferrocene.stdout'
ACTUAL_OUTPUT_FILE = 'OUTPUT_ROSE'

CHARGE_REGEX = 'Partial charges of fragments.*\n{}'.format(
    ('*\n.' * 6) + '*\n')
MO_ENERGIES_REGEX = 'Recanonicalized virtual energies of fragments.*\n{}'. \
    format(('*\n.' * 85) + '*\n')
HF_ENERGIES_REGEX = 'HF energy .*\n{}'.format(('*\n.' * 2) + '*\n')


def run_calculation():
    """Calculation to run."""
    run_ferrocene_rose_no_avas()


@pytest.fixture(scope="session", autouse=True)
def clean_up_files():
    """Runs always at the end of each test."""
    yield
    clean_up()


def test_partial_charges():
    """Test case #1 - Partial charges of fragments."""
    # run the calculation
    run_calculation()

    # read the expected and actual outputs
    expected_output = read_output(os.path.join(
        os.path.dirname(__file__), 'data', EXPECTED_OUTPUT_FILE)
    )
    actual_output = read_output(os.path.join(
        os.path.dirname(__file__), ACTUAL_OUTPUT_FILE)
    )

    # check the actual output against the expected output
    expected_charges = extract_number(CHARGE_REGEX, expected_output)
    actual_charges = extract_number(CHARGE_REGEX, actual_output)
    assert actual_charges == pytest.approx(expected_charges, rel=1e-3)


def test_virtual_energies():
    """Test case #2 - Recanonicalized virtual energies of fragments."""
    # read the expected and actual outputs
    expected_output = read_output(
        os.path.join(os.path.dirname(__file__), 'data', EXPECTED_OUTPUT_FILE)
    )
    actual_output = read_output(
        os.path.join(os.path.dirname(__file__), ACTUAL_OUTPUT_FILE)
    )

    # check the actual output against the expected output
    expected_energies = extract_number(MO_ENERGIES_REGEX, expected_output)
    actual_energies = extract_number(MO_ENERGIES_REGEX, actual_output)
    assert actual_energies == pytest.approx(expected_energies, rel=1.0e-5)


def test_hf_energy():
    """Test case #3 - Check final HF energies, if test = True."""
    # read the expected and actual outputs
    expected_output = read_output(
        os.path.join(os.path.dirname(__file__), 'data', EXPECTED_OUTPUT_FILE)
    )
    actual_output = read_output(
        os.path.join(os.path.dirname(__file__), ACTUAL_OUTPUT_FILE)
    )

    # check the actual output against the expected output
    expected_energy = extract_number(HF_ENERGIES_REGEX, expected_output)
    actual_energy = extract_number(HF_ENERGIES_REGEX, actual_output)
    assert actual_energy == pytest.approx(expected_energy, rel=1.0e-5)
