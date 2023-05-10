import os
import pytest

from .rose_test_functions import clean_up, extract_number, read_output

from ase import Atoms
from qc2.ase.rose import Rose
from qc2.ase.pyscf import PySCF


def run_water_ammonia_rose_no_avas():
    """Water-Ammonia mol_frag example calculation."""

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

    # perform scf calculations on all species
    mol.get_potential_energy()
    frag1.get_potential_energy()
    frag2.get_potential_energy()

    # prepare Rose-ASE calculator
    rose_calc = Rose(rose_calc_type='mol_frag',
                     rose_target=mol,
                     rose_frags=[frag1, frag2],
                     test = True
                     )

    # run the calculator
    rose_calc.calculate()


EXPECTED_OUTPUT_FILE = 'test_rose_ase_pyscf-water-ammonia.stdout'
ACTUAL_OUTPUT_FILE = 'OUTPUT_ROSE'

CHARGE_REGEX = 'Partial charges of fragments.*\n{}'.format(('*\n.' * 6) + '*\n')
MO_ENERGIES_REGEX = 'Recanonicalized virtual energies of fragments.*\n{}'.format(('*\n.' * 9) + '*\n')
HF_ENERGIES_REGEX = 'HF energy .*\n{}'.format(('*\n.' * 2) + '*\n')


def run_calculation():
    """Calculation to run."""
    run_water_ammonia_rose_no_avas()


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
    expected_output = read_output(os.path.join(os.path.dirname(__file__), 'data', EXPECTED_OUTPUT_FILE))
    actual_output = read_output(os.path.join(os.path.dirname(__file__), ACTUAL_OUTPUT_FILE))

    # check the actual output against the expected output
    expected_charges = extract_number(CHARGE_REGEX, expected_output)
    actual_charges = extract_number(CHARGE_REGEX, actual_output)
    assert actual_charges == pytest.approx(expected_charges, rel=1e-3)


def test_virtual_energies():
    """Test case #2 - Recanonicalized virtual energies of fragments."""
    # read the expected and actual outputs
    expected_output = read_output(os.path.join(os.path.dirname(__file__), 'data', EXPECTED_OUTPUT_FILE))
    actual_output = read_output(os.path.join(os.path.dirname(__file__), ACTUAL_OUTPUT_FILE))

    # check the actual output against the expected output
    expected_energies = extract_number(MO_ENERGIES_REGEX, expected_output)
    actual_energies = extract_number(MO_ENERGIES_REGEX, actual_output)
    assert actual_energies == pytest.approx(expected_energies, rel=1.0e-5)


def test_hf_energy():
    """Test case #3 - Check final HF energies, if test = True."""
    # read the expected and actual outputs
    expected_output = read_output(os.path.join(os.path.dirname(__file__), 'data', EXPECTED_OUTPUT_FILE))
    actual_output = read_output(os.path.join(os.path.dirname(__file__), ACTUAL_OUTPUT_FILE))

    # check the actual output against the expected output
    expected_energy = extract_number(HF_ENERGIES_REGEX, expected_output)
    actual_energy = extract_number(HF_ENERGIES_REGEX, actual_output)
    assert actual_energy == pytest.approx(expected_energy, rel=1.0e-8)
