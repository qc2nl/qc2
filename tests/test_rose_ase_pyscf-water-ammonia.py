import pytest

from .rose_test_functions import clean_stuff
from .rose_test_functions import extract_number
from .rose_test_functions import read_output

from ase import Atoms
from qc2.ase.rose import Rose
from qc2.ase.pyscf import PySCF


EXPECTED_OUTPUT_FILE = 'test_rose_ase_pyscf-water-ammonia.stdout'
ACTUAL_OUTPUT_FILE = 'OUTPUT_ROSE'

CHARGE_REGEX = 'Partial charges of fragments.*\n{}'.format(('*\n.' * 5) + '*\n')
MO_ENERGIES_REGEX = 'Recanonicalized virtual energies of fragments.*\n{}'.format(('*\n.' * 18) + '*\n')
HF_ENERGIES_REGEX = 'HF energy.*\n{}'.format(('*\n.' * 2) + '*\n')

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

def test_partial_charges():
    """Test case #1 - Partial charges of fragments."""

    # clean files from previous runs, if any
    clean_stuff()

    # run Rose calculation just once
    run_water_ammonia_rose_no_avas()

    expected_output = read_output(EXPECTED_OUTPUT_FILE)
    actual_output = read_output(ACTUAL_OUTPUT_FILE)

    expected_charges = extract_number(CHARGE_REGEX, expected_output)
    actual_charges = extract_number(CHARGE_REGEX, actual_output)

    assert actual_charges == pytest.approx(expected_charges, rel=1e-3)

def test_virtual_energies():
    """Test case #2 - Recanonicalized virtual energies of fragments."""
    expected_output = read_output(EXPECTED_OUTPUT_FILE)
    actual_output = read_output(ACTUAL_OUTPUT_FILE)

    expected_energies = extract_number(MO_ENERGIES_REGEX, expected_output)
    actual_energies = extract_number(MO_ENERGIES_REGEX, actual_output)

    assert actual_energies == pytest.approx(expected_energies, rel=1.0e-5)

def test_hf_energy():
    """Test case #3 - Check final HF energies, if test = True."""
    expected_output = read_output(EXPECTED_OUTPUT_FILE)
    actual_output = read_output(ACTUAL_OUTPUT_FILE)

    expected_energy = extract_number(HF_ENERGIES_REGEX, expected_output)
    actual_energy = extract_number(HF_ENERGIES_REGEX, actual_output)

    # clean output files
    clean_stuff()

    assert actual_energy == pytest.approx(expected_energy, rel=1.0e-8)
