import os
import pytest

from .rose_test_functions import clean_up, extract_number, read_output

from ase import Atoms
from qc2.ase.rose import Rose
from qc2.ase.pyscf import PySCF


def run_ferrocene_rose_no_avas():
    """Ferrocene mol_frag/relativistic example calculation."""
    # define target molecule
    mol = Atoms('FeC10H10',
                positions=[
                    [ 1.66780866,  0.12282848, -0.27460150],
                    [ 2.24106869,  2.10024551, -0.33511480],
                    [ 3.31616817, -1.00607736, -0.77719710],
                    [ 1.78191728,  1.75217481,  0.97966357],
                    [ 1.16672810,  1.86346489, -1.25728635],
                    [ 2.24424364, -1.24911576, -1.70047730],
                    [ 2.85541376, -1.35028551,  0.53811026],
                    [ 0.42377202,  1.30026137,  0.87001390],
                    [ 0.04359901,  1.36901828, -0.51247557],
                    [ 1.12099915, -1.74357622, -0.95584975],
                    [ 1.49870213, -1.80608183,  0.42766268],
                    [ 3.23295333,  2.46614000, -0.58966882],
                    [ 4.30039797, -0.61794626, -1.02844385],
                    [ 2.36507558,  1.80777918,  1.89582556],
                    [ 1.20126378,  2.02039952, -2.33279589],
                    [ 2.27428727, -1.07886582, -2.77411019],
                    [ 3.42876748, -1.26708471,  1.45836098],
                    [-0.20304943,  0.95502659,  1.68878543],
                    [-0.92220706,  1.08645766, -0.92472956],
                    [ 0.15065240, -2.01282736, -1.36630431],
                    [ 0.86407561, -2.12909803,  1.24950292]
                    ],
                   calculator=PySCF(method='scf.RHF',
                                    basis='unc-sto-3g',
                                    multiplicity=1,
                                    relativistic='x2c',
                                    cart=True,
                                    scf_addons='frac_occ'
                                    )
                   )

    # define atomic/molecular fragments
    frag1 = Atoms('Fe',
                  positions=[[1.66780866, 0.12282848, -0.27460150]],
                  calculator=PySCF(method='scf.RHF',
                                   basis='unc-sto-3g',
                                   multiplicity=1,
                                   cart=True,
                                   relativistic=True,
                                   charge=2,
                                   scf_addons='frac_occ'
                                   )
                   )

    frag2 = Atoms('C5H5',
                positions=[
                    [ 2.24106869,  2.10024551, -0.33511480],
                    [ 1.78191728,  1.75217481,  0.97966357],
                    [ 1.16672810,  1.86346489, -1.25728635],
                    [ 0.42377202,  1.30026137,  0.87001390],
                    [ 0.04359901,  1.36901828, -0.51247557],
                    [ 2.36507558,  1.80777918,  1.89582556],
                    [ 1.20126378,  2.02039952, -2.33279589],
                    [-0.20304943,  0.95502659,  1.68878543],
                    [-0.92220706,  1.08645766, -0.92472956],
                    [ 3.23295333,  2.46614000, -0.58966882]
                    ],
                  calculator=PySCF(method='scf.RHF',
                                   basis='unc-sto-3g',
                                   multiplicity=1,
                                   cart=True,
                                   relativistic=True,
                                   charge=-1,
                                   scf_addons='frac_occ'
                                   )
                  )

    frag3 = Atoms('C5H5',
                  positions=[
                    [1.49870213, -1.80608183,  0.42766268],
                    [2.24424364, -1.24911576, -1.70047730],
                    [3.31616817, -1.00607736, -0.77719710],
                    [1.12099915, -1.74357622, -0.95584975],
                    [2.85541376, -1.35028551,  0.53811026],
                    [4.30039797, -0.61794626, -1.02844385],
                    [2.27428727, -1.07886582, -2.77411019],
                    [3.42876748, -1.26708471,  1.45836098],
                    [0.15065240, -2.01282736, -1.36630431],
                    [0.86407561, -2.12909803,  1.24950292]
                    ],
                  calculator=PySCF(method='scf.RHF',
                                   basis='unc-sto-3g',
                                   multiplicity=1,
                                   cart=True,
                                   relativistic=True,
                                   charge=-1,
                                   scf_addons='frac_occ'
                                   )
                  )

    mol.get_potential_energy()
    frag1.get_potential_energy()
    frag2.get_potential_energy()
    frag3.get_potential_energy()

    # Rose-ASE calculator
    rose_calc = Rose(rose_calc_type='mol_frag',
                     exponent=4,
                     rose_target=mol,
                     rose_frags=[frag1, frag2, frag3],
                     test = True,
                     relativistic = True
                     )
    
    # run the calculator
    rose_calc.calculate()


EXPECTED_OUTPUT_FILE = 'test_rose_ase_pyscf-ferrocene.stdout'
ACTUAL_OUTPUT_FILE = 'OUTPUT_ROSE'

CHARGE_REGEX = 'Partial charges of fragments.*\n{}'.format(('*\n.' * 6) + '*\n')
MO_ENERGIES_REGEX = 'Recanonicalized virtual energies of fragments.*\n{}'.format(('*\n.' * 85) + '*\n')
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
    assert actual_energy == pytest.approx(expected_energy, rel=1.0e-5)
