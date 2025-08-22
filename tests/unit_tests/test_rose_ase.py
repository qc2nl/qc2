"""Tests for the ASE-ROSE interface"""
import subprocess
import shutil
import pytest

from qc2.data.fcidump import FCIDump

# try:
#     from qc2.ase import ROSE, ROSETargetMolecule, ROSEFragment
# except ImportError:
#     pytest.skip("Skipping ASE-ROSE tests...",
#                 allow_module_level=True)

from qc2.ase import ROSE, ROSETargetMolecule, ROSEFragment

# also check if the `genibo.x` and `avas.x` executables are available
# if not shutil.which("genibo.x") or not shutil.which("avas.x"):
#     pytest.skip("ROSE executables not found or not in your path. "
#                 "Skipping tests.", allow_module_level=True)


def clean_up():
    """Remove Rose-ASE calculation outputs."""
    command = ("rm *.xyz *.dfcoef DFCOEF* *.inp INPUT* "
               "MOLECULE.XYZ MRCONEE* *dfpcmo DFPCMO* *.fchk "
               "fort.* timer.dat INFO_MOL *.pyscf *.psi4 "
               "*.npy *.clean OUTPUT_AVAS "
               "OUTPUT_* *.chk ILMO*dat *.out *.fcidump")
    subprocess.run(command, shell=True, capture_output=True)


@pytest.fixture(scope="session", autouse=True)
def clean_up_files():
    """Runs always at the end of all tests."""
    yield
    clean_up()


@pytest.fixture
def rose_calculator():
    """Water atom_frag example calculation."""
    NH3OH2 = ROSETargetMolecule(
        name='water-ammonia',
        atoms=[('N', (-1.39559, -0.02156,  0.00004)),
            ('H', (-1.62981,  0.96110, -0.10622)),
            ('H', (-1.86277, -0.51254, -0.75597)),
            ('H', (-1.83355, -0.33077,  0.86231)),
            ('O', (1.56850,  0.10589,  0.00001)),
            ('H', (0.60674, -0.03396, -0.00063)),
            ('H', (1.94052, -0.78000,  0.000))],
        basis='sto-3g'
    )

    NH3 = ROSEFragment(
        name='ammonia',
        atoms=[('N', (-1.39559, -0.02156,  0.00004)),
            ('H', (-1.62981,  0.96110, -0.10622)),
            ('H', (-1.86277, -0.51254, -0.75597)),
            ('H', (-1.83355, -0.33077,  0.86231))],
        basis='sto-3g'
    )

    H2O = ROSEFragment(
        name='water',
        atoms=[('O', (1.56850,  0.10589,  0.00001)),
            ('H', (0.60674, -0.03396, -0.00063)),
            ('H', (1.94052, -0.78000,  0.000))],
        basis='sto-3g'
    )

    H2O_calculator = ROSE(
        rose_calc_type='mol_frag',
        exponent=2,
        rose_target=NH3OH2,
        rose_frags=[NH3, H2O],
        test=True,
        additional_virtuals_cutoff=2.0,
        frag_threshold=10.0,
        frag_valence=[[1, 7], [2, 6]],
        frag_core=[[1, 1], [2, 1]],
        avas_frag=[1],
        nmo_avas=[[2, 3, 4]],
    )
    H2O_calculator.get_potential_energy()

    return H2O_calculator


def test_ROSE_load_function(rose_calculator):
    """Testing the load method of the ROSE-ASE calculator."""
    # Perform calculation to generate results
    rose_calculator.get_potential_energy()

    # Read results from 'ibo.fcidump'
    # fcidump_filename = 'ibo.fcidump'
    # rose_calculator.schema_format = 'fcidump'
    # fcidump = rose_calculator.load(fcidump_filename)
    # assert isinstance(fcidump, FCIDump)
