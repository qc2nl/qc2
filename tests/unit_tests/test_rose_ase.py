"""Tests for the ASE-ROSE interface"""
import subprocess
import shutil
import pytest

from qiskit_nature.second_q.formats.fcidump import FCIDump

try:
    from qc2.ase import ROSE, ROSETargetMolecule, ROSEFragment
except ImportError:
    pytest.skip("Skipping ASE-ROSE tests...",
                allow_module_level=True)

# also check if the `genibo.x` and `avas.x` executables are available
if not shutil.which("genibo.x") or not shutil.which("avas.x"):
    pytest.skip("ROSE executables not found or not in your path. "
                "Skipping tests.", allow_module_level=True)


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
    h2o = ROSETargetMolecule(
        name='water',
        atoms=[('O', (0.,  0.00000,  0.59372)),
               ('H', (0.,  0.76544, -0.00836)),
               ('H', (0., -0.76544, -0.00836))],
        basis='sto-3g'
    )

    oxygen = ROSEFragment(
        name='oxygen',
        atoms=[('O', (0, 0, 0))],
        multiplicity=1, basis='sto-3g'
    )

    hydrogen = ROSEFragment(
        name='hydrogen',
        atoms=[('H', (0, 0, 0))],
        multiplicity=2, basis='sto-3g'
    )

    h2o_calculator = ROSE(rose_calc_type='atom_frag',
                          exponent=4,
                          rose_target=h2o,
                          rose_frags=[oxygen, hydrogen],
                          test=True,
                          avas_frag=[0],
                          nmo_avas=[3, 4, 5],
                          save_data=True,
                          restricted=True,
                          openshell=True,
                          rose_mo_calculator='pyscf')
    return h2o_calculator


def test_ROSE_load_function(rose_calculator):
    """Testing the load method of the ROSE-ASE calculator."""
    # Perform calculation to generate results
    rose_calculator.get_potential_energy()

    # Read results from 'ibo.fcidump'
    fcidump_filename = 'ibo.fcidump'
    rose_calculator.schema_format = 'fcidump'
    fcidump = rose_calculator.load(fcidump_filename)
    assert isinstance(fcidump, FCIDump)
