import subprocess
import shutil
import pytest

from qiskit_algorithms.optimizers import SLSQP
from qiskit.primitives import Estimator

from qc2.data import qc2Data
from qc2.algorithms.qiskit import VQE
from qc2.algorithms.utils import ActiveSpace

try:
    from qc2.ase import ROSE, ROSETargetMolecule, ROSEFragment
except ImportError:
    pytest.skip("Skipping ASE-ROSE test...",
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


def rose_calculator():
    """Create an instance of ROSE calculator for water."""
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
                          save_data=True,
                          restricted=True,
                          openshell=True,
                          rose_mo_calculator='pyscf')
    return h2o_calculator


@pytest.fixture
def vqe_calculation():
    """VQE using FCIDump schema."""
    # set the ROSE file to load
    fcidump_file = 'ibo.fcidump'

    # create an instance of qc2Data
    qc2data = qc2Data(fcidump_file, schema='fcidump')

    # attach the calculator
    qc2data.molecule.calc = rose_calculator()

    # run the calculator
    qc2data.run()

    # set up VQE calc
    qc2data.algorithm = VQE(
        active_space=ActiveSpace(
            num_active_electrons=(2, 2),
            num_active_spatial_orbitals=3
        ),
        mapper="jw",
        optimizer=SLSQP(),
        estimator=Estimator(),
    )

    # run vqe
    results = qc2data.algorithm.run()

    return results.optimal_energy


def test_vqe_calculation(vqe_calculation):
    """Check that the final vqe energy is correct."""
    assert vqe_calculation == pytest.approx(-75.1697111770602, rel=1e-6)


if __name__ == '__main__':
    pytest.main()
