
import os
import glob
import pytest

from ase.build import molecule

from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorEstimator

from qiskit_aer.primitives import EstimatorV2 as aer_EstimatorV2
from qiskit_aer import QasmSimulator
# from qiskit_ibm_runtime import EstimatorV2 as ibm_runtime_EstimatorV2

from qc2.data import qc2Data
from qc2.ase import PySCF

from qc2.algorithms.qiskit import VQE, oo_VQE
from qc2.algorithms.utils import ActiveSpace


AER_BACKEND = QasmSimulator(method="statevector")


@pytest.fixture(scope="session", autouse=True)
def clean_up_files():
    """Runs at the end of all tests."""
    yield
    # Define the patterns for files to delete
    file_pattern = "*.hdf5 *.dat"
    # Get a list of files that match the patterns
    matching_files = []
    for pattern in file_pattern.split():
        matching_files.extend(glob.glob(pattern))
    # Loop through the matching files and delete each one
    for file_path in matching_files:
        os.remove(file_path)


@pytest.fixture(
    params=[
        StatevectorEstimator(),
        aer_EstimatorV2(),
        # ibm_runtime_EstimatorV2(mode=AER_BACKEND)
    ]
)
def qc2_setup(request):
    """Create qc2 input with different estimators."""
    estimator = request.param

    # instantiate qc2Data class
    qc2data = qc2Data(
        molecule=molecule('H2O')
    )

    # set up and run calculator
    qc2data.molecule.calc = PySCF()
    qc2data.run()

    return qc2data, estimator


@pytest.fixture
def vqe_calculation(qc2_setup):
    """Run VQE calculation."""
    qc2data, estimator = qc2_setup

    # instantiate VQE algorithm
    qc2data.algorithm = VQE(
        active_space=ActiveSpace(
            num_active_electrons=(2, 2),
            num_active_spatial_orbitals=3
        ),
        optimizer=COBYLA(),
        estimator=estimator
    )

    # run VQE
    results = qc2data.algorithm.run()
    return results.optimal_energy


@pytest.fixture
def oo_vqe_calculation(qc2_setup):
    """Run oo-VQE calculation."""
    qc2data, estimator = qc2_setup

    # instantiate oo-VQE algorithm
    qc2data.algorithm = oo_VQE(
        active_space=ActiveSpace(
            num_active_electrons=(1, 1),
            num_active_spatial_orbitals=2
        ),
        optimizer=COBYLA(),
        estimator=estimator,
    )

    # run oo-VQE
    results = qc2data.algorithm.run()
    return results.optimal_energy


def test_vqe_calculation(vqe_calculation):
    """Check that the final vqe energy corresponds to one at CASCI/sto-3g."""
    assert vqe_calculation == pytest.approx(-74.9690743578253, rel=1e-6)


def test_oo_vqe_calculation(oo_vqe_calculation):
    """Check that the oo-vqe energy corresponds to one at CASSCF/sto-3g."""
    assert oo_vqe_calculation == pytest.approx(-74.96565745741862, rel=1e-6)
