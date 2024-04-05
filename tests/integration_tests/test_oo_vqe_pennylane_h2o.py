
import os
import glob
import pytest

from ase.build import molecule

from qc2.ase import PySCF
from qc2.data import qc2Data
from qc2.algorithms.pennylane import oo_VQE
from qc2.algorithms.utils import ActiveSpace


@pytest.fixture(scope="session", autouse=True)
def clean_up_files():
    """Runs at the end of all tests."""
    yield
    # Define the patterns for files to delete
    file_pattern = "*.hdf5"
    # Get a list of files that match the patterns
    matching_files = []
    for pattern in file_pattern.split():
        matching_files.extend(glob.glob(pattern))
    # Loop through the matching files and delete each one
    for file_path in matching_files:
        os.remove(file_path)


@pytest.fixture
def oo_vqe_calculation():
    """Create input for oo-VQE on H2O."""

    # instantiate qc2Data class
    qc2data = qc2Data(
        molecule=molecule('H2O'),
        filename='h2o.hdf5',
        schema='qcschema'
    )

    # set up and run calculator
    qc2data.molecule.calc = PySCF()
    qc2data.run()

    # instantiate oo-VQE algorithm
    qc2data.algorithm = oo_VQE(
        active_space=ActiveSpace(
            num_active_electrons=(1, 1),
            num_active_spatial_orbitals=2
        ),
    )

    # run oo-VQE
    results = qc2data.algorithm.run()
    return results.optimal_energy


def test_oo_vqe_calculation(oo_vqe_calculation):
    """Check that the oo-vqe energy corresponds to one at CASSCF/sto-3g."""
    assert oo_vqe_calculation == pytest.approx(-74.96565745741862, rel=1e-6)
