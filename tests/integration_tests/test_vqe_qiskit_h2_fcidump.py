import os
import glob
import pytest
from ase.build import molecule
from qiskit_algorithms.optimizers import SLSQP
from qiskit.primitives import Estimator
from qc2.ase import PySCF
from qc2.data import qc2Data
from qc2.algorithms.qiskit import VQE
from qc2.algorithms.utils import ActiveSpace


@pytest.fixture(scope="session", autouse=True)
def clean_up_files():
    """Runs at the end of all tests."""
    yield
    # Define the pattern for files to delete
    file_pattern = "*.fcidump"
    # Get a list of files that match the pattern
    matching_files = glob.glob(file_pattern)
    # Loop through the matching files and delete each one
    for file_path in matching_files:
        os.remove(file_path)


@pytest.fixture
def vqe_calculation():
    """Create input for H2 and save/load data using FCIDump schema."""
    # set Atoms object (H2 molecule)
    mol = molecule("H2")

    # file to save data
    fcidump_file = "h2_ase_pyscf_qiskit.fcidump"

    # init the hdf5 file
    qc2data = qc2Data(fcidump_file, mol, schema="fcidump")

    # specify the qchem calculator (default => RHF/STO-3G)
    qc2data.molecule.calc = PySCF()

    # run calculation and save qchem data in the hdf5 file
    qc2data.run()

    # set up VQE calc
    qc2data.algorithm = VQE(
        active_space=ActiveSpace(
            num_active_electrons=(1, 1),
            num_active_spatial_orbitals=2
        ),
        mapper="bk",
        optimizer=SLSQP(),
        estimator=Estimator(),
    )

    # run vqe
    results = qc2data.algorithm.run()

    return results.optimal_energy


def test_vqe_calculation(vqe_calculation):
    """Check that the final vqe energy corresponds to one at FCI/sto-3g."""
    assert vqe_calculation == pytest.approx(-1.137301563740087, rel=1e-6)


if __name__ == '__main__':
    pytest.main()
