import os
import glob
import pytest
from ase import Atoms

from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_algorithms.optimizers import SLSQP
from qiskit.primitives import Estimator

from qc2.data import qc2Data
from qc2.ase import PySCF
from qc2.algorithms.qiskit import VQE
from qc2.algorithms.utils import ActiveSpace


@pytest.fixture(scope="session", autouse=True)
def clean_up_files():
    """Runs at the end of all tests."""
    yield
    # Define the pattern for files to delete
    file_pattern = "*.hdf5"
    # Get a list of files that match the pattern
    matching_files = glob.glob(file_pattern)
    # Loop through the matching files and delete each one
    for file_path in matching_files:
        os.remove(file_path)


@pytest.fixture
def vqe_result():
    """Create input for atomic C and and save/load data using QCSchema."""
    # Input data
    mol = Atoms("C")
    hdf5_file = "carbon_ase_pyscf_qiskit.hdf5"

    qc2data = qc2Data(hdf5_file, mol, schema="qcschema")
    qc2data.molecule.calc = PySCF(
        method='scf.UHF', basis='sto-3g', multiplicity=3, charge=0
    )
    qc2data.run()

    # set up VQE calc
    qc2data.algorithm = VQE(
        active_space=ActiveSpace(
            num_active_electrons=(4, 2),
            num_active_spatial_orbitals=5
        ),
        mapper="jw",
        optimizer=SLSQP(),
        estimator=Estimator(),
    )

    # run vqe
    results = qc2data.algorithm.run()

    return results.optimal_energy


def test_total_ground_state_energy(vqe_result):
    """Check that the final vqe energy corresponds to one at FCI/sto-3g."""
    expected_energy = -37.218733550636
    assert pytest.approx(vqe_result, rel=1e-6) == expected_energy
