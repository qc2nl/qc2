import os
import glob
import pytest
from ase.build import molecule
from qc2.ase import PySCF
from qc2.data import qc2Data
from qc2.algorithms.utils import ActiveSpace

try:
    import pennylane as qml
    from pennylane import numpy as np
    from qc2.algorithms.pennylane import VQE
except ImportError:
    pytest.skip("Skipping Pennylane tests...",
                allow_module_level=True)


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
def vqe_calculation():
    """Create input for H2 and save/load data using QCSchema."""
    # set Atoms object (H2 molecule)
    mol = molecule("H2")

    # file to save data
    hdf5_file = "h2_ase_pennylane.hdf5"

    # init the hdf5 file
    qc2data = qc2Data(hdf5_file, mol, schema="qcschema")

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
        mapper="jw",
        optimizer=qml.GradientDescentOptimizer(stepsize=0.5)
    )

    # run the calc
    results = qc2data.algorithm.run()

    return results.optimal_energy


def test_vqe_calculation(vqe_calculation):
    """Check that the final vqe energy corresponds to one at FCI/sto-3g."""
    assert vqe_calculation == pytest.approx(-1.137301563740087, rel=1e-6)


if __name__ == '__main__':
    pytest.main()
