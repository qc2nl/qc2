import os
import glob
import pytest
from ase.build import molecule
from qc2.data import qc2Data
from qc2.algorithms.utils import ActiveSpace

try:
    import pennylane as qml
    from pennylane import numpy as np
    from qc2.algorithms.pennylane import VQE
except ImportError:
    pytest.skip("Skipping Pennylane tests...",
                allow_module_level=True)

try:
    from qc2.ase import Psi4
except ImportError:
    pytest.skip("Skipping ASE-Psi4 tests...",
                allow_module_level=True)


@pytest.fixture(scope="session", autouse=True)
def clean_up_files():
    """Runs at the end of all tests."""
    yield
    # Define the patterns for files to delete
    file_pattern = "*.fcidump *.dat"
    # Get a list of files that match the patterns
    matching_files = []
    for pattern in file_pattern.split():
        matching_files.extend(glob.glob(pattern))
    # Loop through the matching files and delete each one
    for file_path in matching_files:
        os.remove(file_path)


@pytest.fixture
def vqe_calculation():
    """Create input for H2O and save/load data using FCIdump."""
    # set Atoms object (H2 molecule)
    mol = molecule("H2O")

    # file to save data
    fcidump_file = "h2o_ase_pennylane.fcidump"

    # init the hdf5 file
    qc2data = qc2Data(fcidump_file, mol, schema="fcidump")

    # specify the qchem calculator (default => RHF/STO-3G)
    qc2data.molecule.calc = Psi4(method="hf", basis="sto-3g")

    # run calculation and save qchem data in the hdf5 file
    qc2data.run()

    # set up VQE calc
    qc2data.algorithm = VQE(
        active_space=ActiveSpace(
            num_active_electrons=(2, 2),
            num_active_spatial_orbitals=3
        ),
        mapper="jw",
        optimizer=qml.GradientDescentOptimizer(stepsize=0.5)
    )

    # run the calc
    results = qc2data.algorithm.run()

    return results.optimal_energy


def test_vqe_calculation(vqe_calculation):
    """Check that the final vqe energy corresponds to one at CASCI/sto-3g."""
    assert vqe_calculation == pytest.approx(-74.9690743578253, rel=1e-6)


if __name__ == '__main__':
    pytest.main()
