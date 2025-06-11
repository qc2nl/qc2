from typing import Callable
import os
import pytest

from ase.build import molecule

from qc2.data import qc2Data
from qc2.ase import PySCF
from qc2.algorithms.utils import ActiveSpace

try:
    import pennylane as qml
    from pennylane import numpy as np
    from qc2.algorithms.pennylane import QPE
except ImportError:
    pytest.skip(
        "Skipping PennyLane tests...",
        allow_module_level=True
    )


@pytest.fixture
def qc2data():
    """Fixture to set up qc2Data instance."""
    tmp_filename = str('test_qc2data.h5')
    mol = molecule('H2')
    qc2_data = qc2Data(tmp_filename, mol, schema='qcschema')
    qc2_data.molecule.calc = PySCF()
    yield qc2_data
    os.remove(tmp_filename)


@pytest.fixture
def qpe(qc2data):
    """Fixture to set up QPE instance."""
    qc2data.run()
    active_space = ActiveSpace(
        num_active_electrons=(1, 1),
        num_active_spatial_orbitals=2
    )
    qpe = QPE(qc2data=qc2data, active_space=active_space)
    yield qpe


@pytest.fixture
def active_space():
    return ActiveSpace((1, 1), 2)


def test_initialization(qpe):
    """Test if you can initialize the class."""
    assert isinstance(qpe, QPE)


def test_initialization():
    """Test initialization of the class with custom ansatz."""
    # set up reference state
    reference_state = qml.qchem.hf_state(2, 4)

    qpe = QPE(
        reference_state=reference_state,
        active_space=ActiveSpace(
            num_active_electrons=(1, 1),
            num_active_spatial_orbitals=2
        ),
        mapper="bk",
        device="default.qubit",
    )
    assert isinstance(qpe, QPE)

def test_run_method(qpe):
    """Test main QPE workflow."""
    results = qpe.run()
    assert isinstance(results.optimal_energy, float)
    assert results.optimal_energy == pytest.approx(-0.852942802, 1e-6)
    assert results.phase == pytest.approx(0.75, 1e-6)
