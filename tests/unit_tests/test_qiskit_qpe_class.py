import os
import pytest

from ase.build import molecule

from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit.primitives import Sampler

from qc2.data import qc2Data
from qc2.ase import PySCF
from qc2.algorithms.utils import ActiveSpace
from qc2.algorithms.qiskit import QPE


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
    """Fixture to set up VQE instance."""
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
    reference_state = HartreeFock(
        num_spatial_orbitals=2,
        num_particles=(1, 1),
        qubit_mapper=JordanWignerMapper(),
    )

    qpe = QPE(
        reference_state=reference_state,
        active_space=ActiveSpace(
            num_active_electrons=(1, 1),
            num_active_spatial_orbitals=2
        ),
        mapper="jw",
        sampler=Sampler(),
    )
    assert isinstance(qpe, QPE)


def test_run_method(qpe):
    """Test main VQE workflow."""
    results = qpe.run()
    assert isinstance(results.optimal_energy, float)
    assert results.optimal_energy == pytest.approx(-0.852942802, 1e-6)
    assert results.phase == pytest.approx(0.75, 1e-6)
    