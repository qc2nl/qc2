import os
import pytest

from ase.build import molecule

from qiskit.circuit import QuantumCircuit
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import UCC
from qiskit_algorithms.minimum_eigensolvers import VQEResult
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.mappers import BravyiKitaevMapper
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Estimator

from qc2.data import qc2Data
from qc2.ase import PySCF
from qc2.algorithms.utils import ActiveSpace
from qc2.algorithms.qiskit import VQE


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
def vqe(qc2data):
    """Fixture to set up VQE instance."""
    qc2data.run()
    active_space = ActiveSpace(
        num_active_electrons=(1, 1),
        num_active_spatial_orbitals=2
    )
    vqe = VQE(qc2data=qc2data, active_space=active_space)
    yield vqe


@pytest.fixture
def active_space():
    return ActiveSpace((1, 1), 2)


def test_initialization(vqe):
    """Test if you can initialize the class."""
    assert isinstance(vqe, VQE)


def test_initialization_with_ansatz():
    """Test initialization of the class with custom ansatz."""
    # set up reference state
    reference_state = HartreeFock(
        num_spatial_orbitals=2,
        num_particles=(1, 1),
        qubit_mapper=BravyiKitaevMapper(),
    )
    # set up ansatz
    ansatz = UCCSD(
        num_spatial_orbitals=2,
        num_particles=(1, 1),
        qubit_mapper=BravyiKitaevMapper(),
        initial_state=reference_state
    )
    vqe = VQE(
        ansatz=ansatz,
        reference_state=reference_state,
        active_space=ActiveSpace(
            num_active_electrons=(1, 1),
            num_active_spatial_orbitals=2
        ),
        mapper="bk",
        optimizer=COBYLA(),
        estimator=Estimator(),
    )
    assert isinstance(vqe, VQE)


def test_default_reference(active_space):
    """Test if default reference state works."""
    reference_state = VQE._get_default_reference(
        active_space, JordanWignerMapper()
    )
    assert isinstance(reference_state, QuantumCircuit)


def test_default_ansatz(active_space):
    """Test the generation of default ansatz."""
    reference_state = VQE._get_default_reference(
        active_space, JordanWignerMapper()
    )
    ansatz = VQE._get_default_ansatz(
        active_space, JordanWignerMapper(), reference_state
    )
    assert isinstance(ansatz, UCC)


def test_default_init_params():
    """Test the initialization of circuit parameters."""
    n_params = 10
    init_params = VQE._get_default_init_params(n_params)
    assert len(init_params) == n_params
    assert all(param == 0.0 for param in init_params)


def test_run_method(vqe):
    """Test main VQE workflow."""
    results = vqe.run()
    assert isinstance(results.optimal_energy, float)
    assert results.optimal_energy == pytest.approx(-1.1373015, 1e-6)
    assert all(num != 0 for num in results.optimal_params)
    assert all(isinstance(num, float) for num in results.energy)
    assert all(isinstance(num, list) for num in results.parameters)
