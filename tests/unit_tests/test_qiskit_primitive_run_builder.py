import os
import pytest

from ase.build import molecule

from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD

from qiskit.primitives import Estimator, StatevectorEstimator
from qiskit.primitives import PrimitiveJob

from qiskit_aer.primitives import Estimator as aer_Estimator
from qiskit_aer.primitives import EstimatorV2 as aer_EstimatorV2
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import Estimator as ibm_runtime_Estimator
from qiskit_ibm_runtime import EstimatorV2 as ibm_runtime_EstimatorV2
from qiskit_ibm_runtime.fake_provider import FakeManilaV2

from qc2.data import qc2Data
from qc2.ase import PySCF
from qc2.algorithms.qiskit import EstimatorRunBuilder


AER_BACKEND = AerSimulator(method="statevector")


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
def quantum_circuit(qc2data):
    """Fixture to get the quantum circuit."""
    # run HF
    qc2data.run()
    # set up reference state
    reference_state = HartreeFock(
        num_spatial_orbitals=2,
        num_particles=(1, 1),
        qubit_mapper=JordanWignerMapper(),
    )
    # set up ansatz
    circuit = UCCSD(
        num_spatial_orbitals=2,
        num_particles=(1, 1),
        qubit_mapper=JordanWignerMapper(),
        initial_state=reference_state
    )
    yield circuit


@pytest.fixture
def hamiltonian(qc2data):
    """Fixture to build up the qubit Hamiltonian."""
    # get qubit hamiltonian
    nuc_repulsion, qubit_ham = qc2data.get_qubit_hamiltonian(
        num_electrons=(1, 1),
        num_spatial_orbitals=2,
        mapper=JordanWignerMapper(),
        format='qiskit'
    )
    yield qubit_ham, nuc_repulsion


@pytest.mark.parametrize(
    "estimator", [
        Estimator(),
        StatevectorEstimator(),
        aer_Estimator(),
        aer_EstimatorV2(),
        ibm_runtime_Estimator(mode=AER_BACKEND),
        ibm_runtime_EstimatorV2(mode=AER_BACKEND)
    ]
)
def test_initialization(estimator, quantum_circuit, hamiltonian):
    """Test if you can initialize the class."""
    parameters = [0.0] * quantum_circuit.num_parameters
    primitive_run = EstimatorRunBuilder(
        estimator,
        [quantum_circuit],
        [hamiltonian[0]],
        [parameters],
    )
    assert isinstance(primitive_run, EstimatorRunBuilder)
    assert primitive_run.provenance, (
        estimator.__class__.__module__.split(".")[0],
        estimator.__class__.__name__
    )


@pytest.mark.parametrize(
    "estimator", [
        StatevectorEstimator(),
        aer_EstimatorV2(),
        ibm_runtime_EstimatorV2(mode=FakeManilaV2())
    ]
)
def test_build_run(estimator, quantum_circuit, hamiltonian):
    """Test if you can build and run primitive jobs."""
    parameters = [0.0] * quantum_circuit.num_parameters
    primitive_run = EstimatorRunBuilder(
        estimator,
        [quantum_circuit],
        [hamiltonian[0]],
        [parameters],
    )
    job = primitive_run.build_run()

    result = job.result()
    nuc_repulsion = hamiltonian[1]
    expectation_value = result[0].data.evs + nuc_repulsion

    assert isinstance(job, PrimitiveJob)
    assert expectation_value is not None
