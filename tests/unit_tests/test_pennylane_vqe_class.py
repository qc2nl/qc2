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
    from qc2.algorithms.pennylane import VQE
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
    reference_state = qml.qchem.hf_state(2, 4)

    # set up ansatz
    singles, doubles = qml.qchem.excitations(2, 4)
    s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)

    def ansatz(params):
        qml.UCCSD(
            params,
            wires=range(4),
            s_wires=s_wires,
            d_wires=d_wires,
            init_state=reference_state,
        )

    vqe = VQE(
        ansatz=ansatz,
        active_space=ActiveSpace(
            num_active_electrons=(1, 1),
            num_active_spatial_orbitals=2
        ),
        mapper="bk",
        device="default.qubit",
    )
    assert isinstance(vqe, VQE)



def test_default_ansatz():
    """Test the generation of default ansatz."""
    ansatz, parameters = VQE._get_default_ansatz(None, 4, 2)
    assert isinstance(ansatz, Callable)
    assert isinstance(parameters, np.ndarray)

def test_run_method(vqe):
    """Test main VQE workflow."""
    results = vqe.run()
    assert isinstance(results.optimal_energy, float)
    assert results.optimal_energy == pytest.approx(-1.1373015, 1e-6)
    assert all(num != 0 for num in results.optimal_params)
    assert all(isinstance(num, float) for num in results.energy)
    assert all(isinstance(num, list) for num in results.parameters)
