import os
import pytest
import numpy as np

from ase.build import molecule

from qiskit.quantum_info import SparsePauliOp

from qc2.data import qc2Data
from qc2.ase import PySCF
from qc2.algorithms.utils import ActiveSpace
from qc2.algorithms.utils import OrbitalOptimization

# try importing PennyLane and set a flag
try:
    from pennylane.operation import Operator
    pennylane_available = True
except ImportError:
    pennylane_available = False


@pytest.fixture
def qc2data():
    """Fixture to set up qc2Data instance."""
    # Create a temporary file for testing
    tmp_filename = str('test_qc2data.h5')

    # Create an ASE Atoms instance for testing
    mol = molecule('H2')

    # Create the qc2Data instance
    qc2_data = qc2Data(tmp_filename, mol, schema='qcschema')
    qc2_data.molecule.calc = PySCF()
    yield qc2_data

    # Clean up the temporary file after the tests
    os.remove(tmp_filename)


@pytest.fixture
def orbitaloptimization(qc2data):
    """Fixture to set up OrbitalOptimization instance."""
    qc2data.run()
    active_space = ActiveSpace(
        num_active_electrons=(1, 1),
        num_active_spatial_orbitals=2
    )
    oo = OrbitalOptimization(qc2data, active_space)
    yield oo


# Example RDMs for H2
RDM1 = np.array([[1.97495832+0.j, 0.+0.j], [0.+0.j, 0.02504168+0.j]])
RDM2 = np.array(
    [[[[1.97495832+0.j, 0.+0.j], [0.+0.j, 0.+0.j]], 
      [[0.+0.j, -0.222387662+0.j], [2.08166817e-17+0.j, 0.+0.j]]],
     [[[0.+0.j, 2.08166817e-17+0.j], [-0.222387662+0.j, 0.+0.j]],
      [[0.+0.j, 0.+0.j], [0.+0.j, 0.02504168+0.j]]]]
)


def test_initialization(orbitaloptimization):
    """Test if you can initialize the class."""
    assert isinstance(orbitaloptimization, OrbitalOptimization)


@pytest.mark.parametrize(
    "format", [("qiskit"), ("pennylane")]
)
def test_get_transformed_qubit_hamiltonian(qc2data, format):
    """Test ``get_transformed_qubit_hamiltonian`` method."""
    qc2data.run()
    active_space = ActiveSpace(
        num_active_electrons=(1, 1),
        num_active_spatial_orbitals=2
    )

    if format == "pennylane" and not pennylane_available:
        pytest.skip()

    oo = OrbitalOptimization(qc2data, active_space, format=format)
    kappa = [0.0] * oo.n_kappa
    core_energy, qubit_op = oo.get_transformed_qubit_hamiltonian(kappa)

    assert core_energy == pytest.approx(0.717853, 1e-6)
    if format == "qiskit":
        assert isinstance(qubit_op, SparsePauliOp)
    if format == "pennylane":
        assert isinstance(qubit_op, Operator)


@pytest.mark.parametrize(
    "format", [("qiskit"), ("pennylane")]
)
def test_analytic_deviratives(qc2data, format):
    """Test orbital optimization workflow"""
    qc2data.run()
    active_space = ActiveSpace(
        num_active_electrons=(1, 1),
        num_active_spatial_orbitals=2
    )

    if format == "pennylane" and not pennylane_available:
        pytest.skip()

    oo = OrbitalOptimization(qc2data, active_space, format=format)
    kappa_init = [2.24262778e-01]
    gradients = oo.get_analytic_gradients(kappa_init, RDM1, RDM2)
    hessian = oo.get_analytic_hessian(kappa_init, RDM1, RDM2)

    assert isinstance(gradients, np.ndarray)
    assert gradients.shape == (1,)
    assert isinstance(hessian, np.ndarray)
    assert hessian.shape == (1, 1)


@pytest.mark.parametrize(
    "format", [("qiskit"), ("pennylane")]
)
def test_orbital_optimization(qc2data, format):
    """Test orbital optimization workflow"""
    qc2data.run()
    active_space = ActiveSpace(
        num_active_electrons=(1, 1),
        num_active_spatial_orbitals=2
    )

    if format == "pennylane" and not pennylane_available:
        pytest.skip()

    oo = OrbitalOptimization(qc2data, active_space, format=format)
    kappa_init = [0.0] * oo.n_kappa
    optimized_kappa, energy = oo.orbital_optimization(RDM1, RDM2, kappa_init)
    recovered_energy = oo.get_energy_from_kappa(optimized_kappa, RDM1, RDM2)

    assert all(num != 0 for num in optimized_kappa)
    assert energy == pytest.approx(-1.1373015, 1e-6)
    assert recovered_energy == energy
