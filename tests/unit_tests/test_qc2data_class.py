"""Tests for the qc2Data class"""
import os
import pytest
from ase import Atoms

from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from pennylane.operation import Operator
from qiskit_nature.second_q.formats.qcschema import QCSchema
from qiskit_nature.second_q.formats.fcidump import FCIDump

from qc2.data.data import qc2Data
from qc2.ase.pyscf import PySCF


@pytest.fixture
def qc2_data_qcschema_instance():
    """Fixture to set up qc2Data instance."""
    # Create a temporary file for testing
    tmp_filename = str('test_qc2data.h5')

    # Create an ASE Atoms instance for testing
    atoms = Atoms(symbols="H2", positions=[(0, 0, 0), (0, 0, 0.74)])

    # Create the qc2Data instance
    qc2_data = qc2Data(tmp_filename, atoms, schema='qcschema')
    qc2_data.molecule.calc = PySCF()
    yield qc2_data

    # Clean up the temporary file after the tests
    os.remove(tmp_filename)


@pytest.fixture
def qc2_data_fcidump_instance():
    """Fixture to set up qc2Data instance."""
    # Create a temporary file for testing
    tmp_filename = str('test_qc2data.fcidump')

    # Create an ASE Atoms instance for testing
    atoms = Atoms(symbols="H2", positions=[(0, 0, 0), (0, 0, 0.74)])

    # Create the qc2Data instance
    qc2_data = qc2Data(tmp_filename, atoms, schema='fcidump')
    qc2_data.molecule.calc = PySCF()
    yield qc2_data

    # Clean up the temporary file after the tests
    os.remove(tmp_filename)


def test_init(qc2_data_qcschema_instance, qc2_data_fcidump_instance):
    """Test case # 1 - Testing molecule attribute."""
    qc2_data_qcschema_instance.run()
    qc2_data_fcidump_instance.run()
    assert isinstance(qc2_data_qcschema_instance.molecule, Atoms)
    assert isinstance(qc2_data_fcidump_instance.molecule, Atoms)


def test_run(qc2_data_qcschema_instance, capsys):
    """Test case # 2 - Testing run method."""
    qc2_data_qcschema_instance.run()
    captured = capsys.readouterr()
    main_out = f"Saving qchem data in {qc2_data_qcschema_instance._filename}"
    assert "Reference energy (Hartree)" in captured.out
    assert main_out in captured.out


def test_read_schema(qc2_data_qcschema_instance, qc2_data_fcidump_instance):
    """Test case # 3 - Populating QCSchema and FCIDump dataclasses."""
    qc2_data_qcschema_instance.run()
    qc2_data_fcidump_instance.run()
    qcschema = qc2_data_qcschema_instance.read_schema()
    fcidump = qc2_data_fcidump_instance.read_schema()
    assert qcschema is not None
    assert fcidump is not None
    assert isinstance(qcschema, QCSchema)
    assert isinstance(fcidump, FCIDump)


def test_get_active_space_hamiltonian_qcschema(qc2_data_qcschema_instance):
    """Test case # 4a - Building Active-space Hamiltonian from qcschema."""
    num_electrons = (1, 1)
    num_spatial_orbitals = 2
    qc2_data_qcschema_instance.run()
    (es_problem, core_energy, active_space_hamiltonian
     ) = qc2_data_qcschema_instance.get_active_space_hamiltonian(
        num_electrons, num_spatial_orbitals
    )
    assert isinstance(core_energy, float)
    assert isinstance(es_problem, ElectronicStructureProblem)
    assert isinstance(active_space_hamiltonian, ElectronicEnergy)
    assert core_energy == pytest.approx(0.7151043390810812, 1e-6)


def test_get_active_space_hamiltonian_fcidump(qc2_data_fcidump_instance):
    """Test case # 4b - Building Active-space Hamiltonian from fcidump."""
    num_electrons = (1, 1)
    num_spatial_orbitals = 2
    qc2_data_fcidump_instance.run()
    (es_problem, core_energy, active_space_hamiltonian
     ) = qc2_data_fcidump_instance.get_active_space_hamiltonian(
        num_electrons, num_spatial_orbitals
    )
    assert isinstance(core_energy, float)
    assert isinstance(es_problem, ElectronicStructureProblem)
    assert isinstance(active_space_hamiltonian, ElectronicEnergy)
    assert core_energy == pytest.approx(0.7151043390810812, 1e-6)


def test_get_fermionic_hamiltonian(qc2_data_fcidump_instance):
    """Test case # 5 - Building fermionic Hamiltonian."""
    num_electrons = (1, 1)
    num_spatial_orbitals = 2
    qc2_data_fcidump_instance.run()
    (es_problem, core_energy,
     second_q_op) = qc2_data_fcidump_instance.get_fermionic_hamiltonian(
        num_electrons=num_electrons, num_spatial_orbitals=num_spatial_orbitals
    )
    assert isinstance(core_energy, float)
    assert isinstance(es_problem, ElectronicStructureProblem)
    assert isinstance(second_q_op, FermionicOp)


def test_get_qubit_hamiltonian_qiskit(qc2_data_fcidump_instance):
    """Test case # 6 - Building qubit Hamiltonian using qiskit format."""
    num_electrons = (1, 1)
    num_spatial_orbitals = 2
    qc2_data_fcidump_instance.run()
    core_energy, qubit_op = qc2_data_fcidump_instance.get_qubit_hamiltonian(
        num_electrons=num_electrons, num_spatial_orbitals=num_spatial_orbitals,
        format="qiskit"
    )
    assert isinstance(core_energy, float)
    assert isinstance(qubit_op, SparsePauliOp)


def test_get_qubit_hamiltonian_pennylane(qc2_data_qcschema_instance):
    """Test case # 7 - Building qubit Hamiltonian using pennylane format."""
    num_electrons = (1, 1)
    num_spatial_orbitals = 2
    qc2_data_qcschema_instance.run()
    core_energy, qubit_op = qc2_data_qcschema_instance.get_qubit_hamiltonian(
        num_electrons=num_electrons, num_spatial_orbitals=num_spatial_orbitals,
        format="pennylane"
    )
    assert isinstance(core_energy, float)
    assert isinstance(qubit_op, Operator)
