import os
import glob
import pytest
from ase import Atoms
from qc2.data import qc2Data
from qiskit_nature.second_q.circuit.library import HartreeFock, UCC
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit_algorithms.optimizers import SLSQP
from qiskit.primitives import Estimator
from qc2.ase import PySCF


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
    mol = Atoms('C')
    hdf5_file = 'carbon_ase_pyscf_qiskit.hdf5'

    qc2data = qc2Data(hdf5_file, mol, schema='qcschema')
    qc2data.molecule.calc = PySCF(
        method='scf.UHF', basis='sto-3g', multiplicity=3, charge=0
    )
    qc2data.run()

    n_active_electrons = (4, 2)  # => (n_alpha, n_beta)
    n_active_spatial_orbitals = 5
    mapper = JordanWignerMapper()

    e_core, qubit_op = qc2data.get_qubit_hamiltonian(
        n_active_electrons, n_active_spatial_orbitals, mapper, format='qiskit'
    )
    reference_state = HartreeFock(
        n_active_spatial_orbitals, n_active_electrons, mapper
    )
    ansatz = UCC(
        num_spatial_orbitals=n_active_spatial_orbitals,
        num_particles=n_active_electrons,
        qubit_mapper=mapper, initial_state=reference_state, excitations='sdtq'
    )
    vqe_solver = VQE(Estimator(), ansatz, SLSQP())
    vqe_solver.initial_point = [0.0] * ansatz.num_parameters
    result = vqe_solver.compute_minimum_eigenvalue(qubit_op)
    return result.eigenvalue + e_core


@pytest.mark.skip(reason="Takes a long time because of the TQ excitations...")
def test_total_ground_state_energy(vqe_result):
    """Check that the final vqe energy corresponds to one at FCI/sto-3g."""
    expected_energy = -37.218733550636
    assert pytest.approx(vqe_result, rel=1e-6) == expected_energy
