import os
import glob
import pytest
from ase.build import molecule
from qc2.ase import PySCF
from qc2.data import qc2Data
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.mappers import BravyiKitaevMapper
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit_algorithms.optimizers import SLSQP
from qiskit.primitives import Estimator


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
    # set Atoms object (H2 molecule)
    mol = molecule('CH')

    # file to save data
    fcidump_file = 'CH_ase_qiskit.hdf5'

    # init the hdf5 file
    qc2data = qc2Data(fcidump_file, mol, schema='qcschema')

    # specify the qchem calculator
    qc2data.molecule.calc = PySCF(
        method='scf.UHF',
        multiplicity=2
    )

    # run calculation and save qchem data in the hdf5 file
    qc2data.run()

    # define active space
    n_active_electrons = (3, 2)  # => (n_alpha, n_beta)
    n_active_spatial_orbitals = 5

    # define the type of fermionic-to-qubit transformation
    mapper = BravyiKitaevMapper()

    # set up qubit Hamiltonian and core energy based on given active space
    e_core, qubit_op = qc2data.get_qubit_hamiltonian(
        n_active_electrons, n_active_spatial_orbitals, mapper, format='qiskit'
    )

    reference_state = HartreeFock(
        n_active_spatial_orbitals, n_active_electrons, mapper
    )

    ansatz = UCCSD(
        n_active_spatial_orbitals, n_active_electrons,
        mapper, initial_state=reference_state
    )

    vqe_solver = VQE(Estimator(), ansatz, SLSQP())
    vqe_solver.initial_point = [0.0] * ansatz.num_parameters
    result = vqe_solver.compute_minimum_eigenvalue(qubit_op)

    return result.eigenvalue, e_core


def test_vqe_calculation(vqe_calculation):
    calculated_electronic_energy, e_core = vqe_calculation
    calculated_energy = calculated_electronic_energy+e_core
    assert calculated_energy == pytest.approx(-37.81103335229991, rel=1e-6)


if __name__ == '__main__':
    pytest.main()
