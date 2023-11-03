import os
import glob
import pytest
from ase.build import molecule
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qc2.ase import PySCF
from qc2.data import qc2Data

try:
    import pennylane as qml
    from pennylane import numpy as np
except ImportError:
    pytest.skip("Skipping Pennylane tests...",
                allow_module_level=True)


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
    """Create input for H2 and save/load data using QCSchema."""
    # set Atoms object (H2 molecule)
    mol = molecule('H2')

    # file to save data
    hdf5_file = 'h2_ase_pennylane.hdf5'

    # init the hdf5 file
    qc2data = qc2Data(hdf5_file, mol, schema='qcschema')

    # specify the qchem calculator (default => RHF/STO-3G)
    qc2data.molecule.calc = PySCF()

    # run calculation and save qchem data in the hdf5 file
    qc2data.run()

    # define active space
    n_active_electrons = (1, 1)  # (n_alpha, n_beta)
    n_active_spatial_orbitals = 2

    # define the type of fermionic-to-qubit transformation
    mapper = JordanWignerMapper()

    # set up qubit Hamiltonian and core energy based on given active space
    e_core, qubit_op = qc2data.get_qubit_hamiltonian(
        n_active_electrons, n_active_spatial_orbitals, mapper,
        format='pennylane'
    )

    qubits = 2 * n_active_spatial_orbitals
    electrons = sum(n_active_electrons)

    # Define the HF state
    hf_state = qml.qchem.hf_state(electrons, qubits)

    # Generate single and double excitations
    singles, doubles = qml.qchem.excitations(electrons, qubits)

    # Map excitations to the wires the UCCSD circuit will act on
    s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)

    # Define the device
    dev = qml.device("default.qubit", wires=qubits)

    # Define the qnode
    @qml.qnode(dev)
    def circuit(params, wires, s_wires, d_wires, hf_state):
        qml.UCCSD(params, wires, s_wires, d_wires, hf_state)
        return qml.expval(qubit_op)

    # Define the initial values of the circuit parameters
    params = np.zeros(len(singles) + len(doubles))

    # Define the optimizer
    optimizer = qml.GradientDescentOptimizer(stepsize=0.5)

    # Optimize the circuit parameters and compute the energy
    for n in range(21):
        params, energy = optimizer.step_and_cost(
            circuit, params, wires=range(qubits), s_wires=s_wires,
            d_wires=d_wires, hf_state=hf_state
        )

    return energy, e_core


def test_vqe_calculation(vqe_calculation):
    """Check that the final vqe energy corresponds to one at FCI/sto-3g."""
    calculated_electronic_energy, e_core = vqe_calculation
    calculated_energy = calculated_electronic_energy + e_core
    assert calculated_energy == pytest.approx(-1.137301563740087, rel=1e-6)


if __name__ == '__main__':
    pytest.main()
