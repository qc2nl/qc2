"""Example of a VQE calc using Qiskit-Nature and Psi4-ASE calculator.

Standard restricted calculation => H2O example.

Notes:
    Requires the installation of qc2, ase, psi4, qiskit and h5py.
"""
import subprocess
from ase.build import molecule

import qiskit_nature
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit_algorithms.optimizers import SLSQP
from qiskit.primitives import Estimator

from qc2.ase import Psi4
from qc2.data import qc2Data

# Avoid using the deprecated `PauliSumOp` object
qiskit_nature.settings.use_pauli_sum_op = False


def clean_up_Psi4_files():
    """Remove Psi4 calculation outputs."""
    command = "rm *.dat"
    subprocess.run(command, shell=True, capture_output=True)


# set Atoms object
mol = molecule('H2O')

# file to save data
hdf5_file = 'h2o_ase_psi4_qiskit.fcidump'

# init the hdf5 file
qc2data = qc2Data(hdf5_file, mol, schema='fcidump')

# specify the qchem calculator
qc2data.molecule.calc = Psi4(method='hf', basis='sto-3g')

# run calculation and save qchem data in the hdf5 file
qc2data.run()

# define activate space
n_active_electrons = (2, 2)  # => (n_alpha, n_beta)
n_active_spatial_orbitals = 3

# define the type of fermionic-to-qubit transformation
mapper = JordanWignerMapper()

# set up qubit Hamiltonian and core energy based on given activate space
e_core, qubit_op = qc2data.get_qubit_hamiltonian(n_active_electrons,
                                                 n_active_spatial_orbitals,
                                                 mapper, format='qiskit')

reference_state = HartreeFock(
    n_active_spatial_orbitals,
    n_active_electrons,
    mapper,
)

# print(reference_state.draw())

ansatz = UCCSD(
    n_active_spatial_orbitals,
    n_active_electrons,
    mapper,
    initial_state=reference_state
)

# print(ansatz.draw())

vqe_solver = VQE(Estimator(), ansatz, SLSQP())
vqe_solver.initial_point = [0.0] * ansatz.num_parameters
result = vqe_solver.compute_minimum_eigenvalue(qubit_op)

print("=== QISKIT VQE RESULTS ===")
print(f"* Electronic ground state energy (Hartree): {result.eigenvalue}")
print(f"* Inactive core energy (Hartree): {e_core}")
print(f">>> Total ground state energy (Hartree): {result.eigenvalue+e_core}\n")

# print(f"+++ Final parameters:{result.optimal_parameters}")

clean_up_Psi4_files()
