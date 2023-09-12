"""Example of a VQE calc using Qiskit-Nature and DIRAC-ASE calculator.

Standard restricted calculation => H2 example.

Notes:
    Requires the installation of qc2, ase, qiskit and h5py.
"""
import subprocess
from ase.build import molecule

import qiskit_nature
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.mappers import BravyiKitaevMapper
from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit.algorithms.optimizers import SLSQP
from qiskit.primitives import Estimator
from scm.plams.interfaces.adfsuite.ase_calculator import AMSCalculator
from qc2.ase import AMS
from qc2.data import qc2Data
from scm import plams 
plams.init()


# Avoid using the deprecated `PauliSumOp` object
qiskit_nature.settings.use_pauli_sum_op = False

# set Atoms object
mol = molecule('H2')

# file to save data
hdf5_file = 'h2_ase_ams_qiskit.hdf5'

# init the hdf5 file
qc2data = qc2Data(hdf5_file, mol)

# specify the qchem calculator
settings = plams.Settings()
settings.input.ADF
settings.input.ADF.AOMat2File = True
settings.input.ams.Task = 'SinglePoint'
qc2data.molecule.calc = AMS(settings=settings, name='h2') 

# run calculation and save qchem data in the hdf5 file
qc2data.run()

# # define activate space
# n_active_electrons = (1, 1)  # => (n_alpha, n_beta)
# n_active_spatial_orbitals = 2

# # define the type of fermionic-to-qubit transformation
# mapper = BravyiKitaevMapper()

# # set up qubit Hamiltonian and core energy based on given activate space
# e_core, qubit_op = qc2data.get_qubit_hamiltonian(n_active_electrons,
#                                                  n_active_spatial_orbitals,
#                                                  mapper, format='qiskit')

# reference_state = HartreeFock(
#     n_active_spatial_orbitals,
#     n_active_electrons,
#     mapper,
# )

# # print(reference_state.draw())

# ansatz = UCCSD(
#     n_active_spatial_orbitals,
#     n_active_electrons,
#     mapper,
#     initial_state=reference_state
# )

# # print(ansatz.draw())

# vqe_solver = VQE(Estimator(), ansatz, SLSQP())
# vqe_solver.initial_point = [0.0] * ansatz.num_parameters
# result = vqe_solver.compute_minimum_eigenvalue(qubit_op)

# print("=== QISKIT VQE RESULTS ===")
# print(f"* Electronic ground state energy (Hartree): {result.eigenvalue}")
# print(f"* Inactive core energy (Hartree): {e_core}")
# print(f">>> Total ground state energy (Hartree): {result.eigenvalue+e_core}\n")

# # print(f"+++ Final parameters:{result.optimal_parameters}")

plams.finish()