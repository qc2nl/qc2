"""Example of a VQE calc using Qiskit-Nature and PYSCF-ASE as calculator.

Test case for C atom as an example of a triplet (unrestricted)
calculation.

Notes:
    Requires the installation of qc2, ase, qiskit and h5py.
"""
from ase import Atoms

import qiskit_nature
from qiskit_nature.second_q.circuit.library import HartreeFock, UCC
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit.algorithms.optimizers import SLSQP
from qiskit.primitives import Estimator

from qc2.ase import PySCF
from qc2.data import qc2Data

# Avoid using the deprecated `PauliSumOp` object
qiskit_nature.settings.use_pauli_sum_op = False


# set Atoms object
mol = Atoms('C')

# file to save data
hdf5_file = 'carbon_ase_pyscf_qiskit.hdf5'

# init the hdf5 file
qc2data = qc2Data(hdf5_file, mol)

# specify the qchem calculator and run
qc2data.molecule.calc = PySCF(method='scf.UHF', basis='sto-3g',
                              multiplicity=3, charge=0)

# run calculation and save qchem data in the hdf5 file
qc2data.run()

# define activate space
n_active_electrons = (4, 2)  # => (n_alpha, n_beta)
n_active_spatial_orbitals = 5

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

ansatz = UCC(
    num_spatial_orbitals=n_active_spatial_orbitals,
    num_particles=n_active_electrons,
    qubit_mapper=mapper,
    initial_state=reference_state,
    excitations='sdtq'
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
