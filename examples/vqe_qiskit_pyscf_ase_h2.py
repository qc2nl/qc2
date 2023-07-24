"""Example of a VQE calc using Qiskit-Nature and PYSCF-ASE as calculator.

Standard restricted calculation H2 example.

Notes:
    Requires the installation of qc2, ase, qiskit and h5py.
"""
from ase.build import molecule
from ase.units import Ha
from qc2.ase import PySCF
from qc2.data import qc2Data

import qiskit_nature
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit.algorithms.optimizers import SLSQP
from qiskit.primitives import Estimator

# Avoid using the deprecated `PauliSumOp` object
qiskit_nature.settings.use_pauli_sum_op = False

# set Atoms object
mol = molecule('H2')

# file to save data
hdf5_file = 'h2_ase_pyscf.hdf5'

# init the hdf5 file
qc2data = qc2Data(hdf5_file, mol)

# specify the qchem calculator
qc2data.molecule.calc = PySCF()

# run calculation and save calc data in the hdf5 file
qc2data.run()

n_active_electrons = (1, 1)
n_active_spatial_orbitals = 2

# define the type of fermionic-to-qubit transformation
mapper = JordanWignerMapper()

core, qubit_op = qc2data.get_qubit_hamiltonian(n_active_electrons,
                                               n_active_spatial_orbitals,
                                               mapper=mapper, format='qiskit')

H2_reference_state = HartreeFock(
    num_spatial_orbitals=n_active_spatial_orbitals,
    num_particles=n_active_electrons,
    qubit_mapper=mapper,
)

# print(H2_reference_state.draw())

ansatz = UCCSD(
    num_spatial_orbitals=n_active_spatial_orbitals,
    num_particles=n_active_electrons,
    qubit_mapper=mapper,
    initial_state=H2_reference_state,
)

# print(ansatz.draw())

vqe_solver = VQE(Estimator(), ansatz, SLSQP())
vqe_solver.initial_point = [0.0] * ansatz.num_parameters
result = vqe_solver.compute_minimum_eigenvalue(qubit_op)

print(result.eigenvalue+core)

#calc = GroundStateEigensolver(mapper, vqe_solver)

#res = calc.solve(es_problem)
#print(res)
