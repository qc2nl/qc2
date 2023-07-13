"""Example of a VQE calc using Qiskit-Nature and PySCF-ASE as calculator.

Test case for H2- molecule as an example of a doublet (unrestricted)
calculation.

Notes:
    Requires the installation of qc2, ase, qiskit and h5py.
"""
import h5py

from ase.build import molecule
from ase.units import Ha
from qc2.ase import PySCF
from qc2.data import qc2Data

from qiskit_nature.second_q.formats.qcschema import QCSchema
from qiskit_nature.second_q.formats import qcschema_to_problem
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit.algorithms.optimizers import SLSQP
from qiskit.primitives import Estimator


# set Atoms object
mol = molecule('H2')

# file to save data
hdf5_file = 'h2_minus_ase_pyscf.hdf5'

# init the hdf5 file
qc2data = qc2Data(hdf5_file, mol)

# specify the qchem calculator, run and save
qc2data.molecule.calc = PySCF(method='scf.UHF', basis='6-31g',
                              multiplicity=2, charge=-1)
print(qc2data.molecule.get_potential_energy()/Ha)
qc2data.molecule.calc.save(hdf5_file)

# Open the HDF5 file
file = h5py.File(hdf5_file, 'r')
qcschema = QCSchema._from_hdf5_group(file)
file.close()

es_problem = qcschema_to_problem(qcschema, include_dipole=False)
hamiltonian = es_problem.hamiltonian
second_q_op = hamiltonian.second_q_op()

mapper = JordanWignerMapper()

H2_reference_state = HartreeFock(
    num_spatial_orbitals=es_problem.num_spatial_orbitals,
    num_particles=es_problem.num_particles,
    qubit_mapper=mapper,
)

ansatz = UCCSD(
    num_spatial_orbitals=es_problem.num_spatial_orbitals,
    num_particles=es_problem.num_particles,
    qubit_mapper=mapper,
    initial_state=H2_reference_state,
)

vqe_solver = VQE(Estimator(), ansatz, SLSQP())
vqe_solver.initial_point = [0.0] * ansatz.num_parameters

calc = GroundStateEigensolver(mapper, vqe_solver)

res = calc.solve(es_problem)
print(res)
