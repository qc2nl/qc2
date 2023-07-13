"""Example of a VQE calc using Qiskit-Nature and DIRAC-ASE as calculator.

Standard restricted calculation H2 example.

Notes:
    Requires the installation of qc2, ase, qiskit and h5py.
"""
import subprocess
import h5py

from ase.build import molecule
from ase.units import Ha
from qc2.ase import DIRAC
from qc2.data import qc2Data

from qiskit_nature.second_q.formats.qcschema import QCSchema
from qiskit_nature.second_q.formats import qcschema_to_problem
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit.algorithms.optimizers import SLSQP
from qiskit.primitives import Estimator


def clean_up_DIRAC_files():
    """Remove DIRAC calculation outputs."""
    command = "rm dirac* MDCINT* MRCONEE* FCIDUMP* AOMOMAT* FCI*"
    subprocess.run(command, shell=True, capture_output=True)


# set Atoms object
mol = molecule('H2')

# file to save data
hdf5_file = 'h2_ase_dirac.hdf5'

# init the hdf5 file
qc2data = qc2Data(hdf5_file, mol)

# specify the qchem calculator and run
qc2data.molecule.calc = DIRAC()
print(qc2data.molecule.get_potential_energy()/Ha)

# replace/add required data in the hdf5 file
qc2data.molecule.calc.save(hdf5_file)

# Open the HDF5 file
file = h5py.File(hdf5_file, 'r')

# read data and store it in a QCSchema instance;
# see qiskit_nature/second_q/formats/qcschema/qc_schema.py
qcschema = QCSchema._from_hdf5_group(file)

# Close the file
file.close()

# convert QCSchema into an instance of ElectronicStructureProblem;
# see qiskit_nature/second_q/formats/qcschema_translator.py
es_problem = qcschema_to_problem(qcschema, include_dipole=False)

# convert ElectronicStructureProblem into an instance of ElectronicEnergy
# hamiltonian in second quantization;
# see qiskit_nature/second_q/hamiltonians/electronic_energy.py
hamiltonian = es_problem.hamiltonian
second_q_op = hamiltonian.second_q_op()

# print(second_q_op)

# define the type of fermionic-to-qubit transformation
mapper = JordanWignerMapper()

H2_reference_state = HartreeFock(
    num_spatial_orbitals=es_problem.num_spatial_orbitals,
    num_particles=es_problem.num_particles,
    qubit_mapper=mapper,
)

# print(H2_reference_state.draw())

ansatz = UCCSD(
    num_spatial_orbitals=es_problem.num_spatial_orbitals,
    num_particles=es_problem.num_particles,
    qubit_mapper=mapper,
    initial_state=H2_reference_state,
)

# print(ansatz.draw())

vqe_solver = VQE(Estimator(), ansatz, SLSQP())
vqe_solver.initial_point = [0.0] * ansatz.num_parameters

calc = GroundStateEigensolver(mapper, vqe_solver)

res = calc.solve(es_problem)
print(res)

clean_up_DIRAC_files()
