"""Example of a VQE calc using Pennynale and PYSCF-ASE as calculator.

Standard restricted calculation H2 example.

Notes:
    Requires the installation of qc2, ase, pennylane and h5py.
"""
from ase.build import molecule
from ase.units import Ha
from qc2.ase import PySCF
from qc2.data import qc2Data

from qiskit_nature.second_q.mappers import JordanWignerMapper
import pennylane as qml
from pennylane import numpy as np


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

es_problem, second_q_op = qc2data.get_fermionic_hamiltonian()

# define the type of fermionic-to-qubit transformation
mapper = JordanWignerMapper()

qubit_op = qc2data.get_qubit_hamiltonian(mapper=mapper, format='pennylane')

qubits = 4
electrons = 2

dev = qml.device("default.qubit", wires=qubits)
hf = qml.qchem.hf_state(electrons, qubits)


def circuit(param, wires=qubits):
    qml.BasisState(hf, wires=wires)
    qml.DoubleExcitation(param, wires=range(qubits))

@qml.qnode(dev, interface="autograd")
def cost_fn(param):
    circuit(param, wires=range(qubits))
    return qml.expval(qubit_op)


opt = qml.GradientDescentOptimizer(stepsize=0.4)
theta = np.array(0.0, requires_grad=True)

# store the values of the cost function
energy = [cost_fn(theta)]

# store the values of the circuit parameter
angle = [theta]

max_iterations = 100
conv_tol = 1e-06

for n in range(max_iterations):
    theta, prev_energy = opt.step_and_cost(cost_fn, theta)

    energy.append(cost_fn(theta))
    angle.append(theta)

    conv = np.abs(energy[-1] - prev_energy)

    if n % 2 == 0:
        print(f"Step = {n},  Energy = {energy[-1]:.8f} Ha")

    if conv <= conv_tol:
        break

print("\n" f"Total ground state energy = {(energy[-1]+es_problem.nuclear_repulsion_energy):.8f} Ha")
print("\n" f"Optimal value of the circuit parameter = {angle[-1]:.4f}")
