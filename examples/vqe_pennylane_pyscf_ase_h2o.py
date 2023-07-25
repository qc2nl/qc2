"""Example of a VQE calc using Qiskit-Nature and PYSCF-ASE calculator.

Standard restricted calculation => H2O example.

Notes:
    Requires the installation of qc2, ase, qiskit and h5py.
"""
from ase.build import molecule

from qiskit_nature.second_q.mappers import JordanWignerMapper
import pennylane as qml
from pennylane import numpy as np

from qc2.ase import PySCF
from qc2.data import qc2Data


# set Atoms object
mol = molecule('H2O')

# file to save data
hdf5_file = 'h2o_ase_pyscf_pennylane.hdf5'

# init the hdf5 file
qc2data = qc2Data(hdf5_file, mol)

# specify the qchem calculator
qc2data.molecule.calc = PySCF()  # default => RH/STO-3G

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
                                                 mapper, format='pennylane')

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
        d_wires=d_wires, hf_state=hf_state)
    #if n % 2 == 0:
    #    print("step = {:},  E = {:.8f} Ha".format(n, energy))

print("=== PENNYLANE VQE RESULTS ===")
print(f"* Electronic ground state energy (Hartree): {energy}")
print(f"* Inactive core energy (Hartree): {e_core}")
print(f">>> Total ground state energy (Hartree): {energy+e_core}\n")

# print(f"+++ Final parameters:{params}")
