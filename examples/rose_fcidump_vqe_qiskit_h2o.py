"""Example of a VQE calc using Qiskit-Nature and ROSE-ASE calculator.

Standard restricted calculation => H2O example.

Notes:
    Requires the installation of qc2, ase, rose_ase, qiskit and h5py.
"""
import subprocess

import qiskit_nature
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit.algorithms.optimizers import SLSQP
from qiskit.primitives import Estimator

from qc2.ase import ROSE, ROSETargetMolecule, ROSEFragment
from qc2.data import qc2Data

# Avoid using the deprecated `PauliSumOp` object
qiskit_nature.settings.use_pauli_sum_op = False


def clean_up_ROSE_files():
    """Remove DIRAC calculation outputs."""
    command = 'rm *xyz *out* *dfcoef DFCOEF* *inp INPUT* *chk' \
        'MOLECULE.XYZ MRCONEE* *dfpcmo DFPCMO* *fchk *in fort.100 timer.dat ' \
        'INFO_MOL *.pyscf IAO_Fock SAO *.npy *.clean OUTPUT_AVAS *.chk' \
        'ILMO*dat OUTPUT_* *.chk *.XYZ *.psi4 *.fcidump'
    subprocess.run(command, shell=True, capture_output=True)


# define ROSE target molecule and fragments
H2O = ROSETargetMolecule(
    name='water',
    atoms=[('O', (0.,  0.00000,  0.59372)),
           ('H', (0.,  0.76544, -0.00836)),
           ('H', (0., -0.76544, -0.00836))],
    basis='sto-3g'
)

oxigen = ROSEFragment(
    name='oxygen',
    atoms=[('O', (0, 0, 0))],
    multiplicity=1, basis='sto-3g'
)

hydrogen = ROSEFragment(
    name='hydrogen',
    atoms=[('H', (0, 0, 0))],
    multiplicity=2, basis='sto-3g'
)

# define ROSE final ibos file to be read by qc2Data class
fcidump_file = 'ibo.fcidump'

# instantiate qc2Data - no Atoms() needed
qc2data = qc2Data(fcidump_file, schema='fcidump')

# attach ROSE calculator to an empty Atoms()
qc2data.molecule.calc = ROSE(
    rose_calc_type='atom_frag',
    exponent=4,
    rose_target=H2O,
    rose_frags=[oxigen, hydrogen],
    rose_mo_calculator='pyscf'
)

# run ROSE calculator
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

clean_up_ROSE_files()
