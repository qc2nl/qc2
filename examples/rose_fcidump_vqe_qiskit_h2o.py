"""Example of a VQE calc using Qiskit-Nature and ROSE-ASE calculator.

Standard restricted calculation => H2O example.

Notes:
    Requires the installation of qc2, ase, rose_ase, qiskit and h5py.
"""

import subprocess

from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD


from qc2.ase import ROSE, ROSETargetMolecule, ROSEFragment
from qc2.data import qc2Data
from qc2.algorithms.qiskit import VQE
from qc2.algorithms.utils import ActiveSpace


def clean_up_ROSE_files():
    """Remove DIRAC calculation outputs."""
    command = (
        "rm *xyz *out* *dfcoef DFCOEF* *inp INPUT* *chk"
        "MOLECULE.XYZ MRCONEE* *dfpcmo DFPCMO* *fchk *in fort.100 timer.dat "
        "INFO_MOL *.pyscf IAO_Fock SAO *.npy *.clean OUTPUT_AVAS *.chk"
        "ILMO*dat OUTPUT_* *.chk *.XYZ *.psi4 *.fcidump"
    )
    subprocess.run(command, shell=True, capture_output=True)


# define ROSE target molecule and fragments
H2O = ROSETargetMolecule(
    name="water",
    atoms=[
        ("O", (0.0, 0.00000, 0.59372)),
        ("H", (0.0, 0.76544, -0.00836)),
        ("H", (0.0, -0.76544, -0.00836)),
    ],
    basis="sto-3g",
)

oxygen = ROSEFragment(
    name="oxygen", atoms=[("O", (0, 0, 0))], multiplicity=1, basis="sto-3g"
)

hydrogen = ROSEFragment(
    name="hydrogen", atoms=[("H", (0, 0, 0))], multiplicity=2, basis="sto-3g"
)

# define ROSE final ibos file to be read by qc2Data class
fcidump_file = "ibo.fcidump"

# instantiate qc2Data - no Atoms() needed
qc2data = qc2Data(fcidump_file, schema="fcidump")

# attach ROSE calculator to an empty Atoms()
qc2data.molecule.calc = ROSE(
    rose_calc_type="atom_frag",
    exponent=4,
    rose_target=H2O,
    rose_frags=[oxygen, hydrogen],
    # restricted=False,
    # openshell=False,
    rose_mo_calculator="pyscf",
)

# run ROSE calculator
qc2data.run()

# set up VQE calc
qc2data.algorithm = VQE(
    active_space=ActiveSpace(
        num_active_electrons=(2, 2),
        num_active_spatial_orbitals=3
    )
)

# run the calc
result = qc2data.algorithm.run()

clean_up_ROSE_files()
