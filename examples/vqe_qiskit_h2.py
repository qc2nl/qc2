"""Example of a VQE calc using Qiskit-Nature and DIRAC-ASE calculator.

Standard restricted calculation => H2 example.

Notes:
    Requires the installation of qc2, ase, qiskit and h5py.
"""

import subprocess
from ase.build import molecule

from qiskit_algorithms.optimizers import SLSQP
from qiskit.primitives import Estimator

from qc2.ase import DIRAC
from qc2.data import qc2Data
from qc2.algorithms.qiskit import VQE
from qc2.algorithms.utils import ActiveSpace


def clean_up_DIRAC_files():
    """Remove DIRAC calculation outputs."""
    command = "rm dirac* MDCINT* MRCONEE* FCIDUMP* AOMOMAT* FCI*"
    subprocess.run(command, shell=True, capture_output=True)


# set Atoms object
mol = molecule("H2")

# file to save data
hdf5_file = "h2_dirac_qiskit.hdf5"

# init the hdf5 file
qc2data = qc2Data(hdf5_file, mol)

# specify the qchem calculator
qc2data.molecule.calc = DIRAC()  # default => RHF/STO-3G

# run calculation and save qchem data in the hdf5 file
qc2data.run()

# set up VQE calc
qc2data.algorithm = VQE(
    active_space=ActiveSpace(
        num_active_electrons=(1, 1), num_active_spatial_orbitals=2
    ),
    mapper="bk",
    optimizer=SLSQP(),
    estimator=Estimator(),
)

# run vqe
qc2data.algorithm.run()

clean_up_DIRAC_files()
