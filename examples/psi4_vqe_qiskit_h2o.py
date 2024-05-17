"""Example of a VQE calc using Qiskit-Nature and Psi4-ASE calculator.

Standard restricted calculation => H2O example.

Notes:
    Requires the installation of qc2, ase, psi4, qiskit and h5py.
"""

import subprocess
from ase.build import molecule

from qiskit_algorithms.optimizers import SLSQP
from qiskit.primitives import Estimator

from qc2.ase import Psi4
from qc2.data import qc2Data
from qc2.algorithms.qiskit import VQE
from qc2.algorithms.utils import ActiveSpace


def clean_up_Psi4_files():
    """Remove Psi4 calculation outputs."""
    command = "rm *.dat"
    subprocess.run(command, shell=True, capture_output=True)


# set Atoms object
mol = molecule("H2O")

# file to save data
hdf5_file = "h2o_psi4_qiskit.fcidump"

# init the hdf5 file
qc2data = qc2Data(hdf5_file, mol, schema="fcidump")

# specify the qchem calculator
qc2data.molecule.calc = Psi4(method="hf", basis="sto-3g")

# run calculation and save qchem data in the hdf5 file
qc2data.run()

# set up VQE calc
qc2data.algorithm = VQE(
    active_space=ActiveSpace(
        num_active_electrons=(2, 2), num_active_spatial_orbitals=3
    ),
    optimizer=SLSQP(),
    estimator=Estimator(),
)

# run the calc
result = qc2data.algorithm.run()

clean_up_Psi4_files()

