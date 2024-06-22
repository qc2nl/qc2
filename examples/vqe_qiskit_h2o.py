"""Example of a VQE calc using Qiskit-Nature and PYSCF-ASE calculator.

Standard restricted calculation => H2O example.

Notes:
    Requires the installation of qc2, ase, qiskit and h5py.
"""

from ase.build import molecule

from qc2.ase import PySCF
from qc2.data import qc2Data
from qc2.algorithms.qiskit import VQE
from qc2.algorithms.utils import ActiveSpace

# set Atoms object
mol = molecule("H2O")

# file to save data
hdf5_file = "h2o_pyscf_qiskit.hdf5"

# init the hdf5 file
qc2data = qc2Data(hdf5_file, mol)

# specify the qchem calculator
qc2data.molecule.calc = PySCF()  # default => RHF/STO-3G

# run calculation and save qchem data in the hdf5 file
qc2data.run()

# set up VQE calc
qc2data.algorithm = VQE(
    active_space=ActiveSpace(
        num_active_electrons=(2, 2), num_active_spatial_orbitals=3
    )
)

# run the qc calc
qc2data.algorithm.run()
