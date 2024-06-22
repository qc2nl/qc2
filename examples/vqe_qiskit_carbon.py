"""Example of a VQE calc using Qiskit-Nature and PYSCF-ASE as calculator.

Test case for C atom as an example of a triplet (unrestricted)
calculation.

Notes:
    Requires the installation of qc2, ase, qiskit and h5py.
"""

from ase import Atoms

from qc2.ase import PySCF
from qc2.data import qc2Data
from qc2.algorithms.qiskit import VQE
from qc2.algorithms.utils import ActiveSpace


# set Atoms object
mol = Atoms("C")

# file to save data
hdf5_file = "carbon_pyscf_qiskit.hdf5"

# init the hdf5 file
qc2data = qc2Data(hdf5_file, mol)

# specify the qchem calculator and run
qc2data.molecule.calc = PySCF(
    method="scf.UHF", basis="sto-3g", multiplicity=3, charge=0
)

# run calculation and save qchem data in the hdf5 file
qc2data.run()

# set up VQE calc
qc2data.algorithm = VQE(
    active_space=ActiveSpace(
        num_active_electrons=(4, 2), num_active_spatial_orbitals=5
    )
)

# run the calc
result = qc2data.algorithm.run()
