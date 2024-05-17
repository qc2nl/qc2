from ase.build import molecule

from qiskit_algorithms.optimizers import SLSQP
from qiskit.primitives import Estimator

from qc2.data import qc2Data
from qc2.ase import PySCF

from qc2.algorithms.qiskit import oo_VQE
from qc2.algorithms.utils import ActiveSpace

# instantiate qc2Data class
qc2data = qc2Data(
    molecule=molecule('H2O'),
    filename='h2o.hdf5',
    schema='qcschema'
)

# set up and run calculator
qc2data.molecule.calc = PySCF()
qc2data.run()

# instantiate oo-VQE algorithm
qc2data.algorithm = oo_VQE(
    active_space=ActiveSpace(
        num_active_electrons=(2, 2),
        num_active_spatial_orbitals=3
    ),
    optimizer=SLSQP(),
    estimator=Estimator()
)

# run oo-VQE
results = qc2data.algorithm.run()
