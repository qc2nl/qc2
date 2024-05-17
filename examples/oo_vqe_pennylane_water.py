from ase.build import molecule

import pennylane as qml

from qc2.data import qc2Data
from qc2.ase import PySCF

from qc2.algorithms.pennylane import oo_VQE
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
    optimizer=qml.GradientDescentOptimizer(stepsize=0.5),
    device="default.qubit"
)

# run oo-VQE
results = qc2data.algorithm.run(
    device_kwargs={"shots": None},
    qnode_kwargs={"diff_method": "best"}
)

