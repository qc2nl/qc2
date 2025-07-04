from ase.build import molecule
from ase import Atoms
# import pennylane as qml
from qc2.data import qc2Data
from qc2.ase import PySCF
from qc2.algorithms.netket.vmc import VMC
from qc2.algorithms.utils import ActiveSpace
import numpy as np

# instantiate qc2Data class
qc2data = qc2Data(
    molecule = molecule('H2O'),
    filename='h2o.hdf5',
    schema='qcschema'
)

# set up and run calculator
qc2data.molecule.calc = PySCF(basis='sto-3g')
qc2data.run()

# instantiate the solver
vmc = VMC()
qc2data.algorithm = vmc 
qc2data.algorithm.run()