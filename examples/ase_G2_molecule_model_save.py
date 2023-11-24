"""Example of how to run stand-alone qc2-ASE calculations.

This run saves the data into a qcschema formatted file
and uses the G2 test set to define the molecule.
"""
from ase.build import molecule
from ase.units import Ha
from qc2.ase import PySCF

# set target molecule using G2 molecule dataset
mol = molecule('H2O')

# attach a qchem calculator
mol.calc = PySCF(method='scf.HF', basis='sto-3g')
# define format in which to save the qchem data
mol.calc.schema_format = 'qcschema'

# perform qchem calculation
energy = mol.get_potential_energy()/Ha
print(f"* Single-point energy (Hartree): {energy}")

# save qchem data to a file
mol.calc.save('h2o.hdf5')
