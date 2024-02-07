"""Example of how to run stand-alone qc2-ASE calculations.

This run loads the data from a fcidump formatted file.
"""
from ase.build import molecule
from qc2.ase.qc2_ase_base_class import BaseQc2ASECalculator

# set target molecule
mol = molecule('H2O')

# attach a generic qchem calculator
mol.calc = BaseQc2ASECalculator()
# set the reading format
mol.calc.schema_format = "fcidump"

# load qchem data into a instance of `FCIDump` dataclass
fcidump = mol.calc.load('datafiles/qc2-fcidump_format_h2o.fcidump')
