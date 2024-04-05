"""Example of how to run stand-alone qc2-ASE calculations.

This run saves the data into a fcidump formatted file
and uses `Atoms` class to define the molecule.
"""

import subprocess

from ase import Atoms
from ase.units import Ha
from qc2.ase import Psi4


def clean_up_Psi4_files():
    """Remove Psi4 calculation outputs."""
    command = "rm *.dat"
    subprocess.run(command, shell=True, capture_output=True)


# set target molecule via ASE `Atoms` class
mol = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.737166]])

# attach a qchem calculator to `Atoms` object
mol.calc = Psi4(method="hf", basis="sto-3g")
# define format in which to save the qchem data
mol.calc.schema_format = "fcidump"

# run qchem calculation and print energy in a.u.
energy = mol.get_potential_energy() / Ha
print(f"* Single-point energy (Hartree): {energy}")

# save qchem data to a file
mol.calc.save("h2.fcidump")

# clean up generated files
clean_up_Psi4_files()
