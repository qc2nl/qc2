"""This module defines a cutomized qc2 ASE-ROSE calculator.

For the original calculator see:
https://gitlab.com/quantum_rose/rose/ase_rose/
https://gitlab.com/Cmurilochem/rose/-/blob/ROSE_ase_calculator/ase_rose/ase_rose/rose.py?ref_type=heads#L1
"""
try:
    from ase_rose import ROSE as ROSE_original
    from ase_rose import ROSEFragment as ROSEFragment_original
    from ase_rose import ROSETargetMolecule as ROSETargetMolecule_original
except ImportError as error:
    raise ImportError(
        "Failed to export original ROSE-ASE calculator!"
    ) from error

from dataclasses import dataclass
from typing import Union
import h5py

from qiskit_nature.second_q.formats.qcschema import QCSchema
from qiskit_nature.second_q.formats.fcidump import FCIDump

from .qc2_ase_base_class import BaseQc2ASECalculator


@dataclass
class ROSEFragment (ROSEFragment_original):
    """A dataclass representing an atomic or molecular fragment in ROSE."""


@dataclass
class ROSETargetMolecule (ROSETargetMolecule_original):
    """A dataclass representing the target molecular system in ROSE."""


class ROSE(ROSE_original, BaseQc2ASECalculator):
    """An extended ASE calculator for ROSE.

    Args:
        ROSE (ROSE): Original ROSE ASE calculator.
        BaseQc2ASECalculator (BaseQc2ASECalculator): Base class for
            ase calculartors in qc2.
    """
    def __init__(self, *args, **kwargs) -> None:
        """ASE-Rose Class Constructor.

        Give an example here on how to use ROSE....
        """
        ROSE_original.__init__(self, *args, **kwargs)
        BaseQc2ASECalculator.__init__(self)

    def load(self, datafile: Union[h5py.File, str]) -> Union[
            QCSchema, FCIDump
    ]:
        """Loads electronic structure data from a datafile.

        Notes:
            files are read following the qcschema or fcidump formats.

        Returns:
            `QCSchema` or `FCIDump` dataclasses containing qchem data.

        Example:
        >>> from ase.build import molecule
        >>> from qc2.ase.rose import ROSE
        >>>
        >>> molecule = molecule('H2')
        >>> molecule.calc = PySCF()     # => RHF/STO-3G
        >>> molecule.calc.schema_format = "fcidump"
        >>> fcidump = molecule.calc.load('h2.fcidump')
        """
        return BaseQc2ASECalculator.load(self, datafile)
