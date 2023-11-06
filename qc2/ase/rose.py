"""This module defines a cutomized qc2 ASE-ROSE calculator.

For the original calculator see:
https://gitlab.com/quantum_rose/rose/ase_rose/
https://gitlab.com/Cmurilochem/rose/-/blob/ROSE_ase_calculator/ase_rose/ase_rose/rose.py?ref_type=heads#L1
"""
from dataclasses import dataclass
from typing import Union
import logging
import h5py

from ase_rose import ROSE as ROSE_original
from ase_rose import ROSEFragment as ROSEFragment_original
from ase_rose import ROSETargetMolecule as ROSETargetMolecule_original

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
        """ROSE-ASE calculator.

        Example of a typical ASE-ROSE input:

        >>> from qc2.ase import ROSE
        >>> from qc2.ase import ROSETargetMolecule, ROSEFragment
        >>>
        >>> h2o = ROSETargetMolecule(
        >>> name='water',
        >>> atoms=[('O', (0.,  0.00000,  0.59372)),
        >>>        ('H', (0.,  0.76544, -0.00836)),
        >>>        ('H', (0., -0.76544, -0.00836))],
        >>> basis='sto-3g'
        >>> )
        >>>
        >>> oxygen = ROSEFragment(
        >>>     name='oxygen',
        >>>     atoms=[('O', (0, 0, 0))],
        >>>     multiplicity=1, basis='sto-3g'
        >>> )
        >>>
        >>> hydrogen = ROSEFragment(
        >>>     name='hydrogen',
        >>>     atoms=[('H', (0, 0, 0))],
        >>>     multiplicity=2, basis='sto-3g'
        >>> )
        >>>
        >>> h2o_calculator = ROSE(
        >>>    rose_calc_type='atom_frag',
        >>>    exponent=4,
        >>>    rose_target=h2o,
        >>>    rose_frags=[oxygen, hydrogen],
        >>>    test=True,
        >>>    save_data=True,
        >>>    restricted=True,
        >>>    openshell=True,
        >>>    rose_mo_calculator='pyscf'
        >>> )
        >>>
        >>> h2o_calculator.calculate()
        """
        ROSE_original.__init__(self, *args, **kwargs)
        BaseQc2ASECalculator.__init__(self)

    def save(self, datafile: Union[h5py.File, str]) -> None:
        """Dumps qchem data to a datafile."""
        logging.warning(
                'ROSE.save() method currently inactive. '
                'Datafile %s from ROSE can only be read.', datafile
            )

    def load(self, datafile: str) -> Union[
            QCSchema, FCIDump
    ]:
        """Loads electronic structure data from a fcidump datafile.

        Returns:
            `FCIDump` dataclass containing qchem data.

        Example:
        >>> from qc2.ase import ROSE
        >>> from qc2.ase import ROSETargetMolecule, ROSEFragment
        >>>
        >>> H2 = ROSETargetMolecule(atoms=[('H', (0,0,0.)), ('H', (0,0,1))])
        >>> H = ROSEFragment(atoms=[('H', (0, 0, 0))])
        >>> H2_calculator = ROSE(rose_target=H2, rose_frags=[H])
        >>> H2_calculator.calculate()
        >>> H2_calculator.schema_format = "fcidump"
        >>> fcidump = H2_calculator.load('ibo.fcidump')
        """
        return BaseQc2ASECalculator.load(self, datafile)
