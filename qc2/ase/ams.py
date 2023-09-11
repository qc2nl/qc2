try:
    from scm.plams.interfaces.adfsuite.ase_calculator import AMSCalculator
except ImportError:
    raise ImportError

from typing import Optional, List, Union, Sequence
import os

from ase import Atoms
from ase.calculators.calculator import FileIOCalculator
from ase.io import write
import numpy as np 
from .qc2_ase_base_class import BaseQc2ASECalculator

class AMS(AMSCalculator, BaseQc2ASECalculator):

    def __init__(self, *args, **kwargs):
        super().__init__(args, kwargs)

    def save(self, filename):
        raise NotImplementedError
    
    def _get_molecular_orbitals(self) -> np.ndarray:
        """Read the molecular orbitals from the rkf file
        """

        # number of mos and aos
        naos = int(self.readrkf('Basis', 'naos', file='adf'))
        nmos = int(self.readrkf('A', 'nmos_A', file='adf'))

        # molecular orbital coefficients
        mo_coeffs = np.array(self.readrkf('A', 'Eigen-Bas_A', file='adf')).reshape(nmos, naos)

        # energy of the mos
        energy = self.readrkf('A', 'eps_A', file='adf')

        # ordering of the MOS
        order = self.readrkf('A', 'npart', file='adf')
        order = [o-1 for o in order]

        return energy[order], mo_coeffs[:, order]
    
    