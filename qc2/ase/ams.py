try:
    from scm.plams.interfaces.adfsuite.ase_calculator import AMSCalculator
except ImportError:
    raise ImportError

from typing import Optional, List, Union, Sequence
import os

from ase import Atoms
from ase.calculators.calculator import FileIOCalculator
from ase.io import write

class AMS(AMSCalculator):

    def __init__(self, *args, **kwargs):
        super().__init__(args, kwargs)
