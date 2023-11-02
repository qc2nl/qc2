"""qc2 ASE package."""
import logging

# handling package imports
try:
    from .pyscf import PySCF
except ImportError:
    logging.warning(
        "PySCF is not available. "
        "ASE-PySCF calculator not functional."
    )
try:
    from .psi4 import Psi4
except ImportError:
    logging.warning(
        "Psi4 is not available. "
        "ASE-Psi4 calculator not functional."
    )

from .rose import Rose
from .dirac import DIRAC

__all__ = [
    'PySCF', 'Rose', 'DIRAC', 'Psi4'
]
