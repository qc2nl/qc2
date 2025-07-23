"""qc2 ASE package."""
# # handling package imports

from .pyscf import PySCF

try:
    from .psi4 import Psi4
except ImportError:
    pass

try:
    from .rose import ROSE, ROSETargetMolecule, ROSEFragment
except ImportError:
    pass

try:
    from .dirac import DIRAC
except ImportError:
    pass

__all__ = [
    'PySCF',
    'ROSE',
    'ROSETargetMolecule',
    'ROSEFragment',
    'DIRAC',
    'Psi4'
]
