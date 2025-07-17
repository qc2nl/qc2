"""qc2 ASE package."""
# handling package imports
try:
    from .pyscf import PySCF
except ImportError:
    print('What do you want to do?')
    pass

try:
    from .psi4 import Psi4
except ImportError:
    pass

try:
    from .rose import ROSE, ROSETargetMolecule, ROSEFragment
except ImportError:
    pass

from .dirac import DIRAC

__all__ = [
    'PySCF',
    'ROSE',
    'ROSETargetMolecule',
    'ROSEFragment',
    'DIRAC',
    'Psi4'
]
