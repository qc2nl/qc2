"""
This module defines customized input options for Rose
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, List, Union, Sequence
import numpy as np
from ase import Atoms


@dataclass
class RoseFragment:
    """A dataclass representing an atomic or molecular fragment in Rose."""
    name: str
    atoms: Optional[
        List[Tuple[Union[str, np.ndarray], np.ndarray]]
    ] = field(default_factory=list)
    charge: int = 0
    multiplicity: int = 1
    basis: str = 'sto-3g'


@dataclass
class RoseMolecule:
    """A dataclass representing the target molecular system in Rose."""
    name: str
    atoms: Optional[
        List[Tuple[Union[str, np.ndarray], np.ndarray]]
    ] = field(default_factory=list)
    charge: int = 0
    multiplicity: int = 1
    basis: str = 'sto-3g'


class RoseCalcType(Enum):
    """Enumerator class defining the type of calculation done by Rose."""
    ATOM_FRAG = 'atom_frag'  # calc IAOs from atom-free AOs.
    MOL_FRAG = 'mol_frag'    # calc IFOs from free molecular fragment MOs.


class RoseInterfaceMOCalculators(Enum):
    """Enumerator class defining the program to which Rose is interfaced."""
    PYSCF = 'pyscf'
    PSI4 = 'psi4'
    DIRAC = 'dirac'
    ADF = 'adf'
    GAUSSIAN = 'gaussian'


class RoseIFOVersion(Enum):
    """Enumerator class defining the 'recipe' to obtain IFOs."""
    STNDRD_2013 = "Stndrd_2013"
    SIMPLE_2013 = "Simple_2013"
    SIMPLE_2014 = "Simple_2014"


class RoseILMOExponent(Enum):
    """Enumerator class defining the exponent used to obtain ILMOs."""
    TWO = 2
    THREE = 3
    FOUR = 4


@dataclass
class RoseInputDataClass:
    """A dataclass representing input options for Rose."""
    rose_target: Union[Atoms, RoseMolecule]
    """Target supermolecule => ASE Atoms object."""
    rose_frags: Union[
        Atoms, Sequence[Atoms], RoseFragment, Sequence[RoseFragment]
    ]
    """List of atomic or molecular fragments."""
    target_mo_file_exists: bool = False
    frags_mo_files_exist: bool = False
    """Are canonical mo files already present?"""
    rose_calc_type: RoseCalcType = RoseCalcType.ATOM_FRAG.value
    """Calculation type"""

    exponent: RoseILMOExponent = RoseILMOExponent.TWO.value
    """Exponent used in the localization procedure."""
    restricted: bool = True
    """Restricted calculation option"""
    spatial_orbitals: bool = True
    """Spatial orbitals option - true for non-relativistic calculations."""
    test: bool = False
    """Test to calculate HF energy with the final localized orbitals."""
    include_core: bool = False
    """Frozen core option."""
    relativistic: bool = False
    """Relativistic calculation option."""
    version: RoseIFOVersion = RoseIFOVersion.STNDRD_2013.value
    """Version of the IAO construction."""
    spherical: bool = False
    "Spherical or cartesian coordinate GTOs."
    uncontract: bool = True
    "Use decontracted basis sets."
    get_oeint: bool = False
    "Option to extract the one-electron integrals."
    save: bool = True
    """Option to save final iaos/ibos."""
    additional_virtuals_cutoff: Optional[float] = None  # 2.0  # Eh
    """Add virtual orbitals with energies below this treshold."""
    frag_threshold: Optional[float] = None  # 10.0  # Eh
    """Set reference virtuals as those with energies below this treshold."""
    frag_valence: Optional[List[List[int]]] = field(default_factory=list)
    """# of valence orbitals (valence occ + valence virt) per fragment."""
    frag_core: Optional[List[List[int]]] = field(default_factory=list)
    """# of core orbitals per fragment."""
    frag_bias: Optional[List[List[int]]] = field(default_factory=list)
    """Bias frags when assigning the loc orb (for non-polar bonding orb)."""

    avas_frag: Optional[List[int]] = field(default_factory=list)
    """Fragment IAO file to be extracted from ROSE."""
    nmo_avas: Optional[List[int]] = field(default_factory=list)
    """List of spatial MOs (or spinors if restricted = False) to AVAS."""

    def __post_init__(self):
        """This method is called after the instance is initialized."""
        if self.test:
            self.get_oeint = True
