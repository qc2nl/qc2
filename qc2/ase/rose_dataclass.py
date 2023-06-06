"""Customized input options for Rose"""
from ase import Atoms
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Union, Sequence


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
    ### defining calculator specific options

    # target supermolecule => ASE Atoms object
    rose_target: Atoms
    # list of atomic or molecular fragments
    rose_frags: Union[Atoms, Sequence[Atoms]]
    # are canonical mo files already present?
    target_mo_file_exists: bool = False 
    frags_mo_files_exist: bool = False
    # calculation type
    rose_calc_type: RoseCalcType = RoseCalcType.ATOM_FRAG.value

    ### Rose intrinsic options

    # exponent used in the localization procedure 
    exponent: RoseILMOExponent = RoseILMOExponent.TWO.value
    restricted: bool = True
    spatial_orbitals: bool = True
    test: bool = False
    include_core: bool = False
    relativistic: bool = False
    # version of the IAO construction
    version: RoseIFOVersion = RoseIFOVersion.STNDRD_2013.value
    # spherical or cartesian coordinate GTOs
    spherical: bool = False
    # use decontracted basis sets
    uncontract: bool = True
    # extract the one-electron integrals
    get_oeint: bool = False
    # save final iaos/ibos ?
    save: bool = True
    # add virtual orbitals with energies below this treshold
    additional_virtuals_cutoff: Optional[float] = None  # 2.0  # Eh
    # define reference virtuals as those with energies below this treshold
    frag_threshold: Optional[float] = None  # 10.0  # Eh
    # number of valence orbitals (valence occupied + valence virtuals) per fragment
    frag_valence: Optional[List[List[int]]] = field(default_factory=list)
    # number of core orbitals per fragment
    frag_core: Optional[List[List[int]]] = field(default_factory=list)
    # bias fragments when assigning the loc orb (for non-polar bonding orb)
    frag_bias: Optional[List[List[int]]] = field(default_factory=list)

    # avas related options => made inactive in this version
    ## fragment IAO file to be extracted from ROSE (used for AVAS for instance)
    # avas_frag: Optional[List[int]] = field(default_factory=list)
    ## list of spatial MOs (or spinors if restricted = False) to consider in AVAS.
    # nmo_avas: Optional[List[int]] = field(default_factory=list)

    def __post_init__(self):
        """This method is called after the instance is initialized."""
        if self.test:
            self.get_oeint = True

