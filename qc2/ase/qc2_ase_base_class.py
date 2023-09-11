"""
This module implements the abstract base class for qc2 ase calculators.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Any, Sequence, Optional
import os
import h5py


class BaseQc2ASECalculator(ABC):
    """
    Abstract base class for the qc2 ASE calculators.
    """
    def __init__(self) -> None:
        self._initialize_qcschema_attributes()

    def _initialize_qcschema_attributes(self) -> None:
        """
        Sets ASE special attributes based on QCSchema.
        """
        # => molecule group
        self.symbols: Optional[Sequence[str]] = None
        """The list of atom symbols in this topology."""
        self.geometry: Optional[Sequence[float]] = None
        """The XYZ coordinates (in Bohr units) of the atoms.
        This is a flat list of three times the length of the
        `symbols` list."""
        self.molecular_charge: Optional[int] = None
        """The overall charge of the molecule."""
        self.molecular_multiplicity: Optional[int] = None
        """The overall multiplicity of the molecule."""
        self.atomic_numbers: Optional[Sequence[int]] = None
        """The atomic numbers of all atoms, indicating their nuclear charge.
        Its length must match that of the `symbols` list."""

        # => properties group
        self.calcinfo_nbasis: Optional[int] = None
        """The number of basis functions in the computation."""
        self.calcinfo_nmo: Optional[int] = None
        """The number of molecular orbitals in the computation."""
        self.calcinfo_nalpha: Optional[int] = None
        """The number of alpha-spin electrons in the computation."""
        self.calcinfo_nbeta: Optional[int] = None
        """The number of beta-spin electrons in the computation."""
        self.calcinfo_natom: Optional[int] = None
        """The number of atoms in the computation."""
        self.nuclear_repulsion_energy: Optional[float] = None
        """The nuclear repulsion energy contribution to the
        total SCF energy."""
        self.return_energy: Optional[float] = None
        """The returned energy of the computation."""

        # => model group
        self.basis: Optional[str] = None
        """The basis set used during the computation."""
        self.method: Optional[str] = None
        """The method used for the computation of this object."""

        # => wavefunction group
        self.scf_fock_mo_a: Optional[Sequence[float]] = None
        """The SCF alpha-spin Fock matrix in the MO basis."""
        self.scf_fock_mo_b: Optional[Sequence[float]] = None
        """The SCF beta-spin Fock matrix in the MO basis."""
        self.scf_eri_mo_aa: Optional[Sequence[float]] = None
        """The SCF alpha-alpha electron-repulsion integrals
        in the MO basis."""
        self.scf_eri_mo_bb: Optional[Sequence[float]] = None
        """The SCF beta-beta electron-repulsion integrals
        in the MO basis."""
        self.scf_eri_mo_ab: Optional[Sequence[float]] = None
        """The SCF alpha-beta electron-repulsion integrals
        in the MO basis."""
        self.scf_eri_mo_ba: Optional[Sequence[float]] = None
        """The SCF beta-alpha electron-repulsion integrals
        in the MO basis."""
        self.scf_orbitals_a: Optional[Sequence[float]] = None
        """The SCF alpha-spin orbitals in the AO basis."""
        self.scf_orbitals_b: Optional[Sequence[float]] = None
        """The SCF beta-spin orbitals in the AO basis."""
        self.scf_eigenvalues_a: Optional[Sequence[float]] = None
        """The SCF alpha-spin orbital eigenvalues."""
        self.scf_eigenvalues_b: Optional[Sequence[float]] = None
        """The SCF beta-spin orbital eigenvalues."""
        self.localized_orbitals_a: Optional[Sequence[float]] = None
        """The localized alpha-spin orbitals.
        All `nmo` orbitals are included, even if only a subset
        were localized."""
        self.localized_orbitals_b: Optional[Sequence[float]] = None
        """The localized beta-spin orbitals.
        All `nmo` orbitals are included, even if only a subset
        were localized."""

    @abstractmethod
    def save(self, hdf5_filename: str) -> None:
        """Dumps qchem data to a HDF5 file following QCSchema."""

    @abstractmethod
    def calculate(self, *args, **kwargs) -> None:
        """Perform he scf calculation"""

    def load(self, hdf5_filename: str) -> None:
        """Loads qchem data from a QCSchema-formatted HDF5 file.

        Args:
            hdf5_filename (str): HDF5 file to read the data from.

        Returns:
            None
        """
        # first check if the file exists
        if not os.path.exists(hdf5_filename):
            raise FileNotFoundError(f"{hdf5_filename} file not found!")

        # open the HDF5 file in read mode
        with h5py.File(hdf5_filename, "r") as file:

            # => molecule group
            self.symbols = file['/molecule'].attrs['symbols']
            self.geometry = file['/molecule'].attrs['geometry']
            self.molecular_charge = file['/molecule'].attrs['molecular_charge']
            self.molecular_multiplicity = file['/molecule'].attrs[
                'molecular_multiplicity']
            self.atomic_numbers = file['/molecule'].attrs['atomic_numbers']

            # => properties group
            self.calcinfo_nbasis = file['/properties'].attrs['calcinfo_nbasis']
            self.calcinfo_nmo = file['/properties'].attrs['calcinfo_nmo']
            self.calcinfo_nalpha = file['/properties'].attrs['calcinfo_nalpha']
            self.calcinfo_nbeta = file['/properties'].attrs['calcinfo_nbeta']
            self.calcinfo_natom = file['/properties'].attrs['calcinfo_natom']
            self.nuclear_repulsion_energy = file['/properties'].attrs[
                'nuclear_repulsion_energy']
            self.return_energy = file['/properties'].attrs['return_energy']

            # => model group
            self.basis = file['/model'].attrs['basis']
            self.method = file['/model'].attrs['method']

            # => wavefunction group
            # one-body coefficients (MO basis)
            self.scf_fock_mo_a = file['/wavefunction/scf_fock_mo_a'][...]
            self.scf_fock_mo_b = file['/wavefunction/scf_fock_mo_b'][...]
            # two-body coefficients (MO basis)
            self.scf_eri_mo_aa = file['/wavefunction/scf_eri_mo_aa'][...]
            self.scf_eri_mo_bb = file['/wavefunction/scf_eri_mo_bb'][...]
            self.scf_eri_mo_ab = file['/wavefunction/scf_eri_mo_ab'][...]
            self.scf_eri_mo_ba = file['/wavefunction/scf_eri_mo_ba'][...]
            # mo coefficients in AO basis
            self.scf_orbitals_a = file['/wavefunction/scf_orbitals_a'][...]
            self.scf_orbitals_b = file['/wavefunction/scf_orbitals_b'][...]
            # scf orbital energies
            self.scf_eigenvalues_a = file[
                '/wavefunction/scf_eigenvalues_a'][...]
            self.scf_eigenvalues_b = file[
                '/wavefunction/scf_eigenvalues_b'][...]
            # localized orbitals
            self.localized_orbitals_a = file[
                '/wavefunction/localized_orbitals_a'][...]
            self.localized_orbitals_b = file[
                '/wavefunction/localized_orbitals_b'][...]

    @abstractmethod
    def get_integrals(self) -> Tuple[Any, ...]:
        """Calculates core energy, one- and two-body integrals in MO basis."""
