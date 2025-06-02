"""H5FCIDump"""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import h5py
import numpy as np

from qiskit_nature import QiskitNatureError
from qiskit_nature.second_q.operators.symmetric_two_body import SymmetricTwoBodyIntegrals

@dataclass
class H5FCIDump:
    """
    Qiskit Nature dataclass for representing the FCIDump format.

    The FCIDump format is partially defined in Knowles1989.

    References:
        Knowles1989: Peter J. Knowles, Nicholas C. Handy,
            A determinant based full configuration interaction program,
            Computer Physics Communications, Volume 54, Issue 1, 1989, Pages 75-83,
            ISSN 0010-4655, https://doi.org/10.1016/0010-4655(89)90033-7.
    """

    num_electrons: int
    """The number of electrons."""
    hij: np.ndarray
    """The alpha 1-electron integrals."""
    hijkl: SymmetricTwoBodyIntegrals
    """The alpha/alpha 2-electron integrals ordered in chemist' notation."""
    hij_b: np.ndarray | None = None
    """The beta 1-electron integrals."""
    hijkl_bb: SymmetricTwoBodyIntegrals | None = None
    """The beta/beta 2-electron integrals ordered in chemist' notation."""
    hijkl_ba: SymmetricTwoBodyIntegrals | None = None
    """The beta/alpha 2-electron integrals ordered in chemist' notation."""
    constant_energy: float | None = None
    """The constant energy comprising (for example) the nuclear repulsion energy and inactive
    energies."""
    multiplicity: int = 1
    """The multiplicity."""
    orbsym: Sequence[int] | None = None
    """A list of spatial symmetries of the orbitals."""
    isym: int = 1
    """The spatial symmetry of the wave function."""

    @property
    def num_orbitals(self) -> int:
        """The number of orbitals."""
        return self.hij.shape[0]

    @classmethod
    def from_file(cls, fcidump: str | Path) -> H5FCIDump:
        """Constructs an H5FCIDump object from a file.

        Args:
            fcidump: Path to the input file.

        Returns:
            A :class:`.H5FCIDump` instance.
        """
        try:
            with h5py.File(fcidump,'r') as f5:
                return H5FCIDump(
                    num_electrons=f5["NELEC"][()],	
                    hij=f5["HIJ"][()],
                    hijkl=f5["HIJKL"][()],
                    hij_b=f5["HIJ_B"][()],
                    hijkl_ba=f5["HIJKL_BA"][()],
                    hijkl_bb=f5["HIJKL_BB"][()],
                    multiplicity=f5["MS2"][()] + 1,
                    constant_energy=f5["ecore"][()],
                    orbsym=f5["ORBSYM"][()],
                    isym=f5["ISYM"][()],
                )
        except Exception as ex:
            raise QiskitNatureError(f"Failed to parse {fcidump}: {ex}")

    def to_file(self, fcidump: str | Path) -> None:
        """Dumps an FCIDump object to a file.

        Args:
            fcidump: Path to the output file.
        Raises:
            QiskitNatureError: not all beta-spin related matrices are either None or not None.
            QiskitNatureError: if the dimensions of the provided integral matrices do not match.
        """
        outpath = fcidump if isinstance(fcidump, Path) else Path(fcidump)
        # either all beta variables are None or all of them are not
        if not all(h is None for h in [self.hij_b, self.hijkl_ba, self.hijkl_bb]) and not all(
            h is not None for h in [self.hij_b, self.hijkl_ba, self.hijkl_bb]
        ):
            raise QiskitNatureError("Invalid beta variables.")
        if set(self.hij.shape) != set(self.hijkl.shape):
            raise QiskitNatureError(
                "The number of orbitals of the 1- and 2-body matrices do not match: "
                f"{set(self.hij.shape)} vs. {set(self.hijkl.shape)}"
            )

        with h5py.File(outpath,'w') as f5: 
            f5.create_dataset("NELEC", data=self.num_electrons)
            f5.create_dataset("MS2", data=self.multiplicity - 1)
            f5.create_dataset("ecore", data=self.constant_energy)
            f5.create_dataset("ORBSYM", data=self.orbsym)
            f5.create_dataset("ISYM", data=self.isym)
            f5.create_dataset("HIJ", data=self.hij)
            f5.create_dataset("HIJKL", data=self.hijkl)
            f5.create_dataset("HIJ_B", data=self.hij_b)
            f5.create_dataset("HIJKL_BA", data=self.hijkl_ba)
            f5.create_dataset("HIJKL_BB", data=self.hijkl_bb)