"""This module defines a cutomized qc2 ASE-Psi4 calculator.

For the original calculator see:
https://databases.fysik.dtu.dk/ase/ase/calculators/psi4.html#module-ase.calculators.psi4
"""
try:
    from ase.calculators.psi4 import Psi4 as Psi4_original
    from psi4.driver import fcidump
    from psi4.core import MintsHelper
except ImportError as error:
    raise ImportError(
        "Failed to export original ROSE-Psi4 calculator!"
    ) from error

from typing import Union, Tuple
import numpy as np
import h5py

from qiskit_nature.second_q.formats.qcschema import QCSchema
from qiskit_nature.second_q.formats.fcidump import FCIDump

from .qc2_ase_base_class import BaseQc2ASECalculator


class Psi4(Psi4_original, BaseQc2ASECalculator):
    """An extended ASE calculator for Psi4.

    Args:
        Psi4_original (Psi4_original): Original ROSE Psi4 calculator.
        BaseQc2ASECalculator (BaseQc2ASECalculator): Base class for
            ase calculartors in qc2.
    """
    def __init__(self, *args, **kwargs) -> None:
        """ASE-Psi4 Class Constructor.

        Give an example here on how to use Psi4....
        """
        Psi4_original.__init__(self, *args, **kwargs)
        BaseQc2ASECalculator.__init__(self)

        self.scf_e = None
        self.scf_wfn = None
        self.mints = None

    def calculate(self, *args, **kwargs) -> None:
        """Calculate method for qc2 ASE-Psi4."""
        Psi4_original.calculate(self, *args, **kwargs)

        # save energy and wavefunction
        method = self.parameters['method']
        basis = self.parameters['basis']
        (self.scf_e, self.scf_wfn) = self.psi4.energy(
            f'{method}/{basis}', return_wfn=True
        )

        # instantiate `psi4.core.MintsHelper` class
        self.mints = MintsHelper(self.scf_wfn.basisset())

    def save(self, datafile: Union[h5py.File, str]) -> None:
        """Dumps qchem data to a datafile using QCSchema or FCIDump formats.

        Args:
            datafile (Union[h5py.File, str]): file to save the data to.

        Notes:
            files are written following the QCSchema or FCIDump formats.

        Returns:
            None

        Example:
        >>> from ase.build import molecule
        >>> from qc2.ase import Psi4
        >>>
        >>> molecule = molecule('H2')
        >>> molecule.calc = Psi4(method='hf', basis='sto-3g')
        >>> molecule.calc.schema_format = "qcschema"
        >>> molecule.get_potential_energy()
        >>> molecule.calc.save('h2.hdf5')
        >>>
        >>> molecule = molecule('H2')
        >>> molecule.calc = Psi4(method='hf', basis='sto-3g')
        >>> molecule.calc.schema_format = "fcidump"
        >>> molecule.get_potential_energy()
        >>> molecule.calc.save('h2.fcidump')
        """
        # in case of fcidump format
        if self._schema_format == "fcidump":
            fcidump(self.scf_wfn, datafile)
            return

    def load(self, datafile: Union[h5py.File, str]) -> Union[
            QCSchema, FCIDump
    ]:
        """Loads electronic structure data from a datafile.

        Notes:
            files are read following the qcschema or fcidump formats.

        Returns:
            `QCSchema` or `FCIDump` dataclasses containing qchem data.

        Example:
        >>> from ase.build import molecule
        >>> from qc2.ase import Psi4
        >>>
        >>> molecule = molecule('H2')
        >>> molecule.calc = Psi4(method='hf', basis='sto-3g')
        >>> molecule.calc.schema_format = "qcschema"
        >>> qcschema = molecule.calc.load('h2.h5')
        >>>
        >>> molecule = molecule('H2')
        >>> molecule.calc = Psi4(method='hf', basis='sto-3g')
        >>> molecule.calc.schema_format = "fcidump"
        >>> fcidump = molecule.calc.load('h2.fcidump')
        """
        return BaseQc2ASECalculator.load(self, datafile)

    def get_integrals_ao_basis(self) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieves 1- & 2-e integrals in AO basis from Psi4 routines."""
        one_e_int = np.asarray(self.mints.ao_kinetic()) + \
            np.asarray(self.mints.ao_potential())
        two_e_int = np.asarray(self.mints.ao_eri())
        return one_e_int, two_e_int

    def get_overlap_matrix(self) -> np.ndarray:
        """Retrieves overlap matrix from Psi4 routines."""
        return np.asarray(self.mints.ao_overlap())
