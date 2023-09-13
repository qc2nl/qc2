try:
    from scm.plams.interfaces.adfsuite.ase_calculator import AMSJobCalculator
except ImportError:
    raise ImportError

from typing import Optional, List, Union, Sequence, Tuple, Any
import os
from qiskit_nature.second_q.formats.qcschema import *
import h5py 
from ase import Atoms
from ase.calculators.calculator import FileIOCalculator, all_changes
from ase.io import write
import numpy as np
from scm.plams.interfaces.adfsuite.ase_calculator import all_changes 
from scm import plams 
from .qc2_ase_base_class import BaseQc2ASECalculator
from qiskit_nature import __version__ as qiskit_nature_version
class AMS(AMSJobCalculator, BaseQc2ASECalculator):

    def __init__(self, settings=None, **kwargs):
        settings = self._get_default_setings(settings)
        AMSJobCalculator.__init__(self, settings=settings, **kwargs)
        BaseQc2ASECalculator.__init__(self)

    @staticmethod
    def _get_default_setings(settings) -> plams.Settings :
        """Set up the necessary settings"""
        if settings is None:
            settings = plams.Settings()
        
        settings.input.ams.Task = 'SinglePoint'
        settings.input.ADF
        settings.input.ADF.FULLFOCK = True
        settings.input.ADF.ALLPOINTS = True
        settings.input.ADF.AOMat2File = True

        settings.input.adf.symmetry = "NOSYM"
        settings.input.adf.basis.core = "None"
        settings.input.adf.noprint = "Logfile"
        settings.input.adf.save = "TAPE15"

        return settings


    def save(self, hdf5_filename):
        """Dumps qchem data to a HDF5.

        Args:
            hdf5_filename (str): HDF5 file to save the data to.

        Notes:
            HDF5 files are written following the QCSchema.

        Returns:
            None

        Example:
        >>> from ase.build import molecule
        >>> from qc2.ase.pyscf import PySCF
        >>>
        >>> molecule = molecule('H2')
        >>> molecule.calc = PySCF()  # => RHF/STO-3G
        >>> molecule.calc.get_potential_energy()
        >>> molecule.ca
        """

        topology = QCTopology(
            symbols = self.atoms.get_chemical_symbols(),
            geometry = self.atoms.get_positions().flatten(),
            schema_name = 'qcschma_molecule',
            schema_version = qiskit_nature_version
        )

        model = QCModel(
            method = 'dft',
            basis = None
        )

        provenance = QCProvenance(
            creator = self.amsresults.readrkf('General', 'program', file='adf'),
            routine = self.amsresults.engine_names(), 
            version = self.amsresults.readrkf('General', 'release', file='adf'))
        
        properties = QCProperties(
            calcinfo_nbasis = self.amsresults.readrkf('Basis', 'naos', file='adf'),
            calcinfo_nmo = self.amsresults.readrkf('A', 'nmo_A', file='adf'),
        )

        wavefunction = QCWavefunction(
            orbitals_a = self._get_molecular_orbitals_coefficients(spin='A'),
            orbitals_b = self._get_molecular_orbitals_coefficients(spin='B'),
            eigenvalues_a = self._get_molecular_orbitals_energies(spin='A'),
            eigenvalues_b = self._get_molecular_orbitals_energies(spin='B'),
        )

        qcschema = QCSchema(
            schema_name = 'qcschema',
            schema_version = qiskit_nature_version,
            driver = 'energy',
            return_result = self.amsresults.readrkf('AMSResults', 
                                                  'energy', file='adf'),
            success = self.amsresults.ok(),       
            topology = topology,
            provenance = provenance,
            model = model,
            properties = properties,
            wavefunction = wavefunction
        )

        with h5py.File(hdf5_filename, 'w') as h5file:
            qcschema.to_hdf5(h5file)


    def get_integrals(self) -> Tuple[Any, ...]:
        """Calculates core energy, one- and two-body integrals in MO basis."""
        raise NotImplementedError

    def _get_molecular_orbitals_coefficients(self, spin='A') -> np.ndarray:
        """Read the molecular orbitals from the rkf file
        """

        if spin not in self.amsresults.get_rkf_skeleton(file='adf'):
            return None

        # number of mos and aos
        naos = int(self.amsresults.readrkf('Basis', 'naos', file='adf'))
        nmos = int(self.amsresults.readrkf(spin, 'nmo_{}'.format(spin), file='adf'))

        # molecular orbital coefficients
        mo_coeffs = np.array(self.amsresults.readrkf(spin, 
                                                     'Eigen-Bas_{}'.format(spin), 
                                                     file='adf')).reshape(nmos, naos)

        # energy of the mos
        energy = self.amsresults.readrkf(spin, 
                                         'eps_{}'.format(spin), 
                                         file='adf')

        # ordering of the MOS
        order = self.amsresults.readrkf(spin, 'npart', file='adf')
        order = [o-1 for o in order]

        return mo_coeffs[:, order]
    

    def _get_molecular_orbitals_energies(self, spin='A') -> np.ndarray:
        """Read the molecular orbitals from the rkf file
        """
        if spin not in self.amsresults.get_rkf_skeleton(file='adf'):
            return None

        # energy of the mos
        energy = self.amsresults.readrkf(spin, 
                                         'eps_{}'.format(spin), 
                                         file='adf')

        # ordering of the MOS
        order = self.amsresults.readrkf(spin, 'npart', file='adf')
        order = [o-1 for o in order]

        return energy[order]