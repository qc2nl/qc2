"""This module defines a cutomized qc2 ASE-Psi4 calculator.

For the original ASE calculator see:
https://databases.fysik.dtu.dk/ase/ase/calculators/psi4.html#module-ase.calculators.psi4
"""
from typing import Union, Tuple
import numpy as np
import h5py

from ase.calculators.psi4 import Psi4 as Psi4_original
from psi4.driver import fcidump
from psi4.core import MintsHelper
from psi4 import __version__ as psi4_version

from qiskit_nature.second_q.formats.qcschema import QCSchema
from qiskit_nature import __version__ as qiskit_nature_version
from qiskit_nature.second_q.formats.fcidump import FCIDump

from .qc2_ase_base_class import BaseQc2ASECalculator


class Psi4(Psi4_original, BaseQc2ASECalculator):
    """An extended ASE calculator for Psi4.

    Args:
        Psi4_original (Psi4_original): Original ASE-Psi4 calculator.
        BaseQc2ASECalculator (BaseQc2ASECalculator): Base class for
            ase calculartors in qc2.
    """
    def __init__(self, *args, **kwargs) -> None:
        """Psi4-ASE calculator.

        **Example**

        >>> from ase import Atoms
        >>> from ase.build import molecule
        >>> from qc2.ase import Psi4
        >>>
        >>> molecule = Atoms(...) or molecule = molecule('...')
        >>> molecule.calc = Psi4(method='hf', basis='sto-3g', ...)
        >>> energy = molecule.get_potential_energy()
        >>> gradient = molecule.get_forces()
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

        # instantiate `psi4.core.MintsHelper` class for integrals calculation.
        self.mints = MintsHelper(self.scf_wfn.basisset())

    def save(self, datafile: Union[h5py.File, str]) -> None:
        """Dumps qchem data to a datafile using QCSchema or FCIDump formats.

        Args:
            datafile (Union[h5py.File, str]): file to save the data to.

        Notes:
            files are written following the QCSchema or FCIDump formats.

        Returns:
            None

        **Example**

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

        # in case of qcschema format
        # create instances of QCSchema's component dataclasses
        topology = super().instantiate_qctopology(
            symbols=[
                self.scf_wfn.molecule().fsymbol(n)
                for n in range(self.scf_wfn.molecule().natom())
            ],
            geometry=(
                self.scf_wfn.molecule().full_geometry().np[:].flatten()
            ),
            molecular_charge=self.scf_wfn.molecule().molecular_charge(),
            molecular_multiplicity=self.scf_wfn.molecule().multiplicity(),
            atomic_numbers=[
                self.scf_wfn.molecule().true_atomic_number(n)
                for n in range(self.scf_wfn.molecule().natom())
            ],
            schema_name="qcschema_molecule",
            schema_version=qiskit_nature_version
        )

        provenance = super().instantiate_qcprovenance(
            creator=self.name,
            version=psi4_version,
            routine=f"ASE-{self.__class__.__name__}.save()"
        )

        model = super().instantiate_qcmodel(
            basis=self.parameters['basis'],
            method=self.parameters['method']
        )

        properties = super().instantiate_qcproperties(
            calcinfo_nbasis=self.scf_wfn.basisset().nbf(),
            calcinfo_nmo=self.scf_wfn.basisset().nao(),
            calcinfo_nalpha=self.scf_wfn.nalpha(),
            calcinfo_nbeta=self.scf_wfn.nbeta(),
            calcinfo_natom=self.scf_wfn.molecule().natom(),
            nuclear_repulsion_energy=(
                self.scf_wfn.molecule().nuclear_repulsion_energy()
            ),
            return_energy=self.scf_e
        )

        # get 1- and 2-electron integrals in AO basis
        one_e_int_ao, two_e_int_ao = self.get_integrals_ao_basis()

        # get mo coefficients in AO basis
        alpha_coeff, beta_coeff = self.get_molecular_orbitals_coefficients()

        # get scf mo energies
        alpha_mo, beta_mo = self.get_molecular_orbitals_energies()

        # get 1- and 2-electron integrals in MO basis
        integrals_mo = self.get_integrals_mo_basis()
        one_body_coefficients_a = integrals_mo[0]
        one_body_coefficients_b = integrals_mo[1]
        two_body_coefficients_aa = integrals_mo[2]
        two_body_coefficients_bb = integrals_mo[3]
        two_body_coefficients_ab = integrals_mo[4]
        two_body_coefficients_ba = integrals_mo[5]

        wavefunction = super().instantiate_qcwavefunction(
            basis=self.parameters['basis'],
            scf_fock_a=one_e_int_ao.flatten(),
            # scf_fock_b=one_e_int_ao.flatten(),
            scf_eri=two_e_int_ao.flatten(),
            scf_fock_mo_a=one_body_coefficients_a.flatten(),
            scf_fock_mo_b=one_body_coefficients_b.flatten(),
            scf_eri_mo_aa=two_body_coefficients_aa.flatten(),
            scf_eri_mo_bb=two_body_coefficients_bb.flatten(),
            scf_eri_mo_ba=two_body_coefficients_ba.flatten(),
            scf_eri_mo_ab=two_body_coefficients_ab.flatten(),
            scf_orbitals_a=alpha_coeff.flatten(),
            scf_orbitals_b=beta_coeff.flatten(),
            scf_eigenvalues_a=alpha_mo.flatten(),
            scf_eigenvalues_b=beta_mo.flatten(),
            localized_orbitals_a='',
            localized_orbitals_b=''
        )

        qcschema = super().instantiate_qcschema(
            schema_name='qcschema_molecule',
            schema_version=qiskit_nature_version,
            driver='energy',
            keywords={},
            return_result=self.scf_e,
            molecule=topology,
            wavefunction=wavefunction,
            properties=properties,
            model=model,
            provenance=provenance,
            success=True
        )

        with h5py.File(datafile, 'w') as h5file:
            qcschema.to_hdf5(h5file)

    def load(self, datafile: Union[h5py.File, str]) -> Union[
            QCSchema, FCIDump
    ]:
        """Loads electronic structure data from a datafile.

        Notes:
            files are read following the qcschema or fcidump formats.

        Returns:
            `QCSchema` or `FCIDump` dataclasses containing qchem data.

        **Example**

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

    def get_integrals_mo_basis(self) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray, np.ndarray
    ]:
        """Retrieves 1- & 2-body integrals in MO basis from Psi4 routines.

        Returns:
            A tuple containing `np.ndarray` types:
                - one_body_int_a & one_body_int_b: Numpy arrays containing
                  alpha and beta components of the one-body integrals.
                - two_body_int_aa, two_body_int_bb, two_body_int_ab
                  & two_body_int_ba: Numpy arrays containing
                  alpha-alpha, beta-beta, alpha-beta & beta-alpha
                  components of the two-body integrals.
        """
        # define alpha and beta MO coeffients
        alpha_coeff, beta_coeff = self.get_molecular_orbitals_coefficients()

        # get 1- and 2-electron integrals in AO basis
        one_e_int_ao, two_e_int_ao = self.get_integrals_ao_basis()

        # calculate alpha and beta one-electron integrals in MO basis
        einsum_ao_to_mo = "jk,ji,kl->il"
        one_body_int_a = np.einsum(
            einsum_ao_to_mo, one_e_int_ao, alpha_coeff, alpha_coeff
        )
        one_body_int_b = np.einsum(
            einsum_ao_to_mo, one_e_int_ao, beta_coeff, beta_coeff
        )

        # calculate alpha-alpha, beta-beta, beta-alpha, alpha-beta
        # two-electron integrals in MO basis
        einsum_ao_to_mo = "pqrs,pi,qj,rk,sl->ijkl"
        two_body_int_aa = np.einsum(
            einsum_ao_to_mo, two_e_int_ao,
            alpha_coeff, alpha_coeff, alpha_coeff, alpha_coeff
        )
        two_body_int_bb = np.einsum(
            einsum_ao_to_mo, two_e_int_ao,
            beta_coeff, beta_coeff, beta_coeff, beta_coeff
        )
        two_body_int_ba = np.einsum(
            einsum_ao_to_mo, two_e_int_ao,
            beta_coeff, beta_coeff, alpha_coeff, alpha_coeff
        )
        two_body_int_ab = np.einsum(
            einsum_ao_to_mo, two_e_int_ao,
            alpha_coeff, alpha_coeff, beta_coeff, beta_coeff
        )

        return (
            one_body_int_a, one_body_int_b, two_body_int_aa,
            two_body_int_bb, two_body_int_ab, two_body_int_ba
        )

    def get_integrals_ao_basis(self) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieves 1- & 2-e integrals in AO basis from Psi4 routines."""
        one_e_int = np.asarray(self.mints.ao_kinetic()) + \
            np.asarray(self.mints.ao_potential())
        two_e_int = np.asarray(self.mints.ao_eri())
        return one_e_int, two_e_int

    def get_molecular_orbitals_coefficients(self) -> Tuple[
        np.ndarray, np.ndarray
    ]:
        """Retrieves alpha and beta MO coeffs from Psi4 routines."""
        return np.asarray(self.scf_wfn.Ca()), np.asarray(self.scf_wfn.Cb())

    def get_molecular_orbitals_energies(self) -> Tuple[
        np.ndarray, np.ndarray
    ]:
        """Retrieves alpha and beta MO energies from Psi4 routines."""
        return (
            np.asarray(self.scf_wfn.epsilon_a()),
            np.asarray(self.scf_wfn.epsilon_b())
        )

    def get_overlap_matrix(self) -> np.ndarray:
        """Retrieves overlap matrix from Psi4 routines."""
        return np.asarray(self.mints.ao_overlap())
