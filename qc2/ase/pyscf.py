"""This module defines an ASE interface to PySCF.

https://pyscf.org/ => Official documentation
https://github.com/pyscf/pyscf => GitHub page

Note: Adapted from
https://github.com/pyscf/pyscf/blob/master/pyscf/pbc/tools/pyscf_ase.py
&
https://github.com/pyscf/pyscf/issues/624
"""
from typing import Union, Optional, List, Dict, Any, Tuple
import warnings
import numpy as np
import h5py

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.units import Ha, Bohr
from ase.calculators.calculator import InputError
from ase.calculators.calculator import CalculatorSetupError

from pyscf import gto, scf, dft
from pyscf import __version__ as pyscf_version
from pyscf.tools import fcidump

from qiskit_nature.second_q.formats.qcschema import QCSchema
from qiskit_nature import __version__ as qiskit_nature_version
from qiskit_nature.second_q.formats.fcidump import FCIDump

from .qc2_ase_base_class import BaseQc2ASECalculator


def ase_atoms_to_pyscf(ase_atoms: Atoms) -> List[List[Union[str, np.ndarray]]]:
    """Converts ASE atoms to PySCF atom.

    Args:
        ase_atoms (Atoms): ASE Atoms object.

    Returns:
        List[List[Union[str, np.ndarray]]]: PySCF atom.
    """
    return [[ase_atoms.get_chemical_symbols()[i], ase_atoms.get_positions()[i]]
            for i in range(len(ase_atoms.get_positions()))]


class PySCF(Calculator, BaseQc2ASECalculator):
    """An ASE calculator for the PySCF quantum chemistry package.

    Args:
        Calculator (Calculator): Base-class for all ASE calculators.
        BaseQc2ASECalculator (BaseQc2ASECalculator): Base class for
            ase calculartors in qc2.

    Raises:
        InputError: If attributes other than
            'method', 'xc', 'basis', 'multiplicity',
            'charge', 'relativistic', 'cart', 'scf_addons',
            'verbose', and 'output' are input as Calculator.
        CalculatorSetupError: If abinitio methods other than
            'scf.RHF', 'scf.UHF', 'scf.ROHF',
            'dft.RKS', 'dft.UKS', and 'dft.ROKS' are selected.

    Example of a typical ASE-PySCF input:

    >>> from ase import Atoms
    >>> from ase.build import molecule
    >>> from qc2.ase.pyscf import PySCF
    >>>
    >>> molecule = Atoms(...) or molecule = molecule('...')
    >>> molecule.calc = PySCF(method='dft.RKS',
                            xc='b3lyp',
                            basis='6-31g*',
                            charge=0,
                            multiplicity=1,
                            verbose=0,
                            cart=False,
                            relativistic=False,
                            scf_addons='frac_occ',
                            output='output_file_name_with_no_extension')
    >>> energy = molecule.get_potential_energy()
    >>> gradient = molecule.get_forces()

    where (currently)

    method = 'scf.RHF'
           = 'scf.UHF'
           = 'scf.ROHF'
           = 'dft.RKS'
           = 'dft.UKS'
           = 'dft.ROKS'

    Notes:
        - Scalar relativistic corrections can be added with
            'relativistic = True' keyword. If selected,
            the scf object will be decorated by x2c() method, e.g.,
            mf = scf.RHF(mol).x2c().
            'relativistic' is False by default.
        - pyscf.scf.addons functions can also be included, e.g.:
            if scf_addons='frac_occ' keyword is added, then
            mf = scf.addons.frac_occ(mf).
            'scf_addons' is None by default.
    """
    implemented_properties: List[str] = ['energy', 'forces']

    default_parameters: Dict[str, Any] = {'method': 'scf.HF',
                                          'basis': 'sto-3g',
                                          'xc': 'b3lyp',
                                          'multiplicity': 1,
                                          'charge': 0,
                                          'relativistic': False,
                                          'cart': False,
                                          'scf_addons': None,
                                          'output': None,
                                          'verbose': 0}

    def __init__(self,
                 restart: Optional[bool] = None,
                 ignore_bad_restart: Optional[bool] = False,
                 label: Optional[str] = 'PySCF',
                 atoms: Optional[Atoms] = None,
                 command: Optional[str] = None,
                 directory: str = '.',
                 **kwargs) -> None:
        """ASE-PySCF Class Constructor to initialize the object.

        Notes:
            Basic implementation based on the class Psi4(Calculator);
            see, e.g., ase/ase/calculators/psi4.py.

        Args:
            restart (bool, optional): Prefix for restart file.
                May contain a directory. Defaults to None: don't restart.
            ignore_bad_restart (bool, optional): Deprecated and will
                stop working in the future. Defaults to False.
            label (str, optional): Calculator name. Defaults to 'PySCF'.
            atoms (Atoms, optional): Atoms object to which the calculator
                will be attached. When restarting, atoms will get its
                positions and unit-cell updated from file. Defaults to None.
            command (str, optional): Command used to start calculation.
                Defaults to None.
            directory (str, optional): Working directory in which
                to perform calculations. Defaults to '.'.
        """
        # initializing base class Calculator.
        # see ase/ase/calculators/calculator.py.
        Calculator.__init__(self, restart=restart,
                            ignore_bad_restart=ignore_bad_restart, label=label,
                            atoms=atoms, command=command, directory=directory,
                            **kwargs)
        """Transforms **kwargs into a dictionary with calculation parameters.

        Starting with (attr1=value1, attr2=value2, ...)
            it creates self.parameters['attr1']=value1, and so on.
        """

        # Check self.parameters input keys and values
        self.check_pyscf_attributes()

        self.mol = None
        self.mf = None

        # initialize qc2 base class for ASE calculators.
        # see .qc2_ase_base_class.py
        BaseQc2ASECalculator.__init__(self)

    def check_pyscf_attributes(self) -> None:
        """Checks for any missing and/or mispelling PySCF input attribute.

        Notes:
            it can also be used to eventually set specific
            environment variables, ios, etc.
        """
        recognized_attributes: List[str] = [
            'ignore_bad_restart', 'command', 'method',
            'xc', 'basis', 'multiplicity', 'charge',
            'relativistic', 'cart', 'scf_addons',
            'output', 'verbose', 'kpts', 'nbands', 'smearing'
            ]

        # self.parameters gathers all PYSCF input options in a dictionary.
        # it is defined by class Calculator(BaseCalculator)
        # see ase/ase/calculators/calculator.py
        for key, value in self.parameters.items():
            # check attribute name
            if key not in recognized_attributes:
                raise InputError('Attribute', key,
                                 ' not recognized. Please check input.')

        # dealing with lack of multiplicity and charge info
        if 'multiplicity' not in self.parameters.keys():
            self.parameters['multiplicity'] = 1
            warnings.warn('Multiplicity not provided.'
                          'Assuming default singlet.')

        if 'charge' not in self.parameters.keys():
            self.parameters['charge'] = 0
            warnings.warn('Charge not provided. Assuming default zero.')

        # verbose sets the amount of pyscf printing
        # verbose = 0 prints no info
        if 'verbose' not in self.parameters.keys():
            self.parameters['verbose'] = 0

        # scalar relativistic corrections
        if 'relativistic' not in self.parameters.keys():
            self.parameters['relativistic'] = False

        # cartesian vs spherical basis functions
        if 'cart' not in self.parameters.keys():
            self.parameters['cart'] = False

        # cartesian vs spherical basis functions
        if 'scf_addons' not in self.parameters.keys():
            self.parameters['scf_addons'] = None

        # outputfile
        if 'output' not in self.parameters.keys():
            self.parameters['output'] = None

        # dealing with some ASE specific inputs
        if 'kpts' in self.parameters.keys():
            raise InputError('This ASE-PySCF interface does not yet implement'
                             ' periodic calculations, and thus does not'
                             ' accept k-points as parameters. '
                             'Please remove this argument.')

        if 'nbands' in self.parameters.keys():
            raise InputError('This ASE-PySCF interface '
                             'does not support the keyword "nbands".')

        if 'smearing' in self.parameters.keys():
            raise InputError('Finite temperature DFT is not currently'
                             ' implemented in this PySCF-ASE interface,'
                             ' thus a smearing argument cannot be utilized.'
                             ' Please remove this argument.')

    def calculate(self, atoms: Optional[Atoms] = None,
                  properties: List[str] = ['energy'],
                  system_changes: List[str] = all_changes) -> None:
        """This method is the core responsible for the actual calculation.

        Note: Implementation based on Calculator.calculate() method.
            see also ase/ase/calculators/calculator.py.

        Args:
            atoms (Atoms, optional): Atoms object to which
                the calculator is attached. Defaults to None.
            properties (list[str], optional): List of what properties
                need to be calculated. Defaults to ['energy'].
            system_changes (list[str], optional): List of what has changed
                since last calculation. Can be any combination
                of these six: 'positions', 'numbers', 'cell', 'pbc',
                'initial_charges' and 'initial_magmoms'.
                Defaults to all_changes.

        Raises:
            CalculatorSetupError: If a proper geometry is not provided.
            CalculatorSetupError: If abinitio methods other than
            'scf.RHF', 'scf.UHF', 'scf.ROHF', 'dft.RKS', 'dft.UKS',
                and 'dft.ROKS' are selected.
        """
        # setting up self.atoms attribute from base class Calculator.calculate.
        # this is extracted from the atoms Atoms object.
        Calculator.calculate(self, atoms=atoms)

        # checking that self.atoms has been properly initiated/updated
        if self.atoms is None:
            raise CalculatorSetupError('An Atoms object must be provided to '
                                       'perform a calculation.')

        # the spin variable corresponds to 2S instead of 2S+1
        spin_2s = self.parameters['multiplicity'] - 1

        # if requested, set pyscf output file
        outputfile = None
        if self.parameters['output'] is not None:
            outputfile = self.parameters['output'] + ".out"
            self.parameters['verbose'] = 5

        # passing geometry and other definitions
        self.mol = gto.M(atom=ase_atoms_to_pyscf(self.atoms),
                         basis=self.parameters['basis'],
                         charge=self.parameters['charge'],
                         spin=spin_2s,
                         verbose=self.parameters['verbose'],
                         cart=self.parameters['cart'],
                         output=outputfile)

        # make dictionary of implemented methods
        implemented_methods = {'scf.HF': scf.HF,
                               'scf.RHF': scf.RHF,
                               'scf.UHF': scf.UHF,
                               'scf.ROHF': scf.ROHF,
                               'dft.KS': dft.KS,
                               'dft.RKS': dft.RKS,
                               'dft.UKS': dft.UKS,
                               'dft.ROKS': dft.ROKS}

        # define mf object
        if self.parameters['method'] in implemented_methods:
            self.mf = implemented_methods[self.parameters['method']](self.mol)
        else:
            raise CalculatorSetupError('Method not yet implemented. '
                                       'Current PySCF-ASE calculator '
                                       'only allows for',
                                       implemented_methods,
                                       'wave functions.'
                                       ' Please check input method.')

        if 'dft' in self.parameters['method']:
            self.mf.xc = self.parameters['xc']

        # add scalar relativistic corrections
        if self.parameters['relativistic']:
            self.mf = self.mf.x2c()

        # add decorations into scf object using scf.addons module
        if self.parameters['scf_addons']:
            # get the name of the function to call
            func_name = self.parameters['scf_addons']
            # get the function object from the scf.addons module
            func = getattr(scf.addons, func_name)
            self.mf = func(self.mf)

        # self.mf.chkfile =  self.parameters['output'] + ".chk"

        # calculating energy in eV
        energy = self.mf.kernel() * Ha
        self.results['energy'] = energy

        # calculating forces
        if 'forces' in properties:
            gf = self.mf.nuc_grad_method()
            gf.verbose = self.parameters['verbose']
            if 'dft' in self.parameters['method']:
                gf.grid_response = True
            # analytic gradienta in eV/Angstrom
            forces = -1.0 * gf.kernel() * (Ha / Bohr)
            totalforces = []
            totalforces.extend(forces)
            totalforces = np.array(totalforces)
            self.results['forces'] = totalforces

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
        >>> from qc2.ase.pyscf import PySCF
        >>>
        >>> molecule = molecule('H2')
        >>> molecule.calc = PySCF()  # => RHF/STO-3G
        >>> molecule.calc.schema_format = "qcschema"
        >>> molecule.calc.get_potential_energy()
        >>> molecule.calc.save('h2.hdf5')
        >>>
        >>> molecule = molecule('H2')
        >>> molecule.calc = PySCF()  # => RHF/STO-3G
        >>> molecule.calc.schema_format = "fcidump"
        >>> molecule.calc.get_potential_energy()
        >>> molecule.calc.save('h2.fcidump')
        """
        # in case of fcidump format
        if self._schema_format == "fcidump":
            fcidump.from_scf(self.mf, datafile)
            return

        # in case of qcschema format
        # create instances of QCSchema's component dataclasses
        topology = super().instantiate_qctopology(
            symbols=[
                self.mol.atom_pure_symbol(i) for i in range(self.mol.natm)
            ],
            geometry=self.mol.atom_coords(unit="Bohr").ravel().tolist(),
            molecular_charge=self.mol.charge,
            molecular_multiplicity=(self.mol.spin + 1),
            atomic_numbers=[atom[0] for atom in self.mol._atm],
            schema_name="qcschema_molecule",
            schema_version=qiskit_nature_version
        )

        provenance = super().instantiate_qcprovenance(
            creator=self.name,
            version=pyscf_version,
            routine=f"ASE-{self.__class__.__name__}.save()"
        )

        model = super().instantiate_qcmodel(
            basis=self.mol.basis,
            method=self.mf.__class__.__name__
        )

        properties = super().instantiate_qcproperties(
            calcinfo_nbasis=self.mol.nbas,
            calcinfo_nmo=self.mol.nao,
            calcinfo_nalpha=self.mol.nelec[0],
            calcinfo_nbeta=self.mol.nelec[1],
            calcinfo_natom=self.mol.natm,
            nuclear_repulsion_energy=self.mf.energy_nuc(),
            return_energy=self.mf.e_tot
        )

        # get 1- and 2-electron integrals in AO basis
        one_e_int_ao, two_e_int_ao = self.get_integrals_ao_basis()

        # get mo coefficients in AO basis
        alpha_coeff, beta_coeff = self.get_molecular_orbitals_coefficients()
        if beta_coeff is None:
            beta_coeff = alpha_coeff

        # get scf mo energies
        alpha_mo, beta_mo = self.get_molecular_orbitals_energies()
        if beta_mo is None:
            beta_mo = alpha_mo

        # get 1- and 2-electron integrals in MO basis
        integrals_mo = self.get_integrals_mo_basis()
        one_body_coefficients_a = integrals_mo[0]
        one_body_coefficients_b = integrals_mo[1]
        two_body_coefficients_aa = integrals_mo[2]
        two_body_coefficients_bb = integrals_mo[3]
        two_body_coefficients_ab = integrals_mo[4]
        two_body_coefficients_ba = integrals_mo[5]

        wavefunction = super().instantiate_qcwavefunction(
            basis=self.mol.basis,
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
            return_result=self.mf.e_tot,
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

        Example:
        >>> from ase.build import molecule
        >>> from qc2.ase.pyscf import PySCF
        >>>
        >>> molecule = molecule('H2')
        >>> molecule.calc = PySCF()     # => RHF/STO-3G
        >>> molecule.calc.schema_format = "qcschema"
        >>> qcschema = molecule.calc.load('h2.h5')
        >>>
        >>> molecule = molecule('H2')
        >>> molecule.calc = PySCF()     # => RHF/STO-3G
        >>> molecule.calc.schema_format = "fcidump"
        >>> fcidump = molecule.calc.load('h2.fcidump')
        """
        return BaseQc2ASECalculator.load(self, datafile)

    def get_integrals_mo_basis(self) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray, np.ndarray
    ]:
        """Retrieves 1- & 2-body integrals in MO basis from PySCF routines.

        Returns:
            A tuple containing the following:
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
        if beta_coeff is None:
            one_body_int_b = one_body_int_a
        else:
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
        if beta_coeff is None:
            two_body_int_bb = two_body_int_aa
            two_body_int_ba = two_body_int_aa
            two_body_int_ab = two_body_int_aa
        else:
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
        """Retrieves 1- & 2-e integrals in AO basis from PySCF routines."""
        one_e_int = self.mol.intor('int1e_kin') + self.mol.intor('int1e_nuc')
        two_e_int = self.mol.intor("int2e", aosym=1)
        return one_e_int, two_e_int

    def get_molecular_orbitals_coefficients(self) -> Tuple[
        np.ndarray, np.ndarray
    ]:
        """Retrieves alpha and beta MO coeffs from PySCF routines."""
        return self._expand_mo_object(
            self.mf.mo_coeff, array_dimension=3
        )

    def get_molecular_orbitals_energies(self) -> Tuple[
        np.ndarray, np.ndarray
    ]:
        """Retrieves alpha and beta MO energies from PySCF routines."""
        return self._expand_mo_object(
            self.mf.mo_energy, array_dimension=2
        )

    def get_overlap_matrix(self) -> np.ndarray:
        """Retrieves overlap matrix from PySCF routines."""
        return self.mf.get_ovlp()

    def _expand_mo_object(
        self,
        mo_object: Union[
            Tuple[Optional[np.ndarray], Optional[np.ndarray]], np.ndarray
        ],
        array_dimension: int = 2,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Expands the mo object into alpha- and beta-spin components.

        Notes:
            Adapted from Qiskit-Nature pyscfdriver.py.

        Args:
            mo_object: the molecular orbital object to expand.
            array_dimension:  This argument specifies the dimension of the
                numpy array (if a tuple is not encountered). Making this
                configurable permits this function to be used to expand both,
                MO coefficients (3D array) and MO energies (2D array).

        Returns:
            The (alpha, beta) tuple of MO data.
        """
        if isinstance(mo_object, tuple):
            return mo_object

        if len(mo_object.shape) == array_dimension:
            return mo_object[0], mo_object[1]

        return mo_object, None
