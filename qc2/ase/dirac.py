"""This module defines an ASE interface to DIRAC23.

Official website:
https://www.diracprogram.org/

GitLab repo:
https://gitlab.com/dirac/dirac
"""
import logging
import subprocess
import os
from typing import Optional, List, Dict, Tuple, Union, Any
import h5py
import numpy as np

from ase import Atoms
from ase.calculators.calculator import FileIOCalculator
from ase.calculators.calculator import InputError, CalculationFailed
from ase.units import Bohr
from ase.io import write

from qiskit_nature.second_q.formats.qcschema import QCSchema
from qiskit_nature import __version__ as qiskit_nature_version
from qiskit_nature.second_q.formats.fcidump import FCIDump

from .dirac_io import write_dirac_in, read_dirac_out, _update_dict
from .qc2_ase_base_class import BaseQc2ASECalculator


class DIRAC(FileIOCalculator, BaseQc2ASECalculator):
    """A general ASE calculator for the relativistic qchem DIRAC code.

    Args:
        FileIOCalculator (FileIOCalculator): Base class for calculators
            that write/read input/output files.
        BaseQc2ASECalculator (BaseQc2ASECalculator): Base class for
            ase calculartors in qc2.
    """
    implemented_properties: List[str] = ['energy']
    label: str = 'dirac'
    command: str = 'pam --inp=PREFIX.inp --mol=PREFIX.xyz --silent ' \
        '--get="AOMOMAT MRCONEE MDCINT"'

    def __init__(self,
                 restart: Optional[bool] = None,
                 ignore_bad_restart_file:
                 Optional[bool] = FileIOCalculator._deprecated,
                 label: Optional[str] = None,
                 atoms: Optional[Atoms] = None,
                 command: Optional[str] = None,
                 **kwargs) -> None:
        """DIRAC-ASE calculator.

        Args:
            restart (bool, optional): Prefix for restart file.
                May contain a directory. Defaults to None: don't restart.
            ignore_bad_restart_file (bool, optional): Deprecated and will
                stop working in the future. Defaults to False.
            label (str, optional): Calculator name. Defaults to 'dirac'.
            atoms (Atoms, optional): Atoms object to which the calculator
                will be attached. When restarting, atoms will get its
                positions and unit-cell updated from file. Defaults to None.
            command (str, optional): Command used to start calculation.
                Defaults to None.
            directory (str, optional): Working directory in which
                to perform calculations. Defaults to '.'.

        Example of a typical ASE-DIRAC input:

        >>> from ase import Atoms
        >>> from ase.build import molecule
        >>> from qc2.ase import DIRAC
        >>>
        >>> molecule = Atoms(...) or molecule = molecule('...')
        >>> molecule.calc = DIRAC(dirac={}, wave_function={}...)
        >>> energy = molecule.get_potential_energy()
        """
        # initialize ASE base class Calculator.
        # see ase/ase/calculators/calculator.py.
        FileIOCalculator.__init__(self, restart, ignore_bad_restart_file,
                                  label, atoms, command, **kwargs)

        self.prefix = self.label

        # Check self.parameters input keys and values
        self.check_dirac_attributes()

        # initialize qc2 base class for ASE calculators.
        # see .qc2_ase_base_class.py
        BaseQc2ASECalculator.__init__(self)

    def check_dirac_attributes(self) -> None:
        """Checks for any missing and/or mispelling DIRAC input attribute.

        Notes:
            it can also be used to eventually set specific
            options in the near future.
        """
        recognized_key_sections: List[str] = [
            'dirac', 'general', 'molecule',
            'hamiltonian', 'wave_function', 'analyse', 'properties',
            'visual', 'integrals', 'grid', 'moltra'
            ]

        # check any mispelling
        for key, value in self.parameters.items():
            if key not in recognized_key_sections:
                raise InputError('Keyword', key,
                                 ' not recognized. Please check input.')

        # set default parameters
        if 'dirac' not in self.parameters:
            key = 'dirac'
            value = {'.title': 'DIRAC-ASE calculation',
                     '.wave function': ''}
            # **DIRAC heading must always come first in the dict/input
            self.parameters = _update_dict(self.parameters, key, value)

        if 'hamiltonian' not in self.parameters:
            self.parameters.update(hamiltonian={'.nonrel': ''})

        if 'wave_function' not in self.parameters:
            self.parameters.update(wave_function={'.scf': ''})

        if 'molecule' not in self.parameters:
            self.parameters.update(molecule={'*basis': {'.default': 'sto-3g'}})

        if 'integrals' not in self.parameters:
            # useful to compare with nonrel calc done with other programs
            self.parameters.update(integrals={'.nucmod': '1'})

        if '*charge' not in self.parameters['molecule']:
            self.parameters['molecule']['*charge'] = {'.charge': '0'}

        if '*symmetry' not in self.parameters['molecule']:
            self.parameters['molecule']['*symmetry'] = {'.nosym': '#'}

        if '.4index' not in self.parameters['dirac']:
            # activates the transformation of integrals to MO basis
            self.parameters['dirac'].update({'.4index': ''})

        if ('.4index' in self.parameters['dirac'] and
                'moltra' not in self.parameters):
            # calculates all integrals, including core
            self.parameters.update(moltra={'.active': 'all'})

    def calculate(self, *args, **kwargs) -> None:
        """Execute DIRAC workflow."""
        FileIOCalculator.calculate(self, *args, **kwargs)

    def write_input(
            self,
            atoms: Optional[Atoms] = None,
            properties: Optional[List[str]] = None,
            system_changes: Optional[List[str]] = None
            ) -> None:
        """Generate all necessary inputs for DIRAC."""
        FileIOCalculator.write_input(self, atoms, properties, system_changes)

        # generate xyz geometry file
        xyz_file = self.prefix + ".xyz"
        write(xyz_file, atoms)

        # generate DIRAC inp file
        inp_file = self.prefix + ".inp"
        write_dirac_in(inp_file, **self.parameters)

    def read_results(self):
        """Read energy from DIRAC output file."""
        out_file = self.prefix + "_" + self.prefix + ".out"
        output = read_dirac_out(out_file)
        self.results = output

    def save(self, datafile: h5py.File) -> None:
        """Dumps qchem data to a datafile using QCSchema or FCIDump formats.

        Args:
            datafile (Union[h5py.File, str]): file to save the data to.

        Notes:
            files are written following the QCSchema or FCIDump formats.

        Returns:
            None

        Example:
        >>> from ase.build import molecule
        >>> from qc2.ase import DIRAC
        >>>
        >>> molecule = molecule('H2')
        >>> molecule.calc = DIRAC()  # => RHF/STO-3G
        >>> molecule.calc.schema_format = "qcschema"
        >>> molecule.calc.get_potential_energy()
        >>> molecule.calc.save('h2.h5')
        >>>
        >>> molecule = molecule('H2')
        >>> molecule.calc = DIRAC()  # => RHF/STO-3G
        >>> molecule.calc.schema_format = "fcidump"
        >>> molecule.calc.get_potential_energy()
        >>> molecule.calc.save('h2.fcidump')
        """
        # in case of fcidump format
        if self._schema_format == "fcidump":
            self._get_dirac_fcidump()
            os.rename('FCIDUMP', datafile)
            return

        # in case of qcschema format
        # get info about the basis set used
        if '.default' in self.parameters['molecule']['*basis']:
            basis = self.parameters['molecule']['*basis']['.default']
        else:
            basis = 'special'

        # then get # of molecular orbitals
        nmo = self._get_from_dirac_hdf5_file(
           '/result/wavefunctions/scf/mobasis/n_mo'
        )
        nmo = sum(nmo)
        # in case of relativistic calculations...
        if ('.nonrel' not in self.parameters['hamiltonian'] and
                '.levy-leblond' not in self.parameters['hamiltonian']):
            nmo = nmo // 2
            logging.warning(
                'At the moment, DIRAC-ASE relativistic '
                'calculations may not work properly with '
                'Qiskit and/or Pennylane...'
            )
        # approximate definition of # of alpha and beta electrons
        # does not work for pure triplet ground states!?
        nuc_charge = self._get_from_dirac_hdf5_file(
            '/input/molecule/nuc_charge'
        )
        molecular_charge = int(
            self.parameters['molecule']['*charge']['.charge']
        )
        nelec = int(sum(nuc_charge)) - molecular_charge
        calcinfo_nbeta = nelec // 2
        calcinfo_nalpha = nelec - calcinfo_nbeta

        # get 1- and 2-electron integrals from FCIDUMP file
        integrals = self.get_integrals_mo_basis()
        one_body_integrals = integrals[2]
        two_body_integrals = integrals[3]

        # format these integrals to make them compatible with qcschema
        integrals_mo = self._format_fcidump_mo_integrals(
            one_body_integrals, two_body_integrals, nmo
        )
        one_body_coefficients_a = integrals_mo[0]
        one_body_coefficients_b = integrals_mo[1]
        two_body_coefficients_aa = integrals_mo[2]
        two_body_coefficients_bb = integrals_mo[3]
        two_body_coefficients_ab = integrals_mo[4]
        two_body_coefficients_ba = integrals_mo[5]

        # create instances of QCSchema's component dataclasses
        topology = super().instantiate_qctopology(
            symbols=self._get_from_dirac_hdf5_file(
                '/input/molecule/symbols'
            ),
            geometry=self._get_from_dirac_hdf5_file(
                '/input/molecule/geometry'
            ) / Bohr,
            molecular_charge=int(
                self.parameters['molecule']['*charge']['.charge']
            ),
            molecular_multiplicity='',
            atomic_numbers=self._get_from_dirac_hdf5_file(
                '/input/molecule/nuc_charge'
            ),
            schema_name="qcschema_molecule",
            schema_version=qiskit_nature_version
        )

        provenance = super().instantiate_qcprovenance(
            creator=self.name,
            version='dirac23',
            routine=f"ASE-{self.__class__.__name__}.save()"
        )

        model = super().instantiate_qcmodel(
            basis=basis,
            method=list(
                self.parameters['wave_function'].keys()
            )[-1].strip('.')
        )

        properties = super().instantiate_qcproperties(
            calcinfo_nbasis=self._get_from_dirac_hdf5_file(
                '/input/aobasis/1/n_ao'
            )[0],
            calcinfo_nmo=nmo,
            calcinfo_nalpha=calcinfo_nalpha,
            calcinfo_nbeta=calcinfo_nbeta,
            calcinfo_natom=self._get_from_dirac_hdf5_file(
                '/input/molecule/n_atoms'
            )[0],
            nuclear_repulsion_energy=integrals[0],
            return_energy=self._get_from_dirac_hdf5_file(
                '/result/wavefunctions/scf/energy'
            )[0]
        )

        # TODO: 1. integrals in AO basis (one_e_int_ao, two_e_int_ao)
        #       2. add mo coefficients and mo energies
        #          (alpha_coeff, beta_coeff, alpha_mo, beta_mo)
        wavefunction = super().instantiate_qcwavefunction(
                    basis=basis,
                    # scf_fock_a=one_e_int_ao.flatten(),
                    # #scf_fock_b=one_e_int_ao.flatten(),
                    # scf_eri=two_e_int_ao.flatten(),
                    scf_fock_mo_a=one_body_coefficients_a.flatten(),
                    scf_fock_mo_b=one_body_coefficients_b.flatten(),
                    scf_eri_mo_aa=two_body_coefficients_aa.flatten(),
                    scf_eri_mo_bb=two_body_coefficients_bb.flatten(),
                    scf_eri_mo_ba=two_body_coefficients_ba.flatten(),
                    scf_eri_mo_ab=two_body_coefficients_ab.flatten(),
                    # scf_orbitals_a=alpha_coeff.flatten(),
                    # scf_orbitals_b=beta_coeff.flatten(),
                    # scf_eigenvalues_a=alpha_mo.flatten(),
                    # scf_eigenvalues_b=beta_mo.flatten(),
                    localized_orbitals_a='',
                    localized_orbitals_b=''
                )

        qcschema = super().instantiate_qcschema(
            schema_name='qcschema_molecule',
            schema_version=qiskit_nature_version,
            driver='energy',
            keywords={},
            return_result=self._get_from_dirac_hdf5_file(
                '/result/wavefunctions/scf/energy'
            )[0],
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
        >>> from qc2.ase import DIRAC
        >>>
        >>> molecule = molecule('H2')
        >>> molecule.calc = DIRAC()     # => RHF/STO-3G
        >>> molecule.calc.schema_format = "qcschema"
        >>> qcschema = molecule.calc.load('h2.h5')
        >>>
        >>> molecule = molecule('H2')
        >>> molecule.calc = DIRAC()     # => RHF/STO-3G
        >>> molecule.calc.schema_format = "fcidump"
        >>> fcidump = molecule.calc.load('h2.fcidump')
        """
        if self._schema_format == "fcidump":
            logging.warning(
                'FCIDump.from_file() in Qiskit-Nature may not load '
                'properly integrals from DIRAC FCIDUMP file. '
                'The reason lies in the fact that FCIDump.from_file() '
                'reads integrals as `SymmetricTwoBodyIntegrals`, while, '
                'in DIRAC, FCIDUMP file is generated with the full list '
                'of terms.'
            )

        return BaseQc2ASECalculator.load(self, datafile)

    def get_integrals_mo_basis(self) -> Tuple[
        Union[float, complex], Dict[int, Union[float, complex]],
        Dict[Tuple[int, int], Union[float, complex]],
        Dict[Tuple[int, int, int, int], Union[float, complex]]
    ]:
        """Retrieves 1- and 2-body integrals in MO basis from DIRAC FCIDUMP.

        Notes:
            Adapted from Openfermion-Dirac:
            see: https://github.com/bsenjean/Openfermion-Dirac.

        Returns:
            A tuple containing the following:
                - e_core (Union[float, complex]): Nuclear repulsion energy.
                - spinor (Dict[int, Union[float, complex]]): Dictionary of
                    spinor values with their corresponding indices.
                - one_body_int (Dict[Tuple[int, int], Union[float, complex]]):
                    Dictionary of one-body integrals with their corresponding
                    indices as tuples.
                - two_body_int (Dict[Tuple[int, int, int, int],
                    Union[float, complex]]): Dictionary of two-body integrals
                    with their corresponding indices as tuples.
        """
        self._get_dirac_fcidump()

        e_core = 0
        spinor = {}
        one_body_int = {}
        two_body_int = {}
        num_lines = sum(
            1 for line in open("FCIDUMP", encoding='utf-8')
        )
        with open('FCIDUMP', encoding='utf-8') as f:
            start_reading = 0
            for line in f:
                start_reading += 1
                if "&END" in line:
                    break
            listed_values = [
                [token for token in line.split()] for line in f.readlines()]
            complex_int = False
            if len(listed_values[0]) == 6:
                complex_int = True
            if not complex_int:
                for row in range(num_lines-start_reading):
                    a_1 = int(listed_values[row][1])
                    a_2 = int(listed_values[row][2])
                    a_3 = int(listed_values[row][3])
                    a_4 = int(listed_values[row][4])
                    if a_4 == 0 and a_3 == 0:
                        if a_2 == 0:
                            if a_1 == 0:
                                e_core = float(listed_values[row][0])
                            else:
                                spinor[a_1] = float(listed_values[row][0])
                        else:
                            one_body_int[a_1, a_2] = float(
                                listed_values[row][0])
                    else:
                        two_body_int[a_1, a_2, a_3, a_4] = float(
                            listed_values[row][0])
            if complex_int:
                for row in range(num_lines-start_reading):
                    a_1 = int(listed_values[row][2])
                    a_2 = int(listed_values[row][3])
                    a_3 = int(listed_values[row][4])
                    a_4 = int(listed_values[row][5])
                    if a_4 == 0 and a_3 == 0:
                        if a_2 == 0:
                            if a_1 == 0:
                                e_core = complex(
                                   float(listed_values[row][0]),
                                   float(listed_values[row][1]))
                            else:
                                spinor[a_1] = complex(
                                   float(listed_values[row][0]),
                                   float(listed_values[row][1]))
                        else:
                            one_body_int[a_1, a_2] = complex(
                               float(listed_values[row][0]),
                               float(listed_values[row][1]))
                    else:
                        two_body_int[a_1, a_2, a_3, a_4] = complex(
                           float(listed_values[row][0]),
                           float(listed_values[row][1]))

        return e_core, spinor, one_body_int, two_body_int

    def get_integrals_ao_basis(self) -> Tuple[Any, ...]:
        """Calculates one- and two-electron integrals in AO basis.

        TODO: after Luuks's python interface
        """
        raise NotImplementedError(
            "get_integrals_ao_basis() not yet implemented in DIRAC-ASE"
        )

    def get_molecular_orbitals_coefficients(self) -> Tuple[Any, ...]:
        """Reads alpha and beta molecular orbital coefficients.

        TODO: after Luuks's python interface
        """
        raise NotImplementedError(
            "get_molecular_orbitals_coefficients() not yet "
            "implemented in DIRAC-ASE"
        )

    def get_molecular_orbitals_energies(self) -> Tuple[Any, ...]:
        """Reads alpha and beta orbital energies.

        TODO: after Luuks's python interface
        """
        raise NotImplementedError(
            "get_molecular_orbitals_energies() not yet "
            "implemented in DIRAC-ASE"
        )

    def get_overlap_matrix(self) -> Tuple[Any, ...]:
        """Reads overlap matrix.

        TODO: after Luuks's python interface
        """
        raise NotImplementedError(
            "get_overlap_matrix() not yet implemented in DIRAC-ASE"
        )

    def _get_from_dirac_hdf5_file(self, property_name) -> Any:
        """Helper routine to open dirac HDF5 output and extract property."""
        out_hdf5_file = self.prefix + "_" + self.prefix + ".h5"
        try:
            with h5py.File(out_hdf5_file, "r") as f:
                data = f[property_name][...]
        except (KeyError, IOError):
            data = None
        return data

    def _get_dirac_fcidump(self) -> None:
        """Helper routine to generate DIRAC FCIDUMP file.

        Notes:
            Requires MRCONEE MDCINT files obtained using
            **DIRAC .4INDEX, **MOLTRA .ACTIVE all and
            'pam ... --get="MRCONEE MDCINT"' options.

        Raises:
            EnvironmentError: If the command execution fails.
            CalculationFailed: If the calculator fails with
                a non-zero error code.
        """
        command = "dirac_mointegral_export.x fcidump"
        try:
            proc = subprocess.Popen(command, shell=True, cwd=self.directory)
        except OSError as err:
            msg = f"Failed to execute {command}"
            raise EnvironmentError(msg) from err

        errorcode = proc.wait()

        if errorcode:
            path = os.path.abspath(self.directory)
            msg = (f"Calculator {self.name} failed with "
                   f"command {command} failed in {path} "
                   f"with error code {errorcode}")
            raise CalculationFailed(msg)

    def _format_fcidump_mo_integrals(
        self,
        one_body_integrals: Dict[Tuple[int, int], Union[float, complex]],
        two_body_integrals: Dict[
            Tuple[int, int, int, int], Union[float, complex]
        ],
        nmo: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
               np.ndarray, np.ndarray, np.ndarray]:
        """Helper routine to format DIRAC FCIDUMP integrals.

        Notes:
            Adapted from Openfermion-Dirac:
            see: https://github.com/bsenjean/Openfermion-Dirac.

        Returns:
            A tuple containing the following:
                - one_body_coefficients_a & one_body_coefficients_b:
                    Numpy arrays containing alpha and beta components
                    of the one-body integrals.
                - two_body_coefficients_aa, two_body_coefficients_bb,
                    two_body_coefficients_ab & two_body_coefficients_ba:
                    Numpy arrays containing alpha-alpha, beta-beta,
                    alpha-beta & beta-alpha components of the two-body
                    integrals.
        """
        # tolerance to consider number zero.
        EQ_TOLERANCE = 1e-8

        # slipt 1-body integrals into alpha and beta contributions
        one_body_coefficients_a = np.zeros((nmo, nmo), dtype=np.float64)
        one_body_coefficients_b = np.zeros((nmo, nmo), dtype=np.float64)

        # transform alpha and beta 1-body coeffs into QCSchema format
        for p in range(nmo):
            for q in range(nmo):

                # alpha indexes
                alpha_p = 2 * p + 1
                alpha_q = 2 * q + 1

                # beta indexes
                beta_p = 2 * p + 2
                beta_q = 2 * q + 2

                # alpha and beta 1-body coeffs
                one_body_coefficients_a[p, q] = one_body_integrals[
                    (alpha_p, alpha_q)]
                one_body_coefficients_b[p, q] = one_body_integrals[
                    (beta_p, beta_q)]

        # truncate numbers lower than EQ_TOLERANCE
        one_body_coefficients_a[np.abs(
            one_body_coefficients_a) < EQ_TOLERANCE] = 0.
        one_body_coefficients_b[np.abs(
            one_body_coefficients_b) < EQ_TOLERANCE] = 0.

        # slipt 2-body coeffs into alpha-alpha, beta-beta,
        # alpha-beta and beta-alpha contributions
        two_body_coefficients_aa = np.zeros(
            (nmo, nmo, nmo, nmo), dtype=np.float64)
        two_body_coefficients_bb = np.zeros(
            (nmo, nmo, nmo, nmo), dtype=np.float64)
        two_body_coefficients_ab = np.zeros(
            (nmo, nmo, nmo, nmo), dtype=np.float64)
        two_body_coefficients_ba = np.zeros(
            (nmo, nmo, nmo, nmo), dtype=np.float64)

        # transform alpha-alpha, beta-beta, alpha-beta and beta-alpha
        # 2-body coeffs into QCSchema format
        for p in range(nmo):
            for q in range(nmo):
                for r in range(nmo):
                    for s in range(nmo):

                        # alpha indexes
                        alpha_p = 2 * p + 1
                        alpha_q = 2 * q + 1
                        alpha_r = 2 * r + 1
                        alpha_s = 2 * s + 1

                        # beta indexes
                        beta_p = 2 * p + 2
                        beta_q = 2 * q + 2
                        beta_r = 2 * r + 2
                        beta_s = 2 * s + 2

                        if (alpha_p, alpha_q,
                                alpha_r, alpha_s) in two_body_integrals:

                            # alpha-alpha unique matrix element
                            aa_term = two_body_integrals[
                                (alpha_p, alpha_q, alpha_r, alpha_s)]

                            # exploiting perm symm of 2-body integrals
                            two_body_coefficients_aa[p, q, r, s] = aa_term
                            two_body_coefficients_aa[q, p, s, r] = aa_term
                            two_body_coefficients_aa[r, s, p, q] = np.conj(
                                aa_term
                            )
                            two_body_coefficients_aa[s, r, q, p] = np.conj(
                                aa_term
                            )

                            # restricted non-relativistic case
                            two_body_coefficients_ba[p, q, r, s] = aa_term
                            two_body_coefficients_ba[q, p, s, r] = aa_term
                            two_body_coefficients_ba[r, s, p, q] = np.conj(
                                aa_term
                            )
                            two_body_coefficients_ba[s, r, q, p] = np.conj(
                                aa_term
                            )

                            two_body_coefficients_ab[p, q, r, s] = aa_term
                            two_body_coefficients_ab[q, p, s, r] = aa_term
                            two_body_coefficients_ab[r, s, p, q] = np.conj(
                                aa_term
                            )
                            two_body_coefficients_ab[s, r, q, p] = np.conj(
                                aa_term
                            )

                            two_body_coefficients_bb[p, q, r, s] = aa_term
                            two_body_coefficients_bb[q, p, s, r] = aa_term
                            two_body_coefficients_bb[r, s, p, q] = np.conj(
                                aa_term
                            )
                            two_body_coefficients_bb[s, r, q, p] = np.conj(
                                aa_term
                            )

                        # non-restricted case ?
                        if (beta_p, beta_q,
                                beta_r, beta_s) in two_body_integrals:

                            # beta-beta unique matrix element
                            bb_term = two_body_integrals[
                                (beta_p, beta_q, beta_r, beta_s)]

                            two_body_coefficients_bb[p, q, r, s] = bb_term
                            two_body_coefficients_bb[q, p, s, r] = bb_term
                            two_body_coefficients_bb[r, s, p, q] = np.conj(
                                bb_term
                            )
                            two_body_coefficients_bb[s, r, q, p] = np.conj(
                                bb_term
                            )

                        if (alpha_p, beta_q,
                                beta_r, alpha_s) in two_body_integrals:

                            # alpha-beta unique matrix element
                            ab_term = two_body_integrals[
                                (alpha_p, beta_q, beta_r, alpha_s)]

                            two_body_coefficients_ab[p, q, r, s] = ab_term
                            two_body_coefficients_ab[q, p, s, r] = ab_term
                            two_body_coefficients_ab[r, s, p, q] = np.conj(
                                ab_term
                            )
                            two_body_coefficients_ab[s, r, q, p] = np.conj(
                                ab_term
                            )

                        if (beta_p, alpha_q,
                                alpha_r, beta_s) in two_body_integrals:

                            # beta-alpha unique matrix element
                            ba_term = two_body_integrals[
                                (beta_p, alpha_q, alpha_r, beta_s)]

                            two_body_coefficients_ba[p, q, r, s] = ba_term
                            two_body_coefficients_ba[q, p, s, r] = ba_term
                            two_body_coefficients_ba[r, s, p, q] = np.conj(
                                ba_term
                            )
                            two_body_coefficients_ba[s, r, q, p] = np.conj(
                                ba_term
                            )

        # truncate numbers lower than EQ_TOLERANCE
        two_body_coefficients_aa[np.abs(
            two_body_coefficients_aa) < EQ_TOLERANCE] = 0.
        two_body_coefficients_bb[np.abs(
            two_body_coefficients_bb) < EQ_TOLERANCE] = 0.
        two_body_coefficients_ab[np.abs(
            two_body_coefficients_ab) < EQ_TOLERANCE] = 0.
        two_body_coefficients_ba[np.abs(
            two_body_coefficients_ba) < EQ_TOLERANCE] = 0.

        return (
            one_body_coefficients_a, one_body_coefficients_b,
            two_body_coefficients_aa, two_body_coefficients_bb,
            two_body_coefficients_ab, two_body_coefficients_ba
        )
