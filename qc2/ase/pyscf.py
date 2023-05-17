"""This module defines an ASE interface to PySCF.

https://pyscf.org/ => Official documentation
https://github.com/pyscf/pyscf => GitHub page

Note: Adapted from
https://github.com/pyscf/pyscf/blob/master/pyscf/pbc/tools/pyscf_ase.py
&
https://github.com/pyscf/pyscf/issues/624
"""
from typing import Union, Optional, List, Dict, Any
import warnings
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.units import Ha, Bohr
from ase.calculators.calculator import InputError
from ase.calculators.calculator import CalculatorSetupError
from pyscf import gto, scf, dft, lib


def ase_atoms_to_pyscf(ase_atoms: Atoms) -> List[List[Union[str, np.ndarray]]]:
    """Converts ASE atoms to PySCF atom.

    Args:
        ase_atoms (Atoms): ASE Atoms object.

    Returns:
        List[List[Union[str, np.ndarray]]]: PySCF atom.
    """
    return [[ase_atoms.get_chemical_symbols()[i], ase_atoms.get_positions()[i]]
            for i in range(len(ase_atoms.get_positions()))]


class PySCF(Calculator):
    """An ASE calculator for the PySCF quantum chemistry package.

    Args:
        Calculator (Calculator): Base-class for all ASE calculators.

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

    default_parameters: Dict[str, Any] = {'method': 'dft.RKS',
                                          'basis': '6-31g*',
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

        Note: Basic implementation based on the
            class Psi4(Calculator); see, e.g., ase/ase/calculators/psi4.py.

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
        super().__init__(restart=restart,
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

    def check_pyscf_attributes(self) -> None:
        """Checks for any missing and/or mispelling PySCF input attribute.

        Note: it can also be used to eventually set specific
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
        super().calculate(atoms=atoms)

        # checking that self.atoms has been properly initiated/updated
        if self.atoms is None:
            raise CalculatorSetupError('An Atoms object must be provided to '
                                       'perform a calculation.')

        # the spin variable corresponds to 2S instead of 2S+1
        spin_2s = self.parameters['multiplicity'] - 1

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

    def dump_mol_data(self) -> Dict[str, Any]:
        """Writes relevant molecular info to chkfile in HDF5 format."""
        # Extract molecule info
        charge = self.mol.charge
        natom  = self.mol.natm
        nelec  = self.mol.nelectron
        nalpha = (nelec + self.mol.spin)//2
        nbeta  = (nelec - self.mol.spin)//2
        nao    = self.mol.nao_cart()
        nshells= self.mol.nbas
        multiplicity = self.mol.spin + 1

        cart_coord = []
        for i in range(natom):
            cart_coord.append(self.mol.atom_coords().tolist()[i][0])
            cart_coord.append(self.mol.atom_coords().tolist()[i][1])
            cart_coord.append(self.mol.atom_coords().tolist()[i][2])

        Zint = []
        for i in range(len(self.mol._atm)):
            Zint.append(self.mol._atm[i][0])
        Zreal = Zint

        mol_data = {}
        mol_data = {'charge': charge,
                    'natom': natom,
                    'nelec': nelec,
                    'nalpha': nalpha,
                    'nbeta': nbeta,
                    'nao': nao,
                    'nshells': nshells,
                    'multiplicity': multiplicity,
                    'cart_coord': cart_coord,
                    'Zint': Zint,
                    'Zreal': Zreal}

        if self.mol.output:
            filename = self.mol.output.split(".")
            output_filename = filename[0] + '.chk'
            lib.chkfile.save(output_filename, 'mol_data', mol_data)
        return mol_data

    def dump_basis_set_data(self) -> Dict[str, Any]:
        """Writes basis set data to a chkfile in HDF5 format."""
        # Extract basis set info
        natom  = self.mol.natm
        nshells= self.mol.nbas

        shell_atom_map = []
        orb_momentum = []
        contract_coeff = []
        for i in range(len(self.mol._bas)):
            shell_atom_map.append(self.mol._bas[i][0] + 1)
            orb_momentum.append(self.mol._bas[i][1])
            contract_coeff.append(self.mol._bas[i][3])

        nprim_shell = []
        coord_shell = []
        for i in range(nshells):
            nprim_shell.append(1)
            coord_shell.append(self.mol.atom_coords().tolist()[shell_atom_map[i] - 1][0])
            coord_shell.append(self.mol.atom_coords().tolist()[shell_atom_map[i] - 1][1])
            coord_shell.append(self.mol.atom_coords().tolist()[shell_atom_map[i] - 1][2])

        prim_exp = []
        for i in range(natom):
            atom_type = self.atoms.symbols[i]
            primitives_exp = self.mol._basis[atom_type]
            for j in range(len(primitives_exp)):
                for k in range(1,len(primitives_exp[0])):
                    prim_exp.append(primitives_exp[j][k][0])

        if self.mol.cart:
            pureamd = 1
            pureamf = 1
        else:
            pureamd = 0
            pureamf = 0

        basis_data = {}
        basis_data = {'nshells': nshells,
                      'pureamd': pureamd,
                      'pureamf': pureamf,
                      'orb_momentum': orb_momentum,
                      'nprim_shell': nprim_shell,
                      'shell_atom_map': shell_atom_map,
                      'prim_exp': prim_exp,
                      'contr_coeffs': [1]*len(prim_exp),
                      'coord_shell': coord_shell}
        
        if self.mol.output:
            filename = self.mol.output.split(".")
            output_filename = filename[0] + '.chk'
            lib.chkfile.save(output_filename, 'basis_data', basis_data)
        return basis_data

    def dump_mo_data(self) -> Dict[str, Any]:
        """Writes molecular orbital data to a chkfile in HDF5 format."""
        # Extract MO coeffs
        scf_e = self.mf.e_tot

        alpha_MO = []
        alpha_energies = []
        beta_MO = []
        beta_energies = []

        if ('UHF' not in self.mf._method_name() and
            'UKS' not in self.mf._method_name()):
            alpha_coeff = self.mf.mo_coeff.copy()
            alpha_energies = self.mf.mo_energy.copy()
        else:
            alpha_coeff = self.mf.mo_coeff[0].copy()
            beta_coeff = self.mf.mo_coeff[1].copy()
            alpha_energies = self.mf.mo_energy[0].copy()
            beta_energies = self.mf.mo_energy[1].copy()

        for i in range(alpha_coeff.shape[1]):
            for j in range(alpha_coeff.shape[0]):
                alpha_MO.append(alpha_coeff[j][i])

        if ('UHF' in self.mf._method_name() and
            'UKS' in self.mf._method_name()):
            for i in range(beta_coeff.shape[1]):
                for j in range(beta_coeff.shape[0]):
                    beta_MO.append(beta_coeff[j][i])

        mo_data = {}
        mo_data = {'scf_e': scf_e,
                   'alpha_energies': alpha_energies,
                   'alpha_MO': alpha_MO,
                   'beta_energies': beta_energies,
                   'beta_MO': beta_MO}
        
        if self.mol.output:
            filename = self.mol.output.split(".")
            output_filename = filename[0] + '.chk'
            lib.chkfile.save(output_filename, 'mo_data', mo_data)
        return mo_data

    def dump_oei_data(self) -> Dict[str, Any]:
        """Writes one-electron integrals to a chkfile in HDF5 format."""
        # Extract one-electron integrals
        scf_e = self.mf.e_tot

        if ('UHF' not in self.mf._method_name() and
            'UKS' not in self.mf._method_name()):
            alpha_coeff = self.mf.mo_coeff.copy()
        else:
            alpha_coeff = self.mf.mo_coeff[0].copy()
            beta_coeff = self.mf.mo_coeff[1].copy()

        E_core = scf_e - self.mf.energy_elec()[0]
        one_body_int = []
        if ('UHF' not in self.mf._method_name() and
            'UKS' not in self.mf._method_name()):
            h1e = alpha_coeff.T.dot(self.mf.get_hcore()).dot(alpha_coeff)
            h1e = self.mf.mo_coeff.T.dot(self.mf.get_hcore()).dot(self.mf.mo_coeff)
            for i in range(1, h1e.shape[0]+1):
                for j in range(1, i+1):
                    one_body_int.append(h1e[i-1, j-1])
        else:
            h1e_alpha = alpha_coeff.T.dot(self.mf.get_hcore()).dot(alpha_coeff)
            h1e_beta = beta_coeff.T.dot(self.mf.get_hcore()).dot(beta_coeff)
            h1e_alpha = self.mf.mo_coeff[0].T.dot(self.mf.get_hcore()).dot(self.mf.mo_coeff[0])
            h1e_beta = self.mf.mo_coeff[1].T.dot(self.mf.get_hcore()).dot(self.mf.mo_coeff[1])
            for i in range(1, h1e_alpha.shape[0]+1):
                for j in range(1, i+1):
                    one_body_int.append(h1e_alpha[i-1, j-1])
                    one_body_int.append(h1e_beta[i-1, j-1])

        oei_data = {}
        oei_data = {'E_core': E_core,
                    'one_body_int': one_body_int}
        
        if self.mol.output:
            filename = self.mol.output.split(".")
            output_filename = filename[0] + '.chk'
            lib.chkfile.save(output_filename, 'oei_data', oei_data)
        return oei_data