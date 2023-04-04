"""This module defines an ASE interface to PySCF

https://pyscf.org/ => Official documentation             
https://github.com/pyscf/pyscf => GitHub page 

Note: Adapted from https://github.com/pyscf/pyscf/blob/master/pyscf/pbc/tools/pyscf_ase.py
               and https://github.com/pyscf/pyscf/issues/624
"""
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.units import Ha, Bohr
from ase.calculators.calculator import InputError
from ase.calculators.calculator import CalculatorSetupError
import warnings
from typing import Union, Optional, Type, List, Dict, Any

def ase_atoms_to_pyscf(ase_atoms: Atoms) -> List[List[Union[str, np.ndarray]]]:
    """Converts ASE atoms to PySCF atom.

    Args:
        ase_atoms (Atoms): ASE Atoms object.

    Returns:
        List[List[Union[str, np.ndarray]]]: PySCF atom.
    """    
    return [ [ase_atoms.get_chemical_symbols()[i], ase_atoms.get_positions()[i]] for i in range(len(ase_atoms.get_positions()))]


class PySCF(Calculator):
    """An ASE calculator for the open source (python-based) PySCF quantum chemistry package.

    Args:
        Calculator (Calculator): Base-class for all ASE calculators. 

    Raises:
        InputError: If attributes other than 
            'method', 'xc', 'basis', 'multiplicity', 'charge', and 'verbose' are input as Calculator. 
        CalculatorSetupError: If abinitio methods other than 
            'scf.RHF', 'scf.UHF', 'scf.ROHF', 'dft.RKS', 'dft.UKS', and 'dft.ROKS' are selected.

    Example of a typical ASE-PySCF input:

    >>> from ase import Atoms
    >>> from ase.build import molecule
    >>> from qc2.ase.pyscf import PySCF
    >>>
    >>> molecule = Atoms(...) or molecule = molecule('...')
    >>> molecule.calc = PySCF(method='dft.RKS', xc='b3lyp', basis='6-31g*', charge=0, multiplicity=1, verbose=0)
    >>> energy = molecule.get_potential_energy()
    >>> gradient = molecule.get_forces()

    where (currently) 

    method = 'scf.RHF'
           = 'scf.UHF'
           = 'scf.ROHF'
           = 'dft.RKS'
           = 'dft.UKS'
           = 'dft.ROKS'
    """    
    
    implemented_properties: List[str] = ['energy', 'forces']

    default_parameters: Dict[str, Any] = {'method': 'dft.RKS',
                                          'basis': '6-31g*',
                                          'xc': 'b3lyp', 
                                          'multiplicity': 1,
                                          'charge': 0,
                                          'verbose': 0}
    
    def __init__(self, 
                 restart : Optional[bool] = None,
                 ignore_bad_restart: Optional[bool] = False, 
                 label : Optional[str] = 'PySCF', 
                 atoms : Optional[Atoms] = None, 
                 command : Optional[str] = None, 
                 directory : str = '.', 
                 **kwargs) -> None:
        
        """ASE-PySCF Class Constructor to initialize the object. 
        
        Note: Basic implementation based on the class Psi4(Calculator); see, e.g., ase/ase/calculators/psi4.py.

        Args:
            restart (bool, optional): Prefix for restart file. 
                May contain a directory. Defaults to None: don't restart.
            ignore_bad_restart (bool, optional): Deprecated and will stop working in the future. Defaults to False.
            label (str, optional): Calculator name. Defaults to 'PySCF'.
            atoms (Atoms, optional): Atoms object to which the calculator will be attached. 
                When restarting, atoms will get its positions and unit-cell updated from file. Defaults to None.
            command (str, optional): Command used to start calculation. Defaults to None.
            directory (str, optional): Working directory in which perform calculations. Defaults to '.'.
        """    
        
        # initiating (sub)class Calculator; see ase/ase/calculators/calculator.py.
        Calculator.__init__(self, restart=restart,
                            ignore_bad_restart=ignore_bad_restart, label=label,
                            atoms=atoms, command=command, directory=directory, 
                            **kwargs)
        
        self.set_pyscf()

    def set_pyscf(self):
        """This method sets up PySCF intrinsic attributes.

        1). It converts self.parameters.name set by (sub)class Calculator into self.name, e.g., 
        self.parameter.method => self.method
        self.parameter.xc     => self.xc
        self.parameter.basis  => self.basis
                          ... => self.multiplicity 
                          ... => self.charge
                          ... => self.verbose

        2) It also checks for any missing/mispelling input attributes. 

        Note: it can also be used to set specific environment variables. 
        """

        recognized_attributes: List[str] = ['ignore_bad_restart', 'command', 'method', 
                                            'xc', 'basis', 'multiplicity', 'charge', 
                                            'verbose', 'kpts', 'nbands', 'smearing']
                
        # self.parameters gathers all PYSCF input options in a dictionary. 
        # It is defined in class Calculator(BaseCalculator); see ase/ase/calculators/calculator.py
        for key, value in self.parameters.items():
            # check attribute name
            if key in recognized_attributes:
                # setting PySCF attributes => creating self.name from self.parameters.name 
                setattr(self, key, value)
            else:
                raise InputError('Attribute', key, 'not recognized. Please check input.')

        # dealing with lack of multiplicity and charge info
        if 'multiplicity' not in self.parameters.keys():
            self.multiplicity = 1
            warnings.warn('Multiplicity not provided. Assuming default singlet.')

        if 'charge' not in self.parameters.keys():
            self.charge = 0
            warnings.warn('Charge not provided. Assuming default zero.')

        # verbose sets the amount of pyscf printing => verbose = 0 prints no info
        if 'verbose' not in self.parameters.keys():
            self.verbose = 0  

        # dealing with some ASE specific inputs
        if 'kpts' in self.parameters.keys():
            raise InputError('This ASE-PySCF interface does not yet implement periodic calculations, and thus does not'
                             ' accept k-points as parameters. Please remove this '
                             'argument.')
        
        if 'nbands' in self.parameters.keys():
            raise InputError('This ASE-PySCF interface does not support the keyword "nbands".')

        if 'smearing' in self.parameters.keys():
            raise InputError('Finite temperature DFT is not currently implemented in'
                             ' this PySCF-ASE interface, thus a smearing argument '
                             'cannot be utilized. Please remove this '
                             'argument.')    

    def calculate(self, atoms: Optional[Atoms]=None, properties: List[str]=['energy'], system_changes: List[str]=all_changes) -> None:
        """This method is the core responsible for the actual calculation.

        Note: Implementation based on Calculator.calculate() method; see also ase/ase/calculators/calculator.py. 

        Args:
            atoms (Atoms, optional): Atoms object to which the calculator is attached. Defaults to None.
            properties (list[str], optional): List of what properties need to be calculated. Defaults to ['energy'].
            system_changes (list[str], optional): List of what has changed since last calculation. 
                Can be any combination of these six: 'positions', 'numbers', 'cell', 'pbc', 
                'initial_charges' and 'initial_magmoms'. Defaults to all_changes.

        Raises:
            CalculatorSetupError: If a proper geometry is not provided.
            CalculatorSetupError: If abinitio methods other than 
            'scf.RHF', 'scf.UHF', 'scf.ROHF', 'dft.RKS', 'dft.UKS', and 'dft.ROKS' are selected.    
        """

        implemented_methods: List[str] = ['scf.RHF', 'scf.UHF', 'scf.ROHF', 
                                          'dft.RKS', 'dft.UKS', 'dft.ROKS']
        
        from pyscf import gto, scf, dft

        # setting up self.atoms attribute. This is extracted from the atoms Atoms object. 
        Calculator.calculate(self, atoms=atoms)

        # checking that self.atoms has been properly initiated/updated
        if self.atoms is None:
            raise CalculatorSetupError('An Atoms object must be provided to '
                                       'perform a calculation.')
    
        # the spin variable corresponds to 2S instead of 2S+1
        spin = self.multiplicity - 1

        # passing geometry and other definitions
        molecule = gto.M(atom=ase_atoms_to_pyscf(self.atoms), basis=self.basis, charge=self.charge, spin=spin)

        # checking wf input name => this is case sensitive
        if self.method in implemented_methods:
            wf = eval(self.method)(molecule)
        else:
            raise CalculatorSetupError('Method not yet implemented. '
                                       'Current PySCF-ASE calculator only allows for', 
                                       implemented_methods, 
                                       'wave functions. Please check input method.')
        
        wf.verbose = self.verbose

        if 'dft' in self.method:
            wf.xc = self.xc    
        
        # calculating energy in eV
        energy = wf.kernel() * Ha
        self.results['energy'] = energy

        # calculating forces
        if 'forces' in properties:
            gf = wf.nuc_grad_method()
            gf.verbose = self.verbose
            if 'dft' in self.method:
                gf.grid_response = True
            # analytic gradienta in eV/Angstrom
            forces = -1.0 * gf.kernel() * (Ha / Bohr)
            totalforces = []
            totalforces.extend(forces)
            totalforces = np.array(totalforces)
            self.results['forces'] = totalforces

if __name__ == '__main__':
    
    print(PySCF.__doc__)
    #help(PySCF)

    from ase import Atoms
    from ase.build import molecule
    from ase.units import Ha
    
    mol = molecule('H2')
    mol.calc = PySCF(method='dft.RKS', xc='pbe', basis='sto-3g')

    #print(mol.calc.parameters.method)
    #print(mol.get_potential_energy()/Ha)