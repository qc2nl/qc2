"""This module defines an ASE interface to PySCF

https://pyscf.org/ => Official documentation             
https://github.com/pyscf/pyscf => GitHub page

Note: Adapted from https://github.com/pyscf/pyscf/blob/master/pyscf/pbc/tools/pyscf_ase.py
               and https://github.com/pyscf/pyscf/issues/624
"""
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.units import Ha, Bohr
from ase.calculators.calculator import InputError
from ase.calculators.calculator import CalculatorSetupError
import warnings

def ase_atoms_to_pyscf(ase_atoms):
    """Convert ASE atoms to PySCF atom"""
    return [ [ase_atoms.get_chemical_symbols()[i], ase_atoms.get_positions()[i]] for i in range(len(ase_atoms.get_positions()))]


class PySCF(Calculator):
    """An ASE calculator for the open source (python-based) PySCF quantum chemistry package.

    Units:  ase   => units [eV, Angstrom, eV/Angstrom]
            pyscf => units [Hartree, Bohr, Hartree/Bohr]
    
    Example input ASE calculation:

    >>> molecule = Atoms(...)
    >>> molecule.calc = PySCF(method='dft.RKS', xc='b3lyp', basis='6-31g*', charge=0, multiplicity=1, verbose=0)
    >>> energy = molecule.get_potential_energy()
    >>> gradient = molecule.get_forces()

    Alternatively, the wave function can be defined as:

    >>> molecule.calc = PySCF()
    >>> molecule.calc.method = 'dft.RKS'
    >>> molecule.calc.xc = 'b3lyp'
    >>> molecule.calc.basis = '6-31g*'
    >>> molecule.calc.charge = 0 
    >>> molecule.calc.multiplicity = 1
    >>> molecule.calc.verbose = 0

    where (currently) 

    method = 'scf.RHF'
           = 'scf.UHF'
           = 'scf.ROHF'
           = 'dft.RKS'
           = 'dft.UKS'
           = 'dft.ROKS'              
    """
    implemented_properties = ['energy', 'forces']

    default_parameters = {'method': 'dft.RKS',
                          'basis': '6-31g*',
                          'xc': 'b3lyp', 
                          'multiplicity': 1,
                          'charge': 0,
                          'verbose': 0}

    def __init__(self, restart=None, ignore_bad_restart=False,
                 label='PySCF', atoms=None, command=None, directory='.', 
                 **kwargs):
        
        Calculator.__init__(self, restart=restart,
                            ignore_bad_restart=ignore_bad_restart, label=label,
                            atoms=atoms, command=command, directory=directory, 
                            **kwargs)
        
        self.set_pyscf()

    def set_pyscf(self):
        """This function sets up PySCF intrinsic attributes.

        Note: it can also be used to set specific environment variables. 
        """

        # setting PySCF attributes
        # these are => self.method
        #           => self.xc
        #           => self.basis
        #           => self.multiplicity 
        #           => self.charge
        #           => self.verbose
        #           => self....
        for key, value in self.parameters.items():
            #print(key, value)
            setattr(self, key, value)

        # dealing with lack of multiplicity and charge info
        if 'multiplicity' not in self.parameters.keys():
            self.multiplicity = 1
            warnings.warn('Multiplicity not provided. Assuming default singlet')

        if 'charge' not in self.parameters.keys():
            self.charge = 0
            warnings.warn('Charge not provided. Assuming default zero')

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

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):

        from pyscf import gto, scf, dft

        Calculator.calculate(self, atoms=atoms)

        if self.atoms is None:
            raise CalculatorSetupError('An Atoms object must be provided to '
                                       'perform a calculation')
        atoms = self.atoms

        # the spin variable corresponds to 2S instead of 2S+1
        spin = self.multiplicity - 1

        # passing geometry and wave function definitions
        molecule = gto.M(atom=ase_atoms_to_pyscf(atoms), basis=self.basis, charge=self.charge, spin=spin)
        wf = eval(self.method)(molecule)
        wf.verbose = self.verbose

        if 'dft' in self.method:
            wf.xc = self.xc    
        
        # calculating energy in eV
        energy = wf.kernel() * Ha
        self.results['energy'] = energy

        if 'forces' in properties:
            gf = wf.nuc_grad_method()
            gf.verbose = self.verbose
            if 'dft' in self.method:
                gf.grid_response = True
            # calculating analytic gradient in eV/Angstrom
            forces = -1.0 * gf.kernel() * (Ha / Bohr)
            totalforces = []
            totalforces.extend(forces)
            totalforces = np.array(totalforces)
            self.results['forces'] = totalforces