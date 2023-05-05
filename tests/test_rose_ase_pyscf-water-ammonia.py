import pytest

from ase import Atoms
from qc2.ase.rose import Rose
from qc2.ase.pyscf import PySCF

import re
import subprocess



def clean_stuff():
    """Remove Rose-ASE calculation outputs."""
    command = ("rm *xyz *dfcoef DFCOEF* *inp INPUT* "
    "MOLECULE.XYZ MRCONEE* *dfpcmo DFPCMO* *fchk "
    "*in fort.100 timer.dat INFO_MOL *.pyscf "
    "IAO_Fock SAO *.npy *.clean OUTPUT_AVAS "
    "*.chk ILMO*dat *.out")
    subprocess.run(command, shell=True, capture_output=True)

def extract_number(pattern: str, text: str) -> list():
    """Extract numbers from chunks of text selected from patterns."""
    # Define a regular expression that matches floating point numbers
    number_pattern = re.compile(r'[-+]?(\d*\.\d+|\d+\.\d*|\d+)')

    # find the numbers if the specific pattern
    match = re.search(pattern, text)
    if match:
        sub_string = match.group()
        strings = re.findall(number_pattern, sub_string)
        numbers = []
        for item in strings:
            numbers.append(float(item))
        return numbers
    else:
        raise Exception("Sorry, no pattern found")


def run_water_ammonia_rose_no_avas():
    """ Water-Ammonia example calculation."""

    # define target molecule
    mol = Atoms('NH3OH2',
                positions=[[-1.39559, -0.02156,  0.00004],
                           [-1.62981,  0.96110, -0.10622],
                           [-1.86277, -0.51254, -0.75597],
                           [-1.83355, -0.33077,  0.86231],
                           [ 1.56850,  0.10589,  0.00001],
                           [ 0.60674, -0.03396, -0.00063],
                           [ 1.94052, -0.78000,  0.00022]
                           ],
                calculator=PySCF(method='scf.RHF',
                                 basis='unc-sto-3g',
                                 multiplicity=1)
                )

    # define molecular fragments
    frag1 = Atoms('NH3',
                  positions=[[-1.39559, -0.02156,  0.00004],
                             [-1.62981,  0.96110, -0.10622],
                             [-1.86277, -0.51254, -0.75597],
                             [-1.83355, -0.33077,  0.86231]
                             ],
                  calculator=PySCF(method='scf.RHF',
                                   basis='unc-sto-3g',
                                   multiplicity=1
                                   )
                  )

    frag2 = Atoms('OH2',
                  positions=[[1.56850,  0.10589,  0.00001],
                             [0.60674, -0.03396, -0.00063],
                             [1.94052, -0.78000,  0.00022]
                             ],
                  calculator=PySCF(method='scf.RHF',
                                   basis='unc-sto-3g',
                                   multiplicity=1
                                   )
                  )

    # perform scf calculations on all species
    mol.get_potential_energy()
    frag1.get_potential_energy()
    frag2.get_potential_energy()

    # prepare Rose-ASE calculator
    rose_calc = Rose(rose_calc_type='mol_frag',
                     rose_target=mol,
                     rose_frags=[frag1, frag2],
                     test = True,
                     avas_frag=[0], nmo_avas=[3, 4, 5],
                     frag_valence=[[1,7],[2,6]],
                     frag_core=[[1,1],[2,1]]
                     )

    # run the calculator
    rose_calc.calculate()

def test_rose_output():

    clean_stuff()

    with open('test_rose_ase_pyscf-water-ammonia.stdout', 'r') as f:
        expected_output = f.read()

    run_water_ammonia_rose_no_avas()

    with open('rose.out', 'r') as f:
        actual_output = f.read()

    charge = 'Partial charges of fragments.*\n{}'.format(('*\n.' * 5) + '*\n' )
    mo_energies = 'Recanonicalized virtual energies of fragments.*\n{}'.format(('*\n.' * 18) + '*\n' )

    expected_charges = extract_number(charge, expected_output)
    actual_charges = extract_number(charge, actual_output)

    expected_mo_energies = extract_number(mo_energies, expected_output)
    actual_mo_energies = extract_number(mo_energies, actual_output)

    clean_stuff()

    # Use the pytest.approx method to compare the two lists with tolerance
    assert actual_charges == pytest.approx(expected_charges, rel=1e-3)
    assert actual_mo_energies == pytest.approx(expected_mo_energies, rel=1.0e-5)
