from ase import Atoms
from qc2.ase.rose import Rose
from qc2.ase.pyscf import PySCF

import os
import subprocess

def clean_stuff():
    command = "rm *xyz *dfcoef DFCOEF* *inp INPUT* \
        MOLECULE.XYZ MRCONEE* *dfpcmo DFPCMO* *fchk \
        *in fort.100 timer.dat INFO_MOL *.pyscf \
        IAO_Fock SAO *.npy *.clean *\ 2* OUTPUT_AVAS \
        *.chk ILMO*dat"
    subprocess.Popen(command, shell=True, cwd='.')

def run_water_amonia_rose():
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

    expected_output = open(
        'test_rose_ase_pyscf-water-ammonia.out').read()

    run_water_amonia_rose()

    output = open('rose.out').read()

    clean_stuff()

    assert output == expected_output
