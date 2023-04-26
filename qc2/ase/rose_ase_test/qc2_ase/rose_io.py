"""IO functions necessary for Rose."""
from typing import Dict, Any
from ase.units import Ha  # => needed only for testing; remove latter.


def write_input_genibo_avas(input_data: Dict[str, Any]) -> None:
    """Generates INPUT_GENIBO & INPUT_AVAS fortran files for Rose.

    Args:
        input_data (Dict[str, Any]): Rose input options.
    """
    print("Writing INPUT_GENIBO and INPUT_AVAS....done\n")

    version = input_data.version
    charge = input_data.rose_target.calc.parameters.charge
    exponent = input_data.exponent
    restricted = input_data.restricted
    test = input_data.test
    avas_frag = input_data.avas_frag
    mo_calculator = input_data.rose_target.calc
    natom = len(input_data.rose_target.symbols)
    nmo_avas = input_data.nmo_avas

    # creating INPUT_GENIBO file
    with open("INPUT_GENIBO", "w") as f:
        f.write("**ROSE\n")
        f.write(".VERSION\n")
        f.write(version+"\n")
        f.write(".CHARGE\n")
        f.write(str(charge)+"\n")
        f.write(".EXPONENT\n")
        f.write(str(exponent)+"\n")
        if not restricted:
            f.write(".UNRESTRICTED\n")
        f.write(".FILE_FORMAT\n")
        f.write(mo_calculator+"\n")
        if test == 1:
            f.write(".TEST  \n")
        f.write(".AVAS  \n")
        f.write(str(len(avas_frag))+"\n")
        f.writelines("{:3d}".format(item) for item in avas_frag)
        f.write("\n*END OF INPUT\n")

    # creating INPUT_AVAS file
    with open("INPUT_AVAS", "w") as f:
        f.write(str(natom) + " # natoms\n")
        f.write(str(charge) + " # charge\n")
        if restricted:
            f.write("1 # restricted\n")
        else:
            f.write("0 # restricted\n")
        f.write("1 # spatial orbs\n")
        f.write(mo_calculator+"   # MO file for the full molecule\n")
        f.write(mo_calculator+"   # MO file for the fragments\n")
        f.write(str(len(nmo_avas)) + " # number of valence MOs in B2\n")
        f.writelines("{:3d}".format(item) for item in nmo_avas)


def write_input_mol_frags_xyz(input_data: Dict[str, Any]) -> None:
    """Generates Molecule and Fragment xyz files for Rose.

    Args:
        input_data (Dict[str, Any]): Rose input options.
    """
    print("Generating Molecule.xyz and Frags.xyz....done\n")
    pass


def write_mo_files(input_data: Dict[str, Any]) -> None:
    """Generates atomic and molecular orbitals files for Rose.

    Args:
        input_data (Dict[str, Any]): Rose input options.
    """
    print("Generating orbitals file....done")

    # test
    mol = input_data.rose_target
    print('H2O energy/Eh =', mol.get_potential_energy()/Ha)
    print('H2O orbitals =', mol.calc.wf.mo_coeff)
    for fragment in input_data.rose_frags:
        print(fragment.symbols, 'energy/Eh =',
              fragment.get_potential_energy()/Ha)
        print(fragment.calc.wf.mo_coeff)
    print(" ")
