"""Input/Output functions for Rose."""

import re
from ase import Atoms
from typing import Dict, Any

def call_pyscf_mo_file_generator(atom: Atoms, **kwargs) -> None:
    """Generates pyscf mo input files for Rose."""
    mol = atom.calc.mol
    mf = atom.calc.wf

    print(kwargs)

    print(mol, mf)


def write_int(f, text, var):
    """Writes an integer value to a file in a specific format.

    Args:
        f (file object): The file object to write to.
        text (str): A string of text to precede the integer value.
        var (int): The integer value to be written to the file.

    Returns:
        None
    """
    f.write("{:43}I{:17d}\n".format(text, var))


def write_int_list(f, text, var):
    """Writes a list of integers to a file in a specific format.

    Args:
        f (file object): The file object to write to.
        text (str): A string of text to precede the list of integers.
        var (list): The list of integers to be written to the file.

    Returns:
        None
    """
    f.write("{:43}{:3} N={:12d}\n".format(text, "I", len(var)))
    dim = 0
    buff = 6
    if (len(var) < 6):
        buff = len(var)
    for i in range((len(var)-1)//6+1):
        for j in range(buff):
            f.write("{:12d}".format(var[dim+j]))
        f.write("\n")
        dim = dim + 6
        if (len(var) - dim) < 6:
            buff = len(var) - dim


def write_singlep_list(f, text, var):
    """Writes a list of single-precision floating-point values to a file object.

    Args:
        f: A file object to write to.
        text (str): The text to be written before the list.
        var (list): The list of single-precision floating-point
            values to write.

    Returns:
        None
    """
    f.write("{:43}{:3} N={:12d}\n".format(text, "R", len(var)))
    dim = 0
    buff = 5
    if (len(var) < 5):
        buff = len(var)
    for i in range((len(var)-1)//5+1):
        for j in range(buff):
            f.write("{:16.8e}".format(var[dim+j]))
        f.write("\n")
        dim = dim + 5
        if (len(var) - dim) < 5:
            buff = len(var) - dim


def write_doublep_list(f, text, var):
    """Writes a list of double precision floating point numbers to a file.

    Args:
        f (file object): the file to write the data to
        text (str): a label or description for the data
        var (list): a list of double precision floating point
            numbers to write to file

    Returns:
        None
    """
    f.write("{:43}{:3} N={:12d}\n".format(text, "R", len(var)))
    dim = 0
    buff = 5
    if (len(var) < 5):
        buff = len(var)
    for i in range((len(var)-1)//5+1):
        for j in range(buff):
            f.write("{:24.16e}".format(var[dim+j]))
        f.write("\n")
        dim = dim + 5
        if (len(var) - dim) < 5:
            buff = len(var) - dim


def read_int(text, f):
    """Reads an integer value from a text file.

    Args:
        text (str): The text to search for in the file.
        f (file): The file object to read from.

    Returns:
        int: The integer value found in the file.
    """
    for line in f:
        if re.search(text, line):
            var = int(line.rsplit(None, 1)[-1])
            return var


def read_real(text, f):
    """Reads a floating-point value from a text file.

    Args:
        text (str): The text to search for in the file.
        f (file): The file object to read from.

    Returns:
        float: The floating-point value found in the file.
    """
    for line in f:
        if re.search(text, line):
            var = float(line.rsplit(None, 1)[-1])
            return var


def read_int_list(text, f):
    """Reads a list of integers from a text file.

    Args:
        text (str): The text to search for in the file.
        f (file): The file object to read from.

    Returns:
        list: A list of integers found in the file.
    """
    for line in f:
        if re.search(text, line):
            n = int(line.rsplit(None, 1)[-1])
            var = []
            for i in range((n-1)//6+1):
                line = next(f)
                for j in line.split():
                    var += [int(j)]
            return var


def read_real_list(text, f):
    """Reads a list of floating-point values from a text file.

    Args:
        text (str): The text to search for in the file.
        f (file): The file object to read from.

    Returns:
        list: A list of floating-point values found in the file.
    """
    for line in f:
        if re.search(text, line):
            n = int(line.rsplit(None, 1)[-1])
            var = []
            for i in range((n-1)//5+1):
                line = next(f)
                for j in line.split():
                    var += [float(j)]
            return var
