"""Set of functions usued for rose-ase module package."""
from typing import List
import re
import subprocess


def clean_up():
    """Remove Rose-ASE calculation outputs."""
    command = ("rm *.xyz *.dfcoef DFCOEF* *.inp INPUT* "
    "MOLECULE.XYZ MRCONEE* *dfpcmo DFPCMO* *.fchk "
    "fort.* timer.dat INFO_MOL *.pyscf "
    "*.npy *.clean OUTPUT_AVAS "
    "OUTPUT_ROSE *.chk ILMO*dat *.out")
    subprocess.run(command, shell=True, capture_output=True)


def extract_number(pattern: str, text: str) -> List[float]:
    """Extracts floating point numbers from a chunk of text selected by a pattern."""
    # Define a regular expression that matches floating point numbers
    number_pattern = re.compile(r'([-+]?\d*\.\d+|[-+]?\d+\.\d*|[-+]?\d+)')

    # Find the numbers in the text that match the pattern
    match = re.search(pattern, text)
    if match:
        sub_string = match.group()
        strings = re.findall(number_pattern, sub_string)
        numbers = [float(string) for string in strings]
        return numbers
    else:
        raise ValueError("No pattern found in text.")


def read_output(filename: str) -> str:
    """Reads output files from Rose."""
    with open(filename, 'r') as f:
        return f.read()
