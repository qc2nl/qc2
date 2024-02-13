"""Input/Output functions for Rose."""
from typing import Dict, Any
from copy import deepcopy
import re
from collections import OrderedDict

from ase.calculators.calculator import InputError
from ase.units import Ha


def _update_dict(
        dictionary: Dict[str, Any],
        key: str,
        value: Any
) -> Dict[str, Any]:
    """Updates a dict with a new key-value pair and put it at first position.

    Args:
        dictionary: The original dictionary to be updated.
        key: The key of the new element to be added.
        value: The value of the new element to be added.

    Returns:
        The updated dictionary with the new key-value pair at first position.
    """
    ordered_dict = OrderedDict(dictionary)
    ordered_dict[key] = value
    updated_dict = OrderedDict(
        [(key, ordered_dict[key]) for key in reversed(ordered_dict)]
    )
    return updated_dict


def _replace_underscores(dictionary: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively replaces underscores in dict keys and values with spaces.

    Args:
        dictionary: The dictionary to be processed.

    Returns:
        The dictionary with underscores replaced by spaces.
    """
    new_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, dict):
            new_dict[key.replace("_", " ")] = _replace_underscores(value)
        else:
            new_dict[key.replace("_", " ")] = value.replace("_", " ")
    return new_dict


def _format_value(arg: Any) -> str:
    """Formats the input value.

    Args:
        arg: The value to be formatted.

    Returns:
        The formatted value as a string.
    """
    if isinstance(arg, dict):
        formatted_values = []
        for key, val in arg.items():
            formatted_values.append(
                '{}\n{}'.format(str(key).upper(), str(val))
            )
        format_str = '\n'.join(formatted_values)
    elif isinstance(arg, (float, int, str)):
        if not arg:
            format_str = '{}'.format(str(arg))
        else:
            format_str = '\n{}'.format(str(arg))
    else:
        raise InputError('Format for', arg, 'not allowed.')
    return format_str


def _write_block(name: str, args: Dict[str, Any]) -> str:
    """Writes a block of formatted data.

    Args:
        name: The name of the block.
        args: The dictionary containing the data for the block.

    Returns:
        The formatted block as a string.
    """
    out = ['**{}'.format(name.upper())]
    for key, value in args.items():
        if key.startswith('.'):
            out.append('{}{}'.format(key.upper(), _format_value(value)))
        elif key.startswith('*'):
            out.append('{}\n{}'.format(key.upper(), _format_value(value)))
        else:
            raise InputError(
                'Check input! Options and subsections '
                'must start with . and *, respectively'
            )
    return '\n'.join(out)


def write_dirac_in(input_filename: str, **params: Dict[str, Any]):
    """Writes DIRAC input data to a file.

    Args:
        input_filename: input file to be generated.
        params: Additional parameters for the DIRAC input.

    Returns:
        None.
    """
    tmp_params = deepcopy(params)

    # replace any underscore by a white space
    # important for generating DIRAC input
    tmp_params = _replace_underscores(tmp_params)

    out = []
    out += [_write_block(*item) for item in tmp_params.items()]
    out.append('*END OF INPUT')

    with open(input_filename, 'w') as file:
        file.write('\n'.join(out))


_scf_energy_re = re.compile(r"Total energy\s+:\s+([-\d.]+)")
_mp_energy_re = re.compile(r"@ Total MP2 energy\s+:\s+([-\d.]+)")
_cc_energy_re = re.compile(r"@ Total CCSD energy\s+:\s+([-\d.]+)")
_cct_energy_re = re.compile(r"@ Total CCSD(T) energy\s+:\s+([-\d.]+)")
_ci_energy_re = re.compile(r"@ CI Total Energy\s+:\s+([-\d.]+)")


def read_dirac_out(output_filename: str) -> Dict[str, Any]:
    """Reads DIRAC output data from a file.

    Args:
        output_filename: The DIRAC output file.

    Returns:
        A dictionary containing the extracted data from the DIRAC output.
    """
    energy = None

    with open(output_filename, 'r') as file:
        content = file.read()  # Read the entire content of the file

    ehfmatch = re.search(_scf_energy_re, content)
    if ehfmatch is not None:
        energy = float(ehfmatch.group(1)) * Ha

    empmatch = re.search(_mp_energy_re, content)
    if empmatch is not None:
        energy = float(empmatch.group(1)) * Ha

    eccmatch = re.search(_cc_energy_re, content)
    if eccmatch is not None:
        energy = float(eccmatch.group(1)) * Ha

    ecctmatch = re.search(_cct_energy_re, content)
    if ecctmatch is not None:
        energy = float(ecctmatch.group(1)) * Ha

    ecimatch = re.search(_ci_energy_re, content)
    if ecimatch is not None:
        energy = float(ecimatch.group(1)) * Ha

    output = {'energy': energy}

    return output
