"""Input/Output functions for Rose."""
from typing import Optional, List, Any
import copy
import numpy as np

from .rose_dataclass import RoseInputDataClass
from .rose_dataclass import RoseCalcType, RoseIFOVersion, RoseILMOExponent

def _write_line(file, line: str) -> None:
    """
    Writes a line of text to a file.

    Args:
        file (file object): The file object to write to.
        line (str): The line of text to write.
    """
    file.write(line + "\n")


def _write_section(file, section_name: str, value: Optional[Any] = None) -> None:
    """
    Writes a section name and an optional value to a file.

    Args:
        file (file object): The file object to write to.
        section_name (str): The name of the section.
        value (optional[any]): The value associated with the section (default: None).
    """
    _write_line(file, f".{section_name.upper()}")
    if value is not None:
        _write_line(file, str(value))


def _write_key_value_pairs(file, section_name: str, items: List[List[int]]) -> None:
    """
    Writes a section name and a list of key-value pairs to a file.

    Args:
        file (file object): The file object to write to.
        section_name (str): The name of the section.
        items (list[any]): A list of key-value pairs.
    """
    for item in items:
        _write_section(file, section_name)
        _write_line(file, str(item[0]))
        _write_line(file, str(item[1]))


def write_rose_in(filename: str,
                  inp_data: RoseInputDataClass,
                  **params) -> None:
    """
    Writes ROSE input data to a file.

    Args:
        filename (str): The name of the file to write the input data to.
        inp_data (RoseInputDataClass): The input data to write.
        **params: Additional parameters.

    Returns:
        None
    """
    mol = inp_data.rose_target.calc

    with open(filename, "w") as file:
        _write_line(file, "**ROSE")
        if inp_data.version != RoseIFOVersion.STNDRD_2013.value:
            _write_section(file, "VERSION", inp_data.version)
        _write_section(file, "CHARGE", mol.parameters.charge)
        if inp_data.exponent != RoseILMOExponent.TWO.value:
            _write_section(file, "EXPONENT", inp_data.exponent)
        _write_section(file, "FILE_FORMAT", mol.name.lower())
        if inp_data.test:
            _write_section(file, "TEST")
        if not inp_data.restricted:
            _write_section(file, "UNRESTRICTED")
        if not inp_data.spatial_orbitals:
            _write_section(file, "SPINORS")
        if inp_data.include_core:
            _write_section(file, "INCLUDE_CORE")
        if inp_data.rose_calc_type == RoseCalcType.MOL_FRAG.value:
            _write_section(file, "NFRAGMENTS", len(inp_data.rose_frags))
            if inp_data.additional_virtuals_cutoff:
                _write_section(file,
                              "ADDITIONAL_VIRTUALS_CUTOFF",
                              inp_data.additional_virtuals_cutoff)
            if inp_data.frag_threshold:
                _write_section(file, "FRAG_THRESHOLD", inp_data.frag_threshold)
            if inp_data.frag_valence:
                _write_key_value_pairs(file,
                                      "FRAG_VALENCE",
                                      inp_data.frag_valence)
            if inp_data.frag_core:
                _write_key_value_pairs(file, "FRAG_CORE", inp_data.frag_core)
            if inp_data.frag_bias:
                _write_key_value_pairs(file, "FRAG_BIAS", inp_data.frag_bias)
        _write_line(file, "*END OF INPUT")
