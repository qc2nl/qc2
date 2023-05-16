"""Input/Output functions for Rose."""
from .rose_dataclass import RoseInputDataClass
from .rose_dataclass import RoseCalcType, RoseIFOVersion, RoseILMOExponent

def write_line(file, line):
    file.write(line + "\n")


def write_section(file, section_name, value=None):
    write_line(file, f".{section_name.upper()}")
    if value is not None:
        write_line(file, str(value))


def write_key_value_pairs(file, section_name, items):
    for item in items:
        write_section(file, section_name)
        write_line(file, str(item[0]))
        write_line(file, str(item[1]))


def write_rose_in(filename: str,
                  inp_data: RoseInputDataClass,
                  **params) -> None:
    
    mol = inp_data.rose_target.calc

    with open(filename, "w") as file:
        write_line(file, "**ROSE")
        if inp_data.version != RoseIFOVersion.STNDRD_2013.value:
            write_section(file, "VERSION", inp_data.version)
        write_section(file, "CHARGE", mol.parameters.charge)
        if inp_data.exponent != RoseILMOExponent.TWO.value:
            write_section(file, "EXPONENT", inp_data.exponent)
        write_section(file, "FILE_FORMAT", mol.name.lower())
        if inp_data.test:
            write_section(file, "TEST")
        if not inp_data.restricted:
            write_section(file, "UNRESTRICTED")
        if not inp_data.spatial_orbitals:
            write_section(file, "SPINORS")
        if inp_data.include_core:
            write_section(file, "INCLUDE_CORE")
        if inp_data.rose_calc_type == RoseCalcType.MOL_FRAG.value:
            write_section(file, "NFRAGMENTS", len(inp_data.rose_frags))
            if inp_data.additional_virtuals_cutoff:
                write_section(file,
                              "ADDITIONAL_VIRTUALS_CUTOFF",
                              inp_data.additional_virtuals_cutoff)
            if inp_data.frag_threshold:
                write_section(file, "FRAG_THRESHOLD", inp_data.frag_threshold)
            if inp_data.frag_valence:
                write_key_value_pairs(file,
                                      "FRAG_VALENCE",
                                      inp_data.frag_valence)
            if inp_data.frag_core:
                write_key_value_pairs(file, "FRAG_CORE", inp_data.frag_core)
            if inp_data.frag_bias:
                write_key_value_pairs(file, "FRAG_BIAS", inp_data.frag_bias)
        write_line(file, "*END OF INPUT")