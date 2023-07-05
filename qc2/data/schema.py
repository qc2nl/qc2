"""This module defines funcs to create empty HDF5 files using QCSchema."""
from typing import Dict, Any
import json
import h5py
from h5py._hl.attrs import AttributeManager


def generate_empty_h5(schema_file: str, file_path: str) -> None:
    """
    Creates an empty HDF5 file based on a given schema file.

    Args:
        schema_file (str): Path to the schema file.
        file_path (str): Path to the output HDF5 file.
    """
    with open(schema_file, 'r', encoding='UTF-8') as file:
        schema = json.load(file)

    with h5py.File(file_path, 'w') as file:
        create_attributes(schema, file)


def create_attributes(schema: Dict[str, Any], group: AttributeManager) -> None:
    """
    Creates attributes in the given HDF5 group based on the provided schema.

    Args:
        schema (Dict[str, Any]): The schema specifying the attributes.
        group (AttributeManager): The HDF5 group to create attributes in.
    """
    if 'type' in schema and schema['type'] == 'object':
        for prop, prop_schema in schema.get('properties', {}).items():
            if 'type' in prop_schema and prop_schema['type'] == 'object':
                subgroup = group.create_group(prop)
                create_attributes(prop_schema, subgroup)
            elif 'type' in prop_schema and prop_schema['type'] == 'array':
                group.attrs.create(prop, [],
                                   dtype=h5py.special_dtype(vlen=str))
            else:
                group.attrs.create(prop, None, dtype='f')
