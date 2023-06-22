from typing import Dict, Any
import os

import json
import h5py
import jsonschema


def generate_empty_h5(schema_file: str, output_file: str) -> None:
    """Create an empty HDF5 file following a JSON QCSchema.

    Args:
        schema_file (str): The path to the JSON schema file.
        output_file (str): The path to the output HDF5 file.

    Returns:
        None
    """
    with open(schema_file, 'r', encoding='UTF-8') as file:
        schema = json.load(file)

    jsonschema.Draft4Validator.check_schema(schema)

    with h5py.File(output_file, 'w') as file:
        create_datasets(schema, file)


def create_datasets(schema: dict, parent_group: h5py.Group) -> None:
    """Create datasets in the HDF5 file based on the JSON QCSchema.

    Args:
        schema (dict): The JSON schema.
        parent_group (h5py.Group): The parent HDF5 group where datasets will be created.

    Returns:
        None
    """
    for property_name, property_schema in schema.get('properties', {}).items():
        data_type = property_schema.get('type')
        if data_type == 'array':
            item_type = property_schema.get('items', {}).get('type')
            if item_type == 'integer':
                parent_group.create_dataset(
                    property_name, shape=(0,), dtype='i'
                    )
            elif item_type == 'string':
                parent_group.create_dataset(
                    property_name, shape=(0,),
                    dtype=h5py.string_dtype(encoding='utf-8')
                    )
        elif data_type == 'string':
            parent_group.create_dataset(
                property_name, shape=(0,),
                dtype=h5py.string_dtype(encoding='utf-8')
                )
        # Add support for other data types as needed

        # Recurse into nested objects if present
        if 'properties' in property_schema:
            subgroup = parent_group.create_group(property_name)
            create_datasets(property_schema, subgroup)

