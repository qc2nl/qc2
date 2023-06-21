from typing import Dict, Any
import os

import json
import h5py
import jsonschema
from h5json import Hdf5db
from h5json.jsontoh5.jsontoh5 import Writeh5


def generate_empty_h5(schema_file: str, output_file: str) -> None:
    """Create an empty HDF5 file following a JSON QCSchema.

    Args:
        schema_file (str): The path to the JSON schema file.
        output_file (str): The path to the output HDF5 file.

    Returns:
        None
    """
    with open(schema_file, 'r') as f:
        schema = json.load(f)

    jsonschema.Draft4Validator.check_schema(schema)

    with h5py.File(output_file, 'w') as f:
        create_datasets(schema, f)


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


def old_generate_empty_h5(schema: str, h5name: str) -> None:
    """Generate an empty HDF5 file from a JSON schema.

    Args:
        schema (str): Path to the JSON schema file.
        h5name (str): Path to the output HDF5 file.
    """
    # open schema
    text = open(schema).read()

    # parse the json file into a python dictionary
    h5json = json.loads(text)

    print(h5json)

    if "root" not in h5json:
        raise Exception("No 'root' key in the JSON schema.")
    root_uuid = h5json["root"]

    # create the file, will raise IOError if there's a problem
    Hdf5db.createHDF5File(h5name)

    with Hdf5db(
        h5name, root_uuid=root_uuid, update_timestamps=False, app_logger=None
    ) as db:
        h5writer = Writeh5(db, h5json)
        h5writer.writeFile()

    # open with h5py and remove the _db_ group
    # Note: this will delete any anonymous (un-linked) objects
    f = h5py.File(h5name, "a")
    if "__db__" in f:
        del f["__db__"]
    f.close()
