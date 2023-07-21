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


# this is the original generate_empty_h5

from h5json import Hdf5db
from h5json.jsontoh5.jsontoh5 import Writeh5


def original_generate_empty_h5(schema: str, h5name: str) -> None:
    """Generate an empty HDF5 file from a JSON schema.

    Args:
        schema (str): Path to the JSON schema file.
        h5name (str): Path to the output HDF5 file.
    """
    # open schema
    text = open(schema).read()

    # parse the json file into a python dictionary
    h5json = json.loads(text)

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
