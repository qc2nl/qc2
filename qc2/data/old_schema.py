from typing import Dict, Any
import os

import json
import h5py
from h5json import Hdf5db
from h5json.jsontoh5.jsontoh5 import Writeh5

from .process_schema import write_hdf5
from .process_schema import read_schema, write_schema


def generate_dict_from_text_schema() -> Dict[str, Any]:
    """Convert plain text schema into a dictionary"""
    file = os.path.join(os.path.dirname(__file__), 'QC2schema.txt')
    # file containg the target schema in plain text format
    # not passed as argument...

    # create a valid dictionary containing all required definitions
    schema = read_schema(file)[0]
    schema_dict = schema.copy()
    return schema_dict


def generate_json_schema_file(filename) -> None:
    """Create JSON qc2 schema and save it to a file.
    
    Args:
        filename (str): file in which to write JSON schema
    """
    # generate dictionary from plain text schema in 'QC2schema.txt'
    schema_dict = generate_dict_from_text_schema()
    
    # convert dictionary to JSON schema
    schema_json = json.dumps(schema_dict, indent=2)

    print(schema_json)

    # save the JSON schema to a file
    #with open("{}".format(filename), "w") as f:
    #    f.write(schema_json)


def create_group(parent_group: h5py.Group, schema: Dict[str, Any], current_path: str) -> None:
    """Recursively create groups in the HDF5 file based on the schema.

    Args:
        parent_group (h5py.Group): The parent group where the new group will be created.
        schema (Dict[str, Any]): The JSON schema defining the structure of the group.
        current_path (str): The current path in the HDF5 file.
    """
    for key, value in schema.items():
        path = current_path + key
        if isinstance(value, dict):  # Subgroup
            group = parent_group.create_group(path)
            group.attrs['type'] = value['type']
            group.attrs['rank'] = value['rank']
            group.attrs['use'] = value['use']
            group.attrs['description'] = value['description']
            create_group(group, value, path + '/')
        else:  # Leaf node
            parent_group.attrs[key] = value


def generate_empty_h5(schema_file_path: str, output_file_path: str) -> None:
    """Create an empty HDF5 file according to the provided JSON schema.

    Args:
        schema_file_path (str): The path to the JSON schema file.
        output_file_path (str): The path to the output HDF5 file.
    """
    with open(schema_file_path) as f:
        schema = json.load(f)
    with h5py.File(output_file_path, 'w') as f:
        create_group(f, schema, '')


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
