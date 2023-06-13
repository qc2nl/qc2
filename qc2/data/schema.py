from typing import Dict, Any
import os

import json
import h5py
from h5json import Hdf5db
from h5json.jsontoh5.jsontoh5 import Writeh5

from .process_schema import write_hdf5
from .process_schema import read_schema, write_schema

# testing Luuks scheme ##################################################
def generate_empty_h5(schema: str, h5name: str) -> None:
    """Generate an empty HDF5 file from a JSON schema.

    Args:
        schema (str): Path to the txt schema file.
        h5name (str): Path to the output HDF5 file.
    """
    # generate qc2 data schema
    qc2_schema, qc2_flatschema = read_schema(schema)

    # generated labels text => more easily processed by Fortran programs
    write_schema('QC2labels.txt', qc2_flatschema)

    # create a valid dictionary with all denifitions
    qc2_data = qc2_flatschema.copy()

    # create empty file with dummy data
    write_hdf5(h5name, qc2_data)

def generate_dict_for_qc2_schema() -> Dict[str, Any]:
    """_summary_

    Returns:
        Dict[str, Any]: A valid dictionary containg the qc2 data schema.
    """
    schema = os.path.join(
            os.path.dirname(__file__), 'QC2schema.txt')

    qc2_schema, qc2_flatschema = read_schema(schema)

    # create a valid dictionary with all denifitions
    qc2_dict_schema = qc2_flatschema.copy()

    return qc2_dict_schema
########################################################################

# this is the original generate_empty_h5
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
