import json
import h5py
from h5json import Hdf5db
from h5json.jsontoh5 import Writeh5

def generate_empty_h5(schema: str, h5name: str) -> None:
    """generate an empty hdf5 file from a json schema

    Args:
        schema (str): _description_
        h5name (str): _description_
    """
    # open schema
    text = open(schema).read()

    # parse the json file
    h5json = json.loads(text)

    if "root" not in h5json:
        raise Exception("no root key in input file")
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
