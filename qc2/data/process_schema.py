#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Lucas Visscher, 2021

Functions to work with the  DIRAC data schema.
For instance to create a checkpoint file from DFCOEF, XYZ and AOPROPER files
"""


###### Functions to  the data schema

def make_schema(definitions,section,rootlabel=''):
    """
    Recursive function to define a schema for a particular section given a complete set of defintions
    
    """
    schema={}   
    for variable, definition in definitions[section].items():
        label = rootlabel + '/' + variable
        
        # Add label to the flat dictionary with definitions
        definitions['flat_schema'][label] = definition.copy()
        definition['label'] = label

        if definition['type'] == 'composite':
            # we need an extra layer if we have an array of composites, initialize the first element of this to 1
            if definition['rank'] == 'array':
               label = label + '/' + '1'
               definitions['flat_schema'][label] = definition.copy()
               definitions['flat_schema'][label]['rank'] = 'single'
               definition['label'] = label
               definition['rank'] = 'single'
            schema[variable] = make_schema(definitions,variable,label)
        else:
            # Make also a hierarchical dictionary
            schema[variable]  = definition  
            
    return schema

def read_schema(file_name):
    """
    Read plain text schema file and return schema as a dictionary
    The returned schema is a hierarchical dictionary, the second return value is a flattened dictionary with the same information.
    """
    lines = []
    with open(file_name) as f:
        lines = f.readlines()

    # first split the file into separate blocks
    blocks = []
    new_block = []
    in_block = False
    for line in lines:
        if line[0] == "*":
            if "*end" in line:
                blocks.append(new_block)
                in_block = False
            else:
                in_block = True
                new_block = []
        if in_block:
            new_block.append(line.rstrip())
            
    # parse these blocks to get a list of dictionaries with the definitions
    definitions= {}
    definitions['flat_schema'] = {}
    for block in blocks:
        block_name = block[0][1:]
        block_definition = {}
        for line in block[1:]:
           variable,type,rank,use = line.split()[0:4]
           description = line.split('#')[1]
           variable_definition = {}
           variable_definition['type']         = type
           variable_definition['rank']         = rank
           variable_definition['use']          = use
           variable_definition['description']  = description
           block_definition[variable] = variable_definition
        definitions[block_name] = block_definition
        
    # Use these definitions to make the hierarchical schema and define labels
    schema = make_schema(definitions,'schema')
            
    # Return both the schema as well as the flattened dictionary with all labels and their definitions
    return schema, definitions['flat_schema']


def write_schema(file_name,flat_schema):
    """
    Write schema as list of labels and types such that it can be easily read in Fortran
    """
    f = open(file_name,"w")
    f.write('80 10 10 10 60\n') # length of strings that are written
    # Write all group labels first. Make sure to write them in alphabetical order to faciltate the creation of groups later on.
    for label, dataset in sorted(flat_schema.items()):
        if dataset['type'] == 'composite':
             f.write("{:<80}{:<10}{:<10}{:<10}#{:<60}\n".format(label,dataset['type'],dataset['rank'],dataset['use'],dataset['description']))
     # Then write all other labels
    for label, dataset in sorted(flat_schema.items()):
        if dataset['type'] != 'composite':
            f.write("{:<80}{:<10}{:<10}{:<10}#{:<60}\n".format(label,dataset['type'],dataset['rank'],dataset['use'],dataset['description']))
    f.close()

def group_from_label(label):
    """
    Identify the group to which the dataset or subgroup belongs.
    For top groups it just returns label (as these are their own parents)
    """
    separator_index = label.rfind('/')
    if separator_index > 0:
       group = label[:separator_index]
    else:
       group = label
    return group
     
def data_validity(data):
    """
    Check whether the data read or to be written is valid.
    Should be made more rigorous, for now just check whether all required data is present.
    """
    valid = True
    print ('  Checking validity of data with respect to the DIRAC data schema....')
    # Check whether all required data have values
    for label, dataset in data.items():
        if dataset['type'] == 'composite':
           continue  # composite types have no values
        if 'value' in dataset: 
           #todo: check whether the data has the right type
           continue
        if dataset['use'] == 'required':
              # check whether this data set without a value is required 
              # first check whether (grand)parents are actually required
              optional = False
              parent = label
              while group_from_label(parent) != parent:
                  parent = group_from_label(parent) 
                  if data[parent]['use'] == 'optional':
                      optional = True
              if optional:
                  continue
              # this is indeed a required data set, print a warning and invalidate the data
              else:
                  print(' ..missing required dataset {} : {}'.format(label,dataset['description']))
                  valid = False
    return valid

###### Function to read plaintext xyz file

def read_xyz(file_name):
    """
    Read plain text xyz file and return the data
    """
    from periodic_table import get_element_number
    lines = []
    with open(file_name) as f:
        lines = f.readlines()

    # first line has the number of atoms
    n_atoms = [int(lines[0].rstrip())]
    symbols = []
    coordinates = []
    nuc_charges = []

    for count, line in enumerate(lines[2:]):
        symbol, x, y, z = line.rstrip().split()[0:4]
        # List of strings are tricky in h5py, simplest is to convert them to numpy arrays,
        # this is done below, after first processing them to get nuclear charges.
        symbols.append(symbol.encode('utf-8'))
        coordinates.append(float(x))
        coordinates.append(float(y))
        coordinates.append(float(z))
        nuc_charges.append(float(get_element_number(symbol.title())))
        if count+1 == n_atoms:
           break

    import numpy as np
    symbols = np.array(symbols)

    return n_atoms, symbols, coordinates, nuc_charges

###### Functions to fill the data object with information from DIRAC files
###### Note that all labels are hardwired in this part !
    
def load_XYZ(file_name,data):
    """
    Open XYZ-type file write contents to the flat data dictionary

    """
    # Read xyz file for the geometry
    n_atoms, symbols, coordinates, nuc_charges = read_xyz(file_name)
    # Fill in the data for the molecular topology
    data['/input/molecule/n_atoms']['value']       = n_atoms
    data['/input/molecule/symbols']['value']       = symbols
    data['/input/molecule/geometry']['value']      = coordinates
    data['/input/molecule/nuc_charge']['value']    = nuc_charges

    
def load_DFCOEF(file_name,data):
    """
    Open DFCOEF-type file write contents to the flat data dictionary

    """   
    from dirac_data import read_DFCOEF
    # Read DFCOEF (could have either 32 or 64-bit integers, try both)
    try:
        dfcoef = read_DFCOEF(file_name)
    except:
        dfcoef = read_DFCOEF(file_name,dtype_int='<i8')
    # Fill in the data for the aobasis
    data['/input/aobasis/1/aobasis_id']['value']   = [1] # DFCOEF contains only one basis set definition
    data['/input/aobasis/1/n_shells']['value']     = dfcoef['ao_nshells']
    data['/input/aobasis/1/n_ao']['value']         = dfcoef['ao_nbas']
    data['/input/aobasis/1/angular']['value']      = [1] # hardwired in DIRAC
    data['/input/aobasis/1/center']['value']       = dfcoef['ao_cent']
    data['/input/aobasis/1/orbmom']['value']       = dfcoef['ao_orbmom']
    data['/input/aobasis/1/n_prim']['value']       = dfcoef['ao_nprim']
    data['/input/aobasis/1/n_cont']['value']       = dfcoef['ao_numcf']
    data['/input/aobasis/1/exponents']['value']    = dfcoef['ao_priexp']
    data['/input/aobasis/1/contractions']['value'] = dfcoef['ao_priccf']

    # Fill in the data for the execution
    data['/result/execution/status']['value']    = 2 # Calculations was finished, otherwise we would not have this DFCOEF
    data['/result/execution/end_date']['value']  = dfcoef['date']
    data['/result/execution/end_time']['value']  = dfcoef['time']

    # Fill in the data for the scf wave function
    data['/result/wavefunctions/scf/energy']['value']      = dfcoef['energy']
    data['/result/wavefunctions/scf/mobasis/mobasis_id']['value']  = [1] # DFCOEF contains only one set of MOs
    data['/result/wavefunctions/scf/mobasis/nz']['value']          = dfcoef['nz']
    data['/result/wavefunctions/scf/mobasis/n_fsym']['value']      = dfcoef['nfsym']
    data['/result/wavefunctions/scf/mobasis/orbitals']['value']    = dfcoef['mo_coeff']
    data['/result/wavefunctions/scf/mobasis/eigenvalues']['value'] = dfcoef['eigenval']
    data['/result/wavefunctions/scf/mobasis/symmetry']['value']    = dfcoef['ibeig']
    data['/result/wavefunctions/scf/mobasis/n_basis']['value']     = dfcoef['nao']
    data['/result/wavefunctions/scf/mobasis/n_mo']['value']        = dfcoef['nto']
    data['/result/wavefunctions/scf/mobasis/n_po']['value']        = dfcoef['npo']

    # The AO dimension is not readily available on AOproper so we need to set this here as well
    data['/result/operators/ao_matrices/aobasis_dim']['value'] = dfcoef['ao_nbas']


def load_AOPROPER(file_name,data):
    """
    Open AOPROPER-type file write contents to the flat data dictionary

    """
    from dirac_data import read_DFILE
    # Read AOPROPER
    aoproper = read_DFILE(file_name)
    data['/result/operators/ao_matrices/aobasis_id']['value']   = 1
    # Fill in the data for the matrix representations
    for key, matrix in aoproper.items():
        if ('GABAO1XX' in key):
           pass # this is not a matrix representation (has dimension n_shells)
        else:
           # copy the generic dictionary to the specific dictionary used to store the matrix
           data['/result/operators/ao_matrices/'+key] = data['/result/operators/ao_matrices/operator_name'].copy()
           # fill in the values for this representation
           data['/result/operators/ao_matrices/'+key]['value'] = matrix

def load_diracdata(path):
    """ 
    Checks for existence of old-syle data files and loads the data that is inside
    """
    # Get schema (hierarchical dictionary) and flattened form thereof based on the definition found in the build directory
    from os.path import dirname, join
    valid_schema, valid_data = read_schema(join(dirname(__file__),"DIRACschema.txt"))

    # Make a copy of the flattened form of the schema that is filled with data loaded from the different files
    # Note that the load routines will fail in case data of an unknown kind is encountered. This is intentional, the data 
    # that is written should be properly defined in the data schema file.
    data = valid_data.copy()

    import os
    if os.path.isfile(os.path.join(path,'MOLECULE.XYZ')):
      load_XYZ(os.path.join(path,'MOLECULE.XYZ'),data)
    if os.path.isfile(os.path.join(path,'AOPROPER')):
      load_AOPROPER(os.path.join(path,'AOPROPER'),data)
    if os.path.isfile(os.path.join(path,'DFCOEF')):
      load_DFCOEF(os.path.join(path,'DFCOEF'),data)

    return data

def nohdf5_recursively_load_dict_contents_from_group(data,path):
    """
    Browse through the hdf5 emulator directory and fill the flat dictionary
    """
    import os
    try:
       from scipy.io import FortranFile
       import numpy as np
    except:
       print('  scipy is not installed, HDF5 restart file can not be made')
       return 1
    with os.scandir(path) as it:
      for entry in it:
        if not entry.name.startswith('.') and entry.is_file():
          label=os.path.join(path,entry.name).split('CHECKPOINT.noh5')[1]
          f = FortranFile(os.path.join(path,entry.name), 'r')
          # reading of strings written by Fortran is tricky, this appears to work
          record = f.read_record(np.dtype('a4'),np.int64,np.int64)
          data_type=record[0][0].decode("utf-8")
          data_dim1=record[1][0]
          data_dim2=record[2][0]
          # A more general solution is needed for the case in  which a generic label is not defined explicitly in the schema
          # For now just handle the only two exceptions that can currently occur
          if 'ao_matrices' in label:
             # copy the generic dictionary to the specific dictionary used to store this particular property matrix
             data[label] = data['/result/operators/ao_matrices/operator_name'].copy()
          if 'input/aobasis/2' in label:
             # copy the dictionary of the first instance as dictionary for a later instance
             data[label] = data[label.replace('input/aobasis/2','input/aobasis/1')].copy()
          if data_type=='real':
             data[label]['value'] = f.read_reals()
          elif data_type=='int4':
             data[label]['value'] = f.read_ints(dtype=np.int32)
          elif data_type=='int8':
             data[label]['value'] = f.read_ints(dtype=np.int64)
          elif data_type=='str1':
             str_format = 'a{}'.format(data_dim1)
             data[label]['value'] = f.read_record(str_format)
          elif data_type=='strn':
             if (data_dim2==1):
                str_format = 'a{}'.format(data_dim1)
             else:
                str_format = '{}a{}'.format(data_dim2,data_dim1)
             data[label]['value'] = f.read_record(str_format)
          else:
             print('ERROR in processing restart data: unknown data type',data_type)
             return 2
        elif entry.is_dir():
          nohdf5_recursively_load_dict_contents_from_group(data,os.path.join(path,entry.name))
    return 0

def nohdf5_load_data(path):
    """ 
    Checks for existence of CHECKPOINT.noh5 fall back directory and loads the data that is inside
    """
    # Get schema (hierarchical dictionary) and flattened form thereof based on the definition found in the build directory
    from os.path import dirname, join, isdir
    valid_schema, valid_data = read_schema(join(dirname(__file__),"DIRACschema.txt"))

    # Make a copy of the flattened form of the schema that is filled with data loaded from the different files
    # Note that the load routines will fail in case data of an unknown kind is encountered. This is intentional, the data 
    # that is written should be properly defined in the data schema file.
    data = valid_data.copy()

    if isdir(join(path,'CHECKPOINT.noh5')):
      error = nohdf5_recursively_load_dict_contents_from_group(data,join(path,"CHECKPOINT.noh5"))
    else:
      error = 3

    return error, data

###### Functions to read and write hdf5 files

def read_hdf5(file_name):
    """
    Open hdf5-type file and return dictionary of its contents
    """
    import h5py
    data_dict = {}
    with h5py.File(file_name, 'r') as h5file:
        recursively_load_dict_contents_from_group(h5file, data_dict,'/')
    return data_dict
 
def recursively_load_dict_contents_from_group(h5file, data_dict, path):
    """
    Modified from code found at Stack Exchange to get flat dictionary
    """
    import h5py
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            data_dict[path+key] = {}
            data_dict[path+key]['value'] = item[()]
        elif isinstance(item, h5py._hl.group.Group):
           recursively_load_dict_contents_from_group(h5file, data_dict, path + key + '/')
    return

def write_hdf5(file_name,data):
    """
    Open hdf5-type file and write data dictionary
    """
    import h5py
    cp = h5py.File(file_name, 'w')
    for label, dataset in data.items():
        if 'value' in dataset: 
           # Arrays should be written as extendable datasets, single strings or scalars can be fixed size
           if hasattr(dataset['value'],'__len__'):
               if (len(dataset['value']) == 1 or isinstance(dataset['value'], str)):
                   cp.create_dataset(label, data=[dataset['value']])
               else:
                   cp.create_dataset(label, data=dataset['value'], maxshape=(h5py.h5s.UNLIMITED))
           else:
               cp.create_dataset(label, data=dataset['value'])
    cp.close()

# Simple use of these functions:
# Step 1: load all data residing in directory pathname into a dictionary called data
# data=load_diracdata(pathname)
# Step 2: write this dictionary to a hdf5 file
# write_hdf5(hdf5_filename,data)
# Step 3: Read the data back in to check
# data_check = read_hdf5(hdf5_filename)

# More control: load schema and files in separate steps

# 1a : Get schema (hierarchical dictionary) and flattened form thereof
# schema, flat_schema = read_schema("DIRACschema.txt")
# Optionally write the schema labels
# write_schema('schema_labels.txt',flat_schema)

# 1b: Make a copy of the flattened form of the schema and fill it with data loaded from the different files
#data = flat_schema.copy()
#load_XYZ('MOLECULE.xyz',data)
#load_AOPROPER('AOPROPER',data)
#load_DFCOEF('DFCOEF',data)

# 2: Write the checkpoint file
#write_hdf5('CHECKPOINT.h5',data)
 
# 3: Read the data back in:
#data_read = read_hdf5('CHECKPOINT.h5')
