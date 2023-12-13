.. _init_qc2Data:

Instantiating qc2Data class
===========================

We start with a very simple example:

.. code-block:: python
    :linenos:
    :emphasize-lines: 10-12

    from ase.build import molecule
    from qc2.ase import PySCF
    from qc2.data import qc2Data

    # set ASE Atoms object
    mol = molecule('H2')

    # instantiate qc2Data class
    qc2data = qc2Data(
        molecule=mol,
        filename='h2.hdf5',
        schema='qcschema'
    )

where the first argument ``molecule`` represents an instance of `ASE Atoms <https://wiki.fysik.dtu.dk/ase/ase/atoms.html#module-ase.atoms>`_.
If this attribute is not provided, ``molecule`` will default to an empty ``Atoms()`` object.

In the code snippet above, ``schema`` and ``filename`` are used to specify the format and name of the file in which qchem data is saved.
The available options for ``schema`` are ``qcschema`` and ``fcidump``, which determine whether the filename should have the extensions `*.hdf5` (or `*.h5`) and `*.fcidump`, respectively.
Here is an example using ``fcidump`` format:

.. code-block:: python
    :linenos:
    :emphasize-lines: 10-12

    from ase.build import molecule
    from qc2.ase import PySCF
    from qc2.data import qc2Data

    # instantiate ASE Atoms object
    mol = molecule('H2')

    # now use fcidump format
    qc2data = qc2Data(
        molecule=mol,
        filename='h2.fcidump',
        schema='fcidump'
    )
