Running stand-alone qc2-ASE calculations
========================================

.. note::

    You do not plan to use qc2-ASE calculators alone, you not need to
    worry about all the details provided in this section.
    When using qc2, most of the qc2-ASE features mentioned below are
    naturally abstracted within :class:`qc2.data.data.qc2Data`; see :ref:`qc2data_section`.

Running basic calculations
--------------------------

The first step to run qc2-ASE calculations is to
import the necessary ``ase`` and :mod:`qc2.ase` packages:

.. code-block:: python

    from ase import Atoms
    from qc2.ase import Psi4

where we import the implemented :class:`qc2.ase.psi4.Psi4` qc2-ASE calculator as an example.
To carry out a basic `single-point` energy calculation with it, use:

.. code-block:: python
    :linenos:

    from ase import Atoms
    from ase.units import Ha
    from qc2.ase import Psi4

    # set target molecule using ASE `Atoms` class
    mol = Atoms(
        'H2',
        positions=[
            [0, 0, 0],
            [0, 0, 0.737166]]
    )

    # attach a qchem calculator to `Atoms` object
    mol.calc = Psi4(method='hf', basis='sto-3g')

    # run qchem calculation and print energy in a.u.
    energy = mol.get_potential_energy()/Ha
    print(f"* Single-point energy (Hartree): {energy}")

As you can see, qc2-ASE calculators are executed in the same manner
as traditional `ASE calculators <https://wiki.fysik.dtu.dk/ase/>`_.


Saving qchem data into formatted data files
-------------------------------------------

As mentioned in :ref:`about_section`, the key distinction between qc2-ASE and traditional ASE calculators is
that the former are specifically designed to accommodate extra methods
for easy integration with quantum computing libraries. One of such methods is ``save()`` which
is particularly useful for retrieving and dumping qchem data into formatted data files.

Here is an example of how you can save quantum chemistry data to a formatted `hdf5 <https://portal.hdfgroup.org/hdf5/develop/_u_g.html>`_ file according to
`QCSchema <https://molssi.org/software/qcschema-2/>`_:

.. code-block:: python
    :linenos:
    :emphasize-lines: 11, 18

    from ase.build import molecule
    from ase.units import Ha
    from qc2.ase import PySCF

    # set target molecule using G2 molecule dataset
    mol = molecule('H2O')

    # attach a qchem calculator
    mol.calc = PySCF(method='scf.HF', basis='sto-3g')
    # define format in which to save the qchem data
    mol.calc.schema_format = 'qcschema'

    # perform qchem calculation
    energy = mol.get_potential_energy()/Ha
    print(f"* Single-point energy (Hartree): {energy}")

    # save qchem data to a file
    mol.calc.save('h2o.hdf5')

where the ``schema_format`` attribute of the qc2-ASE calculator is used to set the format in
which to save the data via the ``save()`` method.

If you wish to save data using FCIDump :cite:p:`FCIDump:1989` format, use:

.. code-block:: python
    :linenos:
    :emphasize-lines: 11, 18

    from ase.build import molecule
    from ase.units import Ha
    from qc2.ase import PySCF

    # set target molecule using G2 molecule dataset
    mol = molecule('H2O')

    # attach a qchem calculator
    mol.calc = PySCF(method='scf.HF', basis='sto-3g')
    # define format in which to save the qchem data
    mol.calc.schema_format = 'fcidump'

    # perform qchem calculation
    energy = mol.get_potential_energy()/Ha
    print(f"* Single-point energy (Hartree): {energy}")

    # save qchem data to a file
    mol.calc.save('h2o.fcidump')


Loading qchem data from formatted data files
--------------------------------------------

In addition to the ``save()`` method, qc2-ASE calculators are also equipped with a ``load()`` method.
Its primary function is to read data from qcschema- or fcidump-formatted data files
and store them in ``FCIdump`` and ``QCSchema``
dataclasses; see `Qiskit Nature documentation <https://qiskit.org/ecosystem/nature/apidocs/qiskit_nature.second_q.formats.html>`_.


So, if you have done a quantum chemistry calculation
in the past and have already a formatted data file, *e.g.*,  ``h2o.fcidump``, containing qchem info
you can read data from this file and save it into an instance of ``FCIdump`` dataclass:

.. code-block:: python
    :linenos:
    :emphasize-lines: 8, 10, 13

    from ase.build import molecule
    from qc2.ase import BaseQc2ASECalculator

    # set target molecule
    mol = molecule('H2O')

    # attach a generic qchem calculator
    mol.calc = BaseQc2ASECalculator()
    # set the reading format
    mol.calc.schema_format = "fcidump"

    # load qchem data into a instance of `FCIDump` dataclass
    fcidump = mol.calc.load('h2o.fcidump')

Note that a `dummy` (generic) calculator has been attached to the ASE ``Atoms`` object (``mol``).
The importance of :class:`qc2.ase.BaseQc2ASECalculator` will be emphasized in :ref:`build_ASEs`.

.. note::

    Instances of ``FCIdump`` and ``QCSchema`` dataclasses generated by the ``load()`` method
    have no direct use within qc2-ASE calculators alone. However, they play a crucial role in communication
    with :class:`qc2.data.data.qc2Data` and, subsequently, with quantum computing libraries.
