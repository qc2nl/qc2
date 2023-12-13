.. _run_ase_with_qc2Data:

Running qc2-ASE calculators via qc2Data class
=============================================

One of the key features of :class:`~qc2.data.data.qc2Data` is its ability to run qc2-ASE calculators on-the-fly.
This is achieved by invoking its :meth:`~qc2.data.data.qc2Data.run` method.

An example is given below:

.. code-block:: python
    :linenos:
    :emphasize-lines: 16, 19

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

    # attach a calculator to the molecule attribute
    qc2data.molecule.calc = PySCF()

    # run qc2-ASE calculator
    qc2data.run()

Please note that, before invoking the :meth:`~qc2.data.data.qc2Data.run` method, it is necessary to attach a qc2-ASE calculator to ``qc2data.molecule``
via the ASE ``calc`` attribute. As described in :ref:`run_ase`, we can attach any implemented calculator.
:meth:`~qc2.data.data.qc2Data.run` will then execute the calculator and automatically save the relevant qchem data into ``h2.hdf5``.

.. important::

   If you plan to use qc2 with the qc2-ASE ROSE calculator, instantiate :class:`~qc2.data.data.qc2Data`
   with an empty ``molecule`` argument or set ``molecule = Atoms()``. See the example code below:

    .. code-block:: python
        :linenos:
        :emphasize-lines: 11-12

        from qc2.ase import ROSE, ROSETargetMolecule, ROSEFragment
        from qc2.data import qc2Data

        # define ROSE target molecule and fragments
        molecule = ROSETargetMolecule(...)
        frag1 = ROSEFragment(...)
        fragn = ROSEFragment(...)

        # instantiate qc2Data - no Atoms() needed
        qc2data = qc2Data(
            filename='ibo.fcidump',
            schema='fcidump'
        )

        # attach ROSE calculator to an empty Atoms()
        qc2data.molecule.calc = ROSE(
            rose_calc_type='atom_frag',
            rose_target=molecule,
            rose_frags=[frag1, ..., fragn],
            rose_mo_calculator='pyscf'
        )

        # run ROSE calculator
        qc2data.run()
