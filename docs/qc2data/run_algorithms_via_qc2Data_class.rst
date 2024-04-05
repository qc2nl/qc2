.. _run_algorithms_with_qc2Data:

Running quantum-classical algorithms via qc2Data class
======================================================

As key component of qc2, the :class:`~qc2.data.data.qc2Data` class is designed
not only to smoothly connect with qc2-ASE calculators (see :ref:`run_ase_with_qc2Data`) but also to seamlessly
integrate with its built-in package of native of :ref:`algorithms_section`. This is precisely the goal of qc2,
and it is what users should expect in actual qc2 runs.

This connection is facilitated through the :class:`~qc2.data.data.qc2Data`'s :attr:`~qc2.data.data.qc2Data.algorithm` attribute.
Below, we present examples of complete hybrid quantum-classical runs using qc2.
For more detailed guidance, users are referred to the :ref:`tutorial_section` section and ``examples`` directory.

In the first code snippet, we show how to perform a full VQE calculaton for hydrogen molecule.
This uses ``fcidump`` as qchem data schema and
Qiskit-Nature with the ``SLSQP`` optimizer, ``qiskit.Estimator`` and ``Bravyi-Kitaev`` mapper.

.. code-block:: python
    :linenos:
    :emphasize-lines: 28, 39

    from ase.build import molecule

    from qiskit_algorithms.optimizers import SLSQP
    from qiskit.primitives import Estimator

    from qc2.data import qc2Data
    from qc2.ase import PySCF
    from qc2.algorithms.qiskit import VQE
    from qc2.algorithms.utils import ActiveSpace

    # set ASE Atoms object
    mol = molecule('H2')

    # attach a qc2-ASE calculator to Atoms object
    mol.calc = PySCF()  # => defaults to HF/sto-3g

    # instantiate qc2Data class
    qc2data = qc2Data(
        molecule=mol,
        filename='h2.fcidump',
        schema='fcidump'
    )

    # run qc2-ASE calculator
    qc2data.run()

    # instantiate VQE algorithm class
    qc2data.algorithm = VQE(
        active_space=ActiveSpace(
            num_active_electrons=(1, 1),
            num_active_spatial_orbitals=2
        ),
        mapper="bk",
        optimizer=SLSQP(),
        estimator=Estimator(),
    )

    # run vqe
    results = qc2data.algorithm.run()

The second example shows a oo-VQE run for water and PennyLane.
This now uses ``qcschema`` as data format with the qc2-ASE ``Psi4`` calculator.
Extra options for PennyLane's ``device`` and ``QNode`` are also added.

.. code-block:: python
    :linenos:
    :emphasize-lines: 27, 38-41

    from ase.build import molecule

    import pennylane as qml

    from qc2.data import qc2Data
    from qc2.ase import Psi4
    from qc2.algorithms.pennylane import oo_VQE
    from qc2.algorithms.utils import ActiveSpace

    # set ASE Atoms object
    mol = molecule('H2O')

    # instantiate qc2Data class
    qc2data = qc2Data(
        molecule=mol,
        filename='h2o.hdf5',
        schema='qcschema'
    )

    # one can also attach qc2-ASE calculator later on to the molecule attribute
    qc2data.molecule.calc = Psi4(method="hf", basis="sto-3g")

    # run qc2-ASE calculator
    qc2data.run() 

    # instantiate oo-VQE class
    qc2data.algorithm = oo_VQE(
        active_space=ActiveSpace(
            num_active_electrons=(2, 2),
            num_active_spatial_orbitals=4
        ),
        mapper='jw',
        optimizer=qml.GradientDescentOptimizer(stepsize=0.5),
        device='default.qubit'
    )

    # run oo-VQE algorithm with special device and QNode options if needed
    results = qc2data.algorithm.run(
        device_kwargs={"shots": None},
        qnode_kwargs={"diff_method": "best"}
    )

where ``results`` in both cases correspond to instances of :class:`~qc2.algorithms.algorithms_results.OOVQEResults`
and :class:`~qc2.algorithms.algorithms_results.VQEResults` classes, respectively.