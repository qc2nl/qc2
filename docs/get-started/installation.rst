Installation instructions
=========================

Once a local copy of the qc2 repository has been obtained, qc2 can be installed via `pip`.

.. note::

    Before installation, you may want to create a local conda environment to accommodate all qc2 dependencies. To do so,
    follow the steps:

    .. code-block:: console

        conda create -n qc2 python=3.11 # python version optional
        conda activate qc2

To install qc2 in an editable/develop mode:

.. code-block:: console

    cd qc2
    python3 -m pip install -e .

In its current version, qc2 is compatible with
both `Qiskit Nature <https://qiskit.org/ecosystem/nature/>`_ and `Pennylane <https://pennylane.ai/>`_.
However, the latter is an optional dependency. To install Pennylane alongside qc2
and perform all built-in Pennylane-based automatic testings,
follow these steps:

.. code-block:: console

    cd qc2
    python3 -m pip install -e .[pennylane]


.. note::

    If you are using Mac zsh shell, use instead:

    .. code-block:: console

        cd qc2
        python3 -m pip install -e ".[pennylane]"


If you want to test your installation and run qc2's suite of automatic testings,
run `pytest` while in the main qc2 directory, *e.g.*,

.. code-block:: console

    pytest -v


Another option, particularly suitable for those interested in contributing to qc2,
is to include the `dev` option in your installation, as follows:

.. code-block:: console

    cd qc2
    python3 -m pip install -e .[pennylane,dev]

This will install a set of additional packages such as ``isort`` and ``sphinx``,
enabling users to contribute to the project following best practices.

Note on ASE calculators
-----------------------

The automatic testing by `pytest` attempts to run tests for all supported quantum chemistry programs via
their corresponding qc2-ASE calculators. These tests will of course only run if you have preinstalled
these qchem codes on
your local machine/workstation; please
consult the documentation of each supported quantum chemistry program for the best install procedure.

Examples on how to use all supported qc2-ASE calculators and quantum computing libraries are provided
in the ``examples`` directory.
