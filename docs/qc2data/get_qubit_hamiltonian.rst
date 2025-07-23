.. _get_qubit_hamiltonian:

Building up the molecular Hamiltonian
=====================================

.. note::

    If you aim to take full advantage of qc2 and utilize its suite of algorithm classes,
    you do not need to worry about all the specifics mentioned in this section.
    When instantiating :class:`~qc2.algorithms.base.vqe_base.VQEBASE` and its child classes,
    most of the :class:`~qc2.qc2_driver.QC2`` methods discussed below are executed automatically
    through :meth:`~qc2.algorithms.base.vqe_base.VQEBASE._init_qubit_hamiltonian`

In addition to :meth:`~qc2.qc2_driver.QC2.run`, :class:`~qc2.qc2_driver.QC2`
provides a suite of methods capable of directly reading
from the `QCSchema <https://molssi.org/software/qcschema-2/>`_ or FCIdump data files
and use such info to construct the molecular qubit Hamiltonian. These are :meth:`~qc2.qc2_driver.QC2.get_transformed_hamiltonian`, :meth:`~qc2.qc2_driver.QC2.get_active_space_hamiltonian`,
:meth:`~qc2.qc2_driver.QC2.get_fermionic_hamiltonian` and :meth:`~qc2.qc2_driver.QC2.get_qubit_hamiltonian`.

Of particular relevance is the :meth:`~qc2.qc2_driver.QC2.get_qubit_hamiltonian` method.
Internally, this method takes the active-space electronic Hamiltonian in second quantization,
which is constructed by :meth:`~qc2.qc2_driver.QC2.get_fermionic_hamiltonian`,
and applies an appropriate fermion-to-qubit mapping to it,
such as the Jordan-Wigner or Bravyi-Kitaev transformations :cite:p:`REV_VQE:2022`.

Below, you'll find an example of how to set up and build the molecular Hamiltonian for H\ :sub:`2`
from a `QCSchema <https://molssi.org/software/qcschema-2/>`_ formatted hdf5 file obtained using qc2-ASE :class:`~qc2.ase.dirac.DIRAC`.

.. code-block:: python
    :linenos:
    :emphasize-lines: 24-27

    from ase.build import molecule
    from qc2.qubit_mapper.qiskit import JordanWigner
    from qc2.ase import DIRAC
    from qc2.qc2_driver import QC2 as qc2Data

    # set ASE Atoms object
    mol = molecule('H2')

    # instantiate qc2Data class
    qc2data = qc2Data(
        molecule=mol,
        filename='h2.hdf5',
        schema='qcschema'
    )

    # attach a DIRAC qc2-ASE calculator
    qc2data.molecule.calc = DIRAC()

    # run calculator
    qc2data.run()

    # set up qubit Hamiltonian and core energy based on given activate space
    e_core, qubit_op = qc2data.get_qubit_hamiltonian(
        num_electrons=(1, 1),
        num_spatial_orbitals=2,
        mapper=JordanWigner(),
        format='qiskit'
    )

Here, ``qubit_op`` is a Qiskit-formatted ``SparsePauliOp`` operator, which can be directly used in subsequent hybrid classical-quantum calculations
with `Qiskit Nature <https://qiskit.org/ecosystem/nature/>`_. 
