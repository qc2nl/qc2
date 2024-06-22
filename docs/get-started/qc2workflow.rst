.. _qc2workflow_section:

qc2 workflow
============

.. _code_workflow:

.. figure:: ../figures/qc2_workflow.drawio.png
    :align: center
    :alt: Code workflow diagram

    Code workflow diagram

qc2 is designed around the :class:`~qc2.data.data.qc2Data`. This class connects with traditional quantum chemistry codes via
custom qc2-ASE calculators and the formatted `QCSchema <https://molssi.org/software/qcschema-2/>`_ or FCIdump :cite:p:`FCIDump:1989`
data files generated from them. :class:`~qc2.data.data.qc2Data` is then able to read these files and create instances of
``FCIdump`` and ``QCSchema``
dataclasses; see `Qiskit Nature documentation <https://qiskit.org/ecosystem/nature/apidocs/qiskit_nature.second_q.formats.html>`_.

.. note::

    Since qc2 primarily operates with data files, you could, in principle,
    use any traditional quantum chemistry backend other than `ASE <https://wiki.fysik.dtu.dk/ase/>`_
    to generate qchem data. This is possible as long as your backend dumps the necessary data in
    accordance with the above-mentioned schema.

With the ``FCIdump`` and ``QCSchema`` dataclasses at hand, :class:`~qc2.data.data.qc2Data` can seamlessly integrate with
`Qiskit Nature <https://qiskit.org/ecosystem/nature/>`_ and build up the active-space molecular Hamiltonian in
qubit representation via :meth:`~qc2.data.data.qc2Data.get_qubit_hamiltonian`.
This function uses Qiskit Nature routines and returns the Hamiltonian in Qiskit format.

.. warning::

    By being naturally compatible with Qiskit Nature, :class:`~qc2.data.data.qc2Data`
    does not yet directly communicate with PennyLane,
    a feature that will be available in the near future.

However, If you want to use
`PennyLane <https://pennylane.ai/>`_, :meth:`~qc2.data.data.qc2Data.get_qubit_hamiltonian`
can internally convert the Qiskit Hamiltonian into Pennylane format by resorting to :func:`~qc2.pennylane.convert._qiskit_nature_to_pennylane`.

Once an appropriate form of the qubit Hamiltonian is obtained,
VQE calculations can be performed using the quantum computing library of choice.
Examples on how to perform such calculations will be later given in the :ref:`tutorial_section`
and can also be found in the ``examples`` directory.
