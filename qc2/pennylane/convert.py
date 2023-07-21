""""
Module containing utils for converting Qiskit operators to Pennylane format.

Notes:
    It representes a major extension of pennylane/qchem/convert.py module.
    See https://github.com/PennyLaneAI/pennylane/blob/master/pennylane/qchem/convert.py

    The implemented conversion may only be valid for Fermionic-to-qubit
    transformed hamiltonians since it accounts for the distinct alpha and beta
    qubit (wire) distribution between VQE anzatses in Qiskit and Pennylane.
"""
try:

    import pennylane as qml
    from pennylane import numpy as np
    from pennylane.operation import active_new_opmath
    from pennylane.qchem.convert import _process_wires
    from pennylane.qchem.convert import _openfermion_to_pennylane

except ImportError as Error:
    raise ImportError(
        "This feature requires Pennylane. "
        "It can be installed with: pip install pennylane."
        ) from Error


def _qiskit_to_pennylane(qubit_operator, wires=None):
    """Convert Qiskit `SparsePauliOp` to a 2-tuple of coeffs and PennyLane Paulis.

    Args:
        qubit_operator (qiskit.quantum_info.SparsePauliOp): Qiskit operator
        representing the qubit electronic Hamiltonian.
        wires (Wires, list, tuple, dict): Custom wire mapping used to convert
            the qubit operator to an observable terms measurable in PennyLane.
            For types Wires/list/tuple, each item in the iterable represents a
            wire label corresponding to the qubit number equal to its index.
            For type dict, only int-keyed dict (for qubit-to-wire conversion)
            is accepted. If None, will use identity map (0->0, 1->1, ...).

    Returns:
        tuple[array[float], Iterable[pennylane.operation.Operator]]: coeffs
        and their corresponding PennyLane observables in the Pauli basis.

    **Example**

    >>> from qiskit.quantum_info import SparsePauliOp
    >>> qubit_op = SparsePauliOp.from_list([("XIIZI", 1), ("IYIIY", 2)])
    >>> qubit_op
    SparsePauliOp(['XIIZI', 'IYIIY'],
              coeffs=[1.+0.j, 2.+0.j])
    >>> _qiskit_to_pennylane(qubit_op, wires=['w0', 'w1', 'w2', 'w3', 'w4'])
    (tensor([1., 2.], requires_grad=False),
    [PauliX(wires=[2]) @ PauliZ(wires=[3]),
    PauliY(wires=[0]) @ PauliY(wires=[4])])
    """
    n_wires = qubit_operator.num_qubits
    wires = _process_wires(None, n_wires=n_wires)

    if qubit_operator.coeffs.size == 0:
        return np.array([0.0]), [qml.Identity(wires[0])]

    def _get_op(term, wires):
        """A function to translate Qiskit to Pennylane Pauli terms."""
        if len(term) == n_wires:

            # The Pauli term '...XYZ' in Qiskit is equivalent to [Z0 Y1 X2 ...] in Pennylane
            # So, invert the string...
            term = term[::-1]

            # Wires in Qiskit are grouped by separated blocks of alpha and beta wires,
            # while in Pennylane they are represented by alpha-beta sequences.
            # So, organize the term accordingly...
            n = len(term)//2
            term = ''.join([term[i::n] for i in range(n)])
            # This could also be done by using the `_process_wires` function.

            #if active_new_opmath():
            #    return qml.prod(qml.pauli.string_to_pauli_word(term))
            
            return qml.pauli.string_to_pauli_word(term)

        return qml.Identity(wires[0])

    coeffs, ops = zip(
        *[(coef, _get_op(term, wires))
          for term, coef in qubit_operator.to_list()]
    )

    return np.real(np.array(coeffs, requires_grad=False)), list(ops)


def import_operator(qubit_observable, format="openfermion",
                    wires=None, tol=1e010):
    """Convert an external operator to a PennyLane operator.

    The external format currently supported is openfermion and qiskit.

    Args:
        qubit_observable: external qubit operator that will be converted
        format (str): the format of the operator object to convert from
        wires (.Wires, list, tuple, dict): Custom wire mapping used to convert
            the external qubit operator to a PennyLane operator.
            For types ``Wires``/list/tuple, each item in the iterable
            represents a wire label for the corresponding qubit index.
            For type dict, only int-keyed dictionaries (for qubit-to-wire
            conversion) are accepted. If ``None``, the identity map
            (0->0, 1->1, ...) will be used.
        tol (float): Tolerance in `machine epsilon
            <https://numpy.org/doc/stable/reference/generated/numpy.real_if_close.html>`
            for the imaginary part of the coefficients in ``qubit_observable``.
            Coefficients with imaginary part less than 2.22e-16*tol are
            considered to be real.

    Returns:
        (.Operator): PennyLane operator representing any operator expressed as
        linear comb of Pauli words, e.g., :math:`\sum_{k=0}^{N-1} c_k O_k`

    Example:
    >>> from openfermion import QubitOperator
    >>> h_of = QubitOperator('X0 X1 Y2 Y3', -0.0548)
    + QubitOperator('Z0 Z1', 0.14297)
    >>> h_pl = import_operator(h_of, format='openfermion')
    >>> print(h_pl)
    (0.14297) [Z0 Z1]
    + (-0.0548) [X0 X1 Y2 Y3]

    >>> from qiskit.quantum_info import SparsePauliOp
    >>> h_qt = SparsePauliOp.from_list([("XXYY", -0.0548), ("ZZII", 0.14297)])
    >>> h_pl = import_operator(h_qt, format='qiskit')
    >>> print(h_pl)
    (0.14297) [Z1 Z3]
    + (-0.0548) [Y0 X1 Y2 Y3]

    If the new op-math is active, an arithmetic operator is returned instead.

    >>> qml.operation.enable_new_opmath()
    >>> h_pl = import_operator(h_of, format='openfermion')
    >>> print(h_pl)
    (-0.0548*(PauliX(wires=[0]) @ PauliX(wires=[1]) @ PauliY(wires=[2]) @ PauliY(wires=[3]))) + (0.14297*(PauliZ(wires=[0]) @ PauliZ(wires=[1])))
    """
    if format not in ["openfermion", "qiskit"]:
        raise TypeError(f"Converter does not exist for {format} format.")

    if format == "openfermion":
        # dealing with openfermion `QubitOperator`
        coeffs = np.array([np.real_if_close(coef, tol=tol)
                           for coef in qubit_observable.terms.values()])
    elif format == "qiskit":
        # dealing with qiskit `SparsePauliOp`
        coeffs = np.array([np.real_if_close(coef, tol=tol)
                           for coef in qubit_observable.coeffs])

    if any(np.iscomplex(coeffs)):
        warnings.warn(
            f"The coefficients entering the QubitOperator"
            f" or SparsePauliOp must be real;"
            f" got complex coefficients in the operator"
            f" {list(coeffs[np.iscomplex(coeffs)])}"
        )

    if format == "openfermion":
        if active_new_opmath():
            return qml.dot(*_openfermion_to_pennylane(
                qubit_observable, wires=wires))

        return qml.Hamiltonian(*_openfermion_to_pennylane(
            qubit_observable, wires=wires))

    if format == "qiskit":
        if active_new_opmath():
            return qml.dot(*_qiskit_to_pennylane(
                qubit_observable, wires=wires))

        return qml.Hamiltonian(*_qiskit_to_pennylane(
            qubit_observable, wires=wires))
