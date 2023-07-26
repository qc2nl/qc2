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
    from pennylane.operation import active_new_opmath, Tensor
    from pennylane.ops import Hamiltonian, Prod, SProd, Sum
    from pennylane.qchem.convert import _process_wires
    from pennylane.qchem.convert import _openfermion_to_pennylane
    from pennylane.wires import Wires

except ImportError as Error:
    raise ImportError(
        "This feature requires Pennylane. "
        "It can be installed with: pip install pennylane."
        ) from Error


def _qiskit_nature_to_pennylane(qubit_operator, wires=None):
    """Convert Qiskit SparsePauliOp to 2-tuple of coeffs and PennyLane Paulis.

    This functions is usefull to convert fermionic-to-qubit transformed
    qchem operators from Qiskit-Nature to Pennynale format.

    Args:
        qubit_operator (qiskit.quantum_info.SparsePauliOp): Qiskit operator
        representing the qubit electronic Hamiltonian from Qiskit-Nature.
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
    >>> _qiskit_nature_to_pennylane(qubit_op, wires=['w0', 'w1', 'w2', 'w3', 'w4'])
    (tensor([1., 2.], requires_grad=False),
    [PauliX(wires=['w2']) @ PauliZ(wires=['w3']),
    PauliY(wires=['w0']) @ PauliY(wires=['w4'])])

    If the new op-math is active, the list of operators will be cast as
    :class:`~.Prod` instances instead of :class:`~.Tensor` instances when
    appropriate.
    """
    n_wires = qubit_operator.num_qubits
    wires = _process_wires(wires, n_wires=n_wires)
    wire_map = {wires[x]: y for x, y in zip(range(n_wires), range(n_wires))}

    if qubit_operator.coeffs.size == 0:
        return np.array([0.0]), [qml.Identity(wires[0])]

    def _get_op(term, wires):
        """A function to translate Qiskit to Pennylane Pauli terms."""
        if len(term) == n_wires:

            # the Pauli term '...XYZ' in Qiskit is equivalent to [Z0 Y1 X2 ..]
            # in Pennylane. So, invert the string..
            term = term[::-1]

            # wires in Qiskit-Nature are grouped by separated alpha and beta
            # blocks, e.g., for H2 the circuit is represented by:
            #      ┌───┐
            # q_0: ┤ X ├
            #      └───┘
            # q_1: ─────
            #      ┌───┐
            # q_2: ┤ X ├
            #      └───┘
            # q_3: ─────
            #
            # However, in Pennylane they are represented by alpha-beta
            # sequences. So, organize the term accordingly...
            n = len(term)//2
            term = ''.join([term[i::n] for i in range(n)])
            # this could also be done by using the `_process_wires` function.

            if active_new_opmath():
                return qml.prod(
                    qml.pauli.string_to_pauli_word(term, wire_map=wire_map))

            return qml.pauli.string_to_pauli_word(term, wire_map=wire_map)

        return qml.Identity(wires[0])

    coeffs, ops = zip(
        *[(coef, _get_op(term, wires))
          for term, coef in qubit_operator.to_list()]
    )

    return np.real(np.array(coeffs, requires_grad=False)), list(ops)


def _pennylane_to_qiskit_nature(coeffs, ops, wires=None):
    """XXX"""
    try:
        from qiskit.quantum_info import SparsePauliOp
    except ImportError as Error:
        raise ImportError(
            "This feature requires qiskit. "
            "It can be installed with: pip install qiskit"
        ) from Error

    all_wires = Wires.all_wires([op.wires for op in ops], sort=True)

    if wires is not None:
        qubit_indexed_wires = _process_wires(
            wires,
        )
        if not set(all_wires).issubset(set(qubit_indexed_wires)):
            raise ValueError("Supplied `wires` does not cover all wires"
                             " defined in `ops`.")
    else:
        qubit_indexed_wires = all_wires

    n_wires = len(qubit_indexed_wires)
    wire_map = {qubit_indexed_wires[x]: y for x, y in zip(range(n_wires), range(n_wires))}
    print(n_wires, qubit_indexed_wires, wire_map)

    q_op_list = []
    for coeff, op in zip(coeffs, ops):
        if isinstance(op, (Tensor, Prod, SProd, Hamiltonian)):
            string = qml.pauli.pauli_word_to_string(op, wire_map=wire_map)
            string = string[::-1]
            n = len(string)//2
            string = ''.join([string[i::n] for i in range(n)])
            q_op_list.append((string, coeff.unwrap()))
        if isinstance(op, Sum):
            #string = qml.pauli.pauli_word_to_string(op, wire_map=wire_map)
            print(coeff, op.simplify(), op._build_pauli_rep(), op.terms())
            #raise ValueError(
            #        f"Expected a Pennylane operator with a valid Pauli word "
            #        f"representation, but got {op}.")

    return SparsePauliOp.from_list(q_op_list)


def _qiskit_nature_pennylane_equivalent(
    qiskit_qubit_operator, pennylane_qubit_operator, wires=None
):
    """Check"""
    print(type(pennylane_qubit_operator))
    coeffs, ops = pennylane_qubit_operator.terms()
    return qiskit_qubit_operator == _pennylane_to_qiskit_nature(
        coeffs, ops, wires=wires)


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
    (-0.0548*(PauliX(wires=[0]) @ PauliX(wires=[1]) @ PauliY(wires=[2]) @
    PauliY(wires=[3]))) + (0.14297*(PauliZ(wires=[0]) @ PauliZ(wires=[1])))
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
            return qml.dot(*_qiskit_nature_to_pennylane(
                qubit_observable, wires=wires))

        return qml.Hamiltonian(*_qiskit_nature_to_pennylane(
            qubit_observable, wires=wires))
