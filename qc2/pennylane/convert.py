"""
Module containing utils for converting Qiskit operators to Pennylane format.

Notes:
    It representes a major extension of pennylane/qchem/convert.py module.
    https://github.com/PennyLaneAI/pennylane/blob/master/pennylane/qchem/convert.py

    The implemented functions may only be valid to fermionic-to-qubit
    transformed hamiltonians as it accounts for the distinct alpha and beta
    qubit (wire) distribution between VQE anzatses in Qiskit-Nature and
    Pennylane.
"""
try:

    import warnings
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
    >>> _qiskit_nature_to_pennylane(qubit_op,wires=['w0','w1','w2','w3','w4'])
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
            n = len(term)//2  # => valid for closed shell systems only ?
            term = ''.join([term[i::n] for i in range(n)])
            # this could also be done by using the `_process_wires` function.

            if active_new_opmath():
                return qml.prod(
                    qml.pauli.string_to_pauli_word(term, wire_map=wire_map))

            return Tensor(qml.pauli.string_to_pauli_word(term, wire_map=wire_map))

        return qml.Identity(wires[0])

    coeffs, ops = zip(
        *[(coef, _get_op(term, wires))
          for term, coef in qubit_operator.to_list()]
    )

    return np.array(coeffs).real, list(ops)


def _pennylane_to_qiskit_nature(coeffs, ops, wires):
    """Convert Pennylane to Qiskit-Nature formats.

    Convert a 2-tuple of complex coefficients and PennyLane operations to
    Qiskit ``SparsePauliOp``.

    Args:
        coeffs (array[complex]):
            coefficients for each observable, same length as ops
        ops (Iterable[pennylane.operation.Operations]): list of PennyLane
            operations that have a valid PauliSentence representation.
        wires (Wires, list, tuple, dict): Full wire mapping used to convert
            to qubit operator from an observable terms measurable in a
            PennyLane ansatz. For types Wires/list/tuple, each item in the
            iterable represents a wire label corresponding to the qubit number
            equal to its index. For type dict, only consecutive-int-valued
            dict (for wire-to-qubit conversion) is accepted.

    Returns:
        SparsePauliOp: an instance of Qiskit's ``SparsePauliOp``.

    Notes:
        Instead of having to provide the full list of wires,
        e.g., wires=[0, 1, 2, 3, ..., n_wires],
        this function could be alternativelly implemented using
        ``SparsePauliOp.from_sparse_list()`` in place of
        ``SparsePauliOp.from_list()``,
        e.g.,

        .. code-block:: python

            op = SparsePauliOp.from_sparse_list(
                [("ZX", [1, 4], 1), ("YY", [0, 3], 2)], num_qubits=5)

        with the requirement that users provide the total number of qubits.

    **Example**

    >>> coeffs = np.array([0.1, 0.2, 0.3])
    >>>> ops = [
    ...     qml.operation.Tensor(qml.PauliX(wires=[0])),
    ...     qml.operation.Tensor(qml.PauliY(wires=[0]), qml.PauliZ(wires=[2])),
    ...     qml.prod(qml.PauliX(wires=[0]), qml.PauliZ(wires=[3]))
    ... ]
    >>> _pennylane_to_qiskit_nature(coeffs, ops, wires=[0, 1, 2, 3])
    SparsePauliOp(['IIIX', 'IZIY', 'ZIIX'],
                  coeffs=[0.1+0.j, 0.2+0.j, 0.3+0.j])
    """
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
        raise ValueError(
            "Please, provide the full sequence of wires "
            "so that a complete Pauli term is generated "
            "for Qiskit.")

    n_wires = len(qubit_indexed_wires)
    wire_map = {qubit_indexed_wires[x]: y for x, y in zip(range(n_wires),
                                                          range(n_wires))}

    q_op_list = []
    for coeff, op in zip(coeffs, ops):
        if isinstance(op, (Tensor, Prod, SProd, Hamiltonian)):
            string = qml.pauli.pauli_word_to_string(op, wire_map=wire_map)
            n = len(string)//2  # => valid for closed shell singlets only
            string = ''.join([string[i::n] for i in range(n)])
            string = string[::-1]
            q_op_list.append((string, coeff))
        if isinstance(op, Sum):
            # print(coeff, op.simplify(), op._build_pauli_rep(), op.terms())
            raise ValueError(
                    "Pauli operators representing :class:`.Sum` "
                    "not accepted in the current implementation.")

    return SparsePauliOp.from_list(q_op_list)


def _qiskit_nature_pennylane_equivalent(
    qiskit_qubit_operator, pennylane_qubit_operator, wires=None
):
    """Check functionality of :func:`_pennylane_to_qiskit_nature`.

    Check equivalence between Qiskit :class:`~.SparsePauliOp` and Pennylane
    VQE ``Hamiltonian`` (Tensor product of Pauli matrices).

    Equality is based on Qiskit :class:`~.SparsePauliOp`'s equality.

    Args:
        qiskit_qubit_operator (SparsePauliOp): Qiskit-Natuire qubit operator
            represented as a Pauli summation
        pennylane_qubit_operator (pennylane.Hamiltonian): PennyLane
            Hamiltonian object
        wires (Wires, list, tuple, dict): Full wire mapping used to convert
            to qubit operator from an observable terms measurable in a
            PennyLane ansatz. For types Wires/list/tuple, each item in the
            iterable represents a wire label corresponding to the qubit number
            equal to its index. For type dict, only consecutive-int-valued
            dict (for wire-to-qubit conversion) is accepted.

    Returns:
        (bool): True if equivalent
    """
    coeffs, ops = pennylane_qubit_operator.terms()
    return qiskit_qubit_operator == _pennylane_to_qiskit_nature(
        coeffs, ops, wires=wires)


def import_operator(qubit_observable, format="openfermion",
                    wires=None, tol=1e010):
    r"""Convert an external operator to a PennyLane operator.

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
        linear comb of Pauli words, e.g., :math:`\\sum_{k=0}^{N-1} c_k O_k`

    **Example**

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
