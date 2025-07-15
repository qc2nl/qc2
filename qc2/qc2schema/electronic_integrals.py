
# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from __future__ import annotations
from numbers import Number
import numpy as np
from typing import Dict, Optional, Sequence, Tuple, cast, Callable
# from .polynomial_tensor import PolynomialTensor


class TensorDict(Dict):

    def is_empty(self) -> bool:
        """Returns whether this tensor is empty or not."""
        return len(self) == 0

    def __mul__(self, other: Number) -> TensorDict:
        """Scalar multiplication of a PolynomialTensor with a scalar.

        Args:
            other: scalar to be multiplied with the ``PolynomialTensor``.

        Returns:
            The new ``PolynomialTensor`` product object.

        Raises:
            TypeError: if ``other`` is not a number.
        """
        if not isinstance(other, Number):
            raise TypeError(f"other {other} must be a number")

        prod_dict: dict[str, np.ndarray] = {}
        for key, matrix in self.items():
            prod_dict[key] = other * matrix

        return TensorDict(prod_dict)
    
    def __rmul__(self, other: Number) -> TensorDict:
        return self * other

    def __add__(self, other: TensorDict, qargs=None) -> TensorDict:
        """Addition of ``PolynomialTensor`` instances.

        Args:
            other: second ``PolynomialTensor`` object to be added to the first.

        Returns:
            The new summed ``PolynomialTensor``.

        Raises:
            TypeError: when ``other`` is not a ``PolynomialTensor``.
            ValueError: when values corresponding to keys in ``other`` and the first
                ``PolynomialTensor`` object do not match.
        """
        if not isinstance(other, TensorDict):
            raise TypeError("Incorrect argument type: other should be TensorDict")

        sum_dict: dict[str, np.ndarray] = {}
        for key, value in self.items():
            sum_dict[key] = value + other.get(key, 0)

        return TensorDict(sum_dict)
    
    def __sub__(self, other: TensorDict, qargs=None) -> TensorDict:
        return self + other * (-1)

    @classmethod
    def einsum(
        cls,
        einsum_map: dict[str, tuple[str, ...]],
        *operands: TensorDict,
    ) -> TensorDict:
        """Applies the various Einsum convention operations to the provided tensors.

        This method wraps the :meth:`numpy.einsum` function, allowing very complex operations to be
        applied efficiently to the matrices stored inside the provided ``PolynomialTensor``
        operands.

        As an example, let us compute the exact exchange term of an
        :class:`qiskit_nature.second_q.hamiltonians.ElectronicEnergy` hamiltonian:

        .. code-block:: python

            # a PolynomialTensor containing the two-body terms of an ElectronicEnergy hamiltonian
            two_body = PolynomialTensor({"++--": ...})

            # an electronic density:
            density = PolynomialTensor({"+-": ...})

            # computes the ElectronicEnergy.exchange operator
            exchange = PolynomialTensor.einsum(
                {"pqrs,qs->pr": ("++--", "+-", "+-")},
                two_body,
                density,
            )
            # result will be contained in exchange["+-"]

        Another example is the mapping from the AO to MO basis, as implemented by the
        :class:`qiskit_nature.second_q.transformers.BasisTransformer`.

        .. code-block:: python

            # the one- and two-body integrals of a hamiltonian
            hamiltonian = PolynomialTensor({"+-": ..., "++--": ...})

            # the AO-to-MO transformation coefficients
            mo_coeff = PolynomialTensor({"+-": ...})

            einsum_map = {
                "jk,ji,kl->il": ("+-", "+-", "+-", "+-"),
                "prsq,pi,qj,rk,sl->iklj": ("++--", "+-", "+-", "+-", "+-", "++--"),
            }

            transformed = PolynomialTensor.einsum(
                einsum_map, hamiltonian, mo_coeff, mo_coeff, mo_coeff, mo_coeff
            )
            # results will be contained in transformed["+-"] and transformed["++--"], respectively

        .. note::

           :class:`sparse.SparseArray` supports ``opt_einsum.contract` if ``opt_einsum`` is installed.
           It does not support ``numpy.einsum``. In this case, the resultant
           ``PolynomialTensor`` will contain all dense numpy arrays. If a user would like to work
           with a sparse array instead, they should install ``opt_einsum`` or
           they should convert it explicitly using the :meth:`to_sparse` method.

        Args:
            einsum_map: a dictionary, mapping from :meth:`numpy.einsum` subscripts to a tuple of
                strings. These strings correspond to the keys of matrices to be extracted from the
                provided ``PolynomialTensor`` operands. The last string in this tuple indicates the
                key under which to store the result in the returned ``PolynomialTensor``.
            operands: a sequence of ``PolynomialTensor`` instances on which to operate.
            validate: when set to False the ``data`` will not be validated. Disable this setting
                with care!

        Returns:
            A new ``TensorDict``.
        """
        einsum_func, uses_sparse = get_einsum()
        operand_list = list(operands) if uses_sparse else [op.to_dense() for op in operands]
        new_data: dict[str, np.ndarray] = {}
        for einsum, terms in einsum_map.items():
            *inputs, output = terms
            try:
                ops = []
                for idx, term in enumerate(inputs):
                    op = operand_list[idx][term]
                    ops.append(op)
                result = einsum_func(einsum, *ops)
            except KeyError:
                continue
            if output in new_data:
                new_data += result
            else:
                new_data[output] = result

        return cls(new_data)

class ElectronicIntegrals:

    _VALID_KEYS = {"", "+-", "++--"}

    def __init__(
        self,
        alpha: TensorDict | None = None,
        beta: TensorDict | None = None,
        beta_alpha: TensorDict | None = None,
        *,
        validate: bool = True,
    ) -> None:
        
        self.alpha = alpha
        self.beta = beta
        self.beta_alpha = beta_alpha

        if validate:
            self._validate()

    def _validate(self):

        """Validates the keys of all internal tensors."""
        if not self.alpha.keys() <= ElectronicIntegrals._VALID_KEYS:
            raise KeyError(
                "The only allowed keys for the alpha-spin tensor are '', '+-', and '++--', but your"
                f" tensor has keys: {self.alpha.keys()}"
            )

        if not self.beta.keys() <= ElectronicIntegrals._VALID_KEYS:
            raise KeyError(
                "The only allowed keys for the beta-spin tensor are '', '+-', and '++--', but your"
                f" tensor has keys: {self.beta.keys()}"
            )

        if not self.beta_alpha.keys() <= {"++--"}:
            raise KeyError(
                "The only allowed key for the beta-alpha-spin tensor is '++--', but your "
                f" tensor has keys: {self.beta_alpha.keys()}"
            )

    @property
    def alpha(self) -> TensorDict:
        """The up-spin electronic integrals."""
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: TensorDict | None) -> None:
        self._alpha = alpha if alpha is not None else {}

    @property
    def beta(self) -> TensorDict:
        """The down-spin electronic integrals."""
        return self._beta

    @beta.setter
    def beta(self, beta: TensorDict | None) -> None:
        self._beta = beta if beta is not None else {}

    @property
    def beta_alpha(self) -> TensorDict:
        """The beta-alpha-spin two-body electronic integrals."""
        return self._beta_alpha

    @beta_alpha.setter
    def beta_alpha(self, beta_alpha: TensorDict | None) -> None:
        if beta_alpha is None:
            self._beta_alpha = TensorDict({})
        else:
            keys = set(beta_alpha)
            if keys and keys != {"++--"}:
                raise ValueError(
                    f"The beta_alpha tensor may only contain a `++--` key, not {keys}."
                )
            self._beta_alpha = beta_alpha

    @property
    def one_body(self) -> ElectronicIntegrals:
        """Returns only the one-body integrals."""
        alpha: TensorDict = None
        if "+-" in self.alpha:
            alpha = TensorDict(
                {"+-": self.alpha["+-"]}
            )
        beta: TensorDict = None
        if "+-" in self.beta:
            beta = TensorDict(
                {"+-": self.beta["+-"]},
            )
        return self.__class__(alpha, beta)

    @property
    def two_body(self) -> ElectronicIntegrals:
        """Returns only the two-body integrals."""
        alpha: TensorDict = None
        if "++--" in self.alpha:
            alpha = TensorDict(
                {"++--": self.alpha["++--"]},
            )
        beta: TensorDict = None
        if "++--" in self.beta:
            beta = TensorDict(
                {"++--": self.beta["++--"]},
            )
        beta_alpha: TensorDict = None
        if "++--" in self.beta_alpha:
            beta_alpha = TensorDict(
                {"++--": self.beta_alpha["++--"]},
            )
        return self.__class__(alpha, beta, beta_alpha)

    def __mul__(self, other: Number) -> ElectronicIntegrals:
        if not isinstance(other, Number):
            raise TypeError(f"other {other} must be a number")

        return self.__class__(
            cast(TensorDict, self.alpha * other),
            cast(TensorDict, self.beta * other),
            cast(TensorDict, self.beta_alpha * other),
        )
    
    def __rmul__(self, other: Number) -> ElectronicIntegrals:
        return self * other

    def __add__(self, other: ElectronicIntegrals, qargs=None) -> ElectronicIntegrals:
        if not isinstance(other, ElectronicIntegrals):
            raise TypeError("Incorrect argument type: other should be ElectronicIntegrals")

        # we need to handle beta separately in order to inject alpha where necessary
        beta = TensorDict({})
        beta_self_empty = len(self.beta) == 0
        beta_other_empty = len(other.beta) == 0
        if not (beta_self_empty and beta_other_empty):
            beta_self = self.alpha if beta_self_empty else self.beta
            beta_other = other.alpha if beta_other_empty else other.beta
            beta = beta_self + beta_other

        return self.__class__(
            self.alpha + other.alpha,
            beta,
            self.beta_alpha + other.beta_alpha,
        )
    
    def __sub__(self, other: ElectronicIntegrals, qargs=None) -> ElectronicIntegrals:
        return self + other * (-1)
    

    @classmethod
    def einsum(
        cls,
        einsum_map: dict[str, tuple[str, ...]],
        *operands: ElectronicIntegrals,
        ) -> ElectronicIntegrals:
        """Exposes the :meth:`qiskit_nature.second_q.operators.PolynomialTensor.einsum` method.

        This behaves identical to the ``einsum`` implementation of the ``PolynomialTensor``, applied
        to the :attr:`alpha`, :attr:`beta`, and :attr:`beta_alpha` attributes of the provided
        ``ElectronicIntegrals`` operands.

        This method is special, because it handles the scenario in which any operand has a non-empty
        :attr:`beta` attribute, in which case the empty-beta attributes of any other operands will
        be filled with :attr:`alpha` attributes of those operands.
        The :attr:`beta_alpha` attributes will only be handled if they are non-empty in all supplied
        operands.

        Args:
            einsum_map: a dictionary, mapping from :meth:`numpy.einsum` subscripts to a tuple of
                strings. These strings correspond to the keys of matrices to be extracted from the
                provided ``ElectronicIntegrals`` operands. The last string in this tuple indicates
                the key under which to store the result in the returned ``ElectronicIntegrals``.
            operands: a sequence of ``ElectronicIntegrals`` instances on which to operate.
            validate: when set to False, no validation will be performed. Disable this setting with
                care!

        Returns:
            A new ``ElectronicIntegrals``.
        """
        alpha = TensorDict.einsum(
            einsum_map, *(op.alpha for op in operands)
        )

        beta = TensorDict({})
        if any(not len(op.beta) == 0 for op in operands):
            # If any beta-entry is non-empty, we have to perform this computation.
            # Empty tensors will be populated with their alpha-terms automatically.
            beta = TensorDict.einsum(
                einsum_map,
                *(op.alpha if len(op.beta) == 0 else op.beta for op in operands),
            )

        beta_alpha = TensorDict({})
        if all(not len(op.beta_alpha) == 0 for op in operands):
            # We can only perform this operation, when all beta_alpha tensors are non-empty.
            beta_alpha = TensorDict.einsum(
                einsum_map, *(op.beta_alpha for op in operands)
            )
        return cls(alpha, beta, beta_alpha)


    
def get_einsum() -> tuple[Callable, bool]:
    """Returns the ``einsum`` implementation.

    This returns a tuple of a callable and boolean. The callable is the ``einsum`` implementation.
    If ``opt_einsum`` is installed, ``opt_einsum.contract`` will be used. Otherwise this falls
    back to ``np.einsum``. The boolean indicates support for sparse arrays.

    Returns:
        The pair of the ``einsum`` callable and sparse support indicator.
    """
    try:
        # pylint: disable=import-error
        from opt_einsum import contract
        return contract, True
    except ImportError:
        return np.einsum, False