
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
from .polynomial_tensor import PolynomialTensor

class ElectronicIntegrals:

    _VALID_KEYS = {"", "+-", "++--"}

    def __init__(
        self,
        alpha: PolynomialTensor | None = None,
        beta: PolynomialTensor | None = None,
        beta_alpha: PolynomialTensor | None = None,
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
    def alpha(self) -> PolynomialTensor:
        """The up-spin electronic integrals."""
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: PolynomialTensor | None) -> None:
        self._alpha = alpha if alpha is not None else PolynomialTensor.empty()

    @property
    def beta(self) -> PolynomialTensor:
        """The down-spin electronic integrals."""
        return self._beta

    @beta.setter
    def beta(self, beta: PolynomialTensor | None) -> None:
        self._beta = beta if beta is not None else PolynomialTensor.empty()

    @property
    def beta_alpha(self) -> PolynomialTensor:
        """The beta-alpha-spin two-body electronic integrals."""
        return self._beta_alpha

    @beta_alpha.setter
    def beta_alpha(self, beta_alpha: PolynomialTensor | None) -> None:
        if beta_alpha is None:
            self._beta_alpha = PolynomialTensor.empty()
        else:
            keys = set(beta_alpha)
            if keys and keys != {"++--"}:
                raise ValueError(
                    f"The beta_alpha tensor may only contain a `++--` key, not {keys}."
                )
            self._beta_alpha = beta_alpha

    @property
    def alpha_beta(self) -> PolynomialTensor:
        """The alpha-beta-spin two-body electronic integrals.

        These get reconstructed from :attr:`beta_alpha` by transposing in the physicist' ordering
        convention.
        """
        if self.beta_alpha.is_empty():
            return self.beta_alpha
        return PolynomialTensor({"++--": self.beta_alpha["++--"]}, validate=False)

    @property
    def one_body(self) -> ElectronicIntegrals:
        """Returns only the one-body integrals."""
        alpha: PolynomialTensor = None
        if "+-" in self.alpha:
            alpha = PolynomialTensor(
                {"+-": self.alpha["+-"]},
                validate=False,
            )
        beta: PolynomialTensor = None
        if "+-" in self.beta:
            beta = PolynomialTensor(
                {"+-": self.beta["+-"]},
                validate=False,
            )
        return self.__class__(alpha, beta)

    @property
    def two_body(self) -> ElectronicIntegrals:
        """Returns only the two-body integrals."""
        alpha: PolynomialTensor = None
        if "++--" in self.alpha:
            alpha = PolynomialTensor(
                {"++--": self.alpha["++--"]},
                validate=False,
            )
        beta: PolynomialTensor = None
        if "++--" in self.beta:
            beta = PolynomialTensor(
                {"++--": self.beta["++--"]},
                validate=False,
            )
        beta_alpha: PolynomialTensor = None
        if "++--" in self.beta_alpha:
            beta_alpha = PolynomialTensor(
                {"++--": self.beta_alpha["++--"]},
                validate=False,
            )
        return self.__class__(alpha, beta, beta_alpha)
    
    @property
    def register_length(self) -> int | None:
        """The size of the operator that can be generated from these `ElectronicIntegrals`."""
        alpha_length = self.alpha.register_length
        return alpha_length

    def __eq__(self, other: object) -> bool:
        """Check equality of first ElectronicIntegrals with other

        Args:
            other: second ``ElectronicIntegrals`` object to be compared with the first.
        Returns:
            True when ``ElectronicIntegrals`` objects are equal, False when unequal.
        """
        if not isinstance(other, ElectronicIntegrals):
            return False

        if (
            self.alpha == other.alpha
            and self.beta == other.beta
            and self.beta_alpha == other.beta_alpha
        ):
            return True

        return False

    def equiv(self, other: object) -> bool:
        """Check equivalence of first ElectronicIntegrals with other

        Args:
            other: second ``ElectronicIntegrals`` object to be compared with the first.
        Returns:
            True when ``ElectronicIntegrals`` objects are equivalent, False when not.
        """
        if not isinstance(other, ElectronicIntegrals):
            return False

        if (
            self.alpha.equiv(other.alpha)
            and self.beta.equiv(other.beta)
            and self.beta_alpha.equiv(other.beta_alpha)
        ):
            return True

        return False

    def _multiply(self, other: complex) -> ElectronicIntegrals:
        if not isinstance(other, Number):
            raise TypeError(f"other {other} must be a number")

        return self.__class__(
            cast(PolynomialTensor, other * self.alpha),
            cast(PolynomialTensor, other * self.beta),
            cast(PolynomialTensor, other * self.beta_alpha),
        )

    def _add(self, other: ElectronicIntegrals, qargs=None) -> ElectronicIntegrals:
        if not isinstance(other, ElectronicIntegrals):
            raise TypeError("Incorrect argument type: other should be ElectronicIntegrals")

        # we need to handle beta separately in order to inject alpha where necessary
        beta: PolynomialTensor = None
        beta_self_empty = self.beta.is_empty()
        beta_other_empty = other.beta.is_empty()
        if not (beta_self_empty and beta_other_empty):
            beta_self = self.alpha if beta_self_empty else self.beta
            beta_other = other.alpha if beta_other_empty else other.beta
            beta = beta_self + beta_other

        return self.__class__(
            self.alpha + other.alpha,
            beta,
            self.beta_alpha + other.beta_alpha,
        )

    @classmethod
    def apply(
        cls,
        function: Callable[..., np.ndarray  | complex],
        *operands: ElectronicIntegrals,
        multi: bool = False,
        validate: bool = True,
    ) -> ElectronicIntegrals | list[ElectronicIntegrals]:
        """Exposes the :meth:`qiskit_nature.second_q.operators.PolynomialTensor.apply` method.

        This behaves identical to the ``apply`` implementation of the ``PolynomialTensor``, applied
        to the :attr:`alpha`, :attr:`beta`, and :attr:`beta_alpha` attributes of the provided
        ``ElectronicIntegrals`` operands.

        This method is special, because it handles the scenario in which any operand has a non-empty
        :attr:`beta` attribute, in which case the empty-beta attributes of any other operands will
        be filled with :attr:`alpha` attributes of those operands.
        The :attr:`beta_alpha` attributes will only be handled if they are non-empty in all supplied
        operands.

        Args:
            function: the function to apply to the internal arrays of the provided operands. This
                function must take numpy (or sparse) arrays as its positional arguments. The number
                of arguments must match the number of provided operands.
            operands: a sequence of ``ElectronicIntegrals`` instances on which to operate.
            multi: when set to True this indicates that the provided numpy function will return
                multiple new numpy arrays which will each be wrapped into an ``ElectronicIntegrals``
                instance separately.
            validate: when set to False, no validation will be performed. Disable this setting with
                care!

        Returns:
            A new ``ElectronicIntegrals``.
        """
        alphas = PolynomialTensor.apply(
            function, *(op.alpha for op in operands), multi=multi, validate=validate
        )

        betas: PolynomialTensor | list[PolynomialTensor] | None = None
        if any(not op.beta.is_empty() for op in operands):
            # If any beta-entry is non-empty, we have to perform this computation.
            # Empty tensors will be populated with their alpha-terms automatically.
            betas = PolynomialTensor.apply(
                function,
                *(op.alpha if op.beta.is_empty() else op.beta for op in operands),
                multi=multi,
                validate=validate,
            )

        beta_alphas: PolynomialTensor | list[PolynomialTensor] | None = None
        if all(not op.beta_alpha.is_empty() for op in operands):
            # We can only perform this operation, when all beta_alpha tensors are non-empty.
            beta_alphas = PolynomialTensor.apply(
                function, *(op.beta_alpha for op in operands), multi=multi, validate=validate
            )

        if multi:
            if betas is None:
                betas = [None] * len(alphas)
            if beta_alphas is None:
                beta_alphas = [None] * len(alphas)
            return [
                cls(a, b, ba, validate=validate) for a, b, ba in zip(alphas, betas, beta_alphas)
            ]

        alphas = cast(PolynomialTensor, alphas)
        betas = cast(Optional[PolynomialTensor], betas)
        beta_alphas = cast(Optional[PolynomialTensor], beta_alphas)
        return cls(alphas, betas, beta_alphas, validate=validate)

    @classmethod
    def stack(
        cls,
        function: Callable[..., np.ndarray  | Number],
        operands: Sequence[ElectronicIntegrals],
        *,
        validate: bool = True,
    ) -> ElectronicIntegrals:
        """Exposes the :meth:`qiskit_nature.second_q.operators.PolynomialTensor.stack` method.

        This behaves identical to the ``stack`` implementation of the ``PolynomialTensor``, applied
        to the :attr:`alpha`, :attr:`beta`, and :attr:`beta_alpha` attributes of the provided
        ``ElectronicIntegrals`` operands.

        This method is special, because it handles the scenario in which any operand has a non-empty
        :attr:`beta` attribute, in which case the empty-beta attributes of any other operands will
        be filled with :attr:`alpha` attributes of those operands.
        The :attr:`beta_alpha` attributes will only be handled if they are non-empty in all supplied
        operands.

        .. note::

            When stacking arrays this will likely lead to array shapes which would fail the shape
            validation check. This is considered an advanced use case which is why the user is left
            to disable this check themselves, to ensure they know what they are doing.

        Args:
            function: the stacking function to apply to the internal arrays of the provided
                operands. This function must take a sequence of numpy (or sparse) arrays as its
                first argument. You should use :code:`functools.partial` if you need to provide
                keyword arguments (e.g. :code:`partial(np.stack, axis=-1)`). Common methods to use
                here are :func:`numpy.hstack` and :func:`numpy.vstack`.
            operands: a sequence of ``ElectronicIntegrals`` instances on which to operate.
            validate: when set to False, no validation will be performed. Disable this setting with
                care!

        Returns:
            A new ``ElectronicIntegrals``.
        """
        alpha = PolynomialTensor.stack(function, [op.alpha for op in operands], validate=validate)

        beta: PolynomialTensor = None
        if any(not op.beta.is_empty() for op in operands):
            # If any beta-entry is non-empty, we have to perform this computation.
            # Empty tensors will be populated with their alpha-terms automatically.
            beta = PolynomialTensor.stack(
                function,
                [op.alpha if op.beta.is_empty() else op.beta for op in operands],
                validate=validate,
            )

        beta_alpha: PolynomialTensor = None
        if all(not op.beta_alpha.is_empty() for op in operands):
            # We can only perform this operation, when all beta_alpha tensors are non-empty.
            beta_alpha = PolynomialTensor.stack(
                function, [op.beta_alpha for op in operands], validate=validate
            )
        return cls(alpha, beta, beta_alpha, validate=validate)

    def split(
        self,
        function: Callable[..., np.ndarray  | Number],
        indices_or_sections: int | Sequence[int],
        *,
        validate: bool = True,
    ) -> list[ElectronicIntegrals]:
        """Exposes the :meth:`qiskit_nature.second_q.operators.PolynomialTensor.split` method.

        This behaves identical to the ``split`` implementation of the ``PolynomialTensor``, applied
        to the :attr:`alpha`, :attr:`beta`, and :attr:`beta_alpha` attributes of the provided
        ``ElectronicIntegrals`` operands.

        .. note::

            When splitting arrays this will likely lead to array shapes which would fail the shape
            validation check. This is considered an advanced use case which is why the user is left
            to disable this check themselves, to ensure they know what they are doing.

        Args:
            function: the splitting function to use. This function must take a single numpy (or
                sparse) array as its first input followed by a sequence of indices to split on.
                You should use :code:`functools.partial` if you need to provide keyword arguments
                (e.g. :code:`partial(np.split, axis=-1)`). Common methods to use here are
                :func:`numpy.hsplit` and :func:`numpy.vsplit`.
            indices_or_sections: a single index or sequence of indices to split on.
            validate: when set to False, no validation will be performed. Disable this setting with
                care!

        Returns:
            The new ``ElectronicIntegrals`` instances.
        """
        alphas = self.alpha.split(function, indices_or_sections, validate=validate)

        betas: list[PolynomialTensor | None]
        if self.beta.is_empty():
            betas = [None] * len(alphas)
        else:
            betas = self.beta.split(function, indices_or_sections, validate=validate)

        beta_alphas: list[PolynomialTensor | None]
        if self.beta_alpha.is_empty():
            beta_alphas = [None] * len(alphas)
        else:
            beta_alphas = self.beta_alpha.split(function, indices_or_sections, validate=validate)

        return [
            self.__class__(a, b, ba, validate=validate)
            for a, b, ba in zip(alphas, betas, beta_alphas)
        ]

    @classmethod
    def einsum(
        cls,
        einsum_map: dict[str, tuple[str, ...]],
        *operands: ElectronicIntegrals,
        validate: bool = True,
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
        alpha = PolynomialTensor.einsum(
            einsum_map, *(op.alpha for op in operands), validate=validate
        )

        beta: PolynomialTensor = None
        if any(not op.beta.is_empty() for op in operands):
            # If any beta-entry is non-empty, we have to perform this computation.
            # Empty tensors will be populated with their alpha-terms automatically.
            beta = PolynomialTensor.einsum(
                einsum_map,
                *(op.alpha if op.beta.is_empty() else op.beta for op in operands),
                validate=validate,
            )

        beta_alpha: PolynomialTensor = None
        if all(not op.beta_alpha.is_empty() for op in operands):
            # We can only perform this operation, when all beta_alpha tensors are non-empty.
            beta_alpha = PolynomialTensor.einsum(
                einsum_map, *(op.beta_alpha for op in operands), validate=validate
            )
        return cls(alpha, beta, beta_alpha, validate=validate)

    # pylint: disable=invalid-name
    @classmethod
    def from_raw_integrals(
        cls,
        h1_a: np.ndarray ,
        h2_aa: np.ndarray  | None = None,
        h1_b: np.ndarray  | None = None,
        h2_bb: np.ndarray  | None = None,
        h2_ba: np.ndarray  | None = None,
        *,
        validate: bool = True,
    ) -> ElectronicIntegrals:
        """Loads the provided integral matrices into an ``ElectronicIntegrals`` instance.

        When ``auto_index_order`` is enabled,
        :meth:`qiskit_nature.second_q.operators.tensor_ordering.find_index_order` will be used to
        determine the index ordering of the ``h2_aa`` matrix, based on which the two-body matrices
        will automatically be transformed to the physicist' order, which is required by the
        :class:`qiskit_nature.second_q.operators.PolynomialTensor`.

        Args:
            h1_a: the alpha-spin one-body integrals.
            h2_aa: the alpha-alpha-spin two-body integrals.
            h1_b: the beta-spin one-body integrals.
            h2_bb: the beta-beta-spin two-body integrals.
            h2_ba: the beta-alpha-spin two-body integrals.
            validate: whether or not to validate the integral matrices. Disable this setting with
                care!
            auto_index_order: whether or not to automatically convert the matrices to physicists'
                order.

        Raises:
            QiskitNatureError: if `auto_index_order=True`, upon encountering an invalid
                :class:`qiskit_nature.second_q.operators.tensor_ordering.IndexType`.

        Returns:
            The resulting ``ElectronicIntegrals``.
        """
        alpha_dict = {"+-": h1_a}

        if h2_aa is not None:
            alpha_dict["++--"] = h2_aa

        alpha = PolynomialTensor(alpha_dict, validate=validate)

        beta = None
        beta_dict = {}
        if h1_b is not None:
            beta_dict["+-"] = h1_b
        if h2_bb is not None:
            beta_dict["++--"] = h2_bb
        if beta_dict:
            beta = PolynomialTensor(beta_dict, validate=validate)

        beta_alpha = None
        if h2_ba is not None:
            beta_alpha = PolynomialTensor({"++--": h2_ba}, validate=validate)

        return cls(alpha, beta, beta_alpha, validate=validate)


    def trace_spin(self) -> PolynomialTensor:
        """Returns a :class:`~.PolynomialTensor` where the spin components have been traced out.

        This will sum the :attr:`alpha` and :attr:`beta` components, tracing out the spin.

        Returns:
            A ``PolynomialTensor`` with the spin traced out.
        """
        beta_empty = self.beta.is_empty()
        beta_alpha_empty = self.beta_alpha.is_empty()

        if beta_empty and beta_alpha_empty:
            return cast(PolynomialTensor, 2.0 * self.alpha)

        two_body = self.two_body
        tensor_spin_traced = PolynomialTensor({})
        tensor_spin_traced += self.alpha
        tensor_spin_traced += self.beta
        if beta_alpha_empty:
            tensor_spin_traced += two_body.alpha
            tensor_spin_traced += two_body.beta
        else:
            tensor_spin_traced += two_body.beta_alpha
            tensor_spin_traced += two_body.alpha_beta

        return tensor_spin_traced