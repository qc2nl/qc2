# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Fermionic-particle Operator."""

from __future__ import annotations

import re
from numbers import Number
from collections import defaultdict
from collections.abc import Collection, Mapping
from typing import Iterator, Sequence, Dict

import numpy as np

class FermionicOperator(Dict):

    def __init__(self, data: Mapping[str, float] | None = None, 
                 num_spin_orbitals: int | None = None):
        if data is None:
            data = {}
        super().__init__(data)
        self.num_spin_orbitals = num_spin_orbitals

    def __add__(self, other: FermionicOperator, qargs: None = None) -> FermionicOperator:
        """Return Operator addition of self and other.

        Args:
            other: the second ``SparseLabelOp`` to add to the first.
            qargs: UNUSED.

        Returns:
            The new summed ``SparseLabelOp``.

        Raises:
            ValueError: when ``qargs`` argument is not ``None``
        """
        if self.num_spin_orbitals != other.num_spin_orbitals:
            raise ValueError(
                f"Cannot add operators with different number of spin orbitals: "
                f"{self.num_spin_orbitals} and {other.num_spin_orbitals}"
            )
        if not isinstance(other, self.__class__):
            raise ValueError(
                f"Unsupported operand type(s) for +: '{type(self)}' and '{type(other).__name__}'"
            )

        new_data = {key: value + other.get(key, 0) for key, value in self.items()}
        other_unique = {key: other[key] for key in other.keys() - self.keys()}
        new_data.update(other_unique)

        return self.__class__(new_data, num_spin_orbitals=self.num_spin_orbitals)

    def __sub__(self, other: FermionicOperator, qargs: None = None) -> FermionicOperator:
        """Return Operator subtraction of self and other."""
        return self + other * (-1)

    def __mul__(self, other: Number) -> FermionicOperator:
        """Return scalar multiplication of self and other.

        Args:
            other: the number to multiply the ``SparseLabelOp`` values by.

        Returns:
            The newly multiplied ``SparseLabelOp``.

        Raises:
            TypeError: if ``other`` is not compatible type (int, float or complex)
        """
        if not isinstance(other, (Number)):
            raise TypeError(
                f"Unsupported operand type(s) for *: 'FermionicOperator' and '{type(other).__name__}'"
            )
        new_data = {key: val * other for key, val in self.items()}

        return self.__class__(new_data, num_spin_orbitals=self.num_spin_orbitals)

    def __rmul__(self, other: Number) -> FermionicOperator:
        return self.__mul__(other)

    @property
    def register_length(self) -> int:
        if self.num_spin_orbitals is None:
            self.num_spin_orbitals = max(int(ks[2:]) for k in self.keys() for ks in k.split()) + 1 
        return self.num_spin_orbitals

    def terms(self) -> Iterator[tuple[list[tuple[str, int]]]]:
        """Provides an iterator analogous to :meth:`items` but with the labels already split into
        pairs of operation characters and indices.

        Yields:
            A tuple with two items; the first one being a list of pairs of the form (char, int)
            where char is either `+` or `-` and the integer corresponds to the fermionic mode index
            on which the operator gets applied; the second item of the returned tuple is the
            coefficient of this term.
        """
        for label in iter(self):
            if not label:
                yield ([], self[label])
                continue
            # we hard-code the result of lbl.split("_") as follows:
            #   lbl[0] is either + or -
            #   lbl[2:] corresponds to the index
            terms = [(lbl[0], int(lbl[2:])) for lbl in label.split()]
            yield (terms, self[label])

    @classmethod
    def from_terms(cls, terms: Sequence[tuple[list[tuple[str, int]], float]]) -> FermionicOperator:
        data = {
            " ".join(f"{action}_{index}" for action, index in label): value
            for label, value in terms
        }
        return cls(data)
    
    def adjoint(self) -> FermionicOperator:
        """Return the adjoint of the FermionicOperator."""
        return self.conjugate().transpose()

    def transpose(self) -> FermionicOperator:
        data = {}

        trans = "".maketrans("+-", "-+")

        for label, coeff in self.items():
            data[" ".join(lbl.translate(trans) for lbl in reversed(label.split()))] = coeff

        return self.__class__(data, num_spin_orbitals=self.num_spin_orbitals)

    def conjugate(self) -> FermionicOperator:
        """Returns the conjugate of the ``FermionicOperator``.

        Returns:
            The complex conjugate of the starting ``FermionicOperator``.
        """
        new_data = {key: np.conjugate(val) for key, val in self.items()}

        return self.__class__(new_data, num_spin_orbitals=self.num_spin_orbitals)