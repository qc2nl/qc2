
"""Qubit Mapper interface."""

from __future__ import annotations

from abc import ABC
from ..second_q.fermionic_operator import FermionicOperator
class BaseMapper(ABC):
    """Qubit Mapper interface."""

    def map(self, second_q_ops: FermionicOperator):
        raise NotImplementedError("BaseMapper doens't have a .map() implemented ")