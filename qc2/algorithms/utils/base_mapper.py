
"""Qubit Mapper interface."""

from __future__ import annotations

from abc import ABC
from typing import List, Any, Tuple
from ...second_q.fermionic_operator import FermionicOperator
class BaseMapper(ABC):
    """Qubit Mapper interface."""

    backend: str | None = None

    def map(self, second_q_ops: FermionicOperator):
        raise NotImplementedError("BaseMapper doens't have a .map() implemented ")
    
    def get_representation(self, qubit_op: Any) -> Tuple[List[Any], List[Any]]:
        raise NotImplementedError("BaseMapper doens't have a .get_representation() implemented ")