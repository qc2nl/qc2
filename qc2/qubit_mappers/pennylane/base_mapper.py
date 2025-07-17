
"""Qubit Mapper interface."""

from __future__ import annotations

from abc import ABC
from typing import TypeVar, Dict, Iterable, Generic, Generator

import numpy as np


import pennylane as qml
from pennylane.fermi import from_string

class BaseMapper(ABC):

    mapper = None

    def format_str(op: str) -> str:
        op = op.split('_')
        op.reverse()
        return ''.join(op)

    def _map_single(self, second_q_op):

        single_op_list = second_q_op.split()
        for op in single_op_list:
            self.mapper(from_string(self.format_str(op), ps=True))



    def map(self, second_q_ops: Dict):

        qubit_ops = {}
        for name, second_q_op in second_q_ops.items():
            qubit_ops[name] = self._map_single(second_q_op)

        return qubit_ops
