
"""Qubit Mapper interface."""

from __future__ import annotations

from abc import ABC
from typing import TypeVar, Dict, Iterable, Generic, Generator

import numpy as np


import pennylane as qml
from pennylane.fermi import from_string
from ..base_mapper import BaseMapper

class PennylaneBaseMapper(BaseMapper):

    @staticmethod
    def reformat_str(ops: str) -> str:
        op_list = ops.split()
        reformatted_op_list = []
        for o in op_list:
            o = o.split('_')
            o.reverse()
            reformatted_op_list.append(''.join(o))
        return ' '.join(reformatted_op_list)
