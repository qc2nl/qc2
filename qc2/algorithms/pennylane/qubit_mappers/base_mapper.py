
"""Qubit Mapper interface."""

from __future__ import annotations

from typing import Tuple, List, Dict

import pennylane as qml
import pennylane.numpy as np
from pennylane.pauli.pauli_arithmetic import PauliSentence
from ...utils.base_mapper import BaseMapper

class PennylaneBaseMapper(BaseMapper):

    @staticmethod
    def reformat_str(ops: str) -> str:
        """Reformat the string from `+_0 -_1` to `0+ 1-`"""
        op_list = ops.split()
        reformatted_op_list = []
        for o in op_list:
            o = o.split('_')
            o.reverse()
            reformatted_op_list.append(''.join(o))
        return ' '.join(reformatted_op_list)
    @staticmethod
    def get_wire_map(length: int) -> Dict:
        """get the wire map."""
        n = length - 1
        return { i: 2*i%n for i in range(1, n) }

    def _return_data(self, pauli_sentence: PauliSentence) -> Tuple[np.ndarray, List]:
        """Return data for the qubit mapper."""
        data = (np.array(list(pauli_sentence.values())).real, 
                 list(pauli_sentence.keys()))
        return qml.dot(*data)

