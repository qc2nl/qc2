
"""Qubit Mapper interface."""

from __future__ import annotations

from abc import ABC
from typing import TypeVar, Tuple, List 

import pennylane as qml
import pennylane.numpy as np
from pennylane.operation import active_new_opmath
from pennylane.pauli.pauli_arithmetic import PauliSentence
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

    def _return_data(self, pauli_sentence: PauliSentence) -> Tuple[np.ndarray, List]:
        """Return data for the qubit mapper."""

        data = (np.array(list(pauli_sentence.values())).real, 
                 list(pauli_sentence.keys()))
        
        if active_new_opmath():
            return qml.dot(*data)
        return qml.Hamiltonian(*data)

