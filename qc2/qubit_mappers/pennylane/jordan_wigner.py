from pennylane.pauli.pauli_arithmetic import PauliSentence
from pennylane.fermi import jordan_wigner, from_string
from .base_mapper import PennylaneBaseMapper
from ...second_q.fermionic_operator import FermionicOperator


class JordanWigner(PennylaneBaseMapper):
    
    def map(self, second_q_ops: FermionicOperator):
        qubit_ops = PauliSentence()
        wire_map = self.get_wire_map(second_q_ops.register_length)

        for op, coeff in second_q_ops.items():
            qubit_ops += coeff * jordan_wigner(
                                    from_string(self.reformat_str(op)), 
                                    wire_map=wire_map,
                                    ps=True)

        return self._return_data(qubit_ops)