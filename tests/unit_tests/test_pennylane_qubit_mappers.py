import os
import pytest
from typing import List
from qc2.qc2_driver import QC2
from ase.build import molecule
from qc2.ase.pyscf import PySCF
from qc2.algorithms.second_q.second_quantizer import SecondQuantizer
from qc2.algorithms.second_q.active_space import ActiveSpace
from qc2.algorithms.pennylane.qubit_mappers.jordan_wigner import JordanWigner
from pennylane.ops.op_math.sum import Sum

@pytest.fixture
def qc2data():
    """Fixture to set up qc2Data instance."""
    tmp_filename = str('test_qc2data.h5')
    mol = molecule('H2')
    qc2_data = QC2(tmp_filename, mol, schema='qcschema')
    qc2_data.molecule.calc = PySCF()
    yield qc2_data
    os.remove(tmp_filename)

@pytest.fixture
def active_space():
    return ActiveSpace((1, 1), 2)

@pytest.fixture
def second_q_op(qc2data, active_space):
    qc2data.run()
    second_quantizer = SecondQuantizer(qc2data)
    _, second_q_op = second_quantizer.get_fermionic_hamiltonian(
        active_space.num_active_electrons,
        active_space.num_active_spatial_orbitals
    )
    return second_q_op
@pytest.mark.parametrize("mapper", [JordanWigner()])
def test_mappers_map(second_q_op, mapper):
    qubit_op = mapper.map(second_q_op)
    assert isinstance(qubit_op, Sum)

@pytest.mark.parametrize("mapper", [JordanWigner()])
def test_mappers_representation(second_q_op, mapper):
    coeffs, paulis = mapper.get_representation(mapper.map(second_q_op))
    assert isinstance(coeffs, List)
    assert isinstance(paulis, List)