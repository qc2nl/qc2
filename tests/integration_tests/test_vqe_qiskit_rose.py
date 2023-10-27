import subprocess
import pytest

from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit_algorithms.optimizers import SLSQP
from qiskit.primitives import Estimator

from qc2.ase import ROSE, ROSETargetMolecule, ROSEFragment
from qc2.data import qc2Data


def clean_up():
    """Remove Rose-ASE calculation outputs."""
    command = ("rm *.xyz *.dfcoef DFCOEF* *.inp INPUT* "
               "MOLECULE.XYZ MRCONEE* *dfpcmo DFPCMO* *.fchk "
               "fort.* timer.dat INFO_MOL *.pyscf *.psi4 "
               "*.npy *.clean OUTPUT_AVAS "
               "OUTPUT_* *.chk ILMO*dat *.out *.fcidump")
    subprocess.run(command, shell=True, capture_output=True)


@pytest.fixture(scope="session", autouse=True)
def clean_up_files():
    """Runs always at the end of all tests."""
    yield
    clean_up()


def rose_calculator():
    """Create an instance of ROSE calculator for water."""
    h2o = ROSETargetMolecule(
        name='water',
        atoms=[('O', (0.,  0.00000,  0.59372)),
               ('H', (0.,  0.76544, -0.00836)),
               ('H', (0., -0.76544, -0.00836))],
        basis='sto-3g'
    )

    oxygen = ROSEFragment(
        name='oxygen',
        atoms=[('O', (0, 0, 0))],
        multiplicity=1, basis='sto-3g'
    )

    hydrogen = ROSEFragment(
        name='hydrogen',
        atoms=[('H', (0, 0, 0))],
        multiplicity=2, basis='sto-3g'
    )

    h2o_calculator = ROSE(rose_calc_type='atom_frag',
                          exponent=4,
                          rose_target=h2o,
                          rose_frags=[oxygen, hydrogen],
                          test=True,
                          save_data=True,
                          restricted=True,
                          openshell=True,
                          rose_mo_calculator='pyscf')
    return h2o_calculator


@pytest.fixture
def vqe_calculation():
    """VQE using FCIDump schema."""
    # set the ROSE file to load
    fcidump_file = 'ibo.fcidump'

    # create an instance of qc2Data
    qc2data = qc2Data(fcidump_file, schema='fcidump')

    # attach the calculator
    qc2data.molecule.calc = rose_calculator()

    # run the calculator
    qc2data.run()

    # define active space
    n_active_electrons = (2, 2)  # => (n_alpha, n_beta)
    n_active_spatial_orbitals = 3

    # define the type of fermionic-to-qubit transformation
    mapper = JordanWignerMapper()

    # set up qubit Hamiltonian and core energy based on given active space
    e_core, qubit_op = qc2data.get_qubit_hamiltonian(
        n_active_electrons, n_active_spatial_orbitals, mapper, format='qiskit'
    )

    reference_state = HartreeFock(
        n_active_spatial_orbitals, n_active_electrons, mapper
    )

    ansatz = UCCSD(
        n_active_spatial_orbitals, n_active_electrons,
        mapper, initial_state=reference_state
    )

    vqe_solver = VQE(Estimator(), ansatz, SLSQP())
    vqe_solver.initial_point = [0.0] * ansatz.num_parameters
    result = vqe_solver.compute_minimum_eigenvalue(qubit_op)

    return result.eigenvalue, e_core


def test_vqe_calculation(vqe_calculation):
    """Check that the final vqe energy is correct."""
    calculated_electronic_energy, e_core = vqe_calculation
    calculated_energy = calculated_electronic_energy + e_core
    assert calculated_energy == pytest.approx(-75.1697111770602, rel=1e-6)


if __name__ == '__main__':
    pytest.main()
