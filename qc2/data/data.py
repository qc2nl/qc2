"""This module defines the main qc2 data class."""
from typing import Optional, Tuple, Union
import os
import h5py

from ase import Atoms
from ase.units import Ha

import qiskit_nature
from qiskit_nature.second_q.formats.qcschema import QCSchema
from qiskit_nature.second_q.formats import qcschema_to_problem
from qiskit_nature.second_q.mappers import QubitMapper, JordanWignerMapper
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit.quantum_info import SparsePauliOp

from pennylane.operation import Operator
from ..pennylane.convert import import_operator

from .schema import generate_empty_h5

# avoid using the deprecated `PauliSumOp` object
qiskit_nature.settings.use_pauli_sum_op = False


class qc2Data:
    """Main qc2 class.

    This class orchestrates classical qchem programs and
    python libraries for quantum computing.
    """
    def __init__(self,
                 filename: str,
                 molecule: Optional[Atoms],
                 ):
        """_summary_

        Args:
            filename (str): hdf5 file to save qchem and qc data.
            molecule (Optional[str]): `ase.atoms.Atoms` instance. 
        """
        # this version uses the JSON schema for quantum chemistry (QCSchema)
        # for more details, see https://molssi.org/software/qcschema-2/
        # 'qc_schema_output.schema' taken from
        # https://github.com/MolSSI/QCSchema/tree/master/qcschema/data/v2
        json_file = os.path.join(
            os.path.dirname(__file__), 'qc_schema_output.schema'
            )

        # define attributes
        self._schema = json_file
        self._filename = filename
        self._init_data_file()

        self._molecule = None
        self.molecule = molecule

    def _init_data_file(self):
        """Initializes empty hdf5 file following the QCSchema format."""
        generate_empty_h5(self._schema, self._filename)

    @property
    def molecule(self) -> Atoms:
        """Returs the molecule attribute.

        Returns:
            Molecule as an ASE Atoms object.
        """
        return self._molecule

    @molecule.setter
    def molecule(self, *args, **kwargs) -> None:
        """Sets the molecule attribute."""
        self._molecule = Atoms(*args, **kwargs)

    def run(self) -> None:
        """Runs ASE qchem calculator and save the data into hdf5 file."""
        if self._molecule is None or not isinstance(self._molecule, Atoms):
            raise ValueError(
                "No molecule is available for calculation."
                "Please, set this attribute as an"
                " `ase.atoms.Atoms` instance.")

        # run ase calculator
        reference_energy = self._molecule.get_potential_energy()/Ha
        print(f"Reference energy in Hartrees is: {reference_energy}")

        # dump required data to the hdf5 file
        self._molecule.calc.save(self._filename)
        print(f"Saving qchem data in {self._filename}")

    def get_fermionic_hamiltonian(self) -> Tuple[ElectronicEnergy,
                                                 FermionicOp]:
        """Builds the electronic Hamiltonian in second-quantization."""
        # open the HDF5 file
        with h5py.File(self._filename, 'r') as file:
            # read data and store it in a `QCSchema` instance;
            # see qiskit_nature/second_q/formats/qcschema/qc_schema.py
            qcschema = QCSchema._from_hdf5_group(file)

        # convert `QCSchema` into an instance of `ElectronicStructureProblem`;
        # see qiskit_nature/second_q/formats/qcschema_translator.py
        es_problem = qcschema_to_problem(qcschema, include_dipole=False)

        # convert `ElectronicStructureProblem`` into an instance of
        # `ElectronicEnergy` hamiltonian in second quantization;
        # see qiskit_nature/second_q/problems/electronic_structure_problem.py
        hamiltonian = es_problem.hamiltonian

        # now convert the `ElectronicEnergy` hamiltonian to a `FermionicOp`
        # instance
        # see qiskit_nature/second_q/hamiltonians/electronic_energy.py
        # and qiskit_nature/second_q/operators/fermionic_op.py
        second_q_op = hamiltonian.second_q_op()

        return es_problem, second_q_op

    def get_qubit_hamiltonian(self, mapper: QubitMapper = JordanWignerMapper(),
                              format: str = "qiskit") -> Union[SparsePauliOp,
                                                               Operator]:
        """Generates the qubit Hamiltonian of a target molecule."""

        if format not in ["qiskit", "pennylane"]:
            raise TypeError(f"Format {format} not yet suported.")

        second_q_op = self.get_fermionic_hamiltonian()[1]

        # build fermionic-to-qubit `SparsePauliOp` qiskit hamiltonian
        qubit_op = mapper.map(second_q_op)

        if format == "pennylane":
            # generate pennylane qubit hamiltonian `Operator` instance
            # from qiskit `SparsePauliOp`
            qubit_op = import_operator(qubit_op, "qiskit")

        return qubit_op

    def _get_active_space(self) -> None:
        """Defines active space for given # of active elec and active orb."""
        # see https://github.com/PennyLaneAI/pennylane/blob/master/pennylane/qchem/structure.py#L72
        pass
