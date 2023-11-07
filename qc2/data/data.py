"""This module defines the main qc2 data class."""
from typing import Tuple, Union
import os

from ase import Atoms
from ase.units import Ha

import qiskit_nature
from qiskit.quantum_info import SparsePauliOp

from qiskit_nature.second_q.formats.qcschema import QCSchema
from qiskit_nature.second_q.formats.fcidump import FCIDump
from qiskit_nature.second_q.formats import qcschema_to_problem
from qiskit_nature.second_q.formats import fcidump_to_problem
from qiskit_nature.second_q.mappers import QubitMapper, JordanWignerMapper
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer

from pennylane.operation import Operator
from qc2.pennylane.convert import import_operator

# avoid using the deprecated `PauliSumOp` object
qiskit_nature.settings.use_pauli_sum_op = False


class qc2Data:
    """Main qc2 class.

    This class orchestrates classical qchem programs and
    python libraries for quantum computing.

    Attributes:
        _schema (str): Format in which to save qchem data.
            Options are 'qcschema' or 'fcidump'. Defaults to 'qcschema'

        _filename (str): The path to the HDF5 file used to save qchem and
            quantum computing data.

        _molecule (Optional[Atoms]): An optional attribute representing the
            molecular structure as an `ase.atoms.Atoms` instance.
    """
    def __init__(
            self,
            filename: str,
            molecule: Atoms = Atoms(),
            *,
            schema: str = 'qcschema'
    ):
        """Initializes the `qc2Data` instance.

        Args:
            filename (str): The path to the data file to save qchem and
                quantum computing data.
            molecule (Optional[Atoms]): An optional `ase.atoms.Atoms` instance
                representing the target molecule.
            schema (Optional[str]): An optional attribute defining the format
                in which to save qchem data. Options are 'qcschema' or
                'fcidump'. Defaults to 'qcschema'.

        Example:
        >>> from qc2.data import qc2Data
        >>> from ase.build import molecule
        >>>
        >>> mol = molecule('H2')
        >>> hdf5_file = 'h2.hdf5'
        >>> qc2data = qc2Data(hdf5_file, mol, schema='qcschema')
        >>>
        >>> mol = molecule('H2')
        >>> fcidump_file = 'h2.fcidump'
        >>> qc2data = qc2Data(fcidump_file, mol, schema='fcidump')
        """
        # define attributes
        self._schema = schema
        self._filename = filename
        self._check_filename_extension()

        self._molecule = None
        self.molecule = molecule

    def _check_filename_extension(self) -> None:
        """Ensures that files have proper extensions."""
        # get file extension
        file_extension = os.path.splitext(self._filename)[1]

        # check extension
        if (self._schema == 'qcschema'
                and file_extension not in ['.hdf5', '.h5']):
            raise ValueError(
                f"{file_extension} is not a valid extension. "
                "For QCSchema format provide a file with "
                "*.hdf5 or *.h5 extensions."
            )

        if (self._schema == 'fcidump' and not file_extension == '.fcidump'):
            raise ValueError(
                f"{file_extension} is not a valid extension. "
                "For FCIDump format provide a file with "
                "*.fcidump extension."
            )

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
        """Runs ASE qchem calculator and saves the data into a formated file.

        Example:
        >>> from qc2.data import qc2Data
        >>> from ase.build import molecule
        >>>
        >>> mol = molecule('H2')
        >>> hdf5_file = 'h2.hdf5'
        >>> qc2data = qc2Data(hdf5_file, mol, schema='qcschema')
        >>> qc2data.molecule.calc = DIRAC(...)  # => specify qchem calculator
        >>> qc2data.run()
        >>>
        >>> mol = molecule('H2')
        >>> fcidump_file = 'h2.fcidump'
        >>> qc2data = qc2Data(fcidump_file, mol, schema='fcidump')
        >>> qc2data.molecule.calc = DIRAC(...)  # => specify qchem calculator
        >>> qc2data.run()
        """
        if self._molecule is None:
            raise ValueError(
                "No molecule is available for calculation."
                "Please, set this attribute as an"
                " `ase.atoms.Atoms` instance."
            )

        # run ase calculator
        reference_energy = self._molecule.get_potential_energy()/Ha
        print(f"* Reference energy (Hartree): {reference_energy}")

        # dump required data to the hdf5 or fcidump file
        self._molecule.calc.schema_format = self._schema
        self._molecule.calc.save(self._filename)
        print(f"* Saving qchem data in {self._filename}\n")

    def read_schema(self) -> Union[QCSchema, FCIDump]:
        """Reads data and stores it in `QCSchema` or `FCIDump` instance.

        Notes:
            see qiskit_nature/second_q/formats/qcschema/qc_schema.py
                qiskit_nature/second_q/formats/fcidump/fcidump.py
        """
        # read required data from the hdf5 or fcidump file
        self._molecule.calc.schema_format = self._schema
        return self._molecule.calc.load(self._filename)

    def process_schema(self) -> ElectronicStructureProblem:
        """Creates an instance of `ElectronicStructureProblem`."""
        # read data and store it in a `QCSchema` or `FCIDump`
        # dataclass instances
        schema = self.read_schema()

        if self._schema == 'fcidump':
            # convert `FCIDump` into `ElectronicStructureProblem`;
            # see qiskit_nature/second_q/formats/fcidump_translator.py
            return fcidump_to_problem(schema)

        # convert `QCSchema` into `ElectronicStructureProblem`;
        # see qiskit_nature/second_q/formats/qcschema_translator.py
        return qcschema_to_problem(schema, include_dipole=False)

    def get_active_space_hamiltonian(
            self,
            num_electrons: Union[int, Tuple[int, int]],
            num_spatial_orbitals: int
    ) -> Tuple[ElectronicStructureProblem, float, ElectronicEnergy]:
        """Builds the active-space reduced Hamiltonian."""
        # instantiate `ElectronicStructureProblem`
        es_problem = self.process_schema()

        # convert `ElectronicStructureProblem` into an instance of
        # `ElectronicEnergy` hamiltonian in second quantization;
        # see qiskit_nature/second_q/problems/electronic_structure_problem.py
        hamiltonian = es_problem.hamiltonian

        # in case of space selection, reduce the space extent of the
        # fermionic Hamiltonian based on the number of active electrons
        # and orbitals
        transformer = ActiveSpaceTransformer(num_electrons,
                                             num_spatial_orbitals)

        transformer.prepare_active_space(es_problem.num_particles,
                                         es_problem.num_spatial_orbitals)

        # after preparation, transform hamiltonian
        active_space_hamiltonian = transformer.transform_hamiltonian(
            hamiltonian)

        # just in case also generate a tranformed `ElectronicStructureProblem`
        # active_space_es_problem = transformer.transform(es_problem)

        # set up core energy after transformation
        nuclear_repulsion_energy = active_space_hamiltonian.constants[
            'nuclear_repulsion_energy']
        inactive_space_energy = active_space_hamiltonian.constants[
            'ActiveSpaceTransformer']
        core_energy = nuclear_repulsion_energy + inactive_space_energy

        return es_problem, core_energy, active_space_hamiltonian

    def get_fermionic_hamiltonian(
            self,
            num_electrons: Union[int, Tuple[int, int]],
            num_spatial_orbitals: int
    ) -> Tuple[ElectronicStructureProblem, float, FermionicOp]:
        """Builds the fermionic Hamiltonian of a target molecule.

        This method constructs the electronic Hamiltonian in 2nd-quantization
        based on the provided parameters.

        Args:
            num_electrons (Union[int, Tuple[int, int]]):
                The number of active electrons. If this is a tuple,
                it represents the number of alpha- and beta-spin electrons,
                respectively. If this is a number, it is interpreted as the
                total number of active electrons, should be even, and implies
                that the number of alpha and beta electrons equals half of
                this value, respectively.
            num_spatial_orbitals (int): The number of active orbitals.

        Returns:
            Tuple[float, ElectronicStructureProblem, FermionicOp]:
                A tuple containing the following elements:
                - core_energy (float): The core energy after active space
                    transformation.
                - es_problem (ElectronicStructureProblem): An instance of the
                    `ElectronicStructureProblem`.
                - second_q_op (FermionicOp): An instance of `FermionicOp`
                    representing the ferm. Hamiltonian in 2nd quantization.

        Raises:
            ValueError: If `num_electrons` or `num_spatial_orbitals` is None.

        Notes:
            Based on the qiskit-nature modules:
            qiskit_nature/second_q/problems/electronic_structure_problem.py
            qiskit_nature/second_q/transformers/active_space_transformer.py

        Example:
        >>> from qc2.data import qc2Data
        >>> from ase.build import molecule
        >>>
        >>> mol = molecule('H2')
        >>> hdf5_file = 'h2.hdf5'
        >>> qc2data = qc2Data(hdf5_file, mol, schema='qcschema')
        >>> qc2data.molecule.calc = DIRAC(...)  # => specify qchem calculator
        >>> qc2data.run()
        >>> n_electrons = (1, 1)
        >>> n_spatial_orbitals = 2
        >>> (es_prob, e_core, op) = qc2data.get_fermionic_hamiltonian(
        ...     n_electrons, n_spatial_orbitals
        ... )
        """
        if num_electrons is None:
            raise ValueError(
                "Number of active electrons cannot be 'None'."
                "Please, set the attribute 'num_electrons'.")

        if num_spatial_orbitals is None:
            raise ValueError(
                "Number of active orbitals cannot be 'None'."
                "Please, set the attribute 'num_spatial_orbitals'.")

        # calculate active space `ElectronicEnergy` hamiltonian
        (es_problem, core_energy,
         reduced_hamiltonian) = self.get_active_space_hamiltonian(
             num_electrons, num_spatial_orbitals
         )

        # now convert the reduced Hamiltonian (`Hamiltonian` instance)
        # into a `FermionicOp` instance
        # see qiskit_nature/second_q/hamiltonians/electronic_energy.py
        # and qiskit_nature/second_q/operators/fermionic_op.py
        second_q_op = reduced_hamiltonian.second_q_op()

        return es_problem, core_energy, second_q_op

    def get_qubit_hamiltonian(
            self,
            num_electrons: Union[int, Tuple[int, int]],
            num_spatial_orbitals: int,
            mapper: QubitMapper = JordanWignerMapper(),
            *,
            format: str = "qiskit"
    ) -> Tuple[float, Union[SparsePauliOp, Operator]]:
        """Generates the qubit Hamiltonian of a target molecule.

        This method generates the qubit Hamiltonian representation of a target
        molecule, which is essential for quantum algorithms related to qchem.

        Args:
            num_electrons (Union[int, Tuple[int, int]]):
                The number of active electrons. If this is a tuple,
                it represents the number of alpha- and beta-spin electrons,
                respectively. If this is a number, it is interpreted as the
                total number of active electrons, should be even, and implies
                that the number of alpha and beta electrons equals half of
                this value, respectively.
            num_spatial_orbitals (int): The number of active orbitals.
            mapper (QubitMapper, optional):
                The qubit mapping strategy to convert fermionic operators to
                qubit operators. Defaults to `JordanWignerMapper()`.
            format (str, optional):
                The format in which to return the qubit Hamiltonian.
                Supported formats are "qiskit" and "pennylane".
                Defaults to "qiskit".

        Returns:
            Tuple[float, Union[SparsePauliOp, Operator]]:
                A tuple containing the following elements:
                - core_energy (float): The core energy after after active
                    space transformation.
                - qubit_op (Union[SparsePauliOp, Operator]):
                  - If the format is "qiskit", it returns a `SparsePauliOp`
                  representing the qubit Hamiltonian in the qiskit format.
                  - If the format is "pennylane", it returns a `Operator`
                  instance representing the qubit Hamiltonian in the
                  PennyLane format.

        Raises:
            TypeError: If the provided `format` is not supported (not "qiskit"
            nor "pennylane").

        Example:
        >>> from qc2.data import qc2Data
        >>> from ase.build import molecule
        >>>
        >>> mol = molecule('H2')
        >>> hdf5_file = 'h2.hdf5'
        >>> qc2data = qc2Data(hdf5_file, mol, schema='qcschema')
        >>> qc2data.molecule.calc = DIRAC(...)  # => specify qchem calculator
        >>> qc2data.run()
        >>> n_electrons = (1, 1)
        >>> n_spatial_orbitals = 2
        >>> mapper = BravyiKitaevMapper()
        >>> (e_core, qubit_op) = qc2data.get_qubit_hamiltonian(
        ...     n_electrons, n_spatial_orbitals, mapper, format='qiskit'
        ... )
        """
        if format not in ["qiskit", "pennylane"]:
            raise TypeError(f"Format {format} not yet suported.")

        # get fermionic hamiltonian
        _, core_energy, second_q_op = self.get_fermionic_hamiltonian(
            num_electrons, num_spatial_orbitals)

        # perform fermionic-to-qubit transformation using the given mapper
        # and obtain `SparsePauliOp` qiskit qubit hamiltonian
        qubit_op = mapper.map(second_q_op)

        if format == "pennylane":
            # generate pennylane qubit hamiltonian `Operator` instance
            # from qiskit `SparsePauliOp`;
            # see qc2/pennylane/convert.py
            qubit_op = import_operator(qubit_op, format="qiskit")

        return core_energy, qubit_op
