"""This module defines the main qc2 data class."""
from typing import Optional, Tuple, Union
import os
import h5py

from ase import Atoms
from ase.units import Ha

import qiskit_nature
from qiskit.quantum_info import SparsePauliOp

from qiskit_nature.second_q.formats.qcschema import QCSchema
from qiskit_nature.second_q.formats import qcschema_to_problem
from qiskit_nature.second_q.mappers import QubitMapper, JordanWignerMapper
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer

from pennylane.operation import Operator
from ..pennylane.convert import import_operator

from .schema import generate_empty_h5

# avoid using the deprecated `PauliSumOp` object
qiskit_nature.settings.use_pauli_sum_op = False


class qc2Data:
    """Main qc2 class.

    This class orchestrates classical qchem programs and
    python libraries for quantum computing.

    Attributes:
        _schema (str): The path to the JSON schema file for quantum chemistry
            (QCSchema).
            For more details, see https://molssi.org/software/qcschema-2/.
            The 'qc_schema_output.schema' is taken from
            https://github.com/MolSSI/QCSchema/tree/master/qcschema/data/v2.

        _filename (str): The path to the HDF5 file used to save qchem and
            quantum computing data.

        _molecule (Optional[Atoms]): An optional attribute representing the
            molecular structure as an `ase.atoms.Atoms` instance.
    """
    def __init__(self,
                 filename: str,
                 molecule: Optional[Atoms],
                 ):
        """Initializes the `qc2Data` instance.

        Args:
            filename (str): The path to the HDF5 file to save qchem and
                quantum computing data.
            molecule (Optional[Atoms]): An optional `ase.atoms.Atoms` instance
                representing the target molecule.

        Example:
        >>> from qc2.data import qc2Data
        >>> from ase.build import molecule
        >>>
        >>> mol = molecule('H2')
        >>> hdf5_file = 'h2.hdf5'
        >>> qc2data = qc2Data(hdf5_file, mol)
        """
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
        """Runs ASE qchem calculator and saves the data into hdf5 file.

        Example:
        >>> from qc2.data import qc2Data
        >>> from ase.build import molecule
        >>>
        >>> mol = molecule('H2')
        >>> hdf5_file = 'h2.hdf5'
        >>> qc2data = qc2Data(hdf5_file, mol)
        >>> qc2data.molecule.calc = DIRAC(...)  # => specify qchem calculator
        >>> qc2data.run()
        """
        if self._molecule is None:
            raise ValueError(
                "No molecule is available for calculation."
                "Please, set this attribute as an"
                " `ase.atoms.Atoms` instance.")

        # run ase calculator
        reference_energy = self._molecule.get_potential_energy()/Ha
        print(f"* Reference energy (Hartree): {reference_energy}")

        # dump required data to the hdf5 file
        self._molecule.calc.save(self._filename)
        print(f"* Saving qchem data in {self._filename}\n")

    def get_fermionic_hamiltonian(self,
                                  num_electrons: Union[int, Tuple[int, int]],
                                  num_spatial_orbitals: int
                                  ) -> Tuple[float,
                                             ElectronicEnergy, FermionicOp]:
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
        >>> qc2data = qc2Data(hdf5_file, mol)
        >>> qc2data.molecule.calc = DIRAC(...)  # => specify qchem calculator
        >>> qc2data.run()
        >>> n_electrons = (1, 1)
        >>> n_spatial_orbitals = 2
        >>> (e_core, es_prob, op) = qc2data.get_fermionic_hamiltonian(
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

        # in case of space selection, reduce the space extent of the
        # fermionic Hamiltonian based on the number of active electrons
        # and orbitals
        transformer = ActiveSpaceTransformer(num_electrons,
                                             num_spatial_orbitals)

        transformer.prepare_active_space(es_problem.num_particles,
                                         es_problem.num_spatial_orbitals)

        # after preparation, transform hamiltonian
        reduced_hamiltonian = transformer.transform_hamiltonian(hamiltonian)

        # set up core energy after transformation
        nuclear_repulsion_energy = reduced_hamiltonian.constants[
            'nuclear_repulsion_energy']
        inactive_space_energy = reduced_hamiltonian.constants[
            'ActiveSpaceTransformer']
        core_energy = nuclear_repulsion_energy + inactive_space_energy

        # now convert the reduced Hamiltonian (`Hamiltonian` instance)
        # into a `FermionicOp` instance
        # see qiskit_nature/second_q/hamiltonians/electronic_energy.py
        # and qiskit_nature/second_q/operators/fermionic_op.py
        second_q_op = reduced_hamiltonian.second_q_op()

        return core_energy, es_problem, second_q_op

    def get_qubit_hamiltonian(self,
                              num_electrons: Union[int, Tuple[int, int]],
                              num_spatial_orbitals: int,
                              mapper: QubitMapper = JordanWignerMapper(),
                              *,
                              format: str = "qiskit"
                              ) -> Tuple[float,
                                         Union[SparsePauliOp, Operator]]:
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
        >>> qc2data = qc2Data(hdf5_file, mol)
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
        core_energy, _, second_q_op = self.get_fermionic_hamiltonian(
            num_electrons, num_spatial_orbitals)

        # perform fermionic-to-qubit transformation using the given mapper
        # and obtain `SparsePauliOp` qiskit qubit hamiltonian
        qubit_op = mapper.map(second_q_op)

        if format == "pennylane":
            # generate pennylane qubit hamiltonian `Operator` instance
            # from qiskit `SparsePauliOp`
            # see qc2/pennylane/convert.py
            qubit_op = import_operator(qubit_op, format="qiskit")

        return core_energy, qubit_op
