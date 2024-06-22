"""This module defines the main qc2 data class."""
from typing import Tuple, Union, Optional
import os
import numpy as np

from ase import Atoms
from ase.units import Ha

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
from qiskit_nature.second_q.transformers import BasisTransformer
from qiskit_nature.second_q.problems import ElectronicBasis
from qiskit_nature.second_q.operators import ElectronicIntegrals

# try importing PennyLane and set `PennyLaneOperatorType`
try:
    from pennylane.operation import Operator
    from qc2.pennylane.convert import import_operator
    PennyLaneOperatorType = Operator
except ImportError:
    PennyLaneOperatorType = object

from qc2.algorithms.base.base_algorithm import BaseAlgorithm
from qc2.ase.qc2_ase_base_class import BaseQc2ASECalculator


class qc2Data:
    """Main qc2 class.

    This class orchestrates classical qchem programs and
    python libraries for quantum computing.

    Attributes:
        _schema (str): Format in which to save qchem data.
            Options are ``qcschema`` or ``fcidump``.
            Defaults to ``qcschema``.

        filename (str): The path to the HDF5 or fcidump file used
            to save/read qchem data.

        molecule (Atoms): Attribute representing the
            molecular structure as an ASE :class:`ase.atoms.Atoms` instance.

        algorithm (BaseAlgorithm): Instance of the algorithm to be run.
            Examples are :class:`~qc2.algorithm.qiskit.vqe.VQE` and
            :class:`~qc2.algorithm.pennylane.oo_vqe.oo_VQE`.
    """

    def __init__(
        self,
        filename: str,
        molecule: Atoms = Atoms(),
        algorithm: BaseAlgorithm = BaseAlgorithm(),
        *,
        schema: str = "qcschema",
    ):
        """Initializes the ``qc2Data`` class instance.

        Args:
            filename (str): The path to the data file to save/read qchem
                data.
            molecule (Atoms): An optional :class:`ase.atoms.Atoms`
                instance representing the target molecule.
            algorithm (BaseAlgorithm): Algorithm to be run.
                Examples are :class:`~qc2.algorithm.qiskit.vqe.VQE` and
                :class:`~qc2.algorithm.pennylane.oo_vqe.oo_VQE`.
            schema (Optional[str]): An optional attribute defining the format
                in which to save qchem data. Options are ``qcschema`` or
                ``fcidump``. Defaults to ``qcschema``.

        **Example**

        >>> from ase.build import molecule
        >>> from qc2.data import qc2Data
        >>> from qc2.ase import PySCF
        >>> from qc2.algorithms.utils import ActiveSpace
        >>> from qc2.algorithm.qiskit import VQE
        >>>
        >>> mol = molecule('H2')
        >>>
        >>> hdf5_file = 'h2.hdf5'
        >>> qc2data = qc2Data(hdf5_file, mol, schema='qcschema')
        >>> qc2data.molecule.calc = PySCF()
        >>> qc2data.algorithm = VQE(
        ...     active_space=ActiveSpace(
        ...         num_active_electrons=(1, 1),
        ...         num_active_spatial_orbitals=2
        ...     ),
        ... )
        >>> qc2data.run()            # => run classical qc2-ASE calculator
        >>> qc2data.algorithm.run()  # => run quantum algorithm
        """
        # define attributes
        self._schema = schema
        self._filename = filename
        self._check_filename_extension()

        self._molecule = None
        self.molecule = molecule

        self._algorithm = None
        self.algorithm = algorithm

    @property
    def molecule(self) -> Atoms:
        """Returns the molecule attribute.

        Returns:
            Molecule as an ASE :class:`ase.atoms.Atoms` object.
        """
        return self._molecule

    @molecule.setter
    def molecule(self, *args, **kwargs) -> None:
        """Sets the molecule attribute."""
        self._molecule = Atoms(*args, **kwargs)

    @property
    def algorithm(self) -> BaseAlgorithm:
        """Returns the chosen algorithm.

        Returns:
            Instance of an algorithm class, *e.g.*, VQE.
        """
        return self._algorithm

    @algorithm.setter
    def algorithm(self, algorithm: BaseAlgorithm) -> None:
        """Sets the algorithm attribute."""
        self._algorithm = algorithm
        if hasattr(algorithm, "set_qc2data"):
            algorithm.set_qc2data(self)
        else:
            raise ValueError("{} can't set qc2data".format(algorithm.__name__))

    def _check_filename_extension(self) -> None:
        """Ensures that files have proper extensions."""
        # get file extension
        file_extension = os.path.splitext(self._filename)[1]

        # check extension
        if self._schema == "qcschema" and file_extension not in [".hdf5", ".h5"]:
            raise ValueError(
                f"{file_extension} is not a valid extension. "
                "For QCSchema format provide a file with "
                "*.hdf5 or *.h5 extensions."
            )

        if self._schema == "fcidump" and not file_extension == ".fcidump":
            raise ValueError(
                f"{file_extension} is not a valid extension. "
                "For FCIDump format provide a file with "
                "*.fcidump extension."
            )

    def run(self) -> None:
        """Runs ASE qchem calculator and saves the data into a formated file.

        Returns:
            None

        **Example**

        >>> from ase.build import molecule
        >>> from qc2.ase import DIRAC
        >>> from qc2.data import qc2Data
        >>>
        >>> mol = molecule('H2')
        >>>
        >>> hdf5_file = 'h2.hdf5'
        >>> qc2data = qc2Data(hdf5_file, mol, schema='qcschema')
        >>> qc2data.molecule.calc = DIRAC(...)  # => specify qchem calculator
        >>> qc2data.run()
        >>>
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
        reference_energy = self._molecule.get_potential_energy() / Ha
        print(f"* Reference energy (Hartree): {reference_energy}")

        # dump required data to the hdf5 or fcidump file
        self._molecule.calc.schema_format = self._schema
        self._molecule.calc.save(self._filename)
        print(f"* Saving qchem data in {self._filename}\n")

    def read_schema(self) -> Union[QCSchema, FCIDump]:
        """Reads and stores data in :class:`QCSchema` or :class:`FCIDump`.

        Reads and stores the required data from an HDF5 or FCIDump file as
        either a :class:`QCSchema` or :class:`FCIDump` dataclass instance.

        Returns:
            Union[QCSchema, FCIDump]:
                Instance of :class:`QCSchema` or :class:`FCIDump` dataclass.

        Notes:
            See qiskit_nature/second_q/formats for more information on the
            supported data formats.

        **Example**

        >>> from ase.build import molecule
        >>> from qc2.ase import DIRAC
        >>> from qc2.data import qc2Data
        >>>
        >>> mol = molecule('H2')
        >>>
        >>> hdf5_file = 'h2.hdf5'
        >>> qc2data = qc2Data(hdf5_file, mol, schema='qcschema')
        >>> qc2data.molecule.calc = DIRAC(...)  # => specify qchem calculator
        >>> qc2data.run()
        >>> qcschema = qc2data.read_schema()
        >>>
        >>> fcidump_file = 'h2.fcidump'
        >>> qc2data = qc2Data(fcidump_file, mol, schema='fcidump')
        >>> qc2data.molecule.calc = DIRAC(...)  # => specify qchem calculator
        >>> qc2data.run()
        >>> fcidump = qc2data.read_schema()
        """
        # create a generic calculator
        self._molecule.calc = BaseQc2ASECalculator()

        # read required data from the hdf5 or fcidump file
        self._molecule.calc.schema_format = self._schema
        return self._molecule.calc.load(self._filename)

    def process_schema(
            self,
            *,
            basis: str = "molecular"
    ) -> ElectronicStructureProblem:
        """Creates an instance of :class:`ElectronicStructureProblem`.

        Reads data using the :meth:`~.read_schema` method and converts it into
        an instance of :class:`ElectronicStructureProblem` based on the
        specified schema format (``fcidump`` or ``qcschema``) and
        electronic basis as defined by :class:`qiskit.ElectronicBasis`.

        Args:
            basis (str, optional): The basis in which to construct
                the :class:`ElectronicStructureProblem`. Options are ``atomic``
                or ``molecular``. Defaults to ``molecular``.

        Returns:
            ElectronicStructureProblem:
              An instance representing
              the :class:`ElectronicStructureProblem`.

        Notes:
            - For ``fcidump`` schema, the conversion is done using the
              `fcidump_to_problem` function from
              qiskit_nature/second_q/formats/fcidump_translator.py.
            - For ``qcschema`` schema, the conversion is done using the
              `qcschema_to_problem` function from
              qiskit_nature/second_q/formats/qcschema_translator.py.
            - Dipoles are excluded when converting `QCSchema` data.

        **Example**

        >>> from ase.build import molecule
        >>> from qc2.ase import DIRAC
        >>> from qc2.data import qc2Data
        >>>
        >>> mol = molecule('H2')
        >>>
        >>> hdf5_file = 'h2.hdf5'
        >>> qc2data = qc2Data(hdf5_file, mol, schema='qcschema')
        >>> qc2data.molecule.calc = DIRAC(...)  # => specify qchem calculator
        >>> qc2data.run()
        >>> es_problem = qc2data.process_schema(basis='atomic')
        >>>
        >>> fcidump_file = 'h2.fcidump'
        >>> qc2data = qc2Data(fcidump_file, mol, schema='fcidump')
        >>> qc2data.molecule.calc = DIRAC(...)  # => specify qchem calculator
        >>> qc2data.run()
        >>> es_problem = qc2data.process_schema()
        """
        # read data and store it in a `QCSchema` or `FCIDump`
        # dataclass instances
        schema = self.read_schema()

        # convert electronic basis from string to an instance
        # of :class:`qiskit.ElectronicBasis`
        basis = ElectronicBasis(basis)

        if self._schema == "fcidump":
            # convert `FCIDump` into `ElectronicStructureProblem`;
            # see qiskit_nature/second_q/formats/fcidump_translator.py
            return fcidump_to_problem(schema)

        # convert `QCSchema` into `ElectronicStructureProblem`;
        # see qiskit_nature/second_q/formats/qcschema_translator.py
        return qcschema_to_problem(schema, include_dipole=False, basis=basis)

    def get_active_space_hamiltonian(
            self,
            num_electrons: Union[int, Tuple[int, int]],
            num_spatial_orbitals: int,
            *,
            initial_es_problem: Optional[ElectronicStructureProblem] = None,
    ) -> Tuple[ElectronicStructureProblem, float, ElectronicEnergy]:
        """Builds the active-space reduced Hamiltonian.

        Args:
            num_electrons (Union[int, Tuple[int, int]]): The number of active
                electrons. If a tuple is provided, it represents alpha and
                beta active electrons.
            num_spatial_orbitals (int): The number of spatial orbitals.
            initial_es_problem (Optional[ElectronicStructureProblem]):
                  Initial instance of :class:`ElectronicStructureProblem`.
                  If None, it is instantiated internally. Defaults to None.

        Returns:
            Tuple[ElectronicStructureProblem, float, ElectronicEnergy]:
                - active_space_es_problem (ElectronicStructureProblem): final
                  active space transformed :class:`ElectronicStructureProblem`.
                - core_energy (float): The core energy, which includes the
                  nuclear repulsion energy and the energy of inactive orbitals.
                - active_space_hamiltonian (ElectronicEnergy):
                  Instance of :class:`ElectronicEnergy`,
                  the active-space reduced Hamiltonian.

        Notes:
            - The active-space reduced Hamiltonian is obtained by transforming
              the original electronic structure problem's Hamiltonian using
              an ActiveSpaceTransformer.
            - The core energy is computed as the sum of the nuclear repulsion
              energy and the energy of inactive orbitals.

        **Example**

        >>> from ase.build import molecule
        >>> from qc2.ase import DIRAC
        >>> from qc2.data import qc2Data
        >>>
        >>> mol = molecule('H2')
        >>> hdf5_file = 'h2.hdf5'
        >>> qc2data = qc2Data(hdf5_file, mol, schema='qcschema')
        >>> qc2data.molecule.calc = DIRAC(...)  # => specify qchem calculator
        >>> qc2data.run()
        >>> n_electrons = (1, 1)
        >>> n_spatial_orbitals = 2
        >>> (es_problem, e_core, ham) = qc2data.get_active_space_hamiltonian(
        ...     n_electrons, n_spatial_orbitals
        ... )
        """
        if initial_es_problem is None:
            # instantiate `ElectronicStructureProblem`
            es_problem = self.process_schema()
        else:
            es_problem = initial_es_problem

        # convert `ElectronicStructureProblem` into an instance of
        # `ElectronicEnergy` hamiltonian in second quantization;
        # see qiskit_nature/second_q/problems/electronic_structure_problem.py
        hamiltonian = es_problem.hamiltonian

        # in case of space selection, reduce the space extent of the
        # fermionic Hamiltonian based on the number of active electrons
        # and orbitals
        transformer = ActiveSpaceTransformer(
            num_electrons, num_spatial_orbitals
        )

        transformer.prepare_active_space(
            es_problem.num_particles, es_problem.num_spatial_orbitals
        )

        # after preparation, transform hamiltonian
        active_space_hamiltonian = transformer.transform_hamiltonian(
            hamiltonian
        )

        # just in case also generate a tranformed `ElectronicStructureProblem`
        active_space_es_problem = transformer.transform(es_problem)

        # set up core energy after transformation
        nuclear_repulsion_energy = active_space_hamiltonian.constants[
            "nuclear_repulsion_energy"
        ]
        inactive_space_energy = active_space_hamiltonian.constants[
            "ActiveSpaceTransformer"
        ]
        core_energy = nuclear_repulsion_energy + inactive_space_energy

        return active_space_es_problem, core_energy, active_space_hamiltonian

    def get_transformed_hamiltonian(
            self,
            *,
            initial_es_problem: ElectronicStructureProblem,
            matrix_transform_a: np.ndarray,
            matrix_transform_b: Optional[np.ndarray] = None,
            initial_basis: str = "atomic",
            final_basis: str = "molecular",
    ) -> Tuple[ElectronicStructureProblem, ElectronicEnergy]:
        """Transforms the Hamiltonian from one basis set to another.

        Args:
            initial_es_problem (ElectronicStructureProblem):
                The original electronic structure problem.
            matrix_transform_a (np.ndarray): The transformation matrix
                for alpha spin orbitals.
            matrix_transform_b (np.ndarray, optional): The transformation
                matrix for beta spin orbitals.
            initial_basis (str, optional): The initial basis set.
                Defaults to ``atomic``.
            final_basis (str, optional): The final basis set to transform to.
                Defaults to ``molecular``.

        Returns:
            Tuple[ElectronicStructureProblem, ElectronicEnergy]:
                - transformed_es_problem (ElectronicStructureProblem):
                  An instance of the transformed
                  :class:`ElectronicStructureProblem`.
                - transformed_hamiltonian (ElectronicEnergy):
                  An instance of :class:`ElectronicEnergy`,
                  the transformed Hamiltonian.

        **Example**

        >>> from ase.build import molecule
        >>> from qc2.ase import PySCF
        >>> from qc2.data import qc2Data
        >>>
        >>> mol = molecule('H2')
        >>> hdf5_file = 'h2.hdf5'
        >>> qc2data = qc2Data(hdf5_file, mol, schema='qcschema')
        >>> qc2data.molecule.calc = PySCF(...)  # => specify qchem calculator
        >>> qc2data.run()
        >>> ao_es_problem = qc2data.process_schema(basis='atomic')
        >>> mo_coeff_a = np.array(
        ...     [[0.54884228,  1.21245192],
        ...      [0.54884228, -1.21245192]]
        ... )
        >>> mo_es_problem, hamiltonian = qc2data.get_transformed_hamiltonian(
        ...     initial_es_problem=ao_es_problem,
        ...     matrix_transform_a=mo_coeff_a,
        ...     initial_basis="atomic",
        ...     final_basis="molecular"
        ... )
        """
        initial_basis = ElectronicBasis(initial_basis)
        final_basis = ElectronicBasis(final_basis)

        if matrix_transform_b is None:
            matrix_transform_b = matrix_transform_a

        # create an instance of `BasisTransformer`
        transformer = BasisTransformer(
            initial_basis,
            final_basis,
            ElectronicIntegrals.from_raw_integrals(
                matrix_transform_a, h1_b=matrix_transform_b
            )
        )
        # transform the original `ElectronicStructureProblem`
        # from the original to the new basis
        # first, collect integrals
        alpha = initial_es_problem.hamiltonian.electronic_integrals.alpha
        beta = initial_es_problem.hamiltonian.electronic_integrals.beta
        beta_alpha = (
            initial_es_problem.hamiltonian.electronic_integrals.beta_alpha
        )

        # transform initial integrals to the new basis
        integrals_initial_basis = ElectronicIntegrals(alpha, beta, beta_alpha)
        integrals_final_basis = transformer.transform_electronic_integrals(
            integrals_initial_basis
        )
        # build up the new `ElectronicEnergy` hamiltonian
        transformed_hamiltonian = ElectronicEnergy(integrals_final_basis)
        nuc_rep_energy = (
            initial_es_problem.hamiltonian.nuclear_repulsion_energy
        )
        transformed_hamiltonian.nuclear_repulsion_energy = nuc_rep_energy

        # build up the new `ElectronicStructureProblem`
        transformed_es_problem = ElectronicStructureProblem(
            transformed_hamiltonian
        )
        transformed_es_problem.basis = final_basis
        transformed_es_problem.molecule = initial_es_problem.molecule
        transformed_es_problem.reference_energy = (
            initial_es_problem.reference_energy
        )
        transformed_es_problem.num_particles = initial_es_problem.num_particles
        transformed_es_problem.num_spatial_orbitals = (
            initial_es_problem.num_spatial_orbitals
        )

        return transformed_es_problem, transformed_hamiltonian

    def get_fermionic_hamiltonian(
            self,
            num_electrons: Union[int, Tuple[int, int]],
            num_spatial_orbitals: int,
            *,
            transform: bool = False,
            initial_es_problem: Optional[ElectronicStructureProblem] = None,
            matrix_transform_a: Optional[np.ndarray] = None,
            matrix_transform_b: Optional[np.ndarray] = None,
            initial_basis: str = "atomic",
            final_basis: str = "molecular",
    ) -> Tuple[ElectronicStructureProblem, float, FermionicOp]:
        """Builds the fermionic Hamiltonian of a target molecule.

        This method constructs the electronic Hamiltonian in 2nd-quantization
        based on the provided parameters. It can optionally perform a basis set
        transformation if the `transform` flag is set to True.

        Args:
            num_electrons (Union[int, Tuple[int, int]]):
                The number of active electrons. If this is a tuple,
                it represents the number of alpha- and beta-spin electrons,
                respectively. If this is a number, it is interpreted as the
                total number of active electrons, should be even, and implies
                that the number of alpha and beta electrons equals half of
                this value, respectively.
            num_spatial_orbitals (int): The number of active orbitals.
            transform (bool, optional): If True, performs a basis
                transformation. Defaults to False.
            initial_es_problem (ElectronicStructureProblem, optional):
                The initial electronic structure problem.
                Required if `transform` is True. Defaults to None.
            matrix_transform_a (np.ndarray, optional): Transformation
                matrix for alpha spin orbitals. Required if `transform`
                is True. Defaults to None.
            matrix_transform_b (np.ndarray, optional): Transformation
                matrix for beta spin orbitals. Required if `transform`
                is True. Defaults to None.
            initial_basis (str, optional): The initial basis set. Defaults to
                "atomic".
            final_basis (str, optional): The final basis set to transform to.
                Defaults to "molecular".

        Returns:
            Tuple[float, ElectronicStructureProblem, FermionicOp]:
                - core_energy (float): The core energy after active space
                  transformation.
                - es_problem (ElectronicStructureProblem): An instance of the
                  :class:`ElectronicStructureProblem`.
                - second_q_op (FermionicOp): An instance of
                  :class:`FermionicOp` representing the fermionic Hamiltonian
                  in 2nd quantization.

        Raises:
            ValueError: If :attr:`num_electrons` or
                :attr:`num_spatial_orbitals` is None.
                Or If :attr:`initial_es_problem` is None
                for :attr:`transform` equal True.

        Notes:
            Based on the qiskit-nature modules:
            qiskit_nature/second_q/problems/electronic_structure_problem.py
            qiskit_nature/second_q/transformers/active_space_transformer.py

        **Example**

        >>> from ase.build import molecule
        >>> from qc2.ase import DIRAC
        >>> from qc2.data import qc2Data
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
                "Please, set the attribute 'num_electrons'."
            )

        if num_spatial_orbitals is None:
            raise ValueError(
                "Number of active orbitals cannot be 'None'."
                "Please, set the attribute 'num_spatial_orbitals'."
            )

        transformed_es_problem = None
        if transform is True:
            if initial_es_problem is not None:
                # Transform `ElectronicStructureProblem` to a new basis
                transformed_es_problem, _ = self.get_transformed_hamiltonian(
                    initial_es_problem=initial_es_problem,
                    matrix_transform_a=matrix_transform_a,
                    matrix_transform_b=matrix_transform_b,
                    initial_basis=initial_basis,
                    final_basis=final_basis
                )
            else:
                raise ValueError(
                    "The initial value of 'initial_es_problem' cannot be "
                    "'None' if 'transform' is 'True'."
                )

        # calculate active space `ElectronicEnergy` hamiltonian
        (es_problem, core_energy,
         reduced_hamiltonian) = self.get_active_space_hamiltonian(
             num_electrons,
             num_spatial_orbitals,
             initial_es_problem=transformed_es_problem
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
            format: str = "qiskit",
            transform: bool = False,
            initial_es_problem: Optional[ElectronicStructureProblem] = None,
            matrix_transform_a: Optional[np.ndarray] = None,
            matrix_transform_b: Optional[np.ndarray] = None,
            initial_basis: str = "atomic",
            final_basis: str = "molecular",
    ) -> Tuple[float, Union[SparsePauliOp, PennyLaneOperatorType]]:
        """Generates the qubit Hamiltonian of a target molecule.

        This method generates the qubit Hamiltonian representation of a target
        molecule. It can optionally perform a basis set transformation if the
        ``transform`` flag is True.

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
                qubit operators. Defaults to ``JordanWignerMapper()``.
            format (str, optional):
                The format in which to return the qubit Hamiltonian.
                Supported formats are ``qiskit`` and ``pennylane``.
                Defaults to ``qiskit``.
            transform (bool, optional): If True, performs a basis
                transformation. Defaults to False.
            initial_es_problem (ElectronicStructureProblem, optional): The
                initial electronic structure problem.
                Required if `transform` is True. Defaults to None.
            matrix_transform_a (np.ndarray, optional): Transformation
                matrix for alpha spin orbitals. Required if `transform`
                is True. Defaults to None.
            matrix_transform_b (np.ndarray, optional): Transformation matrix
                for beta spin orbitals. Required if `transform` is True.
                Defaults to None.
            initial_basis (str, optional): The initial basis set.
                Defaults to ``atomic``.
            final_basis (str, optional): The final basis set to transform to.
                Defaults to ``molecular``.

        Returns:
            Tuple[float, Union[SparsePauliOp, Operator]]:
                - core_energy (float): The core energy after active
                  space transformation.
                - qubit_op (Union[SparsePauliOp, Operator]):
                  If the format is ``qiskit``, it returns a
                  :class:`SparsePauliOp` representing the
                  qubit Hamiltonian in the qiskit format.
                  If the format is ``pennylane``, it returns a
                  :class:`Operator` instance representing the
                  qubit Hamiltonian in the PennyLane format.

        Raises:
            TypeError: If the provided `format` is not supported
              (not ``qiskit`` nor ``pennylane``).

        **Example**

        >>> from ase.build import molecule
        >>> from qc2.ase import DIRAC
        >>> from qc2.data import qc2Data
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
            num_electrons,
            num_spatial_orbitals,
            transform=transform,
            initial_es_problem=initial_es_problem,
            matrix_transform_a=matrix_transform_a,
            matrix_transform_b=matrix_transform_b,
            initial_basis=initial_basis,
            final_basis=final_basis
        )

        # perform fermionic-to-qubit transformation using the given mapper
        # and obtain `SparsePauliOp` qiskit qubit hamiltonian
        qubit_op = mapper.map(second_q_op)

        if format == "pennylane":
            # generate pennylane qubit hamiltonian `Operator` instance
            # from qiskit `SparsePauliOp`;
            # see qc2/pennylane/convert.py
            qubit_op = import_operator(qubit_op, format="qiskit")

        return core_energy, qubit_op
