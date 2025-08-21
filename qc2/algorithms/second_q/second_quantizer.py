from typing import Tuple, Union, Any
from ...qc2_driver import QC2
from ...data.qcschema import QCSchema
from .electronic_hamiltonian import ElectronicHamiltonian
from .active_space_transformer import ActiveSpaceTransformer
from .fermionic_operator import FermionicOperator
from ..base.qc2_qubit_mapper_base_class import BaseMapper


class SecondQuantizer:

    def __init__(self, 
                 qc2data: QC2,
                 ):
        
        self.qc2data = qc2data
        self.schema_data = None

    def read_data(self):
        """read the data conained in the schema if necessary."""
        if self.schema_data is None:
            self.schema_data =  self.qc2data.read_schema()

    def get_active_space_hamiltonian(
            self,
            num_electrons: Union[int, Tuple[int, int]],
            num_spatial_orbitals: int,
            initial_hamiltonian: ElectronicHamiltonian | None = None
    ) -> Tuple[float, ElectronicHamiltonian]:
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
        >>> from qc2.qc2_driver import QC2 as qc2Data
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

        if initial_hamiltonian is None:
            # create the initial ElectronicHamiltonian
            self.read_data()
            hamiltonian = ElectronicHamiltonian(schema=self.schema_data,  tol=1E-5)
        else:
            hamiltonian = initial_hamiltonian

        return _get_active_space_hamiltonian(
            num_electrons, num_spatial_orbitals, hamiltonian)

    def get_fermionic_hamiltonian(
            self,
            num_electrons: Union[int, Tuple[int, int]],
            num_spatial_orbitals: int
    ) -> Tuple[float, FermionicOperator]:
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

        # calculate active space `ElectronicHamiltonian`
        (core_energy,
         reduced_hamiltonian) = self.get_active_space_hamiltonian(
             num_electrons,
             num_spatial_orbitals
         )

        # Compute the 2nd quantized operators
        second_q_op = reduced_hamiltonian.second_q_op()

        return core_energy, second_q_op

# needed to be able to compute active space without 
# creating a new instance in OrbitalOptimization
def _get_active_space_hamiltonian(
        num_electrons: Union[int, Tuple[int, int]], 
        num_spatial_orbitals: int,
        initial_hamiltonian: ElectronicHamiltonian
    ) -> Tuple[float, ElectronicHamiltonian]:

        # in case of space selection, reduce the space extent of the
        # fermionic Hamiltonian based on the number of active electrons
        # and orbitals
        transformer = ActiveSpaceTransformer(
            num_electrons, num_spatial_orbitals
        )

        transformer.prepare_active_space(
            initial_hamiltonian.num_particles, initial_hamiltonian.num_spatial_orbitals
        )

        # after preparation, transform hamiltonian
        active_space_hamiltonian = transformer.transform_hamiltonian(
            initial_hamiltonian
        )

        # set up core energy after transformation
        nuclear_repulsion_energy = active_space_hamiltonian.constants[
            "nuclear_repulsion_energy"
        ]
        inactive_space_energy = active_space_hamiltonian.constants[
            "ActiveSpaceTransformer"
        ]
        
        core_energy = nuclear_repulsion_energy + inactive_space_energy

        return core_energy, active_space_hamiltonian