"""This module defines the main qc2 data class."""
from typing import Tuple, Union
import os

import h5py
from ase import Atoms
from ase.units import Ha


from .data.qcschema import QCSchema
from .data.qcschema.qc_second_q import QCSecondQ
from .second_q.electronic_hamiltonian import ElectronicHamiltonian
from .second_q.active_space_transformer import ActiveSpaceTransformer
from .second_q.fermionic_operator import FermionicOperator
from .algorithms.base.base_algorithm import BaseAlgorithm
from .ase.qc2_ase_base_class import BaseQc2ASECalculator


class QC2:
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
            :class:`~qc2.algorithm.pennylane.oo_vqe.OO_VQE`.
    """

    def __init__(
        self,
        filename: str = "qchem_data.hdf5",
        molecule: Atoms = Atoms(),
        algorithm: BaseAlgorithm = BaseAlgorithm(),
        *,
        schema: str = "qcschema",
    ):
        """Initializes the ``qc2Data`` class instance.

        Args:
            filename (str): The path to the data file to save/read qchem
                data. Defaults to ``qchem_data.hdf5``
            molecule (Atoms): An optional :class:`ase.atoms.Atoms`
                instance representing the target molecule.
            algorithm (BaseAlgorithm): Algorithm to be run.
                Examples are :class:`~qc2.algorithm.qiskit.vqe.VQE` and
                :class:`~qc2.algorithm.pennylane.oo_vqe.OO_VQE`.
            schema (Optional[str]): An optional attribute defining the format
                in which to save qchem data. Options are ``qcschema`` or
                ``fcidump``. Defaults to ``qcschema``.

        **Example**

        >>> from ase.build import molecule
        >>> from qc2.qc2_driver import QC2 as qc2Data
        >>> from qc2.ase import PySCF
        >>> from qc2.second_q.active_space import ActiveSpace
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
        if (self._schema == "qcschema"
                and file_extension not in [".hdf5", ".h5"]):
            raise ValueError(
                f"{file_extension} is not a valid extension. "
                "For QCSchema format provide a file with "
                "*.hdf5 or *.h5 extensions."
            )

    def run(self) -> None:
        """Runs ASE qchem calculator and saves the data into a formated file.

        Returns:
            None

        **Example**

        >>> from ase.build import molecule
        >>> from qc2.ase import DIRAC
        >>> from qc2.qc2_driver import QC2 as qc2Data
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

    def read_schema(self) -> QCSchema:
        """Reads and stores data in :class:`QCSchema`

        Reads and stores the required data from an HDF5 or FCIDump file as
        either a :class:`QCSchema` or :class:`FCIDump` dataclass instance.

        Returns:
            QCSchema:
                Instance of :class:`QCSchema`

        Notes:
            See qiskit_nature/second_q/formats for more information on the
            supported data formats.

        **Example**

        >>> from ase.build import molecule
        >>> from qc2.ase import DIRAC
        >>> from qc2.qc2_driver import QC2 as qc2Data
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




 