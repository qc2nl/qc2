from qiskit.circuit.library.blueprintcircuit import BlueprintCircuit
from qiskit.exceptions import QiskitError
from qiskit.circuit import QuantumCircuit, Parameter, QuantumRegister, QuantumCircuit
from qiskit.circuit.library.blueprintcircuit import BlueprintCircuit
from qc2.algorithms.utils.active_space import ActiveSpace

class StateResolution(BlueprintCircuit):
    """State Resolution initial state for SA OO VQE."""

    def __init__(
            self,
            active_space: ActiveSpace,
            name: str | None = "StateResolution",
    ):
        """
        Args:
            active_space (ActiveSpace): Description of the active space.
            name (str): The name of the circuit.
        """
        super().__init__(name=name)
        self.active_space = active_space
        self._num_qubits = 2 * self.active_space.num_active_spatial_orbitals
        if self.num_qubits == 0:    
            self.qregs = []
        else:
            self.qregs = [QuantumRegister(self.num_qubits, name="q")]

    @property
    def num_qubits(self) -> int:
        """The number of qubits."""
        return self._num_qubits

    @num_qubits.setter
    def num_qubits(self, n: int) -> None:
        """Sets the number of qubits."""
        self._num_qubits = n

    def _check_configuration(self, raise_on_failure: bool = True) -> bool:

        """Check if the configuration of the NLocal class is valid.

        Args:
            raise_on_failure: Whether to raise on failure.

        Returns:
            True, if the configuration is valid and the circuit can be constructed. Otherwise
            an ValueError is raised.

        Raises:
            ValueError: If the numbr fo qubit is not set.
            ValueError: If the number of spatial orbitals is lower than the number of particles
        """
        valid = True
        if self.num_qubits is None:
            valid = False
            if raise_on_failure:
                raise ValueError("No number of qubits specified.")
        return valid
    
    def _build(self):
        if self._is_built:
            return
        super()._build()
        
        qc = state_resolution_initializer(self.active_space)

        # append to self
        try:
            block = qc.to_gate()
        except QiskitError:
            block = qc.to_instruction()
        self.append(block, range(self.num_qubits), copy=False)  

def state_resolution_initializer(
    active_space,
) -> QuantumCircuit:
    """
    Set up the default reference state circuit based on Hartree Fock and singlet excitation.

    Parameters:
    active_space (ActiveSpace): Description of the active space.

    Returns:
        QuantumCircuit: The reference state circuit.
    """
    norb = active_space.num_active_spatial_orbitals
    nalpha, nbeta = active_space.num_active_electrons

    phi = Parameter('phi')

    idx_alpha_homo = nalpha - 1
    idx_beta_homo  = norb + nbeta - 1
    idx_alpha_lumo = nalpha
    idx_beta_lumo  = norb + nbeta

    qc = QuantumCircuit(2*norb)

    # create the fixed electrons 
    for i in range(idx_alpha_homo):
        qc.x(i)    

    for i in range(norb, idx_beta_homo):
        qc.x(i)    

    # rotation
    qc.ry(2*phi, idx_alpha_homo)
    qc.x(idx_beta_homo)
    qc.ch(idx_alpha_homo, idx_beta_lumo)

    # cnots
    qc.cx(idx_beta_lumo, idx_beta_homo)
    qc.cx(idx_beta_lumo, idx_alpha_homo)
    qc.cx(idx_alpha_homo, idx_alpha_lumo)

    # last steps
    qc.x(idx_alpha_homo)
    qc.z(idx_beta_lumo)

    return qc