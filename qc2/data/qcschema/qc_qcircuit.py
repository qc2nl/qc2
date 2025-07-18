"""The QCSchema second_q dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, cast

import h5py

from .qc_base import _QCBase


@dataclass
class QCQCircuit(_QCBase):
    """A dataclass to store information about the circuit.

    This is an addition to the original QCSchema
    Matrix quantities are stored as flat, column-major arrays.

    For more information refer to
    [here](https://molssi-qc-schema.readthedocs.io/en/latest/auto_wf.html#wavefunction-schema).
    """

    backend: str
    """Name of the backend., e.g. pennylane, qiskit, ..."""
    hamiltonian_pauli_strings: Sequence[str]
    """Pauli strings, e.g. "IIII"."""
    hamiltonian_coefficients: Sequence[float]
    """Pauli strings coefficients."""
    ansatz_name: str | None = None
    """Name of the ansatz."""
    ansatz_parameters: Sequence[float] | None = None
    """Parameters of the ansatz."""
    ansatz_circuit: Any | None = None
    """Serializd circuit of the ansatz."""