"""The QCSchema second_q dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, cast

import h5py

from .qc_base import _QCBase
from .qc_basis_set import QCBasisSet


@dataclass
class QCSecondQ(_QCBase):
    """A dataclass to store any computed second quantization properties.

    This is an addition to the original QCSchema
    Matrix quantities are stored as flat, column-major arrays.

    For more information refer to
    [here](https://molssi-qc-schema.readthedocs.io/en/latest/auto_wf.html#wavefunction-schema).
    """


    active_electrons: Sequence[int]
    """Number of active electrons, e.g. (2,2)."""
    active_spatial_orbitals: int
    """Number of active spatial orbitals."""
    creation_anhiliton_operators: Sequence[str]
    """Creation and annihilation operators, e.g. +_2 -_3."""
    creation_anhiliton_coefficients: Sequence[float]
    """Creation and annihilation operators."""
    