"""Module containing active space helper functions."""
from dataclasses import dataclass
from typing import Tuple, Union
import numpy as np


def get_active_space_idx(
    nao: int,
    nelectron: Union[int, Tuple[int, int]],
    n_active_orbitals: int,
    n_active_electrons: Union[int, Tuple[int, int]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates indices for occ, active, and virt orbitals in an active space.

    Args:
        nao (int): Total number of atomic orbitals.
        nelectron (Union[int, Tuple[int, int]]): A list representing the
            number of electrons in each atom.
        n_active_orbitals (int): Number of orbitals in the active space.
        n_active_electrons (Union[int, Tuple[int, int]]): A list representing
            the number of active electrons in each atom.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            Three numpy arrays representing the indices of occupied, active, 
            and virtual orbitals, respectively.

    Raises:
        ValueError: If the number of core electrons is odd.
    """
    # Set active space parameters
    nelecore = sum(nelectron) - sum(n_active_electrons)
    if nelecore % 2 == 1:
        raise ValueError('odd number of core electrons')

    occ_idx = np.arange(nelecore // 2)
    act_idx = (occ_idx[-1] + 1 + np.arange(n_active_orbitals)
               if len(occ_idx) > 0
               else np.arange(n_active_orbitals))
    virt_idx = np.arange(act_idx[-1]+1, nao)

    return occ_idx, act_idx, virt_idx


@dataclass
class ActiveSpace:
    """A data class representing the active space."""
    num_active_electrons: tuple[int, int]
    num_active_spatial_orbitals: int
