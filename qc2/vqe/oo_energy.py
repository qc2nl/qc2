""""Module docstring."""

import numpy as np
from qiskit_nature.second_q.formats.qcschema import QCSchema


def _reshape_2(arr, dim, dim_2=None):
    """Docstring."""
    return np.asarray(arr).reshape((dim, dim_2 if dim_2 is not None else dim))


def _reshape_4(arr, dim):
    """Docstring."""
    return np.asarray(arr).reshape((dim,) * 4)


def _get_int1e_mo(qcs_dataclass: QCSchema) -> tuple[np.ndarray, np.ndarray]:
    """Docstring."""
    norb = qcs_dataclass.properties.calcinfo_nmo
    hij = _reshape_2(qcs_dataclass.wavefunction.scf_fock_mo_a, norb)
    hij_b = None

    if qcs_dataclass.wavefunction.scf_fock_mo_b is not None:
        hij_b = _reshape_2(qcs_dataclass.wavefunction.scf_fock_mo_b, norb)

    return hij, hij_b


def _get_int2e_mo(qcs_dataclass: QCSchema) -> tuple[np.ndarray, np.ndarray,
                                               np.ndarray, np.ndarray]:
    """Docstring."""
    norb = qcs_dataclass.properties.calcinfo_nmo
    hijkl = _reshape_4(qcs_dataclass.wavefunction.scf_eri_mo_aa, norb)
    hijkl_bb = None
    hijkl_ba = None

    if qcs_dataclass.wavefunction.scf_eri_mo_bb is not None:
        hijkl_bb = _reshape_4(qcs_dataclass.wavefunction.scf_eri_mo_bb, norb)
    if qcs_dataclass.wavefunction.scf_eri_mo_ba is not None:
        hijkl_ba = _reshape_4(qcs_dataclass.wavefunction.scf_eri_mo_ba, norb)
    #if qcs_dataclass.wavefunction.scf_eri_mo_ab is not None and hijkl_ba is None:
    #    hijkl_ba = np.transpose(_reshape_4(qcs_dataclass.wavefunction.scf_eri_mo_ab, norb))

    return hijkl, hijkl_bb, hijkl_ba


def non_redundant_indices(occ_idx, act_idx, virt_idx, freeze_active):
    """Calculate non-redundant indices for indexing kappa vectors
    for a given active space"""
    no, na, nv = len(occ_idx), len(act_idx), len(virt_idx)
    nao = no + na + nv
    rotation_sizes = [no * na, na * nv, no * nv]
    if not freeze_active:
        rotation_sizes.append(na * (na - 1) // 2)
    n_kappa = sum(rotation_sizes)
    params_idx = np.array([], dtype=int)
    num = 0
    for l_idx, r_idx in zip(*np.tril_indices(nao, -1)):
        if not (
            ((l_idx in act_idx and r_idx in act_idx) and freeze_active)
            or (l_idx in occ_idx and r_idx in occ_idx)
            or (l_idx in virt_idx and r_idx in virt_idx)
        ):  # or (
            # l_idx in virt_idx and r_idx in occ_idx)):
            params_idx = np.append(params_idx, [num])
        num += 1
    assert n_kappa == len(params_idx)
    return params_idx


class oo_energy:
    """Docstring."""

    def __init__(self, qcshema: QCSchema,
                 n_active_orbitals, n_active_electrons):
        """Module Docstrings."""

        # qcschema dataclass object
        self.qcshema = qcshema

        # get molecular integrals
        self.int1e_mo_a = _get_int1e_mo(self.qcshema)[0]
        self.int1e_mo_b = _get_int1e_mo(self.qcshema)[1]
        self.int2e_mo_aa = _get_int2e_mo(self.qcshema)[0]
        self.int2e_mo_bb = _get_int2e_mo(self.qcshema)[1]
        self.int2e_mo_ba = _get_int2e_mo(self.qcshema)[2]

        # Set active space parameters
        self.n_active_orbitals = n_active_orbitals
        self.n_active_electrons = n_active_electrons


if __name__ == "__main__":
    import h5py

    # open the HDF5 file
    with h5py.File('test.h5', 'r') as file:
        qcschema = QCSchema._from_hdf5_group(file)

    n_elec = (1, 1)  # => (n_alpha, n_beta)
    n_orb = 2
    oo = oo_energy(qcschema, n_elec, n_orb)

    print(oo.int2e_mo_aa, oo.int2e_mo_aa.shape)