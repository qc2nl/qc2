# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
from __future__ import annotations

import numpy as np
from typing import Dict, Callable
from ..algorithms.utils import ActiveSpace
from .qcschema import QCSchema
from .electronic_integrals import ElectronicIntegrals, TensorDict
# from .polynomial_tensor import PolynomialTensor
# from .fermionic_op import FermionicOp

def _reshape_2( arr, size):
        return np.asarray(arr).reshape((size, size))

def _reshape_4(arr, size):
    return np.asarray(arr).reshape((size,) * 4)

class ElectronicHamiltonian:

    def __init__(self, schema: QCSchema, tol=1E-6):
        self.schema = schema 
        self.tol = tol
        self.nuclear_repulsion_energy = self.schema.properties.nuclear_repulsion_energy
        self.norb = self.schema.properties.calcinfo_nmo
        self.num_particles = None
        self.num_spatial_orbitals = None

        self.electronic_integrals = self.get_electronic_integrals()

        self.get_second_q_coeffs()

        self.get_second_q_ops()

    def get_electronic_integrals(self):
        # see qcshema_translator.get_mo_hamiltonian_direct

        alpha = TensorDict({'+-': None, '++--': None})
        beta = TensorDict({'+-': None, '++--': None})
        beta_alpha = TensorDict({'++--': None})


        alpha['+-'] = _reshape_2(self.schema.wavefunction.scf_fock_mo_a, self.norb)
        alpha['++--'] = _reshape_4(self.schema.wavefunction.scf_eri_mo_aa, self.norb)
        
        if self.schema.wavefunction.scf_fock_mo_b is not None:
            beta['+-'] = _reshape_2(self.schema.wavefunction.scf_fock_mo_b, self.norb)

        if self.schema.wavefunction.scf_eri_mo_bb is not None:
            beta['++--'] = _reshape_4(self.schema.wavefunction.scf_eri_mo_bb, self.norb)

        if self.schema.wavefunction.scf_eri_mo_ba is not None:
            beta_alpha['++--'] = _reshape_4(self.schema.wavefunction.scf_eri_mo_ba, self.norb)

        if self.schema.wavefunction.scf_eri_mo_ab is not None and beta_alpha['++--'] is None:
            beta_alpha['++--'] = np.transpose(self._reshape_4(self.schema.wavefunction.scf_eri_mo_ab))

        return ElectronicIntegrals(alpha=alpha, beta=beta, beta_alpha=beta_alpha)


    def get_second_q_coeffs(self) -> Dict:

        # see ElectronicIntegrals.second_q_coeff()
        self.second_q_coeffs = {'+-': None, '++--': None}
        
        # one body coefficients
        kron_one_body = np.zeros((2, 2))
        kron_one_body[(0, 0)] = 1
        self.second_q_coeffs['+-'] = np.kron( kron_one_body, self.electronic_integrals.alpha['+-'] ) 

        kron_one_body[(0, 0)] = 0
        kron_one_body[(1, 1)] = 1
        self.second_q_coeffs['+-'] += np.kron( kron_one_body, self.electronic_integrals.beta['+-'] ) 
        
        # two body coefficients pure spin
        kron_two_body = np.zeros((2, 2, 2, 2))
        kron_two_body[(0, 0, 0, 0)] = 0.5
        self.second_q_coeffs['++--'] = np.kron( kron_two_body, self.electronic_integrals.alpha['++--'] )

        kron_two_body[(0, 0, 0, 0)] = 0.0
        kron_two_body[(1, 1, 1, 1)] = 0.5
        self.second_q_coeffs['++--'] += np.kron( kron_two_body, self.electronic_integrals.beta['++--'] )

        # two body coefficients mixed spin
        kron_two_body[(1, 1, 1, 1)] = 0.0
        kron_two_body[(1, 1, 0, 0)] = 0.5
        self.second_q_coeffs['++--'] += np.kron( kron_two_body, self.electronic_integrals.beta_alpha['++--'] )

        kron_two_body[(1, 1, 0, 0)] = 0.0
        kron_two_body[(0, 0, 1, 1)] = 0.5
        self.second_q_coeffs['++--'] += np.kron( kron_two_body, self.electronic_integrals.beta_alpha['++--'].T )
                                                

    
    def get_second_q_ops(self):

        # see FermionicOp.from_polynomial_tensor()
        self.second_q_ops = {}
        norb = self.second_q_coeffs['+-'].shape[0]

        for i in range(norb):
            for j in range(norb):
                if np.abs(self.second_q_coeffs['+-'][i, j]) >= self.tol: 
                    self.second_q_ops[("+_{} -_{}".format(i, j))] = self.second_q_coeffs['+-'][i, j]

        for i in range(norb):
            for j in range(norb):
                for k in range(norb):
                    for l in range(norb):
                        if np.abs(self.second_q_coeffs['++--'][i, j, k, l]) >= self.tol: 
                            self.second_q_ops[("+_{} +_{} -_{} -_{}".format(i, k, l, j))] = self.second_q_coeffs['++--'][i, j, k, l]


    def coulomb(self, density: ElectronicIntegrals) -> ElectronicIntegrals:
        r"""Computes the Coulomb term for the given reduced density matrix.

        .. math::
            J_{qr} = \sum g_{pqrs} D_{ps}

        Args:
            density: the reduced density matrix.

        Returns:
            The Coulomb operator coefficients.

        Raises:
            NotImplementedError: when encountering :class:`.SymmetricTwoBodyIntegrals` inside of
                :attr:`.ElectronicEnergy.electronic_integrals`.
        """
        # two_body_aa = self.electronic_integrals.alpha.get("++--", None)
        # einsum = f"{''.join(two_body_aa._reverse_label_template('pqrs'))},ps->qr"

        einsum = "psqr,ps->qr"
        coulomb = ElectronicIntegrals.einsum(
            {einsum: ("++--", "+-", "+-")}, self.electronic_integrals, density
        )
        
        if self.electronic_integrals.beta_alpha.is_empty() and density.beta.is_empty():
            coulomb *= 2.0  # type: ignore

        else:
            if self.electronic_integrals.beta_alpha.is_empty(): 
                beta_alpha = self.electronic_integrals.two_body.alpha
            else:
                beta_alpha = self.electronic_integrals.beta_alpha

            coulomb.alpha += TensorDict.einsum({einsum: ("++--", "+-", "+-")}, beta_alpha, density.beta)

            einsum = einsum[2:4] + einsum[:2] + einsum[4:]
            coulomb.beta += TensorDict.einsum({einsum: ("++--", "+-", "+-")}, beta_alpha, density.alpha)

        return coulomb

    def exchange(self, density: ElectronicIntegrals) -> ElectronicIntegrals:
        r"""Computes the Exchange term for the given reduced density matrix.

        .. math::
            K_{pr} = \sum g_{pqrs} D_{qs}

        Args:
            density: the reduced density matrix.

        Returns:
            The Exchange operator coefficients.

        Raises:
            NotImplementedError: when encountering :class:`.SymmetricTwoBodyIntegrals` inside of
                :attr:`.ElectronicEnergy.electronic_integrals`.
        """

        exchange = ElectronicIntegrals.einsum(
            {"psqr,qs->pr": ("++--", "+-", "+-")}, self.electronic_integrals, density
        )
        return exchange

    def fock(self, density):
        r"""Computes the Fock operator for the given reduced density matrix.

        .. math::
            F_{pq} = h_{pq} + J_{pq} - K_{pq}

        where :math:`J` and :math:`K` are the :meth:`coulomb` and :meth:`exchange` terms,
        respectively.

        Args:
            density: the reduced density matrix.

        Returns:
            The Fock operator coefficients.
        """
        return self.electronic_integrals.one_body + self.coulomb(density) - self.exchange(density)
    
    