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

"""The basis transformer."""

from __future__ import annotations
from .electronic_integrals import ElectronicIntegrals, TensorDict
from .electronic_hamiltonian import ElectronicHamiltonian

class BasisTransformer():

    def __init__(self,
                 coefficients: ElectronicIntegrals):
        
        self.coefficients = coefficients

    def transform_electronic_integrals(self, integrals: ElectronicIntegrals) -> ElectronicIntegrals:
        """Transforms an :class:`qiskit_nature.second_q.operators.ElectronicIntegrals` instance.

        Args:
            integrals: the ``ElectronicIntegrals`` to transform.

        Raises:
            QiskitNatureError: when using this method on a ``BasisTransformer`` that does not store
                its :attr:`coefficients` as ``ElectronicIntegrals``, too.

        Returns:
            The transformed ``ElectronicIntegrals``.
        """
        if not isinstance(self.coefficients, ElectronicIntegrals):
            raise TypeError(
                "You cannot transform ElectronicIntegrals with a BasisTransformer containing "
                f"coefficients of type, {type(self.coefficients)}, rather than ElectronicIntegrals."
            )

        prsq = "prsq"
        iklj = "iklj"

        two_body_aa = integrals.alpha.get("++--", None)
        if two_body_aa is not None:
            prsq = "pqrs"
            iklj = "ijkl"

        einsum_map = {
            "jk,ji,kl->il": ("+-",) * 4,
            f"{prsq},pi,qj,rk,sl->{iklj}": ("++--", *("+-",) * 4, "++--"),
        }

        transformed_integrals = ElectronicIntegrals.einsum(
            einsum_map, integrals, *(self.coefficients,) * 4
        )

        if not self.coefficients.beta.is_empty() and transformed_integrals.beta_alpha.is_empty():
            transformed_integrals.beta_alpha = TensorDict.einsum(
                {f"{prsq},pi,qj,rk,sl->{iklj}": ("++--", *("+-",) * 4, "++--")},
                integrals.alpha if integrals.beta_alpha.is_empty() else integrals.beta_alpha,
                *(self.coefficients.beta,) * 2,
                *(self.coefficients.alpha,) * 2,
            )

        return transformed_integrals
    
    def transform_hamiltonian(self, hamiltonian: ElectronicHamiltonian) -> ElectronicHamiltonian:
        if isinstance(hamiltonian, ElectronicHamiltonian):
            integrals = hamiltonian.electronic_integrals
            hamiltonian.electronic_integrals = self.transform_electronic_integrals(integrals)
            return hamiltonian
        else:
            raise NotImplementedError(
                f"The hamiltonian of type, {type(hamiltonian)}, is not supported by this "
                "transformer."
            )