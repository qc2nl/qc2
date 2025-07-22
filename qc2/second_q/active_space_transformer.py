# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from typing import  cast
import numpy as np
from copy import deepcopy
from .electronic_integrals import ElectronicIntegrals, TensorDict
from .electronic_hamiltonian import ElectronicHamiltonian


class ActiveSpaceTransformer():

    def __init__(
        self,
        num_electrons: int | tuple[int, int],
        num_spatial_orbitals: int,
        active_orbitals: list[int] | tuple[list[int], list[int]] | None = None,
    ):
        
        self._num_electrons = num_electrons
        self._num_spatial_orbitals = num_spatial_orbitals
        self._active_orbitals = active_orbitals

        self._active_alpha_indices: list[int] = None
        self._active_beta_indices: list[int] = None

        # NOTE: the following attribute is exposed as a read-write property to the user
        # The reason we are not making it a public attribute is to avoid the FreezeCoreTransformer
        # also having to expose it publicly.
        self._active_density: ElectronicIntegrals = None
        self._density_total: ElectronicIntegrals = None

        self.reference_inactive_fock: ElectronicIntegrals | None = None
        self.reference_inactive_energy: float | None = None

    @property
    def active_basis(self):
        """Returns the ``BasisTransformer`` mapping from the total to the active space."""
        return self._active_basis

    @property
    def active_density(self) -> ElectronicIntegrals | None:
        """Returns the active electronic density."""
        return self._active_density

    @active_density.setter
    def active_density(self, active_density: ElectronicIntegrals | None) -> None:
        """Sets the active electronic density."""
        self._active_density = active_density

    def _determine_active_space(
        self, total_num_electrons: int, total_num_spatial_orbitals: int
    ) -> tuple[list[int], list[int]]:
        """Determines the active and inactive orbital indices.

        Args:
            total_num_electrons: the total number of electrons in the system represented by the
                hamiltonian which is to be transformed. If this is a tuple of integers, it encodes
                the number of alpha- and beta-spin electrons separately. Otherwise the integer value
                is assumed to indicate the sum of these two numbers.
            total_num_spatial_orbitals: the total number of spatial orbitals in the system
                represented by the hamiltonian which is to be transformed.

        Returns:
            The pair of active alpha- and beta-spin orbital indices.
        """
        if self._active_orbitals is not None:
            if isinstance(self._active_orbitals, tuple):
                return self._active_orbitals
            return (self._active_orbitals, self._active_orbitals)

        if isinstance(self._num_electrons, tuple):
            num_alpha, num_beta = self._num_electrons
        elif isinstance(self._num_electrons, (int, np.integer)):
            num_alpha = num_beta = self._num_electrons // 2

        # compute number of inactive electrons
        nelec_inactive = total_num_electrons - num_alpha - num_beta

        norbs_inactive = nelec_inactive // 2
        active_orbs_idxs = list(range(norbs_inactive, norbs_inactive + self._num_spatial_orbitals))
        return (active_orbs_idxs, active_orbs_idxs)

    def prepare_active_space(
        self,
        total_num_electrons: int | tuple[int, int],
        total_num_spatial_orbitals: int,
        *,
        occupation_alpha: list[float] | np.ndarray | None = None,
        occupation_beta: list[float] | np.ndarray | None = None,
    ) -> None:
        """Prepares the active space.

        This method must be called manually when using this transformer on a hamiltonian outside of
        a problem instance. In all other cases, the information required here is extracted from the
        problem automatically.

        Args:
            total_num_electrons: the total number of electrons in the system represented by the
                hamiltonian which is to be transformed. If this is a tuple of integers, it encodes
                the number of alpha- and beta-spin electrons separately. Otherwise the integer value
                is assumed to indicate the sum of these two numbers.
            total_num_spatial_orbitals: the total number of spatial orbitals in the system
                represented by the hamiltonian which is to be transformed.
            occupation_alpha: the occupation of the alpha-spin orbitals. If omitted, this
                information is inferred from the required arguments.
            occupation_beta: the occupation of the beta-spin orbitals. If omitted, this
                information is inferred from the required arguments.

        Raises:
            QiskitNatureError: if any of the requirements for a valid active space configuration
                (documented in the class docstring) are not met.
        """
        if isinstance(total_num_electrons, tuple):
            num_alpha, num_beta = total_num_electrons
            sum_electrons = num_alpha + num_beta
        else:
            num_beta = total_num_electrons // 2
            num_alpha = total_num_electrons - num_beta
            sum_electrons = total_num_electrons

        if occupation_alpha is None:
            occupation_alpha = np.asarray(
                [1.0] * num_alpha + [0.0] * (total_num_spatial_orbitals - num_alpha)
            )

        if occupation_beta is None:
            occupation_beta = np.asarray(
                [1.0] * num_beta + [0.0] * (total_num_spatial_orbitals - num_beta)
            )

        self._active_alpha_indices, self._active_beta_indices = self._determine_active_space(
            sum_electrons, total_num_spatial_orbitals
        )

        self._density_total = ElectronicIntegrals(
            alpha = TensorDict({'+-' : np.diag(occupation_alpha)}), 
            beta = TensorDict({'+-' : np.diag(occupation_beta)})
        )


        num_active_alpha = self._num_electrons[0]
        num_frozen_alpha = num_alpha - num_active_alpha

        num_active_beta = self._num_electrons[1]
        num_frozen_beta = num_beta - num_active_beta
        occupation_active_alpha = [0] * num_frozen_alpha + [1] * num_active_alpha + [0] * (total_num_spatial_orbitals - num_alpha)
        occupation_active_beta = [0] * num_frozen_beta + [1] * num_active_beta + [0] * (total_num_spatial_orbitals - num_beta)
       
        self._active_density = ElectronicIntegrals(
            alpha = TensorDict({'+-' : np.diag(occupation_active_alpha)}), 
            beta = TensorDict({'+-' : np.diag(occupation_active_beta)})
        )

        # initialize size-reducing basis transformation
        alpha = TensorDict()
        coeff_alpha = np.zeros((total_num_spatial_orbitals, self._num_spatial_orbitals))
        coeff_alpha[self._active_alpha_indices, range(self._num_spatial_orbitals)] = 1.0
        alpha["+-"] = coeff_alpha

        beta = TensorDict()
        coeff_beta = np.zeros((total_num_spatial_orbitals, self._num_spatial_orbitals))
        coeff_beta[self._active_beta_indices, range(self._num_spatial_orbitals)] = 1.0
        beta["+-"] = coeff_beta

        self.transform_coefficents = ElectronicIntegrals(alpha, beta)
        

    def transform_hamiltonian(self, hamiltonian: ElectronicHamiltonian) -> ElectronicHamiltonian:
       
        reference_inactive_fock = hamiltonian.fock(self._density_total)

        active_fock_operator = (
            hamiltonian.fock(self._active_density) - hamiltonian.electronic_integrals.one_body
        )

        inactive_fock_operator = reference_inactive_fock - active_fock_operator

        reference_inactive_energy = cast(
            ElectronicIntegrals,
            ElectronicIntegrals.einsum(
                {"ij,ji": ("+-", "+-", "")},
                reference_inactive_fock + hamiltonian.electronic_integrals.one_body,
                self._density_total,
            ) * 0.5,
        )

        print('reference_inactive_energy: ', reference_inactive_energy.alpha.get("", 0.0))
        print('reference_inactive_energy: ', reference_inactive_energy.beta.get("", 0.0))
        print('reference_inactive_energy: ', reference_inactive_energy.beta_alpha.get("", 0.0))
        print('')

        reference_inactive_energy = (
            reference_inactive_energy.alpha.get("", 0.0)
            + reference_inactive_energy.beta.get("", 0.0)
            + reference_inactive_energy.beta_alpha.get("", 0.0)
        )
        

        e_inactive = cast(
            ElectronicIntegrals,
            ElectronicIntegrals.einsum(
                {"ij,ji": ("+-", "+-", "")}, reference_inactive_fock, self._active_density
            ) * (-1.0),
        )
        e_inactive += cast(
            ElectronicIntegrals,
            ElectronicIntegrals.einsum(
                {"ij,ji": ("+-", "+-", "")}, active_fock_operator, self._active_density
            ) * 0.5,
        )

        print('e_inactive_sum: ', e_inactive.alpha.get("", 0.0))
        print('e_inactive_sum: ', e_inactive.beta.get("", 0.0))
        print('e_inactive_sum: ', e_inactive.beta_alpha.get("", 0.0))
        print('')

        e_inactive_sum = (
            reference_inactive_energy
            + e_inactive.alpha.get("", 0.0)
            + e_inactive.beta.get("", 0.0)
            + e_inactive.beta_alpha.get("", 0.0)
        )

        new_hamil = ElectronicHamiltonian(
            electronic_integrals=self.transform_electronic_integrals(
                inactive_fock_operator + hamiltonian.electronic_integrals.two_body
            ),
            constants=deepcopy(hamiltonian.constants)
        )

        new_hamil.constants[self.__class__.__name__] = e_inactive_sum

        return new_hamil
    
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
        if not isinstance(self.transform_coefficents, ElectronicIntegrals):
            raise TypeError(
                "You cannot transform ElectronicIntegrals with  "
                f"coefficients of type, {type(self.transform_coefficents)}, rather than ElectronicIntegrals."
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
            einsum_map, integrals, *(self.transform_coefficents,) * 4
        )

        if not self.transform_coefficents.beta.is_empty() and transformed_integrals.beta_alpha.is_empty():
            transformed_integrals.beta_alpha = TensorDict.einsum(
                {f"{prsq},pi,qj,rk,sl->{iklj}": ("++--", *("+-",) * 4, "++--")},
                integrals.alpha if integrals.beta_alpha.is_empty() else integrals.beta_alpha,
                *(self.transform_coefficents.beta,) * 2,
                *(self.transform_coefficents.alpha,) * 2,
            )

        return transformed_integrals