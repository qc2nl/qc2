
import numpy as np
from .qcschema import QCSchema


class ElectronicHamiltonian:

    def __init__(self, schema: QCSchema, tol=1E-6):
        self.schema = schema 
        self.tol = tol
        self.nuclear_repulsion_energy = self.schema.properties.nuclear_repulsion_energy
        self.norb = self.schema.properties.calcinfo_nmo
        self.num_particles = None
        self.num_spatial_orbitals = None

        self.get_mo_hamiltonian()

        self.get_second_q_coeffs()

        self.get_second_q_ops()
   
    def get_mo_hamiltonian(self):
        # see qcshema_translator.get_mo_hamiltonian_direct

        self.alpha = {'+-': None, '++--': None}
        self.beta = {'+-': None, '++--': None}
        self.beta_alpha = {'++--': None}


        self.alpha['+-'] = self._reshape_2(self.schema.wavefunction.scf_fock_mo_a)
        self.alpha['++--'] = self._reshape_4(self.schema.wavefunction.scf_eri_mo_aa)
        
        if self.schema.wavefunction.scf_fock_mo_b is not None:
            self.beta['+-'] = self._reshape_2(self.schema.wavefunction.scf_fock_mo_b)

        if self.schema.wavefunction.scf_eri_mo_bb is not None:
            self.beta['++--'] = self._reshape_4(self.schema.wavefunction.scf_eri_mo_bb)

        if self.schema.wavefunction.scf_eri_mo_ba is not None:
            self.beta_alpha['++--'] = self._reshape_4(self.schema.wavefunction.scf_eri_mo_ba)

        if self.schema.wavefunction.scf_eri_mo_ab is not None and self.beta_alpha['++--'] is None:
            self.beta_alpha['++--'] = np.transpose(self._reshape_4(self.schema.wavefunction.scf_eri_mo_ab))

    def _reshape_2(self, arr):
        return np.asarray(arr).reshape((self.norb, self.norb))

    def _reshape_4(self, arr):
        return np.asarray(arr).reshape((self.norb,) * 4)
    
    def get_second_q_coeffs(self):
        # see ElectronicIntegrals.second_q_coeff()
        self.second_q_coeffs = {'+-': None, '++--': None}
        kron_one_body = np.zeros((2, 2))
        kron_two_body = np.zeros((2, 2, 2, 2))

        kron_one_body[(0, 0)] = 1
        kron_one_body[(1, 1)] = 1
        self.second_q_coeffs['+-'] = np.kron( kron_one_body, self.alpha['+-'] ) 
        
        kron_two_body[(0, 0, 0, 0)] = 0.5
        kron_two_body[(1, 1, 1, 1)] = 0.5

        
        self.second_q_coeffs['++--'] = np.kron( kron_two_body, self.alpha['++--'] )
        # missing terms

    
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
                            self.second_q_ops[("+_{} +_{} -_{} -_{}".format(i, j, k, l))] = self.second_q_coeffs['++--'][i, j, k, l]
