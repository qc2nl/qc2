import netket as nk
import jax.numpy as jnp
import GPSKet.models as qGPS

from GPSKet.hilbert.discrete_fermion import FermionicDiscreteHilbert
from GPSKet.sampler.fermionic_hopping import MetropolisHopping, MetropolisFastHopping
from GPSKet.operator.hamiltonian.ab_initio import (
    AbInitioHamiltonian,
    AbInitioHamiltonianOnTheFly,
)
from GPSKet.models import qGPS
from ..base.base_algorithm import BaseAlgorithm

"""
This is the GPSKet/NetKet bit of the calculation.
The key elements different from standard netket calculations are:
1.) Setting up a fermionic discrete Hilbert space:
    In this Hilbert space configurations are represented as a list of L 8-bit integers
    of which only two bits are used. The first (least significant) bit of each int
    encodes whether the alpha/spin-up channel is occupied at the particular site and the
    second bit encodes whether the beta/spin-down channel is occupied. We might change
    this representation at one point but currently this is the best trade-off between
    memory efficiency and convenience.
2.) Setting up the ab-initio Hamiltonian:
    This requires the defined Hilbert space as well as the one and two electron integrals
    as generated above. For a detailed description of the Hamiltonian definition see
    [Neuscamman (2013), https://doi.org/10.1063/1.4829835].
3.) A sampler to generate configurations:
    Non-autoregressive ansatzes require custom transition rules for the Metropolis-Hastings
    algorithm to generate proposals. Currently only a hopping transition is implemented
    for which a randomly selected electron hops from one site to another (thus always
    conserving the total magnetization and electron number of the initial config).
    Two different versions of this hopping sampler are currently available,
    the MetropolisHopping class which follows the default NetKet design, as well as the
    MetropolisFastHopping class which includes the fast update mechanism which can be
    used with the qGPS ansatz.
    More samplers should probably be implemented in the future.
    For the autoregressive ansatz, the direct sampler can be used for the
    Fermionic systems but needs to be amended to take electron number and magnetization
    conservation into account (if this is wanted).
    TODO: this needs to be implemented and checked.
"""

class VMC(BaseAlgorithm):
    def __init__(
            self,
            qc2data=None,
            model=None,
            sampler=None,
            active_space=None):

        self.qc2data = qc2data
        self.active_space = active_space
        super().__init__()


    def run(self):
        data = self.qc2data.read_schema()

        norb = data.properties.calcinfo_nmo
        nalpha = data.properties.calcinfo_nalpha
        nbeta = data.properties.calcinfo_nbeta
        nuc_en = data.properties.nuclear_repulsion_energy

        h1 = data.wavefunction.scf_fock_mo_a
        h2 = data.wavefunction.scf_eri_mo_aa

        # Set up Hilbert space
        hi = FermionicDiscreteHilbert(norb, n_elec=(nalpha // 2, nbeta // 2))

        # Set up ab-initio Hamiltonian
        ha = AbInitioHamiltonianOnTheFly(hi, h1, h2)


        # If we want, we can compare the exact energies given by the PySCF and the NetKet solver
        # e_mo_nk = nk.exact.lanczos_ed(ha)[0]
        # assert(np.allclose(e_mo_nk, energy_mo))


        # Use Metropolis-Hastings sampler with hopping rule (including fast updates for qGPS)
        sa = MetropolisFastHopping(hi, n_chains_per_rank=1)

        # Define the model and the variational state
        model = qGPS(hi, 10, dtype=jnp.complex128)
        vs = nk.vqs.MCState(sa, model, n_samples=1000)


        # Optimizer
        op = nk.optimizer.Sgd(learning_rate=0.02)
        qgt = nk.optimizer.qgt.QGTJacobianDense(holomorphic=True)
        sr = nk.optimizer.SR(qgt=qgt)

        # Variational Monte Carlo driver
        gs = nk.VMC(ha, op, variational_state=vs, preconditioner=sr)

        # Run optimization
        for it in gs.iter(1000, 1):
            en = gs.energy.mean + nuc_en
            print(
                "Iteration: {}, Energy: {}".format(
                    it, en
                ),
                flush=True,
            )