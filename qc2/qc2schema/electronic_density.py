
import numpy as np
from ..algorithms.utils import ActiveSpace
from .qcschema import QCSchema


class ElectronicDensity:

    def __init__(self, schema: QCSchema):
        self.schema = schema 
        self.norb = self.schema.properties.calcinfo_nmo
        self.num_particles = None
        self.num_spatial_orbitals = None
        self.alpha, self.beta =  self._get_density()

    def _get_density(self):

        _alpha = np.diag(np.asarray(
            [1.0] * self.num_alpha + [0.0] * (total_num_spatial_orbitals - self.num_alpha)
        ))
    
        _beta = np.diag(np.asarray(
            [1.0] * self.num_beta + [0.0] * (total_num_spatial_orbitals - num_beta)
        ))