"""Module defining the QPE algorithm for qiskit"""
from qiskit_algorithms import IterativePhaseEstimation
from .pebase import PEBase

class IQPE(PEBase):
    def __init__(self, 
                 qc2data=None, 
                 num_iterations=None,
                 active_space=None, 
                 mapper=None, 
                 sampler=None, 
                 reference_state=None,  
                 verbose=0):
        super().__init__(qc2data, active_space, mapper, sampler, reference_state, verbose)
        self.num_iterations = num_iterations
        self.solver = IterativePhaseEstimation(self.num_iterations, self.sampler)
