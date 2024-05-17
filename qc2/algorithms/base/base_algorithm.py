"""Module defining base class for all algorithms."""
from abc import ABC


class BaseAlgorithm(ABC):
    """Base class for all qc2 algos."""

    def set_qc2data(self, qc2data):
        """set the data"""
        self.qc2data = qc2data

    def run(self, *args, **kwargs):
        """run it"""
        raise NotImplementedError("BaseAlgorithm doens't have a .run() implemented ")

