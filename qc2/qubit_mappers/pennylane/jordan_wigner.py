from pennylane.fermi import jordan_wigner
from .base_mapper import BaseMapper


class JordanWigner(BaseMapper):
    mapper = jordan_wigner