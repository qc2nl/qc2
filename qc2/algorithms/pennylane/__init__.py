"""qc2 algorithms package for pennylane."""
__all__ = []
# handling package imports
try:
    from .vqe import VQE  # noqa: F401
    from .oo_vqe import oo_VQE  # noqa: F401
    __all__.append(['VQE', 'oo_VQE'])

except ImportError as err:
    raise ImportError(
        "This feature requires PennyLane. "
        "It can be installed with: pip install pennylane."
    ) from err
