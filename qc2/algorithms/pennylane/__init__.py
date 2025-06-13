"""qc2 algorithms package for pennylane."""
__all__ = []
# handling package imports
try:
    from .vqe.vqe import VQE  # noqa: F401
    from .vqe.oo_vqe import OO_VQE  # noqa: F401
    from .vqe.sa_oo_vqe import SA_OO_VQE  # noqa: F401
    from .qpe.qpe import QPE  # noqa: F401
    from .qpe.iqpe import IQPE  # noqa: F401
    __all__.append(['VQE', 'oo_VQE', 'QPE','IQPE','SA_OO_VQE'])

except ImportError as err:
    raise ImportError(
        "This feature requires PennyLane. "
        "It can be installed with: pip install pennylane."
    ) from err
