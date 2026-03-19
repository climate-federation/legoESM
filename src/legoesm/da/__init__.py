"""Data assimilation module for legoESM.

This module will contain ensemble-based and variational data assimilation
methods for parameter estimation and uncertainty quantification.

Current Status: Under development — stubs provided for interface stability.
"""

import logging
import warnings

__all__: list[str] = []

logger = logging.getLogger(__name__)

_DA_NOT_IMPLEMENTED_MSG = (
    "Data assimilation module is under development. "
    "'{name}' is not yet implemented."
)


def __getattr__(name):
    """Data assimilation functionality is not yet implemented."""
    warnings.warn(
        _DA_NOT_IMPLEMENTED_MSG.format(name=name),
        stacklevel=2,
    )
    raise AttributeError(
        f"module 'legoesm.da' has no attribute {name!r}. "
        "The DA module is under development."
    )
