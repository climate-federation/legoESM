"""Data assimilation module for legoESM.

This module will contain ensemble-based and variational data assimilation
methods for parameter estimation and uncertainty quantification.

Current Status: Under development
"""

__all__ = []


def __getattr__(name):
    """Data assimilation functionality is not yet implemented."""
    raise NotImplementedError(
        f"Data assimilation module is under development. "
        f"Cannot access '{name}'."
    )
