"""Data assimilation module for legoESM (**experimental / stub**).

.. warning::

   This module is **experimental** and provides only stub interfaces.
   No data assimilation functionality is implemented yet.  Importing
   this module directly (``from legoesm.da import ...``) is fine, but
   it is intentionally **not** re-exported from the top-level
   ``legoesm`` package to avoid polluting the public namespace.

Future plans include ensemble-based and variational data assimilation
methods for parameter estimation and uncertainty quantification.
"""

import logging
import warnings

__all__: list[str] = []

logger = logging.getLogger(__name__)

_DA_NOT_IMPLEMENTED_MSG = (
    "legoesm.da is an experimental stub. "
    "'{name}' is not yet implemented.  "
    "See the module docstring for current status."
)


def __getattr__(name):
    """Data assimilation functionality is not yet implemented."""
    warnings.warn(
        _DA_NOT_IMPLEMENTED_MSG.format(name=name),
        stacklevel=2,
    )
    raise AttributeError(
        f"module 'legoesm.da' has no attribute {name!r}. "
        "The DA module is an experimental stub."
    )
