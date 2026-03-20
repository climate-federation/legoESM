"""Deprecated alias for FV (C-D grid) ocean primitive equation model.

.. deprecated::
    Import from ``legoesm.ocean.dynamics.ocean_pe_cdgrid`` instead.
    The "FV" ocean module is the same C-D grid implementation.
"""

import warnings as _warnings

from legoesm.ocean.dynamics.ocean_pe_cdgrid import (
    ocean_baroclinic_tendencies_cdgrid as _real_fn,
)


def ocean_baroclinic_tendencies_fv(*args, **kwargs):
    """Deprecated wrapper — delegates to ocean_baroclinic_tendencies_cdgrid."""
    _warnings.warn(
        "ocean_baroclinic_tendencies_fv is deprecated; "
        "use ocean_baroclinic_tendencies_cdgrid from "
        "legoesm.ocean.dynamics.ocean_pe_cdgrid instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _real_fn(*args, **kwargs)


__all__ = ["ocean_baroclinic_tendencies_fv"]
