"""FV3_3D iter 300: AST regression guard for the iter-189
``dt_actual`` parameter on both PE and NH tendency signatures.

iter-296/297/298/299 pinned the runtime semantics of the
``dt_actual`` plumbing.  iter-300 adds a structural guard
ensuring the parameter exists with the right name + default
on both 3D-path tendency functions.

If a future refactor renames ``dt_actual`` to ``dt`` (or
removes the keyword), runtime tests still pass when called
positionally — but downstream callers that use keyword
``dt_actual=...`` break silently.  This signature guard
catches the rename.

Tests
-----

1. ``test_pe_tendency_dt_actual_in_signature`` — parameter
   ``dt_actual`` exists in ``fv3_hydrostatic_tendencies`` with
   default ``None``.
2. ``test_nh_tendency_dt_actual_in_signature`` — same on
   ``cdgrid_compressible_euler_slow_tendencies``.
"""
from __future__ import annotations

import inspect

from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
    cdgrid_compressible_euler_slow_tendencies,
)
from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    fv3_hydrostatic_tendencies,
)


def test_pe_tendency_dt_actual_in_signature():
    """``fv3_hydrostatic_tendencies`` must expose ``dt_actual``
    keyword with default ``None`` (iter-189 plumbing API)."""
    sig = inspect.signature(fv3_hydrostatic_tendencies)
    assert "dt_actual" in sig.parameters, (
        "PE tendency function must keep ``dt_actual`` as a "
        "keyword parameter — iter-189 callers (model.step) and "
        "iter-298 fallback semantics depend on this name."
    )
    assert sig.parameters["dt_actual"].default is None, (
        "``dt_actual`` must default to None to preserve the "
        "iter-298 fallback to ``corner_div_damp_dt_proxy``."
    )


def test_nh_tendency_dt_actual_in_signature():
    """``cdgrid_compressible_euler_slow_tendencies`` must
    expose ``dt_actual`` keyword with default ``None``."""
    sig = inspect.signature(cdgrid_compressible_euler_slow_tendencies)
    assert "dt_actual" in sig.parameters, (
        "NH tendency function must keep ``dt_actual`` as a "
        "keyword parameter — iter-189 callers (model.step) and "
        "iter-299 fallback semantics depend on this name."
    )
    assert sig.parameters["dt_actual"].default is None, (
        "``dt_actual`` must default to None to preserve the "
        "iter-299 fallback to ``corner_div_damp_dt_proxy``."
    )
