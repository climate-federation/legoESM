"""Regression guard: surface temperature is named ``T_sfc`` everywhere (#7).

CLAUDE.md: surface temperature is ``T_sfc`` (no ``T_surface``/``Ts``).  After the
T_surface -> T_sfc unification, lock the public surface pytrees so the old
spelling cannot creep back into a field name (incl. compound names like the
accumulator's ``sum_T_surface``, which a bare-word rename would miss).
"""

from __future__ import annotations

from legoesm.coupler.accumulator import FluxAccumulator
from legoesm.core.coupling_fields import AtmToSurface, SurfaceToAtm, TileResponse


def test_public_surface_pytrees_use_T_sfc_not_T_surface():
    for nt in (TileResponse, SurfaceToAtm, FluxAccumulator):
        for field in nt._fields:
            assert "T_surface" not in field, (
                f"{nt.__name__}.{field} still uses 'T_surface'; canonical is 'T_sfc'")
    # the surface skin temperature is present under the canonical name
    assert "T_sfc" in TileResponse._fields
    assert "T_sfc" in SurfaceToAtm._fields
    assert "sum_T_sfc" in FluxAccumulator._fields
    # AtmToSurface carries the atmosphere's lowest-level temperature, not a
    # surface skin temp, so it simply must not regress to a 'T_surface' field.
    assert all("T_surface" not in f for f in AtmToSurface._fields)
