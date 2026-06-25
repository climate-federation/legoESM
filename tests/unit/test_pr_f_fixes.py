"""PR F: deferred-item fixes — p_ceil config field + duogrid conservation warning.

(The gm_redi surface-complement equivalence, 4D-Var jit-once, topography
cross-face halo, and bitz discriminant floor are covered by their own suites:
test_visbeck_gm, test_incremental/test_pe_4dvar, test_topography, test_bitz_lipscomb.)
"""
from __future__ import annotations

import warnings

import jax.numpy as jnp
import pytest


def test_pe_configs_expose_p_ceil_default():
    """The surface-pressure ceiling is a per-dycore config field now (mirroring
    p_floor), not a hardcoded 2.0e6 literal in the clip — and it is tunable."""
    from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import (
        CDGridPrimitiveEquationConfig,
    )
    from legoesm.atmosphere.dynamics.primitive_eq_latlon_cgrid import (
        CGridLatLonPrimitiveEquationConfig,
    )
    from legoesm.atmosphere.dynamics.primitive_eq_mpas import (
        MPASPrimitiveEquationConfig,
    )

    for cfg_cls in (
        CDGridPrimitiveEquationConfig,
        CGridLatLonPrimitiveEquationConfig,
        MPASPrimitiveEquationConfig,
    ):
        assert cfg_cls().p_ceil == 2.0e6              # default == former literal
        assert cfg_cls(p_ceil=1.5e6).p_ceil == 1.5e6  # tunable


def test_duogrid_monotone_clip_warns_non_conservative():
    """fill_corner_region(monotone_clip=True) must warn that the clip is
    non-conservative; the default (False) is silent."""
    from legoesm.grids.duogrid import (
        create_duogrid_data,
        cube_rmp_vectorized,
        fill_corner_region,
    )

    dg = create_duogrid_data(8, ng=2, k2e_nord=2)
    halo = min(2, dg.ng)
    n_p = dg.n + 2 * halo
    padded = cube_rmp_vectorized(jnp.full((6, n_p, n_p), 7.0), dg, halo)

    with pytest.warns(UserWarning, match="NON-CONSERVATIVE"):
        fill_corner_region(padded, dg, halo, monotone_clip=True)

    # Default path must NOT warn.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fill_corner_region(padded, dg, halo)
