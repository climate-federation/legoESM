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

    # Positional-ABI invariant (codex PR F): existing positional callers pass
    # args up to and including ``p_ceil``, so nothing may be INSERTED before it
    # (that shifts every later field's positional binding).  APPENDING new fields
    # AFTER ``p_ceil`` WITH DEFAULTS is ABI-safe — those callers simply omit them.
    # So the fields after ``p_ceil`` must be EMPTY or exactly a known append-only
    # tail.  #836 appended the lat-lon C-grid top-sponge knobs after ``p_ceil``.
    _APPENDED_AFTER_P_CEIL = {
        # #771: moisture flux-form flag appended after p_ceil (default False).
        "CDGridPrimitiveEquationConfig": ("moisture_flux_form",),
        # #836: lat-lon C-grid top-sponge knobs appended after p_ceil (default OFF).
        "CGridLatLonPrimitiveEquationConfig": (
            "sponge_coeff", "sponge_width_m", "sponge_shape",
            "sponge_scale_height_m",
        ),
        # MPAS: nothing appended -> p_ceil is still last (allowed default ()).
    }
    for cfg_cls in (
        CDGridPrimitiveEquationConfig,
        CGridLatLonPrimitiveEquationConfig,
        MPASPrimitiveEquationConfig,
    ):
        assert cfg_cls().p_ceil == 2.0e6              # default == former literal
        assert cfg_cls(p_ceil=1.5e6).p_ceil == 1.5e6  # tunable
        fields = cfg_cls._fields
        tail = fields[fields.index("p_ceil") + 1:]
        allowed = _APPENDED_AFTER_P_CEIL.get(cfg_cls.__name__, ())
        assert tail == allowed, (
            f"{cfg_cls.__name__}: only the append-only tail {allowed} may follow "
            f"p_ceil (positional constructor ABI); got trailing fields {tail}. "
            "Inserting a field BEFORE p_ceil, or appending one without recording "
            "it here, breaks existing positional callers."
        )


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
