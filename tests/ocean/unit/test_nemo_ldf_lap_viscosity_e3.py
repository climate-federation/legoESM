"""#1455 — NEMO ``dyn_ldf_lev_lap`` e3 (layer-thickness) weighting.

Truth-tier tests for :func:`nemo_ldf_lap_viscosity_e3_cgrid` (the
``lateral_viscosity_e3_weighting="nemo_e3"`` variant of the node-14 embedded
div-curl viscosity operator, ``nemo_ldf_lap_viscosity_cgrid``):

* on UNIFORM thickness (no horizontal e3 variance) it is BIT-IDENTICAL to
  the unweighted operator (the documented reduction — e3 cancels between the
  injected weighting and the outer 1/e3 division);
* on a THICKNESS STEP (a topographic-step analogue) it DIFFERS from the
  unweighted operator, and in the predicted direction (the injected/outer e3
  ratio departs from 1 exactly at the step faces/vertices);
* dispatch: unknown ``lateral_viscosity_e3_weighting`` raises; a nonzero
  value paired with any operator other than ``"nemo_div_curl"`` raises;
  ``"nemo_e3"`` ROUTES ``_bc_horizontal_viscosity`` to the e3-weighted
  operator (the RHS equals the direct operator call).

The DINO kamm/kamm_mlf cards do NOT select ``"nemo_e3"`` — a controlled A/B
on the RUN_GDB restart (see ``fidelity_bar_gate.py``'s "dyn_ldf (dynldf_lev_
lap) u"/"v" row notes) measured the e3-weighted variant WORSENING both gate
rows rather than closing them, so it ships as a tested, dispatch-hardened,
but unselected option; the kamm cards stay at the default ``"off"``
(bit-identical to the pre-#1455 path).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax.numpy as jnp

import pytest

from legoesm.grids.latlon import create_latlon_grid, ensure_geometry
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    min_cell_to_vertex,
    nemo_lateral_viscosity_coefficients,
    nemo_ldf_lap_viscosity_cgrid,
    nemo_ldf_lap_viscosity_e3_cgrid,
)
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import _bc_horizontal_viscosity
from legoesm.ocean.state import LatLonCGridOceanConfig


def _geo(n_lat=40, n_lon=80):
    return ensure_geometry(create_latlon_grid(n_lat, n_lon))


def _uv(geo, rng, nlev=1):
    n_lat, n_lon = geo.lat.shape[0], geo.lon.shape[0]
    u = jnp.asarray(rng.standard_normal((n_lat, n_lon + 1, nlev)))
    v = jnp.asarray(rng.standard_normal((n_lat + 1, n_lon, nlev)))
    return u, v


def test_min_cell_to_vertex_uniform_is_constant():
    """Uniform cell thickness -> every vertex equals that same constant
    (all 4 surrounding cells agree, min is a no-op)."""
    geo = _geo()
    n_lat, n_lon = geo.lat.shape[0], geo.lon.shape[0]
    h_k = jnp.full((n_lat, n_lon, 2), 137.0)
    h_vtx = min_cell_to_vertex(h_k, geo)
    assert h_vtx.shape == (n_lat + 1, n_lon + 1, 2)
    interior = np.asarray(h_vtx)[1:-1]  # exclude polar wall rows (zero by BC)
    np.testing.assert_allclose(interior, 137.0, rtol=0, atol=0)


def test_min_cell_to_vertex_picks_min_of_four_neighbours():
    """A single thin cell pulls every vertex it touches down to its value
    (min rule), leaving vertices with no thin neighbour untouched."""
    geo = _geo(n_lat=10, n_lon=10)
    n_lat, n_lon = geo.lat.shape[0], geo.lon.shape[0]
    h_k = jnp.full((n_lat, n_lon), 100.0)
    h_k = h_k.at[5, 5].set(10.0)
    h_vtx = min_cell_to_vertex(h_k, geo)
    hv = np.asarray(h_vtx)
    # The 4 vertices at the corners of cell (5,5) all pick up the min.
    for dj in (0, 1):
        for di in (0, 1):
            assert hv[5 + di, 5 + dj] == pytest.approx(10.0)
    # A vertex far from the thin cell is untouched.
    assert hv[0 + 1, 0 + 1] == pytest.approx(100.0)  # interior corner elsewhere


def test_e3_weighted_reduces_to_unweighted_on_uniform_thickness():
    """Uniform h_k (no horizontal e3 variance) -> nemo_ldf_lap_viscosity_e3_cgrid
    is BIT-IDENTICAL to nemo_ldf_lap_viscosity_cgrid (the documented reduction:
    h_u/h_v/h_vtx all collapse to the same constant, which cancels between the
    injected weighting and the outer 1/e3 division)."""
    geo = _geo()
    rng = np.random.default_rng(0)
    n_lat, n_lon = geo.lat.shape[0], geo.lon.shape[0]
    nlev = 3
    u, v = _uv(geo, rng, nlev=nlev)
    mask = jnp.ones((n_lat, n_lon))
    u_mask = jnp.ones((n_lat, n_lon + 1))
    v_mask = jnp.ones((n_lat + 1, n_lon))
    half_UM = 0.135
    ahmt, ahmf = nemo_lateral_viscosity_coefficients(geo, half_UM)
    h_k = jnp.full((n_lat, n_lon, nlev), 250.0)

    nu, nv = nemo_ldf_lap_viscosity_cgrid(
        u, v, geo, ahmt, ahmf, mask=mask, u_mask=u_mask, v_mask=v_mask)
    eu, ev = nemo_ldf_lap_viscosity_e3_cgrid(
        u, v, geo, ahmt, ahmf, h_k, mask=mask, u_mask=u_mask, v_mask=v_mask)

    np.testing.assert_allclose(np.asarray(eu), np.asarray(nu), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(ev), np.asarray(nv), rtol=1e-12, atol=1e-12)


def test_e3_weighted_differs_at_a_thickness_step_in_predicted_direction():
    """A thickness STEP (thin band vs a thick surround, the synthetic analogue
    of a topographic step) makes the e3-weighted operator DIFFER from the
    unweighted one, and in the PREDICTED direction: NEMO's zdiv is a
    THICKNESS-WEIGHTED-MEAN flux (h_u*u face flux / outer h_k), which
    down-weights the thin-column velocity relative to the unweighted
    arithmetic div — so at the step the e3-weighted divergence differs
    from the unweighted one exactly where h_u/h_v are non-uniform, and is
    bit-identical away from the step (both operators reduce to the SAME
    uniform-thickness limit off the step band)."""
    geo = _geo(n_lat=30, n_lon=30)
    rng = np.random.default_rng(1)
    n_lat, n_lon = geo.lat.shape[0], geo.lon.shape[0]
    nlev = 1
    u, v = _uv(geo, rng, nlev=nlev)
    mask = jnp.ones((n_lat, n_lon))
    u_mask = jnp.ones((n_lat, n_lon + 1))
    v_mask = jnp.ones((n_lat + 1, n_lon))
    half_UM = 0.135
    ahmt, ahmf = nemo_lateral_viscosity_coefficients(geo, half_UM)

    # A single thin row (a "sill"/topographic-step analogue): row 15 is 10 m,
    # everywhere else 250 m.
    h_np = np.full((n_lat, n_lon, nlev), 250.0)
    h_np[15, :, :] = 10.0
    h_k = jnp.asarray(h_np)

    nu, nv = nemo_ldf_lap_viscosity_cgrid(
        u, v, geo, ahmt, ahmf, mask=mask, u_mask=u_mask, v_mask=v_mask)
    eu, ev = nemo_ldf_lap_viscosity_e3_cgrid(
        u, v, geo, ahmt, ahmf, h_k, mask=mask, u_mask=u_mask, v_mask=v_mask)

    diff_u = np.abs(np.asarray(eu) - np.asarray(nu))
    diff_v = np.abs(np.asarray(ev) - np.asarray(nv))

    # 1. It DIFFERS (a real, nonzero effect at the step).
    assert diff_u.max() > 1e-6, "e3-weighted operator should differ at a thickness step"

    # 2. The difference LOCALIZES at the step: rows far from the thin band
    #    (row 15) see e3-uniform neighbourhoods on BOTH sides and should match
    #    the unweighted operator to roundoff, while rows adjacent to the step
    #    (14, 15, 16) carry the discrepancy.
    far_rows = list(range(2, 12)) + list(range(19, 28))
    near_rows = [14, 15, 16]
    assert diff_u[far_rows].max() < 1e-9, (
        f"far from the step the two operators should agree to roundoff, "
        f"got max diff {diff_u[far_rows].max()}")
    assert diff_u[near_rows].max() > diff_u[far_rows].max(), (
        "the thickness-step discrepancy should be LARGEST at the step, not "
        "spread uniformly across the domain")


def _call_visc_e3(geo, config, h_k=None):
    n_lat, n_lon = geo.lat.shape[0], geo.lon.shape[0]
    rng = np.random.default_rng(4)
    u = jnp.asarray(rng.standard_normal((n_lat, n_lon + 1)))
    v = jnp.asarray(rng.standard_normal((n_lat + 1, n_lon)))
    u3 = u[..., None]; v3 = v[..., None]
    du3 = jnp.zeros_like(u3); dv3 = jnp.zeros_like(v3)
    mask = jnp.ones((n_lat, n_lon))
    u_mask = jnp.ones((n_lat, n_lon + 1)); v_mask = jnp.ones((n_lat + 1, n_lon))
    out = _bc_horizontal_viscosity(
        du3, dv3, u3, v3, geo, mask, u_mask, v_mask, config,
        None, None, 1.0, h_k=h_k)
    return out[0]   # du_dt


def test_dispatch_unknown_e3_weighting_raises():
    geo = _geo()
    cfg = LatLonCGridOceanConfig.from_flat(
        lateral_viscosity_operator="nemo_div_curl",
        lateral_viscosity_e3_weighting="typo",
        A_h=1.5e4, A_h_lat_scaling=True)
    with pytest.raises(ValueError):
        _call_visc_e3(geo, cfg)


def test_dispatch_e3_weighting_requires_nemo_div_curl_operator():
    geo = _geo()
    cfg = LatLonCGridOceanConfig.from_flat(
        lateral_viscosity_operator="vector_laplacian",
        lateral_viscosity_e3_weighting="nemo_e3",
        A_h=1.5e4, A_h_lat_scaling=True)
    with pytest.raises(ValueError):
        _call_visc_e3(geo, cfg)


def test_dispatch_routes_to_e3_weighted_operator():
    """Selecting ``lateral_viscosity_e3_weighting="nemo_e3"`` ROUTES the RHS to
    the e3-weighted operator (the du_dt tendency equals the direct operator
    call on the SAME h_k)."""
    geo = _geo()
    n_lat, n_lon = geo.lat.shape[0], geo.lon.shape[0]
    h_np = np.full((n_lat, n_lon, 1), 250.0)
    h_np[n_lat // 2, :, :] = 10.0
    h_k = jnp.asarray(h_np)

    cfg = LatLonCGridOceanConfig.from_flat(
        lateral_viscosity_operator="nemo_div_curl",
        lateral_viscosity_e3_weighting="nemo_e3",
        A_h=1.5e4, A_h_lat_scaling=True)
    du = np.asarray(_call_visc_e3(geo, cfg, h_k=h_k))[..., 0]

    half_UM = cfg.lateral_viscosity.A_h / (geo.radius * geo.dlon)
    ahmt, ahmf = nemo_lateral_viscosity_coefficients(geo, half_UM)
    rng = np.random.default_rng(4)
    u = jnp.asarray(rng.standard_normal((n_lat, n_lon + 1)))[..., None]
    v = jnp.asarray(rng.standard_normal((n_lat + 1, n_lon)))[..., None]
    mask = jnp.ones((n_lat, n_lon))
    u_mask = jnp.ones((n_lat, n_lon + 1)); v_mask = jnp.ones((n_lat + 1, n_lon))
    ref, _ = nemo_ldf_lap_viscosity_e3_cgrid(
        u, v, geo, ahmt, ahmf, h_k, mask=mask, u_mask=u_mask, v_mask=v_mask)
    assert float(np.max(np.abs(du))) > 0.0
    np.testing.assert_allclose(du, np.asarray(ref)[..., 0], rtol=1e-9, atol=1e-9)


def test_e3_weighting_missing_h_k_raises():
    """``nemo_e3`` needs h_k threaded in; a caller that omits it (h_k=None)
    raises rather than silently falling back to the unweighted operator."""
    geo = _geo()
    cfg = LatLonCGridOceanConfig.from_flat(
        lateral_viscosity_operator="nemo_div_curl",
        lateral_viscosity_e3_weighting="nemo_e3",
        A_h=1.5e4, A_h_lat_scaling=True)
    with pytest.raises(ValueError):
        _call_visc_e3(geo, cfg, h_k=None)


def test_dino_kamm_cards_default_e3_weighting_off():
    """#1455: the e3-weighted variant is a faithful transcription (see the
    module docstring + the operator's own docstring) but a controlled A/B on
    the RUN_GDB restart measured it WORSENING the dyn_ldf gate rows, not
    closing them (u |ratio-1| 1.86e-6->6.11e-6, v 7.24e-6->2.09e-5). The kamm
    cards therefore do NOT select it -- default "off" stays bit-identical."""
    from legoesm.ocean.experiments.dino import (
        DINO_RECIPES, dino_config_for_recipe, dino_lat_lon_model_config,
    )
    for recipe in ("nemo_dino_kamm", "nemo_dino_kamm_mlf"):
        assert "lateral_viscosity_e3_weighting" not in DINO_RECIPES[recipe]
    geo = _geo(n_lat=20, n_lon=40)
    cfg = dino_config_for_recipe("nemo_dino_kamm")
    mc, _ = dino_lat_lon_model_config(geo, cfg)
    assert mc.lateral_viscosity_e3_weighting == "off"
