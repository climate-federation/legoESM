"""Unit tests for the NEMO ``vor_ene`` Sadourny 2-point ENERGY-conserving
PV-flux operator ``pv_flux_ene``.

``pv_flux_ene`` transcribes NEMO ``dynvor.F90::vor_ene`` (the GYRE default
``ln_dynvor_ene``) into the ``latlon_cgrid_operators`` index convention.  It is
the ENE sibling of ``pv_flux_al81_partial_cell`` (EEN / Arakawa-Lamb 12-point
triad).  These tests pin:

1. The EXACT 2-point stencil + weights + indices at interior u/v faces
   (a manufactured ``q = f + ζ`` field passed directly to the operator).
2. Energy conservation (the defining property of the Sadourny 1975 scheme):
   ``Σ h_u·u·F_u·area_u + h_v·v·F_v·area_v ≈ 0`` on a closed flat-bottom
   domain — the vorticity flux does no net work on the kinetic energy.
3. That ENE genuinely DIFFERS from the AL81/EEN default (not an alias).

Reuses the flat-bottom scaffolding from ``test_al81_budget`` so the two
operators are exercised on an identical state.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax
import jax.numpy as jnp

from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    pv_flux_ene,
    pv_flux_ens,
    pv_flux_al81_partial_cell,
    curl_vertex_cgrid,
    nemo_vor_ene_vorticity_cgrid,
)

# Reuse the exact flat-bottom state + energy budget from the AL81 test.
import sys as _sys
_sys.path.insert(0, os.path.dirname(__file__))
from test_al81_budget import _build_test_state, _energy_budget


def _apply_ene(state, f_vtx=None):
    zeta = curl_vertex_cgrid(state["u"], state["v"], state["grid"])
    F_u, F_v = pv_flux_ene(
        zeta, state["h_vtx"], state["h_v"], state["v"],
        state["h_u"], state["u"],
        state["u_mask_3d"], state["v_mask_3d"], state["vtx_mask"],
        f_vtx=f_vtx,
    )
    return zeta, F_u, F_v


def _apply_al81(state):
    zeta = curl_vertex_cgrid(state["u"], state["v"], state["grid"])
    F_u, F_v = pv_flux_al81_partial_cell(
        zeta, state["h_vtx"], state["h_v"], state["v"],
        state["h_u"], state["u"],
        state["u_mask_3d"], state["v_mask_3d"], state["vtx_mask"],
    )
    return zeta, F_u, F_v


# ----------------------------------------------------------------------

def test_ene_exact_2pt_form():
    """The operator reproduces NEMO's exact 2-point averaging at interior
    faces.  We bypass ``curl`` by feeding a known vertex vorticity ``zeta``
    and set ``h = 1``, all masks wet, so ``q = zeta + f_vtx`` — then the
    NEMO ``vor_ene`` formula is a closed hand computation.

        du[j,i] = ¼ ( q[j  ,i]·(v[j,i-1]+v[j,i])
                    + q[j+1,i]·(v[j+1,i-1]+v[j+1,i]) )
        dv[j,i] = -¼ ( q[j,i  ]·(u[j-1,i  ]+u[j,i  ])
                     + q[j,i+1]·(u[j-1,i+1]+u[j,i+1]) )
    """
    n_lat, n_lon, nlev = 6, 8, 1
    rng = np.random.default_rng(0)
    u = jnp.asarray(rng.standard_normal((n_lat, n_lon + 1, nlev)))
    v = jnp.asarray(rng.standard_normal((n_lat + 1, n_lon, nlev)))
    zeta = jnp.asarray(rng.standard_normal((n_lat + 1, n_lon + 1, nlev)))
    f_vtx = jnp.asarray(rng.standard_normal((n_lat + 1, n_lon + 1)))
    ones_h_vtx = jnp.ones_like(zeta)
    ones_h_v = jnp.ones_like(v)
    ones_h_u = jnp.ones_like(u)
    u_mask = jnp.ones_like(u)
    v_mask = jnp.ones_like(v)
    vtx_mask = jnp.ones((n_lat + 1, n_lon + 1))   # all wet → neumann-fill is a no-op

    du, dv = pv_flux_ene(
        zeta, ones_h_vtx, ones_h_v, v, ones_h_u, u,
        u_mask, v_mask, vtx_mask, f_vtx=f_vtx,
    )
    du = np.asarray(du); dv = np.asarray(dv)

    q = np.asarray(zeta) + np.asarray(f_vtx)[..., None]      # h = 1
    vN = np.asarray(v); uN = np.asarray(u)

    # interior u-face (not on the lon-wrap column, not a pole row)
    j, i = 2, 3
    exp_du = 0.25 * (
        q[j, i, 0] * (vN[j, i - 1, 0] + vN[j, i, 0])
        + q[j + 1, i, 0] * (vN[j + 1, i - 1, 0] + vN[j + 1, i, 0])
    )
    assert np.isclose(du[j, i, 0], exp_du, rtol=1e-12, atol=1e-12), (du[j, i, 0], exp_du)

    # interior v-face
    j, i = 3, 4
    exp_dv = -0.25 * (
        q[j, i, 0] * (uN[j - 1, i, 0] + uN[j, i, 0])
        + q[j, i + 1, 0] * (uN[j - 1, i + 1, 0] + uN[j, i + 1, 0])
    )
    assert np.isclose(dv[j, i, 0], exp_dv, rtol=1e-12, atol=1e-12), (dv[j, i, 0], exp_dv)


def test_ens_exact_total_vorticity_form_and_relative_only_mask():
    """Compiled vor_ens: mean(q) times mean(transport), with -V sign."""
    n_lat, n_lon, nlev = 6, 8, 1
    rng = np.random.default_rng(21)
    u = jnp.asarray(rng.standard_normal((n_lat, n_lon + 1, nlev)))
    v = jnp.asarray(rng.standard_normal((n_lat + 1, n_lon, nlev)))
    zeta = jnp.asarray(rng.standard_normal((n_lat + 1, n_lon + 1, nlev)))
    f_vtx = jnp.asarray(rng.standard_normal((n_lat + 1, n_lon + 1)))
    relmask = np.ones((n_lat + 1, n_lon + 1), dtype=np.float64)
    relmask[2, 3] = 0.0
    ones_vtx = jnp.ones_like(zeta)
    ones_u = jnp.ones_like(u)
    ones_v = jnp.ones_like(v)

    du, dv = pv_flux_ens(
        zeta, ones_vtx, ones_v, v, ones_u, u, ones_u, ones_v,
        jnp.asarray(relmask), ln_dynvor_msk=True,
        f_vtx=f_vtx,
    )
    du, dv = np.asarray(du), np.asarray(dv)
    f_np = np.asarray(f_vtx)[..., None]
    # Independent literal port of dynvor.f90:641-646: form CRV first, then
    # subtract f, mask only that recovered relative part, and add f again.
    total = np.add(f_np, np.asarray(zeta))
    relative = np.subtract(total, f_np)
    q = np.add(np.multiply(relative, relmask[..., None]), f_np)
    u_np, v_np = np.asarray(u), np.asarray(v)

    j, i = 2, 3
    expected_u = 0.125 * (q[j, i, 0] + q[j + 1, i, 0]) * (
        v_np[j, i - 1, 0] + v_np[j, i, 0]
        + v_np[j + 1, i - 1, 0] + v_np[j + 1, i, 0]
    )
    assert du[j, i, 0] == expected_u

    j, i = 2, 2
    expected_v = -0.125 * (q[j, i, 0] + q[j, i + 1, 0]) * (
        u_np[j - 1, i, 0] + u_np[j, i, 0]
        + u_np[j - 1, i + 1, 0] + u_np[j, i + 1, 0]
    )
    assert dv[j, i, 0] == expected_v
    # The zero relative mask must retain the planetary term at that vertex.
    assert q[2, 3, 0] == np.asarray(f_vtx)[2, 3]
    unmasked_u, unmasked_v = pv_flux_ens(
        zeta, ones_vtx, ones_v, v, ones_u, u, ones_u, ones_v,
        jnp.asarray(relmask), ln_dynvor_msk=False, f_vtx=f_vtx)
    assert any(not np.array_equal(masked, np.asarray(unmasked))
               for masked, unmasked in zip((du, dv), (unmasked_u, unmasked_v),
                                           strict=True))


def test_ens_is_jit_and_autodiff_safe_and_differs_from_ene():
    state = _build_test_state(n_lat=8, n_lon=12, partial_cells=False, seed=23)
    from legoesm.grids.latlon import ensure_geometry
    grid = ensure_geometry(state["grid"])
    f_vtx = jnp.ones(state["h_vtx"].shape[:2]) * 1.0e-4
    relmask = state["vtx_mask"]

    def operator(u):
        zeta = nemo_vor_ene_vorticity_cgrid(u, state["v"], grid)
        return pv_flux_ens(
            zeta, state["h_vtx"], state["h_v"], state["v"],
            state["h_u"], u, state["u_mask_3d"], state["v_mask_3d"],
            state["vtx_mask"], ln_dynvor_msk=True,
            f_vtx=f_vtx,
        )

    eager = operator(state["u"])
    compiled = jax.jit(operator)(state["u"])
    for left, right in zip(eager, compiled, strict=True):
        assert np.allclose(
            np.asarray(left), np.asarray(right), rtol=1.0e-14, atol=0.0)

    def loss(u):
        du, dv = operator(u)
        return jnp.sum(du * du) + jnp.sum(dv * dv)

    grad = jax.jit(jax.grad(loss))(state["u"])
    assert np.all(np.isfinite(np.asarray(grad)))

    zeta = curl_vertex_cgrid(state["u"], state["v"], state["grid"])
    ene = pv_flux_ene(
        zeta, state["h_vtx"], state["h_v"], state["v"], state["h_u"],
        state["u"], state["u_mask_3d"], state["v_mask_3d"],
        state["vtx_mask"], f_vtx=f_vtx,
    )
    assert any(not np.array_equal(np.asarray(a), np.asarray(b))
               for a, b in zip(eager, ene, strict=True))


def test_ens_total_production_dispatch_matches_direct_operator():
    """The shared dynamics dispatcher selects ENS, not an ENE/EEN alias."""
    from legoesm.grids.latlon import ensure_geometry
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import _bc_pv_flux

    state = _build_test_state(n_lat=8, n_lon=12, partial_cells=False, seed=29)
    grid = ensure_geometry(state["grid"])
    f_vtx = jnp.full(state["h_vtx"].shape[:2], 1.0e-4)
    reciprocals = (1.0 / grid.dx_u, 1.0 / grid.dy_v)
    zeros_u, zeros_v = jnp.zeros_like(state["u"]), jnp.zeros_like(state["v"])
    _, _, dispatched_u, dispatched_v = _bc_pv_flux(
        zeros_u, zeros_v, state["u"], state["v"],
        state["h_u"], state["h_v"], state["h_T"],
        state["u_mask_3d"], state["v_mask_3d"], None, grid,
        "vector_invariant", vertex_mask=state["vtx_mask"],
        vorticity_scheme="ens_total", een_q_boundary="nemo_live",
        ln_dynvor_msk=True,
        een_e3f_scheme="nemo_avg4", een_metric_weighting="nemo",
        dz_ref=jnp.asarray([4000.0]),
        h_vtx_override=state["h_vtx"], f_vtx_override=f_vtx,
        metric_reciprocals=reciprocals,
    )
    zeta = nemo_vor_ene_vorticity_cgrid(state["u"], state["v"], grid)
    direct_u, direct_v = pv_flux_ens(
        zeta, state["h_vtx"], state["h_v"], state["v"], state["h_u"],
        state["u"], state["u_mask_3d"], state["v_mask_3d"],
        state["vtx_mask"], ln_dynvor_msk=True, f_vtx=f_vtx,
        q_boundary="nemo_live",
        metric_widths=(grid.dx_u, grid.dx_v, grid.dy_u, grid.dy_v),
        metric_reciprocals=reciprocals,
    )
    np.testing.assert_array_equal(np.asarray(dispatched_u), np.asarray(direct_u))
    np.testing.assert_array_equal(np.asarray(dispatched_v), np.asarray(direct_v))


def test_ens_requires_explicit_boolean_mask_switch_before_array_use():
    """The production ENS dispatcher has no implicit mask selection."""
    import pytest
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import _bc_pv_flux

    for absent_or_unknown in (None, "true", 1):
        with pytest.raises(ValueError, match="explicit boolean ln_dynvor_msk"):
            _bc_pv_flux(
                None, None, None, None, None, None, None, None, None,
                None, None, None, vorticity_scheme="ens_total",
                ln_dynvor_msk=absent_or_unknown,
            )


def test_mask_switch_refused_outside_ens():
    """ln_dynvor_msk is consumed only by ens_total; elsewhere it would be inert."""
    import pytest
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import _bc_pv_flux

    for scheme in ("al81", "ene", "ene_total", "een_total", "een_planetary"):
        with pytest.raises(ValueError, match="only for ens_total"):
            _bc_pv_flux(
                None, None, None, None, None, None, None, None, None,
                None, None, None, vorticity_scheme=scheme,
                ln_dynvor_msk=True,
            )


def test_ens_direct_thickness_division_bypasses_ene_floor():
    """Compiled dynvor.f90:659-660 divides by e3f without a floor."""
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_swg_card

    swg_mask_switch = (
        build_swg_card().recipe.model_config.vorticity_scheme.ln_dynvor_msk)
    assert swg_mask_switch is True
    n_lat, n_lon, nlev = 3, 4, 1
    tiny = np.float64(1.0e-20)
    zeta = jnp.zeros((n_lat + 1, n_lon + 1, nlev), dtype=jnp.float64)
    f_vtx = jnp.ones((n_lat + 1, n_lon + 1), dtype=jnp.float64)
    h_vtx = jnp.full_like(zeta, tiny)
    u = jnp.ones((n_lat, n_lon + 1, nlev), dtype=jnp.float64)
    v = jnp.ones((n_lat + 1, n_lon, nlev), dtype=jnp.float64)
    ones_u, ones_v = jnp.ones_like(u), jnp.ones_like(v)
    vtx_mask = jnp.ones((n_lat + 1, n_lon + 1), dtype=jnp.float64)

    ens = pv_flux_ens(
        zeta, h_vtx, ones_v, v, ones_u, u, ones_u, ones_v, vtx_mask,
        ln_dynvor_msk=swg_mask_switch, f_vtx=f_vtx,
        q_boundary="nemo_live")
    guarded = pv_flux_ene(
        zeta, h_vtx, ones_v, v, ones_u, u, ones_u, ones_v, vtx_mask,
        f_vtx=f_vtx, q_boundary="nemo_live")
    assert all(np.all(np.isfinite(np.asarray(a))) for a in ens)
    assert any(not np.array_equal(np.asarray(a), np.asarray(b))
               for a, b in zip(ens, guarded, strict=True))
    # The 1e-10 historical ENE denominator versus the 1e-20 direct NEMO
    # denominator yields exactly the planted ten-order response separation.
    assert np.max(np.abs(np.asarray(ens[0]))) > 1.0e9 * np.max(
        np.abs(np.asarray(guarded[0])))


def test_ene_default_total_vorticity_path_is_byte_identical_with_override_seam():
    """Rule 12: the new ENS-only operand seam cannot move an ENE caller."""
    state = _build_test_state(n_lat=8, n_lon=12, partial_cells=False, seed=31)
    f_vtx = jnp.full(state["h_vtx"].shape[:2], 1.0e-4)
    zeta = curl_vertex_cgrid(state["u"], state["v"], state["grid"])
    args = (
        zeta, state["h_vtx"], state["h_v"], state["v"], state["h_u"],
        state["u"], state["u_mask_3d"], state["v_mask_3d"],
        state["vtx_mask"],
    )
    historical = pv_flux_ene(*args, f_vtx=f_vtx)
    explicit = pv_flux_ene(
        *args, f_vtx=f_vtx,
        total_vorticity_override=zeta + f_vtx[..., jnp.newaxis])
    for left, right in zip(historical, explicit, strict=True):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


def test_ene_energy_conservation_flat_bottom():
    """Sadourny 1975: the ENE vorticity flux does zero net work on KE.
    On a closed flat-bottom domain ``Σ h·u·F_u·A + h·v·F_v·A`` is round-off
    relative to the total KE.  (ENE conserves energy but NOT enstrophy — the
    complement of AL81/EEN which conserves both.)
    """
    state = _build_test_state(n_lat=12, n_lon=24, partial_cells=False, seed=7)
    _, F_u, F_v = _apply_ene(state)
    KE, dKE, rel = _energy_budget(state, F_u, F_v)
    assert rel < 1e-7, f"ENE energy injection {rel:.3e} (dKE={dKE:.3e}, KE={KE:.3e})"


def test_ene_differs_from_al81():
    """ENE (2-point Sadourny) and AL81 (12-point triad) are DIFFERENT
    operators — a typo-proofing regression so the dispatch never silently
    aliases them.  They agree only in the smooth limit, not on a random
    grid-scale field.
    """
    state = _build_test_state(n_lat=12, n_lon=24, partial_cells=False, seed=11)
    _, Fu_ene, Fv_ene = _apply_ene(state)
    _, Fu_al, Fv_al = _apply_al81(state)
    assert not np.allclose(np.asarray(Fu_ene), np.asarray(Fu_al), atol=1e-10)
    assert not np.allclose(np.asarray(Fv_ene), np.asarray(Fv_al), atol=1e-10)


def test_ene_differentiable():
    """Pure-jnp / JIT-safe / autodiff-safe (legoESM AD requirement)."""
    state = _build_test_state(n_lat=8, n_lon=12, partial_cells=False, seed=3)

    def loss(u):
        zeta = curl_vertex_cgrid(u, state["v"], state["grid"])
        Fu, Fv = pv_flux_ene(
            zeta, state["h_vtx"], state["h_v"], state["v"],
            state["h_u"], u, state["u_mask_3d"], state["v_mask_3d"],
            state["vtx_mask"],
        )
        return jnp.sum(Fu ** 2) + jnp.sum(Fv ** 2)

    g = jax.jit(jax.grad(loss))(state["u"])
    assert np.all(np.isfinite(np.asarray(g)))


def test_ene_nemo_metric_source_route_eager_jit_grad_and_red_operand():
    """The compiled ENE metric route is eager/JIT/AD safe and reciprocal-live."""
    state = _build_test_state(n_lat=8, n_lon=12, partial_cells=False, seed=13)
    from legoesm.grids.latlon import ensure_geometry
    grid = ensure_geometry(state["grid"])
    widths = (grid.dx_u, grid.dx_v, grid.dy_u, grid.dy_v)
    reciprocals = (1.0 / grid.dx_u, 1.0 / grid.dy_v)

    def operator(u, reciprocal_pair):
        zeta = nemo_vor_ene_vorticity_cgrid(u, state["v"], grid)
        return pv_flux_ene(
            zeta, state["h_vtx"], state["h_v"], state["v"],
            state["h_u"], u, state["u_mask_3d"], state["v_mask_3d"],
            state["vtx_mask"], q_boundary="nemo_live",
            metric_widths=widths, metric_reciprocals=reciprocal_pair,
        )

    eager = operator(state["u"], reciprocals)
    compiled = jax.jit(operator)(state["u"], reciprocals)
    for left, right in zip(eager, compiled, strict=True):
        assert np.all(np.isfinite(np.asarray(left)))
        assert np.allclose(np.asarray(left), np.asarray(right), rtol=1e-14, atol=0.0)

    def loss(u):
        du, dv = operator(u, reciprocals)
        return jnp.sum(du ** 2) + jnp.sum(dv ** 2)

    grad = jax.jit(jax.grad(loss))(state["u"])
    assert np.all(np.isfinite(np.asarray(grad)))

    bumped = np.asarray(reciprocals[0]).copy()
    bumped[2, 3] = np.nextafter(bumped[2, 3], np.inf)
    planted = operator(state["u"], (jnp.asarray(bumped), reciprocals[1]))
    assert any(not np.array_equal(np.asarray(a), np.asarray(b))
               for a, b in zip(eager, planted, strict=True))


def test_bc_pv_flux_rejects_unknown_vorticity_scheme():
    """Dispatch hardening: an unknown ``vorticity_scheme`` raises at ``_bc_pv_flux``
    entry (before any array use), so a typo fails loudly even on the WENO path.
    Registered in ``tests/test_dispatch_hardening.py::BASELINE_DISPATCHERS``.
    (The al81/ene branch selection itself is covered by the direct-operator tests
    above + oracle-scaffold end-to-end verification of the config path.)"""
    import pytest
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import _bc_pv_flux
    with pytest.raises(ValueError, match="unknown vorticity_scheme"):
        _bc_pv_flux(
            None, None, None, None, None, None, None, None, None, None, None, None,
            vorticity_scheme="bogus",
        )
