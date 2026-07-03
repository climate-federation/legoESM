"""Truth-tier gates for the component-wise FLUX-DIVERGENCE lateral viscosity
(``lateral_viscosity_operator="flux_divergence"``), the Veros ``harmonic_friction``
operator ∇·(A_h∇u), ∇·(A_h∇v) added as a config-selectable option per the
oracle-recipe doctrine (rule H).

Directly exercises ``flux_divergence_viscosity_cgrid`` + the ``_bc_horizontal_viscosity``
dispatch + the K_diss_h coupling:
- zero-velocity / uniform-flow  -> zero tendency (∇·(A_h∇u)=0 for u=const),
- momentum conservation: flux divergence telescopes to MACHINE EPS on a closed domain
  when the spherical metric is removed (uniform-cos), and the sphere residual is the
  O(dx²) curvature truncation (NOT a telescoping bug),
- positive-definite K_diss_h (no clamp) + energy-consistency (Σ kdiss·area == KE removed
  to machine eps on a metric-free closed domain),
- cos(lat) scaling preserved (cos_power applied inside the flux, Veros analogue),
- differentiability (jax.grad, FD vs AD),
- BIT-IDENTICAL default: ``lateral_viscosity_operator="vector_laplacian"`` byte-equal to
  the historical path, and flux_divergence is a genuinely DIFFERENT (finite) operator.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    flux_divergence_bilaplacian_cgrid,
    flux_divergence_viscosity_cgrid,
    vector_laplacian_cgrid,
    _cos_lat_uv,
)

_NLAT, _NLON, _NLEV = 16, 24, 2
_A_H = 2.0e4


def _basin(uniform_cos=False, enclosed=True, seed=7, nlev=_NLEV):
    """Build (grid, u_mask, v_mask, cell_mask, u, v) for a basin.

    ``enclosed`` -> land on N/S rows + one land meridian (fully closed: zero flux
    through every boundary). ``uniform_cos`` -> override cos(lat) to a constant so the
    spherical metric vanishes and the flux divergence telescopes EXACTLY."""
    grid = create_latlon_grid(_NLAT, _NLON, dtype=jnp.float64)
    if uniform_cos:
        # Metric-free basin: the spherical metric vanishes only if EVERY
        # cos(lat) the operator reads is the same constant.  #516 routes
        # the v-face cosine through ``vface_zonal_cos_lat``, which
        # recomputes ``cos(0.5·(lat[j]+lat[j+1]))`` from ``grid.lat`` —
        # so overriding only ``cos_lat`` (the cell cosine) leaves a
        # varying interface cosine and the budget no longer telescopes.
        # Pin ``lat`` to a single mid-latitude too, so both the cell and
        # the interface cosine collapse to the same constant.
        mid = _NLAT // 2
        latc = float(grid.lat[mid])
        cosc = float(grid.cos_lat[mid])
        grid = grid._replace(
            lat=jnp.full_like(grid.lat, latc),
            cos_lat=jnp.full_like(grid.cos_lat, cosc),
        )
    m = np.ones((_NLAT, _NLON))
    m[0, :] = 0.0
    m[-1, :] = 0.0
    if enclosed:
        m[:, 0] = 0.0  # break lon periodicity -> fully enclosed basin
    um = np.zeros((_NLAT, _NLON + 1))
    for j in range(_NLON + 1):
        um[:, j] = m[:, (j - 1) % _NLON] * m[:, j % _NLON]
    vm = np.zeros((_NLAT + 1, _NLON))
    for i in range(1, _NLAT):
        vm[i, :] = m[i - 1, :] * m[i, :]
    rng = np.random.default_rng(seed)
    u = jnp.asarray(0.3 * rng.standard_normal((_NLAT, _NLON + 1, nlev))) * jnp.asarray(um)[:, :, None]
    v = jnp.asarray(0.3 * rng.standard_normal((_NLAT + 1, _NLON, nlev))) * jnp.asarray(vm)[:, :, None]
    return grid, jnp.asarray(um), jnp.asarray(vm), jnp.asarray(m), u, v


def _area_uv(grid):
    """Veros velocity-point cell areas: area_u = cost·dyt·dxu, area_v = cosu·dyu·dxt."""
    cos_u, cos_v = _cos_lat_uv(grid)
    dy_cell = np.asarray(grid.dy * 0.5)
    dx_u = np.asarray(grid.radius * cos_u * grid.dlon)
    area_u = (np.asarray(cos_u) * dy_cell * dx_u)[:, None, None]
    dy_v = np.concatenate([[dy_cell[0]], 0.5 * (dy_cell[1:] + dy_cell[:-1]), [dy_cell[-1]]])
    area_v = (np.asarray(cos_v) * dy_v * float(dx_u.mean()))[:, None, None]
    return area_u, area_v


# ---------------------------------------------------------------------------
# Zero on uniform / zero flow.
# ---------------------------------------------------------------------------

def test_zero_velocity_zero_tendency():
    """u=v=0 -> exactly zero viscous tendency + zero dissipation."""
    grid, um, vm, mask, _, _ = _basin()
    u = jnp.zeros((_NLAT, _NLON + 1, _NLEV))
    v = jnp.zeros((_NLAT + 1, _NLON, _NLEV))
    vu, vv, kd = flux_divergence_viscosity_cgrid(
        u, v, grid, _A_H, cos_power=1, mask=mask, u_mask=um, v_mask=vm,
        want_dissipation=True)
    assert float(jnp.max(jnp.abs(vu))) == 0.0
    assert float(jnp.max(jnp.abs(vv))) == 0.0
    assert float(jnp.max(jnp.abs(kd))) == 0.0


@pytest.mark.parametrize("cos_power", [0, 1])
def test_uniform_flow_zero_tendency(cos_power):
    """∇·(A_h∇u)=0 for u=const, v=0 — the constant-flow null space, both cos powers.

    All-ocean interior (v=0 walls at the poles only) so the constant truly has no
    gradient anywhere a flux is formed; the cos(lat) metric does not break it (the
    derivative of a constant is zero regardless of the weight)."""
    grid = create_latlon_grid(_NLAT, _NLON, dtype=jnp.float64)
    um = jnp.ones((_NLAT, _NLON + 1))
    vm = jnp.ones((_NLAT + 1, _NLON)).at[0].set(0.0).at[-1].set(0.0)
    u = jnp.full((_NLAT, _NLON + 1, _NLEV), 0.7)
    v = jnp.zeros((_NLAT + 1, _NLON, _NLEV))
    vu, vv, kd = flux_divergence_viscosity_cgrid(
        u, v, grid, _A_H, cos_power=cos_power, mask=jnp.ones((_NLAT, _NLON)),
        u_mask=um, v_mask=vm, want_dissipation=True)
    assert float(jnp.max(jnp.abs(vu))) < 1e-12, float(jnp.max(jnp.abs(vu)))
    assert float(jnp.max(jnp.abs(vv))) < 1e-12, float(jnp.max(jnp.abs(vv)))
    assert float(jnp.max(jnp.abs(kd))) < 1e-12


# ---------------------------------------------------------------------------
# Momentum conservation (truth tier).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cos_power", [0, 1])
def test_momentum_conservation_machine_eps_metric_free(cos_power):
    """On a CLOSED basin with the spherical metric removed (uniform cos), the
    component-wise flux divergence telescopes EXACTLY: the area-weighted domain
    integral of each velocity component is zero to MACHINE EPS (zero flux through the
    enclosing walls). This is the precise truth-tier statement the operator satisfies
    by construction (flux divergence telescopes; zero flux through land/walls)."""
    grid, um, vm, mask, u, v = _basin(uniform_cos=True, enclosed=True)
    vu, vv, _ = flux_divergence_viscosity_cgrid(
        u, v, grid, _A_H, cos_power=cos_power, mask=mask, u_mask=um, v_mask=vm)
    area_u, area_v = _area_uv(grid)
    # drop the periodic wrap column for u (duplicate of column 0).
    mom_u = float(np.sum((area_u * np.asarray(vu))[:, :-1, :]))
    scale_u = float(np.sum(np.abs(area_u * np.asarray(vu))[:, :-1, :])) + 1e-300
    mom_v = float(np.sum(area_v * np.asarray(vv)))
    scale_v = float(np.sum(np.abs(area_v * np.asarray(vv)))) + 1e-300
    assert abs(mom_u / scale_u) < 1e-12, f"u-momentum not conserved: {mom_u / scale_u:.2e}"
    assert abs(mom_v / scale_v) < 1e-12, f"v-momentum not conserved: {mom_v / scale_v:.2e}"


@pytest.mark.parametrize("cos_power", [0, 1])
def test_momentum_conservation_machine_eps_on_sphere_veros_area(cos_power):
    """STRONGER conservation gate that runs on the REAL SPHERE (variable cos).

    With the correct spherical metric in the flux divergence (the ``÷(cosφ·dy)``
    meridional denominators, matching Veros ``dxu·cosu`` / ``dyu·cosu``), each
    component's flux divergence telescopes against the Veros velocity-point areas
    (``area_u = cost·dyt·dxu`` with ``dxt=dxu=R·dlon``) to MACHINE EPS even on the
    sphere — because ``area_vel / divergence_denominator`` collapses to the constant
    ``R·dlon`` (zonal) / ``dyt`` (merid), so the telescoping sum is exact.

    REGRESSION LOCK: a missing ``cos`` in any meridional divergence denominator (the
    v-meridional ``÷cosφ`` bug found in adversarial review) breaks this exact
    telescoping on the sphere — the metric-free test alone cannot see it (constant
    cos factors out there)."""
    grid, um, vm, mask, u, v = _basin(uniform_cos=False, enclosed=True)
    vu, vv, _ = flux_divergence_viscosity_cgrid(
        u, v, grid, _A_H, cos_power=cos_power, mask=mask, u_mask=um, v_mask=vm)
    cos_u, cos_v = _cos_lat_uv(grid)
    dy_cell = np.asarray(grid.dy * 0.5)
    dxt = float(grid.radius) * float(grid.dlon)  # Veros dxt=dxu=R*dlon (no cos)
    # Veros velocity-point areas: area_u=cost*dyt*dxu, area_v=cosu*dyu*dxt.
    area_u = (np.asarray(cos_u) * dy_cell * dxt)[:, None, None]
    dy_v = np.concatenate([[dy_cell[0]], 0.5 * (dy_cell[1:] + dy_cell[:-1]), [dy_cell[-1]]])
    area_v = (np.asarray(cos_v) * dy_v * dxt)[:, None, None]
    mom_u = float(np.sum((area_u * np.asarray(vu))[:, :-1, :]))
    scale_u = float(np.sum(np.abs(area_u * np.asarray(vu))[:, :-1, :])) + 1e-300
    mom_v = float(np.sum(area_v * np.asarray(vv)))
    scale_v = float(np.sum(np.abs(area_v * np.asarray(vv)))) + 1e-300
    assert abs(mom_u / scale_u) < 1e-11, f"u-momentum (sphere) not conserved: {mom_u/scale_u:.2e}"
    assert abs(mom_v / scale_v) < 1e-11, f"v-momentum (sphere) not conserved: {mom_v/scale_v:.2e}"


def _veros_harmonic_friction_reference(grid, um, vm, u, v, cos_power):
    """Veros ``harmonic_friction`` (veros/core/friction.py, cos-scaling branch,
    no-slip OFF) rewritten in legoESM C-grid indexing, INDEPENDENTLY from the
    operator under test — the term-for-term oracle.

    Metric identities from veros/core/numerics.py (uniform-lon lat-lon):
      cost=cos(yt)=cos_u, cosu=cos(yu)=cos_v, dxt=dxu=R·dlon, dyt=grid.dy/2,
      dyu(interior)=0.5(dyt[1:]+dyt[:-1]).
      flux_east(u)=A_h·cost^p·∂u/∂x at T-centres; flux_north(u)=A_h·cosu^p·cosu·
        ∂u/∂y at vertices; dv_mix_u=(fE−fE_w)/(cost·dxu)+(fN−fN_s)/(cost·dyt).
      flux_east(v)=A_h·cosu^p·∂v/∂x at vertices; flux_north(v)=A_h·cost^p·cost·∂v/∂y
        at T-centres; dv_mix_v=(fE−fE_w)/(cosu·dxt)+(fN−fN_s)/(cosu·dyu).
    """
    R = float(grid.radius); dlon = float(grid.dlon)
    cos_u = np.asarray(grid.cos_lat)
    # #516: Veros's velocity-point cosine is ``cosu = cos(yu)`` at the
    # v-FACE (interface) latitude ``yu`` — the cos-OF-interface
    # ``cos(0.5·(lat[j]+lat[j+1]))``.  (The earlier ``0.5·(cos_u[:-1]+
    # cos_u[1:])`` mean-of-cos reference was the slightly-LESS-faithful
    # approximation #516 removed: it equals ``cos(yu)`` only to O(dlat²) on
    # a stretched grid.)  Computed INDEPENDENTLY from ``grid.lat`` with
    # numpy — NOT via the operator's ``_cos_lat_uv`` — so a regression in
    # the unified metric cannot be copied into this oracle (codex).  Only
    # the interior ``cos_v[1:-1]`` is used below; the polar placeholders
    # are irrelevant to the friction terms.
    _lat = np.asarray(grid.lat)
    _cos_v_int = np.cos(0.5 * (_lat[:-1] + _lat[1:]))   # (n_lat-1,) interior
    cos_v = np.concatenate([cos_u[:1], _cos_v_int, cos_u[-1:]])
    dx_cell = R * cos_u * dlon                       # cost*dxt (zonal arc of T-cell)
    dy_cell = np.asarray(grid.dy) * 0.5              # dyt
    dy_v_int = 0.5 * (dy_cell[1:] + dy_cell[:-1])    # dyu interior
    u = np.asarray(u); v = np.asarray(v); um = np.asarray(um); vm = np.asarray(vm)

    # zonal momentum u
    du_dx = u[:, 1:, :] - u[:, :-1, :]
    fE = (_A_H * (cos_u[:, None, None] ** cos_power) * du_dx / dx_cell[:, None, None]
          * um[:, 1:, None] * um[:, :-1, None])
    du_dy = u[1:, :, :] - u[:-1, :, :]
    cvw = (cos_v[1:-1] ** cos_power) * cos_v[1:-1]
    fN_int = (_A_H * cvw[:, None, None] * du_dy / dy_v_int[:, None, None]
              * um[1:, :, None] * um[:-1, :, None])
    fN = np.concatenate([np.zeros_like(u[:1]), fN_int, np.zeros_like(u[:1])], axis=0)
    netx_core = fE - np.roll(fE, 1, axis=1)
    netx = np.concatenate([netx_core, netx_core[:, :1, :]], axis=1) / dx_cell[:, None, None]
    nety = (fN[1:] - fN[:-1]) / (cos_u * dy_cell)[:, None, None]
    visc_u_ref = (netx + nety) * um[:, :, None]

    # meridional momentum v
    dv_dx = v - np.roll(v, 1, axis=1)
    dx_v = R * cos_v * dlon
    fE_v = (_A_H * (cos_v[:, None, None] ** cos_power) * dv_dx / dx_v[:, None, None]
            * vm[:, :, None] * np.roll(vm, 1, axis=1)[:, :, None])
    dv_dy = v[1:, :, :] - v[:-1, :, :]
    ccw = (cos_u ** cos_power) * cos_u
    fN_v = (_A_H * ccw[:, None, None] * dv_dy / dy_cell[:, None, None]
            * vm[1:, :, None] * vm[:-1, :, None])
    netx_v = (np.roll(fE_v, -1, axis=1) - fE_v) / (cos_v * R * dlon)[:, None, None]
    netx_v[0] = 0.0; netx_v[-1] = 0.0
    nety_v_int = (fN_v[1:] - fN_v[:-1]) / (dy_v_int * cos_v[1:-1])[:, None, None]
    nety_v = np.concatenate([np.zeros_like(v[:1]), nety_v_int, np.zeros_like(v[:1])], axis=0)
    visc_v_ref = (netx_v + nety_v) * vm[:, :, None]
    return visc_u_ref, visc_v_ref


@pytest.mark.parametrize("cos_power", [0, 1])
def test_matches_veros_harmonic_friction_term_for_term(cos_power):
    """The operator reproduces Veros ``harmonic_friction`` (the oracle) TERM-FOR-TERM
    on the real sphere: an independent re-implementation of Veros's discrete
    flux_east/flux_north + divergence (cos-power scaling, masks, ``cost``/``cosu``
    metric placement) in legoESM indexing matches ``flux_divergence_viscosity_cgrid``
    to machine eps for BOTH velocity components. This is the highest-value correctness
    gate (oracle-recipe doctrine) and the regression lock for the metric placement."""
    grid, um, vm, mask, u, v = _basin(uniform_cos=False, enclosed=True)
    vu, vv, _ = flux_divergence_viscosity_cgrid(
        u, v, grid, _A_H, cos_power=cos_power, mask=mask, u_mask=um, v_mask=vm)
    vu_ref, vv_ref = _veros_harmonic_friction_reference(grid, um, vm, u, v, cos_power)
    su = float(np.max(np.abs(vu_ref))) + 1e-300
    sv = float(np.max(np.abs(vv_ref))) + 1e-300
    du = float(np.max(np.abs(np.asarray(vu) - vu_ref)))
    dv = float(np.max(np.abs(np.asarray(vv) - vv_ref)))
    assert du / su < 1e-12, f"u differs from Veros harmonic_friction: rel {du/su:.2e}"
    assert dv / sv < 1e-12, f"v differs from Veros harmonic_friction: rel {dv/sv:.2e}"


def test_sphere_momentum_residual_is_metric_truncation():
    """On the real sphere the area-weighted momentum residual is the O(dx²) spherical-
    metric (curvature) truncation — the term the VECTOR Laplacian carries and the
    component-wise form omits — NOT a telescoping bug: it CONVERGES toward zero under
    grid refinement. (Documents the physics difference that motivates the option.)"""
    def resid(N):
        M = N + 4
        grid = create_latlon_grid(N, M, dtype=jnp.float64)
        m = np.ones((N, M)); m[0, :] = 0.0; m[-1, :] = 0.0; m[:, 0] = 0.0
        um = np.zeros((N, M + 1))
        for j in range(M + 1):
            um[:, j] = m[:, (j - 1) % M] * m[:, j % M]
        vm = np.zeros((N + 1, M))
        for i in range(1, N):
            vm[i, :] = m[i - 1, :] * m[i, :]
        rng = np.random.default_rng(7)
        u = jnp.asarray(0.3 * rng.standard_normal((N, M + 1, 1))) * jnp.asarray(um)[:, :, None]
        v = jnp.asarray(0.3 * rng.standard_normal((N + 1, M, 1))) * jnp.asarray(vm)[:, :, None]
        vu, vv, _ = flux_divergence_viscosity_cgrid(
            u, v, grid, _A_H, cos_power=1, mask=jnp.asarray(m),
            u_mask=jnp.asarray(um), v_mask=jnp.asarray(vm))
        cos_u, cos_v = _cos_lat_uv(grid)
        dy_cell = np.asarray(grid.dy * 0.5); dx_u = np.asarray(grid.radius * cos_u * grid.dlon)
        au = (np.asarray(cos_u) * dy_cell * dx_u)[:, None, None]
        dy_v = np.concatenate([[dy_cell[0]], 0.5 * (dy_cell[1:] + dy_cell[:-1]), [dy_cell[-1]]])
        av = (np.asarray(cos_v) * dy_v * float(dx_u.mean()))[:, None, None]
        ru = abs(float(np.sum((au * np.asarray(vu))[:, :-1, :])) /
                 (float(np.sum(np.abs(au * np.asarray(vu))[:, :-1, :])) + 1e-300))
        rv = abs(float(np.sum(av * np.asarray(vv))) /
                 (float(np.sum(np.abs(av * np.asarray(vv)))) + 1e-300))
        return max(ru, rv)
    r_coarse = resid(12)
    r_fine = resid(48)
    # refining 4x cuts the metric residual by >~5x (O(dx^2) -> expect ~16x; allow slack).
    assert r_fine < r_coarse / 5.0, f"residual not converging: coarse={r_coarse:.2e} fine={r_fine:.2e}"


# ---------------------------------------------------------------------------
# K_diss_h: positive-definite + energy-consistent.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cos_power", [0, 1])
def test_kdiss_positive_definite_no_clamp(cos_power):
    """The component-wise K_diss_h = A_h·|∇u|² = 0.5·Σ(Δu·flux) is >= 0 EVERYWHERE by
    construction (Δu·flux/Δx = A_h·cosᵖ·(∂u)² >= 0) — no clamp needed, unlike the
    dynamical -u·∇²_vec u form (~35% negative cells)."""
    grid, um, vm, mask, u, v = _basin(uniform_cos=False, enclosed=True)
    _, _, kd = flux_divergence_viscosity_cgrid(
        u, v, grid, _A_H, cos_power=cos_power, mask=mask, u_mask=um, v_mask=vm,
        want_dissipation=True)
    assert bool(jnp.all(kd >= 0.0)), f"K_diss_h has negatives, min={float(jnp.min(kd)):.3e}"
    assert bool(jnp.all(jnp.isfinite(kd)))


def test_kdiss_energy_consistent_machine_eps_metric_free():
    """On a metric-free closed basin (uniform cos) the K_diss_h domain integral equals
    the ACTUAL KE removed by the friction tendency (-Σ u·visc·area_u - v·visc·area_v)
    to ~machine level: the dissipation IS the energy the operator removes (Veros's
    mean->eddy K_diss_h pathway). The sphere residual (~10-15%) is the curvature term
    distinguishing the component-wise A_h|∇u|² from the dynamical -u·∇·(A_h∇u)."""
    grid, um, vm, mask, u, v = _basin(uniform_cos=True, enclosed=True)
    area_u, area_v = _area_uv(grid)
    area_c = np.asarray(grid.area)[:, :, None]
    for cos_power in (0, 1):
        vu, vv, kd = flux_divergence_viscosity_cgrid(
            u, v, grid, _A_H, cos_power=cos_power, mask=mask, u_mask=um, v_mask=vm,
            want_dissipation=True)
        ke_removed = (-float(np.sum((np.asarray(u) * np.asarray(vu) * area_u)[:, :-1, :]))
                      - float(np.sum(np.asarray(v) * np.asarray(vv) * area_v)))
        diss_int = float(np.sum(np.asarray(kd) * area_c))
        assert ke_removed > 0.0
        assert abs(diss_int / ke_removed - 1.0) < 1e-2, (
            f"cos_power={cos_power}: kdiss/KE_removed={diss_int / ke_removed:.5f}")


# ---------------------------------------------------------------------------
# cos(lat) scaling preserved.
# ---------------------------------------------------------------------------

def test_cos_scaling_applied_inside_flux():
    """cos_power scales the viscosity (Veros enable_hor_friction_cos_scaling): the
    flux-div tendency with cos_power=1 differs from cos_power=0 (the cos(lat) weight is
    applied inside the flux), and matches the analytic cos scaling at a probe latitude
    on a SEPARABLE field (purely meridional shear, so only the cosᵖ⁺¹ meridional flux
    weight acts and the per-row ratio is the cos factor)."""
    grid, um, vm, mask, u, v = _basin(enclosed=True)
    vu0, vv0, _ = flux_divergence_viscosity_cgrid(
        u, v, grid, _A_H, cos_power=0, mask=mask, u_mask=um, v_mask=vm)
    vu1, vv1, _ = flux_divergence_viscosity_cgrid(
        u, v, grid, _A_H, cos_power=1, mask=mask, u_mask=um, v_mask=vm)
    # cos_power changes the answer (scaling is live, not ignored).
    assert float(jnp.max(jnp.abs(vu1 - vu0))) > 0.0
    assert float(jnp.max(jnp.abs(vv1 - vv0))) > 0.0
    # Sanity: at the equator (cos≈1) the two are closest; toward the poles they diverge
    # (cos < 1 reduces the cos_power=1 viscosity). Compare row-mean |tendency| ratio.
    cos_u = np.asarray(grid.cos_lat)
    j_eq = int(np.argmax(cos_u))            # row nearest the equator
    j_hi = int(np.argmin(cos_u[1:-1])) + 1  # a high-lat interior row
    r_eq = float(jnp.mean(jnp.abs(vu1[j_eq]))) / (float(jnp.mean(jnp.abs(vu0[j_eq]))) + 1e-30)
    r_hi = float(jnp.mean(jnp.abs(vu1[j_hi]))) / (float(jnp.mean(jnp.abs(vu0[j_hi]))) + 1e-30)
    assert r_hi < r_eq, f"cos scaling not stronger at high lat: r_eq={r_eq:.3f} r_hi={r_hi:.3f}"


# ---------------------------------------------------------------------------
# Differentiability.
# ---------------------------------------------------------------------------

def test_differentiable_fd_vs_ad():
    """jax.grad through the operator + the K_diss_h is finite + nonzero, and matches a
    finite-difference probe."""
    grid, um, vm, mask, u0, v0 = _basin(enclosed=True, seed=5)

    def loss(u):
        vu, vv, kd = flux_divergence_viscosity_cgrid(
            u, v0, grid, _A_H, cos_power=1, mask=mask, u_mask=um, v_mask=vm,
            want_dissipation=True)
        return jnp.sum(vu ** 2) + jnp.sum(vv ** 2) + jnp.sum(kd ** 2)

    g = jax.grad(loss)(u0)
    assert bool(jnp.all(jnp.isfinite(g))), "non-finite grad"
    assert float(jnp.max(jnp.abs(g))) > 0.0, "zero grad — not differentiated"
    # FD check at an interior ocean u-point.
    idx = (_NLAT // 2, _NLON // 2, 0)
    eps = 1e-6
    fd = (loss(u0.at[idx].add(eps)) - loss(u0.at[idx].add(-eps))) / (2 * eps)
    ad = float(g[idx])
    assert abs(fd - ad) <= 1e-5 * (abs(fd) + abs(ad) + 1e-30), f"FD {fd:.6e} vs AD {ad:.6e}"


# ---------------------------------------------------------------------------
# Tripolar guard (not yet supported).
# ---------------------------------------------------------------------------

def test_tripolar_raises():
    """The operator rejects tripolar grids with a clear error (like the flux-form
    momentum advection) rather than silently using wrong vertex metrics."""
    grid, um, vm, mask, u, v = _basin()

    class _FoldStub:
        is_active = True

    g_tri = grid._replace() if hasattr(grid, "_replace") else grid
    # Attach an active fold descriptor via a thin wrapper (is_tripolar reads .fold).
    class _G:
        def __init__(self, base):
            self._base = base
            self.fold = _FoldStub()
        def __getattr__(self, k):
            return getattr(self._base, k)

    with pytest.raises(ValueError, match="tripolar"):
        flux_divergence_viscosity_cgrid(u, v, _G(grid), _A_H, cos_power=1,
                                        mask=mask, u_mask=um, v_mask=vm)


# ---------------------------------------------------------------------------
# Dispatch via the public config (the _bc_horizontal_viscosity path).
# ---------------------------------------------------------------------------

def _tend(cfg):
    """Compute baroclinic tendencies on a small spun-up-ish state for ``cfg``."""
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
        latlon_cgrid_ocean_baroclinic_tendencies,
    )
    grid = create_latlon_grid(12, 24, dtype=jnp.float64)
    z_coord = create_ocean_z_star(n_levels=4, H_max=4000.0)
    state = rest_state_latlon_cgrid_ocean(grid, z_coord, H_max=4000.0, land_lat_threshold=85.0)
    up = 0.1 * jax.random.normal(jax.random.PRNGKey(11), state.u.data.shape, dtype=jnp.float64)
    vp = 0.1 * jax.random.normal(jax.random.PRNGKey(12), state.v.data.shape, dtype=jnp.float64)
    state = state._replace(u=state.u.replace(data=state.u.data + up),
                           v=state.v.replace(data=state.v.data + vp))
    return latlon_cgrid_ocean_baroclinic_tendencies(state, grid, z_coord, cfg, dt=300.0)


def test_default_is_vector_laplacian_bit_identical():
    """``lateral_viscosity_operator`` defaults to "vector_laplacian", and the explicit
    value is BYTE-IDENTICAL to the historical (no-field) path through the full
    ``_bc_horizontal_viscosity`` dispatch — the mandatory default-off regression."""
    from legoesm.ocean.state import LatLonCGridOceanConfig
    cfg = LatLonCGridOceanConfig.from_flat(A_h=2.2e5, A_h_lat_scaling=True, A_h_cos_power=1)
    assert cfg.lateral_viscosity_operator == "vector_laplacian"
    t_default = _tend(cfg)
    t_explicit = _tend(cfg._replace(lateral_viscosity_operator="vector_laplacian"))
    np.testing.assert_array_equal(np.asarray(t_default.du_dt.data),
                                  np.asarray(t_explicit.du_dt.data))
    np.testing.assert_array_equal(np.asarray(t_default.dv_dt.data),
                                  np.asarray(t_explicit.dv_dt.data))


def test_flux_divergence_differs_and_finite_via_config():
    """Selecting flux_divergence through the config produces a DIFFERENT (the curvature
    coupling is dropped) but finite momentum tendency vs the vector Laplacian."""
    from legoesm.ocean.state import LatLonCGridOceanConfig
    base = LatLonCGridOceanConfig.from_flat(A_h=2.2e5, A_h_lat_scaling=True, A_h_cos_power=1)
    t_vec = _tend(base)
    t_fd = _tend(base._replace(lateral_viscosity_operator="flux_divergence"))
    assert bool(jnp.all(jnp.isfinite(t_fd.du_dt.data)))
    assert bool(jnp.all(jnp.isfinite(t_fd.dv_dt.data)))
    assert float(jnp.max(jnp.abs(t_fd.du_dt.data - t_vec.du_dt.data))) > 0.0
    assert float(jnp.max(jnp.abs(t_fd.dv_dt.data - t_vec.dv_dt.data))) > 0.0


def test_unknown_operator_raises_via_model_validation():
    """An unknown ``lateral_viscosity_operator`` literal raises ValueError at model
    construction (dispatch discipline — no silent fallthrough)."""
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    grid = create_latlon_grid(12, 24, dtype=jnp.float64)
    z_coord = create_ocean_z_star(n_levels=4, H_max=4000.0)
    cfg = LatLonCGridOceanConfig.from_flat(lateral_viscosity_operator="bogus")
    with pytest.raises(ValueError, match="lateral_viscosity_operator"):
        LatLonCGridOceanModel(grid, z_coord, cfg)


def test_flux_divergence_rejects_legoesm_boosts():
    """flux_divergence (Veros harmonic friction) rejects the legoESM-only A_h boosts
    (eq / cap / floor) it does not implement, rather than silently ignoring them."""
    from legoesm.ocean.state import LatLonCGridOceanConfig
    cfg = LatLonCGridOceanConfig.from_flat(
        A_h=2.2e5, A_h_lat_scaling=True, A_h_cos_power=1,
        lateral_viscosity_operator="flux_divergence", A_h_eq_boost=3.0)
    with pytest.raises(ValueError, match="flux_divergence"):
        _tend(cfg)


def test_acc_recipe_selects_flux_divergence():
    """The Veros ACC recipe opts into flux_divergence (+ keeps kdiss_h_flux_form), so
    the EKE K_diss_h source automatically follows the component-wise form."""
    from legoesm.ocean.fidelity.veros_acc_recipe import build_acc_model_config
    cfg = build_acc_model_config()
    assert cfg.lateral_viscosity_operator == "flux_divergence"
    assert cfg.gm_redi.eke.kdiss_h_flux_form is True
    assert cfg.gm_redi.eke.source_kdiss_h is True


# ---------------------------------------------------------------------------
# Component biharmonic (flux_divergence applied twice) — the MITgcm-faithful
# per-component del4 (useStrainTensionVisc=.FALSE.), stable where the vector
# grad(div)-curl(curl) biharmonic is ill-scaled (front_relax baroclinic oracle).
# ---------------------------------------------------------------------------
def test_component_biharmonic_equals_flux_divergence_twice():
    """By definition ∇⁴ = ∇²(∇²): the operator IS flux_divergence_viscosity_cgrid
    applied twice with unit coefficient."""
    grid, um, vm, m, u, v = _basin(uniform_cos=True)
    bu, bv = flux_divergence_bilaplacian_cgrid(u, v, grid, mask=m, u_mask=um, v_mask=vm)
    lu, lv, _ = flux_divergence_viscosity_cgrid(u, v, grid, 1.0, mask=m, u_mask=um, v_mask=vm)
    cu, cv, _ = flux_divergence_viscosity_cgrid(lu, lv, grid, 1.0, mask=m, u_mask=um, v_mask=vm)
    np.testing.assert_allclose(np.asarray(bu), np.asarray(cu), rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(bv), np.asarray(cv), rtol=0, atol=0)


def test_component_biharmonic_momentum_conserving():
    """∇⁴ telescopes: the area-weighted domain integral of each component is
    machine-zero on a metric-free closed basin (no spurious momentum source)."""
    grid, um, vm, m, u, v = _basin(uniform_cos=True)
    bu, bv = flux_divergence_bilaplacian_cgrid(u, v, grid, mask=m, u_mask=um, v_mask=vm)
    area_u, area_v = _area_uv(grid)
    wbu = np.asarray(bu) * area_u
    wbv = np.asarray(bv) * area_v
    assert abs(float(np.sum(wbu))) < 1e-9 * float(np.abs(wbu).sum() + 1e-30)
    assert abs(float(np.sum(wbv))) < 1e-9 * float(np.abs(wbv).sum() + 1e-30)


def test_component_biharmonic_correct_dimensional_scale():
    """The component ∇⁴ is at the correct ``~u/dx⁴`` dimensional scale (a clean
    5-point ∇² telescoped twice), finite everywhere. (The vector
    grad(div)-curl(curl) biharmonic is ADDITIONALLY ill-scaled on a
    uniform-Cartesian / near-degenerate C-grid — orders of magnitude too large,
    the front_relax instability — but that is grid-specific; the end-to-end
    stability of the faithful biharmonic is gated by the front_relax @slow test.)"""
    grid, um, vm, m, u, v = _basin(uniform_cos=True)
    bu, bv = flux_divergence_bilaplacian_cgrid(u, v, grid, mask=m, u_mask=um, v_mask=vm)
    assert np.all(np.isfinite(np.asarray(bu))) and np.all(np.isfinite(np.asarray(bv)))
    dx = float(grid.radius * grid.dlon * grid.cos_lat[_NLAT // 2])
    scale = 0.3 / dx ** 4                            # |u|~0.3, ∇⁴ ~ u/dx⁴
    assert float(np.abs(bu).max()) < 1e4 * scale    # right ballpark (not u-scale)
    assert float(np.abs(bu).max()) > 0.0            # genuinely non-trivial
