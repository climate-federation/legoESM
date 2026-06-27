"""Findings #6 (single-owner bottom drag) and #8 (continuity-consistent
barotropic transport weights) for the lat-lon C-grid explicit barotropic
substep solver ``barotropic_substeps_latlon_cgrid``.

#8 — the time-averaged transport ``Hu_avg`` returned by the solver must satisfy
the discrete continuity invariant the flux-form tracer step relies on,

    div(Hu_avg) == (eta_old - eta_avg) / dt,

for EVERY ``barotropic_time_filter`` (box / cosine / power_law).  This is what
preserves a uniform tracer.  The earlier flat ``1/n`` transport weight broke it
for box (~95% residual) and cosine (~99%); the SM2005 tail-sum weight fixes it.

#6 — the explicit barotropic substep must NOT re-apply ``implicit_bottom_drag_
factor`` on top of the depth-mean bottom drag already carried in ``F_slow``;
doing so doubled the effective barotropic-mode drag to ~2r/H.  The substep now
owns no drag, so a barotropic mode forced ONLY by ``F_slow_u = -r*U/H`` decays
at the single rate ``r/H``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
    barotropic_substeps_latlon_cgrid,
)
from legoesm.ocean.dynamics.latlon_cgrid_operators import divergence_cgrid


def _flat_basin(n_lat=24, n_lon=48, H=4000.0, lat_cap_deg=80.0):
    grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon)
    z = create_ocean_z_star(n_levels=4, H_max=H, dz_surface=50.0, dz_deep=1500.0)
    latd = np.asarray(grid.lat2d) * 180.0 / np.pi
    H_bathy = jnp.full((grid.n_lat, grid.n_lon), H)
    land_mask = jnp.asarray(np.where(np.abs(latd) < lat_cap_deg, 1.0, 0.0))
    state = rest_state_latlon_cgrid_ocean(
        grid, z, T_water_init_C=10.0, T_deep=10.0, S_uniform=35.0,
        H_max=H, H_bathy_override=H_bathy, land_mask_override=land_mask)
    return grid, z, state


def _cfg(**over):
    # #644 (config grouping) nested the barotropic + bottom-drag knobs into
    # BarotropicConfig / the bottom-drag sub-config. Route flat test kwargs to the
    # right leaf by NamedTuple field membership (robust to exact field placement).
    cfg = LatLonCGridOceanConfig(fix_eta_drift=False)
    baro = cfg.barotropic._replace(
        bebt=0.0, maxvel_barotropic=0.0, barotropic_diffusion_alpha=0.0,
        barotropic_local_subcycle_clamp=False,
        barotropic_solver="explicit_substep",
    )
    drag = cfg.bottom_drag._replace(bottom_drag_r=0.0)
    for k, v in over.items():
        if k in drag._fields:
            drag = drag._replace(**{k: v})
        elif k in baro._fields:
            baro = baro._replace(**{k: v})
        else:
            cfg = cfg._replace(**{k: v})
    return cfg._replace(barotropic=baro, bottom_drag=drag)


class TestBarotropicContinuityInvariant:
    """Finding #8: div(Hu_avg) == (eta_old - eta_avg)/dt for every filter."""

    @pytest.mark.parametrize("time_filter", ["box", "cosine", "power_law"])
    def test_continuity_invariant(self, time_filter):
        grid, z, state = _flat_basin()
        key = jax.random.PRNGKey(1)
        u0 = 0.05 * jax.random.normal(key, state.u.data.shape) * state.u_mask.data[..., None]
        v0 = 0.05 * jax.random.normal(
            jax.random.split(key)[0], state.v.data.shape) * state.v_mask.data[..., None]
        state = state._replace(u=state.u.replace(data=u0), v=state.v.replace(data=v0))
        dt = 1800.0
        n = 30
        cfg = _cfg(barotropic_time_filter=time_filter)
        eta_old = state.eta.data
        sn, (Hu, Hv) = barotropic_substeps_latlon_cgrid(
            state, dt / n, n, grid, z, cfg, add_barotropic_coriolis=True)
        eta_avg = sn.eta.data
        div = divergence_cgrid(
            Hu, Hv, grid, u_mask=state.u_mask.data, v_mask=state.v_mask.data)
        resid = np.asarray(div - (eta_old - eta_avg) / dt)
        m = np.asarray(state.land_mask.data) > 0
        max_resid = float(np.max(np.where(m, np.abs(resid), 0.0)))
        signal = float(np.max(np.abs(np.asarray((eta_old - eta_avg) / dt))))
        rel = max_resid / max(signal, 1e-30)
        assert rel < 1e-5, (
            f"[{time_filter}] continuity invariant residual rel={rel:.3e} "
            f"(abs {max_resid:.3e}); transport weights inconsistent with eta_avg "
            "(finding #8)")

    def test_uniform_tracer_preserved_cosine(self):
        """The invariant's PURPOSE: with continuity-consistent transport, the
        column-integrated thickness change ``eta_avg - eta_old`` equals
        ``-dt*div(Hu_avg)``, so a uniform tracer advected by the resulting mass
        fluxes is preserved.  Pin the column-sum identity directly (cosine)."""
        grid, z, state = _flat_basin()
        key = jax.random.PRNGKey(5)
        u0 = 0.04 * jax.random.normal(key, state.u.data.shape) * state.u_mask.data[..., None]
        v0 = 0.04 * jax.random.normal(
            jax.random.split(key)[0], state.v.data.shape) * state.v_mask.data[..., None]
        state = state._replace(u=state.u.replace(data=u0), v=state.v.replace(data=v0))
        dt = 1800.0
        n = 24
        cfg = _cfg(barotropic_time_filter="cosine")
        eta_old = state.eta.data
        sn, (Hu, Hv) = barotropic_substeps_latlon_cgrid(
            state, dt / n, n, grid, z, cfg, add_barotropic_coriolis=True)
        eta_avg = sn.eta.data
        div = divergence_cgrid(
            Hu, Hv, grid, u_mask=state.u_mask.data, v_mask=state.v_mask.data)
        # h_new - h_old (column sum) == eta_avg - eta_old; tracer continuity needs
        # (eta_avg - eta_old) + dt*div(Hu_avg) == 0.
        m = np.asarray(state.land_mask.data) > 0
        lhs = np.asarray((eta_avg - eta_old) + dt * div)
        err = float(np.max(np.where(m, np.abs(lhs), 0.0)))
        amp = float(np.max(np.where(m, np.abs(np.asarray(eta_avg - eta_old)), 0.0)))
        assert err < 1e-6 * max(amp, 1.0), (
            f"uniform-tracer continuity violated by {err:.3e} (amp {amp:.3e})")


class TestBottomDragSingleOwner:
    """Finding #6: effective barotropic-mode drag is r/H (not 2r/H).

    Force the barotropic mode with ONLY the depth-mean bottom drag
    ``F_slow_u = -r*U/H`` (no PGF, no Coriolis, flat resting eta) and check that
    ``U_bar`` decays per outer step by the SINGLE implicit factor
    ``1/(1 + r*dt/H)`` — i.e. the substep does NOT additionally damp it.
    """

    def test_effective_barotropic_drag_is_r_over_H(self):
        grid, z, state = _flat_basin()
        H = 4000.0
        r = 1.0e-3
        # Uniform barotropic zonal flow; zero eta.
        U0 = 0.2
        u0 = jnp.full(state.u.data.shape, U0) * state.u_mask.data[..., None]
        state = state._replace(u=state.u.replace(data=u0))
        dt = 1800.0
        n = 36
        cfg = _cfg(barotropic_time_filter="box", bottom_drag_r=r)
        # F_slow_u = depth-mean bottom drag = -r*U/H (the ONLY forcing).  No
        # Coriolis (add_barotropic_coriolis=False) and zero PGF (flat eta).
        u_mask = state.u_mask.data
        F_slow_u = (-r * U0 / H) * u_mask
        F_slow_v = jnp.zeros((grid.n_lat + 1, grid.n_lon))
        sn, _ = barotropic_substeps_latlon_cgrid(
            state, dt / n, n, grid, z, cfg,
            F_slow_u=F_slow_u, F_slow_v=F_slow_v,
            add_barotropic_coriolis=False)
        # The time-averaged U_bar_avg is returned through the 3D velocity; read
        # the mean ocean u.  With single-owner drag the barotropic forcing is the
        # explicit -r*U/H applied each substep -> U decays by ~ (1 - r*dt/H).
        m = np.asarray(u_mask) > 0
        u_new = np.asarray(sn.u.data)[..., 0]
        u_mean = float(np.mean(np.where(m, u_new, np.nan)[~np.isnan(np.where(m, u_new, np.nan))]))
        # Expected single-rate decay of the time-MEAN over the step.  Frozen
        # F_slow gives U(t)=U0 - (r U0/H) t, whose step-mean is U0(1 - r*dt/(2H)).
        expected_mean = U0 * (1.0 - r * dt / (2.0 * H))
        # The DOUBLE-counted (old) drag would also apply ~ (1 - r*dt/H) on top,
        # giving a clearly smaller value; assert we are within 5% of the
        # single-owner expectation and NOT near the double-rate value.
        single = abs(u_mean - expected_mean) / expected_mean
        double_mean = U0 * (1.0 - 3.0 * r * dt / (2.0 * H))  # ~2x drag proxy
        assert single < 0.02, (
            f"barotropic drag mean u={u_mean:.6f} vs single-owner "
            f"{expected_mean:.6f} (rel {single:.3e}); drag may be double-counted")
        assert abs(u_mean - expected_mean) < abs(u_mean - double_mean), (
            "barotropic drag closer to the DOUBLE-counted rate than the "
            "single-owner rate (finding #6 regression)")
