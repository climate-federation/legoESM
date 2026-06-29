"""Run the term-by-term analytic ocean tests + dump diagnostic PNGs.

Mirrors the scenarios in ``tests/ocean/unit/test_term_by_term_analytic.py``
and writes one PNG per regime to
``<repo-root>/results/ocean/term_by_term_analytic/``.

Run::

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu \\
        .venv/bin/python scripts/plot_term_by_term_analytic.py
"""

from __future__ import annotations

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from legoesm import constants
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.grids.latlon import create_regional_latlon_grid
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
    LatLonCGridOceanModel,
)
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.vertical import create_ocean_z_star


set_policy(PrecisionPolicy.fp64())

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "results" / "ocean" / "term_by_term_analytic"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# Helpers (mirroring the test file)
# ----------------------------------------------------------------------


def _build_fplane_patch(*, n_lat_inner=16, n_lon=32, lat_half_deg=0.5,
                       lon_extent_deg=2.0, center_lat_deg=45.0,
                       H_max=1000.0, n_levels=2):
    grid, _ = create_regional_latlon_grid(
        n_lat_inner, n_lon,
        lat_south=center_lat_deg - lat_half_deg,
        lat_north=center_lat_deg + lat_half_deg,
        lon_west=0.0, lon_east=lon_extent_deg,
        periodic_x=True, dtype=jnp.float64,
    )
    z_coord = create_ocean_z_star(
        n_levels=n_levels, H_max=H_max,
        dz_surface=H_max / n_levels, dz_deep=H_max / n_levels,
    )
    f0 = 2.0 * float(constants.Omega) * float(np.sin(np.radians(center_lat_deg)))
    return grid, z_coord, f0


def _all_wet_state(grid, z_coord, H_max=1000.0):
    land_mask = jnp.ones((grid.n_lat, grid.n_lon), dtype=jnp.float64)
    H_bathy = jnp.full((grid.n_lat, grid.n_lon), H_max, dtype=jnp.float64)
    return rest_state_latlon_cgrid_ocean(
        grid, z_coord,
        T_water_init_C=10.0, T_deep=10.0, S_uniform=35.0,
        H_max=H_max,
        land_mask_override=land_mask, H_bathy_override=H_bathy,
    )


def _zero_dynamics_config(*, A_h=0.0, B_h=0.0, bottom_drag_r=0.0,
                          momentum_advection="vector_invariant",
                          tracer_advection="upwind",
                          tracer_time_integrator="euler",
                          barotropic_solver="explicit_substep",
                          n_barotropic_substeps=10):
    return LatLonCGridOceanConfig.from_flat(
        A_h=A_h, A_h_lat_scaling=False, A_h_eq_boost=1.0, A_h_merid=0.0,
        B_h=B_h, B_h_barotropic=0.0,
        C_smag=0.0, C_smag_lap=0.0, C_leith=0.0, slope_foot_alpha=0.0,
        A_v=0.0, K_v=0.0, K_h=0.0, K_bih=0.0,
        implicit_vertical_mixing=False,
        bottom_drag_r=bottom_drag_r,
        bottom_drag_bbl_thickness=0.0, bottom_drag_bg_velocity=0.0,
        n_barotropic_substeps=n_barotropic_substeps,
        barotropic_diffusion_alpha=0.0, barotropic_div_damp=0.0,
        bebt=0.0, maxvel_barotropic=0.0,
        barotropic_time_filter="box",
        barotropic_solver=barotropic_solver,
        use_conservation_fixer=False, tracer_advection=tracer_advection,
        tracer_time_integrator=tracer_time_integrator,
        pgf_scheme="adcroft", momentum_advection=momentum_advection,
        ke_gradient_scheme="centered", weno_d_term=False,
        physics=None, gm_redi=None, eos="linear",
    )


def _set_uv_levels(state, u_per_level, v_per_level):
    u_arr = jnp.stack(list(u_per_level), axis=-1).astype(state.u.data.dtype)
    v_arr = jnp.stack(list(v_per_level), axis=-1).astype(state.v.data.dtype)
    return state._replace(
        u=state.u.replace(data=u_arr),
        v=state.v.replace(data=v_arr),
    )


def _broadcast_1d_to_uface(state, u_1d):
    u_1d = np.asarray(u_1d)
    n_lon = u_1d.shape[-1]
    u_face = np.concatenate([u_1d, u_1d[:1]], axis=-1)
    n_lat = state.u.data.shape[0]
    return jnp.asarray(
        np.broadcast_to(u_face[None, :], (n_lat, n_lon + 1)).copy(),
        dtype=state.u.data.dtype,
    )


def _step_many(model, state, dt, n_steps):
    for _ in range(n_steps):
        state = model.step(state, dt)
    return state


# ----------------------------------------------------------------------
# 1. Inertial oscillation
# ----------------------------------------------------------------------


def plot_inertial():
    grid, z_coord, f0 = _build_fplane_patch(center_lat_deg=45.0, n_levels=2)
    state = _all_wet_state(grid, z_coord, H_max=z_coord.H_max)
    config = _zero_dynamics_config()
    model = LatLonCGridOceanModel(grid, z_coord=z_coord, config=config)

    # U0 = 0.01 m/s rather than 0.1: the spherical-metric centripetal
    # term  dv/dt |_metric = tan(lat) · u² / R_earth  is exact in the
    # PE solver on a lat-lon grid.  It vanishes for the planar
    # f-plane analytic solution we compare against, so its strength
    # scales as U0² in the late-time amplitude drift.  Dropping U0
    # by 10× cuts the metric forcing 100×, exposing the underlying
    # Matsuno conservation to plotting precision over two inertial
    # periods.
    U0 = 0.01
    shape2d_u = state.u.data.shape[:2]
    shape2d_v = state.v.data.shape[:2]
    state = _set_uv_levels(
        state,
        [jnp.full(shape2d_u, +U0, dtype=state.u.data.dtype),
         jnp.full(shape2d_u, -U0, dtype=state.u.data.dtype)],
        [jnp.zeros(shape2d_v, dtype=state.v.data.dtype),
         jnp.zeros(shape2d_v, dtype=state.v.data.dtype)],
    )

    T_period = 2.0 * np.pi / f0
    n_per_period = 200
    n_periods = 2
    dt = T_period / n_per_period
    times, u_num, v_num = [], [], []
    s = state
    for k in range(n_periods * n_per_period + 1):
        u_num.append(float(jnp.mean(s.u.data[1:-1, 1:-1, 0])))
        v_num.append(float(jnp.mean(s.v.data[1:-1, 1:-1, 0])))
        times.append(k * dt)
        if k < n_periods * n_per_period:
            s = model.step(s, dt)
    times = np.asarray(times)
    u_num = np.asarray(u_num)
    v_num = np.asarray(v_num)
    u_exact = U0 * np.cos(f0 * times)
    v_exact = -U0 * np.sin(f0 * times)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    ax.plot(times / 3600.0, u_num, "C0-", label="u  num", lw=1.8)
    ax.plot(times / 3600.0, v_num, "C3-", label="v  num", lw=1.8)
    ax.plot(times / 3600.0, u_exact, "k--", label="u  analytic", lw=1.0)
    ax.plot(times / 3600.0, v_exact, "k:", label="v  analytic", lw=1.0)
    ax.set_xlabel("time [h]")
    ax.set_ylabel("velocity [m/s]")
    ax.set_title(f"Inertial oscillation (f = {f0:.3e} 1/s, T = {T_period/3600:.2f} h)")
    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper right", fontsize=9)

    ax = axes[1]
    ax.plot(u_num, v_num, "C0-", label="num", lw=1.8)
    ax.plot(u_exact, v_exact, "k--", label="analytic", lw=1.0)
    ax.set_aspect("equal")
    ax.set_xlabel("u [m/s]")
    ax.set_ylabel("v [m/s]")
    ax.set_title("Hodograph (top level)")
    ax.grid(True, alpha=0.4)
    ax.legend()
    metric = U0 * U0 * np.tan(np.radians(45.0)) / constants.R_earth
    ax.text(
        0.02, 0.02,
        f"U0 = {U0}.  Spherical-metric forcing\n"
        f"|dv/dt| ≈ tan(lat)·U0²/R = {metric:.2e} m/s² → barely visible drift",
        transform=ax.transAxes, fontsize=7, va="bottom",
        family="monospace",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.6", pad=2),
    )

    fig.suptitle("TestInertialOscillation — Coriolis only", fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / "01_inertial_oscillation.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


# ----------------------------------------------------------------------
# 2. Damped inertial oscillation
# ----------------------------------------------------------------------


def plot_damped_inertial():
    H_max = 1000.0
    r = 5.0e-3
    grid, z_coord, f0 = _build_fplane_patch(center_lat_deg=45.0, n_levels=2,
                                             H_max=H_max)
    state = _all_wet_state(grid, z_coord, H_max=H_max)
    config = _zero_dynamics_config(bottom_drag_r=r, n_barotropic_substeps=20)
    model = LatLonCGridOceanModel(grid, z_coord=z_coord, config=config)

    # U0 small — same spherical-metric argument as ``plot_inertial``.
    U0 = 0.01
    shape2d_u = state.u.data.shape[:2]
    shape2d_v = state.v.data.shape[:2]
    state = _set_uv_levels(
        state,
        [jnp.full(shape2d_u, +U0, dtype=state.u.data.dtype),
         jnp.full(shape2d_u, -U0, dtype=state.u.data.dtype)],
        [jnp.zeros(shape2d_v, dtype=state.v.data.dtype),
         jnp.zeros(shape2d_v, dtype=state.v.data.dtype)],
    )

    T_period = 2.0 * np.pi / f0
    n_per_period = 200
    n_periods = 2
    dt = T_period / n_per_period

    times, u_top, v_top, u_bot, v_bot, ke = [], [], [], [], [], []
    s = state
    for k in range(n_periods * n_per_period + 1):
        u = s.u.data[1:-1, 1:-1, :]
        v = s.v.data[1:-1, 1:-1, :]
        u_top.append(float(jnp.mean(u[..., 0])))
        v_top.append(float(jnp.mean(v[..., 0])))
        u_bot.append(float(jnp.mean(u[..., 1])))
        v_bot.append(float(jnp.mean(v[..., 1])))
        ke.append(float(jnp.mean(u ** 2) + jnp.mean(v ** 2)))
        times.append(k * dt)
        if k < n_periods * n_per_period:
            s = model.step(s, dt)
    times = np.asarray(times)
    ke = np.asarray(ke)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    ax.plot(times / 3600.0, u_top, "C0-", label="u top")
    ax.plot(times / 3600.0, v_top, "C0--", label="v top")
    ax.plot(times / 3600.0, u_bot, "C3-", label="u bot")
    ax.plot(times / 3600.0, v_bot, "C3--", label="v bot")
    ax.set_xlabel("time [h]")
    ax.set_ylabel("velocity [m/s]")
    ax.set_title("Per-level velocity components")
    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    ax.semilogy(times / 3600.0, ke, "k-", label="column KE  num")
    ax.semilogy(times / 3600.0, ke[0] * np.exp(-2.0 * r / H_max * times),
                "C2--", label=f"exp(-2 r / H · t),  r={r}, H={H_max}")
    ax.set_xlabel("time [h]")
    ax.set_ylabel("column KE [m²/s²]")
    ax.set_title("Kinetic energy decay")
    ax.grid(True, which="both", alpha=0.4)
    ax.legend(loc="upper right", fontsize=8)

    fig.suptitle("TestDampedInertialOscillation — Coriolis + linear drag", fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / "02_damped_inertial.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


# ----------------------------------------------------------------------
# 3. Burgers advection — tendency check
# ----------------------------------------------------------------------


def plot_burgers():
    H_max = 1000.0
    n_lat_inner = 4
    n_lon = 64
    grid, z_coord, _ = _build_fplane_patch(
        center_lat_deg=0.0, H_max=H_max, n_levels=1,
        n_lat_inner=n_lat_inner, n_lon=n_lon,
        lon_extent_deg=2.0, lat_half_deg=0.05,
    )
    state = _all_wet_state(grid, z_coord, H_max=H_max)
    config = _zero_dynamics_config()
    model = LatLonCGridOceanModel(grid, z_coord=z_coord, config=config)

    lon_rad = np.asarray(grid.lon)
    radius = grid.radius
    dlon = float(grid.dlon)
    x_centres = radius * (lon_rad - 0.5 * dlon)
    Lx = radius * 2.0 * (np.pi / 180.0)
    k = 2.0 * np.pi / Lx
    U0 = 0.05

    u_pattern = U0 * np.sin(k * x_centres)
    u_2d = _broadcast_1d_to_uface(state, u_pattern)
    v_2d = jnp.zeros(state.v.data.shape[:2], dtype=state.v.data.dtype)
    state = _set_uv_levels(state, [u_2d], [v_2d])

    tend = model.tendencies(state, surface_forcing=None, sponge=None, dt=1.0)
    du_dt_num = np.asarray(tend.du_dt.data[1:-1, :-1, 0]).mean(axis=0)
    du_dt_exact = -U0 * U0 * k * np.sin(k * x_centres) * np.cos(k * x_centres)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    ax.plot(x_centres / 1e3, u_pattern, "C0-", lw=1.6, label="u(x, t=0) = U0 sin(kx)")
    ax.set_xlabel("x [km]")
    ax.set_ylabel("u [m/s]")
    ax.set_title("Initial condition")
    ax.grid(True, alpha=0.4)
    ax.legend()

    ax = axes[1]
    ax.plot(x_centres / 1e3, du_dt_num, "C0-", lw=1.8, label="du/dt  num")
    ax.plot(x_centres / 1e3, du_dt_exact, "k--", lw=1.2,
            label="du/dt = -u du/dx  (analytic)")
    ax.set_xlabel("x [km]")
    ax.set_ylabel("du/dt [m/s²]")
    ax.set_title(f"Burgers tendency check (U0={U0}, k={k:.2e})")
    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper right", fontsize=8)

    fig.suptitle("TestBurgersAdvection — vector-invariant advection tendency", fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / "03_burgers_tendency.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


# ----------------------------------------------------------------------
# 4. Laplacian viscous decay
# ----------------------------------------------------------------------


def plot_viscous_decay():
    H_max = 100.0
    n_lon = 32
    n_lat_inner = 4
    lon_extent_deg = 2.0
    A_h = 5.0e2

    grid, z_coord, _ = _build_fplane_patch(
        center_lat_deg=0.0, H_max=H_max, n_levels=2,
        n_lat_inner=n_lat_inner, n_lon=n_lon,
        lon_extent_deg=lon_extent_deg, lat_half_deg=0.05,
    )
    state = _all_wet_state(grid, z_coord, H_max=H_max)
    config = _zero_dynamics_config(A_h=A_h, n_barotropic_substeps=30)
    model = LatLonCGridOceanModel(grid, z_coord=z_coord, config=config)

    lon_rad = np.asarray(grid.lon)
    radius = grid.radius
    dlon = float(grid.dlon)
    x_centres = radius * (lon_rad - 0.5 * dlon)
    Lx = radius * lon_extent_deg * (np.pi / 180.0)
    k = 4.0 * np.pi / Lx                          # m = 2
    U0 = 1.0e-3

    u_pattern = U0 * np.cos(k * x_centres)
    u_top_2d = _broadcast_1d_to_uface(state, u_pattern)
    v_zero = jnp.zeros(state.v.data.shape[:2], dtype=state.v.data.dtype)
    state = _set_uv_levels(state, [u_top_2d, -u_top_2d], [v_zero, v_zero])

    decay_rate = A_h * k * k
    t_e = 1.0 / decay_rate
    n_steps = 400
    dt = t_e / n_steps

    sample_steps = [0, n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps]
    snapshots = {}
    amp_t, time_axis = [], []
    s = state
    for k_step in range(n_steps + 1):
        u_line = np.asarray(s.u.data[1:-1, :-1, 0]).mean(axis=0)
        proj = float(np.mean(u_line * np.cos(k * x_centres))) * 2.0
        amp_t.append(proj)
        time_axis.append(k_step * dt)
        if k_step in sample_steps:
            snapshots[k_step] = u_line.copy()
        if k_step < n_steps:
            s = model.step(s, dt)
    amp_t = np.asarray(amp_t)
    time_axis = np.asarray(time_axis)
    amp_exact = U0 * np.exp(-decay_rate * time_axis)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0.0, 0.95, len(sample_steps)))
    for c, k_step in zip(colors, sample_steps):
        u_exact = U0 * np.exp(-decay_rate * k_step * dt) * np.cos(k * x_centres)
        ax.plot(x_centres / 1e3, snapshots[k_step], "-", color=c,
                lw=1.8, label=f"t = {k_step*dt/3600:.1f} h  num")
        ax.plot(x_centres / 1e3, u_exact, "--", color=c, lw=0.9)
    ax.set_xlabel("x [km]")
    ax.set_ylabel("u (top level) [m/s]")
    ax.set_title(f"u(x, t) snapshots (A_h = {A_h}, k = {k:.2e})")
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=7, loc="upper right")

    ax = axes[1]
    ax.semilogy(time_axis / 3600.0, np.abs(amp_t), "C0-",
                lw=1.8, label="projection on cos(kx)  num")
    ax.semilogy(time_axis / 3600.0, amp_exact, "k--",
                lw=1.0, label=f"U0 · exp(-A_h k² t),  e-fold = {t_e/3600:.2f} h")
    ax.set_xlabel("time [h]")
    ax.set_ylabel("amplitude [m/s]")
    ax.set_title("Amplitude vs analytic decay")
    ax.grid(True, which="both", alpha=0.4)
    ax.legend(loc="upper right", fontsize=8)

    fig.suptitle("TestLaplacianViscousDecay — A_h Laplacian only", fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / "04_viscous_decay.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


CONV_OUT_DIR = REPO_ROOT / "results" / "ocean" / "convergence"
CONV_OUT_DIR.mkdir(parents=True, exist_ok=True)

CONV_N_LONS = (32, 64, 128, 256)


def _l2(a):
    return float(np.sqrt(np.mean(a ** 2)))


# ----------------------------------------------------------------------
# Convergence sweep: transport (tracer) — sweep dst3 + weno5
# ----------------------------------------------------------------------


def _run_transport_sweep(scheme):
    H_max = 10.0; U0 = 0.5
    lon_extent_deg = 4.0; cfl_target = 0.2
    radius = float(constants.R_earth)
    Lx = radius * lon_extent_deg * (np.pi / 180.0)
    T_end = 0.25 * Lx / U0
    sigma = Lx / 16.0; x0 = Lx / 4.0

    dxs, errs = [], []
    for n_lon in CONV_N_LONS:
        n_lat_inner = max(4, n_lon // 8)
        grid, z_coord, _f0 = _build_fplane_patch(
            center_lat_deg=0.0, H_max=H_max, n_levels=1,
            n_lat_inner=n_lat_inner, n_lon=n_lon,
            lon_extent_deg=lon_extent_deg, lat_half_deg=0.05,
        )
        state = _all_wet_state(grid, z_coord, H_max=H_max)

        dx = float(grid.dx[grid.n_lat // 2, grid.n_lon // 2]) / 2.0
        dt = cfl_target * dx / U0
        n_steps = max(1, int(np.ceil(T_end / dt)))
        dt = T_end / n_steps

        c_gw = float(np.sqrt(constants.g * H_max))
        n_subs = max(10, int(np.ceil(10.0 * c_gw * dt / dx)))
        config = _zero_dynamics_config(
            tracer_advection=scheme, tracer_time_integrator="rk3",
            n_barotropic_substeps=n_subs,
        )
        model = LatLonCGridOceanModel(grid, z_coord=z_coord, config=config)

        lon_rad = np.asarray(grid.lon); dlon = float(grid.dlon)
        x_centres = radius * (lon_rad - 0.5 * dlon)
        x_centres = x_centres - x_centres[0]
        T_profile = np.exp(-((x_centres - x0) / sigma) ** 2)
        T_2d = jnp.asarray(np.broadcast_to(
            T_profile[None, :, None], state.T.data.shape).copy())

        u0 = jnp.full(state.u.data.shape, U0)
        v0 = jnp.zeros_like(state.v.data)
        state = state._replace(
            u=state.u.replace(data=u0), v=state.v.replace(data=v0),
            T=state.T.replace(data=T_2d),
        )

        for _ in range(n_steps):
            state = model.step(state, dt)
        T_num = np.asarray(state.T.data[1:-1, :, 0]).mean(axis=0)
        x_shift = (x_centres - U0 * T_end) % Lx
        T_exact = np.exp(-(np.minimum(np.abs(x_shift - x0),
                                       Lx - np.abs(x_shift - x0)) / sigma) ** 2)
        dxs.append(dx); errs.append(_l2(T_num - T_exact))
    return np.array(dxs), np.array(errs)


def _run_wave_sweep():
    H_max = 100.0
    c = float(np.sqrt(constants.g * H_max))
    lon_extent_deg = 4.0; cfl_target = 0.25
    radius = float(constants.R_earth)
    Lx = radius * lon_extent_deg * (np.pi / 180.0)
    T_end = 0.25 * Lx / c
    sigma = Lx / 20.0; x0 = Lx / 2.0
    # Small ε so the O(ε²/H) nonlinear correction stays below the
    # numerical floor — matches the test setup in
    # `tests/ocean/unit/test_term_by_term_analytic.py`.
    eps = 1.0e-4 * H_max

    dxs, errs = [], []
    for n_lon in CONV_N_LONS:
        n_lat_inner = max(4, n_lon // 8)
        grid, z_coord, _f0 = _build_fplane_patch(
            center_lat_deg=0.0, H_max=H_max, n_levels=1,
            n_lat_inner=n_lat_inner, n_lon=n_lon,
            lon_extent_deg=lon_extent_deg, lat_half_deg=0.05,
        )
        state = _all_wet_state(grid, z_coord, H_max=H_max)
        dx = float(grid.dx[grid.n_lat // 2, grid.n_lon // 2]) / 2.0
        dt = cfl_target * dx / c
        n_steps = max(1, int(np.ceil(T_end / dt)))
        dt = T_end / n_steps

        config = _zero_dynamics_config(
            barotropic_solver="implicit_cn", n_barotropic_substeps=1,
        )
        config = config._replace(
            barotropic_implicit_pcg_tol=1.0e-13,
            barotropic_implicit_pcg_maxiter=400,
        )
        model = LatLonCGridOceanModel(grid, z_coord=z_coord, config=config)

        lon_rad = np.asarray(grid.lon); dlon = float(grid.dlon)
        x_centres = radius * (lon_rad - 0.5 * dlon)
        x_centres = x_centres - x_centres[0]
        eta_p = eps * np.exp(-((x_centres - x0) / sigma) ** 2)
        eta_2d = jnp.asarray(np.broadcast_to(
            eta_p[None, :], state.eta.data.shape).copy())
        state = state._replace(eta=state.eta.replace(data=eta_2d))

        for _ in range(n_steps):
            state = model.step(state, dt)
        eta_end = np.asarray(state.eta.data[1:-1, :]).mean(axis=0)

        def _gp(x):
            d = (x - x0 + Lx / 2.0) % Lx - Lx / 2.0
            return np.exp(-(d / sigma) ** 2)

        eta_exact = 0.5 * eps * (_gp(x_centres - c * T_end)
                                  + _gp(x_centres + c * T_end))
        dxs.append(dx); errs.append(_l2(eta_end - eta_exact))
    return np.array(dxs), np.array(errs)


def plot_convergence():
    dxs_dst3, errs_dst3 = _run_transport_sweep("dst3")
    dxs_weno5, errs_weno5 = _run_transport_sweep("weno5")
    dxs_wave, errs_wave = _run_wave_sweep()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.loglog(dxs_dst3 / 1e3, errs_dst3, "C0o-", label="dst3 (~O(dx))")
    ax.loglog(dxs_weno5 / 1e3, errs_weno5, "C3s-", label="weno5 (~O(dx²))")
    ref1 = errs_dst3[-1] * (dxs_dst3 / dxs_dst3[-1])
    ax.loglog(dxs_dst3 / 1e3, ref1, "C0:", lw=0.8, label="O(dx) ref")
    ref2 = errs_weno5[-1] * (dxs_weno5 / dxs_weno5[-1]) ** 2
    ax.loglog(dxs_weno5 / 1e3, ref2, "C3:", lw=0.8, label="O(dx²) ref")
    ax.set_xlabel("dx [km]")
    ax.set_ylabel("L2 error in T(x, T_end)")
    ax.set_title("Transport: C_a = U0 dt/dx = const  (rk3 tracer step)")
    ax.grid(True, which="both", alpha=0.4)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.loglog(dxs_wave / 1e3, errs_wave, "C2o-", label="implicit_cn")
    ref2 = errs_wave[-2] * (dxs_wave / dxs_wave[-2]) ** 2
    ax.loglog(dxs_wave / 1e3, ref2, "k--", lw=0.8, label="O(dx²)")
    ax.set_xlabel("dx [km]")
    ax.set_ylabel("L2 error in η(x, T_end)")
    ax.set_title("Gravity wave: C_g = c dt/dx = const")
    ax.grid(True, which="both", alpha=0.4)
    ax.legend()

    fig.suptitle(
        "Ocean CFL-aware convergence — diffusion deferred "
        "(F_slow → barotropic substep contaminates rate)",
        fontsize=11,
    )
    fig.tight_layout()
    out = CONV_OUT_DIR / "convergence_ocean.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    plot_inertial()
    plot_damped_inertial()
    plot_burgers()
    plot_viscous_decay()
    plot_convergence()
    print(f"\nAll PNGs written to {OUT_DIR} and {CONV_OUT_DIR}")


if __name__ == "__main__":
    main()
