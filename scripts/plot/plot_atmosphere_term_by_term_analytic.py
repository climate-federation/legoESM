"""Run the term-by-term atmosphere SW tests + dump diagnostic PNGs.

Mirrors the scenarios in
``tests/atmosphere/shallow_water/unit/test_term_by_term_analytic.py``.
PNGs land in ``<repo-root>/results/atmosphere/term_by_term_analytic/``.

Run::

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu \\
        .venv/bin/python scripts/plot_atmosphere_term_by_term_analytic.py
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from legoesm import constants
from legoesm.atmosphere.dynamics.gcm.shallow_water_latlon_cgrid import (
    CGridLatLonShallowWaterConfig,
    CGridLatLonShallowWaterModel,
    CGridLatLonShallowWaterState,
    cgrid_latlon_sw_tendencies,
)
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.grids.latlon import create_regional_latlon_grid


set_policy(PrecisionPolicy.fp64())

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "results" / "atmosphere" / "term_by_term_analytic"
OUT_DIR.mkdir(parents=True, exist_ok=True)

H_REST = 100.0


# ----------------------------------------------------------------------
# Helpers (mirroring the test file)
# ----------------------------------------------------------------------


def _build_fplane_patch(*, n_lat_inner=16, n_lon=32, lat_half_deg=0.5,
                        lon_extent_deg=2.0, center_lat_deg=45.0):
    grid, _ = create_regional_latlon_grid(
        n_lat_inner, n_lon,
        lat_south=center_lat_deg - lat_half_deg,
        lat_north=center_lat_deg + lat_half_deg,
        lon_west=0.0, lon_east=lon_extent_deg,
        periodic_x=True, dtype=jnp.float64,
    )
    f0 = 2.0 * float(constants.Omega) * float(np.sin(np.radians(center_lat_deg)))
    return grid, f0


def _rest_state(grid, H=H_REST):
    return CGridLatLonShallowWaterState(
        h=jnp.full((grid.n_lat, grid.n_lon), H, dtype=jnp.float64),
        u=jnp.zeros((grid.n_lat, grid.n_lon + 1), dtype=jnp.float64),
        v=jnp.zeros((grid.n_lat + 1, grid.n_lon), dtype=jnp.float64),
        h_s=jnp.zeros((grid.n_lat, grid.n_lon), dtype=jnp.float64),
    )


def _zero_dynamics_config(*, A_h=0.0):
    return CGridLatLonShallowWaterConfig(
        A_h=A_h, time_integrator="ssp_rk3",
        fix_mass=False, anchor_mass_to_initial=False,
        use_ppm_transport=True, use_polar_filter=False,
    )


def _broadcast_1d_to_uface(state, u_1d):
    u_1d = np.asarray(u_1d)
    n_lon = u_1d.shape[-1]
    u_face = np.concatenate([u_1d, u_1d[:1]], axis=-1)
    n_lat = state.u.shape[0]
    return jnp.asarray(
        np.broadcast_to(u_face[None, :], (n_lat, n_lon + 1)).copy(),
        dtype=state.u.dtype,
    )


# ----------------------------------------------------------------------
# 1. Coriolis tendency
# ----------------------------------------------------------------------


def plot_coriolis_tendency():
    grid, f0 = _build_fplane_patch(center_lat_deg=45.0)
    config = _zero_dynamics_config()

    # Sweep U0 (with v=0): expect dv = -f0 * U0.
    Us = np.linspace(-0.2, 0.2, 9)
    dv_num = []
    for U0 in Us:
        state = _rest_state(grid)
        state = state._replace(
            u=jnp.full(state.u.shape, U0, dtype=state.u.dtype),
        )
        _dh, _du, dv = cgrid_latlon_sw_tendencies(state, grid, config)
        dv_num.append(float(jnp.mean(dv[1:-1, 1:-1])))
    dv_num = np.asarray(dv_num)
    dv_exact = -f0 * Us

    # Same with v=V0, u=0: expect du = +f0 * V0.
    Vs = np.linspace(-0.2, 0.2, 9)
    du_num = []
    for V0 in Vs:
        state = _rest_state(grid)
        state = state._replace(
            v=jnp.full(state.v.shape, V0, dtype=state.v.dtype),
        )
        _dh, du, _dv = cgrid_latlon_sw_tendencies(state, grid, config)
        du_num.append(float(jnp.mean(du[1:-1, 1:-1])))
    du_num = np.asarray(du_num)
    du_exact = +f0 * Vs

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    ax.plot(Us, dv_num, "C0o", label="num", ms=7)
    ax.plot(Us, dv_exact, "k--", label="dv/dt = -f U", lw=1.0)
    ax.set_xlabel("U0 [m/s]")
    ax.set_ylabel("dv/dt [m/s²]")
    ax.set_title(f"v-tendency vs uniform u (f = {f0:.3e} 1/s)")
    ax.grid(True, alpha=0.4)
    ax.legend()

    ax = axes[1]
    ax.plot(Vs, du_num, "C3o", label="num", ms=7)
    ax.plot(Vs, du_exact, "k--", label="du/dt = +f V", lw=1.0)
    ax.set_xlabel("V0 [m/s]")
    ax.set_ylabel("du/dt [m/s²]")
    ax.set_title("u-tendency vs uniform v")
    ax.grid(True, alpha=0.4)
    ax.legend()

    fig.suptitle("TestCoriolisTendencySW — vector-invariant Coriolis term", fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / "01_coriolis_tendency.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


# ----------------------------------------------------------------------
# 2. Burgers advection tendency
# ----------------------------------------------------------------------


def plot_burgers_tendency():
    grid, _f0 = _build_fplane_patch(
        center_lat_deg=0.0,
        n_lat_inner=4, n_lon=64,
        lon_extent_deg=2.0, lat_half_deg=0.05,
    )
    config = _zero_dynamics_config()

    lon_rad = np.asarray(grid.lon)
    radius = grid.radius
    dlon = float(grid.dlon)
    x_centres = radius * (lon_rad - 0.5 * dlon)
    Lx = radius * 2.0 * (np.pi / 180.0)
    k = 2.0 * np.pi / Lx
    U0 = 0.05

    state = _rest_state(grid)
    u_1d = U0 * np.sin(k * x_centres)
    state = state._replace(u=_broadcast_1d_to_uface(state, u_1d))

    _dh, du, _dv = cgrid_latlon_sw_tendencies(state, grid, config)
    du_num = np.asarray(du[1:-1, :-1]).mean(axis=0)
    du_exact = -U0 * U0 * k * np.sin(k * x_centres) * np.cos(k * x_centres)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    ax.plot(x_centres / 1e3, u_1d, "C0-", lw=1.6, label="u(x) = U0 sin(kx)")
    ax.set_xlabel("x [km]")
    ax.set_ylabel("u [m/s]")
    ax.set_title("Initial condition")
    ax.grid(True, alpha=0.4)
    ax.legend()

    ax = axes[1]
    ax.plot(x_centres / 1e3, du_num, "C0-", lw=1.8, label="du/dt num")
    ax.plot(x_centres / 1e3, du_exact, "k--", lw=1.2,
            label="du/dt = -u du/dx  (analytic)")
    ax.set_xlabel("x [km]")
    ax.set_ylabel("du/dt [m/s²]")
    ax.set_title(f"Burgers tendency (U0={U0}, k={k:.2e})")
    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper right", fontsize=8)

    fig.suptitle("TestBurgersAdvectionSW — vector-invariant advection tendency", fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / "02_burgers_tendency.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


# ----------------------------------------------------------------------
# 3. Laplacian viscous tendency
# ----------------------------------------------------------------------


def plot_viscous_tendency():
    n_lon = 64
    A_h = 5.0e2

    # Panel A: tendency shape at m=2.
    grid, _f0 = _build_fplane_patch(
        center_lat_deg=0.0,
        n_lat_inner=4, n_lon=n_lon,
        lon_extent_deg=2.0, lat_half_deg=0.05,
    )
    config_visc = _zero_dynamics_config(A_h=A_h)
    config_inv = _zero_dynamics_config(A_h=0.0)

    lon_rad = np.asarray(grid.lon)
    radius = grid.radius
    dlon = float(grid.dlon)
    x_centres = radius * (lon_rad - 0.5 * dlon)
    Lx = radius * 2.0 * (np.pi / 180.0)
    k = 4.0 * np.pi / Lx
    U0 = 1.0e-3

    state = _rest_state(grid)
    u_1d = U0 * np.cos(k * x_centres)
    state = state._replace(u=_broadcast_1d_to_uface(state, u_1d))

    _dh, du, _dv = cgrid_latlon_sw_tendencies(state, grid, config_visc)
    _dh0, du0, _dv0 = cgrid_latlon_sw_tendencies(state, grid, config_inv)
    du_visc = (np.asarray(du[1:-1, :-1]).mean(axis=0)
               - np.asarray(du0[1:-1, :-1]).mean(axis=0))
    du_visc_exact = -A_h * k * k * U0 * np.cos(k * x_centres)

    # Panel B: amplitude vs k² (scan over m).
    ms = [1, 2, 3, 4, 6, 8]
    amps_num, ks = [], []
    for m in ms:
        grid_m, _ = _build_fplane_patch(
            center_lat_deg=0.0,
            n_lat_inner=4, n_lon=n_lon,
            lon_extent_deg=2.0, lat_half_deg=0.05,
        )
        lon_rad_m = np.asarray(grid_m.lon)
        x_m = grid_m.radius * (lon_rad_m - 0.5 * float(grid_m.dlon))
        k_m = 2.0 * np.pi * m / Lx
        s_m = _rest_state(grid_m)
        s_m = s_m._replace(
            u=_broadcast_1d_to_uface(s_m, U0 * np.cos(k_m * x_m)))
        _h, du_v, _v = cgrid_latlon_sw_tendencies(s_m, grid_m, config_visc)
        _h0, du_0, _v0 = cgrid_latlon_sw_tendencies(s_m, grid_m, config_inv)
        amp = float(np.max(np.abs(
            np.asarray(du_v[1:-1, :-1]).mean(axis=0)
            - np.asarray(du_0[1:-1, :-1]).mean(axis=0)
        )))
        amps_num.append(amp)
        ks.append(k_m)
    amps_num = np.asarray(amps_num)
    ks = np.asarray(ks)
    amps_exact = A_h * ks ** 2 * U0

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    ax.plot(x_centres / 1e3, du_visc, "C0-", lw=1.8, label="du/dt|visc  num")
    ax.plot(x_centres / 1e3, du_visc_exact, "k--", lw=1.2,
            label="A_h · d²u/dx²  (analytic)")
    ax.set_xlabel("x [km]")
    ax.set_ylabel("du/dt [m/s²]")
    ax.set_title(f"Tendency shape (A_h={A_h}, k=4π/Lx, m=2)")
    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    ax.loglog(ks, amps_num, "C0o", label="num", ms=7)
    ax.loglog(ks, amps_exact, "k--", label="A_h · k² · U0", lw=1.0)
    ax.set_xlabel("k [1/m]")
    ax.set_ylabel("|du/dt|_∞ [m/s²]")
    ax.set_title("k² scaling of Laplacian tendency")
    ax.grid(True, which="both", alpha=0.4)
    ax.legend()

    fig.suptitle("TestLaplacianViscousTendencySW — A_h Laplacian only", fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / "03_viscous_tendency.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


# ----------------------------------------------------------------------
# 4. Rayleigh damping — documented absence
# ----------------------------------------------------------------------


def plot_rayleigh_skip():
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.axis("off")
    ax.text(
        0.5, 0.5,
        "Linear (Rayleigh) drag NOT implemented\n"
        "in the atmosphere shallow-water solver.\n\n"
        "CGridLatLonShallowWaterConfig has no drag\n"
        "coefficient field; cubed-sphere FV3, MPAS,\n"
        "and spectral configs likewise.\n\n"
        "Test surfaced as documented pytest.skip\n"
        "so the gap is visible in the matrix.",
        ha="center", va="center", fontsize=11,
        family="monospace",
        bbox=dict(facecolor="lightyellow", edgecolor="0.4", pad=10),
    )
    ax.set_title("TestRayleighDampingSW — not implemented")
    fig.tight_layout()
    out = OUT_DIR / "04_rayleigh_damping_not_implemented.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


CONV_OUT_DIR = REPO_ROOT / "results" / "atmosphere" / "convergence"
CONV_OUT_DIR.mkdir(parents=True, exist_ok=True)

CONV_N_LONS = (32, 64, 128, 256)


def _l2(a):
    return float(np.sqrt(np.mean(a ** 2)))


def _run_wave_sweep_sw():
    H = 100.0
    c = float(np.sqrt(constants.g * H))
    lon_extent_deg = 4.0
    # CFL = 0.1 + lat_half = 0.2°: matches the test setup that
    # recovers clean 2nd-order convergence across all four levels.
    cfl_target = 0.1
    radius = float(constants.R_earth)
    Lx = radius * lon_extent_deg * (np.pi / 180.0)
    T_end = 0.1 * Lx / c
    sigma = Lx / 12.0; x0 = Lx / 2.0
    # ε/H = 5e-5 keeps O(ε²/H) nonlinearity below the numerical
    # floor across the sweep — matches the matching test setup.
    eps = 5.0e-5 * H

    dxs, errs = [], []
    for n_lon in CONV_N_LONS:
        n_lat_inner = max(4, n_lon // 8)
        grid, _f0 = _build_fplane_patch(
            center_lat_deg=0.0,
            n_lat_inner=n_lat_inner, n_lon=n_lon,
            lon_extent_deg=lon_extent_deg, lat_half_deg=0.2,
        )
        config = _zero_dynamics_config()
        model = CGridLatLonShallowWaterModel(grid, config)

        dx = float(grid.dx[grid.n_lat // 2, grid.n_lon // 2]) / 2.0
        dt = cfl_target * dx / c
        n_steps = max(1, int(np.ceil(T_end / dt)))
        dt = T_end / n_steps

        lon_rad = np.asarray(grid.lon); dlon = float(grid.dlon)
        x_centres = radius * (lon_rad - 0.5 * dlon)
        x_centres = x_centres - x_centres[0]
        dh_p = eps * np.exp(-((x_centres - x0) / sigma) ** 2)
        h_1d = H + dh_p

        state = _rest_state(grid, H=H)
        h_2d = jnp.asarray(np.broadcast_to(h_1d[None, :], state.h.shape).copy())
        state = state._replace(h=h_2d)

        for _ in range(n_steps):
            state = model.step(state, dt)
        dh_end = np.asarray(state.h[1:-1, :]).mean(axis=0) - H

        def _gp(x):
            d = (x - x0 + Lx / 2.0) % Lx - Lx / 2.0
            return np.exp(-(d / sigma) ** 2)

        dh_exact = 0.5 * eps * (_gp(x_centres - c * T_end)
                                 + _gp(x_centres + c * T_end))
        dxs.append(dx); errs.append(_l2(dh_end - dh_exact))
    return np.array(dxs), np.array(errs)


def plot_convergence_sw():
    dxs_wave, errs_wave = _run_wave_sweep_sw()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.loglog(dxs_wave / 1e3, errs_wave, "C2o-", label="atmos SW")
    ref2 = errs_wave[-2] * (dxs_wave / dxs_wave[-2]) ** 2
    ax.loglog(dxs_wave / 1e3, ref2, "k--", lw=0.8, label="O(dx²)")
    ax.set_xlabel("dx [km]")
    ax.set_ylabel("L2 error in δh(x, T_end)")
    ax.set_title("Gravity wave: C_g = c dt/dx = const")
    ax.grid(True, which="both", alpha=0.4)
    ax.legend()

    ax = axes[1]
    ax.axis("off")
    ax.text(
        0.5, 0.5,
        "Transport convergence: documented gap\n"
        "  (atmosphere SW has no passive tracer;\n"
        "   momentum-Gaussian setup excites a\n"
        "   spurious gravity wave via continuity).\n\n"
        "Diffusion convergence: documented gap\n"
        "  (gravity-wave CFL is binding so dt ∝ dx,\n"
        "   not dx² → parabolic CFL drifts).",
        ha="center", va="center", fontsize=10, family="monospace",
        bbox=dict(facecolor="lightyellow", edgecolor="0.4", pad=10),
    )
    ax.set_title("Transport + Diffusion — documented gaps")

    fig.suptitle(
        "Atmosphere SW CFL-aware convergence — wave covered, transport / "
        "diffusion deferred",
        fontsize=11,
    )
    fig.tight_layout()
    out = CONV_OUT_DIR / "convergence_atmosphere.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    plot_coriolis_tendency()
    plot_burgers_tendency()
    plot_viscous_tendency()
    plot_rayleigh_skip()
    plot_convergence_sw()
    print(f"\nAll PNGs written to {OUT_DIR} and {CONV_OUT_DIR}")


if __name__ == "__main__":
    main()
