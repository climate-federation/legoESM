"""Publication figures for the plane compressible-Euler (CRM) dycore
term-by-term analytic verification (companion to
``tests/unit/test_compressible_euler_plane_analytic.py``).

Markers = numerical slow-tendency, dashed = closed-form. One multi-panel PNG.

Run::

    JAX_ENABLE_X64=1 .venv/bin/python scripts/plot/plot_crm_dycore_analytic.py \\
        --out results/crm_dycore_analytic
"""
from __future__ import annotations

import argparse
from pathlib import Path

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from legoesm.atmosphere.dynamics.gcm.compressible_euler import CompressibleEulerConfig  # noqa: E402
from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (  # noqa: E402
    make_flat_plane_terrain_metric, make_rest_state,
    plane_compressible_euler_slow_tendencies as slow_tend,
)
from legoesm.grids.plane import create_plane_grid  # noqa: E402
from legoesm.grids.vertical import create_height_coordinate  # noqa: E402


def _cfg(**ov):
    base = dict(sponge_coeff=0.0, hyperdiff_coeff=0.0, hyperdiff_rho_coeff=0.0,
                hyperdiff_w_coeff=0.0, semi_implicit_acoustic=False,
                use_coriolis=False, moist_buoyancy=False, smagorinsky_cs=0.0,
                horizontal_advection_scheme="centered",
                horizontal_momentum_advection_scheme="centered",
                vertical_theta_diffusion=0.0)
    base.update(ov)
    return CompressibleEulerConfig(**base)


def _build(f0=0.0, **cfgkw):
    cmode = "f_plane" if f0 != 0.0 else "none"
    grid = create_plane_grid(nx=32, ny=32, nlev=8, dx=1.0e3, dy=1.0e3,
                             coriolis_mode=cmode, f0=f0, dtype=jnp.float64)
    hc = create_height_coordinate(grid.nlev, H=20.0e3)
    tm = make_flat_plane_terrain_metric(grid, hc)
    return grid, hc, tm, _cfg(**cfgkw), make_rest_state(grid, hc, dtype=jnp.float64)


def _set(st, **f):
    return st._replace(**{k: getattr(st, k).replace(data=v) for k, v in f.items()})


def coriolis_panel(ax):
    F = 1.0e-4
    grid, hc, tm, cfg, st = _build(f0=F, use_coriolis=True)
    sh = st.u.data.shape
    vals = np.linspace(-0.2, 0.2, 9)
    du = [float(np.mean(np.asarray(slow_tend(_set(st, u=jnp.zeros(sh),
          v=jnp.full(sh, V)), grid, hc, tm, cfg).du_dt.data))) for V in vals]
    dv = [float(np.mean(np.asarray(slow_tend(_set(st, u=jnp.full(sh, U),
          v=jnp.zeros(sh)), grid, hc, tm, cfg).dv_dt.data))) for U in vals]
    ax.plot(vals, du, "o", color="tab:blue", label="du/dt num")
    ax.plot(vals, F * vals, "--k", label="+f·V")
    ax.plot(vals, dv, "s", color="tab:red", label="dv/dt num")
    ax.plot(vals, -F * vals, ":k", label="−f·U")
    ax.set_xlabel("U₀ or V₀ [m/s]"); ax.set_ylabel("tendency [m/s²]")
    ax.set_title(f"Coriolis tendency (f={F:.2e} 1/s)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)


def biharm_panel(ax):
    K4, U0 = 1.0e8, 1.0
    grid, hc, tm, cfg, st = _build(hyperdiff_coeff=K4)
    ny, nx, nlev = st.u.data.shape
    yc = jnp.asarray(grid.yc, jnp.float64)
    ks, amps = [], []
    for m in (1, 2, 3, 4, 5):
        ky = 2.0 * np.pi * m / (ny * grid.dy)
        u = U0 * jnp.cos(ky * yc)[:, None, None] * jnp.ones((ny, nx, nlev))
        t = slow_tend(_set(st, u=u), grid, hc, tm, cfg)
        ks.append(ky); amps.append(float(np.max(np.abs(np.asarray(t.du_dt.data)))))
    ks, amps = np.array(ks), np.array(amps)
    ax.loglog(ks, amps, "o", color="tab:blue", label="num")
    ax.loglog(ks, K4 * ks ** 4 * U0, "--k", label="K₄·k⁴·U₀ (analytic)")
    ax.set_xlabel("k [1/m]"); ax.set_ylabel("|du/dt|∞ [m/s²]")
    ax.set_title(f"Biharmonic hyperdiffusion (K₄={K4:.0e})")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")


def advection_panel(ax):
    U, TH0, m = 5.0, 2.0, 2
    grid, hc, tm, cfg, st = _build()
    ny, nx, nlev = st.u.data.shape
    kx = 2.0 * np.pi * m / (nx * grid.dx)
    xc = np.asarray(grid.xc)
    theta_p = TH0 * jnp.sin(kx * jnp.asarray(grid.xc, jnp.float64))[None, :, None] \
        * jnp.ones((ny, nx, nlev))
    t = slow_tend(_set(st, u=jnp.full((ny, nx, nlev), U), theta_prime=theta_p),
                  grid, hc, tm, cfg)
    dthp = np.asarray(t.dtheta_prime_dt.data)[0, :, 0]
    fac = np.sin(kx * grid.dx) / (kx * grid.dx)
    ax.plot(xc / 1e3, dthp, "o", color="tab:blue", ms=4, label="dθ'/dt num")
    ax.plot(xc / 1e3, -U * TH0 * kx * fac * np.cos(kx * xc), "--k",
            label="−U ∂θ'/∂x (analytic)")
    ax.set_xlabel("x [km]"); ax.set_ylabel("dθ'/dt [K/s]")
    ax.set_title(f"Scalar advection (U={U} m/s, centred)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("results/crm_dycore_analytic"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    coriolis_panel(axes[0]); biharm_panel(axes[1]); advection_panel(axes[2])
    fig.suptitle("Plane compressible-Euler (CRM) dycore — term-by-term analytic "
                 "verification", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = args.out / "crm_dycore_analytic.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
