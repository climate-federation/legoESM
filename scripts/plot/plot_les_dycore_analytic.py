"""Publication figures for the spectral-LES dycore term-by-term analytic
verification (companion to ``tests/unit/test_spectral_les_analytic.py``).

Recomputes the same isolated-tendency / decay / convergence diagnostics the
tests assert on, and renders them num-vs-analytic in the house style of
``scripts/plot/plot_term_by_term_analytic.py`` (markers = numerical, dashed =
closed-form). One multi-panel PNG.

Run::

    JAX_ENABLE_X64=1 .venv/bin/python scripts/plot/plot_les_dycore_analytic.py \\
        --out results/les_dycore_analytic
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

from legoesm.atmosphere.dynamics.les import spectral_les_plane as sl  # noqa: E402


def _cfg(**kw):
    base = dict(nx=32, ny=32, nz=24, Lx=320.0, Ly=320.0, Lz=240.0,
                c_s=0.0, c_vreman=0.0, smagorinsky_dynamic=False, nu_floor=0.0,
                buoyancy=False, wall_damping=False, spectral_filter=False,
                dealias=True, time_scheme="rk3")
    base.update(kw)
    return sl.SpectralLESConfig(**base)


def _state(u, v, w):
    return sl.SpectralLESState(u=u, v=v, w=w, theta=None, rhs_theta_prev=None,
                               rhs_u_prev=jnp.zeros_like(u),
                               rhs_v_prev=jnp.zeros_like(v),
                               rhs_w_prev=jnp.zeros_like(w))


def coriolis_panel(ax):
    F = 1.0e-4
    g = sl.make_grid(_cfg())
    ny, nx, nz = g.cfg.ny, g.cfg.nx, g.cfg.nz
    Vs = np.linspace(-0.2, 0.2, 9)
    du, dv = [], []
    for V0 in Vs:
        u = jnp.full((ny, nx, nz), 0.0); v = jnp.full((ny, nx, nz), V0)
        w = jnp.zeros((ny, nx, nz + 1))
        Ru, _, _, _, _, _ = sl.rhs(u, v, w, g, u_geo=(0.0, 0.0), f_cor=F)
        du.append(float(np.mean(np.asarray(Ru)[..., 1:])))
    for U0 in Vs:
        u = jnp.full((ny, nx, nz), U0); v = jnp.full((ny, nx, nz), 0.0)
        w = jnp.zeros((ny, nx, nz + 1))
        _, Rv, _, _, _, _ = sl.rhs(u, v, w, g, u_geo=(0.0, 0.0), f_cor=F)
        dv.append(float(np.mean(np.asarray(Rv)[..., 1:])))
    ax.plot(Vs, du, "o", color="tab:blue", label="du/dt  num")
    ax.plot(Vs, F * Vs, "--", color="k", label="+f·V (analytic)")
    ax.plot(Vs, dv, "s", color="tab:red", label="dv/dt  num")
    ax.plot(Vs, -F * Vs, ":", color="k", label="−f·U (analytic)")
    ax.set_xlabel("U₀ or V₀ [m/s]"); ax.set_ylabel("tendency [m/s²]")
    ax.set_title(f"Coriolis tendency (f = {F:.2e} 1/s)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)


def viscous_k2_panel(ax):
    NU, U0 = 0.5, 0.3
    g = sl.make_grid(_cfg(nu_floor=NU))
    ny, nx, nz = g.cfg.ny, g.cfg.nx, g.cfg.nz
    y = (jnp.arange(ny) * g.dy).astype(jnp.float64)
    ks, amps = [], []
    for m in (1, 2, 3, 4, 5, 6):
        ky = 2.0 * np.pi * m / g.cfg.Ly
        u = U0 * jnp.sin(ky * y)[:, None, None] * jnp.ones((ny, nx, nz))
        Ru, _, _, _, _, _ = sl.rhs(u, jnp.zeros((ny, nx, nz)),
                                jnp.zeros((ny, nx, nz + 1)), g,
                                u_geo=(0.0, 0.0), f_cor=0.0)
        ks.append(ky); amps.append(float(np.max(np.abs(np.asarray(Ru)[..., nz // 2]))))
    ks, amps = np.array(ks), np.array(amps)
    ax.loglog(ks, amps, "o", color="tab:blue", label="num")
    ax.loglog(ks, NU * ks ** 2 * U0, "--", color="k", label="ν·k²·U₀ (analytic)")
    ax.set_xlabel("k [1/m]"); ax.set_ylabel("|du/dt|∞ [m/s²]")
    ax.set_title(f"Constant-ν SGS Laplacian (ν={NU})")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")


def decay_panel(ax):
    NU, U0, m = 8.0, 0.2, 4     # rate·T≈0.5 ⇒ a clear, substantial decay curve
    g = sl.make_grid(_cfg(nu_floor=NU))
    ny, nx, nz = g.cfg.ny, g.cfg.nx, g.cfg.nz
    ky = 2.0 * np.pi * m / g.cfg.Ly
    y = (jnp.arange(ny) * g.dy).astype(jnp.float64)
    shape = jnp.sin(ky * y)[:, None, None] * jnp.ones((ny, nx, nz))
    st = _state(U0 * shape, jnp.zeros((ny, nx, nz)), jnp.zeros((ny, nx, nz + 1)))
    dt, n_out, every = 0.05, 40, 5
    rate = NU * ky ** 2
    ts, amps = [0.0], [U0]
    first, step = True, 0
    for _ in range(n_out * every):
        st, _ = sl.step(st, g=g, dt=dt, u_geo=(0.0, 0.0), f_cor=0.0, first=first)
        first = False; step += 1
        if step % every == 0:
            t = step * dt
            amp = 2.0 * np.mean(np.asarray(st.u)[:, 0, nz // 2]
                                * np.asarray(jnp.sin(ky * y)))
            ts.append(t); amps.append(amp)
    ts = np.array(ts)
    ax.plot(ts, amps, "o", color="tab:blue", ms=4, label="interior amp  num")
    ax.plot(ts, U0 * np.exp(-rate * ts), "--", color="k",
            label=f"U₀·exp(−ν k² t),  ν k²={rate:.4f}")
    ax.set_xlabel("time [s]"); ax.set_ylabel("modal amplitude [m/s]")
    ax.set_title(f"Viscous decay (k mode m={m})")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)


def rk3_convergence_panel(ax):
    NU, U0, m, T = 8.0, 0.2, 6, 10.0
    g = sl.make_grid(_cfg(nu_floor=NU))
    ny, nx, nz = g.cfg.ny, g.cfg.nx, g.cfg.nz
    ky = 2.0 * np.pi * m / g.cfg.Ly
    y = (jnp.arange(ny) * g.dy).astype(jnp.float64)
    shape = jnp.sin(ky * y)[:, None, None] * jnp.ones((ny, nx, nz))
    ana = U0 * np.exp(-NU * ky ** 2 * T)
    dts = np.array([0.4, 0.2, 0.1, 0.05])
    errs = []
    for dt in dts:
        st = _state(U0 * shape, jnp.zeros((ny, nx, nz)), jnp.zeros((ny, nx, nz + 1)))
        n = int(round(T / dt)); first = True
        for _ in range(n):
            st, _ = sl.step(st, g=g, dt=dt, u_geo=(0.0, 0.0), f_cor=0.0, first=first)
            first = False
        amp = 2.0 * np.mean(np.asarray(st.u)[:, 0, nz // 2]
                            * np.asarray(jnp.sin(ky * y)))
        errs.append(abs(amp - ana))
    errs = np.array(errs)
    slope = np.polyfit(np.log(dts), np.log(errs), 1)[0]
    ax.loglog(dts, errs, "o-", color="tab:green", label=f"SSP-RK3 (slope {slope:.2f})")
    ax.loglog(dts, errs[-1] * (dts / dts[-1]) ** 3, "--", color="k", label="O(dt³)")
    ax.set_xlabel("dt [s]"); ax.set_ylabel("|amp − analytic| at T")
    ax.set_title("RK3 temporal convergence")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("results/les_dycore_analytic"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    coriolis_panel(axes[0][0])
    viscous_k2_panel(axes[0][1])
    decay_panel(axes[1][0])
    rk3_convergence_panel(axes[1][1])
    fig.suptitle("Spectral incompressible LES dycore — term-by-term analytic "
                 "verification", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = args.out / "les_dycore_analytic.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
