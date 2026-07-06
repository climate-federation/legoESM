"""Drive the canopy land model through a 3-day diurnal cycle and plot outputs.

Runs four contrasting single-column cases in parallel:
    1. Tropical forest  — DBF, LAI=5, hot humid, CO2=420 ppm
    2. Temperate grass  — GRA_C3, LAI=2, moderate, CO2=420 ppm
    3. C4 savanna       — SAV, LAI=1.5, hot, fC4=1.0
    4. Dry shrubland    — SHR, LAI=1, dry (low initial theta)

The atmospheric forcing uses an idealised diurnal cycle: sinusoidal SW,
constant LW, constant wind, T following a damped sinusoid.

Outputs a 6-panel time-series figure: SW forcing, T_surface, LE, H, GPP,
soil moisture in the top layer.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legoesm import constants
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land.canopy import CanopyConfig, CanopyLandParams
from legoesm.land.config import MultiLayerLandConfig
from legoesm.land.multilayer_land import (
    init_multilayer_land_state,
    step_multilayer_land,
)
from legoesm.land.surface_scheme import TwoLeafCanopyConfig

# Phase 3 script-local shims: ``CanopyLandConfig`` / ``init_canopy_land_state``
# / ``step_canopy_land`` were removed when canopy became a surface scheme.
init_canopy_land_state = init_multilayer_land_state
step_canopy_land = step_multilayer_land


def CanopyLandConfig(*, multilayer=None, canopy=None) -> MultiLayerLandConfig:
    base = multilayer if multilayer is not None else MultiLayerLandConfig()
    cc = canopy if canopy is not None else CanopyConfig()
    return base._replace(surface_scheme=cc)


# ----- case definitions ------------------------------------------------------

CASE_NAMES = ["tropical_DBF", "temperate_GRA_C3", "C4_savanna", "dry_shrub"]

def build_params(ncol: int) -> CanopyLandParams:
    LAI      = jnp.array([5.0, 2.0, 1.5, 1.0])
    hc       = jnp.array([18.0, 1.0, 3.0, 1.5])
    fC4      = jnp.array([0.0, 0.0, 1.0, 0.0])
    FNonVeg  = jnp.zeros(ncol)
    CI       = jnp.array([0.7, 0.85, 0.8, 0.75])
    kn       = jnp.full(ncol, 0.3)
    # Vcmax25 from Jiang & Ryu (2016) Table A1
    Vc3      = jnp.array([66.0, 78.0,  0.0, 62.0])
    Vc4      = jnp.array([ 0.0,  0.0, 40.0,  0.0])
    m_C3     = jnp.full(ncol, 9.0)
    m_C4     = jnp.full(ncol, 4.0)
    b0_C3    = jnp.full(ncol, 0.01)
    b0_C4    = jnp.full(ncol, 0.04)
    alf      = jnp.full(ncol, 0.3)
    TgC      = jnp.array([27.0, 18.0, 28.0, 22.0])
    ALB_VIS  = jnp.array([0.05, 0.10, 0.15, 0.18])
    ALB_NIR  = jnp.array([0.20, 0.25, 0.30, 0.35])
    emis     = jnp.full(ncol, 0.97)
    rz0m     = jnp.array([0.055, 0.12, 0.12, 0.12])
    rd       = jnp.array([0.67,  0.68, 0.68, 0.68])
    return CanopyLandParams(
        LAI=LAI, hc=hc, fC4=fC4, FNonVeg=FNonVeg, CI=CI, kn=kn,
        Vcmax25_C3_leaf=Vc3, Vcmax25_C4_leaf=Vc4,
        m_C3=m_C3, m_C4=m_C4, b0_C3=b0_C3, b0_C4=b0_C4,
        alf=alf, TgC=TgC,
        ALB_VIS=ALB_VIS, ALB_NIR=ALB_NIR, emissivity=emis,
        rz0m=rz0m, rd=rd,
    )


def diurnal_forcing(t_hr: float, ncol: int) -> AtmToSurface:
    """Idealised diurnal forcing at hour t_hr within the day."""
    phase = 2.0 * np.pi * (t_hr - 6.0) / 24.0  # noon at t=12
    cosz  = max(0.0, np.sin(np.pi * (t_hr - 6.0) / 12.0))
    sw_top = 1000.0 * cosz
    T_air  = 295.0 + 6.0 * np.sin((t_hr - 9.0) * np.pi / 12.0)  # peak ~3pm
    q_air  = jnp.array([0.018, 0.010, 0.012, 0.006])   # dry shrub is drier
    return AtmToSurface(
        T_lowest   =jnp.full(ncol, float(T_air)),
        q_lowest   =q_air,
        u_lowest   =jnp.full(ncol, 3.0),
        v_lowest   =jnp.full(ncol, 0.0),
        p_lowest   =jnp.full(ncol, 98000.0),
        p_surface  =jnp.full(ncol, 101325.0),
        rho_lowest =jnp.full(ncol, 1.18),
        sw_down    =jnp.full(ncol, float(sw_top)),
        lw_down    =jnp.full(ncol, 370.0),
        cos_zenith =jnp.full(ncol, float(cosz)),
        precip_total=jnp.zeros(ncol),
        precip_snow =jnp.zeros(ncol),
        co2_ppmv   =jnp.full(ncol, 420.0),
        has_radiation=True,
        has_precipitation=False,
    )


def run(n_days: int = 3, dt_s: float = 1800.0):
    ncol = len(CASE_NAMES)
    cfg = CanopyLandConfig(
        multilayer=MultiLayerLandConfig(),
        canopy=CanopyConfig(max_iters=50, tol=1e-4),
    )
    state = init_canopy_land_state(ncol, cfg, T_init=293.0)

    # Drier initial soil for the shrub case
    theta_init = state.theta_soil
    theta_init = theta_init.at[3, :].set(0.18)  # dry shrub
    state = state._replace(theta_soil=theta_init)

    params = build_params(ncol)

    step_fn = jax.jit(
        lambda s, f: step_canopy_land(
            s, f, cfg, 1.0, dt_s,
            lat=jnp.zeros(ncol), doy=180.0, land_params=params,
        )
    )

    n_steps = int(n_days * 24 * 3600 / dt_s)
    times_hr = np.zeros(n_steps)
    T_surf = np.zeros((n_steps, ncol))
    LE     = np.zeros((n_steps, ncol))
    H      = np.zeros((n_steps, ncol))
    GPP    = np.zeros((n_steps, ncol))  # from -co2_flux * 1e3 [mgC/m2/s]
    theta0 = np.zeros((n_steps, ncol))
    sw_in  = np.zeros(n_steps)

    for k in range(n_steps):
        t_hr = (k * dt_s / 3600.0) % 24.0
        times_hr[k] = k * dt_s / 3600.0
        forcing = diurnal_forcing(t_hr, ncol)
        state, response, _ = step_fn(state, forcing)

        T_surf[k] = np.asarray(response.T_surface)
        LE[k]     = np.asarray(response.lhflx)
        H[k]      = np.asarray(response.shflx)
        GPP[k]    = -np.asarray(response.co2_flux) * 1e3   # mgC/m2/s
        theta0[k] = np.asarray(state.theta_soil[:, 0])
        sw_in[k]  = float(forcing.sw_down[0])

    return times_hr, sw_in, T_surf, LE, H, GPP, theta0


def make_plot(times_hr, sw_in, T_surf, LE, H, GPP, theta0, out_path: Path):
    colors = ["tab:green", "tab:olive", "tab:orange", "tab:brown"]
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)

    ax = axes[0, 0]
    ax.plot(times_hr, sw_in, color="gold", lw=2)
    ax.set_ylabel("SW_down  [W m$^{-2}$]")
    ax.set_title("Atmospheric forcing (idealised diurnal cycle)")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for j, name in enumerate(CASE_NAMES):
        ax.plot(times_hr, T_surf[:, j] - constants.T_freeze, color=colors[j], label=name)
    ax.set_ylabel("T$_{surface}$  [\u00b0C]")
    ax.set_title("Canopy / surface temperature")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    for j in range(4):
        ax.plot(times_hr, LE[:, j], color=colors[j])
    ax.set_ylabel("LE  [W m$^{-2}$]")
    ax.set_title("Latent heat flux")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    for j in range(4):
        ax.plot(times_hr, H[:, j], color=colors[j])
    ax.set_ylabel("H  [W m$^{-2}$]")
    ax.set_title("Sensible heat flux")
    ax.grid(alpha=0.3)

    ax = axes[2, 0]
    for j in range(4):
        ax.plot(times_hr, GPP[:, j], color=colors[j])
    ax.set_ylabel("GPP  [mgC m$^{-2}$ s$^{-1}$]")
    ax.set_xlabel("Hour of simulation")
    ax.set_title("Gross primary productivity")
    ax.grid(alpha=0.3)

    ax = axes[2, 1]
    for j in range(4):
        ax.plot(times_hr, theta0[:, j], color=colors[j])
    ax.set_ylabel(r"$\theta$ top layer  [m$^3$ m$^{-3}$]")
    ax.set_xlabel("Hour of simulation")
    ax.set_title("Top-layer soil moisture")
    ax.grid(alpha=0.3)

    fig.suptitle("legoESM canopy land model — 3-day diurnal cycle",
                 fontsize=13, y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    jax.config.update("jax_enable_x64", True)
    out = run(n_days=3, dt_s=1800.0)
    out_dir = Path(__file__).resolve().parents[1] / "outputs" / "canopy_diurnal"
    out_dir.mkdir(parents=True, exist_ok=True)
    make_plot(*out, out_dir / "canopy_diurnal.png")

    # print daily totals
    times_hr, sw_in, T_surf, LE, H, GPP, theta0 = out
    day_sel = (times_hr >= 24) & (times_hr < 48)
    print("\n=== Day-2 mean fluxes ===")
    print(f"{'case':22s} {'LE':>8s} {'H':>8s} {'GPP':>12s}")
    for j, name in enumerate(CASE_NAMES):
        LE_m  = LE[day_sel, j].mean()
        H_m   = H[day_sel, j].mean()
        # daily GPP in gC/m2/day: mean mgC/m2/s * 86400 / 1000
        GPP_d = GPP[day_sel, j].mean() * 86.4
        print(f"{name:22s} {LE_m:8.1f} {H_m:8.1f} {GPP_d:12.3f} gC/m2/day")
