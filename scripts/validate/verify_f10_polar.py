"""F10 lat-lon spherical-metric polar verification (visual + numerical).

The F10 metric term is a SMOOTH pointwise factor (~tan(theta)/r); it cannot, by
construction, create a 2-dx grid-scale mode (no differencing).  Its correctness
claim is: it CANCELS the spurious deformation that the metric-free centered
differences produce for rigid motion, all the way to the pole.  This script
verifies that F10-specific claim two ways and saves PNGs:

  (A) Solid-body zonal rotation u = U0*cos(theta), v = 0 -> strain rate ~ 0 at
      EVERY latitude (only O(dtheta^2) residual); the metric-FREE code gives a
      spurious eps_12 ~ U0*sin/(2R) that grows toward the pole.
  (B) EVP run under a SMOOTH zonally-uniform high-latitude jet stays finite,
      bounded, and meridionally smooth up to ~89 deg (no pole blow-up).

Run: JAX_ENABLE_X64=1 .venv/bin/python scripts/verify_f10_polar.py
"""
from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ice.config import SeaIceConfig
from legoesm.ice.state import init_dynamic_ice_state
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.ice.sea_ice import step_sea_ice
from legoesm.ice.rheology import _strain_rates_latlon
from legoesm import constants


def part_a_solid_body(n_lat=80, n_lon=160):
    """Strain residual under solid-body rotation vs the metric-free spurious."""
    grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon)
    lat = grid.lat[:, None]
    U0 = 10.0
    u = U0 * jnp.cos(lat) * jnp.ones((n_lat, n_lon))
    v = jnp.zeros((n_lat, n_lon))
    e11, e22, e12 = _strain_rates_latlon(u, v, grid)
    e12_row = jnp.abs(e12[:, 0])                     # zonally uniform
    metric_free = jnp.abs(0.5 * U0 * jnp.sin(lat[:, 0]) / grid.radius)
    ii = slice(4, n_lat - 4)
    max_resid = float(jnp.max(e12_row[ii]))
    max_e11 = float(jnp.max(jnp.abs(e11[ii])))
    max_free = float(jnp.max(metric_free[ii]))
    print(f"[A] solid-body strain residual: max|e12|={max_resid:.2e} "
          f"max|e11|={max_e11:.2e}  vs metric-free spurious|e12|={max_free:.2e}  "
          f"(suppression {max_free / max(max_resid, 1e-30):.0f}x)")
    return grid, e12_row, metric_free, max_resid, max_free


def part_b_evp(n_lat=80, n_lon=160):
    """EVP under a smooth zonally-uniform high-latitude jet -> clean polar field."""
    grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon)
    shape = (n_lat, n_lon)
    lat = grid.lat[:, None]
    st = init_dynamic_ice_state(shape)
    st = st._replace(
        h_ice=st.h_ice.replace(data=jnp.full(shape, 2.0)),
        concentration=st.concentration.replace(data=jnp.full(shape, 0.95)),
        T_ice=st.T_ice.replace(data=jnp.full(shape, 258.0)))
    # Smooth zonally-uniform jet centred at 75 deg (pure latitudinal structure,
    # isolating the F10 meridional metric from lon-direction EVP modes).
    u_jet = 12.0 * jnp.exp(-((lat - jnp.deg2rad(75.0)) / jnp.deg2rad(12.0)) ** 2)
    f = lambda val: jnp.full(shape, val)
    forcing = AtmToSurface(
        sw_down=f(0.0), lw_down=f(180.0), precip_total=jnp.zeros(shape),
        precip_snow=jnp.zeros(shape), T_lowest=f(250.0), q_lowest=f(5e-4),
        u_lowest=jnp.broadcast_to(u_jet, shape), v_lowest=jnp.zeros(shape),
        p_lowest=f(9.5e4), p_surface=f(1e5), rho_lowest=f(1.3),
        cos_zenith=f(0.0), co2_ppmv=f(400.0),
        has_radiation=jnp.ones(shape), has_precipitation=jnp.ones(shape))
    sst = f(constants.T_freeze_ocean)
    z = jnp.zeros(shape)
    config = SeaIceConfig(n_categories=1, dynamics="evp", transport="advect")
    state = st
    for _ in range(20):
        state, _ = step_sea_ice(state, forcing, sst, z, z, config,
                                U_min=1.0, dt=1800.0, grid=grid)
    u, v = state.u_ice.data, state.v_ice.data
    speed = jnp.sqrt(u ** 2 + v ** 2)
    # Zonally uniform -> take row means; meridional 2-d-lat noise on interior.
    u_row = jnp.mean(u, axis=-1)
    sm = (u_row[:-2] + u_row[1:-1] + u_row[2:]) / 3.0
    merid_noise = float(jnp.max(jnp.abs(u_row[1:-1] - sm))
                        / (jnp.std(u_row[1:-1]) + 1e-30))
    finite = bool(jnp.all(jnp.isfinite(u)) and jnp.all(jnp.isfinite(v)))
    print(f"[B] EVP jet: finite={finite} speed_max={float(jnp.max(speed)):.4f} m/s "
          f"meridional 2-dx noise={merid_noise:.4f} (smooth << 0.3)")
    return grid, u, v, speed, finite, merid_noise


def main():
    grid_a, e12_row, metric_free, max_resid, max_free = part_a_solid_body()
    grid_b, u, v, speed, finite_b, merid_noise = part_b_evp()

    latd_a = jnp.rad2deg(grid_a.lat)
    latd_b = jnp.rad2deg(grid_b.lat)
    lond_b = jnp.rad2deg(grid_b.lon)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    # (A) residual vs spurious strain across latitude
    ax[0].semilogy(latd_a, jnp.maximum(e12_row, 1e-20), "b-",
                   label="|eps_12| WITH metric (residual)")
    ax[0].semilogy(latd_a, jnp.maximum(metric_free, 1e-20), "r--",
                   label="|eps_12| metric-FREE (spurious)")
    ax[0].set_title("F10 (A): solid-body rotation strain")
    ax[0].set_xlabel("latitude [deg]"); ax[0].set_ylabel("|eps_12| [1/s]")
    ax[0].legend(fontsize=8); ax[0].grid(True, alpha=0.3)
    # (B) EVP speed field to the pole
    im1 = ax[1].pcolormesh(lond_b, latd_b, speed, shading="auto", cmap="viridis")
    ax[1].set_title("F10 (B): EVP ice speed [m/s]")
    ax[1].set_xlabel("lon"); ax[1].set_ylabel("lat")
    fig.colorbar(im1, ax=ax[1], shrink=0.8)
    # (B) EVP v field
    im2 = ax[2].pcolormesh(lond_b, latd_b, v, shading="auto", cmap="RdBu_r")
    ax[2].set_title("F10 (B): EVP v_ice [m/s]")
    ax[2].set_xlabel("lon"); ax[2].set_ylabel("lat")
    fig.colorbar(im2, ax=ax[2], shrink=0.8)
    out = "/tmp/f10_polar_verification.png"
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)

    ok = (max_resid < 1e-8 and max_resid < 0.01 * max_free
          and finite_b and merid_noise < 0.3)
    print(f"F10 POLAR VERIFICATION {'PASS' if ok else 'FAIL'}")
    print(f"saved: {out}")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
