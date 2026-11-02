"""Single-column ocean model (KPP + MARBL): short integrations, closed-column budgets.

Run with JAX_ENABLE_X64=1.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.ocean.biogeochemistry.marbl import MarblConfig, MarblTracers
from legoesm.ocean.column import (
    ColumnConfig,
    ColumnForcing,
    initial_state,
    make_column_grid,
    run_column,
)
from legoesm.ocean.column.model import cyclic_interp, diurnal_factor

X64 = jax.config.jax_enable_x64
NLEV = 30


def _setup(closed: bool):
    grid = make_column_grid(lat=31.7, lon=-64.2, depth=4000.0, n_levels=NLEV,
                            dz_surface=5.0, dz_deep=300.0)
    z = np.asarray(grid.zt)
    surf = np.exp(-z / 60.0)
    deep = 1.0 - np.exp(-z / 400.0)
    T = 2.5 + 22.0 * np.exp(-z / 400.0)
    S = 35.0 + 1.5 * np.exp(-z / 500.0)
    tr = dict(
        PO4=0.05 + 2.1 * deep, NO3=0.5 + 32.0 * deep, SiO3=1.5 + 60.0 * deep,
        NH4=0.05 * surf, Fe=1.5e-4 + 5e-4 * deep, Lig=2.5e-3 + 0.0 * z,
        O2=230.0 - 100.0 * deep, DIC=2000.0 + 250.0 * deep, DIC_ALT_CO2=1950.0 + 250.0 * deep,
        ALK=2380.0 + 30.0 * deep, ALK_ALT_CO2=2380.0 + 30.0 * deep,
        DOC=40.0 + 30.0 * surf, DON=2.0 + 3.0 * surf, DOP=0.05 + 0.15 * surf,
        DOPr=0.1 + 0.0 * z, DONr=2.0 + 0.0 * z, DOCr=40.0 + 0.0 * z, zooC=0.5 * surf,
        spChl=0.04 * surf, spC=1.0 * surf, spP=surf / 117.0, spFe=2e-5 * surf, spCaCO3=0.05 * surf,
        diatChl=0.02 * surf, diatC=0.4 * surf, diatP=0.4 * surf / 117.0, diatFe=0.8e-5 * surf,
        diatSi=0.06 * surf,
        diazChl=0.002 * surf, diazC=0.05 * surf, diazP=0.05 * surf / 300.0, diazFe=2e-6 * surf,
    )
    bgc = MarblTracers(**{k: jnp.asarray(v) for k, v in tr.items()})
    n = 12
    full = lambda v: jnp.full((n,), v)  # noqa: E731
    wind = 0.0 if closed else 8.0
    forcing = ColumnForcing(
        tau_x=full(0.0 if closed else 0.08), tau_y=full(0.0),
        u10_sqr=full(wind ** 2), sw_net=full(220.0), q_nonsolar=full(-180.0),
        freshwater=full(0.0), atm_press=full(1.0), ifrac=full(0.0), xco2=full(420.0),
        dust_flux=full(0.0 if closed else 3e-11), iron_flux=full(0.0 if closed else 1e-10),
        nox_flux=full(0.0), nhy_flux=full(0.0),
        T_clim=jnp.broadcast_to(jnp.asarray(T), (n, NLEV)),
        S_clim=jnp.broadcast_to(jnp.asarray(S), (n, NLEV)),
    )
    state = initial_state(jnp.asarray(T), jnp.asarray(S), bgc, time0=120.0 * constants.seconds_per_day)
    return grid, forcing, state


def _carbon_inventory(bgc, dz):
    return float(sum(jnp.sum(getattr(bgc, k) * dz)
                     for k in ("DIC", "DOC", "DOCr", "zooC", "spC", "diatC", "diazC", "spCaCO3")))


@pytest.mark.skipif(not X64, reason="budget closure needs float64")
def test_closed_column_carbon_budget():
    grid, forcing, state0 = _setup(closed=True)
    cfg = ColumnConfig(dt=3600.0)
    final, daily, diag = run_column(state0, forcing, grid, cfg, MarblConfig(), 2, 24)
    c0 = _carbon_inventory(state0.bgc, grid.dz)
    c1 = _carbon_inventory(final.bgc, grid.dz)
    burial = float(jnp.sum(diag.POC_sed_loss + diag.CaCO3_sed_loss)) * constants.seconds_per_day
    assert abs((c1 - c0) + burial) < 1e-10 * c0
    assert float(jnp.max(jnp.abs(diag.flux_co2))) == 0.0


def test_forced_column_runs_and_mixes():
    grid, forcing, state0 = _setup(closed=False)
    cfg = ColumnConfig(dt=3600.0)
    final, daily, diag = run_column(state0, forcing, grid, cfg, MarblConfig(), 3, 24)
    for leaf in jax.tree_util.tree_leaves(final):
        assert bool(jnp.all(jnp.isfinite(leaf)))
    assert float(jnp.min(diag.h_bl)) > float(grid.dz[0])
    assert float(jnp.max(diag.h_bl)) < 4000.0
    assert daily.T.shape == (3, NLEV)
    assert float(jnp.sum(diag.NPP)) > 0.0
    # wind drives a surface current
    assert float(jnp.hypot(final.u[0], final.v[0])) > 0.0


def test_cyclic_interp_and_diurnal_factor():
    samples = jnp.arange(12.0)
    assert float(cyclic_interp(samples, jnp.asarray(0.5 / 12.0))) == pytest.approx(0.0)
    assert float(cyclic_interp(samples, jnp.asarray(1.0 / 12.0))) == pytest.approx(0.5)
    assert float(cyclic_interp(samples, jnp.asarray(0.0))) == pytest.approx(5.5)  # wraps Dec->Jan
    # the diurnal factor averages to one over a day
    t = jnp.arange(0.0, constants.seconds_per_day, 600.0) + 170.0 * constants.seconds_per_day
    f = jax.vmap(lambda s: diurnal_factor(s, 31.7, -64.2))(t)
    assert float(jnp.mean(f)) == pytest.approx(1.0, rel=2e-2)
    assert float(jnp.min(f)) == 0.0
