"""Richards solver on the dry end of a Clapp-Hornberger soil, and water
conservation of every Picard iterate.

The global LMIP runs (2 deg, HWSD texture -> Cosby -> Clapp-Hornberger with
theta_r = 0, 2.9 mm top layer) produced dry-season runoff spikes of up to
1e10 mm/day in ~1.5 % of land cells: a bone-dry top layer (theta ~ 0.015,
psi ~ -3e9 m) receiving a little rain.  The pressure-head Picard step
dpsi = residual/(C/dt) used the dry-end capacity (C/dt ~ 1e-16 1/s), so 1 mm
of rain became dpsi ~ +1e10 m, the elastic branch turned that pressure into
water, and the surface link pumped it into the pond as runoff.  The dry-side
floor never bound: it limits drying, and sits at psi_sat*10^(4b).

The solver now steps in theta (the mass the linearised balance solved for),
cascades water a layer cannot hold up the column and onto the surface, bounds
psi at the hydrostatic head of the pond, forbids exfiltration, and uses a
capacity consistent with theta(psi) through saturation (the elastic branch
starts at air entry).  These tests pin the properties that fix guaranteed:
water is conserved at EVERY iterate, runoff never exceeds the water present,
theta never exceeds porosity (plus the elastic allowance), and the wet-side
behaviour (floods, water tables, hydrostatic columns) is physical.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.land.richards import RichardsConfig, solve_richards
from legoesm.land.soil_grid import SoilGridConfig, make_soil_grid
from legoesm.land.soil_hydraulics import (
    SoilHydraulicsConfig,
    air_entry_psi,
    moisture_capacity,
    psi_from_theta,
    theta_from_psi,
)

from legoesm import constants

jax.config.update("jax_enable_x64", True)

_DT = 3600.0
_RHO = float(constants.rho_water)
# The production LMIP column: 10 layers over 3 m, growth 2 (2.93 mm top layer).
_GRID = make_soil_grid(SoilGridConfig(n_layers=10, total_depth=3.0, growth_factor=2.0))
_DZ = np.asarray(_GRID.dz)
# The spike cell's soil (15S 32E; sand 50 %, clay 26 % through Cosby 1984).
_CH = SoilHydraulicsConfig(retention_curve="clapp_hornberger", theta_sat=0.426,
                           psi_sat=-0.169, b_ch=7.1, K_sat=5.36e-6, theta_r=0.0)
_VG = SoilHydraulicsConfig()                       # van Genuchten loam defaults
_BC = SoilHydraulicsConfig(retention_curve="brooks_corey")
# Driest state the solver admits: theta at the -1e6 m pressure floor of the
# cell's own curve (0.047 for this clay).  States below it belong to old
# restarts and are raised once by scripts/data/cap_land_restart.py.
_PSI_FLOOR = -1.0e6
def _floor(hyd):
    return float(theta_from_psi(jnp.asarray(_PSI_FLOOR), hyd))
# End of the dry season in the spike cell: top layers at the floor.
_DRY = jnp.maximum(jnp.array([[0.015, 0.02, 0.03, 0.05, 0.08, 0.12, 0.15, 0.18, 0.20, 0.20]]),
                   _floor(_CH))
_MM_TOL = 1e-6                                     # water-budget closure [mm]


def _step(hyd, theta0, psi0, rain_mm_h, *, pond0_m=0.0, max_iter=10,
          bottom_bc="free_drainage"):
    out = solve_richards(
        psi0, theta0, _GRID, hyd, RichardsConfig(max_iter=max_iter, bottom_bc=bottom_bc),
        jnp.array([rain_mm_h / 1000.0 / 3600.0]), jnp.zeros_like(theta0), _DT,
        surface_water=jnp.array([pond0_m]))
    theta1 = np.asarray(out.theta_new[0])
    runoff_mm = float(out.runoff_surface[0]) / _RHO * _DT * 1000.0
    drain_mm = float(out.runoff_subsurface[0]) / _RHO * _DT * 1000.0
    pond_mm = float(out.surface_water[0]) * 1000.0
    budget_mm = (((theta1 - np.asarray(theta0[0])) * _DZ).sum() * 1000.0
                 + runoff_mm + drain_mm + pond_mm - pond0_m * 1000.0 - rain_mm_h)
    return out, theta1, np.asarray(out.psi_new[0]), runoff_mm, drain_mm, pond_mm, budget_mm


@pytest.mark.parametrize("rain_mm_h", [0.01, 0.1, 1.0, 10.0, 50.0])
def test_rain_on_bone_dry_clapp_hornberger_creates_no_water(rain_mm_h):
    """The spike case.  The old solver turned 1 mm/h into 6e6 mm of runoff and
    theta = 0.84 in the top layer."""
    psi0 = psi_from_theta(_DRY, _CH)
    out, theta1, psi1, runoff, drain, pond, budget = _step(_CH, _DRY, psi0, rain_mm_h)
    assert np.all(np.isfinite(theta1)) and np.all(np.isfinite(psi1))
    assert abs(budget) < _MM_TOL
    assert theta1.max() <= _CH.theta_sat * (1.0 + 1e-3)
    assert runoff <= rain_mm_h + 1e-9
    assert psi1.max() <= float(_GRID.z_node[-1]) + 0.05 + rain_mm_h / 1000.0 + 1e-9
    assert theta1[0] > float(_DRY[0, 0])             # the rain did wet the top layer


@pytest.mark.parametrize("max_iter", list(range(1, 13)))
def test_every_picard_iterate_conserves_water(max_iter):
    """Conservation must not depend on where the fixed-iteration loop stops:
    the excess-water cascade keeps each iterate closed."""
    psi0 = psi_from_theta(_DRY, _CH)
    _, theta1, _, runoff, _, _, budget = _step(_CH, _DRY, psi0, 10.0, max_iter=max_iter)
    assert abs(budget) < _MM_TOL
    assert theta1.max() <= _CH.theta_sat * (1.0 + 1e-3)
    assert runoff <= 10.0 + 1e-9


def test_drying_bone_dry_soil_is_bounded():
    """Evaporation from a column just above the floor: theta stops at the
    floor and nothing is invented while water above it remains."""
    floor = _floor(_CH)
    theta0 = jnp.full((1, 10), floor + 0.05)
    out, theta1, psi1, runoff_mm, drain_mm, pond_mm, budget_mm = _step(
        _CH, theta0, psi_from_theta(theta0, _CH), -0.1)
    assert theta1.min() >= floor - 1e-12
    assert theta1[0] < float(theta0[0, 0])            # the top layer did dry
    assert abs(budget_mm) < _MM_TOL


def test_flood_on_saturated_column_splits_rain_into_runoff_and_drainage():
    """Saturated, hydrostatic column under a full pond and 100 mm/h: the
    column passes K_sat*dt to drainage, the rest ponds and runs off, and the
    water table's pressure survives.  The old solver reported 97 mm of runoff
    AND 19 mm of drainage from 100 mm of rain (+16 mm created)."""
    psi_hs = jnp.asarray(air_entry_psi(_CH) + np.asarray(_GRID.z_node))[None, :]
    theta_hs = theta_from_psi(psi_hs, _CH)
    _, theta1, psi1, runoff, drain, pond, budget = _step(
        _CH, theta_hs, psi_hs, 100.0, pond0_m=0.05)
    assert abs(budget) < _MM_TOL
    assert drain == pytest.approx(_CH.K_sat * _DT * 1000.0, rel=0.02)
    assert runoff == pytest.approx(100.0 - drain, abs=0.5)
    assert pond == pytest.approx(50.0, abs=1e-6)
    assert psi1[-1] > 1.0                                  # water table still there
    assert np.all(theta1 >= _CH.theta_sat - 1e-9)


def test_falling_water_table_desaturates_without_overshoot():
    """A perched water table draining at K_sat: the layer that gives up the
    water desaturates by that amount; the old psi step overshot it to a deeply
    dry psi (tiny elastic capacity) and left +19 mm unaccounted."""
    theta0 = np.array([[0.30, 0.30, 0.30, 0.32, 0.34, 0.36, 0.40, 0.426, 0.426, 0.426]])
    psi0 = np.array(psi_from_theta(jnp.asarray(theta0), _CH))
    zn = np.asarray(_GRID.z_node)
    psi0[0, 7:] = air_entry_psi(_CH) + (zn[7:] - zn[7]) + 0.05
    theta0 = np.asarray(theta_from_psi(jnp.asarray(psi0), _CH))
    _, theta1, psi1, runoff, drain, _, budget = _step(
        _CH, jnp.asarray(theta0), jnp.asarray(psi0), 1.0)
    assert abs(budget) < _MM_TOL
    assert drain == pytest.approx(_CH.K_sat * _DT * 1000.0, rel=0.1)
    assert runoff == 0.0
    assert psi1[7:].min() > -1.0                          # no overshoot to dry
    assert theta1[7:].min() > 0.4


def test_hydrostatic_saturated_column_with_zero_flux_bottom_stays_put():
    psi_hs = jnp.asarray(air_entry_psi(_CH) + np.asarray(_GRID.z_node))[None, :]
    theta_hs = theta_from_psi(psi_hs, _CH)
    out, theta1, psi1, runoff, drain, pond, budget = _step(
        _CH, theta_hs, psi_hs, 0.0, bottom_bc="zero_flux")
    assert abs(budget) < _MM_TOL
    assert np.abs(psi1 - np.asarray(psi_hs[0])).max() < 1e-9
    assert runoff == 0.0 and drain == 0.0 and pond == 0.0


@pytest.mark.parametrize("hyd,theta_uniform", [(_CH, 0.25), (_VG, 0.30)])
def test_moist_column_under_rain_conserves_water(hyd, theta_uniform):
    theta0 = jnp.full((1, 10), theta_uniform)
    _, theta1, _, runoff, _, _, budget = _step(hyd, theta0, psi_from_theta(theta0, hyd), 2.0)
    assert abs(budget) < _MM_TOL
    assert theta1.max() <= float(hyd.theta_sat) * (1.0 + 1e-3)
    assert runoff == 0.0                                  # 2 mm fits in the column


def test_recovers_from_an_over_pressured_restart_state():
    """A top layer at psi = +1e4 m, as the old solver left cells in restarts:
    the next step must relieve it without inventing water."""
    psi0 = np.array(psi_from_theta(_DRY, _CH))
    psi0[0, 0] = 1.0e4
    theta0 = np.asarray(theta_from_psi(jnp.asarray(psi0), _CH))
    _, theta1, psi1, runoff, _, pond, budget = _step(
        _CH, jnp.asarray(theta0), jnp.asarray(psi0), 0.0)
    assert abs(budget) < _MM_TOL
    assert psi1.max() <= float(_GRID.z_node[-1]) + 0.05 + 1e-9
    assert runoff == 0.0


@pytest.mark.parametrize("hyd", [_CH, _VG, _BC],
                         ids=["clapp_hornberger", "van_genuchten", "brooks_corey"])
def test_theta_psi_are_inverses_through_saturation(hyd):
    """theta(psi) is strictly increasing from the unsaturated branch into the
    elastic branch (which starts at air entry), so psi(theta) inverts it, and
    the capacity is its derivative there: the solver's linear system and the
    state then agree on how much water a saturated layer stores."""
    psi_b = float(air_entry_psi(hyd))
    psi = jnp.array([-50.0, -5.0, -1.0, psi_b - 1e-3, psi_b + 1e-3, 0.5, 3.0])
    theta = theta_from_psi(psi, hyd)
    assert np.all(np.diff(np.asarray(theta)) > 0.0)
    back = np.asarray(psi_from_theta(theta, hyd))
    np.testing.assert_allclose(back[-3:], np.asarray(psi[-3:]), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(back[:3], np.asarray(psi[:3]), rtol=1e-3)
    # Capacity on the elastic branch == d theta / d psi == S_s * theta_sat.
    cap = np.asarray(moisture_capacity(psi, theta, hyd))
    np.testing.assert_allclose(cap[-3:], hyd.S_s * hyd.theta_sat, rtol=1e-12)
    eps = 1e-6
    fd = (np.asarray(theta_from_psi(psi[-3:] + eps, hyd))
          - np.asarray(theta_from_psi(psi[-3:] - eps, hyd))) / (2 * eps)
    np.testing.assert_allclose(fd, cap[-3:], rtol=1e-6)


# The second spike family (21S 166E, 2000-01-13 hourly replay): rain of 2.5 mm/h
# on a moist clay column.  The top layer saturates in the first iterate and
# ponds, the linearised surface link then over-infiltrates and leaves a
# negative pond, infiltration is cut, and the saturated layers' elastic
# capacity (4e-5) pushed the whole give-back onto layer 3, which hit the dry
# floor.  That floor sat at psi = -1e30 m, and K_half*(psi_wet - psi_floor) on
# the right-hand side then had to cancel against storage terms 1e25 times
# smaller: the Thomas sweep returned dpsi ~ 1e9 m, cascaded to a 1e7 m pond.
_CLAY = SoilHydraulicsConfig(retention_curve="clapp_hornberger", theta_sat=0.43793,
                             psi_sat=-0.2235, b_ch=8.467, K_sat=2.0e-6, theta_r=0.0)
_MOIST = jnp.array([[0.2057, 0.1905, 0.1727, 0.1704, 0.1816, 0.1962,
                     0.2100, 0.2200, 0.2300, 0.2400]])


def test_rain_on_moist_clay_does_not_collapse_the_column():
    theta, psi, pond = _MOIST, psi_from_theta(_MOIST, _CLAY), 0.0
    for _ in range(6):
        out, theta1, psi1, runoff_mm, drain_mm, pond_mm, budget_mm = _step(
            _CLAY, theta, psi, 2.5, pond0_m=pond)
        assert abs(budget_mm) < _MM_TOL
        assert runoff_mm <= 2.5 + pond * 1000.0 + _MM_TOL
        assert np.all(np.isfinite(psi1))
        # No layer dries below where it started (nothing leaves but drainage).
        assert np.all(theta1 >= np.asarray(_MOIST[0]) - 1e-3)
        assert np.all(theta1 <= _CLAY.theta_sat * (1.0 + _CLAY.S_s * 3.1))
        theta, psi, pond = out.theta_new, out.psi_new, float(out.surface_water[0])
    # Six hours of 2.5 mm/h on a 2.9 mm top layer: the top is near saturation
    # and the wetting front has reached layer 3, which is wetter, not at the floor.
    assert theta1[0] > 0.8 * _CLAY.theta_sat
    assert theta1[2] > np.asarray(_MOIST[0])[2] + 0.05


@pytest.mark.parametrize("rain_mm_h", [0.0, 1.0])
def test_floor_layer_between_wet_layers_is_solvable(rain_mm_h):
    """A layer at the dry floor sandwiched between moist ones (what the
    collapse left behind).  The floor is now -1e6 m, below oven dryness, so the
    coupling terms are a few digits above the storage terms, not twenty-five."""
    floor = _floor(_CLAY)
    theta0 = jnp.array([[0.30, 0.30, floor, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]])
    psi0 = psi_from_theta(theta0, _CLAY)           # -1e6 m in the floor layer
    out, theta1, psi1, runoff_mm, drain_mm, pond_mm, budget_mm = _step(
        _CLAY, theta0, psi0, rain_mm_h)
    assert np.all(np.isfinite(psi1)) and np.all(np.isfinite(theta1))
    assert psi1.min() >= -1.0e6 - 1e-6
    assert abs(budget_mm) < _MM_TOL
    assert runoff_mm <= rain_mm_h + _MM_TOL
    assert theta1.min() >= floor - 1e-12
    assert theta1.max() <= _CLAY.theta_sat * (1.0 + _CLAY.S_s * 3.1)
    # The wet layers stay wet: nothing pumps them into the floor layer's suction.
    assert theta1[0] > 0.25 and theta1[3] > 0.2


def test_restart_psi_below_the_floor_is_bounded_without_moving_water():
    """A restart written before the floor existed carries psi far below it;
    psi is bounded on entry and a dry hour moves nothing."""
    theta0 = _DRY
    psi0 = psi_from_theta(theta0, _CH) * 1.0e5     # an old restart's psi, far below the floor
    assert float(psi0[0, 0]) < -1.0e8
    out, theta1, psi1, runoff_mm, drain_mm, pond_mm, budget_mm = _step(_CH, theta0, psi0, 0.0)
    assert psi1.min() >= -1.0e6 - 1e-6
    # The dry layers exchange nothing; the moist layers redistribute by their own tiny K.
    assert np.allclose(theta1, np.asarray(theta0[0]), atol=1e-4)
    assert abs(budget_mm) < _MM_TOL and runoff_mm == 0.0
