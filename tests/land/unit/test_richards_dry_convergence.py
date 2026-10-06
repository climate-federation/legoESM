"""Richards solve on a dry top layer: damped Picard must not spike or create water.

The inputs are two columns recorded from a production AMIP run (res6, 16 ranks)
whose top layer sat near residual water content over an even drier layer.  The
former solver (10 undamped Picard iterations, last iterate accepted) cycled
there without converging and returned a top-layer head of 3e4-6e4 m, theta
1.5-2.6 (theta_sat 0.39), creating 4-7 mm of water in one step.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.land.config import MultiLayerLandConfig
from legoesm.land.richards import solve_richards
from legoesm.land.soil_grid import SoilGridConfig, make_soil_grid, make_soil_grid_custom
from legoesm.land.soil_hydraulics import SoilHydraulicsConfig, psi_from_theta

jax.config.update("jax_enable_x64", True)

_DZ = (0.002932551319648094, 0.005865102639296188, 0.011730205278592375,
       0.02346041055718475, 0.0469208211143695, 0.093841642228739,
       0.187683284457478, 0.375366568914956, 0.750733137829912,
       1.501466275659824)
_DT = 1800.0
_HYDRO = SoilHydraulicsConfig(theta_r=0.1, theta_sat=0.39, alpha_vg=5.9,
                              n_vg=1.48, K_sat=3.64e-06)

# Converges in 15 damped iterations.
_COL_A = dict(
    psi=(-1190.830374891806, -7434.677586492803, -16.72747592215601,
         -14.064492098361354, -12.981365525264948, -12.485005320140518,
         -12.252760277349012, -12.149435847713908, -12.111269156561526,
         -12.102406103065592),
    theta=(0.10413029235019079, 0.10171468041080506, 0.13198784932670155,
           0.13476050520469096, 0.13612153939565944, 0.13680271434149063,
           0.1371353329629957, 0.13728632108270036, 0.13734257452072104,
           0.13735567513146188),
    sink=(2.463134078244782e-07, 1.0775864896513396e-07, 1.4289199498113514e-07,
          7.344946600651468e-08, 3.5631536538164284e-08, 1.6116858149498773e-08,
          6.466715411091555e-09, 2.0621221735817986e-09, 4.174522097210489e-10,
          3.4151887205668944e-11),
    flux_top=3.6051497751560144e-08)

# Still unconverged after 30 damped iterations (one of 3 in 35 recorded calls).
_COL_B = dict(
    psi=(-50180.852144268676, -166815.96556999325, -90466.11470037345,
         -274.3104863149544, -156.05141289151027, -121.65363612642861,
         -108.58429633100523, -103.45945035408037, -101.81028584714205,
         -101.50857957100571),
    theta=(0.1006856946734424, 0.10038522545570469, 0.10051674440281187,
           0.10835661064178011, 0.11095505985578148, 0.11234583967673872,
           0.11303800002813953, 0.11334408300076443, 0.11344739506243932,
           0.11346656373919044),
    sink=(3.1719428122855497e-07, 1.5379225337504824e-07, 7.630247767140311e-08,
          5.8054338899618824e-08, 3.034774441452275e-08, 1.3959169579515043e-08,
          5.422390465448492e-09, 1.5690190452716896e-09, 2.576998416938422e-10,
          1.3805199577403907e-11),
    flux_top=2.772393189929857e-08)


def _solve(col, richards_config):
    grid = make_soil_grid_custom(_DZ)
    out = solve_richards(
        jnp.asarray([col["psi"]]), jnp.asarray([col["theta"]]), grid, _HYDRO,
        richards_config, jnp.asarray([col["flux_top"]]), jnp.asarray([col["sink"]]),
        _DT, surface_water=jnp.zeros(1))
    dz = np.asarray(_DZ)
    residual = (np.asarray(out.theta_new[0]) @ dz - np.asarray(col["theta"]) @ dz
                - col["flux_top"] * _DT + np.asarray(col["sink"]) @ dz * _DT
                + float(out.surface_water[0])
                + (float(out.runoff_surface[0]) + float(out.runoff_subsurface[0]))
                / constants.rho_water * _DT)
    return out, residual


def _former():
    return MultiLayerLandConfig().richards._replace(max_iter=10, max_dse_per_iter=1e9)


def test_production_config_converges_and_conserves_on_dry_top_layer():
    out, residual = _solve(_COL_A, MultiLayerLandConfig().richards)
    assert bool(out.converged[0])
    assert float(out.n_iter[0]) < MultiLayerLandConfig().richards.max_iter
    assert float(out.theta_new[0, 0]) <= 0.39
    assert abs(residual) < 1e-7  # m of water


def test_unconverged_column_is_flagged_without_a_spike():
    out, residual = _solve(_COL_B, MultiLayerLandConfig().richards)
    assert not bool(out.converged[0])
    assert float(out.theta_new[0, 0]) <= 0.39
    assert float(out.psi_new[0, 0]) < 0.0
    assert abs(residual) < 1e-4  # m; measured 3.4e-5, former solver 7.4e-3


def test_former_undamped_ten_iterations_spike_on_these_columns():
    """The recorded columns do bite: the former settings reproduce the spike."""
    for col in (_COL_A, _COL_B):
        out, residual = _solve(col, _former())
        assert not bool(out.converged[0])
        assert float(out.theta_new[0, 0]) > 1.0
        assert residual > 3e-3


def test_float32_gradient_finite_through_converged_frozen_columns():
    """A column at rest has a zero trial change; the damping factor must not
    divide by a tiny floor there (its float32 VJP overflows to 0 * inf = NaN)."""
    hc = SoilHydraulicsConfig()
    grid = jax.tree.map(lambda x: x.astype(jnp.float32) if hasattr(x, "astype") else x,
                        make_soil_grid(SoilGridConfig(n_layers=8)))
    theta = jnp.full((3, 8), 0.25, dtype=jnp.float32)
    psi = psi_from_theta(theta, hc).astype(jnp.float32)
    sink = jnp.zeros((3, 8), dtype=jnp.float32)
    rc = MultiLayerLandConfig().richards

    def loss(flux_top):
        out = solve_richards(psi, theta, grid, hc, rc, flux_top, sink, _DT)
        assert out.psi_new.dtype == jnp.float32  # the solve itself runs in float32
        return jnp.sum(out.theta_new)

    g = jax.grad(loss)(jnp.zeros(3, dtype=jnp.float32))
    assert g.dtype == jnp.float32
    assert bool(jnp.all(jnp.isfinite(g)))


# Top layer pinned at the dry floor while bare-soil evaporation still draws on it
# (recorded production call: converged in 7 iterations, 0.103 mm of water created).
_HYDRO_SAND = SoilHydraulicsConfig(theta_r=0.065, theta_sat=0.41, alpha_vg=7.5,
                                   n_vg=1.89, K_sat=1.22e-05)
_COL_FLOOR = dict(
    psi=(-4162.179044655679, -26.30658940867679, -2.170612525080939,
         -2.1135974844446457, -2.1113390886565773, -2.1112973291307386,
         -2.1112903434504995, -2.111289578335616, -2.111289562391965,
         -2.1112895623783094),
    theta=(0.06503450000000001, 0.06812714291045885, 0.09373493156972039,
           0.09442014135515249, 0.09444799691578157, 0.09444851251322796,
           0.09444859876604773, 0.09444860821302226, 0.09444860840988069,
           0.09444860841004929),
    sink=(0.0, 0.0, 3.2088441941850865e-10, 1.4121971653219986e-10,
          4.747052382169177e-11, 1.0667773958906095e-11, 1.0773620938583233e-12,
          2.1976588137353353e-14, 1.8288858352457417e-17, 2.5331985392437404e-23),
    flux_top=-5.9147064077228316e-08)


def test_floor_clamp_water_is_reported_for_the_caller_to_charge_back():
    grid = make_soil_grid_custom(_DZ)
    col = _COL_FLOOR
    out = solve_richards(
        jnp.asarray([col["psi"]]), jnp.asarray([col["theta"]]), grid, _HYDRO_SAND,
        MultiLayerLandConfig().richards, jnp.asarray([col["flux_top"]]),
        jnp.asarray([col["sink"]]), _DT, surface_water=jnp.zeros(1))
    dz = np.asarray(_DZ)
    residual = (np.asarray(out.theta_new[0]) @ dz - np.asarray(col["theta"]) @ dz
                - col["flux_top"] * _DT + np.asarray(col["sink"]) @ dz * _DT
                + float(out.surface_water[0])
                + (float(out.runoff_surface[0]) + float(out.runoff_subsurface[0]))
                / constants.rho_water * _DT)
    assert bool(out.converged[0])
    assert residual > 5e-5                      # the clamp created ~0.1 mm
    assert abs(float(out.floor_water[0]) - residual) < 1e-12
    # a converged column off the floor reports nothing
    out_a, _ = _solve(_COL_A, MultiLayerLandConfig().richards)
    assert float(out_a.floor_water[0]) == 0.0


def test_layer_entering_below_the_floor_is_not_charged_as_withdrawal():
    """Lifting a layer that starts below the floor is a state correction; with
    no withdrawal this step, nothing may be charged to evaporation."""
    col = dict(_COL_FLOOR, flux_top=0.0, sink=(0.0,) * 10)
    theta = list(col["theta"]); theta[5] = 0.060          # below theta_r = 0.065
    col["theta"] = tuple(theta)
    grid = make_soil_grid_custom(_DZ)
    out = solve_richards(
        jnp.asarray([col["psi"]]), jnp.asarray([col["theta"]]), grid, _HYDRO_SAND,
        MultiLayerLandConfig().richards, jnp.asarray([col["flux_top"]]),
        jnp.asarray([col["sink"]]), _DT, surface_water=jnp.zeros(1))
    assert bool(out.converged[0])
    assert float(out.floor_water[0]) == 0.0
