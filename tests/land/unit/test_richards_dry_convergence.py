"""Richards solve on a dry top layer: damped Picard must not spike or create water.

The inputs are two columns recorded from a production AMIP run (res6, 16 ranks)
whose top layer sat near residual water content over an even drier layer.  The
former solver (10 undamped Picard iterations, last iterate accepted) cycled
there without converging and returned a top-layer head of 3e4-6e4 m, theta
1.5-2.6 (theta_sat 0.39), creating 4-7 mm of water in one step.  On current
code the post-solve water take-back removes that water again (from the layers
below), but the spike in the top layer remains.
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
                / constants.rho_water * _DT
                - float(out.refill[0]))
    return out, residual


def _former():
    # Former iteration settings (10 undamped iterations).  Main's post-loop
    # water take-back still runs, so this is not the historical solver; the
    # spike survives it on these columns.
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
    # The unconverged iterate lost 3.4e-5 m; the post-solve correction returns it.
    assert abs(residual) < 1e-12  # m
    assert abs(float(out.water_created[0])) < 1e-12


def test_former_undamped_ten_iterations_spike_on_these_columns():
    """The recorded columns do bite: the former iteration settings still spike.

    Main's take-back now closes the water budget of the spiked step (measured
    residual ~1e-17 m on both columns) by draining the layers below, so the
    defect left is the spike itself: a positive top-layer head of 3e4-6e4 m
    and theta 1.5-2.6 against theta_sat 0.39.
    """
    for col in (_COL_A, _COL_B):
        out, _ = _solve(col, _former())
        assert not bool(out.converged[0])
        assert float(out.theta_new[0, 0]) > 1.0
        assert float(out.psi_new[0, 0]) > 1e4


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


def test_float32_gradient_finite_with_no_room_below_saturation():
    """Clapp-Hornberger layers on their saturated plateau (psi < 0, theta =
    theta_sat) leave the post-solve give-back no room; its fraction must not
    divide by a tiny floor (float32 VJP -num/den^2 overflows to NaN)."""
    hc = SoilHydraulicsConfig(retention_curve="clapp_hornberger")
    grid = jax.tree.map(lambda x: x.astype(jnp.float32) if hasattr(x, "astype") else x,
                        make_soil_grid(SoilGridConfig(n_layers=8)))
    theta = jnp.full((2, 8), hc.theta_sat, dtype=jnp.float32)
    psi = jnp.full((2, 8), 0.5 * float(np.max(hc.psi_sat)), dtype=jnp.float32)
    sink = jnp.zeros((2, 8), dtype=jnp.float32)
    rc = MultiLayerLandConfig().richards

    def loss(flux_top):
        out = solve_richards(psi, theta, grid, hc, rc, flux_top, sink, _DT)
        return jnp.sum(out.theta_new) + 1e-6 * jnp.sum(out.psi_new)

    g = jax.grad(loss)(jnp.zeros(2, dtype=jnp.float32))
    assert bool(jnp.all(jnp.isfinite(g)))
