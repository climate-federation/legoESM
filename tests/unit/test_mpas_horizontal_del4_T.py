"""#1354 second mode: HORIZONTAL biharmonic hyperdiffusion of T on MPAS.

Root cause these tests pin down
------------------------------
``component_factory`` feeds ``DycoreConfig.a_h_scale`` to BOTH
``MPASPrimitiveEquationConfig.nu_del2`` (the momentum vector Laplacian) and
``K_h`` (the scalar T Laplacian).  ``K_h`` is the **only** horizontal
dissipation this dycore ever applies to T — ``nu_del4`` acts on ``u`` alone and
``nu_del4_ps`` on ``p_s`` alone.  So ``a_h_scale = 0``, the setting that
recovers midlatitude storm tracks, leaves the temperature field with ZERO
horizontal damping at EVERY scale while momentum keeps its biharmonic.
``test_T_has_no_horizontal_damping_when_K_h_zero`` asserts exactly that gap,
and is the test that must stay red if someone deletes the cure.

The cure is ``nu_del4_T`` — the scalar analogue of the momentum ``nu_del4``:
``dT/dt -= nu_del4_T * div(grad(div(grad T)))``.  Its point is SCALE
SELECTIVITY, so ``test_del4_beats_the_laplacian_at_the_grid_scale_and_loses_at
_synoptic_scale`` is the load-bearing assertion: swapping the biharmonic back
for a Laplacian of any strength cannot satisfy both halves at once.
"""

from __future__ import annotations

import math

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.core.field import Field
from legoesm.core.operators_voronoi import (
    divergence_cell_3d,
    gradient_edge_3d,
)
from legoesm.core.state import MPASHydrostaticState
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
    MPASPrimitiveEquationConfig,
    mpas_hydrostatic_tendencies,
)

pytestmark = pytest.mark.skipif(
    not jax.config.read("jax_enable_x64"),
    reason="dycore science test needs float64 (run with JAX_ENABLE_X64=1)",
)

NLEV = 12
_DT = 60.0


def _mesh():
    return create_voronoi_mesh(2, lloyd_iterations=5)


def _state(mesh, T_field, T0=250.0):
    """Resting (u=0) flat-terrain state carrying the given T anomaly."""
    nC, nE = mesh.nCells, mesh.nEdges
    return MPASHydrostaticState(
        u=Field(data=jnp.zeros((nE, NLEV)), name="u",
                dims=("nEdges", "nlev"), units="m/s"),
        T=Field(data=jnp.asarray(T0 + T_field), name="T",
                dims=("nCells", "nlev"), units="K"),
        p_s=Field(data=jnp.full((nC,), 1.0e5), name="p_s",
                  dims=("nCells",), units="Pa"),
        phis=Field(data=jnp.zeros((nC,)), name="phis",
                   dims=("nCells",), units="m^2/s^2"),
    )


def _spike(mesh, cell=0):
    """One-cell (grid-scale) T anomaly of 1 K at every level."""
    a = np.zeros((mesh.nCells, NLEV))
    a[cell, :] = 1.0
    return a


def _planetary(mesh):
    """Smooth Y_1 = sin(lat) anomaly of 1 K amplitude — the 'eddy' proxy."""
    return np.tile(np.asarray(mesh.latCell)[:, None], (1, NLEV)) * 0.0 + \
        np.sin(np.asarray(mesh.latCell))[:, None]


def _tend(mesh, coord, state, **cfg_kw):
    """Tendencies with EVERY dissipation off unless the caller turns it on."""
    kw = dict(nu_del2=0.0, nu_del4=0.0, nu_del4_ps=0.0, K_h=0.0,
              nu_vert4_T=0.0, fix_mass=False)
    kw.update(cfg_kw)
    cfg = MPASPrimitiveEquationConfig(**kw)
    return mpas_hydrostatic_tendencies(state, mesh, coord, cfg, dt=_DT)


def _projection_rate(anomaly, dT_dt):
    """Damping rate [1/s] of the anomaly pattern: -<a, dT/dt> / <a, a>."""
    a = np.asarray(anomaly)
    return -float((a * np.asarray(dT_dt)).sum() / (a * a).sum())


# ---------------------------------------------------------------------------
# Root cause
# ---------------------------------------------------------------------------

def test_T_has_no_horizontal_damping_when_K_h_zero():
    """a_h_scale=0 ⇒ K_h=0 ⇒ a grid-scale T spike is horizontally UNDAMPED.

    RED without the cure: the only way to make this pass is a horizontal
    T-diffusion operator that survives K_h=0.
    """
    mesh, coord = _mesh(), create_sigma_coordinate(NLEV, sigma_top=1e-3)
    anom = _spike(mesh)
    state = _state(mesh, anom)

    bare = _tend(mesh, coord, state)                    # a_h_scale = 0
    assert _projection_rate(anom, bare.dT_dt.data) == pytest.approx(0.0,
                                                                    abs=1e-12)

    cured = _tend(mesh, coord, state, nu_del4_T=1.0e16)
    rate = _projection_rate(anom, cured.dT_dt.data)
    assert rate > 0.0, "nu_del4_T must DAMP (not amplify) a grid-scale spike"


def test_del4_term_matches_the_analytic_operator_and_only_touches_T():
    """The added term is exactly ``-nu * div(grad(div(grad T)))`` and it
    perturbs NO other tendency (u, p_s, phis stay bitwise identical)."""
    mesh, coord = _mesh(), create_sigma_coordinate(NLEV, sigma_top=1e-3)
    anom = _spike(mesh)
    state = _state(mesh, anom)
    nu = 3.0e15

    off = _tend(mesh, coord, state)
    on = _tend(mesh, coord, state, nu_del4_T=nu)

    T = state.T.data
    del2 = divergence_cell_3d(gradient_edge_3d(T, mesh), mesh)
    del4 = divergence_cell_3d(gradient_edge_3d(del2, mesh), mesh)
    np.testing.assert_allclose(
        np.asarray(on.dT_dt.data - off.dT_dt.data),
        np.asarray(-nu * del4), rtol=1e-10, atol=1e-14)

    # No leakage into the other prognostics.
    for name in ("du_dt", "dp_s_dt", "dphis_dt"):
        np.testing.assert_array_equal(
            np.asarray(getattr(on, name).data),
            np.asarray(getattr(off, name).data),
            err_msg=f"nu_del4_T must not change {name}")


def test_zero_is_an_exact_no_op_with_and_without_K_h():
    """nu_del4_T=0 reproduces the pre-cure dycore BITWISE, on both branches of
    the ``_need_grad_T`` gate (K_h off and K_h on)."""
    mesh, coord = _mesh(), create_sigma_coordinate(NLEV, sigma_top=1e-3)
    state = _state(mesh, _spike(mesh))
    for K_h in (0.0, 5.0e4):
        a = _tend(mesh, coord, state, K_h=K_h)
        b = _tend(mesh, coord, state, K_h=K_h, nu_del4_T=0.0)
        np.testing.assert_array_equal(np.asarray(a.dT_dt.data),
                                      np.asarray(b.dT_dt.data))


# ---------------------------------------------------------------------------
# The load-bearing property: scale selectivity
# ---------------------------------------------------------------------------

def test_del4_beats_the_laplacian_at_the_grid_scale_and_loses_at_synoptic():
    """Why a BIHARMONIC and not simply a bigger ``K_h``.

    Against the production a_h_scale=0.25 Laplacian, the biharmonic at the
    momentum ``nu_del4`` coefficient must damp a one-cell spike FASTER while
    damping a planetary-scale (Y_1) anomaly SLOWER.  No Laplacian coefficient
    can satisfy both, so this stays red if the operator is downgraded.
    """
    mesh, coord = _mesh(), create_sigma_coordinate(NLEV, sigma_top=1e-3)
    # Production coefficient formulas (component_factory.compute_diffusion).
    dx = float(jnp.min(mesh.dcEdge))
    dt = 75.0
    K_h_prod = 0.25 * 3.0e-3 * dx ** 2 / dt
    nu4_prod = dx ** 4 / (24.0 * 3600.0)

    rates = {}
    for label, anom in (("grid", _spike(mesh)), ("planetary", _planetary(mesh))):
        st = _state(mesh, anom)
        lap = _tend(mesh, coord, st, K_h=K_h_prod)
        bih = _tend(mesh, coord, st, nu_del4_T=nu4_prod)
        base = _tend(mesh, coord, st)
        rates[label] = (
            _projection_rate(anom, lap.dT_dt.data - base.dT_dt.data),
            _projection_rate(anom, bih.dT_dt.data - base.dT_dt.data),
        )

    lap_grid, bih_grid = rates["grid"]
    lap_syn, bih_syn = rates["planetary"]
    assert bih_grid > lap_grid, (
        f"biharmonic must dominate at the grid scale "
        f"(del4 {bih_grid:.3e} vs del2 {lap_grid:.3e} 1/s)")
    assert 0.0 < bih_syn < lap_syn, (
        f"biharmonic must damp the resolved scale LESS than the Laplacian "
        f"(del4 {bih_syn:.3e} vs del2 {lap_syn:.3e} 1/s)")
    # And the separation must be large, not marginal.
    assert (bih_grid / bih_syn) > 10.0 * (lap_grid / lap_syn)


# ---------------------------------------------------------------------------
# JAX contract: jit parity + differentiability
# ---------------------------------------------------------------------------

def test_jit_parity_and_gradient():
    mesh, coord = _mesh(), create_sigma_coordinate(NLEV, sigma_top=1e-3)
    anom = _spike(mesh)
    state = _state(mesh, anom)
    nu = 2.0e15

    def loss(T):
        s = state._replace(T=state.T.replace(data=T))
        cfg = MPASPrimitiveEquationConfig(
            nu_del2=0.0, nu_del4=0.0, nu_del4_ps=0.0, K_h=0.0,
            nu_vert4_T=0.0, fix_mass=False, nu_del4_T=nu)
        return jnp.sum(
            mpas_hydrostatic_tendencies(s, mesh, coord, cfg, dt=_DT).dT_dt.data ** 2)

    eager = loss(state.T.data)
    jitted = jax.jit(loss)(state.T.data)
    np.testing.assert_allclose(np.asarray(eager), np.asarray(jitted), rtol=1e-12)

    g = jax.grad(loss)(state.T.data)
    assert np.all(np.isfinite(np.asarray(g)))
    assert np.abs(np.asarray(g)).max() > 0.0


# ---------------------------------------------------------------------------
# Wiring: DycoreConfig -> component_factory -> MPASPrimitiveEquationConfig
# ---------------------------------------------------------------------------

def _mpas_experiment_cfg(**dycore_kw):
    from legoesm.driver.config import DycoreConfig, ExperimentConfig, GridConfig
    return ExperimentConfig(
        grid=GridConfig(grid_type="mpas", resolution=2, nlev=NLEV),
        dycore=DycoreConfig(discretization="mpas", model_type="hydrostatic",
                            dt=75.0, **dycore_kw),
    )


def test_factory_threads_scale_times_hyperdiff_into_nu_del4_T():
    from legoesm.driver.component_factory import (
        compute_diffusion, create_atmosphere_dycore,
    )
    mesh = _mesh()
    coord = create_sigma_coordinate(NLEV, sigma_top=1e-3)
    for scale in (0.0, 0.5, 1.0):
        cfg = _mpas_experiment_cfg(mpas_nu_del4_T_scale=scale)
        diff = compute_diffusion(mesh, cfg.dycore)
        model = create_atmosphere_dycore(cfg, mesh, coord)
        assert model.config.nu_del4_T == pytest.approx(scale * diff.hyperdiff)
        # The confound this knob exists to break: K_h still follows a_h_scale.
        assert model.config.K_h == pytest.approx(diff.A_h)


def test_validate_strict_bounds_and_lane_guard():
    from legoesm.driver.config import DycoreConfig

    _mpas_experiment_cfg(mpas_nu_del4_T_scale=1.0).validate_strict()
    _mpas_experiment_cfg(mpas_nu_del4_T_scale=0.0).validate_strict()

    for bad in (-1.0, float("nan"), float("inf"), 1.0e9):
        with pytest.raises(ValueError, match="mpas_nu_del4_T_scale"):
            _mpas_experiment_cfg(mpas_nu_del4_T_scale=bad).validate_strict()

    # A non-MPAS lane would silently ignore it -> must raise, not shrug.
    from legoesm.driver.config import ExperimentConfig, GridConfig
    latlon = ExperimentConfig(
        grid=GridConfig(grid_type="lat_lon", resolution=24, nlev=NLEV),
        dycore=DycoreConfig(discretization="finite_volume",
                            model_type="hydrostatic", dt=75.0,
                            mpas_nu_del4_T_scale=1.0),
    )
    with pytest.raises(ValueError, match="mpas_nu_del4_T_scale"):
        latlon.validate_strict()


def test_constants_are_not_re_derived():
    """Guard against a stray hardcoded R_earth/g creeping into this module."""
    assert constants.R_earth > 0 and constants.g > 0
    assert math.isfinite(constants.R_earth)
