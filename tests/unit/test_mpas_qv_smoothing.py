"""MPAS horizontal q_v del2 smoothing (post-step).

Covers the 2026-07-23 speckle fix (and its 2026-07-24 hybrid-coordinate
hardening after codex adversarial review):

  - ``legoesm.core.operators_voronoi.scalar_del2_cell_3d``: the UNWEIGHTED
    conservative SCVT scalar Laplacian (no dp weighting — deliberately: the
    default hybrid coordinate's surface dp goes <= 0 over high terrain, so a
    dp-weighted form would divide by it).  Conserves ``sum_c A_c q_c`` per
    level (to fp roundoff), monotone under the CFL bound, coordinate-agnostic.
  - ``legoesm.driver.model_driver._mpas_qv_smooth_step``: the ACTUAL driver
    code (unweighted del2 + q>=0 floor), unit-tested directly — including on
    the pathological hybrid coordinate (codex B2 regression) and in fp32.
  - ``scalar_del2_cell_cfl_factor``: the shared geometry monotonicity
    factor, used by BOTH the driver guard and these tests so the two cannot
    drift.
  - ``ExperimentConfig.validate_strict``: bounds + MPAS-lane-only refusal.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.core.operators_voronoi import (
    scalar_del2_cell_3d, scalar_del2_cell_cfl_factor,
)
from legoesm.grids.vertical import make_hybrid_levels
from legoesm.grids.voronoi import create_voronoi_mesh

NLEV = 3


@pytest.fixture(scope="module")
def mesh():
    """Level-2 SCVT mesh (162 cells) — same fixture family as the
    vector-calculus identity tests."""
    return create_voronoi_mesh(2)


def _rand_q(mesh, seed=0, lo=0.0, hi=0.02, nlev=NLEV):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.uniform(lo, hi, size=(mesh.nCells, nlev)))


class TestOperator:
    # --- plain form: the actual MPAS driver path ------------------------

    def test_constant_field_annihilated(self, mesh):
        q = jnp.full((mesh.nCells, NLEV), 0.011)
        lap = scalar_del2_cell_3d(q, mesh)
        assert float(jnp.max(jnp.abs(lap))) < 1e-18

    def test_plain_form_conserves_area_integral(self, mesh):
        """sum_c A_c lap_c = 0 (exact in exact arithmetic; checked to 1e-12
        rel in x64), AND per level (the operator is purely horizontal — no
        vertical coupling can hide a per-level violation behind an all-level
        sum)."""
        q = _rand_q(mesh)
        lap = scalar_del2_cell_3d(q, mesh)
        area = jnp.asarray(mesh.areaCell)[:, None]
        for k in range(NLEV):
            resid = float(jnp.sum(area[:, 0] * lap[:, k]))
            scale = float(jnp.sum(area[:, 0] * jnp.abs(lap[:, k])))
            assert abs(resid) < 1e-12 * max(scale, 1.0)

    def test_smoothing_sign_damps_extrema(self, mesh):
        """lap < 0 at a strict interior maximum: +nu*lap decreases it."""
        q = jnp.zeros((mesh.nCells, NLEV)).at[37, :].set(1.0e-2)
        lap = scalar_del2_cell_3d(q, mesh)
        assert float(lap[37, 0]) < 0.0

    def test_plain_step_is_monotone_and_positive(self, mesh):
        """Driver path: under nu*dt*g_max <= 0.5 the update
        q + nu*dt*plain_lap is a convex combination — new values stay
        inside [min q, max q] per level, so q >= 0 is preserved with NO
        clipping (the driver's max(.,0) floor is then a no-op, to roundoff)."""
        q = _rand_q(mesh, seed=3)
        # Sparse zeros so positivity is actually stressed.
        q = q.at[::7, :].set(0.0)
        dt = 75.0
        nu = 0.5 / (dt * scalar_del2_cell_cfl_factor(mesh))
        q_new = q + dt * nu * scalar_del2_cell_3d(q, mesh)
        assert float(jnp.min(q_new)) >= float(jnp.min(q)) - 1e-15
        assert float(jnp.max(q_new)) <= float(jnp.max(q)) + 1e-15
        assert float(jnp.min(q_new)) >= -1e-15

    def test_plain_step_conserves_area_integral(self, mesh):
        """Driver path conserves the per-level sum_c A_c q_c integral (the
        floor never binds under the guard, so this is exact in exact
        arithmetic; checked to rel 1e-13 in x64)."""
        q = _rand_q(mesh, seed=5)
        dt = 75.0
        nu = 0.5 / (dt * scalar_del2_cell_cfl_factor(mesh))
        q_new = jnp.maximum(q + dt * nu * scalar_del2_cell_3d(q, mesh), 0.0)
        area = jnp.asarray(mesh.areaCell)[:, None]
        for k in range(NLEV):
            m0 = float(jnp.sum(area[:, 0] * q[:, k]))
            m1 = float(jnp.sum(area[:, 0] * q_new[:, k]))
            assert m1 == pytest.approx(m0, rel=1e-13)

    def test_checkerboard_variance_decreases(self, mesh):
        """Grid-scale noise (the speckle) is what the smoother must damp."""
        rng = np.random.default_rng(7)
        base = 1.0e-2
        q = jnp.asarray(
            base + 5e-3 * rng.choice([-1.0, 1.0], size=(mesh.nCells, NLEV)))
        dt = 75.0
        nu = 0.4 / (dt * scalar_del2_cell_cfl_factor(mesh))
        q_new = q + dt * nu * scalar_del2_cell_3d(q, mesh)
        assert float(jnp.var(q_new)) < 0.9 * float(jnp.var(q))

    def test_autodiff_flows(self, mesh):
        def loss(q):
            return jnp.sum(scalar_del2_cell_3d(q, mesh) ** 2)

        g = jax.grad(loss)(_rand_q(mesh, seed=10))
        assert bool(jnp.all(jnp.isfinite(g)))
        assert float(jnp.max(jnp.abs(g))) > 0.0

    # --- regression: pathological hybrid coordinate (codex B2), driver code -

    def test_driver_helper_finite_and_positive_on_hybrid_low_ps(self, mesh):
        """The default MPAS hybrid coordinate has surface-layer dp <= 0 for
        p_s below ~2/3 p_ref (~660 hPa), reached over high terrain — a
        dp-weighted del2 would divide by it (Inf/NaN).  The ACTUAL driver
        code (module-scope ``_mpas_qv_smooth_step``, unweighted del2 + floor)
        must stay finite and >= 0 there.  Guards the confirmed bug AND that
        the driver never reintroduces dp weighting (the helper takes no dp
        and the plain operator has no dp parameter — a revert is a
        TypeError, not a silent NaN)."""
        from legoesm.driver.model_driver import _mpas_qv_smooth_step
        nlev = 8
        coord = make_hybrid_levels(nlev, p_top_Pa=200.0, stretching=2.0)
        # p_s spanning the dp<=0 crossing across cells (350..1013 hPa).
        ps = jnp.asarray(np.linspace(35000.0, 101300.0, mesh.nCells))
        dp = coord.layer_thickness_dp(ps)                 # (nCells, nlev)
        assert float(jnp.min(dp)) < 0.0                   # non-vacuous: in-regime
        rng = np.random.default_rng(11)
        q = jnp.asarray(rng.uniform(0.0, 0.02, size=(mesh.nCells, nlev)))
        dt = 75.0
        nu = 0.5 / (dt * scalar_del2_cell_cfl_factor(mesh))
        q_new = _mpas_qv_smooth_step(q, mesh, nu, dt)
        assert bool(jnp.all(jnp.isfinite(q_new)))
        assert float(jnp.min(q_new)) >= 0.0
        # It IS the plain-del2 + floor expression (the driver's exact op).
        expect = jnp.maximum(q + dt * nu * scalar_del2_cell_3d(q, mesh), 0.0)
        assert float(jnp.max(jnp.abs(q_new - expect))) == 0.0

    def test_driver_helper_fp32_safe(self, mesh):
        """Production may run fp32: the helper preserves q dtype, stays
        finite and non-negative, and does not upcast."""
        from legoesm.driver.model_driver import _mpas_qv_smooth_step
        rng = np.random.default_rng(12)
        q = jnp.asarray(rng.uniform(0.0, 0.02, size=(mesh.nCells, NLEV)),
                        dtype=jnp.float32)
        dt = 75.0
        nu = 0.5 / (dt * scalar_del2_cell_cfl_factor(mesh))
        q_new = _mpas_qv_smooth_step(q, mesh, nu, dt)
        assert q_new.dtype == jnp.float32
        assert bool(jnp.all(jnp.isfinite(q_new)))
        assert float(jnp.min(q_new)) >= 0.0

    # --- shared CFL factor helper ---------------------------------------

    def test_cfl_factor_positive_and_matches_manual(self, mesh):
        g = scalar_del2_cell_cfl_factor(mesh)
        assert np.isfinite(g) and g > 0.0
        eoc = np.asarray(mesh.edgesOnCell)
        m = eoc >= 0
        safe = np.maximum(eoc, 0)
        gm = np.sum(np.where(m, np.asarray(mesh.dvEdge)[safe]
                             / np.maximum(np.asarray(mesh.dcEdge)[safe], 1e-30),
                             0.0), axis=0) / np.asarray(mesh.areaCell)
        assert g == pytest.approx(float(np.max(gm)), rel=1e-13)


# ---------------------------------------------------------------------------
# validate_strict lane guards (mirrors test_mpas_land_boundary patterns)
# ---------------------------------------------------------------------------

def _mpas_cfg(**kw):
    from legoesm.driver.config import (
        DycoreConfig, ExperimentConfig, GridConfig, OutputConfig,
    )
    kw.setdefault("radiation", "gray")
    return ExperimentConfig(
        grid=GridConfig(grid_type="mpas", resolution=1, nlev=8,
                        vertical_coord="sigma"),
        dycore=DycoreConfig(dt=600.0, discretization="mpas"),
        output=OutputConfig(diag_days=1),
        days=1, dataset="analytical",
        **kw,
    )


def _cdgrid_cfg(**kw):
    from legoesm.driver.config import (
        DycoreConfig, ExperimentConfig, GridConfig, OutputConfig,
    )
    return ExperimentConfig(
        grid=GridConfig(resolution=8, nlev=8),
        dycore=DycoreConfig(dt=600.0),
        output=OutputConfig(diag_days=1),
        days=1, dataset="analytical", radiation="gray",
        **kw,
    )


def test_validate_mpas_accepts_smoothing():
    _mpas_cfg(mpas_qv_smooth_del2_m2s=2.0e5).validate_strict()


def test_validate_mpas_accepts_zero_default():
    _mpas_cfg().validate_strict()


def test_validate_cdgrid_refuses_smoothing():
    with pytest.raises(ValueError, match="MPAS-lane"):
        _cdgrid_cfg(mpas_qv_smooth_del2_m2s=2.0e5).validate_strict()


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf"), 2.0e8])
def test_validate_bounds(bad):
    with pytest.raises(ValueError, match="mpas_qv_smooth_del2_m2s"):
        _mpas_cfg(mpas_qv_smooth_del2_m2s=bad).validate_strict()
