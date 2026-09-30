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


class TestDistributedEquivalence:
    """#1321 item 2: the q_v del2 filter on a partitioned mesh (halo refresh
    + local stencil) must reproduce the serial filter on owned cells.

    Simulated ranks (no MPI needed) — the same wiring the driver applies
    under the distributed Voronoi lane: refresh the q_v CELL halo, run the
    module-scope ``_mpas_qv_smooth_step`` on the rank-local mesh, keep the
    owned cells.  Halo inputs are deliberately POISONED first so the test
    fails if the refresh step is dropped (the driver's exact stale-halo
    scenario: physics updated owned q_v after the last in-step exchange).
    """

    @pytest.mark.parametrize("n_ranks", [2, 3])
    def test_owned_cells_match_serial(self, mesh, n_ranks):
        from legoesm.driver.model_driver import _mpas_qv_smooth_step
        from legoesm.parallel.voronoi_partition import (
            partition_voronoi_mesh, build_local_mesh, scatter_to_local,
        )
        from legoesm.parallel.halo_exchange_voronoi import (
            exchange_local_simulated,
        )

        nlev = 5
        rng = np.random.default_rng(21)
        q_global = jnp.asarray(
            rng.uniform(0.0, 0.02, size=(mesh.nCells, nlev)))
        dt = 75.0
        nu = 0.5 / (dt * scalar_del2_cell_cfl_factor(mesh))
        q_serial = _mpas_qv_smooth_step(q_global, mesh, nu, dt)

        parts = [partition_voronoi_mesh(mesh, n_ranks, r)
                 for r in range(n_ranks)]
        local_meshes = [build_local_mesh(mesh, p) for p in parts]
        locals_q = []
        for p in parts:
            ql = np.asarray(scatter_to_local(q_global, p, "cell")).copy()
            ql[p.n_owned_cells:, :] = 9.99  # poison halo: refresh must fix
            locals_q.append(jnp.asarray(ql))

        refreshed = exchange_local_simulated(parts, locals_q, entity="cell")

        for r, (p, lm, ql) in enumerate(zip(parts, local_meshes, refreshed)):
            q_new_local = _mpas_qv_smooth_step(ql, lm, nu, dt)
            own = p.n_owned_cells
            g_ids = np.asarray(p.local_cells[:own])
            np.testing.assert_allclose(
                np.asarray(q_new_local[:own]),
                np.asarray(q_serial)[g_ids],
                atol=1e-13, rtol=0.0,
                err_msg=f"rank {r}/{n_ranks}: owned-cell filter mismatch",
            )

    def test_poisoned_halo_without_refresh_fails(self, mesh):
        """Non-vacuousness: skipping the halo refresh DOES corrupt
        boundary-owned cells — proving the refresh in the driver is
        load-bearing, not decorative."""
        from legoesm.driver.model_driver import _mpas_qv_smooth_step
        from legoesm.parallel.voronoi_partition import (
            partition_voronoi_mesh, build_local_mesh, scatter_to_local,
        )

        nlev = 5
        rng = np.random.default_rng(22)
        q_global = jnp.asarray(
            rng.uniform(0.0, 0.02, size=(mesh.nCells, nlev)))
        dt = 75.0
        nu = 0.5 / (dt * scalar_del2_cell_cfl_factor(mesh))
        q_serial = _mpas_qv_smooth_step(q_global, mesh, nu, dt)

        p = partition_voronoi_mesh(mesh, 2, 0)
        lm = build_local_mesh(mesh, p)
        ql = np.asarray(scatter_to_local(q_global, p, "cell")).copy()
        ql[p.n_owned_cells:, :] = 9.99          # stale/poisoned halo, no refresh
        q_new_local = _mpas_qv_smooth_step(jnp.asarray(ql), lm, nu, dt)
        own = p.n_owned_cells
        g_ids = np.asarray(p.local_cells[:own])
        diff = np.max(np.abs(np.asarray(q_new_local[:own])
                             - np.asarray(q_serial)[g_ids]))
        assert diff > 1e-6, (
            "poisoned halo did not perturb owned cells — the equivalence "
            "test above would be vacuous"
        )


class TestBiharmonic:
    """The scale-selective companion filter (2026-09-11).

    The Laplacian separates two-cell from four-cell structure by four; the
    biharmonic by sixteen.  That ratio IS the reason the operator exists, so it
    is the thing pinned here — along with the conservation and sign properties
    that let it be used as a filter at all.
    """

    def test_sign_damps(self, mesh):
        """``q + dt*nu4*del4(q)`` must SHRINK a perturbation, not grow it."""
        from legoesm.core.operators_voronoi import scalar_del4_cell_3d
        q = _rand_q(mesh, seed=3)
        anom = q - q.mean(axis=0, keepdims=True)
        nu4, dt = 1.0e14, 100.0
        out = q + dt * nu4 * scalar_del4_cell_3d(q, mesh)
        out_anom = out - out.mean(axis=0, keepdims=True)
        assert float(jnp.std(out_anom)) < float(jnp.std(anom))

    def test_conserves_the_per_level_integral(self, mesh):
        """Each pass is a flux divergence, so the area integral is unchanged."""
        from legoesm.core.operators_voronoi import scalar_del4_cell_3d
        q = _rand_q(mesh, seed=4)
        d4 = scalar_del4_cell_3d(q, mesh)
        area = jnp.asarray(mesh.areaCell)[:, None]
        assert float(jnp.max(jnp.abs((area * d4).sum(axis=0)))) < 1e-20

    def test_eigenvalue_is_minus_the_laplacian_squared(self, mesh):
        """The whole selectivity argument reduces to this identity: because the
        operator is ``-del2(del2)``, its eigenvalue on every mesh mode is minus
        the SQUARE of the Laplacian's, so the damping RATIO between any two
        scales is squared.  Pinned on the discrete operator's own top mode, not
        argued from a continuum ``k^4``; the textbook 4-versus-16 figures do not
        hold here (measured 2.47 and 6.08 between 240 and 479 km)."""
        from legoesm.core.operators_voronoi import scalar_del4_cell_3d
        rng = np.random.default_rng(0)
        v = jnp.asarray(rng.normal(size=(mesh.nCells, 1)))
        lam = 0.0
        for _ in range(300):
            w = scalar_del2_cell_3d(v, mesh)
            lam = float(jnp.linalg.norm(w))
            v = w / lam
        d4 = scalar_del4_cell_3d(v, mesh)
        lam4 = float((d4 * v).sum() / (v * v).sum())
        # del4 = -del2(del2), and del2's eigenvalue is negative, so the
        # biharmonic eigenvalue is -lam^2 on the same mode.
        assert lam4 == pytest.approx(-lam ** 2, rel=1e-6)

    def test_gershgorin_bound_holds(self, mesh):
        """The driver's stability guard assumes |lambda(del2)| <= 2*g_max.  A
        guard resting on a FALSE bound would admit an unstable coefficient, so
        the bound is measured rather than asserted in prose."""
        rng = np.random.default_rng(1)
        v = jnp.asarray(rng.normal(size=(mesh.nCells, 1)))
        lam = 0.0
        for _ in range(300):
            w = scalar_del2_cell_3d(v, mesh)
            lam = float(jnp.linalg.norm(w))
            v = w / lam
        g_max = float(scalar_del2_cell_cfl_factor(mesh))
        assert lam <= 2.0 * g_max
        assert lam > g_max          # and g_max ALONE is not a bound

    def test_driver_step_applies_both_terms(self, mesh):
        """The driver helper with only the biharmonic on must differ from the
        Laplacian-only result and from the untouched field."""
        from legoesm.core.operators_voronoi import scalar_del4_cell_3d
        from legoesm.driver.model_driver import _mpas_qv_smooth_step
        q = _rand_q(mesh, seed=5)
        dt = 100.0
        # Scale the coefficients to THIS mesh: the biharmonic goes as 1/dx^4,
        # so a value sized for a 120 km mesh is invisible on a 2000 km one.
        g_max = float(scalar_del2_cell_cfl_factor(mesh))
        nu2 = 0.1 / (dt * g_max)
        nu4 = 0.1 / (dt * g_max ** 2)
        off = _mpas_qv_smooth_step(q, mesh, 0.0, dt)
        d2 = _mpas_qv_smooth_step(q, mesh, nu2, dt)
        d4 = _mpas_qv_smooth_step(q, mesh, 0.0, dt, nu4=nu4)
        both = _mpas_qv_smooth_step(q, mesh, nu2, dt, nu4=nu4)
        np.testing.assert_allclose(np.asarray(off), np.asarray(q))
        assert not np.allclose(np.asarray(d4), np.asarray(q))
        assert not np.allclose(np.asarray(d4), np.asarray(d2))
        want = np.clip(np.asarray(q + dt * (nu2 * scalar_del2_cell_3d(q, mesh)
                                            + nu4 * scalar_del4_cell_3d(q, mesh))),
                       0.0, None)
        np.testing.assert_allclose(np.asarray(both), want, rtol=1e-12)

    def test_validate_mpas_accepts_biharmonic(self):
        _mpas_cfg(mpas_qv_smooth_del4_m4s=3.6e14).validate_strict()

    def test_validate_cdgrid_refuses_biharmonic(self):
        with pytest.raises(ValueError, match="MPAS-lane"):
            _cdgrid_cfg(mpas_qv_smooth_del4_m4s=3.6e14).validate_strict()

    @pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf"), 2.0e18])
    def test_validate_biharmonic_bounds(self, bad):
        with pytest.raises(ValueError, match="mpas_qv_smooth_del4_m4s"):
            _mpas_cfg(mpas_qv_smooth_del4_m4s=bad).validate_strict()

    @pytest.mark.parametrize("n_ranks", [2, 3])
    def test_biharmonic_owned_cells_match_serial(self, mesh, n_ranks):
        """The biharmonic reaches TWO cells, so the composed operator equals
        ``-del2(del2)`` on owned cells only if the local mesh carries enough
        halo.  MEASURED, not assumed: at this partitioner's default halo depth
        it already does, so the driver's optional mid-operator exchange is
        redundant here and is kept only for a depth-1 partition.  Both paths
        are pinned against serial, and the poisoned halo proves the test can
        fail.
        """
        from legoesm.driver.model_driver import _mpas_qv_smooth_step
        from legoesm.parallel.halo_exchange_voronoi import (
            exchange_local_simulated,
        )
        from legoesm.parallel.voronoi_partition import (
            build_local_mesh, partition_voronoi_mesh, scatter_to_local,
        )

        nlev = 5
        rng = np.random.default_rng(22)
        q_global = jnp.asarray(rng.uniform(0.0, 0.02, size=(mesh.nCells, nlev)))
        dt = 75.0
        g_max = float(scalar_del2_cell_cfl_factor(mesh))
        nu4 = 0.1 / (dt * g_max ** 2)
        q_serial = _mpas_qv_smooth_step(q_global, mesh, 0.0, dt, nu4=nu4)

        parts = [partition_voronoi_mesh(mesh, n_ranks, r) for r in range(n_ranks)]
        local_meshes = [build_local_mesh(mesh, p) for p in parts]
        poisoned = []
        for p in parts:
            ql = np.asarray(scatter_to_local(q_global, p, "cell")).copy()
            ql[p.n_owned_cells:, :] = 9.99
            poisoned.append(jnp.asarray(ql))
        refreshed = exchange_local_simulated(parts, poisoned, entity="cell")

        stale_differs = False
        for r, (p, lm, ql) in enumerate(zip(parts, local_meshes, refreshed)):
            own = p.n_owned_cells
            g_ids = np.asarray(p.local_cells[:own])
            want = np.asarray(q_serial)[g_ids]
            got = _mpas_qv_smooth_step(ql, lm, 0.0, dt, nu4=nu4)
            np.testing.assert_allclose(
                np.asarray(got[:own]), want, atol=1e-13, rtol=0.0,
                err_msg=f"rank {r}/{n_ranks}: biharmonic owned-cell mismatch")
            # Same call on the UNREFRESHED input must NOT match, or the halo
            # exchange the driver performs is doing nothing and this test
            # could never fail.
            bad = _mpas_qv_smooth_step(poisoned[r], lm, 0.0, dt, nu4=nu4)
            if not np.allclose(np.asarray(bad[:own]), want, atol=1e-13, rtol=0.0):
                stale_differs = True
        assert stale_differs, (
            "a poisoned halo changed nothing — the exchange is not load-bearing "
            "and this test proves nothing")

    def test_floor_clipping_is_negligible_at_the_matched_coefficient(self, mesh):
        """Both reviewers asked for the plain ``max(q, 0)`` to be replaced by
        the conserving borrow, because the biharmonic has no maximum principle
        and this model once invented 0.08 kg/m2/day through an unlimited tracer
        clip.  Measured instead of argued.  On the production state at the
        coefficient that matches the Laplacian's grid-scale damping, the
        clipped deficit is 5.0e-11 of the field, and 1.1e-9 even at fifteen
        times that coefficient — far below the concern.  Reproduced here on a
        humidity-like field at the same fraction of the stability guard, with
        the threshold set an order of magnitude above what is measured so a
        real regression trips it.
        """
        from legoesm.core.operators_voronoi import scalar_del4_cell_3d
        rng = np.random.default_rng(23)
        nlev = 5
        lat = np.asarray(mesh.latCell)[:, None]
        # Smooth and positive, with the roughness a real humidity field carries
        # (its structure-function exponent is ~1.5, not white noise).
        q = jnp.asarray(np.clip(
            0.02 * np.cos(lat) ** 2 * np.ones((1, nlev))
            + 2e-4 * rng.normal(size=(mesh.nCells, nlev)), 0.0, None))
        dt = 75.0
        g_max = float(scalar_del2_cell_cfl_factor(mesh))
        nu4 = 0.0133 * 0.5 / (dt * g_max ** 2)   # production's 1.33% of guard
        raw = q + dt * nu4 * scalar_del4_cell_3d(q, mesh)
        ratio = float(jnp.abs(jnp.minimum(raw, 0.0)).sum()) / float(q.sum())
        assert ratio < 1e-8, (
            f"clipped deficit is {ratio:.2e} of the field; the conserving "
            "borrow decision must be revisited")


class TestConservingFloor:
    """The biharmonic branch's positivity floor must not create water: clip,
    then rescale each level's positives so ``sum_c A_c q_c`` is unchanged."""

    def _negative_field(self, mesh):
        # a checkerboard-ish field with a few cells driven well below zero,
        # the state the del4 step can hand the floor
        q = _rand_q(mesh, seed=3, lo=0.005, hi=0.02)
        q = q.at[jnp.array([3, 10, 17, 40]), :].set(-0.01)
        return q

    def test_floor_conserves_per_level_integral_and_is_nonnegative(self, mesh):
        from legoesm.driver.model_driver import _qv_level_conserving_floor
        q = self._negative_field(mesh)
        area = jnp.asarray(mesh.areaCell, dtype=jnp.float64)
        out = _qv_level_conserving_floor(q, area)
        assert float(jnp.min(out)) >= 0.0
        before = jnp.sum(area[:, None] * q, axis=0)
        after = jnp.sum(area[:, None] * out, axis=0)
        np.testing.assert_allclose(np.asarray(after), np.asarray(before),
                                   rtol=1e-12, atol=0.0)
        # non-vacuity: the old floor created water on every level
        created = jnp.sum(area[:, None] * jnp.maximum(q, 0.0), axis=0) - before
        assert bool(jnp.all(created > 0.0))
        # a positive field passes through untouched (factor exactly 1)
        qp = _rand_q(mesh, seed=4, lo=0.001, hi=0.02)
        np.testing.assert_array_equal(np.asarray(_qv_level_conserving_floor(qp, area)),
                                      np.asarray(qp))

    def test_owned_mask_excludes_halo_from_the_sums(self, mesh):
        from legoesm.driver.model_driver import _qv_level_conserving_floor
        q = self._negative_field(mesh)
        area = jnp.asarray(mesh.areaCell, dtype=jnp.float64)
        owned = jnp.arange(mesh.nCells) < mesh.nCells // 2
        out = _qv_level_conserving_floor(q, area, owned_mask=owned)
        w = jnp.where(owned, area, 0.0)
        np.testing.assert_allclose(
            np.asarray(jnp.sum(w[:, None] * out, axis=0)),
            np.asarray(jnp.sum(w[:, None] * q, axis=0)), rtol=1e-12, atol=0.0)

    def test_del4_step_uses_the_conserving_floor(self, mesh):
        from legoesm.driver.model_driver import _mpas_qv_smooth_step
        # a two-cell spike the biharmonic overshoots below zero around
        q = jnp.full((mesh.nCells, NLEV), 1.0e-6).at[7, :].set(0.02)
        area = jnp.asarray(mesh.areaCell, dtype=jnp.float64)
        dt = 100.0
        nu4 = 0.45 / (dt * scalar_del2_cell_cfl_factor(mesh) ** 2)
        out = _mpas_qv_smooth_step(q, mesh, 0.0, dt, nu4=nu4)
        assert float(jnp.min(out)) >= 0.0
        # the unfloored step really goes negative here, so the floor binds
        from legoesm.core.operators_voronoi import scalar_del4_cell_3d
        raw = q + dt * (nu4 * scalar_del4_cell_3d(q, mesh)).astype(q.dtype)
        assert float(jnp.min(raw)) < 0.0
        # the floor conserves the STEP's integral exactly; the fp32 mesh
        # operator itself conserves the input integral only to ~1e-7
        np.testing.assert_allclose(
            np.asarray(jnp.sum(area[:, None] * out, axis=0)),
            np.asarray(jnp.sum(area[:, None] * raw, axis=0)), rtol=1e-12, atol=0.0)
        np.testing.assert_allclose(
            np.asarray(jnp.sum(area[:, None] * out, axis=0)),
            np.asarray(jnp.sum(area[:, None] * q, axis=0)), rtol=1e-6, atol=0.0)

    def test_floor_is_differentiable(self, mesh):
        from legoesm.driver.model_driver import _qv_level_conserving_floor
        q = self._negative_field(mesh)
        area = jnp.asarray(mesh.areaCell, dtype=jnp.float64)
        g = jax.grad(lambda x: jnp.sum(_qv_level_conserving_floor(x, area) ** 2))(q)
        assert bool(jnp.all(jnp.isfinite(g))) and float(jnp.max(jnp.abs(g))) > 0.0
