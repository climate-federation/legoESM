"""Tests for the grid-agnostic physics pipeline refactoring.

Validates:
- ColumnAdapter flatten/unflatten round-trips for cubed-sphere, lat-lon,
  and single-column grids.
- Kernel registries resolve the correct functions for radiation, convection,
  and microphysics.
- The physics pipeline produces identical results across all three grid
  topologies when given the same column data.
- Hard-coded shape assumptions (jnp.prod, explicit reshape dimensions) are
  removed: the adapter handles all geometry.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from legoesm.driver.grid_adapters import (
    ColumnAdapter,
    SingleColumnGrid,
    make_adapter,
)
from legoesm.driver.kernel_registry import (
    RADIATION_REGISTRY,
    CONVECTION_REGISTRY,
    MICROPHYSICS_REGISTRY,
    resolve_kernel,
    available_schemes,
)
from legoesm.driver.physics_pipeline import (
    PhysicsPipeline,
    PhysicsOutput,
    HeldRadiation,
    build_physics_pipeline,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.latlon import create_latlon_grid


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

NLEV = 5
N_CS = 4        # cubed-sphere C4 → 6*4*4 = 96 columns
N_LAT = 8       # lat-lon 8x16 → 128 columns
N_LON = 16


@pytest.fixture(scope="module")
def cs_grid():
    return create_cubed_sphere(N_CS)


@pytest.fixture(scope="module")
def ll_grid():
    return create_latlon_grid(N_LAT, N_LON)


@pytest.fixture(scope="module")
def sc_grid():
    return SingleColumnGrid(
        lat=jnp.array([0.5]),
        lon=jnp.array([1.0]),
    )


def _make_sigma(nlev):
    """Simple sigma coordinate NamedTuple-like object for testing."""
    class _Sigma:
        def __init__(self, nlev):
            self.sigma_full = jnp.linspace(0.1, 0.95, nlev)
            self.sigma_half = jnp.linspace(0.05, 1.0, nlev + 1)
            self.dsigma = jnp.diff(self.sigma_half)
    return _Sigma(nlev)


def _make_config(**overrides):
    """Build a minimal ExperimentConfig for physics pipeline tests."""
    from legoesm.driver.config import ExperimentConfig
    defaults = dict(
        radiation="gray",
        microphysics="none",
        diurnal_cycle=False,
    )
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


# ===================================================================
# Test: ColumnAdapter
# ===================================================================

class TestColumnAdapterCubedSphere:
    """Adapter round-trips for cubed-sphere grid."""

    def test_make_adapter(self, cs_grid):
        ad = make_adapter(cs_grid)
        assert ad.ncol == 6 * N_CS * N_CS
        assert ad.shape_2d == (6, N_CS, N_CS)

    def test_flatten_unflatten_3d(self, cs_grid):
        ad = make_adapter(cs_grid)
        field = jnp.ones((6, N_CS, N_CS, NLEV))
        col = ad.flatten_3d(field)
        assert col.shape == (ad.ncol, NLEV)
        restored = ad.unflatten_3d(col)
        npt.assert_array_equal(field, restored)

    def test_flatten_unflatten_2d(self, cs_grid):
        ad = make_adapter(cs_grid)
        field = jnp.ones((6, N_CS, N_CS))
        col = ad.flatten_2d(field)
        assert col.shape == (ad.ncol,)
        restored = ad.unflatten_2d(col)
        npt.assert_array_equal(field, restored)

    def test_values_preserved(self, cs_grid):
        ad = make_adapter(cs_grid)
        rng = np.random.default_rng(42)
        data = jnp.array(rng.standard_normal((6, N_CS, N_CS, NLEV)))
        col = ad.flatten_3d(data)
        restored = ad.unflatten_3d(col)
        npt.assert_allclose(data, restored, atol=0.0)


class TestColumnAdapterLatLon:
    """Adapter round-trips for lat-lon grid."""

    def test_make_adapter(self, ll_grid):
        ad = make_adapter(ll_grid)
        assert ad.ncol == N_LAT * N_LON
        assert ad.shape_2d == (N_LAT, N_LON)

    def test_flatten_unflatten_3d(self, ll_grid):
        ad = make_adapter(ll_grid)
        field = jnp.ones((N_LAT, N_LON, NLEV))
        col = ad.flatten_3d(field)
        assert col.shape == (ad.ncol, NLEV)
        restored = ad.unflatten_3d(col)
        npt.assert_array_equal(field, restored)

    def test_flatten_unflatten_2d(self, ll_grid):
        ad = make_adapter(ll_grid)
        field = jnp.ones((N_LAT, N_LON))
        col = ad.flatten_2d(field)
        assert col.shape == (ad.ncol,)
        restored = ad.unflatten_2d(col)
        npt.assert_array_equal(field, restored)


class TestColumnAdapterSingleColumn:
    """Adapter round-trips for single-column grid."""

    def test_make_adapter(self, sc_grid):
        ad = make_adapter(sc_grid)
        assert ad.ncol == 1
        assert ad.shape_2d == (1,)

    def test_flatten_unflatten_3d(self, sc_grid):
        ad = make_adapter(sc_grid)
        field = jnp.ones((1, NLEV))
        col = ad.flatten_3d(field)
        assert col.shape == (1, NLEV)
        restored = ad.unflatten_3d(col)
        npt.assert_array_equal(field, restored)

    def test_flatten_unflatten_2d(self, sc_grid):
        ad = make_adapter(sc_grid)
        field = jnp.ones((1,))
        col = ad.flatten_2d(field)
        assert col.shape == (1,)
        restored = ad.unflatten_2d(col)
        npt.assert_array_equal(field, restored)


# ===================================================================
# Test: Kernel registries
# ===================================================================

class TestKernelRegistries:
    """Registry lookup and resolution."""

    def test_radiation_registry_has_gray(self):
        assert "gray" in RADIATION_REGISTRY

    def test_radiation_registry_has_rrtmgp(self):
        assert "rrtmgp" in RADIATION_REGISTRY

    def test_resolve_gray(self):
        fn = resolve_kernel(RADIATION_REGISTRY, "gray")
        assert callable(fn)
        assert fn.__name__ == "gray_radiation"

    def test_convection_registry_has_all(self):
        expected = {"sbm", "dca", "kuo", "mass_flux", "edmf"}
        assert expected == set(CONVECTION_REGISTRY.keys())

    def test_resolve_sbm(self):
        fn = resolve_kernel(CONVECTION_REGISTRY, "sbm")
        assert callable(fn)
        assert fn.__name__ == "sbm_convection"

    def test_microphysics_registry_has_all(self):
        expected = {"kessler", "sundqvist", "seifert_beheng", "morrison", "thompson"}
        assert expected == set(MICROPHYSICS_REGISTRY.keys())

    def test_resolve_kessler(self):
        fn = resolve_kernel(MICROPHYSICS_REGISTRY, "kessler")
        assert callable(fn)
        assert fn.__name__ == "kessler_microphysics"

    def test_unknown_scheme_raises(self):
        with pytest.raises(KeyError, match="nonexistent"):
            resolve_kernel(RADIATION_REGISTRY, "nonexistent")

    def test_available_schemes(self):
        schemes = available_schemes(CONVECTION_REGISTRY)
        assert schemes == sorted(CONVECTION_REGISTRY.keys())


# ===================================================================
# Test: build_physics_pipeline
# ===================================================================

class TestBuildPhysicsPipelineCubedSphere:
    """Physics pipeline constructed from a cubed-sphere grid."""

    def test_builds_with_adapter(self, cs_grid):
        sigma = _make_sigma(NLEV)
        config = _make_config()
        pipeline = build_physics_pipeline(cs_grid, sigma, config)
        assert isinstance(pipeline, PhysicsPipeline)
        assert pipeline.adapter.ncol == 6 * N_CS * N_CS
        assert pipeline.adapter.shape_2d == (6, N_CS, N_CS)

    def test_convection_fn_resolved(self, cs_grid):
        sigma = _make_sigma(NLEV)
        config = _make_config()
        pipeline = build_physics_pipeline(cs_grid, sigma, config)
        assert callable(pipeline.convection_fn)

    def test_no_microphysics_when_none(self, cs_grid):
        sigma = _make_sigma(NLEV)
        config = _make_config(microphysics="none")
        pipeline = build_physics_pipeline(cs_grid, sigma, config)
        assert pipeline.micro_fn is None

    def test_kessler_microphysics(self, cs_grid):
        sigma = _make_sigma(NLEV)
        config = _make_config(microphysics="kessler")
        pipeline = build_physics_pipeline(cs_grid, sigma, config)
        assert pipeline.micro_fn is not None
        assert callable(pipeline.micro_fn)


class TestBuildPhysicsPipelineLatLon:
    """Physics pipeline constructed from a lat-lon grid."""

    def test_builds_with_adapter(self, ll_grid):
        sigma = _make_sigma(NLEV)
        config = _make_config()
        pipeline = build_physics_pipeline(ll_grid, sigma, config)
        assert isinstance(pipeline, PhysicsPipeline)
        assert pipeline.adapter.ncol == N_LAT * N_LON
        assert pipeline.adapter.shape_2d == (N_LAT, N_LON)


class TestBuildPhysicsPipelineSingleColumn:
    """Physics pipeline constructed from a single-column grid."""

    def test_builds_with_adapter(self, sc_grid):
        sigma = _make_sigma(NLEV)
        config = _make_config()
        pipeline = build_physics_pipeline(sc_grid, sigma, config)
        assert isinstance(pipeline, PhysicsPipeline)
        assert pipeline.adapter.ncol == 1
        assert pipeline.adapter.shape_2d == (1,)


# ===================================================================
# Test: physics_step_no_rad through all three grids
# ===================================================================

def _run_physics_step(grid, nlev=NLEV, dt=600.0):
    """Run one physics step on the given grid and return PhysicsOutput."""
    sigma = _make_sigma(nlev)
    config = _make_config()
    pipeline = build_physics_pipeline(grid, sigma, config)
    ad = pipeline.adapter
    shape_2d = ad.shape_2d
    shape_3d = (*shape_2d, nlev)

    rng = np.random.default_rng(123)

    T = jnp.array(280.0 + 10.0 * rng.standard_normal(shape_3d), dtype=jnp.float32)
    p_s = jnp.full(shape_2d, 1e5, dtype=jnp.float32)
    q_v = jnp.full(shape_3d, 0.005, dtype=jnp.float32)
    q_c = jnp.zeros(shape_3d, dtype=jnp.float32)
    q_r = jnp.zeros(shape_3d, dtype=jnp.float32)
    u = jnp.full(shape_3d, 5.0, dtype=jnp.float32)
    v = jnp.full(shape_3d, 0.0, dtype=jnp.float32)
    sst = jnp.full(shape_2d, 300.0, dtype=jnp.float32)
    sic = jnp.zeros(shape_2d, dtype=jnp.float32)
    lat = jnp.full(shape_2d, 0.5, dtype=jnp.float32)

    # Held radiation (zeros)
    dT_dt_rad = jnp.zeros(shape_3d, dtype=jnp.float32)
    sw_net_sfc = jnp.zeros(shape_2d, dtype=jnp.float32)
    lw_net_sfc = jnp.zeros(shape_2d, dtype=jnp.float32)
    sw_up_toa = jnp.zeros(shape_2d, dtype=jnp.float32)
    lw_up_toa = jnp.zeros(shape_2d, dtype=jnp.float32)
    sw_down_toa = jnp.zeros(shape_2d, dtype=jnp.float32)

    out = pipeline.physics_step_no_rad(
        T, p_s, q_v, q_c, q_r, u, v, sst, sic, lat, dt,
        dT_dt_rad, sw_net_sfc, lw_net_sfc,
        sw_up_toa, lw_up_toa, sw_down_toa,
    )
    return out, pipeline


class TestPhysicsStepCubedSphere:
    """Physics step through cubed-sphere grid."""

    def test_output_shapes(self, cs_grid):
        out, pipeline = _run_physics_step(cs_grid)
        shape_3d = (6, N_CS, N_CS, NLEV)
        shape_2d = (6, N_CS, N_CS)
        assert out.dT_dt.shape == shape_3d
        assert out.dq_v_dt.shape == shape_3d
        assert out.dq_c_dt.shape == shape_3d
        assert out.dq_r_dt.shape == shape_3d
        assert out.precip.shape == shape_2d

    def test_finite_values(self, cs_grid):
        out, _ = _run_physics_step(cs_grid)
        assert jnp.all(jnp.isfinite(out.dT_dt))
        assert jnp.all(jnp.isfinite(out.dq_v_dt))
        assert jnp.all(jnp.isfinite(out.precip))


class TestPhysicsStepLatLon:
    """Physics step through lat-lon grid."""

    def test_output_shapes(self, ll_grid):
        out, pipeline = _run_physics_step(ll_grid)
        shape_3d = (N_LAT, N_LON, NLEV)
        shape_2d = (N_LAT, N_LON)
        assert out.dT_dt.shape == shape_3d
        assert out.dq_v_dt.shape == shape_3d
        assert out.precip.shape == shape_2d

    def test_finite_values(self, ll_grid):
        out, _ = _run_physics_step(ll_grid)
        assert jnp.all(jnp.isfinite(out.dT_dt))
        assert jnp.all(jnp.isfinite(out.precip))


class TestPhysicsStepSingleColumn:
    """Physics step through single-column grid."""

    def test_output_shapes(self, sc_grid):
        out, pipeline = _run_physics_step(sc_grid)
        assert out.dT_dt.shape == (1, NLEV)
        assert out.dq_v_dt.shape == (1, NLEV)
        assert out.precip.shape == (1,)

    def test_finite_values(self, sc_grid):
        out, _ = _run_physics_step(sc_grid)
        assert jnp.all(jnp.isfinite(out.dT_dt))
        assert jnp.all(jnp.isfinite(out.precip))


# ===================================================================
# Test: Column-level physics equivalence across grids
# ===================================================================

class TestCrossGridEquivalence:
    """Verify that the same column data produces identical results
    regardless of grid topology.

    Strategy: construct fields on each grid such that every column
    has the same physical state, then verify per-column outputs match.
    """

    def test_uniform_state_gives_same_column_tendencies(self):
        """All grids with identical per-column state → same tendencies."""
        nlev = NLEV
        dt = 600.0
        sigma = _make_sigma(nlev)
        config = _make_config()

        # Canonical column state
        T_val = jnp.array(
            [200.0, 240.0, 260.0, 280.0, 290.0], dtype=jnp.float32
        )
        q_v_val = jnp.full(nlev, 0.005, dtype=jnp.float32)
        p_s_val = 1e5

        results = {}

        for grid_name, grid in [
            ("cs", create_cubed_sphere(N_CS)),
            ("ll", create_latlon_grid(N_LAT, N_LON)),
            ("sc", SingleColumnGrid(lat=jnp.array([0.5]), lon=jnp.array([1.0]))),
        ]:
            pipeline = build_physics_pipeline(grid, sigma, config)
            ad = pipeline.adapter
            shape_2d = ad.shape_2d
            shape_3d = (*shape_2d, nlev)

            T = jnp.broadcast_to(T_val, shape_3d).copy()
            p_s = jnp.full(shape_2d, p_s_val, dtype=jnp.float32)
            q_v = jnp.broadcast_to(q_v_val, shape_3d).copy()
            q_c = jnp.zeros(shape_3d, dtype=jnp.float32)
            q_r = jnp.zeros(shape_3d, dtype=jnp.float32)
            u = jnp.full(shape_3d, 5.0, dtype=jnp.float32)
            v = jnp.zeros(shape_3d, dtype=jnp.float32)
            sst = jnp.full(shape_2d, 300.0, dtype=jnp.float32)
            sic = jnp.zeros(shape_2d, dtype=jnp.float32)
            lat = jnp.full(shape_2d, 0.5, dtype=jnp.float32)

            held_zero_3d = jnp.zeros(shape_3d, dtype=jnp.float32)
            held_zero_2d = jnp.zeros(shape_2d, dtype=jnp.float32)

            out = pipeline.physics_step_no_rad(
                T, p_s, q_v, q_c, q_r, u, v, sst, sic, lat, dt,
                held_zero_3d, held_zero_2d, held_zero_2d,
                held_zero_2d, held_zero_2d, held_zero_2d,
            )

            # Extract first column tendencies
            dT_col = ad.flatten_3d(out.dT_dt)[0]
            dqv_col = ad.flatten_3d(out.dq_v_dt)[0]
            precip_col = ad.flatten_2d(out.precip)[0]

            results[grid_name] = (dT_col, dqv_col, precip_col)

        # Compare cubed-sphere vs lat-lon
        npt.assert_allclose(
            results["cs"][0], results["ll"][0], atol=1e-5,
            err_msg="dT_dt mismatch: cubed-sphere vs lat-lon",
        )
        npt.assert_allclose(
            results["cs"][1], results["ll"][1], atol=1e-5,
            err_msg="dq_v_dt mismatch: cubed-sphere vs lat-lon",
        )
        npt.assert_allclose(
            results["cs"][2], results["ll"][2], atol=1e-5,
            err_msg="precip mismatch: cubed-sphere vs lat-lon",
        )

        # Compare cubed-sphere vs single-column
        npt.assert_allclose(
            results["cs"][0], results["sc"][0], atol=1e-5,
            err_msg="dT_dt mismatch: cubed-sphere vs single-column",
        )
        npt.assert_allclose(
            results["cs"][1], results["sc"][1], atol=1e-5,
            err_msg="dq_v_dt mismatch: cubed-sphere vs single-column",
        )


# ===================================================================
# Test: build_step_unified (JIT-compiled path)
# ===================================================================

class TestStepUnified:
    """The JIT-compiled unified step works through an adapter."""

    def test_unified_step_cubed_sphere(self, cs_grid):
        sigma = _make_sigma(NLEV)
        config = _make_config()
        pipeline = build_physics_pipeline(cs_grid, sigma, config)
        step_fn = pipeline.build_step_unified()

        ad = pipeline.adapter
        shape_2d = ad.shape_2d
        shape_3d = (*shape_2d, NLEV)

        T = jnp.full(shape_3d, 280.0, dtype=jnp.float32)
        p_s = jnp.full(shape_2d, 1e5, dtype=jnp.float32)
        q_v = jnp.full(shape_3d, 0.005, dtype=jnp.float32)
        q_c = jnp.zeros(shape_3d, dtype=jnp.float32)
        q_r = jnp.zeros(shape_3d, dtype=jnp.float32)
        u = jnp.full(shape_3d, 5.0, dtype=jnp.float32)
        v = jnp.zeros(shape_3d, dtype=jnp.float32)
        sst = jnp.full(shape_2d, 300.0, dtype=jnp.float32)
        sic = jnp.zeros(shape_2d, dtype=jnp.float32)
        lat = jnp.full(shape_2d, 0.5, dtype=jnp.float32)
        lon = jnp.full(shape_2d, 1.0, dtype=jnp.float32)

        held_3d = jnp.zeros(shape_3d, dtype=jnp.float32)
        held_2d = jnp.zeros(shape_2d, dtype=jnp.float32)

        solar_w = jnp.array([], dtype=jnp.float32)
        o3 = jnp.zeros((ad.ncol, NLEV), dtype=jnp.float32)
        aerosol = jnp.zeros((ad.ncol, NLEV), dtype=jnp.float32)

        phys_out, new_held = step_fn(
            jnp.bool_(True),
            T, p_s, q_v, q_c, q_r, u, v, sst, sic, lat, lon,
            100.0, 43200.0, 600.0,
            solar_w, 1361.0,
            o3, aerosol,
            held_3d, held_2d, held_2d,
            held_2d, held_2d, held_2d,
        )

        assert isinstance(phys_out, PhysicsOutput)
        assert phys_out.dT_dt.shape == shape_3d
        assert jnp.all(jnp.isfinite(phys_out.dT_dt))

    def test_unified_step_single_column(self, sc_grid):
        sigma = _make_sigma(NLEV)
        config = _make_config()
        pipeline = build_physics_pipeline(sc_grid, sigma, config)
        step_fn = pipeline.build_step_unified()

        ad = pipeline.adapter
        shape_2d = ad.shape_2d
        shape_3d = (*shape_2d, NLEV)

        T = jnp.full(shape_3d, 280.0, dtype=jnp.float32)
        p_s = jnp.full(shape_2d, 1e5, dtype=jnp.float32)
        q_v = jnp.full(shape_3d, 0.005, dtype=jnp.float32)
        q_c = jnp.zeros(shape_3d, dtype=jnp.float32)
        q_r = jnp.zeros(shape_3d, dtype=jnp.float32)
        u = jnp.full(shape_3d, 5.0, dtype=jnp.float32)
        v = jnp.zeros(shape_3d, dtype=jnp.float32)
        sst = jnp.full(shape_2d, 300.0, dtype=jnp.float32)
        sic = jnp.zeros(shape_2d, dtype=jnp.float32)
        lat = jnp.full(shape_2d, 0.5, dtype=jnp.float32)
        lon = jnp.full(shape_2d, 1.0, dtype=jnp.float32)

        held_3d = jnp.zeros(shape_3d, dtype=jnp.float32)
        held_2d = jnp.zeros(shape_2d, dtype=jnp.float32)

        solar_w = jnp.array([], dtype=jnp.float32)
        o3 = jnp.zeros((1, NLEV), dtype=jnp.float32)
        aerosol = jnp.zeros((1, NLEV), dtype=jnp.float32)

        phys_out, new_held = step_fn(
            jnp.bool_(True),
            T, p_s, q_v, q_c, q_r, u, v, sst, sic, lat, lon,
            100.0, 43200.0, 600.0,
            solar_w, 1361.0,
            o3, aerosol,
            held_3d, held_2d, held_2d,
            held_2d, held_2d, held_2d,
        )

        assert phys_out.dT_dt.shape == shape_3d
        assert jnp.all(jnp.isfinite(phys_out.dT_dt))


# ===================================================================
# Test: No jnp.prod shape computation in the refactored pipeline
# ===================================================================

class TestNoHardCodedShapes:
    """Verify that hard-coded shape assumptions have been removed."""

    def test_no_jnp_prod_in_source(self):
        """The refactored physics_pipeline.py should not use jnp.prod
        for computing ncol."""
        import inspect
        from legoesm.driver import physics_pipeline as mod
        source = inspect.getsource(mod)
        # Old pattern: ncol = int(jnp.prod(jnp.array(shape_2d)))
        assert "jnp.prod" not in source, (
            "physics_pipeline.py still contains jnp.prod — "
            "ncol should come from the ColumnAdapter"
        )

    def test_pipeline_uses_adapter_ncol(self, cs_grid):
        sigma = _make_sigma(NLEV)
        config = _make_config()
        pipeline = build_physics_pipeline(cs_grid, sigma, config)
        # The adapter's ncol should be the authoritative column count
        assert pipeline.adapter.ncol == cs_grid.grid_n_columns
