"""Precision mode matrix tests for FV model families.

Verifies that apply_precision() correctly activates each mode, grid builders
honor the policy, and FV dycores run one step in fp32, fp64, and
mixed and mixed_fp64_storage modes with correct stored-state dtypes.
(The #1665 interim refusal of the two mixed modes was lifted by #1675.)
"""

import pytest
import jax
import jax.numpy as jnp

# Force CPU + x64 for the full matrix.
jax.config.update("jax_enable_x64", True)

from legoesm.runtime.precision import apply_precision, get_policy
from legoesm.core.precision import set_policy, PrecisionPolicy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_policy():
    """Reset precision policy to fp32 before each test."""
    set_policy(PrecisionPolicy.fp32())
    yield
    set_policy(PrecisionPolicy.fp32())


# ===========================================================================
# Part 1: apply_precision activation
# ===========================================================================

class TestApplyPrecision:

    def test_fp32(self):
        p = apply_precision("fp32")
        assert p.storage == jnp.float32
        assert p.compute == jnp.float32

    def test_fp64(self):
        p = apply_precision("fp64")
        assert p.storage == jnp.float64
        assert p.compute == jnp.float64

    def test_mixed(self):
        # #1675 lifted the #1665 interim refusal.
        p = apply_precision("mixed")
        assert p.storage == jnp.float32
        assert p.compute == jnp.float32
        assert p.accumulate == jnp.float64
        assert p.control == jnp.float64

    def test_mixed_fp64_storage(self):
        p = apply_precision("mixed_fp64_storage")
        assert p.storage == jnp.float64
        assert p.compute == jnp.float32
        assert p.accumulate == jnp.float64


# ===========================================================================
# Part 2: Grid builders honor precision policy
# ===========================================================================

class TestGridPrecision:

    def test_cubed_sphere_fp32(self):
        apply_precision("fp32")
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        g = create_cubed_sphere(8)
        assert g.area.dtype == jnp.float32

    def test_cubed_sphere_fp64(self):
        apply_precision("fp64")
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        g = create_cubed_sphere(8)
        assert g.area.dtype == jnp.float64

    def test_cubed_sphere_mixed_fp64_storage(self):
        apply_precision("mixed_fp64_storage")
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        g = create_cubed_sphere(8)
        assert g.area.dtype == jnp.float64  # storage role


    def test_latlon_fp32(self):
        apply_precision("fp32")
        from legoesm.grids.latlon import create_latlon_grid
        g = create_latlon_grid(16)
        assert g.lat.dtype == jnp.float32

    def test_latlon_fp64(self):
        apply_precision("fp64")
        from legoesm.grids.latlon import create_latlon_grid
        g = create_latlon_grid(16)
        assert g.lat.dtype == jnp.float64

    def test_latlon_mixed_fp64_storage(self):
        apply_precision("mixed_fp64_storage")
        from legoesm.grids.latlon import create_latlon_grid
        g = create_latlon_grid(16)
        assert g.lat.dtype == jnp.float64  # storage role


    def test_cdgrid_inherits_from_parent(self):
        apply_precision("fp64")
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
        g = create_cubed_sphere(8)
        cdg = create_cubed_sphere_cdgrid(g)
        assert cdg.grad_c00.dtype == jnp.float64
        assert cdg.lon_corner.dtype == jnp.float64

    def test_cdgrid_explicit_metric_dtype(self):
        apply_precision("fp32")
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
        g = create_cubed_sphere(8)
        cdg = create_cubed_sphere_cdgrid(g, metric_dtype=jnp.float64)
        # metric_dtype overrides precision-critical fields
        assert cdg.grad_c00.dtype == jnp.float64
        # base positions follow parent
        assert cdg.lon_corner.dtype == jnp.float32

    def test_gaussian_is_always_float64(self):
        """Gaussian/spectral grids are float64 regardless of policy."""
        apply_precision("fp32")
        # Gaussian requires x64 to be enabled
        from legoesm.grids.gaussian import create_gaussian_grid
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g = create_gaussian_grid(21, 64)
        assert g.lat.dtype == jnp.float64


# ===========================================================================
# Part 3: FV shallow water one-step smoke tests
# ===========================================================================

class TestSWCubedSpherePrecision:
    """CDGrid shallow water one-step in each precision mode."""

    @staticmethod
    def _run_one_step(mode):
        apply_precision(mode)
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
        from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
            CDGridShallowWaterConfig, CDGridShallowWaterModel,
            CDGridShallowWaterState,
        )
        grid = create_cubed_sphere(8)
        cdgrid = create_cubed_sphere_cdgrid(grid)
        n = grid.n
        policy = get_policy()
        sd = policy.storage
        h = jnp.ones((6, n, n), dtype=sd) * 1000.0
        u_d = jnp.zeros((6, n+1, n+1), dtype=sd)
        v_d = jnp.zeros((6, n+1, n+1), dtype=sd)
        h_s = jnp.zeros((6, n, n), dtype=sd)
        state = CDGridShallowWaterState(h=h, u_d=u_d, v_d=v_d, h_s=h_s)
        config = CDGridShallowWaterConfig(use_conservation_fixer=False)
        model = CDGridShallowWaterModel(grid, config)
        state_new = model.step(state, 60.0)
        return state_new, policy

    def test_fp32(self):
        s, p = self._run_one_step("fp32")
        assert s.h.dtype == jnp.float32
        assert jnp.all(jnp.isfinite(s.h))

    def test_fp64(self):
        s, p = self._run_one_step("fp64")
        assert s.h.dtype == jnp.float64
        assert jnp.all(jnp.isfinite(s.h))

    def test_mixed_fp64_storage(self):
        s, p = self._run_one_step("mixed_fp64_storage")
        assert s.h.dtype == jnp.float64  # storage role is f64 in this mode
        assert jnp.all(jnp.isfinite(s.h))


class TestSWMPASPrecision:
    """MPAS shallow water one-step in each precision mode."""

    @staticmethod
    def _run_one_step(mode):
        apply_precision(mode)
        from legoesm.grids.voronoi import create_voronoi_mesh
        from legoesm.atmosphere.dynamics.gcm.shallow_water_mpas import (
            MPASShallowWaterModel, MPASShallowWaterConfig,
        )
        from tests.atmosphere.shallow_water.test_cases.williamson_mpas import (
            williamson_test2_mpas,
        )
        mesh = create_voronoi_mesh(3)
        config = MPASShallowWaterConfig()
        model = MPASShallowWaterModel(mesh, config)
        sw = williamson_test2_mpas(mesh)
        state_new = model.step(sw, 60.0)
        return state_new, get_policy()

    def test_fp32(self):
        # This module forces jax_enable_x64=True for the full matrix, but x64
        # is a global, irreversible switch — so the in-process fp32 step still
        # sees an x64-capable backend and _fix_mass_mpas (correctly) promotes h
        # to float64 to keep mass conserved to ~1e-12 (the fp64 mass correction
        # is load-bearing; truncating it back to fp32 breaks
        # test_sw_mass_conservation_anchored).  The honest "fp32 storage" check
        # therefore needs a backend where x64 was NEVER enabled, which only a
        # fresh subprocess can provide.
        import subprocess, sys, textwrap, os
        code = textwrap.dedent("""
            import jax, jax.numpy as jnp
            from legoesm.runtime.precision import apply_precision
            apply_precision("fp32")
            from legoesm.grids.voronoi import create_voronoi_mesh
            from legoesm.atmosphere.dynamics.gcm.shallow_water_mpas import (
                MPASShallowWaterModel, MPASShallowWaterConfig)
            from tests.atmosphere.shallow_water.test_cases.williamson_mpas import (
                williamson_test2_mpas)
            mesh = create_voronoi_mesh(3)
            model = MPASShallowWaterModel(mesh, MPASShallowWaterConfig())
            s = model.step(williamson_test2_mpas(mesh), 60.0)
            assert s.h.data.dtype == jnp.float32, s.h.data.dtype
            assert bool(jnp.all(jnp.isfinite(s.h.data)))
            print("OK")
        """)
        env = os.environ.copy()
        env.pop("JAX_ENABLE_X64", None)  # ensure x64 is OFF in the child
        r = subprocess.run([sys.executable, "-c", code],
                           capture_output=True, text=True, env=env, timeout=300)
        assert r.returncode == 0 and "OK" in r.stdout, (
            f"fp32 MPAS step did not keep float32 storage.\n"
            f"stdout: {r.stdout}\nstderr: {r.stderr}")

    def test_fp64(self):
        s, p = self._run_one_step("fp64")
        assert s.h.data.dtype == jnp.float64
        assert jnp.all(jnp.isfinite(s.h.data))

    def test_mixed_fp64_storage(self):
        s, p = self._run_one_step("mixed_fp64_storage")
        assert s.h.data.dtype == jnp.float64  # storage role is f64
        assert jnp.all(jnp.isfinite(s.h.data))
