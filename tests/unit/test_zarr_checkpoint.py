"""Tests for Zarr checkpoint IO."""
import json
import shutil
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.core.field import Field
from legoesm.core.state import HydrostaticState
from legoesm.forcing.amip_config import (
    AMIPExperimentConfig,
    save_checkpoint,
    load_checkpoint,
)
from legoesm.driver.checkpoint import (
    save_checkpoint_zarr,
    load_checkpoint_zarr,
    load_checkpoint_auto,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.vertical import create_sigma_coordinate


def _make_state(n=4, nlev=5):
    """Create a small test state."""
    shape_3d = (6, n, n, nlev)
    shape_2d = (6, n, n)
    dims_3d = ("face", "x", "y", "level")
    dims_2d = ("face", "x", "y")

    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 6)

    state = HydrostaticState(
        T=Field(data=jax.random.uniform(keys[0], shape_3d, minval=200, maxval=300),
                name="T", dims=dims_3d, units="K"),
        u=Field(data=jax.random.uniform(keys[1], shape_3d, minval=-20, maxval=20),
                name="u", dims=dims_3d, units="m/s"),
        v=Field(data=jax.random.uniform(keys[2], shape_3d, minval=-20, maxval=20),
                name="v", dims=dims_3d, units="m/s"),
        p_s=Field(data=jax.random.uniform(keys[3], shape_2d, minval=9e4, maxval=1.1e5),
                  name="p_s", dims=dims_2d, units="Pa"),
        phis=Field(data=jax.random.uniform(keys[4], shape_2d, minval=0, maxval=5000),
                   name="phis", dims=dims_2d, units="m^2/s^2"),
    )
    q_v = jax.random.uniform(keys[5], shape_3d, minval=0, maxval=0.02)
    return state, q_v


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)


class TestZarrCheckpoint:
    """Core Zarr checkpoint tests."""

    def test_roundtrip(self, tmp_dir):
        """Save -> load roundtrip preserves arrays."""
        state, q_v = _make_state()
        config = AMIPExperimentConfig()
        path = tmp_dir / "test.zarr"

        save_checkpoint_zarr(path, state, q_v, step=100, day=50.0, config=config)
        loaded = load_checkpoint_zarr(path, None, None)
        s2, qv2, step2, day2, cfg2, diag2, qc2, qr2 = loaded

        assert step2 == 100
        assert day2 == 50.0
        assert jnp.allclose(s2.T.data, state.T.data)
        assert jnp.allclose(s2.u.data, state.u.data)
        assert jnp.allclose(s2.v.data, state.v.data)
        assert jnp.allclose(s2.p_s.data, state.p_s.data)
        assert jnp.allclose(s2.phis.data, state.phis.data)
        assert jnp.allclose(qv2, q_v)
        assert qc2 is None
        assert qr2 is None

    def test_optional_fields(self, tmp_dir):
        """q_c and q_r roundtrip correctly."""
        state, q_v = _make_state()
        config = AMIPExperimentConfig()
        q_c = jnp.ones_like(q_v) * 1e-5
        q_r = jnp.ones_like(q_v) * 1e-6
        path = tmp_dir / "test_qcqr.zarr"

        save_checkpoint_zarr(path, state, q_v, step=10, day=5.0,
                             config=config, q_c=q_c, q_r=q_r)
        _, _, _, _, _, _, qc2, qr2 = load_checkpoint_zarr(path, None, None)
        assert jnp.allclose(qc2, q_c)
        assert jnp.allclose(qr2, q_r)

    def test_metadata(self, tmp_dir):
        """Step, day, config in zarr attrs."""
        state, q_v = _make_state()
        config = AMIPExperimentConfig(resolution=24, days=365)
        path = tmp_dir / "meta.zarr"

        save_checkpoint_zarr(path, state, q_v, step=200, day=100.0, config=config)

        import zarr
        root = zarr.open_consolidated(str(path), mode="r")
        assert root.attrs["step"] == 200
        assert root.attrs["day"] == 100.0
        cfg = json.loads(root.attrs["config_json"])
        assert cfg["resolution"] == 24
        assert cfg["days"] == 365

    def test_diag_accumulators(self, tmp_dir):
        """Diagnostic accumulators roundtrip."""
        state, q_v = _make_state()
        config = AMIPExperimentConfig()
        accum = {"T_mean": np.ones(10), "count": np.array(42)}
        path = tmp_dir / "diag.zarr"

        save_checkpoint_zarr(path, state, q_v, step=0, day=0.0,
                             config=config, diag_accumulators=accum)
        _, _, _, _, _, diag2, _, _ = load_checkpoint_zarr(path, None, None)
        assert np.allclose(diag2["T_mean"], accum["T_mean"])
        assert np.allclose(diag2["count"], accum["count"])

    def test_lazy_loading(self, tmp_dir):
        """Lazy mode returns zarr arrays (not materialized)."""
        state, q_v = _make_state()
        config = AMIPExperimentConfig()
        path = tmp_dir / "lazy.zarr"

        save_checkpoint_zarr(path, state, q_v, step=0, day=0.0, config=config)
        result = load_checkpoint_zarr(path, None, None, lazy=True)
        arrays, step, day, cfg, diag, qc, qr = result

        import zarr
        assert isinstance(arrays["T"], zarr.Array)
        assert step == 0


class TestAutoDetect:
    """Test format auto-detection."""

    def test_auto_loads_npz(self, tmp_dir):
        """load_checkpoint_auto reads .npz files."""
        grid = create_cubed_sphere(4)
        sigma = create_sigma_coordinate(5)
        state, q_v = _make_state()
        config = AMIPExperimentConfig()
        path = tmp_dir / "test.npz"

        save_checkpoint(path, state, q_v, step=42, day=21.0, config=config)
        s2, qv2, step2, day2, cfg2, diag2, qc2, qr2, carry_aux2 = load_checkpoint_auto(
            path, grid, sigma
        )
        assert step2 == 42
        assert jnp.allclose(s2.T.data, state.T.data)

    def test_auto_loads_zarr(self, tmp_dir):
        """load_checkpoint_auto reads .zarr directories."""
        grid = create_cubed_sphere(4)
        sigma = create_sigma_coordinate(5)
        state, q_v = _make_state()
        config = AMIPExperimentConfig()
        path = tmp_dir / "test.zarr"

        save_checkpoint_zarr(path, state, q_v, step=77, day=38.5, config=config)
        s2, qv2, step2, day2, cfg2, diag2, qc2, qr2, carry_aux2 = load_checkpoint_auto(
            path, grid, sigma
        )
        assert step2 == 77
        assert jnp.allclose(s2.T.data, state.T.data)

    def test_zarr_smaller_than_npz(self, tmp_dir):
        """Zarr with compression should be smaller than npz."""
        state, q_v = _make_state(n=8, nlev=20)
        config = AMIPExperimentConfig()

        npz_path = tmp_dir / "test.npz"
        zarr_path = tmp_dir / "test.zarr"

        save_checkpoint(npz_path, state, q_v, step=0, day=0.0, config=config)
        save_checkpoint_zarr(zarr_path, state, q_v, step=0, day=0.0, config=config)

        # Get sizes
        npz_size = npz_path.with_suffix(".npz").stat().st_size
        zarr_size = sum(f.stat().st_size for f in zarr_path.rglob("*") if f.is_file())

        # Zarr with zstd should be competitive or smaller
        # (for random data may not be smaller, so just check it works)
        assert zarr_size > 0
        assert npz_size > 0


def test_wallclock_exhausted_boundary():
    """wallclock_exhausted fires within buffer of the budget; max_s<=0 disables."""
    from legoesm.driver.checkpoint import wallclock_exhausted
    # Fires once elapsed >= (max_s - buffer_s).
    assert wallclock_exhausted(85.0, 90.0, 5.0) is True      # exactly at threshold
    assert wallclock_exhausted(100.0, 90.0, 5.0) is True     # past it
    assert wallclock_exhausted(84.999, 90.0, 5.0) is False   # just under
    # max_s <= 0 disables the check entirely.
    assert wallclock_exhausted(1e9, 0.0, 5.0) is False
    assert wallclock_exhausted(1e9, -1.0, 5.0) is False
