"""Tests for distributed checkpointing.

All tests run on a single process, simulating rank 0 of 1 rank.
Uses ``tmp_path`` fixture for all file I/O so nothing persists.
"""

import json

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.field import Field
from legoesm.core.state import HydrostaticState
from legoesm.forcing.amip_config import AMIPExperimentConfig
from legoesm.driver.distributed_checkpoint import (
    _rank_filename,
    _state_to_arrays,
    save_checkpoint_distributed,
    load_checkpoint_distributed,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_FACES = 6
N = 4
NLEV = 3


def _make_field_3d(name="test", fill=1.0):
    """Create a 3D field (6, N, N, NLEV)."""
    data = jnp.full((N_FACES, N, N, NLEV), fill, dtype=jnp.float64)
    return Field(data, name=name, dims=("face", "x", "y", "level"), units="K")


def _make_field_2d(name="test_2d", fill=0.0):
    """Create a 2D field (6, N, N)."""
    data = jnp.full((N_FACES, N, N), fill, dtype=jnp.float64)
    return Field(data, name=name, dims=("face", "x", "y"), units="Pa")


def _make_state():
    """Create a minimal HydrostaticState."""
    return HydrostaticState(
        u=_make_field_3d("u", fill=1.0),
        v=_make_field_3d("v", fill=-1.0),
        T=_make_field_3d("T", fill=280.0),
        p_s=_make_field_2d("p_s", fill=101325.0),
        phis=_make_field_2d("phis", fill=0.0),
    )


# ===========================================================================
# TestRankFilename
# ===========================================================================

class TestRankFilename:
    """Tests for the rank filename helper."""

    def test_rank_filename_format(self):
        """Rank filenames are zero-padded to 3 digits."""
        assert _rank_filename(0) == "rank_000.npz"
        assert _rank_filename(3) == "rank_003.npz"
        assert _rank_filename(42) == "rank_042.npz"
        assert _rank_filename(999) == "rank_999.npz"


# ===========================================================================
# TestStateToArrays
# ===========================================================================

class TestStateToArrays:
    """Tests for _state_to_arrays helper."""

    def test_extracts_all_fields(self):
        """All state fields are extracted as numpy arrays."""
        state = _make_state()
        arrays = _state_to_arrays(state)

        assert set(arrays.keys()) == {"T", "u", "v", "p_s", "phis"}
        for key, arr in arrays.items():
            assert isinstance(arr, np.ndarray)

    def test_shapes_match(self):
        """Extracted arrays have the correct shapes."""
        state = _make_state()
        arrays = _state_to_arrays(state)

        assert arrays["T"].shape == (N_FACES, N, N, NLEV)
        assert arrays["u"].shape == (N_FACES, N, N, NLEV)
        assert arrays["p_s"].shape == (N_FACES, N, N)
        assert arrays["phis"].shape == (N_FACES, N, N)


# ===========================================================================
# TestDistributedCheckpoint
# ===========================================================================

class TestDistributedCheckpoint:
    """Tests for save/load distributed checkpoint."""

    def test_save_load_roundtrip(self, tmp_path):
        """Save and load round-trip preserves all values."""
        state = _make_state()
        step = 100
        day = 5.5

        save_checkpoint_distributed(
            tmp_path, state, rank=0, n_ranks=1,
            step=step, day=day,
        )

        arrays, loaded_step, loaded_day, loaded_config, diag_acc = \
            load_checkpoint_distributed(tmp_path, rank=0, n_ranks=1)

        assert loaded_step == step
        assert loaded_day == day
        assert loaded_config is None
        assert diag_acc == {}

        # Verify array values
        np.testing.assert_array_equal(arrays["T"], np.asarray(state.T.data))
        np.testing.assert_array_equal(arrays["u"], np.asarray(state.u.data))
        np.testing.assert_array_equal(arrays["v"], np.asarray(state.v.data))
        np.testing.assert_array_equal(arrays["p_s"], np.asarray(state.p_s.data))
        np.testing.assert_array_equal(arrays["phis"], np.asarray(state.phis.data))

    def test_metadata_json_created(self, tmp_path):
        """metadata.json is created with correct contents by rank 0."""
        state = _make_state()
        save_checkpoint_distributed(
            tmp_path, state, rank=0, n_ranks=4,
            step=200, day=10.0,
        )

        meta_path = tmp_path / "metadata.json"
        assert meta_path.exists()

        with open(meta_path) as f:
            meta = json.load(f)

        assert meta["step"] == 200
        assert meta["day"] == 10.0
        assert meta["n_ranks"] == 4

    def test_non_rank0_no_metadata(self, tmp_path):
        """Non-rank-0 processes do NOT write metadata.json."""
        state = _make_state()
        save_checkpoint_distributed(
            tmp_path, state, rank=1, n_ranks=2,
            step=50, day=2.5,
        )

        meta_path = tmp_path / "metadata.json"
        assert not meta_path.exists()

        # But the rank file should exist
        rank_path = tmp_path / "rank_001.npz"
        assert rank_path.exists()

    def test_load_validates_n_ranks_mismatch(self, tmp_path):
        """Loading with wrong n_ranks raises ValueError."""
        state = _make_state()
        save_checkpoint_distributed(
            tmp_path, state, rank=0, n_ranks=4,
            step=100, day=5.0,
        )

        with pytest.raises(ValueError, match="saved with 4 ranks"):
            load_checkpoint_distributed(tmp_path, rank=0, n_ranks=2)

    def test_save_with_config_serialization(self, tmp_path):
        """Config is serialized and deserialized through checkpoint."""
        state = _make_state()
        config = AMIPExperimentConfig(resolution=32, nlev=40, dt=300.0)

        save_checkpoint_distributed(
            tmp_path, state, rank=0, n_ranks=1,
            step=50, day=2.0, config=config,
        )

        _, _, _, loaded_config, _ = load_checkpoint_distributed(
            tmp_path, rank=0, n_ranks=1
        )

        assert loaded_config is not None
        assert loaded_config.resolution == 32
        assert loaded_config.nlev == 40
        assert loaded_config.dt == 300.0

    def test_save_with_q_v_q_c_q_r(self, tmp_path):
        """Optional moisture arrays are saved and loaded correctly."""
        state = _make_state()
        q_v = jnp.ones((N_FACES, N, N, NLEV)) * 0.01
        q_c = jnp.ones((N_FACES, N, N, NLEV)) * 1e-5
        q_r = jnp.ones((N_FACES, N, N, NLEV)) * 1e-6

        save_checkpoint_distributed(
            tmp_path, state, rank=0, n_ranks=1,
            step=10, day=0.5,
            q_v=q_v, q_c=q_c, q_r=q_r,
        )

        arrays, _, _, _, _ = load_checkpoint_distributed(
            tmp_path, rank=0, n_ranks=1
        )

        assert "q_v" in arrays
        assert "q_c" in arrays
        assert "q_r" in arrays
        np.testing.assert_allclose(arrays["q_v"], np.asarray(q_v), atol=1e-15)
        np.testing.assert_allclose(arrays["q_c"], np.asarray(q_c), atol=1e-15)
        np.testing.assert_allclose(arrays["q_r"], np.asarray(q_r), atol=1e-15)

    def test_save_without_optional_arrays(self, tmp_path):
        """Checkpoint without q_c/q_r loads fine (they are absent)."""
        state = _make_state()

        save_checkpoint_distributed(
            tmp_path, state, rank=0, n_ranks=1,
            step=10, day=0.5,
        )

        arrays, _, _, _, _ = load_checkpoint_distributed(
            tmp_path, rank=0, n_ranks=1
        )

        assert "q_c" not in arrays
        assert "q_r" not in arrays
        assert "q_v" not in arrays

    def test_save_with_diag_accumulators(self, tmp_path):
        """Diagnostic accumulators are saved and loaded."""
        state = _make_state()
        diag_acc = {
            "precip_total": np.array(42.0),
            "T_mean": np.ones((N_FACES, N, N)) * 280.0,
        }

        save_checkpoint_distributed(
            tmp_path, state, rank=0, n_ranks=1,
            step=10, day=0.5,
            diag_accumulators=diag_acc,
        )

        _, _, _, _, loaded_diag = load_checkpoint_distributed(
            tmp_path, rank=0, n_ranks=1
        )

        assert "precip_total" in loaded_diag
        assert "T_mean" in loaded_diag
        np.testing.assert_allclose(
            loaded_diag["precip_total"], 42.0, atol=1e-12
        )

    def test_load_missing_checkpoint_dir(self, tmp_path):
        """Loading from a nonexistent directory raises FileNotFoundError."""
        fake_path = tmp_path / "nonexistent"
        with pytest.raises(FileNotFoundError):
            load_checkpoint_distributed(fake_path, rank=0, n_ranks=1)

    def test_load_missing_rank_file(self, tmp_path):
        """Loading a missing per-rank file raises FileNotFoundError."""
        state = _make_state()
        # Save rank 0 only
        save_checkpoint_distributed(
            tmp_path, state, rank=0, n_ranks=2,
            step=10, day=0.5,
        )

        # Rank 1 file was not saved
        with pytest.raises(FileNotFoundError, match="rank_001"):
            load_checkpoint_distributed(tmp_path, rank=1, n_ranks=2)

    def test_checkpoint_directory_created(self, tmp_path):
        """The checkpoint directory is created if it does not exist."""
        state = _make_state()
        ckpt_dir = tmp_path / "sub" / "deep" / "checkpoint"

        save_checkpoint_distributed(
            ckpt_dir, state, rank=0, n_ranks=1,
            step=1, day=0.1,
        )

        assert ckpt_dir.exists()
        assert (ckpt_dir / "metadata.json").exists()
        assert (ckpt_dir / "rank_000.npz").exists()

    def test_multiple_ranks_independent(self, tmp_path):
        """Each rank writes its own file independently."""
        state = _make_state()

        # Simulate rank 0
        save_checkpoint_distributed(
            tmp_path, state, rank=0, n_ranks=2,
            step=10, day=1.0,
        )
        # Simulate rank 1
        state_r1 = state._replace(
            T=state.T.replace(data=state.T.data + 5.0)
        )
        save_checkpoint_distributed(
            tmp_path, state_r1, rank=1, n_ranks=2,
            step=10, day=1.0,
        )

        # Load each rank
        arr0, _, _, _, _ = load_checkpoint_distributed(
            tmp_path, rank=0, n_ranks=2
        )
        arr1, _, _, _, _ = load_checkpoint_distributed(
            tmp_path, rank=1, n_ranks=2
        )

        # Rank 1 should have different T values
        assert not np.array_equal(arr0["T"], arr1["T"])
        np.testing.assert_allclose(arr1["T"] - arr0["T"], 5.0, atol=1e-12)
