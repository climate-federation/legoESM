"""Tests for the generic state checkpoint system (io/state_checkpoint.py).

Round-trip tests for multiple state types, static field handling,
config hash validation, shape mismatch detection, and precision casting.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from legoesm.core.field import Field
from legoesm.io.state_checkpoint import (
    save_state_checkpoint,
    load_state_checkpoint,
    validate_state_checkpoint,
    _extract_arrays,
    _extract_field_metadata,
    _meta_path,
)


# ---------------------------------------------------------------------------
# Minimal state types for testing (mirrors real legoESM states)
# ---------------------------------------------------------------------------


class _MiniAtmState(NamedTuple):
    """Atmosphere-like: all Field objects."""
    T: Field
    u: Field
    p_s: Field


class _MiniOceanState(NamedTuple):
    """Ocean-like: Fields + static fields."""
    T: Field
    S: Field
    eta: Field
    H_bathy: Field
    land_mask: Field


class _MiniLandState(NamedTuple):
    """Land-like: mix of Field and raw jax.Array."""
    T_soil: Field
    W_bucket: Field
    runoff: jax.Array | None


class _MiniConfig(NamedTuple):
    """Minimal config for hash tests."""
    resolution: int
    dt: float
    scheme: str


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def atm_state():
    return _MiniAtmState(
        T=Field(data=jnp.ones((6, 4, 4, 5)) * 280.0, name="T",
                dims=("face", "x", "y", "level"), units="K"),
        u=Field(data=jnp.ones((6, 4, 4, 5)) * 10.0, name="u",
                dims=("face", "x", "y", "level"), units="m/s"),
        p_s=Field(data=jnp.ones((6, 4, 4)) * 1e5, name="p_s",
                  dims=("face", "x", "y"), units="Pa"),
    )


@pytest.fixture
def ocean_state():
    return _MiniOceanState(
        T=Field(data=jnp.ones((6, 4, 4, 3)) * 15.0, name="T",
                dims=("face", "x", "y", "level"), units="degC"),
        S=Field(data=jnp.ones((6, 4, 4, 3)) * 35.0, name="S",
                dims=("face", "x", "y", "level"), units="PSU"),
        eta=Field(data=jnp.zeros((6, 4, 4)), name="eta",
                  dims=("face", "x", "y"), units="m"),
        H_bathy=Field(data=jnp.ones((6, 4, 4)) * 4000.0, name="H_bathy",
                      dims=("face", "x", "y"), units="m"),
        land_mask=Field(data=jnp.zeros((6, 4, 4)), name="land_mask",
                        dims=("face", "x", "y"), units=""),
    )


@pytest.fixture
def land_state():
    return _MiniLandState(
        T_soil=Field(data=jnp.ones((6, 4, 4)) * 285.0, name="T_soil",
                     dims=("face", "x", "y"), units="K"),
        W_bucket=Field(data=jnp.ones((6, 4, 4)) * 50.0, name="W_bucket",
                       dims=("face", "x", "y"), units="kg/m2"),
        runoff=None,
    )


@pytest.fixture
def config():
    return _MiniConfig(resolution=4, dt=1800.0, scheme="rk4")


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_atmosphere_round_trip(self, tmp_path, atm_state):
        path = tmp_path / "atm_ckpt.npz"
        save_state_checkpoint(path, atm_state, step=100, elapsed_time_s=3600.0, dt=900.0)

        loaded, step, elapsed, meta = load_state_checkpoint(path, atm_state)

        assert step == 100
        assert elapsed == 3600.0
        assert meta["dt"] == 900.0
        for name in atm_state._fields:
            orig = getattr(atm_state, name)
            reloaded = getattr(loaded, name)
            assert isinstance(reloaded, Field)
            npt.assert_array_equal(np.asarray(orig.data), np.asarray(reloaded.data))
            assert reloaded.name == orig.name
            assert reloaded.dims == orig.dims
            assert reloaded.units == orig.units

    def test_ocean_round_trip(self, tmp_path, ocean_state):
        path = tmp_path / "ocean_ckpt.npz"
        save_state_checkpoint(path, ocean_state, step=50, elapsed_time_s=7200.0, dt=600.0)

        loaded, step, elapsed, meta = load_state_checkpoint(path, ocean_state)

        assert step == 50
        for name in ocean_state._fields:
            orig = getattr(ocean_state, name)
            reloaded = getattr(loaded, name)
            npt.assert_array_equal(np.asarray(orig.data), np.asarray(reloaded.data))

    def test_land_round_trip_with_none(self, tmp_path, land_state):
        path = tmp_path / "land_ckpt.npz"
        save_state_checkpoint(path, land_state, step=10, elapsed_time_s=1800.0, dt=1800.0)

        loaded, step, elapsed, meta = load_state_checkpoint(path, land_state)

        assert step == 10
        npt.assert_array_equal(
            np.asarray(land_state.T_soil.data),
            np.asarray(loaded.T_soil.data),
        )
        # runoff was None — should remain None (not in checkpoint)
        assert loaded.runoff is None

    def test_land_with_runoff_populated(self, tmp_path):
        state = _MiniLandState(
            T_soil=Field(data=jnp.ones((6, 4, 4)) * 285.0, name="T_soil",
                         dims=("face", "x", "y"), units="K"),
            W_bucket=Field(data=jnp.ones((6, 4, 4)) * 50.0, name="W_bucket",
                           dims=("face", "x", "y"), units="kg/m2"),
            runoff=jnp.ones((6, 4, 4)) * 0.001,
        )
        path = tmp_path / "land_runoff.npz"
        save_state_checkpoint(path, state, step=1, elapsed_time_s=100.0, dt=100.0)

        loaded, step, elapsed, meta = load_state_checkpoint(path, state)

        npt.assert_allclose(np.asarray(loaded.runoff), np.asarray(state.runoff))


# ---------------------------------------------------------------------------
# Static field handling
# ---------------------------------------------------------------------------


class TestStaticFields:
    def test_static_fields_from_template(self, tmp_path, ocean_state):
        """Static fields (H_bathy, land_mask) come from template, not file."""
        path = tmp_path / "ocean_static.npz"
        save_state_checkpoint(path, ocean_state, step=1, elapsed_time_s=600.0, dt=600.0)

        # Create a modified template with different bathymetry
        modified = ocean_state._replace(
            H_bathy=Field(data=jnp.ones((6, 4, 4)) * 9999.0, name="H_bathy",
                          dims=("face", "x", "y"), units="m"),
        )

        loaded, _, _, _ = load_state_checkpoint(
            path, modified, static_fields={"H_bathy", "land_mask"}
        )

        # Static fields should come from the modified template
        npt.assert_array_equal(np.asarray(loaded.H_bathy.data), 9999.0)
        # Prognostic fields should come from the file
        npt.assert_array_equal(
            np.asarray(loaded.T.data), np.asarray(ocean_state.T.data)
        )

    def test_restore_static_overrides(self, tmp_path, ocean_state):
        """restore_static=True loads static fields from file."""
        path = tmp_path / "ocean_restore.npz"
        save_state_checkpoint(path, ocean_state, step=1, elapsed_time_s=600.0, dt=600.0)

        modified = ocean_state._replace(
            H_bathy=Field(data=jnp.ones((6, 4, 4)) * 9999.0, name="H_bathy",
                          dims=("face", "x", "y"), units="m"),
        )

        loaded, _, _, _ = load_state_checkpoint(
            path, modified, restore_static=True,
            static_fields={"H_bathy"},
        )

        # With restore_static=True, file values win
        npt.assert_allclose(np.asarray(loaded.H_bathy.data), 4000.0)


# ---------------------------------------------------------------------------
# Prognostic-only save
# ---------------------------------------------------------------------------


class TestPrognosticFields:
    def test_prognostic_subset(self, tmp_path, ocean_state):
        path = tmp_path / "ocean_prog.npz"
        save_state_checkpoint(
            path, ocean_state, step=1, elapsed_time_s=600.0, dt=600.0,
            prognostic_fields={"T", "S", "eta"},
        )

        # H_bathy and land_mask not in the file — must come from template
        loaded, _, _, meta = load_state_checkpoint(path, ocean_state)

        npt.assert_array_equal(
            np.asarray(loaded.T.data), np.asarray(ocean_state.T.data)
        )
        # H_bathy should be from template (not in file)
        npt.assert_array_equal(
            np.asarray(loaded.H_bathy.data), np.asarray(ocean_state.H_bathy.data)
        )


# ---------------------------------------------------------------------------
# Config hash validation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    def test_matching_config(self, tmp_path, atm_state, config):
        path = tmp_path / "cfg_match.npz"
        save_state_checkpoint(
            path, atm_state, step=1, elapsed_time_s=100.0, dt=100.0,
            config=config,
        )

        # Loading with same config — no warnings
        loaded, _, _, _ = load_state_checkpoint(path, atm_state, config=config)
        assert loaded is not None

    def test_mismatched_config_warns(self, tmp_path, atm_state, config):
        path = tmp_path / "cfg_mismatch.npz"
        save_state_checkpoint(
            path, atm_state, step=1, elapsed_time_s=100.0, dt=100.0,
            config=config,
        )

        different_config = _MiniConfig(resolution=8, dt=900.0, scheme="rk4")
        with pytest.warns(match="config hash mismatch"):
            load_state_checkpoint(path, atm_state, config=different_config)


# ---------------------------------------------------------------------------
# Shape mismatch detection
# ---------------------------------------------------------------------------


class TestShapeMismatch:
    def test_shape_mismatch_raises(self, tmp_path, atm_state):
        path = tmp_path / "shape_bad.npz"
        save_state_checkpoint(path, atm_state, step=1, elapsed_time_s=100.0, dt=100.0)

        # Create template with different shape
        wrong_template = _MiniAtmState(
            T=Field(data=jnp.ones((6, 8, 8, 5)) * 280.0, name="T",
                    dims=("face", "x", "y", "level"), units="K"),
            u=Field(data=jnp.ones((6, 8, 8, 5)) * 10.0, name="u",
                    dims=("face", "x", "y", "level"), units="m/s"),
            p_s=Field(data=jnp.ones((6, 8, 8)) * 1e5, name="p_s",
                      dims=("face", "x", "y"), units="Pa"),
        )

        with pytest.raises(ValueError, match="shape mismatch"):
            load_state_checkpoint(path, wrong_template)


# ---------------------------------------------------------------------------
# Validate without loading
# ---------------------------------------------------------------------------


class TestValidateOnly:
    def test_validate_compatible(self, tmp_path, atm_state):
        path = tmp_path / "valid.npz"
        save_state_checkpoint(path, atm_state, step=1, elapsed_time_s=100.0, dt=100.0)

        issues = validate_state_checkpoint(path, atm_state)
        # No ERROR issues expected
        assert not any(i.startswith("ERROR") for i in issues)

    def test_validate_missing_meta(self, tmp_path, atm_state):
        path = tmp_path / "no_meta.npz"
        np.savez(path, T=np.ones(5))

        issues = validate_state_checkpoint(path, atm_state)
        assert any("no companion metadata" in i for i in issues)

    def test_validate_shape_mismatch(self, tmp_path, atm_state):
        path = tmp_path / "bad_shape.npz"
        save_state_checkpoint(path, atm_state, step=1, elapsed_time_s=100.0, dt=100.0)

        wrong_template = _MiniAtmState(
            T=Field(data=jnp.ones((6, 8, 8, 5)), name="T",
                    dims=("face", "x", "y", "level"), units="K"),
            u=Field(data=jnp.ones((6, 8, 8, 5)), name="u",
                    dims=("face", "x", "y", "level"), units="m/s"),
            p_s=Field(data=jnp.ones((6, 8, 8)), name="p_s",
                      dims=("face", "x", "y"), units="Pa"),
        )

        issues = validate_state_checkpoint(path, wrong_template)
        assert any("ERROR" in i and "shape mismatch" in i for i in issues)


# ---------------------------------------------------------------------------
# Metadata content
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_metadata_fields(self, tmp_path, atm_state, config):
        path = tmp_path / "meta_check.npz"
        save_state_checkpoint(
            path, atm_state, step=42, elapsed_time_s=12345.0, dt=900.0,
            config=config, model_version="1.2.3",
        )

        meta_file = _meta_path(path)
        with open(meta_file) as f:
            meta = json.load(f)

        assert meta["step"] == 42
        assert meta["elapsed_time_s"] == 12345.0
        assert meta["dt"] == 900.0
        assert meta["model_version"] == "1.2.3"
        assert meta["state_type"] == "_MiniAtmState"
        assert set(meta["saved_fields"]) == {"T", "u", "p_s"}
        assert "config_hash" in meta
        assert meta["config_hash"] != ""
        assert "state_digest" in meta

    def test_digest_changes_with_data(self, tmp_path, atm_state):
        path1 = tmp_path / "d1.npz"
        save_state_checkpoint(path1, atm_state, step=1, elapsed_time_s=0.0, dt=100.0)

        modified = atm_state._replace(
            T=Field(data=jnp.ones((6, 4, 4, 5)) * 300.0, name="T",
                    dims=("face", "x", "y", "level"), units="K"),
        )
        path2 = tmp_path / "d2.npz"
        save_state_checkpoint(path2, modified, step=1, elapsed_time_s=0.0, dt=100.0)

        with open(_meta_path(path1)) as f:
            m1 = json.load(f)
        with open(_meta_path(path2)) as f:
            m2 = json.load(f)

        assert m1["state_digest"] != m2["state_digest"]


# ---------------------------------------------------------------------------
# Precision casting
# ---------------------------------------------------------------------------


class TestPrecisionCasting:
    def test_load_casts_to_active_dtype(self, tmp_path):
        """Saved fp64 data is cast to active storage dtype on load."""
        state = _MiniAtmState(
            T=Field(data=jnp.ones((2, 2, 2, 2), dtype=jnp.float64) * 280.0,
                    name="T", dims=("face", "x", "y", "level"), units="K"),
            u=Field(data=jnp.ones((2, 2, 2, 2), dtype=jnp.float64) * 10.0,
                    name="u", dims=("face", "x", "y", "level"), units="m/s"),
            p_s=Field(data=jnp.ones((2, 2, 2), dtype=jnp.float64) * 1e5,
                      name="p_s", dims=("face", "x", "y"), units="Pa"),
        )
        path = tmp_path / "precision.npz"
        save_state_checkpoint(path, state, step=1, elapsed_time_s=0.0, dt=100.0)

        loaded, _, _, _ = load_state_checkpoint(path, state, strict=False)

        from legoesm.core.precision import get_policy
        expected_dtype = get_policy().storage
        for name in state._fields:
            val = getattr(loaded, name)
            if isinstance(val, Field):
                assert val.data.dtype == expected_dtype


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestDigestVerification:
    def test_corrupted_checkpoint_raises(self, tmp_path, atm_state):
        """Tampering with saved arrays triggers digest mismatch."""
        path = tmp_path / "corrupt.npz"
        save_state_checkpoint(path, atm_state, step=1, elapsed_time_s=0.0, dt=100.0)

        # Tamper with the npz file by rewriting one array
        npz = dict(np.load(path))
        npz["T"] = np.zeros_like(npz["T"])
        np.savez(path, **npz)

        with pytest.raises(ValueError, match="digest mismatch"):
            load_state_checkpoint(path, atm_state)

    def test_digest_skipped_with_static_fields(self, tmp_path, ocean_state):
        """Digest check is skipped when static fields are excluded."""
        path = tmp_path / "static_skip.npz"
        save_state_checkpoint(path, ocean_state, step=1, elapsed_time_s=0.0, dt=100.0)

        # This should not raise even though we skip some fields
        loaded, _, _, _ = load_state_checkpoint(
            path, ocean_state, static_fields={"H_bathy"}
        )
        assert loaded is not None


class TestHelpers:
    def test_extract_arrays(self, atm_state):
        arrays = _extract_arrays(atm_state)
        assert set(arrays.keys()) == {"T", "u", "p_s"}
        assert arrays["T"].shape == (6, 4, 4, 5)

    def test_extract_arrays_none_skipped(self, land_state):
        arrays = _extract_arrays(land_state)
        assert "runoff" not in arrays
        assert set(arrays.keys()) == {"T_soil", "W_bucket"}

    def test_field_metadata_extraction(self, atm_state):
        meta = _extract_field_metadata(atm_state)
        assert meta["T"]["is_field"] is True
        assert meta["T"]["units"] == "K"
        assert meta["T"]["dims"] == ["face", "x", "y", "level"]

    def test_field_metadata_none(self, land_state):
        meta = _extract_field_metadata(land_state)
        assert meta["runoff"]["is_none"] is True
