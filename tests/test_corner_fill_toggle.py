"""Unit tests for the FV3_3D iter-7 corner-fill mode toggle.

Verifies that:

1. Default mode is ``"avg"`` (legacy 2-point average) so all existing
   behaviour is preserved bit-for-bit.
2. The setter ``set_corner_fill_mode`` raises ValueError for unknown
   modes.
3. Switching to ``"fv3_agrid_xdir"`` produces the FV3-faithful
   diagonal-mirror result; switching back to ``"avg"`` restores the
   legacy values.
4. The two modes give DIFFERENT results on a non-trivial input
   (regression guard against silent no-op).

The toggle is at module scope in ``legoesm.grids.halo``, exposed via
``set_corner_fill_mode`` / ``get_corner_fill_mode``.
"""
from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.halo import (
    fill_corners_h1,
    fill_corners_h2,
    get_corner_fill_mode,
    set_corner_fill_mode,
)


@pytest.fixture(autouse=True)
def reset_mode():
    """Always reset to the default ``avg`` mode after each test so the
    process-wide toggle does not leak between tests."""
    yield
    set_corner_fill_mode("avg")


def test_default_mode_is_avg():
    assert get_corner_fill_mode() == "avg"


def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="Unknown corner fill mode"):
        set_corner_fill_mode("not_a_real_mode")


def test_bgrid_xdir_h1_uses_depth2_mirror():
    """BGRID-XDir h1 mode: SW corner = q[0, 2] (depth-2 mirror)."""
    set_corner_fill_mode("fv3_bgrid_xdir")
    n = 6
    padded = jnp.zeros((6, n + 2, n + 2))
    padded = padded.at[:, 0, 2].set(33.0)   # depth-2 along XDir
    out = fill_corners_h1(padded)
    np.testing.assert_array_equal(np.asarray(out[:, 0, 0]), 33.0 * np.ones(6))
    # NE corner: q[-1, -1] = q[-1, -3]
    padded2 = jnp.zeros((6, n + 2, n + 2))
    padded2 = padded2.at[:, -1, -3].set(77.0)
    out2 = fill_corners_h1(padded2)
    np.testing.assert_array_equal(np.asarray(out2[:, -1, -1]), 77.0 * np.ones(6))


def test_bgrid_xdir_h2_falls_through_to_avg():
    """BGRID-XDir h2 mode: falls through to legacy avg path.

    The iter-10 diagnostic found that applying FV3 BGRID depth-3/4
    mirror to the 2×2 cube-vertex L-block destabilises the 3D HS
    dycore.  Combined with the iter-7 finding that no 3D operator
    actually reads the 2×2 corner block, h2 BGRID is gated to fall
    through to the legacy avg path.  This regression-guards that
    routing — h2 in BGRID mode must produce the SAME output as h2
    in avg mode.
    """
    n = 6
    rng = np.random.default_rng(seed=43)
    padded_np = rng.uniform(-1.0, 1.0, size=(6, n + 4, n + 4))
    padded = jnp.asarray(padded_np)

    set_corner_fill_mode("avg")
    out_avg = fill_corners_h2(padded)
    set_corner_fill_mode("fv3_bgrid_xdir")
    out_bgrid = fill_corners_h2(padded)

    np.testing.assert_array_equal(np.asarray(out_avg), np.asarray(out_bgrid))


def test_bgrid_xdir_modes_differ_from_other_modes():
    """BGRID-XDir must produce results distinct from both AVG and AGRID-XDir."""
    n = 8
    rng = np.random.default_rng(seed=44)
    padded = jnp.asarray(rng.uniform(-1.0, 1.0, size=(6, n + 2, n + 2)))

    set_corner_fill_mode("avg")
    out_avg = np.asarray(fill_corners_h1(padded))
    set_corner_fill_mode("fv3_agrid_xdir")
    out_agrid = np.asarray(fill_corners_h1(padded))
    set_corner_fill_mode("fv3_bgrid_xdir")
    out_bgrid = np.asarray(fill_corners_h1(padded))

    assert float(np.max(np.abs(out_avg - out_bgrid))) > 0
    assert float(np.max(np.abs(out_agrid - out_bgrid))) > 0
    assert float(np.max(np.abs(out_avg - out_agrid))) > 0


def test_avg_mode_preserves_legacy_2point_average():
    """In ``avg`` mode, SW corner = 0.5*(adjacent_west + adjacent_south)."""
    set_corner_fill_mode("avg")
    n = 6
    # Pre-fill the halo array with distinct values.
    padded = jnp.zeros((6, n + 2, n + 2))
    padded = padded.at[:, 0, 1].set(10.0)   # west halo, j=1
    padded = padded.at[:, 1, 0].set(20.0)   # south halo, i=1
    out = fill_corners_h1(padded)
    # SW corner should be 0.5 * (10.0 + 20.0) = 15.0
    np.testing.assert_array_equal(np.asarray(out[:, 0, 0]), 15.0 * np.ones(6))


def test_fv3_agrid_xdir_uses_diagonal_mirror():
    """In ``fv3_agrid_xdir`` mode, SW corner = q[0, 1] (XDir mirror)."""
    set_corner_fill_mode("fv3_agrid_xdir")
    n = 6
    padded = jnp.zeros((6, n + 2, n + 2))
    padded = padded.at[:, 0, 1].set(10.0)   # west halo, j=1
    padded = padded.at[:, 1, 0].set(20.0)   # south halo, i=1
    out = fill_corners_h1(padded)
    # SW corner should be padded[0, 1] = 10.0 (NOT the average).
    np.testing.assert_array_equal(np.asarray(out[:, 0, 0]), 10.0 * np.ones(6))
    # NW corner should be padded[0, n] = padded[0, -2].
    padded_check = jnp.zeros((6, n + 2, n + 2))
    padded_check = padded_check.at[:, 0, -2].set(7.0)
    out2 = fill_corners_h1(padded_check)
    np.testing.assert_array_equal(
        np.asarray(out2[:, 0, -1]), 7.0 * np.ones(6),
    )


def test_modes_give_different_results_on_random_input():
    """avg and fv3_agrid_xdir must NOT be the same function."""
    n = 8
    rng = np.random.default_rng(seed=7)
    padded_np = rng.uniform(-1.0, 1.0, size=(6, n + 2, n + 2))
    padded = jnp.asarray(padded_np)

    set_corner_fill_mode("avg")
    out_avg = fill_corners_h1(padded)
    set_corner_fill_mode("fv3_agrid_xdir")
    out_xdir = fill_corners_h1(padded)

    # The two modes must differ at the cube vertices (24 cells per
    # 6 faces × 4 corners = 24, always).
    diff = np.asarray(out_avg) - np.asarray(out_xdir)
    # Only cube vertices differ; rest is identical.
    assert float(np.max(np.abs(diff))) > 0.0
    # Specifically, only the 24 corner positions differ.
    diff_mask = diff != 0.0
    expected = np.zeros((6, n + 2, n + 2), dtype=bool)
    expected[:, 0, 0] = True
    expected[:, 0, -1] = True
    expected[:, -1, 0] = True
    expected[:, -1, -1] = True
    np.testing.assert_array_equal(diff_mask, expected)


def test_round_trip_mode_change_restores_legacy():
    """Switching to fv3_agrid_xdir then back to avg restores legacy."""
    n = 5
    rng = np.random.default_rng(seed=3)
    padded = jnp.asarray(rng.uniform(-1.0, 1.0, size=(6, n + 2, n + 2)))

    set_corner_fill_mode("avg")
    out_before = fill_corners_h1(padded)

    set_corner_fill_mode("fv3_agrid_xdir")
    _ = fill_corners_h1(padded)

    set_corner_fill_mode("avg")
    out_after = fill_corners_h1(padded)

    np.testing.assert_array_equal(out_before, out_after)


# --- iter 8: h2 toggle ---


def test_h2_default_mode_is_avg_legacy():
    """h2 default mode reproduces the inside-out 2-point average."""
    set_corner_fill_mode("avg")
    n = 6
    # Random halo strip values; corner block initially zero.
    rng = np.random.default_rng(seed=23)
    padded_np = rng.uniform(-1.0, 1.0, size=(6, n + 4, n + 4))
    # Zero out the SW 2x2 block to verify the fill writes them.
    padded_np[:, 0:2, 0:2] = 0.0
    padded = jnp.asarray(padded_np)
    out = fill_corners_h2(padded)
    out_np = np.asarray(out)

    # Inside-out: (1,1) ← 0.5*(padded[1,2] + padded[2,1])
    # Then (0,1) ← 0.5*(padded[0,2] + (1,1)_filled)
    # Then (1,0) ← 0.5*(padded[2,0] + (1,1)_filled)
    # Finally (0,0) ← 0.5*((0,1)_filled + (1,0)_filled)
    f = 0
    inner = 0.5 * (padded_np[f, 1, 2] + padded_np[f, 2, 1])
    np.testing.assert_allclose(out_np[f, 1, 1], inner)
    expected_01 = 0.5 * (padded_np[f, 0, 2] + inner)
    np.testing.assert_allclose(out_np[f, 0, 1], expected_01)
    expected_10 = 0.5 * (padded_np[f, 2, 0] + inner)
    np.testing.assert_allclose(out_np[f, 1, 0], expected_10)
    expected_00 = 0.5 * (expected_01 + expected_10)
    np.testing.assert_allclose(out_np[f, 0, 0], expected_00)


def test_h2_xdir_mode_uses_diagonal_mirror():
    """h2 XDir mode SW block uses FV3 AGRID-XDir for ng=2."""
    set_corner_fill_mode("fv3_agrid_xdir")
    n = 6
    rng = np.random.default_rng(seed=23)
    padded_np = rng.uniform(-1.0, 1.0, size=(6, n + 4, n + 4))
    padded = jnp.asarray(padded_np)
    out = fill_corners_h2(padded)
    out_np = np.asarray(out)

    # FV3 AGRID-XDir for ng=2 (Fortran q(1-i, 1-j) = q(1-j, i)
    # with i,j in {1,2}; padded index = Fortran index + 1):
    #   (1, 1) ← (1, 2)
    #   (1, 0) ← (0, 2)
    #   (0, 1) ← (1, 3)
    #   (0, 0) ← (0, 3)
    np.testing.assert_array_equal(out_np[:, 1, 1], padded_np[:, 1, 2])
    np.testing.assert_array_equal(out_np[:, 1, 0], padded_np[:, 0, 2])
    np.testing.assert_array_equal(out_np[:, 0, 1], padded_np[:, 1, 3])
    np.testing.assert_array_equal(out_np[:, 0, 0], padded_np[:, 0, 3])


def test_h2_modes_differ_on_random_input():
    """h2 avg and fv3_agrid_xdir must produce different results."""
    n = 8
    rng = np.random.default_rng(seed=99)
    padded = jnp.asarray(rng.uniform(-1.0, 1.0, size=(6, n + 4, n + 4)))

    set_corner_fill_mode("avg")
    out_avg = fill_corners_h2(padded)

    set_corner_fill_mode("fv3_agrid_xdir")
    out_xdir = fill_corners_h2(padded)

    diff = float(jnp.max(jnp.abs(out_avg - out_xdir)))
    assert diff > 0.0, "h2 modes must differ on random input"


def test_invalid_env_value_raises_at_import():
    import os
    import subprocess
    env = dict(os.environ, LEGOESM_CORNER_FILL="fv3_bgrid", JAX_PLATFORMS="cpu")
    r = subprocess.run([sys.executable, "-c", "import legoesm.grids.halo"],
                       env=env, capture_output=True, text=True)
    assert r.returncode != 0 and "LEGOESM_CORNER_FILL" in r.stderr


def test_env_var_conflicting_with_config_raises(monkeypatch):
    from legoesm.grids.halo import apply_corner_fill_config
    monkeypatch.setenv("LEGOESM_CORNER_FILL", "avg")
    with pytest.raises(ValueError, match="disagrees with the run config"):
        apply_corner_fill_config("fv3_bgrid_xdir")


def test_config_door_refuses_a_mode_another_trace_already_used(monkeypatch):
    """The fill reads the mode at trace time, so a compiled function keeps the
    mode it traced with; a run configured for another mode in the same process
    would silently mix two modes (codex review of #1811)."""
    from legoesm.grids import halo
    monkeypatch.setattr(halo, "_corner_fill_mode", halo._corner_fill_mode)
    monkeypatch.setattr(halo, "_corner_fill_claimed", None)
    monkeypatch.setattr(halo, "_corner_fill_traced", {"fv3_bgrid_xdir"})
    halo.apply_corner_fill_config("fv3_bgrid_xdir")          # same mode: fine
    monkeypatch.setattr(halo, "_corner_fill_claimed", None)
    with pytest.raises(ValueError, match="already traced"):
        halo.apply_corner_fill_config("avg")


def test_model_driver_applies_config_corner_fill(monkeypatch):
    import legoesm.grids.halo as halo
    monkeypatch.delenv("LEGOESM_CORNER_FILL", raising=False)
    monkeypatch.setattr(halo, "_corner_fill_mode", halo._corner_fill_mode)
    monkeypatch.setattr(halo, "_corner_fill_claimed", None)
    monkeypatch.setattr(halo, "_corner_fill_traced", set())    # earlier tests traced other modes
    from legoesm.driver.config import DycoreConfig, ExperimentConfig
    from legoesm.driver.model_driver import ModelDriver
    ModelDriver(ExperimentConfig(dycore=DycoreConfig(corner_fill="fv3_bgrid_xdir")))
    assert get_corner_fill_mode() == "fv3_bgrid_xdir"


def test_second_model_with_other_corner_fill_raises(monkeypatch):
    import legoesm.grids.halo as halo
    monkeypatch.delenv("LEGOESM_CORNER_FILL", raising=False)
    monkeypatch.setattr(halo, "_corner_fill_mode", halo._corner_fill_mode)
    monkeypatch.setattr(halo, "_corner_fill_claimed", None)
    monkeypatch.setattr(halo, "_corner_fill_traced", set())    # earlier tests traced other modes
    from legoesm.driver.config import DycoreConfig, ExperimentConfig
    from legoesm.driver.model_driver import ModelDriver
    ModelDriver(ExperimentConfig(dycore=DycoreConfig(corner_fill="fv3_bgrid_xdir")))
    ModelDriver(ExperimentConfig(dycore=DycoreConfig(corner_fill="fv3_bgrid_xdir")))
    with pytest.raises(ValueError, match="already built in this process"):
        ModelDriver(ExperimentConfig(dycore=DycoreConfig(corner_fill="avg")))
    assert get_corner_fill_mode() == "fv3_bgrid_xdir"


def test_traced_corner_fill_mode_reaches_the_run_manifest(monkeypatch, tmp_path):
    import legoesm.grids.halo as halo
    from legoesm.driver.config import ExperimentConfig
    from legoesm.driver.model_driver import ModelDriver
    from legoesm.driver.restart import read_run_manifest, write_run_manifest
    monkeypatch.delenv("LEGOESM_CORNER_FILL", raising=False)
    monkeypatch.setattr(halo, "_corner_fill_mode", halo._corner_fill_mode)
    monkeypatch.setattr(halo, "_corner_fill_claimed", None)
    monkeypatch.setattr(halo, "_corner_fill_traced", set())
    driver = ModelDriver(ExperimentConfig())
    set_corner_fill_mode("fv3_agrid_xdir")          # changed after the build
    jax.jit(halo.fill_corners_h1)(jnp.zeros((6, 6, 6)))
    driver._output_dir = tmp_path
    driver._mpi_rank = None
    write_run_manifest(tmp_path, driver._input_config)
    driver._record_final_state_digest()
    manifest = read_run_manifest(tmp_path)
    assert manifest["result"]["corner_fill_traced"] == ["fv3_agrid_xdir"]
    from legoesm.driver.restart import validate_run_manifest
    validate_run_manifest(manifest)               # the extra key stays valid
