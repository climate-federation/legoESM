"""Unit tests for the pure metrics behind the cube-artifact visual-regression gate
(``scripts/validate/visual_regression.py``).

The full Williamson-2 cube sim is too heavy for unit CI and runs in a nightly job;
these tests pin the *metrics* (SSIM / perceptual hash / Hamming) on synthetic
fields so the gate is provably non-vacuous — identical fields score perfectly,
and a deliberately injected cube-edge imprint is detected (SSIM drops and the
perceptual hash diverges). The script's metric functions are loaded directly from
its file (scripts/ is not an importable package).
"""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "validate" / "visual_regression.py"


def _load_metrics():
    spec = importlib.util.spec_from_file_location("_visreg", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


vr = _load_metrics()


def _smooth_field(seed: int = 0) -> np.ndarray:
    """A smooth, deterministic (6, 16, 16) field resembling a cube v-wind."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((6, 4, 4))
    # upsample 4x4 -> 16x16 per panel (smooth, no grid-scale noise)
    out = np.zeros((6, 16, 16))
    for p in range(6):
        out[p] = np.kron(base[p], np.ones((4, 4)))
    return out


def test_ssim_identical_is_one() -> None:
    f = _smooth_field()
    assert vr.ssim(f, f) == pytest.approx(1.0, abs=1e-9)


def test_ssim_drops_under_edge_imprint() -> None:
    """Injecting a grid-scale cube-edge imprint must drop SSIM below the gate."""
    f = _smooth_field()
    g = f.copy()
    g[:, 0, :] += 5.0   # spurious signal on every panel's edge row
    g[:, -1, :] += 5.0
    g[:, :, 0] += 5.0
    g[:, :, -1] += 5.0
    assert vr.ssim(f, g) < vr.SSIM_MIN


def test_phash_identical_zero_hamming() -> None:
    f = _smooth_field()
    assert vr.hamming(vr.phash(f), vr.phash(f)) == 0


def test_phash_detects_pattern_change() -> None:
    f = _smooth_field(seed=0)
    g = _smooth_field(seed=1)
    assert vr.hamming(vr.phash(f), vr.phash(g)) > vr.HAMMING_MAX


def test_ssim_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        vr.ssim(np.zeros((6, 8, 8)), np.zeros((6, 16, 16)))


def test_phash_is_deterministic() -> None:
    f = _smooth_field()
    assert vr.phash(f) == vr.phash(f.copy())
