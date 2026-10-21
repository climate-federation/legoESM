"""Coherence code of the SWG twin scale-predictability probe: identical -> 1, independent -> ~0."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.fft import dctn, idctn

PROBE = (Path(__file__).resolve().parents[2] / "scripts/validate/ocean_fidelity/testcases"
         / "swg_twin_scale_predictability.py")
spec = importlib.util.spec_from_file_location("swg_twin_scale_predictability", PROBE)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)

N = 39


@pytest.mark.parametrize("basis", ["dct", "dst"])
def test_identical_fields_are_fully_coherent(basis):
    a = np.random.default_rng(0).normal(size=(3, N, N))
    assert np.allclose(probe.band_coherence(a, a, basis), 1.0, atol=1e-12)


def test_independent_fields_are_incoherent():
    rng = np.random.default_rng(1)
    c = probe.band_coherence(rng.normal(size=(200, N, N)), rng.normal(size=(200, N, N))).mean(axis=0)
    # bands 0 (one coefficient) and the corner bands hold few coefficients: judge bands with >= 20
    counts = np.bincount(probe.band_index(N, N).ravel())
    assert np.all(np.abs(c[counts >= 20]) < 0.1)


def test_scale_selective_noise_gives_the_right_cutoff():
    # b = a + noise confined above band 10: coherence ~1 below, <1 above, so the cutoff sits at 10
    rng = np.random.default_rng(2)
    a = rng.normal(size=(1, N, N))
    fa = dctn(a, type=2, axes=(-2, -1), norm="ortho")
    hi = probe.band_index(N, N) > 10
    noise = idctn(np.where(hi, rng.normal(size=fa.shape) * 5.0, 0.0), type=2, axes=(-2, -1), norm="ortho")
    c = probe.band_coherence(a, a + noise)
    assert probe.cutoff(c, 0.99, contiguous=True)[0] == 10
    assert c[0, 15:35].mean() < 0.35  # expected 1/sqrt(1+25) = 0.2


def test_length_conversion():
    assert np.isnan(probe.length_km(0, N, 50e3))
    assert probe.length_km(N, N, 50e3) == pytest.approx(100.0)  # Nyquist 2*dx


def test_control_starts_default_and_validation():
    assert probe.control_starts(None, 3) == [0, 0, 0]
    assert probe.control_starts([731, 1168], 2) == [731, 1168]
    with pytest.raises(ValueError):
        probe.control_starts([731], 2)
    with pytest.raises(ValueError):
        probe.control_starts([-1, 5], 2)
