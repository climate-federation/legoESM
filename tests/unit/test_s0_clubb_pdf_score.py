"""s0_clubb_pdf_score.py: the cold-layer mean and the moment alarms are non-vacuous."""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

_TOOL = (pathlib.Path(__file__).resolve().parents[2]
         / "scripts" / "validate" / "amip_bias" / "s0_clubb_pdf_score.py")
_spec = importlib.util.spec_from_file_location("s0_score", _TOOL)
s0 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s0)


def _grid():
    lat = np.array([80.0, 80.0, 40.0])          # two polar cells, one mid-latitude
    area = np.array([1.0, 3.0, 1.0])
    T = np.full((3, 4), 260.0)
    T[:, 2:] = 240.0                             # lowest two layers cold
    return lat, area, T


def test_cold_cap_mean_weights_by_area_and_masks_warm_layers():
    lat, area, T = _grid()
    f = np.zeros((3, 4))
    f[0, 2:] = 1.0                               # cell 0 cold layers cloudy
    f[1, :2] = 1.0                               # cell 1 only WARM layers cloudy -> masked
    f[2, :] = 1.0                                # mid-latitude -> masked
    mean, n = s0.cold_cap_mean(f, T, lat, area, 75.0, 253.0)
    assert n == 4
    assert mean == pytest.approx(1.0 * 2 / (2 * 1.0 + 2 * 3.0))


def test_cold_cap_mean_refuses_empty_mask():
    lat, area, T = _grid()
    with pytest.raises(SystemExit):
        s0.cold_cap_mean(np.zeros_like(T), T, lat, area, 75.0, 100.0)


def _packed(ncol, nlev, **fields):
    from legoesm.atmosphere.physics.turbulence.clubb import (
        CLUBBMomentState, pack_clubb_moments)
    import jax.numpy as jnp
    kw = {}
    for name in CLUBBMomentState._fields:
        nz = nlev if name in ("rtm", "thlm", "um", "vm", "wp3") else nlev + 1
        quiet = 1e-6 if name == "rtp2" else 0.1        # every field below its alarm
        kw[name] = jnp.asarray(fields.get(name, np.full((ncol, nz), quiet)))
    return np.asarray(pack_clubb_moments(CLUBBMomentState(**kw)))


def test_moment_alarm_fires_and_is_silent_when_quiet():
    lat, area, T = _grid()
    quiet = _packed(3, 4)
    res, alarms = s0.moment_stats(quiet, T, lat, area, 75.0, 253.0)
    assert alarms == []
    assert res["wp2"][0] == pytest.approx(0.1)
    hot = np.full((3, 5), 0.1); hot[1, 0] = 250.0   # zm index 0 = surface
    loud = _packed(3, 4, thlp2=hot)
    _, alarms = s0.moment_stats(loud, T, lat, area, 75.0, 253.0)
    assert alarms and alarms[0].startswith("thlp2 |max| 250")


def test_non_finite_moment_is_an_alarm():
    lat, area, T = _grid()
    bad = np.full((3, 4), 0.1); bad[0, 1] = np.nan
    _, alarms = s0.moment_stats(_packed(3, 4, wp3=bad), T, lat, area, 75.0, 253.0)
    assert any("non-finite" in a for a in alarms)


def test_surface_layer_lands_top_down():
    # zm surface value must be averaged into the LOWEST top-down layer (index -1).
    lat, area, T = _grid()
    T[:] = 260.0; T[:, -1] = 240.0                # only the lowest layer is cold
    w = np.full((3, 5), 0.0); w[:, 0] = 2.0        # ascending zm index 0 = surface
    res, _ = s0.moment_stats(_packed(3, 4, wp2=w), T, lat, area, 75.0, 253.0)
    assert res["wp2"][0] == pytest.approx(1.0)     # 0.5*(2+0) on the surface layer
