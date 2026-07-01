"""SAM fixed-zenith perpetual RCE insolation tests (iter-11 RAD-2).

SAM's ``doperpetual`` RCE uses a FIXED solar zenith angle and reduced solar
constant: zenith 51.7° (cosθ=0.620), S_0=685 W/m² ⇒ uniform TOA insolation
685·0.620 ≈ 425 W/m², with cosθ used directly as the SW optical-path cosine.
The legacy path used a latitude-based daily-mean with a daytime-effective
cos(SZA).
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics.radiation.config import (
    RadiationConfig, RRTMGPConfig,
)
from legoesm.atmosphere.physics.radiation.integration import (
    _compute_insolation,
)

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "run"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from run_rcemip_plane import _build_radiation_config  # type: ignore  # noqa: E402


def test_fixed_zenith_uniform_insolation_and_cos_sza():
    """rce_fixed_cos_zenith ⇒ TOA = S_0·cosθ UNIFORM (lat-independent) and
    cosθ returned directly as cos_sza, with no f_day rescaling."""
    cfg = RadiationConfig(
        scheme="rrtmgp", rrtmgp=RRTMGPConfig(S_0=685.0),
        rce_fixed_cos_zenith=0.620,
    )
    lat = jnp.array([0.0, 0.6, -0.4, 1.2])
    insol, cos_sza, f_day, _ = _compute_insolation(lat, cfg)
    assert np.allclose(np.asarray(insol), 685.0 * 0.620)   # 424.7 uniform
    assert cos_sza is not None
    assert np.allclose(np.asarray(cos_sza), 0.620)         # fixed optical path
    assert f_day is None                                   # no daytime rescale


def test_off_keeps_latitude_daily_mean_path():
    """Default (None) keeps the legacy latitude-based path: cos_sza derived
    downstream (None here) and f_day set, full S_0."""
    cfg = RadiationConfig(scheme="rrtmgp", rrtmgp=RRTMGPConfig())
    _, cos_sza, f_day, _ = _compute_insolation(jnp.array([0.0]), cfg)
    assert cos_sza is None
    assert f_day is not None


def test_default_insolation_is_rcemip():
    """The DEFAULT is the actual RCEMIP1/prm setup (551.58, 42.05°), not the
    generic 'sam' 685/51.7° — RCEMIP1/prm uses doperpetual+dosolarconstant
    with solar_constant=551.58, zenith_angle=42.05."""
    cfg = _build_radiation_config("rrtmgp")          # no insolation arg
    assert cfg.rrtmgp.S_0 == pytest.approx(551.58)
    assert cfg.rce_fixed_cos_zenith == pytest.approx(0.7425)


def test_driver_presets_sam_and_rcemip():
    """The driver builder wires SAM (685, 0.620⇒425) and RCEMIP
    (551.58, 0.7425⇒409.6) presets; 'off' = full S_0, no fixed zenith."""
    sam = _build_radiation_config("rrtmgp", insolation="sam")
    assert sam.rrtmgp.S_0 == 685.0
    assert sam.rce_fixed_cos_zenith == pytest.approx(0.620)
    insol, _, _, _ = _compute_insolation(jnp.array([0.0]), sam)
    assert float(insol[0]) == pytest.approx(425.0, abs=1.0)

    rce = _build_radiation_config("rrtmgp", insolation="rcemip")
    assert rce.rrtmgp.S_0 == pytest.approx(551.58)
    i2, _, _, _ = _compute_insolation(jnp.array([0.0]), rce)
    assert float(i2[0]) == pytest.approx(409.6, abs=1.0)

    off = _build_radiation_config("rrtmgp", insolation="off")
    assert off.rce_fixed_cos_zenith is None
    assert off.rrtmgp.S_0 == RRTMGPConfig().S_0     # full solar constant


def test_driver_sets_sam_mls_ghg_and_albedo_for_rce():
    """RAD-3/RAD-4: the fixed-zenith RCE RRTMGP config carries SAM's MLS trace
    gases (CO2≈355, CH4≈1700, N2O≈320 — from rrtmg_lw.nc AbsorberAmountMLS, vs
    legoESM's modern 415/1900/332) and the Briegleb direct ocean albedo
    (≈0.033) with a 0.07 diffuse split. 'off' keeps the modern defaults."""
    rce = _build_radiation_config("rrtmgp", insolation="rcemip")
    assert rce.rrtmgp.co2_ppmv == pytest.approx(355.0)
    assert rce.rrtmgp.ch4_ppbv == pytest.approx(1700.0)
    assert rce.rrtmgp.n2o_ppbv == pytest.approx(320.0)
    assert rce.rrtmgp.sfc_albedo_direct == pytest.approx(0.0329, abs=1e-3)
    assert rce.rrtmgp.sfc_albedo == pytest.approx(0.07)
    # SAM-RCEMIP CO2 is LOWER than legoESM's modern default ⇒ less LW forcing.
    assert rce.rrtmgp.co2_ppmv < RRTMGPConfig().co2_ppmv

    off = _build_radiation_config("rrtmgp", insolation="off")
    assert off.rrtmgp.co2_ppmv == pytest.approx(415.0)   # modern default kept
    assert off.rrtmgp.sfc_albedo_direct is None


def test_gray_scheme_fixed_zenith_sets_gray_S0():
    """For gray radiation the reduced S_0 lives on the gray sub-config."""
    cfg = _build_radiation_config("gray", insolation="sam")
    assert cfg.gray.S_0 == 685.0
    assert cfg.rce_fixed_cos_zenith == pytest.approx(0.620)


def test_unknown_insolation_raises():
    with pytest.raises(ValueError, match="Unknown --insolation"):
        _build_radiation_config("rrtmgp", insolation="perpetual_typo")
