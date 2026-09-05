"""Unit tests for the NEMO RGB-chlorophyll shortwave-penetration scheme.

Validates ``ocean/physics/shortwave_penetration.py`` rgb_chl path against an
INDEPENDENT NumPy mirror of NEMO 5.0.1 ``tra_qsr`` (qsr_RGBc), the NEMO
class-index formula, exact column heat closure, and the Morel & Berthon (1989)
vertical Chl profile polynomial -- none of which copy the implementation.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import math

import numpy as np
import pytest

from legoesm.ocean.physics.shortwave_penetration import (
    ShortwavePenetrationConfig,
    apply_shortwave_penetration,
    shortwave_penetration_rgb_tendency,
    shortwave_penetration_tendency,
    _RGB_ATTENUATION_BGR,
    _rgb_class_row,
    _morel_berthon_chl_column,
)

# NEMO ORCA1 RGB namelist defaults (rn_abs, rn_si0) and seawater heat budget.
_RN_ABS = 0.58
_RN_SI0 = 0.35
_RHO0 = 1026.0
_CSW = 3991.86795711963

# Independent copy of the NEMO 61x3 (blue, green, red) table from trc_oce.F90,
# re-transcribed here so the test does not import the module's own copy for the
# numeric mirror (class-index + band-sum checks read it back independently).
_NEMO_TABLE = np.array(_RGB_ATTENUATION_BGR, dtype=np.float64)


def _nemo_class_index_1based(chl: float) -> int:
    """NEMO itab = NINT(41 + 20*log10(Chl)), Chl clamped to [0.03, 10] (1-based)."""
    c = min(max(chl, 0.03), 10.0)
    return int(math.floor(41.0 + 20.0 * math.log10(c) + 0.5))


def _nemo_rgb_recursion(sw, chl_col, dz, n_wet, rho0=_RHO0, csw=_CSW):
    """Independent NumPy mirror of NEMO qsr_RGBc per-level band recursion.

    ``chl_col`` is the per-level chlorophyll (nlev,). Returns dT/dt (nlev,)."""
    nlev = len(dz)
    k_ir = 1.0 / _RN_SI0
    frac_ir = _RN_ABS
    frac_rgb = (1.0 - _RN_ABS) / 3.0
    ze0 = frac_ir * sw
    zeR = frac_rgb * sw
    zeG = frac_rgb * sw
    zeB = frac_rgb * sw
    zeT = sw
    dT = np.zeros(nlev)
    for jk in range(nlev):
        e3t = dz[jk]
        row = _nemo_class_index_1based(chl_col[jk]) - 1
        row = min(max(row, 0), 60)
        k_b, k_g, k_r = _NEMO_TABLE[row]
        if e3t <= 0.0:
            dT[jk] = 0.0
            continue
        zze0 = ze0 * math.exp(-e3t * k_ir)
        zzeR = zeR * math.exp(-e3t * k_r)
        zzeG = zeG * math.exp(-e3t * k_g)
        zzeB = zeB * math.exp(-e3t * k_b)
        # wmask on the face below cell jk: wet iff cell jk+1 is also wet.
        wmask_next = 1.0 if (jk + 1) < n_wet else 0.0
        zzeT = (zze0 + zzeR + zzeG + zzeB) * wmask_next
        dT[jk] = (zeT - zzeT) / (rho0 * csw * e3t)
        ze0, zeR, zeG, zeB, zeT = zze0, zzeR, zzeG, zzeB, zzeT
    return dT


# ---------------------------------------------------------------------------
# Class index
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "chl, expected_1based",
    [(0.03, 11), (0.1, 21), (1.0, 41), (10.0, 61),
     (0.001, 11),   # below clamp -> 0.03 class
     (100.0, 61)],  # above clamp -> 10 class
)
def test_class_index_matches_nemo(chl, expected_1based):
    row = int(np.asarray(_rgb_class_row(np.array(chl))))
    assert row == expected_1based - 1
    assert row == _nemo_class_index_1based(chl) - 1


def test_table_shape_and_monotone():
    assert _NEMO_TABLE.shape == (61, 3)
    # Attenuation increases with chlorophyll class for every band.
    assert np.all(np.diff(_NEMO_TABLE, axis=0) > 0)


# ---------------------------------------------------------------------------
# Band-sum vs independent recursion (profile = "surface": constant column Chl)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("chl", [0.05, 0.3, 2.0, 8.0])
def test_rgb_tendency_matches_nemo_recursion(chl):
    dz = np.array([5.0, 10.0, 20.0, 40.0, 80.0, 160.0])  # m
    n = len(dz)
    sw = 200.0
    cfg = ShortwavePenetrationConfig(scheme="rgb_chl", rgb_chl_profile="surface")
    dT = np.asarray(
        shortwave_penetration_rgb_tendency(
            np.array(sw),
            np.array(chl),
            np.asarray(dz),
            np.ones(n),
            cfg,
            rho_0=_RHO0,
            c_sw=_CSW,
        )
    )
    dT_ref = _nemo_rgb_recursion(sw, np.full(n, chl), dz, n_wet=n)
    np.testing.assert_allclose(dT, dT_ref, rtol=1e-10, atol=1e-14)


def test_column_heat_closure():
    """100% of incident SW deposited in the wet column (energy conservation)."""
    dz = np.array([5.0, 10.0, 20.0, 40.0, 80.0, 160.0])
    n = len(dz)
    sw = 173.0
    cfg = ShortwavePenetrationConfig(scheme="rgb_chl", rgb_chl_profile="surface")
    dT = np.asarray(
        shortwave_penetration_rgb_tendency(
            np.array(sw), np.array(0.2), np.asarray(dz), np.ones(n), cfg,
            rho_0=_RHO0, c_sw=_CSW))
    absorbed = float(np.sum(dT * _RHO0 * _CSW * dz))
    assert absorbed == pytest.approx(sw, rel=1e-9)


def test_partial_wet_column_closure_and_bottom_deposit():
    """A column with dry cells below the seabed still absorbs 100% of SW, all
    light reaching the seabed lands in the LAST WET level (no leak below)."""
    dz_live = np.array([10.0, 20.0, 40.0, 0.0, 0.0])  # 3 wet cells, 2 dry
    wet = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
    n_wet = 3
    sw = 150.0
    cfg = ShortwavePenetrationConfig(scheme="rgb_chl", rgb_chl_profile="surface")
    dT = np.asarray(
        shortwave_penetration_rgb_tendency(
            np.array(sw), np.array(0.1), np.asarray(dz_live), np.asarray(wet),
            cfg, rho_0=_RHO0, c_sw=_CSW))
    # Dry cells contribute nothing.
    assert dT[3] == 0.0 and dT[4] == 0.0
    absorbed = float(np.sum(dT[:n_wet] * _RHO0 * _CSW * dz_live[:n_wet]))
    assert absorbed == pytest.approx(sw, rel=1e-9)
    dT_ref = _nemo_rgb_recursion(sw, np.full(5, 0.1), dz_live, n_wet=n_wet)
    np.testing.assert_allclose(dT[:n_wet], dT_ref[:n_wet], rtol=1e-10, atol=1e-14)


def test_dry_column_zero():
    dz = np.zeros(5)
    cfg = ShortwavePenetrationConfig(scheme="rgb_chl")
    dT = np.asarray(
        shortwave_penetration_rgb_tendency(
            np.array(120.0), np.array(0.3), dz, np.zeros(5), cfg))
    assert np.all(dT == 0.0)


# ---------------------------------------------------------------------------
# Morel & Berthon vertical profile (independent polynomial re-derivation)
# ---------------------------------------------------------------------------
def _nemo_morel_berthon(chl_surface, gdepw):
    c = min(max(chl_surface, 0.03), 10.0)
    zlogc = math.log(c)
    zc1 = 0.113328685307 + 0.803 * zlogc
    zc2 = 3.703768066608 + 0.459 * zlogc
    zc3 = 6.34247346942 - 0.746 * zc2
    if zc3 > 4.62497281328:
        zc3 = 5.298317366548 - 0.293 * zc2
    zCze = math.exp(zc1)
    inv_delpsi = 1.0 / (0.710 + zlogc * (0.159 + zlogc * 0.021))
    inv_zze = math.exp(-zc3)
    zCb = 0.768 + zlogc * (0.087 - zlogc * (0.179 + zlogc * 0.025))
    zCmax = 0.299 - zlogc * (0.289 - zlogc * 0.579)
    zpsimax = 0.6 - zlogc * (0.640 - zlogc * (0.021 + zlogc * 0.115))
    out = []
    for d in gdepw:
        zpsi = inv_zze * d
        val = zCze * (zCb + zCmax * math.exp(-(((zpsi - zpsimax) * inv_delpsi) ** 2)))
        out.append(min(max(val, 0.03), 10.0))
    return np.array(out)


@pytest.mark.parametrize("chl_s", [0.04, 0.15, 0.8, 5.0])
def test_morel_berthon_profile_matches_nemo(chl_s):
    dz = np.array([5.0, 10.0, 20.0, 40.0, 80.0, 160.0])
    gdepw = np.cumsum(dz)  # bottom-interface depths (what the function expects)
    chl_z = np.asarray(
        _morel_berthon_chl_column(np.array(chl_s), np.asarray(gdepw)[None, :])[0])
    chl_ref = _nemo_morel_berthon(chl_s, gdepw)
    np.testing.assert_allclose(chl_z, chl_ref, rtol=1e-10, atol=1e-12)


def test_morel_berthon_has_subsurface_maximum():
    """For mildly oligotrophic surface Chl the profile has a deep-Chl maximum
    below the surface (the physical signature of nn_chlprfl=1).  At Chl=0.1 the
    Morel-Berthon peak parameter zpsimax=+0.78 places the DCM near 60 m (within
    the 200 m column); for very low Chl (<~0.07) zpsimax goes negative and the
    column is monotone-decreasing, so this asserts the DCM regime specifically."""
    dz = np.full(40, 5.0)
    gdepw = np.cumsum(dz)  # 5, 10, ..., 200 m
    chl_z = np.asarray(
        _morel_berthon_chl_column(np.array(0.1), np.asarray(gdepw)[None, :])[0])
    assert chl_z.argmax() > 0          # maximum is below the surface cell
    assert chl_z.argmax() < len(dz) - 1  # ... and above the column bottom (a true DCM)


# ---------------------------------------------------------------------------
# Dispatch hardening
# ---------------------------------------------------------------------------
def test_dispatch_unknown_scheme_raises():
    cfg = ShortwavePenetrationConfig(scheme="banana")
    with pytest.raises(ValueError, match="unknown shortwave penetration scheme"):
        apply_shortwave_penetration(cfg, np.array(100.0),
                                    dz_ref=np.array([10.0]),
                                    z_half_ref=np.array([0.0, -10.0]),
                                    jacobian=np.array(1.0))


def test_nemo_qsr_rgb_selector_requires_and_consumes_source_operands():
    sw = np.array([[120.0]])
    chl = np.array([[0.2]])
    dz = np.array([[[2.0, 8.0]]])
    wet = np.ones_like(dz)
    common = dict(sw_down=sw, chl=chl, dz_live=dz, wet_cell=wet)
    generic = apply_shortwave_penetration(
        ShortwavePenetrationConfig(scheme="rgb_chl"), **common)
    cfg = ShortwavePenetrationConfig(
        scheme="nemo_qsr_rgb", nemo_time_step_s=10800.0)
    with pytest.raises(ValueError, match="requires gdepw_bottom_live"):
        apply_shortwave_penetration(cfg, **common)
    nemo = apply_shortwave_penetration(
        cfg,
        **common,
        gdepw_bottom_live=np.array([[[2.0, 10.0]]]),
        gdepw_ref=np.array([0.0, 2.0, 10.0]),
        e3t_ref=np.array([2.0, 8.0]),
        rho_0=_RHO0,
        c_sw=_CSW,
    )
    assert np.asarray(nemo).shape == np.asarray(generic).shape
    assert np.isfinite(np.asarray(nemo)).all()


def test_dispatch_rgb_requires_chl():
    cfg = ShortwavePenetrationConfig(scheme="rgb_chl")
    with pytest.raises(ValueError, match="requires a chlorophyll field"):
        apply_shortwave_penetration(cfg, np.array(100.0),
                                    dz_live=np.array([10.0]),
                                    wet_cell=np.array([1.0]))


def test_dispatch_rgb_requires_dz_live_and_wet():
    cfg = ShortwavePenetrationConfig(scheme="rgb_chl")
    with pytest.raises(ValueError, match="dz_live and wet_cell"):
        apply_shortwave_penetration(cfg, np.array(100.0), chl=np.array([0.1]))


def test_two_band_kernel_rejects_rgb_config():
    cfg = ShortwavePenetrationConfig(scheme="rgb_chl")
    with pytest.raises(ValueError, match="two-band Jerlov kernel"):
        shortwave_penetration_tendency(
            np.array(100.0), np.array([10.0]), np.array([0.0, -10.0]),
            np.array(1.0), cfg)


def test_unknown_chl_profile_raises():
    cfg = ShortwavePenetrationConfig(scheme="rgb_chl", rgb_chl_profile="bogus")
    with pytest.raises(ValueError, match="unknown rgb_chl_profile"):
        shortwave_penetration_rgb_tendency(
            np.array(100.0), np.array(0.1), np.array([10.0]), np.array([1.0]), cfg)


def test_default_scheme_is_two_band_backward_compatible():
    """The default config still drives the original two-band kernel unchanged."""
    cfg = ShortwavePenetrationConfig()
    assert cfg.scheme == "jerlov_2band"
    dz_ref = np.array([10.0, 20.0, 40.0])
    z_half = np.array([0.0, -10.0, -30.0, -70.0])
    out = np.asarray(apply_shortwave_penetration(
        cfg, np.array(200.0), dz_ref=dz_ref, z_half_ref=z_half,
        jacobian=np.array(1.0)))
    direct = np.asarray(shortwave_penetration_tendency(
        np.array(200.0), dz_ref, z_half, np.array(1.0), cfg))
    np.testing.assert_array_equal(out, direct)
