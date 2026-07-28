"""Unit tests for the #1226 dyn_hpg dump comparator
(``scripts/validate/ocean_fidelity/dino_1226/hpg_tendency_compare.py``).

Covers the three pieces of non-trivial logic that a silent bug would hide
inside a plausible-looking correlation: the NEMO ``usr_def_istate`` CASE(4)
transcription, the Fortran dump index ordering, and the corr/ratio convention.
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

_SCRIPT = (pathlib.Path(__file__).resolve().parents[3] / "scripts" / "validate"
           / "ocean_fidelity" / "dino_1226" / "hpg_tendency_compare.py")


def _load():
    spec = importlib.util.spec_from_file_location("hpg_cmp", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


hpg = _load()


# ---------------------------------------------------------------- istate ----
def test_istate_case4_equator_is_the_unblended_profile():
    """At |phi| = 0 the meridional blend is the identity, so T/S must equal the
    depth-only profile (usrdef_istate.F90:174: the (phiMAX-|phi|)/phiMAX factor
    is 1 there)."""
    gdept = np.array([5.0, 100.0, 1000.0, 3000.0])
    gphit = np.array([[0.0, 60.0]])          # col 0 = equator, col 1 sets phiMAX
    tmask = np.ones((1, 2, 4))
    t, s = hpg.nemo_istate_case4(gdept, gphit, tmask)
    t1d, s1d = hpg._istate_profiles_1d(gdept)
    np.testing.assert_allclose(t[0, 0], t1d, rtol=1e-14)
    np.testing.assert_allclose(s[0, 0], s1d, rtol=1e-14)


def test_istate_case4_rejects_degenerate_phi_max():
    """A non-positive MAXVAL(gphit) must RAISE, not hand back an all-NaN
    state through a divide-by-zero in the blend."""
    with pytest.raises(ValueError, match="MAXVAL"):
        hpg.nemo_istate_case4(np.array([5.0]), np.array([[0.0]]),
                              np.ones((1, 1, 1)))


def test_istate_case4_pole_collapses_to_the_abyssal_anchor():
    """At |phi| = phiMAX the blend collapses every level onto zTbot/zSbot --
    the isothermal polar column that drives DINO's deep convection."""
    gdept = np.array([5.0, 100.0, 1000.0, 3000.0])
    gphit = np.array([[0.0, 60.0]])
    tmask = np.ones((1, 2, 4))
    t, s = hpg.nemo_istate_case4(gdept, gphit, tmask)
    assert np.allclose(t[0, 1], t[0, 1][0])
    assert np.allclose(s[0, 1], s[0, 1][0])
    # and the anchor is the MINIMUM of the profile (NEMO's MINVAL)
    t1d, s1d = hpg._istate_profiles_1d(gdept)
    np.testing.assert_allclose(t[0, 1][0], t1d.min(), rtol=1e-14)
    np.testing.assert_allclose(s[0, 1][0], s1d.min(), rtol=1e-14)


def test_istate_case4_anchor_ignores_dry_cells():
    """zTbot = MINVAL(T + 100*(1-tmask)) takes the min over WET cells only.
    A dry deepest level must NOT set the anchor -- that was the documented
    full-step trap (deepest reference level globally dry)."""
    gdept = np.array([5.0, 100.0, 1000.0, 3000.0])
    gphit = np.array([[60.0]])
    wet = np.ones((1, 1, 4))
    wet[0, 0, 3] = 0.0                      # deepest level DRY
    t_wet, _ = hpg.nemo_istate_case4(gdept, gphit, wet)
    t1d, _ = hpg._istate_profiles_1d(gdept)
    # anchor is the min over the WET levels (0..2), not t1d[3]
    np.testing.assert_allclose(t_wet[0, 0, 0], t1d[:3].min(), rtol=1e-14)
    assert not np.isclose(t_wet[0, 0, 0], t1d.min())


# ------------------------------------------------------------------ dump ----
def test_read_dump_index_order(tmp_path):
    """The Fortran write is jk outer / jj / ji fastest; read_dump must return
    (n_lat, n_lon, nlev). A transposed reader still correlates ~1 on a smooth
    field, so pin it on an asymmetric one."""
    jpk, jpj, jpi = 3, 4, 5
    raw = np.arange(jpk * jpj * jpi, dtype=np.float64)
    p = tmp_path / "d.bin"
    raw.tofile(p)
    out = hpg.read_dump(str(p), jpk, jpj, jpi)
    assert out.shape == (jpj, jpi, jpk)
    # element (jj=1, ji=2, jk=0) is raw[0*jpj*jpi + 1*jpi + 2]
    assert out[1, 2, 0] == raw[1 * jpi + 2]
    assert out[3, 4, 2] == raw[2 * jpj * jpi + 3 * jpi + 4]


def test_read_dump_rejects_wrong_size(tmp_path):
    p = tmp_path / "d.bin"
    np.arange(7, dtype=np.float64).tofile(p)
    with pytest.raises(ValueError, match="expected"):
        hpg.read_dump(str(p), 3, 4, 5)


# ----------------------------------------------------------- corr / ratio ----
def test_corr_ratio_is_rms_ratio_over_wet_points():
    """ratio = rms(lego)/rms(nemo) over the wet mask (the #1226 convention),
    and dry points must not contribute."""
    nemo = np.array([[[1.0, 2.0]], [[3.0, 4.0]]])
    lego = 2.0 * nemo
    wet = np.ones_like(nemo, dtype=bool)
    c, r, n = hpg.corr_ratio(lego, nemo, wet)
    assert n == 4
    assert c == pytest.approx(1.0)
    assert r == pytest.approx(2.0)
    # poisoning a DRY point changes nothing
    wet2 = wet.copy()
    wet2[0, 0, 0] = False
    lego2 = lego.copy()
    lego2[0, 0, 0] = 1e6
    c2, r2, n2 = hpg.corr_ratio(lego2, nemo, wet2)
    assert n2 == 3
    assert r2 == pytest.approx(2.0)


def test_alignment_scan_finds_a_planted_offset():
    """The scan must recover a known shift with a sharp corr peak -- the guard
    against the offset-artifact class that produced false #1226 findings."""
    rng = np.random.default_rng(0)
    n_lat, n_lon, nlev = 6, 5, 3
    lego = rng.standard_normal((n_lat, n_lon, nlev))
    halo = 2
    full = np.zeros((n_lat + 2 * halo, n_lon + 2 * halo, nlev))
    # plant lego at offset (halo+1, halo-1) -> scan should report dj=1, di=-1
    full[halo + 1:halo + 1 + n_lat, halo - 1:halo - 1 + n_lon, :] = lego
    wet = np.ones_like(lego, dtype=bool)
    scan = hpg.alignment_scan(lego, full, wet, halo, span=1)
    dj, di, c, r, _ = scan[0]
    assert (dj, di) == (1, -1)
    assert c == pytest.approx(1.0)
    assert r == pytest.approx(1.0)
    assert c - scan[1][2] > 1e-3          # peak is sharp
