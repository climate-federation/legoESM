"""NEMO 5.0.1 ORCA1 oracle for ``si3_ice_update_flx_columns`` (``ice_update_flx``, ln_cndflx=F, jpl=1).

Fixture ``fixtures/orca1_t4/si3_update_flx_orca1.npz`` (+ ``.json`` provenance): an 8x8 interior window of the
tier-4 U4 record (NEMO 5.0.1 SI3 ORCA1, ice step kt 17281, nn_fsbc 96) holding 32 lead-warming (fhld > 0) and
30 other ice cells.  The routine is pointwise, so the window needs no halo (``nn_hls = 0``).  Pass rule = the tier-4
class rule rel <= max(1e-12, NEMO seed spread); qsr per cell |d| <= 1e-12 * max(|a|, |b|, 1 W/m2) because NEMO's
night-cell qsr is a ~1e-17 cancellation residue (U4 per-cell criterion).
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.ice.config import SI3_HARD_SWITCHES, SI3SwitchConfig
from legoesm.ice.si3_finalize import SI3Fluxes, SI3UpdateFluxForcing, si3_ice_update_flx_columns
from legoesm.ice.si3_orca2_state import SI3ORCA2StateConfig

jax.config.update("jax_enable_x64", True)

_FIX = Path(__file__).resolve().parent / "fixtures" / "orca1_t4" / "si3_update_flx_orca1"
_SPREAD = dict(qns=4.7e-12, emp=2.1e-14, sfx=8.9e-11, fwfice=1.5e-14, wfx_ice=1.8e-14, wfx_snw=1.8e-14,
               snwice_mass=6.4e-16, snwice_mass_b=6.4e-16, snwice_fmass=1.1e-9, fr_i=0.0, tn_ice=0.0)


def _rel(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
    d = np.maximum(np.abs(a), np.abs(b))
    return float(np.max(np.where(d > 0, np.abs(a - b) / np.where(d > 0, d, 1.0), 0.0)))


def _load():
    d = dict(np.load(str(_FIX) + ".npz"))
    assert json.loads(Path(str(_FIX) + ".json").read_text())["oracle"].startswith("NEMO 5.0.1")
    return d


def _config(d):
    cfg = SI3ORCA2StateConfig()._replace(nn_hls=0, ln_icethd=True,   # ORCA1 ln_icethd = T
                                         rDt_ice=1.0 / float(d["in_r1_Dt_ice"]))
    c = cfg.ice_constants
    # NEMO's rhos/rhoi/rhow are the constant set the routine uses (no override)
    assert (c.rho_snow, c.rho_ice, constants.rho_water) == (
        float(d["in_rhos"]), float(d["in_rhoi"]), float(d["in_rhow"]))
    return cfg


def _call(d, switches, **over):
    g = lambda k: jnp.asarray(over.get(k, d["in_" + k]))  # noqa: E731
    z = jnp.zeros_like(g("at_i"))
    fl = {f: (g(f) if "in_" + f in d else z) for f in SI3Fluxes._fields}
    fo = {f: (g(f) if "in_" + f in d else z) for f in SI3UpdateFluxForcing._fields}
    fo["tmask"] = jnp.asarray(d["in_tmask1"])
    # jpl=1: category axis of a_i_b / qsr_ice / qtr_ice_bot / qevap_ice / t_su
    for k in ("a_i_b", "qsr_ice", "qtr_ice_bot", "qevap_ice"):
        fo[k] = fo[k][..., None]
    return si3_ice_update_flx_columns(
        at_i=g("at_i"), vt_i=g("vt_i"), vt_s=g("vt_s"), vt_ip=g("vt_ip"), vt_il=g("vt_il"),
        t_su=g("t_su")[..., None], fluxes=SI3Fluxes(**fl), forcing=SI3UpdateFluxForcing(**fo),
        config=_config(d), switches=switches)


def test_window_scores_both_branches():
    d = _load()
    ice = d["in_at_i"] > 0
    assert int((d["in_fhld"] > 0).sum()) >= 30 and int((ice & (d["in_fhld"] <= 0)).sum()) >= 30


def test_hard_mode_matches_nemo_orca1():
    d = _load()
    _, out = _call(d, SI3_HARD_SWITCHES)
    for k, sp in _SPREAD.items():
        got = np.asarray(getattr(out, k))
        if k == "tn_ice":
            got = got[..., 0]
        assert _rel(got, d["out_" + k]) <= max(1e-12, sp), k
    a, b = np.asarray(out.qsr), d["out_qsr"]
    assert np.all(np.abs(a - b) <= 1e-12 * np.maximum(np.maximum(np.abs(a), np.abs(b)), 1.0))
    assert out.alb_ice is None


def test_lead_warming_branch_is_load_bearing():
    """Dropping the fhld>0 branch (fhld forced to 0) breaks qsr in exactly the 32 warming cells."""
    d = _load()
    _, out = _call(d, SI3_HARD_SWITCHES, fhld=np.zeros_like(d["in_fhld"]))
    a, b = np.asarray(out.qsr), d["out_qsr"]
    bad = np.abs(a - b) > 1e-12 * np.maximum(np.maximum(np.abs(a), np.abs(b)), 1.0)
    assert int(bad.sum()) > 0 and np.all((d["in_fhld"] > 0)[bad])


def test_smooth_default_keeps_column_heat_and_matches_hard_off_ramp():
    d = _load()
    _, hard = _call(d, SI3_HARD_SWITCHES)
    _, smooth = _call(d, SI3SwitchConfig())
    tot_h = np.asarray(hard.qsr) + np.asarray(hard.qns)
    tot_s = np.asarray(smooth.qsr) + np.asarray(smooth.qns)
    assert _rel(tot_s, tot_h) <= 1e-13                   # the blend only moves heat between SW and non-solar
    w_f = SI3SwitchConfig().relative_width * SI3SwitchConfig().flux_scale_W_m2
    off_ramp = (d["in_fhld"] <= 0) | (d["in_fhld"] >= w_f)
    np.testing.assert_array_equal(np.asarray(smooth.qsr)[off_ramp], np.asarray(hard.qsr)[off_ramp])


def test_blend_conserves_column_heat_inside_both_ramps():
    """Cells placed inside the fhld ramp and inside the at_i ramp: qsr moves, qsr + qns does not."""
    d = _load()
    sw = SI3SwitchConfig()
    w_f, w_a = sw.relative_width * sw.flux_scale_W_m2, sw.relative_width * sw.ice_switch_scale
    fh = d["in_fhld"].copy()
    at = d["in_at_i"].copy()
    warm = np.argwhere(d["in_fhld"] > 0)
    (i1, j1), (i2, j2) = warm[0], warm[1]
    fh[i1, j1] = 0.4 * w_f                      # fhld ramp
    at[i2, j2] = 0.6 * w_a                      # at_i ramp
    _, hard = _call(d, SI3_HARD_SWITCHES, fhld=fh, at_i=at)
    _, smooth = _call(d, sw, fhld=fh, at_i=at)
    _, zq = _call(d, SI3_HARD_SWITCHES, fhld=np.where(np.ones_like(fh, bool), -1.0, fh), at_i=at)
    _, wq = _call(d, SI3_HARD_SWITCHES, fhld=np.where(np.ones_like(fh, bool), 1.0, fh),
                  at_i=np.where(at > 0, np.maximum(at, 1.0e-6), at))
    # pinned per-case gap: qsr = zqsr + (warm - zqsr) * C1(x), C1(x) = 3x^2 - 2x^3 at x = 0.4 and 0.6
    for (i, j), weight in (((i1, j1), 0.352), ((i2, j2), 0.648)):
        z, wv = float(zq.qsr[i, j]), float(wq.qsr[i, j])
        assert z != wv
        np.testing.assert_allclose(float(smooth.qsr[i, j]), z + (wv - z) * weight, rtol=1e-12, atol=1e-12)
    for i, j in ((i1, j1), (i2, j2)):
        assert float(smooth.qsr[i, j]) != float(hard.qsr[i, j])          # the ramp is active here
        np.testing.assert_allclose(float(smooth.qsr[i, j] + smooth.qns[i, j]),
                                   float(hard.qsr[i, j] + hard.qns[i, j]), rtol=1e-13, atol=0)


def test_smooth_blend_inside_the_ramp_and_gradient():
    d = _load()
    i, j = np.argwhere(d["in_fhld"] > 0)[0]
    w_f = SI3SwitchConfig().relative_width * SI3SwitchConfig().flux_scale_W_m2

    def qsr_at(f):
        fh = jnp.asarray(d["in_fhld"]).at[i, j].set(f)
        return _call(d, SI3SwitchConfig(), fhld=fh)[1].qsr[i, j]

    lo, hi, mid = float(qsr_at(0.0)), float(qsr_at(w_f)), float(qsr_at(0.5 * w_f))
    assert lo != hi and min(lo, hi) < mid < max(lo, hi)  # strictly between the two NEMO forms
    x = jnp.asarray(0.3 * w_f)
    g = float(jax.grad(qsr_at)(x))
    eps = 1e-6 * w_f
    fd = (float(qsr_at(x + eps)) - float(qsr_at(x - eps))) / (2 * eps)
    assert np.isfinite(g) and g != 0.0
    np.testing.assert_allclose(g, fd, rtol=1e-6)


def test_jit_matches_eager():
    d = _load()
    eager = _call(d, SI3SwitchConfig())[1]
    jitted = jax.jit(lambda: _call(d, SI3SwitchConfig())[1])()
    for k in ("qsr", "qns", "emp", "sfx", "snwice_fmass"):
        np.testing.assert_array_equal(getattr(eager, k), getattr(jitted, k))
