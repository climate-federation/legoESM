"""Rung O3: NEMO `ice_var_itd_1cMc` fit and `ice_istate` fill (`si3_o3_istate_fill_fields`)."""

from __future__ import annotations

import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ice.constants_config import NEMO_SI3_CONSTANTS_CONFIG as C
from legoesm.ice.si3_orca2_state import (
    SI3ORCA2StateConfig,
    si3_hfn_category_bounds,
    si3_o3_istate_fill_fields,
    si3_var_itd_1cmc,
)
from legoesm.ocean.constants_config import NEMO_CONSTANTS_CONFIG

CFG = SI3ORCA2StateConfig.for_orca2_sas_ice()
HI_MAX = np.asarray(
    si3_hfn_category_bounds(jpl=CFG.jpl, rn_himean=CFG.rn_himean, opt_hips=CFG.opt_hips, rn_himax=CFG.rn_himax).hi_max
)
# Inexact columns: deck values, thin/thick ice, heavy snow (transfer active), no ice.
PHTI = np.array([2.0, 1.0, 0.37, 3.1415926535, 0.83, 0.0])
PHTS = np.array([0.2, 0.2, 0.61, 0.0123, 1.9, 0.0])
PATI = np.array([0.9, 0.9, 0.731, 0.55, 0.999, 0.0])
PATIP = PATI * np.array([0.2, 0.2, 0.13, 0.31, 0.07, 0.0])
PHTIP = np.array([0.05, 0.05, 0.021, 0.11, 0.3, 0.0])
PHTIL = np.array([0.0, 0.0, 0.004, 0.017, 0.0, 0.0])


@pytest.fixture(autouse=True)
def _libm():
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))


def _ref_1cmc(phti, phts, pati, patip, phtip, phtil):
    """Independent scalar oracle of `icevar.f90:1182-1327` in CPython binary64 (C `exp`, no FMA)."""
    jpl = len(HI_MAX) - 1
    hm = [float(x) for x in HI_MAX]
    rhos, rhoi, rho0 = C.rho_snow, C.rho_ice, NEMO_CONSTANTS_CONFIG.rho_0
    r1_rho0, r1_rhos = 1.0 / rho0, 1.0 / rhos
    z1 = 1.0 / phti if phti != 0.0 else 0.0
    a = [0.0] * jpl
    h = [0.0] * jpl
    s = [0.0] * jpl
    if phti > 0.0:
        for jl in range(1, jpl + 1):
            h0 = hm[jl - 1]
            e0 = math.exp(-(2.0 * h0 * z1))
            p0 = phti * phti + 2.0 * phti * h0 + 2.0 * h0 * h0
            if jl < jpl:
                h1 = hm[jl]
                e1 = math.exp(-(2.0 * h1 * z1))
                p1 = phti * phti + 2.0 * phti * h1 + 2.0 * h1 * h1
                pa = pati * z1 * ((phti + 2.0 * h0) * e0 - (phti + 2.0 * h1) * e1)
                zv = pati * z1 * (p0 * e0 - p1 * e1)
            else:
                pa = pati * z1 * (phti + 2.0 * h0) * e0
                zv = pati * z1 * p0 * e0
            if pa > 1e-6:
                a[jl - 1], h[jl - 1] = pa, zv / pa
    for jl in range(jpl):
        if a[jl] > 0.0:
            s[jl] = h[jl] * phts * z1
            zdh = max(0.0, (rhos * s[jl] + (rhoi - rho0) * h[jl]) * r1_rho0)
            h[jl] = min(hm[jl + 1], h[jl] + zdh)
            s[jl] = max(0.0, s[jl] - zdh * rhoi * r1_rhos)
    zf = patip / pati if pati != 0.0 else 0.0
    aip = [zf * x for x in a]
    den = phti * pati
    zp = (phtip * patip) / den if den != 0.0 else 0.0
    zl = (phtil * patip) / den if den != 0.0 else 0.0
    hip = [zp * (h[k] * a[k]) / aip[k] if aip[k] != 0.0 else 0.0 for k in range(jpl)]
    hil = [zl * (h[k] * a[k]) / aip[k] if aip[k] != 0.0 else 0.0 for k in range(jpl)]
    return a, h, s, aip, hip, hil


def _fit(plant=False):
    f = lambda *x: si3_var_itd_1cmc(  # noqa: E731
        phti=x[0], phts=x[1], pati=x[2], ptmi=x[3], ptms=x[3], ptmsu=x[3], psmi=x[3],
        patip=x[4], phtip=x[5], phtil=x[6], hi_max=jnp.asarray(HI_MAX), ice_constants=C,
        rho0=NEMO_CONSTANTS_CONFIG.rho_0, _plant_skip_snow_transfer=plant,
    )
    args = tuple(jnp.asarray(v) for v in (PHTI, PHTS, PATI, np.full(6, 270.0), PATIP, PHTIP, PHTIL))
    return jax.jit(f)(*args), f(*args)


def test_fit_bit_exact_vs_independent_oracle_and_jit_equals_eager():
    jit_out, eager_out = _fit()
    for a, b in zip(jax.tree_util.tree_leaves(jit_out), jax.tree_util.tree_leaves(eager_out)):
        assert np.array_equal(np.asarray(a).view(np.int64), np.asarray(b).view(np.int64))
    for c in range(len(PHTI)):
        ra, rh, rs, raip, rhip, rhil = _ref_1cmc(PHTI[c], PHTS[c], PATI[c], PATIP[c], PHTIP[c], PHTIL[c])
        for name, ref in (("a_i", ra), ("h_i", rh), ("h_s", rs), ("a_ip", raip), ("h_ip", rhip), ("h_il", rhil)):
            got = np.asarray(getattr(jit_out, name))[c]
            assert np.array_equal(got.view(np.int64), np.asarray(ref).view(np.int64)), (name, c, got, ref)


def test_snow_transfer_branch_live_and_mass_conserving():
    good, _ = _fit()
    bad, _ = _fit(plant=True)
    col = 4  # heavy snow: rhos*h_s > (rho0-rhoi)*h_i
    assert not np.array_equal(np.asarray(good.h_i)[col], np.asarray(bad.h_i)[col])
    m_good = C.rho_snow * np.asarray(good.h_s)[col] + C.rho_ice * np.asarray(good.h_i)[col]
    m_bad = C.rho_snow * np.asarray(bad.h_s)[col] + C.rho_ice * np.asarray(bad.h_i)[col]
    # `MIN( hi_max(jl), ph_i + zdh )` (icevar.f90:1282) clips without returning the snow: mass is
    # conserved only where the clip does not bind.
    clipped = np.asarray(good.h_i)[col] == HI_MAX[1:]
    assert clipped.any() and (~clipped).any()
    np.testing.assert_allclose(m_good[~clipped], m_bad[~clipped], rtol=1e-14)
    assert np.all(m_good[clipped] < m_bad[clipped])
    assert np.array_equal(np.asarray(good.h_i)[0], np.asarray(bad.h_i)[0])  # deck column: no transfer


def test_gamma_fit_conserves_area_and_volume_without_snow():
    out, _ = _fit()
    for c in (0, 1, 3):  # negligible-snow columns, no epsi06 truncation
        np.testing.assert_allclose(np.sum(np.asarray(out.a_i)[c]), PATI[c], rtol=1e-13)
    np.testing.assert_allclose(np.sum(np.asarray(out.a_i * out.h_i)[3]), PATI[3] * PHTI[3], rtol=1e-9)
    assert np.all(np.asarray(out.a_i)[5] == 0.0) and np.all(np.asarray(out.h_i)[5] == 0.0)


def _fill_inputs(scale=1.0, ati=0.9):
    shape = (6, 6)
    tm = np.ones(shape)
    tm[0, :] = 0.0
    zm = tm.copy()
    zm[1, :] = 0.0
    return dict(
        zht_i_ini=jnp.asarray(2.0 * scale * zm), zht_s_ini=jnp.asarray(0.2 * zm), zat_i_ini=jnp.asarray(ati * zm),
        ztm_i_ini=jnp.asarray(270.0 * zm), ztm_s_ini=jnp.asarray(270.0 * zm), zt_su_ini=jnp.asarray(270.0 * zm),
        zsm_i_ini=jnp.asarray(6.3 * zm), zapnd_ini=jnp.asarray(0.2 * ati * zm), zhpnd_ini=jnp.asarray(0.05 * zm),
        zhlid_ini=jnp.asarray(0.0 * zm), tmask=jnp.asarray(tm), sss_m=jnp.asarray(np.full(shape, 33.7)),
        rn_amax_2d=jnp.asarray(np.full(shape, 0.997)),
    )


def test_amax_cap_branch_live():
    good = si3_o3_istate_fill_fields(**_fill_inputs(ati=1.0), config=CFG)
    bad = si3_o3_istate_fill_fields(**_fill_inputs(ati=1.0), config=CFG, _plant_skip_amax=True)
    assert float(np.max(np.asarray(good.at_i))) <= 0.997 + 1e-15
    assert float(np.max(np.asarray(bad.at_i))) > 0.997
    good9 = si3_o3_istate_fill_fields(**_fill_inputs(ati=0.9), config=CFG)
    bad9 = si3_o3_istate_fill_fields(**_fill_inputs(ati=0.9), config=CFG, _plant_skip_amax=True)
    assert np.array_equal(np.asarray(good9.a_i), np.asarray(bad9.a_i))  # ORCA2 deck: cap never binds


def test_gradient_finite_and_central_fd():
    def loss(s):
        o = si3_o3_istate_fill_fields(**_fill_inputs(scale=s), config=CFG)
        return jnp.sum(o.v_i) + 1e-9 * jnp.sum(o.e_i) + jnp.sum(o.v_ip)

    s0, eps = 1.0, 1e-6
    g = float(jax.jit(jax.grad(loss))(s0))
    fd = (float(loss(s0 + eps)) - float(loss(s0 - eps))) / (2 * eps)
    assert np.isfinite(g) and g != 0.0
    assert abs(g / fd - 1.0) < 1e-5, (g, fd)


def test_refuses_missing_or_unsupported_inputs():
    inp = _fill_inputs()
    with pytest.raises(TypeError):
        si3_o3_istate_fill_fields(**inp)  # no config: no default
    with pytest.raises(TypeError):
        si3_o3_istate_fill_fields(**{k: v for k, v in inp.items() if k != "sss_m"}, config=CFG)
    with pytest.raises(ValueError, match="nn_icesal"):
        si3_o3_istate_fill_fields(**inp, config=CFG._replace(nn_icesal=1))


_O2_RECORD = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round7/oracle_o2_orca2_sas_ice")


@pytest.mark.skipif(not (_O2_RECORD / "o2_admission.json").is_file(), reason="admitted O2 record absent")
def test_o3_scores_zero_ulp_on_admitted_o2_record_and_plants_fire():
    from scripts.validate.ocean_fidelity.testcases.nemo_si3_o3_orca2_sas_gate import score_o3

    rep = score_o3(_O2_RECORD)
    assert rep["status"] == "PASS" and rep["max_ulp_wet_full"] == [0, 0] and rep["arrays_scored"] == 55
    assert rep["jit_vs_eager_max_ulp"] == 0
    for plant in ("1ulp-v_i", "native-exp"):
        with pytest.raises(ValueError, match="PLANT-FIRED"):
            score_o3(_O2_RECORD, plant=plant)



@pytest.mark.skipif(not (_O2_RECORD / "o2_admission.json").is_file(), reason="admitted O2 record absent")
def test_o2_admission_reads_the_fit_only_in_its_initial_state_comparison():
    """Perturbing the fit's output moves the O2 initial-state candidates and no step/restart candidate."""
    from scripts.validate.ocean_fidelity.testcases.nemo_si3_o3_orca2_sas_gate import o2_admission_fit_dependence

    dep = o2_admission_fit_dependence(_O2_RECORD)
    assert set(dep) == {"o2_istate", "kt1", "kt2", "rst_entry", "rst_step1", "rst_vs_kt2", "restart_variables"}
    # vt_il cannot move: the deck's lid volume is identically zero.
    assert sorted(dep["o2_istate"]["moved"]) == sorted((
        "vt_i", "vt_s", "at_i", "at_ip", "vt_ip", "snwice_mass", "snwice_mass_b", "zsshadj", "ssh_kbb", "ssh_kmm",
    ))
    assert [k for k, r in dep.items() if r["depends_on_fit"]] == ["o2_istate"]


@pytest.mark.skipif(not (_O2_RECORD / "o2_admission.json").is_file(), reason="admitted O2 record absent")
def test_o3_built_entry_steps_zero_ulp_and_perturbed_fit_refuses():
    from scripts.validate.ocean_fidelity.testcases.nemo_si3_o3_orca2_sas_gate import score_o3_entry_steps

    rep = score_o3_entry_steps(_O2_RECORD)
    assert rep["status"] == "PASS" and rep["max_ulp_kt1"] == [0, 0] and rep["max_ulp_kt2"] == [0, 0]
    with pytest.raises(ValueError, match="PLANT-FIRED: prognostic field"):
        score_o3_entry_steps(_O2_RECORD, bump_fit=True)


O3_PARENT_COMMIT = "62a28e1324b203e5a6930b734680a4af8196630a"


def test_o3_parent_vs_current_other_callers_bit_identical(monkeypatch):
    """Non-SI3, SCM, lane-3 `si3_bl99` and O1 callers match this round's parent (O2 harness, its P1 plant)."""
    import importlib.util

    path = Path(__file__).resolve().parents[2] / "ocean/fidelity/test_nemo_si3_o2_orca2_sas_gate.py"
    spec = importlib.util.spec_from_file_location("o2_tests_for_o3_parent", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "PARENT_COMMIT", O3_PARENT_COMMIT)
    mod.test_o2_parent_vs_current_all_callers_bit_identical(monkeypatch)
