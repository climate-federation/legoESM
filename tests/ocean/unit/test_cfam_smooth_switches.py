"""RULE 5 smooth switches of double diffusion, PP81 and Simmons tidal mixing.

Every switch on traced state is a compact C1 ramp / C1 min-max whose width is a
config field (0 = the reference's hard form).  Per switch, three tests:

1. hard limit (widths 0) = the COMPILED oracle at fixture points around the
   switch, rel 1e-12: CVMix fce422195a (ddiff aligned to Rc 1.6, exp1 2; shear
   'PP'; tidal 'Simmons') and NEMO 5.0.1 ORCA1 ``zdf_ddm``.  Values pinned
   from oracle_audit/cvmix_family_oracle/pins.py (job 11291996);
2. the default-width gap, per case, stored (rel 1e-6, abs floor 1e-12);
3. ``jax.test_util.check_grads`` at the default inside each ramp, with the
   input scaled by the width and the output to O(1), plus a nonzero slope.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.ocean.physics.vertical_mixing import double_diffusion as dd
from legoesm.ocean.physics.vertical_mixing import tidal as td
from legoesm.ocean.physics.vertical_mixing.config import RichardsonVerticalMixingConfig
from legoesm.ocean.physics.vertical_mixing.richardson import _pp81_diffusivities

# ---- pinned oracle values and measured default-width outputs ----
DDM_NUM = [9.970000000000001e-06, 9.999700000000002e-06, 1.00003e-05, 1.003e-05, 1.0500000000000001e-05, -9.970000000000001e-06, -9.999700000000002e-06, -1.00003e-05, -1.003e-05, -1.0500000000000001e-05, 3.0000000000000004e-08, 1e-08, -1e-08, -3.0000000000000004e-08, -4.970000000000001e-06, -4.9950000000000005e-06, -5.0049999999999995e-06, -5.03e-06, -9e-06]
DDM_DEN = [1e-05, 1e-05, 1e-05, 1e-05, 1e-05, -1e-05, -1e-05, -1e-05, -1e-05, -1e-05, -1e-05, -1e-05, -1e-05, -1e-05, -1e-05, -1e-05, -1e-05, -1e-05, -1e-05]
DDM_N2 = [-2.941847999999972e-07, -2.9418479999898384e-09, 2.9418479999898384e-09, 2.941847999999972e-07, 4.903080000000003e-06, 2.941847999999972e-07, 2.9418479999898384e-09, -2.9418479999898384e-09, -2.941847999999972e-07, -4.903080000000003e-06, 9.835578480000001e-05, 9.815966160000001e-05, 9.796353840000001e-05, 9.776741520000002e-05, 4.93249848e-05, 4.9079830800000006e-05, 4.898176920000001e-05, 4.873661520000001e-05, 9.806160000000007e-06]
# CVMix Tdiff/Sdiff (cfg aligned); for R > 1 Tdiff/R (port heat ratio 0.7/R vs CVMix 0.7, documented)
DDM_AVT_ORACLE = [0.0, 0.0, 6.999789953801386e-05, 6.978539394940069e-05, 6.528740051297439e-05, 0.0001346375791089086, 0.00013563675612871095, 0.0, 0.0, 0.0, 0.0, 0.0, 1.3635000000000002e-06, 1.3635000000000002e-06, 1.955592501570877e-05, 1.98419911823062e-05, 1.9957213284479178e-05, 2.0247261974095308e-05, 0.00010376946726422928]
DDM_AVS_ORACLE = [0.0, 0.0, 9.999999925000001e-05, 9.999250018749842e-05, 9.79311007694616e-05, 0.00013389034054485418, 0.00013562922828874586, 0.0, 0.0, 0.0, 0.0, 0.0, 2.04525e-10, 6.13575e-10, 1.457894209921089e-06, 1.4866611893342917e-06, 1.515251418624082e-06, 1.6309169520133795e-06, 8.457211582034685e-05]
DDM_AVT_DEFAULT = [0.0, 1.1305820912889251e-05, 9.129005371015164e-05, 7.674027549320357e-05, 6.528740051297439e-05, 0.00012793662896949572, 9.180895933084476e-05, 1.1241920242834745e-05, 0.0, 0.0, 1.4180400000000004e-07, 4.799520000000002e-07, 8.835480000000002e-07, 1.2216960000000004e-06, 1.955592501570877e-05, 1.98419911823062e-05, 1.9957213284479178e-05, 2.0247261974095308e-05, 0.00010376946726422928]
DDM_AVS_DEFAULT = [0.0, 1.2935134797022744e-05, 0.00010476986915183981, 0.00010388472240449454, 9.79311007694616e-05, 0.00013036496514768917, 0.00010503980857904546, 1.2901893086806674e-05, 0.0, 0.0, 0.0, 0.0, 4.771159200000002e-11, 4.618010880000001e-10, 1.4475217472927565e-06, 1.4794890535965577e-06, 1.5080376344172395e-06, 1.6201778042623173e-06, 8.457211582034685e-05]
DDM_KCAP = 0.00012
DDM_AVT_KCAP_DEFAULT = [0.0, 1.1305820912889251e-05, 8.902372177524594e-05, 7.651311892458217e-05, 6.528740051297439e-05, 0.00011115336408775067, 8.943080579537255e-05, 1.1241920242834745e-05, 0.0, 0.0, 1.4180400000000004e-07, 4.799520000000002e-07, 8.835480000000002e-07, 1.2216960000000004e-06, 1.955592501570877e-05, 1.98419911823062e-05, 1.9957213284479178e-05, 2.0247261974095308e-05, 9.806758266768001e-05]
DDM_AVS_KCAP_DEFAULT = [0.0, 1.2935134797022744e-05, 9.872515014765951e-05, 9.814385034234627e-05, 9.403036881849252e-05, 0.00011214532006128053, 9.890086765276472e-05, 1.2901893086806674e-05, 0.0, 0.0, 0.0, 0.0, 4.771159200000002e-11, 4.618010880000001e-10, 1.4475217472927565e-06, 1.4794890535965577e-06, 1.5080376344172395e-06, 1.6201778042623173e-06, 8.351038302293884e-05]
# point indices per switch
DDM_CASES = {"S1_R1_edge": [0, 1, 2, 3, 5, 6, 7, 8], "S2_R0_edge": [10, 11, 12, 13],
             "S3_R05_join": [14, 15, 16, 17], "S4_N2_gate": [1, 2, 6, 7]}

NEMO_ZDT = [-9.970000000000001e-06, -1.003e-05, -4.970000000000001e-06, -5.03e-06, -3e-06, -9.990000000000001e-06, 1.2e-05, 1.2e-05, -7e-06, -7e-06]
NEMO_ZDS = [-1e-05, -1e-05, -1e-05, -1e-05, -1e-05, -1e-05, 1e-05, 1e-05, -1e-05, -1e-05]
NEMO_RN2 = [1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, -3e-09, 3e-09, -3e-09, 3e-09]
NEMO_ORACLE = [0.0001346375791089086, 6.579765732416344e-05, 1.9555925015708763e-05, 2.0247261974095325e-05, 5.027201388127168e-06, 0.00013531009017653687, 0.0, 4.951986183074266e-05, 0.0, 5.2441171668212294e-05, 0.00013389034054485415, 9.427864328019418e-05, 1.4578942099210883e-06, 1.6309169520133797e-06, 2.2622406246572255e-07, 0.0001350597665097103, 0.0, 8.489119170984456e-05, 0.0, 2.333632139235447e-05, 0.0001346375791089086, 9.427864328019418e-05, 1.9555925015708763e-05, 2.0247261974095325e-05, 5.027201388127168e-06, 0.00013531009017653687, 0.0, 8.489119170984456e-05, 0.0, 5.2441171668212294e-05]  # compiled zdf_ddm avt | avs | avm
NEMO_DEFAULT = [0.00012753336378739177, 7.316726347699105e-05, 1.9555925015708767e-05, 2.0247261974095304e-05, 5.027201388127168e-06, 0.00011096601120802644, 5.154820428391461e-06, 4.437454921562462e-05, 5.458917149810436e-06, 4.6992323223152406e-05, 0.00011731416281641803, 8.595997437164884e-05, 1.4475217472927565e-06, 1.620177804262319e-06, 2.2622406246572255e-07, 8.994326148343185e-05, 8.836835020099647e-06, 7.607065579821363e-05, 2.528719343549348e-07, 1.8738786251133088e-05, 0.00012753336378739177, 8.595997437164884e-05, 1.9555925015708767e-05, 2.0247261974095304e-05, 5.027201388127168e-06, 0.00011096601120802644, 8.836835020099647e-06, 7.607065579821363e-05, 5.458917149810436e-06, 4.6992323223152406e-05]
NEMO_CASES = {"S1_R1_edge": [0, 1], "S3_R05_join": [2, 3], "S4_N2_gate": [6, 7, 8, 9]}

PP_RI = [-0.004, -0.001, 0.0, 0.001, 0.004, 0.3]
PP_ORACLE = [0.0051, 0.0051, 0.0051, 0.005050372515531795, 0.004905843906189927, 0.0009000000000000001, 0.00511, 0.00511, 0.00511, 0.005035246284111239, 0.004819650888421496, 0.00037000000000000005]  # CVMix Mdiff | Tdiff
PP_DEFAULT = [0.005097500937187597, 0.005060238726369423, 0.005038081092550441, 0.005011200582271942, 0.004903488966175691, 0.0009000000000000001, 0.0051062268804674795, 0.005050078412718548, 0.005016788663404164, 0.004976502063698654, 0.0048161641422942315, 0.00037000000000000005]

TD_ZW = [-0.0, -428.57142857142856, -857.1428571428571, -1285.7142857142858, -1714.2857142857142, -2142.8571428571427, -2571.4285714285716, -3000.0]
TD_N2 = [1e-05, 1e-05, -1e-08, 3e-08, 1.003e-07, 9.97e-08, 2e-09, 1e-05]  # at interfaces 1..nl+1
TD_E = 41.42406511645347
TD_ORACLE = [0.005, 0.0, 0.005, 0.005, 0.005, 0.005]  # CVMix Tdiff at interfaces 2..nl
TD_ORACLE_FLOOR_GATED = [0.005, 0.0, 0.005, 0.005, 0.005, 0.005]  # CVMix fed max(N2, N_squared_min), K=0 at N2<=0
TD_DEFAULT = [0.00499775, 0.0, 0.00108, 0.005, 0.00499986527, 5.92e-06]
TD_E_SMALL = 0.001
TD_ORACLE_FLOOR_GATED_SMALL = [1.2094418995131522e-07, 0.0, 6.715677560514366e-05, 0.0001577761361585061, 0.0003729019568029751, 0.0008787130482114862]
TD_DEFAULT_SMALL = [1.2094418995131517e-07, 0.0, 1.450586353071103e-05, 0.00015774468158590672, 0.00037281734511782343, 1.0403962490823994e-06]
TD_CASES = {"S7_N2_floor": [3, 4], "S8_K_cap": [0], "S9_N2_gate": [1, 2, 5]}

HARD_DD = dict(regime_width=0.0, n2_gate_width=0.0, k_max_width=0.0, avm_max_width=0.0)
HARD_TD = dict(n2_floor_width=0.0, k_max_width=0.0, n2_gate_width=0.0)


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    den = np.maximum(np.abs(a), np.abs(b))
    return np.where(den > 0, np.abs(a - b) / np.where(den > 0, den, 1.0), 0.0)


def _gap_ok(got, stored):
    got, stored = np.asarray(got), np.asarray(stored)
    return np.all((np.abs(got - stored) <= 1e-6 * np.abs(stored)) | (np.abs(got - stored) <= 1e-12))


def _ddm(cfg, num=DDM_NUM, n2=DDM_N2):
    a, s = dd.compute_ddm_diffusivity(jnp.asarray(n2), jnp.asarray(num), jnp.asarray(DDM_DEN), cfg)
    return np.asarray(a), np.asarray(s)


def _nemo(cfg):
    z = jnp.zeros(len(NEMO_ZDT))
    out = dd.nemo_zdf_ddm(z, z, jnp.asarray(NEMO_ZDT), jnp.asarray(NEMO_ZDS), jnp.asarray(NEMO_RN2),
                          jnp.ones(len(NEMO_ZDT)), cfg)
    return np.concatenate([np.asarray(x) for x in out])


def _tidal(N2_int, E=TD_E, **kw):
    zw = np.asarray(TD_ZW)
    nl = zw.size - 1
    zt = 0.5 * (zw[:-1] + zw[1:])
    return td.compute_tidal_diffusivity(
        jnp.asarray(E), jnp.asarray(-zw[1:nl]), jnp.asarray(zt[:-1] - zt[1:]),
        jnp.asarray(-zw[nl]), N2_int, config=td.TidalMixingConfig(enabled=True, **kw))


TD_N2_INT = jnp.asarray(TD_N2[1:-1])


# ---------------- 1. hard limit = compiled oracle ----------------
@pytest.mark.parametrize("switch", sorted(DDM_CASES))
def test_ddm_hard_limit_matches_cvmix(switch):
    idx = DDM_CASES[switch]
    a, s = _ddm(dd.DoubleDiffusionConfig(enabled=True, **HARD_DD))
    assert _rel(a[idx], np.asarray(DDM_AVT_ORACLE)[idx]).max() <= 1e-12
    assert _rel(s[idx], np.asarray(DDM_AVS_ORACLE)[idx]).max() <= 1e-12


def test_ddm_k_max_cap_hard_limit_is_capped_cvmix():
    a, s = _ddm(dd.DoubleDiffusionConfig(enabled=True, k_max=DDM_KCAP, **HARD_DD))
    assert _rel(a, np.minimum(DDM_AVT_ORACLE, DDM_KCAP)).max() <= 1e-12
    assert _rel(s, np.minimum(DDM_AVS_ORACLE, DDM_KCAP)).max() <= 1e-12


@pytest.mark.parametrize("switch", sorted(NEMO_CASES))
def test_nemo_literal_hard_limit_matches_compiled_nemo(switch):
    k = len(NEMO_ZDT)
    idx = np.concatenate([np.asarray(NEMO_CASES[switch]) + j * k for j in range(3)])
    got = _nemo(dd.DoubleDiffusionConfig(enabled=True, form="nemo_literal", **HARD_DD))
    assert _rel(got[idx], np.asarray(NEMO_ORACLE)[idx]).max() <= 1e-12


def test_pp81_ri_clip_hard_limit_matches_cvmix():
    got = np.concatenate([np.asarray(x) for x in _pp81_diffusivities(
        jnp.asarray(PP_RI), RichardsonVerticalMixingConfig(ri_clip_width=0.0))])
    assert _rel(got, PP_ORACLE).max() <= 1e-12


def test_tidal_hard_limit_is_cvmix():
    """S8 cap and S9 gate: all widths 0 and no floor = compiled CVMix Simmons."""
    got = np.asarray(_tidal(TD_N2_INT, N_squared_min=0.0, **HARD_TD))
    assert _rel(got, TD_ORACLE).max() <= 1e-12


def test_tidal_floor_hard_limit_is_cvmix_on_floored_n2():
    """S7: hard floor at N_squared_min = CVMix fed max(N2, N_squared_min), K = 0 at N2 <= 0;
    both E values (the large one caps, the small one keeps K far below K_max)."""
    got = np.asarray(_tidal(TD_N2_INT, **HARD_TD))
    assert _rel(got, TD_ORACLE_FLOOR_GATED).max() <= 1e-12
    got = np.asarray(_tidal(TD_N2_INT, E=TD_E_SMALL, **HARD_TD))
    assert _rel(got, TD_ORACLE_FLOOR_GATED_SMALL).max() <= 1e-12
    assert np.all(got[TD_CASES["S7_N2_floor"]] < 0.1 * td.TidalMixingConfig().K_max)


# ---------------- 2. default-width gap, per case ----------------
@pytest.mark.parametrize("switch", sorted(DDM_CASES))
def test_ddm_default_gap_per_case(switch):
    idx = DDM_CASES[switch]
    a, s = _ddm(dd.DoubleDiffusionConfig(enabled=True))
    assert _gap_ok(a[idx], np.asarray(DDM_AVT_DEFAULT)[idx])
    assert _gap_ok(s[idx], np.asarray(DDM_AVS_DEFAULT)[idx])
    # the smoothing is not a no-op at these points
    assert _rel(np.r_[a[idx], s[idx]], np.r_[np.asarray(DDM_AVT_ORACLE)[idx],
                                               np.asarray(DDM_AVS_ORACLE)[idx]]).max() > 1e-6


def test_ddm_k_max_cap_default_gap():
    a, s = _ddm(dd.DoubleDiffusionConfig(enabled=True, k_max=DDM_KCAP))
    assert _gap_ok(a, DDM_AVT_KCAP_DEFAULT) and _gap_ok(s, DDM_AVS_KCAP_DEFAULT)


@pytest.mark.parametrize("switch", sorted(NEMO_CASES))
def test_nemo_literal_default_gap_per_case(switch):
    k = len(NEMO_ZDT)
    idx = np.concatenate([np.asarray(NEMO_CASES[switch]) + j * k for j in range(3)])
    got = _nemo(dd.DoubleDiffusionConfig(enabled=True, form="nemo_literal"))
    assert _gap_ok(got[idx], np.asarray(NEMO_DEFAULT)[idx])
    assert _rel(got[idx], np.asarray(NEMO_ORACLE)[idx]).max() > 1e-6


def test_pp81_default_gap_per_case():
    got = np.concatenate([np.asarray(x) for x in _pp81_diffusivities(
        jnp.asarray(PP_RI), RichardsonVerticalMixingConfig())])
    assert _gap_ok(got, PP_DEFAULT)
    assert _rel(got, PP_ORACLE).max() > 1e-6


@pytest.mark.parametrize("switch", sorted(TD_CASES))
def test_tidal_default_gap_per_case(switch):
    idx = TD_CASES[switch]
    got = np.asarray(_tidal(TD_N2_INT))
    assert _gap_ok(got[idx], np.asarray(TD_DEFAULT)[idx])
    got = np.asarray(_tidal(TD_N2_INT, E=TD_E_SMALL))
    assert _gap_ok(got[idx], np.asarray(TD_DEFAULT_SMALL)[idx])


# ---------------- 3. gradients at the default, inside each ramp ----------------
def _check(f, x0=0.0):
    check_grads(f, (jnp.asarray(x0),), order=1, modes=("fwd", "rev"))
    assert float(jnp.max(jnp.abs(jax.jacfwd(f)(jnp.asarray(x0))))) > 0.0


@pytest.mark.parametrize("R0,den", [(1.0, -1e-5), (1.0, 1e-5), (0.0, -1e-5), (0.5, -1e-5)])
def test_ddm_gradient_through_regime_edges(R0, den):
    """S1 (R = 1), S2 (R = 0), S3 (R = 0.5): R swept across the ramp, N2 held stable."""
    cfg = dd.DoubleDiffusionConfig(enabled=True)
    w = cfg.regime_width

    def f(x):
        a, s = dd.compute_ddm_diffusivity(jnp.asarray(1e-5), (R0 + x * w) * den, jnp.asarray(den), cfg)
        return jnp.stack([a, s]) / 1e-4

    _check(f, 0.1)


def test_ddm_gradient_through_n2_gate():
    """S4: N2 swept across the gate ramp at a fingering R."""
    cfg = dd.DoubleDiffusionConfig(enabled=True)

    def f(x):
        a, s = dd.compute_ddm_diffusivity(x * cfg.n2_gate_width, jnp.asarray(1.2e-5), jnp.asarray(1e-5), cfg)
        return jnp.stack([a, s]) / 1e-4

    _check(f, 0.1)


def test_ddm_gradient_through_k_max_cap():
    """S5: k_max set to the fingering avs; rn_avts swept so avs crosses the cap ramp."""
    args = (jnp.asarray(1e-5), jnp.asarray(1.2e-5), jnp.asarray(1e-5))
    base = float(dd.compute_ddm_diffusivity(*args, dd.DoubleDiffusionConfig(enabled=True))[1])
    w = dd.DoubleDiffusionConfig().k_max_width
    ks = dd.DoubleDiffusionConfig().rn_avts

    def f(x):
        cfg = dd.DoubleDiffusionConfig(enabled=True, k_max=base, rn_avts=ks * (1.0 + x * w / base))
        return jnp.stack(dd.compute_ddm_diffusivity(*args, cfg)) / 1e-4

    _check(f, 0.1)


def test_ddm_k_max_width_bound():
    """smooth_min_c1 would dip below 0 if k_max_width > 2*k_max: refused."""
    with pytest.raises(ValueError, match="k_max_width"):
        dd.compute_ddm_diffusivity(jnp.asarray(1e-5), jnp.asarray(1.2e-5), jnp.asarray(1e-5),
                                   dd.DoubleDiffusionConfig(enabled=True, k_max=1e-5))
    with pytest.raises(ValueError, match="k_max_width"):
        _tidal(TD_N2_INT, K_max=1e-5)


def _ddm_main_formula(N2, a, b, cfg):
    """origin/main e78cd6a664's hybrid kernel, verbatim (the hard reference)."""
    eps = float(jnp.finfo(jnp.float32).eps)
    denom = jnp.where(b >= 0.0, jnp.maximum(b, eps), jnp.minimum(b, -eps))
    R = a / denom
    stable = N2 > 0.0
    Rc = cfg.rn_hsbfr
    finger = stable & (R > 1.0) & (R < Rc)
    Rf = jnp.clip(R, 1.0 + eps, Rc - eps)
    frac = (Rf - 1.0) / (Rc - 1.0)
    avs_f = cfg.rn_avts * (1.0 - frac * frac) ** 3
    avt_f = 0.7 * avs_f / jnp.where(finger, R, 1.0)
    dc = stable & (R > 0.0) & (R < 1.0)
    Rd = jnp.clip(R, eps, 1.0 - eps)
    avt_dc = 1.5e-6 * 0.909 * jnp.exp(4.6 * jnp.exp(-0.54 * (1.0 / Rd - 1.0)))
    avs_dc = jnp.where(Rd >= 0.5, avt_dc * (1.85 - 0.85 / Rd) * Rd, avt_dc * 0.15 * Rd)
    avt = jnp.where(finger, avt_f, jnp.where(dc, avt_dc, 0.0))
    avs = jnp.where(finger, avs_f, jnp.where(dc, avs_dc, 0.0))
    return jnp.clip(avt, 0.0, cfg.k_max), jnp.clip(avs, 0.0, cfg.k_max)


def test_ddm_width_zero_is_bitwise_main():
    rng = np.random.default_rng(0)
    R = np.concatenate([rng.uniform(-0.5, 2.0, 2000), [0.0, 0.5, 1.0, 1.6]])
    den = np.where(rng.random(R.size) < 0.5, 1.0, -1.0) * 10.0 ** rng.uniform(-7, -4, R.size)
    num = R * den
    N2 = constants.g * (num - den) * np.where(rng.random(R.size) < 0.9, 1.0, -1.0)
    cfg = dd.DoubleDiffusionConfig(enabled=True, **HARD_DD)
    args = (jnp.asarray(N2), jnp.asarray(num), jnp.asarray(den))
    for got, ref in zip(dd.compute_ddm_diffusivity(*args, cfg), _ddm_main_formula(*args, cfg)):
        assert np.array_equal(np.asarray(got), np.asarray(ref))


@pytest.mark.parametrize("which", ["regime_1", "regime_05", "gate"])
def test_nemo_literal_gradient_through_masks(which):
    cfg = dd.DoubleDiffusionConfig(enabled=True, form="nemo_literal")
    z = jnp.zeros(())

    def f(x):
        if which == "gate":
            zdt, zds, rn2 = jnp.asarray(1.2e-5), jnp.asarray(1e-5), x * cfg.n2_gate_width
        else:
            r0 = 1.0 if which == "regime_1" else 0.5
            zdt, zds, rn2 = (r0 + x * cfg.regime_width) * -1e-5, jnp.asarray(-1e-5), jnp.asarray(1e-5)
        return jnp.stack(dd.nemo_zdf_ddm(z, z, zdt, zds, rn2, jnp.ones(()), cfg)) / 1e-4

    _check(f, 0.1)


# S10 avm MAX alone smooth (other widths 0): at zrau = 0.999, heat - salt =
# 1.85*zavdt*(1 - zrau) ~ 2.5e-7, inside the (0, 1e-6) band.
NEMO_AVM_ONLY = [0.00013458983927485085, 9.427864328019418e-05, 1.9555925015708767e-05, 2.0247261974095304e-05, 5.027201388127168e-06, 0.00013516940461992327, 0.0, 8.489119170984456e-05, 0.0, 5.244117166821231e-05]
HARD_MASKS = dict(regime_width=0.0, n2_gate_width=0.0, k_max_width=0.0)


def test_nemo_literal_avm_max_hard_limit_and_gap():
    z = jnp.zeros(len(NEMO_ZDT))
    args = (z, z, jnp.asarray(NEMO_ZDT), jnp.asarray(NEMO_ZDS), jnp.asarray(NEMO_RN2), jnp.ones(len(NEMO_ZDT)))
    k = len(NEMO_ZDT)
    hard = np.asarray(dd.nemo_zdf_ddm(*args, dd.DoubleDiffusionConfig(enabled=True, form="nemo_literal", **HARD_DD))[2])
    assert _rel(hard, np.asarray(NEMO_ORACLE)[2 * k:]).max() <= 1e-12
    got = np.asarray(dd.nemo_zdf_ddm(*args, dd.DoubleDiffusionConfig(enabled=True, form="nemo_literal", **HARD_MASKS))[2])
    assert _gap_ok(got, NEMO_AVM_ONLY)
    assert _rel(got[5], np.asarray(NEMO_ORACLE)[2 * k + 5]) > 1e-6


def test_nemo_literal_avm_max_keeps_structural_zeros():
    """heat = salt = 0 (masked interface) must leave avm unchanged at the default."""
    z = jnp.zeros(2)
    cfg = dd.DoubleDiffusionConfig(enabled=True, form="nemo_literal")
    _, _, avm = dd.nemo_zdf_ddm(z, z, jnp.asarray([1.2e-5, -0.7e-5]), jnp.asarray([1e-5, -1e-5]),
                                jnp.asarray([1e-5, 1e-5]), jnp.zeros(2), cfg)
    assert np.all(np.asarray(avm) == 0.0)


def test_nemo_literal_gradient_through_avm_max():
    """S10: zrau = 0.999 puts heat - salt (~2.7e-7) inside the one-sided avm MAX band."""
    cfg = dd.DoubleDiffusionConfig(enabled=True, form="nemo_literal", **HARD_MASKS)
    z = jnp.zeros(())

    def f(x):
        zdt = (0.999 + x * 1e-4) * -1e-5
        return dd.nemo_zdf_ddm(z, z, zdt, jnp.asarray(-1e-5), jnp.asarray(1e-5), jnp.ones(()), cfg)[2] / 1e-4

    _check(f, 0.0)


def test_pp81_gradient_through_ri_clip():
    cfg = RichardsonVerticalMixingConfig()

    def f(x):
        return jnp.stack(_pp81_diffusivities(x * cfg.ri_clip_width, cfg)) / 1e-3

    _check(f, 0.1)


@pytest.mark.parametrize("i,which", [(3, "floor"), (2, "gate"), (0, "cap")])
def test_tidal_gradient_through_switches(i, which):
    """Output normalised by its own value at the ramp centre, so the floor/gate
    slopes are O(1) and cannot hide under check_grads' absolute tolerance."""
    cfg = td.TidalMixingConfig()
    scale = float(_tidal(TD_N2_INT.at[i].set(cfg.N_squared_min if which == "floor"
                                              else 0.5 * cfg.n2_gate_width), E=TD_E_SMALL)[i])

    def f(x):
        if which == "cap":
            return _tidal(TD_N2_INT, E=TD_E * (1.0 + x * cfg.k_max_width / cfg.K_max))[i] / cfg.K_max
        if which == "floor":
            n2 = TD_N2_INT.at[i].set(cfg.N_squared_min + x * cfg.n2_floor_width)
        else:
            n2 = TD_N2_INT.at[i].set(x * cfg.n2_gate_width)
        return _tidal(n2, E=TD_E_SMALL)[i] / scale

    _check(f, 0.1 if which != "gate" else 0.3)
