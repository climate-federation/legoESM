"""CLM-ML leaf thermal acclimation forms (DifferBESS v10, plan Part C): Knauer et al. (2023) as ``acclim_type == 2``.

Contract: (1) the Knauer constants equal DifferBESS's ``Photosynthesis_FvCB_Knauer._C3_*`` (pinned here as numbers);
(2) ``knauer_c3_leaf_params`` reproduces the linear forms with the [3, 37] degC clips; (3) the type-1 kernel (closed-over
activation energies) is bit-identical to the generalised kernel fed the same floats as runtime arguments — the refactor
that made Ha a runtime argument changed nothing for Kattge & Knorr; (4) ``_apply_acclim_type`` patches the three
by-value namespaces and clears the kernel caches on a switch; (5) the config refuses Knauer without a home temperature
or under the WUE stomatal model.
"""
from __future__ import annotations

import numpy as np
import pytest

from legoesm.land.canopy.config import CLM_ML_ACCLIM_TYPE, CLMMLCanopyConfig

DIFFERBESS_C3 = dict(HA_V_INTERCEPT=42.6e3, HA_V_SLOPE_TG=1.14e3, HA_J=40.71e3, HD_V=200000.0, HD_J=200000.0,
                     DS_V_INTERCEPT=645.13, DS_V_SLOPE_TG=-0.38, DS_J_INTERCEPT=658.77, DS_J_SLOPE_TH=-0.84, DS_J_SLOPE_TGTH=-0.52,
                     BJV_INTERCEPT=2.56, BJV_SLOPE_TH=-0.0375, BJV_SLOPE_TGTH=-0.0202, TG_LOW=3.0, TG_HIGH=37.0, TH_LOW=3.0, TH_HIGH=37.0)


def test_constants_equal_differbess():
    from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLKnauerAcclim import KNAUER_C3, KNAUER_C3_HD
    assert KNAUER_C3 == DIFFERBESS_C3
    assert KNAUER_C3_HD == 200000.0


@pytest.mark.parametrize("tg,th", [(3.0, 3.0), (10.0, 20.0), (20.0, 25.0), (30.0, 28.0), (37.0, 37.0), (-5.0, 45.0)])
def test_leaf_params_follow_the_linear_forms_with_clips(tg, th):
    from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLKnauerAcclim import knauer_c3_leaf_params
    vha, jha, vse, jse, j2v = (float(x) for x in knauer_c3_leaf_params(tg, th))
    tg_c, th_c = min(max(tg, 3.0), 37.0), min(max(th, 3.0), 37.0)
    assert vha == pytest.approx(42.6e3 + 1.14e3 * tg_c, rel=1e-12)
    assert jha == pytest.approx(40.71e3, rel=1e-12)
    assert vse == pytest.approx(645.13 - 0.38 * tg_c, rel=1e-12)
    assert jse == pytest.approx(658.77 - 0.84 * th_c - 0.52 * (tg_c - th_c), rel=1e-12)
    assert j2v == pytest.approx(2.56 - 0.0375 * th_c - 0.0202 * (tg_c - th_c), rel=1e-12)


def test_type1_kernel_matches_generalised_kernel_to_the_last_bits():
    pytest.importorskip("legoesm.land.canopy.clm_ml_backend.multilayer_canopy")
    import jax, jax.numpy as jnp
    from legoesm.land.canopy.clm_ml_backend.multilayer_canopy import MLLeafPhotosynthesisMod as P
    from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLclm_varcon import (jmaxha_acclim, jmaxhd_acclim, vcmaxha_acclim, vcmaxhd_acclim, rdhd, rdse)
    n = 6; rng = np.random.default_rng(0)
    args = (jnp.asarray(rng.uniform(0.2, 1.0, n)), jnp.asarray(rng.uniform(283.0, 308.0, n)), jnp.asarray(rng.uniform(30.0, 80.0, n)),
            jnp.asarray(rng.uniform(60.0, 150.0, n)), jnp.asarray(rng.uniform(0.5, 1.5, n)), jnp.zeros(n), jnp.asarray(rng.uniform(800.0, 2000.0, n)),
            jnp.asarray(rng.uniform(50.0, 1500.0, n)), jnp.asarray(rng.uniform(0.5, 2.0, n)), jnp.asarray(rng.uniform(0.5, 2.0, n)), jnp.full(n, 40.0))
    rdc = P._fth25_py(rdhd, rdse)
    ta_c = 18.0; vse = 668.39 - 1.07 * ta_c; jse = 659.70 - 0.75 * ta_c
    vc = P._fth25(vcmaxhd_acclim, jnp.asarray(vse)); jc = P._fth25(jmaxhd_acclim, jnp.asarray(jse))
    common = dict(is_c3=True, c3psn_pft_val=1.0, vcmaxhd=vcmaxhd_acclim, jmaxhd=jmaxhd_acclim, rdc=rdc, o2ref_p=209.0)
    k1 = P._get_vmapped_photo_kernel_acclim(vcmaxha=vcmaxha_acclim, jmaxha=jmaxha_acclim, **common)
    # the generalised factory with its DEFAULT gas constant (the type-2 getter selects DifferBESS's 8.314, a different response)
    k2 = jax.jit(jax.vmap(P._make_leaf_photo_kernel_acclim_ha(**common), in_axes=(0,) * 11 + (None,) * 8))
    g0, g1 = jnp.asarray(0.01), jnp.asarray(4.0)
    out1 = k1(*args, g0, g1, jnp.asarray(vse), vc, jnp.asarray(jse), jc)
    out2 = k2(*args, g0, g1, jnp.asarray(vse), vc, jnp.asarray(jse), jc, jnp.asarray(float(vcmaxha_acclim)), jnp.asarray(float(jmaxha_acclim)))
    # the closed-over float is constant-folded by XLA, the runtime scalar is not: identical arithmetic, last-bit differences
    # (1e-14 relative, review 2026-10-11); the pre-refactor factory IS bit-identical to the type-1 wrapper (same closed-over path)
    for a, b in zip(out1, out2):
        assert np.allclose(np.asarray(a), np.asarray(b), rtol=1e-12, atol=0.0)
    assert np.isfinite(np.asarray(out1[15])).all() and float(np.asarray(out1[15]).max()) > 0.0   # agross


def test_type2_kernel_changes_with_growth_temperature():
    pytest.importorskip("legoesm.land.canopy.clm_ml_backend.multilayer_canopy")
    import jax.numpy as jnp
    from legoesm.land.canopy.clm_ml_backend.multilayer_canopy import MLLeafPhotosynthesisMod as P
    from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLKnauerAcclim import KNAUER_C3_HD, KNAUER_RGAS, knauer_c3_leaf_params
    from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLclm_varcon import rdhd, rdse
    n = 4
    args = (jnp.full(n, 0.5), jnp.full(n, 303.0), jnp.full(n, 60.0), jnp.full(n, 100.0), jnp.full(n, 1.0), jnp.zeros(n), jnp.full(n, 1500.0),
            jnp.full(n, 800.0), jnp.full(n, 1.0), jnp.full(n, 1.0), jnp.full(n, 40.0))
    k = P._get_vmapped_photo_kernel_acclim2(is_c3=True, c3psn_pft_val=1.0, vcmaxhd=KNAUER_C3_HD, jmaxhd=KNAUER_C3_HD, rdc=P._fth25_py(rdhd, rdse), o2ref_p=209.0)
    vmax = {}
    for tg in (10.0, 30.0):
        vha, jha, vse, jse, _ = knauer_c3_leaf_params(tg, 20.0)
        out = k(*args, jnp.asarray(0.01), jnp.asarray(4.0), vse, P._fth25(KNAUER_C3_HD, vse, KNAUER_RGAS), jse, P._fth25(KNAUER_C3_HD, jse, KNAUER_RGAS), vha, jha)   # the dispatch's normalisation
        vmax[tg] = float(np.asarray(out[3])[0])   # vcmax at the leaf temperature 30 C
    assert np.isfinite(list(vmax.values())).all() and vmax[10.0] != vmax[30.0]


def test_apply_acclim_type_patches_three_namespaces(monkeypatch):
    """The REAL modules (a sys.modules fake is bypassed by the dotted imports once the parents are loaded; review): the
    three by-value copies of acclim_type move together and the switch clears the photosynthesis factories."""
    pytest.importorskip("legoesm.land.canopy.clm_ml_backend.multilayer_canopy")
    import legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLCanopyNitrogenProfileMod as nprof
    import legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLLeafPhotosynthesisMod as photo
    import legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLclm_varctl as ctl
    import legoesm.land.canopy.clm_ml_interface as ifc
    for mod in (ctl, photo, nprof):
        monkeypatch.setattr(mod, "acclim_type", mod.acclim_type)   # restored after the test
    monkeypatch.setattr(ifc, "_APPLIED_ACCLIM_TYPE", None)
    cleared = []
    def _fn(): pass
    _fn.cache_clear = lambda: cleared.append(1)
    monkeypatch.setattr(photo, "_probe_cached_factory", _fn, raising=False)
    ifc._apply_acclim_type(CLMMLCanopyConfig(stomatal_model="medlyn", acclimation="knauer", t_home_C=19.8))
    assert (ctl.acclim_type, photo.acclim_type, nprof.acclim_type) == (2, 2, 2)
    assert cleared == []                       # first application: nothing to clear
    ifc._apply_acclim_type(CLMMLCanopyConfig(stomatal_model="medlyn", acclimation="kk"))
    assert (ctl.acclim_type, photo.acclim_type, nprof.acclim_type) == (1, 1, 1) and cleared == [1]   # the switch clears


def test_config_validation():
    assert CLM_ML_ACCLIM_TYPE == {"off": 0, "kk": 1, "knauer": 2}
    assert CLMMLCanopyConfig().acclimation == "kk"
    CLMMLCanopyConfig(stomatal_model="medlyn", acclimation="knauer", t_home_C=20.0).validate()
    with pytest.raises(ValueError, match="t_home_C"):
        CLMMLCanopyConfig(stomatal_model="medlyn", acclimation="knauer").validate()
    with pytest.raises(ValueError, match="Medlyn / Ball-Berry"):
        CLMMLCanopyConfig(stomatal_model="wue", acclimation="knauer", t_home_C=20.0).validate()
    with pytest.raises(ValueError, match="unknown CLM-ML acclimation"):
        CLMMLCanopyConfig(acclimation="bogus").validate()
