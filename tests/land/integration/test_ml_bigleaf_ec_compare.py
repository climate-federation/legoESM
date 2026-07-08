"""Tests for the multilayer-vs-big-leaf EC comparison driver
(``scripts/validate/compare_ml_bigleaf_ec.py``).

The CI-runnable half exercises the big-leaf side (``run_bigleaf`` /
``build_bigleaf_forcing``) on a synthetic CHATS-like captured-forcing dict — no
``clm-ml-jax`` needed.  A separate importorskip smoke drives ``capture_clm_ml``
against a real ``create_mlcanopy`` container when the optional model is present.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

# Load the script module by path (scripts/ is not an importable package).
_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "validate" / "compare_ml_bigleaf_ec.py"
_spec = importlib.util.spec_from_file_location("compare_ml_bigleaf_ec", _SCRIPT)
cmp_mod = importlib.util.module_from_spec(_spec)
sys.modules["compare_ml_bigleaf_ec"] = cmp_mod
_spec.loader.exec_module(cmp_mod)


def _capture(sw, t_air=295.0, *, lai=2.0):
    """A synthetic CHATS-like captured-forcing dict (midday if sw high).

    Includes the canopy STRUCTURE keys ``run_bigleaf`` matches to CLM-ML (pai,
    vcmax_top, ztop, zref) so the test exercises the matched-parameter path.
    """
    return dict(Ta=t_air, q=0.010, u=3.0, P=99000.0, co2=400.0, lw=350.0,
                sw=sw, coszen=0.9 if sw > 0 else 0.0, lai=lai, Tg=t_air - 1.0,
                pai=lai + 0.7, vcmax_top=100.0, ztop=10.0, zref=23.0,
                gpp_ml=None, sh_ml=None, lh_ml=None, sif_ml=None,
                shveg_ml=None, lhveg_ml=None, shsoi_ml=None, lhsoi_ml=None,
                rnveg_ml=None)


def test_build_bigleaf_forcing_density_and_fields():
    c = _capture(800.0)
    f = cmp_mod.build_bigleaf_forcing(c)
    # rho = P / (R_d T (1 + 0.61 q)) — moist air ~1.1-1.2 kg/m3 here.
    assert 1.05 < float(f.rho_lowest[0]) < 1.25
    assert float(f.sw_down[0]) == 800.0 and float(f.co2_ppmv[0]) == 400.0


def test_run_bigleaf_daytime_plausible():
    out = cmp_mod.run_bigleaf(_capture(850.0))
    # Daytime, well-lit, matched vcmax/PAI: positive productive fluxes, finite.
    assert out["gpp_bl"] is not None and out["gpp_bl"] > 0.0          # gC/m2/s
    assert 0.0 < out["gpp_bl"] / cmp_mod._UMOL_CO2_TO_GC < 80.0       # plausible umol range
    assert out["lh_bl"] > 0.0                                         # transpiring
    assert out["sif_bl"] is not None and out["sif_bl"] > 0.0
    assert out["sif_bl"] < 40.0                                       # SIF << APAR (a few %)
    # Vegetation/soil split is populated and sums to the totals.
    assert jnp.isfinite(jnp.asarray(out["shveg_bl"]))
    assert out["lhveg_bl"] > 0.0                                      # leaf transpiration
    assert out["sh_bl"] == pytest.approx(out["shveg_bl"] + out["shsoi_bl"], abs=1e-4)
    assert out["lh_bl"] == pytest.approx(out["lhveg_bl"] + out["lhsoi_bl"], abs=1e-4)
    for v in out.values():
        assert v is None or jnp.isfinite(jnp.asarray(v))


def test_run_bigleaf_vcmax_raises_gpp():
    # The matched top-canopy Vcmax must actually reach photosynthesis: a higher
    # vcmax_top produces more GPP (guards the canopy_params wiring from silently
    # falling back to the scalar default).
    lo = cmp_mod.run_bigleaf({**_capture(850.0), "vcmax_top": 40.0})
    hi = cmp_mod.run_bigleaf({**_capture(850.0), "vcmax_top": 120.0})
    assert hi["gpp_bl"] > lo["gpp_bl"] > 0.0


def test_run_bigleaf_vcmax_gap_falls_back():
    # A capture gap (vcmax_top nan / non-positive) must fall back to a positive
    # default, NOT silently run with zero photosynthetic capacity.
    for bad in (float("nan"), 0.0, -5.0):
        out = cmp_mod.run_bigleaf({**_capture(850.0), "vcmax_top": bad})
        assert out["gpp_bl"] is not None and out["gpp_bl"] > 0.0


def test_run_bigleaf_night_zero_sif_and_gpp():
    out = cmp_mod.run_bigleaf(_capture(0.0, t_air=288.0))
    # No light -> no fluorescence, no gross assimilation.
    assert out["sif_bl"] == pytest.approx(0.0, abs=1e-9)
    assert out["gpp_bl"] is not None and out["gpp_bl"] == pytest.approx(0.0, abs=1e-9)


def test_umol_to_gc_uses_carbon_molar_mass():
    from legoesm import constants
    assert cmp_mod._UMOL_CO2_TO_GC == pytest.approx(constants.M_C * 1e-6, rel=1e-12)


def test_local_solar_hours_centers_noon_on_coszen_max():
    import numpy as np

    # coszen peaks at step 40 -> that step is local solar noon (12.0 h); the
    # rest offset by (step - 40) * 0.5 h, wrapped into [0, 24).  Steps need not
    # be contiguous (a night gap between the two partial days).
    steps = [38, 39, 40, 41, 42, 5, 6]
    coszen = [0.88, 0.91, 0.92, 0.90, 0.86, 0.10, 0.20]
    recs = [{"step": s, "coszen": c} for s, c in zip(steps, coszen)]
    lt = cmp_mod._local_solar_hours(recs)
    assert lt[2] == pytest.approx(12.0)          # the coszen-max step
    assert lt[0] == pytest.approx(11.0)          # step 38 -> noon - 1 h
    assert lt[4] == pytest.approx(13.0)          # step 42 -> noon + 1 h
    assert lt[5] == pytest.approx((12.0 + (5 - 40) * 0.5) % 24.0)  # wrapped
    assert np.all((lt >= 0.0) & (lt < 24.0))


def test_local_solar_hours_all_nan_coszen_falls_back():
    import numpy as np

    # No finite coszen (solar geometry absent) must NOT raise on an all-NaN
    # nanargmax; fall back to elapsed half-hours from the earliest step.
    recs = [{"step": 10, "coszen": float("nan")}, {"step": 12, "coszen": float("nan")}]
    lt = cmp_mod._local_solar_hours(recs)
    assert lt[0] == pytest.approx(0.0)           # earliest step -> 0 h elapsed
    assert lt[1] == pytest.approx(1.0)           # +2 steps * 0.5 h
    assert np.all((lt >= 0.0) & (lt < 24.0))


def test_plot_writes_figure(tmp_path):
    # Smoke: the publication plotter renders daytime records to a PNG.  Night
    # (sw<=50) rows are dropped; at least one daytime row must remain.
    recs = [
        {"step": 30, "coszen": 0.4, "sw": 300.0, "gpp_ml": 2.0e-4, "gpp_bl": 2.2e-4,
         "sif_ml": 8.0, "sif_bl": 8.3, "lhveg_ml": 200.0, "lhveg_bl": 180.0,
         "shveg_ml": 40.0, "shveg_bl": 55.0},
        {"step": 40, "coszen": 0.9, "sw": 800.0, "gpp_ml": 3.2e-4, "gpp_bl": 3.7e-4,
         "sif_ml": 15.0, "sif_bl": 15.4, "lhveg_ml": 320.0, "lhveg_bl": 300.0,
         "shveg_ml": 90.0, "shveg_bl": 100.0},
        {"step": 60, "coszen": 0.01, "sw": 0.0, "gpp_ml": 0.0, "gpp_bl": 0.0},  # night, dropped
    ]
    out = tmp_path / "fig.png"
    cmp_mod._plot(recs, str(out))
    assert out.exists() and out.stat().st_size > 0


def test_capture_clm_ml_reads_real_forcing_fields():
    mlt = pytest.importorskip("multilayer_canopy.MLCanopyFluxesType")
    ml = mlt.create_mlcanopy(begp=1, endp=1)  # 1-based patch: column 0 at index 1
    # Fill the forcing + a filled sunlit leaf so capture returns finite values.
    ml = ml._replace(
        tref_forcing=ml.tref_forcing.at[1].set(298.0),
        qref_forcing=ml.qref_forcing.at[1].set(0.010),
        uref_forcing=ml.uref_forcing.at[1].set(2.5),
        pref_forcing=ml.pref_forcing.at[1].set(99000.0),
        co2ref_forcing=ml.co2ref_forcing.at[1].set(400.0),
        lwsky_forcing=ml.lwsky_forcing.at[1].set(350.0),
        solar_zen_forcing=ml.solar_zen_forcing.at[1].set(0.5),
        swskyb_forcing=ml.swskyb_forcing.at[1, 1].set(400.0).at[1, 2].set(200.0),
        swskyd_forcing=ml.swskyd_forcing.at[1, 1].set(100.0).at[1, 2].set(50.0),
        dlai_profile=ml.dlai_profile.at[1, 1].set(2.0),     # green LAI
        dpai_profile=ml.dpai_profile.at[1, 1].set(2.7),     # plant area (incl. stems)
        vcmax25_leaf=ml.vcmax25_leaf.at[1, 1, 1].set(95.0),  # sunlit top layer
        ztop_canopy=ml.ztop_canopy.at[1].set(10.0),
        zref_forcing=ml.zref_forcing.at[1].set(23.0),
        tg_soil=ml.tg_soil.at[1].set(297.0),
        gppveg_canopy=ml.gppveg_canopy.at[1].set(15.0),
        shflx_canopy=ml.shflx_canopy.at[1].set(60.0),
        shveg_canopy=ml.shveg_canopy.at[1].set(40.0),
        lhflx_canopy=ml.lhflx_canopy.at[1].set(200.0),
        lhveg_canopy=ml.lhveg_canopy.at[1].set(180.0),
    )
    c = cmp_mod.capture_clm_ml(ml, 0)
    assert c["Ta"] == pytest.approx(298.0) and c["co2"] == pytest.approx(400.0)
    assert c["sw"] == pytest.approx(750.0)             # 400+200+100+50
    assert c["lai"] == pytest.approx(2.0)   # green LAI from dlai
    assert c["pai"] == pytest.approx(2.7)   # plant area from dpai
    assert c["coszen"] == pytest.approx(jnp.cos(jnp.asarray(0.5)), rel=1e-6)
    assert c["gpp_ml"] == pytest.approx(15.0 * cmp_mod._UMOL_CO2_TO_GC)
    # Matched-structure fields captured for driving the two-leaf identically.
    assert c["vcmax_top"] == pytest.approx(95.0)
    assert c["ztop"] == pytest.approx(10.0) and c["zref"] == pytest.approx(23.0)
    assert c["shveg_ml"] == pytest.approx(40.0) and c["lhveg_ml"] == pytest.approx(180.0)
    # Now the big-leaf runs off the captured forcing without error.
    b = cmp_mod.run_bigleaf(c)
    assert b["sif_bl"] is not None and b["lh_bl"] > 0.0
