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
    """A synthetic CHATS-like captured-forcing dict (midday if sw high)."""
    return dict(Ta=t_air, q=0.010, u=3.0, P=99000.0, co2=400.0, lw=350.0,
                sw=sw, coszen=0.9 if sw > 0 else 0.0, lai=lai, Tg=t_air - 1.0,
                gpp_ml=None, sh_ml=None, lh_ml=None, sif_ml=None)


def test_build_bigleaf_forcing_density_and_fields():
    c = _capture(800.0)
    f = cmp_mod.build_bigleaf_forcing(c)
    # rho = P / (R_d T (1 + 0.61 q)) — moist air ~1.1-1.2 kg/m3 here.
    assert 1.05 < float(f.rho_lowest[0]) < 1.25
    assert float(f.sw_down[0]) == 800.0 and float(f.co2_ppmv[0]) == 400.0


def test_run_bigleaf_daytime_plausible():
    out = cmp_mod.run_bigleaf(_capture(850.0))
    # Daytime, well-lit, LAI=2: positive productive fluxes, all finite.
    assert out["gpp_bl"] is not None and out["gpp_bl"] > 0.0          # gC/m2/s
    assert 0.0 < out["gpp_bl"] / cmp_mod._UMOL_CO2_TO_GC < 60.0       # plausible umol range
    assert out["lh_bl"] > 0.0                                         # transpiring
    assert out["sif_bl"] is not None and out["sif_bl"] > 0.0
    assert out["sif_bl"] < 40.0                                       # SIF << APAR (a few %)
    for v in out.values():
        assert v is None or jnp.isfinite(jnp.asarray(v))


def test_run_bigleaf_night_zero_sif_and_gpp():
    out = cmp_mod.run_bigleaf(_capture(0.0, t_air=288.0))
    # No light -> no fluorescence, no gross assimilation.
    assert out["sif_bl"] == pytest.approx(0.0, abs=1e-9)
    assert out["gpp_bl"] is not None and out["gpp_bl"] == pytest.approx(0.0, abs=1e-9)


def test_umol_to_gc_uses_carbon_molar_mass():
    from legoesm import constants
    assert cmp_mod._UMOL_CO2_TO_GC == pytest.approx(constants.M_C * 1e-6, rel=1e-12)


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
        dpai_profile=ml.dpai_profile.at[1, 1].set(2.0),
        tg_soil=ml.tg_soil.at[1].set(297.0),
        gppveg_canopy=ml.gppveg_canopy.at[1].set(15.0),
        shflx_canopy=ml.shflx_canopy.at[1].set(60.0),
        lhflx_canopy=ml.lhflx_canopy.at[1].set(200.0),
    )
    c = cmp_mod.capture_clm_ml(ml, 0)
    assert c["Ta"] == pytest.approx(298.0) and c["co2"] == pytest.approx(400.0)
    assert c["sw"] == pytest.approx(750.0)             # 400+200+100+50
    assert c["lai"] == pytest.approx(2.0)
    assert c["coszen"] == pytest.approx(jnp.cos(jnp.asarray(0.5)), rel=1e-6)
    assert c["gpp_ml"] == pytest.approx(15.0 * cmp_mod._UMOL_CO2_TO_GC)
    # Now the big-leaf runs off the captured forcing without error.
    b = cmp_mod.run_bigleaf(c)
    assert b["sif_bl"] is not None and b["lh_bl"] > 0.0
