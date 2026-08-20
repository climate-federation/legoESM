"""Direct tests for compare_tendencies_nemo.prandtl_split_report.

The report exists to say whether the Stage-A K_H excess is an over-energetic
closure or a wrong tracer/momentum split, so it is run here on synthetic
fields with a KNOWN answer before it is trusted on NEMO's.

Its runtime control is also checked for NON-VACUITY.  An earlier revision
asserted the factorisation identity K_H/avt == (K_M/avm)*(Pr_n/Pr_o), which is
true for ANY four arrays and therefore could not fail; the control it was
replaced with (NEMO's own avt <= avm off the floors) is shown below to fire
when that property is broken.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts/validate/ocean_fidelity"))
_SPEC = importlib.util.spec_from_file_location(
    "compare_tendencies_nemo",
    _ROOT / "scripts/validate/ocean_fidelity/compare_tendencies_nemo.py")
ctn = importlib.util.module_from_spec(_SPEC)
sys.modules["compare_tendencies_nemo"] = ctn
_SPEC.loader.exec_module(ctn)

NCOL, NIFACE = 40, 6
PR_NEMO, PR_OURS = 3.0, 1.5
AVMB, AVTB = 1.4e-6, 1.2e-7        # the "floors" the report must detect


def _fields(avt_scale=1.0):
    """avm/avt with a known Prandtl, K_M == avm, K_H == K_M/PR_OURS.

    Interface 0 is put ON both floors so the report has something to exclude;
    the remaining five carry the signal.
    """
    avm = np.full((NCOL, NIFACE), 1.0e-2)
    avm[:, 0] = AVMB
    avt = avm / PR_NEMO * avt_scale
    avt[:, 0] = AVTB
    K_M = avm.copy()
    K_H = K_M / PR_OURS
    wet = np.ones((NCOL, NIFACE), dtype=bool)
    lat = np.zeros(NCOL)                    # equator: tropics + nino3 + eqpac
    lon = np.full(NCOL, -120.0)             # inside nino3 (-150..-90)
    return K_H, K_M, avt, avm, wet, lat, lon


def _row(rows, name):
    hit = [r for r in rows if r["region"] == name]
    assert hit, f"{name!r} missing from {[r['region'] for r in rows]}"
    return hit[0]


def test_known_answer_recovers_the_planted_prandtl():
    K_H, K_M, avt, avm, wet, lat, lon = _fields()
    rows = ctn.prandtl_split_report(K_H, K_M, avt, avm, wet, lat, lon)
    r = _row(rows, "nino3")
    # Floors excluded => 5 of 6 interfaces survive.
    assert r["n"] == NCOL * (NIFACE - 1)
    assert r["K_M_over_avm_rms"] == pytest.approx(1.0, rel=1e-12)
    assert r["K_H_over_avt_rms"] == pytest.approx(PR_NEMO / PR_OURS, rel=1e-12)
    assert r["Pr_nemo_median"] == pytest.approx(PR_NEMO, rel=1e-12)
    assert r["Pr_ours_median"] == pytest.approx(PR_OURS, rel=1e-12)
    # Nothing is on a clamp here, so the medians mean what they say.
    assert r["Pr_ours_frac_at_ceiling"] == 0.0
    assert r["Pr_ours_frac_at_floor"] == 0.0


def test_saturated_prandtl_is_flagged_not_hidden():
    """A Pr median sitting on the 10.0 clamp must report at_ceil == 1."""
    K_H, K_M, avt, avm, wet, lat, lon = _fields()
    K_H = K_M / 10.0
    rows = ctn.prandtl_split_report(K_H, K_M, avt, avm, wet, lat, lon)
    r = _row(rows, "nino3")
    assert r["Pr_ours_median"] == pytest.approx(10.0, rel=1e-12)
    assert r["Pr_ours_frac_at_ceiling"] == 1.0


def test_control_fires_when_nemo_prandtl_sign_is_broken():
    """NON-VACUITY: avt > avm off the floors is impossible under nn_pdl=1."""
    K_H, K_M, avt, avm, wet, lat, lon = _fields()
    avt[:, 1:] = avm[:, 1:] * 2.0            # Pr_nemo = 0.5: cannot happen
    with pytest.raises(SystemExit, match="avt exceeds its avm"):
        ctn.prandtl_split_report(K_H, K_M, avt, avm, wet, lat, lon)


def test_control_tolerates_the_inversion_in_evd_columns():
    """EVD legitimately inverts it (rn_evd on the tracer, nn_evdm=0), so the
    abort must be scoped to the calm columns and not fire on convection."""
    K_H, K_M, avt, avm, wet, lat, lon = _fields()
    evd = np.zeros(NCOL, dtype=bool)
    evd[NCOL // 2:] = True
    avt[evd, 1:] = 100.0                     # rn_evd
    rows = ctn.prandtl_split_report(K_H, K_M, avt, avm, wet, lat, lon,
                                    evd_cols=evd)
    assert _row(rows, "nino3/calm")["Pr_nemo_median"] == pytest.approx(PR_NEMO)


def test_depth_split_separates_the_thermocline_from_the_abyss():
    """The abyss is where Pr saturates for real; the cut must isolate it."""
    K_H, K_M, avt, avm, wet, lat, lon = _fields()
    z = np.tile(np.array([10.0, 50.0, 150.0, 1000.0, 2000.0, 3000.0]),
                (NCOL, 1))
    K_H = K_M / np.where(z <= 300.0, PR_OURS, 10.0)   # saturated only deep
    rows = ctn.prandtl_split_report(K_H, K_M, avt, avm, wet, lat, lon,
                                    z_iface=z, z_cuts_m=(300.0,))
    shallow = _row(rows, "nino3<300m")
    assert shallow["Pr_ours_median"] == pytest.approx(PR_OURS, rel=1e-12)
    assert shallow["Pr_ours_frac_at_ceiling"] == 0.0
    # Full column: interface 0 is dropped as floored, leaving 2 shallow and 3
    # deep, so the median lands ON the clamp -- the exact statistic that made
    # the first NEMO run unreadable.
    full = _row(rows, "nino3")
    assert full["Pr_ours_frac_at_ceiling"] == pytest.approx(0.6, rel=1e-12)


def test_ri_ratio_uses_only_the_both_unclamped_subset():
    """Pr = clamp(4.5*Ri,1,10) on both sides, so the Pr ratio is the Ri ratio
    ONLY where neither is clamped. A clamped point must not enter it."""
    K_H, K_M, avt, avm, wet, lat, lon = _fields()
    # Half the interfaces get OUR Pr pinned on the ceiling; those must be
    # excluded, leaving the planted PR_OURS/PR_NEMO on the rest.
    K_H = K_M / PR_OURS
    K_H[: NCOL // 2, :] = K_M[: NCOL // 2, :] / 10.0
    rows = ctn.prandtl_split_report(K_H, K_M, avt, avm, wet, lat, lon)
    r = _row(rows, "nino3")
    assert r["Pr_ours_frac_at_ceiling"] == pytest.approx(0.5, rel=1e-12)
    assert r["n_both_unclamped"] == (NCOL // 2) * (NIFACE - 1)
    assert r["Ri_ratio_median"] == pytest.approx(PR_OURS / PR_NEMO, rel=1e-12)
