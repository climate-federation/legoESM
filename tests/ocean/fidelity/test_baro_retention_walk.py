"""Direct test for #1455's barotropic retention walk.

The module's job is to keep two things apart that look identical in a single
arm: a LINEAR RETENTION of an injected barotropic increment, and CHAOTIC
DIVERGENCE of two trajectories.  Only the first may be multiplied by a
per-step injection rate to predict an accumulation.  Every assertion below is
written so it FAILS if that discrimination is removed or inverted.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pytest

_MOD_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))),
    "scripts", "validate", "ocean_fidelity", "dino_1226",
    "baro_retention_walk.py")


def _load():
    spec = importlib.util.spec_from_file_location("_baro_ret", _MOD_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_baro_ret"] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load()


def _arm(acc, scale, key="dU_avg"):
    return {"path": "<synthetic>", "acc": np.asarray(acc, dtype=np.float64),
            "gate": np.asarray(acc, dtype=np.float64),
            "scale": float(scale), "key": key}


# --------------------------------------------------------------- retention ---
def test_retention_recovers_a_planted_linear_response():
    """A response that is exactly f(t) * injection must give R(t) = f(t)."""
    inj = 5.0e-3
    f = np.array([0.5, 0.25, 0.1, 0.0])
    ctrl = _arm(np.full(4, 60.0), 0.0)
    pert = _arm(60.0 + f * inj, 1.0)
    r = M.retention(pert, ctrl, inj)
    assert r["R"] == pytest.approx(f, rel=1e-12)
    # and it must scale out: the SAME f at 100x injection gives the SAME R
    pert100 = _arm(60.0 + f * inj * 100.0, 100.0)
    r100 = M.retention(pert100, ctrl, inj)
    assert r100["R"] == pytest.approx(f, rel=1e-12)


def test_retention_refuses_a_zero_injection_arm():
    ctrl = _arm(np.full(4, 60.0), 0.0)
    with pytest.raises(SystemExit):
        M.retention(_arm(np.full(4, 60.0), 0.0), ctrl, 5.0e-3)


# --------------------------------------------------------------- linearity ---
def test_linearity_passes_a_planted_linear_pair_and_fails_a_chaotic_one():
    """The load-bearing discrimination, both arms of it.

    LINEAR: response proportional to injection => R identical => ratio 1.
    CHAOTIC: the large-amplitude arm saturates instead of scaling, so its R
    is far smaller and the ratio leaves the tolerance band.
    """
    inj = 5.0e-3
    big = 40.0 * M.FLOOR_SV        # comfortably readable
    f = np.array([1.0, 0.8, 0.6, 0.5]) * big / inj
    ctrl = _arm(np.full(4, 60.0), 0.0)

    a1 = M.retention(_arm(60.0 + f * inj, 1.0), ctrl, inj)
    a100_lin = M.retention(_arm(60.0 + f * inj * 100.0, 100.0), ctrl, inj)
    L = M.linearity(a100_lin, a1)
    assert L["n"] == 4, "all four days must be readable in this fixture"
    assert L["median_ratio"] == pytest.approx(1.0, rel=1e-9)
    assert L["pass"] is True

    # chaotic: the 100x arm saturates at 3x the 1x response, not 100x.
    a100_sat = M.retention(_arm(60.0 + f * inj * 3.0, 100.0), ctrl, inj)
    Lc = M.linearity(a100_sat, a1)
    assert Lc["median_ratio"] == pytest.approx(0.03, rel=1e-9)
    assert Lc["pass"] is False, (
        "a saturating (chaotic) response must NOT be certified linear -- "
        "multiplying it by a per-step rate reports a Lyapunov exponent as a "
        "retention factor")


def test_linearity_reports_no_readable_day_rather_than_inventing_one():
    """Below the floor there is no information, and the code must say so."""
    inj = 5.0e-3
    tiny = 0.1 * M.FLOOR_SV
    ctrl = _arm(np.full(3, 60.0), 0.0)
    a = M.retention(_arm(60.0 + np.full(3, tiny), 1.0), ctrl, inj)
    b = M.retention(_arm(60.0 + np.full(3, tiny * 100), 100.0), ctrl, inj)
    L = M.linearity(a, b)
    assert L["n"] == 0 and L["pass"] is False
    assert "readable" in L["reason"]


# -------------------------------------------------------------- accounting ---
def test_accumulate_matches_the_hand_integral():
    """S = d * steps_per_day * sum(R), and a decayed R cannot pay a gap."""
    R = np.array([0.1, 0.01, 0.0, 0.0])
    d = 4.55e-3
    assert M.accumulate(R, d) == pytest.approx(d * M.STEPS_PER_DAY * 0.11)
    # the physical statement the campaign needs: a retention that decays
    # inside two days cannot reach a tenth of a Sverdrup from this deposit.
    assert abs(M.accumulate(R, d)) < 0.1 * abs(M.GAP_FULL_SECTION)


def test_accumulate_is_sign_preserving():
    """A POSITIVE deposit under a POSITIVE retention cannot pay a NEGATIVE gap.

    This is the tension the whole measurement exists to resolve, so it gets an
    assertion rather than a comment.
    """
    S = M.accumulate(np.full(90, 0.02), 4.55e-3)
    assert S > 0 and M.GAP_FULL_SECTION < 0
    assert S * M.GAP_FULL_SECTION < 0, "sign contradiction must be detectable"


# ------------------------------------------------------------------- loader ---
def test_load_arm_rejects_an_unstable_arm_and_a_nan_arm(tmp_path):
    good = dict(acc_dep_daily=np.zeros(3), acc_gate_daily=np.zeros(3),
                perturb_baro_scale=np.float64(1.0),
                perturb_baro_key=np.str_("dU_avg"),
                stable=np.array(True), blew_up_at_step=-1)
    p = tmp_path / "ok.npz"
    np.savez(p, **good)
    assert M.load_arm(str(p))["scale"] == 1.0

    bad = dict(good); bad["stable"] = np.array(False)
    p2 = tmp_path / "unstable.npz"; np.savez(p2, **bad)
    with pytest.raises(SystemExit):
        M.load_arm(str(p2))

    nan = dict(good); nan["acc_dep_daily"] = np.array([0.0, np.nan, 0.0])
    p3 = tmp_path / "nan.npz"; np.savez(p3, **nan)
    with pytest.raises(SystemExit):
        M.load_arm(str(p3))

    missing = {k: v for k, v in good.items() if k != "acc_dep_daily"}
    p4 = tmp_path / "missing.npz"; np.savez(p4, **missing)
    with pytest.raises(SystemExit):
        M.load_arm(str(p4))


def test_module_actually_uses_its_own_linearity_function():
    """A source assertion: main() must call the tested function.

    Without this the tests above could pass while main() re-implemented the
    arithmetic inline -- the exact defect this branch blocked on twice.
    """
    import inspect
    src = inspect.getsource(M.main)
    assert "linearity(" in src, "main() does not call linearity()"
    assert "accumulate(" in src, "main() does not call accumulate()"
    assert "retention(" in src, "main() does not call retention()"
