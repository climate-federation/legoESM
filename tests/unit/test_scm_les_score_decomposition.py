"""Direct tests for ``scripts/validate/scm_les_score_decomposition.py``.

The point of the probe is that it reproduces the TUNER's arithmetic rather than
an independent re-derivation of it, so the tests check exactly that: the
per-variable score it reports must equal what
``scm_rce_metrics.normalized_profile_rmse`` returns on the same inputs, and a
structureless reference must be visible as a small ``sigma_ref`` instead of
being hidden behind the scorer's floor.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = (Path(__file__).resolve().parents[2]
           / "scripts" / "validate" / "scm_les_score_decomposition.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("_scm_les_decomp", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load_module()


def _write_case(path: Path, *, nlev=8, n_in=6, scheme="louis",
                scored=("theta", "u")):
    """A profiles_<case>.npz in the layout the tuner writes."""
    mask = np.zeros(nlev, dtype=bool)
    mask[:n_in] = True
    weights = np.where(mask, 1.0, 0.0)
    weights = weights / weights.sum()
    z = np.linspace(10.0, 800.0, nlev)
    # theta: real vertical structure. u: nearly uniform -> tiny sigma_ref.
    theta_ref = np.where(mask, 300.0 + 0.01 * z, np.nan)
    u_ref = np.where(mask, 10.0 + 1.0e-3 * z, np.nan)
    payload = {
        "z_scm": z, "mask": mask, "weights": weights,
        "window_hours": np.array([4.0, 6.0]),
        "scored": np.array(list(scored)),
        "les_scmlev_theta": theta_ref,
        "les_scmlev_u": u_ref,
        f"scm_{scheme}_theta": np.nan_to_num(theta_ref, nan=0.0) + 0.5,
        f"scm_{scheme}_u": np.nan_to_num(u_ref, nan=0.0) + 0.5,
    }
    np.savez(path, **payload)
    return mask, weights


def test_score_matches_the_tuners_own_metric(tmp_path, mod):
    """The probe's score must equal normalized_profile_rmse, not an analogue."""
    from legoesm.training.scm_rce_metrics import normalized_profile_rmse

    npz = tmp_path / "profiles_toy.npz"
    _write_case(npz)
    got = mod.decompose_case(npz, "louis")

    data = np.load(npz, allow_pickle=True)
    mask = data["mask"].astype(bool)
    weights = data["weights"]
    for var in ("theta", "u"):
        ref = np.nan_to_num(data[f"les_scmlev_{var}"], nan=0.0)
        scm = np.where(mask, data[f"scm_louis_{var}"], 0.0)
        expect = float(normalized_profile_rmse(
            ref, scm, weights, profile_floor=1.0e-8))
        assert got[var]["score"] == pytest.approx(expect, rel=1e-10)


def test_absolute_rmse_is_in_physical_units(tmp_path, mod):
    """rmse_abs is the un-normalized error: a uniform +0.5 offset gives 0.5."""
    npz = tmp_path / "profiles_toy.npz"
    _write_case(npz)
    got = mod.decompose_case(npz, "louis")
    for var in ("theta", "u"):
        assert got[var]["rmse_abs"] == pytest.approx(0.5, rel=1e-10)
    assert got["theta"]["units"] == "K"
    assert got["u"]["units"] == "m/s"


def test_low_structure_reference_shows_a_small_sigma_not_a_big_error(
        tmp_path, mod):
    """The discrimination the probe exists for.

    Both variables carry the SAME physical error (0.5). The variable whose
    reference has less vertical structure must report a proportionally larger
    normalized score while its rmse_abs stays identical -- which is how a
    normalization artifact is told apart from a genuinely worse fit.
    """
    npz = tmp_path / "profiles_toy.npz"
    _write_case(npz)
    got = mod.decompose_case(npz, "louis")
    assert got["u"]["sigma_ref"] < got["theta"]["sigma_ref"]
    assert got["u"]["score"] > got["theta"]["score"]
    assert got["u"]["rmse_abs"] == pytest.approx(got["theta"]["rmse_abs"])
    # and the score ratio is exactly the inverse sigma ratio
    assert (got["u"]["score"] / got["theta"]["score"]) == pytest.approx(
        got["theta"]["sigma_ref"] / got["u"]["sigma_ref"], rel=1e-10)


def test_missing_per_case_profiles_is_a_hard_error(tmp_path, mod, capsys):
    """A missing case must fail loudly, never be silently skipped."""
    (tmp_path / "tuned_parameters.json").write_text(json.dumps({
        "protocol": {"cases": ["bomex", "ekman"]},
        "schemes": [{"scheme": "louis",
                     "per_case_default": {"bomex": 0.1, "ekman": 1.0}}],
    }))
    _write_case(tmp_path / "profiles_bomex.npz")   # ekman deliberately absent
    with pytest.raises(SystemExit) as exc:
        mod.main([str(tmp_path)])
    assert "ekman" in str(exc.value)


def test_end_to_end_report_shares_sum_to_100(tmp_path, mod):
    (tmp_path / "tuned_parameters.json").write_text(json.dumps({
        "protocol": {"cases": ["bomex", "ekman"]},
        "schemes": [{"scheme": "louis",
                     "per_case_default": {"bomex": 0.1, "ekman": 1.0}}],
    }))
    for case in ("bomex", "ekman"):
        _write_case(tmp_path / f"profiles_{case}.npz")
    out = tmp_path / "decomp.json"
    assert mod.main([str(tmp_path), "--json-out", str(out)]) == 0
    report = json.loads(out.read_text())["report"]["louis"]
    total = sum(r["joint_share_pct"] for r in report.values())
    assert total == pytest.approx(100.0)
    # ekman carries 10x the score of bomex, so ~90.9% of the joint loss
    assert report["ekman"]["joint_share_pct"] == pytest.approx(
        100.0 * 1.0 / 1.1, rel=1e-9)
