"""Self-tests for the W2 duo-target gate (codex w2gate P0/P1).

Synthetic pass, synthetic fail, and every input-contract rejection — the
tripwire-with-self-test doctrine.  The pure ``score`` takes the reference
stats explicitly, so nothing here touches the Zenodo file.
"""

import importlib.util
import os

import numpy as np
import pytest

REPO = os.path.join(os.path.dirname(__file__), "..", "..")
REF = (0.0236, 0.0096)          # published duo C48 hord6 envelope


def _load_gate():
    path = os.path.join(REPO, "scripts", "validate", "fv3_native",
                        "w2_duo_oracle_gate.py")
    spec = importlib.util.spec_from_file_location("w2_duo_oracle_gate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_npz(tmp_path, *, v=None, t=None, lat=None, lon=None, drop=()):
    nt = 11
    if t is None:
        t = np.linspace(0.0, 5.0, nt)
    nt = len(t)
    if v is None:
        v = np.zeros((nt, 181, 360))
    if lat is None:
        lat = np.linspace(-90, 90, 181)
    if lon is None:
        lon = np.linspace(-180, 179, 360)
    data = dict(v=v, times_days=t, lat=lat, lon=lon)
    for k in drop:
        data.pop(k)
    p = tmp_path / "snap.npz"
    np.savez(p, **data)
    return str(p)


def test_synthetic_pass_and_fail(tmp_path):
    g = _load_gate()
    # PASS: tiny error field well inside the C36 envelope
    v = np.full((11, 181, 360), 1e-4)
    v5 = g.load_run_day5_v(_write_npz(tmp_path, v=v))
    assert g.score(v5, 36, 1.5, REF)["ok"]
    # FAIL: production-magnitude field (0.5 m/s) is far outside
    v_bad = np.full((11, 181, 360), 0.5)
    v5b = g.load_run_day5_v(_write_npz(tmp_path, v=v_bad))
    r = g.score(v5b, 36, 1.5, REF)
    assert not r["ok"]
    assert r["got_max"] / r["env_max"] > 5     # quantified distance


def test_contract_rejects_wrong_day(tmp_path):
    g = _load_gate()
    p = _write_npz(tmp_path, t=np.linspace(0.0, 1.0, 3))   # ends at day 1
    with pytest.raises(g.ContractError, match="times_days"):
        g.load_run_day5_v(p)


def test_contract_rejects_day0_zeros(tmp_path):
    """The codex P0 attack: a day-0-only npz of zeros must be REJECTED,
    not scored as a pass."""
    g = _load_gate()
    p = _write_npz(tmp_path, v=np.zeros((1, 181, 360)), t=np.array([0.0]))
    with pytest.raises(g.ContractError):
        g.load_run_day5_v(p)


def test_contract_rejects_wrong_shape(tmp_path):
    g = _load_gate()
    p = _write_npz(tmp_path, v=np.zeros((11, 91, 180)))
    with pytest.raises(g.ContractError, match="181, 360"):
        g.load_run_day5_v(p)


def test_contract_rejects_nonfinite(tmp_path):
    g = _load_gate()
    v = np.zeros((11, 181, 360))
    v[-1, 5, 5] = np.nan
    with pytest.raises(g.ContractError, match="non-finite"):
        g.load_run_day5_v(_write_npz(tmp_path, v=v))


def test_contract_rejects_bad_coords_and_missing_keys(tmp_path):
    g = _load_gate()
    p = _write_npz(tmp_path, lat=np.linspace(0, 90, 181))
    with pytest.raises(g.ContractError, match="lat"):
        g.load_run_day5_v(p)
    p2 = _write_npz(tmp_path, drop=("times_days",))
    with pytest.raises(g.ContractError, match="times_days"):
        g.load_run_day5_v(p2)


def test_contract_rejects_malformed_times(tmp_path):
    """codex w2gate-r2: NaN / non-monotone / length-mismatched times."""
    g = _load_gate()
    t_nan = np.linspace(0.0, 5.0, 11)
    t_nan[3] = np.nan
    with pytest.raises(g.ContractError, match="finite"):
        g.load_run_day5_v(_write_npz(tmp_path, t=t_nan))
    t_dec = np.array([0.0, 2.0, 1.0, 5.0])
    with pytest.raises(g.ContractError, match="increasing"):
        g.load_run_day5_v(_write_npz(tmp_path,
                                     v=np.zeros((4, 181, 360)), t=t_dec))
    # nt mismatch between v and times_days
    with pytest.raises(g.ContractError, match="aligned"):
        g.load_run_day5_v(_write_npz(tmp_path, v=np.zeros((7, 181, 360))))


def test_contract_rejects_bad_lon_values(tmp_path):
    """codex w2gate-r2: 360 points with wrong VALUES must be rejected."""
    g = _load_gate()
    lon_bad = np.linspace(0.0, 35.9, 360)     # wrong span
    with pytest.raises(g.ContractError, match="lon"):
        g.load_run_day5_v(_write_npz(tmp_path, lon=lon_bad))
    lon_dec = np.linspace(179.0, -180.0, 360)  # decreasing
    with pytest.raises(g.ContractError, match="lon"):
        g.load_run_day5_v(_write_npz(tmp_path, lon=lon_dec))


def test_contract_rejects_noncanonical_interior_lat(tmp_path):
    """codex w2gate-r2: correct endpoints but warped interior spacing."""
    g = _load_gate()
    lat = np.linspace(-90.0, 90.0, 181) ** 3 / 90.0 ** 2   # warped, same ends
    with pytest.raises(g.ContractError, match="lat"):
        g.load_run_day5_v(_write_npz(tmp_path, lat=lat))


def test_score_validates_args():
    g = _load_gate()
    v5 = np.zeros((181, 360))
    with pytest.raises(ValueError, match="--res"):
        g.score(v5, 0, 1.5, REF)
    with pytest.raises(ValueError, match="tol-factor"):
        g.score(v5, 36, -1.0, REF)
