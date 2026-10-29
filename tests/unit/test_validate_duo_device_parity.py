import numpy as np
import pytest

from scripts.validate import validate_duo_device_parity as v

N = v.N_COLUMNS


def _pair():
    """Synthetic 1-device / 2-device dumps holding every grouped key, an
    atmosphere field, and land fields with the dry-soil columns."""
    rng = np.random.default_rng(0)
    ref = {f: {} for f in ("model_step.npz", "driver_checkpoint.npz")}
    for f in ref:
        for k in v.RADIATION_KEYS[f] | v.CLOUD_VARIANCE_KEYS[f]:
            ref[f][k] = rng.uniform(1.0, 2.0, 8)
        ref[f]["s0_dyn_state_u" if f == "model_step.npz" else "u"] = \
            rng.uniform(1.0, 2.0, (8, 3))
    ck = ref["driver_checkpoint.npz"]
    ck["land_soil_dz"] = np.full(3, 0.5)
    for name in ("land_ml_theta_soil", "land_ml_psi_soil", "land_ml_T_soil"):
        ck[name] = rng.uniform(1.0, 2.0, (N, 3))
    ck["land_held_albedo"] = rng.uniform(1.0, 2.0, N)
    ck["land_held_lhflx"] = rng.uniform(1.0, 2.0, N)   # not in the defect record
    ck["land_taumag_valid"] = np.ones(N, bool)
    canopy = rng.uniform(1.0, 2.0, (N, 2))
    canopy[:5] = np.nan                       # never-converged columns
    ck[v.CANOPY_NAN_KEY] = canopy
    got = {f: {k: x.copy() for k, x in d.items()} for f, d in ref.items()}
    # a gap inside every bound, and the dry-soil jump at half its measured size
    for f in got:
        for k, x in got[f].items():
            if x.dtype.kind == "f" and not k.startswith("land_"):
                tol = v.group_of(f, k)[1]
                x.flat[0] += 0.5 * tol * np.max(np.abs(ref[f][k]))
    for (k, c), size in v.DRY_SOIL_DEFECT.items():
        peak = np.max(np.abs(ck[k]))
        x = got["driver_checkpoint.npz"][k]
        if x.ndim == 1:
            x[c] += 0.5 * size * peak
        else:
            x[c, 0] += 0.5 * size * peak
    return ref, got


@pytest.fixture(autouse=True)
def _recorded(monkeypatch):
    """The synthetic dumps' own key lists and canopy NaN count stand in for
    the measured C24 records."""
    ref, _ = _pair()
    monkeypatch.setattr(v, "MODEL_STEP_KEYS", frozenset(ref["model_step.npz"]))
    monkeypatch.setattr(v, "CHECKPOINT_KEYS", frozenset(ref["driver_checkpoint.npz"]))
    monkeypatch.setattr(v, "CANOPY_NAN_COUNT", 10)


def _run(tmp_path, ref, got):
    for side, data in (("a", ref), ("b", got)):
        (tmp_path / side).mkdir(exist_ok=True)
        for f, d in data.items():
            np.savez(tmp_path / side / f, **d)
    return v.main(["compare", str(tmp_path / "a"), str(tmp_path / "b")])


def test_gaps_inside_their_bounds_pass(tmp_path, capsys):
    ref, got = _pair()
    assert _run(tmp_path, ref, got) == 0
    out = capsys.readouterr().out
    for (k, c) in v.DRY_SOIL_DEFECT:                # named on every run
        assert f"KNOWN DEFECT {k} column {c}" in out
    assert "column 470 soil water" in out and "column 495 soil water" in out


@pytest.mark.parametrize("violation", [
    "atmosphere", "radiation", "cloud_variance", "land_other_column",
    "dry_column_above_measured", "dry_column_new_field", "missing_key",
    "grouped_key_absent", "both_missing", "extra_key", "shape", "nan",
    "nan_both_arms", "inf", "inf_both_arms", "zero_peak", "bool", "canopy_nan_moved",
    "canopy_nan_count", "canopy_inf", "canopy_finite_diverges", "dtype"])
def test_each_violation_fails(tmp_path, violation):
    ref, got = _pair()
    m, c = got["model_step.npz"], got["driver_checkpoint.npz"]
    rm, rc = ref["model_step.npz"], ref["driver_checkpoint.npz"]
    if violation == "atmosphere":
        m["s0_dyn_state_u"][0, 0] += 2e-10 * np.max(np.abs(rm["s0_dyn_state_u"]))
    elif violation == "radiation":
        c["T"][0] += 2e-4 * np.max(np.abs(rc["T"]))
    elif violation == "cloud_variance":
        m["s0_physout_relvar"][0] += 1e-8 * np.max(np.abs(rm["s0_physout_relvar"]))
    elif violation == "land_other_column":
        c["land_ml_theta_soil"][7, 1] += 2e-6 * np.max(rc["land_ml_theta_soil"])
    elif violation == "dry_column_above_measured":
        k = "land_ml_T_soil"
        c[k][495, 2] += 2 * v.DRY_SOIL_DEFECT[(k, 495)] * np.max(rc[k])
    elif violation == "dry_column_new_field":
        c["land_held_lhflx"][470] += 1e-3
    elif violation == "missing_key":
        del c["u"]
    elif violation == "grouped_key_absent":
        del rm["s0_physout_rad_heating"], m["s0_physout_rad_heating"]
    elif violation == "shape":
        m["s0_dyn_state_u"] = m["s0_dyn_state_u"][:, :2]
    elif violation == "nan":
        c["u"][1, 1] = np.nan
    elif violation == "both_missing":
        del rc["u"], c["u"]
    elif violation == "extra_key":
        rm["s0_new_extra"] = np.ones(2)
        m["s0_new_extra"] = np.ones(2)
    elif violation == "nan_both_arms":
        rc["u"][:] = np.nan
        c["u"][:] = np.nan
    elif violation == "canopy_nan_moved":
        c[v.CANOPY_NAN_KEY][4, 0] = 1.5
        c[v.CANOPY_NAN_KEY][9, 0] = np.nan
    elif violation == "canopy_nan_count":
        rc[v.CANOPY_NAN_KEY][9, 0] = np.nan
        c[v.CANOPY_NAN_KEY][9, 0] = np.nan
    elif violation == "canopy_finite_diverges":     # NaN entries still match
        c[v.CANOPY_NAN_KEY][100:] *= 1.01
    elif violation == "dtype":
        m["s0_dyn_state_u"] = m["s0_dyn_state_u"].astype(np.float32)
    elif violation == "canopy_inf":
        c[v.CANOPY_NAN_KEY][7, 1] = np.inf
    elif violation == "inf":
        c["u"][1, 1] = np.inf
    elif violation == "inf_both_arms":
        rc["u"][1, 1] = np.inf
        c["u"][1, 1] = np.inf
    elif violation == "zero_peak":
        rm["s0_dyn_state_u"][:] = 0.0
        m["s0_dyn_state_u"][:] = 0.0
        m["s0_dyn_state_u"][2, 1] = 1e-300
    elif violation == "bool":
        c["land_taumag_valid"][3] = False
    assert _run(tmp_path, ref, got) == 1


def test_non_c24_land_is_refused(tmp_path):
    ref, got = _pair()
    for d in (ref, got):
        d["driver_checkpoint.npz"]["land_ml_theta_soil"] = np.ones((10, 3))
    assert _run(tmp_path, ref, got) == 1


def test_soil_water_uses_layer_depth_and_water_density():
    from legoesm import constants
    ck = {"land_ml_theta_soil": np.array([[0.1, 0.2]]), "land_soil_dz": np.array([1.0, 2.0])}
    assert v.soil_water_kg_m2(ck, 0) == pytest.approx(0.5 * constants.rho_water)


def test_flatten_names_leaves_and_refuses_text():
    rec = {}
    v._flatten("s0_pre", {"state": {"pt": np.ones(2)}, "q": [np.zeros(1)]}, rec)
    assert set(rec) == {"s0_pre_state_pt", "s0_pre_q_0"}
    with pytest.raises(TypeError):
        v._flatten("x", {"name": np.array(["a"])}, {})


def test_dump_needs_run_amip_arguments():
    with pytest.raises(SystemExit):
        v.main(["dump", "--out-dir", "/nonexistent/never-created"])
