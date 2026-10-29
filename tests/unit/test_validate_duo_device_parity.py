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
    ck["land_taumag_valid"] = np.ones(N, bool)
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
    "grouped_key_absent", "shape", "nan", "bool"])
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
        c["land_held_albedo2"] = rc["land_held_albedo"].copy()
        rc["land_held_albedo2"] = rc["land_held_albedo"].copy()
        c["land_held_albedo2"][470] += 1e-3
    elif violation == "missing_key":
        del c["u"]
    elif violation == "grouped_key_absent":
        del rm["s0_physout_rad_heating"], m["s0_physout_rad_heating"]
    elif violation == "shape":
        m["s0_dyn_state_u"] = m["s0_dyn_state_u"][:, :2]
    elif violation == "nan":
        c["u"][1, 1] = np.nan
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
