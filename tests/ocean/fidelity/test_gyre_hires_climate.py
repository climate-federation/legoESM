import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_hires_climate.py"
SPEC = importlib.util.spec_from_file_location("gyre_hires_climate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CLIMATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLIMATE)


def _write_legoesm_provider_fixture(root: Path, *, mutation: str | None = None) -> Path:
    from legoesm.ocean.fidelity.nemo_testcase_recipe import GYRE_R4_SMAG_NAMELIST

    run = root / "run"
    run.mkdir(parents=True)
    shape = CLIMATE.GLOBAL_SHAPE
    values = np.ones(shape, dtype=np.float64)
    if mutation == "nonfinite":
        values[4, 5] = np.nan
    day_value = 2 if mutation == "time-counter" else 1
    dt_s = 1800.0 if mutation == "cadence" else 3600.0
    step_value = 23 if mutation == "step-counter" else 24
    sample_count = 23 if mutation == "sample-count" else 24
    daily = run / "day0001.npz"
    np.savez_compressed(
        daily,
        day=np.int64(day_value), dt_s=np.float64(dt_s),
        step=np.int64(step_value),
        daily_mean_sample_count=np.int64(sample_count),
        ssh_m_daily_mean=values,
        surface_u_m_s_daily_mean=np.ones(shape),
        surface_v_m_s_daily_mean=np.ones(shape),
        surface_temperature_c_daily_mean=np.ones(shape),
    )
    if mutation == "file-sequence":
        np.savez_compressed(run / "day0002.npz", day=np.int64(2))
    namelist = GYRE_R4_SMAG_NAMELIST._asdict()
    if mutation == "deck":
        namelist = dict(namelist, nn_GYRE=3)
    run_namelist = dict(namelist, nn_itend=24, nn_write=24)
    if mutation == "namelist-cadence":
        run_namelist["nn_write"] = 23
    manifest = {
        "schema": "legoesm-gyre-hires-daily-v1",
        "status": "pending" if mutation == "manifest" else "complete",
        "completed_days": 1,
        "days": 1,
        "steps_per_day": 23 if mutation == "manifest-cadence" else 24,
        "sampling": ["end_of_day_snapshot", "daily_mean"],
        "namelist": run_namelist,
        "legoesm_git_sha": "a" * 40,
        "mode": "gpu-native", "backend": "gpu",
        "precision_policy": "fp32" if mutation == "precision" else "fp64",
        "transcendentals": "native", "certified_policy": False,
        "physical_gpu_index": 0,
        "daily_mean_convention": {
            "surface_temperature_c": CLIMATE.LEGOESM_SST_CONVENTION,
        },
        "daily_sha256": {
            "day0001.npz": (
                "0" * 64 if mutation == "digest" else CLIMATE.sha256(daily)
            ),
        },
    }
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    registry = {
        "schema": "gyre-hires-registered-runs-v1",
        "runs": {"test": {
            "run_directory": str(run.resolve()), "commit": "a" * 40,
            "deck": "r4_smag", "mode": "gpu-native", "backend": "gpu",
            "precision_policy": "fp64",
            "transcendentals": "native", "certified_policy": False,
            "physical_gpu_index": 0, "label": "test",
        }},
    }
    registry_path = root / "registry.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    return registry_path


def test_domain_assembly_uses_one_based_xy_positions_and_refuses_overlap():
    tiles = [
        (np.array([1, 1]), np.array([[1.0, 2.0], [3.0, 4.0]])),
        (np.array([3, 1]), np.array([[5.0], [6.0]])),
    ]
    assert np.array_equal(
        CLIMATE.assemble_domain_tiles(tiles, shape=(2, 3)),
        np.array([[1.0, 2.0, 5.0], [3.0, 4.0, 6.0]]),
    )
    with pytest.raises(Exception, match="overlapping"):
        CLIMATE.assemble_domain_tiles(tiles + [(np.array([1, 1]), np.ones((2, 1)))],
                                      shape=(2, 3))


def test_weighted_sst_is_divided_by_same_file_thickness():
    weighted = np.array([[20.0, 42.0]])
    thickness = np.array([[2.0, 3.0]])
    assert np.array_equal(
        CLIMATE.decode_weighted_surface_temperature(weighted, thickness),
        np.array([[10.0, 14.0]]),
    )
    with pytest.raises(Exception, match="thickness"):
        CLIMATE.decode_weighted_surface_temperature(weighted, np.zeros_like(thickness))


def test_alternative_nemo_manifest_is_explicit_labelled_and_noncertified(
    tmp_path, monkeypatch,
):
    root = tmp_path / "scalar"
    root.mkdir()
    payloads = {
        "nemo": b"scalar-math binary",
        "namelist_cfg": b"exact deck",
        "namelist_ref": b"exact reference deck",
    }
    digests = {
        name: hashlib.sha256(payload).hexdigest()
        for name, payload in payloads.items()
    }
    for name, payload in payloads.items():
        (root / name).write_bytes(payload)
    (root / "resolved_namelist_lines.txt").write_text("", encoding="utf-8")
    manifest = root / "manifest.sha256"
    manifest.write_text(
        "".join(f"{digests[name]}  {name}\n" for name in payloads),
        encoding="utf-8",
    )
    monkeypatch.setattr(CLIMATE, "NEMO_MANIFEST", {
        "nemo": "0" * 64,
        "namelist_cfg": digests["namelist_cfg"],
        "namelist_ref": digests["namelist_ref"],
    })

    admitted = CLIMATE._require_nemo_deck(
        root, alternative_manifest=manifest, alternative_label="scalar_math")
    assert admitted["reference"] == {
        "label": "scalar_math",
        "certified": False,
        "classification": "labelled non-certified NEMO reference",
        "manifest_path": str(manifest.resolve()),
    }
    assert admitted["resolved_namelist_sha256"] is None
    with pytest.raises(Exception, match="explicit label"):
        CLIMATE._require_nemo_deck(root, alternative_manifest=manifest)
    with pytest.raises(Exception, match="not r4_smag"):
        CLIMATE._require_nemo_deck(root)
    with pytest.raises(Exception, match="label requires"):
        CLIMATE._require_nemo_deck(root, alternative_label="scalar_math")

    monkeypatch.setattr(CLIMATE, "NEMO_MANIFEST", {
        "nemo": "0" * 64,
        "namelist_cfg": "1" * 64,
        "namelist_ref": digests["namelist_ref"],
    })
    with pytest.raises(Exception, match="deck differs at namelist_cfg"):
        CLIMATE._require_nemo_deck(
            root, alternative_manifest=manifest,
            alternative_label="scalar_math")

    manifest.write_text(
        manifest.read_text(encoding="utf-8") + f"{'2' * 64}  extra\n",
        encoding="utf-8",
    )
    with pytest.raises(Exception, match="inventory differs"):
        CLIMATE._require_nemo_deck(
            root, alternative_manifest=manifest,
            alternative_label="scalar_math")


def test_alternative_nemo_manifest_cannot_name_another_root(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    elsewhere = tmp_path / "elsewhere.sha256"
    elsewhere.write_text("", encoding="utf-8")
    with pytest.raises(Exception, match="ROOT/manifest"):
        CLIMATE._require_nemo_deck(
            root, alternative_manifest=elsewhere,
            alternative_label="scalar_math")


def test_diagnose_applies_native_velocity_highpass_and_registered_statistics(monkeypatch):
    from scipy.ndimage import uniform_filter1d

    shape = (6, 8)
    wet = np.zeros(shape, dtype=bool)
    wet[1:5, 1:7] = True
    geom = {
        "wet": wet,
        "u_mask": wet.copy(),
        "v_mask": wet.copy(),
        "area": np.ones(shape),
        "latitude": np.broadcast_to(np.arange(shape[0])[:, None], shape),
        "f": np.ones(shape),
        "spacing_m": 1.0,
    }
    frames = []
    for day in range(1, 36):
        phase = np.sin(2.0 * np.pi * day / 7.0)
        ssh = np.broadcast_to((np.arange(shape[0]) - 2.5)[:, None], shape).copy()
        frames.append(CLIMATE.DailyFrame(
            day, ssh, np.full(shape, 2.0 + phase), np.full(shape, phase),
            np.full(shape, 10.0 + day),
        ))
    provider = CLIMATE.DailyProvider("synthetic", "daily_mean", {}, lambda: iter(frames))
    monkeypatch.setattr(
        CLIMATE, "cell_velocity",
        lambda u, v, geom: (np.zeros_like(u), np.zeros_like(v)),
    )
    result = CLIMATE.diagnose(provider, target_start=16, target_days=20, geom=geom)
    raw_u = np.stack([frame.u for frame in frames])
    raw_v = np.stack([frame.v for frame in frames])
    raw_u = np.where(wet, raw_u, 0.0)
    raw_v = np.where(wet, raw_v, 0.0)
    expected_eke = np.mean(0.5 * (
        (raw_u - uniform_filter1d(raw_u, 31, axis=0, mode="nearest"))[15:35] ** 2
        + (raw_v - uniform_filter1d(raw_v, 31, axis=0, mode="nearest"))[15:35] ** 2
    ), axis=0)
    assert result["basin_mean_ke_series_m2_s2"].shape == (20,)
    assert result["highpass_eke_map_m2_s2"].shape == shape
    assert np.allclose(result["highpass_eke_map_m2_s2"][wet], expected_eke[wet])
    assert np.all(np.isnan(result["highpass_eke_map_m2_s2"][~wet]))
    assert np.all(np.isnan(result["time_mean_sst_c"][~wet]))
    assert float(result["eddy_fraction"]) > 0.0
    assert float(result["window_mean_basin_ke_m2_s2"]) > 0.0
    assert np.all(result["time_mean_speed_m_s"][wet] == 0.0)
    expected_rms = np.sqrt(np.mean(raw_u[15:35] ** 2 + raw_v[15:35] ** 2, axis=0))
    assert np.allclose(result["time_mean_rms_native_speed_m_s"][wet], expected_rms[wet])
    wrong_context_rms = np.sqrt(np.mean(raw_u ** 2 + raw_v ** 2, axis=0))
    assert np.any(expected_rms[wet] != wrong_context_rms[wet])
    assert float(result["ssh_zero_crossing_row"]) == 2.5


def test_preregistered_score_has_two_relative_and_two_row_bars():
    base = {
        "window_mean_basin_ke_m2_s2": np.asarray(1.0),
        "eddy_fraction": np.asarray(0.1),
        "absolute_u_jet_row": np.asarray(10),
        "ssh_zero_crossing_row": np.asarray(11.0),
    }
    report = CLIMATE.score_statistics({"NEMO": base, "legoESM": dict(base)})
    assert report["verdict"] == "PASS"
    assert [row["bar"] for row in report["rows"]] == [0.2, 0.2, 2.0, 2.0]


def test_preregistered_window_and_gpu_run_metadata_fail_closed():
    CLIMATE.require_registered_window(1461, 365)
    with pytest.raises(Exception, match="preregistered"):
        CLIMATE.require_registered_window(1, 365)
    registration = CLIMATE.load_registered_legoesm_run("pre_fix_7b772afb8")
    manifest = {
        "legoesm_git_sha": registration["commit"],
        "mode": "gpu-native",
        "backend": "gpu",
        "precision_policy": "fp64",
        "transcendentals": "native",
        "certified_policy": False,
        "daily_mean_convention": {
            "surface_temperature_c": CLIMATE.LEGOESM_SST_CONVENTION,
        },
    }
    CLIMATE._require_legoesm_run_metadata(manifest, registration)
    for field, bad_value in (
        ("legoesm_git_sha", "wrong"),
        ("backend", "cpu"),
        ("precision_policy", "fp32"),
        ("transcendentals", "certified"),
        ("certified_policy", True),
    ):
        candidate = dict(manifest)
        candidate[field] = bad_value
        with pytest.raises(Exception):
            CLIMATE._require_legoesm_run_metadata(candidate, registration)


def test_run_registry_refuses_unknown_label_and_metadata_mismatch(tmp_path):
    with pytest.raises(Exception, match="unregistered"):
        CLIMATE.load_registered_legoesm_run("post_fix_not_yet_acquired")
    registry = {
        "schema": "gyre-hires-registered-runs-v1",
        "runs": {
            "test": {
                "run_directory": str(tmp_path),
                "commit": "a" * 40,
                "deck": "r4_smag",
                "mode": "gpu-native",
                "backend": "gpu",
                "precision_policy": "fp64",
                "transcendentals": "native",
                "certified_policy": False,
                "physical_gpu_index": 2,
                "label": "test",
            }
        },
    }
    path = tmp_path / "registry.json"
    path.write_text(__import__("json").dumps(registry))
    registered = CLIMATE.load_registered_legoesm_run("test", path)
    bad_manifest = dict(registered, legoesm_git_sha="a" * 40, physical_gpu_index=1)
    with pytest.raises(Exception, match="physical GPU"):
        CLIMATE._require_legoesm_run_metadata(bad_manifest, registered)


def test_post_fix_run_registration_pins_fix_commit_and_gpu_zero():
    registered = CLIMATE.load_registered_legoesm_run("post_fix_b401ec3f3")
    assert registered["commit"] == "b401ec3f3b794b6e4918db372a7cacc802307e15"
    assert registered["physical_gpu_index"] == 0
    assert "post_fix" in registered["run_directory"]


def test_report_only_pattern_comparison_uses_common_finite_support():
    score = CLIMATE.paired_comparison(
        np.array([1.0, 2.0, np.nan]),
        np.array([2.0, 4.0, np.nan]),
    )
    assert score["common_points"] == 2
    assert score["bias"] == 1.5
    assert score["rmse"] == pytest.approx(np.sqrt(2.5))
    assert score["correlation"] == pytest.approx(1.0)


def test_legoesm_provider_every_registered_refusal_fires(tmp_path):
    """Manifest/deck/clock/sequence/digest/non-finite controls are executable."""
    for mutation, pattern, iterate in (
        ("manifest", "incomplete", False),
        ("deck", "r4_smag", False),
        ("manifest-cadence", "manifest daily cadence", False),
        ("namelist-cadence", "r4_smag at nn_write", False),
        ("cadence", "embedded clock", True),
        ("time-counter", "embedded clock", True),
        ("step-counter", "embedded daily cadence", True),
        ("sample-count", "embedded daily cadence", True),
        ("precision", "precision_policy", False),
        ("file-sequence", "file sequence", False),
        ("digest", "digest mismatch", False),
        ("nonfinite", "non-finite", True),
    ):
        root = tmp_path / mutation
        registry = _write_legoesm_provider_fixture(root, mutation=mutation)
        if iterate:
            provider = CLIMATE.legoesm_npz_provider(
                "test", 1, start_day=1, registry_path=registry)
            with pytest.raises(Exception, match=pattern):
                list(provider.iterate())
        else:
            with pytest.raises(Exception, match=pattern):
                CLIMATE.legoesm_npz_provider(
                    "test", 1, start_day=1, registry_path=registry)

    # Positive control: the same fixture without a plant yields exactly one
    # finite, correctly clocked day.  This keeps the refusals able to fail.
    registry = _write_legoesm_provider_fixture(tmp_path / "clean")
    provider = CLIMATE.legoesm_npz_provider(
        "test", 1, start_day=1, registry_path=registry)
    frames = list(provider.iterate())
    assert len(frames) == 1 and frames[0].day == 1
