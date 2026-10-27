import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT / "scripts/validate/ocean_fidelity/testcases/"
    "nemo_testcase_l2_gyre_hires_budget_preflight.py"
)
SPEC = importlib.util.spec_from_file_location("gyre_budget_preflight", SCRIPT)
PREFLIGHT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PREFLIGHT)


def write_nemo_restart(path: Path, *, kt: float = 4320.0) -> None:
    from netCDF4 import Dataset

    with Dataset(path, "w") as dataset:
        for name in sorted(PREFLIGHT.NEMO_REQUIRED_RESTART_VARIABLES):
            variable = dataset.createVariable(name, "f8")
            variable.assignValue(kt if name == "kt" else 1.0)


def write_archive(root: Path, *, drop: str | None = None, non_finite: str | None = None):
    fields = list(PREFLIGHT.LEGO_FULL_FIELDS)
    manifest = {
        "status": "complete",
        "completed_days": 180,
        "per_step": True,
        "full_state_window": [1, 180],
        "full_state_fields": fields,
        "namelist": {"nn_GYRE": 4, "ln_dynldf_blp": True},
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    for day in (1, 30, 180):
        arrays = {}
        for key in PREFLIGHT.required_legoesm_keys():
            if key != drop:
                arrays[key] = np.array([np.nan if key == non_finite else day])
        np.savez(root / f"day{day:04d}.npz", **arrays)


def test_legoesm_archive_passes_only_with_every_registered_field(tmp_path):
    write_archive(tmp_path)
    report = PREFLIGHT.audit_legoesm_archive(tmp_path)
    assert report["status"] == "PASS"
    assert report["errors"] == []

    missing = tmp_path / "missing"
    missing.mkdir()
    write_archive(missing, drop="v_m_s_3d_daily_mean")
    report = PREFLIGHT.audit_legoesm_archive(missing)
    assert report["status"] == "REFUSE"
    assert any("v_m_s_3d_daily_mean" in error for error in report["errors"])


def test_legoesm_archive_refuses_non_finite_active_provider(tmp_path):
    write_archive(tmp_path, non_finite="temperature_c_3d_daily_mean")
    report = PREFLIGHT.audit_legoesm_archive(tmp_path)
    assert report["status"] == "REFUSE"
    assert any("non-finite temperature_c_3d_daily_mean" in error
               for error in report["errors"])


def test_joint_report_cannot_pass_one_refusing_provider(tmp_path, monkeypatch):
    write_archive(tmp_path)
    monkeypatch.setattr(
        PREFLIGHT, "audit_nemo_archive",
        lambda root: {"root": str(root), "status": "REFUSE", "errors": ["plant"]},
    )
    report = PREFLIGHT.build_report(tmp_path, tmp_path / "nemo")
    assert report["verdict"] == "ACQUISITION_NEEDED"
    assert "do not admit" in report["disposition"]


def test_complete_state_archives_do_not_admit_unobserved_production_terms(
        tmp_path, monkeypatch):
    """This is the round-13 regression: full daily state is not a budget."""
    lego = tmp_path / "lego"
    lego.mkdir()
    write_archive(lego)
    monkeypatch.setattr(
        PREFLIGHT, "audit_nemo_archive",
        lambda root: {
            "root": str(root), "status": "PASS", "errors": [],
            "online_tendency_variables": [],
            "trend_flags": {flag: False for flag in PREFLIGHT.NEMO_TREND_FLAGS},
        },
    )
    report = PREFLIGHT.build_report(lego, tmp_path / "nemo")
    assert report["state_archive_status"] == "PASS"
    assert report["budget_observability"]["status"] == "REFUSE"
    assert report["owner"] == report["closure"] == "UNMEASURED"
    assert report["verdict"] == "ACQUISITION_NEEDED"


def test_online_contract_can_pass_and_each_semantic_plant_refuses(
        tmp_path, monkeypatch):
    producer_commit = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True).stdout.strip()
    roots = {provider: tmp_path / provider for provider in ("legoesm", "nemo")}
    for provider, root in roots.items():
        root.mkdir()
        shape = (151, 1, 1)
        terms = {
            key: np.zeros(shape) for key in PREFLIGHT.BUDGET_TERM_KEYS}
        np.savez(
            root / "terms.npz",
            day=np.arange(30, 181),
            surface_v_wet_mask=np.ones((1, 1), dtype=bool),
            **terms,
        )
        passive = {
            "provider": provider, "status": "PASS", "unequal_words": 0,
            "compared_words": 10,
        }
        if provider == "nemo":
            control = tmp_path / "control"
            control.mkdir(exist_ok=True)
            (root / "nemo").write_bytes(b"same-binary")
            (control / "nemo").write_bytes(b"same-binary")
            write_nemo_restart(root / "instrumented.nc")
            (control / "control.nc").write_bytes(
                (root / "instrumented.nc").read_bytes())
            passive.update({
                "control_sha256": PREFLIGHT.sha256(control / "control.nc"),
                "instrumented_sha256": PREFLIGHT.sha256(
                    root / "instrumented.nc"),
                "control_restart": "control.nc",
                "instrumented_restart": "instrumented.nc",
                "compared_variables": sorted(
                    PREFLIGHT.NEMO_REQUIRED_RESTART_VARIABLES),
                "compared_words": len(
                    PREFLIGHT.NEMO_REQUIRED_RESTART_VARIABLES),
            })
        else:
            passive.update({
                "window_entry_day": 30, "steps": 24,
                "compared_leaves": 3,
            })
        (root / "passive.json").write_text(json.dumps(passive))
        (root / "preregistration.json").write_text(json.dumps({
            "terms": list(PREFLIGHT.REGISTERED_TERMS),
            "closure_bar_abs_m_s": PREFLIGHT.REGISTERED_CLOSURE_BAR_ABS_M_S,
            "registered_before_data": True,
            "registration_commit": PREFLIGHT.REGISTRATION_COMMIT,
            "source_sha256": PREFLIGHT.REGISTRATION_SOURCE_SHA256,
        }))
        if provider == "legoesm":
            source_rows = {
                path: PREFLIGHT.git_blob_sha256(producer_commit, path)
                for path in PREFLIGHT.LEGO_SOURCE_PATHS
            }
        else:
            source_rows = {
                **{
                    f"repo:{path}": PREFLIGHT.git_blob_sha256(
                        producer_commit, path)
                    for path in PREFLIGHT.NEMO_TRACKED_SOURCE_PATHS
                },
                **PREFLIGHT.NEMO_EXTERNAL_SOURCE_HASHES,
            }
        assert all(source_rows.values())
        (root / "source.sha256").write_text("".join(
            f"{digest}  {name}\n" for name, digest in source_rows.items()))
        contract = {
            "provider": provider,
            **PREFLIGHT.BUDGET_CONTRACT_REQUIREMENTS,
            "producer_commit": producer_commit,
            "deck": "r4_smag",
            "precision_flavour": (
                "native-fp64-gpu" if provider == "legoesm"
                else "vector-math-conda"),
            "cadence": dict(PREFLIGHT.BUDGET_CADENCE),
            "closure_bar_abs_m_s": PREFLIGHT.REGISTERED_CLOSURE_BAR_ABS_M_S,
            "closure_max_abs_m_s": 0.0,
            "data_artifact": "terms.npz",
            "data_sha256": PREFLIGHT.sha256(root / "terms.npz"),
            "passive_identity_artifact": "passive.json",
            "passive_identity_sha256": PREFLIGHT.sha256(root / "passive.json"),
            "preregistration_artifact": "preregistration.json",
            "preregistration_sha256": PREFLIGHT.sha256(
                root / "preregistration.json"),
            "writer_source_manifest_artifact": "source.sha256",
            "writer_source_manifest_sha256": PREFLIGHT.sha256(
                root / "source.sha256"),
        }
        if provider == "legoesm":
            contract["physical_gpu_index"] = 0
            contract["resolved_namelist"] = {
                "nn_GYRE": 4,
                "rn_Dt_s": 3600.0,
                "rn_Ld_m": 25000.0,
                "ln_dynldf_lap": False,
                "ln_dynldf_blp": True,
                "ln_dynldf_lev": True,
                "nn_ahm_ijk_t": 32,
                "rn_csmc": 3.5,
                "rn_minfac": 1.0,
                "rn_maxfac": 1.0,
            }
        else:
            namelist = root / "resolved_namelist_lines.txt"
            namelist.write_text(
                "nn_GYRE = 4\nrn_Dt =  3600.0000000000000,\n"
                "rn_Ld =  25000.000000000000,\n"
                "ln_dynldf_lap = .false.\nln_dynldf_blp = .true.\n"
                "ln_dynldf_lev = .true.\nnn_ahm_ijk_t = 32\n"
                "rn_csmc =  3.5000000000000000,\n"
                "rn_minfac =  1.0000000000000000,\n"
                "rn_maxfac =  1.0000000000000000,\n")
            contract["resolved_namelist_artifact"] = namelist.name
            contract["resolved_namelist_sha256"] = PREFLIGHT.sha256(namelist)
            binary_digest = PREFLIGHT.sha256(root / "nemo")
            contract["binary_sha256"] = binary_digest
            contract["control_binary_sha256"] = binary_digest
            contract["instrumented_binary_sha256"] = binary_digest
            contract["same_binary_control"] = True
            contract["nemo_engine_commit"] = PREFLIGHT.NEMO_ENGINE_COMMIT
        (root / PREFLIGHT.BUDGET_CONTRACT).write_text(json.dumps(contract))

    monkeypatch.setattr(
        PREFLIGHT, "audit_legoesm_archive",
        lambda root: {"root": str(root), "status": "PASS", "errors": []},
    )
    monkeypatch.setattr(
        PREFLIGHT, "audit_nemo_archive",
        lambda root: {"root": str(root), "status": "PASS", "errors": []},
    )
    assert PREFLIGHT.build_report(roots["legoesm"], roots["nemo"])["verdict"] == "PASS"
    nemo_contract = json.loads(
        (roots["nemo"] / PREFLIGHT.BUDGET_CONTRACT).read_text())

    # The scalar twin is explicit: its source manifest must replace (not add
    # to) the vector arch row, and the caller must request that flavour.
    nemo_source = roots["nemo"] / "source.sha256"
    vector_source = nemo_source.read_text()
    scalar_source = vector_source.replace(
        PREFLIGHT.NEMO_EXTERNAL_SOURCE_HASHES[
            "oracle:arch/arch-conda.fcm"]
        + "  oracle:arch/arch-conda.fcm",
        PREFLIGHT.NEMO_SCALARMATH_ARCH[
            "oracle:arch/arch-conda-scalarmath.fcm"]
        + "  oracle:arch/arch-conda-scalarmath.fcm",
    )
    assert scalar_source != vector_source
    nemo_source.write_text(scalar_source)
    scalar_contract = dict(nemo_contract)
    scalar_contract["precision_flavour"] = "scalar-math-conda"
    scalar_contract["writer_source_manifest_sha256"] = PREFLIGHT.sha256(
        nemo_source)
    (roots["nemo"] / PREFLIGHT.BUDGET_CONTRACT).write_text(
        json.dumps(scalar_contract))
    assert PREFLIGHT.audit_budget_contract(
        roots["nemo"], "nemo", "scalar-math-conda")["status"] == "PASS"
    assert PREFLIGHT.audit_budget_contract(
        roots["nemo"], "nemo")["status"] == "REFUSE"
    nemo_source.write_text(vector_source)
    (roots["nemo"] / PREFLIGHT.BUDGET_CONTRACT).write_text(
        json.dumps(nemo_contract))

    for key in (
        "status", "window_days", "terms", "production_online",
        "external_mode_included",
        "implicit_zdf_stress_separated", "closure_total_recorded",
        "passive_trajectory_identity",
    ):
        planted = dict(nemo_contract)
        planted[key] = False if isinstance(planted[key], bool) else None
        (roots["nemo"] / PREFLIGHT.BUDGET_CONTRACT).write_text(
            json.dumps(planted))
        assert PREFLIGHT.build_report(
            roots["legoesm"], roots["nemo"])["verdict"] == "ACQUISITION_NEEDED"

    for key, bad in (
        ("provider", "legoesm"),
        ("producer_commit", "not-a-sha"),
        ("deck", "r4"),
        ("precision_flavour", "scalar-math"),
        ("cadence", {}),
        ("closure_bar_abs_m_s", None),
        ("data_sha256", "0" * 64),
        ("passive_identity_sha256", "0" * 64),
        ("preregistration_sha256", "0" * 64),
        ("writer_source_manifest_sha256", "0" * 64),
        ("producer_commit", "0" * 40),
        ("binary_sha256", "not-a-digest"),
        ("control_binary_sha256", "c" * 64),
        ("same_binary_control", False),
        ("nemo_engine_commit", "0" * 40),
    ):
        planted = dict(nemo_contract)
        planted[key] = bad
        (roots["nemo"] / PREFLIGHT.BUDGET_CONTRACT).write_text(
            json.dumps(planted))
        assert PREFLIGHT.build_report(
            roots["legoesm"], roots["nemo"])["verdict"] == "ACQUISITION_NEEDED"
    (roots["nemo"] / PREFLIGHT.BUDGET_CONTRACT).write_text(
        json.dumps(nemo_contract))

    # The resolved artifact is a value gate, not a spelling gate: padded
    # list-directed output must pass, while a self-consistently rehashed
    # changed value must still refuse.
    namelist_path = roots["nemo"] / "resolved_namelist_lines.txt"
    good_namelist = namelist_path.read_text()
    namelist_path.write_text(good_namelist.replace(
        "3.5000000000000000", "3.6000000000000000"))
    planted = dict(nemo_contract)
    planted["resolved_namelist_sha256"] = PREFLIGHT.sha256(namelist_path)
    (roots["nemo"] / PREFLIGHT.BUDGET_CONTRACT).write_text(
        json.dumps(planted))
    assert PREFLIGHT.build_report(
        roots["legoesm"], roots["nemo"])["verdict"] == "ACQUISITION_NEEDED"
    namelist_path.write_text(good_namelist)
    (roots["nemo"] / PREFLIGHT.BUDGET_CONTRACT).write_text(
        json.dumps(nemo_contract))

    # Self-consistent invented source and binary digests used to pass because
    # only their syntax was checked.  Bind both to admitted artifacts/blobs.
    nemo_root = roots["nemo"]
    source_original = (nemo_root / "source.sha256").read_bytes()
    (nemo_root / "source.sha256").write_text(f"{'a' * 64}  writer.py\n")
    planted = dict(nemo_contract)
    planted["writer_source_manifest_sha256"] = PREFLIGHT.sha256(
        nemo_root / "source.sha256")
    (nemo_root / PREFLIGHT.BUDGET_CONTRACT).write_text(json.dumps(planted))
    assert PREFLIGHT.build_report(
        roots["legoesm"], nemo_root)["verdict"] == "ACQUISITION_NEEDED"
    (nemo_root / "source.sha256").write_bytes(source_original)

    planted = dict(nemo_contract)
    for key in ("binary_sha256", "control_binary_sha256",
                "instrumented_binary_sha256"):
        planted[key] = "c" * 64
    (nemo_root / PREFLIGHT.BUDGET_CONTRACT).write_text(json.dumps(planted))
    assert PREFLIGHT.build_report(
        roots["legoesm"], nemo_root)["verdict"] == "ACQUISITION_NEEDED"

    passive_original = json.loads((nemo_root / "passive.json").read_text())
    planted_passive = dict(passive_original)
    planted_passive["control_sha256"] = planted_passive[
        "instrumented_sha256"] = "c" * 64
    planted_passive["compared_variables"] = []
    (nemo_root / "passive.json").write_text(json.dumps(planted_passive))
    planted = dict(nemo_contract)
    planted["passive_identity_sha256"] = PREFLIGHT.sha256(
        nemo_root / "passive.json")
    (nemo_root / PREFLIGHT.BUDGET_CONTRACT).write_text(json.dumps(planted))
    assert PREFLIGHT.build_report(
        roots["legoesm"], nemo_root)["verdict"] == "ACQUISITION_NEEDED"
    (nemo_root / "passive.json").write_text(json.dumps(passive_original))
    (nemo_root / PREFLIGHT.BUDGET_CONTRACT).write_text(
        json.dumps(nemo_contract))

    # Matching hashes and a self-consistent JSON claim cannot turn arbitrary
    # bytes into a valid pair of NEMO restart providers.
    control_restart = tmp_path / "control" / "control.nc"
    instrumented_restart = nemo_root / "instrumented.nc"
    restart_control_original = control_restart.read_bytes()
    restart_instrumented_original = instrumented_restart.read_bytes()
    control_restart.write_bytes(b"same-invalid-restart")
    instrumented_restart.write_bytes(b"same-invalid-restart")
    planted_passive = dict(passive_original)
    planted_passive["control_sha256"] = PREFLIGHT.sha256(control_restart)
    planted_passive["instrumented_sha256"] = PREFLIGHT.sha256(
        instrumented_restart)
    (nemo_root / "passive.json").write_text(json.dumps(planted_passive))
    planted = dict(nemo_contract)
    planted["passive_identity_sha256"] = PREFLIGHT.sha256(
        nemo_root / "passive.json")
    (nemo_root / PREFLIGHT.BUDGET_CONTRACT).write_text(json.dumps(planted))
    assert PREFLIGHT.build_report(
        roots["legoesm"], nemo_root)["verdict"] == "ACQUISITION_NEEDED"
    control_restart.write_bytes(restart_control_original)
    instrumented_restart.write_bytes(restart_instrumented_original)
    (nemo_root / "passive.json").write_text(json.dumps(passive_original))
    (nemo_root / PREFLIGHT.BUDGET_CONTRACT).write_text(
        json.dumps(nemo_contract))

    # A correctly hashed but non-closing active term must still refuse.
    arrays = {
        key: np.zeros((151, 1, 1)) for key in PREFLIGHT.BUDGET_TERM_KEYS}
    arrays[PREFLIGHT.BUDGET_TERM_KEYS[0]][0, 0, 0] = 1.0
    np.savez(
        nemo_root / "terms.npz", day=np.arange(30, 181),
        surface_v_wet_mask=np.ones((1, 1), dtype=bool), **arrays)
    planted = dict(nemo_contract)
    planted["data_sha256"] = PREFLIGHT.sha256(nemo_root / "terms.npz")
    (nemo_root / PREFLIGHT.BUDGET_CONTRACT).write_text(json.dumps(planted))
    assert PREFLIGHT.build_report(
        roots["legoesm"], nemo_root)["verdict"] == "ACQUISITION_NEEDED"

    # A correctly hashed passive artifact with zero compared state words is
    # still vacuous and must refuse.
    np.savez(
        nemo_root / "terms.npz", day=np.arange(30, 181),
        surface_v_wet_mask=np.ones((1, 1), dtype=bool),
        **{key: np.zeros((151, 1, 1))
           for key in PREFLIGHT.BUDGET_TERM_KEYS})
    empty_passive = json.loads((nemo_root / "passive.json").read_text())
    empty_passive["compared_words"] = 0
    (nemo_root / "passive.json").write_text(json.dumps(empty_passive))
    planted = dict(nemo_contract)
    planted["data_sha256"] = PREFLIGHT.sha256(nemo_root / "terms.npz")
    planted["passive_identity_sha256"] = PREFLIGHT.sha256(
        nemo_root / "passive.json")
    (nemo_root / PREFLIGHT.BUDGET_CONTRACT).write_text(json.dumps(planted))
    assert PREFLIGHT.build_report(
        roots["legoesm"], nemo_root)["verdict"] == "ACQUISITION_NEEDED"


def test_separate_budget_roots_are_used(tmp_path, monkeypatch):
    monkeypatch.setattr(
        PREFLIGHT, "audit_legoesm_archive",
        lambda root: {"root": str(root), "status": "PASS", "errors": []},
    )
    monkeypatch.setattr(
        PREFLIGHT, "audit_nemo_archive",
        lambda root: {"root": str(root), "status": "PASS", "errors": []},
    )
    calls = []

    def fake_budget(root, provider):
        calls.append((root, provider))
        return {"root": str(root), "status": "REFUSE", "errors": ["plant"]}

    monkeypatch.setattr(PREFLIGHT, "audit_budget_contract", fake_budget)
    lego_budget = tmp_path / "lego-budget"
    nemo_budget = tmp_path / "nemo-budget"
    report = PREFLIGHT.build_report(
        tmp_path / "lego-state", tmp_path / "nemo-state",
        lego_budget, nemo_budget)
    assert calls == [(lego_budget, "legoesm"), (nemo_budget, "nemo")]
    assert report["verdict"] == "ACQUISITION_NEEDED"


def test_daily_means_cannot_identify_a_nonlinear_daily_mean_product():
    control = PREFLIGHT.nonlinear_daily_mean_control()
    assert control["provider_means_a"] == control["provider_means_b"] == [0.0, 0.0]
    assert control["mean_product_a"] == 1.0
    assert control["mean_product_b"] == -1.0
    assert control["ambiguity"] == 2.0
    assert control["control_fires"] is True
