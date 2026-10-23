#!/usr/bin/env python3
"""Audit whether stored GYRE runs can support the registered momentum budget."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

START_DAY = 30
END_DAY = 180
FULL_WINDOW = (1, END_DAY)
LEGO_FULL_FIELDS = (
    "temperature_c_3d",
    "salinity_psu_3d",
    "u_m_s_3d",
    "v_m_s_3d",
    "tke_m2_s2_3d",
    "layer_thickness_m_3d",
)
NEMO_REQUIREMENTS = {
    "T": ("votemper", "vosaline", "vovvle3t", "sossheig"),
    "U": ("vozocrtx",),
    "V": ("vomecrty",),
    "W": ("votkeavt", "votkeavm", "vovecrtz"),
}
REGISTERED_TERMS = (
    "advection",
    "pressure_kinetic_energy_gradient",
    "total_vorticity_coriolis",
    "smagorinsky_lateral_diffusion",
    "vertical_mixing",
    "surface_stress",
    "external_mode_correction",
    "total_increment",
)
BUDGET_TERM_KEYS = tuple(
    f"surface_v_{term}_daily_mean" for term in REGISTERED_TERMS)
BUDGET_DAY_KEY = "day"
BUDGET_MASK_KEY = "surface_v_wet_mask"
BUDGET_CONTRACT = "momentum_budget_manifest.json"
REGISTERED_CLOSURE_BAR_ABS_M_S = 1.0e-12
REGISTRATION_COMMIT = "3293fd5f49c332038cc61735ed1dc7bbc0f7b798"
REGISTRATION_SOURCE_SHA256 = (
    "480a45f3d7f5e02bd5c4771316bae668e59ee33aaeab87a73c47a6212edf5174"
)
NEMO_ENGINE_COMMIT = "dcc7fb8c1779fa8409e41e4ce3ab7d45b9ceb796"
REPO = Path(__file__).resolve().parents[4]
LEGO_SOURCE_PATHS = (
    "scripts/run/run_gyre_hires.py",
    "packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py",
    "scripts/run/run_gyre_hires_budget_gpu.sh",
    "docs/ocean/fidelity/testcases/"
    "nemo_testcases_l2_gyre_hires_round14_preregistration.md",
)
NEMO_TRACKED_SOURCE_PATHS = (
    "scripts/validate/ocean_fidelity/testcases/"
    "nemo_testcase_l2_gyre_hires_budget/gyre_r14_budget.F90",
    "scripts/validate/ocean_fidelity/testcases/"
    "nemo_testcase_l2_gyre_hires_budget/stprk3_stg_round14.patch",
    "scripts/validate/ocean_fidelity/testcases/"
    "nemo_testcase_l2_gyre_hires_budget/dynadv_round14.patch",
    "scripts/validate/ocean_fidelity/testcases/"
    "nemo_testcase_l2_gyre_hires_budget/dynzdf_round14.patch",
    "scripts/validate/ocean_fidelity/testcases/"
    "nemo_testcase_l2_gyre_hires_budget/namelist_cfg_r4_smag_180d.patch",
    "scripts/validate/ocean_fidelity/testcases/"
    "nemo_testcase_l2_gyre_hires_budget/package_records.py",
    "scripts/validate/ocean_fidelity/testcases/"
    "nemo_testcase_l2_gyre_hires_budget/run.sh",
    "scripts/validate/ocean_fidelity/testcases/"
    "nemo_testcase_l2_gyre_hires/namelist_cfg_r4_smag_kt10.patch",
    "scripts/validate/ocean_fidelity/testcases/"
    "nemo_testcase_l2_gyre_hires/deck.sh",
    "scripts/validate/ocean_fidelity/testcases/configs/"
    "gyre_omip_l2_phase3_namelist_cfg",
    "docs/ocean/fidelity/testcases/"
    "nemo_testcases_l2_gyre_hires_round14_preregistration.md",
)
NEMO_EXTERNAL_SOURCE_HASHES = {
    "oracle:cfgs/GYRE_OMIP_L2_P3/MY_SRC/dynhpg.F90":
        "6c15901bfcb58e768523ef1afab7f8365085afe6a8e86be810a7ee2078e4bd73",
    "oracle:cfgs/GYRE_OMIP_L2_P3/MY_SRC/dynspg_ts.F90":
        "c50912ba02772fe11eb8a164b05b05077a993ca88058e5ed01aad1d16d6b8b82",
    "oracle:cfgs/GYRE_OMIP_L2_P3/MY_SRC/dynvor.F90":
        "06b1fd76c44e62d926e896e53006153ace4ec87d759b7d17765ad37462c6b8de",
    "oracle:cfgs/GYRE_OMIP_L2_P3/MY_SRC/eosbn2.F90":
        "8ffe1b56ce90b8924ecebc37c618c6be67a521aa190c102d48e136533fcf5067",
    "oracle:cfgs/GYRE_OMIP_L2_P3/MY_SRC/stprk3.F90":
        "11c6a66407eda69f3c8ceb57a37ed5f6565cefe4d8e32c0dee453ad021906fee",
    "oracle:cfgs/GYRE_OMIP_L2_P3/MY_SRC/stprk3_stg.F90":
        "4d3493de0e837654e961a30ac4684801d87b4226ed9e8c226922db780d699afe",
    "oracle:cfgs/GYRE_OMIP_L2_P3/MY_SRC/traadv.F90":
        "4807fa59ce15237ac04ef8e84f39c77b3da4080aa9ad1effa517fb79e1a6968d",
    "oracle:cfgs/GYRE_OMIP_L2_P3/EXP00/namelist_ref":
        "b04f2ce4d12aa247ae3b707483d25cf2d38d0b3cf8817d71d33e66918fd81cfd",
    "oracle:cfgs/GYRE_OMIP_L2_P3/cpp_GYRE_OMIP_L2_P3.fcm":
        "54bd2cead92cee1eaa8cb257c7f7cdea8fe39cc9fb0f47c7c0d5853b909b34b7",
    "oracle:src/OCE/DYN/dynadv.F90":
        "d6802c22ce6a3b0307b556cbb20326f04f916711e2288f68e7a67dc7beb94a6a",
    "oracle:src/OCE/DYN/dynzdf.F90":
        "4489773bb4cafe46123cfd658b1561188215c9ce0db11f6859fb27cb019bb174",
    "oracle:arch/arch-conda.fcm":
        "64cf1b90f611936bbb92a7514800f8a8e9c8365be7b3cf9a9963f1615c5c836a",
}
NEMO_SCALARMATH_ARCH = {
    "oracle:arch/arch-conda-scalarmath.fcm":
        "132f7a0500c4f0e86d8d3bf7864974a82e1dea5d83166dcfdfaf409e2ca04561",
}
NEMO_REQUIRED_RESTART_VARIABLES = {
    "kt", "sshn", "un", "vn", "tn", "sn", "en", "avt_k", "avm_k"}
BUDGET_CONTRACT_REQUIREMENTS = {
    "status": "complete",
    "window_days": [START_DAY, END_DAY],
    "terms": list(REGISTERED_TERMS),
    "production_online": True,
    "external_mode_included": True,
    "implicit_zdf_stress_separated": True,
    "closure_total_recorded": True,
    "passive_trajectory_identity": True,
}
BUDGET_CADENCE = {
    "dt_s": 3600.0,
    "steps_per_day": 24,
    "first_day": START_DAY,
    "last_day": END_DAY,
    "first_step": 697,
    "last_step": 4320,
    "samples_per_record": 24,
    "records": END_DAY - START_DAY + 1,
}
LEGO_CARRIED_ZDF_FIELDS = (
    "tke_avm",
    "tke_avt",
    "tke_dissl",
    "tke_avm_surface",
)
NEMO_TREND_FLAGS = ("ln_glo_trd", "ln_dyn_trd", "ln_KE_trd", "ln_vor_trd")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolved_namelist_number(text: str, name: str) -> float | None:
    """Read one scalar numeric assignment from NEMO's resolved-name dump."""
    match = re.search(
        rf"\b{re.escape(name)}\s*=\s*"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[de][-+]?\d+)?)",
        text,
        re.IGNORECASE,
    )
    if match is None:
        return None
    try:
        return float(match.group(1).replace("d", "e").replace("D", "E"))
    except ValueError:
        return None


def producer_descends_from_registration(producer_commit: str) -> bool:
    """Return true only for a local commit containing the preregistration."""
    if re.fullmatch(r"[0-9a-f]{40}", producer_commit) is None:
        return False
    return subprocess.run(
        ["git", "-C", str(REPO), "merge-base", "--is-ancestor",
         REGISTRATION_COMMIT, producer_commit],
        check=False, capture_output=True).returncode == 0


def git_blob_sha256(commit: str, path: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{commit}:{path}"],
        check=False, capture_output=True)
    if result.returncode != 0:
        return None
    return hashlib.sha256(result.stdout).hexdigest()


def audit_source_manifest(
    path: Path, provider: str, producer_commit: str,
    nemo_precision_flavour: str = "vector-math-conda",
) -> list[str]:
    """Bind the declared source set to producer blobs or pinned oracle files."""
    errors: list[str] = []
    rows: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            errors.append("writer source manifest has a malformed row")
            continue
        digest, name = match.groups()
        if name in rows:
            errors.append(f"writer source manifest repeats {name}")
        rows[name] = digest
    if provider == "legoesm":
        expected_names = set(LEGO_SOURCE_PATHS)
        expected = {
            name: git_blob_sha256(producer_commit, name)
            for name in LEGO_SOURCE_PATHS
        }
    else:
        external = dict(NEMO_EXTERNAL_SOURCE_HASHES)
        if nemo_precision_flavour == "scalar-math-conda":
            external.pop("oracle:arch/arch-conda.fcm")
            external.update(NEMO_SCALARMATH_ARCH)
        expected_names = {
            *(f"repo:{name}" for name in NEMO_TRACKED_SOURCE_PATHS),
            *external,
        }
        expected = {
            **{
                f"repo:{name}": git_blob_sha256(producer_commit, name)
                for name in NEMO_TRACKED_SOURCE_PATHS
            },
            **external,
        }
    if set(rows) != expected_names:
        errors.append(
            "writer source manifest registry differs: "
            f"missing={sorted(expected_names - set(rows))}, "
            f"extra={sorted(set(rows) - expected_names)}")
    for name in sorted(expected_names & set(rows)):
        if expected[name] is None or rows[name] != expected[name]:
            errors.append(f"writer source manifest digest differs for {name}")
    return errors


def required_legoesm_keys() -> tuple[str, ...]:
    return tuple(
        key
        for name in LEGO_FULL_FIELDS
        for key in (name, f"{name}_daily_mean")
    )


def audit_nemo_restart_pair(control: Path, instrumented: Path) -> dict:
    """Independently reopen and compare the two terminal NEMO restarts."""
    from netCDF4 import Dataset

    with Dataset(control) as left, Dataset(instrumented) as right:
        left.set_auto_mask(False)
        right.set_auto_mask(False)
        left_names = {
            name for name, variable in left.variables.items()
            if np.dtype(variable.dtype).kind in "biufc"
        }
        right_names = {
            name for name, variable in right.variables.items()
            if np.dtype(variable.dtype).kind in "biufc"
        }
        if left_names != right_names:
            raise ValueError(
                "restart numeric-variable registry changed: "
                f"only_control={sorted(left_names - right_names)}, "
                f"only_instrumented={sorted(right_names - left_names)}")
        missing = sorted(NEMO_REQUIRED_RESTART_VARIABLES - left_names)
        if missing:
            raise ValueError(
                "restart lacks required prognostic providers: "
                + ", ".join(missing))

        compared_words = 0
        unequal_words = 0
        for name in sorted(left_names):
            a = np.asarray(left.variables[name][:])
            b = np.asarray(right.variables[name][:])
            if a.shape != b.shape or a.dtype != b.dtype:
                raise ValueError(
                    f"restart variable {name} changed shape or dtype")
            a_words = np.ascontiguousarray(a).view(np.uint8).reshape(
                a.size, a.dtype.itemsize)
            b_words = np.ascontiguousarray(b).view(np.uint8).reshape(
                b.size, b.dtype.itemsize)
            unequal_words += int(np.count_nonzero(
                np.any(a_words != b_words, axis=1)))
            compared_words += int(a.size)

        kt = np.asarray(left.variables["kt"][:])
        if kt.size != 1 or float(kt.reshape(-1)[0]) != 4320.0:
            raise ValueError("restart terminal kt is not exactly 4320")
        if compared_words <= 0:
            raise ValueError("restart comparison contains zero numeric words")
        return {
            "compared_variables": sorted(left_names),
            "compared_words": compared_words,
            "unequal_words": unequal_words,
        }


def audit_legoesm_archive(root: Path) -> dict:
    """Return a fail-closed audit of the matched 3-D legoESM archive."""
    manifest_path = root / "manifest.json"
    errors: list[str] = []
    manifest = None
    if not manifest_path.is_file():
        errors.append("missing manifest.json")
    else:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") != "complete":
            errors.append("manifest status is not complete")
        if manifest.get("completed_days", 0) < END_DAY:
            errors.append(f"manifest has fewer than {END_DAY} completed days")
        if manifest.get("per_step") is not True:
            errors.append("manifest does not certify observer-free per-step integration")
        if manifest.get("full_state_window") != list(FULL_WINDOW):
            errors.append(
                f"manifest full_state_window is not {list(FULL_WINDOW)}")
        if manifest.get("full_state_fields") != list(LEGO_FULL_FIELDS):
            errors.append("manifest full_state_fields do not match the registered set")
        namelist = manifest.get("namelist", {})
        if namelist.get("nn_GYRE") != 4:
            errors.append("manifest does not resolve nn_GYRE=4")
        if namelist.get("ln_dynldf_blp") is not True:
            errors.append("manifest is not the r4_smag momentum deck")

    required = set(required_legoesm_keys())
    day_rows = []
    available_keys: set[str] = set()
    for day in (1, START_DAY, END_DAY):
        path = root / f"day{day:04d}.npz"
        row = {"day": day, "path": str(path), "missing": [], "non_finite": []}
        if not path.is_file():
            errors.append(f"missing {path.name}")
            row["missing"] = sorted(required)
        else:
            with np.load(path) as archive:
                available_keys.update(archive.files)
                row["missing"] = sorted(required - set(archive.files))
                for key in sorted(required & set(archive.files)):
                    if not bool(np.all(np.isfinite(archive[key]))):
                        row["non_finite"].append(key)
            if row["missing"]:
                errors.append(
                    f"{path.name} lacks {', '.join(row['missing'])}")
            if row["non_finite"]:
                errors.append(
                    f"{path.name} has non-finite {', '.join(row['non_finite'])}")
            row["sha256"] = sha256(path)
        day_rows.append(row)
    return {
        "root": str(root.resolve()),
        "manifest_sha256": sha256(manifest_path) if manifest_path.is_file() else None,
        "required_fields": list(LEGO_FULL_FIELDS),
        "available_keys": sorted(available_keys),
        "missing_carried_zdf_fields": sorted(
            set(LEGO_CARRIED_ZDF_FIELDS) - available_keys),
        "sampled_days": day_rows,
        "errors": errors,
        "status": "PASS" if not errors else "REFUSE",
    }


def audit_nemo_archive(root: Path) -> dict:
    """Check tile count, variables, cadence, and placement metadata."""
    from netCDF4 import Dataset

    errors: list[str] = []
    grids = {}
    available_variables: set[str] = set()
    for grid, variables in NEMO_REQUIREMENTS.items():
        paths = sorted(root.glob(f"*_grid_{grid}_*.nc"))
        if len(paths) != 8:
            errors.append(f"grid_{grid} has {len(paths)} tiles, expected 8")
        tile_rows = []
        for path in paths:
            with Dataset(path) as dataset:
                available_variables.update(dataset.variables)
                missing = sorted(set(variables) - set(dataset.variables))
                time_count = len(dataset.dimensions.get("time_counter", ()))
                has_position = "DOMAIN_position_first" in dataset.ncattrs()
                has_global = "DOMAIN_size_global" in dataset.ncattrs()
                if missing:
                    errors.append(f"{path.name} lacks {', '.join(missing)}")
                if time_count < END_DAY:
                    errors.append(
                        f"{path.name} has {time_count} daily records, expected >= {END_DAY}")
                if not (has_position and has_global):
                    errors.append(f"{path.name} lacks DOMAIN placement attributes")
                tile_rows.append({
                    "name": path.name,
                    "time_records": time_count,
                    "missing_variables": missing,
                    "domain_position_first": has_position,
                    "domain_size_global": has_global,
                })
        grids[grid] = tile_rows
    manifest = root / "manifest.sha256"
    if not manifest.is_file():
        errors.append("missing manifest.sha256")
    ocean_output = root / "ocean.output"
    trend_flags = {}
    if ocean_output.is_file():
        output = ocean_output.read_text(errors="replace")
        for flag in NEMO_TREND_FLAGS:
            match = re.search(rf"\b{re.escape(flag)}\s*=\s*([TF])\b", output)
            trend_flags[flag] = None if match is None else match.group(1) == "T"
    else:
        trend_flags = {flag: None for flag in NEMO_TREND_FLAGS}
    return {
        "root": str(root.resolve()),
        "manifest_sha256": sha256(manifest) if manifest.is_file() else None,
        "requirements": {key: list(value) for key, value in NEMO_REQUIREMENTS.items()},
        "available_variables": sorted(available_variables),
        "online_tendency_variables": sorted(
            name for name in available_variables
            if name.startswith(("utrd_", "vtrd_"))),
        "trend_flags": trend_flags,
        "grids": grids,
        "errors": errors,
        "status": "PASS" if not errors else "REFUSE",
    }


def nonlinear_daily_mean_control() -> dict:
    """Exhibit equal daily means with different nonlinear mean products."""
    velocity = np.asarray([-1.0, 1.0])
    gradient_a = np.asarray([-1.0, 1.0])
    gradient_b = np.asarray([1.0, -1.0])
    means_a = [float(np.mean(velocity)), float(np.mean(gradient_a))]
    means_b = [float(np.mean(velocity)), float(np.mean(gradient_b))]
    product_a = float(np.mean(velocity * gradient_a))
    product_b = float(np.mean(velocity * gradient_b))
    return {
        "provider_means_a": means_a,
        "provider_means_b": means_b,
        "provider_means_identical": means_a == means_b,
        "mean_product_a": product_a,
        "mean_product_b": product_b,
        "ambiguity": abs(product_a - product_b),
        "control_fires": means_a == means_b and product_a != product_b,
    }


def audit_budget_contract(
    root: Path,
    provider: str,
    nemo_precision_flavour: str = "vector-math-conda",
) -> dict:
    """Require an online-production contract, not merely daily state means."""
    if nemo_precision_flavour not in (
            "vector-math-conda", "scalar-math-conda"):
        raise ValueError(
            f"unsupported NEMO precision flavour {nemo_precision_flavour!r}")
    path = root / BUDGET_CONTRACT
    errors: list[str] = []
    contract = None
    if not path.is_file():
        errors.append(f"missing {BUDGET_CONTRACT}")
    else:
        contract = json.loads(path.read_text())
        if contract.get("provider") != provider:
            errors.append(f"contract provider is not {provider}")
        for key, expected in BUDGET_CONTRACT_REQUIREMENTS.items():
            if contract.get(key) != expected:
                errors.append(
                    f"contract {key} is {contract.get(key)!r}, expected {expected!r}")
        producer_commit = contract.get("producer_commit")
        if (not isinstance(producer_commit, str)
                or re.fullmatch(r"[0-9a-f]{40}", producer_commit) is None):
            errors.append("contract producer_commit is not a lowercase Git SHA")
        elif not producer_descends_from_registration(producer_commit):
            errors.append(
                "contract producer_commit does not descend from the "
                "preregistration commit")
        if contract.get("deck") != "r4_smag":
            errors.append("contract deck is not r4_smag")
        expected_flavour = {
            "legoesm": "native-fp64-gpu",
            "nemo": nemo_precision_flavour,
        }[provider]
        if contract.get("precision_flavour") != expected_flavour:
            errors.append(
                f"contract precision_flavour is not {expected_flavour}")
        cadence = contract.get("cadence")
        if not isinstance(cadence, dict):
            errors.append("contract cadence is not a mapping")
        else:
            for key, expected in BUDGET_CADENCE.items():
                if cadence.get(key) != expected:
                    errors.append(
                        f"contract cadence {key} is {cadence.get(key)!r}, "
                        f"expected {expected!r}")
        if provider == "legoesm":
            namelist = contract.get("resolved_namelist")
            expected_namelist = {
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
            if not isinstance(namelist, dict):
                errors.append("contract resolved_namelist is not a mapping")
            else:
                for key, expected in expected_namelist.items():
                    if namelist.get(key) != expected:
                        errors.append(
                            f"contract resolved_namelist {key} is "
                            f"{namelist.get(key)!r}, expected {expected!r}")
            gpu = contract.get("physical_gpu_index")
            if isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0:
                errors.append("contract lacks an explicit physical GPU index")
        closure_bar = contract.get("closure_bar_abs_m_s")
        if closure_bar != REGISTERED_CLOSURE_BAR_ABS_M_S:
            errors.append(
                "contract closure_bar_abs_m_s is not the registered "
                f"{REGISTERED_CLOSURE_BAR_ABS_M_S}")
        claimed_closure = contract.get("closure_max_abs_m_s")
        if (isinstance(claimed_closure, bool)
                or not isinstance(claimed_closure, (int, float))
                or not np.isfinite(claimed_closure)
                or claimed_closure < 0
                or claimed_closure > REGISTERED_CLOSURE_BAR_ABS_M_S):
            errors.append("contract closure_max_abs_m_s is absent or over bar")
        verified_artifacts: dict[str, Path] = {}
        for name_key, digest_key in (
            ("data_artifact", "data_sha256"),
            ("passive_identity_artifact", "passive_identity_sha256"),
            ("preregistration_artifact", "preregistration_sha256"),
            ("writer_source_manifest_artifact",
             "writer_source_manifest_sha256"),
        ):
            name = contract.get(name_key)
            digest = contract.get(digest_key)
            if (not isinstance(name, str) or not name
                    or Path(name).name != name):
                errors.append(f"contract {name_key} is not a safe local filename")
                continue
            artifact = root / name
            if not artifact.is_file():
                errors.append(f"contract {name_key} is missing: {name}")
            elif digest != sha256(artifact):
                errors.append(f"contract {digest_key} does not match {name}")
            else:
                verified_artifacts[name_key] = artifact

        source_path = verified_artifacts.get("writer_source_manifest_artifact")
        if source_path is not None:
            errors.extend(audit_source_manifest(
                source_path, provider, producer_commit,
                nemo_precision_flavour))

        if provider == "nemo":
            if contract.get("nemo_engine_commit") != NEMO_ENGINE_COMMIT:
                errors.append(
                    "NEMO contract does not pin the certified engine commit")
            binary_digest = contract.get("binary_sha256")
            if (not isinstance(binary_digest, str)
                    or re.fullmatch(r"[0-9a-f]{64}", binary_digest) is None):
                errors.append("NEMO contract lacks a binary SHA-256")
            elif (contract.get("control_binary_sha256") != binary_digest
                  or contract.get("instrumented_binary_sha256")
                  != binary_digest):
                errors.append(
                    "NEMO control and instrumented binary hashes differ")
            if contract.get("same_binary_control") is not True:
                errors.append("NEMO contract does not prove a same-binary control")
            instrumented_binary = root / "nemo"
            control_binary = root.parent / "control" / "nemo"
            for label, binary in (
                    ("instrumented", instrumented_binary),
                    ("control", control_binary)):
                if not binary.is_file():
                    errors.append(f"NEMO {label} binary artifact is missing")
                elif sha256(binary) != binary_digest:
                    errors.append(f"NEMO {label} binary artifact hash differs")
            name = contract.get("resolved_namelist_artifact")
            digest = contract.get("resolved_namelist_sha256")
            if (not isinstance(name, str) or not name
                    or Path(name).name != name):
                errors.append(
                    "contract resolved_namelist_artifact is not a safe local filename")
            else:
                artifact = root / name
                if not artifact.is_file():
                    errors.append("contract resolved namelist artifact is missing")
                elif digest != sha256(artifact):
                    errors.append("contract resolved_namelist_sha256 does not match")
                else:
                    text = artifact.read_text(errors="replace")
                    for expression in (
                        r"\bnn_GYRE\s*=\s*4\b",
                        r"\bln_dynldf_lap\s*=\s*(?:\.false\.|F)(?:\s|,)",
                        r"\bln_dynldf_blp\s*=\s*(?:\.true\.|T)(?:\s|,)",
                        r"\bln_dynldf_lev\s*=\s*(?:\.true\.|T)(?:\s|,)",
                        r"\bnn_ahm_ijk_t\s*=\s*32\b",
                    ):
                        if re.search(expression, text, re.IGNORECASE) is None:
                            errors.append(
                                "resolved namelist lacks registered expression "
                                f"{expression}")
                    for name, expected in (
                        ("rn_Dt", 3600.0),
                        ("rn_Ld", 25000.0),
                        ("rn_csmc", 3.5),
                        ("rn_minfac", 1.0),
                        ("rn_maxfac", 1.0),
                    ):
                        actual = resolved_namelist_number(text, name)
                        if actual != expected:
                            errors.append(
                                "resolved namelist scalar "
                                f"{name} is {actual!r}, expected {expected!r}")

        data_path = verified_artifacts.get("data_artifact")
        if data_path is not None:
            try:
                with np.load(data_path, allow_pickle=False) as data:
                    required = {
                        BUDGET_DAY_KEY, BUDGET_MASK_KEY, *BUDGET_TERM_KEYS}
                    missing = sorted(required - set(data.files))
                    if missing:
                        errors.append(
                            f"budget data lacks {', '.join(missing)}")
                    else:
                        days = np.asarray(data[BUDGET_DAY_KEY])
                        expected_days = np.arange(START_DAY, END_DAY + 1)
                        if not np.array_equal(days, expected_days):
                            errors.append("budget data day axis is not exactly 30..180")
                        mask = np.asarray(data[BUDGET_MASK_KEY])
                        if mask.ndim != 2 or mask.dtype != np.bool_ or not np.any(mask):
                            errors.append("budget wet mask must be a non-empty 2-D bool array")
                        arrays = [np.asarray(data[key]) for key in BUDGET_TERM_KEYS]
                        expected_shape = (expected_days.size, *mask.shape)
                        for key, values in zip(BUDGET_TERM_KEYS, arrays):
                            if values.shape != expected_shape:
                                errors.append(
                                    f"budget {key} shape is {values.shape}, "
                                    f"expected {expected_shape}")
                            elif not bool(np.all(np.isfinite(values))):
                                errors.append(f"budget {key} has non-finite values")
                        if (all(values.shape == expected_shape for values in arrays)
                                and closure_bar
                                == REGISTERED_CLOSURE_BAR_ABS_M_S):
                            components = np.sum(np.stack(arrays[:-1]), axis=0)
                            closure_max = float(np.max(np.abs(
                                components[:, mask] - arrays[-1][:, mask])))
                            if closure_max > closure_bar:
                                errors.append(
                                    f"budget closure {closure_max} exceeds {closure_bar}")
                            if closure_max != claimed_closure:
                                errors.append(
                                    "contract closure_max_abs_m_s does not "
                                    "match the data artifact")
            except (OSError, ValueError) as exc:
                errors.append(f"cannot read budget data artifact: {exc}")

        passive_path = verified_artifacts.get("passive_identity_artifact")
        if passive_path is not None:
            try:
                passive = json.loads(passive_path.read_text())
                if (passive.get("provider") != provider
                        or passive.get("status") != "PASS"
                        or passive.get("unequal_words") != 0
                        or not isinstance(passive.get("compared_words"), int)
                        or passive.get("compared_words") <= 0):
                    errors.append("passive identity artifact does not prove zero moved words")
                if provider == "nemo":
                    control_digest = passive.get("control_sha256")
                    instrumented_digest = passive.get("instrumented_sha256")
                    if (not isinstance(control_digest, str)
                            or re.fullmatch(r"[0-9a-f]{64}", control_digest) is None
                            or control_digest != instrumented_digest):
                        errors.append(
                            "NEMO passive artifact does not hash identical restarts")
                    compared_variables = passive.get("compared_variables")
                    if (not isinstance(compared_variables, list)
                            or not NEMO_REQUIRED_RESTART_VARIABLES.issubset(
                                compared_variables)):
                        errors.append(
                            "NEMO passive artifact lacks the required provider registry")
                    control_name = passive.get("control_restart")
                    instrumented_name = passive.get("instrumented_restart")
                    if (not isinstance(control_name, str)
                            or Path(control_name).name != control_name
                            or not isinstance(instrumented_name, str)
                            or Path(instrumented_name).name != instrumented_name):
                        errors.append("NEMO passive restart names are unsafe")
                    else:
                        actual_restarts = (
                            (root.parent / "control" / control_name,
                             control_digest),
                            (root / instrumented_name, instrumented_digest),
                        )
                        for restart, digest in actual_restarts:
                            if (not restart.is_file()
                                    or sha256(restart) != digest):
                                errors.append(
                                    f"NEMO passive restart artifact differs: {restart}")
                        if all(restart.is_file() for restart, _ in actual_restarts):
                            try:
                                live = audit_nemo_restart_pair(
                                    actual_restarts[0][0], actual_restarts[1][0])
                                if live["unequal_words"] != 0:
                                    errors.append(
                                        "NEMO passive restarts contain moved words")
                                if (live["compared_variables"]
                                        != compared_variables
                                        or live["compared_words"]
                                        != passive.get("compared_words")):
                                    errors.append(
                                        "NEMO passive summary differs from the "
                                        "reopened restart pair")
                            except (OSError, RuntimeError, ValueError) as exc:
                                errors.append(
                                    f"cannot verify NEMO restart contents: {exc}")
                elif (passive.get("window_entry_day") != START_DAY
                      or passive.get("steps") != 24
                      or not isinstance(passive.get("compared_leaves"), int)
                      or passive.get("compared_leaves") <= 0):
                    errors.append(
                        "legoesm passive artifact does not cover day 30 for 24 steps")
            except (OSError, ValueError) as exc:
                errors.append(f"cannot read passive identity artifact: {exc}")

        prereg_path = verified_artifacts.get("preregistration_artifact")
        if prereg_path is not None:
            try:
                prereg = json.loads(prereg_path.read_text())
                commit = prereg.get("registration_commit")
                if (prereg.get("terms") != list(REGISTERED_TERMS)
                        or prereg.get("closure_bar_abs_m_s") != closure_bar
                        or prereg.get("registered_before_data") is not True
                        or commit != REGISTRATION_COMMIT
                        or prereg.get("source_sha256")
                        != REGISTRATION_SOURCE_SHA256):
                    errors.append(
                        "preregistration artifact does not pin terms, bar, "
                        "and prior commit")
            except (OSError, ValueError) as exc:
                errors.append(f"cannot read preregistration artifact: {exc}")
    return {
        "provider": provider,
        "path": str(path),
        "contract": contract,
        "errors": errors,
        "status": "PASS" if not errors else "REFUSE",
    }


def build_report(
    legoesm_root: Path,
    nemo_root: Path,
    legoesm_budget_root: Path | None = None,
    nemo_budget_root: Path | None = None,
) -> dict:
    """Audit state archives and independently acquired online budgets."""
    legoesm_budget_root = (
        legoesm_root if legoesm_budget_root is None else legoesm_budget_root)
    nemo_budget_root = nemo_root if nemo_budget_root is None else nemo_budget_root
    legoesm = audit_legoesm_archive(legoesm_root)
    nemo = audit_nemo_archive(nemo_root)
    legoesm_budget = audit_budget_contract(legoesm_budget_root, "legoesm")
    nemo_budget = audit_budget_contract(nemo_budget_root, "nemo")
    state_admitted = legoesm["status"] == nemo["status"] == "PASS"
    budget_admitted = (
        state_admitted
        and legoesm_budget["status"] == nemo_budget["status"] == "PASS"
    )
    mean_control = nonlinear_daily_mean_control()
    return {
        "schema": "legoesm-gyre-hires-momentum-budget-preflight-v3",
        "registered_window_days": [START_DAY, END_DAY],
        "leading_field": "native surface V",
        "leading_region": "western boundary wet-span bin",
        "registered_terms": list(REGISTERED_TERMS),
        "legoesm": legoesm,
        "nemo": nemo,
        "state_archive_status": "PASS" if state_admitted else "REFUSE",
        "budget_observability": {
            "legoesm": legoesm_budget,
            "nemo": nemo_budget,
            "nonlinear_daily_mean_control": mean_control,
            "status": "PASS" if budget_admitted else "REFUSE",
        },
        "owner": "UNMEASURED",
        "closure": "UNMEASURED",
        "verdict": "PASS" if budget_admitted else "ACQUISITION_NEEDED",
        "disposition": (
            "online production records admit the registered term attribution"
            if budget_admitted else
            "daily state archives do not admit production-term attribution; "
            "require matched online term records, external-mode coverage, "
            "implicit vertical-mixing/stress separation, an independent total, "
            "passive-trajectory identity, and a preregistered numeric closure bar"
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legoesm-root", type=Path, required=True)
    parser.add_argument("--nemo-root", type=Path, required=True)
    parser.add_argument("--legoesm-budget-root", type=Path)
    parser.add_argument("--nemo-budget-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = build_report(
            args.legoesm_root, args.nemo_root,
            args.legoesm_budget_root, args.nemo_budget_root)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["verdict"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
