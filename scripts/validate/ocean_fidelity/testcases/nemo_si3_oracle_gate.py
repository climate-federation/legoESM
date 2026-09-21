#!/usr/bin/env python3
"""Fail-closed oracle-only gate for SI3 lane-3 dynamics rungs 3.1--3.4.

The gate inventories the files NEMO actually wrote.  Every mesh/restart/
resolved-namelist item must have exactly one VERIFIED, WAIVED, or UNMEASURED
disposition.  Phase 1 makes no legoESM comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import struct
import subprocess
from pathlib import Path

import netCDF4
import numpy as np
from legoesm.ocean.fidelity.provenance import worktree_stamp

_LANE1_PATH = Path(__file__).with_name("nemo_testcase_oracle_gate.py")
_SPEC = importlib.util.spec_from_file_location("nemo_testcase_oracle_gate_l1", _LANE1_PATH)
assert _SPEC and _SPEC.loader
_lane1 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_lane1)

GateError = _lane1.GateError
require = _lane1.require
sha256 = _lane1.sha256
namelist_inventory = _lane1.namelist_inventory
namelist_values = _lane1.namelist_values
nc_inventory = _lane1.nc_inventory
parse_logical = _lane1.parse_logical

VALID = {"VERIFIED", "WAIVED", "UNMEASURED"}
# NEMO's own enumeration of the non-state metadata variables it writes into every
# restart/mesh file (src/OCE/IOM/iom.F90:365-375, meta(1:11)); `numcat` is the SI3
# category axis defined at src/OCE/IOM/iom_nf90.F90:139.  Copied verbatim so the
# waiver set is sourced, not invented.
META = {
    "nav_lat",
    "nav_lon",
    "nav_lev",
    "time_instant",
    "time_instant_bounds",
    "time_counter",
    "time_counter_bounds",
    "x",
    "y",
    "numcat",
    "nav_hgt",
}
FORMAT = "nemo-si3-l3-coverage-v1"
MAGIC = "NEMO_L3_ICE_1"
_SI3_HALO = 2
_SI3_FOUR_POINT_WEIGHT = 0.25
_SI3_ICE_PRESENCE_EPSILON = 1.0e-10
_FRAME_STORAGE_BITS = 64  # lane-3 fp64 dump contract, icestp.F90:154-171
_DIAGNOSTIC_P95 = 95.0  # receipt-requested descriptive percentile, not a gate
_DIAGNOSTIC_P99 = 99.0  # receipt-requested descriptive percentile, not a gate

# Executed ORCA1 overlay values. Sources: ORCA1 namelist_ice_ref:57,108-116
# and namelist_ice_cfg:52-55; the rung-3.4 additions are ref:87-102 and
# cfg:57-65. These are validation targets, not tunable gate thresholds.
_ORCA1_RHEOLOGY_NUMERIC_VALUES = (
    ("namdyn.rn_ishlat", 2),
    ("namdyn_rdgrft.rn_pstar", 2.0e4),
    ("namdyn_rdgrft.rn_crhg", 20),
    ("namdyn_rhg.rn_creepl", 2.0e-9),
    ("namdyn_rhg.rn_ecc", 2),
    ("namdyn_rhg.nn_nevp", 100),
    ("namdyn_rhg.rn_relast", 0.333),
    ("namdyn_rhg.nn_rhg_chkcvg", 0),
)
_ORCA1_RIDGING_NUMERIC_VALUES = (
    ("namdyn_rdgrft.rn_murdg", 3.0),
    ("namdyn_rdgrft.rn_csrdg", 0.5),
    ("namdyn_rdgrft.rn_gstar", 0.15),
    ("namdyn_rdgrft.rn_astar", 0.03),
    ("namdyn_rdgrft.rn_hstar", 25.0),
    ("namdyn_rdgrft.rn_porordg", 0.0),
    ("namdyn_rdgrft.rn_fsnwrdg", 0.5),
    ("namdyn_rdgrft.rn_fpndrdg", 0.5),
    ("namdyn_rdgrft.rn_hraft", 0.75),
    ("namdyn_rdgrft.rn_craft", 5.0),
    ("namdyn_rdgrft.rn_fsnwrft", 0.5),
    ("namdyn_rdgrft.rn_fpndrft", 0.5),
)

# Each field is written in this exact order by the committed icestp.F90:154-170.
# The dump precedes store_fields at upstream icestp.F90:151, so ordinary state
# is current-entry and already-carried diagnostic fields retain the preceding
# completed step's value.  The second citation names the defining/restart site.
FRAME_REGISTRY = (
    ("v_i", ("jpi", "jpj", "jpl"), "STEP_ENTRY_CURRENT", "icestp.F90:154-171; icerst.F90:143"),
    ("v_s", ("jpi", "jpj", "jpl"), "STEP_ENTRY_CURRENT", "icestp.F90:154-171; icerst.F90:144"),
    ("a_i", ("jpi", "jpj", "jpl"), "STEP_ENTRY_CURRENT", "icestp.F90:154-171; icerst.F90:145"),
    ("t_su", ("jpi", "jpj", "jpl"), "STEP_ENTRY_CURRENT", "icestp.F90:154-171; icerst.F90:146"),
    ("oa_i", ("jpi", "jpj", "jpl"), "STEP_ENTRY_CURRENT", "icestp.F90:154-171; icerst.F90:149"),
    ("a_ip", ("jpi", "jpj", "jpl"), "STEP_ENTRY_CURRENT", "icestp.F90:154-171; icerst.F90:150"),
    ("v_ip", ("jpi", "jpj", "jpl"), "STEP_ENTRY_CURRENT", "icestp.F90:154-171; icerst.F90:151"),
    ("v_il", ("jpi", "jpj", "jpl"), "STEP_ENTRY_CURRENT", "icestp.F90:154-171; icerst.F90:152"),
    ("sv_i", ("jpi", "jpj", "jpl"), "STEP_ENTRY_CURRENT", "icestp.F90:154-171; icerst.F90:168"),
    ("u_ice", ("jpi", "jpj"), "STEP_ENTRY_CURRENT", "icestp.F90:154-171; icerst.F90:147"),
    ("v_ice", ("jpi", "jpj"), "STEP_ENTRY_CURRENT", "icestp.F90:154-171; icerst.F90:148"),
    (
        "stress1_i",
        ("jpi", "jpj"),
        "CARRIED_PREVIOUS_STEP",
        "icestp.F90:154-171; icedyn_rhg_evp.F90:1110",
    ),
    (
        "stress2_i",
        ("jpi", "jpj"),
        "CARRIED_PREVIOUS_STEP",
        "icestp.F90:154-171; icedyn_rhg_evp.F90:1111",
    ),
    (
        "stress12_i",
        ("jpi", "jpj"),
        "CARRIED_PREVIOUS_STEP",
        "icestp.F90:154-171; icedyn_rhg_evp.F90:1112",
    ),
    (
        "snwice_mass",
        ("jpi", "jpj"),
        "CARRIED_PREVIOUS_STEP",
        "icestp.F90:154-171; iceupdate.F90:479",
    ),
    (
        "snwice_mass_b",
        ("jpi", "jpj"),
        "CARRIED_PREVIOUS_BEFORE_LEVEL",
        "icestp.F90:154-171; iceupdate.F90:480",
    ),
    (
        "e_s",
        ("jpi", "jpj", "nlay_s", "jpl"),
        "STEP_ENTRY_CURRENT",
        "icestp.F90:154-171; icerst.F90:153-159",
    ),
    (
        "e_i",
        ("jpi", "jpj", "nlay_i", "jpl"),
        "STEP_ENTRY_CURRENT",
        "icestp.F90:154-171; icerst.F90:160-166",
    ),
    (
        "szv_i",
        ("jpi", "jpj", "nlay_i", "jpl"),
        "STEP_ENTRY_CURRENT",
        "icestp.F90:154-171; icerst.F90:170-175",
    ),
)

# `nz` is the RESOLVED nav_lev NEMO writes, not the case's jpkglo: ICE_ADV1D asks
# for kpk=1 (tests/ICE_ADV1D/MY_SRC/usrdef_nam.F90:76) and NEMO raises it via
# `jpk = MAX( 2, jpkglo )` (src/OCE/LBC/mppini.F90:80,375), which the run's own
# ocean.output confirms ("jpk : 2   jpkglo : 1").
RUNGS = {
    "3.1": {
        "steps": 40,
        "nx": 59,
        "ny": 59,
        "nz": 2,
        "dx": 4.0,
        "dt": 2.0,
        "nlay_i": 3,
        "nlay_s": 3,
    },
    "3.2": {
        "steps": 485,
        "nx": 99,
        "ny": 99,
        "nz": 2,
        "dx": 3000.0,
        "dt": 1200.0,
        "nlay_i": 3,
        "nlay_s": 3,
    },
    "3.3": {
        "steps": 485,
        "nx": 99,
        "ny": 99,
        "nz": 2,
        "dx": 3000.0,
        "dt": 1200.0,
        "nlay_i": 3,
        "nlay_s": 3,
    },
    # tests/ICE_RHEO/MY_SRC/usrdef_nam.F90:73-84; namelist_cfg:19-20,29-37;
    # ORCA1 namelist_ice_ref:25-26 supplies the requested layer counts.
    "3.4": {
        "steps": 720,
        "nx": 1000,
        "ny": 1000,
        "nz": 2,
        "dx": 2000.0,
        "dt": 30.0,
        "nlay_i": 10,
        "nlay_s": 5,
    },
}

TARGET_ICE_NML = {
    "nampar.jpl",
    "nampar.nlay_i",
    "nampar.nlay_s",
    "nampar.ln_icedyn",
    "nampar.ln_icethd",
    "namdyn.ln_dynall",
    "namdyn.ln_dynrhgadv",
    "namdyn.ln_dynadv1d",
    "namdyn.ln_dynadv2d",
    "namdyn_adv.ln_adv_pra",
    "namdyn_adv.ln_adv_umx",
    "namdia.ln_icediachk",
}

TARGET_RHG_NML = {
    "namdyn.rn_ishlat",
    "namdyn.ln_landfast_l16",
    "namdyn_rdgrft.ln_str_h79",
    "namdyn_rdgrft.rn_pstar",
    "namdyn_rdgrft.rn_crhg",
    "namdyn_rdgrft.ln_str_smooth",
    "namdyn_rhg.ln_rhg_evp",
    "namdyn_rhg.ln_rhg_eap",
    "namdyn_rhg.ln_aevp",
    "namdyn_rhg.rn_creepl",
    "namdyn_rhg.rn_ecc",
    "namdyn_rhg.nn_nevp",
    "namdyn_rhg.rn_relast",
    "namdyn_rhg.nn_rhg_chkcvg",
}

TARGET_RDGRFT_NML = {
    "namdyn_rdgrft.ln_distf_lin",
    "namdyn_rdgrft.ln_distf_exp",
    "namdyn_rdgrft.rn_murdg",
    "namdyn_rdgrft.rn_csrdg",
    "namdyn_rdgrft.ln_partf_lin",
    "namdyn_rdgrft.rn_gstar",
    "namdyn_rdgrft.ln_partf_exp",
    "namdyn_rdgrft.rn_astar",
    "namdyn_rdgrft.ln_ridging",
    "namdyn_rdgrft.rn_hstar",
    "namdyn_rdgrft.rn_porordg",
    "namdyn_rdgrft.rn_fsnwrdg",
    "namdyn_rdgrft.rn_fpndrdg",
    "namdyn_rdgrft.ln_rafting",
    "namdyn_rdgrft.rn_hraft",
    "namdyn_rdgrft.rn_craft",
    "namdyn_rdgrft.rn_fsnwrft",
    "namdyn_rdgrft.rn_fpndrft",
}

TARGET_RUNG34_STATE_NML = {
    "namthd_sal.nn_icesal",
    "namthd_pnd.ln_pnd",
    "namini.ln_iceini",
}

TARGET_DYN_NML = {
    "namrun.cn_exp",
    "namrun.nn_itend",
    "namrun.nn_stock",
    "namdom.rn_dt",
    "namdom.ln_meshmask",
    "namsbc.nn_fsbc",
    "namsbc.nn_ice",
    "namsbc_sas.l_sasread",
}


REPO_ROOT = Path(__file__).resolve().parents[4]
CONFIG_DIR = Path(__file__).with_name("configs")
CONFIG_STEM = {
    "3.1": "ice_adv1d_l3",
    "3.2": "ice_adv2d_l3",
    "3.3": "ice_adv2d_rhg_l3",
    "3.4": "ice_rheo_l3_clean",
}
# The copied dynamic deck names the clean/excluded source arm, while the ice
# deck is the one ORCA1 overlay shared by that arm.  Keep the deliberate names
# explicit rather than duplicating a byte-identical config file.
CONFIG_STEM_OVERRIDE = {("3.4", "namelist_ice_cfg"): "ice_rheo_l3"}

# Ice-ocean drag is quadratic and both stress terms carry the same U-point ice
# fraction `zaU`, so it cancels and free drift is a closed-form identity:
#   utau_ice = rho0 * rn_Cd_io * |u_ice|^2      (icedyn_rhg_evp.F90:580-590,310)
# `utau_ice` is a case constant, not a namelist entry.
ICE_ADV2D_UTAU_ICE_PA = 1.3  # tests/ICE_ADV2D/MY_SRC/usrdef_sbc.F90:93
# Band on that identity.  The rheology is what is under test, not the third
# digit: a broken stress balance, drag law or aEVP solve moves the steady speed
# by tens of percent, while the measured margin is ~2e-10 (receipt section 4).
FREE_DRIFT_REL_TOL = 1.0e-2


def git_sha() -> str:
    """Repo commit that produced this artifact (nemo_testcase_full_statistics.py:118-121)."""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    ).stdout.strip()


def input_namelists(root: Path, rung: str) -> dict[str, str]:
    """Bind the committed input decks to the run that consumed them.

    The gate otherwise only reads NEMO's RESOLVED `output.namelist.*`, so the
    committed `configs/` copies would be documentation that nothing checks.
    """
    digests: dict[str, str] = {}
    for name in ("namelist_cfg", "namelist_ice_cfg"):
        used = root / name
        stem = CONFIG_STEM_OVERRIDE.get((rung, name), CONFIG_STEM[rung])
        committed = CONFIG_DIR / f"{stem}_{name}"
        require(used.is_file(), f"run input deck absent: {used}")
        require(committed.is_file(), f"committed input deck absent: {committed}")
        digest = sha256(used)
        require(
            digest == sha256(committed),
            f"{name} used by the run differs from committed {committed.name}",
        )
        digests[name] = digest
    return digests


def find_one(root: Path, patterns: tuple[str, ...]) -> Path:
    hits: list[Path] = []
    for pattern in patterns:
        hits.extend(root.glob(pattern))
    hits = sorted(set(hits))
    require(len(hits) == 1, f"expected one of {patterns}, found {len(hits)}")
    return hits[0]


def run_files(root: Path) -> dict[str, Path]:
    files = {
        "mesh": find_one(root, ("mesh_mask*.nc",)),
        "restart": find_one(root, ("*restart_ice*.nc", "*restart_ice_*.nc")),
        "namelist_dyn": root / "output.namelist.dyn",
        "namelist_ice": root / "output.namelist.ice",
        "ocean_output": root / "ocean.output",
    }
    for name, path in files.items():
        require(path.is_file(), f"{name} file is absent: {path}")
    return files


def inventory(root: Path) -> tuple[dict[str, Path], dict[str, set[str]]]:
    files = run_files(root)
    return files, {
        "mesh": nc_inventory(files["mesh"]),
        "restart": nc_inventory(files["restart"]),
        "namelist_dyn": namelist_inventory(files["namelist_dyn"]),
        "namelist_ice": namelist_inventory(files["namelist_ice"]),
    }


def _contract_item(status: str, reason: str) -> dict[str, str]:
    return {"status": status, "reason": reason}


def restart_contract(rung: str, nlay_i: int, nlay_s: int, nn_icesal: int, ponds: bool) -> dict:
    rhg = rung in {"3.3", "3.4"}
    result: dict[str, dict[str, str]] = {}

    def add(name: str, active: bool, reason: str) -> None:
        result[name] = _contract_item("VERIFIED" if active else "WAIVED", reason)

    for name in (
        "nn_fsbc",
        "kt_ice",
        "v_i",
        "v_s",
        "a_i",
        "t_su",
        "u_ice",
        "v_ice",
        "oa_i",
        "a_ip",
        "v_ip",
        "v_il",
        "sv_i",
        "snwice_mass",
        "snwice_mass_b",
    ):
        add(name, True, "Appendix A active restart prognostic; loaded from final SI3 restart")
    for level in range(1, nlay_s + 1):
        add(f"e_s_l{level:02d}", True, "Appendix A snow-enthalpy layer")
        add(
            f"t_s_l{level:02d}",
            False,
            "WAIVED-NOT-RESTARTED: Appendix A t_s(l) is reconstructed; "
            "icerst.F90:153-159 writes e_s(l)",
        )
    for level in range(1, nlay_i + 1):
        add(f"e_i_l{level:02d}", True, "Appendix A ice-enthalpy layer")
        add(
            f"szv_i_l{level:02d}",
            True,
            "Appendix A layer salt-content field; file-loaded even when inert",
        )
    for name in ("cnd_ice", "t1_ice"):
        add(name, False, "WAIVED-INACTIVE: ln_cpl=F; icerst.F90:178-181")
    for name in ("stress1_i", "stress2_i", "stress12_i"):
        add(
            name,
            rhg,
            "Appendix A EVP stress"
            if rhg
            else "WAIVED-INACTIVE: advection-only rung never calls ice_dyn_rhg",
        )

    for stem in ("ice", "sn", "a", "age"):
        for prefix in ("sx", "sy", "sxx", "syy", "sxy"):
            add(f"{prefix}{stem}", True, "Appendix A mandatory Prather moment")
    for stem, layers in (("c0", nlay_s), ("e", nlay_i)):
        for level in range(1, layers + 1):
            for prefix in ("sx", "sy", "sxx", "syy", "sxy"):
                add(f"{prefix}{stem}_l{level:02d}", True, "Appendix A layered Prather moment")
    for prefix in ("sx", "sy", "sxx", "syy", "sxy"):
        add(
            f"{prefix}sal",
            nn_icesal != 4,
            "Appendix A bulk-salinity Prather moment"
            if nn_icesal != 4
            else "WAIVED-INACTIVE: nn_icesal=4 selects layer moments",
        )
    for level in range(1, nlay_i + 1):
        for prefix in ("sx", "sy", "sxx", "syy", "sxy"):
            add(
                f"{prefix}si_l{level:02d}",
                nn_icesal == 4,
                "Appendix A layer-salinity Prather moment"
                if nn_icesal == 4
                else "WAIVED-INACTIVE: nn_icesal/=4 selects bulk moments",
            )
    for stem in ("ap", "vp", "vl"):
        for prefix in ("sx", "sy", "sxx", "syy", "sxy"):
            add(
                f"{prefix}{stem}",
                ponds,
                "Appendix A active pond Prather moment"
                if ponds
                else "WAIVED-INACTIVE: both pond schemes are off",
            )
    if rung == "3.4":
        # icedyn_rdgrft.F90:46-75 declares these SAVE work/diagnostic arrays,
        # but the module has no restart writer.  The redistribution is carried
        # only by the already-VERIFIED a/v/e/salt/pond prognostics above
        # (icedyn_rdgrft.F90:779-888).  Enumerating the ephemeral arrays here
        # prevents "ridging state" from disappearing behind a generic waiver.
        for name in (
            "closing_net",
            "opning",
            "closing_gross",
            "apartf",
            "hrmin",
            "hrmax",
            "hrexp",
            "hraft",
            "hi_hrdg",
            "aridge",
            "araft",
            "airdg1",
            "airft1",
            "airdg2",
            "airft2",
            "opning_2d",
            "dairdg1dt",
            "dairft1dt",
            "dairdg2dt",
            "dairft2dt",
        ):
            add(
                name,
                False,
                "WAIVED-NOT-RESTARTED: icedyn_rdgrft.F90:46-75 ephemeral "
                "work/diagnostic array; carried redistribution is in the active "
                "Appendix-A prognostics (icedyn_rdgrft.F90:779-888)",
            )
    return result


def _resolved_ice_settings(root: Path) -> tuple[int, int, int, bool]:
    values = namelist_values(run_files(root)["namelist_ice"])
    needed = ("nampar.nlay_i", "nampar.nlay_s", "namthd_sal.nn_icesal", "namthd_pnd.ln_pnd")
    for key in needed:
        require(key in values, f"missing resolved selector {key}")
    return (
        int(values["nampar.nlay_i"]),
        int(values["nampar.nlay_s"]),
        int(values["namthd_sal.nn_icesal"]),
        parse_logical(values["namthd_pnd.ln_pnd"], "namthd_pnd.ln_pnd"),
    )


def disposition_template(root: Path, rung: str) -> dict:
    files, actual = inventory(root)
    entries: dict[str, dict[str, dict[str, str]]] = {}
    for namespace, names in actual.items():
        entries[namespace] = {}
        for name in sorted(names):
            key = name.lower()
            if namespace == "mesh":
                waived = key in META
                reason = (
                    "coordinate/time metadata; explicitly waived"
                    if waived
                    else "file-loaded, finite, and covered by geometry checks"
                )
            elif namespace == "restart":
                waived = key in META
                if waived:
                    reason = "coordinate/time metadata; explicitly waived"
                elif name.startswith("DELAY__"):
                    # iom's delayed-global-reduction buffer, not model state: the
                    # ice-CFL mpp_max carries cdelay='cflice'
                    # (icedyn_adv_pra.F90:122, icedyn_adv_umx.F90:185) and iom
                    # writes it at icerst.F90:140 / reads it at iom.F90:1953.
                    reason = (
                        "iom delayed-reduction restart buffer (icerst.F90:140, "
                        "iom.F90:1939-1955), not an Appendix-A prognostic; file-loaded and finite"
                    )
                else:
                    reason = "file-loaded final SI3 restart state in fp64"
            else:
                if namespace == "namelist_dyn":
                    targets = TARGET_DYN_NML
                else:
                    targets = TARGET_ICE_NML
                    if rung in {"3.3", "3.4"}:
                        targets = targets | TARGET_RHG_NML
                    if rung == "3.4":
                        targets = targets | TARGET_RDGRFT_NML | TARGET_RUNG34_STATE_NML
                waived = key not in targets
                reason = (
                    "resolved NEMO default outside the rung claim; full file hash pinned"
                    if waived
                    else (
                        "resolved rung selector verified against the shipped case plus "
                        "declared overlay"
                    )
                )
            entries[namespace][name] = _contract_item("WAIVED" if waived else "VERIFIED", reason)
    nlay_i, nlay_s, nn_icesal, ponds = _resolved_ice_settings(root)
    return {
        "format": FORMAT,
        "worktree": worktree_stamp(),
        "git_sha": git_sha(),
        "rung": rung,
        "files": {name: sha256(path) for name, path in files.items()},
        "entries": entries,
        "restart_contract": restart_contract(rung, nlay_i, nlay_s, nn_icesal, ponds),
    }


def check_manifest(root: Path, rung: str, manifest: dict, plant_unaccounted: bool = False) -> dict:
    files, actual = inventory(root)
    if plant_unaccounted:
        actual["mesh"].add("PLANTED_UNACCOUNTED_FILE_ARRAY")
    require(manifest.get("format") == FORMAT, "bad manifest format")
    require(manifest.get("rung") == rung, "manifest rung mismatch")
    # Provenance is a claim like any other: a manifest may not carry a made-up
    # commit string.
    stamped = manifest.get("git_sha", "")
    require(
        isinstance(stamped, str) and re.fullmatch(r"[0-9a-f]{40}", stamped) is not None,
        f"manifest git_sha is not a 40-hex commit: {stamped!r}",
    )
    # Every disposition is REGENERATED from the run rather than trusted from the
    # file, so a VERIFIED row cannot be silently downgraded to WAIVED (which
    # would skip its numeric/finite/fp64 check) by editing the manifest.  Same
    # ratchet the Appendix-A contract already gets below.
    expected_entries = disposition_template(root, rung)["entries"]
    for namespace, names in actual.items():
        ledger = manifest.get("entries", {}).get(namespace, {})
        require(
            set(ledger) == names,
            f"{namespace} coverage mismatch: missing={sorted(names - set(ledger))}, "
            f"extra={sorted(set(ledger) - names)}",
        )
        require(
            ledger == expected_entries[namespace],
            f"{namespace} dispositions differ from the regenerated coverage template",
        )
        for name, item in ledger.items():
            require(item.get("status") in VALID, f"{namespace}.{name}: bad disposition")
            if namespace in {"mesh", "restart"}:
                require(
                    item.get("status") in {"VERIFIED", "WAIVED"},
                    f"{namespace}.{name}: UNMEASURED is forbidden",
                )
            require(bool(item.get("reason", "").strip()), f"{namespace}.{name}: empty reason")
            if item["status"] != "VERIFIED" or namespace.startswith("namelist"):
                continue
            with netCDF4.Dataset(files[namespace]) as ds:
                data = np.asarray(ds.variables[name][:])
            require(np.issubdtype(data.dtype, np.number), f"{namespace}.{name} is not numeric")
            require(np.all(np.isfinite(data)), f"{namespace}.{name} contains non-finite values")
            if namespace == "restart" and np.issubdtype(data.dtype, np.floating):
                require(data.dtype == np.float64, f"restart.{name} is not fp64")

    for name, path in files.items():
        require(
            manifest.get("files", {}).get(name) == sha256(path),
            f"{name} SHA256 mismatch",
        )

    restart_names = actual["restart"]
    restart_ledger = manifest["entries"]["restart"]
    nlay_i, nlay_s, nn_icesal, ponds = _resolved_ice_settings(root)
    expected_contract = restart_contract(rung, nlay_i, nlay_s, nn_icesal, ponds)
    contract = manifest.get("restart_contract", {})
    require(
        contract == expected_contract,
        "restart contract differs from exhaustive Appendix-A contract",
    )
    for name, item in contract.items():
        require(item.get("status") in VALID, f"restart contract {name}: bad disposition")
        require(bool(item.get("reason", "").strip()), f"restart contract {name}: empty reason")
        if item["status"] == "VERIFIED":
            require(name in restart_names, f"restart contract {name}: active field absent")
            require(
                restart_ledger[name]["status"] == "VERIFIED",
                f"restart contract {name}: not VERIFIED-loaded",
            )
    return {name: len(values) for name, values in actual.items()}


def _number(values: dict[str, str], key: str, expected: float) -> None:
    require(key in values, f"missing resolved selector {key}")
    got = float(values[key].replace("D", "E").replace("d", "e"))
    require(got == expected, f"resolved selector {key}={values[key]}")


def _logical(values: dict[str, str], key: str, expected: bool) -> None:
    require(key in values, f"missing resolved selector {key}")
    require(parse_logical(values[key], key) is expected, f"resolved selector {key}={values[key]}")


def resolved_selectors(root: Path, rung: str) -> dict:
    spec = RUNGS[rung]
    dyn = namelist_values(run_files(root)["namelist_dyn"])
    ice = namelist_values(run_files(root)["namelist_ice"])
    _number(dyn, "namrun.nn_itend", spec["steps"])
    _number(dyn, "namrun.nn_stock", spec["steps"])
    _number(dyn, "namdom.rn_dt", spec["dt"])
    _number(dyn, "namsbc.nn_fsbc", 1)
    _number(dyn, "namsbc.nn_ice", 2)
    _logical(dyn, "namdom.ln_meshmask", True)
    _logical(dyn, "namsbc_sas.l_sasread", False)
    _number(ice, "nampar.nlay_i", spec["nlay_i"])
    _number(ice, "nampar.nlay_s", spec["nlay_s"])
    _number(ice, "nampar.jpl", 1)
    _logical(ice, "nampar.ln_icedyn", True)
    _logical(ice, "nampar.ln_icethd", False)
    _logical(ice, "namdyn_adv.ln_adv_pra", True)
    _logical(ice, "namdyn_adv.ln_adv_umx", False)
    _logical(ice, "namdia.ln_icediachk", rung != "3.4")
    for key in (
        "namdyn.ln_dynall",
        "namdyn.ln_dynrhgadv",
        "namdyn.ln_dynadv1d",
        "namdyn.ln_dynadv2d",
    ):
        expected = {
            "3.1": "namdyn.ln_dynadv1d",
            "3.2": "namdyn.ln_dynadv2d",
            "3.3": "namdyn.ln_dynrhgadv",
            "3.4": "namdyn.ln_dynall",
        }[rung]
        _logical(ice, key, key == expected)
    if rung in {"3.3", "3.4"}:
        for key, expected in _ORCA1_RHEOLOGY_NUMERIC_VALUES:
            _number(ice, key, expected)
        for key, expected in (
            ("namdyn.ln_landfast_l16", False),
            ("namdyn_rdgrft.ln_str_h79", True),
            ("namdyn_rdgrft.ln_str_smooth", False),
            ("namdyn_rhg.ln_rhg_evp", True),
            ("namdyn_rhg.ln_rhg_eap", False),
            ("namdyn_rhg.ln_aevp", True),
        ):
            _logical(ice, key, expected)
    if rung == "3.4":
        for key, expected in _ORCA1_RIDGING_NUMERIC_VALUES:
            _number(ice, key, expected)
        for key, expected in (
            ("namdyn_rdgrft.ln_distf_lin", False),
            ("namdyn_rdgrft.ln_distf_exp", True),
            ("namdyn_rdgrft.ln_partf_lin", False),
            ("namdyn_rdgrft.ln_partf_exp", True),
            ("namdyn_rdgrft.ln_ridging", True),
            ("namdyn_rdgrft.ln_rafting", True),
            ("namini.ln_iceini", True),
        ):
            _logical(ice, key, expected)
    expected_exp = {
        "3.1": "ICE_ADV1D_OMIP_L3",
        "3.2": "ICE_ADV2D_OMIP_L3",
        "3.3": "ICE_ADV2D_RHG_OMIP_L3",
        "3.4": "ICE_RHEO_OMIP_L3",
    }[rung]
    # NEMO writes cn_exp back as a quoted, blank-padded CHARACTER(lc) field.
    resolved_exp = dyn.get("namrun.cn_exp", "").strip().strip("'\"").strip()
    require(resolved_exp == expected_exp, f"resolved cn_exp={resolved_exp!r}")
    nlay_i, nlay_s, nn_icesal, ponds = _resolved_ice_settings(root)
    return {
        "jpl": 1,
        "nlay_i": nlay_i,
        "nlay_s": nlay_s,
        "nn_icesal": nn_icesal,
        "ln_pnd": ponds,
    }


def cpp_keys(case_dir: Path) -> dict:
    path = case_dir / "BLD/cpp.fcm"
    require(path.is_file(), f"compiled cpp key file absent: {path}")
    line = next((raw for raw in path.read_text().splitlines() if "fppkeys" in raw), "")
    keys = line.split("fppkeys", 1)[-1].split()
    require(set(keys) == {"key_si3", "key_linssh", "key_vco_1d"}, f"resolved cpp keys={keys}")
    return {"keys": keys, "sha256": sha256(path)}


def _array(ds: netCDF4.Dataset, name: str) -> np.ndarray:
    require(name in ds.variables, f"missing mesh array {name}")
    return np.asarray(ds.variables[name][:])


def geometry(root: Path, rung: str, plant_field: bool = False) -> dict:
    spec = RUNGS[rung]
    mesh = run_files(root)["mesh"]
    with netCDF4.Dataset(mesh) as ds:
        require(len(ds.dimensions["x"]) == spec["nx"], "mesh x dimension")
        require(len(ds.dimensions["y"]) == spec["ny"], "mesh y dimension")
        require(len(ds.dimensions["nav_lev"]) == spec["nz"], "mesh vertical dimension")
        for name in ("e1t", "e1u", "e1v", "e1f", "e2t", "e2u", "e2v", "e2f"):
            data = _array(ds, name)
            if plant_field and name == "e1t":
                data = data.copy()
                data.flat[0] += 1.0
            require(data.dtype == np.float64, f"{name} is not fp64")
            require(np.array_equal(data, np.full_like(data, spec["dx"])), f"{name} metric")
        for name in ("ff_t", "ff_f"):
            require(
                np.array_equal(_array(ds, name), np.zeros_like(_array(ds, name))),
                f"{name} must be zero",
            )
        for name in ("tmask", "umask", "vmask"):
            require(set(np.unique(_array(ds, name))).issubset({0, 1}), f"{name} is not binary")
    return {
        "mesh_sha256": sha256(mesh),
        "nx": spec["nx"],
        "ny": spec["ny"],
        "nz": spec["nz"],
        "dx_m": spec["dx"],
    }


def read_frame(
    path: Path, fields: frozenset[str] | None = None
) -> tuple[dict[str, int], dict[str, np.ndarray]]:
    if fields is not None:
        known = {row[0] for row in FRAME_REGISTRY}
        require(not (fields - known), f"unknown requested frame fields: {sorted(fields - known)}")
    with path.open("rb") as fh:
        magic = fh.read(16).decode("ascii").rstrip()
        raw = fh.read(9 * 4)
        require(len(raw) == 9 * 4, f"truncated frame header: {path}")
        version, kt, jpi, jpj, jpl, nlay_i, nlay_s, storage, nfields = struct.unpack("=9i", raw)
        require(magic == MAGIC, f"bad frame magic: {magic!r}")
        require(
            version == 1 and storage == _FRAME_STORAGE_BITS,
            f"bad frame version/storage: {version}/{storage}",
        )
        require(nfields == len(FRAME_REGISTRY), f"frame registry count {nfields}")
        dims = {"jpi": jpi, "jpj": jpj, "jpl": jpl, "nlay_i": nlay_i, "nlay_s": nlay_s}
        arrays: dict[str, np.ndarray] = {}
        for name, axes, _, _ in FRAME_REGISTRY:
            shape = tuple(dims[axis] for axis in axes)
            count = int(np.prod(shape))
            if fields is not None and name not in fields:
                fh.seek(count * np.dtype(np.float64).itemsize, 1)
                continue
            values = np.fromfile(fh, dtype=np.float64, count=count)
            require(values.size == count, f"truncated frame field {name}: {path}")
            arrays[name] = values.reshape(shape, order="F")
        require(fh.read(1) == b"", f"unregistered trailing frame bytes: {path}")
    require(
        all(np.all(np.isfinite(value)) for value in arrays.values()), f"non-finite frame: {path}"
    )
    return {"kt": kt, **dims, "storage_bits": storage}, arrays


def _interior(array: np.ndarray) -> np.ndarray:
    require(
        array.shape[0] >= 5 and array.shape[1] >= 5, f"frame lacks two-cell halos: {array.shape}"
    )
    return array[2:-2, 2:-2, ...]


def _relative_drift(initial: np.ndarray, final: np.ndarray) -> float:
    scale = max(abs(float(np.sum(initial, dtype=np.float64))), 1.0)
    return abs(float(np.sum(final, dtype=np.float64) - np.sum(initial, dtype=np.float64))) / scale


def _restart_array(path: Path, name: str) -> np.ndarray:
    with netCDF4.Dataset(path) as ds:
        require(name in ds.variables, f"final restart lacks phenomenology field {name}")
        variable = ds.variables[name]
        selection: list[int | slice] = []
        kept_dims: list[str] = []
        for dim in variable.dimensions:
            if "time" in dim.lower():
                selection.append(0)
            else:
                selection.append(slice(None))
                kept_dims.append(dim)
        data = np.asarray(variable[tuple(selection)])
    x_axis = next((i for i, dim in enumerate(kept_dims) if dim.lower() == "x"), None)
    y_axis = next((i for i, dim in enumerate(kept_dims) if dim.lower() == "y"), None)
    require(x_axis is not None and y_axis is not None, f"{name}: cannot identify x/y dimensions")
    order = [x_axis, y_axis] + [i for i in range(data.ndim) if i not in {x_axis, y_axis}]
    return np.transpose(data, order)


def final_restart_names(rung: str) -> tuple[str, ...]:
    """Fields consumed by the endpoint phenomenology for each rung."""

    names = ("a_i", "v_i", "u_ice")
    return names + (("v_ice",) if rung == "3.4" else ())


def wet_window(root: Path) -> tuple[slice, slice]:
    """Surface-tmask bounding box, in the frames' (x, y) order.

    ICE_ADV1D closes its basin with a one-cell land rim (`mesh_mask.tmask` rows
    and columns 0 and 58 are 0), and `dommsk.F90` zeroes every ice field there,
    so a whole-array reduction reports the land rim rather than the documented
    ice-domain behaviour.  ICE_ADV2D is doubly periodic and fully wet, for which
    this window is the whole array.
    """
    with netCDF4.Dataset(run_files(root)["mesh"]) as ds:
        tmask = np.asarray(_array(ds, "tmask"))
    while tmask.ndim > 2:
        tmask = tmask[0]
    wet = tmask.T.astype(bool)  # (y, x) file order -> (x, y) frame order
    require(bool(wet.any()), "mesh tmask has no wet cell")
    xs = np.flatnonzero(wet.any(axis=1))
    ys = np.flatnonzero(wet.any(axis=0))
    window = (slice(int(xs[0]), int(xs[-1]) + 1), slice(int(ys[0]), int(ys[-1]) + 1))
    require(
        bool(wet[window].all()),
        "surface wet domain is not a filled rectangle; the phenomenology window "
        "would mix land and ocean cells",
    )
    return window


def free_drift_speed(root: Path) -> dict[str, float]:
    """Steady free-drift speed implied by the run's OWN resolved constants.

    Nothing here is defaulted: `rn_Cd_io` is read from the resolved ice
    namelist and `rho0` from the run's `ocean.output`.
    """
    values = namelist_values(run_files(root)["namelist_ice"])
    require("namsbc.rn_cd_io" in values, "missing resolved selector namsbc.rn_cd_io")
    cd_io = float(values["namsbc.rn_cd_io"].replace("D", "E").replace("d", "e"))
    text = run_files(root)["ocean_output"].read_text(errors="replace")
    match = re.search(r"volumic mass of reference\s+rho0\s*=\s*([-+.\deEdD]+)", text)
    require(match is not None, "ocean.output does not print the resolved rho0")
    assert match is not None
    rho0 = float(match.group(1).replace("D", "E").replace("d", "e"))
    require(rho0 > 0.0 and cd_io > 0.0, f"non-positive rho0={rho0} or rn_Cd_io={cd_io}")
    return {
        "rho0_kg_m3": rho0,
        "rn_cd_io": cd_io,
        "utau_ice_pa": ICE_ADV2D_UTAU_ICE_PA,
        "u_free_drift_m_s": float(np.sqrt(ICE_ADV2D_UTAU_ICE_PA / (rho0 * cd_io))),
        "relative_tolerance": FREE_DRIFT_REL_TOL,
    }


def cell_area(root: Path) -> np.ndarray:
    """T-cell area in the frames' (x, y) order, from the mesh NEMO wrote."""
    with netCDF4.Dataset(run_files(root)["mesh"]) as ds:
        e1t, e2t = _array(ds, "e1t"), _array(ds, "e2t")
    while e1t.ndim > 2:
        e1t, e2t = e1t[0], e2t[0]
    return np.asarray(e1t * e2t).T


def conservation_diagnostics(root: Path) -> dict:
    """Preregistered predicate: any printed SI3 `: violation` line is REFUTE.

    The verdict is returned rather than raised so the receipt still carries the
    coverage and trajectory numbers; `main` exits nonzero on anything but
    CONFIRM, so the gate stays fail-closed.
    """
    files = run_files(root)
    contents = files["ocean_output"].read_text(errors="replace")
    hits = re.findall(r"^.*:\s*violation\s+.*$", contents, flags=re.IGNORECASE | re.MULTILINE)
    # The instrument stamps every ice step entry, so each printed violation can be
    # attributed to the step it was raised in rather than only counted.
    heat: list[float] = []
    events: list[dict[str, float]] = []
    step = float("nan")
    for line in contents.splitlines():
        stamp = re.search(r"LANE3_ICE_STEP_ENTRY_DUMP\s+(\d+)", line)
        if stamp:
            step = float(stamp.group(1))
            continue
        found = re.search(r":\s*violation heat cons\. \[J\]\s*=\s*([-+.\deEdD]+)", line)
        if found:
            value = float(found.group(1).replace("D", "E").replace("d", "e"))
            heat.append(value)
            events.append({"kt": step, "violation_j": value})
    ice_values = namelist_values(files["namelist_ice"])
    require("namdia.ln_icediachk" in ice_values, "missing resolved ln_icediachk")
    enabled = parse_logical(ice_values["namdia.ln_icediachk"], "namdia.ln_icediachk")
    require(enabled or not hits, "SI3 printed conservation violations while its check was disabled")
    return {
        "status": ("CONFIRM" if not hits else "REFUTE") if enabled else "WAIVED-INACTIVE",
        "criterion": (
            "zero SI3 native-threshold conservation violations"
            if enabled
            else "WAIVED-INACTIVE: resolved ln_icediachk=F; absence of lines is not a check"
        ),
        "enabled": enabled,
        "source": "icectl.F90:67-78,166-190,232-236; namelist_ice_ref:318-319",
        "violation_count": len(hits),
        "violations": [hit.strip() for hit in hits],
        "violation_heat_j": heat,
        "violation_heat_j_total": float(sum(heat)),
        "violation_heat_events": events,
    }


def rung34_shear_diagnostic(root: Path, window: tuple[slice, slice]) -> dict[str, object]:
    """Evaluate NEMO's maximum-shear diagnostic without inventing a README band."""

    paths = sorted(root.glob("*_6h_*.nc"))
    require(paths, "rung 3.4 has no final six-hour output file")
    path: Path | None = None
    raw: np.ma.MaskedArray | None = None
    for candidate in paths:
        with netCDF4.Dataset(candidate) as dataset:
            if "sishea" not in dataset.variables:
                continue
            variable = dataset.variables["sishea"]
            require(
                variable.shape[0] > 0,
                f"documented shear field has no time record: {candidate}",
            )
            raw = np.ma.asarray(variable[-1])
            path = candidate
            break
    if path is None or raw is None:
        mesh_names = ("e1u", "e2u", "e1v", "e2v", "e1f", "e2f", "e1t", "e2t", "fmask")
        with netCDF4.Dataset(run_files(root)["mesh"]) as dataset:
            metrics: dict[str, np.ndarray] = {}
            for name in mesh_names:
                value = np.asarray(_array(dataset, name))
                while value.ndim > 2:
                    value = value[0]
                metrics[name] = np.pad(value.T.astype(np.float64), _SI3_HALO, mode="wrap")
        frame_paths = sorted(root.glob("oracle_ice_step_entry_kt*.bin"))
        require(len(frame_paths) == RUNGS["3.4"]["steps"], "rung 3.4 shear frame count")
        selected = frozenset(("a_i", "u_ice", "v_ice"))
        series: list[dict[str, object]] = []
        final_shear: np.ndarray | None = None
        for expected_kt, frame_path in enumerate(frame_paths, start=1):
            header, frame = read_frame(frame_path, selected)
            require(header["kt"] == expected_kt, "rung 3.4 shear frame clock")
            u_ice = frame["u_ice"]
            v_ice = frame["v_ice"]
            ice_area = frame["a_i"][..., 0]
            shear = _rung34_sishea(u_ice, v_ice, ice_area, metrics)
            physical = shear[
                _SI3_HALO : -_SI3_HALO, _SI3_HALO : -_SI3_HALO
            ][window]
            require(np.all(np.isfinite(physical)), "derived NEMO sishea is non-finite")
            series.append(
                {
                    "kt": expected_kt,
                    "maximum_s-1": float(np.max(physical)),
                    "p99_s-1": float(np.percentile(physical, _DIAGNOSTIC_P99)),
                }
            )
            final_shear = physical
        assert final_shear is not None
        p95, p99 = np.percentile(
            final_shear, (_DIAGNOSTIC_P95, _DIAGNOSTIC_P99)
        )
        return {
            "source": (
                "tests/ICE_RHEO/EXPREF/README:51-53; "
                "icedyn_rhg_evp.F90:190-191,793-816"
            ),
            "field": "sishea reconstructed from NEMO entry-frame u_ice/v_ice and mesh",
            "formula": "sqrt(zdt**2 + zds**2) * zmsk",
            "fast_and_icb_masks": (
                "both zero on this landfast-off, iceberg-free copied case; their source "
                "multipliers at icedyn_rhg_evp.F90:805,813 are therefore one"
            ),
            "dtype": str(final_shear.dtype),
            "frames": len(series),
            "series": series,
            "minimum_s-1": float(np.min(final_shear)),
            "maximum_s-1": float(np.max(final_shear)),
            "p95_s-1": float(p95),
            "p99_s-1": float(p99),
            "cells_above_p99": int(np.count_nonzero(final_shear > p99)),
            "available_six_hour_outputs": [
                {"path": str(candidate), "sha256": sha256(candidate)}
                for candidate in paths
            ],
            "classification": (
                "MEASURED-UNCLASSIFIED: all 720 NEMO frames evaluate the source formula, "
                "but README supplies neither a numeric sharpness threshold nor an EAP "
                "comparator for this aEVP run"
            ),
        }
    shear = np.asarray(np.ma.filled(raw, np.nan), dtype=np.float64).T[window]
    require(np.all(np.isfinite(shear)), "final NEMO sishea contains non-finite wet cells")
    p95, p99 = np.percentile(shear, (_DIAGNOSTIC_P95, _DIAGNOSTIC_P99))
    return {
        "source": "tests/ICE_RHEO/EXPREF/README:51-53; icedyn.F90:172-193",
        "field": "sishea (NEMO maximum shear of sea-ice velocity, final 6-hour record)",
        "output_path": str(path),
        "output_sha256": sha256(path),
        "dtype_on_disk": str(raw.dtype),
        "minimum_s-1": float(np.min(shear)),
        "maximum_s-1": float(np.max(shear)),
        "p95_s-1": float(p95),
        "p99_s-1": float(p99),
        "cells_above_p99": int(np.count_nonzero(shear > p99)),
        "classification": (
            "MEASURED-UNCLASSIFIED: README says EVP maxima are less defined but supplies "
            "neither a numeric sharpness threshold nor an EAP comparator for this aEVP run"
        ),
    }


def _rung34_sishea(
    u_ice: np.ndarray,
    v_ice: np.ndarray,
    ice_area: np.ndarray,
    metrics: dict[str, np.ndarray],
) -> np.ndarray:
    """Literal NumPy replay of icedyn_rhg_evp.F90:793-816."""

    zds = (
        (
            np.roll(u_ice, -1, axis=1) / np.roll(metrics["e1u"], -1, axis=1)
            - u_ice / metrics["e1u"]
        )
        * metrics["e1f"]
        * metrics["e1f"]
        + (
            np.roll(v_ice, -1, axis=0) / np.roll(metrics["e2v"], -1, axis=0)
            - v_ice / metrics["e2v"]
        )
        * metrics["e2f"]
        * metrics["e2f"]
    ) / (metrics["e1f"] * metrics["e2f"])
    zds = zds * metrics["fmask"]
    tension = (
        (
            u_ice / metrics["e2u"]
            - np.roll(u_ice, 1, axis=0) / np.roll(metrics["e2u"], 1, axis=0)
        )
        * metrics["e2t"]
        * metrics["e2t"]
        - (
            v_ice / metrics["e1v"]
            - np.roll(v_ice, 1, axis=1) / np.roll(metrics["e1v"], 1, axis=1)
        )
        * metrics["e1t"]
        * metrics["e1t"]
    ) / (metrics["e1t"] * metrics["e2t"])
    weighted_shear_square = zds * zds * (metrics["e1f"] * metrics["e2f"])
    shear_square_t = (
        weighted_shear_square
        + np.roll(weighted_shear_square, 1, axis=0)
        + np.roll(weighted_shear_square, 1, axis=1)
        + np.roll(np.roll(weighted_shear_square, 1, axis=0), 1, axis=1)
    ) * _SI3_FOUR_POINT_WEIGHT / (metrics["e1t"] * metrics["e2t"])
    zmsk = (ice_area >= _SI3_ICE_PRESENCE_EPSILON).astype(np.float64)
    return np.sqrt(tension * tension + shear_square_t) * zmsk


def phenomenology(
    rung: str,
    first: dict[str, np.ndarray],
    final: dict[str, np.ndarray],
    window: tuple[slice, slice] = (slice(None), slice(None)),
    free_drift: dict[str, float] | None = None,
    maximum_trajectory: dict[str, float | str | bool] | None = None,
    rung34_shear: dict[str, object] | None = None,
) -> dict:
    a0, a1 = _interior(first["a_i"])[window], final["a_i"][window]
    v0, v1 = _interior(first["v_i"])[window], final["v_i"][window]
    refuted: list[str] = []

    def predicate(holds: bool, claim: str) -> None:
        """Record a preregistered CONFIRM/REFUTE predicate without aborting.

        The verdict is reported instead of raised so a refuted rung still emits
        its measured numbers; `main` exits nonzero on any REFUTE, so the gate
        stays fail-closed.
        """
        if not holds:
            refuted.append(claim)

    report: dict[str, object] = {
        "status": "CONFIRM",
        "sample": "step-entry kt=1 versus completed-run final ice restart",
        "domain": "surface-tmask wet window (mesh_mask.tmask bounding box)",
        "concentration_relative_drift_unclassified": _relative_drift(a0, a1),
        "ice_volume_relative_drift_unclassified": _relative_drift(v0, v1),
    }
    if rung == "3.1":
        h0 = np.divide(v0, a0, out=np.zeros_like(v0), where=a0 != 0.0)
        h1 = np.divide(v1, a1, out=np.zeros_like(v1), where=a1 != 0.0)
        spreads = {
            name: float(np.max(np.ptp(array, axis=1)))
            for name, array in (("a_i", a1), ("v_i", v1), ("h_i", h1))
        }
        predicate(
            all(value == 0.0 for value in spreads.values()), f"y-homogeneity refuted: {spreads}"
        )
        x = np.arange(a0.shape[0], dtype=np.float64)
        center = 0.5 * (a0.shape[0] - 1)
        centroids: dict[str, tuple[float, float]] = {}
        for name, initial, completed in (("a_i", a0, a1), ("v_i", v0, v1), ("h_i", h0, h1)):
            weights0 = np.sum(initial, axis=(1, 2), dtype=np.float64)
            weights1 = np.sum(completed, axis=(1, 2), dtype=np.float64)
            c0 = float(np.sum(weights0 * x) / np.sum(weights0))
            c1 = float(np.sum(weights1 * x) / np.sum(weights1))
            # h_i is excluded from the strict-decrease predicate because its
            # INITIAL distance from the basin centre is exactly 0: the shipped IC
            # lays a symmetric thickness notch (hti=0.2 on x in [15,43] of the
            # 59-cell basin, make_initice.py:97-103) on a uniform hti=1 field, so
            # its weighted centroid starts on the centre and nothing can strictly
            # decrease from zero.  The predicate is ill-posed, not refuted; the
            # measured baseline is emitted so the claim stays checkable.
            if name != "h_i":
                predicate(
                    abs(c1 - center) < abs(c0 - center),
                    f"{name} did not converge toward centre",
                )
            else:
                require(
                    abs(c0 - center) == 0.0,
                    "h_i convergence is only voided while its baseline distance is "
                    f"exactly zero; measured {abs(c0 - center)!r}",
                )
            predicate(float(np.ptp(completed)) > 0.0, f"{name} initial shape collapsed")
            centroids[name] = (c0, c1)
        report.update(
            basin_centre_index=center,
            h_i_centroid_convergence="VOID-ILL-POSED: baseline distance is exactly 0",
            y_homogeneity_max_abs=max(spreads.values()),
            initial_a_centroid_index=centroids["a_i"][0],
            final_a_centroid_index=centroids["a_i"][1],
            initial_v_centroid_index=centroids["v_i"][0],
            final_v_centroid_index=centroids["v_i"][1],
            initial_h_centroid_index=centroids["h_i"][0],
            final_h_centroid_index=centroids["h_i"][1],
        )
    elif rung == "3.2":
        h0 = np.divide(v0, a0, out=np.zeros_like(v0), where=a0 != 0.0)
        h1 = np.divide(v1, a1, out=np.zeros_like(v1), where=a1 != 0.0)
        overshoot = float(np.max(h1) - np.max(h0))
        require(maximum_trajectory is not None, "rung 3.2 needs its step-boundary max series")
        assert maximum_trajectory is not None
        predicate(overshoot > 0.0, "documented Prather ice-thickness overshoot not detected")
        report.update(
            initial_a_i_max=float(np.max(a0)),
            final_a_i_max=float(np.max(a1)),
            maximum_concentration_documentation_status=(
                "UNMEASURED: the shipped qualitative Prather-versus-UM remark supplies "
                "no quantitative max-preservation band"
            ),
            maximum_concentration_step_boundary_diagnostic=maximum_trajectory,
            initial_h_i_max=float(np.max(h0)),
            final_h_i_max=float(np.max(h1)),
            h_i_overshoot=overshoot,
            upper_side_lobe_cell_count=int(np.count_nonzero(h1 > np.max(h0))),
        )
    elif rung == "3.3":
        response = float(np.max(final["u_ice"][window]))
        state_change = float(np.max(np.abs(v1 - v0)))
        predicate(response > 0.0, "constant positive x-stress produced no positive ice velocity")
        predicate(state_change > 0.0, "rheology+advection state did not change")
        report.update(final_u_ice_max=response, ice_volume_max_abs_change=state_change)
        # The documented claim is that the rheology CALCULATES the velocity the
        # constant stress implies, so the bar is the free-drift identity the run's
        # own constants fix -- not merely "the ice moved".
        require(free_drift is not None, "rung 3.3 needs the resolved free-drift constants")
        assert free_drift is not None
        predicted = free_drift["u_free_drift_m_s"]
        miss = abs(response - predicted) / predicted
        predicate(
            miss <= free_drift["relative_tolerance"],
            f"steady ice speed {response} misses the free-drift identity {predicted} "
            f"by {miss} (band {free_drift['relative_tolerance']})",
        )
        report.update(
            u_free_drift_predicted_m_s=predicted,
            u_free_drift_relative_miss=miss,
            u_free_drift_source="icedyn_rhg_evp.F90:310,580-590; usrdef_sbc.F90:93",
        )
    else:
        require(rung == "3.4", f"unhandled phenomenology rung {rung}")
        require(rung34_shear is not None, "rung 3.4 needs NEMO's final sishea field")
        velocity_response = max(
            float(np.max(np.abs(final["u_ice"][window]))),
            float(np.max(np.abs(final["v_ice"][window]))),
        )
        state_change = float(np.max(np.abs(v1 - v0)))
        require(velocity_response > 0.0, "ICE_RHEO wind produced no velocity response")
        require(state_change > 0.0, "ICE_RHEO dynamics did not change ice volume")
        report.update(
            status=(
                "UNMEASURED"
                if str(rung34_shear["classification"]).startswith("UNMEASURED")
                else "MEASURED-UNCLASSIFIED"
            ),
            readme_source="tests/ICE_RHEO/EXPREF/README:51-53",
            final_max_abs_ice_velocity_m_s=velocity_response,
            ice_volume_max_abs_change=state_change,
            shear=rung34_shear,
            eap_angle_contrast=(
                "OUT-OF-SCOPE: default-duration requested arm is aEVP; README reserves "
                "the EAP intersection-angle contrast for a longer paired run"
            ),
        )
    report["refuted_predicates"] = refuted
    if refuted:
        report["status"] = "REFUTE"
    elif rung == "3.2":
        report["status"] = "UNMEASURED"
    return report


def maximum_trajectory_diagnostic(
    samples: list[dict[str, float | str]],
) -> dict[str, float | str | bool]:
    """Describe, but do not classify, max-concentration changes at step boundaries.

    The shipped README supplies no numerical tolerance, and SI3 itself notes that
    Prather fields are not perfectly bounded (``icedyn_adv_pra.F90:418-420``).
    Consequently this diagnostic cannot turn the qualitative documentation into
    an exact floating-point predicate.
    """
    require(len(samples) >= 2, "maximum trajectory needs at least two samples")
    baseline = float(samples[0]["a_i_max"])
    require(
        baseline != 0.0,
        "relative maximum trajectory is undefined for a zero baseline",
    )
    deviations = [float(row["a_i_max"]) - baseline for row in samples]
    worst_index = max(range(len(samples)), key=lambda index: abs(deviations[index]))
    worst = samples[worst_index]
    return {
        "sample": "all SI3 step-entry boundaries plus the completed-run final restart",
        "coverage": "initial, post-steps 1..N-1, and post-step N; no within-step split states",
        "baseline_a_i_max": baseline,
        "all_sampled_maxima_exactly_equal": all(value == 0.0 for value in deviations),
        "worst_signed_excursion": deviations[worst_index],
        "worst_abs_relative_excursion": abs(deviations[worst_index]) / abs(baseline),
        "worst_sample_kt": float(worst["kt"]),
        "worst_sample_phase": str(worst["phase"]),
        "minimum_sampled_a_i_max": min(float(row["a_i_max"]) for row in samples),
        "maximum_sampled_a_i_max": max(float(row["a_i_max"]) for row in samples),
    }


def trajectory(root: Path, rung: str) -> dict:
    expected = RUNGS[rung]["steps"]
    paths = sorted(root.glob("oracle_ice_step_entry_kt*.bin"))
    require(len(paths) == expected, f"expected {expected} ice-entry frames, found {len(paths)}")
    first_arrays = final_arrays = None
    frame_hashes: list[str] = []
    header0: dict[str, int] | None = None
    window = wet_window(root)
    area = cell_area(root)[window]
    scan: list[dict[str, float]] = []
    for step, path in enumerate(paths, start=1):
        header, arrays = read_frame(path)
        require(header["kt"] == step, f"frame sequence mismatch at {path.name}: kt={header['kt']}")
        require(header["jpi"] == RUNGS[rung]["nx"] + 4, f"local jpi={header['jpi']}")
        require(header["jpj"] == RUNGS[rung]["ny"] + 4, f"local jpj={header['jpj']}")
        require(
            header["nlay_i"] == RUNGS[rung]["nlay_i"]
            and header["nlay_s"] == RUNGS[rung]["nlay_s"],
            "frame ice-layer count",
        )
        if header0 is None:
            header0 = header
            first_arrays = arrays
        else:
            require(
                {k: header[k] for k in header if k != "kt"}
                == {k: header0[k] for k in header0 if k != "kt"},
                "frame shape changed",
            )
        final_arrays = arrays
        frame_hashes.append(sha256(path))
        a_i = _interior(arrays["a_i"])[window]
        v_i = _interior(arrays["v_i"])[window]
        h_i = np.divide(v_i, a_i, out=np.zeros_like(v_i), where=a_i != 0.0)
        enthalpy = _interior(arrays["e_i"])[window].sum(axis=(2, 3)) + _interior(arrays["e_s"])[
            window
        ].sum(axis=(2, 3))
        scan.append(
            {
                "kt": float(header["kt"]),
                "a_i_max": float(np.max(a_i)),
                "h_i_max": float(np.max(h_i)),
                "a_i_sum": float(np.sum(a_i, dtype=np.float64)),
                "v_i_sum": float(np.sum(v_i, dtype=np.float64)),
                "ice_snow_enthalpy_j": float(np.sum(enthalpy * area, dtype=np.float64)),
            }
        )
    assert header0 is not None and first_arrays is not None and final_arrays is not None
    aggregate = hashlib.sha256("\n".join(frame_hashes).encode()).hexdigest()
    peak_a = max(scan, key=lambda row: row["a_i_max"])
    peak_h = max(scan, key=lambda row: row["h_i_max"])
    free_drift = free_drift_speed(root) if rung == "3.3" else None
    final_restart = {
        name: _restart_array(run_files(root)["restart"], name)
        for name in final_restart_names(rung)
    }
    maximum_diagnostic = None
    if rung == "3.2":
        maximum_samples: list[dict[str, float | str]] = [
            {"phase": "step_entry", "kt": row["kt"], "a_i_max": row["a_i_max"]}
            for row in scan
        ]
        maximum_samples.append(
            {
                "phase": "post_step_final_restart",
                "kt": float(expected),
                "a_i_max": float(np.max(final_restart["a_i"][window])),
            }
        )
        maximum_diagnostic = maximum_trajectory_diagnostic(maximum_samples)
    shear_diagnostic = rung34_shear_diagnostic(root, window) if rung == "3.4" else None
    return {
        "frame_count": len(paths),
        "first_sha256": frame_hashes[0],
        "last_sha256": frame_hashes[-1],
        "ordered_frame_hash_aggregate": aggregate,
        "header": header0,
        "registry": [
            {"name": name, "shape": list(shape), "time_level": level, "source": source}
            for name, shape, level, source in FRAME_REGISTRY
        ],
        "free_drift": free_drift,
        # Measured over EVERY frame, so no claim about the trajectory rests on an
        # endpoint sample or on an uncommitted probe.  Reported, not scored.
        "full_scan_unclassified": {
            "domain": "surface-tmask wet window, interior of the two-cell halo",
            "trajectory_a_i_max": peak_a["a_i_max"],
            "trajectory_a_i_max_kt": peak_a["kt"],
            "trajectory_h_i_max": peak_h["h_i_max"],
            "trajectory_h_i_max_kt": peak_h["kt"],
            "a_i_sum_first": scan[0]["a_i_sum"],
            "a_i_sum_second": scan[1]["a_i_sum"] if len(scan) > 1 else None,
            "a_i_sum_last": scan[-1]["a_i_sum"],
            "v_i_sum_first": scan[0]["v_i_sum"],
            "v_i_sum_last": scan[-1]["v_i_sum"],
            "ice_snow_enthalpy_j_first": scan[0]["ice_snow_enthalpy_j"],
            "ice_snow_enthalpy_j_last": scan[-1]["ice_snow_enthalpy_j"],
            "ice_snow_enthalpy_j_loss": scan[0]["ice_snow_enthalpy_j"]
            - scan[-1]["ice_snow_enthalpy_j"],
            "ice_snow_enthalpy_j_step_losses": [
                {
                    "kt": scan[i]["kt"],
                    "loss_j": scan[i]["ice_snow_enthalpy_j"] - scan[i + 1]["ice_snow_enthalpy_j"],
                }
                for i in range(len(scan) - 1)
                if scan[i]["ice_snow_enthalpy_j"] != scan[i + 1]["ice_snow_enthalpy_j"]
            ],
        },
        "phenomenology": phenomenology(
            rung,
            first_arrays,
            final_restart,
            window,
            free_drift,
            maximum_diagnostic,
            shear_diagnostic,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--rung", choices=tuple(RUNGS), required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--emit-manifest", action="store_true")
    parser.add_argument("--plant-unaccounted", action="store_true")
    parser.add_argument("--plant-field", action="store_true")
    args = parser.parse_args()
    root = args.run_dir.resolve()
    if args.emit_manifest:
        manifest = disposition_template(root, args.rung)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    require(args.manifest is not None, "--manifest is required")
    manifest = json.loads(args.manifest.read_text())
    counts = check_manifest(root, args.rung, manifest, plant_unaccounted=args.plant_unaccounted)
    selectors = resolved_selectors(root, args.rung)
    keys = cpp_keys(args.case_dir.resolve())
    decks = input_namelists(root, args.rung)
    geom = geometry(root, args.rung, plant_field=args.plant_field)
    traj = trajectory(root, args.rung)
    conservation = conservation_diagnostics(root)
    losses = {
        row["kt"]: row["loss_j"]
        for row in traj["full_scan_unclassified"]["ice_snow_enthalpy_j_step_losses"]
    }
    # Reported, not scored: does each printed heat violation equal the ice+snow
    # enthalpy that left the frames during that same step?  The last step has no
    # successor frame to difference, so it carries no ratio.
    attribution = [
        {
            "kt": event["kt"],
            "violation_j": event["violation_j"],
            "frame_enthalpy_loss_j": losses.get(event["kt"]),
            "ratio": (
                event["violation_j"] / losses[event["kt"]] if losses.get(event["kt"]) else None
            ),
        }
        for event in conservation["violation_heat_events"]
    ]
    unmeasured = [
        "legoesm_alignment",
        "post-dynamics_stage_arrays",
        "landfast_L16_deferred_lane4",
    ]
    if args.rung == "3.2":
        unmeasured.extend(
            [
                "maximum_concentration_documentation_conformance",
                "within_step_prather_split_states",
            ]
        )
    if args.rung == "3.4":
        unmeasured.extend(
            [
                "README_EVP_less_defined_numeric_conformance_no_threshold",
                "README_long_run_EVP_vs_EAP_intersection_angle_out_of_scope",
                "SI3_native_conservation_check_resolved_off",
            ]
        )
    if args.rung == "3.4":
        overall_status = str(traj["phenomenology"]["status"])
    else:
        overall_status = (
            "VERIFIED"
            if conservation["status"] == traj["phenomenology"]["status"] == "CONFIRM"
            else "DEBT"
        )
    print(
        json.dumps(
            {
                "status": overall_status,
                "coverage_status": "VERIFIED",
                "scope": "NEMO_ORACLE_ONLY",
                "git_sha": git_sha(),
                "rung": args.rung,
                "inventory_counts": counts,
                "resolved_inherited_settings": selectors,
                "cpp": keys,
                "input_decks_sha256": decks,
                "geometry": geom,
                "trajectory": traj,
                "conservation_diagnostics": conservation,
                "heat_residual_attribution_unclassified": attribution,
                "unmeasured": unmeasured,
            },
            indent=2,
            sort_keys=True,
        )
    )
    verdicts = (conservation["status"], traj["phenomenology"]["status"])
    return 0 if set(verdicts) == {"CONFIRM"} else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GateError as exc:
        print(f"DEBT: {exc}", file=__import__("sys").stderr)
        raise SystemExit(1)
