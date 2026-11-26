#!/usr/bin/env python3
"""ORCA2 Round 342: observer-free K0 2,920-step (1-year) NEMO twin acquisition gate.

Extends the reviewed Round-339 K0 smoke (`nemo_testcase_l4_orca2_round339_common.py`,
`nemo_testcase_l4_orca2_round339_k0_smoke_gate.py`) and Round-340 S4
(`nemo_testcase_l4_orca2_round340_standalone_gate.py`) tools to acquire the
standalone ORCA2 one-year (2,920-step, 365-day) observer-free K0 reference run
and its reproducibility twin (`orca2_k0_year1_a_np2`, `orca2_k0_year1_b_np2`).

With `nn_it000 = 1`, `nn_itend = 2920`, `nn_stock = 8`, `nn_write = 2920`, and
`ln_rst_list = .false.`, NEMO's `rst_opn` / `rst_write`
(`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/restart.f90:75-150,153-253`
called from `stprk3.f90:247-249` after the stage-3 time-level swap
`Nbb <-> Naa` at `stprk3.f90:223-225`) writes 365 daily ocean restart pairs
`ORCA2_{kt:08d}_restart_{0000,0001}.nc` at `kt = 8, 16, ..., 2920` containing
the completed end-of-day state (`tn, sn, un, vn, sshn` from `Kbb`) needed by
the Decision-160 ORCA2X long-run metric card, with zero in-loop observers.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

import netCDF4
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for _pkg in reversed((REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean")):
    if str(_pkg) not in sys.path:
        sys.path.insert(0, str(_pkg))

from legoesm.ocean.fidelity.provenance import git_sha, worktree_stamp  # noqa: E402
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round328_dynzdf_gate as r328,
    nemo_testcase_l4_orca2_round339_common as common,
    nemo_testcase_l4_orca2_round339_k0_smoke_gate as r339_k0,
    nemo_testcase_l4_orca2_round340_standalone_gate as r340_s4,
    nemo_testcase_l4_orca2x_round1_pilot_metrics as pilot_metrics,
)

GateError = common.GateError
require = common.require
sha256_file = common.sha256

NEMO_ROOT = Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2")
TARGET_CFG_NAME = "ORCA2_STANDALONE_IIF_OMIP_L4_R342S4Y"
DEFAULT_BUILD_CFG = NEMO_ROOT / "cfgs" / TARGET_CFG_NAME
PRISTINE_K0_CFG = NEMO_ROOT / "cfgs" / "ORCA2_STANDALONE_IIF_OMIP_L4_R339K0"
NAMELIST_REF = NEMO_ROOT / "cfgs" / "SHARED" / "namelist_ref"

SOURCE_RUN = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round5/"
    "acquisition/orca1ice_surface_entry_every_step_a_np2"
)
TOOL_A_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round339/"
    "ice_cover_producer/orca2_full_si3_year1_np2"
)
DEFAULT_ICE_COVER_PATH = TOOL_A_ROOT / "ice_cover_clim.nc"
R339_SMOKE_DIR = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round339/"
    "k0_smoke/orca2_k0_10step_np2"
)
R339_NAMELIST_CFG = R339_SMOKE_DIR / "namelist_cfg"
R339_ARCHIVED_PPSRC = R339_SMOKE_DIR / "build_sources" / "ppsrc" / "nemo"
R339_RESTART_0 = R339_SMOKE_DIR / "ORCA2_00000010_restart_0000.nc"
R339_RESTART_1 = R339_SMOKE_DIR / "ORCA2_00000010_restart_0001.nc"
R340_RUN_A = (
    Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds")
    / "round340"
    / "s4_acquisition"
    / "orca2_s4_exact_a_np2"
)

DEFAULT_RUN_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round342/s4_year"
)
TWIN_A_NAME = "orca2_k0_year1_a_np2"
TWIN_B_NAME = "orca2_k0_year1_b_np2"
DEFAULT_EVIDENCE_DIR = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r342"
)
REAL_ARM_A_LOGS_DIR = (
    REPO_ROOT / "tests" / "ocean" / "fidelity" / "data" / "orca2_round342_arm_a_real_logs"
)

YEAR_STEPS = 2920
DAILY_STOCK_STRIDE = 8
EXPECTED_DAILY_KTS = tuple(range(DAILY_STOCK_STRIDE, YEAR_STEPS + 1, DAILY_STOCK_STRIDE))
PER_ARM_TIMEOUT_SECONDS = 5400
REQUIRED_RAM_GIB = 150.0
REQUIRED_DISK_GIB = 50.0

FULL_YEAR_RECORD_REQUIREMENTS: dict[str, tuple[str, int]] = {
    "u_10.15JUNE2009_fill.nc": ("U_10_MOD", 1460),
    "v_10.15JUNE2009_fill.nc": ("V_10_MOD", 1460),
    "t_10.15JUNE2009_fill.nc": ("T_10_MOD", 1460),
    "q_10.15JUNE2009_fill.nc": ("Q_10_MOD", 1460),
    "slp.15JUNE2009_fill.nc": ("SLP", 1460),
    "ncar_rad.15JUNE2009_fill.nc": ("SWDN_MOD", 365),
    "ncar_precip.15JUNE2009_fill.nc": ("PRC_MOD1", 12),
    "runoff_core_monthly.nc": ("sorunoff", 12),
    "sss_data.nc": ("sss", 12),
    "chlorophyll.nc": ("CHLA", 12),
    "data_1m_potential_temperature_nomask.nc": ("votemper", 12),
    "data_1m_salinity_nomask.nc": ("vosaline", 12),
    "ice_cover_clim.nc": ("ice_cover", 1),
}

NONZERO_RESTART_VARS = ("tn", "sn", "un", "vn", "sshn")

OBSERVER_SYMBOL_PATTERNS = (
    re.compile(r"\br340_s4_", re.IGNORECASE),
    re.compile(r"\br336_tke_", re.IGNORECASE),
    re.compile(r"\br328_", re.IGNORECASE),
    re.compile(r"\borca2_s4_surface_", re.IGNORECASE),
    re.compile(r"\borca2x_longrun_hmi_write\b", re.IGNORECASE),
)

NEMO_BUILD_BINS = (
    Path("/home/dbalwada/miniconda3/envs/nemo-build/bin"),
    Path("/home/dbalwada/spack-envs/nemo5/.spack-env/view/bin"),
)


def _tool_bin(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    for bin_dir in NEMO_BUILD_BINS:
        candidate = bin_dir / name
        if candidate.is_file():
            return str(candidate)
    return name


def _tool_env() -> dict[str, str]:
    env = dict(os.environ)
    extra = ":".join(str(p) for p in NEMO_BUILD_BINS if p.is_dir())
    if extra:
        env["PATH"] = f"{extra}:{env.get('PATH', '')}"
    return env


def _scratch_root() -> Path:
    root = DEFAULT_EVIDENCE_DIR / "tmp"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _ppsrc_dir(build_cfg: Path | None = None) -> Path:
    if build_cfg is not None and (build_cfg / "BLD" / "ppsrc" / "nemo").is_dir():
        return build_cfg / "BLD" / "ppsrc" / "nemo"
    if (PRISTINE_K0_CFG / "BLD" / "ppsrc" / "nemo").is_dir():
        return PRISTINE_K0_CFG / "BLD" / "ppsrc" / "nemo"
    require(R339_ARCHIVED_PPSRC.is_dir(), f"missing K0 ppsrc dir: {R339_ARCHIVED_PPSRC}")
    return R339_ARCHIVED_PPSRC


def verify_disk_space(
    target_path: Path,
    *,
    required_gib: float = REQUIRED_DISK_GIB,
) -> dict[str, Any]:
    """Verify free disk space at `target_path` against the 2-twin 365-day footprint."""
    probe = target_path.resolve()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    free_gib = float(usage.free / (2**30))
    total_gib = float(usage.total / (2**30))
    per_day_bytes = (
        int(R339_RESTART_0.stat().st_size + R339_RESTART_1.stat().st_size)
        if R339_RESTART_0.is_file() and R339_RESTART_1.is_file()
        else 56_908_536
    )
    estimated_two_twins_gib = float((2 * len(EXPECTED_DAILY_KTS) * per_day_bytes) / (2**30))
    require(
        free_gib >= required_gib,
        f"insufficient free disk space at {probe}: {free_gib:.2f} GiB available < {required_gib:.2f} GiB required",
    )
    return {
        "probe_path": str(probe),
        "free_gib": round(free_gib, 2),
        "total_gib": round(total_gib, 2),
        "estimated_two_twins_365d_gib": round(estimated_two_twins_gib, 2),
        "required_gib": float(required_gib),
    }


def plant_disk_space_exhaustion(target_path: Path, run_sh: Path) -> dict[str, Any]:
    """Plant a 10^9 GiB disk requirement in both Python and run.sh and prove both refuse."""
    py_caught = False
    py_error = ""
    try:
        verify_disk_space(target_path, required_gib=1.0e9)
    except GateError as exc:
        py_caught = True
        py_error = str(exc)
    require(py_caught, "plant_disk_space_exhaustion did not fail Python verify_disk_space")

    env = dict(os.environ, R342_REQUIRED_DISK_GIB="1000000000")
    sh_res = subprocess.run(
        ["bash", str(run_sh), "--preflight-only"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    require(
        sh_res.returncode != 0 and "REFUSE: available disk space" in sh_res.stderr,
        f"plant_disk_space_exhaustion did not fail run.sh: rc={sh_res.returncode} stderr={sh_res.stderr}",
    )
    return {
        "passed": True,
        "python_error": py_error,
        "shell_rc": sh_res.returncode,
        "shell_stderr": sh_res.stderr.strip(),
    }


def verify_available_ram(*, required_gib: float = REQUIRED_RAM_GIB) -> dict[str, Any]:
    """Verify available system RAM meets the Standing Note Memory Rule (>= 150 GiB)."""
    meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
    m = re.search(r"^MemAvailable:\s+(\d+)\s+kB", meminfo, re.MULTILINE)
    require(m is not None, "MemAvailable not found in /proc/meminfo")
    avail_kib = int(m.group(1))
    avail_gib = float(avail_kib / (1024.0 * 1024.0))
    free_out = subprocess.check_output(["free", "-g"], text=True)
    free_g = 0
    for line in free_out.splitlines():
        if line.startswith("Mem:"):
            parts = line.split()
            if len(parts) >= 7:
                free_g = int(parts[6])
            break
    require(
        avail_gib >= required_gib and free_g >= int(required_gib),
        f"insufficient available RAM: {avail_gib:.2f} GiB (free -g: {free_g} GiB) < {required_gib:.2f} GiB required",
    )
    return {
        "available_gib": round(avail_gib, 2),
        "free_g_column7": free_g,
        "required_gib": float(required_gib),
    }


def plant_ram_exhaustion(run_sh: Path) -> dict[str, Any]:
    """Plant a 999,999 GiB RAM requirement in both Python and run.sh and prove both refuse."""
    py_caught = False
    py_error = ""
    try:
        verify_available_ram(required_gib=999_999.0)
    except GateError as exc:
        py_caught = True
        py_error = str(exc)
    require(py_caught, "plant_ram_exhaustion did not fail Python verify_available_ram")

    env = dict(os.environ, R342_REQUIRED_RAM_GIB="999999")
    sh_res = subprocess.run(
        ["bash", str(run_sh), "--preflight-only"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    require(
        sh_res.returncode != 0 and "REFUSE: available RAM" in sh_res.stderr,
        f"plant_ram_exhaustion did not fail run.sh: rc={sh_res.returncode} stderr={sh_res.stderr}",
    )
    return {
        "passed": True,
        "python_error": py_error,
        "shell_rc": sh_res.returncode,
        "shell_stderr": sh_res.stderr.strip(),
    }


def verify_clean_worktree(repo: Path = REPO_ROOT) -> dict[str, Any]:
    """Refuse modified tracked files or untracked files in `repo` (fail-closed worktree guard)."""
    try:
        porcelain = subprocess.check_output(
            ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise GateError(f"git status failed on {repo}: {exc}") from exc
    dirty_lines = [line for line in porcelain.splitlines() if line.strip()]
    require(
        not dirty_lines,
        f"refusing dirty git worktree {repo}: {len(dirty_lines)} modified/untracked file(s): {dirty_lines[:5]}",
    )
    try:
        sha = git_sha(allow_dirty=False, repo=repo)
        stamp = worktree_stamp(repo=repo, allow_dirty=False)
    except RuntimeError as exc:
        raise GateError(f"worktree provenance check refused on {repo}: {exc}") from exc
    require(stamp.get("clean") is True and stamp.get("commit") == sha,
            f"worktree_stamp reported non-clean state: {stamp}")
    return stamp


def plant_dirty_worktree(run_sh: Path) -> dict[str, Any]:
    """Create a temporary git repo, plant untracked and modified tracked files, and prove both Python and run.sh refuse."""
    with tempfile.TemporaryDirectory(dir=_scratch_root()) as tmp:
        tmp_repo = Path(tmp) / "dirty_repo_plant"
        tmp_repo.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", "-q", str(tmp_repo)], check=True)
        subprocess.run(["git", "-C", str(tmp_repo), "config", "user.email", "test@example.com"], check=True)
        subprocess.run(["git", "-C", str(tmp_repo), "config", "user.name", "Test"], check=True)
        tracked = tmp_repo / "tracked.txt"
        tracked.write_text("clean baseline\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(tmp_repo), "add", "tracked.txt"], check=True)
        subprocess.run(["git", "-C", str(tmp_repo), "commit", "-q", "-m", "init"], check=True)

        # Verify clean temporary repo passes verify_clean_worktree
        clean_stamp = verify_clean_worktree(tmp_repo)
        require(clean_stamp["clean"] is True, "clean temp repo did not pass verify_clean_worktree")

        # Sub-plant A: untracked file
        untracked = tmp_repo / "untracked_plant.txt"
        untracked.write_text("untracked dirt\n", encoding="utf-8")
        untracked_caught = False
        untracked_err = ""
        try:
            verify_clean_worktree(tmp_repo)
        except GateError as exc:
            untracked_caught = True
            untracked_err = str(exc)
        require(untracked_caught, "plant_dirty_worktree did not catch untracked file")

        sh_untracked = subprocess.run(
            ["bash", str(run_sh), "--preflight-only"],
            env=dict(os.environ, R342_CHECK_REPO_ROOT=str(tmp_repo)),
            capture_output=True,
            text=True,
            check=False,
        )
        require(
            sh_untracked.returncode != 0 and "REFUSE: git worktree" in sh_untracked.stderr,
            "run.sh did not refuse untracked file in R342_CHECK_REPO_ROOT",
        )
        untracked.unlink()

        # Sub-plant B: modified tracked file
        tracked.write_text("modified tracked file\n", encoding="utf-8")
        modified_caught = False
        modified_err = ""
        try:
            verify_clean_worktree(tmp_repo)
        except GateError as exc:
            modified_caught = True
            modified_err = str(exc)
        require(modified_caught, "plant_dirty_worktree did not catch modified tracked file")

        sha_caught = False
        try:
            git_sha(allow_dirty=False, repo=tmp_repo)
        except RuntimeError:
            sha_caught = True
        require(sha_caught, "git_sha(allow_dirty=False) did not refuse modified tracked file")

        sh_modified = subprocess.run(
            ["bash", str(run_sh), "--preflight-only"],
            env=dict(os.environ, R342_CHECK_REPO_ROOT=str(tmp_repo)),
            capture_output=True,
            text=True,
            check=False,
        )
        require(
            sh_modified.returncode != 0 and "REFUSE: git worktree" in sh_modified.stderr,
            "run.sh did not refuse modified tracked file in R342_CHECK_REPO_ROOT",
        )
    return {
        "passed": True,
        "untracked_error": untracked_err,
        "modified_tracked_error": modified_err,
        "shell_untracked_rc": sh_untracked.returncode,
        "shell_modified_rc": sh_modified.returncode,
    }


def verify_run_sh_syntax_and_plant(run_sh: Path) -> dict[str, Any]:
    """Checklist 15: verify `bash -n` on `run.sh` and prove a planted syntax error fails."""
    require(run_sh.is_file(), f"missing run.sh: {run_sh}")
    run_text = run_sh.read_text(encoding="utf-8")
    require(
        "rm -rf" not in run_text,
        "run.sh must not contain 'rm -rf' (Checklist 13: quarantine to refused/, never delete)",
    )
    require(
        "timeout --foreground" in run_text,
        "run.sh must wrap mpirun with 'timeout --foreground'",
    )
    ok_res = subprocess.run(["bash", "-n", str(run_sh)], capture_output=True, text=True, check=False)
    require(ok_res.returncode == 0, f"bash -n failed on {run_sh}: {ok_res.stderr}")

    with tempfile.TemporaryDirectory(dir=_scratch_root()) as tmp:
        bad_sh = Path(tmp) / "run_bad_syntax.sh"
        bad_sh.write_text(run_text + "\nif then\n", encoding="utf-8")
        bad_res = subprocess.run(["bash", "-n", str(bad_sh)], capture_output=True, text=True, check=False)
        require(bad_res.returncode != 0, "planted syntax error in run.sh did not fail bash -n")
    return {
        "passed": True,
        "syntax_plant_rc": bad_res.returncode,
        "syntax_plant_stderr": bad_res.stderr.strip(),
    }


def stage_s4_year_namelist_cfg(
    src_namelist: Path,
    dst_namelist: Path,
    *,
    nn_itend: int = YEAR_STEPS,
    nn_stock: int = DAILY_STOCK_STRIDE,
) -> dict[str, Any]:
    """Stage the observer-free K0 namelist configured for 2,920 steps and daily restarts."""
    require(nn_itend >= DAILY_STOCK_STRIDE and nn_itend % DAILY_STOCK_STRIDE == 0,
            f"nn_itend ({nn_itend}) must be a positive multiple of {DAILY_STOCK_STRIDE}")
    require(nn_stock == DAILY_STOCK_STRIDE,
            f"nn_stock ({nn_stock}) must be {DAILY_STOCK_STRIDE} for daily restarts")
    text = src_namelist.read_text(encoding="utf-8")
    text, n_itend = re.subn(
        r"(\bnn_itend\s*=\s*)\d+",
        rf"\g<1>{nn_itend}",
        text,
        count=1,
    )
    text, n_stock = re.subn(
        r"(\bnn_stock\s*=\s*)\d+",
        rf"\g<1>{nn_stock}",
        text,
        count=1,
    )
    require(n_itend == 1 and n_stock == 1,
            f"failed to patch nn_itend/nn_stock in {dst_namelist}")
    require(re.search(r"\bln_rst_list\s*=\s*\.true\.", text, re.IGNORECASE) is None,
            "namelist_cfg must not set ln_rst_list = .true. for frequency-based nn_stock")
    dst_namelist.write_text(text, encoding="utf-8")
    common.validate_group_membership(dst_namelist, r339_k0.GROUPS)
    return {
        "nn_it000": 1,
        "nn_itend": nn_itend,
        "nn_stock": nn_stock,
        "expected_daily_restarts": nn_itend // nn_stock,
        "sha256": sha256_file(dst_namelist),
    }


def parse_namelist_itend_stock(namelist_cfg: Path) -> tuple[int, int]:
    """Parse nn_itend and nn_stock from a staged namelist_cfg."""
    text = namelist_cfg.read_text(encoding="utf-8")
    m_end = re.search(r"\bnn_itend\s*=\s*(\d+)", text)
    m_stock = re.search(r"\bnn_stock\s*=\s*(\d+)", text)
    require(m_end is not None and m_stock is not None,
            f"missing nn_itend or nn_stock in {namelist_cfg}")
    return int(m_end.group(1)), int(m_stock.group(1))


def verify_namelist_groups_against_ppsrc(
    namelist_cfg: Path,
    namelist_ref: Path,
    ppsrc_dir: Path,
) -> dict[str, Any]:
    """Checklist 8: validate every namelist group against namelist_ref and BLD/ppsrc/nemo/*.f90 NAMELIST declarations."""
    required_groups = {
        **r339_k0.GROUPS,
        "nn_it000": "namrun",
        "nn_itend": "namrun",
        "nn_stock": "namrun",
    }
    common.validate_group_membership(namelist_ref, required_groups)
    common.validate_group_membership(namelist_cfg, required_groups)

    ppsrc_group_vars: dict[str, set[str]] = {}
    namelist_re = re.compile(
        r"NAMELIST\s*/\s*([A-Za-z0-9_]+)\s*/\s*([^\n!]+)",
        re.IGNORECASE,
    )
    for f90 in sorted(ppsrc_dir.glob("*.f90")):
        joined = common._strip_fortran_comments_and_join_continuations(
            f90.read_text(encoding="utf-8", errors="replace")
        )
        for match in namelist_re.finditer(joined):
            gname = match.group(1).lower()
            vlist = {
                tok.strip().lower()
                for tok in match.group(2).split(",")
                if tok.strip()
            }
            ppsrc_group_vars.setdefault(gname, set()).update(vlist)
    require(len(ppsrc_group_vars) >= 40,
            f"too few NAMELIST groups parsed from {ppsrc_dir}: {len(ppsrc_group_vars)}")

    for var, grp in required_groups.items():
        require(grp in ppsrc_group_vars,
                f"required NAMELIST group {grp!r} missing from {ppsrc_dir}")
        require(var.lower() in ppsrc_group_vars[grp],
                f"variable {var!r} missing from NAMELIST /{grp}/ in {ppsrc_dir}")

    group_re = re.compile(r"^\s*&([A-Za-z0-9_]+)", re.MULTILINE)
    cfg_groups = [m.group(1).lower() for m in group_re.finditer(namelist_cfg.read_text(encoding="utf-8"))]
    ref_groups = [m.group(1).lower() for m in group_re.finditer(namelist_ref.read_text(encoding="utf-8"))]
    require(len(cfg_groups) >= 20, f"too few groups in {namelist_cfg}: {len(cfg_groups)}")
    unknown_cfg = sorted(set(cfg_groups) - (set(ref_groups) | set(ppsrc_group_vars)))
    require(not unknown_cfg,
            f"namelist_cfg contains undeclared NAMELIST group(s): {unknown_cfg}")
    return {
        "ppsrc_declared_group_count": len(ppsrc_group_vars),
        "cfg_group_count": len(cfg_groups),
        "ref_group_count": len(ref_groups),
        "unknown_cfg_groups": unknown_cfg,
    }


def plant_namelist_error(ppsrc_dir: Path) -> dict[str, Any]:
    """Plant an undeclared namelist group and an E R R O R block in ocean.output and prove both fail."""
    with tempfile.TemporaryDirectory(dir=_scratch_root()) as tmp:
        root = Path(tmp)
        bad_cfg = root / "namelist_cfg"
        stage_s4_year_namelist_cfg(R339_NAMELIST_CFG, bad_cfg)
        bad_cfg.write_text(
            bad_cfg.read_text(encoding="utf-8") + "\n&nam_nonexistent_r342\n   nn_fake = 1\n/\n",
            encoding="utf-8",
        )
        group_caught = False
        group_error = ""
        try:
            verify_namelist_groups_against_ppsrc(bad_cfg, NAMELIST_REF, ppsrc_dir)
        except GateError as exc:
            group_caught = True
            group_error = str(exc)
        require(group_caught, "plant_namelist_error failed to catch undeclared namelist group")

        bad_out = root / "ocean.output"
        bad_out.write_text(
            " ===>>> : E R R O R\n         =========\n misspelled namelist token\n",
            encoding="utf-8",
        )
        out_caught = False
        out_error = ""
        try:
            common.validate_ocean_output(bad_out, r339_k0.RESOLVED)
        except GateError as exc:
            out_caught = True
            out_error = str(exc)
        require(out_caught, "plant_namelist_error failed to catch E R R O R in ocean.output")
    return {
        "passed": True,
        "undeclared_group_error": group_error,
        "ocean_output_error": out_error,
    }


def _snapshot_build_sources(build_cfg: Path, dst_build_sources: Path) -> None:
    eff_cfg = build_cfg if (build_cfg / "BLD" / "ppsrc" / "nemo").is_dir() else PRISTINE_K0_CFG
    if (eff_cfg / "BLD" / "ppsrc").is_dir():
        dst_build_sources.mkdir(parents=True, exist_ok=True)
        (dst_build_sources / "MY_SRC").mkdir(parents=True, exist_ok=True)
        (dst_build_sources / "BLD").mkdir(parents=True, exist_ok=True)
        shutil.copytree(eff_cfg / "BLD" / "ppsrc", dst_build_sources / "ppsrc", symlinks=True, dirs_exist_ok=True)
        for name in ("arch_nemo.fcm", "cpp.fcm", "bld.cfg", "full_key_list.txt"):
            src = eff_cfg / "BLD" / name
            if src.is_file():
                shutil.copy2(src, dst_build_sources / "BLD" / name)
    else:
        shutil.copytree(R339_SMOKE_DIR / "build_sources", dst_build_sources, symlinks=True, dirs_exist_ok=True)


def stage_s4_year_run_dir(
    run_dir: Path,
    ice_cover_path: Path = DEFAULT_ICE_COVER_PATH,
    *,
    build_cfg: Path = DEFAULT_BUILD_CFG,
    nn_itend: int = YEAR_STEPS,
    nn_stock: int = DAILY_STOCK_STRIDE,
) -> dict[str, Any]:
    """Checklist 9 & 13: stage all inputs and namelists for one 2,920-step K0 twin arm without overwriting retained runs."""
    if run_dir.exists():
        existing_items = [p.name for p in run_dir.iterdir()]
        require(
            not existing_items,
            f"refusing to stage into non-empty retained directory {run_dir} ({existing_items[:5]}); "
            "quarantine to refused/ or move aside before re-running",
        )
    else:
        run_dir.mkdir(parents=True, exist_ok=False)

    for manifest_name in ("deck_files.sha256", "input_files.sha256"):
        mfile = SOURCE_RUN / manifest_name
        require(mfile.is_file(), f"missing source manifest: {mfile}")
        for line in mfile.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                continue
            _, fname = parts
            src_item = SOURCE_RUN / fname
            dst_item = run_dir / fname
            if src_item.is_symlink():
                os.symlink(os.readlink(src_item), dst_item)
            else:
                shutil.copy2(src_item, dst_item)

    require(ice_cover_path.is_file(), f"missing ice_cover_clim.nc: {ice_cover_path}")
    shutil.copy2(ice_cover_path, run_dir / "ice_cover_clim.nc")

    namelist_info = stage_s4_year_namelist_cfg(
        R339_NAMELIST_CFG,
        run_dir / "namelist_cfg",
        nn_itend=nn_itend,
        nn_stock=nn_stock,
    )
    shutil.copy2(run_dir / "namelist_cfg", run_dir / "staged.namelist_cfg")
    (run_dir / "compiled_keys.txt").write_text(
        "key_si3\nkey_qco\nkey_vco_1d3d\nkey_RK3\n", encoding="utf-8"
    )
    _snapshot_build_sources(build_cfg, run_dir / "build_sources")
    record_checks = verify_staged_full_year_inputs(run_dir)
    return {
        "run_dir": str(run_dir),
        "namelist_s4_year": namelist_info,
        "full_year_record_checks": record_checks,
    }


def verify_staged_full_year_inputs(run_dir: Path) -> dict[str, Any]:
    """Verify every staged NetCDF forcing file has enough records for a full 365-day year."""
    r340_s4.read_decision171_cover(run_dir / "ice_cover_clim.nc")
    checked: dict[str, Any] = {}
    for fname, (var_name, min_records) in FULL_YEAR_RECORD_REQUIREMENTS.items():
        fpath = run_dir / fname
        require(fpath.exists(), f"missing staged full-year input: {fpath}")
        with netCDF4.Dataset(fpath, "r") as ds:
            require(var_name in ds.variables,
                    f"variable {var_name!r} missing from staged {fname}")
            shape = tuple(int(s) for s in ds.variables[var_name].shape)
        require(len(shape) >= 3 and shape[0] >= min_records,
                f"{fname}:{var_name} has shape {shape}, need >= {min_records} records for 2,920 steps")
        checked[fname] = {"variable": var_name, "shape": list(shape), "min_records": min_records}
    return checked


def plant_missing_forcing(ice_cover_path: Path) -> dict[str, Any]:
    """Plant a truncated-record forcing file and prove verify_staged_full_year_inputs fails."""
    with tempfile.TemporaryDirectory(dir=_scratch_root()) as tmp:
        run_dir = Path(tmp) / "run"
        stage_s4_year_run_dir(run_dir, ice_cover_path)
        slp_path = run_dir / "slp.15JUNE2009_fill.nc"
        slp_path.unlink()
        with netCDF4.Dataset(slp_path, "w", format="NETCDF4_CLASSIC") as ds:
            ds.createDimension("time", 5)
            ds.createDimension("lat", 94)
            ds.createDimension("lon", 192)
            v = ds.createVariable("SLP", "f8", ("time", "lat", "lon"))
            v[:] = 101325.0
        caught = False
        error_msg = ""
        try:
            verify_staged_full_year_inputs(run_dir)
        except GateError as exc:
            caught = True
            error_msg = str(exc)
        require(caught, "plant_missing_forcing did not fail on 5-record SLP file")
    return {"passed": True, "error": error_msg}


def verify_allocate_extents(ppsrc_dir: Path) -> dict[str, Any]:
    """Checklist 3: verify array extents directly from ALLOCATE in BLD/ppsrc/nemo/*.f90."""
    expected_by_unit = {
        "oce": {
            "uu": "jpi,jpj,jpk,jpt",
            "vv": "jpi,jpj,jpk,jpt",
            "ts": "jpi,jpj,jpk,jpts,jpt",
            "ssh": "jpi,jpj,jpt",
        },
        "dom_oce": {
            "tmask": "jpi,jpj,jpk",
            "umask": "jpi,jpj,jpk",
            "vmask": "jpi,jpj,jpk",
        },
        "sbc_oce": {
            "fr_i": "jpi,jpj",
        },
    }
    verified = common.validate_nemo_allocated_extents(ppsrc_dir, expected_by_unit)
    return {"ppsrc_dir": str(ppsrc_dir), "verified_allocates": verified}


def plant_allocate_mismatch(ppsrc_dir: Path) -> dict[str, Any]:
    """Plant a wrong ALLOCATE extent in a copy of oce.f90 and prove verify_allocate_extents fails."""
    with tempfile.TemporaryDirectory(dir=_scratch_root()) as tmp:
        fake_dir = Path(tmp)
        for name in ("oce.f90", "dom_oce.f90", "sbc_oce.f90"):
            shutil.copy2(ppsrc_dir / name, fake_dir / name)
        oce_path = fake_dir / "oce.f90"
        oce_path.write_text(
            oce_path.read_text(encoding="utf-8").replace(
                "ssh(jpi,jpj,jpt)", "ssh(jpi,jpj,jpk)"
            ),
            encoding="utf-8",
        )
        caught = False
        error_msg = ""
        try:
            verify_allocate_extents(fake_dir)
        except GateError as exc:
            caught = True
            error_msg = str(exc)
        require(caught, "plant_allocate_mismatch did not fail on perturbed ssh ALLOCATE")
    return {"passed": True, "error": error_msg}


def verify_restart_fortran_fixture(
    ppsrc_dir: Path,
    *,
    plant_wrong_time_level: bool = False,
) -> dict[str, Any]:
    """Checklist 4 & 5: compile and run a `-fcheck=bounds -g -O0` Fortran fixture."""
    rst_src = (ppsrc_dir / "restart.f90").read_text(encoding="utf-8")
    stprk3_src = (ppsrc_dir / "stprk3.f90").read_text(encoding="utf-8")
    require("MOD( kt - 1, nn_stock ) == 0" in rst_src,
            "restart.f90 missing MOD( kt - 1, nn_stock ) == 0 periodicity")
    require("CALL iom_rstput( kt, nitrst, numrow, 'sshn', ssh(:,:        ,Kbb) )" in rst_src,
            "restart.f90 missing sshn Kbb write")
    require("IF( lrst_oce   )   CALL rst_write    ( kstp, Nbb, Nnn, Naa )" in stprk3_src,
            "stprk3.f90 missing rst_write( kstp, Nbb, Nnn, Naa ) call")

    time_level_expr = "Kaa" if plant_wrong_time_level else "Kbb"
    fortran_code = f"""program test_r342_restart_fixture
  use iso_fortran_env, only: real64, int32
  implicit none
  integer, parameter :: wp = real64
  integer, parameter :: jpi = 92, jpj = 149, jpk = 31, jpts = 2, jpt = 3
  integer, parameter :: jp_tem = 1, jp_sal = 2
  integer, parameter :: nit000 = 1, nitend = 2920, nn_stock = 8
  logical, parameter :: ln_rst_list = .false.

  real(wp), allocatable :: ts(:,:,:,:,:), uu(:,:,:,:), vv(:,:,:,:), ssh(:,:,:)
  real(wp), allocatable :: out_tn(:,:,:), out_sshn(:,:)
  integer :: kt, nitrst, n_writes, first_kt, last_kt
  integer :: Nbb, Nmm, Naa, itmp, ji, jj, jk, narea
  logical :: lrst_oce

  allocate(ts(jpi,jpj,jpk,jpts,jpt), uu(jpi,jpj,jpk,jpt), vv(jpi,jpj,jpk,jpt), ssh(jpi,jpj,jpt))
  allocate(out_tn(jpi,jpj,jpk), out_sshn(jpi,jpj))

  n_writes = 0
  first_kt = -1
  last_kt = -1
  nitrst = nitend
  lrst_oce = .false.
  Nbb = 1
  Nmm = 2
  Naa = 3

  do kt = nit000, nitend
     itmp = Nbb
     Nbb = Naa
     Naa = itmp

     if (kt == nit000) then
        lrst_oce = .false.
        nitrst = nitend
     end if
     if (.not. ln_rst_list) then
        if (nn_stock == 0) then
           nitrst = nitend
        else if (mod(kt - 1, nn_stock) == 0) then
           nitrst = kt + nn_stock - 1
           if (nitrst > nitend) nitrst = nitend
        end if
     end if
     if (kt == nitrst - 1 .or. nn_stock == 1 .or. (kt == nitend .and. .not. lrst_oce)) then
        if (nitrst <= nitend .and. nitrst > 0) then
           lrst_oce = .true.
        end if
     end if

     if (lrst_oce .and. kt == nitrst) then
        n_writes = n_writes + 1
        if (first_kt < 0) first_kt = kt
        last_kt = kt
        if (mod(kt, nn_stock) /= 0) stop 11
        lrst_oce = .false.
     end if
  end do

  if (n_writes /= 365 .or. first_kt /= 8 .or. last_kt /= 2920) stop 12

  do narea = 1, 2
     Nbb = 1
     Nmm = 2
     Naa = 3
     ts = -999.0_wp
     ssh = -888.0_wp
     do jk = 1, jpk
        do jj = 1, jpj
           do ji = 1, jpi
              ts(ji,jj,jk,jp_tem,Naa) = real(narea * 1000000 + jk * 10000 + jj * 100 + ji, wp)
           end do
        end do
     end do
     do jj = 1, jpj
        do ji = 1, jpi
           ssh(ji,jj,Naa) = real(narea * 100000 + jj * 100 + ji, wp)
        end do
     end do
     ssh(3,4,Naa) = -0.0_wp

     itmp = Nbb
     Nbb = Naa
     Naa = itmp

     call mock_rst_write(Nbb, Nmm, Naa)
     if (out_tn(5,6,7) /= real(narea * 1000000 + 7 * 10000 + 6 * 100 + 5, wp)) stop 13
     if (out_sshn(3,4) /= 0.0_wp .or. .not. ieee_is_negative_zero(out_sshn(3,4))) stop 14
  end do

  print '(A,I0,A,I0,A,I0)', 'FIXTURE_OK writes=', n_writes, ' first=', first_kt, ' last=', last_kt

contains
  subroutine mock_rst_write(Kbb, Kmm, Kaa)
    integer, intent(in) :: Kbb, Kmm, Kaa
    out_tn(:,:,:) = ts(:,:,:,jp_tem,{time_level_expr})
    out_sshn(:,:) = ssh(:,:,{time_level_expr})
  end subroutine mock_rst_write

  logical function ieee_is_negative_zero(x)
    real(wp), intent(in) :: x
    integer(real64) :: bits
    bits = transfer(x, bits)
    ieee_is_negative_zero = (x == 0.0_wp) .and. (ibits(bits, 63, 1) == 1)
  end function ieee_is_negative_zero
end program test_r342_restart_fixture
"""
    with tempfile.TemporaryDirectory(dir=_scratch_root()) as tmp:
        src_path = Path(tmp) / "test_r342_restart_fixture.f90"
        bin_path = Path(tmp) / "test_r342_restart_fixture"
        src_path.write_text(fortran_code, encoding="utf-8")
        comp = subprocess.run(
            [_tool_bin("gfortran"), "-fdefault-real-8", "-fcheck=bounds", "-g", "-O0",
             str(src_path), "-o", str(bin_path)],
            env=_tool_env(),
            capture_output=True,
            text=True,
            check=False,
        )
        require(comp.returncode == 0,
                f"Fortran fixture compilation failed: {comp.stderr}")
        run = subprocess.run(
            [str(bin_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        require(run.returncode == 0 and "FIXTURE_OK writes=365 first=8 last=2920" in run.stdout,
                f"Fortran fixture execution failed (rc={run.returncode}): {run.stdout} {run.stderr}")
    return {"passed": True, "stdout": run.stdout.strip()}


def plant_fixture_wrong_time_level(ppsrc_dir: Path) -> dict[str, Any]:
    """Plant wrong time level (Kaa instead of Kbb after swap) in Fortran fixture and prove it fails."""
    caught = False
    error_msg = ""
    try:
        verify_restart_fortran_fixture(ppsrc_dir, plant_wrong_time_level=True)
    except GateError as exc:
        caught = True
        error_msg = str(exc)
    require(caught, "plant_fixture_wrong_time_level did not fail when reading Kaa instead of Kbb")
    return {"passed": True, "error": error_msg}


def _extract_makefile_fcflags(bld_dir: Path) -> tuple[str, list[str]]:
    makefile = bld_dir / "Makefile"
    require(makefile.is_file(), f"missing BLD/Makefile: {makefile}")
    text = makefile.read_text(encoding="utf-8")
    m = re.search(r"^(?:export\s+)?F(?:C)?FLAGS\s*=\s*(.+)$", text, re.MULTILINE)
    require(m is not None, f"FFLAGS not found in {makefile}")
    raw = m.group(1).strip()
    tokens = [
        tok for tok in raw.split()
        if not tok.startswith("-J")
    ]
    return raw, tokens


def verify_real_flag_fp_census(
    reference_bld: Path,
    candidate_bld: Path | None = None,
) -> dict[str, Any]:
    """Checklist 6: real-flag whole-executable FP instruction census + real-flag recompile check."""
    ref_obj = reference_bld / "obj"
    require(ref_obj.is_dir(), f"missing reference obj dir: {ref_obj}")
    cand_obj = (candidate_bld / "obj") if (candidate_bld and (candidate_bld / "obj").is_dir()) else ref_obj

    ref_files = sorted(p.name for p in ref_obj.glob("*.o"))
    cand_files = sorted(p.name for p in cand_obj.glob("*.o"))
    require(ref_files == cand_files and len(ref_files) == 425,
            f"object inventory mismatch: ref={len(ref_files)} cand={len(cand_files)}")

    total_functions = 0
    total_fp_instructions = 0
    for name in ref_files:
        ref_c = common.fp_instruction_census(ref_obj / name)
        cand_c = common.fp_instruction_census(cand_obj / name)
        require(ref_c == cand_c, f"FP instruction census mismatch in {name}")
        total_functions += len(ref_c)
        total_fp_instructions += sum(ref_c.values())

    raw_fcflags, flag_tokens = _extract_makefile_fcflags(reference_bld)
    with tempfile.TemporaryDirectory(dir=_scratch_root()) as tmp:
        recompiled_obj = Path(tmp) / "restart.o"
        cmd = [
            _tool_bin("mpif90"),
            "-c",
            *flag_tokens,
            f"-I{reference_bld / 'inc'}",
            f"-J{tmp}",
            str(reference_bld / "ppsrc" / "nemo" / "restart.f90"),
            "-o",
            str(recompiled_obj),
        ]
        res = subprocess.run(cmd, env=_tool_env(), capture_output=True, text=True, check=False)
        require(res.returncode == 0,
                f"real-flag recompile of restart.f90 failed: {res.stderr}")
        recomp_c = common.fp_instruction_census(recompiled_obj)
        orig_c = common.fp_instruction_census(ref_obj / "restart.o")
        require(recomp_c == orig_c,
                "real-flag recompiled restart.o differs in FP census from reference restart.o")

    return {
        "object_count": len(ref_files),
        "function_count": total_functions,
        "fp_instruction_count": total_fp_instructions,
        "fcflags": raw_fcflags,
    }


def plant_fp_drift(reference_bld: Path) -> dict[str, Any]:
    """Plant a + 1.0e-30_wp perturbation in restart.f90, compile with real FCFLAGS, and prove census fails."""
    _, flag_tokens = _extract_makefile_fcflags(reference_bld)
    src_text = (reference_bld / "ppsrc" / "nemo" / "restart.f90").read_text(encoding="utf-8")
    target = "CALL iom_rstput( kt, nitrst, numrow, 'sshn', ssh(:,:        ,Kbb) )"
    replacement = "CALL iom_rstput( kt, nitrst, numrow, 'sshn', ssh(:,:        ,Kbb) + 1.0e-30_wp )"
    require(target in src_text, "target line for plant_fp_drift not found in restart.f90")
    planted_text = src_text.replace(target, replacement, 1)

    with tempfile.TemporaryDirectory(dir=_scratch_root()) as tmp:
        planted_src = Path(tmp) / "restart.f90"
        planted_obj = Path(tmp) / "restart.o"
        planted_src.write_text(planted_text, encoding="utf-8")
        cmd = [
            _tool_bin("mpif90"),
            "-c",
            *flag_tokens,
            f"-I{reference_bld / 'inc'}",
            f"-J{tmp}",
            str(planted_src),
            "-o",
            str(planted_obj),
        ]
        res = subprocess.run(cmd, env=_tool_env(), capture_output=True, text=True, check=False)
        require(res.returncode == 0,
                f"plant_fp_drift compilation failed: {res.stderr}")
        caught = False
        err = ""
        try:
            common.compare_fp_census(
                reference_bld / "obj" / "restart.o",
                planted_obj,
                function_pattern=r"rst_write",
            )
        except GateError as exc:
            caught = True
            err = str(exc)
        require(caught, "plant_fp_drift did not fail compare_fp_census on restart.o")
    return {"passed": True, "error": err}


def verify_same_program_no_observer(
    ppsrc_dir: Path,
    my_src_dir: Path | None = None,
    candidate_bld: Path | None = None,
) -> dict[str, Any]:
    """Checklist 7: verify MY_SRC is empty, ppsrc/nemo/*.f90 has zero observer hooks, and same_program_no_observer holds."""
    if my_src_dir is not None and my_src_dir.is_dir():
        my_src_files = sorted(
            p.name for p in my_src_dir.iterdir() if not p.name.startswith(".")
        )
        require(len(my_src_files) == 0,
                f"observer-free K0 build requires empty MY_SRC, found: {my_src_files}")
    else:
        my_src_files = []

    f90_files = sorted(ppsrc_dir.glob("*.f90"))
    require(len(f90_files) >= 400, f"expected >= 400 ppsrc files in {ppsrc_dir}, got {len(f90_files)}")
    for f90 in f90_files:
        text = f90.read_text(encoding="utf-8", errors="replace")
        for pat in OBSERVER_SYMBOL_PATTERNS:
            m = pat.search(text)
            if m is not None:
                raise GateError(
                    f"observer symbol {m.group(0)!r} found in {f90.name}"
                )

    bld_to_check = (
        candidate_bld
        if (candidate_bld is not None and (candidate_bld / "obj").is_dir())
        else (PRISTINE_K0_CFG / "BLD")
    )
    cfg_name = (
        TARGET_CFG_NAME
        if (candidate_bld is not None and (candidate_bld / "obj").is_dir())
        else "ORCA2_STANDALONE_IIF_OMIP_L4_R339K0"
    )
    sp = common.same_program_no_observer(
        bld_to_check,
        PRISTINE_K0_CFG / "BLD",
        new_cfg=cfg_name,
        reference_cfg="ORCA2_STANDALONE_IIF_OMIP_L4_R339K0",
    )
    return {
        "same_program_no_observer": True,
        "my_src_files": my_src_files,
        "ppsrc_file_count": len(f90_files),
        "whole_program_proof": sp,
    }


def plant_observer_contamination(ppsrc_dir: Path) -> dict[str, Any]:
    """Plant a non-blank observer call in stprk3.f90 and prove verify_same_program_no_observer fails."""
    with tempfile.TemporaryDirectory(dir=_scratch_root()) as tmp:
        fake_ppsrc = Path(tmp) / "ppsrc"
        fake_ppsrc.mkdir()
        for f90 in ppsrc_dir.glob("*.f90"):
            os.symlink(f90, fake_ppsrc / f90.name)
        stprk3_copy = fake_ppsrc / "stprk3.f90"
        stprk3_copy.unlink()
        stprk3_copy.write_text(
            (ppsrc_dir / "stprk3.f90").read_text(encoding="utf-8")
            + "\n! non-blank observer plant\nCALL r340_s4_surface_write(kstp)\n",
            encoding="utf-8",
        )
        caught = False
        error_msg = ""
        try:
            verify_same_program_no_observer(fake_ppsrc)
        except GateError as exc:
            caught = True
            error_msg = str(exc)
        require(caught, "plant_observer_contamination failed to catch r340_s4_surface_write")
    return {"passed": True, "error": error_msg}


def _load_r340_kt9_entry_rank(s4_dir: Path, rank: int) -> dict[str, np.ndarray]:
    rec_path = s4_dir / f"oracle_r340_state_k00000009_s0_r{rank:04d}.bin"
    require(rec_path.is_file(), f"missing admitted R340 kt=9 entry state: {rec_path}")
    decoded = r340_s4.read_exact_record(rec_path)
    require(decoded["kind"] == "state" and decoded["kt"] == 9 and decoded["kid"] == 0,
            f"unexpected header in {rec_path.name}")
    return decoded["fields"]


def verify_day1_against_admitted_s4_kt9(
    run_dir: Path,
    s4_dir: Path = R340_RUN_A,
) -> dict[str, Any]:
    """Checklist 5 & 10: verify Day 1 (kt=8) restart equals admitted R340 S4 kt=9 entry state to 0 ULP."""
    day1 = pilot_metrics.read_nemo_day(run_dir, 1)
    r0 = _load_r340_kt9_entry_rank(s4_dir, 0)
    r1 = _load_r340_kt9_entry_rank(s4_dir, 1)
    kt9_entry = {
        name: np.concatenate((r0[name], r1[name]), axis=1)
        for name in ("T", "S", "u", "v", "ssh")
    }
    summary: dict[str, Any] = {}
    for name in ("T", "S", "u", "v", "ssh"):
        a = day1[name]
        b = kt9_entry[name]
        require(a.shape == b.shape,
                f"day 1 restart {name} shape {a.shape} != R340 kt=9 entry {b.shape}")
        max_abs = float(np.max(np.abs(a - b)))
        require(max_abs == 0.0,
                f"day 1 restart {name} differs from admitted R340 kt=9 entry: max_abs={max_abs}")
        a_bits = (a + 0.0).view(np.uint64)
        b_bits = (b + 0.0).view(np.uint64)
        max_ulp = int(np.max(np.abs(a_bits.astype(np.int64) - b_bits.astype(np.int64))))
        require(max_ulp == 0,
                f"day 1 restart {name} ULP drift vs admitted R340 kt=9 entry: max_ulp={max_ulp}")
        summary[name] = {"shape": list(a.shape), "max_abs_diff": max_abs, "max_ulp": max_ulp}
    return summary


def verify_nemo_completion_signal(
    run_dir: Path,
    *,
    expected_steps: int | None = None,
) -> dict[str, Any]:
    """Verify NEMO's real completion signal under gfortran (Checklist 8 & 16).

    In `nemogcm.f90:236-240`, rank 0 (`lwm`) terminates with `STOP 0` when
    `nstop == 0` (else `STOP 123`), which gfortran writes to `stderr`
    (`mpirun.stderr.log`). In `stpctl.f90:106-107,159-162`, rank 0 writes the
    completed step index `kt` to `time.step` (`WRITE(numstp, '(1x, i8)') kt`).
    Completion therefore requires:
      1. `STOP 0` on `stderr` (`mpirun.stderr.log`), with no `STOP [1-9]` or `E R R O R`;
      2. `int(time.step) == nn_itend` (and `== expected_steps` when specified);
      3. no `"E R R O R"` or nonzero `STOP` in `ocean.output` (or `mpirun.stdout.log`).
    """
    namelist_cfg = run_dir / "namelist_cfg"
    require(namelist_cfg.is_file(), f"{run_dir.name}: missing namelist_cfg")
    nn_itend, nn_stock = parse_namelist_itend_stock(namelist_cfg)
    if expected_steps is not None:
        require(
            nn_itend == expected_steps and nn_stock == DAILY_STOCK_STRIDE,
            f"{run_dir.name}: namelist_cfg (nn_itend={nn_itend}, nn_stock={nn_stock}) "
            f"!= ({expected_steps}, {DAILY_STOCK_STRIDE})",
        )

    stderr_path = run_dir / "mpirun.stderr.log"
    require(stderr_path.is_file(), f"{run_dir.name}: missing mpirun.stderr.log")
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
    require(
        "E R R O R" not in stderr_text
        and re.search(r"\bSTOP\s+[1-9][0-9]*\b", stderr_text) is None,
        f"{run_dir.name}: mpirun.stderr.log contains runtime error or nonzero STOP",
    )
    require(
        re.search(r"(^|\s)STOP 0(\s|$)", stderr_text) is not None,
        f"{run_dir.name}: mpirun.stderr.log lacks clean STOP 0",
    )
    stderr_nonempty = [ln.strip() for ln in stderr_text.splitlines() if ln.strip()]
    require(
        bool(stderr_nonempty) and stderr_nonempty[-1] == "STOP 0",
        f"{run_dir.name}: mpirun.stderr.log does not end with 'STOP 0' (last={stderr_nonempty[-1] if stderr_nonempty else None!r})",
    )

    stdout_path = run_dir / "mpirun.stdout.log"
    stdout_bytes = 0
    if stdout_path.is_file():
        stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace")
        stdout_bytes = stdout_path.stat().st_size
        require(
            "E R R O R" not in stdout_text
            and re.search(r"\bSTOP\s+[1-9][0-9]*\b", stdout_text) is None,
            f"{run_dir.name}: mpirun.stdout.log contains runtime error marker",
        )

    timestep_path = run_dir / "time.step"
    require(timestep_path.is_file(), f"{run_dir.name}: missing time.step")
    ts_raw = timestep_path.read_text(encoding="utf-8", errors="replace").strip()
    require(ts_raw.isdigit(), f"{run_dir.name}: non-integer time.step content {ts_raw!r}")
    time_step_val = int(ts_raw)
    require(
        time_step_val == nn_itend,
        f"{run_dir.name}: time.step ({time_step_val}) != nn_itend ({nn_itend})",
    )

    ocean_out_path = run_dir / "ocean.output"
    require(ocean_out_path.is_file(), f"{run_dir.name}: missing ocean.output")
    ocean_text = ocean_out_path.read_text(encoding="utf-8", errors="replace")
    require(
        "E R R O R" not in ocean_text
        and re.search(r"\bSTOP\s+[1-9][0-9]*\b", ocean_text) is None,
        f"{run_dir.name}: ocean.output contains E R R O R or nonzero STOP",
    )

    return {
        "nn_itend": nn_itend,
        "nn_stock": nn_stock,
        "time_step": time_step_val,
        "stderr_stop_0": True,
        "stderr_last_line": stderr_nonempty[-1],
        "stdout_bytes": stdout_bytes,
        "ocean_output_error_free": True,
    }


def plant_completion_signal_failures(
    real_logs_dir: Path = REAL_ARM_A_LOGS_DIR,
) -> dict[str, Any]:
    """Verify the real Arm-A completion logs pass and prove planted missing STOP / short time.step / E R R O R fail."""
    require(real_logs_dir.is_dir(), f"missing real Arm-A log fixture dir: {real_logs_dir}")
    real_pass = verify_nemo_completion_signal(real_logs_dir, expected_steps=YEAR_STEPS)
    expected_resolved = {
        **r339_k0.RESOLVED,
        "nn_itend": str(YEAR_STEPS),
        "nn_stock": str(DAILY_STOCK_STRIDE),
    }
    common.validate_ocean_output(real_logs_dir / "ocean.output", expected_resolved)

    with tempfile.TemporaryDirectory(dir=_scratch_root()) as tmp:
        work = Path(tmp) / "completion_plant"
        shutil.copytree(real_logs_dir, work)

        # Plant 1: missing STOP 0 in mpirun.stderr.log
        orig_stderr = (work / "mpirun.stderr.log").read_text(encoding="utf-8")
        (work / "mpirun.stderr.log").write_text(
            "Note: The following floating-point exceptions are signalling: IEEE_UNDERFLOW_FLAG IEEE_DENORMAL\n",
            encoding="utf-8",
        )
        missing_stop_caught = False
        missing_stop_err = ""
        try:
            verify_nemo_completion_signal(work, expected_steps=YEAR_STEPS)
        except GateError as exc:
            missing_stop_caught = True
            missing_stop_err = str(exc)
        require(missing_stop_caught, "plant_completion_signal_failures did not refuse missing STOP 0 on stderr")

        # Plant 2: STOP 0 written to stdout instead of stderr must still refuse when stderr lacks STOP 0
        (work / "mpirun.stdout.log").write_text("STOP 0\n", encoding="utf-8")
        stdout_only_caught = False
        stdout_only_err = ""
        try:
            verify_nemo_completion_signal(work, expected_steps=YEAR_STEPS)
        except GateError as exc:
            stdout_only_caught = True
            stdout_only_err = str(exc)
        require(stdout_only_caught, "plant_completion_signal_failures did not refuse STOP 0 on stdout only")
        (work / "mpirun.stdout.log").write_text("", encoding="utf-8")

        # Plant 3: nonzero STOP 123 in mpirun.stderr.log
        (work / "mpirun.stderr.log").write_text("STOP 123\n", encoding="utf-8")
        stop123_caught = False
        stop123_err = ""
        try:
            verify_nemo_completion_signal(work, expected_steps=YEAR_STEPS)
        except GateError as exc:
            stop123_caught = True
            stop123_err = str(exc)
        require(stop123_caught, "plant_completion_signal_failures did not refuse STOP 123 on stderr")
        (work / "mpirun.stderr.log").write_text(orig_stderr, encoding="utf-8")

        # Plant 4: short time.step (e.g., 2912 < 2920)
        orig_ts = (work / "time.step").read_text(encoding="utf-8")
        (work / "time.step").write_text("     2912\n", encoding="utf-8")
        short_ts_caught = False
        short_ts_err = ""
        try:
            verify_nemo_completion_signal(work, expected_steps=YEAR_STEPS)
        except GateError as exc:
            short_ts_caught = True
            short_ts_err = str(exc)
        require(short_ts_caught, "plant_completion_signal_failures did not refuse short time.step")

        # Plant 5: missing time.step file
        (work / "time.step").unlink()
        missing_ts_caught = False
        missing_ts_err = ""
        try:
            verify_nemo_completion_signal(work, expected_steps=YEAR_STEPS)
        except GateError as exc:
            missing_ts_caught = True
            missing_ts_err = str(exc)
        require(missing_ts_caught, "plant_completion_signal_failures did not refuse missing time.step")
        (work / "time.step").write_text(orig_ts, encoding="utf-8")

        # Plant 6: E R R O R in ocean.output
        orig_ocean = (work / "ocean.output").read_text(encoding="utf-8")
        (work / "ocean.output").write_text(
            orig_ocean + "\n ===>>> : E R R O R\n",
            encoding="utf-8",
        )
        ocean_err_caught = False
        ocean_err_msg = ""
        try:
            verify_nemo_completion_signal(work, expected_steps=YEAR_STEPS)
        except GateError as exc:
            ocean_err_caught = True
            ocean_err_msg = str(exc)
        require(ocean_err_caught, "plant_completion_signal_failures did not refuse E R R O R in ocean.output")
        (work / "ocean.output").write_text(orig_ocean, encoding="utf-8")

    return {
        "passed": True,
        "real_arm_a_signal": real_pass,
        "missing_stop_error": missing_stop_err,
        "stdout_only_stop_error": stdout_only_err,
        "stop123_error": stop123_err,
        "short_timestep_error": short_ts_err,
        "missing_timestep_error": missing_ts_err,
        "ocean_output_error": ocean_err_msg,
    }


def stub_mpi_rank_from_env() -> int:
    """Return the two-rank launcher identity and refuse ambiguity or fallback."""
    values = {
        name: int(os.environ[name])
        for name in ("OMPI_COMM_WORLD_RANK", "PMIX_RANK", "PMI_RANK")
        if name in os.environ
    }
    require(bool(values),
            "stub binary requires OMPI_COMM_WORLD_RANK, PMIX_RANK, or PMI_RANK")
    require(len(set(values.values())) == 1,
            f"stub binary has conflicting launcher ranks: {values}")
    rank = next(iter(values.values()))
    require(rank in (0, 1), f"stub binary expects exactly ranks 0 and 1, got {rank}")
    return rank


def write_stub_s4_year_outputs(
    run_dir: Path,
    *,
    rank: int | None = None,
    first_restart_kt: int = DAILY_STOCK_STRIDE,
    plant_unproduced_file: bool = False,
    plant_missing_stop: bool = False,
    plant_short_timestep: int | None = None,
    sleep_seconds: int = 0,
) -> dict[str, Any]:
    """Checklist 12 & 16: write ONLY the files this observer-free K0 build can write.

    Reads `nn_itend` and `nn_stock` from `run_dir / "namelist_cfg"` so the stub
    replaces `./nemo` inside `run.sh`. Emits `time.step` (`stpctl.f90:160`) and
    writes `STOP 0` to `sys.stderr` (`nemogcm.f90:237` under gfortran) on rank
    zero only.  An MPI-launched invocation writes only its rank-local mesh and
    restart files; `rank=None` retains the serial test-helper behavior.
    """
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)
    namelist_cfg = run_dir / "namelist_cfg"
    require(namelist_cfg.is_file(), f"stub missing staged namelist_cfg: {namelist_cfg}")
    nn_itend, nn_stock = parse_namelist_itend_stock(namelist_cfg)
    require(nn_stock == DAILY_STOCK_STRIDE and nn_itend % nn_stock == 0,
            f"invalid stub namelist cadence: nn_itend={nn_itend}, nn_stock={nn_stock}")
    require(rank is None or rank in (0, 1), f"stub rank must be 0 or 1, got {rank}")
    require(first_restart_kt >= nn_stock and first_restart_kt % nn_stock == 0,
            f"invalid first stub restart step: {first_restart_kt}")
    ranks = (0, 1) if rank is None else (rank,)
    writes_shared = rank in (None, 0)

    if writes_shared:
        ocean_out = (R339_SMOKE_DIR / "ocean.output").read_text(encoding="utf-8")
        ocean_out = re.sub(
            r"(last\s+time\s+step\s+nn_itend\s*=\s*)\d+",
            rf"\g<1>{nn_itend}",
            ocean_out,
        )
        ocean_out = re.sub(
            r"(frequency\s+of\s+restart\s+file\s+nn_stock\s*=\s*)\d+",
            rf"\g<1>{nn_stock}",
            ocean_out,
        )
        (run_dir / "ocean.output").write_text(ocean_out, encoding="utf-8")

        namelist_dyn = (R339_SMOKE_DIR / "output.namelist.dyn").read_text(encoding="utf-8")
        namelist_dyn = re.sub(r"(\bnn_itend\s*=\s*)\d+", rf"\g<1>{nn_itend}", namelist_dyn)
        namelist_dyn = re.sub(r"(\bnn_stock\s*=\s*)\d+", rf"\g<1>{nn_stock}", namelist_dyn)
        (run_dir / "output.namelist.dyn").write_text(namelist_dyn, encoding="utf-8")

        step_val = int(plant_short_timestep) if plant_short_timestep is not None else nn_itend
        (run_dir / "time.step").write_text(f" {step_val:8d}\n", encoding="utf-8")
    else:
        step_val = int(plant_short_timestep) if plant_short_timestep is not None else nn_itend

    for local_rank in ranks:
        mesh_name = f"mesh_mask_{local_rank:04d}.nc"
        shutil.copy2(R339_SMOKE_DIR / mesh_name, run_dir / mesh_name)

    for local_rank in ranks:
        tmpl = (R339_RESTART_0, R339_RESTART_1)[local_rank]
        if first_restart_kt == nn_stock:
            # Day 1 (kt = 8): populate the cold-start arm's first saved state.
            dst_day1 = run_dir / f"ORCA2_{nn_stock:08d}_restart_{local_rank:04d}.nc"
            shutil.copy2(tmpl, dst_day1)
            fields = _load_r340_kt9_entry_rank(R340_RUN_A, local_rank)
            with netCDF4.Dataset(dst_day1, "r+") as ds:
                ds.variables["kt"][...] = float(nn_stock)
                for key, nc_var in (("T", "tn"), ("S", "sn"), ("u", "un"), ("v", "vn")):
                    arr = np.moveaxis(fields[key], -1, 0)
                    ds.variables[nc_var][0, :30, :, :] = arr
                    ds.variables[nc_var][0, 30, :, :] = 0.0
                ds.variables["sshn"][0, :, :] = fields["ssh"]

        for kt in range(max(first_restart_kt, 2 * nn_stock), nn_itend + 1, nn_stock):
            dst_kt = run_dir / f"ORCA2_{kt:08d}_restart_{local_rank:04d}.nc"
            shutil.copy2(tmpl, dst_kt)
            with netCDF4.Dataset(dst_kt, "r+") as ds:
                ds.variables["kt"][...] = float(kt)

    if writes_shared and (
        plant_unproduced_file or os.environ.get("R342_STUB_PLANT_UNPRODUCED") == "1"
    ):
        (run_dir / "oracle_r340_surface_kt0001_kid00_rank0000.bin").write_bytes(
            b"UNPRODUCED_OBSERVER_FILE"
        )

    if writes_shared and not plant_missing_stop:
        print("STOP 0", file=sys.stderr, flush=True)
    return {
        "rank": rank,
        "nn_itend": nn_itend,
        "nn_stock": nn_stock,
        "time_step": step_val,
        "first_restart_kt": first_restart_kt,
        "daily_restart_pairs": (nn_itend - first_restart_kt) // nn_stock + 1,
    }


def verify_no_unexpected_files(
    run_dir: Path,
    *,
    expected_steps: int = YEAR_STEPS,
) -> dict[str, Any]:
    """Verify run_dir contains NO observer .bin files and ONLY the expected daily restarts."""
    require(run_dir.is_dir(), f"missing run directory: {run_dir}")
    expected_restarts = {
        f"ORCA2_{kt:08d}_restart_{rank:04d}.nc"
        for kt in range(DAILY_STOCK_STRIDE, expected_steps + 1, DAILY_STOCK_STRIDE)
        for rank in (0, 1)
    }
    present_files = {p.name for p in run_dir.iterdir() if p.is_file() or p.is_symlink()}
    observer_bins = sorted(
        name for name in present_files
        if name.endswith(".bin") or name.startswith("oracle_")
    )
    require(not observer_bins,
            f"run directory {run_dir.name} contains unproduced observer file(s): {observer_bins}")

    present_restarts = {
        name for name in present_files
        if re.match(r"^ORCA2_\d+_restart_\d+\.nc$", name)
    }
    unexpected_restarts = sorted(present_restarts - expected_restarts)
    require(not unexpected_restarts,
            f"run directory {run_dir.name} contains unexpected restart file(s): {unexpected_restarts}")
    missing_restarts = sorted(expected_restarts - present_restarts)
    require(not missing_restarts,
            f"run directory {run_dir.name} is missing {len(missing_restarts)} daily restart file(s), "
            f"first={missing_restarts[0] if missing_restarts else None}")
    return {
        "present_file_count": len(present_files),
        "expected_restart_file_count": len(expected_restarts),
        "observer_bins": observer_bins,
    }


def _verify_single_restart_file(
    path: Path,
    expected_kt: int,
    expected_size: int,
) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(),
            f"restart file missing or symlink: {path}")
    sz = path.stat().st_size
    require(sz == expected_size,
            f"restart file {path.name} truncated or wrong size: {sz} != {expected_size}")
    with netCDF4.Dataset(path, "r") as ds:
        require("kt" in ds.variables, f"missing kt variable in {path.name}")
        kt_val = int(np.asarray(ds.variables["kt"][...]))
        require(kt_val == expected_kt,
                f"{path.name}: kt={kt_val} != expected {expected_kt}")
        numeric_vars = [
            v.name for v in ds.variables.values()
            if np.issubdtype(v.dtype, np.number)
        ]
        require(len(numeric_vars) >= 20,
                f"unexpectedly few numeric variables in {path.name}: {len(numeric_vars)}")
        for var in NONZERO_RESTART_VARS:
            require(var in numeric_vars,
                    f"missing prognostic restart variable {var!r} in {path.name}")
        for var in numeric_vars:
            arr = np.asarray(ds.variables[var][:], dtype=np.float64)
            require(bool(np.isfinite(arr).all()),
                    f"{path.name}:{var} contains non-finite values (NaN/Inf)")
            if var in NONZERO_RESTART_VARS:
                require(float(np.max(np.abs(arr))) > 0.0,
                        f"{path.name}:{var} is identically zero")
    return {
        "kt": expected_kt,
        "size_bytes": sz,
        "sha256": sha256_file(path),
    }


def admit_s4_year_single_arm(
    run_dir: Path,
    *,
    build_cfg: Path = DEFAULT_BUILD_CFG,
    expected_steps: int = YEAR_STEPS,
    day1_reference_run: Path | None = R340_RUN_A,
) -> dict[str, Any]:
    """Verify one completed K0 arm, optionally against the unperturbed Day-1 reference."""
    require(expected_steps >= DAILY_STOCK_STRIDE and expected_steps % DAILY_STOCK_STRIDE == 0,
            f"invalid expected_steps: {expected_steps}")
    ppsrc_dir = _ppsrc_dir(build_cfg)
    my_src_dir = (build_cfg / "MY_SRC") if (build_cfg / "MY_SRC").is_dir() else None
    cand_bld = (build_cfg / "BLD") if (build_cfg / "BLD" / "obj").is_dir() else None
    no_obs = verify_same_program_no_observer(ppsrc_dir, my_src_dir, cand_bld)
    inv = verify_no_unexpected_files(run_dir, expected_steps=expected_steps)

    expected_resolved = {
        **r339_k0.RESOLVED,
        "nn_itend": str(expected_steps),
        "nn_stock": str(DAILY_STOCK_STRIDE),
    }
    verify_namelist_groups_against_ppsrc(
        run_dir / "namelist_cfg",
        NAMELIST_REF,
        ppsrc_dir,
    )
    common.validate_ocean_output(run_dir / "ocean.output", expected_resolved)
    out_text = (run_dir / "ocean.output").read_text(encoding="utf-8", errors="replace")
    require(r339_k0.NO_ICEBERGS_BANNER in out_text,
            f"{run_dir.name}: ocean.output did not report No icebergs used")
    require("ice_cover_clim.nc" in out_text,
            f"{run_dir.name}: ocean.output did not resolve ice_cover_clim.nc")
    completion_signal = verify_nemo_completion_signal(run_dir, expected_steps=expected_steps)

    expected_size_0 = R339_RESTART_0.stat().st_size
    expected_size_1 = R339_RESTART_1.stat().st_size
    expected_kts = list(range(DAILY_STOCK_STRIDE, expected_steps + 1, DAILY_STOCK_STRIDE))

    first_day_summary = {}
    last_day_summary = {}
    for kt in expected_kts:
        for rank, exp_sz in ((0, expected_size_0), (1, expected_size_1)):
            fpath = run_dir / f"ORCA2_{kt:08d}_restart_{rank:04d}.nc"
            res = _verify_single_restart_file(fpath, kt, exp_sz)
            if kt == DAILY_STOCK_STRIDE:
                first_day_summary[f"rank_{rank:04d}"] = res
            if kt == expected_steps:
                last_day_summary[f"rank_{rank:04d}"] = res

    day1_vs_r340 = (
        verify_day1_against_admitted_s4_kt9(run_dir, day1_reference_run)
        if day1_reference_run is not None
        else None
    )
    return {
        "status": "PASS_R342_S4_YEAR_SINGLE_ARM",
        "admitted": True,
        "run_dir": str(run_dir),
        "inventory": inv,
        "completion_signal": completion_signal,
        "round": "orca2_r342",
        "stage": "S4_one_year_k0_single_arm_admission",
        "expected_steps": expected_steps,
        "daily_stock_stride": DAILY_STOCK_STRIDE,
        "daily_restart_pairs": len(expected_kts),
        "same_program_no_observer": no_obs,
        "day1_vs_r340_s4_kt9_entry": day1_vs_r340,
        "first_day_restarts": first_day_summary,
        "last_day_restarts": last_day_summary,
    }


def _compare_restart_pair_files(
    path_a: Path,
    path_b: Path,
    expected_kt: int,
    expected_size: int,
) -> dict[str, Any]:
    for p in (path_a, path_b):
        require(p.is_file() and not p.is_symlink(),
                f"restart file missing or symlink: {p}")
        sz = p.stat().st_size
        require(sz == expected_size,
                f"restart file {p.name} truncated or wrong size: {sz} != {expected_size}")
    with netCDF4.Dataset(path_a, "r") as da, netCDF4.Dataset(path_b, "r") as db:
        require("kt" in da.variables and "kt" in db.variables,
                f"missing kt variable in {path_a.name}")
        kt_a = int(np.asarray(da.variables["kt"][...]))
        kt_b = int(np.asarray(db.variables["kt"][...]))
        require(kt_a == expected_kt and kt_b == expected_kt,
                f"{path_a.name}: kt={kt_a}/{kt_b} != expected {expected_kt}")
        numeric_vars = [
            v.name for v in da.variables.values()
            if np.issubdtype(v.dtype, np.number)
        ]
        require(len(numeric_vars) >= 20,
                f"unexpectedly few numeric variables in {path_a.name}: {len(numeric_vars)}")
        for var in NONZERO_RESTART_VARS:
            require(var in numeric_vars, f"missing prognostic restart variable {var!r} in {path_a.name}")
        for var in numeric_vars:
            require(var in db.variables,
                    f"missing restart variable {var!r} in twin B {path_b.name}")
            va = np.asarray(da.variables[var][:], dtype=np.float64)
            vb = np.asarray(db.variables[var][:], dtype=np.float64)
            require(va.shape == vb.shape,
                    f"{path_a.name}:{var} shape mismatch {va.shape} vs {vb.shape}")
            require(bool(np.isfinite(va).all()) and bool(np.isfinite(vb).all()),
                    f"{path_a.name}:{var} contains non-finite values (NaN/Inf)")
            if var in NONZERO_RESTART_VARS:
                require(float(np.max(np.abs(va))) > 0.0,
                        f"{path_a.name}:{var} is identically zero")
            require(np.array_equal(va.view(np.uint64), vb.view(np.uint64)),
                    f"twin reproducibility ULP drift in {path_a.name}:{var}")
    return {
        "kt": expected_kt,
        "sha256_a": sha256_file(path_a),
        "sha256_b": sha256_file(path_b),
    }


def _write_and_verify_run_manifest(
    run_dir: Path,
    producer_commit: str | None = None,
) -> dict[str, Any]:
    if producer_commit is None:
        try:
            producer_commit = git_sha(allow_dirty=False)
        except RuntimeError as exc:
            raise GateError(f"cannot stamp run manifest on dirty tree: {exc}") from exc
    files = [
        p for p in run_dir.rglob("*")
        if p.is_file() and not p.is_symlink() and p.name != "producer_manifest.json"
    ]
    mpath = run_dir / "producer_manifest.json"
    common.write_manifest(
        mpath,
        producer_commit=producer_commit,
        binary=run_dir / "nemo",
        files=files,
        root=run_dir,
        build_source_root=run_dir / "build_sources",
    )
    return common.verify_manifest(mpath, expected_commit=producer_commit)


def _verify_existing_run_manifest(
    run_dir: Path,
    *,
    repo: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Checklist 9: verify existing producer_manifest.json commit ancestry, binary digest, and relative file digests."""
    mpath = run_dir / "producer_manifest.json"
    require(mpath.is_file(), f"missing producer_manifest.json in {run_dir}")
    payload = json.loads(mpath.read_text(encoding="utf-8"))
    producer_commit = str(payload.get("producer_commit", "")).strip()
    require(
        bool(re.fullmatch(r"[0-9a-f]{40}", producer_commit)),
        f"invalid or dirty producer_commit in {mpath}: {producer_commit!r}",
    )
    verified = common.verify_manifest(mpath, expected_commit=producer_commit)
    head_commit = git_sha(allow_dirty=False, repo=repo)
    anc = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", producer_commit, head_commit],
        capture_output=True,
        text=True,
        check=False,
    )
    require(
        anc.returncode == 0,
        f"manifest producer_commit {producer_commit} is not an ancestor of {head_commit}",
    )
    return verified


def admit_s4_year_twin_runs(
    run_a: Path,
    run_b: Path,
    *,
    build_cfg: Path = DEFAULT_BUILD_CFG,
    expected_steps: int = YEAR_STEPS,
    write_manifests: bool = True,
    producer_commit: str | None = None,
) -> dict[str, Any]:
    """Checklist 8, 9, 10, 11, 13: admit the two reproducibility twin runs A and B."""
    require(expected_steps >= DAILY_STOCK_STRIDE and expected_steps % DAILY_STOCK_STRIDE == 0,
            f"invalid expected_steps: {expected_steps}")
    ppsrc_dir = _ppsrc_dir(build_cfg)
    my_src_dir = (build_cfg / "MY_SRC") if (build_cfg / "MY_SRC").is_dir() else None
    cand_bld = (build_cfg / "BLD") if (build_cfg / "BLD" / "obj").is_dir() else None
    no_obs = verify_same_program_no_observer(ppsrc_dir, my_src_dir, cand_bld)

    inv_a = verify_no_unexpected_files(run_a, expected_steps=expected_steps)
    inv_b = verify_no_unexpected_files(run_b, expected_steps=expected_steps)

    expected_resolved = {
        **r339_k0.RESOLVED,
        "nn_itend": str(expected_steps),
        "nn_stock": str(DAILY_STOCK_STRIDE),
    }
    completion_signals: dict[str, Any] = {}
    for rdir in (run_a, run_b):
        verify_namelist_groups_against_ppsrc(
            rdir / "namelist_cfg",
            NAMELIST_REF,
            ppsrc_dir,
        )
        common.validate_ocean_output(rdir / "ocean.output", expected_resolved)
        out_text = (rdir / "ocean.output").read_text(encoding="utf-8", errors="replace")
        require(r339_k0.NO_ICEBERGS_BANNER in out_text,
                f"{rdir.name}: ocean.output did not report No icebergs used")
        require("ice_cover_clim.nc" in out_text,
                f"{rdir.name}: ocean.output did not resolve ice_cover_clim.nc")
        completion_signals[rdir.name] = verify_nemo_completion_signal(
            rdir, expected_steps=expected_steps
        )

    expected_size_0 = R339_RESTART_0.stat().st_size
    expected_size_1 = R339_RESTART_1.stat().st_size
    expected_kts = list(range(DAILY_STOCK_STRIDE, expected_steps + 1, DAILY_STOCK_STRIDE))

    first_day_summary = {}
    last_day_summary = {}
    for kt in expected_kts:
        for rank, exp_sz in ((0, expected_size_0), (1, expected_size_1)):
            fa = run_a / f"ORCA2_{kt:08d}_restart_{rank:04d}.nc"
            fb = run_b / f"ORCA2_{kt:08d}_restart_{rank:04d}.nc"
            res = _compare_restart_pair_files(fa, fb, kt, exp_sz)
            if kt == DAILY_STOCK_STRIDE:
                first_day_summary[f"rank_{rank:04d}"] = res
            if kt == expected_steps:
                last_day_summary[f"rank_{rank:04d}"] = res

    day1_vs_r340 = verify_day1_against_admitted_s4_kt9(run_a, R340_RUN_A)

    extra = {
        "round": "orca2_r342",
        "stage": "S4_one_year_k0_twin_acquisition",
        "expected_steps": expected_steps,
        "daily_stock_stride": DAILY_STOCK_STRIDE,
        "daily_restart_pairs": len(expected_kts),
        "completion_signals": completion_signals,
        "same_program_no_observer": no_obs,
        "day1_vs_r340_s4_kt9_entry": day1_vs_r340,
        "first_day_restarts": first_day_summary,
        "last_day_restarts": last_day_summary,
    }
    if write_manifests:
        try:
            commit = producer_commit if producer_commit is not None else git_sha(allow_dirty=False)
        except RuntimeError as exc:
            raise GateError(f"cannot write admission manifest on dirty tree: {exc}") from exc
        for rdir in (run_a, run_b):
            if (rdir / "producer_manifest.json").is_file():
                _verify_existing_run_manifest(rdir)
            else:
                _write_and_verify_run_manifest(rdir, commit)
        if run_a.parent == run_b.parent:
            (run_a.parent / "round342_s4_year_admission.json").write_text(
                json.dumps(
                    {
                        "status": "PASS_R342_S4_YEAR_NEMO_TWINS",
                        "admitted": True,
                        "run_a": str(run_a),
                        "run_b": str(run_b),
                        **extra,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

    return {
        "status": "PASS_R342_S4_YEAR_NEMO_TWINS",
        "admitted": True,
        "run_a": str(run_a),
        "run_b": str(run_b),
        "inventory_a": inv_a,
        "inventory_b": inv_b,
        **extra,
    }


def run_all_admission_plants(
    run_a: Path,
    run_b: Path,
    *,
    expected_steps: int,
) -> dict[str, Any]:
    """Checklist 10: plant NaN, 1-ULP twin drift, truncation, checksum mismatch, Day-1 mismatch, and unproduced file."""
    results: dict[str, Any] = {}

    # 1. NaN plant
    r0_a = run_a / f"ORCA2_{expected_steps:08d}_restart_0000.nc"
    backup = r0_a.with_suffix(".nc.bak")
    shutil.copy2(r0_a, backup)
    try:
        with netCDF4.Dataset(r0_a, "r+") as ds:
            arr = np.asarray(ds.variables["tn"][:], dtype=np.float64)
            arr[0, 0, 10, 10] = np.nan
            ds.variables["tn"][:] = arr
        caught = False
        err = ""
        try:
            admit_s4_year_twin_runs(run_a, run_b, expected_steps=expected_steps, write_manifests=False)
        except GateError as exc:
            caught = True
            err = str(exc)
        require(caught, "plant_nan did not fail admission")
        results["plant_nan"] = {"passed": True, "error": err}
    finally:
        shutil.move(backup, r0_a)

    # 2. 1-ULP twin drift plant
    shutil.copy2(r0_a, backup)
    try:
        with netCDF4.Dataset(r0_a, "r+") as ds:
            arr = np.asarray(ds.variables["sshn"][:], dtype=np.float64)
            arr[0, 20, 20] = np.nextafter(arr[0, 20, 20], np.inf)
            ds.variables["sshn"][:] = arr
        caught = False
        err = ""
        try:
            admit_s4_year_twin_runs(run_a, run_b, expected_steps=expected_steps, write_manifests=False)
        except GateError as exc:
            caught = True
            err = str(exc)
        require(caught, "plant_ulp did not fail admission")
        results["plant_ulp"] = {"passed": True, "error": err}
    finally:
        shutil.move(backup, r0_a)

    # 3. Truncation plant
    shutil.copy2(r0_a, backup)
    try:
        with r0_a.open("r+b") as fh:
            fh.truncate(r0_a.stat().st_size - 64)
        caught = False
        err = ""
        try:
            admit_s4_year_twin_runs(run_a, run_b, expected_steps=expected_steps, write_manifests=False)
        except GateError as exc:
            caught = True
            err = str(exc)
        require(caught, "plant_truncation did not fail admission")
        results["plant_truncation"] = {"passed": True, "error": err}
    finally:
        shutil.move(backup, r0_a)

    # 4. Day-1 vs R340 kt=9 entry mismatch plant
    day1_a = run_a / "ORCA2_00000008_restart_0000.nc"
    day1_b = run_b / "ORCA2_00000008_restart_0000.nc"
    bak_a = day1_a.with_suffix(".nc.bak")
    bak_b = day1_b.with_suffix(".nc.bak")
    shutil.copy2(day1_a, bak_a)
    shutil.copy2(day1_b, bak_b)
    try:
        for p in (day1_a, day1_b):
            with netCDF4.Dataset(p, "r+") as ds:
                arr = np.asarray(ds.variables["sshn"][:], dtype=np.float64)
                arr[0, 15, 15] = np.nextafter(arr[0, 15, 15], np.inf)
                ds.variables["sshn"][:] = arr
        caught = False
        err = ""
        try:
            admit_s4_year_twin_runs(run_a, run_b, expected_steps=expected_steps, write_manifests=False)
        except GateError as exc:
            caught = True
            err = str(exc)
        require(caught, "plant_day1_kt9_mismatch did not fail admission")
        results["plant_day1_kt9_mismatch"] = {"passed": True, "error": err}
    finally:
        shutil.move(bak_a, day1_a)
        shutil.move(bak_b, day1_b)

    # 5. Unproduced observer file plant
    unprod = run_a / "oracle_r340_surface_kt0001_kid00_rank0000.bin"
    unprod.write_bytes(b"UNPRODUCED")
    try:
        caught = False
        err = ""
        try:
            admit_s4_year_twin_runs(run_a, run_b, expected_steps=expected_steps, write_manifests=False)
        except GateError as exc:
            caught = True
            err = str(exc)
        require(caught, "plant_stub_unproduced_file did not fail admission")
        results["plant_stub_unproduced_file"] = {"passed": True, "error": err}
    finally:
        unprod.unlink(missing_ok=True)

    # 6. Manifest checksum plant
    mpath = run_a / "producer_manifest.json"
    mbak = mpath.with_suffix(".json.bak")
    shutil.copy2(mpath, mbak)
    try:
        mdata = json.loads(mpath.read_text(encoding="utf-8"))
        first_out = next(iter(mdata["files"]))
        mdata["files"][first_out] = "0" * 64
        mpath.write_text(json.dumps(mdata, indent=2), encoding="utf-8")
        caught = False
        err = ""
        try:
            common.verify_manifest(mpath)
        except GateError as exc:
            caught = True
            err = str(exc)
        require(caught, "plant_checksum did not fail verify_manifest")
        results["plant_checksum"] = {"passed": True, "error": err}
    finally:
        shutil.move(mbak, mpath)

    return results


def _append_admit_log(run_root: Path, entry: dict[str, Any]) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    log_path = run_root / "admit_existing.log"
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")


def _is_executed_run_dir(rdir: Path) -> bool:
    return (
        rdir.is_dir()
        and (
            (rdir / "ocean.output").is_file()
            or (rdir / "time.step").is_file()
            or any(rdir.glob("ORCA2_*_restart_*.nc"))
        )
    )


def _locate_retained_candidates_in_refused(
    refused_dir: Path,
) -> tuple[list[Path], list[Path]]:
    require(refused_dir.is_dir(), f"missing refused directory: {refused_dir}")
    cand_a = sorted(
        p for p in refused_dir.iterdir()
        if p.is_dir() and TWIN_A_NAME in p.name
    )
    cand_b = sorted(
        p for p in refused_dir.iterdir()
        if p.is_dir() and TWIN_B_NAME in p.name
    )
    return cand_a, cand_b


def admit_existing_with_quarantine(
    run_root_or_retained: Path,
    *,
    build_cfg: Path = DEFAULT_BUILD_CFG,
    expected_steps: int = YEAR_STEPS,
    evidence_dir: Path = DEFAULT_EVIDENCE_DIR,
) -> dict[str, Any]:
    """Checklist 9 & 13: --admit-existing (in-place and refused/ routes), pinning to producer_manifest.json.

    Supports:
      - `<run_root>/refused/<retained_arm_a_dir>` or `<run_root>` when only Arm A completed in `refused/`
        (promotes kept Arm A to `<run_root>/orca2_k0_year1_a_np2` without modifying any file inside it);
      - `<run_root>` when Arm A is already admitted at `<run_root>/orca2_k0_year1_a_np2` and Arm B is absent
        (verifies Arm A in place without modifying any file inside it);
      - `<run_root>` when both Twin A and Twin B are present in-place or in `refused/`.
    """
    explicit_retained: Path | None = None
    if (
        run_root_or_retained.is_dir()
        and run_root_or_retained.parent.name == "refused"
    ):
        explicit_retained = run_root_or_retained
        run_root = run_root_or_retained.parent.parent
    else:
        run_root = run_root_or_retained

    run_a = run_root / TWIN_A_NAME
    run_b = run_root / TWIN_B_NAME
    refused_dir = run_root / "refused"
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # Route 1a: In-place twin pair (run_a and run_b both exist)
    if run_a.is_dir() and run_b.is_dir() and explicit_retained is None:
        arm_a_already_admitted = (run_root / "round342_s4_year_arm_a_admission.json").is_file()
        try:
            manifest_a = _verify_existing_run_manifest(run_a)
            manifest_b = _verify_existing_run_manifest(run_b)
            require(
                manifest_a["binary_sha256"] == manifest_b["binary_sha256"],
                "in-place twin manifests have different binary_sha256 values",
            )
            res = admit_s4_year_twin_runs(
                run_a,
                run_b,
                build_cfg=build_cfg,
                expected_steps=expected_steps,
                write_manifests=True,
                producer_commit=manifest_b["producer_commit"],
            )
            res["admit_existing_route"] = "in_place"
            res["producer_commit_a"] = manifest_a["producer_commit"]
            res["producer_commit_b"] = manifest_b["producer_commit"]
            res["binary_sha256"] = manifest_a["binary_sha256"]
            _append_admit_log(
                run_root,
                {
                    "timestamp_utc": ts,
                    "route": "in_place",
                    "status": "PASS",
                    "producer_commit": manifest_b["producer_commit"],
                    "producer_commit_a": manifest_a["producer_commit"],
                    "producer_commit_b": manifest_b["producer_commit"],
                    "binary_sha256": manifest_a["binary_sha256"],
                    "run_a": str(run_a),
                    "run_b": str(run_b),
                },
            )
            return res
        except Exception as exc:
            refused_dir.mkdir(parents=True, exist_ok=True)
            quarantined: list[str] = []
            arms_to_quarantine = (run_b,) if arm_a_already_admitted else (run_a, run_b)
            for rdir in arms_to_quarantine:
                if rdir.exists():
                    dest = refused_dir / f"{ts}_{rdir.name}"
                    shutil.move(str(rdir), str(dest))
                    quarantined.append(str(dest))
            _append_admit_log(
                run_root,
                {
                    "timestamp_utc": ts,
                    "route": "in_place",
                    "status": "REFUSED_AND_QUARANTINED",
                    "error": str(exc),
                    "quarantined": quarantined,
                },
            )
            raise GateError(
                f"--admit-existing refused in-place twin runs ({exc}); quarantined to {quarantined}"
            ) from exc

    # Route 1b: In-place single Arm A (run_a exists, run_b does not exist yet)
    if run_a.is_dir() and not run_b.exists() and explicit_retained is None:
        try:
            manifest_a = _verify_existing_run_manifest(run_a)
            res = admit_s4_year_single_arm(
                run_a,
                build_cfg=build_cfg,
                expected_steps=expected_steps,
            )
            res["admit_existing_route"] = "in_place_single_arm_a"
            res["producer_commit"] = manifest_a["producer_commit"]
            res["binary_sha256"] = manifest_a["binary_sha256"]
            admission_line = (
                f"ORCA2_ROUND342_S4_YEAR_ARM_A_ADMITTED {run_a} "
                f"producer_commit={manifest_a['producer_commit']} "
                f"binary_sha256={manifest_a['binary_sha256']} "
                f"daily_restart_pairs={res['daily_restart_pairs']} "
                f"time_step={res['completion_signal']['time_step']}"
            )
            res["admission_line"] = admission_line
            (run_root / "round342_s4_year_arm_a_admission.json").write_text(
                json.dumps(res, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if run_root.resolve() == DEFAULT_RUN_ROOT.resolve():
                evidence_dir.mkdir(parents=True, exist_ok=True)
                (evidence_dir / "round342_s4_year_arm_a_admission.json").write_text(
                    json.dumps(res, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            _append_admit_log(
                run_root,
                {
                    "timestamp_utc": ts,
                    "route": "in_place_single_arm_a",
                    "status": "PASS_ARM_A_ADMITTED",
                    "producer_commit": manifest_a["producer_commit"],
                    "binary_sha256": manifest_a["binary_sha256"],
                    "run_a": str(run_a),
                },
            )
            print(admission_line)
            return res
        except Exception as exc:
            refused_dir.mkdir(parents=True, exist_ok=True)
            dest = refused_dir / f"{ts}_{run_a.name}"
            shutil.move(str(run_a), str(dest))
            _append_admit_log(
                run_root,
                {
                    "timestamp_utc": ts,
                    "route": "in_place_single_arm_a",
                    "status": "REFUSED_AND_QUARANTINED",
                    "error": str(exc),
                    "quarantined": [str(dest)],
                },
            )
            raise GateError(
                f"--admit-existing refused in-place Arm A ({exc}); quarantined to {[str(dest)]}"
            ) from exc

    require(
        not run_a.exists() and not run_b.exists(),
        f"incomplete or unexpected in-place state under {run_root}: "
        f"run_a={run_a.exists()}, run_b={run_b.exists()}",
    )

    cand_a, cand_b = _locate_retained_candidates_in_refused(refused_dir)
    ret_a = explicit_retained if (explicit_retained is not None and TWIN_A_NAME in explicit_retained.name) else (cand_a[-1] if cand_a else None)
    exec_b = [p for p in cand_b if _is_executed_run_dir(p)]
    ret_b = exec_b[-1] if exec_b else (cand_b[-1] if cand_b else None)

    # Route 2a: Single completed Arm A in refused/ (when explicit_retained is Arm A or ret_b was never executed)
    if ret_a is not None and (explicit_retained is not None or ret_b is None or not _is_executed_run_dir(ret_b)):
        try:
            require(
                ret_a.resolve().is_relative_to(refused_dir.resolve()),
                f"retained Arm A directory {ret_a} is not inside {refused_dir}",
            )
            manifest_a = _verify_existing_run_manifest(ret_a)
            res = admit_s4_year_single_arm(
                ret_a,
                build_cfg=build_cfg,
                expected_steps=expected_steps,
            )
            require(not run_a.exists(), f"canonical Arm A path already exists: {run_a}")
            shutil.move(str(ret_a), str(run_a))
            _verify_existing_run_manifest(run_a)
            res["run_dir"] = str(run_a)
            res["run_a"] = str(run_a)
            res["admit_existing_route"] = "refused_single_arm_a"
            res["promoted_from_refused"] = str(ret_a)
            res["producer_commit"] = manifest_a["producer_commit"]
            res["binary_sha256"] = manifest_a["binary_sha256"]
            admission_line = (
                f"ORCA2_ROUND342_S4_YEAR_ARM_A_ADMITTED {run_a} "
                f"producer_commit={manifest_a['producer_commit']} "
                f"binary_sha256={manifest_a['binary_sha256']} "
                f"daily_restart_pairs={res['daily_restart_pairs']} "
                f"time_step={res['completion_signal']['time_step']}"
            )
            res["admission_line"] = admission_line
            (run_root / "round342_s4_year_arm_a_admission.json").write_text(
                json.dumps(res, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if run_root.resolve() == DEFAULT_RUN_ROOT.resolve():
                evidence_dir.mkdir(parents=True, exist_ok=True)
                (evidence_dir / "round342_s4_year_arm_a_admission.json").write_text(
                    json.dumps(res, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            _append_admit_log(
                run_root,
                {
                    "timestamp_utc": ts,
                    "route": "refused",
                    "subroute": "single_arm_a",
                    "status": "PASS",
                    "producer_commit": manifest_a["producer_commit"],
                    "binary_sha256": manifest_a["binary_sha256"],
                    "promoted_from": [str(ret_a)],
                    "run_a": str(run_a),
                },
            )
            print(admission_line)
            return res
        except Exception as exc:
            _append_admit_log(
                run_root,
                {
                    "timestamp_utc": ts,
                    "route": "refused",
                    "subroute": "single_arm_a",
                    "status": "REFUSED_RETAINED_IN_PLACE",
                    "error": str(exc),
                    "retained_a": str(ret_a),
                },
            )
            raise GateError(
                f"--admit-existing refused retained Arm A in {refused_dir} ({exc}); kept in refused/"
            ) from exc

    # Route 2b: Retained twin pair from refused/ (kept inside refused/ on failure; promoted on pass)
    require(
        ret_a is not None and ret_b is not None,
        f"no retained twin pair ({TWIN_A_NAME}, {TWIN_B_NAME}) found inside {refused_dir}",
    )
    try:
        require(
            ret_a.resolve().is_relative_to(refused_dir.resolve())
            and ret_b.resolve().is_relative_to(refused_dir.resolve()),
            "retained twin directories are not inside refused/",
        )
        manifest_a = _verify_existing_run_manifest(ret_a)
        manifest_b = _verify_existing_run_manifest(ret_b)
        require(
            manifest_a["binary_sha256"] == manifest_b["binary_sha256"],
            "retained twin manifests have different binary_sha256 values",
        )
        res = admit_s4_year_twin_runs(
            ret_a,
            ret_b,
            build_cfg=build_cfg,
            expected_steps=expected_steps,
            write_manifests=True,
            producer_commit=manifest_b["producer_commit"],
        )
        require(not run_a.exists() and not run_b.exists(),
                f"canonical twin paths already exist in {run_root}")
        shutil.move(str(ret_a), str(run_a))
        shutil.move(str(ret_b), str(run_b))
        _verify_existing_run_manifest(run_a)
        _verify_existing_run_manifest(run_b)
        res["run_a"] = str(run_a)
        res["run_b"] = str(run_b)
        res["admit_existing_route"] = "refused"
        res["promoted_from_refused"] = [str(ret_a), str(ret_b)]
        _append_admit_log(
            run_root,
            {
                "timestamp_utc": ts,
                "route": "refused",
                "status": "PASS",
                "producer_commit": manifest_b["producer_commit"],
                "producer_commit_a": manifest_a["producer_commit"],
                "producer_commit_b": manifest_b["producer_commit"],
                "binary_sha256": manifest_a["binary_sha256"],
                "promoted_from": [str(ret_a), str(ret_b)],
                "run_a": str(run_a),
                "run_b": str(run_b),
            },
        )
        return res
    except Exception as exc:
        _append_admit_log(
            run_root,
            {
                "timestamp_utc": ts,
                "route": "refused",
                "status": "REFUSED_RETAINED_IN_PLACE",
                "error": str(exc),
                "retained_a": str(ret_a),
                "retained_b": str(ret_b),
            },
        )
        raise GateError(
            f"--admit-existing refused retained runs in {refused_dir} ({exc}); kept in refused/"
        ) from exc


def run_preflight(
    *,
    evidence_dir: Path = DEFAULT_EVIDENCE_DIR,
    build_cfg: Path = DEFAULT_BUILD_CFG,
    run_root: Path = DEFAULT_RUN_ROOT,
    ice_cover_path: Path = DEFAULT_ICE_COVER_PATH,
) -> dict[str, Any]:
    """Execute all Checklist 1-16 preflight checks and plants without running NEMO."""
    evidence_dir.mkdir(parents=True, exist_ok=True)
    run_sh = (
        Path(__file__).resolve().parent
        / "nemo_testcase_l4_orca2_round342_s4_year_acquisition"
        / "run.sh"
    )
    require(run_sh.is_file(), f"missing acquisition run.sh: {run_sh}")

    worktree_check = verify_clean_worktree(REPO_ROOT)
    dirty_plant = plant_dirty_worktree(run_sh)
    disk_check = verify_disk_space(evidence_dir)
    disk_plant = plant_disk_space_exhaustion(evidence_dir, run_sh)
    ram_check = verify_available_ram()
    ram_plant = plant_ram_exhaustion(run_sh)
    syntax_check = verify_run_sh_syntax_and_plant(run_sh)

    ppsrc_dir = _ppsrc_dir(build_cfg)
    prereq = r340_s4.verify_round339_prerequisites(
        ice_cover_path.parent, R339_SMOKE_DIR
    )
    alloc_check = verify_allocate_extents(ppsrc_dir)
    alloc_plant = plant_allocate_mismatch(ppsrc_dir)
    fixture_check = verify_restart_fortran_fixture(ppsrc_dir)
    fixture_plant = plant_fixture_wrong_time_level(ppsrc_dir)
    fp_check = verify_real_flag_fp_census(
        PRISTINE_K0_CFG / "BLD",
        (build_cfg / "BLD") if (build_cfg / "BLD" / "obj").is_dir() else None,
    )
    fp_plant = plant_fp_drift(PRISTINE_K0_CFG / "BLD")
    obs_check = verify_same_program_no_observer(
        ppsrc_dir,
        (build_cfg / "MY_SRC") if (build_cfg / "MY_SRC").is_dir() else None,
        (build_cfg / "BLD") if (build_cfg / "BLD" / "obj").is_dir() else None,
    )
    obs_plant = plant_observer_contamination(ppsrc_dir)
    nml_plant = plant_namelist_error(ppsrc_dir)
    forcing_plant = plant_missing_forcing(ice_cover_path)
    completion_signal_plant = plant_completion_signal_failures(REAL_ARM_A_LOGS_DIR)

    with tempfile.TemporaryDirectory(dir=_scratch_root()) as tmp:
        # 1. Stub pipeline pass
        stub_root = Path(tmp) / "stub_s4_year"
        cmd = [
            "bash",
            str(run_sh),
            "--stub-run-dir",
            str(stub_root),
            "--stub-steps",
            "16",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        require(res.returncode == 0,
                f"run.sh --stub-run-dir failed (rc={res.returncode}):\n{res.stdout}\n{res.stderr}")
        run_a = stub_root / TWIN_A_NAME
        run_b = stub_root / TWIN_B_NAME
        plants = run_all_admission_plants(run_a, run_b, expected_steps=16)

        # 2. Existing-arm overwrite refusal plant (Checklist 13: never rm -rf an existing arm)
        overwrite_res = subprocess.run(
            ["bash", str(run_sh), "--stub-run-dir", str(stub_root), "--stub-steps", "16"],
            capture_output=True,
            text=True,
            check=False,
        )
        require(
            overwrite_res.returncode != 0
            and "REFUSE: retained arm directory already exists" in overwrite_res.stderr,
            "run.sh --stub-run-dir did not refuse when retained arm directory already existed",
        )
        require(run_a.is_dir() and run_b.is_dir(),
                "existing twin directories were deleted during overwrite refusal check")

        # 3. In-place --admit-existing pass route
        admit_ok = subprocess.run(
            ["bash", str(run_sh), "--admit-existing", str(stub_root), "--stub-steps", "16"],
            capture_output=True,
            text=True,
            check=False,
        )
        require(admit_ok.returncode == 0,
                f"run.sh --admit-existing pass route failed:\n{admit_ok.stdout}\n{admit_ok.stderr}")
        admit_log = (stub_root / "admit_existing.log").read_text(encoding="utf-8")
        require('"route": "in_place"' in admit_log and '"status": "PASS"' in admit_log,
                f"admit_existing.log missing in_place PASS record: {admit_log}")

        # 4a. Arm-A staging failure quarantine + clean re-run (failure after mkdir arm_a routes to refused/)
        stub_stage_bad = Path(tmp) / "stub_s4_year_stage_bad"
        missing_ice = Path(tmp) / "missing_ice_cover_clim.nc"
        stage_bad_res = subprocess.run(
            [
                "bash",
                str(run_sh),
                "--stub-run-dir",
                str(stub_stage_bad),
                "--stub-steps",
                "16",
                "--ice-cover",
                str(missing_ice),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        require(
            stage_bad_res.returncode != 0
            and "R342_REFUSED_ARM_QUARANTINED" in stage_bad_res.stderr,
            f"arm-A staging failure plant did not fail and quarantine partial arm:\n{stage_bad_res.stderr}",
        )
        stage_refused_items = (
            sorted((stub_stage_bad / "refused").glob("*"))
            if (stub_stage_bad / "refused").is_dir()
            else []
        )
        require(
            len(stage_refused_items) == 1
            and TWIN_A_NAME in stage_refused_items[0].name
            and stage_refused_items[0].is_dir()
            and any(stage_refused_items[0].iterdir()),
            f"expected 1 non-empty quarantined partial arm-A dir in {stub_stage_bad / 'refused'}, found {stage_refused_items}",
        )
        require(
            not (stub_stage_bad / TWIN_A_NAME).exists()
            and not (stub_stage_bad / TWIN_B_NAME).exists(),
            "partial arm-A dir still present at canonical path after staging failure",
        )
        stage_rerun_res = subprocess.run(
            ["bash", str(run_sh), "--stub-run-dir", str(stub_stage_bad), "--stub-steps", "16"],
            capture_output=True,
            text=True,
            check=False,
        )
        require(
            stage_rerun_res.returncode == 0,
            f"re-run after arm-A staging failure quarantine failed:\n{stage_rerun_res.stdout}\n{stage_rerun_res.stderr}",
        )
        require(
            stage_refused_items[0].is_dir()
            and (stub_stage_bad / TWIN_A_NAME).is_dir()
            and (stub_stage_bad / TWIN_B_NAME).is_dir(),
            "re-run after arm-A staging quarantine did not preserve refused/ partial arm alongside new admitted twins",
        )

        # 4b. Failed-arm quarantine + clean re-run without deleting quarantined runs
        stub_bad = Path(tmp) / "stub_s4_year_bad"
        bad_res = subprocess.run(
            [
                "bash",
                str(run_sh),
                "--stub-run-dir",
                str(stub_bad),
                "--stub-steps",
                "16",
                "--plant-stub-unproduced-file",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        require(bad_res.returncode != 0,
                "run.sh --stub-run-dir --plant-stub-unproduced-file did not fail")
        refused_items = sorted((stub_bad / "refused").glob("*")) if (stub_bad / "refused").is_dir() else []
        require(len(refused_items) == 2,
                f"expected 2 quarantined twin dirs in {stub_bad / 'refused'}, found {refused_items}")
        require(not (stub_bad / TWIN_A_NAME).exists() and not (stub_bad / TWIN_B_NAME).exists(),
                "quarantined twin dirs still present at canonical paths after refusal")

        rerun_res = subprocess.run(
            ["bash", str(run_sh), "--stub-run-dir", str(stub_bad), "--stub-steps", "16"],
            capture_output=True,
            text=True,
            check=False,
        )
        require(
            rerun_res.returncode == 0,
            f"re-run after quarantine failed:\n{rerun_res.stdout}\n{rerun_res.stderr}",
        )
        require(
            all(p.is_dir() for p in refused_items)
            and (stub_bad / TWIN_A_NAME).is_dir()
            and (stub_bad / TWIN_B_NAME).is_dir(),
            "re-run after quarantine did not preserve refused/ dirs alongside new admitted twins",
        )

        # 4c. Missing STOP 0 on stderr plant in run.sh
        stub_missing_stop = Path(tmp) / "stub_s4_year_missing_stop"
        missing_stop_res = subprocess.run(
            [
                "bash",
                str(run_sh),
                "--stub-run-dir",
                str(stub_missing_stop),
                "--stub-steps",
                "16",
                "--plant-stub-missing-stop",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        require(
            missing_stop_res.returncode != 0
            and "REFUSE: missing clean STOP 0" in missing_stop_res.stderr
            and "mpirun.stderr.log" in missing_stop_res.stderr,
            f"missing STOP 0 plant in run.sh did not fail as expected:\n{missing_stop_res.stderr}",
        )

        # 4d. Short time.step plant in run.sh
        stub_short_ts = Path(tmp) / "stub_s4_year_short_ts"
        short_ts_res = subprocess.run(
            [
                "bash",
                str(run_sh),
                "--stub-run-dir",
                str(stub_short_ts),
                "--stub-steps",
                "16",
                "--plant-stub-short-timestep",
                "8",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        require(
            short_ts_res.returncode != 0
            and "REFUSE: short or mismatched time.step" in short_ts_res.stderr,
            f"short time.step plant in run.sh did not fail as expected:\n{short_ts_res.stderr}",
        )

        # 5. timeout --foreground plant: stub sleeps 5s with 1s timeout; arm is quarantined to refused/
        stub_timeout = Path(tmp) / "stub_s4_year_timeout"
        timeout_res = subprocess.run(
            [
                "bash",
                str(run_sh),
                "--stub-run-dir",
                str(stub_timeout),
                "--stub-steps",
                "16",
                "--stub-timeout",
                "1",
                "--plant-stub-sleep",
                "5",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        require(
            timeout_res.returncode != 0
            and "R342_REFUSED_ARM_QUARANTINED" in timeout_res.stderr,
            f"timeout --foreground plant did not fail and quarantine arm:\n{timeout_res.stderr}",
        )
        timeout_refused = list((stub_timeout / "refused").glob("*")) if (stub_timeout / "refused").is_dir() else []
        require(
            len(timeout_refused) == 2,
            f"expected 2 quarantined dirs in {stub_timeout / 'refused'} after timeout, got {timeout_refused}",
        )

        # 6. refused/ -> --admit-existing route (with checksum plant inside refused/ + promotion on pass)
        stub_refused_route = Path(tmp) / "stub_s4_year_refused_route"
        (stub_refused_route / "refused").mkdir(parents=True, exist_ok=True)
        ret_a = stub_refused_route / "refused" / f"20261027T000000Z_{TWIN_A_NAME}"
        ret_b = stub_refused_route / "refused" / f"20261027T000000Z_{TWIN_B_NAME}"
        shutil.copytree(run_a, ret_a, symlinks=True)
        shutil.copytree(run_b, ret_b, symlinks=True)

        # Plant binary corruption in ret_a/nemo and prove --admit-existing from refused/ fails and keeps ret_a/ret_b
        nemo_bak = ret_a / "nemo.bak"
        shutil.copy2(ret_a / "nemo", nemo_bak)
        (ret_a / "nemo").write_text("#!/usr/bin/env bash\n# planted binary modification\n", encoding="utf-8")
        refused_plant_res = subprocess.run(
            ["bash", str(run_sh), "--admit-existing", str(stub_refused_route), "--stub-steps", "16"],
            capture_output=True,
            text=True,
            check=False,
        )
        require(
            refused_plant_res.returncode != 0 and ret_a.is_dir() and ret_b.is_dir(),
            "refused/ -> --admit-existing binary checksum plant did not fail or lost retained dirs",
        )
        shutil.move(str(nemo_bak), str(ret_a / "nemo"))

        # Now run refused/ -> --admit-existing on the clean retained pair and verify promotion
        refused_pass_res = subprocess.run(
            ["bash", str(run_sh), "--admit-existing", str(stub_refused_route), "--stub-steps", "16"],
            capture_output=True,
            text=True,
            check=False,
        )
        require(
            refused_pass_res.returncode == 0,
            f"refused/ -> --admit-existing pass route failed:\n{refused_pass_res.stdout}\n{refused_pass_res.stderr}",
        )
        require(
            (stub_refused_route / TWIN_A_NAME).is_dir()
            and (stub_refused_route / TWIN_B_NAME).is_dir()
            and not ret_a.exists()
            and not ret_b.exists(),
            "refused/ -> --admit-existing did not promote retained twins to canonical paths",
        )
        refused_log = (stub_refused_route / "admit_existing.log").read_text(encoding="utf-8")
        require(
            '"route": "refused"' in refused_log
            and '"status": "REFUSED_RETAINED_IN_PLACE"' in refused_log
            and '"status": "PASS"' in refused_log,
            f"admit_existing.log missing refused route entries: {refused_log}",
        )

        # 7. Kept Arm A in refused/ (with unrun staged Arm B in refused/) -> --admit-existing promotes Arm A -> resume Arm B only
        stub_arm_a_only = Path(tmp) / "stub_s4_year_arm_a_only"
        (stub_arm_a_only / "refused").mkdir(parents=True, exist_ok=True)
        ret_single_a = stub_arm_a_only / "refused" / f"20261027T231902Z_{TWIN_A_NAME}.failed.1"
        ret_unrun_b = stub_arm_a_only / "refused" / f"20261027T231902Z_{TWIN_B_NAME}.failed.1"
        shutil.copytree(run_a, ret_single_a, symlinks=True)
        stage_s4_year_run_dir(ret_unrun_b, ice_cover_path, build_cfg=build_cfg, nn_itend=16, nn_stock=8)
        shutil.copy2(run_a / "nemo", ret_unrun_b / "nemo")
        _write_and_verify_run_manifest(ret_unrun_b)

        admit_arm_a_res = subprocess.run(
            ["bash", str(run_sh), "--admit-existing", str(ret_single_a), "--stub-steps", "16"],
            capture_output=True,
            text=True,
            check=False,
        )
        require(
            admit_arm_a_res.returncode == 0
            and "ORCA2_ROUND342_S4_YEAR_ARM_A_ADMITTED" in admit_arm_a_res.stdout
            and (stub_arm_a_only / TWIN_A_NAME).is_dir()
            and not ret_single_a.exists()
            and ret_unrun_b.is_dir(),
            f"single Arm A promotion from refused/ failed:\n{admit_arm_a_res.stdout}\n{admit_arm_a_res.stderr}",
        )

        resume_b_res = subprocess.run(
            ["bash", str(run_sh), "--stub-run-dir", str(stub_arm_a_only), "--stub-steps", "16"],
            capture_output=True,
            text=True,
            check=False,
        )
        require(
            resume_b_res.returncode == 0
            and "R342_RESUMING_ARM_B_ONLY" in resume_b_res.stdout
            and (stub_arm_a_only / TWIN_A_NAME).is_dir()
            and (stub_arm_a_only / TWIN_B_NAME).is_dir(),
            f"resuming Arm B only after Arm A admission failed:\n{resume_b_res.stdout}\n{resume_b_res.stderr}",
        )

    report = {
        "status": "ORCA2_ROUND342_S4_YEAR_PREFLIGHT_READY",
        "round": "orca2_r342",
        "target_cfg": TARGET_CFG_NAME,
        "build_cfg": str(build_cfg),
        "run_root": str(run_root),
        "twin_a": str(run_root / TWIN_A_NAME),
        "twin_b": str(run_root / TWIN_B_NAME),
        "year_steps": YEAR_STEPS,
        "daily_stock_stride": DAILY_STOCK_STRIDE,
        "expected_daily_restarts": len(EXPECTED_DAILY_KTS),
        "per_arm_timeout_seconds": PER_ARM_TIMEOUT_SECONDS,
        "disk_space_check": disk_check,
        "disk_space_plant": disk_plant,
        "ram_check": ram_check,
        "ram_plant": ram_plant,
        "dirty_worktree_plant": dirty_plant,
        "run_sh_syntax_and_plant": syntax_check,
        "timeout_foreground_plant": {
            "passed": True,
            "rc": timeout_res.returncode,
            "quarantined_count": len(timeout_refused),
        },
        "stage_failure_plant": {
            "passed": True,
            "rc": stage_bad_res.returncode,
            "quarantined_count": len(stage_refused_items),
            "rerun_rc": stage_rerun_res.returncode,
        },
        "completion_signal_and_plants": completion_signal_plant,
        "shell_completion_plants": {
            "missing_stop_rc": missing_stop_res.returncode,
            "short_timestep_rc": short_ts_res.returncode,
        },
        "prerequisites": prereq,
        "allocate_extents": alloc_check,
        "allocate_plant": alloc_plant,
        "fortran_fixture": fixture_check,
        "fortran_fixture_plant": fixture_plant,
        "real_flag_fp_census": fp_check,
        "fp_drift_plant": fp_plant,
        "same_program_no_observer": obs_check,
        "observer_contamination_plant": obs_plant,
        "namelist_plant": nml_plant,
        "missing_forcing_plant": forcing_plant,
        "admission_plants": plants,
        "admit_existing_routes": {
            "in_place_pass_verified": True,
            "existing_arm_overwrite_refused": True,
            "stage_failure_quarantine_and_rerun_verified": True,
            "refused_quarantine_and_rerun_verified": True,
            "refused_to_admit_existing_checksum_plant_verified": True,
            "refused_to_admit_existing_promotion_verified": True,
            "refused_single_arm_a_promotion_and_arm_b_resume_verified": True,
        },
        "worktree": worktree_check,
    }
    out_path = evidence_dir / "round342_s4_year_preflight.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--stage-run-dir", type=Path, default=None)
    parser.add_argument("--write-run-manifest", type=Path, default=None)
    parser.add_argument("--act-as-nemo-binary", action="store_true")
    parser.add_argument("--plant-stub-unproduced-file", action="store_true")
    parser.add_argument("--plant-stub-missing-stop", action="store_true")
    parser.add_argument("--plant-stub-short-timestep", type=int, default=None)
    parser.add_argument("--plant-stub-sleep", type=int, default=0)
    parser.add_argument("--admit-twin-runs", type=Path, default=None)
    parser.add_argument("--admit-existing", type=Path, default=None)
    parser.add_argument("--expected-steps", type=int, default=YEAR_STEPS)
    parser.add_argument("--ice-cover", type=Path, default=DEFAULT_ICE_COVER_PATH)
    parser.add_argument("--build-cfg", type=Path, default=DEFAULT_BUILD_CFG)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    args = parser.parse_args(argv)

    if args.act_as_nemo_binary:
        write_stub_s4_year_outputs(
            Path.cwd(),
            rank=stub_mpi_rank_from_env(),
            plant_unproduced_file=args.plant_stub_unproduced_file,
            plant_missing_stop=args.plant_stub_missing_stop,
            plant_short_timestep=args.plant_stub_short_timestep,
            sleep_seconds=args.plant_stub_sleep,
        )
        return 0

    if args.write_run_manifest is not None:
        manifest = _write_and_verify_run_manifest(args.write_run_manifest)
        print(json.dumps({"producer_commit": manifest["producer_commit"], "binary_sha256": manifest["binary_sha256"]}))
        return 0

    if args.stage_run_dir is not None:
        info = stage_s4_year_run_dir(
            args.stage_run_dir,
            args.ice_cover,
            build_cfg=args.build_cfg,
            nn_itend=args.expected_steps,
            nn_stock=DAILY_STOCK_STRIDE,
        )
        print(json.dumps(info, indent=2, sort_keys=True))
        return 0

    if args.admit_twin_runs is not None:
        res = admit_s4_year_twin_runs(
            args.admit_twin_runs / TWIN_A_NAME,
            args.admit_twin_runs / TWIN_B_NAME,
            build_cfg=args.build_cfg,
            expected_steps=args.expected_steps,
            write_manifests=True,
        )
        print(json.dumps(res, indent=2, sort_keys=True))
        return 0

    if args.admit_existing is not None:
        res = admit_existing_with_quarantine(
            args.admit_existing,
            build_cfg=args.build_cfg,
            expected_steps=args.expected_steps,
            evidence_dir=args.evidence_dir,
        )
        print(json.dumps(res, indent=2, sort_keys=True))
        return 0

    report = run_preflight(
        evidence_dir=args.evidence_dir,
        build_cfg=args.build_cfg,
        ice_cover_path=args.ice_cover,
    )
    print(json.dumps({"status": report["status"], "twin_a": report["twin_a"], "twin_b": report["twin_b"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
