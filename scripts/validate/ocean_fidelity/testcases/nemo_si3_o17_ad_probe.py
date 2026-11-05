#!/usr/bin/env python3
"""CPU-fp64 value-and-grad probe and prefix/slice/pass compile diagnostics for O17."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time
from typing import Any, Iterable, Sequence


def _preconfigure_xla_logging(argv: list[str]) -> Path | None:
    """Enable C++ XLA pass/compiler VLOG before jaxlib initializes if --xla-pass-log is set."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--xla-pass-log", type=Path)
    parser.add_argument("--vmodule", type=str, default=None)
    known, _ = parser.parse_known_args(argv)
    if known.xla_pass_log is None:
        return None
    resolved = str(known.xla_pass_log.resolve())
    if resolved == "/tmp" or resolved.startswith("/tmp/"):
        raise ValueError(f"XLA pass log must not be under /tmp: {known.xla_pass_log}")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
    vmod = known.vmodule or "hlo_pass_pipeline=1,cpu_compiler=2,buffer_assignment=1,ir_compiler=2"
    os.environ["TF_CPP_VMODULE"] = vmod
    # Keep Python's sys.stderr attached to the real stderr fd while silencing C++ fd 2
    # to /dev/null outside _compile_with_optional_xla_log.
    real_stderr_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)
    sys.stderr = os.fdopen(real_stderr_fd, "w", buffering=1)
    return known.xla_pass_log


if __name__ == "__main__":
    _PRECONFIGURED_LOG = _preconfigure_xla_logging(sys.argv[1:])
else:
    _PRECONFIGURED_LOG = None

import jax
import jax.numpy as jnp
from jax.extend import core as jax_core
import numpy as np
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ice import si3_o17 as o17_module
from legoesm.ice.dynamics import SI3CGridAEVPForcing, SI3CGridAEVPState, SI3CGridMetrics
from legoesm.ice.si3_finalize import SI3StressInputs
from legoesm.ice.si3_o17 import (
    SI3O17LiveForcing,
    SI3O17StepState,
    SI3O17SurfaceForcing,
    build_orca2_sas_ice_o17_card,
    si3_o17_step,
)
from legoesm.ice.si3_orca2_state import (
    SI3ORCA2AdvectionGrid,
    SI3State,
    si3_lbc_lnk_orca2,
    si3_update_snwice_mass,
    si3_var_agg,
    si3_var_glo2eqv,
)
from legoesm.ice.si3_surface import SI3SurfaceConfig

from scripts.validate.ocean_fidelity.testcases.nemo_si3_o8_o12_master_scorer import (
    DEFAULT_PPSRC,
    parse_resolved_ice_namelist,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MASTER = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round18/oracle_o8_o16_master"
)

# Canonical execution-order partition of si3_o17_step into N=14 rungs covering
# all 10 major operators and the 4 inter-rung glue/halo/surface/tail stages.
# Every function listed in ``funcs`` is called exactly once at the top level of
# ``si3_o17_step``.
RUNG_SPECS: tuple[dict[str, Any], ...] = (
    {
        "k": 1,
        "name": "o7_blk1_strength",
        "funcs": (
            "si3_eos_fzp",
            "si3_blk_ice_1",
            "si3_ice_sbc_tau_lbc",
            "si3_orca2_ice_strength",
        ),
        "description": "Entry freezing point, O7 bulk_ice_1 + wind-stress LBC, O5 H79 ice strength",
    },
    {
        "k": 2,
        "name": "o5_aevp",
        "funcs": ("si3_orca2_aevp_solver", "si3_cgrid_deformation"),
        "description": "O5 100-subcycle aEVP rheology solver and C-grid deformation",
    },
    {
        "k": 3,
        "name": "o4_prather",
        "funcs": ("si3_dyn_orca2",),
        "description": "O4 Prather advection (checkpointed 3-subcycle scan)",
    },
    {
        "k": 4,
        "name": "o6_rdgrft",
        "funcs": ("si3_ice_dyn_rdgrft",),
        "description": "O6 ridging and rafting redistribution",
    },
    {
        "k": 5,
        "name": "o6_ice_cor_dyn",
        "funcs": ("si3_ice_cor_dyn",),
        "description": "O6 post-dynamics correction ice_cor(kt, 1)",
    },
    {
        "k": 6,
        "name": "postdyn_halos_agg",
        "funcs": ("si3_apply_state_halos_orca2", "si3_var_glo2eqv", "si3_var_agg"),
        "description": "Post-dynamics full state LBC (69 fields), glo2eqv(2), and var_agg(1)",
    },
    {
        "k": 7,
        "name": "o7_surface_o8",
        "funcs": ("si3_ice_alb", "si3_blk_ice_2", "si3_ice_flx_other", "si3_ice_thd_frazil"),
        "description": "O7 ice albedo, blk_ice_2, flx_other, and O8 frazil",
    },
    {
        "k": 8,
        "name": "o12_thermo",
        "funcs": ("_thermo_entries", "si3_ice_thd_o12_stages"),
        "description": "Thermo category unpack/scatter and O9-O12 category scan + LEV ponds",
    },
    {
        "k": 9,
        "name": "o13_remap",
        "funcs": ("si3_ice_itd_rem",),
        "description": "Category state packing and O13 ITD thickness remapping",
    },
    {
        "k": 10,
        "name": "o14_open_water",
        "funcs": ("si3_ice_thd_do",),
        "description": "Remap scatter and O14 open-water ice growth",
    },
    {
        "k": 11,
        "name": "o15_cor",
        "funcs": ("si3_o15_step",),
        "description": "Open-water scatter and O15 post-thermo correction ice_cor(kt, 2)",
    },
    {
        "k": 12,
        "name": "thermo_tail",
        "funcs": ("_thermodynamic_tail",),
        "description": "Post-ice_cor(kt, 2) ageing, 14 LBC calls, category sum, velocity zeroing",
    },
    {
        "k": 13,
        "name": "o16_update_flx",
        "funcs": ("si3_o16_step",),
        "description": "O16 glo2eqv(2), var_agg(2), and ice_update_flx ocean exchange",
    },
    {
        "k": 14,
        "name": "o16_update_tau",
        "funcs": ("si3_ice_update_tau",),
        "description": "Final ocean-ice stress update ice_update_tau and full-step publication",
    },
)

ALL_SINGLE_CALL_FUNCS: tuple[str, ...] = tuple(
    fn for spec in RUNG_SPECS for fn in spec["funcs"]
)
D198_TEN_RUNG_FUNCS: tuple[str, ...] = (
    "si3_orca2_aevp_solver",
    "si3_dyn_orca2",
    "si3_ice_dyn_rdgrft",
    "si3_ice_cor_dyn",
    "si3_ice_thd_o12_stages",
    "si3_ice_itd_rem",
    "si3_ice_thd_do",
    "si3_o15_step",
    "si3_o16_step",
    "si3_ice_update_tau",
)

_HLO_INSTR_RE = re.compile(r"^\s+(?:ROOT\s+)?%?[\w\.\-]+\s+=\s+")


def _provenance() -> dict[str, Any]:
    """Stamp git SHA, working-tree cleanliness, CLI flags, and relevant env vars."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        sha = "UNKNOWN"
    try:
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
    except Exception:
        dirty = ["UNKNOWN"]
    env_keys = (
        "CUDA_VISIBLE_DEVICES",
        "JAX_PLATFORMS",
        "JAX_ENABLE_X64",
        "XLA_FLAGS",
        "TF_CPP_MIN_LOG_LEVEL",
        "TF_CPP_VMODULE",
        "TMPDIR",
        "PYTHONPATH",
    )
    return {
        "git_sha": sha,
        "git_dirty_count": len(dirty),
        "git_dirty_files": dirty[:20],
        "argv": list(sys.argv),
        "env": {k: os.environ.get(k) for k in env_keys},
        "jax_version": getattr(jax, "__version__", "unknown"),
        "default_backend": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
    }


def _count_hlo_text(hlo_text: str) -> dict[str, int]:
    """Count total HLO instructions, ENTRY instructions, and computations in HLO text."""
    total_instr = 0
    entry_instr = 0
    computations = 0
    in_entry = False
    brace_depth = 0
    for line in hlo_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("//", "/*", "HloModule")):
            continue
        if brace_depth == 0 and stripped.endswith("{"):
            computations += 1
            in_entry = stripped.startswith("ENTRY ")
            brace_depth = 1
            continue
        if brace_depth > 0:
            brace_depth += line.count("{") - line.count("}")
            if _HLO_INSTR_RE.match(line):
                total_instr += 1
                if in_entry:
                    entry_instr += 1
            if brace_depth <= 0:
                in_entry = False
                brace_depth = 0
    return {
        "instructions_total": total_instr,
        "instructions_entry": entry_instr,
        "computations": computations,
    }


def _nested_jaxprs(value: Any) -> Iterable[jax_core.Jaxpr]:
    if isinstance(value, jax_core.ClosedJaxpr):
        yield value.jaxpr
    elif isinstance(value, jax_core.Jaxpr):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _nested_jaxprs(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _nested_jaxprs(child)


def _count_jaxpr_equations(closed_jaxpr: jax_core.ClosedJaxpr) -> dict[str, int]:
    top_eqns = len(closed_jaxpr.jaxpr.eqns)
    rec_eqns = 0

    def walk(j: jax_core.Jaxpr) -> None:
        nonlocal rec_eqns
        for eqn in j.eqns:
            rec_eqns += 1
            for child in _nested_jaxprs(eqn.params):
                walk(child)

    walk(closed_jaxpr.jaxpr)
    return {"top_eqns": top_eqns, "recursive_eqns": rec_eqns}


def _parse_duration_us(text: str) -> float:
    """Extract microsecond value from '(12345 us)' or fallback to unit conversion."""
    m = re.search(r"\((\d+)\s*us\)", text)
    if m:
        return float(m.group(1))
    m2 = re.search(r"time:\s*([0-9.]+)\s*(us|ms|s)\b", text)
    if m2:
        val = float(m2.group(1))
        unit = m2.group(2)
        if unit == "us":
            return val
        if unit == "ms":
            return val * 1e3
        if unit == "s":
            return val * 1e6
    return 0.0


def _parse_vlog_timestamp(line: str) -> float | None:
    """Parse 'I1103 07:34:49.600038 ...' into seconds since midnight (handling day wrap)."""
    m = re.match(r"^[IWEF](\d{2})(\d{2})\s+(\d{2}):(\d{2}):(\d{2}\.\d+)", line)
    if not m:
        return None
    _month, day, hh, mm, ss = m.groups()
    return int(day) * 86400.0 + int(hh) * 3600.0 + int(mm) * 60.0 + float(ss)


def parse_xla_pass_log(log_path: Path) -> dict[str, Any]:
    """Parse C++ VLOG from hlo_pass_pipeline, buffer_assignment, cpu_compiler, ir_compiler."""
    if not log_path.exists():
        return {"error": f"log file {log_path} does not exist"}
    lines = log_path.read_text(errors="replace").splitlines()
    # Locate the last main module compilation (or last 'Running HLO pass pipeline on module')
    target_module = None
    for line in lines:
        m = re.search(r"Running HLO pass pipeline on module ([\w\.\-]+):", line)
        if m:
            target_module = m.group(1)
    # Find the index where target_module's first HLO pass pipeline starts
    start_idx = 0
    if target_module:
        for i, line in enumerate(lines):
            if f"Running HLO pass pipeline on module {target_module}:" in line:
                start_idx = i
                break
    relevant = lines[start_idx:]

    pass_totals_us: dict[str, float] = defaultdict(float)
    pass_max_us: dict[str, float] = defaultdict(float)
    pass_calls: Counter[str] = Counter()
    subpipeline_totals_us: dict[str, float] = defaultdict(float)
    # Compound pipeline wrappers whose time is the sum of inner passes
    compound_pipelines = {
        "after layout assignment",
        "simplification after layout assignment",
        "simplification",
        "post_scatter_expansion_simplification",
        "pipeline",
        "HLO passes through layout assignment",
        "HLO passes after layout assignment",
        "hlo normalization",
        "dot-library-passes",
        "sharding-removal",
        "async-collective",
        "SubbytePacker pipeline",
    }
    for line in relevant:
        m_pipe = re.search(r"Running HLO pass pipeline on module [\w\.\-]+:\s*(.+)$", line)
        if m_pipe:
            compound_pipelines.add(m_pipe.group(1).strip())

    first_hlo_ts = None
    last_hlo_ts = None
    last_hlo_pass_started = None
    compile_backend_start_ts = None
    buf_assign_start_ts = None
    buf_assign_end_ts = None
    buffers_to_assign = None
    total_allocation_str = None
    llvm_compile_start_ts = None
    llvm_compile_end_ts = None
    backend_jit_time_s = None
    llvm_modules: list[dict[str, Any]] = []

    for line in relevant:
        ts = _parse_vlog_timestamp(line)
        if "hlo_pass_pipeline.cc" in line:
            if first_hlo_ts is None and ts is not None:
                first_hlo_ts = ts
            if ts is not None:
                last_hlo_ts = ts
            m_start = re.search(r"hlo_pass_pipeline\.cc:\d+\]\s+HLO pass\s+(.+)$", line)
            if m_start:
                pname_start = m_start.group(1).strip()
                if pname_start not in compound_pipelines:
                    last_hlo_pass_started = pname_start
            m_done = re.search(r"HLO pass:\s+(.+?)\s+time:\s+(.+)$", line)
            if m_done:
                pname = m_done.group(1).strip()
                dur_us = _parse_duration_us(m_done.group(2))
                if pname in compound_pipelines:
                    subpipeline_totals_us[pname] += dur_us
                else:
                    pass_totals_us[pname] += dur_us
                    pass_max_us[pname] = max(pass_max_us[pname], dur_us)
                    pass_calls[pname] += 1
        elif "cpu_compiler.cc" in line:
            if "Compiling: " in line and ts is not None:
                compile_backend_start_ts = ts
            elif "Compile LLVM module with" in line and ts is not None:
                llvm_compile_start_ts = ts
            elif "Compilation finished" in line and ts is not None:
                llvm_compile_end_ts = ts
            elif "for CPU using JIT time:" in line:
                backend_jit_time_s = round(_parse_duration_us(line) / 1e6, 4)
        elif "buffer_assignment.cc" in line:
            if "Assigning buffers to module" in line and ts is not None:
                buf_assign_start_ts = ts
            elif "Number of buffers to assign:" in line:
                m_buf = re.search(r"Number of buffers to assign:\s*(\d+)", line)
                if m_buf:
                    buffers_to_assign = int(m_buf.group(1))
            elif "total allocation:" in line:
                m_tot = re.search(r"total allocation:\s*(\S+)", line)
                if m_tot:
                    total_allocation_str = m_tot.group(1)
            elif "Buffer assignment done." in line and ts is not None:
                buf_assign_end_ts = ts
        elif "ir_compiler.cc" in line and "Compiled LLVM module:" in line:
            m_ir = re.search(r"Compiled LLVM module:\s*(\S+)\s+time:\s*(.+)$", line)
            if m_ir:
                llvm_modules.append(
                    {
                        "module": m_ir.group(1),
                        "seconds": round(_parse_duration_us(m_ir.group(2)) / 1e6, 4),
                    }
                )

    top_passes = sorted(pass_totals_us.items(), key=lambda kv: kv[1], reverse=True)[:10]
    llvm_modules_sorted = sorted(llvm_modules, key=lambda d: d["seconds"], reverse=True)

    def _diff(a: float | None, b: float | None) -> float | None:
        return round(b - a, 4) if (a is not None and b is not None and b >= a) else None

    return {
        "module": target_module,
        "hlo_passes_wall_s": _diff(first_hlo_ts, compile_backend_start_ts or last_hlo_ts),
        "hlo_passes_sum_s": round(sum(pass_totals_us.values()) / 1e6, 4),
        "last_hlo_pass_started": last_hlo_pass_started,
        "buffer_assignment_s": _diff(buf_assign_start_ts, buf_assign_end_ts),
        "buffers_to_assign": buffers_to_assign,
        "total_allocation": total_allocation_str,
        "llvm_ir_emit_s": _diff(buf_assign_end_ts, llvm_compile_start_ts),
        "llvm_codegen_wall_s": _diff(llvm_compile_start_ts, llvm_compile_end_ts),
        "llvm_modules_compiled": len(llvm_modules),
        "llvm_modules_sum_s": round(sum(m["seconds"] for m in llvm_modules), 4),
        "backend_jit_total_s": backend_jit_time_s,
        "top_hlo_passes": [
            {
                "pass": name,
                "total_s": round(tot_us / 1e6, 4),
                "max_s": round(pass_max_us[name] / 1e6, 4),
                "calls": int(pass_calls[name]),
            }
            for name, tot_us in top_passes
        ],
        "top_llvm_modules": llvm_modules_sorted[:5],
    }


def _case(master: Path, shape: tuple[int, int] = (7, 7)):
    """Mixed-regime O17 step entry; ``shape`` (184, 152) is the real ORCA2 rank grid."""
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    run = master / "record_100step"
    nml = parse_resolved_ice_namelist(
        run / "output.namelist.ice", run / "output.namelist.dyn", DEFAULT_PPSRC
    )
    surface_cfg = SI3SurfaceConfig.from_resolved_namelists(
        run / "output.namelist.dyn", run / "output.namelist.ice"
    )
    card = build_orca2_sas_ice_o17_card(surface=surface_cfg, thermo_namelist=nml)
    # The owned tile exercises three regimes at separated columns: open water,
    # active ice, and a nonzero category column below epsi10 (icethd.F90:121-138).
    (ni, nj), h = shape, 2
    ii = (ni - 2 * h, nj - 2 * h)
    full = (ni, nj)
    zf, of = jnp.zeros(full, jnp.float64), jnp.ones(full, jnp.float64)
    oi = jnp.ones(ii, jnp.float64)
    ix, iy = jnp.meshgrid(
        jnp.arange(ni, dtype=jnp.float64), jnp.arange(nj, dtype=jnp.float64), indexing="ij"
    )
    ocean_u = 0.10 + 0.001 * ix + 0.0002 * iy
    ocean_v = 0.05 + 0.0003 * ix - 0.0007 * iy

    def lbc(value, nature="T", sign=1.0):
        return si3_lbc_lnk_orca2(
            jnp.asarray(value, dtype=jnp.float64),
            nature,
            sign,
            nn_hls=card.state.nn_hls,
            ldfull=True,
        )

    pattern = np.asarray(  # icethd.F90:123,131: ice, open water, nonzero <= epsi10
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0e-12]], dtype=np.float64
    )
    reps = (-(-ii[0] // 3), -(-ii[1] // 3))
    ice_owned = jnp.asarray(np.tile(pattern, reps)[: ii[0], : ii[1]])
    ice_mask = lbc(zf.at[h:-h, h:-h].set(ice_owned))
    category_area = jnp.asarray([0.12, 0.16, 0.18, 0.14, 0.10])
    category_thickness = jnp.asarray([0.25, 0.80, 1.60, 2.80, 4.50])
    a_i = lbc(ice_mask[..., None] * category_area)
    v_i = lbc(a_i * category_thickness)
    v_s = lbc(a_i * 0.12)
    sv_i = lbc(v_i * 5.0)
    a_ip = lbc(a_i * 0.08)
    v_ip = lbc(a_ip * 0.04)
    v_il = lbc(a_ip * 0.008)
    oa_i = lbc(a_i * 30.0)
    t_su = lbc(jnp.where(a_i > 0.0, card.constants.T0 - 5.0, card.constants.T0))
    e_i = lbc(
        jnp.broadcast_to(
            v_i[..., None, :] * (3.0e8 / card.state.nlay_i),
            (*full, card.state.nlay_i, card.state.jpl),
        )
    )
    e_s = lbc(
        jnp.broadcast_to(
            v_s[..., None, :] * (1.1e8 / card.state.nlay_s),
            (*full, card.state.nlay_s, card.state.jpl),
        )
    )
    szv_i = lbc(
        jnp.broadcast_to(
            v_i[..., None, :] * (5.0 / card.state.nlay_i),
            (*full, card.state.nlay_i, card.state.jpl),
        )
    )
    u_ice = lbc((0.04 + 0.0005 * ix) * ice_mask, "U", -1.0)
    v_ice = lbc((-0.02 + 0.0003 * iy) * ice_mask, "V", -1.0)
    eqv = si3_var_glo2eqv(
        v_i=v_i,
        v_s=v_s,
        sv_i=sv_i,
        a_i=a_i,
        oa_i=oa_i,
        a_ip=a_ip,
        v_ip=v_ip,
        v_il=v_il,
        e_s=e_s,
        e_i=e_i,
        szv_i=szv_i,
        kn=2,
        hi_max_top=99.0,
        sss_m=jnp.full(full, 34.0),
        config=card.state,
    )
    t_si = jnp.broadcast_to(
        jnp.where(ice_mask[..., None] > 0.0, card.constants.T0 - 1.0, 0.0),
        (*full, card.state.jpl),
    )
    agg = si3_var_agg(
        v_i=v_i,
        v_s=eqv.v_s,
        sv_i=sv_i,
        a_i=eqv.a_i,
        oa_i=oa_i,
        a_ip=a_ip,
        v_ip=v_ip,
        v_il=v_il,
        t_su=t_su,
        t_si=t_si,
        e_s=e_s,
        e_i=e_i,
        szv_i=szv_i,
        a_ip_eff=eqv.a_ip_eff,
        t_i=eqv.t_i,
        t_s=eqv.t_s,
        kn=2,
        config=card.state,
    )
    snwice_mass, snwice_mass_b, snwice_fmass = si3_update_snwice_mass(
        vt_i=agg.vt_i,
        vt_s=agg.vt_s,
        vt_ip=agg.vt_ip,
        vt_il=agg.vt_il,
        snwice_mass_prev=zf,
        tmask=of,
        config=card.state,
        is_init=True,
    )
    base = SI3State.zeros(ni, nj, card.state)
    state = base._replace(
        v_i=v_i,
        v_s=eqv.v_s,
        sv_i=sv_i,
        a_i=eqv.a_i,
        t_su=t_su,
        u_ice=u_ice,
        v_ice=v_ice,
        oa_i=oa_i,
        a_ip=a_ip,
        v_ip=v_ip,
        v_il=v_il,
        e_s=e_s,
        e_i=e_i,
        szv_i=szv_i,
        t1_ice=t_su,
        t_si=t_si,
        snwice_mass=snwice_mass,
        snwice_mass_b=snwice_mass_b,
        snwice_fmass=snwice_fmass,
        equivalent=eqv,
        aggregated=agg,
    )
    length = jnp.full(full, 2.0e5)
    area = jnp.full(full, 4.0e10)
    metrics = SI3CGridMetrics(
        length,
        length,
        length,
        length,
        length,
        length,
        length,
        length,
        area,
        area,
        area,
        area,
    )
    aevp_forcing = SI3CGridAEVPForcing(
        concentration_t=agg.at_i,
        ice_volume_t=agg.vt_i,
        snow_volume_t=agg.vt_s,
        pond_volume_t=agg.vt_ip,
        lid_volume_t=agg.vt_il,
        air_stress_u_t=jnp.full(full, 0.08),
        air_stress_v_t=jnp.full(full, 0.03),
        drag_io_t=jnp.full(full, 0.005),
        ocean_u_u=ocean_u,
        ocean_v_v=ocean_v,
        ssh_t=None,
        coriolis_t=jnp.full(full, 1.0e-4),
        tmask_t=of,
        umask_u=of,
        vmask_v=of,
        fast_tmask=zf,
        depth_t=jnp.full(full, 1000.0),
        depth_u=jnp.full(full, 1000.0),
        depth_v=jnp.full(full, 1000.0),
        iceberg_tmask=zf,
        iceberg_umask=zf,
        iceberg_vmask=zf,
    )
    aevp = SI3CGridAEVPState(
        u_ice,
        v_ice,
        jnp.full(full, 100.0),
        jnp.full(full, -60.0),
        jnp.full(full, 20.0),
    )
    grid = SI3ORCA2AdvectionGrid(
        e1u=length,
        e2v=length,
        r1_e1u=1.0 / length,
        r1_e2v=1.0 / length,
        e2u=length,
        e1v=length,
        e1e2t=area,
        r1_e1e2t=1.0 / area,
        tmask=of,
        umask=of,
        vmask=of,
    )
    surface = SI3O17SurfaceForcing(
        wndi=jnp.full(ii, 5.0),
        wndj=jnp.full(ii, 2.0),
        theta_air_zt=jnp.full(ii, 270.0),
        q_air_zt=jnp.full(ii, 0.002),
        slp=jnp.full(ii, 101000.0),
        rhoa=jnp.full(ii, 1.3),
        cloud_fra=jnp.full(ii, 0.5),
        smask0=oi,
        qlw=jnp.full(ii, 250.0),
        precip=jnp.full(ii, 1.0e-7),
        snow=jnp.full(ii, 4.0e-8),
        qsr=jnp.full(ii, 80.0),
        emp=jnp.full(ii, 2.0e-8),
        qns_oce=jnp.full(ii, 20.0),
        qsr_oce=jnp.full(ii, 80.0),
        ssu_m=ocean_u,
        ssv_m=ocean_v,
        utau=jnp.full(ii, 0.08),
        vtau=jnp.full(ii, 0.03),
        frq_m=jnp.full(ii, 0.02),
        e3t_m=jnp.full(ii, 10.0),
    )
    stress = SI3StressInputs(
        uu_kbb=ocean_u,
        vv_kbb=ocean_v,
        umask_k1=of,
        vmask_k1=of,
        u_ice=u_ice,
        v_ice=v_ice,
        u_oce=ocean_u,
        v_oce=ocean_v,
        drag_io=jnp.full(full, 0.005),
        at_i=agg.at_i,
        utau=jnp.full(full, 0.08),
        vtau=jnp.full(full, 0.03),
        tmod_io=jnp.full(full, 0.04),
        taum=jnp.full(ii, 0.09),
        utau_oce=jnp.full(ii, 0.04),
        vtau_oce=jnp.full(ii, 0.02),
    )
    forcing = SI3O17LiveForcing(
        kt=1,
        sss_m=jnp.full(full, 34.0),
        sst_m=jnp.full(full, -1.8),
        rn_amax_2d=jnp.full(full, 0.95),
        advection_grid=grid,
        aevp_forcing=aevp_forcing,
        aevp_metrics=metrics,
        ssh_m=zf,
        surface=surface,
        stress=stress,
    )
    return SI3O17StepState(state, aevp), forcing, card


class _SliceStop(Exception):
    """Internal control-flow signal raised during JAX tracing at a rung boundary."""

    def __init__(self, payload: Any):
        self.payload = payload


def _is_float_leaf(leaf: Any) -> bool:
    return (
        hasattr(leaf, "dtype")
        and hasattr(leaf, "shape")
        and jnp.issubdtype(leaf.dtype, jnp.floating)
    )


def _sum_unique_floating_leaves(tree: Any) -> jax.Array:
    seen: set[int] = set()
    terms: list[jax.Array] = []
    for leaf in jax.tree_util.tree_leaves(tree):
        if _is_float_leaf(leaf) and id(leaf) not in seen:
            seen.add(id(leaf))
            terms.append(jnp.sum(leaf))
    if not terms:
        return jnp.asarray(0.0, dtype=jnp.float64)
    return sum(terms)


def _record_eager_interceptions(
    entry: SI3O17StepState,
    forcing: SI3O17LiveForcing,
    card: Any,
    func_names: Iterable[str],
) -> tuple[dict[str, tuple[Any, Any, Any]], float]:
    """Run one concrete eager step and record (args, kwargs, output) for named functions."""
    names = tuple(func_names)
    recorded: dict[str, tuple[Any, Any, Any]] = {}
    originals = {name: getattr(o17_module, name) for name in names}

    def make_recorder(name: str, fn: Any):
        def wrapped(*args, **kwargs):
            out = fn(*args, **kwargs)
            recorded.setdefault(name, (args, kwargs, out))
            return out

        return wrapped

    t0 = time.monotonic()
    try:
        for name in names:
            setattr(o17_module, name, make_recorder(name, originals[name]))
        o17_module.si3_o17_step(entry, SI3O17LiveForcing(int(forcing.kt), *forcing[1:]), card)
    finally:
        for name, fn in originals.items():
            setattr(o17_module, name, fn)
    return recorded, time.monotonic() - t0


def _compile_with_optional_xla_log(
    lowered: Any,
    xla_pass_log: Path | None,
) -> tuple[Any, float, dict[str, Any] | None]:
    """Compile a lowered JAX computation, optionally redirecting C++ stderr to xla_pass_log."""
    if xla_pass_log is not None:
        resolved = str(xla_pass_log.resolve())
        if resolved == "/tmp" or resolved.startswith("/tmp/"):
            raise ValueError(f"XLA pass log must not be under /tmp: {xla_pass_log}")
        xla_pass_log.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(xla_pass_log), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        sys.stderr.flush()
        saved_stderr = os.dup(2)
        os.dup2(fd, 2)
        os.close(fd)
        t0 = time.monotonic()
        try:
            compiled = lowered.compile()
        finally:
            sys.stderr.flush()
            os.dup2(saved_stderr, 2)
            os.close(saved_stderr)
        compile_s = time.monotonic() - t0
        return compiled, compile_s, parse_xla_pass_log(xla_pass_log)
    t0 = time.monotonic()
    compiled = lowered.compile()
    compile_s = time.monotonic() - t0
    return compiled, compile_s, None


def _log_progress(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    sys.stderr.write(f"[{ts}] {msg}\n")
    sys.stderr.flush()


def measure_reverse_compile_unit(
    label: str,
    objective_fn: Any,
    diff_inputs: Any,
    *,
    xla_pass_log: Path | None = None,
    execute: bool = True,
    on_compiled: Any = None,
) -> dict[str, Any]:
    """Measure JAXPR counts, lower(), HLO counts, compile(), peak RSS, and optional execution."""
    _log_progress(f"{label}: start make_jaxpr(fwd)")
    t_jaxpr0 = time.monotonic()
    fwd_closed = jax.make_jaxpr(objective_fn)(diff_inputs)
    fwd_jaxpr_s = time.monotonic() - t_jaxpr0
    fwd_counts = _count_jaxpr_equations(fwd_closed)

    vg_fn = jax.value_and_grad(objective_fn)
    _log_progress(
        f"{label}: done make_jaxpr(fwd) in {fwd_jaxpr_s:.2f}s "
        f"(top={fwd_counts['top_eqns']}, rec={fwd_counts['recursive_eqns']}); start make_jaxpr(vg)"
    )
    t_vg_jaxpr0 = time.monotonic()
    vg_closed = jax.make_jaxpr(vg_fn)(diff_inputs)
    vg_jaxpr_s = time.monotonic() - t_vg_jaxpr0
    vg_counts = _count_jaxpr_equations(vg_closed)

    transformed = jax.jit(vg_fn)
    _log_progress(
        f"{label}: done make_jaxpr(vg) in {vg_jaxpr_s:.2f}s "
        f"(top={vg_counts['top_eqns']}, rec={vg_counts['recursive_eqns']}); start lower()"
    )
    t_low0 = time.monotonic()
    lowered = transformed.lower(diff_inputs)
    lower_s = time.monotonic() - t_low0

    hlo_pre_text = lowered.as_text(dialect="hlo")
    hlo_pre = _count_hlo_text(hlo_pre_text)
    _log_progress(
        f"{label}: done lower() in {lower_s:.2f}s "
        f"(hlo_pre={hlo_pre['instructions_total']}); start compile()"
    )

    compiled, compile_s, xla_breakdown = _compile_with_optional_xla_log(lowered, xla_pass_log)
    _log_progress(f"{label}: done compile() in {compile_s:.2f}s; start compiled.as_text()")
    hlo_post_text = compiled.as_text()
    hlo_post = _count_hlo_text(hlo_post_text)
    _log_progress(
        f"{label}: done compiled.as_text() (hlo_post={hlo_post['instructions_total']})"
    )

    rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    row: dict[str, Any] = {
        "label": label,
        "fwd_make_jaxpr_seconds": round(fwd_jaxpr_s, 3),
        "vg_make_jaxpr_seconds": round(vg_jaxpr_s, 3),
        "tracing_seconds": round(vg_jaxpr_s, 3),
        "lower_seconds": round(lower_s, 3),
        "lowering_seconds": round(lower_s, 3),
        "lower_total_seconds": round(vg_jaxpr_s + lower_s, 3),
        "compile_seconds": round(compile_s, 3),
        "compilation_seconds": round(compile_s, 3),
        "lower_plus_compile_seconds": round(lower_s + compile_s, 3),
        "trace_plus_lower_plus_compile_seconds": round(vg_jaxpr_s + lower_s + compile_s, 3),
        "execution_seconds": None,
        "objective": None,
        "peak_rss_kib": rss_kib,
        "peak_rss_gib": round(rss_kib / (1024.0 * 1024.0), 3),
        "fwd_jaxpr_top_eqns": fwd_counts["top_eqns"],
        "fwd_jaxpr_rec_eqns": fwd_counts["recursive_eqns"],
        "vg_jaxpr_top_eqns": vg_counts["top_eqns"],
        "vg_jaxpr_rec_eqns": vg_counts["recursive_eqns"],
        "hlo_pre_opt_instructions": hlo_pre["instructions_total"],
        "hlo_pre_opt_entry_instructions": hlo_pre["instructions_entry"],
        "hlo_pre_opt_computations": hlo_pre["computations"],
        "hlo_post_opt_instructions": hlo_post["instructions_total"],
        "hlo_post_opt_entry_instructions": hlo_post["instructions_entry"],
        "hlo_post_opt_computations": hlo_post["computations"],
    }
    if xla_breakdown is not None:
        row["xla_pass_log"] = str(xla_pass_log)
        row["xla_breakdown"] = xla_breakdown
    if on_compiled is not None:
        on_compiled(row)

    if execute:
        _log_progress(f"{label}: start execute()")
        t_exec0 = time.monotonic()
        val, _grad = compiled(diff_inputs)
        jax.block_until_ready(val)
        exec_s = round(time.monotonic() - t_exec0, 4)
        obj_val = float(np.asarray(val))
        grad_arrays = [
            np.asarray(g)
            for g in jax.tree_util.tree_leaves(_grad)
            if _is_float_leaf(g)
        ]
        grad_nan = sum(int(np.isnan(a).sum()) for a in grad_arrays)
        grad_inf = sum(int(np.isinf(a).sum()) for a in grad_arrays)
        grad_finite = sum(int(np.isfinite(a).sum()) for a in grad_arrays)
        grad_nonzero = sum(int(np.count_nonzero(a)) for a in grad_arrays)
        rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        row["execution_seconds"] = exec_s
        row["objective"] = obj_val
        row["objective_is_finite"] = bool(np.isfinite(obj_val))
        row["gradient_nan_entries"] = grad_nan
        row["gradient_inf_entries"] = grad_inf
        row["gradient_finite_entries"] = grad_finite
        row["gradient_nonzero_entries"] = grad_nonzero
        row["peak_rss_kib"] = rss_kib
        row["peak_rss_gib"] = round(rss_kib / (1024.0 * 1024.0), 3)
        _log_progress(f"{label}: done execute() in {exec_s:.4f}s")
    return row


def diagnose_slices(
    master: Path = DEFAULT_MASTER,
    *,
    prefixes: Sequence[int] = (),
    slices: Sequence[tuple[int, int]] = (),
    isolated_funcs: Sequence[str] = (),
    glue_only: bool = False,
    checkpoint_rungs: Sequence[int] = (),
    xla_pass_log: Path | None = None,
    execute: bool = True,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Run reverse-compile diagnostics on prefixes 1..k, slices k1..k2, isolated funcs, or glue-only."""
    entry, forcing, card = _case(master)
    kt = int(forcing.kt)
    base_inputs = (entry, forcing[1:])

    all_slice_specs: list[tuple[str, int, int]] = []
    for k in prefixes:
        if not (1 <= k <= len(RUNG_SPECS)):
            raise ValueError(f"prefix k={k} out of range 1..{len(RUNG_SPECS)}")
        all_slice_specs.append((f"prefix_1_{k}_{RUNG_SPECS[k - 1]['name']}", 1, k))
    for k1, k2 in slices:
        if not (1 <= k1 <= k2 <= len(RUNG_SPECS)):
            raise ValueError(f"slice {k1}..{k2} out of range 1..{len(RUNG_SPECS)}")
        n1 = RUNG_SPECS[k1 - 1]["name"]
        n2 = RUNG_SPECS[k2 - 1]["name"]
        label = f"rung_{k1}_{n1}" if k1 == k2 else f"slice_{k1}_{k2}_{n1}_to_{n2}"
        all_slice_specs.append((label, k1, k2))

    need_eager_record = (
        any(k1 > 1 for _, k1, _ in all_slice_specs)
        or bool(isolated_funcs)
        or bool(glue_only)
    )
    recorded_eager: dict[str, tuple[Any, Any, Any]] = {}
    eager_record_s = None
    if need_eager_record:
        recorded_eager, eager_record_s = _record_eager_interceptions(
            entry, forcing, card, ALL_SINGLE_CALL_FUNCS
        )

    results: list[dict[str, Any]] = []
    checkpoint_set = set(checkpoint_rungs)

    def _log_for(item_label: str, total_items: int) -> Path | None:
        if xla_pass_log is None:
            return None
        if total_items == 1:
            return xla_pass_log
        stem = xla_pass_log.stem
        suffix = xla_pass_log.suffix or ".log"
        return xla_pass_log.with_name(f"{stem}_{item_label}{suffix}")

    total_runs = len(all_slice_specs) + len(isolated_funcs) + (1 if glue_only else 0)
    prov = _provenance()

    def _build_report() -> dict[str, Any]:
        exec_ok = all(
            (r.get("execution_seconds") is None)
            or (
                r.get("objective_is_finite") is True
                and r.get("gradient_nan_entries", 0) == 0
                and r.get("gradient_inf_entries", 0) == 0
            )
            for r in results
        )
        return {
            "status": "PASS" if exec_ok else "FAIL",
            "mode": "diagnostic",
            "provenance": prov,
            "eager_recording_seconds": round(eager_record_s, 3)
            if eager_record_s is not None
            else None,
            "rung_registry": [
                {"k": s["k"], "name": s["name"], "funcs": list(s["funcs"])} for s in RUNG_SPECS
            ],
            "results": results,
        }

    def _flush_output() -> None:
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(_build_report(), indent=2, sort_keys=True) + "\n")

    for label, k1, k2 in all_slice_specs:
        mocked_funcs: list[str] = [
            fn for r in range(1, k1) for fn in RUNG_SPECS[r - 1]["funcs"]
        ]
        active_rungs = [RUNG_SPECS[r - 1] for r in range(k1, k2 + 1)]
        stop_before_func = (
            RUNG_SPECS[k2]["funcs"][0] if k2 < len(RUNG_SPECS) else None
        )

        # Build template for mocked outputs of rungs 1..k1-1
        mock_templates: list[tuple[str, list[Any], list[bool], Any]] = []
        mock_float_Arrays: list[Any] = []
        for fn_name in mocked_funcs:
            _a, _kw, out_val = recorded_eager[fn_name]
            leaves, treedef = jax.tree_util.tree_flatten(out_val)
            mask = [_is_float_leaf(l) for l in leaves]
            fl = [l for l, m in zip(leaves, mask) if m]
            mock_templates.append((fn_name, leaves, mask, treedef))
            mock_float_Arrays.append(fl)

        diff_inputs = (base_inputs, mock_float_Arrays) if mocked_funcs else base_inputs

        def make_slice_objective(
            k1_val=k1,
            k2_val=k2,
            mock_templates_val=mock_templates,
            stop_fn=stop_before_func,
            active_specs=active_rungs,
        ):
            def objective(x):
                if mock_templates_val:
                    (step_entry, dynamic), mocked_fl_lists = x
                else:
                    step_entry, dynamic = x
                    mocked_fl_lists = ()

                originals = {name: getattr(o17_module, name) for name in ALL_SINGLE_CALL_FUNCS}
                captured_outputs: list[Any] = []
                try:
                    # 1. Mock all functions in rungs 1..k1-1 to return their tracer outputs in 0 eqns
                    for (fn_name, leaves, mask, treedef), fl_list in zip(
                        mock_templates_val, mocked_fl_lists
                    ):
                        it = iter(fl_list)
                        reconstructed = jax.tree_util.tree_unflatten(
                            treedef, [next(it) if m else l for l, m in zip(leaves, mask)]
                        )
                        setattr(
                            o17_module,
                            fn_name,
                            lambda *a, _ret=reconstructed, **kw: _ret,
                        )

                    # 2. Wrap active rungs k1..k2 to capture outputs (and optionally jax.checkpoint)
                    for spec in active_specs:
                        r_idx = spec["k"]
                        for fn_name in spec["funcs"]:
                            orig_fn = originals[fn_name]
                            if r_idx in checkpoint_set:
                                orig_fn = jax.checkpoint(orig_fn)

                            def make_active_wrapper(f):
                                def wrapped(*a, **kw):
                                    res = f(*a, **kw)
                                    captured_outputs.append(res)
                                    return res

                                return wrapped

                            setattr(o17_module, fn_name, make_active_wrapper(orig_fn))

                    # 3. If k2 < N, stop right at the entry of rung k2 + 1
                    if stop_fn is not None:

                        def stop_wrapper(*a, **kw):
                            raise _SliceStop((captured_outputs, a, kw))

                        setattr(o17_module, stop_fn, stop_wrapper)

                    try:
                        out, trace = si3_o17_step(
                            step_entry, SI3O17LiveForcing(kt, *dynamic), card
                        )
                    except _SliceStop as stop_exc:
                        return _sum_unique_floating_leaves(stop_exc.payload)

                    exchange = trace.finalize.final
                    primal = (out, exchange.fluxes, exchange.outputs, trace.finalize.stress)
                    return _sum_unique_floating_leaves(primal)
                finally:
                    for name, fn in originals.items():
                        setattr(o17_module, name, fn)

            return objective

        def _on_slice_compiled(
            r: dict[str, Any], k1_v=k1, k2_v=k2, active_v=active_rungs
        ) -> None:
            r["k_start"] = k1_v
            r["k_end"] = k2_v
            r["rungs"] = [s["name"] for s in active_v]
            r["checkpoint_rungs"] = sorted(checkpoint_set)
            results.append(r)
            _flush_output()

        measure_reverse_compile_unit(
            label,
            make_slice_objective(),
            diff_inputs,
            xla_pass_log=_log_for(label, total_runs),
            execute=execute,
            on_compiled=_on_slice_compiled,
        )
        _flush_output()

    for fn_name in isolated_funcs:
        if fn_name not in recorded_eager:
            raise ValueError(f"unknown or unrecorded function {fn_name!r}")
        args, kwargs, _out = recorded_eager[fn_name]
        leaves, treedef = jax.tree_util.tree_flatten((args, kwargs))
        mask = [_is_float_leaf(l) for l in leaves]
        diff = [l for l, m in zip(leaves, mask) if m]
        orig_fn = getattr(o17_module, fn_name)

        def isolated_obj(diff_leaves, leaves=leaves, mask=mask, treedef=treedef, fn=orig_fn):
            it = iter(diff_leaves)
            a, kw = jax.tree_util.tree_unflatten(
                treedef, [next(it) if m else l for l, m in zip(leaves, mask)]
            )
            out = fn(*a, **kw)
            return _sum_unique_floating_leaves(out)

        label = f"isolated_{fn_name}"

        def _on_iso_compiled(r: dict[str, Any], fn_v=fn_name) -> None:
            r["isolated_function"] = fn_v
            results.append(r)
            _flush_output()

        measure_reverse_compile_unit(
            label,
            isolated_obj,
            diff,
            xla_pass_log=_log_for(label, total_runs),
            execute=execute,
            on_compiled=_on_iso_compiled,
        )
        _flush_output()

    if glue_only:
        # Mock the 10 D198 rungs so ONLY the uncounted glue/halos/surface/packing/outer step runs
        mock_templates = []
        mock_float_arrays = []
        for fn_name in D198_TEN_RUNG_FUNCS:
            _a, _kw, out_val = recorded_eager[fn_name]
            leaves, treedef = jax.tree_util.tree_flatten(out_val)
            mask = [_is_float_leaf(l) for l in leaves]
            fl = [l for l, m in zip(leaves, mask) if m]
            mock_templates.append((fn_name, leaves, mask, treedef))
            mock_float_arrays.append(fl)

        glue_inputs = (base_inputs, mock_float_arrays)

        def glue_objective(x):
            (step_entry, dynamic), mocked_fl_lists = x
            originals = {name: getattr(o17_module, name) for name in D198_TEN_RUNG_FUNCS}
            mock_inputs_received: list[Any] = []
            try:
                for (fn_name, leaves, mask, treedef), fl_list in zip(
                    mock_templates, mocked_fl_lists
                ):
                    it = iter(fl_list)
                    reconstructed = jax.tree_util.tree_unflatten(
                        treedef, [next(it) if m else l for l, m in zip(leaves, mask)]
                    )

                    def make_mock(ret_val):
                        def _mock(*a, **kw):
                            mock_inputs_received.append((a, kw))
                            return ret_val

                        return _mock

                    setattr(o17_module, fn_name, make_mock(reconstructed))
                out, trace = si3_o17_step(step_entry, SI3O17LiveForcing(kt, *dynamic), card)
                exchange = trace.finalize.final
                primal = (out, exchange.fluxes, exchange.outputs, trace.finalize.stress)
                return _sum_unique_floating_leaves((primal, mock_inputs_received))
            finally:
                for name, fn in originals.items():
                    setattr(o17_module, name, fn)

        label = "glue_and_uncounted_rungs_only"

        def _on_glue_compiled(r: dict[str, Any]) -> None:
            r["mocked_d198_rungs"] = list(D198_TEN_RUNG_FUNCS)
            results.append(r)
            _flush_output()

        measure_reverse_compile_unit(
            label,
            glue_objective,
            glue_inputs,
            xla_pass_log=_log_for(label, total_runs),
            execute=execute,
            on_compiled=_on_glue_compiled,
        )
        _flush_output()

    return _build_report()


def probe(
    master: Path,
    *,
    execution_mode: str = "compiled",
    xla_pass_log: Path | None = None,
) -> dict:
    if execution_mode not in {"compiled", "eager"}:
        raise ValueError(f"unknown AD execution mode {execution_mode!r}")
    entry, forcing, card = _case(master)
    # Scheduling/routing values are literal static inputs.  Only floating
    # state/forcing leaves enter reverse-mode AD; this prevents ``kt`` and the
    # deck's category/iteration counts from becoming JAX tracers.
    static_inputs = {
        "forcing.kt": int(forcing.kt),
        "card.state.jpl": int(card.state.jpl),
        "card.state.nlay_i": int(card.state.nlay_i),
        "card.state.nlay_s": int(card.state.nlay_s),
        "card.state.nn_hls": int(card.state.nn_hls),
        "card.aevp.n_subcycles": int(card.aevp.n_subcycles),
        "card.aevp.category_count": int(card.aevp.category_count),
        "card.state.ln_icedyn": bool(card.state.ln_icedyn),
        "card.state.ln_icethd": bool(card.state.ln_icethd),
    }
    inputs = (entry, forcing[1:])

    def objective(x):
        step_entry, live_forcing_dynamic = x
        live_forcing = SI3O17LiveForcing(static_inputs["forcing.kt"], *live_forcing_dynamic)
        out, trace = si3_o17_step(step_entry, live_forcing, card)
        exchange = trace.finalize.final
        primal = (out, exchange.fluxes, exchange.outputs, trace.finalize.stress)
        leaves = [
            v
            for v in jax.tree_util.tree_leaves(primal)
            if hasattr(v, "dtype") and jnp.issubdtype(v.dtype, jnp.floating)
        ]
        return sum(jnp.sum(v) for v in leaves), primal

    jax.eval_shape(objective, inputs)
    if execution_mode == "compiled":
        vg_fn = jax.value_and_grad(objective, has_aux=True)
        _log_progress("full_o17_probe: start make_jaxpr(vg)")
        trace_started = time.monotonic()
        vg_closed = jax.make_jaxpr(vg_fn)(inputs)
        tracing_seconds = time.monotonic() - trace_started
        vg_counts = _count_jaxpr_equations(vg_closed)
        _log_progress(
            f"full_o17_probe: done make_jaxpr(vg) in {tracing_seconds:.2f}s "
            f"(top={vg_counts['top_eqns']}, rec={vg_counts['recursive_eqns']}); start lower()"
        )
        transformed = jax.jit(vg_fn)
        lower_started = time.monotonic()
        lowered = transformed.lower(inputs)
        lower_seconds = time.monotonic() - lower_started
        hlo_pre = _count_hlo_text(lowered.as_text(dialect="hlo"))
        _log_progress(
            f"full_o17_probe: done lower() in {lower_seconds:.2f}s "
            f"(hlo_pre={hlo_pre['instructions_total']}); start compile()"
        )
        compiled, compile_only_seconds, xla_breakdown = _compile_with_optional_xla_log(
            lowered, xla_pass_log
        )
        compile_seconds = lower_seconds + compile_only_seconds
        hlo_post = _count_hlo_text(compiled.as_text())
        _log_progress(
            f"full_o17_probe: done compile() in {compile_only_seconds:.2f}s "
            f"(hlo_post={hlo_post['instructions_total']}); start execute()"
        )
        execute_started = time.monotonic()
        (value, primal1), grad = compiled(inputs)
        jax.block_until_ready(((value, primal1), grad))
        execution_seconds = time.monotonic() - execute_started
        _log_progress(f"full_o17_probe: done execute() in {execution_seconds:.4f}s")
    else:
        tracing_seconds = None
        vg_counts = None
        lower_seconds = None
        compile_only_seconds = None
        compile_seconds = None
        hlo_pre = None
        hlo_post = None
        xla_breakdown = None
        execute_started = time.monotonic()
        with jax.disable_jit():
            (value, primal1), grad = jax.value_and_grad(objective, has_aux=True)(inputs)
        execution_seconds = time.monotonic() - execute_started

    def primal_outputs(x):
        result = si3_o17_step(x[0], SI3O17LiveForcing(static_inputs["forcing.kt"], *x[1]), card)
        return (
            result[0],
            result[1].finalize.final.fluxes,
            result[1].finalize.final.outputs,
            result[1].finalize.stress,
        )

    if execution_mode == "compiled":
        primal0 = jax.jit(primal_outputs)(inputs)
    else:
        with jax.disable_jit():
            primal0 = primal_outputs(inputs)
    raw_uint64_differing = 0
    signed_zero_differing = 0
    both_nan_differing = 0
    finite_value_differing = 0
    primal0_nan_entries = 0
    primal1_nan_entries = 0
    for a, b in zip(jax.tree_util.tree_leaves(primal0), jax.tree_util.tree_leaves(primal1)):
        if hasattr(a, "dtype") and a.dtype == jnp.float64:
            aa = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
            bb = np.ascontiguousarray(np.asarray(b, dtype=np.float64))
            primal0_nan_entries += int(np.isnan(aa).sum())
            primal1_nan_entries += int(np.isnan(bb).sum())
            ab = aa.view(np.uint64)
            bw = bb.view(np.uint64)
            diff_mask = ab != bw
            if np.any(diff_mask):
                raw_uint64_differing += int(np.sum(diff_mask))
                ad = aa[diff_mask]
                bd = bb[diff_mask]
                zero_pair = (ad == 0.0) & (bd == 0.0)
                nan_pair = np.isnan(ad) & np.isnan(bd)
                signed_zero_differing += int(np.sum(zero_pair))
                both_nan_differing += int(np.sum(nan_pair))
                finite_value_differing += int(np.sum(~(zero_pair | nan_pair)))
    unchanged = (
        finite_value_differing == 0
        and primal0_nan_entries == 0
        and primal1_nan_entries == 0
    )
    rows = []
    input_paths, _ = jax.tree_util.tree_flatten_with_path(inputs)
    grad_leaves = jax.tree_util.tree_leaves(grad)
    if len(input_paths) != len(grad_leaves):
        raise AssertionError("input/gradient pytree leaf count changed")
    for (path, input_leaf), leaf in zip(input_paths, grad_leaves):
        a = np.asarray(leaf)
        input_dtype = getattr(input_leaf, "dtype", None)
        differentiable = input_dtype is not None and jnp.issubdtype(input_dtype, jnp.floating)
        leaf_name = jax.tree_util.keystr(path)
        if leaf_name.startswith("[0]"):
            leaf_name = "entry" + leaf_name[3:]
        else:
            for index, name in enumerate(SI3O17LiveForcing._fields[1:]):
                prefix = f"[1][{index}]"
                if leaf_name.startswith(prefix):
                    leaf_name = f"forcing.{name}" + leaf_name[len(prefix) :]
                    break
        row = {
            "leaf": leaf_name,
            "shape": list(np.shape(input_leaf)),
            "input_dtype": str(
                input_dtype if input_dtype is not None else type(input_leaf).__name__
            ),
            "gradient_dtype": str(a.dtype),
            "differentiable": bool(differentiable),
        }
        if differentiable:
            row.update(
                finite=int(np.isfinite(a).sum()),
                nan=int(np.isnan(a).sum()),
                inf=int(np.isinf(a).sum()),
                nonzero=int(np.count_nonzero(a)),
            )
        else:
            raise AssertionError(f"non-floating leaf entered differentiated pytree: {path}")
        rows.append(row)
    finite_rows = [row for row in rows if row["differentiable"]]

    def zero_reason(leaf_name: str) -> str:
        if any(
            leaf_name.startswith(f"forcing.aevp_forcing.{name}")
            for name in (
                "concentration_t",
                "ice_volume_t",
                "snow_volume_t",
                "pond_volume_t",
                "lid_volume_t",
                "air_stress_u_t",
                "air_stress_v_t",
            )
        ):
            return "overwritten from the live ice state or O7 stress before the aEVP call"
        if any(
            leaf_name.startswith(f"forcing.stress.{name}")
            for name in (
                "u_ice",
                "v_ice",
                "at_i",
            )
        ):
            return "overwritten from the completed O16 state before the stress exchange"
        if any(
            token in leaf_name
            for token in (
                ".tmask",
                ".umask",
                ".vmask",
                ".smask",
                "iceberg_",
                "fast_tmask",
            )
        ):
            return "fixed routing/mask leaf; branch selection is intentionally non-differentiable"
        if any(token in leaf_name for token in ("depth_t", "depth_u", "depth_v")):
            return "landfast basal drag is disabled by the frozen deck"
        if ".prather.sal." in leaf_name:
            return "bulk-salinity Prather moments are inactive when nn_icesal=4 uses layer salt"
        return "active program has an exact zero local derivative at this mixed-regime state"

    zero_gradient_leaves = [
        {"leaf": row["leaf"], "reason": zero_reason(row["leaf"])}
        for row in finite_rows
        if row["nonzero"] == 0
    ]
    rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    report = {
        "status": "PASS"
        if np.isfinite(float(value))
        and unchanged
        and all(r["nan"] == r["inf"] == 0 for r in finite_rows)
        else "FAIL",
        "provenance": _provenance(),
        "objective": float(value),
        "forward_bits_unchanged": unchanged,
        "forward_primal_comparison": {
            "raw_uint64_differing": raw_uint64_differing,
            "decision_200_signed_zero_differing": signed_zero_differing,
            "both_nan_differing": both_nan_differing,
            "finite_value_differing": finite_value_differing,
            "primal0_nan_entries": primal0_nan_entries,
            "primal1_nan_entries": primal1_nan_entries,
        },
        "tracing_seconds": tracing_seconds,
        "lower_seconds": lower_seconds,
        "lowering_seconds": lower_seconds,
        "compile_only_seconds": compile_only_seconds,
        "compilation_seconds": compile_only_seconds,
        "compile_seconds": compile_seconds,
        "lower_plus_compile_seconds": compile_seconds,
        "trace_plus_lower_plus_compile_seconds": (
            tracing_seconds + compile_seconds
            if tracing_seconds is not None and compile_seconds is not None
            else None
        ),
        "execution_seconds": execution_seconds,
        "execution_mode": execution_mode,
        "vg_jaxpr_top_eqns": vg_counts["top_eqns"] if vg_counts else None,
        "vg_jaxpr_rec_eqns": vg_counts["recursive_eqns"] if vg_counts else None,
        "hlo_pre_opt_instructions": hlo_pre["instructions_total"] if hlo_pre else None,
        "hlo_post_opt_instructions": hlo_post["instructions_total"] if hlo_post else None,
        "peak_rss_kib": rss_kib,
        "peak_rss_gib": round(rss_kib / (1024.0 * 1024.0), 3),
        "memory_rule_reason": (
            "exact full O17 reverse-mode compile; run alone on a small conforming mixed-regime tile"
        ),
        "synthetic_regime": {
            "full_shape": [7, 7],
            "owned_shape": [3, 3],
            "ice_owned_columns": 1,
            "open_water_owned_columns": 7,
            "near_empty_owned_columns": 1,
            "ice_categories": 5,
            "snow": True,
            "ponds": True,
            "nonzero_velocity_and_stress": True,
        },
        "static_literal_inputs": static_inputs,
        "zero_gradient_leaves": zero_gradient_leaves,
        "leaves": rows,
    }
    if xla_breakdown is not None:
        report["xla_pass_log"] = str(xla_pass_log)
        report["xla_breakdown"] = xla_breakdown
    return report


def build_negative_zero_var_agg_fixture(
    recorded_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Build a conforming si3_var_agg(kn=2) input dict containing IEEE-754 -0.0 in every input.

    Exercises both:
    - Halo cell (0, 0), where interior == +0.0 and negative category/layer entries
      paired with -0.0 across categories 1..4 / layers 1..nlay propagate -0.0
      through the 50-step tm_i_acc and 25-step tm_s_acc accumulations and
      survive `* interior` across all 12 halo-masked output fields; and
    - Interior cell (2, 2), where normal float64 -1e-300 in category 0 paired with
      -0.0 in categories 1..4 and 1e30 in category 1 of a_i/v_i/a_ip underflows
      to -0.0 in both eager and XLA CPU JIT across tm_su, tm_si, om_i, sm_i,
      hm_ip, and hm_il.
    """
    neg_zero = np.float64(-0.0)
    kw: dict[str, Any] = dict(recorded_kwargs)
    array_names = (
        "v_i",
        "v_s",
        "sv_i",
        "a_i",
        "oa_i",
        "a_ip",
        "v_ip",
        "v_il",
        "t_su",
        "t_si",
        "e_s",
        "e_i",
        "szv_i",
        "a_ip_eff",
        "t_i",
        "t_s",
    )
    np_arrays = {k: np.array(kw[k], dtype=np.float64, copy=True) for k in array_names}

    # Ensure every one of the 16 input arrays contains -0.0 in category/layer > 0 on halo cell (0, 0)
    for k in array_names:
        arr = np_arrays[k]
        if arr.ndim == 3:
            arr[0, 0, :] = neg_zero
        else:
            arr[0, 0, :, :] = neg_zero

    # Halo cell (0, 0): active volume/concentration so tm_i_acc and tm_s_acc take the
    # `vt_i > epsi20` / `vt_s > epsi20` branch across all 50 / 25 steps, with -0.0 in steps 1..49 / 1..24.
    np_arrays["v_i"][0, 0, 0] = 1.0
    np_arrays["v_s"][0, 0, 0] = 1.0
    np_arrays["a_i"][0, 0, 0] = 0.5
    np_arrays["a_ip"][0, 0, 0] = 0.5
    np_arrays["t_i"][0, 0, 0, 0] = -10.0
    np_arrays["t_s"][0, 0, 0, 0] = -5.0
    np_arrays["e_s"][0, 0, 0, 0] = -1.0
    np_arrays["e_i"][0, 0, 0, 0] = -1.0
    np_arrays["a_ip_eff"][0, 0, 0] = -0.1
    np_arrays["t_su"][0, 0, 0] = -1.0
    np_arrays["t_si"][0, 0, 0] = -1.0
    np_arrays["oa_i"][0, 0, 0] = -0.1
    np_arrays["szv_i"][0, 0, 0, 0] = -0.1
    np_arrays["v_ip"][0, 0, 0] = -0.1
    np_arrays["v_il"][0, 0, 0] = -0.1

    # Interior cell (2, 2): normal float64 -1e-300 divided by 1e30 underflows to -0.0 in eager and XLA CPU JIT.
    np_arrays["a_i"][2, 2, :] = np.array([1.0, 1.0e30, neg_zero, neg_zero, neg_zero], dtype=np.float64)
    np_arrays["v_i"][2, 2, :] = np.array([1.0, 1.0e30, neg_zero, neg_zero, neg_zero], dtype=np.float64)
    np_arrays["a_ip"][2, 2, :] = np.array([1.0, 1.0e30, neg_zero, neg_zero, neg_zero], dtype=np.float64)
    np_arrays["t_su"][2, 2, :] = np.array([-1.0e-300, neg_zero, neg_zero, neg_zero, neg_zero], dtype=np.float64)
    np_arrays["t_si"][2, 2, :] = np.array([-1.0e-300, neg_zero, neg_zero, neg_zero, neg_zero], dtype=np.float64)
    np_arrays["oa_i"][2, 2, :] = np.array([-1.0e-300, neg_zero, neg_zero, neg_zero, neg_zero], dtype=np.float64)
    np_arrays["v_ip"][2, 2, :] = np.array([-1.0e-300, neg_zero, neg_zero, neg_zero, neg_zero], dtype=np.float64)
    np_arrays["v_il"][2, 2, :] = np.array([-1.0e-300, neg_zero, neg_zero, neg_zero, neg_zero], dtype=np.float64)
    np_arrays["szv_i"][2, 2, :, :] = neg_zero
    np_arrays["szv_i"][2, 2, 0, 0] = -1.0e-300

    for k, arr in np_arrays.items():
        kw[k] = jnp.asarray(arr, dtype=jnp.float64)
    return kw


def _pytree_bit_stats(reference: Any, candidate: Any) -> dict[str, Any]:
    ref_leaves = [
        np.asarray(x, dtype=np.float64)
        for x in jax.tree_util.tree_leaves(reference)
        if _is_float_leaf(x)
    ]
    cand_leaves = [
        np.asarray(x, dtype=np.float64)
        for x in jax.tree_util.tree_leaves(candidate)
        if _is_float_leaf(x)
    ]
    if len(ref_leaves) != len(cand_leaves):
        raise AssertionError(f"float leaf count mismatch: {len(ref_leaves)} vs {len(cand_leaves)}")
    bit_mismatches = 0
    pos_zeros_ref = 0
    neg_zeros_ref = 0
    neg_zeros_cand = 0
    nan_count = 0
    for r, c in zip(ref_leaves, cand_leaves):
        rb = r.view(np.uint64)
        cb = c.view(np.uint64)
        bit_mismatches += int(np.count_nonzero(rb != cb))
        pos_zeros_ref += int(np.count_nonzero(rb == np.uint64(0)))
        neg_zeros_ref += int(np.count_nonzero(rb == np.uint64(0x8000000000000000)))
        neg_zeros_cand += int(np.count_nonzero(cb == np.uint64(0x8000000000000000)))
        nan_count += int(np.count_nonzero(np.isnan(r)))
    return {
        "float_leaves": len(ref_leaves),
        "bit_mismatches": bit_mismatches,
        "positive_zeros_ref": pos_zeros_ref,
        "negative_zeros_ref": neg_zeros_ref,
        "negative_zeros_cand": neg_zeros_cand,
        "nan_count": nan_count,
    }


def reproduce_d198_r5_var_agg_and_o16(
    master: Path = DEFAULT_MASTER,
    *,
    execute_scan: bool = True,
    include_o15: bool = True,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Committed DECISION 198 round 5 reproducer for si3_var_agg(kn=2), si3_o16_step, and si3_o15_step.

    1. Intercepts real recorded O15 and O16 inputs from a single eager si3_o17_step call.
    2. Verifies 0-ULP forward bit parity (eager vs jit, unrolled vs scan/traced, signed zeros
       and NaNs included) on both recorded inputs and the negative-zero fixture.
    3. Measures reverse-mode AD compile (tracing, lowering, compilation separately) for
       unrolled (compile-only) vs scan/traced (compile + execute).
    """
    from legoesm.ice import si3_finalize as finalize_mod
    from legoesm.ice import si3_orca2_state as state_mod

    entry, forcing, card = _case(master)
    recorded, eager_s = _record_eager_interceptions(
        entry, forcing, card, ("si3_o15_step", "si3_o16_step")
    )
    o15_args, o15_kw, o15_out_recorded = recorded["si3_o15_step"]
    o16_args, o16_kw, o16_out_recorded = recorded["si3_o16_step"]
    o16_state = o16_args[0]
    o16_cfg = o16_kw["config"]
    o16_sss_m = o16_kw["sss_m"]
    eqv = state_mod.si3_var_glo2eqv(
        v_i=o16_state.v_i,
        v_s=o16_state.v_s,
        sv_i=o16_state.sv_i,
        a_i=o16_state.a_i,
        oa_i=o16_state.oa_i,
        a_ip=o16_state.a_ip,
        v_ip=o16_state.v_ip,
        v_il=o16_state.v_il,
        e_s=o16_state.e_s,
        e_i=o16_state.e_i,
        szv_i=o16_state.szv_i,
        kn=2,
        hi_max_top=o16_state.bounds.hi_max[-1],
        sss_m=o16_sss_m,
        config=o16_cfg,
    )
    var_agg_kwargs: dict[str, Any] = {
        "v_i": o16_state.v_i,
        "v_s": eqv.v_s,
        "sv_i": o16_state.sv_i,
        "a_i": eqv.a_i,
        "oa_i": o16_state.oa_i,
        "a_ip": o16_state.a_ip,
        "v_ip": o16_state.v_ip,
        "v_il": o16_state.v_il,
        "t_su": o16_state.t_su,
        "t_si": o16_state.t_si,
        "e_s": o16_state.e_s,
        "e_i": o16_state.e_i,
        "szv_i": o16_state.szv_i,
        "a_ip_eff": eqv.a_ip_eff,
        "t_i": eqv.t_i,
        "t_s": eqv.t_s,
        "kn": 2,
        "config": o16_cfg,
    }
    var_agg_keys = tuple(k for k in var_agg_kwargs if k not in ("kn", "config"))
    var_agg_diff = tuple(var_agg_kwargs[k] for k in var_agg_keys)

    def _call_var_agg(diff_tuple: tuple[Any, ...]) -> Any:
        call_kw = {k: v for k, v in zip(var_agg_keys, diff_tuple)}
        call_kw["kn"] = 2
        call_kw["config"] = o16_cfg
        return state_mod.si3_var_agg(**call_kw)

    def _var_agg_obj(diff_tuple: tuple[Any, ...]) -> jax.Array:
        return _sum_unique_floating_leaves(_call_var_agg(diff_tuple))

    # Forward bit parity on recorded inputs and negative-zero fixture
    agg_eager = _call_var_agg(var_agg_diff)
    agg_scan_jit = jax.jit(_call_var_agg)(var_agg_diff)
    orig_state_traced = state_mod._si3_traced
    try:
        state_mod._si3_traced = lambda *_: False
        agg_unrolled_jit = jax.jit(_call_var_agg)(var_agg_diff)
    finally:
        state_mod._si3_traced = orig_state_traced

    neg_zero_kw = build_negative_zero_var_agg_fixture(var_agg_kwargs)
    neg_zero_diff = tuple(neg_zero_kw[k] for k in var_agg_keys)
    input_neg_zeros = {
        k: int(np.count_nonzero(np.asarray(neg_zero_kw[k]).view(np.uint64) == np.uint64(0x8000000000000000)))
        for k in var_agg_keys
    }
    agg_neg_eager = _call_var_agg(neg_zero_diff)
    agg_neg_scan_jit = jax.jit(_call_var_agg)(neg_zero_diff)
    try:
        state_mod._si3_traced = lambda *_: False
        agg_neg_unrolled_jit = jax.jit(_call_var_agg)(neg_zero_diff)
    finally:
        state_mod._si3_traced = orig_state_traced
    output_neg_zero_fields = {
        name: int(
            np.count_nonzero(
                np.asarray(getattr(agg_neg_scan_jit, name)).view(np.uint64)
                == np.uint64(0x8000000000000000)
            )
        )
        for name in agg_neg_scan_jit._fields
        if int(
            np.count_nonzero(
                np.asarray(getattr(agg_neg_scan_jit, name)).view(np.uint64)
                == np.uint64(0x8000000000000000)
            )
        )
        > 0
    }

    parity = {
        "var_agg_recorded_eager_vs_scan_jit": _pytree_bit_stats(agg_eager, agg_scan_jit),
        "var_agg_recorded_unrolled_jit_vs_scan_jit": _pytree_bit_stats(
            agg_unrolled_jit, agg_scan_jit
        ),
        "var_agg_neg_zero_eager_vs_scan_jit": _pytree_bit_stats(agg_neg_eager, agg_neg_scan_jit),
        "var_agg_neg_zero_unrolled_jit_vs_scan_jit": _pytree_bit_stats(
            agg_neg_unrolled_jit, agg_neg_scan_jit
        ),
        "var_agg_neg_zero_input_counts": input_neg_zeros,
        "var_agg_neg_zero_surviving_output_fields": output_neg_zero_fields,
    }

    # Reverse-mode AD benchmarks: unrolled (compile-only) vs scan/traced (compile + execute)
    benchmarks: dict[str, Any] = {}
    try:
        state_mod._si3_traced = lambda *_: False
        benchmarks["si3_var_agg_kn2_unrolled"] = measure_reverse_compile_unit(
            "si3_var_agg_kn2_unrolled",
            _var_agg_obj,
            var_agg_diff,
            execute=False,
        )
    finally:
        state_mod._si3_traced = orig_state_traced

    benchmarks["si3_var_agg_kn2_scan"] = measure_reverse_compile_unit(
        "si3_var_agg_kn2_scan",
        _var_agg_obj,
        var_agg_diff,
        execute=execute_scan,
    )

    o16_leaves, o16_treedef = jax.tree_util.tree_flatten((o16_args, o16_kw))
    o16_mask = [_is_float_leaf(x) for x in o16_leaves]
    o16_diff = [x for x, m in zip(o16_leaves, o16_mask) if m]

    def _call_o16(diff_list: list[Any]) -> Any:
        it = iter(diff_list)
        r_args, r_kw = jax.tree_util.tree_unflatten(
            o16_treedef, [next(it) if m else v for v, m in zip(o16_leaves, o16_mask)]
        )
        return finalize_mod.si3_o16_step(*r_args, **r_kw)

    def _o16_obj(diff_list: list[Any]) -> jax.Array:
        return _sum_unique_floating_leaves(_call_o16(diff_list))

    o16_scan_jit_out = jax.jit(_call_o16)(o16_diff)
    parity["si3_o16_step_recorded_eager_vs_scan_jit"] = _pytree_bit_stats(
        o16_out_recorded, o16_scan_jit_out
    )

    try:
        state_mod._si3_traced = lambda *_: False
        benchmarks["si3_o16_step_unrolled"] = measure_reverse_compile_unit(
            "si3_o16_step_unrolled",
            _o16_obj,
            o16_diff,
            execute=False,
        )
    finally:
        state_mod._si3_traced = orig_state_traced

    benchmarks["si3_o16_step_scan"] = measure_reverse_compile_unit(
        "si3_o16_step_scan",
        _o16_obj,
        o16_diff,
        execute=execute_scan,
    )

    if include_o15:
        o15_leaves, o15_treedef = jax.tree_util.tree_flatten((o15_args, o15_kw))
        o15_mask = [_is_float_leaf(x) for x in o15_leaves]
        o15_diff = [x for x, m in zip(o15_leaves, o15_mask) if m]

        def _call_o15(diff_list: list[Any]) -> Any:
            it = iter(diff_list)
            r_args, r_kw = jax.tree_util.tree_unflatten(
                o15_treedef, [next(it) if m else v for v, m in zip(o15_leaves, o15_mask)]
            )
            return finalize_mod.si3_o15_step(*r_args, **r_kw)

        def _o15_obj(diff_list: list[Any]) -> jax.Array:
            return _sum_unique_floating_leaves(_call_o15(diff_list))

        o15_traced_jit_out = jax.jit(_call_o15)(o15_diff)
        parity["si3_o15_step_recorded_eager_vs_traced_jit"] = _pytree_bit_stats(
            o15_out_recorded, o15_traced_jit_out
        )
        orig_finalize_traced = finalize_mod._si3_traced
        try:
            finalize_mod._si3_traced = lambda *_: False
            benchmarks["si3_o15_step_unrolled"] = measure_reverse_compile_unit(
                "si3_o15_step_unrolled",
                _o15_obj,
                o15_diff,
                execute=False,
            )
        finally:
            finalize_mod._si3_traced = orig_finalize_traced

        benchmarks["si3_o15_step_traced"] = measure_reverse_compile_unit(
            "si3_o15_step_traced",
            _o15_obj,
            o15_diff,
            execute=execute_scan,
        )

        # Codex sub-candidate measurements on recorded O15/O16 inputs:
        # 1. O15 _ice_itd_reb (si3_finalize.py:440-594)
        o15_state, o15_fluxes = o15_args
        o15_cfg = o15_kw["config"]
        o15_sss_m = o15_kw["sss_m"]
        o15_amax = o15_kw["rn_amax_2d"]

        def _reb_obj(x: tuple[Any, Any]) -> jax.Array:
            return _sum_unique_floating_leaves(finalize_mod._ice_itd_reb(x[0], x[1], o15_cfg))

        try:
            finalize_mod._si3_traced = lambda *_: False
            benchmarks["o15_itd_reb_unrolled"] = measure_reverse_compile_unit(
                "o15_itd_reb_unrolled",
                _reb_obj,
                (o15_state, o15_amax),
                execute=execute_scan,
            )
        finally:
            finalize_mod._si3_traced = orig_finalize_traced

        benchmarks["o15_itd_reb_traced"] = measure_reverse_compile_unit(
            "o15_itd_reb_traced",
            _reb_obj,
            (o15_state, o15_amax),
            execute=execute_scan,
        )

        # 2. O15 50-step salinity bounds loop (si3_finalize.py:665-720 / itd.py:1352-1380)
        import legoesm.constants as canonical_constants
        from legoesm.ice.itd import SI3IceCorConfig, si3_ice_cor_salinity_bounds

        cor_cfg = SI3IceCorConfig(
            rn_himin=float(o15_kw["rn_himin"]),
            rn_simin=float(o15_cfg.rn_simin),
            rn_sinew=float(o15_cfg.rn_sinew),
            nn_icesal=int(o15_cfg.nn_icesal),
            ln_pnd=bool(o15_cfg.ln_pnd),
            ln_pnd_lids=bool(o15_cfg.ln_pnd_lids),
            rdt_ice=float(o15_cfg.rDt_ice),
            rhoi=float(o15_cfg.ice_constants.rho_ice),
            rhos=float(o15_cfg.ice_constants.rho_snow),
            rhow=float(canonical_constants.rho_water),
            rt0=float(o15_cfg.ice_constants.T0),
        )
        jpi_c, jpj_c, _ = o15_state.a_i.shape
        h_c = int(o15_cfg.nn_hls)
        sl_c = (slice(h_c, -h_c), slice(h_c, -h_c)) if h_c else (slice(None), slice(None))
        interior_c = jnp.zeros((jpi_c, jpj_c), dtype=jnp.bool_).at[sl_c].set(True)

        from legoesm.core.source_rounding import nemo_source_round as sr

        def _sal_unrolled_obj(x: tuple[Any, Any, Any, Any]) -> jax.Array:
            a_i_in, v_i_in, szv_i_in, sfx_in = x
            dtype = a_i_in.dtype
            r1_nlay_i = sr(jnp.asarray(1.0 / float(o15_cfg.nlay_i), dtype=dtype))
            r1_dt = sr(jnp.asarray(1.0 / float(o15_cfg.rDt_ice), dtype=dtype))
            rhoi = jnp.asarray(o15_cfg.ice_constants.rho_ice, dtype=dtype)
            sss = jnp.asarray(o15_sss_m, dtype=dtype)
            szv_i = szv_i_in
            sfx_res = sfx_in
            for jl in range(o15_cfg.jpl):
                for jk in range(o15_cfg.nlay_i):
                    old = szv_i[..., jk, jl]
                    lower = sr(sr(o15_cfg.rn_simin * v_i_in[..., jl]) * r1_nlay_i)
                    upper = sr(
                        sr(sr(sr(o15_cfg.rn_sinew * sss) * v_i_in[..., jl]) * r1_nlay_i)
                    )
                    bounded_low = jnp.where(lower > old, lower, old)
                    bounded = jnp.where(upper < bounded_low, upper, bounded_low)
                    bounded = jnp.where(interior_c, bounded, old)
                    szv_i = szv_i.at[..., jk, jl].set(bounded)
                    delta = sr(bounded - old)
                    correction = sr(sr(delta * rhoi) * r1_dt)
                    sfx_res = jnp.where(interior_c, sr(sfx_res - correction), sfx_res)
            return _sum_unique_floating_leaves((szv_i, sfx_res))

        def _sal_traced_obj(x: tuple[Any, Any, Any, Any]) -> jax.Array:
            out = si3_ice_cor_salinity_bounds(
                {"a_i": x[0], "v_i": x[1], "szv_i": x[2], "sfx_res": x[3]},
                interior=interior_c,
                sss_m=o15_sss_m,
                cfg=cor_cfg,
                traced=True,
            )
            return _sum_unique_floating_leaves((out["szv_i"], out["sfx_res"]))

        sal_inputs = (o15_state.a_i, o15_state.v_i, o15_state.szv_i, o15_fluxes.sfx_res)
        benchmarks["o15_salinity_50step_unrolled"] = measure_reverse_compile_unit(
            "o15_salinity_50step_unrolled",
            _sal_unrolled_obj,
            sal_inputs,
            execute=execute_scan,
        )
        benchmarks["o15_salinity_50step_traced"] = measure_reverse_compile_unit(
            "o15_salinity_50step_traced",
            _sal_traced_obj,
            sal_inputs,
            execute=execute_scan,
        )

        # 3. O12 10-step pond permeability loop (si3_thd_pnd.py:247-260)
        from legoesm.core.source_rounding import nemo_source_round as sr

        def _o12_perm_10step_obj(x: tuple[Any, Any]) -> jax.Array:
            t_i_cat, sz_i_cat = x
            zt1 = sr(t_i_cat - float(o16_cfg.ice_constants.T0))
            zs1 = sz_i_cat
            zperm_layers = []
            for jk in range(o16_cfg.nlay_i):
                zt_k = zt1[:, :, jk]
                zs_k = zs1[:, :, jk]
                zfac = jnp.where(zt_k < -2.0, -5.0, -18.0)
                zb_k = jnp.maximum(0.0, jnp.minimum(1.0, sr(sr(zs_k / jnp.minimum(-0.01, zt_k)) / zfac)))
                zperm_layers.append(sr(3.0e-8 * jnp.power(zb_k, 3.0)))
            zperm = zperm_layers[0]
            for jk in range(1, o16_cfg.nlay_i):
                zperm = jnp.minimum(zperm, zperm_layers[jk])
            return jnp.sum(zperm)

        benchmarks["o12_pnd_10step_perm_loop"] = measure_reverse_compile_unit(
            "o12_pnd_10step_perm_loop",
            _o12_perm_10step_obj,
            (eqv.t_i[:, :, :, 0], eqv.sz_i[:, :, :, 0]),
            execute=execute_scan,
        )

        # 4. 10-step aggregation sums (si3_orca2_state.py:768-777, 997-998)
        benchmarks["fortran_sum_categories_10step"] = measure_reverse_compile_unit(
            "fortran_sum_categories_10step",
            lambda e_arr: jnp.sum(
                state_mod.fortran_sum_categories(
                    state_mod.fortran_sum_categories(e_arr, axis=-1), axis=-1
                )
            ),
            o16_state.e_i,
            execute=execute_scan,
        )

        def _glo2eqv_si_10step_obj(sz_i_in: Any) -> jax.Array:
            r1_nlay_i = sr(jnp.asarray(1.0 / float(o16_cfg.nlay_i), dtype=sz_i_in.dtype))
            s_i_acc = jnp.zeros_like(o16_state.v_i)
            for jk in range(o16_cfg.nlay_i):
                s_i_acc = sr(s_i_acc + sr(sz_i_in[:, :, jk, :] * r1_nlay_i))
            return jnp.sum(s_i_acc)

        benchmarks["glo2eqv_s_i_sum_10step"] = measure_reverse_compile_unit(
            "glo2eqv_s_i_sum_10step",
            _glo2eqv_si_10step_obj,
            eqv.sz_i,
            execute=execute_scan,
        )

    all_parity_ok = (
        parity["var_agg_recorded_eager_vs_scan_jit"]["bit_mismatches"] == 0
        and parity["var_agg_recorded_unrolled_jit_vs_scan_jit"]["bit_mismatches"] == 0
        and parity["var_agg_neg_zero_eager_vs_scan_jit"]["bit_mismatches"] == 0
        and parity["var_agg_neg_zero_unrolled_jit_vs_scan_jit"]["bit_mismatches"] == 0
        and parity["var_agg_neg_zero_eager_vs_scan_jit"]["negative_zeros_cand"] > 0
        and all(v > 0 for v in input_neg_zeros.values())
        and parity["si3_o16_step_recorded_eager_vs_scan_jit"]["bit_mismatches"] == 0
        and (
            not include_o15
            or parity["si3_o15_step_recorded_eager_vs_traced_jit"]["bit_mismatches"] == 0
        )
    )
    report = {
        "status": "PASS" if all_parity_ok else "FAIL",
        "provenance": _provenance(),
        "eager_interception_seconds": round(eager_s, 3),
        "parity": parity,
        "benchmarks": benchmarks,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _parse_int_list(spec: str) -> list[int]:
    items: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            items.extend(range(int(a), int(b) + 1))
        else:
            items.append(int(part))
    return items


def _parse_slice_list(spec: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.append((int(a), int(b)))
        elif ":" in part:
            a, b = part.split(":", 1)
            out.append((int(a), int(b)))
        else:
            k = int(part)
            out.append((k, k))
    return out


def eager_jit_probe(master: Path, shape: tuple[int, int] = (184, 152)) -> dict:
    """One production-card O17 step (100 aEVP subcycles) eager and compiled, bit-compared.

    The entry is the mixed-regime fixture laid out on the real ORCA2 rank grid
    (jpi=184, jpj=152 from Ni0glo=180, Nj0glo=148, nn_hls=2), with its fold LBC
    applied, so ice-covered, open-water and nonzero-below-epsi10 columns all occur.
    """
    entry, forcing, card = _case(master, shape)
    h = card.state.nn_hls
    conc = np.asarray(entry.ice.a_i)[h:-h, h:-h].sum(axis=-1)
    dynamic = tuple(forcing[1:])

    def step(step_entry, live_dynamic):
        return si3_o17_step(step_entry, SI3O17LiveForcing(1, *live_dynamic), card)

    started = time.monotonic()
    with jax.disable_jit():
        eager = step(entry, dynamic)
    jax.block_until_ready(eager)
    eager_seconds = time.monotonic() - started
    eager_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    started = time.monotonic()
    executable = jax.jit(step).lower(entry, dynamic).compile()
    compile_seconds = time.monotonic() - started
    started = time.monotonic()
    compiled = executable(entry, dynamic)
    jax.block_until_ready(compiled)
    execute_seconds = time.monotonic() - started
    mismatched, leaves, nonfinite = [], 0, []
    for (path, a), (_, b) in zip(
        jax.tree_util.tree_flatten_with_path(eager)[0],
        jax.tree_util.tree_flatten_with_path(compiled)[0],
        strict=True,
    ):
        a, b = np.asarray(a), np.asarray(b)
        leaves += 1
        bits = (a.view(np.uint64) != b.view(np.uint64)) if a.dtype == np.float64 else (a != b)
        if a.shape != b.shape or np.count_nonzero(bits):
            signed_zero_only = bool(a.dtype == np.float64 and np.all(a[bits] == 0.0) and np.all(b[bits] == 0.0))
            mismatched.append({
                "leaf": jax.tree_util.keystr(path), "count": int(np.count_nonzero(bits)),
                "signed_zero_only": signed_zero_only,
                "eager_negative_zero": int(np.count_nonzero(np.signbit(a[bits]))) if signed_zero_only else None,
                "jit_negative_zero": int(np.count_nonzero(np.signbit(b[bits]))) if signed_zero_only else None,
            })
        if a.dtype.kind == "f" and not np.isfinite(a).all():
            nonfinite.append(jax.tree_util.keystr(path))
    return {
        "status": "PASS" if not mismatched and not nonfinite else "FAIL",
        "full_shape": list(shape),
        "owned_shape": [shape[0] - 2 * h, shape[1] - 2 * h],
        "aevp_subcycles": card.aevp.n_subcycles,
        "owned_columns": {
            "ice_covered": int(np.count_nonzero(conc > 1.0e-10)),
            "ice_free": int(np.count_nonzero(conc == 0.0)),
            "near_empty_le_epsi10": int(np.count_nonzero((conc > 0.0) & (conc <= 1.0e-10))),
        },
        "leaves_compared": leaves,
        "bitwise_mismatched_leaves": mismatched,
        "all_mismatches_signed_zero_only": all(row["signed_zero_only"] for row in mismatched),
        "nonfinite_leaves": nonfinite,
        "eager_seconds": eager_seconds,
        "eager_peak_rss_kib": eager_rss,
        "jit_compile_seconds": compile_seconds,
        "jit_execute_seconds": execute_seconds,
        "process_peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execution-mode", choices=("compiled", "eager"), default="compiled")
    parser.add_argument("--eager-jit-probe", action="store_true")
    parser.add_argument(
        "--prefixes",
        type=str,
        default=None,
        help="Comma/range list of prefix lengths k in 1..14 (e.g. '1-5' or '1,2,3').",
    )
    parser.add_argument(
        "--slices",
        type=str,
        default=None,
        help="Comma list of rung slices/pairs k1-k2 in 1..14 (e.g. '3-4,8-9,8-14').",
    )
    parser.add_argument(
        "--isolated-funcs",
        type=str,
        default=None,
        help="Comma list of single-call function names in si3_o17_step to compile in isolation.",
    )
    parser.add_argument(
        "--glue-only",
        action="store_true",
        help="Mock the 10 D198 rungs to measure reverse compile of glue/halos/packing/outer step only.",
    )
    parser.add_argument(
        "--checkpoint-rungs",
        type=str,
        default=None,
        help="Optional comma/range list of rung indices to wrap in jax.checkpoint during slice runs.",
    )
    parser.add_argument(
        "--xla-pass-log",
        type=Path,
        default=None,
        help="Optional scratch file path to capture C++ XLA pass timings (never under /tmp).",
    )
    parser.add_argument("--vmodule", dest="vmodule", type=str, default=None)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument(
        "--reproduce-d198r5",
        action="store_true",
        help="Run the committed DECISION 198 round 5 reproducer for si3_var_agg(kn=2), si3_o16_step, and si3_o15_step.",
    )
    args = parser.parse_args(argv)

    is_diag = any(
        (
            args.prefixes is not None,
            args.slices is not None,
            args.isolated_funcs is not None,
            args.glue_only,
        )
    )
    if args.eager_jit_probe:
        report = eager_jit_probe(args.master)
    elif args.reproduce_d198r5:
        report = reproduce_d198_r5_var_agg_and_o16(
            args.master,
            execute_scan=not args.compile_only,
            include_o15=True,
            output_path=args.output,
        )
    elif is_diag:
        prefixes = _parse_int_list(args.prefixes) if args.prefixes else []
        slices = _parse_slice_list(args.slices) if args.slices else []
        isolated = (
            [s.strip() for s in args.isolated_funcs.split(",") if s.strip()]
            if args.isolated_funcs
            else []
        )
        ckpt = _parse_int_list(args.checkpoint_rungs) if args.checkpoint_rungs else []
        report = diagnose_slices(
            args.master,
            prefixes=prefixes,
            slices=slices,
            isolated_funcs=isolated,
            glue_only=args.glue_only,
            checkpoint_rungs=ckpt,
            xla_pass_log=args.xla_pass_log,
            execute=not args.compile_only,
            output_path=args.output,
        )
    else:
        report = probe(
            args.master,
            execution_mode=args.execution_mode,
            xla_pass_log=args.xla_pass_log,
        )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
