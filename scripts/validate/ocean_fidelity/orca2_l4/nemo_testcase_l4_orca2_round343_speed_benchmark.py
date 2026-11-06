#!/usr/bin/env python3
"""ORCA2 standalone K0 speed & 8-day accuracy benchmark.

Measures execution speed (compile time, steady-state s/step over >= 20 steps
after warm-up, host/GPU memory, and projected 1-year / 2,920-step wall-clock)
and 8-day (64-step) SST/T accuracy cost against the exact-mode daily snapshots
from Round 342 across four arms on the standalone ORCA2 K0 composed card:

  (a) exact_cpu_fp64  : Exact NEMO mode on CPU (float64, transcendentals='libm')
  (b) normal_cpu_fp64 : Normal mode on CPU (float64, transcendentals='native')
  (c) normal_gpu_fp64 : Normal mode on 1 idle GPU (float64, transcendentals='native')
  (d) normal_gpu_fp32 : Normal mode on 1 idle GPU (float32, transcendentals='native')

Round 344 reuses the same implementation for native-transcendental
fast-literal fp64 CPU/GPU arms while retaining the round-343 interface.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import resource
import shutil
import subprocess
import sys
import time
import warnings
from typing import Any, Mapping, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import netCDF4
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for _pkg in reversed((REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean")):
    if str(_pkg) not in sys.path:
        sys.path.insert(0, str(_pkg))

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy  # noqa: E402
from legoesm.ocean import bulk_flux_omip as bulk  # noqa: E402
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (  # noqa: E402
    LatLonCGridOceanModel,
)
from legoesm.ocean.fidelity.nemo_testcase_recipe import (  # noqa: E402
    build_orca2_damping_for_step,
)
from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402
from legoesm.ocean.forcing.nemo_surface import NemoSurfaceMeans  # noqa: E402
from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig  # noqa: E402
from legoesm.ocean.physics.lateral_mixing.mle import MLEConfig  # noqa: E402
from legoesm.ocean.physics.vertical_mixing.config import TKEConfig  # noqa: E402
from legoesm.ocean.physics.vertical_mixing.double_diffusion import (  # noqa: E402
    DoubleDiffusionConfig,
)
from legoesm.ocean.physics.vertical_mixing.internal_wave_mixing import (  # noqa: E402
    IWMConfig,
)
from legoesm.ocean.state import (  # noqa: E402
    BarotropicConfig,
    LatLonCGridOceanConfig,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_o1_numerical_gate as o1,
    nemo_testcase_l4_orca2_round1_ladder_gate as ladder,
    nemo_testcase_l4_orca2_round339_common as common,
    nemo_testcase_l4_orca2_round340_standalone_gate as r340_s4,
)
from scripts.bench.sample_cpu_cores import (  # noqa: E402
    CLK_TCK,
    parse_cpu_ticks,
)
from scripts.validate.fv3_native.summarize_perfetto_trace import (  # noqa: E402
    load_perfetto_events,
    perfetto_thread_names,
)

GateError = common.GateError
require = common.require
sha256_file = common.sha256

DEFAULT_DECK_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round5/"
    "acquisition/orca1ice_surface_entry_every_step_a_np2"
)
DEFAULT_CORE_MANIFEST = DEFAULT_DECK_ROOT / "input_files.sha256"
DEFAULT_ICE_COVER_PATH = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round339/"
    "ice_cover_producer/orca2_full_si3_year1_np2/ice_cover_clim.nc"
)
DEFAULT_EXACT_REF_DIR = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r342/"
    "legoesm_year1_2920step"
)
DEFAULT_OUTPUT_DIR = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r343"
)
DEFAULT_ROUND343_REPORT = DEFAULT_OUTPUT_DIR / "orca2_round343_speed_benchmark_report.json"

YEAR_STEPS = 2920
STEPS_PER_DAY = 8
WARMUP_STEPS = 2
MIN_STEADY_STEPS = 20
EXACT_BENCHMARK_STEPS = WARMUP_STEPS + MIN_STEADY_STEPS  # 22 steps (kt=1..22)
NORMAL_BENCHMARK_STEPS = 8 * STEPS_PER_DAY               # 64 steps (8 days)
DAILY_CHECK_STEPS_8D = tuple(range(STEPS_PER_DAY, NORMAL_BENCHMARK_STEPS + 1, STEPS_PER_DAY))
DAILY_CHECK_STEPS_EXACT = (8, 16)

MONTH_DAYS = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=np.int64)
MONTH_END_SEC = (np.cumsum(MONTH_DAYS) * 86400).astype(np.int64)
MONTH_CENTRES_SEC = ((np.cumsum(MONTH_DAYS) - MONTH_DAYS / 2) * 86400).astype(np.int64)
EXTENDED_MONTH_CENTRES_SEC = np.concatenate((
    [MONTH_CENTRES_SEC[-1] - 365 * 86400],
    MONTH_CENTRES_SEC,
    [MONTH_CENTRES_SEC[0] + 365 * 86400],
))

ARM_IDS = (
    "exact_cpu_fp64",
    "normal_cpu_fp64",
    "normal_gpu_fp64",
    "normal_gpu_fp32",
)
FAST_LITERAL_ARM_IDS = (
    "fast_literal_cpu_fp64",
    "fast_literal_gpu_fp64",
)

ROUND349_CORE_COUNTS = (1, 2, 4, 8, 16, 40)
ROUND349_MODES = ("exact", "fast_literal")
ROUND349_FULL_STEPS = 18
ROUND349_STEADY_STEPS = 16
ROUND349_SMOKE_STEPS = 4
ROUND349_PROFILE_STEP = 3
ROUND349_PROFILE_ANNOTATION = "orca2_round349_profile_step"
ROUND349_SURFACE_ANNOTATION = "orca2_round349_surface_eager"
# Guards the selected physical cores and their siblings from background load.
ROUND349_PINNED_CORE_BUSY_FRACTION = 0.10
# Guards the quiet-machine sample long enough to expose sustained CPU load.
ROUND349_PREFLIGHT_SAMPLE_SECONDS = 10.0
# Guards the benchmark from any unrelated process occupying one full core.
ROUND349_OTHER_PROCESS_BUSY_CORES = 1.0
ROUND349_JAXLIB_CPU_CLIENT_SOURCE = (
    "/home/dbalwada/legoESM/.venv/lib/python3.13/site-packages/"
    "jaxlib/xla_client.py:64-87"
)
ROUND349_SCRATCH_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/orca2_r349_scratch"
)
DEFAULT_ROUND349_OUTPUT_DIR = ROUND349_SCRATCH_ROOT / "legoesm_cpu_sweep"
DEFAULT_ROUND349_NEMO_SOURCE = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round342/"
    "s4_year/orca2_k0_year1_a_np2"
)
ROUND349_REPORT_NAME = "orca2_round349_cpu_benchmark_report.json"
ROUND349_REPORT_MARKDOWN_NAME = "orca2_round349_cpu_benchmark_report.md"


class CpuTopologyRow(NamedTuple):
    """One online logical CPU and its physical-core identity."""

    cpu_id: int
    core_id: int
    package_id: int
    siblings: tuple[int, ...]


class ProcessCpuSample(NamedTuple):
    """CPU ticks, executable name, and immediate parent for one process."""

    ticks: int
    command: str
    parent_pid: int


def parse_cpu_id_list(value: str) -> tuple[int, ...]:
    """Parse Linux CPU-list syntax (``0,2-4``) into unique sorted IDs."""
    ids: list[int] = []
    for token in value.split(","):
        token = token.strip()
        require(bool(token), f"empty CPU-list token in {value!r}")
        if "-" in token:
            lo_s, hi_s = token.split("-", maxsplit=1)
            lo, hi = int(lo_s), int(hi_s)
            require(0 <= lo <= hi, f"invalid CPU range {token!r}")
            ids.extend(range(lo, hi + 1))
        else:
            cpu = int(token)
            require(cpu >= 0, f"negative CPU id {cpu}")
            ids.append(cpu)
    require(len(ids) == len(set(ids)), f"duplicate CPU id in {value!r}")
    return tuple(sorted(ids))


def format_cpu_id_list(cpu_ids: Sequence[int]) -> str:
    """Return a stable comma-separated CPU list accepted by taskset/Open MPI."""
    values = tuple(int(cpu) for cpu in cpu_ids)
    require(bool(values), "CPU id list must not be empty")
    require(len(values) == len(set(values)), f"duplicate CPU ids: {values}")
    require(all(cpu >= 0 for cpu in values), f"negative CPU id in {values}")
    return ",".join(str(cpu) for cpu in values)


def read_cpu_topology(
    sysfs_root: Path = Path("/sys/devices/system/cpu"),
) -> tuple[CpuTopologyRow, ...]:
    """Read online logical CPUs and physical-core identities from Linux sysfs."""
    rows: list[CpuTopologyRow] = []
    for cpu_dir in sorted(
        sysfs_root.glob("cpu[0-9]*"), key=lambda path: int(path.name[3:])
    ):
        cpu_id = int(cpu_dir.name[3:])
        online_path = cpu_dir / "online"
        if online_path.is_file() and online_path.read_text(encoding="utf-8").strip() != "1":
            continue
        topo = cpu_dir / "topology"
        core_id = int((topo / "core_id").read_text(encoding="utf-8").strip())
        package_id = int(
            (topo / "physical_package_id").read_text(encoding="utf-8").strip()
        )
        siblings = parse_cpu_id_list(
            (topo / "thread_siblings_list").read_text(encoding="utf-8").strip()
        )
        rows.append(CpuTopologyRow(cpu_id, core_id, package_id, siblings))
    require(bool(rows), f"no online CPUs found below {sysfs_root}")
    return tuple(rows)


def select_physical_cpu_ids(
    n_cores: int,
    topology: Sequence[CpuTopologyRow] | None = None,
) -> tuple[int, ...]:
    """Select the lowest logical representative of each physical core."""
    require(n_cores > 0, f"n_cores must be positive, got {n_cores}")
    rows = tuple(read_cpu_topology() if topology is None else topology)
    representatives: dict[tuple[int, int], CpuTopologyRow] = {}
    for row in rows:
        key = (row.package_id, row.core_id)
        current = representatives.get(key)
        if current is None or row.cpu_id < current.cpu_id:
            representatives[key] = row
    selected_rows = sorted(representatives.values(), key=lambda row: row.cpu_id)[:n_cores]
    require(
        len(selected_rows) == n_cores,
        f"requested {n_cores} physical cores but found {len(representatives)}",
    )
    selected = tuple(row.cpu_id for row in selected_rows)
    for index, row in enumerate(selected_rows):
        overlap = set(row.siblings).intersection(selected)
        require(
            overlap == {row.cpu_id},
            f"selected CPUs share a physical core: cpu={row.cpu_id}, "
            f"siblings={row.siblings}, selected={selected}",
        )
        require(
            (row.package_id, row.core_id)
            not in {(other.package_id, other.core_id) for other in selected_rows[:index]},
            f"duplicate physical core selected for cpu={row.cpu_id}",
        )
    return selected


def read_thread_cpu_ticks(pid: int | str = "self") -> dict[int, int]:
    """Read per-thread user+system CPU ticks for a process."""
    ticks: dict[int, int] = {}
    task_root = Path(f"/proc/{pid}/task")
    for task_dir in task_root.iterdir():
        if not task_dir.name.isdigit():
            continue
        try:
            user_ticks, system_ticks = parse_cpu_ticks(
                (task_dir / "stat").read_text(encoding="utf-8")
            )
        except (FileNotFoundError, ProcessLookupError):
            continue
        ticks[int(task_dir.name)] = user_ticks + system_ticks
    require(bool(ticks), f"no thread CPU counters found below {task_root}")
    return ticks


def read_process_total_cpu_ticks(pid: int | str = "self") -> int:
    """Read total process user+system CPU ticks from ``/proc/<pid>/stat``."""
    user_ticks, system_ticks = parse_cpu_ticks(
        Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    )
    return user_ticks + system_ticks


def read_thread_affinity_record(
    expected_cpu_ids: Sequence[int], pid: int | str = "self"
) -> dict[str, Any]:
    """Record every live thread name and affinity; refuse any affinity escape."""
    expected = tuple(sorted(int(cpu) for cpu in expected_cpu_ids))
    require(bool(expected), "expected thread affinity must not be empty")
    task_root = Path(f"/proc/{pid}/task")
    threads: list[dict[str, Any]] = []
    for task_dir in sorted(
        task_root.iterdir(), key=lambda path: int(path.name) if path.name.isdigit() else -1
    ):
        if not task_dir.name.isdigit():
            continue
        tid = int(task_dir.name)
        try:
            name = (task_dir / "comm").read_text(encoding="utf-8").strip()
            affinity = tuple(sorted(os.sched_getaffinity(tid)))
        except (FileNotFoundError, ProcessLookupError):
            continue
        threads.append({"tid": tid, "name": name, "affinity": list(affinity)})
    require(bool(threads), f"no thread affinity records found below {task_root}")
    mismatches = [row for row in threads if tuple(row["affinity"]) != expected]
    require(
        not mismatches,
        f"thread affinity differs from selected CPU set {expected}: {mismatches}",
    )
    name_counts: dict[str, int] = {}
    for row in threads:
        name = str(row["name"])
        name_counts[name] = name_counts.get(name, 0) + 1
    main_tid = int(pid) if isinstance(pid, int) else os.getpid()
    return {
        "expected_cpu_ids": list(expected),
        "affinity_exact": True,
        "os_thread_count": len(threads),
        "tf_XLAEigen_thread_count": name_counts.get("tf_XLAEigen", 0),
        "other_worker_thread_count": sum(
            1 for row in threads if row["tid"] != main_tid and row["name"] != "tf_XLAEigen"
        ),
        "thread_name_counts": dict(sorted(name_counts.items())),
        "threads": threads,
    }


def summarize_thread_cpu_ticks(
    before: Mapping[int, int],
    after: Mapping[int, int],
    wall_s: float,
    *,
    process_ticks_before: int,
    process_ticks_after: int,
) -> dict[str, Any]:
    """Report total busy cores and every thread's CPU seconds, without a cutoff."""
    require(np.isfinite(wall_s) and wall_s > 0.0, f"invalid wall_s={wall_s}")
    cpu_seconds = {
        int(tid): max(0, int(ticks) - int(before.get(tid, 0))) / CLK_TCK
        for tid, ticks in after.items()
    }
    total_process_cpu_seconds = (
        max(0, int(process_ticks_after) - int(process_ticks_before)) / CLK_TCK
    )
    measured_cores = total_process_cpu_seconds / wall_s
    ordered = sorted(cpu_seconds.items(), key=lambda item: (-item[1], item[0]))
    return {
        "cores_busy": float(measured_cores),
        "os_threads_after": len(after),
        "total_process_cpu_seconds": float(total_process_cpu_seconds),
        "thread_cpu_seconds": [
            {"tid": tid, "cpu_seconds": float(value)} for tid, value in ordered
        ],
    }


class SwitchSpec(NamedTuple):
    """Inventory entry for a switch distinguishing exact-NEMO mode from normal mode."""

    name: str
    owner: str
    path_in_card: str
    category: str
    exact_value: Any
    normal_value: Any
    switched_in_normal_mode: bool
    exact_citation: str
    normal_citation: str
    notes: str


def build_orca2_exact_vs_normal_switch_inventory() -> tuple[SwitchSpec, ...]:
    """Return the complete code-sourced inventory of exact-mode switches on ORCA2 K0."""
    return (
        # --- Category A: Hard libm-guarded switches (6) ---
        SwitchSpec(
            name="PrecisionPolicy.transcendentals",
            owner="PrecisionPolicy",
            path_in_card="policy.transcendentals",
            category="A_libm_guarded",
            exact_value="libm",
            normal_value="native",
            switched_in_normal_mode=True,
            exact_citation="packages/core/legoesm/core/precision.py:117-125",
            normal_citation="packages/core/legoesm/core/precision.py:103-114",
            notes=(
                "libm dispatches tx.exp/log/log10/sin/cos/tanh/atan/pow through "
                "jax.pure_callback + Python scalar ctypes loop over libm.so.6 on CPU "
                "(packages/core/legoesm/core/transcendentals.py:48-85); native uses "
                "vectorized XLA jnp primitives on CPU/GPU."
            ),
        ),
        SwitchSpec(
            name="NEMOTestcaseCard.transcendentals",
            owner="NEMOTestcaseCard",
            path_in_card="ocean.transcendentals",
            category="A_libm_guarded",
            exact_value="libm",
            normal_value="native",
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:66",
            normal_citation="packages/core/legoesm/core/precision.py:103",
            notes="Card metadata stamping the active transcendental backend.",
        ),
        SwitchSpec(
            name="tke_htau_evaluation",
            owner="TKEConfig",
            path_in_card="ocean.recipe.model_config.physics.vertical_mixing.tke.tke_htau_evaluation",
            category="A_libm_guarded",
            exact_value="nemo_scalar_libm",
            normal_value="jax_expression",
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2152",
            normal_citation="packages/ocean/legoesm/ocean/physics/vertical_mixing/config.py:658",
            notes=(
                "nemo_scalar_libm is guarded by _require_libm_policy "
                "(packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py:2708-2717,2740)."
            ),
        ),
        SwitchSpec(
            name="tke_etau_exponential_evaluation",
            owner="TKEConfig",
            path_in_card="ocean.recipe.model_config.physics.vertical_mixing.tke.tke_etau_exponential_evaluation",
            category="A_libm_guarded",
            exact_value="nemo_scalar_libm",
            normal_value="jax_expression",
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2153",
            normal_citation="packages/ocean/legoesm/ocean/physics/vertical_mixing/config.py:650",
            notes=(
                "nemo_scalar_libm is guarded by _require_libm_policy "
                "(packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py:2815-2817)."
            ),
        ),
        SwitchSpec(
            name="ddm.(enabled,form)",
            owner="DoubleDiffusionConfig",
            path_in_card="ocean.recipe.model_config.physics.vertical_mixing.ddm",
            category="A_libm_guarded",
            exact_value=(True, "nemo_literal"),
            normal_value=(False, "hybrid"),
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2174-2175",
            normal_citation="packages/ocean/legoesm/ocean/physics/vertical_mixing/double_diffusion.py:245-251",
            notes=(
                "form='nemo_literal' raises RuntimeError when transcendentals!='libm' "
                "(double_diffusion.py:325-329); enabled=True with form='hybrid' is "
                "refused by zdf_implicit_solver_evaluation='nemo_literal' "
                "(ocean_model_latlon_cgrid.py:14828-14832)."
            ),
        ),
        SwitchSpec(
            name="iwm.(form,tsdiff)",
            owner="IWMConfig",
            path_in_card="ocean.recipe.model_config.physics.vertical_mixing.iwm",
            category="A_libm_guarded",
            exact_value=("nemo_literal", True),
            normal_value=("generic", False),
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2167-2173",
            normal_citation="packages/ocean/legoesm/ocean/physics/vertical_mixing/internal_wave_mixing.py:187-203",
            notes=(
                "form='nemo_literal' raises RuntimeError when transcendentals!='libm' "
                "(internal_wave_mixing.py:411-415); k_profiles.py:582-592 requires "
                "tsdiff=False when form='generic'."
            ),
        ),
        # --- Category B: Source-rounding / operand-association / literal-order switches (36) ---
        SwitchSpec(
            name="nemo_bn2_source_rounded",
            owner="TKEConfig",
            path_in_card="ocean.recipe.model_config.physics.vertical_mixing.tke.nemo_bn2_source_rounded",
            category="B_source_rounding_or_order",
            exact_value=True,
            normal_value=False,
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2137",
            normal_citation="packages/ocean/legoesm/ocean/physics/vertical_mixing/config.py:675",
            notes="Statement-level source rounding in N^2 buoyancy frequency.",
        ),
        SwitchSpec(
            name="nemo_tke_solve_source_rounded",
            owner="TKEConfig",
            path_in_card="ocean.recipe.model_config.physics.vertical_mixing.tke.nemo_tke_solve_source_rounded",
            category="B_source_rounding_or_order",
            exact_value=True,
            normal_value=False,
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2139",
            normal_citation="packages/ocean/legoesm/ocean/physics/vertical_mixing/config.py:682",
            notes="Source rounding inside the TKE tridiagonal Thomas solve.",
        ),
        SwitchSpec(
            name="nemo_tke_matrix_rhs_source_rounded",
            owner="TKEConfig",
            path_in_card="ocean.recipe.model_config.physics.vertical_mixing.tke.nemo_tke_matrix_rhs_source_rounded",
            category="B_source_rounding_or_order",
            exact_value=True,
            normal_value=False,
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2141",
            normal_citation="packages/ocean/legoesm/ocean/physics/vertical_mixing/config.py:689",
            notes="Source rounding when assembling the TKE tridiagonal matrix and RHS.",
        ),
        SwitchSpec(
            name="nemo_sh2_r3_source_rounded",
            owner="TKEConfig",
            path_in_card="ocean.recipe.model_config.physics.vertical_mixing.tke.nemo_sh2_r3_source_rounded",
            category="B_source_rounding_or_order",
            exact_value=True,
            normal_value=False,
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2142",
            normal_citation="packages/ocean/legoesm/ocean/physics/vertical_mixing/config.py:693",
            notes="Source rounding in zdf_sh2 shear production.",
        ),
        SwitchSpec(
            name="nemo_sh2_r3_stored_reciprocal",
            owner="TKEConfig",
            path_in_card="ocean.recipe.model_config.physics.vertical_mixing.tke.nemo_sh2_r3_stored_reciprocal",
            category="B_source_rounding_or_order",
            exact_value=True,
            normal_value=False,
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2143",
            normal_citation="packages/ocean/legoesm/ocean/physics/vertical_mixing/config.py:697",
            notes="Multiply by stored reciprocal vs direct division in zdf_sh2.",
        ),
        SwitchSpec(
            name="tke_mxl_raw_evaluation",
            owner="TKEConfig",
            path_in_card="ocean.recipe.model_config.physics.vertical_mixing.tke.tke_mxl_raw_evaluation",
            category="B_source_rounding_or_order",
            exact_value="nemo_literal",
            normal_value="factored",
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2156",
            normal_citation="packages/ocean/legoesm/ocean/physics/vertical_mixing/config.py:662",
            notes="Mixing-length raw formula operand grouping.",
        ),
        SwitchSpec(
            name="tke_solver_evaluation",
            owner="TKEConfig",
            path_in_card="ocean.recipe.model_config.physics.vertical_mixing.tke.tke_solver_evaluation",
            category="B_source_rounding_or_order",
            exact_value="nemo_literal",
            normal_value="shared_thomas",
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:422",
            normal_citation="packages/ocean/legoesm/ocean/physics/vertical_mixing/config.py:642",
            notes="NEMO literal tridiagonal sweep vs shared Thomas solver.",
        ),
        SwitchSpec(
            name="tke_langmuir_evaluation",
            owner="TKEConfig",
            path_in_card="ocean.recipe.model_config.physics.vertical_mixing.tke.tke_langmuir_evaluation",
            category="B_source_rounding_or_order",
            exact_value="nemo_literal",
            normal_value="vectorized",
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:1946",
            normal_citation="packages/ocean/legoesm/ocean/physics/vertical_mixing/config.py:702",
            notes="Langmuir mixing-length loop vs vectorized evaluation.",
        ),
        SwitchSpec(
            name="slope_uv_source_rounding",
            owner="GMRediConfig",
            path_in_card="ocean.recipe.model_config.gm_redi.slope_uv_source_rounding",
            category="B_source_rounding_or_order",
            exact_value=True,
            normal_value=False,
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2934",
            normal_citation="packages/ocean/legoesm/ocean/physics/lateral_mixing/config.py:417",
            notes="Source rounding on U/V isoneutral slope components.",
        ),
        SwitchSpec(
            name="redi_a33_source_ordering",
            owner="GMRediConfig",
            path_in_card="ocean.recipe.model_config.gm_redi.redi_a33_source_ordering",
            category="B_source_rounding_or_order",
            exact_value=True,
            normal_value=False,
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2935",
            normal_citation="packages/ocean/legoesm/ocean/physics/lateral_mixing/config.py:489",
            notes="NEMO statement ordering in Redi K33 diagonal.",
        ),
        SwitchSpec(
            name="redi_a33_evaluation",
            owner="GMRediConfig",
            path_in_card="ocean.recipe.model_config.gm_redi.redi_a33_evaluation",
            category="B_source_rounding_or_order",
            exact_value="nemo_literal",
            normal_value="normalized_square",
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2912",
            normal_citation="packages/ocean/legoesm/ocean/physics/lateral_mixing/config.py:485",
            notes="Literal triad square vs normalized square in Redi K33.",
        ),
        SwitchSpec(
            name="redi_vertical_skew_evaluation",
            owner="GMRediConfig",
            path_in_card="ocean.recipe.model_config.gm_redi.redi_vertical_skew_evaluation",
            category="B_source_rounding_or_order",
            exact_value="nemo_literal",
            normal_value="normalized_sums",
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2911",
            normal_citation="packages/ocean/legoesm/ocean/physics/lateral_mixing/config.py:480",
            notes="Literal triad sums vs normalized sums in Redi vertical skew flux.",
        ),
        SwitchSpec(
            name="slope_metric_evaluation",
            owner="GMRediConfig",
            path_in_card="ocean.recipe.model_config.gm_redi.slope_metric_evaluation",
            category="B_source_rounding_or_order",
            exact_value="nemo_reciprocal",
            normal_value="division",
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2898",
            normal_citation="packages/ocean/legoesm/ocean/physics/lateral_mixing/config.py:412",
            notes="Stored metric reciprocal vs direct division in GM/Redi slopes.",
        ),
        SwitchSpec(
            name="mle.nemo_association",
            owner="MLEConfig",
            path_in_card="ocean.recipe.model_config.physics.mle.nemo_association",
            category="B_source_rounding_or_order",
            exact_value=True,
            normal_value=False,
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2990",
            normal_citation="packages/ocean/legoesm/ocean/physics/lateral_mixing/mle.py:121",
            notes="NEMO parenthesis association in Fox-Kemper MLE streamfunction.",
        ),
        SwitchSpec(
            name="nemo_r3t_stored_product",
            owner="ZStarCoordinate",
            path_in_card="ocean.recipe.z_coord.nemo_r3t_stored_product",
            category="B_source_rounding_or_order",
            exact_value=True,
            normal_value=False,
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2209",
            normal_citation="packages/ocean/legoesm/ocean/vertical.py:1531",
            notes="Source-rounded stored r3t stretch product in z* thickness.",
        ),
        SwitchSpec(
            name="nemo_eos_gdept_rounded_product",
            owner="LatLonCGridOceanConfig",
            path_in_card="ocean.recipe.model_config.nemo_eos_gdept_rounded_product",
            category="B_source_rounding_or_order",
            exact_value=True,
            normal_value=False,
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:1983",
            normal_citation="packages/ocean/legoesm/ocean/state.py:3224",
            notes="Source-rounded gdept_0 * (1 + r3t) product in EOS depth.",
        ),
        SwitchSpec(
            name="zdf_tracer_nemo_rounded_thickness",
            owner="LatLonCGridOceanConfig",
            path_in_card="ocean.recipe.model_config.zdf_tracer_nemo_rounded_thickness",
            category="B_source_rounding_or_order",
            exact_value=True,
            normal_value=False,
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:1976",
            normal_citation="packages/ocean/legoesm/ocean/state.py:3233",
            notes="Source-rounded layer thickness in implicit tracer vertical diffusion.",
        ),
        SwitchSpec(
            name="barotropic_rhs_ascending_level_sum",
            owner="LatLonCGridOceanConfig",
            path_in_card="ocean.recipe.model_config.barotropic_rhs_ascending_level_sum",
            category="B_source_rounding_or_order",
            exact_value=True,
            normal_value=False,
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:1977",
            normal_citation="packages/ocean/legoesm/ocean/state.py:3237",
            notes="Sequential ascending k=1..jpkm1 loop with barriers vs jnp.sum for barotropic RHS.",
        ),
        SwitchSpec(
            name="rk3_stage1_rhs_nemo_literal",
            owner="LatLonCGridOceanConfig",
            path_in_card="ocean.recipe.model_config.rk3_stage1_rhs_nemo_literal",
            category="B_source_rounding_or_order",
            exact_value=True,
            normal_value=False,
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:1978",
            normal_citation="packages/ocean/legoesm/ocean/state.py:3179",
            notes="Stage-1 momentum RHS accumulation in NEMO written order.",
        ),
        SwitchSpec(
            name="rk3_stage_velocity_written_order",
            owner="LatLonCGridOceanConfig",
            path_in_card="ocean.recipe.model_config.rk3_stage_velocity_written_order",
            category="B_source_rounding_or_order",
            exact_value=True,
            normal_value=False,
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:1979",
            normal_citation="packages/ocean/legoesm/ocean/state.py:3184",
            notes="Source-rounded written order for WS-RK3 stage velocity update.",
        ),
        SwitchSpec(
            name="rk3_stage_tracer_written_order",
            owner="LatLonCGridOceanConfig",
            path_in_card="ocean.recipe.model_config.rk3_stage_tracer_written_order",
            category="B_source_rounding_or_order",
            exact_value=True,
            normal_value=False,
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:1991",
            normal_citation="packages/ocean/legoesm/ocean/state.py:3264",
            notes="Source-rounded written order for WS-RK3 stage-1/2 tracer update.",
        ),
        SwitchSpec(
            name="rk3_stage_r3f_nemo_literal",
            owner="LatLonCGridOceanConfig",
            path_in_card="ocean.recipe.model_config.rk3_stage_r3f_nemo_literal",
            category="B_source_rounding_or_order",
            exact_value=True,
            normal_value=False,
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:1981",
            normal_citation="packages/ocean/legoesm/ocean/state.py:3214",
            notes="Interpolate stored r3fb/r3fa ratios vs ratio of interpolated SSH.",
        ),
        SwitchSpec(
            name="rk3_stage_r3t_nemo_literal",
            owner="LatLonCGridOceanConfig",
            path_in_card="ocean.recipe.model_config.rk3_stage_r3t_nemo_literal",
            category="B_source_rounding_or_order",
            exact_value=True,
            normal_value=False,
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:1982",
            normal_citation="packages/ocean/legoesm/ocean/state.py:3219",
            notes="Interpolate stored r3t ratios vs ratio of interpolated stage SSH.",
        ),
        SwitchSpec(
            name="rk3_stage_r3uv_nemo_literal",
            owner="LatLonCGridOceanConfig",
            path_in_card="ocean.recipe.model_config.rk3_stage_r3uv_nemo_literal",
            category="B_source_rounding_or_order",
            exact_value=True,
            normal_value=False,
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:1984",
            normal_citation="packages/ocean/legoesm/ocean/state.py:3229",
            notes="Interpolate stored r3u/r3v ratios vs ratio of interpolated stage SSH.",
        ),
        SwitchSpec(
            name="nemo_bottom_drag_written_statements",
            owner="LatLonCGridOceanConfig",
            path_in_card="ocean.recipe.model_config.nemo_bottom_drag_written_statements",
            category="B_source_rounding_or_order",
            exact_value=True,
            normal_value=False,
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2002",
            normal_citation="packages/ocean/legoesm/ocean/state.py:3301",
            notes="Statement-rounded bottom drag coefficient evaluation.",
        ),
        SwitchSpec(
            name="rk3_vector_rhs_nemo_order",
            owner="LatLonCGridOceanConfig",
            path_in_card="ocean.recipe.model_config.rk3_vector_rhs_nemo_order",
            category="B_source_rounding_or_order",
            exact_value=True,
            normal_value=False,
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2014",
            normal_citation="packages/ocean/legoesm/ocean/state.py:3342",
            notes="NEMO written addition order for vector-form momentum RHS.",
        ),
        SwitchSpec(
            name="een_vorticity_written_order",
            owner="LatLonCGridOceanConfig",
            path_in_card="ocean.recipe.model_config.een_vorticity_written_order",
            category="B_source_rounding_or_order",
            exact_value=True,
            normal_value=False,
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2015",
            normal_citation="packages/ocean/legoesm/ocean/state.py:3190",
            notes="NEMO written triad summation order in EEN vorticity Coriolis/vorticity.",
        ),
        SwitchSpec(
            name="nemo_wzv_stored_r3_ratios",
            owner="LatLonCGridOceanConfig",
            path_in_card="ocean.recipe.model_config.nemo_wzv_stored_r3_ratios",
            category="B_source_rounding_or_order",
            exact_value=True,
            normal_value=False,
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2017",
            normal_citation="packages/ocean/legoesm/ocean/state.py:3321",
            notes="Use stored r3t/r3u/r3v ratios in wzv vertical velocity diagnostic.",
        ),
        SwitchSpec(
            name="nemo_zad_written_association",
            owner="LatLonCGridOceanConfig",
            path_in_card="ocean.recipe.model_config.nemo_zad_written_association",
            category="B_source_rounding_or_order",
            exact_value=True,
            normal_value=False,
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2018",
            normal_citation="packages/ocean/legoesm/ocean/state.py:3327",
            notes="NEMO written parenthesis association in vertical momentum advection (dynzad).",
        ),
        SwitchSpec(
            name="nemo_stage2_momentum_wzv_half_step",
            owner="LatLonCGridOceanConfig",
            path_in_card="ocean.recipe.model_config.nemo_stage2_momentum_wzv_half_step",
            category="B_source_rounding_or_order",
            exact_value=True,
            normal_value=False,
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2019",
            normal_citation="packages/ocean/legoesm/ocean/state.py:3333",
            notes="Stage-2 wzv half-step thickness ratio rounding.",
        ),
        SwitchSpec(
            name="nemo_stage_momentum_wzv_stored_kmm_ratios",
            owner="LatLonCGridOceanConfig",
            path_in_card="ocean.recipe.model_config.nemo_stage_momentum_wzv_stored_kmm_ratios",
            category="B_source_rounding_or_order",
            exact_value=True,
            normal_value=False,
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2020",
            normal_citation="packages/ocean/legoesm/ocean/state.py:3338",
            notes="Stored Kmm ratios in stage wzv vertical velocity.",
        ),
        SwitchSpec(
            name="nemo_ldf_stored_f_area",
            owner="LatLonCGridOceanConfig",
            path_in_card="ocean.recipe.model_config.nemo_ldf_stored_f_area",
            category="B_source_rounding_or_order",
            exact_value=True,
            normal_value=False,
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2026",
            normal_citation="packages/ocean/legoesm/ocean/state.py:3348",
            notes="Use stored F-cell area reciprocal in lateral friction.",
        ),
        SwitchSpec(
            name="nemo_ldf_stage_r3f",
            owner="LatLonCGridOceanConfig",
            path_in_card="ocean.recipe.model_config.nemo_ldf_stage_r3f",
            category="B_source_rounding_or_order",
            exact_value=True,
            normal_value=False,
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2027",
            normal_citation="packages/ocean/legoesm/ocean/state.py:3367",
            notes="Stage r3f ratio in lateral momentum diffusion.",
        ),
        SwitchSpec(
            name="rk3_stage1_ldf_source_order",
            owner="LatLonCGridOceanConfig",
            path_in_card="ocean.recipe.model_config.rk3_stage1_ldf_source_order",
            category="B_source_rounding_or_order",
            exact_value=True,
            normal_value=False,
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2028",
            normal_citation="packages/ocean/legoesm/ocean/state.py:3361",
            notes="Stage-1 lateral diffusion accumulation order in Krhs.",
        ),
        SwitchSpec(
            name="barotropic_substep_dt_nemo_literal",
            owner="BarotropicConfig",
            path_in_card="ocean.recipe.model_config.barotropic.barotropic_substep_dt_nemo_literal",
            category="B_source_rounding_or_order",
            exact_value=True,
            normal_value=False,
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:1957",
            normal_citation="packages/ocean/legoesm/ocean/state.py:1646",
            notes="NEMO literal substep rdt_e rounding in barotropic solver.",
        ),
        SwitchSpec(
            name="barotropic_een_ratio_written_order",
            owner="BarotropicConfig",
            path_in_card="ocean.recipe.model_config.barotropic.barotropic_een_ratio_written_order",
            category="B_source_rounding_or_order",
            exact_value=True,
            normal_value=False,
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2053",
            normal_citation="packages/ocean/legoesm/ocean/state.py:1667",
            notes="NEMO written order for barotropic EEN thickness ratio.",
        ),
        # --- Category C: Unconditional source-rounding barriers (1 representative entry) ---
        SwitchSpec(
            name="nemo_source_round / nemo_source_barrier",
            owner="source_rounding",
            path_in_card="unconditional_helper",
            category="C_unconditional_barrier",
            exact_value="optimization_barrier + copysign_zero",
            normal_value="optimization_barrier + copysign_zero",
            switched_in_normal_mode=False,
            exact_citation="packages/core/legoesm/core/source_rounding.py:14-35",
            normal_citation="packages/core/legoesm/core/source_rounding.py:14-35",
            notes=(
                "Direct calls in bulk_flux_omip.py:1086-1096, eos.py, and "
                "shortwave_penetration.py have no config switch; retained as-is "
                "because packages/ is unmodified."
            ),
        ),
        # --- Category D: Structurally coupled WS-RK3 / partial-cell / tripolar selectors (3 entries) ---
        SwitchSpec(
            name="zdf_implicit_solver_evaluation (+ companion zdf flags)",
            owner="LatLonCGridOceanConfig",
            path_in_card="ocean.recipe.model_config.zdf_implicit_solver_evaluation",
            category="D_structural_scheme_selector",
            exact_value="nemo_literal",
            normal_value="shared_thomas",
            switched_in_normal_mode=False,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:544",
            normal_citation="packages/ocean/legoesm/ocean/state.py:3163",
            notes=(
                "Retained at 'nemo_literal' on ORCA2 K0 because _zdf_literal is the "
                "only branch in _apply_implicit_vertical_mixing "
                "(ocean_model_latlon_cgrid.py:14827-14975) that consumes "
                "nemo_tracer_content_rhs, nemo_aimp_*, and 3-D partial-cell masks."
            ),
        ),
        SwitchSpec(
            name="orca2_l8_krhs_selection (+ stage-3 tracer/EIV/Treguier bundle)",
            owner="LatLonCGridOceanConfig",
            path_in_card="ocean.recipe.model_config.rk3_stage3_damping_after_geothermal_nemo_order",
            category="D_structural_scheme_selector",
            exact_value=True,
            normal_value=False,
            switched_in_normal_mode=False,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2954-2975",
            normal_citation="packages/ocean/legoesm/ocean/state.py:3423",
            notes=(
                "Required by LatLonCGridOceanModel constructor validation when "
                "rk3_ws + gm_bolus_advection='through_fct' + bbl_diffusive_option=1 "
                "are active (ocean_model_latlon_cgrid.py:2606-2622,4116-4293)."
            ),
        ),
        SwitchSpec(
            name="tripolar_and_ws_rk3_scheme_selectors (kappa_redi_horizontal_evaluation, tke_matrix_evaluation, barotropic_continuity_evaluation)",
            owner="GMRediConfig",
            path_in_card="ocean.recipe.model_config.gm_redi.kappa_redi_horizontal_evaluation",
            category="D_structural_scheme_selector",
            exact_value="nemo_metric_literal",
            normal_value="cosine_scaled",
            switched_in_normal_mode=False,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2909",
            normal_citation="packages/ocean/legoesm/ocean/physics/lateral_mixing/config.py:473",
            notes=(
                "Required by constructor validation on tripolar / WS-RK3 cards "
                "(ocean_model_latlon_cgrid.py:3248-3256,4507-4523; tke.py:3283-3287)."
            ),
        ),
    )


def _extract_switch_value(card: Any, spec: SwitchSpec) -> Any:
    if spec.path_in_card == "policy.transcendentals":
        return get_policy().transcendentals
    if spec.path_in_card == "unconditional_helper":
        return spec.exact_value
    if spec.path_in_card == "ocean.recipe.model_config.physics.vertical_mixing.ddm":
        ddm = card.ocean.recipe.model_config.physics.vertical_mixing.ddm
        return (ddm.enabled, ddm.form)
    if spec.path_in_card == "ocean.recipe.model_config.physics.vertical_mixing.iwm":
        iwm = card.ocean.recipe.model_config.physics.vertical_mixing.iwm
        return (iwm.form, iwm.tsdiff)
    obj = card
    for attr in spec.path_in_card.split("."):
        obj = getattr(obj, attr)
    return obj


def _extract_class_default_value(spec: SwitchSpec) -> Any:
    attr = spec.path_in_card.split(".")[-1]
    if spec.owner == "PrecisionPolicy":
        return PrecisionPolicy.fp64().transcendentals
    if spec.owner == "NEMOTestcaseCard":
        return PrecisionPolicy.fp64().transcendentals
    if spec.owner == "TKEConfig":
        return getattr(TKEConfig(), attr)
    if spec.owner == "DoubleDiffusionConfig":
        ddm = DoubleDiffusionConfig()
        return (ddm.enabled, ddm.form)
    if spec.owner == "IWMConfig":
        iwm = IWMConfig()
        return (iwm.form, iwm.tsdiff)
    if spec.owner == "GMRediConfig":
        return getattr(GMRediConfig(), attr)
    if spec.owner == "MLEConfig":
        return getattr(MLEConfig(), attr)
    if spec.owner == "ZStarCoordinate":
        return False
    if spec.owner == "LatLonCGridOceanConfig":
        return getattr(LatLonCGridOceanConfig(), attr)
    if spec.owner == "BarotropicConfig":
        return getattr(BarotropicConfig(), attr)
    return spec.normal_value


def verify_switch_inventory_against_card(
    base_card: Any,
    normal_card: Any,
    *,
    inventory: Sequence[SwitchSpec] | None = None,
) -> dict[str, Any]:
    """Verify exact vs normal card switch settings and confirm packages/ defaults are untouched."""
    specs = (
        build_orca2_exact_vs_normal_switch_inventory()
        if inventory is None
        else tuple(inventory)
    )
    require(len(specs) > 0, "switch inventory must not be empty")
    switched_count = 0
    cat_a_count = 0
    cat_b_count = 0
    for spec in specs:
        class_default = _extract_class_default_value(spec)
        require(
            class_default == spec.normal_value,
            f"Switch {spec.name!r} normal_value {spec.normal_value!r} != "
            f"class default {class_default!r} (was a default in packages/ changed?)",
        )
        if spec.path_in_card in ("policy.transcendentals", "unconditional_helper"):
            if spec.switched_in_normal_mode:
                switched_count += 1
                if spec.category == "A_libm_guarded":
                    cat_a_count += 1
            continue
        base_val = _extract_switch_value(base_card, spec)
        norm_val = _extract_switch_value(normal_card, spec)
        require(
            base_val == spec.exact_value,
            f"base_card switch {spec.name!r} has {base_val!r}, expected exact_value {spec.exact_value!r}",
        )
        if spec.switched_in_normal_mode:
            switched_count += 1
            if spec.category == "A_libm_guarded":
                cat_a_count += 1
            elif spec.category == "B_source_rounding_or_order":
                cat_b_count += 1
            require(
                norm_val == spec.normal_value,
                f"normal_card switch {spec.name!r} has {norm_val!r}, expected normal_value {spec.normal_value!r}",
            )
            require(
                base_val != norm_val,
                f"switched entry {spec.name!r} did not change between exact and normal card",
            )
        else:
            require(
                norm_val == spec.exact_value,
                f"retained switch {spec.name!r} changed on normal_card: {norm_val!r} != {spec.exact_value!r}",
            )
    require(cat_a_count == 6, f"expected 6 Category A switches, got {cat_a_count}")
    require(cat_b_count == 36, f"expected 36 Category B switches, got {cat_b_count}")
    require(switched_count == 42, f"expected 42 switched normal-mode settings, got {switched_count}")
    return {
        "total_inventory_entries": len(specs),
        "switched_in_normal_mode": switched_count,
        "category_a_count": cat_a_count,
        "category_b_count": cat_b_count,
    }


def build_normal_mode_orca2_k0_card(
    base_card: Any,
    *,
    variant: str = "cat_a_plus_b",
    precision_policy_name: str = "fp64",
) -> Any:
    """Return a copy of ``base_card`` with exact-mode switches set to normal defaults."""
    require(
        variant in ("cat_a_plus_b", "cat_a_only"),
        f"unknown normal-mode variant {variant!r}; expected 'cat_a_plus_b' or 'cat_a_only'",
    )
    require(
        precision_policy_name in ("fp64", "fp32"),
        f"unknown precision_policy_name {precision_policy_name!r}; expected 'fp64' or 'fp32'",
    )
    mc = base_card.ocean.recipe.model_config
    zc = base_card.ocean.recipe.z_coord
    vm = mc.physics.vertical_mixing

    tke = vm.tke._replace(
        tke_htau_evaluation="jax_expression",
        tke_etau_exponential_evaluation="jax_expression",
    )
    ddm = vm.ddm._replace(enabled=False, form="hybrid")
    iwm = vm.iwm._replace(form="generic", tsdiff=False)

    if variant == "cat_a_plus_b":
        tke = tke._replace(
            nemo_bn2_source_rounded=False,
            nemo_tke_solve_source_rounded=False,
            nemo_tke_matrix_rhs_source_rounded=False,
            nemo_sh2_r3_source_rounded=False,
            nemo_sh2_r3_stored_reciprocal=False,
            tke_mxl_raw_evaluation="factored",
            tke_solver_evaluation="shared_thomas",
            tke_langmuir_evaluation="vectorized",
        )
        gm = mc.gm_redi._replace(
            slope_uv_source_rounding=False,
            redi_a33_source_ordering=False,
            redi_a33_evaluation="normalized_square",
            redi_vertical_skew_evaluation="normalized_sums",
            slope_metric_evaluation="division",
        )
        mle = mc.physics.mle._replace(nemo_association=False)
        bt = mc.barotropic._replace(
            barotropic_substep_dt_nemo_literal=False,
            barotropic_een_ratio_written_order=False,
        )
        zc = zc._replace(nemo_r3t_stored_product=False)
        mc = mc._replace(
            nemo_eos_gdept_rounded_product=False,
            zdf_tracer_nemo_rounded_thickness=False,
            barotropic_rhs_ascending_level_sum=False,
            rk3_stage1_rhs_nemo_literal=False,
            rk3_stage_velocity_written_order=False,
            rk3_stage_tracer_written_order=False,
            rk3_stage_r3f_nemo_literal=False,
            rk3_stage_r3t_nemo_literal=False,
            rk3_stage_r3uv_nemo_literal=False,
            nemo_bottom_drag_written_statements=False,
            rk3_vector_rhs_nemo_order=False,
            een_vorticity_written_order=False,
            nemo_wzv_stored_r3_ratios=False,
            nemo_zad_written_association=False,
            nemo_stage2_momentum_wzv_half_step=False,
            nemo_stage_momentum_wzv_stored_kmm_ratios=False,
            nemo_ldf_stored_f_area=False,
            nemo_ldf_stage_r3f=False,
            rk3_stage1_ldf_source_order=False,
            gm_redi=gm,
            barotropic=bt,
            physics=mc.physics._replace(
                vertical_mixing=vm._replace(tke=tke, ddm=ddm, iwm=iwm),
                mle=mle,
            ),
        )
    else:
        mc = mc._replace(
            physics=mc.physics._replace(
                vertical_mixing=vm._replace(tke=tke, ddm=ddm, iwm=iwm),
            ),
        )

    recipe = base_card.ocean.recipe._replace(
        model_config=mc,
        physics_config=mc.physics,
        z_coord=zc,
    )
    ocean = base_card.ocean._replace(
        recipe=recipe,
        transcendentals="native",
        precision_policy=precision_policy_name,
    )
    return base_card._replace(ocean=ocean)


def build_fast_literal_mode_orca2_k0_card(base_card: Any) -> Any:
    """Keep the exact K0 recipe and select only native XLA transcendentals."""
    require(
        base_card.ocean.transcendentals == "libm",
        "fast-literal source card must be the exact scalar-libm card",
    )
    require(
        base_card.ocean.precision_policy == "fp64",
        "fast-literal source card must be fp64",
    )
    return base_card._replace(
        ocean=base_card.ocean._replace(transcendentals="native")
    )


def verify_fast_literal_card(base_card: Any, fast_card: Any) -> dict[str, Any]:
    """Prove native transcendentals are the fast card's only configuration delta."""
    require(base_card.ocean.transcendentals == "libm", "base card is not exact/libm")
    require(fast_card.ocean.transcendentals == "native", "fast card is not native")
    require(
        fast_card.ocean.precision_policy == base_card.ocean.precision_policy == "fp64",
        "fast card changed the fp64 precision policy",
    )
    retained = 0
    for spec in build_orca2_exact_vs_normal_switch_inventory():
        if spec.path_in_card in ("policy.transcendentals", "ocean.transcendentals"):
            continue
        if spec.path_in_card == "unconditional_helper":
            retained += 1
            continue
        base_value = _extract_switch_value(base_card, spec)
        fast_value = _extract_switch_value(fast_card, spec)
        require(
            fast_value == base_value,
            f"fast-literal card changed retained selector {spec.name}: "
            f"{base_value!r} -> {fast_value!r}",
        )
        retained += 1
    require(
        fast_card.ocean.recipe is base_card.ocean.recipe,
        "fast-literal card replaced the exact recipe object",
    )
    require(fast_card.k0 == base_card.k0, "fast-literal card changed K0 selectors")
    require(
        fast_card._replace(ocean=base_card.ocean) == base_card,
        "fast-literal card changed a non-ocean top-level field",
    )
    return {"changed_fields": ["ocean.transcendentals"], "retained_inventory_entries": retained}


def compute_step_timing_summary(
    step_times_s: Sequence[float],
    *,
    warmup_steps: int = WARMUP_STEPS,
    min_steady_steps: int = MIN_STEADY_STEPS,
    year_steps: int = YEAR_STEPS,
) -> dict[str, Any]:
    """Compute compile time, steady-state s/step stats, and 1-year projections."""
    times = np.asarray(step_times_s, dtype=np.float64)
    require(times.ndim == 1, "step_times_s must be 1-D")
    require(warmup_steps >= 1, f"warmup_steps must be >= 1, got {warmup_steps}")
    require(min_steady_steps >= 1, f"min_steady_steps must be >= 1, got {min_steady_steps}")
    require(
        times.size >= warmup_steps + min_steady_steps,
        f"need >= {warmup_steps + min_steady_steps} total steps "
        f"({warmup_steps} warmup + >={min_steady_steps} steady), got {times.size}",
    )
    require(bool(np.isfinite(times).all()) and bool((times > 0.0).all()),
            "all step times must be finite and positive")

    warmup = times[:warmup_steps]
    steady = times[warmup_steps:]
    steady_kts = np.arange(warmup_steps + 1, times.size + 1)
    odd_mask = (steady_kts % 2) == 1
    even_mask = (steady_kts % 2) == 0

    median_s = float(np.median(steady))
    mean_s = float(np.mean(steady))
    std_s = float(np.std(steady))
    q25, q75 = np.percentile(steady, [25.0, 75.0])
    iqr_s = float(q75 - q25)
    min_s = float(np.min(steady))
    max_s = float(np.max(steady))
    odd_median_s = float(np.median(steady[odd_mask])) if np.any(odd_mask) else median_s
    even_median_s = float(np.median(steady[even_mask])) if np.any(even_mask) else median_s

    compile_warmup_total_s = float(np.sum(warmup))
    compile_overhead_est_s = float(max(0.0, compile_warmup_total_s - warmup_steps * median_s))

    proj_steady_s = float(year_steps * median_s)
    proj_with_compile_s = float(compile_warmup_total_s + (year_steps - warmup_steps) * median_s)
    proj_mean_s = float(year_steps * mean_s)

    return {
        "total_steps": int(times.size),
        "warmup_steps": int(warmup_steps),
        "steady_state_steps": int(steady.size),
        "step1_s": float(times[0]),
        "step2_s": float(times[1]) if times.size >= 2 else 0.0,
        "compile_warmup_total_s": compile_warmup_total_s,
        "compile_overhead_est_s": compile_overhead_est_s,
        "steady_median_s": median_s,
        "steady_mean_s": mean_s,
        "steady_std_s": std_s,
        "steady_iqr_s": iqr_s,
        "steady_min_s": min_s,
        "steady_max_s": max_s,
        "steady_odd_kt_median_s": odd_median_s,
        "steady_even_kt_median_s": even_median_s,
        "projected_year_steps": int(year_steps),
        "projected_year_steady_s": proj_steady_s,
        "projected_year_steady_min": proj_steady_s / 60.0,
        "projected_year_steady_hr": proj_steady_s / 3600.0,
        "projected_year_with_compile_s": proj_with_compile_s,
        "projected_year_with_compile_min": proj_with_compile_s / 60.0,
        "projected_year_with_compile_hr": proj_with_compile_s / 3600.0,
        "projected_year_from_mean_s": proj_mean_s,
        "projected_year_from_mean_min": proj_mean_s / 60.0,
        "step_times_s": [float(x) for x in times],
    }


def score_daily_temperature_against_reference(
    candidate_t: np.ndarray,
    reference_t: np.ndarray,
    wet_mask_3d: np.ndarray,
    *,
    kt: int,
) -> dict[str, Any]:
    """Compute SST and 3-D T RMS and max absolute differences at step ``kt``."""
    cand = np.asarray(candidate_t, dtype=np.float64)
    ref = np.asarray(reference_t, dtype=np.float64)
    wet3 = np.asarray(wet_mask_3d, dtype=bool)
    require(cand.shape == (148, 180, 30), f"candidate T shape {cand.shape} != (148, 180, 30)")
    require(ref.shape == (148, 180, 30), f"reference T shape {ref.shape} != (148, 180, 30)")
    require(wet3.shape == (148, 180, 30), f"wet_mask_3d shape {wet3.shape} != (148, 180, 30)")
    require(bool(np.isfinite(cand).all()), f"candidate T at kt={kt} contains non-finite values")
    require(bool(np.isfinite(ref).all()), f"reference T at kt={kt} contains non-finite values")
    require(bool(np.any(wet3)), "wet_mask_3d must have at least one active cell")

    wet2 = wet3[..., 0]
    diff_t = cand - ref
    diff_sst = diff_t[..., 0]

    return {
        "kt": int(kt),
        "day": int(kt // STEPS_PER_DAY),
        "sst_rms_all_c": float(np.sqrt(np.mean(diff_sst ** 2))),
        "sst_rms_wet_c": float(np.sqrt(np.mean(diff_sst[wet2] ** 2))),
        "sst_max_abs_c": float(np.max(np.abs(diff_sst))),
        "t3d_rms_all_c": float(np.sqrt(np.mean(diff_t ** 2))),
        "t3d_rms_wet_c": float(np.sqrt(np.mean(diff_t[wet3] ** 2))),
        "t3d_max_abs_c": float(np.max(np.abs(diff_t))),
    }


def read_memory_telemetry() -> dict[str, Any]:
    """Read host RSS/VmHWM and optional GPU memory stats for the current process."""
    ru_maxrss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    vm_hwm_kib = ru_maxrss_kib
    vm_rss_kib = 0
    status_path = Path("/proc/self/status")
    if status_path.is_file():
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmHWM:"):
                parts = line.split()
                if len(parts) >= 2:
                    vm_hwm_kib = max(vm_hwm_kib, int(parts[1]))
            elif line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    vm_rss_kib = int(parts[1])
    telemetry: dict[str, Any] = {
        "host_peak_rss_kib": vm_hwm_kib,
        "host_peak_rss_mib": vm_hwm_kib / 1024.0,
        "host_peak_rss_gib": vm_hwm_kib / (1024.0 * 1024.0),
        "host_current_rss_mib": vm_rss_kib / 1024.0,
        "gpu_peak_bytes_in_use": None,
        "gpu_peak_mib": None,
        "gpu_bytes_limit": None,
    }
    if jax.default_backend() == "gpu":
        dev = jax.devices()[0]
        if hasattr(dev, "memory_stats"):
            stats = dev.memory_stats() or {}
            if "peak_bytes_in_use" in stats:
                peak_b = int(stats["peak_bytes_in_use"])
                telemetry["gpu_peak_bytes_in_use"] = peak_b
                telemetry["gpu_peak_mib"] = peak_b / (1024.0 * 1024.0)
            if "bytes_limit" in stats:
                telemetry["gpu_bytes_limit"] = int(stats["bytes_limit"])
    return telemetry


def load_core_window_for_steps(
    deck_root: Path,
    manifest: Path,
    max_steps: int,
) -> dict[str, np.ndarray]:
    """Load and interpolate all CORE records reachable across steps 1..max_steps."""
    require(1 <= max_steps <= YEAR_STEPS, f"max_steps must be in 1..{YEAR_STEPS}, got {max_steps}")
    declared: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            declared[parts[1]] = parts[0]
    required = {filename for _, filename, _, _ in o1.FIELD_SPECS}
    required |= {
        "weights_core2_orca2_bicub.nc",
        "weights_core2_orca2_bilin.nc",
        "ORCA_R2_zps_domcfg.nc",
    }
    for name in required:
        require(name in declared, f"CORE input manifest omits {name}")
        require(sha256_file(deck_root / name) == declared[name],
                f"CORE input digest moved: {name}")

    n_6h = min(1460, max(5, (max_steps - 1) // 2 + 1))
    n_daily = min(365, max(2, (max_steps - 1) // 8 + 1))
    n_monthly = 1 if max_steps <= 10 else 12

    mapped: dict[str, np.ndarray] = {}
    weights = {bicubic: o1.load_weights(deck_root, bicubic) for bicubic in (False, True)}
    for field, filename, variable, bicubic in o1.FIELD_SPECS:
        with netCDF4.Dataset(deck_root / filename, "r") as ds:
            src_var = ds[variable]
            total_recs = int(src_var.shape[0])
            count = {1460: n_6h, 365: n_daily, 12: n_monthly}.get(total_recs)
            require(count is not None, f"CORE cadence moved for {variable}: {src_var.shape}")
            raw_slice = np.asarray(src_var[:count].data, dtype=np.float64)
        indices, coefficients = weights[bicubic]
        mapped[field] = np.stack([
            np.asarray(o1.nemo_fld_interp(raw_slice[idx], indices, coefficients, bicubic=bicubic))
            for idx in range(count)
        ])

    with netCDF4.Dataset(deck_root / "ORCA_R2_zps_domcfg.nc", "r") as ds:
        cosine, sine = o1.nemo_t_rotation_from_domain(*(
            np.asarray(ds[name][:].data, dtype=np.float64)
            for name in ("glamt", "gphit", "glamv", "gphiv")
        ))
    wind_shape = mapped["wndi"].shape
    mapped["wndi"], mapped["wndj"] = map(
        np.asarray,
        o1.rotate_en_to_ij(
            mapped["wndi"],
            mapped["wndj"],
            np.broadcast_to(cosine, wind_shape),
            np.broadcast_to(sine, wind_shape),
        ),
    )
    require(all(np.isfinite(val).all() for val in mapped.values()),
            "mapped CORE normal-year input contains non-finite values")
    return mapped


def _load_monthly_12(path: Path, variable: str) -> np.ndarray:
    require(path.is_file(), f"missing K0 monthly file: {path}")
    with netCDF4.Dataset(path, "r") as ds:
        require(variable in ds.variables, f"K0 variable {variable!r} absent from {path.name}")
        src = ds.variables[variable]
        src.set_auto_maskandscale(False)
        val = np.asarray(src[:], dtype=np.float64)
    if val.ndim == 4:
        val = val[:, 0]
    require(val.shape == (12, 148, 180), f"unexpected K0 monthly shape in {path.name}: {val.shape}")
    require(bool(np.isfinite(val).all()), f"non-finite monthly values in {path.name}")
    return val


def _monthly_pair_for_clock(
    records_12: np.ndarray,
    isecsbc: int,
) -> bulk.NemoStandaloneFieldPair:
    after_ext = int(np.searchsorted(EXTENDED_MONTH_CENTRES_SEC, isecsbc, side="right"))
    before_ext = after_ext - 1
    before_s = int(EXTENDED_MONTH_CENTRES_SEC[before_ext])
    after_s = int(EXTENDED_MONTH_CENTRES_SEC[after_ext])
    before_idx = (before_ext - 1) % 12
    after_idx = (after_ext - 1) % 12
    return bulk.NemoStandaloneFieldPair(
        jnp.asarray(records_12[before_idx], dtype=jnp.float64),
        jnp.asarray(records_12[after_idx], dtype=jnp.float64),
        isecsbc,
        before_s,
        after_s,
    )


def precompute_step_controls(
    card: Any,
    deck_root: Path,
    core_manifest: Path,
    ice_cover_path: Path,
    steps: int,
) -> tuple[dict[str, Any], ...]:
    """Precompute per-step physical forcing controls for steps 1..steps outside the step timer."""
    recipe = card.ocean.recipe
    state = recipe.initial_state
    mask_np = np.asarray(state.land_mask.data, dtype=np.float64)
    ny, nx = mask_np.shape
    require((ny, nx) == (148, 180), f"unexpected ORCA2 grid shape: {(ny, nx)}")
    active_np = np.asarray(recipe.z_coord.is_active) > 0.5
    wet_2d = mask_np > 0.5

    mapped_core = load_core_window_for_steps(deck_root, core_manifest, steps)
    cover = r340_s4.read_decision171_cover(ice_cover_path)
    runoff_12 = _load_monthly_12(deck_root / "runoff_core_monthly.nc", "sorunoff")
    sss_12 = _load_monthly_12(deck_root / "sss_data.nc", "sss")
    geothermal = jnp.asarray(
        ladder.read_geothermal_heatflow(deck_root, (148, 180)),
        dtype=jnp.float64,
    )
    with netCDF4.Dataset(deck_root / "ORCA_R2_zps_domcfg.nc", "r") as ds:
        ff_f = jnp.asarray(np.asarray(ds.variables["ff_f"][:], dtype=np.float64))
    with netCDF4.Dataset(deck_root / "runoff_core_monthly.nc", "r") as ds:
        mouth = jnp.asarray(np.asarray(ds.variables["socoefr0"][:], dtype=np.float64))

    stress_lbc, tmask_east, tmask_north, umask, vmask = r340_s4._build_stress_lbc_and_masks(
        mask_np,
        np.asarray(state.u_mask.data, dtype=np.float64),
        np.asarray(state.v_mask.data, dtype=np.float64),
        recipe.grid.fold,
    )
    zero = jnp.zeros((ny, nx), dtype=jnp.float64)
    one = jnp.ones((ny, nx), dtype=jnp.float64)
    mask_jnp = jnp.asarray(mask_np, dtype=jnp.float64)
    cover_jnp = jnp.asarray(cover, dtype=jnp.float64)

    means_init = NemoSurfaceMeans(*(zero for _ in range(7)))
    s0_base = bulk.NemoS0Inputs(
        prior=means_init,
        kt=1,
        nit000=1,
        forcing_stride=2,
        restart_means_present=False,
        use_conservative_temperature=False,
        remove_inverse_barometer=False,
        u_before=zero,
        v_before=zero,
        temperature_now=zero,
        salinity_now=zero,
        ssh_now=zero,
        ssh_inverse_barometer=zero,
        e3t_reference=one,
        r3t_now=zero,
        surface_mask=mask_jnp,
        shortwave_fraction_top=one,
    )
    s2_base = bulk.NemoS2Inputs(
        stress_lbc=stress_lbc,
        post_ice_sfx=zero,
        runoff_now=zero,
        prior_runoff_mass=zero,
        prior_runoff_temperature_content=zero,
        prior_runoff_salinity_content=zero,
        prior_restoring_volume_flux=zero,
        prior_restoring_heat_flux=zero,
        prior_restoring_coefficient=one,
        sss_target=zero,
        runoff_mouth_mask=mouth,
        surface_mask=mask_jnp,
        tmask_east=jnp.asarray(tmask_east, dtype=jnp.float64),
        tmask_north=jnp.asarray(tmask_north, dtype=jnp.float64),
        umask=jnp.asarray(umask, dtype=jnp.float64),
        vmask=jnp.asarray(vmask, dtype=jnp.float64),
        freshwater_budget_correction=jnp.asarray(0.0, dtype=jnp.float64),
        runoff_factor=jnp.asarray(1.0, dtype=jnp.float64),
        reference_density=jnp.asarray(recipe.model_config.physics.constants.rho_0, dtype=jnp.float64),
        restoring_mm_per_day=jnp.asarray(-166.67, dtype=jnp.float64),
        restoring_bound_mm_per_day=jnp.asarray(4.0, dtype=jnp.float64),
        seconds_per_day=jnp.asarray(86400.0, dtype=jnp.float64),
        salinity_floor=jnp.asarray(1.0e-20, dtype=jnp.float64),
        heat_capacity=jnp.asarray(recipe.model_config.physics.constants.c_sw, dtype=jnp.float64),
    )

    dt_s = int(card.ocean.dt_s)
    controls: list[dict[str, Any]] = []
    for kt in range(1, steps + 1):
        isecsbc = kt * dt_s
        retained_kt = kt if kt % 2 else kt - 1
        retained_clock = retained_kt * dt_s

        idx_6h = retained_clock // 21600
        idx_daily = retained_clock // 86400
        idx_monthly = int(np.searchsorted(MONTH_END_SEC, retained_clock, side="right"))
        if mapped_core["precip_raw"].shape[0] == 1:
            idx_monthly = 0

        atmosphere = bulk.NemoStandaloneAtmosphereFields(
            jnp.asarray(mapped_core["wndi"][idx_6h], dtype=jnp.float64),
            jnp.asarray(mapped_core["wndj"][idx_6h], dtype=jnp.float64),
            jnp.asarray(mapped_core["tair"][idx_6h], dtype=jnp.float64),
            jnp.asarray(mapped_core["humi"][idx_6h], dtype=jnp.float64),
            jnp.asarray(mapped_core["slp"][idx_6h], dtype=jnp.float64),
            jnp.asarray(mapped_core["qsr_down"][idx_daily], dtype=jnp.float64),
            jnp.asarray(mapped_core["qlw_down"][idx_daily], dtype=jnp.float64),
            jnp.asarray(mapped_core["precip_raw"][idx_monthly], dtype=jnp.float64),
            jnp.asarray(mapped_core["snow_raw"][idx_monthly], dtype=jnp.float64),
        )
        pair_runoff = _monthly_pair_for_clock(runoff_12, isecsbc)
        pair_sss = _monthly_pair_for_clock(sss_12, isecsbc)
        s0f = bulk.NemoStandaloneS0FInputs(
            atmosphere,
            atmosphere,
            isecsbc,
            isecsbc - 1,
            isecsbc + 1,
            pair_runoff,
            pair_sss,
        )
        ice_before_s, ice_after_s = (
            (-15768000, 15768000)
            if isecsbc <= 15768000
            else (15768000, 47304000)
        )
        ice_if = bulk.NemoSbcIceIfInputs(
            cover_jnp,
            cover_jnp,
            isecsbc,
            ice_before_s,
            ice_after_s,
            zero,
            zero,
            zero,
            ff_f,
            mask_jnp,
            mask_jnp,
        )
        chl = jnp.asarray(
            ladder.chlorophyll_at_step(deck_root, kt, wet_2d, dt_s=card.ocean.dt_s),
            dtype=jnp.float64,
        )
        sponge = build_orca2_damping_for_step(
            deck_root,
            active_np,
            kt=kt,
            dt_s=card.ocean.dt_s,
            mask_evaluation="nemo_tmask_multiply",
        )
        controls.append({
            "s0f": s0f,
            "ice_if": ice_if,
            "s0_base": s0_base,
            "s2_base": s2_base,
            "chl": chl,
            "geothermal_heat_flux_mwm2": geothermal,
            "sponge": sponge,
        })
    return tuple(controls)


def run_single_benchmark_arm(
    arm_id: str,
    *,
    deck_root: Path = DEFAULT_DECK_ROOT,
    core_manifest: Path = DEFAULT_CORE_MANIFEST,
    ice_cover_path: Path = DEFAULT_ICE_COVER_PATH,
    exact_ref_dir: Path = DEFAULT_EXACT_REF_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    variant: str = "cat_a_plus_b",
    steps_override: int | None = None,
    min_steady_steps: int = MIN_STEADY_STEPS,
    collect_accuracy: bool = True,
    requested_cores: int | None = None,
    expected_cpu_ids: Sequence[int] | None = None,
    profile_dir: Path | None = None,
    artifact_name: str | None = None,
    allow_dirty_provenance: bool = False,
) -> dict[str, Any]:
    """Execute one benchmark arm in the current process and save its JSON artifact."""
    valid_arms = (*ARM_IDS, *FAST_LITERAL_ARM_IDS)
    require(arm_id in valid_arms, f"unknown benchmark arm {arm_id!r}; expected one of {valid_arms}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if arm_id == "exact_cpu_fp64":
        require(jax.config.jax_enable_x64, "exact_cpu_fp64 requires JAX_ENABLE_X64=1")
        require(jax.default_backend() == "cpu", f"exact_cpu_fp64 requires CPU backend, got {jax.default_backend()!r}")
        set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
        steps = EXACT_BENCHMARK_STEPS
        daily_check_steps = DAILY_CHECK_STEPS_EXACT
    elif arm_id == "fast_literal_cpu_fp64":
        require(jax.config.jax_enable_x64, "fast_literal_cpu_fp64 requires JAX_ENABLE_X64=1")
        require(jax.default_backend() == "cpu", f"fast_literal_cpu_fp64 requires CPU backend, got {jax.default_backend()!r}")
        set_policy(PrecisionPolicy.fp64(transcendentals="native"))
        steps = NORMAL_BENCHMARK_STEPS
        daily_check_steps = DAILY_CHECK_STEPS_8D
    elif arm_id == "fast_literal_gpu_fp64":
        require(jax.config.jax_enable_x64, "fast_literal_gpu_fp64 requires JAX_ENABLE_X64=1")
        require(jax.default_backend() == "gpu", f"fast_literal_gpu_fp64 requires GPU backend, got {jax.default_backend()!r}")
        set_policy(PrecisionPolicy.fp64(transcendentals="native"))
        steps = NORMAL_BENCHMARK_STEPS
        daily_check_steps = DAILY_CHECK_STEPS_8D
    elif arm_id == "normal_cpu_fp64":
        require(jax.config.jax_enable_x64, "normal_cpu_fp64 requires JAX_ENABLE_X64=1")
        require(jax.default_backend() == "cpu", f"normal_cpu_fp64 requires CPU backend, got {jax.default_backend()!r}")
        set_policy(PrecisionPolicy.fp64(transcendentals="native"))
        steps = NORMAL_BENCHMARK_STEPS
        daily_check_steps = DAILY_CHECK_STEPS_8D
    elif arm_id == "normal_gpu_fp64":
        require(jax.config.jax_enable_x64, "normal_gpu_fp64 requires JAX_ENABLE_X64=1")
        require(jax.default_backend() == "gpu", f"normal_gpu_fp64 requires GPU backend, got {jax.default_backend()!r}")
        set_policy(PrecisionPolicy.fp64(transcendentals="native"))
        steps = NORMAL_BENCHMARK_STEPS
        daily_check_steps = DAILY_CHECK_STEPS_8D
    else:  # normal_gpu_fp32
        require(not jax.config.jax_enable_x64, "normal_gpu_fp32 requires JAX_ENABLE_X64=0")
        require(jax.default_backend() == "gpu", f"normal_gpu_fp32 requires GPU backend, got {jax.default_backend()!r}")
        warnings.filterwarnings(
            "ignore",
            message="Explicitly requested dtype",
            category=UserWarning,
        )
        set_policy(PrecisionPolicy.fp32(transcendentals="native"))
        steps = NORMAL_BENCHMARK_STEPS
        daily_check_steps = DAILY_CHECK_STEPS_8D

    if steps_override is not None:
        require(
            steps_override >= WARMUP_STEPS + min_steady_steps,
            f"steps_override={steps_override} must cover {WARMUP_STEPS} warmup + "
            f"{min_steady_steps} steady steps",
        )
        steps = int(steps_override)
    if not collect_accuracy:
        daily_check_steps = ()

    affinity = tuple(sorted(os.sched_getaffinity(0)))
    if requested_cores is not None:
        require(requested_cores > 0, f"requested_cores must be positive, got {requested_cores}")
        require(
            len(affinity) == requested_cores,
            f"CPU affinity has {len(affinity)} CPUs, expected requested_cores={requested_cores}: "
            f"{affinity}",
        )
    if expected_cpu_ids is not None:
        expected = tuple(sorted(int(cpu) for cpu in expected_cpu_ids))
        require(
            affinity == expected,
            f"CPU affinity {affinity} != expected physical CPU ids {expected}",
        )
    if profile_dir is not None:
        require(
            steps >= ROUND349_PROFILE_STEP,
            f"profile requires step {ROUND349_PROFILE_STEP}, got steps={steps}",
        )
        require(not profile_dir.exists(), f"profile directory already exists: {profile_dir}")
        profile_dir.mkdir(parents=True)

    t_setup_start = time.perf_counter()
    base_card = bulk.build_nemo_orca2_k0_ocean_card(deck_root)
    normal_card_check = build_normal_mode_orca2_k0_card(base_card, variant="cat_a_plus_b")
    inventory_check = verify_switch_inventory_against_card(base_card, normal_card_check)

    if arm_id == "exact_cpu_fp64":
        card = base_card
    elif arm_id in FAST_LITERAL_ARM_IDS:
        card = build_fast_literal_mode_orca2_k0_card(base_card)
        inventory_check = verify_fast_literal_card(base_card, card)
    else:
        pol_name = "fp32" if arm_id == "normal_gpu_fp32" else "fp64"
        card = build_normal_mode_orca2_k0_card(
            base_card,
            variant=variant,
            precision_policy_name=pol_name,
        )

    wet_3d = np.asarray(base_card.ocean.recipe.z_coord.is_active, dtype=bool)
    controls = precompute_step_controls(card, deck_root, core_manifest, ice_cover_path, steps)
    recipe = card.ocean.recipe
    model = LatLonCGridOceanModel(
        recipe.grid,
        recipe.z_coord,
        recipe.model_config,
        iwm_forcing=recipe.iwm_forcing,
    )
    state = recipe.initial_state
    setup_wall_s = time.perf_counter() - t_setup_start

    expected_dtype = np.dtype("float32") if arm_id == "normal_gpu_fp32" else np.dtype("float64")
    require(
        np.dtype(state.T.data.dtype) == expected_dtype,
        f"arm {arm_id}: initial state.T.data.dtype={state.T.data.dtype} != {expected_dtype}",
    )

    carry_s0 = None
    carry_s2 = None
    prior_surface = None
    step_times_s: list[float] = []
    step_cpu_telemetry: list[dict[str, Any]] = []
    daily_accuracy: list[dict[str, Any]] = []

    for kt in range(1, steps + 1):
        ctrl = controls[kt - 1]
        s0_base = ctrl["s0_base"]
        s2_base = ctrl["s2_base"]
        s0 = (
            s0_base
            if carry_s0 is None
            else s0_base._replace(
                prior=carry_s0.prior,
                shortwave_fraction_top=carry_s0.shortwave_fraction_top,
            )
        )
        s2 = (
            s2_base
            if carry_s2 is None
            else s2_base._replace(
                prior_runoff_mass=carry_s2.prior_runoff_mass,
                prior_runoff_temperature_content=carry_s2.prior_runoff_temperature_content,
                prior_runoff_salinity_content=carry_s2.prior_runoff_salinity_content,
                prior_restoring_volume_flux=carry_s2.prior_restoring_volume_flux,
                prior_restoring_heat_flux=carry_s2.prior_restoring_heat_flux,
                prior_restoring_coefficient=carry_s2.prior_restoring_coefficient,
            )
        )

        thread_ticks_before = read_thread_cpu_ticks() if requested_cores is not None else None
        process_ticks_before = (
            read_process_total_cpu_ticks() if requested_cores is not None else None
        )
        is_profile_step = profile_dir is not None and kt == ROUND349_PROFILE_STEP
        trace_context = (
            jax.profiler.trace(profile_dir, create_perfetto_trace=True)
            if is_profile_step
            else nullcontext()
        )
        original_surface_card = bulk.nemo_orca2_k0_surface_card
        if is_profile_step:
            def profiled_surface_card(*args, **kwargs):
                with jax.profiler.TraceAnnotation(ROUND349_SURFACE_ANNOTATION):
                    surface_result = original_surface_card(*args, **kwargs)
                    jax.block_until_ready(surface_result)
                    return surface_result

            bulk.nemo_orca2_k0_surface_card = profiled_surface_card
        with trace_context:
            annotation = (
                jax.profiler.TraceAnnotation(ROUND349_PROFILE_ANNOTATION)
                if is_profile_step
                else nullcontext()
            )
            try:
                with annotation:
                    t0 = time.perf_counter()
                    res = bulk.nemo_orca2_k0_ocean_step(
                        card,
                        state,
                        model=model,
                        nemo_kt=kt,
                        s0f=ctrl["s0f"],
                        ice_if=ctrl["ice_if"],
                        s0=s0,
                        s2=s2,
                        chl=ctrl["chl"],
                        geothermal_heat_flux_mwm2=ctrl["geothermal_heat_flux_mwm2"],
                        prior_surface=prior_surface,
                        sponge=ctrl["sponge"],
                    )
                    state = res.ocean.state_after if hasattr(res.ocean, "state_after") else res.ocean
                    jax.block_until_ready(res)
                    dt_wall = time.perf_counter() - t0
            finally:
                bulk.nemo_orca2_k0_surface_card = original_surface_card
        step_times_s.append(dt_wall)
        if thread_ticks_before is not None:
            require(process_ticks_before is not None, "process CPU baseline is missing")
            cpu_row = summarize_thread_cpu_ticks(
                thread_ticks_before,
                read_thread_cpu_ticks(),
                dt_wall,
                process_ticks_before=process_ticks_before,
                process_ticks_after=read_process_total_cpu_ticks(),
            )
            cpu_row["kt"] = kt
            cpu_row["thread_affinity_record"] = read_thread_affinity_record(affinity)
            require(
                cpu_row["thread_affinity_record"]["tf_XLAEigen_thread_count"]
                == requested_cores,
                f"kt={kt}: tf_XLAEigen pool does not follow the process affinity: "
                f"{cpu_row['thread_affinity_record']['tf_XLAEigen_thread_count']} "
                f"workers for {requested_cores} CPUs",
            )
            step_cpu_telemetry.append(cpu_row)

        carry_s0 = res.next_s0
        carry_s2 = res.next_s2
        prior_surface = res.surface

        if kt in daily_check_steps:
            ref_path = exact_ref_dir / f"candidate_step{kt:08d}.npz"
            require(ref_path.is_file(), f"missing exact reference snapshot: {ref_path}")
            with np.load(ref_path) as ref_npz:
                t_ref = np.asarray(ref_npz["T"], dtype=np.float64)
            t_cand = np.asarray(state.T.data, dtype=np.float64)
            score = score_daily_temperature_against_reference(t_cand, t_ref, wet_3d, kt=kt)
            score["reference_snapshot"] = ref_path.name
            score["reference_sha256"] = sha256_file(ref_path)
            daily_accuracy.append(score)

    for field_name in ("T", "S", "u", "v", "eta"):
        arr = np.asarray(getattr(state, field_name).data)
        require(bool(np.isfinite(arr).all()), f"arm {arm_id}: non-finite terminal {field_name}")

    timing = compute_step_timing_summary(
        step_times_s, min_steady_steps=min_steady_steps
    )
    memory = read_memory_telemetry()
    dev = jax.devices()[0]

    arm_report = {
        "arm_id": arm_id,
        "variant": (
            "exact" if arm_id == "exact_cpu_fp64"
            else "fast_literal" if arm_id in FAST_LITERAL_ARM_IDS
            else variant
        ),
        "backend": jax.default_backend(),
        "device_kind": getattr(dev, "device_kind", str(dev)),
        "dtype": str(expected_dtype),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "transcendentals": get_policy().transcendentals,
        "setup_wall_s": setup_wall_s,
        "timing": timing,
        "memory": memory,
        "daily_accuracy": daily_accuracy,
        "cpu_affinity": list(affinity),
        "requested_cores": requested_cores,
        "thread_environment": {
            key: os.environ.get(key)
            for key in (
                "XLA_FLAGS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "step_cpu_telemetry": step_cpu_telemetry,
        "switch_inventory_check": inventory_check,
        "provenance": worktree_stamp(allow_dirty=allow_dirty_provenance),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    if step_cpu_telemetry:
        steady_cpu = step_cpu_telemetry[WARMUP_STEPS:]
        arm_report["cpu_utilization"] = {
            "measurement": (
                "total /proc/self/stat process utime+stime delta divided by "
                "steady-step wall time; no per-thread activity cutoff"
            ),
            "steady_median_cores_busy": float(
                np.median([row["cores_busy"] for row in steady_cpu])
            ),
            "steady_mean_cores_busy": float(
                np.mean([row["cores_busy"] for row in steady_cpu])
            ),
            "steady_thread_affinity_record": steady_cpu[-1]["thread_affinity_record"],
            "steady_thread_cpu_seconds": [
                {"kt": row["kt"], "threads": row["thread_cpu_seconds"]}
                for row in steady_cpu
            ],
        }
    if profile_dir is not None:
        trace_path, _ = load_perfetto_events(profile_dir)
        arm_report["profile"] = {
            "step": ROUND349_PROFILE_STEP,
            "annotation": ROUND349_PROFILE_ANNOTATION,
            "trace_path": str(trace_path),
        }

    out_path = output_dir / (artifact_name or f"arm_{arm_id}.json")
    out_path.write_text(json.dumps(arm_report, indent=2) + "\n", encoding="utf-8")
    return arm_report


def validate_benchmark_report(
    report: Mapping[str, Any],
    *,
    require_exact_zero_on_arm_a: bool = True,
) -> None:
    """Validate a combined 4-arm benchmark report against preregistered gates."""
    require("arms" in report and isinstance(report["arms"], Mapping), "report missing 'arms' mapping")
    arms = report["arms"]
    for arm_id in ARM_IDS:
        require(arm_id in arms, f"report missing required arm {arm_id!r}")
        arm = arms[arm_id]
        timing = arm.get("timing", {})
        require(
            int(timing.get("steady_state_steps", 0)) >= MIN_STEADY_STEPS,
            f"arm {arm_id}: steady_state_steps < {MIN_STEADY_STEPS}",
        )
        require(
            float(timing.get("steady_median_s", 0.0)) > 0.0
            and np.isfinite(float(timing.get("steady_median_s", np.nan))),
            f"arm {arm_id}: invalid steady_median_s",
        )
        require(
            float(timing.get("compile_warmup_total_s", 0.0)) > 0.0,
            f"arm {arm_id}: invalid compile_warmup_total_s",
        )
        mem = arm.get("memory", {})
        require(
            float(mem.get("host_peak_rss_mib", 0.0)) > 0.0,
            f"arm {arm_id}: invalid host_peak_rss_mib",
        )
        daily = arm.get("daily_accuracy", [])
        if arm_id == "exact_cpu_fp64":
            require(
                tuple(d["kt"] for d in daily) == DAILY_CHECK_STEPS_EXACT,
                f"arm {arm_id}: expected daily_accuracy kts {DAILY_CHECK_STEPS_EXACT}",
            )
            if require_exact_zero_on_arm_a:
                for entry in daily:
                    require(
                        entry["sst_max_abs_c"] == 0.0 and entry["t3d_max_abs_c"] == 0.0,
                        f"arm {arm_id} kt={entry['kt']} drifted from exact reference: {entry}",
                    )
        else:
            require(
                tuple(d["kt"] for d in daily) == DAILY_CHECK_STEPS_8D,
                f"arm {arm_id}: expected 8 daily_accuracy kts {DAILY_CHECK_STEPS_8D}",
            )
            for entry in daily:
                for key in (
                    "sst_rms_all_c",
                    "sst_rms_wet_c",
                    "sst_max_abs_c",
                    "t3d_rms_all_c",
                    "t3d_rms_wet_c",
                    "t3d_max_abs_c",
                ):
                    val = float(entry[key])
                    require(
                        np.isfinite(val) and val >= 0.0,
                        f"arm {arm_id} kt={entry['kt']} invalid {key}={val}",
                    )
                require(
                    float(entry["sst_max_abs_c"]) >= float(entry["sst_rms_all_c"]),
                    f"arm {arm_id} kt={entry['kt']}: sst_max_abs_c < sst_rms_all_c",
                )
                require(
                    float(entry["t3d_max_abs_c"]) >= float(entry["t3d_rms_all_c"]),
                    f"arm {arm_id} kt={entry['kt']}: t3d_max_abs_c < t3d_rms_all_c",
                )


_ACCURACY_METRICS = (
    "sst_rms_all_c",
    "sst_rms_wet_c",
    "sst_max_abs_c",
    "t3d_rms_all_c",
    "t3d_rms_wet_c",
    "t3d_max_abs_c",
)


def validate_fast_literal_benchmark_report(report: Mapping[str, Any]) -> None:
    """Validate round 344 arms and its preregistered direction vs round 343."""
    arms = report.get("arms", {})
    baseline = report.get("round343_normal_fp64", {})
    pairs = {
        "fast_literal_cpu_fp64": "normal_cpu_fp64",
        "fast_literal_gpu_fp64": "normal_gpu_fp64",
    }
    for fast_id, normal_id in pairs.items():
        require(fast_id in arms, f"round344 report missing {fast_id}")
        require(normal_id in baseline, f"round344 report missing baseline {normal_id}")
        fast = arms[fast_id]
        normal = baseline[normal_id]
        require(fast.get("variant") == "fast_literal", f"{fast_id} is not fast_literal")
        require(fast.get("dtype") == "float64", f"{fast_id} is not fp64")
        require(fast.get("transcendentals") == "native", f"{fast_id} is not native")
        require(
            int(fast.get("timing", {}).get("steady_state_steps", 0)) >= MIN_STEADY_STEPS,
            f"{fast_id} has fewer than {MIN_STEADY_STEPS} steady steps",
        )
        fast_daily = fast.get("daily_accuracy", [])
        normal_daily = normal.get("daily_accuracy", [])
        require(
            tuple(row.get("kt") for row in fast_daily) == DAILY_CHECK_STEPS_8D,
            f"{fast_id} does not contain all eight daily scores",
        )
        require(
            tuple(row.get("kt") for row in normal_daily) == DAILY_CHECK_STEPS_8D,
            f"{normal_id} does not contain all eight daily scores",
        )
        for fast_row, normal_row in zip(fast_daily, normal_daily, strict=True):
            for metric in _ACCURACY_METRICS:
                fast_value = float(fast_row[metric])
                normal_value = float(normal_row[metric])
                require(
                    np.isfinite(fast_value) and fast_value >= 0.0,
                    f"{fast_id} kt={fast_row['kt']} has invalid {metric}={fast_value}",
                )
                require(
                    fast_value < normal_value,
                    f"{fast_id} kt={fast_row['kt']} {metric}={fast_value} is not "
                    f"closer than {normal_id}={normal_value}",
                )


def assemble_fast_literal_benchmark_report(
    *,
    output_dir: Path,
    round343_report: Path = DEFAULT_ROUND343_REPORT,
) -> dict[str, Any]:
    """Assemble the two round-344 fp64 arms beside round 343's normal arms."""
    arms = {}
    for arm_id in FAST_LITERAL_ARM_IDS:
        path = output_dir / f"arm_{arm_id}.json"
        require(path.is_file(), f"missing fast-literal arm artifact {path}")
        arms[arm_id] = json.loads(path.read_text(encoding="utf-8"))
    require(round343_report.is_file(), f"missing round-343 report {round343_report}")
    old = json.loads(round343_report.read_text(encoding="utf-8"))
    baseline = {
        arm_id: old["arms"][arm_id]
        for arm_id in ("normal_cpu_fp64", "normal_gpu_fp64")
    }
    report = {
        "round": "orca2_r344",
        "author": "Codex",
        "description": "Standalone ORCA2 K0 native-transcendental fast-literal benchmark",
        "provenance": worktree_stamp(),
        "round343_report": str(round343_report),
        "round343_report_sha256": sha256_file(round343_report),
        "arms": arms,
        "round343_normal_fp64": baseline,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    validate_fast_literal_benchmark_report(report)
    path = output_dir / "orca2_round344_fast_literal_benchmark_report.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def assemble_benchmark_report_from_arm_files(
    *,
    deck_root: Path = DEFAULT_DECK_ROOT,
    core_manifest: Path = DEFAULT_CORE_MANIFEST,
    ice_cover_path: Path = DEFAULT_ICE_COVER_PATH,
    exact_ref_dir: Path = DEFAULT_EXACT_REF_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Assemble and validate the combined 4-arm report from saved arm JSON artifacts."""
    arms_out: dict[str, Any] = {}
    for arm_id in ARM_IDS:
        arm_file = output_dir / f"arm_{arm_id}.json"
        require(arm_file.is_file(), f"missing arm artifact {arm_file}")
        arms_out[arm_id] = json.loads(arm_file.read_text(encoding="utf-8"))

    inventory = [spec._asdict() for spec in build_orca2_exact_vs_normal_switch_inventory()]
    combined = {
        "round": "orca2_r343",
        "author": "Argon",
        "description": "Standalone ORCA2 K0 speed and 8-day accuracy benchmark (exact vs normal mode)",
        "provenance": worktree_stamp(),
        "deck_root": str(deck_root),
        "core_manifest": str(core_manifest),
        "ice_cover_path": str(ice_cover_path),
        "exact_ref_dir": str(exact_ref_dir),
        "switch_inventory": inventory,
        "arms": arms_out,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    validate_benchmark_report(combined)
    combined_path = output_dir / "orca2_round343_speed_benchmark_report.json"
    combined_path.write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
    return combined


def run_all_arms_via_subprocesses(
    *,
    deck_root: Path = DEFAULT_DECK_ROOT,
    core_manifest: Path = DEFAULT_CORE_MANIFEST,
    ice_cover_path: Path = DEFAULT_ICE_COVER_PATH,
    exact_ref_dir: Path = DEFAULT_EXACT_REF_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    gpu_device: str = "0",
    variant: str = "cat_a_plus_b",
) -> dict[str, Any]:
    """Run all four benchmark arms in isolated subprocesses and assemble the combined report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).resolve()

    arm_envs = {
        "exact_cpu_fp64": {"JAX_ENABLE_X64": "1", "JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": ""},
        "normal_cpu_fp64": {"JAX_ENABLE_X64": "1", "JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": ""},
        "normal_gpu_fp64": {"JAX_ENABLE_X64": "1", "CUDA_VISIBLE_DEVICES": str(gpu_device)},
        "normal_gpu_fp32": {"JAX_ENABLE_X64": "0", "CUDA_VISIBLE_DEVICES": str(gpu_device)},
    }

    for arm_id in ARM_IDS:
        env = dict(os.environ)
        env.pop("JAX_PLATFORMS", None)
        env.update(arm_envs[arm_id])
        cmd = [
            sys.executable,
            str(script_path),
            "--arm",
            arm_id,
            "--deck-root",
            str(deck_root),
            "--core-manifest",
            str(core_manifest),
            "--ice-cover-path",
            str(ice_cover_path),
            "--exact-ref-dir",
            str(exact_ref_dir),
            "--output-dir",
            str(output_dir),
            "--variant",
            variant,
        ]
        print(f"[round343] launching arm={arm_id} ...", flush=True)
        proc = subprocess.run(cmd, env=env, check=True, text=True, capture_output=True)
        if proc.stdout:
            print(proc.stdout, end="", flush=True)

    return assemble_benchmark_report_from_arm_files(
        deck_root=deck_root,
        core_manifest=core_manifest,
        ice_cover_path=ice_cover_path,
        exact_ref_dir=exact_ref_dir,
        output_dir=output_dir,
    )


def compute_sypd(seconds_per_step: float, *, year_steps: int = YEAR_STEPS) -> float:
    """Convert steady seconds/step to simulated years per wall day."""
    require(
        np.isfinite(seconds_per_step) and seconds_per_step > 0.0,
        f"seconds_per_step must be finite and positive, got {seconds_per_step}",
    )
    require(year_steps > 0, f"year_steps must be positive, got {year_steps}")
    return float(86400.0 / (year_steps * seconds_per_step))


def round349_performance_row(
    *,
    model: str,
    mode: str,
    cores: int,
    seconds_per_step: float,
    busy_cores: float,
    steps: int,
    steady_steps: int,
    cpu_ids: Sequence[int],
    background_load: Mapping[str, Any],
    artifact: str,
    measurement_provenance: Mapping[str, Any],
    timing_window: Mapping[str, int],
    seconds_per_step_statistic: str,
) -> dict[str, Any]:
    """Build one schema-checked row for the round-349 comparison table."""
    require(model in ("NEMO", "legoESM"), f"unknown model {model!r}")
    valid_modes = ("nemo_5.0.2", *ROUND349_MODES)
    require(mode in valid_modes, f"unknown mode {mode!r}; expected one of {valid_modes}")
    require(cores in ROUND349_CORE_COUNTS, f"unregistered core count {cores}")
    ids = tuple(int(cpu) for cpu in cpu_ids)
    require(len(ids) == cores, f"row has {len(ids)} CPU ids for cores={cores}")
    require(len(set(ids)) == cores, f"row CPU ids are not unique: {ids}")
    require(ids == select_physical_cpu_ids(cores), f"row CPU ids are not registered physical cores: {ids}")
    require(steps > 0, f"row steps must be positive, got {steps}")
    require(
        0 < steady_steps <= steps,
        f"row steady_steps={steady_steps} must be in 1..{steps}",
    )
    require(np.isfinite(busy_cores) and busy_cores >= 0.0, f"invalid busy_cores={busy_cores}")
    require(bool(background_load), "row is missing background-load provenance")
    require(bool(artifact), "row is missing artifact provenance")
    require(
        all(measurement_provenance.get(key) for key in ("backend", "dtype", "identity")),
        "row is missing backend/dtype/identity provenance",
    )
    require(set(timing_window) == {"first_step", "last_step"}, "row timing window is missing")
    require(bool(seconds_per_step_statistic), "row timing statistic is missing")
    sypd = compute_sypd(seconds_per_step)
    return {
        "model": model,
        "mode": mode,
        "cores": int(cores),
        "cpu_ids": list(ids),
        "steps": int(steps),
        "steady_steps": int(steady_steps),
        "seconds_per_step": float(seconds_per_step),
        "sypd": sypd,
        "core_seconds_per_step": float(cores * seconds_per_step),
        "busy_cores": float(busy_cores),
        "background_load": dict(background_load),
        "artifact": artifact,
        "measurement_provenance": dict(measurement_provenance),
        "timing_window": dict(timing_window),
        "seconds_per_step_statistic": seconds_per_step_statistic,
    }


def validate_round349_report(
    report: Mapping[str, Any], *, require_complete: bool = True
) -> None:
    """Validate report schema, arithmetic, full sweep, and profile coverage."""
    require(report.get("round") == "orca2_r349", "round-349 report has wrong round")
    rows = report.get("rows")
    require(isinstance(rows, list) and bool(rows), "round-349 report has no rows")
    seen: set[tuple[str, str, int]] = set()
    for row in rows:
        key = (str(row.get("model")), str(row.get("mode")), int(row.get("cores", 0)))
        require(key not in seen, f"duplicate round-349 row {key}")
        seen.add(key)
        rebuilt = round349_performance_row(
            model=key[0],
            mode=key[1],
            cores=key[2],
            seconds_per_step=float(row.get("seconds_per_step", 0.0)),
            busy_cores=float(row.get("busy_cores", -1.0)),
            steps=int(row.get("steps", 0)),
            steady_steps=int(row.get("steady_steps", 0)),
            cpu_ids=row.get("cpu_ids", []),
            background_load=row.get("background_load", {}),
            artifact=str(row.get("artifact", "")),
            measurement_provenance=row.get("measurement_provenance", {}),
            timing_window=row.get("timing_window", {}),
            seconds_per_step_statistic=str(row.get("seconds_per_step_statistic", "")),
        )
        expected_window = {"first_step": 7, "last_step": 92} if key[0] == "NEMO" else {"first_step": 3, "last_step": 18}
        expected_statistic = "MPI AVG elapsed / frequency" if key[0] == "NEMO" else "steady median"
        require(row["timing_window"] == expected_window, f"row {key} has wrong timing window")
        require(row["seconds_per_step_statistic"] == expected_statistic, f"row {key} has wrong statistic")
        if key[0] == "legoESM":
            thread_record = row.get("thread_affinity_record", {})
            require(
                thread_record.get("affinity_exact") is True,
                f"row {key} lacks an exact thread-affinity record",
            )
            require(
                thread_record.get("expected_cpu_ids") == row["cpu_ids"],
                f"row {key} thread affinity differs from row CPU IDs",
            )
            require(
                len(row.get("per_thread_cpu_seconds_by_step", []))
                == int(row["steady_steps"]),
                f"row {key} lacks one per-thread CPU-seconds list per steady step",
            )
        for arithmetic_key in ("sypd", "core_seconds_per_step"):
            require(
                np.isclose(
                    float(row.get(arithmetic_key, np.nan)),
                    float(rebuilt[arithmetic_key]),
                    rtol=1.0e-12,
                    atol=0.0,
                ),
                f"row {key} has invalid {arithmetic_key}",
            )
        if key[0] == "legoESM" and require_complete:
            require(
                int(row.get("steady_steps", 0)) >= ROUND349_STEADY_STEPS,
                f"row {key} has fewer than {ROUND349_STEADY_STEPS} steady steps",
            )
    if require_complete:
        expected = {
            ("NEMO", "nemo_5.0.2", n) for n in ROUND349_CORE_COUNTS
        } | {
            ("legoESM", mode, n)
            for mode in ROUND349_MODES
            for n in ROUND349_CORE_COUNTS
        }
        require(
            seen == expected,
            f"report row set mismatch: missing={sorted(expected-seen)} "
            f"extra={sorted(seen-expected)}",
        )
        profiles = report.get("profiles", [])
        require(isinstance(profiles, list), "profiles must be a list")
        profile_cores = {int(profile.get("cores", 0)) for profile in profiles}
        require(profile_cores == {1, 40}, f"profiles must cover cores 1 and 40, got {profile_cores}")
        for profile in profiles:
            require(profile.get("mode") == "exact", "profile must use exact mode")
            require(bool(profile.get("cost_centres")), "profile has no cost centres")
            require(profile.get("profile_step") == ROUND349_PROFILE_STEP, "profile step is not kt=3")
            require(bool(profile.get("trace_path")), "profile trace path is missing")
            require(bool(profile.get("provenance")), "profile provenance is missing")
            require(bool(profile.get("background_load")), "profile background load is missing")
            require(bool(profile.get("measurement_provenance")), "profile identity is missing")
            for key in (
                "compiled_kernel_wall_fraction",
                "eager_or_dispatch_fraction",
                "eager_surface_wall_fraction",
            ):
                value = float(profile.get(key, np.nan))
                require(np.isfinite(value) and 0.0 <= value <= 1.0, f"invalid profile {key}={value}")
            compiled = float(profile["compiled_kernel_wall_fraction"])
            eager = float(profile["eager_or_dispatch_fraction"])
            require(
                np.isclose(compiled + eager, 1.0, rtol=0.0, atol=1.0e-12),
                "compiled and eager/dispatch profile fractions must sum to one",
            )


_PROFILE_COST_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("surface bulk flux", ("bulk_flux", "nemo_surface", "nemo_blk", "sbc", "surface_card")),
    ("tracer advection", ("tracer", "traadv", "fct", "advection")),
    ("TKE / vertical mixing", ("tke", "vertical_mixing", "zdf", "thomas", "tridiagonal")),
    ("EOS / density", ("eos", "density", "rho", "bn2", "equation_of_state")),
    ("barotropic / free surface", ("barotropic", "free_surface", "dynspg", "ssh", "eta")),
    ("momentum / vorticity", ("momentum", "vorticity", "dynvor", "wzv", "velocity")),
    ("lateral mixing / GM-Redi", ("lateral_mixing", "gm_redi", "redi", "isoneutral", "mle")),
    ("shortwave penetration", ("shortwave", "fraqsr", "traqsr", "chlorophyll")),
    ("freshwater / runoff", ("freshwater", "runoff", "restoring", "sfx", "emp")),
    ("compiled libm callback", ("pure_callback", "scalar_libm", "custom-call", "custom_call")),
    ("optimization barrier / copy", ("optimization_barrier", "copy", "bitcast")),
    ("mask / halo / north fold", ("mask", "halo", "north_fold", "lbc")),
    ("pressure gradient", ("pressure", "dynhpg", "hydrostatic")),
    ("bottom drag", ("bottom_drag", "drag", "drg")),
)


def _profile_event_text(event: Mapping[str, Any]) -> str:
    args = event.get("args", {})
    try:
        args_text = json.dumps(args, sort_keys=True, default=str)
    except TypeError:
        args_text = str(args)
    return f"{event.get('name', '')} {args_text}".lower()


def _profile_cost_centre(event: Mapping[str, Any], *, compiled: bool) -> str:
    text_value = _profile_event_text(event)
    for category, patterns in _PROFILE_COST_PATTERNS:
        if any(pattern in text_value for pattern in patterns):
            return category
    return "other compiled XLA kernels" if compiled else "Python / eager dispatch"


def _interval_union_duration(intervals: Sequence[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    ordered = sorted((float(start), float(end)) for start, end in intervals if end > start)
    if not ordered:
        return 0.0
    total = 0.0
    start, end = ordered[0]
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def summarize_round349_perfetto(
    trace_path: Path,
    *,
    cores: int,
    mode: str = "exact",
    top_n: int = 20,
) -> dict[str, Any]:
    """Aggregate the registered one-step trace into profile cost centres."""
    resolved, events = load_perfetto_events(trace_path)
    thread_names = perfetto_thread_names(events)
    annotations = [
        event
        for event in events
        if event.get("ph") == "X" and event.get("name") == ROUND349_PROFILE_ANNOTATION
    ]
    require(len(annotations) == 1, f"expected one profile annotation, found {len(annotations)}")
    annotation = annotations[0]
    step_start = float(annotation["ts"])
    step_end = step_start + float(annotation["dur"])
    require(step_end > step_start, "profile annotation has non-positive duration")
    surface_annotations = [
        event
        for event in events
        if event.get("ph") == "X" and event.get("name") == ROUND349_SURFACE_ANNOTATION
    ]
    require(
        len(surface_annotations) == 1,
        f"expected one explicit surface annotation, found {len(surface_annotations)}",
    )
    surface_annotation = surface_annotations[0]

    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    compiled_intervals: list[tuple[float, float]] = []
    surface_eager_intervals: list[tuple[float, float]] = []
    for event in events:
        if event.get("ph") != "X" or event is annotation:
            continue
        event_start = float(event.get("ts", 0.0))
        event_end = event_start + float(event.get("dur", 0.0))
        clipped_start = max(step_start, event_start)
        clipped_end = min(step_end, event_end)
        if clipped_end <= clipped_start:
            continue
        thread_name = thread_names.get(
            (event.get("pid"), event.get("tid")),
            f"{event.get('pid')}/{event.get('tid')}",
        )
        thread_lower = thread_name.lower()
        compiled = any(
            marker in thread_lower
            for marker in ("xla", "eigen", "pjrt", "tf_foreach", "worker")
        )
        centre = _profile_cost_centre(event, compiled=compiled)
        duration = clipped_end - clipped_start
        totals[centre] = totals.get(centre, 0.0) + duration
        counts[centre] = counts.get(centre, 0) + 1
        if compiled:
            compiled_intervals.append((clipped_start, clipped_end))
        if event is surface_annotation:
            surface_eager_intervals.append((clipped_start, clipped_end))

    total_inclusive = sum(totals.values())
    require(total_inclusive > 0.0, "profile contains no complete events inside step")
    step_duration = step_end - step_start
    compiled_wall = min(step_duration, _interval_union_duration(compiled_intervals))
    surface_eager_wall = min(step_duration, _interval_union_duration(surface_eager_intervals))
    rows = [
        {
            "rank": rank,
            "cost_centre": name,
            "inclusive_cpu_s": duration / 1.0e6,
            "fraction_of_classified_inclusive_cpu": duration / total_inclusive,
            "event_count": counts[name],
        }
        for rank, (name, duration) in enumerate(
            sorted(totals.items(), key=lambda item: (-item[1], item[0]))[:top_n],
            start=1,
        )
    ]
    compiled_fraction = compiled_wall / step_duration
    return {
        "mode": mode,
        "cores": int(cores),
        "profile_step": ROUND349_PROFILE_STEP,
        "trace_path": str(resolved),
        "step_wall_s": step_duration / 1.0e6,
        "compiled_kernel_wall_fraction": compiled_fraction,
        "eager_or_dispatch_fraction": 1.0 - compiled_fraction,
        "eager_surface_wall_fraction": surface_eager_wall / step_duration,
        "inclusive_cpu_time_note": (
            "inclusive trace-event times can overlap; shares rank named cost centres; "
            "unclassified work remains in other compiled/Python categories; compiled/eager "
            "coverage is heuristic and the synchronized surface fraction is non-exclusive"
        ),
        "cost_centres": rows,
    }


_NEMO_STEADY_HEADER_RE = re.compile(
    r"Performance statistics between time step\s+(\d+)\s+and\s+(\d+)", re.IGNORECASE
)
_NEMO_STEP_ROW_RE = re.compile(
    r"^\s*step\s*\|\s*([0-9.Ee+-]+)s.*\|\s*([0-9]+)\s*$"
)


def parse_nemo_timing_output(text: str) -> dict[str, Any]:
    """Parse NEMO's steady timing block, refusing the whole-run block."""
    lines = text.splitlines()
    header_index = None
    first_step = last_step = None
    for index, line in enumerate(lines):
        match = _NEMO_STEADY_HEADER_RE.search(line)
        if match:
            header_index = index
            first_step, last_step = int(match.group(1)), int(match.group(2))
            break
    require(header_index is not None, "NEMO timing output lacks steady time-step block")
    steady_lines = lines[header_index + 1 :]
    # timing.F90 writes the AVG block only when jpnij > 1; a single rank writes only its local block.
    block = "Timing : AVG values"
    if not any(line.strip().startswith(block) for line in steady_lines):
        block = "Timing : values for local MPI process"
    in_average = False
    for line in steady_lines:
        if line.strip().startswith(block):
            in_average = True
            continue
        if not in_average:
            continue
        match = _NEMO_STEP_ROW_RE.match(line)
        if match:
            elapsed_s = float(match.group(1))
            frequency = int(match.group(2))
            require(frequency > 0 and elapsed_s > 0.0, "invalid NEMO steady timing row")
            require(
                frequency == last_step - first_step + 1,
                f"NEMO steady frequency {frequency} does not match window "
                f"{first_step}..{last_step}",
            )
            return {
                "first_step": first_step,
                "last_step": last_step,
                "steady_steps": frequency,
                "steady_elapsed_s": elapsed_s,
                "seconds_per_step": elapsed_s / frequency,
            }
    raise GateError("NEMO timing output lacks AVG steady step row")


def parse_gnu_time_metrics(text: str) -> dict[str, float]:
    """Parse the operator script's stable GNU-time key/value format."""
    values: dict[str, float] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, raw = line.split("=", maxsplit=1)
        if key in {"elapsed_s", "user_s", "system_s", "max_rss_kib"}:
            values[key] = float(raw)
    require(
        set(values) == {"elapsed_s", "user_s", "system_s", "max_rss_kib"},
        f"GNU time metrics incomplete: {sorted(values)}",
    )
    require(values["elapsed_s"] > 0.0, "GNU time elapsed_s must be positive")
    require(values["user_s"] >= 0.0 and values["system_s"] >= 0.0, "negative GNU CPU time")
    return values


def build_round349_nemo_artifact(
    run_dir: Path,
    *,
    cores: int,
    cpu_ids: Sequence[int],
    preflight_path: Path,
) -> dict[str, Any]:
    """Build a NEMO arm artifact from timing.output and GNU time telemetry."""
    timing_path = run_dir / "timing.output"
    time_path = run_dir / "gnu_time.txt"
    require(timing_path.is_file(), f"missing {timing_path}")
    require(time_path.is_file(), f"missing {time_path}")
    require(preflight_path.is_file(), f"missing {preflight_path}")
    timing = parse_nemo_timing_output(timing_path.read_text(encoding="utf-8"))
    require(
        (timing["first_step"], timing["last_step"], timing["steady_steps"]) == (7, 92, 86),
        f"NEMO steady window is not registered 7..92/86: {timing}",
    )
    time_metrics = parse_gnu_time_metrics(time_path.read_text(encoding="utf-8"))
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    binding_paths = sorted(run_dir.glob("binding_rank*.txt"))
    require(len(binding_paths) == cores, f"expected {cores} rank binding records")
    rank_bindings: dict[str, list[int]] = {}
    for path in binding_paths:
        match = re.search(r"Cpus_allowed_list:\s*(\S+)", path.read_text(encoding="utf-8"))
        require(match is not None, f"invalid rank binding record {path}")
        allowed = list(parse_cpu_id_list(match.group(1)))
        require(len(allowed) == 1, f"rank binding is not one hardware thread: {path}: {allowed}")
        rank_bindings[path.stem] = allowed
    observed_cpu_ids = sorted({cpu for allowed in rank_bindings.values() for cpu in allowed})
    require(observed_cpu_ids == sorted(cpu_ids), "observed NEMO rank affinity differs from request")
    rank_time_paths = sorted(run_dir.glob("rank_time*.txt"))
    require(len(rank_time_paths) == cores, f"expected {cores} rank CPU-time records")
    rank_times = {
        path.stem: parse_gnu_time_metrics(path.read_text(encoding="utf-8"))
        for path in rank_time_paths
    }
    manifest = json.loads((run_dir / "producer_manifest.json").read_text(encoding="utf-8"))
    binary_sha = sha256_file(run_dir / "nemo")
    require(binary_sha == manifest.get("binary_sha256"), "NEMO binary fails producer manifest")
    busy = sum(value["user_s"] + value["system_s"] for value in rank_times.values()) / time_metrics["elapsed_s"]
    row = round349_performance_row(
        model="NEMO",
        mode="nemo_5.0.2",
        cores=cores,
        seconds_per_step=float(timing["seconds_per_step"]),
        busy_cores=busy,
        steps=96,
        steady_steps=int(timing["steady_steps"]),
        cpu_ids=observed_cpu_ids,
        background_load=preflight,
        artifact=str(run_dir),
        measurement_provenance={
            "backend": "NEMO 5.0.2 / Open MPI",
            "dtype": "float64",
            "identity": {
                "binary_sha256": binary_sha,
                "producer_manifest_binary_sha256": manifest["binary_sha256"],
                "namelist_sha256": sha256_file(run_dir / "namelist_cfg"),
            },
        },
        timing_window={"first_step": 7, "last_step": 92},
        seconds_per_step_statistic="MPI AVG elapsed / frequency",
    )
    return {
        "round": "orca2_r349",
        "kind": "nemo_cpu_arm",
        "row": row,
        "timing": timing,
        "gnu_time": time_metrics,
        "binary_sha256": binary_sha,
        "producer_manifest_binary_sha256": manifest["binary_sha256"],
        "rank_bindings": rank_bindings,
        "rank_cpu_times": rank_times,
        "namelist_sha256": sha256_file(run_dir / "namelist_cfg"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def _read_proc_cpu_counters() -> dict[int, tuple[int, int]]:
    """Return logical-CPU ``(idle, total)`` tick counters from /proc/stat."""
    counters: dict[int, tuple[int, int]] = {}
    for line in Path("/proc/stat").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields or not re.fullmatch(r"cpu\d+", fields[0]):
            continue
        values = [int(value) for value in fields[1:]]
        require(len(values) >= 5, f"short /proc/stat row: {line}")
        cpu_id = int(fields[0][3:])
        idle = values[3] + values[4]
        counters[cpu_id] = (idle, sum(values))
    require(bool(counters), "/proc/stat contains no per-CPU counters")
    return counters


def _read_process_cpu_ticks() -> dict[int, ProcessCpuSample]:
    """Return live process CPU ticks, names, and parents for load preflight."""
    rows: dict[int, ProcessCpuSample] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat_line = (entry / "stat").read_text(encoding="utf-8")
            user_ticks, system_ticks = parse_cpu_ticks(stat_line)
            stat_rest = stat_line[stat_line.rindex(")") + 1:].split()
            parent_pid = int(stat_rest[1])
            command = (entry / "comm").read_text(encoding="utf-8").strip()
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        rows[int(entry.name)] = ProcessCpuSample(
            user_ticks + system_ticks, command, parent_pid
        )
    return rows


def find_nemo_processes(processes: Mapping[int, ProcessCpuSample]) -> list[int]:
    """Find NEMO executables and an immediate Open MPI launcher parent."""
    nemo_names = {"nemo", "nemo.exe"}
    launcher_names = {"mpirun", "orterun", "prterun"}
    direct = {
        pid for pid, sample in processes.items() if sample.command in nemo_names
    }
    launchers = {
        sample.parent_pid
        for pid, sample in processes.items()
        if pid in direct
        and sample.parent_pid in processes
        and processes[sample.parent_pid].command in launcher_names
    }
    return sorted(direct | launchers)


def _available_memory_gib() -> float:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return float(line.split()[1]) / (1024.0 * 1024.0)
    raise GateError("/proc/meminfo lacks MemAvailable")


def round349_nemo_preflight(
    cpu_ids: Sequence[int], *, sample_seconds: float = ROUND349_PREFLIGHT_SAMPLE_SECONDS
) -> dict[str, Any]:
    """Refuse occupied NEMO benchmark resources using a measured window."""
    ids = tuple(int(cpu) for cpu in cpu_ids)
    require(bool(ids), "preflight CPU list must not be empty")
    require(sample_seconds > 0.0, f"sample_seconds must be positive, got {sample_seconds}")
    available_gib = _available_memory_gib()
    require(available_gib >= 150.0, f"available RAM {available_gib:.1f} GiB < 150 GiB")

    before_processes = _read_process_cpu_ticks()
    existing_nemo = find_nemo_processes(before_processes)
    require(not existing_nemo, f"NEMO process already running: pids={existing_nemo}")
    topology = {row.cpu_id: row for row in read_cpu_topology()}
    require(all(cpu in topology for cpu in ids), "selected CPU absent from topology")
    sampled_ids = tuple(sorted({sibling for cpu in ids for sibling in topology[cpu].siblings}))
    before_cpu = _read_proc_cpu_counters()
    started = time.monotonic()
    time.sleep(sample_seconds)
    elapsed = time.monotonic() - started
    after_cpu = _read_proc_cpu_counters()
    after_processes = _read_process_cpu_ticks()

    utilization: dict[int, float] = {}
    for cpu in sampled_ids:
        require(cpu in before_cpu and cpu in after_cpu, f"CPU {cpu} absent from /proc/stat")
        idle0, total0 = before_cpu[cpu]
        idle1, total1 = after_cpu[cpu]
        delta_total = total1 - total0
        require(delta_total > 0, f"CPU {cpu} total ticks did not advance")
        utilization[cpu] = 1.0 - (idle1 - idle0) / delta_total
    busy_pinned = {
        cpu: value
        for cpu, value in utilization.items()
        if value > ROUND349_PINNED_CORE_BUSY_FRACTION
    }
    require(
        not busy_pinned,
        f"selected physical cores or hyperthread siblings exceed 10% utilization: {busy_pinned}",
    )

    after_nemo = find_nemo_processes(after_processes)
    require(not after_nemo, f"NEMO process appeared during preflight: pids={after_nemo}")

    excluded = {os.getpid(), os.getppid()}
    heavy: list[dict[str, Any]] = []
    for pid, sample in after_processes.items():
        if pid in excluded:
            continue
        prior = before_processes.get(
            pid, ProcessCpuSample(0, sample.command, sample.parent_pid)
        )
        average_cores = max(0, sample.ticks - prior.ticks) / CLK_TCK / elapsed
        if average_cores >= ROUND349_OTHER_PROCESS_BUSY_CORES:
            heavy.append(
                {"pid": pid, "command": sample.command, "average_cores": average_cores}
            )
    require(not heavy, f"unrelated heavy process active during preflight: {heavy}")
    return {
        "sample_seconds": elapsed,
        "available_memory_gib": available_gib,
        "pinned_cpu_utilization": {str(cpu): value for cpu, value in utilization.items()},
        "requested_cpu_ids": list(ids),
        "sampled_cpu_ids_including_siblings": list(sampled_ids),
        "max_pinned_cpu_utilization": max(utilization.values()),
        "pinned_core_busy_fraction": ROUND349_PINNED_CORE_BUSY_FRACTION,
        "heavy_process_threshold_cores": ROUND349_OTHER_PROCESS_BUSY_CORES,
        "heavy_processes": heavy,
        "nemo_processes_before": existing_nemo,
        "nemo_processes_after": after_nemo,
        "loadavg": list(os.getloadavg()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def _round349_background_snapshot() -> dict[str, Any]:
    return {
        "loadavg": list(os.getloadavg()),
        "available_memory_gib": _available_memory_gib(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def round349_steady_timing_window(steps: int) -> dict[str, int]:
    """Derive a non-empty, parity-balanced steady window after warm-up."""
    steady_window_steps = int(steps) - WARMUP_STEPS
    require(steady_window_steps > 0, "steady timing window is empty")
    require(
        steady_window_steps % 2 == 0,
        f"steady timing window must have equal nn_fsbc parities, got "
        f"{steady_window_steps} steps",
    )
    return {"first_step": WARMUP_STEPS + 1, "last_step": int(steps)}


def run_round349_legoesm_child(
    *,
    mode: str,
    cores: int,
    cpu_ids: Sequence[int],
    steps: int,
    output_dir: Path,
    smoke: bool = False,
    profile: bool = False,
    deck_root: Path = DEFAULT_DECK_ROOT,
    core_manifest: Path = DEFAULT_CORE_MANIFEST,
    ice_cover_path: Path = DEFAULT_ICE_COVER_PATH,
    preflight_path: Path | None = None,
) -> dict[str, Any]:
    """Run one affinity-pinned round-349 legoESM child process."""
    require(mode in ROUND349_MODES, f"unknown round-349 mode {mode!r}")
    require(cores in ROUND349_CORE_COUNTS, f"unregistered core count {cores}")
    require(len(tuple(cpu_ids)) == cores, "CPU-id count does not match cores")
    if smoke:
        require(mode == "fast_literal", "round-349 smoke is fast-literal only")
        require(steps == ROUND349_SMOKE_STEPS, f"smoke requires {ROUND349_SMOKE_STEPS} steps")
        min_steady = steps - WARMUP_STEPS
    elif profile:
        require(mode == "exact", "round-349 profiles use exact/libm mode")
        require(cores in (1, 40), "round-349 profiles are registered only at n=1 and n=40")
        require(steps == ROUND349_PROFILE_STEP, f"profile requires {ROUND349_PROFILE_STEP} steps")
        min_steady = 1
    else:
        require(steps >= ROUND349_FULL_STEPS, f"full arm requires >= {ROUND349_FULL_STEPS} steps")
        min_steady = ROUND349_STEADY_STEPS
    timing_window = None
    if not profile:
        timing_window = round349_steady_timing_window(steps)
    if not smoke:
        require(preflight_path is not None and preflight_path.is_file(), "measured preflight is required")

    arm_id = "exact_cpu_fp64" if mode == "exact" else "fast_literal_cpu_fp64"
    suffix = f"{mode}_n{cores}"
    artifact_name = (
        f"legoesm_profile_driver_{suffix}.json"
        if profile
        else f"legoesm_{suffix}.json"
    )
    trace_dir = output_dir / f"trace_{suffix}" if profile else None
    result = run_single_benchmark_arm(
        arm_id,
        deck_root=deck_root,
        core_manifest=core_manifest,
        ice_cover_path=ice_cover_path,
        output_dir=output_dir,
        steps_override=steps,
        min_steady_steps=min_steady,
        collect_accuracy=False,
        requested_cores=cores,
        expected_cpu_ids=cpu_ids,
        profile_dir=trace_dir,
        artifact_name=artifact_name,
        allow_dirty_provenance=smoke,
    )
    background_load = (
        _round349_background_snapshot()
        if preflight_path is None
        else json.loads(preflight_path.read_text(encoding="utf-8"))
    )
    result["round349"] = {
        "mode": mode,
        "cores": cores,
        "smoke": smoke,
        "profile": profile,
        "background_load": background_load,
    }
    if profile:
        profile_report = summarize_round349_perfetto(
            Path(result["profile"]["trace_path"]), cores=cores, mode=mode
        )
        profile_report["provenance"] = result["provenance"]
        profile_report["background_load"] = background_load
        profile_report["measurement_provenance"] = {
            "backend": result["backend"],
            "dtype": result["dtype"],
            "identity": {
                "worktree": result["provenance"],
                "core_manifest_sha256": sha256_file(core_manifest),
                "ice_cover_sha256": sha256_file(ice_cover_path),
            },
        }
        profile_path = output_dir / f"profile_{suffix}.json"
        profile_path.write_text(json.dumps(profile_report, indent=2) + "\n", encoding="utf-8")
        result["round349"]["profile_report"] = str(profile_path)
    else:
        utilization = result["cpu_utilization"]
        require(timing_window is not None, "performance arm lacks a timing window")
        row = round349_performance_row(
            model="legoESM",
            mode=mode,
            cores=cores,
            seconds_per_step=float(result["timing"]["steady_median_s"]),
            busy_cores=float(utilization["steady_mean_cores_busy"]),
            steps=int(result["timing"]["total_steps"]),
            steady_steps=int(result["timing"]["steady_state_steps"]),
            cpu_ids=cpu_ids,
            background_load=result["round349"]["background_load"],
            artifact=str(output_dir / artifact_name),
            measurement_provenance={
                "backend": result["backend"],
                "dtype": result["dtype"],
                "identity": {
                    "worktree": result["provenance"],
                    "core_manifest": str(core_manifest),
                    "core_manifest_sha256": sha256_file(core_manifest),
                    "ice_cover": str(ice_cover_path),
                    "ice_cover_sha256": sha256_file(ice_cover_path),
                },
            },
            timing_window=timing_window,
            seconds_per_step_statistic="steady median",
        )
        row["thread_affinity_record"] = utilization["steady_thread_affinity_record"]
        row["per_thread_cpu_seconds_by_step"] = utilization[
            "steady_thread_cpu_seconds"
        ]
        result["round349_row"] = row
    artifact_path = output_dir / artifact_name
    artifact_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _round349_child_environment(cores: int) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "TMPDIR": str(ROUND349_SCRATCH_ROOT),
            "CUDA_VISIBLE_DEVICES": "",
            "JAX_PLATFORMS": "cpu",
            "JAX_ENABLE_X64": "1",
            # Installed jaxlib xla_client.py:64-87 exposes no pool-size input;
            # the measured XLA worker pool follows this process's taskset mask.
            "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=true",
            "OMP_NUM_THREADS": str(cores),
            "MKL_NUM_THREADS": str(cores),
            "OPENBLAS_NUM_THREADS": str(cores),
            "NUMEXPR_NUM_THREADS": str(cores),
        }
    )
    return env


def build_round349_legoesm_commands(
    *,
    output_dir: Path,
    core_counts: Sequence[int],
    steps: int,
    smoke: bool,
    include_profiles: bool,
) -> list[tuple[list[str], dict[str, str]]]:
    """Build the isolated taskset commands used by dry-run and acquisition."""
    taskset = shutil.which("taskset")
    require(taskset is not None, "taskset is required for round-349 CPU binding")
    script_path = Path(__file__).resolve()
    commands: list[tuple[list[str], dict[str, str]]] = []
    modes = ("fast_literal",) if smoke else ROUND349_MODES
    for cores in core_counts:
        require(cores in ROUND349_CORE_COUNTS, f"unregistered core count {cores}")
        cpu_ids = select_physical_cpu_ids(cores)
        cpu_csv = format_cpu_id_list(cpu_ids)
        for mode in modes:
            preflight_path = output_dir / f"preflight_{mode}_n{cores}.json"
            cmd = [
                taskset,
                "-c",
                cpu_csv,
                sys.executable,
                str(script_path),
                "--round349-child",
                "--round349-mode",
                mode,
                "--round349-cores",
                str(cores),
                "--round349-cpu-ids",
                cpu_csv,
                "--round349-steps",
                str(steps),
                "--round349-preflight-path",
                str(preflight_path),
                "--output-dir",
                str(output_dir),
            ]
            if smoke:
                cmd.append("--round349-smoke")
            commands.append((cmd, _round349_child_environment(cores)))
        if include_profiles and cores in (1, 40):
            profile_preflight = output_dir / f"preflight_profile_exact_n{cores}.json"
            profile_cmd = [
                taskset,
                "-c",
                cpu_csv,
                sys.executable,
                str(script_path),
                "--round349-child",
                "--round349-mode",
                "exact",
                "--round349-cores",
                str(cores),
                "--round349-cpu-ids",
                cpu_csv,
                "--round349-steps",
                str(ROUND349_PROFILE_STEP),
                "--round349-profile",
                "--round349-preflight-path",
                str(profile_preflight),
                "--output-dir",
                str(output_dir),
            ]
            commands.append((profile_cmd, _round349_child_environment(cores)))
    return commands


def run_round349_legoesm_sweep(
    *,
    output_dir: Path,
    core_counts: Sequence[int] = ROUND349_CORE_COUNTS,
    steps: int = ROUND349_FULL_STEPS,
    smoke: bool = False,
    include_profiles: bool = True,
    dry_run: bool = False,
) -> list[list[str]]:
    """Run or print the registered isolated legoESM CPU sweep."""
    output_resolved = output_dir.resolve()
    require(
        output_resolved.is_relative_to(ROUND349_SCRATCH_ROOT.resolve()),
        f"round-349 outputs must be below {ROUND349_SCRATCH_ROOT}",
    )
    if smoke:
        require(tuple(core_counts) == (1, 2), "registered smoke core counts are exactly 1,2")
        require(steps == ROUND349_SMOKE_STEPS, f"smoke steps must be {ROUND349_SMOKE_STEPS}")
        include_profiles = False
    else:
        require(steps >= ROUND349_FULL_STEPS, f"full sweep requires >= {ROUND349_FULL_STEPS} steps")
    commands = build_round349_legoesm_commands(
        output_dir=output_dir,
        core_counts=core_counts,
        steps=steps,
        smoke=smoke,
        include_profiles=include_profiles,
    )
    if dry_run:
        for command, env in commands:
            print(
                "ROUND349_DRY_RUN "
                + " ".join(f"{key}={env[key]}" for key in ("CUDA_VISIBLE_DEVICES", "JAX_PLATFORMS", "JAX_ENABLE_X64", "OMP_NUM_THREADS", "XLA_FLAGS"))
                + " "
                + " ".join(command)
            )
        return [command for command, _ in commands]

    require(
        not output_dir.exists() or not any(output_dir.iterdir()),
        f"round-349 output directory must be absent or empty: {output_dir}",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for command, env in commands:
        cpu_csv = command[command.index("--round349-cpu-ids") + 1]
        preflight_path = Path(command[command.index("--round349-preflight-path") + 1])
        preflight = round349_nemo_preflight(
            parse_cpu_id_list(cpu_csv),
            sample_seconds=ROUND349_PREFLIGHT_SAMPLE_SECONDS,
        )
        preflight_path.write_text(json.dumps(preflight, indent=2) + "\n", encoding="utf-8")
        print(f"[round349] launching: {' '.join(command)}", flush=True)
        subprocess.run(command, env=env, check=True, timeout=1200)
    return [command for command, _ in commands]


def _round349_hypothesis(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    exact = {
        int(row["cores"]): row
        for row in rows
        if row.get("model") == "legoESM" and row.get("mode") == "exact"
    }
    if 1 not in exact or 40 not in exact:
        return {"verdict": "UNMEASURED", "reason": "missing exact n=1 or n=40 row"}
    speedup = float(exact[1]["seconds_per_step"]) / float(exact[40]["seconds_per_step"])
    busy_40 = float(exact[40]["busy_cores"])
    verdict = "CONFIRM" if speedup < 1.5 and busy_40 < 2.0 else "REFUTE"
    return {
        "verdict": verdict,
        "n1_to_n40_speedup": speedup,
        "n40_busy_cores": busy_40,
        "confirm_rule": "speedup < 1.5 and n40 busy cores < 2.0",
    }


def _round349_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# ORCA2 round 349 CPU benchmark report",
        "",
        f"Installed jaxlib `{ROUND349_JAXLIB_CPU_CLIENT_SOURCE}` passes no CPU-pool size to "
        "`get_tfrt_cpu_client`; the measured XLA worker pool follows the process "
        "affinity selected by `taskset`.",
        "",
        "| Model | Mode | Cores | CPU IDs | s/step | Statistic/window | SYPD | core-s/step | busy cores | XLA/other workers | steps (total/steady) |",
        "|---|---|---:|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    ordered_rows = sorted(
        report["rows"], key=lambda item: (item["cores"], item["model"], item["mode"])
    )
    for row in ordered_rows:
        thread_record = row.get("thread_affinity_record", {})
        worker_counts = (
            f"{thread_record['tf_XLAEigen_thread_count']}/"
            f"{thread_record['other_worker_thread_count']}"
            if thread_record
            else "--"
        )
        lines.append(
            f"| {row['model']} | {row['mode']} | {row['cores']} | "
            f"{','.join(str(cpu) for cpu in row['cpu_ids'])} | "
            f"{row['seconds_per_step']:.6f} | {row['seconds_per_step_statistic']} "
            f"kt={row['timing_window']['first_step']}..{row['timing_window']['last_step']} | "
            f"{row['sypd']:.4f} | "
            f"{row['core_seconds_per_step']:.6f} | {row['busy_cores']:.3f} | "
            f"{worker_counts} | "
            f"{row['steps']}/{row['steady_steps']} |"
        )
    lines.extend(
        [
            "",
            "## Per-row provenance",
            "",
        ]
    )
    for row in ordered_rows:
        lines.append(
            f"- {row['model']} {row['mode']} n={row['cores']}: "
            f"artifact `{row['artifact']}`; background "
            f"`{json.dumps(row['background_load'], sort_keys=True)}`; identity "
            f"`{json.dumps(row['measurement_provenance'], sort_keys=True)}`; "
            f"thread affinity/counts "
            f"`{json.dumps(row.get('thread_affinity_record'), sort_keys=True)}`; "
            f"per-thread CPU seconds "
            f"`{json.dumps(row.get('per_thread_cpu_seconds_by_step'), sort_keys=True)}`."
        )
    lines.extend(["", f"Hypothesis verdict: **{report['hypothesis']['verdict']}**.", ""])
    for profile in report.get("profiles", []):
        profile_provenance = profile.get("provenance", {})
        lines.extend(
            [
                f"## legoESM exact profile, n={profile['cores']}",
                "",
                f"Profiled step: kt={profile['profile_step']}; "
                f"step wall={profile['step_wall_s']:.6f} s; "
                f"trace `{profile['trace_path']}`; "
                f"commit `{profile_provenance.get('commit', 'unknown')}`; "
                f"clean={profile_provenance.get('clean', 'unknown')}.",
                "",
                f"Background `{json.dumps(profile['background_load'], sort_keys=True)}`; "
                f"identity `{json.dumps(profile['measurement_provenance'], sort_keys=True)}`.",
                "",
                f"Compiled-kernel wall fraction: {profile['compiled_kernel_wall_fraction']:.4f}; "
                f"eager/dispatch fraction: {profile['eager_or_dispatch_fraction']:.4f}; "
                f"eager surface fraction: {profile['eager_surface_wall_fraction']:.4f}.",
                "",
                "| Rank | Cost centre | inclusive CPU s | share | events |",
                "|---:|---|---:|---:|---:|",
            ]
        )
        for centre in profile["cost_centres"]:
            lines.append(
                f"| {centre['rank']} | {centre['cost_centre']} | "
                f"{centre['inclusive_cpu_s']:.6f} | "
                f"{centre['fraction_of_classified_inclusive_cpu']:.4f} | "
                f"{centre['event_count']} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def assemble_round349_report(
    *,
    legoesm_dir: Path,
    nemo_dir: Path,
    output_dir: Path,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Assemble legoESM/NEMO artifacts into JSON and Markdown reports."""
    require(
        output_dir.resolve().is_relative_to(ROUND349_SCRATCH_ROOT.resolve()),
        f"round-349 report output must be below {ROUND349_SCRATCH_ROOT}",
    )
    rows: list[dict[str, Any]] = []
    for path in sorted(legoesm_dir.glob("legoesm_*.json")):
        artifact = json.loads(path.read_text(encoding="utf-8"))
        if "round349_row" in artifact and not artifact.get("round349", {}).get("smoke"):
            rows.append(artifact["round349_row"])
    for path in sorted(nemo_dir.glob("nemo_n*.json")):
        artifact = json.loads(path.read_text(encoding="utf-8"))
        if "row" in artifact:
            rows.append(artifact["row"])
    profiles = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(legoesm_dir.glob("profile_exact_n*.json"))
    ]
    report = {
        "round": "orca2_r349",
        "decision": 219,
        "description": "Per-core CPU speed of standalone ORCA2 K0",
        "formula": "SYPD = 86400 / (2920 * seconds_per_step)",
        "rows": rows,
        "profiles": profiles,
        "hypothesis": _round349_hypothesis(rows),
        "historical_nemo_np2": {
            "elapsed_s": 1767.523521,
            "steps": 2920,
            "note": "certified year run with eight-step restart output; not a round-349 row",
            "source": str(DEFAULT_ROUND349_NEMO_SOURCE / "timing.output"),
        },
        "provenance": worktree_stamp(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    validate_round349_report(report, require_complete=require_complete)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / ROUND349_REPORT_NAME
    markdown_path = output_dir / ROUND349_REPORT_MARKDOWN_NAME
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_round349_markdown(report), encoding="utf-8")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        choices=(*ARM_IDS, *FAST_LITERAL_ARM_IDS, "all", "assemble", "assemble-fast"),
        default="all",
        help=("Benchmark arm to execute, 'all' to run round 343's four arms, "
              "'assemble' for round 343, or 'assemble-fast' for round 344."),
    )
    parser.add_argument("--deck-root", type=Path, default=DEFAULT_DECK_ROOT)
    parser.add_argument("--core-manifest", type=Path, default=DEFAULT_CORE_MANIFEST)
    parser.add_argument("--ice-cover-path", type=Path, default=DEFAULT_ICE_COVER_PATH)
    parser.add_argument("--exact-ref-dir", type=Path, default=DEFAULT_EXACT_REF_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--round343-report", type=Path, default=DEFAULT_ROUND343_REPORT,
    )
    parser.add_argument("--gpu-device", type=str, default="0")
    parser.add_argument(
        "--variant",
        choices=("cat_a_plus_b", "cat_a_only"),
        default="cat_a_plus_b",
    )
    parser.add_argument("--round349-child", action="store_true")
    parser.add_argument("--round349-sweep", action="store_true")
    parser.add_argument("--round349-preflight", action="store_true")
    parser.add_argument("--round349-parse-nemo", action="store_true")
    parser.add_argument("--round349-assemble", action="store_true")
    parser.add_argument("--round349-print-cpu-ids", action="store_true")
    parser.add_argument("--round349-dry-run", action="store_true")
    parser.add_argument("--round349-smoke", action="store_true")
    parser.add_argument("--round349-profile", action="store_true")
    parser.add_argument("--round349-allow-partial", action="store_true")
    parser.add_argument("--round349-mode", choices=ROUND349_MODES, default="fast_literal")
    parser.add_argument("--round349-cores", type=int, default=1)
    parser.add_argument("--round349-cpu-ids", type=str, default="")
    parser.add_argument("--round349-core-counts", type=str, default="1,2,4,8,16,40")
    parser.add_argument("--round349-steps", type=int, default=ROUND349_FULL_STEPS)
    parser.add_argument(
        "--round349-sample-seconds",
        type=float,
        default=ROUND349_PREFLIGHT_SAMPLE_SECONDS,
    )
    parser.add_argument("--round349-output-file", type=Path, default=None)
    parser.add_argument("--round349-nemo-run-dir", type=Path, default=None)
    parser.add_argument("--round349-preflight-path", type=Path, default=None)
    parser.add_argument("--round349-legoesm-dir", type=Path, default=DEFAULT_ROUND349_OUTPUT_DIR)
    parser.add_argument(
        "--round349-nemo-dir", type=Path, default=ROUND349_SCRATCH_ROOT / "nemo_cpu_sweep"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.round349_print_cpu_ids:
        print(format_cpu_id_list(select_physical_cpu_ids(args.round349_cores)))
    elif args.round349_preflight:
        cpu_ids = parse_cpu_id_list(args.round349_cpu_ids)
        report = round349_nemo_preflight(
            cpu_ids, sample_seconds=args.round349_sample_seconds
        )
        payload = json.dumps(report, indent=2) + "\n"
        if args.round349_output_file is not None:
            args.round349_output_file.parent.mkdir(parents=True, exist_ok=True)
            args.round349_output_file.write_text(payload, encoding="utf-8")
        print(payload, end="")
    elif args.round349_parse_nemo:
        require(args.round349_nemo_run_dir is not None, "--round349-nemo-run-dir is required")
        require(args.round349_preflight_path is not None, "--round349-preflight-path is required")
        require(args.round349_output_file is not None, "--round349-output-file is required")
        cpu_ids = parse_cpu_id_list(args.round349_cpu_ids)
        artifact = build_round349_nemo_artifact(
            args.round349_nemo_run_dir,
            cores=args.round349_cores,
            cpu_ids=cpu_ids,
            preflight_path=args.round349_preflight_path,
        )
        args.round349_output_file.parent.mkdir(parents=True, exist_ok=True)
        args.round349_output_file.write_text(
            json.dumps(artifact, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"ROUND349_NEMO_RESULT n={args.round349_cores} "
            f"seconds_per_step={artifact['row']['seconds_per_step']:.6f} "
            f"SYPD={artifact['row']['sypd']:.4f} "
            f"busy_cores={artifact['row']['busy_cores']:.3f}"
        )
    elif args.round349_child:
        cpu_ids = parse_cpu_id_list(args.round349_cpu_ids)
        result = run_round349_legoesm_child(
            mode=args.round349_mode,
            cores=args.round349_cores,
            cpu_ids=cpu_ids,
            steps=args.round349_steps,
            output_dir=args.output_dir,
            smoke=args.round349_smoke,
            profile=args.round349_profile,
            deck_root=args.deck_root,
            core_manifest=args.core_manifest,
            ice_cover_path=args.ice_cover_path,
            preflight_path=args.round349_preflight_path,
        )
        if args.round349_profile:
            profile_path = result["round349"]["profile_report"]
            print(
                f"ROUND349_PROFILE mode={args.round349_mode} n={args.round349_cores} "
                f"path={profile_path}"
            )
        else:
            row = result["round349_row"]
            thread_record = row["thread_affinity_record"]
            label = "SMOKE" if args.round349_smoke else "RESULT"
            print(
                f"ROUND349_{label} mode={args.round349_mode} n={args.round349_cores} "
                f"steps={row['steps']} steady={row['steady_steps']} "
                f"seconds_per_step={row['seconds_per_step']:.6f} "
                f"SYPD={row['sypd']:.4f} busy_cores={row['busy_cores']:.3f} "
                f"tf_XLAEigen_threads={thread_record['tf_XLAEigen_thread_count']} "
                f"other_worker_threads={thread_record['other_worker_thread_count']} "
                f"thread_affinity_exact={thread_record['affinity_exact']}"
            )
    elif args.round349_sweep:
        core_counts = parse_cpu_id_list(args.round349_core_counts)
        run_round349_legoesm_sweep(
            output_dir=args.output_dir,
            core_counts=core_counts,
            steps=args.round349_steps,
            smoke=args.round349_smoke,
            include_profiles=not args.round349_smoke,
            dry_run=args.round349_dry_run,
        )
    elif args.round349_assemble:
        report = assemble_round349_report(
            legoesm_dir=args.round349_legoesm_dir,
            nemo_dir=args.round349_nemo_dir,
            output_dir=args.output_dir,
            require_complete=not args.round349_allow_partial,
        )
        print(
            f"ROUND349_REPORT rows={len(report['rows'])} "
            f"profiles={len(report['profiles'])} verdict={report['hypothesis']['verdict']} "
            f"path={args.output_dir / ROUND349_REPORT_NAME}"
        )
    elif args.arm == "assemble-fast":
        report = assemble_fast_literal_benchmark_report(
            output_dir=args.output_dir,
            round343_report=args.round343_report,
        )
        print(
            f"[round344] completed 2-arm report -> "
            f"{args.output_dir / 'orca2_round344_fast_literal_benchmark_report.json'}"
        )
        for arm_id, arm in report["arms"].items():
            timing = arm["timing"]
            print(
                f"  {arm_id}: compile_warmup={timing['compile_warmup_total_s']:.2f}s "
                f"median={timing['steady_median_s']:.4f}s/step "
                f"1yr_steady={timing['projected_year_steady_min']:.2f}min"
            )
    elif args.arm in ("all", "assemble"):
        if args.arm == "all":
            report = run_all_arms_via_subprocesses(
                deck_root=args.deck_root,
                core_manifest=args.core_manifest,
                ice_cover_path=args.ice_cover_path,
                exact_ref_dir=args.exact_ref_dir,
                output_dir=args.output_dir,
                gpu_device=args.gpu_device,
                variant=args.variant,
            )
        else:
            report = assemble_benchmark_report_from_arm_files(
                deck_root=args.deck_root,
                core_manifest=args.core_manifest,
                ice_cover_path=args.ice_cover_path,
                exact_ref_dir=args.exact_ref_dir,
                output_dir=args.output_dir,
            )
        print(
            f"[round343] completed 4-arm report -> "
            f"{args.output_dir / 'orca2_round343_speed_benchmark_report.json'}"
        )
        for arm_id, arm in report["arms"].items():
            t = arm["timing"]
            print(
                f"  {arm_id}: compile_warmup={t['compile_warmup_total_s']:.2f}s "
                f"median={t['steady_median_s']:.4f}s/step "
                f"(odd={t['steady_odd_kt_median_s']:.4f}s, even={t['steady_even_kt_median_s']:.4f}s) "
                f"1yr_steady={t['projected_year_steady_min']:.2f}min"
            )
    else:
        arm_report = run_single_benchmark_arm(
            args.arm,
            deck_root=args.deck_root,
            core_manifest=args.core_manifest,
            ice_cover_path=args.ice_cover_path,
            exact_ref_dir=args.exact_ref_dir,
            output_dir=args.output_dir,
            variant=args.variant,
        )
        t = arm_report["timing"]
        print(
            f"[round343] {args.arm}: compile_warmup={t['compile_warmup_total_s']:.2f}s "
            f"median={t['steady_median_s']:.4f}s/step "
            f"(odd={t['steady_odd_kt_median_s']:.4f}s, even={t['steady_even_kt_median_s']:.4f}s) "
            f"1yr_steady={t['projected_year_steady_min']:.2f}min"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
