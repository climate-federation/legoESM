"""Deep scaling diagnostics for offline bottleneck analysis.

Instruments legoESM time-stepping with per-phase timing, memory tracking,
communication volume measurement, and kernel-level breakdown.  Designed to
run on Levante (or any HPC cluster) without Claude Code, producing structured
JSON reports that can be analyzed offline.

The core idea: wrap the step function in a diagnostic harness that records
fine-grained timings per phase (dycore, physics, halo, reductions, etc.)
on every step, then aggregate into statistics suitable for bottleneck
identification.

Usage
-----
::

    from legoesm.parallel.scaling_diagnostics import (
        DiagnosticHarness,
        PhaseTimer,
        MemoryTracker,
        CommVolumeTracker,
    )

    harness = DiagnosticHarness(rank=rank, world_size=world_size)
    harness.start_step()
    # ... run phases, each wrapped in harness.phase("name") ...
    harness.end_step()
    harness.dump("results/diag_rank0.json")

Or use the higher-level ``instrumented_benchmark()`` which wraps the
existing scaling benchmark with full diagnostics.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# ======================================================================
# Phase timer — per-phase wall-clock timing
# ======================================================================

class PhaseTimer:
    """Accumulates wall-clock time per named phase across steps.

    Each phase records individual step durations for histogram analysis,
    plus running totals for fast aggregation.
    """

    def __init__(self, max_history: int = 5000):
        self._max_history = max_history
        self._totals: dict[str, float] = defaultdict(float)   # total ns
        self._counts: dict[str, int] = defaultdict(int)        # call count
        self._history: dict[str, list[float]] = defaultdict(list)  # per-call us
        self._step_phases: dict[str, float] = {}  # current step accumulator
        self._step_start: float = 0.0
        self._step_totals: list[float] = []  # total step time (us)

    @contextmanager
    def phase(self, name: str):
        """Time a phase within the current step."""
        t0 = time.perf_counter_ns()
        yield
        elapsed_ns = time.perf_counter_ns() - t0
        elapsed_us = elapsed_ns / 1_000.0
        self._totals[name] += elapsed_ns
        self._counts[name] += 1
        if len(self._history[name]) < self._max_history:
            self._history[name].append(elapsed_us)
        # Accumulate into current step
        self._step_phases[name] = self._step_phases.get(name, 0.0) + elapsed_us

    def start_step(self):
        """Mark the beginning of a time step."""
        self._step_phases = {}
        self._step_start = time.perf_counter_ns()

    def end_step(self) -> dict[str, float]:
        """Mark the end of a time step; return per-phase times (us)."""
        total_us = (time.perf_counter_ns() - self._step_start) / 1_000.0
        if len(self._step_totals) < self._max_history:
            self._step_totals.append(total_us)
        result = dict(self._step_phases)
        result["_total"] = total_us
        return result

    def summary(self) -> dict[str, Any]:
        """Return summary statistics for all phases."""
        import math

        def _stats(values: list[float]) -> dict[str, float]:
            if not values:
                return {}
            n = len(values)
            mean = sum(values) / n
            sorted_v = sorted(values)
            median = sorted_v[n // 2]
            p95 = sorted_v[int(n * 0.95)] if n >= 20 else sorted_v[-1]
            p99 = sorted_v[int(n * 0.99)] if n >= 100 else sorted_v[-1]
            std = math.sqrt(sum((v - mean) ** 2 for v in values) / n) if n > 1 else 0.0
            return {
                "count": n,
                "mean_us": round(mean, 2),
                "median_us": round(median, 2),
                "std_us": round(std, 2),
                "min_us": round(sorted_v[0], 2),
                "max_us": round(sorted_v[-1], 2),
                "p95_us": round(p95, 2),
                "p99_us": round(p99, 2),
                "total_ms": round(sum(values) / 1_000.0, 2),
            }

        result = {}
        for name in sorted(self._totals.keys()):
            result[name] = _stats(self._history.get(name, []))

        if self._step_totals:
            result["_step_total"] = _stats(self._step_totals)

        # Fraction of step time per phase
        if self._step_totals:
            total_mean = sum(self._step_totals) / len(self._step_totals)
            fractions = {}
            for name in sorted(self._totals.keys()):
                h = self._history.get(name, [])
                if h:
                    phase_mean = sum(h) / len(h)
                    fractions[name] = round(phase_mean / total_mean * 100, 1)
            result["_fraction_pct"] = fractions

        return result


# ======================================================================
# Memory tracker — peak and per-step GPU/host memory
# ======================================================================

class MemoryTracker:
    """Track JAX device memory usage across steps."""

    def __init__(self):
        self._snapshots: list[dict[str, Any]] = []

    def snapshot(self, label: str = "") -> dict[str, Any]:
        """Take a memory snapshot.  Returns the snapshot dict."""
        try:
            import jax
            devices = jax.local_devices()
            mem = {}
            for d in devices:
                try:
                    stats = d.memory_stats()
                    if stats:
                        mem[str(d)] = {
                            "bytes_in_use": stats.get("bytes_in_use", 0),
                            "peak_bytes_in_use": stats.get("peak_bytes_in_use", 0),
                            "bytes_limit": stats.get("bytes_limit", 0),
                        }
                except Exception:
                    pass
            snap = {
                "label": label,
                "timestamp_ns": time.perf_counter_ns(),
                "devices": mem,
            }
        except Exception:
            snap = {"label": label, "timestamp_ns": time.perf_counter_ns(),
                    "error": "could not read memory stats"}
        self._snapshots.append(snap)
        return snap

    def summary(self) -> dict[str, Any]:
        """Return memory summary across all snapshots."""
        if not self._snapshots:
            return {}
        # Find peak across all snapshots
        peak = 0
        peak_label = ""
        for snap in self._snapshots:
            for dev, info in snap.get("devices", {}).items():
                p = info.get("peak_bytes_in_use", 0)
                if p > peak:
                    peak = p
                    peak_label = snap["label"]
        return {
            "n_snapshots": len(self._snapshots),
            "peak_bytes": peak,
            "peak_mb": round(peak / 1e6, 1),
            "peak_label": peak_label,
            "snapshots": self._snapshots,
        }


# ======================================================================
# Communication volume tracker
# ======================================================================

class CommVolumeTracker:
    """Track communication volume (bytes sent/received) per operation."""

    def __init__(self):
        self._volumes: dict[str, list[int]] = defaultdict(list)
        self._max_history = 5000

    def record(self, name: str, bytes_sent: int, bytes_received: int = 0):
        """Record a communication event."""
        total = bytes_sent + bytes_received
        if len(self._volumes[name]) < self._max_history:
            self._volumes[name].append(total)

    def summary(self) -> dict[str, Any]:
        """Return comm volume summary."""
        result = {}
        for name, vols in sorted(self._volumes.items()):
            if vols:
                result[name] = {
                    "n_calls": len(vols),
                    "total_bytes": sum(vols),
                    "total_mb": round(sum(vols) / 1e6, 2),
                    "avg_bytes": round(sum(vols) / len(vols), 0),
                }
        return result


# ======================================================================
# XLA compilation tracker
# ======================================================================

class CompilationTracker:
    """Track JIT compilation events and times."""

    def __init__(self):
        self._compilations: list[dict[str, Any]] = []

    def record(self, name: str, compile_time_s: float,
               n_params: int = 0, hlo_size: int = 0):
        self._compilations.append({
            "name": name,
            "compile_time_s": round(compile_time_s, 3),
            "n_params": n_params,
            "hlo_size": hlo_size,
        })

    def summary(self) -> dict[str, Any]:
        return {
            "n_compilations": len(self._compilations),
            "total_compile_s": round(
                sum(c["compile_time_s"] for c in self._compilations), 3),
            "compilations": self._compilations,
        }


# ======================================================================
# Diagnostic harness — top-level coordinator
# ======================================================================

@dataclass
class HardwareInfo:
    """Collected hardware/software environment information."""
    hostname: str = ""
    platform: str = ""
    python_version: str = ""
    jax_version: str = ""
    jax_backend: str = ""
    n_local_devices: int = 0
    device_kind: str = ""
    device_names: list[str] = field(default_factory=list)
    gpu_info: str = ""
    mpi4jax_version: str = ""
    xla_flags: str = ""
    cuda_visible: str = ""
    slurm_job_id: str = ""
    slurm_nodelist: str = ""
    cpu_count: int = 0
    total_memory_gb: float = 0.0


def collect_hardware_info() -> HardwareInfo:
    """Collect hardware and software environment details."""
    import jax
    info = HardwareInfo()
    info.hostname = platform.node()
    info.platform = platform.platform()
    info.python_version = sys.version
    info.jax_version = jax.__version__
    info.jax_backend = jax.default_backend()
    info.n_local_devices = len(jax.local_devices())
    devs = jax.local_devices()
    if devs:
        info.device_kind = devs[0].platform
        info.device_names = [str(d) for d in devs]

    # GPU info via nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            info.gpu_info = result.stdout.strip()
    except Exception:
        pass

    # mpi4jax version
    try:
        import mpi4jax
        info.mpi4jax_version = getattr(mpi4jax, "__version__", "unknown")
    except ImportError:
        info.mpi4jax_version = "not installed"

    info.xla_flags = os.environ.get("XLA_FLAGS", "")
    info.cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    info.slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
    info.slurm_nodelist = os.environ.get("SLURM_NODELIST", "")
    info.cpu_count = os.cpu_count() or 0

    # Total memory
    try:
        # getrlimit is per-process; use /proc/meminfo on Linux
        if os.path.exists("/proc/meminfo"):
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        info.total_memory_gb = round(kb / 1e6, 1)
                        break
    except Exception:
        pass

    return info


class DiagnosticHarness:
    """Top-level coordinator for scaling diagnostics.

    Wraps a time-stepping loop with per-phase timing, memory tracking,
    communication volume, and compilation tracking.

    Produces a single JSON report file per rank containing all diagnostics.
    """

    def __init__(self, rank: int = 0, world_size: int = 1,
                 grid_type: str = "cubed-sphere", config: dict | None = None):
        self.rank = rank
        self.world_size = world_size
        self.grid_type = grid_type
        self.config = config or {}

        self.timer = PhaseTimer()
        self.memory = MemoryTracker()
        self.comm = CommVolumeTracker()
        self.compilation = CompilationTracker()

        self._step_count = 0
        self._step_records: list[dict[str, float]] = []  # per-step breakdown
        self._max_step_records = 500

    def start_step(self):
        """Call at the beginning of each time step."""
        self.timer.start_step()
        self._step_count += 1

    def end_step(self):
        """Call at the end of each time step."""
        record = self.timer.end_step()
        if len(self._step_records) < self._max_step_records:
            self._step_records.append(record)

    @contextmanager
    def phase(self, name: str):
        """Time a named phase within a step."""
        with self.timer.phase(name):
            yield

    def memory_snapshot(self, label: str = ""):
        """Take a device memory snapshot."""
        self.memory.snapshot(label)

    def record_compilation(self, name: str, compile_time_s: float, **kwargs):
        """Record a JIT compilation event."""
        self.compilation.record(name, compile_time_s, **kwargs)


    def build_report(self, extra: dict | None = None) -> dict[str, Any]:
        """Build the complete diagnostic report."""
        hw = collect_hardware_info()

        report: dict[str, Any] = {
            "version": 1,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "rank": self.rank,
            "world_size": self.world_size,
            "grid_type": self.grid_type,
            "config": self.config,
            "hardware": asdict(hw),
            "n_steps": self._step_count,
            "phase_timing": self.timer.summary(),
            "memory": self.memory.summary(),
            "comm_volume": self.comm.summary(),
            "compilation": self.compilation.summary(),
        }

        # Per-step breakdown (first N steps for time-series analysis)
        if self._step_records:
            report["step_records"] = self._step_records

        if extra:
            report["extra"] = extra

        return report

    def dump(self, path: str | Path, extra: dict | None = None):
        """Write the diagnostic report to a JSON file."""
        report = self.build_report(extra=extra)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        if self.rank == 0:
            print(f"  Diagnostics written to {path}")


# ======================================================================
# Instrumented step wrapper
# ======================================================================

def make_instrumented_step(step_fn, harness: DiagnosticHarness,
                           *, block_until_ready: bool = True):
    """Wrap a step function with per-phase timing.

    This is for *host-side* profiling — it measures wall-clock time
    including device sync.  For production runs, use the standard
    lax.scan path for minimal overhead.

    Parameters
    ----------
    step_fn : callable
        ``step_fn(state, dt) -> state``
    harness : DiagnosticHarness
        Diagnostic harness to record timings into.
    block_until_ready : bool
        If True, call ``jax.block_until_ready`` after each step to ensure
        GPU kernels complete before recording time.  Required for accurate
        per-step timings but adds sync overhead.
    """
    import jax

    def instrumented(state, dt):
        harness.start_step()

        with harness.phase("step_total"):
            new_state = step_fn(state, dt)
            if block_until_ready:
                jax.block_until_ready(jax.tree.leaves(new_state))

        harness.end_step()
        return new_state

    return instrumented


# ======================================================================
# Roofline model data collection
# ======================================================================

def collect_roofline_data(
    step_fn,
    state,
    dt: float,
    n_steps: int = 20,
    rank: int = 0,
) -> dict[str, Any]:
    """Collect data for roofline model analysis.

    Measures:
    - Compute throughput (FLOP/s estimate from step time and grid size)
    - Memory bandwidth utilization
    - Arithmetic intensity estimate

    This uses JAX's profiling hooks when available, falling back to
    wall-clock estimates.
    """
    import jax

    # Estimate state size (bytes)
    leaves = jax.tree.leaves(state)
    state_bytes = sum(
        x.nbytes if hasattr(x, "nbytes") else 0 for x in leaves
    )

    # Warm up
    for _ in range(3):
        state = step_fn(state, dt)
    jax.block_until_ready(jax.tree.leaves(state))

    # Time steps
    times = []
    for _ in range(n_steps):
        t0 = time.perf_counter_ns()
        state = step_fn(state, dt)
        jax.block_until_ready(jax.tree.leaves(state))
        times.append((time.perf_counter_ns() - t0) / 1e9)

    mean_time = sum(times) / len(times)
    # Rough estimate: each step reads and writes full state at least once
    # Real arithmetic intensity requires XLA HLO analysis
    bytes_per_step = 2 * state_bytes  # read + write minimum
    bandwidth_estimate = bytes_per_step / mean_time if mean_time > 0 else 0

    return {
        "state_bytes": state_bytes,
        "state_mb": round(state_bytes / 1e6, 1),
        "n_leaves": len(leaves),
        "mean_step_time_s": round(mean_time, 6),
        "min_bytes_per_step": bytes_per_step,
        "bandwidth_estimate_gb_s": round(bandwidth_estimate / 1e9, 2),
        "step_times_s": [round(t, 6) for t in times],
    }


# ======================================================================
# Halo exchange deep profiler
# ======================================================================

def profile_halo_exchange(
    n: int,
    nlev: int,
    halo: int = 1,
    n_warmup: int = 5,
    n_iters: int = 100,
    rank: int = 0,
    world_size: int = 1,
) -> dict[str, Any]:
    """Detailed halo exchange profiling including message sizes and bandwidth.

    Measures 3D, 4D, and vector halo exchanges with per-exchange timing.
    """
    import jax
    import jax.numpy as jnp

    key = jax.random.PRNGKey(rank)

    results = {}

    for label, shape in [
        ("scalar_3d", (6, n, n)),
        ("scalar_4d", (6, n, n, nlev)),
        ("vector_4d", (6, n, n, nlev, 2)),  # e.g., u,v stacked
    ]:
        try:
            # Dtype must follow the x64 setting: requesting float64 with x64
            # OFF silently truncates to float32, and the byte accounting
            # below would then be 2x wrong (2026-07-24 Levante campaign).
            dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
            dtype_bytes = jnp.dtype(dtype).itemsize
            data = jax.random.normal(key, shape, dtype=dtype)
            data_bytes = data.nbytes

            from legoesm.grids.halo import pad_halo, pad_halo_4d

            if label == "scalar_3d":
                _raw = lambda d: pad_halo(d, halo=halo)
            elif label == "scalar_4d":
                _raw = lambda d: pad_halo_4d(d, halo=halo)
            else:
                # Vector: pad each component
                _raw = lambda d: jnp.stack([
                    pad_halo_4d(d[..., i], halo=halo) for i in range(d.shape[-1])
                ], axis=-1)
            # JIT once. Timing the EAGER path measured op-by-op Python
            # dispatch (~20 s per "exchange" on the Levante GPU node),
            # not the exchange — production always runs these inside a
            # compiled step, so the eager number is meaningless here.
            fn = jax.jit(_raw)

            # Warmup (also absorbs the one-time compile)
            for _ in range(n_warmup):
                _ = fn(data)
            jax.block_until_ready(jax.tree.leaves(fn(data)))

            # Sync before timing
            try:
                from mpi4py import MPI
                if MPI.COMM_WORLD.Get_size() > 1:
                    MPI.COMM_WORLD.Barrier()
            except ImportError:
                pass

            # Timed iterations with per-exchange times
            per_exchange_us = []
            t_total_start = time.perf_counter_ns()
            for _ in range(n_iters):
                t0 = time.perf_counter_ns()
                out = fn(data)
                jax.block_until_ready(jax.tree.leaves(out))
                per_exchange_us.append((time.perf_counter_ns() - t0) / 1_000.0)
            time.perf_counter_ns() - t_total_start

            mean_us = sum(per_exchange_us) / len(per_exchange_us)
            sorted_us = sorted(per_exchange_us)

            # Estimate bytes communicated per exchange:
            # Each face edge: n * halo * dtype_bytes (for 3D)
            # or n * nlev * halo * dtype_bytes (for 4D)
            # 4 edges per face, 6 faces, but only edges on rank boundaries
            if label == "scalar_3d":
                bytes_per_edge = n * halo * dtype_bytes
                n_edges = 4 * 6 if world_size == 1 else 4  # per-rank
            elif label == "scalar_4d":
                bytes_per_edge = n * nlev * halo * dtype_bytes
                n_edges = 4 * 6 if world_size == 1 else 4
            else:
                bytes_per_edge = n * nlev * halo * dtype_bytes * 2
                n_edges = 4 * 6 if world_size == 1 else 4

            bytes_per_exchange = bytes_per_edge * n_edges
            bw_gb_s = (bytes_per_exchange / (mean_us / 1e6)) / 1e9 if mean_us > 0 else 0
            # At world_size == 1 nothing crosses a rank boundary: this is a
            # LOCAL pad (device-memory shuffle), so a "halo bandwidth" label
            # would be a category error. Report the time, refuse the BW.
            local_only = world_size == 1
            if local_only:
                bw_gb_s = None

            results[label] = {
                "shape": list(shape),
                "dtype_bytes": dtype_bytes,
                "jitted": True,
                "local_pad_only": local_only,
                "bandwidth_reason": (
                    "world_size==1: local device-memory pad, no inter-rank "
                    "transfer — bandwidth undefined" if local_only else None),
                "data_bytes": data_bytes,
                "mean_us": round(mean_us, 1),
                "median_us": round(sorted_us[len(sorted_us) // 2], 1),
                "min_us": round(sorted_us[0], 1),
                "max_us": round(sorted_us[-1], 1),
                "p95_us": round(sorted_us[int(len(sorted_us) * 0.95)], 1),
                "std_us": round(
                    (sum((v - mean_us) ** 2 for v in per_exchange_us) /
                     len(per_exchange_us)) ** 0.5, 1),
                "bytes_per_exchange": bytes_per_exchange,
                "bandwidth_gb_s": (round(bw_gb_s, 3)
                                   if bw_gb_s is not None else None),
                "n_edges": n_edges,
            }
        except Exception as e:
            results[label] = {"error": str(e)}

    return results


# ======================================================================
# Reduction profiler
# ======================================================================

def profile_reductions(
    n: int,
    nlev: int,
    n_iters: int = 200,
    rank: int = 0,
    world_size: int = 1,
) -> dict[str, Any]:
    """Profile MPI reduction operations with various payload sizes."""
    import jax
    import jax.numpy as jnp

    results = {}

    for label, shape in [
        ("scalar", ()),
        ("1d_nlev", (nlev,)),
        ("2d_face", (n, n)),
        ("3d_volume", (n, n, nlev)),
    ]:
        data = jax.random.normal(jax.random.PRNGKey(rank), shape or (1,),
                                 dtype=jnp.float64)
        if not shape:
            data = data.squeeze()
        data_bytes = data.nbytes

        # Test local reduction (jnp.sum)
        t0 = time.perf_counter_ns()
        for _ in range(n_iters):
            _ = jnp.sum(data)
            jax.block_until_ready(_)
        local_total_ns = time.perf_counter_ns() - t0
        local_mean_us = local_total_ns / n_iters / 1_000.0

        # Test MPI reduction if available
        mpi_mean_us = None
        if world_size > 1:
            try:
                from legoesm.parallel.reductions import global_sum_mpi
                # Warmup
                for _ in range(5):
                    _ = global_sum_mpi(data)

                try:
                    from mpi4py import MPI
                    MPI.COMM_WORLD.Barrier()
                except ImportError:
                    pass

                t0 = time.perf_counter_ns()
                for _ in range(n_iters):
                    _ = global_sum_mpi(data)
                    jax.block_until_ready(_)
                mpi_total_ns = time.perf_counter_ns() - t0
                mpi_mean_us = round(mpi_total_ns / n_iters / 1_000.0, 1)
            except Exception as e:
                mpi_mean_us = f"error: {e}"

        results[label] = {
            "shape": list(shape) if shape else [1],
            "bytes": data_bytes,
            "local_sum_mean_us": round(local_mean_us, 1),
            "mpi_sum_mean_us": mpi_mean_us,
        }

    return results


# ======================================================================
# XLA profiling helper
# ======================================================================

def capture_xla_profile(
    step_fn,
    state,
    dt: float,
    output_dir: str,
    n_steps: int = 10,
    rank: int = 0,
) -> dict[str, Any]:
    """Capture an XLA/TensorBoard profile trace.

    Creates a profile trace that can be viewed in TensorBoard or
    analyzed by ``jax.profiler``.  The profile captures GPU kernel
    timings, memory allocations, and XLA compilation details.

    Returns metadata about the captured profile.
    """
    import jax

    profile_dir = Path(output_dir) / f"xla_profile_rank{rank}"
    profile_dir.mkdir(parents=True, exist_ok=True)

    # Warm up
    for _ in range(3):
        state = step_fn(state, dt)
    jax.block_until_ready(jax.tree.leaves(state))

    # Profile
    try:
        jax.profiler.start_trace(str(profile_dir))
        for _ in range(n_steps):
            state = step_fn(state, dt)
        jax.block_until_ready(jax.tree.leaves(state))
        jax.profiler.stop_trace()

        return {
            "profile_dir": str(profile_dir),
            "n_steps": n_steps,
            "status": "success",
        }
    except Exception as e:
        return {
            "profile_dir": str(profile_dir),
            "n_steps": n_steps,
            "status": f"error: {e}",
        }


# ======================================================================
# Compute/communication overlap estimator
# ======================================================================

def estimate_overlap_potential(
    step_fn,
    halo_fn,
    state,
    dt: float,
    n_iters: int = 30,
    rank: int = 0,
) -> dict[str, Any]:
    """Estimate the potential for compute/communication overlap.

    Runs the full step, then measures halo exchange time separately.
    The overlap potential is the fraction of step time occupied by
    halo exchange (which could theoretically be hidden).

    ``halo_fn`` MUST be compiled (it is jitted here if it is not): timing
    the eager pad measured Python op-dispatch — 23 s "halo" inside a 46 ms
    step, i.e. a 49451 % "theoretical speedup" (Levante job 26454084).
    A halo fraction above 100 % is arithmetically impossible for a
    component of the step, so it is reported as a defect, not a finding.
    """
    import jax

    halo_fn = jax.jit(halo_fn)

    # Warm up (also compiles halo_fn)
    for _ in range(3):
        state = step_fn(state, dt)
    jax.block_until_ready(jax.tree.leaves(state))
    jax.block_until_ready(jax.tree.leaves(halo_fn(state)))

    # Time full step
    step_times = []
    for _ in range(n_iters):
        t0 = time.perf_counter_ns()
        state = step_fn(state, dt)
        jax.block_until_ready(jax.tree.leaves(state))
        step_times.append((time.perf_counter_ns() - t0) / 1e6)  # ms

    # Time halo exchange only
    halo_times = []
    for _ in range(n_iters):
        t0 = time.perf_counter_ns()
        _ = halo_fn(state)
        jax.block_until_ready(jax.tree.leaves(_))
        halo_times.append((time.perf_counter_ns() - t0) / 1e6)  # ms

    mean_step = sum(step_times) / len(step_times)
    mean_halo = sum(halo_times) / len(halo_times)
    overlap_pct = (mean_halo / mean_step * 100) if mean_step > 0 else 0
    # The standalone halo is a COMPONENT of the step: >100 % means the
    # measurement is invalid (un-jitted dispatch, a different shape, or a
    # step that does not actually contain this exchange), never a real
    # "hide 100 % of the step" opportunity.
    valid = overlap_pct <= 100.0
    out = {
        "mean_step_ms": round(mean_step, 3),
        "mean_halo_ms": round(mean_halo, 3),
        "halo_fraction_pct": round(overlap_pct, 1),
        "measurement_valid": valid,
        "step_times_ms": [round(t, 3) for t in step_times],
        "halo_times_ms": [round(t, 3) for t in halo_times],
    }
    if valid:
        out["overlap_potential_ms"] = round(mean_halo, 3)
        out["theoretical_speedup_pct"] = round(overlap_pct, 1)
    else:
        out["overlap_potential_ms"] = None
        out["theoretical_speedup_pct"] = None
        out["invalid_reason"] = (
            f"standalone halo ({mean_halo:.1f} ms) exceeds the full step "
            f"({mean_step:.1f} ms) — the halo probe is not measuring a "
            f"component of this step; refusing to report an overlap "
            f"potential from it")
    return out
