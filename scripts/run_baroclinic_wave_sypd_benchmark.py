#!/usr/bin/env python
"""Sweep resolutions, GPU counts, and CPU counts. Collect SYPD benchmarks.

Produces a two-panel SYPD scaling figure (like CliMA/Yatunin et al. 2026,
Figure 13) with GPU scaling on top and CPU scaling on bottom, plus
individual panel PNGs and a CSV/JSON of all results.

GPU scaling:  controlled via CUDA_VISIBLE_DEVICES so JAX sees N GPUs.
CPU scaling:  controlled via CUDA_VISIBLE_DEVICES="" (force CPU backend)
              plus XLA_FLAGS to set the intra-op thread count, and
              OMP/MKL thread variables for NumPy/BLAS parallelism.

Usage:
    cd /path/to/legoESM
    source .venv/bin/activate

    # GPU-only sweep (single GPU, multiple resolutions)
    python scripts/run_sypd_benchmark.py --resolutions 16 32 48 96 --days 10

    # Multi-GPU scaling
    python scripts/run_sypd_benchmark.py -n 16 32 48 --gpus 1 2 4 --days 10

    # CPU-only scaling
    python scripts/run_sypd_benchmark.py -n 16 32 48 --cpus 16 32 64 128 --days 5

    # Full benchmark: GPU + CPU (produces the two-panel figure)
    python scripts/run_sypd_benchmark.py -n 16 32 48 96 \\
        --gpus 1 2 4 8 --cpus 16 32 64 128 256 --days 10

    # Dry run
    python scripts/run_sypd_benchmark.py -n 16 32 --gpus 1 2 --cpus 16 32 --dry-run
"""

import argparse
import csv
import json
import multiprocessing
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def compute_sypd(n_days: int, wall_clock_seconds: float) -> float:
    """Compute Simulated Years Per Day."""
    simulated_years = n_days / 365.25
    wall_clock_days = wall_clock_seconds / 86400.0
    if wall_clock_days == 0:
        return float("inf")
    return simulated_years / wall_clock_days


def detect_gpus() -> list[int]:
    """Return list of available CUDA GPU indices."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return [int(x.strip()) for x in result.stdout.strip().splitlines()]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    try:
        import jax

        gpus = [d for d in jax.devices() if d.platform == "gpu"]
        return list(range(len(gpus)))
    except Exception:
        pass
    return []


def detect_cpu_count() -> int:
    """Return total number of CPU cores available."""
    try:
        # On SLURM, respect allocated cores
        slurm_cpus = os.environ.get("SLURM_CPUS_ON_NODE")
        if slurm_cpus:
            return int(slurm_cpus)
    except (ValueError, TypeError):
        pass
    return multiprocessing.cpu_count()


def make_cuda_visible_str(n_gpus: int, available_gpus: list[int]) -> str:
    """Build CUDA_VISIBLE_DEVICES string for the first n_gpus devices."""
    selected = available_gpus[:n_gpus]
    return ",".join(str(g) for g in selected)


def create_cpu_wrapper(script_path: str) -> str:
    """Create a temporary wrapper script that suppresses the CUDA plugin.

    On GPU nodes, JAX's xla_cuda12 plugin crashes during initialization
    when CUDA_VISIBLE_DEVICES is set to hide GPUs. This wrapper
    monkey-patches the plugin's initialize() function to a no-op before
    running the actual script via runpy, preserving argv and __file__.

    Returns the path to the temporary wrapper script.
    """
    # Resolve to absolute path so the wrapper works regardless of cwd
    abs_script = os.path.abspath(script_path)
    script_dir = os.path.dirname(abs_script)
    project_dir = os.getcwd()

    wrapper_code = f"""\
import sys
import os

# Ensure CPU-only environment
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["JAX_PLATFORMS"] = "cpu"

# Set working directory to the project root (where the original command ran)
os.chdir({repr(project_dir)})

# Add the script's own directory to sys.path so relative imports work
# (e.g. "from _deprecated_wrapper import ...")
script_dir = {repr(script_dir)}
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Monkey-patch the CUDA plugin so it doesn't try to init
try:
    import jax_plugins.xla_cuda12 as _cuda_plugin
    _cuda_plugin.initialize = lambda: None
except (ImportError, ModuleNotFoundError):
    pass

# Run the actual script via runpy so argparse, __file__ etc. work
import runpy
sys.argv[0] = {repr(abs_script)}
runpy.run_path({repr(abs_script)}, run_name="__main__")
"""
    fd, path = tempfile.mkstemp(suffix="_cpu_wrapper.py", prefix="legoesm_")
    with os.fdopen(fd, "w") as f:
        f.write(wrapper_code)
    return path


# ─────────────────────────────────────────────────────────────────────
# Core benchmark runner
# ─────────────────────────────────────────────────────────────────────


def run_single_benchmark(
    resolution: int,
    days: int,
    device_type: str = "gpu",  # "gpu" or "cpu"
    n_devices: int = 1,  # number of GPUs or CPU threads
    available_gpus: list[int] | None = None,
    levels: int = 26,
    dt: float | None = None,
    perturbed: bool = True,
    script_path: str = "scripts/run_baroclinic_wave.py",
    extra_args: list[str] | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Run a single baroclinic wave benchmark and parse the output.

    For GPU runs: sets CUDA_VISIBLE_DEVICES to expose n_devices GPUs.
    For CPU runs: hides all GPUs, sets XLA intra-op parallelism threads
                  and OMP/MKL thread counts to n_devices.
    """
    cmd = [
        sys.executable,
        script_path,
        "--resolution",
        str(resolution),
        "--levels",
        str(levels),
        "--days",
        str(days),
    ]
    if dt is not None:
        cmd += ["--dt", str(dt)]
    if not perturbed:
        cmd.append("--no-perturbation")

    # Give each run its own output directory so results don't overwrite
    pert_str = "perturbed" if perturbed else "steady_state"
    run_subdir = (
        f"baroclinic_wave_{pert_str}_C{resolution}_L{levels}_{device_type}{n_devices}"
    )
    if output_dir is not None:
        run_output = str(output_dir / run_subdir)
    else:
        run_output = f"results/{run_subdir}"
    cmd += ["--output", run_output]

    if extra_args:
        cmd.extend(extra_args)

    # Build environment
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    if device_type == "gpu":
        wrapper_path = None
        if available_gpus and n_devices > 0:
            cuda_str = make_cuda_visible_str(n_devices, available_gpus)
            env["CUDA_VISIBLE_DEVICES"] = cuda_str
            device_label = f"{n_devices} GPU(s) [{cuda_str}]"
        else:
            env["CUDA_VISIBLE_DEVICES"] = ""
            env["JAX_PLATFORMS"] = "cpu"
            device_label = "CPU fallback (no GPUs)"
    else:
        # CPU mode: Control logical device partitioning and thread counts
        env["CUDA_VISIBLE_DEVICES"] = "-1"
        env["JAX_PLATFORMS"] = "cpu"

        # IMPORTANT: This forces JAX to partition the CPU into N devices.
        # This allows pmap/sharding to actually distribute work.
        xla_flags = env.get("XLA_FLAGS", "")
        # Add the device count flag while preserving existing flags
        xla_flags += f" --xla_force_host_platform_device_count={n_devices}"
        env["XLA_FLAGS"] = xla_flags.strip()

        # Thread control for XLA and BLAS
        n_threads = str(n_devices)
        env["XLA_NUM_THREADS"] = n_threads
        env["OMP_NUM_THREADS"] = "1"  # Often better to let XLA handle parallelism
        env["MKL_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"

        device_label = f"{n_devices} Logical CPU Device(s)"
        wrapper_path = create_cpu_wrapper(script_path)
        cmd[1] = wrapper_path

    print(f"\n{'='*70}")
    print(
        f"  Running C{resolution}, L{levels}, {days} days, "
        f"{'perturbed' if perturbed else 'steady-state'}"
    )
    print(f"  Devices: {device_label}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*70}")

    t_start = time.time()
    stdout_lines = []
    stderr_lines = []

    def _cleanup_wrapper():
        if wrapper_path and os.path.exists(wrapper_path):
            try:
                os.unlink(wrapper_path)
            except OSError:
                pass

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )

        for line in proc.stdout:
            print(f"    {line}", end="")
            stdout_lines.append(line)

        proc.wait(timeout=14400)  # 4h timeout for CPU runs
        wall_time = time.time() - t_start

        stderr_text = proc.stderr.read()
        if stderr_text:
            stderr_lines.append(stderr_text)

        stdout = "".join(stdout_lines)
        stderr = "".join(stderr_lines)

        if proc.returncode != 0:
            print(f"  *** FAILED (exit code {proc.returncode}) ***")
            print(f"  stderr: {stderr[:500]}")
            _cleanup_wrapper()
            return {
                "resolution": resolution,
                "levels": levels,
                "days": days,
                "device_type": device_type,
                "n_devices": n_devices,
                "perturbed": perturbed,
                "status": "FAILED",
                "wall_time": wall_time,
                "sypd": None,
                "stdout": stdout,
                "stderr": stderr,
            }

    except subprocess.TimeoutExpired:
        proc.kill()
        print(f"  *** TIMEOUT ***")
        _cleanup_wrapper()
        return {
            "resolution": resolution,
            "levels": levels,
            "days": days,
            "device_type": device_type,
            "n_devices": n_devices,
            "perturbed": perturbed,
            "status": "TIMEOUT",
            "wall_time": 14400,
            "sypd": None,
        }

    # Parse output
    info = {
        "resolution": resolution,
        "levels": levels,
        "days": days,
        "device_type": device_type,
        "n_devices": n_devices,
        "perturbed": perturbed,
        "status": "OK",
        "wall_time": wall_time,
    }

    for line in stdout.splitlines():
        s = line.strip()

        if "Integration complete:" in s:
            parts = s.split()
            for i, p in enumerate(parts):
                if p.endswith("s") and i > 0 and parts[i - 1] == "complete:":
                    try:
                        info["integration_time"] = float(p.rstrip("s"))
                    except ValueError:
                        pass
                if p.startswith("(") and "steps/s" in s:
                    try:
                        info["steps_per_sec"] = float(p.strip("("))
                    except ValueError:
                        pass
        if "Mass drift:" in s:
            try:
                info["mass_drift"] = float(s.split()[-1])
            except (ValueError, IndexError):
                pass
        if "Max wind:" in s:
            try:
                info["max_wind"] = float(s.split()[-2])
            except (ValueError, IndexError):
                pass
        if "Mean T:" in s:
            try:
                info["mean_T"] = float(s.split()[-2])
            except (ValueError, IndexError):
                pass
        if "Grid resolution:" in s:
            try:
                info["grid_km"] = float(s.split("~")[1].split()[0])
            except (ValueError, IndexError):
                pass
        if "BLOWUP" in s:
            info["status"] = "BLOWUP"

    integration_time = info.get("integration_time", wall_time)
    info["sypd"] = compute_sypd(days, integration_time)
    info["sypd_total"] = compute_sypd(days, wall_time)

    print(f"\n  Result: {info['status']}")
    print(f"  Devices: {device_label}")
    print(f"  Wall time: {wall_time:.1f}s " f"(integration: {integration_time:.1f}s)")
    print(f"  SYPD (integration only): {info['sypd']:.4f}")
    print(f"  SYPD (total wallclock):  {info['sypd_total']:.4f}")
    if "steps_per_sec" in info:
        print(f"  Steps/sec: {info['steps_per_sec']:.1f}")

    _cleanup_wrapper()
    return info


# ─────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────

# Resolution colors (coarsest to finest, matching CliMA style)
RESOLUTION_COLORS = [
    "#00BCD4",  # cyan
    "#4CAF50",  # green
    "#F44336",  # red
    "#FF9800",  # orange
    "#9C27B0",  # purple
    "#795548",  # brown
    "#009688",  # teal
    "#E91E63",  # pink
]
MARKER_STYLES = ["o", "s", "D", "^", "v", "P", "*", "X"]


def _group_results(results, device_type):
    """Group results: {resolution: {n_devices: result}}."""
    grouped = {}
    for r in results:
        if r["status"] != "OK" or r["sypd"] is None:
            continue
        if not r.get("perturbed", True):
            continue
        if r.get("device_type") != device_type:
            continue
        res = r["resolution"]
        nd = r["n_devices"]
        if res not in grouped:
            grouped[res] = {}
        grouped[res][nd] = r
    return grouped


def _draw_scaling_panel(
    ax, grouped, resolutions, device_counts, res_color, res_marker, xlabel, title
):
    """Draw a single SYPD-vs-device-count panel on the given axes."""

    for res in sorted(resolutions, reverse=True):
        if res not in grouped:
            continue
        dev_data = grouped[res]
        devs_sorted = sorted(dev_data.keys())
        sypds = [dev_data[nd]["sypd"] for nd in devs_sorted]

        grid_km = dev_data[devs_sorted[0]].get("grid_km", 10000.0 / res)
        label = f"~{grid_km:.0f} km"

        ax.plot(
            devs_sorted,
            sypds,
            marker=res_marker[res],
            color=res_color[res],
            linewidth=2,
            markersize=9,
            markeredgecolor="black",
            markeredgewidth=0.8,
            label=label,
            zorder=5,
        )

    # Ideal linear scaling reference
    if len(device_counts) >= 2:
        dev_arr = np.array(device_counts, dtype=float)
        for res in resolutions:
            if res not in grouped:
                continue
            dev_data = grouped[res]
            base = min(dev_data.keys())
            base_sypd = dev_data[base]["sypd"]
            ideal = base_sypd * (dev_arr / base)
            ax.plot(
                dev_arr,
                ideal,
                "--",
                color=res_color[res],
                alpha=0.25,
                linewidth=1,
                zorder=2,
            )

        ax.plot([], [], "--", color="gray", alpha=0.5, label="Ideal scaling")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("SYPD", fontsize=12)
    ax.set_title(title, fontsize=13)

    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{int(x)}" if x == int(x) else "")
    )
    ax.xaxis.set_major_locator(ticker.FixedLocator(device_counts))

    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax.grid(True, which="both", alpha=0.3)


def plot_all(results, output_dir, gpu_label="", cpu_label=""):
    """Generate all plots.

    If both GPU and CPU results exist, produces the two-panel combined
    figure. Individual panel PNGs are always saved as well.
    """
    ok = [
        r
        for r in results
        if r["status"] == "OK" and r["sypd"] is not None and r.get("perturbed", True)
    ]
    if not ok:
        print("No successful results to plot.")
        return

    resolutions = sorted(set(r["resolution"] for r in ok))

    # Assign colors per resolution
    res_color = {}
    res_marker = {}
    for i, res in enumerate(resolutions):
        res_color[res] = RESOLUTION_COLORS[i % len(RESOLUTION_COLORS)]
        res_marker[res] = MARKER_STYLES[i % len(MARKER_STYLES)]

    gpu_grouped = _group_results(results, "gpu")
    cpu_grouped = _group_results(results, "cpu")

    has_gpu = len(gpu_grouped) > 0
    has_cpu = len(cpu_grouped) > 0

    gpu_devs = (
        sorted(set(nd for rd in gpu_grouped.values() for nd in rd.keys()))
        if has_gpu
        else []
    )
    cpu_devs = (
        sorted(set(nd for rd in cpu_grouped.values() for nd in rd.keys()))
        if has_cpu
        else []
    )

    # ── Individual panels ──
    if has_gpu and len(gpu_devs) > 1:
        fig, ax = plt.subplots(figsize=(10, 7))
        title = "legoESM: SYPD vs. Number of GPUs"
        if gpu_label:
            title += f"\n({gpu_label})"
        _draw_scaling_panel(
            ax,
            gpu_grouped,
            resolutions,
            gpu_devs,
            res_color,
            res_marker,
            "Number of GPUs",
            title,
        )
        plt.tight_layout()
        fig.savefig(output_dir / "sypd_vs_gpus.png", dpi=150)
        plt.close()
        print("  Saved sypd_vs_gpus.png")

    if has_cpu and len(cpu_devs) > 1:
        fig, ax = plt.subplots(figsize=(10, 7))
        title = "legoESM: SYPD vs. Number of CPUs"
        if cpu_label:
            title += f"\n({cpu_label})"
        _draw_scaling_panel(
            ax,
            cpu_grouped,
            resolutions,
            cpu_devs,
            res_color,
            res_marker,
            "Number of CPUs",
            title,
        )
        plt.tight_layout()
        fig.savefig(output_dir / "sypd_vs_cpus.png", dpi=150)
        plt.close()
        print("  Saved sypd_vs_cpus.png")

    # ── Combined two-panel figure (CliMA Figure 13 style) ──
    if has_gpu and has_cpu and len(gpu_devs) > 1 and len(cpu_devs) > 1:
        fig, (ax_gpu, ax_cpu) = plt.subplots(2, 1, figsize=(10, 12), sharex=False)

        gpu_title = "SYPD vs. Number of GPUs"
        if gpu_label:
            gpu_title += f" ({gpu_label})"
        _draw_scaling_panel(
            ax_gpu,
            gpu_grouped,
            resolutions,
            gpu_devs,
            res_color,
            res_marker,
            "Number of GPUs",
            gpu_title,
        )

        cpu_title = "SYPD vs. Number of CPUs"
        if cpu_label:
            cpu_title += f" ({cpu_label})"
        _draw_scaling_panel(
            ax_cpu,
            cpu_grouped,
            resolutions,
            cpu_devs,
            res_color,
            res_marker,
            "Number of CPUs",
            cpu_title,
        )

        fig.suptitle("legoESM: Baroclinic Wave Benchmark", fontsize=15, y=1.01)
        plt.tight_layout()
        fig.savefig(output_dir / "sypd_combined.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved sypd_combined.png")

    # ── SYPD vs Resolution ──
    fig, ax = plt.subplots(figsize=(10, 7))
    all_lines = []

    if has_gpu:
        for ng in gpu_devs:
            rl, sl = [], []
            for res in resolutions:
                if res in gpu_grouped and ng in gpu_grouped[res]:
                    rl.append(res)
                    sl.append(gpu_grouped[res][ng]["sypd"])
            if rl:
                lbl = f"{ng} GPU{'s' if ng > 1 else ''}"
                ax.plot(
                    rl,
                    sl,
                    "o-",
                    linewidth=2,
                    markersize=8,
                    markeredgecolor="black",
                    markeredgewidth=0.7,
                    label=lbl,
                    zorder=5,
                )

    if has_cpu:
        for nc in cpu_devs:
            rl, sl = [], []
            for res in resolutions:
                if res in cpu_grouped and nc in cpu_grouped[res]:
                    rl.append(res)
                    sl.append(cpu_grouped[res][nc]["sypd"])
            if rl:
                lbl = f"{nc} CPU{'s' if nc > 1 else ''}"
                ax.plot(
                    rl,
                    sl,
                    "s--",
                    linewidth=2,
                    markersize=8,
                    markeredgecolor="black",
                    markeredgewidth=0.7,
                    label=lbl,
                    zorder=4,
                )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Cubed-sphere resolution (N)", fontsize=13)
    ax.set_ylabel("SYPD", fontsize=13)
    ax.set_title("legoESM: SYPD vs. Resolution", fontsize=14)
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"C{int(x)}" if x == int(x) else "")
    )
    ax.xaxis.set_major_locator(ticker.FixedLocator(resolutions))
    ax.legend(fontsize=9, loc="best", framealpha=0.9)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "sypd_vs_resolution.png", dpi=150)
    plt.close()
    print("  Saved sypd_vs_resolution.png")

    # ── Breakdown bar chart ──
    combined = {**gpu_grouped, **cpu_grouped}
    if combined:
        configs, si, st = [], [], []
        for res in resolutions:
            if res in combined:
                ng = min(combined[res].keys())
                r = combined[res][ng]
                configs.append(f"C{res}")
                si.append(r["sypd"])
                st.append(r["sypd_total"])

        if configs:
            fig, ax = plt.subplots(figsize=(8, 6))
            x = np.arange(len(configs))
            w = 0.35
            ax.bar(
                x - w / 2, si, w, label="Integration only", color="#2196F3", alpha=0.85
            )
            ax.bar(
                x + w / 2,
                st,
                w,
                label="Total (incl. JIT, I/O)",
                color="#FF9800",
                alpha=0.85,
            )
            ax.set_yscale("log")
            ax.set_xlabel("Resolution", fontsize=12)
            ax.set_ylabel("SYPD", fontsize=12)
            ax.set_title("legoESM: SYPD Breakdown", fontsize=13)
            ax.set_xticks(x)
            ax.set_xticklabels(configs)
            ax.legend(fontsize=11)
            ax.grid(True, axis="y", alpha=0.3)
            plt.tight_layout()
            fig.savefig(output_dir / "sypd_breakdown.png", dpi=150)
            plt.close()
            print("  Saved sypd_breakdown.png")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Run SYPD scaling benchmarks for legoESM baroclinic wave"
    )
    parser.add_argument(
        "--resolutions",
        "-n",
        type=int,
        nargs="+",
        default=[16, 32, 48],
        help="Cubed-sphere resolutions to test (default: 16 32 48)",
    )
    parser.add_argument(
        "--gpus",
        "-g",
        type=int,
        nargs="+",
        default=None,
        help="GPU counts to sweep. E.g. --gpus 1 2 4 8. "
        "Omit to skip GPU benchmarks (unless no --cpus either, "
        "then defaults to single GPU).",
    )
    parser.add_argument(
        "--cpus",
        "-c",
        type=int,
        nargs="+",
        default=None,
        help="CPU thread counts to sweep. E.g. --cpus 16 32 64 128 256. "
        "Forces CPU backend (no GPU). "
        "Omit to skip CPU benchmarks.",
    )
    parser.add_argument(
        "--gpu-label",
        type=str,
        default="",
        help="Label for GPU hardware, shown in plot title. "
        "E.g. 'NVIDIA A100' or 'DKRZ Levante A100'",
    )
    parser.add_argument(
        "--cpu-label",
        type=str,
        default="",
        help="Label for CPU hardware, shown in plot title. "
        "E.g. 'AMD EPYC 7763' or 'DKRZ Levante'",
    )
    parser.add_argument(
        "--levels",
        "-l",
        type=int,
        default=26,
        help="Number of vertical levels (default: 26)",
    )
    parser.add_argument(
        "--days",
        "-d",
        type=int,
        default=10,
        help="Integration time in days (default: 10)",
    )
    parser.add_argument(
        "--include-steady-state",
        action="store_true",
        help="Also run the steady-state (unperturbed) test",
    )
    parser.add_argument(
        "--script",
        type=str,
        default="scripts/run_baroclinic_wave.py",
        help="Path to run_baroclinic_wave.py",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="results/sypd_benchmark",
        help="Output directory for results",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without executing"
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Detect hardware
    available_gpus = detect_gpus()
    n_gpus_available = len(available_gpus)
    n_cpus_available = detect_cpu_count()

    # Determine what to run
    gpu_counts = sorted(args.gpus) if args.gpus else []
    cpu_counts = sorted(args.cpus) if args.cpus else []

    # Default: if neither --gpus nor --cpus given, do single GPU
    if not gpu_counts and not cpu_counts:
        if n_gpus_available >= 1:
            gpu_counts = [1]
        else:
            cpu_counts = [n_cpus_available]

    resolutions = sorted(args.resolutions)

    # Validate
    for ng in gpu_counts:
        if ng > n_gpus_available:
            print(
                f"WARNING: requested {ng} GPUs but only "
                f"{n_gpus_available} available. Will cap at {n_gpus_available}."
            )
    for nc in cpu_counts:
        if nc > n_cpus_available:
            print(
                f"WARNING: requested {nc} CPU threads but only "
                f"{n_cpus_available} cores detected."
            )

    total_runs = len(resolutions) * (len(gpu_counts) + len(cpu_counts))
    if args.include_steady_state:
        total_runs *= 2

    # Build run plan: list of (device_type, n_devices) tuples
    run_plan = []
    for ng in gpu_counts:
        run_plan.append(("gpu", min(ng, n_gpus_available)))
    for nc in cpu_counts:
        run_plan.append(("cpu", nc))

    print("=" * 70)
    print("legoESM SYPD Benchmark Suite")
    print("=" * 70)
    print(f"Resolutions:    {['C' + str(r) for r in resolutions]}")
    if gpu_counts:
        print(
            f"GPU counts:     {gpu_counts}  (available: {n_gpus_available} "
            f"{available_gpus})"
        )
    if cpu_counts:
        print(f"CPU counts:     {cpu_counts}  (available cores: {n_cpus_available})")
    print(f"Levels:         {args.levels}")
    print(f"Days:           {args.days}")
    print(f"Output:         {output_dir}")
    print(f"Total runs:     {total_runs}")
    if args.gpu_label:
        print(f"GPU label:      {args.gpu_label}")
    if args.cpu_label:
        print(f"CPU label:      {args.cpu_label}")

    if args.dry_run:
        print("\n*** DRY RUN ***\n")
        idx = 0
        for res in resolutions:
            for dtype, ndev in run_plan:
                idx += 1
                pert_str = "perturbed"
                run_out = str(
                    output_dir
                    / f"baroclinic_wave_{pert_str}_C{res}_L{args.levels}_{dtype}{ndev}"
                )
                if dtype == "gpu":
                    cuda = make_cuda_visible_str(ndev, available_gpus)
                    env_str = f"CUDA_VISIBLE_DEVICES={cuda}"
                else:
                    env_str = (
                        f"CUDA_VISIBLE_DEVICES=-1 JAX_PLATFORMS=cpu "
                        f"OMP_NUM_THREADS={ndev} "
                        f"MKL_NUM_THREADS={ndev} "
                        f"(via CUDA-suppressing wrapper)"
                    )
                cmd = (
                    f"python {args.script} --resolution {res} "
                    f"--levels {args.levels} --days {args.days} "
                    f"--output {run_out}"
                )
                print(
                    f"  [{idx}/{total_runs}] [{dtype.upper()} x{ndev}] "
                    f"{env_str}\n"
                    f"    {cmd}"
                )
        return

    # ── Run benchmarks ──
    all_results = []
    t_total_start = time.time()
    idx = 0

    # Define a baseline for the CFL condition.
    # C16 typically runs well at 600s.
    BASE_RES = 16
    BASE_DT = 600.0

    for res in resolutions:
        # Calculate a stable dt inversely proportional to resolution
        # Example: C16 -> 600s, C32 -> 300s, C64 -> 150s, C96 -> 100s
        stable_dt = BASE_DT * (BASE_RES / res)

        print(f"\n--- Scaling for C{res}: Using dt = {stable_dt:.1f}s ---")
        for dtype, ndev in run_plan:
            idx += 1
            print(f"\n  [{idx}/{total_runs}]", end="")

            result = run_single_benchmark(
                resolution=res,
                days=args.days,
                device_type=dtype,
                n_devices=ndev,
                available_gpus=available_gpus,
                levels=args.levels,
                dt=stable_dt,  # Passes the calculated stable time step
                perturbed=True,
                script_path=args.script,
                output_dir=output_dir,
            )
            all_results.append(result)

            if args.include_steady_state:
                idx += 1
                print(f"\n  [{idx}/{total_runs}]", end="")
                result_ss = run_single_benchmark(
                    resolution=res,
                    days=args.days,
                    device_type=dtype,
                    n_devices=ndev,
                    available_gpus=available_gpus,
                    levels=args.levels,
                    dt=stable_dt,  # Also apply to steady state
                    perturbed=False,
                    script_path=args.script,
                    output_dir=output_dir,
                )
                all_results.append(result_ss)

    total_time = time.time() - t_total_start

    # ── Save results ──
    csv_path = output_dir / "sypd_results.csv"
    fieldnames = [
        "resolution",
        "levels",
        "days",
        "device_type",
        "n_devices",
        "perturbed",
        "status",
        "wall_time",
        "integration_time",
        "sypd",
        "sypd_total",
        "steps_per_sec",
        "grid_km",
        "max_wind",
        "mean_T",
        "mass_drift",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)
    print(f"\nResults saved to {csv_path}")

    json_results = []
    for r in all_results:
        jr = {k: v for k, v in r.items() if k not in ("stdout", "stderr")}
        json_results.append(jr)
    with open(output_dir / "sypd_results.json", "w") as f:
        json.dump(json_results, f, indent=2, default=str)

    # ── Generate plots ──
    print("\nGenerating plots...")
    plot_all(
        all_results, output_dir, gpu_label=args.gpu_label, cpu_label=args.cpu_label
    )

    # ── Print summary table ──
    print(f"\n{'='*70}")
    print("SYPD Benchmark Summary")
    print(f"{'='*70}")
    print(
        f"{'Res':>6} {'Type':>4} {'#Dev':>5} {'Grid km':>8} {'Status':>8} "
        f"{'Wall(s)':>8} {'Int(s)':>8} {'SYPD':>10} {'steps/s':>8}"
    )
    for r in all_results:
        if not r.get("perturbed", True):
            continue
        dtype = r.get("device_type", "gpu").upper()[:3]
        sypd_val = r.get("sypd")
        sypd_str = f"{sypd_val:>10.4f}" if sypd_val is not None else f"{'N/A':>10}"
        sps_val = r.get("steps_per_sec")
        sps_str = f"{sps_val:>8.1f}" if sps_val is not None else f"{'N/A':>8}"
        int_val = r.get("integration_time")
        int_str = f"{int_val:>8.1f}" if int_val is not None else f"{'N/A':>8}"
        print(
            f"C{r['resolution']:>4} "
            f"{dtype:>4} "
            f"{r.get('n_devices', 1):>5} "
            f"{r.get('grid_km', 0) or 0:>7.0f} "
            f"{r['status']:>8} "
            f"{r.get('wall_time', 0):>8.1f} "
            f"{int_str} "
            f"{sypd_str} "
            f"{sps_str}"
        )

    print(f"\nTotal benchmark time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"All outputs in: {output_dir}/")


if __name__ == "__main__":
    main()
