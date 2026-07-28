#!/usr/bin/env python
"""Offline analysis of scaling diagnostics for Claude Code review.

Reads the JSON reports produced by ``run_scaling_diagnosis.py`` on Levante
and generates a structured bottleneck analysis report.  Designed to be run
locally where Claude Code can review the output.

Usage
-----
Analyze a single diagnostic run::

    python scripts/analyze_scaling_results.py results/scaling_diagnosis/20260416T120000Z/

Compare two runs (e.g., before/after optimization)::

    python scripts/analyze_scaling_results.py \
        --baseline results/scaling_diagnosis/20260415T100000Z/ \
        --current  results/scaling_diagnosis/20260416T120000Z/

Generate a Claude-friendly report (markdown)::

    python scripts/analyze_scaling_results.py results/scaling_diagnosis/20260416T120000Z/ \
        --format markdown --output bottleneck_report.md

Output
------
Produces a structured analysis covering:

1. **Throughput summary**: SYPD, ms/step, Mcells/s
2. **Phase breakdown**: What fraction of time is spent in each phase
3. **Communication analysis**: Halo exchange bandwidth, reduction latency
4. **Memory analysis**: Peak usage, allocation patterns
5. **Scaling efficiency**: Weak/strong scaling from multi-run data
6. **Bottleneck ranking**: Ordered list of optimization opportunities
7. **Roofline position**: Compute vs memory bound assessment
8. **Recommendations**: Specific optimization suggestions with priority

Each section includes raw data and interpretation suitable for
Claude Code to propose concrete optimization patches.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


# ======================================================================
# Data loading
# ======================================================================

def load_run(run_dir: Path) -> dict[str, Any]:
    """Load all JSON files from a diagnostic run directory."""
    data = {"dir": str(run_dir)}

    for p in sorted(run_dir.glob("*.json")):
        key = p.stem
        with open(p) as f:
            data[key] = json.load(f)

    # Check for XLA profile directories
    for d in sorted(run_dir.glob("xla_profile_*")):
        data.setdefault("xla_profiles", []).append(str(d))

    return data


def load_multi_rank(run_dir: Path) -> list[dict]:
    """Load per-rank diagnostic files."""
    ranks = []
    for p in sorted(run_dir.glob("diag_rank*.json")):
        with open(p) as f:
            ranks.append(json.load(f))
    return ranks


# ======================================================================
# Analysis modules
# ======================================================================

def analyze_throughput(data: dict) -> dict[str, Any]:
    """Analyze throughput from scan timing."""
    scan = data.get("summary", {}).get("scan_throughput") or \
           data.get("diag_rank0", {}).get("extra", {}).get("scan_throughput")
    if not scan:
        return {"status": "no scan timing data"}

    config = data.get("summary", {}).get("config") or \
             data.get("diag_rank0", {}).get("config", {})

    total_cells = config.get("total_cells", 0)
    world_size = config.get("cells_per_rank", 0)
    ms_per_step = scan.get("time_per_step_ms", 0)
    mcells_per_s = (total_cells / (ms_per_step / 1000) / 1e6) if ms_per_step > 0 else 0

    return {
        "time_per_step_ms": ms_per_step,
        "sypd": scan.get("sypd", 0),
        "mcells_per_s": round(mcells_per_s, 1),
        "compile_time_s": scan.get("compile_time_s", 0),
        "total_cells": total_cells,
        "assessment": _throughput_assessment(ms_per_step, total_cells, scan.get("sypd", 0)),
    }


def _throughput_assessment(ms_step: float, cells: int, sypd: float) -> str:
    """Qualitative throughput assessment."""
    if cells == 0 or ms_step == 0:
        return "insufficient data"
    us_per_cell = ms_step * 1000 / cells
    if us_per_cell < 0.01:
        return "excellent (<10 ns/cell)"
    elif us_per_cell < 0.1:
        return "good (10-100 ns/cell)"
    elif us_per_cell < 1.0:
        return "moderate (0.1-1 us/cell)"
    else:
        return f"slow ({us_per_cell:.1f} us/cell — likely bottlenecked)"


def analyze_phase_breakdown(data: dict) -> dict[str, Any]:
    """Analyze per-phase timing breakdown."""
    rank0 = data.get("diag_rank0", {})
    phases = rank0.get("phase_timing", {})

    if not phases:
        return {"status": "no phase timing data"}

    fractions = phases.get("_fraction_pct", {})
    step_total = phases.get("_step_total", {})

    # Find dominant phase
    if fractions:
        dominant = max(fractions, key=fractions.get)
        dominant_pct = fractions[dominant]
    else:
        dominant = "unknown"
        dominant_pct = 0

    # Identify bottleneck categories
    comm_phases = {k: v for k, v in fractions.items()
                   if any(x in k.lower() for x in ["halo", "comm", "exchange", "reduce", "allreduce"])}
    compute_phases = {k: v for k, v in fractions.items()
                      if any(x in k.lower() for x in ["dycore", "physics", "step", "compute"])}

    comm_total = sum(comm_phases.values()) if comm_phases else 0
    compute_total = sum(compute_phases.values()) if compute_phases else 0

    return {
        "fractions_pct": fractions,
        "step_total": step_total,
        "dominant_phase": dominant,
        "dominant_pct": dominant_pct,
        "comm_total_pct": round(comm_total, 1),
        "compute_total_pct": round(compute_total, 1),
        "phase_details": {k: v for k, v in phases.items()
                          if not k.startswith("_")},
        "assessment": _phase_assessment(comm_total, compute_total, fractions),
    }


def _phase_assessment(comm_pct: float, compute_pct: float,
                       fractions: dict) -> str:
    if comm_pct > 50:
        return f"COMMUNICATION BOUND: {comm_pct:.0f}% of step in communication. Priority: reduce halo exchange cost or enable overlap."
    elif comm_pct > 30:
        return f"Mixed compute/comm: {comm_pct:.0f}% communication. Overlap optimization could help."
    else:
        return f"COMPUTE BOUND: {compute_pct:.0f}% of step in compute. Focus on kernel optimization."


def analyze_halo_exchange(data: dict) -> dict[str, Any]:
    """Analyze halo exchange performance."""
    halo = data.get("halo_profile") or \
           data.get("diag_rank0", {}).get("extra", {}).get("halo_profile")
    if not halo:
        return {"status": "no halo exchange data"}

    results = {}
    for key, info in halo.items():
        if isinstance(info, dict) and "mean_us" in info:
            # bandwidth_gb_s is None for a world_size==1 profile: a local
            # device-memory pad, not an inter-rank transfer, so "bandwidth"
            # is undefined (scaling_diagnostics 2026-07-24 fix). Preserve
            # None rather than coercing to 0 — 0 GB/s would read as a
            # pathologically slow link when in fact nothing crossed a rank.
            results[key] = {
                "mean_us": info["mean_us"],
                "p95_us": info.get("p95_us", 0),
                "bandwidth_gb_s": info.get("bandwidth_gb_s"),
                "local_pad_only": info.get("local_pad_only", False),
                "bytes_per_exchange": info.get("bytes_per_exchange", 0),
                "variability_pct": round(
                    info.get("std_us", 0) / info["mean_us"] * 100, 1
                ) if info["mean_us"] > 0 else 0,
            }

    # Bandwidth assessment — skip None (undefined for local-pad profiles);
    # default None when no exchange reported a real cross-rank bandwidth.
    _bws = [r["bandwidth_gb_s"] for r in results.values()
            if r.get("bandwidth_gb_s") is not None]
    max_bw = max(_bws) if _bws else None

    return {
        "exchanges": results,
        "peak_bandwidth_gb_s": max_bw,
        "assessment": _halo_assessment(results, max_bw),
    }


def _halo_assessment(results: dict, peak_bw: float | None) -> str:
    issues = []

    # No cross-rank bandwidth to assess (world_size==1 local-pad profile):
    # the exchanges are device-memory shuffles, so a bandwidth verdict would
    # be a category error. Report that and skip the bandwidth checks.
    if peak_bw is None:
        issues.append(
            "No inter-rank bandwidth measured (single-process / local-pad "
            "profile) — re-run with world_size>1 to assess halo bandwidth.")

    # Check bandwidth utilization (A100 NVLink: ~600 GB/s, PCIe: ~32 GB/s, InfiniBand: ~25 GB/s)
    if peak_bw is not None:
        if peak_bw < 1.0:
            issues.append(f"Very low bandwidth ({peak_bw:.1f} GB/s). Check: small message sizes, blocking sendrecv overhead, host-device copies.")
        elif peak_bw < 10.0:
            issues.append(f"Low bandwidth ({peak_bw:.1f} GB/s). Consider message aggregation or async overlap.")

    # Check variability
    for key, info in results.items():
        var = info.get("variability_pct", 0)
        if var > 50:
            issues.append(f"{key}: high variability ({var:.0f}% CV). Check: OS jitter, network contention, load imbalance.")

    # Check 3D vs 4D ratio
    if "scalar_3d" in results and "scalar_4d" in results:
        ratio = results["scalar_4d"]["mean_us"] / results["scalar_3d"]["mean_us"] if results["scalar_3d"]["mean_us"] > 0 else 0
        expected_ratio = results["scalar_4d"].get("bytes_per_exchange", 1) / max(results["scalar_3d"].get("bytes_per_exchange", 1), 1)
        if ratio > expected_ratio * 1.5:
            issues.append(f"4D halo {ratio:.1f}x slower than 3D but only {expected_ratio:.1f}x more data. Possible: memory layout, cache effects.")

    return " | ".join(issues) if issues else "Halo exchange performance appears reasonable."


def analyze_reductions(data: dict) -> dict[str, Any]:
    """Analyze MPI reduction performance."""
    red = data.get("reduction_profile") or \
          data.get("diag_rank0", {}).get("extra", {}).get("reduction_profile")
    if not red:
        return {"status": "no reduction data"}

    results = {}
    for key, info in red.items():
        if isinstance(info, dict):
            results[key] = {
                "local_us": info.get("local_sum_mean_us", 0),
                "mpi_us": info.get("mpi_sum_mean_us"),
                "overhead_factor": round(
                    info["mpi_sum_mean_us"] / info["local_sum_mean_us"], 1
                ) if info.get("mpi_sum_mean_us") and info.get("local_sum_mean_us") and info["local_sum_mean_us"] > 0 else None,
            }

    return {
        "reductions": results,
        "assessment": _reduction_assessment(results),
    }


def _reduction_assessment(results: dict) -> str:
    issues = []
    for key, info in results.items():
        if info.get("overhead_factor") and info["overhead_factor"] > 100:
            issues.append(f"{key}: MPI {info['overhead_factor']}x slower than local. Check: mpi4jax host callback overhead.")
    if not issues:
        return "Reduction overhead appears acceptable."
    return " | ".join(issues)


def analyze_memory(data: dict) -> dict[str, Any]:
    """Analyze memory usage patterns."""
    rank0 = data.get("diag_rank0", {})
    mem = rank0.get("memory", {})
    if not mem or not mem.get("snapshots"):
        return {"status": "no memory data"}

    return {
        "peak_mb": mem.get("peak_mb", 0),
        "peak_label": mem.get("peak_label", ""),
        "n_snapshots": mem.get("n_snapshots", 0),
        "snapshots": [
            {"label": s.get("label", ""), "devices": s.get("devices", {})}
            for s in mem.get("snapshots", [])[:10]  # first 10
        ],
    }


def analyze_roofline(data: dict) -> dict[str, Any]:
    """Analyze roofline model data."""
    roof = data.get("roofline") or \
           data.get("diag_rank0", {}).get("extra", {}).get("roofline")
    if not roof:
        return {"status": "no roofline data"}

    return {
        "state_mb": roof.get("state_mb", 0),
        "n_leaves": roof.get("n_leaves", 0),
        "mean_step_time_s": roof.get("mean_step_time_s", 0),
        "bandwidth_estimate_gb_s": roof.get("bandwidth_estimate_gb_s", 0),
        "assessment": _roofline_assessment(roof),
    }


def _roofline_assessment(roof: dict) -> str:
    bw = roof.get("bandwidth_estimate_gb_s", 0)
    # A100 HBM2e: ~2 TB/s peak
    if bw > 1000:
        return f"High bandwidth utilization ({bw:.0f} GB/s). Likely memory-bandwidth bound — optimize arithmetic intensity."
    elif bw > 100:
        return f"Moderate bandwidth ({bw:.0f} GB/s). Room for improvement in memory access patterns."
    else:
        return f"Low bandwidth ({bw:.0f} GB/s). Likely compute or latency bound, not memory-bandwidth limited."


def analyze_overlap(data: dict) -> dict[str, Any]:
    """Analyze compute/communication overlap potential."""
    olap = data.get("diag_rank0", {}).get("extra", {}).get("overlap_estimate")
    if not olap:
        return {"status": "no overlap data"}

    return {
        "halo_fraction_pct": olap.get("halo_fraction_pct", 0),
        "mean_step_ms": olap.get("mean_step_ms", 0),
        "mean_halo_ms": olap.get("mean_halo_ms", 0),
        # None when the standalone-halo probe exceeded the full step, i.e.
        # it was not measuring a component of the step (scaling_diagnostics
        # 2026-07-24 fix). Pass it through as-is; do not coerce to a number.
        "measurement_valid": olap.get("measurement_valid", True),
        "theoretical_speedup_pct": olap.get("theoretical_speedup_pct"),
        "assessment": _overlap_assessment(olap),
    }


def _overlap_assessment(olap: dict) -> str:
    # Refuse to report an overlap verdict from an invalid measurement — the
    # halo probe was not timing a component of this step, so its fraction is
    # meaningless (a >100% "halo fraction" is arithmetically impossible).
    if not olap.get("measurement_valid", True):
        return (olap.get("invalid_reason")
                or "Overlap measurement invalid (halo probe not a component "
                   "of the step) — re-run the diagnosis; no verdict.")
    frac = olap.get("halo_fraction_pct", 0)
    if frac > 30:
        return f"HIGH overlap potential ({frac:.0f}% halo). Implementing async halo exchange could yield significant speedup."
    elif frac > 10:
        return f"Moderate overlap potential ({frac:.0f}% halo). Async exchange would help but not transformative."
    else:
        return f"Low overlap potential ({frac:.0f}% halo). Communication is not the bottleneck."


def analyze_compilation(data: dict) -> dict[str, Any]:
    """Analyze JIT compilation costs."""
    rank0 = data.get("diag_rank0", {})
    comp = rank0.get("compilation", {})
    if not comp:
        return {"status": "no compilation data"}
    return {
        "n_compilations": comp.get("n_compilations", 0),
        "total_compile_s": comp.get("total_compile_s", 0),
        "compilations": comp.get("compilations", []),
    }


def analyze_cross_rank(data: dict) -> dict[str, Any]:
    """Analyze cross-rank load balance from summary.json."""
    summary = data.get("summary", {})
    per_rank = summary.get("per_rank_timing", [])
    if not per_rank:
        return {"status": "no cross-rank data"}

    # Compare step totals across ranks
    step_means = []
    for rd in per_rank:
        pt = rd.get("phase_timing", {})
        st = pt.get("_step_total", {})
        if st and "mean_us" in st:
            step_means.append(st["mean_us"])

    if not step_means:
        return {"status": "no step timing in cross-rank data"}

    mean = sum(step_means) / len(step_means)
    slowest = max(step_means)
    fastest = min(step_means)
    imbalance = (slowest - fastest) / mean * 100 if mean > 0 else 0

    return {
        "n_ranks": len(step_means),
        "mean_step_us": round(mean, 1),
        "fastest_us": round(fastest, 1),
        "slowest_us": round(slowest, 1),
        "imbalance_pct": round(imbalance, 1),
        "per_rank_means": [round(x, 1) for x in step_means],
        "assessment": _balance_assessment(imbalance, len(step_means)),
    }


def _balance_assessment(imbalance: float, n_ranks: int) -> str:
    if imbalance < 5:
        return "Well balanced across ranks."
    elif imbalance < 15:
        return f"Moderate imbalance ({imbalance:.0f}%). Check: different face sizes, asymmetric halo patterns."
    else:
        return f"SIGNIFICANT imbalance ({imbalance:.0f}%). Priority: investigate rank-specific bottlenecks, face assignment, memory pressure."


# ======================================================================
# Bottleneck ranking
# ======================================================================

def rank_bottlenecks(analyses: dict[str, Any]) -> list[dict[str, Any]]:
    """Rank optimization opportunities by estimated impact."""
    bottlenecks = []

    # Communication overhead
    phase = analyses.get("phase_breakdown", {})
    comm_pct = phase.get("comm_total_pct", 0)
    if comm_pct > 10:
        bottlenecks.append({
            "priority": 1 if comm_pct > 30 else 2,
            "category": "communication",
            "description": f"Communication takes {comm_pct:.0f}% of step time",
            "potential_speedup_pct": round(comm_pct * 0.5, 0),  # 50% of comm could be hidden
            "actions": [
                "Implement async halo exchange (overlap with interior compute)",
                "Aggregate messages: one sendrecv per neighbor instead of per-edge",
                "When mpi4jax gains Isend/Irecv, convert to non-blocking",
                "Reduce halo width if numerical scheme permits",
            ],
        })

    # Load imbalance
    balance = analyses.get("cross_rank_balance", {})
    imbalance = balance.get("imbalance_pct", 0)
    if imbalance > 10:
        bottlenecks.append({
            "priority": 1 if imbalance > 20 else 2,
            "category": "load_balance",
            "description": f"Cross-rank load imbalance of {imbalance:.0f}%",
            "potential_speedup_pct": round(imbalance * 0.3, 0),
            "actions": [
                "Check face assignment: some faces may have more computation (polar regions, physics)",
                "Profile per-rank memory pressure: OOM on one rank slows all ranks",
                "Consider dynamic load balancing or work stealing",
            ],
        })

    # Halo bandwidth — None means no inter-rank transfer was measured
    # (local-pad / single-process profile); a "low bandwidth" bottleneck is
    # not inferable from it, so skip the check rather than treat None as 0.
    halo = analyses.get("halo_exchange", {})
    peak_bw = halo.get("peak_bandwidth_gb_s")
    if peak_bw is not None and peak_bw < 5.0 and comm_pct > 10:
        bottlenecks.append({
            "priority": 2,
            "category": "halo_bandwidth",
            "description": f"Halo exchange bandwidth only {peak_bw:.1f} GB/s",
            "potential_speedup_pct": 10,
            "actions": [
                "Check message sizes: very small messages have high per-message overhead",
                "Pack multiple edges into single buffer per neighbor rank",
                "Verify GPU direct RDMA is active (not routing through host memory)",
                "Check NCCL vs UCX transport selection",
            ],
        })

    # Compilation cost
    comp = analyses.get("compilation", {})
    compile_s = comp.get("total_compile_s", 0)
    if compile_s > 30:
        bottlenecks.append({
            "priority": 3,
            "category": "compilation",
            "description": f"JIT compilation takes {compile_s:.0f}s",
            "potential_speedup_pct": 0,  # one-time cost
            "actions": [
                "Cache compiled functions across runs (jax.experimental.serialize)",
                "Reduce function complexity: fewer lax.scan steps, simpler physics",
                "Avoid shape-polymorphic code that prevents caching",
            ],
        })

    # Memory
    mem = analyses.get("memory", {})
    peak_mb = mem.get("peak_mb", 0)
    if peak_mb > 30000:  # > 30 GB
        bottlenecks.append({
            "priority": 2,
            "category": "memory",
            "description": f"Peak memory {peak_mb:.0f} MB",
            "potential_speedup_pct": 5,
            "actions": [
                "Enable gradient checkpointing for long scan sequences",
                "Reduce state duplication: avoid creating copies during halo exchange",
                "Use float32 for non-critical computations",
            ],
        })

    # Reduction overhead
    red = analyses.get("reductions", {})
    for key, info in red.get("reductions", {}).items():
        factor = info.get("overhead_factor")
        if factor and factor > 50:
            bottlenecks.append({
                "priority": 3,
                "category": "reduction_overhead",
                "description": f"MPI reduction '{key}' has {factor}x overhead vs local",
                "potential_speedup_pct": 2,
                "actions": [
                    "Reduce global reduction frequency (e.g., check CFL every N steps)",
                    "Batch multiple reductions into one allreduce",
                    "Use async allreduce where result is not immediately needed",
                ],
            })

    # Sort by priority
    bottlenecks.sort(key=lambda b: (b["priority"], -b.get("potential_speedup_pct", 0)))
    return bottlenecks


# ======================================================================
# Report generation
# ======================================================================

def generate_report(analyses: dict, bottlenecks: list, format: str = "json") -> str:
    """Generate the analysis report in the specified format."""
    if format == "json":
        report = {
            "analyses": analyses,
            "bottlenecks": bottlenecks,
            "recommendations": _generate_recommendations(analyses, bottlenecks),
        }
        return json.dumps(report, indent=2)

    elif format == "markdown":
        return _generate_markdown(analyses, bottlenecks)

    else:
        raise ValueError(f"Unknown format: {format}")


def _generate_recommendations(analyses: dict, bottlenecks: list) -> list[str]:
    """Generate prioritized optimization recommendations."""
    recs = []
    for b in bottlenecks[:5]:  # top 5
        recs.append(
            f"[P{b['priority']}] {b['description']} "
            f"(~{b.get('potential_speedup_pct', '?')}% speedup potential): "
            + b["actions"][0]
        )
    return recs


def _generate_markdown(analyses: dict, bottlenecks: list) -> str:
    """Generate a markdown report for Claude Code review."""
    lines = []
    lines.append("# legoESM Scaling Diagnostics Report\n")

    # Throughput
    tp = analyses.get("throughput", {})
    lines.append("## 1. Throughput Summary\n")
    lines.append(f"- **Time/step**: {tp.get('time_per_step_ms', '?')} ms")
    lines.append(f"- **SYPD**: {tp.get('sypd', '?')}")
    lines.append(f"- **Mcells/s**: {tp.get('mcells_per_s', '?')}")
    lines.append(f"- **Compile time**: {tp.get('compile_time_s', '?')} s")
    lines.append(f"- **Assessment**: {tp.get('assessment', 'N/A')}\n")

    # Phase breakdown
    pb = analyses.get("phase_breakdown", {})
    lines.append("## 2. Phase Breakdown\n")
    fracs = pb.get("fractions_pct", {})
    if fracs:
        lines.append("| Phase | % of Step |")
        lines.append("|-------|-----------|")
        for k, v in sorted(fracs.items(), key=lambda x: -x[1]):
            lines.append(f"| {k} | {v}% |")
        lines.append("")
    lines.append(f"- **Dominant**: {pb.get('dominant_phase', '?')} ({pb.get('dominant_pct', '?')}%)")
    lines.append(f"- **Communication**: {pb.get('comm_total_pct', '?')}%")
    lines.append(f"- **Compute**: {pb.get('compute_total_pct', '?')}%")
    lines.append(f"- **Assessment**: {pb.get('assessment', 'N/A')}\n")

    # Halo exchange
    he = analyses.get("halo_exchange", {})
    lines.append("## 3. Halo Exchange\n")
    for k, v in he.get("exchanges", {}).items():
        lines.append(f"- **{k}**: {v.get('mean_us', '?')} us (p95: {v.get('p95_us', '?')} us), "
                      f"BW: {v.get('bandwidth_gb_s', '?')} GB/s, "
                      f"variability: {v.get('variability_pct', '?')}%")
    lines.append(f"- **Assessment**: {he.get('assessment', 'N/A')}\n")

    # Reductions
    red = analyses.get("reductions", {})
    lines.append("## 4. MPI Reductions\n")
    for k, v in red.get("reductions", {}).items():
        lines.append(f"- **{k}**: local={v.get('local_us', '?')} us, "
                      f"MPI={v.get('mpi_us', '?')} us, "
                      f"overhead={v.get('overhead_factor', '?')}x")
    lines.append(f"- **Assessment**: {red.get('assessment', 'N/A')}\n")

    # Cross-rank balance
    bal = analyses.get("cross_rank_balance", {})
    if bal.get("n_ranks"):
        lines.append("## 5. Cross-Rank Balance\n")
        lines.append(f"- **Ranks**: {bal['n_ranks']}")
        lines.append(f"- **Fastest**: {bal.get('fastest_us', '?')} us")
        lines.append(f"- **Slowest**: {bal.get('slowest_us', '?')} us")
        lines.append(f"- **Imbalance**: {bal.get('imbalance_pct', '?')}%")
        lines.append(f"- **Assessment**: {bal.get('assessment', 'N/A')}\n")

    # Overlap
    olap = analyses.get("overlap", {})
    if olap.get("halo_fraction_pct") is not None:
        lines.append("## 6. Compute/Communication Overlap\n")
        lines.append(f"- **Halo fraction**: {olap['halo_fraction_pct']}% of step")
        lines.append(f"- **Theoretical speedup**: {olap.get('theoretical_speedup_pct', '?')}%")
        lines.append(f"- **Assessment**: {olap.get('assessment', 'N/A')}\n")

    # Roofline
    roof = analyses.get("roofline", {})
    if roof.get("state_mb"):
        lines.append("## 7. Roofline Position\n")
        lines.append(f"- **State size**: {roof['state_mb']} MB")
        lines.append(f"- **BW estimate**: {roof.get('bandwidth_estimate_gb_s', '?')} GB/s")
        lines.append(f"- **Assessment**: {roof.get('assessment', 'N/A')}\n")

    # Memory
    mem = analyses.get("memory", {})
    if mem.get("peak_mb"):
        lines.append("## 8. Memory\n")
        lines.append(f"- **Peak**: {mem['peak_mb']} MB ({mem.get('peak_label', '')})\n")

    # Bottlenecks
    lines.append("## 9. Bottleneck Ranking\n")
    for i, b in enumerate(bottlenecks, 1):
        lines.append(f"### {i}. [{b['category']}] {b['description']}")
        lines.append(f"- **Priority**: P{b['priority']}")
        lines.append(f"- **Potential speedup**: ~{b.get('potential_speedup_pct', '?')}%")
        lines.append("- **Actions**:")
        for a in b.get("actions", []):
            lines.append(f"  - {a}")
        lines.append("")

    # Recommendations
    lines.append("## 10. Top Recommendations\n")
    recs = _generate_recommendations(analyses, bottlenecks)
    for r in recs:
        lines.append(f"- {r}")
    lines.append("")

    # Raw data pointers
    lines.append("## Raw Data Files\n")
    lines.append("The following JSON files contain detailed data for deeper analysis:")
    lines.append("- `diag_rank0.json` — Full per-rank diagnostics (phase timing, memory, etc.)")
    lines.append("- `summary.json` — Cross-rank aggregated data")
    lines.append("- `halo_profile.json` — Detailed halo exchange profiling")
    lines.append("- `reduction_profile.json` — MPI reduction latency data")
    lines.append("- `roofline.json` — Roofline model data")
    lines.append("- `env.json` — Full environment snapshot\n")

    return "\n".join(lines)


# ======================================================================
# Comparison mode
# ======================================================================

def compare_runs(baseline: dict, current: dict) -> dict[str, Any]:
    """Compare two diagnostic runs to assess optimization impact."""
    comparison = {}

    # Throughput comparison
    b_scan = (baseline.get("summary", {}).get("scan_throughput") or
              baseline.get("diag_rank0", {}).get("extra", {}).get("scan_throughput", {}))
    c_scan = (current.get("summary", {}).get("scan_throughput") or
              current.get("diag_rank0", {}).get("extra", {}).get("scan_throughput", {}))

    if b_scan and c_scan:
        b_ms = b_scan.get("time_per_step_ms", 0)
        c_ms = c_scan.get("time_per_step_ms", 0)
        if b_ms > 0:
            speedup = (b_ms - c_ms) / b_ms * 100
            comparison["throughput"] = {
                "baseline_ms": b_ms,
                "current_ms": c_ms,
                "speedup_pct": round(speedup, 1),
                "baseline_sypd": b_scan.get("sypd", 0),
                "current_sypd": c_scan.get("sypd", 0),
            }

    # Halo exchange comparison
    b_halo = (baseline.get("halo_profile") or
              baseline.get("diag_rank0", {}).get("extra", {}).get("halo_profile", {}))
    c_halo = (current.get("halo_profile") or
              current.get("diag_rank0", {}).get("extra", {}).get("halo_profile", {}))

    if b_halo and c_halo:
        halo_comp = {}
        for key in set(b_halo) | set(c_halo):
            b_info = b_halo.get(key, {})
            c_info = c_halo.get(key, {})
            if isinstance(b_info, dict) and isinstance(c_info, dict):
                b_us = b_info.get("mean_us", 0)
                c_us = c_info.get("mean_us", 0)
                if b_us > 0:
                    halo_comp[key] = {
                        "baseline_us": b_us,
                        "current_us": c_us,
                        "change_pct": round((c_us - b_us) / b_us * 100, 1),
                    }
        comparison["halo_exchange"] = halo_comp

    return comparison


# ======================================================================
# CLI
# ======================================================================

def build_parser():
    p = argparse.ArgumentParser(
        description="Analyze scaling diagnostics from Levante runs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("run_dir", nargs="?", type=str,
                   help="Path to diagnostic run directory")
    p.add_argument("--baseline", type=str, default=None,
                   help="Baseline run directory for comparison")
    p.add_argument("--current", type=str, default=None,
                   help="Current run directory for comparison")
    p.add_argument("--format", choices=["json", "markdown"], default="markdown")
    p.add_argument("--output", type=str, default=None,
                   help="Output file (default: stdout)")
    return p


def main() -> int:
    args = build_parser().parse_args()

    # Comparison mode
    if args.baseline and args.current:
        baseline = load_run(Path(args.baseline))
        current = load_run(Path(args.current))
        comparison = compare_runs(baseline, current)
        output = json.dumps(comparison, indent=2)
        if args.output:
            Path(args.output).write_text(output)
        else:
            print(output)
        return 0

    # Single run analysis
    run_dir = Path(args.run_dir or args.current or ".")
    if not run_dir.exists():
        print(f"Error: {run_dir} does not exist", file=sys.stderr)
        return 1

    data = load_run(run_dir)

    # Run all analyses
    analyses = {
        "throughput": analyze_throughput(data),
        "phase_breakdown": analyze_phase_breakdown(data),
        "halo_exchange": analyze_halo_exchange(data),
        "reductions": analyze_reductions(data),
        "memory": analyze_memory(data),
        "roofline": analyze_roofline(data),
        "overlap": analyze_overlap(data),
        "compilation": analyze_compilation(data),
        "cross_rank_balance": analyze_cross_rank(data),
    }

    bottlenecks = rank_bottlenecks(analyses)
    report = generate_report(analyses, bottlenecks, format=args.format)

    if args.output:
        Path(args.output).write_text(report)
        print(f"Report written to {args.output}")
    else:
        print(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
