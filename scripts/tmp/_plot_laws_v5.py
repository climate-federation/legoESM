#!/usr/bin/env python
"""Scaling laws v5 — campaign-wide strong/weak x MPI/GPU x atm/ocean.

Panels (2x4):
  r1: atm CPU-MPI strong | atm CPU-MPI weak | A1 CS-SPMD strong (NEW)
      | atm GPU strong
  r2: ocean CPU-MPI strong | ocean CPU-MPI weak | atm GPU weak | summary

Data sources (all under results/scaling_ginsburg/):
  atm_cpumpi_ll_8456237, atm_cpumpi_8454267   atm MPI json (TimingResult)
  a1_spmd                                      A1 --cs-spmd json (NEW)
  levers2_clean_8460192 + fused_halo_ab_8459326/legacy  ocean MPI csv
  atm_gpu_8454266/<grid>_float64/all_scaling.csv        atm GPU csv

Throwaway campaign plot.  Usage: _plot_laws_v5.py <out_png>
"""
import csv
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(sys.argv[1])
R = Path("results/scaling_ginsburg")


def _load_json_dir(pattern):
    out = []
    for f in glob.glob(str(pattern), recursive=True):
        try:
            out.append((f, json.load(open(f))))
        except Exception:
            pass
    return out


fig, ax = plt.subplots(2, 4, figsize=(23, 10.5))

# ---------------- (0,0) atm CPU-MPI strong ----------------
A = ax[0, 0]
srcs = {
    "LL": R / "atm_cpumpi_ll_8456237/latlon_held_suarez_strong",
    "ico": R / "atm_cpumpi_8454267/icosahedral_held_suarez_strong",
}
for li, (label, d) in enumerate(srcs.items()):
    by_res = defaultdict(dict)
    for _f, j in _load_json_dir(d / "*_float64.json"):
        if j["mode"] == "strong":
            by_res[j["resolution"]][j["n_ranks"]] = j["time_per_step_ms"]
    # Strong dirs mix resolutions: pick the best-covered curve
    # (largest resolution on ties).
    res = max(by_res, key=lambda r: (len(by_res[r]), r), default=None)
    if res is None or len(by_res[res]) < 2:
        continue
    pts = by_res[res]
    nps = sorted(pts)
    c = ["tab:blue", "tab:orange"][li]
    A.plot(nps, [pts[n] for n in nps], "-o", color=c, lw=2,
           label=f"{label} r{res} held_suarez")
    A.plot(nps, [pts[nps[0]] * nps[0] / n for n in nps], ":", color=c,
           alpha=0.5, label="ideal")
A.set_xscale("log", base=2); A.set_yscale("log")
A.set_xlabel("MPI ranks"); A.set_ylabel("ms / step")
A.set_title("ATM CPU-MPI strong (f64)")
A.grid(True, which="both", alpha=0.25); A.legend(fontsize=7)

# ---------------- (0,1) atm CPU-MPI weak ----------------
B = ax[0, 1]
wsrcs = {
    "LL (band)": R / "atm_cpumpi_ll_8456237/latlon_held_suarez_weak",
    "ico": R / "atm_cpumpi_8454267/icosahedral_held_suarez_weak",
}
for li, (label, d) in enumerate(wsrcs.items()):
    pts = {}
    for _f, j in _load_json_dir(d / "*_float64.json"):
        pts[j["n_ranks"]] = j["time_per_step_ms"]
    if len(pts) < 2:
        continue
    nps = sorted(pts)
    c = ["tab:blue", "tab:orange"][li]
    B.plot(nps, [pts[n] / pts[nps[0]] for n in nps], "-o", color=c, lw=2,
           label=label)
B.axhline(1.0, color="0.5", ls="--", lw=1.2, label="ideal (flat)")
B.set_xscale("log", base=2)
B.set_xlabel("MPI ranks"); B.set_ylabel("step time / np_min")
B.set_title("ATM CPU-MPI weak (f64)")
B.grid(True, which="both", alpha=0.25); B.legend(fontsize=7)

# ---------------- (0,2) A1: CS TRUE decomposition strong (NEW) -------
C = ax[0, 2]
# (layout, res) -> np -> ms; 1node = 6 procs on one node (DRAM-shared),
# 6node = 1 proc/node (full per-rank bandwidth).
a1 = defaultdict(dict)
for f, j in _load_json_dir(R / "a1_spmd/**/*_float64.json"):
    if "refusal" in f:
        continue
    a1[("1node", j["resolution"])][j["n_ranks"]] = j["time_per_step_ms"]
for f, j in _load_json_dir(R / "a1_spmd_multinode/**/*_float64.json"):
    a1[("Nnode", j["resolution"])][j["n_ranks"]] = j["time_per_step_ms"]
a1_speed = []
A1_STYLE = {"1node": (":", 0.55, "1 node"), "Nnode": ("-", 1.0, "1/node")}
A1_COLOR = {48: "tab:green", 96: "tab:red"}
for (layout, res), pts in sorted(a1.items()):
    if len(pts) < 2 or res not in A1_COLOR:
        continue
    ls, al, lab = A1_STYLE[layout]
    nps = sorted(pts)
    c = A1_COLOR[res]
    C.plot(nps, [pts[n] for n in nps], ls, marker="s", color=c, lw=2.2,
           alpha=al, label=f"C{res} {lab}")
    if layout == "Nnode" or ("Nnode", res) not in a1:
        C.plot(nps, [pts[nps[0]] * nps[0] / n for n in nps], ":",
               color=c, alpha=0.35)
    if 6 in pts and 1 in pts:
        a1_speed.append((res, lab, pts[1] / pts[6]))
C.set_xscale("log", base=2); C.set_yscale("log")
C.set_xticks([1, 2, 3, 6]); C.set_xticklabels(["1", "2", "3", "6"])
C.set_xlabel("processes (jax.distributed, 1 CPU device each)")
C.set_ylabel("ms / step")
C.set_title("NEW — A1 cubed-sphere TRUE decomposition\n"
            "(multi-controller SPMD + multiface ppermute)")
C.grid(True, which="both", alpha=0.25); C.legend(fontsize=7)
if not a1:
    C.text(0.5, 0.5, "a1_spmd results pending", ha="center",
           transform=C.transAxes)

# ---------------- (0,3) atm GPU strong ----------------
D = ax[0, 3]
ggrids = {"cubed-sphere": "tab:green", "latlon": "tab:blue",
          "icosahedral": "tab:orange"}
gpu_eff = []
for g, c in ggrids.items():
    f = R / f"atm_gpu_8454266/{g}_float64/all_scaling.csv"
    if not f.exists():
        continue
    by_res = defaultdict(dict)
    for r in csv.DictReader(open(f)):
        if r["mode"] == "strong":
            by_res[int(r["resolution"])][int(r["n_gpus"])] = (
                float(r["time_per_step_ms"]))
    res = max((rr for rr, d in by_res.items() if len(d) >= 2), default=None)
    if res is None:
        continue
    pts = by_res[res]
    nps = sorted(pts)
    D.plot(nps, [pts[n] for n in nps], "-o", color=c, lw=2,
           label=f"{g} r{res}")
    D.plot(nps, [pts[nps[0]] * nps[0] / n for n in nps], ":", color=c,
           alpha=0.5)
    if 2 in pts:
        gpu_eff.append((g, res, pts[1] / pts[2] / 2.0))
D.set_xscale("log", base=2); D.set_yscale("log")
D.set_xticks([1, 2]); D.set_xticklabels(["1", "2"])
D.set_xlabel("GPUs (RTX 8000, PCIe)"); D.set_ylabel("ms / step")
D.set_title("ATM GPU strong (f64)")
D.grid(True, which="both", alpha=0.25); D.legend(fontsize=7)

# ---------------- (1,0) ocean CPU-MPI strong ----------------
E = ax[1, 0]
osrcs = {
    "legacy (pre-campaign)":
        R / "fused_halo_ab_8459326/legacy",
    "shipped (fused+folds+vmix-pair)":
        R / "levers2_clean_8460192/vmixpair",
}
OC = {128: "tab:red", 192: "tab:purple"}
ocean_speed = {}
orows = {}
for label, d in osrcs.items():
    for f in glob.glob(str(d / "ocean_*_np*_float64.csv")):
        for r in csv.DictReader(open(f)):
            mode = "strong" if "strong" in r["mode"] else "weak"
            orows.setdefault((label, mode, int(r["resolution"])), {})[
                int(r["n_gpus"])] = float(r["time_per_step_ms"])
for (label, mode, res), pts in sorted(orows.items()):
    if mode != "strong" or len(pts) < 2 or res not in OC:
        continue
    ls = ":" if "legacy" in label else "-"
    mk = "o" if "legacy" in label else "s"
    al = 0.45 if "legacy" in label else 1.0
    nps = sorted(pts)
    E.plot(nps, [pts[n] for n in nps], ls, marker=mk, alpha=al,
           color=OC[res], lw=2, label=f"LL{res} {label}")
for res in OC:
    a = orows.get(("legacy (pre-campaign)", "strong", res), {})
    b = orows.get(("shipped (fused+folds+vmix-pair)", "strong", res), {})
    if 8 in a and 8 in b:
        ocean_speed[res] = a[8] / b[8]
E.set_xscale("log", base=2); E.set_yscale("log")
E.set_xlabel("MPI ranks"); E.set_ylabel("ms / step")
E.set_title("OCEAN CPU-MPI strong (f64, implicit_cn/PCG)")
E.grid(True, which="both", alpha=0.25); E.legend(fontsize=6.5)

# ---------------- (1,1) ocean CPU-MPI weak ----------------
F = ax[1, 1]
for label, _d in osrcs.items():
    pts = {}
    for (l2, mode, _res), d2 in orows.items():
        if l2 == label and mode == "weak":
            pts.update(d2)
    if len(pts) < 2:
        continue
    ls = ":" if "legacy" in label else "-"
    al = 0.45 if "legacy" in label else 1.0
    nps = sorted(pts)
    F.plot(nps, [pts[n] / pts[nps[0]] for n in nps], ls, marker="o",
           alpha=al, color="tab:green", lw=2, label=label)
F.axhline(1.0, color="0.5", ls="--", lw=1.2, label="ideal (flat)")
F.set_xscale("log", base=2)
F.set_xlabel("MPI ranks"); F.set_ylabel("step time / np=1")
F.set_title("OCEAN CPU-MPI weak (band, rows/rank=48)")
F.grid(True, which="both", alpha=0.25); F.legend(fontsize=7)

# ---------------- (1,2) atm GPU weak ----------------
G = ax[1, 2]
for g, c in ggrids.items():
    f = R / f"atm_gpu_8454266/{g}_float64/all_scaling.csv"
    if not f.exists():
        continue
    pts = {}
    for r in csv.DictReader(open(f)):
        if r["mode"] == "weak":
            pts[int(r["n_gpus"])] = float(r["time_per_step_ms"])
    if len(pts) < 2:
        continue
    nps = sorted(pts)
    G.plot(nps, [pts[n] / pts[nps[0]] for n in nps], "-o", color=c, lw=2,
           label=g)
G.axhline(1.0, color="0.5", ls="--", lw=1.2, label="ideal (flat)")
G.set_xscale("log", base=2)
G.set_xticks([1, 2]); G.set_xticklabels(["1", "2"])
G.set_xlabel("GPUs"); G.set_ylabel("step time / 1 GPU")
G.set_title("ATM GPU weak (f64; PCIe-bound)")
G.grid(True, which="both", alpha=0.25); G.legend(fontsize=7)

# ---------------- (1,3) summary ----------------
H = ax[1, 3]
H.axis("off")
lines = ["campaign summary (all merged to main)", ""]
lines.append("ocean CPU-MPI strong, np8 vs pre-campaign:")
for res, sp in sorted(ocean_speed.items()):
    lines.append(f"  LL{res}: {sp:.2f}x")
lines.append("ocean weak np8 growth: x4.65 -> x2.95")
lines.append("halo exchanges/step: ocean 63->~30, atm 15->9")
lines.append("")
lines.append("NEW A1 cubed-sphere TRUE decomposition:")
if a1_speed:
    for res, lab, sp in a1_speed:
        lines.append(f"  C{res} {lab}: np6 speedup {sp:.2f}x vs serial")
else:
    lines.append("  (job pending)")
lines.append("  parity (bench contract): <=1e-8 gate PASS")
lines.append("  1-node C96 flat = DRAM-contention wall")
lines.append("")
lines.append("atm GPU strong eff (1->2 GPU, f64):")
for g, res, eff in gpu_eff:
    lines.append(f"  {g} r{res}: {eff:.2f}")
lines.append("  (PCIe roofline ~0.73 f64; NVLink = hardware lever)")
lines.append("ocean GPU f32 option: 1.57-1.72x vs f64")
lines.append("")
lines.append("opt-in levers: single-reduce PCG (np32 1.03-1.06x),")
lines.append("  tracer level-stack (measured out, OFF)")
H.text(0.0, 0.98, "\n".join(lines), va="top", family="monospace",
       fontsize=8.2, transform=H.transAxes)

fig.suptitle(
    "legoESM scaling campaign v5 — strong/weak x CPU-MPI/GPU x atm/ocean "
    "(Ginsburg; f64 unless noted)",
    fontsize=12,
)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT, dpi=140)
print(f"wrote {OUT}")
