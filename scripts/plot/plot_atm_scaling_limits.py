#!/usr/bin/env python
"""MPAS AMIP res-6 strong scaling on Derecho CPU vs provisional limit models.

Neither model line is a lower bound (codex + GLM review, 2026-10-26):
  memory model = every materialised array written (+ read) once through DRAM
    at the node's concurrent STREAM bandwidth; cache reuse can beat it.
  communication estimate = fastest observed call times + bytes / 6 GB/s.
Measured numbers, all 2026-10-26, main 9f2018f11a, 16 ranks x 8 cores/node:
  step times: slowest-rank mean over steps 48-111, untraced runs.
  W, MPI counts/bytes/times: rank 0, steps 48-79 (16 two-step windows, one
    radiation step), scripts/bench/bytes_census.py on traced runs (traced
    then with an uncommitted per-call wrapper; the committed route is
    scripts/cluster/scaling_derecho/amip_trace_census.sh).
  BW: scripts/cluster/scaling_derecho/stream_sync.sh, 1 node (common window).
  6 GB/s: scripts/bench/roofline_probe.py, 16 MB sendrecv, 2 ranks on 2 nodes.
  Raw logs and traces were in session scratch and are not archived.
usage: plot_atm_scaling_limits.py [out.png]
"""
import sys

import numpy as np

NODES = np.array([1, 2, 4, 8])
TOTAL_MS = np.array([876.1, 522.6, 322.7, 214.9])   # whole step
CSTEP_MS = np.array([764.0, 439.6, 270.5, 179.3])   # compiled model step only
W_GB = np.array([4.844, 2.867, 1.584, 0.892])       # bytes written per step
BW_RANK = 234.9 / 16                                # GB/s per rank
NSR = np.array([51.4, 43.0, 61.2, 61.2])            # sendrecv calls per step
BSR = np.array([12.508, 9.182, 6.773, 4.913]) * 1e6  # bytes received per step
TSR = np.array([9.8, 35.2, 17.6, 15.8]) * 1e-6      # fastest sendrecv call [s]
NAR = np.array([22.5, 23.2, 24.0, 24.0])            # allreduce calls per step
TAR = np.array([3.1, 7.9, 10.8, 13.4]) * 1e-6       # fastest allreduce call [s]
IN_MPI_MS = np.array([66.8 + 5.4, 52.5 + 21.3, 45.1 + 16.1, 27.4 + 12.9])  # incl. waiting
NET_BW = 6.0e9                                      # B/s, 16 MB inter-node sendrecv


def limits() -> dict:
    return dict(mem2=2 * W_GB / BW_RANK * 1e3, mem1=W_GB / BW_RANK * 1e3,
                comm=(NSR * TSR + BSR / NET_BW + NAR * TAR) * 1e3)


def main(out="atm_scaling_limits.png") -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    L = limits()
    for i, n in enumerate(NODES):
        print(f"{n} nodes: total {TOTAL_MS[i]:.0f}  compiled {CSTEP_MS[i]:.0f}  mem model "
              f"{L['mem1'][i]:.0f}-{L['mem2'][i]:.0f}  comm est {L['comm'][i]:.2f}  in MPI "
              f"{IN_MPI_MS[i]:.0f} ms; compiled/mem2 {CSTEP_MS[i]/L['mem2'][i]:.2f}")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    a = ax[0]
    a.loglog(NODES, TOTAL_MS, "s-", label="measured step")
    a.loglog(NODES, CSTEP_MS, "^-", label="measured: compiled model step only")
    a.loglog(NODES, TOTAL_MS[0] / NODES, "k:", lw=0.8, label="ideal scaling")
    a.fill_between(NODES, L["mem1"], L["mem2"], color="C2", alpha=0.25)
    a.loglog(NODES, L["mem2"], "-", color="C2",
             label="memory model (not a bound): every array written + read once via DRAM")
    a.loglog(NODES, L["mem1"], "--", color="C2", lw=0.8,
             label="memory model: every array written once via DRAM")
    a.loglog(NODES, L["comm"], "-", color="C3", label="communication wire-time estimate")
    a.loglog(NODES, IN_MPI_MS, ":", color="C3",
             label="measured: time inside MPI calls, rank 0 (incl. waiting)")
    a.set_xticks(NODES); a.set_xticklabels(NODES)
    a.set_xlabel("nodes (16 ranks x 8 cores)"); a.set_ylabel("ms per step")
    a.set_title("Strong scaling vs provisional limit models"); a.legend(fontsize=7)
    a = ax[1]
    a.plot(NODES, CSTEP_MS / L["mem2"], "o-", label="compiled step / memory model")
    a.plot(NODES, TOTAL_MS / L["mem2"], "s-", label="whole step / memory model")
    for x, y in zip(NODES, CSTEP_MS / L["mem2"]):
        a.annotate(f"{y:.2f}x", (x, y), textcoords="offset points", xytext=(4, -12), fontsize=8)
    a.axhline(1, color="k", ls=":", lw=0.8); a.set_xscale("log")
    a.set_xticks(NODES); a.set_xticklabels(NODES)
    a.set_xlabel("nodes"); a.set_title("Ratio to memory model"); a.legend(fontsize=8)
    fig.suptitle("legoESM MPAS atmosphere AMIP res 6, Derecho CPU. Bytes per step and MPI "
                 "counts measured on rank 0 at each node count; bandwidth measured on 1 node",
                 fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return L


if __name__ == "__main__":
    main(*sys.argv[1:2])
