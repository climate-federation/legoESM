# Distance-to-limit assessment — legoESM scaling campaign (2026-06-13)

Honest per-axis "how close to the theoretical limit" for weak/strong ×
MPI/GPU × atm/ocean, anchored to measured rooflines (roofline_probe.py,
cross-node collective clock 8473330) and phase splits. Every number has a
job receipt; "AT LIMIT" means the gap is hardware or algorithm-fundamental,
not engineering debt.

## Atmosphere

| Axis | State | Limit verdict |
|---|---|---|
| Cubed-sphere per-device | 7.2e-9 s/cell/step (JAX-Fluids-class) | **AT JAX-peer limit** — per-device XLA-fusion closed; CS dycore not launch-bound (command-buffer A/B null) |
| Cubed-sphere 2-GPU strong | eff 0.73 (f64) | **AT PCIe-link roofline** — ppermute 70-245× < HBM (measured); headroom needs NVLink/IB (absent on Ginsburg) |
| Cubed-sphere multinode SPMD | C96 np6 2.42× (1 proc/node) | bandwidth-class, scales with size; PRODUCTION path shipped (`distributed_mode='spmd'`); halo-count lever (F1 packing shipped) |
| Lat-lon CPU-MPI | comm 15→6 exch/step | 1-D band; **2-D decomposition is the open structural lever** (task #7) for the strong limit at high ranks |
| Icosahedral/MPAS 2-GPU | eff 0.88 | at link limit; distributed PCG shipped |
| Spectral | single-device | **by design** (global SH transform = cuBLAS-bound matmul); multi-rank N/A documented |

## Ocean (lat-lon C-grid, production tile rows/rank=48, phase-split job 8475898)

Production-tile phase split (np4 LL192): implicit_vmix 44.9%, baroclinic
16.7%, barotropic 10.1%, tracer 8.4%, residual 16.9%.

| Phase | %step | Rank-behavior | Limit verdict |
|---|---|---|---|
| implicit_vmix | 38-61% | shards ~perfectly (0 comm np1) | per-device THROUGHPUT cost, NOT a scaling limiter; f32 storage = 1.7× (opt-in; f64 = scientific choice); easy compute wins shipped (T+S shared-factor +15%); remaining = memory-traffic refactor (fuse K-profile into Thomas) — task #12, bounded by being shard-friendly |
| baroclinic | 16-21% | 27 halos = #1 rank-GROWING | **THE weak-scaling lever** — halo fusion (vertex-mask hoist shipped 39→37; neumann_fill_cgrid 12-exch cluster = next, same static-mask-hoist pattern) |
| barotropic | 9-10% | 120 reductions | reduction wall NEUTRALIZED (zonal_line / single_reduce, opt-in; regime-crossover job 8475875: +20-26% small tiles, neutral production tile because the phase is only ~6-10%) — **ceiling bounded by phase fraction** |
| tracer_advection | 8% | 4 halos | minor; tracer-pair restructure measured non-win |

Weak np16→32 production tile: ×1.6-1.8 (cross-node Gloo-TCP latency-bound;
≤8 ranks/node policy — DRAM contention above that, job 8474286).

## Where the real remaining headroom is (ranked)

1. **Ocean baroclinic halo fusion** (weak-scaling): 27 halos → fewer via the
   static-mask-sequence hoist (vertex-mask done; neumann_fill_cgrid 12-exch
   next). Measurable at production tile; bounded by baroclinic's 16-21%.
2. **Ocean vmix memory-traffic** (throughput, not scaling): 38-61% of step,
   shards fine; fuse K-profile build into the solve, cut intermediates.
3. **Lat-lon 2-D decomposition** (atm+ocean strong limit at high ranks):
   structural, big effort (task #7).
4. **Multi-node GPU SPMD**: architecture proven (A1) but interconnect-capped
   on Ginsburg PCIe (ppermute < SYS/PCIe < network); **defer to NVLink/IB
   hardware** — not an engineering gap here.

## Bottom line

The high-ROI axes are AT their limits: atm per-device (JAX-peer), atm GPU
strong (PCIe roofline), ocean barotropic reductions (neutralized). Remaining
gaps are (a) deep multi-operator halo-fusion refactors with payoff bounded by
each phase's step-fraction, or (b) hardware-blocked (no NVLink/IB). The
campaign is in diminishing-returns territory on this hardware; each further
lever is a single-digit-% production gain for multi-session effort.
