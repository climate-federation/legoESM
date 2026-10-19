# ORCA2 round 317 — post-fix gradient picture (measurement only)

Base `d0db4b99fb53` (round 316). Prereg `d85c4fa0f323` (frozen before any
measurement). No model file changed. Evidence:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round317/`
(`jobs/`; round 316's detached results stay in `round316/jobs/`). CPU, fp64,
x64, production JIT, libm policy. The machine is shared and loaded (load
~20 on 80 cores): wall times are upper bounds.

## Verdict
- No NaN remains on any measured card, in either mode, at any N.
- Rung 0 and OMT-1: reverse = forward mode to <= 1.5e-13 on every row.
- Rung 10: forward mode is finite at every N (round 315's forward NaN is
  gone after C2). Reverse and forward mode differ by up to 4e-9 at N = 1
  and 2.6e-5 from N = 2 (T and SSH losses).
- Training cost with per-step remat is flat in memory: 17.6 GB (rung 0) and
  23.6 GB (rung 10) at N = 10; backward ~8 s (rung 0) and ~37 s (rung 10)
  per extra step.
- Drag still cannot be trained on rung 10 (TKE cards): five more `float()`
  casts at `ocean_model_latlon_cgrid.py:11589-11593`.

## Round 316's detached results (item 1)
| card (commit) | finite, all N | reverse vs forward (worst) | FD, best eps | peak |
|---|---|---|---|---|
| rung 0 (`7a1825f4d6f2`) | yes | 8.5e-14 (64 rows) | 2.4e-9 .. 1.6e-3 | 149 GiB |
| rung 10 C2 (`2f1b26df90a2`) | yes, both modes | 2.6e-5 (T-dir rows); scalars 1.0e-12 | 3.0e-9 .. 8.7e-3 | 241 GiB |
| fused rung 0, N = 10 | yes | 3.5e-15 (T), 8.2e-16 (SSH) | — | — |
| fused rung 10 C2, N = 10 | yes | 2.0e-5 (T), 1.7e-5 (SSH) | — | — |

- FD at the flip-free eps: the branch-flip counter ran on OMT-2 only
  (round 316: no flips at 1e-7, T FD 7.8e-7, pass). The probe's sweep stops
  at 1e-6, where OMT-2 still has 30 flips, so no other card has an FD row at
  a flip-free eps. At 1e-6 the T-loss FD misses by 1e-5 .. 8e-4 (rung 0,
  rung 10), the size round 316 tied to the flips. NOT certified either way.
- Drag (rung 0): traceable now; construction parity is not bit-identical
  (rebuilt and traced paths round differently), but the rebuilt-model FD
  matches the traced gradient to FD noise (T, N = 10: 5.35799e-5 vs
  5.35779e-5).
- Rung 10 inert inputs: the isoneutral diffusivity (AD = FD = 0; the card
  does not run it) and the background diffusivity floor 1e-10 m2/s (AD
  exactly 0; FD = one-ulp loss changes / 2 eps). Under the no-inert rule a
  rung-10 trainer must freeze both out.

## OMT re-runs, one at a time (item 2)
Round 316's command (`--remat`), memory guard per process (stop if the
process exceeds its cap or available memory < 150 GiB).
| card | status | finite | reverse vs forward | peak |
|---|---|---|---|---|
| OMT-1 (`d85c4fa0f323`) | complete, 6,617 s | yes, N = 1, 2, 5 | 1.5e-13 (42 rows); 4 parity controls pass | 95.5 GiB |
| OMT-2 | running at commit | — | — | — |
| OMT-3, OMT-4 | queued after OMT-2 | — | — | — |

OMT-1's FD at 1e-6 misses by up to 0.3 on T_dir1, N = 5 (best 1.4e-3 at
0.1): a switch, since reverse = forward there to 1e-13.

## Rung 10 forward mode (item 3)
Finite at N = 1, 2, 5, 10 on both losses, three T directions and three
scalars (chained probe) and at N = 10 fused. Nothing left to locate.

The reverse/forward gap (cost-mode runs, fused, T loss, T_dir0):
| card, N | gap | primal loss, reverse vs forward executable |
|---|---|---|
| rung 10, 1 | 3.3e-10 | bit-identical |
| rung 10, 10 | 2.0e-5 | differ by 2 ulp (production differs from both) |
| rung 0, 10 | 3.5e-15 | differ too |

- CONFIRMED: the reverse, forward and production executables compute
  last-bit-different trajectories by N = 10 on both cards.
- PLAUSIBLE (P4): rung 10's switches (convective adjustment etc.) turn that
  drift into a 2e-5 gradient gap; rung 0 has no such switches. The 3.3e-10
  gap at N = 1 with identical primal loss is not explained (P4's N = 1 bar
  of 1e-10 failed). For training, both modes are derivatives of branches
  within rounding of each other.

## Online-training cost (item 4)
Fused `jit(value_and_grad)` of the T loss, step 1 + scan over steps 2..N,
one N per process (`--cost`). "Run" = warm backward (forward included),
compile excluded; "memory" = the run's sampled RSS growth (XLA's own buffer
estimate agrees to within 6 %).
| card | N | backward, remat | memory, remat | backward / memory, no remat | |dL/dT0| |
|---|---|---|---|---|---|
| rung 0 | 1 | 10.7 s | 6.4 GB | same executable | 1.93417e-2 |
| rung 0 | 2 | 40.1 s | 17.1 GB | 20.9 s / 16.8 GB | 1.93409e-2 |
| rung 0 | 5 | 65.7 s | 17.2 GB | compile > 95 GiB cap; retry PENDING | 1.93394e-2 |
| rung 0 | 10 | 103.9 s | 17.6 GB | PENDING | 1.93395e-2 |
| rung 10 | 1 | 28.3 s | 9.8 GB | same executable | 1.93357e-2 |
| rung 10 | 2 | 102.0 s | 21.6 GB | PENDING | 1.93348e-2 |
| rung 10 | 5 | 224.2 s | 22.9 GB | PENDING | 1.93330e-2 |
| rung 10 | 10 | 398.5 s | 23.6 GB | PENDING | 1.93325e-2 |

- Compile: 6-11 min at N = 1, 25-36 min for N >= 2 (one per shape).
  Process peaks with remat: 24-76 GiB (rung 0), 85-89 GiB (rung 10).
- Gradient norm growth with N (chained probe, every card): T loss flat
  (1.9332e-2 .. 1.9342e-2 from N = 1 to 10); SSH loss grows ~8-11x
  (6.86e-4 at N = 1; 5.5e-3 rung 0 / OMT-4, 6.7e-3 OMT-3, 7.7e-3 OMT-2,
  5.5e-3 rung 10 at N = 10).
- Chained-probe backward (OMT cards, remat, compile excluded): N = 10
  135-226 s (rung 0 / OMT-2..4, round 316), N = 5 43 s (OMT-1).
- The fused and chained instruments agree on |dL/dT0| (rung 10, N = 10:
  1.93325e-2 both).

## Instrument (`nemo_testcase_l4_orca2_round316_fused_grad.py --cost`)
Direct tests (`tests/ocean/fidelity/test_orca2_round316_instruments.py`,
9 pass): gradient = analytic with and without remat; cost rows = analytic,
forward mode and primal bits; the memory sampler sees a known 0.4 GB
allocation; `--no-remat` raises XLA's buffer estimate > 2x on a toy scan.
Plants (`test_plants/plants.log`): remat ignored, sampler not polling,
wrong N, `--no-remat` ignored — each fails one test.
Infrastructure: several N in one process hit the per-process memory-map
limit (exit 134, the round-315 failure); fixed by releasing each N's
executable (`db34474d9d70`) and running one N per process.

## Memory rule
At most one process above 100 GiB at a time (round 316's rung-10 probe,
then OMT-2); cost runs and OMT-1 ran beside it under a 95 GiB cap (units as
`free -g`; OMT-1's 95.5 GiB is 102.5 GB decimal). Lowest available memory
seen: 666 GiB. No OOM kill.

## PENDING (detached, results land in `round317/jobs/`)
OMT-2, OMT-3, OMT-4 probes (serial, 160 GiB cap); rung 0 no-remat N = 10;
rung 10 no-remat N = 2, 5, 10 (95 GiB cap); every cap-stopped cost row is
retried under 150 GiB after the OMT probes.

## Gates
- Citation gate on this receipt: see `gates/citation_gate_receipt.json`.
- Default citation-gate test + private-import ratchet: 23 passed.

codex review: operator
