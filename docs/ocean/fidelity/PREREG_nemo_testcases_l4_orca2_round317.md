# PREREG — ORCA2 round 317: post-fix gradient picture (frozen before measuring)

Base `d0db4b99fb53` (round 316 landed). MEASUREMENT ONLY: no model file
changes. Instruments: round 315's chained probe (unchanged) and round 316's
fused instrument with a new `--cost` mode (per-N compile and warm-run time,
sampled run peak RSS, XLA's buffer estimate, gradient norm; remat on/off).

Already read before freezing (not predictions): round 316's partial OMT
reverse rows (finite, all N), its fused rung-0 and rung-10 N = 10 results
(finite; rung 10 reverse . T_dir0 vs forward mode 2.0e-5 T, 1.7e-5 SSH;
fused forward not bit-identical to production on either card).
Not read before freezing: the completed rung-0 probe's check rows, the
rung-10 probe's check rows, every re-run and cost row.

Memory rule: at most one process > 100 GB at a time (round 316's rung-10
probe counts); a guard stops my own process if available memory < 150 GB.

Predictions:
- P1 rung 0 (round 316 probe): reverse = forward mode to <= 1e-10 on every
  T-direction and scalar row, N = 1, 2, 5, 10 (falsifier: any larger gap or NaN).
- P2 OMT-1..4 re-run one at a time (round 316's command, `--remat`):
  finite at every N; reverse = forward mode <= 1e-10 on every row; peak
  RSS <= ~150 GB each (falsifier: NaN, a larger gap, or a cap stop).
- P3 rung 10 (C2): forward mode finite at N = 1, 2, 5, 10 for both losses;
  reverse = forward mode <= 1e-10 at N = 1 (falsifier: NaN or larger gap).
- P4 (PLAUSIBLE mechanism for the rung-10 2e-5 gap): the reverse and
  forward executables compute last-bit-different primal trajectories and
  rung 10's switches (convective-adjustment trigger etc.) land on different
  sides, so each mode differentiates a different branch. Discriminator,
  fused, T loss: N = 1 gap <= 1e-10, and at N = 10 the forward-mode primal
  loss differs in bits from the reverse primal loss. Falsifier: identical
  primal bits at N = 10 with the gap still > 1e-10 (a derivative-rule
  asymmetry instead), or a gap > 1e-10 already at N = 1.
- P5 cost (fused, T loss, rung 0 and rung 10, N = 1, 2, 5, 10): warm
  backward time roughly linear in N; with remat the run's memory growth
  is roughly flat in N; without remat it grows roughly linearly; rung 10
  without remat exceeds 95 GB by N = 10.
