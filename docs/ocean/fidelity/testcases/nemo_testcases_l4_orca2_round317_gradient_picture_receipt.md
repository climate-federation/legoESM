# ORCA2 round 317 — post-fix gradient picture (measurement only)

Base `d0db4b99fb53` (round 316). Prereg `d85c4fa0f323` (frozen before any
measurement). No model file changed. Evidence:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round317/`
(`jobs/`; round 316's detached results stay in `round316/jobs/`). CPU, fp64,
x64, production JIT, libm policy. Machine shared and loaded (load ~20 on 80
cores): wall times are upper bounds.

STATUS: PARTIAL — written before the long runs finished; sections marked
PENDING are filled on resume.

## Verdict (so far)
- Rung 0: finite at every N; reverse = forward mode to <= 8.5e-14 on all 64
  rows (P1 holds). Drag now carries a gradient.
- Rung 10: forward mode is finite at every N on both losses — the round-315
  forward-mode NaN is gone after C2. Reverse vs forward disagree: <= 2.1e-9
  at N = 1 (T loss), up to 2.6e-5 from N = 2 (P3 falsified on its 1e-10 bar).
- Rung 10: drag still cannot be passed traced: a sixth set of `float()`
  casts in the TKE bottom boundary condition.

PENDING: OMT-1..4 re-runs, rung-10 scalar rows, cost table, P4.
