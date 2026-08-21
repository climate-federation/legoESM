# PRE-REGISTRATION -- #1455 ladder-promotion gate runs (2026-08-21)

Written and COMMITTED BEFORE the runs below were launched.  Everything after
"expected" is a prediction, not a result.

Tree: branch `fidelity/dino-step-walk`, HEAD as of this commit.
Harness: `run_fp64.py kamm_twin_90d.py nemo_dino_kamm_mlf <out>.npz --days 90
--bridge-before --save-3d`, `CUDA_VISIBLE_DEVICES=0`, `JAX_ENABLE_X64=1`,
corrected seasonal clock (harness default, `seasonal_t0_seconds = 15552000`).
Gate: `acceptance_gate_90d.py <out>.npz --level 5`, constants UNMODIFIED.
Channel band: `acc_thermal_wind.acc_band` (rows 14..48, true partial-cell
thickness weighting), the same call the walk used.

## Task 1 -- the 0.2 Sv "baseline discrepancy" discriminator

Established by reading the two runs' own provenance, BEFORE any new compute:

| run | HEAD | precision policy | E3T mode | day-90 ACC gap |
|---|---|---|---|---|
| recorded baseline (`/tmp/dino_clock_baseline/twin90_corrected.log`) | `8e56daafa` | **fp64** (`[run_fp64] control dtype = float64`) | off | +1.872 |
| walk shipped-grid arm (`twin90_off.log`) | `d120f9242` | **fp32 control** (bridge emitted the NON-fp64 warning) | off | +1.659 |

So the pair differs in TWO variables, not one:
 (C1) the precision policy -- `kamm_twin_90d.py` never sets a policy, it
      inherits the process default, and the walk did not wrap it in
      `run_fp64.py`;
 (C2) model code -- `git diff 8e56daafa..HEAD -- packages/ src/` is 22 files /
      2176 insertions, including
      `ocean/dynamics/barotropic_latlon_cgrid.py` (+187),
      `ocean/physics/tidal_forcing.py` (+145),
      `ocean/fidelity/time_levels.py` (+17) from PR #1628.
`d120f9242..HEAD` in `packages/`+`src/` is comment-only (26 lines of prose in
`nemo_state_bridge.py`), so the walk arms ARE at HEAD's model behaviour.

Determinism established: the walk's `off` arm and its `off_rep2` repeat are
bit-identical in every saved field.  Day 0 of baseline vs walk is bit-identical
(`max|d u3d_day0| = max|d T3d_day0| = 0.0`); they part company by day 30
(`max|d u3d| = 2.28e-2`).  So this is a real code/precision response, not noise.

**Discriminating run:** the shipped-grid (`off`) arm at HEAD *under fp64*.
That changes only (C1) relative to the walk arm and only (C2) relative to the
baseline.

* ACC gap within 0.02 Sv of **+1.872** => the PRECISION POLICY owns the 0.213 Sv
  and the #1628 model diff is inert on this card. CONFIRMED-precision.
* ACC gap within 0.02 Sv of **+1.659** => the MODEL CODE owns it (fp32/fp64 is
  inert here). CONFIRMED-model-code; then name the #1628 edit only if it can be
  pointed at in code, otherwise label PLAUSIBLE.
* anything else => both contribute; report the split, claim neither alone.

Either way the recorded +1.872 and the walk's +1.659 were measured on
DIFFERENT INSTRUMENTS and must never again be differenced as if they were one.

## Task 2 -- expected gate results

Both arms re-run at HEAD under fp64 so the two gate columns differ in ONE
variable (the ladder mode).  The walk's fp32 numbers below are the prediction
anchors, not the answer -- if the fp64 re-run moves them, the fp64 number is
the one that counts and the walk's table needs the same correction.

| metric | shipped `off` expected | NEMO/NEMO `both` expected |
|---|---|---|
| ACC full-section gap [Sv] | ~ +1.66 .. +1.87, FAIL (18-21x the 0.091 floor) | **~ -0.516, still FAIL** (~5.7x floor), improved from ~20.6x |
| channel-band ACC gap [Sv] | ~ +3.48 | **~ +0.099**, i.e. at the 0.091 floor |
| upper contrast <1400 m | PASS | PASS |
| deep contrast >1400 m | PASS | PASS |
| S-band surface sigma MAX | PASS | PASS |
| S-band surface sigma MEAN | PASS | PASS |

ACC is expected to remain the lone FAIL on the `both` arm.  The three density
metrics are NOT expected to move past their floors -- but the ladder change
alters the vertical geometry the density metrics are computed on, so if any of
them degrades past its floor that is a Rule-8 FAITHFUL-BUT-WORSE finding:
report it loudly, keep the faithful ladder, do NOT revert to hide it.

Refutation condition for the whole ladder story: if the `both` arm's channel
band comes back far from +0.099 under fp64, the walk's ladder ranking was a
precision artifact and Task 3 must not ship.

## Task 3 -- what the gate runs decide

Task 3 (twin-harness default -> `both`) is written against the numbers above.
If the fp64 re-run refutes the ladder ranking, the default does not move.
