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

---

# RESULT (2026-08-21, appended after the runs, predictions above left unedited)

All runs: HEAD `0283a5223`, clean tree, `run_fp64.py` wrapper (control dtype
float64 confirmed in every log), `CUDA_VISIBLE_DEVICES=0`, `JAX_ENABLE_X64=1`,
corrected clock (`seasonal_t0_seconds = 15552000` stamped), `--bridge-before`,
90 days = 2880 steps, both arms STABLE, both bit-identical to NEMO at day 0.

## Task 1 -- OWNER: THE PRECISION POLICY. Confirmed by bit-identity.

The shipped-grid arm re-run at HEAD under fp64 is **BIT-IDENTICAL** to the
recorded `8e56daafa` baseline -- `max|d u3d|` at days 30 and 90, `max|d T3d|`
and `max|d eta3d|` at day 90 are all EXACTLY 0.0, and its ACC gap is +1.8723,
the recorded number.  Against the walk's fp32 arm the same fields differ by
2.28e-2, 2.65e-2, 1.63e-1 and 5.31e-3.

So:

* the 22-file / 2176-line model diff from `8e56daafa` to HEAD, PR #1628's
  barotropic C-grid and tidal-forcing edits included, is **EXACTLY INERT** on
  this card -- not "small", not "within noise": zero difference in every saved
  field.  No attribution to any #1628 edit is needed or warranted.
* the precision policy owns **100%** of the 0.213 Sv.  `kamm_twin_90d.py`
  never set a policy, so an unwrapped run silently built the whole oracle
  comparison at the fp32 control dtype.

There was never a physics discrepancy here.  CONFIRMED.

## Task 2 -- the dual gate, at 5x the noise floor, both arms at HEAD under fp64

| metric | floor | shipped 1-D ladder | x floor | NEMO/NEMO ladders | x floor |
|---|---|---|---|---|---|
| ACC full-section gap [Sv] | 0.091 | **+1.8723 FAIL** | 20.6x | **-0.5965 FAIL** | 6.6x |
| upper contrast <1400 m | 1.1e-4 | +4.00e-4 PASS | 3.6x | +2.92e-4 PASS | 2.7x |
| deep contrast >1400 m | 4.5e-5 | +6.12e-5 PASS | 1.4x | +1.78e-5 PASS | 0.4x |
| S-band surface sigma MAX | 9.5e-5 | +3.64e-4 PASS | 3.8x | +2.60e-4 PASS | 2.7x |
| S-band surface sigma MEAN | 9.5e-5 | +2.90e-4 PASS | 3.1x | +2.82e-4 PASS | 3.0x |
| | | PASS 4 / FAIL 1 | | PASS 4 / FAIL 1 | |

Channel band (`acc_thermal_wind`, rows 14..48, true partial-cell weighting,
mean over longitudes 2..-2), signed gap vs NEMO's own day-90 state:

| arm | baroclinic | barotropic | band total | median band | acc_full |
|---|---|---|---|---|---|
| shipped 1-D | +0.3530 | +2.5789 | **+2.9319** | +3.6117 | +1.8723 |
| NEMO/NEMO | -0.0187 | +0.3062 | **+0.2874** | +0.2446 | -0.5965 |

ACC is the lone FAIL on both arms, as predicted.  **No Rule-8 finding: all four
density metrics IMPROVE on the NEMO/NEMO arm**, none degrades, none crosses its
floor.  Gate exit status 1 on both (the ACC FAIL); the gate's own
synthetic-violation self-test passes (PASS 0 / FAIL 5) and both instrument
self-checks pass (NEMO y10 ACC 121.07 Sv; band volume 2.694775e16 m3).

## CORRECTIONS to the predictions above -- the walk's ladder table was fp32

Every number in the walk's four-arm table was measured at the fp32 control
dtype.  The two arms re-measured at fp64 both moved:

| quantity | walk (fp32) | this run (fp64) |
|---|---|---|
| `off` acc_full gap | +1.659 | **+1.8723** |
| `both` acc_full gap | -0.516 | **-0.5965** |
| `off` channel band | +3.483 | **+2.9319** |
| `both` channel band | +0.099 | **+0.2874** |

RETRACTED: "the NEMO/NEMO channel-band gap is +0.099 Sv, i.e. AT the 0.091 Sv
floor."  At fp64 it is +0.2874 Sv, **3.2x that floor** -- a real, resolvable
residual, not noise.  The 90% cut against the shipped ladder survives; the
"at floor" claim does not, and neither does any statement resting on it.

NOT RE-MEASURED: the `e3t_only` and `gdept_only` half-ladder arms exist only at
fp32.  Their split (thickness -> barotropic, depth -> baroclinic) is quoted in
the bridge note explicitly as fp32 and should be re-run before it is leaned on.

The ladder result is NOT refuted: the ranking, the sign, and the 90% cut all
survive fp64.  Task 3 proceeds.

## Standing hazard this exposed

`kamm_twin_90d.py` does not set or check a precision policy.  Every past and
future twin run is fp32 unless someone remembers `run_fp64.py`, and the only
signal is a bridge warning on stderr that scrolls past.  Two numbers recorded
weeks apart differed by 0.21 Sv for exactly this reason.  Making the twin
runner gate its own precision (`ocean.fidelity.precision_gate.require_fp64`)
is NOT done here -- it is out of this task's scope -- but it is the obvious
next fix and it is named so it is not lost.
