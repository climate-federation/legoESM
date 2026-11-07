# NEMO testcases L2 GYRE round 80 — held-candidate record recovery receipt

Date: 2026-09-13

Starting commit: `ac501360f1212e58c3ec979a0b1c56604383818c`

Writable clone: `/tmp/autopilot-work-Qfw6Ujw3`

Evidence root: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round80`

## Verdict

**STOPPED_FOR_RECORD.** Round 79's absolute-history candidate remains HOLD.
The operator's acquisition did not run the requested parent ladder or review:
the script required a positional parent-clone argument, while the autonomous
operator invokes the named script without arguments. Round 80's own mandatory
read-only review again failed before model startup. No physics or carried-state
change was made, and the next source-order walk did not start.

No configuration choice was made. Decision 37 remains the user's explicit YES;
this round neither changes nor reinterprets it. NEMO source/build/run, the year
harness, card reconciliation gate, freshwater pair, #1484 guard, held
manifests, defaults, thresholds, masks, and scientific configuration were
untouched.

## Frozen preregistration

Round 80 was preregistered before attempting a review or new comparison at
commit `11dd6a59f09dd2b2aa40b0e25318b1c3ac9fcc4d`. It froze the exact parent and
after arms, the shared cellwise comparison, both plants, the literal Round 79
kt=2 U/V movement prediction, and STOPPED_FOR_RECORD when either prerequisite
could not be acquired.

The preimplementation search found the existing reusable instrument
`nemo_testcase_offline_compare.py`, which calls the same shared
`ulp_move_gate.compare_gate_reports` path as the production gates. No second
comparator or model toggle was written. The only new executable is the external
operator recovery contract at
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round80/run.sh`.

## Failed acquisition retained

The inherited operator log contains one decisive line:

> /data/abyssal/dbalwada/nemo-testcases-l2/phase3/round79/run.sh: line 4: 1: usage: run.sh PARENT_CLONE_AT_0078F9CC [SEALED_CLONE]

This is a shell argument-parsing failure before the first gate invocation. No
`round79_gyre_parent_0078f9cc.json` or parent residual sidecar exists in the
evidence root, and the available clean temporary clones were enumerated: none
is at `0078f9cc851c921176afd8e30660373129321712`. The result is not promoted,
imputed, or replaced with a scratch toggle.

Round 80's replacement `run.sh` takes no arguments. It creates its own
disposable shared clone at the exact parent commit, requires a clean exact
stamp, captures all ten steps and a hashed residual sidecar, compares that
parent against the already sealed Round 79 after pair, runs the shared
three-ULP and AT-BAR-to-DEBT plants, records whether the original kt=2 U/V
movement prediction is CONFIRMED or REFUTED, and only then invokes the review.
It performs no `makenemo`, `mpirun`, or NEMO-source mutation.

## Independent review

The required command was attempted after preregistration:

`codex exec --sandbox read-only -C /tmp/autopilot-work-Qfw6Ujw3 <Round 79 adversarial prompt>`

It exited 1 before reading the diff. Its terminal line, quoted verbatim, was:

> Error: failed to initialize in-process app-server client: Read-only file system (os error 30)

Therefore no reviewer `SHIP`, `HOLD`, or `DO NOT SHIP` verdict exists. This is
not treated as a waiver. The corrected operator script reruns the same required
review after the parent comparison exists and refuses a response that does not
end in one exact verdict.

A separate pass was then invoked on the committed Round 80 preregistration,
receipt, recovery script, and Rule 12 table with an explicit instruction to
refute the STOPPED_FOR_RECORD claim and plants. It also exited 1 before reading
the diff with the same terminal line quoted above. Thus Round 80 has no
adversarial verdict to quote; `round80_codex_round_review_attempt.txt` retains
the failed invocation rather than inventing one.

## First non-bit statement and magnitude rank

No new model measurement supersedes Round 79. Its admitted direct gate remains
the active statement register: kt=2 external substep 1 coefficients,
`un_e`, `ub_e`, `ubb_e`, and `ua_e` are bit-exact. The first non-bit row is
substep-2 `un_e`, 580/580 wet U faces, maximum absolute difference
`3.032539284029834e-09`. The next eligible magnitude-ranked walk begins after
the substep-1 midpoint and ends at the swap into that row, but cannot start
until this HOLD is mechanically disposed.

The Round 79 preregistration required kt=2 whole-step U/V to move at the bit
level. That prediction remains untested because the parent artifact is absent.
The corrected acquisition records it literally as CONFIRMED or REFUTED; it may
not silently relabel kt=3 as the originally predicted boundary.

## Rule 12 table

| lane | Round 80 disposition |
|---|---|
| GYRE source-order kt=2 substeps | PRESERVED: substep-1 midpoint exact; first non-bit substep-2 `un_e`; no new walk |
| GYRE kt=1..10 | UNMEASURED parent/after comparison: after arm sealed at `add5cbd5`; exact `0078f9cc` parent absent; candidate remains HOLD |
| GYRE days 1..30 | PRESERVED controlled result: Round 79 measured every scored field improved on all 30 days against the recorded before arm; no rerun or new claim |
| LOCK_EXCHANGE-zco | PRESERVED: Round 79's 50-row comparison passed; no changed statement this round |
| OVERFLOW-zps | PRESERVED non-execution: its boxcar card does not enter the changed cross-window AB3/AM4 continuation branch |
| DINO | SHARED-STATEMENT RISK: separate leapfrog carry means the changed continuation statement does not execute, but repository restart compatibility remains shared; no numerical neutrality claimed |
| ORCA2 | UNMEASURED-WITH-SPEC: independently aligned T/S/U/V/SSH and six absolute histories, native staggered masks, elementwise fp64 equality plus normalized L-infinity for kt=1..10; reject AT-BAR loss, earlier first-over-bar, or any wet-face history mismatch |

## Gates and focused checks

- `bash -n /data/abyssal/dbalwada/nemo-testcases-l2/phase3/round80/run.sh`:
  PASS.
- The run contract's unexpected-argument plant exited 64 before any clone or
  computation and printed `REFUSE: run.sh takes no arguments`.
- Receipt citation gate: PASS, 3/3 compiled citations mapped, zero map-audit or
  self-test failures.
- Shifting the compiled initialization citation by two lines exited 1 with
  `SYMBOL-NOT-AT-LINE`.
- Focused tests for the reused offline comparator, shared cellwise admission
  gate, and receipt citation gate: **38 passed in 1.91 s** under CPU/fp64.
- `git diff --check`: PASS.
- No model-physics suite was rerun because this round changes no production or
  test code. Round 79's 87-test result remains prior evidence, not a Round 80
  test claim.

## Choices and uncertainty

Choices made: none. The parent commit, after arm, bar, fields, steps, CPU/fp64
policy, and carried-state authorization were inherited verbatim. There is no
mechanical gate for the no-unasked-choice claim; this paragraph is the required
honour-system disclosure.

UNVERIFIED: the parent/after cellwise result; the original kt=2 movement
prediction; and the independent reviewer verdict. Each is deliberately absent,
not inferred.

## OPEN — exact handoff to round 81

1. Operator runs
   `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round80/run.sh` with no
   arguments. Success ends with `ROUND80_PARENT_AND_REVIEW_READY` and records
   the exact verdict and retained disposable parent-clone path.
2. Next round reads `round80_parent_after_comparison.json`,
   `round80_kt2_prediction_disposition.json`, and
   `round80_codex_review.txt`. It quotes the review verdict verbatim and
   dispositions every named finding. A `DO NOT SHIP` verdict forbids the
   candidate; `HOLD` keeps its correction open.
3. If the comparison passes and review says SHIP, promote the paired Decision
   37 candidate while retaining any failed kt=2 movement prediction as
   REFUTED. Do not resurrect the Round 51 raw-history-only manifest.
4. Only after promotion, preregister the next source walk from the exact
   substep-1 midpoint through pressure, Coriolis/drag/forcing, velocity update,
   and swap into substep-2 `un_e`. Every fix must come from the first non-bit
   compiled statement, not from the downstream day-30 field.
5. Keep DINO risk explicit and ORCA2 UNMEASURED-with-spec until their independent
   trajectory records exist.

## Compiled source citations

The running GYRE branch initializes absolute histories only for a cold start,
loads the current external velocity from the selected baroclinic level, and
zeros after accumulators separately:
`GYRE_OMIP_L2_P3_SM_R77UAMID5/BLD/ppsrc/nemo/dynspg_ts.f90:339-378`.

Its midpoint consumes the absolute current, before, and twice-before velocity
arrays in the written association:
`GYRE_OMIP_L2_P3_SM_R77UAMID5/BLD/ppsrc/nemo/dynspg_ts.f90:481-509`.

It rotates those absolute histories and assigns the next substep current value
after every external substep:
`GYRE_OMIP_L2_P3_SM_R77UAMID5/BLD/ppsrc/nemo/dynspg_ts.f90:783-795`.
