# ORCA2 round 49 receipt — OVERFLOW cancelling-pair boundary

Date: 2026-09-27  
Base: `ab8dd260e8ef198182e9e1d1962c19f3f95d75bf`  
Disposition: **STOPPED_FOR_RECORD**  
ORCA2 claim label: **given NEMO's entry**  
OVERFLOW claim label: **independent**

## Frozen question

Round 48's source-exact QCO tracer candidate made ORCA2's direct stage-1
statement bit-exact, but worsened five strict OVERFLOW U rows.  Round 49 asked
whether its first downstream movement enters the next stage through EOS/HPG,
as compiled: the QCO assignment is at
`OVERFLOW_OMIP_L1/BLD/ppsrc/nemo/stprk3_stg.f90:501-508`, and stages 2/3 call
EOS then HPG before vorticity and advection at
`OVERFLOW_OMIP_L1/BLD/ppsrc/nemo/stprk3_stg.f90:311-338`.  The selected SCO
HPG recurrence is the compiled
`OVERFLOW_OMIP_L1/BLD/ppsrc/nemo/dynhpg.f90:341-414`.

No configuration, threshold, carried state, selector, stabiliser, or sea-ice
field changed.  The ORCA2 card's `unmeasured_features` tuple is unchanged.

## Candidate reproduction

The preregistration was committed at `a96a954463` before measurement.  Commit
`3958b34bde` restored round 48's candidate exactly.  Its OVERFLOW trajectory
artifact has the same residual sidecar SHA-256 as round 48,
`db5dc334c0d63fa93b0a1ede5e130da023d5ee3ccbfc12f829cadf20fb3b3a7c`.
The oracle-relative comparison reproduced 50 scored rows, the unchanged first
over-bar set `{T,u}` at kt=2, and maximum worsening 57.8125 row-scale ULP.
Thus R49-P1 is **CONFIRMED**.

The candidate was then removed by `85d4bd9cbf`; the final tree has no model
diff from the base.

## Boundary instrument and control

The existing phase-3 stage sweep gained one narrow mode.  In one ordinary
compiled model step, private WRITE-only exposure returns stage-1 T/S/ssh and
the immediately following stage-2 HPG U/V only after production completes.
It stamps fp64/libm, CPU backend, commit/worktree state, oracle hashes, and a
hashed NPZ sidecar.  Its comparison uses `np.array_equal` and bitwise unequal
counts in frozen compiled order.

The focused synthetic control passed (`1 passed, 10 deselected`).  The real
+1 K tracer plant exited 2 and made only its intended T row DEBT at absolute
maximum 1.0 K.  The unplanted candidate and base both exited 0.

## Measurement

Candidate commit `a7e8741818` and restored-base commit `85d4bd9cbf` gave:

| boundary | cells compared | base/candidate unequal | max movement |
|---|---:|---:|---:|
| kt=1 stage-1 T | 60,600 | 0 | 0 |
| kt=1 stage-1 S | 60,600 | 0 | 0 |
| kt=1 stage-1 ssh | 606 | 0 | 0 |
| kt=1 stage-2 post-HPG U | 60,900 | 0 | 0 |

The two compressed sidecars are themselves byte-identical, SHA-256
`16bd7bfb0970750cc63765ad7e40f0ac320074721d90d82df0d3f4689db126f1`.
The comparison's `first_moved_boundary` is null.

Against NEMO at kt=1, both candidate and base have exact stage-1 T and S.
Stage-1 ssh is AT-BAR but not exact (6/200 active cells, maximum
1.3877787807814457e-17 m).  The calibrated independent SCO replay is AT-BAR
but not exact at the stage-2 HPG U boundary (164/16,900 active cells, maximum
2.1775137872720348e-16 m s-1).  This latter result is not promoted to a
bit-exact HPG claim.

Consequences for the frozen predictions:

- R49-P2 is **REFUTED at kt=1**: neither tracer nor the following HPG boundary
  moves.  The hook ordering is correct, but kt=1 is earlier than the held
  candidate's causal effect on this card.
- R49-P3 is **UNMEASURED**, not refuted: the first card-level movement occurs
  at kt=2, and no admitted kt=2 source-order stage stream exists.
- R49-P4 is **CONFIRMED**: the record has kt=1 stage endpoints, kt=1 RHS and
  transports, kt=1..10 step entries and barotropic frames, but no kt=2
  stage-local T/S/rhd or post-HPG U/V accumulator.
- R49-P5 is **CONFIRMED**: no unmeasured pair landed.

The first non-bit statement is therefore not named this round.  The evidence
only bounds it after the bit-identical kt=1 stage-2 HPG boundary and at or
before the already-observed kt=2 whole-step movement.  Assigning EOS, HPG, or
a later operator without the missing kt=2 record would violate the first-
statement rule.

## Record stop

The fail-closed acquisition contract is:

`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round49/acquisition/run.sh`

It names new config `OVERFLOW_OMIP_L1_P3_R49PAIR` and new run
`oracle_overflow_kt2_pair`.  It requests kt=2, stages 1..3, source-ordered
tracer accumulator boundaries plus EOS `rhd`, every momentum accumulator from
entry through HPG/vorticity/advection/LDF/ZDF, raw Kaa, and post-barotropic
Kaa.  Its preflight exits 66 with the named refusal
`committed additions-only writer patch is missing`; it creates no config or
run until both the writer patch and schema/admission gate exist.  This is an
acquisition specification, not a produced oracle record.

## Verification

- Candidate OVERFLOW trajectory: reproduced round 48 exactly; expected DEBT.
- Narrow boundary gate: candidate exit 0; base exit 0; real plant exit 2.
- Focused sidecar/hash test: PASS.
- Default citation gate: PASS, 274 citations, no unmapped citation or map-audit
  failure.  This receipt: PASS, 3 citations.  The shifted QCO citation plant
  exits 1 with one failure.
- Shared-card battery: 160 passed in 363.51 s; the separate tank-removal file:
  10 passed in 8.60 s.
- `tests/ocean/fidelity -n 12`: 1,931 items reached 99%, showing the five
  documented failures, then reproduced the known xdist controller stall and
  was interrupted.  The five IDs were rerun serially and retained the known
  signatures: round-129 stale certification, round-51 private trace registry,
  SI3 scalar-math provenance, the worktree-stamp ratchet (only its three
  pre-existing offenders after this round's comparison stamp was repaired),
  and the `hires_lane_surface` case-board ratchet.
- Separate read-only `codex exec` review: **independent review unavailable
  in-sandbox** (`failed to initialize in-process app-server client: Read-only
  file system`).

## OPEN

1. Commit the additions-only OVERFLOW kt=2 writer and its fail-closed parser,
   then complete and run the acquisition contract under the new target name.
2. Compare held candidate and base at kt=2 in NEMO's compiled order.  Name the
   first unequal statement; test it bit-exact on NEMO operands before forming
   any cancelling pair.
3. Keep the QCO candidate held until the full shared-statement landing gate
   passes.  No model change from this round may be carried forward.
4. Return afterward to the first whole ORCA2 non-bit checkpoint (kt=1 stage-1
   T), Decision-52 independent initial state/year, and round-20 slow forcing.
5. Sea ice remains out of scope at `STOP_SELECTOR_GAP`; its six selectors and
   `unmeasured_features` tuple remain frozen.

ASKED: continue the preregistered cancelling-pair walk and stop for a missing
record rather than infer.  
UNASKED: none.
