# NEMO testcase Lane 4 — ORCA2 card round 22 exact-transition receipt

Date: 2026-09-25

Parent: `1454e2b3f734c3e2d5a98197b54086e6dc13dbb3`

Status: **LANDED — ORCA2 MERGED LADDER RECERTIFIED.**  Given the independent
ORCA2 initial state with Decision-52's recorded NEMO SSH entry, all 185 ladder
rows moved by the GYRE-lane merge are now attributed.  Round 21's fold-layout
and bridge-carried F-thickness controls plus this round's live
lateral-diffusion-thickness control restore all 200 round-20 row summaries.
The fresh current-tip ladder exactly reproduces round 21's merged baseline and
is the lane's new certified reference.  Decision 58 is eligible next round.

No file under `packages/` changed.  The six sea-ice selectors and the card's
`unmeasured_features` tuple remain unchanged: `staged_gm_eiv`,
`linear_implicit_bottom_drag`, `internal_wave_mixing`,
`spatial_lateral_viscosity`, `freshwater_budget_carry`, and
`si3_jpl5_layered_prather_state`.

Every ladder and candidate number below is labelled **independent with
Decision-52 SSH**.  No given-NEMO-entry solver number is mixed into the table.

## 1. Compiled statement and first changed operation

NEMO calls stage 3, writes the stage dump, swaps the completed `Naa` level into
`Nbb`, and extrapolates only the spare SSH level at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/stprk3.f90:230-241`.
The admitted record confirms that stage 3 and kt=2 entry are bit-identical for
T, S, u, v and ssh.  There is no NEMO state-changing statement at the boundary
round 21 named.

The first changed legoESM operation is the merged routing of six live
thickness arrays into lateral momentum diffusion.  NEMO completes the stage-3
momentum RHS with `dyn_ldf` before the implicit vertical solve at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/stprk3_stg.f90:493-506`.
The executing operator uses Kbb T/U/V/F thicknesses to build curl and
divergence, then divides the update by Kmm U/V thicknesses at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynldf_lev.f90:123-140`.
Commit `94b7761bc7` began passing that literal six-array bundle into every RK3
call.  Before it, the same shared operator received no bundle and used its
algebraic thickness path.

The round-22 control withholds only that argument bundle.  Card, state,
operator, coefficient, masks, forcing, precision and every other call argument
stay current.  It executes on live calls, moves five exact returned-state
arrays, and together with round 21's two controls restores the pre-merge bits.
This is attribution, not a revert: Rule 12 keeps NEMO's faithful live-thickness
routing in production.

## 2. Exact transition and ladder result

The exact-array probe ran the archived clean pre-merge tree at `b03f78bb5` and
the current committed tree.  It saved entry, stages 1-3 and the separately
compiled ordinary returned state for every field.

| field in ordinary returned state | unequal cells, current combined vs pre-merge | maximum absolute movement |
|---|---:|---:|
| T | 16,684 / 799,200 | `1.7763568394002505e-14` degC |
| S | 5,474 / 799,200 | `3.552713678800501e-14` PSU |
| u | 328,801 / 799,200 | `4.1321113197767545e-14` m/s |
| v | 304,322 / 799,200 | `4.1907449732647706e-14` m/s |
| ssh | 12,584 / 26,640 | `7.771561172376096e-16` m |

R22-P2 is **REFUTED and retained**.  Entry and all three diagnostic stage
arrays are exactly equal between the combined and pre-merge arms; the first
exact movement is the separately compiled ordinary returned T array, followed
by S, u, v and ssh at the same boundary.  The model's existing stage observer
changes the compiled return graph enough to hide these last-bit movements.
Therefore stage-row summaries are observer-conditioned diagnostics, not a
bitwise proxy for the ordinary production return.  The production entry rows
remain the authoritative carried trajectory.

With the live lateral-diffusion bundle withheld, **0 of 25 exact candidate
arrays differ** from pre-merge.  The ten-step arm restores **200 of 200 row
summaries** exactly, including round 20's kt=10 entry-T maximum
`3.9430791763114783` degC.  A one-ULP plant changes exactly one cell and is
refused.

The fresh uncontrolled current-tip ladder is exactly equal to round 21's
merged baseline.  Its certified reference values are:

| checkpoint | current reference maximum error |
|---|---:|
| kt=1 stage-2 u | `0.06463349988292608` m/s |
| kt=1 stage-2 v | `0.03401471577804818` m/s |
| kt=10 entry T | `3.947126188631776` degC |

The first non-bit NEMO comparison remains kt=1 stage-1 T: 233,341 / 399,600
cells, maximum `0.0014770192519697467` degC.  The compiled runoff source is
still ruled out because 231,291 unequal cells lie outside every cell it can
reach.  No NEMO-fidelity statement changes in this round.

## 3. Frozen predictions

| ID | verdict | measurement |
|---|---|---|
| R22-P1 | **CONFIRMED** | Both exact captures reproduce their admitted scalar summaries, and the fresh current ladder equals round 21's merged baseline. |
| R22-P2 | **REFUTED** | No diagnostic stage array moves; the first exact movement is the ordinary returned T state. |
| R22-P3 | **CONFIRMED** | Withholding only live LDF thickness routing restores 25/25 exact arrays and 200/200 ladder rows. |
| R22-P4 | **CONFIRMED** | NEMO stage 3 and kt=2 entry are bit-identical in all five fields. |
| R22-P5 | **CONFIRMED** | The LDF control moves five arrays; the one-ULP plant changes exactly one cell. |

No prediction was rewritten after measurement.

## 4. Review and gates

The separate `codex exec --sandbox read-only` review was attempted at the
committed measurement tip.  It failed before reading the diff because the
in-process app-server client could not initialize on a read-only filesystem.
Verdict: **independent review unavailable in-sandbox**.

The final round-22 classifier exits 0 with `RECERTIFIED`.  It binds the four
exact snapshots, both admitted ladder documents, the restored ten-step ladder,
and the fresh current-tip ladder.  The receipt citation gate exits 0 with all
three compiled-source citations mapped and audited.  Its rigid-line plant
exits 1 and reports `SYMBOL-NOT-AT-LINE` for the planted stage-call anchor.

The final focused battery passes **11 / 11** tests.  The one required
`tests/ocean/fidelity -n 12` battery completes with **1,843 passed, 7 skipped,
5 failed** in 2,375.33 s.  All five failures are inherited and unrelated to
the round-22 diff: the round-129 spread-floor record stamp, the round-51 live
trace field suffix, SI3 scalar-math source provenance, the known three-emitter
worktree-stamp ratchet, and the unregistered `hires_lane_surface` case-board
row.  No round-22 test fails.

No GYRE trajectory rerun is required: the parent-to-tip `packages/` diff is
empty.  The certified GYRE reference remains ladder digest
`cf06a8fc7d0e90f2`, day-30 `6.572574374770603e-05` K, day-240
`1.644836070117868e-02` K, and day-360 `1.1225660018551306e-02` K.

Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round22/codex`.

## Choices

ASKED: instrument the exact stage-3 to kt=2-entry transition, name the first
changed operation, and reconcile the merged ladder before Decisions 58 and 54.

UNASKED: none.  No configuration value, state field, stabilizer, sea-ice
selector, scoring definition, or production statement changed.

## OPEN

1. Decision 58 is next: set the ORCA2 card's already-authorized second
   per-stage continuity solve to True and run the ten-step ladder; GYRE must
   remain byte-identical.
2. Decision 54 follows Decision 58: land the whole three-part `dyn_ldf`
   attribution under the required GYRE, DINO, lock-exchange and overflow gates.
3. Stage checkpoints are observer-conditioned below about `5e-14`; future
   last-bit production claims must use the ordinary returned state, while the
   stage observer remains useful above that measured floor.
4. Round 20's ranked slow-forcing producer walk remains open.
5. The northern-fold mask and wind-stress operands (668 / 35 cells) remain
   reported, not landed.
6. The independent ORCA2 year still depends on the scheduled independent
   initial-state completion.
7. The seven inherited duplicate citation-map literal keys remain open.
