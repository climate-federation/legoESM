# ORCA2 round 43 — stage-1 vertical-transport handoff

Date: 2026-09-27  
Card: `orca2_vector_een_c2`  
Claim label: **given NEMO's entry** (Decision 52)

## Verdict

**HELD.**  The endpoint arm closes both horizontal stage-1 metric transports,
but not the vertical transport.  Its first non-bit boundary is `zFw`:
188,408 / 220,028 support-safe cells differ, with maximum
`1.862645149230957e-09`.  Replaying NEMO's source-stage WZV recurrence from
the admitted operands is bit-exact (0 / 228,641), and substituting the
recorded `zFw` makes both tracers bit-exact immediately after centered
advection.  That substitution exposes a later surface/runoff-source residual,
so the preregistered full-stage landing condition is not met.

No model, configuration, selector, carried-state rule, sea-ice field, or
stabiliser changed in this round.

## Preregistered predictions

| prediction | outcome |
|---|---|
| R43-P1 landed round-42 ladder reproduces | CONFIRMED: kt=1 stage-1 T is 233,341 / 399,600 at `0.0014770192519700243 K`; S is 233,341 / 399,600 at `0.0003277315524314872` |
| R43-P2 admitted schema and scorer bind; plant fires | CONFIRMED |
| R43-P3 endpoint arm closes every downstream row | **REFUTED**: `zFu` and `zFv` close, but `zFw` is the first nonzero row |
| R43-P4 production first differs at metric transport and endpoint closes downstream tracer boundaries | **REFUTED**: production first differs at `zFu`, but the endpoint arm leaves `zFw` non-bit |
| R43-P5 land only if one cited statement closes the full stage-1 T/S row and every required gate passes | CONFIRMED as a hold: exact `zFw` exposes a later source residual |

The failed predictions remain in the preregistration unchanged.

## Ordered boundary scores

All rows below are **given NEMO's entry**, on the record-backed, support-safe
rank-0 wet interior.  They were produced on CPU with fp64 and scalar libm.

| arm / ordered boundary | unequal / scored | maximum absolute difference |
|---|---:|---:|
| production `zFu` | 221,640 / 251,670 | `321205.47896199464` |
| production `zFv` | 222,048 / 252,330 | `293841.76375678886` |
| production `zFw` | 219,975 / 220,028 | `699293.0825276154` |
| production stage-1 T | 228,641 / 228,641 | `0.0011814592259422607` |
| production stage-1 S | 228,641 / 228,641 | `0.0003277315524314872` |
| endpoint `zFu` | 0 / 251,670 | 0 |
| endpoint `zFv` | 0 / 252,330 | 0 |
| endpoint `zFw` | 188,408 / 220,028 | `1.862645149230957e-09` |
| endpoint after-advection T | 164,350 / 228,641 | `1.4230153513872246e-19` |
| endpoint after-advection S | 132,204 / 228,641 | `1.8973538018496328e-19` |
| endpoint stage-1 T | 316 / 228,641 | `3.552713678800501e-15` |
| endpoint stage-1 S | 103 / 228,641 | `7.105427357601002e-15` |

The endpoint arm changes only the final sea surface and depth-integrated U/V
transports.  Thus it mechanically moves the first internal non-bit boundary
from horizontal metric transport to vertical transport; no downstream tracer
row is attributed across that boundary.

## Source-stage WZV discriminator

The record contains the materialized stage-1 WZV operands, including NEMO's
`Kbb` and `Kaa` thickness ratios.  Replaying the executing QCO recurrence with
NEMO's `dt/3` stage interval is exact: 0 / 228,641 cells differ.  Replacing
only the endpoint arm's non-bit `zFw` by the recorded field then gives:

| recorded-W boundary | T unequal / scored | S unequal / scored | maximum T / S |
|---|---:|---:|---:|
| after centered advection | 0 / 228,641 | 0 / 228,641 | 0 / 0 |
| after surface/runoff sources | 2,514 / 228,641 | 2,418 / 228,641 | `4.235164736271502e-22` / `1.6940658945086007e-21` |
| stage-1 output | 58 / 228,641 | 26 / 228,641 | `3.552713678800501e-15` / `7.105427357601002e-15` |

This distinguishes arithmetic from association: NEMO's WZV statement is
bit-exact when supplied the recorded source-stage pair, while the production
handoff supplies a different stage association.  It also proves that landing
only that association would not close the full stage-1 T/S row.

## Instrument correction retained

The first direct replay of the older phase-2l instrument omitted the binding
Decision-52 entry-SSH bridge and reported an all-cell difference near
`1.5e-09`.  That is an instrument mismatch, not a physics result.  The output
is retained as `phase2l_supplied_replay.json`; the round-43 gate repairs the
bridge explicitly and reproduces the round-42 ladder before scoring.  No
number from the unbridged replay is used as a fidelity claim.

## Controls, tests, and review

- The one-ULP stage-1 tracer plant exits nonzero with
  `REFUSE: planted stage-1 tracer cell rejected through scorer`.
- Focused round-43 plus citation-gate tests: 18 passed.
- The single `tests/ocean/fidelity -n 12` run dispatched 1,927 items and
  reached 99%, then repeated the known xdist-controller hang after its Python
  workers had exited; it was interrupted without a terminal summary.  Five
  red markers were rerun by exact node ID in one serial battery and remained
  red: the known SI3 scalar-math provenance and worktree-stamp ratchets, plus
  pre-existing GYRE round-129 stale-certified-gate, round-51 private-field
  registry, and recipe-case-board (`hires_lane_surface`) failures.  None reads
  a round-43 file or a changed model file; this incomplete battery is not
  called green.
- Separate read-only `codex exec` review: **independent review unavailable
  in-sandbox** (`failed to initialize in-process app-server client: Read-only
  file system`).

Because no `packages/` file changed, this held measurement round is not a GYRE
trajectory landing and does not invoke the shared-model non-regression gate.

## Evidence

Durable artifacts are under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round43/`.
The decisive files are `tracer_handoff_final.json`,
`tracer_handoff_final.log`, `orca2_ladder_kt1.json`,
`orca2_ladder_kt1.log`, `tracer_handoff_plant.log`, and
`codex_review.log`.

## OPEN

1. Continue in compiled order at the first downstream residual: split the
   stage-1 EMP/SFX/QNS and runoff source association on the admitted record.
   Only after that statement is exact may the source-stage WZV association
   and source residual land together under the complete ORCA2 and GYRE gates.
2. The first whole-card non-bit checkpoint remains kt=1 stage-1 temperature,
   `0.0014770192519700243 K`; this round localizes its first endpoint-arm
   transport handoff but does not close the checkpoint.
3. Sea ice remains exactly the card's six-item `unmeasured_features` tuple.

## Compiled-source citations

The executing HYB branch forms `zFu` and `zFv` from its stage-associated
barotropic correction at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3_stg.f90:265-284`.
It then calls the transport builder before centered advection and the surface
boundary condition at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3_stg.f90:551-645`.

For this vector-invariant card, the transport builder enables `pFw`, calls
`wzv`, multiplies its output by T-cell area, and records the live result at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/traadv.f90:267-315`.
The executing nonlinear QCO WZV branch first takes horizontal divergence,
then integrates bottom-up with the `Kaa-Kbb` thickness-ratio increment at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sshwzv.f90:271-299`.

The CEN2 consumer applies horizontal flux divergence at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/traadv_cen.f90:143-160`
and vertical flux divergence at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/traadv_cen.f90:202-227`.
The following stage-1 surface dilution and river-runoff source statements are
at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/trasbc.f90:275-328`.
