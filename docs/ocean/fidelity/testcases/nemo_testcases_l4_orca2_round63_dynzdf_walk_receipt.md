# NEMO testcase L4 ORCA2 round 63 — `dyn_zdf` internal walk

Date: 2026-09-28  
Incoming tip: `6ac4ecd1b8d76eb35b48888b28892fdec78aa4e8`  
Frozen preregistration: `63996b04e0`  
Disposition: **HELD; private observer only, no physics lands**  
Evidence: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round63/`

## Outcome

Round 63 admits the operator-acquired round-62 record and closes the four
source-ordered `dyn_zdf` read-outs.  The numbers below are **independent**:
both legoESM arms start from the same round-60 kt=3 state; NEMO supplies oracle
frames, not the legoESM entry state.

The first measured statement is already non-bit.  Immediately after NEMO's
explicit Kaa update, base legoESM differs at 563 / 16,900 active U faces with
maximum absolute error `3.0650474349552814e-08` m/s.  The barotropic
subtraction and explicit drag each have 744 unequal U faces and maximum error
`3.063907276154576e-08`; the implicit solve retains 744 and
`3.0639067359616856e-08`.  This controlled OVERFLOW record has no active V
faces, so all four V read-outs are explicitly `UNMEASURED_NO_ACTIVE_FACE`.

The held source-ordered UP3 arm is **MIXED at the first moved boundary**, not
the preregistered strict TOWARD.  At explicit U it moves 209 faces: 91 toward
and 118 away; L2 changes `9.649682186799621e-08` to
`9.648831757914849e-08`, while L-infinity slightly worsens from
`3.0650474349552814e-08` to `3.065047435302226e-08`.  Later aggregate rows
move slightly toward, but MIXED does not name an owner.  R63-P3 is therefore
**REFUTED**, the candidate was removed, and no physics statement lands.

## Frozen predictions

| ID | verdict | evidence |
|---|---|---|
| R63-P1 | **CONFIRMED** | Round-62 admission is `AT_BAR`; same-build implicit U/V endpoints are array-equal to the parent raw-Kaa endpoints. Payload and stamp plants each exit 2. |
| R63-P2 | **CONFIRMED** | All 12 returned state arrays are bit-identical between ordinary and observed runs. The final ordinary digest equals the base digest `e3859619145d3de1489b140fc50983f080341f3905b0f59c231865e792f56b15`. The one-ULP active-U plant adds exactly one refusal, 563 -> 564, and exits 2. |
| R63-P3 | **REFUTED** | Explicit U is MIXED (91 toward / 118 away), not TOWARD; it is the first moved boundary. |
| R63-P4 | **CONFIRMED with clarified statement identity** | Same-build implicit endpoints equal raw Kaa. The older `pre_zdf_u` is Krhs, not explicit Kaa: 3,171 / 60,600 values differ, maximum `0.24980363380356374`; this failed provisional reconciliation is retained as two distinct compiled statements. Cross-build pre-ZDF and raw-Kaa comparisons are informational and happen to be exact. |
| R63-P5 | **CONFIRMED** | Final package physics equals the incoming tree; only the false-by-default observer remains. |
| R63-P6 | **CONFIRMED** | GYRE has 70 / 70 unmoved rows, array-equal residuals, and 30 / 30 byte-identical daily snapshots. |

## Source order and instrument

The acquired, executing source performs the explicit update at
`OVERFLOW_OMIP_L1_P3_R62ZDF/BLD/ppsrc/nemo/dynzdf.f90:140-159`, the
barotropic subtraction at
`OVERFLOW_OMIP_L1_P3_R62ZDF/BLD/ppsrc/nemo/dynzdf.f90:165-171`, the explicit
drag at `OVERFLOW_OMIP_L1_P3_R62ZDF/BLD/ppsrc/nemo/dynzdf.f90:172-192`, and
the U/V solves at
`OVERFLOW_OMIP_L1_P3_R62ZDF/BLD/ppsrc/nemo/dynzdf.f90:344-362` and
`OVERFLOW_OMIP_L1_P3_R62ZDF/BLD/ppsrc/nemo/dynzdf.f90:513-531`.  These are
live statements in the acquired test build.

The private hook and constructor guard are at
`ocean_model_latlon_cgrid.py:1305-1310` and
`ocean_model_latlon_cgrid.py:2832-2836`.  The production solve binds the hook
at `ocean_model_latlon_cgrid.py:11117-11121`, captures explicit,
barotropic-subtracted, and drag-complete operands at
`ocean_model_latlon_cgrid.py:11610-11613`,
`ocean_model_latlon_cgrid.py:11800-11803`, and
`ocean_model_latlon_cgrid.py:11919-11925`, then uses an ordered write-only
callback on the completed solve at `ocean_model_latlon_cgrid.py:12107-12114`.
No public card constructs this hook.

The gate requires clean commit identity, CPU, fp64/libm, production JIT, the
admitted producer stamp, and the ordinary/observer commit match at
`nemo_testcase_l1_overflow_round63_dynzdf_walk_gate.py:171-230`.  It checks
returned-state noninterference and scores the eight ordered frames at
`nemo_testcase_l1_overflow_round63_dynzdf_walk_gate.py:232-248`; its one-ULP
plant requires exactly one added refusal at
`nemo_testcase_l1_overflow_round63_dynzdf_walk_gate.py:280-292`.

## Shared-code gate

The final GYRE ten-step report compares PASS against the certified round-61
baseline: 70 rows, zero movement, no status changes, and first-over-bar still
kt=3.  Both residual archives have SHA-256
`43f37831256832c31f9a983352d8949e9947040faf01d5f4f39751dc1a70935c`.
All 30 daily snapshots are byte-identical; both day-30 files have SHA-256
`e2daff3f91ec3d806c109d1f7dff592ed705d236a6bfdb6a8c82614596fb3ec3`.

## Review and verification

The required separate read-only Codex review could not initialize a reviewer.
Its exact terminal result was:

> Error: failed to initialize in-process app-server client: Read-only file system (os error 30)

Therefore **independent review unavailable in-sandbox**.  The focused
round-60/61/62/63 and citation tests passed **39 / 39**.  The default citation
gate passed 274 citations and the round-63 gate passed 15 citations, each with
zero unmapped entries and zero failures.  A real +2-line plant on the observer
callback exited 1 with `SYMBOL-NOT-AT-LINE`.

The one permitted `tests/ocean/fidelity -n 12` battery collected 1,991 tests,
showed five failure marks and seven skips, reached 97%, and reproduced the
registered terminal zero-progress stall before a summary.  It was stopped
after a final observation window and was not rerun.  The focused round-63
tests are clean; the five broad-suite marks match the count already recorded
on the incoming lane and no round-63 test emitted a failure.

## Process record

The first full observer attempts exposed a per-process compiler-map limit and
were retained.  The committed gate therefore serializes ordinary and observer
compiles in separate clean processes.  A first reconciliation assumption that
round-50 `pre_zdf` and round-62 `explicit` named the same statement was
falsified and is retained above; the gate now requires only the valid
same-build implicit/raw-Kaa identity and labels cross-build checks information.

## Scope ledger and OPEN

**ASKED.** Admit and walk the four compiled `dyn_zdf` boundaries.  **UNASKED
and unchanged.** Configuration, carried state, forcing, stabilisers, public
state, ocean physics, and the six ORCA2 ice selectors.  Sea ice remains at
`STOP_SELECTOR_GAP`.

1. The first non-bit statement is now the explicit Kaa update.  Its inputs
   (Kbb, Krhs, `rDt`, mask) are the next source-order operand walk; do not retry
   the UP3 pair without a strict measured owner.
2. Decision 52's independent-start ORCA2 ladder is still owed after the
   step-level walk, followed by the month-scale ORCA2 magnitude ranking.
3. The ORCA2 card's exact-input ice gate remains `STOP_SELECTOR_GAP`; do not
   change `unmeasured_features`.

ASKED: close the four `dyn_zdf` internal read-outs.  
UNASKED: configuration, carried state, forcing, stabiliser, physics, and ice.
