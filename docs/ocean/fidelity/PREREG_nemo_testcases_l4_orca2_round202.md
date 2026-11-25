# ORCA2 round 202 preregistration — admit and score OMT-0

Date: 2026-10-09. Frozen base: `55fc7fe0b3`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round202/`.
All state comparisons are labelled **independent OMT-0** unless a row is
explicitly labelled **given NEMO's entry**. These labels are never combined in
one score table. The shipped rung-10 ORCA2 card, sea ice, its six selectors and
`unmeasured_features` remain unchanged.

Round 200's committed recovery contract and the operator's note that both
ten-step twins completed are prior evidence. This round freezes the admission,
card and scoring predicates before reading the record payloads or running a
legoESM trajectory.

## Frozen scope

OMT-0 is Decision 103's rung-0 card with exactly five modules disabled:
momentum advection, lateral momentum diffusion, tracer advection, lateral
tracer diffusion and bottom drag. It retains the ORCA2 mesh and partial cells,
EEN vorticity, split-explicit free surface, partial-cell pressure gradient,
constant vertical mixing and NEMO-replacement enhanced vertical diffusion.
No forcing, damping, restoring, ice or stabiliser is added. The OFF arms must
be selected through existing shared configuration fields; no OMT-only physics
branch is permitted.

The compiled NEMO selections remain the ones cited and gated in rounds 199 and
200: `dynadv.f90:162-190`, `ldfdyn.f90:177-228`, `traadv.f90:586-633`,
`ldftra.f90:214-268`, `zdfdrg.f90:371-401`, and the kt=11 safety predicate in
`stpctl.f90:176-184,243-250,293-316` under the admitted
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/` tree.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes / action |
|---|---|---|---|
| R202-P1 | The existing round-200 record satisfies its committed contract. | Two normal ten-step twins provide 40 rank-complete shards; all five fp64 finite fields are array-equal in 100 twin comparisons; the month step-10 state is array-equal in 20 comparisons; the unchanged month target exits through the exact kt=11 `stp_ctl` boundary with the frozen extrema and location. | Any missing/wrong shard, non-finite value, unequal twin/calibration field, clean continuation or changed stop: **REFUTED**; stop without a card score. |
| R202-P2 | Every admission control binds. | The eleven record plants named by the round-200 launcher each refuse and the clean record emits `PASS_R200_OMT0_TEN_STEP_RECORD__STOP_MONTH_AT_KT11`. | Any green plant or absent success token: instrument invalid; no record claim. |
| R202-P3 | The explicit OMT-0 legoESM card resolves exactly the five OFF modules and otherwise equals the corrected hierarchy rung-0 card. | A resolved-config diff contains only the five Decision-103 module selections; mesh arrays and initial active T/S/u/v/ssh are array-equal to the admitted rung-0 identity inputs. | Any extra resolved choice or geometry/entry mismatch: stop; no trajectory claim and `DECISION_NEEDED` for a physical choice. |
| R202-P4 | Both ten-step ladders execute in fp64 without a legoESM guard before kt=10. | Independent and given-entry artifacts each contain all registered kt=1..10 rows and finite active state through kt=10. | Earlier guard/non-finite: **REFUTED**; name the exact first boundary and do not score later rows. |
| R202-P5 | Removing the five downstream modules does not create an earlier source-ordered debt than the retained split-explicit external stage. | Independent entry is bit-exact; all earlier retained stage inputs stay at bar; first non-bit is the external-stage SSH or a later statement. | Any earlier non-bit boundary: **REFUTED**; that boundary owns the OMT-0 walk. |
| R202-P6 | The candidate remains finite through kt=10 and reaches its first safety/live-thickness boundary no earlier than NEMO's kt=11 boundary. | Ten-step candidate state is finite and a step-11 probe either takes a mechanically named existing guard or produces a finite state; the result is reported beside NEMO's kt=11 stop without calling the boundaries equivalent unless their predicates and state agree. | Candidate fails before kt=11: **REFUTED** and first boundary is named. Candidate continues: prediction partly confirmed through kt=10, with the differing kt=11 outcome registered. |
| R202-P7 | OMT-0 changes no shared production behavior. | No `packages/` diff is required, or any unavoidable shared change passes the full GYRE/DINO/tank and ORCA2 gates before landing. | Ungated model diff: stop and restore production. |
| R202-P8 | New instruments are non-vacuous. | Every new gate has a planted violation that changes one real, non-zero operand or selector and refuses. | Any green/no-op plant: no measurement from that gate is cited. |

## Landing predicate

This round may land pure card selection and measurement machinery after P1-P8
pass. A physics statement may land only if its changed operator is bit-exact
given NEMO operands on every executing card and Decision 96 passes: among rows
whose RMS score moves, a majority moves toward NEMO; the first-over-bar row
moves toward or stays; no exact row is lost beyond the floor; the SSH maximum
does not worsen; every touched and score-equal bit-moved row is registered.
Otherwise the statement remains private and the round is **HELD** with the
first non-bit boundary. OMT-1 does not begin in this round.

ASKED choices: Decision 103's OMT-0 definition, ten-step protocol and exact
kt=11 oracle stop handling.
UNASKED choices: empty.
