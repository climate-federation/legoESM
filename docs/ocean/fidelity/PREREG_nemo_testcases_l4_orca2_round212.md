# Preregistration — ORCA2 round 212 OMT-1 card and vector-branch boundary

Date: 2026-10-09. Frozen base: `68981c39b`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round212/`.
Every trajectory number is labelled either **independent OMT-1** or
**given NEMO's entry OMT-1**. This round changes no sea-ice selector, shipped
ORCA2 card, carried state, stabiliser, or unmeasured-feature declaration.

## Admitted record and source order

Round 211's operator record reports
`PASS_R211_OMT1_ENTRY_STAGE_RECORD__STOP_AT_KT9`: two ranks, eight steps,
four stages, 64 self-describing frames per twin, 320 twin field comparisons,
and four terminal-restart byte comparisons. The uninstrumented and both
instrumented executions complete kt=8; NEMO's own `stp_ctl` stops the preserved
boundary run at kt=9. Round 212 first re-runs that admission mechanically and
does not read a trajectory if it refuses.

The compiled OMT-1 selector executes vector KEG+ZAD+VOR at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynadv.f90:162-201`. Inside the
split-explicit loop NEMO constructs midpoint velocities and depths, transports,
continuity, surface pressure, Coriolis and then the direct vector U/V update at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:445-679`. The walk is
offline from the admitted passive streams in that source order. No executable
observer is permitted.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes / action |
|---|---|---|---|
| R212-P1 | The round-211 record remains admissible and passive. | Admission reproduces 64 frames/twin, 320 array-identical field comparisons, four byte-identical terminal comparisons, and the kt=9 boundary; a payload plant refuses. | Any changed hash, malformed/missing/non-finite field, twin bit, terminal byte, or boundary: **REFUTED**; stop without scoring. |
| R212-P2 | OMT-1 is exactly Decision 109's one-module edge. | The instantiated CPU/fp64/libm card restores vector-invariant KEG/ZAD/EEN while drag, momentum LDF, tracer advection and tracer LDF remain OFF; its selector plant refuses. | Any extra live module, silent default, or deck/card mismatch: **REFUTED**; stop. |
| R212-P3 | Both labelled kt=1..8 ladders complete through NEMO's admitted finite interval. | Each label produces 32 checkpoints / 160 rows, active entry is bit-exact, every value is finite, and the first non-bit checkpoint is kt=1 stage-1 SSH. | Earlier entry debt, a different first field, an incomplete row set, or a non-finite candidate: **REFUTED**; retain the measured first boundary and stop. |
| R212-P4 | The vector edge restores the rung-0-sized first SSH debt. | kt=1 stage-1 SSH maximum is of order `1.31e-1 m`, rather than OMT-0's post-landing `6.3e-3 m`; the exact value is measured, not assumed. | Any other magnitude: **REFUTED**; retain the measured value. |
| R212-P5 | The first internal numerical debt lies in the compiled vector-form split-explicit path. | Offline source-ordered replay names the earliest non-bit operand or statement before the kt=1 stage-1 SSH output and reproduces the candidate-side completed boundary within the registered floor. | Missing required oracle operand: **UNMEASURED_WITH_SPEC** and write one acquisition; replay non-closure: **HELD_INSTRUMENT**, no attribution. |
| R212-P6 | Every new gate binds. | Card-module, entry-bit, and any new replay plant each refuse. | Any plant stays green: no round-212 claim is citable. |

## Landing predicate

The explicit OMT-1 card may land when P1-P4 and their plants pass. A physics
statement lands only if its oracle operand is exact, it satisfies Decision 96
on OMT-1 and rung 0, and the full shared GYRE/DINO/tank/rung-10 gates pass.
Otherwise this round is **HELD** at the first named statement, or
**STOPPED_FOR_RECORD** with the exact missing stream. No configuration choice
is made in this round.

ASKED choices: Decisions 103 and 109. UNASKED choices: empty.
