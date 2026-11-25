# ORCA2 round 304 — G5 certified-statement import under Decision 122

Date: 2026-10-17

Status: **LANDED** (two of three G5 statements; the ice-ocean top drag stays off)

Preregistration `36400d0ac` (frozen before measurement). Landing commit
`18e4631cb` (measured clean). Citation re-anchor `80da64e3f`.
Evidence: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round304/`.
Codex review: paused (quota). Independent review: operator (Claude).

## What landed, and what stayed off

| G5 statement | (a) G5 certification | (b) ORCA2_OMIP_L4 line, SUBROUTINE, switches | (c)+(d) on 7 pairs | verdict |
|---|---|---|---|---|
| B. stage-1/2 Kmm divisor + EMP dilution, with stage-1 `r3t(Kaa)` | `nemo_testcases_gridladder_g5_round4` §3 item 1 and §5 rows 1, 2, 5 | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/trasbc.f90:284-286` (`tra_sbc_RK3`, `:220`); `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:177` (`stp_RK3_stg`, `:78`; `n_baro_upd = np_HYB` at `:49`); runs because the build defines no linear-free-surface key (`cfgs/ORCA2_OMIP_L4/cpp_ORCA2_OMIP_L4.fcm:1`), so `lk_linssh = .FALSE.` (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/dom_oce.f90:137`) and the stage-1/2 guard `ORCA2_OMIP_L4/BLD/ppsrc/nemo/trasbc.f90:282` is true. No namelist switch is involved | pass | **ON** (all cards) |
| C. stage-3 salt flux `sfx*z1_rho0_e3t` | `nemo_testcases_gridladder_g5_round4` §3 item 2 and §5 row 3 | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/trasbc.f90:308` (`tra_sbc_RK3`, the `ELSE` of `:292`, taken because `lk_linssh = .FALSE.`); `sfx` comes from SI3 (`nn_ice = 2`, `ORCA2_OMIP_L4/EXP00/namelist_cfg:85`) | pass | **ON** (rung 10 only) |
| A. implicit ice-ocean top drag (`rCdU_top`, residual before wind, substep, `dyn_zdf`, barotropic) | `nemo_testcases_gridladder_g5_round3` "Decision 122" table; certified at round 4 | same lines run (G5 round 3 table) | not measured | **OFF** |

Why A stays off: ORCA2 runs linear bottom drag (`ln_lin = .true.`,
`ORCA2_OMIP_L4/EXP00/namelist_cfg:251`; `ln_drgimp = .true.`,
`ORCA2_OMIP_L4/EXP00/namelist_ref:817`), so NEMO's U/V coefficient sums a
nonzero bottom pair with the top pair in one statement. G5 runs with bottom
drag off and certified only the top pair. Its code refuses a card with bottom
drag in the matrix or substeps, and the shipped ORCA2 card has both. The
statement ORCA2 executes is therefore not the certified one. A was not ported.

The ORCA2 card takes `sfx` the same way G5 does: from NEMO's recorded post-SI3
surface frame (`sfx` and `rCdU_ice` are both in that record). C reads the raw
recorded `sfx` through the new optional forcing field `salt_flux_sfx`. Rung 0
and OMT-1..4 have no surface forcing, so C is scoped off there, as round 285
did for the stage-3 heat unit.

The package hunks are those of gridladder commits `6c1cebfd7` and `62060a46c`.
They are imported, not re-implemented, and both default to False.

## Decision-122 census (B+C as one set; same-tree control = both off)

Counts are RMS toward / away / score-equal. They are registered, not a veto.
Result: `PASS_R278_D122_DECISION` (`decision.json` sha256 ab1a56e1dedd3a0f).

| card | label | toward / away / equal | early kt 1-3 | exact rows lost | first over-bar row | kt-10 stage-3 ssh max before -> after (m) |
|---|---|---:|---:|---:|---|---|
| rung 0 | independent | 0 / 0 / 0 | 0 / 0 / 0 | 0 | kt1 stage1 T, equal | 0.38835166609370336, identical |
| OMT-1 | independent | 0 / 0 / 0 | 0 / 0 / 0 | 0 | kt1 stage1 T, equal | 0.7471516624784271 (kt 8), identical |
| OMT-2 | independent | 0 / 0 / 0 | 0 / 0 / 0 | 0 | kt1 stage1 T, equal | 0.34887168162722165, identical |
| OMT-3 | independent | 0 / 0 / 0 | 0 / 0 / 0 | 0 | kt1 stage1 T, equal | 0.363396286656436, identical |
| OMT-4 | independent | 0 / 0 / 0 | 0 / 0 / 0 | 0 | kt1 stage1 T, equal | 0.3941246940211905, identical |
| rung 10 | independent | 1 / 0 / 1 | 0 / 0 / 0 | 0 | kt1 entry ssh, equal | 0.2933479046003188, identical |
| rung 10 | given NEMO's entry | 75 / 25 / 14 | 2 / 1 / 5 | 0 | kt1 stage1 T, equal | 0.2855333068505322 -> 0.2855333068505171 |

Rows touched: rung 10 independent 2/200, given NEMO's entry 114/200, all others 0.

Predictions:
- R304-P1 **CONFIRMED**: no rows moved on rung 0 or OMT-1..4.
- R304-P2 **CONFIRMED**: no exact rows lost, and the first over-bar row is equal on both rung-10 labels.
- R304-P3 **CONFIRMED at summary level**: on the given-entry label the kt-1
  stage-3 S max error is unchanged (0.059707391246320185) and its RMS moves in
  the 17th digit (3.015410766732258e-4 -> 3.0154107667322577e-4). The card's
  stage-3 salt source therefore held no term besides `sfx` and runoff. This is
  not a field-level diff.
- R304-P4 **CONFIRMED**: GYRE is byte-identical.

B and C were measured as one set, as G5 certified them. C acts only on
rung 10 (the only card with surface forcing), and there B and C were measured
together and never apart: C's own effect is unmeasured on every card.

Plants: an exact-row loss exits 2 with `STATUS PLANT-FIRED: previously exact
rows left the bar: [(1, 'entry', 'T')]`. A control that keeps a selector on
exits 2 with `STATUS PLANT-FIRED: rung10: control retains a G5 selector`.

## Shared-card and software gates

- GYRE: base `89420312e` and tip `18e4631cb` ten-step residual archives are
  byte-identical (sha256 3b3d865cf5c82d38…). The 30 day-30 daily snapshots are
  30/30 byte-identical. The gate script exits 1 on both base and tip; that exit
  predates this round (round 302's base also exits 1).
- Citation gate + private-import ratchet + the round-304/285 import tests:
  28 passed. The gate run on THIS receipt first failed (six cited lines had
  no map entry); after the review the entries were added and the gate passes
  on this receipt (see the answer commit). One mechanical re-anchor was needed (`reanchor_default_receipt.py`
  in the evidence directory). One span grew 174 -> 199 lines; its pinned
  endpoint statements are unchanged, the same growth G5 round 4 recorded.
- Rest of the push battery (TKE, recipe, freshwater, stage face mask,
  prognostic barotropic, ORCA2 parallel receipt gate): 121 passed in 584.18 s.

## Choices

ASKED (in the round prompt): import B and C; leave A off because ORCA2
composes it with bottom drag differently; no change to the after-SSH selection
(D132), runoff depth (D126) or the private fold unit.

UNASKED, offered for revert:
1. B stays live on rung 0 and OMT-1..4 (round 285's scoping pattern). It moved
   zero rows there.
2. The rung-10 forcing builder now also passes the raw recorded `sfx`. It is
   inert unless C is on.

## OPEN

1. Next round, not this one: ORCA2 runs G5's ice-ocean top drag (A)
   TOGETHER with ORCA2's linear bottom drag. The ladder rung G7 (G5 plus
   ORCA2's linear bottom drag, in byte-identical compiled files) was certified
   on 2026-10-16 (gridladder receipt `nemo_testcases_gridladder_g7_round1`;
   brief addendum 100). Importing A together with G7's two bottom-drag
   statements (bottom residual and bottom coefficient) under D122 is the next
   round's work.
2. Unchanged: promote fold unit + transport mask + top-V atomically (round
   303); D126 and D132 pending unchanged.
