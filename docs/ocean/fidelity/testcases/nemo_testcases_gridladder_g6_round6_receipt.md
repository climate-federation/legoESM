# Grid ladder rung G6 — round 6 receipt (gridladder_r42)

Status: **LANDED (Decision 132, G6-selected); G6 NOT CERTIFIED.**  The
kt 2..6 whole-domain debt (sea level 4.461e-3 m at kt 2) is gone.  Under
GIVEN-NEMO-ENTRY every field is at the bar kt 1..6 on level 1 and the 2-D
fields; one all-level row (kt-4 two-step, partial-cell u/v, 1.86e-15) is
just over.  INDEPENDENT is at the bar kt 1..2 and over it from kt 3
(worst 6.8e-14, v at kt 5-6).  The residual is a last-place difference
that starts INSIDE `stp_2D` at kt 2; NEMO records nothing there after
kt 1, so the walk stops at `stp_2D`'s exit.  ACQUISITION_NEEDED.

Preregistration: `PREREG_nemo_testcases_gridladder_g6_round6.md` (commit
`b1fa72b09`, before any run).  Record: round 36
`gridladder_rounds/round36/oracle_g6_r36/` (unchanged).  Evidence:
`gridladder_rounds/round42/`.  Runs stamped `40c9af16cf51` (clean tree).
SHA-256 prefixes: independent kt 1-3 `0512bbd66804d089`, kt 4-6
`ee7e154ee0b8d55d`; given-entry kt 3-6 `a5c6cb78f99040e8`; stages kt 3
(given) `d899888aeb430eb4`; probe `probe_r1_ht0.py` `cb216ccae8608e79`
(json `059392c134723f99`).

## 1. Landed (`40c9af16c`)

The G6 card selects `nemo_first_wzv_after_ssh='rk3_extrapolated_carried'`
(`nemo_testcase_recipe.py`, G6 branch only).  The shared model already writes
the slot at step end and reads it at the first `wzv` (the GYRE/VORTEX
vector cards use it); no model code changed.

| NEMO statement | G5 r31 build | ORCA2_OMIP_L4 | SUBROUTINE |
|---|---|---|---|
| write `ssh(:,:,Naa) = 2*ssh(:,:,Nbb) - ssh(:,:,Naa)` | `stprk3.f90:241` | `:237` | `stp_RK3` :62 |
| read `r3t(:,:,Kaa) = ssh(:,:,Kaa)*r1_ht_0`, then `CALL wzv` | `stp2d.f90:152`, `:156` | same file | `stp_2D` |
| kt-1 fill from rest (`ln_rstart=.false.`): `ssh(:,:,Kaa) = ssh(:,:,Kbb)` | `restart.f90:406` | `:405` | `rst_read_ssh` :321 |
| restart entry: `iom_get(...,'ssha', ssh(:,:,Kaa))` | `restart.f90:367` | `:367` | `rst_read_ssh` :321 |
| restart write `'ssha'` (after the :241 extrapolation) | `restart.f90:184` | `:184` | `rst_write` :153 |

- kt 1: the card's unseeded slot is the step-entry height, which is NEMO's
  from-rest fill (`:406`).  kt 1 is identical to round 40 (below).
- GIVEN-NEMO-ENTRY: the G2/G6 ladder's NEMO entry now seeds the slot from the
  restart's `ssha`, as `rst_read_ssh` does; the ladder refuses unless only G6
  carries it.  Round 36's restarts hold `ssha` (checked at kt 2, 4, 6).
- Caller grep: `nemo_first_wzv_after_ssh=` in the card file — GYRE and VORTEX
  `rk3_extrapolated_carried` (unchanged), `orca2_vector_een_c2`
  `rk3_extrapolated` (unchanged, so the ORCA2 card and G1b keep it), the new
  G6 branch.  Generic default `""` unchanged.  No dynamics file changed, so
  every other card's step is the same code on the same config; the
  card-equality and opt-in-default tests pass unchanged.
- Test: `test_g6_card_is_g5_plus_g1b_vector_selectors` now pins the field.
  FAILS with the selection reverted to `rk3_extrapolated`
  (`'rk3_extrapolated' != 'rk3_extrapolated_carried'`); restored,
  `git status --porcelain` showed only the round's own edits.

## 2. The round-39 scratch figure on the committed card

| kt 2 | round 40 (card) | round-39 scratch (at `5f471f290`) | committed card |
|---|---:|---:|---:|
| ssh | 4.461e-03 / 16433 | 4.394e-08 / 2396 | **1.110e-15 / 1369** |
| u | 8.087e-03 / 15853 | 8.355e-07 / 2323 | 2.220e-16 / 1334 |
| T | 1.044e-03 / 16433 | 5.270e-10 / 946 | 4.441e-16 / 1 |

P2 is REFUTED in the good direction: the committed card is 4e7 times closer
than the scratch arm, not "about 4.394e-8".  The scratch ran before round
40's fold-row landing, whose kt-1 fold residual (u 4.6e-9 on 135 cells) the
scratch carried into kt 2.  PLAUSIBLE (the scratch was not re-run on HEAD
minus round 40).

## 3. Both labels, kt 1..6 (level 1 + 2-D fields; `max_abs / unequal`)

| kt | label | ssh | uu_b | vv_b | u | v | T | S |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | indep. = given | 0/0 | 0/0 | 0/0 | 0/0 | 1.39e-17/3 | 0/0 | 0/0 |
| 2 | indep. = given two-step | 1.11e-15/1369 | 6.2e-17/1574 | 6.9e-17/1717 | 2.2e-16/1334 | 2.2e-16/1364 | 4.4e-16/1 | 7.1e-15/1 |
| 3 | independent | **6.55e-15**/8728 | 1.7e-16/9760 | 2.2e-16/9654 | **6.22e-15**/6874 | **8.42e-15**/7117 | 7.1e-15/4 | 7.1e-15/4 |
| 4 | independent | **3.99e-14**/13426 | **6.6e-15**/14210 | **4.2e-15**/13988 | **1.72e-14**/10948 | **6.54e-14**/11269 | 3.6e-15/9 | 7.1e-15/6 |
| 5 | independent | **3.55e-15**/14065 | **6.5e-15**/14776 | **4.2e-15**/14779 | **1.97e-14**/12375 | **6.69e-14**/12683 | 3.6e-15/17 | 7.1e-15/22 |
| 6 | independent | **4.24e-14**/14355 | **6.2e-15**/14938 | **4.2e-15**/15072 | **2.43e-14**/12928 | **6.80e-14**/13268 | 7.1e-15/24 | 7.1e-15/24 |
| 3 | given (restart 2) | 6.7e-16/1666 | 1.1e-16/1997 | 8.3e-17/1980 | 2.9e-15/1636 | 2.8e-15/1494 | 2.2e-16/1 | 7.1e-15/1 |
| 4 | given two-step | 1.3e-15/10003 | 4.4e-16/10991 | 2.5e-16/10694 | 2.9e-15/7951 | 2.7e-15/8129 | 3.6e-15/6 | 7.1e-15/6 |
| 5 | given (restart 4) | 8.9e-16/2080 | 1.1e-16/2282 | 1.1e-16/2234 | 6.7e-16/1638 | 8.9e-16/1680 | 2.2e-16/1 | 7.1e-15/1 |
| 6 | given two-step | 1.6e-15/10321 | 2.2e-16/11264 | 4.4e-16/11334 | 2.8e-15/7984 | 2.6e-15/8365 | 7.1e-15/5 | 7.1e-15/5 |

Absolute differences; the scorer's verdict uses the normalised error.  Bold =
DEBT (normalised > 1e-15); everything else AT-BAR.  Every given-entry row is
AT-BAR.  All-level rows (normalised, full / partial cells): given kt-4
two-step u 7.3e-16 / **1.86e-15**, v 5.2e-16 / **1.38e-15** (DEBT); given
kt 6 AT-BAR; independent kt 4 and 6 u, v DEBT (≤2.7e-14); T, S AT-BAR
everywhere.  Flushed subnormals (Decision 127): 0 in every run.  The kt-1
and kt-2 given rows equal the independent rows: NEMO restarts exist only at
even steps, so both start from the card's initial state.

**Certification: NOT certified.**  Independent is over the bar from kt 3;
given-entry is over it at one all-level row.

## 4. Walk of the kt-2 residual

| boundary (kt 3, from NEMO's restart 2 incl. `ssha`) | result |
|---|---|
| step entry | NEMO's restart, exact by construction |
| stage 1: uu_b / vv_b (`stp_2D` exit) | 1997 / 1980 cells, 1.1e-16 |
| stage 1: ssh / u / v | 847 / 872 / 814 cells, ≤2.2e-16 |
| stage 2 | ssh 1080, u 1290, v 1297 cells, ≤4.4e-16 |

The cells are scattered interior (fold 0, south 0, seam 49 of 1997),
one-ulp-scale.  **The first non-bit boundary NEMO records is `stp_2D`'s
exit (`stp2d.f90`, SUBROUTINE `stp_2D`; frame written by
`l1_dump_bt_frames`, `stprk3.f90:206`).**  The statement inside it is not
named.

Why it first shows at kt 2: G6 starts from rest, so at kt 1 every
`stp_2D` momentum-advection operand at Kbb (velocity, `r3t/r3u/r3v(Kbb)`,
`ww`'s sea-level-change term) was zero.  Candidates in compiled order
(`stp2d.f90`): `dyn_hpg`, `dyn_vor`, the first `wzv` (:156), `dyn_keg`,
`dyn_zad`, `Ue_rhs` depth sum, drag/wind, the solver.  Ruled out by
measurement or reading:

- `r1_ht_0`: NEMO `ssmask/(ht_0+1-ssmask)` (`domain.f90:212`) vs the
  `wzv` builder's `1/ht_0`: 0 of 16 433 wet columns differ (probe above).
- `wzv` qco recurrence: the card's form is NEMO's written association
  (`sshwzv.f90:297-298`): `(r1_Dt*e3t_3d)*(r3a-r3b)`, then
  `pww(k+1) - (ze3div + .)*tmask`.  Read, not measured.
- `r3u/r3v(Kbb)`: NEMO carries the previous stage-3 `dom_qco_r3c_RK3`
  ratio (`stprk3_stg.f90:242-244`; `domqco.f90:266-269`); the card's builder
  has the same operation order.  Read, not measured.

NEMO writes no `stp_2D` operand after kt 1: the slow-forcing dump
(`stp2d.F90` MY_SRC, guards at lines 181, 210, 222, 226, 234) is gated `kt == nit000`, and the
stage-2 operator dumps are kt 1 only.  So no transcription defect was
landed this round.

## 5. Acquisition

`scripts/validate/ocean_fidelity/testcases/nemo_testcase_g6_orca2vec/run_r42_stp2d.sh`
(commit `5b0fa6506`): builds `ORCA2G6_OMIP_L4_R42STP2D` from G6's record
config (G5 r31 FIX1, binary pinned `41cd9a63…`) with ONE patched source,
`MY_SRC/stp2d.F90`.  The patch is write-only: the 3-D right-hand side after
HPG, VOR, KEG and ZAD; the first `wzv`'s `ww` with `ssh/r3t(Kaa,Kbb)` and
`r3u/r3v(Kbb)` (`oracle_stp2d_terms_kt*.bin`, kt 1-3); and the existing
slow-forcing dump widened to `kt <= nit000 + 2` (five write-only guards).
Gates in the script: compiled `stp2d.f90` differs from FIX1's only by added
lines plus those five guards; every other ppsrc file byte-identical; FIX1's
record files kt 1-4 byte-identical to round 36's twin_a; restarts 2 and 4
equal round 36's writer-free reference in every variable; term-dump payload
size and finiteness.  Dry-tested here: the patch applied to a copy of
`stp2d.F90` (anchors unique, 5 guards) and `bash -n`.  Not built, not run.

Next round: replay each `stp_2D` term at kt 2 from those records (kt 1 is the
calibration, since its terms except HPG are zero), land the first non-bit
statement G6-selected, re-score kt 1..6.

## D122 pointer (ORCA2 lane)

The selected statements run on ORCA2: `stprk3.f90:237`, `stp2d.f90:152/156`,
`restart.f90:367/405` in `ORCA2_OMIP_L4/BLD/ppsrc/nemo/`.  Import waits for
G6 certification.

## Tests and review

- G6 card + G2 ladder + G2-G7 card tests: 102 passed (CPU fp64, 8 workers).
- G1 card, G1b card, G1 tools, after-SSH carry, opt-in defaults, card
  constructibility: 162 passed.  G1 ladder: 86 passed (16 workers).
- Receipt-citation gate + private-import ratchet: 23 passed, after a
  mechanical re-anchor of three gate pins (`5c0b23703`: recipe 4126-4131 ->
  4132-4137, 3971 -> 3977; ladder 655-663 -> 664-672).
- codex review: paused (quota).  The operator runs an independent Claude
  review of this diff before landing.

ASKED: Decision 132 on the G6 card with the carried field initialised as
NEMO does and loaded on the NEMO-entry label; a revert-failing test; the
scratch figure re-measured; the kt-2 walk; both labels kt 1..6.

UNASKED, offered for revert: the gate re-anchor (mechanical); the
acquisition script (needed to continue the walk); the ladder docstring line.

UNVERIFIED: the 4e7 gap between the scratch arm and the committed card
is attributed to round 40, not measured; the two "read, not measured"
items in §4; the acquisition has not run.
