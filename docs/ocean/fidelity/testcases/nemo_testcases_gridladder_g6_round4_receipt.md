# Grid ladder rung G6 — round 4 receipt (gridladder_r39)

Status: **LANDED (kt-1 ZAD pivot V face); G6 NOT CERTIFIED.**  The kt-1
stage-2 KEG+ZAD boundary is split: **KEG is value-exact at every wet face
and level; ZAD owns the whole remaining difference.**  ZAD's first non-bit
operand on the V pivot row (the card's V face mask, dry there) is landed.
The kt-2 whole-domain debt is a NEW statement at kt 2 stage 1, not carried
from the fold rows; its owner is measured (scratch) and needs a user
decision because the fix adds carried model state.

Preregistration: `PREREG_nemo_testcases_gridladder_g6_round4.md` (commit
`5f471f290`, before any comparison).  Record: round 36
`gridladder_rounds/round36/oracle_g6_r36/` (unchanged).  Evidence:
`gridladder_rounds/round39/`.  Final runs stamped `24690e4329` (clean tree).
SHA-256 prefixes: independent `8845cbf9f722c399`, given-entry
`bea174338c4f3406`, operators kt 1 `3e4e3b6b9a7445f4`, KEG split
`d91be205ef83ffae` (probe `probe_keg_split.py` `b444285b09415369`), ZAD
operand swaps `b618a969de16d3f2` (`probe_zad_operands.py`
`6e5191307624fd05`), w inversion `a1f8ac9cf35dd7d6` (`probe_invert_w.py`
`149fdea9a6c48ad0`), kt-2 stages `05e2055d0b7667f6`, kt-2 scratch arm
`32e763ae0b4fffa5` (`scratch_carried_after_ssh.py` `9bb1e7ef9dfa0aaa`).
Card captures (write-only hooks / passive debug callbacks):
`capture_keg_zad.py`, `capture_zad_ops.py`, `capture_wzv_rec.py`.

## 1. KEG vs ZAD at kt 1, stage 2 (instrument and split)

NEMO records only the combined accumulator (`after_advection`).  The split
is an OFFLINE literal numpy transcription of `dyn_keg` nkeg_C2
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynkeg.f90:61`, block :117-131, `zhke` :125,
pivot V :130; deck `nn_dynkeg=0`) applied to NEMO's recorded stage-2
`uu/vv(Kmm)` on top of NEMO's recorded `after_vorticity`.  The record writer
(`l4_canon_3d`) zeroes halos and dry cells, so the halos `dyn_keg` reads are
rebuilt with NEMO's own `lbc_lnk`: E-W cyclic, then `lbc_nfd_dp` T-pivot
'U'/'V', sign -1 (`lbcnfd.f90:561`).  The first replay without that halo
rebuild failed calibration (38 456 cells) and was not quoted.

| check | result |
|---|---|
| calibration: replayed NEMO after-KEG + card ZAD increment vs NEMO `after_advection`, where the card already equals NEMO (797 593 U / 796 009 V cells) | 0 / 0 unequal |
| **card after-KEG vs NEMO after-KEG, wet cells, all levels** | **U 0, V 0 unequal** |
| card after-ZAD vs NEMO (before landing) | U 1607 / V 3191 all levels (84 / 195 surface) |

Card after-ZAD differences by row (all levels): U pivot T row 1292, V row
below the pivot 1272, V pivot row 1319, plus a scattered interior set
(U 314 / V 600 cells, max 8.5e-22 absolute, ≤2e-11 of the increment).

## 2. ZAD operands (`dyn_zad`, `dynzad.f90:59`, loop :105-127, V update :125-126)

The card's stage-2 ZAD kernel operands were captured; the replay reproduces
the captured output exactly (U and V).  One operand swapped at a time,
scored against NEMO (wet cells):

| operand swapped | V pivot row max | U pivot T row | V row below |
|---|---:|---:|---:|
| none (card) | 8.594e-09 (card increment exactly 0) | 5.509e-12 | 3.5e-13 |
| **V face mask -> NEMO raw vmask** | **4.915e-12** | unchanged | unchanged |
| velocities -> NEMO's recorded | unchanged (card u,v equal NEMO's, 0 unequal) | unchanged | unchanged |
| U face thickness -> fold image | — | unchanged (already symmetric) | — |

**First non-bit operand (V pivot row): the V face mask.**  NEMO's `dyn_zad`
updates the pivot V row (raw vmask wet, 68 faces at the surface); the card
handed the ZAD routine the compact mask, dry there, so the increment was
exactly zero.  The residual on the U pivot T row, the V row below and the V
pivot row reads the card's `ww` on the pivot T row: inverting NEMO's
recorded ZAD increment for the face pair `e1e2t*ww(i)+(i+1)` agrees with the
card to ~1e-12 on interior rows 140 and 145 (instrument control) and
differs by median 4.3e-4 (max 0.12) on the pivot T row.  Ruled out as its
source, by measurement: the card's sea-level operands to that `w` (equal to
NEMO's recorded stage sea levels at every wet cell), the pivot V transport
(included; its omission would be ~1e5 times larger), the velocities.  NEMO
computes `ww` on the pivot T row and the first halo row directly in
`wzv_RK3_t` (`sshwzv.f90:236`, `div_hor` :278, recurrence :298), with no
fold.  Statement not yet named (OPEN 2).

## 3. Landed statement (G6-card-selected)

Under `rk3_vector_raw_vmask` (selected only by the G6 card,
`nemo_testcase_recipe.py:3834`; default False; caller grep: the only other
`rk3_vector_raw_vmask=` is the ladder's `--without-vector-raw-vmask` plant)
the vector-form vertical momentum advection receives NEMO's raw vmask (the
literal mask already handed to the stage update and HPG) for its V faces.
Every other card executes the identical expression (`else v_mask_3d`).

| prediction (preregistered) | measured |
|---|---|
| P1 replay calibrates | yes (0/0 where the card equals NEMO) |
| P2 owner = KEG on the fold rows | **REFUTED**: KEG exact everywhere; ZAD owns the fold rows |
| P3 ZAD equal on fold rows, or owns the interior set | ZAD owns both the fold rows and the interior last-place set |
| P4 G6-only selection | yes |

kt-1 stage-2 operators after landing (surface max / unequal; all levels):
HPG 0/0, VOR 0/0 (unchanged); after-ZAD U 3.488e-13/84 (unchanged), **V
6.271e-13/195 (was 1.214e-09/195)**; all levels U 5.23e-12 (unchanged),
**V 4.91e-12 (was 6.80e-09)**.

## 4. Both labels, every recorded surface field (after landing)

`max_abs / unequal cells`, bar 1e-15.  Identical to round 38 at the printed
precision: the pivot V row is overwritten by the stage-end fold, so its own
increment does not reach the step-end state.

| kt | label | ssh | uu_b | vv_b | u | v | T | S |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | indep. = given | 0/0 | 0/0 | 0/0 | 4.613e-09/135 | 2.625e-09/202 | 0/0 | 0/0 |
| 2 | indep. = given two-step | 4.461e-03/16433 | 4.032e-03/15853 | 3.326e-03/15943 | 8.087e-03/15853 | 4.912e-03/15943 | 1.044e-03/16433 | 3.117e-03/16433 |
| 3 | independent | 9.152e-03/16433 | 1.080e-02/15853 | 5.253e-03/15943 | 1.311e-02/15853 | 6.716e-03/15943 | 4.009e-03/16433 | 7.847e-03/16433 |
| 4 | independent | 1.855e-02/16433 | 1.229e-02/15853 | 9.786e-03/15943 | 1.682e-02/15853 | 9.214e-03/15943 | 7.130e-03/16433 | 1.096e-02/16433 |
| 5 | independent | 2.883e-02/16433 | 1.427e-02/15853 | 7.508e-03/15943 | 1.150e-02/15853 | 1.078e-02/15943 | 9.387e-03/16433 | 1.736e-02/16433 |
| 6 | independent | 5.074e-02/16433 | 2.714e-02/15853 | 2.234e-02/15943 | 2.860e-02/15853 | 2.236e-02/15943 | 1.172e-02/16433 | 1.771e-02/16433 |
| 3 | given (restart 2) | 9.078e-03/16433 | 8.415e-03/15853 | 5.584e-03/15943 | 1.273e-02/15853 | 6.931e-03/15943 | 2.609e-03/16433 | 7.744e-03/16433 |
| 4 | given two-step | 1.770e-02/16433 | 1.030e-02/15853 | 9.732e-03/15943 | 1.579e-02/15853 | 8.976e-03/15943 | 5.848e-03/16433 | 1.091e-02/16433 |
| 5 | given (restart 4) | 2.671e-02/16433 | 1.580e-02/15853 | 7.635e-03/15943 | 1.642e-02/15853 | 8.693e-03/15943 | 4.980e-03/16433 | 6.989e-03/16433 |
| 6 | given two-step | 5.330e-02/16433 | 2.691e-02/15853 | 2.714e-02/15943 | 2.924e-02/15853 | 3.072e-02/15943 | 9.900e-03/16433 | 1.496e-02/16433 |

Correction to round 38's table: NEMO restarts exist only at even steps, so
the kt-2 "given" row is the card's own chain from kt 1 (not an independent
NEMO entry), and the even-kt given rows are two-step rows.

## 5. kt 2: first step, stage, field, region, owner

Stages arm resumed from the card's own kt-1 chain (INDEPENDENT label),
against NEMO's kt-2 stage records:

| stage (kt 2) | ssh | u | v | T | S |
|---|---:|---:|---:|---:|---:|
| 1 | 1.487e-03/16433 | 2.059e-02/15853 | 1.115e-02/15943 | 3.634e-04/16433 | 9.807e-04/16433 |
| 2 | 2.231e-03 | 5.444e-03 | 8.884e-03 | 5.389e-04 | 1.493e-03 |
| 3 | 4.461e-03 | 8.087e-03 | 4.912e-03 | 1.044e-03 | 3.117e-03 |

- **First: kt 2, stage 1, every prognostic field, whole wet domain, all
  depths.**  NOT carried from the kt-1 fold rows (kt-1 entry is exact
  except ~1e-9 on the fold rows; kt-1 fold error fell 1000x in round 38 and
  kt 2 did not move).  Not a step-size error: the card's stage-1 increments
  match NEMO's on median (ratio 1.00001) with ~1e-3 (T, S, ssh) and ~10 %
  (u, v) scatter.
- **Owner (CONFIRMED by a one-field scratch arm, not landed):** the G6 card
  resolves NEMO's first vector-form `wzv` call (`stp2d.f90:149-156`: the
  after-height ratio `r3t(Kaa)` is set from the `ssh(Kaa)` slot at :152 and
  read by `wzv` at :156) with `nemo_first_wzv_after_ssh='rk3_extrapolated'`
  (a fresh extrapolation).  The ORCA2 card resolves the same statement as
  `'rk3_extrapolated_carried'` (the slot the previous step left, carried as
  NEMO's `ssha`).  The two agree at kt 1 and differ from kt 2.  Changing
  ONLY that field on the G6 card (scratch, `5f471f290` + in-memory card
  edit):

  | kt | field | card now | carried slot |
  |---|---|---:|---:|
  | 2 | ssh | 4.461e-03/16433 | 4.394e-08/2396 |
  | 2 | u | 8.087e-03/15853 | 8.355e-07/2323 |
  | 2 | T | 1.044e-03/16433 | 5.270e-10/946 |
  | 3 | ssh | 9.152e-03/16433 | 1.061e-07/9360 |
  | 3 | u | 1.311e-02/15853 | 1.994e-06/7472 |

  kt 1 is unchanged by it (identical rows).  It adds a carried field to
  G6's state (the after-SSH slot, restart-loaded on the ORCA2 card), so it
  is a carried-state change and is NOT landed: DECISION_NEEDED.

## 6. OPEN (ranked by magnitude)

1. **kt 2+ whole domain** — owner measured (§5); lands on the user's answer.
   Then re-walk kt 2 for the residual (scratch: ssh 4.394e-08 on 2396 cells).
2. kt-1 ZAD residual on the fold rows (U pivot T row 5.5e-12, V rows
   ≤4.9e-12, 1292-1319 cells per row): the card's `ww` on the pivot T row
   (median 4.3e-4 of the transport pair).  Next: capture the card's
   per-level divergence on that row against a literal replay of
   `div_hor_RK3` np_velocity (`divhor.f90:125-130`) + `sbc_rnf_div`
   (`sbcrnf.f90:238`, `ln_rnf=T`) from NEMO's recorded operands.
3. Interior ZAD last-place set (U 314 / V 600 cells, ≤8.5e-22 absolute).

## D122 pointer (ORCA2 lane)

The same compiled line runs on ORCA2: `ORCA2_OMIP_L4/BLD/ppsrc/nemo/
dynzad.f90:125-126` (SUBROUTINE `dyn_zad` :59) updates the pivot V row with
the raw `vmask` only in the thickness factor.  Any ORCA2 card carrying NEMO's
raw V mask must hand it to the ZAD V faces too, not only to the stage
update, HPG and vorticity.  Not imported by this round.

## Tests and review

- New `test_g6_dyn_zad_v_face_mask_keeps_the_pivot_row` (G1b card with the
  G6 flag): the V face mask reaching the ZAD routine is wet on the pivot row
  with the flag, dry without.  Shown to FAIL with the fix reverted
  (`assert all(p > 0 ...)`), then restored (`git status --porcelain` clean).
- `tests/ocean/unit/test_nemo_orca2g1b_card.py` +
  `tests/ocean/fidelity/test_nemo_testcase_g6_orca2vec.py`: 26 passed (CPU
  fp64).
- Receipt-citation gate + private-import ratchet: 23 passed, after a
  mechanical +4 re-anchor of three `ocean_pe_latlon_cgrid.py` gate pins and
  two audited receipt citations past the inserted lines.
- codex review: paused (quota).  The operator runs an independent Claude
  review of this diff before landing.

ASKED: KEG/ZAD split at kt 1 with citations; first non-bit operand; its
G6-selected landing; re-score both labels kt 1..6; kt-2 first step, stage,
field, region and owner.

UNASKED, offered for revert: the citation re-anchor (mechanical); the
scratch one-field arm in §5 (evidence only, no card change).  No defaults,
no other cards, no deck changes.

UNVERIFIED: the pivot-row `ww` statement (OPEN 2); whether the carried
after-SSH slot also needs the ORCA2 card's restart loader on G6 (it will if
landed).
