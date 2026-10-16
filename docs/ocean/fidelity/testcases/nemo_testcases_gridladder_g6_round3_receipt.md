# Grid ladder rung G6 — round 3 receipt (gridladder_r38)

Status: **LANDED; G6 NOT CERTIFIED.**  The kt-1 stage-2 EEN vorticity term
is now bit-exact in value at every face and depth (was 68 U / 136 V surface
faces, 1292 U / 2630 V all levels).  The next non-bit statement at kt 1 is
the KEG+ZAD accumulator on the same fold rows.  kt 2..6 are unchanged and
remain DEBT under both labels (whole-domain, ~4.5e-3 m sea level at kt 2).

Preregistration: `PREREG_nemo_testcases_gridladder_g6_round3.md` (committed
before the card change was measured).  Record: round 36
`gridladder_rounds/round36/oracle_g6_r36/` (admitted round 37, unchanged).
Evidence: `gridladder_rounds/round38/`.  SHA-256 prefixes: independent
`d01958fe4e20b864`, given-entry `f56b66c033596287`, stages kt 1
`c94f5f69f0a51531`, operators kt 1 `bd4d5cdfd2ca2498`, pre-registration
attribution `probe_pivot_zeta.json` (probe `probe_pivot_zeta.py`, capture
`capture.py`).  Final runs stamped `e31b570a09` (clean tree).

## 1. The walk (kt 1, stage 2, `vor_een` in compiled order)

NEMO: `vor_een` opens at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynvor.f90:692`
(identical body in the record build `ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1`).
np_CRV forms the F-point vorticity at :762-766 from `pv(ji+1,jj)`,
`pv(ji,jj)`, `pu(ji,jj+1)`, `pu(ji,jj)` with no mask; the transports at
:783-786; the triads at :789-794; the update at :796-803.  On the pivot row
`pv = vv(Kmm)` is the stage-end `lbc_lnk` image: `lbc_nfd_dp` T-pivot 'V'
(`lbcnfd.f90:684-721`, enclosing SUBROUTINE at :561) overwrites the last
ihls+1 rows INCLUDING the pivot row ipj-ihls from the row below, sign -1.

Operands, compared one at a time (card captured passively by a debug
callback on the stage-2 operator call; the capture reproduces the round-37
dumped card accumulator `after_hpg + out*mask == after_vor` in value at every
cell, asserted in the probe):

| operand at the fold rows | card vs NEMO's statement |
|---|---|
| ff_f, F thickness (z1_e3f), metric widths | equal (pivot row = fold image of the row below, 0 unequal) |
| pivot-row V transport `zwy` (raw mask, fold image) | equal (round-37 landing) |
| **pivot-row F vorticity `zwz`** | **FIRST NON-BIT**: the card's equals the vorticity of a V field whose pivot row is ZERO (0 unequal cells) |

Cause, read off the code and confirmed by the swap: the WS-RK3 stage state
handed to the momentum tendency multiplied V by the compact `v_mask_3d`,
which is dry on the pivot row, while the stage update itself already keeps
NEMO's raw pivot support.  Replaying the SAME card operator with only `zeta`
recomputed from the stage V (pivot row kept) gives 0 unequal U and 0 unequal
V cells against NEMO's `after_vorticity` record at every level.  With that
vorticity the pivot F row equals the F fold image of the row below in every
cell, i.e. NEMO's direct halo arithmetic.

## 2. Landed statement (G6-card-selected)

Under `rk3_vector_raw_vmask` (selected only by the G6 card,
`nemo_testcase_recipe.py:3834`; default False; caller grep: no other
selector) the stage state given to the momentum tendency keeps V on the raw
stage support `_ws_stage_v_mask` instead of the compact mask.  Every other
card executes the identical expression as before.

Prediction table (preregistered):

| | prediction | measured |
|---|---|---|
| P1 VOR kt 1 stage 2 | 0/0 every level | U 0, V 0 (value) every level; AT-BAR, max 0 |
| P2 HPG | stays 0/0 | 0/0 |
| P3 KEG+ZAD | falls, may not reach 0 | surface U 9.961e-10/84 -> 3.488e-13/84; V 3.578e-09/197 -> 1.214e-09/195; all-level U 1.335e-09 -> 5.225e-12 (full), V 9.696e-09 -> 6.804e-09 (full) |
| P4 other cards | unchanged | flag selected by G6 only; G1b/G6 focused files pass |

The bitwise counts in the operator JSON (~47-50k) are signed zeros on dry or
zero cells; they are identical at `after_hpg`, which is value-exact, and are
not differences.

## 3. Both labels, every recorded surface field

`max_abs / unequal cells`, bar 1e-15.

| kt | label | ssh | uu_b | vv_b | u | v | T | S |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | indep. = given | 0/0 | 0/0 | 0/0 | 4.613e-09/135 (was 7.089e-06) | 2.625e-09/202 (was 7.961e-06/203) | 0/0 | 0/0 |
| 2 | indep. = given (2x) | 4.461e-03/16433 | 4.032e-03/15853 | 3.326e-03/15943 | 8.087e-03/15853 | 4.912e-03/15943 | 1.044e-03/16433 | 3.117e-03/16433 |
| 3 | independent | 9.152e-03/16433 | 1.080e-02/15853 | 5.253e-03/15943 | 1.311e-02/15853 | 6.716e-03/15943 | 4.009e-03/16433 | 7.847e-03/16433 |
| 4 | independent | 1.855e-02/16433 | 1.229e-02/15853 | 9.786e-03/15943 | 1.682e-02/15853 | 9.214e-03/15943 | 7.130e-03/16433 | 1.096e-02/16433 |
| 5 | independent | 2.883e-02/16433 | 1.427e-02/15853 | 7.508e-03/15943 | 1.150e-02/15853 | 1.078e-02/15943 | 9.387e-03/16433 | 1.736e-02/16433 |
| 6 | independent | 5.074e-02/16433 | 2.714e-02/15853 | 2.234e-02/15943 | 2.860e-02/15853 | 2.236e-02/15943 | 1.172e-02/16433 | 1.771e-02/16433 |
| 3d | given | 9.078e-03/16433 | 8.415e-03/15853 | 5.584e-03/15943 | 1.273e-02/15853 | 6.931e-03/15943 | 2.609e-03/16433 | 7.744e-03/16433 |
| 4d | given | 1.703e-02/16433 | 1.043e-02/15853 | 7.449e-03/15943 | 1.806e-02/15853 | 7.102e-03/15943 | 3.229e-03/16433 | 4.528e-03/16433 |
| 5d | given | 2.671e-02/16433 | 1.580e-02/15853 | 7.635e-03/15943 | 1.642e-02/15853 | 8.693e-03/15943 | 4.980e-03/16433 | 6.989e-03/16433 |
| 6d | given | 8.026e-02/16433 | 3.793e-02/15853 | 1.828e-02/15943 | 4.433e-02/15853 | 2.087e-02/15943 | 1.478e-02/16433 | 2.076e-02/16433 |

kt 2..6 are identical to round 37 at the printed precision under both labels.

kt-1 regions (surface): u — fold pivot row 4.613e-09/68, fold row below
2.944e-10/67, seam/south/coast/open 0; v — pivot row 2.625e-09/68, row below
2.625e-09/68, third row 8.087e-13/63, interior 1.388e-17/3 (coast 2, open 1;
present unchanged in round 37).

## 4. Depth classes, kt 1 (stage exposures; relative max / unequal; dry 0)

| stage | u full | u partial | v full | v partial | T, S |
|---|---:|---:|---:|---:|---:|
| 1 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| 2 | 2.57e-08/1306 | 2.93e-08/63 | 1.18e-09/2562 | 8.91e-10/124 | 0/0 |
| 3 | 2.23e-08/2716 (was 4.72e-06) | 2.59e-08/129 (was 5.51e-06) | 1.23e-08/3874 (was 5.89e-06) | 4.41e-09/186 (was 4.95e-06) | 0/0 |

G6 remains DEBT under both labels from kt 1 (fold rows) and over the whole
domain from kt 2.

## 5. OPEN (next round, ranked by magnitude)

1. **kt 2 whole domain** (ssh 4.461e-03, every wet cell, both labels): NOT
   seeded by the fold — kt-1 fold error fell 1000x and kt-2 numbers did not
   move.  The stages arm cannot enter kt 2 (no kt-1 restart; admission
   recorded stage-3 kt 1 = kt-2 entry, which the arm does not yet read).
   First move: let the stages/operators arms enter kt from the admitted
   previous stage-3 frame, then walk kt 2.  Cause unknown (PLAUSIBLE: an
   every-other-step path, nn_fsbc=2, that kt 1 does not run).
2. kt-1 stage-2 KEG+ZAD on the fold rows (V 1.214e-09 surface on the pivot
   and the row below; U 3.5e-13 pivot row), plus a scattered interior
   last-place set (present before).  NEMO: `dyn_keg` (`dynkeg.f90:61`) forms
   `zhke` on the halo T row directly (:120-125) and the pivot V row reads
   `zhke(ji,jj+1)` (:130); `dyn_zad` (`dynzad.f90:59`) reads `ww(ji,jj+1)`
   on the halo T row (:105-127).  Neither operand walked yet.

## D122 pointer (ORCA2 lane fold unit)

The same compiled line runs on ORCA2: `ORCA2_OMIP_L4/BLD/ppsrc/nemo/
dynvor.f90:762-766` (SUBROUTINE `vor_een` :692) reads the pivot-row
`vv(Kmm)`, which is the `lbc_nfd_dp` 'V' fold image
(`lbcnfd.f90:684-721`, SUBROUTINE :561).  Any ORCA2 card that carries NEMO's
raw surface V mask (rounds 289-299) must hand that row to the vorticity
term, not only to the transport and the stage update; with the compact mask
the pivot F vorticity is silently computed from a zero V row (relative error
up to 2.6e-4 on the vorticity increment here).  Not imported by this round.

## Tests and review

- New `test_g6_stage_state_hands_vor_een_the_folded_pivot_v` (G1b one-level
  card with the G6 flag): the stage V reaching the tendency keeps the pivot
  row equal to the fold image of the row below; with the flag off it is
  zero.  Shown to FAIL with the fix reverted (`assert wet.any()`), then
  restored.
- The round-37 synthetic word pin `test_g6_raw_vmask_consumers_are_live_
  under_traced_jit` moved (expected: the pivot V now reaches the stage-2 RHS);
  re-pinned (rhs `0xbdff8bd7c5fbf193...`, raw `0xbec4cb6e7bc1d37e...`).
- `tests/ocean/unit/test_nemo_orca2g1b_card.py` +
  `tests/ocean/fidelity/test_nemo_testcase_g6_orca2vec.py`: 25 passed (CPU
  fp64).
- Receipt-citation gate + private-import ratchet: 23 passed, after a
  mechanical +4 re-anchor of 60 `ocean_model_latlon_cgrid.py` pins past the
  inserted lines (gate map and audited receipt section).
- single review (codex): see the review section appended below.

ASKED: the vor_een walk at kt 1 in compiled order; first non-bit operand
with its citation; G6-card-selected landing tied to `rk3_vector_raw_vmask`;
both labels kt 1..6; D122 pointer.

UNASKED, offered for revert: the re-pin of the round-37 synthetic word test
(forced by the landing); the citation re-anchor (mechanical).  No defaults,
no other cards, no deck changes.

UNVERIFIED: the kt-2 owner (item 1); the KEG/ZAD fold operands (item 2).
