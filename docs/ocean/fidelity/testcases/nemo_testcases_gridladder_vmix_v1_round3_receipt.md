# Vertical-mixing lane — V1 round 3 (vmix_r3)

Status: **HELD. V1 is not certified.** The record is admitted (passivity
PASS). Two cited TKE statements landed, both selected by V1 alone. Both
labels are still DEBT from kt 1. The next owner is named (PLAUSIBLE) and is
not landed.

Decision 129 is still pending. This round used the as-built deck: NEMO's own
TKE floors, internal-wave mixing off.

No new preregistration was written. The round was scored against round 1's
frozen P1/P2, and both are falsified (section 2).

Commits:
- `ae07cb6c8`: the landing, instrument and tests;
- `94d20fbfb`: citation re-anchor;
- `8e1dc0ee0`: ladder summary printer.

Record: `gridladder_vmix_rounds/round1/oracle_v1_r1` (operator's
`run.sh --resume`; log `v1_acq2_run.log`: `V1_ACQUISITION_PASS steps=1..10
restarts=2..10`).

Evidence `gridladder_vmix_rounds/round3/` (`manifest.sha256`):

| file | sha256 prefix |
|---|---|
| `v1_admit.json` | 7467feb92edadf02 |
| `v1_independent_kt1_5.json` | 44ac42a1f768f9c1 |
| `v1_independent_kt6_10.json` | 0246496a9187626a |
| `v1_given_e13.json` | a684d94caa108f9c |
| `v1_given_e57.json` | 55cb0cb2ba3d2b7f |
| `v1_given_e9.json` | 136a48782edeb792 |
| `summary.txt` | 5d19f9ec3de48edc |
| `plant_mxlice.log` | 26c2f818ecfacc81 |
| `reanchor.py` | 0f749fec790f9719 |
| `v1_indep_kt1_dirty.json` (before) | 36fc5c1c91fe722b |
| `v1_indep_kt1_lc_dirty.json` (Langmuir only) | 809e15c088bf32bb |
| `v1_indep_kt1_mxlice_dirty.json` (both) | b9f8f8f9b8ad82e3 |

The kt 1..10 ladders are stamped clean: kt 1-5 and the given-entry runs at
`94d20fbfb`, kt 6-10 at `8e1dc0ee0`. Both commits have the same `packages/`
tree.

## 1. Passivity (the reviewer's must-do): PASS

G5's admission (`g5_records.py admit`) was run unchanged on the V1 evidence:

- **Twin A vs writer-free reference, kt 10:** `restarts_identical` passes on
  all 42 restart variables, including NEMO's TKE state (`en`, `avm_k`,
  `avt_k`, `dissl`).
- **Twin B vs reference:** identical at kt 2, 4, 6, 8 and 10 (42 variables
  each). Twin C shares no restart step with the reference (twin C writes odd
  steps, the reference even ones).
- **Twin bytes:** all 60 per-step frames are byte-identical across the three
  twins. The `hm_i` frames for kt 1, 5 and 10 hash identically across the
  twins; the acquisition also checks every `hm_i` frame.
- **Frame/restart consistency:** stage 3 equals the next step's entry at
  kt 1..9, and restarts equal the stage-3 frames at kt 2..10.
- **Plant:** one ULP moved in a copy of twin A's restart is refused
  (`PASSIVITY FAILED ... sshn differs`, exit 1).

## 2. Admission facts and the round-1 predictions

- **P1 (boundary) FALSIFIED.** NEMO's writer-free V1 reference stops in
  `stp_ctl` at **kt 26** (ocean.output:3080-3082; `time.step` = 26), not kt 11
  as in G5. The last completed step is 25.
- **Scoring stops at kt 10.** The twins record frames and NEMO's per-step
  surface fluxes only for kt 1..10. The card is driven by those fluxes
  (D123), so kt 11..25 cannot be scored from this record.
- **River-mouth mixing is inert.** `socoefr0` is zero everywhere, so NEMO's
  `ln_rnf_mouth` diffusivity term adds nothing, and NEMO's `avt` equals the
  closure's `avt_k`.
- **P2 FALSIFIED (falsifier a).** At kt 1, before any fix, 11,682 ice-free
  columns had unequal TKE. 20 of the 26 columns with over-bar TKE were
  ice-free.

## 3. TKE state at kt 1, card vs NEMO

NEMO's frame is taken right after `zdf_phy` at kt 1 (record build
`stprk3.f90:371-388`, `l2_dump_zdf`). The card's values are its closure
state after step 1, on w-levels 2..30.

"Unequal" counts every cell that is not bit-identical, including last-place
differences. "Over-bar" counts cells above 1e-15 times max(|NEMO|, 1).

| field | stage | unequal cells / columns | over-bar cells / columns | max abs diff | NEMO max | card max |
|---|---|---:|---:|---:|---:|---:|
| en | before | 49,924 / 13,341 | 130 / 26 | 1.66e-2 | 0.06755072080608174 | 0.06755072080608177 |
| en | after both | 35,577 / 12,579 | **0 / 0** | 6.94e-18 (AT-BAR) | 0.06755072080608174 | 0.06755072080608174 |
| avm | before | 120,494 / 15,653 | 476 / 314 (243 under ice) | 4.83e-1 | 1.5753286834874598 | 1.57532868348746 |
| avm | Langmuir fix only | — | 413 / 309 (238 under ice) | 3.41e-1 | — | — |
| avm | after both | 114,718 / 15,629 | **148 / 92 (21 under ice)** | 3.41e-1 | 1.5753286834874598 | 1.5753286834874598 |
| avt | after both | 109,930 / 15,584 | 147 / 91 (20 under ice) | 1.24e-1 | 0.8914170986973796 | 0.8914170986973796 |

Surface and ice inputs that TKE reads:

- `taum` and `fr_i` reach the card through G5's certified route. G5's
  identity run showed them equal to NEMO's record at every step.
- `hm_i` is now read from NEMO's record. It is nonzero in 3,630 cells at
  kt 1, exactly the 3,630 ice-covered cells, with max 2.007 m.
- The surface `avm_k` row is exact at every step.

Mixing lengths are not in NEMO's record at kt 1. The dissipation length
`dissl` is scored from kt 2 in section 5.

## 4. First non-bit statements, in compiled order

### 4.1 Langmuir depth when no level crosses (LANDED)

NEMO initialises `imlc(:) = mbkt + 1` in every column, then searches upward
(`ORCA2V1_OMIP_L4_R1FULLENTRY/BLD/ppsrc/nemo/zdftke.f90:354-357`; SUBROUTINE
`tke_tke` at :199). So when no level crosses the energy threshold, the
Langmuir depth is that column's own bottom.

The card inherited the compact Langmuir arm, which falls back to the
**grid's** deepest interface. In shallow columns (7-10 wet levels in the
worst cases) that injects a weak, monotone source where NEMO's peaks
mid-column. NEMO's en at w4 in column (118,138) is 0.0090; the card's was
0.0011.

V1 now selects the shared `nemo_literal` arm, which already transcribes
`:354`. Over-bar TKE cells go from 130 to 0.

### 4.2 Under-ice surface mixing length, `nn_mxlice=2` (LANDED, transcribed)

NEMO statement (`zdftke.f90:620-623` then `:635-637`; SUBROUTINE `tke_avn`
at :531):

`zmxlm(1) = ((1-fr_i)*zraug*taum + fr_i*hm_i*2)*tmask(1)`, then
`MAX(rn_mxl0, zmxlm(1))`.

What landed:
- a new `TKEConfig.nemo_mxlice` field. The default 0 is the existing
  CASE 0 path, unchanged. Unknown values raise.
- arm 2 reads the raw `fr_i` (not the `nn_eice` tanh fraction) and `hm_i`.
  `hm_i` comes from a new optional `OceanSurfaceForcing.ice_mean_thickness_hm_i`.
  Missing inputs raise.

Over-bar avm columns under ice go from 238 to 21.

Gate: `test_nn_mxlice2_anchor_is_nemos_case2_and_refuses_missing_inputs`
compares bits against the NEMO expression. Planting `hm_i*1.0` turns it red
(`plant_mxlice.log`).

### 4.3 Next owner (NOT landed; PLAUSIBLE)

The bottom-up mixing-length sweep reads the thickness of the cell just below
each wet w-level:
`zmxlm(jk) = MIN(zmxlm(jk+1) + e3t_3d(jk+1)*(1+r3t*tmask(jk+1)), zmxlm(jk))`
(`zdftke.f90:694-697`, `tke_avn`).

At a column's bottom, NEMO uses:
- the rest thickness of the first dry cell, for partial columns;
- level 31's thickness, `e3t(jpk)`, for full-depth columns.

NEMO's values run from 1.3 m to 1000 m and vary by column. The card has no
input for either; it falls back to level 30's thickness.

Evidence: of the 71 ice-free over-bar columns, 28 are full-depth with the
first over-bar cell at w30. 23 of the other 43 have it at their own bottom
w-level.

The shared TKE already accepts an extra true `e3t(jpk)` row; no model path
supplies it. The fix needs a new carried mesh input, so it is left for the
next round. It has not been tested by replay; label **PLAUSIBLE**.

### 4.4 Registered, not owners

- **Dry-cell `dissl`.** The card leaves values around 1e-145 on 346,979 dry
  w-cells where NEMO has exactly 0. Nothing reads dry `dissl` (round 1's
  argument), but it fails the dry-cell check on every restart row.
- **Lane-A merge touchpoint.** The full ORCA2 card keeps the compact
  Langmuir arm and has no `nn_mxlice` arm, so both defects are present there
  too. That is unmeasured and was not changed (D122).

## 5. Both labels, kt 1..10 (the record's scoreable span)

Format: unequal cells / max abs diff. "lev" = all-level rows, scored where
NEMO wrote a restart.

INDEPENDENT:

| kt | ssh | u (level 1) | T (level 1) | lev en | lev avm_k | lev T | lev dissl (dry unequal) |
|---|---|---|---|---|---|---|---|
| 1 | **0** | 3911 / 1.2e-3 | 1485 / 1.2e-2 | — | — | — | — |
| 2 | 14867 / 1.0e-3 | 15653 / 3.8e-3 | 15607 / 1.9e-2 | 124320 / 2.3e-3 | 155306 / 4.6e-1 | 201454 / 1.7e-1 | 324750 (346979) |
| 5 | 16433 / 3.0e-3 | 15853 / 7.1e-3 | 16432 / 3.9e-2 | 194235 / 9.5e-3 | 175482 / 5.2e-1 | 430401 / 1.2e-1 | 413368 (346979) |
| 10 | 16433 / 5.6e-3 | 15853 / 1.3e-2 | — | 238093 / 1.0e-2 | 178235 / 5.6e-1 | 430552 / 2.2e-1 | 414119 (346979) |

GIVEN-NEMO-ENTRY (NEMO's restart, including NEMO's TKE state):

| entry kt | ssh | u (level 1) | T (level 1) | two-step lev en | two-step lev avm_k |
|---|---|---|---|---|---|
| 1 | 0 | 3911 / 1.2e-3 | 1485 / 1.2e-2 | 124320 / 2.3e-3 | 155306 / 4.6e-1 |
| 3 | 0 | 15208 / 3.3e-3 | 15490 / 9.7e-3 | 127480 / 1.7e-3 | 157900 / 4.4e-1 |
| 5 | 0 | 15273 / 4.7e-3 | 15709 / 1.3e-2 | 128816 / 9.3e-3 | 158753 / 4.8e-1 |
| 7 | 0 | 15344 / 2.1e-2 | 15831 / 2.6e-2 | 130539 / 1.2e-2 | 160727 / 6.3e-1 |
| 9 | 0 | 15460 / 3.8e-2 | 15935 / 2.4e-1 | 131916 / 9.4e-3 | 161872 / 5.8e-1 |

Every step not shown is in `summary.txt`. Every row is DEBT except ssh,
`uu_b` and `vv_b` at the first step of each entry, and surface `avm_k`.

**V1 is NOT certified.**

The score plant (`--plant score`) exits 1, but every row is already DEBT, so
here it proves nothing.

## 6. Decision 122 (ORCA2_OMIP_L4 build)

`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90` is byte-identical to the V1
record's (`cmp`), so the line numbers carry over.

| statement | ORCA2_OMIP_L4 line | SUBROUTINE |
|---|---|---|
| Langmuir no-crossing depth `imlc = mbkt+1` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:354` | `tke_tke` (:199) |
| `nn_mxlice=2` surface length | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:620-623` | `tke_avn` (:531) |
| next owner, bottom-up sweep | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:694-697` | `tke_avn` (:531) |

## 7. Selection and caller grep (no-silent-default rule)

- `tke_langmuir_evaluation="nemo_literal"` and `nemo_mxlice=2` are set only
  in the V1 card's vertical block.
- `_orca2v1_tke_selected` pins both, plus the masked anchor. Planting either
  back to its old value fires the selector check (test).
- G1-G5, the full ORCA2 card and every library default are unchanged. The
  G5 card still has `nemo_mxlice == 0` (test).
- The label `tke_nn_mxlice2_under_ice_mixing_length` is removed. The V1
  unmeasured list is now `("nemo_post_si3_surface_records",)`.
- The label `v1_record_not_admitted` is removed, because the record is
  admitted.

## Tests and review

All runs were synchronous, on CPU in fp64 (`JAX_ENABLE_X64=1`):

| suite | result |
|---|---|
| V1 file (13) + TKE NEMO terms | 82 passed |
| G2-ladder + G4 + G5, not slow | 51 passed, 11 deselected |
| TKE unit suites (carried, under-ice, identity, integration, prognostic, dry-wmask) | 163 passed |
| Citation gate + private imports, clean tree, after re-anchor | 23 passed |

- New tests: the `hm_i` reader, with four refusal plants; the ZDF-frame
  reader and the w-class split; the under-ice arm, with a red plant.
- Citation re-anchor: 3 citations moved mechanically (`reanchor.py`). One
  pinned span grew from 9 to 11 lines; both of its endpoint statements are
  unchanged.
- I amended my own minutes-old, unpushed re-anchor commit once to fix that
  span length. Disclosed here.

Single review (codex): see the addendum below.

## Scope

ASKED: passivity first; admission and kt-1 TKE state identity; both labels;
first non-bit statement with its citation; transcribing `nn_mxlice=2`; the
Decision-122 lines; tests; one review.

UNASKED, offered for revert:
- landing the Langmuir arm selection (section 4.1) in addition to
  `nn_mxlice`. It is a cited statement that comes earlier in compiled order;
  it is card-selected, and the arm itself already existed;
- the new forcing field `ice_mean_thickness_hm_i`. The transcription needs
  it, and it is inert unless selected.

UNVERIFIED:
- the next owner (4.3);
- the full ORCA2 card's response (pointer only).
