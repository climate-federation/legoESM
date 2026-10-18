# Grid ladder rung G9 — round 2 resume receipt (gridladder_r47)

Status: **LANDED as asked, G9 NOT CERTIFIED.**  The FCT's pivot-row
half-overwrite is transcribed (G9 only).  It is a no-op on every
self-mirrored input, and that is what NEMO's FCT always receives: NEMO
re-mirrors T and S at the end of every stage.  The card does not.  That
missing end-of-stage fold is why the card's pivot row loses its mirror.

- Landing commit `2cdf9d2afd`.  Record: round 46's, unchanged; no acquisition.
- codex review: paused (quota).  The operator runs an independent Claude
  review of this diff before landing.
- Evidence: `gridladder_rounds/round47/resume/` (manifest `manifest.sha256`).

## 1. The half-row overwrite (item 1)

`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:311` exchanges `ztFu` (U, −1),
`ztFv` (V, −1) and `zta_up1` (T, +1).  Round 2 transcribed the V part.  On
the T-pivot fold the same call also rewrites the right half of the stored
pivot row from its left half:

- T: native cells N/2+1 .. N−1 take cells N/2−1 .. 1
  (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/lbcnfd.f90:625`), sign +1, on `zta_up1`.
- U: native faces N/2 .. N−1 take faces N/2−1 .. 0
  (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/lbcnfd.f90:670`), sign −1, on the
  antidiffusive `ztFu`; the periodic west column repeats the last face.

Both feed the limiter and the final trend.  Landed under the existing
`fct_fold_exchange` (G9 only) through one new shared helper in the core
fold operators.  The external-mode U fold association already had the
U arm inline; it now calls the helper with the same operations in the
same order.  Searched first: `half`, `n_lon // 2`, `perm_T`,
`pivot_row_stored`, `fold_perm_u` across `packages/`; the only existing
copy was that U arm.

Other cards: traced step jaxpr, addresses masked, unchanged — G5
`339a8016d262c4f6`, G8 `03ccdb94113eab61` (equal to round 47's).  A first
version of the helper swapped two operations and moved both hashes; the
order was restored.

## 2. Where NEMO re-mirrors T and S (item 2)

After every stage, G9 runs
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:795` with T and S on
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:796` (`ldfull`): the full
exchange, including the T-pivot half-row.  G9 takes this branch because
constant vertical mixing sets `l_zdfsh2` false
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:276`) and `ln_shuman` is off.
NEMO's stage records are written after the stage returns
(`ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/stprk3.f90:338`), so
they are post-exchange.

**The card does not do this.**  Its end-of-stage exchange covers the top
V row only; no T, S (or U) pivot half-row exists anywhere in the model.
Not landed: it is a new statement, not asked for this resume.

## 3. Pivot-row self-mirror counts, kt 1..10 (item 3)

Wet cells on the stored pivot row, all levels, whose value differs from
their mirror partner.  Card = independent label, NEMO's surface input.

| kt | card T / S before | card T / S now | NEMO stages 1, 2, 3 | NEMO restart |
|---|---|---|---|---|
| 1 | 2 / 0 | 2 / 0 | 0 | — |
| 2 | 2 / 0 | 2 / 0 | 0 | 0 |
| 3–5 | 0 / 4 | 0 / 4 | 0 | 0 |
| 6 | 2 / 4 | 2 / 4 | 0 | 0 |
| 7 | 4 / 4 | 4 / 6 | 0 | 0 |
| 8 | 68 / 8 | 102 / 14 | 0 | 0 |
| 9 | 148 / 26 | 126 / 26 | 0 | 0 |
| 10 | 152 / 34 | 146 / 34 | 0 | 0 |

NEMO is self-mirrored after every stage of every step.  The card's own
pivot-row update breaks the mirror from kt 1; only the end-of-stage fold
would remove it.

## 4. Score at `2cdf9d2afd` (both labels, kt 1..10)

Against round 47 (`5070601653`), row by row, 212 rows:

- given-NEMO-entry, all entries and two-step rows: **bit-identical**.
  From NEMO's restarts the overwrite moves 0 cells (kt 3, 8, 10 checked
  directly, with and without it).
- independent kt 1..6: bit-identical.
- independent kt 7..10: no status changes, no worst value changes.
  Cell counts move: 13 rows by +1..+30 cells, 6 rows by −1..−6.
  On the cells it moves, the overwrite is a last-place lottery (kt 10 T:
  126 exact with it, 156 without; S: 130 vs 156).  Largest move
  2.5e-14 absolute, under the bar.

This is the expected shape: the overwrite only acts on the card's
asymmetric chain, which NEMO never has.

Score SHA-256 (first 16) in `resume/manifest.sha256`; row diff `resume/cmp.txt`.

## 5. Test and plants (item 4)

A value test against NEMO's record cannot fail here: every NEMO entry is
self-mirrored, so the overwrite does nothing there.  The independent chain
first meets it at kt 7, and only below the bar.  So the input is
constructed.

New `test_g9_fct_pivot_row_right_half_is_the_image_of_its_left_half`: the
card's kt-1 stage-3 FCT inputs, pivot row's right half of both tracer
levels scaled by 1+1e-6; jitted FCT, tracers traced, x64.  The guess and
the antidiffusive U flux on the pivot row must equal the image of the left
half, with expected indices written from NEMO's loops, not the helper.

| copy (scratch worktrees, `resume/plants/`) | result |
|---|---|
| restored | 1 passed |
| T overwrite removed | failed on the guess |
| U overwrite removed | failed on the U flux |

## 6. Gates

Results in `gridladder_rounds/round47/resume/` (`JAX_ENABLE_X64=1`,
`land.sh` PYTHONPATH):

| battery | result |
|---|---|
| G9 file | see `g9_file.log` |
| default gate (land.sh list) | see `gate_*.log` |
| receipt citation gate, this receipt | see `receipt_gate.log` |

## Choices

ASKED: transcribe the half-row (item 1); the value-test route (item 4).

UNASKED, offered for revert:
- the half-row shares the existing G9 field rather than a new one;
- the U arm of the external-mode association now calls the shared helper
  (no jaxpr change);
- the test input is constructed, because no NEMO entry can make it fail.

UNVERIFIED: that the card's kt-1 mirror break starts in its stage-3
pivot-row update (not located this resume).
