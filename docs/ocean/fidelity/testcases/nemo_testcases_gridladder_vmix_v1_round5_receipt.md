# Vertical-mixing lane — V1 round 5 (vmix_r5)

Status: **HELD. V1 is not certified.** The 118-cell owner is named, cited
and landed (V1-selected). At kt 1 the over-bar viscosity/diffusivity cells
fall from 120/119 to 2/1, and every scored kt-1 surface field (u, v, T, S,
ssh, `uu_b`, `vv_b`) is now AT-BAR. Both labels are still DEBT from kt 2.

Decision 129 is still pending; the as-built deck (NEMO's own TKE floors,
internal-wave mixing off) was used.

Preregistration: `PREREG_nemo_testcases_gridladder_vmix_v1_round5.md` at
`2edb67c4d`, committed before measuring.

Commits:
- `2edb67c4d`: preregistration;
- `6fcc5d378`: the landing (masked penetration, literal raw length) and the
  round-4 housekeeping;
- `18c3d4dec`: review fixes (two more refusals).

Record: `gridladder_vmix_rounds/round1/oracle_v1_r1` (admitted round 3).
Evidence: `gridladder_vmix_rounds/round5/` (`manifest.sha256`). The ladders
are stamped clean at `6fcc5d378efdf940b634a4870836ee262debdbe1`, packages
tree `6de06f7718189efcca4e0b81cdb5fc24de1f2633`. `18c3d4dec` changes only
raise paths that V1 never reaches (V1 sets `etau_mode="below_ml"` and
`tke_mxl_choice=3`; the V1 test file passes at it).

| file | sha256 prefix |
|---|---|
| `probe_closure.py` | 304aecdf03fae627 |
| `v1_mixing_length_dirty.json` (mask only) | ed97563336a990db |
| `v1_mixing_length_raw_dirty.json` (mask + literal raw length) | cbdef8587f4b4423 |
| `v1_independent_kt1_5.json` | c107137cd9a139aa |
| `v1_independent_kt6_10.json` | 43bd10bc77cfa74f |
| `v1_given_entries.json` | de09cbfc135636cd |
| `summary.txt` | f84bf5a211464f58 |
| `codex_review.log` | c2cada5272e3b097 |

`cap.pkl` (658 MB, the probe's capture) stays in the evidence directory only.

## 1. Housekeeping from the round-4 review (done)

- **The append step can no longer be deleted silently.** A new closure test
  feeds a step-entry bundle with a terminal row of 77 m and asserts both
  length calls receive `[1, 2, 3, 77]`. Planting the removal of the append
  turns it red.
- **The exact-equality sweep test** now runs on traced jit inputs and is
  skipped unless x64 is on.
- **The flag now refuses every path that would ignore it:** the
  implicit-solve stage, the post-mixing (Veros-order) path, and (review fix)
  a length choice other than NEMO's 3/4. Each refusal has a planted-red check.

## 2. The 118-cell owner: penetration energy below the seafloor (LANDED)

### Measurement (P1 CONFIRMED, P2 CONFIRMED)

`probe_closure.py` reran round 4's kt-1 mixing-length arm with a write-only
capture of every operand and result of the closure's length and coefficient
calls. The capture is passive: every row count equals round 4's
(`card_en_avm_replay_vs_card` 46,593 / 118 over the bar).

- **P1:** the closure's final viscosity equals the published `avm`
  bit-for-bit. The publish path is not the owner.
- **P2:** the closure's final length differs from the replay's by up to
  133.67 m. Of its four operands, rn2 and e3t are bit-identical and the
  surface anchor differs by one ULP in 117 columns. The **energy** differs in
  all **346,979 dry W cells**: the closure carries up to `8.43e-4` m²/s²
  there, while the published `en` is 0. Swapping only that operand moves
  exactly the 118 over-bar cells; swapping any other operand moves none.

So the card adds surface-penetration energy to dry W rows. NEMO's bottom-up
length sweep (`zdftke.f90:690-692`) then carries those large dry lengths up
into wet cells.

### NEMO statement

`ORCA2V1_OMIP_L4_R1FULLENTRY/BLD/ppsrc/nemo/zdftke.f90:498-501`
(SUBROUTINE `tke_tke`, :199), nn_etau=1:

`en(jk) = en(jk) + rn_efr*en(1)*EXP(-gdepw/htau)*MAX(0,1-zice_fra)*wmask(jk)*tmask(1)`

The card's penetration had no `wmask(jk)*tmask(1)`. That is a transcription
defect.

### What landed

- New `TKEConfig.nemo_etau_wmask` (default False). When selected, the
  increment is multiplied by the closure's dry-W mask and the surface tmask,
  after the ice factor. Both factors are exactly 0 or 1, so wet-cell bits are
  unchanged. Missing masks raise.
- In the same commit, V1 selects the existing literal raw-length arm
  `tke_mxl_raw_evaluation="nemo_literal"`: NEMO's
  `MAX(rmxl_min, SQRT(2*en/zrn2))`, `zrn2 = MAX(rn2, rsmall)`
  (`zdftke.f90:643-645`, `tke_avn` :531). `rsmall = 0.5*EPSILON(1.e0)` is
  binary64 because the build compiles with `-fdefault-real-8` (`phycst.f90:38`).
  It strictly improves every row and moves no over-bar count (table below).

kt-1 cellwise rows (414,119 wet W cells; unequal / over-bar / max abs):

| row | round 4 | + mask | + mask + literal raw length |
|---|---|---|---|
| card `avm` vs NEMO record | 114,690 / 120 / 3.41e-1 | 114,575 / 2 / 1.59e-14 | 99,800 / 2 / 1.59e-14 |
| card `avt` vs NEMO record | 109,902 / 119 / 1.24e-1 | 109,791 / 1 / 1.59e-15 | 95,369 / 1 / 1.59e-15 |
| card `avm` vs replay from its own `en` | 46,593 / 118 / 3.41e-1 | 46,464 / 0 / 4.4e-16 | **0 / 0 / 0** |
| NEMO-input replay `avm` vs record | 109,920 / 2 / 1.59e-14 | same | 93,453 / 2 / 1.59e-14 |
| `en` (kt-1 ZDF frame) | 35,577 / 0 / 6.94e-18 | — | 35,106 / 0 / 6.94e-18 |

Gates: `test_etau_wmask_leaves_dry_rows_and_wet_bits_as_nemo` checks wet
bits bitwise against the unmasked form, checks that dry rows are untouched
(with a red control), and checks the refusal; planting the dropped `tmask(1)`
factor turns it red. The V1 config-delta test names both new selections. The
selector check fires when either is planted back.

### Decision 122 (ORCA2_OMIP_L4)

`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90` is byte-identical to the V1 file
(round 4: both SHA-256 `0f290190…`), so the lines carry over:

| statement | ORCA2_OMIP_L4 line | SUBROUTINE |
|---|---|---|
| nn_etau=1 increment `* wmask(jk) * tmask(1)` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:498-501` | `tke_tke` (:199) |
| raw length `MAX(rmxl_min,SQRT(2*en/zrn2))` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:643-645` | `tke_avn` (:531) |

The full ORCA2 card is unchanged: its injection has the same defect
(pointer only, unmeasured there).

## 3. Both labels, kt 1..10

All-level wet cells: unequal / max abs. Every row is DEBT except where noted.
Dry `dissl` is now **0 unequal at every restart** (round 4: 346,979).

INDEPENDENT:

| kt | `en` | `avm_k` | `avt_k` | `dissl` (wet) | T |
|---:|---|---|---|---|---|
| 1 (ZDF frame) | 35,106 / 6.94e-18 AT-BAR | 99,800 / 1.59e-14 | 95,369 / 1.59e-15 | — | — |
| 2 | 124,256 / 6.40e-6 | 146,413 / 7.04e-5 | 144,587 / 1.15e-3 | 292,072 / 8.92e-7 | 137,035 / 3.32e-4 |
| 3 | 125,680 / 3.64e-5 | 148,908 / 2.90e-3 | 147,372 / 1.65e-3 | 319,354 / 2.45e-5 | 400,328 / 1.05e-3 |
| 4 | 169,893 / 1.28e-5 | 171,529 / 1.15e-3 | 171,790 / 1.32e-3 | 399,483 / 2.22e-5 | 427,363 / 1.58e-3 |
| 5 | 183,648 / 8.37e-5 | 175,360 / 4.24e-3 | 176,067 / 4.24e-3 | 412,806 / 3.64e-5 | 430,228 / 2.50e-3 |
| 6 | 191,362 / 1.28e-4 | 175,363 / 5.60e-3 | 176,205 / 5.60e-3 | 413,950 / 3.64e-5 | 430,470 / 9.72e-3 |
| 7 | 196,293 / 7.87e-4 | 177,354 / 1.62e-2 | 178,369 / 1.62e-2 | 414,060 / 1.42e-4 | 430,510 / 3.45e-2 |
| 8 | 199,077 / 4.88e-4 | 177,213 / 8.22e-3 | 178,392 / 8.22e-3 | 414,102 / 1.65e-4 | 430,546 / 1.06e-1 |
| 9 | 203,637 / 2.42e-3 | 178,367 / 5.25e-2 | 179,667 / 5.25e-2 | 414,102 / 2.34e-4 | 430,546 / 1.43e-1 |
| 10 | 207,290 / 8.10e-4 | 178,233 / 1.30e-2 | 179,606 / 9.06e-3 | 414,111 / 8.87e-4 | 430,551 / 1.59e-1 |

kt-1 surface rows (independent and given entry 1 share them):
- ssh, `uu_b`, `vv_b`: 0 unequal;
- u: AT-BAR, 3,494 unequal, max 9.99e-16 (round 4: DEBT 1.2e-3);
- v: AT-BAR, 3,470 / 4.16e-16;
- T: AT-BAR, 1,318 / 1.07e-14 (round 4: DEBT 1.2e-2);
- S: AT-BAR, 1,054 / 2.13e-14.

Compared with round 4 (same record, same windows): kt-2..10 `avm_k` maximum
falls from 0.459..0.606 to 7.0e-5..5.3e-2, and all-level T at kt 10 from
2.22e-1 to 1.59e-1. The kt-10 **level-1 u** maximum is 2.84e-2 versus 1.3e-2
in round 3: that row got worse, and it is not attributed.

GIVEN-NEMO-ENTRY (two-step projection; all-level, full-depth class shown as
in the log, partial in JSON):

| entry -> score | `en` | `avm_k` | `avt_k` | T |
|---|---|---|---|---|
| 1 -> 2 | 6.40e-6 (123,838) | 3.70e-5 (138,770) | 6.07e-4 (137,358) | 1.10e-5 (136,539) |
| 3 -> 4 | 4.92e-6 (126,864) | 1.31e-5 (141,609) | 2.14e-4 (140,748) | 3.13e-5 (409,833) |
| 5 -> 6 | 1.17e-4 (128,122) | 1.24e-3 (142,728) | 1.24e-3 (142,035) | 2.86e-4 (413,025) |
| 7 -> 8 | 8.46e-4 (129,785) | 1.25e-3 (144,245) | 1.25e-3 (143,896) | 4.20e-3 (412,602) |
| 9 -> 10 | 5.74e-4 (131,083) | 1.36e-4 (145,587) | 1.36e-4 (145,379) | 7.98e-3 (411,939) |

Round 4's given-entry `avm_k` maxima were 0.439..0.634; now 1.3e-5..1.25e-3.
The single-step surface T from NEMO's entry at kt 7 (2.57e-2) and kt 9
(2.37e-1) is unchanged from round 3. So those two steps belong to a different
owner (registered, not measured).

**V1 is NOT certified.**

## 4. Next owner (registered, UNMEASURED)

The 2 remaining kt-1 `avm` cells are (10,120) and (21,113), both at w-level 20.
NEMO's own-input replay misses them identically, so the owner is upstream of
the length.

At (10,120) w20: `en` is exactly `rn_emin` on both sides and N² = 7.52e-10.
The length there is the raw `SQRT(2*en/rn2)`, not bounded by the sweeps.
Inverting NEMO's recorded `avm` gives a length 22,363 ULP from the card's,
which needs NEMO's rn2 to differ from the card's by ~6e-12 relative
(~5e-21 absolute). A last-place N² difference that is AT-BAR in absolute
terms, amplified by 1/√N² in a near-neutral cell.

- **Statement:** `bn2_t`, `ORCA2V1_OMIP_L4_R1FULLENTRY/BLD/ppsrc/nemo/eosbn2.f90:1632-1640`
  (SUBROUTINE `bn2_t` :1604; `ORCA2_OMIP_L4` file byte-identical, same lines).
- **Ruled out:** the card's depth ladder and `e3w` are bit-identical to NEMO's
  `gdept_1d`/`gdepw_1d`/`e3w_1d` (the build is `key_vco_1d3d`); the EOS is
  EOS-80 on both sides.
- **Not ruled out:** the expansion coefficients (`rab_3d_t`, :1240 onward)
  and the T/S operands.
- **Discriminating measurement:** NEMO's rn2 (and `pab`) at kt 1. The record
  does not contain it. Recording it needs a record patch, so the next round
  either adds rn2 to the kt-1 ZDF frame (acquisition) or replays `rab_3d_t`
  offline.

## 5. Selection and caller grep (no-silent-default rule)

- `nemo_etau_wmask=True` and `tke_mxl_raw_evaluation="nemo_literal"` are set
  only in `_orca2v1_tke_vertical`. `_orca2v1_tke_selected` pins both.
- G1-G5, the full ORCA2 card, DINO and every library default are unchanged.
  The G5 card asserts `(nemo_mxlice, nemo_mxl_terminal_e3t, nemo_etau_wmask)
  == (0, False, False)` and raw length `factored`.
- **RULE 3 question (fix behind a default that keeps the old behaviour):**
  `nemo_etau_wmask` defaults to False (lane rule: shared flags default off
  until each card is certified). Default stays off (X) or moves to True for
  every NEMO-identity card (Y)? My pick: X until each card's own ladder
  measures it.

## Tests and review

All runs synchronous, CPU, fp64 (`JAX_ENABLE_X64=1`):

| suite | result |
|---|---|
| V1 file + TKE carried + identity + NEMO terms | 203 passed (130 s) |
| G2-ladder + G4 + G5, not slow | 51 passed, 11 deselected |
| TKE closure, dry-wmask, integration, N² before advection, nn_mxl2, post-mixing, prognostic, under-ice, dz slots, advection, EVD gate | 129 passed (292 s) |
| After the review fixes: the first suite + citation gate + private imports | 224 passed; the 2 gate failures were the clean-tree stamp refusal |
| Citation gate + private imports, clean tree at `18c3d4dec` | 23 passed |

No citation was shifted; no re-anchor was needed.

Single review (codex), `codex_review.log`: **"VERDICT: DO NOT SHIP"**, two
CONFIRMED findings, both fixed in `18c3d4dec` with planted-red tests:
- `nemo_etau_wmask=True` was silently ignored when `etau_mode="none"`; it now
  raises.
- `nemo_mxl_terminal_e3t=True` was accepted with length choice 1/2 and then
  unused; it now raises.

Neither finding touches V1's numbers.

## Choices made

- ASKED: housekeeping (task 1); naming the owner and landing it if it was a
  transcription defect (task 2); re-scoring both labels.
- UNASKED, offered for revert:
  - selecting the literal raw-length arm on V1, in the same commit as the
    mask. It is a cited statement and a strict improvement with no over-bar
    change.
  - the new refusal for the post-mixing path (an extra guard beyond the
    implicit-solve one asked for).

UNVERIFIED:
- the next owner (section 4) — no NEMO rn2 in the record;
- the full ORCA2 card's response to the mask (pointer only);
- the given-entry kt-7/9 owner.
