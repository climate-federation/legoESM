# Grid ladder — G5 round 2 (gridladder_r32)

Status: **HELD. G5 is not certified.** The record is admitted (passivity
passes). Every ocean-read surface/ice field the card takes from NEMO's record
is bit-identical at kt 1..10. The first non-bit statement is the preregistered
one: the implicit ice-ocean top drag, which the card has no input for. Every
surface and all-level row is DEBT under both labels from kt 1. Nothing landed:
the fix is a new top-drag path (input + four consumers), not one statement.

Preregistration `PREREG_nemo_testcases_gridladder_g5_round2.md` (`ef4097de4`,
frozen after the passivity check, which the brief makes the gate, and before
everything else). Record `gridladder_rounds/round31/oracle_g5_r31`
(`G5_ACQUISITION_PASS steps=1..10`; reference stopped in `stp_ctl` at kt 11,
`|V| = 10.25 m/s`). Instruments `c24c154fa`. Evidence
`gridladder_rounds/round32/` (`manifest.sha256`): `g5_admission.json`
a18ae39aab3d5d51, `g5_forcing_identity.json` f96770cc244d4ea3,
`g5_top_drag_kt1.json` b344e97a5db97945, `g5_stages_kt1.json` 125e142bf1a070b0,
`g5_stage1_kt1.json` d50de3cc863324b2, `g5_independent_kt1_4.json`
6a13d59fb93b6790, `g5_independent_kt5_10.json` f23e3755d2c0e353,
`g5_given_13579.json` 27fee1b25e0118f3, `g5_tables.md` d227225225302915
(from `probe/summarize.py` 4da2a9d9c15c46b1).

## 1. Passivity (the reviewer's must-do) — PASS

`check_records.restarts_identical(ref_100step, twin_a, [10])`: all 38 restart
variables byte-identical at step 10, the reference's last common step with
twin A. Also identical: reference vs twin B at steps 2, 4, 6, 8, 10. Twin C's
odd steps have no reference restart to compare. Plant (one ulp on `sshn` in a
copy of twin A's restart) refuses: `PASSIVITY FAILED ... sshn differs`.

## 2. Admission — PASS

- 60 per-step frames (entry, surface, barotropic, stages 1-3; kt 1..10)
  byte-identical across twins A/B/C.
- NEMO's stage-3 frame of kt equals its entry frame of kt+1 (kt 1..9: ssh and
  level-1 T, S, u, v).
- Every restart 2..10 equals that step's stage-3 frame (all levels T, S, u, v;
  ssh) and barotropic frame (`uu_n`, `vv_n`).
- The run.sh admission did not include the passivity check; it is now in
  `g5_records.py admit` (exit 73 on refusal). No record was re-admitted.

## 3. Ice-ocean forcing identity, every step — 0 unequal

Card input (through the ORCA2 lane's route, exactly as G4) vs NEMO's post-SI3
record on the 16433 wet cells. Cell = unequal cells.

| field (card input vs NEMO record) | kt1 | kt2 | kt3 | kt4 | kt5 | kt6 | kt7 | kt8 | kt9 | kt10 | max abs, all kt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| emp | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0e+00 |
| fr_i | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0e+00 |
| q_net = round(qns+qsr) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0e+00 |
| qns | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0e+00 |
| qsr | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0e+00 |
| rnf | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0e+00 |
| rnf_tsc S | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0e+00 |
| rnf_tsc T | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0e+00 |
| sfx (card salt_flux vs 1e-3*sfx) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0e+00 |
| taum | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0e+00 |
| utau (native i) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0e+00 |
| utauU (mirror on NEMO T stress; instrument) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0e+00 |
| vtau (native j) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0e+00 |
| vtauV (mirror on NEMO T stress; instrument) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0e+00 |
| ice-covered wet cells (fr_i > 0) | 3630 | 3630 | 3769 | 3769 | 3883 | 3883 | 3918 | 3918 | 3949 | 3949 | |
| rCdU_ice nonzero on those cells | 3630 | 3630 | 3769 | 3769 | 3883 | 3883 | 3918 | 3918 | 3949 | 3949 | min -1.550e-03 m/s |

- U/V stress: the T->U/V statement mirrored on NEMO's own T stress reproduces
  NEMO's recorded `utauU`/`vtauV` on every wet face, kt 1..10 (instrument
  check; the card's inner T stress is NEMO's, and its own T->U/V statement is
  G4's certified one).
- Rule 11, my instrument: the first identity run reported `sfx` 474 cells
  unequal (my `1e3*(1e-3*sfx)` round trip) and U/V stress 1148 faces (mirror
  on halo-less arrays, then land faces NEMO stores as 0). Both were probe
  defects; the table is the fixed probe.
- Plant (`emp` + 1e-12 on one wet cell, kt 1) refuses.
- `snwice_mass` is not read by this ocean (`ln_ice_embd = F`, run log).

## 4. First non-bit statement (kt 1, stage 1)

The card's entry is NEMO's at kt 1 (ssh, T, S on all levels: 0 unequal).
Stage 1 is already DEBT: ssh 10059 cells (max 6.05e-3 m), u 9753 (2.30e-2
m/s), v 9828, T 9000, S 8983. Stage 3: ssh 10109 (1.81e-2), u 9668 (2.94e-1).

Locality of the stage-1 ssh error (chessboard distance to the nearest
ice-covered cell, `fr_i > 0`):

| distance (cells) | wet | unequal | max abs (m) |
|---|---:|---:|---:|
| 0 (under ice) | 3630 | 3630 | 6.05e-3 |
| 1-5 | 1695 | 1690 | 4.67e-3 |
| 6-15 | 3250 | 3114 | 4.33e-5 |
| 16-30 | 4522 | 1625 | 5.36e-7 |
| > 30 | 3336 | 0 | 0 |

The card's model config is identical to the certified G4 card's (field-by-
field diff: no difference); only the card labels differ.

NEMO's statement, in compiled order: SI3 sets
`ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/iceupdate.f90:403`
(`rCdU_ice = -r1_rho0*tmod_io*at_i*tmask`); `zdf_phy` copies it at
`ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/zdfdrg.f90:187`
(`rCdU_top = rCdU_ice`, called at
`ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/stprk3.f90:183`, before
`stp_2D`); `stp_2D` calls `dyn_drg_init` at
`ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/stp2d.f90:219`, whose first
ice-dependent statement is the external-mode drag coefficient
`ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/dynspg_ts.f90:1431-1432`.

- CONFIRMED (measured): NEMO's recorded coefficient at kt 1 is nonzero on
  3519 U and 3593 V faces (range -1.51e-3 .. -2.4e-10 m/s). An offline replay
  of :1431-1432 from the recorded `rCdU_ice` (with `rCdU_bot = 0`) equals it
  bit-for-bit on all 3519 U and 3525 V faces; the 68 fold-row V faces are not
  replayed (the record holds `rCdU_ice` before its fold exchange). The card's
  value is 0 on all of them. Plant (1e-9 on one ice cell) refuses.
- READ OFF THE CODE: the statements before it in `stp_2D`
  (`ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/stp2d.f90:140-166`: eos, hpg, vor, wzv, keg, zad) read only the entry
  state, which is bit-identical. Not separately measured this round.
- Why the error is large (READ OFF THE CODE, PLAUSIBLE as the whole cause):
  with `ln_drgice_imp = T` SI3 sets `zflagi = 0`
  (`ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/iceupdate.f90:405`), so
  the surface stress it hands the ocean
  (`ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/iceupdate.f90:417`)
  carries only the ice-velocity half of the ice-ocean stress. The ocean-velocity
  half is the implicit top drag. Without it the card's under-ice ocean has no
  drag, and errors reach ~1 m/s by kt 8.

Prediction 3 CONFIRMED. Prediction 4 CONFIRMED (not certified).

The same coefficient's other consumers, all absent from the card: the
baroclinic top-drag RHS term `ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/dynspg_ts.f90:1487-1488`, the implicit drag in
the substep loop, and `dyn_zdf`'s top-drag terms
(`ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/dynzdf.f90:175-177`, `ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/dynzdf.f90:312`).

## 5. Ladders, kt 1..10, both labels (scored at `c24c154fa`)

Cell = `status max_abs/unequal`; D = DEBT; bar 1e-15 normalized. T/S at kt 10
have no kt-11 entry frame; their all-level rows come from the kt-10 restart.

### INDEPENDENT  (cell = status max_abs/unequal)
| kt | ssh | uu_b | vv_b | u | v | T | S |
| 1 | D 1.81e-02/10109 | D 2.30e-02/9760 | D 3.59e-02/9840 | D 2.94e-01/9668 | D 3.34e-01/9739 | D 4.36e-04/9117 | D 1.48e-02/9104 |
| 2 | D 1.02e-01/14338 | D 7.80e-02/13833 | D 1.22e-01/14043 | D 5.43e-01/13485 | D 7.36e-01/13751 | D 2.26e-03/12596 | D 8.07e-02/12573 |
| 3 | D 1.88e-01/16111 | D 1.04e-01/15595 | D 1.14e-01/15688 | D 6.59e-01/15555 | D 7.85e-01/15680 | D 3.83e-03/15855 | D 1.30e-01/15848 |
| 4 | D 1.63e-01/16326 | D 1.09e-01/15776 | D 1.43e-01/15859 | D 7.94e-01/15755 | D 8.82e-01/15854 | D 4.59e-03/16245 | D 1.47e-01/16243 |
| 5 | D 2.33e-01/16419 | D 1.49e-01/15843 | D 2.07e-01/15935 | D 6.95e-01/15843 | D 1.08e+00/15931 | D 6.65e-03/16394 | D 1.53e-01/16394 |
| 6 | D 3.59e-01/16433 | D 1.75e-01/15853 | D 1.92e-01/15943 | D 9.30e-01/15853 | D 1.28e+00/15943 | D 1.16e-02/16429 | D 3.06e-01/16429 |
| 7 | D 3.56e-01/16433 | D 1.57e-01/15853 | D 1.77e-01/15943 | D 1.23e+00/15853 | D 1.08e+00/15943 | D 1.00e-02/16433 | D 3.14e-01/16433 |
| 8 | D 3.53e-01/16433 | D 2.23e-01/15853 | D 2.37e-01/15943 | D 1.30e+00/15853 | D 1.00e+00/15943 | D 1.20e-02/16433 | D 2.93e-01/16433 |
| 9 | D 3.28e-01/16433 | D 1.76e-01/15853 | D 2.33e-01/15943 | D 1.17e+00/15853 | D 9.04e-01/15943 | D 1.18e-02/16433 | D 3.06e-01/16433 |
| 10 | D 3.76e-01/16433 | D 2.23e-01/15853 | D 2.19e-01/15943 | D 9.82e-01/15853 | D 8.42e-01/15943 | — | — |
surface rows 68, DEBT 68, bit-identical 0
| region | rows | debt rows | worst normalized | at (kt field unequal/cells max_abs) |
| fold | 68 | 68 | 4.170e-02 | kt1 v 200/540 5.64e-02 |
| seam | 68 | 68 | 4.009e-02 | kt8 u 270/580 1.22e-01 |
| south | 68 | 68 | 3.550e-02 | kt2 v 46/528 7.11e-02 |
| interior | 68 | 68 | 4.272e-01 | kt8 u 15343/24992 1.30e+00 |
| interior_coast | 68 | 68 | 4.272e-01 | kt8 u 1571/3935 1.30e+00 |
| interior_open | 68 | 68 | 2.365e-01 | kt8 u 13772/21057 7.18e-01 |
| fold_pivot_row | 68 | 68 | 3.640e-02 | kt1 v 68/180 4.92e-02 |
| fold_row_below | 68 | 68 | 3.640e-02 | kt1 v 68/180 4.92e-02 |
| fold_third_row | 68 | 68 | 4.170e-02 | kt1 v 64/180 5.64e-02 |
| field/class | rows | debt rows | worst relative (kt, unequal/cells, level) |
| u full | 9 | 9 | 3.029e-01 (8, 397940, 397940, 1) |
| u partial | 9 | 9 | 1.122e-01 (8, 15090, 15090, 4) |
| v full | 9 | 9 | 3.670e-01 (2, 351157, 399994, 1) |
| v partial | 9 | 9 | 8.113e-02 (8, 15181, 15181, 4) |
| T full | 9 | 9 | 5.250e-04 (10, 414998, 414998, 1) |
| T partial | 9 | 9 | 2.650e-04 (6, 15551, 15554, 4) |
| S full | 9 | 9 | 9.789e-03 (10, 414998, 414998, 3) |
| S partial | 9 | 9 | 7.896e-03 (7, 15554, 15554, 4) |

### GIVEN-NEMO-ENTRY  (cell = status max_abs/unequal)
| kt | ssh | uu_b | vv_b | u | v | T | S |
| 1 | D 1.81e-02/10109 | D 2.30e-02/9760 | D 3.59e-02/9840 | D 2.94e-01/9668 | D 3.34e-01/9739 | D 4.36e-04/9117 | D 1.48e-02/9104 |
| 2 | D 1.02e-01/14338 | D 7.80e-02/13833 | D 1.22e-01/14043 | D 5.43e-01/13485 | D 7.36e-01/13751 | D 2.26e-03/12596 | D 8.07e-02/12573 |
| 3 | D 7.27e-02/10602 | D 5.57e-02/10260 | D 8.35e-02/10358 | D 3.01e-01/10149 | D 5.59e-01/10224 | D 2.43e-03/9762 | D 8.27e-02/9752 |
| 4 | D 1.38e-01/14879 | D 6.84e-02/14432 | D 1.30e-01/14526 | D 6.49e-01/14135 | D 1.02e+00/14373 | D 4.59e-03/13597 | D 1.06e-01/13573 |
| 5 | D 1.30e-01/10618 | D 6.37e-02/10297 | D 1.07e-01/10342 | D 4.39e-01/10141 | D 8.58e-01/10241 | D 4.37e-03/9782 | D 1.07e-01/9766 |
| 6 | D 2.19e-01/14911 | D 1.07e-01/14418 | D 1.11e-01/14563 | D 7.76e-01/14057 | D 1.50e+00/14358 | D 7.14e-03/13563 | D 1.88e-01/13546 |
| 7 | D 8.00e-02/10682 | D 2.75e-02/10364 | D 5.95e-02/10405 | D 4.02e-01/10204 | D 7.41e-01/10293 | D 3.03e-03/9827 | D 6.80e-02/9813 |
| 8 | D 1.43e-01/14984 | D 5.72e-02/14414 | D 6.34e-02/14624 | D 8.56e-01/14059 | D 1.18e+00/14371 | D 4.54e-03/13615 | D 1.02e-01/13587 |
| 9 | D 5.50e-02/10722 | D 1.06e-01/10384 | D 7.03e-02/10457 | D 3.93e-01/10265 | D 5.68e-01/10325 | D 2.80e-03/9877 | D 7.28e-02/9863 |
| 10 | D 1.28e-01/14971 | D 1.76e-01/14510 | D 1.57e-01/14620 | D 7.77e-01/14118 | D 7.57e-01/14392 | — | — |
surface rows 68, DEBT 68, bit-identical 0
| region | rows | debt rows | worst normalized | at (kt field unequal/cells max_abs) |
| fold | 68 | 68 | 4.239e-02 | kt10 u 198/540 2.10e-01 |
| seam | 68 | 68 | 3.460e-02 | kt2 v 198/580 6.93e-02 |
| south | 68 | 68 | 3.550e-02 | kt2 v 46/528 7.11e-02 |
| interior | 68 | 68 | 3.670e-01 | kt2 v 13307/24992 7.36e-01 |
| interior_coast | 68 | 68 | 3.670e-01 | kt2 v 1206/3935 7.36e-01 |
| interior_open | 68 | 68 | 2.791e-01 | kt8 u 12390/21057 8.48e-01 |
| fold_pivot_row | 68 | 68 | 3.640e-02 | kt1 v 68/180 4.92e-02 |
| fold_row_below | 68 | 68 | 3.640e-02 | kt1 v 68/180 4.92e-02 |
| fold_third_row | 68 | 68 | 4.239e-02 | kt10 u 63/180 2.10e-01 |
| field/class | rows | debt rows | worst relative (kt, unequal/cells, level) |
| u full | 5 | 5 | 1.997e-01 (8, 362416, 397940, 1) |
| u partial | 5 | 5 | 3.081e-02 (4, 13739, 15090, 4) |
| v full | 5 | 5 | 3.670e-01 (2, 351157, 399994, 1) |
| v partial | 5 | 5 | 3.636e-02 (2, 13291, 15181, 4) |
| T full | 5 | 5 | 2.371e-04 (6, 343336, 414998, 1) |
| T partial | 5 | 5 | 1.297e-04 (6, 12855, 15554, 4) |
| S full | 5 | 5 | 4.652e-03 (6, 342906, 414998, 3) |
| S partial | 5 | 5 | 4.684e-03 (6, 12837, 15554, 4) |

Every region (fold and its three rows, seam, south, coast, open interior) and
every depth class is DEBT under both labels; worst is the open/coast interior.
The kt-1 rows are the stage-walk numbers: the error starts at the ice edge.

## 6. Decision 122 — same statements in the ORCA2_OMIP_L4 build

Each line opened at its enclosing SUBROUTINE:

| statement | ORCA2_OMIP_L4 line | SUBROUTINE |
|---|---|---|
| external-mode drag coefficient | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynspg_ts.f90:1431-1432` | `dyn_drg_init` (:1404) |
| its call in the external mode | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stp2d.f90:219` | `stp_2D` (:63) |
| `rCdU_top = rCdU_ice` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfdrg.f90:187` | `zdf_drg` (:150) |
| `rCdU_ice` formed | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/iceupdate.f90:403` | `ice_update_tau` (:332) |

Identical text at identical lines (only `stprk3.f90` shifts, :183 -> :179).
No legoESM statement landed, so nothing moves on the full ORCA2 card.

## 7. Tests and review

- Focused G5 file, CPU fp64 (`JAX_ENABLE_X64=1`): 12 passed (3 new: the
  surface reader on a synthetic frame, admission + passivity plant, top-drag
  replay + plant). Planting a wrong halo cut in the reader fails the new reader
  test; restored and verified clean with `git status --porcelain`.
- G2/G3/G4 ladder fast tests: 47 passed, 11 slow deselected.
- Private-import + hardcoded-constant ratchets: 5634 passed, 1 failed — the
  pre-existing `tests/unit/test_jra55_do.py` row (file untouched here).
- Ladder edit: adds `ORCA2G5-ice` and lets G4/G5 run the stages arm with
  NEMO's forcing; G2/G3/G4 independent and given-entry paths unchanged.

REVIEW_PLACEHOLDER

ASKED: passivity first; admission; per-step ice-ocean forcing identity; both
ladders with region/depth splits; first non-bit statement with citation;
Decision-122 pointers; tests; one review.
UNASKED, offered for revert: the ladder's stages arm now accepts G4 (it
refused before) and passes NEMO's forcing for G4/G5.

UNVERIFIED: that the statements before `dyn_drg_init` are bit-exact at kt 1
(read off the code, not measured); that the missing top drag owns ALL of the
debt (one mechanism, measured only at kt 1 stage 1 by locality).

## OPEN

Next round: transcribe NEMO's implicit ice-ocean top drag (input
`rCdU_ice` from the record; the coefficient, its baroclinic RHS term,
the substep implicit use, and `dyn_zdf`'s top terms), G5-selected, default
off, by extending the existing bottom-drag path (searched: no top-drag
machinery exists in `packages/ocean`, only the unmeasured label).
