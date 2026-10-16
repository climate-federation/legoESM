# ORCA2 round 301 preregistration — kt=7 stage-1 U-transport factor split

Date: 2026-10-16. Frozen base: `c930ae521`. Scope: OMT-4, labels
**independent** and **given NEMO's entry** (round 300's two labels).
Round 300 named NEMO's stage-1 U transport `zFu` (differenced at
`divhor.f90:132-139`) as the first non-bit W-builder operand, wrong by
orders of magnitude in the fold band (last three rows). This round splits
that product into its factors at matched program points. It changes no
production package, card, deck, selector, carried state, stabiliser,
sea-ice field, runoff depth (D126) or after-SSH selection (D132).

## Resolved compiled statements

Record build `ORCA2_OMIP_L4_R299W7`, subroutine `stp_RK3_stg`:

- `BLD/ppsrc/nemo/stprk3_stg.f90:284`: `zub = un_adv*(r1_hu_0/(1+r3u(Kmm))) - uu_b(Kmm)`
  (np_LIN/np_HYB branch; `n_baro_upd = np_HYB` at `:51`).
- `BLD/ppsrc/nemo/stprk3_stg.f90:290`: `zFu = e2u*(e3u_3d*(1+r3u(Kmm)*umask)) * (uu(Kmm) + zub*umask)`.
- Stage 1 runs with `Kmm = Nbb` (`stprk3.f90:215`).

Same statements in `ORCA2_OMIP_L4` at `stprk3_stg.f90:276` and `:282`.

## NEMO factor sources (no new acquisition)

From the admitted round-299 record directory: `e2u`, `e3u_0`, `umask`,
`e1t/e2t/e1u` (rank mesh files); `uu(Nbb)` and `ssh(Nbb)` (kt-7 stage-0
rank frames, `stprk3.f90:108`); `un_adv` (kt-7 bt frame, written right after
`stp_2D`); `uu_b(Nbb)` (kt-6 bt frame `uu_b(Naa)`; stages 1-3 of kt 6 reassign
the same `ua_b`, `stprk3_stg.f90:141,157,238`). The bt frames are written by
rank 0 only (`IF(.NOT.lwp) RETURN`), so `un_adv`, `uu_b` and every factor
derived from them are scored on the western half (columns 0-89) only.
`hu_0` is rebuilt by `domain.f90:199,213`; `r3u(Kmm)` by
`dom_qco_r3c_RK3` (`domqco.f90:266-267`) from `ssh(Nbb)`.

## Frozen predictions and falsifiers

1. **R301-P1 — instrument closure.** NEMO's factors recomposed in NEMO's
   association reproduce the recorded `trp_u` bit-for-bit on every scored
   western-half wet face, fold band included. A miss anywhere is an
   instrument failure: no factor verdict is quoted on the missed cells.
2. **R301-P2 — static factors.** `e2u`, `e3u_0`, `umask` and rebuilt `hu_0`
   are exact between card and NEMO in the fold band. A non-bit static factor
   is the finding.
3. **R301-P3 — first non-bit factor.** Ranked by fold-band size relative to
   NEMO's own fold-band magnitude, the dominant non-bit factor is the
   carried step-entry U velocity `uu(Kmm)`; i.e. the fold debt enters the
   stage-1 product through the state carried from kt 6, not through the
   kt-7 product statement. Falsifier: `uu(Kmm)`'s fold-band max difference is
   below 10% of NEMO's fold-band `uu` maximum, or another factor's relative
   fold-band error is larger.
4. **R301-P4 — external correction secondary.** `zub` (and its operands
   `un_adv`, `uu_b`, `r3u`) differ in the fold band by less, relatively, than
   `uu(Kmm)`. Falsifier: `zub`'s relative fold-band error exceeds `uu`'s.
5. **R301-P5 — labels.** Both labels give the same table (they were
   array-identical in round 300). Disagreement refuses the claim.
6. **R301-P6 — measurement only.** If P3 holds, no kt-7 statement lands: the
   owner is upstream in time and the next walk targets the kt-6 producer of
   the fold-band U. Controls: an admission/coverage plant, a closure plant
   (one NEMO factor perturbed one ulp must break P1) and the one-bit control
   must refuse with a distinct plant code.

## Addendum (2026-10-16, appended after the kt-7 factor table, before the history below is measured)

The kt-7 factor table shows every dynamic factor already non-bit in the fold
band at stage-1 entry, and the production helpers for `:284`/`:290` replay
NEMO's `trp_u` exactly from NEMO's operands. To place the carried entry U in
time, the same traced run records the card's step-entry `u`, `ssh` and
`uu_b` at kt 1..7 and scores them against NEMO's stage-0 frames (and, for
`uu_b`, NEMO's kt-1 bt frame, west half), fold band versus interior.

7. **R301-P7 — onset.** Predicted: the fold-band entry U is exact at kt 1
   (admitted entry) and first exceeds 1% of NEMO's fold-band U maximum at
   kt 2 (the end of the first full step), well before kt 7. Falsifier: the
   first >1% kt is 5 or later.
