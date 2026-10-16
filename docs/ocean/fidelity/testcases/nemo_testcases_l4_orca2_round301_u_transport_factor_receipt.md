# ORCA2 round 301 — kt=7 stage-1 U-transport factor split

Date: 2026-10-16 (Claude round; codex review: paused (quota)).

Status: **HELD**.  The kt-7 stage-1 U-transport statements are exact: the
card's own production code, fed NEMO's operands, reproduces NEMO's recorded
`trp_u` bit for bit on all 226,236 western wet faces (fold band 1,972 of
1,972).  The fold-band debt arrives through the inputs.  Every dynamic
factor is already non-bit at kt-7 entry; the static factors are exact.  The
card's fold-band entry U first leaves the bar at kt 2, together with the
interior.  CORRECTED after the independent review: measured relative to
NEMO's own signal in each region, the fold band is already far worse than
the interior at kt 2 (RMS error 12% vs 1.7%), then 59% vs 2.7% at kt 3 and
281% vs 4.8% at kt 6 (reviewer scratch; to be committed as a probe); only
the MAXIMUM jumps during step 6 (fold 0.256 -> 6.23 m/s; interior 0.519 ->
0.652 m/s).  Both labels are free runs from kt 1 with bit-identical kt-1
entries, so their runs are identical.  No package, card,
selector, carried state, sea ice, `unmeasured_features`, runoff depth (D126)
or after-SSH selection (D132) changes.

## Provenance

- Base `c930ae521`; frozen preregistration `725c8bf25`, addendum (R301-P7)
  appended before the history was measured, in `0248d2073`.
- Measurement tree `0248d207308683de97723e46946a0967114fd283`, clean, CPU,
  fp64, libm, production JIT; both labels, about 5 minutes each.
- Record: round 299's admitted
  `orca2_omt4_stage7_w_10step_np2` (the same `trp_u` round 300 scored), its
  rank mesh files, kt-1..7 stage-0 rank frames, and the kt-6/kt-7 bt frames.
  Deck and frame root: round 222's `orca2_omt4_frames_10step_a_np2`.
- Raw SHA-256: independent
  `ad93db3f3823d48aeb2835b4f522130863bc8d3d273885bc55d9451d7650ed63`,
  given NEMO's entry
  `d15033be4b973df831cb4f81c8c74f6c5d9d79fd07e40ecb2eeb01e902d5e46a`;
  classification
  `13b1dafa2b25aea2ceaf5d8aa2ce020fe20aaa57fd136fc3c42b526ebdecb986`
  (`HELD_R301_FOLD_FACTOR`).  Evidence: `orca2_rounds/round301/`.
- Later commits change only the instrument's structure (NEMO association
  pulled into one tested function) and tests.  On the final tree, the
  NEMO-only closure and replay still give 0 unequal of 226,236.

## Statements and NEMO sources

Stage 1 runs with Kmm = Nbb
(`ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/stprk3.f90:215`).  In subroutine
`stp_RK3_stg`, the external correction is
`ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/stprk3_stg.f90:284` and the product is
`ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/stprk3_stg.f90:290`.  The same lines in
the production build are
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:276` and
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:282`.  The time-mean
transport `un_adv` is accumulated and normalised in subroutine
`dyn_spg_ts` at
`ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/dynspg_ts.f90:570` and
`ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/dynspg_ts.f90:831`, identical in
production at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynspg_ts.f90:570` and
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynspg_ts.f90:831`.

Where NEMO's factors come from (no new acquisition):

- From the rank mesh files: the metric, the reference U thickness and the
  U mask.
- From the kt-7 stage-0 frame: the entry velocity and entry sea level.
- From the kt-7 bt frame: `un_adv`.
- From the kt-6 bt frame: `uu_b(Nbb)`.
- `hu_0` and `r3u` are rebuilt by NEMO's own formulas: the domain-init
  depth sum and reciprocal, and the stage-1 sea-level ratio.

The bt frames are written by rank 0 only, so every row is scored on
columns 0-89.  Round 300's fold argmax `[147,47-48]` is inside that range.

## Instrument checks

| Check | Result |
|---|---|
| NEMO factors, recomposed in NEMO's order, vs recorded `trp_u` | 0 / 226,236 unequal (fold band 0 / 1,972) |
| Card production helpers (jit) on NEMO's operands vs recorded `trp_u` | 0 / 226,236 unequal |
| Closure plant: one `e2u` raised one ulp | fires: 24 faces unequal, exit 2 |
| One-bit plant, label-disagreement plant | both fire, exit 2 |
| A plant that stays green | exits 3 with `PLANT-GREEN` (tested) |

## Factor table, fold band (last three rows), western half, kt 7 stage 1

The two labels produced identical tables, histories included (the
classifier refuses otherwise).  "Relative" means the fold-band maximum
difference divided by NEMO's own fold-band maximum.

| Factor (compiled order) | Unequal / support | Max abs difference | NEMO fold max | Relative | Interior max |
|---|---:|---:|---:|---:|---:|
| `un_adv` [m2/s] | 99 / 99 | 7825.04533613859 | 26.133002865194513 | 299.4 | 204.96704743420864 |
| entry sea level [m] | 99 / 99 | 0.5673251146752318 | 0.28152711786919493 | 2.015 | 0.12730778061892484 |
| `uu_b(Kmm)` [m/s] | 99 / 99 | 0.2129466492772054 | 0.0634497935229831 | 3.356 | 0.03758737700721003 |
| `zub` (as the product sees it) [m/s] | 99 / 99 | 1.7199805516153064 | 0.031762477455018825 | 54.15 | 0.12858907411258055 |
| `e2u` [m] | 0 / 99 | 0 | 93414.47523058299 | 0 | 0 |
| U mask | 0 / 8,100 | 0 | 1 | 0 | 0 |
| `e3u(Kmm)` [m] | 1,972 / 1,972 | 0.05064108395123412 | 496.699986337979 | 1.0e-4 | 0.02529392018107046 |
| entry U `uu(Kmm)` [m/s] | 1,972 / 1,972 | 6.233678382807669 | 0.08447270648030095 | 73.8 | 0.6515743557370255 |
| corrected U [m/s] | 1,972 / 1,972 | 6.514138667107963 | 0.09470227616299201 | 68.8 | 0.6525643944564721 |
| `zFu` [m3/s] | 1,972 / 1,972 | 79097748.84596531 | 504667.9334835084 | 156.7 | 25693183.08591891 |

In compiled order, the first non-bit factor is `un_adv`, from `dyn_spg_ts`.
It is a kt-7 product of an entry state that is already non-bit (entry sea
level and `uu_b`, rows above).  So it is not a statement defect that can be
named at kt 7.  Among the product's direct factors, entry U is the largest
(73.8 against `zub` 54.2 and `e3u(Kmm)` 1.0e-4).  CONFIRMED (measured).

## When the fold-band entry U goes wrong (R301-P7)

The card's step-entry state compared with NEMO's stage-0 frames, whole
domain for U and sea level, western half for `uu_b`:

| kt entry | U fold max diff | NEMO U fold max | U interior max diff | sea level fold / interior max diff | `uu_b` fold max diff |
|---:|---:|---:|---:|---|---:|
| 1 | 0 | 0 | 0 | 0 / 0 | — |
| 2 | 0.018327801035952572 | 0.0477326767252484 | 0.06639487446559043 | 0 / 0 | 0 |
| 3 | 0.09726837092324699 | 0.08153887495050562 | 0.1834617565645883 | 0.1275 / 0.2404 | 0.01963610565553299 |
| 4 | 0.11368139231513295 | 0.10344782336454528 | 0.4494542110558444 | 0.1504 / 0.1637 | 0.024778801630429323 |
| 5 | 0.1335607233135891 | 0.10249789875924162 | 0.631939740545014 | 0.1829 / 0.1490 | 0.05294695256194604 |
| 6 | 0.2560188339261232 | 0.12506486986862386 | 0.519019265612116 | 0.2430 / 0.1260 | 0.11242102613598812 |
| 7 | 6.233678382807669 | 0.13504175517826694 | 0.6515743557370255 | 0.5673 / 0.1349 | 0.2129466492772054 |

- After step 1, the 3-D U differs over the whole domain (203,314 interior
  faces), while sea level and `uu_b` are still exact.  This is the known
  kt-1 3-D velocity debt.  It is not specific to the fold.
- WITHDRAWN after the independent review: "the fold band breaks away during
  step 6".  Only its MAXIMUM U difference jumps in step 6 (24x, 0.256 -> 6.23
  m/s; interior 1.3x).  By relative RMS against NEMO's signal in each region
  the fold band is already 12% off at kt 2 (interior 1.7%), 59% at kt 3
  (2.7%) and 281% at kt 6 (4.8%) -- reviewer scratch, not yet a committed
  probe.  The fold departure therefore starts at kt 2-3, not kt 6.

## Prediction ledger

| ID | Verdict | Result |
|---|---|---|
| R301-P1 | **CONFIRMED** | NEMO closure 0 / 226,236, fold band included. |
| R301-P2 | **PARTIAL** | `e2u` and U mask are exact.  The card's reference thickness and `hu_0` are not exposed separately, so they are UNMEASURED directly.  The statement replay shows that NEMO's operands give NEMO's product. |
| R301-P3 | **REFUTED** | The largest relative fold error is `un_adv` (299x), not entry U (73.8x).  Entry U is the largest of the product's direct factors. |
| R301-P4 | **REFUTED** | `un_adv` exceeds entry U.  `zub` itself (54.2) stays below it. |
| R301-P5 | **CONFIRMED** | The labels give identical rows and histories. |
| R301-P6 | **CONFIRMED** | No kt-7 statement lands; every plant fires. |
| R301-P7 | **CONFIRMED** | Exact at kt 1; first above 1% at kt 2 (38%). |

## Tests and gates

- New tests: 8 passed.  Five planted breaks were each restored and checked
  with `cmp`: product association, `zub` without `uu_b`, a bt-frame halo
  offset, a dropped closure check, and first-factor order.  Each one gives
  `1 failed, 7 passed` (`test_breaks.log`).
- The refactor of round 300's traced step is behaviour-neutral: round 300's
  tests pass (4 passed).
- Citation gate at `0e4827e52`: this receipt PASS (9 citations, each at its
  pinned subroutine, 0 unmapped, 0 failing map entries); default receipt
  PASS (274).  A line plant on the `dyn_spg_ts` accumulation fails
  (`SYMBOL-NOT-AT-LINE`, exit 1), and so does a routine plant on the `zub`
  statement (`ROUTINE-CONTEXT-MISMATCH`, exit 1).
- The citation-gate tests, private-import ratchet, round 298-301 tests:
  `41 passed`.  The landing script's battery (`-n 12`): `144 passed`.
- GYRE ten-step / 30-day member: not required (no file under `packages/`
  changes).
- codex review: paused (quota).  The operator runs an independent Claude
  review before landing.

## Configuration choices

- ASKED: none.  UNASKED: none.  D126 and D132 are unchanged and not
  re-raised.

## OPEN

1. Commit a per-kt, per-region relative-error history probe (fold band vs
   interior, both labels, kt 1-7) to replace the review's scratch numbers.
2. Walk the EARLIEST fold-band departure, at kt 2 (U non-bit on 3,463 of
   3,933 fold faces and 203,314 of 409,097 interior faces), by stage, from
   the admitted stage frames; split the first growing stage's writer there.
3. Keep the complete fold unit private and atomic.
