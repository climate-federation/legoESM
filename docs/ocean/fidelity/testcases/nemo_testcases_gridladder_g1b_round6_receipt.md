# Grid ladder rung G1b — round 6 receipt (gridladder_r35)

Status: **LANDED; G1b remains DEBT**.  The first independent unequal value
before this round was localized to kt 7, stage 3, U cell `[130,113]`.  Its
source is a C2 kinetic-energy-gradient subnormal retained by NEMO's gradual
underflow and flushed by XLA CPU.  The G1b-only preservation makes that raw
and final stage value bit-identical.  The full rescore exposes a distinct
external-mode subnormal leave at kt 8 and retains the prior over-bar verdicts,
so G1b is not certified.

Preregistration:
`docs/ocean/fidelity/PREREG_nemo_testcases_gridladder_g1b_round6.md`
(`b4d2755c5`, before round-6 measurement).  Evidence:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round35/`.
Final scored artifacts were produced from clean commit `3f00fe185` in fp64 on
CPU.

## First unequal bit and carried-state decision

The frozen prediction R6-F1 is **REFUTED**: kt 1..6 are bit-identical, but kt
7 remains exact through stages 1 and 2.  The first unequal stage output is
stage 3 U cell `[130,113]`: NEMO is
`-7.767256946156943e-297`, the card is
`-7.767256946156942e-297`, one binary64 bit and
`1.35807730622e-312` apart.  Sea level, V, T and S remain exact there.

R6-C1 and R6-C2 are **CONFIRMED**.  At the kt-7 entry, every restart-held leaf
is bit-identical: T, S, U, V, sea level, `uu_b`, `vv_b`, and all six
barotropic-history arrays.  Of the leaves a restart does not hand the card,
only diagnostic W differs (`10,541` cells, maximum
`1.7040453360573697e-21`); all other represented leaves are exact.  Four
one-step arms—complete carried state, complete restart state, carried state
with all restart-held leaves replaced, and the inverse replacement—produce
the same pre-fix one-cell U leave.  The planted entry perturbation fires.
Thus no carried leaf owns kt 7; the leave is same-step arithmetic.

## Compiled-order stage walk

The corrected offline replay selects G1b's vector RHS and vector velocity
statement.  Its stage-2 and stage-3 controls both reproduce NEMO with zero
unequal output faces.  At kt 7 stage 3, in executed order:

| boundary | U / V unequal before | maximum at U `[130,113]` |
|---|---:|---:|
| HPG | 0 / 0 | 0 |
| relative-vorticity trend | 0 / 0 | 0 |
| C2 KE-gradient term | 100 / 99 | `6.279798654e-314` |
| completed RHS | 4 / 0 | `6.279798654e-314` |
| post-`dyn_zdf` raw velocity | 1 / 0 | `6.7768057580267e-310` |
| post-barotropic-correction velocity | 1 / 0 | `1.35807730622e-312` |

At the named cell, HPG and vorticity are exactly zero.  NEMO's KE-gradient
contribution is `-6.279798654e-314`; the ordinary card contribution is zero.
The source first forms and stores C2 `zu`, `zv`, and `zhke`, then subtracts
their metric-scaled gradient from `Krhs`: record build
`ORCA2G1_OMIP_L1_R4REF/BLD/ppsrc/nemo/dynkeg.f90:117-130`.  The enclosing
`SUBROUTINE dyn_keg` opens at :61 and ends at :175.  NEMO then consumes that
RHS in the vector update and removes the barotropic velocity at
`ORCA2G1_OMIP_L1_R4REF/BLD/ppsrc/nemo/dynzdf.f90:138-160`; the enclosing
`SUBROUTINE dyn_zdf` opens at :67.

R6-S1 is therefore **REFUTED as predicted and resolved by measurement**: the
first non-bit statement is stage-3 `dyn_keg`, not a stage-1 velocity update.
The ordinary term observer still truthfully reports XLA's flushed zero.  The
selected path evaluates the same C2 statements after an exact `2^500`
velocity scaling, retains the `2^1000`-scaled KE-gradient increment, and adds
it at `dyn_zdf`'s first normal-valued residual boundary.  The final division
by `2^1000` is exact.  Its boolean selector defaults false and only G1b selects
it; the path refuses non-C2, non-vector, non-NEMO-order, and multilevel use.

After landing, the stage-3 raw and final U/V rows are bit-identical.  Removing
only `rk3_vector_keg_subnormal_preservation` restores exactly the original
one-cell final U leave.  A portable compiled control independently proves
that ordinary XLA addition flushes the physical increment, the scaled addition
matches binary64 arithmetic, and its derivative with respect to the residual
is one.  R6-L1 is **CONFIRMED locally** and no exact GIVEN row is lost.

## Decision 122 full-ORCA2 source pointer

The full build's enclosing routine was opened, not inferred from the record
build.  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynkeg.f90:61` begins
`SUBROUTINE dyn_keg`, which ends at :175; the identical C2 statements are
:117-130.  Its enclosing
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynzdf.f90:67` likewise opens there, with the
same vector update and removal at :138-160.  The full ORCA2 card is not
selected by this G1b-only round because Decision 122 requires its own measured
execution chain.

## Both-label rescore

The GIVEN table excludes only kt-1 T/S land cells, the standing convention;
no wet or scored-field row is excluded.

| label / scope | round 5 (`e4ed232c0`) | round 6 (`3f00fe185`) |
|---|---:|---:|
| GIVEN entry frames 1..50, first over bar | none | none |
| GIVEN entry exact rows / unequal cells | 261 / 2,267 | **262 / 2,266** |
| GIVEN restart steps 3..100, first over bar | none | none |
| GIVEN restart exact rows / unequal cells | 305 / 5,034 | **306 / 5,033** |
| INDEPENDENT first unequal output | kt 7 U `[130,113]` | **kt 8 ssh `[90,152]`** |
| INDEPENDENT first over bar, ssh / u / v | 81 / none / 99 | **81 / none / 99** |
| INDEPENDENT worst ssh / u / v | `1.4407e-14` / `9.9028e-16` / `1.1367e-15` | **unchanged** |
| INDEPENDENT worst T / S | 0 / 0 | **0 / 0** |
| INDEPENDENT exact rows / total unequal cells | 170 / 3,216,622 | **171 / 3,216,621** |

R6-L2 is **REFUTED for certification**.  The new first unequal output is kt 8
sea level cell `[90,152]`, `1.913340147207e-312`.  A clean carried-state audit
shows both a complete card entry and a complete NEMO restart produce the same
five-cell sea-level leave.  The kt-8 stage record is already unequal at the
stage-1 external handoff, so this is a separate external-mode gradual-
underflow debt, not a surviving KEG error or carried-state owner.  It is the
next discriminating measurement; it is not hidden or waived here.

## Controls, tests, provenance, and choices

- Focused CPU tests ran synchronously with `JAX_ENABLE_X64=1`: the complete
  G1b card unit file **15 passed**; the scaled-add, carried-leaf, and kt-7
  record/removal controls **3 passed**.  The clean CLI removal artifact also
  restores the one-cell U leave.
- The carried arm has a direct every-leaf synthetic test and a planted entry
  perturbation.  The stage replay's executed vector predicate is pinned by the
  record-backed control.
- Search-before-build reused `_nemo_c2_kinetic_energy`, the shared C-grid
  gradients, `nemo_source_round`, the existing stage/raw observers, and the
  G1 ladder.  There is no duplicate model or scorer.
- independent review: operator (Claude).  No nested Codex review was run, per
  the round instruction; the operator reviews this diff before landing.
- Controlled comparison: one G1b-selected numerical transcription and its
  instruments changed; record, deck, timestep, masks, bar, precision and both
  label definitions are unchanged.
- Choices made: ASKED — locate the first unequal bit, compare all carried
  state, cite and land a demonstrated source defect, rescore both labels.
  UNASKED: none.

Final artifact SHA-256 (first 16): `carried_kt7.json` `06e07b3ebeb8aef8`;
`stages_kt7.json` `680ddbd42c60fa0c`; `stages_kt7_removal.json`
`00ad82fc2a485881`; `independent100.json` `8c34a8adfa035ff9`;
`given_entry50.json` `9fbf4cb963b3dce`; `restart_entry100.json`
`fa4c0e1f4ec37aa4`.

UNVERIFIED:
- the full ORCA2 card executes the cited statements but was read, not run;
- the exact external-mode statement responsible for the newly exposed kt-8
  sea-level subnormal has not been localized;
- G1b remains uncertified because INDEPENDENT ssh and V exceed the bar.

COMMITS_IN: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/gridladder_r35-2196213372
ACQUISITION_NEEDED: NONE
DECISION_NEEDED: NONE
ROUND_STATUS: LANDED — kt-7 KEG is exact; G1b remains DEBT on the INDEPENDENT label.
