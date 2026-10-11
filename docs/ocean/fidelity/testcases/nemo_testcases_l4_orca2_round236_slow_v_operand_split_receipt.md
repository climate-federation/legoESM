# ORCA2 round 236 — OMT-4 fold-local slow-V operand split

Date: 2026-10-10  
Base: `956afff2723d40cc909d55e6cc6944212124d45b`  
Measurement commit: `93bc384eb4cd39a5ba3461401c64b352ce3c817d`  
Status: **HELD** (first upstream unit named; no production physics changed)

## Scope and labels

This round executes round 235's alpha-independent OPEN: split substep 1's
fold-local slow-V difference in NEMO source order. The frozen preregistration is
`docs/ocean/fidelity/PREREG_nemo_testcases_l4_orca2_round236.md`. Decisions 114
and 115 remain pending with the user and were not acted on. The shipped ORCA2
card, every sea-ice selector, and its `unmeasured_features` tuple are unchanged.

Every number is reported separately under **independent OMT-4** (legoESM's own
corrected initial state) and **given NEMO's entry OMT-4**. The results are
identical under the two labels; they were measured independently and never
pooled.

## Executing compiled statements

The vector-form OMT-4 build forms `Ve_rhs` from live V-face thickness,
completed three-dimensional V RHS, `vmask`, and `r1_hv_0` at
`ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/stp2d.f90:194-199`. It then applies
baroclinic drag and wind at
`ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/stp2d.f90:219-230`. The external
solver copies that field to `zv_frc` at
`ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/dynspg_ts.f90:289-294` and removes
the two-dimensional Coriolis term through `ssvmask` at
`ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/dynspg_ts.f90:323-328`.

OMT-1 through OMT-4 and the rung-0/rung-10 vector cards execute this source
path. This round scores OMT-4 only. No result is claimed for OMT-0's flux/linear
branch or for another card's trajectory.

## Record and instrument

No acquisition was needed. The gate reads the admitted OMT-4 twin's
`NEMO_L2_SLOW_2` and `NEMO_L2_BTSUB_2` streams under round 222, parses their
self-described headers, and requires exact defined payloads between twins.
Both labels run CPU, production JIT, fp64/libm. The existing completed-stage
side output is accepted only after traced and untraced T/S/u/v/SSH are
array-identical.

The owner support is defined by the already admitted final `slow_v` boundary:
35/13,320 cells unequal, all in the northern fold band, maximum
`1.6557659420864476e-06 m s-2` at `[147,54]`. Each side's own source statement
replays its completed depth mean and final slow forcing bit-for-bit. Thus the
table is an operand attribution, not a second executable.

Evidence SHA256:

- independent: `9cc7bb2af423c7ef8d323cf8b414b604d8cd8ecaeb67a1eb6627b04114bec043`
- given NEMO's entry: `10ae88a978bc7baa7662794a6e9dd259d918c0b4dd051b662fa1b49aeeb3726c`
- classification: `c037adc2185a4d5af427397af4ce1a8e64fa156fa9ea037d81eb7c79de6e007b`

All live evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round236/`.

## Source-ordered operand table

The table is identical for both labels. Three-dimensional operands are scored
over the 35 owner faces and all 30 physical levels (1,050 slots); two-dimensional
boundaries are scored over the 35 faces.

| source row | unequal / scored | maximum absolute difference | disposition |
|---|---:|---:|---|
| live `e3v_3d` | 382 / 1,050 | 499.9785217898161 m | first raw non-bit operand |
| completed V RHS | 668 / 1,050 | 2.3624915336282472e-06 m s-2 | second raw non-bit operand |
| `vmask` | 0 / 1,050 | 0 | bit-exact |
| `r1_hv_0` | 0 / 35 | 0 | bit-exact |
| completed depth mean | 35 / 35 | 1.6557659420864476e-06 m s-2 | debt |
| post-drag | 35 / 35 | 1.6557659420864476e-06 m s-2 | unchanged debt |
| post-wind / incoming `Ve_rhs` | 35 / 35 | 1.6557659420864476e-06 m s-2 | unchanged debt |
| pre-step Coriolis trend | 0 / 35 | 0 | bit-exact |
| raw `ssvmask` | 0 / 35 | 0 | bit-exact |
| final `zv_frc` / `slow_v` | 35 / 35 | 1.6557659420864476e-06 m s-2 | carried incoming debt |

The frozen Coriolis-unit prediction R236-P2 is **REFUTED**. The Coriolis trend,
raw mask, and final subtraction are exact; the difference enters before them.

## Cumulative replay and owner disposition

Replacing only live `e3v_3d` with NEMO's record leaves 35/35 depth-mean cells
unequal with the same `1.6557659420864476e-06` maximum. Replacing the completed
V RHS next closes the depth mean at 0/35 unequal. Mask and reciprocal are
already exact. Therefore the first raw non-bit operand is `e3v_3d`, but no
single-statement landing is justified: the first output-closing object is the
source-ordered **live-thickness + completed-V-RHS unit**.

At the downstream statement, replacing only incoming `Ve_rhs` closes final
`zv_frc` at 0/35 unequal; replacing Coriolis or mask afterward changes nothing.
R236-P3's predicted Coriolis/mask cancelling pair is **REFUTED**. The 35-cell
slow-V difference is carried entirely from the incoming depth average.

This unit overlaps Decision 114's separately pending exact geometry question,
but also contains an independent completed-RHS difference. Landing geometry
alone would not close this boundary, and coupling it to an unwalked RHS would
pre-empt that pending decision. No package change lands.

## Instrument retractions

Three refused attempts are retained, not promoted:

1. The first invocation used round 235's temporary raw-mask hook after that
   hook had been removed during production restoration. Construction refused
   before a model step or number.
2. The first operand table treated raw mesh thickness as the candidate operand
   on every owner face instead of applying the executing raw-vs-reconstructed
   fold selection. Its candidate-side replay did not close and is retracted.
3. The next checker read the producer field named `depth_v` as a forcing; that
   field is the water-column depth in metres. The units/replay gate refused it.
   The final gate reconstructs the actual depth-average statement from the
   certified operands and includes all 30 physical levels.

## Preregistered predictions

| prediction | disposition |
|---|---|
| R236-P1 | **CONFIRMED**: inherited streams admit, twin defined payloads are exact, and both traces are passive. |
| R236-P2 | **REFUTED**: the first raw unequal operand is live `e3v_3d`, not the Coriolis-removal unit. |
| R236-P3 | **REFUTED**: incoming `Ve_rhs` alone closes final `zv_frc`; Coriolis and raw mask are exact. |
| R236-P4 | **CONFIRMED**: both labels have identical support, counts, maxima, first operand, and cumulative closure. |
| R236-P5 | **CONFIRMED**: measurement only; final production-package diff is empty. |

## Verification

- focused gate tests: **4 passed**
- production measurement: **PASS_R236_SLOW_V_SPLIT** under both labels
- classification: **HELD_R236_SLOW_V_OWNER_NAMED**
- citation gate and rigid-shift plant: pending below
- `tests/ocean/fidelity -n 12`: pending below
- independent review: pending below
- production-package diff from base: empty

## OPEN

1. Walk the completed three-dimensional V RHS on the same 35 fold faces in
   NEMO accumulation order (HPG, LDF, VOR, KEG, ZAD) from the existing passive
   stage trace. Score an RHS-only replay as well as the cumulative
   live-thickness + RHS unit; do not land a partial operand.
2. Decision 114 remains pending with the user. If it authorizes the exact raw
   geometry separately, land and register that unit first, then remeasure this
   slow-forcing boundary; otherwise keep geometry private while walking the
   independent RHS debt.
3. Decision 115 remains pending with the user. Do not resume the global
   external-mode walk past substep 3 until its explicit filter-alpha field is
   authorized.

No acquisition is needed for item 1. The current passive stage/RHS record is
sufficient; if its fold rank lacks one named operator accumulator, report the
exact missing stream before writing a new acquisition.
