# Preregistration: NEMO-testcases L2 GYRE round 92 Round-89 damage bisection

Date: 2026-09-14. Frozen at incoming tip `dc62e1838499` before applying any
held candidate, repairing the Round-90 record reader, or running a Round-92
scientific comparison. Evidence belongs under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round92/`.

## Question and magnitude rank

Decision 40 refuses source-proof landings that fail the trajectory gate and
orders the held union to be bisected for damage one member at a time. This
round tests only the Round-89 held Kaa/W/WZV/ZAD/source-assignment bundle on
the restored Round-85 production implementation. Round 91 measured the union
at exactly Round 89's six headline ladder values even after adding the Round
60 TKE member. Round 89 is therefore the first and largest already-observed
damage candidate; Round 60 remains held and untested alone at this tip.

The immutable Round-85 comparator has kt2 T/S/U/V maxima
`1.4210854715202004e-14`, `2.1316282072803006e-14`,
`2.7377110452773967e-12`, and `3.284922138989399e-12`; kt3 T/S maxima
`1.627497246303733e-4` and `6.327735185607253e-6`; and day-30 T RMS
`1.2397011295506804e-2 K`.

## Compiled source and candidate boundary

The compiled Round-90 GYRE vector branch assigns stage-one Kaa U/V from Kbb
plus `rDt*Krhs`, then masks, at
`GYRE_OMIP_L2_P3_SM_R90BARO/BLD/ppsrc/nemo/stprk3_stg.f90:668-687`.
It then computes the depth-mean correction from current `uu_b/vv_b(Kaa)`,
reference thickness, and reference reciprocal, and adds it to every active
level at
`GYRE_OMIP_L2_P3_SM_R90BARO/BLD/ppsrc/nemo/stprk3_stg.f90:728-765`.
The target `uu_b/vv_b(Kaa)` is accumulated and normalized by the executing
external solver at
`GYRE_OMIP_L2_P3_SM_R90BARO/BLD/ppsrc/nemo/dynspg_ts.f90:763-804`;
the reference reciprocals are constructed at
`GYRE_OMIP_L2_P3_SM_R90BARO/BLD/ppsrc/nemo/domain.f90:195-215`.

The Round-89 manifest is applied whole and alone because it is the held-union
member named by Decision 40. Its given-NEMO-input local assignment and
Kaa/W/WZV/ZAD proofs must still report zero unequal cells, and every plant
must still exit nonzero. No part is eligible to land this round.

## Record-reader repair and operand discriminator

The already acquired Round-90 record is self-describing. Its global header is
`36x26x31`, while the writer's assumed-shape `zub/zvb` arguments are the
compiled owned correction arrays and declare `32x22`; other rank-two arrays
declare `36x26`, and rank-three arrays declare `36x26x31`. The existing reader
incorrectly requires global extents for every rank-two payload. Extend that
reader, not NEMO or the record, with a per-field extent contract derived from
the committed writer. Preserve magic, header, exact field inventory, finite
value, producer-commit, digest, and final-add checks. Add a synthetic mixed-
extent record and prove a wrong owned extent fails. The recorded one-ULP final
add plant must exit nonzero.

After admission, extend the existing Round-90 split (not a new implementation)
to compare the Round-89 candidate's live seeded kt2 stage-one operands in
compiled order against the direct record: Kbb, Krhs, raw Kaa, reference
thickness, reference reciprocal, target, correction, and final U/V. The
given-NEMO-input correction replay must be bit-exact or the instrument is
refuted. The first direct live non-bit operand is the reason a locally exact
assignment can disagree on the trajectory; downstream rows are not owners.

## Frozen predictions and falsifiers

1. The Round-89 member alone reproduces its prior damage: kt2 T/S/U/V
   `3.0652394795183113e-3`, `4.835710022078388e-3`,
   `6.733005735178965e-7`, `1.3183569256688065e-6`; kt3 T/S
   `1.566544749833554e-2`, `4.324733116938262e-3`. First-over-bar moves from
   kt2 U/V to kt2 T/S/U/V. Any different printed value refutes this numerical
   prediction and is retained.
2. The repaired acquired-record gate reports `READY`, its ordinary final-add
   replay is bit-exact on U and V, its final-ULP plant fires with nonzero exit,
   and a synthetic wrong owned extent is rejected. Any miss refutes admission.
3. On NEMO's own recorded assignment/correction inputs, the Round-89 shared
   helpers remain bit-exact. On the independent live candidate, Kbb remains
   exact and the first non-bit direct input is the stage-one momentum RHS
   (`Krhs`), before raw Kaa and the correction. If Kbb is non-bit, Krhs is
   exact, or the NEMO-given-input replay is non-bit, this operand prediction is
   refuted and no cause is assigned beyond the measured first boundary.
4. Because the ladder is predicted to lose AT-BAR kt2 T/S, the candidate is
   held. Days 1--30 are nevertheless measured as Decision 40's damage rank;
   day-30 T RMS is predicted to equal Round 91's union value
   `1.2589964172802363e-2 K`. A different value is retained as a refutation.

## Rule-12 and card dispositions

| lane | frozen disposition |
|---|---|
| GYRE local source rows | Re-prove every Round-89 assignment and Kaa/W/WZV/ZAD row at zero unequal; every inherited plant exits nonzero |
| GYRE kt=1..10 | Compare only the Round-89 member against Round 85; register every moved row; expected refusal from new kt2 T/S debt and earlier first-over-bar |
| GYRE days 1..30 | Measure damage against the immutable Round-85 day-gap arm; no result can override the failed ladder |
| LOCK_EXCHANGE-zco and OVERFLOW-zps | No landing is possible; retain Round-91 union evidence that the superset moved zero certified tank rows, and make no stronger single-member claim |
| DINO | The W/WZV/ZAD helper is shared; numerical neutrality is not claimed, and the 96--98% regional cancellation warning remains in force |
| ORCA2 | **UNMEASURED-WITH-SPEC:** select its compiled integrator and independently align Kaa/Kbb SSH, WZV operands/carries, ZAD, source assignment, correction operands, restarts, and kt1..10 T/S/U/V/SSH on native masks; reject missing state, any wet-input mismatch, AT-BAR loss, or earlier first-over-bar |

No configuration, coefficient, timestep, stabilizer, carried-state policy,
year harness, reconciliation gate, freshwater pair, #1484 guard, canonical
NEMO source, or NEMO executable changes. The candidate is restored after
measurement. The only lasting code change may be the acquired-record reader,
its diagnostic split, tests, manifest-header evidence, and receipt.
