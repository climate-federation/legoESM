# ORCA2 round 248 — reference reciprocal refuted; substep-2 exit reciprocal is first non-finite writer

Date: 2026-10-11. Frozen base: `02d3962d6`. Preregistration commit:
`16de99b07`. Final measurement commit: `db723fe61`. Candidate retraction:
`6d90b412c`. Status: **HELD**. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round248/`.

This round changes no effective package/model file, deck, selector, carried
state, stabiliser, sea-ice setting, or `unmeasured_features` entry. The exact
round-247 stage-mask candidate was re-applied only to measure the complete
private atomic unit, then removed. Every result is separately labelled
**independent** and **given NEMO's entry**; both labels produced identical
values. No NEMO acquisition was required.

## Compiled statements and source order

The executing record build constructs `hu_0`/`hv_0` from the associated
three-dimensional masks and stores mask-safe reference reciprocals at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/domain.f90:193-215`. Its live QCO
ratios multiply the surface height by those stored reciprocals at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/domqco.f90:207-218`. The external
substep then writes live face depths and their mask-safe reciprocals at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:712-718`.

The committed gate is
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round248_land_support_gate.py`.
It advances the ordinary private atomic candidate to its finite kt=5 entry,
then reads the pure frozen geometry/EOS bundle and the established 65-frame
barotropic scan trace. It inserts no callback or new arithmetic. CPU fp64/x64
and the production D114 resolved-mesh route are explicit.

## Frozen predictions

1. **R248-P1 — REFUTED.** Candidate and NEMO surface-mask predicates are
   bit-exact on all 26,640 U and 26,640 V columns. Every reference and live
   reciprocal is finite. At `[j=1,i=49]`, `hv_0=0`, candidate wet predicate,
   resolved `ssvmask`, reference reciprocal, and live reciprocal are all 0.
2. **R248-P2 — CONFIRMED AS A DIFFERENT WRITER.** The finite entry and frozen
   geometry/EOS bundle remain finite; every 3-D slow-forcing operand remains
   finite. The first non-finite trace value is the U exit-depth reciprocal at
   external substep 2, `[j=7,i=114]`, under both labels.
3. **R248-P3 — NOT ACTIVATED.** The candidate does not cross kt=5, so the
   kt=8/10 completion predicate is false.
4. **R248-P4 — NOT ACTIVATED.** OMT/rung/GYRE/DINO/tank/month landing gates
   cannot make a failed atomic prerequisite eligible. The candidate was
   removed and the final package diff against the frozen base is empty.
5. **R248-P5 — CONFIRMED.** Both-label, full-support, finite-entry,
   source-order, and writer plants each exit 2 with `PLANT-FIRED`.

## Measurements

The finite kt=5 entry is already explosive at the northern fold under both
labels: maximum absolute S is 36,038.4014800172 PSU at `[147,49,2]`, T is
1,949.1355638489615 K at the same cell, U is 72,020.748458079 m/s at
`[147,49,0]`, and V is 78,500.20607733581 m/s at `[147,48,0]`. SSH remains
finite with maximum 1.8769447886623258 m.

The frozen EOS/HPG intermediates are also finite: maximum absolute density
anomaly is 43,628,917,366.14404, pressure anomaly is
4,340,930,527,786.3716, and `rhd` is 42,523,311.27304488. No slow-forcing
field is non-finite before the external loop.

At the first writer, the direct operand row is identical under both labels:

| operand | value |
|---|---:|
| substep / compact cell | 2 / `[7,114]` |
| `hu_0` | 554 m |
| `ssumask` | 1 |
| face SSH | -554 m |
| live U depth | 0 m |
| live reciprocal | `+Inf` |
| previous-substep live U depth | 553.6813609810112 m |
| previous-substep reciprocal | 0.001806092945278495 m^-1 |

Thus the literal oracle statement itself divides by zero on the candidate's
already-explosive private state. NEMO's admitted kt=5 state is finite. The
reciprocal is the first non-finite writer, but it is downstream damage: adding
a floor or clip would be a stabiliser NEMO does not use, and changing the
reference reciprocal would contradict the exact full-support table.

## Landing disposition

Nothing lands in production. The reference-mask hypothesis is retained as a
refutation and the exact stage-mask/transport candidate is retracted again.
Because the final package tree equals the base, GYRE, DINO, tanks, rung 0,
rung 10, and the ORCA2 month retain their certified base trajectories by
construction; no claim is made from an unrun landing gate.

## Controls, tests, citations, and review

The round-248 focused gate tests pass 6/6. The classifier passes on both
labelled reports and all five named plants fire. The mandatory fidelity
battery and citation/review results are recorded in the final amendment below.

## OPEN

The origin precedes the substep-2 reciprocal: the kt=5 entry is finite but
already explosive at the northern fold. On the same complete private unit,
build the passive kt=1..4 stage/substep growth table against NEMO, find the
first boundary whose error grows by more than 10x, and replay that boundary in
compiled source order. Do not change the exit reciprocal, add a depth floor,
or continue from the global kt=5 NaNs. Decision 116 remains pending with the
user and is not re-emitted.
