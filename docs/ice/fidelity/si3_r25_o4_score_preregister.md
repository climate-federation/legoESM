# SI3 Ladder Round 25 (`si3_r25`) — O4 production-score preregistration

Date: 2026-10-29. Scoring base after the required lane rebase:
`0deec1acfcb0406fa1b2f690a4fbcfadc23d5e81`, whose lane parent is
`0d339a87e5c1d718a0b9aa4bb49f8fb8788562be`. The admitted record was produced
at the reviewed acquisition commit
`896c2acf024274db71e4b39b4cd8eaf8bd0620fd` and is read-only evidence at
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round25/oracle_o4_prather`.

No score from that evidence was run while preparing this document. The
operator-provided admission JSON was inspected as acquisition provenance, not
used as this round's production-score verdict.

## Frozen ladder bar

The O4 row of `docs/ice/fidelity/si3_ladder_design.md` defines five-category
Prather advection across the ORCA2 cyclic seam and north fold, including both
sweep orders and every Prather moment.

The exact score bar is:

> 0 ULP on every NEMO-written wet cell for every recorded state/residual exit
> at the four stages advection/zapsmall/lbc/step, seam and fold regions
> reported separately.

Signed-zero differences count as unequal and as one ULP. There is no
normalised-error tolerance and no Decision-127 subnormal floor in this local
statement score. The 55 Prather moment arrays recorded at the advection exit
are part of the advection stage. The recorded CFL values, subcycle count, and
sweep order are exact controls and must also agree.

## Exact production entry point scored (M4)

The scored production entry point is
`packages/ice/legoesm/ice/si3_orca2_state.py`'s landed `si3_o4_step`, with the
explicit O4 configuration (`jpl=5`, `nlay_i=10`, `nlay_s=5`, `nn_hls=2`,
Prather selected, prescribed `rn_uice=rn_vice=0.5`, thermodynamics off) and
the admitted record's grid, state, salinity, temperature, and delayed-CFL
inputs. This is the production O4 call chain
`si3_o4_step -> si3_dyn_orca2 -> si3_dyn_adv_pra_orca2`; no copied test
implementation is eligible for the verdict. Intermediate boundary scores use
the same landed production functions called by that chain, and the end-of-step
score comes from `si3_o4_step` itself.

The scorer must run CPU fp64/x64 in two modes:

1. eager, with the whole production scoring call inside `jax.disable_jit()`;
2. JIT, with that same production call wrapped by `jax.jit` where the admitted
   static record contract permits compilation.

The scorer itself must be committed before either mode is run. It must report,
for each step, record kind, and stage: scored wet scalar cells, unequal cells,
maximum ULP, and the first differing `(field, i, j, trailing indices)` cell or
`NONE`. Seam and fold rows must report the same quantities separately. The
receipt may aggregate fields only after preserving the first field/index in
the record schema's declared order.

## Frozen classification

`PASS_O4_EXACT` requires all of the following in both applicable modes and both
sweep steps:

* every full wet, wet seam, and wet fold row has `unequal=0`, `max_ulp=0`, and
  `first_differing_cell=NONE` at advection, zapsmall, lbc, and step;
* every advection moment row is exact under the same regional split;
* CFL, delayed CFL, subcycle count, and sweep-order controls are exact;
* eager and JIT output summaries are identical; and
* the scorer's one-wet-bit plants still turn the corresponding stage red.

`O4_DEBT` is any other result, including one signed zero, one unequal wet cell,
an empty required seam/fold region, a mode disagreement, or a missing scored
record. On debt, the receipt must name the earliest stage, field, and cell and
attribute the first differing production statement as `CONFIRMED` only when a
direct statement-level discriminator proves ownership; otherwise it is
`PLAUSIBLE`. No model fix is in scope for this round.

## Frozen evidence and reporting scope

All 18 admitted record files are required. Kinds 1--4 are the validated input
contract (grid, state entry, moment entry, CFL/control); kinds 5--9 are scored
as advection state, advection moments, zapsmall state, post-LBC state, and
end-of-step state. Both step 1 (`x-y`) and step 2 (`y-x`) are mandatory. No
record, field, wet cell, seam cell, fold cell, or mode may be waived after
measurement.

Post-rebase provenance amendment: the historical acquisition commit above is
superseded by the exact acquisition patch content retained on the landing lane.

PROVENANCE-SUPERSESSION: 896c2acf024274db71e4b39b4cd8eaf8bd0620fd -> blob 4f5f458723219d8afdd09973d33afd13c8f5f1d1 `scripts/validate/ocean_fidelity/testcases/nemo_si3_o4_orca2_prather_acquisition/icedyn_adv_o4.patch`
