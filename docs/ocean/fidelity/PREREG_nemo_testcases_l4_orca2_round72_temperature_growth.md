# ORCA2 round 72 preregistration — independent temperature growth

Date frozen: 2026-09-28

Base: `5fcad2d985fe5dbc1b408ac532d435b7a6886159`

Claim label: **independent**. legoESM starts from the ORCA2 card's own ocean
state and consumes the admitted exact surface operands. No NEMO entry field is
substituted. NEMO's own from-rest restarts are the comparators.

Sea ice remains out of scope. The card's six-entry `unmeasured_features`
tuple, selectors, 10,800 s step, and carried state are frozen.

## Existing evidence, not predictions

Round 71 admitted the 480 rank-step surface frames and completed a production
CPU/JIT/fp64/libm 240-step ocean run. Its terminal all-cell temperature row is
non-bit (`21.637832697714707 K` maximum, `0.17564842520962015 K` unweighted
RMS). Those values are inputs to this round, not new evidence. Round 71 did
not locate or wet-mask that maximum and did not save an intermediate error
atlas.

The search-before-build audit found the reusable round-71 month runner and
restart reader, the round-43 source-boundary exposure, and the campaign's
volume-weighted RMS convention. This round extends those instruments rather
than adding a second trajectory implementation.

## Frozen metric and partitions

The temperature residual is legoESM minus NEMO on the shared 148 x 180 x 30
T-cell grid. All headline statistics use only the card's active T cells and
fp64 reference volumes `area_T * dz_ref`; dry cells are reported separately
and cannot own the wet-ocean ranking. The global statistic is
`sqrt(sum(volume * residual**2) / sum(volume))`.

The only full-domain exact-state checkpoints available without a new NEMO
run are frozen as entry (`kt=0`), the admitted two-rank step-10 restart, and
the admitted two-rank step-240 restart. The ranked intervals are therefore
`0->10` and `10->240`; no interpolated NEMO state is permitted.

Depth uses every one of the card's 30 native levels separately. Region uses
the supplied ORCA2 input file `subbasins.nc`, variables `atlmsk`, `pacmsk`,
and `indmsk`, restricted to wet cells. The gate requires their values to be
exactly binary, requires the three masks to be mutually exclusive on wet
cells, and adds the exact wet complement as `other`. The four regions must
partition every wet column exactly or the instrument refuses. No latitude,
longitude, depth, or value threshold is introduced.

For each interval, depth and region are ranked by increase in their
volume-weighted squared-error numerator. The report also records each bin's
end-point weighted RMS, wet volume, unequal count, and maximum. Negative
growth is retained, not clipped. Contributions must sum back to the global
squared-error numerator to fp64 reduction roundoff.

## Compiled source boundary

The admitted compiled NEMO program writes completed before fields `sshn`,
`un`, `vn`, `tn`, and `sn` at
`ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/restart.f90:170-184`,
after stage 3 and the `Nbb = Naa` swap at
`ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/stprk3.f90:223-271`.

At stage 1 the first compiled statement that changes a tracer accumulator is
the `tra_adv` call at
`ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/stprk3_stg.f90:633-643`;
surface sources follow at :645-651 and the QCO/RK assignment follows at
:670-680. The record exposes all three boundaries on rank 0. The source-order
walk compares entry T, after-advection T, after-SBC T, and stage-1 T on the
same recorded wet support.

## Frozen predictions and falsifiers

1. **Record/time-level calibration.** The two-rank step-10 restart predicts
   raw-bit equality with the admitted rank-0 stage-3 T frame on their common
   owned wet cells. A mismatch refuses all growth results.
2. **Execution replay.** Re-running the corrected round-71 month predicts the
   same terminal all-cell temperature row exactly. Any differing count,
   maximum, mean, RMS, first unequal index, or bit flag invalidates the run.
3. **Interval owner.** `10->240` predicts the largest positive increase in
   global volume-weighted temperature mean-square error. `0->10` ranking
   first is **REFUTED**, not reinterpreted.
4. **Depth owner.** Native level `k=0` predicts the largest positive total
   `0->240` volume-weighted squared-error contribution. A different level is
   **REFUTED** and becomes the measured owner.
5. **Region owner.** The supplied Pacific mask predicts the largest positive
   total `0->240` volume-weighted squared-error contribution. A different
   supplied-mask region is **REFUTED** and becomes the measured owner.
6. **Source-order owner.** Independent kt=1 entry T predicts bit identity;
   after-advection T predicts the first non-bit temperature boundary. If it
   stays exact, the walk proceeds to after-SBC and QCO/RK in compiled order.
   The first observed non-bit boundary, not this prediction, owns the next
   statement walk.
7. **Disposition.** This is diagnostic-only and predicts **HELD**. No physics,
   card, configuration, selector, threshold, stabiliser, or carried-state
   change is authorized.

Failed predictions remain **REFUTED** in the receipt. The interval, depth,
region, dry-cell, restart-calibration, and source-boundary rows are separate
predicates; none may waive another.

## Controls and validation

- Reuse the round-71 production step and forcing readers; run the 240-step
  model trajectory once.
- Plants must fire for a restart time-level mismatch, dry-cell inclusion,
  region overlap, interval-order mutation, temperature-boundary mutation, and
  one-ULP terminal score mutation.
- Run focused tests, both citation gates with a real firing citation plant,
  the 170-test card/tank battery, and the required read-only Codex review.
  No `packages/` change is authorized, so trajectory landing gates do not
  apply.

ASKED: rank the independent ORCA2 temperature-error growth by available exact
interval, native depth, and supplied basin region, then name the first
source-ordered non-bit temperature statement.

UNASKED: model arithmetic, configuration, forcing reconstruction, new NEMO
output, selectors, sea ice, and the held shared tracer QCO/RK candidate.
