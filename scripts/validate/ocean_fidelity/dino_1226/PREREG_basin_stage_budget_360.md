# PRE-REGISTRATION — the accumulated stage budget of the southern basin over a full year

Written BEFORE either run started. Companion to
`PREREG_basin_seasonal_decomp.md` (the offline phase, already scored).
Lane: `fidelity/dino-basin-budget-1yr`.

## What phase 1 changed about this run's target, and why that is disclosed here

The offline decomposition (`basin_seasonal_decomp.py`, output
`/tmp/bsd_run.log`) found that the −0.95 Sv southern-basin transport gap at
day 360 is the residual of a large near-cancellation:

| day | depth-mean part [Sv] | shear part [Sv] | net [Sv] |
|---:|---:|---:|---:|
| 90 | −0.628 | +0.214 | −0.426 |
| 270 | −2.285 | +2.200 | −0.057 |
| 360 | −2.415 | +1.480 | −0.952 |

Both parts clear their own ensemble floor by 4–9×. legoESM's depth-mean
southern circulation is a fixed 27–41% too strong from day 90 onward; its
shear is 8–14% too strong.

**The instrument this run uses reduces to the DEPTH-INTEGRATED row
circulation.** It therefore measures the depth-mean part — the LARGER of the
two and the one that appears first — and is BLIND to the shear part, which
owns 85% of the day-270→360 compounding. That limitation is registered here in
advance so it cannot be presented afterwards as a result. Naming the shear
owner needs a per-level accumulator that does not exist; building one is not in
this lane.

## The runs

* **legoESM**: `southern_term_torque_accum.py --days 360 --interval-days 10`,
  shipped `nemo_dino_kamm_mlf` card, `LEGOESM_NEMO_E3T=both`, fp64, from
  `DINO_00005760_restart.nc`. Measured cost 3.5 s/step → ~11 h on one GPU.
* **NEMO**: the trddump-instrumented oracle, `RUN_ACC90`'s namelist with
  `nn_itend` extended from 8640 to 17280 and nothing else changed.
* Both are the SAME restart and the SAME 2700 s step as the verdict run.

## Gate that runs FIRST and can veto the long run

**T1 trajectory identity.** A separate 10-day accumulator run must reproduce
the verdict run's member-0 day-10 state under the recorded row-circulation
reducer, to fp64 roundoff. If it does not, the budget is being taken on a
different trajectory from the one whose gap is being explained, and the
360-day run is discarded rather than reported. This runs concurrently on the
second GPU; a failure kills the long run at ~35 minutes rather than 11 hours.

## Predictions, registered before the numbers exist

* **P1** — The southern rows' accumulated stage table will show a single stage
  carrying a lego-minus-NEMO surplus of the sign that strengthens the
  depth-mean circulation, present in EVERY 10-day window rather than switching
  on late. CONFIRM: one stage carries ≥60% of the year-total difference and its
  per-window sign is constant in ≥30 of 36 windows. REFUTE: no stage exceeds
  40%, or the sign alternates.
* **P2** — The per-window difference will be roughly CONSTANT in time rather
  than growing, because phase 1 measured a saturating ratio and a saturating
  Δ depth-mean. CONFIRM: the last-quarter window mean is within ±50% of the
  first-quarter window mean. REFUTE: it grows by more than 2×.
* **P3** — *No prediction* on which of BARO / BCLIN / ZDF-bt / ZDF-bc it is.
  The recorded 90-day arms put the whole lego-minus-NEMO barotropic gap at
  −1.0 to −1.4 m³/s² on every arm, but on a DIFFERENT card
  (`transport_avg`, `off` ladder) and a different ladder, so quoting them as a
  prediction here would be a protocol confound.
* **P4** — The day-270 window will NOT be special in this table. Phase 1
  showed the day-270 near-zero is a cancellation between two parts, only one
  of which this budget sees. CONFIRM: the day-260→270 window's difference is
  within the interquartile range of all 36. REFUTE: it is an outlier, which
  would mean the depth-mean part does have a mid-year feature the transport
  decomposition missed.

## Known limitation, restated so it is not re-discovered as a finding

NEMO's depth-integrated circulation after a step is set ENTIRELY by its
barotropic solve; legoESM keeps part of the vertical solve's barotropic
deposit. The two models' stage tables are therefore NOT row-comparable stage by
stage. The comparable pairing is REALIZED-vs-REALIZED, as
`nemo_accum_torque.py` established. Any stage-by-stage lego-minus-NEMO number
is labelled CONTAMINATED, exactly as that probe labels it.
