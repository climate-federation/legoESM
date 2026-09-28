# Round 90 preregistration: stage-one barotropic-correction operand split

Date: 2026-09-14

Frozen production commit: `3f30fbea4433d78682a22b6657f84cf0c07df743`

Evidence root: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round90/`

## Inherited boundary and magnitude

The production baseline remains the landed Round 85 arm.  Its certified kt2
T/S/U/V maxima are `1.4210854715202004e-14`,
`2.1316282072803006e-14`, `2.7377110452773967e-12`, and
`3.284922138989399e-12`; kt3 T/S are `1.627497246303733e-4` and
`6.327735185607253e-6`; day-30 T is `1.2397011295506804e-2 K`.

Round 89 proved the vector RK3 assignment bit-exact on every recorded NEMO
input but measured the independently executing stage-one completed U/V at
`1.2184073888699132e-10` / `1.6794210466741788e-10`.  That is the first
unresolved statement boundary after the locally exact assignment and precedes
the stage-two growth to `6.920627565571227e-6` / `1.0321290368606531e-5`.
The held Kaa/W/ZAD/assignment bundle remains unapplied and is not a baseline.

## Compiled source and executing branch

The cited program is the admitted record producer at
`cfgs/GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo`.

* `stprk3_stg.f90:666-675` selects the executing vector branch and writes the
  raw Kaa velocity from Kbb plus `rDt*Krhs`, followed by the three-dimensional
  face mask.
* `stprk3_stg.f90:734-741` forms `zub/zvb` as the separately prognostic
  `uu_b/vv_b(Kaa)` target minus the ascending-level reference-thickness
  velocity sum multiplied by `r1_hu_0/r1_hv_0`.
* `stprk3_stg.f90:757-760` adds that correction at every wet level and applies
  the same three-dimensional masks.
* `dynspg_ts.f90:765-780,797-804` accumulates and finalizes the boxcar
  `puu_b/pvv_b(Kaa)` target that the correction consumes.  The resolved
  namelist selects `ln_dynadv_vec=.true.` at `namelist_cfg:161`, so the
  velocity accumulator at `dynspg_ts.f90:767-769` is live and the transport
  arm at `:770-777` is dead.
* `domain.f90:212-214` materializes the reference-depth reciprocals used by
  the correction.

This registration names an observed boundary, not a cause.  The operand split
will decide whether the live raw Kaa velocity, external-mode target, weighted
mean, correction, or final add is the first new non-bit statement.

## Reuse and instrument contract

The repository search found the single shared
`rk3_stage_barotropic_correction`, its fixed-order level reduction, the
Round-51 live-stage trace, the Round-89 stage walk, and the Round-46 named
stage records.  The instrument will extend the existing live trace with the
already-computed raw stage velocities and diagnostic correction operands.  It
will not add another barotropic correction or record reader.

For kt2 stage one the ordered rows are:

1. Kbb velocity and completed stage-one RHS;
2. raw Kaa velocity immediately after the vector assignment;
3. every reference face-thickness term and the ascending level sum;
4. stored `r1_hu_0/r1_hv_0` and the resulting diagnosed own mean;
5. separately prognostic `uu_b/vv_b(Kaa)` target;
6. `zub/zvb` difference; and
7. the masked final U/V add.

Direct record payloads are used for raw Kaa, target, `zub/zvb`, and final U/V.
A derived row is labelled `DERIVED`, never discharged by itself.  A separate
given-NEMO-input replay must reproduce recorded `zub/zvb` and final U/V with
zero unequal cells before any live residual is trusted.  The trace observer
must prove returned ordinary T/S/U/V/SSH are unchanged.  One nonzero ULP is
planted in a nonzero wet target cell; it must print `PLANT_FIRED` and exit
nonzero.  The commit stamp is fail-closed.

## Frozen prediction and falsifiers

The inherited kt2 Kbb U/V gap is already `2.7377e-12` / `3.2849e-12`, so it is
not eligible to be renamed by this split.  The prediction is that the raw Kaa
row remains in that magnitude class, while the separately accumulated
barotropic target is the first magnitude-bearing operand: its maximum gap is
at least four times the corresponding raw Kaa maximum on both faces and it
accounts for at least 75 percent of the final post-correction maximum.  The
given-input correction replay is predicted bit-exact.

The target-owner prediction is **CONFIRMED** only if the instrument calibration
and observer controls are exact, the plant exits nonzero, both target/raw
ratios are at least four, and substituting only NEMO's recorded target into
the live raw/weight correction removes at least 75 percent of the final U and
V maxima.  It is **REFUTED** if any calibration row is non-bit, an earlier
weighted-mean statement exceeds the target boundary, either ratio is below
four, either substitution closure is below 75 percent, or the plant exits
zero.  A refutation remains in the receipt and hands the measured first
non-bit statement forward.

If and only if the split names a source statement whose shared implementation
is non-bit and a NEMO-given-input version of that statement is bit-exact, test
the smallest source-cited correction.  Do not retry any held Kaa/W/ZAD member
unless this new statement is locally exact and explicitly paired with it.

## Rule-12 disposition

Any production candidate must first pass its local exactness proof and plant,
then the complete 954-row GYRE kt=1..10 comparison against Round 85.  No AT-BAR
row may leave the bar, first-over-bar may not move earlier or gain a field, and
every moved row is registered.  Only an admissible ladder reaches the frozen
days 1--30 run and scorer.  kt2 U/V, kt3 T/S, and day-30 T must improve in the
registered directions above.

LOCK_EXCHANGE and OVERFLOW execute the shared WS-RK3 correction and therefore
require their full certified tank lanes for a production change.  DINO uses
MLF, not this RK3 stage statement, but any shared helper or held WZV member is
an explicit DINO risk and must be scored before a neutrality claim.  ORCA2 is
`UNMEASURED-WITH-SPEC`: prove its selected integrator, then align raw Kaa,
reference-thickness reduction, reciprocal, external target, correction, final
add, restart state, and kt1..10 T/S/U/V/SSH on native masks.

No configuration, coefficient, timestep, threshold, carried state, restart
contract, year harness, reconciliation gate, freshwater pair, #1484 guard,
NEMO source, or NEMO executable changes are authorized or made in this round.
