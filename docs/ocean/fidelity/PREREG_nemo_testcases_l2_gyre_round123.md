# Preregistration — NEMO testcase L2 GYRE round 123 day-240 process budget

Date: 2026-09-19

Incoming lane tip: `d6765dd6fd740b6e105a990c491cee44e450b093`

This document is frozen before any Round-123 process-boundary measurement.
Evidence will live under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round123/`.  Round 122
localized the day-240 temperature gap to a southern/western upper-ocean mode
that appears between days 180 and 240, but it did not name a process.  This
round stays on magnitude: it extends the existing year-owner instrument with
the raw process boundaries needed to budget that interval on NEMO's and
legoESM's independently evolving production trajectories.  It does not resume
the last-bit stage walk or infer a process from the spatial pattern.

## P0 — the missing measurement and the stop boundary

A census of the committed GYRE instruments and the admitted full-year records
found daily prognostic snapshots and early-step stage/operator records, but no
stage-3 tracer-process boundaries between days 180 and 240.  Daily T/S/U/V/SSH
cannot recover an accumulated RHS after individual calls, and a process label
cannot be reconstructed from their difference.  The existing records are
therefore insufficient for the requested causal ranking.

Frozen prediction: a new NEMO record is required.  Round 123 will define and
validate the reader, writer card, layout, controls and score before requesting
the record, then stop `STOPPED_FOR_RECORD`; it will not name a process owner or
land physics.  REFUTE this prediction only if an already admitted record is
found that contains every raw boundary in P2 for all steps 1081--1440 on the
same seed-0 trajectory.  An early-step record, a one-step tendency, spatial
resemblance, or a legoESM-only omit arm does not refute it.

## P1 — exact producer and passive trajectory control

The acquisition source is the already admitted full-year seed-0 card and run:

* source configuration:
  `GYRE_OMIP_L2_P3_SM_YRPERT`;
* source run:
  `phase3/year_fromrest/nemo_seed0`;
* source binary SHA-256:
  `578c88f17ecaa8052276ff43e6b6c928f5be49fb218d4af33bc8718472613c4a`;
* step-1080 restart SHA-256:
  `6c0c7a950b30b9d59dbf2673833ddf462a5f8ea5650f496f2f772e1e17092976`;
* step-1440 restart SHA-256:
  `96529a98da0e0d89b328632a826a9d41593f81f0d1a917350f28f184d49b163a`.

The new configuration name is
`GYRE_OMIP_L2_P3_SM_R123PROC`; its run root is
`phase3/round123/oracle_process_budget`.  The run starts from rest with the
source run's actual seed-0 prepared namelists.  Exactly one namelist assignment
changes: `nn_itend` is shortened from 2160 to 1440.  `nn_stock=180`,
`nn_write=2160`, `nn_pert_seed=0`, the timestep and every physical choice stay
unchanged.  This is truncation of the admitted trajectory, not a new model
configuration.

The writer is WRITE-only.  It must be an additive patch to the source card's
`MY_SRC/stprk3_stg.F90`, compile under the source card's exact preprocessor
keys, appear in the target `BLD/ppsrc/nemo` branch, and write only on rank zero
with tiling disabled.  The run is admissible only if both the step-1080 and
step-1440 restart files are byte-identical to the source run and reproduce the
two hashes above.  Any mismatch means the instrument was not passive: REFUSE
the entire process record and require a new target name after correcting the
instrument.  No number from a rejected record may be scored.

Frozen prediction: both restart comparisons are byte-identical.  This is
REFUTED, not excused as compiler noise, by one differing byte.  A restart-
consumption plant must also make admission exit nonzero.

## P2 — compiled order and frozen byte layout

The source run's compiled stage program clears `Krhs`, calls tracer advection,
then the RK3 surface boundary condition at
`GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/stprk3_stg.f90:810-853`.
At stage 3 it calls penetrative shortwave, lateral diffusion and implicit
vertical diffusion in that order at
`GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/stprk3_stg.f90:898-953`.
The source run's resolved log must continue to prove `ln_tile=F`,
`ln_traqsr=T`, and `ln_bdy`, `ln_trabbc`, `ln_trabbl`, `ln_tradmp`,
`ln_zdfmfc`, `ln_zdfosm`, and `ln_zdfnpc` all false.  REFUSE if any branch
state differs; the frozen rows below would no longer span the compiled program.

For every `kstp=1081..1440`, inclusive, the writer emits exactly one Fortran
unformatted stream file named
`oracle_process_budget_ktXXXXXXXX.bin`.  Its byte sequence is frozen as:

1. `CHARACTER(LEN=16)` magic `NEMO_L2_R123PROC`;
2. eleven 32-bit native integers: version `1`, `kstp`, `kstg`, `Kbb`, `Kmm`,
   `Krhs`, `Kaa`, `jpi`, `jpj`, `jpk`, and `STORAGE_SIZE(1._wp)`;
3. one binary64 scalar `rDt`;
4. temperature `ts(:,:,:,jp_tem,Kbb)`;
5. the three two-dimensional free-surface ratios `r3t(:,:,Kbb)`,
   `r3t(:,:,Kmm)`, `r3t(:,:,Kaa)`;
6. temperature `Krhs` immediately after `tra_adv`;
7. temperature `Krhs` immediately after `tra_sbc_RK3`;
8. temperature `Krhs` immediately after `tra_qsr`;
9. temperature `Krhs` immediately after `tra_ldf`; and
10. temperature `ts(:,:,:,jp_tem,Kaa)` immediately after `tra_zdf`.

The GYRE dimensions are frozen at `jpi=36`, `jpj=26`, `jpk=31`, and storage
size at 64 bits.  Thus each file is exactly
`16 + 11*4 + 8 + (6*36*26*31 + 3*36*26)*8 = 1,415,300` bytes.  Exactly 360
files total `509,508,000` bytes.  The parser must validate magic, version,
step sequence, stage `3`, dimensions, storage size, `rDt=14,400 s`, EOF and a
clean committed producer stamp; it may never infer a shortened or extra field
list from the observed file size.

The compiled implicit RHS uses the before thickness and the accumulated
`Krhs` at the middle thickness before its vertical solve at
`GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/trazdf.f90:545-579`.
This is why all three `r3t` slots, the before tracer, and each accumulated RHS
boundary are recorded.  A truncation plant, a producer-stamp plant and an
in-memory one-ULP change to the post-surface-boundary RHS must each print
`STATUS PLANT-FIRED` and exit nonzero.  The ULP plant is non-vacuous only if it
moves the decoded surface-boundary contribution and the interval closure.

## P3 — independent-trajectory process budget and ranking metric

The scorer extends
`nemo_testcase_l2_gyre_year_owners.py`; it is not a second year harness.  NEMO
and legoESM must each produce the same raw boundaries while evolving from rest
on their own seed-0 production trajectory.  Substitution of NEMO state into
legoESM, replay from one recorded entry, or an isolated eager closure is not
this measurement.  Production CPU/fp64 stepping and the production JIT path
remain required on the legoESM side.

On wet interior cells, for each model and step, let `q=1+r3t` and reconstruct
the accumulated explicit-temperature boundaries in the compiled order:

```
B0   = qbb*Tbb/qaa
Badv = (qbb*Tbb + rDt*qmm*Radv)/qaa
Bsbc = (qbb*Tbb + rDt*qmm*Rsbc)/qaa
Bqsr = (qbb*Tbb + rDt*qmm*Rqsr)/qaa
Bldf = (qbb*Tbb + rDt*qmm*Rldf)/qaa
```

The per-step rows are `geometry=B0-Tbb`, `advection=Badv-B0`,
`surface_boundary=Bsbc-Badv`, `shortwave=Bqsr-Bsbc`,
`lateral_diffusion=Bldf-Bqsr`, and `vertical_diffusion=Taa-Bldf`.
The explicit rounding residual is
`Taa-Tbb-sum(rows)` and is retained rather than hidden.  Summing each row over
steps 1081--1440 on each independently evolving model gives process component
`C_p = sum(delta_p_lego - delta_p_nemo)`.  Together with the incoming day-180
state gap and a closure residual, these components must reconstruct the
measured day-240 error array.

The primary ranking quantity is the signed whole-domain wet-T3D projection

```
carry_p = dot(C_p, E240) / dot(E240, E240) * RMS(E240)
```

in kelvin, where `E240=T_lego-T_nemo` and the RMS uses the existing owners
instrument's wet mask.  Signed carries, including incoming and closure, must
sum to `RMS(E240)` within fp64 summation; process rows are ranked by absolute
`carry_p`.  Component RMS is reported beside it so cancellation is visible.
The round-122 west-third / 0--100-m / 15--29-N intersection is reported only
as a diagnostic and never replaces the whole-domain ranking.

Frozen instrument checks: the same scorer must reproduce the existing
day-240 headline `1.6446741930292448e-2 K`; the interval budget must close at
every step and at day 240; and each active boundary must move at least one wet
cell somewhere in the interval.  REFUTE the ranking if the headline differs,
if closure exceeds the measured fp64 rounding residual, if the process rows do
not reconstruct the endpoint, or if the production ULP plant does not move
the registered surface-boundary row.  No advance guess is made about which
process is largest: the first owner is the mechanically largest admitted
absolute projection, not the process that best resembles the spatial map.

## P4 — scope, controls and landing boundary

Round 123 may add only the committed acquisition card, its exact reader and
controls, tests, citation map and receipt.  It does not modify the production
ocean model, the year harness, reconciliation gate, freshwater pair, `#1484`
guard, restart schema, card defaults, constants or stabilizers.  No NEMO source
is modified in place; the agent must not run `makenemo` or `mpirun`.

The reader's synthetic exact-layout test, wrong-size test, ULP propagation
test and CLI nonzero plant tests must pass before the operator is asked to
spend the NEMO run.  The receipt citation gate must map every compiled-source
citation; its shifted-citation plant must print `status=FAIL` and exit nonzero.
A separate read-only Codex pass must try to refute the field list, byte count,
passive admission, projection algebra and STOPPED_FOR_RECORD verdict.

Because this round changes no production statement, there is no ladder,
month, year, DINO or recipe blast-radius trajectory to score.  LOCK_EXCHANGE,
OVERFLOW and DINO execute no new statement.  ORCA2 is
`UNMEASURED-WITH-SPEC`: its corresponding process-budget acquisition would
need native dimensions, mask, compiled call order and passive restart hashes.
Only after both independent trajectory records exist may the largest measured
process become a candidate.  A card or carried-state choice then requires
`DECISION_NEEDED`; any implementation otherwise remains subject to the full
Decision-43 and Decision-45 gates.
