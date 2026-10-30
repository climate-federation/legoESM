# GYRE high-resolution round 17 preregistration

Date: 2026-10-19.  This document is committed before either round-17 NEMO
record is acquired or the discriminator is scored.

## Question and invariant input

Round 16 measured the days-30--180, western-boundary, native-surface-V signed
mean of `total_vorticity_coriolis`: legoESM minus vector-math NEMO is
`1.6614925847170393e-6 m s-1`, while scalar-math minus vector-math NEMO is
`3.567844130008744e-8 m s-1` (ratio `46.56853057964074`).  That identifies a
term carrier, not a physics owner.

The round-17 discriminator asks whether the already formed ENE potential
vorticity operand `zwz` owns that excess.  Vector and scalar NEMO and legoESM
must all start from this *same* developed vector-math restart:

```
/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round15/oracle_gyre_r4_smag_budget_180d_corrected/control/GYRE_R4_SMAG_R14_BUDGET_00004320_restart.nc
sha256 a8e33a3d22e4483bac777839f6cdc9bdf8c2e348c3a1d638adeafa114fff1c12
```

The scalar run's own day-180 restart is forbidden here because it already
contains 180 days of trajectory separation.  Both NEMO runs execute only
`kt=4321`, one rank, on the resolved `r4_smag` deck.  The writer records stage
3 (`Kmm=N+1/2`) immediately before and after `dyn_vor`, plus the executed
`vor_ene` arrays after the live-F-thickness division (`zwz`, hence q) and the
two metric transports (`zwx`, `zwy`).  The legoESM run uses the certified CPU
fp64/libm policy and the production-jitted WS-RK3 step from the same restart.

## Registered reduction and verdict

The primary row is exactly the round-16 reduction narrowed to the one common
step: native surface V, the existing western-boundary wet-span bin, signed
mean.  Let `L` be legoESM's stage-3 vorticity/Coriolis term, `Nv` vector-math
NEMO's term, and `Ns` scalar-math NEMO's term.  The arithmetic envelope is
`E = abs(mean(Ns - Nv))` on that row.

One variable is changed: replace only legoESM's already formed stage-3 q with
the scalar-NEMO `zwz` record.  All velocities, live thicknesses, metric
transports, masks, and the operator stencil remain legoESM's production
values.

- **CONFIRM q formation as the owner boundary:**
  `abs(mean(L[q := Ns.q] - Ns)) <= E`, and the already computed stage-3 HPG
  and advection component arrays are bitwise unchanged between the baseline
  and substitution runs.
- **REFUTE q formation:** the substituted row remains outside `E`.  No shared
  physics change follows; the next discriminator must substitute the recorded
  metric transports or the completed operator term.
- **INVALID:** either NEMO build does not start from the pinned restart, the
  recorded before/after difference does not reproduce its recorded vorticity
  term, a provider has a non-finite active value, the wet masks/region differ,
  or an untouched component moves.

The existing round-16 `4x` screen is reported for context, but it is not the
causal verdict bar: this controlled substitution must enter the measured
NEMO-vs-NEMO envelope itself.  A planted active q-word perturbation must make
the substitution verdict refuse.

## Source-execution registration

The active compiled GYRE source calls `dyn_vor` between `dyn_hpg` and
`dyn_adv` in every RK stage
(`/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/GYRE_OMIP_L2_P3_R14BUD/BLD/ppsrc/nemo/stprk3_stg.f90:399-409`).
Stage 3 reads `Kmm=N+1/2`
(`/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/GYRE_OMIP_L2_P3_R14BUD/BLD/ppsrc/nemo/stprk3_stg.f90:218-241`).
The resolved ENE branch calls `vor_ene(...,ntot,...)`
(`/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/GYRE_OMIP_L2_P3_R14BUD/BLD/ppsrc/nemo/dynvor.f90:230-238`).
It forms total vorticity, divides it by live F thickness, forms the two metric
transports, then applies the signed two-point stencil
(`/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/GYRE_OMIP_L2_P3_R14BUD/BLD/ppsrc/nemo/dynvor.f90:536-573`).

## Scope guard

This round may add private diagnostic/injection seams and write-only NEMO
writer patches.  It does not authorize a public configuration option or a
physics change.  A shared-path fix is allowed only after the registered
substitution localises a transcription difference and the affected cards'
certified gates are measured before and after.
