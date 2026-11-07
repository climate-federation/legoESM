# SI3 lane 3 — rung 3.2 DEBT ownership preregistration

Issue: climate-federation/legoESM #1699

Oracle root: `/data/abyssal/dbalwada/nemo-testcases-l3/ice_adv2d/final`

This document is committed before the written-order replay is implemented or
run.  The fixed gate at commit `9f9c4a4a1681f7cb88bb381e369dbbf378dec2a7`
first exceeds the normalized pointwise `1e-15` bar after completed step 16 in
`szv_i_l01`, at `1.1095589247903137e-15`.  The purpose of this measurement is
to classify that DEBT, not to alter the bar.

## Fixed replay

The probe will build the pinned fp64/CPU ICE_ADV2D card and advance its current
kernel through 15 completed steps.  From that identical candidate input it
will execute completed step 16 twice:

1. the production JAX SI3 Prather arm; and
2. a NumPy replay whose scalar assignment sequence follows NEMO
   `icedyn_adv_pra.F90`: y limiter `:757-791`, positive-y extraction/removal
   before negative-y extraction/removal `:793-882`, receiver merges `:884-943`,
   then x limiter `:534-568`, positive-x extraction/removal before negative-x
   extraction/removal `:570-664`, and receiver merges `:666-717`.  Step 16 is
   even, hence y then x by `:253-351`.  Hbig/Hsnow/zapneg retain their existing
   source order `:405-421,946-1142`.

The replay must use the same fp64 arrays, halo masks, periodic boundaries,
tracer order, and one-cycle CFL as the card.  It may not load an oracle field
as candidate state.  It will record hashes of the input, the oracle `kt=17`
frame, the production result, and the replay result.

## Fixed measurements and classification

For every one of the 16 transported families, the probe will report the
production-versus-oracle and replay-versus-oracle maximum absolute,
normalized-maximum-absolute, and maximum ULP distance on the 99 x 99 physical
domain after step 16.  It will separately report all five moment arrays for
all 16 tracers between production and replay; step-entry frames do not contain
moments, so no oracle-moment claim will be made at step 16.

The preselected discriminator is the first gate-DEBT field `szv_i_l01` at its
production maximum-error cell:

* **RE-ASSOCIATION** only if the written-order replay is within two fp64 ULP of
  the oracle at that cell and the production result is farther away.  This is
  an arithmetic classification, not an AT-BAR or fidelity claim.
* **IMPLEMENTATION/INPUT DEBT** otherwise.  The probe must then expose the
  first production-versus-replay operand or assignment outside two ULP in the
  y limiter/sweep, x limiter/sweep, or post-advection corrections, and name
  that owner.  If the first differing operand is already present in the common
  step-16 input, ownership remains **UNMEASURED** rather than being guessed.

The probe must include a planted arithmetic-order control that changes a
nonzero common operand and demonstrably changes the reported ULP result.

## End-of-task ASKED / UNASKED choice list

**ASKED choices:** replay one first-DEBT step in NEMO-written x/y/limiter
order; use the immutable oracle; classify at two ULP; first-diverge an operand
if the replay does not close; fp64 and CPU only; preserve the `1e-15` gate.

**UNASKED choices:** no tolerance change; no production-kernel edit before the
measurement; no oracle-file modification; no claim about within-step NEMO
moments absent from the frames; no attribution from scaling alone; no rung-3.3
claim from this probe.
