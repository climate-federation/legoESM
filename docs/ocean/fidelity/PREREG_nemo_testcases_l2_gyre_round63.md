# Preregistration: NEMO GYRE L2 round 63 Krhs split

**Frozen before instrument implementation and before acquisition.**

## Question

Which active stage-3 term at kt=2 first makes NEMO's temperature
Krhs differ from legoESM, causing the round-62 maximum incoming-content
difference of 1.679392692e-3?

The compiled call order is fixed by
cfgs/GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:

1. zero ts(:,:,:,:,Krhs) (825--827);
2. tra_adv (853--857), whose active FCT branch calls the upstream
   first guess and then adds the corrected anti-diffusive flux
   (traadv_fct.f90:171, 316--327);
3. tra_sbc_RK3 (863);
4. tra_qsr (924--929);
5. tra_ldf (944);
6. tra_zdf (958), which assembles
   e3t(Kbb)*T(Kbb) + p2dt*e3t(Kmm)*Krhs
   (trazdf.f90:545--548).

The resolved configuration makes this the complete active accumulation list:
ln_traqsr=.true., while BDY, BBC, BBL, damping, MFC, OSM, and NPC are
false (round46/oracle_kt2_stage/ocean.output:528, 546, 555--568,
734, 743, 748).

## Prediction and exact decision

**Predicted owner: FCT advection.** Its two-stage, limiter-dependent update is
the first and structurally most complex writer of Krhs; the remaining active
calls add forcing/diffusion to that accumulator.

CONFIRM only if a term-boundary replay shows that replacing legoESM's complete
FCT advection contribution (first guess plus correction) by NEMO's makes the
temperature content RHS bit-identical through the following accumulator
boundaries and final NEMO-associated assembly (zero unequal values).

REFUTE if that substitution leaves any unequal final content value, or if a
surface-boundary, solar, or lateral-diffusion increment is the first boundary
whose like-for-like replay makes the content RHS bit-identical. If no single
term passes, report unresolved/coupled; do not choose the largest correlation.

Salt is a mandatory control. The result is inadmissible unless rebuilding both
T and S assembled content from recorded operands is bit-identical to the
recorded assembly (zero unequal values). A one-ULP perturbation plant must exit
nonzero.

## Acquisition identity

- source base: GYRE_OMIP_L2_P3_SM_R46KT2
- new target: GYRE_OMIP_L2_P3_SM_R63KRHS
- run: two steps, one MPI rank
- record point: kt=2, RK3 stage 3, inner domain
- evidence target:
  /data/abyssal/dbalwada/nemo-testcases-l2/phase3/round63/oracle_krhs_split

No production-physics change is in scope.
