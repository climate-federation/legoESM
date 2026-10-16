# ORCA2 round 298 preregistration — kt=7 stage-1 writer replay

Date: 2026-10-16. Frozen base: `e3cba1315`. Scope: OMT-4, separately
labelled **independent** and **given NEMO's entry**. The operator reports that
round 297's rank-complete kt=7 stage-1 record admitted. This round reads that
record only after this preregistration is committed, replays the complete
private fold unit from its finite kt=7 entry, and names the first source-order
boundary carrying the stage-1 growth. It changes no production package, card,
deck, selector, carried state, stabiliser, sea-ice field, or
`unmeasured_features` entry.

## Frozen source order

The compiled OMT-4 stage program carries the completed momentum RHS into the
stage-1 vector update at
`ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/stprk3_stg.f90:477-480`, applies the
barotropic correction at `:521-528` and `:547-549`, builds the active-tracer
transport at `:565-592`, accumulates tracer advection at `:617-662`, applies
the surface source at `:664-672`, performs the QCO stage update at `:693-720`,
and associates the completed U/V/T/S state at `:812-820`. The record names
those boundaries `rhs`, `pre`, `cor`, `trp`, `adv`, `sbc`, `qco`, and `final`.

The candidate is evaluated with the same private atomic unit as round 296 and
with production JIT on CPU in fp64/libm. Candidate intermediate values come
from already-existing passive stage operands and an offline call to the shared
pure tracer-stage implementation. The offline tracer replay must reproduce the
candidate's own completed stage below the 2e-10 campaign floor before any
oracle residual is interpreted.

## Frozen predictions and falsifiers

1. **R298-P1 — admission.** The round-297 record has two rank records with 23
   fields, exactly-once 148x180 coverage, and byte-identical inherited frames
   and restarts. Any admission refusal stops the round without reading values.
2. **R298-P2 — label agreement.** Independent and given-entry OMT-4 name the
   same first source-order boundary. Different first boundaries refute this
   prediction and no attribution is made.
3. **R298-P3 — first growth carrier.** The momentum RHS/update/correction and
   tracer transports do not first create the kt=7 stage-1 fold-band growth;
   `adv_t`/`adv_s` are the first recorded boundary whose disagreement carries
   it. An earlier `rhs/pre/cor/trp` or later `sbc/qco/final` owner is retained
   as **REFUTED** and becomes the finding.
4. **R298-P4 — program-point controls.** Candidate tracer replay reproduces
   its own completed stage below 2e-10; NEMO `adv` equals `sbc` on this
   unforced OMT-4 deck; both labels cover the full boundary order; a one-bit
   perturbation fires. Failure of any control makes the instrument refuse.
5. **R298-P5 — measurement only.** This round lands no private fold-unit
   constituent. If the first boundary is a multi-statement operator, the OPEN
   item is its operand split in compiled order, not a partial landing.

## Stop conditions

The complete fold unit remains private and atomic. Decision 125's top-V
exchange and the resolved stage-mask statement remain off in production.
Decision 126 remains pending and is not re-raised; shipped runoff and sea ice
are unchanged. A source boundary is named only when both labels agree and all
controls fire.
