# Preregistration: NEMO-testcases L2 GYRE round 70 FCT/LDF cancelling pair

Date: 2026-09-12. Frozen before adding the private pair seam, running any
round-70 measurement, or editing production physics.

## Ranked owner and immutable record

Round 69 confirmed that routing the live LDF rate through the stage-3 content
association reduces the local temperature-content maximum 28.2-fold, from
`1.679392691670500e-3 K m` to `5.954039670541533e-5 K m`, but the route alone
worsened at least one cell in each of 53 canonical kt3--10 rows. It was fully
withdrawn. Complete FCT advection is the largest measured residual after that
route (`6.196948294061e-11 K s-1`, versus SBC `1.052042188174e-12`, QSR
`1.168556066120e-16`, and LDF `3.901917581501e-12`). The inherited first
non-bit FCT write is the complete `pt_rhs` increment, but round 65 established
that its Kmm tracer/transport/thickness inputs are already unequal. Therefore
this round tests the FCT/LDF pair before changing either production statement;
an oracle-output substitution is causal evidence, not an eligible fix.

The admitted record remains
`round64/oracle_krhs_split/oracle_krhs_split_kt00000002.bin`, produced by
`3b3b045bd9e03b60330204e7590e4c4470b7a0ca`. Admission must reproduce 43 of
63 inherited records exactly, classify 20 as changed, and admit all 132
consumed values. Wrong-worktree, wrong-record-producer, truncation,
record-stamp, and one-ULP plants must exit nonzero.

## Compiled statements

Compiled GYRE clears `Krhs`, then calls advection and SBC at
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:827-868`; stage 3
then calls QSR, LDF, and ZDF in that order at `:917-965`. The active two-step
FCT path forms first-step upstream faces at
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/traadv_fct.f90:497-540`, averages
the second-step faces at `:564-611`, limits the antidiffusive faces at
`:798-933`, and adds the final flux divergence to `Krhs` at `:318-329`.
The implicit tracer solve forms
`e3t(Kbb)*T(Kbb) + p2dt*e3t(Kmm)*Krhs` at
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/trazdf.f90:547-565`.

legoESM's one shared WS helper forms its stage-3 complete-FCT content and adds
the stage-3 source at `ocean_model_latlon_cgrid.py:2018-2030`, returning both
the content and its advection half at `:1918-1922`. The GYRE path calls that
helper with the live stage transports and source tuple at
`ocean_model_latlon_cgrid.py:7880-7923`. Its existing private content override
acts later at `:8049-8052`, so it cannot certify the helper's own association.

## Four-arm JIT-native matrix and frozen falsifiers

Extend the existing round-67--69 instrument and the existing private hook
container; do not create a second FCT implementation or a new public option.
Add a private data override for the helper's already-materialized stage-3
advection content and restore round 69's private source-route boolean. Both
must act inside the same JAX helper before it adds the source. The FCT operand
is the oracle's thickness-form advection content reconstructed from the
admitted Kbb tracer/thickness, `p2dt`, Kmm thickness, and `after_adv Krhs`;
the reconstruction must be bit-exact to the same compiled content statement
when all oracle operands are supplied. No host reconstruction of a legoESM
output is admissible as a target.

From the identical oracle-seeded kt=2 state run four arms whose constructor
arguments differ only in these two private controls:

1. untouched: live complete-FCT content, live stage-3 source;
2. LDF-only: live complete-FCT content, source plus the same-step live LDF;
3. FCT-only: admitted oracle complete-FCT content, live source;
4. paired: admitted oracle complete-FCT content, source plus same-step live LDF.

The untouched private arm must be bit-identical to a separately constructed
ordinary model in every state leaf and both returned content arrays. The LDF
arrays captured from every routed arm must be bit-identical to the ordinary
same-step arrays. The FCT-only helper advection content must be bit-identical
to the admitted oracle operand. Each active half must move both wet T and S
content and kt3 state; the pair must move both halves, remain finite, and use
no changed public card, state, timestep, or threshold.

Prediction: paired temperature content is at most `6.0e-6 K m` and at least
eight times smaller in maximum norm than LDF-only; paired salinity content is
at most `1.0e-6`. At the kt3-before boundary the pair must reduce the maximum
T and S oracle residual relative to both untouched and LDF-only. It must also
remove at least 99% of the wet cells that LDF-only worsens relative to
untouched by more than two row-scale float64 ULPs. CONFIRM only if all exact
controls, admission checks, movement checks, magnitude predictions, and both
plants pass. Otherwise REFUTE the pair and leave production unchanged.

Plant 1 replaces the oracle FCT operand by the untouched arm's live advection
content. FCT-only must then equal untouched, paired must equal LDF-only, and
the required FCT movement check must make the command exit nonzero. Plant 2
moves the first wet oracle FCT-content cell by one `nextafter` ULP after the
oracle reconstruction check; exactly one injected cell must change and the
frozen exact-target check must make the command exit nonzero.

## Landing and Rule 12

The admitted oracle-output override can never land. If the four-arm prediction
confirms, freeze every helper-content and kt3 metric dictionary and use the
result only to justify the next source walk into the already-unequal FCT Kmm
inputs. A production edit becomes eligible only after that walk names the
first differing compiled statement and proves the shared legoESM statement
bit-exact on NEMO inputs.

Before any eligible production edit is measured, freeze its complete kt1--10
prediction against Decision 36's recorded after arm
`f78547b752f733c4d86f024df7effc6f5b2e376a`: all 53 round-69 regression rows
must cease worsening cellwise by more than two row-scale float64 ULPs, no
other row may newly worsen, no AT-BAR row may leave the bar, and first-over-bar
may not move earlier. Then run the canonical GYRE ladder and fixed member-0
days 1--30 harness and report every day. Until a source-derived candidate
exists, those trajectory runs are **UNREACHED**, never passed.

LOCK_EXCHANGE and OVERFLOW execute the shared FCT2/WS program but no GM/Redi;
an eligible FCT statement therefore requires exact kt1--10 before/after
artifacts on both tanks. DINO uses the separate modified-leapfrog program; any
shared FCT-statement edit requires its execution gate and per-row accounting,
with its known cancellation risk explicit. ORCA2 remains **UNMEASURED WITH
SPEC**: resolve its native card; record post-SBC/QSR/LDF stage-3 `Krhs` and
pre/post-ZDF T/S for kt1--10; replay the content statement and independent
trajectory gate; require exact replay, every moved row registered, no AT-BAR
loss, and no earlier first-over-bar boundary.

No NEMO source/build/run, configuration/default, carried state, stabilizer,
year harness, reconciliation gate, freshwater pair, #1484 guard, or held
manifest is modified. Failed predictions remain recorded as REFUTED.
