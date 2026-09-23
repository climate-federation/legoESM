# Preregistration: NEMO-testcases L2 GYRE round 69 JIT-native LDF routing

Date: 2026-09-12. Frozen before adding the private causal seam, running its
measurement, editing production routing, or measuring any trajectory.

## Ranked boundary and immutable record

The first non-bit statement inherited from round 68 is the host reconstruction
of legoESM's production stage-3 content association. It moved 3 of 18,000
temperature cells by one ULP while salt stayed exact, so that host expression
is retracted as an exact predictor. This round does not reuse it. The largest
measured stage-3 content discrepancy remains temperature
`1.679392691670500e-3 K m`; putting the already-computed LDF rate into the
stage-3 source reduced the host estimate 28.2x to
`5.954039670541533e-5 K m`, after which complete FCT advection is the largest
measured residual. The LDF routing boundary therefore precedes the remaining
FCT walk by magnitude and source order.

The admitted oracle remains
`round64/oracle_krhs_split/oracle_krhs_split_kt00000002.bin`, produced by
`3b3b045bd9e03b60330204e7590e4c4470b7a0ca`. Admission must reproduce 43 of 63
inherited records bit-for-bit, classify the other 20 as changed, and admit all
132 consumed values. Wrong-worktree, wrong-record-producer, truncation,
record-stamp, and one-ULP plants must exit nonzero.

## Compiled statement

The compiled GYRE stage zeros `Krhs`, then calls advection and SBC at
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:827-868`.
Stage 3 calls QSR, LDF, and ZDF in that order at the same compiled file's
`:917-965`. The active LDF implementation reads the Kbb tracer gradients at
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/traldf_iso.f90:175-204`, forms its
Kmm-metric horizontal fluxes at `:227-246`, and adds the flux divergence with
a positive sign to `Krhs` at `:257-305`. The compiled ZDF statement forms
`e3t(Kbb)*T(Kbb) + p2dt*e3t(Kmm)*Krhs` at
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/trazdf.f90:547-565` before the back
substitution at `:575-580`.

legoESM's single WS helper forms its stage-3 advection content and source
association at `ocean_model_latlon_cgrid.py:1991-2003`, returning the actual
content at `:1918-1922`. The production GYRE path constructs its stage-3
SBC/QSR source at `:6181-6208`, computes the Kbb-input GM/Redi rate at
`:7117-7128,7340-7422`, and calls the helper with the source tuple at
`:7598-7678`. It then replaces the earlier concentration update with the
helper output at `:7833-7842`. Therefore the measured LDF rate must enter the
helper's stage-3 source tuple to execute NEMO's compiled order.

## JIT-native causal arm and frozen falsifiers

Extend the existing round-67/68 instrument rather than create a second
reimplementation. Add one private, default-false test hook that, only on the
`rk3_ws` path with an executing GM/Redi operator, passes the live JAX
`dT_gm/dS_gm` arrays into the existing helper's stage-3 source tuple. It may not
reconstruct content on the host, call a second FCT operator, alter K33, or
change any public card, default, threshold, state, or carry. Capture the
helper's own returned content and the ordinary post-ZDF kt=3 T/S fields.

Run two pre-edit arms from the identical oracle-seeded kt=2 state:

1. default hook false, which must be bit-identical to a separately constructed
   ordinary model in every returned field and in the captured content;
2. JIT-native hook true, whose only constructor-argument difference is that
   private boolean and whose injected arrays must be bit-identical to the
   GM/Redi arrays returned in the same traced step.

The one-ULP content plant must change exactly one wet cell and make the gate
exit nonzero. A routing-null plant that replaces the live LDF addition by exact
zeros must make the routed arm equal the untouched arm and must fail the
required-movement check. The old host reconstruction must remain reported as
REFUTED with its 3-cell temperature census.

Prediction: the native routed content and kt=3 state improve temperature by at
least 20x from the admitted live baseline; their maximum discrepancies are no
larger than the round-67 routed floors (T
`5.954039670541533e-5 K m`, S `7.651457963220310e-6`) plus only the measured
native-versus-host association row. The routed arm must move at least one wet
T and S cell and remain finite. CONFIRM only if both untouched exact controls,
both plants, the same-step injection identity, the magnitude rule, and the
admission rules pass. Any unequal untouched cell, a non-firing plant, a host
array round-trip in the causal path, or failure to improve T by 20x REFUTES the
candidate and no production physics edit may remain.

If confirmed, freeze the complete native routed content and kt=3 metric
dictionaries. A production edit is eligible only if it removes the private
arm while routing the same already-computed signed LDF arrays through the one
shared `rk3_ws` implementation, and then reproduces those frozen dictionaries
exactly. Zero unequal cells is required; an equal maximum with a different
census or RMS is failure.

## Rule 12 and scope

Only after exact pre-edit confirmation, run GYRE kt=1..10 with
`nemo_testcase_l2_gyre_phase3_gate.py` and compare every moved row against
decision 36's recorded after arm `f78547b752f733c4d86f024df7effc6f5b2e376a`.
No AT-BAR row may leave the bar and the first-over-bar boundary may not move
earlier. Run the fixed daily year harness for member 0 through day 30 and score
days 1..30 against `year_owners`; report every day and the day-30 T/S gap.

LOCK_EXCHANGE and OVERFLOW select the shared WS program but configure no
GM/Redi operator. Require exact before/after kt=1..10 artifacts and show the
routing statement does not execute; register any moved row. DINO uses the
separate modified-leapfrog program, so require a source-path proof and its
existing execution gate or byte-identical artifact; the shared GM/Redi routine
alone is not sufficient proof of risk. DINO's known per-row cancellation debt
remains explicit and no band statistic may waive a moved row.

ORCA2 remains **UNMEASURED WITH SPEC** only if the receipt records this exact
future gate: resolve its native card; record post-SBC, post-QSR, and post-LDF
stage-3 `Krhs` plus pre/post-ZDF T/S for kt=1..10; replay the content statement
and independent trajectory gate; require exact statement replay, every moved
row registered, no AT-BAR loss, and no earlier first-over-bar boundary.

No NEMO source/build/run, configuration choice, carried-state change,
stabilizer, year harness, reconciliation gate, freshwater pair, #1484 guard,
or held/refuted manifest is modified. A failed prediction is kept as REFUTED.

## Post-edit landing-gate correction after refused run

The first production invocation wrote status **REFUTED** even though both
frozen native content dictionaries and both frozen kt=3 dictionaries matched
exactly. Two bookkeeping checks had compared different arms. First, the 20x
check divided the routed result by the post-edit production baseline, which
already contains the route, rather than round 68's frozen un-routed
`1.679392691670500e-3 K m` baseline. Second, it required the post-edit host
reconstruction to retain round 68's 3-cell un-routed association census; the
post-edit quantity is the routed association and correctly reproduces the
pre-edit native arm's separate 2-cell association row instead. No threshold,
physics prediction, native target, or model code changes. Before the first
successful post-edit run, calculate improvement against the frozen un-routed
report, keep the 3-cell row explicitly as the historical retraction, and check
the current 2-cell row against the frozen native association census. The
refused report remains in `round69_native_source_after.json` until a stamped
corrected run supersedes it by filename and checksum in the receipt.
