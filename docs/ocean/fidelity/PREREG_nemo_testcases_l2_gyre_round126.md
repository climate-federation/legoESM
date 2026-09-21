# Preregistration — NEMO testcase L2 GYRE round 126

**Frozen before scientific scoring.**  The operator reports the Round-125
record ready at
`phase3/round125/oracle_vertical_decomposition`; this document fixes how it
will be admitted and decomposed.  Merely confirming that the supplied path
exists is not a scientific result.  The incoming legoESM commit is
`4cac617cd928007506f2de7ccb098f87e04204d1`.

## Question and inherited magnitude

Round 124 assigned a signed `+2.4168271578053416e-2 K` of the day-240
temperature-error projection to vertical diffusion over steps 1081--1440.
Round 126 asks which sub-boundary carries that number, and whether a divergent
mixing coefficient is an independently wrong TKE closure or a response to the
already divergent circulation/front position.

The compiled Round-125 program calls the explicit tracer operators before
`tra_zdf` at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/stprk3_stg.f90:937-944`.
Inside `tra_zdf`, temperature selects `avt + ah_wslp2`, zeros the surface
coefficient and constructs the three diagonals at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/trazdf.f90:418-481`; it then
forms the LU diagonal, content RHS, forward recurrence and back solve at
`:527-592`.  The compiled closure dispatch copies the TKE result into `avt`
at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/zdfphy.f90:325-369`, and the
TKE source computes that diffusivity from energy and mixing length at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/zdftke.f90:679-693`.

## P0 — record admission before use

The existing year-owner instrument must admit exactly 362 frames (steps 1,
2, and 1081--1440), each 6,965,416 bytes, with the Round-125 producer stamp,
binary hash, exact file manifest, passive restart hashes, per-step Round-123
trajectory alignment, and zero unequal cells in all eight compiled-arithmetic
calibration rows.  Its four controls — wrong stamp, truncation,
consumed-matrix ULP and trajectory ULP — must print `STATUS PLANT-FIRED` and
exit nonzero.  Any failure **REFUTES admission and stops the round**; no subset
of the record will be scored.

Frozen prediction: all admission rows pass and every plant exits nonzero.

## P1 — one extended production trace, not a second harness

The existing `--produce-process-trace` path will be extended with a static
`--produce-vertical-trace` output option.  It will still drive
`LatLonCGridOceanModel.step`, hence `_step_jitted`, independently from rest
through step 1440 and carry state only from a separately compiled ordinary
production call.  During steps 1081--1440 the write-only trace will expose the
actual values already consumed by the production solve:

1. pre-solve temperature content and normalized pre-solve temperature;
2. TKE heat diffusivity, isoneutral vertical diffusivity and their sum;
3. `e3t(Kaa)`, `e3w(Kmm)`, wet mask and the consumed lower/diagonal/upper
   matrix; and
4. the solved temperature.

The new path reuses the existing private process-trace hook and the production
`nemo_tracer_tridiagonal`/ordered solve.  It does not add a public selector or
a duplicate vertical solver.  Every traced step must leave the separately
compiled ordinary carried state byte-identical, and the regenerated day-180,
day-210 and day-240 core snapshots must agree with the immutable year member
where an immutable snapshot exists.  The old Round-124 process fields must
also remain bit-identical over all 360 frames.

Controls are frozen before the run: a wrong trace stamp and one changed trace
matrix bit must be rejected; a production-step effective-diffusivity plant
must move a real wet interface, a consumed matrix coefficient and the
diagnostic solve while leaving every upstream process boundary unchanged.
Each plant exits nonzero and prints `STATUS PLANT-FIRED`.

Frozen prediction: ordinary-state unequal bytes are zero, all inherited
process fields are bit-identical, and all three new controls fire.  Any state
change or upstream movement **REFUTES the write-only claim and stops the
round**.

## P2 — frozen vertical-increment telescope

The scored object at each step is the vertical increment, not raw
temperature:

`V = solved_temperature - pre_solve_content / e3t(Kaa)`.

That definition makes a no-diffusion solve exactly zero even when the two
independent trajectories enter with different temperatures.  NEMO's recorded
RHS, LU, forward sweep and solution must first reproduce bit-for-bit.  The
legoESM trace must reproduce its production content, matrix and solution.

Starting from the NEMO increment, the following substitutions are made in
this fixed source order.  Each row is the new hybrid minus the preceding
hybrid; rows are never reordered after seeing their sizes.

1. **live column / gradient acted on** — replace only normalized pre-solve
   temperature by legoESM's;
2. **free-surface weighting** — replace `e3t(Kaa)` and `e3w(Kmm)` by
   legoESM's, rebuilding the literal matrix;
3. **TKE heat diffusivity** — replace only `avt`;
4. **isoneutral vertical diffusivity** — replace only `ah_wslp2`/the model's
   already-computed K33 contribution;
5. **matrix construction/association** — replace the rebuilt matrix by the
   matrix emitted inside the production JIT;
6. **implicit solve/association** — replace the source-ordered offline solve
   by the production-JIT solved column; and
7. **interaction/order residual** — retain, under this name, whatever is
   needed to close `V_lego - V_NEMO`; it is not distributed among preferred
   owners.

The surface/bottom row is scored separately from the recorded boundary
coefficients and active flags.  The compiled program imposes the surface zero
at `trazdf.f90:454` and describes the masked bottom condition at `:465-481`;
no nonzero boundary flux will be inferred from an explicit tendency that ran
before `tra_zdf`.

Each 360-step component is summed and projected onto the same day-240 endpoint
error used by Round 124.  The signed carry sum must reproduce
`+2.4168271578053416e-2 K`; the full-field telescope and projected telescope
must report their residuals.  Six ten-day blocks, depth bands, west/interior/
east thirds and latitude bands are retained for the largest row.

Frozen magnitude prediction: the TKE-`avt` row is largest, is positive and
lies in `[+1.5e-2,+3.5e-2] K`; the live-column row is at most `1.0e-2 K` in
absolute carry, the isoneutral row at most `5.0e-3 K`, and each geometry,
matrix, solve and surface/bottom row at most `1.0e-6 K`.  Any value outside
these ranges is **REFUTED**, retained verbatim in the receipt, and the measured
ranking wins.

## P3 — closure-versus-upstream discriminator

At the available day-180 and day-210 restart boundaries, the card's production
`diagnose_vertical_K` closure will be driven from a bridged NEMO step-entry
state: NEMO T/S/u/v/ssh, depth means, `en`, `avm_k`, `avt_k` and `dissl`, with
the forcing phase of steps 1081 and 1261.  The call runs through the full
production step closure under `jax.jit`; an isolated local formula is not
called “production.”  Its heat diffusivity is compared with that step's
recorded NEMO `avt`, alongside the model-own-trajectory comparison.

The upstream-response prediction is frozen as follows: at both checkpoints,
substituting the NEMO entry state reduces the wet-interface RMS `avt` error by
at least 90 percent and reduces its maximum error.  A reduction below 50
percent at either checkpoint **REFUTES** “only responds upstream” and assigns
the next walk to the TKE closure.  A reduction of 50--90 percent is
`INCONCLUSIVE`, not rounded into either verdict.

To test the more specific jet-position interpretation, upper-100-m speed is
mapped to T points and the speed-weighted centroid of the fastest decile in
the western third is reported at days 180 and 210 for both trajectories.
Consistent nonzero centroid displacement together with the >=90-percent
conditional collapse confirms “upstream jet-position response.”  Otherwise
the receipt says only “upstream state response”; it will not promote visual
resemblance to attribution.

## P4 — round boundary

This is a diagnostic round.  No production physics, card, restart schema or
configuration value lands.  The largest measured sub-owner becomes Round
127's only candidate.  If that candidate needs a configuration or carried
state choice, the receipt and final report ask `DECISION_NEEDED` instead of
choosing one.

Because no production statement changes, the Decision-43 ladder/month and
Decision-45 year landing gates do not run.  GYRE, generic NEMO-GYRE, DINO,
LOCK_EXCHANGE and OVERFLOW production rows do not move.  ORCA2 is
`UNMEASURED-WITH-SPEC`: repeat this native interval record, independent
production trace, conditional-closure checkpoints and endpoint projection
before transferring the ranking.

A separate read-only Codex pass must try to refute the trace passivity,
telescope, controls and closure-versus-upstream verdict.  Every compiled
source citation in the receipt is mapped by the citation gate, and its shifted
citation plant must exit nonzero.
