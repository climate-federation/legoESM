# NEMO testcase L2 GYRE high resolution — round 11 preregistration

Date: 2026-10-16

Incoming lane tip: `854d2da150da603813f2637e8ae764a7c5459ccf`.

No measurement registered here changes a deck, a physics option, a bar, or a
shared production path.  The oracle is the admitted scalar-math step-2 record
at
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round7/oracle_gyre_r4_scalar_step2/run/`.
The climate reference remains the certified vector-math five-year NEMO run;
the scalar-math run is the admitted arithmetic twin.

## R11.1 — exact-rung minimal subset

The scored boundary is the stage-3 `tra_zdf` output at entry `kt=2`, under the
certified fp64/libm CPU policy.  The bar is zero unequal active float64 words.
The executed source forms one shared matrix and then separate temperature and
salinity contents (`trazdf.f90:400-458,504-559` in the compiled
`GYRE_OMIP_L2_P3_SMW_R7STEP2` source).

The controlled arms, in source order, are:

1. replace formed diffusivity plus recorded temperature content;
2. add recorded salinity content;
3. score the proper subsets of any exact arm so that “minimal” means that
   removing any named member loses exactness;
4. only if temperature is still not exact, add the two bit-exact geometry
   inputs, interface thickness then cell thickness, one set at a time.

`CONFIRM_T_PAIR` means arm 1 makes temperature bit-exact while both singleton
arms remain non-exact.  `CONFIRM_TS_TRIPLE` means arm 2 makes both tracers
bit-exact and every proper subset fails for at least one tracer.  Any other
result names the smallest exact subset actually measured.  If no subset is
exact, the result is `OPERATOR_COMPOSITION`.  A formed input that is itself a
different expression from NEMO may be fixed only after its executed expression
is traced; otherwise a multi-input result is recorded as rounding-order debt
and no shared kernel changes.

Every new selector has a disabled-path identity control and an active-word ULP
plant.  A selector or plant that does not reach the intended tracer invalidates
the experiment.

## R11.2 — day-30 to day-180 divergence excess

The comparison uses identical daily means on days 30 through 180.  NEMO vector
is the reference.  NEMO scalar minus NEMO vector is the arithmetic-twin floor;
the registered `post_round9_219db484b` run minus NEMO vector is the candidate.
The existing divergence definition remains unchanged.

The available like-for-like fields are surface SSH, native surface U, native
surface V and thickness-weighted SST.  The first field is the earliest one
whose candidate normalized RMS exceeds four times its same-day twin value,
reusing round 10's disclosed operational envelope.  Ties are reported, not
broken by field order.

Spatial ownership uses squared-error excess
`SSE(candidate-vector) - SSE(scalar-vector)` without clipping.  Each row's wet
zonal span is divided into five equal fractional-distance bins from its western
wet boundary; all five bins and their signed shares are reported.  A western
boundary result requires the westernmost bin to carry the largest positive
excess.  An interior result requires one of the middle three bins.  Surface
and prognostic depth-mean velocities are scored separately.  The stored
legoESM run has no bottom velocity, so bottom ownership is explicitly
`UNMEASURED`, never inferred from the surface or depth mean.

Growth is measured from the absolute RMS difference of the first field over
days 30–180, both basin-wide and in its leading zonal bin.  Two fits use the
same positive daily series and the same log-residual metric:

- exponential: `log(error) = a + b day`;
- algebraic: `log(error) = a + p log(day)`.

`EXPONENTIAL` requires the exponential fit to have the lower log RMSE in both
the basin and leading region with positive growth.  `ALGEBRAIC` requires the
algebraic fit in both.  Disagreement is `MIXED`; non-positive or non-finite
series refuse.  Exact synthetic exponential and algebraic series must select
their own class, and a planted spatial error must move the regional owner.

These statistics are post-hoc diagnostics of the completed run.  They do not
alter the preregistered year-5 climate bars.
