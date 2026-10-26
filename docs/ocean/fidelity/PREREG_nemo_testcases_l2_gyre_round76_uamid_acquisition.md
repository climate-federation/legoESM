# Preregistration: NEMO-testcases L2 GYRE round 76 `ua_e` acquisition

Date: 2026-09-12. Frozen after the committed kt=2 transport-mean probe
refuted the predicted wet-cell count and its trace-identity calibration, and
before writing or preflighting the replacement acquisition card.

## Measured stop and record need

The first probe preserved the predicted source location but refuted its
magnitude: substep-1 `zhU` differs at 2 of 580 wet U faces, not 580 of 580,
with maximum `9.947598300641403e-14`. Its `ua_e` operand differs at the same
two wet faces by at most `2.117582368135751e-22`; `zhup2_e` is bit-exact.
The shared legoESM metric-transport and accumulator statements are bit-exact
for all 50 rows when fed NEMO's recorded operands.

The production-JIT substep trace and the separate live-stage trace disagree in
their final transport average by one ULP at 352 U faces and 354 V faces. The
same disagreement remains without the tracer callback, so the 2-face `ua_e`
residual is below the instrument's demonstrated compilation floor and is not a
citable first-live-statement verdict. The robust round-73 final `un_adv`
boundary remains open; no production edit is eligible from this trace.

The compiled NEMO target evaluates the kt=2 mid-step U velocity as
`ua_e = za1*un_e + za2*ub_e + za3*ubb_e` at
`GYRE_OMIP_L2_P3_SM_R75ADV3/BLD/ppsrc/nemo/dynspg_ts.f90:484-493`. Those raw
inputs are not present in the admitted kt=2 records, so an oracle-input shared
statement test requires a new record rather than another intrusive live trace.

## Frozen acquisition and falsifiers

Prepare a new target `GYRE_OMIP_L2_P3_SM_R76UAMID4` from the unchanged
`GYRE_OMIP_L2_P3_SM_R75ADV3` source card and a new evidence directory
`round76/oracle_uamid_kt2`. Copy EXP00 and MY_SRC file by file; preserve the
namelist byte-for-byte-identically; add only a WRITE-only stream to the
config-local `dynspg_ts.F90`; never modify canonical NEMO source.

At kt=2 only, after the compiled `ua_e`/`va_e` extrapolation, record one
16-byte magic, six int32 header values `(version, kt, ncycle, jpi, jpj, bits)`,
then for each of 50 substeps one int32 `jn`, three float64 coefficients
`za1,za2,za3`, and four 36 by 26 float64 U fields in order
`un_e,ub_e,ubb_e,ua_e`. The exact stream size is 1,499,040 bytes:
`16 + 6*4 + 50*(4 + 3*8 + 4*36*26*8)`. There is one header and no record
marker.

Prediction: a strict reader replays
`((za1*un_e)+(za2*ub_e))+(za3*ubb_e)` bit-for-bit at all 50 substeps. Exact
preprocessing using the R75 build keys and includes followed by
`gfortran -fsyntax-only` exits zero with empty stdout and stderr. Preflight
proves the four-field writer/reader registry and exact byte count agree. A
layout plant removing `ubb_e`, a one-ULP `ua_e` target plant, a shifted header,
a truncated-size plant, and a wrong-stamp plant must each exit nonzero.

The operator run must retain the exact final restart and mesh mask, admit only
the new U-midpoint record against the R75 source run, make the consumed-field
plant exit nonzero, and stamp the clean committed checkout that actually runs
the acquisition. Any failed record or admission gate refuses readiness.

No production statement, configuration/default, carried state, stabilizer,
year harness, reconciliation gate, freshwater pair, #1484 guard, or held
manifest may change. GYRE kt=1--10 and days 1--30, LOCK_EXCHANGE, OVERFLOW,
and DINO remain UNREACHED because this is acquisition only. DINO retains the
shared-statement and cancellation risks. ORCA2 remains UNMEASURED WITH SPEC as
registered in the main round-76 preregistration. The round-70 patch stays held.
