# Preregistration — VORTEX_SMT round 17 (lane round 229): horizontal LDF tensor walk

Frozen before reading scientific values from the corrected Round-228 R16
payload or running legoESM against it.  Base:
`be95a1a1685adcdb24ceb1a0276fea81c600c836` (round 228).  Evidence lives under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round229/`; the immutable NEMO
record is `phase3/round228/oracle_vortex_smt3_ldf_internal/`.

## Compiled branch read first

The R16 build computes the horizontal tensor factors inside the same cell loop
that consumes them.  It forms A11/A22 from the live Kmm U/V thickness at
`VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:243-244`, the
two four-mask reciprocals at `:246-249`, A13/A23 from the already-certified
native slopes at `:251-252`, and the horizontal fluxes with the parenthesised
gradient sums at `:254-259`.  The corrected writer copies all six scalar
factors and both fluxes before the loop closes at `:260-266`; it does not
repeat Round 227's outside-loop scalar copy.

The existing production operator exposes these same rows through its private
write-only LDF diagnostic hook.  Pre-implementation search found and extends
`nemo_testcase_l1_vortex_smt_round228_ldf_internal_walk.py`; no second parser,
stage harness, or isolated operator is created.

## Predictions and falsifiers

* **R17-P1 — admission and aggregate reproduction.**  The R16 admission is
  passive and self-consistent, and the production-JIT rerun reproduces the
  frozen pre-LDF `7.418332614861356e-11 K`, post-LDF
  `2.0915088416728622e-07 K`, and additional-LDF
  `2.090767008411376e-07 K` maxima.  REFUTED by a failed stamp/plant/restart
  check or any different aggregate value.
* **R17-P2 — corrected factor record.**  The six corrected groups are genuine
  cellwise arrays: each recorded factor equals a direct reconstruction from
  its own recorded operands on the executed wet support, and a one-ULP plant
  on one factor changes its row and exits nonzero.  REFUTED by a spatially
  replicated last scalar, a failed reconstruction, or a missed plant.
* **R17-P3 — source-order boundary.**  A11, A22, A13, A23 and the two mask
  reciprocals are bit-exact under the production step; the first magnitude
  boundary is the horizontal-flux association at zfu or zfv.  REFUTED by any
  earlier factor row that is non-bit and carries at least 90% of the additional
  LDF maximum, or by bit-exact horizontal fluxes.
* **R17-P4 — magnitude closure.**  Replaying NEMO's parenthesised horizontal
  flux association as a one-variable production-step arm removes at least 90%
  of the `2.090767008411376e-07 K` additional-LDF maximum.  REFUTED if the
  production arm removes less than 90%, moves an upstream row, or needs a
  second operand or statement.
* **R17-P5 — landing discipline.**  A physics statement proceeds to the full
  landing gates only if its production-JIT row is bit-exact with NEMO's own
  preceding seam and P4 confirms magnitude closure.  Otherwise production is
  unchanged and the round is HELD with the next measured seam named.

Failed predictions remain in the receipt.  Eager and isolated-JIT results may
support a production row but cannot replace it.  The plant must print
`STATUS PLANT-FIRED` and exit nonzero.

## No hidden choices

Decision 93 already authorises the SMT-3 rung.  This round changes no scheme,
coefficient, timestep, threshold, stabiliser, carried state, card default, or
record format.  If the measured source row needs a new configurable physics
choice rather than a literal transcription, the round stops with
`DECISION_NEEDED` instead of selecting one.
