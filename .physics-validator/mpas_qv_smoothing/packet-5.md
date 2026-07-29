# Adversarial review packet — MPAS q_v smoothing (round 5, doc-precision final)

Independent adversarial reviewer. Round 4: "no substantive implementation,
conservation/sign, differentiability, dtype, config/CLI, or test-vacuity defect"
— the ONLY open item was two more spots with unqualified "exactly"/"no-op"
conservation wording. Those are now fixed, plus one more I found myself. This
round is DOCUMENTATION ONLY (no code/logic change).

## Applied since round 4 (documentation only)

1. `config.py` field comment (was "conserves the per-level mixing-ratio area
   integral sum_c A_c q_c exactly") → now "conserves the per-level mixing-ratio
   area integral sum_c A_c q_c (exact in exact arithmetic; to floating-point
   roundoff — ~1e-7 relative in fp32)."
2. `test_mpas_qv_smoothing.py` `test_plain_form_conserves_area_integral`
   docstring (was "sum_c A_c lap_c = 0 exactly") → "= 0 (exact in exact
   arithmetic; checked to 1e-12 rel in x64)".
3. `test_mpas_qv_smoothing.py` `test_plain_step_is_monotone_and_positive`
   docstring (was "floor is then a strict no-op") → "floor is then a no-op, to
   roundoff".
4. `test_mpas_qv_smoothing.py` `test_plain_step_conserves_area_integral`
   docstring (was "so this is exact") → "so this is exact in exact arithmetic;
   checked to rel 1e-13 in x64".

The remaining unqualified word "EXACT" in the driver setup comment
("the CFL guard below is EXACT for the applied op" / "EXACT for the plain form
actually applied") refers to the GUARD FORMULA matching the operator's
monotonicity row-sum factor g_c — an algebraic identity you confirmed correct in
round 2 ("CFL review: correct for the applied plain operator"), NOT a
floating-point conservation claim. Please confirm that is a correct usage and
does not need the roundoff qualifier.

## Ask

Confirm the doc-precision correction is now FULLY addressed and there are no
remaining substantive OR documentation findings. If clean, state "no
substantive findings" and that the documentation is fully qualified.
