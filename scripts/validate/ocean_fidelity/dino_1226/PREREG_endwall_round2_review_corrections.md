# PREREGISTRATION CORRECTION — reachable coastal and tau gates

Status: frozen after evidence review placed the first coastal classification
on HOLD, before the corrected offline execution.

## Coastal-unmask gate retraction

The previous gate required the wet `j=1` argmax to be in the coastal-unmask
set. That is impossible by construction: the argmax mask requires `umask=1`,
whereas the coastal set requires `umask=0`. Its
`REFUTED_COASTAL_UNMASK` label is **RETRACTED**; the printed overlap values and
one-cell localization remain measurements, not verdicts.

The corrected candidate number is only the fraction of each operand's
full-plane material residual support in the exact NEMO coastal-unmask set.
Material support remains the already registered
`abs(delta)>=0.01*max(abs(delta))`. For each of direct `dynzdf` and `F_slow`,
**CONFIRMS_COASTAL_UNMASK** iff fraction `>=0.90`, **REFUTES** iff fraction
`<=0.10`, otherwise **UNRESOLVED**. The overall hypothesis CONFIRMS only if
both operands confirm, REFUTES if either refutes, otherwise is UNRESOLVED.
The wet argmax's mask/T-neighbour/multiplier state remains descriptive context
and cannot enter this gate.

A synthetic material-support set entirely inside coastal points must CONFIRM.
Moving every synthetic support point to known noncoastal points must REFUTE.
Both planted branches must fire or all coastal labels are invalid.

## Tau arithmetic wording correction

The live-slot preregistration's “maximum relative error <=1e-8” wording is
**RETRACTED**. The review-provided replacement number is pointwise wet ratio
standard deviation, measured as approximately `9.98e-9`; maximum pointwise
deviation from 2 is separately descriptive and is approximately `1.64e-7`.
The verifier confirms `NAMED_IS_TWICE_APPLIED_RATIO_STD_LE_1E-8` iff the ratio
standard deviation is `<=1e-8`. A planted `1e-3` ratio mutation must exceed the
same standard-deviation bar. It must never print or document maximum agreement
to `1e-8`.

The corrected coastal run reuses the same committed kt=5761 bridge and arrays,
prints all prior retractions plus this impossible-gate retraction, stamps this
preregistration commit, and runs CPU-only. The tau verifier may be rerun against
the retained isolated CPU artifact; no NEMO or GPU execution is authorized.
