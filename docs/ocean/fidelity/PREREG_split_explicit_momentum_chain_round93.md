# Preregistration: exact W-slope/Shapiro × A33 floor split, round 93

Date: 2026-08-30. Frozen before measurement. Inputs are the retained day-180
raw/final W-slope dumps, the round-87 component streams, and the round-92
production diagnostic. No NEMO, GPU, or held run is authorized.

First replay `ldfslp.F90:320-367` as a 2x2 existing-dump matrix: production
versus NEMO raw `zwz/zww` input, crossed with production versus host
source-ordered Shapiro association. Score raw and final W fields separately at
`1e-15`. The raw main effect owns only if it removes at least 90% of the final
W error; the Shapiro main effect and interaction use the same threshold. A
one-cell roll and a changed central Shapiro weight must be red.

Then hold the winning W pair and replay `traldf_iso.F90:285-332` as a 2^2
association split: written `zahu/zahv` and `ah_wslp2` multiplication grouping
(`H`) versus written `akz_h`, `zcoef0`, and reciprocal post-factor grouping
(`K`). Score `ah_wslp2`, `akz`, the explicit A33 component, skew, and full
`zfw_tem`. Main effects plus HxK interaction are recorded. Exact ownership
requires full `zfw_tem` to pass the unchanged `1e-15` bar and all planted
roll/sign/coefficient controls to fire.

On confirmation, implement one coupled DINO-only literal builder shared by
the explicit Redi operator and implicit K33 consumer; generic cards remain
byte-pinned. Then release 78.S.1-S.3 in order and resume the ZDF/Asselin tail.
If no arm closes, new operand instrumentation is required and must use the
`__MEASURED_` SLOT convention, reserved units, full-halo writers, tracked-only
porcelain gate, checkout `PYTHONPATH`, and SHA-pinned producer.
