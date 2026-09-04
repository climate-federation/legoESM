# NEMO testcase lane 2 GYRE — phase-3 round-15 preregistration

Date: 2026-09-04
Session: `01a05cb9-7625-7f40-9e83-b3fa767b1945`
Reviewed baseline: `2b322c42f1c5c6d82ad6b930436b6f74295c486c`

## ASKED oracle and precision-policy change

User Decision 4 requires a new scalar-math oracle and a certification-only
system-libm transcendental policy.  The old vectorized GYRE oracle remains
retained and is flagged V1; it is not silently overwritten.  V2 will be built
as `cfgs/GYRE_OMIP_L2_P3_SM` from shipped `GYRE_PISCES` with
`arch-conda-scalarmath.fcm`, then receive byte-identical copies of V1's
config-local `MY_SRC`, `EXP00`, and `cpp_GYRE_OMIP_L2_P3.fcm`.  The target
preprocessor keys remain `key_qco key_vco_1d3d key_RK3`.

Pre-implementation search over `packages/` and `src/` found no
`pure_callback`, `ctypes`, or `CDLL` implementation.  The shared
`PrecisionPolicy` at `packages/core/legoesm/core/precision.py:80` is presently
dtype-only.  One core implementation will therefore add a `native|libm`
platform policy, defaulting to `native`; NEMO-identity testcase harnesses will
request `libm` explicitly.  This is a precision/platform selection, not a
physics option and not a per-card operator arm.

The scalar-math arch SHA-256 is
`132f7a0500c4f0e86d8d3bf7864974a82e1dea5d83166dcfdfaf409e2ca04561`.
Its Fortran flags are `-fdefault-real-8 -O3 -funroll-all-loops
-fcray-pointer -ffree-line-length-none -fallow-argument-mismatch
-fno-tree-vectorize`.  V2 is eligible only if `nm -D nemo.exe` reports zero
`_ZGV*` symbols and all copied deck/source SHA-256 values equal V1.

## V1-to-V2 record predictions

The resolved deck selects two-band shortwave (`ln_qsr_2bd=.true.`) and rejects
RGB (`ln_qsr_rgb=.false.`).  The executing `qsr_2BD` evaluates the attenuation
EXP calls at `src/OCE/TRA/traqsr.F90:665-680,696-702`, then adds that increment
to the RK3 tracer RHS.  Therefore the stage-3
`oracle_qsr_stage3_kt00000001.bin` record is predicted to have an identical
first payload (`qsr`) and a **DIFFERENT** second payload (the temperature RHS
increment).  Stage-3 tracer checkpoints at and after `tra_qsr`, stage-3 state,
and step-entry records `kt=2…10` are predicted DIFFERENT as that increment
propagates.  The `kt=1` entry and records completed before the stage-3
shortwave call are predicted IDENTICAL.

The GYRE analytic SBC uses COS/SIN at
`src/OCE/USR/usrdef_sbc.F90:90-120,122-144,161-176`; the user's machine-level
discriminator found JAX and scalar glibc SIN/COS bit-identical.  Its `qsr`,
`qns`, `emp`, `utau`, and `vtau` fields are therefore predicted IDENTICAL
between oracle binaries.  EOS/HPG/vorticity and the first-step external-mode
records are likewise predicted IDENTICAL because those executed chains do not
contain the changed EXP/TANH calls before forcing propagates.

The briefing named `zdftke nn_etau=1` as a possible changed path, but the
actual resolved GYRE deck says `nn_etau=0`
(`cfgs/GYRE_OMIP_L2_P3/EXP00/namelist_cfg:220` and
`output.namelist.dyn:311`).  Thus the EXP arms at
`src/OCE/ZDF/zdftke.F90:492-511` are dead for this run and TKE records are
predicted IDENTICAL.  That source-resolved correction is part of the result,
not a deck change.

Every `oracle_*.bin` produced by like-for-like ten-step V1 and V2 runs will be
byte-compared and classified IDENTICAL or DIFFERENT; for a structured changed
record the first differing named payload will be reported.  The production-JIT
fp64 whole-step register and `kt=1…10` sweep will then be re-scored against V2.

## Eligibility falsifier

With oracle V2 and `PrecisionPolicy(transcendentals="libm")`, both of these
must be bit-exact before further owner work:

1. the two-band shortwave increment in the `l2_qsr_before/after` boundary;
2. all executed `usrdef_sbc` forcing fields (`qsr`, `qns`, `emp`, `utau`, and
   `vtau`).

Mismatch counts are reported before and after.  Any nonzero after-count stops
the round without tuning.

## Conditional ordered walk

Only after eligibility passes, `_nemo_source_round` will be renamed to public
`nemo_source_round` with no alias and used for the corrected-velocity plus
`zub/zvb` composition.  The registered measurement is the change from the
Round-14 corrected-velocity maximum `2.799952110443815e-17`.

The next physical boundary is the stage-1 slow forcing passed into the
external mode.  The preregistered candidates, in order, are: the NEMO
source-order depth-weighted 3-D RHS sum, its reference-depth `hu/hv` division,
surface-stress addition, bottom-drag contribution/time level, and the handoff
to the substep update.  A config-local WRITE-only record will expose each
operand before any source change.  CONFIRM requires the first non-bit-exact
operand plus residual-scale movement in a one-variable private arm; otherwise
the boundary remains DEBT.  Any fix must be shared, literal, and free of a
GYRE/card guard.

No independent review of this preregistration has occurred.
