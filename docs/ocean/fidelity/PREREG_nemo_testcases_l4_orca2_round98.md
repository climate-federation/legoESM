# Preregistration — ORCA2 round 98 barotropic Coriolis residual walk

Date: 2026-10-02. Base: `96ded53362ecda9d826788b4b2543db5fe9e7bff`.
Scope is ocean only. Every number is labelled **independent** because hierarchy
rung 0 starts from NEMO's own from-rest state.

## Frozen source order and constraints

The admitted round-96 record and round-97 observer are the instrument. NEMO
forms the eight frozen EEN coefficients in
`ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:1213-1265`, then applies
the four U products and four V products with the written pairwise association
in `ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:1369-1392`.
For `nn_e3f_typ=0`, the divisor is the masked four-cell `e3f_0vor`, F-fold
exchange, and zero substitution in
`ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynvor.f90:912-937`.

Round 97 measured the current mesh-`e3f_0` baseline and a source-`e3f_0vor`
one-variable arm. The source arm reduced substep-2 U from 15,692 unequal cells
at maximum `1.6425537782543814e-7` to 68 cells at maximum
`2.9617669311254642e-8`, with the remaining maximum on the northern fold row
`(j=147, i=134)`. It did not close the substep-1 signed-zero boundary.
Decision 84's shared production landing is not present at this base and will
not be duplicated here.

## Frozen measurements and predictions

1. Extend the existing walker, not the NEMO record, to expose the eight
   legoESM literal coefficients and the eight source-associated products at
   substeps 1 and 2. Keep the admitted entry, slow forcing, raw barotropic
   histories, masks, precision, and CPU/JIT protocol unchanged.
2. Reproduce round 97's baseline and source-divisor rows before accepting any
   new result. A one-ULP coefficient/product mutation must fire.
3. Evaluate the substep-1 four-pair application in the compiled association
   from the same bit-exact mid-step velocities. Distinguish a coefficient-sign
   cause from a product/sum signed-zero cause; do not treat zero magnitude as
   bit identity.
4. At substep 2, test one association at a time after the source divisor:
   first the north-fold operand used by the live F stretch / frozen
   coefficient builder, then the final coefficient-to-velocity pairing. No
   package statement may land unless the changed operator is bit-exact given
   NEMO inputs and all shared-card gates pass.

| ID | Frozen prediction | Confirmation | Falsification / action |
|---|---|---|---|
| R98-P1 | Round 97 is reproducible at this base. | Baseline and source-divisor score dictionaries equal the committed round-97 artifact exactly. | Any changed value is **REFUTED**; stop and reconcile the instrument before recording new numbers. |
| R98-P2 | The substep-1 mismatch is a signed-zero construction issue in the compiled four-pair application, not a nonzero coefficient error. | All unequal substep-1 cells have zero magnitude; a source-associated scalar/strict application from the same inputs changes their sign pattern and is scored independently. | Any nonzero error or upstream non-bit velocity is **REFUTED**; return to the first moved operand. |
| R98-P3 | After source `e3f_0vor`, the first nonzero residual is owned by a northern-fold association in the coefficient builder. | A fold-only arm reduces the 68-cell, `2.9617669311254642e-8` substep-2 U row and moves the `(147,134)` maximum toward NEMO without changing non-fold coefficients. | No improvement, any non-fold movement, or an earlier row leaving bit identity is **REFUTED**; test coefficient-to-velocity association next and keep the fold arm out of production. |
| R98-P4 | GYRE does not execute a tripolar-fold correction. | If a package fix becomes eligible, GYRE's certified ladder and day-30/240/360 outputs are byte-identical to this branch's baseline. | Any GYRE movement invokes the full Decision 43/45/55/59 predicate; otherwise hold the change. |

## Landing and refusal bar

The first non-bit statement remains owner of the walk. A zero-valued signed
bit mismatch is non-bit. No configuration choice, stabilizer, threshold,
carried-state convention, sea-ice selector, or `unmeasured_features` entry may
change. The absent Decision-84 divisor landing cannot be silently folded into
another package change. If the admitted record cannot distinguish coefficient
construction from application, write a rank-complete, self-describing
coefficient/product acquisition under a new target and report
`ACQUISITION_NEEDED`.
