# Preregistration — round 166, developed shear-production walk

Committed before scoring any new Round-166 row.  The entry is NEMO's admitted
day-180 restart at step 1080 and the measured program is legoESM's production-
jitted step 1081.  This extends the existing developed-state bridge and the
Round-104/105 shear walk; it does not introduce a second bridge or an isolated
closure as scientific evidence.

## Compiled program and reference calibration

The executed Round-164 compiled program calls `zdf_phy(kstp,Nbb,Nbb,Nrhs)` at
`GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/stprk3.f90:167-168`, which calls
`zdf_sh2(Kbb,Kmm,avm_k,sh2)` at
`GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/zdfphy.f90:317-320`.  Both
velocity and live-metric time levels are therefore the step-entry `Nbb` slot.
The no-Stokes branch writes the U-face shear at
`GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/zdfsh2.f90:97-103`, the V-face
shear at `:104-108`, combines both onto T points at `:111-114`, and zeroes the
two boundary levels at `:116-119`.  Its masks are the compiled W-face products
at `GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/dommsk.f90:234-242`.  Its
live face ratios are the vector-form writes at
`GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/domqco.f90:211-218`.

Before a model row is interpreted, rebuild NEMO's own `zsh2u`, `zsh2v`, and
`p_sh2` in the literal compiled association from the admitted day-180 restart,
the Round-164 `avm_entry`, the card's recorded geometry/masks, and the record's
own `sh2`.  Prediction: the reconstructed `p_sh2` is BIT against all 17,400
wet solved interfaces.  REFUTED if it is not; attribution stops because the
reference-side reconstruction is then not calibrated.

## Production-JIT compiled-order walk

Score, separately and in source order:

1. step-entry U/V face velocities and their vertical differences;
2. carried `avm_k` and the U/V face sums;
3. step-entry `r3u/r3v`, the four live `e3w*(1+r3)` factors, and their U/V
   divisor products;
4. `wumask/wvmask` and the two coast factors;
5. the U-face and V-face `zsh2` statements;
6. the final T-point `p_sh2` assignment.

Every production row comes through `LatLonCGridOceanModel.step` under its
production JIT.  Any trace extension is accepted only if its returned state is
bit-identical to an ordinary unobserved production step.  Isolated NumPy/JAX
replays calibrate the reference and attribute operands but do not replace a
production row.

Frozen predictions:

1. The existing restart bridge checks remain BIT for U/V, SSH and carried
   `avm_k`; the velocity, viscosity and mask operand rows are therefore BIT.
   REFUTED if any differs.
2. Round 165's headline reproduces exactly: `p_sh2` differs in
   17,400/17,400 wet interfaces with maximum
   `2.1204821986655657e-15` s-2, and its TKE RHS differs in 14,649/18,000
   cells with maximum `3.053494349730679e-11` m2 s-2.  REFUTED if either
   changes.
3. The first non-bit shear boundary is predicted to be the live Kmm face
   metric/divisor consumed at `zdfsh2.f90:102,107`, inherited from the QCO
   face-ratio path.  If all four face metrics and both divisors are BIT, this
   prediction is REFUTED and the first non-bit U/V face or final T-point
   statement replaces it.
4. A one-ULP change to a nonzero consumed step-entry velocity cell must move a
   registered vertical-difference row and the final production `p_sh2`, then
   exit nonzero with `STATUS PLANT-FIRED`.  A plant on a zero, dry, unconsumed,
   or isolated-only value is invalid.

## Candidate and verdict

If the first non-bit value is inherited from an upstream operand, no shear
patch is formed.  The held Round-105 routing split is re-evaluated only if the
walk reaches its exact free-surface routing statement and closes under the
production JIT.  If the first non-bit value is instead a one-variable shear
statement given bit-exact operands, that statement alone may become a
candidate and must pass Decision 43/45 as amended by Decision 55: full ladder,
30 days, 360 days, every moved row, and every executing card including DINO.
No configuration or carried-state choice is authorized.  Decisions 53, 54,
57 and 58 remain pending and untouched.

The immutable production headline is unchanged: kt2 T/S/U/V AT-BAR,
first-over-bar kt3, day-30 T rms `6.572574374770603e-05` K, day-240
`1.644836070117868e-02` K, and day-360 `1.122566001855131e-02` K.
