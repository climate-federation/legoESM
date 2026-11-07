# NEMO testcase L2 GYRE phase-3 round-50 receipt

## Verdict

The recorded-input LDF operator is **CONFIRMED bit-exact** at kt=2 stages 1
and 3.  The first pre-fix departure is the outer divergence scale in compiled
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynldf_lev.f90:121-140`:
NEMO rounds `(ahmt*r1_e1e2t)/e3t` before multiplying the flux bracket.  XLA's
closed-constant route instead lowered it as reciprocal multiplication.  The
shared implementation now accepts the six recorded reciprocal operands as
dynamic given inputs and spells the compiled multiply/divide tree literally.
The U/V output divisions follow the same association at
the same compiled statement.

| intermediate | stage 1 unequal before | stage 1 unequal after |
|---|---:|---:|
| e2u/e3u/u and e1v/e3v/v products | 0 / 0 | 0 / 0 |
| divergence bracket | 0 | 0 |
| `(ahmt*r1_e1e2t)/e3t` | 4,336 | 0 |
| curl bracket / scale | 0 / 0 | 0 / 0 |
| post-LDF U / V | 974 / 866 | 0 / 0 |

Before the fix, stage-1 U split 616 first-wet-ring / 358 interior (20 wet
corners), and V 526 / 340 (25); rows span native j=4..23 (U) and 4..22 (V),
with boundary rows dominant.  Output residuals were usually ±1 ULP but
cancellation tails reached 10,240 ULP.  The scalar NumPy replay calibrates 0
cells at every stage/face; the model path now does too.  Its nextafter plant
exits 1.

## Rule 12 and trajectory

| card | disposition |
|---|---|
| GYRE-zco | TESTED: kt=2 stage 1/3 U/V 0 unequal |
| LOCK_EXCHANGE-zco | VALUE-INERT: resolved `ln_dynldf_OFF=T`; optional operands absent |
| OVERFLOW-zps | VALUE-INERT: resolved `ln_dynldf_OFF=T`; optional operands absent |
| ORCA2 | UNMEASURED-WITH-SPEC: acquire the same six reciprocals, six thicknesses, Kbb U/V, and post-LDF U/V |
| DINO | SHARED-STATEMENT RISK: separate branch has no recorded-input score; do not infer |

LOCK and OVERFLOW's executed switches are printed at `lock_kt1_10/ocean.output:615`
and `overflow_kt1_10/ocean.output:727`.  GYRE's ordinary kt=1..10 path remains
DEBT and first crosses at kt=2 U/V, unchanged at
`2.7478404751243857e-12 / 3.305560306813421e-12`; exact selection still needs
the live WS stage operands rather than a record injection.

The preserved stage-ZAD patch was reapplied only in a scratch clone.  It moves
kt=2 U/V to `2.7377110452773967e-12 / 3.284922138989399e-12`, but the Rule-12
comparison still has 57 worsening rows (largest `1.839493133678369e10` row
ULP).  It is therefore **REFUTED again and not landed**.

## Barotropic memory

The repaired admission reader registers the round-46 named/ranked stream; the
previously failing round-48 acquisition now passes.  NEMO internally preserves
all six histories, commits Kaa, seeds Kmm/current, and zeroes Kaa/adv at kt=2.
Against legoESM, adv/Kaa resets are exact.  U/V current and Kmm/Kaa differ only
in 4/35 signed-zero cells.  Numeric non-bit debt remains: Ubb/Vbb 6/5 cells
(`8.47e-22/4.24e-22`) and SSH b/bb/current/Kmm/Kaa 597--600 cells
(`4.34e-19`); Ub/Vb have only 5/36 signed-zero differences.  Owners are NEMO's
history rotation at
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynspg_ts.f90:749-757` and
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynspg_ts.f90:759-761`,
the AB3-AM4 weights at
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynspg_ts.f90:456-480`, and legoESM's
lossy deviation subtract/reconstruct path.  The one-bit memory plant exits 1.

## Disposition

ASKED: LDF localization/replay/fix, conditional ZAD retest, kt=1..10,
barotropic reader/score, Rule 12.  UNASKED: no NEMO run/build/source mutation,
no TKE/year-harness edit, no ZAD landing, no configuration choice.  Open:
thread live WS LDF operands; remove barotropic deviation reconstruction loss;
acquire ORCA2 and DINO evidence.
