# Preregistration — round 173, wet-only retry of the paired solve-input record

Committed before constructing or running a corrected Round-172 directed arm.
The operator's Round-172 baseline reached step 1440 and wrote `STOP 0`, but
the e3t arm stopped at step 1082 with NaNs.  The failed arm is retained as
evidence and is never scored.

## Failure named from the compiled program

The compiled adaptive-implicit branch forms the ordinary diagonal with
`e3t_3d * (1 + r3t(Kaa) * tmask)` at
`GYRE_OMIP_L2_P3_SM_R172SOLVEPAIR/BLD/ppsrc/nemo/trazdf.f90:490-504`.
Therefore dry cells retain positive reference thickness.  The Round-172
reader instead selected the imported array without a mask at lines 494--497.
The imported legoESM array has exactly 3,120 zero values in its first frame,
equal to 104 dry horizontal cells times 30 active levels.  Feeding those zeros
to NEMO made dry-column diagonals singular; the compiled recurrence at
`trazdf.f90:561-565` then divided by the zero pivot, and NEMO's own step
control reported NaNs at step 1082.

This round changes no compiled source and does not rebuild NEMO.  It reuses
the existing Round-172 binary after proving its hash equals the baseline and
failed-arm copies.  A new input conversion starts from NEMO's recorded
`e3t_Kaa` and `rhs_T` at every step and replaces only the 18,000 wet active
cells with the corresponding ordinary legoESM values.  Thus the unselected
dry cells remain NEMO's own operands and the directed family is unchanged.

## Frozen predictions and falsifiers

1. The corrected raw-input admission reports exactly 18,000 selected wet
   cells and 3,120 retained dry active-level cells per family per frame.  All
   retained cells are bit-identical to the Round-125 record.  Any other count
   or any retained-cell difference refuses the retry.
2. The existing Round-172 baseline step-1080 and step-1440 restarts are
   byte-identical to the Round-125 producing run, and its copied executable is
   byte-identical to the installed Round-172 binary.  Any mismatch refuses
   reuse and requests a new acquisition target instead.
3. New `e3t_wet` and `content_wet` arms each reach step 1440 with `STOP 0`.
   Either arm stopping early, producing a NaN, or lacking the two required
   restarts refuses the scientific comparison.
4. Each directed step-1440 restart differs from the admitted baseline.  An
   identical restart makes that arm non-discriminating and refuses it.
5. Multiplying the first frame's selected wet values by exactly
   `1 + 2**-20` changes all 18,000 selected values, while every retained dry
   value remains bit-identical; the converter prints `STATUS PLANT-FIRED` and
   exits nonzero.  A success exit or a dry-cell movement invalidates the
   converter.
6. After acquisition, the day-240 T3D RMS is scored for baseline,
   `e3t_wet`, and `content_wet` against the same Round-125 restart.  The arm
   removing more of the production `1.241262968697578e-03` K complete-K/e3w
   remainder is promoted to its producer walk.  The frozen ordering remains
   e3t greater than content; equality or the reverse ordering refutes it.

The paired arms are forced-input magnitude sensitivities, not source-exact
landing proof.  No production physics, configuration, carried state, restart
schema, card default, immutable before arm, or pending user decision changes.
