# Preregistration: NEMO-testcases L2 GYRE round 107 TKE RHS intermediate walk

Date: 2026-09-17. Frozen at incoming lane tip
`a89ab761ca98920016f4eee2f0ecae3ccd781dad`, before any round-107
measurement. Evidence belongs under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round107/`.

## Ordered statement and compiled source

This round stays at the first unresolved statement in the authorized TKE
subwalk. The record-producing compiled program assigns
`zfact3 = 0.5_wp * rn_ediss` at
`GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90:261`, writes the three
matrix rows at `:434-436`, then evaluates the right-hand side at `:439-442`:

```fortran
en = en + rn_Dt * ( p_sh2
                   - p_avt * rn2
                   + zfact3 * dissl * en
                 ) * wmask
```

Round 106 established that this final assignment is BIT in eager execution,
non-bit in 5,117 of 20,416 cells under an isolated JIT, and non-bit in 11,024
cells at `5.551115123125783e-17` through the production step after the held
round-105 shear routing makes `p_sh2` BIT. Source-order identity boundaries and
the literal NEMO dissipation multiplication tree did not close it. Round 107
therefore scores the expression's intermediate values themselves in compiled
order, one returned intermediate per arm.

The existing consolidated round-46/51 stage gate, existing private RHS
selector, and existing NEMO-from-NEMO block replay are extended. No second
harness or public configuration is created.

## Immutable trajectory arm and stage order

The immutable trajectory arm remains
`phase3/merge_main_2026-09-17/after/{ladder.json,day_gap.json}`. Its headline
rows are kt2 T/S/U/V `1.4210854715202004e-14` /
`2.1316282072803006e-14` / `2.7377110452773967e-12` /
`3.2849219221489645e-12`, kt3 T/S `1.627497246303733e-4` /
`6.327735185607253e-6`, and day-30 T RMS
`1.2397011296352737e-2` K.

Decision 41 still makes kt1 stage-1 U the global first owned non-bit output:
7,620 unequal cells at `5.421010862427522e-20`. This downstream TKE walk is
authorized attribution work only. No round-107 arithmetic candidate may land
while that upstream output remains owned non-bit. If that condition has
unexpectedly changed, the candidate must instead pass the complete 954-row
Rule-12 comparison and days 1-30 before promotion.

## Frozen predictions and falsifiers

**P1 — reproduced boundary.** Reapplying the held round-105 shear-routing
patch only in the measurement tree will reproduce BIT `zd_up`, `zd_lw`,
`zdiag`, and `p_sh2`; eager `en_rhs` will be BIT; isolated-JIT `en_rhs` will be
5,117/20,416 unequal; and production-JIT `en_rhs` will be 11,024/20,416
unequal with maximum `5.551115123125783e-17`. REFUTED if any count or maximum
differs.

**P2 — one-output intermediate walk.** Eight separate arms will return exactly
one selected value from the expression while still returning the final RHS:
`p_avt*rn2`, `zfact3*dissl`, the complete dissipation product, the
shear-minus-stratification subtotal, the complete parenthesized sum, the
timestep product, the masked increment, and the final accumulation. No arm
will return a tuple of all intermediates. NEMO references will be rebuilt in
the source association from the admitted recorded operands; the existing
replay must still reproduce the recorded final output BIT before any
intermediate row is interpreted.

Every isolated-eager row is predicted BIT. Under isolated JIT the first three
rows are predicted BIT and the shear-minus-stratification subtotal is
predicted to be the first non-bit row, because lowering may contract
`p_sh2 - p_avt*rn2` while eager execution rounds the product before the
subtraction. The production-step JIT is predicted to have the same first
non-bit boundary, though its unequal-cell count may differ. REFUTED if an
earlier row is non-bit, if the subtotal remains BIT, or if the replay no
longer closes.

**P3 — optimized-code discriminator.** The isolated first-non-bit arm will be
compiled with XLA text and LLVM dumping enabled. Its pre-optimization operation
order and eager result are predicted to contain a separately rounded multiply
then subtraction, while optimized HLO or LLVM is predicted to show a fused or
reassociated multiply-subtract path with no binary64 rounding boundary between
them. REFUTED if the optimized representation preserves separate operations
and the emitted machine-level path contains no contraction/reassociation. A
compiler dump without a module/function match is UNMEASURED, not evidence.

**P4 — full-precision `reduce_precision` discriminator.** Only if P2 names the
predicted contraction boundary, one candidate will wrap the `p_avt*rn2`
product in `jax.lax.reduce_precision(..., exponent_bits=11,
mantissa_bits=52)` before subtraction. Before the candidate is measured, a
committed control must prove bitwise primal identity for normal finite values,
the smallest and largest subnormals, both signed zeros, both infinities, and
multiple NaN bit patterns; it must also prove finite reverse-mode derivative
one on finite physical inputs. The candidate is predicted to make the
shear-minus-stratification subtotal and final RHS BIT in both isolated JIT and
the production step while leaving earlier rows BIT. REFUTED by one unequal
cell, one changed special-value bit pattern, or a non-finite/non-unit physical
gradient. If P2 names another boundary, this candidate is not run and a frozen
addendum is required before any replacement.

**P5 — plants.** The returned-intermediate scorer will advance one finite,
exact reference cell by one ULP. The target row must gain exactly one unequal
cell, print `STATUS PLANT-FIRED`, and exit nonzero. The final RHS plant must
continue to gain exactly one unequal cell and exit nonzero. REFUTED if either
plant perturbs a zero/no-op, prints PASS, or exits zero.

**P6 — landing and magnitude.** Even if P4 closes the TKE RHS, the arithmetic
remains HELD because kt1 stage-1 U is earlier in stage order and the required
round-105 routing itself failed Rule 12 in 59 rows / 203,983 cells. The
one-ULP RHS residue is not a demonstrated owner of kt3 T or the day-30 gap.
No trajectory arm is run for a private discriminator. If the upstream stage
condition unexpectedly closes, the full ladder and days 1-30 become mandatory:
no AT-BAR row may leave the bar, first-over-bar may not move earlier, and every
moved row must be registered.

**P7 — other configurations.** DINO shares the literal prognostic-TKE
statement, so any eventual unconditional rounding primitive requires a DINO
production-step before/after row. This round calls no private selector safe for
DINO. LOCK_EXCHANGE and OVERFLOW are predicted not to instantiate the
statement; ORCA2 remains UNMEASURED-WITH-SPEC and requires a one-step
production comparison before promotion. REFUTED for either small tank if its
resolved card executes the statement.

**P8 — instrumentation neutrality.** With both new private selectors empty,
the restored-production stage twin and focused tests will reproduce the
incoming tip. REFUTED by any default-path bit movement. Measurement-only
arithmetic will be removed from production; a useful candidate may be retained
only as a held manifest with its veto in the header.

## Measurement order and controls

1. Commit this preregistration.
2. Extend the consolidated gate and literal TKE trace with a private
   one-intermediate selector and fail-closed ULP plant; add direct tests.
3. Reapply the held round-105 routing only in the measurement tree and
   reproduce P1 under eager, isolated JIT, and production JIT.
4. Run the eight intermediate arms separately in compiled order. Stop at and
   name the first non-bit statement; preserve every row.
5. Dump and inspect optimized HLO/LLVM for that arm. Run P4 only if P2 names
   its predicted boundary and the identity/gradient controls pass first.
6. Restore landed production, rerun the canonical stage twin, then run the
   separate read-only Codex refutation pass, citation gate and shifted-citation
   plant, and focused tests. Quote terminal summaries rather than shell status.

## Scope limits

CPU only, fp64, `JAX_ENABLE_X64=1`. No NEMO source is modified and neither
`makenemo` nor `mpirun` is run. No public configuration, default, scheme
selection, threshold, carried state, restart schema, year harness,
reconciliation gate, freshwater pair, #1484 guard, or immutable trajectory
baseline is changed. No stabilizer is introduced. GitHub issue #1455 is not a
measurement source for this round; the clone has only a local remote and the
direct issue fetch is unavailable.
