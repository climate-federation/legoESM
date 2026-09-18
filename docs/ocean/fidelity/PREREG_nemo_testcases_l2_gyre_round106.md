# Preregistration: NEMO-testcases L2 GYRE round 106 production-JIT TKE RHS walk

Date: 2026-09-17. Frozen at incoming lane tip
`27ffba40e75b0927a313273e3a7822298ce9764b`, before any round-106
measurement. Evidence belongs under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round106/`.

## Ordered statement under test

Round 105 made the shear operand bit-exact given NEMO's recorded stage entry
in both eager and production-JIT execution, but the immediately consuming TKE
RHS remained non-bit only in production JIT: eager was 0 unequal cells while
production JIT was 11,024 of 20,416 unequal cells with maximum absolute error
`5.551115123125783e-17`. The routing split failed Rule 12 and is held at
`scripts/validate/ocean_fidelity/testcases/manifests/nemo_testcase_l2_gyre_round105_held_shear_routing_split.patch`.

The record build's compiled program writes the matrix at
`GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90:434-436`, then executes
one RHS assignment at `:439-442`:

```fortran
en = en + rn_Dt * ( p_sh2
                   - p_avt * rn2
                   + zfact3 * dissl * en
                 ) * wmask
```

The round-106 walk stays in the consolidated round-46/51 stage tool and drives
that statement through the full production step from NEMO's recorded entry.
It first reapplies the held round-105 routing in the measurement arm, then
inserts one IEEE-identity source-rounding boundary at a time in compiled
expression order. An isolated JIT of a copied expression is not production
evidence.

## Immutable trajectory arm and stage order

The immutable trajectory arm remains
`phase3/merge_main_2026-09-17/after/{ladder.json,day_gap.json}`. Its headline
rows are kt2 T/S/U/V `1.4210854715202004e-14` /
`2.1316282072803006e-14` / `2.7377110452773967e-12` /
`3.2849219221489645e-12`, kt3 T/S `1.627497246303733e-4` /
`6.327735185607253e-6`, and day-30 T RMS
`1.2397011296352737e-2` K.

Decision 41 still makes kt1 stage-1 U the global first owned non-bit stage
output. This downstream vertical-physics walk is authorized attribution work,
not permission to land a downstream statement. No round-106 trajectory arm
will be promoted unless the full stage condition unexpectedly closes and the
954-row Rule-12 gate is then run and passes.

## Frozen predictions and falsifiers

**P1 — reproduced production boundary.** With the held routing reapplied only
in the measurement arm, `zd_up`, `zd_lw`, `zdiag`, and `p_sh2` will remain BIT
in eager and production JIT. `en_rhs` will reproduce 0 unequal cells in eager
and 11,024/20,416 unequal cells, maximum `5.551115123125783e-17`, in production
JIT. REFUTED if any exact predecessor moves or either RHS count differs.

**P2 — one-boundary source-order walk.** Separate production-step arms will
materialize exactly one of these values with the shared IEEE-identity
`nemo_source_round`: (a) `p_avt*rn2`, (b) `zfact3*dissl`, (c)
`zfact3*dissl*en`, (d) `p_sh2-p_avt*rn2`, (e) the complete parenthesized
sum, (f) `rn_Dt*sum`, or (g) the trailing `*wmask` increment immediately
before the final `en + increment`. All other associations remain the current
production expression. The prediction is that (g) alone makes `en_rhs` BIT,
because it prevents contraction/reassociation across the final accumulation;
arms (a)-(f) are predicted to leave at least one unequal cell. This prediction
is REFUTED if another arm is the first bit-exact arm or if (g) remains non-bit.
Every failed arm remains reported.

**P3 — complete-source discriminator.** If no individual arm closes the row,
a diagnostic arm will materialize every listed compiled-expression boundary.
It is predicted BIT. REFUTED if one cell remains unequal; that result would
exclude missing source-round boundaries as a complete explanation and force an
operand/lifetime walk rather than another barrier guess.

**P4 — production plant.** The exact reference for the first newly exact
production row will be advanced by one ULP at one wet, finite cell. The target
row must move from 0 to exactly 1 unequal cell, the gate must print
`STATUS PLANT-FIRED`, and the process must exit nonzero. REFUTED if the plant
does not change exactly one scored cell, prints PASS, or exits zero.

**P5 — attribution and landing.** A boundary is named only when its
production-step arm, not an isolated closure, closes the row while all earlier
compiled outputs remain BIT. Even then the result is HELD because kt1 stage-1
U remains an earlier owned non-bit output. If that upstream condition is found
to have changed, the candidate must instead run the full 954-row Rule-12 gate
and days 1-30 against the immutable arm; no AT-BAR row may leave the bar,
first-over-bar may not move earlier, and every moved row must be registered.
Local exactness never overrides either stage order or Rule 12.

**P6 — DINO and other tanks.** The compiled RHS statement is shared by the
DINO literal TKE path, so any eventual unconditional source-rounding boundary
is a DINO risk until measured through DINO's production step; round 106 will
not call it safe from GYRE evidence. LOCK_EXCHANGE and OVERFLOW are predicted
not to instantiate the literal prognostic-TKE statement. ORCA2 remains
UNMEASURED-WITH-SPEC: run a one-step before/after state comparison through its
production closure before promotion. REFUTED for either small tank if its card
does execute the statement. No unmeasured tank is called a pass.

**P7 — instrumentation neutrality.** With the new diagnostic selector at its
default, the restored-production stage twin and focused production-step tests
must remain bit-identical to the incoming tip. The selector is private and no
card/configuration field is added. REFUTED by any default-path bit movement.

## Measurement order and controls

1. Commit this preregistration.
2. Extend the existing consolidated stage gate with a private source-boundary
   selector, defaulting to no change, and a nonzero production-JIT plant.
3. Reapply the held round-105 routing only in the measurement tree and
   reproduce P1 under eager and production JIT.
4. Run P2 arms in compiled order; run P3 only if needed; run the plant against
   the first exact arm.
5. Restore landed production. Preserve any useful combined candidate only as
   a held manifest, and rerun the kt1/kt2 recorded-entry and chained stage
   tables on the clean default path.
6. Run the separate read-only Codex refutation pass, citation gate and shifted-
   citation plant, and focused tests. Record exact suite summaries, including
   pre-existing failures.

## Scope limits

CPU only, fp64, `JAX_ENABLE_X64=1`. No NEMO source is modified and neither
`makenemo` nor `mpirun` is run. No public configuration, default, scheme
selection, threshold, carried state, restart schema, year harness,
reconciliation gate, freshwater pair, #1484 guard, or trajectory baseline is
changed. The round does not promote the held shear routing or any downstream
TKE arithmetic while the upstream stage output remains owned non-bit.

## Frozen addendum: dissipation association discriminator

Frozen after P2/P3 were measured and refuted, and before this new arm is
implemented or measured. The original predictions above are unchanged. The
seven original arms left at least 10,402 cells unequal; the complete-source
arm equalled the `dissipation_product` arm at 10,402 cells and
`1.1102230246251565e-16`.

Source reinspection found that the first instrument inherited legoESM's
existing dissipation tree while labelling it as NEMO's tree. The compiled
record build first assigns `zfact3 = 0.5_wp * rn_ediss` at
`GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90:260-261`, then the RHS
evaluates `zfact3 * dissl * en` at `:441`. legoESM first evaluates
`diss_rate = c_eps * dissl_old` and its RHS evaluates
`0.5 * diss_rate * en`. Those are algebraically equal but have different
binary64 multiplication trees. Therefore the original arm called
`zfact3_dissl` did **not** implement its name; it materialized
`0.5 * (rn_ediss * dissl)`.

**P9 — one-variable NEMO dissipation tree.** A new production-step arm will
change only the dissipation term from
`0.5 * (rn_ediss*dissl) * en` to NEMO's
`(0.5*rn_ediss) * dissl * en`; the rest of the RHS and every operand remain
unchanged. It is predicted to make the production-JIT `en_rhs` BIT while the
eager arm remains BIT. REFUTED by one unequal cell in either execution mode.
If JIT remains non-bit, a second arm may materialize NEMO's separately written
`zfact3` scalar and the two multiplication boundaries to distinguish source
association from residual fusion; its prediction is BIT and its falsifier is
again one unequal cell. Neither arm is eligible to land under the unchanged
stage-order hold.
