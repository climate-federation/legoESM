# Preregistration — ORCA2 round 245 midpoint V-depth operand split

Date: 2026-10-11. Frozen base: `a702e3a9d`. Scope: OMT-4, kt=1 stage 1,
external substeps 3 through 6, measured separately for the independent and
given-NEMO-entry labels. This round first splits the substep-3 midpoint V
depth, then follows the measured consequence into the registered substep-6
`va_e`/`zhvp2_e` pair. No deck, selector, carried state, sea-ice field, or
NEMO record changes are authorized.

## Source statement and fixed protocol

The executing OMT-4 build forms the V depth in this source order at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:487-519`:

```
zhvp2_e = hv_0 + 0.5 * r1_e1e2v
         * (e1e2t(j) * zsshp2_e(j) + e1e2t(j+1) * zsshp2_e(j+1))
         * ssvmask
```

The split therefore scores `hv_0`, `r1_e1e2v`, south `e1e2t`, south
midpoint SSH, north `e1e2t`, north midpoint SSH, and `ssvmask` in that written
order. Candidate operands must be the exact values consumed by the production
D114 route on the current tree. NEMO geometry operands come from the card's
admitted raw mesh; midpoint SSH and final depth come from the admitted passive
substep and advective-mean streams. The existing round-244 observer must remain
bit-passive under CPU production JIT, fp64, x64, and libm.

## Frozen predictions and falsifiers

1. **R245-P1 — instrument.** Both record twins remain exact on defined
   payload, completed traced and ordinary states are array-identical, every
   operand is float64, and both labels agree. Any failure **INVALIDATES** the
   measurement.
2. **R245-P2 — already landed inputs.** Raw `hv_0` is bit-exact through the
   existing atomic fold unit, and the D114 masks make `ssvmask` bit-exact.
   Any unequal cell in either row **REFUTES** this prediction and stops at that
   earlier geometry statement.
3. **R245-P3 — first debt.** The first unequal operand at substep 3 is a
   fold-halo horizontal-area operand: candidate `e1e2t` on row `jpj`, or its
   associated `r1_e1e2v`; it is not the midpoint SSH. If all area operands are
   exact, or midpoint SSH is first, **REFUTE** P3 and retain the measured first
   operand instead.
4. **R245-P4 — causal closure.** Substituting the first unequal recorded/raw
   operand alone closes the substep-3 depth from 35 unequal cells to zero and,
   when replayed through substeps 4–6, collapses the registered substep-6 pair
   (`va_e` 129, depth 120, `zhV` 129) to zero. Any residual at substep 3
   **REFUTES** single-operand closure; an unchanged substep-6 pair
   **REFUTES** ownership of the pair.
5. **R245-P5 — landing predicate.** A package change is eligible only after
   one cited NEMO statement is isolated, its direct output is bit-exact given
   NEMO operands, the cancelling pair is scored atomically, and the full
   OMT-1..4/rung-0/rung-10/GYRE/DINO/tank gates satisfy Decision 96. Otherwise
   this round is **HELD** with the next operand named.
6. **R245-P6 — controls.** Plants for label coverage, source order, one-bit
   operand movement, replay closure, production-route identity, and passivity
   must each refuse.

## Stop conditions

No configuration choice, NEMO source edit, acquisition, or partial member of
the measured cancelling pair may land. If the admitted records cannot support
the split, report the exact missing stream and write a fail-closed acquisition
instead of inferring an operand. Failed predictions remain in the receipt.
