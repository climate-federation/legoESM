# Preregistration: GM `zn` square-root forward value, round 74

Date: 2026-08-30. Frozen before measurement. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

Round 73 localizes row 8.8 to `zn`: source-recomputed and stored T-point
Coriolis arms are identical, while oracle `zn` alone makes `zRo`, `zaeiw`,
the U-face average, and row 8.8 pass.  The direct Kmm `rn2b` and `e3w`
operands are bit-exact and the source-ordered left reduction has already been
selected.

The remaining forward difference is explicit.  NEMO evaluates
`SQRT(MAX(rn2b,0))*e3w` (`ldftra.F90:747-752`).  legoESM evaluates
`sqrt(max(zn2,1e-30))*e3w` to avoid the undefined zero derivative.  The
`1e-15` forward value at every zero N² slot is small enough for `zn`'s own
bar, but round 73 proves its accumulated remainder is amplified by `zRo^2`.

Reuse exact captured `rn2b/e3w`.  Build `zn` with the existing source-order
left loop under two arms: the current `1e-30` floor (required downstream red)
and exact-forward `sqrt(max(rn2b,0))`.  Propagate each through the registered
literal `zRo/zaeiw` and bolus chain.  The exact-forward arm must pass `zn`,
`zRo`, `zaeiw`, `aeiu`, and row 8.8 at `1e-12`; a negative-N² perturbation
must remain exactly clipped and a zero-N² planted point must distinguish the
two arms.

If it passes, own `ZN_SQRT_FORWARD_FLOOR`.  The production fix must use an
exact NEMO forward value with an explicitly finite zero derivative (custom
JVP or equivalent), be selector-scoped with faithful defaults only on the two
DINO cards, retain byte identity elsewhere, and ship JIT/grad plus planted-red
tests.

Frozen round-73 receipt SHA-256:
`c7445a0e26e4b74999bbe89f79c043ad4e9d9754f357d0e94ae136471cc9961d`.
