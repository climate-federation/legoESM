# Preregistration: GM post-chain source association, round 72

Date: 2026-08-30. Frozen before measurement. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

Round 71 makes both direct Kmm geometry operands exact (`e3w_Kmm` and
`rn2b`) and reduces row 8.8 from `9.66343e-10` to `2.42076e-11`.  The first
red coefficient operand is now `zaeiw` (`4.66204e-12`, 437 columns), while
`zhw` is bit-exact and `zn`, `zah`, and `zRo` are individually at their
`1e-12` bars.  NEMO does not evaluate the remaining chain in legoESM's
normalized expression.  It evaluates, in order
(`ldftra.F90:776-789`):

1. `zRo = max(2e3,min(.4*zn/zfw,40e3))`;
2. `zaeiw = zRo*zRo*sqrt(zah/zhw)*ssmask`;
3. `z1_f20 = 1/(2*omega*sin(rad*20))`;
4. `zzaei = min(1,abs(ff_t*z1_f20))*zaeiw`;
5. `zaeiw = min(zzaei,paei0)`.

Reuse the admitted round-71 production capture and the existing direct
`zRo/zah/zhw/zaeiw` dumps.  Score a frozen substitution matrix at the
unchanged `1e-12` coefficient bar and `1e-12` row bar:

- current production association (required red plant);
- literal NEMO post-chain with own `zRo/zah/zhw`;
- literal post-chain with oracle `zRo` only;
- literal post-chain with oracle `zah/zhw` only;
- literal post-chain with all three oracle operands (must reproduce oracle
  `zaeiw`, otherwise the proposed formula or staggering is invalid).

Ownership is `POST_CHAIN_ASSOCIATION` only if the own-operand literal arm
passes `zaeiw`, its U-face average, and row 8.8, while the production arm
remains red.  If only an oracle-operand arm passes, localize to the first
required operand.  If the all-oracle arm fails, stop `INVALID_LITERAL_REPLAY`.
No production change is admitted before this classifier fires.  Round 70's
`e3t(1)` proxy remains a mandatory strong-red planted violation.

Frozen round-71 receipt SHA-256:
`6500acfa930c0342430fd1e57cfb1da023b0978e8fda3561e6133ffe12368821`.
