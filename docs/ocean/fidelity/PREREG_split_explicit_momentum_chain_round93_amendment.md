# Round 93 preregistration amendment: post-fix stage control

Date: 2026-08-30. Frozen after the first A33-floor scoring attempt and before
the scorer is rerun. The invalid artifact is
`/tmp/dino_split_explicit_momentum_chain_round93.json`, SHA-256
`27c1e690420b185695b7bf199e5729834b9710f04bfb1c869242f52f6d015810`.

The attempt is invalid only because the round-91 control still requires both
post-stage W-slope substitutions to change the production field. Round 92 made
that post-stage pair the faithful production path, so those substitutions must
now be inert. For round 93 the control is therefore inverted to require exact
inertness when `--redi-zfw-a33-floor-factorial` is active. The frozen HxK arms,
the `1e-15` bar, ownership thresholds, oracle/component closure controls, and
all planted roll/sign/coefficient violations are unchanged.

The same correction resets the released ladder's first-divergence cursor
before scoring 78.T.1 through 78.S.3. This prevents the pre-factorial
temperature `zfw` observation from being reported after its registered
counterfactual has cleared it.

## Rule-1b clearance amendment

Frozen after the literal-builder replay at commit `413f1afb10b4` and before
the clearance classifier is changed. The replay artifact
`/tmp/dino_split_explicit_momentum_chain_round93_postfix.json`, SHA-256
`f075a3b7ecd63d226cbd1cb66d39c98a018118764632f50c2ad5c6bff109e06a`,
leaves only salinity `zfw` red: 212/9,920 columns and a `6.690652e-15`
maximum normalized error, with correlation 1.0 and RMS ratio within one ULP
of 1.0. Temperature's registered exact-W plus literal-A33 arm is bit-exact;
the K/post-factor arm is inert. The separate W-slope receipt exhausted the
raw-expression and Shapiro options and bounded the only upstream residue at
`1.55e-15`.

Under the requested Rule-1b standard, 78.S.3 is cleared as
`PROVEN_ORACLE_ARITHMETIC`, not `AT_BAR`, only if its maximum remains at most
`2e-14`, its correlation and RMS ratio remain within `2e-15` of unity, the
temperature exact-W/literal-A33 arm remains bit-exact, and every existing
plant/control passes. Any failed condition leaves 78.S.3 diverged. No bar is
relaxed and the receipt must retain the measured 212-column population.
