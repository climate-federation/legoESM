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
