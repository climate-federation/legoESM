# DINO split-explicit / momentum-commit chain: rounds 9--10 result

Date: 2026-08-29. Session:
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

## Verdict

The frozen round-9 scorer was correct to print `OPEN_UNRESOLVED`. Its formal
flux disposition required `Q_depth_held == REFUTES` *and*
`F_flux_held == CONFIRMS`. F confirmed whole-domain and at both walls, but Q
was `UNRESOLVED`: prediction error >= 0.90 and explained RMS <= 0.10 met two
refutation clauses, while correlations 0.832738 whole-domain, 0.998268 at
`j=1`, and 0.850052 at `j=197` all exceeded the preregistered <= 0.20 clause.
That historical disposition is retained; no bar was relaxed after seeing the
result.

The production counterfactual **CONFIRMS the V-face zonal width
as the dominant owner of the pre-fix flux-composition residual**. The already
reviewed fix from `fidelity/dino-register-items` was carried as original
commits `c1cbef72ae4` and `6917c20714c` (local cherry-picks `4ff604100fe` and
`9445d8228c8`). On the same six NEMO dumps, CPU/fp64, it removes:

| Registered axis | Required | Measured reduction |
|---|---:|---:|
| 9.5 `zhV` normalized RMS error | >= 0.99 | 0.999999999993981 |
| 9.6 `zhdiv` normalized RMS error | >= 0.99 | 0.999999999958132 |
| `j=1` divergence-residual RMS | >= 0.99 | 0.999999999978738 |
| `j=197` divergence-residual RMS | >= 0.99 | 0.999999999959972 |

The Q-depth result is now stronger than the round-9 amplitude bound:
`zhvp2_e` becomes numerically exact on the registered wet mask and the
post-fix Q hold is an exact zero
prediction, hence `REFUTES` at both walls and whole-domain. The prior 3.207e-9
`zhvp2_e` error was not an independent depth formula defect; the face-depth
weighting itself consumed the same incorrect V metric.

## Named operand and production fix

The active NEMO DINO expressions are:

- `cfgs/DINO/MY_SRC/dynspg_ts.F90:699-701`:
  `zhU = e2u * ua_e * zhup2_e`;
- `cfgs/DINO/MY_SRC/dynspg_ts.F90:702-704`:
  `zhV = e1v * va_e * zhvp2_e`.

NEMO constructs the V latitude at the half-integer Mercator row in
`cfgs/DINO/MY_SRC/usrdef_hgr.F90:98,108`, then constructs `e1v` and `e2v`
from that same latitude at `:113,118`. legoESM forms the velocity-depth
products at
`packages/ocean/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py:851-852`
and divergence consumes `grid.dx_v` at
`packages/core/legoesm/grids/operators_latlon_cgrid.py:959-965`.

Before the carried fix, legoESM evaluated `dx_v` at the arithmetic mean of two
adjacent T latitudes. The Mercator transform is nonlinear, so this is not the
latitude of the half-index. The carried implementation uses the true V-face
latitude for the `nemo_isotropic`, variable-spacing contract at
`packages/core/legoesm/grids/latlon.py:1739-1806`. It is the same reviewed fix,
not a lane-local reimplementation. The follow-up scopes its existing
bit-identity claim to CPU; GPU can differ by about one ulp and was not used
here.

Merge-order dependency: if `fidelity/dino-register-items` lands first, drop
the equivalent local carry commits while retaining this lane's preregistration,
rescore, and result commits. Do not resolve the overlap by restoring the old
T-latitude-midpoint `dx_v` construction.

## Ordered post-fix rows

The exact class gates remain unchanged.

| Subrow | NEMO operand/site | Post-fix status | E | max error / NEMO RMS |
|---:|---|---|---:|---:|
| 9.1 | `zsshp2_e`, `dynspg_ts.F90:655-669` | AT BAR | 0 | 0 |
| 9.2 | `zhup2_e`, `:678-682` | AT BAR | 0 | 0 |
| 9.3 | `zhvp2_e`, `:683-687` | AT BAR | 0 | 0 |
| 9.4 | `zhU`, `:699-701` | **DEBT; ordered stop** | 9.21235e-17 | 2.07017e-15 |
| 9.5 | `zhV`, `:702-704` | DEBT | 9.36377e-17 | 2.67802e-15 |
| 9.6 | `zhdiv`, `:722-723` | DEBT | 1.89696e-11 | 3.09469e-10 |
| 9.7 | `ssha_e`, `:724` | AT BAR | 4.17903e-17 | 4.24828e-16 |

The next ordered discriminator is an offline, existing-dump multiplication
association arm at 9.4: reproduce NEMO's literal `e2u * ua_e * zhup2_e`
association and compare it with the production `H * U`, then metric multiply,
under the unchanged POINTWISE bar. The analogous 9.5 arm follows only after
9.4. Subrow 9.6 then needs its own divergence summation/association peel; its
remaining E is about 10.4 orders smaller than the pre-fix 0.453 but still above
the accumulating bar. No new NEMO instrumentation is indicated for these
offline arms.

Row 1.4 and registry rows 2--6 remain **ORDERED-BLOCKED** behind 9.4. Their
previous dump inventory and targeting receipts remain valid but are not
promoted into chain verdicts. Thus the row-1.3 metric-flux owner is fixed, while
the chain itself is not complete.

## Provenance and host-template correction

The fixed-tree rescore reused the round-9 ON/OFF 197/197 bracket, NEMO
executables, source hash, six operand dumps, round-8 artifact, and session ID.
It ran from clean commit `7b10cdcd08ea7031fede782b5bc104c0ed039454`
with JAX CPU/fp64. No NEMO run, GPU, `mpirun`, or push occurred.

Round 9's executed host corrections are now permanent in the handoff and
canonical SLOT source-build template: use `grep`, make every block `cd` into
its own repository context, use absolute repository paths for cross-directory
inputs, and explicitly export `CODEX_SESSION_ID`. The lean-donor copy guard
and cumulative 28-file deterministic-writer source manifest remain mandatory.

Targeted CPU tests passed 37/37 with this worktree's `packages/core` and
`packages/ocean` paths pinned. A preceding ambient-editable-install invocation
is invalid: it imported the old checkout, and its four failures reproduced the
pre-fix 3.3175e-5 V-width defect. No test result from that invocation is used.

Machine receipts:

- `dino_split_explicit_momentum_chain_round9_artifact.json` — the immutable
  host-produced frozen result;
- `dino_split_explicit_momentum_chain_round10_fixed_rescore.json` — unchanged
  round-9 scorer on fixed production geometry;
- `dino_split_explicit_momentum_chain_round10_artifact.json` — committed
  preregistered before/after adjudication.
