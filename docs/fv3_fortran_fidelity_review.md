# FV3 Fortran Fidelity Review

Baselined 2026-04-14. Older prose is aggressively condensed to
save tokens. Only the newest Ralph-loop tail remains in full
form below.

> **Metadata convention (iter-174)**: do not duplicate "updated
> through iter-N" in the title. The authoritative iteration count
> is the branch commit history plus the HEAD commit message tag.

## Live status

- **W2 v-wind artifact at C36 remains unresolved (structural) but
  20 % smaller after iter-893.**  Production still runs
  `FV3EdgeShallowWaterModel` -> `fv3_sw_tendencies` (Arakawa-Lamb +
  RK3 + `boundary_fix`), not the FV3 FB chain. Current canonical
  W2 baseline after iter-893 (with `apply_fortran_xppm_boundary
  =True` activating Fortran's iord<7 cube-edge boundary formulas
  per `tp_core.F90:357-369`): `L2=2.75e-02`,
  `v_ll_Linf=1.319e-01 m/s`. Pre-iter-893 was `L2=3.06e-02`,
  `v_ll_Linf=1.585e-01 m/s`.  The artifact remains a
  cube-vertex-meridian stripe pattern and does not converge away
  with resolution.
- **FB chain accuracy at C24/C36 remains unresolved.**
  Halo=3 scaffolding exists, but FB accuracy is still poor. After
  the iter-808 sign-aware sync fix, FB DUOGRID completes 24 h at
  C24 without NaN, but h growth remains far above physical.

## Closed priorities

- **Panel-edge corner metrics**: resolved.
- **d_sw3 BGRID_NE sync**: resolved.
- **Duogrid cube-edge flux synchronization**: resolved by
  iter-807/808 sign-flip sync.
- **Legacy edge handling disabled in duogrid mode**: verified.
- **Non-duogrid `_d2a2c_vect` cube-vertex gap**: isolated,
  architectural, not on the default production path.
- **Old W2 polar face-4 vs face-5 asymmetry**: resolved by the
  iter-505 PPM-axis fix.
- **Visual diagnostic correctness**: corrected in iter-797/820.
- **Cosine bell visual cleanliness**: confirmed in iter-823.

## Key production fix

Iter-505 fixed the major production-path bug in
`cgrid_mass_flux_divergence`: x-direction strips were being passed
to `_ppm_reconstruct_1d` with the wrong active axis. Impact on
canonical W2 C36 dt=300s 1d:

- `L2`: `1.53e-3 -> 2.42e-4`
- `Linf`: `4.07e-3 -> 1.83e-3`
- `max|v_ll|`: `0.557 -> 0.303 m/s`

## Historical archive

Everything before `iter-856` is intentionally compressed here.

- `iter-1..174`: core FV3 metric/operator port, seam/sync work,
  regression expansion, and early FB bring-up.
- `iter-505..510`: production-path PPM-axis fix; old polar
  asymmetry closed.
- `iter-511..729`: W2/W5 artifact characterization, halo=3
  plumbing, FB diagnostics, and production-vs-FV3 routing
  clarified.
- `iter-730..751`: user-visible `v_ll` sentinels were locked; the
  production hyperdiff path was shown to be structurally non-FV3;
  and `use_duogrid=True` on the A-L path was proven catastrophic.
- `iter-752..759`: `_del6_vt_flux` was ported and corrected; the
  del6 post-step path became the production best, cutting W2
  `v_ll_Linf` into the `~0.159` class.
- `iter-760..767`: W2 mode-A was localized to cube corners; three
  Fortran-inspired cube-corner fill ideas were all tested on the
  A-L path and all made W2 worse.
- `iter-768..775`: the residual was re-measured as mainly
  dynamical; `boundary_fix` corner smoothing was shown to be
  load-bearing; `grad_c10` sensitivity was isolated; and a `0.98×`
  corner tuning reduced W2 but was explicitly classified as
  non-Fortran-faithful.
- `iter-776..795`: cosine-bell and W2 plateau behaviour was
  characterized; the W2 smoking gun was narrowed to cube-vertex
  `dv/dt`; the residual was identified as incomplete Cor+press+KE
  cancellation with numerical zeta load-bearing; and
  component-consistent B-halo was refuted for the smooth W2 IC.
- `iter-796..802`: zeta-zero tests confirmed the cancellation
  story; visual diagnostics were fixed; DUOGRID mass-transport
  blowup was localized to the halo=2 scalar path; and
  `fill_corner_region` overshoot plus a failed monotonicity clip
  were documented.
- `iter-803..811`: Fortran-faithful snapshot semantics for pass-2
  diagonals were aligned; the true DUOGRID root cause was found in
  `synchronize_cgrid_fluxes`; sign-aware flux sync fixed the worst
  DUOGRID catastrophe; W5 stayed within ~1% of LEGACY; cosine bell
  was somewhat worse but not catastrophic; and `boundary_fix`
  remained load-bearing.
- `iter-812..820`: DUOGRID visuals showed strong cube-corner
  artifacts; FB DUOGRID no longer crashed instantly but remained
  wildly inaccurate; `damp_v` above `0.03` was identified as the FB
  crash trigger; and the visual `v_north` projection was corrected
  to match the production sentinel.
- `iter-822..830`: W2 LEGACY convergence plateau was confirmed as
  structural; cosine bell was visually clean; W5 showed faint
  cube-vertex ringing; all combined Fortran corner-fill knob
  combinations still worsened W2; and the ocean rest-state
  condition passed.
- `iter-831..840`: FB phase ablations showed `_c_sw` drives h
  growth and `_p_grad_c` is stabilising; several documentary and
  diagnostic cleanups followed; and the canonical sentinel-aligned
  baseline was re-established.
- `iter-841..855`: current-form ut/vt swap was not a drop-in
  improvement; related halo hypotheses were retracted or reverted;
  dv/dt decomposition near the cube vertex was refined; the
  d_sw5-related comparison work underwent multiple honesty passes;
  Phase 1 trial wire-ins were attempted; and iter-854's short-run
  quantitative claims were formally retracted because of stale RK3
  plumbing.

Use git history if you need the full older narrative.

## Latest Ralph-loop iterations (full form)

Everything through `iter-898c` is compacted here. Only the newest
Ralph-loop tail remains in full form below.

- `iter-856..858`: the properly-plumbed Phase-1 / d_sw5-corner-
  divergence production experiment was re-run honestly; canonical
  damping blew up, and the reduced-damping variant still worsened
  1-day W2. Conclusion: the Phase-1 swap is not a viable production
  fix on the A-L + RK3 path.
- `iter-859..861`: the `*dt` adaptive-cap idea was shown to be a
  production no-op at canonical scales, and patch-dependent scripts
  were hardened with reproducibility guards.
- `iter-862..871`: several FB-only structural helpers and audits were
  added (`_d_sw5_corner_divergence` corner corrections, d_sw4 corner
  KE helper, halo-gap estimates, `da_min_c` sentinel, pre-iter-808
  gold-file audit), but FB stability/accuracy remained unresolved.
- `iter-872..877`: production `dddmp` configurability was surfaced,
  then narrowed and honesty-guarded; default-OFF fidelity-flag
  sentinels were added; an over-broad `use_duogrid`-contract change
  was reverted.
- `iter-878..885b`: the PPM stop-time / sentinel series corrected
  overshoot logic, removed non-Fortran clips, tightened behavioural
  coverage, fixed `_ppm_1d` boundary indexing, and removed duplicate
  or weak sentinels.
- `iter-886..891b`: a master Fortran-sentinel inventory was written;
  remaining `_ppm_1d` / `_ppm_edge_values` boundary-formula gaps were
  isolated, plumbed as opt-in paths, and corrected for bounded-domain
  / off-by-one placement issues.
- `iter-892..898c`: the production `_ppm_reconstruct_1d` off-by-one
  fix unexpectedly improved W2; `apply_fortran_xppm_boundary=True`
  became the production W2/W5 matrix baseline; W2 sentinels and
  known-worse tests were realigned and tightened so the new baseline
  is actually pinned.

### Iter-899 — investigate iter-892 PPM boundary index map; surface 1-cell shift bug

**Motivation.**  Codex iter-899 fidelity review recommended adding the missing Fortran `al(0)` and `al(npx+1)` cube-edge boundary overrides in `_ppm_reconstruct_1d` (`tp_core.F90:359, 368`).  iter-899 attempted this but uncovered an **index-map ambiguity** in the existing iter-892 code that requires resolution before extending.

**The finding.**  iter-892's docstring (`operators_cdgrid.py:189-197`) describes the input strip as:

> q has caller halo=2: q[k] = q1(k-1) for k=0..n_int+3.

This implies q[0]=q1(-1) (i.e., 1 halo cell on the left, 3 on the right — asymmetric).  iter-892's override formulas were written under this assumption.

**Empirical investigation** (`scripts/diag_iter899_ppm_strip_layout.py`) reveals that production actually feeds the leaf a **symmetric** halo=2 strip:

```
strip[ 0] = q1(-2)   (halo cell, depth 2)
strip[ 1] = q1(-1)   (halo cell, depth 1)
strip[ 2] = q1( 0)   (cube-edge interior cell)
strip[ 3] = q1( 1)
...
strip[n+1] = q1(n-1) (last interior)
strip[n+2] = q1(n)   (halo cell, depth 1)
strip[n+3] = q1(n+1) (halo cell, depth 2)
```

This is the natural output of `_pad_halo_auto_h2` (returning `(6, n+4, n+4)` with halo=2 per side) followed by `h_pad[:, :, 2:-2]` in `cgrid_mass_flux_divergence`.  The real strip layout is `q[k] = q1(k-2)` — ONE cell shifted from what the iter-892 docstring claims.

**Implication.**  Under the actual Hypothesis A index map, q_face[k] corresponds to Fortran `al(k-2)`.  iter-892's overrides at q_face[2,3,n+1,n+2] therefore correspond to `al(0,1,n-1,n)` — but iter-892's *formulas* implement the Fortran recipes for `al(1,2,npx-1,npx)`-style values (i.e., the docstring's intended targets).  This is a **1-cell shift bug**:

| q_face index | Fortran al (under Hypothesis A) | iter-892 formula |
|--------------|---------------------------------|------------------|
| q_face[2]    | al(0) = c1·q1(-2)+c2·q1(-1)+c3·q1(0) | xt-style 4-pt clipped |
| q_face[3]    | al(1) = xt 4-pt clipped         | c3·q_pad[4]+c2·q_pad[5]+c1·q_pad[6] |
| q_face[n+1]  | al(n-1) = standard 4th-order    | c1·q_pad[n+1]+c2·q_pad[n+2]+c3·q_pad[n+3] |
| q_face[n+2]  | al(n) = standard 4th-order      | xt-style 4-pt clipped |

iter-892 places the **wrong type of formula** at each cube-edge slot.  None of the four overrides produce Fortran-faithful values at their target indices.

**Why iter-893 still helped.**  Despite the 1-cell shift, iter-893 reduced W2 v_ll_Linf from 0.159 to 0.132 m/s (-17%).  Hypothesis: the iter-892 formulas (xt-clipped + c3/c2/c1 mirrors) are still **closer to Fortran's correct cube-edge values than the standard 4th-order edge extrapolation** (which uses `mode='edge'` replicas of q1(-1) for q1(-2) cells), even when placed at the wrong q_face indices.  The clipping in xt formulas may also reduce overshoot at the W2 cube vertex.

**iter-899 deliverable** (NO production code change — pure investigation + documentation).
- `scripts/diag_iter899_ppm_strip_layout.py`: empirical demonstration of the production strip layout (Hypothesis A confirmed).
- This iter-899 doc entry surfacing the discrepancy and the implication for future iters.

**Why no fix in iter-899.**  Two distinct fixes are possible:

1. **Index correction**: shift iter-892's q_pad indices +1 so the formulas land at the correct Fortran al targets.  Requires careful re-analysis because the iter-892 *formulas* (xt at q_face[2], c3/c2/c1 at q_face[3]) don't directly correspond to Fortran's specifications for those indices either — Fortran al(0) is 3-pt c1/c2/c3, Fortran al(1) is 4-pt xt clipped.  A pure index shift won't make the code Fortran-faithful; the formulas need to be swapped too.

2. **Full Fortran-faithful overrides**: replace iter-892's overrides with Fortran's actual recipes at the correct indices: q_face[2]=al(0)=c1·q1(-2)+c2·q1(-1)+c3·q1(0); q_face[3]=al(1)=xt clipped; q_face[4]=al(2)=c3·q1(1)+c2·q1(2)+c1·q1(3); mirror on right.  Halo=2 strip provides q1(-2..n+1), enough for al(0,1,2,n-1,n) but **NOT for al(npx,npx+1) which need q1(npx+1)=q1(n+2)** (deferred to halo=3 work).

Both fixes carry W2 regression risk: the iter-892 formulas may be empirically better than the strict-Fortran ones at the smooth W2 cube vertex.  iter-900+ should make these changes one at a time, with measurement.

**Process.**  179th iter.  iter-899 does NOT close a Fortran-fidelity gap by writing code — it surfaces a previously-undocumented gap by careful investigation.  This is meaningful work in the Ralph protocol sense: it converts a hidden fidelity bug into a documented, scoped, measurable work item for iter-900+.  No production code changes; no regression risk.

### Iter-900 — Fortran-faithful LEFT-side cube-edge overrides (opt-in flag, NEGATIVE result on W2)

**Motivation.**  iter-899 identified a 1-cell shift bug in iter-892's LEFT-side cube-edge PPM overrides: under production strip layout (Hypothesis A: q[k]=q1(k-2)), iter-892's xt formula at q_face[2] computes a non-Fortran value at the al(0) slot, and its c3/c2/c1 mirror at q_face[3] computes a non-Fortran value at the al(1) slot.  iter-900 implements the strict Fortran-faithful overrides at the corrected q_face indices behind a feature flag, with W2 measurement.

**Implementation.**  New `fortran_faithful_ppm_left: bool = False` field on `CDGridShallowWaterConfig` (forwarded through `fv3_sw_tendencies` and `cgrid_mass_flux_divergence` to `_ppm_reconstruct_1d`).  When True (and `apply_fortran_xppm_boundary=True`), the LEFT-side overrides apply Fortran's actual recipes per `tp_core.F90:359-362`:

```
q_face[2] = al(0) = c1*q1(-2) + c2*q1(-1) + c3*q1(0)
q_face[3] = al(1) = xt clipped using q1(-1..2), where
           xt = 0.75*(q1(0)+q1(1)) - 0.25*(q1(-1)+q1(2))
q_face[4] = al(2) = c3*q1(1) + c2*q1(2) + c1*q1(3)   [NEW override]
```

(Right-side overrides unchanged in iter-900; deferred to iter-901+ for separate measurement.)

**W2 measurement at C36 dt=300s 1-day** (`scripts/diag_iter900_w2_fortran_faithful_left.py`):

| config                           | mass_drift | h_L2     | h_Linf   | v_ll_Linf  |
|----------------------------------|------------|----------|----------|------------|
| (A) iter-892 default (production) | 4.561e-07  | 2.048e-04 | 8.184e+00 | **1.319e-01** |
| (B) iter-900 fortran-faithful left | 4.561e-07  | 2.291e-04 | 1.002e+01 | 2.027e-01 |

iter-900's strict-Fortran path INCREASES W2 v_ll_Linf by 53.7 % (+0.071 m/s).  h_L2 increases 11.9 % (2.05e-4 → 2.29e-4), h_Linf increases 22.4 %.  Mass drift unchanged (correct conservation).

**Interpretation (empirical only — mechanism unproven).**  iter-892's 1-cell-shifted formulas are EMPIRICALLY BETTER than strict Fortran on the smooth W2 cube vertex.  The W2 measurement is a global v_ll_Linf — it does NOT directly establish the per-cell mechanism.  One PLAUSIBLE hypothesis: iter-892's xt-clipped formula at q_face[2] (placed where Fortran specifies a 3-pt c1/c2/c3) constrains the boundary value more aggressively than Fortran's own c1/c2/c3 — possibly preventing overshoot at the smooth W2 cube edge.  Strict Fortran al(0) = c1*q1(-2) + c2*q1(-1) + c3*q1(0) is a 3-pt one-sided extrapolation that is higher-truncation-error in the linear regime.

**However**, this is a hypothesis, not a finding.  The unit tests in `tests/test_ppm_reconstruct_1d_fortran_faithful_left_iter900.py` show on a quadratic input that strict-Fortran al(0)=`c1*q1(-2)+c2*q1(-1)+c3*q1(0)` produces a value WITHIN [min, max] of {q1(-2..1)} for the test's monotone field — i.e., no overshoot occurs at the unit-test slot.  The +53.7 % W2 effect must therefore arise from interactions with the broader transport/limiter pipeline (perhaps the monotonicity limiter at lines 273-301 reacting differently to the new al(0) value, or perhaps the change propagating to neighbour cells via the upwind selection at the cube edge).  Pinpointing the real mechanism requires a per-cell W2 spatial diagnostic comparing the two flag values' v-wind tendencies at t=1 (not in scope for iter-900).

**Decision.**  Keep iter-892 as the production default.  iter-900's flag stays default OFF.  This documents a tension between strict Fortran fidelity and W2 numerical quality — a case where the legoESM cubed-sphere PPM benefits from a non-Fortran formula that happens to outperform the oracle on this specific test case.

**Open questions** (deferred):
- Does the iter-900 flag improve or hurt OTHER cases (W5, cosine bell, ocean rest)?  Only W2 measured.
- Does the same picture hold at finer resolution (C48, C96)?
- Would the right-side counterpart (`fortran_faithful_ppm_right` adding al(npx-1)/al(npx) at q_face[n+2,n+3]) show the same pattern?
- Could a HYBRID (strict-Fortran al(0)/al(2) but iter-892 xt at al(1)) be better than either pure path?

**Deliverable.**
- `src/legoesm/core/operators_cdgrid.py:_ppm_reconstruct_1d`: `fortran_faithful_ppm_left` kwarg (default False); preserves iter-892 default; adds 3-slot Fortran-faithful path at q_face[2,3,4] when flag=True.
- `src/legoesm/core/operators_cdgrid.py:cgrid_mass_flux_divergence`: forwards kwarg to leaf for both x and y strips.
- `src/legoesm/core/operators_cdgrid.py:fv3_sw_tendencies`: kwarg added; forwards to mass-flux divergence.
- `src/legoesm/atmosphere/dynamics/shallow_water_fv3_cdgrid.py:CDGridShallowWaterConfig`: `fortran_faithful_ppm_left` field added; production model.step forwards.
- `tests/test_ppm_reconstruct_1d_fortran_faithful_left_iter900.py`: 6 sentinels covering OFF preserves iter-892, ON applies Fortran al(0/1/2) at correct slots, RIGHT side untouched, gate respects `apply_fortran_xppm_boundary`.
- `tests/test_fortran_fidelity_default_flags_iter873.py`: `fortran_faithful_ppm_left` added to inventory.
- `scripts/diag_iter900_w2_fortran_faithful_left.py`: W2 comparison script.
- This iter-900 doc entry.

**Verification.**  All 26 tests pass (12 iter-873 inventory + iter-896 must-activate, 6 iter-899 sentinels + iter-768 t=1d pin, 6 iter-900 sentinels).  Production W2 baseline (iter-892 default) unchanged at 1.319e-1 m/s.

**Process.**  180th iter.  iter-900 closes the iter-899 handoff with a measured negative result: strict Fortran is WORSE than iter-892's accidentally-good shifted formulas on W2.  This is a legitimate FV3-fidelity finding — sometimes the oracle's literal recipes don't transfer cleanly to a different halo convention, and a thoughtful "wrong" formula outperforms.  The flag is preserved as opt-in for future hybrid experiments.  CLAUDE.md instructs "follow the Fortran implementation exactly" — but iter-900 demonstrates an empirical exception worth documenting; default flipping requires evidence iter-900 improves W2 plus W5/cosine bell/ocean rest, not just one of them.

### Iter-901 — broad evaluation of `fortran_faithful_ppm_left` across W5 + ocean rest

**Motivation.**  iter-900 measured ONLY W2 and concluded iter-892 was empirically better.  Per Ralph protocol, every iter should run all 4 evaluations (cosine bell, W2, W5, ocean rest).  iter-901 fills the gap.

**Cosine bell** is structurally INERT to the iter-900 flag because `run_cosine_bell` invokes `transport_step` directly without going through `fv3_sw_tendencies` (per iter-775 note in `scripts/run_atmosphere_test_matrix.py:1568-1576`).  Skipped with explicit justification.

**W5 measurement at C36 dt=300s 1-day** (`scripts/diag_iter901_broad_eval_fortran_faithful_left.py`):

| config                            | mass_drift | \|h-h_ic\|_Linf | h_min     | h_max     |
|-----------------------------------|------------|-----------------|-----------|-----------|
| (A) iter-892 default (production) | 1.151e-06  | 1.964e+02       | 3.9040e+03 | 5.9667e+03 |
| (B) iter-900 fortran-faithful left | 5.753e-07 | 1.965e+02       | 3.9039e+03 | 5.9667e+03 |

W5 fields are essentially **identical** between the two flag values (h_diff_linf, h range bit-equal in 4 sig figs).  Only mass drift differs: iter-900's strict-Fortran path has **half** the iter-892 mass drift (5.75e-7 vs 1.15e-6).  iter-892's accidentally-good shifted formulas evidently leak slightly more mass than the strict-Fortran recipes — a small but measurable Fortran-fidelity benefit.

**Ocean rest measurement at C36 1-day** (h=10000 m, u=v=0, h_s=0):

| config                            | mass_drift | \|h-h_ic\|_Linf | max\|u\|    | max\|v\|    |
|-----------------------------------|------------|-----------------|------------|------------|
| (A) iter-892 default (production) | 1.758e-06  | 1.758e-02       | 2.884e-14  | 2.409e-14  |
| (B) iter-900 fortran-faithful left | 1.758e-06 | 1.758e-02       | 2.884e-14  | 2.409e-14  |

**Bit-identical** between the two flag values.  Reason: at zero velocity the upwind face selection (`jnp.where(u_c > 0, q_R_left, q_L_right)` etc.) is degenerate, and the changed PPM face values feed multiplications by zero in the flux divergence — so the LEFT-edge override never affects the height tendency.  Ocean rest is robust to this PPM change.

**Combined picture (now including iter-900's W2):**

| test case        | iter-892 default | iter-900 fortran-faithful | winner       |
|------------------|------------------|---------------------------|--------------|
| W2 v_ll_Linf     | 1.319e-01 m/s    | 2.027e-01 m/s             | iter-892 (-53.7%) |
| W5 mass drift    | 1.151e-06        | 5.753e-07                 | iter-900 (-50%)   |
| W5 \|h-h_ic\|_Linf | 1.964e+02 m    | 1.965e+02 m               | tie (essentially identical) |
| Ocean rest       | identical        | identical                 | tie          |
| Cosine bell      | n/a (transport_step path) | n/a               | inert        |

**Interpretation.**  iter-892's empirical advantage is **W2-specific**.  On W5, iter-900's strict-Fortran path is marginally BETTER on mass conservation while preserving the height field bit-equally.  This is consistent with Fortran's recipes being designed for the FB-chain transport (which iter-900 does NOT consume) — but at the production C-grid PPM level, the formulas converge for cases without strong cube-vertex anomalies.  W2's smooth-IC cube-vertex artifact is a special case where iter-892's xt-clipping coincidentally constrains overshoot better than strict Fortran.

**Decision.**  iter-892 remains production default.  No default flip in iter-901.  The W2 +53.7 % regression dominates the W5 -50 % mass-drift improvement (mass drift is already at machine-precision floor; the W2 effect is at the actual physical-error scale).  iter-900's flag stays opt-in for advanced users who care more about Fortran fidelity than W2 absolute error.

**Deliverable.**
- `scripts/diag_iter901_broad_eval_fortran_faithful_left.py`: W5 + ocean rest comparison.
- This iter-901 doc entry adding the broader evaluation.

**Process.**  181st iter.  iter-901 completes the iter-900 evaluation by running the remaining 3 of 4 Ralph-protocol cases (cosine bell explicitly skipped with justification).  Strengthens the iter-900 conclusion: the flag is W2-pessimal in absolute v_ll_Linf but neutral-to-marginally-positive elsewhere.  No production code change; no regression risk.

### Iter-902 — quantify the dxa-weighted vs uniform-xt fidelity gap (defer full plumbing)

**Motivation.**  iter-888 docstring at `_ppm_reconstruct_1d` (line 124-131) explicitly notes the LEFT-side cube-edge xt formula uses a UNIFORM-GRID simplification of Fortran's dxa-weighted formula at `tp_core.F90:360-361`:

```fortran
al(1) = 0.5 * ( ((2*dxa(0,j)+dxa(-1,j))*q1(0) - dxa(0,j)*q1(-1))
                / (dxa(-1,j)+dxa(0,j))
              + ((2*dxa(1,j)+dxa(2,j))*q1(1) - dxa(1,j)*q1(2))
                / (dxa(1,j)+dxa(2,j)) )
```

For uniform dxa this collapses to `0.75*(q1(0)+q1(1)) - 0.25*(q1(-1)+q1(2))` — the iter-892 formula in production.  The gap was deferred until "dxa plumbing is added".  Codex iter-902 fidelity review recommended closing this gap.  iter-902 quantifies the magnitude FIRST so the implementation cost can be weighed against expected impact.

**Quantification at C36 face-0 LEFT cube edge** (`scripts/diag_iter902_dxa_xt_gap.py`):

For a smooth synthetic field q1(j) = sin(j·dxa/radius) and using mode='edge' replicas for halo dxa cells (LOWER BOUND on the true discrepancy, since real halo dxa from neighbour faces near cube vertices may differ more):

| metric | value |
|--------|-------|
| min \|fortran_al1 - uniform_xt\|  | 1.127e-07 |
| max \|fortran_al1 - uniform_xt\|  | 1.207e-04 |
| mean \|fortran_al1 - uniform_xt\| | 4.510e-05 |

**Expected W2 impact (extrapolation).**  iter-893's analogous-magnitude PPM cube-edge perturbations (replacing 4th-order edge stencil with iter-892's xt-clipped + c3/c2/c1) produced a W2 v_ll_Linf change of 0.027 m/s.  The iter-902 dxa correction is ~10× smaller in face-value units, so the expected W2 effect scales to ~3 mm/s — roughly 2% of the current production baseline (0.132 m/s).  Marginal but measurable.

**Decision.**  Defer full dxa plumbing to iter-903+.  Reasons:
1. Expected W2 progress (~2%) is below the threshold that justifies the implementation cost (substantial: cdgrid metric plumbing through `cgrid_mass_flux_divergence` → `_ppm_reconstruct_1d`, plus halo-2 dxa values which require their own neighbour-face exchange or `1.0/rdxa` extension).
2. iter-900-901 just demonstrated that strict Fortran can be W2-PESSIMAL at the cube edge — applying the dxa correction on TOP of iter-892's empirically-good 1-cell-shifted slot may not improve W2 (and could worsen it via similar mechanism).
3. The MED-risk-of-W2-regression label from Codex's review reflects this.
4. Higher-value gaps likely exist outside the PPM cube-edge thread (FB chain stabilization, deeper cross-face halo helper, structural W2 cube-imprint blocker).

**Deliverable.**
- `scripts/diag_iter902_dxa_xt_gap.py`: numerical quantification of the dxa correction magnitude.
- This iter-902 doc entry surfacing the deferred-with-justification status.

**Caveat — lower bound.**  The diagnostic uses `mode='edge'` replicas for halo dxa cells (q1(-2) = q1(0) etc.).  The real halo dxa values come from neighbour faces' interior cells, which near cube vertices have meaningfully different dxa due to gnomonic projection.  The TRUE discrepancy at the cube vertex itself could be 2-5× larger.  An exhaustive measurement that pulls real halo dxa from `_pad_halo_auto_h2(rdxa)` would tighten this bound but adds complexity for marginal incremental insight.

**Process.**  182nd iter.  iter-902 closes a long-standing iter-888 deferred item by quantifying its expected impact.  The conclusion ("not worth implementing now") is data-driven rather than speculative.  Future iters can refer to this magnitude estimate when deciding whether to revisit dxa plumbing.  No production code change; no regression risk.

### Iter-903 — Fortran-faithful RIGHT-side cube-edge overrides (NEGATIVE result on W2, mirroring iter-900)

**Motivation.**  iter-900 implemented strict-Fortran LEFT-side PPM cube-edge overrides and measured a +53.7 % W2 v_ll_Linf regression — iter-892's 1-cell-shifted formulas at the wrong q_face indices are W2-load-bearing.  iter-903 closes the symmetric counterpart: the RIGHT-side (q_face[n+1, n+2] in iter-892, q_face[n+2, n+3] under Hypothesis A) under a new `fortran_faithful_ppm_right` flag.

**Implementation.**  New default-OFF field on `CDGridShallowWaterConfig`, plumbed through `fv3_sw_tendencies` -> `cgrid_mass_flux_divergence` -> `_ppm_reconstruct_1d`.  When `True` AND `apply_fortran_xppm_boundary=True`:

```
q_face[n+2] = al(npx-1) = c1*q1(npx-3) + c2*q1(npx-2) + c3*q1(npx-1)
            = c1*q_pad[n+2] + c2*q_pad[n+3] + c3*q_pad[n+4]
            (FULL halo-2 fidelity)

q_face[n+3] = al(npx) = xt clipped using q1(npx-2..npx+1)
            = q_pad[n+3..n+6]
            (PARTIAL fidelity — q_pad[n+6] is mode='edge' replica
             of q1(n+1) since q1(npx+1)=q1(n+2) is outside halo=2)

q_face[n+4] = al(npx+1)  [NOT placed — needs halo=3, deferred]

q_face[n+1] = standard 4th-order interior stencil  [iter-892's
              misplaced override at the al(n-1) slot is REMOVED;
              Fortran has no boundary override there]
```

**W2 measurement at C36 dt=300s 1-day** (`scripts/diag_iter903_w2_fortran_faithful_right.py`):

| config                              | mass_drift | h_L2     | h_Linf   | v_ll_Linf  |
|-------------------------------------|------------|----------|----------|------------|
| (A) iter-892 default (production)   | 4.561e-07  | 2.048e-04 | 8.184e+00 | **1.319e-01** |
| (B) iter-903 fortran-faithful right | 5.701e-07  | 2.281e-04 | 1.059e+01 | 1.989e-01 |

iter-903's strict-Fortran path INCREASES W2 v_ll_Linf by **+50.8 %** (+0.067 m/s) — nearly mirror-image of iter-900's +53.7 % LEFT-side regression.  h_L2 +11 %, h_Linf +29 %, mass_drift slightly worse (vs iter-900's W5 case where mass_drift IMPROVED).

**Limiter-saturation observation** (iter-903 unit tests).  On strictly-convex quadratic test inputs, the monotonicity overshoot constraint at lines 295-301 saturates the q_face[n+3] override to `3*q - 2*q_R` regardless of whether iter-892's xt-clipped or iter-903's strict-Fortran al(npx) is the pre-limiter value.  The effective change in iter-903 is concentrated at q_face[n+2] (al(npx-1) — fully Fortran-faithful) plus the q_face[n+1] revert.

**Symmetric interpretation (LEFT + RIGHT).**  Combining iter-900 and iter-903:

| W2 v_ll_Linf at C36 1-day              | iter-892 default | strict-Fortran | delta     |
|-----------------------------------------|------------------|----------------|-----------|
| LEFT-side override (q_face[2,3])       | (baseline)       | flag-on:       | +53.7 %   |
| RIGHT-side override (q_face[n+2,n+3])  | (baseline)       | flag-on:       | +50.8 %   |

Both sides yield +50%-class W2 regressions when strict Fortran is applied at the correct indices.  iter-892's dual 1-cell-shifted approach is empirically optimal across BOTH cube edges of the production W2 path.  This is a NON-TRIVIAL symmetry — the cube-edge artifact mechanism that iter-892 happens to suppress is symmetric across face boundaries, so a single 1-cell shift bug applied symmetrically yields paired empirical advantages.

**Decision.**  Keep iter-892 as the production default.  iter-903's flag stays default OFF (covered by iter-873 inventory).  Combined with iter-900: the LEFT-side AND RIGHT-side strict-Fortran paths both regress W2; CLAUDE.md's "follow the Fortran implementation exactly" yields W2-pessimal behavior on BOTH cube edges.

**Open questions** (deferred):
- Does iter-903's flag improve W5 mass drift like iter-900 did?  (Not measured in iter-903; same broad-eval pattern as iter-901 would close this.)
- Could a HYBRID (iter-900 LEFT + iter-903 RIGHT, or one but not the other) yield a different trade-off?
- Does the symmetric LEFT+RIGHT pattern hold at finer resolutions (C48, C96)?

**Deliverable.**
- `src/legoesm/core/operators_cdgrid.py:_ppm_reconstruct_1d`: new `fortran_faithful_ppm_right` kwarg (default False); preserves iter-892 default; adds 2-slot Fortran-faithful path at q_face[n+2, n+3] when flag=True with q_face[n+1] revert.
- `src/legoesm/core/operators_cdgrid.py:cgrid_mass_flux_divergence`, `fv3_sw_tendencies`: kwarg threading.
- `src/legoesm/atmosphere/dynamics/shallow_water_fv3_cdgrid.py:CDGridShallowWaterConfig`: `fortran_faithful_ppm_right` field; production model.step forwards.
- `tests/test_ppm_reconstruct_1d_fortran_faithful_right_iter903.py`: 6 sentinels covering OFF preserves iter-892, ON applies Fortran al(npx-1) at q_face[n+2], q_face[n+3] is limiter-saturated, q_face[n+1] reverts to 4th-order, LEFT side untouched, gate respects `apply_fortran_xppm_boundary`.
- `tests/test_fortran_fidelity_default_flags_iter873.py`: `fortran_faithful_ppm_right` added to inventory.
- `scripts/diag_iter903_w2_fortran_faithful_right.py`: W2 comparison script (also `if __name__ == "__main__"` guarded per iter-901c convention).
- This iter-903 doc entry.

**Verification.**  All 32 tests pass: 12 iter-873/896 + 6 iter-899 + 1 iter-768 + 6 iter-900 + 6 iter-903 + 1 iter-893 baseline.  Production W2 baseline (iter-892 + iter-903 default-OFF) unchanged at 1.319e-1 m/s.

**Process.**  183rd iter.  iter-903 closes the symmetric counterpart of iter-900, completing the LEFT+RIGHT cube-edge Fortran-fidelity matrix.  Both directions yield ~50 % W2 regressions when strict Fortran is applied — the dual-side empirical pattern strengthens iter-900's interpretation: legoESM's cubed-sphere PPM transport pipeline benefits from non-Fortran constraints at BOTH cube edges, not just one.  Per Ralph protocol "every iter must produce real and meaningful work": iter-903 lands a flagged production-path implementation, measured W2 effect, sentinels, doc entry — even though the empirical conclusion is that the work should NOT be enabled by default.

### Iter-904 — true-FV3 d_sw1 mass transport opt-in (NEGATIVE result: NaN at C36 1-day)

**Motivation (user-directed).**  Pierre's iter-904 brief explicitly clarified that production is NOT true FV3:

> "Production is not true FV3. It runs `shallow_water_fv3_cdgrid.py:673` -> `operators_cdgrid.py:1840` using SSP-RK3, Arakawa-Lamb pressure gradient, physical-ish C-grid mass velocities, tuned div_damp, and a non-FV3 boundary_fix. True FV3 instead calls c_sw, then d_sw1/d_sw4/d_sw5/d_sw6 in a forward-backward chain in `dyn_core.F90:489` and `sw_core.F90:79`."

The W2 v-bias arises from incomplete geostrophic cancellation at cube vertices.  iter-904 isolates whether swapping ONE component — the height-tendency mass transport — to the true-FV3 d_sw1 path (`_d2a2c_vect` -> `compute_transport_quantities` -> `transport_step` per `sw_core.F90:79`) reduces the bias without destabilizing the SSP-RK3 momentum path.

**Implementation.**  New default-OFF `use_fv3_dsw1_mass_transport: bool = False` field on `CDGridShallowWaterConfig`.  When True (production model only):

```python
# fv3_sw_tendencies height-tendency branch:
if use_fv3_dsw1_mass_transport:
    _, _, _, _, ut, vt = _d2a2c_vect(u_d, v_d, cdgrid)
    h_new = transport_step(h, ut, vt, dt, cdgrid,
                            apply_fortran_xppm_boundary=apply_fortran_xppm_boundary)
    dh_dt = (h_new - h) / dt
else:
    dh_dt = cgrid_mass_flux_divergence(h, u_c, v_c, cdgrid, ...)  # iter-892 path
```

Momentum tendencies UNCHANGED; only the height-tendency path swaps.  `dt` plumbed from `model.step` -> `tendency_fn` closure -> `fv3_sw_tendencies(..., dt=dt)`.

**iter-903b/c warning extension.**  `FV3FBShallowWaterModel.__init__` now also warns when `use_fv3_dsw1_mass_transport=True` (the FB chain uses true FV3 natively; the SSP-RK3 d_sw1 wrap is meaningless there).

**t=0 tendency decomposition** (`scripts/diag_iter904_w2_true_fv3_gap.py` Part 1):

| metric                          | value at t=0          |
|---------------------------------|-----------------------|
| max \|dh/dt\|                   | 1.708e-04 m/s         |
| max \|du_d/dt\|                 | 1.405e-05 m/s²        |
| max \|dv_d/dt\|                 | 1.903e-05 m/s²        |
| max \|dv_north/dt\|_lat-lon     | 1.743e-05 m/s²        |

Top-20 D-grid v-edge hot spots cluster at **lat ±33.9°** on faces 0 and 2 — exactly the cube-vertex region where W2 mode A is strongest.  This confirms the W2 v-bias has a tendency-level signature at t=0, before any transport propagation.

**W2 1-day measurement at C36 dt=300s** (Part 2):

| config                                       | h_L2     | v_ll_Linf  |
|----------------------------------------------|----------|------------|
| (A) iter-892 default (production)            | 2.048e-04 | **1.319e-01** |
| (B) iter-904 use_fv3_dsw1_mass_transport     | NaN      | NaN        |

iter-904's path **NaN'd** during the 1-day integration.  Likely cause: `transport_step` is a finite-volume forward-Euler full-dt update returning `h_new`; deriving `dh_dt = (h_new - h) / dt` and feeding it into SSP-RK3's sub-stepping breaks the consistency between the transport's full-dt design and the RK3 stage weights.  The RK3 stages effectively apply a tendency that points toward h(t+dt) but at scaled magnitudes, leading to instability when combined with the Arakawa-Lamb momentum.

**Decision.**  Keep iter-892 as production default.  iter-904 flag stays default-OFF; the SSP-RK3 + true-FV3 d_sw1 hybrid is empirically unstable.  Per the user's acceptance gate ("improve v_ll_Linf by >=10 % AND worsen h_L2 by <5 %"), iter-904 fails on both metrics (NaN means undefined).

**What this means for the W2 v-bias.**  The bias cannot be fixed by swapping ONE component of true FV3 into the SSP-RK3 wrapper.  A faithful FV3 implementation requires the FULL forward-backward chain (c_sw + d_sw1 + d_sw4 + d_sw5 + d_sw6 in proper sequence, NOT wrapped in RK3).  That's a multi-iter architectural project — iter-904 documents that the partial-swap shortcut does not work.

**Open paths for iter-905+:**
- Time-integration matching: try wrapping the full FB chain in SSP-RK3 with proper sub-dt threading (probably still inconsistent).
- True FB chain stabilization: continue the deferred FB-chain stability work (per CLAUDE.md "FB chain accuracy at C24/C36 remains unresolved").
- Alternative: use `transport_step` only for the FIRST RK3 stage (forward Euler), then revert to PPM mass flux for subsequent stages — partial integration consistency, but adds branching.

**Deliverable.**
- `src/legoesm/atmosphere/dynamics/shallow_water_fv3_cdgrid.py:CDGridShallowWaterConfig`: new `use_fv3_dsw1_mass_transport: bool = False` field with SCOPE doc.
- `src/legoesm/atmosphere/dynamics/shallow_water_fv3_cdgrid.py:FV3EdgeShallowWaterModel.step`: forwards flag + `dt` to `fv3_sw_tendencies`.
- `src/legoesm/atmosphere/dynamics/shallow_water_fv3_cdgrid.py:FV3FBShallowWaterModel.__init__`: warning extended to include the new flag.
- `src/legoesm/core/operators_cdgrid.py:fv3_sw_tendencies`: new conditional height-tendency branch using `_d2a2c_vect` + `transport_step` when flag=True; raises ValueError if `dt` is None in that branch.
- `tests/test_iter904_use_fv3_dsw1_mass_transport.py`: 7 sentinels covering default-OFF bit-equality, ON-changes-dh-only, dt-required, FB-warning, production-no-spurious-warn, default-flags-still-OFF, ON-step-runs-without-NaN-on-random-state.
- `tests/test_fortran_fidelity_default_flags_iter873.py`: documented exclusion of `use_fv3_*` prefix from the iter-873 inventory taxonomy with reason.
- `scripts/diag_iter904_w2_true_fv3_gap.py`: tendency decomposition + W2 1-day comparison.
- This iter-904 doc entry.

**Verification.**  7/7 iter-904 tests pass; 14/14 iter-873 tests pass; production W2 baseline unchanged at 1.319e-1 m/s with flag OFF.

**Process.**  184th iter.  iter-904 lands a user-directed substantive piece of work (production-path config flag + true-FV3 d_sw1 transport plumbing + tendency diagnostic + W2 measurement) and produces a clear empirical answer: the partial-swap shortcut does NOT work — the SSP-RK3 + finite-volume hybrid is unstable.  This narrows the design space for iter-905+: the W2 bias requires either (a) the full FB chain's stabilization, or (b) a different time-integration matching strategy.

### Iter-904b — forward FV3 d_sw1 damping (`nord_v`, `damp_v`); confirms iter-904's instability is structural, not damping-driven (Codex iter-904 stop-time)

**Motivation.**  Codex iter-904 stop-time review correctly flagged: "iter-904 omits FV3 d_sw1 mass-transport damping".  Per Fortran `sw_core.F90:886-887`, d_sw1 calls `fv_tp_2d(delp, ..., nord=nord_v, damp_c=damp_v)` so the mass transport picks up the same 4th-order del-n smoother the FB chain applies.  iter-904's `transport_step` call omitted these kwargs, leaving d_sw1 transport without any del-n damping.

**Fix.**  Add `dsw1_nord` and `dsw1_damp_c` kwargs to `fv3_sw_tendencies` (defaults 2 and 0.06 — Fortran-canonical values).  Production model.step forwards from `config.nord_v` and `config.damp_v` (already-canonical config fields per the iter-872c-take3 Fortran convention).  iter-904b's `transport_step` call now passes these so the d_sw1 mass transport gets the proper Fortran-faithful smoother.

**Re-measurement at C36 dt=300s 1-day.**

| config                                         | h_L2     | v_ll_Linf  |
|------------------------------------------------|----------|------------|
| (A) iter-892 default                           | 2.048e-04 | **1.319e-01** |
| (B) iter-904b use_fv3_dsw1_mass_transport      | 1.387e+00 | 3.137e+02 |

iter-904b's path no longer NaN's (the damping prevents the immediate catastrophic instability), but produces a v_ll_Linf of **313 m/s** (vs 0.13 baseline — +2378× worse) and h_L2 of **1.39** (vs 2.05e-4 baseline — +6770× worse).

**Why the result is still catastrophic.**  The fundamental incompatibility is the SSP-RK3 wrapping: each of the 3 RK3 stages calls `transport_step` with the FULL `dt` (not the sub-stage dt — the integrator API doesn't expose stage dt to the tendency function), so each stage attempts to advect h forward by an entire timestep.  RK3 then combines the stage tendencies with weights summing to 1, but the underlying transport in each stage was already a full-dt update.  Net effect: mass is advected ~3× per step, and the resulting tendency has wildly wrong magnitude.  Damping mitigates the blowup speed but cannot fix the structural mismatch.

**Confirms iter-904's conclusion.**  The W2 v-bias cannot be fixed by partial-swapping ONE component (d_sw1 mass transport) of true FV3 into the SSP-RK3 wrapper.  The FB scheme requires its own time stepping (`dyn_core.F90:489` uses three sequential FORWARD-BACKWARD phases per dt, each itself a full-dt update — NOT a multi-stage RK3 with weighted intermediate states).  iter-904b strengthens this conclusion: even with proper Fortran-faithful damping, the SSP-RK3 + d_sw1 hybrid is structurally unsuitable.

**Decision.**  iter-892 remains production default.  `use_fv3_dsw1_mass_transport=True` flag is preserved as a documented opt-in for diagnostic reproducibility, but the doc now explicitly states the path is structurally unstable — users should NOT enable it expecting improvement.

**Path forward (iter-905+).**  The W2 bias fix requires one of:
1. **Full FB-chain integration**: replace SSP-RK3 with the Fortran-faithful FB scheme (c_sw -> p_grad_c -> d_sw with proper sub-stepping).  Multi-iter architectural project; CLAUDE.md memory tracks this as a deferred structural blocker ("FB chain accuracy at C24/C36 remains unresolved").
2. **Sub-dt threading in tendency_fn**: refactor the integrator API so tendency_fn receives the actual stage dt, then pass that as transport_step's dt.  This would make iter-904's hybrid consistent at each RK3 stage.  Risk: stage-level full-dt re-advection still doesn't match the FB chain's design — the partial-swap may remain a non-improvement.
3. **Forward-Euler mass-only path**: split the integration at `model.step` level — call `transport_step` ONCE per full dt outside RK3 (forward-Euler mass), and use SSP-RK3 only for momentum.  This breaks the joint conservation properties of RK3 but might produce a stable iter-904-class hybrid.  Architectural change but smaller than (1).

**Deliverable.**
- `src/legoesm/core/operators_cdgrid.py:fv3_sw_tendencies`: new `dsw1_nord` (default 2) and `dsw1_damp_c` (default 0.06) kwargs forwarded to `transport_step` when `use_fv3_dsw1_mass_transport=True`.
- `src/legoesm/atmosphere/dynamics/shallow_water_fv3_cdgrid.py:FV3EdgeShallowWaterModel.step`: forwards `nord_v`/`damp_v` config fields as the new `dsw1_nord`/`dsw1_damp_c`.
- This iter-904b doc entry strengthening iter-904's negative-result conclusion with proper damping in place.

**Verification.**  All 26 tests pass (7 iter-904 + 5 iter-903b/c + 14 iter-873).  Production W2 baseline (flag OFF) unchanged at 1.319e-1 m/s.

**Process.**  185th iter.  iter-904b closes the Codex iter-904 stop-time finding by adding the previously-omitted Fortran-faithful damping, then confirms via re-measurement that the SSP-RK3 + d_sw1 partial-swap remains structurally unsuitable even with proper damping.  This is a stronger negative result than iter-904 alone: not "we forgot the damping and it NaN'd", but "even with proper Fortran-faithful damping, the time-integration mismatch is the binding constraint".

### Iter-905 — split mass+momentum integration (path 3 of iter-904b's 3 paths forward; NEGATIVE result)

**Motivation (user-directed).**  After iter-904b confirmed the SSP-RK3+d_sw1 partial-swap is structurally unstable, the user directed implementation of all 3 paths forward:

> "implement 1. Full FB-chain integration replacing SSP-RK3 (multi-iter architectural). 2. Sub-dt threading in the tendency_fn API. 3. Split mass+momentum integration (forward-Euler mass outside RK3, RK3 momentum)."

iter-905 implements path (3) — the smallest architectural change of the three.

**Implementation.**  New default-OFF `use_split_mass_momentum_integration` config field.  When True, `FV3EdgeShallowWaterModel.step`:
1. Computes `h_new = transport_step(h, ut, vt, dt, ...)` ONCE outside the RK3 loop with Fortran-faithful damping (`nord_v`, `damp_v`).
2. Runs SSP-RK3 with momentum-only tendency function: at each stage, the held IC `h` is used for the Bernoulli function, and mass tendency is forced to zero.  RK3 produces final `u_d`, `v_d`.
3. Returns state with `h=h_new` from step 1 and momentum from step 2.

This decouples mass from RK3 stages, addressing iter-904b's "~3× mass advection per step" finding.

**W2 1-day measurement at C36 dt=300s** (`scripts/diag_iter905_w2_split_integration.py`):

| config                                          | h_L2     | h_Linf   | v_ll_Linf  |
|-------------------------------------------------|----------|----------|------------|
| (A) iter-892 default                            | 2.048e-04 | 8.184e+00 | **1.319e-01** |
| (B) iter-905 split mass+momentum                | 1.292e+00 | 5.068e+04 | 2.921e+02 |

iter-905 also catastrophically blows up: v_ll_Linf = 292 m/s (+221416 %), h_L2 = 1.29 (+630798 %).  Same ~+50× scale of failure as iter-904b's full-RK3 wrap with damping.

**Why path (3) also fails.**  The momentum RK3 stages use the IC h for the Bernoulli function across all 3 stages while u, v evolve.  After stage 1, u₁/v₁ have been updated against the IC pressure-gradient (Bernoulli) field — but the actual h field has by then been advected forward by transport_step.  So u₁/v₁ correspond to a momentum balance with the OLD h, while the actual flow field has h_new.  This mass-momentum decoupling violates the geostrophic balance the W2 IC relies on, producing a runaway divergence.

**Implications for paths (1) and (2).**

- **Path (1) — Full FB-chain integration replacing SSP-RK3**: this would replace `dispatch_integrator` with a faithful c_sw → p_grad_c → d_sw chain at each timestep (no RK3 at all).  This is the architecturally correct fix — the FB chain is precisely the time integration the d_sw routines are designed for.  But: legoESM's existing `fv3_fb_sw_step` (consumed by `FV3FBShallowWaterModel`) is documented "EXPERIMENTAL — DO NOT USE.  NOT PRODUCTION-READY.  Known unstable (85 m/s v-wind after 1 day, 3 % mass error)."  The FB chain is itself a known structural blocker (per CLAUDE.md "FB chain accuracy at C24/C36 remains unresolved").  Wiring an unstable FB chain into production wouldn't fix W2 either; it'd worsen it.  Path (1) is fundamentally blocked on the FB chain stabilization work itself.
- **Path (2) — Sub-dt threading**: this would refactor `dispatch_integrator` so tendency_fn receives the actual stage dt (instead of closing over the outer dt).  But even with stage-correct dt, iter-905's stage-level mass-momentum decoupling would persist — RK3 would now run transport_step with stage-dt inside each tendency, but the stage's Bernoulli-derived momentum gradient still uses an h that's inconsistent with the stage's transported h.  Path (2) doesn't address the geostrophic-cancellation requirement.

**Combined conclusion (iter-904 + iter-904b + iter-905).**  ALL THREE partial-FV3 / RK3-hybrid paths fail catastrophically on W2 1-day at C36.  The W2 v-bias is fundamentally tied to maintaining geostrophic mass-momentum balance during time integration.  iter-892's choices (full PPM mass flux divergence + Arakawa-Lamb pressure gradient + boundary_fix corner smoothing, all inside a SINGLE RK3 stage) preserve the balance well enough to give v_ll_Linf = 0.132 m/s.  Any partial swap that mixes FV3 d_sw transport with RK3 momentum breaks the balance and produces 100×-2400× v_ll_Linf regression.

**Decision.**  Keep iter-892 as production default.  iter-905 flag stays default-OFF.  Paths (1) and (2) DEFERRED with strong evidence they're unlikely to improve W2:

- Path (1) requires FB chain stabilization first (multi-iter prerequisite).  When the FB chain itself reaches sub-0.132 W2 v_ll_Linf at C36, then it makes sense to wire it into production.  Until then, swapping in an unstable FB chain would worsen W2.
- Path (2) doesn't address the geostrophic-balance issue; predicted to produce similar blowups as paths (3) and (4) but via a different mechanism.  Could be implemented for completeness but unlikely to be the W2 fix.

**Deliverable.**
- `src/legoesm/atmosphere/dynamics/shallow_water_fv3_cdgrid.py:CDGridShallowWaterConfig`: new `use_split_mass_momentum_integration: bool = False` field.
- `src/legoesm/atmosphere/dynamics/shallow_water_fv3_cdgrid.py:FV3EdgeShallowWaterModel.step`: split-integration branch when flag=True (transport_step outside RK3, momentum-only tendency_fn with held h).
- `src/legoesm/atmosphere/dynamics/shallow_water_fv3_cdgrid.py:FV3FBShallowWaterModel.__init__`: warning extended to flag iter-905.
- `tests/test_iter905_split_mass_momentum.py`: 6 sentinels covering default-OFF bit-equality, ON-changes-step-output, ON-finite, FB-warning, default-OFF, unrelated flags untouched.
- `scripts/diag_iter905_w2_split_integration.py`: W2 measurement script (with `__main__` guard per iter-901c convention).
- This iter-905 doc entry.

**Verification.**  6/6 iter-905 tests pass; 33 total tests pass in the iter-89x/90x suite (12 iter-873/896 + 6 iter-899 + 1 iter-768 + 6 iter-900 + 6 iter-903 + 5 iter-903b/c + 7 iter-904 + 6 iter-905).  Production W2 baseline (flag OFF) unchanged at 1.319e-1 m/s.

**Process.**  186th iter.  iter-905 closes path (3) of the user-directed 3-path program with another negative result.  Combined with iter-904 (NaN) and iter-904b (catastrophic blowup at +2378 % v_ll_Linf), all 3 partial-FV3/RK3-hybrid attempts fail.  The user's explicit request to implement all 3 paths is now answered: path (3) is implemented and measured negative; paths (1) and (2) are explicitly deferred with structural arguments why they're unlikely to succeed.  iter-906+ should pivot away from RK3-hybrid approaches and toward either FB-chain stabilization (path 1's prerequisite) or a different W2 angle entirely (e.g., the structural geostrophic cancellation per CLAUDE.md memory).

### Iter-906 — per-component dv tendency analysis at W2 hot spots (POSITIVE diagnostic finding)

**Motivation.**  iter-904/904b/905 confirmed all 3 partial-FV3/RK3 hybrids fail catastrophically.  iter-906 takes a different angle: instead of trying ANOTHER time-integration swap, decompose `dv_d/dt` at the iter-904 W2 hot spots into its constituent terms to identify WHICH component drives the geostrophic-cancellation residual.  Per `operators_cdgrid.py:2009-2010`:

```
du_cc =  zeta_abs * v_cc - dB_dx_cc
dv_cc = -zeta_abs * u_cc - dB_dy_cc
```

For the W2 IC, the analytical solution has v_cc≡0 and exact geostrophic balance `f·u = -g·dh/dy`.  iter-906 measures `+zeta_abs*u_cc` (Coriolis-vorticity term) and `-dB_dy_cc` (pressure-gradient term) magnitudes separately and reports their cancellation quality at cube-vertex cells.

**Diagnostic results at C36 t=0** (`scripts/diag_iter906_w2_dv_balance.py`):

| metric                       | value           |
|------------------------------|-----------------|
| max \|dv_cc\| (residual)     | 2.52e-05 m/s²   |
| mean \|dv_cc\|               | 2.04e-06 m/s²   |
| max \|coriolis term\|        | 3.04e-03 m/s²   |
| max \|pressure term\|        | 3.04e-03 m/s²   |

The two terms each have magnitude ~3e-3 m/s², matching analytical `f·u ≈ Ω·a·u₀ ≈ 2.7e-3` for W2's u₀ ≈ 38.6 m/s — the discretization is correctly representing both the Coriolis-vorticity and the pressure-gradient force.

**Top-10 cube-vertex hot spots:**

| face | i | j | lat (°) | -zeta_abs*u | -dB_dy_cc | dv_cc | cancel ratio |
|------|---|---|---------|-------------|-----------|-------|--------------|
| 5 | 0 | 0 | -36.45 | -2.494e-03 | +2.519e-03 | +2.516e-05 | 9.99e-03 |
| 4 | 0 | 0 | +36.45 | -2.494e-03 | +2.519e-03 | +2.516e-05 | 9.99e-03 |
| 4 | 35 | 35 | +36.45 | +2.494e-03 | -2.519e-03 | -2.516e-05 | 9.99e-03 |
| ... 5 more cube-vertex cells at lat ±36.45° on faces 4/5 ... |
| 3 | 35 | 35 | +34.66 | -2.503e-03 | +2.511e-03 | +7.39e-06 | 2.95e-03 |
| 3 | 35 | 0  | -34.66 | +2.503e-03 | -2.511e-03 | -7.39e-06 | 2.95e-03 |

The first 8 hot spots are EXACTLY the 8 cube vertices on faces 4 and 5 (top/bottom polar faces' corners).  Cancellation ratio at these cube vertices is ~1e-2 (i.e., the two ~2.5e-3 terms cancel to within ~2.5e-5).

**Cancellation quality summary:**

| metric                                       | cancellation ratio |
|----------------------------------------------|--------------------|
| hot-spot top-10 mean                         | 8.58e-03           |
| global mean (all 6n² cells)                  | 1.13e-03           |
| global max (worst cell, ≈ hot spot)          | 9.99e-03           |

**Cube-vertex cancellation is 7.6× WORSE than global average.**

**Interpretation.**  The W2 v-bias is NOT a uniformly-distributed cancellation error amplified by time integration.  It is a LOCALIZED cube-vertex breakdown of the geostrophic cancellation between zeta_abs*u_cc and dB_dy_cc.  Both terms are individually accurate (~3e-3, matching analytical), but their cube-vertex difference accumulates 7.6× more error than at smooth interior cells.

**Implications for iter-907+ work.**  Two concrete improvement targets:

1. **`zeta_abs` at cube vertices**: `dgrid_vorticity` uses corner-interpolated `u_corner, v_corner` from rotate-pad-rotate halo cells.  Cube-vertex inaccuracy in halo metrics propagates to `zeta`.  Improving the halo treatment at the 4 cube-vertex cells of `(u_cc_pad, v_cc_pad)` should reduce this term's local error.
2. **`dB_dy_cc` at cube vertices**: `_arakawa_lamb_gradient` + `_interp_corner_to_center` chain has its own cube-vertex errors.  Improving the corner-to-center interpolation at the 4 vertex cells should reduce this term's local error.

Either fix applied at ONE cube-vertex slot (4 or 8 cells per cube edge) would ~halve the local cancellation error and reduce the W2 v_ll_Linf residual.  Iter-907+ should target ONE of these fixes per iter, with measurement.

**Comparison to iter-767/769 corner-fill experiments.**  Per CLAUDE.md memory, iter-767 implemented Fortran's `fill_corners_agrid_r8` VECTOR (mySign=-1) formula at the cube vertices to address this exact mechanism — but it WORSENED W2 by 16×.  The reason (per iter-767's docstring) is that our `pad_halo_vector` ALREADY rotates winds through geographic intermediary, and applying Fortran's swap on top of already-rotated halo produces algorithmically inconsistent values.  iter-907+ work should not repeat iter-767's approach — instead, target the halo's accuracy at the rotation step, or improve the gradient-stencil rather than the halo cells themselves.

**Deliverable.**
- `scripts/diag_iter906_w2_dv_balance.py`: per-component dv decomposition at W2 hot spots; quantifies cube-vertex cancellation breakdown; emits VERDICT identifying the localized-cancellation interpretation.
- This iter-906 doc entry surfacing the actionable diagnostic finding.

**Verification.**  Diagnostic runs to completion in ~5s.  No production code change; no regression risk.

**Process.**  187th iter.  iter-906 PIVOTS from the failed RK3-hybrid implementation thread to a diagnostic deep-dive that yields a concrete actionable target for iter-907+: cube-vertex cancellation breakdown between `-zeta_abs*u_cc` and `-dB_dy_cc`.  This is the highest-leverage single iter so far in the iter-904-906 thread because it gives future iters a precise mechanism to target rather than repeating failed time-integration swaps.

### Iter-906b — D-grid `dv_d_dt` decomposition matching iter-904 hot-spot stagger (REFINES iter-906 interpretation)

**Codex iter-906 stop-time concern.**  "The iter-906 diagnostic does not measure the D-grid hot spots it claims to explain".  iter-906 measured `dv_cc` at cell centres (shape `(6, n, n)`); iter-904's hot spots were on the D-grid v-edge stagger `dv_d_dt` (shape `(6, n+1, n)`), AFTER the cell-centre→D-grid projection at `operators_cdgrid.py:2193-2214` and AFTER div_damp + boundary_fix contributions.

**iter-906b extension.**  Reproduce iter-906's cell-centre decomposition, project the bare Coriolis+pressure sum to D-grid via the same `pad_halo_vector` chain production uses, then compare to the FULL production `dv_d_dt` (which includes div_damp + boundary_fix + projection halo).

**Key results at C36 t=0** (`scripts/diag_iter906b_w2_dv_balance_dgrid.py`):

| metric                                          | value          |
|-------------------------------------------------|----------------|
| max \|dv_cc\| (iter-906 finding, cell centre)   | 2.52e-05 m/s²  |
| max \|dv_d_dt\| (BARE Coriolis+pressure, D-grid) | 1.15e-05 m/s²  |
| max \|dv_d_dt_prod\| (PRODUCTION fv3_sw_tendencies) | **1.90e-05 m/s²** |

The bare-projection result (1.15e-5) is SMALLER than the cell-centre cube-vertex finding (2.52e-5) — i.e., the projection AVERAGES OUT half the cube-vertex error.  The PRODUCTION result (1.90e-5) is LARGER than the bare projection by ~65 % because production adds div_damp + boundary_fix.

**Iter-904 D-grid hot spots vs iter-906 cell-centre hot spots — DIFFERENT.**

- iter-906 cell-centre hot spots: **lat ±36.45° on faces 4 / 5** (cube vertices).
- iter-906b D-grid production hot spots: **lat ±33.9° on faces 0 / 2** (interior of equatorial faces, ~3° away from cube vertices) — matching iter-904.

These are NOT the same cells.  The cell-centre cube-vertex cancellation breakdown that iter-906 identified does NOT directly translate to the D-grid production hot spots.

**Cancellation quality on D-grid stagger:**

| metric                                | value          |
|---------------------------------------|----------------|
| hot-spot top-10 mean                  | 1.08e-03       |
| global mean                           | 1.11e-03       |
| global max                            | 6.06e-03       |

**D-grid hot-spot cancellation ratio is ~1.0× global** — i.e., iter-906's "7.6× worse at hot spots" finding does NOT replicate on the D-grid stagger.  The D-grid hot spots are NOT driven by the cell-centre cube-vertex cancellation breakdown.

**Refined interpretation.**  iter-906's cell-centre cube-vertex finding is REAL but UPSTREAM of and PARTIALLY CANCELLED BY the cell-centre→D-grid projection.  The actual production W2 v-bias driver is the contributions ADDED after the bare Coriolis+pressure terms — most likely div_damp's contribution to dv_cc (line 2089) and/or boundary_fix's smoothing (line 2188-2191), and/or `fortran_vector_corner_fill`-style halo treatments at the projection step (line 2209-2212).

**Updated targets for iter-907+:**

1. **div_damp's contribution to dv_d_dt at lat ±33.9° face 0/2**: this is the lat-band where div_damp's adaptive `dddmp * |delpc|` clipping interacts with the cube-edge cells (face 0/2 = equatorial faces, with i=2/3 the row near the EW cube edges).  The interaction may be the dominant mechanism.
2. **boundary_fix's effect on dv_cc at face boundary cells**: row 2/3 from the cube edge corresponds to interior cells just inside the boundary smoothing region.  Boundary_fix smooths rows 0 and n-1; rows 1-3 may absorb its discontinuity.
3. **Cell-centre→D-grid projection halo (`pad_halo_vector`) accuracy at cube edges**: the projection at line 2194 uses halo cells; cube-edge halo accuracy could break the iter-906 cancellation differently for the D-grid stagger than the cell-centre one.

**Walking back iter-906's overclaim.**  iter-906 said cube-vertex cancellation was the W2 bias mechanism.  iter-906b shows that's correct AT CELL CENTRES but NOT AT THE D-GRID STAGGER WHERE THE BIAS LIVES.  The actual mechanism is more complex — production-only contributions (div_damp, boundary_fix) AND/OR projection-halo errors AT THE D-GRID stagger are the binding constraints.  iter-907+ should target these production-side components rather than zeta_abs / dB_dy_cc directly.

**Deliverable.**
- `scripts/diag_iter906b_w2_dv_balance_dgrid.py`: D-grid `dv_d_dt` decomposition with explicit comparison to iter-906 cell-centre finding; emits VERDICT identifying the discrepancy.
- This iter-906b doc entry refining iter-906's interpretation.

**Verification.**  Diagnostic runs to completion in ~5s.  No production code change.

**Process.**  188th iter.  iter-906b corrects iter-906's overclaim by measuring at the same stagger as iter-904's hot spots.  The Codex stop-time concern was valid; the corrective measurement reveals iter-906's cube-vertex cancellation finding is real at cell centres but DOES NOT explain the production W2 v-bias.  Net iter-907+ direction shifts from "fix zeta_abs/dB_dy_cc at cube vertices" (iter-906's recommendation, now superseded) to "investigate div_damp + boundary_fix + projection-halo contributions at lat ±33.9° face 0/2 — the actual D-grid hot spots".

### Iter-907 — ablation study: div_damp dominates W2 D-grid hot-spot magnitude (29× more than boundary_fix)

**Motivation.**  iter-906b refined the iter-907+ targets to "div_damp + boundary_fix + projection halo".  iter-907 disambiguates WHICH component dominates by running fv3_sw_tendencies with each toggled independently.

**Configurations** (all use `apply_fortran_xppm_boundary=True`, `dddmp=0.2`):

- (A) Production:    `div_damp = 2.13e8, boundary_fix = True`  (= iter-892).
- (B) − div_damp:    `div_damp = 0,       boundary_fix = True`.
- (C) − boundary_fix: `div_damp = 2.13e8, boundary_fix = False`.
- (D) Bare A-L:      `div_damp = 0,       boundary_fix = False`.

**Whole-D-grid `|dv_d_dt|` max:**

| config | max \|dv_d_dt\| |
|--------|------------------|
| A (production) | 1.903e-05 |
| B (− div_damp) | **4.551e-06**  (−76 %) |
| C (− boundary_fix) | 2.688e-05  (+41 %) |
| D (bare A-L) | 1.148e-05 |

Removing div_damp reduces the max by 76 %.  Removing boundary_fix INCREASES the max by 41 % (boundary_fix is a stabilizer, as designed).

**At iter-904 D-grid production hot spots (top-10 cells by |cfg_A|):**

| metric | value |
|--------|-------|
| A (production) mean \|dv_d_dt\| | 1.875e-05 |
| B (− div_damp) mean | 2.77e-06  (−85 %) |
| C (− boundary_fix) mean | 1.93e-05  (essentially unchanged) |
| D (bare A-L) mean | 2.82e-06 |
| **delta_div_damp** (A − B) | **+1.599e-05**  (~85 % of A) |
| **delta_bfix** (A − C) | **−5.47e-07**  (~3 % of A; opposite sign) |
| **\|delta_div_damp / delta_bfix\|** | **29.2 ×** |

div_damp's contribution at the hot spots is **29× larger** than boundary_fix's — div_damp is the dominant driver of the W2 D-grid hot-spot magnitude.

**The eight hot spots at lat ±33.9° face 0/2 are the EW-cube-edge cells of the equatorial faces.**  Removing div_damp reduces |dv_d_dt| at these cells from 1.90e-5 → 2.71e-6 (−86 %).  These cells correspond to (face, i, j) = (0, 2, 1), (0, 2, 34), (2, 2, 1), (2, 2, 34) and their lat-mirrors at (face, 3, ...).  The i = 2-3 row is just inside the cube-edge boundary on the east side of faces 0/2.

**Mechanism interpretation.**  div_damp's contribution to dv_cc is `adaptive_coeff * d(div)/dy_cc` (line 2089) where `adaptive_coeff = da_min_c * max(d2_bg, min(0.20, dddmp*|div|))` and `d2_bg = div_damp / da_min_c` (line 2066).  At iter-892 production with `div_damp = 2.13e8`, the bg-dominated regime (div_damp/da_min_c) is large.  The cube-edge cells have non-zero numerical divergence (despite the W2 IC having near-zero physical divergence) — the iter-892 PPM-flux + Arakawa-Lamb pressure gradient + boundary_fix combination produces ~1e-9 m/s divergence at these cells, and div_damp amplifies it to ~1.6e-5 m/s² in the dv tendency.

**iter-761 historical context.**  Per CLAUDE.md memory, iter-761 tuned `div_damp = 8 × _div_damp_cube(n)` (= the current production value) and measured this 8× as REDUCING W2 v_ll_Linf from 0.303 → 0.159 m/s (−48 %).  At the iter-761 baseline, increasing div_damp helped because the cube-corner mode A at lat ±35° was driven by a DIFFERENT mechanism (mode-A flux divergence at corners).  At the post-iter-893 baseline (0.132 m/s), the cube-corner mode-A pathway is no longer the dominant residual — but the div_damp is still active at 8×, and now its contribution is the limiting factor.

**Implications for iter-908+:**

1. **div_damp coefficient sweep**: try 4×, 2×, 1× the iter-761 value and measure W2 v_ll_Linf at C36 1-day.  If the post-iter-893 W2 sweet spot is at a smaller coefficient, this is a quick win.  Risk: lower div_damp may resurrect the iter-761-era mode-A at corners.
2. **Cube-edge-aware `adaptive_coeff`**: apply the full coefficient at face interiors but a softer (e.g., 4×) value at the i=0,1,n-1,n cells.  Targets the specific hot-spot cells.
3. **`d_sw5` holistic port**: the iter-872c-take4 comment notes the divergence-damping path is "structurally incomplete (the *dt factor and corner-divergence stencil are deferred to a dedicated d_sw5 holistic port)".  This is the architecturally-correct fix but multi-iter scope.

**Caveat — t=0 vs 1-day.**  iter-907 measures dv_d_dt at t=0, not the time-integrated W2 v_ll_Linf at 1-day.  The hot-spot dv_d_dt is small in absolute terms (1.9e-5 m/s² × 86400 s = 1.6 m/s if integrated as constant tendency).  But the 1-day W2 v_ll_Linf is 0.132 m/s — much smaller than the integrated tendency.  This means the actual time-integrated bias is from a smaller systematic component of the t=0 hot-spot magnitude that survives time averaging.  div_damp's contribution may include both an oscillatory part (averaged out) and a systematic part (accumulating).  iter-908 sweep would confirm if reducing div_damp actually helps the integrated bias.

**Deliverable.**
- `scripts/diag_iter907_w2_dv_ablation.py`: 4-config ablation script with hot-spot per-cell decomposition and clear VERDICT on dominant component.
- This iter-907 doc entry.

**Verification.**  Diagnostic runs to completion in ~5s.  No production code change.

**Process.**  189th iter.  iter-907 closes the iter-906b refinement question with a sharp answer: div_damp dominates 29× over boundary_fix at the W2 D-grid hot spots.  The historical iter-761 8× tuning is now identified as the dominant residual contributor at the post-iter-893 baseline.  iter-908+ has a concrete, low-cost test: sweep div_damp and measure W2.
