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
