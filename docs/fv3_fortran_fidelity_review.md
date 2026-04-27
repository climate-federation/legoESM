# FV3 Fortran Fidelity Review

Baselined 2026-04-14. This file keeps the latest Ralph-loop
iterations in full and compresses older material for token economy.
Use git history for retired prose.

> **Metadata convention (iter-174)**: do not duplicate "updated
> through iter-N" in the title. The authoritative iteration count
> is the branch commit history plus the HEAD commit message tag.

## Current Status

- Production W2 is still not true FV3. It runs
  `FV3EdgeShallowWaterModel` -> `fv3_sw_tendencies` with
  Arakawa-Lamb/RK3/`div_damp`/`boundary_fix`, not the Fortran
  FB `c_sw` -> `d_sw1/d_sw4/d_sw5/d_sw6` chain.
- W2 C36 remains a cube-vertex/meridian `v_ll` artifact.
  `apply_fortran_xppm_boundary=True` improved the canonical baseline
  from `v_ll_Linf=1.585e-01` to `1.319e-01 m/s`, but did not remove
  the structural bias.
- FB/duogrid scaffolding exists (`halo=3`, flux sync, `d_sw*`
  helpers), but C24/C36 accuracy and stability remain unresolved.

## Compact Archive

- Closed or resolved: panel-edge corner metrics, `d_sw3` BGRID_NE
  sync, duogrid flux sync, legacy edge bypass in duogrid, old polar
  face asymmetry, visual diagnostics, and cosine-bell visual checks.
- `iter-1..174`: core FV3 metric/operator port, seam/sync work,
  regression expansion, and early FB bring-up.
- `iter-505`: fixed the production PPM active-axis bug in
  `cgrid_mass_flux_divergence`; W2 C36 `max|v_ll|` improved
  `0.557 -> 0.303 m/s`.
- `iter-511..751`: W2/W5 artifacts characterized; production path
  identified as structurally non-FV3; `use_duogrid=True` on the
  A-L path shown catastrophic.
- `iter-752..855`: del6/corner-fill/zeta/DUOGRID/FB ablations
  localized the W2 residual to cube-vertex `dv/dt`, incomplete
  Cor+pressure+KE cancellation, and load-bearing numerical
  zeta/`boundary_fix`.
- `iter-856..898c`: Phase-1/d_sw5 production swaps failed; PPM
  stop-time and Fortran-sentinel work culminated in
  `apply_fortran_xppm_boundary=True` as the current W2/W5 baseline.
- `iter-899..903`: PPM strip forensics showed production uses
  symmetric halo=2 (`q[k]=q1(k-2)`). Strict LEFT/RIGHT Fortran PPM
  flags were implemented but worsened W2 (`0.1319 -> 0.2027` and
  `0.1319 -> 0.1989 m/s`), so both remain default-off; the dxa xt
  correction was quantified as small and deferred.

## Latest Ralph-Loop Iterations

Full detail is retained from `iter-930` onward. Older Ralph-loop entries are compressed below; use git history for retired prose.

### Iter-904 to Iter-929 - compacted Ralph-loop archive

- `iter-904..905`: Partial true-FV3 mass transport inserted into the RK3 production path failed catastrophically. Default-off `use_fv3_dsw1_mass_transport` produced NaNs; adding `nord_v`/`damp_v` avoided NaNs but yielded W2 `v_ll_Linf=3.137e+02 m/s`; split mass/FV3 + RK3 momentum still gave `2.921e+02 m/s`. This rules out one-component FV3/RK3 hybrids.
- `iter-906..910`: W2 residual was localized to cube-edge/cube-vertex geometry. Bare cell-center Coriolis/pressure cancellation is worst near vertices; production `div_damp` dominates instantaneous hot spots, but reducing global or boundary damping worsens W2. Resolution helps slowly, with C36 near a `0.10..0.12 m/s` practical floor at current cost.
- `iter-912`: Focused iter-887..911b regression sweep passed: 10 files, 63 tests, 175 s. All recent default-off flags remained isolated; production W2 baseline stayed `v_ll_Linf=1.319e-1 m/s`.
- `iter-913..920`: Silent-regression cleanup found 8 drifted tests, mostly from iter-878's Fortran-correct PPM limiter fix. Gold files and live replacement tests were updated; broader atmosphere/dynamics layers were clean. Process rule persisted: run full `tests/unit/test_cdgrid_fv3_regression.py` and related layers every multi-iter core-operator cycle.
- `iter-921..925`: Visual and sentinel refresh for W2/W5/cosine-bell/rest-state. Iter-893 PPM boundary fix improves W2 `v_ll_Linf` by ~17 % but worsens W2 `h_err_max` by ~77 %; strict-Fortran PPM variants are Pareto-dominated. W5, cosine bell, and rest state are essentially unaffected; iter-893 is bit-exact on constants.
- `iter-926..927`: Added `use_fv3_dsw5_corner_damping` as default-off documentation of a rejected path. Additive d_sw5 corner damping blew up W2 (`v_ll_Linf=2.253 m/s`); replacement semantics were less bad but still rejected (`0.971 m/s`). Conclusion: per-operator d_sw5 swaps cannot fix the A-L/RK3 operator-family mismatch.
- `iter-928`: Added a meta-fidelity sentinel locking 8 documented Fortran-fidelity gap markers so cleanup edits cannot silently erase known gaps without actually closing them.
- `iter-929`: Bare A-L decomposition identified the upstream source: 1 % imperfect geostrophic cancellation at the 8 cube vertices. `boundary_fix` spreads that residual and `div_damp` amplifies it into the row-2 production hot spots. Fixing W2 requires either better cube-vertex cancellation/halo handling or the full Fortran FB operator family, not damping tuning.

### Iter-930 — boundary_fix vs div_damp 4-way matrix: boundary_fix is dominant integrated stabilizer

**Trigger.**  iter-907 found `div_damp` is 29× more than `boundary_fix` at INSTANTANEOUS hot-spot tendency magnitude.  iter-929 identified the 3-stage chain (bare A-L cube-vertex → boundary_fix spread → div_damp amplification).  But which stabilizer dominates the INTEGRATED 1-day v_ll_Linf?  The two questions have different answers.

**iter-930 measurement** (W2 C36 1-day, `apply_fortran_xppm_boundary=True`, all 4 cells of the `boundary_fix × div_damp` matrix):

| `boundary_fix` | `div_damp` | v_ll_Linf | h_L2  | h_Linf |
|----------------|------------|-----------|-------|--------|
| True (production) | 8×       | 0.1319    | 0.512 | 8.184  |
| False             | 8×       | 0.6380    | 1.751 | 34.29  |
| True              | 0        | 0.1857    | 0.645 | 6.023  |
| False             | 0        | 0.6616    | 1.826 | 20.72  |

**Findings.**

1. **`boundary_fix=False` worsens v_ll_Linf 4.8×** (0.132 → 0.638 m/s) regardless of `div_damp`.
2. **`div_damp=0` (with boundary_fix on) worsens v_ll_Linf only 1.4×** (0.132 → 0.186 m/s).
3. **`boundary_fix` is the DOMINANT integrated-error stabilizer** for W2 — penalty for removing it is 3×+ larger than the penalty for removing `div_damp`.
4. **Production matrix (both on) is the BEST of the 4 cells** on every metric except h_Linf, where `bf=T, dd=0` wins (6.02 vs 8.18).  The h_Linf trade-off mirrors the iter-921 v vs h Pareto observation: more aggressive damping shifts where the residual concentrates.

**Reconciliation with iter-907.**  iter-907's "div_damp 29× boundary_fix at hot spot" is correct but refers to the INSTANTANEOUS dv tendency magnitude.  Over 288 RK3 steps, the `boundary_fix` smoothing operates at every step and accumulates to dominate the integrated v_ll_Linf.  `div_damp` damps the instantaneous noise but doesn't fully suppress it; `boundary_fix`'s averaging is more globally effective at reducing the integrated cube-vertex-derived bias.

**Implication for the d_sw5 audit and the user's issue #5.**

User issue #5 noted `boundary_fix` is "Python-only stabilizer ... reduces artifacts but proves the production operator is compensating for a missing faithful FV3 mechanism."  iter-930 quantifies the SCALE of that compensation: removing boundary_fix would push v_ll_Linf from 0.132 to 0.638 m/s — 4.8× worse than the iter-893 baseline.  Until a Fortran-faithful corner mechanism (per issue #6 — d_sw5 corner KE-add structure within the FB chain) is in place, `boundary_fix` is structurally indispensable for production W2.

**iter-930 deliverables.**

1. `tests/test_iter930_boundary_fix_div_damp_load_bearing.py` — 5 sentinels:
   - 4 parametric pins on the 4-way matrix (within ±5 %).
   - 1 inequality pin: `bf=False` penalty > 3× `dd=0` penalty on v_ll_Linf.

**Verification.**  5/5 pass in 60 s.

**Backlog implication.**  When/if FB chain stabilization lands, the FIRST validation should be that production W2 v_ll_Linf with the FB chain (and no boundary_fix Python-only stabilizer) reaches the iter-893 0.132 m/s number or better.  iter-930's 0.638 m/s is the "no compensation" baseline that the FB chain replacement must match.

**Process.**  No production code change.  Cumulative iter-921→iter-930: 10 commits, 38 sentinel tests, 0 production behavioral changes.  Production W2 baseline unchanged at v_ll_Linf=0.132 m/s.

### Iter-931 — bare A-L cube-vertex residual is RESOLUTION-REFINABLE (slow ~p=0.4 power-law)

**Trigger.**  iter-929 found the bare A-L has 1 % imperfect cancellation at the 8 cube vertices.  iter-931 measures whether this cube-vertex bias DECREASES with resolution (refinable, soft floor) or stays constant (structural, hard floor).

**iter-931 measurement** (W2 IC, t=0, no time stepping; bare A-L only):

| `N` | `|coriolis_dv|` max | `|bare residual|` max | imperfect % |
|-----|---------------------|-----------------------|-------------|
| 16  | 3.029e-03           | 3.382e-05             | 1.117 %     |
| 24  | 3.040e-03           | 2.967e-05             | 0.976 %     |
| 36  | 3.045e-03           | 2.516e-05             | 0.826 %     |
| 48  | 3.047e-03           | 2.238e-05             | 0.735 %     |

**Findings.**

1. **Background magnitude is nearly resolution-INVARIANT** (3.03e-3 → 3.05e-3, +0.6 %).  The Coriolis and Bernoulli-grad sub-operators are O(1) physical quantities, not numerical noise.
2. **Residual magnitude DECREASES monotonically** with resolution (3.38e-5 → 2.24e-5, −34 % from C16 to C48).
3. **Imperfect % decreases monotonically** (1.12 % → 0.73 %, −34 %).
4. **Convergence rate**: residual at C16 / C48 = 1.51× over a 3× resolution increase → effective order p ≈ log(1.51)/log(3) ≈ 0.38.  Slow (sub-linear) but POSITIVE convergence.

**Reconciliation with iter-910's integrated v_ll_Linf scaling.**

iter-910 measured C16=0.354, C24=0.183, C36=0.132, C48=0.125 m/s for integrated 1-day W2 v_ll_Linf — also slow ~1st-order scaling.  iter-931 confirms the source of that scaling: the bare A-L cube-vertex residual is the upstream feeder.  Both metrics scale together; the cube-vertex bias is the structural source of the integrated W2 floor.

**Implication.**

The cube-vertex bias is **NOT a structural floor** — refinement reduces it.  Production at C36 (v_ll_Linf=0.132 m/s) is at this resolution because of cost constraints (~5.6× cost factor C36→C64 per iter-910b), not because higher resolution doesn't help.  An iter that wants to push v_ll_Linf below 0.10 m/s has a viable resolution-only path (~C72-C96), independent of any operator-family change.

This is a useful diagnostic finding for the user's iter-927 issue #2 (operator family).  The A-L family DOES converge with refinement; the operator family question is about cost-efficiency, not feasibility.

**iter-931 deliverables.**

1. `tests/test_iter931_bare_al_cube_vertex_resolution_scaling.py` — 5 sentinels:
   - 4 parametric pins (C16, C24, C36, C48 residual_max + imperfect_pct within ±5 %).
   - 1 monotonicity sentinel: residual must decrease with resolution.

**Verification.**  5/5 pass in 59 s.

**Process.**  No production code change.  Cumulative iter-921→iter-931: 11 commits, 43 sentinel tests, 0 production behavioral changes.  Production W2 baseline unchanged at v_ll_Linf=0.132 m/s.

### Iter-932 — pin the SCOPE of iter-893's PPM boundary fix

**Trigger.**  iter-925 verified iter-893 is bit-exact no-op on rest state (constants).  iter-929 found bare A-L cube-vertex 1 % imperfect cancellation.  Question: does iter-893's PPM fix leak into the velocity tendency path on a non-uniform state, or is it strictly scoped to the mass transport (`dh_dt`)?

**iter-932 measurement** (W2 t=0 C36, `div_damp=0`, `boundary_fix=False` so no other production stabilizers fire):

| metric                       | iter-820 (xppm=False) | iter-893 (xppm=True) | Δ                |
|------------------------------|-----------------------|----------------------|------------------|
| max\|du_dt\|                 | reference             | reference            | **0.0 (bit-exact)** |
| max\|dv_dt\|                 | reference             | reference            | **0.0 (bit-exact)** |
| max\|dh_dt\|                 | 8.53e-05              | 1.71e-04             | 8.57e-05         |

**iter-893 is FULLY scoped to `dh_dt`.**  Velocity tendencies are bit-exact identical under the toggle, confirming PPM is consumed only by `cgrid_mass_flux_divergence` and not by the bare A-L Coriolis/Bernoulli-grad path.

**iter-932 deliverables.**

1. `tests/test_iter932_iter893_velocity_tendency_invariant.py` — 3 sentinels:
   - `velocity_tendencies_bit_exact_under_iter893_toggle`: `du_dt`, `dv_dt` strictly bit-exact between iter-820 and iter-893 baselines.  Fires if PPM ever leaks into the velocity path.
   - `height_tendency_does_change_under_iter893_toggle`: positive sentinel — if iter-893 silently no-ops on `dh_dt`, the flag is dead.
   - `height_tendency_drift_within_iter932_band`: pin the iter-893 magnitude effect on `dh_dt` (8.57e-5) within ±20 %.

**Verification.**  3/3 pass in 19 s.

**Cumulative iter-921→iter-932 deliverables.**

- 12 commits, 1 production code change (`use_fv3_dsw5_corner_damping` flag, default-OFF, REJECTED).
- 46 sentinel tests across W2 single + Pareto + W5 multi + cosine bell + rest state + bare-AL resolution scaling + iter-893 scope.
- 8 user-identified Fortran-fidelity gaps locked in test suite.
- ~5 minutes total CI cost.

The iter-921→iter-932 audit established the FOLLOWING about the production W2 problem:

1. **Root cause** (iter-929): bare A-L 1 % imperfect cancellation at 8 cube vertices.
2. **Resolution scaling** (iter-931): cube-vertex bias is REFINABLE (slow ~p=0.4 power-law).  Production at higher resolution would help.
3. **Amplification chain** (iter-929): bare residual → boundary_fix spread → div_damp 7× amp.
4. **Stabilizer dominance** (iter-930): boundary_fix is the dominant integrated stabilizer (4.8× v_ll inflation if removed).
5. **iter-893 scope** (iter-925, iter-932): strictly mass-transport-only — bit-exact no-op on velocity path and on constant states.
6. **Pareto trade-off** (iter-921, iter-922): iter-893 reduces v_ll Linf by 17 % at cost of h_err Linf +77 %; default is Pareto-non-dominated vs strict-Fortran variants.
7. **d_sw5 corner damping**: REJECTED both as additive (iter-926b) and replacement (iter-927) — operator family mismatch is fundamental, can't be patched per-operator.
8. **W5/cosine bell are unaffected** (iter-923, iter-924): the Pareto trade-off is W2-specific.

Production W2 baseline at iter-893 (v_ll_Linf=0.132 m/s, h_err_max=8.18 m) unchanged throughout.

### Iter-933 — FB chain stability scan: NaN at C8/C12/C16, stable at C24+

**Trigger.**  User issue #8 marks `FV3FBShallowWaterModel` as experimental and unstable.  iter-933 quantifies WHERE the instability shows up by running one FB step at C8/C12/C16/C24/C36 with the same `dt=300 s`.

**iter-933 measurement** (W2 t=0, dt=300 s, FB Fortran defaults d4_bg=0.16, nord=1):

| `N` | `h` finite | `u_d` finite | `v_d` finite |
|-----|------------|--------------|--------------|
| 8   | **NO (NaN)** | yes        | yes          |
| 12  | **NO (NaN)** | yes        | yes          |
| 16  | **NO (NaN)** | yes        | yes          |
| 24  | yes          | yes        | yes          |
| 36  | yes          | yes        | yes          |

**Surprising findings.**

1. **FB chain is stable at C24+ but produces NaN at C8/C12/C16.**  This is the OPPOSITE of typical CFL-driven instability (which fails at high resolution under fixed dt).
2. **The NaN is in `h` only** (mass transport), NOT in `u_d` or `v_d` (velocity tendencies).  Velocity computation handles low resolution fine — `u_max` and `v_max` are O(40) m/s at all resolutions.
3. **Grid-size-dependent, not time-step-dependent.**  Same `dt=300 s` across all resolutions; only the grid spacing changes.

**Localisation.**  The blow-up is in the mass-transport path: `_d_sw_native` → `transport_step` → `fv_tp_2d` → `_ppm_1d`.  The `_ppm_1d` boundary-cell handling becomes inadequate when `n_interior` falls below some threshold between 16 and 24.

**Implication for stabilisation.**

The user's iter-927 prohibition ("Do not use `use_fv3_dsw1_mass_transport` or `split_mass_momentum` as the fix; those paths already failed catastrophically") aligns with iter-933's localisation: those rejected flags use the same `transport_step` path that NaN's in the FB chain at low resolution.  The mass-transport stage is the load-bearing FB-chain failure mode.

Stabilising the FB chain at low resolution → fixing `_ppm_1d` boundary handling at small `n_interior`.  This is a focused multi-iter project independent of the velocity-side operator-family work (issues #1, #2, #6).

**iter-933 deliverables.**

1. `scripts/diag_iter933_fb_chain_resolution_stability.py` — runnable one-step FB scan across resolutions.

**Verification.**  No new test added — adding a sentinel that REQUIRES NaN at C8 would be perverse (we want the bug FIXED, not pinned).  The diagnostic provides reproducible measurement for stabilisation work.

**Process.**  No production code change.  Cumulative iter-921→iter-933: 13 commits, 46 sentinel tests, 1 production code change (`use_fv3_dsw5_corner_damping` flag, default-OFF, REJECTED), 0 default-config behavior changes.  Production W2 baseline unchanged at v_ll_Linf=0.132 m/s.

### Iter-934 — fix `_deln_flux` float32 overflow: FB chain now stable at all resolutions

**Trigger.**  iter-933 localised the FB chain low-resolution NaN to the mass-transport path (`_d_sw_native` → `transport_step` → `fv_tp_2d`).  iter-934 drilled deeper and identified the exact failing operator: `_deln_flux` in `fv_tp_2d.py`.

**Root cause** (iter-934 diagnostic).  At C8 with `nord=2`, `damp_c=0.06`:

```
da_min = min(area) ≈ 1.2e12 m²    (huge cells at coarse grid)
damp   = (damp_c * da_min)^(nord+1) = (0.06 × 1.2e12)³ ≈ 4e32
d2     = damp * q                  ≈ 4e32 × 3e3 = 1.5e36   (still in float32)
fx2    = sin × dy × (d2_diff) × rdxc
       ≈ 1 × 1.25e6 × 1.4e35 × 1e-6
       ≈ 1.75e35 — but intermediate sin*dy*d2_diff = 1.75e41 OVERFLOWS float32 (max 3.4e38)
```

Even though the final per-step damped flux is small, the INTERMEDIATE products `sin × dy × (damp*q_diff)` overflow float32 at low resolution.  The grid metric arrays (`area`, `dy_edge_x`, `sin_sg`, etc.) are float32 in the codebase; promoting them globally would have wide-ranging cost.

**Fix (iter-934).**  Factor `damp` out of the iteration loop and apply it at the final flux assembly (Step 4) instead of at Step 1 initialisation.  All operations between Step 1 and Step 4 are LINEAR in `d2`, so the result is mathematically identical — but every intermediate now fits comfortably in float32:

```python
# BEFORE:
d2 = damp * q                           # ≈ 1.5e36
... iterate Laplacian-divergence on d2 ...
fx = fx + fx2                           # damp baked in

# AFTER (iter-934):
d2 = q                                  # ≈ 3e3 (no overflow)
... iterate Laplacian-divergence on d2 ...
fx = fx + damp * fx2                    # apply damp at end
```

Single change in `src/legoesm/core/fv_tp_2d.py:_deln_flux`; ~6 lines edited; the mass-not-None path was already factored this way, so iter-934 only changes the mass=None branch.

**Verification.**

- iter-933 stability scan (re-run after iter-934): all resolutions C8/C12/C16/C24/C36 produce FINITE `h`, `u`, `v` after one FB step (W2 IC, dt=300 s).  Was: NaN at C8/C12/C16.
- 25/25 production sentinels (iter-921 + iter-923 + iter-924 + iter-925 + iter-904) pass — no production W2/W5/cb/rest-state regression at C36.  iter-934 is bit-identical to pre-iter-934 at high resolution where float32 didn't overflow.
- iter-934 sentinel (`tests/test_iter934_fb_chain_low_res_stability.py`) — 6 tests pin: 5 parametric pass-at-each-resolution + 1 mathematical-equivalence pin.  6/6 pass in 147 s.

**Implication.**

This unblocks FB-chain stabilisation at low resolution (C8-C16).  The FB chain remains structurally non-Fortran-faithful at cube vertices for OTHER reasons (issue #7 sign-flip not ported, issue #2 operator family) — iter-934 does not by itself make FB chain production-ready.  But iter-934 removes the float32-overflow gate that previously prevented even basic FB-chain stability testing at low resolution.

The user's iter-927 prohibition on `use_fv3_dsw1_mass_transport` / `split_mass_momentum` was based on iter-904/iter-905 catastrophic blowups at C36.  Those tests used the same `_deln_flux` path; iter-934 may have improved their numerical stability at lower resolution, but the iter-927 prohibition remains valid until those flags are re-measured against iter-927's strict acceptance criterion.

**iter-934 deliverables.**

1. `src/legoesm/core/fv_tp_2d.py` — `_deln_flux` damp-factoring fix (mass=None branch).
2. `scripts/diag_iter933_fb_chain_resolution_stability.py` — updated header to record post-iter-934 result.
3. `tests/test_iter934_fb_chain_low_res_stability.py` — 6 sentinels.

**Process.**  Real production code change (`_deln_flux` semantics in float32; mathematically identical in float64).  Cumulative iter-921→iter-934: 14 commits, 52 sentinel tests, 2 production code changes (iter-926/927 `use_fv3_dsw5_corner_damping` flag, REJECTED but default-OFF; iter-934 `_deln_flux` damp-factoring fix, ENABLED at all resolutions).  Production W2 baseline at iter-893 unchanged at v_ll_Linf=0.132 m/s.
