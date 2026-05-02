# SMC03 Density-Jacobian PGF — Code Review

Reviewer: ocean-model-expert / dycore-expert
Branch: `ocean-pgf-smc03`
Scope: review-only (no code edits) of the four new functions in
`src/legoesm/ocean/dynamics/latlon_cgrid_operators.py`, the dispatch
hook in `src/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py`, the
config field in `src/legoesm/ocean/state.py`, and the four
`tests/ocean/unit/test_pgf_smc03_phase{1,2,3,4}.py` test files.

Question being adjudicated: is the BH stress-test residual of 525 mm/s
(target < 5 mm/s) a fundamental accuracy limit of harmonic-linear
piecewise-linear S&M03 — meaning the next step is genuinely cubic
spline — or is the implementation carrying a fixable bug that, once
removed, would close the gap on linear-piece reconstruction alone?

The short answer up front: **there is one critical bug that almost
certainly accounts for most of the 525 mm/s, plus two correctness
concerns and a handful of sub-optimal choices.** The current Phase
3 / Phase 2 unit tests do not exercise the broken regime because the
test bathymetry is hand-tuned to keep the partial-cell thickness ratio
above the failure threshold I identify below. I recommend fixing the
critical bug and re-running BH before declaring cubic spline necessary.

---

## Issues by severity

### Critical

#### C1. Option-B target depth can fall below the partial column's seafloor → asymmetric clamp breaks rest-state cancellation

`src/legoesm/ocean/dynamics/latlon_cgrid_operators.py:2200-2211` and
`:2252-2258` (the y-direction analog).

The new Option-B target depth at face j is
`z_target_face = 0.5 * (z_c_W + z_c_E)`. The plan §2.3 / commit message
claims this midpoint is "by construction inside both columns' partial
cells." **It isn't.** Concretely, take a face between a full-cell W
column and a partial-bottom E column at level k = bot_active_E:

- Cells 0..k-1 are full in both columns and align, so
  `z_top_E[k] = z_top_W[k] = z_top`.
- `z_c_W[k]   = z_top + 0.5 · dz_ref[k]`
- `z_c_E[k]   = z_top + 0.5 · h_E[k]`     (h_E < dz_ref)
- `z_seafloor_E = z_top + h_E`
- Midpoint:
  `z_target  = z_top + (h_E + dz_ref)/4`
- `z_target − z_seafloor_E  =  (dz_ref − 3·h_E) / 4`

So `z_target > z_seafloor_E` whenever `h_E < dz_ref / 3`. In that
regime the call

```python
P_E = compute_pressure_at_target_smc03(rho, h_partial, ..., z_target_face, g)
```

clamps internally (`latlon_cgrid_operators.py:2117-2121`) so that
`z_t_clamped[E] = z_seafloor_E` while
`z_t_clamped[W] = z_target` (W is deeper, no clamp). The two columns are
then evaluated at *different* depths, and even for an exactly-linear
ρ(z) you get

```
P_E − P_W   ≈   ρ_W · g · (z_target − z_seafloor_E)
            =   ρ_W · g · (dz_ref − 3·h_E) / 4
```

For `ρ_0·g ≈ 10⁴ Pa/m` and `dz_ref/4 ≈ 100 m`, that is ~10⁶ Pa per face
even before any stratification. Divided by `dx_u` (a few × 10⁵ m at
1°) and by `ρ_0`, this becomes O(10⁻³ – 10⁻²) m/s² — easily enough to
drive a steady-state O(0.5 m/s) bottom-trapped current under r=10⁻³
linear drag. This matches the observed 525 mm/s magnitude almost too
well to be coincidence.

Why the tests miss it: the Phase 3
`TestLinearRhoSteppedBathymetryMachineZero` uses
`H_shallow=1750, dz=200`, giving `h_partial = 150 m`, ratio
`h/dz = 0.75 ≫ 1/3`. The Phase 2
`TestCrossColumnConsistency` uses `h_E = 50 m, dz=100, h/dz = 0.5`,
also above threshold; targets explicitly stay in `[0, 500m]`, never
probing the fragile region near either column's seafloor. So the unit
suite passes while BH (smoothing=5, r_max=0.54) generates many faces
with `h/dz < 1/3`.

**Recommended fix (in priority order):**

1. Switch the per-face target depth to
   `z_target = min(z_c_W, z_c_E)`  (the *shallower* of the two
   centroids). For a partial cell on E, `z_c_E < z_c_W`, so
   `z_target = z_c_E < z_seafloor_E < z_c_W < z_seafloor_W`, inside
   both columns by construction. This is in fact the depth choice
   Adcroft & Campin 2004 use for their face correction — and it
   reduces to the standard centroid for full-cell columns
   (backwards compat preserved). Check the existing
   `partial_cell_pgf_correction_x` at `:1877`:
   `face_ref = jnp.minimum(centroid_east, centroid_west)`.
   The SMC03 path should adopt the same convention.

2. As a guard, even after switching to `min(z_c_W, z_c_E)`, drop the
   clamp pathology at the seafloor — replace it with an
   "evaluate-as-if-cell-bot extrapolation": the in-cell linear
   formula already evaluates correctly at `z = z_seafloor` (the
   cell-bottom integral of the linear deviation vanishes). Saturating
   to the seafloor pressure as the function currently does is fine
   provided the target is genuinely below — but the bug above means
   you saturate when the target is still above the *deeper* column's
   physical depth, which is the asymmetry that doesn't cancel.

Once C1 is fixed, redo the BH headline run before considering cubic
spline. I expect the residual to drop below 50 mm/s (still above the
< 5 mm/s target, but in a regime where the remaining O(h²·ρ'') residual
is the genuine fundamental piece, not a clamp artifact).

### High

#### H1. σ at the partial-bottom is one-sided and not cross-column consistent

`latlon_cgrid_operators.py:2036-2040`.

For the partial-bottom cell the implementation uses
`σ = Δρ_top` (one-sided). For linear ρ this gives the true slope and
two adjacent columns agree by accident. For the nonlinear BH
exponential thermocline they don't:

- W (full at k, partial at k+1): σ_W[k] = harmonic-mean of slopes
  from the cells above and below.
- E (partial at k): σ_E[k] = (ρ_E[k-1] − ρ_E[k]) /
  (z_c_E[k-1] − z_c_E[k]).

`z_c_E[k] = z_top + 0.5·h_E` is *shallower* than `z_c_W[k]`, and
ρ_E[k] is the cell-mean over a thinner cell, so σ_E[k] is
the secant slope between two non-collocated points. σ_W[k] uses three
cells centered on a different depth. Even at machine precision the two
slopes disagree by O(h·ρ''), and the in-cell linear evaluation at any
common z is shifted by O(h²·ρ'').

The plan acknowledges this in §4 risk #4. The "Cell-above σ extension"
fallback you describe in `partial_cells_results.md` (line 307) is the
right idea but you reverted it without seeing a clear improvement —
plausibly because C1 (the Option-B clamp asymmetry) was masking the
σ-asymmetry. With C1 fixed, this becomes the next-leading residual.

**Recommended investigation:** with C1 fixed, retry the σ_{k-1}
extension at each column's partial-bottom. The argument *for* it: at
the face between W (full) and E (partial), if you set
`σ_E[bot_E] = σ_E[bot_E - 1]` and `σ_W[bot_E] = σ_W[bot_E - 1]` (use
the cell-above slope), both columns reconstruct ρ at z_target with the
*same* σ as long as cells bot_E - 1 are full in both columns (they
are). For linear ρ this stays machine-zero (σ is constant anyway). For
exponential ρ it kills the σ-mismatch term. The previous attempt
recorded as "negligible effect" likely failed only because the C1 clamp
bug was the dominant residual.

#### H2. `compute_pressure_at_target_smc03` argmax with inactive (h=0) cells: the "first True" semantics work only by accident

`latlon_cgrid_operators.py:2125-2132`.

After the partial-bottom cell, all inactive cells share
`z_top = z_bot = z_seafloor`. For `z_t_clamped = z_seafloor` (anytime
the target is at or below the seafloor — including the C1 clamp
case), `in_cell` is True at the partial-bottom *and* at every inactive
cell. `argmax` picks the first True → the partial-bottom cell.

This is correct (you want partial-bottom σ, ρ, P_top). But it depends
on the convention that `argmax` returns the lowest index on ties, plus
the convention that `cumsum` of `h_partial=0` for inactive cells
correctly gives `z_top_inactive = z_seafloor`. Both currently hold,
but neither is stated as a load-bearing invariant in the function
docstring. Add a comment documenting that this is the intended
behaviour, and add a unit test specifically for the
"all-inactive-tied-at-seafloor" case so an innocuous future change
to the cell-top accumulation (e.g. switching to `cumsum_exclusive`)
won't silently move the argmax to k=nlev-1 (which would have σ=0,
ρ undefined, and would *silently* break the rest state).

Suggested test: pad a 4-active-level column with 4 inactive levels,
evaluate at z_t = z_seafloor and z_t = z_seafloor + 100 m (forces
clamp), and verify P equals the analytic P_bot.

### Medium

#### M1. The dispatch in `ocean_pe_latlon_cgrid.py` *appends* the SMC03 result onto a previously-computed centred gradient

`ocean_pe_latlon_cgrid.py:950-1012`.

The structure today is:

```
# unconditional (centred KE+pressure batched call)
dp_dx, dp_dy = gradient_*_cgrid(p_prime_filled, grid)

# only for partial-cell coord:
if pgf_scheme == "smc03":
    dp_dx = density_jacobian_pgf_smc03_x(...)   # OVERWRITES
    dp_dy = density_jacobian_pgf_smc03_y(...)
else:
    dp_dx += partial_cell_pgf_correction_x(...)
    dp_dy += partial_cell_pgf_correction_y(...)
```

Functionally this is correct — the SMC03 branch fully overwrites
`dp_dx`/`dp_dy`. But it means we *waste compute* always doing the
centred-diff gradient on `p_prime_filled` and never using its result
(in the SMC03 branch). More worryingly it means the
`p_prime_filled = _neumann_fill_cgrid(p_prime, mask)` step at line 829
is also unused in the SMC03 branch — but `p_prime` itself is computed
unconditionally (line 821, via `iterate_eos_and_pressure_anomaly`),
which is fine because we *do* need ρ' from the same EOS pass. There is
no correctness issue, but:

- The dtype-cast at line 1011-1012
  `dp_dx = dp_dx_smc.astype(dp_dx.dtype)` is functioning as the bridge.
  Verify that `dp_dx.dtype` is always `float32` in the standard policy
  (it inherits from `p_prime`, which inherits from `T`). If the SMC03
  internals ever escalate to float64 (e.g. via `jnp.cumsum` on x64),
  this cast is correct but silently loses precision — and the AD pass
  will need to handle the cast. Unit test: run the Phase 4 dispatch
  test under `PrecisionPolicy.fp32()` and verify
  `tend.du_dt.dtype == float32`.

- Cosmetic but worth a TODO: skip the `gradient_x_cgrid(p_prime_filled,
  ...)` batched call when `pgf_scheme="smc03"` and `isinstance(z_coord,
  PartialCell)`. Saves a ~free op now, but on real-ETOPO it stops
  being free. Not a correctness issue.

#### M2. SMC03 relies on `rho_prime` from `iterate_eos_and_pressure_anomaly`, but the in-cell σ formula assumes cell-mean ρ values

`ocean_pe_latlon_cgrid.py:1001-1008` passes `rho_prime` (the EOS
output minus rho_0). The harmonic σ stencil treats this as the
*cell-mean*. That's fine because `iterate_eos_and_pressure_anomaly`
in fact returns ρ evaluated at the cell *centroid* (it iterates
`rho ← EOS(T, S, p_hydro(rho))` with p_hydro at the centroid).
*Cell-mean ρ for a piecewise-linear reconstruction equals
ρ-at-centroid* (the linear deviation integrates to zero across the
cell), so this is consistent — but only because the σ slope estimator
uses centroid-to-centroid finite differences. If a future change moves
the σ estimator to use rho-at-interfaces, or moves
`iterate_eos_and_pressure_anomaly` to compute cell-averaged ρ via
quadrature, the consistency breaks silently. Add a one-line invariant
assertion / comment in `density_jacobian_pgf_smc03_x` like:

> Assumes `rho_per_cell` is ρ at the cell centroid (which equals the
> cell-mean for a piecewise-linear reconstruction). Required by the
> σ stencil and the in-cell integral both being centroid-anchored.

Not a current bug. Future-proofing.

#### M3. `z_target_face` shape interaction inside `compute_pressure_at_target_smc03`

`latlon_cgrid_operators.py:2125-2132`.

Inside the function, `z_target` is interpreted as `(..., n_targets)`,
and the broadcast against cells gives a `(..., nlev, n_targets)` mask.
For the SMC03 face PGF the caller passes `z_target_face` with shape
`(n_lat, n_lon, nlev)` — i.e. **n_targets = nlev**. That allocates an
intermediate of shape `(n_lat, n_lon, nlev, nlev)`. At BH (e.g.
`64×64×20`) this is ~2 MB, fine. At 1°×30 levels it's 47 MB, tight but
fine. At ¼°×60 it's ~30 GB — already flagged as risk #3 in plan §4.

The correctness implication today: none. But the
`take_along_axis(rho_per_cell, k_t, axis=-1)` call at line 2135 expects
`k_t` to have shape `(..., n_targets)` (which it does after argmax over
the cells axis, yielding `(..., n_targets)`). Confirmed correct. Note
for future ¼° work: replace the per-target argmax with a per-cell
contribution accumulation (`P(z_t) − P_top_kt` only depends on cells
≤ k_t, expressible as a `lax.cumsum`-friendly form).

### Low / nits

#### L1. The σ harmonic-mean denominator floor uses `eps=1e-30` but accepts negative `sum_slopes`

`latlon_cgrid_operators.py:2030-2032`:

```python
sum_slopes = delta_top + delta_bot
safe_sum = jnp.where(jnp.abs(sum_slopes) > eps, sum_slopes, eps)
```

When both slopes are nonzero same-sign (the regime where σ_harm is
used), `sum_slopes` cannot be < eps in magnitude *unless* the slopes
exactly cancel — in which case the next `same_sign` mask (line 2033)
sends σ to 0 anyway. So the floor is dead code in the
`σ_harm` branch and acts only as a NaN guard for the AD pass. Fine.
But: when `sum_slopes` is negative (but `|·| > eps`), `safe_sum =
sum_slopes` — i.e. you allow negative sums. For same-sign slopes both
negative (a stable density profile is monotone increasing downward,
so `δ_top, δ_bot > 0`; for an inverted profile both would be < 0), the
sum is negative, `2·δ_top·δ_bot/sum` is **negative-of-positive** =
negative. Same sign as δ. ✓

So this works. It's correct but obscure. Add a one-line comment:
"Both slopes can be negative when ρ decreases with depth (top of
mixed layer with stronger inversion); harmonic mean preserves sign
naturally."

#### L2. `density_jacobian_pgf_smc03_y` pole padding zeros only the boundary v-faces but does not mask interior v-faces over land

`latlon_cgrid_operators.py:2261-2262`. The function pads the boundary
to zero (correct: pole walls). It does not multiply by `v_mask_3d`
internally. That is consistent with `gradient_y_cgrid` (also unmasked)
and the final masking happens at `ocean_pe_latlon_cgrid.py:1532-1533`
on the assembled `du_dt/dv_dt`. Documented in the SMC03 docstring
("face mask in the integrating PE step gates the result downstream").
No bug, just confirm with a test that interior land-face contributions
don't matter — they are clobbered by `v_mask_3d` regardless.

#### L3. The "Option B reduces to Option A on full-cell faces" claim

Plan §2.3 / docstring at `:2168`: "At full-cell faces this reduces to
the standard reference-cell centroid (both centroids equal
|z_full_ref[k]|)." True for `z*` coord *only if eta=0*. Under non-zero
η the live centroid depends on η through the Jacobian, and with eta=0
reference (Adcroft path comment at `ocean_pe_latlon_cgrid.py:1019`)
SMC03 would also use η=0. But SMC03 reads `z_centroid =
cumsum(h_partial) - 0.5*h_partial` (line 2191) using `h_partial`
*directly*, which is η=0 by definition for partial cells. Adcroft path
calls `compute_centroid_depth(zeros_like(eta_safe), H_bathy, z_coord)`
to enforce the same. **Both paths are eta=0.** OK — but worth a
docstring line explicitly aligning the two.

#### L4. The Phase 5 fallback list in `partial_cells_results.md` lists "curvature term κ in the in-cell integral" as negligible

`partial_cells_results.md:301-306`. Almost certainly negligible
*because* C1 was the dominant error. With C1 fixed, the κ correction
becomes interesting again — it's the leading remainder term once σ
is consistent. Worth re-trying.

#### L5. `gradient_x_cgrid` inside the unconditional batch is wasted compute when SMC03 path takes over

Already noted in M1. Cosmetic.

#### L6. The Phase 2 continuity test uses `eps = 1e-6` and `atol = 0.1 Pa`

`tests/ocean/unit/test_pgf_smc03_phase2.py:133-147`. Fine, but a
tighter test using `eps = 1e-9` and `atol = 1e-3 Pa` would catch a
class of off-by-one cell-lookup bugs. Trivial improvement.

---

## Answers to specific questions in the prompt

1. **Math bugs.** Harmonic σ formula correct. In-cell integral
   correct. P_top accumulation correct (cell-mean integral of linear
   deviation is zero). Sign convention consistent (positive
   downward). The `0.5·σ·(z_t + z_top - 2·z_c)` factor is the
   correct closed-form for the linear deviation integrated from
   `z_top` to `z_t`. ✓

2. **Boundary handling.** Top cell ✓, partial-bottom σ one-sided
   (correct but see H1), inactive σ → 0 ✓, land columns: the
   final `du_dt * u_mask_3d` masks them downstream — fine. The
   one-sided σ formulation at partial-bottom is the structural
   weakness for nonlinear ρ.

3. **Cross-column consistency.** Implementation produces identical
   ρ(z) reconstructions in adjacent columns *only* in the linear-ρ
   case. For nonlinear ρ, σ at the partial-bottom is a one-sided
   secant in the partial column vs a three-point harmonic in the
   full column → mismatch (H1). And worse: when h_E < dz_ref/3,
   the Option-B target depth falls below the partial seafloor and
   the asymmetric clamp creates the residual that I believe drives
   most of the 525 mm/s (C1). The W-roll itself is correct
   (periodic-wrap matches the Adcroft and centred-gradient
   conventions).

4. **Option B specifically.** Implementation matches the stated
   formula `0.5·(z_c_W + z_c_E)`. At pole rows the v-direction
   is zeroed via `jnp.pad`. At full-cell faces both centroids equal
   `|z_full_ref[k]|`, so it reduces to Option A there. **But the
   "by construction inside both columns" claim is wrong** — see C1.
   I recommend switching to `min(z_c_W, z_c_E)` (Adcroft-style),
   which IS by construction inside both columns and reduces to the
   same Option A on full cells.

5. **`compute_pressure_at_target_smc03` correctness.** Argmax-based
   lookup is correct for interior targets and at interface ties.
   Behaviour at z_t = z_seafloor is correct *by accident* due to
   argmax-first-True semantics over inactive-cell ties (H2).
   `take_along_axis` works correctly with the (..., nlev) → (..., n_t)
   reduction. Clamp to [0, z_seafloor] gracefully handles
   below-seafloor targets but, per C1, clamping in only one of the
   two paired columns is the asymmetry that doesn't cancel.

6. **AD pitfalls.** `same_sign` test produces piecewise gradients;
   safe-divide pattern keeps both branches finite; argmax has zero
   gradient w.r.t. inputs (only ρ-values flow through). The Phase 1
   AD smoke tests pass. ✓ I don't see an AD bug here.

7. **Dispatch correctness.** SMC03 path overwrites `dp_dx/dy` cleanly
   (line 1011-1012). The previously-computed centred gradient is
   discarded — wasted compute, not contamination. Adcroft branch in
   the `else` is unchanged from the prior PR. ✓

8. **Integration with rest of model.** The downstream H&A column-sum
   identity test (Phase 4 test 4) passes — confirms barotropic /
   tracer-flux paths are unchanged. KE gradient is independent of
   PGF. Vertical advection uses w from divergence, also independent
   of PGF. Drag, hyperdiff are independent. The only coupling is
   that `du_dt/dv_dt` enters the slow-forcing depth-average for the
   barotropic solver (`F_slow_u/v` at line 254). A larger-magnitude
   PGF tendency under SMC03 produces a larger `F_slow`, which the
   barotropic solver then drives — but that's *correct* coupling, not
   contamination.

9. **Bug vs fundamental.** I believe **C1 is the dominant error**.
   For BH at smoothing=5, r_max=0.54, the geometry generates many
   faces with `h_partial / dz_ref < 1/3` between full and partial
   columns. Each such face contributes a residual of order
   `ρ_0·g·(dz − 3·h_partial)/(4·dx_u·ρ_0) ≈ 10⁻³ – 10⁻²` m/s² that
   does not vanish for any ρ profile (linear or otherwise) and
   doesn't cancel between adjacent columns. This is consistent in
   sign with a "smooth bottom-trapped current" (because the residual
   is concentrated at the partial-bottom level on every seamount-
   slope face) and consistent in magnitude with O(0.5 m/s) under
   r=10⁻³ linear drag. **My prediction**: with C1 fixed, BH at
   smoothing=5 gives < 50 mm/s. With C1 + H1 fixed (σ extension
   at partial-bottom), < 10 mm/s. Cubic spline only becomes
   necessary if both fixes still leave residual > 5 mm/s.

10. **Missed accuracy.**
    - σ at the partial-bottom: the third-order one-sided FD
      (using ρ_{k-2}, ρ_{k-1}, ρ_k centroids) would buy you
      O(h²·ρ''') vs the current O(h·ρ''). Cheap; worth trying after
      C1.
    - P_top cumsum uses cell-mean = ρ_centroid, valid for
      piecewise-linear. For piecewise-parabolic this would need a
      genuine quadrature. Not relevant for the current scheme.
    - σ at faces vs cells: S&M03's full §4.2 puts σ at faces and
      reconstructs cubic-Hermite-style. That's the cubic-spline
      upgrade.

11. **Optimization-as-bug.** The `z_full_ref` parameter dropped from
    the Option-B refactor: confirmed by grep that `z_full_ref` is no
    longer referenced in the SMC03 path; nothing else breaks. The
    cell axis assumption `axis=-1`: works because the lat-lon shape
    is `(n_lat, n_lon, nlev)` everywhere SMC03 is called. The σ
    function uses `axis=-1` for cell rolls and is called with
    arrays whose trailing axis is `nlev`. ✓

---

## Top-3 recommended changes

In priority order:

1. **Fix C1 first.** Replace
   `z_target_face = 0.5 * (z_c_W + z_c_E)` with
   `z_target_face = jnp.minimum(z_c_W, z_c_E)` in both
   `density_jacobian_pgf_smc03_x` (`:2203`) and `_y` (`:2252`).
   This matches the Adcroft & Campin convention, is by construction
   inside both columns, and reduces to the standard centroid for
   full cells. Add a Phase 3 test that builds a face with
   `h_partial / dz_ref = 0.2` (well inside the failure regime) and
   verifies SMC03 gives machine-zero rest-state PGF for linear ρ.

2. **Try σ extension at the partial-bottom (re-do the abandoned
   fallback).** With C1 fixed, set
   `sigma[bot_active] = sigma[bot_active − 1]` for each column
   *after* the harmonic-mean computation. For linear ρ this is a
   no-op (σ already constant); for nonlinear ρ this makes both
   adjacent columns use the same σ at the partial-bottom. Re-run BH;
   I expect this to take residual from "≤ 50 mm/s" down to "≤ 10 mm/s"
   range.

3. **Add a regression test that probes the failure regime**
   (`h_partial / dz_ref = 0.2`, full + partial face pair, linear ρ)
   to lock in C1 as a permanent invariant. This is the test that
   the current Phase 3 suite *should* have included from the start.

If after (1) + (2) + (3) the BH residual is still > 5 mm/s, the
cubic-spline upgrade is genuinely warranted and the existing scaffolding
(harmonic σ → in-cell integral → face PGF dispatch) is the right
foundation: only `reconstruct_harmonic_slopes` and
`compute_pressure_at_target_smc03` need to be replaced.

---

## Honest assessment

The 525 mm/s figure is **almost certainly not fundamental** to the
harmonic-linear scheme. The current Option B design has a clamp
asymmetry that fires whenever the partial-cell-thickness ratio drops
below 1/3 — exactly the regime BH at smoothing=5, r_max=0.54
generates. The unit tests do not exercise this regime, so the
"machine zero on linear ρ" claim of the Phase 3 test holds only on
the friendly bathymetry chosen there. On BH the same property fails
catastrophically because a face between W (full cell) and E (thin
partial cell) evaluates P_W at z_target inside W but P_E at the
clamped seafloor — the difference is O(ρ·g·dz/4) per face.

I would not implement cubic spline before fixing C1 and re-running BH.
If C1 + H1 leave residual > 5 mm/s on linear ρ — *now* cubic spline
is the answer. If they leave residual > 5 mm/s on exponential ρ but
machine-zero on linear ρ, that's the genuine O(h²·ρ'') accuracy floor
of harmonic-linear and cubic spline buys you the next decade.

The implementation craftsmanship is otherwise solid: AD-clean, dtype-
clean, matches the documented plan, integrates cleanly with the
existing pipeline, preserves backwards compat under
`pgf_scheme="adcroft"`. The one critical bug is a subtle geometric
case the test bathymetry happens not to probe.
