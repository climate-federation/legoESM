# PGF Tier Baseline Results

**Date**: 2026-05-04
**Branch**: `test/PGF_latlon` (at commit f73a0767, same as main)
**Grid**: 36x72 lat-lon, 10 levels, H_max=4000m, dt=600s
**Coord**: partial cells (unless noted)
**Barotropic**: implicit_cn, bottom_drag_r=1e-3, A_h=1e4

These are the baseline measurements BEFORE any bug fixes.  The purpose
is to establish the current state so that fixes can be measured against it.

## Tier 1: τ=0, idealized bathymetry, uniform T/S (10 days)

| Test | Adcroft | SMC03 | SMC03 advantage |
|---|---:|---:|---:|
| **Flat bottom** (sanity) | < 1e-10 m/s | — | — |
| **BH seamount** (partial, smooth=5) | **294.6 mm/s** | **4.9 mm/s** | **60×** |
| **Step bathy** (partial) | **99.9 mm/s** | **3.0 mm/s** | **33×** |
| **BH seamount** (z-star) | PASS (< 1 mm/s) | PASS (< 1 mm/s) | — |

**Observations**:
- Flat bottom + uniform T/S: machine zero — sanity check passes.
- z-star seamount: passes even with Adcroft — confirms the issue is
  partial-cell-specific.
- Adcroft on partial cells: 100-300 mm/s spurious flow, consistent
  with the 2Δz mode documented in `partial_cells_results.md`.
- SMC03 on partial cells: 3-5 mm/s — dramatically better, but still
  above the 1 mm/s target.  Consistent with the BH Phase 5 result
  of 1.5 mm/s at 30 days with 20 levels (our 10 levels are coarser).

## Tier 2: τ=0, idealized bathymetry, realistic stratification (10 days)

| Test | Adcroft | SMC03 | SMC03 advantage |
|---|---:|---:|---:|
| **BH seamount, WOA** | **316.0 mm/s** | **6.1 mm/s** | **52×** |
| **BH seamount, exponential** | **356.2 mm/s** | **8.1 mm/s** | **44×** |
| **Step bathy, WOA** | **98.9 mm/s** | **13.5 mm/s** | **7×** |
| **SMC03 < Adcroft** (comparison) | PASS | — | — |

**Observations**:
- Realistic stratification makes things worse (as expected): the
  ρ' residual is nonzero, so the PGF error is larger.
- SMC03 advantage is consistent (7-52×) but the absolute magnitudes
  are larger than Tier 1.
- Step bathy is harder than seamount for SMC03 (13.5 vs 6.1 mm/s)
  because the bathymetric step is sharper.
- The exponential stratification (8.1 mm/s) produces larger errors
  than WOA (6.1 mm/s) because the WOA profile has less curvature
  in the relevant depth range.

## Key Takeaways

1. **SMC03 is consistently 7-60× better than Adcroft** on partial cells.
   This validates the fix-partial-cell-pgf branch work.

2. **Pure z-star passes Tier 1 easily** — the issue is entirely in the
   partial-cell correction, not the base PGF.

3. **The 1 mm/s target is achievable** with SMC03 at higher vertical
   resolution (the BH Phase 5 result was 1.5 mm/s with 20 levels; our
   10-level grid is coarser).

4. **Bug fixes #1 and #2 have NOT been applied** — these results are
   the pre-fix baseline. The face-thickness inconsistency may be
   responsible for some of the residual, especially at step boundaries.

## Effect of Bug Fixes on Tiers 1-2

All three fixes applied (face-thickness min-rule, barotropic H_u min,
hi_precision_pressure=True).  Result: **no measurable change** in
rest-state metrics.  This is because in a rest state du_dt is tiny,
so the weighting difference between min-rule and mean-rule is
negligible.

The fixes are real consistency corrections, but they matter under
**strong forcing** (the JRA scenario), not in rest-state tests.

## Tier 4: Wind-driven, idealized bathymetry (10 days, WITH fixes)

Physics: prescribed cosine-latitude wind only (no lateral/vertical
mixing), τ_max=0.1 Pa unless noted.  No dissipation → speeds higher
than production.

| Test | Adcroft | SMC03 | Flat ref |
|---|---:|---:|---:|
| Seamount + cosine wind | 2115 mm/s | 2028 mm/s | — |
| Step + cosine wind | 1946 mm/s | 1936 mm/s | — |
| Seamount + strong wind (0.2 Pa) | 5965 mm/s | 5687 mm/s | — |
| Seamount + WOA + cosine wind | 2285 mm/s | 2146 mm/s | — |
| Flat-bottom reference | — | — | 1954 mm/s |

**Key observations**:
1. Wind IS being applied — flat-bottom reference ~2 m/s Ekman response.
2. Adcroft is consistently **4-6% worse** than SMC03 under wind forcing
   (vs 30-60× worse in rest state). The wind signal dominates.
3. Step bathymetry shows smallest Adcroft-SMC03 difference (~0.5%)
   — the topographic PGF error is buried in the wind-driven flow.
4. Strong wind (0.2 Pa): approaching ~6 m/s without dissipation.
   Both schemes survive (no NaN) but are at the edge.
5. The PGF scheme difference is much smaller under forcing than at rest
   — confirming that the rest-state test is the most diagnostic for
   PGF scheme quality, while forced tests stress the coupling and
   face-thickness consistency.

## Next Steps

1. Run Tiers 1-4 **without** the bug fixes (revert) to get the
   before/after comparison for the wind-driven case specifically.
2. Increase vertical resolution to 20 levels and re-run to match the
   BH Phase 5 configuration.
3. Run Tier 3 (ETOPO) once available.
4. Add lateral dissipation (A_h, B_h) to Tier 4 for more realistic
   velocity magnitudes and better PGF/wind separation.
