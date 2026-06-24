# Ralph task: internal_tide — fix grid-scale w-noise at the partial-cell staircase

## Goal
Make legoESM faithfully reproduce the Oceananigans `internal_tide` validation case
(scoreboard row 2d), against the REAL Oceananigans reference, by removing the
grid-scale (2Δx) vertical-velocity noise generated at the partial-cell staircase.
FAITHFUL ONLY — no backstops, no smoothing that smears the physical signal, no
fabricated numbers. The case is EXCEPTIONALLY sensitive: the real radiating tide
is ~2e-5 in b', so any spurious grid-scale noise swamps it.

## You have (use them)
- **Oceananigans SOURCE** (read the CORRECT implementation): `/tmp/ocn_j11_depot/packages/Oceananigans/NCFoc/src/`
  — continuity / w: `Models/HydrostaticFreeSurfaceModels/compute_w_from_continuity.jl` (find it), `Operators/`, the immersed-boundary continuity in `ImmersedBoundaries/`.
- **Run Oceananigans at will** (generate references / single-step tendencies):
  `JULIA_DEPOT_PATH=/tmp/ocn_j11_depot julia +1.10.11 --project=/tmp/ocn_j11_gen <script.jl> <args>`
  Oracle deck: `scripts/data/generate_oceananigans_internal_tide_reference.jl` (writes to `<root>/internal_tide/`).
- **Reference root**: `LEGOESM_OCEAN_FIDELITY_OCEANANIGANS_REF=/tmp/ocn_fidelity_ref`
- **Probes** (gitignored `scripts/tmp/`): `_it_ycheck.py` (max|v| + b' y-structure), `_it_tidal_pattern.py` (time-varying M2 b' rms + pattern_corr + figure /tmp/internal_tide_pattern.png).
- **Comparator**: `scripts/validate/ocean_fidelity/compare_oceananigans_internal_tide.py` (env hooks: MOM_ADV, PARTIAL_CELL, IT_H0, IT_K_V, IT_A_V, PGF_SCHEME, RECON_ZETA, ...).

## VERIFIED state — DO NOT re-tread (these are settled)
- **Faithful momentum scheme = FLUX-FORM** (`MOM_ADV=flux_form`, now the comparator default). The oracle uses `momentum_advection = WENO()` = flux-form on the RectilinearGrid (NOT WENOVectorInvariant). Do NOT investigate the vector-invariant vortcor / reconstruct_zeta / Hollingsworth / conditional_flux — that was a WRONG-SCHEME dead end (real but irrelevant).
- **PGF is correct** (smc03 density-Jacobian, machine-zero at rest; PR #573). Not the issue.
- **Mixing ruled out** (K_v=A_v=0 oracle no-closure; spurious growth persists without it). NOTE K_v=0 NaNs the implicit tracer solve at rest — use a tiny K_v for stability OR fix that separately; it's a rest-only artifact.
- **tracer_wall_neumann_fill is ALREADY ON** and does NOT fix the staircase noise.
- Pure z-star (no staircase) = no noise but generates no tide; partial cells = the tide + staircase noise. The bump MUST be partial cells (matches oracle PartialCellBottom).

## PROGRESS LOG (iterations so far — start from here, don't redo)
- **iter 1: w-diagnosis RULED OUT.** Single-step w match (harness: `generate_oceananigans_internal_tide_w_reference.jl` + `compare_oceananigans_internal_tide_w.py`): max|w| matches within 10% (lego 1.97e-3 vs oracle 2.18e-3), legoESM SMOOTHER at the staircase. Both use the SAME min-rule face thickness (`Δrᶠᶜᶜ = min(Δr[i-1],Δr[i])`, Oceananigans `partial_cell_bottom.jl`) + bottom-up integration. NOT the w.
- **iter 2-3: TRACER ADVECTION is the source — ~4× too strong at the staircase.** Real one-step Δb/dt (oracle via `time_step!`): pattern_corr **0.76** but legoESM magnitude ~4× the oracle (3.4e-7 vs 7.8e-8; 4.4× at the bump flank x≈±20km/z=-1739m, Hb=1845m). CONFIRMED real (not an IC artifact — b set at the physical centroid `compute_centroid_depth` gives the same 4×) and SCHEME-INDEPENDENT (weno5 and dst3 both give ~4× → NOT the face reconstruction). w_baro (=state.w averaged) matches the oracle.
- **REFINED TARGET:** the discrepancy is the flux-form `b·∂w/∂z` / continuity-consistency at the staircase — the oracle's `u·∂b/∂x` cancels part of `w·∂b/∂z` so its db/dt (7.8e-8) is LESS than w·N² (2.2e-7); legoESM's does NOT cancel so db/dt (3.4e-7) is MORE than w·N². Next: compare the HALF-LEVEL w_baro (not the cell-center-averaged state.w) and the per-level mass-flux divergence at the staircase vs the oracle; find where legoESM's vertical+horizontal tracer flux fails to telescope for vertically-varying b.

## The blocker (precisely located)
With `MOM_ADV=flux_form`: STABLE, max|v| over bump 0.70 m/s (vs 2.7 vector-invariant),
b' rms 2.5e-4 (vs oracle 2e-5, ~12×), pattern_corr ~0. The b' shows a grid-scale
CHECKERBOARD at the bump's partial-cell staircase flank. The diagnosed `w` has the
RIGHT magnitude (max|w| 2.28e-3 ≈ oracle 2.69e-3) but its 2Δx roughness at the steps
≈ 2.59e-3 (as large as w itself, localized). Source = the min-rule flux divergence
`h_u = min(h_W, h_E)` over the staircase steps in `_diagnose_w_from_flux_div` /
the continuity. Fix = make legoESM's partial-cell continuity / w-diagnosis at the
staircase match Oceananigans' immersed-boundary continuity, WITHOUT smearing.

## Method (each iteration)
1. Read Oceananigans' w-from-continuity + immersed-boundary face/flux handling; find the
   exact difference from legoESM's `_diagnose_w_from_flux_div` + `min_cell_to_uface` min-rule
   (`ocean_pe_latlon_cgrid.py`, `latlon_cgrid_operators.py`).
2. Consider a single-step CONTINUITY/w tendency match: prescribe an identical (u,v,b) on
   the bump in both codes, dump Oceananigans' `w` (and div) via the Julia harness, compare
   to legoESM's `w` cell-by-cell at the staircase to pinpoint the operator discrepancy.
3. Implement the faithful fix as a gated, selectable option (default off = bit-identical);
   verify with `_it_ycheck.py` (max|v| → ~0.2 tidal-ellipse, not 0.70) and the 2Δx w-roughness
   dropping to << max|w|.
4. Run the full `_it_tidal_pattern.py` (MOM_ADV=flux_form): target pattern_corr ≥ 0.6, rms within ~2×.
5. NO REGRESSION: `tests/ocean/unit/test_pgf_tiers.py` + partial-cell suites + `test_pgf_seamount_at_rest.py` stay green.
6. Run the codex/adversarial review on any dycore change; commit incrementally on
   `feat/oceananigans-fidelity-harness`; explicit pathspecs (never `git add .`).

## Acceptance (the completion promise is ONLY true when ALL hold, on a REAL run)
- `compare_oceananigans_internal_tide.py` with `MOM_ADV=flux_form`: time-varying M2 b'
  **pattern_corr ≥ 0.6** vs the oracle AND rms within ~2× of the oracle.
- `_it_ycheck.py`: max|v| over the bump ≤ ~0.3 m/s (the tidal ellipse), 2Δx w-roughness << max|w|.
- No regression in the partial-cell / PGF unit suites.
- The fix is FAITHFUL + gated + adversarially reviewed.

If a faithful fix proves genuinely out of reach, document precisely why (with evidence)
in issue #576 and the scoreboard — but do NOT output a false completion promise.
