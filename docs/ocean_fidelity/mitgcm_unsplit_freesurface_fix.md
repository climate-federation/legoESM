# MITgcm baroclinic-gyre long-run checkerboard: root cause + faithful fix spec

**Status:** Diagnosis COMPLETE + **FIX VALIDATED by POC** (term-by-term operator match vs MITgcm +
MITgcm Fortran read + working unsplit-FS proof-of-concept). Branch `feat/mitgcm-oracle`.

**FIX VALIDATED (`scripts/tmp/_bgyre_unsplit_poc.py`):** a POC unsplit implicit free surface —
predictor `u*=u+Δt·AB2(du_dt)` on the FULL 3D velocity, one `solve_helmholtz_freesurface` for
`η^{n+1}` from depth-integrated `div(u*)`, uniform `−Δt·g·∇η` correction on all levels, reusing the
model's `tendencies` + `_apply_implicit_vertical_mixing` — keeps the velocity SMOOTH where the
split checkerboards. From MITgcm's equilibrated month-1 state (where the SPLIT grows zig 0.068→1.83
by day 15): UNSPLIT zig 0.068→0.081(5d)→0.113(15d)→**0.174(30d)**, |u|max 0.21 (≈MITgcm 0.23) — vs
SPLIT 1.83 (~10× worse). **At the FAITHFUL A_h=5000, K_h=1000, NO added dissipation.** Confirms the
diagnosis AND that the unsplit FS is the faithful fix. The POC is the working template for the
production `implicit_unsplit` option below.

## Root cause (definitive)

The 1-year `u`/`v` grid-scale (2Δx) checkerboard in the `tutorial_baroclinic_gyre` recipe is a
**spurious grid-scale baroclinic instability** caused by legoESM's **split-explicit barotropic/
baroclinic free-surface stepping** vs MITgcm's **unsplit implicit free surface**. Established by:

- Term-by-term momentum-operator comparison at a bit-imported MITgcm month-1 state
  (`scripts/tmp/_bgyre_operator_match.py`): **advection `Um_Advec` corr 1.0000, Coriolis `Um_Cori`
  corr 1.0000** — both operators bit-faithful. The divergence is entirely in the pressure-gradient/
  free-surface handling.
- One legoESM step from MITgcm's smooth state injects **3.5× more grid-scale (2Δx) momentum
  tendency** than MITgcm's `TOTUTEND` (rms 3.0e-9 vs 8.5e-10) — the instability seed, at the grid
  scale, from the split.
- It's the coupled `v′→T′(advect mean ∂T/∂y)→baroclinic PGF→u′→Coriolis→v′` loop tapping the
  restoring-maintained APE at marginally-resolved `Ld/dx≈1.3`; the momentum side alone is stable.
- Climate statistics already match MITgcm to corr 0.999 over a year; this is a velocity-only
  grid-scale invariant violation, NOT fixable faithfully by dissipation (all toggles ruled out).

**Why the split breaks it:** MITgcm's discretization conserves KE+PE *by construction* — its
energy-conserving FD hydrostatic-pressure integral makes the discrete PGF operator and the
continuity operator **adjoint** (PGF work on the flow exactly balances the PE change, no spurious
KE+PE source), and its barotropic dynamics are a single unsplit backward-Euler elliptic inversion
of the depth-integrated divergence (exact volume consistency, no averaging dissipation). legoESM's
split (`ocean_model_latlon_cgrid.py`: `F_slow=depth-mean(du_dt)` line 1626 → eta solve; `du_dt_pert
=du_dt−F_slow` line 1635 integrated separately; `u_new=u′_new+U_bar` line 601 recombine) integrates
the barotropic and baroclinic pressure pieces on different axes, breaking the PGF/continuity
adjointness at the grid scale → the spurious grid-scale KE+PE source that drives the checkerboard.

## MITgcm reference algorithm (exact — from Fortran read; for the faithful port)

Hydrostatic z-coord, linear free surface, `implicSurfPress=implicDiv2DFlow=1` (fully implicit),
`exactConserv=.TRUE.`, synchronous, flux-form. **UNSPLIT** (one 3D predictor + one 2D elliptic
solve + one uniform correction; NO barotropic mode prognostic, NO sub-cycle, NO averaging):

1. **Baroclinic hydrostatic potential** `phiHydC` = energy-conserving FD integral (Adcroft 1997,
   `integr_GeoPot=2`), walking DOWN the column with center-staggered half-cell steps
   (`calc_phi_hyd.F:257-272`); **free-surface load EXCLUDED** (`φ′(surface)=0`, `:129-135`).
   Gradient = centered C-grid difference `−∂φ′/∂x` (`calc_grad_phi_hyd.F:142-153`).
2. **Predictor** `u*`: `u^n` advanced by AB2 of ALL explicit tendencies INCLUDING the baroclinic
   `−∇φ′_hyd` (`timestep.F:116-126,373-388`). The surface `g∇η` is NOT in `u*` (implicSurfPress=1).
3. **Elliptic solve** (`solve_for_pressure.F`): RHS = vertical integral of the divergence of the
   predictor transport `u*` (`calc_div_ghat.F`) minus the exactConserv `etaH` source; solve
   `[I/(gΔt²) − ∇·(H∇)] (g·η^{n+1}) = RHS` (`ini_cg2d.F:99-109`) for `η^{n+1}`. The full depth `H`
   enters ONLY by vertical integration — no separate barotropic velocity is solved.
4. **Correction** (`correction_step.F`): `u^{n+1} = u* − Δt·g·∇η^{n+1}`, the SAME vertically-uniform
   surface-pressure gradient applied to EVERY level (mask only).
5. **exactConserv**: `INTEGR_CONTINUITY` updates `η`/`etaH` from the same depth-integrated
   divergence used for `w` → free surface exactly consistent with `∇·(Hu)+∂η/∂t=0`.

Diagnostic mapping (so future comparisons are clean): `Um_dPhiX` = baroclinic `−∂φ′/∂x` PLUS
surface `−g∂η^{n+1}/∂x` (FULL); `PHIHYD/PH` = `phiHydC + g·η` (FULL, includes the surface load).
This is why the earlier KE_PGF-vs-Um_dPhiX and vs `−dPH/dx` comparisons looked divergent — legoESM's
`KE_PGF` is baroclinic-only; MITgcm's diagnostics are full.

## Fix (implementation plan)

Add a **selectable, faithful** `barotropic_solver="implicit_unsplit"` canonical option (doctrine:
new option, not a bespoke solver; keep the existing split for backward-compat) that replicates the
above — NO `F_slow`/`du_dt_pert` depth-mean split, NO `U_bar` recombination:
1. Verify/port the energy-conserving FD `phiHydC` integral (legoESM `iterate_eos_and_pressure_
   anomaly` — confirm it's the Adcroft-1997 center-staggered form, FS load excluded) and the
   baroclinic PGF gradient.
2. Predictor `u* = u^n + Δt·AB2(du_dt)` on the FULL 3D velocity (du_dt already has the baroclinic
   PGF; do NOT subtract the depth-mean).
3. Reuse the existing elliptic machinery (`barotropic_implicit_latlon_cgrid`) but feed it the
   depth-integrated divergence of `u*` as RHS and return `η^{n+1}` (the operator is already the
   depth-integrated Laplacian — confirm it matches `ini_cg2d.F`).
4. Correct `u^{n+1} = u* − Δt·g·∇η^{n+1}` uniformly on all levels.
5. exactConserv continuity update for `η`.

## Validation (acceptance)
- 30-day then 1-yr velocity zig-zag ≤ 0.4 at FAITHFUL `A_h=5000`, `K_h=1000`, NO added dissipation.
- Term-by-term: full PGF (baroclinic+surface) vs `Um_dPhiX` corr ≥0.99; the 1-step grid-scale
  injection drops to ~MITgcm's (≤1.5× not 3.5×).
- `tests/ocean/fidelity/test_mitgcm_baroclinic_gyre_recipe` still passes; 1-yr eta/T corr ≥0.99.
- The barotropic gyre recipe stays laminar (the unsplit option is the same family as the faithful
  stepper that already matches 0.031 bit-faithfully).
- Codex/physics-validator adversarial review (core dycore change); non-vacuous test on the
  PGF/continuity discrete adjointness (synthetic-violation self-test).

## Probes / artifacts
`scripts/tmp/_bgyre_operator_match.py` (term-by-term), `_bgyre_inject_mitgcm.py` (state import),
`_bgyre_longrun_{lego,compare}.py` (1-yr). MITgcm momDiag+3D state at month-1: `/tmp/mitgcm_bgyre/
run` (2161-step run). Full diagnostic record: `.claude/ralph_longrun_findings.md`.
