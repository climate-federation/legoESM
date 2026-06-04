# Tropical OMIP (Phase A) — Implementation Plan

**Status**: scoping
**Date**: 2026-05-02
**Parent docs**: `omip_1deg_plan.md` (full-OMIP scope), `omip_protocol_review.md`
(Griffies 2016 protocol gap analysis), `realistic_geometry_lat_lon_plan.md`
(bathymetry / partial-cells foundation just merged).

This is **Phase A** of the two-phase OMIP strategy laid out in
`omip_protocol_review.md` §3:
- **Phase A** (this plan): 60°S–60°N forced ocean, no sea ice, 1 cycle. Technical
  shakedown of the full physics stack + diagnostics infrastructure. ~6–8 weeks
  dev + ~4 days GPU.
- **Phase B** (deferred): global with sea ice, 5 cycles, OMIP-2 compliant
  submission. Gated on Phase A success + sea-ice validation against NSIDC /
  PIOMAS. ~3 months additional dev.

## 1. Goal

Run a forced ocean simulation at 1° lat-lon over **60°S–60°N** driven by
JRA55-do v1.4+ corrected forcing for one full OMIP-2 cycle (1958–2018, 61 yr
on a noleap calendar). Deliver:

| Diagnostic | Target / reference |
|---|---|
| AMOC@26.5°N | RAPID 16–22 Sv |
| Drake transport | Meredith 2011: 136.7 ± 6.9 Sv |
| Indonesian Throughflow | Sprintall et al. 2009: ~15 Sv |
| Equatorial Undercurrent | Johnson 2002: ~30 Sv core |
| Northward MHT @ 26.5°N | RAPID + Trenberth-Caron: ~1.2 PW |
| σ₂ MOC by basin | Bench against CORE-II ensemble |
| Mixed layer depth | de Boyer Montégut 2004 climatology |
| Drake meridional T,S structure | Orsi 1995 fronts |

**This is not a protocol-compliant OMIP submission.** It is the technical
shakedown that proves the forced-ocean infrastructure works end-to-end.
Compliance comes in Phase B.

## 2. Why a 60°S–60°N domain de-risks Phase A

The boundary closure removes, in one stroke, the four hardest pieces of full
OMIP:

1. **No polar grid singularity** in the active domain. cos²(lat) collapses to
   0.25 at 60° (still 4× nominal A_h — fine), not to 0 at the pole. No need
   for the zonal Fourier filter, no need to lock a polar cap, no need for
   tripolar.
2. **No sea-ice model** required. Surface T capped at T_freeze inside the
   sponge prevents spurious convection at the closure.
3. **No Bering Strait / Fram / Davis** transport bookkeeping (all north of
   60°N). Reduces the OMIP 16-strait list to 6 tropical-domain straits.
4. **No high-latitude brine rejection / freshwater issues** that would
   require ice. SSS restoring alone is sufficient for closure of the
   tropical+midlatitude salt budget.

The cost: we cannot diagnose AMOC pathway through GIN/Labrador, Arctic
freshwater export, or Antarctic Bottom Water formation. We *can* diagnose
AMOC strength at 26.5°N (well inside domain), Southern Ocean MOC return cell
(captured at 60°S, comfortably south of the typical 45°S closure), and
basin-integrated MHT.

## 3. What we already have (realistic-geometry-full)

From the just-merged branch:

- `LatLonCGridOceanModel` with full physics stack:
  - Primitive equations + split-explicit barotropic (BEBT, implicit_cn solver)
  - Wright nonlinear EOS, z* vertical coordinate
  - KPP boundary layer, Richardson interior mixing
  - GM/Redi (Visbeck adaptive)
  - Plume convection (with the May 2 `expm1` entrainment fix)
  - Biharmonic + Smagorinsky + Leith viscosity
  - PPM / DST-3 / WENO5/7 tracer advection
  - Quadratic bottom drag, shortwave penetration
  - Conservation fixers + freshwater virtual-salt flux
- ETOPO bathymetry loader + partial cells (Adcroft)
- AL81 PV-flux on partial cells (production-grade Coriolis)
- 100-yr Wolfe-Cessi spinup demonstrated stable on real ETOPO
- cos²(lat) A_h scaling + ≥80°N polar cap (not needed for Phase A but in tree)
- Restart I/O (`io/restart.py`)

Bulk flux scaffolding exists (`coupler/bulk_flux.py`,
`ocean/physics/surface_forcing/bulk_formulas.py`) but **needs LY09 audit** —
this is item 1 below.

## 4. Critical path — 8 work items

| # | Item | Effort | Critical-path? |
|---|------|--------|---|
| 1 | Bulk-formula audit (LY09 compliance) | 1–2 d | yes — gates 4 |
| 2 | JRA55-do forcing pipeline | 1 wk | yes — gates 4 |
| 3 | WOA18 init + basin masks + strait paths | 3–4 d | yes — gates 4 |
| 4 | `ForcedOceanDriver` | 1.5 wk | yes |
| 5 | Sponge boundaries at 60°N / 60°S | 2–3 d | yes |
| 6 | SSS restoring | 3 d | yes |
| 7 | Online diagnostics module | 1.5–2 wk | yes — design from day 1 |
| 8 | Output infrastructure | 3 d | yes |

**Total dev**: 6–8 weeks single-developer (some parallelism possible).
**Total compute**: ~85 GPU-hr (12 spin-up + 73 cycle-1 production).

### Item 1 — Bulk-formula audit (LY09 compliance)

**Effort**: 1–2 d. Gates item 4.

OMIP §2.2 mandates Large & Yeager (2009) bulk formulae, not COARE 3.0 or
LY04 (the 2004 version). LY09 hand-tuned drag/heat-transfer coefficients to
balance against CORE-II/JRA55-do forcing. Substituting COARE breaks the
intended energy balance and is a known source of inter-model spread.

**Audit targets**:

- `src/legoesm/coupler/bulk_flux.py`
- `src/legoesm/ocean/physics/surface_forcing/bulk_formulas.py`

**Verify**:

- Neutral 10 m drag coefficient:
  `C_d_n10 = 1e-3 * (2.7/U + 0.142 + U/13.09 − 3.14807e-10·U⁶)`
- Neutral 10 m heat transfer:
  `C_h_n10 = 18.0e-3 * sqrt(C_d_n10)` (stable), `32.7e-3 * sqrt(C_d_n10)` (unstable)
- Neutral 10 m moisture transfer:
  `C_e_n10 = 34.6e-3 * sqrt(C_d_n10)`
- Stability function: Businger 1971 / Large & Pond 1981 (LY09 Table 4)
- Iteration: 5 fixed-point passes (LY09 §3)
- Sea-air specific humidity: 0.98 × q_sat(SST, p_air) (the 0.98 is for salinity
  reduction of saturation vapor pressure)

**If LY04 is what's there**: the formulas are similar in form but use
different constants (`C_d_n10` peak vs. monotone). Need updating to LY09
constants.

**Deliverable**: a short audit memo
(`docs/ocean_experiments/bulk_flux_ly09_audit.md`) with a side-by-side
comparison and a list of any required code changes.

### Item 2 — JRA55-do forcing pipeline

**Effort**: 1 wk. Gates item 4.

`scripts/data/prepare_omip_forcing.py`. Operations:

1. Download/locate JRA55-do v1.4+ "corrected" files (Tsujino et al. 2018, 2020).
   Variables: `uas`, `vas` (3-hourly winds); `tas`, `huss`, `psl`, `prra`,
   `prsn`, `rsds`, `rlds`, `friver` (6-hourly).
2. Conservative regrid 0.5° → 1° (reuse `grids/topography.py` machinery).
3. Pre-process to noleap (drop Feb 29 from leap years). Document in run README.
4. Apply LY09 forcing-correction tables if not already in the file (the v1.4
   "corrected" distribution should already include these — verify).
5. Quick-fix river runoff: redistribute `friver` uniformly within 3 cells of any
   coast in each 1° lat band, weighted by the band's total flux. Conserves the
   ~1.2×10⁶ m³/s global flux without river routing.
6. Write to a single Zarr store with consolidated metadata. Total ~30 GB.

Reuse `training/era5_to_state.py` for the loading patterns (already handles
Zarr + lazy chunked reads with local cache).

**Output**: `data/jra55_do_v14_omip2_1deg_noleap.zarr`

### Item 3 — WOA18 initialization + basin masks + strait paths

**Effort**: 3–4 d. Gates item 4.

`src/legoesm/ocean/init_omip.py`:

1. **`create_omip_state(domain="tropical")`**:
   - Global 1° lat-lon, but mask everything outside 60°S–60°N as land
     (preserves grid topology; the closed boundary is just a coastline).
   - ETOPO bathymetry via `grids/topography.py` (already production).
   - WOA18 0.25° T,S → conservatively interpolate to model 1° (reuse ETOPO
     conservative remap; bilinear is **not** acceptable per OMIP App. A3.2).
   - Zero velocity, zero η.
   - Land mask from bathy (H > 10 m = ocean), then apply 60° latitude mask.
2. **Basin masks**: Atlantic-Arctic, Indian-Pacific, Global. Use the standard
   NCAR/CICE convention (closed at Bering, open at Indo-Pacific throughflow).
   Static `(ny, nx) int8` arrays. Computed once at init, attached to state.
3. **Strait section paths** for the 6 tropical-domain straits: Drake,
   Indonesian Throughflow, Florida-Bahamas, Mozambique, Caribbean, Gibraltar.
   Pre-compute the (i,j) zigzag path along native grid lines per OMIP §C4.
   Static arrays, no AD pain, no recompilation.
4. **`create_omip_physics()`** and **`create_omip_config()`**: as specified in
   `omip_1deg_plan.md` Phase 3, with `barotropic_solver="implicit_cn"` (proven
   in Phase 4(c)).

### Item 4 — `ForcedOceanDriver`

**Effort**: 1.5 wk. Critical path.

`scripts/run_omip_ocean.py` + the driver class (location TBD —
`src/legoesm/ocean/forced_driver.py` if it grows large enough to warrant a
module).

Templated on `scripts/run_amip.py` but reversed (ocean forced by atmosphere
instead of atmosphere forced by SST).

**Per-step loop** (must be JIT-clean per CLAUDE.md):

```python
@jax.jit
def _step(state, forcing_fields_at_t, diagnostics_carry):
    # 1. Compute bulk fluxes from forcing + SST
    surf_forcing = bulk_flux(forcing_fields_at_t, state.T[..., 0], cfg)
    # 2. Apply sponge tendencies + SSS restoring
    surf_forcing = apply_sponge(surf_forcing, state, sponge_cfg)
    surf_forcing = apply_sss_restoring(surf_forcing, state, restoring_cfg)
    # 3. Step the ocean
    new_state = model.step(state, surf_forcing)
    # 4. Accumulate diagnostics
    new_diag = diagnostics.accumulate(diagnostics_carry, new_state, surf_forcing)
    return new_state, new_diag
```

**Critical constraints** (from CLAUDE.md):
- Build the JIT-compiled step **once outside** the time loop. Pass changing
  forcing as explicit arguments, not closure captures.
- Use the non-donating `.raw` variant if AD is ever needed (probably not for
  Phase A, but the pattern should be in place for Phase B's training work).
- `lax.scan` over the inner segment (e.g., 1 day = 24 steps at dt=1h);
  outer Python loop over days for forcing reads + diagnostics flush.

**Forcing interpolation**: linear in time between the 3-hourly wind / 6-hourly
other-fields, computed at the sub-step level. Implementation note: to avoid
recompilation, the interpolation weights are computed in Python and passed as
floats; the forcing arrays themselves are passed as tracers.

### Item 5 — Sponge boundaries at 60°N / 60°S

**Effort**: 2–3 d. Critical path.

In the polar 5° (55°–60°) of each hemisphere:

- **T,S relaxation** to WOA monthly climatology with τ ramped from 30 d at
  55° to 5 d at 60° (linear ramp).
- **u, v = 0** in the last 2 cells (60°–60° row + adjacent).
- **SST cap at T_freeze** (271.35 K, `constants.T_freeze_ocean`) everywhere in
  the sponge to prevent spurious deep convection at the closure (this is the
  "freeze-T cap" hack standing in for sea ice).
- **Salinity** allowed to drift inside the sponge (T,S relaxation handles it),
  but no explicit ice-derived salt flux.

Implement as a new module `src/legoesm/ocean/physics/surface_forcing/sponge.py`
or extend the existing `restoring.py` if structurally cleaner. Must be
differentiable (linear relaxation is AD-clean by construction).

### Item 6 — SSS restoring (interior)

**Effort**: 3 d. Critical path.

OMIP §2.2 mandates SSS damping to monthly climatology. Recommended:

- **Piston velocity**: 5×10⁻⁷ m/s (NEMO/ORCA standard; protocol prefers
  "weak" restoring).
- **Target**: PHC3.0 monthly SSS or WOA18 monthly SSS.
- **Apply globally** in the active 60°S–60°N domain (the sponge handles polar).
- **Diagnose `wfcorr` and `vsfcorr`** separately (Tables K1, K2) so the salt
  budget closes. This is a *correction* flux, not a virtual salt flux.

Likely 70% reuse from `ocean/physics/surface_forcing/restoring.py` — verify.

### Item 7 — Online diagnostics module

**Effort**: 1.5–2 wk. **Must be designed from day 1.**

This is the work item the parent plan most under-estimated. OMIP requires
~150 fields across Tables H–N; we need the subset that is achievable in the
tropical domain without ice. **Retrofitting after the run = re-running.**

`src/legoesm/ocean/diagnostics/omip.py` — `OmipDiagnostics` carry struct
attached to `SegmentCarry` (per CLAUDE.md, this is a cross-cutting NamedTuple
edit — update `pack_carry`, `unpack_carry`, the per-step Python reference
loop, and all existing test files that construct `SegmentCarry`).

**Online accumulators**:

| Accumulator | Bin / dimension | Notes |
|---|---|---|
| σ₂ MOC | 80 σ₂ bins × ny × 3 basins | Bin edges 24.0–28.5 kg/m³ at 0.05 resolution. Each step: select bin from local σ₂; accumulate `v · h · dx`. Monthly mean = `msftmrho`. |
| Depth-space MOC | nz × ny × 3 basins | Standard `msftmyz` accumulation. |
| MLD | (ny, nx) | Levitus σₜ criterion, ΔB_crit = 3×10⁻⁴ m/s². Monthly mean + monthly max + monthly min (`mlotst`, `mlotstmax`, `mlotstmin`). |
| Ideal-age tracer | 3D | One passive tracer; surface boundary `A=0`, interior `∂A/∂t = 1`. Add to advection/diffusion stack. |
| Heat budget partition | (ny, nx, nz) per term | Resolved advection, GM, Redi, KPP, convection, surface flux. Tables L1. |
| Salt budget partition | same | Tables L2. |
| Strait transports | 6 straits × {mass, heat, salt} | Online line-integral over pre-computed (i,j) path. |
| Boundary fluxes | 12 fields | `tauuo, tauvo, hfds, fsitherm=0, sfdsi=0, hfsifrazil=0, hfsnthermds=0, ficeberg=0, wfo, wfonocorr, wfcorr, vsfcorr`. (Ice-related = 0 in Phase A.) |

**Implementation pattern**: each accumulator is a separate function that
takes `(state, forcing, diag_carry)` and returns updated carry. Composed in
the driver. Reset at month boundaries; emit to NetCDF at year boundaries.

### Item 8 — Output infrastructure

**Effort**: 3 d. Critical path.

- **Native-grid monthly NetCDF**: one file per year, `<run_id>_<year>.nc`,
  with all monthly-mean fields. Use xarray + NetCDF4. Single-rank writes
  (fine at 1°).
- **1° spherical regrid** (post-process, offline): conservative regrid native
  → standard 1° spherical. OMIP App. A3.3 mandates this for Priority-1 tracers.
  Reuse the conservative remap from `grids/topography.py`.
- **CMOR variable names** for all named diagnostics (`thetao`, `so`, `uo`,
  `vo`, `zos`, `tauuo`, `tauvo`, `hfds`, `wfo`, `mlotst`, `msftmyz`,
  `msftmrho`, `agessc`, etc.). One name per variable, per the CMOR-3 / CMIP6
  data-request controlled vocabulary.
- **Decadal-mean output** at end of cycle: per OMIP §3.3.

## 5. Compute budget

| Stage | Steps | GPU-hr |
|---|---|---|
| Cold-start ramp (year 1, wind 0→full) | 8760 (dt=1h) | 12 |
| Cycle 1 production (1958–2018, 61 yr noleap) | 534360 | 73 |
| **Total** | | **~85 GPU-hr** |

Storage:
- Forcing Zarr: ~30 GB
- Per-year monthly NetCDF: ~5 GB (12 monthly × ~150 fields × 1° × 50 lev)
- Cycle 1 total: ~310 GB
- Restart files (5 yr cadence): ~1 GB each, ~12 over the cycle = 12 GB

## 6. Risk register (Phase A specific)

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| 1 | **Spurious deep convection at 60°N sponge** | High | T_freeze cap inside the full 5° sponge (not just at the boundary edge). Diagnose Bering/Okhotsk/N. Pacific MLD as red flag. |
| 2 | **Drake transport overshoots (>200 Sv)** | High | Drake is ~3 cells wide at 1°, no bottom form drag. Ramp wind 0→full over year 1. If overshoot persists in cycle 1, add Munk-Palmen form drag in topographically-controlled cells. Document residual. |
| 3 | **AMOC overshoot from naive SSS restoring** | Medium-High | Pin sponge SSS to WOA monthly (already in spec). Weak (5×10⁻⁷ m/s) global restoring elsewhere. Monitor AMOC@26.5N daily during ramp. |
| 4 | **Cycle-1 not equilibrated for NADW lower limb** | Known | This is shakedown, not publication. Document. Phase B's 5-cycle production is the equilibrated answer. |
| 5 | **Diagnostic retrofit cost** | Medium-High | Design item 7 from day 1, before any production run. Sentinel: every diagnostic in the OMIP table that's achievable in tropical domain has an entry in `OmipDiagnostics` *before* item 4 ships. |
| 6 | **JIT recompilation in driver** | Medium | Build segment fn once outside loop; pass forcing as explicit traced args (not closure capture); use `SegmentForcing` pattern from training infrastructure. |
| 7 | **GM/Redi κ tuning at 1°** | Medium | Visbeck adaptive (already in tree). Default κ_GM = 800 m²/s; sensitivity test at ±200 if cycle-1 AMOC is off. |
| 8 | **Forcing I/O bandwidth** | Medium | Pre-stage Zarr locally; lazy chunked reads; profile per-step I/O before launching production. |

## 7. Sequencing (8-week schedule, single developer)

| Week | Work |
|---|---|
| 1 | Items 1 + 2 in parallel (audit while building forcing pipeline) |
| 2 | Items 3 + 4a + 5 in parallel (init, driver skeleton, sponges) |
| 3 | Items 4b + 7a (driver integration; first half of diagnostics: σ₂ MOC, MLD, ideal age) |
| 4 | Items 7b + 6 (remaining diagnostics: budgets, straits, boundary fluxes; SSS restoring) |
| 5 | Item 8 + integration testing + cold-start shakedown (1 mo of forcing) |
| 6 | Cold-start ramp (year 1) + debug |
| 7 | Cycle-1 production launch (continuous run) |
| 8 | Cycle-1 analysis + validation against AMOC/Drake/ITF/MHT/EUC targets |

## 8. Reuse for Phase B

When Phase B (full global with ice) starts, **~80% of Phase A's dev work is
reused unchanged**:

- Item 1 (LY09 audit): reused
- Item 2 (forcing pipeline): reused with extended polar coverage
- Item 3 (init): extended to global; basin masks already include high-lat
- Item 4 (driver): reused; just needs ice coupling
- Item 5 (sponges): **discarded** — replaced by ice + polar grid hardening
- Item 6 (SSS restoring): retuned for under-ice application
- Item 7 (diagnostics): extended with sea-ice diagnostics block (Tables K1–K3
  ice fluxes that are 0 in Phase A become real)
- Item 8 (output): reused

So Phase A is not throwaway work — it's the infrastructure layer Phase B needs
anyway, plus a publishable forced-ocean validation result of its own
("legoESM's first global-scale forced ocean run, validated against JRA55-do
for AMOC, Drake, ITF, MHT").

## 9. Success criteria for Phase A → Phase B gate

Phase A is a success if:

1. Cycle 1 completes without numerical instability for the full 61 yr.
2. AMOC@26.5°N falls in 12–22 Sv (CORE-II ensemble range allowing for
   under-equilibration).
3. Drake transport falls in 100–200 Sv.
4. ITF falls in 10–20 Sv.
5. Northward MHT @ 26.5°N falls in 0.8–1.4 PW.
6. σ₂ MOC shows a clear NADW cell (positive in upper Atlantic, negative below
   ~σ₂=27.5).
7. Equatorial Undercurrent has a coherent core in Pacific and Atlantic.
8. No diagnostic retrofitting required to assess (1)–(7).

Phase B starts when Phase A passes (1)–(8) **and** sea-ice validation against
NSIDC + PIOMAS passes the minimum-viable set in `omip_protocol_review.md` §3.

---

## 9. GPU performance: `lax.scan` with partial cells — RESOLVED

**Status**: fixed (2026-05-03)
**Result**: ~130× speedup (180 s/day → 1.4 s/day on V100)

### Root cause

The scan-block path was never actually exercised with partial cells.
Two bugs combined to create the appearance of a scan-specific divergence:

1. **Missing `block_fn` construction**: `_build_jra55_block_fn()` was never
   called inside `_run_omip_loop()`, so `block_fn` was undefined.  The scan
   path would have raised `NameError` on the first call.

2. **`_use_single_step` masking the bug**: when `--bathymetry` was set,
   `run_omip_single()` injected `jra55_state["_use_single_step"] = True`,
   routing all partial-cell runs to the Python for-loop path.  The
   divergence reported in the original diagnosis was from an earlier
   iteration of the code before the `_use_single_step` fallback was
   added, and was never re-tested after the workaround went in.

### Fix (3 changes)

1. **`_run_omip_loop`**: build `block_fn` via `_build_jra55_block_fn()`
   at the top of the scan-blocks section.
2. **`_build_jra55_block_fn`**: scan body now calls `model._step_impl()`
   instead of `model.step()` to avoid nested JIT boundaries (preventive —
   testing showed both produce identical results, but `_step_impl` is the
   correct pattern per CLAUDE.md's buffer-donation / JIT-reuse guidance).
3. **`LatLonCGridOceanModel`**: extracted `_step_impl()` (no JIT) from
   `step()`, which is now a thin `@jax.jit` wrapper.  `step_impl()` is
   the entry point for any outer-JIT context (scan blocks, training).

### Verified config

Same as the original plan config, now running via scan:

```python
LatLonCGridOceanConfig(
    A_h=2.0e5, A_h_lat_scaling=True,
    B_h=5.0e9,
    bottom_drag_r=2.5e-3, bottom_drag_bbl_thickness=100.0,
    A_v=1e-3, K_v=1e-4,
    barotropic_solver="implicit_cn",
    pgf_scheme="smc03",
    physics=None,
)
```

Grid: 180×360 (1°), 20 levels, H_max=5000, dz_surface=20, dz_deep=500.
ETOPO: H_min=50, 5 smoothing passes, r_factor_max=0.2, polar caps ±80°.
Partial cells via `create_partial_cell_coordinate(z_base, H_bathy)`.

### Performance (V100 GPU)

| Metric | Single-step (old) | Scan-block (new) |
|--------|-------------------|------------------|
| Time per day | ~180 s | ~1.4 s |
| s/step (after JIT) | ~0.6 s | 0.005 s |
| 7-day wall time | ~21 min | ~75 s |
| 1-yr projection | ~18 hr | ~8.5 min |

### Stability test

7-day ETOPO run with JRA55-do RYF forcing, T_ramp=1 day:
- max_speed: 0.015 → 0.032 m/s (day 1→7, no exponential growth)
- SST: 19.82 → 19.77°C (healthy adjustment)
- SSH: −0.001 → −0.002 m (smooth evolution)
- All blocks finite, no blowups.

---

**Next action**: scope item 1 (LY09 bulk-formula audit). Audit memo target:
`docs/ocean_experiments/bulk_flux_ly09_audit.md`.
