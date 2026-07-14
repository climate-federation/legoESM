# Multi-layer land + prognostic carbon — equilibrium realism & allocation audit

Driver: `scripts/validate/land_carbon_equilibrium.py`
(SLURM wrapper `scripts/cluster/land_carbon/run_equilibrium.sbatch`).

Spins the full coupled multi-layer land model (soil thermal + Richards
hydrology + snow + surface energy) with the DifferLand 6-pool prognostic
carbon cycle (Farquhar–Ball/Berry photosynthesis) to a repeating-annual-climate
equilibrium at climate pixels spanning tropical rainforest → arctic tundra, and
audits allocation closure, per-pool input/output balance, NEE→0, and each pool
/ flux against published biome ranges. Per-step carbon diagnostics are
reconstructed from the model's exact end-of-step surface temperature and
`root_zone_beta_soil`, so the diagnostic GPP/NPP/allocation breakdown matches
the coupled model's carbon step (`diag.nee == response.co2_flux`).

## What the audit found (before → after)

The **carbon-allocation arithmetic was never the problem** — allocation closes
to machine precision (Σ allocation == max(NPP,0)) and the column conserves
carbon exactly (Σ ΔC_pools == −NEE·dt) in every regime, verified by the
existing conservation tests. The realism failures were in photosynthesis
scaling, the sub-daily respiration/deficit handling, PFT-blind allocation, and
woody-debris decomposition:

### 1. Big-leaf GPP under-scaling (photosynthesis) — FIXED
`farquhar_photosynthesis` drove the light-limited rate `Wj` with the
canopy-absorbed `APAR = fAPAR·PAR` but left the Rubisco/electron-transport
capacities `Vc_max`/`J_max`/`Rd` at **leaf** level, so a dense canopy (LAI≫1)
was capped at a single leaf's assimilation. Tropical GPP came out ~5 gC/m²/day
(≈40 % low). Fix: multiply the capacity terms by the big-leaf canopy integral
`L_c = (1−e^{−k·LAI})/k` (Sellers 1992; Bonan 2011). Tropical GPP → ~9–10
gC/m²/day. `L_c→LAI→0` for a leafless column, so it degrades correctly.

> **Update (Phase 3 canonical-FvCB adoption):** the C3-only
> `farquhar_photosynthesis` described here has since been retired. The big-leaf
> coupled solver now delegates to the canonical FvCB kernels
> (`canopy.photosynthesis.c3_assimilation` / `c4_assimilation`), and the
> canopy-integral scaling `L_c` is preserved by folding it (with the soil-water
> stress) into an effective `Vcmax25` passed to those kernels — see
> `land/stomata.py::solve_coupled_farquhar_ci`.
>
> **Param migration:** the big-leaf FvCB kinetics no longer live in
> `StomataConfig`. The retired tunables `J_max25`, `Rd25`, `alpha_q`, `theta_j`
> (and the Bernacchi constants `Kc25`/`Ko25`/`Gamma_star25`/`O2_conc`/`Ha_*`/
> `Hd_J`/`S_J`/`co_limitation_eps`) are now the canonical module-level constants
> in `canopy/photosynthesis.py`. Any externally-saved tuned-params JSON keyed on
> `land.stomata.J_max25` (etc.) will no longer resolve — the surviving big-leaf
> photosynthesis tunable is `land.stomata.Vc_max25` (Jmax/Rd derive from it via
> the fixed canonical ratios). `Vcmax25` remains the trainable capacity knob.

### 2. Sub-daily maintenance-respiration death-spiral — FIXED
DifferLand is a daily-timestep DALEC model run here at the land model's hourly
`dt`. Every night GPP=0 while the biomass-proportional maintenance respiration
stays on, so NPP<0 and the deficit cascade fires to pay respiration from
stored carbon. The old cascade drew **foliage second** (labile→foliage→…):
once the labile reserve emptied (which the seasonal leaf-fall guarantees),
nightly maintenance ate the leaves → LAI→0 → GPP→0 → the whole column died.
**All seven pixels collapsed to a dead biosphere at year 100** with canopy
scaling alone. Fix: reorder the cascade to **labile → wood → root → foliage**
— maintenance is paid from the mobile reserve then the largest storage pool
(wood, which also net-accumulates because its allocation share exceeds the
nightly draw whenever daily NPP>0); functional roots and the photosynthetic
foliage are protected, touched only in genuine starvation. Conservation is
untouched (the total drawn = deficit − unmet, independent of draw order).

### 3. PFT-blind allocation grew wood on grasslands — FIXED
Allocation fractions were globally uniform, so herbaceous PFTs (grass, crop,
tundra) allocated ~57 % of NPP to a **woody pool** and built a multi-kgC
phantom tree. Fix: a static per-PFT `woody` flag (feature-gate, not a traced
branch); when `False` the structural fraction that would form wood is invested
belowground (roots) instead. Allocation still closes exactly. Exposed as
`--carbon-woody/--no-carbon-woody` on `run_lmip`.

### 4. Wood humified 100 % into millennial SOM — FIXED
Wood turnover was routed entirely into the stable SOM pool with **zero
respiration**, so with the ~550-yr SOM turnover the implied soil-carbon stock
was ~10–20× too large. Fix: split wood turnover like the litter pathway — a
fraction `cwd_humification_eff` (~0.3, Harmon et al. 1986) humifies to SOM, the
rest respires as coarse-woody-debris heterotrophic respiration (added to NEE;
conserves exactly). Exposed as `--cwd-humification-eff` on `run_lmip`.

### 5. SOM turnover too slow + temperature-insensitive — FIXED
Even after the CWD split, the single soil-C pool turned over too slowly
(`tor_som` ~550 yr) with a weak, shared temperature sensitivity, so at true
equilibrium the soil-carbon stock was far too large (tropical ~46 kgC/m² vs
observed ~10–15). Two coupled fixes:
- **Faster bulk turnover**: `tor_som` 5e-6 → 4e-5 /day (~68-yr reference MRT →
  ~17 yr in warm tropical soil via the Q10 below), matching the fast
  bulk (active+slow) turnover of warm-wet topsoil. A single pool does not
  resolve the passive millennial fraction (see below).
- **Decoupled soil Q10**: a heterotrophic-decomposition temperature sensitivity
  `Q10_het_exp` (Q10≈2.5), separate from the autotrophic `Q10_exp`, so **warm
  tropical soils turn SOM over fast** (low equilibrium SOM despite high
  productivity) while **cold high-latitude soils retain carbon** — the observed
  SOC gradient direction, which the old weak shared Q10≈1.5 could not produce.
Exposed as `--carbon-q10-het` on `run_lmip`.

### 6. Soil-carbon spin-up too short — FIXED (semi-analytic)
A soil-C pool with ~68–270-yr turnover needs ~1500–3000 model years to
equilibrate by brute-force integration; 100 yr only shows the transient (the
100-yr "SOM" was ~2× below its true steady state). Standard practice avoids the
millennia via a **semi-analytic spin-up** (Xia et al. 2012, GMD; cf.
accelerated decomposition, Thornton & Rosenbloom 2005; Koven et al. 2013):
after a transient that stationarises the fast pools + wood, the LINEAR
slow-pool steady state is solved analytically (`dC/dt = I − k·C` ⇒ `C_eq = I/k
= C_spinup·(I_annual/loss_annual)`) and the pools are reset to it, then a short
verification segment confirms the drift → 0. Implemented as the reusable
`legoesm.land.carbon.spinup.analytic_slow_pool_equilibrium`, wired into
`run_lmip` (`--carbon-spinup semi_analytic`) and reused by the validator.

### Residual / recommended (not code bugs)
- **Multi-pool SOM**: at true (semi-analytic) equilibrium a SINGLE soil-C pool
  cannot reproduce the full observed SOC pattern — fast-turnover in warm
  productive forests AND a large cold/waterlogged high-latitude reservoir. The
  fixes bring the productive forests to realistic SOC, but grassland and
  tundra/boreal SOC are then underestimated (they need frozen/anaerobic
  preservation the single pool lacks). The right follow-up is a multi-pool
  (active/slow/passive) SOM with moisture + freeze decomposition controls
  (CENTURY/CLM-style); the semi-analytic spin-up already generalises to the
  multi-pool matrix solution (Xia et al. 2012).
- The DALEC Gaussian phenology (`Bday`/`Fday`) is applied uniformly; tropical
  evergreen PFTs should use near-continuous leaf turnover rather than a
  temperate leaf-fall. Per-PFT phenology is a follow-up.

## Supporting infrastructure added
- `land.carbon.CarbonDiagnostics` + `return_diagnostics` on `step_carbon[_differland]`:
  the exact GPP/NPP/Ra/Rh/allocation/turnover breakdown, single source of truth.
- `land.carbon.spinup` (`SlowPoolFluxes`, `analytic_slow_pool_equilibrium`): the
  semi-analytic soil-C equilibrium, used by both `run_lmip` and the validator.
- `land.carbon_diagnostics.reconstruct_carbon_diagnostics`: rebuilds the carbon
  flux breakdown from an end-of-step land state (shared by the spin-up + validator).
- `multilayer_land.root_zone_beta_soil`: factored the (previously duplicated)
  root-zone moisture-stress helper; reused by the model (both time levels).
- `land.lmip_forcing.make_synthetic_lmip_forcing`: promoted the LMIP synthetic
  forcing out of the driver so the validator and `run_lmip` share one generator.

## Blast radius
The canopy-scaling fix raises GPP (and, via Ball-Berry, stomatal conductance →
ET) for **every** stomata-enabled land configuration (AMIP, coupled), not just
this validator. This is a correctness fix (the big-leaf was under-scaled), but
downstream runs will show higher GPP/ET and should be re-baselined.

## Per-region initial conditions

ICs must be regionally realistic (a tundra must not start with a tropical
forest's wood + SOM). `build_pixel_config` seeds each pool via
`_biome_carbon_init`: the **slow** pools that retain their IC over the spin-up
carry the regional signal — SOM from the biome soil-carbon range, wood from its
biomass range — while the **fast** maintenance-bearing pools (foliage at the
biome's mean LAI; root/labile modest) are seeded conservatively so the stand
grows into steady state. Seeding a maintenance-bearing pool *above* equilibrium
(e.g. a uniform 10 kgC/m² wood, or a savanna tree's root mass on a grass pixel)
makes maintenance respiration exceed a climate-suppressed GPP and sinks the
column into the death-spiral's dead attractor.

## Equilibrium scorecard (semi-analytic spin-up, per-region ICs)

200-yr transient + analytic slow-pool solve + 40-yr verification (`drift` →0
confirms the equilibrium). Hard physical-consistency checks (allocation closes,
`NPP ≤ GPP`, herbaceous carry no wood, pools ≥ 0): **0 failures across all 7
biomes** — the carbon allocation is bug-free to machine precision
(`alloc_resid ≲ 1e-14`).

| pixel | GPP | NPP | LAImax | biomass kgC | SOC kgC | drift %/yr | status |
|---|---|---|---|---|---|---|---|
| tropical_rainforest | 3582 | 1019 | 9.1 | 14.5 | 9.5 | −1.2 | alive |
| tropical_savanna | 366 | 85 | 1.0 | 0.2 | 0.4 | 0.0 | alive‡ |
| temperate_deciduous | 1347 | 580 | 3.1 | 9.4 | 21.3 | −0.2 | alive, in-range |
| temperate_grassland | 322 | 95 | 1.1 | 0.2 | 1.3 | 0.0 | alive‡ |
| boreal_forest | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | collapsed† |
| semiarid_shrubland | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | collapsed† |
| arctic_tundra | 177 | 73 | 0.6 | 0.2 | 4.3 | 0.0 | alive‡ |

GPP/NPP [gC/m²/yr], biomass/SOC [kgC/m²]. **Forest SOC is now realistic**
(tropical 9.5, temperate 21.3 kgC — vs the old single-pool 46 / 88 at true
equilibrium). `‡` grassland/savanna/tundra SOC is UNDER-estimated by the single
pool (needs the multi-pool + freeze/anaerobic controls in Residual §; the
grasses are also under-productive under the idealized C3/C4-as-C3 forcing). `†`
boreal/semiarid: annual GPP < annual R_maint (see Residual §) — NOT an
allocation defect; the column stays carbon-conserving and non-negative.

Artifacts: `results/land_carbon_equilibrium/{scorecard.json,report.md,
pool_trajectories.png,gpp_npp.png,allocation.png}`.

## Global initial-condition map (Stage A)

The per-region audit above equilibrates a handful of hand-placed pixels. Stage A
generalizes it to a **global** near-equilibrium IC map, so a coupled run starts
every land cell from its own spun-up carbon pools instead of an unphysical
uniform cold start. Design spec + plan:
`docs/superpowers/specs/2026-07-07-global-carbon-ic-map-design.md`,
`docs/superpowers/plans/2026-07-07-global-carbon-ic-map.md`.

**Pipeline** (all in `legoesm.land.carbon`, resolution-agnostic — every core
function takes `(ncell, …)`):

1. `climate_features.reduce_climatology_to_features` reduces a monthly
   climatology to five per-cell features `(mat_k, map_yr, t_seasonal_amp_k,
   aridity, sw_mean_w)` (aridity = MAP/PET, Priestley–Taylor PET).
2. `global_init.build_archetypes` clusters the occupied **PFT × climate** space
   with a per-PFT, seeded k-means over standardized features → an
   `ArchetypeTable` of O(hundreds) representative (PFT, climate, soil)
   archetypes + each cell's cover-weighted archetype membership.
3. `global_init.equilibrate_archetypes` spins up only those archetypes with the
   shared **semi-analytic** driver (`spinup.run_semi_analytic_spinup`, Xia 2012 —
   the same slow-pool solve the per-region scorecard uses), batched by
   `(is_woody, soil_class)` group with per-archetype physiology threaded through
   `LandSurfaceParams` (verified per-column, not collapsed).
4. `global_init.map_to_grid` cover-weight-mixes the archetype equilibria onto
   every cell: `pools[c] = Σ_p w[c,p]·eq[id[c,p]]` (bare fraction holds less
   carbon; no renormalization).

**Driver:** `scripts/data/build_global_carbon_ic.py` loads real CLM5 PFT cover +
HWSD/CLM soil texture (17-PFT `CLM5_PFT_NAMES` axis reconstructed from
`natpft=15 + cft=2`) + a climatology, runs the pipeline, and writes
`global_carbon_ic.npz` (per-pool `(ncell,)` + geometry) and `archetypes.npz`.

**Validator:** `scripts/validate/global_carbon_ic_map.py` re-integrates each
archetype from the mapped IC vs. a cold start on the *stored* soil column. This
is **indicative archetype-level QC, not a per-cell proof**: `map_to_grid` is
linear in the IC pools, but re-integrating a *mixed* cell is nonlinear
(GPP/LAI/stomata/respiration depend nonlinearly on the summed foliar carbon and
a shared soil column), so small per-archetype drift is a *necessary condition*
and strong signal that each archetype equilibrium is stationary — **not** a
proof that every mixed grid cell starts at equilibrium. Rigorous per-cell
validation (re-integrating a sample of *actual* mixed grid cells) is a
documented follow-up. On the real CLM5 cover (13 824 cells, 17 PFTs, 51
archetypes) the mapped IC drifts **0.76 %/yr vs. 4.51 %/yr cold** — every
archetype starts ~6× closer to equilibrium than a cold start.

**Known Stage-A approximations.** (1) *Sub-`w_min` cover is dropped.* PFT
fractions below `--w-min` cover in a cell get no archetype (→ zero carbon), and
many small fractions can sum to material area; the driver prints the mean/max
dropped vegetated land-cover fraction so the omitted area is visible (F3). The
follow-up maps each trace PFT to its nearest same-PFT archetype instead of
dropping it. (2) *Bare ground (PFT 0) is excluded outright* — it is inert
(`Vc_max25 = 0`, no carbon), so it neither clusters nor equilibrates. (3)
*Archetype climate means are cover-weighted* by each PFT's cover in its member
cells, so an archetype reflects where the PFT dominates; fully cover-weighted
k-means *assignment* is a follow-up. (4) The drift validator is indicative
archetype-level QC, not a per-cell proof (see Validator above).

**Demonstration caveat.** `--climate-from-latitude` runs the pipeline on real
cover with a **zonal** (latitude-only) climatology reusing the idealized LMIP
forcing — a labeled demonstration that collapses longitudinal climate (deserts
and rainforests at one latitude share a climate). The science-grade map needs a
real 2-D monthly climatology (ERA5/GSWP3 assembled to the target grid via the
existing `--climatology <nc>` path); assembling that forcing is the documented
next data-prep step. Stage B (global multi-observation parameter training,
warm-started from this IC map) gets its own spec.

Artifacts (gitignored): `results/global_carbon_ic*/`.
