# DINO config reachability audit (#1226)

Systematic sweep for the defect class that has bitten #1226 three times:
a recipe card sets a field, but the field never reaches its intended
consumer on the *real* dispatch path — silently running different physics
than the card claims.

## Precedent (three confirmed instances, all fixed on this branch)

| # | Field | Bug pattern | Cost | Fix commit |
|---|---|---|---|---|
| 1 | `gm_redi_slope_n2` (→ `GMRediConfig.slope_n2`) | `getattr(cfg, 'slope_n2', 'adiabatic')` inside `compute_treguier_kappa_gm_nemo_native`'s call site received the 2-field `TreguierConfig` (`_treg`) instead of the parent `GMRediConfig`, so it silently fell back to `'adiabatic'` while the slopes two lines above correctly used `'nemo_bn2'` — an internally inconsistent kappa. | 1.6% aeiu deficit (the whole #1226 GM/Redi residual at the time) | `b9005e528` |
| 2 | `gm_bolus_kappa_face_average` | Same getattr-with-default silent-fallback class; per-face kappa averaging (NEMO `ldftra.F90:716-718`) was not threaded to the bolus-transport builder. | eiv transport relative error median 3.4% → 0.16% once fixed | `294d09e59` / `c0eaaa9b7` (the user-cited hash `c7f86ca9f` does not match this field in the current repo history — see note below) |
| 3 | `eos_depth` | `gm_redi_density_and_jacobian()` hardcoded `eos_depth="insitu"`, ignoring `LatLonCGridOceanConfig.eos_depth="geometric"` selected by the recipe. NEMO's S-EOS is written in `gdept`; the PGF path (`ocean_pe_latlon_cgrid.py`) already threaded this, the GM/Redi path did not. | eiv transport v ratio 0.982 → 0.991 (largest #1226 bar gap at the time) | `db4146440` |

**Note on hash `c7f86ca9f`** cited in the task brief: this repo's history has that hash as the *barotropic_seed_face_depth* inertness-demotion commit, not `gm_bolus_kappa_face_average`. The `gm_bolus_kappa_face_average` fix is `294d09e59`/`c0eaaa9b7`. Treat the task brief's hash as a paraphrase of the bug class, not a literal pointer — verified against `git log --all`.

All three share the same signature: a `getattr(cfg, "X", default)` (or an outright hardcoded literal) where the config actually reaching the call site is either the wrong object or missing the field, and the fallback is *plausible enough* that per-term numerical review didn't catch it — only oracle tendency-matching against NEMO's own dumped intermediates did.

## Method

1. Enumerated every field set by `DINO_RECIPES["nemo_dino_kamm"]` and `["nemo_dino_kamm_mlf"]` (`packages/ocean/legoesm/ocean/experiments/dino.py:895-1294`) and every `DINOConfig` field they touch.
2. Traced the REAL dispatch: `dino_config_for_recipe` → `dino_lat_lon_model_config` (builds `LatLonCGridOceanConfig` + `GMRediConfig`/`TreguierConfig`/`VerticalMixingConfig`/`TKEConfig`) → `LatLonCGridOceanModel` → `outer_integrator="leapfrog"` (only on the `_mlf` card) → `_leapfrog_step` → `_step_impl` called **twice per outer step** (`_ab2_scope_override="advective"` both times: an advective Nnn pass whose full output is kept, and a dissipative Nbb pass whose only surviving output is `diss_incr_bb`, per `ocean_model_latlon_cgrid.py:6847-6910`). The FE `nemo_dino_kamm` card runs the same `_step_impl` once per step directly (`outer_integrator="forward_euler"`).
3. Grepped every `getattr(cfg`/`getattr(config`/`getattr(self.config` site in the GM/Redi, vertical-mixing, and dycore modules the recipe touches, plus every DINOConfig field assignment in the two recipe dicts, then read each hit's actual call site to confirm the object type and whether the field/branch is reachable under this recipe's static config (`vmix_scheme="tke"`, `lateral_tracer_mixing="isoneutral"`, `gm_kappa_scheme="treguier"`, `barotropic_solver="explicit_substep"`, `outer_integrator` per card).
4. Cross-checked the three fixed instances directly against `git show` to confirm they are non-regressed (all use explicit named params or direct attribute access now, no defensive `getattr` on a possibly-wrong object).

## Table (representative fields traced; full field list is the ~90 keys in the two recipe dicts — only fields with a notable reachability property are itemized individually, the rest reach their consumer via a direct, unconditional `cfg.<field>` read with no intervening dispatch and are omitted for brevity)

| Field | Card value | Consumer | Reaches runtime? | Notes |
|---|---|---|---|---|
| `gm_redi_slope_n2` | `"nemo_bn2"` | `gm_redi_latlon_cgrid.py:3265`, `:925` (`GMRediConfig.slope_n2`, explicit param, no getattr-on-wrong-object) | Yes | Bug #1, fixed (`b9005e528`); re-verified not regressed. |
| `gm_bolus_kappa_face_average` | `True` | `gm_redi_latlon_cgrid.py:3453` (`cfg.gm_bolus_kappa_face_average`, direct attribute, `cfg`=`GMRediConfig`) | Yes | Bug #2, fixed; direct attribute access now, no getattr. |
| `eos_depth` | `"geometric"` | `LatLonCGridOceanConfig.from_flat(..., eos_depth=cfg.eos_depth)` (`dino.py:2840`) → `ocean_model_latlon_cgrid.py:3826,3858,3894,7184` (`getattr(self.config, "eos_depth", "insitu")`) → threaded into `gm_redi_density_and_jacobian`, `compute_isoneutral_K33_latlon` | Yes | Bug #3, fixed (`db4146440`). `self.config` genuinely carries the field (from `from_flat`), so the getattr default is inert on this recipe — but it is the SAME defensive-default idiom as bugs #1/#2; a future refactor that constructs `LatLonCGridOceanConfig` without threading `eos_depth` would silently regress to `"insitu"` here with no error. |
| `wind_through_step` + `forcing_annual_cycle` | Both `True` | `dino_lat_lon_surface_forcing_arrays` (annual-mean-only `dino_wind_stress`) + `dino_step_surface_forcing` (no `t_seconds` param at all) vs. `apply_dino_lat_lon_surface_forcing` (does recompute `T_star`/`Q_sr` seasonally via `dino_T_star_seasonal`/`dino_Q_sr_seasonal`, and **fail-loud raises** if `t_seconds` is missing when `forcing_annual_cycle=True`) | **NO for wind; YES for T\*/Q_sr** | **Genuine hit (new).** See ranked finding #1 below. |
| `convection_two_level_trigger` | `True` | `k_profiles.py:970` (`getattr(cfg, "two_level_trigger", False) and before_tracers is not None`) fed by `_enhanced_diffusion_K(..., before_tracers=n2_tracers)` at `k_profiles.py:299`, where `n2_tracers = self._n2_before_advection_tracers(state)` | Yes | Initially looked like a dead/silent-noop pattern (guarded by `before_tracers is not None`, and the *other* call site `convection/integration.py:142` never passes `before_tracers` at all — dead on that path). But `nemo_dino_kamm` runs through the *implicit* K-profiles path (`vmix_scheme="tke"` ⇒ `implicit_vertical_mixing`), which DOES thread `before_tracers=n2_tracers`, and `n2_tracers` is non-None because `n2_before_advection=True` is also set on the card and `vmix.scheme=="tke"`. Confirmed reachable — not a bug, but fragile (the explicit-mixing path silently no-ops the same flag; a config that flips `implicit_vertical_mixing=False` on this recipe would regress silently). |
| `n2_before_advection` | `True` | `_n2_before_advection_tracers()` (`ocean_model_latlon_cgrid.py:4490-4519`) — gated on `vmix.scheme=="tke"` AND `vmix.tke.n2_before_advection`, raises if `n2_mode` isn't `adiabatic`/`nemo_bn2` | Yes | Feeds BOTH TKE bn2 and EVD's two-level trigger from the same entry-state T/S snapshot, as documented. Verified wired for both. |
| `barotropic_auto_cmax` | not set (stays default `0.0`) | `_dino_barotropic_substeps()` (`dino.py:2505`: `if cfg.barotropic_auto_cmax <= 0.0: return cfg.n_barotropic_substeps`) | N/A — correctly dead by design | Card hardcodes `n_barotropic_substeps=23` directly (comment cites NEMO's own computed `nn_e=23` for this exact grid/dt) rather than re-deriving via the auto formula. Not a bug — a deliberate precomputed-constant choice, documented. |
| `redi_S_max` | `0.01` | `GMRediConfig.S_max=cfg.redi_S_max` (`dino.py:2615,2665`) | Yes | Card does set this (`"redi_S_max": 0.01` at `dino.py:1073`) — initially suspected missing during the audit, confirmed present. |
| `tke_kappaM_max` | `float("inf")` | `_dino_vertical_mixing_config` (`dino.py:2406-2407`): `kappaM_max=(cfg.K_conv if cfg.tke_kappaM_max is None else cfg.tke_kappaM_max)` | Yes | `None`-sentinel pattern, correctly threaded (`inf` overrides the `K_conv` default ceiling). |
| `barotropic_coriolis="een_metric"` | set on FE card, inherited by MLF | `ocean_model_latlon_cgrid.py:2081` (membership + `raise` on unknown), `:3261` (`metric_complete=(_bt_cor_split=="een_metric")`) | Yes | Dispatch raises on unrecognized values; confirmed not silently collapsed to `"een"`. |
| `treguier_aei0` | `1500.0` | `TreguierConfig(aei0=cfg.treguier_aei0)` (`dino.py:2655,2681`) | Yes | Field-name mismatch (`treguier_aei0` → `aei0`) is intentional and correctly mapped, not a typo-drift. |
| Every `getattr(self.config, "X", default)` site enumerated in `ocean_model_latlon_cgrid.py` (~90 sites) | various | `self.config` is always the model's own fully-populated `LatLonCGridOceanConfig` (built via `from_flat` from the SAME `DINOConfig`) | Yes | These are NOT instances of the bug pattern — the getattr default only matters for OLDER configs lacking the field; on this recipe's dispatch `self.config` genuinely carries every field checked. Distinguished from bugs #1-3 by object identity: the bug pattern is getattr on the WRONG or a PARTIAL sub-config, not defensive defaults on the correctly-threaded top-level config. |

## Ranked findings (genuine hits only)

**1 genuine new hit found** (plus reconfirmation that the three known-fixed instances remain fixed with no sibling regressions in the areas traced: GM/Redi kappa/slope/eos_depth threading, TKE closure config threading, barotropic scheme dispatch, convection two-level trigger).

### 1. `forcing_annual_cycle=True` + `wind_through_step=True`: wind stress never varies seasonally, only T*/Q_sr do (MEDIUM-HIGH impact, currently latent — driver-dependent)

- **Field**: `DINOConfig.forcing_annual_cycle` (card value `True`, `dino.py:1170`) and `DINOConfig.wind_through_step` (card value `True`, `dino.py:1171`).
- **What the card intends**: NEMO's `ln_ann_cyc=T` — a full seasonal cycle for every forcing component (`namusr_def`).
- **What actually happens**: `apply_dino_lat_lon_surface_forcing` (`dino.py:3264-3306`) DOES seasonally recompute `T_star_2d`/`Q_sr_2d` via `dino_T_star_seasonal`/`dino_Q_sr_seasonal` when `forcing_annual_cycle=True`, and even fail-loud `raise`s if the caller omits `t_seconds` (`dino.py:3296-3300`) — a deliberate anti-silent-fallback guard. But **wind stress has no seasonal implementation at all**: `dino_wind_stress` (`dino.py:1526`) is annual-mean-only, there is no `dino_wind_stress_seasonal` counterpart anywhere in the module, `dino_lat_lon_surface_forcing_arrays` (`dino.py:3204-3238`) builds `tau_u_cell_2d`/`taum_2d`/`tau_u_face` once from the annual mean, and `dino_step_surface_forcing(forcing)` (`dino.py:3241-3261`, the function that builds the `OceanSurfaceForcing` routed through `model.step` when `wind_through_step=True`) takes **no time argument whatsoever**.
- **Consequence**: on the canonical 5-year Y5 driver (`scripts/validate/ocean_fidelity/dino_1226/kamm_run5y_v3.py:46,50`), `sf = dino_step_surface_forcing(forcing)` is built ONCE outside the JIT'd step closure (comment: `# sf constant (annual tau)`) and reused for all 5 years/57,600 steps. The momentum forcing AND the TKE `taum` surface-BC channel (which under `tke_surface_bc="nemo_dirichlet"` — also set by this card — directly sets `en(1)=MAX(rn_emin0, rn_ebb·taum/rho0)`) both silently stay at the annual mean, even though the card explicitly selects `taum_westerly_boost` and the full seasonal machinery for everything else.
- **Physics impact**: NEMO's actual wind forcing has a real seasonal cycle (the westerlies/trades shift with the ITCZ/storm-track seasonal migration); an annual-mean wind under a card that claims `ln_ann_cyc=T` will under-represent seasonal Ekman transport variability and seasonal TKE surface-injection variability — this is a genuine, silent, direction-known deficiency (wind seasonality is strictly REMOVED relative to what the card/NEMO does), magnitude not cheaply estimable without a seasonal-wind implementation to diff against (no faithful wind function exists yet to compare).
- **Scope caveat**: this is a driver/harness-adjacent gap as much as a model gap — `apply_dino_lat_lon_surface_forcing`'s per-step call already threads `t_seconds` for T*/Q_sr; the missing piece is (a) a `dino_wind_stress_seasonal` function (none currently exists — this is a genuinely unimplemented DINO forcing component, not a wiring bug in the sense of bugs #1-3) and (b) rebuilding `sf` per-step (or per-N-steps) with that seasonal wind inside the driver loop instead of once outside the JIT. Ranked below the three precedent bugs in confidence-of-fix-size because unlike #1-3 (one-line/one-parameter threading fixes), this needs a new seasonal wind-stress formula (paper eq. 7 gives no explicit seasonal form; would need sourcing from NEMO's `usrdef_sbc.F90` cubic-Hermite knot table if it has one, or the paper's SI).
- **Not flagged as fully "silent"**: the analogous T*/Q_sr path has an explicit fail-loud guard proving the author was alert to exactly this staleness risk for those two components; the wind axis appears to have been out of scope for that guard rather than an oversight in the same commit, but the net effect on `nemo_dino_kamm(_mlf)` cards, which BOTH set `forcing_annual_cycle=True`, is the same class of silent-partial-implementation the user is hunting for.

### No other genuine hits found in the areas traced

Everything else enumerated in the table above — the entire GM/Redi kappa/slope/eos_depth chain (bugs #1-3's neighborhood), the TKE closure config (12+ fields: `tke_prognostic`, `tke_mxl_choice`, `tke_surface_bc`, `tke_dissipation`, `tke_kappa_convention`, `tke_alpha`, `tke_surface_bc_level`, `vmix_background_mode`, `tke_buoyancy_sink`, `tke_mxl_min_m`/`tke_mxl0_min_m`, `tke_bottom_bc`, `tke_kappaM_max`, `tke_shear_production`, `tke_n2_time_level`, `tke_prandtl_ri`), the barotropic scheme block (`barotropic_face_depth`, `barotropic_seed_face_depth`, `barotropic_diffusion_alpha`, `barotropic_time_filter`, `barotropic_coriolis`, `barotropic_coriolis_split`, `barotropic_forcing_centred`, `barotropic_een_seed`, `n_barotropic_substeps`), the momentum/Coriolis/integrator identity block (`vorticity_scheme`, `een_q_boundary`, `een_e3f_scheme`, `coriolis_scheme`, `outer_integrator`, `asselin_gamma`, `tracer_combine`), the dynzdf composition (`zdf_drag_in_matrix`, `zdf_baroclinic_only`, `barotropic_drag_substep`), the bottom drag (`bottom_drag_scheme`), and `pgf_scheme`/`pgf_quadrature` — were each traced to a direct, unconditional read (`cfg.<field>` or an explicit named parameter) on an object confirmed to be the correctly-populated config for this recipe's actual dispatch path (leapfrog ×2 `_step_impl` calls with `_ab2_scope_override="advective"` for the MLF card; single forward-Euler `_step_impl` call for the FE card). No wrong-object `getattr` fallbacks, no hardcoded literals overriding a card selection, and no dead-code-path consumption were found among these.

## Commit history verification

```
$ git show --stat b9005e528   # slope_n2 fix
$ git show --stat 294d09e59   # gm_bolus_kappa_face_average (first plumbed)
$ git show --stat c0eaaa9b7   # gm_bolus_kappa_face_average (card default flip) + Shapiro seam fix
$ git show --stat db4146440   # eos_depth fix
```
All four confirmed present in `git log --all` on this repository; diffs read in full as part of this audit.


## CORRECTION (verified 2026-07-27, after the audit was written)

**The single reported hit — `forcing_annual_cycle` not reaching wind stress — is
REFUTED. It is not a defect.**

NEMO's DINO wind is ALSO annual-mean. `cfgs/DINO/MY_SRC/usrdef_sbc.F90:221`:

    utau(ji,jj) = znl_cbc( znds_wnd_phi, znds_wnd_val, gphiu(ji,jj) )

a pure function of latitude over STATIC knot values (`znds_wnd_val`, set once at
:156). `compute_day_of_year`'s seasonal cosines (`zcos_sais1/zcos_sais2`) feed
only `ztstar` and `zqsr_dayMean` — never `utau`/`vtau`/`taum`. So legoESM
applying the annual cycle to T*/Q_sr while leaving the wind at its annual mean
reproduces NEMO exactly, and the absence of a `dino_wind_stress_seasonal` is
correct, not a gap.

**Audit result therefore: ZERO genuine reachability hits.** The three known
instances (`slope_n2` b9005e528, `gm_bolus_kappa_face_average` 294d09e59 +
c0eaaa9b7, `eos_depth` db4146440) were the complete set; no fourth was hiding in
the GM/Redi, TKE, barotropic or Coriolis blocks.

Method note: the audit inferred "NEMO has a seasonal cycle, lego's wind does
not" from the card's `ln_ann_cyc=T` comment without checking whether NEMO's WIND
consumes it. Rule 0 — read the oracle's source for the specific quantity, not
the flag that nominally governs it.
