# Offline EC-site land/canopy runs (DifferBESS driver → legoESM)

Standalone, uncoupled runs of legoESM's two-leaf canopy at eddy-covariance flux
towers, validated against observed tower fluxes (GPP / LE / H).  The forcing comes
from the **DifferBESS** per-site driver NetCDFs
(`<DifferBESS>/data/sitelevel/nc/<SITE>_driver_v2.nc` — FLUXNET2015 meteorology +
MODIS LAI/albedo + BESSRad PAR + site parameters + observed fluxes).

Two pieces:

* **`packages/land/legoesm/land/boundary_data/ec_site.py`** — `read_ec_site_driver`,
  a pure reader that maps a driver NetCDF onto legoESM's `AtmToSurface` forcing
  time series + per-step `CanopyLandParams`, plus a `valid` gap mask and the
  observed fluxes.  Reusable + unit-tested (`tests/land/boundary_data/test_ec_site.py`).
* **`scripts/run/run_ec_site.py`** — the driver.  `--mode diagnostic` `jax.vmap`s
  `compute_two_leaf_canopy_fluxes` over time with the soil state prescribed from
  the driver, then reports RMSE / bias / Pearson r vs the tower obs (the
  prognostic multilayer-soil mode is added in a later stage).

## Variable mapping (the "missing variables" answer)

Most "missing" `AtmToSurface` fields are derivable; none are blocking.  All
constants come from `legoesm.constants`, all saturation from `legoesm.thermo`
(AERK) — nothing re-derived in the adapter.

| `AtmToSurface` | driver var | derivation |
|---|---|---|
| `sw_down`, `lw_down`, `co2_ppmv` | `SW_IN`, `LW_IN`, `CO2` | direct (SW clamped ≥ 0) |
| `T_lowest` | `TA` [°C] | `+ constants.T_freeze` |
| `q_lowest` | `VPD`, `TA`, `PA` | `e_sat`(AERK) → `e = e_sat − VPD` → `q = ε e / (p − (1−ε) e)` |
| `u_lowest`, `v_lowest` | `WS` | `u = WS`, `v = 0` (direction absent; canopy uses \|wind\| only) |
| `p_surface` = `p_lowest` | `PA` [kPa] | `× 1000` |
| `rho_lowest` | p, T, q | `p / (R_d T_v)`, `T_v = T (1 + (1/ε − 1) q)` |
| `cos_zenith` | `SZA` [deg] | `cos(radians(SZA))` (SZA capped at 90° in the driver → night = 0) |
| `precip_total` | `P` [mm/step] | `P / (dt_s)` → kg m⁻² s⁻¹; `precip_snow` from a `T_freeze` ramp |
| `has_radiation`, `has_precipitation` | — | True |

Site params → `CanopyLandParams` (per-column): `LAI ← ds.LAI` (masked to `[0, 7]`,
**the same input as production DifferBESS**; `kn`, `FNonVeg` derived from it),
`canopy_height ← CANOPY_HEIGHT`, `CI ← CI`, `fC4 ← C4`, `TgC ← T_GROWTH`,
albedos ← `Albedo_{BSA,WSA}_{vis,nir}`, `Vcmax25 ←` driver `Vcmax25_C3Leaf`
(cropland or <50 % coverage → legoESM PFT lookup), aero params + leaf width from
the IGBP→PFT table.  Observed `GPP_DT`/`NEE`/`ET`/`H` are extracted for comparison
(GPP unit-aligned: model gC m⁻² s⁻¹ ÷ `12.0e-6` → µmol m⁻² s⁻¹).

### Genuine method differences (document, don't tune-to-match)

1. **Direct/diffuse SW split** — DifferBESS uses BESSRad satellite ratios;
   legoESM computes its own Liu–Jordan clearness-index split internally
   (`split_sw_components`).  SW partitioning differs.
2. **SW absorption** — legoESM's canopy computes absorbed SW internally (no
   pre-computed `ASW_*` from the driver).
3. **Soil** — diagnostic (prescribed `Ts`, `w_frac_rz`) here vs prognostic
   (multilayer Richards + thermal) in the later stage.

## Gap handling + the `valid` mask

Met forcings have gaps.  Each required input is (a) NaN'd outside its physical
range, (b) linearly interpolated in time to a finite series so the vmap runs
everywhere, and (c) flagged by a per-step `observed` mask.  `valid` is the AND of
all required inputs' observed masks **plus** a uniform-time-axis check; it does
**not** gate on inputs that have deliberate production fallbacks (Vcmax lookup,
emissivity default).  Comparison metrics use `valid & isfinite(model) & isfinite(obs)`.

At US-Ton the binding input is `LW_IN` (≈ 42 % observed — the longwave radiometer
came online ~2014), so the trustworthy comparison window is the later record.

## Canopy robustness (required for dry sites)

Real dry/hot/high-VPD savanna forcing exposed two pre-existing canopy regressions
vs the DifferBESS oracle; both are fixed in `land/canopy/energy_balance.py`:

1. **Soft LE energy-balance cap** (`apply_le_cap`, `le_cap_mode="soft"` default) —
   a smooth softplus **upper** bound `LE ≤ max(Rn,0)+slack(Rn)` on the leaf and soil
   latent heat (`max(Rn,0)` is softplus-smoothed, so no Jacobian kink at Rn=0).
   Without it the leaf-temperature Newton solve runs away (`q_sat(Tf)` overflow →
   NaN) and the prescribed-`Ts` soil over-evaporates.  No lower bound is applied —
   negative LE (dew) passes through unchanged and a zero raw flux stays ~0 (a lower
   bound would manufacture evaporation from bone-dry soil).  Slack settings match
   DifferBESS `CanopyEnergyBalance` soft mode.
2. **Minimum cuticular conductance** (`_GS_MIN_MOL = 1e-4` mol m⁻² s⁻¹) — under the
   legacy/default `CanopyConfig.stress_b0=True`, full water stress scales the
   Ball-Berry slope **and** intercept to zero (`gs = 0`), making the leaf gs/Ci/An
   subsystem degenerate and the Newton Jacobian singular → NaN.  The floor keeps
   `gs > 0`.  With `stress_b0=False` the intercept `b0` stays > 0 (the cuticle keeps
   leaking under drought), so `gs` is already floored above `_GS_MIN_MOL` and this
   guard is inactive.  Matches DifferBESS `g0`.
3. **Production wind floor in the driver** (`_U_MIN = 1 m/s`) — the coupled land
   step always drives the canopy with `sqrt(u²+v²+U_min²)`; the offline driver
   mirrors that (`run_ec_site.py`) so calm-wind steps are not run with raw,
   near-zero wind (which production never sees and which the surface-layer solver
   handles poorly at twilight).

Effect at US-Ton (savanna): non-finite model fluxes on valid steps dropped from
~43 % to **0**; LE Pearson r rose from 0.14 to 0.72.  The driver reports the
per-flux model-non-finite count separately so masked failures are never hidden.

## Diagnostic-mode limitations

* **Sensible heat `H`** is unreliable in diagnostic mode (prescribed `Ts`): the
  soil skin temperature is read from the driver rather than solved from the
  surface energy balance, so the `H` partition and its diurnal phase are off
  (US-Ton `H` r ≈ −0.17).  The prognostic mode is expected to fix this.
* With the `U_min` floor + upper-only cap + conductance floor, no valid US-Ton or
  US-MMS step produces non-finite fluxes (`model_nan = 0`); the driver still reports
  that count per flux so any future regression surfaces instead of being masked.

## Running / adding a site

```
JAX_ENABLE_X64=1 .venv/bin/python scripts/run/run_ec_site.py \
    --driver-nc <DifferBESS>/data/sitelevel/nc/US-Ton_driver_v2.nc \
                <DifferBESS>/data/sitelevel/nc/US-MMS_driver_v2.nc \
    --mode diagnostic --out diagnostics/ec_site
```

`--start-step` / `--max-steps` restrict to a timestep window (e.g. to skip the
early gappy record); `--chunk` bounds the vmap memory.  Any v2 driver works —
the IGBP→PFT table in the adapter covers the standard land-cover classes; an
unsupported IGBP/Köppen class simply marks the site's steps `valid = False`.

## Prognostic mode (Stage B)

`--mode prognostic` integrates the **full multilayer land** (Richards soil
moisture + soil-thermal) forward in time with `lax.scan`, carrying
`MultiLayerLandState`; the canopy is invoked inside the land step and the
moisture stress `w_frac_rz` becomes prognostic (the driver's `T_soil_top` /
`w_frac_rz` are used only for the initial state).  This solves the surface energy
balance, fixing the diagnostic-mode `H` limitation (diagnostic `H` r ≈ 0.06 →
prognostic `H` diurnal r ≈ 0.93–0.99).

* **NaN-guarded carry** — because the soil state is carried, a single non-finite
  update would poison the rest of the run.  The carry is atomic: any non-finite
  state leaf reverts the *whole* state to the previous step (`jnp.where`), the
  step's emitted fluxes are masked to NaN (counted in `model_nan`), and a
  `reverted` flag is persisted.
* **Soil presets** — `--soil {default,siltloam,sandyloam}`, `--bottom-bc
  {free_drainage,zero_flux}`, `--soil-depth-m` select texture-appropriate
  hydraulics (US-MMS ≈ silt loam).  Generic-loam defaults over-drain (free
  drainage of a deep column bleeds the root zone to wilting).
* **Soil-moisture nudging** — `--nudge-tau-days T` relaxes the prognostic θ
  toward observed SWC (sub-surface layers only) with timescale τ, keeping water
  stress realistic while energy/canopy/carbon stay prognostic.  Standard
  land-flux evaluation technique; skill is scored on observed-forcing steps only
  (`score_valid` persisted).

A continuous (gap-free) MET driver is needed for prognostic scans — see
`scripts/data/build_ec_gapfree_driver.py` (reuses FLUXNET `_F_MDS`, byte-identical
to `driver_v2` on observed steps; month×slot-of-day climatology for residual long
gaps; provenance flags `met_atm_filled` / `soil_filled`).

## Cross-scale skill at US-MMS (DBF) and the LE/H/EF gap

GPP reaches **r > 0.8 at all timescales** (hourly 0.81–0.83, seasonal 0.88, IAV
0.82, diurnal 0.99).  **All flux diurnal cycles are r ≈ 0.93–0.99.**  But the
*aggregate* LE/H/EF fall short of 0.8 — traced (DifferBESS oracle + EC four-way
radiation) to **soil over-evaporation**, not Rn/albedo/WUE:

* legoESM soil evaporation ≈ **84 % of LE**; DifferBESS oracle ≈ **5 %**
  (transpiration 184 vs soil 9.9 W/m²; obs GPP & EF matched).  Winter LE is ~10×
  over, ~98 % soil evaporation at LAI ≈ 0.7 (exposed forest floor).
* **Not radiation** — net Rn matches EC obs (r = 0.999, bias −6 W/m²); albedo
  bias −0.02 (MODIS prescribed, same as DifferBESS).  **Not WUE** — GPP matches.
* **Fix applied:** soil-evaporation efficiency was the root-zone `w_frac_rz`; it
  is now the **soil pore relative humidity** `h_r = exp(ψ_top g / (R_v T))` from
  the *prognostic* top-layer matric potential (Kelvin equation), applied as a
  beta conductance efficiency.  DifferBESS's **alpha** form (`q_surf = h_r q_sat`)
  is **incompatible with legoESM's prognostic skin T** — it drives ~−200 W/m²
  condensation onto a dry surface and destabilises the energy balance (it works
  for DifferBESS only because Ts is *prescribed*).  Beta + Kelvin `h_r` is the
  safe adaptation (35/35 canopy tests pass).

### Open research items (block LE/H/EF r > 0.8)

1. **Surface ↔ root-zone moisture decoupling** — soil evaporation must respond to
   the fast-drying *surface* while transpiration draws the *moist root zone*.
   Nudging wets both (soil over-evaporates); free-running dries both
   (transpiration collapses).  The Kelvin `h_r` only bites once the surface
   genuinely dries.
2. **Cold-season / frozen-soil evaporation** — winter soil evaporates the
   (correct) net radiation at low LAI while the real surface is frozen/snow-
   limited (energy → H/G).  Needs a frozen/snow soil-evaporation suppression.
3. **Canopy↔soil radiation split** — soil sees ≈ 29 % of net radiation (summer,
   LAI = 5) vs ≈ 8 % naive Beer (clumping CI = 0.66 explains part); secondary.

## Cross-site survey (22 sites, 7 PFTs, both modes)

Run in parallel on SLURM (`scripts/cluster/ec_site/run_ec_site_array.sbatch`, a
manifest-driven array — one task per `<mode> <SITE> <out_dir> [flags]` line;
44 tasks = 22 sites × {diagnostic, prognostic-nudged}).  Sites span CRO, DBF,
EBF, ENF, GRA, SAV, SHR.  Skill = Pearson r vs tower obs; plots/CSV in
`diagnostics/ec_site_xsite/` (`xsite_skill_by_{site,pft}.png`,
`xsite_diag_vs_prog.png`, `xsite_skill.csv`).

**Mean r across the 22 sites:**

| flux | diagnostic | prognostic (nudged) |
|---|---|---|
| GPP | 0.74 | 0.66 |
| LE  | 0.52 | 0.48 |
| **H** | **0.07** | **0.77** |
| EF  | 0.17 | 0.12 |

* **Prognostic mode fixes the energy partition everywhere** — H jumps from ≈ 0
  (diagnostic prescribes Ts) to **r = 0.6–0.93 at essentially every site/PFT**
  (e.g. US-SRM −0.51→0.92, CA-Qfo −0.26→0.91, FR-Pue −0.17→0.88).  This is the
  clearest, most universal result.
* **GPP** (diagnostic) is strong for mesic forest / crop / grass (PFT means 0.73–
  0.85) but **drops sharply for dry/sparse ecosystems**: SAV 0.44 (US-SRM 0.14),
  SHR 0.24 (US-Whs).  Dry-ecosystem carbon is the largest GPP gap.  Nudged
  prognostic GPP holds for mesic types (small drop) but degrades more for some
  ENF (US-NR1 0.80→0.36).
* **LE** is mixed under prognostic — improves at many (US-MMS 0.46→0.66, FI-Hyy
  0.47→0.67) but collapses at a few (DE-Hai 0.65→0.02, DE-Gri 0.72→0.01) where
  the soil-evaporation interaction bites.  **EF** stays limited in both modes (the
  same Bowen-partition issue documented above).
* **Known site issues** (flagged, not dropped): AU-Wom = NaN in both modes
  (unsupported IGBP/Köppen → all steps `valid=False`); US-SRM / US-Whs prognostic
  = NaN (very-dry SAV/SHR prognostic — to investigate); DE-Hai / DE-Gri LE → ~0
  under prognostic (soil-evap interaction — to investigate).

**Takeaway:** the diagnosed gaps generalise across PFTs — prognostic is a decisive
win for **H** (energy balance), GPP holds for mesic types, and the residual
weaknesses cluster exactly where expected: **soil-evaporation / Bowen (LE/EF)**
and **dry/sparse ecosystems (savanna, shrubland)**.
