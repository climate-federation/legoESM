# Bulk-Flux Audit for OMIP Compliance (LY09)

**Audit target**: Tropical OMIP Phase A (`tropical_omip_plan.md` Item 1)
**Date**: 2026-05-02
**Files audited**:
- `src/legoesm/coupler/bulk_flux.py` (MOST iteration + transfer coefficients)
- `src/legoesm/ocean/physics/surface_forcing/bulk_formulas.py` (ocean-side adapter)

OMIP §2.2 (Griffies 2016): *"Bulk formulae for computing turbulent fluxes
for heat and momentum must follow Large and Yeager (2009)."*

## TL;DR (revised after broader audit)

**Initial scope of this audit was too narrow.** I first audited only
`coupler/bulk_flux.py` and the idealized adapter
`ocean/physics/surface_forcing/bulk_formulas.py` and concluded six gaps
existed. A subsequent survey of the full coupler / freshwater stack
(`coupler/coupler.py`, `coupler/surface_exchange.py`, `ocean/freshwater.py`,
`coupler/mpas_adapter.py`, and the `model.step(freshwater=...)` API on
`LatLonCGridOceanModel`) showed that **half of those "gaps" are already
solved infrastructure** wired through the realistic-coupler path. See
"Errata" below.

**Genuine remaining gaps (confirmed)**:

1. The C_DN drag formula in `compute_most_fluxes` is **LY04**, missing
   LY09's high-wind quintic correction term. **One-line fix** in
   `coupler/bulk_flux.py:210`.
2. The **0.98 salinity correction** on saturation specific humidity at the
   sea surface is missing from the realistic coupler path. **One-line fix**
   in `coupler/coupler.py:183`.
3. `compute_most_fluxes` takes a **single `z_ref`** for momentum, heat, and
   moisture. JRA55-do delivers winds at 10 m and T,q at 2 m, so we need
   separate `z_u`, `z_t`, `z_q`. Not a one-line fix — generalize the API.
   **~1 day.**

**Already-solved (was incorrectly flagged)**:

- 2D atmospheric forcing — `coupler.ocean_tile_response()` already takes
  fully 2D `AtmToSurface` fields including u_lowest, v_lowest, T_lowest,
  q_lowest, p_surface, rho_lowest, sw_down, lw_down, precip, runoff.
- `psl` plumbing — `surface_exchange.extract_atm_to_surface()` returns
  `p_surface`, and `coupler.py:183` passes it directly to
  `saturation_mixing_ratio(ocean_sst, forcing.p_surface)`.
- Freshwater flux — `ocean/freshwater.py` provides `FreshwaterForcing` +
  `freshwater_from_coupler()` + `virtual_salt_flux()`, and
  `LatLonCGridOceanModel.step()` already accepts a `freshwater=` kwarg
  (`ocean_model_latlon_cgrid.py:364, 482, 806`) that drives both the η and
  S tendencies. No new path needed.

The `bulk_formulas.py` adapter (with `dS_dt = 0` hard-coded and scalar
forcing) is **the idealized-experiment path**, not what realistic coupled
runs use. For tropical OMIP we route through `ocean_tile_response` →
`compute_most_fluxes` → `freshwater_from_coupler` → `model.step(freshwater=...)`,
which is mostly done.

**Net effort revision**: ~1.5 days of bulk-flux fixes (the three confirmed
gaps), not the 6 days I previously estimated. The other 4–5 days I had
allocated to "build a 2D-aware adapter and wire freshwater" are unnecessary
because that infrastructure exists; what remains is **driver-level glue**
(building `AtmToSurface` and `FreshwaterForcing` from JRA55-do fields), which
properly belongs to Item 4 (driver), not Item 1 (bulk flux).

## Errata — initial findings I retracted after broader audit

The original draft of this memo audited only two files
(`coupler/bulk_flux.py` and `ocean/physics/surface_forcing/bulk_formulas.py`)
and flagged six gaps. A subsequent Explore pass over the full coupler and
freshwater stack overturned three of them. Documenting here so the trail is
clear:

| Originally flagged | Reality | Where it actually lives |
|---|---|---|
| "psl is hard-coded to 101325 Pa" | **Already plumbed** | `coupler/surface_exchange.py:90` returns `p_surface`; `coupler/coupler.py:183` uses it in `saturation_mixing_ratio(ocean_sst, forcing.p_surface)` |
| "Atmospheric forcing is scalar zonal-only" | **Fully 2D in the realistic path** | `coupler/coupler.py:169-256` `ocean_tile_response()` takes the `AtmToSurface` 2D struct populated by `extract_atm_to_surface()` |
| "Freshwater flux missing (`dS_dt = 0` hard-coded)" | **Fully implemented** | `ocean/freshwater.py` provides `FreshwaterForcing` + `freshwater_from_coupler()` + `virtual_salt_flux()`; `LatLonCGridOceanModel.step()` accepts `freshwater=` and applies both η and S tendencies (`ocean_model_latlon_cgrid.py:364, 482, 806`) |

The `bulk_formulas.py` file I audited is the **idealized-experiment adapter**
(scalar wind for sin² gyres etc.). It is not on the realistic-coupling path.
For OMIP the call chain is:

```
JRA55-do fields
  → AtmToSurface struct (built by driver glue)
  → coupler.ocean_tile_response()           # τ, Q_sh, Q_lh
      → coupler.bulk_flux.compute_most_fluxes()
  → ocean.freshwater.freshwater_from_coupler()  # P, E, runoff → FreshwaterForcing
  → LatLonCGridOceanModel.step(freshwater=...)  # η + dS tendencies
```

Most of this chain already exists. The driver-side glue (build
`AtmToSurface` from forcing reads, build `FreshwaterForcing` from
`P − E + runoff`) is part of Item 4, not Item 1.

## Component-by-component findings

### 1. Drag coefficient C_DN(U_10N) — LY04, not LY09

**Current** (`bulk_flux.py:210`):
```python
C_DN = (2.7 / U_10N + 0.142 + 0.0764 * U_10N) * 1e-3
```

The first three terms agree with LY09 (note 0.0764 ≈ 1/13.09 = 0.07640183,
identical at machine precision). **Missing**: LY09's high-wind quintic
correction term:

```python
C_DN_LY09 = (2.7/U_10N + 0.142 + U_10N/13.09 - 3.14807e-10 * U_10N**6) * 1e-3
```

**Why it matters**: the −3.14807e-10·U⁶ term suppresses unphysical drag
growth at high winds (U > ~33 m/s). At U = 50 m/s the LY04 formula gives
C_DN ≈ 4.0×10⁻³ (clipped to 3.0×10⁻³ in our implementation), while LY09
gives C_DN ≈ 0.97×10⁻³. JRA55-do contains realistic storm-track winds up
to ~30 m/s in the Southern Ocean and North Atlantic; the ~5–8 m/s tail
beyond 30 m/s is rare but produces large per-event flux differences.

**Fix**:
```python
C_DN = (
    2.7 / U_10N
    + 0.142
    + U_10N / 13.09
    - 3.14807e-10 * U_10N**6
) * 1e-3
C_DN = jnp.clip(C_DN, 0.5e-3, 3.0e-3)
```

The clip remains correct as a safety floor/ceiling.

**Status**: 1-line fix. Verifiable with Tsujino et al. 2018 Fig. 3.

### 2. Stability-dependent Stanton/Dalton numbers — LY09-compliant ✓

**Current** (`bulk_flux.py:216–217`):
```python
CHN10 = jnp.where(zeta < 0.0, 32.7e-3, 18.0e-3) * rdn
CEN10 = 34.6e-3 * rdn
```

LY09 published values:
- Stanton (heat): 32.7×10⁻³ unstable, 18.0×10⁻³ stable ✓
- Dalton (moisture): 34.6×10⁻³ ✓

**Note**: LY09 uses 18.0×10⁻³ for *stable conditions over open ocean*. Some
later implementations (e.g., MOM6/SIS2 default) use a slightly different
stable Stanton number; we are LY09-correct here.

**Status**: no fix needed.

### 3. Stability functions — LY09-compliant ✓

`psi_m`, `psi_h` use Businger-Dyer (unstable) and Dyer 1974 (stable). LY09
Table 4 cites Large & Pond 1982 which uses the same Businger 1971 stability
functions. Mathematical forms match.

**Status**: no fix needed.

### 4. MOST iteration count — LY09-compliant ✓

`n_iter=5` default in `compute_most_fluxes`. LY09 §3 explicitly uses 5
fixed-point iterations. The `jax.lax.fori_loop` implementation preserves AD
compatibility.

**Status**: no fix needed.

### 5. Sea-surface saturation humidity — missing 0.98 correction

**Current** (`coupler/coupler.py:183`, the realistic-coupling path):
```python
q_sfc = saturation_mixing_ratio(ocean_sst, forcing.p_surface)
```

(The idealized adapter `bulk_formulas.py:49` has the same omission, but the
realistic coupler is the path tropical OMIP uses.)

LY09 §3 (and Large 2006 §3.4) specify that the *effective* sea-surface
saturation humidity is reduced by 2 % due to ocean salinity:
```python
q_sat_sea = 0.98 * q_sat(SST, p_surface)
```

Without this, latent heat fluxes are systematically biased high by ~2 % in
absolute terms, which translates to ~2 W/m² over the tropical ocean — small
but nonzero, and protocol-required.

**Fix** in `coupler/coupler.py:183`:
```python
q_sfc = 0.98 * saturation_mixing_ratio(ocean_sst, forcing.p_surface)
```

(And the same fix in `bulk_formulas.py:49` for consistency in the idealized
path, though that path doesn't go to OMIP.)

**Status**: 1-line fix. The saturation thermodynamics helper
(`legoesm.thermo`) is already correct — only the application site at the
air-sea interface needs the 0.98.

### 6. Reference-height pressure for q_sat — already plumbed (no fix)

**Initially flagged** as a gap in the idealized adapter (`bulk_formulas.py:14`
hard-codes `_P_ATM = 101325.0`). **In the realistic coupling path this is
already correct**: `surface_exchange.extract_atm_to_surface()` populates
`p_surface` from the atmospheric state and `coupler.py:183` passes it to
`saturation_mixing_ratio`. JRA55-do `psl` lands in `forcing.p_surface` via
the driver glue, no change required.

The only residual cleanup is in the idealized adapter, which is irrelevant
for OMIP. Fix optional, low priority.

**Status**: not a gap on the OMIP path. Idealized adapter cleanup is cosmetic.

### 7. Wind reference height — implementation question

**Current** (`bulk_formulas.py`): caller passes `cfg.z_ref` (default 10 m).
JRA55-do `uas`/`vas` are at 10 m. `tas`/`huss` are at 2 m. Currently the
adapter passes everything to `compute_most_fluxes` with the same `z_ref`,
which is wrong for the heat/moisture variables.

LY09 standard practice (and JRA55-do convention): convert all atmospheric
state to **10 m** values before applying bulk formulas. The wind is already
at 10 m; T and q must be height-adjusted from 2 m to 10 m using the MOST
profile shape (a self-consistency loop).

**Alternative**: pass separate `z_u = 10`, `z_t = 2`, `z_q = 2` to the MOST
solver and let it height-correct internally. This is what LY09 §3 actually
prescribes (Eqs. 13–15).

**Fix** (medium effort, ~1 day): generalize `compute_most_fluxes` to accept
separate reference heights for momentum vs scalar variables.

**Status**: medium fix. Can be deferred for shakedown but should be in
before final cycle 1.

### 8. Atmospheric state input — already 2D in the realistic path (no fix)

**Initially flagged** based on the idealized adapter
(`bulk_formulas.py:56-60` builds 2D arrays from scalar `cfg.U_a`, etc.).
**The realistic coupling path is already 2D**:

`coupler/surface_exchange.py:18-96` `extract_atm_to_surface()` returns an
`AtmToSurface` struct of 2D (ny, nx) fields:
- `u_lowest`, `v_lowest` — wind components
- `T_lowest`, `q_lowest` — air state
- `p_lowest`, `p_surface`, `rho_lowest` — pressure/density
- `sw_down`, `lw_down` — radiation
- `precip_total`, `precip_snow` — precipitation
- `cos_zenith` — for albedo

`coupler/coupler.py:169-256` `ocean_tile_response()` consumes these directly
and calls `compute_most_fluxes` per cell.

**For OMIP**, the driver builds an `AtmToSurface` struct from JRA55-do reads
each step (driver-level glue, ~½ day). No new bulk-flux infrastructure
needed. The idealized scalar-config path coexists for unit tests / sin² gyre
experiments.

**Status**: not a gap on the OMIP path. Driver glue lives in Item 4.

### 9. Freshwater flux — already implemented (no fix)

**Initially flagged** based on the idealized adapter
(`bulk_formulas.py:105` has `dS_dt = jnp.zeros(...)`). **The realistic path
already has the full freshwater chain**:

- `ocean/freshwater.py` — `FreshwaterForcing` struct (`precip`, `evap`,
  `runoff`, `ice_fw`); `freshwater_from_coupler()` (lines 141-234) computes
  `evap = lhflx / L_v` and assembles `precip + runoff + ice_fw - evap` into
  the freshwater forcing; `virtual_salt_flux()` (lines 108-138) returns
  `dS/dt = -S_ref * F_fw / (ρ_0 · dz_0)`; `freshwater_eta_tendency()`
  returns the η contribution.
- `ocean/dynamics/ocean_model_latlon_cgrid.py` — `LatLonCGridOceanModel.step()`
  takes `freshwater=` kwarg (line 363). Inside the step, `F_slow_eta` is
  applied to the barotropic split (line 482-484) and `dS_fw` is added to
  the salinity tendency (line 806-811). Behavior is gated by
  `config.freshwater_closure ∈ {"none", "virtual_salt_flux"}`.
- `coupler/mpas_adapter.py:137-195` — `compute_mpas_freshwater()` shows the
  pattern of bridging coupler outputs (`AtmToSurface`, `SurfaceToAtm`, ice
  state) into `FreshwaterForcing`. For lat-lon the bridge is a few lines of
  driver code (no MPAS-specific structure needed).

For tropical OMIP without ice: `ice_fw = 0`. Then:
```
F_fw = P_liq + P_sol + runoff - evap         # all 2D, all from JRA55-do or lhflx
fw_forcing = FreshwaterForcing(precip=P, evap=E, runoff=R, ice_fw=0)
state_new = model.step(state, dt, freshwater=fw_forcing, ...)
```

**OMIP boundary-flux diagnostics** `wfo`, `wfonocorr`, `wfcorr` (Tables K1)
are computed by accumulating these terms separately in the diagnostics
module (Item 7), not by the bulk-flux code itself.

**Status**: not a gap. Wiring `freshwater_from_coupler()` into the JRA55-do
driver is ~½ day of glue in Item 4.

### 10. Latent heat of vaporization — uses constant L_v (acceptable)

`bulk_flux.py:282`: `_L = constants.L_v if L_latent is None else L_latent`.
LY09 uses a constant L_v = 2.5×10⁶ J/kg. We use `constants.L_v = 2.5×10⁶`
(same value). For snow, additional latent heat of fusion is needed
(L_f = 3.34×10⁵ J/kg) to convert snow at 0°C to liquid before evaporation
accounting. Not critical for tropical Phase A (no snow); needed for Phase B.

**Status**: acceptable for Phase A; revisit for Phase B.

### 11. Long-wave up-welling — emissivity hard-coded via cfg.emissivity ✓

`bulk_formulas.py:51`: `Q_lw_up = cfg.emissivity * constants.sigma_sb * T_s ** 4`.
LY09 uses ε_ocean = 1.0 (true blackbody assumption). `BulkFormulaConfig`
default emissivity is 1.0 per the surface_forcing audit. **Status**: OK.

## Summary table — fixes needed for OMIP compliance (revised)

| # | Issue | File | Severity | Effort | Phase A blocker? |
|---|---|---|---|---|---|
| 1 | LY04 → LY09 drag formula (add U⁶ term) | `coupler/bulk_flux.py:210` | High | 1 line | yes |
| 2 | Add 0.98 salinity correction to q_sfc | `coupler/coupler.py:183` | Medium-High | 1 line | yes |
| 3 | Separate z_u (10 m) vs z_t,z_q (2 m) heights | `coupler/bulk_flux.py` API + `CouplerConfig` | Medium | ~1 day | yes (cycle-1 quality) |

**Retracted from original list**:
- ~~psl plumbing~~ — already done via `forcing.p_surface`.
- ~~2D atmospheric forcing~~ — `ocean_tile_response` already consumes 2D
  `AtmToSurface`. Driver-level glue to build it from JRA55-do is Item 4.
- ~~Freshwater flux~~ — `FreshwaterForcing` + `freshwater_from_coupler` +
  `model.step(freshwater=...)` already exist and are wired into the
  lat-lon C-grid model. Driver-level glue is Item 4.

**Total effort**: ~1.5 days for the bulk-flux fixes (was 6 days in the
original draft of this memo). The remaining 4–5 days I had allocated to
"build a 2D-aware adapter and wire freshwater" are unnecessary because that
infrastructure exists; that work is **driver glue**, properly assigned to
Item 4 (driver), not Item 1 (bulk flux).

## Recommended execution order (revised)

1. **Half-day**: LY09 C_DN (#1) + 0.98 q_sfc (#2) — two 1-line changes in
   `bulk_flux.py:210` and `coupler.py:183`. Add unit tests:
   `test_ly09_drag_high_wind` (verify C_DN against Tsujino 2018 Fig. 3 at
   U = 10, 20, 30, 40 m/s within 1 %), `test_98_pct_qsat_correction`.
2. **1 day**: separate reference heights (#3). Generalize
   `compute_most_fluxes` to accept `z_u`, `z_t`, `z_q`. Default
   `z_t = z_q = z_u` so existing call sites are unaffected. Plumb
   `CouplerConfig.z_u`, `z_t`, `z_q` defaults of 10, 2, 2.
3. **Half-day**: integration test — full bulk-flux pipeline against a
   single timestep of JRA55-do forcing on a small global subset; compare
   τ, Q_sh, Q_lh against Tsujino 2018 reference within ±5 %.

## Critical-path implications for the tropical OMIP plan (revised)

The parent plan (`tropical_omip_plan.md`) sized Item 1 at 1–2 days. **After
the broader audit, that estimate is roughly correct** — split into:

- **Item 1a**: audit memo (done, this document).
- **Item 1b**: LY09 + 0.98 + z_u/z_t implementation, ~2 days.

Item 1b is small enough that it can be folded back into Item 1 without a
separate critical-path entry. The "structural" work I had originally added
(2D forcing adapter, freshwater path) **does not exist as a gap** — it
exists as driver glue inside Item 4, where the JRA55-do reader builds an
`AtmToSurface` and a `FreshwaterForcing` per step.

The MOST iteration in `compute_most_fluxes` is sound, AD-compatible, and
reusable with no structural changes — only the LY09 coefficient fix and the
reference-height generalization.

## Tests to add (post-implementation)

- `test_ly09_drag_high_wind`: verify C_DN at U = 10, 20, 30, 40 m/s matches
  Tsujino 2018 Fig. 3 within 1 %.
- `test_98_pct_qsat_correction`: verify q_sat_sea = 0.98 × q_sat at given
  SST.
- `test_bulk_flux_2d_forcing`: verify the 2D-field path produces the same
  result as the scalar path when forcing fields are uniform.
- `test_freshwater_balance`: verify global integral of P − E over a single
  step matches the input forcing P − E to machine precision (no spurious
  sources/sinks).
- `test_jra55do_single_column`: a single 1° column at 30°N, 30°W with
  one timestep of JRA55-do forcing; verify tau, Q_sh, Q_lh against the
  Tsujino reference within ±5 %.

These slot into `tests/ocean/unit/test_bulk_formulas.py` and a new
`tests/ocean/unit/test_bulk_flux_ly09.py`.
