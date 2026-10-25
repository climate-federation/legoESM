# Ocean biogeochemistry on the forced tripole / lat-lon drivers (`ggn/bgc-tripole`)

Status, 25 Oct 2026. Branch = `main` (faabec69) + the commits listed at the end.
Three BGC schemes run online on the eORCA1 tripole ocean + sea ice under the
standard NEMO-faithful card, from GLODAPv2 / WOA23 initial fields, with
restart chaining: `--biogeo {abiotic,npzd,npzd_v2,marbl}`.

## Architecture (ShyBFM-style, Alessandri et al. 2026)

```
physics step (unchanged)  ->  TracerTransporter  ->  scheme source/sink  ->  next step
  exports mass_flux_u/v/w      flux-form advection      abiotic / npzd / npzd_v2 /
  and vmix_K_v (opt-in)        + implicit K_v mixing    marbl column step
```

* `packages/ocean/legoesm/ocean/biogeochemistry/transport.py` — `TracerTransporter`:
  advects (the card's own `tracer_advection` / RK3) and mixes (backward Euler,
  the step's composed K_v, floored at `--bgc-k-bg`) a stacked tracer array on
  partial-cell thicknesses. Operator split, first order; the physics never sees
  the tracers. Conserves `sum(h*c)` to the barotropic solver's tolerance
  (ALK: ~1e-8 relative per 1,000 steps, linear in steps).
* `.../omip_hooks.py` — `BGCRunner`: `before_step` / `after_step` / `save` /
  `load`; forcing from the surface-forcing object (PAR = 0.43 SW (1-A),
  U10^2 = |tau| / rho_a C_D, ice fraction from the prognostic sea-ice state);
  inventories, wet volume, air-sea CO2 in PgC/yr, negative-cell counts.
* `.../marbl_hooks.py` — `MARBLRunner`: the 32-tracer MARBL port
  (`biogeochemistry/marbl`, CESM2.1 PFTs, mmol/m^3) on the same transport;
  settling stays inside MARBL's column step.
* Restart: BGC tracers in `<restart_leg>.bgc.npz` beside the ocean restart,
  with the absolute step; the ocean restart, its slot policy and its config
  fingerprint are untouched. The BGC CLI args are excluded from the
  fingerprint (they do not change the ocean trajectory).
* `scripts/data/build_bgc_init_eorca1.py` — GLODAPv2.2016b (PI_TCO2, TAlk) +
  WOA23 (NO3, PO4, Si, O2) -> `bgc_init.npz` on the model grid [mol/m^3].

## Deviations from `main` that upstream should know about

1. **`LatLonCGridOceanState.vmix_K_v` + `LatLonCGridOceanConfig.store_vmix_K`**
   (8 lines in `_step_impl`, slot registered DIAGNOSTIC in `restart.py` and
   `run_omip.py`). The composed vertical tracer diffusivity the implicit solve
   used, exported like `mass_flux_*` (#1442). Off by default; inert for every
   existing run. Needed because the TKE card carries `tke_avt` only under
   `carried_previous_step`.
2. **Two carbonate solvers.** `biogeochemistry/carbonate.py` (main: surface
   solver, Follows iteration, reviewed July 2026) serves abiotic / npzd / npzd_v2.
   `biogeochemistry/marbl/carbonate_marbl.py` (the ggn/bgc_modelling solver:
   pressure-dependent constants, full-depth speciation, CO3 saturation) serves
   MARBL, which needs saturation state at depth. Reconciling them is a separate
   task; both are validated in their own suites.
3. **Positivity.** `step_ocean_biogeochemistry` keeps main's softplus limiter
   (alpha = 1e-6 mol/m^3) for abiotic / npzd. For any tracer at or below that
   scale (Fe ~1e-10, Chl, NH4) the softplus injects ~0.7 alpha per step and the
   tracer runs away (measured: Fe 50 pM -> 1.8 uM in 48 steps); npzd_v2 uses a
   plain clip. Any sub-micromolar tracer added to the v1 schemes will hit this.
4. **npzd_v2 fixes vs. the branch it came from:** grazing total was multiplied
   by P twice (zooplankton never fed, N leaked at the grazing rate); Fe was
   overwritten each step by a static profile (now prognostic + dust source).
   `tests/ocean/unit/test_bgc_npzd_v2_column.py` pins N conservation (1e-12)
   and the Fe scale.
5. **MARBL support on main:** `gas_exchange.piston_velocity` (branch signature)
   and the MARBL constants (`rho_sw_marbl`, molar masses, ...) in
   `legoesm/constants.py`.
6. **`run_omip_core2.py`:** `--biogeo` family + 18 lines of hooks;
   `--biogeo` forces `store_mass_flux` and `store_vmix_K` on.

## Known limitations / next

* First-order operator splitting; GM/Redi bolus fluxes not applied to tracers
  (off on the standard card).
* No river / atmospheric deposition sources (ShyBFM eq. 6) and no burial sink:
  closed system apart from air-sea exchange. The dilution term is NOT needed:
  the z* flux-form update conserves `h*c` under freshwater volume change.
* NPZD-family settling is inside the source/sink on reference `dz`; moving it
  into the transport (real thicknesses) is planned.
* MARBL deposition forcing (`--marbl-forcing`, monthly dust/Fe/NOx/NHy on the
  model grid) not yet built; Fe starts from a 0.1-0.6 nM profile.
* Cost (1 A100, eORCA1 L75, dt 150 s): physics 0.65 steps/s = 0.28 SYPD;
  +15 % MARBL at `--bgc-every 4`, +25 % npzd_v2 at every step.

## Validation runs (Levante, `/work/bb1596/b384036/diffESM/runs/step1/`)

| run | scheme | IC | length | result |
|---|---|---|---|---|
| B1_abiotic_smoke | abiotic | uniform | 2 d + resume | ALK conserved 1e-8; outgassing -43 PgC/yr (uniform IC) |
| B2_npzdv2_smoke | npzd_v2 | uniform | 2 d | 18 tracers, no negatives |
| B3_npzdv2_glodap | npzd_v2 | GLODAP/WOA | 10 d | air-sea +8 -> +3.7 PgC/yr |
| B4_marbl_smoke | marbl | GLODAP/WOA + seeds | 2 d + resume | NPP 22->48, export 1.0->2.0, CO2 +2.1 PgC/yr, pCO2 272, P conserved 1e-7 |

## Commits over main

```
d448152 bgc: import MARBL 3-D port
87044fa bgc: operator-split tracer transport + --biogeo in run_omip_core2
e84c593 ocean: opt-in vmix_K_v diagnostic slot (store_vmix_K)
9dd49fe bgc: GLODAPv2.2016b + WOA23 -> eORCA1 initial fields
8d7a862 bgc: npzd_v2 on main; grazing/positivity fixes; cumulative inventory ref
5c77d91 bgc: --biogeo marbl (MARBLRunner)
4ecb70a marbl: carbonate_marbl.py, piston_velocity, MARBL constants
cff76c5 marbl: fix inverted ocean mask; sidecar pco2 for MarblConfig
```
