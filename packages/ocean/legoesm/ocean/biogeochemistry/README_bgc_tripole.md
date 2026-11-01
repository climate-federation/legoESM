# Ocean biogeochemistry on the forced tripole / lat-lon drivers (`ggn/bgc-tripole`)

Status, 1 Nov 2026. Branch = `main` (faabec69) + the commits listed at the end.
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
* Surface deposition for MARBL via `--marbl-forcing` (dust, soluble Fe, NOy, NHx;
  `scripts/data/build_marbl_forcing_eorca1.py` from the CESM2 gx1v6 set: 528 Tg dust,
  7.9 Gmol Fe, 13.5 Tg N per year in 1850). Not yet wired: sediment / hydrothermal Fe
  (3-D interior source), rivers (GNEWS2000) and burial -> Step 4. npzd_v2 still uses a
  uniform `Fe_dust`.
* NPZD-family settling is inside the source/sink on reference `dz`; moving it
  into the transport (real thicknesses) is planned.
* The GPU runs are not bit-reproducible (XLA atomics): two legs with identical
  configuration differ at ~1e-6 in point maxima and in the last printed digit of
  mean SST after weeks, without growing. Any restart round-trip / one-way-coupling
  check must use a tolerance, or `--xla_gpu_deterministic_ops=true`.
* Seeds: MARBL's semi-labile DOM must start small (DOC 1, DON 0.1, DOP 0.01
  mmol/m^3). A 40 mmol/m^3 DOC seed (~640 PgC) remineralised at ~100 PgC/yr and
  drove surface pCO2 from 273 to 295 ppm within six months (C2_marbl_6mo).
* npzd_v2 keeps phytoplankton near its refuge value outside the subpolar North
  Atlantic (zooplankton > phytoplankton at the surface, NO3/PO4 accumulating): a
  grazing/mortality calibration issue of the scheme, to be done in the 1-D harness
  (`tests/ocean/unit/test_bgc_npzd_v2_column.py`). K_Fe (80/50 pM) is not the cause:
  MARBL grows normally under the same Fe field.
* Cost (1 A100-40GB, eORCA1 L75, dt 150 s): physics 0.65 steps/s = 0.28 SYPD;
  +15 % MARBL at `--bgc-every 4`, +25 % npzd_v2 at every step. At dt = 1200 s
  (Phase-1 plan) use `--bgc-every 3` (1-h BGC step, CESM's interval) to keep the
  MARBL share at that level.

## Validation runs (Levante, `/work/bb1596/b384036/diffESM/runs/step1/`)

| run | scheme | IC | length | result |
|---|---|---|---|---|
| B1_abiotic_smoke | abiotic | uniform | 2 d + resume | ALK conserved 1e-8; outgassing -43 PgC/yr (uniform IC) |
| B2_npzdv2_smoke | npzd_v2 | uniform | 2 d | 18 tracers, no negatives |
| B3_npzdv2_glodap | npzd_v2 | GLODAP/WOA | 10 d | air-sea +8 -> +3.7 PgC/yr |
| B4_marbl_smoke | marbl | GLODAP/WOA + seeds | 2 d + resume | NPP 22->48, export 1.0->2.0, CO2 +2.1 PgC/yr, pCO2 272, P conserved 1e-7 |
| C2_npzdv2_6mo | npzd_v2 | GLODAP/WOA, branched from B3 | 180 d, 7 legs | C/N/P/Si closed; uptake 3.8 -> 0.7 PgC/yr; surface Pd+Pn flat at 0.12 mmol N/m^3; deep seed dies (correct) |
| C2_marbl_6mo | marbl (DOC seed 40) | same parent | 180 d, 7 legs | NPP 82 -> 44, export 9 -> 4 PgC/yr, Chl 0.6 -> 0.15; P 3e-6, Si 9e-6; DIC +1.3e-3 linear = DOC-seed artefact; pCO2 273 -> 295 |
| C2 one-way split | both | | 180 d | ocean diagnostics of the two chains identical to GPU roundoff (SST last digit, max\|u\| 1e-6, no growth) |
| B5_marbl_seedfix | marbl (DOC seed 1) | | 2 d | total C conserved 2.7e-8; DIC rise now the dying 3-D living seed only |
| B6_marbl_dep | marbl + `--marbl-forcing` | | 2 d | N deposition exact (+0.9e-7); Fe +3.1e-5 of +4.5e-5 deposited retained (rest: dust-scaled scavenging/burial) |
| C1 + C1b | physics only | WOA | 1.08 yr, 10 legs | dt = 150 reference: 0.66 steps/s; AMOC@26N 23 -> 34 -> 26 Sv, Drake 136 -> 146 -> 144 Sv over year 1 |

Figures: `docs/figures/bgc_tripole/c2_marbl_vs_npzdv2_6mo.png` (time series),
`docs/figures/bgc_tripole/c2_endstate_surface_maps.png` (day-180 surface fields);
`scripts/analysis` equivalent: `/work/bb1596/b384036/diffESM/runs/step1/compare_c2.py`.

Decision (Step 1): MARBL carries the Step 4 BGC spin-up; npzd_v2 is the cheap
baseline pending calibration.

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
114e0b6 docs: README
ba082b3 bgc: branch legs start the BGC fresh when the parent sidecar holds another scheme
33ec391 restart fingerprint: exclude LEGOESM_ALLOW_VERSION_DRIFT
5a27384 marbl: realistic semi-labile DOM seeds; total-carbon inventory in diagnostics
1fed170 bgc: CESM2/MARBL deposition forcing (dust, soluble Fe, NOy, NHx) -> eORCA1
(this)  docs: README after Block C
```
