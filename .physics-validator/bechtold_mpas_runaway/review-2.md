Not converged yet. A and B are real defects, but the proposed 2×2 does not identify their interaction.

A new confirmed activation bug matters for B: on MPAS, `_ext_forcing` omits `_aerosol_lw_active`. Thus volcanic LW is only materialized if some other external channel or experiment is also active; with LW volcanic forcing alone, `_precompute_external_forcing()` is never called and no `aerosol_lw_od` reaches radiation. [model_driver.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:6157) [model_driver.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:6471)

So B’s actual MPAS gate is stricter than “flag + file.” For `probe_bech40`, record all three of:

- resolved `volcanic_aerosol_lw` and file path;
- `_aerosol_lw_active`;
- area-mean/max `aerosol_lw_od` actually passed on day 1.

On B itself: malformed LW OD is a confirmed radiation-input defect, but “therefore reduced OLR” remains an inference. A pure absorber also emits; its net TOA sign depends on the temperature/cloud state. Diagnose paired radiation calls on the identical state: current LW OD versus zero LW OD, and report ΔOLR, Δsurface-LW, and vertical ΔLW heating. `_aerosol_lw_od × heating` is not a meaningful budget quantity.

Your factors are presently mislabeled:

- SBM vs Bechtold is a scheme contrast, not A.
- Bechtold `dx_m=0` vs physical `dx_m` is A.
- pinned vs transient deck is a bundled deck factor, not B.

A valid A×B test is four Bechtold runs with every other forcing fixed:

| Contrast | Levels |
|---|---|
| A: resolution closure | `dx_m=0` vs `dx_m≈sqrt(mean cell area)` |
| B: volcanic-LW defect | malformed LW OD on vs zero OD |

Then add SBM as a baseline, and test pinned/transient deck separately. Otherwise, a deck interaction could be ozone, SW aerosol, GHG, or radiation cadence—not volcanic LW.

Also, a full identical checkpoint cannot be restarted across SBM/Bechtold: the MPAS restart guard rejects a convection-scheme mismatch when the saved physics carry is present. [model_driver.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:6224) You need identical atmospheric/dynamical fields but deliberately constructed scheme-compatible carries; stripping all physics state also resets radiation/turbulence memory and creates a transient.

Your leaf result supports a narrower conclusion: Bechtold is not a finite-call moist-enthalpy source on that exercised path. It does not prove the full MPAS step is energy-conservative. The MPAS bridge routes convective condensate/rain into tracers for later microphysical processing, so water closure must include all condensate species, not only vapor. [integration.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/atmosphere/legoesm/atmosphere/physics/convection/integration.py:664)

For the MPAS ledger, take state deltas after each operator, area-weighted with `areaCell * dp/g`, and include:

- all water species, surface precipitation, and pressure-mass changes;
- dry enthalpy plus moist, kinetic, and geopotential terms if calling it an energy closure;
- explicit rows for the temperature floor, tracer floors, and pressure mass fixer.

Those three numerical operations are outside your listed process rows and can alter stored energy/water. [primitive_eq_mpas.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_mpas.py:887) The existing `--budget-ledger` is not sufficient: it is dry-enthalpy-only and is not wired through this MPAS combined-physics path. [process_ledger.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/tools/legoesm/diagnostics/process_ledger.py:23)

Two remaining causal cautions:

- Fixed SST genuinely can supply LHF in MPAS—the turbulence bridge uses forced `T_sfc`—but this must be measured, not assumed. It currently does not export SH/LH as an MPAS tendency diagnostic, so instrument the turbulence output directly. [integration.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/atmosphere/legoesm/atmosphere/physics/turbulence/integration.py:586)
- The wind spike is not evidence for Bechtold CMT: MPAS gives Bechtold zero cell winds for CMT and zeroes the incompatible edge-wind tendency. It instead keeps GWD/dycore pressure-gradient pathways live. [integration.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/atmosphere/legoesm/atmosphere/physics/convection/integration.py:400) [integration.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/atmosphere/legoesm/atmosphere/physics/convection/integration.py:720)

My ranking is therefore: A is a confirmed amplifier/wiring bug; B is a confirmed, activation-dependent radiation bug; neither is yet the demonstrated source of this runaway. The live direct-source candidates remain actual SST turbulent flux, diagnosed Hines wave-energy closure, malformed LW forcing if active, and MPAS clipping/mass-fixer/dycore residuals. The corrected A×B experiment plus exact TOA/surface flux differences will decide it.
