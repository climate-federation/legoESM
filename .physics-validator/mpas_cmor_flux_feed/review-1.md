## Confirmed defects

- **CONFIRMED — `rsdt` uses the wrong TOA value.** [radiation/integration.py:1202](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1202) exports the top-halo downwelling flux, while the backend explicitly preserves prescribed `toa_insolation` for this CMOR diagnostic at [integration.py:735-739](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:735). The compiled lane correctly prefers it at [physics_pipeline.py:2221-2227](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/physics_pipeline.py:2221). This does not restore the old unconstrained 2× failure, but it does retain the documented biased halo value.  
  Fix: use `rad_out.toa_insolation if ... is not None else rad_out.sw_flux_down[:, 0]`.

- **CONFIRMED — Held–Suarez runs discard all five new diagnostics.** The MPAS Held–Suarez wrapper rebuilds `HydrostaticTendencies` and forwards only `sw_net_sfc`, `lw_net_sfc`, and precipitation at [model_driver.py:6205-6223](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:6205). Thus `rlut`, `rsut`, `rsdt`, `hfss`, and `hfls` are all `None` downstream whenever `held_suarez_forcing=True`.  
  Fix: forward the five new fields, preferably via `_replace` on the original tendency after adding HS tendencies.

- **CONFIRMED — the MPI-Voronoi producer still implements the old 3-slot contract.** It initializes and emits `(sw, lw, precip)` only at [voronoi_mpi.py:1029-1058](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/parallel/voronoi_mpi.py:1029), then merges with an unpadded `zip` at [voronoi_mpi.py:1135-1138](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/parallel/voronoi_mpi.py:1135). The driver explicitly enables CMOR feeding for a one-rank Voronoi layout at [model_driver.py:5448-5467](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:5448), so that valid mode silently loses all new fields. Multi-rank CMOR is already loudly unsupported, not silently wrong.  
  Fix: mirror the serial extraction/order and padded slot-wise merge.

- **CONFIRMED — turbulent diagnostic `Field` metadata says Pa, not W/m².** [turbulence/integration.py:671-674](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/atmosphere/legoesm/atmosphere/physics/turbulence/integration.py:671) uses `state.p_s.replace`; `Field.replace` preserves the original units ([field.py:90-98](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/core/field.py:90)). Numerically the current driver strips `.data`, and CMOR writer metadata masks this, but the in-memory contract is false.  
  Fix: pass `units="W/m^2"` (and an appropriate `long_name`) in both replacements.

## Candidate adjudication

- **C1: CONFIRMED.** See first finding.

- **C2: CONFIRMED for one-rank MPI; qualified for multi-rank.** Multi-rank is explicitly refused with a warning, but one-rank MPI is accepted and broken.

- **C3: FALSE-POSITIVE.** The zonal accumulator intentionally contains only `T_low`, precip, and `psl` ([diagnostics.py:1550-1559](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/diagnostics.py:1550)); the compiled `collect()` path has the same zonal scope. The new fields are Amon spatial outputs, which the writer consumes. Extend `z2d` only if zonal flux products are now a requirement.

- **C4: FALSE-POSITIVE as an ice/sublimation complaint for the default MPAS lane.** The MPAS turbulence water boundary condition itself uses `lhflx / L_v`, so no separate sublimation mass-flux channel exists to export.  
  **AMBIGUOUS configuration caveat:** `thermo_convention="aerobulk"` forms `lhflx` with temperature-dependent latent heat ([bulk_flux.py:1096-1115](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/core/bulk_flux.py:1096)). The collector’s constant-`L_v` conversion is then not an energy-to-evaporation inversion. Fix, if AeroBulk CMOR fidelity matters: export a native evaporation/mass-flux diagnostic from turbulence rather than deriving it from `hfls`.

- **C5: FALSE-POSITIVE.** `_sfc_diag` is not checkpointed, but each restarted MPAS job deliberately performs radiation on local step zero ([model_driver.py:6642-6648](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:6642)) before diagnostics/feed run. It is absent briefly, not stale, and cannot cause a missing normal diagnostic sample.

- **C6: serial tuple order is correct.** Producer order is LW-up, SW-up, SW-down, SH, LH ([primitive_eq_mpas.py:868-875](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_mpas.py:868)); consumer maps slots 3–7 consistently ([model_driver.py:5528-5532](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:5528)). No sign flip is needed: radiation and turbulence conventions match the requested CMOR signs.

`reshape`/`astype` in the packer are JAX-safe; the host-side accumulator is intentionally non-differentiable diagnostics code. Shape validation is transactional before accumulator mutation.

Tests are insufficient: the new test covers only direct packer/collector calls, not the serial `sfc_diag` bridge, Held–Suarez wrapper, MPI producer, `toa_insolation` preference, or turbulent-field units. I could not run pytest because it is not installed in this environment; `git diff --check` is clean.
