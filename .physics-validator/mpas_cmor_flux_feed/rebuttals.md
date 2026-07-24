# Rebuttals — MPAS CMOR flux feed

## Round 1 (codex agreed these are NOT bugs)

**C3 — zonal accumulator omits the 5 new fields → FALSE POSITIVE.**
The zonal `z2d` bundle (diagnostics.py:1550-1559) intentionally carries only
`T_low`, precip, `psl`; the compiled `collect()` path has the identical zonal
scope. rlut/rsut/rsdt/hfss/hfls/evspsbl are CMOR **Amon spatial** products,
which the ClimateEval radiation-budget + surface-flux suites consume from the
spatial `fields_2d` regrid. Adding them to `z2d` would be scope creep, not a fix.
Codex concurred. Would only extend if zonal-mean flux products become a
requirement.

**C4 — evspsbl = hfls / L_v approximation → FALSE POSITIVE for the campaign lane.**
The MPAS default bulk turbulence water boundary condition itself forms lhflx
with a constant L_v (surface_layer.py), and the compiled path derives evap the
same way (`evap_rate = lhflx / constants.L_v`, physics_pipeline.py:1466). So the
lean derivation is EXACTLY consistent with both the scheme's own partition and
the compiled path — no information is lost. Documented caveat (codex): under
`thermo_convention="aerobulk"` lhflx uses a temperature-dependent latent heat,
so a constant-L_v inversion is not exact; if AeroBulk CMOR fidelity is later
required, export a native evaporation mass-flux from turbulence rather than
inverting hfls. Not in scope for the rrtmgp+bulk campaign lane.

**C5 — restart staleness of `_sfc_diag` → FALSE POSITIVE.**
`_sfc_diag` is a Python instance attribute, not checkpointed. But a restarted
MPAS job performs radiation on local step 0 (model_driver.py:6642-6648) BEFORE
diagnostics/feed run, so the TOA extras are briefly *absent*, never *stale*, and
no normal diagnostic sample is missed. Same non-persistence as the pre-existing
sw_net/lw_net/precip slots, which have never caused a restart artifact.

## Round 1 (codex CONFIRMED — all fixed, see REPORT)

C1 rsdt→toa_insolation, NEW-1 HS wrapper drop, C2 MPI-voronoi 3-slot producer,
NEW-2 turbulence Field units="Pa". C6 tuple order verified correct (no change).
