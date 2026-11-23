# Adversarial review r1: volcanic SW aerosol with the CMIP6 file's own per-band optics
READ-ONLY review of the uncommitted diff in /glade/derecho/scratch/pg2328/wt_volc (git diff vs HEAD,
plus new files tests/unit/test_volcanic_sw_optics.py, scripts/validate/probe_volcanic_sw_optics.py). Do not edit.

What changed: volcanic stratospheric SW aerosol used to be a solar-weighted GRAY extinction added
into `aerosol_od` and solved with the grey tropospheric SSA 0.93 / g 0.70. Now
`_load_volcanic_sw_optics_profile` (forcing/external.py) builds per-file-layer, per-band moments
(tau, tau*omega, tau*omega*g) from ext_sun/omega_sun/g_sun, bands sorted ascending wavenumber and
validated 1:1 against RRTMGP SW band limits (one documented 2600/2680 cm-1 edge difference);
`aerosol_layer_fields` (forcing/surface_utils.py) remaps each moment with the existing conservative
log-p overlap remap x tropopause weight to (ncol,nlev,3,14) `aerosol_sw_volc`, keeping volcanic OUT
of `aerosol_od`; new optional field threaded model_driver -> SegmentForcing/pack_forcing ->
physics_pipeline (step_unified kwarg captured by closure, compute_radiation_core, rrtmgp radiation_fn;
gray/zero radiation raise) and integration.py (forcing dict key `aerosol_sw_volc`) -> RRTMGP.solve_columns
(`aerosol_volcanic_moments_sw`, halos, overhead layer, subcolumn expansion key, column chunking)
-> two_stream `_mix_sw_aerosol_and_clip` adds the g-point's band moments to tau / tau*ssa / tau*ssa*g.
Grey ext_sun-only SW profile path deleted (a file with ext_sun but no omega/g now raises).
Voted detail choices: Q1 sorted 1:1 + edge validation (author A, codex C, GLM B); Q2 raise (codex, GLM);
Q3 float32 cache; Q4 None when absent; Q5 non-RRTMGP lanes raise; Q6 non-finite raises (moot: 0 NaN in
production file); Q7 rtol 1e-10 equivalence, closure |res| <= 1e-10*incident + 1e-10.

Check hard: (1) band mapping/order correctness incl. g_point_to_bnd; (2) every radiation call path that
receives aerosol_od also receives the new field (MPAS, cube, lat-lon, spectral, compiled, per-step,
operator-split SPMD, chunked, subcolumns, MC optics, clear-sky) -- name any that silently drop it;
(3) energy: SW budget closure, no double counting of volcanic in aerosol_od, CCN unchanged;
(4) AD/JIT: pytree stability (None vs array across segments), dtype under compute_fp32, gradient;
(5) tests vacuous? (6) memory of (ncol,nsrc=70,3,14) host interpolation at res6; (7) any silent
behaviour change for runs WITHOUT a volcanic file or with legacy column `aod` files.
Reply: numbered findings with severity (BLOCKER/MAJOR/MINOR), file:line, failure scenario; then VERDICT.
