# Adversarial review r2: volcanic SW aerosol per-band optics (READ-ONLY, do not edit)
Worktree /glade/derecho/scratch/pg2328/wt_volc, uncommitted diff vs HEAD (+ new files
tests/unit/test_volcanic_sw_optics.py, scripts/validate/probe_volcanic_sw_optics.py).
r1 findings and what was done:
- codex 1 (training SegmentForcing constructors drop the field): FIXED model_driver build_training_segment
  pack_forcing and scale_build pack_forcing now pass aerosol_sw_volc. (Note: neither passes aerosol_lw_od --
  pre-existing, out of scope, reported.)
- codex 2 (integration gray/simple_lw return before refusing): FIXED, guard before the returns
  (scheme != "rrtmgp" raises); tests for gray and simple_lw.
- codex 3 (band-mapping tests vacuous): ADDED single-band absorbing injection test vs the independent
  RRTMGPConfig.aerosol_ssa_bands grey path, bands 0 and 10, rtol 1e-10.
- codex 4 / GLM 2 (host interpolation memory): FIXED by column chunks of _VOLC_SW_COL_CHUNK=4096 in
  aerosol_layer_fields; test asserts chunk=2 result array-equal to unchunked. Subcolumn expansion: production
  deck uses McICA (cloud_vertical_overlap_optics: mcica), where expand_kwargs does not run; device field is
  (ncol, nlev, 3, 14).
- GLM 1 (GRID_SHAPED_FORCING_FIELDS): REFUTED by reading -- aerosol_lw_od (ncol,nlev) IS in the tuple; the
  flat-face sharding is P("face") on dim 0 for any ndim; shard_pytree has a shape gate.
- GLM 3 (overhead zeros only inside _zero(aerosol_od)): REFUTED -- the volcanic prepend is its own
  statement inside `if overhead:`, independent of aerosol_od.
- GLM 4: loaded[:4]+(loaded[4],) simplified? (cosmetic; say if you want it).
Measured: Jan 1992 equator clear-sky cosz 0.5, CAM6 L32, full tables: file optics vs grey 0.93/0.70:
TOA SW up +2.5 W/m2, surface SW down +7.3 W/m2, SW absorbed above 100 hPa 26.5 vs 39.6 W/m2 (none 25.0).
Tests: new file 19 passed; affected suites 203 passed, 5 failed = pre-existing on origin/main (3 cross_lane
fv3_duo refusals, 2 multilayer-land training/winds). Non-vacuity: reverting the two_stream mixing fails 3 tests.
VOTE (one line): column chunk size for the host interpolation: (A) 4096 module constant; (B) other (name it).
Review the CURRENT diff for remaining defects (same checklist as r1: call-path coverage, energy/double count,
AD/JIT/pytree, dtype, test vacuity, behaviour without volcanic file). Reply numbered findings with severity,
file:line, failure scenario; then VERDICT (APPROVE / HOLD).
