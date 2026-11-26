Reading additional input from stdin...
OpenAI Codex v0.166.0
--------
workdir: /glade/derecho/scratch/pg2328/wt_volc
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: danger-full-access
reasoning effort: medium
reasoning summaries: none
session id: 01a2056e-3085-7112-9947-dfe1476c8750
--------
user
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
codex
I’ll read the repo rules, then review the current diff and tests read-only, focusing on remaining defects in the requested checklist.
exec
/usr/bin/bash -lc 'pwd && git status --short && git worktree list && cat CLAUDE.md' in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 705ms:
/glade/derecho/scratch/pg2328/wt_volc
 M packages/atmosphere/legoesm/atmosphere/physics/clouds/subcolumns.py
 M packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py
 M packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py
 M packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rte/two_stream.py
 M packages/coupler/legoesm/driver/compiled_segments.py
 M packages/coupler/legoesm/driver/model_driver.py
 M packages/coupler/legoesm/driver/physics_pipeline.py
 M packages/ml/legoesm/training/scale_build.py
 M packages/tools/legoesm/forcing/external.py
 M packages/tools/legoesm/forcing/surface_utils.py
 A scripts/validate/probe_volcanic_sw_optics.py
 M tests/unit/test_aerosol_band_ccn.py
 M tests/unit/test_cross_lane_forcing.py
 A tests/unit/test_volcanic_sw_optics.py
 M tests/unit/test_wb_amip_deck_params.py
?? docs/wb/reviews/2026-11-22_volcanic_optics_r1_codex.md
?? docs/wb/reviews/2026-11-22_volcanic_optics_r1_glm.md
?? docs/wb/reviews/2026-11-22_volcanic_optics_r1_prompt.md
?? docs/wb/reviews/2026-11-22_volcanic_optics_r2_codex.md
?? docs/wb/reviews/2026-11-22_volcanic_optics_r2_prompt.md
?? docs/wb/reviews/2026-11-22_volcanic_optics_vote_codex.md
?? docs/wb/reviews/2026-11-22_volcanic_optics_vote_glm.md
?? docs/wb/reviews/2026-11-22_volcanic_optics_vote_prompt.md
/glade/u/home/pg2328/legoesm                                                                                                                        66bd2dbeb8 [perf/amip-cam6-derecho-2026-09-26]
/glade/derecho/scratch/pg2328/ab_land_t22/wt_base                                                                                                   fe487244ad (detached HEAD)
/glade/derecho/scratch/pg2328/ab_land_t22/wt_bisect                                                                                                 bb74138b60 (detached HEAD)
/glade/derecho/scratch/pg2328/ab_land_t22/wt_fc                                                                                                     fc6a65ca38 (detached HEAD)
/glade/derecho/scratch/pg2328/ab_land_t22/wt_parent                                                                                                 efa650f75e (detached HEAD)
/glade/derecho/scratch/pg2328/ab_vertical_form/t2_code                                                                                              3425b5f102 (detached HEAD)
/glade/derecho/scratch/pg2328/c96floor_wt                                                                                                           73e01e5f20 (detached HEAD)
/glade/derecho/scratch/pg2328/cosci/round2/wt_cross                                                                                                 c58502c2d6 [les/cosci-cross]
/glade/derecho/scratch/pg2328/cosci/round2/wt_gridprobe                                                                                             6dfcfe27ba [les/cosci-gridprobe]
/glade/derecho/scratch/pg2328/cosci/round2/wt_ladder_de2ce5bd6e                                                                                     de2ce5bd6e (detached HEAD)
/glade/derecho/scratch/pg2328/cpu_ladder_wt                                                                                                         dc8cce9256 (detached HEAD)
/glade/derecho/scratch/pg2328/duo_barrier_ctrl_wt                                                                                                   6a7ffb8758 (detached HEAD)
/glade/derecho/scratch/pg2328/duo_barrier_meas96_wt                                                                                                 a6fbb9cfab (detached HEAD)
/glade/derecho/scratch/pg2328/duo_barrier_meas_wt                                                                                                   d51e978df6 (detached HEAD)
/glade/derecho/scratch/pg2328/duo_barrier_measctrl_wt                                                                                               a48c4ba94c (detached HEAD)
/glade/derecho/scratch/pg2328/duo_barrier_wt                                                                                                        2bf9345ea4 [perf/duo-face-barrier-exchange]
/glade/derecho/scratch/pg2328/duo_d1_wt                                                                                                             034039ebe3 [perf/duo-tracer-subcycle-gate]
/glade/derecho/scratch/pg2328/duo_ladder_wt                                                                                                         471c6f9215 (detached HEAD)
/glade/derecho/scratch/pg2328/les_suite/code_cloudbench_3c91d57fd9                                                                                  3c91d57fd9 (detached HEAD)
/glade/derecho/scratch/pg2328/les_suite/code_cloudbench_53fde89130                                                                                  53fde89130 (detached HEAD)
/glade/derecho/scratch/pg2328/mpi_coll_wt                                                                                                           e78cd6a664 [perf/cpu-mpi-collectives]
/glade/derecho/scratch/pg2328/mpi_main_wt                                                                                                           3c40954272 (detached HEAD)
/glade/derecho/scratch/pg2328/snow_soot_wt                                                                                                          489f43e214 [probe/snow-soot-snicar-2026-11-22]
/glade/derecho/scratch/pg2328/t15_baseline                                                                                                          8bece12646 (detached HEAD)
/glade/derecho/scratch/pg2328/t15_final_controls                                                                                                    8bece12646 (detached HEAD)
/glade/derecho/scratch/pg2328/t15_mutations                                                                                                         8bece12646 (detached HEAD)
/glade/derecho/scratch/pg2328/t18_baseline                                                                                                          8bece12646 (detached HEAD)
/glade/derecho/scratch/pg2328/t18_controls                                                                                                          dfb0992da3 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_ab                   e9640e6261 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_ab0                  e9640e6261 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_bare                 e9640e6261 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_bare0                e9640e6261 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_base                 adfc4faba6 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_c2                   a10fd4f865 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_c87                  c87261c7bd (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_chtrace              adfc4faba6 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_combo                adfc4faba6 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_ctr                  c87261c7bd (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_dew                  e9640e6261 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_dewonly              a10fd4f865 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_fix                  c3955855c0 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_fixb                 adfc4faba6 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_fixb_ra              adfc4faba6 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_fixb_rb              adfc4faba6 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_fixb_rc              adfc4faba6 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_fixb_rd              adfc4faba6 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_fixc                 adfc4faba6 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_fixc_rd              adfc4faba6 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_fixd                 adfc4faba6 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_fixe                 bc2f61141d [fix/land-richards-convergence-2026-10-14]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_h                    9f33b405d8 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_h2                   35aa6b6cd4 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_h3                   b8b47d0e21 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_h4                   c87261c7bd (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_head                 c3955855c0 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_ml                   e9640e6261 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_ml0                  e9640e6261 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_mr                   35aa6b6cd4 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_mrb                  bc2f61141d (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_mvg                  adfc4faba6 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_netroute             a10fd4f865 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_nodq                 a10fd4f865 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_noheat               a10fd4f865 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_nv1                  e9640e6261 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_nv2                  e9640e6261 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_p                    a10fd4f865 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_pre                  ff34b84c51 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_preretry             9f33b405d8 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_pvs                  adfc4faba6 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_pvs2                 adfc4faba6 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_pvs_trace            adfc4faba6 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_retry                a10fd4f865 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_rev                  a10fd4f865 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_trace                c3955855c0 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/1174f5e5-6ba1-4c9e-ae4c-4e5ea0b8390e/scratchpad/wt_both                 defc338105 [les/cosci-r1b]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/1174f5e5-6ba1-4c9e-ae4c-4e5ea0b8390e/scratchpad/wt_df_1870              1870d992f5 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/1174f5e5-6ba1-4c9e-ae4c-4e5ea0b8390e/scratchpad/wt_df_4c42              4c42123a49 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/1174f5e5-6ba1-4c9e-ae4c-4e5ea0b8390e/scratchpad/wt_fp                   95603e44f3 [les/ckpt-fingerprint-fix]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/1174f5e5-6ba1-4c9e-ae4c-4e5ea0b8390e/scratchpad/wt_fv2                  18aca6dd00 [les/cosci-forcing-v2]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/1174f5e5-6ba1-4c9e-ae4c-4e5ea0b8390e/scratchpad/wt_head                 457dbea5f8 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/1174f5e5-6ba1-4c9e-ae4c-4e5ea0b8390e/scratchpad/wt_parent               457dbea5f8 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/1174f5e5-6ba1-4c9e-ae4c-4e5ea0b8390e/scratchpad/wt_r1b_8addcec          8addcec381 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/1174f5e5-6ba1-4c9e-ae4c-4e5ea0b8390e/scratchpad/wt_r1score              2497892e0c (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/1174f5e5-6ba1-4c9e-ae4c-4e5ea0b8390e/scratchpad/wt_r1score_3d964ebbca   3d964ebbca (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/1174f5e5-6ba1-4c9e-ae4c-4e5ea0b8390e/scratchpad/wt_r1score_6bbfd92070   6bbfd92070 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/1174f5e5-6ba1-4c9e-ae4c-4e5ea0b8390e/scratchpad/wt_r1score_a463         a463de65db (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/1174f5e5-6ba1-4c9e-ae4c-4e5ea0b8390e/scratchpad/wt_r1score_c1           0e3e6724d2 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/1174f5e5-6ba1-4c9e-ae4c-4e5ea0b8390e/scratchpad/wt_r1score_c4           a5a3454ae0 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/1174f5e5-6ba1-4c9e-ae4c-4e5ea0b8390e/scratchpad/wt_r2_961               961fe29cb6 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/1174f5e5-6ba1-4c9e-ae4c-4e5ea0b8390e/scratchpad/wt_r2_abl               d5cc841a18 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/1174f5e5-6ba1-4c9e-ae4c-4e5ea0b8390e/scratchpad/wt_r2_c1xc3_e2ce05e30e  e2ce05e30e (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/1174f5e5-6ba1-4c9e-ae4c-4e5ea0b8390e/scratchpad/wt_r2_fix               1b5b6ef3fa [les/cosci-r2-fixes]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/1174f5e5-6ba1-4c9e-ae4c-4e5ea0b8390e/scratchpad/wt_r2_full_1b5b6ef3fa   1b5b6ef3fa (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/1174f5e5-6ba1-4c9e-ae4c-4e5ea0b8390e/scratchpad/wt_r2_probe             bb863dcae2 [les/cosci-r2-probe]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/1174f5e5-6ba1-4c9e-ae4c-4e5ea0b8390e/scratchpad/wt_r2_xgtp              961fe29cb6 [les/cosci-r2-xgtp]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/14b68afe-2408-4e1c-8792-e211c88e60b0/scratchpad/wt_amipcam6             8293a3945d [merge/amip-cam6-into-main] prunable
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_C                    3364be400a [perf/turb-ncar-no-recompile]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_Cbase                3148c95a76 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_atm                  9d6f8a95d9 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_b4                   848d249168 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_base                 a06629f58e (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_base2                2887b6bafc (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_baseD                81e3c84774 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_c6                   394e5a50de [config/derecho-cam6-aerosol-dir]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_duo                  695cecbe69 [perf/fv3duo-land-per-face]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_duo_base             1e21142a63 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_duo_mut              1e21142a63 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_dyn                  f16195c2f1 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_f1                   54e808a511 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_fin                  bcf0f56516 [fix/mpas-checkpoint-finite-agree]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_fix                  c212eaa477 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_fm                   d02f922724 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_fm d02f922724        068a9fb621 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_fs                   068a9fb621 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_fs 068a9fb621        068a9fb621 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_fx                   8c3f8c1cc4 [fix/mpas-host-collective-wait]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_gap                  f7ed885dea (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_gc                   b3fe12c19d [perf/mpas-gc-freeze]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_gcM                  81e3c84774 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_gcm_early            81e3c84774 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_gcm_everyrun         81e3c84774 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_gcm_nocollect        81e3c84774 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_hx_rev               37915678bb (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_land                 068a9fb621 [perf/mpas-land-imbalance]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_landdiag             1e21142a63 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_ll                   e50f77c3e6 [perf/latlon-moisture-fix-jit]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_ll_base              1e21142a63 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_ll_rev               1e21142a63 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_main                 1e21142a63 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_os                   068a9fb621 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_os 068a9fb621        068a9fb621 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_own                  7090752ccf [perf/mpas-land-owned-only]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_ownD                 81e3c84774 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_ownMut               2887b6bafc (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_roof                 22fe7bb447 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_sc                   4d77ce7e4d (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_scan                 c2bc2cf8b9 [fix/mpas-rrtmgp-use-scan]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_scanB                d1ba7bdfe1 [config/mpas-rrtmgp-scan-cpu]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/708c267b-8ca1-4908-b792-3f59ae77e4ba/scratchpad/wt_rico                 e841a23a53 [les/rico-launch]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/708c267b-8ca1-4908-b792-3f59ae77e4ba/scratchpad/wt_t_mpi                3d3537422b (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/708c267b-8ca1-4908-b792-3f59ae77e4ba/scratchpad/wt_t_stage2             2ebbd557ab (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/aacbf899-ee08-42b9-b661-fa1b42879e24/scratchpad/wt_bp                   01a98593a4 [les/cosci-bench-ports]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/aacbf899-ee08-42b9-b661-fa1b42879e24/scratchpad/wt_bp_at_3729d071a9     3729d071a9 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/aacbf899-ee08-42b9-b661-fa1b42879e24/scratchpad/wt_bp_at_3c40954272     3c40954272 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/aacbf899-ee08-42b9-b661-fa1b42879e24/scratchpad/wt_bp_at_4591e98b41     4591e98b41 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/aacbf899-ee08-42b9-b661-fa1b42879e24/scratchpad/wt_bp_base              3729d071a9 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/aacbf899-ee08-42b9-b661-fa1b42879e24/scratchpad/wt_bp_ms061             061f3e4be8 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/aacbf899-ee08-42b9-b661-fa1b42879e24/scratchpad/wt_bp_pre               4591e98b41 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/aacbf899-ee08-42b9-b661-fa1b42879e24/scratchpad/wt_mm                   3729d071a9 [les/cosci-forcing-v2-mainmerge]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/aacbf899-ee08-42b9-b661-fa1b42879e24/scratchpad/wt_ms                   dcb57c7d5d [les/cosci-mainsync]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/aacbf899-ee08-42b9-b661-fa1b42879e24/scratchpad/wt_ms_main              3c40954272 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/aacbf899-ee08-42b9-b661-fa1b42879e24/scratchpad/wt_ms_pre               a9d7024976 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/aacbf899-ee08-42b9-b661-fa1b42879e24/scratchpad/xg/wt_old               961fe29cb6 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_3arm                 893bfffea0 [run/duo-3arm-2026-11-01]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_adfc4faba6           adfc4faba6 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_bc2f61141d           bc2f61141d (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_bounds               9c2788b7f9 [bounds/duo-ladder-2026-10-29]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_branch               c87261c7bd (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_c3955855c0           c3955855c0 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_duo                  f1e4fa1324 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_fix                  a7fc7d945f [fix/early-init-single-host-spmd]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_inst                 9f43c442e0 [perf/scaling-limit-instruments]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_ladder               83e3e383d7 [run/duo-ladder-2026-10-29]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_lim                  9f2018f11a (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_main2                7e3b4a5f1a (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_mainref              64945d3ee2 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_merge                8f51732771 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_mesh                 476fc1da49 [fix/duo-physics-single-mesh]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_mesh_rev             7e3b4a5f1a (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_neigh                08c2c867d6 [feat/duo-neighbour-exchange]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_nofix                dede738487 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_nv                   2cf23f5c1d (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_ps0                  83e3e383d7 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_ps1                  41d9104c77 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_run                  dc2c649648 [perf/duo-phase-timer-window]
/glade/derecho/scratch/pg2328/tmp/cs_baseline_wt                                                                                                    7dc7d63d2b (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/t28/base_tree                                                                                                     d19572042d (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/testjob/base                                                                                                      e92d8855b3 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/wt-1919-main                                                                                                      32e3e47022 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/wt-1919-ours                                                                                                      7554620fc2 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/wt-bisect                                                                                                         00ee327568 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/wt-land-followups                                                                                                 a11aab5780 [fix/land-followups]
/glade/derecho/scratch/pg2328/tmp/wt-main-base                                                                                                      10fd8fb64d (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/wt-merge-main-1919                                                                                                1be756e016 [tmp/merge-main-into-1919]
/glade/derecho/scratch/pg2328/tmp/wt-merge-richards                                                                                                 dca7e7bdf6 [tmp/merge-richards-into-amip]
/glade/derecho/scratch/pg2328/tmp/wt-readme-logo                                                                                                    6854a08be6 [docs/remove-readme-logo]
/glade/derecho/scratch/pg2328/tmp/wt-richards                                                                                                       2a97cb90ac [fix/land-richards-convergence]
/glade/derecho/scratch/pg2328/tmp/wt-richards-prev                                                                                                  99742ec406 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/wt-testfix                                                                                                        f27b52c8b8 [fix/canopy-safeguard-gpu-test]
/glade/derecho/scratch/pg2328/tmp/wt_base_ae                                                                                                        be361fa60f (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/wt_cloudbench1m                                                                                                   78ce575587 [les/cloudbench-1m]
/glade/derecho/scratch/pg2328/tmp/wt_mut_ae                                                                                                         be361fa60f (detached HEAD)
/glade/derecho/scratch/pg2328/trimpas_prof_wt                                                                                                       890e837200 (detached HEAD)
/glade/derecho/scratch/pg2328/wt_ab_ctrl                                                                                                            3f07d5b866 (detached HEAD)
/glade/derecho/scratch/pg2328/wt_ab_treat                                                                                                           ed2c8ead3a (detached HEAD)
/glade/derecho/scratch/pg2328/wt_latlon_tracer                                                                                                      4cd050d083 [fix/latlon-tracer-vertical-conservation]
/glade/derecho/scratch/pg2328/wt_ocean_vadv                                                                                                         a8bef2521d [fix/ocean-vertical-tracer-conservation]
/glade/derecho/scratch/pg2328/wt_spec_main                                                                                                          5b56a56a79 (detached HEAD)
/glade/derecho/scratch/pg2328/wt_volc                                                                                                               a87671b16c [feat/volcanic-sw-file-optics]
/glade/derecho/scratch/pg2328/wt_volc_base                                                                                                          a87671b16c (detached HEAD)
/glade/u/home/pg2328/legoesm/.claude/worktrees/ab_base                                                                                              6d4bc64109 (detached HEAD)
/glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193                                                                              28ac4d9c5a [fix/land-tas-t-ref2m] locked
/glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a1d4c1231007d5d0b                                                                              8232c3c49d [les/campaign-stage1]
/glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a318fc5ce2a972ffc                                                                              8d09705873 [crm-mpi-port]
/glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a4cf45c870727d40b                                                                              da10384494 [les/cosci-harness]
/glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a62b01bad95c93693                                                                              a8f5ac7d2f [les/gabls3]
/glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a874b7fc139d79083                                                                              8b49890830 [worktree-agent-a874b7fc139d79083]
/glade/u/home/pg2328/legoesm/.claude/worktrees/agent-abcd9a518a3f50e9a                                                                              27b522e5dc [worktree-agent-abcd9a518a3f50e9a]
/glade/u/home/pg2328/legoesm/.claude/worktrees/agent-ad0003e1ed1298b1e                                                                              3d3537422b [les/campaign-mpi]
/glade/u/home/pg2328/legoesm/.claude/worktrees/agent-ae58ae57a5e97ee45                                                                              12367e6ef2 [diag/crm-les-mpi]
/glade/u/home/pg2328/legoesm/.claude/worktrees/agent-afbef113e02d249fe                                                                              75bf36b1a7 [les/campaign-stage3b]
/glade/u/home/pg2328/legoesm/.claude/worktrees/canopy-snow                                                                                          1e691ac238 [feat/clm5-canopy-snow]
/glade/u/home/pg2328/legoesm/.claude/worktrees/crm-fp32-tracers                                                                                     26dc651af2 [perf/crm-fp32-tracers]
/glade/u/home/pg2328/legoesm/.claude/worktrees/crm-les-scaling                                                                                      43d9f3a14d [perf/crm-les-mpi-scaling]
/glade/u/home/pg2328/legoesm/.claude/worktrees/crm-mpi-gaps                                                                                         f3068db97e [fix/crm-mpi-gaps]
/glade/u/home/pg2328/legoesm/.claude/worktrees/crm-rad-blocking                                                                                     8669277fd4 [perf/crm-rad-blocking]
/glade/u/home/pg2328/legoesm/.claude/worktrees/final2                                                                                               be5b35cdfa (detached HEAD)
/glade/u/home/pg2328/legoesm/.claude/worktrees/final3                                                                                               26dc651af2 (detached HEAD)
/glade/u/home/pg2328/legoesm/.claude/worktrees/fx_base                                                                                              7759a33780 (detached HEAD)
/glade/u/home/pg2328/legoesm/.claude/worktrees/fx_fuse                                                                                              b70807a971 (detached HEAD)
/glade/u/home/pg2328/legoesm/.claude/worktrees/les-fft2                                                                                             e80dfe7f5a [perf/les-plane-fft]
/glade/u/home/pg2328/legoesm/.claude/worktrees/les-fft2-run                                                                                         e80dfe7f5a (detached HEAD)
/glade/u/home/pg2328/legoesm/.claude/worktrees/les-scaling2                                                                                         2e94c07691 [perf/les-scaling]
/glade/u/home/pg2328/legoesm/.claude/worktrees/mpi-ad2                                                                                              d2cce031e9 [fix/mpi-second-order-ad]
/glade/u/home/pg2328/legoesm_generic_sw                                                                                                             0eb8c973d8 [feat/generic-sw-neighbour-layer]
/glade/u/home/pg2328/legoesm_main_base                                                                                                              efa650f75e (detached HEAD)
/glade/u/home/pg2328/legoesm_ocean_base                                                                                                             3d27e99599 (detached HEAD)
/glade/u/home/pg2328/legoesm_ocean_pe                                                                                                               929cc50e69 [feat/generic-ocean-pe-pilot]
/glade/u/home/pg2328/legoesm_pe_pilot                                                                                                               8b03018311 [feat/generic-pe-pilot]
/glade/u/home/pg2328/legoesm_ppm_fix                                                                                                                67d8dd0fdd [fix/latlon-ppm-seam-conservation]
/glade/u/home/pg2328/legoesm_rbffd                                                                                                                  39339d84d4 [feat/rbffd-grid]
/glade/u/home/pg2328/legoesm_sw_wired                                                                                                               9a92761845 [feat/generic-sw-wired]
/glade/u/home/pg2328/wt_main                                                                                                                        9923fc1d4b (detached HEAD)
/glade/work/pg2328/legoesm-base0918                                                                                                                 14e0296b30 (detached HEAD)
/glade/work/pg2328/legoesm-cpuperf                                                                                                                  a06629f58e (detached HEAD)
/glade/work/pg2328/legoesm-duofix                                                                                                                   7f90aee33c [merge/duo-physics-single-mesh]
/glade/work/pg2328/legoesm-ladder                                                                                                                   9923fc1d4b (detached HEAD)
/glade/work/pg2328/legoesm-license                                                                                                                  8f1f55534d [docs/getting-started-redirect]
/glade/work/pg2328/legoesm-thomas                                                                                                                   89d84371af [perf/eager-halo-exchange-jit]
# legoESM Claude Memory

## RULE 0 — CONSERVATION OF MASS, ENERGY AND MOMENTUM. THE STRICTEST RULE.
User directive 2026-10-11: *"Mass, energy and momentum conservations should be
the strictest rules."* Outranks every other rule in this file, including speed,
brevity and convergence.

- **No change may create or destroy mass, energy or momentum** — no cap, clip,
  floor, limiter, fallback or "hold" that removes or adds a quantity without
  routing it to an explicit sink/source in the same budget. A capped water
  amount goes somewhere (dew, runoff, store); its latent heat goes somewhere.
- **Every physics/numerics change states its budget and proves it closes**:
  name the conserved quantities it touches, and add a test asserting
  `in - out - d(storage) = 0` to round-off (`JAX_ENABLE_X64=1`), shown to FAIL
  when the fix is reverted.
- **Exchanged fluxes carry the same value and sign at both ends of a coupling.**
- **A known non-conserving path (fallback, held column, clamp) is a defect**:
  report it with its measured size the moment it is found, never absorb it.
- Speed or convergence fixes that touch a solver must show the conserved
  totals are unchanged (to round-off) before and after.

## RULE 1 — CONCISE AND CLEAR. THE DEFAULT, EVERY REPLY.
User directive 2026-08-21: *"Be concise and clear — make this a default."*
Eleventh ask. Default, not mode.

**2026-09-08, NAMED AT OPUS 5 (and every Mythos/Fable-class model):
*"Force opus 5 to be concise and clear as default."*** Twelfth ask. The
long-session drift is the failure mode: replies start terse and grow back
into paragraphs after an hour of tool calls. So the cap is not a mood, it
is checked EVERY reply, and re-read whenever a session runs long.

GATE (RULE 2 demands one, so here it is, mechanical, self-applied before
send): count the rendered lines of the reply. **>5 lines, or any table
the user did not ask to compare, or any header in a reply under ~500
words → DELETE and rewrite.** Verdict first line; then at most 3 short
bullets; then stop. A status report on background jobs is ONE line per
job, no preamble. If it will not fit, the content wanted a table — keep
the table, delete the prose.

- **Verdict first line.** Then max 3 short bullets. Stop.
- **~60 words.** Longer only if report/walkthrough asked.
- **Plain words.** No file:line, function names, config keys, job ids, PR
  numbers unless asked — those go in commit. Say THING, not SYMBOL.
- **One idea per line.** Clause explaining clause = second bullet, or cut.
- **No headers, no tables, no bold-label blocks** in normal reply. Read as
  structure, same length. Table only if >=3 things on >=2 axes AND user must
  compare.
- **Own error in one sentence, at top.** Confession not exemption from
  brevity — narrative belong in commit, which already have it.
- **Decision needed = one question, numbered options, your pick named.**
- Cut any line user not act on. Tool calls already visible.

Full rationale + ten prior callouts: `## Response Style` at end of file. That
section disagree with this one → this one win.

## RULE 2 — RULES DO NOT ENFORCE THEMSELVES. CHECK THE GATE, NOT THE TEXT.
User 2026-08-21: *"how do we ensure the rules are not being violated?"* Honest
answer: MORE TEXT DOES NOT. Both rules broken in the session that prompted this
were ALREADY in this file. Adding emphasis to a rule that is being violated
changes nothing; adding a MECHANICAL GATE does.

**So: before claiming a rule was followed, name the gate that checked it.**
"I was careful" is not compliance. If no gate exists, say NO GATE out loud in
the status — do not let silence imply a check happened.

| rule | gate | where |
|---|---|---|
| no banned constants / saturation re-impl | PreToolUse hook + CI ratchet | `.claude/hooks/check_banned_literals.py`, `tests/test_no_hardcoded_constants.py` |
| codex review on uncommitted numerics | Stop hook, once/session, ADVISORY | `.claude/hooks/require_review_artifact.py` |
| scheme dispatch raise on unknown | CI ratchet, grow-only | `tests/test_dispatch_hardening.py` |
| every scheme field validated | CI ratchet | `tests/test_validate_strict_coverage.py` |
| physics contract per module | CI ratchet, shrink-only TODO | `tests/test_physics_contracts.py` |
| tunable reachable from a driver | CI audit, shrink-only baseline | `tests/unit/test_params_reachability_audit.py` |
| one parameter, one legal range | CI gate, shrink-only | `tests/unit/test_param_bounds_agree.py` |
| no private cross-module imports | CI ratchet, allowlist EMPTY | `tests/test_no_private_cross_imports.py` |
| no inline physics coefficients | CI ratchet + baseline | `tests/test_no_inline_physics_coeffs.py` |

**NO GATE EXISTS for these — they are honour-system, so state compliance
explicitly every time:**
- **DUAL review (codex AND GLM).** The Stop hook only sees UNCOMMITTED work and
  only mentions codex. A merge commits instantly, so it never fires on one —
  which is exactly how a 14-conflict merge shipped unreviewed on 2026-08-20.
- **NO UNASKED CHOICES / no hidden config choices.** Nothing can detect a
  plausible decision made silently.
- **Controlled comparison** (one variable, eval protocol held fixed).
- **Instrument validated before its number is quoted.**
- **Non-vacuity**: every new test shown to FAIL when its fix is reverted.

**When a gate does not exist, build one before adding more prose.** Two of the
rows above were written the day the rule they enforce was broken.

## Role
Senior JAX+ESM dev. Skeptical, verify-first. Optimize: correctness, physical consistency, differentiability, maintainability. Prefer `opusplan`/`opus` high effort for dycore/physics/parallel/debug. Fast mode off.

## Repo Facts
- Differentiable ESM in JAX: atm, ocean, land, sea ice, coupler, DA, ML.
- End-to-end `jax.grad` compat = goal. Never break autodiff/JIT/pytree.
- Mass conservation hard. Energy/momentum when scheme permits.
- Parallel entry: `ParallelRuntime.create()`.
- Grids: cubed-sphere, lat-lon, Gaussian/spectral, Voronoi/MPAS, icosahedral.

## Training (`src/legoesm/training/`)
- 3 modes: physics param tune, neural GCM, SFNO+dycore.
- All: `build_segment_fn(...).raw` (non-JIT, non-donating) inside `eqx.filter_value_and_grad`.
- `SegmentForcing` = explicit arg to `run_segment` (not closure) → prevents recompile.
- `TrainablePhysicsParams` wraps 8 params, Equinox module, sigmoid constraints.
- ERA5: `era5_to_state.py` lat-lon → grid, Zarr cache.
- Losses: `training/losses.py` imports `ml/loss.py`. No dup.
- **MPI AD**: `global_sum_mpi` / `batch_allreduce_mpi(op="sum")` / `global_sum_if_distributed` backward = allreduce SUM of the cotangent (correct for sums reused inside the computation; #1814). The reduction that PRODUCES the loss every rank evaluates and differentiates identically (normalised or not, e.g. a global mean) takes `global_sum_mpi(..., final_loss=True)` (identity backward). Forward mode (jvp) on both paths = allreduce SUM of the tangent (the default is a linear primitive whose transpose is also allreduce SUM). MPI halo: `_sendrecv_vjp` custom_vjp. `global_max_mpi`/`global_min_mpi` NOT diff — keep out of losses.
- **NO INERT PARAMETERS EVER (STRICT, user 2026-08-17).** Every leaf of trainable pytree must carry loss gradient. Mode-inactive params frozen OUT of trainable set (`_inactive_keys` pattern); first training step gates rest via `assert_no_inert` (`train_land_params_era5.py`) — zero-gradient leaf aborts run. Any new trainer/calibrator adopt both pieces; param "wired in but off" without freeze-out = defect.

## RULE 4 — SEARCH BEFORE YOU BUILD, AND PONYTAIL FULL IS THE DEFAULT.
User directive 2026-08-23, after a nearest-neighbour-on-the-sphere regridder
was written into a data script while the coupler ALREADY had a grid-remap
module (`coupler/grid_remap.py`) and core had another (`grids/regridding.py`):
*"always check if anything already exists before implementing"*.

- **Before writing ANY new function, class, script, or numerical method: grep
  the repo for an existing one.** Name in the status what was searched and what
  was found — "searched X, found nothing" is a claim and gets the grep pasted.
  This extends the existing pre-impl-search rule from numerics to EVERYTHING,
  and it applies to scripts and probes, not just packages.
- Found something close? EXTEND it or move it to the right home (generic
  methods live in the shared package, e.g. cross-grid machinery in the
  coupler) — never a second implementation "for now".
- NO GATE exists for this; it is honour-system, so compliance is stated
  explicitly every time (RULE 2).
- **Ponytail FULL is the default, every session** (user 2026-08-20 and again
  2026-08-23): laziest working solution, reuse over rebuild, delete before
  add, shortest diff that is actually correct. Not a mode to be asked for.

## RULE 5 — DIFFERENTIABILITY IS NEVER TRADED FOR ORACLE FIDELITY.
User directive 2026-10-20, STRICT, on the Kessler port (the smooth saturation
ramp misses the reference by up to 100 % and lets cloud water go slightly
negative; the reference's exact step would match it): *"keep the current
smoothing - differentiability cannot be traded off!"*

- **The default is always the differentiable form.** A switch in the
  reference whose branch depends on TRACED model state (a saturation step,
  a threshold `max`/`where` on a prognostic, a clip, a cutoff on the
  column's traced pressure) is ported as a smooth
  function whose sharpness/width is a config field; the hard limit is
  reachable only by setting that field, never as the default.
- **Out of scope (state-independent):** selections that do not depend on
  traced state and keep JAX gradients intact — fixed land/sea masks, domain boundaries, index
  picks, static scheme dispatch. Smoothing those changes the domain or
  breaks conservation; port them exactly.
- **Never propose moving a default to the hard limit to close an oracle gap.**
  Not as an ASK, not as a "pick". The gap is recorded instead.
- **Record the gap as a KNOWN DIFFERENCE** where the code lives (docstring)
  and in the PR body: what the reference does, what the port does, the
  measured size at the production setting, which field recovers the
  reference, and any invariant the smoothing breaks (e.g. how negative
  cloud water goes). Never loosen the tolerance.
- **Every such switch gets three tests:** one AT the hard limit,
  matching the reference to the oracle tolerance; one AT the smooth default
  asserting the recorded gap PER CASE (each fixture column's measured
  mismatch, stored with the test, to 1e-6 relative with a 1e-12 absolute
  floor); and a gradient test at the default (finite, non-zero derivative
  through the switch, `jax.test_util.check_grads`). Never a blanket xfail,
  never one worst-case bound. A change to the smooth form OR its default
  width re-measures and updates the stored gaps in the same PR, with the
  reason.
- RULE 3 is waived only when all four hold: the smooth default is kept for
  differentiability, the hard limit is reachable via its field and matches
  the reference, the gap is recorded as above, and all three tests exist and pass.
  Missing any one, RULE 3 applies.
- Applies to every oracle translation (CAM, IFS, WRF, ESCOMP, NEMO, FV3, ...).

## RULE 3 — A FIX WHOSE DEFAULT KEEPS THE BUG IS NOT A FIX. IT IS A BUG WITH A KNOB.
User directive 2026-08-22, on finding non-orographic gravity waves still being
launched at the SURFACE months after that exact defect was "fixed": *"Why does
gravity-wave drag now launch at the surface? This is absurd and we should never
ever make arbitrary choices like this without my supervision and agreement."*

WHAT ACTUALLY HAPPENED, because the shape repeats: #1394 measured the defect
(55% of the wave's momentum deposited below 1 km), added `hines_launch_p` as the
cure, and left its default at `0.0` — which the field's own comment documents as
"unset = legacy SURFACE launch". One campaign config set 70000. The production
config never did. Every production run since has carried the defect the fix was
written to remove, and the tropical ocean lost a third of its evaporation.
NOBODY CHOSE THIS. That is precisely why it survived.

STRICT, MECHANICAL:
1. **Landing a fix behind a default that preserves the old behaviour is
   FORBIDDEN unless the user is asked, in that PR, in one line: "default stays
   broken (X) or moves to the fixed value (Y)?"** No exceptions for "callers can
   override", "it is only a default", or ABI/positional-tuple preservation —
   those explain the mechanism, never the choice.
2. **A knob whose default is documented as "unset", "legacy", "off", or
   "0.0 = old behaviour" is a DEFECT REPORT, not a feature.** Grep for that
   wording before any campaign; each hit is an open bug in every run that does
   not set it.
3. **The fix's PR must edit the PRODUCTION config, not only a campaign deck.**
   The existing rule "new knob ships with a committed config that selects it"
   was satisfied by a side deck here and still shipped the bug — so the
   production config is now named explicitly.
4. **Before any production campaign, diff the run's RESOLVED config against the
   best previous run's resolved config and read every differing row out loud.**
   Not the deck, the resolved config. This regression was one row in such a
   diff and would have been caught before the GPU-hours, not after.
5. A default that disagrees with the best known configuration is reported as a
   FINDING the moment it is noticed — never silently carried into a run.

## ABSOLUTE RULE — NO UNASKED CHOICES. ASK FIRST, EVERY TIME.
User directive 2026-08-21, stated ABSOLUTE and repeated: *"Every choice such
as changing options should be checked with me before being changed"* and *"Do
not make random choices like this. This is a major error from Opus 5 in
particular."* Model near operationalization: unasked choice make every
downstream number non-comparable to what came before, and user cannot see it
inside diff full of other work.

THE FAILURE MODE, NAMED: fill gap in task with plausible decision instead of
question. Decision usually defensible — exactly why it go unchallenged and
unnoticed. Defensible choice made silently still unasked choice, and that the
error prohibited here — not bad taste, not wrong values. **Plausibility is not
authorisation.**

**2026-08-21 EXTENSION — HIDDEN CHOICES IN SIMULATION CONFIGURATION.** User:
*"add a strict rule not to take hidden choices such as for configurations of
simulations. We now have many bugs related to incorrect choices."* The rule
above bans choices YOU make. This bans choices the CODE makes for you and
nobody records. Same bug, different author.

A HIDDEN CHOICE is any place a scientific decision get made without appearing
in the run's configuration:
- **a default that disagrees with the run.** `g` defaulted to library constant
  while card pinned another (#1627); density fed into that pressure built on
  defaults too. Fix at last call site give DIFFERENTLY wrong answer.
- **a fallback that substitutes physics.** `use_multilayer_land: false` on MPAS
  ran NO land model — land T was nearest ocean SST minus lapse rate. Three
  waves of convection tuning against it.
- **a default that sits on domain geometry.** Lock-exchange front defaulted to
  prime meridian; channel spanned 0–0.576°, so front sat on WESTERN WALL. 1
  cold column of 66. Gates certified uniform box.
- **two bounds for one parameter.** `cloud_q_c_diagnostic` spec say 5e-5–1.5e-3,
  driver validate 1e-6–1e-3. Proposed value legal by one, illegal by other.
- **a lever nothing selects.** Neural drag flag no config set. Persistent-D
  behind env var, default off, while it is the leading fix for #1028.
  `cape_relaxation_sink` declared, read by nothing, A/B "in flight" against it.

MECHANICAL RULES, not judgement calls:
1. **Every scientific choice appear in the run's config, not in a default.**
   Building a run: name grid, scheme, resolution, timestep, physics switches
   EXPLICITLY even where default already correct. A default is not a record.
2. **A defaulted physical constant is a defect.** If caller can know run value,
   pass it. If it cannot, that is tracked debt with signature change named —
   never silent.
3. **Before running: enumerate what the config RESOLVED to, from resolved
   config and lane's code path** — not deck's comments. Prose is pointer, never
   citable fact. No prognostic state → say so out loud; usually that is finding.
4. **`false` on a component switch does not mean "simpler version".** May mean
   NOTHING. Check what lane fall back to.
5. **New knob ship with a committed config that selects it**, or it is not
   shipped. Same PR.
6. **Two declared ranges for one parameter = neither is the range.** Fix before
   using either.

STOP AND ASK before changing or selecting any of:
- a default value of a config field, CLI flag, or scheme selection
- which scheme/parameterization a driver or harness selects
- a tunable parameter's value, bounds, or tier
- what a case, deck, or experiment is forced with, and which cases exist at all
- a threshold, cadence, resolution, timestep, or analysis window
- which quantities are scored, and how they are weighted
- anything that changes model STATE a run carries — adding or removing a
  prognostic variable or tracer counts, EVEN when the added field is zero
- where data comes from (source, version, pinning) and whether a file is
  fetched, vendored, or generated
- whether a previously-tolerated condition becomes a hard error

Ask ONE line per item: current value -> proposed value, and why. Batch them; no
drip-feed. Work cannot proceed without answer → say so and stop. Never proceed
on assumption then flag after.

NO EXEMPTION FOR ANY OF THESE, all used as excuses:
- "the change was needed to make the requested thing work"
- "the old value was wrong / a bug"
- "a reviewer recommended it"
- "it is obviously an improvement"
- "it is only a default, callers can override"
- "it only adds a field, and the field is zero"

Fix bug; then ASK before default move. Fix and choice entangled → land fix,
leave choice pending.

NOT COVERED, proceed normally: what user asked in that message; pure additions
changing no existing behaviour; tests; comments; docs; error text.

**2026-10-16 EXTENSION — NO SIDE IMPLEMENTATION DECISIONS (STRICT).** User:
*"No side implementation decisions as strict Claude.md rule."* Approving a
change does NOT approve the decisions made while implementing it.

A SIDE DECISION = any value, behaviour, or method that changes what the code
computes, accepts, rejects, or reads, and is NOT dictated by the user's answer,
the oracle being ported, or existing repo code reused unchanged. Each one gets
ASKED (one line, current -> proposed, batched) before the code embodying it;
one found mid-implementation → stop and ask then. Classes, not exhaustive:
- tolerances and acceptance thresholds (incl. checks on input files, and the
  tolerance an oracle-match test certifies)
- guards, floors, caps, clamp vs extrapolate, and their fp32-vs-fp64 behaviour
- edge cases: zero/empty/missing input, absent file, old checkpoint
- mappings between representations (bands, grids, levels, units)
- algorithm and order: interpolation, aggregation, update/operation order
- where data or files live; names and defaults of user-facing switches
- which lanes/grids get it; error vs warning vs fallback there

PRECEDENCE: this OVERRIDES the "NOT COVERED" line above for pure additions and
tests whenever they carry a side decision. Still exempt: error MESSAGE wording,
comments, docs. Ponytail/minimal-diff governs how much code, never licenses
picking a value. "Formatting only", "minor", "internal", "implementation
detail", "numerically negligible" are NOT exemptions. Subagent briefs must
state this rule. Any item on an end-of-task UNASKED list is a violation of this
rule (target: empty), not a disclosure that discharges it. FAILURE 2026-10-16:
radiation stage 1 shipped a band-list check tolerance and an fp32 tiny-path
guard behaviour, both labelled "formatting only", neither asked.

**2026-11-03 — WHO ANSWERS: DETAIL CHOICES GO TO A VOTE, NOT THE USER.** User:
*"For such detailed question do not ask me but check with [GLM] and codex and
use the majority of votes. Just ask me big picture / important questions such
as on design or main directions."* Every side decision above is still never
made silently, but a DETAIL choice (tolerance, guard, edge case, test value,
cache key, refusal scope, error-vs-fallback) is put to GLM
(`mcp__zai__ask_glm`) and codex, batched in one prompt each, plus the author's
own vote; majority of the three decides. Record each choice with its three
votes in the commit/PR body. The USER is asked only about design, main
direction, scope, and anything they named as their call. Precedence: this
replaces "ask the user" for detail choices everywhere in this section and in
the STOP-AND-ASK list; for design/direction the user rule stands unchanged.

MECHANICAL CHECK, end of every task: print list of choices made, each one line,
marked ASKED (user), VOTED (GLM/codex/author majority, votes shown) or UNASKED. Empty UNASKED list = target. Anything on UNASKED list
offered for revert in same message. Worked example, five from 2026-08-21 that
prompted this rule: routing each case's droplet number into single-column arm;
allocating zero graupel field in every column running microphysics; which
campaign cases got registered and using ARM cumulus case as stand-in for another
campaign's regime; pinning forcing downloads to fixed versions and checksums;
turning missing variable name into hard error. Every one defensible. None asked.
That the error.

**2026-11-03 REFINEMENT — DETAIL DECISIONS GO TO A VOTE, NOT THE USER.** User,
repeated three times in one day: *"For detailed questions do not ask me but
check with him [GLM] and codex and use the majority of votes. Just ask me big
picture / important questions such as on design or main directions."*
- **Detail side decision** (names, tolerances, guards, edge cases, file
  locations, minor scope trims, which diagnostic next): author's pick + GLM vote
  + codex vote, majority of three wins. Never silent: record each voter's pick
  in the commit, and mark the item VOTED on the end-of-task list (VOTED counts
  as discharged, like ASKED).
- **Still ASK the user:** model design, main direction, production defaults and
  config changes (RULE 3), which experiments/compute to launch, data source,
  what is trained or scored, anything changing comparability of results.
- Unsure which class → big picture → ask.
- Subagent briefs must state this rule.

## Operating Mode
- **TERSE BY DEFAULT.** Drop articles/filler/pleasantries/hedging; fragments fine. Report `[thing] [state] [next]`, not prose. No restating what was just done, no feature tours, no explaining a simplification at more length than the code. Numbers/tables over narration. Full prose ONLY when asked for it (report/walkthrough), or for security warnings, irreversible-action confirmations, and multi-step sequences where fragments risk misread. Code/commits/PRs/docstrings: written normally.
- **DO WHAT WAS ASKED — DO NOT EXTRAPOLATE.** Deliver requested thing, then STOP and report. No inventing adjacent work because it seem useful: no unrequested helper/launcher/benchmark scripts, no extra files staged "for later", no speculative refactors, no scope widened from "fix X" to "also improve Y". Adjacent work look warranted → NAME it in one line, let user choose. Do not pre-build. Uncertain whether in scope? It not; ask. Over-delivery not helpfulness: cost review time, bury actual change, create artifacts nobody vetted.
- **Irreversible / outward-facing actions need explicit permission EVERY time**: rewriting published history (`push --force*`), amending/squashing pushed commits, deleting or overwriting files the user created, `git reset --hard`, closing/merging PRs, installing or upgrading system packages, reboots. Approval for one such action does NOT carry to the next. Propose, then wait.
- Nontrivial task: short plan before edit. Read nearby impl+tests first. Ambiguous numerics/physics/API: ask.
- Minimal diffs. No unrelated refactor in bug fix.
- **DUAL adversarial review is the DEFAULT for EVERY substantive change — TWO
  independent reviewers, never one** (user directive 2026-08-12, restated
  STRICT 2026-08-13, again 2026-08-20 as "make this the default behaviour").
  Author NEVER review own code. Route by who WROTE code — two reviewers always
  the other two:
  **Claude-authored → codex + GLM-5.2** (`mcp__zai__ask_glm`);
  **GLM-authored → codex + Claude**;
  **codex-authored → Claude + GLM-5.2**.
  Both BEFORE the PR, not after; report both verdicts in PR body and status
  line. Subagents doing implementation must be told to run BOTH — they default
  to codex only. Applies to measurement harnesses and probes too: instrument
  decide what we believe.
  Why two: they catch DIFFERENT classes. Codex find diff defects (broken
  contracts, vacuous tests that cannot fail, silent fallbacks, binding gates
  accepting wrong arm). GLM find MECHANISM defects (wrong objective, wrong
  regime, lever whose premise receipts already falsified — it retracted own
  top-ranked lever once measurement contradicted it).
  Reviewer disagreement = SIGNAL, not noise: name disputed point and the
  measurement that discriminates, run it if cheap, never average the two.
  A REVIEWER'S FINDING IS A HYPOTHESIS, NOT AN INSTRUCTION: measure before fix.
  2026-08-20, GLM predicted rounding error 1.4e-5 that would refuse valid
  restart files; measured over every column model builds it was 7.9e-8, so guard
  added for it was DELETED and measurement pinned instead — unmeasured finding
  buy knob that never bind.
  **"Substantive" includes MERGE CONFLICT RESOLUTION**, and rebases,
  cherry-picks, back-ports, probes, test fixtures. Conflict resolution =
  highest-risk case in list: someone chose between two versions of same code and
  diff not record what was discarded. FAILURE 2026-08-20: merged 17-day-old
  branch, hand-resolved 14 conflicts across 7 files including two independent
  fixes for one defect, shipped unreviewed until asked.
- **Codex adversarial review MANDATORY after any major code implementation/change.** Trigger: new module/feature, dycore/physics/parallel/ocean/land/ice/coupler/training edit, >~50 LOC, multi-file, or anything touching numerics/AD/JIT/pytree/conservation. Run **iterate-with-codex agent** loop below (`/codex:adversarial-review --wait` → fix flagged → `/codex:review --wait` → repeat until clean or 30 iter) BEFORE declaring done; report that review ran + verdict.
  **Review SUBAGENT dies (spend limit, API error) = NOT review waiver — codex
  CLI is separate binary with separate credentials, usually still reachable:
  `codex exec --sandbox read-only -C <repo> "<prompt>"` (`which codex`,
  `~/.codex/auth.json`). Try CLI directly before ever proceeding unreviewed. BOTH
  unavailable → say "UNREVIEWED" in every status until one succeed.**
  2026-07-26: subagent hit monthly spend limit, many iterations ran unreviewed
  while CLI worked fine whole time. Exempt: trivial/mechanical edits (typo,
  comment, rename, doc/markdown/`.tex`-only, single config value).
- **Pre-impl search mandatory**: before new fn/helper/class/operator/diagnostic/init/load/loss/numerical routine, grep `src/legoesm/` for similar names/docstrings/formulas in `thermo.py`, `constants.py`, `eos.py`, `ml/loss.py`, `diagnostics/`, `core/`, `atmosphere/physics/_shared.py`. State searched+found. Similar exists → extend/factor.
- **Shared utilities — never re-derive** (prod, scripts, validators, plotters, tests, notebooks, probes):
  - Constants: `from legoesm import constants` → `T_freeze`, `R_d`, `c_pd`, `L_v`, `R_v`, `epsilon`, `g`, `p_ref`, `kappa`, `sigma_sb`, `T_freeze_ocean`. No literals `273.15`/`287.0`/`1004.64`/`2.501e6`/`461.51`/`0.622`/`9.80616`/`6.371e6`/`7.292e-5`.
  - Saturation: `from legoesm.thermo import saturation_vapor_pressure, saturation_mixing_ratio, saturation_mixing_ratio_ice`. No re-impl Tetens/Magnus/Clausius–Clapeyron (plotters incl). Why: re-derived `e_sat=611.2*exp(17.67*Tc/(Tc+243.5))` diverged from model → false supersat in CI.
  - Column integrals: `legoesm.diagnostics.column_integrals` (`column_water_vapor`). No inline `jnp.sum(q*p_s*dsigma)/g`.
  - Losses: `ml/loss.py` (`area_weighted_mse`, `spectral_loss`, `per_variable_mse`).
  - Optimizer: `ml/training.create_optimizer()` (warmup+cosine+clip).
  - SCM-RCE gradient tuning: reuse `scripts/run/run_scm_rce_campaign.py` for CRM
    reference extraction / SCM evaluation and `legoesm.training.scm_rce_metrics`
    for normalized profile score. No duplicated RCE profile numerics.
  - SCM-RCE param training defaults to MUON via `ml.training.create_optimizer()`,
    init from `results/scm_rce_campaign/tuned_parameters.json`, writes recommended
    trained JSON under `results/`, never mutates production `*Config` defaults.
    Apply trainable overrides inside loss so leaves traced; static frozen leaves
    stay outside.
  - Every new `.py`, including `scripts/run/*.py` drivers, gets direct test.
    Scheme/factory dispatch must raise on unknown selections.
  - Atm column (h, ρ, virtual T): `atmosphere.physics._shared`.
  - Ocean EOS/pressure: `ocean.eos` (`compute_ocean_rho`, `compute_ocean_rho_and_pressure`).
  - SFNO: `ml/sfno.py`. No new neural op archs in training.
  - Channel packing: `ml/channel_packing.py` (`PE3DChannelSpec`, `pack_pe_state`, `unpack_pe_output`).
  - Ocean baroclinic (#214): `ocean/dynamics/ocean_tendency_common.py` (`iterate_eos_and_pressure_anomaly`, `apply_sponge_tracer_relaxation`, `apply_freshwater_virtual_salt_top`, `implicit_bottom_drag_factor`) in new `ocean_pe_*.py`.
  - Ocean barotropic (#214): `ocean/dynamics/barotropic_common.py` (`compute_filter_weights`, `bebt_blend`, `maxvel_clip`) in new `barotropic_*.py`. `tests/ocean/unit/test_no_scheme_duplication.py` enforces.
  - Plotters NOT exempt. Use model helpers for q_sat, RH, ρ, virtual T, MSE.
- No duplicate numerics across dycores/physics/grids/tests. Indexing/naming-only copy-paste forbidden.
- **No laziness on hard/large code** (>100 LOC, multi-component, full operator chains): no `pass`/`NotImplementedError` stubs, no partial-called-done, no skip edge cells/boundary halos/corner stencils/non-duogrid/MPI-sharded/AD-VJP. No happy-path-only tests. Too big → say so, list remainder, quantify risk.

## COMPARING TWO CONFIGURATIONS = A SIDE-BY-SIDE TABLE, ALWAYS
User directive 2026-08-21. Any time two runs / configs / templates / lanes are
compared, give a table: one row per field that DIFFERS, one column per
configuration, named in the header. Omit fields that agree — it is a diff, not
an inventory. Prose comparisons hide the one line that matters; the land
coupling week was lost to a surface-scheme mismatch a five-row table exposes at
a glance. This is also the required answer to "why did A work and B not".
If the table has more than one row, no single output difference is attributable
yet — say so.

## ACCURACY FIRST — A CLAIM THAT DRIVES A CODE CHANGE GETS REVIEWED BEFORE THE CODE
User directive 2026-08-21, after a week was spent coupling the WRONG land model
and then chasing two wrong causes in a row: *"we focus now on accuracy and we
must check with adversarial reviews our claims, especially when they imply
coding decisions."*

The existing dual-review rule covers DIFFS. This one covers the CLAIM that
justifies the diff, and it comes FIRST:

- **State the claim, then get it reviewed, THEN write code.** If a sentence of
  the form "X causes Y, so I will change Z" is about to become an edit, it goes
  to codex and GLM as a claim, before the edit exists. Both. A claim reviewed
  only after the code is written is a claim defended, not tested.
- **Say which parts are MEASURED and which are READ OFF THE CODE.** A scaling
  argument, a mechanism, a "this is the only consumer" — each is a separate
  claim with its own evidence.
- **Quote the number's PROVENANCE next to it.** "62 K leaf temperature" from a
  synthetic sweep that deliberately includes unreachable forcings is not the
  same statement as 62 K in a run. Report the reachable-subset number and the
  full-box number separately, and say which is which. A frightening number with
  no provenance wastes the reader's attention and can send the work sideways.
- **A number a well-posed scheme could never produce is a BUG IN THE
  INSTRUMENT until proven otherwise.** Check the probe before reporting the
  physics.
- **When a working configuration exists, diff against it FIRST.** See
  [[reuse-the-working-path-dont-rebuild]]: the offline land driver was healthy
  the whole time; the coupled lane simply called a different parameter provider.
  The first question on any coupling defect is "what does the working lane do
  differently", not "what could be wrong with the physics".
- **Reviewer disagreement is the signal to MEASURE, not to average.** Name the
  discriminating check and run it; it is usually a grep.

## Epistemic rules (non-negotiable)

### Never infer an API — read it
Before calling any function from JAX, Equinox, Optax, Diffrax, jaxKAN, or any
other dependency: grep installed source in site-packages, read actual signature.
No reconstruct from memory. Applies to argument names, argument order,
keyword-only args, return arity.

Symbol under `jax.experimental.*` → assume API changed since training data.
Verify or search. No guessing module paths.

### Report uncertainty explicitly
End any non-trivial code response with `UNVERIFIED:` block listing:
- APIs used but not read from source
- assumptions about library versions or runtime behavior
- anything that would silently produce wrong numbers rather than an error

Empty block = valid answer. Missing block = not.

My premise wrong — bug misdiagnosed, or thing asked for won't work — say so
before writing code.

### Diagnose before patching
Something fail: state candidate causes and how to discriminate between them,
then test. No going straight to fix. No agreeing with cause I suggested unless
evidence support it.


**Applies to THIS repo's own API too** — large enough that memory unreliable.
FAILURES in ONE session (2026-07-30): `AerosolConfig(reference_aod=)` (really
`reference_aod_550`), `McFarlaneConfig(N_ref=)` (no such field),
`run_amip.build_parser` (really `build_arg_parser`), `from legoesm.grids import
create_grid` (really `legoesm.grids.factory`). Each cost full probe round-trip.
Worse, `RRTMGPConfig()` default `include_clouds=False`, so offline harness
omitting it return CLEAR-SKY fluxes and EVERY cloud gradient exactly 0.0 —
silently wrong number, not error. Read NamedTuple `_fields` / `_field_defaults`
before constructing config.

User's premise wrong — bug misdiagnosed, or thing asked for won't work — say so
BEFORE writing code.
## Verification (JAX-specific)

Code not done until it run. Correctness claims require output.

- **Shapes/dtypes**: check with `jax.eval_shape` before running anything
  expensive. Cheap, catch most errors.
- **Gradients**: any new `custom_vjp`/`custom_jvp`, adjoint, or hand-derived
  derivative must pass `jax.test_util.check_grads(f, args, order=2)` before you
  claim it work. Gradient that run is not gradient that correct — single most
  common way to ship silently wrong result here. (Applies directly to
  `_sendrecv_vjp` in `halo_exchange.py` and any new MPI-AD path.)
- **jit parity**: run function eager and under `jit`, compare outputs.
  Divergence = tracer bug (Python-side branching, `.item()`, `if` on traced
  value, host callbacks).
- **Sharding**: verify with `jax.debug.visualize_array_sharding` or by printing
  `.sharding`, not by reasoning about what annotation should do.
- **Numerics**: default float32. State tolerance you compare at. No `==` on
  floats. Test need float64 → say so explicitly, no silent `jax_enable_x64`.
- **donate_argnums / buffer donation**: never add without confirming donated
  buffer not reused. Fail silently or crash far from cause.

## Scope
One change at a time. No refactor of adjacent code, no rename, no "improve" of
code I didn't ask about. Long unbroken generations drift into invention — prefer
small verified diff over large plausible one.

**DO NOT EXTRAPOLATE. Do only what was asked** (user directive 2026-08-06).
The ask = deliverable, not starting point to reason outward from.

- Related-looking problem you notice = ONE-LINE report, not work item. Name it
  and stop; do not start it.
- No widening scope because fix "would only be complete if" something adjacent
  also done. Ship the ask; state the boundary.
- New artifacts (scripts, benches, plots, panels, public APIs, config knobs)
  only when asked or genuinely required to finish the ask. Unsure whether
  required → it not; ask in one line.
- No proposing or launching compute user did not ask for.
- FAILURES 2026-08-05/06: scoped 128-GPU coupled ladder nobody requested off a
  question about existing plots; added coupled panel to figure when asked to
  assess the figure; wrote probes and helper scripts for questions never posed.
  Each cost round-trip and buried actual answer.

## Attribution Gates — MANDATORY, each from a real 2026-07 failure
Model near operational. Every rule below mechanical: satisfy it or state
explicitly that you did not. "I was careful" is not compliance.

- **PROVE THE PATH EXECUTES before blaming a line.** Naming file:line as cause
  require showing that line run in configuration under test: print ENCLOSING
  FUNCTION (`awk` nearest `def` above it), confirm active lane/driver call it.
  FAILURE: blamed positivity clamps at `model_driver.py:10923` for century's
  water source; they live in `_run_per_step` while century run `_run_column`,
  which contain no moisture clamp at all. Fix nearly written for lane the run
  never touch. Same class as reading entry point instead of full path.
- **REUSING A REFERENCE IMPL MEANS PORTING ITS EXCLUSIONS, not just its
  formula.** State which of reference's guards/scope conditions you kept and
  which dropped, with reason for each. FAILURE: copied
  `spectral_les_moist.conserving_positive` but not its `n_water` split, so
  column-conserving borrow applied to number concentrations
  (`N_c`/`N_i`/`N_r`) — unphysical, and it fed M2005 deposition (~N_i^(2/3)),
  producing fake "accelerating dry bias" reported before being caught.
- **A TEST THAT INSPECTS SOURCE MUST NAME THE SYMBOL THAT RUNS, and must be
  shown to FAIL when feature removed.** `inspect.getsource(X)` assertion where X
  is delegating wrapper pass while proving nothing. FAILURE: asserted against
  `MPASPrimitiveEquationModel.step`; floors are in `_step_jit`.
- **TOOL STATUS IS NOT EVIDENCE — read output tail.** Exit code without tool's
  own success line (pytest's `N passed`, "COMPLETED in Xs") = UNVERIFIED; OOM
  kills and timeouts can surface as success. FAILURE: reported regression suite
  green on exit-0 that was actually `Out Of Memory` mid-run. Quote decisive line
  when claiming suite passed.
- **EVERY BASELINE/ALLOW-LIST REASON STRING IS A CLAIM — verify in code before
  writing it.** Plausible-sounding reason permanently hide real defect. FAILURE:
  classified `convective_buoyancy_death_memory` as "carried in SegmentCarry" (it
  not — leaf `BechtoldConfig.buoyancy_death_memory` exist and nothing map to
  it), asserted `SBMConfig.precip_efficiency` leaf that not exist, and credited
  `micro_substeps` to consumer that read `args.`, not the config field.
- **RATE / TENDENCY / SKILL COMPARISONS: identical windows on BOTH sides, and
  print window next to number.** Differing spans = confound, not result.
  FAILURE: TCW over days 190-530 vs CMOR year 1 gave "+38.7 kg/m2/yr"; matched
  windows gave +13.4. Extends existing controlled-comparison rule to derived
  rates.
- **A DIAGNOSTIC'S PRINTED PRECISION BOUNDS THE RATE YOU CAN CLAIM.** Log CWV at
  0.1 kg/m2 over 8 days resolve only ~±4.6 kg/m2/yr — no reporting trend inside
  one quantum. Prefer fp64 from model state (checkpoints) over parsed log lines.
  Same class as throughput-quantization error.
- **`JAX_ENABLE_X64=1` on any numerics/conservation test.** fp32 mismatch NOT a
  failure until re-run with x64; and *new* failure not yours until reproduced
  with your change stashed. Do both before reporting regression.
- **RUN-TARGET PARAMS ARE ABSOLUTE (`TARGET_DAYS`), and "latest checkpoint"
  MOVES.** For controlled pair, COPY pinned checkpoint into each arm dir; never
  use `PREV_CKPT_DIR`-style newest-wins pointer while another run advancing.
  FAILURE (twice): arms exited instantly at "Already at/past target".
- **A LAUNCHER FLAG THAT SWITCHES ONE FORCING CHANNEL MUST SWITCH ALL OF THEM.**
  Verify resolved paths in run log, not the flag you passed. FAILURE:
  `CENTURY_DECK=1` set era-correct ozone+volcanic but left 1979-2016 SST.
- **PROSE IS A POINTER, NEVER A CITABLE FACT.** Code comment, docstring,
  `AMIP.md`/`docs/` entry or "Known issue" naming a limitation, guard, or
  missing feature MUST be re-verified in CURRENT code at point of use before
  repeated as finding — this repo routinely fix things without updating prose.
  Comment name module imposing guard → OPEN THAT MODULE. FAILURES (2026-07-30,
  three in one day): quoted `_run_column` "turbulent surface fluxes are
  intentionally NOT applied" comment and `AMIP.md` Known #3 to claim MPAS has no
  turbulence — MPAS turbulence path exist (`turbulence/integration.py`, Perot
  edge->cell) and run resolve `turbulence=louis` + `surface_bulk_scheme=coare3`;
  and doubted a cloud_fraction comment that was exactly right.
- **THE FIRST GUARD YOU FIND IS NOT THE ONLY GUARD — follow value to its
  CONSUMER before declaring it unclamped/unchecked.** FAILURE: reported "no
  upper bound on r_eff" from `rrtmgp.py`'s `clip(x, 1e-6, None)`; real clamp to
  lookup-table range is one call deeper in `rrtmgp/optics/cloud_optics.py`. Same
  class as blaming a line without proving its enclosing function runs.
- **BEFORE ATTRIBUTING A BIAS TO A COMPONENT, PROVE THE COMPONENT EXISTS AND
  RUNS IN THE CONFIG UNDER TEST.** ABSENT component and BADLY-TUNED one give
  SAME symptom, and every tuning arm against absent component return null.
  Enumerate, from RESOLVED config and LANE'S code path (not deck's comments):
  the scheme, its PROGNOSTIC STATE, its inputs. No prognostic state -> say so
  out loud; that usually the finding.
  Corollaries, each earned:
  (a) **A `false` on a component switch does not mean "the simpler version" —
  it may mean NOTHING.** Check what lane actually fall back to.
  (b) **A DECK COMMENT NAMING A FALLBACK IS A POINTER, NOT A FACT** (prose rule,
  applied to configs): grep named symbol in that lane's setup.
  (c) **When N different schemes for component A all fail to move a bias, STOP
  TUNING A.** Limiter upstream, or A's inputs wrong.
  (d) **A parameter wrong at BOTH ENDS of a physical range is a MISSING
  DEPENDENCE, not a mistuning** — thing it should depend on not read.
  FAILURE 2026-08-15, most expensive of campaign: MPAS AMIP deck set
  `use_multilayer_land: false` with comment "this deck runs the slab". There is
  NO slab land on MPAS lane (`slab_land_active` wired only on cube/general
  path), so runs had **no land surface model at all** — land T_sfc was nearest
  OCEAN's prescribed SST minus 6.5 K/km, land evaporation fixed 0.6 of
  saturation, no soil, no water store, no runoff, no stomata, no surface energy
  balance. Tropical deserts evaporated 3.5x observed and rainforest 0.73x FROM
  SAME CONSTANT (corollary d, unnoticed); sensible heat ~25 W/m2 on Sahara and
  Amazon alike. Waves 9-11 spent GPU-hours swapping FIVE convection schemes and
  dozen trigger knobs against tropical-land rain deficit no convection scheme
  could ever fix (corollary c, unnoticed). Interactive soil moisture, bucket
  hydrology, stomatal control are DEFAULTS for AMIP case, not options.
- **A GLOBAL STATISTIC ON A NON-UNIFORM GRID NEEDS AREA WEIGHTS.** Never
  `np.mean(field)` for global mean on lat-lon (or any stretched grid) — use
  `cos(lat)` or model's `areacella`. FAILURE: reported "+17 hPa of dry mass
  created" from unweighted `p_s` mean; AREA-WEIGHTED mass was invariant at
  983.493 hPa to 6 digits, i.e. defect did not exist. Habits carried from
  quasi-uniform MPAS/SCVT mesh are INVALID on lat-lon.
- **A PROPOSED MECHANISM MUST SURVIVE A SCALING / PERTURBATION TEST BEFORE IT IS
  CITED AS THE CAUSE.** X claimed to drive Y → change X by known factor, check Y
  respond as mechanism predict. FAILURE: proposed "damp-to-rest pumps mass
  convergence" (predict ~linear in sponge coefficient); quartering coefficient
  slowed growth only 1.5x, refuting it — fix would have shipped on false
  mechanism. Label every uncaught claim PLAUSIBLE; honest "cause unknown" is
  cheap, confident wrong cause buy a code change and a relaunch.

## Implementation Discipline — the CODE and its PROSE are both claims
User, 2026-08-06: *"Be much more conscientious and careful when implementing.
Be systematic, check, do not be sloppy or too fast."* Every rule below from
defect shipped in ONE session that prompted it, and every one caught by
adversarial reviewer rather than by me — i.e. each avoidable by reading two more
lines before typing. Slow down at these exact points.

- **A DOCSTRING/COMMENT THAT DESCRIBES BEHAVIOUR IS A TESTABLE CLAIM. Trace data
  flow before writing it.** FAILURE: wrote "dropping this call now makes the
  test fail" about `divg_d` exchange — false, because unit 4 hands only
  `uc`/`vc` to `d_sw1` and `divg_d` not consumed until `d_sw5`. Sentence written
  from intent, not from call chain. Before asserting "X is covered by this
  test", name consumer of X and confirm it execute inside test's span.
- **NEVER WRITE "this removes the question entirely" ABOUT AN API YOU HAVE NOT
  READ.** FAILURE: claimed `np.ascontiguousarray` give unconditional copy; it is
  copy-IF-NEEDED, so it aliased at km=1 and copied at km>1 — one line of code
  with two aliasing behaviours, asserted as safe. Extends *Never infer an API*
  to STRENGTH of a guarantee, not just signature.
- **SCOPE WORDS — "both", "all", "every", "cannot", "unified", "always" — GET A
  GREP BEFORE THEY GET TYPED.** FAILURE: "both lanes now route through one
  helper" while `csw_step_sixface` still called interim helpers directly. Went
  into COMMIT MESSAGE, which cannot be edited after. Weaken to what you verified
  ("the post-p_grad_c site") or run the grep.
- **RE-READ A BOOLEAN IMPLICATION IN THE SOURCE BEFORE RESTATING IT; ONE-WAY IS
  NOT TWO-WAY.** FAILURE: `bounded_domain = regional .or. nested .or.
  duogrid` mean duogrid⇒bounded, and I wrote converse pair "unreachable" —
  inverting it and mislabelling legitimate regional/nested category as
  impossible. Quote the line next to restatement so direction checkable.
- **A CONTROL THAT PERTURBS A ZERO IS NOT A CONTROL. Print baseline value the
  perturbation multiplies BEFORE trusting result.** FAILURE: divg-corner probe
  scaled `divg_d(1,1)` by 3 where that cell is exactly 0.0, so control was no-op
  and proved nothing; claim actually rested on predicted==actual identity. Say
  which check carried the claim.
- **AN A/B MUST DIFFER IN ONE FIELD OF THE CONSTRUCTOR, AND YOU MUST DIFF THE
  CONSTRUCTOR ARGS TO KNOW.** FAILURE: compared two contexts differing in grid
  conventions AND in whether ext bundle existed, while asserting it isolated the
  exchange. Applies to fixtures, not just runs — controlled-comparison rule
  cover `pytest` fixtures too.
- **A NEW TEST/FIXTURE IS PART OF THE DIFF: RUN THE SUITE THAT CONSUMES IT, NOT
  ONLY THE TEST YOU WROTE.** Fixture edit = change to every test in the module.
- **SHELL COMMANDS THAT CAN PROMPT WILL SILENTLY NOT RUN.** Use `git checkout
  --`/`git restore`, `cp -f`, `rm -f`; never bare `cp`/`mv` for revert. FAILURE:
  bare `cp` hit interactive overwrite prompt and "revert" never applied — one
  `git status` short of reporting clean tree that was not clean. VERIFY EVERY
  REVERT with `git status --porcelain`.
- **A LOG-SCRAPING GUARD MUST EXCLUDE THE PROMPT/COMMAND IT ECHOES.** FAILURE:
  review wrapper grepped whole log for failure token its own prompt contained,
  so it "retried" every time and never printed exit status. Same class as test
  that cannot fail.
- **BUDGET THE REVIEW LOOP INTO THE WORK.** Adversarial reviewer found real
  defects in EACH of three rounds here; rounds 2 and 3 not ceremony. No
  presenting first-round implementation as finished, no treating "tests pass" as
  terminal condition — round-1 diff passed 401 tests while still containing
  silent fallback and false docstring.

## Assumption Gates — from six wrong assumptions in ONE session (2026-08-05/06)
User callout: *"you keep making a lot of assumptions that prove to be wrong."*
Gates above stop wrong CLAIMS about the model; these stop cheaper, more frequent
error — being wrong about CODE AND DATA IN FRONT OF YOU. Every rule below
mechanical, each cost full round-trip.

- **AN ARRAY'S LAYOUT IS AN API — READ IT, INCLUDING AXIS ORDER.** "Never infer
  an API" rule cover signatures; it also cover array SHAPE, AXIS ORDER, index
  base, padding convention. Before first index of any mesh/state field, print
  its `.shape`. FAILURE: indexed `mesh.cellsOnCell[c, k]` assuming
  `(nCells, 6)`; it is `(6, nCells)`, so probe raised `IndexError` on every
  method. `cellsOnEdge` is `(2,
  nEdges)` — one unambiguous pair per edge, usually the better handle.
- **A PROXY IS NOT THE QUANTITY. Real number produced by specific code path →
  GET IT BY CALLING THAT PATH.** Re-deriving lookalike from first principles
  silently answer different question. FAILURE: computed `max_degree` from 1-ring
  `cellsOnEdge` adjacency and reported it as "ppermute round count" — real
  exchange is halo-depth-aware, so proxy said 8 rounds for every method/rank
  count while real schedule said 12→14. Proxy could not even reproduce known
  answer — that the tell: **run proxy against a case whose real value you
  already know BEFORE using it on the unknown one.**
- **AN OPTIMIZER ONLY HELPS IF ITS OBJECTIVE IS THE BINDING TERM — state which
  quantity it minimizes, confirm that quantity is measured bottleneck.**
  FAILURE: assumed METIS would cut MPAS ppermute rounds; METIS minimize EDGE
  CUT, bottleneck is MAX_DEGREE, and measured they move OPPOSITELY (metis 19
  rounds @64 vs sfc 14, while metis has lower cut). "Better partitioner" is not
  a mechanism.
- **A TABLE YOU PARSED IS NOT DATA UNTIL YOU SPOT-CHECK IT.** Any derived
  summary quoted to human, or fed to review agent, need ≥2 rows verified by eye
  against raw source first. FAILURE: regex over multi-line series lists attached
  subdiv-9's `9.60/11.48 ms` to `s8` label, and that mislabelled pair went into
  codex prompt as fact.
- **PRE-IMPL GREP COVERS TESTS, NOT JUST FUNCTIONS.** Before writing test, grep
  for one already asserting same invariant. FAILURE: added two tests counting
  PCG reduction sites duplicating existing `test_reduction_count_halved`; codex
  had to point it out.
- **WHEN A TEST FAILS, DECIDE WHETHER THE EXPECTATION OR THE CODE IS WRONG, AND
  SAY WHICH.** Assertion you just wrote is claim with no more standing than the
  code. FAILURE: asserted `LEGOESM_..._FUSED_HALO=""` meant OFF; resolver is
  `!= "0"`, so `""` means ON — test was wrong, not the default.
- **BEFORE PROPOSING COMPUTE, CHECK THE PATH IS BUILT — grep driver for
  decomposition/sharding the run would need.** FAILURE: scoped 128-GPU coupled
  ladder before finding `CoupledESM` contain three occurrences of "shard", all
  in comments — coupled ocean is replicated, so ladder would have measured
  unbuilt path.
- **WHEN TWO MECHANISMS COULD EXPLAIN A NUMBER, NAME THE MEASUREMENT THAT
  DISCRIMINATES THEM AND RUN IT BEFORE REPORTING EITHER.** FAILURE: reported
  serial-vs-SPMD gap as "cadence PHASE disagreement"; predicates agree with ZERO
  offset and real cause was serial short-tail fallback. One `grep` of two
  predicates would have settled it.

## Compute Discipline — speculation costs GPU-hours, not just credibility
User, 2026-07-30 (THIRD callout in five days): *"You keep making very
speculative assumptions... be much more precise so we do not waste time with
useless simulations."* Gates above stop wrong CLAIMS; these stop wrong RUNS.
Simulation launched on hypothesis no measurement can refute = pure waste, and it
also cost WALL-CLOCK of queue slot it occupied.

- **NO COMPUTE ON AN UNFALSIFIABLE HYPOTHESIS. Before submitting ANY job costing
  >1 GPU-hour, write down three things: (a) exact number the run will produce,
  (b) value that CONFIRMS and value that REFUTES, (c) why a cheaper offline/CPU
  test on an EXISTING checkpoint cannot answer it. Cannot fill all three -> DO
  NOT SUBMIT; run cheap test first.** Nearly every question asked so far
  (fluxes, tendencies, radii, momentum budgets, cloud optics) was answerable
  offline from saved checkpoint in minutes. FAILURE 2026-07-30: submitted two
  5-YEAR full-physics runs (8 h walltime each) while model had KNOWN unfixed +56
  W/m2 albedo error and no low-level circulation — five simulated years of
  broken climate, answering no question that had been asked.
- **RANK ERRORS BY MAGNITUDE BEFORE CHOOSING WHAT TO WORK ON.** Run full
  scorecard FIRST, work LARGEST term; re-rank after every fix. FAILURE
  2026-07-30: spent most of session on hfls deficit (-40 W/m2) while dominant
  error was rsut (+56 W/m2) — and scorecard naming it was already sitting in run
  directory, unread.
- **ONE VARIABLE PER PRODUCTION RUN.** N simultaneous config changes answer ZERO
  questions, because no output attributable to any one of them. Multi-change
  config legitimate ONLY as deliberate new BASELINE labelled as such and never
  compared term-by-term against the old one. FAILURE 2026-07-30: one launch
  flipped ~8 switches at once.
- **A LONG RUN ON A MODEL WITH AN UNFIXED DOMINANT ERROR IS WASTE.** Before
  extending past ~30 simulated days, state largest outstanding scorecard term
  and why run still worth its GPU-hours. Fix big term, then extend. Short
  validation windows (days) are for "does it run and is new physics behaving";
  multi-year windows are for model that already pass.
- **NEVER LEAD WITH AN ARITHMETIC COINCIDENCE.** Hand-computed ratio that
  "matches" observed ratio is not evidence when calculation omit factors code
  actually applies. State it as arithmetic, or don't state it. FAILURE
  2026-07-30: "cover x tau ~ 1.9x matches the 1.9x albedo" ignored sub-grid
  inhomogeneity factor radiation applies to cloud paths.
- **PREFER THE INSTRUMENT THAT ALREADY EXISTS.** Before writing probe, check run
  directory for scorecard/manifest/diagnostic answering the question, and
  `scripts/validate/` for validator. Reading existing artifact cost seconds; new
  probe cost an hour and need its own controls.

## Diagnosis Discipline — one session, ~10 GPU arms, 5 of them wasted
2026-08-05/06, FESOM2-match OMIP blowup. Every rule mechanical, from specific
failure in that one session. Root cause turned out to be one-line IC defect
findable offline in seconds; arms bought nothing.

- **SANITY-CHECK EVERY PROBE NUMBER AGAINST A PHYSICAL RANGE BEFORE BUILDING ON
  IT — one line of arithmetic, before the conclusion.** Convert to quantity
  whose plausible span you know, compare. FAILURE: PGF probe returned 0.02
  m/s^2 and I built whole attribution on it; that value imply ~14 kg/m^3 density
  difference between ADJACENT 1-degree cells when entire ocean span ~6 kg/m^3.
  Check take seconds, number sat unused for hours, and running it immediately
  would have exposed real defect (cells initialised to T=0/S=0) before a single
  GPU arm.

- **A FAILURE STEP OR LOCATION IS NOT A SIGNATURE UNTIL THE DIAGNOSTIC IS SHOWN
  TO RESOLVE IT.** State instrument's resolution, confirm reported step is not
  merely its first sample. FAILURE: five one-variable arms all reported "BLOWUP
  at step 36" and I read that config-invariance as physics. `run_omip` set
  `block_size = max(1, diag_every)` and only test at block boundaries, so 36 was
  FIRST LOOK. At `--diag-every 1` answer was step 1. Corollary: arms run at
  coarse diagnostic stride prove only "this setting alone does not fix it",
  NEVER "this setting has no effect".

- **REFUTE WITH STATES, NOT EXIT CODES OR FAILURE STEPS.** Two runs failing at
  same step can fail for different reasons. FAILURE: declared open North Pole
  "refuted" because capping did not move blowup step; states showed capping HAD
  removed that mode (|u|max 19.27 -> 0.0137 m/s) and merely uncovered second,
  unrelated one at same step number.

- **A PROBE IS PRODUCTION CODE FOR THE "NEVER INFER AN API" RULE, AND ITS FIRST
  OUTPUT IS UNTRUSTED.** Read signature of every function a probe call,
  including this repo's own. FAILURE: called
  `build_runoff_map(lat, lon, mask, area)` when signature is
  `(ocean_mask, cell_area, lat_rad, lon_rad)`, so `cell_area` received
  `deg2rad(lon)` — exactly 0 at lon 0 — and I reported NaN "conservation defect"
  in shipped code that did not exist.

- **A PLAUSIBLE-LOOKING VALUE IS MORE DANGEROUS THAN A NaN, IN CODE AND IN
  TESTS.** Sentinel that is valid number of right dtype pass every finite/NaN
  guard downstream. FAILURE: `_interp_profile_to_z_coord` returned `0.0` for
  no-data column; `0.0` is not NaN, so caller's `where(isnan, fill, x)` never
  fired and 39032 wet cells entered model as fresh water at 0 degC. TWO
  COMMITTED TESTS ASSERTED THAT BEHAVIOUR, one calling it "documented degenerate
  fallback" and one commenting "still no NaN leaks" — exact inversion of truth.
  Degenerate branch must return something → return NaN (or raise) so downstream
  guard can see it, and treat any test asserting magic sentinel as suspect.

- **BUDGET CLOSURE BEFORE HYPOTHESES.** For any conservation-relevant blowup or
  drift, run closure/redistribution check FIRST — it name the operator class
  instead of ranking guesses. Canonical probe:
  `scripts/validate/ocean_fidelity/omip_conservation_closure.py`. On first
  deployment it refuted hypothesis I was about to spend a GPU arm on (dipole
  whose two-cell sum GREW is not a diffusion instability) and localised
  config-invariant source to single column. See
  `docs/ocean/fidelity/fidelity_to_fesom2jax_level_plan.md` (#1492) Phase 0.1.

- **A THROWAWAY PROBE'S NUMBER IS UNMEASURED. COMMIT THE PROBE.** Per #1492
  Phase 0.3: one committed probe per row, locked conventions, provenance (git
  SHA + inputs + flags) stamped on every run. Inline heredoc probes not citable,
  cannot be re-run against changed model, and hide own bugs — both defects above
  came from uncommitted ones, and committing closure probe immediately surfaced
  two more inside it.

- **DAMAGE-FIELD CORRELATIONS ARE NOT MECHANISMS.** In blown-up field, "worst
  cells are coastal / shallow / polar" describe where damage LANDED, usually
  downstream of single upstream defect. FAILURE: measured P(bad|coastal)=33% vs
  0.67% interior, real 50x enrichment, and treated it as coastal process; those
  were simply columns the IC defect had corrupted. Before citing spatial
  pattern, confirm snapshot is ORIGIN state (`snapshot_final.npz` stamp
  `_step = n_steps` regardless of when run stopped — verify with dedicated
  1-step run).

## JAX
- Pure pytree fns. `lax.scan` time integration. `vmap`/batched arrays over Python loops on array dims. `jnp.where`/`lax.cond`/`fori_loop`/`scan` not Python control flow on traced.
- **Feature gating exception** (`fix_mass`, `fix_moisture`): Python `if` on static bool in closure — NOT `jnp.where` (traces both branches). `jnp.where` only for data-dependent traced selection.
- Stable shapes. No retrace. Dtype: spectral=x64+complex128; finite-volume can float32.
- No host/device thrash, NumPy in traced code, hidden non-JAX side effects.
- **Buffer donation + `jax.grad`**: `donate_argnums` conflicts reverse-mode AD. JIT fn inside `jax.grad`/`eqx.filter_value_and_grad` → provide non-donating variant (`.raw`). See `build_segment_fn`.
- **Closures vs explicit args**: closure captures = compile-time consts. Per-iter changing val (SST, solar) → pass as traced arg. See `SegmentForcing`.

## Earth System
- Conservation, metric consistency, staggered-grid consistency, halo correctness = first-class.
- No silent clip/damp/coerce unless justified+validated.
- Preserve units, sign conventions, monotonicity/positivity, hydrostatic/nonhydrostatic.
- Cubed-sphere/curvilinear: assume edge+metric errors first.
- Physics coupling: column closure + consistent flux signs.
- DA/diff: preserve smoothness. No gratuitous nondiff.
- **Sign-convention check MANDATORY on every equation/flux/tendency edit.** Before declaring done on any code touching PDE term, flux, source/sink, BC, or budget update: (1) state coordinate convention in scope (z up/down, flux positive-up/down/into-body) as comment at the term; (2) walk EACH term, confirm its sign match that convention — gravity vs capillary/diffusion divergence, top vs bottom BC, source vs sink, the `±` in `state = state ± dt·tend`; (3) confirm budget closes (`in − out − Δstorage = 0`) and exchanged fluxes carry SAME sign at both ends of a coupling (land runoff `+into ocean` must arrive `+into ocean`). Comment label and code must agree — flux commented "upward" computed as downward = defect, fix label or math. Mechanical gate where feasible: sign/conservation unit test (analytic column, manufactured solution, or `assert` budget residual ≈ 0) — passing norms alone never certify sign right (flipped flux can still be small). Common flips: z-axis direction, evap positive-up vs moistening, brine/salt vs freshwater dilution, stress atmospheric vs ocean convention (`-tau`), free-drainage vs gravity double-count.

## Parallel/HPC
- Correctness across serial/multi-device/MPI/hybrid.
- Sharded: reason about halo exchange, reductions, partition specs, global invariants.
- Validate single-rank → smallest distributed.
- Backends differ (Metal/GPU/CPU/spectral/MPI). Apple Silicon: spectral on CPU.
- **MPI**: `initialize_distributed(global_n=N)` → `scatter_to_local()` → rank-local step → `gather_to_global()` for I/O only. Never full global per rank.
- **4D halo**: `pad_halo_4d()`+`pad_halo_vector_4d()` all vert in one msg. All 3D ops in `operators_3d.py` use 4D. Never `vmap(pad_halo)`.
- **MPI halo AD**: all `sendrecv` via `_sendrecv_vjp` (`@jax.custom_vjp` in `halo_exchange.py`). Only `allreduce(SUM)` AD-safe (via `global_sum_mpi`; `final_loss=True` only for a sum seeded as the loss); `MAX`/`MIN`/`allgather`/`bcast` = diagnostics only.
- **Device mesh under MPI**: per-rank count to `create_device_mesh()`, not total.

## Oracle-Recipe Fidelity (ocean) — see docs/ocean/fidelity/oracle_recipe_strategy.md
- ADDITIVE to Validation Rules: oracle work NEVER replaces unit tests, ocean matrix, conservation checks, or visual verification. Truth tiers (conservation/equivariance/analytic) outrank oracle-matching.
- Recipe = pure config selecting shared canonical blocks (never bespoke `veros_*` solver). Oracle-matching numerics go in canonical module (`eos.py`, advection/limiter dispatch, `vertical_mixing/`, integrator dispatch) as selectable options.
- Mimicry-only glue (halo strip, axis transpose, time-level handling) lives in fidelity harness, never model. Test: "would a user with different goal ever select this?" No → harness.
- Conventions handled only in bridge, verified by equivariance tests (`physics(φ(x))=φ(physics(x))` to tol); a "convention" changing wet domain/answers is physics → config, not bridge.
- Constants are config (`ConstantsConfig`), not module-global monkey-patches (no `override_constants` in shippable paths); defaults reference `legoesm.constants`; base only, derived (κ,ε) recomputed.
- Oracle tendency-match (tier 3) trusted only for block that also clear truth tiers (0–2). MACHINE-ENFORCED (#388 Ask#4): `ocean/fidelity/precedence.py::evaluate_precedence` LOCKS oracle tiers (≥3) on any truth-tier (0–2) failure; surfaced + exit-gated by `scripts/validate/ocean_fidelity/build_fidelity_scorecard.py` (the one generated scorecard).

## Recipe×Setup template adapters (#388)
- **One shared selector, never re-implemented.** Every component YAML experiment adapter exposing a `setup:` block (atmosphere `Config`, ocean `OceanExperimentConfig`, sea-ice `SeaIceExperimentConfig`, …) MUST route validation + command-building + signature through `legoesm.core.setup_selector` (`MatrixRunnerSpec` + `validate_setup`/`build_matrix_command`/`setup_signature`/`require_positive_finite`). Per-component CLI differences (case flag `--only`/`--test`, exact-match `exact_prefix`, which `--levels`/`--dt`/`--days`/`--resolution` flags exist) are a `MatrixRunnerSpec`, NOT a copy-pasted `parts=[...]` builder or private `_SETUP_KEYS`/`_validate_setup`. Component-specific case-name validation goes through `known_names=` kwarg. Re-implemented selector = REJECTED in review. Factory naming: `_<component>_matrix_spec()`. FOLLOW-UP: dedup `OceanExperimentConfig`'s inline selector onto shared helper once PR #465 + shared-helper PR both land on main (inline copy predates extraction).
- **Every setup template's `(name, grid)` MUST be a real matrix case**, enforced by catalog-backed test (`run_<comp>_test_matrix._build_test_matrix()` → assert pair exists) so `--test/--only =<name> --grid <grid>` never select nothing. New template without this gate → REJECTED.
- **An exact case selector matching nothing is a hard error** (`raise SystemExit`) in every matrix-runner mode — never silent no-op (dispatch-hardening). MPI-safe: key guard off global pre-slice match count; empty-slice ranks fall through to barrier.
- **New `experiment_registry` mode → same-PR `get_adapter` test** in `test_experiment_registry.test_get_adapter_resolves_classes` (template test importing class directly does NOT cover dispatch).
- Land has no standalone idealized-case surface (no registry/matrix runner) → recipe×setup N/A there; do not invent one to match pattern. Truth-tier precedence stays ocean-scoped per `oracle_recipe_strategy.md`; use `TRUTH_TIERS`/`ORACLE_TIER_FLOOR` by name (no hardcoded `3`).

## Validation
- Narrowest test after edits. Numerical changes: analytical/benchmark > unit tests alone. `JAX_ENABLE_X64=1` unless float32/Metal task.
- Dycore: Williamson, Galewsky, Jablonowski-Williamson, DCMIP, Held-Suarez, ocean benchmarks.
- Conservation/reductions/coupler: mass+energy diagnostics.
- Parallel: unsharded vs sharded, single-rank vs MPI.
- Too expensive: say what ran/didn't, residual risk.
- **CRITICAL — Controlled comparison: change ONE variable, hold eval protocol FIXED to baseline.** To claim a change (resolution, params, scheme, days) improved/degraded/"is comparable" vs prior result, keep EVERYTHING else byte-identical to that baseline: forcing data + its sampling (years/days/hours/climatology), evaluation grid, metric definition (bias vs RMSE), region masks, timestepping. Metric that moved because protocol/sampling changed = **CONFOUND, not result** — NEVER compare number computed on one sampling/grid/metric to number from another and call difference an effect. Resource limit (network cap, compute, time) force lighter or different sampling → **re-run BASELINE at that SAME sampling before comparing** — fresh baseline cheap insurance; confounded claim not. Before writing "improved"/"degraded"/"better"/"comparable" vs any earlier number, explicitly confirm two configs differ ONLY in variable under test; cannot → say so, claim NO direction. Report full config (data+sampling, grid, days/steps, params, metric) next to every number so reader know exactly what compared. Assuming two runs comparable when setup drifted = error that turn "we improved it" into "we degraded it."
- **CRITICAL — Precision gate for comparison/skill/causal claims (do NOT be sloppy — 2026-07-20 EC-site lesson).** (a) **Harness self-check FIRST**: before reporting NEW scheme's skill vs validated baseline, run KNOWN baseline through YOUR OWN harness, confirm it reproduce baseline's established/published number (±small tol). Your harness score validated scheme wildly off — e.g. two-leaf H looked "broken" (NSE≈−1) when paper figure track obs — the HARNESS is wrong; fix BEFORE any new-scheme claim. (b) **Match reference metric EXACTLY**: metric aggregation (daily-mean vs half-hourly point-wise NSE), window (multi-year JJA vs one summer), masks (valid-forcing) ALL part of protocol. Report SAME metric reference used, plus any alternative, WITH sample N. Single-window point-wise metric is NOT skill verdict, and never rank schemes by metric punishing one scheme's known artifact (point-wise scatter/spikes) while ignoring dimension of interest (mean diurnal shape). (c) **Never let QC mask flatter one side**: drop scheme's unphysical spikes from scoring → report BOTH failures-penalized primary score AND separately-labeled plausible-only sensitivity score. (d) **Do NOT INFER causal origin — INSTRUMENT it**: "the spikes come from X" require LOGGING X and alternatives (raw vs intermediate vs final), not deduction from reading one code path; "correctly hooked up" require reading FULL path, not entry point. (e) **Label every claim CONFIRMED (evidence shown) vs PLAUSIBLE (inferred)** — adversarial review WILL refute over-confident inferences; state uncertainty up front rather than presenting verdict table a five-minute check overturn.
- **CRITICAL — A gradient/AD-tuning result that underperforms a known-good baseline is a BUG in your method until PROVEN otherwise — never attribute it to physics.** Before interpreting ANY optimizer/AD tuning outcome, or attributing residual error to physical cause (radiation, resolution, scheme's structure): (1) **Differentiated loss MUST be exact quantity you evaluate and report.** Truncated-BPTT / short-window / stop-gradient'd-spinup / cached-state proxy differing from full evaluation = OBJECTIVE MISMATCH, not result — optimizer minimize proxy while eval barely move. Differentiate real objective (e.g. full RCE equilibrium the eval scores), or prove proxy tight. (2) **Verify AD actually flows:** finite-difference check on ≥1 parameter (analytic grad vs `(L(x+ε)−L(x−ε))/2ε`, ratio≈1), confirm gradients finite AND non-negligible (not silently zeroed by `stop_gradient`, frozen/`enable_*=False` leaf, `lax.stop_gradient`/`jax.lax.cond` dead branch, or detached recompute). (3) **Sanity vs known-good baseline FIRST:** derivative-free or prior method got large improvements and your gradient method barely move loss → YOUR optimization broken (weak objective, tiny effective LR, over-aggressive line-search backtracking to ~0 step, frozen leaves) — fix before drawing conclusions. Concluding "residual bias is radiative/structural, not tunable" from run whose optimizer never actually worked = exact error that ship FALSE scientific claim. Near operationalization this disqualifying: barely-moving loss, suspiciously-flat before/after, or result contradicting known-good baseline BLOCKS delivery until gradient path verified end-to-end. See [[feedback_verify_ad_before_physics]].
- **CRITICAL — Visual verify spatial/grid artifacts**: passing tests+norms NECESSARY ≠ SUFFICIENT for cubed-sphere ops, halo exchange, diffusion coeffs, grid metrics. Edge artifacts/cube imprint/grid-scale noise only detected visually (v-wind W2, wind_speed W5). Run `--only sw --grid cubed_sphere --quick` + inspect PNGs vs baseline. Norms can improve while artifacts worsen. Never claim "tests pass, edge fixed" from pytest alone.
- **Diffusion sensitivity**: div damping + hyperdiff AMPLIFY halo errors at cubed-sphere face boundaries. Check W2 v-wind visually when touching `_hyperdiff_cube`, `_div_damp_cube`, diffusion params.
- **Visual-regression gate (cube imprint)**: `scripts/validate/visual_regression.py --check` numericises W2 v-wind cube-imprint check (SSIM + per-panel perceptual hash + edge-artifact ratio vs tiny committed ref in `tests/visual_baselines/`). Deterministic metric math gated in CI (`tests/test_visual_regression_metrics.py`); full cube-SW `--check` runs as NIGHTLY non-blocking CI job until tolerances calibrated across CI hardware. Tiny numeric baselines (.npy+json) ARE tracked — the one carve-out to "no tracked visual baselines".
- **CRITICAL — VALIDATE THE INSTRUMENT BEFORE QUOTING ITS NUMBER (2026-07-25 duogrid lesson: 8 confident claims, all retracted).** Diagnostic script = UNTRUSTED CODE until it pass own controls. Never state finding — never write "measured", "confirmed", "proven", "VERDICT" — from probe's first output. **Before quoting any diagnostic number, run these five checks and say in the message that you ran them:**
  1. **Right conserved/invariant quantity?** Budget what SYSTEM conserves, not convenient proxy. (Failed: reported "vertex creates energy" from **KE alone** — KE NOT conserved in shallow water, it trade with PE. Total `E=∫area(½h|V|²+½gh²)` reversed sign of conclusion.)
  2. **Same transform / units / staggering on BOTH sides?** Two "A-grid winds" from different operators = DIFFERENT QUANTITIES. (Failed: ours `c2l_ord2` vs oracle `C2L_ORD=4` — SAME raw state gave 1.96e-2 vs 5.79e-2, 3× swing that WAS the reported effect. Also: never budget across stage boundary where state change representation — mid-step FV3 winds in circulation form, produced ±5.6e10 garbage. REPEAT OFFENCE 2026-08-05, this time SHIPPED then retracted: paired predicted face velocity (41.87 m/s) against model's `max_speed` timeseries diagnostic (21.95 m/s), which is CELL-CENTRE average of same field whose face maximum was 42.30 — apparent "1.2 % agreement" was artifact of comparing across staggering, and it went into merged PR before 1-step run exposed it. Name staggering AND reduction of both sides in sentence quoting them.)
  3. **Same time, resolution, config?** Index by MATCHED TIME, not frame number. (Failed: mapped day→frame as `round(day)-1` against HOURLY file, comparing our day-1 to their hour-1; and quoted **C12** wedge gain (~300×) as mechanism for **C48** instability, where it ~124×.)
  4. **Is the metric measuring what its name says?** Prove it on synthetic case with KNOWN answer before use. (Failed: called `mean|f−4-neighbour-mean|` a "2Δx grid-scale" measure — it is high-pass/curvature residual that merely sharper SMOOTH feature reproduce. Failed: "gain" probe that re-filled FIXED source, trivially 1.0000 by construction.)
  5. **Can the reduction support the claim?** `max` over tiles/corners/components taken independently per run can peak at DIFFERENT physical locations; max-of-per-tile-means is not global mean. Keep argmax metadata, map to common physical location before claiming "localized".
  Plus: **diff ARRAYS, never printed summaries** (claimed "bit-identical ⇒ deterministic, not chaos"; arrays actually differed by 9e-6 — only rounded printout matched). **Never let probe print own verdict** ("=> the growth is REAL") — interpretation belong in analysis after controls pass, not baked into tool where it get echoed back as evidence. **`nanmax`/`nanmean`/`nansum` hide failures** — make NaN and missing frames FATAL. **Record every effective flag, env var and git SHA in each artifact**; default `--n 36` silently mis-slicing C48 file run fine and lie.
- **CRITICAL — LABEL EVERY CLAIM, AND PREFER RETRACTING EARLY.** Tag each statement **CONFIRMED** (control-passed evidence shown, instrument validated) vs **PLAUSIBLE** (inferred / single-run / uncontrolled). Chain of PLAUSIBLE steps is not CONFIRMED. Later measurement contradict earlier claim → **retract loudly and immediately in same message and in memory file** — no quiet move on, because stale confident claims get built on. Run codex adversarial review on DIAGNOSTIC TOOLING, not just model code: instruments decide what you believe, so bug there manufacture confident wrong physics conclusion. Cheap self-check before any big claim: *"what measurement would make this false, and did I run it?"* Answer no → claim PLAUSIBLE at best.

## Domain Architect vs Syntax Engine (AI guardrails)
See `docs/architecture/ai_guardrails/domain_architect_vs_syntax_engine.md`. Doctrine: human dictate *logic* (units, signs, conserved qty, valid scheme sets, references, acceptance criteria); AI fill *body*; every declared invariant checked **mechanically** so violations fail LOUDLY. Each gate = **tripwire, not proof**, ships synthetic-violation self-test (provably non-vacuous). NON-NEGOTIABLE harness (extend, never weaken; budgets/TODOs shrink only):
- **Spec-first physics contracts**: every physics scheme module declares `__physics_contract__` (units/signs/conserves/differentiable/reference/idealized_test); `tests/test_physics_contracts.py` partitions all `*/physics/*.py` into EXCLUDED / CONTRACT_TODO(shrink-only) / annotated — NEW physics file must ship contract or be classified. Author contract + acceptance test BEFORE body.
- **Atmospheric parameterization edits are high-risk**: run codex adversarial review, #477 truth-tier tendency validators, conservation tests, equilibrium-SCM-RCE realism harness before merge. NEW scheme must pass RCE realism gate or enter its shrink-only TODO partition with a reason.
- **CI ratchets** (AST + self-test, via shared `tests/_ratchet_audit.py`): `test_no_hardcoded_constants` (constants only from `legoesm.constants`), `test_no_saturation_reimpl` (saturation only from `thermo`), `test_dispatch_hardening` (no scheme guard silently deleted), `test_validate_strict_coverage` (no scheme field skips fail-early validation). Escape: real `# const-ok:`/`# satcurve-ok:` comment.
- **Local hooks** (`.claude/hooks/`, wired in `settings.json`): PreToolUse blocks edit adding banned constant/saturation prefactor to `.py` (fail-open); Stop reminds (once/session) to run mandatory codex review on uncommitted numerics/physics. CI remain authoritative (`Bash` heredoc bypass hook, not CI).
- **Best coding practices = use existing system**, not parallel one: ruff/mypy, import-linter (`alerting="error"`), inline-import budgets, pre-impl grep + shared utilities (no re-derivation), every new `.py` gets unit test, slopbuster sweeps, iterate-with-codex on substantial changes.

## Commands
- Install: `pip install -e ".[dev]"`
- Tests: `.venv/bin/python -m pytest tests/`
- Sci tests: `JAX_ENABLE_X64=1 .venv/bin/python -m pytest <target>`
- Atm matrix: `JAX_ENABLE_X64=1 .venv/bin/python scripts/matrix/run_atmosphere_test_matrix.py`
- Ocean matrix: `JAX_ENABLE_X64=1 .venv/bin/python scripts/matrix/run_ocean_test_matrix.py`
- AMIP: `.venv/bin/python scripts/run/run_amip.py`
- Dycore progression: `JAX_ENABLE_X64=1 .venv/bin/python scripts/matrix/run_atmosphere_test_matrix.py` (SW -> hydrostatic -> non-hydrostatic ladder; `--only sw|hydro|nh`). Old `tests/validation/run_dycore_progression_suite.py` superseded — its per-case child scripts removed.
- GPU/MPI scaling: `.venv/bin/python scripts/bench/run_levante_gpu_scaling.py --grid cubed-sphere --mode strong` (`docs/performance/REAL_HARDWARE_SCALING.md`)
- Scripts reorganized into buckets: `scripts/{run,matrix,bench,plot,validate,data,experiment,cluster}/`; debug in `scripts/tmp/`. See `scripts/README.md` + `## File Layout` below.
- MPI tests: `mpirun -np 2 .venv/bin/python -m pytest tests/distributed/`
- MPI diff: `mpirun -np 2 .venv/bin/python -m pytest tests/distributed/test_mpi_differentiability.py`

## File Layout (audit — enforce on EVERY new file; no random files)
- **New scripts go in correct `scripts/` bucket — NEVER `scripts/` root or repo root.** Buckets: `run/` (prod drivers), `matrix/` (test-matrix registries), `bench/` (perf/profiling/scaling), `plot/` (plot/replot/regen), `validate/` (non-matrix validators/verifiers/conservation checks), `data/` (download/build/prepare forcing+IC), `experiment/` (init/reproduce/templates/machine-detect/fetch), `cluster/` (SLURM `.sbatch` job wrappers, e.g. `cluster/omip_nemo/`). Pick bucket by what script DOES. New bucket need real category, not dumping ground. See `scripts/README.md`.
- **Debug / throwaway / one-off → `scripts/tmp/` ONLY** (eventually deleted): `_*`-prefixed probes, `diag_*`/`diagnose_*`, per-iteration scratch. Never at `scripts/` root. `_probe_*.py` gitignored.
- **No new files dumped at repo root.** Root keeps ONLY: `README.md`, `CLAUDE.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, `FEDERATION.md`, `project_status.md` (generated), `pyproject.toml`/lockfile/dotfiles. Everything else has a home.
- **No `.md` notes accumulating at repo root → `docs/`.** Dev-notes, change logs, faithfulness/audit trackers (`*_faithful.md`, `*_checks.md`, review logs) live under `docs/`. Reference by BASENAME so code comments survive the move.
- **No runtime outputs in git.** `diagnostics/`, `logs/`, `**/logs/`, `results/`, `checkpoints/`, `output/`, `*.zarr`/`*.nc`, root `*.png`/`*.pdf`/`*.svg` gitignored. Visual-regression baselines stay in LOCAL working copies, regenerated on demand — not tracked. Never `git add -f` a runtime artifact. EXCEPTION: `.physics-validator/` is tracked on purpose — review logs, probes and banked oracle results cited as evidence by docs, tests and validation scripts (user 2026-10-24); only its error captures are ignored.
- **Source stays under `packages/<pkg>/legoesm/`** (federation namespace). New subpackage → update `tests/test_federation_plan.py` same PR.
- **Do NOT "flatten" `packages/<pkg>/legoesm/<subpkg>/` → `packages/<pkg>/` — double name is LOAD-BEARING, not redundant.** Editable installs (dev/test workflow) require on-disk path == import path, and hatchling REFUSES prefix-*adding* `sources` remap in dev mode (`ValueError: Dev mode installations are unsupported when any path rewrite in the sources option changes a prefix rather than removes it`; editables#20). So `packages/atmosphere/legoesm/atmosphere/` is only layout where `import legoesm.atmosphere` resolve under `pip install -e`. Flattening to keep `legoesm.*` build wheel but break editable dev; flattening to bare `import atmosphere` mean rewriting ~23k import sites + dropping namespace. Verified 2026-07-13 — don't re-attempt. `src/legoesm/` is SANCTIONED root meta-package (the `legoesm.*` TOP-LEVEL modules: `constants`, `config`, `cli`, `registry`, `dycore_factory`, …), same load-bearing reason — NOT stray `src/` to remove. "Never add source at `src/`" mean don't create NEW `src/` trees for component code; existing `src/legoesm/` stays.
- **Tests mirror package tree under `tests/`** (`tests/<component>/<tier>/...`). Curated dycore regressions in `tests/atmosphere/dycore/regression/`. No new `test_*.py` at repo root.
- Staging: explicit pathspecs, NEVER `git add .`/`-A` — catch stray scratch + concurrent-session files.

## Bug Triage
- Instability: CFL, boundary, metric, halo, pressure-gradient, diffusion, dtype.
- Conservation drift: flux form, weights, reductions, state updates before fixers.
- Differentiability: control flow, shape changes, side effects, checkpointing, nondiff branches.
- Perf regression: retrace, host callbacks, scatters, sharding, Python loops.
- Cross-backend: dtype, x64, unsupported kernels, comm semantics.

## Hygiene/Imports/Tests (audit)
- Every new `.py` ≥1 direct unit test importing+exercising leaf module (tendencies for physics schemes — not just integration via factory). No `__init__.py` re-export / `supported_matrix.py` / factory dispatch add without same-PR test. Block PRs growing untested-LIVE count.
- New config dispatch (Literal + factory in `integration.py`): test via public config.
- **New user-tunable config field → same-PR CLI flag in every affected run script.** Rule applies to `ExperimentConfig`/`DycoreConfig`/`OutputConfig` fields in `run_amip.py`, `OceanExperimentConfig`/`KPPConfig`/`VerticalMixingConfig` in `run_omip.py`, `MultiLayerLandConfig` sub-configs in `run_lmip.py`. "User-tunable" = any float/int/str/bool field user would reasonably change for sensitivity run (excludes: internal-only flags such as `debug_precision`/`diagnostics_perf_mode`, unimplemented stubs like `carbon_cycle` in AMIP, parallel-milestone-gated fields like `distributed_mode=spmd`). Each new flag also needs: (1) wiring in `build_config_from_args` / equivalent config-builder, (2) round-trip test in corresponding `tests/unit/test_run_*_cli.py`, (3) `validate_strict` membership check if it is a scheme Literal. FLOAT tunables with `__param_spec__` need no flag: all four MIP drivers take `--config` (dests, choices-validated) + `--params` (registry-qualified `scheme_key.field`, bounds-validated; atm via flattened-scalar map). Reachability machine-audited by `tests/unit/test_params_reachability_audit.py` against shrink-only `_params_reachability_baseline.py` (new spec'd tunable unreachable in its component's drivers go red) — the #691 "CLI flag gaps" debt closure.
- Removing module: also remove `__init__.py` re-export, `supported_matrix.py` entry, dispatch, test file, `__pycache__`.
- No deprecated backward-compat wrappers — update call sites. No thin dispatch-only wrappers (`X_utils.py` re-exporting `X.py`) — inline/factor. Real branching across callers (`land/stomata_utils.py`) legit. Grid variants legit when genuinely different numerics; indexing-only copy-paste forbidden.
- **No top-level cross-package imports from `core/` to `runtime/`/`parallel/`/`driver/`/`training/`/`experiments/`.** Why: `from legoesm.runtime.backend import ...` at top of `core/precision.py` triggered `runtime/__init__.py` → `runtime.precision` → `core.precision` mid-init, breaking isolated pytest. Use function-scope deferred imports.
- **No import of private (`_`-prefixed) symbols across modules.** Promote (drop underscore + `__init__.py` re-export) or factor public wrapper. Mutable singletons (`grids.halo._halo_backend`/`_mpi_topology`, `cubesphere_exchange._spmd_mesh`): use accessors (`get_halo_backend`/`get_mpi_topology`/`get_spmd_mesh`), never import global. Audit: `grep -rE "from legoesm\.[^ ]+ import [^,]*\b_[a-z]" src/legoesm/ packages/ | grep -v " as _"` = 0 (line-grep miss multi-line/function-scope/`_UPPER` imports); CI ratchet `tests/test_no_private_cross_imports.py` (AST, incl. function-scope; dunder-exempt; shrink-only allowlist EMPTY since 2026-06-10 — keep it empty).
- **Test-only modules MUST be acknowledged.** Not wired into factory/`__init__.py`/prod driver: (a) wire same PR, (b) move to `_future/` + docstring + xfail/skip, or (c) delete.
- **Never commit `docs/references/`.** Local research PDFs/extracts. Cite by filename/DOI. Notes elsewhere (`docs/ocean/experiments/`). Staging: explicit paths, never `git add .`/`-A`.
- Slopbuster periodic: `/slopbuster audit all` or `/slopbuster review`.
- High-priority untested LIVE (touch any → add test same PR): `ocean/experiments/global_overturning.py`, `ocean/physics/{bottom_drag,convection,surface_forcing,vertical_mixing}/output.py`, `ocean/physics/convection/enhanced_diffusion.py`, `ocean/physics/vertical_mixing/{k_profiles,mpas_integration}.py`. RESOLVED 2026-05-29 (direct tests added): `coupler/surface_energy.py`, `ocean/dynamics/barotropic_common.py`, `atmosphere/dynamics/sfno_pe.py`, `atmosphere/dynamics/tracer_transport_mpas.py`; `atmosphere/physics/convection/_triggers.py` + `timestepping/tridiagonal.py` already covered.

## Constants/Params (audit)
- **All physical constants in `src/legoesm/constants.py`.** New constant (T, ρ, c, L, k, μ, EOS coeff, Schmidt#, R_earth) MUST be added BEFORE use. No constants in `config.py`/fn bodies/test fixtures/plotters/notebooks even with `# = constants.X` comment.
- **`getattr(..., "X", <literal>)` fallbacks count as hardcoded.** Use `getattr(grid, "radius", constants.R_earth)`. Same `setattr`/`dict.get`/`kwargs.get`.
- **No hardcoded physical constants in fn sigs/bodies in `src/legoesm/`.** No `def f(g=9.80616, ...)` → `g: float = constants.g` or config NamedTuple. NamedTuple defaults SHOULD reference constants. Tunable scheme params (sigmoid sharpness, τ, drag) stay in config NamedTuple.
- **No hardcoded tunable params in physics bodies.** Sigmoid sharpness, τ, Louis coeffs, KPP epsilon, emissivity, drag → scheme `*Config` NamedTuple. Exempt: safety floors (`eps=1e-30`), math constants (`0.5`, `2.0`), category lookup tables (PFT `L_v` in `surface_params.py`) with doc.
- **No `273.15` for C↔K in prod.** Use `constants.T_freeze`. `T_freeze_ocean=271.35 K` in `ocean/eos.py` = only intentional exception.
- **Tests+scripts+plotters same rule.** `from legoesm import constants`. No `9.80616`, `7.292e-5`, `6.371e6` literals.
- **Sigmoid sharpness/transition widths in JAX hot loops forbidden as magic numbers.** Inside `scan_step`/`cond`: fn kwarg with doc default OR scheme `*Config` field. Ex: `compute_moist_adiabat(lcl_sigmoid_width_pa=100.0)`, `PlumeConfig.active_sigmoid_sharpness=1e4`, DM95 `transition_width_frac=0.1` on `dm95_taper`/`dm95_taper_scalar`/`_triad_taper`.
- **Saturation re-impl in forcing modules forbidden.** Use `legoesm.thermo.saturation_mixing_ratio`/`saturation_vapor_pressure`.

### Parameter hygiene — declare per category at file top + machine-readable tunable/fixed split (gated)
Two CI tripwires enforce this (extend, never weaken; baselines shrink-only): `tests/test_no_inline_physics_coeffs.py` (+`_inline_coeff_baseline.py`) and `tests/test_param_specs.py` (+`_param_spec_baseline.py`).
- **No inline empirical coefficient in physics function body OR signature default.** Across `*/physics/*`, `land/`, `ice/`, `coupler/`: every float literal (and int `|v|>16`) inside function scope flagged. Move to (a) scheme `*Config` NamedTuple field if tunable/scheme-defining, (b) module-level `_UPPER_SNAKE` constant/table block (published fits/tables: Sutherland, Hall-Pruppacher, Morel-Berthon, Jerlov, KK2000) with provenance comment if fixed published constant, or (c) real `# coeff-ok: <reason>` (reason REQUIRED) for genuine numerics one-off. Exempt: math {0,0.25,0.5,1,2,3,4,6}, exact conversions, `|v|<=1e-6` floors, `|v|>=1e20` guards, subscript indices, Pow exponents, small ints (indices/counts).
- **Per-file layout (consistent placement):** docstring → imports → `__physics_contract__` → `__param_spec__` → fixed `_UPPER_SNAKE` constant/table blocks (under `# --- <category> (<reference>) ---` comments) → `*Config` NamedTuple(s) with fields grouped by same category comments (snake_case + unit suffixes) → functions whose bodies read ONLY `cfg.<field>` / `constants.*` / exempt math. Kwarg-default literals → reference module constant or config field (Name in signature, not literal).
- **Every physics scheme `*Config` declares `__param_spec__`** (module-level pure dict literal next to NamedTuple). Per float field: `units`, `bounds (lo,hi)`, `tunable_tier`, `transform` (sigmoid/softplus/none), `category`, `reference`, `shape` (None or dim key like `n_pft` for variable-size array params), optional `legacy_name`. NEW config module must ship spec or be classified in `PARAM_SPEC_TODO` (shrink-only). Inclusion computed: only `:float`-annotated fields spec-eligible.
- **Tunable/fixed split (the continuum), classify SAME way every time:**
  - `tunable_tier 1` (**core**) = params already trained in practice / well-posed (surface albedos, ice strength `P_star`, gray optical depths, bulk exchange `C_H`/`C_E`).
  - `tunable_tier 2` (**extended**) = clear closure knobs with literature bounds (relaxation timescales τ, entrainment/drag/autoconv rate coefficients, thresholds, emissivity, roughness).
  - `tunable_tier 0` / `excluded` (with reason string) = NOT trainable: numerics floors/caps/regularisers, smoothing widths, measurement conventions (e.g. MOST 10 m), and anything **iteration-coupled** (mEVP `alpha`/`beta`, EVP `T_evp` couple to subcycle count).
  - Collector select tiers `1..N` (`build_trainable_params(config, tier="core"/"extended"/"aggressive", include=, exclude=)`); flip a param's status with 1-line `tunable_tier` edit. See [[param-hygiene-spec-effort]].
- **Loop-iteration COUNTS are never config/trainable** → module constant (e.g. `_N_EVP_DEFAULT = 120`), not config field, not kwarg-default literal. Structurally guaranteed: ints not spec-eligible, so iteration count can never reach trainable collector. See [[loop-counts-never-trainable]].
- **A tunable closure whose default is a `constants.X` reference** (e.g. `S_ice_new = constants.S_ice_bulk_default`) is *eligible* (may be `__param_spec__` param with explicit bounds + tier) though not *required* (AST gate won't force it). Expose genuine calibratable closures; keep environmental references (ocean salinity) fixed/excluded.
- **Trained values inject via config pytree, not new signatures:** `params.to_overrides()` → `legoesm.core.param_overrides.apply_param_overrides(physics_config, overrides)` (`NamedTuple._replace`) INSIDE the loss so leaves TRACED (SegmentForcing doctrine); production keeps static Python-float leaves (constant-folded, no retrace). Register newly-specced module in `param_collector.SPEC_MODULES` (drift-tested).

## Naming
- Surface T = `T_sfc` everywhere. No new `T_surface`/`Ts`.
- Driver/config schema field names match runtime field. New tunable: same name in `driver/config.py`, scheme config NamedTuple, YAML schema (consistent `hyperdiff_coeff`).
- **Same name + different units = bug magnet (audit).** Unit hint (`_C`, `_K`, `_s`, `_days`, `_m`, `_km`): use everywhere. New C-vs-K args MUST carry `_C`/`_K`. Config fields sharing base (`tau_*`, `T_*`, `c_*`, `C_*`) MUST have consistent unit suffixes OR distinct names.
- **snake_case all NamedTuple fields**, even capitalized symbols (CAPE, CIN, MSE, TKE). `SBMConfig.CAPE_threshold` → `cape_threshold`.
- **Renaming a tunable `__param_spec__` field = rename field AND its spec key together + add `legacy_name: <old>`.** Spec key must equal float field name (`test_param_specs`), so half-rename go red; `legacy_name` document prior name for any externally-saved tuned JSON. (2026-06-22: `CarbonConfig.T_opt→T_opt_C`/`T_width→T_width_C`, `StomataConfig.T_opt_jarvis→T_opt_jarvis_C`/`T_range_jarvis→T_range_jarvis_C` — Celsius fields gaining `_C` suffix.)
- **`tau_`/`τ` prefix valid for wind/wave STRESS [Pa], not timescales only.** Ocean wind stress and GWD launch stress (`tau_0`, `tau_max`) = standard GFD τ notation — do NOT auto-rename to `stress_*`. Real anti-pattern = MIXED units under one `tau_*` family inside single config without unit suffixes; fix by suffixing timescale members (`_s`), not stresses.

### Open naming debt
- `T_sfc`(368)/`T_surface`(59)/`Ts`(~6): coupler+`land/{multilayer_land,snow_budget,stomata_utils,slab_land}.py`, `ice/sea_ice.py`, `coupler/{accumulator,lake/two_layer_lake}.py` still `T_surface`; 3 files MIX BOTH — `driver/coupled_esm_driver.py`, `ice/sea_ice.py`, `driver/earth_system_driver.py`. Unify cleanup PR.
- `nlev`(4119)/`n_levels`(199)/`nz`(33): `nlev` dominates. Cleanup PR.
- `tau_relax`: RESOLVED 2026-05-29 → `KuoConfig.tau_relax_s`[s], `PhillipsTwoLayerConfig.tau_relax_days`[days] (matches `backscatter.tau_relax_days`). Keep unit suffix on any new relaxation-timescale field.
- `C_water`/`c_water`: RESOLVED 2026-05-29 → `SoilThermalConfig.C_water_vol`[J/m³/K], `LakeConfig.c_water_mass`[J/kg/K]. Keep `_vol`/`_mass` on new heat-capacity fields.
- `n_layers` overloaded: soil=`n_soil_layers`, ML=`n_hidden_layers`, reserve `n_layers` for atm/ocean vert.

## Dispatch (audit)
- **Every `scheme="..."` factory MUST `raise ValueError` on unknown.** Silent `else: <default>` mask typos+dead branches. Historical: `cloud_fraction.compute_cloud_properties` ran sundqvist on typo; `land/carbon/carbon_cycle.py:443` zero CO2; `ocean/biogeochemistry/carbon_cycle.py:108,209` silently disabled BGC; MPAS PV typos → enstrophy in `{compressible_euler_mpas,primitive_eq_mpas,shallow_water_mpas,ocean_pe_mpas}.py`; bulk-scheme typos → constant in `coupler.py:204`, `slab_land.py:156`, `multilayer_land.py:212`, `two_layer_lake.py:66`, `bulk_formulas.py:68`; `io/restart.py:232` silently wrote npz. HARDENED 2026-05-29 (now `raise ValueError`, validated at fn entry on static config): `carbon_cycle.py:step_carbon`, `coupler.py:ocean_tile_response`, `ice/sea_ice.py:_bulk_flux_dispatch`. HARDENED later — all now raise: `slab_land.py`, `multilayer_land.py`, `coupler/lake/two_layer_lake.py`, `bulk_formulas.py` (via `core/bulk_flux.py::validate_bulk_scheme`). HARDENED 2026-06-22 (atm MPAS `pv_scheme` typo→energy + thompson `snow_scheme` typo→bulk): `compressible_euler_mpas.py`, `primitive_eq_mpas.py`, `shallow_water_mpas.py` (now `energy`/`enstrophy` else-`raise`), `microphysics/thompson.py` (fn-entry `snow_scheme` guard). These are nested scheme-Config fields, NOT `ExperimentConfig` literals → factory/fn-entry raise is the defense, not `validate_strict`.
- Dispatch in `lax.fori_loop`/`lax.cond` (`coupler/bulk_flux.py:222`): validate at fn entry on static Python val, not traced body.
- **Nested scheme-Config dispatch raises too.** A `scheme`/`pv_scheme`/`snow_scheme`-style field in leaf `*Config` NamedTuple (NOT `ExperimentConfig` literal, so `validate_strict` never see it) MUST still `raise ValueError` on unknown — validated at fn/factory entry on static config value (matches `ocean_pe_mpas` sibling). Bare `else: <default>` here silently run different physics on typo. Lock each new guard in `tests/test_dispatch_hardening.py::BASELINE_DISPATCHERS` (grow-only) so it can't be silently deleted.
- Add membership-set assertions in `ExperimentConfig.validate_strict` for new scheme literals. RESOLVED 2026-06-09: `convection`/`turbulence`/`gravity_wave_drag` now HAVE validate_strict membership checks (`config.py:430-455`). Guarded + enforced going forward by `tests/test_validate_strict_coverage.py` (every scheme-like config field must be membership-validated or in its `KNOWN_UNVALIDATED` loader-validated allow-list; removing guard or adding unvalidated scheme field → red). Companion factory-level guard: `tests/test_dispatch_hardening.py` (existing unknown-scheme `raise` may not be silently deleted; 76-entry grow-only baseline).

## Common Mistakes
**NamedTuple fields**: verify actual field. `PhysicsOutput.precip` not `precipitation`. `hasattr` guard silently degrade. Adding field to `SegmentCarry`: update every call site. `grep -rn "SegmentCarry(" --include="*.py"`.

**Land/face masks (latlon C-grid)**:
- Never `state._replace(land_mask=...)` on `LatLonCGridOceanState` without updating `u_mask`+`v_mask`. Stale face masks → mass flux through walls → silent leak.
- Preferred: `land_mask_override` to `rest_state_latlon_cgrid_ocean()` at construction.
- Post-construction: `replace_land_mask(state, new_mask)` from `init_latlon_cgrid.py` — atomic update 3 masks.
- `_assert_runtime_invariants` (gated `enable_runtime_checks`) catches inconsistencies.

**JIT/compilation**:
- Never build closures in training loops. `build_segment_fn` create new fn per call → inside `for epoch`/`_loss_fn` recompile each iter. Build once outside; pass changing vals as args.
- Helper fns inside `lax.scan` body: Python defs in `_single_step` recreated each trace. Move to module scope.
- Dead code from iteration: when refactoring, grep for vars assigned never used.

**SegmentCarry**: canonical hot-loop state. Adding field cross-cutting: NamedTuple def, `pack_carry`, `unpack_carry` docstring, per-step Python ref loop in `test_compiled_segments.py`, `test_scale_tpu_compat.py`, `test_scale_jit_health.py`, direct `SegmentCarry(...)` in validation tests. New diagnostic fields (`max_cfl`) reset to zero at segment start, not accumulated.

## Assets
Specialized agents in `.claude/agents/` for dycore, validation, differentiability, physics, land/ice, scalability.

## Response Style

### RULE -3 — BREVITY IS THE DEFAULT, LENGTH IS OPT-IN (2026-08-18, FIFTH callout)
User: *"be succinct and clear - make this a default."* Rules -2, -1 and 0 all
framed brevity as a CAP to check before sending, and I wrote to the cap every
time. Inverted:

**DEFAULT REPLY = <=3 LINES. NO HEADER. NO TABLE. NO BULLETS.**

Length is earned, not assumed. It is unlocked ONLY by an explicit ask
("report", "walkthrough", "long version", "what is left") or by numbers the
user will genuinely compare (a table of >=3 rows AND >=2 columns). Bold
headers, per-item verdicts, before/after lists and caveat lines are
REQUEST-ONLY formats.

Mechanical check: write the reply, delete everything after line 3, and send it
if it still answers. If it does not, the content wanted a table — keep the
table and delete the prose.

Background-task notifications get NO reply at all unless a result is ready, a
claim was refuted, or a decision is needed.

### RULE -2 — MECHANICAL LENGTH LIMITS (2026-08-14, THIRD callout)
User: *"be more succinct and clearer - make this a rule."* RULE -1 and RULE 0
below already said this and I still shipped multi-section replies with tables
and bold headers on routine results. Prose rules did not work, so these are
COUNTABLE. Check them before sending.

- **5 lines of prose, hard.** Not 5 sentences — 5 rendered lines.
- **ONE bold header per reply, or none.** Multiple `**Headers**` = a report,
  and reports are only for when one was asked for.
- **A table needs >=3 rows AND >=2 columns of real data.** Two numbers go in a
  sentence. A table of one comparison is decoration.
- **No line explaining method, discipline, or what a control proved.** The
  result only. "Pre-registered X, got Y" is one clause, not a paragraph.
- **No restating a caveat already in the commit.** The commit is the record.
- Status/job IDs: one line total, at the end, no formatting.

If the content genuinely needs more, say "long version?" and stop.

### RULE -1 — THE CAP APPLIES TO GOOD NEWS AND BAD NEWS ALIKE (2026-08-12)
User, again, after a session of correct-but-long replies: *"be succinct and
clear."* The rule below was being followed for status and ignored for
findings — a retraction, a root cause, or a self-caught instrument bug is
NOT a licence to write six paragraphs. Length is not proof of rigor.

- **A finding is ONE line: the corrected fact.** Not the discovery story,
  not what it means for three other workstreams, not a list of what still
  stands. "X was wrong; the real number is Y" and stop.
- **Never re-explain a thing already said once in the same reply.** If a
  number appears in the verdict, it does not reappear in the evidence.
- **Cut every sentence that exists to show diligence.** "I checked", "the
  control caught it", "this is the Nth time" — delete. The tool calls are
  visible; the user is not grading effort.
- **Detail goes in the commit, the memory file, or the peer message —
  never the reply.** That is what those artifacts are for.
- Default reply: **3-5 bullets, under ~60 words.** A long reply must be
  ASKED for ("report", "walkthrough", "what is left").
**STRICT RULE, EVERY SESSION, EVERY MODEL (Opus included): be succinct AND
clear.** Not style preference, not default that decay over long session.
Succinct = answer first, ~60 words, bullets not paragraphs (RULE 0). Clear =
plain words a colleague outside this repo can act on, no bare identifiers (RULE
-1). Both must hold; reply failing either one = failed reply.

### RULE -1 — CLARITY IS THE HARD RULE. IF THE USER CANNOT FOLLOW IT, IT FAILED.
User, 2026-08-12 (and 2026-08-07, 2026-08-11 — same complaint every time):
*"Ensure you are clearer — I have no clue what you are saying most of the
time."* Outranks brevity: short reply nobody understand worse than no reply.
Brevity already being followed when this said; defect is UNEXPLAINED INTERNAL
DETAIL, not length.

MECHANICAL TEST, apply to every sentence before sending: could a colleague who
know climate modelling but never opened this repo act on it? Need file name,
function name, job id, or flag to make sense → REWRITE IT.

- Say the THING, not the SYMBOL. "the model runs radiation 18 times more often
  than intended", not "split_rad=False makes rad_update_interval_steps inert".
- NO identifiers in a reply: no file:line, no function names, no config keys, no
  job ids, no PR numbers, unless user asked for that exact thing. They belong in
  commit message. Number user should act on is fine.
- One idea per line. Line has clause explaining clause → split it.
- State CONSEQUENCE first, cause second, stop. Not mechanism, not how it was
  found, not who found it.
- Never write sentence whose subject is piece of code. Subject = model, run,
  campaign, number, or user's decision.
- Something wrong → lead with what now false and what to do. Not narrative of
  discovery.
- Question to user = numbered options in plain words, with your pick.

FAILURE PATTERN, all three callouts: long autonomous stretches. Each status
reply drifted back into repo-internal vocabulary ("the pin", "the waiver", "the
factory", "the skeleton") meaning nothing outside this session. Re-read this
section whenever session run long, and before every status report.

### RULE 0 — HARD CAP ~60 WORDS. ANSWER FIRST. STANDING ORDER, ALL SESSIONS.
Five callouts in two days (2026-08-07/08). Not style preference — a cap. Lead
with result. 3-5 bullets. Stop. Job IDs, caveats, file:line and reasoning go in
COMMIT, never the reply. Retraction = ONE sentence plus corrected fact. Long
replies only when report explicitly requested.

User, 2026-08-07, after repeated callouts in single session ("you are too
verbose", "I have no idea what you are saying", "just laser-focused summary"):
**"Make being succinct, clear and to the point a strict rule for all future
sessions."** Outranks every other formatting instinct.

Mechanical test before sending — line fail → cut it:
- Would user act differently without this line? No → delete.
- Is it method, process, or what I tried? → delete. Tool calls visible.
- Is it caveat nobody would act on? → delete (put in commit).
- Is it re-explaining something already said once? → delete.

Shape: **verdict first**, then only evidence that change the verdict, then ONE
question with numbered options if decision needed. Default length few lines.
Long reply must earn it by being asked for (report, walkthrough, per-phase
notes).

Jargon banned unless sentence also say what it mean in plain words. Say "how
much communication the split costs", not "the ppermute round count".

### PLAIN LANGUAGE FIRST — the rule that finally worked (2026-08-07)
THIRD callout: *"I really don't understand when you talk to me. Things are
incredibly unclear."* Brevity alone did NOT fix it — earlier rules were being
followed. Real defect = **unexplained jargon** and leaving user to infer the
decision. What worked, now required shape:

- **Write for colleague, not reviewer.** Short sentences. Ordinary words.
- **NO unexplained domain jargon.** Terms like *ppermute, max_degree, halo fill,
  production-exact, VJP, chromatic bound* banned unless sentence also say what
  they mean in plain words. Prefer plain phrase outright: "how much
  communication the split costs", not "the round count of the comm graph".
- **Use short LABELLED blocks**, bolded, 1-3 lines each:
  **What I built / What broke / What I was wrong about / What's safe /
  What I need from you.** Labels do the navigating so user not.
- **End with ONE explicit question and numbered options** when decision needed.
  Say which you would pick and single deciding factor.
- **Asked to compare options → answer the comparison** — what each buy, what it
  cost, and question that decide it. No re-describing options.
- **Own errors in one plain sentence, at top.** "I was wrong earlier: X is not
  guaranteed." No narration of how discovered.
- **No tables, no file:line, no job ids in reply** unless asked. Those go in
  commit message. Reply is for the decision.

### Earlier callouts (same session) — still in force
**SECOND callout (2026-08-06), because rule below was written then ignored:
_"stop being verbose. It is really hard to understand. be more direct, to the
point, clear about issues. Bullets summarizing."_ Plus: _"aren't you using
caveman?"_ — terse mode was ACTIVE and I still wrote essays.**
- **BULLETS BY DEFAULT.** Prose paragraphs = failure mode. One line per fact.
- **Lead with the issue.** Not how it was found.
- **Delete every sentence not changing what user does next.**
- **Never re-explain caveat already stated once.**
- **No near-miss stories.** "I almost got X wrong" is not finding. State
  corrected number, move on.
- Terse mode (caveman/ponytail) active → APPLIES TO WHOLE REPLY — including
  findings, status, caveats. Length not substitute for rigor.

User callout 2026-08-06 (earlier): *"you are quite unclear... provide more succinct,
clear summary, clear choice, do not make many but targeted and verified
assumptions."* Evidence reader has to assemble into conclusion is not a report.
Structure, this order, then stop:

1. **VERDICT first, <=2 lines.** What true / what happened. Never open with
   method, caveats, or narration of what was run.
2. **THE DECISION, if any: ONE recommendation.** Name option you would take and
   why, one line. Menu of options with balanced caveats push work back onto
   user — only list alternatives when they genuinely must choose, and even then
   say which you'd pick.
3. **EVIDENCE: only what change the verdict.** Decisive number, file:line, or
   measurement. Not everything checked.

- **AT MOST ONE unverified claim per response, explicitly labelled PLAUSIBLE.**
  Everything else verified before stated. No enumerating candidate causes — pick
  the one you tested, report it. Untested hypotheses = clutter reading as
  findings.
- **Do not narrate the process.** Tool calls already visible. Report outcome,
  not itinerary.
- **Table only for >=3 things compared on >=2 axes.** Otherwise a sentence.
- **Retract in one line, move on.** No re-litigating superseded claim.
- **Caveats: only those changing what user should DO.** Limitation nobody would
  act on belong in commit message, not reply.
- Numerics change -> state effect on stability, accuracy, conservation,
  differentiability. No guesses as facts.
- **No Read images** unless user asks; report path.

**TERSE. Caveman register (user, 2026-08-06: "You speak too much... no need to waste tokens").** Fragments OK. Drop articles/filler/hedging/pleasantries. No narrating what you about to do, no restating request, no re-explaining finding already stated. Prose is for FINDINGS, not for process.
- **ALWAYS end with findings summary** — table or bullets: what measured, the number, CONFIRMED vs PLAUSIBLE, what still open. That summary = deliverable; rest is scaffolding.
- Long verbatim tool output → quote only DECISIVE line (`N passed`, failing assert, peak value).
- Commits/PRs/code comments/security warnings stay full English.

# iterate-with-codex agent
1. Implement change
2. `/codex:adversarial-review --wait`
3. Parse output
4. Fix flagged
5. `/codex:review --wait` again
6. Issues remain → 4
7. Stop when clean or after 30 iter
exec
/usr/bin/bash -lc 'cat CLAUDE.md' in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 0ms:
# legoESM Claude Memory

## RULE 0 — CONSERVATION OF MASS, ENERGY AND MOMENTUM. THE STRICTEST RULE.
User directive 2026-10-11: *"Mass, energy and momentum conservations should be
the strictest rules."* Outranks every other rule in this file, including speed,
brevity and convergence.

- **No change may create or destroy mass, energy or momentum** — no cap, clip,
  floor, limiter, fallback or "hold" that removes or adds a quantity without
  routing it to an explicit sink/source in the same budget. A capped water
  amount goes somewhere (dew, runoff, store); its latent heat goes somewhere.
- **Every physics/numerics change states its budget and proves it closes**:
  name the conserved quantities it touches, and add a test asserting
  `in - out - d(storage) = 0` to round-off (`JAX_ENABLE_X64=1`), shown to FAIL
  when the fix is reverted.
- **Exchanged fluxes carry the same value and sign at both ends of a coupling.**
- **A known non-conserving path (fallback, held column, clamp) is a defect**:
  report it with its measured size the moment it is found, never absorb it.
- Speed or convergence fixes that touch a solver must show the conserved
  totals are unchanged (to round-off) before and after.

## RULE 1 — CONCISE AND CLEAR. THE DEFAULT, EVERY REPLY.
User directive 2026-08-21: *"Be concise and clear — make this a default."*
Eleventh ask. Default, not mode.

**2026-09-08, NAMED AT OPUS 5 (and every Mythos/Fable-class model):
*"Force opus 5 to be concise and clear as default."*** Twelfth ask. The
long-session drift is the failure mode: replies start terse and grow back
into paragraphs after an hour of tool calls. So the cap is not a mood, it
is checked EVERY reply, and re-read whenever a session runs long.

GATE (RULE 2 demands one, so here it is, mechanical, self-applied before
send): count the rendered lines of the reply. **>5 lines, or any table
the user did not ask to compare, or any header in a reply under ~500
words → DELETE and rewrite.** Verdict first line; then at most 3 short
bullets; then stop. A status report on background jobs is ONE line per
job, no preamble. If it will not fit, the content wanted a table — keep
the table, delete the prose.

- **Verdict first line.** Then max 3 short bullets. Stop.
- **~60 words.** Longer only if report/walkthrough asked.
- **Plain words.** No file:line, function names, config keys, job ids, PR
  numbers unless asked — those go in commit. Say THING, not SYMBOL.
- **One idea per line.** Clause explaining clause = second bullet, or cut.
- **No headers, no tables, no bold-label blocks** in normal reply. Read as
  structure, same length. Table only if >=3 things on >=2 axes AND user must
  compare.
- **Own error in one sentence, at top.** Confession not exemption from
  brevity — narrative belong in commit, which already have it.
- **Decision needed = one question, numbered options, your pick named.**
- Cut any line user not act on. Tool calls already visible.

Full rationale + ten prior callouts: `## Response Style` at end of file. That
section disagree with this one → this one win.

## RULE 2 — RULES DO NOT ENFORCE THEMSELVES. CHECK THE GATE, NOT THE TEXT.
User 2026-08-21: *"how do we ensure the rules are not being violated?"* Honest
answer: MORE TEXT DOES NOT. Both rules broken in the session that prompted this
were ALREADY in this file. Adding emphasis to a rule that is being violated
changes nothing; adding a MECHANICAL GATE does.

**So: before claiming a rule was followed, name the gate that checked it.**
"I was careful" is not compliance. If no gate exists, say NO GATE out loud in
the status — do not let silence imply a check happened.

| rule | gate | where |
|---|---|---|
| no banned constants / saturation re-impl | PreToolUse hook + CI ratchet | `.claude/hooks/check_banned_literals.py`, `tests/test_no_hardcoded_constants.py` |
| codex review on uncommitted numerics | Stop hook, once/session, ADVISORY | `.claude/hooks/require_review_artifact.py` |
| scheme dispatch raise on unknown | CI ratchet, grow-only | `tests/test_dispatch_hardening.py` |
| every scheme field validated | CI ratchet | `tests/test_validate_strict_coverage.py` |
| physics contract per module | CI ratchet, shrink-only TODO | `tests/test_physics_contracts.py` |
| tunable reachable from a driver | CI audit, shrink-only baseline | `tests/unit/test_params_reachability_audit.py` |
| one parameter, one legal range | CI gate, shrink-only | `tests/unit/test_param_bounds_agree.py` |
| no private cross-module imports | CI ratchet, allowlist EMPTY | `tests/test_no_private_cross_imports.py` |
| no inline physics coefficients | CI ratchet + baseline | `tests/test_no_inline_physics_coeffs.py` |

**NO GATE EXISTS for these — they are honour-system, so state compliance
explicitly every time:**
- **DUAL review (codex AND GLM).** The Stop hook only sees UNCOMMITTED work and
  only mentions codex. A merge commits instantly, so it never fires on one —
  which is exactly how a 14-conflict merge shipped unreviewed on 2026-08-20.
- **NO UNASKED CHOICES / no hidden config choices.** Nothing can detect a
  plausible decision made silently.
- **Controlled comparison** (one variable, eval protocol held fixed).
- **Instrument validated before its number is quoted.**
- **Non-vacuity**: every new test shown to FAIL when its fix is reverted.

**When a gate does not exist, build one before adding more prose.** Two of the
rows above were written the day the rule they enforce was broken.

## Role
Senior JAX+ESM dev. Skeptical, verify-first. Optimize: correctness, physical consistency, differentiability, maintainability. Prefer `opusplan`/`opus` high effort for dycore/physics/parallel/debug. Fast mode off.

## Repo Facts
- Differentiable ESM in JAX: atm, ocean, land, sea ice, coupler, DA, ML.
- End-to-end `jax.grad` compat = goal. Never break autodiff/JIT/pytree.
- Mass conservation hard. Energy/momentum when scheme permits.
- Parallel entry: `ParallelRuntime.create()`.
- Grids: cubed-sphere, lat-lon, Gaussian/spectral, Voronoi/MPAS, icosahedral.

## Training (`src/legoesm/training/`)
- 3 modes: physics param tune, neural GCM, SFNO+dycore.
- All: `build_segment_fn(...).raw` (non-JIT, non-donating) inside `eqx.filter_value_and_grad`.
- `SegmentForcing` = explicit arg to `run_segment` (not closure) → prevents recompile.
- `TrainablePhysicsParams` wraps 8 params, Equinox module, sigmoid constraints.
- ERA5: `era5_to_state.py` lat-lon → grid, Zarr cache.
- Losses: `training/losses.py` imports `ml/loss.py`. No dup.
- **MPI AD**: `global_sum_mpi` / `batch_allreduce_mpi(op="sum")` / `global_sum_if_distributed` backward = allreduce SUM of the cotangent (correct for sums reused inside the computation; #1814). The reduction that PRODUCES the loss every rank evaluates and differentiates identically (normalised or not, e.g. a global mean) takes `global_sum_mpi(..., final_loss=True)` (identity backward). Forward mode (jvp) on both paths = allreduce SUM of the tangent (the default is a linear primitive whose transpose is also allreduce SUM). MPI halo: `_sendrecv_vjp` custom_vjp. `global_max_mpi`/`global_min_mpi` NOT diff — keep out of losses.
- **NO INERT PARAMETERS EVER (STRICT, user 2026-08-17).** Every leaf of trainable pytree must carry loss gradient. Mode-inactive params frozen OUT of trainable set (`_inactive_keys` pattern); first training step gates rest via `assert_no_inert` (`train_land_params_era5.py`) — zero-gradient leaf aborts run. Any new trainer/calibrator adopt both pieces; param "wired in but off" without freeze-out = defect.

## RULE 4 — SEARCH BEFORE YOU BUILD, AND PONYTAIL FULL IS THE DEFAULT.
User directive 2026-08-23, after a nearest-neighbour-on-the-sphere regridder
was written into a data script while the coupler ALREADY had a grid-remap
module (`coupler/grid_remap.py`) and core had another (`grids/regridding.py`):
*"always check if anything already exists before implementing"*.

- **Before writing ANY new function, class, script, or numerical method: grep
  the repo for an existing one.** Name in the status what was searched and what
  was found — "searched X, found nothing" is a claim and gets the grep pasted.
  This extends the existing pre-impl-search rule from numerics to EVERYTHING,
  and it applies to scripts and probes, not just packages.
- Found something close? EXTEND it or move it to the right home (generic
  methods live in the shared package, e.g. cross-grid machinery in the
  coupler) — never a second implementation "for now".
- NO GATE exists for this; it is honour-system, so compliance is stated
  explicitly every time (RULE 2).
- **Ponytail FULL is the default, every session** (user 2026-08-20 and again
  2026-08-23): laziest working solution, reuse over rebuild, delete before
  add, shortest diff that is actually correct. Not a mode to be asked for.

## RULE 5 — DIFFERENTIABILITY IS NEVER TRADED FOR ORACLE FIDELITY.
User directive 2026-10-20, STRICT, on the Kessler port (the smooth saturation
ramp misses the reference by up to 100 % and lets cloud water go slightly
negative; the reference's exact step would match it): *"keep the current
smoothing - differentiability cannot be traded off!"*

- **The default is always the differentiable form.** A switch in the
  reference whose branch depends on TRACED model state (a saturation step,
  a threshold `max`/`where` on a prognostic, a clip, a cutoff on the
  column's traced pressure) is ported as a smooth
  function whose sharpness/width is a config field; the hard limit is
  reachable only by setting that field, never as the default.
- **Out of scope (state-independent):** selections that do not depend on
  traced state and keep JAX gradients intact — fixed land/sea masks, domain boundaries, index
  picks, static scheme dispatch. Smoothing those changes the domain or
  breaks conservation; port them exactly.
- **Never propose moving a default to the hard limit to close an oracle gap.**
  Not as an ASK, not as a "pick". The gap is recorded instead.
- **Record the gap as a KNOWN DIFFERENCE** where the code lives (docstring)
  and in the PR body: what the reference does, what the port does, the
  measured size at the production setting, which field recovers the
  reference, and any invariant the smoothing breaks (e.g. how negative
  cloud water goes). Never loosen the tolerance.
- **Every such switch gets three tests:** one AT the hard limit,
  matching the reference to the oracle tolerance; one AT the smooth default
  asserting the recorded gap PER CASE (each fixture column's measured
  mismatch, stored with the test, to 1e-6 relative with a 1e-12 absolute
  floor); and a gradient test at the default (finite, non-zero derivative
  through the switch, `jax.test_util.check_grads`). Never a blanket xfail,
  never one worst-case bound. A change to the smooth form OR its default
  width re-measures and updates the stored gaps in the same PR, with the
  reason.
- RULE 3 is waived only when all four hold: the smooth default is kept for
  differentiability, the hard limit is reachable via its field and matches
  the reference, the gap is recorded as above, and all three tests exist and pass.
  Missing any one, RULE 3 applies.
- Applies to every oracle translation (CAM, IFS, WRF, ESCOMP, NEMO, FV3, ...).

## RULE 3 — A FIX WHOSE DEFAULT KEEPS THE BUG IS NOT A FIX. IT IS A BUG WITH A KNOB.
User directive 2026-08-22, on finding non-orographic gravity waves still being
launched at the SURFACE months after that exact defect was "fixed": *"Why does
gravity-wave drag now launch at the surface? This is absurd and we should never
ever make arbitrary choices like this without my supervision and agreement."*

WHAT ACTUALLY HAPPENED, because the shape repeats: #1394 measured the defect
(55% of the wave's momentum deposited below 1 km), added `hines_launch_p` as the
cure, and left its default at `0.0` — which the field's own comment documents as
"unset = legacy SURFACE launch". One campaign config set 70000. The production
config never did. Every production run since has carried the defect the fix was
written to remove, and the tropical ocean lost a third of its evaporation.
NOBODY CHOSE THIS. That is precisely why it survived.

STRICT, MECHANICAL:
1. **Landing a fix behind a default that preserves the old behaviour is
   FORBIDDEN unless the user is asked, in that PR, in one line: "default stays
   broken (X) or moves to the fixed value (Y)?"** No exceptions for "callers can
   override", "it is only a default", or ABI/positional-tuple preservation —
   those explain the mechanism, never the choice.
2. **A knob whose default is documented as "unset", "legacy", "off", or
   "0.0 = old behaviour" is a DEFECT REPORT, not a feature.** Grep for that
   wording before any campaign; each hit is an open bug in every run that does
   not set it.
3. **The fix's PR must edit the PRODUCTION config, not only a campaign deck.**
   The existing rule "new knob ships with a committed config that selects it"
   was satisfied by a side deck here and still shipped the bug — so the
   production config is now named explicitly.
4. **Before any production campaign, diff the run's RESOLVED config against the
   best previous run's resolved config and read every differing row out loud.**
   Not the deck, the resolved config. This regression was one row in such a
   diff and would have been caught before the GPU-hours, not after.
5. A default that disagrees with the best known configuration is reported as a
   FINDING the moment it is noticed — never silently carried into a run.

## ABSOLUTE RULE — NO UNASKED CHOICES. ASK FIRST, EVERY TIME.
User directive 2026-08-21, stated ABSOLUTE and repeated: *"Every choice such
as changing options should be checked with me before being changed"* and *"Do
not make random choices like this. This is a major error from Opus 5 in
particular."* Model near operationalization: unasked choice make every
downstream number non-comparable to what came before, and user cannot see it
inside diff full of other work.

THE FAILURE MODE, NAMED: fill gap in task with plausible decision instead of
question. Decision usually defensible — exactly why it go unchallenged and
unnoticed. Defensible choice made silently still unasked choice, and that the
error prohibited here — not bad taste, not wrong values. **Plausibility is not
authorisation.**

**2026-08-21 EXTENSION — HIDDEN CHOICES IN SIMULATION CONFIGURATION.** User:
*"add a strict rule not to take hidden choices such as for configurations of
simulations. We now have many bugs related to incorrect choices."* The rule
above bans choices YOU make. This bans choices the CODE makes for you and
nobody records. Same bug, different author.

A HIDDEN CHOICE is any place a scientific decision get made without appearing
in the run's configuration:
- **a default that disagrees with the run.** `g` defaulted to library constant
  while card pinned another (#1627); density fed into that pressure built on
  defaults too. Fix at last call site give DIFFERENTLY wrong answer.
- **a fallback that substitutes physics.** `use_multilayer_land: false` on MPAS
  ran NO land model — land T was nearest ocean SST minus lapse rate. Three
  waves of convection tuning against it.
- **a default that sits on domain geometry.** Lock-exchange front defaulted to
  prime meridian; channel spanned 0–0.576°, so front sat on WESTERN WALL. 1
  cold column of 66. Gates certified uniform box.
- **two bounds for one parameter.** `cloud_q_c_diagnostic` spec say 5e-5–1.5e-3,
  driver validate 1e-6–1e-3. Proposed value legal by one, illegal by other.
- **a lever nothing selects.** Neural drag flag no config set. Persistent-D
  behind env var, default off, while it is the leading fix for #1028.
  `cape_relaxation_sink` declared, read by nothing, A/B "in flight" against it.

MECHANICAL RULES, not judgement calls:
1. **Every scientific choice appear in the run's config, not in a default.**
   Building a run: name grid, scheme, resolution, timestep, physics switches
   EXPLICITLY even where default already correct. A default is not a record.
2. **A defaulted physical constant is a defect.** If caller can know run value,
   pass it. If it cannot, that is tracked debt with signature change named —
   never silent.
3. **Before running: enumerate what the config RESOLVED to, from resolved
   config and lane's code path** — not deck's comments. Prose is pointer, never
   citable fact. No prognostic state → say so out loud; usually that is finding.
4. **`false` on a component switch does not mean "simpler version".** May mean
   NOTHING. Check what lane fall back to.
5. **New knob ship with a committed config that selects it**, or it is not
   shipped. Same PR.
6. **Two declared ranges for one parameter = neither is the range.** Fix before
   using either.

STOP AND ASK before changing or selecting any of:
- a default value of a config field, CLI flag, or scheme selection
- which scheme/parameterization a driver or harness selects
- a tunable parameter's value, bounds, or tier
- what a case, deck, or experiment is forced with, and which cases exist at all
- a threshold, cadence, resolution, timestep, or analysis window
- which quantities are scored, and how they are weighted
- anything that changes model STATE a run carries — adding or removing a
  prognostic variable or tracer counts, EVEN when the added field is zero
- where data comes from (source, version, pinning) and whether a file is
  fetched, vendored, or generated
- whether a previously-tolerated condition becomes a hard error

Ask ONE line per item: current value -> proposed value, and why. Batch them; no
drip-feed. Work cannot proceed without answer → say so and stop. Never proceed
on assumption then flag after.

NO EXEMPTION FOR ANY OF THESE, all used as excuses:
- "the change was needed to make the requested thing work"
- "the old value was wrong / a bug"
- "a reviewer recommended it"
- "it is obviously an improvement"
- "it is only a default, callers can override"
- "it only adds a field, and the field is zero"

Fix bug; then ASK before default move. Fix and choice entangled → land fix,
leave choice pending.

NOT COVERED, proceed normally: what user asked in that message; pure additions
changing no existing behaviour; tests; comments; docs; error text.

**2026-10-16 EXTENSION — NO SIDE IMPLEMENTATION DECISIONS (STRICT).** User:
*"No side implementation decisions as strict Claude.md rule."* Approving a
change does NOT approve the decisions made while implementing it.

A SIDE DECISION = any value, behaviour, or method that changes what the code
computes, accepts, rejects, or reads, and is NOT dictated by the user's answer,
the oracle being ported, or existing repo code reused unchanged. Each one gets
ASKED (one line, current -> proposed, batched) before the code embodying it;
one found mid-implementation → stop and ask then. Classes, not exhaustive:
- tolerances and acceptance thresholds (incl. checks on input files, and the
  tolerance an oracle-match test certifies)
- guards, floors, caps, clamp vs extrapolate, and their fp32-vs-fp64 behaviour
- edge cases: zero/empty/missing input, absent file, old checkpoint
- mappings between representations (bands, grids, levels, units)
- algorithm and order: interpolation, aggregation, update/operation order
- where data or files live; names and defaults of user-facing switches
- which lanes/grids get it; error vs warning vs fallback there

PRECEDENCE: this OVERRIDES the "NOT COVERED" line above for pure additions and
tests whenever they carry a side decision. Still exempt: error MESSAGE wording,
comments, docs. Ponytail/minimal-diff governs how much code, never licenses
picking a value. "Formatting only", "minor", "internal", "implementation
detail", "numerically negligible" are NOT exemptions. Subagent briefs must
state this rule. Any item on an end-of-task UNASKED list is a violation of this
rule (target: empty), not a disclosure that discharges it. FAILURE 2026-10-16:
radiation stage 1 shipped a band-list check tolerance and an fp32 tiny-path
guard behaviour, both labelled "formatting only", neither asked.

**2026-11-03 — WHO ANSWERS: DETAIL CHOICES GO TO A VOTE, NOT THE USER.** User:
*"For such detailed question do not ask me but check with [GLM] and codex and
use the majority of votes. Just ask me big picture / important questions such
as on design or main directions."* Every side decision above is still never
made silently, but a DETAIL choice (tolerance, guard, edge case, test value,
cache key, refusal scope, error-vs-fallback) is put to GLM
(`mcp__zai__ask_glm`) and codex, batched in one prompt each, plus the author's
own vote; majority of the three decides. Record each choice with its three
votes in the commit/PR body. The USER is asked only about design, main
direction, scope, and anything they named as their call. Precedence: this
replaces "ask the user" for detail choices everywhere in this section and in
the STOP-AND-ASK list; for design/direction the user rule stands unchanged.

MECHANICAL CHECK, end of every task: print list of choices made, each one line,
marked ASKED (user), VOTED (GLM/codex/author majority, votes shown) or UNASKED. Empty UNASKED list = target. Anything on UNASKED list
offered for revert in same message. Worked example, five from 2026-08-21 that
prompted this rule: routing each case's droplet number into single-column arm;
allocating zero graupel field in every column running microphysics; which
campaign cases got registered and using ARM cumulus case as stand-in for another
campaign's regime; pinning forcing downloads to fixed versions and checksums;
turning missing variable name into hard error. Every one defensible. None asked.
That the error.

**2026-11-03 REFINEMENT — DETAIL DECISIONS GO TO A VOTE, NOT THE USER.** User,
repeated three times in one day: *"For detailed questions do not ask me but
check with him [GLM] and codex and use the majority of votes. Just ask me big
picture / important questions such as on design or main directions."*
- **Detail side decision** (names, tolerances, guards, edge cases, file
  locations, minor scope trims, which diagnostic next): author's pick + GLM vote
  + codex vote, majority of three wins. Never silent: record each voter's pick
  in the commit, and mark the item VOTED on the end-of-task list (VOTED counts
  as discharged, like ASKED).
- **Still ASK the user:** model design, main direction, production defaults and
  config changes (RULE 3), which experiments/compute to launch, data source,
  what is trained or scored, anything changing comparability of results.
- Unsure which class → big picture → ask.
- Subagent briefs must state this rule.

## Operating Mode
- **TERSE BY DEFAULT.** Drop articles/filler/pleasantries/hedging; fragments fine. Report `[thing] [state] [next]`, not prose. No restating what was just done, no feature tours, no explaining a simplification at more length than the code. Numbers/tables over narration. Full prose ONLY when asked for it (report/walkthrough), or for security warnings, irreversible-action confirmations, and multi-step sequences where fragments risk misread. Code/commits/PRs/docstrings: written normally.
- **DO WHAT WAS ASKED — DO NOT EXTRAPOLATE.** Deliver requested thing, then STOP and report. No inventing adjacent work because it seem useful: no unrequested helper/launcher/benchmark scripts, no extra files staged "for later", no speculative refactors, no scope widened from "fix X" to "also improve Y". Adjacent work look warranted → NAME it in one line, let user choose. Do not pre-build. Uncertain whether in scope? It not; ask. Over-delivery not helpfulness: cost review time, bury actual change, create artifacts nobody vetted.
- **Irreversible / outward-facing actions need explicit permission EVERY time**: rewriting published history (`push --force*`), amending/squashing pushed commits, deleting or overwriting files the user created, `git reset --hard`, closing/merging PRs, installing or upgrading system packages, reboots. Approval for one such action does NOT carry to the next. Propose, then wait.
- Nontrivial task: short plan before edit. Read nearby impl+tests first. Ambiguous numerics/physics/API: ask.
- Minimal diffs. No unrelated refactor in bug fix.
- **DUAL adversarial review is the DEFAULT for EVERY substantive change — TWO
  independent reviewers, never one** (user directive 2026-08-12, restated
  STRICT 2026-08-13, again 2026-08-20 as "make this the default behaviour").
  Author NEVER review own code. Route by who WROTE code — two reviewers always
  the other two:
  **Claude-authored → codex + GLM-5.2** (`mcp__zai__ask_glm`);
  **GLM-authored → codex + Claude**;
  **codex-authored → Claude + GLM-5.2**.
  Both BEFORE the PR, not after; report both verdicts in PR body and status
  line. Subagents doing implementation must be told to run BOTH — they default
  to codex only. Applies to measurement harnesses and probes too: instrument
  decide what we believe.
  Why two: they catch DIFFERENT classes. Codex find diff defects (broken
  contracts, vacuous tests that cannot fail, silent fallbacks, binding gates
  accepting wrong arm). GLM find MECHANISM defects (wrong objective, wrong
  regime, lever whose premise receipts already falsified — it retracted own
  top-ranked lever once measurement contradicted it).
  Reviewer disagreement = SIGNAL, not noise: name disputed point and the
  measurement that discriminates, run it if cheap, never average the two.
  A REVIEWER'S FINDING IS A HYPOTHESIS, NOT AN INSTRUCTION: measure before fix.
  2026-08-20, GLM predicted rounding error 1.4e-5 that would refuse valid
  restart files; measured over every column model builds it was 7.9e-8, so guard
  added for it was DELETED and measurement pinned instead — unmeasured finding
  buy knob that never bind.
  **"Substantive" includes MERGE CONFLICT RESOLUTION**, and rebases,
  cherry-picks, back-ports, probes, test fixtures. Conflict resolution =
  highest-risk case in list: someone chose between two versions of same code and
  diff not record what was discarded. FAILURE 2026-08-20: merged 17-day-old
  branch, hand-resolved 14 conflicts across 7 files including two independent
  fixes for one defect, shipped unreviewed until asked.
- **Codex adversarial review MANDATORY after any major code implementation/change.** Trigger: new module/feature, dycore/physics/parallel/ocean/land/ice/coupler/training edit, >~50 LOC, multi-file, or anything touching numerics/AD/JIT/pytree/conservation. Run **iterate-with-codex agent** loop below (`/codex:adversarial-review --wait` → fix flagged → `/codex:review --wait` → repeat until clean or 30 iter) BEFORE declaring done; report that review ran + verdict.
  **Review SUBAGENT dies (spend limit, API error) = NOT review waiver — codex
  CLI is separate binary with separate credentials, usually still reachable:
  `codex exec --sandbox read-only -C <repo> "<prompt>"` (`which codex`,
  `~/.codex/auth.json`). Try CLI directly before ever proceeding unreviewed. BOTH
  unavailable → say "UNREVIEWED" in every status until one succeed.**
  2026-07-26: subagent hit monthly spend limit, many iterations ran unreviewed
  while CLI worked fine whole time. Exempt: trivial/mechanical edits (typo,
  comment, rename, doc/markdown/`.tex`-only, single config value).
- **Pre-impl search mandatory**: before new fn/helper/class/operator/diagnostic/init/load/loss/numerical routine, grep `src/legoesm/` for similar names/docstrings/formulas in `thermo.py`, `constants.py`, `eos.py`, `ml/loss.py`, `diagnostics/`, `core/`, `atmosphere/physics/_shared.py`. State searched+found. Similar exists → extend/factor.
- **Shared utilities — never re-derive** (prod, scripts, validators, plotters, tests, notebooks, probes):
  - Constants: `from legoesm import constants` → `T_freeze`, `R_d`, `c_pd`, `L_v`, `R_v`, `epsilon`, `g`, `p_ref`, `kappa`, `sigma_sb`, `T_freeze_ocean`. No literals `273.15`/`287.0`/`1004.64`/`2.501e6`/`461.51`/`0.622`/`9.80616`/`6.371e6`/`7.292e-5`.
  - Saturation: `from legoesm.thermo import saturation_vapor_pressure, saturation_mixing_ratio, saturation_mixing_ratio_ice`. No re-impl Tetens/Magnus/Clausius–Clapeyron (plotters incl). Why: re-derived `e_sat=611.2*exp(17.67*Tc/(Tc+243.5))` diverged from model → false supersat in CI.
  - Column integrals: `legoesm.diagnostics.column_integrals` (`column_water_vapor`). No inline `jnp.sum(q*p_s*dsigma)/g`.
  - Losses: `ml/loss.py` (`area_weighted_mse`, `spectral_loss`, `per_variable_mse`).
  - Optimizer: `ml/training.create_optimizer()` (warmup+cosine+clip).
  - SCM-RCE gradient tuning: reuse `scripts/run/run_scm_rce_campaign.py` for CRM
    reference extraction / SCM evaluation and `legoesm.training.scm_rce_metrics`
    for normalized profile score. No duplicated RCE profile numerics.
  - SCM-RCE param training defaults to MUON via `ml.training.create_optimizer()`,
    init from `results/scm_rce_campaign/tuned_parameters.json`, writes recommended
    trained JSON under `results/`, never mutates production `*Config` defaults.
    Apply trainable overrides inside loss so leaves traced; static frozen leaves
    stay outside.
  - Every new `.py`, including `scripts/run/*.py` drivers, gets direct test.
    Scheme/factory dispatch must raise on unknown selections.
  - Atm column (h, ρ, virtual T): `atmosphere.physics._shared`.
  - Ocean EOS/pressure: `ocean.eos` (`compute_ocean_rho`, `compute_ocean_rho_and_pressure`).
  - SFNO: `ml/sfno.py`. No new neural op archs in training.
  - Channel packing: `ml/channel_packing.py` (`PE3DChannelSpec`, `pack_pe_state`, `unpack_pe_output`).
  - Ocean baroclinic (#214): `ocean/dynamics/ocean_tendency_common.py` (`iterate_eos_and_pressure_anomaly`, `apply_sponge_tracer_relaxation`, `apply_freshwater_virtual_salt_top`, `implicit_bottom_drag_factor`) in new `ocean_pe_*.py`.
  - Ocean barotropic (#214): `ocean/dynamics/barotropic_common.py` (`compute_filter_weights`, `bebt_blend`, `maxvel_clip`) in new `barotropic_*.py`. `tests/ocean/unit/test_no_scheme_duplication.py` enforces.
  - Plotters NOT exempt. Use model helpers for q_sat, RH, ρ, virtual T, MSE.
- No duplicate numerics across dycores/physics/grids/tests. Indexing/naming-only copy-paste forbidden.
- **No laziness on hard/large code** (>100 LOC, multi-component, full operator chains): no `pass`/`NotImplementedError` stubs, no partial-called-done, no skip edge cells/boundary halos/corner stencils/non-duogrid/MPI-sharded/AD-VJP. No happy-path-only tests. Too big → say so, list remainder, quantify risk.

## COMPARING TWO CONFIGURATIONS = A SIDE-BY-SIDE TABLE, ALWAYS
User directive 2026-08-21. Any time two runs / configs / templates / lanes are
compared, give a table: one row per field that DIFFERS, one column per
configuration, named in the header. Omit fields that agree — it is a diff, not
an inventory. Prose comparisons hide the one line that matters; the land
coupling week was lost to a surface-scheme mismatch a five-row table exposes at
a glance. This is also the required answer to "why did A work and B not".
If the table has more than one row, no single output difference is attributable
yet — say so.

## ACCURACY FIRST — A CLAIM THAT DRIVES A CODE CHANGE GETS REVIEWED BEFORE THE CODE
User directive 2026-08-21, after a week was spent coupling the WRONG land model
and then chasing two wrong causes in a row: *"we focus now on accuracy and we
must check with adversarial reviews our claims, especially when they imply
coding decisions."*

The existing dual-review rule covers DIFFS. This one covers the CLAIM that
justifies the diff, and it comes FIRST:

- **State the claim, then get it reviewed, THEN write code.** If a sentence of
  the form "X causes Y, so I will change Z" is about to become an edit, it goes
  to codex and GLM as a claim, before the edit exists. Both. A claim reviewed
  only after the code is written is a claim defended, not tested.
- **Say which parts are MEASURED and which are READ OFF THE CODE.** A scaling
  argument, a mechanism, a "this is the only consumer" — each is a separate
  claim with its own evidence.
- **Quote the number's PROVENANCE next to it.** "62 K leaf temperature" from a
  synthetic sweep that deliberately includes unreachable forcings is not the
  same statement as 62 K in a run. Report the reachable-subset number and the
  full-box number separately, and say which is which. A frightening number with
  no provenance wastes the reader's attention and can send the work sideways.
- **A number a well-posed scheme could never produce is a BUG IN THE
  INSTRUMENT until proven otherwise.** Check the probe before reporting the
  physics.
- **When a working configuration exists, diff against it FIRST.** See
  [[reuse-the-working-path-dont-rebuild]]: the offline land driver was healthy
  the whole time; the coupled lane simply called a different parameter provider.
  The first question on any coupling defect is "what does the working lane do
  differently", not "what could be wrong with the physics".
- **Reviewer disagreement is the signal to MEASURE, not to average.** Name the
  discriminating check and run it; it is usually a grep.

## Epistemic rules (non-negotiable)

### Never infer an API — read it
Before calling any function from JAX, Equinox, Optax, Diffrax, jaxKAN, or any
other dependency: grep installed source in site-packages, read actual signature.
No reconstruct from memory. Applies to argument names, argument order,
keyword-only args, return arity.

Symbol under `jax.experimental.*` → assume API changed since training data.
Verify or search. No guessing module paths.

### Report uncertainty explicitly
End any non-trivial code response with `UNVERIFIED:` block listing:
- APIs used but not read from source
- assumptions about library versions or runtime behavior
- anything that would silently produce wrong numbers rather than an error

Empty block = valid answer. Missing block = not.

My premise wrong — bug misdiagnosed, or thing asked for won't work — say so
before writing code.

### Diagnose before patching
Something fail: state candidate causes and how to discriminate between them,
then test. No going straight to fix. No agreeing with cause I suggested unless
evidence support it.


**Applies to THIS repo's own API too** — large enough that memory unreliable.
FAILURES in ONE session (2026-07-30): `AerosolConfig(reference_aod=)` (really
`reference_aod_550`), `McFarlaneConfig(N_ref=)` (no such field),
`run_amip.build_parser` (really `build_arg_parser`), `from legoesm.grids import
create_grid` (really `legoesm.grids.factory`). Each cost full probe round-trip.
Worse, `RRTMGPConfig()` default `include_clouds=False`, so offline harness
omitting it return CLEAR-SKY fluxes and EVERY cloud gradient exactly 0.0 —
silently wrong number, not error. Read NamedTuple `_fields` / `_field_defaults`
before constructing config.

User's premise wrong — bug misdiagnosed, or thing asked for won't work — say so
BEFORE writing code.
## Verification (JAX-specific)

Code not done until it run. Correctness claims require output.

- **Shapes/dtypes**: check with `jax.eval_shape` before running anything
  expensive. Cheap, catch most errors.
- **Gradients**: any new `custom_vjp`/`custom_jvp`, adjoint, or hand-derived
  derivative must pass `jax.test_util.check_grads(f, args, order=2)` before you
  claim it work. Gradient that run is not gradient that correct — single most
  common way to ship silently wrong result here. (Applies directly to
  `_sendrecv_vjp` in `halo_exchange.py` and any new MPI-AD path.)
- **jit parity**: run function eager and under `jit`, compare outputs.
  Divergence = tracer bug (Python-side branching, `.item()`, `if` on traced
  value, host callbacks).
- **Sharding**: verify with `jax.debug.visualize_array_sharding` or by printing
  `.sharding`, not by reasoning about what annotation should do.
- **Numerics**: default float32. State tolerance you compare at. No `==` on
  floats. Test need float64 → say so explicitly, no silent `jax_enable_x64`.
- **donate_argnums / buffer donation**: never add without confirming donated
  buffer not reused. Fail silently or crash far from cause.

## Scope
One change at a time. No refactor of adjacent code, no rename, no "improve" of
code I didn't ask about. Long unbroken generations drift into invention — prefer
small verified diff over large plausible one.

**DO NOT EXTRAPOLATE. Do only what was asked** (user directive 2026-08-06).
The ask = deliverable, not starting point to reason outward from.

- Related-looking problem you notice = ONE-LINE report, not work item. Name it
  and stop; do not start it.
- No widening scope because fix "would only be complete if" something adjacent
  also done. Ship the ask; state the boundary.
- New artifacts (scripts, benches, plots, panels, public APIs, config knobs)
  only when asked or genuinely required to finish the ask. Unsure whether
  required → it not; ask in one line.
- No proposing or launching compute user did not ask for.
- FAILURES 2026-08-05/06: scoped 128-GPU coupled ladder nobody requested off a
  question about existing plots; added coupled panel to figure when asked to
  assess the figure; wrote probes and helper scripts for questions never posed.
  Each cost round-trip and buried actual answer.

## Attribution Gates — MANDATORY, each from a real 2026-07 failure
Model near operational. Every rule below mechanical: satisfy it or state
explicitly that you did not. "I was careful" is not compliance.

- **PROVE THE PATH EXECUTES before blaming a line.** Naming file:line as cause
  require showing that line run in configuration under test: print ENCLOSING
  FUNCTION (`awk` nearest `def` above it), confirm active lane/driver call it.
  FAILURE: blamed positivity clamps at `model_driver.py:10923` for century's
  water source; they live in `_run_per_step` while century run `_run_column`,
  which contain no moisture clamp at all. Fix nearly written for lane the run
  never touch. Same class as reading entry point instead of full path.
- **REUSING A REFERENCE IMPL MEANS PORTING ITS EXCLUSIONS, not just its
  formula.** State which of reference's guards/scope conditions you kept and
  which dropped, with reason for each. FAILURE: copied
  `spectral_les_moist.conserving_positive` but not its `n_water` split, so
  column-conserving borrow applied to number concentrations
  (`N_c`/`N_i`/`N_r`) — unphysical, and it fed M2005 deposition (~N_i^(2/3)),
  producing fake "accelerating dry bias" reported before being caught.
- **A TEST THAT INSPECTS SOURCE MUST NAME THE SYMBOL THAT RUNS, and must be
  shown to FAIL when feature removed.** `inspect.getsource(X)` assertion where X
  is delegating wrapper pass while proving nothing. FAILURE: asserted against
  `MPASPrimitiveEquationModel.step`; floors are in `_step_jit`.
- **TOOL STATUS IS NOT EVIDENCE — read output tail.** Exit code without tool's
  own success line (pytest's `N passed`, "COMPLETED in Xs") = UNVERIFIED; OOM
  kills and timeouts can surface as success. FAILURE: reported regression suite
  green on exit-0 that was actually `Out Of Memory` mid-run. Quote decisive line
  when claiming suite passed.
- **EVERY BASELINE/ALLOW-LIST REASON STRING IS A CLAIM — verify in code before
  writing it.** Plausible-sounding reason permanently hide real defect. FAILURE:
  classified `convective_buoyancy_death_memory` as "carried in SegmentCarry" (it
  not — leaf `BechtoldConfig.buoyancy_death_memory` exist and nothing map to
  it), asserted `SBMConfig.precip_efficiency` leaf that not exist, and credited
  `micro_substeps` to consumer that read `args.`, not the config field.
- **RATE / TENDENCY / SKILL COMPARISONS: identical windows on BOTH sides, and
  print window next to number.** Differing spans = confound, not result.
  FAILURE: TCW over days 190-530 vs CMOR year 1 gave "+38.7 kg/m2/yr"; matched
  windows gave +13.4. Extends existing controlled-comparison rule to derived
  rates.
- **A DIAGNOSTIC'S PRINTED PRECISION BOUNDS THE RATE YOU CAN CLAIM.** Log CWV at
  0.1 kg/m2 over 8 days resolve only ~±4.6 kg/m2/yr — no reporting trend inside
  one quantum. Prefer fp64 from model state (checkpoints) over parsed log lines.
  Same class as throughput-quantization error.
- **`JAX_ENABLE_X64=1` on any numerics/conservation test.** fp32 mismatch NOT a
  failure until re-run with x64; and *new* failure not yours until reproduced
  with your change stashed. Do both before reporting regression.
- **RUN-TARGET PARAMS ARE ABSOLUTE (`TARGET_DAYS`), and "latest checkpoint"
  MOVES.** For controlled pair, COPY pinned checkpoint into each arm dir; never
  use `PREV_CKPT_DIR`-style newest-wins pointer while another run advancing.
  FAILURE (twice): arms exited instantly at "Already at/past target".
- **A LAUNCHER FLAG THAT SWITCHES ONE FORCING CHANNEL MUST SWITCH ALL OF THEM.**
  Verify resolved paths in run log, not the flag you passed. FAILURE:
  `CENTURY_DECK=1` set era-correct ozone+volcanic but left 1979-2016 SST.
- **PROSE IS A POINTER, NEVER A CITABLE FACT.** Code comment, docstring,
  `AMIP.md`/`docs/` entry or "Known issue" naming a limitation, guard, or
  missing feature MUST be re-verified in CURRENT code at point of use before
  repeated as finding — this repo routinely fix things without updating prose.
  Comment name module imposing guard → OPEN THAT MODULE. FAILURES (2026-07-30,
  three in one day): quoted `_run_column` "turbulent surface fluxes are
  intentionally NOT applied" comment and `AMIP.md` Known #3 to claim MPAS has no
  turbulence — MPAS turbulence path exist (`turbulence/integration.py`, Perot
  edge->cell) and run resolve `turbulence=louis` + `surface_bulk_scheme=coare3`;
  and doubted a cloud_fraction comment that was exactly right.
- **THE FIRST GUARD YOU FIND IS NOT THE ONLY GUARD — follow value to its
  CONSUMER before declaring it unclamped/unchecked.** FAILURE: reported "no
  upper bound on r_eff" from `rrtmgp.py`'s `clip(x, 1e-6, None)`; real clamp to
  lookup-table range is one call deeper in `rrtmgp/optics/cloud_optics.py`. Same
  class as blaming a line without proving its enclosing function runs.
- **BEFORE ATTRIBUTING A BIAS TO A COMPONENT, PROVE THE COMPONENT EXISTS AND
  RUNS IN THE CONFIG UNDER TEST.** ABSENT component and BADLY-TUNED one give
  SAME symptom, and every tuning arm against absent component return null.
  Enumerate, from RESOLVED config and LANE'S code path (not deck's comments):
  the scheme, its PROGNOSTIC STATE, its inputs. No prognostic state -> say so
  out loud; that usually the finding.
  Corollaries, each earned:
  (a) **A `false` on a component switch does not mean "the simpler version" —
  it may mean NOTHING.** Check what lane actually fall back to.
  (b) **A DECK COMMENT NAMING A FALLBACK IS A POINTER, NOT A FACT** (prose rule,
  applied to configs): grep named symbol in that lane's setup.
  (c) **When N different schemes for component A all fail to move a bias, STOP
  TUNING A.** Limiter upstream, or A's inputs wrong.
  (d) **A parameter wrong at BOTH ENDS of a physical range is a MISSING
  DEPENDENCE, not a mistuning** — thing it should depend on not read.
  FAILURE 2026-08-15, most expensive of campaign: MPAS AMIP deck set
  `use_multilayer_land: false` with comment "this deck runs the slab". There is
  NO slab land on MPAS lane (`slab_land_active` wired only on cube/general
  path), so runs had **no land surface model at all** — land T_sfc was nearest
  OCEAN's prescribed SST minus 6.5 K/km, land evaporation fixed 0.6 of
  saturation, no soil, no water store, no runoff, no stomata, no surface energy
  balance. Tropical deserts evaporated 3.5x observed and rainforest 0.73x FROM
  SAME CONSTANT (corollary d, unnoticed); sensible heat ~25 W/m2 on Sahara and
  Amazon alike. Waves 9-11 spent GPU-hours swapping FIVE convection schemes and
  dozen trigger knobs against tropical-land rain deficit no convection scheme
  could ever fix (corollary c, unnoticed). Interactive soil moisture, bucket
  hydrology, stomatal control are DEFAULTS for AMIP case, not options.
- **A GLOBAL STATISTIC ON A NON-UNIFORM GRID NEEDS AREA WEIGHTS.** Never
  `np.mean(field)` for global mean on lat-lon (or any stretched grid) — use
  `cos(lat)` or model's `areacella`. FAILURE: reported "+17 hPa of dry mass
  created" from unweighted `p_s` mean; AREA-WEIGHTED mass was invariant at
  983.493 hPa to 6 digits, i.e. defect did not exist. Habits carried from
  quasi-uniform MPAS/SCVT mesh are INVALID on lat-lon.
- **A PROPOSED MECHANISM MUST SURVIVE A SCALING / PERTURBATION TEST BEFORE IT IS
  CITED AS THE CAUSE.** X claimed to drive Y → change X by known factor, check Y
  respond as mechanism predict. FAILURE: proposed "damp-to-rest pumps mass
  convergence" (predict ~linear in sponge coefficient); quartering coefficient
  slowed growth only 1.5x, refuting it — fix would have shipped on false
  mechanism. Label every uncaught claim PLAUSIBLE; honest "cause unknown" is
  cheap, confident wrong cause buy a code change and a relaunch.

## Implementation Discipline — the CODE and its PROSE are both claims
User, 2026-08-06: *"Be much more conscientious and careful when implementing.
Be systematic, check, do not be sloppy or too fast."* Every rule below from
defect shipped in ONE session that prompted it, and every one caught by
adversarial reviewer rather than by me — i.e. each avoidable by reading two more
lines before typing. Slow down at these exact points.

- **A DOCSTRING/COMMENT THAT DESCRIBES BEHAVIOUR IS A TESTABLE CLAIM. Trace data
  flow before writing it.** FAILURE: wrote "dropping this call now makes the
  test fail" about `divg_d` exchange — false, because unit 4 hands only
  `uc`/`vc` to `d_sw1` and `divg_d` not consumed until `d_sw5`. Sentence written
  from intent, not from call chain. Before asserting "X is covered by this
  test", name consumer of X and confirm it execute inside test's span.
- **NEVER WRITE "this removes the question entirely" ABOUT AN API YOU HAVE NOT
  READ.** FAILURE: claimed `np.ascontiguousarray` give unconditional copy; it is
  copy-IF-NEEDED, so it aliased at km=1 and copied at km>1 — one line of code
  with two aliasing behaviours, asserted as safe. Extends *Never infer an API*
  to STRENGTH of a guarantee, not just signature.
- **SCOPE WORDS — "both", "all", "every", "cannot", "unified", "always" — GET A
  GREP BEFORE THEY GET TYPED.** FAILURE: "both lanes now route through one
  helper" while `csw_step_sixface` still called interim helpers directly. Went
  into COMMIT MESSAGE, which cannot be edited after. Weaken to what you verified
  ("the post-p_grad_c site") or run the grep.
- **RE-READ A BOOLEAN IMPLICATION IN THE SOURCE BEFORE RESTATING IT; ONE-WAY IS
  NOT TWO-WAY.** FAILURE: `bounded_domain = regional .or. nested .or.
  duogrid` mean duogrid⇒bounded, and I wrote converse pair "unreachable" —
  inverting it and mislabelling legitimate regional/nested category as
  impossible. Quote the line next to restatement so direction checkable.
- **A CONTROL THAT PERTURBS A ZERO IS NOT A CONTROL. Print baseline value the
  perturbation multiplies BEFORE trusting result.** FAILURE: divg-corner probe
  scaled `divg_d(1,1)` by 3 where that cell is exactly 0.0, so control was no-op
  and proved nothing; claim actually rested on predicted==actual identity. Say
  which check carried the claim.
- **AN A/B MUST DIFFER IN ONE FIELD OF THE CONSTRUCTOR, AND YOU MUST DIFF THE
  CONSTRUCTOR ARGS TO KNOW.** FAILURE: compared two contexts differing in grid
  conventions AND in whether ext bundle existed, while asserting it isolated the
  exchange. Applies to fixtures, not just runs — controlled-comparison rule
  cover `pytest` fixtures too.
- **A NEW TEST/FIXTURE IS PART OF THE DIFF: RUN THE SUITE THAT CONSUMES IT, NOT
  ONLY THE TEST YOU WROTE.** Fixture edit = change to every test in the module.
- **SHELL COMMANDS THAT CAN PROMPT WILL SILENTLY NOT RUN.** Use `git checkout
  --`/`git restore`, `cp -f`, `rm -f`; never bare `cp`/`mv` for revert. FAILURE:
  bare `cp` hit interactive overwrite prompt and "revert" never applied — one
  `git status` short of reporting clean tree that was not clean. VERIFY EVERY
  REVERT with `git status --porcelain`.
- **A LOG-SCRAPING GUARD MUST EXCLUDE THE PROMPT/COMMAND IT ECHOES.** FAILURE:
  review wrapper grepped whole log for failure token its own prompt contained,
  so it "retried" every time and never printed exit status. Same class as test
  that cannot fail.
- **BUDGET THE REVIEW LOOP INTO THE WORK.** Adversarial reviewer found real
  defects in EACH of three rounds here; rounds 2 and 3 not ceremony. No
  presenting first-round implementation as finished, no treating "tests pass" as
  terminal condition — round-1 diff passed 401 tests while still containing
  silent fallback and false docstring.

## Assumption Gates — from six wrong assumptions in ONE session (2026-08-05/06)
User callout: *"you keep making a lot of assumptions that prove to be wrong."*
Gates above stop wrong CLAIMS about the model; these stop cheaper, more frequent
error — being wrong about CODE AND DATA IN FRONT OF YOU. Every rule below
mechanical, each cost full round-trip.

- **AN ARRAY'S LAYOUT IS AN API — READ IT, INCLUDING AXIS ORDER.** "Never infer
  an API" rule cover signatures; it also cover array SHAPE, AXIS ORDER, index
  base, padding convention. Before first index of any mesh/state field, print
  its `.shape`. FAILURE: indexed `mesh.cellsOnCell[c, k]` assuming
  `(nCells, 6)`; it is `(6, nCells)`, so probe raised `IndexError` on every
  method. `cellsOnEdge` is `(2,
  nEdges)` — one unambiguous pair per edge, usually the better handle.
- **A PROXY IS NOT THE QUANTITY. Real number produced by specific code path →
  GET IT BY CALLING THAT PATH.** Re-deriving lookalike from first principles
  silently answer different question. FAILURE: computed `max_degree` from 1-ring
  `cellsOnEdge` adjacency and reported it as "ppermute round count" — real
  exchange is halo-depth-aware, so proxy said 8 rounds for every method/rank
  count while real schedule said 12→14. Proxy could not even reproduce known
  answer — that the tell: **run proxy against a case whose real value you
  already know BEFORE using it on the unknown one.**
- **AN OPTIMIZER ONLY HELPS IF ITS OBJECTIVE IS THE BINDING TERM — state which
  quantity it minimizes, confirm that quantity is measured bottleneck.**
  FAILURE: assumed METIS would cut MPAS ppermute rounds; METIS minimize EDGE
  CUT, bottleneck is MAX_DEGREE, and measured they move OPPOSITELY (metis 19
  rounds @64 vs sfc 14, while metis has lower cut). "Better partitioner" is not
  a mechanism.
- **A TABLE YOU PARSED IS NOT DATA UNTIL YOU SPOT-CHECK IT.** Any derived
  summary quoted to human, or fed to review agent, need ≥2 rows verified by eye
  against raw source first. FAILURE: regex over multi-line series lists attached
  subdiv-9's `9.60/11.48 ms` to `s8` label, and that mislabelled pair went into
  codex prompt as fact.
- **PRE-IMPL GREP COVERS TESTS, NOT JUST FUNCTIONS.** Before writing test, grep
  for one already asserting same invariant. FAILURE: added two tests counting
  PCG reduction sites duplicating existing `test_reduction_count_halved`; codex
  had to point it out.
- **WHEN A TEST FAILS, DECIDE WHETHER THE EXPECTATION OR THE CODE IS WRONG, AND
  SAY WHICH.** Assertion you just wrote is claim with no more standing than the
  code. FAILURE: asserted `LEGOESM_..._FUSED_HALO=""` meant OFF; resolver is
  `!= "0"`, so `""` means ON — test was wrong, not the default.
- **BEFORE PROPOSING COMPUTE, CHECK THE PATH IS BUILT — grep driver for
  decomposition/sharding the run would need.** FAILURE: scoped 128-GPU coupled
  ladder before finding `CoupledESM` contain three occurrences of "shard", all
  in comments — coupled ocean is replicated, so ladder would have measured
  unbuilt path.
- **WHEN TWO MECHANISMS COULD EXPLAIN A NUMBER, NAME THE MEASUREMENT THAT
  DISCRIMINATES THEM AND RUN IT BEFORE REPORTING EITHER.** FAILURE: reported
  serial-vs-SPMD gap as "cadence PHASE disagreement"; predicates agree with ZERO
  offset and real cause was serial short-tail fallback. One `grep` of two
  predicates would have settled it.

## Compute Discipline — speculation costs GPU-hours, not just credibility
User, 2026-07-30 (THIRD callout in five days): *"You keep making very
speculative assumptions... be much more precise so we do not waste time with
useless simulations."* Gates above stop wrong CLAIMS; these stop wrong RUNS.
Simulation launched on hypothesis no measurement can refute = pure waste, and it
also cost WALL-CLOCK of queue slot it occupied.

- **NO COMPUTE ON AN UNFALSIFIABLE HYPOTHESIS. Before submitting ANY job costing
  >1 GPU-hour, write down three things: (a) exact number the run will produce,
  (b) value that CONFIRMS and value that REFUTES, (c) why a cheaper offline/CPU
  test on an EXISTING checkpoint cannot answer it. Cannot fill all three -> DO
  NOT SUBMIT; run cheap test first.** Nearly every question asked so far
  (fluxes, tendencies, radii, momentum budgets, cloud optics) was answerable
  offline from saved checkpoint in minutes. FAILURE 2026-07-30: submitted two
  5-YEAR full-physics runs (8 h walltime each) while model had KNOWN unfixed +56
  W/m2 albedo error and no low-level circulation — five simulated years of
  broken climate, answering no question that had been asked.
- **RANK ERRORS BY MAGNITUDE BEFORE CHOOSING WHAT TO WORK ON.** Run full
  scorecard FIRST, work LARGEST term; re-rank after every fix. FAILURE
  2026-07-30: spent most of session on hfls deficit (-40 W/m2) while dominant
  error was rsut (+56 W/m2) — and scorecard naming it was already sitting in run
  directory, unread.
- **ONE VARIABLE PER PRODUCTION RUN.** N simultaneous config changes answer ZERO
  questions, because no output attributable to any one of them. Multi-change
  config legitimate ONLY as deliberate new BASELINE labelled as such and never
  compared term-by-term against the old one. FAILURE 2026-07-30: one launch
  flipped ~8 switches at once.
- **A LONG RUN ON A MODEL WITH AN UNFIXED DOMINANT ERROR IS WASTE.** Before
  extending past ~30 simulated days, state largest outstanding scorecard term
  and why run still worth its GPU-hours. Fix big term, then extend. Short
  validation windows (days) are for "does it run and is new physics behaving";
  multi-year windows are for model that already pass.
- **NEVER LEAD WITH AN ARITHMETIC COINCIDENCE.** Hand-computed ratio that
  "matches" observed ratio is not evidence when calculation omit factors code
  actually applies. State it as arithmetic, or don't state it. FAILURE
  2026-07-30: "cover x tau ~ 1.9x matches the 1.9x albedo" ignored sub-grid
  inhomogeneity factor radiation applies to cloud paths.
- **PREFER THE INSTRUMENT THAT ALREADY EXISTS.** Before writing probe, check run
  directory for scorecard/manifest/diagnostic answering the question, and
  `scripts/validate/` for validator. Reading existing artifact cost seconds; new
  probe cost an hour and need its own controls.

## Diagnosis Discipline — one session, ~10 GPU arms, 5 of them wasted
2026-08-05/06, FESOM2-match OMIP blowup. Every rule mechanical, from specific
failure in that one session. Root cause turned out to be one-line IC defect
findable offline in seconds; arms bought nothing.

- **SANITY-CHECK EVERY PROBE NUMBER AGAINST A PHYSICAL RANGE BEFORE BUILDING ON
  IT — one line of arithmetic, before the conclusion.** Convert to quantity
  whose plausible span you know, compare. FAILURE: PGF probe returned 0.02
  m/s^2 and I built whole attribution on it; that value imply ~14 kg/m^3 density
  difference between ADJACENT 1-degree cells when entire ocean span ~6 kg/m^3.
  Check take seconds, number sat unused for hours, and running it immediately
  would have exposed real defect (cells initialised to T=0/S=0) before a single
  GPU arm.

- **A FAILURE STEP OR LOCATION IS NOT A SIGNATURE UNTIL THE DIAGNOSTIC IS SHOWN
  TO RESOLVE IT.** State instrument's resolution, confirm reported step is not
  merely its first sample. FAILURE: five one-variable arms all reported "BLOWUP
  at step 36" and I read that config-invariance as physics. `run_omip` set
  `block_size = max(1, diag_every)` and only test at block boundaries, so 36 was
  FIRST LOOK. At `--diag-every 1` answer was step 1. Corollary: arms run at
  coarse diagnostic stride prove only "this setting alone does not fix it",
  NEVER "this setting has no effect".

- **REFUTE WITH STATES, NOT EXIT CODES OR FAILURE STEPS.** Two runs failing at
  same step can fail for different reasons. FAILURE: declared open North Pole
  "refuted" because capping did not move blowup step; states showed capping HAD
  removed that mode (|u|max 19.27 -> 0.0137 m/s) and merely uncovered second,
  unrelated one at same step number.

- **A PROBE IS PRODUCTION CODE FOR THE "NEVER INFER AN API" RULE, AND ITS FIRST
  OUTPUT IS UNTRUSTED.** Read signature of every function a probe call,
  including this repo's own. FAILURE: called
  `build_runoff_map(lat, lon, mask, area)` when signature is
  `(ocean_mask, cell_area, lat_rad, lon_rad)`, so `cell_area` received
  `deg2rad(lon)` — exactly 0 at lon 0 — and I reported NaN "conservation defect"
  in shipped code that did not exist.

- **A PLAUSIBLE-LOOKING VALUE IS MORE DANGEROUS THAN A NaN, IN CODE AND IN
  TESTS.** Sentinel that is valid number of right dtype pass every finite/NaN
  guard downstream. FAILURE: `_interp_profile_to_z_coord` returned `0.0` for
  no-data column; `0.0` is not NaN, so caller's `where(isnan, fill, x)` never
  fired and 39032 wet cells entered model as fresh water at 0 degC. TWO
  COMMITTED TESTS ASSERTED THAT BEHAVIOUR, one calling it "documented degenerate
  fallback" and one commenting "still no NaN leaks" — exact inversion of truth.
  Degenerate branch must return something → return NaN (or raise) so downstream
  guard can see it, and treat any test asserting magic sentinel as suspect.

- **BUDGET CLOSURE BEFORE HYPOTHESES.** For any conservation-relevant blowup or
  drift, run closure/redistribution check FIRST — it name the operator class
  instead of ranking guesses. Canonical probe:
  `scripts/validate/ocean_fidelity/omip_conservation_closure.py`. On first
  deployment it refuted hypothesis I was about to spend a GPU arm on (dipole
  whose two-cell sum GREW is not a diffusion instability) and localised
  config-invariant source to single column. See
  `docs/ocean/fidelity/fidelity_to_fesom2jax_level_plan.md` (#1492) Phase 0.1.

- **A THROWAWAY PROBE'S NUMBER IS UNMEASURED. COMMIT THE PROBE.** Per #1492
  Phase 0.3: one committed probe per row, locked conventions, provenance (git
  SHA + inputs + flags) stamped on every run. Inline heredoc probes not citable,
  cannot be re-run against changed model, and hide own bugs — both defects above
  came from uncommitted ones, and committing closure probe immediately surfaced
  two more inside it.

- **DAMAGE-FIELD CORRELATIONS ARE NOT MECHANISMS.** In blown-up field, "worst
  cells are coastal / shallow / polar" describe where damage LANDED, usually
  downstream of single upstream defect. FAILURE: measured P(bad|coastal)=33% vs
  0.67% interior, real 50x enrichment, and treated it as coastal process; those
  were simply columns the IC defect had corrupted. Before citing spatial
  pattern, confirm snapshot is ORIGIN state (`snapshot_final.npz` stamp
  `_step = n_steps` regardless of when run stopped — verify with dedicated
  1-step run).

## JAX
- Pure pytree fns. `lax.scan` time integration. `vmap`/batched arrays over Python loops on array dims. `jnp.where`/`lax.cond`/`fori_loop`/`scan` not Python control flow on traced.
- **Feature gating exception** (`fix_mass`, `fix_moisture`): Python `if` on static bool in closure — NOT `jnp.where` (traces both branches). `jnp.where` only for data-dependent traced selection.
- Stable shapes. No retrace. Dtype: spectral=x64+complex128; finite-volume can float32.
- No host/device thrash, NumPy in traced code, hidden non-JAX side effects.
- **Buffer donation + `jax.grad`**: `donate_argnums` conflicts reverse-mode AD. JIT fn inside `jax.grad`/`eqx.filter_value_and_grad` → provide non-donating variant (`.raw`). See `build_segment_fn`.
- **Closures vs explicit args**: closure captures = compile-time consts. Per-iter changing val (SST, solar) → pass as traced arg. See `SegmentForcing`.

## Earth System
- Conservation, metric consistency, staggered-grid consistency, halo correctness = first-class.
- No silent clip/damp/coerce unless justified+validated.
- Preserve units, sign conventions, monotonicity/positivity, hydrostatic/nonhydrostatic.
- Cubed-sphere/curvilinear: assume edge+metric errors first.
- Physics coupling: column closure + consistent flux signs.
- DA/diff: preserve smoothness. No gratuitous nondiff.
- **Sign-convention check MANDATORY on every equation/flux/tendency edit.** Before declaring done on any code touching PDE term, flux, source/sink, BC, or budget update: (1) state coordinate convention in scope (z up/down, flux positive-up/down/into-body) as comment at the term; (2) walk EACH term, confirm its sign match that convention — gravity vs capillary/diffusion divergence, top vs bottom BC, source vs sink, the `±` in `state = state ± dt·tend`; (3) confirm budget closes (`in − out − Δstorage = 0`) and exchanged fluxes carry SAME sign at both ends of a coupling (land runoff `+into ocean` must arrive `+into ocean`). Comment label and code must agree — flux commented "upward" computed as downward = defect, fix label or math. Mechanical gate where feasible: sign/conservation unit test (analytic column, manufactured solution, or `assert` budget residual ≈ 0) — passing norms alone never certify sign right (flipped flux can still be small). Common flips: z-axis direction, evap positive-up vs moistening, brine/salt vs freshwater dilution, stress atmospheric vs ocean convention (`-tau`), free-drainage vs gravity double-count.

## Parallel/HPC
- Correctness across serial/multi-device/MPI/hybrid.
- Sharded: reason about halo exchange, reductions, partition specs, global invariants.
- Validate single-rank → smallest distributed.
- Backends differ (Metal/GPU/CPU/spectral/MPI). Apple Silicon: spectral on CPU.
- **MPI**: `initialize_distributed(global_n=N)` → `scatter_to_local()` → rank-local step → `gather_to_global()` for I/O only. Never full global per rank.
- **4D halo**: `pad_halo_4d()`+`pad_halo_vector_4d()` all vert in one msg. All 3D ops in `operators_3d.py` use 4D. Never `vmap(pad_halo)`.
- **MPI halo AD**: all `sendrecv` via `_sendrecv_vjp` (`@jax.custom_vjp` in `halo_exchange.py`). Only `allreduce(SUM)` AD-safe (via `global_sum_mpi`; `final_loss=True` only for a sum seeded as the loss); `MAX`/`MIN`/`allgather`/`bcast` = diagnostics only.
- **Device mesh under MPI**: per-rank count to `create_device_mesh()`, not total.

## Oracle-Recipe Fidelity (ocean) — see docs/ocean/fidelity/oracle_recipe_strategy.md
- ADDITIVE to Validation Rules: oracle work NEVER replaces unit tests, ocean matrix, conservation checks, or visual verification. Truth tiers (conservation/equivariance/analytic) outrank oracle-matching.
- Recipe = pure config selecting shared canonical blocks (never bespoke `veros_*` solver). Oracle-matching numerics go in canonical module (`eos.py`, advection/limiter dispatch, `vertical_mixing/`, integrator dispatch) as selectable options.
- Mimicry-only glue (halo strip, axis transpose, time-level handling) lives in fidelity harness, never model. Test: "would a user with different goal ever select this?" No → harness.
- Conventions handled only in bridge, verified by equivariance tests (`physics(φ(x))=φ(physics(x))` to tol); a "convention" changing wet domain/answers is physics → config, not bridge.
- Constants are config (`ConstantsConfig`), not module-global monkey-patches (no `override_constants` in shippable paths); defaults reference `legoesm.constants`; base only, derived (κ,ε) recomputed.
- Oracle tendency-match (tier 3) trusted only for block that also clear truth tiers (0–2). MACHINE-ENFORCED (#388 Ask#4): `ocean/fidelity/precedence.py::evaluate_precedence` LOCKS oracle tiers (≥3) on any truth-tier (0–2) failure; surfaced + exit-gated by `scripts/validate/ocean_fidelity/build_fidelity_scorecard.py` (the one generated scorecard).

## Recipe×Setup template adapters (#388)
- **One shared selector, never re-implemented.** Every component YAML experiment adapter exposing a `setup:` block (atmosphere `Config`, ocean `OceanExperimentConfig`, sea-ice `SeaIceExperimentConfig`, …) MUST route validation + command-building + signature through `legoesm.core.setup_selector` (`MatrixRunnerSpec` + `validate_setup`/`build_matrix_command`/`setup_signature`/`require_positive_finite`). Per-component CLI differences (case flag `--only`/`--test`, exact-match `exact_prefix`, which `--levels`/`--dt`/`--days`/`--resolution` flags exist) are a `MatrixRunnerSpec`, NOT a copy-pasted `parts=[...]` builder or private `_SETUP_KEYS`/`_validate_setup`. Component-specific case-name validation goes through `known_names=` kwarg. Re-implemented selector = REJECTED in review. Factory naming: `_<component>_matrix_spec()`. FOLLOW-UP: dedup `OceanExperimentConfig`'s inline selector onto shared helper once PR #465 + shared-helper PR both land on main (inline copy predates extraction).
- **Every setup template's `(name, grid)` MUST be a real matrix case**, enforced by catalog-backed test (`run_<comp>_test_matrix._build_test_matrix()` → assert pair exists) so `--test/--only =<name> --grid <grid>` never select nothing. New template without this gate → REJECTED.
- **An exact case selector matching nothing is a hard error** (`raise SystemExit`) in every matrix-runner mode — never silent no-op (dispatch-hardening). MPI-safe: key guard off global pre-slice match count; empty-slice ranks fall through to barrier.
- **New `experiment_registry` mode → same-PR `get_adapter` test** in `test_experiment_registry.test_get_adapter_resolves_classes` (template test importing class directly does NOT cover dispatch).
- Land has no standalone idealized-case surface (no registry/matrix runner) → recipe×setup N/A there; do not invent one to match pattern. Truth-tier precedence stays ocean-scoped per `oracle_recipe_strategy.md`; use `TRUTH_TIERS`/`ORACLE_TIER_FLOOR` by name (no hardcoded `3`).

## Validation
- Narrowest test after edits. Numerical changes: analytical/benchmark > unit tests alone. `JAX_ENABLE_X64=1` unless float32/Metal task.
- Dycore: Williamson, Galewsky, Jablonowski-Williamson, DCMIP, Held-Suarez, ocean benchmarks.
- Conservation/reductions/coupler: mass+energy diagnostics.
- Parallel: unsharded vs sharded, single-rank vs MPI.
- Too expensive: say what ran/didn't, residual risk.
- **CRITICAL — Controlled comparison: change ONE variable, hold eval protocol FIXED to baseline.** To claim a change (resolution, params, scheme, days) improved/degraded/"is comparable" vs prior result, keep EVERYTHING else byte-identical to that baseline: forcing data + its sampling (years/days/hours/climatology), evaluation grid, metric definition (bias vs RMSE), region masks, timestepping. Metric that moved because protocol/sampling changed = **CONFOUND, not result** — NEVER compare number computed on one sampling/grid/metric to number from another and call difference an effect. Resource limit (network cap, compute, time) force lighter or different sampling → **re-run BASELINE at that SAME sampling before comparing** — fresh baseline cheap insurance; confounded claim not. Before writing "improved"/"degraded"/"better"/"comparable" vs any earlier number, explicitly confirm two configs differ ONLY in variable under test; cannot → say so, claim NO direction. Report full config (data+sampling, grid, days/steps, params, metric) next to every number so reader know exactly what compared. Assuming two runs comparable when setup drifted = error that turn "we improved it" into "we degraded it."
- **CRITICAL — Precision gate for comparison/skill/causal claims (do NOT be sloppy — 2026-07-20 EC-site lesson).** (a) **Harness self-check FIRST**: before reporting NEW scheme's skill vs validated baseline, run KNOWN baseline through YOUR OWN harness, confirm it reproduce baseline's established/published number (±small tol). Your harness score validated scheme wildly off — e.g. two-leaf H looked "broken" (NSE≈−1) when paper figure track obs — the HARNESS is wrong; fix BEFORE any new-scheme claim. (b) **Match reference metric EXACTLY**: metric aggregation (daily-mean vs half-hourly point-wise NSE), window (multi-year JJA vs one summer), masks (valid-forcing) ALL part of protocol. Report SAME metric reference used, plus any alternative, WITH sample N. Single-window point-wise metric is NOT skill verdict, and never rank schemes by metric punishing one scheme's known artifact (point-wise scatter/spikes) while ignoring dimension of interest (mean diurnal shape). (c) **Never let QC mask flatter one side**: drop scheme's unphysical spikes from scoring → report BOTH failures-penalized primary score AND separately-labeled plausible-only sensitivity score. (d) **Do NOT INFER causal origin — INSTRUMENT it**: "the spikes come from X" require LOGGING X and alternatives (raw vs intermediate vs final), not deduction from reading one code path; "correctly hooked up" require reading FULL path, not entry point. (e) **Label every claim CONFIRMED (evidence shown) vs PLAUSIBLE (inferred)** — adversarial review WILL refute over-confident inferences; state uncertainty up front rather than presenting verdict table a five-minute check overturn.
- **CRITICAL — A gradient/AD-tuning result that underperforms a known-good baseline is a BUG in your method until PROVEN otherwise — never attribute it to physics.** Before interpreting ANY optimizer/AD tuning outcome, or attributing residual error to physical cause (radiation, resolution, scheme's structure): (1) **Differentiated loss MUST be exact quantity you evaluate and report.** Truncated-BPTT / short-window / stop-gradient'd-spinup / cached-state proxy differing from full evaluation = OBJECTIVE MISMATCH, not result — optimizer minimize proxy while eval barely move. Differentiate real objective (e.g. full RCE equilibrium the eval scores), or prove proxy tight. (2) **Verify AD actually flows:** finite-difference check on ≥1 parameter (analytic grad vs `(L(x+ε)−L(x−ε))/2ε`, ratio≈1), confirm gradients finite AND non-negligible (not silently zeroed by `stop_gradient`, frozen/`enable_*=False` leaf, `lax.stop_gradient`/`jax.lax.cond` dead branch, or detached recompute). (3) **Sanity vs known-good baseline FIRST:** derivative-free or prior method got large improvements and your gradient method barely move loss → YOUR optimization broken (weak objective, tiny effective LR, over-aggressive line-search backtracking to ~0 step, frozen leaves) — fix before drawing conclusions. Concluding "residual bias is radiative/structural, not tunable" from run whose optimizer never actually worked = exact error that ship FALSE scientific claim. Near operationalization this disqualifying: barely-moving loss, suspiciously-flat before/after, or result contradicting known-good baseline BLOCKS delivery until gradient path verified end-to-end. See [[feedback_verify_ad_before_physics]].
- **CRITICAL — Visual verify spatial/grid artifacts**: passing tests+norms NECESSARY ≠ SUFFICIENT for cubed-sphere ops, halo exchange, diffusion coeffs, grid metrics. Edge artifacts/cube imprint/grid-scale noise only detected visually (v-wind W2, wind_speed W5). Run `--only sw --grid cubed_sphere --quick` + inspect PNGs vs baseline. Norms can improve while artifacts worsen. Never claim "tests pass, edge fixed" from pytest alone.
- **Diffusion sensitivity**: div damping + hyperdiff AMPLIFY halo errors at cubed-sphere face boundaries. Check W2 v-wind visually when touching `_hyperdiff_cube`, `_div_damp_cube`, diffusion params.
- **Visual-regression gate (cube imprint)**: `scripts/validate/visual_regression.py --check` numericises W2 v-wind cube-imprint check (SSIM + per-panel perceptual hash + edge-artifact ratio vs tiny committed ref in `tests/visual_baselines/`). Deterministic metric math gated in CI (`tests/test_visual_regression_metrics.py`); full cube-SW `--check` runs as NIGHTLY non-blocking CI job until tolerances calibrated across CI hardware. Tiny numeric baselines (.npy+json) ARE tracked — the one carve-out to "no tracked visual baselines".
- **CRITICAL — VALIDATE THE INSTRUMENT BEFORE QUOTING ITS NUMBER (2026-07-25 duogrid lesson: 8 confident claims, all retracted).** Diagnostic script = UNTRUSTED CODE until it pass own controls. Never state finding — never write "measured", "confirmed", "proven", "VERDICT" — from probe's first output. **Before quoting any diagnostic number, run these five checks and say in the message that you ran them:**
  1. **Right conserved/invariant quantity?** Budget what SYSTEM conserves, not convenient proxy. (Failed: reported "vertex creates energy" from **KE alone** — KE NOT conserved in shallow water, it trade with PE. Total `E=∫area(½h|V|²+½gh²)` reversed sign of conclusion.)
  2. **Same transform / units / staggering on BOTH sides?** Two "A-grid winds" from different operators = DIFFERENT QUANTITIES. (Failed: ours `c2l_ord2` vs oracle `C2L_ORD=4` — SAME raw state gave 1.96e-2 vs 5.79e-2, 3× swing that WAS the reported effect. Also: never budget across stage boundary where state change representation — mid-step FV3 winds in circulation form, produced ±5.6e10 garbage. REPEAT OFFENCE 2026-08-05, this time SHIPPED then retracted: paired predicted face velocity (41.87 m/s) against model's `max_speed` timeseries diagnostic (21.95 m/s), which is CELL-CENTRE average of same field whose face maximum was 42.30 — apparent "1.2 % agreement" was artifact of comparing across staggering, and it went into merged PR before 1-step run exposed it. Name staggering AND reduction of both sides in sentence quoting them.)
  3. **Same time, resolution, config?** Index by MATCHED TIME, not frame number. (Failed: mapped day→frame as `round(day)-1` against HOURLY file, comparing our day-1 to their hour-1; and quoted **C12** wedge gain (~300×) as mechanism for **C48** instability, where it ~124×.)
  4. **Is the metric measuring what its name says?** Prove it on synthetic case with KNOWN answer before use. (Failed: called `mean|f−4-neighbour-mean|` a "2Δx grid-scale" measure — it is high-pass/curvature residual that merely sharper SMOOTH feature reproduce. Failed: "gain" probe that re-filled FIXED source, trivially 1.0000 by construction.)
  5. **Can the reduction support the claim?** `max` over tiles/corners/components taken independently per run can peak at DIFFERENT physical locations; max-of-per-tile-means is not global mean. Keep argmax metadata, map to common physical location before claiming "localized".
  Plus: **diff ARRAYS, never printed summaries** (claimed "bit-identical ⇒ deterministic, not chaos"; arrays actually differed by 9e-6 — only rounded printout matched). **Never let probe print own verdict** ("=> the growth is REAL") — interpretation belong in analysis after controls pass, not baked into tool where it get echoed back as evidence. **`nanmax`/`nanmean`/`nansum` hide failures** — make NaN and missing frames FATAL. **Record every effective flag, env var and git SHA in each artifact**; default `--n 36` silently mis-slicing C48 file run fine and lie.
- **CRITICAL — LABEL EVERY CLAIM, AND PREFER RETRACTING EARLY.** Tag each statement **CONFIRMED** (control-passed evidence shown, instrument validated) vs **PLAUSIBLE** (inferred / single-run / uncontrolled). Chain of PLAUSIBLE steps is not CONFIRMED. Later measurement contradict earlier claim → **retract loudly and immediately in same message and in memory file** — no quiet move on, because stale confident claims get built on. Run codex adversarial review on DIAGNOSTIC TOOLING, not just model code: instruments decide what you believe, so bug there manufacture confident wrong physics conclusion. Cheap self-check before any big claim: *"what measurement would make this false, and did I run it?"* Answer no → claim PLAUSIBLE at best.

## Domain Architect vs Syntax Engine (AI guardrails)
See `docs/architecture/ai_guardrails/domain_architect_vs_syntax_engine.md`. Doctrine: human dictate *logic* (units, signs, conserved qty, valid scheme sets, references, acceptance criteria); AI fill *body*; every declared invariant checked **mechanically** so violations fail LOUDLY. Each gate = **tripwire, not proof**, ships synthetic-violation self-test (provably non-vacuous). NON-NEGOTIABLE harness (extend, never weaken; budgets/TODOs shrink only):
- **Spec-first physics contracts**: every physics scheme module declares `__physics_contract__` (units/signs/conserves/differentiable/reference/idealized_test); `tests/test_physics_contracts.py` partitions all `*/physics/*.py` into EXCLUDED / CONTRACT_TODO(shrink-only) / annotated — NEW physics file must ship contract or be classified. Author contract + acceptance test BEFORE body.
- **Atmospheric parameterization edits are high-risk**: run codex adversarial review, #477 truth-tier tendency validators, conservation tests, equilibrium-SCM-RCE realism harness before merge. NEW scheme must pass RCE realism gate or enter its shrink-only TODO partition with a reason.
- **CI ratchets** (AST + self-test, via shared `tests/_ratchet_audit.py`): `test_no_hardcoded_constants` (constants only from `legoesm.constants`), `test_no_saturation_reimpl` (saturation only from `thermo`), `test_dispatch_hardening` (no scheme guard silently deleted), `test_validate_strict_coverage` (no scheme field skips fail-early validation). Escape: real `# const-ok:`/`# satcurve-ok:` comment.
- **Local hooks** (`.claude/hooks/`, wired in `settings.json`): PreToolUse blocks edit adding banned constant/saturation prefactor to `.py` (fail-open); Stop reminds (once/session) to run mandatory codex review on uncommitted numerics/physics. CI remain authoritative (`Bash` heredoc bypass hook, not CI).
- **Best coding practices = use existing system**, not parallel one: ruff/mypy, import-linter (`alerting="error"`), inline-import budgets, pre-impl grep + shared utilities (no re-derivation), every new `.py` gets unit test, slopbuster sweeps, iterate-with-codex on substantial changes.

## Commands
- Install: `pip install -e ".[dev]"`
- Tests: `.venv/bin/python -m pytest tests/`
- Sci tests: `JAX_ENABLE_X64=1 .venv/bin/python -m pytest <target>`
- Atm matrix: `JAX_ENABLE_X64=1 .venv/bin/python scripts/matrix/run_atmosphere_test_matrix.py`
- Ocean matrix: `JAX_ENABLE_X64=1 .venv/bin/python scripts/matrix/run_ocean_test_matrix.py`
- AMIP: `.venv/bin/python scripts/run/run_amip.py`
- Dycore progression: `JAX_ENABLE_X64=1 .venv/bin/python scripts/matrix/run_atmosphere_test_matrix.py` (SW -> hydrostatic -> non-hydrostatic ladder; `--only sw|hydro|nh`). Old `tests/validation/run_dycore_progression_suite.py` superseded — its per-case child scripts removed.
- GPU/MPI scaling: `.venv/bin/python scripts/bench/run_levante_gpu_scaling.py --grid cubed-sphere --mode strong` (`docs/performance/REAL_HARDWARE_SCALING.md`)
- Scripts reorganized into buckets: `scripts/{run,matrix,bench,plot,validate,data,experiment,cluster}/`; debug in `scripts/tmp/`. See `scripts/README.md` + `## File Layout` below.
- MPI tests: `mpirun -np 2 .venv/bin/python -m pytest tests/distributed/`
- MPI diff: `mpirun -np 2 .venv/bin/python -m pytest tests/distributed/test_mpi_differentiability.py`

## File Layout (audit — enforce on EVERY new file; no random files)
- **New scripts go in correct `scripts/` bucket — NEVER `scripts/` root or repo root.** Buckets: `run/` (prod drivers), `matrix/` (test-matrix registries), `bench/` (perf/profiling/scaling), `plot/` (plot/replot/regen), `validate/` (non-matrix validators/verifiers/conservation checks), `data/` (download/build/prepare forcing+IC), `experiment/` (init/reproduce/templates/machine-detect/fetch), `cluster/` (SLURM `.sbatch` job wrappers, e.g. `cluster/omip_nemo/`). Pick bucket by what script DOES. New bucket need real category, not dumping ground. See `scripts/README.md`.
- **Debug / throwaway / one-off → `scripts/tmp/` ONLY** (eventually deleted): `_*`-prefixed probes, `diag_*`/`diagnose_*`, per-iteration scratch. Never at `scripts/` root. `_probe_*.py` gitignored.
- **No new files dumped at repo root.** Root keeps ONLY: `README.md`, `CLAUDE.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, `FEDERATION.md`, `project_status.md` (generated), `pyproject.toml`/lockfile/dotfiles. Everything else has a home.
- **No `.md` notes accumulating at repo root → `docs/`.** Dev-notes, change logs, faithfulness/audit trackers (`*_faithful.md`, `*_checks.md`, review logs) live under `docs/`. Reference by BASENAME so code comments survive the move.
- **No runtime outputs in git.** `diagnostics/`, `logs/`, `**/logs/`, `results/`, `checkpoints/`, `output/`, `*.zarr`/`*.nc`, root `*.png`/`*.pdf`/`*.svg` gitignored. Visual-regression baselines stay in LOCAL working copies, regenerated on demand — not tracked. Never `git add -f` a runtime artifact. EXCEPTION: `.physics-validator/` is tracked on purpose — review logs, probes and banked oracle results cited as evidence by docs, tests and validation scripts (user 2026-10-24); only its error captures are ignored.
- **Source stays under `packages/<pkg>/legoesm/`** (federation namespace). New subpackage → update `tests/test_federation_plan.py` same PR.
- **Do NOT "flatten" `packages/<pkg>/legoesm/<subpkg>/` → `packages/<pkg>/` — double name is LOAD-BEARING, not redundant.** Editable installs (dev/test workflow) require on-disk path == import path, and hatchling REFUSES prefix-*adding* `sources` remap in dev mode (`ValueError: Dev mode installations are unsupported when any path rewrite in the sources option changes a prefix rather than removes it`; editables#20). So `packages/atmosphere/legoesm/atmosphere/` is only layout where `import legoesm.atmosphere` resolve under `pip install -e`. Flattening to keep `legoesm.*` build wheel but break editable dev; flattening to bare `import atmosphere` mean rewriting ~23k import sites + dropping namespace. Verified 2026-07-13 — don't re-attempt. `src/legoesm/` is SANCTIONED root meta-package (the `legoesm.*` TOP-LEVEL modules: `constants`, `config`, `cli`, `registry`, `dycore_factory`, …), same load-bearing reason — NOT stray `src/` to remove. "Never add source at `src/`" mean don't create NEW `src/` trees for component code; existing `src/legoesm/` stays.
- **Tests mirror package tree under `tests/`** (`tests/<component>/<tier>/...`). Curated dycore regressions in `tests/atmosphere/dycore/regression/`. No new `test_*.py` at repo root.
- Staging: explicit pathspecs, NEVER `git add .`/`-A` — catch stray scratch + concurrent-session files.

## Bug Triage
- Instability: CFL, boundary, metric, halo, pressure-gradient, diffusion, dtype.
- Conservation drift: flux form, weights, reductions, state updates before fixers.
- Differentiability: control flow, shape changes, side effects, checkpointing, nondiff branches.
- Perf regression: retrace, host callbacks, scatters, sharding, Python loops.
- Cross-backend: dtype, x64, unsupported kernels, comm semantics.

## Hygiene/Imports/Tests (audit)
- Every new `.py` ≥1 direct unit test importing+exercising leaf module (tendencies for physics schemes — not just integration via factory). No `__init__.py` re-export / `supported_matrix.py` / factory dispatch add without same-PR test. Block PRs growing untested-LIVE count.
- New config dispatch (Literal + factory in `integration.py`): test via public config.
- **New user-tunable config field → same-PR CLI flag in every affected run script.** Rule applies to `ExperimentConfig`/`DycoreConfig`/`OutputConfig` fields in `run_amip.py`, `OceanExperimentConfig`/`KPPConfig`/`VerticalMixingConfig` in `run_omip.py`, `MultiLayerLandConfig` sub-configs in `run_lmip.py`. "User-tunable" = any float/int/str/bool field user would reasonably change for sensitivity run (excludes: internal-only flags such as `debug_precision`/`diagnostics_perf_mode`, unimplemented stubs like `carbon_cycle` in AMIP, parallel-milestone-gated fields like `distributed_mode=spmd`). Each new flag also needs: (1) wiring in `build_config_from_args` / equivalent config-builder, (2) round-trip test in corresponding `tests/unit/test_run_*_cli.py`, (3) `validate_strict` membership check if it is a scheme Literal. FLOAT tunables with `__param_spec__` need no flag: all four MIP drivers take `--config` (dests, choices-validated) + `--params` (registry-qualified `scheme_key.field`, bounds-validated; atm via flattened-scalar map). Reachability machine-audited by `tests/unit/test_params_reachability_audit.py` against shrink-only `_params_reachability_baseline.py` (new spec'd tunable unreachable in its component's drivers go red) — the #691 "CLI flag gaps" debt closure.
- Removing module: also remove `__init__.py` re-export, `supported_matrix.py` entry, dispatch, test file, `__pycache__`.
- No deprecated backward-compat wrappers — update call sites. No thin dispatch-only wrappers (`X_utils.py` re-exporting `X.py`) — inline/factor. Real branching across callers (`land/stomata_utils.py`) legit. Grid variants legit when genuinely different numerics; indexing-only copy-paste forbidden.
- **No top-level cross-package imports from `core/` to `runtime/`/`parallel/`/`driver/`/`training/`/`experiments/`.** Why: `from legoesm.runtime.backend import ...` at top of `core/precision.py` triggered `runtime/__init__.py` → `runtime.precision` → `core.precision` mid-init, breaking isolated pytest. Use function-scope deferred imports.
- **No import of private (`_`-prefixed) symbols across modules.** Promote (drop underscore + `__init__.py` re-export) or factor public wrapper. Mutable singletons (`grids.halo._halo_backend`/`_mpi_topology`, `cubesphere_exchange._spmd_mesh`): use accessors (`get_halo_backend`/`get_mpi_topology`/`get_spmd_mesh`), never import global. Audit: `grep -rE "from legoesm\.[^ ]+ import [^,]*\b_[a-z]" src/legoesm/ packages/ | grep -v " as _"` = 0 (line-grep miss multi-line/function-scope/`_UPPER` imports); CI ratchet `tests/test_no_private_cross_imports.py` (AST, incl. function-scope; dunder-exempt; shrink-only allowlist EMPTY since 2026-06-10 — keep it empty).
- **Test-only modules MUST be acknowledged.** Not wired into factory/`__init__.py`/prod driver: (a) wire same PR, (b) move to `_future/` + docstring + xfail/skip, or (c) delete.
- **Never commit `docs/references/`.** Local research PDFs/extracts. Cite by filename/DOI. Notes elsewhere (`docs/ocean/experiments/`). Staging: explicit paths, never `git add .`/`-A`.
- Slopbuster periodic: `/slopbuster audit all` or `/slopbuster review`.
- High-priority untested LIVE (touch any → add test same PR): `ocean/experiments/global_overturning.py`, `ocean/physics/{bottom_drag,convection,surface_forcing,vertical_mixing}/output.py`, `ocean/physics/convection/enhanced_diffusion.py`, `ocean/physics/vertical_mixing/{k_profiles,mpas_integration}.py`. RESOLVED 2026-05-29 (direct tests added): `coupler/surface_energy.py`, `ocean/dynamics/barotropic_common.py`, `atmosphere/dynamics/sfno_pe.py`, `atmosphere/dynamics/tracer_transport_mpas.py`; `atmosphere/physics/convection/_triggers.py` + `timestepping/tridiagonal.py` already covered.

## Constants/Params (audit)
- **All physical constants in `src/legoesm/constants.py`.** New constant (T, ρ, c, L, k, μ, EOS coeff, Schmidt#, R_earth) MUST be added BEFORE use. No constants in `config.py`/fn bodies/test fixtures/plotters/notebooks even with `# = constants.X` comment.
- **`getattr(..., "X", <literal>)` fallbacks count as hardcoded.** Use `getattr(grid, "radius", constants.R_earth)`. Same `setattr`/`dict.get`/`kwargs.get`.
- **No hardcoded physical constants in fn sigs/bodies in `src/legoesm/`.** No `def f(g=9.80616, ...)` → `g: float = constants.g` or config NamedTuple. NamedTuple defaults SHOULD reference constants. Tunable scheme params (sigmoid sharpness, τ, drag) stay in config NamedTuple.
- **No hardcoded tunable params in physics bodies.** Sigmoid sharpness, τ, Louis coeffs, KPP epsilon, emissivity, drag → scheme `*Config` NamedTuple. Exempt: safety floors (`eps=1e-30`), math constants (`0.5`, `2.0`), category lookup tables (PFT `L_v` in `surface_params.py`) with doc.
- **No `273.15` for C↔K in prod.** Use `constants.T_freeze`. `T_freeze_ocean=271.35 K` in `ocean/eos.py` = only intentional exception.
- **Tests+scripts+plotters same rule.** `from legoesm import constants`. No `9.80616`, `7.292e-5`, `6.371e6` literals.
- **Sigmoid sharpness/transition widths in JAX hot loops forbidden as magic numbers.** Inside `scan_step`/`cond`: fn kwarg with doc default OR scheme `*Config` field. Ex: `compute_moist_adiabat(lcl_sigmoid_width_pa=100.0)`, `PlumeConfig.active_sigmoid_sharpness=1e4`, DM95 `transition_width_frac=0.1` on `dm95_taper`/`dm95_taper_scalar`/`_triad_taper`.
- **Saturation re-impl in forcing modules forbidden.** Use `legoesm.thermo.saturation_mixing_ratio`/`saturation_vapor_pressure`.

### Parameter hygiene — declare per category at file top + machine-readable tunable/fixed split (gated)
Two CI tripwires enforce this (extend, never weaken; baselines shrink-only): `tests/test_no_inline_physics_coeffs.py` (+`_inline_coeff_baseline.py`) and `tests/test_param_specs.py` (+`_param_spec_baseline.py`).
- **No inline empirical coefficient in physics function body OR signature default.** Across `*/physics/*`, `land/`, `ice/`, `coupler/`: every float literal (and int `|v|>16`) inside function scope flagged. Move to (a) scheme `*Config` NamedTuple field if tunable/scheme-defining, (b) module-level `_UPPER_SNAKE` constant/table block (published fits/tables: Sutherland, Hall-Pruppacher, Morel-Berthon, Jerlov, KK2000) with provenance comment if fixed published constant, or (c) real `# coeff-ok: <reason>` (reason REQUIRED) for genuine numerics one-off. Exempt: math {0,0.25,0.5,1,2,3,4,6}, exact conversions, `|v|<=1e-6` floors, `|v|>=1e20` guards, subscript indices, Pow exponents, small ints (indices/counts).
- **Per-file layout (consistent placement):** docstring → imports → `__physics_contract__` → `__param_spec__` → fixed `_UPPER_SNAKE` constant/table blocks (under `# --- <category> (<reference>) ---` comments) → `*Config` NamedTuple(s) with fields grouped by same category comments (snake_case + unit suffixes) → functions whose bodies read ONLY `cfg.<field>` / `constants.*` / exempt math. Kwarg-default literals → reference module constant or config field (Name in signature, not literal).
- **Every physics scheme `*Config` declares `__param_spec__`** (module-level pure dict literal next to NamedTuple). Per float field: `units`, `bounds (lo,hi)`, `tunable_tier`, `transform` (sigmoid/softplus/none), `category`, `reference`, `shape` (None or dim key like `n_pft` for variable-size array params), optional `legacy_name`. NEW config module must ship spec or be classified in `PARAM_SPEC_TODO` (shrink-only). Inclusion computed: only `:float`-annotated fields spec-eligible.
- **Tunable/fixed split (the continuum), classify SAME way every time:**
  - `tunable_tier 1` (**core**) = params already trained in practice / well-posed (surface albedos, ice strength `P_star`, gray optical depths, bulk exchange `C_H`/`C_E`).
  - `tunable_tier 2` (**extended**) = clear closure knobs with literature bounds (relaxation timescales τ, entrainment/drag/autoconv rate coefficients, thresholds, emissivity, roughness).
  - `tunable_tier 0` / `excluded` (with reason string) = NOT trainable: numerics floors/caps/regularisers, smoothing widths, measurement conventions (e.g. MOST 10 m), and anything **iteration-coupled** (mEVP `alpha`/`beta`, EVP `T_evp` couple to subcycle count).
  - Collector select tiers `1..N` (`build_trainable_params(config, tier="core"/"extended"/"aggressive", include=, exclude=)`); flip a param's status with 1-line `tunable_tier` edit. See [[param-hygiene-spec-effort]].
- **Loop-iteration COUNTS are never config/trainable** → module constant (e.g. `_N_EVP_DEFAULT = 120`), not config field, not kwarg-default literal. Structurally guaranteed: ints not spec-eligible, so iteration count can never reach trainable collector. See [[loop-counts-never-trainable]].
- **A tunable closure whose default is a `constants.X` reference** (e.g. `S_ice_new = constants.S_ice_bulk_default`) is *eligible* (may be `__param_spec__` param with explicit bounds + tier) though not *required* (AST gate won't force it). Expose genuine calibratable closures; keep environmental references (ocean salinity) fixed/excluded.
- **Trained values inject via config pytree, not new signatures:** `params.to_overrides()` → `legoesm.core.param_overrides.apply_param_overrides(physics_config, overrides)` (`NamedTuple._replace`) INSIDE the loss so leaves TRACED (SegmentForcing doctrine); production keeps static Python-float leaves (constant-folded, no retrace). Register newly-specced module in `param_collector.SPEC_MODULES` (drift-tested).

## Naming
- Surface T = `T_sfc` everywhere. No new `T_surface`/`Ts`.
- Driver/config schema field names match runtime field. New tunable: same name in `driver/config.py`, scheme config NamedTuple, YAML schema (consistent `hyperdiff_coeff`).
- **Same name + different units = bug magnet (audit).** Unit hint (`_C`, `_K`, `_s`, `_days`, `_m`, `_km`): use everywhere. New C-vs-K args MUST carry `_C`/`_K`. Config fields sharing base (`tau_*`, `T_*`, `c_*`, `C_*`) MUST have consistent unit suffixes OR distinct names.
- **snake_case all NamedTuple fields**, even capitalized symbols (CAPE, CIN, MSE, TKE). `SBMConfig.CAPE_threshold` → `cape_threshold`.
- **Renaming a tunable `__param_spec__` field = rename field AND its spec key together + add `legacy_name: <old>`.** Spec key must equal float field name (`test_param_specs`), so half-rename go red; `legacy_name` document prior name for any externally-saved tuned JSON. (2026-06-22: `CarbonConfig.T_opt→T_opt_C`/`T_width→T_width_C`, `StomataConfig.T_opt_jarvis→T_opt_jarvis_C`/`T_range_jarvis→T_range_jarvis_C` — Celsius fields gaining `_C` suffix.)
- **`tau_`/`τ` prefix valid for wind/wave STRESS [Pa], not timescales only.** Ocean wind stress and GWD launch stress (`tau_0`, `tau_max`) = standard GFD τ notation — do NOT auto-rename to `stress_*`. Real anti-pattern = MIXED units under one `tau_*` family inside single config without unit suffixes; fix by suffixing timescale members (`_s`), not stresses.

### Open naming debt
- `T_sfc`(368)/`T_surface`(59)/`Ts`(~6): coupler+`land/{multilayer_land,snow_budget,stomata_utils,slab_land}.py`, `ice/sea_ice.py`, `coupler/{accumulator,lake/two_layer_lake}.py` still `T_surface`; 3 files MIX BOTH — `driver/coupled_esm_driver.py`, `ice/sea_ice.py`, `driver/earth_system_driver.py`. Unify cleanup PR.
- `nlev`(4119)/`n_levels`(199)/`nz`(33): `nlev` dominates. Cleanup PR.
- `tau_relax`: RESOLVED 2026-05-29 → `KuoConfig.tau_relax_s`[s], `PhillipsTwoLayerConfig.tau_relax_days`[days] (matches `backscatter.tau_relax_days`). Keep unit suffix on any new relaxation-timescale field.
- `C_water`/`c_water`: RESOLVED 2026-05-29 → `SoilThermalConfig.C_water_vol`[J/m³/K], `LakeConfig.c_water_mass`[J/kg/K]. Keep `_vol`/`_mass` on new heat-capacity fields.
- `n_layers` overloaded: soil=`n_soil_layers`, ML=`n_hidden_layers`, reserve `n_layers` for atm/ocean vert.

## Dispatch (audit)
- **Every `scheme="..."` factory MUST `raise ValueError` on unknown.** Silent `else: <default>` mask typos+dead branches. Historical: `cloud_fraction.compute_cloud_properties` ran sundqvist on typo; `land/carbon/carbon_cycle.py:443` zero CO2; `ocean/biogeochemistry/carbon_cycle.py:108,209` silently disabled BGC; MPAS PV typos → enstrophy in `{compressible_euler_mpas,primitive_eq_mpas,shallow_water_mpas,ocean_pe_mpas}.py`; bulk-scheme typos → constant in `coupler.py:204`, `slab_land.py:156`, `multilayer_land.py:212`, `two_layer_lake.py:66`, `bulk_formulas.py:68`; `io/restart.py:232` silently wrote npz. HARDENED 2026-05-29 (now `raise ValueError`, validated at fn entry on static config): `carbon_cycle.py:step_carbon`, `coupler.py:ocean_tile_response`, `ice/sea_ice.py:_bulk_flux_dispatch`. HARDENED later — all now raise: `slab_land.py`, `multilayer_land.py`, `coupler/lake/two_layer_lake.py`, `bulk_formulas.py` (via `core/bulk_flux.py::validate_bulk_scheme`). HARDENED 2026-06-22 (atm MPAS `pv_scheme` typo→energy + thompson `snow_scheme` typo→bulk): `compressible_euler_mpas.py`, `primitive_eq_mpas.py`, `shallow_water_mpas.py` (now `energy`/`enstrophy` else-`raise`), `microphysics/thompson.py` (fn-entry `snow_scheme` guard). These are nested scheme-Config fields, NOT `ExperimentConfig` literals → factory/fn-entry raise is the defense, not `validate_strict`.
- Dispatch in `lax.fori_loop`/`lax.cond` (`coupler/bulk_flux.py:222`): validate at fn entry on static Python val, not traced body.
- **Nested scheme-Config dispatch raises too.** A `scheme`/`pv_scheme`/`snow_scheme`-style field in leaf `*Config` NamedTuple (NOT `ExperimentConfig` literal, so `validate_strict` never see it) MUST still `raise ValueError` on unknown — validated at fn/factory entry on static config value (matches `ocean_pe_mpas` sibling). Bare `else: <default>` here silently run different physics on typo. Lock each new guard in `tests/test_dispatch_hardening.py::BASELINE_DISPATCHERS` (grow-only) so it can't be silently deleted.
- Add membership-set assertions in `ExperimentConfig.validate_strict` for new scheme literals. RESOLVED 2026-06-09: `convection`/`turbulence`/`gravity_wave_drag` now HAVE validate_strict membership checks (`config.py:430-455`). Guarded + enforced going forward by `tests/test_validate_strict_coverage.py` (every scheme-like config field must be membership-validated or in its `KNOWN_UNVALIDATED` loader-validated allow-list; removing guard or adding unvalidated scheme field → red). Companion factory-level guard: `tests/test_dispatch_hardening.py` (existing unknown-scheme `raise` may not be silently deleted; 76-entry grow-only baseline).

## Common Mistakes
**NamedTuple fields**: verify actual field. `PhysicsOutput.precip` not `precipitation`. `hasattr` guard silently degrade. Adding field to `SegmentCarry`: update every call site. `grep -rn "SegmentCarry(" --include="*.py"`.

**Land/face masks (latlon C-grid)**:
- Never `state._replace(land_mask=...)` on `LatLonCGridOceanState` without updating `u_mask`+`v_mask`. Stale face masks → mass flux through walls → silent leak.
- Preferred: `land_mask_override` to `rest_state_latlon_cgrid_ocean()` at construction.
- Post-construction: `replace_land_mask(state, new_mask)` from `init_latlon_cgrid.py` — atomic update 3 masks.
- `_assert_runtime_invariants` (gated `enable_runtime_checks`) catches inconsistencies.

**JIT/compilation**:
- Never build closures in training loops. `build_segment_fn` create new fn per call → inside `for epoch`/`_loss_fn` recompile each iter. Build once outside; pass changing vals as args.
- Helper fns inside `lax.scan` body: Python defs in `_single_step` recreated each trace. Move to module scope.
- Dead code from iteration: when refactoring, grep for vars assigned never used.

**SegmentCarry**: canonical hot-loop state. Adding field cross-cutting: NamedTuple def, `pack_carry`, `unpack_carry` docstring, per-step Python ref loop in `test_compiled_segments.py`, `test_scale_tpu_compat.py`, `test_scale_jit_health.py`, direct `SegmentCarry(...)` in validation tests. New diagnostic fields (`max_cfl`) reset to zero at segment start, not accumulated.

## Assets
Specialized agents in `.claude/agents/` for dycore, validation, differentiability, physics, land/ice, scalability.

## Response Style

### RULE -3 — BREVITY IS THE DEFAULT, LENGTH IS OPT-IN (2026-08-18, FIFTH callout)
User: *"be succinct and clear - make this a default."* Rules -2, -1 and 0 all
framed brevity as a CAP to check before sending, and I wrote to the cap every
time. Inverted:

**DEFAULT REPLY = <=3 LINES. NO HEADER. NO TABLE. NO BULLETS.**

Length is earned, not assumed. It is unlocked ONLY by an explicit ask
("report", "walkthrough", "long version", "what is left") or by numbers the
user will genuinely compare (a table of >=3 rows AND >=2 columns). Bold
headers, per-item verdicts, before/after lists and caveat lines are
REQUEST-ONLY formats.

Mechanical check: write the reply, delete everything after line 3, and send it
if it still answers. If it does not, the content wanted a table — keep the
table and delete the prose.

Background-task notifications get NO reply at all unless a result is ready, a
claim was refuted, or a decision is needed.

### RULE -2 — MECHANICAL LENGTH LIMITS (2026-08-14, THIRD callout)
User: *"be more succinct and clearer - make this a rule."* RULE -1 and RULE 0
below already said this and I still shipped multi-section replies with tables
and bold headers on routine results. Prose rules did not work, so these are
COUNTABLE. Check them before sending.

- **5 lines of prose, hard.** Not 5 sentences — 5 rendered lines.
- **ONE bold header per reply, or none.** Multiple `**Headers**` = a report,
  and reports are only for when one was asked for.
- **A table needs >=3 rows AND >=2 columns of real data.** Two numbers go in a
  sentence. A table of one comparison is decoration.
- **No line explaining method, discipline, or what a control proved.** The
  result only. "Pre-registered X, got Y" is one clause, not a paragraph.
- **No restating a caveat already in the commit.** The commit is the record.
- Status/job IDs: one line total, at the end, no formatting.

If the content genuinely needs more, say "long version?" and stop.

### RULE -1 — THE CAP APPLIES TO GOOD NEWS AND BAD NEWS ALIKE (2026-08-12)
User, again, after a session of correct-but-long replies: *"be succinct and
clear."* The rule below was being followed for status and ignored for
findings — a retraction, a root cause, or a self-caught instrument bug is
NOT a licence to write six paragraphs. Length is not proof of rigor.

- **A finding is ONE line: the corrected fact.** Not the discovery story,
  not what it means for three other workstreams, not a list of what still
  stands. "X was wrong; the real number is Y" and stop.
- **Never re-explain a thing already said once in the same reply.** If a
  number appears in the verdict, it does not reappear in the evidence.
- **Cut every sentence that exists to show diligence.** "I checked", "the
  control caught it", "this is the Nth time" — delete. The tool calls are
  visible; the user is not grading effort.
- **Detail goes in the commit, the memory file, or the peer message —
  never the reply.** That is what those artifacts are for.
- Default reply: **3-5 bullets, under ~60 words.** A long reply must be
  ASKED for ("report", "walkthrough", "what is left").
**STRICT RULE, EVERY SESSION, EVERY MODEL (Opus included): be succinct AND
clear.** Not style preference, not default that decay over long session.
Succinct = answer first, ~60 words, bullets not paragraphs (RULE 0). Clear =
plain words a colleague outside this repo can act on, no bare identifiers (RULE
-1). Both must hold; reply failing either one = failed reply.

### RULE -1 — CLARITY IS THE HARD RULE. IF THE USER CANNOT FOLLOW IT, IT FAILED.
User, 2026-08-12 (and 2026-08-07, 2026-08-11 — same complaint every time):
*"Ensure you are clearer — I have no clue what you are saying most of the
time."* Outranks brevity: short reply nobody understand worse than no reply.
Brevity already being followed when this said; defect is UNEXPLAINED INTERNAL
DETAIL, not length.

MECHANICAL TEST, apply to every sentence before sending: could a colleague who
know climate modelling but never opened this repo act on it? Need file name,
function name, job id, or flag to make sense → REWRITE IT.

- Say the THING, not the SYMBOL. "the model runs radiation 18 times more often
  than intended", not "split_rad=False makes rad_update_interval_steps inert".
- NO identifiers in a reply: no file:line, no function names, no config keys, no
  job ids, no PR numbers, unless user asked for that exact thing. They belong in
  commit message. Number user should act on is fine.
- One idea per line. Line has clause explaining clause → split it.
- State CONSEQUENCE first, cause second, stop. Not mechanism, not how it was
  found, not who found it.
- Never write sentence whose subject is piece of code. Subject = model, run,
  campaign, number, or user's decision.
- Something wrong → lead with what now false and what to do. Not narrative of
  discovery.
- Question to user = numbered options in plain words, with your pick.

FAILURE PATTERN, all three callouts: long autonomous stretches. Each status
reply drifted back into repo-internal vocabulary ("the pin", "the waiver", "the
factory", "the skeleton") meaning nothing outside this session. Re-read this
section whenever session run long, and before every status report.

### RULE 0 — HARD CAP ~60 WORDS. ANSWER FIRST. STANDING ORDER, ALL SESSIONS.
Five callouts in two days (2026-08-07/08). Not style preference — a cap. Lead
with result. 3-5 bullets. Stop. Job IDs, caveats, file:line and reasoning go in
COMMIT, never the reply. Retraction = ONE sentence plus corrected fact. Long
replies only when report explicitly requested.

User, 2026-08-07, after repeated callouts in single session ("you are too
verbose", "I have no idea what you are saying", "just laser-focused summary"):
**"Make being succinct, clear and to the point a strict rule for all future
sessions."** Outranks every other formatting instinct.

Mechanical test before sending — line fail → cut it:
- Would user act differently without this line? No → delete.
- Is it method, process, or what I tried? → delete. Tool calls visible.
- Is it caveat nobody would act on? → delete (put in commit).
- Is it re-explaining something already said once? → delete.

Shape: **verdict first**, then only evidence that change the verdict, then ONE
question with numbered options if decision needed. Default length few lines.
Long reply must earn it by being asked for (report, walkthrough, per-phase
notes).

Jargon banned unless sentence also say what it mean in plain words. Say "how
much communication the split costs", not "the ppermute round count".

### PLAIN LANGUAGE FIRST — the rule that finally worked (2026-08-07)
THIRD callout: *"I really don't understand when you talk to me. Things are
incredibly unclear."* Brevity alone did NOT fix it — earlier rules were being
followed. Real defect = **unexplained jargon** and leaving user to infer the
decision. What worked, now required shape:

- **Write for colleague, not reviewer.** Short sentences. Ordinary words.
- **NO unexplained domain jargon.** Terms like *ppermute, max_degree, halo fill,
  production-exact, VJP, chromatic bound* banned unless sentence also say what
  they mean in plain words. Prefer plain phrase outright: "how much
  communication the split costs", not "the round count of the comm graph".
- **Use short LABELLED blocks**, bolded, 1-3 lines each:
  **What I built / What broke / What I was wrong about / What's safe /
  What I need from you.** Labels do the navigating so user not.
- **End with ONE explicit question and numbered options** when decision needed.
  Say which you would pick and single deciding factor.
- **Asked to compare options → answer the comparison** — what each buy, what it
  cost, and question that decide it. No re-describing options.
- **Own errors in one plain sentence, at top.** "I was wrong earlier: X is not
  guaranteed." No narration of how discovered.
- **No tables, no file:line, no job ids in reply** unless asked. Those go in
  commit message. Reply is for the decision.

### Earlier callouts (same session) — still in force
**SECOND callout (2026-08-06), because rule below was written then ignored:
_"stop being verbose. It is really hard to understand. be more direct, to the
point, clear about issues. Bullets summarizing."_ Plus: _"aren't you using
caveman?"_ — terse mode was ACTIVE and I still wrote essays.**
- **BULLETS BY DEFAULT.** Prose paragraphs = failure mode. One line per fact.
- **Lead with the issue.** Not how it was found.
- **Delete every sentence not changing what user does next.**
- **Never re-explain caveat already stated once.**
- **No near-miss stories.** "I almost got X wrong" is not finding. State
  corrected number, move on.
- Terse mode (caveman/ponytail) active → APPLIES TO WHOLE REPLY — including
  findings, status, caveats. Length not substitute for rigor.

User callout 2026-08-06 (earlier): *"you are quite unclear... provide more succinct,
clear summary, clear choice, do not make many but targeted and verified
assumptions."* Evidence reader has to assemble into conclusion is not a report.
Structure, this order, then stop:

1. **VERDICT first, <=2 lines.** What true / what happened. Never open with
   method, caveats, or narration of what was run.
2. **THE DECISION, if any: ONE recommendation.** Name option you would take and
   why, one line. Menu of options with balanced caveats push work back onto
   user — only list alternatives when they genuinely must choose, and even then
   say which you'd pick.
3. **EVIDENCE: only what change the verdict.** Decisive number, file:line, or
   measurement. Not everything checked.

- **AT MOST ONE unverified claim per response, explicitly labelled PLAUSIBLE.**
  Everything else verified before stated. No enumerating candidate causes — pick
  the one you tested, report it. Untested hypotheses = clutter reading as
  findings.
- **Do not narrate the process.** Tool calls already visible. Report outcome,
  not itinerary.
- **Table only for >=3 things compared on >=2 axes.** Otherwise a sentence.
- **Retract in one line, move on.** No re-litigating superseded claim.
- **Caveats: only those changing what user should DO.** Limitation nobody would
  act on belong in commit message, not reply.
- Numerics change -> state effect on stability, accuracy, conservation,
  differentiability. No guesses as facts.
- **No Read images** unless user asks; report path.

**TERSE. Caveman register (user, 2026-08-06: "You speak too much... no need to waste tokens").** Fragments OK. Drop articles/filler/hedging/pleasantries. No narrating what you about to do, no restating request, no re-explaining finding already stated. Prose is for FINDINGS, not for process.
- **ALWAYS end with findings summary** — table or bullets: what measured, the number, CONFIRMED vs PLAUSIBLE, what still open. That summary = deliverable; rest is scaffolding.
- Long verbatim tool output → quote only DECISIVE line (`N passed`, failing assert, peak value).
- Commits/PRs/code comments/security warnings stay full English.

# iterate-with-codex agent
1. Implement change
2. `/codex:adversarial-review --wait`
3. Parse output
4. Fix flagged
5. `/codex:review --wait` again
6. Issues remain → 4
7. Stop when clean or after 30 iter
exec
/usr/bin/bash -lc 'git diff HEAD --stat && git diff HEAD -- packages/atmosphere packages/coupler packages/ml' in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 88ms:
 .../atmosphere/physics/clouds/subcolumns.py        |   2 +-
 .../atmosphere/physics/radiation/integration.py    |  23 ++
 .../atmosphere/physics/radiation/rrtmgp/rrtmgp.py  |  35 +++
 .../physics/radiation/rrtmgp/rte/two_stream.py     |  31 ++-
 .../coupler/legoesm/driver/compiled_segments.py    |  18 +-
 packages/coupler/legoesm/driver/model_driver.py    |  15 +-
 .../coupler/legoesm/driver/physics_pipeline.py     |  21 +-
 packages/ml/legoesm/training/scale_build.py        |  10 +-
 packages/tools/legoesm/forcing/external.py         | 206 +++++++++++----
 packages/tools/legoesm/forcing/surface_utils.py    |  66 +++--
 scripts/validate/probe_volcanic_sw_optics.py       |  89 +++++++
 tests/unit/test_aerosol_band_ccn.py                |  48 ++--
 tests/unit/test_cross_lane_forcing.py              |   2 +-
 tests/unit/test_volcanic_sw_optics.py              | 294 +++++++++++++++++++++
 tests/unit/test_wb_amip_deck_params.py             |   2 +-
 15 files changed, 763 insertions(+), 99 deletions(-)
diff --git a/packages/atmosphere/legoesm/atmosphere/physics/clouds/subcolumns.py b/packages/atmosphere/legoesm/atmosphere/physics/clouds/subcolumns.py
index f0a808310b..448d9d718f 100644
--- a/packages/atmosphere/legoesm/atmosphere/physics/clouds/subcolumns.py
+++ b/packages/atmosphere/legoesm/atmosphere/physics/clouds/subcolumns.py
@@ -413,7 +413,7 @@ PER_COLUMN_SOLVER_KEYS = frozenset({
     "T", "p_full", "p_half", "sfc_temperature", "q_v", "cos_zenith",
     "sfc_albedo", "sfc_albedo_direct", "sfc_emissivity", "o3_vmr", "o3_top_vmr",
     "aerosol_optical_depth", "aerosol_absorption_optical_depth_lw",
-    "cloud_path_liq", "cloud_path_ice", "cloud_r_eff_liq", "cloud_r_eff_ice",
+    "aerosol_volcanic_moments_sw", "cloud_path_liq", "cloud_path_ice", "cloud_r_eff_liq", "cloud_r_eff_ice",
     "cloud_fraction",
 })
 
diff --git a/packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py b/packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py
index c87b5dd24a..b4f343470c 100644
--- a/packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py
+++ b/packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py
@@ -977,6 +977,7 @@ def _call_radiation_backend(
     o3_top_vmr: jnp.ndarray | None = None,
     aerosol_od: jnp.ndarray | None = None,
     aerosol_lw_od: jnp.ndarray | None = None,
+    aerosol_sw_volc: jnp.ndarray | None = None,
     solar_spectral_fraction: jnp.ndarray | None = None,
     eccf: float | jnp.ndarray = 1.0,
     tsi: jnp.ndarray | None = None,
@@ -1045,6 +1046,10 @@ def _call_radiation_backend(
         ``aerosol_absorption_optical_depth_lw``.  ``None`` (default) is a
         no-op in the solver — byte-identical to no volcanic LW aerosol.
         Ignored by gray radiation.
+    aerosol_sw_volc : jnp.ndarray or None
+        Volcanic SHORTWAVE aerosol with its own per-band optics from the
+        CMIP6 file, (ncol, nlev, 3, n_bnd_sw) moments (tau, tau*ssa,
+        tau*ssa*g), passed to the solver as ``aerosol_volcanic_moments_sw``.
     solar_spectral_fraction : jnp.ndarray or None
         Per-g-point solar weights for spectral solar-cycle forcing,
         passed through to ``solve_columns``.
@@ -1061,6 +1066,11 @@ def _call_radiation_backend(
             "open_ocean_frac (direct/diffuse ocean albedo split) is wired for "
             f"the rrtmgp solver only; scheme {radiation_config.scheme!r} has a "
             "single surface albedo and would silently drop the direct beam.")
+    if aerosol_sw_volc is not None and radiation_config.scheme != "rrtmgp":
+        raise ValueError(
+            "aerosol_sw_volc (volcanic SW per-band optics) is wired for the "
+            f"rrtmgp solver only; scheme {radiation_config.scheme!r} would "
+            "silently drop the volcanic shortwave aerosol.")
     if radiation_config.scheme == "simple_lw":
         return _simple_lw_radiation(
             T=T, p_half=p_half, q_v=q_v, q_cloud=q_cloud, q_ice=q_ice,
@@ -1225,6 +1235,8 @@ def _call_radiation_backend(
         solar_spectral_fraction=solar_spectral_fraction,
         **cloud_kwargs,
     )
+    if aerosol_sw_volc is not None:
+        _rad_kwargs["aerosol_volcanic_moments_sw"] = aerosol_sw_volc
     if cloud_config is not None:
         # One bulk ice density for the PSD radius and the optics size (CAM6).
         _rad_kwargs["cloud_rho_ice"] = cloud_config.rho_cloud_ice
@@ -1814,6 +1826,9 @@ def _make_hydrostatic_radiation(
         _aer_lw_ext = (
             forcing.get("aerosol_lw_od") if forcing is not None else None
         )
+        _aer_volc_ext = (
+            forcing.get("aerosol_sw_volc") if forcing is not None else None
+        )
         _ghg_ext = forcing.get("ghg_vmr") if forcing is not None else None
         # Tile-blended SURFACE SHORTWAVE ALBEDO as a per-step TRACED forcing
         # (same channel as T_sfc / o3 / ghg above, so the JIT'd step never
@@ -2088,6 +2103,8 @@ def _make_hydrostatic_radiation(
                 _aer_ext = shard_columns(_aer_ext, column_mesh)
             if _aer_lw_ext is not None:
                 _aer_lw_ext = shard_columns(_aer_lw_ext, column_mesh)
+            if _aer_volc_ext is not None:
+                _aer_volc_ext = shard_columns(_aer_volc_ext, column_mesh)
             if _cf_ovr is not None:
                 _cf_ovr = shard_columns(_cf_ovr, column_mesh)
             if q_snow_col is not None:
@@ -2133,6 +2150,7 @@ def _make_hydrostatic_radiation(
             o3_top_vmr=_o3_top_ext,
             aerosol_od=_aer_ext,
             aerosol_lw_od=_aer_lw_ext,
+            aerosol_sw_volc=_aer_volc_ext,
             ghg_vmr_override=_ghg_ext,
             cloud_fraction_override=_cf_ovr,
             conv_precip=_conv_precip_col,
@@ -2193,6 +2211,7 @@ def _make_hydrostatic_radiation(
                 o3_top_vmr=_o3_top_ext,
                 aerosol_od=_aer_ext,
                 aerosol_lw_od=_aer_lw_ext,
+                aerosol_sw_volc=_aer_volc_ext,
                 ghg_vmr_override=_ghg_ext,
                 cloud_fraction_override=None,
                 conv_precip=None,
@@ -3264,6 +3283,9 @@ def _make_spectral_pe_radiation(
         _aer_lw_ext = (
             forcing.get("aerosol_lw_od") if forcing is not None else None
         )
+        _aer_volc_ext = (
+            forcing.get("aerosol_sw_volc") if forcing is not None else None
+        )
         _ghg_ext = forcing.get("ghg_vmr") if forcing is not None else None
 
         # Surface albedo / emissivity overrides: a per-step TRACED
@@ -3427,6 +3449,7 @@ def _make_spectral_pe_radiation(
             tsi=_tsi_ext,
             aerosol_od=_aer_ext,
             aerosol_lw_od=_aer_lw_ext,
+            aerosol_sw_volc=_aer_volc_ext,
             ghg_vmr_override=_ghg_ext,
             cloud_fraction_override=_cf_ovr,
             conv_mass_flux_up=_conv_mf_col,
diff --git a/packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py b/packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py
index a85a6f5555..1b40686a37 100644
--- a/packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py
+++ b/packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py
@@ -617,6 +617,7 @@ class RRTMGP:
       cloud_d_eff_snow: jnp.ndarray | None = None,
       snow_optics=None,
       cloud_snow_weight: jnp.ndarray | None = None,
+      aerosol_volcanic_moments_sw: jnp.ndarray | None = None,
   ):
       """Compute radiation for legoESM column arrays.
 
@@ -678,6 +679,12 @@ class RRTMGP:
           byte-identical.
       solar_spectral_fraction : jnp.ndarray | None
           Per-g-point solar source weights (ngpt_sw,).
+      aerosol_volcanic_moments_sw : jnp.ndarray | None
+          Shortwave aerosol with its OWN per-band optics (CMIP6 volcanic
+          sulfate), ``(ncol, nlev, 3, n_bnd_sw)`` per-layer moments
+          ``(tau, tau*ssa, tau*ssa*g)``, bands in ascending wavenumber (the
+          SW gas-optics band order).  Mixed with the grey
+          ``aerosol_optical_depth`` as a second component.  ``None`` = absent.
       ghg_vmr_override : dict | None
           Runtime GHG VMR overrides (e.g. ``{"co2": 4.15e-4}``).
       mcica_cloud_fraction : jnp.ndarray | None
@@ -757,6 +764,9 @@ class RRTMGP:
       if aerosol_absorption_optical_depth_lw is not None:
           aerosol_absorption_optical_depth_lw = jnp.asarray(
               aerosol_absorption_optical_depth_lw).astype(_table_dtype)
+      if aerosol_volcanic_moments_sw is not None:
+          aerosol_volcanic_moments_sw = jnp.asarray(
+              aerosol_volcanic_moments_sw).astype(_table_dtype)
       if cloud_path_liq is not None:
           cloud_path_liq = jnp.asarray(cloud_path_liq).astype(_table_dtype)
       if cloud_path_ice is not None:
@@ -868,6 +878,10 @@ class RRTMGP:
           aerosol_optical_depth = _zero(aerosol_optical_depth)
           aerosol_absorption_optical_depth_lw = _zero(
               aerosol_absorption_optical_depth_lw)
+          if aerosol_volcanic_moments_sw is not None:
+              aerosol_volcanic_moments_sw = jnp.concatenate(
+                  [jnp.zeros_like(aerosol_volcanic_moments_sw[:, :1]),
+                   aerosol_volcanic_moments_sw], axis=1)
           if cloud_r_eff_liq is not None:
               cloud_r_eff_liq = _top(cloud_r_eff_liq, cloud_r_eff_liq[:, :1])
           if cloud_r_eff_ice is not None:
@@ -1117,6 +1131,25 @@ class RRTMGP:
       # full SW solve so the per-band selection is defined in exactly one place.
       aer_ssa, aer_g = _resolve_aerosol_sw_optics(config, optics_lib)
 
+      # Per-band aerosol component (volcanic): (ncol, nlev, 3, n_bnd) ->
+      # (3, n_bnd, ncol, 1, nlev+2), the same halo / bottom-first layout as
+      # the grey aerosol OD; the moments are linear in OD, so the linear halo
+      # extrapolation + clip is the treatment the grey OD gets.
+      if aerosol_volcanic_moments_sw is not None:
+          n_bnd = int(optics_lib.gas_optics_sw.n_bnd)
+          if aerosol_volcanic_moments_sw.shape[-2:] != (3, n_bnd):
+              raise ValueError(
+                  "aerosol_volcanic_moments_sw trailing shape "
+                  f"{aerosol_volcanic_moments_sw.shape[-2:]}, expected (3, "
+                  f"{n_bnd}) for this SW gas-optics table.")
+          _nc, _nl = aerosol_volcanic_moments_sw.shape[:2]
+          _vm = jnp.moveaxis(aerosol_volcanic_moments_sw, (2, 3), (0, 1))
+          _vm = _add_halos(_vm[..., ::-1].reshape(3 * n_bnd * _nc, 1, _nl))
+          volc_moments_3d = jnp.clip(
+              _vm.reshape(3, n_bnd, _nc, 1, _nl + 2), 0.0, None)
+      else:
+          volc_moments_3d = None
+
       # Optional aerosol optical depth (longwave, pure absorber)
       if aerosol_absorption_optical_depth_lw is not None:
           aerosol_od_lw_3d = jnp.clip(
@@ -1157,6 +1190,7 @@ class RRTMGP:
               aerosol_optical_depth=aerosol_od_3d,
               aerosol_single_scattering_albedo=aer_ssa,
               aerosol_asymmetry_factor=aer_g,
+              aerosol_volcanic_moments=volc_moments_3d,
           )
           # Strip the singleton Y axis + vertical halos, flip back to the
           # legoESM TOA-first convention -> (n_gpt, ncol, nlev).
@@ -1257,6 +1291,7 @@ class RRTMGP:
           aerosol_optical_depth=aerosol_od_3d,
           aerosol_single_scattering_albedo=aer_ssa,
           aerosol_asymmetry_factor=aer_g,
+          aerosol_volcanic_moments=volc_moments_3d,
           solar_fraction_by_gpt=solar_weights,
           use_scan=config.use_scan,
           gpoint_batch_size=getattr(config, "gpoint_batch_size", 0),
diff --git a/packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rte/two_stream.py b/packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rte/two_stream.py
index 44aa2ef1ab..1920aede7b 100644
--- a/packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rte/two_stream.py
+++ b/packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rte/two_stream.py
@@ -630,6 +630,7 @@ def compute_sw_optical_props_gpt(
     aerosol_single_scattering_albedo: float | Array,
     aerosol_asymmetry_factor: float | Array,
     cloud_snow=None,
+    aerosol_volcanic_moments: Array | None = None,
 ) -> dict[str, Array]:
   """Per-g-point shortwave optical properties (optical_depth / ssa / asymmetry).
 
@@ -653,7 +654,8 @@ def compute_sw_optical_props_gpt(
   )
   return _mix_sw_aerosol_and_clip(
       igpt, optics_lib, sw_optical_props, aerosol_optical_depth,
-      aerosol_single_scattering_albedo, aerosol_asymmetry_factor)
+      aerosol_single_scattering_albedo, aerosol_asymmetry_factor,
+      aerosol_volcanic_moments)
 
 
 def _mix_sw_aerosol_and_clip(
@@ -663,8 +665,18 @@ def _mix_sw_aerosol_and_clip(
     aerosol_optical_depth: Array | None,
     aerosol_single_scattering_albedo: float | Array,
     aerosol_asymmetry_factor: float | Array,
+    aerosol_volcanic_moments: Array | None = None,
 ) -> dict[str, Array]:
-  """Mix aerosol into one g-point's SW gas(+cloud) optics, then clip ssa/g."""
+  """Mix aerosol into one g-point's SW gas(+cloud) optics, then clip ssa/g.
+
+  ``aerosol_volcanic_moments`` ``(3, n_bnd_sw, *layer_shape)``: a second
+  aerosol component with its OWN per-band optics, given as scattering
+  moments ``(tau, tau*ssa, tau*ssa*g)`` (stratospheric volcanic sulfate from
+  the CMIP6 file); added to the same tau / tau*ssa / tau*ssa*g sums as the
+  grey aerosol, so the mix is the exact optical-depth-weighted combination.
+  """
+  if aerosol_optical_depth is None and aerosol_volcanic_moments is not None:
+    aerosol_optical_depth = jnp.zeros_like(sw_optical_props['optical_depth'])
   if aerosol_optical_depth is not None:
     # Per-band aerosol optics: ``aerosol_single_scattering_albedo`` /
     # ``aerosol_asymmetry_factor`` may be a SCALAR (grey aerosol, historical
@@ -687,6 +699,10 @@ def _mix_sw_aerosol_and_clip(
     w_bg = sw_optical_props['ssa']
     g_bg = sw_optical_props['asymmetry_factor']
     w_num = tau_bg * w_bg + tau_aer * ssa_aer
+    if aerosol_volcanic_moments is not None:
+      volc = aerosol_volcanic_moments[:, optics_lib.gas_optics_sw.g_point_to_bnd[igpt]]
+      tau_tot = tau_tot + volc[0]
+      w_num = w_num + volc[1]
     # AD-safe SW optical-property mixing (restored from commit 59407953
     # after AIMIP-#312 merge reverted it).  ``a / jnp.maximum(b, eps)``
     # has a ``-a/b**2`` VJP that overflows when ``b`` is at the floor —
@@ -705,6 +721,8 @@ def _mix_sw_aerosol_and_clip(
         tau_bg * w_bg * g_bg
         + tau_aer * ssa_aer * g_aer
     )
+    if aerosol_volcanic_moments is not None:
+      g_num = g_num + volc[2]
     g_denom = tau_tot * jnp.maximum(w_tot, 1.0e-12)
     g_tot = jnp.clip(
         safe_divide(g_num, g_denom, eps=1.0e-12, fill=0.0),
@@ -748,6 +766,7 @@ def compute_sw_optical_field(
     aerosol_optical_depth: Array | None = None,
     aerosol_single_scattering_albedo: float | Array = _AEROSOL_SSA_DEFAULT,
     aerosol_asymmetry_factor: float | Array = _AEROSOL_ASYM_DEFAULT,
+    aerosol_volcanic_moments: Array | None = None,
 ) -> dict[str, Array]:
   """Stack per-g-point shortwave optical fields for the 3D MC ray tracer.
 
@@ -770,6 +789,7 @@ def compute_sw_optical_field(
         cloud_r_eff_liq, cloud_path_liq, cloud_r_eff_ice, cloud_path_ice,
         cloud_fraction, aerosol_optical_depth,
         aerosol_single_scattering_albedo, aerosol_asymmetry_factor,
+        aerosol_volcanic_moments=aerosol_volcanic_moments,
     )
     # Gas Rayleigh scattering optical depth (the gas scatter is ALL Rayleigh) as
     # a fraction of TOTAL scattering (gas + cloud + aerosol) -> the per-cell
@@ -866,6 +886,7 @@ def solve_sw(
     cloud_path_fn=None,
     clear_sky: bool = False,
     cloud_snow=None,
+    aerosol_volcanic_moments: Array | None = None,
 ) -> dict[str, Array]:
   """Solves the two-stream radiative transfer equation for shortwave.
 
@@ -899,6 +920,9 @@ def solve_sw(
       shape as `temperature`), added to SW extinction.
     aerosol_single_scattering_albedo: Bulk aerosol single-scattering albedo.
     aerosol_asymmetry_factor: Bulk aerosol asymmetry factor.
+    aerosol_volcanic_moments: Optional second aerosol component with per-band
+      optics, ``(3, n_bnd_sw, *temperature.shape)`` moments
+      ``(tau, tau*ssa, tau*ssa*g)`` (see ``_mix_sw_aerosol_and_clip``).
     solar_fraction_by_gpt: Optional external spectral solar weights by g-point.
     use_scan: Whether to use scan or for loops for the recurrent operation.
 
@@ -997,6 +1021,7 @@ def solve_sw(
           cloud_r_eff_liq, liq_g, cloud_r_eff_ice, ice_g,
           cloud_fraction, aerosol_optical_depth,
           aerosol_single_scattering_albedo, aerosol_asymmetry_factor,
+          aerosol_volcanic_moments=aerosol_volcanic_moments,
           **snow_kw,
       )
       sw_fluxes = _sw_gpoint_fluxes(igpt, sw_optical_props)
@@ -1009,7 +1034,7 @@ def solve_sw(
           igpt, gas_props, False, cloud_r_eff_liq, liq_g,
           cloud_r_eff_ice, ice_g, cloud_fraction=cloud_fraction, **snow_kw)
       _aer = (aerosol_optical_depth, aerosol_single_scattering_albedo,
-              aerosol_asymmetry_factor)
+              aerosol_asymmetry_factor, aerosol_volcanic_moments)
       sw_fluxes = _sw_gpoint_fluxes(
           igpt, _mix_sw_aerosol_and_clip(igpt, optics_lib, cloudy_props, *_aer))
       sw_fluxes_clr = _sw_gpoint_fluxes(
diff --git a/packages/coupler/legoesm/driver/compiled_segments.py b/packages/coupler/legoesm/driver/compiled_segments.py
index c08a7d8667..9043592c7e 100644
--- a/packages/coupler/legoesm/driver/compiled_segments.py
+++ b/packages/coupler/legoesm/driver/compiled_segments.py
@@ -883,6 +883,10 @@ class SegmentForcing(NamedTuple):
     # Column-mean ozone VMR above the model top (ncol,) for the RRTMGP
     # overhead layer.  ``None`` => the solver falls back to the top-layer o3.
     o3_top_vmr: jax.Array | None = None
+    # Volcanic SHORTWAVE aerosol with the CMIP6 file's own per-band optics,
+    # (ncol, nlev, 3, n_bnd_sw) moments (tau, tau*ssa, tau*ssa*g).  ``None`` =>
+    # no such source (the volcanic SW, if any, is inside ``aerosol_od``).
+    aerosol_sw_volc: jax.Array | None = None
 
 
 # Canonical GHG species ordering for the ghg_vmr array.  The three halogens
@@ -940,6 +944,7 @@ def pack_forcing(
     land_ml_params=None,
     aerosol_ccn_aod=None,
     o3_top_vmr=None,
+    aerosol_sw_volc=None,
 ) -> SegmentForcing:
     """Pack per-segment forcing into a SegmentForcing pytree.
 
@@ -1066,6 +1071,9 @@ def pack_forcing(
         o3_top_vmr=(
             None if o3_top_vmr is None else jnp.asarray(o3_top_vmr)
         ),
+        aerosol_sw_volc=(
+            None if aerosol_sw_volc is None else jnp.asarray(aerosol_sw_volc)
+        ),
     )
 
 
@@ -1081,7 +1089,7 @@ GRID_SHAPED_FORCING_FIELDS = (
     "sfc_shflx_override", "sfc_lhflx_override", "sfc_evap_override",
     "sfc_taux_override", "sfc_tauy_override",
     "sfc_lw_up", "sfc_sw_up", "sfc_sw_down", "land_frac", "aerosol_ccn_aod",
-    "o3_top_vmr",
+    "o3_top_vmr", "aerosol_sw_volc",
 )
 
 
@@ -1603,6 +1611,7 @@ def split_physics_single_rank(carry, T_new, u_new, v_new, p_s_new,
         albedo_ice=statics.albedo_ice, albedo_ocean=statics.albedo_ocean,
         ghg_vmr_override=statics.ghg_vmr_override,
         aerosol_lw_od=statics.forcing.aerosol_lw_od,
+        aerosol_sw_volc=statics.forcing.aerosol_sw_volc,
         aerosol_ccn_aod=statics.forcing.aerosol_ccn_aod,
         o3_top_vmr=statics.forcing.o3_top_vmr,
         sfc_albedo_override=statics.forcing.sfc_albedo_override,
@@ -1658,6 +1667,7 @@ def split_physics_single_rank(carry, T_new, u_new, v_new, p_s_new,
                 statics.forcing.solar_weights, statics.forcing.s_0,
                 statics.forcing.o3_vmr, statics.forcing.aerosol_od,
                 aerosol_lw_od_precomputed=statics.forcing.aerosol_lw_od,
+                aerosol_sw_volc_precomputed=statics.forcing.aerosol_sw_volc,
                 aerosol_ccn_aod=statics.forcing.aerosol_ccn_aod,
                 o3_top_vmr=statics.forcing.o3_top_vmr,
                 tau_equator=statics.tau_equator, tau_pole=statics.tau_pole,
@@ -2592,6 +2602,7 @@ def build_segment_fn(
                     albedo_ice=_albedo_ice, albedo_ocean=_albedo_ocean,
                     ghg_vmr_override=_ghg_vmr_override,
                     aerosol_lw_od=forcing.aerosol_lw_od,
+                    aerosol_sw_volc=forcing.aerosol_sw_volc,
                     aerosol_ccn_aod=forcing.aerosol_ccn_aod,
                     o3_top_vmr=forcing.o3_top_vmr,
                     sfc_albedo_override=forcing.sfc_albedo_override,
@@ -2634,6 +2645,7 @@ def build_segment_fn(
                             forcing.solar_weights, forcing.s_0,
                             forcing.o3_vmr, forcing.aerosol_od,
                             aerosol_lw_od_precomputed=forcing.aerosol_lw_od,
+                            aerosol_sw_volc_precomputed=forcing.aerosol_sw_volc,
                             aerosol_ccn_aod=forcing.aerosol_ccn_aod,
                             o3_top_vmr=forcing.o3_top_vmr,
                             tau_equator=_tau_equator, tau_pole=_tau_pole,
@@ -3073,6 +3085,7 @@ def build_segment_fn(
                     forcing.solar_weights, forcing.s_0,
                     forcing.o3_vmr, forcing.aerosol_od,
                     aerosol_lw_od_precomputed=forcing.aerosol_lw_od,
+                    aerosol_sw_volc_precomputed=forcing.aerosol_sw_volc,
                     aerosol_ccn_aod=forcing.aerosol_ccn_aod,
                     o3_top_vmr=forcing.o3_top_vmr,
                     tau_equator=_tau_equator, tau_pole=_tau_pole,
@@ -3106,6 +3119,7 @@ def build_segment_fn(
                         forcing.solar_weights, forcing.s_0,
                         forcing.o3_vmr, forcing.aerosol_od,
                         aerosol_lw_od_precomputed=forcing.aerosol_lw_od,
+                        aerosol_sw_volc_precomputed=forcing.aerosol_sw_volc,
                         aerosol_ccn_aod=forcing.aerosol_ccn_aod,
                         o3_top_vmr=forcing.o3_top_vmr,
                         tau_equator=_tau_equator, tau_pole=_tau_pole,
@@ -3160,6 +3174,7 @@ def build_segment_fn(
                     forcing.solar_weights, forcing.s_0,
                     forcing.o3_vmr, forcing.aerosol_od,
                     aerosol_lw_od_precomputed=forcing.aerosol_lw_od,
+                    aerosol_sw_volc_precomputed=forcing.aerosol_sw_volc,
                     aerosol_ccn_aod=forcing.aerosol_ccn_aod,
                     o3_top_vmr=forcing.o3_top_vmr,
                     tau_equator=_tau_equator, tau_pole=_tau_pole,
@@ -3193,6 +3208,7 @@ def build_segment_fn(
                         forcing.solar_weights, forcing.s_0,
                         forcing.o3_vmr, forcing.aerosol_od,
                         aerosol_lw_od_precomputed=forcing.aerosol_lw_od,
+                        aerosol_sw_volc_precomputed=forcing.aerosol_sw_volc,
                         aerosol_ccn_aod=forcing.aerosol_ccn_aod,
                         o3_top_vmr=forcing.o3_top_vmr,
                         tau_equator=_tau_equator, tau_pole=_tau_pole,
diff --git a/packages/coupler/legoesm/driver/model_driver.py b/packages/coupler/legoesm/driver/model_driver.py
index 8290a8f0a3..939f3dfca6 100644
--- a/packages/coupler/legoesm/driver/model_driver.py
+++ b/packages/coupler/legoesm/driver/model_driver.py
@@ -5848,6 +5848,7 @@ class ModelDriver:
             and bool(cfg.volcanic_aerosol_file)
         )
         self._aerosol_lw_od = None
+        self._aerosol_sw_volc = None
         self._o3_top_vmr = None
         self._aerosol_ccn_aod = None
         self._aerosol_config = AerosolConfig(
@@ -6155,7 +6156,8 @@ class ModelDriver:
         # (tracer_data.F90 bilinear lon/lat lininterp).
         lon_col = (jnp.asarray(self._owned_lon() if lon is None else lon)
                    .reshape(ncol) if self._aerosol_active else None)
-        aerosol_od, _ccn_aod, self._aerosol_lw_od = aerosol_layer_fields(
+        (aerosol_od, _ccn_aod, self._aerosol_lw_od,
+         self._aerosol_sw_volc) = aerosol_layer_fields(
             self._aerosol_config, day, lat_col, p_half_col, w_strat,
             active=self._aerosol_active, lw_active=self._aerosol_lw_active,
             dtype=p_s.dtype, lon_col=lon_col)
@@ -16479,6 +16481,10 @@ class ModelDriver:
                         _aer_lw = getattr(self, "_aerosol_lw_od", None)
                         if self._aerosol_lw_active and _aer_lw is not None:
                             _forcing_daily["aerosol_lw_od"] = _aer_lw
+                        # Volcanic SW with its own per-band optics (None =>
+                        # absent; the volcanic SW is then in aerosol_od).
+                        if getattr(self, "_aerosol_sw_volc", None) is not None:
+                            _forcing_daily["aerosol_sw_volc"] = self._aerosol_sw_volc
                         # Tropospheric visible-band CCN AOD (absent => sum(aerosol_od)).
                         if getattr(self, "_aerosol_ccn_aod", None) is not None:
                             _forcing_daily["aerosol_ccn_aod"] = self._aerosol_ccn_aod
@@ -18463,6 +18469,8 @@ class ModelDriver:
                         _aer_lw = getattr(self, "_aerosol_lw_od", None)
                         if self._aerosol_lw_active and _aer_lw is not None:
                             _ext_daily["aerosol_lw_od"] = _aer_lw
+                        if getattr(self, "_aerosol_sw_volc", None) is not None:
+                            _ext_daily["aerosol_sw_volc"] = self._aerosol_sw_volc
                         # Tropospheric visible-band CCN AOD (absent => sum(aerosol_od)).
                         if getattr(self, "_aerosol_ccn_aod", None) is not None:
                             _ext_daily["aerosol_ccn_aod"] = self._aerosol_ccn_aod
@@ -19436,6 +19444,7 @@ class ModelDriver:
                 solar_weights=solar_weights, s_0=current_s_0,
                 o3_vmr=o3_vmr, aerosol_od=aerosol_od,
                 aerosol_lw_od=getattr(self, "_aerosol_lw_od", None),
+                aerosol_sw_volc=getattr(self, "_aerosol_sw_volc", None),
                 aerosol_ccn_aod=getattr(self, "_aerosol_ccn_aod", None),
                 o3_top_vmr=getattr(self, "_o3_top_vmr", None),
                 ghg_vmr=ghg_vmr,
@@ -20275,6 +20284,7 @@ class ModelDriver:
             o3_vmr=ctx["o3_vmr"], aerosol_od=ctx["aerosol_od"],
             aerosol_ccn_aod=getattr(self, "_aerosol_ccn_aod", None),
             o3_top_vmr=getattr(self, "_o3_top_vmr", None),
+            aerosol_sw_volc=getattr(self, "_aerosol_sw_volc", None),
             ghg_vmr=ctx["ghg_vmr"],
             # NOTE: transient cover is deliberately NOT injected here.  The training
             # segment returns run_segment.raw (un-jitted), so the land calibration
@@ -20651,6 +20661,7 @@ class ModelDriver:
                 solar_weights=solar_weights, s_0=current_s_0,
                 o3_vmr=o3_vmr, aerosol_od=aerosol_od,
                 aerosol_lw_od=getattr(self, "_aerosol_lw_od", None),
+                aerosol_sw_volc=getattr(self, "_aerosol_sw_volc", None),
                 aerosol_ccn_aod=getattr(self, "_aerosol_ccn_aod", None),
                 o3_top_vmr=getattr(self, "_o3_top_vmr", None),
                 ghg_vmr=ghg_vmr,
@@ -21540,6 +21551,7 @@ class ModelDriver:
                 held_sw_up_toa, held_lw_up_toa, held_sw_down_toa,
                 ghg_vmr_override=ghg_vmr,
                 aerosol_lw_od=getattr(self, "_aerosol_lw_od", None),
+                aerosol_sw_volc=getattr(self, "_aerosol_sw_volc", None),
                 aerosol_ccn_aod=getattr(self, "_aerosol_ccn_aod", None),
                 o3_top_vmr=getattr(self, "_o3_top_vmr", None),
                 land_ml=(self._land_ml_state
@@ -21685,6 +21697,7 @@ class ModelDriver:
                     held_sw_up_toa, held_lw_up_toa, held_sw_down_toa,
                     ghg_vmr_override=ghg_vmr,
                     aerosol_lw_od=getattr(self, "_aerosol_lw_od", None),
+                    aerosol_sw_volc=getattr(self, "_aerosol_sw_volc", None),
                     aerosol_ccn_aod=getattr(self, "_aerosol_ccn_aod", None),
                     o3_top_vmr=getattr(self, "_o3_top_vmr", None),
                     land_ml=(self._land_ml_state
diff --git a/packages/coupler/legoesm/driver/physics_pipeline.py b/packages/coupler/legoesm/driver/physics_pipeline.py
index 650a26ad92..95a4f44f65 100644
--- a/packages/coupler/legoesm/driver/physics_pipeline.py
+++ b/packages/coupler/legoesm/driver/physics_pipeline.py
@@ -2896,6 +2896,7 @@ class PhysicsPipeline:
                                solar_weights, s_0,
                                o3_vmr_precomputed, aerosol_od_precomputed,
                                aerosol_lw_od_precomputed=None,
+                               aerosol_sw_volc_precomputed=None,
                                aerosol_ccn_aod=None,
                                o3_top_vmr=None,
                                tau_equator=None, tau_pole=None,
@@ -3318,6 +3319,7 @@ class PhysicsPipeline:
             o3_vmr_precomputed = _shard(o3_vmr_precomputed)
             aerosol_od_precomputed = _shard(aerosol_od_precomputed)
             aerosol_lw_od_precomputed = _shard(aerosol_lw_od_precomputed)
+            aerosol_sw_volc_precomputed = _shard(aerosol_sw_volc_precomputed)
             o3_top_vmr = _shard(o3_top_vmr)
             if cloud_kwargs:
                 cloud_kwargs = {k: (v if k == "cloud_rho_ice" else _shard(v))
@@ -3334,6 +3336,8 @@ class PhysicsPipeline:
             ghg_vmr_override=ghg_vmr_override,
             aerosol_lw_od_col=aerosol_lw_od_precomputed,
             o3_top_vmr_col=o3_top_vmr,
+            **({} if aerosol_sw_volc_precomputed is None
+               else {"aerosol_sw_volc_col": aerosol_sw_volc_precomputed}),
             # only the rrtmgp radiation_fn takes it (cesm_ocean_albedo)
             **({} if open_ocean_frac_col is None
                else {"open_ocean_frac_col": open_ocean_frac_col}),
@@ -3545,6 +3549,7 @@ class PhysicsPipeline:
                          albedo_ocean=pipeline.albedo_ocean,
                          ghg_vmr_override=None,
                          aerosol_lw_od=None,
+                         aerosol_sw_volc=None,
                          aerosol_ccn_aod=None,
                          o3_top_vmr=None,
                          T_land=None,
@@ -3626,6 +3631,7 @@ class PhysicsPipeline:
                         day_of_year, seconds_of_day,
                         solar_weights, s_0, o3_vmr, aerosol_od,
                         aerosol_lw_od_precomputed=aerosol_lw_od,
+                        aerosol_sw_volc_precomputed=aerosol_sw_volc,
                         aerosol_ccn_aod=aerosol_ccn_aod,
                         o3_top_vmr=o3_top_vmr,
                         tau_equator=tau_equator, tau_pole=tau_pole,
@@ -3841,8 +3847,11 @@ def _build_none_radiation_fn(config):
                      cloud_path_liq_lw=None, cloud_path_ice_lw=None,
                      cloud_r_eff_liq=None, cloud_r_eff_ice=None,
                      cloud_rho_ice=None,
-                     cloud_fraction=None):
+                     cloud_fraction=None, aerosol_sw_volc_col=None):
         del aerosol_lw_od_col, o3_top_vmr_col  # zero-radiation: no-ops
+        if aerosol_sw_volc_col is not None:
+            raise ValueError("zero radiation cannot consume the volcanic SW "
+                             "per-band aerosol optics; use rrtmgp radiation.")
         ncol, nlev = T_col.shape
         z_full = jnp.zeros((ncol, nlev), dtype=T_col.dtype)
         z_half = jnp.zeros((ncol, nlev + 1), dtype=T_col.dtype)
@@ -3896,9 +3905,12 @@ def _build_gray_radiation_fn(config):
                      cloud_path_liq_lw=None, cloud_path_ice_lw=None,
                      cloud_r_eff_liq=None, cloud_r_eff_ice=None,
                      cloud_rho_ice=None,
-                     cloud_fraction=None):
+                     cloud_fraction=None, aerosol_sw_volc_col=None):
         del ghg_vmr_override  # gray radiation does not use GHG concentrations
         del aerosol_lw_od_col, o3_top_vmr_col  # gray: no aerosol LW od / ozone
+        if aerosol_sw_volc_col is not None:
+            raise ValueError("gray radiation cannot consume the volcanic SW "
+                             "per-band aerosol optics; use rrtmgp radiation.")
         del cloud_path_liq, cloud_path_ice, cloud_r_eff_liq, cloud_r_eff_ice, cloud_fraction
         del cloud_rho_ice
         del cloud_path_liq_lw, cloud_path_ice_lw  # gray: no cloud optics
@@ -4026,7 +4038,8 @@ def _build_rrtmgp_radiation_fn(config):
                      cloud_path_liq_lw=None, cloud_path_ice_lw=None,
                      cloud_r_eff_liq=None, cloud_r_eff_ice=None,
                      cloud_rho_ice=None,
-                     cloud_fraction=None, open_ocean_frac_col=None):
+                     cloud_fraction=None, open_ocean_frac_col=None,
+                     aerosol_sw_volc_col=None):
         del tau_equator, tau_pole  # RRTMGP does not use gray optical depth
         _sw_scale = None
         # Eccentricity scales the incident SW *flux* by (a/r)^2; the orbital
@@ -4092,6 +4105,8 @@ def _build_rrtmgp_radiation_fn(config):
             o3_vmr=o3_vmr_col,
             aerosol_optical_depth=aerosol_od_col,
             aerosol_absorption_optical_depth_lw=aerosol_lw_od_col,
+            **({} if aerosol_sw_volc_col is None
+               else {"aerosol_volcanic_moments_sw": aerosol_sw_volc_col}),
             solar_spectral_fraction=solar_weights if solar_weights.size > 0 else None,
             ghg_vmr_override=ghg_vmr_override,
             cloud_path_liq=cloud_path_liq,
diff --git a/packages/ml/legoesm/training/scale_build.py b/packages/ml/legoesm/training/scale_build.py
index 7ab2138bbc..a36d3ecbe3 100644
--- a/packages/ml/legoesm/training/scale_build.py
+++ b/packages/ml/legoesm/training/scale_build.py
@@ -1607,8 +1607,13 @@ def wb_aerosol_forcing(spec, grid, sigma, carry, time_ns):
     per_year = [aerosol_layer_fields(
         acfg._replace(start_year=y), sim_day, lat_col, p_half, w_strat,
         active=True, lw_active=lw_active, dtype=p_s.dtype) for y in years]
-    od, ccn, lw = (jnp.mean(jnp.stack(f), axis=0) for f in zip(*per_year))
-    return {"aerosol_od": od, "aerosol_ccn_aod": ccn, "aerosol_lw_od": lw}
+    od, ccn, lw = (jnp.mean(jnp.stack(f), axis=0)
+                   for f in list(zip(*per_year))[:3])
+    out = {"aerosol_od": od, "aerosol_ccn_aod": ccn, "aerosol_lw_od": lw}
+    volc = [f[3] for f in per_year]
+    if volc[0] is not None:
+        out["aerosol_sw_volc"] = jnp.mean(jnp.stack(volc), axis=0)
+    return out
 
 
 def build_spectral_forcing(sl, grid, time_ns, year, *, aerosol=None,
@@ -1850,6 +1855,7 @@ def load_era5_samples(cfg, yml, grid, sigma, *,
             solar_weights=ctx["solar_weights"], s_0=ctx["current_s_0"],
             o3_vmr=ctx["o3_vmr"], aerosol_od=ctx["aerosol_od"],
             o3_top_vmr=getattr(driver, "_o3_top_vmr", None),
+            aerosol_sw_volc=getattr(driver, "_aerosol_sw_volc", None),
             **plane_kwargs)
         return (ic, target, forcing)
 

exec
/usr/bin/bash -lc 'git diff HEAD -- packages/tools/legoesm/forcing/external.py packages/tools/legoesm/forcing/surface_utils.py' in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 0ms:
diff --git a/packages/tools/legoesm/forcing/external.py b/packages/tools/legoesm/forcing/external.py
index c4ac00c97e..6f5e4ae027 100644
--- a/packages/tools/legoesm/forcing/external.py
+++ b/packages/tools/legoesm/forcing/external.py
@@ -824,11 +824,8 @@ def _load_volcanic_profile(
 ) -> tuple[np.ndarray, object, np.ndarray, np.ndarray, np.ndarray]:
     """CMIP6 volcanic LW aerosol as a PRESSURE-RESOLVED absorption-OD profile.
 
-    ``kind="sw"`` returns the SHORTWAVE ``ext_sun`` EXTINCTION profile instead
-    (the SW slot takes extinction; SSA/g come from the radiation config), with
-    the solar bands collapsed solar-weighted via :func:`_collapse_sw_bands`
-    (flat mean if the file has no ``wl1_sun``/``wl2_sun``).  The B1/B3 notes
-    below are LW-only; B2 (stratospheric placement) applies to both.
+    The shortwave counterpart, with the file's own per-band optics, is
+    :func:`_load_volcanic_sw_optics_profile`.
 
     Fixes three defects in feeding ``bc_aeropt_cmip6_volc_lw_b16`` to the
     RRTMGP longwave ABSORPTION optical-depth slot:
@@ -865,10 +862,12 @@ def _load_volcanic_profile(
       = top / lowest pressure), aligned so ``aod_profile[..., j]`` occupies
       ``[p_edges[j], p_edges[j+1]]``.
 
-    Raises ``ValueError`` if the file lacks ``ext_earth`` (``ext_sun`` for
-    SW; caller treats as no profile source and stays byte-identical).
+    Raises ``ValueError`` if the file lacks ``ext_earth`` (caller treats as no
+    profile source and stays byte-identical).
     """
-    ext_name = "ext_sun" if kind == "sw" else "ext_earth"
+    if kind != "lw":
+        raise ValueError(f"_load_volcanic_profile: unknown kind {kind!r}")
+    ext_name = "ext_earth"
     ds = _open_forcing_dataset(path)
     if ext_name not in ds.data_vars:
         ds.close()
@@ -880,9 +879,7 @@ def _load_volcanic_profile(
     dims = list(var.dims)
 
     # B1: single-scattering albedo -> absorption fraction (1-omega) per band.
-    if kind == "sw":
-        omega = 0.0  # SW slot wants extinction, not absorption
-    elif "omega_earth" in ds.data_vars:
+    if "omega_earth" in ds.data_vars:
         omega = np.clip(
             np.asarray(ds["omega_earth"].values, dtype=np.float64), 0.0, 1.0,
         )
@@ -926,12 +923,7 @@ def _load_volcanic_profile(
          if d not in (time_name, lat_name, alt_name)),
         None,
     )
-    if band_name is not None and kind == "sw":
-        collapsed = _collapse_sw_bands(ds, abs_ext, dims, band_name)
-        abs_ext = (np.nanmean(abs_ext, axis=dims.index(band_name))
-                   if collapsed is None else collapsed)
-        dims = [d for d in dims if d != band_name]
-    elif band_name is not None:
+    if band_name is not None:
         band_axis = dims.index(band_name)
         if "wl1_earth" in ds and "wl2_earth" in ds:
             weights = _planck_band_weights(
@@ -952,22 +944,135 @@ def _load_volcanic_profile(
     order = [dims.index(time_name), dims.index(lat_name), dims.index(alt_name)]
     prof = np.transpose(abs_ext, order)  # (ntime, nlat, nalt) [1/km]
 
-    # Altitude layer edges (midpoints, outer half-steps) and thickness [km].
+    dz_km, p_edges = _volcanic_layer_geometry(alt_km)
+    # Per-layer AOD [-] = absorption extinction [1/km] * dz [km]; layer axis
+    # flipped to ASCENDING pressure (top first) to match ``p_edges``.
+    aod_profile = (prof * dz_km[None, None, :])[..., ::-1]
+    return mid_days, first_date, lat, aod_profile, p_edges
+
+
+def _volcanic_layer_geometry(alt_km: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
+    """Layer thickness ``dz_km`` (nalt,) [km] in file (altitude-ascending)
+    order, and the US-Std-Atm-1976 layer pressure edges (nalt+1,) [Pa] in
+    ASCENDING pressure (top first) -- the caller flips its layer axis to
+    match.  Edges are altitude midpoints with outer half-steps."""
     edges_km = np.empty(alt_km.size + 1, dtype=np.float64)
     edges_km[1:-1] = 0.5 * (alt_km[:-1] + alt_km[1:])
     edges_km[0] = alt_km[0] - 0.5 * (alt_km[1] - alt_km[0])
     edges_km[-1] = alt_km[-1] + 0.5 * (alt_km[-1] - alt_km[-2])
-    dz_km = np.abs(np.diff(edges_km))  # (nalt,)
+    dz_km = np.abs(np.diff(edges_km))
+    p_edges = _ussa1976_pressure(edges_km * 1000.0)[::-1]
+    return dz_km, p_edges
+
+
+# Solar-band optics variables of the CMIP6 volcanic file that give the
+# stratospheric aerosol its OWN shortwave single-scattering properties.
+_VOLC_SW_OPTICS_VARS = ("ext_sun", "omega_sun", "g_sun", "wl1_sun", "wl2_sun")
+# RRTMGP shortwave band limits [cm-1], ascending (``bnd_limits_wavenumber`` of
+# rrtmgp-gas-sw-g112/g224; a test pins this table to both files).  The volcanic
+# file's bands are mapped 1:1 onto these in ascending-wavenumber order.
+_RRTMGP_SW_BAND_LIMITS_CM1 = (
+    (820.0, 2680.0), (2680.0, 3250.0), (3250.0, 4000.0), (4000.0, 4650.0),
+    (4650.0, 5150.0), (5150.0, 6150.0), (6150.0, 7700.0), (7700.0, 8050.0),
+    (8050.0, 12850.0), (12850.0, 16000.0), (16000.0, 22650.0),
+    (22650.0, 29000.0), (29000.0, 38000.0), (38000.0, 50000.0),
+)
+# The CMIP6 file is on the RRTMG-SW grid, whose one differing edge is
+# 2600 cm-1 where RRTMGP has 2680 cm-1 (RRTMG band 29 is 820-2600, band 16
+# 2600-3250); every other edge must agree to this relative tolerance.
+_RRTMG_RRTMGP_EDGE_DIFF_CM1 = ((2680.0, 2600.0),)
+_VOLC_SW_EDGE_RTOL = 1.0e-3
 
-    # Per-layer AOD [-] = absorption extinction [1/km] * dz [km].
-    aod_profile = prof * dz_km[None, None, :]
-    p_edges = _ussa1976_pressure(edges_km * 1000.0)  # [Pa]
 
-    # Return ASCENDING pressure (top first): altitude ascending -> pressure
-    # descending, so flip the layer axis and the edges together.
-    aod_profile = aod_profile[..., ::-1]
-    p_edges = p_edges[::-1]
-    return mid_days, first_date, lat, aod_profile, p_edges
+@lru_cache(maxsize=4)
+def _load_volcanic_sw_optics_profile(path: str):
+    """CMIP6 volcanic SHORTWAVE optics as per-band scattering MOMENTS.
+
+    Returns ``(mid_days, first_date, lat, moments, p_edges, band_wn)``:
+
+    * ``moments`` ``(ntime, nlat, nlayer, 3, nband)`` float32, per file layer
+      and solar band: ``(tau, tau*omega, tau*omega*g)`` with
+      ``tau = ext_sun*dz``.  Moments (not tau/omega/g) because the layer remap
+      and the tropopause weight are linear in them, so remapping each moment
+      and mixing afterwards is exact.
+    * layer axis ASCENDING pressure (top first) aligned with ``p_edges``
+      ``(nlayer+1,)`` [Pa] (same geometry as the LW profile);
+    * bands sorted to ASCENDING wavenumber (the file ships RRTMG/E3SM order,
+      820-2600 cm-1 last; RRTMGP orders 820-2680 cm-1 first), with
+      ``band_wn`` ``(nband, 2)`` [cm-1] the sorted band edges so the radiation
+      can check the 1:1 band match against its own gas-optics table.
+
+    Returns None when the file has no ``ext_sun`` (a legacy column ``aod``
+    file, which stays in the column path).  A file with ``ext_sun`` but
+    without the other ``_VOLC_SW_OPTICS_VARS``, with bands that do not match
+    :data:`_RRTMGP_SW_BAND_LIMITS_CM1`, with non-finite values, or with a
+    negative asymmetry factor (the layer remap is for non-negative optical
+    depths) raises.
+    """
+    ds = _open_forcing_dataset(path)
+    if "ext_sun" not in ds:
+        ds.close()
+        return None
+    missing = [v for v in _VOLC_SW_OPTICS_VARS if v not in ds]
+    if missing:
+        ds.close()
+        raise ValueError(
+            f"Volcanic file {path!r} has ext_sun but no {missing}: the "
+            "stratospheric aerosol needs its own solar-band optics.")
+    var = ds["ext_sun"]
+    dims = list(var.dims)
+    alt_name = next((v for v in ("altitude", "alt", "height", "lev") if v in dims),
+                    None)
+    lat_name = next((v for v in ("lat", "latitude") if v in dims), None)
+    time_name = next((v for v in ("time", "month") if v in dims), None)
+    band_name = ds["wl1_sun"].dims[0]
+    if None in (alt_name, lat_name, time_name) or band_name not in dims:
+        ds.close()
+        raise ValueError(
+            f"CMIP6 volcanic file {path!r}: ext_sun dims {dims} lack an "
+            "altitude/latitude/time/solar-band axis.")
+    order = [dims.index(n) for n in (time_name, lat_name, alt_name, band_name)]
+    ext, omega, g = (
+        np.transpose(np.asarray(ds[v].values, dtype=np.float64), order)
+        for v in ("ext_sun", "omega_sun", "g_sun"))
+    for name, a in (("ext_sun", ext), ("omega_sun", omega), ("g_sun", g)):
+        if not np.all(np.isfinite(a)):
+            ds.close()
+            raise ValueError(f"Non-finite {name} in volcanic file {path!r}.")
+    if np.any(g < 0.0) or np.any(ext < 0.0):
+        ds.close()
+        raise ValueError(f"Negative ext_sun or g_sun in volcanic file {path!r}.")
+    if time_name == "time":
+        mid_days, first_date = _read_time_axis(ds)
+    else:
+        mid_days = np.array([15.5 + 30.4375 * m for m in range(ext.shape[0])])
+        first_date = None
+    lat = np.asarray(ds[lat_name].values, dtype=np.float64)
+    alt_km = np.asarray(ds[alt_name].values, dtype=np.float64)
+    wl1 = np.asarray(ds["wl1_sun"].values, dtype=np.float64)
+    wl2 = np.asarray(ds["wl2_sun"].values, dtype=np.float64)
+    ds.close()
+
+    dz_km, p_edges = _volcanic_layer_geometry(alt_km)
+    tau = ext * dz_km[None, None, :, None]
+    ssa = np.clip(omega, 0.0, 1.0)
+    moments = np.stack([tau, tau * ssa, tau * ssa * np.clip(g, -1.0, 1.0)],
+                       axis=-2)                       # (t, lat, alt, 3, band)
+    # um -> cm-1: wn = 1e4 / wl.
+    band_wn = np.stack([1.0e4 / np.maximum(wl1, wl2),
+                        1.0e4 / np.minimum(wl1, wl2)], axis=-1)
+    srt = np.argsort(band_wn[:, 0])
+    band_wn = band_wn[srt]
+    ref = np.asarray(_RRTMGP_SW_BAND_LIMITS_CM1)
+    for a, b in _RRTMG_RRTMGP_EDGE_DIFF_CM1:
+        ref = np.where(ref == a, b, ref)
+    if band_wn.shape != ref.shape or not np.allclose(
+            band_wn, ref, rtol=_VOLC_SW_EDGE_RTOL, atol=0.0):
+        raise ValueError(
+            f"Volcanic file {path!r} solar bands {band_wn.tolist()} [cm-1] do "
+            "not map 1:1 onto the RRTMGP shortwave bands.")
+    moments = moments[..., srt][:, :, ::-1].astype(np.float32)
+    return mid_days, first_date, lat, moments, p_edges, band_wn
 
 
 @lru_cache(maxsize=16)
@@ -2341,29 +2446,40 @@ def get_aerosol_lw_at_time(config: AerosolConfig, day: float,
     return _volcanic_profile_at_time(config, day, lat_grid, "lw")
 
 
-def get_aerosol_sw_volcanic_at_time(config: AerosolConfig, day: float,
-                                    lat_grid: jnp.ndarray | None = None):
-    """SHORTWAVE volcanic extinction as a pressure-resolved profile.
+def get_aerosol_sw_volcanic_optics_at_time(config: AerosolConfig, day: float,
+                                           lat_grid: jnp.ndarray):
+    """SHORTWAVE volcanic aerosol with its OWN per-band optics from the file.
 
-    Same contract as :func:`get_aerosol_lw_at_time` but for ``ext_sun``
-    (solar-weighted gray band collapse), gated on ``volcanic_enabled``.  None
-    for a legacy column ``aod(time, lat)`` volcanic file (no profile) -- the
-    caller then keeps that column inside :func:`get_aerosol_at_time`.
+    Returns ``(moments_col, p_edges, band_wn)``: ``moments_col``
+    ``(ncol, nsrc, 3, nband)`` per-file-layer ``(tau, tau*omega,
+    tau*omega*g)`` (see :func:`_load_volcanic_sw_optics_profile`), ``p_edges``
+    ``(nsrc+1,)`` [Pa] ascending, ``band_wn`` ``(nband, 2)`` [cm-1].  None
+    when volcanic forcing is off or the file carries no ``omega_sun`` /
+    ``g_sun`` / band edges (the caller then keeps the gray path).
     """
     if not config.volcanic_enabled or not config.volcanic_path:
         return None
-    return _volcanic_profile_at_time(config, day, lat_grid, "sw")
+    loaded = _load_volcanic_sw_optics_profile(config.volcanic_path)
+    if loaded is None:
+        return None
+    out = _volcanic_profile_at_time(config, day, lat_grid, "sw_optics",
+                                    loaded=loaded[:4] + (loaded[4],))
+    return out[0], out[1], loaded[5]
 
 
-def _volcanic_profile_at_time(config, day, lat_grid, kind):
-    """Shared body of the LW / SW volcanic profile samplers."""
-    try:
-        mid_days_v, first_date_v, lat_v, prof_v, p_edges = (
-            _load_volcanic_profile(config.volcanic_path, kind)
-        )
-    except ValueError:
-        # File has no ext_earth / ext_sun band — treat as "no source".
-        return None
+def _volcanic_profile_at_time(config, day, lat_grid, kind, loaded=None):
+    """Shared body of the LW / SW volcanic profile samplers.  Trailing axes
+    after the layer axis (the SW-optics moment/band axes) are carried along."""
+    if loaded is not None:
+        mid_days_v, first_date_v, lat_v, prof_v, p_edges = loaded
+    else:
+        try:
+            mid_days_v, first_date_v, lat_v, prof_v, p_edges = (
+                _load_volcanic_profile(config.volcanic_path, kind)
+            )
+        except ValueError:
+            # File has no ext_earth / ext_sun band — treat as "no source".
+            return None
 
     # Interpolate the (ntime, nlat, nlayer) profile in time -> (nlat, nlayer).
     if len(mid_days_v) > 12:
@@ -2385,7 +2501,7 @@ def _volcanic_profile_at_time(config, day, lat_grid, kind):
         # intentionally NOT summed away (unlike the gray SW/column path).
         prof_col = _interp_zonal_to_grid(lat_v, prof_t, lat_grid)
         prof_col = jnp.reshape(
-            jnp.clip(prof_col, 0.0, None), (-1, prof_t.shape[-1]),
+            jnp.clip(prof_col, 0.0, None), (-1, *prof_t.shape[1:]),
         )
         return prof_col, jnp.asarray(p_edges)
 
diff --git a/packages/tools/legoesm/forcing/surface_utils.py b/packages/tools/legoesm/forcing/surface_utils.py
index a763e6724d..19be83f055 100644
--- a/packages/tools/legoesm/forcing/surface_utils.py
+++ b/packages/tools/legoesm/forcing/surface_utils.py
@@ -805,43 +805,71 @@ def stratospheric_layer_weight(p_half_col: jnp.ndarray,
     return jnp.where(bot <= p_tp, 1.0, jnp.where(top >= p_tp, 0.0, 0.5))
 
 
+# Columns per host-side chunk of the volcanic SW optics interpolation + remap
+# (memory bound only; the result does not depend on it).
+_VOLC_SW_COL_CHUNK = 4096
+
+
 def aerosol_layer_fields(acfg, day, lat_col, p_half_col, w_strat, *,
                          active, lw_active, dtype, lon_col=None):
     """Per-layer aerosol forcing for one day: ``(aerosol_od, ccn_aod,
-    aerosol_lw_od)``, as the AMIP driver feeds radiation and the aerosol-CCN
-    droplet number.  ``ccn_aod`` is None when ``active`` is off; the two
-    optical-depth fields are zeros (ncol, nlev) when their switch is off.
-    ``lon_col`` [rad] reads the SW and CCN aerosol at each column's longitude;
-    None reads their zonal mean."""
+    aerosol_lw_od, aerosol_sw_volc)``, as the AMIP driver feeds radiation and
+    the aerosol-CCN droplet number.  ``ccn_aod`` is None when ``active`` is
+    off; the two optical-depth fields are zeros (ncol, nlev) when their switch
+    is off.  ``aerosol_sw_volc`` is the volcanic SW aerosol with its own
+    per-band optics, ``(ncol, nlev, 3, nband)`` moments ``(tau, tau*omega,
+    tau*omega*g)`` in ascending-wavenumber band order, kept out of
+    ``aerosol_od``; None when volcanic forcing is off or the volcanic file is a
+    legacy column ``aod`` file (which then stays in ``aerosol_od`` with the
+    grey aerosol SSA/g).  ``lon_col`` [rad] reads the
+    SW and CCN aerosol at each column's longitude; None reads their zonal
+    mean."""
     from legoesm.forcing.external import (
         get_aerosol_at_time,
         get_aerosol_lw_at_time,
-        get_aerosol_sw_volcanic_at_time,
+        get_aerosol_sw_volcanic_optics_at_time,
     )
 
     ncol, nlev = p_half_col.shape[0], p_half_col.shape[1] - 1
     ccn_aod = None
+    aerosol_sw_volc = None
     aerosol_od = jnp.zeros((ncol, nlev), dtype=dtype)
     if active:
         _acfg = acfg
-        # Volcanic SW (CMIP6 ``ext_sun``) is placed at its true
-        # stratospheric pressure like the LW profile below, NOT spread
-        # with the tropospheric column by pressure mass.  A legacy
-        # column-only volcanic file (profile None) stays in the column.
-        volc_sw = get_aerosol_sw_volcanic_at_time(_acfg, day, lat_grid=lat_col)
+        # Volcanic SW with the file's own per-band extinction / SSA / g:
+        # each moment is placed at its true stratospheric pressure (linear
+        # remap + tropopause weight, so mixing the remapped moments is exact)
+        # and kept OUT of the grey ``aerosol_od``.
+        # Column chunks bound the host interpolation's (chunk, nsrc, 3, nb)
+        # float64 temporaries (the full res6 field is ~1 GiB).
+        lat_flat = jnp.reshape(jnp.asarray(lat_col), (ncol,))
+        chunks = []
+        for c0 in range(0, ncol, _VOLC_SW_COL_CHUNK):
+            sl = slice(c0, min(c0 + _VOLC_SW_COL_CHUNK, ncol))
+            volc_opt = get_aerosol_sw_volcanic_optics_at_time(
+                _acfg, day, lat_flat[sl])
+            if volc_opt is None:
+                break
+            mom, p_edges_v, _band_wn = volc_opt          # (nc, nsrc, 3, nb)
+            nc, nsrc, n_m, n_b = mom.shape
+            flat = jnp.moveaxis(jnp.asarray(mom).reshape(nc, nsrc, n_m * n_b),
+                                -1, 0)                   # (3*nb, nc, nsrc)
+            placed = jax.vmap(place_stratospheric_aod_profile_to_layers,
+                              in_axes=(0, None, None))(
+                flat, jnp.asarray(p_edges_v), p_half_col[sl]) * w_strat[sl]
+            chunks.append(jnp.moveaxis(placed, 0, -1).reshape(
+                nc, nlev, n_m, n_b).astype(dtype))
+        if chunks:
+            aerosol_sw_volc = jnp.concatenate(chunks, axis=0)
+        # A legacy column-only volcanic file (no ext_sun) stays in the column.
         _trop_cfg = _acfg._replace(volcanic_enabled=False)
         aerosol_col = get_aerosol_at_time(
-            _acfg if volc_sw is None else _trop_cfg, day, lat_grid=lat_col,
-            lon_grid=lon_col,
+            _acfg if aerosol_sw_volc is None else _trop_cfg,
+            day, lat_grid=lat_col, lon_grid=lon_col,
         )
         aerosol_od = distribute_column_aod_to_layers(
             jnp.asarray(aerosol_col), p_half_col,
         )
-        if volc_sw is not None:
-            aerosol_od = aerosol_od + (place_stratospheric_aod_profile_to_layers(
-                jnp.asarray(volc_sw[0]), jnp.asarray(volc_sw[1]), p_half_col,
-            ) * w_strat
-            ).astype(aerosol_od.dtype)
         # AOD->CCN input: TROPOSPHERIC column in the visible (550 nm)
         # band, volcanic excluded, from ``ccn_path`` if set.  The Andreae 2009
         # fit is local total AOT: a zonal mean gives polluted E China and the
@@ -872,4 +900,4 @@ def aerosol_layer_fields(acfg, day, lat_col, p_half_col, w_strat, *,
                 jnp.asarray(prof_col), jnp.asarray(p_edges), p_half_col,
             ) * w_strat
             ).astype(dtype)
-    return aerosol_od, ccn_aod, aerosol_lw_od
+    return aerosol_od, ccn_aod, aerosol_lw_od, aerosol_sw_volc

exec
/usr/bin/bash -lc 'git diff HEAD -- packages/atmosphere' in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 0ms:
diff --git a/packages/atmosphere/legoesm/atmosphere/physics/clouds/subcolumns.py b/packages/atmosphere/legoesm/atmosphere/physics/clouds/subcolumns.py
index f0a808310b..448d9d718f 100644
--- a/packages/atmosphere/legoesm/atmosphere/physics/clouds/subcolumns.py
+++ b/packages/atmosphere/legoesm/atmosphere/physics/clouds/subcolumns.py
@@ -413,7 +413,7 @@ PER_COLUMN_SOLVER_KEYS = frozenset({
     "T", "p_full", "p_half", "sfc_temperature", "q_v", "cos_zenith",
     "sfc_albedo", "sfc_albedo_direct", "sfc_emissivity", "o3_vmr", "o3_top_vmr",
     "aerosol_optical_depth", "aerosol_absorption_optical_depth_lw",
-    "cloud_path_liq", "cloud_path_ice", "cloud_r_eff_liq", "cloud_r_eff_ice",
+    "aerosol_volcanic_moments_sw", "cloud_path_liq", "cloud_path_ice", "cloud_r_eff_liq", "cloud_r_eff_ice",
     "cloud_fraction",
 })
 
diff --git a/packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py b/packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py
index c87b5dd24a..b4f343470c 100644
--- a/packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py
+++ b/packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py
@@ -977,6 +977,7 @@ def _call_radiation_backend(
     o3_top_vmr: jnp.ndarray | None = None,
     aerosol_od: jnp.ndarray | None = None,
     aerosol_lw_od: jnp.ndarray | None = None,
+    aerosol_sw_volc: jnp.ndarray | None = None,
     solar_spectral_fraction: jnp.ndarray | None = None,
     eccf: float | jnp.ndarray = 1.0,
     tsi: jnp.ndarray | None = None,
@@ -1045,6 +1046,10 @@ def _call_radiation_backend(
         ``aerosol_absorption_optical_depth_lw``.  ``None`` (default) is a
         no-op in the solver — byte-identical to no volcanic LW aerosol.
         Ignored by gray radiation.
+    aerosol_sw_volc : jnp.ndarray or None
+        Volcanic SHORTWAVE aerosol with its own per-band optics from the
+        CMIP6 file, (ncol, nlev, 3, n_bnd_sw) moments (tau, tau*ssa,
+        tau*ssa*g), passed to the solver as ``aerosol_volcanic_moments_sw``.
     solar_spectral_fraction : jnp.ndarray or None
         Per-g-point solar weights for spectral solar-cycle forcing,
         passed through to ``solve_columns``.
@@ -1061,6 +1066,11 @@ def _call_radiation_backend(
             "open_ocean_frac (direct/diffuse ocean albedo split) is wired for "
             f"the rrtmgp solver only; scheme {radiation_config.scheme!r} has a "
             "single surface albedo and would silently drop the direct beam.")
+    if aerosol_sw_volc is not None and radiation_config.scheme != "rrtmgp":
+        raise ValueError(
+            "aerosol_sw_volc (volcanic SW per-band optics) is wired for the "
+            f"rrtmgp solver only; scheme {radiation_config.scheme!r} would "
+            "silently drop the volcanic shortwave aerosol.")
     if radiation_config.scheme == "simple_lw":
         return _simple_lw_radiation(
             T=T, p_half=p_half, q_v=q_v, q_cloud=q_cloud, q_ice=q_ice,
@@ -1225,6 +1235,8 @@ def _call_radiation_backend(
         solar_spectral_fraction=solar_spectral_fraction,
         **cloud_kwargs,
     )
+    if aerosol_sw_volc is not None:
+        _rad_kwargs["aerosol_volcanic_moments_sw"] = aerosol_sw_volc
     if cloud_config is not None:
         # One bulk ice density for the PSD radius and the optics size (CAM6).
         _rad_kwargs["cloud_rho_ice"] = cloud_config.rho_cloud_ice
@@ -1814,6 +1826,9 @@ def _make_hydrostatic_radiation(
         _aer_lw_ext = (
             forcing.get("aerosol_lw_od") if forcing is not None else None
         )
+        _aer_volc_ext = (
+            forcing.get("aerosol_sw_volc") if forcing is not None else None
+        )
         _ghg_ext = forcing.get("ghg_vmr") if forcing is not None else None
         # Tile-blended SURFACE SHORTWAVE ALBEDO as a per-step TRACED forcing
         # (same channel as T_sfc / o3 / ghg above, so the JIT'd step never
@@ -2088,6 +2103,8 @@ def _make_hydrostatic_radiation(
                 _aer_ext = shard_columns(_aer_ext, column_mesh)
             if _aer_lw_ext is not None:
                 _aer_lw_ext = shard_columns(_aer_lw_ext, column_mesh)
+            if _aer_volc_ext is not None:
+                _aer_volc_ext = shard_columns(_aer_volc_ext, column_mesh)
             if _cf_ovr is not None:
                 _cf_ovr = shard_columns(_cf_ovr, column_mesh)
             if q_snow_col is not None:
@@ -2133,6 +2150,7 @@ def _make_hydrostatic_radiation(
             o3_top_vmr=_o3_top_ext,
             aerosol_od=_aer_ext,
             aerosol_lw_od=_aer_lw_ext,
+            aerosol_sw_volc=_aer_volc_ext,
             ghg_vmr_override=_ghg_ext,
             cloud_fraction_override=_cf_ovr,
             conv_precip=_conv_precip_col,
@@ -2193,6 +2211,7 @@ def _make_hydrostatic_radiation(
                 o3_top_vmr=_o3_top_ext,
                 aerosol_od=_aer_ext,
                 aerosol_lw_od=_aer_lw_ext,
+                aerosol_sw_volc=_aer_volc_ext,
                 ghg_vmr_override=_ghg_ext,
                 cloud_fraction_override=None,
                 conv_precip=None,
@@ -3264,6 +3283,9 @@ def _make_spectral_pe_radiation(
         _aer_lw_ext = (
             forcing.get("aerosol_lw_od") if forcing is not None else None
         )
+        _aer_volc_ext = (
+            forcing.get("aerosol_sw_volc") if forcing is not None else None
+        )
         _ghg_ext = forcing.get("ghg_vmr") if forcing is not None else None
 
         # Surface albedo / emissivity overrides: a per-step TRACED
@@ -3427,6 +3449,7 @@ def _make_spectral_pe_radiation(
             tsi=_tsi_ext,
             aerosol_od=_aer_ext,
             aerosol_lw_od=_aer_lw_ext,
+            aerosol_sw_volc=_aer_volc_ext,
             ghg_vmr_override=_ghg_ext,
             cloud_fraction_override=_cf_ovr,
             conv_mass_flux_up=_conv_mf_col,
diff --git a/packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py b/packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py
index a85a6f5555..1b40686a37 100644
--- a/packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py
+++ b/packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py
@@ -617,6 +617,7 @@ class RRTMGP:
       cloud_d_eff_snow: jnp.ndarray | None = None,
       snow_optics=None,
       cloud_snow_weight: jnp.ndarray | None = None,
+      aerosol_volcanic_moments_sw: jnp.ndarray | None = None,
   ):
       """Compute radiation for legoESM column arrays.
 
@@ -678,6 +679,12 @@ class RRTMGP:
           byte-identical.
       solar_spectral_fraction : jnp.ndarray | None
           Per-g-point solar source weights (ngpt_sw,).
+      aerosol_volcanic_moments_sw : jnp.ndarray | None
+          Shortwave aerosol with its OWN per-band optics (CMIP6 volcanic
+          sulfate), ``(ncol, nlev, 3, n_bnd_sw)`` per-layer moments
+          ``(tau, tau*ssa, tau*ssa*g)``, bands in ascending wavenumber (the
+          SW gas-optics band order).  Mixed with the grey
+          ``aerosol_optical_depth`` as a second component.  ``None`` = absent.
       ghg_vmr_override : dict | None
           Runtime GHG VMR overrides (e.g. ``{"co2": 4.15e-4}``).
       mcica_cloud_fraction : jnp.ndarray | None
@@ -757,6 +764,9 @@ class RRTMGP:
       if aerosol_absorption_optical_depth_lw is not None:
           aerosol_absorption_optical_depth_lw = jnp.asarray(
               aerosol_absorption_optical_depth_lw).astype(_table_dtype)
+      if aerosol_volcanic_moments_sw is not None:
+          aerosol_volcanic_moments_sw = jnp.asarray(
+              aerosol_volcanic_moments_sw).astype(_table_dtype)
       if cloud_path_liq is not None:
           cloud_path_liq = jnp.asarray(cloud_path_liq).astype(_table_dtype)
       if cloud_path_ice is not None:
@@ -868,6 +878,10 @@ class RRTMGP:
           aerosol_optical_depth = _zero(aerosol_optical_depth)
           aerosol_absorption_optical_depth_lw = _zero(
               aerosol_absorption_optical_depth_lw)
+          if aerosol_volcanic_moments_sw is not None:
+              aerosol_volcanic_moments_sw = jnp.concatenate(
+                  [jnp.zeros_like(aerosol_volcanic_moments_sw[:, :1]),
+                   aerosol_volcanic_moments_sw], axis=1)
           if cloud_r_eff_liq is not None:
               cloud_r_eff_liq = _top(cloud_r_eff_liq, cloud_r_eff_liq[:, :1])
           if cloud_r_eff_ice is not None:
@@ -1117,6 +1131,25 @@ class RRTMGP:
       # full SW solve so the per-band selection is defined in exactly one place.
       aer_ssa, aer_g = _resolve_aerosol_sw_optics(config, optics_lib)
 
+      # Per-band aerosol component (volcanic): (ncol, nlev, 3, n_bnd) ->
+      # (3, n_bnd, ncol, 1, nlev+2), the same halo / bottom-first layout as
+      # the grey aerosol OD; the moments are linear in OD, so the linear halo
+      # extrapolation + clip is the treatment the grey OD gets.
+      if aerosol_volcanic_moments_sw is not None:
+          n_bnd = int(optics_lib.gas_optics_sw.n_bnd)
+          if aerosol_volcanic_moments_sw.shape[-2:] != (3, n_bnd):
+              raise ValueError(
+                  "aerosol_volcanic_moments_sw trailing shape "
+                  f"{aerosol_volcanic_moments_sw.shape[-2:]}, expected (3, "
+                  f"{n_bnd}) for this SW gas-optics table.")
+          _nc, _nl = aerosol_volcanic_moments_sw.shape[:2]
+          _vm = jnp.moveaxis(aerosol_volcanic_moments_sw, (2, 3), (0, 1))
+          _vm = _add_halos(_vm[..., ::-1].reshape(3 * n_bnd * _nc, 1, _nl))
+          volc_moments_3d = jnp.clip(
+              _vm.reshape(3, n_bnd, _nc, 1, _nl + 2), 0.0, None)
+      else:
+          volc_moments_3d = None
+
       # Optional aerosol optical depth (longwave, pure absorber)
       if aerosol_absorption_optical_depth_lw is not None:
           aerosol_od_lw_3d = jnp.clip(
@@ -1157,6 +1190,7 @@ class RRTMGP:
               aerosol_optical_depth=aerosol_od_3d,
               aerosol_single_scattering_albedo=aer_ssa,
               aerosol_asymmetry_factor=aer_g,
+              aerosol_volcanic_moments=volc_moments_3d,
           )
           # Strip the singleton Y axis + vertical halos, flip back to the
           # legoESM TOA-first convention -> (n_gpt, ncol, nlev).
@@ -1257,6 +1291,7 @@ class RRTMGP:
           aerosol_optical_depth=aerosol_od_3d,
           aerosol_single_scattering_albedo=aer_ssa,
           aerosol_asymmetry_factor=aer_g,
+          aerosol_volcanic_moments=volc_moments_3d,
           solar_fraction_by_gpt=solar_weights,
           use_scan=config.use_scan,
           gpoint_batch_size=getattr(config, "gpoint_batch_size", 0),
diff --git a/packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rte/two_stream.py b/packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rte/two_stream.py
index 44aa2ef1ab..1920aede7b 100644
--- a/packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rte/two_stream.py
+++ b/packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rte/two_stream.py
@@ -630,6 +630,7 @@ def compute_sw_optical_props_gpt(
     aerosol_single_scattering_albedo: float | Array,
     aerosol_asymmetry_factor: float | Array,
     cloud_snow=None,
+    aerosol_volcanic_moments: Array | None = None,
 ) -> dict[str, Array]:
   """Per-g-point shortwave optical properties (optical_depth / ssa / asymmetry).
 
@@ -653,7 +654,8 @@ def compute_sw_optical_props_gpt(
   )
   return _mix_sw_aerosol_and_clip(
       igpt, optics_lib, sw_optical_props, aerosol_optical_depth,
-      aerosol_single_scattering_albedo, aerosol_asymmetry_factor)
+      aerosol_single_scattering_albedo, aerosol_asymmetry_factor,
+      aerosol_volcanic_moments)
 
 
 def _mix_sw_aerosol_and_clip(
@@ -663,8 +665,18 @@ def _mix_sw_aerosol_and_clip(
     aerosol_optical_depth: Array | None,
     aerosol_single_scattering_albedo: float | Array,
     aerosol_asymmetry_factor: float | Array,
+    aerosol_volcanic_moments: Array | None = None,
 ) -> dict[str, Array]:
-  """Mix aerosol into one g-point's SW gas(+cloud) optics, then clip ssa/g."""
+  """Mix aerosol into one g-point's SW gas(+cloud) optics, then clip ssa/g.
+
+  ``aerosol_volcanic_moments`` ``(3, n_bnd_sw, *layer_shape)``: a second
+  aerosol component with its OWN per-band optics, given as scattering
+  moments ``(tau, tau*ssa, tau*ssa*g)`` (stratospheric volcanic sulfate from
+  the CMIP6 file); added to the same tau / tau*ssa / tau*ssa*g sums as the
+  grey aerosol, so the mix is the exact optical-depth-weighted combination.
+  """
+  if aerosol_optical_depth is None and aerosol_volcanic_moments is not None:
+    aerosol_optical_depth = jnp.zeros_like(sw_optical_props['optical_depth'])
   if aerosol_optical_depth is not None:
     # Per-band aerosol optics: ``aerosol_single_scattering_albedo`` /
     # ``aerosol_asymmetry_factor`` may be a SCALAR (grey aerosol, historical
@@ -687,6 +699,10 @@ def _mix_sw_aerosol_and_clip(
     w_bg = sw_optical_props['ssa']
     g_bg = sw_optical_props['asymmetry_factor']
     w_num = tau_bg * w_bg + tau_aer * ssa_aer
+    if aerosol_volcanic_moments is not None:
+      volc = aerosol_volcanic_moments[:, optics_lib.gas_optics_sw.g_point_to_bnd[igpt]]
+      tau_tot = tau_tot + volc[0]
+      w_num = w_num + volc[1]
     # AD-safe SW optical-property mixing (restored from commit 59407953
     # after AIMIP-#312 merge reverted it).  ``a / jnp.maximum(b, eps)``
     # has a ``-a/b**2`` VJP that overflows when ``b`` is at the floor —
@@ -705,6 +721,8 @@ def _mix_sw_aerosol_and_clip(
         tau_bg * w_bg * g_bg
         + tau_aer * ssa_aer * g_aer
     )
+    if aerosol_volcanic_moments is not None:
+      g_num = g_num + volc[2]
     g_denom = tau_tot * jnp.maximum(w_tot, 1.0e-12)
     g_tot = jnp.clip(
         safe_divide(g_num, g_denom, eps=1.0e-12, fill=0.0),
@@ -748,6 +766,7 @@ def compute_sw_optical_field(
     aerosol_optical_depth: Array | None = None,
     aerosol_single_scattering_albedo: float | Array = _AEROSOL_SSA_DEFAULT,
     aerosol_asymmetry_factor: float | Array = _AEROSOL_ASYM_DEFAULT,
+    aerosol_volcanic_moments: Array | None = None,
 ) -> dict[str, Array]:
   """Stack per-g-point shortwave optical fields for the 3D MC ray tracer.
 
@@ -770,6 +789,7 @@ def compute_sw_optical_field(
         cloud_r_eff_liq, cloud_path_liq, cloud_r_eff_ice, cloud_path_ice,
         cloud_fraction, aerosol_optical_depth,
         aerosol_single_scattering_albedo, aerosol_asymmetry_factor,
+        aerosol_volcanic_moments=aerosol_volcanic_moments,
     )
     # Gas Rayleigh scattering optical depth (the gas scatter is ALL Rayleigh) as
     # a fraction of TOTAL scattering (gas + cloud + aerosol) -> the per-cell
@@ -866,6 +886,7 @@ def solve_sw(
     cloud_path_fn=None,
     clear_sky: bool = False,
     cloud_snow=None,
+    aerosol_volcanic_moments: Array | None = None,
 ) -> dict[str, Array]:
   """Solves the two-stream radiative transfer equation for shortwave.
 
@@ -899,6 +920,9 @@ def solve_sw(
       shape as `temperature`), added to SW extinction.
     aerosol_single_scattering_albedo: Bulk aerosol single-scattering albedo.
     aerosol_asymmetry_factor: Bulk aerosol asymmetry factor.
+    aerosol_volcanic_moments: Optional second aerosol component with per-band
+      optics, ``(3, n_bnd_sw, *temperature.shape)`` moments
+      ``(tau, tau*ssa, tau*ssa*g)`` (see ``_mix_sw_aerosol_and_clip``).
     solar_fraction_by_gpt: Optional external spectral solar weights by g-point.
     use_scan: Whether to use scan or for loops for the recurrent operation.
 
@@ -997,6 +1021,7 @@ def solve_sw(
           cloud_r_eff_liq, liq_g, cloud_r_eff_ice, ice_g,
           cloud_fraction, aerosol_optical_depth,
           aerosol_single_scattering_albedo, aerosol_asymmetry_factor,
+          aerosol_volcanic_moments=aerosol_volcanic_moments,
           **snow_kw,
       )
       sw_fluxes = _sw_gpoint_fluxes(igpt, sw_optical_props)
@@ -1009,7 +1034,7 @@ def solve_sw(
           igpt, gas_props, False, cloud_r_eff_liq, liq_g,
           cloud_r_eff_ice, ice_g, cloud_fraction=cloud_fraction, **snow_kw)
       _aer = (aerosol_optical_depth, aerosol_single_scattering_albedo,
-              aerosol_asymmetry_factor)
+              aerosol_asymmetry_factor, aerosol_volcanic_moments)
       sw_fluxes = _sw_gpoint_fluxes(
           igpt, _mix_sw_aerosol_and_clip(igpt, optics_lib, cloudy_props, *_aer))
       sw_fluxes_clr = _sw_gpoint_fluxes(

exec
/usr/bin/bash -lc 'cat tests/unit/test_volcanic_sw_optics.py' in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 0ms:
"""Volcanic stratospheric aerosol in the SHORTWAVE with the CMIP6 file's own
per-band extinction / single-scattering albedo / asymmetry.

- loader: bands sorted to RRTMGP order and checked against its band limits,
  moments (tau, tau*ssa, tau*ssa*g), refusals (no SSA/g, wrong bands, g<0);
- the band-limit table equals both RRTMGP SW gas-optics files;
- RRTMGP: moments carrying the grey ssa/g reproduce the grey path; the file's
  non-absorbing sulfate absorbs less than the grey 0.93; SW flux closure;
  jit parity; finite non-zero gradient;
- gray / zero radiation refuse the field.
"""

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import netCDF4
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

_REPO = Path(__file__).resolve().parents[2]
# RRTMG-SW / CMIP6-file band edges [um], in the file's own (E3SM) order.
_WL1 = np.array([3.07692, 2.5, 2.15054, 1.94175, 1.62602, 1.2987, 1.24224,
                 0.77821, 0.625, 0.441501, 0.344828, 0.263158, 0.2, 3.84615])
_WL2 = np.array([3.84615, 3.07692, 2.5, 2.15054, 1.94175, 1.62602, 1.2987,
                 1.24224, 0.77821, 0.625, 0.441501, 0.344828, 0.263158, 12.1951])
_LAT = np.linspace(-90.0, 90.0, 7)
_PROD_FILE = "/glade/work/pg2328/amip_inputs/volcanic_cmip6_from_cesm_1978-2014.nc"


def volc_sw_file(path, ext=0.01, omega=None, g=None, alt=(18.0, 20.0, 22.0, 24.0),
                 wl=(_WL1, _WL2), drop=()):
    """CMIP6-shaped volcanic file: ``ext_sun/omega_sun/g_sun(solar_bands,
    lat, altitude, month)``; per-band ``omega``/``g`` (file band order)."""
    nb = len(wl[0])
    omega = np.full(nb, 0.99) if omega is None else np.asarray(omega, float)
    g = np.full(nb, 0.7) if g is None else np.asarray(g, float)
    shape = (nb, _LAT.size, len(alt), 12)
    with netCDF4.Dataset(path, "w") as ds:
        for d, n in (("solar_bands", nb), ("lat", _LAT.size),
                     ("altitude", len(alt)), ("month", 12)):
            ds.createDimension(d, n)
        ds.createVariable("lat", "f8", ("lat",))[:] = _LAT
        ds.createVariable("altitude", "f8", ("altitude",))[:] = np.asarray(alt)
        dims = ("solar_bands", "lat", "altitude", "month")
        fields = {"ext_sun": np.broadcast_to(np.asarray(ext, float), shape),
                  "omega_sun": np.broadcast_to(omega[:, None, None, None], shape),
                  "g_sun": np.broadcast_to(g[:, None, None, None], shape)}
        for k, v in fields.items():
            if k not in drop:
                ds.createVariable(k, "f8", dims)[:] = v
        ds.createVariable("wl1_sun", "f8", ("solar_bands",))[:] = wl[0]
        ds.createVariable("wl2_sun", "f8", ("solar_bands",))[:] = wl[1]
    return str(path)


def _load(path):
    from legoesm.forcing.external import _load_volcanic_sw_optics_profile
    _load_volcanic_sw_optics_profile.cache_clear()
    return _load_volcanic_sw_optics_profile(path)


def test_loader_sorts_bands_and_builds_moments(tmp_path):
    omega = np.linspace(0.2, 1.0, 14)
    g = np.linspace(0.3, 0.8, 14)
    _, _, lat, mom, p_edges, band_wn = _load(
        volc_sw_file(tmp_path / "v.nc", omega=omega, g=g))
    assert mom.shape == (12, _LAT.size, 4, 3, 14) and mom.dtype == np.float32
    assert np.all(np.diff(p_edges) > 0)           # top first
    assert np.all(np.diff(band_wn[:, 0]) > 0)     # ascending wavenumber
    srt = np.argsort(1.0e4 / np.maximum(_WL1, _WL2))
    assert srt[0] == 13                           # 820-2600 cm-1 band first
    tau = 0.01 * 2.0                               # ext [1/km] * dz 2 km
    np.testing.assert_allclose(mom[0, 0, 0, 0], tau, rtol=1e-6)
    np.testing.assert_allclose(mom[0, 0, 0, 1], tau * omega[srt], rtol=1e-6)
    np.testing.assert_allclose(mom[0, 0, 0, 2], tau * omega[srt] * g[srt], rtol=1e-6)


@pytest.mark.parametrize("kw, match", [
    ({"drop": ("omega_sun",)}, "omega_sun"),
    ({"drop": ("g_sun",)}, "g_sun"),
    ({"g": np.full(14, -0.1)}, "Negative"),
    ({"wl": (_WL1[:13], _WL2[:13])}, "map 1:1"),
    ({"wl": (_WL1 * 1.05, _WL2 * 1.05)}, "map 1:1"),
])
def test_loader_refuses(tmp_path, kw, match):
    with pytest.raises(ValueError, match=match):
        _load(volc_sw_file(tmp_path / "v.nc", **kw))


def test_legacy_column_file_is_not_an_optics_source(tmp_path):
    p = str(tmp_path / "legacy.nc")
    with netCDF4.Dataset(p, "w") as ds:
        ds.createDimension("time", 12)
        ds.createDimension("lat", 3)
        ds.createVariable("lat", "f8", ("lat",))[:] = [-60.0, 0.0, 60.0]
        ds.createVariable("aod", "f8", ("time", "lat"))[:] = 0.01
    assert _load(p) is None


@pytest.mark.parametrize("table", ["rrtmgp-gas-sw-g112.nc", "rrtmgp-gas-sw-g224.nc"])
def test_band_limit_table_equals_rrtmgp_files(table):
    from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import (
        _default_data_path,
    )
    from legoesm.forcing.external import _RRTMGP_SW_BAND_LIMITS_CM1
    from legoesm.forcing.external import _open_forcing_dataset
    ds = _open_forcing_dataset(_default_data_path(table))
    lims = np.asarray(ds["bnd_limits_wavenumber"].values)
    ds.close()
    np.testing.assert_array_equal(lims, np.asarray(_RRTMGP_SW_BAND_LIMITS_CM1))


@pytest.mark.skipif(not os.path.exists(_PROD_FILE), reason="production file not staged")
def test_production_file_maps_and_sulfate_scatters():
    _, _, _, mom, _, band_wn = _load(_PROD_FILE)
    assert band_wn.shape == (14, 2)
    tau = mom[..., 0, :].astype(np.float64).sum(axis=2)   # (t, lat, band)
    ssa = mom[..., 1, :].astype(np.float64).sum(axis=2) / tau
    # Visible bands (ascending index 9-11 = 12850-29000 cm-1): non-absorbing.
    assert ssa[..., 9:12].min() > 0.999


def _columns(ncol=2, nlev=10):
    T = jnp.broadcast_to(jnp.linspace(210.0, 290.0, nlev)[None, :], (ncol, nlev))
    p_half = jnp.broadcast_to(jnp.geomspace(5.0e2, 1.0e5, nlev + 1)[None, :],
                              (ncol, nlev + 1))
    p_full = 0.5 * (p_half[:, 1:] + p_half[:, :-1])
    return dict(T=T, p_full=p_full, p_half=p_half,
                sfc_temperature=jnp.full((ncol,), 295.0),
                q_v=jnp.full((ncol, nlev), 2.0e-3),
                cos_zenith=jnp.array([0.6, 0.3])[:ncol],
                aerosol_optical_depth=jnp.full((ncol, nlev), 0.01))


def _volc_moments(kw, tau_b, ssa_b, g_b, layers=slice(1, 4)):
    ncol, nlev = kw["T"].shape
    tau = jnp.zeros((ncol, nlev, 14)).at[:, layers, :].set(jnp.asarray(tau_b))
    return jnp.stack([tau, tau * jnp.asarray(ssa_b),
                      tau * jnp.asarray(ssa_b) * jnp.asarray(g_b)], axis=2)


def _solver():
    from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
    from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP
    cfg = RRTMGPConfig()
    return RRTMGP.from_legoesm_config(cfg), cfg


def test_grey_moments_reproduce_grey_path():
    """Volcanic tau with the grey ssa/g == the same tau in aerosol_od."""
    solver, cfg = _solver()
    kw = _columns()
    volc = _volc_moments(kw, 0.05, cfg.aerosol_ssa, cfg.aerosol_g)
    a = solver.solve_columns(**kw, aerosol_volcanic_moments_sw=volc)
    kw_grey = {**kw, "aerosol_optical_depth":
               kw["aerosol_optical_depth"] + volc[:, :, 0, 0]}
    b = solver.solve_columns(**kw_grey)
    for f in ("sw_flux_up", "sw_flux_down", "sw_heating_rate"):
        np.testing.assert_allclose(getattr(a, f), getattr(b, f), rtol=1e-10,
                                   atol=1e-10)


def test_file_optics_change_sw_and_close_budget():
    """Non-absorbing volcanic sulfate (ssa 1) absorbs less SW and reflects
    more than the same optical depth at the grey ssa 0.93; both close the SW
    column budget (TOA net - surface net = sum of layer heating * cp dp / g)."""
    from legoesm.atmosphere.physics.radiation.rrtmgp import constants as rc
    solver, cfg = _solver()
    kw = _columns()
    grey = solver.solve_columns(**kw, aerosol_volcanic_moments_sw=_volc_moments(
        kw, 0.1, cfg.aerosol_ssa, cfg.aerosol_g))
    file_ = solver.solve_columns(**kw, aerosol_volcanic_moments_sw=_volc_moments(
        kw, 0.1, 1.0, 0.72))
    dp = kw["p_half"][:, 1:] - kw["p_half"][:, :-1]

    def absorbed(o):
        net = o.sw_flux_down - o.sw_flux_up
        return net[:, 0] - net[:, -1]

    for o in (grey, file_):
        col = jnp.sum(o.sw_heating_rate * dp, axis=1) * rc.CP_D / rc.G
        incident = o.sw_flux_down[:, 0]
        assert np.all(np.abs(np.asarray(absorbed(o) - col))
                      <= 1e-10 * np.asarray(incident) + 1e-10)
    assert np.all(np.asarray(absorbed(grey) - absorbed(file_)) > 0.5)
    assert np.all(np.asarray(file_.sw_flux_up[:, 0] - grey.sw_flux_up[:, 0]) > 0.5)


def test_jit_parity_and_gradient():
    solver, _ = _solver()
    kw = _columns()

    def toa_up(scale):
        volc = _volc_moments(kw, 0.08 * scale, 0.99, 0.7)
        return jnp.sum(solver.solve_columns(
            **kw, aerosol_volcanic_moments_sw=volc).sw_flux_up[:, 0])

    eager = toa_up(1.0)
    np.testing.assert_allclose(jax.jit(toa_up)(1.0), eager, rtol=1e-12)
    g = jax.grad(toa_up)(1.0)
    assert np.isfinite(g) and g > 0.0
    fd = (toa_up(1.0 + 1e-5) - toa_up(1.0 - 1e-5)) / 2e-5
    np.testing.assert_allclose(g, fd, rtol=1e-5)


def test_wrong_band_count_raises():
    solver, _ = _solver()
    kw = _columns()
    with pytest.raises(ValueError, match="trailing shape"):
        solver.solve_columns(**kw, aerosol_volcanic_moments_sw=jnp.zeros(
            kw["T"].shape + (3, 13)))


@pytest.mark.parametrize("builder", ["_build_gray_radiation_fn",
                                     "_build_none_radiation_fn"])
def test_non_rrtmgp_radiation_refuses(builder):
    from legoesm.driver import physics_pipeline as pp
    from legoesm.driver.config import ExperimentConfig
    fn, _ = getattr(pp, builder)(ExperimentConfig(radiation="gray"))
    kw = _columns()
    ncol, nlev = kw["T"].shape
    args = (kw["T"], kw["p_full"], kw["p_half"], kw["q_v"],
            kw["sfc_temperature"], jnp.zeros(ncol), jnp.zeros(ncol), 180.0,
            43200.0, jnp.full((ncol,), 0.1), jnp.ones(ncol),
            jnp.zeros((ncol, nlev)), jnp.zeros((ncol, nlev)), jnp.zeros(0))
    with pytest.raises(ValueError, match="volcanic SW"):
        fn(*args, aerosol_sw_volc_col=jnp.zeros((ncol, nlev, 3, 14)))


def test_probe_runs_on_synthetic_file(tmp_path, capsys):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "probe", _REPO / "scripts/validate/probe_volcanic_sw_optics.py")
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    probe.main(volc_sw_file(tmp_path / "v.nc", omega=np.full(14, 1.0)), 1979, 1)
    out = capsys.readouterr().out
    assert "file-grey" in out and "ssa 1.0000" in out


def test_band_mapping_matches_per_band_grey_path():
    """Absorbing volcanic aerosol in ONE band == the grey aerosol with that
    band's per-band ssa (RRTMGPConfig.aerosol_ssa_bands, an independent path
    through the same g-point -> band map): a wrong band order or a fixed band
    for every g-point fails."""
    from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP
    solver, cfg = _solver()
    kw = {**_columns(), "aerosol_optical_depth": jnp.zeros((2, 10))}
    for k in (0, 10):
        ssa = np.ones(14)
        ssa[k] = 0.4
        volc = _volc_moments(kw, 0.2, ssa, cfg.aerosol_g)
        a = solver.solve_columns(**kw, aerosol_volcanic_moments_sw=volc)
        ref = RRTMGP.from_legoesm_config(cfg._replace(
            aerosol_ssa_bands=tuple(ssa))).solve_columns(
            **{**kw, "aerosol_optical_depth": volc[:, :, 0, 0]})
        np.testing.assert_allclose(a.sw_flux_down, ref.sw_flux_down,
                                   rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(a.sw_flux_up, ref.sw_flux_up,
                                   rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("scheme", ["gray", "simple_lw"])
def test_integration_non_rrtmgp_refuses(scheme):
    from legoesm.atmosphere.physics.radiation import integration
    from legoesm.atmosphere.physics.radiation.config import RadiationConfig
    kw = _columns()
    with pytest.raises(ValueError, match="aerosol_sw_volc"):
        integration._call_radiation_backend(
            RadiationConfig(scheme=scheme), kw["T"], kw["p_full"], kw["p_half"],
            kw["sfc_temperature"], jnp.zeros(2), kw["q_v"], jnp.full(2, 400.0),
            aerosol_sw_volc=jnp.zeros((2, 10, 3, 14)))


def test_volcanic_chunking_is_exact(tmp_path, monkeypatch):
    from legoesm.forcing import surface_utils as su
    from legoesm.forcing.external import AerosolConfig
    p = volc_sw_file(tmp_path / "v.nc", omega=np.linspace(0.3, 1.0, 14))
    acfg = AerosolConfig(enabled=True, volcanic_enabled=True, volcanic_path=p,
                         use_reference_if_missing=True)
    ncol, nlev = 5, 12
    lat = jnp.radians(jnp.linspace(-70.0, 70.0, ncol))
    p_half = jnp.broadcast_to(jnp.geomspace(1.0e2, 1.0e5, nlev + 1), (ncol, nlev + 1))
    w = su.stratospheric_layer_weight(p_half, lat)
    args = (acfg, 15.0, lat, p_half, w)
    kw = dict(active=True, lw_active=False, dtype=jnp.float64)
    whole = su.aerosol_layer_fields(*args, **kw)[3]
    monkeypatch.setattr(su, "_VOLC_SW_COL_CHUNK", 2)
    np.testing.assert_array_equal(su.aerosol_layer_fields(*args, **kw)[3], whole)
    assert whole.shape == (ncol, nlev, 3, 14) and float(whole[..., 0, :].sum()) > 0

codex
The revised mixing and band-mapping tests address the reported r1 issues. I’m checking the remaining forcing paths and the loader/remap edge cases.
exec
/usr/bin/bash -lc "rg -n 'aerosol_layer_fields\\(|get_aerosol_sw_volcanic_at_time|aerosol_sw_volc|aerosol_lw_od' packages/ml/legoesm/training/scale_build.py packages/coupler/legoesm/driver/model_driver.py packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py" in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 0ms:
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:979:    aerosol_lw_od: jnp.ndarray | None = None,
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:980:    aerosol_sw_volc: jnp.ndarray | None = None,
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1042:    aerosol_lw_od : jnp.ndarray or None
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1049:    aerosol_sw_volc : jnp.ndarray or None
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1069:    if aerosol_sw_volc is not None and radiation_config.scheme != "rrtmgp":
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1071:            "aerosol_sw_volc (volcanic SW per-band optics) is wired for the "
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1234:        aerosol_absorption_optical_depth_lw=aerosol_lw_od,
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1238:    if aerosol_sw_volc is not None:
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1239:        _rad_kwargs["aerosol_volcanic_moments_sw"] = aerosol_sw_volc
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1827:            forcing.get("aerosol_lw_od") if forcing is not None else None
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1830:            forcing.get("aerosol_sw_volc") if forcing is not None else None
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:2152:            aerosol_lw_od=_aer_lw_ext,
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:2153:            aerosol_sw_volc=_aer_volc_ext,
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:2213:                aerosol_lw_od=_aer_lw_ext,
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:2214:                aerosol_sw_volc=_aer_volc_ext,
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:3284:            forcing.get("aerosol_lw_od") if forcing is not None else None
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:3287:            forcing.get("aerosol_sw_volc") if forcing is not None else None
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:3451:            aerosol_lw_od=_aer_lw_ext,
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:3452:            aerosol_sw_volc=_aer_volc_ext,
packages/ml/legoesm/training/scale_build.py:1582:    """``aerosol_od`` / ``aerosol_ccn_aod`` / ``aerosol_lw_od`` for one sample:
packages/ml/legoesm/training/scale_build.py:1607:    per_year = [aerosol_layer_fields(
packages/ml/legoesm/training/scale_build.py:1612:    out = {"aerosol_od": od, "aerosol_ccn_aod": ccn, "aerosol_lw_od": lw}
packages/ml/legoesm/training/scale_build.py:1615:        out["aerosol_sw_volc"] = jnp.mean(jnp.stack(volc), axis=0)
packages/ml/legoesm/training/scale_build.py:1858:            aerosol_sw_volc=getattr(driver, "_aerosol_sw_volc", None),
packages/coupler/legoesm/driver/model_driver.py:5850:        self._aerosol_lw_od = None
packages/coupler/legoesm/driver/model_driver.py:5851:        self._aerosol_sw_volc = None
packages/coupler/legoesm/driver/model_driver.py:6159:        (aerosol_od, _ccn_aod, self._aerosol_lw_od,
packages/coupler/legoesm/driver/model_driver.py:6160:         self._aerosol_sw_volc) = aerosol_layer_fields(
packages/coupler/legoesm/driver/model_driver.py:16479:                        # ``forcing.get("aerosol_lw_od")`` is None ⇒ RRTMGP
packages/coupler/legoesm/driver/model_driver.py:16481:                        _aer_lw = getattr(self, "_aerosol_lw_od", None)
packages/coupler/legoesm/driver/model_driver.py:16483:                            _forcing_daily["aerosol_lw_od"] = _aer_lw
packages/coupler/legoesm/driver/model_driver.py:16486:                        if getattr(self, "_aerosol_sw_volc", None) is not None:
packages/coupler/legoesm/driver/model_driver.py:16487:                            _forcing_daily["aerosol_sw_volc"] = self._aerosol_sw_volc
packages/coupler/legoesm/driver/model_driver.py:18469:                        _aer_lw = getattr(self, "_aerosol_lw_od", None)
packages/coupler/legoesm/driver/model_driver.py:18471:                            _ext_daily["aerosol_lw_od"] = _aer_lw
packages/coupler/legoesm/driver/model_driver.py:18472:                        if getattr(self, "_aerosol_sw_volc", None) is not None:
packages/coupler/legoesm/driver/model_driver.py:18473:                            _ext_daily["aerosol_sw_volc"] = self._aerosol_sw_volc
packages/coupler/legoesm/driver/model_driver.py:19446:                aerosol_lw_od=getattr(self, "_aerosol_lw_od", None),
packages/coupler/legoesm/driver/model_driver.py:19447:                aerosol_sw_volc=getattr(self, "_aerosol_sw_volc", None),
packages/coupler/legoesm/driver/model_driver.py:20287:            aerosol_sw_volc=getattr(self, "_aerosol_sw_volc", None),
packages/coupler/legoesm/driver/model_driver.py:20663:                aerosol_lw_od=getattr(self, "_aerosol_lw_od", None),
packages/coupler/legoesm/driver/model_driver.py:20664:                aerosol_sw_volc=getattr(self, "_aerosol_sw_volc", None),
packages/coupler/legoesm/driver/model_driver.py:21553:                aerosol_lw_od=getattr(self, "_aerosol_lw_od", None),
packages/coupler/legoesm/driver/model_driver.py:21554:                aerosol_sw_volc=getattr(self, "_aerosol_sw_volc", None),
packages/coupler/legoesm/driver/model_driver.py:21699:                    aerosol_lw_od=getattr(self, "_aerosol_lw_od", None),
packages/coupler/legoesm/driver/model_driver.py:21700:                    aerosol_sw_volc=getattr(self, "_aerosol_sw_volc", None),

exec
/usr/bin/bash -lc "sed -n '2450,2535p' packages/tools/legoesm/forcing/external.py && rg -n 'def _interp_zonal_to_grid|def place_stratospheric|def expand_kwargs|PER_COLUMN_SOLVER_KEYS|def _add_halos' packages/tools/legoesm/forcing/external.py packages/tools/legoesm/forcing/surface_utils.py packages/atmosphere/legoesm/atmosphere/physics/clouds/subcolumns.py packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py && cat scripts/validate/probe_volcanic_sw_optics.py" in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 0ms:
                                           lat_grid: jnp.ndarray):
    """SHORTWAVE volcanic aerosol with its OWN per-band optics from the file.

    Returns ``(moments_col, p_edges, band_wn)``: ``moments_col``
    ``(ncol, nsrc, 3, nband)`` per-file-layer ``(tau, tau*omega,
    tau*omega*g)`` (see :func:`_load_volcanic_sw_optics_profile`), ``p_edges``
    ``(nsrc+1,)`` [Pa] ascending, ``band_wn`` ``(nband, 2)`` [cm-1].  None
    when volcanic forcing is off or the file carries no ``omega_sun`` /
    ``g_sun`` / band edges (the caller then keeps the gray path).
    """
    if not config.volcanic_enabled or not config.volcanic_path:
        return None
    loaded = _load_volcanic_sw_optics_profile(config.volcanic_path)
    if loaded is None:
        return None
    out = _volcanic_profile_at_time(config, day, lat_grid, "sw_optics",
                                    loaded=loaded[:4] + (loaded[4],))
    return out[0], out[1], loaded[5]


def _volcanic_profile_at_time(config, day, lat_grid, kind, loaded=None):
    """Shared body of the LW / SW volcanic profile samplers.  Trailing axes
    after the layer axis (the SW-optics moment/band axes) are carried along."""
    if loaded is not None:
        mid_days_v, first_date_v, lat_v, prof_v, p_edges = loaded
    else:
        try:
            mid_days_v, first_date_v, lat_v, prof_v, p_edges = (
                _load_volcanic_profile(config.volcanic_path, kind)
            )
        except ValueError:
            # File has no ext_earth / ext_sun band — treat as "no source".
            return None

    # Interpolate the (ntime, nlat, nlayer) profile in time -> (nlat, nlayer).
    if len(mid_days_v) > 12:
        file_day_v = _simday_to_file_day(day, config.start_year, first_date_v)
        prof_t = _interp_monthly_noncyclic(
            mid_days_v, prof_v, file_day_v, config.volcanic_path,
            config.hold_outside_file_years)
    else:
        mid_anchored_v, prof_anchored_v = _cyclic_phase_anchor(
            mid_days_v, prof_v, first_date_v,
        )
        prof_t = _interp_monthly_cyclic(mid_anchored_v, prof_anchored_v, day)
    prof_t = prof_t * config.volcanic_scale  # (nlat, nlayer)

    if lat_grid is not None:
        # (nlat, nlayer) -> (*lat_grid.shape, nlayer); the driver hands a
        # raveled column-latitude grid, so flatten leading dims to
        # (ncol, nlayer).  The nlayer axis is the vertical PROFILE and is
        # intentionally NOT summed away (unlike the gray SW/column path).
        prof_col = _interp_zonal_to_grid(lat_v, prof_t, lat_grid)
        prof_col = jnp.reshape(
            jnp.clip(prof_col, 0.0, None), (-1, *prof_t.shape[1:]),
        )
        return prof_col, jnp.asarray(p_edges)

    return {
        "lat": lat_v,
        "aod_profile": np.clip(prof_t, 0.0, None),
        "p_edges": np.asarray(p_edges),
    }


# ==============================================================================
# Solar irradiance
# ==============================================================================

class SolarConfig(NamedTuple):
    """Solar irradiance configuration.

    The seasonal cycle is handled by the radiation scheme's insolation
    calculation. This config adds support for TSI variations (e.g.,
    solar cycle, volcanic dimming).

    Constant TSI is active by default. Time-varying TSI from file is
    supported via source="file".

    Fields
    ------
    S_0 : float
        Total solar irradiance [W/m2]. Default: 1360.
    source : str
        "constant", "file", or "spectral_file".
    path : str
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:136:def _add_halos(f_3d):
packages/tools/legoesm/forcing/surface_utils.py:494:def place_stratospheric_aod_profile_to_layers(
packages/atmosphere/legoesm/atmosphere/physics/clouds/subcolumns.py:412:PER_COLUMN_SOLVER_KEYS = frozenset({
packages/atmosphere/legoesm/atmosphere/physics/clouds/subcolumns.py:421:def expand_kwargs(kwargs: dict, n_sub: int, ncol: int,
packages/atmosphere/legoesm/atmosphere/physics/clouds/subcolumns.py:422:                  keys=PER_COLUMN_SOLVER_KEYS) -> dict:
packages/tools/legoesm/forcing/external.py:1600:def _interp_zonal_to_grid(lat_src: np.ndarray, field: np.ndarray,
"""Probe: SW effect of giving CMIP6 volcanic aerosol its own per-band optics.

One clear-sky equatorial column (CAM6 L32, p_s 1000 hPa, US-Std-like T,
no tropospheric aerosol), mid-month of a given year/month, RRTMGP full
tables.  Arms, same column / same placement / same tropopause weight:
  grey: OLD path -- ext_sun collapsed solar-weighted (5778 K Planck) to one
        gray tau, solved with the grey aerosol ssa/g (RRTMGPConfig 0.93/0.70);
  file: NEW path -- per-band (tau, tau*ssa, tau*ssa*g) from the file.
Prints TOA upward SW, surface downward SW and SW absorbed above 100 hPa
[W/m2] at cos(zenith) = 0.5 and the daily-mean-equivalent (x insolation
weight not applied; instantaneous).  Usage:
  python scripts/validate/probe_volcanic_sw_optics.py FILE YEAR MONTH
"""
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def main(path, year, month):
    from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
    from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import (
        RRTMGP, gas_table_files)
    from legoesm.forcing.external import (
        AerosolConfig, _SOLAR_PLANCK_TEMP_K, _planck_band_weights,
        get_aerosol_sw_volcanic_optics_at_time)
    from legoesm.forcing.surface_utils import (
        place_stratospheric_aod_profile_to_layers, stratospheric_layer_weight)
    from legoesm.grids.vertical import make_cam6_l32_levels

    sig = make_cam6_l32_levels()
    p_s = jnp.array([1.0e5])
    p_half = jnp.asarray(sig.pressure_at_half(p_s)).reshape(1, -1)
    p_full = jnp.asarray(sig.pressure_at_full(p_s)).reshape(1, -1)
    T = jnp.maximum(300.0 * (p_full / 1.0e5) ** 0.19, 200.0)
    q_v = jnp.clip(0.018 * (p_full / 1.0e5) ** 3, 3.0e-6, None)
    lat = jnp.array([0.0])
    day = 365.0 * (year - 1979) + 30.4 * (month - 1) + 15.0
    acfg = AerosolConfig(enabled=True, volcanic_enabled=True,
                         volcanic_path=path, start_year=1979)
    mom, p_edges, band_wn = get_aerosol_sw_volcanic_optics_at_time(acfg, day, lat)
    w = stratospheric_layer_weight(p_half, lat, p_full, T)
    nsrc = mom.shape[1]
    flat = jnp.moveaxis(jnp.asarray(mom, jnp.float64).reshape(1, nsrc, 42), -1, 0)
    placed = jax.vmap(place_stratospheric_aod_profile_to_layers,
                      in_axes=(0, None, None))(flat, jnp.asarray(p_edges), p_half) * w
    volc = jnp.moveaxis(placed, 0, -1).reshape(1, -1, 3, 14)
    wl = 1.0e4 / band_wn                       # cm-1 -> um, (14, 2)
    wts = _planck_band_weights(wl[:, 1], wl[:, 0], _SOLAR_PLANCK_TEMP_K)
    tau_gray = jnp.tensordot(volc[:, :, 0, :], jnp.asarray(wts / wts.sum()),
                             axes=(2, 0))

    lw, sw = gas_table_files("full")
    cfg = RRTMGPConfig(lw_gas_file=lw, sw_gas_file=sw)
    solver = RRTMGP.from_legoesm_config(cfg)
    kw = dict(T=T, p_full=p_full, p_half=p_half,
              sfc_temperature=jnp.array([300.0]), q_v=q_v,
              cos_zenith=jnp.array([0.5]), sfc_albedo=jnp.array([0.07]))
    arms = {
        "none": solver.solve_columns(**kw),
        "grey": solver.solve_columns(**kw, aerosol_optical_depth=tau_gray),
        "file": solver.solve_columns(**kw, aerosol_optical_depth=jnp.zeros_like(tau_gray),
                                     aerosol_volcanic_moments_sw=volc),
    }
    k100 = int(jnp.argmin(jnp.abs(p_half[0] - 1.0e4)))
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    print(f"file={path} {year}-{month:02d} lat=0 cosz=0.5 code={sha}")
    tau_vis = float(volc[0, :, 0, 10].sum())
    ssa_vis = float(volc[0, :, 1, 10].sum() / max(tau_vis, 1e-30))
    print(f"volcanic column tau 16000-22650 cm-1 {tau_vis:.4f} ssa {ssa_vis:.4f};"
          f" gray tau {float(tau_gray.sum()):.4f}")
    for k, o in arms.items():
        net = o.sw_flux_down[0] - o.sw_flux_up[0]
        print(f"{k:5s} toa_up {float(o.sw_flux_up[0, 0]):8.3f}  sfc_down "
              f"{float(o.sw_flux_down[0, -1]):8.3f}  abs_above_100hPa "
              f"{float(net[0] - net[k100]):7.3f}  abs_column "
              f"{float(net[0] - net[-1]):8.3f}")
    g, f = arms["grey"], arms["file"]
    print(f"file-grey: toa_up {float(f.sw_flux_up[0,0]-g.sw_flux_up[0,0]):+.3f} "
          f"sfc_down {float(f.sw_flux_down[0,-1]-g.sw_flux_down[0,-1]):+.3f} W/m2")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))

exec
/usr/bin/bash -lc 'JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/unit/test_volcanic_sw_optics.py' in /glade/derecho/scratch/pg2328/wt_volc
 exited 127 in 0ms:
/usr/bin/bash: .venv/bin/python: No such file or directory

exec
/usr/bin/bash -lc "sed -n '470,600p' packages/tools/legoesm/forcing/surface_utils.py && sed -n '1580,1645p' packages/tools/legoesm/forcing/external.py && sed -n '410,465p' packages/atmosphere/legoesm/atmosphere/physics/clouds/subcolumns.py && sed -n '125,165p' packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py && command -v python && ls /glade/u/home/pg2328/legoesm/.venv/bin/python" in /glade/derecho/scratch/pg2328/wt_volc
 exited 2 in 0ms:

def distribute_column_aod_to_layers(
    aod_col: jnp.ndarray,
    p_half_col: jnp.ndarray,
) -> jnp.ndarray:
    """Distribute column aerosol optical depth to layers by pressure thickness.

    Parameters
    ----------
    aod_col : array, shape (ncol,)
        Total column AOD per column.
    p_half_col : array, shape (ncol, nlev+1)
        Half-level pressures.

    Returns
    -------
    aod_layers : array, shape (ncol, nlev)
        Layer-distributed AOD.
    """
    dp = jnp.clip(p_half_col[:, 1:] - p_half_col[:, :-1], 1.0e-12, None)
    w = dp / jnp.sum(dp, axis=1, keepdims=True)
    return jnp.clip(aod_col, 0.0, None)[:, None] * w


def place_stratospheric_aod_profile_to_layers(
    aod_profile: jnp.ndarray,
    p_edges: jnp.ndarray,
    p_half_col: jnp.ndarray,
) -> jnp.ndarray:
    """Conservatively remap a fixed-pressure-edge AOD profile onto model layers.

    Volcanic stratospheric aerosol is supplied as a per-layer absorption
    optical depth on the forcing file's OWN (altitude -> pressure) grid.  This
    bins it onto the model's pressure layers by fractional pressure overlap,
    so the aerosol lands at its true stratospheric pressure -- unlike
    :func:`distribute_column_aod_to_layers`, which spreads a single column AOD
    by full-column pressure mass and thus dumps ~90 % of a stratospheric layer
    into the troposphere.

    Method: the cumulative-OD curve is sampled at each model half level and the
    per-layer OD is the difference of the cumulative at the layer's two edges.
    The cumulative is interpolated in LOG-pressure, because the source OD is
    ``ext*dz`` -- uniform in geometric height (~ log-pressure) within a source
    layer, NOT uniform in pressure; a linear-pressure CDF would misallocate ~1 %
    of a 0.5 km source layer's OD across a cutting model interface (Codex review
    iter-1).  The remap is exact at source-layer edges and conserves the total
    column OD that falls within the model's pressure range (OD above the model
    top or below the surface edge is dropped -- it cannot be represented).

    The op set (searchsorted / gather / clip / min / max) is piecewise-linear:
    a.e.-differentiable in both ``aod_profile`` and ``p_half_col`` with
    well-defined one-sided gradients (kinks at source/model edges), and
    JIT/vmap-safe.

    Sign/units: optical depth is >= 0 and additive; pressure [Pa].

    Parameters
    ----------
    aod_profile : array, shape (ncol, nsrc)
        Per-source-layer absorption optical depth [-] (>= 0).
    p_edges : array, shape (nsrc+1,)
        Source-layer pressure edges [Pa], ASCENDING (index 0 = top / lowest
        pressure), aligned so ``aod_profile[:, j]`` occupies
        ``[p_edges[j], p_edges[j+1]]``.
    p_half_col : array, shape (ncol, nlev+1)
        Model half-level pressures [Pa] (index 0 = top).

    Returns
    -------
    aod_layers : array, shape (ncol, nlev)
        Layer absorption optical depth [-].
    """
    pe = jnp.asarray(p_edges)                                   # (nsrc+1,)
    aod = jnp.clip(jnp.asarray(aod_profile), 0.0, None)         # (ncol, nsrc)
    # Cumulative OD from the top: cum[:, k] = OD in source layers 0..k-1, so
    # cum aligns to the nsrc+1 source edges ``pe``.
    cum = jnp.concatenate(
        [jnp.zeros((aod.shape[0], 1), aod.dtype), jnp.cumsum(aod, axis=1)],
        axis=1,
    )                                                          # (ncol, nsrc+1)
    # Interpolate the CDF in log-pressure (source OD is uniform in geometric
    # height ~ log-pressure within a layer).
    l_edges = jnp.log(jnp.clip(pe, 1.0e-12, None))             # (nsrc+1,)

    ph = jnp.asarray(p_half_col)
    p_lo = jnp.minimum(ph[:, :-1], ph[:, 1:])                  # (ncol, nlev)
    p_hi = jnp.maximum(ph[:, :-1], ph[:, 1:])

    def _cdf_at(p):
        # Per-column log-pressure interp of the shared-edge cumulative at
        # pressures ``p``; clamps to [0, total] outside [pe[0], pe[-1]].
        lp = jnp.log(jnp.clip(p, 1.0e-12, None))
        idx = jnp.clip(jnp.searchsorted(l_edges, lp) - 1, 0, pe.shape[0] - 2)
        l0 = l_edges[idx]
        l1 = l_edges[idx + 1]
        frac = jnp.clip((lp - l0) / jnp.clip(l1 - l0, 1.0e-12, None), 0.0, 1.0)
        c0 = jnp.take_along_axis(cum, idx, axis=1)
        c1 = jnp.take_along_axis(cum, idx + 1, axis=1)
        return c0 + frac * (c1 - c0)

    return jnp.clip(_cdf_at(p_hi) - _cdf_at(p_lo), 0.0, None)


# --- prognostic sea-ice skin temperature (Semtner 1976 zero-layer + thermal
#     inertia; conductivity Untersteiner 1961 via constants.k_ice_default) ---

# Numerics floor for the skin update [K]: below the coldest observed polar
# surface (~185 K is the terrestrial record, Antarctic plateau) — a guard
# against a pathological flux spike, never active in normal operation.
_ICE_SKIN_FLOOR_K = 185.0


def prognostic_ice_skin_temperature(
    T_skin: jnp.ndarray,
    F_net_down_W_m2: jnp.ndarray,
    sic: jnp.ndarray,
    dt_s: float,
    h_ice_m: float,
    T_freeze_K: float = constants.T_freeze_ocean,
    k_ice_W_m_K: float = constants.k_ice_default,
    T_melt_surface_K: float = constants.T_freeze,
    melt_cap_width_K: float = 0.0,
) -> jnp.ndarray:
    """Advance a prescribed-ice AMIP skin temperature one integration step.

    Semtner (1976) zero-layer thermodynamics — conductive flux through a
    climatological ice slab of thickness ``h_ice_m`` toward the seawater
    freezing point ``T_freeze_K`` at the BASE (ice/ocean interface) — plus
    the slab's half thermal inertia (``C = rho_ice * c_pi * h/2``), which
    turns the diagnostic zero-layer balance into a stably integrable
    prognostic skin:
        else:
            sim_epoch = cftime.DatetimeGregorian(int(start_year), 1, 1)
        if hasattr(first_date, "year") and hasattr(first_date, "month"):
            delta = sim_epoch - first_date
            delta_days = delta.days + delta.seconds / 86400.0
            return delta_days + sim_day
    except Exception:
        pass
    # numpy.datetime64 fallback path: take scalar difference in days
    # (reached only if the by-calendar-date branch above raised).
    try:
        anchor = np.datetime64(f"{int(start_year):04d}-01-01")
        delta_days = float(
            (anchor - np.datetime64(first_date)) / np.timedelta64(1, "D"),
        )
        return delta_days + sim_day
    except Exception:
        return sim_day


def _interp_zonal_to_grid(lat_src: np.ndarray, field: np.ndarray,
                           lat_grid: jnp.ndarray) -> jnp.ndarray:
    """Interpolate a zonal-mean field to model grid latitudes.

    Parameters
    ----------
    lat_src : (nlat_src,)
        Source latitudes [degrees].  May be ascending (``-90 → +90``)
        or descending (``+89.5 → -89.5``, as in Kinne aerosol files);
        this helper detects a descending axis and flips both ``lat_src``
        and ``field`` along the leading axis before calling
        :func:`numpy.interp` (which requires strictly ascending ``xp``).
        Without this guard ``np.interp`` would silently extrapolate to
        the endpoint and produce a near-constant field at every model
        latitude, not a NaN / error (issue #207 bug 6).
    field : (nlat_src,) or (nlat_src, nlev) or (nlat_src, nlev, nband)
        Zonal-mean field at source latitudes.  Handles arbitrary trailing
        dimensions (e.g., 3D Kinne aerosol files with a band axis, #178).
    lat_grid : jax array, any shape
        Model grid latitudes [radians]. Will be converted to degrees.

    Returns
    -------
    jax array with shape (*lat_grid.shape,) or (*lat_grid.shape, *trailing)
    """
    lat_deg = np.asarray(jnp.degrees(lat_grid)).ravel()

    lat_src = np.asarray(lat_src)
    if lat_src.size > 1 and lat_src[0] > lat_src[-1]:
        # Descending source axis (e.g. Kinne aerosol files): flip so
        # ``np.interp``'s strictly-ascending-xp contract is satisfied.
        lat_src = lat_src[::-1]
        field = field[::-1]

    if field.ndim == 1:
        result = np.interp(lat_deg, lat_src, field)
        return jnp.array(result).reshape(lat_grid.shape)

    # Vectorised lerp: locate the bracketing source latitude once, then
    # gather both endpoints and combine.  Replaces the previous
    # ``for k in range(n_cols): np.interp(...)`` loop, which scaled
    # quadratically on T170+ (n_cols ~ 10⁴) at every external-forcing
    # update.
    nsrc = lat_src.shape[0]
    if nsrc == 1:
        result_flat = np.broadcast_to(
#: mis-expand ``solar_spectral_fraction`` -- that is ``(ngpt_sw,)``, a G-POINT
#: array of order 112, so a 112-column case would have corrupted it.
PER_COLUMN_SOLVER_KEYS = frozenset({
    "T", "p_full", "p_half", "sfc_temperature", "q_v", "cos_zenith",
    "sfc_albedo", "sfc_albedo_direct", "sfc_emissivity", "o3_vmr", "o3_top_vmr",
    "aerosol_optical_depth", "aerosol_absorption_optical_depth_lw",
    "aerosol_volcanic_moments_sw", "cloud_path_liq", "cloud_path_ice", "cloud_r_eff_liq", "cloud_r_eff_ice",
    "cloud_fraction",
})


def expand_kwargs(kwargs: dict, n_sub: int, ncol: int,
                  keys=PER_COLUMN_SOLVER_KEYS) -> dict:
    """Replicate the PER-COLUMN entries of a solver kwargs dict to ``n_sub*ncol``.

    Only keys in ``keys`` are touched, and only when they are arrays whose
    leading dimension is ``ncol`` (a scalar ``sfc_albedo`` is left alone).
    Everything else -- ``None``, scalars, g-point arrays, GHG dicts -- passes
    through unchanged.

    A key in ``keys`` whose array has a leading dimension OTHER than ``ncol``
    is a hard error rather than a silent pass-through: that means the caller's
    layout changed and the subcolumn expansion would be wrong.
    """
    out = {}
    for k, v in kwargs.items():
        if k not in keys or v is None or isinstance(v, (int, float, bool, str)):
            out[k] = v
            continue
        a = jnp.asarray(v)
        if a.ndim == 0:
            out[k] = v
            continue
        if a.shape[0] != ncol:
            raise ValueError(
                f"per-column solver kwarg {k!r} has leading dimension "
                f"{a.shape[0]}, expected ncol={ncol}; the subcolumn expansion "
                "cannot be applied safely"
            )
        out[k] = jnp.reshape(
            jnp.broadcast_to(a[None, ...], (n_sub,) + a.shape),
            (n_sub * ncol,) + a.shape[1:])
    return out


def average_output(result, n_sub: int, ncol: int):
    """Average every per-column field of a ``RadiationOutput`` over subcolumns.

    Fluxes and heating rates are averaged because the independent-column
    approximation IS the subcolumn-weighted mean of independent solutions.
    ``toa_insolation`` is identical across subcolumns, so averaging is a no-op
    on it rather than a special case.
    """
    return type(result)(*[
        None if f is None else average_over_subcolumns(f, n_sub, ncol)
        for f in result
    return GAS_TABLE_FILES[tables]
_DEFAULT_SW_CLOUD = _default_data_path("rrtmgp-clouds-sw-bnd.nc")

# Module-level optics cache (shared across RRTMGP instances)
_legoesm_optics_cache: dict = {}


# ---------------------------------------------------------------------------
# Helper functions for the legoESM column solver
# ---------------------------------------------------------------------------

def _add_halos(f_3d):
    """Add 1-cell vertical halos via linear extrapolation.

    Input shape (ncol, 1, nlev), output shape (ncol, 1, nlev+2).
    """
    bottom_halo = 2 * f_3d[:, :, 0:1] - f_3d[:, :, 1:2]
    top_halo = 2 * f_3d[:, :, -1:] - f_3d[:, :, -2:-1]
    return jnp.concatenate([bottom_halo, f_3d, top_halo], axis=2)


def cam_interface_temperature(T, p_full, p_half, sfc_temperature):
    """Longwave level temperatures, CAM6 host recipe (TOA-first).

    Port of CAM cam_cesm2_1_rel_60 ``rrtmg_state.F90`` (rrtmg_state_create,
    lines 113-119): the top interface takes the top-layer temperature, the
    surface interface takes the surface temperature, and each interior
    interface k (between layers k-1 above and k below) is

        tint(k) = t(k) - dy * (t(k) - t(k-1)),
        dy = (ln pint(k) - ln pmid(k)) / (ln pmid(k-1) - ln pmid(k)).

    CAM's surface value is (lwup/stebol)**0.25; here it is the
    ``sfc_temperature`` the radiation is handed (equal for a surface whose
    upward LW is stebol*T**4).

    Args:
      T: ``(ncol, nlev)`` layer temperature [K], index 0 = model top.
      p_full: ``(ncol, nlev)`` layer-midpoint pressure [Pa].
      p_half: ``(ncol, nlev+1)`` interface pressure [Pa].
      sfc_temperature: ``(ncol,)`` surface temperature [K].
/glade/u/apps/derecho/25.10/opt/view/bin/python
ls: cannot access '/glade/u/home/pg2328/legoesm/.venv/bin/python': No such file or directory

exec
/usr/bin/bash -lc "git diff HEAD -- tests/unit/test_aerosol_band_ccn.py tests/unit/test_cross_lane_forcing.py tests/unit/test_wb_amip_deck_params.py && rg -n 'aerosol_layer_fields\\(|get_aerosol_sw_volcanic_at_time' --glob '*.py' && rg -n '"'^##|''^# '"' CLAUDE.md" in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 5223ms:
diff --git a/tests/unit/test_aerosol_band_ccn.py b/tests/unit/test_aerosol_band_ccn.py
index f7d141cac1..58cd451d91 100644
--- a/tests/unit/test_aerosol_band_ccn.py
+++ b/tests/unit/test_aerosol_band_ccn.py
@@ -2,8 +2,9 @@
 
 - solar-weighted gray / 550 nm band from the file's own ``wl1_sun``/``wl2_sun``
   (order-independent; no edges -> old flat mean);
-- volcanic SW (``ext_sun``) placed at its stratospheric pressure, not spread
-  through the troposphere, and excluded from ``aerosol_ccn_aod``;
+- volcanic SW (``ext_sun`` + own optics) placed at its stratospheric pressure,
+  not spread through the troposphere, kept out of the grey ``aerosol_od`` and
+  excluded from ``aerosol_ccn_aod``;
 - ``aerosol_ccn_aod`` drives the AOD->CCN proxy when present, byte-identical
   sum(aerosol_od) when absent;
 - ``--aerosol-ccn-file`` -> ExperimentConfig -> AerosolConfig.ccn_path -> CCN.
@@ -59,16 +60,10 @@ def _kinne_file(path, band_aod, edges=True, lon=False):
 
 
 def _volc_file(path, ext=0.01, alt=(18.0, 20.0, 22.0, 24.0)):
-    """CMIP6-shaped ``ext_sun(solar_bands, lat, altitude, month)`` [1/km]."""
-    with netCDF4.Dataset(path, "w") as ds:
-        for d, n in (("solar_bands", 3), ("lat", _LAT.size), ("altitude", len(alt)), ("month", 12)):
-            ds.createDimension(d, n)
-        ds.createVariable("lat", "f8", ("lat",))[:] = _LAT
-        ds.createVariable("altitude", "f8", ("altitude",))[:] = np.asarray(alt)
-        ds.createVariable("ext_sun", "f8", ("solar_bands", "lat", "altitude", "month"))[:] = ext
-        ds.createVariable("wl1_sun", "f8", ("solar_bands",))[:] = _WL1
-        ds.createVariable("wl2_sun", "f8", ("solar_bands",))[:] = _WL2
-    return str(path)
+    """CMIP6-shaped volcanic file with its own solar-band optics (14 RRTMG
+    bands, see ``test_volcanic_sw_optics.volc_sw_file``)."""
+    from tests.unit.test_volcanic_sw_optics import volc_sw_file
+    return volc_sw_file(path, ext=ext, alt=alt)
 
 
 def test_solar_weighted_and_visible_band_collapse(tmp_path):
@@ -139,9 +134,14 @@ def test_volcanic_sw_above_tropopause_and_out_of_ccn(tmp_path):
     volc = _volc_file(tmp_path / "v.nc")  # 17-25 km, column ~0.08
     f0, aer0, _ = _driver_forcing(_cfg(aerosol_file=trop))
     f1, aer1, p_full = _driver_forcing(_cfg(aerosol_file=trop, volcanic_aerosol_file=volc))
-    dv = aer1 - aer0
-    np.testing.assert_allclose(dv.sum(-1), 0.01 * 8.0, rtol=1e-6)  # all of it lands
-    assert np.abs(dv[p_full > 3.0e4]).max() < 1e-12  # none below 300 hPa
+    assert f0._aerosol_sw_volc is None
+    # Volcanic SW is carried with its own optics, NOT in the grey field.
+    np.testing.assert_array_equal(aer1, aer0)
+    mom = np.asarray(f1._aerosol_sw_volc)            # (ncol, nlev, 3, 14)
+    assert mom.shape == aer0.shape + (3, 14)
+    np.testing.assert_allclose(mom[..., 0, :].sum(1), 0.01 * 8.0, rtol=1e-6)  # all lands
+    np.testing.assert_allclose(mom[..., 1, :], 0.99 * mom[..., 0, :], rtol=1e-6)
+    assert np.abs(mom[p_full > 3.0e4]).max() < 1e-12  # none below 300 hPa
     # CCN: tropospheric 550 nm column, volcanic excluded.
     np.testing.assert_allclose(np.asarray(f1._aerosol_ccn_aod), 0.20, rtol=1e-12)
     np.testing.assert_array_equal(np.asarray(f1._aerosol_ccn_aod),
@@ -268,19 +268,23 @@ def test_driver_volcanic_cut_at_state_tropopause(tmp_path):
     volc = _volc_file(tmp_path / "v.nc")
     trop = _kinne_file(tmp_path / "t.nc", [0.05, 0.20, 0.40], lon=True)
     cfg = _cfg(aerosol_file=trop, volcanic_aerosol_file=volc)
-    _, aer0, _ = _driver_forcing(_cfg(aerosol_file=trop))
-    _, aer_c, _ = _driver_forcing(cfg)
-    _, aer_w, p_full = _driver_forcing(
+
+    def vtau(f):  # per-layer volcanic tau of one band (all bands equal here)
+        return np.asarray(f._aerosol_sw_volc)[..., 0, 0]
+
+    f_c, _, _ = _driver_forcing(cfg)
+    f_w, _, p_full = _driver_forcing(
         cfg, temp_fn=lambda p: _std_temp(p, z_tp_m=19.0e3, t0=300.0)[0])
-    np.testing.assert_allclose((aer_c - aer0).sum(-1), 0.01 * 8.0, rtol=1e-6)
-    dv = aer_w - aer0
+    np.testing.assert_allclose(vtau(f_c).sum(-1), 0.01 * 8.0, rtol=1e-6)
+    dv = vtau(f_w)
     assert (dv.sum(-1) < 0.99 * 0.08).all() and (dv.sum(-1) > 0.0).all()
     p_tp = _std_temp(1.0e5, z_tp_m=19.0e3, t0=300.0)[1]
     assert np.abs(dv[p_full > 2.0 * p_tp]).max() < 1e-12
     # Explicit ``temp=`` (spectral caller) == state T; off-grid shape -> climatology.
     tp = _std_temp(jnp.asarray(p_full), z_tp_m=19.0e3, t0=300.0)[0]
-    np.testing.assert_array_equal(_driver_forcing(cfg, temp=tp)[1], aer_w)
-    np.testing.assert_array_equal(_driver_forcing(cfg, temp=tp[:, :-1])[1], aer_c)
+    np.testing.assert_array_equal(vtau(_driver_forcing(cfg, temp=tp)[0]), dv)
+    np.testing.assert_array_equal(
+        vtau(_driver_forcing(cfg, temp=tp[:, :-1])[0]), vtau(f_c))
 
 
 def test_owned_temp_selects_owned_faces():
diff --git a/tests/unit/test_cross_lane_forcing.py b/tests/unit/test_cross_lane_forcing.py
index d436c2a65b..5fd911e129 100644
--- a/tests/unit/test_cross_lane_forcing.py
+++ b/tests/unit/test_cross_lane_forcing.py
@@ -75,7 +75,7 @@ def _patch_readers(monkeypatch):
         assert np.shape(lon_grid) == np.shape(lat_grid)
         return _aod(np.asarray(lat_grid), day, ccn)
     monkeypatch.setattr(ext, "get_aerosol_at_time", _aer)
-    monkeypatch.setattr(ext, "get_aerosol_sw_volcanic_at_time", lambda c, day, lat_grid=None: None)
+    monkeypatch.setattr(ext, "get_aerosol_sw_volcanic_optics_at_time", lambda c, day, lat_grid=None: None)
     monkeypatch.setattr(ext, "get_aerosol_lw_at_time", lambda c, day, lat_grid=None: (
         np.full((np.size(lat_grid), 2), 0.01), np.array([2000.0, 3000.0, 4000.0])))
     monkeypatch.setattr(ext, "get_ghg_at_time", lambda c, day: dict(GHG))
diff --git a/tests/unit/test_wb_amip_deck_params.py b/tests/unit/test_wb_amip_deck_params.py
index 5d4822e2db..be92dbd847 100644
--- a/tests/unit/test_wb_amip_deck_params.py
+++ b/tests/unit/test_wb_amip_deck_params.py
@@ -409,7 +409,7 @@ def test_aerosol_climatology_averages_the_years_on_the_noleap_clock(
         return np.full(np.shape(lat_grid), 1e-3 * (c.start_year - 2000)
                        + (0.5 if band == "vis" else 0.0))
     monkeypatch.setattr(ext, "get_aerosol_at_time", _aod)
-    monkeypatch.setattr(ext, "get_aerosol_sw_volcanic_at_time",
+    monkeypatch.setattr(ext, "get_aerosol_sw_volcanic_optics_at_time",
                         lambda c, day, lat_grid=None: None)
     monkeypatch.setattr(ext, "get_aerosol_lw_at_time",
                         lambda c, day, lat_grid=None: None)
tests/unit/test_volcanic_sw_optics.py:291:    whole = su.aerosol_layer_fields(*args, **kw)[3]
tests/unit/test_volcanic_sw_optics.py:293:    np.testing.assert_array_equal(su.aerosol_layer_fields(*args, **kw)[3], whole)
packages/coupler/legoesm/driver/model_driver.py:6160:         self._aerosol_sw_volc) = aerosol_layer_fields(
packages/ml/legoesm/training/scale_build.py:1607:    per_year = [aerosol_layer_fields(
packages/tools/legoesm/forcing/surface_utils.py:813:def aerosol_layer_fields(acfg, day, lat_col, p_half_col, w_strat, *,
1:# legoESM Claude Memory
3:## RULE 0 — CONSERVATION OF MASS, ENERGY AND MOMENTUM. THE STRICTEST RULE.
22:## RULE 1 — CONCISE AND CLEAR. THE DEFAULT, EVERY REPLY.
56:## RULE 2 — RULES DO NOT ENFORCE THEMSELVES. CHECK THE GATE, NOT THE TEXT.
92:## Role
95:## Repo Facts
102:## Training (`src/legoesm/training/`)
112:## RULE 4 — SEARCH BEFORE YOU BUILD, AND PONYTAIL FULL IS THE DEFAULT.
132:## RULE 5 — DIFFERENTIABILITY IS NEVER TRADED FOR ORACLE FIDELITY.
170:## RULE 3 — A FIX WHOSE DEFAULT KEEPS THE BUG IS NOT A FIX. IT IS A BUG WITH A KNOB.
205:## ABSOLUTE RULE — NO UNASKED CHOICES. ASK FIRST, EVERY TIME.
356:## Operating Mode
428:## COMPARING TWO CONFIGURATIONS = A SIDE-BY-SIDE TABLE, ALWAYS
438:## ACCURACY FIRST — A CLAIM THAT DRIVES A CODE CHANGE GETS REVIEWED BEFORE THE CODE
470:## Epistemic rules (non-negotiable)
472:### Never infer an API — read it
481:### Report uncertainty explicitly
492:### Diagnose before patching
510:## Verification (JAX-specific)
531:## Scope
552:## Attribution Gates — MANDATORY, each from a real 2026-07 failure
660:## Implementation Discipline — the CODE and its PROSE are both claims
715:## Assumption Gates — from six wrong assumptions in ONE session (2026-08-05/06)
766:## Compute Discipline — speculation costs GPU-hours, not just credibility
808:## Diagnosis Discipline — one session, ~10 GPU arms, 5 of them wasted
880:## JAX
888:## Earth System
897:## Parallel/HPC
907:## Oracle-Recipe Fidelity (ocean) — see docs/ocean/fidelity/oracle_recipe_strategy.md
915:## Recipe×Setup template adapters (#388)
922:## Validation
943:## Domain Architect vs Syntax Engine (AI guardrails)
951:## Commands
964:## File Layout (audit — enforce on EVERY new file; no random files)
975:## Bug Triage
982:## Hygiene/Imports/Tests (audit)
995:## Constants/Params (audit)
1005:### Parameter hygiene — declare per category at file top + machine-readable tunable/fixed split (gated)
1019:## Naming
1027:### Open naming debt
1034:## Dispatch (audit)
1040:## Common Mistakes
1056:## Assets
1059:## Response Style
1061:### RULE -3 — BREVITY IS THE DEFAULT, LENGTH IS OPT-IN (2026-08-18, FIFTH callout)
1081:### RULE -2 — MECHANICAL LENGTH LIMITS (2026-08-14, THIRD callout)
1099:### RULE -1 — THE CAP APPLIES TO GOOD NEWS AND BAD NEWS ALIKE (2026-08-12)
1123:### RULE -1 — CLARITY IS THE HARD RULE. IF THE USER CANNOT FOLLOW IT, IT FAILED.
1153:### RULE 0 — HARD CAP ~60 WORDS. ANSWER FIRST. STANDING ORDER, ALL SESSIONS.
1178:### PLAIN LANGUAGE FIRST — the rule that finally worked (2026-08-07)
1201:### Earlier callouts (same session) — still in force
1248:# iterate-with-codex agent

exec
/usr/bin/bash -lc "sed -n '1570,1680p' packages/ml/legoesm/training/scale_build.py && sed -n '356,428p' CLAUDE.md && sed -n '510,552p' CLAUDE.md && sed -n '880,1005p' CLAUDE.md && python -c 'import jax,pytest,netCDF4; print(jax.__version__)' " in /glade/derecho/scratch/pg2328/wt_volc
 exited 1 in 0ms:
    acfg = AerosolConfig(
        enabled=True, source="climatology", path=str(cl["aerosol_file"]),
        ccn_path=str(cl.get("aerosol_ccn_file", "") or ""),
        use_reference_if_missing=True,
        reference_aod_550=float(cl.get("aerosol_reference_aod", 0.03)),
        volcanic_enabled=bool(volc), volcanic_path=volc,
        volcanic_lw_enabled=(lw and bool(volc)),
        hold_outside_file_years=False)
    return acfg, tuple(range(years[0], years[1] + 1)), (lw and bool(volc))


def wb_aerosol_forcing(spec, grid, sigma, carry, time_ns):
    """``aerosol_od`` / ``aerosol_ccn_aod`` / ``aerosol_lw_od`` for one sample:
    the AMIP driver's fields (``aerosol_layer_fields``) at the sample's
    calendar date in each climatology year, averaged.  Columns are the
    initial state's (p_s, T) -- the stratospheric placement uses its
    tropopause, as the driver's daily refresh does -- and the fields are held
    over the forecast window.  The date is put on the readers' noleap clock;
    29 February reads 28 February."""
    import jax.numpy as jnp

    from legoesm.forcing.surface_utils import (
        aerosol_layer_fields,
        stratospheric_layer_weight,
    )

    acfg, years, lw_active = spec
    t = np.datetime64(time_ns, "s").astype(object)
    day = min(t.day, 28) if t.month == 2 else t.day
    sim_day = (_NOLEAP_MONTH_START[t.month - 1] + day - 1
               + (t.hour * 3600 + t.minute * 60 + t.second) / 86400.0)
    p_s = jnp.asarray(carry.p_s).reshape(-1)
    p_half = jnp.asarray(sigma.pressure_at_half(p_s))
    p_full = jnp.asarray(sigma.pressure_at_full(p_s))
    lat_col = jnp.repeat(jnp.asarray(grid.lat), int(grid.n_lon))
    temp = jnp.asarray(carry.T).reshape(p_full.shape)
    w_strat = stratospheric_layer_weight(p_half, lat_col, p_full, temp)
    per_year = [aerosol_layer_fields(
        acfg._replace(start_year=y), sim_day, lat_col, p_half, w_strat,
        active=True, lw_active=lw_active, dtype=p_s.dtype) for y in years]
    od, ccn, lw = (jnp.mean(jnp.stack(f), axis=0)
                   for f in list(zip(*per_year))[:3])
    out = {"aerosol_od": od, "aerosol_ccn_aod": ccn, "aerosol_lw_od": lw}
    volc = [f[3] for f in per_year]
    if volc[0] is not None:
        out["aerosol_sw_volc"] = jnp.mean(jnp.stack(volc), axis=0)
    return out


def build_spectral_forcing(sl, grid, time_ns, year, *, aerosol=None,
                           sigma=None, ic_carry=None):
    """The traced per-sample forcing dict of the spectral core, from a slice.

    ONE builder for the training loader and the WB2 forecast eval, so a model
    is scored with exactly the forcing it was trained with: the prescribed
    surface-flux planes and the land fraction ride along whenever the slice
    carries them (i.e. whenever its ``TrainingERA5Config`` asked for them),
    regridded and flattened exactly like ``T_sfc``.  Signs are already the
    model's: ``era5_to_state`` did the ERA5 flips.
    """
    import jax.numpy as jnp

    from legoesm.training.era5_to_state import regrid_2d_to_gaussian

    def _plane(field):
        return jnp.asarray(regrid_2d_to_gaussian(
            field, sl.lat, sl.lon, grid)).reshape(-1)

    sst = _plane(sl.sst)
    doy_1based, sod = era5_time_to_forcing_calendar(time_ns, year)
    forcing = {
        "T_sfc": sst,
        "sic": jnp.zeros_like(sst),
        "day_of_year": jnp.asarray(doy_1based),
        "seconds_of_day": jnp.asarray(sod),
    }
    # Shared plane builder (grid-shaped there), flattened to columns here so
    # the output dict is identical to the pre-helper version.
    for _key, _plane_2d in prescribed_surface_planes(sl, grid).items():
        forcing[_key] = _plane_2d.reshape(-1)
    if aerosol is not None:
        forcing.update(wb_aerosol_forcing(aerosol, grid, sigma, ic_carry,
                                          time_ns))
    return forcing


def _load_era5_samples_spectral(cfg, yml, grid, sigma, *,
                                rank=0, nproc=1, host_resident=False):
    """(ic, target, forcing) samples on the Gaussian grid for the spectral core.

    ``ic``/``target`` are SegmentCarry on the Gaussian grid
    (``era5_to_spectral_carry``); ``forcing`` is the traced dict the spectral
    rollout's prescribed-SST pathway consumes (``forcing_base``: flat
    ``T_sfc``/``sic`` (ncol,) + calendar scalars — SegmentForcing doctrine, all
    traced so per-sample values never retrace).  Consumed only inside the
    spectral ``make_run_seg(...).raw`` (opaque to the trainer loop).

    #1286: with ``nproc > 1`` this builds ONLY ``rank``'s contiguous shard (fix
    B — no global materialization); with ``host_resident`` each built sample is
    moved off-device to host numpy (fix A — the loop ``device_put``s per batch).
    """
    from legoesm.training.era5_to_state import (
        era5_to_spectral_carry,
        open_era5_zarr,
    era5_terrain_product,
)

    # ic/target slices need the state only; the sample-start slice (2-D
    # fields only: asking it for condensate would re-read the cloud store per
    # sample for data it throws away) is the one that carries the prescribed
    # surface planes.
## Operating Mode
- **TERSE BY DEFAULT.** Drop articles/filler/pleasantries/hedging; fragments fine. Report `[thing] [state] [next]`, not prose. No restating what was just done, no feature tours, no explaining a simplification at more length than the code. Numbers/tables over narration. Full prose ONLY when asked for it (report/walkthrough), or for security warnings, irreversible-action confirmations, and multi-step sequences where fragments risk misread. Code/commits/PRs/docstrings: written normally.
- **DO WHAT WAS ASKED — DO NOT EXTRAPOLATE.** Deliver requested thing, then STOP and report. No inventing adjacent work because it seem useful: no unrequested helper/launcher/benchmark scripts, no extra files staged "for later", no speculative refactors, no scope widened from "fix X" to "also improve Y". Adjacent work look warranted → NAME it in one line, let user choose. Do not pre-build. Uncertain whether in scope? It not; ask. Over-delivery not helpfulness: cost review time, bury actual change, create artifacts nobody vetted.
- **Irreversible / outward-facing actions need explicit permission EVERY time**: rewriting published history (`push --force*`), amending/squashing pushed commits, deleting or overwriting files the user created, `git reset --hard`, closing/merging PRs, installing or upgrading system packages, reboots. Approval for one such action does NOT carry to the next. Propose, then wait.
- Nontrivial task: short plan before edit. Read nearby impl+tests first. Ambiguous numerics/physics/API: ask.
- Minimal diffs. No unrelated refactor in bug fix.
- **DUAL adversarial review is the DEFAULT for EVERY substantive change — TWO
  independent reviewers, never one** (user directive 2026-08-12, restated
  STRICT 2026-08-13, again 2026-08-20 as "make this the default behaviour").
  Author NEVER review own code. Route by who WROTE code — two reviewers always
  the other two:
  **Claude-authored → codex + GLM-5.2** (`mcp__zai__ask_glm`);
  **GLM-authored → codex + Claude**;
  **codex-authored → Claude + GLM-5.2**.
  Both BEFORE the PR, not after; report both verdicts in PR body and status
  line. Subagents doing implementation must be told to run BOTH — they default
  to codex only. Applies to measurement harnesses and probes too: instrument
  decide what we believe.
  Why two: they catch DIFFERENT classes. Codex find diff defects (broken
  contracts, vacuous tests that cannot fail, silent fallbacks, binding gates
  accepting wrong arm). GLM find MECHANISM defects (wrong objective, wrong
  regime, lever whose premise receipts already falsified — it retracted own
  top-ranked lever once measurement contradicted it).
  Reviewer disagreement = SIGNAL, not noise: name disputed point and the
  measurement that discriminates, run it if cheap, never average the two.
  A REVIEWER'S FINDING IS A HYPOTHESIS, NOT AN INSTRUCTION: measure before fix.
  2026-08-20, GLM predicted rounding error 1.4e-5 that would refuse valid
  restart files; measured over every column model builds it was 7.9e-8, so guard
  added for it was DELETED and measurement pinned instead — unmeasured finding
  buy knob that never bind.
  **"Substantive" includes MERGE CONFLICT RESOLUTION**, and rebases,
  cherry-picks, back-ports, probes, test fixtures. Conflict resolution =
  highest-risk case in list: someone chose between two versions of same code and
  diff not record what was discarded. FAILURE 2026-08-20: merged 17-day-old
  branch, hand-resolved 14 conflicts across 7 files including two independent
  fixes for one defect, shipped unreviewed until asked.
- **Codex adversarial review MANDATORY after any major code implementation/change.** Trigger: new module/feature, dycore/physics/parallel/ocean/land/ice/coupler/training edit, >~50 LOC, multi-file, or anything touching numerics/AD/JIT/pytree/conservation. Run **iterate-with-codex agent** loop below (`/codex:adversarial-review --wait` → fix flagged → `/codex:review --wait` → repeat until clean or 30 iter) BEFORE declaring done; report that review ran + verdict.
  **Review SUBAGENT dies (spend limit, API error) = NOT review waiver — codex
  CLI is separate binary with separate credentials, usually still reachable:
  `codex exec --sandbox read-only -C <repo> "<prompt>"` (`which codex`,
  `~/.codex/auth.json`). Try CLI directly before ever proceeding unreviewed. BOTH
  unavailable → say "UNREVIEWED" in every status until one succeed.**
  2026-07-26: subagent hit monthly spend limit, many iterations ran unreviewed
  while CLI worked fine whole time. Exempt: trivial/mechanical edits (typo,
  comment, rename, doc/markdown/`.tex`-only, single config value).
- **Pre-impl search mandatory**: before new fn/helper/class/operator/diagnostic/init/load/loss/numerical routine, grep `src/legoesm/` for similar names/docstrings/formulas in `thermo.py`, `constants.py`, `eos.py`, `ml/loss.py`, `diagnostics/`, `core/`, `atmosphere/physics/_shared.py`. State searched+found. Similar exists → extend/factor.
- **Shared utilities — never re-derive** (prod, scripts, validators, plotters, tests, notebooks, probes):
  - Constants: `from legoesm import constants` → `T_freeze`, `R_d`, `c_pd`, `L_v`, `R_v`, `epsilon`, `g`, `p_ref`, `kappa`, `sigma_sb`, `T_freeze_ocean`. No literals `273.15`/`287.0`/`1004.64`/`2.501e6`/`461.51`/`0.622`/`9.80616`/`6.371e6`/`7.292e-5`.
  - Saturation: `from legoesm.thermo import saturation_vapor_pressure, saturation_mixing_ratio, saturation_mixing_ratio_ice`. No re-impl Tetens/Magnus/Clausius–Clapeyron (plotters incl). Why: re-derived `e_sat=611.2*exp(17.67*Tc/(Tc+243.5))` diverged from model → false supersat in CI.
  - Column integrals: `legoesm.diagnostics.column_integrals` (`column_water_vapor`). No inline `jnp.sum(q*p_s*dsigma)/g`.
  - Losses: `ml/loss.py` (`area_weighted_mse`, `spectral_loss`, `per_variable_mse`).
  - Optimizer: `ml/training.create_optimizer()` (warmup+cosine+clip).
  - SCM-RCE gradient tuning: reuse `scripts/run/run_scm_rce_campaign.py` for CRM
    reference extraction / SCM evaluation and `legoesm.training.scm_rce_metrics`
    for normalized profile score. No duplicated RCE profile numerics.
  - SCM-RCE param training defaults to MUON via `ml.training.create_optimizer()`,
    init from `results/scm_rce_campaign/tuned_parameters.json`, writes recommended
    trained JSON under `results/`, never mutates production `*Config` defaults.
    Apply trainable overrides inside loss so leaves traced; static frozen leaves
    stay outside.
  - Every new `.py`, including `scripts/run/*.py` drivers, gets direct test.
    Scheme/factory dispatch must raise on unknown selections.
  - Atm column (h, ρ, virtual T): `atmosphere.physics._shared`.
  - Ocean EOS/pressure: `ocean.eos` (`compute_ocean_rho`, `compute_ocean_rho_and_pressure`).
  - SFNO: `ml/sfno.py`. No new neural op archs in training.
  - Channel packing: `ml/channel_packing.py` (`PE3DChannelSpec`, `pack_pe_state`, `unpack_pe_output`).
  - Ocean baroclinic (#214): `ocean/dynamics/ocean_tendency_common.py` (`iterate_eos_and_pressure_anomaly`, `apply_sponge_tracer_relaxation`, `apply_freshwater_virtual_salt_top`, `implicit_bottom_drag_factor`) in new `ocean_pe_*.py`.
  - Ocean barotropic (#214): `ocean/dynamics/barotropic_common.py` (`compute_filter_weights`, `bebt_blend`, `maxvel_clip`) in new `barotropic_*.py`. `tests/ocean/unit/test_no_scheme_duplication.py` enforces.
  - Plotters NOT exempt. Use model helpers for q_sat, RH, ρ, virtual T, MSE.
- No duplicate numerics across dycores/physics/grids/tests. Indexing/naming-only copy-paste forbidden.
- **No laziness on hard/large code** (>100 LOC, multi-component, full operator chains): no `pass`/`NotImplementedError` stubs, no partial-called-done, no skip edge cells/boundary halos/corner stencils/non-duogrid/MPI-sharded/AD-VJP. No happy-path-only tests. Too big → say so, list remainder, quantify risk.

## COMPARING TWO CONFIGURATIONS = A SIDE-BY-SIDE TABLE, ALWAYS
## Verification (JAX-specific)

Code not done until it run. Correctness claims require output.

- **Shapes/dtypes**: check with `jax.eval_shape` before running anything
  expensive. Cheap, catch most errors.
- **Gradients**: any new `custom_vjp`/`custom_jvp`, adjoint, or hand-derived
  derivative must pass `jax.test_util.check_grads(f, args, order=2)` before you
  claim it work. Gradient that run is not gradient that correct — single most
  common way to ship silently wrong result here. (Applies directly to
  `_sendrecv_vjp` in `halo_exchange.py` and any new MPI-AD path.)
- **jit parity**: run function eager and under `jit`, compare outputs.
  Divergence = tracer bug (Python-side branching, `.item()`, `if` on traced
  value, host callbacks).
- **Sharding**: verify with `jax.debug.visualize_array_sharding` or by printing
  `.sharding`, not by reasoning about what annotation should do.
- **Numerics**: default float32. State tolerance you compare at. No `==` on
  floats. Test need float64 → say so explicitly, no silent `jax_enable_x64`.
- **donate_argnums / buffer donation**: never add without confirming donated
  buffer not reused. Fail silently or crash far from cause.

## Scope
One change at a time. No refactor of adjacent code, no rename, no "improve" of
code I didn't ask about. Long unbroken generations drift into invention — prefer
small verified diff over large plausible one.

**DO NOT EXTRAPOLATE. Do only what was asked** (user directive 2026-08-06).
The ask = deliverable, not starting point to reason outward from.

- Related-looking problem you notice = ONE-LINE report, not work item. Name it
  and stop; do not start it.
- No widening scope because fix "would only be complete if" something adjacent
  also done. Ship the ask; state the boundary.
- New artifacts (scripts, benches, plots, panels, public APIs, config knobs)
  only when asked or genuinely required to finish the ask. Unsure whether
  required → it not; ask in one line.
- No proposing or launching compute user did not ask for.
- FAILURES 2026-08-05/06: scoped 128-GPU coupled ladder nobody requested off a
  question about existing plots; added coupled panel to figure when asked to
  assess the figure; wrote probes and helper scripts for questions never posed.
  Each cost round-trip and buried actual answer.

## Attribution Gates — MANDATORY, each from a real 2026-07 failure
## JAX
- Pure pytree fns. `lax.scan` time integration. `vmap`/batched arrays over Python loops on array dims. `jnp.where`/`lax.cond`/`fori_loop`/`scan` not Python control flow on traced.
- **Feature gating exception** (`fix_mass`, `fix_moisture`): Python `if` on static bool in closure — NOT `jnp.where` (traces both branches). `jnp.where` only for data-dependent traced selection.
- Stable shapes. No retrace. Dtype: spectral=x64+complex128; finite-volume can float32.
- No host/device thrash, NumPy in traced code, hidden non-JAX side effects.
- **Buffer donation + `jax.grad`**: `donate_argnums` conflicts reverse-mode AD. JIT fn inside `jax.grad`/`eqx.filter_value_and_grad` → provide non-donating variant (`.raw`). See `build_segment_fn`.
- **Closures vs explicit args**: closure captures = compile-time consts. Per-iter changing val (SST, solar) → pass as traced arg. See `SegmentForcing`.

## Earth System
- Conservation, metric consistency, staggered-grid consistency, halo correctness = first-class.
- No silent clip/damp/coerce unless justified+validated.
- Preserve units, sign conventions, monotonicity/positivity, hydrostatic/nonhydrostatic.
- Cubed-sphere/curvilinear: assume edge+metric errors first.
- Physics coupling: column closure + consistent flux signs.
- DA/diff: preserve smoothness. No gratuitous nondiff.
- **Sign-convention check MANDATORY on every equation/flux/tendency edit.** Before declaring done on any code touching PDE term, flux, source/sink, BC, or budget update: (1) state coordinate convention in scope (z up/down, flux positive-up/down/into-body) as comment at the term; (2) walk EACH term, confirm its sign match that convention — gravity vs capillary/diffusion divergence, top vs bottom BC, source vs sink, the `±` in `state = state ± dt·tend`; (3) confirm budget closes (`in − out − Δstorage = 0`) and exchanged fluxes carry SAME sign at both ends of a coupling (land runoff `+into ocean` must arrive `+into ocean`). Comment label and code must agree — flux commented "upward" computed as downward = defect, fix label or math. Mechanical gate where feasible: sign/conservation unit test (analytic column, manufactured solution, or `assert` budget residual ≈ 0) — passing norms alone never certify sign right (flipped flux can still be small). Common flips: z-axis direction, evap positive-up vs moistening, brine/salt vs freshwater dilution, stress atmospheric vs ocean convention (`-tau`), free-drainage vs gravity double-count.

## Parallel/HPC
- Correctness across serial/multi-device/MPI/hybrid.
- Sharded: reason about halo exchange, reductions, partition specs, global invariants.
- Validate single-rank → smallest distributed.
- Backends differ (Metal/GPU/CPU/spectral/MPI). Apple Silicon: spectral on CPU.
- **MPI**: `initialize_distributed(global_n=N)` → `scatter_to_local()` → rank-local step → `gather_to_global()` for I/O only. Never full global per rank.
- **4D halo**: `pad_halo_4d()`+`pad_halo_vector_4d()` all vert in one msg. All 3D ops in `operators_3d.py` use 4D. Never `vmap(pad_halo)`.
- **MPI halo AD**: all `sendrecv` via `_sendrecv_vjp` (`@jax.custom_vjp` in `halo_exchange.py`). Only `allreduce(SUM)` AD-safe (via `global_sum_mpi`; `final_loss=True` only for a sum seeded as the loss); `MAX`/`MIN`/`allgather`/`bcast` = diagnostics only.
- **Device mesh under MPI**: per-rank count to `create_device_mesh()`, not total.

## Oracle-Recipe Fidelity (ocean) — see docs/ocean/fidelity/oracle_recipe_strategy.md
- ADDITIVE to Validation Rules: oracle work NEVER replaces unit tests, ocean matrix, conservation checks, or visual verification. Truth tiers (conservation/equivariance/analytic) outrank oracle-matching.
- Recipe = pure config selecting shared canonical blocks (never bespoke `veros_*` solver). Oracle-matching numerics go in canonical module (`eos.py`, advection/limiter dispatch, `vertical_mixing/`, integrator dispatch) as selectable options.
- Mimicry-only glue (halo strip, axis transpose, time-level handling) lives in fidelity harness, never model. Test: "would a user with different goal ever select this?" No → harness.
- Conventions handled only in bridge, verified by equivariance tests (`physics(φ(x))=φ(physics(x))` to tol); a "convention" changing wet domain/answers is physics → config, not bridge.
- Constants are config (`ConstantsConfig`), not module-global monkey-patches (no `override_constants` in shippable paths); defaults reference `legoesm.constants`; base only, derived (κ,ε) recomputed.
- Oracle tendency-match (tier 3) trusted only for block that also clear truth tiers (0–2). MACHINE-ENFORCED (#388 Ask#4): `ocean/fidelity/precedence.py::evaluate_precedence` LOCKS oracle tiers (≥3) on any truth-tier (0–2) failure; surfaced + exit-gated by `scripts/validate/ocean_fidelity/build_fidelity_scorecard.py` (the one generated scorecard).

## Recipe×Setup template adapters (#388)
- **One shared selector, never re-implemented.** Every component YAML experiment adapter exposing a `setup:` block (atmosphere `Config`, ocean `OceanExperimentConfig`, sea-ice `SeaIceExperimentConfig`, …) MUST route validation + command-building + signature through `legoesm.core.setup_selector` (`MatrixRunnerSpec` + `validate_setup`/`build_matrix_command`/`setup_signature`/`require_positive_finite`). Per-component CLI differences (case flag `--only`/`--test`, exact-match `exact_prefix`, which `--levels`/`--dt`/`--days`/`--resolution` flags exist) are a `MatrixRunnerSpec`, NOT a copy-pasted `parts=[...]` builder or private `_SETUP_KEYS`/`_validate_setup`. Component-specific case-name validation goes through `known_names=` kwarg. Re-implemented selector = REJECTED in review. Factory naming: `_<component>_matrix_spec()`. FOLLOW-UP: dedup `OceanExperimentConfig`'s inline selector onto shared helper once PR #465 + shared-helper PR both land on main (inline copy predates extraction).
- **Every setup template's `(name, grid)` MUST be a real matrix case**, enforced by catalog-backed test (`run_<comp>_test_matrix._build_test_matrix()` → assert pair exists) so `--test/--only =<name> --grid <grid>` never select nothing. New template without this gate → REJECTED.
- **An exact case selector matching nothing is a hard error** (`raise SystemExit`) in every matrix-runner mode — never silent no-op (dispatch-hardening). MPI-safe: key guard off global pre-slice match count; empty-slice ranks fall through to barrier.
- **New `experiment_registry` mode → same-PR `get_adapter` test** in `test_experiment_registry.test_get_adapter_resolves_classes` (template test importing class directly does NOT cover dispatch).
- Land has no standalone idealized-case surface (no registry/matrix runner) → recipe×setup N/A there; do not invent one to match pattern. Truth-tier precedence stays ocean-scoped per `oracle_recipe_strategy.md`; use `TRUTH_TIERS`/`ORACLE_TIER_FLOOR` by name (no hardcoded `3`).

## Validation
- Narrowest test after edits. Numerical changes: analytical/benchmark > unit tests alone. `JAX_ENABLE_X64=1` unless float32/Metal task.
- Dycore: Williamson, Galewsky, Jablonowski-Williamson, DCMIP, Held-Suarez, ocean benchmarks.
- Conservation/reductions/coupler: mass+energy diagnostics.
- Parallel: unsharded vs sharded, single-rank vs MPI.
- Too expensive: say what ran/didn't, residual risk.
- **CRITICAL — Controlled comparison: change ONE variable, hold eval protocol FIXED to baseline.** To claim a change (resolution, params, scheme, days) improved/degraded/"is comparable" vs prior result, keep EVERYTHING else byte-identical to that baseline: forcing data + its sampling (years/days/hours/climatology), evaluation grid, metric definition (bias vs RMSE), region masks, timestepping. Metric that moved because protocol/sampling changed = **CONFOUND, not result** — NEVER compare number computed on one sampling/grid/metric to number from another and call difference an effect. Resource limit (network cap, compute, time) force lighter or different sampling → **re-run BASELINE at that SAME sampling before comparing** — fresh baseline cheap insurance; confounded claim not. Before writing "improved"/"degraded"/"better"/"comparable" vs any earlier number, explicitly confirm two configs differ ONLY in variable under test; cannot → say so, claim NO direction. Report full config (data+sampling, grid, days/steps, params, metric) next to every number so reader know exactly what compared. Assuming two runs comparable when setup drifted = error that turn "we improved it" into "we degraded it."
- **CRITICAL — Precision gate for comparison/skill/causal claims (do NOT be sloppy — 2026-07-20 EC-site lesson).** (a) **Harness self-check FIRST**: before reporting NEW scheme's skill vs validated baseline, run KNOWN baseline through YOUR OWN harness, confirm it reproduce baseline's established/published number (±small tol). Your harness score validated scheme wildly off — e.g. two-leaf H looked "broken" (NSE≈−1) when paper figure track obs — the HARNESS is wrong; fix BEFORE any new-scheme claim. (b) **Match reference metric EXACTLY**: metric aggregation (daily-mean vs half-hourly point-wise NSE), window (multi-year JJA vs one summer), masks (valid-forcing) ALL part of protocol. Report SAME metric reference used, plus any alternative, WITH sample N. Single-window point-wise metric is NOT skill verdict, and never rank schemes by metric punishing one scheme's known artifact (point-wise scatter/spikes) while ignoring dimension of interest (mean diurnal shape). (c) **Never let QC mask flatter one side**: drop scheme's unphysical spikes from scoring → report BOTH failures-penalized primary score AND separately-labeled plausible-only sensitivity score. (d) **Do NOT INFER causal origin — INSTRUMENT it**: "the spikes come from X" require LOGGING X and alternatives (raw vs intermediate vs final), not deduction from reading one code path; "correctly hooked up" require reading FULL path, not entry point. (e) **Label every claim CONFIRMED (evidence shown) vs PLAUSIBLE (inferred)** — adversarial review WILL refute over-confident inferences; state uncertainty up front rather than presenting verdict table a five-minute check overturn.
- **CRITICAL — A gradient/AD-tuning result that underperforms a known-good baseline is a BUG in your method until PROVEN otherwise — never attribute it to physics.** Before interpreting ANY optimizer/AD tuning outcome, or attributing residual error to physical cause (radiation, resolution, scheme's structure): (1) **Differentiated loss MUST be exact quantity you evaluate and report.** Truncated-BPTT / short-window / stop-gradient'd-spinup / cached-state proxy differing from full evaluation = OBJECTIVE MISMATCH, not result — optimizer minimize proxy while eval barely move. Differentiate real objective (e.g. full RCE equilibrium the eval scores), or prove proxy tight. (2) **Verify AD actually flows:** finite-difference check on ≥1 parameter (analytic grad vs `(L(x+ε)−L(x−ε))/2ε`, ratio≈1), confirm gradients finite AND non-negligible (not silently zeroed by `stop_gradient`, frozen/`enable_*=False` leaf, `lax.stop_gradient`/`jax.lax.cond` dead branch, or detached recompute). (3) **Sanity vs known-good baseline FIRST:** derivative-free or prior method got large improvements and your gradient method barely move loss → YOUR optimization broken (weak objective, tiny effective LR, over-aggressive line-search backtracking to ~0 step, frozen leaves) — fix before drawing conclusions. Concluding "residual bias is radiative/structural, not tunable" from run whose optimizer never actually worked = exact error that ship FALSE scientific claim. Near operationalization this disqualifying: barely-moving loss, suspiciously-flat before/after, or result contradicting known-good baseline BLOCKS delivery until gradient path verified end-to-end. See [[feedback_verify_ad_before_physics]].
- **CRITICAL — Visual verify spatial/grid artifacts**: passing tests+norms NECESSARY ≠ SUFFICIENT for cubed-sphere ops, halo exchange, diffusion coeffs, grid metrics. Edge artifacts/cube imprint/grid-scale noise only detected visually (v-wind W2, wind_speed W5). Run `--only sw --grid cubed_sphere --quick` + inspect PNGs vs baseline. Norms can improve while artifacts worsen. Never claim "tests pass, edge fixed" from pytest alone.
- **Diffusion sensitivity**: div damping + hyperdiff AMPLIFY halo errors at cubed-sphere face boundaries. Check W2 v-wind visually when touching `_hyperdiff_cube`, `_div_damp_cube`, diffusion params.
- **Visual-regression gate (cube imprint)**: `scripts/validate/visual_regression.py --check` numericises W2 v-wind cube-imprint check (SSIM + per-panel perceptual hash + edge-artifact ratio vs tiny committed ref in `tests/visual_baselines/`). Deterministic metric math gated in CI (`tests/test_visual_regression_metrics.py`); full cube-SW `--check` runs as NIGHTLY non-blocking CI job until tolerances calibrated across CI hardware. Tiny numeric baselines (.npy+json) ARE tracked — the one carve-out to "no tracked visual baselines".
- **CRITICAL — VALIDATE THE INSTRUMENT BEFORE QUOTING ITS NUMBER (2026-07-25 duogrid lesson: 8 confident claims, all retracted).** Diagnostic script = UNTRUSTED CODE until it pass own controls. Never state finding — never write "measured", "confirmed", "proven", "VERDICT" — from probe's first output. **Before quoting any diagnostic number, run these five checks and say in the message that you ran them:**
  1. **Right conserved/invariant quantity?** Budget what SYSTEM conserves, not convenient proxy. (Failed: reported "vertex creates energy" from **KE alone** — KE NOT conserved in shallow water, it trade with PE. Total `E=∫area(½h|V|²+½gh²)` reversed sign of conclusion.)
  2. **Same transform / units / staggering on BOTH sides?** Two "A-grid winds" from different operators = DIFFERENT QUANTITIES. (Failed: ours `c2l_ord2` vs oracle `C2L_ORD=4` — SAME raw state gave 1.96e-2 vs 5.79e-2, 3× swing that WAS the reported effect. Also: never budget across stage boundary where state change representation — mid-step FV3 winds in circulation form, produced ±5.6e10 garbage. REPEAT OFFENCE 2026-08-05, this time SHIPPED then retracted: paired predicted face velocity (41.87 m/s) against model's `max_speed` timeseries diagnostic (21.95 m/s), which is CELL-CENTRE average of same field whose face maximum was 42.30 — apparent "1.2 % agreement" was artifact of comparing across staggering, and it went into merged PR before 1-step run exposed it. Name staggering AND reduction of both sides in sentence quoting them.)
  3. **Same time, resolution, config?** Index by MATCHED TIME, not frame number. (Failed: mapped day→frame as `round(day)-1` against HOURLY file, comparing our day-1 to their hour-1; and quoted **C12** wedge gain (~300×) as mechanism for **C48** instability, where it ~124×.)
  4. **Is the metric measuring what its name says?** Prove it on synthetic case with KNOWN answer before use. (Failed: called `mean|f−4-neighbour-mean|` a "2Δx grid-scale" measure — it is high-pass/curvature residual that merely sharper SMOOTH feature reproduce. Failed: "gain" probe that re-filled FIXED source, trivially 1.0000 by construction.)
  5. **Can the reduction support the claim?** `max` over tiles/corners/components taken independently per run can peak at DIFFERENT physical locations; max-of-per-tile-means is not global mean. Keep argmax metadata, map to common physical location before claiming "localized".
  Plus: **diff ARRAYS, never printed summaries** (claimed "bit-identical ⇒ deterministic, not chaos"; arrays actually differed by 9e-6 — only rounded printout matched). **Never let probe print own verdict** ("=> the growth is REAL") — interpretation belong in analysis after controls pass, not baked into tool where it get echoed back as evidence. **`nanmax`/`nanmean`/`nansum` hide failures** — make NaN and missing frames FATAL. **Record every effective flag, env var and git SHA in each artifact**; default `--n 36` silently mis-slicing C48 file run fine and lie.
- **CRITICAL — LABEL EVERY CLAIM, AND PREFER RETRACTING EARLY.** Tag each statement **CONFIRMED** (control-passed evidence shown, instrument validated) vs **PLAUSIBLE** (inferred / single-run / uncontrolled). Chain of PLAUSIBLE steps is not CONFIRMED. Later measurement contradict earlier claim → **retract loudly and immediately in same message and in memory file** — no quiet move on, because stale confident claims get built on. Run codex adversarial review on DIAGNOSTIC TOOLING, not just model code: instruments decide what you believe, so bug there manufacture confident wrong physics conclusion. Cheap self-check before any big claim: *"what measurement would make this false, and did I run it?"* Answer no → claim PLAUSIBLE at best.

## Domain Architect vs Syntax Engine (AI guardrails)
See `docs/architecture/ai_guardrails/domain_architect_vs_syntax_engine.md`. Doctrine: human dictate *logic* (units, signs, conserved qty, valid scheme sets, references, acceptance criteria); AI fill *body*; every declared invariant checked **mechanically** so violations fail LOUDLY. Each gate = **tripwire, not proof**, ships synthetic-violation self-test (provably non-vacuous). NON-NEGOTIABLE harness (extend, never weaken; budgets/TODOs shrink only):
- **Spec-first physics contracts**: every physics scheme module declares `__physics_contract__` (units/signs/conserves/differentiable/reference/idealized_test); `tests/test_physics_contracts.py` partitions all `*/physics/*.py` into EXCLUDED / CONTRACT_TODO(shrink-only) / annotated — NEW physics file must ship contract or be classified. Author contract + acceptance test BEFORE body.
- **Atmospheric parameterization edits are high-risk**: run codex adversarial review, #477 truth-tier tendency validators, conservation tests, equilibrium-SCM-RCE realism harness before merge. NEW scheme must pass RCE realism gate or enter its shrink-only TODO partition with a reason.
- **CI ratchets** (AST + self-test, via shared `tests/_ratchet_audit.py`): `test_no_hardcoded_constants` (constants only from `legoesm.constants`), `test_no_saturation_reimpl` (saturation only from `thermo`), `test_dispatch_hardening` (no scheme guard silently deleted), `test_validate_strict_coverage` (no scheme field skips fail-early validation). Escape: real `# const-ok:`/`# satcurve-ok:` comment.
- **Local hooks** (`.claude/hooks/`, wired in `settings.json`): PreToolUse blocks edit adding banned constant/saturation prefactor to `.py` (fail-open); Stop reminds (once/session) to run mandatory codex review on uncommitted numerics/physics. CI remain authoritative (`Bash` heredoc bypass hook, not CI).
- **Best coding practices = use existing system**, not parallel one: ruff/mypy, import-linter (`alerting="error"`), inline-import budgets, pre-impl grep + shared utilities (no re-derivation), every new `.py` gets unit test, slopbuster sweeps, iterate-with-codex on substantial changes.

## Commands
- Install: `pip install -e ".[dev]"`
- Tests: `.venv/bin/python -m pytest tests/`
- Sci tests: `JAX_ENABLE_X64=1 .venv/bin/python -m pytest <target>`
- Atm matrix: `JAX_ENABLE_X64=1 .venv/bin/python scripts/matrix/run_atmosphere_test_matrix.py`
- Ocean matrix: `JAX_ENABLE_X64=1 .venv/bin/python scripts/matrix/run_ocean_test_matrix.py`
- AMIP: `.venv/bin/python scripts/run/run_amip.py`
- Dycore progression: `JAX_ENABLE_X64=1 .venv/bin/python scripts/matrix/run_atmosphere_test_matrix.py` (SW -> hydrostatic -> non-hydrostatic ladder; `--only sw|hydro|nh`). Old `tests/validation/run_dycore_progression_suite.py` superseded — its per-case child scripts removed.
- GPU/MPI scaling: `.venv/bin/python scripts/bench/run_levante_gpu_scaling.py --grid cubed-sphere --mode strong` (`docs/performance/REAL_HARDWARE_SCALING.md`)
- Scripts reorganized into buckets: `scripts/{run,matrix,bench,plot,validate,data,experiment,cluster}/`; debug in `scripts/tmp/`. See `scripts/README.md` + `## File Layout` below.
- MPI tests: `mpirun -np 2 .venv/bin/python -m pytest tests/distributed/`
- MPI diff: `mpirun -np 2 .venv/bin/python -m pytest tests/distributed/test_mpi_differentiability.py`

## File Layout (audit — enforce on EVERY new file; no random files)
- **New scripts go in correct `scripts/` bucket — NEVER `scripts/` root or repo root.** Buckets: `run/` (prod drivers), `matrix/` (test-matrix registries), `bench/` (perf/profiling/scaling), `plot/` (plot/replot/regen), `validate/` (non-matrix validators/verifiers/conservation checks), `data/` (download/build/prepare forcing+IC), `experiment/` (init/reproduce/templates/machine-detect/fetch), `cluster/` (SLURM `.sbatch` job wrappers, e.g. `cluster/omip_nemo/`). Pick bucket by what script DOES. New bucket need real category, not dumping ground. See `scripts/README.md`.
- **Debug / throwaway / one-off → `scripts/tmp/` ONLY** (eventually deleted): `_*`-prefixed probes, `diag_*`/`diagnose_*`, per-iteration scratch. Never at `scripts/` root. `_probe_*.py` gitignored.
- **No new files dumped at repo root.** Root keeps ONLY: `README.md`, `CLAUDE.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, `FEDERATION.md`, `project_status.md` (generated), `pyproject.toml`/lockfile/dotfiles. Everything else has a home.
- **No `.md` notes accumulating at repo root → `docs/`.** Dev-notes, change logs, faithfulness/audit trackers (`*_faithful.md`, `*_checks.md`, review logs) live under `docs/`. Reference by BASENAME so code comments survive the move.
- **No runtime outputs in git.** `diagnostics/`, `logs/`, `**/logs/`, `results/`, `checkpoints/`, `output/`, `*.zarr`/`*.nc`, root `*.png`/`*.pdf`/`*.svg` gitignored. Visual-regression baselines stay in LOCAL working copies, regenerated on demand — not tracked. Never `git add -f` a runtime artifact. EXCEPTION: `.physics-validator/` is tracked on purpose — review logs, probes and banked oracle results cited as evidence by docs, tests and validation scripts (user 2026-10-24); only its error captures are ignored.
- **Source stays under `packages/<pkg>/legoesm/`** (federation namespace). New subpackage → update `tests/test_federation_plan.py` same PR.
- **Do NOT "flatten" `packages/<pkg>/legoesm/<subpkg>/` → `packages/<pkg>/` — double name is LOAD-BEARING, not redundant.** Editable installs (dev/test workflow) require on-disk path == import path, and hatchling REFUSES prefix-*adding* `sources` remap in dev mode (`ValueError: Dev mode installations are unsupported when any path rewrite in the sources option changes a prefix rather than removes it`; editables#20). So `packages/atmosphere/legoesm/atmosphere/` is only layout where `import legoesm.atmosphere` resolve under `pip install -e`. Flattening to keep `legoesm.*` build wheel but break editable dev; flattening to bare `import atmosphere` mean rewriting ~23k import sites + dropping namespace. Verified 2026-07-13 — don't re-attempt. `src/legoesm/` is SANCTIONED root meta-package (the `legoesm.*` TOP-LEVEL modules: `constants`, `config`, `cli`, `registry`, `dycore_factory`, …), same load-bearing reason — NOT stray `src/` to remove. "Never add source at `src/`" mean don't create NEW `src/` trees for component code; existing `src/legoesm/` stays.
- **Tests mirror package tree under `tests/`** (`tests/<component>/<tier>/...`). Curated dycore regressions in `tests/atmosphere/dycore/regression/`. No new `test_*.py` at repo root.
- Staging: explicit pathspecs, NEVER `git add .`/`-A` — catch stray scratch + concurrent-session files.

## Bug Triage
- Instability: CFL, boundary, metric, halo, pressure-gradient, diffusion, dtype.
- Conservation drift: flux form, weights, reductions, state updates before fixers.
- Differentiability: control flow, shape changes, side effects, checkpointing, nondiff branches.
- Perf regression: retrace, host callbacks, scatters, sharding, Python loops.
- Cross-backend: dtype, x64, unsupported kernels, comm semantics.

## Hygiene/Imports/Tests (audit)
- Every new `.py` ≥1 direct unit test importing+exercising leaf module (tendencies for physics schemes — not just integration via factory). No `__init__.py` re-export / `supported_matrix.py` / factory dispatch add without same-PR test. Block PRs growing untested-LIVE count.
- New config dispatch (Literal + factory in `integration.py`): test via public config.
- **New user-tunable config field → same-PR CLI flag in every affected run script.** Rule applies to `ExperimentConfig`/`DycoreConfig`/`OutputConfig` fields in `run_amip.py`, `OceanExperimentConfig`/`KPPConfig`/`VerticalMixingConfig` in `run_omip.py`, `MultiLayerLandConfig` sub-configs in `run_lmip.py`. "User-tunable" = any float/int/str/bool field user would reasonably change for sensitivity run (excludes: internal-only flags such as `debug_precision`/`diagnostics_perf_mode`, unimplemented stubs like `carbon_cycle` in AMIP, parallel-milestone-gated fields like `distributed_mode=spmd`). Each new flag also needs: (1) wiring in `build_config_from_args` / equivalent config-builder, (2) round-trip test in corresponding `tests/unit/test_run_*_cli.py`, (3) `validate_strict` membership check if it is a scheme Literal. FLOAT tunables with `__param_spec__` need no flag: all four MIP drivers take `--config` (dests, choices-validated) + `--params` (registry-qualified `scheme_key.field`, bounds-validated; atm via flattened-scalar map). Reachability machine-audited by `tests/unit/test_params_reachability_audit.py` against shrink-only `_params_reachability_baseline.py` (new spec'd tunable unreachable in its component's drivers go red) — the #691 "CLI flag gaps" debt closure.
- Removing module: also remove `__init__.py` re-export, `supported_matrix.py` entry, dispatch, test file, `__pycache__`.
- No deprecated backward-compat wrappers — update call sites. No thin dispatch-only wrappers (`X_utils.py` re-exporting `X.py`) — inline/factor. Real branching across callers (`land/stomata_utils.py`) legit. Grid variants legit when genuinely different numerics; indexing-only copy-paste forbidden.
- **No top-level cross-package imports from `core/` to `runtime/`/`parallel/`/`driver/`/`training/`/`experiments/`.** Why: `from legoesm.runtime.backend import ...` at top of `core/precision.py` triggered `runtime/__init__.py` → `runtime.precision` → `core.precision` mid-init, breaking isolated pytest. Use function-scope deferred imports.
- **No import of private (`_`-prefixed) symbols across modules.** Promote (drop underscore + `__init__.py` re-export) or factor public wrapper. Mutable singletons (`grids.halo._halo_backend`/`_mpi_topology`, `cubesphere_exchange._spmd_mesh`): use accessors (`get_halo_backend`/`get_mpi_topology`/`get_spmd_mesh`), never import global. Audit: `grep -rE "from legoesm\.[^ ]+ import [^,]*\b_[a-z]" src/legoesm/ packages/ | grep -v " as _"` = 0 (line-grep miss multi-line/function-scope/`_UPPER` imports); CI ratchet `tests/test_no_private_cross_imports.py` (AST, incl. function-scope; dunder-exempt; shrink-only allowlist EMPTY since 2026-06-10 — keep it empty).
- **Test-only modules MUST be acknowledged.** Not wired into factory/`__init__.py`/prod driver: (a) wire same PR, (b) move to `_future/` + docstring + xfail/skip, or (c) delete.
- **Never commit `docs/references/`.** Local research PDFs/extracts. Cite by filename/DOI. Notes elsewhere (`docs/ocean/experiments/`). Staging: explicit paths, never `git add .`/`-A`.
- Slopbuster periodic: `/slopbuster audit all` or `/slopbuster review`.
- High-priority untested LIVE (touch any → add test same PR): `ocean/experiments/global_overturning.py`, `ocean/physics/{bottom_drag,convection,surface_forcing,vertical_mixing}/output.py`, `ocean/physics/convection/enhanced_diffusion.py`, `ocean/physics/vertical_mixing/{k_profiles,mpas_integration}.py`. RESOLVED 2026-05-29 (direct tests added): `coupler/surface_energy.py`, `ocean/dynamics/barotropic_common.py`, `atmosphere/dynamics/sfno_pe.py`, `atmosphere/dynamics/tracer_transport_mpas.py`; `atmosphere/physics/convection/_triggers.py` + `timestepping/tridiagonal.py` already covered.

## Constants/Params (audit)
- **All physical constants in `src/legoesm/constants.py`.** New constant (T, ρ, c, L, k, μ, EOS coeff, Schmidt#, R_earth) MUST be added BEFORE use. No constants in `config.py`/fn bodies/test fixtures/plotters/notebooks even with `# = constants.X` comment.
- **`getattr(..., "X", <literal>)` fallbacks count as hardcoded.** Use `getattr(grid, "radius", constants.R_earth)`. Same `setattr`/`dict.get`/`kwargs.get`.
- **No hardcoded physical constants in fn sigs/bodies in `src/legoesm/`.** No `def f(g=9.80616, ...)` → `g: float = constants.g` or config NamedTuple. NamedTuple defaults SHOULD reference constants. Tunable scheme params (sigmoid sharpness, τ, drag) stay in config NamedTuple.
- **No hardcoded tunable params in physics bodies.** Sigmoid sharpness, τ, Louis coeffs, KPP epsilon, emissivity, drag → scheme `*Config` NamedTuple. Exempt: safety floors (`eps=1e-30`), math constants (`0.5`, `2.0`), category lookup tables (PFT `L_v` in `surface_params.py`) with doc.
- **No `273.15` for C↔K in prod.** Use `constants.T_freeze`. `T_freeze_ocean=271.35 K` in `ocean/eos.py` = only intentional exception.
- **Tests+scripts+plotters same rule.** `from legoesm import constants`. No `9.80616`, `7.292e-5`, `6.371e6` literals.
- **Sigmoid sharpness/transition widths in JAX hot loops forbidden as magic numbers.** Inside `scan_step`/`cond`: fn kwarg with doc default OR scheme `*Config` field. Ex: `compute_moist_adiabat(lcl_sigmoid_width_pa=100.0)`, `PlumeConfig.active_sigmoid_sharpness=1e4`, DM95 `transition_width_frac=0.1` on `dm95_taper`/`dm95_taper_scalar`/`_triad_taper`.
- **Saturation re-impl in forcing modules forbidden.** Use `legoesm.thermo.saturation_mixing_ratio`/`saturation_vapor_pressure`.

### Parameter hygiene — declare per category at file top + machine-readable tunable/fixed split (gated)
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'jax'

exec
/usr/bin/bash -lc "sed -n '740,920p' packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py && sed -n '1080,1162p' packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py && sed -n '1640,1688p' packages/tools/legoesm/forcing/external.py" in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 0ms:
      # table dtype.
      if mcica_cloud_fraction is None:
          _mcica_shift = None
      else:
          from legoesm.atmosphere.physics.clouds.subcolumns import (
              cam_seed_shift)
          _mcica_shift = cam_seed_shift(p_full)
      T = T.astype(_table_dtype)
      p_full = p_full.astype(_table_dtype)
      p_half = p_half.astype(_table_dtype)
      q_v = q_v.astype(_table_dtype)
      sfc_temperature = jnp.asarray(sfc_temperature).astype(_table_dtype)
      cos_zenith = jnp.asarray(cos_zenith).astype(_table_dtype)
      # Optional external-forcing arrays must match the working dtype
      # too: under ``compute_fp32`` a float64 ozone / aerosol column
      # (built by the driver under JAX x64) would re-promote the optical
      # depth and break the RTE ``lax.scan`` carry dtype — exactly the
      # PR #343 leak class, resurfaced when the MPAS/spectral paths
      # started threading per-step o3_vmr / aerosol_od (2026-06-10).
      if o3_vmr is not None:
          o3_vmr = jnp.asarray(o3_vmr).astype(_table_dtype)
      if aerosol_optical_depth is not None:
          aerosol_optical_depth = jnp.asarray(
              aerosol_optical_depth).astype(_table_dtype)
      if aerosol_absorption_optical_depth_lw is not None:
          aerosol_absorption_optical_depth_lw = jnp.asarray(
              aerosol_absorption_optical_depth_lw).astype(_table_dtype)
      if aerosol_volcanic_moments_sw is not None:
          aerosol_volcanic_moments_sw = jnp.asarray(
              aerosol_volcanic_moments_sw).astype(_table_dtype)
      if cloud_path_liq is not None:
          cloud_path_liq = jnp.asarray(cloud_path_liq).astype(_table_dtype)
      if cloud_path_ice is not None:
          cloud_path_ice = jnp.asarray(cloud_path_ice).astype(_table_dtype)
      if cloud_path_liq_lw is not None:
          cloud_path_liq_lw = jnp.asarray(
              cloud_path_liq_lw).astype(_table_dtype)
      if cloud_path_ice_lw is not None:
          cloud_path_ice_lw = jnp.asarray(
              cloud_path_ice_lw).astype(_table_dtype)
      if cloud_r_eff_liq is not None:
          cloud_r_eff_liq = jnp.asarray(cloud_r_eff_liq).astype(_table_dtype)
      if cloud_r_eff_ice is not None:
          cloud_r_eff_ice = jnp.asarray(cloud_r_eff_ice).astype(_table_dtype)
          if cloud_rho_ice is not None:
              # The cloud optics read the table at 2 r 500/917 (CAM rhoi =
              # 500); r*rho/500 makes that 2 r rho/917 for the configured
              # bulk density (factor exactly 1.0 at rho = 500).
              cloud_r_eff_ice = cloud_r_eff_ice * jnp.asarray(
                  cloud_rho_ice / cloud_optics.RHO_ICE_BULK_CAM_MG2,
                  dtype=_table_dtype)
      if cloud_fraction is not None:
          cloud_fraction = jnp.asarray(cloud_fraction).astype(_table_dtype)
      if mcica_cloud_fraction is not None:
          mcica_cloud_fraction = jnp.asarray(
              mcica_cloud_fraction).astype(_table_dtype)
      _snow_parts = (cloud_path_snow, cloud_d_eff_snow, snow_optics)
      if any(x is not None for x in _snow_parts):
          if any(x is None for x in _snow_parts):
              raise ValueError(
                  "snow in the radiation needs cloud_path_snow, "
                  "cloud_d_eff_snow and snow_optics together")
          if cloud_path_liq is None and cloud_path_ice is None:
              raise ValueError("cloud_path_snow needs the cloud paths too")
          if (cloud_fraction is not None or cloud_path_liq_lw is not None
                  or cloud_path_ice_lw is not None
                  or sw_optical_field_only or lw_optical_field_only):
              raise ValueError(
                  "cloud_path_snow is wired for in-cloud (McICA) or plain "
                  "grid-mean paths only, not cloud_fraction scaling, "
                  "separate LW paths or the optical-field outputs")
          from legoesm.atmosphere.physics.radiation.rrtmgp.optics import (
              mitchell_ice_optics as _mio)
          cloud_path_snow = jnp.asarray(cloud_path_snow).astype(_table_dtype)
          cloud_d_eff_snow = jnp.asarray(cloud_d_eff_snow).astype(_table_dtype)
          snow_optics = _mio.cast_table(snow_optics, _table_dtype)
          cloud_snow_weight = (jnp.ones_like(cloud_path_snow) if cloud_snow_weight is None
                               else jnp.asarray(cloud_snow_weight).astype(_table_dtype))
      if ghg_vmr_override is not None:
          # Cast every numeric override (array, Python float/int, list)
          # to the working dtype; only floating leaves are retyped so
          # integer flags (if any ever appear) pass through unchanged.
          def _cast_ghg(v):
              arr = jnp.asarray(v)
              if jnp.issubdtype(arr.dtype, jnp.floating):
                  return arr.astype(_table_dtype)
              return arr
          ghg_vmr_override = {
              k: _cast_ghg(v) for k, v in ghg_vmr_override.items()
          }
      if solar_spectral_fraction is not None:
          solar_spectral_fraction = jnp.asarray(
              solar_spectral_fraction).astype(_table_dtype)

      # --- 0b. CAM-style transported layer ABOVE the model top (opt-in) ---
      # CAM RRTMG's "extra layer": one layer from p_top up to a 1 Pa lid with
      # T, q_v of the top model layer, no cloud / aerosol, and the overhead
      # column-mean ozone, so the beam reaching the model top has already
      # been depleted by the ozone above it.  Its heating is discarded and
      # the returned TOA face is the lid (true top).
      # ponytail: the ~2 W/m2 SW absorbed above the model top LEAVES the
      # column budget (TOA net - surface net != sum of model-layer heating),
      # exactly as in CAM.  Not applied to the optics-only (3D MC) paths,
      # whose per-layer output must stay on the model grid.
      overhead = (bool(getattr(config, "overhead_layer", False))
                  and not (sw_optical_field_only or lw_optical_field_only))
      # LW level temperatures from the model column (CAM6 rrtmg_state).
      T_lev = cam_interface_temperature(T, p_full, p_half, sfc_temperature)
      if overhead:
          # CAM's extra layer (rrtmg_state.F90 L122-136, kk = max(k-1, 1)):
          # both of its interfaces take tint(1), the top-layer temperature.
          T_lev = jnp.concatenate([T_lev[:, :1], T_lev], axis=1)
          p_top = p_half[:, :1]
          p_lid = jnp.minimum(p_top, 1.0)  # coeff-ok: 1 Pa lid (CAM-like ~0 Pa top)

          def _top(a, k0):
              return None if a is None else jnp.concatenate([k0, a], axis=1)

          def _zero(a):
              return None if a is None else _top(a, jnp.zeros_like(a[:, :1]))

          if o3_vmr is None:  # resolve model ozone first: fallback = o3(k0)
              o3_vmr = _standard_o3_profile(p_full)
          p_half = _top(p_half, p_lid)
          p_full = _top(p_full, 0.5 * (p_lid + p_top))
          T = _top(T, T[:, :1])
          q_v = _top(q_v, q_v[:, :1])
          o3_vmr = _top(o3_vmr, o3_vmr[:, :1] if o3_top_vmr is None
                        else jnp.asarray(o3_top_vmr).astype(
                            _table_dtype).reshape(ncol, 1))
          cloud_path_liq, cloud_path_ice = _zero(cloud_path_liq), _zero(cloud_path_ice)
          cloud_path_snow = _zero(cloud_path_snow)
          if cloud_d_eff_snow is not None:
              cloud_d_eff_snow = _top(cloud_d_eff_snow, cloud_d_eff_snow[:, :1])
              cloud_snow_weight = _top(cloud_snow_weight, cloud_snow_weight[:, :1])
          cloud_path_liq_lw = _zero(cloud_path_liq_lw)
          cloud_path_ice_lw = _zero(cloud_path_ice_lw)
          cloud_fraction = _zero(cloud_fraction)
          aerosol_optical_depth = _zero(aerosol_optical_depth)
          aerosol_absorption_optical_depth_lw = _zero(
              aerosol_absorption_optical_depth_lw)
          if aerosol_volcanic_moments_sw is not None:
              aerosol_volcanic_moments_sw = jnp.concatenate(
                  [jnp.zeros_like(aerosol_volcanic_moments_sw[:, :1]),
                   aerosol_volcanic_moments_sw], axis=1)
          if cloud_r_eff_liq is not None:
              cloud_r_eff_liq = _top(cloud_r_eff_liq, cloud_r_eff_liq[:, :1])
          if cloud_r_eff_ice is not None:
              cloud_r_eff_ice = _top(cloud_r_eff_ice, cloud_r_eff_ice[:, :1])
          nlev = nlev + 1

      # --- 1. Reshape (ncol, nlev) -> (ncol, 1, nlev+2) with halos ---
      T_3d = _add_halos(T[:, None, ::-1])
      p_3d = _add_halos(p_full[:, None, ::-1])
      p_3d = jnp.clip(p_3d, 1.0, None)
      # Upper-clip q_v strictly below 1 so the (1 - q_v) denominator in
      # the VMR conversion is bounded away from zero.  0.99 is far above
      # any physically plausible specific humidity (peak tropical surface
      # values are ~0.025); the bound only ever fires on numerical
      # pathology during spin-up and prevents singular/negative VMRs.
      #
      # **Clip AFTER ``_add_halos``**: linear extrapolation of a steep
      # boundary profile can produce halo values OUTSIDE [0, 0.99]
      # (e.g. q_v = [0.99, 0.0, ...] extrapolates to halo = 1.98), which
      # would yield singular / negative h2o_vmr via the
      # ``1 − q_v`` denominator.  Clipping the interior alone is not
      # enough — the halo cells are passed straight to the RRTMGP
      # solve.  Codex iter-79 stop-time review.
      q_v_3d = _add_halos(q_v[:, None, ::-1])
      q_v_3d = jnp.clip(q_v_3d, 0.0, 0.99)  # coeff-ok: specific-humidity cap

      # --- 2. Build VMR fields ---
      mol_ratio = constants.R_V / constants.R_D
      h2o_vmr = mol_ratio * q_v_3d / (1.0 - q_v_3d)

      if o3_vmr is not None:
          # Clip AFTER ``_add_halos``: linear halo extrapolation of a steep
          # boundary ozone profile (e.g. [1e-5, 1e-10]) can extrapolate to a
          # NEGATIVE halo VMR (2·1e-10 − 1e-5 < 0), which is an unphysical
          # absorber concentration in the optics lookup.  Same fix class as
          # the q_v post-clip above (codex atm-radiation review).
                  "mcica_cloud_fraction samples clouds per g-point inside the "
                  "flux solve; the optical-field-only outputs would return "
                  "unsampled in-cloud optics")
          from legoesm.atmosphere.physics.clouds.subcolumns import (
              apply_mask, generate_subcolumns)
          if mcica_mask_width is None:
              raise ValueError(
                  "mcica_cloud_fraction needs mcica_mask_width "
                  "(CloudConfig.cloud_subcolumn_mask_width; 0 = CAM's hard mask)")

          def _mcica_paths(n_gpt):
              mask = generate_subcolumns(mcica_cloud_fraction, n_gpt,
                                         shift=_mcica_shift, width=mcica_mask_width)
              if overhead:
                  # the overhead layer is cloud-free; sample the MODEL layers
                  # only, so each g-point's model-layer subcolumn is the one
                  # it draws with the layer off (merge of fu/halo with main's
                  # McICA: mcica_cloud_fraction stays (ncol, nlev))
                  mask = jnp.concatenate(
                      [jnp.zeros_like(mask[:, :, :1]), mask], axis=2)

              # snow: its cap weight is sampled with the path (a smooth
              # mask m gives m w min(p a, 50) = min(m w p a, 50 m w))
              _sampled = ((_cpl, _cpi) if cloud_snow is None
                          else (_cpl, _cpi, cloud_path_snow, cloud_snow_weight))

              def paths(igpt):
                  m = mask[igpt]
                  return tuple(
                      jnp.clip(_add_halos(
                          apply_mask(m, x)[:, None, ::-1]), 0.0, None)
                      for x in _sampled)
              return paths

          lw_path_fn = _mcica_paths(optics_lib.n_gpt_lw)
          sw_path_fn = _mcica_paths(optics_lib.n_gpt_sw)

      # Optional aerosol optical depth (shortwave).  Clip AFTER ``_add_halos``
      # (same fix class as q_v / o3 / cf / cloud paths): linear halo
      # extrapolation can drive a boundary aerosol OD negative, which is an
      # unphysical (negative) optical depth feeding the interior recurrence.
      if aerosol_optical_depth is not None:
          aerosol_od_3d = jnp.clip(
              _add_halos(aerosol_optical_depth[:, None, ::-1]), 0.0, None,
          )
      else:
          aerosol_od_3d = None

      # Aerosol SW single-scattering albedo + asymmetry: scalar (grey, default)
      # or per-shortwave-band arrays (config.aerosol_ssa_bands/aerosol_g_bands).
      # Resolved once here and shared by the MC-optics short-circuit and the
      # full SW solve so the per-band selection is defined in exactly one place.
      aer_ssa, aer_g = _resolve_aerosol_sw_optics(config, optics_lib)

      # Per-band aerosol component (volcanic): (ncol, nlev, 3, n_bnd) ->
      # (3, n_bnd, ncol, 1, nlev+2), the same halo / bottom-first layout as
      # the grey aerosol OD; the moments are linear in OD, so the linear halo
      # extrapolation + clip is the treatment the grey OD gets.
      if aerosol_volcanic_moments_sw is not None:
          n_bnd = int(optics_lib.gas_optics_sw.n_bnd)
          if aerosol_volcanic_moments_sw.shape[-2:] != (3, n_bnd):
              raise ValueError(
                  "aerosol_volcanic_moments_sw trailing shape "
                  f"{aerosol_volcanic_moments_sw.shape[-2:]}, expected (3, "
                  f"{n_bnd}) for this SW gas-optics table.")
          _nc, _nl = aerosol_volcanic_moments_sw.shape[:2]
          _vm = jnp.moveaxis(aerosol_volcanic_moments_sw, (2, 3), (0, 1))
          _vm = _add_halos(_vm[..., ::-1].reshape(3 * n_bnd * _nc, 1, _nl))
          volc_moments_3d = jnp.clip(
              _vm.reshape(3, n_bnd, _nc, 1, _nl + 2), 0.0, None)
      else:
          volc_moments_3d = None

      # Optional aerosol optical depth (longwave, pure absorber)
      if aerosol_absorption_optical_depth_lw is not None:
          aerosol_od_lw_3d = jnp.clip(
              _add_halos(aerosol_absorption_optical_depth_lw[:, None, ::-1]),
              0.0, None,
          )
      else:
          aerosol_od_lw_3d = None

      # Optional spectral solar forcing
    # ``for k in range(n_cols): np.interp(...)`` loop, which scaled
    # quadratically on T170+ (n_cols ~ 10⁴) at every external-forcing
    # update.
    nsrc = lat_src.shape[0]
    if nsrc == 1:
        result_flat = np.broadcast_to(
            field.reshape(1, -1), (lat_deg.size, np.prod(field.shape[1:])),
        ).copy()
    else:
        idx = np.clip(np.searchsorted(lat_src, lat_deg) - 1, 0, nsrc - 2)
        x0 = lat_src[idx]
        x1 = lat_src[idx + 1]
        # Avoid division by zero on duplicated source points.
        dx = np.where(x1 > x0, x1 - x0, 1.0)
        w = np.where(x1 > x0, (lat_deg - x0) / dx, 0.0)
        # Clamp at the endpoints (matches np.interp's flat extrapolation).
        w = np.clip(w, 0.0, 1.0)
        # Reshape ``field`` to (nsrc, n_cols) and gather along leading.
        trailing_shape = field.shape[1:]
        n_cols = int(np.prod(trailing_shape))
        field_flat = np.asarray(field).reshape(nsrc, n_cols)
        f0 = field_flat[idx]                    # (n_lat, n_cols)
        f1 = field_flat[idx + 1]                # (n_lat, n_cols)
        result_flat = (1.0 - w[:, None]) * f0 + w[:, None] * f1
    return jnp.array(result_flat).reshape((*lat_grid.shape, *field.shape[1:]))


def _interp_vertical(field_plev: jnp.ndarray, plev_src: np.ndarray,
                      p_target: jnp.ndarray) -> jnp.ndarray:
    """Interpolate vertically in log-pressure space.

    Parameters
    ----------
    field_plev : (..., nlev_src)
        Field on source pressure levels.
    plev_src : (nlev_src,)
        Source pressure levels [Pa].
    p_target : (..., nlev_target)
        Target pressure levels [Pa].

    Returns
    -------
    (..., nlev_target) interpolated field.
    """
    log_p_src = np.log(np.maximum(plev_src, 1e-10))
    log_p_tgt = jnp.log(jnp.maximum(p_target, 1e-10))

    shape_prefix = field_plev.shape[:-1]
    nlev_tgt = p_target.shape[-1]

exec
/usr/bin/bash -lc "ls /glade/work/pg2328 && ls -d /glade/derecho/scratch/pg2328/*venv* /glade/u/home/pg2328/*venv* 2>/dev/null && rg -n 'python|venv|pytest' docs/wb/reviews/2026-11-22_volcanic_optics_r1_prompt.md" in /glade/derecho/scratch/pg2328/wt_volc
 exited 2 in 0ms:
amip_inputs
conda-envs
cpuperf_env.sh
env_backups
envs
headroom-venv
legoesm-base0918
legoesm-cpuperf
legoesm-duofix
legoesm-ladder
legoesm-license
legoesm-thomas
mpiwrapper
ollama
venvs

exec
/usr/bin/bash -lc "rg -n 'aerosol_od|forcing\\._replace|SegmentForcing\\(|pack_forcing\\(' packages/coupler/legoesm/driver/compiled_segments.py packages/ml/legoesm/training --glob '*.py'" in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 0ms:
packages/coupler/legoesm/driver/compiled_segments.py:782:class SegmentForcing(NamedTuple):
packages/coupler/legoesm/driver/compiled_segments.py:796:    aerosol_od: jax.Array
packages/coupler/legoesm/driver/compiled_segments.py:798:    # stratospheric, gap #9).  Mirrors ``aerosol_od`` (the SHORTWAVE
packages/coupler/legoesm/driver/compiled_segments.py:800:    # ``aerosol_od`` (materialised to zeros by :func:`pack_forcing` when
packages/coupler/legoesm/driver/compiled_segments.py:881:    # (volcanic excluded).  ``None`` => CCN uses sum(aerosol_od) (unchanged).
packages/coupler/legoesm/driver/compiled_segments.py:888:    # no such source (the volcanic SW, if any, is inside ``aerosol_od``).
packages/coupler/legoesm/driver/compiled_segments.py:927:def pack_forcing(
packages/coupler/legoesm/driver/compiled_segments.py:929:    solar_weights, s_0, o3_vmr, aerosol_od,
packages/coupler/legoesm/driver/compiled_segments.py:1002:    # to zeros of the SAME shape as ``aerosol_od`` when None so the
packages/coupler/legoesm/driver/compiled_segments.py:1006:    _aer_od = jnp.asarray(aerosol_od)
packages/coupler/legoesm/driver/compiled_segments.py:1011:    return SegmentForcing(
packages/coupler/legoesm/driver/compiled_segments.py:1019:        aerosol_od=_aer_od,
packages/coupler/legoesm/driver/compiled_segments.py:1087:    "sst", "sic", "solar_weights", "o3_vmr", "aerosol_od", "aerosol_lw_od",
packages/coupler/legoesm/driver/compiled_segments.py:1159:    return forcing._replace(**updates) if updates else forcing
packages/coupler/legoesm/driver/compiled_segments.py:1605:        statics.forcing.o3_vmr, statics.forcing.aerosol_od,
packages/coupler/legoesm/driver/compiled_segments.py:1668:                statics.forcing.o3_vmr, statics.forcing.aerosol_od,
packages/coupler/legoesm/driver/compiled_segments.py:2594:                    forcing.o3_vmr, forcing.aerosol_od,
packages/coupler/legoesm/driver/compiled_segments.py:2646:                            forcing.o3_vmr, forcing.aerosol_od,
packages/coupler/legoesm/driver/compiled_segments.py:3086:                    forcing.o3_vmr, forcing.aerosol_od,
packages/coupler/legoesm/driver/compiled_segments.py:3120:                        forcing.o3_vmr, forcing.aerosol_od,
packages/coupler/legoesm/driver/compiled_segments.py:3175:                    forcing.o3_vmr, forcing.aerosol_od,
packages/coupler/legoesm/driver/compiled_segments.py:3209:                        forcing.o3_vmr, forcing.aerosol_od,
packages/ml/legoesm/training/sfno_dycore_coupling.py:217:            aerosol_od,
packages/ml/legoesm/training/sfno_dycore_coupling.py:268:            o3_vmr, aerosol_od,
packages/ml/legoesm/training/sfno_dycore_coupling.py:335:            solar_weights, s_0, o3_vmr, aerosol_od,
packages/ml/legoesm/training/scale_build.py:1582:    """``aerosol_od`` / ``aerosol_ccn_aod`` / ``aerosol_lw_od`` for one sample:
packages/ml/legoesm/training/scale_build.py:1612:    out = {"aerosol_od": od, "aerosol_ccn_aod": ccn, "aerosol_lw_od": lw}
packages/ml/legoesm/training/scale_build.py:1852:        forcing = pack_forcing(
packages/ml/legoesm/training/scale_build.py:1856:            o3_vmr=ctx["o3_vmr"], aerosol_od=ctx["aerosol_od"],

exec
/usr/bin/bash -lc "sed -n '140,360p' packages/ml/legoesm/training/sfno_dycore_coupling.py && ls /glade/work/pg2328/venvs /glade/work/pg2328/envs && sed -n '6120,6180p' packages/coupler/legoesm/driver/model_driver.py" in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 0ms:
        out = self.sfno(packed, self.grid)  # (n_lat, n_lon, n_channels)

        # Unpack tendency channels
        dT_dt = out[..., spec.T_slice]
        dq_v_dt = out[..., spec.q_slice]

        zeros_3d = jnp.zeros(T.shape, dtype=T.dtype)
        zeros_2d = jnp.zeros(p_s.shape, dtype=p_s.dtype)

        # Flux head: per-column MLP on the packed normalized state -> 4
        # radiation fluxes (sw_net_sfc, lw_net_sfc, sw_up_toa, lw_up_toa),
        # scaled to W/m^2.  None -> zeros (state-only).  precip / sw_down_toa
        # stay zero (insolation is an external forcing, not predicted here).
        if self.flux_head is not None:
            n_lat_g, n_lon_g, n_ch = packed.shape
            flat = packed.reshape(n_lat_g * n_lon_g, n_ch)
            f = jax.vmap(self.flux_head)(flat) * self.flux_output_scale
            f = f.reshape(n_lat_g, n_lon_g, 4)
            sw_net_sfc, lw_net_sfc = f[..., 0], f[..., 1]
            sw_up_toa, lw_up_toa = f[..., 2], f[..., 3]
        else:
            sw_net_sfc = lw_net_sfc = sw_up_toa = lw_up_toa = zeros_2d

        return PhysicsOutput(
            **_physics_output_kwargs(
                dT_dt=dT_dt,
                dq_v_dt=dq_v_dt,
                dq_c_dt=zeros_3d,
                dq_r_dt=zeros_3d,
                precip=zeros_2d,
                sw_net_sfc=sw_net_sfc,
                lw_net_sfc=lw_net_sfc,
                sw_up_toa=sw_up_toa,
                lw_up_toa=lw_up_toa,
                sw_down_toa=zeros_2d,
                reference_3d=T,
            )
        )


def make_sfno_step_unified(
    sfno_physics: SFNOPhysics,
    mode: Literal["correction", "replacement"] = "correction",
    traditional_step_unified=None,
):
    """Build a step_unified callable that incorporates the SFNO.

    Same signature as ``PhysicsPipeline.build_step_unified()`` output,
    so it can be passed directly to ``build_segment_fn``.

    Parameters
    ----------
    sfno_physics : SFNOPhysics
    mode : "correction" or "replacement"
    traditional_step_unified : callable or None
        Required when mode="correction".
    """
    if mode == "correction" and traditional_step_unified is None:
        raise ValueError(
            "traditional_step_unified is required for mode='correction'"
        )

    def step_unified(need_rad, T, p_s, q_v, q_c, q_r, *args, **kwargs):
        conv_prog, tail = _parse_step_unified_tail(args)
        (
            u,
            v,
            sst,
            sic,
            lat,
            lon,
            day_of_year,
            seconds_of_day,
            dt,
            solar_weights,
            s_0,
            o3_vmr,
            aerosol_od,
            held_dT_rad,
            held_sw_net_sfc,
            held_lw_net_sfc,
            held_sw_up_toa,
            held_lw_up_toa,
            held_sw_down_toa,
        ) = tail
        # SFNO tendency prediction from prognostic fields
        phis = kwargs.get("phis", jnp.zeros_like(p_s))
        # Prescribed planes (grid-shaped, already on the SFNO's own Gaussian
        # grid on this lane); a flag-on network names a missing one.
        land_frac = kwargs.get("land_frac")
        if sfno_physics.spatial_embedding and land_frac is None:
            raise ValueError(
                "SFNOPhysics(spatial_embedding=True) requires 'land_frac' in "
                "the step_unified kwargs, got None")
        sfc_fluxes = None
        if sfno_physics.era5_surface_fluxes:
            from legoesm.atmosphere.physics.neural_physics import (
                SFC_FLUX_STEP_UNIFIED_KEYS,
            )
            sfc_fluxes = []
            for key in SFC_FLUX_STEP_UNIFIED_KEYS:
                plane = kwargs.get(key)
                if plane is None:
                    raise ValueError(
                        f"SFNOPhysics(era5_surface_fluxes=True) requires "
                        f"{key!r} in the step_unified kwargs, got None")
                sfc_fluxes.append(plane)
            sfc_fluxes = tuple(sfc_fluxes)
        sfno_out = sfno_physics(T, u, v, q_v, p_s, phis, dt,
                                land_frac=land_frac, sfc_fluxes=sfc_fluxes)
        if "conv_prog" in _PHYSICS_OUTPUT_FIELDS and conv_prog is not None:
            sfno_out = sfno_out._replace(conv_prog=conv_prog)

        if mode == "replacement":
            held_new = (
                held_dT_rad, held_sw_net_sfc, held_lw_net_sfc,
                held_sw_up_toa, held_lw_up_toa, held_sw_down_toa,
            )
            return sfno_out, held_new

        # mode == "correction": traditional physics + SFNO correction
        trad_args = [need_rad, T, p_s, q_v, q_c, q_r]
        if conv_prog is not None:
            trad_args.append(conv_prog)
        trad_args.extend([
            u, v, sst, sic, lat, lon,
            day_of_year, seconds_of_day, dt,
            solar_weights, s_0,
            o3_vmr, aerosol_od,
            held_dT_rad, held_sw_net_sfc, held_lw_net_sfc,
            held_sw_up_toa, held_lw_up_toa, held_sw_down_toa,
        ])
        _trad = traditional_step_unified(*trad_args, **kwargs)
        trad_out, held_new = _trad[0], _trad[1]
        # The traditional PhysicsPipeline step returns a 3rd value (the
        # slab-land skin temperature, #325); carry it through so an
        # SFNO-correction run with an active land tile still evolves
        # T_land.  Older 2-tuple steps leave it None (land inert).
        _trad_T_land = _trad[2] if len(_trad) > 2 else None

        corrected = PhysicsOutput(
            **_physics_output_kwargs(
                dT_dt=trad_out.dT_dt + sfno_out.dT_dt,
                dq_v_dt=trad_out.dq_v_dt + sfno_out.dq_v_dt,
                dq_c_dt=trad_out.dq_c_dt + sfno_out.dq_c_dt,
                dq_r_dt=trad_out.dq_r_dt + sfno_out.dq_r_dt,
                precip=trad_out.precip,
                sw_net_sfc=trad_out.sw_net_sfc,
                lw_net_sfc=trad_out.lw_net_sfc,
                sw_up_toa=trad_out.sw_up_toa,
                lw_up_toa=trad_out.lw_up_toa,
                sw_down_toa=trad_out.sw_down_toa,
                reference_3d=trad_out.dT_dt,
                template=trad_out,
                conv_prog=conv_prog,
            )
        )
        return corrected, held_new, _trad_T_land

    return step_unified


def make_sfno_step_unified_latlon(
    sfno_physics: SFNOPhysics,
    w_latlon_to_gauss,
    w_gauss_to_latlon,
):
    """step_unified for a LAT-LON model grid with the SFNO on a Gaussian grid.

    The SFNO's spherical-harmonic transforms need Gaussian quadrature
    latitudes, so the model's lat-lon prognostics are remapped onto the
    SFNO's own Gaussian grid, the SFNO predicts tendencies there, and the
    dT/dt, dq_v/dt tendencies are remapped back to the lat-lon grid.
    Replacement mode only (the SFNO IS the physics; there is no lat-lon
    traditional pipeline to correct against in the WB scale trainer).

    Parameters
    ----------
    sfno_physics : SFNOPhysics
        Wrapper whose ``grid`` is the Gaussian grid the weights target.
    w_latlon_to_gauss, w_gauss_to_latlon : RegridWeights
        Precomputed IDW weights (``compute_latlon_to_voronoi_weights``)
        with ``target_shape`` set to the 2-D destination grid shape, so
        ``regrid_scalar`` returns (n_lat, n_lon[, nlev]) fields directly.
    """
    from legoesm.atmosphere.physics.neural_physics import (
        SFC_FLUX_STEP_UNIFIED_KEYS,
    )
    from legoesm.grids.regridding import regrid_scalar

    def step_unified(need_rad, T, p_s, q_v, q_c, q_r, *args, **kwargs):
        conv_prog, tail = _parse_step_unified_tail(args)
        (
            u, v, sst, sic, lat, lon,
            day_of_year, seconds_of_day, dt,
            solar_weights, s_0, o3_vmr, aerosol_od,
            held_dT_rad, held_sw_net_sfc, held_lw_net_sfc,
            held_sw_up_toa, held_lw_up_toa, held_sw_down_toa,
        ) = tail

        # Legacy callers may omit phis (zero orography); the trainer passes
        # the carry's surface geopotential.
        phis = kwargs.get("phis", jnp.zeros_like(p_s))

        def to_gauss(f):
            return regrid_scalar(f, w_latlon_to_gauss)

        # Prescribed input planes for SFNOPhysics's extra channels: the
        # driver-path kwargs regridded to the Gaussian grid with the same
        # IDW weights as the prognostics (driver keys mapped onto the
        # SFC_FLUX_FORCING_KEYS order via SFC_FLUX_STEP_UNIFIED_KEYS).  A
        # flag on with its keyword missing is an error naming it — never
        # silent zeros.  Flags off -> nothing extra regridded/passed.
        land_frac = kwargs.get("land_frac")
        if sfno_physics.spatial_embedding:
            if land_frac is None:
                raise ValueError(
                    "SFNOPhysics(spatial_embedding=True) requires "
                    "'land_frac' in the step_unified kwargs, got None")
            land_frac = to_gauss(land_frac)
        if sfno_physics.era5_surface_fluxes:
/glade/work/pg2328/envs:
headroom

/glade/work/pg2328/venvs:
context-compress
                # With p_top = 0 there is no overhead layer: it keeps the top
                # model layer's value, and the extra layer's 0/0 is never formed
                # (p_top replaced by 1 Pa only inside that branch).
                from legoesm.atmosphere.physics.radiation.ozone_mls import (
                    mls_ozone_layer_vmr,
                )
                _lay = _compute_ozone_vmr(
                    p_full_col, lat_col, OzoneProfileConfig(source="mls"),
                    p_half=p_half_col,
                )
                _ptop = p_half_col[:, 0]
                _ph_safe = p_half_col.at[:, 0].set(jnp.where(_ptop > 0.0, _ptop, 1.0))
                _above = jnp.where(_ptop > 0.0, mls_ozone_layer_vmr(_ph_safe)[1],
                                   _lay[:, 0])
                _o3_inline = jnp.concatenate(
                    [_lay, jnp.broadcast_to(_above[:, None], (ncol, _p_o3.shape[1] - nlev))],
                    axis=1)
            else:
                _o3_inline = _compute_ozone_vmr(
                    _p_o3, lat_col,
                    OzoneProfileConfig(source=self.config.ozone_source),
                )
            if _o3_inline is None:
                # source="standard": the backend contract is o3_vmr=None →
                # its built-in US Std Atm 1976 fit; materialize the same
                # profile for the stable-signature override path.
                from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import (
                    standard_o3_profile,
                )
                _o3_inline = standard_o3_profile(_p_o3)
            o3_vmr = jnp.asarray(_o3_inline).astype(p_s.dtype)
        self._o3_top_vmr = jnp.mean(o3_vmr[:, nlev:], axis=1)
        o3_vmr = o3_vmr[:, :nlev]

        # Radiation and the droplet proxy read the aerosol at each column's
        # own longitude, as CAM's prescribed aerosol reaches radiation
        # (tracer_data.F90 bilinear lon/lat lininterp).
        lon_col = (jnp.asarray(self._owned_lon() if lon is None else lon)
                   .reshape(ncol) if self._aerosol_active else None)
        (aerosol_od, _ccn_aod, self._aerosol_lw_od,
         self._aerosol_sw_volc) = aerosol_layer_fields(
            self._aerosol_config, day, lat_col, p_half_col, w_strat,
            active=self._aerosol_active, lw_active=self._aerosol_lw_active,
            dtype=p_s.dtype, lon_col=lon_col)
        if _ccn_aod is not None:
            self._aerosol_ccn_aod = _ccn_aod

        ghg_vmr = self.radiation_ghg_vmr(day)
        return o3_vmr, aerosol_od, ghg_vmr

    def radiation_ghg_vmr(self, day):
        """GHG mole fractions radiation uses on ``day`` (None = fixed config gases)."""
        from legoesm.forcing.external import (
            get_ghg_at_time, ghg_concentrations_to_vmr,
        )
        # GHG VMR override (None for gray radiation / constant forcing)
        ghg_vmr = None
        if self._ghg_active:
            ghg_conc = get_ghg_at_time(self._ghg_config, day)
            ghg_vmr = ghg_concentrations_to_vmr(ghg_conc)


exec
/usr/bin/bash -lc "ls /glade/work/pg2328/legoesm-base0918/bin/python /glade/work/pg2328/legoesm-thomas/bin/python /glade/work/pg2328/conda-envs/*/bin/python /glade/work/pg2328/headroom-venv/bin/python 2>/dev/null && rg -n 'PYTHONPATH|legoesm-base|legoesm-thomas|conda|venv' docs/wb/reviews/2026-11-22_volcanic_optics_r1_prompt.md tests/conftest.py pyproject.toml" in /glade/derecho/scratch/pg2328/wt_volc
 exited 2 in 0ms:
/glade/work/pg2328/conda-envs/legoesm-gpu/bin/python
/glade/work/pg2328/headroom-venv/bin/python

exec
/usr/bin/bash -lc 'JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONDONTWRITEBYTECODE=1 /glade/work/pg2328/conda-envs/legoesm-gpu/bin/python -m pytest -q -p no:cacheprovider tests/unit/test_volcanic_sw_optics.py' in /glade/derecho/scratch/pg2328/wt_volc
exec
/usr/bin/bash -lc "rg -n 'solve_sw\\(|compute_sw_optical_props_gpt\\(|compute_sw_optical_field\\(' packages --glob '*.py' && sed -n '658,746p' packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rte/two_stream.py && sed -n '1095,1165p' packages/coupler/legoesm/driver/compiled_segments.py" in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 0ms:
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rte/two_stream.py:617:def compute_sw_optical_props_gpt(
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rte/two_stream.py:755:def compute_sw_optical_field(
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rte/two_stream.py:787:    props = compute_sw_optical_props_gpt(
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rte/two_stream.py:867:def solve_sw(
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rte/two_stream.py:1019:      sw_optical_props = compute_sw_optical_props_gpt(
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:1185:          sw_props = two_stream.compute_sw_optical_field(
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:1279:      sw_fluxes = two_stream.solve_sw(
      aerosol_volcanic_moments)


def _mix_sw_aerosol_and_clip(
    igpt: Array,
    optics_lib,
    sw_optical_props: dict[str, Array],
    aerosol_optical_depth: Array | None,
    aerosol_single_scattering_albedo: float | Array,
    aerosol_asymmetry_factor: float | Array,
    aerosol_volcanic_moments: Array | None = None,
) -> dict[str, Array]:
  """Mix aerosol into one g-point's SW gas(+cloud) optics, then clip ssa/g.

  ``aerosol_volcanic_moments`` ``(3, n_bnd_sw, *layer_shape)``: a second
  aerosol component with its OWN per-band optics, given as scattering
  moments ``(tau, tau*ssa, tau*ssa*g)`` (stratospheric volcanic sulfate from
  the CMIP6 file); added to the same tau / tau*ssa / tau*ssa*g sums as the
  grey aerosol, so the mix is the exact optical-depth-weighted combination.
  """
  if aerosol_optical_depth is None and aerosol_volcanic_moments is not None:
    aerosol_optical_depth = jnp.zeros_like(sw_optical_props['optical_depth'])
  if aerosol_optical_depth is not None:
    # Per-band aerosol optics: ``aerosol_single_scattering_albedo`` /
    # ``aerosol_asymmetry_factor`` may be a SCALAR (grey aerosol, historical
    # default -> byte-identical) OR a ``(n_bnd_sw,)`` array of per-shortwave-band
    # values (real aerosols absorb/scatter very differently in the UV/visible vs
    # the near-IR).  When per-band, select this g-point's band value via the
    # SAME g-point->band map the LW diffusivity path uses (g_point_to_bnd);
    # a 0-D input broadcasts unchanged.
    ssa_aer = jnp.asarray(aerosol_single_scattering_albedo)
    g_aer = jnp.asarray(aerosol_asymmetry_factor)
    if ssa_aer.ndim >= 1 or g_aer.ndim >= 1:
      band_idx = optics_lib.gas_optics_sw.g_point_to_bnd[igpt]
      if ssa_aer.ndim >= 1:
        ssa_aer = ssa_aer[band_idx]
      if g_aer.ndim >= 1:
        g_aer = g_aer[band_idx]
    tau_bg = jnp.maximum(sw_optical_props['optical_depth'], 1.0e-12)
    tau_aer = jnp.maximum(aerosol_optical_depth, 0.0)
    tau_tot = tau_bg + tau_aer
    w_bg = sw_optical_props['ssa']
    g_bg = sw_optical_props['asymmetry_factor']
    w_num = tau_bg * w_bg + tau_aer * ssa_aer
    if aerosol_volcanic_moments is not None:
      volc = aerosol_volcanic_moments[:, optics_lib.gas_optics_sw.g_point_to_bnd[igpt]]
      tau_tot = tau_tot + volc[0]
      w_num = w_num + volc[1]
    # AD-safe SW optical-property mixing (restored from commit 59407953
    # after AIMIP-#312 merge reverted it).  ``a / jnp.maximum(b, eps)``
    # has a ``-a/b**2`` VJP that overflows when ``b`` is at the floor —
    # for cloud-free, low-water-vapor stratospheric layers ``tau_tot``
    # can reach the 1e-12 floor and the backward propagates NaN to every
    # upstream traced parameter whose state path touches gas absorption.
    # ``safe_divide`` masks the bad branch before the divide.  The outer
    # ``jnp.clip`` preserves the original output range; with ``fill=0.0``
    # the bad branch lands inside that range.
    w_tot = jnp.clip(
        safe_divide(w_num, tau_tot, eps=1.0e-12, fill=0.0),
        0.0,
        1.0,
    )
    g_num = (
        tau_bg * w_bg * g_bg
        + tau_aer * ssa_aer * g_aer
    )
    if aerosol_volcanic_moments is not None:
      g_num = g_num + volc[2]
    g_denom = tau_tot * jnp.maximum(w_tot, 1.0e-12)
    g_tot = jnp.clip(
        safe_divide(g_num, g_denom, eps=1.0e-12, fill=0.0),
        -1.0,
        1.0,
    )
    sw_optical_props = {
        'optical_depth': tau_tot,
        'ssa': w_tot,
        'asymmetry_factor': g_tot,
    }
  # Bound the single-scattering albedo / asymmetry to their physical ranges
  # before downstream use.  The aerosol-mixing branch above already clips
  # ssa∈[0,1] / g∈[-1,1], but with no aerosol forcing the cloud optics feed
  # through UNCLAMPED — and on a drifted coupled state the cloud
  # parameterisation can emit ssa>1 / |g|>1, which drives the Meador-Weaver
  # reflectance/transmittance above 1 so the two-stream AMPLIFIES the flux
  # (super-physical TOA SW down; BUG-B, 2026-06-15).  Physical optics are
  # already in range ⇒ a no-op for valid inputs.
  sw_optical_props = {
      **sw_optical_props,

def shard_forcing(forcing: SegmentForcing, device_config) -> SegmentForcing:
    """Commit the grid-shaped ``SegmentForcing`` leaves to the SAME device
    layout as the sharded state (SPMD sharding step 3 of the production
    cs_spmd design).

    ``run_segment``'s sharded JIT wrapper honours a committed
    ``NamedSharding`` on the build-time mesh per leaf and pins anything
    else REPLICATED (``_input_sharding``).  Replicated forcing means every
    device receives the FULL global forcing array at each segment
    boundary (host->device bandwidth and memory x n_devices) and GSPMD
    re-slices it against the face-sharded state inside the step.
    Committing the grid-shaped leaves with the state's own sharding rules
    (:func:`legoesm.parallel.mesh.shard_pytree` — face-first on
    cubed-sphere, lat-axis on lat-lon, tiling-aware) makes the transfer
    and memory per device ``1/n_devices`` and removes the reshard.

    No-ops (returns ``forcing`` unchanged) when: ``device_config`` is
    ``None`` (byte-identical single-device contract), the mesh has no
    face sharding (single device), or the run is mpi4jax-distributed
    (``is_distributed`` — forcing there is rank-local, never device-mesh
    sharded).  ``None`` optional fields and non-grid leaves pass through
    untouched; a grid-NAMED leaf whose shape does not match the grid rule
    (e.g. the ``(0,)`` o3 placeholder) is left alone by ``shard_pytree``'s
    shape gate.
    """
    if device_config is None or getattr(device_config, "is_distributed", False):
        return forcing
    if getattr(device_config, "face_sharding", None) is None:
        return forcing
    from legoesm.parallel.mesh import shard_pytree

    # Production cubed-sphere packs the 3-D radiation forcing FLATTENED to
    # ``(ncol=6*n*n, nlev)`` (``_precompute_external_forcing``), which the
    # leading-6 ``shard_pytree`` rule cannot see (codex 2026-07-01 Medium).
    # A flat face-major leaf shards ``P("face")`` on dim 0 — the exact
    # layout the driver already commits for flat carry leaves (see
    # ``_input_sharding``: "a flat [6*n*n, nlev] carry leaf the driver
    # placed on P('face')").  ``ncol % 6 == 0`` chunks land exactly one
    # face's columns per face-mesh slot; the face axis size divides 6 so
    # divisibility always holds.  Gate: cubed-sphere only, named grid
    # field, flat (shape[0] a positive non-6 multiple of 6).
    _flat_face = None
    if getattr(device_config, "grid_type", None) == "cubed_sphere" and \
            getattr(device_config, "mesh", None) is not None:
        from jax.sharding import NamedSharding, PartitionSpec
        _flat_face = NamedSharding(device_config.mesh, PartitionSpec("face"))

    updates = {}
    for name in GRID_SHAPED_FORCING_FIELDS:
        leaf = getattr(forcing, name)
        if leaf is None or not isinstance(leaf, (jax.Array, jnp.ndarray)):
            continue
        if (_flat_face is not None and leaf.ndim >= 1
                and leaf.shape[0] != 6 and leaf.shape[0] > 0
                and leaf.shape[0] % 6 == 0):
            # multiprocess_safe_device_put, NOT jax.device_put: under
            # multi-controller SPMD the cross-process bit-equality assert
            # in device_put trips on host-precomputed forcing (#693,
            # Levante 2-node receipt job 26030677).
            from legoesm.parallel.mesh import multiprocess_safe_device_put
            updates[name] = multiprocess_safe_device_put(leaf, _flat_face)
        else:
            updates[name] = shard_pytree(leaf, device_config)
    return forcing._replace(**updates) if updates else forcing


@dataclass(frozen=True)
class _SplitStepStatics:
    """Closure statics for the operator-split physics body, bundled so the
    single-rank physics + the shared finalizer live at MODULE SCOPE and can be

exec
/usr/bin/bash -lc "head -80 tests/conftest.py && rg -n 'sw_optical_field_only|lw_optical_field_only|return RadiationOutput|clear_sky' packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py && git diff HEAD --numstat" in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 78ms:
"""Shared test fixtures for legoESM."""

import functools
import os
import resource
import sys
from pathlib import Path

#: Cap the CPU affinity mask BEFORE JAX is imported (below).  XLA sizes its
#: Eigen thread pool from the affinity mask, so on a many-core login node a
#: single JAX process opens far more threads than ``ulimit -u`` allows once a
#: few run at once, and the pool constructor aborts the interpreter:
#:
#:   F env.cc:93] Check failed: ret == 0 (11 vs. 0) Thread tf_XLAPjRtCpuClient
#:   creation via pthread_create() failed.
#:     @ Eigen::ThreadPoolTempl<>::ThreadPoolTempl()
#:
#: (11 = EAGAIN.)  It surfaces as "Fatal Python error: Aborted" mid-run, which
#: reads like a test failure but is the process running out of thread slots.
#:
#: Measured on a 256-core Levante login node (``ulimit -u`` = 2048), threads
#: opened by one client for a trivial jitted matmul:
#:
#:     CPUs    4     8    16    32    64   256
#:     threads 29    45    77   141   237   621
#:
#: Uncapped that is 621 threads each, so FOUR concurrent clients (2484) are the
#: first count over the 2048 ceiling; five were observed to fail 5/5.  At 32
#: CPUs a client takes ~141, so ~14 fit.  Capping costs nothing measurable --
#: these tests are XLA-compile-bound, not intra-op-parallelism-bound
#: (tests/test_corner_div_damp_nh.py: 166.9 s on 256 CPUs, 166.3 s on 8,
#: 152.7 s on 32).
#:
#: Set ``LEGOESM_TEST_CPU_CAP`` to a positive integer to force a cap, or to 0
#: (or any value <= 0) to disable and inherit the machine's full affinity mask.
_DEFAULT_TEST_CPU_CAP = 32

#: Threads one XLA CPU client opens ~= _THREADS_PER_CPU * ncpu + _THREADS_BASE,
#: fitted to the table above (slope (621-29)/(256-4) = 2.35).  Used to size the
#: cap against the ACTUAL thread ceiling rather than trusting a fixed 32, which
#: is only evidenced safe for ulimit -u = 2048.
_THREADS_PER_CPU = 2.4
_THREADS_BASE = 20.0
#: Fraction of the thread ceiling the test session may claim, leaving room for
#: the user's other processes (a login node already carries ~50 threads).
_RLIMIT_SHARE = 0.5

#: Env vars every common launcher sets in a rank's environment.  Under MPI the
#: launcher (or the batch scheduler) owns placement: each rank already gets its
#: own binding, and clamping every rank to the SAME low-numbered CPUs here would
#: pile all of them onto one core set and serialise the run.  So the cap is
#: skipped for MPI ranks -- they are separate processes with a launcher-assigned
#: mask, which is exactly the placement this function would otherwise destroy.
#: MPI does not standardise a rank env var, so detection is BEST-EFFORT: this
#: covers the launchers in use here.  For an unlisted launcher, disable the cap
#: explicitly with ``LEGOESM_TEST_CPU_CAP=0``.
_MPI_RANK_ENV = (
    "OMPI_COMM_WORLD_RANK",     # Open MPI
    "PMI_RANK",                 # MPICH / Intel MPI
    "PMIX_RANK",                # PMIx
    "SLURM_PROCID",             # srun-launched
    "MV2_COMM_WORLD_RANK",      # MVAPICH2
    "MPI_RANKID",               # IBM Platform MPI
    "LAMRANK",                  # legacy LAM/MPI
)


def _is_mpi_rank() -> bool:
    """True when this interpreter was launched as a rank of an MPI job."""
    return any(v in os.environ for v in _MPI_RANK_ENV)


def _requested_cap() -> int | None:
    """The user's explicit ``LEGOESM_TEST_CPU_CAP``, or None when unset/garbage.

    A value <= 0 disables capping entirely and is returned as 0.
    """
    raw = os.environ.get("LEGOESM_TEST_CPU_CAP")
    if raw is None:
        return None
608:      sw_optical_field_only: bool = False,
609:      lw_optical_field_only: bool = False,
613:      clear_sky: bool = False,
806:                  or sw_optical_field_only or lw_optical_field_only):
845:                  and not (sw_optical_field_only or lw_optical_field_only))
1078:          if sw_optical_field_only or lw_optical_field_only:
1178:      # default path (sw_optical_field_only=False) is byte-identical.
1179:      if clear_sky and (sw_optical_field_only or lw_optical_field_only):
1181:              "clear_sky adds clear-sky TOA fluxes to the RadiationOutput "
1184:      if sw_optical_field_only:
1226:      # (lw_optical_field_only=False) is byte-identical.
1227:      if lw_optical_field_only:
1275:          clear_sky=clear_sky,
1300:          clear_sky=clear_sky,
1332:      return RadiationOutput(
1342:                              if clear_sky else None),
1344:                              if clear_sky else None),
1417:      if opt.get("sw_optical_field_only") or opt.get("lw_optical_field_only"):
1420:              "sw_optical_field_only / lw_optical_field_only return a "
1	1	packages/atmosphere/legoesm/atmosphere/physics/clouds/subcolumns.py
23	0	packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py
35	0	packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py
28	3	packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rte/two_stream.py
17	1	packages/coupler/legoesm/driver/compiled_segments.py
14	1	packages/coupler/legoesm/driver/model_driver.py
18	3	packages/coupler/legoesm/driver/physics_pipeline.py
8	2	packages/ml/legoesm/training/scale_build.py
161	45	packages/tools/legoesm/forcing/external.py
47	19	packages/tools/legoesm/forcing/surface_utils.py
89	0	scripts/validate/probe_volcanic_sw_optics.py
26	22	tests/unit/test_aerosol_band_ccn.py
1	1	tests/unit/test_cross_lane_forcing.py
294	0	tests/unit/test_volcanic_sw_optics.py
1	1	tests/unit/test_wb_amip_deck_params.py

exec
/usr/bin/bash -lc "sed -n '5820,5900p' packages/coupler/legoesm/driver/model_driver.py && sed -n '1390,1475p' packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py && rg -n 'sys.path|packages|src' tests/conftest.py && sed -n '951,964p' CLAUDE.md" in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 0ms:
                and not cfg.ozone_file):
            # the loader would substitute its built-in reference profile
            # (use_reference_if_missing below) for the input4MIPs file
            raise ValueError(
                "radiation='ecrad_ifs' reads the input4MIPs ozone file; "
                "ozone_file is empty (--ozone-file).")
        self._ozone_ext_config = OzoneConfig(
            enabled=self._ozone_ext_active,
            source="climatology", path=cfg.ozone_file,
            use_reference_if_missing=True,
            # Required for the non-cyclic branch in ``get_ozone_at_time``
            # so that CMIP6 multi-year ozone files (>12 months, e.g. the
            # UReading 1850–2014 vmro3 file) are sampled at the actual
            # simulation calendar year instead of falling back to a
            # 1850 climatology.
            start_year=cfg.start_year,
            hold_outside_file_years=cfg.forcing_hold_outside_file_years,
        )

        # Aerosol external forcing
        self._aerosol_active = (cfg.radiation in ("rrtmg", "rrtmgp")
                                and cfg.aerosol_forcing == "external")
        # Volcanic stratospheric LONGWAVE aerosol (gap #9): active only on a
        # gas-radiation scheme (gray ignores aerosol) with the LW switch and
        # a volcanic file present.  Default OFF ⇒ no LW aerosol path.
        self._aerosol_lw_active = (
            cfg.radiation in ("rrtmg", "rrtmgp")
            and bool(cfg.volcanic_aerosol_lw)
            and bool(cfg.volcanic_aerosol_file)
        )
        self._aerosol_lw_od = None
        self._aerosol_sw_volc = None
        self._o3_top_vmr = None
        self._aerosol_ccn_aod = None
        self._aerosol_config = AerosolConfig(
            enabled=self._aerosol_active,
            source="climatology", path=cfg.aerosol_file,
            ccn_path=cfg.aerosol_ccn_file,
            use_reference_if_missing=True,
            reference_aod_550=cfg.aerosol_reference_aod,
            volcanic_enabled=bool(cfg.volcanic_aerosol_file),
            volcanic_path=cfg.volcanic_aerosol_file,
            volcanic_scale=cfg.volcanic_aerosol_scale,
            # Volcanic stratospheric LONGWAVE aerosol (gap #9): only when
            # the LW switch is set AND a volcanic file is present.  Default
            # OFF ⇒ ``get_aerosol_lw_at_time`` returns None ⇒ zeros LW od
            # ⇒ byte-identical (RRTMGP no-op).
            volcanic_lw_enabled=(bool(cfg.volcanic_aerosol_lw)
                                 and bool(cfg.volcanic_aerosol_file)),
            # Calendar anchor for the non-cyclic dispatch in
            # ``get_aerosol_at_time``.  Multi-year volcanic time-series
            # (e.g. 1850–2014 CMIP6 ``bc_aeropt_cmip6_volc_*``) are
            # sampled at their actual eruption calendars instead of
            # being collapsed onto a 12-month cycle.
            start_year=cfg.start_year,
            hold_outside_file_years=cfg.forcing_hold_outside_file_years,
        )

        # Solar init
        solar_init = get_solar_forcing_at_time(self._solar_config, cfg.start_day)
        if self._use_solar_spectral:
            self._solar_weights_template = jnp.asarray(
                solar_init["solar_fraction_by_gpt"]
            )
        else:
            self._solar_weights_template = jnp.array([], dtype=jnp.float32)

        # CMIP experiment GHG override
        self._experiment = cfg.experiment
        self._start_year = cfg.start_year
        if self._experiment:
            from legoesm.forcing.experiments import ghg_at_year
            co2, ch4, n2o = ghg_at_year(self._experiment, self._start_year)
            self.config = cfg._replace(co2_ppmv=co2, ch4_ppbv=ch4, n2o_ppbv=n2o)
            cfg = self.config
            logger.info(f"  CMIP: {self._experiment} (year {self._start_year}), "
                  f"CO2={co2:.1f} ppmv")

        # GHG forcing config
        self._ghg_active = (cfg.radiation in ("rrtmg", "rrtmgp", "ecrad_ifs")
                            and cfg.ghg_forcing == "external")
          alongside the required inputs; everything else (Python/0-D
          scalars, ``None``, the ``ghg_vmr_override`` dict, the per-g-point
          ``solar_spectral_fraction`` whose leading axis is ``n_gpt`` not
          ``ncol``) is broadcast unchanged to every block.

      Returns
      -------
      RadiationOutput
          Reassembled over ``ncol`` — identical to :meth:`solve_columns`.
      """
      ncol = T.shape[0]
      # Disabled / degenerate → exact plain call.  This precedes every
      # chunking-specific check so a disabled call is a TRUE passthrough:
      # it honours any optics-only flags and returns exactly what
      # solve_columns returns (no reshape overhead, no contract change).
      if (not column_chunk_size or column_chunk_size <= 0
              or ncol <= column_chunk_size):
          return self.solve_columns(
              T=T, p_full=p_full, p_half=p_half,
              sfc_temperature=sfc_temperature, q_v=q_v,
              cos_zenith=cos_zenith, **opt,
          )
      # --- chunking is ACTIVE from here down ---
      # The optics-only short-circuits return a per-g-point TUPLE whose leaf
      # layout ((n_gpt, ncol, nlev)) carries no leading (n_block, chunk)
      # column axis, so the reassembly reshape below would corrupt it.
      # Chunking supports the RadiationOutput path only.
      if opt.get("sw_optical_field_only") or opt.get("lw_optical_field_only"):
          raise ValueError(
              "solve_columns_chunked supports only the RadiationOutput path; "
              "sw_optical_field_only / lw_optical_field_only return a "
              "per-g-point tuple incompatible with column chunking — call "
              "solve_columns directly for the optics-only tracers."
          )
      if ncol % column_chunk_size != 0:
          raise ValueError(
              f"column_chunk_size {column_chunk_size} must divide ncol "
              f"{ncol} exactly (radiation columns are independent, but the "
              "lax.map reshape requires equal-size blocks). Note that ncol "
              "here is the SOLVER's column count: with max-random overlap "
              "it is cloud_n_subcolumns * the grid's column count, so a "
              "block size chosen against the grid alone need not divide it."
          )
      n_block = ncol // column_chunk_size

      # Per-column CONFIG surface fallbacks (sfc_albedo / sfc_emissivity /
      # sfc_albedo_direct) are read at GLOBAL ncol inside solve_columns via
      # ``_resolve_surface_field(override, self._config.<field>)`` whenever the
      # explicit per-call override is None.  Only EXPLICIT overrides (in opt)
      # are chunked here, so a per-column ``(ncol, ...)`` config fallback would
      # be read at global ncol inside each chunk-local solve → size mismatch
      # and a broken / non-exact result.  Fail loudly and tell the caller to
      # pass the field explicitly (which IS chunked); the production AMIP path
      # already passes sfc_albedo / sfc_emissivity as explicit args, so this
      # never fires there — it protects the direct-call contract + exactness.
      cfg = getattr(self, "_config", None)
      for _sfc_name in ("sfc_albedo", "sfc_emissivity", "sfc_albedo_direct"):
          _fallback = getattr(cfg, _sfc_name, None)
          if (opt.get(_sfc_name) is None and hasattr(_fallback, "shape")
                  and getattr(_fallback, "ndim", 0) >= 1
                  and _fallback.shape[0] == ncol):
              raise ValueError(
                  f"solve_columns_chunked: config fallback '{_sfc_name}' is a "
                  f"per-column ({ncol},...) array but no explicit '{_sfc_name}' "
                  "override was passed. The config surface field is read at "
                  "global ncol inside each chunk block (via "
                  "_resolve_surface_field) → size mismatch under chunking. "
                  f"Pass '{_sfc_name}' as an explicit per-call argument (it is "
                  "chunked) instead of relying on the per-column config field."
              )

      # Split every per-column array on its leading axis into
      # (n_block, column_chunk_size, ...).  This row-major reshape is
      # order-preserving and the reassembly reshape below is its exact
      # inverse, so no column is reordered — the exactness guarantee.
      def _is_col_array(x):
          return (hasattr(x, "shape") and getattr(x, "ndim", 0) >= 1
                  and x.shape[0] == ncol)

      def _split(x):
          return x.reshape(n_block, column_chunk_size, *x.shape[1:])

      mapped = {
          "T": _split(T), "p_full": _split(p_full), "p_half": _split(p_half),
          "sfc_temperature": _split(sfc_temperature), "q_v": _split(q_v),
          "cos_zenith": _split(cos_zenith),
191:    from jax._src.nn import functions as _nnf
## Commands
- Install: `pip install -e ".[dev]"`
- Tests: `.venv/bin/python -m pytest tests/`
- Sci tests: `JAX_ENABLE_X64=1 .venv/bin/python -m pytest <target>`
- Atm matrix: `JAX_ENABLE_X64=1 .venv/bin/python scripts/matrix/run_atmosphere_test_matrix.py`
- Ocean matrix: `JAX_ENABLE_X64=1 .venv/bin/python scripts/matrix/run_ocean_test_matrix.py`
- AMIP: `.venv/bin/python scripts/run/run_amip.py`
- Dycore progression: `JAX_ENABLE_X64=1 .venv/bin/python scripts/matrix/run_atmosphere_test_matrix.py` (SW -> hydrostatic -> non-hydrostatic ladder; `--only sw|hydro|nh`). Old `tests/validation/run_dycore_progression_suite.py` superseded — its per-case child scripts removed.
- GPU/MPI scaling: `.venv/bin/python scripts/bench/run_levante_gpu_scaling.py --grid cubed-sphere --mode strong` (`docs/performance/REAL_HARDWARE_SCALING.md`)
- Scripts reorganized into buckets: `scripts/{run,matrix,bench,plot,validate,data,experiment,cluster}/`; debug in `scripts/tmp/`. See `scripts/README.md` + `## File Layout` below.
- MPI tests: `mpirun -np 2 .venv/bin/python -m pytest tests/distributed/`
- MPI diff: `mpirun -np 2 .venv/bin/python -m pytest tests/distributed/test_mpi_differentiability.py`

## File Layout (audit — enforce on EVERY new file; no random files)

codex
The forcing paths I’ve inspected carry the new field through, and it stays out of the grey aerosol and CCN inputs. The worktree has no `.venv`; I’m checking a shared environment before relying on its test results.
exec
/usr/bin/bash -lc "JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 /glade/work/pg2328/conda-envs/legoesm-gpu/bin/python -c 'import legoesm.forcing.external as e; print(e.__file__)' && ls -l src/legoesm/forcing && rg -n 'pythonpath|addopts' pyproject.toml" in /glade/derecho/scratch/pg2328/wt_volc
 exited 2 in 1924ms:
/glade/u/home/pg2328/legoesm/packages/tools/legoesm/forcing/external.py
ls: cannot access 'src/legoesm/forcing': No such file or directory

 exited 130 in 41152ms:
============================= test session starts ==============================
platform linux -- Python 3.12.14, pytest-9.1.1, pluggy-1.6.0
rootdir: /glade/derecho/scratch/pg2328/wt_volc
configfile: pyproject.toml
plugins: zarr-3.4.0, timeout-2.4.0, jaxtyping-0.3.11, xdist-3.8.0
collected 21 items

tests/unit/test_volcanic_sw_optics.py FFFFFFFFFFFFFFFFFFFFF              [100%]

=================================== FAILURES ===================================
__________________ test_loader_sorts_bands_and_builds_moments __________________
tests/unit/test_volcanic_sw_optics.py:69: in test_loader_sorts_bands_and_builds_moments
    _, _, lat, mom, p_edges, band_wn = _load(
tests/unit/test_volcanic_sw_optics.py:61: in _load
    from legoesm.forcing.external import _load_volcanic_sw_optics_profile
E   ImportError: cannot import name '_load_volcanic_sw_optics_profile' from 'legoesm.forcing.external' (/glade/u/home/pg2328/legoesm/packages/tools/legoesm/forcing/external.py)
______________________ test_loader_refuses[kw0-omega_sun] ______________________
tests/unit/test_volcanic_sw_optics.py:91: in test_loader_refuses
    _load(volc_sw_file(tmp_path / "v.nc", **kw))
tests/unit/test_volcanic_sw_optics.py:61: in _load
    from legoesm.forcing.external import _load_volcanic_sw_optics_profile
E   ImportError: cannot import name '_load_volcanic_sw_optics_profile' from 'legoesm.forcing.external' (/glade/u/home/pg2328/legoesm/packages/tools/legoesm/forcing/external.py)
________________________ test_loader_refuses[kw1-g_sun] ________________________
tests/unit/test_volcanic_sw_optics.py:91: in test_loader_refuses
    _load(volc_sw_file(tmp_path / "v.nc", **kw))
tests/unit/test_volcanic_sw_optics.py:61: in _load
    from legoesm.forcing.external import _load_volcanic_sw_optics_profile
E   ImportError: cannot import name '_load_volcanic_sw_optics_profile' from 'legoesm.forcing.external' (/glade/u/home/pg2328/legoesm/packages/tools/legoesm/forcing/external.py)
______________________ test_loader_refuses[kw2-Negative] _______________________
tests/unit/test_volcanic_sw_optics.py:91: in test_loader_refuses
    _load(volc_sw_file(tmp_path / "v.nc", **kw))
tests/unit/test_volcanic_sw_optics.py:61: in _load
    from legoesm.forcing.external import _load_volcanic_sw_optics_profile
E   ImportError: cannot import name '_load_volcanic_sw_optics_profile' from 'legoesm.forcing.external' (/glade/u/home/pg2328/legoesm/packages/tools/legoesm/forcing/external.py)
_______________________ test_loader_refuses[kw3-map 1:1] _______________________
tests/unit/test_volcanic_sw_optics.py:91: in test_loader_refuses
    _load(volc_sw_file(tmp_path / "v.nc", **kw))
tests/unit/test_volcanic_sw_optics.py:61: in _load
    from legoesm.forcing.external import _load_volcanic_sw_optics_profile
E   ImportError: cannot import name '_load_volcanic_sw_optics_profile' from 'legoesm.forcing.external' (/glade/u/home/pg2328/legoesm/packages/tools/legoesm/forcing/external.py)
_______________________ test_loader_refuses[kw4-map 1:1] _______________________
tests/unit/test_volcanic_sw_optics.py:91: in test_loader_refuses
    _load(volc_sw_file(tmp_path / "v.nc", **kw))
tests/unit/test_volcanic_sw_optics.py:61: in _load
    from legoesm.forcing.external import _load_volcanic_sw_optics_profile
E   ImportError: cannot import name '_load_volcanic_sw_optics_profile' from 'legoesm.forcing.external' (/glade/u/home/pg2328/legoesm/packages/tools/legoesm/forcing/external.py)
_______________ test_legacy_column_file_is_not_an_optics_source ________________
tests/unit/test_volcanic_sw_optics.py:101: in test_legacy_column_file_is_not_an_optics_source
    assert _load(p) is None
           ^^^^^^^^
tests/unit/test_volcanic_sw_optics.py:61: in _load
    from legoesm.forcing.external import _load_volcanic_sw_optics_profile
E   ImportError: cannot import name '_load_volcanic_sw_optics_profile' from 'legoesm.forcing.external' (/glade/u/home/pg2328/legoesm/packages/tools/legoesm/forcing/external.py)
_______ test_band_limit_table_equals_rrtmgp_files[rrtmgp-gas-sw-g112.nc] _______
tests/unit/test_volcanic_sw_optics.py:109: in test_band_limit_table_equals_rrtmgp_files
    from legoesm.forcing.external import _RRTMGP_SW_BAND_LIMITS_CM1
E   ImportError: cannot import name '_RRTMGP_SW_BAND_LIMITS_CM1' from 'legoesm.forcing.external' (/glade/u/home/pg2328/legoesm/packages/tools/legoesm/forcing/external.py)
_______ test_band_limit_table_equals_rrtmgp_files[rrtmgp-gas-sw-g224.nc] _______
tests/unit/test_volcanic_sw_optics.py:109: in test_band_limit_table_equals_rrtmgp_files
    from legoesm.forcing.external import _RRTMGP_SW_BAND_LIMITS_CM1
E   ImportError: cannot import name '_RRTMGP_SW_BAND_LIMITS_CM1' from 'legoesm.forcing.external' (/glade/u/home/pg2328/legoesm/packages/tools/legoesm/forcing/external.py)
________________ test_production_file_maps_and_sulfate_scatters ________________
tests/unit/test_volcanic_sw_optics.py:119: in test_production_file_maps_and_sulfate_scatters
    _, _, _, mom, _, band_wn = _load(_PROD_FILE)
                               ^^^^^^^^^^^^^^^^^
tests/unit/test_volcanic_sw_optics.py:61: in _load
    from legoesm.forcing.external import _load_volcanic_sw_optics_profile
E   ImportError: cannot import name '_load_volcanic_sw_optics_profile' from 'legoesm.forcing.external' (/glade/u/home/pg2328/legoesm/packages/tools/legoesm/forcing/external.py)
____________________ test_grey_moments_reproduce_grey_path _____________________
tests/unit/test_volcanic_sw_optics.py:158: in test_grey_moments_reproduce_grey_path
    a = solver.solve_columns(**kw, aerosol_volcanic_moments_sw=volc)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   TypeError: RRTMGP.solve_columns() got an unexpected keyword argument 'aerosol_volcanic_moments_sw'
_________________ test_file_optics_change_sw_and_close_budget __________________
tests/unit/test_volcanic_sw_optics.py:174: in test_file_optics_change_sw_and_close_budget
    grey = solver.solve_columns(**kw, aerosol_volcanic_moments_sw=_volc_moments(
E   TypeError: RRTMGP.solve_columns() got an unexpected keyword argument 'aerosol_volcanic_moments_sw'
_________________________ test_jit_parity_and_gradient _________________________
tests/unit/test_volcanic_sw_optics.py:202: in test_jit_parity_and_gradient
    eager = toa_up(1.0)
            ^^^^^^^^^^^
tests/unit/test_volcanic_sw_optics.py:199: in toa_up
    return jnp.sum(solver.solve_columns(
E   TypeError: RRTMGP.solve_columns() got an unexpected keyword argument 'aerosol_volcanic_moments_sw'
_________________________ test_wrong_band_count_raises _________________________
tests/unit/test_volcanic_sw_optics.py:214: in test_wrong_band_count_raises
    solver.solve_columns(**kw, aerosol_volcanic_moments_sw=jnp.zeros(
E   TypeError: RRTMGP.solve_columns() got an unexpected keyword argument 'aerosol_volcanic_moments_sw'
_________ test_non_rrtmgp_radiation_refuses[_build_gray_radiation_fn] __________
tests/unit/test_volcanic_sw_optics.py:231: in test_non_rrtmgp_radiation_refuses
    fn(*args, aerosol_sw_volc_col=jnp.zeros((ncol, nlev, 3, 14)))
E   TypeError: _build_gray_radiation_fn.<locals>.radiation_fn() got an unexpected keyword argument 'aerosol_sw_volc_col'
E   --------------------
E   For simplicity, JAX has removed its internal frames from the traceback of the following exception. Set JAX_TRACEBACK_FILTERING=off to include these.
_________ test_non_rrtmgp_radiation_refuses[_build_none_radiation_fn] __________
tests/unit/test_volcanic_sw_optics.py:231: in test_non_rrtmgp_radiation_refuses
    fn(*args, aerosol_sw_volc_col=jnp.zeros((ncol, nlev, 3, 14)))
E   TypeError: _build_none_radiation_fn.<locals>.radiation_fn() got an unexpected keyword argument 'aerosol_sw_volc_col'
E   --------------------
E   For simplicity, JAX has removed its internal frames from the traceback of the following exception. Set JAX_TRACEBACK_FILTERING=off to include these.
______________________ test_probe_runs_on_synthetic_file _______________________
tests/unit/test_volcanic_sw_optics.py:240: in test_probe_runs_on_synthetic_file
    probe.main(volc_sw_file(tmp_path / "v.nc", omega=np.full(14, 1.0)), 1979, 1)
scripts/validate/probe_volcanic_sw_optics.py:28: in main
    from legoesm.forcing.external import (
E   ImportError: cannot import name 'get_aerosol_sw_volcanic_optics_at_time' from 'legoesm.forcing.external' (/glade/u/home/pg2328/legoesm/packages/tools/legoesm/forcing/external.py)
_________________ test_band_mapping_matches_per_band_grey_path _________________
tests/unit/test_volcanic_sw_optics.py:257: in test_band_mapping_matches_per_band_grey_path
    a = solver.solve_columns(**kw, aerosol_volcanic_moments_sw=volc)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   TypeError: RRTMGP.solve_columns() got an unexpected keyword argument 'aerosol_volcanic_moments_sw'
__________________ test_integration_non_rrtmgp_refuses[gray] ___________________
tests/unit/test_volcanic_sw_optics.py:273: in test_integration_non_rrtmgp_refuses
    integration._call_radiation_backend(
E   TypeError: _call_radiation_backend() got an unexpected keyword argument 'aerosol_sw_volc'
________________ test_integration_non_rrtmgp_refuses[simple_lw] ________________
tests/unit/test_volcanic_sw_optics.py:273: in test_integration_non_rrtmgp_refuses
    integration._call_radiation_backend(
E   TypeError: _call_radiation_backend() got an unexpected keyword argument 'aerosol_sw_volc'
_______________________ test_volcanic_chunking_is_exact ________________________
tests/unit/test_volcanic_sw_optics.py:291: in test_volcanic_chunking_is_exact
    whole = su.aerosol_layer_fields(*args, **kw)[3]
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   IndexError: tuple index out of range
=============================== warnings summary ===============================
tests/unit/test_volcanic_sw_optics.py: 22 warnings
  /glade/derecho/scratch/pg2328/wt_volc/tests/unit/test_volcanic_sw_optics.py:54: DeprecationWarning: Setting the shape on a NumPy array has been deprecated in NumPy 2.5.
  As an alternative, you can create a new view using np.reshape (with copy=False if needed).
    ds.createVariable(k, "f8", dims)[:] = v

tests/unit/test_volcanic_sw_optics.py::test_legacy_column_file_is_not_an_optics_source
  /glade/derecho/scratch/pg2328/wt_volc/tests/unit/test_volcanic_sw_optics.py:100: DeprecationWarning: Setting the shape on a NumPy array has been deprecated in NumPy 2.5.
  As an alternative, you can create a new view using np.reshape (with copy=False if needed).
    ds.createVariable("aod", "f8", ("time", "lat"))[:] = 0.01

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED tests/unit/test_volcanic_sw_optics.py::test_loader_sorts_bands_and_builds_moments
FAILED tests/unit/test_volcanic_sw_optics.py::test_loader_refuses[kw0-omega_sun]
FAILED tests/unit/test_volcanic_sw_optics.py::test_loader_refuses[kw1-g_sun]
FAILED tests/unit/test_volcanic_sw_optics.py::test_loader_refuses[kw2-Negative]
FAILED tests/unit/test_volcanic_sw_optics.py::test_loader_refuses[kw3-map 1:1]
FAILED tests/unit/test_volcanic_sw_optics.py::test_loader_refuses[kw4-map 1:1]
FAILED tests/unit/test_volcanic_sw_optics.py::test_legacy_column_file_is_not_an_optics_source
FAILED tests/unit/test_volcanic_sw_optics.py::test_band_limit_table_equals_rrtmgp_files[rrtmgp-gas-sw-g112.nc]
FAILED tests/unit/test_volcanic_sw_optics.py::test_band_limit_table_equals_rrtmgp_files[rrtmgp-gas-sw-g224.nc]
FAILED tests/unit/test_volcanic_sw_optics.py::test_production_file_maps_and_sulfate_scatters
FAILED tests/unit/test_volcanic_sw_optics.py::test_grey_moments_reproduce_grey_path
FAILED tests/unit/test_volcanic_sw_optics.py::test_file_optics_change_sw_and_close_budget
FAILED tests/unit/test_volcanic_sw_optics.py::test_jit_parity_and_gradient - ...
FAILED tests/unit/test_volcanic_sw_optics.py::test_wrong_band_count_raises - ...
FAILED tests/unit/test_volcanic_sw_optics.py::test_non_rrtmgp_radiation_refuses[_build_gray_radiation_fn]
FAILED tests/unit/test_volcanic_sw_optics.py::test_non_rrtmgp_radiation_refuses[_build_none_radiation_fn]
FAILED tests/unit/test_volcanic_sw_optics.py::test_probe_runs_on_synthetic_file
FAILED tests/unit/test_volcanic_sw_optics.py::test_band_mapping_matches_per_band_grey_path
FAILED tests/unit/test_volcanic_sw_optics.py::test_integration_non_rrtmgp_refuses[gray]
FAILED tests/unit/test_volcanic_sw_optics.py::test_integration_non_rrtmgp_refuses[simple_lw]
FAILED tests/unit/test_volcanic_sw_optics.py::test_volcanic_chunking_is_exact
======================= 21 failed, 23 warnings in 19.13s =======================

exec
/usr/bin/bash -lc "ls packages && ls src && head -80 packages/core/legoesm/__init__.py && rg -n 'pythonpath|addopts' pyproject.toml" in /glade/derecho/scratch/pg2328/wt_volc
 exited 1 in 0ms:
atmosphere
core
coupler
ice
land
ml
ocean
tools
legoesm
head: cannot open 'packages/core/legoesm/__init__.py' for reading: No such file or directory

exec
/usr/bin/bash -lc "cat src/legoesm/__init__.py && rg -n 'pythonpath|addopts' pyproject.toml && ls -ld /glade/work/pg2328/conda-envs/legoesm-gpu/lib/python3.12/site-packages/*legoesm* " in /glade/derecho/scratch/pg2328/wt_volc
 exited 1 in 0ms:
cat: src/legoesm/__init__.py: No such file or directory

exec
/usr/bin/bash -lc 'PYTHONPATH="$PWD/src:$PWD/packages/tools:$PWD/packages/atmosphere:$PWD/packages/core:$PWD/packages/coupler:$PWD/packages/ml:$PWD/packages/ocean:$PWD/packages/land:$PWD/packages/ice" PYTHONDONTWRITEBYTECODE=1 /glade/work/pg2328/conda-envs/legoesm-gpu/bin/python -c '"'import legoesm.forcing.external as e; print(e.__file__)'" in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 2021ms:
/glade/derecho/scratch/pg2328/wt_volc/packages/tools/legoesm/forcing/external.py

exec
/usr/bin/bash -lc 'PYTHONPATH="$PWD/src:$PWD/packages/tools:$PWD/packages/atmosphere:$PWD/packages/core:$PWD/packages/coupler:$PWD/packages/ml:$PWD/packages/ocean:$PWD/packages/land:$PWD/packages/ice" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONDONTWRITEBYTECODE=1 /glade/work/pg2328/conda-envs/legoesm-gpu/bin/python -m pytest -q -p no:cacheprovider tests/unit/test_volcanic_sw_optics.py' in /glade/derecho/scratch/pg2328/wt_volc
exec
/usr/bin/bash -lc "git diff --check HEAD && nl -ba scripts/validate/probe_volcanic_sw_optics.py | sed -n '34,77p' && sed -n '425,456p' tests/unit/test_wb_amip_deck_params.py && sed -n '1320,1350p' packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py" in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 89ms:
    34	
    35	    sig = make_cam6_l32_levels()
    36	    p_s = jnp.array([1.0e5])
    37	    p_half = jnp.asarray(sig.pressure_at_half(p_s)).reshape(1, -1)
    38	    p_full = jnp.asarray(sig.pressure_at_full(p_s)).reshape(1, -1)
    39	    T = jnp.maximum(300.0 * (p_full / 1.0e5) ** 0.19, 200.0)
    40	    q_v = jnp.clip(0.018 * (p_full / 1.0e5) ** 3, 3.0e-6, None)
    41	    lat = jnp.array([0.0])
    42	    day = 365.0 * (year - 1979) + 30.4 * (month - 1) + 15.0
    43	    acfg = AerosolConfig(enabled=True, volcanic_enabled=True,
    44	                         volcanic_path=path, start_year=1979)
    45	    mom, p_edges, band_wn = get_aerosol_sw_volcanic_optics_at_time(acfg, day, lat)
    46	    w = stratospheric_layer_weight(p_half, lat, p_full, T)
    47	    nsrc = mom.shape[1]
    48	    flat = jnp.moveaxis(jnp.asarray(mom, jnp.float64).reshape(1, nsrc, 42), -1, 0)
    49	    placed = jax.vmap(place_stratospheric_aod_profile_to_layers,
    50	                      in_axes=(0, None, None))(flat, jnp.asarray(p_edges), p_half) * w
    51	    volc = jnp.moveaxis(placed, 0, -1).reshape(1, -1, 3, 14)
    52	    wl = 1.0e4 / band_wn                       # cm-1 -> um, (14, 2)
    53	    wts = _planck_band_weights(wl[:, 1], wl[:, 0], _SOLAR_PLANCK_TEMP_K)
    54	    tau_gray = jnp.tensordot(volc[:, :, 0, :], jnp.asarray(wts / wts.sum()),
    55	                             axes=(2, 0))
    56	
    57	    lw, sw = gas_table_files("full")
    58	    cfg = RRTMGPConfig(lw_gas_file=lw, sw_gas_file=sw)
    59	    solver = RRTMGP.from_legoesm_config(cfg)
    60	    kw = dict(T=T, p_full=p_full, p_half=p_half,
    61	              sfc_temperature=jnp.array([300.0]), q_v=q_v,
    62	              cos_zenith=jnp.array([0.5]), sfc_albedo=jnp.array([0.07]))
    63	    arms = {
    64	        "none": solver.solve_columns(**kw),
    65	        "grey": solver.solve_columns(**kw, aerosol_optical_depth=tau_gray),
    66	        "file": solver.solve_columns(**kw, aerosol_optical_depth=jnp.zeros_like(tau_gray),
    67	                                     aerosol_volcanic_moments_sw=volc),
    68	    }
    69	    k100 = int(jnp.argmin(jnp.abs(p_half[0] - 1.0e4)))
    70	    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
    71	                         capture_output=True, text=True).stdout.strip()
    72	    print(f"file={path} {year}-{month:02d} lat=0 cosz=0.5 code={sha}")
    73	    tau_vis = float(volc[0, :, 0, 10].sum())
    74	    ssa_vis = float(volc[0, :, 1, 10].sum() / max(tau_vis, 1e-30))
    75	    print(f"volcanic column tau 16000-22650 cm-1 {tau_vis:.4f} ssa {ssa_vis:.4f};"
    76	          f" gray tau {float(tau_gray.sum()):.4f}")
    77	    for k, o in arms.items():
        T=jnp.full((grid.n_lat, grid.n_lon, 4), 250.0))
    sl = SimpleNamespace(sst=np.full((4, 8), 290.0), lat=np.linspace(-80, 80, 4),
                         lon=np.linspace(0, 315, 8), sfc_shf=None, land_frac=None)
    for stamp, want in (("2016-03-01T00", 59.0), ("2016-02-29T06", 58.25),
                        ("2015-12-31T18", 364.75)):
        calls.clear()
        f = build_spectral_forcing(sl, grid, np.datetime64(stamp, "ns"), 2016,
                                   aerosol=spec, sigma=sigma, ic_carry=carry)
        assert sorted({y for y, _, _ in calls}) == list(range(2005, 2015))
        assert {d for _, d, _ in calls} == {want}
        col = np.asarray(f["aerosol_od"]).sum(axis=1)
        np.testing.assert_allclose(col, 1e-3 * 9.5, rtol=1e-12)
        np.testing.assert_allclose(np.asarray(f["aerosol_ccn_aod"]),
                                   0.5 + 1e-3 * 9.5, rtol=1e-12)
    plain = build_spectral_forcing(sl, grid, np.datetime64("2016-03-01", "ns"),
                                   2016)
    assert "aerosol_od" not in plain


def test_the_cam6_tier3_deck_runs_aerosol_ccn_like_amip(monkeypatch):
    """The droplet-number-from-aerosol switch reaches the BUILT physics (both
    radiation and microphysics read it), and the deck's aerosol keys are
    accepted.  Non-vacuity: drop ``nc_from_aerosol`` from the deck's
    param_fixed and the gate reads False."""
    import copy
    from types import SimpleNamespace

    import legoesm.atmosphere.physics.combined as combined
    from legoesm.training.scale_build import (
        build_mode_components, validate_wb_campaign_yaml)

    yml = copy.deepcopy(_tier3())
      sw_hr = sw_hr_3d[:, 0, hw:-hw][:, ::-1]

      if overhead:
          # Drop the overhead layer's heating and the model-top face: face 0
          # stays the TRUE top (lid), so TOA diagnostics see the full column.
          def _faces(a):
              return jnp.concatenate([a[:, :1], a[:, 2:]], axis=1)

          lw_up, lw_down = _faces(lw_up), _faces(lw_down)
          sw_up, sw_down = _faces(sw_up), _faces(sw_down)
          lw_hr, sw_hr = lw_hr[:, 1:], sw_hr[:, 1:]

      return RadiationOutput(
          lw_flux_up=lw_up,
          lw_flux_down=lw_down,
          sw_flux_up=sw_up,
          sw_flux_down=sw_down,
          heating_rate=lw_hr + sw_hr,
          lw_heating_rate=lw_hr,
          sw_heating_rate=sw_hr,
          # internal index -1 is the TOA face (== *_flux_up[:, 0] above)
          lw_flux_up_toa_clr=(lw_fluxes['flux_up_clr'][:, 0, -1]
                              if clear_sky else None),
          sw_flux_up_toa_clr=(sw_fluxes['flux_up_clr'][:, 0, -1]
                              if clear_sky else None),
      )

  def solve_columns_chunked(
      self,
      *,
      column_chunk_size: int,

exec
/usr/bin/bash -lc "PYTHONDONTWRITEBYTECODE=1 /glade/work/pg2328/conda-envs/legoesm-gpu/bin/python - <<'PY'
import netCDF4
p='/glade/work/pg2328/amip_inputs/volcanic_cmip6_from_cesm_1978-2014.nc'
with netCDF4.Dataset(p) as d:
 print([(k,len(v)) for k,v in d.dimensions.items()])
 print('time',d.variables['time'].__dict__)
 print('alt',d.variables['altitude'][:3],d.variables['altitude'][-3:])
PY" in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 250ms:
[('terrestrial_bands', 16), ('latitude', 36), ('altitude', 70), ('time', 444), ('solar_bands', 14), ('altitude_int', 71)]
time {'_FillValue': np.float64(nan), 'units': 'days since 1850-01-01 00:00:00', 'calendar': 'noleap'}
alt [5.  5.5 6. ] [38.5 39.  39.5]

exec
/usr/bin/bash -lc "git status --short && sed -n '1470,1508p' packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py && sed -n '1168,1220p' packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py" in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 26ms:
 M packages/atmosphere/legoesm/atmosphere/physics/clouds/subcolumns.py
 M packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py
 M packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py
 M packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rte/two_stream.py
 M packages/coupler/legoesm/driver/compiled_segments.py
 M packages/coupler/legoesm/driver/model_driver.py
 M packages/coupler/legoesm/driver/physics_pipeline.py
 M packages/ml/legoesm/training/scale_build.py
 M packages/tools/legoesm/forcing/external.py
 M packages/tools/legoesm/forcing/surface_utils.py
 A scripts/validate/probe_volcanic_sw_optics.py
 M tests/unit/test_aerosol_band_ccn.py
 M tests/unit/test_cross_lane_forcing.py
 A tests/unit/test_volcanic_sw_optics.py
 M tests/unit/test_wb_amip_deck_params.py
?? docs/wb/reviews/2026-11-22_volcanic_optics_r1_codex.md
?? docs/wb/reviews/2026-11-22_volcanic_optics_r1_glm.md
?? docs/wb/reviews/2026-11-22_volcanic_optics_r1_prompt.md
?? docs/wb/reviews/2026-11-22_volcanic_optics_r2_codex.md
?? docs/wb/reviews/2026-11-22_volcanic_optics_r2_glm.md
?? docs/wb/reviews/2026-11-22_volcanic_optics_r2_prompt.md
?? docs/wb/reviews/2026-11-22_volcanic_optics_vote_codex.md
?? docs/wb/reviews/2026-11-22_volcanic_optics_vote_glm.md
?? docs/wb/reviews/2026-11-22_volcanic_optics_vote_prompt.md
          return x.reshape(n_block, column_chunk_size, *x.shape[1:])

      mapped = {
          "T": _split(T), "p_full": _split(p_full), "p_half": _split(p_half),
          "sfc_temperature": _split(sfc_temperature), "q_v": _split(q_v),
          "cos_zenith": _split(cos_zenith),
      }
      # ``solar_spectral_fraction`` is a per-g-point vector (leading axis
      # n_gpt, NOT ncol) that must be broadcast to every block, never split.
      # Excluded by name so it stays static even in the degenerate
      # ncol == n_gpt_sw case where the shape heuristic alone would misfire
      # (splitting it → each block sees a short weight vector → solve_columns
      # raises on the wrong g-point length).
      _non_column_opt = ("solar_spectral_fraction",)
      static = {}
      for k, v in opt.items():
          if k not in _non_column_opt and _is_col_array(v):
              mapped[k] = _split(v)
          else:
              static[k] = v

      # Checkpointed per block: lax.map is a scan, so without this the
      # backward pass keeps every block's radiation activations at once and
      # the chunking buys compile time but no memory.  With it the backward
      # recomputes one block at a time, so reverse-mode scratch is bounded by
      # ``column_chunk_size`` columns instead of ``ncol`` -- the AMIP WB arm
      # needs it because max-random overlap hands the solver n_sub*ncol
      # sub-columns (measured 490 GiB of scratch at 8 sub-columns, T63/L32).
      # Forward-only callers are unaffected (checkpoint is a no-op there) and
      # the values are unchanged either way.  prevent_cse=False because this
      # body already runs inside a scan (same reason as the g-point block
      # loop in rte/two_stream.py).
      @functools.partial(
          jax.checkpoint,
          policy=jax.checkpoint_policies.nothing_saveable,
          prevent_cse=False,
      )
      def _one(chunk):
          return self.solve_columns(**chunk, **static)
              raise ValueError(
                  "solar_spectral_fraction has wrong length: "
                  f"{solar_weights.shape[0]} (expected {optics_lib.n_gpt_sw})",
              )
      else:
          solar_weights = None

      # --- 3b. Optics-only short-circuit for the 3D MC ray tracer ---
      # Reuses the EXACT state above (no duplicated numerics); returns the
      # per-g-point shortwave optical field instead of solving transport. The
      # default path (sw_optical_field_only=False) is byte-identical.
      if clear_sky and (sw_optical_field_only or lw_optical_field_only):
          raise ValueError(
              "clear_sky adds clear-sky TOA fluxes to the RadiationOutput "
              "path; the optical-field-only outputs have no fluxes.")

      if sw_optical_field_only:
          sw_props = two_stream.compute_sw_optical_field(
              p_3d, T_3d, molecules, optics_lib, vmr_fields,
              cloud_r_eff_liq=crl_3d, cloud_path_liq=cpl_3d,
              cloud_r_eff_ice=cri_3d, cloud_path_ice=cpi_3d,
              cloud_fraction=cf_3d,
              aerosol_optical_depth=aerosol_od_3d,
              aerosol_single_scattering_albedo=aer_ssa,
              aerosol_asymmetry_factor=aer_g,
              aerosol_volcanic_moments=volc_moments_3d,
          )
          # Strip the singleton Y axis + vertical halos, flip back to the
          # legoESM TOA-first convention -> (n_gpt, ncol, nlev).
          hw = 1

          def _strip(a):
              return a[:, :, 0, hw:-hw][:, :, ::-1]

          tau_gpt = _strip(sw_props['optical_depth'])
          ssa_gpt = _strip(sw_props['ssa'])
          g_gpt = _strip(sw_props['asymmetry_factor'])
          rayleigh_frac_gpt = _strip(sw_props['rayleigh_frac'])
          # Cloud liquid effective radius [um] per cell for Mie-LUT sampling
          # (microhh LUT spans 2.5..21.5 um). Clear-sky -> the LUT floor.
          # crl_3d is 3D (ncol, 1, nlev+2) -- NOT per-g-point 4D -- so it needs a
          # 3D strip (drop singleton, trim halo, z-reverse to TOA-first), not the
          # 4D _strip used for the per-g-point optics.
          if crl_3d is not None:
              r_eff_um = crl_3d[:, 0, hw:-hw][:, ::-1] * 1.0e6   # (ncol, nlev)
          else:
              r_eff_um = jnp.full((ncol, nlev), 2.5, dtype=tau_gpt.dtype)  # coeff-ok: microhh Mie LUT lower r_eff bound [um]
          weights = solar_weights
          if weights is None:
              weights = optics_lib.solar_fraction_by_gpt
          # Beam-normal solar flux per g-point [W/m^2].
          solar_flux_normal = config.S_0 * weights
          return (tau_gpt, ssa_gpt, g_gpt, rayleigh_frac_gpt, r_eff_um,

codex
The shared environment initially imported the main checkout. With imports pinned to this worktree, the tests are progressing without failures so far. I haven’t found a remaining blocking defect in the code review.
exec
/usr/bin/bash -lc "sed -n '1200,1268p' packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py && sed -n '1550,1578p' packages/ml/legoesm/training/scale_build.py && sed -n '922,940p' CLAUDE.md" in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 0ms:
            q_snow=q_snow,
            n_snow=n_snow,
            n_cloud_specified=n_cloud_specified,
        )
        if cloud_config.cap_floor_on:
            from legoesm.atmosphere.physics.clouds.cloud_fraction import apply_cap_cloud_floor
            cloud_props = apply_cap_cloud_floor(cloud_props, lat, p_full, dp, cloud_config)
        # ``to_rrtmg_kwargs`` builds the kwargs without ``cloud_fraction``
        # (commit 4c9591bb, lost in AIMIP-#312 merge, restored iter-15
        # in ``physics_pipeline.py`` and iter-16 here) — see docstring
        # for why.  iter-17 centralised the helper so the bug can't
        # resurface at a third call site.
        cloud_kwargs = cloud_props.to_rrtmg_kwargs()

    # RRTMGP path: use solver directly (config is baked in).
    if rrtmgp_solver is None:
        # Fallback: construct a solver on the fly (e.g. called without
        # the pre-built solver from make_radiation_physics).
        from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP
        rrtmgp_solver = RRTMGP.from_legoesm_config(radiation_config.rrtmgp)

    _rad_kwargs = dict(
        T=T,
        p_full=p_full,
        p_half=p_half,
        sfc_temperature=sfc_temperature,
        q_v=q_v_safe,
        cos_zenith=cos_sza,
        sfc_albedo=sfc_albedo_override,
        sfc_albedo_direct=sfc_albedo_direct,
        sfc_emissivity=sfc_emissivity_override,
        o3_vmr=o3_vmr,
        ghg_vmr_override=ghg_vmr_override,
        aerosol_optical_depth=aerosol_od,
        aerosol_absorption_optical_depth_lw=aerosol_lw_od,
        solar_spectral_fraction=solar_spectral_fraction,
        **cloud_kwargs,
    )
    if aerosol_sw_volc is not None:
        _rad_kwargs["aerosol_volcanic_moments_sw"] = aerosol_sw_volc
    if cloud_config is not None:
        # One bulk ice density for the PSD radius and the optics size (CAM6).
        _rad_kwargs["cloud_rho_ice"] = cloud_config.rho_cloud_ice
    if clear_sky_toa:
        _rad_kwargs["clear_sky"] = True
    if o3_top_vmr is not None and o3_vmr_override is not None:
        _rad_kwargs["o3_top_vmr"] = o3_top_vmr
    # Maximum-random-overlap SUBCOLUMNS (opt-in).  The default path hands
    # every layer's GRID-MEAN water path to ONE homogeneous column, so cloud
    # spread thinly over many partly cloudy layers is solved as one deep
    # uniform cloud.  Measured against a Monte-Carlo independent-column
    # reference on a real model state, that costs ~30% of the cloud albedo
    # (0.3995 -> 0.2786) and ~18 W/m2 of OLR (219.2 -> 237.8, reference 239.0).
    # See clouds/subcolumns.py for the validation and for the closed-form
    # alternative that was measured and rejected.
    #
    # The kwargs are chosen BEFORE the solve so exactly ONE solve runs: an
    # earlier version solved the base column and then the subcolumns and threw
    # the base away, doubling the cost and compiling an extra shape.
    _ov = ("none" if cloud_props is None else
           getattr(cloud_config, "cloud_vertical_overlap_optics", "none"))
    _n_sub = 1
    if _ov == "max_random":
        from legoesm.atmosphere.physics.clouds import subcolumns as _sub
        _n_sub = int(getattr(cloud_config, "cloud_n_subcolumns",
                             _sub.N_SUBCOLUMNS_DEFAULT))
        _ncol = T.shape[0]
        # CAM draws different uniforms in every column; the per-column shift
        # (seeded from the bottom pressure, as for mcica) does the same, so a
    The AMIP production aerosol, as its driver builds it, but on a
    calendar-month CLIMATOLOGY over ``aerosol_climatology_years`` [first,
    last]: the files end 2014 and the WB windows are 2015-2020 (owner
    decision 2026-10-16)."""
    cl = dict(yml.get("classical", {}) or {})
    if not cl.get("aerosol_file"):
        return None
    from legoesm.forcing.external import AerosolConfig

    years = cl.get("aerosol_climatology_years")
    lw = cl.get("volcanic_aerosol_lw", False)
    if (not isinstance(years, list) or len(years) != 2
            or not all(isinstance(y, int) for y in years)
            or years[0] > years[1]):
        raise ValueError("classical.aerosol_climatology_years must be "
                         f"[first, last] integer years, got {years!r}")
    if not isinstance(lw, bool):
        raise ValueError(
            f"classical.volcanic_aerosol_lw must be a YAML boolean, got {lw!r}")
    volc = str(cl.get("volcanic_aerosol_file", "") or "")
    acfg = AerosolConfig(
        enabled=True, source="climatology", path=str(cl["aerosol_file"]),
        ccn_path=str(cl.get("aerosol_ccn_file", "") or ""),
        use_reference_if_missing=True,
        reference_aod_550=float(cl.get("aerosol_reference_aod", 0.03)),
        volcanic_enabled=bool(volc), volcanic_path=volc,
        volcanic_lw_enabled=(lw and bool(volc)),
        hold_outside_file_years=False)
    return acfg, tuple(range(years[0], years[1] + 1)), (lw and bool(volc))
## Validation
- Narrowest test after edits. Numerical changes: analytical/benchmark > unit tests alone. `JAX_ENABLE_X64=1` unless float32/Metal task.
- Dycore: Williamson, Galewsky, Jablonowski-Williamson, DCMIP, Held-Suarez, ocean benchmarks.
- Conservation/reductions/coupler: mass+energy diagnostics.
- Parallel: unsharded vs sharded, single-rank vs MPI.
- Too expensive: say what ran/didn't, residual risk.
- **CRITICAL — Controlled comparison: change ONE variable, hold eval protocol FIXED to baseline.** To claim a change (resolution, params, scheme, days) improved/degraded/"is comparable" vs prior result, keep EVERYTHING else byte-identical to that baseline: forcing data + its sampling (years/days/hours/climatology), evaluation grid, metric definition (bias vs RMSE), region masks, timestepping. Metric that moved because protocol/sampling changed = **CONFOUND, not result** — NEVER compare number computed on one sampling/grid/metric to number from another and call difference an effect. Resource limit (network cap, compute, time) force lighter or different sampling → **re-run BASELINE at that SAME sampling before comparing** — fresh baseline cheap insurance; confounded claim not. Before writing "improved"/"degraded"/"better"/"comparable" vs any earlier number, explicitly confirm two configs differ ONLY in variable under test; cannot → say so, claim NO direction. Report full config (data+sampling, grid, days/steps, params, metric) next to every number so reader know exactly what compared. Assuming two runs comparable when setup drifted = error that turn "we improved it" into "we degraded it."
- **CRITICAL — Precision gate for comparison/skill/causal claims (do NOT be sloppy — 2026-07-20 EC-site lesson).** (a) **Harness self-check FIRST**: before reporting NEW scheme's skill vs validated baseline, run KNOWN baseline through YOUR OWN harness, confirm it reproduce baseline's established/published number (±small tol). Your harness score validated scheme wildly off — e.g. two-leaf H looked "broken" (NSE≈−1) when paper figure track obs — the HARNESS is wrong; fix BEFORE any new-scheme claim. (b) **Match reference metric EXACTLY**: metric aggregation (daily-mean vs half-hourly point-wise NSE), window (multi-year JJA vs one summer), masks (valid-forcing) ALL part of protocol. Report SAME metric reference used, plus any alternative, WITH sample N. Single-window point-wise metric is NOT skill verdict, and never rank schemes by metric punishing one scheme's known artifact (point-wise scatter/spikes) while ignoring dimension of interest (mean diurnal shape). (c) **Never let QC mask flatter one side**: drop scheme's unphysical spikes from scoring → report BOTH failures-penalized primary score AND separately-labeled plausible-only sensitivity score. (d) **Do NOT INFER causal origin — INSTRUMENT it**: "the spikes come from X" require LOGGING X and alternatives (raw vs intermediate vs final), not deduction from reading one code path; "correctly hooked up" require reading FULL path, not entry point. (e) **Label every claim CONFIRMED (evidence shown) vs PLAUSIBLE (inferred)** — adversarial review WILL refute over-confident inferences; state uncertainty up front rather than presenting verdict table a five-minute check overturn.
- **CRITICAL — A gradient/AD-tuning result that underperforms a known-good baseline is a BUG in your method until PROVEN otherwise — never attribute it to physics.** Before interpreting ANY optimizer/AD tuning outcome, or attributing residual error to physical cause (radiation, resolution, scheme's structure): (1) **Differentiated loss MUST be exact quantity you evaluate and report.** Truncated-BPTT / short-window / stop-gradient'd-spinup / cached-state proxy differing from full evaluation = OBJECTIVE MISMATCH, not result — optimizer minimize proxy while eval barely move. Differentiate real objective (e.g. full RCE equilibrium the eval scores), or prove proxy tight. (2) **Verify AD actually flows:** finite-difference check on ≥1 parameter (analytic grad vs `(L(x+ε)−L(x−ε))/2ε`, ratio≈1), confirm gradients finite AND non-negligible (not silently zeroed by `stop_gradient`, frozen/`enable_*=False` leaf, `lax.stop_gradient`/`jax.lax.cond` dead branch, or detached recompute). (3) **Sanity vs known-good baseline FIRST:** derivative-free or prior method got large improvements and your gradient method barely move loss → YOUR optimization broken (weak objective, tiny effective LR, over-aggressive line-search backtracking to ~0 step, frozen leaves) — fix before drawing conclusions. Concluding "residual bias is radiative/structural, not tunable" from run whose optimizer never actually worked = exact error that ship FALSE scientific claim. Near operationalization this disqualifying: barely-moving loss, suspiciously-flat before/after, or result contradicting known-good baseline BLOCKS delivery until gradient path verified end-to-end. See [[feedback_verify_ad_before_physics]].
- **CRITICAL — Visual verify spatial/grid artifacts**: passing tests+norms NECESSARY ≠ SUFFICIENT for cubed-sphere ops, halo exchange, diffusion coeffs, grid metrics. Edge artifacts/cube imprint/grid-scale noise only detected visually (v-wind W2, wind_speed W5). Run `--only sw --grid cubed_sphere --quick` + inspect PNGs vs baseline. Norms can improve while artifacts worsen. Never claim "tests pass, edge fixed" from pytest alone.
- **Diffusion sensitivity**: div damping + hyperdiff AMPLIFY halo errors at cubed-sphere face boundaries. Check W2 v-wind visually when touching `_hyperdiff_cube`, `_div_damp_cube`, diffusion params.
- **Visual-regression gate (cube imprint)**: `scripts/validate/visual_regression.py --check` numericises W2 v-wind cube-imprint check (SSIM + per-panel perceptual hash + edge-artifact ratio vs tiny committed ref in `tests/visual_baselines/`). Deterministic metric math gated in CI (`tests/test_visual_regression_metrics.py`); full cube-SW `--check` runs as NIGHTLY non-blocking CI job until tolerances calibrated across CI hardware. Tiny numeric baselines (.npy+json) ARE tracked — the one carve-out to "no tracked visual baselines".
- **CRITICAL — VALIDATE THE INSTRUMENT BEFORE QUOTING ITS NUMBER (2026-07-25 duogrid lesson: 8 confident claims, all retracted).** Diagnostic script = UNTRUSTED CODE until it pass own controls. Never state finding — never write "measured", "confirmed", "proven", "VERDICT" — from probe's first output. **Before quoting any diagnostic number, run these five checks and say in the message that you ran them:**
  1. **Right conserved/invariant quantity?** Budget what SYSTEM conserves, not convenient proxy. (Failed: reported "vertex creates energy" from **KE alone** — KE NOT conserved in shallow water, it trade with PE. Total `E=∫area(½h|V|²+½gh²)` reversed sign of conclusion.)
  2. **Same transform / units / staggering on BOTH sides?** Two "A-grid winds" from different operators = DIFFERENT QUANTITIES. (Failed: ours `c2l_ord2` vs oracle `C2L_ORD=4` — SAME raw state gave 1.96e-2 vs 5.79e-2, 3× swing that WAS the reported effect. Also: never budget across stage boundary where state change representation — mid-step FV3 winds in circulation form, produced ±5.6e10 garbage. REPEAT OFFENCE 2026-08-05, this time SHIPPED then retracted: paired predicted face velocity (41.87 m/s) against model's `max_speed` timeseries diagnostic (21.95 m/s), which is CELL-CENTRE average of same field whose face maximum was 42.30 — apparent "1.2 % agreement" was artifact of comparing across staggering, and it went into merged PR before 1-step run exposed it. Name staggering AND reduction of both sides in sentence quoting them.)
  3. **Same time, resolution, config?** Index by MATCHED TIME, not frame number. (Failed: mapped day→frame as `round(day)-1` against HOURLY file, comparing our day-1 to their hour-1; and quoted **C12** wedge gain (~300×) as mechanism for **C48** instability, where it ~124×.)
  4. **Is the metric measuring what its name says?** Prove it on synthetic case with KNOWN answer before use. (Failed: called `mean|f−4-neighbour-mean|` a "2Δx grid-scale" measure — it is high-pass/curvature residual that merely sharper SMOOTH feature reproduce. Failed: "gain" probe that re-filled FIXED source, trivially 1.0000 by construction.)
  5. **Can the reduction support the claim?** `max` over tiles/corners/components taken independently per run can peak at DIFFERENT physical locations; max-of-per-tile-means is not global mean. Keep argmax metadata, map to common physical location before claiming "localized".
  Plus: **diff ARRAYS, never printed summaries** (claimed "bit-identical ⇒ deterministic, not chaos"; arrays actually differed by 9e-6 — only rounded printout matched). **Never let probe print own verdict** ("=> the growth is REAL") — interpretation belong in analysis after controls pass, not baked into tool where it get echoed back as evidence. **`nanmax`/`nanmean`/`nansum` hide failures** — make NaN and missing frames FATAL. **Record every effective flag, env var and git SHA in each artifact**; default `--n 36` silently mis-slicing C48 file run fine and lie.

exec
/usr/bin/bash -lc 'PYTHONPATH="$PWD/src:$PWD/packages/tools:$PWD/packages/atmosphere:$PWD/packages/core:$PWD/packages/coupler:$PWD/packages/ml:$PWD/packages/ocean:$PWD/packages/land:$PWD/packages/ice" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONDONTWRITEBYTECODE=1 /glade/work/pg2328/conda-envs/legoesm-gpu/bin/python -m pytest -q -p no:cacheprovider tests/unit/test_aerosol_band_ccn.py' in /glade/derecho/scratch/pg2328/wt_volc
exec
/usr/bin/bash -lc "rg -n 'scheme.*rrtmg|radiation == \"rrtmg\"|radiation in.*rrtmg|rrtmg.*rrtmgp' packages/atmosphere/legoesm/atmosphere/physics/radiation/config.py packages/coupler/legoesm/driver/physics_pipeline.py packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py | head -55" in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 0ms:
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:141:    if radiation_config.scheme not in ("rrtmgp", "gray"):
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:159:        if radiation_config.scheme != "rrtmgp" and not _gray_e_is_one:
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:172:                   if radiation_config.scheme == "rrtmgp"
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:215:    if radiation_config.scheme not in ("rrtmgp", "ecrad_ifs"):
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:329:    elif config.scheme == "rrtmgp":
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:333:        return rrtmgp_radiation, config.rrtmgp
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:366:    S_0 = config.rrtmgp.S_0 if config.scheme == "rrtmgp" else config.gray.S_0
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1064:    if open_ocean_frac is not None and radiation_config.scheme != "rrtmgp":
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1069:    if aerosol_sw_volc is not None and radiation_config.scheme != "rrtmgp":
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1218:        from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1219:        rrtmgp_solver = RRTMGP.from_legoesm_config(radiation_config.rrtmgp)
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1277:            from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import (
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1423:        radiation_config.scheme in ("rrtmgp", "mc3d")
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1554:    if radiation_config.scheme in ("rrtmgp", "mc3d"):
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1555:        from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1558:            rrtmgp_solver = RRTMGP.from_legoesm_config(radiation_config.rrtmgp)
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1570:    if radiation_config.ozone.source == "ml" and radiation_config.scheme == "rrtmgp":
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1698:    When ``nc_from_aerosol`` is True AND the scheme is ``rrtmgp`` the
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1890:                       if radiation_config.scheme == "rrtmgp"
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1968:        if nc_from_aerosol and radiation_config.scheme == "rrtmgp":
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:2146:            rrtmgp_solver=rrtmgp_solver,
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:2207:                rrtmgp_solver=rrtmgp_solver,
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:2572:            rrtmgp_solver=rrtmgp_solver,
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:2944:                rrtmgp_solver=rrtmgp_solver, lon=lon_col,
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:3085:            radiation_config.scheme == "rrtmgp"
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:3090:                f"Cloud-aware radiation (scheme='rrtmgp', "
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:3137:            rrtmgp_solver=rrtmgp_solver, lon=lon_col,
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:3360:                       if radiation_config.scheme == "rrtmgp"
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:3443:            rrtmgp_solver=rrtmgp_solver,
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:3485:    if radiation_config.scheme == "rrtmgp":
packages/coupler/legoesm/driver/physics_pipeline.py:3831:    and silently built the full RRTMGP scheme. Matches the gray/rrtmgp
packages/coupler/legoesm/driver/physics_pipeline.py:3969:    from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import gas_table_files
packages/coupler/legoesm/driver/physics_pipeline.py:3991:    from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import gas_table_files
packages/coupler/legoesm/driver/physics_pipeline.py:4005:    from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP
packages/coupler/legoesm/driver/physics_pipeline.py:4019:    rrtmg_config = rrtmgp_config_from_experiment(
packages/coupler/legoesm/driver/physics_pipeline.py:4163:    return radiation_fn, {"scheme": rrtmg_config, "orbit": orbit}
packages/coupler/legoesm/driver/physics_pipeline.py:4186:    "rrtmgp": _build_rrtmgp_radiation_fn,
packages/coupler/legoesm/driver/physics_pipeline.py:4187:    "rrtmg": _build_rrtmgp_radiation_fn,  # common alias
packages/coupler/legoesm/driver/physics_pipeline.py:5862:    pipeline.rrtmgp_radiation = rad_scheme in ("rrtmgp", "rrtmg")
packages/atmosphere/legoesm/atmosphere/physics/radiation/config.py:390:        Active radiation scheme: "gray", "rrtmgp", "mc3d", "simple_lw" or

exec
/usr/bin/bash -lc "sed -n '2470,2590p' packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py && sed -n '2850,2957p' packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py && sed -n '3050,3155p' packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py" in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 0ms:
        p = pressure_from_eos(rho_total, theta_total)
        exner = (p / constants.p_ref) ** constants.kappa
        T = theta_total * exner

        nlev = height_coord.n_levels
        shape_3d = theta_p.shape  # (6, n, n, nlev)
        shape_w = state.w.data.shape  # (6, n, n, nlev+1)
        shape_2d = state.phis.data.shape  # (6, n, n)

        # Interface pressure from evolving column state (not fixed reference).
        p_half = reconstruct_half_level_pressure_hydrostatic(
            p_full=p,
            rho_full=rho_total,
            z_half=terrain_metric.z_half_3d,
        )

        # Surface temperature = lowest-level temperature (default)
        # with optional SCM-driver override via set_T_sfc_override hook.
        T_sfc = _apply_T_sfc_override(T[..., -1], _T_sfc_override_cell[0])

        # Insolation (and optionally cos_sza for diurnal cycle).
        insol, cos_sza, f_day, eccf = _compute_insolation(
            lat, radiation_config,
            lon=lon,
            day_of_year=_time["day_of_year"],
            seconds_of_day=_time["seconds_of_day"],
        )

        # Reshape to columns
        ncol = shape_2d[0] * shape_2d[1] * shape_2d[2]
        T_col = T.reshape(ncol, nlev)
        p_full_col = p.reshape(ncol, nlev)
        p_half_col = p_half.reshape(ncol, nlev + 1)
        T_sfc_col = T_sfc.reshape(ncol)
        lat_col = lat.reshape(ncol)
        lon_col = lon.reshape(ncol)
        insol_col = insol.reshape(ncol)
        cos_sza_col = cos_sza.reshape(ncol) if cos_sza is not None else None

        # NH state stores water vapor in tracer slot 0 when moist tracers exist.
        n_tracers = state.tracers.data.shape[-1]
        if n_tracers > 0:
            q_v = jnp.clip(state.tracers.data[..., 0], 0.0, None)
        else:
            q_v = jnp.zeros_like(T)
        q_v_col = q_v.reshape(ncol, nlev)

        # Cloud condensate from tracer slots 1 (q_c) and 3 (q_i).
        q_cloud_col = None
        q_ice_col = None
        if n_tracers > 1:
            q_cloud_col = jnp.clip(
                state.tracers.data[..., 1], 0.0, None
            ).reshape(ncol, nlev)
        if n_tracers > 3:
            q_ice_col = jnp.clip(
                state.tracers.data[..., 3], 0.0, None
            ).reshape(ncol, nlev)
        # Cloud-ice NUMBER (per-mass [1/kg]) from tracer slot 8 = N_i in the
        # canonical 9-slot double-moment layout (q_v,q_c,q_r,q_i,q_s,q_g,
        # N_c,N_r,N_i). ALL double-moment schemes share N_c=6 / N_i=8:
        # morrison, seifert_beheng, thompson, ml_emulator, and p3 (p3 reuses
        # slots 4/5 for q_rim/B_rim but keeps N_c=6/N_r=7/N_i=8, with N_i
        # per-mass via N_i_target/ρ). Feeds the M2005 PSD ice effective radius
        # EFFI=1.5/LAMI (RAD-1-ice). The ``> 8`` guard keeps single-moment
        # layouts (kessler/sundqvist, 3 slots) on the constant r_eff. CAVEAT
        # (codex iter-12): a hypothetical non-standard ≥9-slot layout that did
        # NOT place N_i at slot 8 would mis-read it — a tracer-metadata/scheme
        # key would be more robust but is deferred; every current 9-slot scheme
        # honours this layout.
        n_ice_col = None
        if n_tracers > 8:
            n_ice_col = jnp.clip(
                state.tracers.data[..., 8], 0.0, None
            ).reshape(ncol, nlev)
        # Cloud-droplet NUMBER (per-VOLUME [#/m³]) from slot 6 (N_c in the
        # Morrison double-moment layout) ⇒ the M2005 PSD liquid effective radius
        # reffc=(PGAM+3)/(2·LAMC). Same ``> 8`` (Morrison) guard as N_i.
        n_cloud_col = None
        if n_tracers > 8:
            # Stored per MASS; the PSD wants per VOLUME (see the extractor).
            n_cloud_col = jnp.clip(
                state.tracers.data[..., 6], 0.0, None
            ).reshape(ncol, nlev) * rho_total.reshape(ncol, nlev)

        f_day_col = f_day.reshape(ncol) if f_day is not None else None
        rad_out = _call_radiation_backend(
            radiation_config=radiation_config,
            eccf=eccf,
            T=T_col,
            p_full=p_full_col,
            p_half=p_half_col,
            sfc_temperature=T_sfc_col,
            lat=lat_col,
            q_v=q_v_col,
            insolation=insol_col,
            cos_sza=cos_sza_col,
            q_cloud=q_cloud_col,
            q_ice=q_ice_col,
            n_ice=n_ice_col,
            n_cloud=n_cloud_col,
            f_day=f_day_col,
            rrtmgp_solver=rrtmgp_solver,
            lon=lon_col,
            ml_ozone_coefs=ml_ozone_coefs,
        )

        # Convert dT/dt -> dtheta'/dt using local Exner (T = theta * exner).
        dT_dt = rad_out.heating_rate.reshape(shape_3d)
        dtheta_prime_dt = dT_dt / jnp.clip(exner, 1e-6, None)

        dims_3d = ("face", "x", "y", "level")
        dims_w = ("face", "x", "y", "level_half")
        dims_2d = ("face", "x", "y")
        dims_tr = ("face", "x", "y", "level", "tracer")
        # Pin zero-tendency placeholders to the upstream state precision
        # so x64 default zeros do not leak into the f32 column path.
        _state_dtype = T.dtype
        _phis_dtype = state.phis.data.dtype

        return NonHydrostaticTendencies(
        exner = (p / constants.p_ref) ** constants.kappa
        T = theta_total * exner

        nlev = height_coord.n_levels
        shape_3d = theta_p.shape          # (ny, nx, nlev)
        shape_w = state.w.data.shape      # (ny, nx, nlev+1)
        shape_2d = state.phis.data.shape  # (ny, nx)
        ny, nx = shape_2d
        ncol = ny * nx

        p_half = reconstruct_half_level_pressure_hydrostatic(
            p_full=p, rho_full=rho_total,
            z_half=terrain_metric.z_half_3d,
        )
        T_sfc = _apply_T_sfc_override(T[..., -1], _T_sfc_override_cell[0])

        # Plane lat/lon: PlaneGrid.grid_lat returns constant lat0 over
        # (ny, nx), already in radians (deg2rad applied in property).
        lat, lon = _get_grid_lat_lon(grid, shape_2d)
        insol, cos_sza, f_day, eccf = _compute_insolation(
            lat, radiation_config, lon=lon,
            day_of_year=_time["day_of_year"],
            seconds_of_day=_time["seconds_of_day"],
        )

        T_col = T.reshape(ncol, nlev)
        p_full_col = p.reshape(ncol, nlev)
        p_half_col = p_half.reshape(ncol, nlev + 1)
        T_sfc_col = T_sfc.reshape(ncol)
        lat_col = lat.reshape(ncol)
        lon_col = lon.reshape(ncol)
        insol_col = insol.reshape(ncol)
        cos_sza_col = (
            cos_sza.reshape(ncol) if cos_sza is not None else None
        )
        f_day_col = f_day.reshape(ncol) if f_day is not None else None

        # NH plane state stores water vapor in tracer slot 0; q_c at 1,
        # q_i at 3 — mirroring the cubed-sphere NH tracer layout used
        # by ``_make_nonhydrostatic_radiation``.
        n_tracers = state.tracers.data.shape[-1]
        if n_tracers > 0:
            q_v = jnp.clip(state.tracers.data[..., 0], 0.0, None)
        else:
            q_v = jnp.zeros_like(T)
        q_v_col = q_v.reshape(ncol, nlev)

        q_cloud_col = None
        q_ice_col = None
        if n_tracers > 1:
            q_cloud_col = jnp.clip(
                state.tracers.data[..., 1], 0.0, None,
            ).reshape(ncol, nlev)
        if n_tracers > 3:
            q_ice_col = jnp.clip(
                state.tracers.data[..., 3], 0.0, None,
            ).reshape(ncol, nlev)
        # Double-moment NUMBER columns for the M2005 PSD effective radii, same
        # layout/units/guard as the cubed-sphere NH path: N_c per-VOLUME [#/m³]
        # at slot 6, N_i per-MASS [#/kg] at slot 8 — the canonical 9-slot layout
        # (q_v,q_c,q_r,q_i,q_s,q_g,N_c,N_r,N_i) shared by ALL double-moment
        # schemes (morrison, seifert_beheng, thompson, p3, ml_emulator). The
        # ``> 8`` guard keeps single-moment layouts (kessler/sundqvist, 3 slots)
        # on the constant r_eff. Without this, plane-CRM RRTMGP used a fixed
        # r_eff regardless of droplet number — unfaithful to SAM's PSD reffc/EFFI.
        n_cloud_col = None
        n_ice_col = None
        if n_tracers > 8:
            # Stored per MASS; the PSD wants per VOLUME (see the extractor).
            n_cloud_col = jnp.clip(
                state.tracers.data[..., 6], 0.0, None,
            ).reshape(ncol, nlev) * rho_total.reshape(ncol, nlev)
            n_ice_col = jnp.clip(
                state.tracers.data[..., 8], 0.0, None,
            ).reshape(ncol, nlev)

        if radiation_config.scheme == "mc3d":
            # 3D Monte-Carlo shortwave + gray longwave (plane LES/CRM only).
            dT_dt = _mc3d_plane_heating(
                radiation_config, grid, terrain_metric, rrtmgp_solver,
                p_full_col, p_half_col, T_col, T_sfc_col, lat_col, q_v_col,
                q_cloud_col, q_ice_col, n_cloud_col, n_ice_col,
                insol_col, cos_sza_col, rho_total, ny, nx, nlev,
                _time["day_of_year"], _time["seconds_of_day"],
            )
        else:
            rad_out = _call_radiation_backend(
                radiation_config=radiation_config,
                eccf=eccf,
                T=T_col, p_full=p_full_col, p_half=p_half_col,
                sfc_temperature=T_sfc_col, lat=lat_col, q_v=q_v_col,
                insolation=insol_col, cos_sza=cos_sza_col,
                q_cloud=q_cloud_col, q_ice=q_ice_col,
                n_cloud=n_cloud_col, n_ice=n_ice_col, f_day=f_day_col,
                rrtmgp_solver=rrtmgp_solver, lon=lon_col,
                ml_ozone_coefs=ml_ozone_coefs,
            )
            dT_dt = rad_out.heating_rate.reshape(shape_3d)

        dtheta_prime_dt = dT_dt / jnp.clip(exner, 1e-6, None)

        # Match the PlaneNonHydrostaticState field convention
        # (("y","x","z"), ("y","x","z_half"), ...) so summed tendencies
        # across surface_flux + radiation + microphysics carry
        # consistent metadata for the dycore composer.
        dims_3d = ("y", "x", "z")
        dims_w = ("y", "x", "z_half")
        dims_2d = ("y", "x")
            z_half=terrain_metric.z_half_3d,
        )
        T_sfc = _apply_T_sfc_override(T[..., -1], _T_sfc_override_cell[0])

        # MPAS lat/lon at cells handled by `_get_grid_lat_lon` via the
        # `hasattr(grid_or_mesh, 'latCell')` branch.
        lat, lon = _get_grid_lat_lon(mesh, shape_2d)
        insol, cos_sza, f_day, eccf = _compute_insolation(
            lat, radiation_config, lon=lon,
            day_of_year=_time["day_of_year"],
            seconds_of_day=_time["seconds_of_day"],
        )

        # Columns are already (ncol, nlev) — no reshape needed for the
        # cell-centred quantities.
        T_col = T
        p_full_col = p
        p_half_col = p_half
        T_sfc_col = T_sfc
        lat_col = lat
        lon_col = lon
        insol_col = insol
        cos_sza_col = cos_sza
        f_day_col = f_day

        # Tracer slot layout (matches the cubed-sphere NH variant):
        #   [0] q_v   [1] q_c   [2] q_r   [3] q_i
        # Codex review 2026-05-24 iter-3: cloud-aware backends MUST
        # have access to q_c (slot 1) AND q_i (slot 3); otherwise a
        # caller wiring ``RRTMGPConfig(include_clouds=True)`` with
        # fewer slots would get a silent CLEAR-SKY run instead of an
        # error. Hard-fail at call time before we lose the
        # information.
        n_tracers = state.tracers.data.shape[-1]
        _wants_clouds = (
            radiation_config.scheme == "rrtmgp"
            and getattr(radiation_config.rrtmgp, "include_clouds", False)
        )
        if _wants_clouds and n_tracers < 4:
            raise ValueError(
                f"Cloud-aware radiation (scheme='rrtmgp', "
                f"include_clouds=True) requires state.tracers with "
                f"at least 4 slots (q_v, q_c, q_r, q_i); MPAS NH "
                f"state carries {n_tracers}. Allocate the state with "
                f">= 4 tracers or disable cloud optics."
            )
        if n_tracers > 0:
            q_v = jnp.clip(state.tracers.data[..., 0], 0.0, None)
        else:
            q_v = jnp.zeros_like(T)
        q_v_col = q_v

        q_cloud_col = None
        q_ice_col = None
        if n_tracers > 1:
            q_cloud_col = jnp.clip(
                state.tracers.data[..., 1], 0.0, None,
            )
        if n_tracers > 3:
            q_ice_col = jnp.clip(
                state.tracers.data[..., 3], 0.0, None,
            )
        # Double-moment NUMBER columns for the M2005 PSD effective radii (same
        # slot layout/units/guard as the cubed-sphere NH + plane paths): N_c
        # per-VOLUME [#/m³] slot 6, N_i per-MASS [#/kg] slot 8 — the canonical
        # 9-slot layout shared by all double-moment schemes (morrison,
        # seifert_beheng, thompson, p3, ml_emulator); ``> 8`` guard keeps
        # single-moment schemes (kessler/sundqvist) on the constant r_eff.
        # MPAS columns are already (nCells, nlev) — no reshape needed.
        n_cloud_col = None
        n_ice_col = None
        if n_tracers > 8:
            # Stored per MASS; the PSD wants per VOLUME (see the extractor).
            from legoesm.atmosphere.physics._shared import compute_rho
            n_cloud_col = jnp.clip(
                state.tracers.data[..., 6], 0.0, None
            ) * compute_rho(T_col, p_full_col, q_v_col)
            n_ice_col = jnp.clip(state.tracers.data[..., 8], 0.0, None)

        rad_out = _call_radiation_backend(
            radiation_config=radiation_config,
            eccf=eccf,
            T=T_col, p_full=p_full_col, p_half=p_half_col,
            sfc_temperature=T_sfc_col, lat=lat_col, q_v=q_v_col,
            insolation=insol_col, cos_sza=cos_sza_col,
            q_cloud=q_cloud_col, q_ice=q_ice_col,
            n_cloud=n_cloud_col, n_ice=n_ice_col, f_day=f_day_col,
            rrtmgp_solver=rrtmgp_solver, lon=lon_col,
            ml_ozone_coefs=ml_ozone_coefs,
        )

        dT_dt = rad_out.heating_rate          # (nCells, nlev)
        dtheta_prime_dt = dT_dt / jnp.clip(exner, 1e-6, None)

        dims_cell = ("nCells", "nlev")
        dims_edge = ("nEdges", "nlev")
        dims_w = ("nCells", "nlev_half")
        dims_2d = ("nCells",)
        dims_tr = ("nCells", "nlev", "tracer")
        _sd = T.dtype
        _pd = state.phis.data.dtype

        return MPASNonHydrostaticTendencies(
            du_dt=Field(
                data=jnp.zeros(shape_edge_3d, dtype=_sd),
                name="du_dt_rad", dims=dims_edge, units="m/s^2",

 succeeded in 64235ms:
============================= test session starts ==============================
platform linux -- Python 3.12.14, pytest-9.1.1, pluggy-1.6.0
rootdir: /glade/derecho/scratch/pg2328/wt_volc
configfile: pyproject.toml
plugins: zarr-3.4.0, timeout-2.4.0, jaxtyping-0.3.11, xdist-3.8.0
collected 13 items

tests/unit/test_aerosol_band_ccn.py .............                        [100%]

=============================== warnings summary ===============================
tests/unit/test_aerosol_band_ccn.py::test_solar_weighted_and_visible_band_collapse
tests/unit/test_aerosol_band_ccn.py::test_solar_weighted_and_visible_band_collapse
tests/unit/test_aerosol_band_ccn.py::test_volcanic_sw_above_tropopause_and_out_of_ccn
tests/unit/test_aerosol_band_ccn.py::test_ccn_file_flows_from_cli_to_ccn
tests/unit/test_aerosol_band_ccn.py::test_ccn_file_flows_from_cli_to_ccn
tests/unit/test_aerosol_band_ccn.py::test_driver_volcanic_cut_at_state_tropopause
  /glade/derecho/scratch/pg2328/wt_volc/tests/unit/test_aerosol_band_ccn.py:55: DeprecationWarning: Setting the shape on a NumPy array has been deprecated in NumPy 2.5.
  As an alternative, you can create a new view using np.reshape (with copy=False if needed).
    ds.createVariable("aod", "f8", dims)[:] = np.broadcast_to(band, shape)

tests/unit/test_aerosol_band_ccn.py::test_solar_weighted_and_visible_band_collapse
  /glade/derecho/scratch/pg2328/wt_volc/tests/unit/test_aerosol_band_ccn.py:91: DeprecationWarning: Setting the shape on a NumPy array has been deprecated in NumPy 2.5.
  As an alternative, you can create a new view using np.reshape (with copy=False if needed).
    ds.createVariable("aod", "f8", ("time", "lnwl", "lat"))[:] = np.broadcast_to(

tests/unit/test_aerosol_band_ccn.py::test_volcanic_sw_above_tropopause_and_out_of_ccn
tests/unit/test_aerosol_band_ccn.py::test_volcanic_sw_above_tropopause_and_out_of_ccn
tests/unit/test_aerosol_band_ccn.py::test_volcanic_sw_above_tropopause_and_out_of_ccn
tests/unit/test_aerosol_band_ccn.py::test_driver_volcanic_cut_at_state_tropopause
tests/unit/test_aerosol_band_ccn.py::test_driver_volcanic_cut_at_state_tropopause
tests/unit/test_aerosol_band_ccn.py::test_driver_volcanic_cut_at_state_tropopause
  /glade/derecho/scratch/pg2328/wt_volc/tests/unit/test_volcanic_sw_optics.py:54: DeprecationWarning: Setting the shape on a NumPy array has been deprecated in NumPy 2.5.
  As an alternative, you can create a new view using np.reshape (with copy=False if needed).
    ds.createVariable(k, "f8", dims)[:] = v

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
======================= 13 passed, 13 warnings in 49.09s =======================

 succeeded in 139382ms:
============================= test session starts ==============================
platform linux -- Python 3.12.14, pytest-9.1.1, pluggy-1.6.0
rootdir: /glade/derecho/scratch/pg2328/wt_volc
configfile: pyproject.toml
plugins: zarr-3.4.0, timeout-2.4.0, jaxtyping-0.3.11, xdist-3.8.0
collected 21 items

tests/unit/test_volcanic_sw_optics.py .....................              [100%]

=============================== warnings summary ===============================
tests/unit/test_volcanic_sw_optics.py: 22 warnings
  /glade/derecho/scratch/pg2328/wt_volc/tests/unit/test_volcanic_sw_optics.py:54: DeprecationWarning: Setting the shape on a NumPy array has been deprecated in NumPy 2.5.
  As an alternative, you can create a new view using np.reshape (with copy=False if needed).
    ds.createVariable(k, "f8", dims)[:] = v

tests/unit/test_volcanic_sw_optics.py::test_legacy_column_file_is_not_an_optics_source
  /glade/derecho/scratch/pg2328/wt_volc/tests/unit/test_volcanic_sw_optics.py:100: DeprecationWarning: Setting the shape on a NumPy array has been deprecated in NumPy 2.5.
  As an alternative, you can create a new view using np.reshape (with copy=False if needed).
    ds.createVariable("aod", "f8", ("time", "lat"))[:] = 0.01

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
================= 21 passed, 23 warnings in 134.75s (0:02:14) ==================

codex
All 13 aerosol/CCN regression tests passed. The optics suite is still running with no failures reported. My chunk-size vote is A: keep 4096 as a module constant.
codex
1. No remaining actionable defects found in the current diff. CPU/fp64 verification: 21 optics tests and 13 aerosol/CCN tests passed. No files edited.

VOTE: A — keep `_VOLC_SW_COL_CHUNK = 4096` as a module constant.

VERDICT: APPROVE.
tokens used
104,959
1. No remaining actionable defects found in the current diff. CPU/fp64 verification: 21 optics tests and 13 aerosol/CCN tests passed. No files edited.

VOTE: A — keep `_VOLC_SW_COL_CHUNK = 4096` as a module constant.

VERDICT: APPROVE.
