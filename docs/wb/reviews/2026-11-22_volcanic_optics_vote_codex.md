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
session id: 01a20546-5e12-7dd3-83e5-62b8cc7cb282
--------
user
# Vote: volcanic SW aerosol optics from the CMIP6 file (legoESM, read-only, do not edit files)

User-approved change: volcanic stratospheric aerosol in the SHORTWAVE currently enters RRTMGP as a
solar-weighted GRAY extinction (ext_sun collapsed over 14 bands) ADDED into the single `aerosol_od`
(ncol,nlev) leaf, which then uses the grey tropospheric aerosol SSA=0.93 / g=0.70 (RRTMGPConfig).
The ETH CMIP6 file (volcanic_cmip6_from_cesm_1978-2014.nc: ext_sun/omega_sun/g_sun
(solar_bands=14, latitude=36, altitude=70, time=444)) carries per-band SSA and g. Bands are in
RRTMG/E3SM order (wl2_sun last band 12.195 um = 820 cm-1, i.e. 820-2600 cm-1 band LAST);
RRTMGP SW bands are ascending wavenumber 820-2680 first.
Code: packages/tools/legoesm/forcing/external.py (_load_volcanic_profile, _volcanic_profile_at_time,
get_aerosol_sw_volcanic_at_time), packages/tools/legoesm/forcing/surface_utils.py
(aerosol_layer_fields, place_stratospheric_aod_profile_to_layers, stratospheric_layer_weight),
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rte/two_stream.py
(_mix_sw_aerosol_and_clip), rrtmgp/rrtmgp.py (_resolve_aerosol_sw_optics, solve_columns).

Proposed design (author):
D1. New optional forcing field `aerosol_sw_volc` (ncol, nlev, 3, 14): per-band volcanic moments
    (tau, tau*ssa, tau*ssa*g) per model layer, placed by the existing conservative log-p overlap remap
    and tropopause weight applied to each moment (remap is linear => mixing exact). Threaded like
    `aerosol_lw_od` through SegmentForcing / physics_pipeline / integration / RRTMGP.solve_columns /
    two_stream (solve_sw + compute_sw_optical_field). Volcanic is then REMOVED from `aerosol_od`
    (trop only). In _mix_sw_aerosol_and_clip the volcanic moments of the g-point's band are added as a
    second aerosol component: tau_tot += tau_v, w_num += (tau ssa)_v, g_num += (tau ssa g)_v.
    No config knob: it is THE behaviour whenever the volcanic file has ext_sun+omega_sun+g_sun.
Detail choices — vote A/B (or propose C) on each, one line each with reason:
Q1 band mapping: (A) reorder file bands to ascending wavenumber and map 1:1 to RRTMGP band index,
   raising unless the file has 14 bands with wl edges and each sorted file band centre falls inside the
   matching RRTMGP band limits (checked in RRTMGP against gas_optics_sw band_lims_wvn); (B) overlap-weighted
   remap of file bands onto RRTMGP bands.
Q2 file has ext_sun but no omega_sun/g_sun: (A) keep today's gray path in aerosol_od (grey 0.93/0.70)
   with a logged warning; (B) raise.
Q3 storage dtype of the cached (time,lat,alt,3,14) moments table (~235 MB f32 / 470 MB f64 per rank):
   (A) float32 (file is float32); (B) float64.
Q4 field absent (no volcanic file): (A) None leaf (pytree static, no memory) ; (B) zeros (ncol,nlev,3,14).
Q5 radiation lanes that cannot consume it (gray, zero, ecRad/other, neural): (A) raise if the leaf is
   non-None and radiation is not RRTMGP; (B) silently ignore.
Q6 NaN fill in file (fill values below tropopause / extrapolated tail): (A) treat NaN as 0 OD before
   moments (existing loader behaviour? check) ; (B) raise.
Q7 test tolerances: SW flux closure (TOA net - sfc net - column absorption) at x64 rtol 1e-10;
   equivalence test: with volcanic ssa=0.93,g=0.70 and gray tau the new path reproduces the old path to
   rtol 1e-10; (A) accept; (B) propose others.
Also: flag any design defect (wrong physics, budget, gradient, band order) in D1.
Reply format: Q1: A|B|C - reason ... then DEFECTS: list.
codex
I’ll check the loader, remapping, and radiation code, then give a read-only vote on each choice.
exec
/usr/bin/bash -lc 'pwd && git status --porcelain && git worktree list && cat CLAUDE.md' in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 1043ms:
/glade/derecho/scratch/pg2328/wt_volc
?? docs/wb/reviews/2026-11-22_volcanic_optics_vote_codex.md
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
/glade/derecho/scratch/pg2328/wt_ocean_vadv                                                                                                         a87671b16c [fix/ocean-vertical-tracer-conservation]
/glade/derecho/scratch/pg2328/wt_spec_main                                                                                                          5b56a56a79 (detached HEAD)
/glade/derecho/scratch/pg2328/wt_volc                                                                                                               a87671b16c [feat/volcanic-sw-file-optics]
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
/usr/bin/bash -lc "wc -l CLAUDE.md; sed -n '1,230p' CLAUDE.md" in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 0ms:
1254 CLAUDE.md
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

exec
/usr/bin/bash -lc "rg -n 'def (_load_volcanic_profile|_volcanic_profile_at_time|get_aerosol_sw_volcanic_at_time|aerosol_layer_fields|place_stratospheric_aod_profile_to_layers|stratospheric_layer_weight|_mix_sw_aerosol_and_clip|_resolve_aerosol_sw_optics|solve_columns)|nan_to_num|wl[12]_sun' packages/tools/legoesm/forcing/{external,surface_utils}.py packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/{rrtmgp.py,rte/two_stream.py}" in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 0ms:
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rte/two_stream.py:659:def _mix_sw_aerosol_and_clip(
packages/tools/legoesm/forcing/surface_utils.py:494:def place_stratospheric_aod_profile_to_layers(
packages/tools/legoesm/forcing/surface_utils.py:770:def stratospheric_layer_weight(p_half_col: jnp.ndarray,
packages/tools/legoesm/forcing/surface_utils.py:808:def aerosol_layer_fields(acfg, day, lat_col, p_half_col, w_strat, *,
packages/tools/legoesm/forcing/external.py:167:    edges ``wl1_sun``/``wl2_sun`` [um] on that dim: ``band="gray"`` -> solar-
packages/tools/legoesm/forcing/external.py:171:    if "wl1_sun" not in ds or ds["wl1_sun"].dims != (band_dim,):
packages/tools/legoesm/forcing/external.py:173:    wl1 = np.asarray(ds["wl1_sun"].values, dtype=np.float64)
packages/tools/legoesm/forcing/external.py:174:    wl2 = np.asarray(ds["wl2_sun"].values, dtype=np.float64)
packages/tools/legoesm/forcing/external.py:534:    band dim carrying ``wl1_sun``/``wl2_sun`` edges is instead collapsed by
packages/tools/legoesm/forcing/external.py:682:       the file carries ``wl1_sun``/``wl2_sun`` band edges.
packages/tools/legoesm/forcing/external.py:780:    # weighted when the file carries ``wl1_sun``/``wl2_sun``, else the
packages/tools/legoesm/forcing/external.py:822:def _load_volcanic_profile(
packages/tools/legoesm/forcing/external.py:830:    (flat mean if the file has no ``wl1_sun``/``wl2_sun``).  The B1/B3 notes
packages/tools/legoesm/forcing/external.py:2344:def get_aerosol_sw_volcanic_at_time(config: AerosolConfig, day: float,
packages/tools/legoesm/forcing/external.py:2358:def _volcanic_profile_at_time(config, day, lat_grid, kind):
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:312:def _resolve_aerosol_sw_optics(config, optics_lib):
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:585:  def solve_columns(
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:1312:  def solve_columns_chunked(

exec
/usr/bin/bash -lc "sed -n '822,1010p' packages/tools/legoesm/forcing/external.py; sed -n '2340,2425p' packages/tools/legoesm/forcing/external.py; sed -n '494,650p' packages/tools/legoesm/forcing/surface_utils.py; sed -n '770,925p' packages/tools/legoesm/forcing/surface_utils.py" in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 0ms:
def _load_volcanic_profile(
    path: str, kind: str = "lw",
) -> tuple[np.ndarray, object, np.ndarray, np.ndarray, np.ndarray]:
    """CMIP6 volcanic LW aerosol as a PRESSURE-RESOLVED absorption-OD profile.

    ``kind="sw"`` returns the SHORTWAVE ``ext_sun`` EXTINCTION profile instead
    (the SW slot takes extinction; SSA/g come from the radiation config), with
    the solar bands collapsed solar-weighted via :func:`_collapse_sw_bands`
    (flat mean if the file has no ``wl1_sun``/``wl2_sun``).  The B1/B3 notes
    below are LW-only; B2 (stratospheric placement) applies to both.

    Fixes three defects in feeding ``bc_aeropt_cmip6_volc_lw_b16`` to the
    RRTMGP longwave ABSORPTION optical-depth slot:

    * **B1 extinction -> absorption.**  ``ext_earth`` is EXTINCTION; the LW slot
      wants ABSORPTION (single-scattering albedo 0).  Scaled per band by the
      co-located ``omega_earth``: ``abs = ext*(1-omega)``.  ``omega_earth``
      reaches ~0.4 in the window bands, so feeding raw extinction overstates
      the absorption OD by up to that factor.  If the file lacks
      ``omega_earth`` the sulfate LW ``omega ~ 0`` approximation is used
      (``abs == ext``) and documented.
    * **B2 stratospheric placement.**  The file is HEIGHT-RESOLVED (altitude
      5-40 km).  The per-layer profile is retained and each file layer tagged
      with a US-Std-Atm-1976 pressure edge, so the driver places the aerosol
      at its true stratospheric pressure instead of spreading a column AOD by
      tropospheric pressure mass (which lands ~90 % of it below the
      tropopause).
    * **B3 spectral collapse.**  The 16 terrestrial bands ARE collapsed to one
      gray value (the LW slot is applied to every g-point), but via a
      Planck-emission-weighted mean at :data:`_VOLC_LW_PLANCK_TEMP_K` rather
      than a flat band mean.  This is better-motivated than a flat mean (which
      overweights the near-IR bands that carry ~no LW flux at stratospheric
      temperature) but is a HEURISTIC, not the exact gray equivalent -- the
      exact single-tau response would fold in the full spectral radiative
      kernel (surface/gas/cloud transmission + local emission), not Planck
      emission alone.  Residual per-band spread is a documented approximation;
      threading true per-band tau through the g-point loop is deferred as
      disproportionate for this background-magnitude forcing.

    Returns ``(mid_days, first_date, lat, aod_profile, p_edges)``:

    * ``aod_profile`` ``(ntime, nlat, nlayer)`` per-layer absorption OD [-];
    * ``p_edges`` ``(nlayer+1,)`` [Pa] layer pressure edges, ASCENDING (index 0
      = top / lowest pressure), aligned so ``aod_profile[..., j]`` occupies
      ``[p_edges[j], p_edges[j+1]]``.

    Raises ``ValueError`` if the file lacks ``ext_earth`` (``ext_sun`` for
    SW; caller treats as no profile source and stays byte-identical).
    """
    ext_name = "ext_sun" if kind == "sw" else "ext_earth"
    ds = _open_forcing_dataset(path)
    if ext_name not in ds.data_vars:
        ds.close()
        raise ValueError(
            f"No CMIP6 volcanic extinction {ext_name!r} in {path!r}.",
        )
    var = ds[ext_name]
    ext = np.asarray(var.values, dtype=np.float64)  # [1/km]
    dims = list(var.dims)

    # B1: single-scattering albedo -> absorption fraction (1-omega) per band.
    if kind == "sw":
        omega = 0.0  # SW slot wants extinction, not absorption
    elif "omega_earth" in ds.data_vars:
        omega = np.clip(
            np.asarray(ds["omega_earth"].values, dtype=np.float64), 0.0, 1.0,
        )
    else:
        omega = np.zeros_like(ext)  # sulfate LW omega ~ 0 approximation
    abs_ext = ext * (1.0 - omega)  # absorption extinction [1/km]

    alt_name = next(
        (v for v in ("altitude", "alt", "height", "lev") if v in ds), None,
    )
    if alt_name is None:
        ds.close()
        raise ValueError(f"No altitude coordinate in volcanic file {path!r}")
    alt_km = np.asarray(ds[alt_name].values, dtype=np.float64)

    if "lat" in ds:
        lat = np.asarray(ds["lat"].values, dtype=np.float64)
        lat_name = "lat"
    elif "latitude" in ds:
        lat = np.asarray(ds["latitude"].values, dtype=np.float64)
        lat_name = "latitude"
    else:
        ds.close()
        raise ValueError(f"No 'lat'/'latitude' in volcanic file {path!r}")

    if "time" in dims:
        time_name = "time"
        mid_days, first_date = _read_time_axis(ds)
    elif "month" in dims:
        time_name = "month"
        nm = ext.shape[dims.index("month")]
        mid_days = np.array([15.5 + 30.4375 * m for m in range(nm)])
        first_date = None
    else:
        ds.close()
        raise ValueError(f"No time/month dimension in volcanic file {path!r}")

    # B3: Planck-emission-weighted collapse of the terrestrial-band axis.
    band_name = next(
        (d for d in dims
         if d not in (time_name, lat_name, alt_name)),
        None,
    )
    if band_name is not None and kind == "sw":
        collapsed = _collapse_sw_bands(ds, abs_ext, dims, band_name)
        abs_ext = (np.nanmean(abs_ext, axis=dims.index(band_name))
                   if collapsed is None else collapsed)
        dims = [d for d in dims if d != band_name]
    elif band_name is not None:
        band_axis = dims.index(band_name)
        if "wl1_earth" in ds and "wl2_earth" in ds:
            weights = _planck_band_weights(
                np.asarray(ds["wl1_earth"].values),
                np.asarray(ds["wl2_earth"].values),
                _VOLC_LW_PLANCK_TEMP_K,
            )
        else:
            weights = np.ones(abs_ext.shape[band_axis], dtype=np.float64)
        # Weighted mean over bands (band axis -> gray).
        abs_ext = np.tensordot(
            weights, np.moveaxis(abs_ext, band_axis, 0), axes=(0, 0),
        ) / weights.sum()
        dims = [d for d in dims if d != band_name]
    ds.close()

    # Canonical (time, lat, altitude) order.
    order = [dims.index(time_name), dims.index(lat_name), dims.index(alt_name)]
    prof = np.transpose(abs_ext, order)  # (ntime, nlat, nalt) [1/km]

    # Altitude layer edges (midpoints, outer half-steps) and thickness [km].
    edges_km = np.empty(alt_km.size + 1, dtype=np.float64)
    edges_km[1:-1] = 0.5 * (alt_km[:-1] + alt_km[1:])
    edges_km[0] = alt_km[0] - 0.5 * (alt_km[1] - alt_km[0])
    edges_km[-1] = alt_km[-1] + 0.5 * (alt_km[-1] - alt_km[-2])
    dz_km = np.abs(np.diff(edges_km))  # (nalt,)

    # Per-layer AOD [-] = absorption extinction [1/km] * dz [km].
    aod_profile = prof * dz_km[None, None, :]
    p_edges = _ussa1976_pressure(edges_km * 1000.0)  # [Pa]

    # Return ASCENDING pressure (top first): altitude ascending -> pressure
    # descending, so flip the layer axis and the edges together.
    aod_profile = aod_profile[..., ::-1]
    p_edges = p_edges[::-1]
    return mid_days, first_date, lat, aod_profile, p_edges


@lru_cache(maxsize=16)
def _load_volcanic_auto(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dispatch ``path`` to the right volcanic loader.

    Returns a ``(mid_days, lat, data)`` tuple (the anchor-free form of
    :func:`_load_monthly_zonal_anchored`) regardless of which on-disk
    schema the file follows.  See :func:`_load_volcanic_auto_anchored` for a
    variant that also returns the file's CF anchor (used by the
    multi-year non-cyclic dispatch in :func:`get_aerosol_at_time`).
    """
    mid_days, _first_date, lat, data = _load_volcanic_auto_anchored(path)
    return mid_days, lat, data


@lru_cache(maxsize=16)
def _load_volcanic_auto_anchored(
    path: str,
) -> tuple[np.ndarray, object, np.ndarray, np.ndarray]:
    """Like :func:`_load_volcanic_auto` but also returns ``first_date``.

    Criterion: presence of the CMIP6 ``ext_sun`` variable triggers
    :func:`_load_volcanic_cmip6_anchored`; otherwise fall back to the
    legacy ``aod(time, lat)`` schema via
    :func:`_load_monthly_zonal_anchored`.  Cached so that repeated
    calls in an AMIP time-step loop don't re-open the dataset just to
    dispatch.
    """
    ds = _open_forcing_dataset(path)
    dvars = set(ds.data_vars)
    ds.close()
    if "ext_sun" in dvars:
        return _load_volcanic_cmip6_anchored(path)
    if "ext_earth" in dvars and "aod" not in dvars:
        raise ValueError(
            f"Volcanic file {path!r} has only LW extinction "
            "('ext_earth') and no SW extinction ('ext_sun'); cannot be "
            "used as a SW AOD source.",
        )
        return None
    return _volcanic_profile_at_time(config, day, lat_grid, "lw")


def get_aerosol_sw_volcanic_at_time(config: AerosolConfig, day: float,
                                    lat_grid: jnp.ndarray | None = None):
    """SHORTWAVE volcanic extinction as a pressure-resolved profile.

    Same contract as :func:`get_aerosol_lw_at_time` but for ``ext_sun``
    (solar-weighted gray band collapse), gated on ``volcanic_enabled``.  None
    for a legacy column ``aod(time, lat)`` volcanic file (no profile) -- the
    caller then keeps that column inside :func:`get_aerosol_at_time`.
    """
    if not config.volcanic_enabled or not config.volcanic_path:
        return None
    return _volcanic_profile_at_time(config, day, lat_grid, "sw")


def _volcanic_profile_at_time(config, day, lat_grid, kind):
    """Shared body of the LW / SW volcanic profile samplers."""
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
            jnp.clip(prof_col, 0.0, None), (-1, prof_t.shape[-1]),
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
        Path to forcing NetCDF file.
    tsi_var : str
        Variable name for TSI time series.
    spectral_var : str
        Variable name for per-g-point solar fractions when source="spectral_file".
    spectral_band_order : str
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

        C dT_s/dt = F_net_down(atm) + (k_i / h) * (T_base - T_s)

    Signs (surface conventions of the exported ``_sfc_diag`` fluxes):
    ``F_net_down_W_m2 = sw_net_sfc + lw_net_sfc - shflx - lhflx`` — sw/lw net
    positive INTO the surface, turbulent fluxes positive UPWARD out of it, so
    ``F_net_down`` is the net energy gain of the skin from the atmosphere.
    Winter polar night: F_net_down < 0, the skin cools below ``T_freeze_K``
    until conduction from the ocean balances the loss — the equilibrium
    ``T_s = T_base + F_net_down * h / k_i`` of the classic zero-layer model.

    Discretization: backward-Euler in the CONDUCTIVE term (unconditionally
    stable for any ``dt_s``; the atmospheric flux is the lagged explicit
    forcing, refreshed by the physics each step):

        T_new = (T_s + (dt/C)*(F_net + (k_i/h)*T_base)) / (1 + (dt/C)*k_i/h)

    The FULL coupled system (skin + atmosphere recomputing F_net at the new
    skin) is only CONDITIONALLY stable, and the threshold depends on WHEN the
    atmosphere sees the new skin.  If the caller refreshes the surface anchor
    at the SAME cadence as this advance — the MPAS driver re-blends T_sfc every
    model step — the per-step amplification is ``|1 - r*lambda|/(1 + r*g)`` for
    a surface-flux feedback ``lambda = -dF_net/dT_s > 0`` (LW + sensible),
    ``r = dt/C``, ``g = k_i/h``; at the short model step ``dt_s = DT`` this
    keeps ``r*lambda << 1``, stable and monotone for any realistic ``lambda``.
    If instead the anchor is HELD while the skin advances sub-cadence (e.g.
    per-step advance under a once-daily anchor), the effective map reverts to
    the coarse-cadence thresholds (order tens of W/m^2/K, tightening as h
    shrinks) and the melt cap only BOUNDS — does not cure — a cap/floor
    oscillation.  Advance and anchor-refresh cadence MUST match.

    Bounds: melt cap ``T_new <= T_melt_surface_K`` — the FRESH-ICE SURFACE
    melting point (``constants.T_freeze``, i.e. 0 C), NOT the basal seawater
    freezing point ``T_freeze_K`` (``constants.T_freeze_ocean``).  Distinct
    boundary conditions: conduction is toward the warmer seawater-freezing
    base, the surface melts at 0 C (mirrors ``ice/sea_ice.py`` ``T_base`` vs
    ``T_melt_surface``).  A melting skin sheds energy at 0 C — Semtner's melt
    branch, whose meltwater bookkeeping a prescribed-ice run does not carry.
    ``_ICE_SKIN_FLOOR_K`` guards a pathological cold spike.  Open water
    (``sic <= 0``) snaps to ``T_freeze_K`` (seawater freezing) so a cell
    freezing later starts from the freezing point, not a stale skin.

    Limitations. (1) Single-tile: ``F_net_down`` is the BLENDED-cell surface
    flux (computed at ``sic*T_skin + (1-sic)*sst``), used here as the ice-tile
    forcing.  Exact only at ``sic=1``; at marginal ice the true ice-tile
    LW/turbulent flux differs — a documented closure error, most consequential
    during seasonal advance/retreat, second-order because the skin enters the
    anchor weighted by ``sic``.  (2) ``h_ice_m`` is a single global
    climatological thickness (no Arctic~2 m / Antarctic~1 m asymmetry).
    (3) Differentiability: the interior is smooth (``dT_new/dF = r/(1+r*g)``,
def stratospheric_layer_weight(p_half_col: jnp.ndarray,
                               lat_rad: jnp.ndarray,
                               p_full_col: jnp.ndarray | None = None,
                               temp_col: jnp.ndarray | None = None,
                               ) -> jnp.ndarray:
    """Per-layer weight for CMIP6 stratospheric (volcanic) aerosol at the
    tropopause, as the ETH/SAGE-3lambda data notes prescribe: 1 above the
    tropopause, 0.5 in the layer holding it, 0 below (the file's values below
    the tropopause are an extrapolated matching tail, not tropospheric
    aerosol).  ``p_half_col`` (ncol, nlev+1) [Pa] in either vertical order,
    ``lat_rad`` (ncol,).

    The tropopause is the column's WMO lapse-rate tropopause
    (``wmo_tropopause_pressure``) when ``p_full_col``/``temp_col`` (ncol, nlev)
    are given, falling back per column to the climatological
    p_tp = 300 - 200 cos^2(lat) hPa (100 hPa tropics, 300 hPa poles) where
    none is found or no T is given.

    ponytail: no temporal smoothing -- the ETH notes ask for the
    instantaneous tropopause.  Measured (scripts/validate/
    check_wmo_tropopause_fv3duo.py, code e4d7a50ff, x64) on FV3-duo C24 L32
    daily checkpoints duo_5d days 1-5: 0% of 3456 columns fall back (also 0%
    at duo_60d days 5-30); the tropopause layer moves >=1 level day-to-day in
    30% of columns, >=2 in 4% (30-60 deg 43%/7%, tropics 20%/2%), 24% of
    moves flip back next day -- consistent with synoptic motion at L32 layer
    spacing; threshold chatter is not separately quantified.  Smooth (e.g. a
    running mean of p_tp) only if that proves material.
    """
    p_tp = (300.0 - 200.0 * jnp.cos(lat_rad) ** 2) * 100.0
    if temp_col is not None:
        p_wmo = wmo_tropopause_pressure(p_full_col, temp_col)
        p_tp = jnp.where(jnp.isnan(p_wmo), p_tp, p_wmo)
    p_tp = p_tp[:, None]
    top = jnp.minimum(p_half_col[:, :-1], p_half_col[:, 1:])
    bot = jnp.maximum(p_half_col[:, :-1], p_half_col[:, 1:])
    return jnp.where(bot <= p_tp, 1.0, jnp.where(top >= p_tp, 0.0, 0.5))


def aerosol_layer_fields(acfg, day, lat_col, p_half_col, w_strat, *,
                         active, lw_active, dtype, lon_col=None):
    """Per-layer aerosol forcing for one day: ``(aerosol_od, ccn_aod,
    aerosol_lw_od)``, as the AMIP driver feeds radiation and the aerosol-CCN
    droplet number.  ``ccn_aod`` is None when ``active`` is off; the two
    optical-depth fields are zeros (ncol, nlev) when their switch is off.
    ``lon_col`` [rad] reads the SW and CCN aerosol at each column's longitude;
    None reads their zonal mean."""
    from legoesm.forcing.external import (
        get_aerosol_at_time,
        get_aerosol_lw_at_time,
        get_aerosol_sw_volcanic_at_time,
    )

    ncol, nlev = p_half_col.shape[0], p_half_col.shape[1] - 1
    ccn_aod = None
    aerosol_od = jnp.zeros((ncol, nlev), dtype=dtype)
    if active:
        _acfg = acfg
        # Volcanic SW (CMIP6 ``ext_sun``) is placed at its true
        # stratospheric pressure like the LW profile below, NOT spread
        # with the tropospheric column by pressure mass.  A legacy
        # column-only volcanic file (profile None) stays in the column.
        volc_sw = get_aerosol_sw_volcanic_at_time(_acfg, day, lat_grid=lat_col)
        _trop_cfg = _acfg._replace(volcanic_enabled=False)
        aerosol_col = get_aerosol_at_time(
            _acfg if volc_sw is None else _trop_cfg, day, lat_grid=lat_col,
            lon_grid=lon_col,
        )
        aerosol_od = distribute_column_aod_to_layers(
            jnp.asarray(aerosol_col), p_half_col,
        )
        if volc_sw is not None:
            aerosol_od = aerosol_od + (place_stratospheric_aod_profile_to_layers(
                jnp.asarray(volc_sw[0]), jnp.asarray(volc_sw[1]), p_half_col,
            ) * w_strat
            ).astype(aerosol_od.dtype)
        # AOD->CCN input: TROPOSPHERIC column in the visible (550 nm)
        # band, volcanic excluded, from ``ccn_path`` if set.  The Andreae 2009
        # fit is local total AOT: a zonal mean gives polluted E China and the
        # clean N Pacific one droplet number.
        ccn_aod = jnp.asarray(get_aerosol_at_time(
            _trop_cfg._replace(path=_acfg.ccn_path or _acfg.path),
            day, lat_grid=lat_col, band="vis", lon_grid=lon_col,
        )).reshape(ncol).astype(dtype)

    # Volcanic stratospheric LONGWAVE aerosol (gap #9): same (ncol,
    # nlev) shape as ``aerosol_od``, default zeros so the SegmentForcing
    # / forcing-dict leaf is a concrete fixed-shape array (no retrace)
    # and a run without volcanic LW aerosol is byte-identical (zeros LW
    # od is a RRTMGP no-op).  ``get_aerosol_lw_at_time`` returns the
    # file's height-resolved ABSORPTION profile (ext*(1-omega), Planck-
    # weighted gray band collapse) plus its pressure edges; the profile
    # is placed at its true stratospheric pressure by a conservative
    # overlap remap -- NOT spread by full-column pressure mass, which
    # dumped ~90% of a stratospheric aerosol into the troposphere.
    aerosol_lw_od = jnp.zeros((ncol, nlev), dtype=dtype)
    if lw_active:
        aerosol_lw_prof = get_aerosol_lw_at_time(
            acfg, day, lat_grid=lat_col,
        )
        if aerosol_lw_prof is not None:
            prof_col, p_edges = aerosol_lw_prof
            aerosol_lw_od = (place_stratospheric_aod_profile_to_layers(
                jnp.asarray(prof_col), jnp.asarray(p_edges), p_half_col,
            ) * w_strat
            ).astype(dtype)
    return aerosol_od, ccn_aod, aerosol_lw_od

exec
/usr/bin/bash -lc "sed -n '620,800p' packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rte/two_stream.py; sed -n '300,365p' packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py; rg -n 'aerosol|delta|band_lims' packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rte/two_stream.py" in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 0ms:
    pressure: Array,
    temperature: Array,
    molecules: Array,
    vmr_fields: dict | None,
    cloud_r_eff_liq: Array | None,
    cloud_path_liq: Array | None,
    cloud_r_eff_ice: Array | None,
    cloud_path_ice: Array | None,
    cloud_fraction: Array | None,
    aerosol_optical_depth: Array | None,
    aerosol_single_scattering_albedo: float | Array,
    aerosol_asymmetry_factor: float | Array,
    cloud_snow=None,
) -> dict[str, Array]:
  """Per-g-point shortwave optical properties (optical_depth / ssa / asymmetry).

  Shared by ``solve_sw`` (two-stream transport) and ``compute_sw_optical_field``
  (3D Monte-Carlo ray tracer) so the gas+cloud optics lookup, aerosol mixing,
  and physical-range clipping are defined exactly once (no duplicated numerics).
  ``vmr_fields`` must already be reindexed to RRTM gas indices by the caller.
  """
  sw_optical_props = optics_lib.compute_sw_optical_properties(
      pressure,
      temperature,
      molecules,
      igpt,
      vmr_fields,
      cloud_r_eff_liq,
      cloud_path_liq,
      cloud_r_eff_ice,
      cloud_path_ice,
      cloud_fraction=cloud_fraction,
      **({} if cloud_snow is None else {'cloud_snow': cloud_snow}),
  )
  return _mix_sw_aerosol_and_clip(
      igpt, optics_lib, sw_optical_props, aerosol_optical_depth,
      aerosol_single_scattering_albedo, aerosol_asymmetry_factor)


def _mix_sw_aerosol_and_clip(
    igpt: Array,
    optics_lib,
    sw_optical_props: dict[str, Array],
    aerosol_optical_depth: Array | None,
    aerosol_single_scattering_albedo: float | Array,
    aerosol_asymmetry_factor: float | Array,
) -> dict[str, Array]:
  """Mix aerosol into one g-point's SW gas(+cloud) optics, then clip ssa/g."""
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
      'ssa': jnp.clip(sw_optical_props['ssa'], 0.0, 1.0),
      'asymmetry_factor': jnp.clip(
          sw_optical_props['asymmetry_factor'], -1.0, 1.0,
      ),
  }
  return sw_optical_props


def compute_sw_optical_field(
    pressure: Array,
    temperature: Array,
    molecules: Array,
    optics_lib: optics_base.OpticsScheme,
    vmr_fields: dict[str, Array] | None = None,
    cloud_r_eff_liq: Array | None = None,
    cloud_path_liq: Array | None = None,
    cloud_r_eff_ice: Array | None = None,
    cloud_path_ice: Array | None = None,
    cloud_fraction: Array | None = None,
    aerosol_optical_depth: Array | None = None,
    aerosol_single_scattering_albedo: float | Array = _AEROSOL_SSA_DEFAULT,
    aerosol_asymmetry_factor: float | Array = _AEROSOL_ASYM_DEFAULT,
) -> dict[str, Array]:
  """Stack per-g-point shortwave optical fields for the 3D MC ray tracer.

  Runs the SAME per-g-point optics path as ``solve_sw`` (via
  ``compute_sw_optical_props_gpt``) but, instead of solving two-stream
  transport, stacks the optical depth / ssa / asymmetry over all shortwave
  g-points. Returns a dict with keys ``optical_depth`` / ``ssa`` /
  ``asymmetry_factor``, each shaped ``(n_gpt_sw, X, Y, Z)`` (same X,Y,Z as the
  input fields, including any vertical halo cells — the caller strips them).
  """
  optics_lib = cast(optics.RRTMOptics, optics_lib)
  if vmr_fields is not None:
    vmr_fields = _reindex_vmr_fields(vmr_fields, optics_lib.gas_optics_sw)
  n_gpt = optics_lib.n_gpt_sw
  optics_lib = optics_lib.with_all_gpts(False, pressure, temperature, molecules, vmr_fields)

  def gpt_props(igpt):
    props = compute_sw_optical_props_gpt(
        igpt, optics_lib, pressure, temperature, molecules, vmr_fields,
        cloud_r_eff_liq, cloud_path_liq, cloud_r_eff_ice, cloud_path_ice,
        cloud_fraction, aerosol_optical_depth,
        aerosol_single_scattering_albedo, aerosol_asymmetry_factor,
    )
    # Gas Rayleigh scattering optical depth (the gas scatter is ALL Rayleigh) as
    # a fraction of TOTAL scattering (gas + cloud + aerosol) -> the per-cell
    # probability that a scatter event is Rayleigh, for the explicit-phase split.
    rayl_scat = optics_lib.gas_gpt('rayleigh', igpt)
    tot_scat = props['ssa'] * props['optical_depth']
    # safe_divide (not a floored raw divide) so the masked-branch VJP can't
    # overflow as tot_scat -> 0, consistent with ssa/g above. fill=1.0: a cell
    # with negligible total scattering is gas-only -> any scatter there is
    # Rayleigh (matches the old clip-to-1 limit, not 0).
    rayleigh_frac = jnp.clip(
        safe_divide(rayl_scat, tot_scat, eps=1.0e-30, fill=1.0), 0.0, 1.0)
    return {**props, 'rayleigh_frac': rayleigh_frac}

  # Sequential map over g-points: stacks leaves to (n_gpt, X, Y, Z).
  return jax.lax.map(gpt_props, jnp.arange(n_gpt))


def compute_lw_optical_field(
    pressure: Array,
    temperature: Array,
    molecules: Array,
    optics_lib: optics_base.OpticsScheme,
    sfc_temperature: Array,
    vmr_fields: dict[str, Array] | None = None,
    cloud_r_eff_liq: Array | None = None,
    cloud_path_liq: Array | None = None,
    cloud_r_eff_ice: Array | None = None,
    # by the callable guard above when recursed into.
    if hasattr(obj, "__dict__"):
        for _a, _v in vars(obj).items():
            if callable(_v):
                continue
            _nv = _cast_optics_f64_to_f32(_v, _seen)
            if _nv is not _v:
                setattr(obj, _a, _nv)
        return obj
    return obj


def _resolve_aerosol_sw_optics(config, optics_lib):
    """SW aerosol single-scattering albedo + asymmetry fed to the two-stream
    solver.  Returns Python-float SCALARS (grey aerosol, the historical default
    -> byte-identical) or ``(n_bnd_sw,)`` per-band jnp arrays when
    ``config.aerosol_ssa_bands`` / ``aerosol_g_bands`` are set.  The per-band
    length is validated against the loaded SW gas-optics band count (a wrong
    length is a config error, not a silent out-of-range gather)."""
    ssa = config.aerosol_ssa
    g = config.aerosol_g
    bands_ssa = getattr(config, "aerosol_ssa_bands", None)
    bands_g = getattr(config, "aerosol_g_bands", None)
    if bands_ssa is None and bands_g is None:
        return ssa, g
    n_bnd = int(optics_lib.gas_optics_sw.n_bnd)
    dtype = optics_lib.gas_optics_sw.kmajor.dtype
    if bands_ssa is not None:
        if len(bands_ssa) != n_bnd:
            raise ValueError(
                f"RRTMGPConfig.aerosol_ssa_bands has length {len(bands_ssa)}, "
                f"expected the SW band count {n_bnd}.")
        ssa = jnp.asarray(bands_ssa, dtype=dtype)
    if bands_g is not None:
        if len(bands_g) != n_bnd:
            raise ValueError(
                f"RRTMGPConfig.aerosol_g_bands has length {len(bands_g)}, "
                f"expected the SW band count {n_bnd}.")
        g = jnp.asarray(bands_g, dtype=dtype)
    return ssa, g


class RRTMGP:
  """Rapid Radiative Transfer Model for General Circulation Models (RRTMGP).

  legoESM integration API — construct with
  ``RRTMGP.from_legoesm_config(rrtmgp_config)`` and drive via
  ``solve_columns(...)``.  The swirl_jatmos-style ``__init__`` and
  ``compute_heating_rate`` API were removed in iter-22 after zero
  callers were found across the codebase.
  """

  @staticmethod
  def _optics_cache_key(config):
      """Hashable key for the optics-table cache.

      Includes only fields that change the loaded NetCDF tables
      (gas files, cloud files, table-shape inputs) plus the active
      x64 precision (table dtype depends on it).  Solver-behavior
      fields like ``use_scan`` deliberately omitted — the tables
      themselves are independent of how the solver traverses them.

      Iter-40: re-introduced ``include_clouds`` here BUT for a
      different reason than the pre-iter-36 state — iter-40 made
      ``RRTMOptics.__init__`` skip the cloud-table load when
      ``include_clouds=False``, so the constructed optics_lib is
40:        "gas/cloud/aerosol optics to Meador-Weaver reflectance/transmittance and "
50:        "cloud_fraction": "1", "aerosol_optical_depth": "1",
81:# Default aerosol optical properties (fixed scheme defaults).
372:    aerosol_absorption_optical_depth: Array | None = None,
416:    aerosol_absorption_optical_depth: Optional prescribed longwave aerosol
419:      albedo 0.  Longwave aerosol scattering is neglected (the dominant LW
420:      aerosol effect is absorption/emission); a caller holding extinction
465:    if aerosol_absorption_optical_depth is not None:
466:      # Prescribed longwave aerosol as a pure-absorbing layer
470:      # is unchanged because the aerosol contributes no scattering
473:      # the per-band diffusivity sees the aerosol-inclusive transmissivity
477:      # (same rationale as the shortwave aerosol mix in ``solve_sw``).
479:      tau_aer = jnp.maximum(aerosol_absorption_optical_depth, 0.0)
494:    # (same BUG-B robustness fix as the shortwave path): with no LW aerosol
629:    aerosol_optical_depth: Array | None,
630:    aerosol_single_scattering_albedo: float | Array,
631:    aerosol_asymmetry_factor: float | Array,
637:  (3D Monte-Carlo ray tracer) so the gas+cloud optics lookup, aerosol mixing,
654:  return _mix_sw_aerosol_and_clip(
655:      igpt, optics_lib, sw_optical_props, aerosol_optical_depth,
656:      aerosol_single_scattering_albedo, aerosol_asymmetry_factor)
659:def _mix_sw_aerosol_and_clip(
663:    aerosol_optical_depth: Array | None,
664:    aerosol_single_scattering_albedo: float | Array,
665:    aerosol_asymmetry_factor: float | Array,
667:  """Mix aerosol into one g-point's SW gas(+cloud) optics, then clip ssa/g."""
668:  if aerosol_optical_depth is not None:
669:    # Per-band aerosol optics: ``aerosol_single_scattering_albedo`` /
670:    # ``aerosol_asymmetry_factor`` may be a SCALAR (grey aerosol, historical
672:    # values (real aerosols absorb/scatter very differently in the UV/visible vs
676:    ssa_aer = jnp.asarray(aerosol_single_scattering_albedo)
677:    g_aer = jnp.asarray(aerosol_asymmetry_factor)
685:    tau_aer = jnp.maximum(aerosol_optical_depth, 0.0)
720:  # before downstream use.  The aerosol-mixing branch above already clips
721:  # ssa∈[0,1] / g∈[-1,1], but with no aerosol forcing the cloud optics feed
748:    aerosol_optical_depth: Array | None = None,
749:    aerosol_single_scattering_albedo: float | Array = _AEROSOL_SSA_DEFAULT,
750:    aerosol_asymmetry_factor: float | Array = _AEROSOL_ASYM_DEFAULT,
771:        cloud_fraction, aerosol_optical_depth,
772:        aerosol_single_scattering_albedo, aerosol_asymmetry_factor,
775:    # a fraction of TOTAL scattering (gas + cloud + aerosol) -> the per-cell
803:    aerosol_absorption_optical_depth: Array | None = None,
832:    if aerosol_absorption_optical_depth is not None:
833:      # LW aerosol is a pure absorber (ssa=0): add its absorption OD directly,
834:      # matching the solve_lw aerosol treatment.
835:      abs_od = abs_od + jnp.maximum(aerosol_absorption_optical_depth, 0.0)
859:    aerosol_optical_depth: Array | None = None,
860:    aerosol_single_scattering_albedo: float | Array = _AEROSOL_SSA_DEFAULT,
861:    aerosol_asymmetry_factor: float | Array = _AEROSOL_ASYM_DEFAULT,
898:    aerosol_optical_depth: Optional aerosol optical depth per layer (same
900:    aerosol_single_scattering_albedo: Bulk aerosol single-scattering albedo.
901:    aerosol_asymmetry_factor: Bulk aerosol asymmetry factor.
937:    """One g-point's SW fluxes from its final (aerosol-mixed) optics."""
994:      # Per-g-point gas+cloud+aerosol optics (shared with the MC ray tracer).
998:          cloud_fraction, aerosol_optical_depth,
999:          aerosol_single_scattering_albedo, aerosol_asymmetry_factor,
1005:      # solve_lw); aerosol is mixed into each, as the separate passes did.
1011:      _aer = (aerosol_optical_depth, aerosol_single_scattering_albedo,
1012:              aerosol_asymmetry_factor)
1014:          igpt, _mix_sw_aerosol_and_clip(igpt, optics_lib, cloudy_props, *_aer))
1016:          igpt, _mix_sw_aerosol_and_clip(igpt, optics_lib, gas_props, *_aer))

exec
/usr/bin/bash -lc "sed -n '230,620p' CLAUDE.md" in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 0ms:
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

exec
/usr/bin/bash -lc "rg -n 'volcanic_cmip6_from_cesm_1978-2014|nan|fillna|where' packages/tools/legoesm/forcing/external.py | head -65; sed -n '585,670p' packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py; rg -n 'aerosol_lw_od|aerosol_optical_depth|reshape|chunk' packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py | tail -65; rg --files -g '*volcanic*' -g '*AGENTS.md' | head -40" in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 18ms:
75:# sits at ~16-30 km where US Std Atm 1976 T ~ 217-227 K.  This is a documented
124:    p = np.full_like(z, np.nan)
129:        p = np.where(in_layer, _layer_p(p_base[i], z_b, t_b, lapse, z), p)
461:    (times, data) where times is shape (N,) in days and data maps
469:    # Build a case-insensitive lookup for variable names to handle files where
542:    (mid_days, first_date, lat, data) where ``first_date`` is a
556:        # nanmean as on the whole array, without holding a full-resolution
560:            np.nanmean(np.asarray(var.isel({dims[0]: i}).values, dtype=np.float64), axis=ax)
583:            data = np.nanmean(data, axis=ax)
598:        data = np.nanmean(data, axis=ax) if collapsed is None else collapsed
642:            slab = (np.nanmean(slab, axis=sdims.index(d))
648:    if np.isnan(data).any():
650:        bad = np.argwhere(np.isnan(data))
679:       averages non-(time, lat) axes via ``np.nanmean``) and avoids the
789:        aod_with_bands = (np.nanmean(aod_with_bands, axis=band_axis)
931:        abs_ext = (np.nanmean(abs_ext, axis=dims.index(band_name))
1164:            data = np.nanmean(data, axis=lon_ax)
1202:    read MPI-M files where the stored names are ``"TSI"`` / ``"SSI_frac"``.
1548:        dx = np.where(x1 > x0, x1 - x0, 1.0)
1549:        w = np.where(x1 > x0, (lat_deg - x0) / dx, 0.0)
1609:        dx = np.where(x1 > x0, x1 - x0, 1.0)
1610:        w = np.where(x1 > x0, (log_p_tgt_flat - x0) / dx, 0.0)
1634:    where lat=1, lon=1 and time is in fractional years.
1689:    "nmol/mol": 1.0e-9, "nmol mol-1": 1.0e-9, "nanomol/mol": 1.0e-9,
1701:    dimensions (time, lat, lon) where lat=1, lon=1.  The time axis
1707:    (years, data) where years is shape (N,) and data maps variable
1754:            ref = float(np.nanmax(np.abs(arr)))
2682:    its 14 fractions in **RRTMG-SW** band order, where the long-wave
  def solve_columns(
      self,
      T: jnp.ndarray,
      p_full: jnp.ndarray,
      p_half: jnp.ndarray,
      sfc_temperature: jnp.ndarray,
      q_v: jnp.ndarray,
      cos_zenith: jnp.ndarray,
      sfc_albedo: jnp.ndarray | float | None = None,
      sfc_albedo_direct: jnp.ndarray | float | None = None,
      sfc_emissivity: jnp.ndarray | float | None = None,
      o3_vmr: jnp.ndarray | None = None,
      cloud_path_liq: jnp.ndarray | None = None,
      cloud_path_ice: jnp.ndarray | None = None,
      cloud_path_liq_lw: jnp.ndarray | None = None,
      cloud_path_ice_lw: jnp.ndarray | None = None,
      cloud_r_eff_liq: jnp.ndarray | None = None,
      cloud_r_eff_ice: jnp.ndarray | None = None,
      cloud_fraction: jnp.ndarray | None = None,
      aerosol_optical_depth: jnp.ndarray | None = None,
      aerosol_absorption_optical_depth_lw: jnp.ndarray | None = None,
      solar_spectral_fraction: jnp.ndarray | None = None,
      ghg_vmr_override: dict | None = None,
      sw_optical_field_only: bool = False,
      lw_optical_field_only: bool = False,
      *,
      mcica_cloud_fraction: jnp.ndarray | None = None,
      mcica_mask_width: float | None = None,
      clear_sky: bool = False,
      o3_top_vmr: jnp.ndarray | None = None,
      cloud_rho_ice: float | None = None,
      cloud_path_snow: jnp.ndarray | None = None,
      cloud_d_eff_snow: jnp.ndarray | None = None,
      snow_optics=None,
      cloud_snow_weight: jnp.ndarray | None = None,
  ):
      """Compute radiation for legoESM column arrays.

      This is the canonical entry point for the RRTMGP solver when used
      from legoESM's physics integration layer.  It handles:

      1. Reshaping (ncol, nlev) arrays to jax-rrtmgp's (ncol, 1, nlev+2)
      2. Adding 1-cell vertical halos
      3. Building VMR dict from config concentrations
      4. Computing atmospheric state and solving two-stream
      5. Stripping halos and reshaping back

      Parameters
      ----------
      T : jnp.ndarray
          Temperature at full levels (ncol, nlev) [K].
      p_full : jnp.ndarray
          Pressure at full levels (ncol, nlev) [Pa].
      p_half : jnp.ndarray
          Pressure at interface levels (ncol, nlev+1) [Pa].
      sfc_temperature : jnp.ndarray
          Surface temperature (ncol,) [K].
      q_v : jnp.ndarray
          Water vapor specific humidity (ncol, nlev) [kg/kg].
      cos_zenith : jnp.ndarray
          Cosine of solar zenith angle (ncol,).
      sfc_albedo : jnp.ndarray | float | None
          Per-column surface albedo override.
      sfc_emissivity : jnp.ndarray | float | None
          Per-column surface emissivity override.
      o3_vmr : jnp.ndarray | None
          External ozone VMR (ncol, nlev), index 0 = TOA.
      cloud_path_liq : jnp.ndarray | None
          Liquid water path per layer (ncol, nlev) [kg/m^2].
      cloud_path_ice : jnp.ndarray | None
          Ice water path per layer (ncol, nlev) [kg/m^2].
      cloud_path_liq_lw : jnp.ndarray | None
          Separate liquid water path for the LONGWAVE solve (ncol, nlev)
          [kg/m^2].  ``None`` (default) reuses ``cloud_path_liq`` for both
          streams (byte-identical legacy behaviour).  Populated by the
          ``two_column`` partial-coverage optics, whose coverage inversion
          is stream-specific (reflectance vs emissivity space).
      cloud_path_ice_lw : jnp.ndarray | None
          Separate ice water path for the LONGWAVE solve; see
          ``cloud_path_liq_lw``.
      cloud_r_eff_liq : jnp.ndarray | None
          Liquid cloud effective radius (ncol, nlev) [m].
      cloud_r_eff_ice : jnp.ndarray | None
          Ice cloud effective radius (ncol, nlev) [m].
      aerosol_optical_depth : jnp.ndarray | None
          Prescribed shortwave aerosol optical depth per layer
54:        "aerosol_optical_depth": "1",
174:        [T[:, :1], t_int, jnp.reshape(sfc_temperature, (-1, 1))], axis=1)
604:      aerosol_optical_depth: jnp.ndarray | None = None,
669:      aerosol_optical_depth : jnp.ndarray | None
754:      if aerosol_optical_depth is not None:
755:          aerosol_optical_depth = jnp.asarray(
756:              aerosol_optical_depth).astype(_table_dtype)
859:                            _table_dtype).reshape(ncol, 1))
868:          aerosol_optical_depth = _zero(aerosol_optical_depth)
966:              return val.reshape(ncol, 1)
1107:      if aerosol_optical_depth is not None:
1109:              _add_halos(aerosol_optical_depth[:, None, ::-1]), 0.0, None,
1157:              aerosol_optical_depth=aerosol_od_3d,
1257:          aerosol_optical_depth=aerosol_od_3d,
1277:      # --- 7. Strip halos, flip back to legoESM convention, reshape ---
1312:  def solve_columns_chunked(
1315:      column_chunk_size: int,
1324:      """Column-chunked wrapper over :meth:`solve_columns`.
1326:      Splits the leading column axis into ``ncol // column_chunk_size``
1331:      unchunked call, not an approximation.  The point is COMPILE TIME: XLA
1332:      lowers the per-block body ONCE at ``column_chunk_size`` columns, so the
1337:      LEVELS are NOT chunked: the vertical is coupled through the two-stream
1345:      column_chunk_size : int
1347:          disables chunking (a plain :meth:`solve_columns` call).  Must
1354:          axis equals ``ncol`` is treated as a per-column array and chunked
1367:      # chunking-specific check so a disabled call is a TRUE passthrough:
1369:      # solve_columns returns (no reshape overhead, no contract change).
1370:      if (not column_chunk_size or column_chunk_size <= 0
1371:              or ncol <= column_chunk_size):
1377:      # --- chunking is ACTIVE from here down ---
1379:      # layout ((n_gpt, ncol, nlev)) carries no leading (n_block, chunk)
1380:      # column axis, so the reassembly reshape below would corrupt it.
1384:              "solve_columns_chunked supports only the RadiationOutput path; "
1386:              "per-g-point tuple incompatible with column chunking — call "
1389:      if ncol % column_chunk_size != 0:
1391:              f"column_chunk_size {column_chunk_size} must divide ncol "
1393:              "lax.map reshape requires equal-size blocks). Note that ncol "
1398:      n_block = ncol // column_chunk_size
1404:      # are chunked here, so a per-column ``(ncol, ...)`` config fallback would
1405:      # be read at global ncol inside each chunk-local solve → size mismatch
1407:      # pass the field explicitly (which IS chunked); the production AMIP path
1417:                  f"solve_columns_chunked: config fallback '{_sfc_name}' is a "
1420:                  "global ncol inside each chunk block (via "
1421:                  "_resolve_surface_field) → size mismatch under chunking. "
1423:                  "chunked) instead of relying on the per-column config field."
1427:      # (n_block, column_chunk_size, ...).  This row-major reshape is
1428:      # order-preserving and the reassembly reshape below is its exact
1435:          return x.reshape(n_block, column_chunk_size, *x.shape[1:])
1458:      # the chunking buys compile time but no memory.  With it the backward
1460:      # ``column_chunk_size`` columns instead of ``ncol`` -- the AMIP WB arm
1472:      def _one(chunk):
1473:          return self.solve_columns(**chunk, **static)
1475:      out = jax.lax.map(_one, mapped)  # leaves: (n_block, chunk, ...)
1476:      # Merge (n_block, chunk) back to ncol — the exact inverse of _split.
1477:      return jax.tree.map(lambda a: a.reshape(ncol, *a.shape[2:]), out)
tests/unit/test_volcanic_lw_aerosol.py
AGENTS.md
data/forcing_amip/volcanic_amip_1979-1980.nc
docs/wb/reviews/2026-11-22_volcanic_optics_vote_prompt.md
docs/wb/reviews/2026-11-22_volcanic_optics_vote_codex.md

codex
The profile loader does not currently replace NaNs. The remapper also clips negative inputs, so it cannot be reused unchanged for signed τ·SSA·g moments.
exec
/usr/bin/bash -lc "sed -n '620,1020p' CLAUDE.md" in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 0ms:
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

exec
/usr/bin/bash -lc "rg -n 'def _open_forcing_dataset|volcanic_cmip6_from_cesm_1978-2014|delta_scal|delta_edd' packages scripts configs tests | head -65; sed -n '150,200p' packages/tools/legoesm/forcing/external.py; sed -n '2655,2720p' packages/tools/legoesm/forcing/external.py; sed -n '930,980p' packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rte/two_stream.py; sed -n '1,100p' tests/unit/test_volcanic_lw_aerosol.py" in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 2ms:
rg: configs: No such file or directory (os error 2)
tests/ice/unit/test_ice_shortwave_faithful.py:5:``delta_eddington`` (Briegleb & Light 2007 two-band VIS/NIR surrogate with
tests/ice/unit/test_ice_shortwave_faithful.py:50:    delta_eddington_albedo,
tests/ice/unit/test_ice_shortwave_faithful.py:98:@pytest.mark.parametrize("scheme", ["constant", "maykut_untersteiner", "delta_eddington"])
tests/ice/unit/test_ice_shortwave_faithful.py:127:    # The delta_eddington branch has its OWN min() clamp (line 400); with physical
tests/ice/unit/test_ice_shortwave_faithful.py:135:        _sw_mod, "delta_eddington_albedo",
tests/ice/unit/test_ice_shortwave_faithful.py:138:    out = _sw(400.0, 260.0, 1.0, 0.0, 0.0, 0.0, "delta_eddington")
tests/ice/unit/test_ice_shortwave_faithful.py:175:        alpha, _ = delta_eddington_albedo(
tests/ice/unit/test_ice_shortwave_faithful.py:194:    alpha, _ = delta_eddington_albedo(
tests/ice/unit/test_ice_shortwave_faithful.py:213:    alpha, trans = delta_eddington_albedo(_arr(T), _arr(1.0), _arr(0.10),
tests/ice/unit/test_ice_shortwave_faithful.py:223:    alpha, _ = delta_eddington_albedo(_arr(T), _arr(1.0), _arr(0.0),
tests/ice/unit/test_ice_shortwave_faithful.py:234:    alpha, _ = delta_eddington_albedo(_arr(T), _arr(1.0), _arr(0.0),
tests/ice/unit/test_ice_shortwave_faithful.py:244:    alpha, _ = delta_eddington_albedo(_arr(T_FREEZE), _arr(1.0), _arr(0.10),
tests/ice/unit/test_ice_shortwave_faithful.py:252:    alpha, _ = delta_eddington_albedo(_arr(T_FREEZE), _arr(1.0), _arr(0.0),
tests/ice/unit/test_ice_shortwave_faithful.py:314:# 5. Beer-Lambert interior penetration (delta_eddington).
tests/ice/unit/test_ice_shortwave_faithful.py:320:    _, t1 = delta_eddington_albedo(_arr(T), _arr(0.5), _arr(0.0), _arr(0.0), _arr(0.0))
tests/ice/unit/test_ice_shortwave_faithful.py:321:    _, t2 = delta_eddington_albedo(_arr(T), _arr(1.0), _arr(0.0), _arr(0.0), _arr(0.0))
tests/ice/unit/test_ice_shortwave_faithful.py:330:    _, t = delta_eddington_albedo(_arr(T), _arr(h), _arr(0.0), _arr(0.0), _arr(0.0))
tests/ice/unit/test_ice_shortwave_faithful.py:338:    ts = [float(delta_eddington_albedo(_arr(T), _arr(h), _arr(0.0), _arr(0.0),
tests/ice/unit/test_ice_shortwave_faithful.py:350:    shallow = float(delta_eddington_albedo(_arr(T), _arr(1.0), _arr(0.0),
tests/ice/unit/test_ice_shortwave_faithful.py:352:    deep = float(delta_eddington_albedo(_arr(T), _arr(1.0), _arr(0.0),
tests/ice/unit/test_ice_shortwave_faithful.py:363:    a_shallow = float(delta_eddington_albedo(_arr(T), _arr(2e-4), _arr(0.0),
tests/ice/unit/test_ice_shortwave_faithful.py:365:    a_deep = float(delta_eddington_albedo(_arr(T), _arr(2e-4), _arr(0.0),
tests/ice/unit/test_ice_shortwave_faithful.py:379:    a_thin_ice = float(delta_eddington_albedo(_arr(T), _arr(0.1), _arr(0.10),
tests/ice/unit/test_ice_shortwave_faithful.py:381:    a_thick_ice = float(delta_eddington_albedo(_arr(T), _arr(3.0), _arr(0.10),
tests/ice/unit/test_ice_shortwave_faithful.py:397:    a_sat = float(delta_eddington_albedo(_arr(T), _arr(0.5), _arr(0.0),
tests/ice/unit/test_ice_shortwave_faithful.py:399:    a_thick = float(delta_eddington_albedo(_arr(T), _arr(2.0), _arr(0.0),
tests/ice/unit/test_ice_shortwave_faithful.py:401:    a_quarter = float(delta_eddington_albedo(_arr(T), _arr(0.5 / 4), _arr(0.0),
tests/ice/unit/test_ice_shortwave_faithful.py:413:    a_sat = float(delta_eddington_albedo(_arr(T), _arr(1.0), _arr(0.05),
tests/ice/unit/test_ice_shortwave_faithful.py:415:    a_deep = float(delta_eddington_albedo(_arr(T), _arr(1.0), _arr(0.5),
tests/ice/unit/test_ice_shortwave_faithful.py:417:    a_half = float(delta_eddington_albedo(_arr(T), _arr(1.0), _arr(0.025),
tests/ice/unit/test_ice_shortwave_faithful.py:432:        alpha, _ = delta_eddington_albedo(_arr(T_FREEZE - 3.0), _arr(1.0),
tests/ice/unit/test_ice_shortwave_faithful.py:444:    a_sat = float(delta_eddington_albedo(_arr(T), _arr(1.0), _arr(0.0),
tests/ice/unit/test_ice_shortwave_faithful.py:446:    a_deeper = float(delta_eddington_albedo(_arr(T), _arr(1.0), _arr(0.0),
tests/ice/unit/test_ice_shortwave_faithful.py:448:    a_half = float(delta_eddington_albedo(_arr(T), _arr(1.0), _arr(0.0),
tests/ice/unit/test_ice_shortwave_faithful.py:489:        alpha, trans = delta_eddington_albedo(_arr(260.0), h, _arr(0.0),
tests/ice/unit/test_omip_ice_optics.py:16:from legoesm.ice.shortwave import compute_ice_sw, delta_eddington_albedo
tests/ice/unit/test_omip_ice_optics.py:24:SCHEMES = ("constant", "maykut_untersteiner", "delta_eddington")
tests/ice/unit/test_omip_ice_optics.py:73:                              "--ice-shortwave", "delta_eddington"])
tests/ice/unit/test_omip_ice_optics.py:86:    alpha, transmission = delta_eddington_albedo(z + temperature, z + thickness, hs, z, z)
tests/ice/unit/test_omip_ice_optics.py:160:    if scheme == "delta_eddington":
tests/ice/unit/test_omip_ice_optics.py:201:    cfg = _main_config(["--prognostic-sea-ice", "--ice-shortwave", "delta_eddington",
tests/ice/unit/test_omip_ice_optics.py:222:        return jnp.stack(delta_eddington_albedo(SKIN_T, x[0], x[1], 0., 0.))
tests/sea_ice/validation/test_sea_ice_truth_tiers.py:364:    a_bare = float(SW.delta_eddington_albedo(
tests/sea_ice/validation/test_sea_ice_truth_tiers.py:367:    a_snow = float(SW.delta_eddington_albedo(
tests/sea_ice/validation/test_sea_ice_truth_tiers.py:370:    a_pond = float(SW.delta_eddington_albedo(
tests/sea_ice/validation/test_sea_ice_truth_tiers.py:379:@pytest.mark.parametrize("scheme", ["constant", "maykut_untersteiner", "delta_eddington"])
tests/sea_ice/validation/test_sea_ice_truth_tiers.py:407:@pytest.mark.parametrize("scheme", ["maykut_untersteiner", "delta_eddington"])
tests/ice/unit/test_sw_transmittance_const.py:72:    the budget cannot go negative (mirrors the delta_eddington bound)."""
tests/ice/unit/test_shortwave_pond_albedo.py:9:from legoesm.ice.shortwave import _band_albedo_pond, delta_eddington_albedo
tests/ice/unit/test_shortwave_pond_albedo.py:35:def test_delta_eddington_deeper_pond_lowers_tile_albedo():
tests/ice/unit/test_shortwave_pond_albedo.py:39:    shallow, _ = delta_eddington_albedo(*args, jnp.asarray(0.02))
tests/ice/unit/test_shortwave_pond_albedo.py:40:    deep, _ = delta_eddington_albedo(*args, jnp.asarray(0.4))
packages/atmosphere/legoesm/atmosphere/physics/radiation/ecrad_ifs/cloud_optics.py:7:false``, ``do_lw_cloud_scattering = true``, ``do_sw_delta_scaling_with_gases =
packages/atmosphere/legoesm/atmosphere/physics/radiation/ecrad_ifs/cloud_optics.py:78:        "radiation_delta_eddington.h, radiation_cloud.F90 crop_cloud_fraction; "
packages/atmosphere/legoesm/atmosphere/physics/radiation/ecrad_ifs/cloud_optics.py:170:def _delta_eddington_scat_od(od, scat_od, asym):
packages/atmosphere/legoesm/atmosphere/physics/radiation/ecrad_ifs/cloud_optics.py:171:    """radiation_delta_eddington.h: delta_eddington_scat_od."""
packages/atmosphere/legoesm/atmosphere/physics/radiation/ecrad_ifs/cloud_optics.py:273:        liq, _delta_eddington_scat_od(*_liq_socrates(co["liq.coeff_sw"], lwp_s, re_liq_s, sharpness)))
packages/atmosphere/legoesm/atmosphere/physics/radiation/ecrad_ifs/cloud_optics.py:275:        ice, _delta_eddington_scat_od(*_ice_fu_sw(co["ice.coeff_sw"], iwp_s, re_ice_s, sharpness)))
packages/atmosphere/legoesm/atmosphere/physics/radiation/ecrad_ifs/cloud_optics.py:277:        ice, _delta_eddington_scat_od(*_ice_fu_lw(co["ice.coeff_lw"], iwp_s, re_ice_s, sharpness)))
packages/atmosphere/legoesm/atmosphere/physics/radiation/ecrad_ifs/tripleclouds_solvers.py:14:* ``radiation_tripleclouds_sw.F90`` for do_sw_delta_scaling_with_gases = false,
packages/atmosphere/legoesm/atmosphere/physics/radiation/ecrad_ifs/aerosol_optics.py:10:``do_sw_delta_scaling_with_gases = false`` (delta-Eddington on the aerosol
packages/atmosphere/legoesm/atmosphere/physics/radiation/ecrad_ifs/aerosol_optics.py:11:alone, ``radiation_delta_eddington.h:delta_eddington_extensive_vec``).  No
packages/atmosphere/legoesm/atmosphere/physics/radiation/ecrad_ifs/aerosol_optics.py:94:        "radiation_delta_eddington.h; Bozzo et al. (2020) GMD 13."
packages/atmosphere/legoesm/atmosphere/physics/radiation/ecrad_ifs/aerosol_optics.py:107:# max(scat_od, 1.0e-24) in delta_eddington_extensive_vec: a DEFAULT-REAL
packages/atmosphere/legoesm/atmosphere/physics/radiation/ecrad_ifs/aerosol_optics.py:108:# (single-precision) literal in the oracle (radiation_delta_eddington.h), promoted.
    t = float(temp_k)
    w1 = np.asarray(wl1_um, dtype=np.float64) * 1.0e-6  # -> m
    w2 = np.asarray(wl2_um, dtype=np.float64) * 1.0e-6
    nu_lo = 1.0 / np.maximum(w1, w2)  # smaller wavenumber [1/m]
    nu_hi = 1.0 / np.minimum(w1, w2)  # larger wavenumber [1/m]
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    weights = np.empty(nu_lo.shape[0], dtype=np.float64)
    for b in range(nu_lo.shape[0]):
        nu = np.linspace(nu_lo[b], nu_hi[b], _PLANCK_QUAD_NSUB)
        x = h * c * nu / (k_b * t)
        b_nu = 2.0 * h * c**2 * nu**3 / np.expm1(x)
        weights[b] = _trapz(b_nu, nu)
    return weights


def _collapse_sw_bands(ds, data, dims, band_dim, band="gray"):
    """Collapse SW band axis ``band_dim`` of ``data`` using the file's OWN band
    edges ``wl1_sun``/``wl2_sun`` [um] on that dim: ``band="gray"`` -> solar-
    weighted (5778 K Planck) mean, ``band="vis"`` -> the band containing 550 nm.
    Returns None when the file carries no edges for ``band_dim`` (caller keeps
    its flat band mean)."""
    if "wl1_sun" not in ds or ds["wl1_sun"].dims != (band_dim,):
        return None
    wl1 = np.asarray(ds["wl1_sun"].values, dtype=np.float64)
    wl2 = np.asarray(ds["wl2_sun"].values, dtype=np.float64)
    ax = dims.index(band_dim)
    if band == "vis":
        lo, hi = np.minimum(wl1, wl2), np.maximum(wl1, wl2)
        i_vis = int(np.flatnonzero((lo <= _VISIBLE_WL_UM) & (hi > _VISIBLE_WL_UM))[0])
        return np.take(data, i_vis, axis=ax)
    w = _planck_band_weights(wl1, wl2, _SOLAR_PLANCK_TEMP_K)
    return np.tensordot(w, np.moveaxis(data, ax, 0), axes=(0, 0)) / w.sum()


# ==============================================================================
# Data loading helpers (Zarr-first, NetCDF fallback via xarray)
# ==============================================================================

def _open_forcing_dataset(path: str):
    """Open a forcing file as xarray Dataset (Zarr or NetCDF).

    Always opens with ``decode_times=False`` so that time coordinates are
    returned as raw numeric values (float/int) that can be safely cast to
    ``np.float64``.  Files whose dates extend beyond numpy's datetime64[ns]
    range (~2262) would otherwise cause xarray to fall back to
    ``cftime.datetime`` objects, which cannot be cast to float64.
    """
    import os
    import xarray as xr
    if os.path.isdir(path) or path.endswith(".zarr"):
        return xr.open_zarr(path, decode_times=False)
        lo0, n_in = lo - 1, int(hi - lo + 1)
        if solar_src is not None:
            w = solar_src[lo0:hi]
            s = w.sum()
            w = w / s if s > 0 else np.full(n_in, 1.0 / n_in)
        else:
            w = np.full(n_in, 1.0 / n_in)
        band_weights.append((lo0, hi, w))

    if spec_bands.ndim == 1:
        out = np.zeros(n_gpt, dtype=np.float64)
        for i, (lo0, hi, w) in enumerate(band_weights):
            out[lo0:hi] = spec_bands[i] * w
        return out
    # Trailing-axis case: (..., n_bands) → (..., n_gpt).
    leading_shape = spec_bands.shape[:-1]
    out = np.zeros(leading_shape + (n_gpt,), dtype=np.float64)
    for i, (lo0, hi, w) in enumerate(band_weights):
        out[..., lo0:hi] = spec_bands[..., i:i + 1] * w
    return out


def _cmip_sw_band_order_to_rrtmgp(spec_bands: np.ndarray) -> np.ndarray:
    """Re-order per-band SW fractions from CMIP6 / RRTMG-SW band order
    to the RRTMGP-SW gas-optics-table band order.

    The MPI-M CMIP6 spectral-solar file (``swflux_14band_cmip6``) stores
    its 14 fractions in **RRTMG-SW** band order, where the long-wave
    overlap band (820-2680 cm^-1) is the LAST band.  The RRTMGP-SW
    gas-optics table (``rrtmgp-gas-sw-g112.nc``, ``bnd_limits_wavenumber``)
    lists the SAME 14 wavenumber intervals but with that band FIRST, so
    the two orderings differ by exactly a one-band cyclic rotation
    (issue #322):

        RRTMGP band ``i``  ==  CMIP band ``(i - 1) mod 14``

    hence ``np.roll(spec, +1, axis=-1)`` converts CMIP order -> RRTMGP
    order.  Without it, ``_expand_bands_to_gpoints`` (which maps by array
    index) assigns each CMIP band's flux to the wrong RRTMGP band,
    shifting the whole spectrum one band toward longer wavelengths — e.g.
    UV flux lands in the near-IR water-vapour band, roughly doubling
    clear-sky shortwave absorption.

    This is applied ONLY on the ``spectral_file`` ingest path (per-band
    14-element MPI-M input).  ``_expand_bands_to_gpoints`` itself stays a
    pure index-preserving expansion so its band-integral contract is
    unchanged.

    Parameters
    ----------
    spec_bands : np.ndarray, shape ``(..., 14)``
        Per-band fractions in CMIP / RRTMG-SW order (trailing axis).

    Returns
    -------
    np.ndarray
        Same shape, re-ordered to RRTMGP-SW band order.
    """
    spec_bands = np.asarray(spec_bands, dtype=np.float64)
    return np.roll(spec_bands, 1, axis=-1)


def get_tsi_at_time(config: SolarConfig, day: float) -> float:
    """Return total solar irradiance at a given simulation day.

    Parameters
  is_day_col = zenith < 0.5 * jnp.pi  # shape () or (ncol, 1)

  # Check whether *any* column is illuminated to short-circuit a global
  # nighttime domain (preserves the original optimisation).
  any_day = jnp.any(is_day_col)

  def _sw_gpoint_fluxes(igpt, sw_optical_props):
    """One g-point's SW fluxes from its final (aerosol-mixed) optics."""
    optical_props_2stream = monochromatic_two_stream.sw_cell_properties(
        safe_zenith,
        sw_optical_props['optical_depth'],
        sw_optical_props['ssa'],
        sw_optical_props['asymmetry_factor'],
    )

    # Surface albedo: broadcast per-column array or scalar to 2D plane
    # with the same horizontal sharding as the temperature. The DIFFUSE
    # reflection uses sfc_alb; the DIRECT beam uses sfc_alb_dir when set
    # (RAD-3 SAM direct/diffuse split), else falls back to sfc_alb.
    sfc_albedo = atmos_state.sfc_alb * jnp.ones_like(temperature[:, :, 0])
    _alb_dir_src = (
        atmos_state.sfc_alb if atmos_state.sfc_alb_dir is None
        else atmos_state.sfc_alb_dir
    )
    sfc_albedo_dir = _alb_dir_src * jnp.ones_like(temperature[:, :, 0])

    # Monochromatic top of atmosphere flux.
    if solar_fraction_by_gpt is None:
      spectral_weight = optics_lib.solar_fraction_by_gpt[igpt]
    else:
      spectral_weight = solar_fraction_by_gpt[igpt]
    solar_flux = atmos_state.irrad * spectral_weight
    toa_flux = solar_flux * jnp.ones_like(temperature[:, :, 0])

    sources_2stream = monochromatic_two_stream.sw_cell_source(
        t_dir=optical_props_2stream['t_dir'],
        r_dir=optical_props_2stream['r_dir'],
        optical_depth=sw_optical_props['optical_depth'],
        toa_flux=toa_flux,
        sfc_albedo_direct=sfc_albedo_dir,
        zenith=safe_zenith,
        use_scan=use_scan,
        mu_incident=jnp.cos(zenith),
    )

    sw_fluxes = monochromatic_two_stream.sw_transport(
        t_diff=optical_props_2stream['t_diff'],
        r_diff=optical_props_2stream['r_diff'],
        src_up=sources_2stream['src_up'],
        src_down=sources_2stream['src_down'],
        sfc_src=sources_2stream['sfc_src'],
"""Volcanic stratospheric LONGWAVE aerosol (#9) + the B1/B2/B3 correctness fix.

The optional per-layer ``aerosol_lw_od`` is OFF by default (byte-identical
no-op) and, when supplied, materialises as a JIT-stable (ncol, nlev) field.

Fix coverage (Codex-reviewed):
- ``TestUSStdAtmPressure`` / ``TestPlanckBandWeights`` -- the z->p (geopotential-
  corrected) and Planck-weighted gray-collapse helpers.
- ``TestLoaderAbsorptionScaling`` -- B1: loader feeds ext*(1-omega), not raw
  extinction (synthetic-violation guard).
- ``TestStratosphericPlacement`` -- B2: conservative log-pressure overlap remap
  puts >80% of OD above 100 hPa vs <15% for the old pressure-mass helper.
- ``TestPlacementDifferentiability`` -- grad matches FD, jit/vmap match eager.
- ``TestExternalForcingGate`` -- the ``_ext_forcing`` gate includes LW aerosol.
- ``TestLwAerosolOlrSign`` -- solve_columns: a stratospheric LW absorption bump
  lowers OLR (correct warming sign) by a small amount.
- ``TestRealFileEndToEnd`` -- the real 1979 file lands in the stratosphere.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


class TestVolcanicLwDefaultOff(unittest.TestCase):
    """The loader is OFF by default => no LW aerosol => no-op."""

    def test_get_aerosol_lw_disabled_returns_none(self):
        from legoesm.forcing.external import AerosolConfig, get_aerosol_lw_at_time
        cfg = AerosolConfig()  # volcanic_lw_enabled defaults False
        self.assertFalse(cfg.volcanic_lw_enabled)
        out = get_aerosol_lw_at_time(cfg, day=0.0, lat_grid=np.zeros(4))
        self.assertIsNone(out)

    def test_get_aerosol_lw_enabled_without_path_returns_none(self):
        from legoesm.forcing.external import AerosolConfig, get_aerosol_lw_at_time
        cfg = AerosolConfig(volcanic_lw_enabled=True)  # no volcanic_path
        out = get_aerosol_lw_at_time(cfg, day=0.0, lat_grid=np.zeros(4))
        self.assertIsNone(out)


class TestPackForcingLwField(unittest.TestCase):
    """pack_forcing materialises aerosol_lw_od as a JIT-stable concrete array."""

    def _pack(self, aerosol_lw_od):
        from legoesm.driver.compiled_segments import pack_forcing
        shp = (6, 4, 4)
        aer = jnp.zeros((6 * 4 * 4, 5))  # (ncol, nlev) per-layer SW aerosol od
        return pack_forcing(
            sst=jnp.full(shp, 290.0), sic=jnp.zeros(shp),
            day_of_year=1, seconds_of_day=0.0,
            solar_weights=jnp.ones(1), s_0=1361.0,
            o3_vmr=jnp.zeros((6 * 4 * 4, 5)), aerosol_od=aer,
            aerosol_lw_od=aerosol_lw_od,
        )

    def test_none_materialises_zeros_same_shape(self):
        f = self._pack(None)
        self.assertEqual(f.aerosol_lw_od.shape, f.aerosol_od.shape)
        self.assertTrue(jnp.all(f.aerosol_lw_od == 0.0))

    def test_supplied_value_preserved(self):
        lw = jnp.full((6 * 4 * 4, 5), 0.1)
        f = self._pack(lw)
        self.assertEqual(f.aerosol_lw_od.shape, (6 * 4 * 4, 5))
        self.assertTrue(jnp.allclose(f.aerosol_lw_od, 0.1))


# ---------------------------------------------------------------------------
# Fix B1/B2/B3: extinction->absorption, stratospheric placement, band collapse
# ---------------------------------------------------------------------------

_REAL_VOLC_LW = Path(
    "/pool/data/ICON/grids/public/mpim/common/aerosol_volcanic_cmip6/"
    "bc_aeropt_cmip6_volc_lw_b16_sw_b14_1979.nc"
)


def _make_synth_volc_lw(path, omega=0.3, ext_val=1.0e-3, nband=16,
                        nlat=4, nalt=6, nmonth=12, alt_lo=15.0, alt_hi=25.0):
    """Write a minimal CMIP6-shaped volcanic LW file for loader tests."""
    import xarray as xr
    alt = np.linspace(alt_lo, alt_hi, nalt)          # km, stratospheric
    lat = np.linspace(-60.0, 60.0, nlat)
    month = np.arange(1, nmonth + 1)
    ev = np.asarray(ext_val, dtype=np.float64)
    if ev.ndim == 0:                                 # scalar -> band-uniform
        ext = np.full((nband, nlat, nalt, nmonth), float(ev))
    else:                                            # per-band profile
        ext = np.broadcast_to(

exec
/usr/bin/bash -lc "rg -n 'volcanic_cmip6_from_cesm_1978-2014' configs scripts packages docs/wb --glob '"'!reviews/*'"' | head -20; sed -n '197,212p' packages/tools/legoesm/forcing/external.py; sed -n '1020,1075p' CLAUDE.md; sed -n '850,875p' packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py; rg -n 'band_lims_wvn|bnd_limits_wavenumber' packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/optics/gas_optics.py" in /glade/derecho/scratch/pg2328/wt_volc
 exited 1 in 199ms:
rg: configs: No such file or directory (os error 2)
docs/wb/reviews/2026-11-22_volcanic_optics_vote_prompt.md:6:The ETH CMIP6 file (volcanic_cmip6_from_cesm_1978-2014.nc: ext_sun/omega_sun/g_sun
docs/wb/reviews/2026-11-22_volcanic_optics_vote_codex.md:19:The ETH CMIP6 file (volcanic_cmip6_from_cesm_1978-2014.nc: ext_sun/omega_sun/g_sun
docs/wb/reviews/2026-11-22_volcanic_optics_vote_codex.md:3063:/usr/bin/bash -lc "rg -n 'volcanic_cmip6_from_cesm_1978-2014|nan|fillna|where' packages/tools/legoesm/forcing/external.py | head -65; sed -n '585,670p' packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py; rg -n 'aerosol_lw_od|aerosol_optical_depth|reshape|chunk' packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py | tail -65; rg --files -g '*volcanic*' -g '*AGENTS.md' | head -40" in /glade/derecho/scratch/pg2328/wt_volc
docs/wb/reviews/2026-11-22_volcanic_optics_vote_codex.md:3648:/usr/bin/bash -lc "rg -n 'def _open_forcing_dataset|volcanic_cmip6_from_cesm_1978-2014|delta_scal|delta_edd' packages scripts configs tests | head -65; sed -n '150,200p' packages/tools/legoesm/forcing/external.py; sed -n '2655,2720p' packages/tools/legoesm/forcing/external.py; sed -n '930,980p' packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rte/two_stream.py; sed -n '1,100p' tests/unit/test_volcanic_lw_aerosol.py" in /glade/derecho/scratch/pg2328/wt_volc
scripts/cluster/derecho_amip/amip_mpas_gpu_chain.pbs:56:export VOLCANIC="${AMIP_INPUTS}/volcanic_cmip6_from_cesm_1978-2014.nc"
scripts/cluster/scaling_derecho/amip_trace_census.sh:35:export OZONE=$AI/ozone_cam6_plev_1978-2014.nc VOLCANIC=$AI/volcanic_cmip6_from_cesm_1978-2014.nc
    import os
    import xarray as xr
    if os.path.isdir(path) or path.endswith(".zarr"):
        return xr.open_zarr(path, decode_times=False)
    return xr.open_dataset(path, decode_times=False)


def _to_days_float(time_values, *, units: str | None = None,
                   calendar: str | None = None) -> np.ndarray:
    """Convert xarray time values to float days since the first record.

    Handles four cases that arise in practice:

    * ``numpy.datetime64`` arrays — standard CF time within the
      ``datetime64[ns]`` representable range (~year 100–2262).
    * ``cftime.datetime`` object arrays — xarray falls back to cftime
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
          if cloud_r_eff_liq is not None:
              cloud_r_eff_liq = _top(cloud_r_eff_liq, cloud_r_eff_liq[:, :1])
          if cloud_r_eff_ice is not None:
              cloud_r_eff_ice = _top(cloud_r_eff_ice, cloud_r_eff_ice[:, :1])
          nlev = nlev + 1

codex
Q1: C - Use A’s sorted 1:1 mapping, but validate the expected edge pairs, explicitly allowing documented RRTMG/RRTMGP boundary differences. Centre containment alone can accept incorrect intervals; overlap remapping assumes unresolved spectral structure.

Q2: B - Raise: extinction alone cannot supply the approved volcanic optics; a warning still permits scientifically different forcing.

Q3: A - Cache float32; promote to the radiation computation dtype before interpolation/remapping/mixing. Float64 storage cannot recover source precision.

Q4: A - Use None when absent for the entire run; keep an array through zero-volcanic months so the pytree structure stays fixed.

Q5: A - Raise during setup; silently ignoring the leaf would discard volcanic SW forcing after its removal from aerosol_od.

Q6: C - Zero documented no-aerosol fill regions only; reject unexplained missing extinction or missing SSA/g at positive extinction. The current profile loader does **not** replace NaNs.

Q7: C - Keep 1e-10 for float64 algebraic equivalence using float64 synthetic moments. For closure use `|residual| ≤ 1e-10 × incident_flux + 1e-10 W/m²`; relative tolerance against zero is ineffective. Test float32 cache quantization separately.

DEFECTS:

- The existing remapper clips both inputs and outputs nonnegative, corrupting signed τ·SSA·g when g<0. Reuse its overlap weights through a signed linear operation. [Source](/glade/derecho/scratch/pg2328/wt_volc/packages/tools/legoesm/forcing/surface_utils.py:550)
- The existing sampler catches **every ValueError**, so new malformed-file/band-validation errors could silently become “no profile.” Separate absent-schema detection from validation failures. [Source](/glade/derecho/scratch/pg2328/wt_volc/packages/tools/legoesm/forcing/external.py:2358)
- Volcanic mixing must execute when tropospheric aerosol is None, and enter both cloudy and clear-sky branches, optical-field export, column chunking, vertical reversal, halos, and synthetic top-layer handling.
- Preserve safe division for transparent/non-scattering layers; validate `0≤τSSA≤τ` and `|τSSA g|≤τSSA`. Test JIT and gradients, including zero OD.
- Conservative remapping preserves represented moments; subsequent tropopause weighting removes some. Test these operations separately. Validate energy closure against integrated heating, with a planted missing-heating violation.

UNVERIFIED: Actual file metadata/fill masks and runtime numerical tolerances; this was source inspection only.
tokens used
61,359
Q1: C - Use A’s sorted 1:1 mapping, but validate the expected edge pairs, explicitly allowing documented RRTMG/RRTMGP boundary differences. Centre containment alone can accept incorrect intervals; overlap remapping assumes unresolved spectral structure.

Q2: B - Raise: extinction alone cannot supply the approved volcanic optics; a warning still permits scientifically different forcing.

Q3: A - Cache float32; promote to the radiation computation dtype before interpolation/remapping/mixing. Float64 storage cannot recover source precision.

Q4: A - Use None when absent for the entire run; keep an array through zero-volcanic months so the pytree structure stays fixed.

Q5: A - Raise during setup; silently ignoring the leaf would discard volcanic SW forcing after its removal from aerosol_od.

Q6: C - Zero documented no-aerosol fill regions only; reject unexplained missing extinction or missing SSA/g at positive extinction. The current profile loader does **not** replace NaNs.

Q7: C - Keep 1e-10 for float64 algebraic equivalence using float64 synthetic moments. For closure use `|residual| ≤ 1e-10 × incident_flux + 1e-10 W/m²`; relative tolerance against zero is ineffective. Test float32 cache quantization separately.

DEFECTS:

- The existing remapper clips both inputs and outputs nonnegative, corrupting signed τ·SSA·g when g<0. Reuse its overlap weights through a signed linear operation. [Source](/glade/derecho/scratch/pg2328/wt_volc/packages/tools/legoesm/forcing/surface_utils.py:550)
- The existing sampler catches **every ValueError**, so new malformed-file/band-validation errors could silently become “no profile.” Separate absent-schema detection from validation failures. [Source](/glade/derecho/scratch/pg2328/wt_volc/packages/tools/legoesm/forcing/external.py:2358)
- Volcanic mixing must execute when tropospheric aerosol is None, and enter both cloudy and clear-sky branches, optical-field export, column chunking, vertical reversal, halos, and synthetic top-layer handling.
- Preserve safe division for transparent/non-scattering layers; validate `0≤τSSA≤τ` and `|τSSA g|≤τSSA`. Test JIT and gradients, including zero OD.
- Conservative remapping preserves represented moments; subsequent tropopause weighting removes some. Test these operations separately. Validate energy closure against integrated heating, with a planted missing-heating violation.

UNVERIFIED: Actual file metadata/fill masks and runtime numerical tolerances; this was source inspection only.
