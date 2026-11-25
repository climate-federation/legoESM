Reading additional input from stdin...
OpenAI Codex v0.166.0
--------
workdir: /glade/u/home/pg2328/legoesm
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: danger-full-access
reasoning effort: medium
reasoning summaries: none
session id: 01a204e6-34f3-7250-b0bf-006c42e85cbd
--------
user
Adversarially review the claims in docs/wb/reviews/2026-11-22_forcing_audit_prompt.md. Read-only: do not edit any file. Verify each claim in the code cited and reply AGREE/REFUTE/UNSURE per claim with file:line evidence, then list missed channels. Under 500 words.
codex
I’ll read the review prompt and repository rules, then check each claim against the cited code and look for missed channels. I won’t edit any files.
exec
/usr/bin/bash -lc 'pwd; git status --porcelain; git worktree list; cat CLAUDE.md; cat docs/wb/reviews/2026-11-22_forcing_audit_prompt.md' in /glade/u/home/pg2328/legoesm
 succeeded in 345ms:
/glade/u/home/pg2328/legoesm
?? .claude/.headroom_wrap_settings.lock
?? c1.pbs
?? data/les_cases/AYOTTE/
?? data/les_cases/GABLS3/
?? data/les_cases/GABLS4/
?? data/les_cases/PROVENANCE_2026-10-15.md
?? docs/datasets.json
?? docs/wb/reviews/2026-11-22_cold_bias_claims_codex.md
?? docs/wb/reviews/2026-11-22_cold_bias_claims_glm.md
?? docs/wb/reviews/2026-11-22_cold_bias_claims_prompt.md
?? docs/wb/reviews/2026-11-22_cold_bias_vote_codex.md
?? docs/wb/reviews/2026-11-22_cold_bias_vote_glm.md
?? docs/wb/reviews/2026-11-22_cold_bias_vote_prompt.md
?? docs/wb/reviews/2026-11-22_forcing_audit_codex.md
?? docs/wb/reviews/2026-11-22_forcing_audit_prompt.md
?? scripts/cluster/les_suite/derecho_les_case.pbs
?? scripts/cluster/les_suite/les_timing.o7827219
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
/glade/derecho/scratch/pg2328/wt_spec_main                                                                                                          5b56a56a79 (detached HEAD)
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
/glade/u/home/pg2328/legoesm/.claude/worktrees/crm-fp32-tracers                                                                                     be5b35cdfa [perf/crm-fp32-tracers]
/glade/u/home/pg2328/legoesm/.claude/worktrees/crm-les-scaling                                                                                      43d9f3a14d [perf/crm-les-mpi-scaling]
/glade/u/home/pg2328/legoesm/.claude/worktrees/crm-mpi-gaps                                                                                         f3068db97e [fix/crm-mpi-gaps]
/glade/u/home/pg2328/legoesm/.claude/worktrees/crm-rad-blocking                                                                                     8669277fd4 [perf/crm-rad-blocking]
/glade/u/home/pg2328/legoesm/.claude/worktrees/final2                                                                                               be5b35cdfa (detached HEAD)
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
7. Stop when clean or after 30 iter# Forcing audit: legoESM production AMIP vs CAM6 (CESM2.1 cam_cesm2_1_rel_60 hist_cam6) — claims for adversarial review

Context: production deck config/amip/amip_production.yaml + paths config/amip/amip_production.sh (Levante).
1979 bias: tas -1.1 K, cold NH high-latitude land/Antarctica/Tibet, TOA albedo +0.10-0.15 over Siberia/Canada/Arctic.
CAM6 reference: CAM namelist use_case hist_cam6.xml (/glade/derecho/scratch/pg2328/cesm_src/CAM/bld/namelist_files/use_cases/hist_cam6.xml)
and CESM inputdata on /glade/campaign/cesm/cesmdata/inputdata. Levante input files are NOT readable from Derecho.

Please verify or refute EACH claim from the code (packages/tools/legoesm/forcing/external.py, surface_utils.py,
packages/coupler/legoesm/driver/model_driver.py::_precompute_external_forcing, radiation/rrtmgp/rrtmgp.py,
radiation/config.py, forcing/amip.py::get_forcing_at_time). Answer per claim: AGREE / REFUTE (with file:line) / UNSURE.
Also name any forcing channel I MISSED.

C1 GHG: ours = annual global-mean CO2/CH4/N2O/CFC-11/CFC-12 from ICON greenhouse_historical_plus.nc, linear interp in
   fractional year (start_year + day/365), transient. File lacks CFC-22/CCl4/CF4 so RRTMGP holds them at FIXED present-day
   (cfc22 240, ccl4 75, cf4 85 pptv; rrtmgp.py:504-508). CAM6 = LBC_1750-2015_CMIP6_GlobAnnAvg (lat-resolved, SERIAL):
   1979 CO2 336.6 ppm, CH4 1566 ppb, N2O 300.4 ppb, CFC11eq 358 pptv, CFC12 287 pptv (measured, unweighted lat mean).
   Verdict: EQUIVALENT within ~0.05 W/m2 (halogen bookkeeping differs: CAM CFC11eq lumps all minor halogens;
   ours physical CFC-11 + fixed present-day minor halogens ~ +0.02 W/m2 vs CAM in 1979). Not cold-bias relevant.
   PLAUSIBLE (Levante file values not read).
C2 Solar: ours = ICON swflux_14band_cmip6_1850-2299-v3.2 (SOLARIS-HEPPA 3.2) monthly TSI + 14-band SSI_frac,
   RRTMG->RRTMGP one-band rotation; CAM6 = SolarForcingCMIP6 (same SOLARIS-HEPPA 3.2). Orbit: ours present-day Berger
   elements (orbit_year null); CAM6 fixed-year orbit. EQUIVALENT (<0.05 W/m2).
C3 Ozone: ours = UReading-CCMI-1-0 vmro3 (CMIP6 input4MIPs) monthly 1979-2014, zonal-mean, non-cyclic, pressure-interpolated,
   plus 16 overhead levels for the above-top layer; CAM6 = ozone_strataero_WACCM_L70_zm5day CMIP6ensAvg (WACCM6-derived,
   5-day zonal mean). DIFFERENT product, same family; O(0.1) W/m2 and stratospheric T differences. Not cold-bias primary.
C4 Volcanic: ours = ETH/IACETH CMIP6 v4 ext_sun/ext_earth profiles, transient, SW placed as a pressure profile * strat
   weight, LW absorption on (--volcanic-aerosol-lw). BUT SW uses the GLOBAL gray aerosol SSA 0.93 / g 0.70
   (radiation/config.py:258), not the file's omega_sun/g_sun (~0.99 for sulfate): volcanic aerosol is spuriously absorbing.
   CAM6 = WACCM prescribed_strataero (MAM-mode sulfate, own optics). DIFFERENT. 1979 quiet (AOD~0.005): negligible;
   El Chichon/Pinatubo years: too little SW cooling (~5-7% of the volcanic SW forcing absorbed), too much strat heating.
C5 Tropospheric aerosol: ours = MACv2-SP plumes on MACv1 background (1850-2014 transient, lat-lon), solar-weighted gray
   column AOD distributed by layer pressure thickness (uniform mixing ratio, surface to top; surface_utils.py:489-491),
   single gray SSA 0.93 / g 0.70 for all bands, no LW; CCN via Andreae (2009) fit on total AOT (aerosol_ccn: true).
   CAM6 = MAM4 interactive (CMIP6 emissions), per-band optics, PBL-concentrated profile, prognostic activation (ARG),
   dust/sea-salt, ice nucleation on dust. DIFFERENT (by design; licence decision). Direct effect: ours more absorbing
   and spread aloft -> if anything a WARMING bias vs CAM; indirect: tuned (rsut bias +0.78 W/m2). Cold-bias relevance
   PLAUSIBLE only via CDNC over high latitudes.
C6 SST/SIC: ours = PCMDI-AMIP-1-1-10 tosbcs/siconcbcs (mid-month BC values), linear interp in time, SST floored at
   T_freeze_ocean, SIC clipped [0,1]; CAM6 = Hurrell-merged HadOIBl bcs file, linear interp. EQUIVALENT (version drift only).
C7 Land use: ours = surfdata_1.9x2.5_16pfts_CMIP6_simyr2000 fixed; CAM6 FHIST = CLM5 transient land use 1850-2015
   (78 PFT incl. crops). DIFFERENT; small global forcing (<0.1 W/m2), not cold-bias primary.
C8 MISSING: aerosol (BC/dust) deposition on snow (CLM5 SNICAR fed by MAM4 deposition): our snow albedo has no impurity
   term (grep snicar/black carbon/impurity in packages/land: no hits). Snow is therefore brighter than CAM6/CLM5 snow,
   most in spring over Eurasia/Tibet; literature: global +0.05 W/m2, regional spring snow forcing O(1-5) W/m2.
   CAM6/CLM5 also use SNICAR grain-size ageing vs our BATS ageing (land-model, not forcing). PLAUSIBLE contributor to the
   NH land TOA-albedo / cold bias.
C9 Nitrogen deposition: N/A (our land has no N cycle).
Overall: forcing is NOT the main cold-bias driver; GHG/solar/SST equivalent; aerosol/volcanic optics different; the
only forcing-side candidate that plausibly feeds the NH snow cold bias is the missing aerosol-on-snow darkening.

exec
/usr/bin/bash -lc "sed -n '1,220p' CLAUDE.md" in /glade/u/home/pg2328/legoesm
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

exec
/usr/bin/bash -lc "rg --files -g AGENTS.md -g '*external.py' -g '*surface_utils.py' -g '*rrtmgp.py' -g '*amip.py'; cat config/amip/amip_production.yaml; cat config/amip/amip_production.sh" in /glade/u/home/pg2328/legoesm
 succeeded in 0ms:
AGENTS.md
tests/distributed/test_mpas_mpi_amip.py
tests/distributed/test_latlon_mpi_amip.py
tests/tools/test_surface_utils.py
tests/unit/test_plot_amip.py
tests/unit/test_orbital_integration_rrtmgp.py
tests/unit/test_monitor_century_amip.py
tests/validation/test_precision_amip.py
tests/unit/test_surface_utils.py
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py
packages/tools/legoesm/forcing/external.py
packages/tools/legoesm/forcing/surface_utils.py
packages/tools/legoesm/forcing/amip.py
packages/ocean/legoesm/ocean/physics/surface_forcing/external.py
scripts/run/run_amip.py
scripts/validate/monitor_century_amip.py
scripts/plot/plot_amip.py
# =============================================================================
# legoESM AMIP — AUTHORITATIVE production configuration (single source of truth)
# =============================================================================
# Run with:
#   run_amip.py --config config/amip/amip_production.yaml \
#       <machine path flags> --days N --output DIR
#
# Keys are run_amip argument DESTS (set as defaults via --config); any explicit
# CLI flag still overrides a key here.  Machine-specific PATHS (ERA5 IC,
# SST/SIC, solar/ozone/GHG/aerosol/volcanic, topography) are deliberately NOT
# pinned here — they live in config/amip/amip_production.sh (env-overridable)
# so this physics config stays machine-independent and reproducible across
# deployments.
#
# THE ATMOSPHERIC PHYSICS SUITE IS CAM6 (user decision 2026-09-23).  The lane
# (MPAS res6 Voronoi, prescribed SST, two-leaf multilayer land, external
# forcings) is unchanged from the previous production deck; the atmospheric
# parameterizations, the vertical grid and the physics cadence are CAM6
# (CESM2.1, cam_cesm2_1_rel_60, f09 namelist) wherever a faithful port exists.
# The outgoing configuration is kept RUNNABLE and unchanged at
# config/amip/amip_sundqvist_l36.yaml, so every earlier result reproduces; the
# decks that used to `include: amip_production.yaml` now include that file,
# because they document results measured against it.
#
# THIS DEFAULT IS ADOPTED ON FAITHFULNESS TO CAM6, NOT ON DEMONSTRATED SKILL.
# The 60-day CAM6 arm is NOT finished and has NOT yet been scored against
# observations.  Over its first twenty-five days it sits about 25 W/m2 out of
# top-of-atmosphere energy balance, against about 2 W/m2 for the Sundqvist/L36
# arm, because its clouds reflect far too little sunlight.  The cause is known
# and is NOT yet fixed: cloud water is being rimed onto falling ice.  The
# supporting measurement, on the two arms' day-5 states: 71% of the liquid
# deficit lies below freezing, 70% of that cold deficit is accounted for by
# the snow + graupel increase, and graupel more than doubles column-wide.
#
# EVIDENCE AND PAIR RECORD live in the RUN TREE, which is deliberately NOT in
# this repository (runtime outputs are gitignored):
#   /work/bd1083/b309178/diffESM/legoesm_pg/amip_runs/_cam6/PREREGISTRATION.md
# on the Levante deployment.  It records the pair (cam6_base60 = the
# Sundqvist/L36 deck, cam6_suite60 = this one; 60-day cold starts, scored
# days 31-60 vs ERA5), the resolved-config diff, and every measurement cited
# above.  A reader without that tree should treat the numbers here as
# unverifiable rather than as repository facts.
#
# NOT YET CAM6 (run 2+): MG2 microphysics (Morrison mg flavour stands in),
# ridge-based orographic drag (needs the ridge-statistics dataset), MAM4
# aerosols (licence decision), CLUBB tuning table.
#
# EVERY scientific choice is spelled out below even where it equals a code
# default: a default is not a record.  In particular the two numerical levers
# (mpas_div_damp4_scale, mpas_vert_advection_scheme: sb) keep their OFF code
# defaults and are selected HERE.
# ---- grid / dycore ----------------------------------------------------------
grid_type: voronoi
resolution: 6               # 40962 cells (~1.1 deg) ~ CAM6 f09 (0.9x1.25)
nlev: 32                    # CAM6: the L32 hybrid table (cam_vcoords_L32_c180105)
vertical_coord: cam_l32     # CAM6: hyai/hybi verbatim, p_top 2.255 hPa
discretization: mpas
vertical_form: as83          # Arakawa-Suarez; production MPAS switch approved 2026-11-11.
                             # 30-day A/B: no detectable global-mean difference;
                             # energy fixer masks the energy-gain comparison. Evidence:
                             # /glade/derecho/scratch/pg2328/ab_vertical_form/README.md
dt: 112.5
precision: fp64
corner_fill: avg                     # cube-vertex halo fill; inert on MPAS,
                                     # pinned so the run records it
a_h_scale: 0.375
mpas_sponge_del2_top_layers: 2       # index-based sponge: 2 layers on L32 span
                                     # 2.3-10.2 hPa (the Sundqvist/L36 deck: 3 layers on
                                     # sigma-36 = 2.0-7.6 hPa).  CAM6 f09 FV has
                                     # NO top sponge (div24del2flag=4 = uniform
                                     # del4 divergence damping; the tanh top
                                     # profile lives only in the ldel2/ldiv2
                                     # branches) -- 2 layers is the closest
                                     # offered setting (user 2026-09-21)
mpas_sponge_del2_top_factor: 8.0
tracer_vertical_scheme: van_leer
tracer_van_leer_width: 1.0e-10  # VOTED W4: mixing-ratio increments [kg/kg]
tracer_van_leer_flux_width: 1.0e-6  # VOTED W4: mass flux [Pa/s]
mpas_vert_advection_scheme: sb       # conservative Simmons-Burridge flux form
                                     # of the hybrid theta transport (van Leer
                                     # is sigma-only on this lane).  The
                                     # advective upwind form violates the
                                     # discrete product rule: measured column
                                     # residual 3.8e-2 of the interfacial
                                     # exchange on this L32 table against
                                     # 6.7e-16 conservative.  USER-APPROVED
                                     # 2026-09-23; the CODE default stays
                                     # 'upwind' and this deck selects it.
mpas_vert4_flux_form_tracers: []     # vapour takes the log form below instead
mpas_vert4_log_form_tracers: [q_v]   # vertical del4 filter on ln(q_v + 1e-7), water-weighted
                                     # mass-conserving flux.  The flux form [q_v] removed 8-10 %/day
                                     # of UTLS water offline; one 60-day pair vs it (2026-10-14,
                                     # PLAUSIBLE): Feb ClimateEval median RMSE ratio 0.97, 200 hPa
                                     # T RMSE -26..-32 %.  USER-APPROVED 2026-10-15; the CODE
                                     # default moved to this too (same day).
mpas_div_damp4_scale: 1.0            # CAM6: fv_div24del2flag=4 selects ldiv4,
                                     # 4th-order DIVERGENCE damping at every
                                     # level (cd_core.F90 tau4 = 0.01/dt times
                                     # the squared cell area).  1.0 = CAM's
                                     # nondimensional RATE, not CAM's behaviour
                                     # (CAM applies it per acoustic substep,
                                     # per cell); 7.235e15 m^4/s here.  RATE
                                     # PENDING USER APPROVAL.  Our del2/del4
                                     # are VECTOR Laplacians and damp balanced
                                     # flow too; this is the curl-free-only
                                     # term CAM relies on, and the L32 table's
                                     # 14 pure-pressure top layers have no way
                                     # to absorb divergence into their own mass.
                                     # MEASURED, physics off, from the day-5
                                     # state: alone it cuts the 24 hPa
                                     # hot-patch drift from 3.14 to 1.96 K/h,
                                     # a 38% reduction -- real but NOT
                                     # sufficient.  WITH the conservative
                                     # transport above the drift REVERSES to
                                     # -1.10 K/h, which is why this deck
                                     # carries both or neither.
energy_fixer: true                   # CAM6 check_energy_fix analogue (review
                                     # 2026-10-10 F15, by vote): the energy each
                                     # dynamics step loses (sponge/del4/div4 KE,
                                     # numerics) returns as a uniform heating
                                     # before physics, and the energy the dry-
                                     # mass fixer moves (outside that window)
                                     # returns too (2026-10-14 F15 V8: 0.065
                                     # W/m2 at res6).  Named, not inherited
                                     # (the code default is also true).

# ---- physics cadence ---------------------------------------------------------
physics_update_steps: 16    # CAM6: physics every 1800 s (16 x 112.5 s); the
                            # held-physics steps re-apply the last tendencies
rad_update_steps: 32        # CAM6 f09: radiation hourly (iradsw = -1)
cld_macmic_num_steps: 3     # CAM6 namelist_defaults_cam.xml:1254 (mg2 + clubb_sgs,
                            # dtime 1800): CLUBB + microphysics sub-cycled 3 x
                            # at 600 s inside each physics step (tphysbc macmic)

# ---- radiation / clouds / microphysics --------------------------------------
radiation: rrtmg            # rrtmgp builder (CAM6 uses RRTMG; same k-lineage)
rrtmgp_gas_tables: full     # reference RRTMGP tables (LW g256 / SW g224): RFMIP
                            # forcing error <=0.04 W/m2 vs up to 0.24 with the
                            # reduced g128/g112 set (user approval 2026-11-12)
rrtmgp_overhead_layer: true # CAM RRTMG extra layer above the model top (ozone
                            # column above p_top absorbs; ~2 W/m2 leaves budget)
# Saturation curve the cloud-cover diagnosis measures RH against.  NAMED, not
# inherited: the code default is "mixed_phase" and that is the value this deck
# wants, but the default moved on 2026-09-17 and a default is not a record.
# The preserved pre-CAM6 deck pins "liquid" for the same reason.
cloud_saturation_scheme: mixed_phase
clouds: cam6_clubb          # CAM6: liquid = CLUBB PDF fraction, ice = CAM aist,
                            # deep-convective = clubb_intr deepcu from the
                            # scheme's updraft mass flux + in-cloud condensate
use_clubb_cloud_fraction: true   # routes the CLUBB cloud-fraction carry to
                                 # radiation (required by cam6_clubb)
cloud_cam6_rhmini: 0.80     # CAM6 aist ramp (cldfrc2m.F90:846-856), namelist
cloud_cam6_rhmaxi: 1.0      # cldfrc2m_rhmini/rhmaxi/rhminis/rhmaxis under
cloud_cam6_rhminis: 1.0     # clubb_sgs (namelist_defaults_cam.xml:1521-1533);
cloud_cam6_rhmaxis: 1.0     # = the code defaults, now recorded (user 2026-09-30)
microphysics: morrison      # mg flavour (MG2 port = run 2)
morrison_flavor: mg         # NAMED, equals the code default.  Resolves the
                            # MG rows (resolve_morrison_flavor), incl. since
                            # 2026-10-19 (oracle audit, user decision) CAM6
                            # MG2 rain: slope bounds 1/[500, 20] um (was
                            # 1/[2800, 20] um) and Beheng self-collection
                            # -8*nric*qric*rho on in-precip rain (was SAM
                            # SB2001 with breakup).
morrison_do_graupel: true   # NAMED, equals the code default: this deck's
                            # measured behaviour.  CAM6's MG2 has NO graupel
                            # (eight constituents, micro_mg_cam.F90:139); our
                            # port rimes cloud water onto graupel, ~half the
                            # cold-sector cloud-water sink on the day-5 state.
                            # The graupel-off arm is this deck plus
                            # --no-morrison-do-graupel; this value stays
                            # until the user decides (2026-09-24).
morrison_warm_rain_scheme: kk2000_cam6  # CAM6 MG2 law (user decision
                                   # 2026-10-19, oracle audit; was kk2000, the
                                   # SAM/E3SM 1350*qc^2.47*Nc^-1.79 form, 0.003-
                                   # 13x MG2): 13.5*var_coef(relvar,2.47)*
                                   # qc^2.47*Nc^-1.1 (micro_mg_utils.F90), with
                                   # per-cell relvar = rcm^2/min(1, rcp2) from
                                   # CLUBB (clubb_intr.F90:2416-2425).  Named
                                   # explicitly; the code default (unset)
                                   # resolves to the same law under mg.
morrison_autocon_fact: 1.0         # kk2000 autoconversion multiplier (1 = unscaled)
morrison_accre_enhan_fact: 1.0     # kk2000 accretion multiplier (MG2 accre_enhan)
morrison_warm_rain_incloud: true   # CAM6 MG2 in-cloud warm rain (user decision
                                   # 2026-09-30): rates on qc/lcldm, qr/lcldm,
                                   # lcldm = max(ast, 1e-4), ast = max(CLUBB
                                   # liquid, aist ice-stratus) as CAM6
                                   # (clubb_intr.F90:2575, micro_mg_cam.F90:
                                   # 1810), times lcldm (micro_mg2_0.F90:878,
                                   # :1224-1322).  Grid-mean arm:
                                   # --no-morrison-warm-rain-incloud.
                                   # Since 2026-10-19 (oracle audit, user
                                   # decision) it is the whole CAM6 MG2 rain
                                   # mode: in_cloud precip_frac, rain
                                   # evaporation only in the precipitating
                                   # clear air, rain sedimenting the post-
                                   # source state, two-sided number re-adjust.
                                   # Since 2026-10-19 also CAM6 MG2 TWO-MOMENT
                                   # SNOW (user decision): the run carries the
                                   # snow number N_s (tracer slot [9]); MG2
                                   # riming, Bergeron onto snow, sublimation,
                                   # self-aggregation, melt above +2 C, fall
                                   # speeds, snow sedimenting the post-source
                                   # state.
morrison_ice_snow_d_auto: 500.0e-6 # CAM6 MG2 micro_mg_dcs = 500 um ice->snow
                                   # size (namelist_defaults_cam.xml:1244;
                                   # user decision 2026-10-19, was 250 um).
morrison_sed_cfl_substeps: true  # MG2 micro_mg2_0.F90 sedimentation loop:
                                 # nstep = 1 + int(max V dt/dz) per column,
                                 # so precipitation crosses >1 layer per 600 s.
                                 # Rain and snow sediment the POST-source state
                                 # as MG2 (:2204, via warm_rain_incloud); ice/
                                 # graupel still sediment PRE-source pools.
morrison_sed_cfl_substeps_strict: false # the abort path is an equinox error_if,
                                        # i.e. a host callback, and the GPU lane
                                        # runs JAX_PLATFORMS=cuda with no CPU
                                        # device to place it on (this killed the
                                        # first 60-day arm at day 2).  The
                                        # required sub-step count is still
                                        # reported as a diagnostic.
                                 # (L32 at 600 s needs ~90 for graupel; a clamp
                                 # would silently slow the fall)
hard_saturation_adjustment: false  # CAM6 clubb_do_liqsupersat = .false.
                                   # (namelist_defaults_cam.xml:1225): no
                                   # supersaturation removal after the physics
                                   # step.  On this lane the flag is the
                                   # POST-STEP drain (model_driver
                                   # _mpas_hard_saturation_poststep), which is
                                   # our counterpart of CAM's post-CLUBB
                                   # "iceadj" block (clubb_intr.F90:2258).
                                   # Was true; user table 2026-09-24, row 2.
cloud_partial_coverage_optics: none
cloud_optics_inhomogeneity: constant  # CAM6: RRTMG applies NO in-cloud
                                       # inhomogeneity factor (none anywhere
                                       # in src/physics/rrtmg/).  "constant"
                                       # with factor 1.0 is the no-thinning
                                       # path (chi = 1, byte-identical).  Was
                                       # two_region: on day-15 states it
                                       # removed 52% of the liquid in run 1
                                       # and 42% in run 3, so the water
                                       # reaching radiation stayed ~14.5 g/m2
                                       # across the liquid hand-off.  User
                                       # decision 2026-09-25, for run 4.
cloud_inhomogeneity_factor: 1.0        # the "constant" scheme's scalar: 1.0
                                       # = homogeneous, no reduction.  Set
                                       # explicitly so the no-thinning row
                                       # rests on no default.
cloud_fsd: 1.0                         # INERT under "constant" (consumed by
                                       # two_region only); kept explicit so
                                       # the old value is on record.
cloud_vertical_overlap_optics: mcica  # CAM6 McICA (user 2026-09-25): each
                                     # g-point sees its own max-random cloud
                                     # subcolumn in ONE solve (was max_random:
                                     # 8 full solves, ~8.5x the radiation cost).
                                     # Synthetic 512-column check vs a 64-subcolumn
                                     # reference: bias within sampling noise,
                                     # per-column scatter ~1.8x the 8-subcolumn one.
cloud_n_subcolumns: 8                # read only by max_random; inert under mcica
rrtmgp_gpoint_batch_size: 16         # GPU: 16 g-points per vmap block (production
                                     # runs on GPU; user decision 2026-11-18).
                                     # Measured on A100, MPAS res6, 1 simulated day, 0 vs
                                     # 16 only: 0.161 -> 0.132 s/step on 1 GPU, 0.112 ->
                                     # 0.092 on 4 (2026-11-13, before the full gas
                                     # tables); 16/32/64/128 within 3% (2026-11-16).
                                     # COSTS CPU runs: 2026-10-06, res6 16 ranks x 8
                                     # cores, radiation step 13.8 s at 16 vs 7.9 s at 0
                                     # (~139 s per simulated day) -- a CPU run passes
                                     # --rrtmgp-gpoint-batch-size 0.  Fluxes differ
                                     # only by summation order (<= 6.1e-4 W/m2).
rrtmgp_use_scan: true                # scan the RRTMGP column recurrence (code default:
                                     # unrolled on CPU, scan on GPU, so GPU is unchanged).
                                     # Measured 2026-10-29, this deck, MPAS res6 CPU: step
                                     # -4% at 1 node, -7% at 8; round-off only (fp32 fluxes
                                     # <= 1.1e-7 of max after one step), dry mass unchanged,
                                     # 127-step divergence (0.66 K) matched by a 1e-14 T
                                     # perturbation control (0.27 K).  User 2026-10-30.
hard_sat_ice_curve: false          # the ice-curve blend of that same
                                   # post-step drain; inert once the drain
                                   # is off, set explicitly so no row rests
                                   # on a default.  User table row 3.
homogeneous_ice_nucleation: true
aerosol_ccn: true                  # droplet number diagnosed from the prescribed
                                   # aerosol (user decision 2026-11-01, #2216,
                                   # reverses 2026-10-24): Jan 1979 30-day arm,
                                   # global rsut bias vs CERES-EBAF -7.99 W/m2
                                   # (prognostic) -> +0.78 (this setting).
morrison_prognostic_droplet_number: false  # CAM6 MG2 prognostic droplet number
                                   # OFF until its droplet budget closes (#2216);
                                   # the prognostic arm is
                                   # amip_production_prognostic_nc.yaml.
                                   # cam6_aerosol_dir below is read only by it.
morrison_mg2_drop_early_reset_log_width: 0.01  # NAMED, equals the code default:
                                   # RULE 5 width of MG2's early droplet size
                                   # reset (autoconversion + snow riming on
                                   # this deck's specified droplet number; 0 =
                                   # MG2's hard clip; user 2026-11-01, #2216)
morrison_mg2_drop_sed_reset_log_width: 0.01  # NAMED, equals the code default:
                                   # RULE 5 width of MG2's droplet size reset in
                                   # the sedimentation fall speeds (read only by
                                   # the prognostic droplet number, so inert on
                                   # this deck; 0 = MG2's hard clip; #2216)
cam6_aerosol_dir: /work/bb1596/b309178/diffESM/legoesm_ap/data/cam6_optics
energy_consistent_moisture_clip: true

# ---- convection ---------------------------------------------------------------
zm_land_fraction: required  # ZM reads the land mask (c0_lnd vs c0_ocn); "none" = aquaplanet only
convection: zhang_mcfarlane # CAM6: faithful zm_conv port (f09 mg2+clubb namelist:
                            # c0_lnd 0.0075, c0_ocn 0.03, ke 5e-6 (dyn=fv),
                            # tau 3600,
                            # capelmt 70, tiedke_add 0.5, momcu = momcd = 0.7,
                            # num_cin 1); scheme defaults carry those values.
                            # ZM convective momentum transport (momtran) is ON
                            # and reaches the MPAS winds (mpas_cmt resolves True;
                            # the Sundqvist/L36 deck's Bechtold runs CMT
                            # off, lane default)
convective_rain_to_surface: true
convective_cloud: false     # refused by cam6_clubb (deepcu is CAM6's version)
# Bechtold/IFS settings written explicitly (inert here: convection is ZM).
bechtold_use_ifs_ascent: false         # faithful OpenIFS chain off
bechtold_ifs_shallow_nosupply: zero    # no sub-cloud supply -> shallow flux ramps to ~0 (OpenIFS: first_guess)
# ZM tunables written explicitly (values = the scheme defaults = the CAM6 f09
# namelist above), so a campaign arm's change is one visible row.
zm_c0_lnd: 0.0075           # rain conversion over land [1/m]
zm_c0_ocn: 0.03             # rain conversion over ocean [1/m]
zm_ke: 5.0e-6               # rain evaporation efficiency
zm_dmpdz: -1.0e-3           # dilute-plume entrainment [1/m]
zm_tau: 3600.0              # CAPE consumption timescale [s]
zm_capelmt: 70.0            # deep-convection CAPE threshold [J/kg]
# ZM detrainment smoothing widths (RULE 5; values = the scheme defaults; 0 =
# CAM's hard switch), read by the CLUBB macmic sub-step's detrainment booking.
zm_detrain_phase_width_K: 0.1              # liquid/ice ramp corners [K]
zm_detrain_number_width_kgkg_s: 1.0e-12    # max(0, dlf) in the number source [kg/kg/s]
# ZM trigger width (RULE 5; value = the scheme default; 0 = CAM's hard switch):
# the closure mass flux ramps from 0 at zm_capelmt to CAM's at zm_capelmt + width.
zm_trigger_width_J_kg: 10.0
zm_transport_cloud_tracers: true  # ZM convtran of carried cloud/precip water + numbers (CAM6)

# ---- turbulence ---------------------------------------------------------------
turbulence: clubb
clubb_prognostic: true
clubb_liquid_partition: true       # CAM6 clubb_intr: the closure's diagnosed
                                   # cloud liquid is handed to the host as a
                                   # tendency, and the Morrison saturation
                                   # adjustment is switched off, both
                                   # condensation and (since 2026-10-19, user
                                   # decision) cloud-water evaporation (MG2
                                   # allow_sed_supersat = .false. under CLUBB,
                                   # micro_mg_cam.F90:668-672).  One flag,
                                   # both halves.  User table row 1.
# The partition REFUSES an active radiative condensate floor (it would be
# imposed on top of the liquid the closure now supplies, making clouds opaque
# twice).  Both floors default to a non-zero scheme value when unset, so they
# are pinned to 0 here.  Not in the user's three-row table: REQUIRED BY ROW 1.
cloud_q_c_diagnostic: 0.0
conv_cloud_condensate: 0.0
# CLUBB upper domain limit = CAM6's trop_cloud_top_press 1.D2 (100 Pa;
# cam_cesm2_1_rel_60 namelist_defaults_cam.xml:1295).  User decision
# 2026-10-20 ("use cam6 rule"), was 15000 Pa.  Offline 1/3-day A/B on a
# day-105 production checkpoint: stable, sparse stratospheric CLUBB cloud
# (scripts/validate/oracle/clubb_glue/top_press_scm_ab.py).
clubb_trop_cloud_top_press: 100.0
# CLUBB tunables written explicitly (values = the port's CLUBBParams defaults,
# the CAM6 clubb namelist).  clubb_c_K10h (0.3) is read by the eddy-scalar
# rtm/thlm diffusion (CAM l_do_expldiff_rtm_thlm) but has no deck key, so the
# CLUBBParams default applies.
clubb_c14: 2.2
clubb_c8: 4.2
clubb_c11: 0.7
clubb_c11b: 0.35
clubb_gamma_coef: 0.308
clubb_gamma_coefb: 0.32
clubb_beta: 2.4
clubb_c_k10: 0.5

# ---- gravity-wave drag ---------------------------------------------------------
# CAM6 f09 L32 (non-WACCM, nlev < 60) runs OROGRAPHIC drag ONLY: build-namelist
# sets use_gw_front=false and use_gw_convect_dp=false (bld/build-namelist ~3230;
# the f09 frontal/Beres coefficient rows apply to WACCM / >=60-level builds).
# So no non-orographic member here (the Sundqvist/L36 deck carries the
# e3sm_cam
# background source; that row is dropped on purpose).
gravity_wave_drag: mcfarlane            # orographic member = CAM6 gw_oro port
                                        # (cam_cesm2_1_rel_60, oracle-matched;
                                        # user decision 2026-10-19): effgw_oro
                                        # 0.125 x landfrac, sgh/sqrt(landfrac),
                                        # interface N with n2min 5e-5, no launch
                                        # clip, tndmax 400 m/s/day, limiter
                                        # remainder carried down, gw eddy
                                        # diffusion of dse + tracers, critical-
                                        # level offset 0 (CAM c = 0; follow-up
                                        # 2026-10-20, was 0.5 m/s).  These are
                                        # the McFarlaneConfig code defaults (no
                                        # deck rows exist for them, except
                                        # effgw_oro below); the fv3duo deck
                                        # inherits this row.  Since #2546
                                        # (2026-11-16) the code defaults also
                                        # smooth the umcfac |u|, its sign and
                                        # the taumin zeroing (widths 0.01 m/s,
                                        # 1e-3 m/s, 1e-10 Pa; RULE 5).
mcfarlane_effgw_oro: 0.125              # CAM6 effgw_oro (namelist_defaults_cam
                                        # non-WACCM L32); explicit so the run
                                        # records it (#1712).
hines_launch_p: 70000.0                 # inert (hines not selected); kept equal
                                        # to the Sundqvist/L36 deck.  The e3sm_cam_*
                                        # rows
                                        # also resolve (to run_amip defaults) but
                                        # are dead while e3sm_cam is not in the
                                        # selector; re-pin them from the CAM6
                                        # namelist before any run that adds it.

# ---- surface ------------------------------------------------------------------
surface_bulk_scheme: large_yeager_cesm  # CAM6/CESM shr_flux_atmOcn (LY04)
# (surface_gustiness_zi: 300 removed, user 2026-10-19: inert on this scheme --
# CESM2.1 runs gust_fac = 0, and the large_yeager_cesm law never reads it.)
surface_tiled: false
surface_z0_land: 0.1
mpas_land_lapse_K_per_km: 6.5
mpas_land_beta: 0.6
mpas_ice_skin_prognostic: true
albedo_ice: 0.80
# Melt-dependent sea-ice albedo on the prognostic skin (#2106, user decision
# 2026-10-24 after the July 1979 A/B: Arctic clear-sky reflected SW -19.2
# W/m2).  Named explicitly although "auto" resolves to it here.
sea_ice_albedo_scheme: maykut_untersteiner
sea_ice_melt_ramp_width_K: 1.0
ice_skin_melt_cap_width_K: 0.01
dynamic_albedo: true
snow_albedo_feedback: true
land_snow_tau_days: 150.0
snow_age_activation_K: 5000.0

# ---- land (unchanged) -----------------------------------------------------------
use_multilayer_land: true
land_update_seconds: 1800.0  # CAM6: CLM steps once per 1800 s coupling
                             # interval (user 2026-09-25).  = 16 x 112.5 s,
                             # forced with the interval-mean atmosphere; the
                             # offline LMIP driver also runs the tile at 1800 s.
                             # Was 300 (-> 337.5 s actual, every 3 steps).
multilayer_n_layers: 10
multilayer_soil_depth: 3.0
# Soil freeze/thaw ON (user 2026-09-26), as in CLM5 and as the land IC spin-up
# ran; the land calibration's tables were fitted with it off.
land_soil_freeze_thaw: true
# Snowpack: explicit bulk (one SWE reservoir), as every prior production run.
# Moving production to the layered pack (bulk -> layered) is an OPEN user decision.
land_snow_scheme: bulk
# Frozen soil impedes water flow, CLM5 form 10**(-e * ice fraction), e = 6
# (user 2026-09-28).  Active because freeze/thaw is on.
land_soil_ice_impedance_exponent: 6.0
# Soil-water numerics (user 2026-10-14): modified van Genuchten air-entry head [m].
land_soil_vg_air_entry_m: -0.02
# One-layer snow thermal node OFF (record only; A/B switch added 2026-10-03,
# user decision: off until the Sep-Dec arm verdict).
land_snow_insulation: false
land_soil_moisture_init_frac: 0.25
land_stomatal_beta: false
land_calibrated_physics: true
land_surface_scheme: two_leaf
land_canopy_max_iters: 10   # NEW BASELINE 2026-10-05 (user): A/B vs 60 = 4.2x/2.7x faster, land LH -1.5/-3.4 %; see fv3_duo_gaps/q_review/deck_ledgers.md
# Pinned at the current values (b0 stressed, no interception) so the deck
# records them; changing either is a labelled land baseline, not a default flip.
land_canopy_stress_b0: true
land_canopy_interception: false
# Two-leaf canopy solver smoothing widths, named so every run records them
# (user 2026-09-30): canopy-air RH <= 1 cap and stable zeta <= 0.5 cap.
land_canopy_rh_cap_smoothing_width: 0.01
land_canopy_zeta_cap_smoothing_width: 0.05
# Above-canopy Monin-Obukhov fixed-point iterations: 5 -> 10 (user 2026-09-30;
# 5 left unstable columns up to 11% off in aerodynamic resistance).
land_canopy_most_n_iters: 10
# Canopy Newton solve columns per batch: 64 -> 0 = one batch per GPU (user
# 2026-11-07; AMIP 4 A100 step 0.242 -> 0.138 s, answers equal to rounding).
land_canopy_solve_group: 0
# Two-leaf canopy fixes from I. Di Carlo's legoESM-enso fork (user 2026-11-20:
# fixed value is the default): semi-implicit ground heat flux (CLM5 eqs.
# 2.5.123-2.5.124, 2.6.15; explicit flux heated the 2.9 mm top soil layer
# 17-43 K in one step under sparse dry canopy) and unconverged columns keep
# their last admissible iterate (CLM5 2.5.3.2; the cold-state fallback gave
# 1,100-1,430 W/m2 sensible heat and was held).  NEW BASELINE.
land_canopy_implicit_ground_flux: true
land_canopy_fallback_from_last_iterate: true
mpas_land_beta_soil: true
# Surface drag over land from the land model's own roughness / canopy drag, not
# the ocean bulk law (user 2026-10-03: "a major default"; Dec 2001 45-70N land:
# the boundary layer received 0.04 Pa vs land model 0.23 vs ERA5 0.26).
mpas_land_stress_from_land: true
# Coastal cells: the non-land fraction's bulk fluxes on the ocean/ice surface,
# not on the land-blended T_sfc/q_sfc (#1320; CESM computes atm-ocn fluxes on
# ocean points only).  User 2026-10-19.
ocean_flux_on_ocean_surface: true
# Partly ice-covered sea cells: turbulent exchange on the ice skin and on the
# open water separately, the ice skin charged its own sensible/latent heat
# (#2573, user 2026-11-19).  Radiation is not split.
sea_ice_flux_split: true
mpas_land_params_refresh: true  # seasonal LAI / canopy height / wet-soil
                                # albedo rebuilt every land step, as the
                                # offline calibration runs the tables (user
                                # 2026-09-29; before, January LAI all year)
land_soil_init: aridity
# Land starting state: the res-6 spun-up soil water/snow (soil_ic_mpas6.npz) with
# soil TEMPERATURE replaced by ERA5 1979-01-01 00 UTC stl1-4 on non-glacier land
# (glaciers keep the spin-up). Built by scripts/data/regrid_land_ic.py
# --target-grid mpas --target-resolution 6 --target-clm-surfdata <the run's
# CLM surfdata> (soil water carried as relative wetness between dry floor and
# saturation, rebuilt on this run's soil, user 2026-11-12), then
# --era5-soil-t; file md5 0d3a2af0a438c1653ddc3f5c122966c9; provenance is in
# the file's metadata. data/ is gitignored:
# stage the file in the launch directory.
# _v3 (2026-11-15): rebuilt with --vg-air-entry-m -0.02 against main's moved
# dry floor (re-land of #2497/#2526; supersedes the _v2 lift-only file).
land_ic: data/lmip_soil_ic/soil_ic_mpas6_era5T_19790101_v3.npz

# ---- moisture smoothing / clamps (unchanged) -----------------------------------
mpas_qv_smooth_del2_m2s: 0.0
mpas_qv_smooth_del4_m4s: 3.6421e14
mpas_conservative_tracer_clamp: true
positivity_condensate_from_vapour: true  # pay net-negative column condensate from the column's vapour (ENSO-fork port, 2026-11-20)

# ---- forcing / IC / output (unchanged) ------------------------------------------
diurnal_cycle: true
orbital_insolation: true
# Orbit: present day, stated explicitly (null = present-day element; set
# orbit_year OR the three elements for a paleo/idealized orbit).
orbit_year: null
orbit_eccentricity: null
orbit_obliquity_deg: null
orbit_long_perihelion_deg: null
orbit_solver: berger_series
ic: era5
dataset: custom
sst_var: tosbcs
sst_offset: 0.0
sic_var: siconcbcs
sic_scale: 0.01
solar_source: spectral_file
solar_tsi_var: TSI
solar_spectral_var: SSI_frac
solar_spectral_band_order: rrtmg_sw   # explicit: SSI_frac is in RRTMG-SW band order (adapt_cmip6_solar.py); 'auto' only inferred it
ozone_forcing: external
ghg_forcing: external
aerosol_forcing: external
aerosol_reference_aod: 0.03  # 550 nm background AOD of the built-in reference
                             # aerosol field, read only when no aerosol file is
                             # given (the production launchers give one). Written
                             # out so it rests on no default (user decision
                             # 2026-10-15); equals the run_amip default.
# A date outside a multi-year forcing file's years raises (explicit: the
# end-record hold is an opt-in for deliberate experiments only; F37).
forcing_hold_outside_file_years: false
start_year: 1979
diag_days: 1
checkpoint_days: 30         # run control, carried from the outgoing production
                            # deck (not a CAM6 choice)
monthly_means: true
cmip_output: true
clear_sky_diag: true
#!/usr/bin/env bash
# =============================================================================
# legoESM AMIP — machine paths + launcher for the AUTHORITATIVE production config
# =============================================================================
# The authoritative PHYSICS / grid / dycore / forcing-selector / diagnostics
# parameter set lives in ONE machine-independent file:
#
#     config/amip/amip_production.yaml   (--config single source of truth)
#
# This shell file supplies ONLY the machine-specific PATHS (which are not
# committed into the physics config) and a ready launcher. Do NOT fork per-run
# physics here — edit the YAML (under review) so all runs stay comparable.
#
#   source config/amip/amip_production.sh
#   "${PY}" -u scripts/run/run_amip.py \
#       --config config/amip/amip_production.yaml \
#       "${AMIP_PATH_FLAGS[@]}" --days "${DAYS}" --output "${OUTDIR}"
#
# Provenance + validation: see the header of config/amip/amip_production.yaml
# (VALIDATED at C48/L40, job 25918469: albedo 0.292, rsut 99, pr 3.2, net TOA
# +6). Tracking: GitHub issue #636.
#
# OPEN ITEMS — need Pierre/Veronika to finalize (do NOT silently diverge):
#   * RUNNER: this is the run_amip.py (prescribed-SST) --config path. Pierre's
#     tuned config is a run_coupled.py --config YAML (cmip_ocean_slab.yaml).
#   * LAND: since 2026-07-07 the YAML runs the FULL multilayer Richards land
#     (use_multilayer_land: true — CLM PFT/texture surfdata, prognostic snow,
#     snow_albedo_feedback: true, stomatal-beta OFF per #741) under tiled
#     COARE3(ocean)/MOST(land) fluxes. Pierre's coupled config uses
#     slab_richards — the land-model choice is the remaining divergence to
#     reconcile, not snow albedo (the old "no snow-albedo feedback" caveat
#     applied to the retired simplified land tile).
#   * convective_cloud: the tracked YAML now ships TRUE (mirror the canonical
#     tuned base config/cmip/cmip_tuned_physics.yaml for parameter-identity with
#     the coupled CMIP slab). Prescribed-SST AMIP SCIENCE runs OVERRIDE it to
#     FALSE (best TOA): a 30-day C48 A/B (jobs 25929869 ON vs 25929870 OFF) showed
#     ON warms the tropics only +0.55 K while pushing albedo 0.295->0.370 and OLR
#     234.7->213.3 (~25 W/m^2 off CERES); SBM-alone is CMIP6-class (albedo 0.295
#     vs 0.290). `--convective-cloud` is a one-directional store_true and cannot
#     flip the YAML's true back off, so the OFF runner (amip_cmip6_chain.sbatch)
#     uses a tiny `include: amip_production.yaml` + `convective_cloud: false`
#     override. The residual tropical cold bias is a convective-heating /
#     surface-flux issue, not cloud-radiative.
# =============================================================================

# --- Machine-specific paths (Levante defaults; override via the environment) ---
: "${ICON_ROOT:=/pool/data/ICON/grids/public/mpim}"
# ERA5 IC with the analysis's 1..30 hPa levels added (scripts/data/
# add_era5_upper_levels.py) to the 13-level proxy, whose top is 50 hPa and
# whose L32 top layers start ~40/20 K cold.  SAME product as the Ginsburg twin
# (review 2026-10-10 F33).  Not staged on Levante.  Levante does have
# /scratch/b/b309178/era5_ic_1979-01-01.zarr: 37 pressure levels, 1-1000 hPa,
# 1979-01-01 00Z, 1 deg, built with cdo from /pool/data/ERA5 (amip_sota.sh uses
# it).
: "${ERA5_IC:=/scratch/b/b309178/era5_ic_1979-01-01_gproxy_L21.zarr}"
# Topography: 0.25 deg, built with scripts/data/prep_etopo_topography.py (its
# defaults) from NOAA ETOPO 2022 60 arc-second ice-surface relief.  The 1 deg
# file is coarser than the MPAS res-6 cells, so the per-grid terrain product
# (bin_latlon_to_cells) leaves 2193 cells with no source point and refuses to
# build: every production-resolution run crashed at setup.
: "${ETOPO:=/work/bb1596/b309178/diffESM/legoesm_ap/data/bathymetry/etopo2022_60s_0p25deg.nc}"
# SST/SIC: PCMDI CMIP7 AMIP boundary conditions (the protocol-standard tosbcs/
# siconcbcs mid-month files, 1870-2022) — the dataset every proven MPAS-lane
# run uses (Pierre's climeval_bech day-247+ and the replica chains).  The file
# is in degC -> the PAIRED offset below converts to K; keep SST and
# AMIP_SST_OFFSET consistent when overriding (the legacy ICON bc_sst files are
# already Kelvin and need offset 0).
: "${PCMDI_BC:=/pool/data/INPUT4MIP/data/input4MIPs/CMIP7/CMIP/PCMDI/PCMDI-AMIP-1-1-10}"
: "${SST:=${PCMDI_BC}/ocean/mon/tosbcs/gn/v20250807/tosbcs_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-10_gn_187001-202212.nc}"
: "${SIC:=${PCMDI_BC}/seaIce/mon/siconcbcs/gn/v20250807/siconcbcs_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-10_gn_187001-202212.nc}"
: "${AMIP_SST_OFFSET:=273.15}"
# Legacy alternative (Kelvin; pair with AMIP_SST_OFFSET=0):
#   SST=${ICON_ROOT}/0019/sst_and_seaice/r0001/bc_sst_1979_2016.nc
#   SIC=${ICON_ROOT}/0019/sst_and_seaice/r0001/bc_sic_1979_2016.nc
: "${SOLAR:=${ICON_ROOT}/common/solar_radiation/swflux_14band_cmip6_1850-2299-v3.2.nc}"
: "${GHG:=${ICON_ROOT}/independent/greenhouse_gases/greenhouse_historical_plus.nc}"
# Ozone defaults to the merged 1979-2014 transient file (432 monthly records),
# aerosol and volcanic to 1850-2014 products (below), so the DEFAULT run is
# CMIP6-compliant for the full AMIP period. The old single-year (Kinne 1979,
# volcanic 1979) and 1950-1999 ozone pool files silently freeze/hold forcing
# past their coverage — only override back to them for deliberate fixed-forcing
# sensitivity runs.
: "${FORCING_1979_2014:=/work/bb1596/b309178/diffESM/legoesm_ap/data/forcing/cmip6_1979-2014}"
: "${OZONE:=${FORCING_1979_2014}/vmro3_input4MIPs_ozone_CMIP_UReading-CCMI-1-0_gn_197901-201412.nc}"
# MACv2-SP anthropogenic plumes on the Kinne MACv1 pre-industrial background,
# 1850-2014: fine+coarse for radiation AND for the droplet-number (CCN) path,
# the TOTAL AOT the Andreae (2009) fit was built on (review 2026-10-10 F33/F37;
# CCN was fine mode until the 2026-10-18 oracle audit, user decision 2026-10-19).  Same basenames on both machines; build
# them here with scripts/data/adapt_cmip6_aerosol.py (see the Ginsburg twin).
: "${AEROSOL:=${FORCING_1979_2014}/aeropt_macv2sp_sw_b14_tot_1850-2014_MACv1bg.nc}"
: "${AEROSOL_CCN:=${FORCING_1979_2014}/aeropt_macv2sp_sw_b14_tot_1850-2014_MACv1bg.nc}"
# Volcanic: ETH CMIP6 stratospheric aerosol v4.0.0 (ECHAM6 SW + LW bands),
# 1850-2014, the SAME product as the Ginsburg twin (review 2026-10-10 F33/F37).
# Its 1979-2014 records equal the old *_1979-2014.nc file; the loader output
# differs only by calendar stamps and the first half of January 1979 (#1948).
# Build it here before a launch: scripts/data/adapt_cesm_forcing.py volcanic
# --years 1850 2014 on the raw CMIP_ECHAM6_radiation_v4.nc (see the twin).
: "${VOLCANIC:=${FORCING_1979_2014}/bc_aeropt_cmip6_volc_lw_b16_sw_b14_1850-2014_IACETH-v4.nc}"
# CLM surfdata for the multilayer Richards land model (use_multilayer_land): the
# PFT/texture/glacier surface map. Compute nodes have NO internet, so the auto-
# download in clm_surface_map.download_clm_surfdata() FAILS (SSL) — this file
# MUST be staged locally and passed via --clm-surfdata-path (staged 2026-07-02).
: "${CLM_SURFDATA:=/work/bb1596/b309178/diffESM/legoesm_ap/data/clm/surfdata_1.9x2.5_16pfts_CMIP6_simyr2000.nc}"
# Subgrid-orography stddev (SSO) for the orographic GWD launch (#1514). With
# gravity_wave_drag=mcfarlane+hines and NO SSO file, McFarlane falls back to
# the scalar h_topo=500 m on EVERY column — a fictional 500-m mountain over
# the open ocean, measured at -0.29 Pa column drag over 40-60S (2x the entire
# observed surface stress): it removed the eddy-driven westerly belt within a
# week of the ERA5 IC and suppressed the storm tracks in BOTH hemispheres.
# File built by scripts/data/prep_subgrid_orography.py from etopo_1deg_clean
# (--fine-res-deg 1.0 --block-deg 2.0; Southern-Ocean mean sgh 500 -> 3.6 m).
# ADOPTING THIS CHANGES THE MODEL CLIMATE (validated 90-d + 274-d restart
# arms, issue #1514): any comparison spanning it is confounded. Opt OUT with
# AMIP_SSO="" (pre-fix reproduction runs only).
# _seamfix (2026-10-07, #1712): same recipe rebuilt after the regridder seam
# fix; differs from sso_stdh_2deg_from1deg.nc only in the 358-360E column,
# where the old file was built with a fake 0-m strip. sha256 of both in #1884.
: "${AMIP_SSO:=/work/bb1596/b309178/diffESM/legoesm_ap/data/orography/sso_stdh_2deg_from1deg_seamfix.nc}"
# Land-sea mask (sftlf, fraction) built from the SAME CLM surfdata by
# scripts/data/build_sftlf_from_surfdata.py (2026-07-22). Without it f_land is
# derived from ETOPO elevation>0, and ETOPO's inland-sea BATHYMETRY (Caspian
# -28 m, Aral) classifies those basins as OCEAN -> prescribed nearest-neighbour
# SST -> uncapped potential evaporation (the 2000 W/m^2 hfls / 177 mm prw
# central-Asia hotspots in the 2-yr pilot). The CLM mask keeps the surface
# tiling consistent with the land model's own footprint (Caspian/Aral = land).
# Opt back to the legacy elevation mask with AMIP_LAND_MASK="" (deliberate
# reproduction runs only).
: "${AMIP_LAND_MASK:=/work/bb1596/b309178/diffESM/legoesm_ap/data/clm/sftlf_clm_1.9x2.5.nc}"

# --- The machine-specific PATH flags (everything else is in the YAML) ---------
# Harmonized surfdata: the CANOPY parameter source (per-PFT canopy height,
# roughness ratio, Vcmax25, band albedos).  The default land surface scheme is
# the two-leaf canopy, which is REFUSED without this — the CLM surfdata below
# supplies soil texture and PFT cover, not canopy structure.  Two datasets, two
# jobs; a canopy run needs both staged.
: "${AMIP_SURFDATA:=${REPO:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM}/data/legoesm_surfdata_c260716.nc}"

AMIP_PATH_FLAGS=(
  --surfdata "${AMIP_SURFDATA}"
  --ic-path "${ERA5_IC}"
  --forcing-path "${SST}"
  --sic-path "${SIC}"
  --sst-offset "${AMIP_SST_OFFSET}"
  --solar-file "${SOLAR}"
  --ozone-file "${OZONE}"
  --ghg-file "${GHG}"
  --aerosol-file "${AEROSOL}"
  --aerosol-ccn-file "${AEROSOL_CCN}"
  --volcanic-aerosol-file "${VOLCANIC}"
  --topography "${ETOPO}"
  --clm-surfdata-path "${CLM_SURFDATA}"
)
# Volcanic LONGWAVE absorption (ext_earth band; post-eruption stratospheric
# heating). ON by default for the campaign (decision 2026-07-21): reader
# validated against the merged 1979-2014 file (Pinatubo-peak LW AOD 0.020 vs
# 0.0005 quiet-1979; El Chichon visible). LW AOD ~0 outside 1982-84/1991-93,
# so quiet years are unaffected; the pilot decade exercises El Chichon before
# the full 36-yr. No forward run has exercised the LW channel yet — first
# post-1982 pilot output is the validation gate. Opt OUT (e.g. to reproduce
# pre-2026-07-21 SW-only runs) with AMIP_VOLCANIC_LW=0.
if [[ "${AMIP_VOLCANIC_LW:-1}" == "1" ]]; then
  AMIP_PATH_FLAGS+=( --volcanic-aerosol-lw )
fi
# CLM-derived land mask (see AMIP_LAND_MASK above). Side-effect scope audited
# 2026-07-22: with use_multilayer_land=true the ONLY climate-relevant change is
# f_land itself (the forced slab_land_active is shadowed by the multilayer tile
# branch; surfdata albedo needs --surfdata, which this launcher never passes).
if [[ -n "${AMIP_LAND_MASK}" ]]; then
  AMIP_PATH_FLAGS+=( --land-mask-file "${AMIP_LAND_MASK}" )
fi
# Subgrid orography for the orographic GWD launch (see AMIP_SSO above, #1514).
# Without this flag the run trips the loud scalar-fallback warning (PR #1540)
# and reproduces the pseudo-mountain climate. Opt-out is DELIBERATE only.
if [[ -n "${AMIP_SSO}" ]]; then
  AMIP_PATH_FLAGS+=( --default-subgrid-orography-file "${AMIP_SSO}" )
fi

# Refuse at job start, before any compile or step time is spent, when the
# aerosol / CCN aerosol / volcanic product is not staged: the run would
# otherwise die at the first forcing load.  Call it right after sourcing:
#   amip_require_forcing_files || exit 3
amip_require_forcing_files() {
  local _f
  for _f in "${AEROSOL}" "${AEROSOL_CCN}" "${VOLCANIC}"; do
    [[ -f "${_f}" ]] || { echo "missing forcing file ${_f}: stage it (see config/amip/amip_production.sh)" >&2; return 3; }
  done
}

exec
/usr/bin/bash -lc "rg -n 'ghg|cfc|year|SSI|spectral|ozone|overhead|volcan|omega|g_sun|ext_sun|ext_earth' packages/tools/legoesm/forcing/external.py" in /glade/u/home/pg2328/legoesm
 succeeded in 0ms:
15:- Aerosol: zonal-mean AOD climatology with optional volcanic contribution
16:- Solar: constant/file TSI and full spectral (per g-point) forcing
39:# CF calendars whose year length differs from the model's noleap 365-day
54:# volcanic aerosol is fed to RRTMGP as an ABSORPTION optical depth
55:# (ext*(1-omega)) placed at its true stratospheric pressure.
63:# volcanic file occupies lies in the 11-47 km stratospheric layers.
77:# spectral radiative kernel): on the real 1979 file a +/-10 K change moves the
84:# per-band SW aerosol OD (Kinne / CMIP6 volcanic ``ext_sun``).
96:    to map the CMIP6 volcanic file's altitude axis onto pressure for
136:    """Un-normalised Planck-emission weight per spectral band at ``temp_k``.
144:    would fold in the full spectral radiative kernel.  Only relative weights
211:      ``datetime64[ns]`` representable range (~year 100–2262).
219:      and converted to days since the first record, so e.g. CMIP6 ozone
223:      as-is after a simple ``float64`` cast (e.g. ``"year as %Y.%f"``
253:        # "months since" / "years since" are only supported by cftime on
256:        # convert with the 30.4375-day average month (year/12).  For
263:        if "years since" in ul:
304:    :func:`get_ozone_at_time` to map a simulation day onto the file's
327:    ``start_year-01-01``) onto the file's absolute time axis.
350:    practice (CMIP6 input4MIPs ozone / aerosol / GHG annual default
380:    * Numeric arrays with CF units of ``months since`` / ``years since``
382:      parse ``<ref>`` manually and add the integer month / year offset
392:    falls back to the 1850 CMIP6 reference year in
409:        if "months since" in ul or "years since" in ul:
416:            # Add whole months / years, then fractional part as days so
424:                ny = anchor.year + total_m // 12
428:            else:  # years since
432:                    anchor.year + whole, anchor.month, anchor.day,
500:    fired per simulated step with up to 112 g-points (CMIP6 spectral
501:    SSI), forcing 112 Python calls into ``np.interp`` per step.
529:    when the file spans multiple years.
538:    used by the ozone loader.
587:    # Collapse any remaining non-(time, lat) dimensions (e.g. spectral
665:def _load_volcanic_cmip6(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
666:    """Load a CMIP6 / MPI-M stratospheric volcanic aerosol file.
668:    These files (``bc_aeropt_cmip6_volc_lw_b16_sw_b14_<year>.nc``) do not
670:    SW extinction ``ext_sun(solar_bands, lat, altitude, month)`` in
676:    2. Mean over the spectral-band axis to produce a representative
690:    for non-cyclic multi-year volcanic time-series), see
691:    :func:`_load_volcanic_cmip6_anchored`.
693:    mid_days, _first_date, lat, aod = _load_volcanic_cmip6_anchored(path)
698:def _load_volcanic_extinction_anchored(
703:    """Load a CMIP6 volcanic stratospheric-extinction variable (anchored).
705:    Shared body for the SW (``ext_sun``) and LW (``ext_earth``) loaders.
707:    (trapezoid → AOD), collapses the spectral-band axis to a representative
713:    map a simulation day onto a multi-year file's absolute axis (1982 El
717:    The SW/LW bands are NOT interchangeable: reading ``ext_earth`` as a SW
718:    quantity (or ``ext_sun`` as LW) would inject the wrong band into the
729:            f"No CMIP6 volcanic {kind_label} extinction variable "
743:            f"No altitude coordinate in CMIP6 volcanic file {path!r}",
754:            f"No 'lat'/'latitude' variable in CMIP6 volcanic file {path!r}",
757:    # Month / time axis — CMIP6 volcanic may use ``month`` or ``time``.
769:            f"No time/month dimension in CMIP6 volcanic file {path!r}",
779:    # Collapse the spectral-band axis to a single broadband AOD: solar-
806:def _load_volcanic_cmip6_anchored(
809:    """SHORTWAVE volcanic extinction (``ext_sun``); see
810:    :func:`_load_volcanic_extinction_anchored` for the shared body."""
811:    return _load_volcanic_extinction_anchored(path, "ext_sun", "SW")
814:def _load_volcanic_lw_absorption_profile(
817:    """LW absorption profile; see :func:`_load_volcanic_profile`."""
818:    return _load_volcanic_profile(path, "lw")
822:def _load_volcanic_profile(
825:    """CMIP6 volcanic LW aerosol as a PRESSURE-RESOLVED absorption-OD profile.
827:    ``kind="sw"`` returns the SHORTWAVE ``ext_sun`` EXTINCTION profile instead
836:    * **B1 extinction -> absorption.**  ``ext_earth`` is EXTINCTION; the LW slot
838:      co-located ``omega_earth``: ``abs = ext*(1-omega)``.  ``omega_earth``
841:      ``omega_earth`` the sulfate LW ``omega ~ 0`` approximation is used
849:    * **B3 spectral collapse.**  The 16 terrestrial bands ARE collapsed to one
855:      exact single-tau response would fold in the full spectral radiative
868:    Raises ``ValueError`` if the file lacks ``ext_earth`` (``ext_sun`` for
871:    ext_name = "ext_sun" if kind == "sw" else "ext_earth"
876:            f"No CMIP6 volcanic extinction {ext_name!r} in {path!r}.",
882:    # B1: single-scattering albedo -> absorption fraction (1-omega) per band.
884:        omega = 0.0  # SW slot wants extinction, not absorption
885:    elif "omega_earth" in ds.data_vars:
886:        omega = np.clip(
887:            np.asarray(ds["omega_earth"].values, dtype=np.float64), 0.0, 1.0,
890:        omega = np.zeros_like(ext)  # sulfate LW omega ~ 0 approximation
891:    abs_ext = ext * (1.0 - omega)  # absorption extinction [1/km]
898:        raise ValueError(f"No altitude coordinate in volcanic file {path!r}")
909:        raise ValueError(f"No 'lat'/'latitude' in volcanic file {path!r}")
921:        raise ValueError(f"No time/month dimension in volcanic file {path!r}")
974:def _load_volcanic_auto(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
975:    """Dispatch ``path`` to the right volcanic loader.
979:    schema the file follows.  See :func:`_load_volcanic_auto_anchored` for a
981:    multi-year non-cyclic dispatch in :func:`get_aerosol_at_time`).
983:    mid_days, _first_date, lat, data = _load_volcanic_auto_anchored(path)
988:def _load_volcanic_auto_anchored(
991:    """Like :func:`_load_volcanic_auto` but also returns ``first_date``.
993:    Criterion: presence of the CMIP6 ``ext_sun`` variable triggers
994:    :func:`_load_volcanic_cmip6_anchored`; otherwise fall back to the
1003:    if "ext_sun" in dvars:
1004:        return _load_volcanic_cmip6_anchored(path)
1005:    if "ext_earth" in dvars and "aod" not in dvars:
1008:            "('ext_earth') and no SW extinction ('ext_sun'); cannot be "
1014:def _ozone_unit_factor(units: str, varname: str) -> float:
1015:    """Return the multiplicative factor that converts ozone from the
1018:    The legoESM radiation kernel (RRTMG / RRTMGP) consumes ozone as a
1019:    dimensionless mole fraction.  CMIP6 input4MIPs ozone is supplied
1021:    ozone as ``tro3`` in kg/kg (mass mixing ratio).  Without unit
1035:       - ``vmro3`` / ``o3`` / ``O3`` / ``ozone`` → 1.0 (vmr — CMIP6 default).
1069:            if varname in {"vmro3", "ozone", "o3", "O3"}:
1085:        f"Unrecognised ozone units {units!r} on variable {varname!r}. "
1094:    for multi-year files.
1103:      Used by :func:`get_ozone_at_time` to map the simulation day onto
1104:      the file's absolute time axis for multi-year (``ntime > 12``)
1105:      ozone files.
1108:      When the file's ozone variable carries ``units = "kg kg-1"`` (or
1124:    factor = _ozone_unit_factor(var_units, varname)
1127:            f"[ozone-loader] Converting {varname!r} from units={var_units!r} "
1188:    spectral_var: str,
1190:    """Load time-varying solar spectral forcing (Zarr or NetCDF).
1196:      ``TSI`` in canonical CMIP6 / MPI-M spectral-solar files
1197:    - spectral weights (2-D ``(time, ngpt)`` or ``(time, nband)``) —
1198:      e.g. ``solar_fraction_by_gpt`` or the CMIP6 ``SSI_frac``
1202:    read MPI-M files where the stored names are ``"TSI"`` / ``"SSI_frac"``.
1207:        raise ValueError(f"No 'time' variable in spectral solar file {path!r}")
1211:    # MPI-M CMIP6 spectral-solar file stores ``TSI`` / ``SSI_frac`` while
1218:    spectral_actual = _resolve(spectral_var)
1219:    if spectral_actual is None:
1222:            f"Spectral variable {spectral_var!r} not found in {path!r}",
1224:    spec = np.asarray(ds[spectral_actual].values, dtype=np.float64)
1228:            f"Spectral variable {spectral_actual!r} must have shape "
1246:    """Interpolate a monthly-cyclic field (12, ...) to a day of year.
1273:    """Linear interpolation along a non-cyclic multi-year monthly axis.
1275:    For multi-year forcing files (e.g. CMIP6 ozone 1850–2014, 1980
1284:    holding an end record across years is a different forcing, not an
1286:    129 years without a word).  ``hold_outside=True`` is the explicit opt-in
1287:    that holds the end record instead (``forcing_hold_outside_file_years``).
1305:                "Stage a file covering the run's years, check start_year, or "
1307:                "(forcing_hold_outside_file_years)."
1313:    # ``for k in range(...): np.interp(...)`` loop.  For CMIP6 ozone
1330:    """Return ``(year, month, day_of_month, day_frac)`` for a calendar-aware
1333:    if hasattr(date, "year") and hasattr(date, "month"):
1340:        return int(date.year), int(date.month), int(date.day), day_frac
1343:    year = months_int // 12 + 1970
1351:    return year, month, dom, day_frac
1354:def _model_noleap_date(sim_day: float, start_year: int) -> tuple[int, int, int, float]:
1355:    """Calendar date ``(year, month, day_of_month, day_frac)`` of a model
1356:    day count (days since ``start_year-01-01`` on the model's noleap
1359:    year_off = int(np.floor(sim_day / NOLEAP_DAYS_PER_YEAR))
1360:    rem = sim_day - year_off * NOLEAP_DAYS_PER_YEAR
1369:    return start_year + year_off, month, dom, day_frac
1376:    day-of-year (audit F4), returning ``(mid_days, data)`` co-sorted.
1381:    phase shift.  Offsetting by the first record's noleap day-of-year
1385:    mid-year (e.g. a July-first file) still satisfies the ascending-axis
1387:    units-less axis (``first_date is None``) already carries day-of-year
1391:    re-anchored; within-year spacing keeps the file's native elapsed days, so a
1392:    leap-year-dated climatology's post-February records sit <=1 day late.
1395:    ``months since`` / ``years since`` axes to average-length elapsed days, so
1399:    residual on a rare leap-year-dated file.
1416:def _simday_to_file_day(sim_day: float, start_year: int,
1418:    """Map a simulation day (days since ``start_year-01-01`` on the model's
1441:      4 years (the file gains leap days the model clock never lives
1454:            y, m, d, day_frac = _model_noleap_date(sim_day, int(start_year))
1474:            sim_epoch = ctor(int(start_year), 1, 1)
1476:            sim_epoch = cftime.DatetimeGregorian(int(start_year), 1, 1)
1477:        if hasattr(first_date, "year") and hasattr(first_date, "month"):
1486:        anchor = np.datetime64(f"{int(start_year):04d}-01-01")
1588:    # Canonicalise to ascending log_p_src.  CMIP6 ozone files (e.g.
1589:    # ``vmro3_input4MIPs_ozone_*.nc``) ship plev descending (1000→0.1 hPa);
1592:    # near-zero ozone in the synthetic AMIP deck unit test).
1634:    where lat=1, lon=1 and time is in fractional years.
1646:    cfc11_pptv : float
1648:    cfc12_pptv : float
1652:        annual global-mean file with year-indexed time axis).
1656:    start_year : int
1657:        Simulation start year, used to convert simulation day → calendar
1658:        year when source="annual_file".
1663:    cfc11_pptv: float = 240.0
1664:    cfc12_pptv: float = 530.0
1667:    start_year: int = 1979
1674:# spellings).  Used by ``_load_ghg_annual_file`` so the rest of the
1697:def _load_ghg_annual_file(path: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
1702:    uses units ``"year as %Y.%f"`` giving fractional year values
1707:    (years, data) where years is shape (N,) and data maps variable
1721:    years = np.asarray(ds["time"].values, dtype=np.float64)
1740:        if arr.ndim != 1 or arr.shape[0] != years.shape[0]:
1745:                f"({years.shape[0]},))"
1777:    return years, data
1780:def get_ghg_at_time(config: GHGConfig, day: float) -> dict:
1791:    dict with keys "co2_ppmv", "ch4_ppbv", "n2o_ppbv", "cfc11_pptv",
1792:    "cfc12_pptv".
1799:            "cfc11_pptv": config.cfc11_pptv,
1800:            "cfc12_pptv": config.cfc12_pptv,
1808:        result["cfc11_pptv"] = config.cfc11_pptv
1809:        result["cfc12_pptv"] = config.cfc12_pptv
1816:        years, data = _load_ghg_annual_file(config.path)
1817:        # Convert simulation day → fractional year on the model's noleap
1820:        # forcing channels use.  365.25 here drifted ~9 days over a 35-year
1822:        year = config.start_year + day / 365.0
1823:        # ``_load_ghg_annual_file`` returns DIMENSIONLESS mole fractions
1826:        # fixed factors — the inverse of ``ghg_concentrations_to_vmr``.
1831:            "co2_ppmv": _interp_1d(years, data["CO2"], year) * 1.0e6,
1832:            "ch4_ppbv": _interp_1d(years, data["CH4"], year) * 1.0e9,
1833:            "n2o_ppbv": _interp_1d(years, data["N2O"], year) * 1.0e9,
1834:            "cfc11_pptv": _interp_1d(years, data["CFC_11"], year) * 1.0e12,
1835:            "cfc12_pptv": _interp_1d(years, data["CFC_12"], year) * 1.0e12,
1836:            **{key: _interp_1d(years, data[var], year) * 1.0e12
1844:# File variable -> (get_ghg_at_time key [pptv], RRTMGP gas name).  Physical
1848:    "CFC_22": ("cfc22_pptv", "cfc22"),
1854:def ghg_concentrations_to_vmr(ghg: dict) -> dict:
1859:    ghg : dict
1860:        As returned by ``get_ghg_at_time`` (ppmv/ppbv/pptv units).
1867:        "co2": ghg["co2_ppmv"] * 1.0e-6,
1868:        "ch4": ghg["ch4_ppbv"] * 1.0e-9,
1869:        "n2o": ghg["n2o_ppbv"] * 1.0e-9,
1871:    if "cfc11_pptv" in ghg:
1872:        vmr["cfc11"] = ghg["cfc11_pptv"] * 1.0e-12
1873:    if "cfc12_pptv" in ghg:
1874:        vmr["cfc12"] = ghg["cfc12_pptv"] * 1.0e-12
1876:        if key in ghg:
1877:            vmr[gas] = ghg[key] * 1.0e-12
1888:    When enabled, provides column ozone or 3D ozone mixing ratio
1894:    - Variable: 'ozone', 'vmro3', 'o3', 'O3', or 'tro3' (auto-detected)
1902:        Whether ozone forcing is active.
1906:        Path to ozone NetCDF file.
1908:        If True and path is empty, use a built-in reference ozone profile.
1912:        Peak ozone VMR [mol/mol] for the reference profile.
1914:        Width of reference ozone in log-pressure space.
1916:        If True, reference ozone increases toward poles.
1917:    start_year : int
1918:        Simulation start year.  Used for the non-cyclic dispatch: when
1919:        the ozone file has ``ntime > 12`` records (e.g. CMIP6
1920:        ``vmro3_input4MIPs_ozone_CMIP_UReading-CCMI-1-0_gn_185001-201412.nc``
1921:        with 1980 months), ``get_ozone_at_time`` treats ``mid_days`` as a
1924:        interannual ozone evolution instead of collapsing to a fixed
1935:    start_year: int = 1850
1936:    # Opt-in: hold a multi-year file's end record outside its years instead
1938:    hold_outside_file_years: bool = False
1941:def _reference_ozone_profile(
1946:    """Reference ozone profile used when external ozone is active without a file."""
1964:_OZONE_VARNAMES = ("ozone", "vmro3", "o3", "O3", "tro3")
1968:def _detect_ozone_varname(path: str) -> str:
1969:    """Auto-detect the ozone variable name in a Zarr store or NetCDF file.
1971:    Tries common names: 'ozone', 'vmro3' (CMIP6), 'o3', 'O3', 'tro3'.
1980:        f"No ozone variable found in {path!r}. "
1985:def get_ozone_at_time(config: OzoneConfig, day: float,
1988:    """Return ozone field at a given simulation day.
1994:        Day of year (fractional).
1999:        Model pressure levels [Pa]. If provided and the ozone file
2004:    None if ozone is disabled.
2006:    (lat_grid.shape, nlev) if 3D, or dict with "lat"/"ozone" keys
2015:            return {"lat": lat_ref, "ozone": np.asarray(_reference_ozone_profile(lat_ref_rad, None, config))}
2016:        return _reference_ozone_profile(lat_grid, p_grid, config)
2023:    varname = _detect_ozone_varname(config.path)
2027:    # Dispatch cyclic (12-month climatology) vs non-cyclic (multi-year
2028:    # interannually varying file, e.g. CMIP6 ozone 1850–2014 @ 1980 months).
2029:    # Keyed on ``len(mid_days) > 12`` AND ``config.start_year`` is required
2032:        file_day = _simday_to_file_day(day, config.start_year, first_date)
2033:        ozone_interp = _interp_monthly_noncyclic(
2035:            config.hold_outside_file_years)
2038:        # day-of-year (audit F4: days-since-first-record put mid-Jan at Jan 1).
2042:        ozone_interp = _interp_monthly_cyclic(mid_anchored, data_anchored, day)
2046:        ozone_on_grid = _interp_zonal_to_grid(lat, ozone_interp, lat_grid)
2048:        if plev is not None and p_grid is not None and ozone_on_grid.ndim > len(lat_grid.shape):
2049:            ozone_on_grid = _interp_vertical(ozone_on_grid, plev, p_grid)
2050:        return ozone_on_grid
2052:        return {"lat": lat, "ozone": ozone_interp}
2084:    volcanic_enabled : bool
2085:        If True, add volcanic aerosol forcing on top of baseline aerosol.
2086:    volcanic_path : str
2087:        NetCDF path for volcanic AOD climatology/time-series (same schema as aerosol).
2088:    volcanic_scale : float
2089:        Multiplicative scaling applied to volcanic AOD.
2090:    volcanic_lw_enabled : bool
2091:        If True, ALSO load the LONGWAVE volcanic stratospheric extinction
2092:        (``ext_earth`` in CMIP6 ``bc_aeropt_cmip6_volc_*`` files) as a
2095:        when the volcanic file has no LW band) the LW aerosol path returns
2098:        ``volcanic_enabled`` path, which is unaffected.  (gap #9)
2106:    volcanic_enabled: bool = False
2107:    volcanic_path: str = ""
2108:    volcanic_scale: float = 1.0
2109:    volcanic_lw_enabled: bool = False
2114:    # Calendar year of simulation day 0.  Required for the non-cyclic
2115:    # dispatch in :func:`get_aerosol_at_time` so multi-year volcanic
2119:    # cyclic axis.  Mirrors :class:`OzoneConfig.start_year`.
2120:    start_year: int = 1979
2121:    # Opt-in: hold a multi-year file's end record outside its years instead
2123:    hold_outside_file_years: bool = False
2144:        Day of year (fractional).
2172:        # (multi-year, e.g. real CMIP6 input4MIPs aerosol).  Keyed on
2173:        # ``len(mid_days) > 12``; ``config.start_year`` maps the
2175:        # anchor is present.  Without this dispatch, a 36-year aerosol
2192:            file_day = _simday_to_file_day(day, config.start_year, first_date)
2195:                config.hold_outside_file_years)
2198:            # day-of-year (audit F4).
2233:    # Optional volcanic contribution.
2234:    if config.volcanic_enabled:
2235:        if config.volcanic_path:
2236:            # ``_load_volcanic_auto_anchored`` handles both the legacy
2238:            # ``ext_sun(solar_bands, lat, altitude, month)`` schema
2240:            # CF time anchor so multi-year volcanic time series with
2243:            # path mod-365.25-wrapped a multi-year file onto 12 months
2246:            mid_days_v, first_date_v, lat_v, data_v = _load_volcanic_auto_anchored(
2247:                config.volcanic_path
2251:                    day, config.start_year, first_date_v,
2255:                        mid_days_v, data_v, file_day_v, config.volcanic_path,
2256:                        config.hold_outside_file_years)
2257:                    * config.volcanic_scale
2265:                    * config.volcanic_scale
2269:                # Sum trailing dims for multi-dimensional volcanic files
2285:                "AerosolConfig.volcanic_path must be set when volcanic_enabled=True "
2301:    """Return the LONGWAVE volcanic aerosol as a pressure-resolved profile.
2303:    Mirrors :func:`get_aerosol_at_time` but returns the volcanic
2305:    loaded from the ``ext_earth`` band of the CMIP6
2307:    :func:`_load_volcanic_lw_absorption_profile` (extinction -> absorption via
2308:    ``1-omega``, Planck-weighted gray band collapse, and the file's native
2310:    climatology, so the baseline is zero and the result is the volcanic LW
2317:        Day of year (fractional).
2324:        If LW aerosol is disabled (``volcanic_lw_enabled`` False), or enabled
2325:        with no ``volcanic_path``, or the file carries no ``ext_earth`` LW
2327:        byte-identical to no volcanic LW aerosol.
2336:    # Default OFF: only the explicit LW switch + a real volcanic file with
2337:    # an ``ext_earth`` band produces nonzero LW aerosol.  Anything else
2339:    if not config.volcanic_lw_enabled or not config.volcanic_path:
2341:    return _volcanic_profile_at_time(config, day, lat_grid, "lw")
2344:def get_aerosol_sw_volcanic_at_time(config: AerosolConfig, day: float,
2346:    """SHORTWAVE volcanic extinction as a pressure-resolved profile.
2348:    Same contract as :func:`get_aerosol_lw_at_time` but for ``ext_sun``
2349:    (solar-weighted gray band collapse), gated on ``volcanic_enabled``.  None
2350:    for a legacy column ``aod(time, lat)`` volcanic file (no profile) -- the
2353:    if not config.volcanic_enabled or not config.volcanic_path:
2355:    return _volcanic_profile_at_time(config, day, lat_grid, "sw")
2358:def _volcanic_profile_at_time(config, day, lat_grid, kind):
2359:    """Shared body of the LW / SW volcanic profile samplers."""
2362:            _load_volcanic_profile(config.volcanic_path, kind)
2365:        # File has no ext_earth / ext_sun band — treat as "no source".
2370:        file_day_v = _simday_to_file_day(day, config.start_year, first_date_v)
2372:            mid_days_v, prof_v, file_day_v, config.volcanic_path,
2373:            config.hold_outside_file_years)
2379:    prof_t = prof_t * config.volcanic_scale  # (nlat, nlayer)
2408:    solar cycle, volcanic dimming).
2418:        "constant", "file", or "spectral_file".
2423:    spectral_var : str
2424:        Variable name for per-g-point solar fractions when source="spectral_file".
2425:    spectral_band_order : str
2426:        Band ordering of a *per-band* (14-band) ``spectral_file`` input,
2432:          ``SSI_frac`` or ``swflux_14band`` filename); otherwise leave
2439:    normalize_spectral : bool
2440:        If True, normalize interpolated spectral fractions to sum to one.
2441:    start_year : int
2442:        Simulation start year.  Used to convert simulation day (days since
2450:    spectral_var: str = "solar_fraction_by_gpt"
2451:    spectral_band_order: str = "auto"
2452:    normalize_spectral: bool = True
2453:    start_year: int = 1979
2461:    """Return solar forcing dict with broadband TSI and optional spectral weights.
2473:    if config.source not in ("file", "spectral_file"):
2481:    # Map simulation day (days since ``start_year-01-01``) onto the
2483:    # (e.g. CMIP6 spectral-solar "days since 1850-01-01") anchor the
2489:    abs_day = _simday_to_file_day(day, config.start_year, anchor)
2495:    if config.source == "spectral_file":
2499:            config.spectral_var,
2506:                f"{config.path!r} (solar_source='spectral_file')",
2513:        # Expand only when the loaded spectral length matches the table's
2522:        if config.spectral_band_order not in ("auto", "as_is", "rrtmg_sw"):
2524:                "SolarConfig.spectral_band_order must be 'auto', 'as_is' "
2525:                f"or 'rrtmg_sw'; got {config.spectral_band_order!r}",
2543:            #                 MPI-M CMIP6 signature (variable ``SSI_frac``
2549:                "ssi_frac" in (config.spectral_var or "").lower()
2552:            if config.spectral_band_order == "rrtmg_sw" or (
2553:                config.spectral_band_order == "auto" and _looks_cmip
2555:                if config.spectral_band_order == "auto":
2558:                        "(spectral_var=%r); applying 'rrtmg_sw' band order "
2560:                        "#322). Set spectral_band_order explicitly to "
2561:                        "override.", config.path, config.spectral_var,
2565:        if config.normalize_spectral:
2569:                    f"Interpolated spectral forcing has non-positive sum at day={day}",
2584:    input spectral axis needs the band→g-point expansion.  Cached so
2596:    """Expand per-band spectral fractions to per-g-point fractions
2599:    CMIP6 solar files (e.g. ``SSI_frac``) store one *band-integrated*
2619:    ``Σ_{g ∈ band b} f_g = f_b`` AND the physically correct spectral
2643:    # Per-g-point solar source = the physical within-band spectral shape.
2681:    The MPI-M CMIP6 spectral-solar file (``swflux_14band_cmip6``) stores
2698:    This is applied ONLY on the ``spectral_file`` ingest path (per-band
2743:    ghg: GHGConfig = GHGConfig()
2744:    ozone: OzoneConfig = OzoneConfig()

exec
/usr/bin/bash -lc "rg -n 'aerosol|volcan|strat|ghg|365|ozone|solar|overhead' packages/tools/legoesm/forcing/surface_utils.py packages/coupler/legoesm/driver/model_driver.py | head -180; rg -n 'cfc22|ccl4|cf4|aerosol_ssa|aerosol_g|volcan|aerosol_tau|solar' packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py packages/atmosphere/legoesm/atmosphere/physics/radiation/config.py" in /glade/u/home/pg2328/legoesm
 succeeded in 0ms:
packages/tools/legoesm/forcing/surface_utils.py:475:    """Distribute column aerosol optical depth to layers by pressure thickness.
packages/tools/legoesm/forcing/surface_utils.py:494:def place_stratospheric_aod_profile_to_layers(
packages/tools/legoesm/forcing/surface_utils.py:501:    Volcanic stratospheric aerosol is supplied as a per-layer absorption
packages/tools/legoesm/forcing/surface_utils.py:504:    so the aerosol lands at its true stratospheric pressure -- unlike
packages/tools/legoesm/forcing/surface_utils.py:506:    by full-column pressure mass and thus dumps ~90 % of a stratospheric layer
packages/tools/legoesm/forcing/surface_utils.py:726:# that surface as fallback).  This WMO locator places volcanic aerosol only;
packages/tools/legoesm/forcing/surface_utils.py:770:def stratospheric_layer_weight(p_half_col: jnp.ndarray,
packages/tools/legoesm/forcing/surface_utils.py:775:    """Per-layer weight for CMIP6 stratospheric (volcanic) aerosol at the
packages/tools/legoesm/forcing/surface_utils.py:779:    aerosol).  ``p_half_col`` (ncol, nlev+1) [Pa] in either vertical order,
packages/tools/legoesm/forcing/surface_utils.py:808:def aerosol_layer_fields(acfg, day, lat_col, p_half_col, w_strat, *,
packages/tools/legoesm/forcing/surface_utils.py:810:    """Per-layer aerosol forcing for one day: ``(aerosol_od, ccn_aod,
packages/tools/legoesm/forcing/surface_utils.py:811:    aerosol_lw_od)``, as the AMIP driver feeds radiation and the aerosol-CCN
packages/tools/legoesm/forcing/surface_utils.py:814:    ``lon_col`` [rad] reads the SW and CCN aerosol at each column's longitude;
packages/tools/legoesm/forcing/surface_utils.py:817:        get_aerosol_at_time,
packages/tools/legoesm/forcing/surface_utils.py:818:        get_aerosol_lw_at_time,
packages/tools/legoesm/forcing/surface_utils.py:819:        get_aerosol_sw_volcanic_at_time,
packages/tools/legoesm/forcing/surface_utils.py:824:    aerosol_od = jnp.zeros((ncol, nlev), dtype=dtype)
packages/tools/legoesm/forcing/surface_utils.py:828:        # stratospheric pressure like the LW profile below, NOT spread
packages/tools/legoesm/forcing/surface_utils.py:830:        # column-only volcanic file (profile None) stays in the column.
packages/tools/legoesm/forcing/surface_utils.py:831:        volc_sw = get_aerosol_sw_volcanic_at_time(_acfg, day, lat_grid=lat_col)
packages/tools/legoesm/forcing/surface_utils.py:832:        _trop_cfg = _acfg._replace(volcanic_enabled=False)
packages/tools/legoesm/forcing/surface_utils.py:833:        aerosol_col = get_aerosol_at_time(
packages/tools/legoesm/forcing/surface_utils.py:837:        aerosol_od = distribute_column_aod_to_layers(
packages/tools/legoesm/forcing/surface_utils.py:838:            jnp.asarray(aerosol_col), p_half_col,
packages/tools/legoesm/forcing/surface_utils.py:841:            aerosol_od = aerosol_od + (place_stratospheric_aod_profile_to_layers(
packages/tools/legoesm/forcing/surface_utils.py:843:            ) * w_strat
packages/tools/legoesm/forcing/surface_utils.py:844:            ).astype(aerosol_od.dtype)
packages/tools/legoesm/forcing/surface_utils.py:846:        # band, volcanic excluded, from ``ccn_path`` if set.  The Andreae 2009
packages/tools/legoesm/forcing/surface_utils.py:849:        ccn_aod = jnp.asarray(get_aerosol_at_time(
packages/tools/legoesm/forcing/surface_utils.py:854:    # Volcanic stratospheric LONGWAVE aerosol (gap #9): same (ncol,
packages/tools/legoesm/forcing/surface_utils.py:855:    # nlev) shape as ``aerosol_od``, default zeros so the SegmentForcing
packages/tools/legoesm/forcing/surface_utils.py:857:    # and a run without volcanic LW aerosol is byte-identical (zeros LW
packages/tools/legoesm/forcing/surface_utils.py:858:    # od is a RRTMGP no-op).  ``get_aerosol_lw_at_time`` returns the
packages/tools/legoesm/forcing/surface_utils.py:861:    # is placed at its true stratospheric pressure by a conservative
packages/tools/legoesm/forcing/surface_utils.py:863:    # dumped ~90% of a stratospheric aerosol into the troposphere.
packages/tools/legoesm/forcing/surface_utils.py:864:    aerosol_lw_od = jnp.zeros((ncol, nlev), dtype=dtype)
packages/tools/legoesm/forcing/surface_utils.py:866:        aerosol_lw_prof = get_aerosol_lw_at_time(
packages/tools/legoesm/forcing/surface_utils.py:869:        if aerosol_lw_prof is not None:
packages/tools/legoesm/forcing/surface_utils.py:870:            prof_col, p_edges = aerosol_lw_prof
packages/tools/legoesm/forcing/surface_utils.py:871:            aerosol_lw_od = (place_stratospheric_aod_profile_to_layers(
packages/tools/legoesm/forcing/surface_utils.py:873:            ) * w_strat
packages/tools/legoesm/forcing/surface_utils.py:875:    return aerosol_od, ccn_aod, aerosol_lw_od
packages/coupler/legoesm/driver/model_driver.py:3:Orchestrates grid creation, vertical coordinate, dycore, physics,
packages/coupler/legoesm/driver/model_driver.py:40:    solar_config_from_experiment,
packages/coupler/legoesm/driver/model_driver.py:70:    ozone_active: bool,
packages/coupler/legoesm/driver/model_driver.py:71:    aerosol_active: bool,
packages/coupler/legoesm/driver/model_driver.py:72:    aerosol_lw_active: bool,
packages/coupler/legoesm/driver/model_driver.py:73:    ghg_active: bool,
packages/coupler/legoesm/driver/model_driver.py:76:    """Whether the per-step external-forcing dict (o3/aerosol/ghg) must be built.
packages/coupler/legoesm/driver/model_driver.py:81:    spectral path is rrtmgp-only).  ``aerosol_lw_active`` MUST be included:
packages/coupler/legoesm/driver/model_driver.py:82:    volcanic stratospheric LW aerosol supplied alone (no ozone / SW aerosol /
packages/coupler/legoesm/driver/model_driver.py:86:        ozone_active or aerosol_active or aerosol_lw_active
packages/coupler/legoesm/driver/model_driver.py:87:        or ghg_active or experiment_active
packages/coupler/legoesm/driver/model_driver.py:975:    solar cosine, per cell -- the MPAS ``dynamic_albedo`` path for the
packages/coupler/legoesm/driver/model_driver.py:988:    from legoesm.atmosphere.physics.radiation.solar import (
packages/coupler/legoesm/driver/model_driver.py:1145:    its radiation optics receive the aerosol field.  Untouched config -> the
packages/coupler/legoesm/driver/model_driver.py:1154:    if cfg.nc_from_aerosol and lane != "spectral":
packages/coupler/legoesm/driver/model_driver.py:1156:            f"nc_from_aerosol (--aerosol-ccn) is not supported on the {lane} "
packages/coupler/legoesm/driver/model_driver.py:1157:            "lane: it runs Kessler, which has no aerosol droplet-number mode; "
packages/coupler/legoesm/driver/model_driver.py:1158:            "drop --aerosol-ccn.")
packages/coupler/legoesm/driver/model_driver.py:1159:    if cfg.nc_from_aerosol:
packages/coupler/legoesm/driver/model_driver.py:1161:            "nc_from_aerosol (--aerosol-ccn) is not wired on the spectral lane: "
packages/coupler/legoesm/driver/model_driver.py:1162:            "its microphysics and radiation optics never see the aerosol "
packages/coupler/legoesm/driver/model_driver.py:2183:        from legoesm.atmosphere.physics.radiation.solar import (
packages/coupler/legoesm/driver/model_driver.py:2188:        # The canopy sees the run's orbit and solar constant; the circular
packages/coupler/legoesm/driver/model_driver.py:2199:            solar_s0=float(config.S_0))
packages/coupler/legoesm/driver/model_driver.py:2687:        if getattr(config, "ifs_ozone_chem", False):
packages/coupler/legoesm/driver/model_driver.py:2688:            # IFS Cariolle ozone (tracer only), a passenger after the water
packages/coupler/legoesm/driver/model_driver.py:2694:                    "IFS Cariolle prognostic ozone"),))
packages/coupler/legoesm/driver/model_driver.py:2704:        # AMIP run from a non-January ERA5 date can run the matching solar season
packages/coupler/legoesm/driver/model_driver.py:3612:                    "daily external forcing (ozone/aerosol/GHG) is active on "
packages/coupler/legoesm/driver/model_driver.py:3619:                    "days while daily external forcing (ozone/aerosol/GHG) "
packages/coupler/legoesm/driver/model_driver.py:3629:                    "external forcing (ozone/aerosol/GHG) is active on the "
packages/coupler/legoesm/driver/model_driver.py:3792:            # SST/SIC, land, ozone -- lands on the columns the model
packages/coupler/legoesm/driver/model_driver.py:4295:            # provisional ones, so every later consumer (ozone p-grid,
packages/coupler/legoesm/driver/model_driver.py:4523:        atmosphere (constant-lapse-rate troposphere + isothermal stratosphere +
packages/coupler/legoesm/driver/model_driver.py:4529:        and stratospheric floor use Earth-like defaults.
packages/coupler/legoesm/driver/model_driver.py:4697:        # troposphere + isothermal stratosphere + equator-pole gradient, BEFORE
packages/coupler/legoesm/driver/model_driver.py:5720:        ``cover_year = start_year + day/365`` selects the annual cover
packages/coupler/legoesm/driver/model_driver.py:5733:        cover_year = float(self._start_year) + day / 365.0
packages/coupler/legoesm/driver/model_driver.py:5822:        """Configure external forcing: solar, ozone, aerosol, GHG."""
packages/coupler/legoesm/driver/model_driver.py:5825:            get_solar_forcing_at_time,
packages/coupler/legoesm/driver/model_driver.py:5826:            get_ghg_at_time,
packages/coupler/legoesm/driver/model_driver.py:5830:        self._solar_config = solar_config_from_experiment(cfg)
packages/coupler/legoesm/driver/model_driver.py:5831:        self._use_solar_spectral = (cfg.solar_source == "spectral_file")
packages/coupler/legoesm/driver/model_driver.py:5834:        # ecrad_ifs reads the same input4MIPs ozone and GHG (user 2026-11-04,
packages/coupler/legoesm/driver/model_driver.py:5835:        # vote M6a); aerosols and the GHG overrides stay rrtmg/rrtmgp-only.
packages/coupler/legoesm/driver/model_driver.py:5836:        self._ozone_ext_active = (
packages/coupler/legoesm/driver/model_driver.py:5838:            and cfg.ozone_forcing == "external")
packages/coupler/legoesm/driver/model_driver.py:5839:        if (self._ozone_ext_active and cfg.radiation == "ecrad_ifs"
packages/coupler/legoesm/driver/model_driver.py:5840:                and not cfg.ozone_file):
packages/coupler/legoesm/driver/model_driver.py:5844:                "radiation='ecrad_ifs' reads the input4MIPs ozone file; "
packages/coupler/legoesm/driver/model_driver.py:5845:                "ozone_file is empty (--ozone-file).")
packages/coupler/legoesm/driver/model_driver.py:5846:        self._ozone_ext_config = OzoneConfig(
packages/coupler/legoesm/driver/model_driver.py:5847:            enabled=self._ozone_ext_active,
packages/coupler/legoesm/driver/model_driver.py:5848:            source="climatology", path=cfg.ozone_file,
packages/coupler/legoesm/driver/model_driver.py:5850:            # Required for the non-cyclic branch in ``get_ozone_at_time``
packages/coupler/legoesm/driver/model_driver.py:5851:            # so that CMIP6 multi-year ozone files (>12 months, e.g. the
packages/coupler/legoesm/driver/model_driver.py:5860:        self._aerosol_active = (cfg.radiation in ("rrtmg", "rrtmgp")
packages/coupler/legoesm/driver/model_driver.py:5861:                                and cfg.aerosol_forcing == "external")
packages/coupler/legoesm/driver/model_driver.py:5862:        # Volcanic stratospheric LONGWAVE aerosol (gap #9): active only on a
packages/coupler/legoesm/driver/model_driver.py:5863:        # gas-radiation scheme (gray ignores aerosol) with the LW switch and
packages/coupler/legoesm/driver/model_driver.py:5864:        # a volcanic file present.  Default OFF ⇒ no LW aerosol path.
packages/coupler/legoesm/driver/model_driver.py:5865:        self._aerosol_lw_active = (
packages/coupler/legoesm/driver/model_driver.py:5867:            and bool(cfg.volcanic_aerosol_lw)
packages/coupler/legoesm/driver/model_driver.py:5868:            and bool(cfg.volcanic_aerosol_file)
packages/coupler/legoesm/driver/model_driver.py:5870:        self._aerosol_lw_od = None
packages/coupler/legoesm/driver/model_driver.py:5872:        self._aerosol_ccn_aod = None
packages/coupler/legoesm/driver/model_driver.py:5873:        self._aerosol_config = AerosolConfig(
packages/coupler/legoesm/driver/model_driver.py:5874:            enabled=self._aerosol_active,
packages/coupler/legoesm/driver/model_driver.py:5875:            source="climatology", path=cfg.aerosol_file,
packages/coupler/legoesm/driver/model_driver.py:5876:            ccn_path=cfg.aerosol_ccn_file,
packages/coupler/legoesm/driver/model_driver.py:5878:            reference_aod_550=cfg.aerosol_reference_aod,
packages/coupler/legoesm/driver/model_driver.py:5879:            volcanic_enabled=bool(cfg.volcanic_aerosol_file),
packages/coupler/legoesm/driver/model_driver.py:5880:            volcanic_path=cfg.volcanic_aerosol_file,
packages/coupler/legoesm/driver/model_driver.py:5881:            volcanic_scale=cfg.volcanic_aerosol_scale,
packages/coupler/legoesm/driver/model_driver.py:5882:            # Volcanic stratospheric LONGWAVE aerosol (gap #9): only when
packages/coupler/legoesm/driver/model_driver.py:5883:            # the LW switch is set AND a volcanic file is present.  Default
packages/coupler/legoesm/driver/model_driver.py:5884:            # OFF ⇒ ``get_aerosol_lw_at_time`` returns None ⇒ zeros LW od
packages/coupler/legoesm/driver/model_driver.py:5886:            volcanic_lw_enabled=(bool(cfg.volcanic_aerosol_lw)
packages/coupler/legoesm/driver/model_driver.py:5887:                                 and bool(cfg.volcanic_aerosol_file)),
packages/coupler/legoesm/driver/model_driver.py:5889:            # ``get_aerosol_at_time``.  Multi-year volcanic time-series
packages/coupler/legoesm/driver/model_driver.py:5898:        solar_init = get_solar_forcing_at_time(self._solar_config, cfg.start_day)
packages/coupler/legoesm/driver/model_driver.py:5899:        if self._use_solar_spectral:
packages/coupler/legoesm/driver/model_driver.py:5900:            self._solar_weights_template = jnp.asarray(
packages/coupler/legoesm/driver/model_driver.py:5901:                solar_init["solar_fraction_by_gpt"]
packages/coupler/legoesm/driver/model_driver.py:5904:            self._solar_weights_template = jnp.array([], dtype=jnp.float32)
packages/coupler/legoesm/driver/model_driver.py:5910:            from legoesm.forcing.experiments import ghg_at_year
packages/coupler/legoesm/driver/model_driver.py:5911:            co2, ch4, n2o = ghg_at_year(self._experiment, self._start_year)
packages/coupler/legoesm/driver/model_driver.py:5918:        self._ghg_active = (cfg.radiation in ("rrtmg", "rrtmgp", "ecrad_ifs")
packages/coupler/legoesm/driver/model_driver.py:5919:                            and cfg.ghg_forcing == "external")
packages/coupler/legoesm/driver/model_driver.py:5920:        if self._ghg_active:
packages/coupler/legoesm/driver/model_driver.py:5921:            self._ghg_config = GHGConfig(
packages/coupler/legoesm/driver/model_driver.py:5926:                path=cfg.ghg_file,
packages/coupler/legoesm/driver/model_driver.py:5930:            ghg_init = get_ghg_at_time(self._ghg_config, cfg.start_day)
packages/coupler/legoesm/driver/model_driver.py:5932:                f"  GHG external: CO2={ghg_init['co2_ppmv']:.1f}ppmv, "
packages/coupler/legoesm/driver/model_driver.py:5933:                f"CH4={ghg_init['ch4_ppbv']:.0f}ppbv, "
packages/coupler/legoesm/driver/model_driver.py:5934:                f"N2O={ghg_init['n2o_ppbv']:.1f}ppbv, "
packages/coupler/legoesm/driver/model_driver.py:5935:                f"CFC11={ghg_init['cfc11_pptv']:.0f}pptv, "
packages/coupler/legoesm/driver/model_driver.py:5936:                f"CFC12={ghg_init['cfc12_pptv']:.0f}pptv"
packages/coupler/legoesm/driver/model_driver.py:5939:            self._ghg_config = GHGConfig(
packages/coupler/legoesm/driver/model_driver.py:5956:            # (``get_ozone_at_time`` & co) are host/NumPy interpolators —
packages/coupler/legoesm/driver/model_driver.py:5985:        """Start-up constants of CAM6 prescribed-aerosol droplet activation
packages/coupler/legoesm/driver/model_driver.py:5987:        horizontal weights of the aerosol climatology for the physics columns.
packages/coupler/legoesm/driver/model_driver.py:5989:        from legoesm.atmosphere.physics.microphysics.cam6_aerosol_climatology import (
packages/coupler/legoesm/driver/model_driver.py:5994:        from legoesm.atmosphere.physics.microphysics.cam6_modal_aerosol import (
packages/coupler/legoesm/driver/model_driver.py:6001:        data_dir = self.config.cam6_aerosol_dir
packages/coupler/legoesm/driver/model_driver.py:6031:    def _ecrad_aerosol_forcing(self, day):
packages/coupler/legoesm/driver/model_driver.py:6032:        """ecrad_ifs aerosols (user 2026-11-04 item 3, votes C2-C8): the IFS
packages/coupler/legoesm/driver/model_driver.py:6040:        365.25-day year (the IFS's leap-day wobble, kept for parity).  Returns
packages/coupler/legoesm/driver/model_driver.py:6042:        from legoesm.atmosphere.physics.radiation.ecrad_ifs import ifs_aerosol
packages/coupler/legoesm/driver/model_driver.py:6046:            self._ecrad_cams = ifs_aerosol.load_cams_2d(cfg.ecrad_ifs_aerosol_file)
packages/coupler/legoesm/driver/model_driver.py:6056:        cm, cum = ifs_aerosol.climatology_at_columns(
packages/coupler/legoesm/driver/model_driver.py:6058:        return {"ecrad_aerosol_column_mass": jnp.asarray(cm),
packages/coupler/legoesm/driver/model_driver.py:6059:                "ecrad_aerosol_cum_fraction": jnp.asarray(cum)}
packages/coupler/legoesm/driver/model_driver.py:6062:        """Pre-compute ozone/aerosol/GHG fields outside JIT boundary.
packages/coupler/legoesm/driver/model_driver.py:6066:        from legoesm.forcing.external import get_ozone_at_time
packages/coupler/legoesm/driver/model_driver.py:6068:            aerosol_layer_fields,
packages/coupler/legoesm/driver/model_driver.py:6069:            stratospheric_layer_weight,
packages/coupler/legoesm/driver/model_driver.py:6083:        # is the overhead column-mean VMR for the RRTMGP overhead layer
packages/coupler/legoesm/driver/model_driver.py:6084:        # (RRTMGPConfig.overhead_layer; unused when that is off).  The inline sources other than
packages/coupler/legoesm/driver/model_driver.py:6092:        # Volcanic aerosol is cut at each column's WMO lapse-rate tropopause
packages/coupler/legoesm/driver/model_driver.py:6096:        # tropopause.  Only when volcanic aerosol can be on (config-uniform,
packages/coupler/legoesm/driver/model_driver.py:6098:        w_strat = None
packages/coupler/legoesm/driver/model_driver.py:6099:        if self._aerosol_active or self._aerosol_lw_active:
packages/coupler/legoesm/driver/model_driver.py:6105:            w_strat = stratospheric_layer_weight(
packages/coupler/legoesm/driver/model_driver.py:6108:        if self._ozone_ext_active:
packages/coupler/legoesm/driver/model_driver.py:6109:            o3_vmr = jnp.asarray(get_ozone_at_time(
packages/coupler/legoesm/driver/model_driver.py:6110:                self._ozone_ext_config, day,
packages/coupler/legoesm/driver/model_driver.py:6114:            # External ozone forcing inactive — build the inline profile
packages/coupler/legoesm/driver/model_driver.py:6118:            # 1e-10 and effectively disabled stratospheric ozone heating
packages/coupler/legoesm/driver/model_driver.py:6121:            # Honor ``cfg.ozone_source`` (2026-07-21 audit): this precomputed
packages/coupler/legoesm/driver/model_driver.py:6124:            # ``_compute_ozone_vmr`` — so building only the "standard"
packages/coupler/legoesm/driver/model_driver.py:6125:            # profile here silently ignored ``--ozone-source analytical``/
packages/coupler/legoesm/driver/model_driver.py:6130:                compute_ozone_vmr as _compute_ozone_vmr,
packages/coupler/legoesm/driver/model_driver.py:6135:            if self.config.ozone_source == "mls":
packages/coupler/legoesm/driver/model_driver.py:6137:                # model interfaces; the overhead layer takes gSAM's own extra
packages/coupler/legoesm/driver/model_driver.py:6139:                # With p_top = 0 there is no overhead layer: it keeps the top
packages/coupler/legoesm/driver/model_driver.py:6142:                from legoesm.atmosphere.physics.radiation.ozone_mls import (
packages/coupler/legoesm/driver/model_driver.py:6143:                    mls_ozone_layer_vmr,
packages/atmosphere/legoesm/atmosphere/physics/radiation/config.py:31:    from legoesm.atmosphere.physics.radiation.solar import OrbitalParameters
packages/atmosphere/legoesm/atmosphere/physics/radiation/config.py:71:        "obliquity": {"units": "degree", "bounds": (0.0, 90.0), "tunable_tier": 0, "transform": "sigmoid", "category": "solar", "reference": "gray radiation scheme default", "shape": None},
packages/atmosphere/legoesm/atmosphere/physics/radiation/config.py:95:            "aerosol_ssa": "RRTMGP instance-cache key (Python-hashed); not trainable until consumed as a traced input",
packages/atmosphere/legoesm/atmosphere/physics/radiation/config.py:96:            "aerosol_g": "RRTMGP instance-cache key (Python-hashed); not trainable until consumed as a traced input",
packages/atmosphere/legoesm/atmosphere/physics/radiation/config.py:147:        ``solar_exponent`` default 4.0). Controls how SW absorption is
packages/atmosphere/legoesm/atmosphere/physics/radiation/config.py:151:        Total solar irradiance [W/m^2] (default constants.S_0 = 1361.0).
packages/atmosphere/legoesm/atmosphere/physics/radiation/config.py:200:        Total solar irradiance [W/m^2] (default constants.S_0 = 1361.0).
packages/atmosphere/legoesm/atmosphere/physics/radiation/config.py:201:    aerosol_ssa : float
packages/atmosphere/legoesm/atmosphere/physics/radiation/config.py:204:    aerosol_g : float
packages/atmosphere/legoesm/atmosphere/physics/radiation/config.py:207:    aerosol_ssa_bands : tuple[float, ...] | None
packages/atmosphere/legoesm/atmosphere/physics/radiation/config.py:211:        belongs to instead of the scalar ``aerosol_ssa`` — real aerosols scatter
packages/atmosphere/legoesm/atmosphere/physics/radiation/config.py:213:        grey aerosol at the scalar ``aerosol_ssa`` (byte-identical).
packages/atmosphere/legoesm/atmosphere/physics/radiation/config.py:214:    aerosol_g_bands : tuple[float, ...] | None
packages/atmosphere/legoesm/atmosphere/physics/radiation/config.py:216:        semantics as ``aerosol_ssa_bands``). ``None`` (default) => grey aerosol
packages/atmosphere/legoesm/atmosphere/physics/radiation/config.py:217:        at the scalar ``aerosol_g``.
packages/atmosphere/legoesm/atmosphere/physics/radiation/config.py:258:    aerosol_ssa: float = 0.93
packages/atmosphere/legoesm/atmosphere/physics/radiation/config.py:259:    aerosol_g: float = 0.70
packages/atmosphere/legoesm/atmosphere/physics/radiation/config.py:313:    # the shipped table).  None => grey aerosol at the scalar aerosol_ssa/g
packages/atmosphere/legoesm/atmosphere/physics/radiation/config.py:315:    # APPENDED at the end of the field list (not next to aerosol_ssa/g) so no
packages/atmosphere/legoesm/atmosphere/physics/radiation/config.py:317:    aerosol_ssa_bands: tuple[float, ...] | None = None
packages/atmosphere/legoesm/atmosphere/physics/radiation/config.py:318:    aerosol_g_bands: tuple[float, ...] | None = None
packages/atmosphere/legoesm/atmosphere/physics/radiation/config.py:399:        If True, compute instantaneous insolation from the solar zenith
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:48:        "q_v": "kg/kg", "cos_zenith": "1 (cos solar zenith angle)",
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:316:    ``config.aerosol_ssa_bands`` / ``aerosol_g_bands`` are set.  The per-band
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:319:    ssa = config.aerosol_ssa
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:320:    g = config.aerosol_g
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:321:    bands_ssa = getattr(config, "aerosol_ssa_bands", None)
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:322:    bands_g = getattr(config, "aerosol_g_bands", None)
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:330:                f"RRTMGPConfig.aerosol_ssa_bands has length {len(bands_ssa)}, "
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:336:                f"RRTMGPConfig.aerosol_g_bands has length {len(bands_g)}, "
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:394:      - ``S_0``, ``aerosol_ssa``, ``aerosol_g`` (iter-32 audit): these
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:398:        ``aerosol_ssa = 0.95`` would silently reuse a solver instance
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:399:        built with ``aerosol_ssa = 0.93`` and apply the stale value.
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:454:          config.aerosol_ssa,
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:455:          config.aerosol_g,
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:458:          getattr(config, "aerosol_ssa_bands", None),
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:459:          getattr(config, "aerosol_g_bands", None),
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:504:              "ccl4": 7.5e-11,
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:507:              "cfc22": 2.4e-10,
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:508:              "cf4": 8.5e-11,
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:606:      solar_spectral_fraction: jnp.ndarray | None = None,
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:645:          Cosine of solar zenith angle (ncol,).
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:679:      solar_spectral_fraction : jnp.ndarray | None
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:680:          Per-g-point solar source weights (ngpt_sw,).
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:820:      if solar_spectral_fraction is not None:
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:821:          solar_spectral_fraction = jnp.asarray(
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:822:              solar_spectral_fraction).astype(_table_dtype)
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:1115:      # or per-shortwave-band arrays (config.aerosol_ssa_bands/aerosol_g_bands).
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:1129:      # Optional spectral solar forcing
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:1130:      if solar_spectral_fraction is not None:
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:1131:          solar_weights = jnp.clip(jnp.asarray(solar_spectral_fraction), 0.0, None)
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:1132:          denom = jnp.maximum(jnp.sum(solar_weights), 1.0e-30)
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:1133:          solar_weights = solar_weights / denom
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:1134:          if solar_weights.shape[0] != optics_lib.n_gpt_sw:
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:1136:                  "solar_spectral_fraction has wrong length: "
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:1137:                  f"{solar_weights.shape[0]} (expected {optics_lib.n_gpt_sw})",
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:1140:          solar_weights = None
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:1181:          weights = solar_weights
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:1183:              weights = optics_lib.solar_fraction_by_gpt
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:1184:          # Beam-normal solar flux per g-point [W/m^2].
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:1185:          solar_flux_normal = config.S_0 * weights
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:1187:                  solar_flux_normal)
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:1260:          solar_fraction_by_gpt=solar_weights,
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:1357:          ``solar_spectral_fraction`` whose leading axis is ``n_gpt`` not
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:1442:      # ``solar_spectral_fraction`` is a per-g-point vector (leading axis
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:1448:      _non_column_opt = ("solar_spectral_fraction",)

exec
/usr/bin/bash -lc "sed -n '221,440p' CLAUDE.md" in /glade/u/home/pg2328/legoesm
 succeeded in 0ms:
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

exec
/usr/bin/bash -lc "nl -ba packages/tools/legoesm/forcing/surface_utils.py | sed -n '390,545p'; nl -ba packages/coupler/legoesm/driver/model_driver.py | sed -n '6060,6225p'; rg -n 'aerosol|activation|co2|overhead' config/amip/amip_production.yaml" in /glade/u/home/pg2328/legoesm
 succeeded in 0ms:
   390	) -> "jnp.ndarray | None":
   391	    """The snow field a coupled driver may pass to ``static_surface_albedo``.
   392	
   393	    Shortwave twin of the emissivity selector above, and the single place both
   394	    coupled drivers make this call, so they cannot drift apart.
   395	
   396	    The driver's ``_carry_aux["snow"]`` is the raw segment carry.  On an
   397	    ENSEMBLE run (``ensemble_size > 1``) the driver unpacks
   398	    ``ensemble_mean(carry)`` for the state and the held fluxes but stashes the
   399	    snow carry member-shaped — deliberately, since the next segment restarts
   400	    from the full ensemble.  Handing that member-shaped field to the albedo
   401	    blend would make ``albedo`` member-shaped too and broadcast the 2-D mean
   402	    ``sw_net_sfc`` up to 3-D, mis-shaping every downstream surface forcing
   403	    field.  So ensembles fall back to the bare vegetation albedo, matching the
   404	    existing gate that disables the multilayer land tile for ensembles.
   405	
   406	    Scope of that fallback, stated narrowly: ensembles lose only the SNOW
   407	    brightening.  They still gain the ocean/ice/land blend itself, which is the
   408	    first-order fix and the whole point — an ensemble run is NOT byte-identical
   409	    to the pre-fix behaviour, it is corrected without the snow term.
   410	
   411	    ``<= 1`` rather than ``== 1``, matching ``ModelDriver._create_ensemble``,
   412	    which treats any size at or below one as serial and leaves the carry
   413	    single-member shaped.
   414	
   415	    Parameters
   416	    ----------
   417	    snow : array or None
   418	        Snow water equivalent [kg/m^2] from the driver's carry, or ``None``.
   419	    ensemble_size : int
   420	        The atmosphere driver's ``_ensemble_size``.
   421	
   422	    Returns
   423	    -------
   424	    snow : array or None
   425	        ``snow`` on a single-member run; ``None`` otherwise.
   426	    """
   427	    if snow is None or int(ensemble_size) > 1:
   428	        return None
   429	    return snow
   430	
   431	
   432	def snow_fraction(
   433	    T_low: jnp.ndarray,
   434	    T_freeze: float,
   435	    transition_center_offset_K: float = 2.0,
   436	    transition_width_K: float = 4.0,
   437	) -> jnp.ndarray:
   438	    """Smooth rain/snow partition fraction (Wigmosta 1994 / Dai 2008).
   439	
   440	    Linear ramp from 0 (all rain) at
   441	    ``T_low = T_freeze + transition_center_offset_K`` down to 1 (all snow)
   442	    at ``T_low = T_freeze + transition_center_offset_K - transition_width_K``.
   443	    Replaces the hard step ``where(T_low < T_freeze, 1, 0)``, which zeroes
   444	    ``d(snow)/d(T_low)`` on training/DA paths and miscounts mixed-phase
   445	    precipitation in the 0–4 °C band. Single source of truth shared by the
   446	    coupled and earth-system drivers so the two cannot silently diverge.
   447	
   448	    Parameters
   449	    ----------
   450	    T_low : array
   451	        Lowest-model-level air temperature [K].
   452	    T_freeze : float
   453	        Freezing point [K] (pass ``constants.T_freeze``).
   454	    transition_center_offset_K : float
   455	        Upper edge of the mixed-phase band above freezing [K] (default 2.0).
   456	    transition_width_K : float
   457	        Total width of the linear ramp [K] (default 4.0).
   458	
   459	    Returns
   460	    -------
   461	    snow_frac : array
   462	        Snow fraction in [0, 1].
   463	    """
   464	    return jnp.clip(
   465	        (T_freeze + transition_center_offset_K - T_low) / transition_width_K,
   466	        0.0,
   467	        1.0,
   468	    )
   469	
   470	
   471	def distribute_column_aod_to_layers(
   472	    aod_col: jnp.ndarray,
   473	    p_half_col: jnp.ndarray,
   474	) -> jnp.ndarray:
   475	    """Distribute column aerosol optical depth to layers by pressure thickness.
   476	
   477	    Parameters
   478	    ----------
   479	    aod_col : array, shape (ncol,)
   480	        Total column AOD per column.
   481	    p_half_col : array, shape (ncol, nlev+1)
   482	        Half-level pressures.
   483	
   484	    Returns
   485	    -------
   486	    aod_layers : array, shape (ncol, nlev)
   487	        Layer-distributed AOD.
   488	    """
   489	    dp = jnp.clip(p_half_col[:, 1:] - p_half_col[:, :-1], 1.0e-12, None)
   490	    w = dp / jnp.sum(dp, axis=1, keepdims=True)
   491	    return jnp.clip(aod_col, 0.0, None)[:, None] * w
   492	
   493	
   494	def place_stratospheric_aod_profile_to_layers(
   495	    aod_profile: jnp.ndarray,
   496	    p_edges: jnp.ndarray,
   497	    p_half_col: jnp.ndarray,
   498	) -> jnp.ndarray:
   499	    """Conservatively remap a fixed-pressure-edge AOD profile onto model layers.
   500	
   501	    Volcanic stratospheric aerosol is supplied as a per-layer absorption
   502	    optical depth on the forcing file's OWN (altitude -> pressure) grid.  This
   503	    bins it onto the model's pressure layers by fractional pressure overlap,
   504	    so the aerosol lands at its true stratospheric pressure -- unlike
   505	    :func:`distribute_column_aod_to_layers`, which spreads a single column AOD
   506	    by full-column pressure mass and thus dumps ~90 % of a stratospheric layer
   507	    into the troposphere.
   508	
   509	    Method: the cumulative-OD curve is sampled at each model half level and the
   510	    per-layer OD is the difference of the cumulative at the layer's two edges.
   511	    The cumulative is interpolated in LOG-pressure, because the source OD is
   512	    ``ext*dz`` -- uniform in geometric height (~ log-pressure) within a source
   513	    layer, NOT uniform in pressure; a linear-pressure CDF would misallocate ~1 %
   514	    of a 0.5 km source layer's OD across a cutting model interface (Codex review
   515	    iter-1).  The remap is exact at source-layer edges and conserves the total
   516	    column OD that falls within the model's pressure range (OD above the model
   517	    top or below the surface edge is dropped -- it cannot be represented).
   518	
   519	    The op set (searchsorted / gather / clip / min / max) is piecewise-linear:
   520	    a.e.-differentiable in both ``aod_profile`` and ``p_half_col`` with
   521	    well-defined one-sided gradients (kinks at source/model edges), and
   522	    JIT/vmap-safe.
   523	
   524	    Sign/units: optical depth is >= 0 and additive; pressure [Pa].
   525	
   526	    Parameters
   527	    ----------
   528	    aod_profile : array, shape (ncol, nsrc)
   529	        Per-source-layer absorption optical depth [-] (>= 0).
   530	    p_edges : array, shape (nsrc+1,)
   531	        Source-layer pressure edges [Pa], ASCENDING (index 0 = top / lowest
   532	        pressure), aligned so ``aod_profile[:, j]`` occupies
   533	        ``[p_edges[j], p_edges[j+1]]``.
   534	    p_half_col : array, shape (ncol, nlev+1)
   535	        Model half-level pressures [Pa] (index 0 = top).
   536	
   537	    Returns
   538	    -------
   539	    aod_layers : array, shape (ncol, nlev)
   540	        Layer absorption optical depth [-].
   541	    """
   542	    pe = jnp.asarray(p_edges)                                   # (nsrc+1,)
   543	    aod = jnp.clip(jnp.asarray(aod_profile), 0.0, None)         # (ncol, nsrc)
   544	    # Cumulative OD from the top: cum[:, k] = OD in source layers 0..k-1, so
   545	    # cum aligns to the nsrc+1 source edges ``pe``.
  6060	
  6061	    def _precompute_external_forcing(self, day, p_s, lat, temp=None, lon=None):
  6062	        """Pre-compute ozone/aerosol/GHG fields outside JIT boundary.
  6063	
  6064	        ``lon`` [rad] matches ``lat``; None -> :meth:`_owned_lon` (the layout
  6065	        of :meth:`_owned_p_s_and_lat`, which every caller but spectral uses)."""
  6066	        from legoesm.forcing.external import get_ozone_at_time
  6067	        from legoesm.forcing.surface_utils import (
  6068	            aerosol_layer_fields,
  6069	            stratospheric_layer_weight,
  6070	        )
  6071	
  6072	        nlev = self.sigma.sigma_full.shape[0]
  6073	        shape_2d = p_s.shape
  6074	        ncol = int(np.prod(np.array(shape_2d)))
  6075	
  6076	        p_full = self.sigma.pressure_at_full(p_s)
  6077	        p_half = self.sigma.pressure_at_half(p_s)
  6078	        p_full_col = p_full.reshape(ncol, nlev)
  6079	        p_half_col = p_half.reshape(ncol, nlev + 1)
  6080	        lat_col = lat.reshape(ncol)
  6081	        # Ozone is evaluated on the model levels PLUS 16 midpoints of equal
  6082	        # pressure sub-layers above the model top (0 < p < p_top); their mean
  6083	        # is the overhead column-mean VMR for the RRTMGP overhead layer
  6084	        # (RRTMGPConfig.overhead_layer; unused when that is off).  The inline sources other than
  6085	        # "mls" are pointwise in p, so the model-level values are unchanged;
  6086	        # "mls" (gSAM layer means) is handled from the interfaces below.
  6087	        _p_o3 = jnp.concatenate([
  6088	            p_full_col,
  6089	            p_half_col[:, :1] * ((jnp.arange(16) + 0.5) / 16.0),
  6090	        ], axis=1)
  6091	
  6092	        # Volcanic aerosol is cut at each column's WMO lapse-rate tropopause
  6093	        # diagnosed from T at this precompute (the same state snapshot as
  6094	        # ``p_s``; callers refresh daily).  ``temp`` defaults to the state's
  6095	        # grid-point T; absent or off the ``p_s`` grid -> climatological
  6096	        # tropopause.  Only when volcanic aerosol can be on (config-uniform,
  6097	        # so the SPMD gather in ``_owned_temp`` stays collective).
  6098	        w_strat = None
  6099	        if self._aerosol_active or self._aerosol_lw_active:
  6100	            if temp is None:
  6101	                temp = self._owned_temp()
  6102	            temp_col = None
  6103	            if temp is not None and tuple(temp.shape) == tuple(p_full.shape):
  6104	                temp_col = jnp.asarray(temp).reshape(ncol, nlev)
  6105	            w_strat = stratospheric_layer_weight(
  6106	                p_half_col, lat_col, p_full_col, temp_col)
  6107	
  6108	        if self._ozone_ext_active:
  6109	            o3_vmr = jnp.asarray(get_ozone_at_time(
  6110	                self._ozone_ext_config, day,
  6111	                lat_grid=lat_col, p_grid=_p_o3,
  6112	            ))
  6113	        else:
  6114	            # External ozone forcing inactive — build the inline profile
  6115	            # HERE so the radiation JIT signature stays stable (it always
  6116	            # sees a real array).  Previously ``o3_vmr`` was initialised to
  6117	            # zeros and threaded into the solver, which then clipped it to
  6118	            # 1e-10 and effectively disabled stratospheric ozone heating
  6119	            # (audit 2026-05-12 HIGH #3).
  6120	            #
  6121	            # Honor ``cfg.ozone_source`` (2026-07-21 audit): this precomputed
  6122	            # array reaches the compiled radiation as ``o3_vmr_override``,
  6123	            # which takes precedence over the backend's own
  6124	            # ``_compute_ozone_vmr`` — so building only the "standard"
  6125	            # profile here silently ignored ``--ozone-source analytical``/
  6126	            # ``none`` on the cd-grid/latlon pipeline while MPAS/spectral
  6127	            # honored them (grid-dependent physics from the same config).
  6128	            # Reuse the backend's dispatcher for exact parity.
  6129	            from legoesm.atmosphere.physics.radiation.integration import (
  6130	                compute_ozone_vmr as _compute_ozone_vmr,
  6131	            )
  6132	            from legoesm.atmosphere.physics.radiation.config import (
  6133	                OzoneProfileConfig,
  6134	            )
  6135	            if self.config.ozone_source == "mls":
  6136	                # gSAM layer means are not pointwise in p: evaluate them on the
  6137	                # model interfaces; the overhead layer takes gSAM's own extra
  6138	                # layer above the model top (p_top to min(1e-4 hPa, p_top/8)).
  6139	                # With p_top = 0 there is no overhead layer: it keeps the top
  6140	                # model layer's value, and the extra layer's 0/0 is never formed
  6141	                # (p_top replaced by 1 Pa only inside that branch).
  6142	                from legoesm.atmosphere.physics.radiation.ozone_mls import (
  6143	                    mls_ozone_layer_vmr,
  6144	                )
  6145	                _lay = _compute_ozone_vmr(
  6146	                    p_full_col, lat_col, OzoneProfileConfig(source="mls"),
  6147	                    p_half=p_half_col,
  6148	                )
  6149	                _ptop = p_half_col[:, 0]
  6150	                _ph_safe = p_half_col.at[:, 0].set(jnp.where(_ptop > 0.0, _ptop, 1.0))
  6151	                _above = jnp.where(_ptop > 0.0, mls_ozone_layer_vmr(_ph_safe)[1],
  6152	                                   _lay[:, 0])
  6153	                _o3_inline = jnp.concatenate(
  6154	                    [_lay, jnp.broadcast_to(_above[:, None], (ncol, _p_o3.shape[1] - nlev))],
  6155	                    axis=1)
  6156	            else:
  6157	                _o3_inline = _compute_ozone_vmr(
  6158	                    _p_o3, lat_col,
  6159	                    OzoneProfileConfig(source=self.config.ozone_source),
  6160	                )
  6161	            if _o3_inline is None:
  6162	                # source="standard": the backend contract is o3_vmr=None →
  6163	                # its built-in US Std Atm 1976 fit; materialize the same
  6164	                # profile for the stable-signature override path.
  6165	                from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import (
  6166	                    standard_o3_profile,
  6167	                )
  6168	                _o3_inline = standard_o3_profile(_p_o3)
  6169	            o3_vmr = jnp.asarray(_o3_inline).astype(p_s.dtype)
  6170	        self._o3_top_vmr = jnp.mean(o3_vmr[:, nlev:], axis=1)
  6171	        o3_vmr = o3_vmr[:, :nlev]
  6172	
  6173	        # Radiation and the droplet proxy read the aerosol at each column's
  6174	        # own longitude, as CAM's prescribed aerosol reaches radiation
  6175	        # (tracer_data.F90 bilinear lon/lat lininterp).
  6176	        lon_col = (jnp.asarray(self._owned_lon() if lon is None else lon)
  6177	                   .reshape(ncol) if self._aerosol_active else None)
  6178	        aerosol_od, _ccn_aod, self._aerosol_lw_od = aerosol_layer_fields(
  6179	            self._aerosol_config, day, lat_col, p_half_col, w_strat,
  6180	            active=self._aerosol_active, lw_active=self._aerosol_lw_active,
  6181	            dtype=p_s.dtype, lon_col=lon_col)
  6182	        if _ccn_aod is not None:
  6183	            self._aerosol_ccn_aod = _ccn_aod
  6184	
  6185	        ghg_vmr = self.radiation_ghg_vmr(day)
  6186	        return o3_vmr, aerosol_od, ghg_vmr
  6187	
  6188	    def radiation_ghg_vmr(self, day):
  6189	        """GHG mole fractions radiation uses on ``day`` (None = fixed config gases)."""
  6190	        from legoesm.forcing.external import (
  6191	            get_ghg_at_time, ghg_concentrations_to_vmr,
  6192	        )
  6193	        # GHG VMR override (None for gray radiation / constant forcing)
  6194	        ghg_vmr = None
  6195	        if self._ghg_active:
  6196	            ghg_conc = get_ghg_at_time(self._ghg_config, day)
  6197	            ghg_vmr = ghg_concentrations_to_vmr(ghg_conc)
  6198	
  6199	        # CMIP experiment transient GHG override — applies even when
  6200	        # ghg_forcing is "constant" so that built-in experiment
  6201	        # trajectories (historical, SSP, 1pctCO2) drive radiation.
  6202	        cfg = self.config
  6203	        if (self._experiment
  6204	                and cfg.radiation in ("rrtmg", "rrtmgp")
  6205	                and ghg_vmr is None):
  6206	            from legoesm.forcing.experiments import (
  6207	                ghg_at_year, EXPERIMENT_TEMPLATES,
  6208	            )
  6209	            tmpl = EXPERIMENT_TEMPLATES.get(self._experiment)
  6210	            if tmpl is not None and tmpl.forcing_type == "transient":
  6211	                current_year = self._start_year + day / 365.0
  6212	                co2, ch4, n2o = ghg_at_year(self._experiment, current_year)
  6213	                ghg_vmr = ghg_concentrations_to_vmr({
  6214	                    "co2_ppmv": co2,
  6215	                    "ch4_ppbv": ch4,
  6216	                    "n2o_ppbv": n2o,
  6217	                })
  6218	
  6219	        # Interactive carbon-radiation coupling (#3 / C4MIP): when the coupled
  6220	        # driver runs a prognostic CO2 tracer it sets ``self._co2_vmr_override``
  6221	        # (a global-mean CO2 mole fraction) each segment; inject it into the
  6222	        # radiation GHG so emitted / absorbed CO2 actually changes radiative
  6223	        # forcing.  Gated on a gas-radiation scheme (gray ignores GHG) and on the
  6224	        # override being present, so fixed-CO2 runs are byte-identical.
  6225	        co2_vmr_override = getattr(self, "_co2_vmr_override", None)
10:# SST/SIC, solar/ozone/GHG/aerosol/volcanic, topography) are deliberately NOT
46:# aerosols (licence decision), CLUBB tuning table.
140:rrtmgp_overhead_layer: true # CAM RRTMG extra layer above the model top (ozone
281:aerosol_ccn: true                  # droplet number diagnosed from the prescribed
282:                                   # aerosol (user decision 2026-11-01, #2216,
290:                                   # cam6_aerosol_dir below is read only by it.
301:cam6_aerosol_dir: /work/bb1596/b309178/diffESM/legoesm_ap/data/cam6_optics
429:snow_age_activation_K: 5000.0
540:aerosol_forcing: external
541:aerosol_reference_aod: 0.03  # 550 nm background AOD of the built-in reference
542:                             # aerosol field, read only when no aerosol file is

exec
/usr/bin/bash -lc "rg -n 'SERIAL|LBC_|SolarForcing|ozone_strat|prescribed_strat|orb_|bndtvs|ndep|flanduse' /glade/derecho/scratch/pg2328/cesm_src/CAM/bld/namelist_files/use_cases/hist_cam6.xml; rg -n 'get_forcing_at_time|freeze|clip|interp|cyclic' packages/tools/legoesm/forcing/amip.py; rg -n -i 'impurit|snicar|black.carbon|deposition|snow.*age|bats' packages/land/legoesm | head -65" in /glade/u/home/pg2328/legoesm
 succeeded in 0ms:
6:  <solar_irrad_data_file    >   'atm/cam/solar/SolarForcingCMIP6_18491230-23000102_c20200615.nc'</solar_irrad_data_file>
14:  <prescribed_ozone_datapath>   'atm/cam/ozone_strataero'                                                    </prescribed_ozone_datapath>
15:  <prescribed_ozone_file    >   'ozone_strataero_WACCM_L70_zm5day_18500101-20150103_CMIP6ensAvg_c180923.nc'  </prescribed_ozone_file>
17:  <prescribed_ozone_type    >   'SERIAL'                                                                     </prescribed_ozone_type>
22:  <prescribed_strataero_type    >     'SERIAL'                                                                     </prescribed_strataero_type>
23:  <prescribed_strataero_datapath>     'atm/cam/ozone_strataero'                                                    </prescribed_strataero_datapath>
24:  <prescribed_strataero_file>         'ozone_strataero_WACCM_L70_zm5day_18500101-20150103_CMIP6ensAvg_c180923.nc'  </prescribed_strataero_file> 
25:  <prescribed_strataero_use_chemtrop> .true.                                                                       </prescribed_strataero_use_chemtrop>
27:  <scenario_ghg            >    'CHEM_LBC_FILE'   </scenario_ghg>
28:  <flbc_file>atm/waccm/lb/LBC_1750-2015_CMIP6_GlobAnnAvg_c180926.nc</flbc_file>
29:  <flbc_type>'SERIAL'</flbc_type>
5:and provides linear time interpolation for use during integration.
86:    # the value tracks ``constants.T_freeze_ocean`` (CLAUDE.md naming
88:    T_ice: float = constants.T_freeze_ocean
107:        interpolated value in :func:`get_forcing_at_time` (Taylor 2000).
110:        [0, 1] (PCMDI mid-month bcs convention); clipped to [0, 1] only
111:        AFTER time interpolation in :func:`get_forcing_at_time`.
147:            sst_offset=constants.T_freeze,    # Celsius -> Kelvin
207:    stays strictly increasing for the ``searchsorted`` interpolation in
208:    :func:`get_forcing_at_time`.  Monthly (bcs) axes are unaffected.
345:    low side by the freeze floor; SIC→100×, masked by the [0,1] clip).
364:    # (265-304 K) — trusting the attribute would demand +T_freeze and cook
385:            f"+{constants.T_freeze}.",
389:    # A Celsius file needs the +T_freeze offset (tight tolerance,
392:    if _sst_is_celsius and abs(config.sst_offset - constants.T_freeze) > 1.0:
396:            f"+{constants.T_freeze} K Celsius->Kelvin offset. Pass "
397:            f"--sst-offset {constants.T_freeze} (constants.T_freeze)."
403:            f"spurious +{constants.T_freeze}. Pass --sst-offset 0 for a "
410:    # unphysical for an ocean surface (the downstream T_freeze_ocean
454:    # overshoots are clipped only AFTER time interpolation, per protocol,
544:    # AFTER time interpolation (get_forcing_at_time), NOT to these mid-month
545:    # anchors — the same Taylor et al. (2000) bcs clip-after-interp
592:        # mid-month bcs anchor), else interpolation would silently clamp to
659:    from scipy.interpolate import RegularGridInterpolator
782:        # AFTER time interpolation (get_forcing_at_time), NOT to these mid-month
784:        # overshoot the physical range so that clip(linear-interp(anchors))
813:            # mid-month bcs anchor), else the interpolation would silently clamp
816:            # files — see get_forcing_at_time), not a jump back to the file era.
829:            # End-of-record hold is SILENT by design (get_forcing_at_time
848:    # SIC [0,1] clip and SST freezing floor are applied after time interpolation
849:    # in get_forcing_at_time.
873:    from scipy.interpolate import RegularGridInterpolator
878:    # Wrap longitude for interpolation continuity (pad one column each end).
902:        interp = RegularGridInterpolator(
906:        out[t] = interp(target_points).reshape(target_shape)
910:def climatology_interp_indices(times: jnp.ndarray, day: float):
911:    """Bracketing indices + weight for cyclic/transient monthly interpolation.
913:    Shared by the AMIP SST/SIC path (``get_forcing_at_time``) and any other
915:    cyclic-wrap / transient-hold logic lives in ONE place (no duplicated
941:    idx = jnp.where(wrap, jnp.clip(idx, 0, ntime - 1),
942:                    jnp.clip(idx, 0, ntime - 2))
947:    weight = jnp.clip((day - t_i) / dt, 0.0, 1.0)
951:def get_forcing_at_time(
955:    """Linearly interpolate SST and SIC to a given day.
977:    t_freeze = forcing.config.T_ice
980:    # to shape (1, ...) by load_amip_forcing): no interpolation is possible, so
983:        return (jnp.maximum(forcing.sst[0], t_freeze),
984:                jnp.clip(forcing.sic[0], 0.0, 1.0))
987:    # with the slab q-flux path (climatology_interp_indices).
988:    idx, idx_next, weight = climatology_interp_indices(times, day)
990:    # Linear interpolation
995:    # point) — on the interpolated value only. Cast back to the STORED forcing
998:    # through the interpolation weight.
999:    sst = jnp.maximum(sst, t_freeze).astype(forcing.sst.dtype)
1000:    sic = jnp.clip(sic, 0.0, 1.0).astype(forcing.sic.dtype)
packages/land/legoesm/land/clm_surface_map.py:139:_BIOPHYS_LMIP_SNOW_ALBEDO_MIN = 0.5207    # [-] aged-snow albedo
packages/land/legoesm/land/clm_surface_map.py:141:_BIOPHYS_LMIP_SNOW_TAU_S = 317424.96      # [s] snow-age decay (3.674 days)
packages/land/legoesm/land/clm_surface_map.py:202:# aged-snow albedo floor, tanh snow-cover SWE half-scale [kg/m2], snow-albedo age
packages/land/legoesm/land/clm_surface_map.py:209:TUNED_SNOW_TAU_DAYS_MULTILAYER = 11.6391   # snow-albedo age e-folding [days]   # snow-albedo age e-folding [days]
packages/land/legoesm/land/clm_surface_map.py:650:    # Calibrated SNOW albedo (fresh/aged brightness + cover threshold + age decay) baked
packages/land/legoesm/land/snow_bands.py:259:    snow_age_bands: jax.Array  # (ncol, n_bands) per-band snow age [s]
packages/land/legoesm/land/snow_bands.py:260:    snow_age: jax.Array        # (ncol,) SWE-weighted aggregate age [s] (diagnostic)
packages/land/legoesm/land/snow_bands.py:270:    snow_age_bands: jnp.ndarray,
packages/land/legoesm/land/snow_bands.py:279:    snow_age_activation_K: float,
packages/land/legoesm/land/snow_bands.py:299:    Seasonal-snow age resets on GROSS band snowfall (fresh snow always brightens the
packages/land/legoesm/land/snow_bands.py:330:    # 1. Seasonal snow accumulate + energy-limited melt.  update_snow resets the age on
packages/land/legoesm/land/snow_bands.py:333:        swe_bands, snow_age_bands, T_sfc_band, snowfall_bands, dt,
packages/land/legoesm/land/snow_bands.py:335:        snow_age_activation_K=snow_age_activation_K,
packages/land/legoesm/land/snow_bands.py:371:    # 2. Firnification: seasonal snow above the snow cap densifies into ice storage.
packages/land/legoesm/land/snow_bands.py:417:        snow_age_bands=age_new_bands,
packages/land/legoesm/land/snow_bands.py:418:        snow_age=age_agg,
packages/land/legoesm/land/snow_bands.py:442:    snow_age_bands: jnp.ndarray,
packages/land/legoesm/land/snow_bands.py:454:    mean_k [ snow_albedo(age_k) * cover(swe_k) ]``.  At uniform bands (flat cell,
packages/land/legoesm/land/snow_bands.py:457:    ``snow_cover_fn`` maps SWE -> cover fraction and ``snow_albedo_fn`` maps age ->
packages/land/legoesm/land/snow_bands.py:465:    contrib = snow_albedo_fn(snow_age_bands) * f
packages/land/legoesm/land/snow_bands.py:471:    snow_age_bands: jnp.ndarray,
packages/land/legoesm/land/snow_bands.py:481:    ``alpha_k = ground_k * (1 - cover(swe_k)) + snow_albedo(age_k) * cover(swe_k)``.
packages/land/legoesm/land/snow_bands.py:502:    return ground * (1.0 - cover) + jnp.maximum(snow_albedo_fn(snow_age_bands), ground) * cover
packages/land/legoesm/land/snow_budget.py:30:    snow_age: jnp.ndarray,
packages/land/legoesm/land/snow_budget.py:38:    snow_age_activation_K: float,
packages/land/legoesm/land/snow_budget.py:40:    """Update snow depth and age, returning the melt amount.
packages/land/legoesm/land/snow_budget.py:46:    snow_age : jnp.ndarray
packages/land/legoesm/land/snow_budget.py:47:        Current snow age [s].
packages/land/legoesm/land/snow_budget.py:68:    snow_age_activation_K : float
packages/land/legoesm/land/snow_budget.py:69:        Snow-age metamorphism activation temperature [K], from
packages/land/legoesm/land/snow_budget.py:70:        ``LandAlbedoConfig.snow_age_activation_K``; 0.0 = calendar clock.
packages/land/legoesm/land/snow_budget.py:75:    (snow_new, snow_age_new, snow_melt)
packages/land/legoesm/land/snow_budget.py:76:        Updated snow depth [kg/m2], snow age [s], and amount of snow
packages/land/legoesm/land/snow_budget.py:100:    snow_age_new = update_snow_age(
packages/land/legoesm/land/snow_budget.py:101:        snow_new, snow_age, precip_snow, dt,
packages/land/legoesm/land/snow_budget.py:102:        T_snow=T_sfc, age_activation_K=snow_age_activation_K)
packages/land/legoesm/land/snow_budget.py:104:    return snow_new, snow_age_new, snow_melt
packages/land/legoesm/land/snow_budget.py:108:    """Temperature scaling of snow grain growth, BATS/CLM form.
packages/land/legoesm/land/snow_budget.py:111:    of Dickinson et al. (1993) BATS with activation temperature ``A`` [K]
packages/land/legoesm/land/snow_budget.py:112:    (BATS uses 5000 K).  It is 1 at the freezing point and falls steeply as the
packages/land/legoesm/land/snow_budget.py:113:    snow cools: 0.08 at 240 K, 0.03 at 230 K, so cold dry polar snow ages ~12-30
packages/land/legoesm/land/snow_budget.py:114:    times more slowly than melting snow.  Bounded to (0, 1]: snow never ages
packages/land/legoesm/land/snow_budget.py:127:def update_snow_age(
packages/land/legoesm/land/snow_budget.py:129:    snow_age: jnp.ndarray,
packages/land/legoesm/land/snow_budget.py:136:    """Snow-age clock: mass-weighted grain-age mixing of fresh + existing snow.
packages/land/legoesm/land/snow_budget.py:139:    scheme (``snow_bands``), which keeps a single cell-level age for the albedo
packages/land/legoesm/land/snow_budget.py:142:    Fresh snowfall must NOT hard-reset the albedo age to zero on a trace flurry: the
packages/land/legoesm/land/snow_budget.py:145:    grain age mixes by MASS (CLM/BATS): existing snow ages by ``dt`` while fresh snow
packages/land/legoesm/land/snow_budget.py:148:        new_age = (snow_age + dt) * swe_old / (swe_old + fresh_swe)
packages/land/legoesm/land/snow_budget.py:153:    ``snow_new`` net of the fresh addition).  Trace snow on a deep aged pack barely
packages/land/legoesm/land/snow_budget.py:158:    # Units: fresh_swe, swe_old, denom all [kg/m2]; snow_age, dt [s]; ratio is
packages/land/legoesm/land/snow_budget.py:159:    # dimensionless in [0, 1], so new_age stays in [0, snow_age + dt] -- age never
packages/land/legoesm/land/snow_budget.py:166:    # (BATS/CLM).  Cold dry snow therefore stays near its fresh albedo for
packages/land/legoesm/land/snow_budget.py:171:    _rate = (1.0 if (T_snow is None or age_activation_K == 0.0)
packages/land/legoesm/land/snow_budget.py:172:             else metamorphism_rate(T_snow, age_activation_K))
packages/land/legoesm/land/snow_budget.py:173:    aged = snow_age + dt * _rate                             # existing snow ages
packages/land/legoesm/land/snow_budget.py:177:    # If all snow has melted, reset age to zero
packages/land/legoesm/land/state.py:21:    snow_age: Field        # Time since last snowfall [s]
packages/land/legoesm/land/state.py:44:    snow_age: jax.Array        # Time since last snowfall [s], (ncol,)
packages/land/legoesm/land/state.py:57:    # Per-band snow age [s], (ncol, n_bands) — companion to ``snow_bands`` (fresh
packages/land/legoesm/land/state.py:59:    # ``None`` (legacy) uses the single cell-mean ``snow_age`` clock.
packages/land/legoesm/land/state.py:60:    snow_age_bands: jax.Array = None
packages/land/legoesm/land/multilayer_land.py:66:from legoesm.land.snow_budget import update_snow_age
packages/land/legoesm/land/multilayer_land.py:584:    snow_age = state.snow_age    # (ncol,)
packages/land/legoesm/land/multilayer_land.py:970:        # reflected the exported snow-aged 0.52 on the same cell -- nothing

exec
/usr/bin/bash -lc "nl -ba packages/tools/legoesm/forcing/surface_utils.py | sed -n '453,550p'; nl -ba packages/coupler/legoesm/driver/model_driver.py | sed -n '6070,6115p'; nl -ba config/amip/amip_production.yaml | sed -n '270,310p'; rg -n 'def .*solar|roll|rrtmg_sw|365' packages/tools/legoesm/forcing/external.py | tail -30" in /glade/u/home/pg2328/legoesm
 succeeded in 0ms:
   453	        Freezing point [K] (pass ``constants.T_freeze``).
   454	    transition_center_offset_K : float
   455	        Upper edge of the mixed-phase band above freezing [K] (default 2.0).
   456	    transition_width_K : float
   457	        Total width of the linear ramp [K] (default 4.0).
   458	
   459	    Returns
   460	    -------
   461	    snow_frac : array
   462	        Snow fraction in [0, 1].
   463	    """
   464	    return jnp.clip(
   465	        (T_freeze + transition_center_offset_K - T_low) / transition_width_K,
   466	        0.0,
   467	        1.0,
   468	    )
   469	
   470	
   471	def distribute_column_aod_to_layers(
   472	    aod_col: jnp.ndarray,
   473	    p_half_col: jnp.ndarray,
   474	) -> jnp.ndarray:
   475	    """Distribute column aerosol optical depth to layers by pressure thickness.
   476	
   477	    Parameters
   478	    ----------
   479	    aod_col : array, shape (ncol,)
   480	        Total column AOD per column.
   481	    p_half_col : array, shape (ncol, nlev+1)
   482	        Half-level pressures.
   483	
   484	    Returns
   485	    -------
   486	    aod_layers : array, shape (ncol, nlev)
   487	        Layer-distributed AOD.
   488	    """
   489	    dp = jnp.clip(p_half_col[:, 1:] - p_half_col[:, :-1], 1.0e-12, None)
   490	    w = dp / jnp.sum(dp, axis=1, keepdims=True)
   491	    return jnp.clip(aod_col, 0.0, None)[:, None] * w
   492	
   493	
   494	def place_stratospheric_aod_profile_to_layers(
   495	    aod_profile: jnp.ndarray,
   496	    p_edges: jnp.ndarray,
   497	    p_half_col: jnp.ndarray,
   498	) -> jnp.ndarray:
   499	    """Conservatively remap a fixed-pressure-edge AOD profile onto model layers.
   500	
   501	    Volcanic stratospheric aerosol is supplied as a per-layer absorption
   502	    optical depth on the forcing file's OWN (altitude -> pressure) grid.  This
   503	    bins it onto the model's pressure layers by fractional pressure overlap,
   504	    so the aerosol lands at its true stratospheric pressure -- unlike
   505	    :func:`distribute_column_aod_to_layers`, which spreads a single column AOD
   506	    by full-column pressure mass and thus dumps ~90 % of a stratospheric layer
   507	    into the troposphere.
   508	
   509	    Method: the cumulative-OD curve is sampled at each model half level and the
   510	    per-layer OD is the difference of the cumulative at the layer's two edges.
   511	    The cumulative is interpolated in LOG-pressure, because the source OD is
   512	    ``ext*dz`` -- uniform in geometric height (~ log-pressure) within a source
   513	    layer, NOT uniform in pressure; a linear-pressure CDF would misallocate ~1 %
   514	    of a 0.5 km source layer's OD across a cutting model interface (Codex review
   515	    iter-1).  The remap is exact at source-layer edges and conserves the total
   516	    column OD that falls within the model's pressure range (OD above the model
   517	    top or below the surface edge is dropped -- it cannot be represented).
   518	
   519	    The op set (searchsorted / gather / clip / min / max) is piecewise-linear:
   520	    a.e.-differentiable in both ``aod_profile`` and ``p_half_col`` with
   521	    well-defined one-sided gradients (kinks at source/model edges), and
   522	    JIT/vmap-safe.
   523	
   524	    Sign/units: optical depth is >= 0 and additive; pressure [Pa].
   525	
   526	    Parameters
   527	    ----------
   528	    aod_profile : array, shape (ncol, nsrc)
   529	        Per-source-layer absorption optical depth [-] (>= 0).
   530	    p_edges : array, shape (nsrc+1,)
   531	        Source-layer pressure edges [Pa], ASCENDING (index 0 = top / lowest
   532	        pressure), aligned so ``aod_profile[:, j]`` occupies
   533	        ``[p_edges[j], p_edges[j+1]]``.
   534	    p_half_col : array, shape (ncol, nlev+1)
   535	        Model half-level pressures [Pa] (index 0 = top).
   536	
   537	    Returns
   538	    -------
   539	    aod_layers : array, shape (ncol, nlev)
   540	        Layer absorption optical depth [-].
   541	    """
   542	    pe = jnp.asarray(p_edges)                                   # (nsrc+1,)
   543	    aod = jnp.clip(jnp.asarray(aod_profile), 0.0, None)         # (ncol, nsrc)
   544	    # Cumulative OD from the top: cum[:, k] = OD in source layers 0..k-1, so
   545	    # cum aligns to the nsrc+1 source edges ``pe``.
   546	    cum = jnp.concatenate(
   547	        [jnp.zeros((aod.shape[0], 1), aod.dtype), jnp.cumsum(aod, axis=1)],
   548	        axis=1,
   549	    )                                                          # (ncol, nsrc+1)
   550	    # Interpolate the CDF in log-pressure (source OD is uniform in geometric
  6070	        )
  6071	
  6072	        nlev = self.sigma.sigma_full.shape[0]
  6073	        shape_2d = p_s.shape
  6074	        ncol = int(np.prod(np.array(shape_2d)))
  6075	
  6076	        p_full = self.sigma.pressure_at_full(p_s)
  6077	        p_half = self.sigma.pressure_at_half(p_s)
  6078	        p_full_col = p_full.reshape(ncol, nlev)
  6079	        p_half_col = p_half.reshape(ncol, nlev + 1)
  6080	        lat_col = lat.reshape(ncol)
  6081	        # Ozone is evaluated on the model levels PLUS 16 midpoints of equal
  6082	        # pressure sub-layers above the model top (0 < p < p_top); their mean
  6083	        # is the overhead column-mean VMR for the RRTMGP overhead layer
  6084	        # (RRTMGPConfig.overhead_layer; unused when that is off).  The inline sources other than
  6085	        # "mls" are pointwise in p, so the model-level values are unchanged;
  6086	        # "mls" (gSAM layer means) is handled from the interfaces below.
  6087	        _p_o3 = jnp.concatenate([
  6088	            p_full_col,
  6089	            p_half_col[:, :1] * ((jnp.arange(16) + 0.5) / 16.0),
  6090	        ], axis=1)
  6091	
  6092	        # Volcanic aerosol is cut at each column's WMO lapse-rate tropopause
  6093	        # diagnosed from T at this precompute (the same state snapshot as
  6094	        # ``p_s``; callers refresh daily).  ``temp`` defaults to the state's
  6095	        # grid-point T; absent or off the ``p_s`` grid -> climatological
  6096	        # tropopause.  Only when volcanic aerosol can be on (config-uniform,
  6097	        # so the SPMD gather in ``_owned_temp`` stays collective).
  6098	        w_strat = None
  6099	        if self._aerosol_active or self._aerosol_lw_active:
  6100	            if temp is None:
  6101	                temp = self._owned_temp()
  6102	            temp_col = None
  6103	            if temp is not None and tuple(temp.shape) == tuple(p_full.shape):
  6104	                temp_col = jnp.asarray(temp).reshape(ncol, nlev)
  6105	            w_strat = stratospheric_layer_weight(
  6106	                p_half_col, lat_col, p_full_col, temp_col)
  6107	
  6108	        if self._ozone_ext_active:
  6109	            o3_vmr = jnp.asarray(get_ozone_at_time(
  6110	                self._ozone_ext_config, day,
  6111	                lat_grid=lat_col, p_grid=_p_o3,
  6112	            ))
  6113	        else:
  6114	            # External ozone forcing inactive — build the inline profile
  6115	            # HERE so the radiation JIT signature stays stable (it always
   270	                                     # unrolled on CPU, scan on GPU, so GPU is unchanged).
   271	                                     # Measured 2026-10-29, this deck, MPAS res6 CPU: step
   272	                                     # -4% at 1 node, -7% at 8; round-off only (fp32 fluxes
   273	                                     # <= 1.1e-7 of max after one step), dry mass unchanged,
   274	                                     # 127-step divergence (0.66 K) matched by a 1e-14 T
   275	                                     # perturbation control (0.27 K).  User 2026-10-30.
   276	hard_sat_ice_curve: false          # the ice-curve blend of that same
   277	                                   # post-step drain; inert once the drain
   278	                                   # is off, set explicitly so no row rests
   279	                                   # on a default.  User table row 3.
   280	homogeneous_ice_nucleation: true
   281	aerosol_ccn: true                  # droplet number diagnosed from the prescribed
   282	                                   # aerosol (user decision 2026-11-01, #2216,
   283	                                   # reverses 2026-10-24): Jan 1979 30-day arm,
   284	                                   # global rsut bias vs CERES-EBAF -7.99 W/m2
   285	                                   # (prognostic) -> +0.78 (this setting).
   286	morrison_prognostic_droplet_number: false  # CAM6 MG2 prognostic droplet number
   287	                                   # OFF until its droplet budget closes (#2216);
   288	                                   # the prognostic arm is
   289	                                   # amip_production_prognostic_nc.yaml.
   290	                                   # cam6_aerosol_dir below is read only by it.
   291	morrison_mg2_drop_early_reset_log_width: 0.01  # NAMED, equals the code default:
   292	                                   # RULE 5 width of MG2's early droplet size
   293	                                   # reset (autoconversion + snow riming on
   294	                                   # this deck's specified droplet number; 0 =
   295	                                   # MG2's hard clip; user 2026-11-01, #2216)
   296	morrison_mg2_drop_sed_reset_log_width: 0.01  # NAMED, equals the code default:
   297	                                   # RULE 5 width of MG2's droplet size reset in
   298	                                   # the sedimentation fall speeds (read only by
   299	                                   # the prognostic droplet number, so inert on
   300	                                   # this deck; 0 = MG2's hard clip; #2216)
   301	cam6_aerosol_dir: /work/bb1596/b309178/diffESM/legoesm_ap/data/cam6_optics
   302	energy_consistent_moisture_clip: true
   303	
   304	# ---- convection ---------------------------------------------------------------
   305	zm_land_fraction: required  # ZM reads the land mask (c0_lnd vs c0_ocn); "none" = aquaplanet only
   306	convection: zhang_mcfarlane # CAM6: faithful zm_conv port (f09 mg2+clubb namelist:
   307	                            # c0_lnd 0.0075, c0_ocn 0.03, ke 5e-6 (dyn=fv),
   308	                            # tau 3600,
   309	                            # capelmt 70, tiedke_add 0.5, momcu = momcd = 0.7,
   310	                            # num_cin 1); scheme defaults carry those values.
1251:    uses 365.0; the previous 365.25 period drifted the seasonal phase by
1254:    period = 365.0
1276:    months), ``_interp_monthly_cyclic`` wraps with period 365 d and
1357:    365-day clock).  Handles negative ``sim_day`` (dates before the sim
1363:    # NOLEAP_MONTH_STARTS is length 13 ([12] == 365), so side="right" gives
1384:    [0, 365) and co-sorted with ``data`` so that a climatology starting
1409:    anchored = (np.asarray(mid_days, dtype=np.float64) + offset) % 365.0
1818:        # (365-day) clock — matching the experiment-transient override in
1819:        # model_driver (day/365.0) and the calendar-date mapping the other
1820:        # forcing channels use.  365.25 here drifted ~9 days over a 35-year
1822:        year = config.start_year + day / 365.0
2176:        # file would be sampled cyclically (mod 365.25), throwing away
2243:            # path mod-365.25-wrapped a multi-year file onto 12 months
2435:        - ``"rrtmg_sw"``: always rotate (caller asserts the file is in
2460:def get_solar_forcing_at_time(config: SolarConfig, day: float) -> dict:
2522:        if config.spectral_band_order not in ("auto", "as_is", "rrtmg_sw"):
2525:                f"or 'rrtmg_sw'; got {config.spectral_band_order!r}",
2528:        _n_bands, _n_gpt = _rrtmg_sw_band_counts(_sw_table)
2540:            #   "rrtmg_sw" -> always rotate (caller asserts CMIP order).
2552:            if config.spectral_band_order == "rrtmg_sw" or (
2558:                        "(spectral_var=%r); applying 'rrtmg_sw' band order "
2580:def _rrtmg_sw_band_counts(rrtmg_sw_path: str) -> tuple[int, int]:
2588:    ds = (xr.open_dataset(rrtmg_sw_path) if not rrtmg_sw_path.endswith(".zarr")
2589:          else xr.open_zarr(rrtmg_sw_path))
2595:def _expand_bands_to_gpoints(spec_bands: np.ndarray, rrtmg_sw_path: str) -> np.ndarray:
2629:    rrtmg_sw_path : str
2639:    ds = xr.open_dataset(rrtmg_sw_path) if not rrtmg_sw_path.endswith(".zarr") \
2640:        else xr.open_zarr(rrtmg_sw_path)
2691:    hence ``np.roll(spec, +1, axis=-1)`` converts CMIP order -> RRTMGP
2714:    return np.roll(spec_bands, 1, axis=-1)

codex
The cited code supports several implementation claims, but it doesn’t establish the forcing bounds or bias attribution. I’m checking the CAM6 paths and land inputs before assigning the final verdicts.
exec
/usr/bin/bash -lc "rg -n 'def aerosol_layer_fields' packages/tools/legoesm/forcing/surface_utils.py; nl -ba packages/tools/legoesm/forcing/surface_utils.py | sed -n '665,795p'; nl -ba packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py | sed -n '490,525p'; nl -ba packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py | sed -n '1100,1130p'; nl -ba packages/tools/legoesm/forcing/external.py | sed -n '1828,1854p'; rg -n -i 'nitrogen|co2|snow_albedo' packages/land/legoesm/land/two_leaf.py packages/land/legoesm/land/albedo.py | head -40" in /glade/u/home/pg2328/legoesm
 succeeded in 0ms:
808:def aerosol_layer_fields(acfg, day, lat_col, p_half_col, w_strat, *,
   665	    T_melt_surface_K : fresh-ice surface melting point [K] (upper cap).
   666	    melt_cap_width_K : width [K] of the smooth melt cap.  ``0`` = the hard
   667	        ``clip`` cap; ``> 0`` = ``smooth_min(T_new, T_melt_surface_K,
   668	        sharpness=1/width)`` (LogSumExp, never above the melting point), so
   669	        the skin keeps a nonzero gradient at the cap.  KNOWN DIFFERENCE vs the
   670	        hard cap: a melting skin sits slightly below 0 C, by about
   671	        ``width * ln(1 / (1 - exp(-dT/width)))`` for a per-step warming dT
   672	        (measured per case in tests/unit/test_ice_skin_smooth_cap.py).
   673	
   674	    Returns
   675	    -------
   676	    (ncol,) updated skin temperature [K].
   677	    """
   678	    C_areal = 0.5 * constants.rho_ice * constants.c_pi * h_ice_m  # [J/m^2/K]
   679	    g_cond = k_ice_W_m_K / h_ice_m                                # [W/m^2/K]
   680	    r = dt_s / C_areal
   681	    T_new = (T_skin + r * (F_net_down_W_m2 + g_cond * T_freeze_K)) \
   682	        / (1.0 + r * g_cond)
   683	    if melt_cap_width_K == 0.0:
   684	        T_new = jnp.clip(T_new, _ICE_SKIN_FLOOR_K, T_melt_surface_K)
   685	    else:
   686	        from legoesm.core.smooth import smooth_min
   687	        T_new = smooth_min(jnp.maximum(T_new, _ICE_SKIN_FLOOR_K),
   688	                           T_melt_surface_K, 1.0 / melt_cap_width_K)
   689	    return jnp.where(sic > 0.0, T_new, T_freeze_K)
   690	
   691	
   692	def advance_ice_skin(
   693	    T_skin: jnp.ndarray,
   694	    sw_net: jnp.ndarray,
   695	    lw_net: jnp.ndarray,
   696	    shflx: jnp.ndarray | None,
   697	    lhflx: jnp.ndarray | None,
   698	    sic: jnp.ndarray,
   699	    dt_s: float,
   700	    h_ice_m: float,
   701	    melt_cap_width_K: float = 0.0,
   702	) -> jnp.ndarray:
   703	    """One step of the prognostic ice skin from a lane's surface fluxes.
   704	
   705	    Grid-agnostic per-column physics shared by every lane: ``F_net_down =
   706	    sw_net + lw_net - shflx - lhflx`` (sw/lw net [W/m^2, +into surface];
   707	    turbulent fluxes [W/m^2, +upward]; an absent turbulent flux contributes
   708	    nothing), then :func:`prognostic_ice_skin_temperature`.  Inputs are
   709	    flattened to columns and the result has the shape of ``T_skin``.
   710	    """
   711	    shape = jnp.shape(T_skin)
   712	    f_net = (jnp.asarray(sw_net).reshape(-1) + jnp.asarray(lw_net).reshape(-1))
   713	    if shflx is not None:
   714	        f_net = f_net - jnp.asarray(shflx).reshape(-1)
   715	    if lhflx is not None:
   716	        f_net = f_net - jnp.asarray(lhflx).reshape(-1)
   717	    out = prognostic_ice_skin_temperature(
   718	        jnp.asarray(T_skin).reshape(-1), f_net, jnp.asarray(sic).reshape(-1),
   719	        dt_s=dt_s, h_ice_m=h_ice_m, melt_cap_width_K=melt_cap_width_K)
   720	    return out.reshape(shape)
   721	
   722	# NOTE: the model has OTHER tropopauses (clouds/cloud_fraction.py): the CAM6
   723	# aist cloud switch (rhminis/rhmaxis) uses the analytic 25000 - 15000
   724	# cos^2(lat) Pa surface (cam6_tropopause_analytic), and CLUBB's clubbtop uses
   725	# CAM's chemistry tropopause (cam_chem_tropopause_level: twmo + polar rule,
   726	# that surface as fallback).  This WMO locator places volcanic aerosol only;
   727	# unifying them is an open follow-up.
   728	def wmo_tropopause_pressure(p_full_col: jnp.ndarray, temp_col: jnp.ndarray,
   729	                            p_bot: float = 5.0e4,
   730	                            p_top: float = 5.0e3) -> jnp.ndarray:
   731	    """WMO (1957) lapse-rate tropopause pressure [Pa] per column: the lowest
   732	    full level in [p_top, p_bot] [Pa] whose lapse rate to the next level up
   733	    is <= 2 K/km and whose mean lapse rate to every higher level within 2 km
   734	    (and to 2 km itself) stays <= 2 K/km.
   735	    NaN where no level qualifies -- incl. any NaN T in the column (it
   736	    poisons z above and every within-2-km test below).
   737	    ``p_full_col``/``temp_col`` (ncol, nlev) in either vertical order;
   738	    heights by hypsometric integration of T.
   739	
   740	    ponytail: the tropopause snaps to a full level (no sub-layer interpolation
   741	    of the 2 K/km crossing as in CAM's twmo), i.e. it is located to within one
   742	    layer -- all that the 1/0.5/0 layer weighting below resolves; dry T (not
   743	    virtual) in the hypsometric dz (<0.1% at these heights); the within-2-km
   744	    test loops over all nlev level offsets, O(nlev^2) per column (daily).
   745	    """
   746	    order = jnp.argsort(-p_full_col, axis=1)  # surface first
   747	    p = jnp.take_along_axis(p_full_col, order, axis=1)
   748	    temp = jnp.take_along_axis(temp_col, order, axis=1)
   749	    dz = (constants.R_d / constants.g) * 0.5 * (temp[:, :-1] + temp[:, 1:]) \
   750	        * jnp.log(p[:, :-1] / p[:, 1:])
   751	    z = jnp.concatenate([jnp.zeros_like(dz[:, :1]),
   752	                         jnp.cumsum(dz, axis=1)], axis=1)
   753	    z2 = z[:, :-1] + 2000.0
   754	    temp_2km = jax.vmap(jnp.interp)(z2, z, temp)  # T 2 km above each base level
   755	    # Mean lapse rate j->m <= 2 K/km  <=>  s_m >= s_j with s = T + 2 K/km * z.
   756	    s = temp + 2e-3 * z
   757	    ok = (((temp[:, :-1] - temp[:, 1:]) / dz <= 2e-3)
   758	          & (temp_2km + 2e-3 * z2 >= s[:, :-1])
   759	          & (z2 <= z[:, -1:])  # the 2 km test must fit in the column
   760	          & (p[:, :-1] <= p_bot) & (p[:, :-1] >= p_top))
   761	    n = s.shape[1]
   762	    for m in range(2, n):  # level offsets (static nlev), not columns
   763	        hit = (s[:, m:] >= s[:, :-m]) | (z[:, m:] - z[:, :-m] > 2000.0)
   764	        ok = ok & jnp.pad(hit, ((0, 0), (0, m - 1)), constant_values=True)
   765	    k = jnp.argmax(ok, axis=1)  # first True = lowest qualifying level
   766	    p_tp = jnp.take_along_axis(p[:, :-1], k[:, None], axis=1)[:, 0]
   767	    return jnp.where(ok.any(axis=1), p_tp, jnp.nan)
   768	
   769	
   770	def stratospheric_layer_weight(p_half_col: jnp.ndarray,
   771	                               lat_rad: jnp.ndarray,
   772	                               p_full_col: jnp.ndarray | None = None,
   773	                               temp_col: jnp.ndarray | None = None,
   774	                               ) -> jnp.ndarray:
   775	    """Per-layer weight for CMIP6 stratospheric (volcanic) aerosol at the
   776	    tropopause, as the ETH/SAGE-3lambda data notes prescribe: 1 above the
   777	    tropopause, 0.5 in the layer holding it, 0 below (the file's values below
   778	    the tropopause are an extrapolated matching tail, not tropospheric
   779	    aerosol).  ``p_half_col`` (ncol, nlev+1) [Pa] in either vertical order,
   780	    ``lat_rad`` (ncol,).
   781	
   782	    The tropopause is the column's WMO lapse-rate tropopause
   783	    (``wmo_tropopause_pressure``) when ``p_full_col``/``temp_col`` (ncol, nlev)
   784	    are given, falling back per column to the climatological
   785	    p_tp = 300 - 200 cos^2(lat) hPa (100 hPa tropics, 300 hPa poles) where
   786	    none is found or no T is given.
   787	
   788	    ponytail: no temporal smoothing -- the ETH notes ask for the
   789	    instantaneous tropopause.  Measured (scripts/validate/
   790	    check_wmo_tropopause_fv3duo.py, code e4d7a50ff, x64) on FV3-duo C24 L32
   791	    daily checkpoints duo_5d days 1-5: 0% of 3456 columns fall back (also 0%
   792	    at duo_60d days 5-30); the tropopause layer moves >=1 level day-to-day in
   793	    30% of columns, >=2 in 4% (30-60 deg 43%/7%, tropics 20%/2%), 24% of
   794	    moves flip back next day -- consistent with synoptic motion at L32 layer
   795	    spacing; threshold chatter is not separately quantified.  Smooth (e.g. a
   490	              cloud_longwave_nc_filepath=lw_cloud,
   491	              cloud_shortwave_nc_filepath=sw_cloud,
   492	          )
   493	          optics_params = OpticsParameters(optics=rrtm_optics)
   494	
   495	          # Build VMR library with global means from legoESM config.
   496	          global_means = {
   497	              optics_constants.DRY_AIR_KEY: optics_constants.DRY_AIR_VMR,
   498	              "co2": config.co2_ppmv * 1.0e-6,
   499	              "ch4": config.ch4_ppbv * 1.0e-9,
   500	              "n2o": config.n2o_ppbv * 1.0e-9,
   501	              "o2": _O2_MOLE_FRACTION,
   502	              "n2": _N2_MOLE_FRACTION,
   503	              "co": 1.5e-7,
   504	              "ccl4": 7.5e-11,
   505	              "cfc11": 2.2e-10,
   506	              "cfc12": 5.0e-10,
   507	              "cfc22": 2.4e-10,
   508	              "cf4": 8.5e-11,
   509	              "no2": 3.0e-10,
   510	          }
   511	          vmr_lib = LookupVolumeMixingRatio(
   512	              global_means=global_means, profiles=None,
   513	          )
   514	
   515	          # iter-40: pass include_clouds to skip cloud-table load for
   516	          # clear-sky-only workflows; ~MB saved per cached entry.
   517	          optics_lib = optics_factory(
   518	              optics_params, vmr_lib,
   519	              include_clouds=config.include_clouds,
   520	          )
   521	          if getattr(config, "compute_fp32", False):
   522	              # Cast the optics tables float64 -> float32 (the dycore keeps
   523	              # fp64).  GATED OFF in production: the MPAS driver sets
   524	              # ``compute_fp32=False`` because the fp32 RTE path still crashes
   525	              # (the shortwave direct-beam recurrence re-promotes the scan carry
  1100	          lw_path_fn = _mcica_paths(optics_lib.n_gpt_lw)
  1101	          sw_path_fn = _mcica_paths(optics_lib.n_gpt_sw)
  1102	
  1103	      # Optional aerosol optical depth (shortwave).  Clip AFTER ``_add_halos``
  1104	      # (same fix class as q_v / o3 / cf / cloud paths): linear halo
  1105	      # extrapolation can drive a boundary aerosol OD negative, which is an
  1106	      # unphysical (negative) optical depth feeding the interior recurrence.
  1107	      if aerosol_optical_depth is not None:
  1108	          aerosol_od_3d = jnp.clip(
  1109	              _add_halos(aerosol_optical_depth[:, None, ::-1]), 0.0, None,
  1110	          )
  1111	      else:
  1112	          aerosol_od_3d = None
  1113	
  1114	      # Aerosol SW single-scattering albedo + asymmetry: scalar (grey, default)
  1115	      # or per-shortwave-band arrays (config.aerosol_ssa_bands/aerosol_g_bands).
  1116	      # Resolved once here and shared by the MC-optics short-circuit and the
  1117	      # full SW solve so the per-band selection is defined in exactly one place.
  1118	      aer_ssa, aer_g = _resolve_aerosol_sw_optics(config, optics_lib)
  1119	
  1120	      # Optional aerosol optical depth (longwave, pure absorber)
  1121	      if aerosol_absorption_optical_depth_lw is not None:
  1122	          aerosol_od_lw_3d = jnp.clip(
  1123	              _add_halos(aerosol_absorption_optical_depth_lw[:, None, ::-1]),
  1124	              0.0, None,
  1125	          )
  1126	      else:
  1127	          aerosol_od_lw_3d = None
  1128	
  1129	      # Optional spectral solar forcing
  1130	      if solar_spectral_fraction is not None:
  1828	        # fraction labelled "co2_ppmv", so radiation saw CO2 ≈ 3.4e-10
  1829	        # vmr — effectively zero CO2, a ~1 K/day spurious LW cooling.)
  1830	        return {
  1831	            "co2_ppmv": _interp_1d(years, data["CO2"], year) * 1.0e6,
  1832	            "ch4_ppbv": _interp_1d(years, data["CH4"], year) * 1.0e9,
  1833	            "n2o_ppbv": _interp_1d(years, data["N2O"], year) * 1.0e9,
  1834	            "cfc11_pptv": _interp_1d(years, data["CFC_11"], year) * 1.0e12,
  1835	            "cfc12_pptv": _interp_1d(years, data["CFC_12"], year) * 1.0e12,
  1836	            **{key: _interp_1d(years, data[var], year) * 1.0e12
  1837	               for var, (key, _) in _GHG_OPTIONAL_HALOGENS.items()
  1838	               if var in data},
  1839	        }
  1840	    else:
  1841	        raise ValueError(f"Unknown GHG source: {config.source!r}")
  1842	
  1843	
  1844	# File variable -> (get_ghg_at_time key [pptv], RRTMGP gas name).  Physical
  1845	# species, not CFC-equivalents: RRTMGP carries each of these itself (fixed
  1846	# global means in rrtmgp.py otherwise), so equivalents would double count.
  1847	_GHG_OPTIONAL_HALOGENS = {
  1848	    "CFC_22": ("cfc22_pptv", "cfc22"),
  1849	    "CCl4": ("ccl4_pptv", "ccl4"),
  1850	    "CF4": ("cf4_pptv", "cf4"),
  1851	}
  1852	
  1853	
  1854	def ghg_concentrations_to_vmr(ghg: dict) -> dict:
rg: packages/land/legoesm/land/two_leaf.py: No such file or directory (os error 2)
rg: packages/land/legoesm/land/albedo.py: No such file or directory (os error 2)

exec
/usr/bin/bash -lc "nl -ba /glade/derecho/scratch/pg2328/cesm_src/CAM/bld/namelist_files/use_cases/hist_cam6.xml | head -85; rg -n -i 'co2.*(pp|vmr|constant)|co2_pp|ndep|landuse|transient' packages/land/legoesm/land/canopy* packages/land/legoesm/land/two_leaf* packages/land/legoesm/land/clm_surface_map.py | head -65; rg -n 'andreae|Andreae|aerosol_ccn' packages/atmosphere/legoesm/atmosphere/physics/microphysics/aerosol.py; rg -n 'def .*albedo|impurit|age|exp' packages/land/legoesm/land/albedo.py | head -45" in /glade/u/home/pg2328/legoesm
 succeeded in 0ms:
     1	<?xml version="1.0"?>
     2	<namelist_defaults>
     3	
     4	  <ncdata hgrid="1.9x2.5" nlev="32">cesm2_init/b.e20.B1850.f19_g17.release_cesm2_1_0.020/0301-01-01/b.e20.B1850.f19_g17.release_cesm2_1_0.020.cam.i.0301-01-01-00000.nc</ncdata>
     5	
     6	  <solar_irrad_data_file    >   'atm/cam/solar/SolarForcingCMIP6_18491230-23000102_c20200615.nc'</solar_irrad_data_file>
     7	
     8	  <tracer_cnst_datapath     >   'atm/cam/tracer_cnst'                                        </tracer_cnst_datapath>
     9	  <tracer_cnst_file         >   'tracer_cnst_halons_3D_L70_1849-2015_CMIP6ensAvg_c180927.nc' </tracer_cnst_file>
    10	  <tracer_cnst_specifier    >   'O3','OH','NO3','HO2'                                        </tracer_cnst_specifier>
    11	  <tracer_cnst_type         >   'INTERP_MISSING_MONTHS'                                      </tracer_cnst_type>
    12	  <tracer_cnst_filelist     >   ''                                                           </tracer_cnst_filelist>
    13	
    14	  <prescribed_ozone_datapath>   'atm/cam/ozone_strataero'                                                    </prescribed_ozone_datapath>
    15	  <prescribed_ozone_file    >   'ozone_strataero_WACCM_L70_zm5day_18500101-20150103_CMIP6ensAvg_c180923.nc'  </prescribed_ozone_file>
    16	  <prescribed_ozone_name    >   'O3'                                                                         </prescribed_ozone_name>
    17	  <prescribed_ozone_type    >   'SERIAL'                                                                     </prescribed_ozone_type>
    18	
    19	  <srf_emis_type            >    INTERP_MISSING_MONTHS  </srf_emis_type>
    20	  <ext_frc_type             >    INTERP_MISSING_MONTHS  </ext_frc_type>
    21	
    22	  <prescribed_strataero_type    >     'SERIAL'                                                                     </prescribed_strataero_type>
    23	  <prescribed_strataero_datapath>     'atm/cam/ozone_strataero'                                                    </prescribed_strataero_datapath>
    24	  <prescribed_strataero_file>         'ozone_strataero_WACCM_L70_zm5day_18500101-20150103_CMIP6ensAvg_c180923.nc'  </prescribed_strataero_file> 
    25	  <prescribed_strataero_use_chemtrop> .true.                                                                       </prescribed_strataero_use_chemtrop>
    26	
    27	  <scenario_ghg            >    'CHEM_LBC_FILE'   </scenario_ghg>
    28	  <flbc_file>atm/waccm/lb/LBC_1750-2015_CMIP6_GlobAnnAvg_c180926.nc</flbc_file>
    29	  <flbc_type>'SERIAL'</flbc_type>
    30	  <flbc_list>'CO2','CH4','N2O','CFC11eq','CFC12'</flbc_list>
    31	
    32	<fincl1 hgrid="1.9x2.5">
    33	         'dst_a1', 'dst_c1', 'dst_a3', 'dst_c3', 'dst_a1_SRF', 'dst_a3_SRF', 'dst_a1DDF', 'dst_a1SFWET', 'dst_c1DDF',
    34	         'dst_c1SFWET', 'dst_a3DDF', 'dst_a3SFWET', 'dst_c3DDF', 'dst_c3SFWET', 'SSAVIS', 'AODABS', 'AODDUST2', 'bc_a1',
    35	         'bc_a4', 'dst_a1', 'dst_a2', 'dst_a3', 'ncl_a1', 'ncl_a1', 'ncl_a2', 'ncl_a3', 'pom_a1',
    36	         'pom_a4', 'so4_a1', 'so4_a2', 'so4_a3', 'soa_a1', 'soa_a2', 'bc_c1', 'bc_c4', 'dst_c1',
    37	         'dst_c2', 'dst_c3', 'ncl_c1', 'ncl_c1', 'ncl_c2', 'ncl_c3', 'pom_c1', 'pom_c4', 'so4_c1',
    38	         'so4_c2', 'so4_c3', 'soa_c1', 'soa_c2', 'num_a1', 'num_a2', 'num_a4', 'num_c1', 'num_c2',
    39	         'num_c3', 'bc_a1SFWET', 'bc_a4SFWET', 'dst_a1SFWET', 'dst_a2SFWET', 'dst_a3SFWET', 'ncl_a1SFWET', 'ncl_a2SFWET', 'ncl_a3SFWET',
    40	         'pom_a1SFWET', 'pom_a4SFWET', 'so4_a1SFWET', 'so4_a2SFWET', 'so4_a3SFWET', 'soa_a1SFWET', 'soa_a2SFWET', 'bc_c1SFWET', 'bc_c4SFWET',
    41	         'dst_c1SFWET', 'dst_c2SFWET', 'dst_c3SFWET', 'ncl_c1SFWET', 'ncl_c2SFWET', 'ncl_c3SFWET', 'pom_c1SFWET', 'pom_c4SFWET', 'so4_c1SFWET',
    42	         'so4_c2SFWET', 'so4_c3SFWET', 'soa_c1SFWET', 'soa_c2SFWET', 'bc_a1DDF', 'bc_a4DDF', 'dst_a1DDF', 'dst_a2DDF', 'dst_a3DDF',
    43	         'ncl_a1DDF', 'ncl_a2DDF', 'ncl_a3DDF', 'pom_a1DDF', 'pom_a4DDF', 'so4_a1DDF', 'so4_a2DDF', 'so4_a3DDF', 'soa_a1DDF',
    44	         'soa_a2DDF', 'bc_c1DDF', 'bc_c4DDF', 'dst_c1DDF', 'dst_c2DDF', 'dst_c3DDF', 'ncl_c1DDF', 'ncl_c2DDF', 'ncl_c3DDF',
    45	         'pom_c1DDF', 'pom_c4DDF', 'so4_c1DDF', 'so4_c2DDF', 'so4_c3DDF', 'soa_c1DDF', 'soa_c2DDF', 'bc_a4_CLXF', 'pom_a4_CLXF',
    46	         'so4_a1_CLXF', 'so4_a2_CLXF', 'SFbc_a4', 'SFpom_a4', 'SFso4_a1', 'SFso4_a2', 'so4_a1_sfgaex1', 'so4_a2_sfgaex1', 'so4_a3_sfgaex1',
    47	         'soa_a1_sfgaex1', 'soa_a2_sfgaex1', 'so4_a2_sfnnuc1', 'so4_c1AQH2SO4', 'so4_c2AQH2SO4', 'so4_c3AQH2SO4', 'so4_c1AQSO4', 'so4_c2AQSO4', 'so4_c3AQSO4',
    48	         'ADRAIN', 'ADSNOW', 'ANRAIN', 'ANSNOW', 'AODDUST', 'AODVIS', 'AQRAIN', 'AQSNOW', 'AQSO4_H2O2',
    49	         'AQSO4_O3', 'AREI', 'AREL', 'AWNC', 'AWNI', 'CCN3', 'CDNUMC', 'CLDHGH', 'CLDICE',
    50	         'CLDLIQ', 'CLDLOW', 'CLDMED', 'CLDTOT', 'CLOUD', 'CONCLD', 'DMS', 'FICE', 'FLDS',
    51	         'FLNS', 'FLNS', 'FLNT', 'FLNT', 'FLUT', 'FLUT', 'FREQI', 'FREQL', 'FREQR',
    52	         'FREQS', 'FSDS', 'FSDS', 'FSNS', 'FSNS', 'FSNT', 'FSNT', 'FSNTOA', 'FSNTOA',
    53	         'H2SO4', 'H2SO4_sfgaex1', 'ICEFRAC', 'ICIMR', 'ICWMR', 'IWC', 'LWCF', 'OMEGA', 'OMEGAT',
    54	         'PBLH', 'PRECC', 'PRECL', 'PRECSC', 'PRECSL', 'PS', 'PSL', 'Q', 'QFLX',
    55	         'QRL', 'QRS', 'QT', 'RAINQM', 'RCM_CLUBB', 'RELHUM', 'RELVAR', 'RTP2_CLUBB', 'RTPTHLP_CLUBB',
    56	         'SFDMS', 'SFSOAG', 'SNOWQM', 'SO2', 'SO2_CLXF', 'SOLIN', 'SWCF', 'T', 'TAUBLJX',
    57	         'TAUBLJY', 'TAUGWX', 'TAUGWY', 'TAUX', 'TAUY', 'TGCLDCWP', 'TGCLDIWP', 'TGCLDLWP', 'TMQ',
    58	         'TREFHT', 'TS', 'U', 'U10', 'UP2_CLUBB', 'UPWP_CLUBB', 'UU', 'V', 'VP2_CLUBB',
    59	         'VPWP_CLUBB', 'VQ', 'VU', 'VV', 'WP2_CLUBB', 'WP3_CLUBB', 'WPRCP_CLUBB', 'WPRTP_CLUBB', 'WPTHLP_CLUBB',
    60	         'WPTHVP_CLUBB', 'WSUB', 'Z3', 'bc_a1', 'bc_a1DDF', 'bc_a1SFWET', 'bc_a1_SRF', 'bc_a4', 'bc_a4DDF',
    61	         'bc_a4SFWET', 'bc_a4_SRF', 'bc_c1', 'bc_c1DDF', 'bc_c1SFWET', 'num_a2', 'num_a3', 'num_a4', 'so4_a1',
    62	         'so4_a2', 'so4_a3', 'so4_c1', 'O3', 'O3colAbove'
    63	</fincl1>
    64	
    65	  <!-- sim_year used for CO2 cycle namelist settings -->
    66	  <sim_year>1850-2000</sim_year>
    67	
    68	</namelist_defaults>
rg: packages/land/legoesm/land/two_leaf*: No such file or directory (os error 2)
packages/land/legoesm/land/clm_surface_map.py:679:# Transient land-use cover (LULC): re-weight the vegetation params by year
packages/land/legoesm/land/clm_surface_map.py:681:# The CLM map above bakes a single-year PFT cover.  For a transient land-use run
packages/land/legoesm/land/clm_surface_map.py:689:# vegetation, not soil texture; transient LAI is a documented follow-up).
packages/land/legoesm/land/clm_surface_map.py:692:def load_transient_cover_on_columns(surfdata_path, tgt_lat_deg, tgt_lon_deg):
packages/land/legoesm/land/clm_surface_map.py:693:    """Read a transient ``legoesm_surfdata`` cover and regrid it onto columns.
packages/land/legoesm/land/clm_surface_map.py:714:                f"transient cover has {pft.shape[1]} PFTs on the npft axis; expected "
packages/land/legoesm/land/clm_surface_map.py:739:    mirrors the consumer's non-transient provider so enabling transient cover with
packages/land/legoesm/land/clm_surface_map.py:752:class TransientCoverProvider(eqx.Module):
packages/land/legoesm/land/canopy/config.py:265:    # negligible CO2 uptake — light/LAI-limited GPP is essentially unchanged
packages/land/legoesm/land/canopy/config.py:600:    # light-limited rate in C4; it is independent of Vcmax25.
packages/land/legoesm/land/canopy/config.py:978:    # against independent single-column runs), so it needs no driver CLI flag.  When
packages/land/legoesm/land/canopy/clm_ml_backend/clm_src_cpl/lnd_comp_nuopc.py:25:    performs domain/grid setup independent of decomposition bounds, while
packages/land/legoesm/land/canopy/energy_balance.py:125:        # (-0.1..-0.4 W m-2) at LE = 0, a leaf-area-independent spurious dew
packages/land/legoesm/land/canopy/photosynthesis.py:39:against an independent scalar oracle (rel 1e-9):
packages/land/legoesm/land/canopy/photosynthesis.py:56:oracle supplies them as independent literals and canaries them against the
packages/land/legoesm/land/canopy/stability.py:20:per-regime resistance FORMS to round-off (rel 1e-9) against an independent scalar
packages/land/legoesm/land/canopy/radiative_transfer.py:5:  - Handles PAR, NIR, and UV independently.
packages/land/legoesm/land/canopy/radiative_transfer.py:17:``tests/land/unit/test_canopy_rt_faithful.py`` against an independent scalar
packages/land/legoesm/land/canopy/radiative_transfer.py:40:as independent oracle literals and canaried against the module constants
packages/land/legoesm/land/canopy/radiative_transfer.py:1093:    # (1-rho*r_c)).  The remaining LW_out - R_col*La is the La-independent
packages/land/legoesm/land/canopy/sif.py:83:INDEPENDENT scalar reimplementation to round-off (rel ``1e-12``), with every
packages/land/legoesm/land/canopy/sif.py:382:    high-light daytime elements, where ``fluorescence_yield(0)`` is je-INDEPENDENT
packages/land/legoesm/land/canopy/clm_ml_interface.py:1071:    # co2_ppmv × p_surface × 1e-6 → Pa
packages/land/legoesm/land/canopy/clm_ml_interface.py:1073:    co2_pa = forcing.co2_ppmv * forcing.p_surface * 1.0e-6  # Pa
packages/land/legoesm/land/canopy/clm_ml_interface.py:1506:    # GPP: µmol CO2/m²/s → gC/m²/s  (standard atomic weight of C = constants.M_C g/mol)
packages/land/legoesm/land/canopy/clm_ml_interface.py:1914:    # INDEPENDENT (no horizontal coupling — each patch is a standalone 1-D canopy),
packages/land/legoesm/land/canopy/clm_ml_interface.py:2500:        # Canopy columns are physically INDEPENDENT (no horizontal coupling), so the
packages/land/legoesm/land/canopy/clm_ml_interface.py:2510:        # ~(PFTs × nbot values), INDEPENDENT of ncol — the heterogeneous path to
packages/land/legoesm/land/canopy/clm_ml_interface.py:2513:        # anyway (explicit-count layering, htop-INDEPENDENT: MLinitVerticalMod
packages/land/legoesm/land/canopy/clm_ml_interface.py:2517:        # NO-OP vs the loop, validated column-for-column against independent
packages/land/legoesm/land/canopy/clm_ml_backend/multilayer_canopy/MLCanopyFluxesMod.py:988:        here is independent of the multilayer canopy soil energy
packages/land/legoesm/land/canopy/clm_ml_backend/multilayer_canopy/MLCanopyFluxesMod.py:1694:       Three independent checks with tolerances 0.001, 0.001, and
packages/land/legoesm/land/canopy/clm_ml_backend/offline_driver/TowerMetMod.py:71:    Return the atmospheric CO2 concentration (ppm).
packages/land/legoesm/land/canopy/clm_ml_backend/offline_driver/TowerMetMod.py:77:        Atmospheric CO2 concentration in ppm (umol/mol).
packages/land/legoesm/land/canopy/clm_ml_backend/offline_driver/TowerMetMod.py:218:        forc_pco2 = (CO2_ppm / 1e6) * forc_pbot
packages/land/legoesm/land/canopy/clm_ml_backend/offline_driver/TowerMetMod.py:471:    CO2 is stored as ppm (not Pa) in the multilayer canopy container,
packages/land/legoesm/land/canopy/clm_ml_backend/offline_driver/TowerMetMod.py:490:    co2ref_next: Array = mlcanopy_inst.co2ref_next_forcing  # CO2 (umol/mol = ppm)
packages/land/legoesm/land/canopy/clm_ml_backend/offline_driver/TowerMetMod.py:539:        # CO2 in ppm for multilayer canopy — Fortran line 280
packages/land/legoesm/land/canopy/clm_ml_backend/multilayer_canopy/MLGetAtmForcingMod.py:359:        # dispatch these independently (no dependency chain).
packages/land/legoesm/land/canopy/clm_ml_backend/multilayer_canopy/MLLeafPhotosynthesisMod.py:1339:    this function can be cached independently of their values, and the kernel
rg: packages/atmosphere/legoesm/atmosphere/physics/microphysics/aerosol.py: IO error for operation on packages/atmosphere/legoesm/atmosphere/physics/microphysics/aerosol.py: No such file or directory (os error 2)
rg: packages/land/legoesm/land/albedo.py: IO error for operation on packages/land/legoesm/land/albedo.py: No such file or directory (os error 2)

exec
/usr/bin/bash -lc "nl -ba packages/tools/legoesm/forcing/surface_utils.py | sed -n '808,900p'; rg -n 'co2_ppmv|co2_vmr|co2 =|co2=' packages/coupler/legoesm/coupler/*land* packages/coupler/legoesm/driver/model_driver.py packages/land/legoesm/land/canopy/types.py; rg -n 'Andreae|andreae' packages/atmosphere; rg -n 'def snow_albedo' packages; rg -n 'nitrogen|ndep' packages/land/legoesm/land --glob '*.py' | head -15; rg -n 'transient_cover|surfdata_path' packages/coupler/legoesm/driver/model_driver.py | head -25" in /glade/u/home/pg2328/legoesm
 succeeded in 44ms:
   808	def aerosol_layer_fields(acfg, day, lat_col, p_half_col, w_strat, *,
   809	                         active, lw_active, dtype, lon_col=None):
   810	    """Per-layer aerosol forcing for one day: ``(aerosol_od, ccn_aod,
   811	    aerosol_lw_od)``, as the AMIP driver feeds radiation and the aerosol-CCN
   812	    droplet number.  ``ccn_aod`` is None when ``active`` is off; the two
   813	    optical-depth fields are zeros (ncol, nlev) when their switch is off.
   814	    ``lon_col`` [rad] reads the SW and CCN aerosol at each column's longitude;
   815	    None reads their zonal mean."""
   816	    from legoesm.forcing.external import (
   817	        get_aerosol_at_time,
   818	        get_aerosol_lw_at_time,
   819	        get_aerosol_sw_volcanic_at_time,
   820	    )
   821	
   822	    ncol, nlev = p_half_col.shape[0], p_half_col.shape[1] - 1
   823	    ccn_aod = None
   824	    aerosol_od = jnp.zeros((ncol, nlev), dtype=dtype)
   825	    if active:
   826	        _acfg = acfg
   827	        # Volcanic SW (CMIP6 ``ext_sun``) is placed at its true
   828	        # stratospheric pressure like the LW profile below, NOT spread
   829	        # with the tropospheric column by pressure mass.  A legacy
   830	        # column-only volcanic file (profile None) stays in the column.
   831	        volc_sw = get_aerosol_sw_volcanic_at_time(_acfg, day, lat_grid=lat_col)
   832	        _trop_cfg = _acfg._replace(volcanic_enabled=False)
   833	        aerosol_col = get_aerosol_at_time(
   834	            _acfg if volc_sw is None else _trop_cfg, day, lat_grid=lat_col,
   835	            lon_grid=lon_col,
   836	        )
   837	        aerosol_od = distribute_column_aod_to_layers(
   838	            jnp.asarray(aerosol_col), p_half_col,
   839	        )
   840	        if volc_sw is not None:
   841	            aerosol_od = aerosol_od + (place_stratospheric_aod_profile_to_layers(
   842	                jnp.asarray(volc_sw[0]), jnp.asarray(volc_sw[1]), p_half_col,
   843	            ) * w_strat
   844	            ).astype(aerosol_od.dtype)
   845	        # AOD->CCN input: TROPOSPHERIC column in the visible (550 nm)
   846	        # band, volcanic excluded, from ``ccn_path`` if set.  The Andreae 2009
   847	        # fit is local total AOT: a zonal mean gives polluted E China and the
   848	        # clean N Pacific one droplet number.
   849	        ccn_aod = jnp.asarray(get_aerosol_at_time(
   850	            _trop_cfg._replace(path=_acfg.ccn_path or _acfg.path),
   851	            day, lat_grid=lat_col, band="vis", lon_grid=lon_col,
   852	        )).reshape(ncol).astype(dtype)
   853	
   854	    # Volcanic stratospheric LONGWAVE aerosol (gap #9): same (ncol,
   855	    # nlev) shape as ``aerosol_od``, default zeros so the SegmentForcing
   856	    # / forcing-dict leaf is a concrete fixed-shape array (no retrace)
   857	    # and a run without volcanic LW aerosol is byte-identical (zeros LW
   858	    # od is a RRTMGP no-op).  ``get_aerosol_lw_at_time`` returns the
   859	    # file's height-resolved ABSORPTION profile (ext*(1-omega), Planck-
   860	    # weighted gray band collapse) plus its pressure edges; the profile
   861	    # is placed at its true stratospheric pressure by a conservative
   862	    # overlap remap -- NOT spread by full-column pressure mass, which
   863	    # dumped ~90% of a stratospheric aerosol into the troposphere.
   864	    aerosol_lw_od = jnp.zeros((ncol, nlev), dtype=dtype)
   865	    if lw_active:
   866	        aerosol_lw_prof = get_aerosol_lw_at_time(
   867	            acfg, day, lat_grid=lat_col,
   868	        )
   869	        if aerosol_lw_prof is not None:
   870	            prof_col, p_edges = aerosol_lw_prof
   871	            aerosol_lw_od = (place_stratospheric_aod_profile_to_layers(
   872	                jnp.asarray(prof_col), jnp.asarray(p_edges), p_half_col,
   873	            ) * w_strat
   874	            ).astype(dtype)
   875	    return aerosol_od, ccn_aod, aerosol_lw_od
rg: packages/coupler/legoesm/coupler/*land*: No such file or directory (os error 2)
rg: packages/land/legoesm/land/canopy/types.py: No such file or directory (os error 2)
packages/coupler/legoesm/driver/model_driver.py:1339:        # Radiation's CO2 (daily GHG file, else cfg.co2_ppmv).
packages/coupler/legoesm/driver/model_driver.py:1340:        co2_ppmv=jnp.full_like(T_air, co2),
packages/coupler/legoesm/driver/model_driver.py:1833:                          "co2_ppmv", "has_radiation", "has_precipitation",
packages/coupler/legoesm/driver/model_driver.py:5912:            self.config = cfg._replace(co2_ppmv=co2, ch4_ppbv=ch4, n2o_ppbv=n2o)
packages/coupler/legoesm/driver/model_driver.py:5922:                co2_ppmv=cfg.co2_ppmv,
packages/coupler/legoesm/driver/model_driver.py:5932:                f"  GHG external: CO2={ghg_init['co2_ppmv']:.1f}ppmv, "
packages/coupler/legoesm/driver/model_driver.py:5940:                co2_ppmv=cfg.co2_ppmv,
packages/coupler/legoesm/driver/model_driver.py:6214:                    "co2_ppmv": co2,
packages/coupler/legoesm/driver/model_driver.py:6220:        # driver runs a prognostic CO2 tracer it sets ``self._co2_vmr_override``
packages/coupler/legoesm/driver/model_driver.py:6225:        co2_vmr_override = getattr(self, "_co2_vmr_override", None)
packages/coupler/legoesm/driver/model_driver.py:6226:        if (co2_vmr_override is not None
packages/coupler/legoesm/driver/model_driver.py:6228:            ghg_vmr = {**(ghg_vmr or {}), "co2": co2_vmr_override}
packages/coupler/legoesm/driver/model_driver.py:13592:        # (``cfg.co2_ppmv`` etc. — set at setup by ``ghg_at_year`` / the
packages/coupler/legoesm/driver/model_driver.py:15415:                    land_co2_ppmv(_forcing_daily.get("ghg_vmr"), cfg.co2_ppmv),
packages/coupler/legoesm/driver/model_driver.py:15425:            from legoesm.core.coupling_fields import land_co2_ppmv
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1687:    cloud-optics droplet number is overridden with the per-column Andreae
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1942:        # with the per-column Andreae (2009) AOD->CCN diagnostic so the
packages/atmosphere/legoesm/atmosphere/physics/microphysics/arg_activation.py:5::mod:`aerosol_activation` (Andreae 2009 power law).  The proxy carries no
packages/atmosphere/legoesm/atmosphere/physics/microphysics/config.py:646:    # (Andreae 2009 AOT–CCN inversion, ``aerosol_activation.ccn_from_aod``)
packages/atmosphere/legoesm/atmosphere/physics/microphysics/config.py:974:    # Andreae (2009) AOD->CCN diagnostic byte-identical; "arg" switches to the
packages/atmosphere/legoesm/atmosphere/physics/microphysics/integration.py:537:    # Aerosol-CCN specified droplet number (Andreae 2009 AOD->CCN): active
packages/atmosphere/legoesm/atmosphere/physics/microphysics/integration.py:706:        # Andreae (2009) AOD->CCN diagnostic.  The override is unconditional
packages/atmosphere/legoesm/atmosphere/physics/microphysics/integration.py:714:            # byte-identical to the Andreae (2009) ``specified_nc_field``;
packages/atmosphere/legoesm/atmosphere/physics/microphysics/aerosol_activation.py:6:link with the published global AOT–CCN correlation of Andreae (2009):
packages/atmosphere/legoesm/atmosphere/physics/microphysics/aerosol_activation.py:27:Andreae, M. O. (2009): Correlation between cloud condensation nuclei
packages/atmosphere/legoesm/atmosphere/physics/microphysics/aerosol_activation.py:44:        "aerosol optical depth (Andreae 2009 AOT-CCN relation, inverted)."
packages/atmosphere/legoesm/atmosphere/physics/microphysics/aerosol_activation.py:54:        "Andreae (2009), ACP 9, 543-556, doi:10.5194/acp-9-543-2009: "
packages/atmosphere/legoesm/atmosphere/physics/microphysics/aerosol_activation.py:71:            # CCN = (AOT / c)^(1/b) inversion of the Andreae (2009) global fit AOT = 0.0027*CCN^0.640
packages/atmosphere/legoesm/atmosphere/physics/microphysics/aerosol_activation.py:72:            "aot_coeff": {"units": "1", "bounds": (0.000891, 0.0081), "tunable_tier": 1, "transform": "sigmoid", "category": "ccn_activation", "reference": "Andreae (2009) AOD-CCN power law", "shape": None},
packages/atmosphere/legoesm/atmosphere/physics/microphysics/aerosol_activation.py:73:            "aot_exponent": {"units": "1", "bounds": (0.2112, 1.92), "tunable_tier": 2, "transform": "sigmoid", "category": "ccn_activation", "reference": "Andreae (2009) AOD-CCN power law", "shape": None},
packages/atmosphere/legoesm/atmosphere/physics/microphysics/aerosol_activation.py:82:    """Configuration for the Andreae (2009) AOD → CCN inversion.
packages/atmosphere/legoesm/atmosphere/physics/microphysics/aerosol_activation.py:88:        (0.0027, 0.640 from Andreae 2009 Fig. 1).  Override only to
packages/atmosphere/legoesm/atmosphere/physics/microphysics/aerosol_activation.py:112:    Inverts the Andreae (2009) global power-law fit
packages/atmosphere/legoesm/atmosphere/physics/microphysics/aerosol_activation.py:148:    number via :func:`ccn_from_aod` (Andreae 2009) and broadcast across all
packages/core/legoesm/surface_albedo.py:192:def snow_albedo(
packages/land/legoesm/land/leaf_biophysics.py:48:# independent; consumers that add Kc to a mole-fraction Ci already work in
packages/land/legoesm/land/boundary_data/ec_site.py:66:# --- Diagnostic moisture-stress proxy defaults (texture-independent) -----------
packages/land/legoesm/land/config.py:313:    independent booleans, ``slab_land_active`` and ``use_multilayer_land``,
packages/land/legoesm/land/land_use_change.py:249:    annual and independent of the sub-daily biophysics scan), area-weighting each
packages/land/legoesm/land/soil_hydraulics.py:19:``tests/land/unit/test_soil_hydraulics_faithful.py`` against an independent
packages/land/legoesm/land/boundary_data/step_updater.py:298:        # CLM5 default table is data-independent, so XLA hoists it out of the loop).
packages/land/legoesm/land/pedotransfer.py:29:to round-off (rel 1e-9) against an independent scalar reimplementation and
packages/land/legoesm/land/snow_budget.py:118:    ``activation_K = 0`` returns exactly 1.0 (the temperature-independent clock).
packages/land/legoesm/land/boundary_data/_internals.py:29:KN_DEFAULT = 0.3           # nitrogen extinction coefficient [-]
packages/land/legoesm/land/state.py:27:    # step.  Independent of any externally prescribed
packages/land/legoesm/land/snow_bands.py:227:    independent of the forcing's phase convention).  Total precip is conserved per
packages/land/legoesm/land/scm.py:6:history — so the land model can be exercised **independently of the coupler**
packages/land/legoesm/land/lmip_forcing.py:27:components must stay independent of each other (import-linter contract).
packages/land/legoesm/land/lmip_forcing.py:64:    helper's result independent of the caller's site longitude (which
packages/land/legoesm/land/canopy/clm_ml_backend/clm_src_cpl/lnd_comp_nuopc.py:25:    performs domain/grid setup independent of decomposition bounds, while
2208:    # (``clm_surfdata_path``) is required on compute nodes with no outbound
2210:    surfdata_path = config.clm_surfdata_path
2211:    if surfdata_path:
2214:        if not os.path.exists(surfdata_path):
2216:                f"clm_surfdata_path={surfdata_path!r} does not exist; stage the "
2219:        surfdata_file = surfdata_path
2439:        _sd_path = getattr(config, "surfdata_path", "")
5095:            #   2. ``surfdata_path`` — harmonized legoesm_surfdata (per-column
5102:            surfdata_path = getattr(self.config, "surfdata_path", "")
5115:            elif surfdata_path:
5117:                    surfdata_path, lat_albedo
5119:            elif getattr(self.config, "clm_surfdata_path", ""):
5132:                    surfdata_path=self.config.clm_surfdata_path,
5425:                load_transient_cover_on_columns, clm_provider_rebuild,
5427:            _cover, _years = load_transient_cover_on_columns(
5751:    def _surfdata_land_albedo(self, surfdata_path: str, lat_albedo):
5787:            surfdata_path, self._surface_data_grid(), land_cfg, float(self.config.start_day),
5815:            f"  Land albedo: SURFDATA ({surfdata_path}); "

exec
/usr/bin/bash -lc "nl -ba packages/tools/legoesm/forcing/external.py | sed -n '2460,2518p'; nl -ba packages/tools/legoesm/forcing/external.py | sed -n '878,893p'; rg -n 'orbit_year|1950|2000' packages/atmosphere/legoesm/atmosphere/physics/radiation/solar.py | head -25; rg -n 'co2_ppmv|co2_vmr' packages/coupler/legoesm | head -55; rg -n -i 'impurit|snicar|black carbon' packages/land --glob '*.py' | head -12" in /glade/u/home/pg2328/legoesm
 succeeded in 0ms:
  2460	def get_solar_forcing_at_time(config: SolarConfig, day: float) -> dict:
  2461	    """Return solar forcing dict with broadband TSI and optional spectral weights.
  2462	
  2463	    Returns
  2464	    -------
  2465	    dict
  2466	        Keys:
  2467	        - ``"tsi"``: float [W/m^2]
  2468	        - ``"solar_fraction_by_gpt"``: jax array (ngpt,) or None
  2469	    """
  2470	    if config.source == "constant":
  2471	        return {"tsi": float(config.S_0), "solar_fraction_by_gpt": None}
  2472	
  2473	    if config.source not in ("file", "spectral_file"):
  2474	        raise ValueError(f"Unknown solar source: {config.source!r}")
  2475	
  2476	    if not config.path:
  2477	        raise ValueError(
  2478	            f"SolarConfig.path must be set when source={config.source!r}",
  2479	        )
  2480	
  2481	    # Map simulation day (days since ``start_year-01-01``) onto the
  2482	    # file's absolute time axis.  CF ``units`` on the time variable
  2483	    # (e.g. CMIP6 spectral-solar "days since 1850-01-01") anchor the
  2484	    # conversion; units-less test fixtures fall through to
  2485	    # ``file_day = sim_day`` (file times interpreted in simulation
  2486	    # reference).  This replaces the earlier hardcoded
  2487	    # ``_REF_YEAR = 1850`` assumption, which broke units-less files.
  2488	    anchor = _load_time_anchor(config.path)
  2489	    abs_day = _simday_to_file_day(day, config.start_year, anchor)
  2490	
  2491	    if config.source == "file":
  2492	        times, data = _load_timeseries(config.path, (config.tsi_var,))
  2493	        return {"tsi": _interp_1d(times, data[config.tsi_var], abs_day), "solar_fraction_by_gpt": None}
  2494	
  2495	    if config.source == "spectral_file":
  2496	        times, tsi_series, spec_series = _load_time_gpt(
  2497	            config.path,
  2498	            config.tsi_var,
  2499	            config.spectral_var,
  2500	        )
  2501	        if tsi_series is None:
  2502	            # Was a silent fall-back to config.S_0: a file without its TSI
  2503	            # variable ran a constant sun while claiming a transient one.
  2504	            raise ValueError(
  2505	                f"Solar TSI variable {config.tsi_var!r} not found in "
  2506	                f"{config.path!r} (solar_source='spectral_file')",
  2507	            )
  2508	        tsi_val = _interp_1d(times, tsi_series, abs_day)
  2509	        spec = _interp_2d_time(times, spec_series, abs_day)
  2510	        spec = np.clip(spec, 0.0, None)
  2511	        # CMIP6 solar files store one fraction per RRTMG-SW band (14 bands).
  2512	        # The RRTMG solver expects one fraction per g-point (112 g-points).
  2513	        # Expand only when the loaded spectral length matches the table's
  2514	        # ``n_bands`` EXACTLY.  Loose gates like ``shape[1] < 50`` catch
  2515	        # unit fixtures with 2 gpoints and crash the expansion because
  2516	        # ``bnd_limits_gpt`` indexes beyond the input array (Codex
  2517	        # review 2026-04-24); the strict match makes the heuristic
  2518	        # robust to arbitrary-sized inputs.
   878	    var = ds[ext_name]
   879	    ext = np.asarray(var.values, dtype=np.float64)  # [1/km]
   880	    dims = list(var.dims)
   881	
   882	    # B1: single-scattering albedo -> absorption fraction (1-omega) per band.
   883	    if kind == "sw":
   884	        omega = 0.0  # SW slot wants extinction, not absorption
   885	    elif "omega_earth" in ds.data_vars:
   886	        omega = np.clip(
   887	            np.asarray(ds["omega_earth"].values, dtype=np.float64), 0.0, 1.0,
   888	        )
   889	    else:
   890	        omega = np.zeros_like(ext)  # sulfate LW omega ~ 0 approximation
   891	    abs_ext = ext * (1.0 - omega)  # absorption extinction [1/km]
   892	
   893	    alt_name = next(
96:    Defaults are the present-day (≈ year 2000) values from
packages/coupler/legoesm/coupler/surface_exchange.py:107:        co2_ppmv=jnp.array(config.co2_ppmv_default),
packages/coupler/legoesm/coupler/config.py:24:            "co2_ppmv_default": "forcing: default atmospheric CO2 [ppmv]",
packages/coupler/legoesm/coupler/config.py:107:    co2_ppmv_default: float = 400.0   # Default CO2 concentration
packages/coupler/legoesm/driver/coupled_esm_driver.py:1429:        self._co2_mass = (jnp.asarray(cfg.co2_ppmv_init * 1.0e-6, jnp.float64)
packages/coupler/legoesm/driver/coupled_esm_driver.py:1432:        logger.info(f"  CO2 box: init={cfg.co2_ppmv_init:.1f} ppmv "
packages/coupler/legoesm/driver/coupled_esm_driver.py:1933:            co2_ppmv = jnp.full_like(p_s, self._co2_box_vmr() * 1.0e6)
packages/coupler/legoesm/driver/coupled_esm_driver.py:1936:            from legoesm.core.coupling_fields import land_co2_ppmv
packages/coupler/legoesm/driver/coupled_esm_driver.py:1937:            co2_ppmv = jnp.full_like(p_s, land_co2_ppmv(
packages/coupler/legoesm/driver/coupled_esm_driver.py:1938:                self._atm.radiation_ghg_vmr(day), self.atm_config.co2_ppmv))
packages/coupler/legoesm/driver/coupled_esm_driver.py:1957:            co2_ppmv=co2_ppmv,
packages/coupler/legoesm/driver/coupled_esm_driver.py:2511:    def _co2_vmr_from_mass(self, co2_mass, p_s):
packages/coupler/legoesm/driver/coupled_esm_driver.py:2518:        return self._co2_vmr_from_mass(self._co2_mass, self._atm.state.p_s.data)
packages/coupler/legoesm/driver/coupled_esm_driver.py:2537:            self._atm._co2_vmr_override = float(self._co2_box_vmr())
packages/coupler/legoesm/driver/coupled_esm_driver.py:2864:            diag["co2_ppmv_mean"] = float(host[idx]) * 1e6
packages/coupler/legoesm/driver/model_driver.py:1339:        # Radiation's CO2 (daily GHG file, else cfg.co2_ppmv).
packages/coupler/legoesm/driver/model_driver.py:1340:        co2_ppmv=jnp.full_like(T_air, co2),
packages/coupler/legoesm/driver/model_driver.py:1833:                          "co2_ppmv", "has_radiation", "has_precipitation",
packages/coupler/legoesm/driver/model_driver.py:5912:            self.config = cfg._replace(co2_ppmv=co2, ch4_ppbv=ch4, n2o_ppbv=n2o)
packages/coupler/legoesm/driver/model_driver.py:5922:                co2_ppmv=cfg.co2_ppmv,
packages/coupler/legoesm/driver/model_driver.py:5932:                f"  GHG external: CO2={ghg_init['co2_ppmv']:.1f}ppmv, "
packages/coupler/legoesm/driver/model_driver.py:5940:                co2_ppmv=cfg.co2_ppmv,
packages/coupler/legoesm/driver/model_driver.py:6214:                    "co2_ppmv": co2,
packages/coupler/legoesm/driver/model_driver.py:6220:        # driver runs a prognostic CO2 tracer it sets ``self._co2_vmr_override``
packages/coupler/legoesm/driver/model_driver.py:6225:        co2_vmr_override = getattr(self, "_co2_vmr_override", None)
packages/coupler/legoesm/driver/model_driver.py:6226:        if (co2_vmr_override is not None
packages/coupler/legoesm/driver/model_driver.py:6228:            ghg_vmr = {**(ghg_vmr or {}), "co2": co2_vmr_override}
packages/coupler/legoesm/driver/model_driver.py:13592:        # (``cfg.co2_ppmv`` etc. — set at setup by ``ghg_at_year`` / the
packages/coupler/legoesm/driver/model_driver.py:15415:                    land_co2_ppmv(_forcing_daily.get("ghg_vmr"), cfg.co2_ppmv),
packages/coupler/legoesm/driver/model_driver.py:15425:            from legoesm.core.coupling_fields import land_co2_ppmv
packages/coupler/legoesm/driver/config.py:912:    co2_ppmv: float = 415.0
packages/coupler/legoesm/driver/config.py:5822:            co2_ppmv=amip_cfg.co2_ppmv,
packages/coupler/legoesm/driver/config.py:6044:            co2_ppmv=self.co2_ppmv,
packages/coupler/legoesm/driver/coupled_config.py:47:    co2_ppmv_init : float
packages/coupler/legoesm/driver/coupled_config.py:124:    co2_ppmv_init: float = 415.0
packages/coupler/legoesm/driver/physics_pipeline.py:167:        co2_ppmv=None,
packages/coupler/legoesm/driver/physics_pipeline.py:173:        # the land canopy reads the same value (land_co2_ppmv).
packages/coupler/legoesm/driver/physics_pipeline.py:174:        self.co2_ppmv = co2_ppmv
packages/coupler/legoesm/driver/physics_pipeline.py:927:            AtmToSurface, land_co2_ppmv, lowest_level_height)
packages/coupler/legoesm/driver/physics_pipeline.py:960:        _co2 = land_co2_ppmv(ghg_vmr_override, self.co2_ppmv)
packages/coupler/legoesm/driver/physics_pipeline.py:962:            raise ValueError("PhysicsPipeline.co2_ppmv is unset; the land canopy "
packages/coupler/legoesm/driver/physics_pipeline.py:971:            cos_zenith=_cosz, co2_ppmv=_co2 * ones,
packages/coupler/legoesm/driver/physics_pipeline.py:3962:        co2_ppmv=config.co2_ppmv,
packages/coupler/legoesm/driver/physics_pipeline.py:5810:        co2_ppmv=config.co2_ppmv,
packages/coupler/legoesm/driver/earth_system_driver.py:277:            land_co2_ppmv, lowest_level_height,
packages/coupler/legoesm/driver/earth_system_driver.py:296:            co2_ppmv=jnp.full_like(p_s, land_co2_ppmv(
packages/coupler/legoesm/driver/earth_system_driver.py:297:                self._atm.radiation_ghg_vmr(day), self.config.co2_ppmv)),

exec
/usr/bin/bash -lc "nl -ba packages/coupler/legoesm/driver/model_driver.py | sed -n '5400,5465p'; nl -ba packages/core/legoesm/surface_albedo.py | sed -n '192,240p'; nl -ba packages/land/legoesm/land/state.py | head -80; rg -n '78|num.*pft|numpft' /glade/derecho/scratch/pg2328/cesm_src/../components/clm/src/main/pftconMod.F90 /glade/derecho/scratch/pg2328/cesm_src/CLM/src/main/pftconMod.F90; nl -ba packages/tools/legoesm/forcing/external.py | sed -n '1697,1720p'; rg -n 'solar.*month|monthly' scripts/data/adapt_cmip6_solar.py" in /glade/u/home/pg2328/legoesm
 exited 1 in 0ms:
  5400	        self.physics.land_ml_cfg = cfg
  5401	        self.physics.land_ml_params = params
  5402	        self.physics.land_ml_lat = jnp.asarray(lat_rad, dtype=storage_dtype)
  5403	        self.physics.land_ml_doy = 0.0
  5404	        # CONCRETE dynamics timestep [s].  The jitted segment passes ``dt`` as a
  5405	        # TRACER, but the CLM-ML canopy needs a concrete dt to resolve its static
  5406	        # ML sub-step count (num_ml_steps = ceil(dt/dtime_ml)).  The timestep is
  5407	        # fixed, so the concrete config dt is numerically identical to the traced
  5408	        # one; the pipeline uses it only on the clm_ml path (byte-identical for
  5409	        # simple_seb / two_leaf, which keep the traced dt).
  5410	        self.physics.land_ml_dt = float(self.config.dycore.dt)
  5411	
  5412	        # Transient land-use cover (LULC): load the annual cover series onto the SAME
  5413	        # model columns (identical _nearest_regrid targets => cell-for-cell aligned
  5414	        # with surface_map) and build the per-year vegetation-param rebuild (soil /
  5415	        # LAI frozen).  ``params`` above is the year=None fallback baked on the
  5416	        # pipeline; the run loop re-weights per segment via
  5417	        # ``_transient_land_ml_params`` and passes the result as a TRACED
  5418	        # SegmentForcing arg so the jitted step reads the evolving cover (not the
  5419	        # closure-baked attribute).  ``include_soil_albedo=True`` mirrors
  5420	        # clm_multilayer_setup (which weights the soil-colour albedo), so the
  5421	        # rebuild at the base cover reproduces ``params`` exactly.
  5422	        self._land_cover_transient = None
  5423	        if getattr(self.config, "transient_land_cover", False):
  5424	            from legoesm.land.clm_surface_map import (
  5425	                load_transient_cover_on_columns, clm_provider_rebuild,
  5426	            )
  5427	            _cover, _years = load_transient_cover_on_columns(
  5428	                self.config.land_cover_surfdata, lat_deg, lon_deg)
  5429	            _rebuild = clm_provider_rebuild(
  5430	                surface_map, variant="multilayer", include_soil_albedo=True)
  5431	            self._land_cover_transient = (_cover, _years, _rebuild)
  5432	            logger.info(
  5433	                "  Land tile: TRANSIENT cover ACTIVE (%d cover years, %d..%d) "
  5434	                "from %s", int(_years.shape[0]), int(_years[0]), int(_years[-1]),
  5435	                self.config.land_cover_surfdata,
  5436	            )
  5437	
  5438	        ncol = lat_deg.shape[0]
  5439	        T_init = _flat_cols(self.state.T.data[..., -1]).astype(storage_dtype)
  5440	        # Aridity-aware cold-start soil moisture (#730 / #837).  Seed theta from
  5441	        # the near-surface RH of the IC atmosphere, mapped into the per-column
  5442	        # plant-available range [theta_wp, theta_fc] the tile's beta reads
  5443	        # (params.theta_wp/theta_fc from clm_multilayer_setup).  The legacy
  5444	        # moisture-uniform 0.5*theta_sat seed leaves subtropical deserts
  5445	        # rainforest-wet, so a hot bare-soil skin drives a runaway
  5446	        # potential-evaporation blowup (~day 8).  RH here uses the model's own
  5447	        # saturation_mixing_ratio -- the SAME law the land bulk flux uses -- so it
  5448	        # is consistent with the running physics (q_v is the atmospheric lowest
  5449	        # level; p_s is a close proxy for the lowest-level pressure).  Fall back to
  5450	        # the frac*theta_sat uniform seed (``land_soil_moisture_init_frac``) only
  5451	        # when the IC carries no q_v tracer (the aridity map needs RH).
  5452	        _qv = self.q_v  # canonical tracer store: raw (...,nlev) array, same column
  5453	        # layout as self.state.T.data; populated by both the analytical and ERA5 IC.
  5454	        _soil_init = getattr(self.config, "land_soil_init", "aridity")
  5455	        if _soil_init == "saturation_fraction":
  5456	            # Seed as a fraction of POROSITY, so the soil can start above field
  5457	            # capacity — the way to keep a run out of the dry-soil attractor
  5458	            # (low soil water -> weak evaporation -> dry boundary layer -> weaker
  5459	            # evaporation).  The aridity seed below cannot do this: it caps at
  5460	            # field capacity by construction.
  5461	            #
  5462	            # Held just below saturation because the van-Genuchten retention is
  5463	            # singular AT saturation — psi_from_theta needs theta < theta_sat —
  5464	            # and just above the residual for the same reason at the dry end.
  5465	            # Same guard the offline calibrator uses on its own seed.
   192	def snow_albedo(
   193	    snow_age: jnp.ndarray,
   194	    config: LandAlbedoConfig = LandAlbedoConfig(),
   195	    cos_zenith: jnp.ndarray | None = None,
   196	) -> jnp.ndarray:
   197	    """Compute snow albedo as a function of snow age (and optionally solar zenith).
   198	
   199	    Diffuse (age-decayed) part::
   200	
   201	        alpha_diffuse = alpha_snow_min + (alpha_snow_max - alpha_snow_min) * exp(-t/tau)
   202	
   203	    the exponential age decay standing in for grain-size metamorphism (fresh snow
   204	    bright, aged/firn snow darker).  When ``cos_zenith`` is supplied, the BATS /
   205	    Dickinson (1983) solar-zenith brightening (as in CLM) is added — snow is brighter
   206	    at a low sun (high zenith), the leading control on the polar ice-sheet shortwave
   207	    budget that an overhead-sun albedo misses::
   208	
   209	        alpha = alpha_diffuse + factor * f(mu) * (1 - alpha_diffuse)
   210	        f(mu) = max( (1/b) * ((1 + b) / (1 + 2 b mu) - 1), 0 ),  mu = clip(cos_zenith, 0, 1)
   211	
   212	    ``f`` is 0 at overhead sun (mu = 1) and rises to 1 at grazing incidence (mu = 0),
   213	    and the denominator ``1 + 2 b mu >= 1`` is never zero — differentiable everywhere.
   214	
   215	    Parameters
   216	    ----------
   217	    snow_age : jnp.ndarray
   218	        Time since last snowfall [s].
   219	    config : LandAlbedoConfig
   220	    cos_zenith : jnp.ndarray or None
   221	        Cosine of the solar zenith angle.  ``None`` disables the zenith brightening
   222	        (returns the diffuse albedo — the legacy behaviour).
   223	
   224	    Returns
   225	    -------
   226	    alpha_snow : jnp.ndarray
   227	        Snow albedo.
   228	    """
   229	    alpha_diffuse = config.alpha_snow_min + (
   230	        config.alpha_snow_max - config.alpha_snow_min
   231	    ) * jnp.exp(-snow_age / config.tau_snow_decay)
   232	    if cos_zenith is None:
   233	        return alpha_diffuse
   234	    mu = jnp.clip(cos_zenith, 0.0, 1.0)
   235	    b = config.snow_zenith_b
   236	    f_zen = jnp.maximum((1.0 / b) * ((1.0 + b) / (1.0 + 2.0 * b * mu) - 1.0), 0.0)
   237	    return alpha_diffuse + config.snow_zenith_factor * f_zen * (1.0 - alpha_diffuse)
   238	
   239	
   240	def snow_cover_fraction(
     1	"""Land model state container."""
     2	
     3	from __future__ import annotations
     4	
     5	from typing import Any, NamedTuple
     6	
     7	import jax
     8	
     9	from legoesm.core.field import Field
    10	
    11	
    12	class LandState(NamedTuple):
    13	    """Slab land state.
    14	
    15	    Field shape is grid-dependent: (ncol,) on the column/SCM path,
    16	    (6, n, n) on the cubed-sphere.
    17	    """
    18	    T_soil: Field          # Soil slab temperature [K]
    19	    W_bucket: Field        # Bucket soil moisture [kg/m2]
    20	    snow_depth: Field      # Snow water equivalent [kg/m2]
    21	    snow_age: Field        # Time since last snowfall [s]
    22	    runoff: jax.Array | None = None  # Surface runoff [kg/m2/s]
    23	    # 30-day exponential moving average of near-surface air temperature
    24	    # in [°C] — feeds the Leuning Vcmax growth-temperature acclimation
    25	    # term when the two-leaf canopy surface scheme is active.  ``None``
    26	    # by default; populated by ``init_*_land_state`` and advanced each
    27	    # step.  Independent of any externally prescribed
    28	    # ``CanopyLandParams.TgC`` (which takes precedence when both are set).
    29	    TgC: jax.Array | None = None
    30	
    31	
    32	class MultiLayerLandState(NamedTuple):
    33	    """Multi-layer land state (Task 8).
    34	
    35	    2D fields have shape (ncol,).
    36	    3D fields have shape (ncol, n_soil_layers).
    37	    """
    38	    T_soil: jax.Array          # Soil temperature [K], (ncol, n_soil_layers)
    39	    psi_soil: jax.Array        # Soil matric potential [m], (ncol, n_soil_layers)
    40	    theta_soil: jax.Array      # Volumetric water content [m3/m3], (ncol, n_soil_layers)
    41	    runoff_surface: jax.Array  # Surface runoff [kg/m2/s], (ncol,)
    42	    runoff_subsurface: jax.Array  # Subsurface runoff [kg/m2/s], (ncol,)
    43	    snow_depth: jax.Array      # Snow water equivalent [kg/m2], (ncol,)
    44	    snow_age: jax.Array        # Time since last snowfall [s], (ncol,)
    45	    # 30-day exponential moving average of near-surface air temperature
    46	    # in [°C] — see ``LandState.TgC``.  Optional; ``None`` by default.
    47	    TgC: jax.Array | None = None
    48	    # Surface ponding depth [m], (ncol,) — a coupled surface cell (ParFlow/CliMA
    49	    # overland store): excess precip ponds, infiltrates on later steps, and
    50	    # overflows to runoff above ``pond_max``.  ``None`` (legacy) is zero ponding.
    51	    surface_water: jax.Array = None
    52	    # Sub-grid elevation-band SWE [kg/m2], (ncol, n_bands) — present iff the
    53	    # elevation-band snow scheme is enabled (``config.elev_bands``); ``None``
    54	    # (legacy) runs the single cell-mean snowpack.  ``snow_depth`` always carries
    55	    # the area-weighted aggregate for diagnostics/restart compatibility.
    56	    snow_bands: jax.Array = None
    57	    # Per-band snow age [s], (ncol, n_bands) — companion to ``snow_bands`` (fresh
    58	    # snow rejuvenates only its own band's albedo while perennial firn keeps aging).
    59	    # ``None`` (legacy) uses the single cell-mean ``snow_age`` clock.
    60	    snow_age_bands: jax.Array = None
    61	    # Sub-grid elevation-band firn/glacier-ice reservoir [kg/m2], (ncol, n_bands) —
    62	    # companion to ``snow_bands`` (gap 4): seasonal snow above the snow cap firnifies
    63	    # into this store, which discharges slowly (delayed glacier outflow) and exposes
    64	    # dark ablation ice where the seasonal snow melts off.  ``None`` (legacy / bands
    65	    # off) has no perennial-ice reservoir.
    66	    ice_bands: jax.Array = None
    67	    # Prognostic state for the CLM-ML-JAX multilayer canopy scheme.
    68	    # Holds the ``mlcanopy_type`` instance carried forward between steps.
    69	    # ``None`` when the CLM-ML canopy scheme is not active.
    70	    canopy_state: Any | None = None
    71	    # Intercepted canopy water store [kg m-2], (ncol,) — the shared canopy
    72	    # interception scheme (``config.interception``; two-leaf / SimpleSEB path).
    73	    # ``None`` (legacy / interception off) carries no store.  The CLM-ML canopy
    74	    # keeps its own internal ``h2ocan`` store inside ``canopy_state`` and does
    75	    # NOT use this field.
    76	    W_canopy: jax.Array | None = None
    77	    # Warm-start cache for the two-leaf canopy Newton closure: the last
    78	    # CONVERGED solution per column, ``(ncol, 6)``, NaN where a column has never
    79	    # converged (which the canopy reads as "cold start").  NOT a prognostic
    80	    # variable — it carries no physics, only the seed the iteration starts from
rg: /glade/derecho/scratch/pg2328/cesm_src/../components/clm/src/main/pftconMod.F90: No such file or directory (os error 2)
rg: /glade/derecho/scratch/pg2328/cesm_src/CLM/src/main/pftconMod.F90: No such file or directory (os error 2)
  1697	def _load_ghg_annual_file(path: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
  1698	    """Load CMIP6-style annual global-mean GHG file (Zarr or NetCDF).
  1699	
  1700	    Expected format: variables CO2, CH4, N2O, CFC_11, CFC_12 with
  1701	    dimensions (time, lat, lon) where lat=1, lon=1.  The time axis
  1702	    uses units ``"year as %Y.%f"`` giving fractional year values
  1703	    (e.g. 1850.0, 1851.0, ...).
  1704	
  1705	    Returns
  1706	    -------
  1707	    (years, data) where years is shape (N,) and data maps variable
  1708	    names to 1-D numpy arrays of length N.  Values are normalized to
  1709	    DIMENSIONLESS mole fractions (mol/mol) using each variable's
  1710	    ``units`` attribute (audit 2026-06-11) so the caller can convert to
  1711	    ppmv/ppbv/pptv with fixed factors regardless of the file's storage
  1712	    unit.  An unrecognised ``units`` string falls back to a magnitude
  1713	    heuristic (CO2-scale value < 1 ⇒ already mole fraction) and logs a
  1714	    warning rather than silently feeding a 1e6-wrong concentration to
  1715	    radiation.
  1716	    """
  1717	    ds = _open_forcing_dataset(path)
  1718	    if "time" not in ds.dims:
  1719	        ds.close()
  1720	        raise ValueError(f"GHG file {path!r} has no 'time' dimension")

exec
/usr/bin/bash -lc "nl -ba config/amip/amip_production.sh | sed -n '140,180p'; nl -ba packages/land/legoesm/land/clm_surface_map.py | sed -n '125,145p'; nl -ba packages/atmosphere/legoesm/atmosphere/physics/radiation/solar.py | sed -n '90,115p'; nl -ba packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py | sed -n '830,867p'; rg -n 'bndtvs|orb_iyear|orb_mode' /glade/derecho/scratch/pg2328/cesm_src/CAM/bld/namelist_files/namelist_defaults_cam.xml | head -12; git status --porcelain" in /glade/u/home/pg2328/legoesm
 succeeded in 62ms:
   140	  --ic-path "${ERA5_IC}"
   141	  --forcing-path "${SST}"
   142	  --sic-path "${SIC}"
   143	  --sst-offset "${AMIP_SST_OFFSET}"
   144	  --solar-file "${SOLAR}"
   145	  --ozone-file "${OZONE}"
   146	  --ghg-file "${GHG}"
   147	  --aerosol-file "${AEROSOL}"
   148	  --aerosol-ccn-file "${AEROSOL_CCN}"
   149	  --volcanic-aerosol-file "${VOLCANIC}"
   150	  --topography "${ETOPO}"
   151	  --clm-surfdata-path "${CLM_SURFDATA}"
   152	)
   153	# Volcanic LONGWAVE absorption (ext_earth band; post-eruption stratospheric
   154	# heating). ON by default for the campaign (decision 2026-07-21): reader
   155	# validated against the merged 1979-2014 file (Pinatubo-peak LW AOD 0.020 vs
   156	# 0.0005 quiet-1979; El Chichon visible). LW AOD ~0 outside 1982-84/1991-93,
   157	# so quiet years are unaffected; the pilot decade exercises El Chichon before
   158	# the full 36-yr. No forward run has exercised the LW channel yet — first
   159	# post-1982 pilot output is the validation gate. Opt OUT (e.g. to reproduce
   160	# pre-2026-07-21 SW-only runs) with AMIP_VOLCANIC_LW=0.
   161	if [[ "${AMIP_VOLCANIC_LW:-1}" == "1" ]]; then
   162	  AMIP_PATH_FLAGS+=( --volcanic-aerosol-lw )
   163	fi
   164	# CLM-derived land mask (see AMIP_LAND_MASK above). Side-effect scope audited
   165	# 2026-07-22: with use_multilayer_land=true the ONLY climate-relevant change is
   166	# f_land itself (the forced slab_land_active is shadowed by the multilayer tile
   167	# branch; surfdata albedo needs --surfdata, which this launcher never passes).
   168	if [[ -n "${AMIP_LAND_MASK}" ]]; then
   169	  AMIP_PATH_FLAGS+=( --land-mask-file "${AMIP_LAND_MASK}" )
   170	fi
   171	# Subgrid orography for the orographic GWD launch (see AMIP_SSO above, #1514).
   172	# Without this flag the run trips the loud scalar-fallback warning (PR #1540)
   173	# and reproduces the pseudo-mountain climate. Opt-out is DELIBERATE only.
   174	if [[ -n "${AMIP_SSO}" ]]; then
   175	  AMIP_PATH_FLAGS+=( --default-subgrid-orography-file "${AMIP_SSO}" )
   176	fi
   177	
   178	# Refuse at job start, before any compile or step time is spent, when the
   179	# aerosol / CCN aerosol / volcanic product is not staged: the run would
   180	# otherwise die at the first forcing load.  Call it right after sourcing:
   125	    0.088, 1.706, 1.469, 1.414, 1.631, 1.675, 1.527, 1.408, 1.088,
   126	    0.716, 0.622, 0.723, 0.457, 0.507, 0.504, 0.488, 0.477)  # const-ok: baked per-PFT calibration table (LMIP biophysics 2-deg), not a physical constant
   127	_BIOPHYS_LMIP_PFT_THETA_WP = (
   128	    0.0846, 0.0915, 0.0942, 0.0907, 0.1154, 0.0988, 0.1162, 0.1062, 0.1035,
   129	    0.0988, 0.0838, 0.0839, 0.0865, 0.0874, 0.0869, 0.0822, 0.0988)  # const-ok: baked per-PFT calibration table (LMIP biophysics 2-deg)
   130	_BIOPHYS_LMIP_PFT_THETA_FC = (
   131	    0.1706, 0.2249, 0.2351, 0.2253, 0.2890, 0.2492, 0.2918, 0.2676, 0.2599,
   132	    0.2635, 0.1995, 0.2045, 0.2164, 0.2171, 0.2193, 0.2123, 0.2469)  # const-ok: baked per-PFT calibration table (LMIP biophysics 2-deg)
   133	
   134	# Scalar snow / soil albedo settings of the SAME calibration. These are
   135	# LandAlbedoConfig field values; the offline LMIP consumes them through
   136	# ``lmip_config`` (physics.albedo), and the coupled path applies them in
   137	# ``land.config.apply_biophysics_lmip_two_leaf``.
   138	_BIOPHYS_LMIP_SNOW_ALBEDO_MAX = 0.8077    # [-] fresh-snow albedo
   139	_BIOPHYS_LMIP_SNOW_ALBEDO_MIN = 0.5207    # [-] aged-snow albedo
   140	_BIOPHYS_LMIP_SNOW_DCRIT = 15.4229        # [kg/m2] Niu-Yang half-cover SWE
   141	_BIOPHYS_LMIP_SNOW_TAU_S = 317424.96      # [s] snow-age decay (3.674 days)
   142	_BIOPHYS_LMIP_SOIL_DRY_BOOST = 0.1458     # [-] dry-soil brightening
   143	_BIOPHYS_LMIP_GLACIER_ALBEDO = (0.8178, 0.6178)   # [-] (visible, near-infrared)
   144	
   145	
    90	
    91	class OrbitalParameters(NamedTuple):
    92	    """Earth orbital elements for realistic (non-circular) insolation.
    93	
    94	    Berger (1978) convention (as in CESM ``shr_orb_mod`` / climlab): the
    95	    longitude of perihelion is measured from the moving vernal equinox.
    96	    Defaults are the present-day (≈ year 2000) values from
    97	    ``legoesm.constants``.  ``eccentricity = 0`` recovers a circular orbit
    98	    (the AMIP-II realistic insolation uses the present-day ellipse).
    99	
   100	    ``solver`` picks how the true solar longitude is obtained:
   101	    ``"berger_series"`` (CESM ``shr_orb_decl``'s cubic series in e, matched
   102	    to the compiled oracle; accurate only for small e: 1.1e-7 rad at present
   103	    day, 0.008 deg at e=0.1, 0.68 deg at e=0.3) or ``"kepler"`` (exact Newton
   104	    solve of Kepler's equation, valid to high e).  Daily-mean insolation
   105	    still holds the declination AND the Earth-Sun distance factor fixed over
   106	    the day, which degrades at high e near perihelion where both change fast
   107	    (at e=0.9 (a/r)^2 swings by several per cent within the perihelion day).
   108	    """
   109	
   110	    eccentricity: float = constants.orbital_eccentricity
   111	    obliquity_deg: float = constants.orbital_obliquity_deg
   112	    long_perihelion_deg: float = constants.orbital_long_perihelion_deg
   113	    solver: str = "berger_series"
   114	
   115	
   830	      # ponytail: the ~2 W/m2 SW absorbed above the model top LEAVES the
   831	      # column budget (TOA net - surface net != sum of model-layer heating),
   832	      # exactly as in CAM.  Not applied to the optics-only (3D MC) paths,
   833	      # whose per-layer output must stay on the model grid.
   834	      overhead = (bool(getattr(config, "overhead_layer", False))
   835	                  and not (sw_optical_field_only or lw_optical_field_only))
   836	      # LW level temperatures from the model column (CAM6 rrtmg_state).
   837	      T_lev = cam_interface_temperature(T, p_full, p_half, sfc_temperature)
   838	      if overhead:
   839	          # CAM's extra layer (rrtmg_state.F90 L122-136, kk = max(k-1, 1)):
   840	          # both of its interfaces take tint(1), the top-layer temperature.
   841	          T_lev = jnp.concatenate([T_lev[:, :1], T_lev], axis=1)
   842	          p_top = p_half[:, :1]
   843	          p_lid = jnp.minimum(p_top, 1.0)  # coeff-ok: 1 Pa lid (CAM-like ~0 Pa top)
   844	
   845	          def _top(a, k0):
   846	              return None if a is None else jnp.concatenate([k0, a], axis=1)
   847	
   848	          def _zero(a):
   849	              return None if a is None else _top(a, jnp.zeros_like(a[:, :1]))
   850	
   851	          if o3_vmr is None:  # resolve model ozone first: fallback = o3(k0)
   852	              o3_vmr = _standard_o3_profile(p_full)
   853	          p_half = _top(p_half, p_lid)
   854	          p_full = _top(p_full, 0.5 * (p_lid + p_top))
   855	          T = _top(T, T[:, :1])
   856	          q_v = _top(q_v, q_v[:, :1])
   857	          o3_vmr = _top(o3_vmr, o3_vmr[:, :1] if o3_top_vmr is None
   858	                        else jnp.asarray(o3_top_vmr).astype(
   859	                            _table_dtype).reshape(ncol, 1))
   860	          cloud_path_liq, cloud_path_ice = _zero(cloud_path_liq), _zero(cloud_path_ice)
   861	          cloud_path_snow = _zero(cloud_path_snow)
   862	          if cloud_d_eff_snow is not None:
   863	              cloud_d_eff_snow = _top(cloud_d_eff_snow, cloud_d_eff_snow[:, :1])
   864	              cloud_snow_weight = _top(cloud_snow_weight, cloud_snow_weight[:, :1])
   865	          cloud_path_liq_lw = _zero(cloud_path_liq_lw)
   866	          cloud_path_ice_lw = _zero(cloud_path_ice_lw)
   867	          cloud_fraction = _zero(cloud_fraction)
423:<bndtvsox          hgrid="128x256"  >atm/cam/scyc/SOx_emissions_128x256_L2_1850-2000_c040321.nc</bndtvsox>
424:<bndtvsox          hgrid="64x128"   >atm/cam/scyc/SOx_emissions_64x128_L2_c030722.nc</bndtvsox>
425:<bndtvsox          hgrid="32x64"    >atm/cam/scyc/SOx_emissions_32x64_L2_c030722.nc</bndtvsox>
426:<bndtvsox dyn="fv" hgrid="4x5"      >atm/cam/scyc/SOx_emissions_4x5_noncon_c050306.nc</bndtvsox>
1882:<orb_iyear >1990</orb_iyear>
1892:<bndtvs hgrid="0.23x0.31"  sim_year="2000">atm/cam/sst/sst_HadOIBl_bc_0.23x0.31_clim_c061106.nc</bndtvs>
1893:<bndtvs hgrid="0.47x0.63"  sim_year="2000">atm/cam/sst/sst_HadOIBl_bc_0.47x0.63_clim_c061106.nc</bndtvs>
1894:<bndtvs hgrid="0.9x1.25"   sim_year="2000">atm/cam/sst/sst_HadOIBl_bc_0.9x1.25_clim_c040926a.nc</bndtvs>
1895:<bndtvs hgrid="1.9x2.5"    sim_year="2000">atm/cam/sst/sst_HadOIBl_bc_1.9x2.5_clim_c061031.nc</bndtvs>
1896:<bndtvs hgrid="2.5x3.33"   sim_year="2000">atm/cam/sst/sst_HadOIBl_bc_2.5x3.33_clim_c091210.nc</bndtvs>
1897:<bndtvs hgrid="4x5"        sim_year="2000">atm/cam/sst/sst_HadOIBl_bc_4x5_clim_c061031.nc</bndtvs>
1898:<bndtvs hgrid="10x15"      sim_year="2000">atm/cam/sst/sst_HadOIBl_bc_10x15_clim_c050526.nc</bndtvs>
?? .claude/.headroom_wrap_settings.lock
?? c1.pbs
?? data/les_cases/AYOTTE/
?? data/les_cases/GABLS3/
?? data/les_cases/GABLS4/
?? data/les_cases/PROVENANCE_2026-10-15.md
?? docs/datasets.json
?? docs/wb/reviews/2026-11-22_cold_bias_claims_codex.md
?? docs/wb/reviews/2026-11-22_cold_bias_claims_glm.md
?? docs/wb/reviews/2026-11-22_cold_bias_claims_prompt.md
?? docs/wb/reviews/2026-11-22_cold_bias_vote_codex.md
?? docs/wb/reviews/2026-11-22_cold_bias_vote_glm.md
?? docs/wb/reviews/2026-11-22_cold_bias_vote_prompt.md
?? docs/wb/reviews/2026-11-22_forcing_audit_codex.md
?? docs/wb/reviews/2026-11-22_forcing_audit_glm.md
?? docs/wb/reviews/2026-11-22_forcing_audit_prompt.md
?? scripts/cluster/les_suite/derecho_les_case.pbs
?? scripts/cluster/les_suite/les_timing.o7827219

codex
- **C1 — UNSURE.** Annual interpolation is confirmed ([external.py:1822](/glade/u/home/pg2328/legoesm/packages/tools/legoesm/forcing/external.py:1822)); fixed minor-halogen defaults are confirmed ([rrtmgp.py:504](/glade/u/home/pg2328/legoesm/packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:504)). However, the loader supports optional minor halogens ([external.py:1847](/glade/u/home/pg2328/legoesm/packages/tools/legoesm/forcing/external.py:1847)). Without the Levante file, its missing species, concentrations, and both forcing estimates remain unverified.

- **C2 — UNSURE.** Spectral interpolation and rotation are implemented ([external.py:2508](/glade/u/home/pg2328/legoesm/packages/tools/legoesm/forcing/external.py:2508), [external.py:2714](/glade/u/home/pg2328/legoesm/packages/tools/legoesm/forcing/external.py:2714)). Present-day orbital defaults are explicit ([solar.py:96](/glade/u/home/pg2328/legoesm/packages/atmosphere/legoesm/atmosphere/physics/radiation/solar.py:96)). These establish neither identical input values nor the <0.05 W/m² bound.

- **C3 — UNSURE.** Noncyclic ozone interpolation is confirmed ([external.py:2032](/glade/u/home/pg2328/legoesm/packages/tools/legoesm/forcing/external.py:2032)); the 16 samples are averaged into **one** overhead layer ([model_driver.py:6081](/glade/u/home/pg2328/legoesm/packages/coupler/legoesm/driver/model_driver.py:6081)). Product differences are supported; forcing magnitude and exclusion as a primary cause are unmeasured.

- **C4 — AGREE on the optics defect; UNSURE on magnitude.** SW ignores file SSA ([external.py:883](/glade/u/home/pg2328/legoesm/packages/tools/legoesm/forcing/external.py:883)); volcanic and tropospheric OD are combined ([surface_utils.py:841](/glade/u/home/pg2328/legoesm/packages/tools/legoesm/forcing/surface_utils.py:841)) under shared optics ([rrtmgp.py:1118](/glade/u/home/pg2328/legoesm/packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:1118)). A 7% single-scattering absorption fraction does not establish a 5–7% forcing error.

- **C5 — REFUTE the warming inference.** Pressure-thickness distribution and visible-total-AOD CCN are confirmed ([surface_utils.py:489](/glade/u/home/pg2328/legoesm/packages/tools/legoesm/forcing/surface_utils.py:489), [surface_utils.py:845](/glade/u/home/pg2328/legoesm/packages/tools/legoesm/forcing/surface_utils.py:845)). Atmospheric absorption does not determine surface-temperature response. Global rsut tuning cannot constrain regional cloud-mediated cooling.

- **C6 — UNSURE.** Interpolation, freezing floor, and SIC clipping agree ([amip.py:988](/glade/u/home/pg2328/legoesm/packages/tools/legoesm/forcing/amip.py:988)). Different boundary datasets require numerical comparison before declaring equivalence.

- **C7 — UNSURE.** Production selects fixed surfdata; transient cover requires a separate switch ([model_driver.py:5423](/glade/u/home/pg2328/legoesm/packages/coupler/legoesm/driver/model_driver.py:5423)). Neither the global forcing bound nor regional irrelevance follows.

- **C8 — AGREE on missing explicit impurities; REFUTE “therefore brighter.”** Snow albedo depends on age and zenith ([surface_albedo.py:192](/glade/u/home/pg2328/legoesm/packages/core/legoesm/surface_albedo.py:192)), with calibrated brightness parameters ([clm_surface_map.py:138](/glade/u/home/pg2328/legoesm/packages/land/legoesm/land/clm_surface_map.py:138)). Relative brightness requires matched snow states; contribution remains plausible.

- **C9 — AGREE for production:** no prognostic nitrogen pool in the land state ([state.py:32](/glade/u/home/pg2328/legoesm/packages/land/legoesm/land/state.py:32)).

- **Overall — REFUTE the exclusion argument.** These checks do not establish that snow impurities are the sole plausible forcing contributor.

Missed channels: physiological CO₂ effects on canopy exchange ([model_driver.py:1340](/glade/u/home/pg2328/legoesm/packages/coupler/legoesm/driver/model_driver.py:1340)); harmonized canopy/soil optical inputs, beyond CLM PFT cover; topography, land fraction, and subgrid-orography boundary fields ([amip_production.sh:138](/glade/u/home/pg2328/legoesm/config/amip/amip_production.sh:138), [amip_production.sh:168](/glade/u/home/pg2328/legoesm/config/amip/amip_production.sh:168)).
tokens used
86,647
- **C1 — UNSURE.** Annual interpolation is confirmed ([external.py:1822](/glade/u/home/pg2328/legoesm/packages/tools/legoesm/forcing/external.py:1822)); fixed minor-halogen defaults are confirmed ([rrtmgp.py:504](/glade/u/home/pg2328/legoesm/packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:504)). However, the loader supports optional minor halogens ([external.py:1847](/glade/u/home/pg2328/legoesm/packages/tools/legoesm/forcing/external.py:1847)). Without the Levante file, its missing species, concentrations, and both forcing estimates remain unverified.

- **C2 — UNSURE.** Spectral interpolation and rotation are implemented ([external.py:2508](/glade/u/home/pg2328/legoesm/packages/tools/legoesm/forcing/external.py:2508), [external.py:2714](/glade/u/home/pg2328/legoesm/packages/tools/legoesm/forcing/external.py:2714)). Present-day orbital defaults are explicit ([solar.py:96](/glade/u/home/pg2328/legoesm/packages/atmosphere/legoesm/atmosphere/physics/radiation/solar.py:96)). These establish neither identical input values nor the <0.05 W/m² bound.

- **C3 — UNSURE.** Noncyclic ozone interpolation is confirmed ([external.py:2032](/glade/u/home/pg2328/legoesm/packages/tools/legoesm/forcing/external.py:2032)); the 16 samples are averaged into **one** overhead layer ([model_driver.py:6081](/glade/u/home/pg2328/legoesm/packages/coupler/legoesm/driver/model_driver.py:6081)). Product differences are supported; forcing magnitude and exclusion as a primary cause are unmeasured.

- **C4 — AGREE on the optics defect; UNSURE on magnitude.** SW ignores file SSA ([external.py:883](/glade/u/home/pg2328/legoesm/packages/tools/legoesm/forcing/external.py:883)); volcanic and tropospheric OD are combined ([surface_utils.py:841](/glade/u/home/pg2328/legoesm/packages/tools/legoesm/forcing/surface_utils.py:841)) under shared optics ([rrtmgp.py:1118](/glade/u/home/pg2328/legoesm/packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:1118)). A 7% single-scattering absorption fraction does not establish a 5–7% forcing error.

- **C5 — REFUTE the warming inference.** Pressure-thickness distribution and visible-total-AOD CCN are confirmed ([surface_utils.py:489](/glade/u/home/pg2328/legoesm/packages/tools/legoesm/forcing/surface_utils.py:489), [surface_utils.py:845](/glade/u/home/pg2328/legoesm/packages/tools/legoesm/forcing/surface_utils.py:845)). Atmospheric absorption does not determine surface-temperature response. Global rsut tuning cannot constrain regional cloud-mediated cooling.

- **C6 — UNSURE.** Interpolation, freezing floor, and SIC clipping agree ([amip.py:988](/glade/u/home/pg2328/legoesm/packages/tools/legoesm/forcing/amip.py:988)). Different boundary datasets require numerical comparison before declaring equivalence.

- **C7 — UNSURE.** Production selects fixed surfdata; transient cover requires a separate switch ([model_driver.py:5423](/glade/u/home/pg2328/legoesm/packages/coupler/legoesm/driver/model_driver.py:5423)). Neither the global forcing bound nor regional irrelevance follows.

- **C8 — AGREE on missing explicit impurities; REFUTE “therefore brighter.”** Snow albedo depends on age and zenith ([surface_albedo.py:192](/glade/u/home/pg2328/legoesm/packages/core/legoesm/surface_albedo.py:192)), with calibrated brightness parameters ([clm_surface_map.py:138](/glade/u/home/pg2328/legoesm/packages/land/legoesm/land/clm_surface_map.py:138)). Relative brightness requires matched snow states; contribution remains plausible.

- **C9 — AGREE for production:** no prognostic nitrogen pool in the land state ([state.py:32](/glade/u/home/pg2328/legoesm/packages/land/legoesm/land/state.py:32)).

- **Overall — REFUTE the exclusion argument.** These checks do not establish that snow impurities are the sole plausible forcing contributor.

Missed channels: physiological CO₂ effects on canopy exchange ([model_driver.py:1340](/glade/u/home/pg2328/legoesm/packages/coupler/legoesm/driver/model_driver.py:1340)); harmonized canopy/soil optical inputs, beyond CLM PFT cover; topography, land fraction, and subgrid-orography boundary fields ([amip_production.sh:138](/glade/u/home/pg2328/legoesm/config/amip/amip_production.sh:138), [amip_production.sh:168](/glade/u/home/pg2328/legoesm/config/amip/amip_production.sh:168)).
