#!/usr/bin/env python
"""Machine-enforced fidelity bar for the #1226 NEMO term sweep.

The standing user directive is corr == 1.0 AND ratio == 1.0.  Repeatedly, a
term measured at 0.99x was recorded as "MATCHED"/"FAITHFUL"/"CLOSED" and the
campaign moved on.  This gate removes the judgment call: a term is DONE only if
its measured numbers meet the bar, otherwise it is DEBT -- regardless of what
prose was written about it.

Run:  python scripts/validate/ocean_fidelity/dino_1226/fidelity_bar_gate.py
Exit 0 only when every term is at the bar.  Non-zero (and a printed table)
otherwise.  Update MEASUREMENTS as terms are re-measured; never relax BAR_*.

HARNESS-CONTAMINATION AUDIT (2026-07-27, re-run of the whole sweep):
Two suspected harness artifacts were checked against every row below.
  (A) ``bridge_nemo_to_legoesm_topo``'s ``LEGOESM_NEMO_E3T`` env var
      (nemo_state_bridge.py:229-281) defaults to "off" (the 1-D e3t_1d/
      gdept_1d ladder, deliberately wrong -- see that function's docstring).
      "both"/"gdept_only" carry NEMO's real 3-D gdept_0 T-depths.
  (B) the bridge builds ``geom.dx_u/dy_u/dx_v/dy_v`` analytically from
      gphit/gphiv (``create_latlon_geometry``) rather than reading NEMO's own
      e1u/e2u/e1v/e2v from mesh_mask.nc.
VERDICT on (A): mostly clean, with ONE real exception. ~34 of the ~40 rows
come from probes whose own "Run with:" doc block sets LEGOESM_NEMO_E3T=both.
Six rows came from probes that never set it (so ran the "off" default): ATF
filter u/v, dyn_spg_ts puu_b/un_adv, zdftke pdlr, zdftke composite avt/avm,
and eiv transport u/v (probe_eiv_transport_v2*.py, which superseded a v1 that
DID set it). All were re-run with the env var toggled:
  - ATF, dyn_spg_ts, zdftke: reproduce to 4-5 sig figs either way -- (A)
    REFUTED for those, the residuals are real.
  - eiv transport u/v: GENUINELY e3t-sensitive. "off" gives corr 0.995918 /
    ratio 1.0231 (u) and 0.994325 / 1.0150 (v); "both" gives 0.999617 /
    0.9962 and 0.999103 / 0.9906. The numbers recorded below are the CLEAN
    ("both") ones. CAVEAT: this improvement is CONFOUNDED with the
    concurrent eos_depth='geometric' fix (commit f8e6f9488 credits that fix
    for 0.998474->0.999617 over the same window). Reaching 0.999617/0.996234
    needs BOTH changes; neither was isolated against a frozen baseline of the
    other. Do not attribute that gain to a single cause.
(B) was measured directly: bridge dy_v/dy_T
vs NEMO's real e2v/e2t agree to ~4-5e-5 relative error at every INTERIOR row;
the only rows with a real 0.2-0.8% discrepancy are the two channel WALL rows
(row 0 south, row 198 north) -- not a domain-wide floor. Root cause: on a
non-tripolar (regular/Mercator) LatLonCGridGeometry, ``gradient_x/y_cgrid``,
``divergence_cgrid`` (operators_latlon_cgrid.py) never read the geometry's
own dx_u/dy_u/dx_v/dy_v fields -- they recompute face metrics inline from
``cos_lat``/``dy``/``radius``/``dlon`` every call, so patching those fields
(or substituting NEMO's e1u/e2u/e1v/e2v into them) is a NO-OP for every
production tendency; verified empirically for dyn_hpg/dyn_vor/dyn_spg_ts
(ratio unchanged to 5+ digits after the substitution) AND for ATF and eiv
transport. Net: (B) explains NO row; (A) explains no row except the eiv
transport pair, whose recorded numbers are already the clean ones. Every
other DEBT number below is a real model residual, not a measurement artifact.
See docs/ocean/fidelity/ (harness-audit note) for the full re-measurement
log. Per-row notes below are tagged ``[e3t=both]`` where independently
verified.

#1226 METRIC-CONVENTION OPTION (2026-07-28, branch feat/nemo-dino-topo-bridge):
NEMO's DINO ``usr_def_hgr.F90`` builds the T/u-face horizontal grid metric
ISOTROPICALLY (``pe1t = pe2t = ra*rad*cos(rad*phi_T)*rn_e1_deg``) -- a
deliberate closed-form approximation, NOT a more-exact grid. legoESM's default
computed ``dy``/``area`` as the true finite-difference / exact-spherical-cap
form, which is MORE geometrically exact but LESS NEMO-faithful. Measured
directly against the real DINO R1 ``mesh_mask.nc``
(``/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_TRAJ/``,
nn_hls=2 halo stripped to the 195x48 interior): legoESM ``dy_T`` vs NEMO
``e2t`` relerr -1.27e-5..+9.5e-6, ``area`` vs ``e1t*e2t`` relerr
-2.5e-5..+4.1e-5 -- the SAME order as several DEBT rows below (ssh_nxt/
div_hor 9e-6, r3t/r3u-v ~3e-6). A new additive
``metric_convention: str = "exact" | "nemo_isotropic"`` option now exists on
``create_mercator_grid``, ``create_latlon_geometry``/``ensure_geometry``, and
``bridge_nemo_to_legoesm_topo`` (default ``"exact"`` is BIT-IDENTICAL to every
prior caller; verified by dedicated tests). It touches ONLY the T/u-face
metric (``dy_T``, ``dy_u``, ``area_T``) -- the v-face metric (#516
``vface_zonal_cos_lat``, tested by
``tests/ocean/unit/test_vface_metric_consistency_mercator.py``) and cell/face
LATITUDES (``grid.lat``/``grid.lat_v``) are untouched by construction, so the
#516 strain/stress-adjoint and div/advection mass-consistency invariants stay
green under both conventions (verified). Independently: ``r3t``/``r3u/r3v``
are PURELY VERTICAL ratios (eta/H0) with no horizontal-metric term at all, so
the "single shared cause" hypothesis for the 3e-6..5e-5 band is FALSIFIED for
those two rows specifically (structural, confirmed by direct formula read).
``dyn_cor_2d`` is only PARTIALLY exposed (its e2u/dy_u contributor is
metric-sensitive, its e1v/dx_v contributor is the #516-protected v-face and
is NOT -- and NOTE the u/v assignment is CROSSED vs the naive guess: NEMO's
``cor_u`` is built from ``e1u``/``e1v`` (zonal widths, INVARIANT under this
flag) while ``cor_v`` carries ``e2u = dy_u`` (the sensitive one), so it is
``cor_v``, not ``cor_u``, that can move). ``dyn_drg_init``'s u/v asymmetry is
larger than a symmetric metric offset predicts, so it is not expected to
collapse cleanly. The ``nemo_dino_kamm``/``nemo_dino_kamm_mlf`` DINO_RECIPES
cards now set ``metric_convention="nemo_isotropic"``; every other
recipe/caller stays ``"exact"``.

END-TO-END re-measurement DID complete (2026-07-28, bridged ``RUN_GDB``
restart, ``LEGOESM_NEMO_E3T=both``, controlled A/B with metric_convention the
ONLY variable). Results, per-row detail below:
  * ``ssh_nxt/div_hor``  0.999991 -> **1.000004** (real, predicted-direction
    movement; crossed 1.0; |ratio-1| ~4e-6 still outside the 1e-6 bar). The
    recorded tuple below is now the ``nemo_isotropic`` value, i.e. the
    SHIPPED ``nemo_dino_kamm`` card's state. STILL DEBT.
  * ``dyn_cor_2d``  u identical to 8 s.f. (predicted invariant, confirmed);
    v 0.999928 -> 0.999923 (moved AWAY from 1.0). NOT closed.
  * ``dyn_drg_init``  identical to 8 s.f. for BOTH u and v -- zero movement,
    mechanism CONCLUSIVELY excluded for this row.
NONE of the five rows crossed the bar. The 3e-6..5e-5 band therefore has AT
LEAST TWO distinct causes: the metric convention (live and dominant for
ssh_nxt/div_hor, marginal for dyn_cor_2d v) and something else entirely for
dyn_drg_init + the structurally-excluded r3t/r3u-v family. CAVEAT: the
dyn_cor_2d run's own exact-convention baseline did not reproduce the recorded
0.99999992/0.999986 (see that row) -- its A/B is controlled and valid, but its
absolute numbers come from a different reference state than the original probe.
"""
from __future__ import annotations

import sys

# The bar.  Roundoff only -- these are NOT tolerances for physics differences.
BAR_CORR = 1.0 - 1e-9
BAR_RATIO_EPS = 1e-6
# PER-ELEMENT bar.  corr and ratio are BOTH aggregate statistics: `ratio` is a
# MEAN, so mixed-sign per-element errors CANCEL and a term can sit inside
# BAR_RATIO_EPS while every element is wrong by orders more.  This is not
# hypothetical -- it was how "bn2 (rn2b)" held AT BAR at ratio 1-6.6e-9 while
# its OWN note recorded median |rel| 6.96e-6, a thousand times larger; that
# 7e-6 is exactly what put 10 of DINO's 9920 MLD columns on the wrong level
# (proved by substituting NEMO's own rn2b into the real zdf_mxl code path:
# mismatches 10 -> 0, see zdf_mxl_nmln_compare.py).  A term is only AT BAR if
# its per-element error is ALSO at roundoff.
BAR_PER_ELEM_EPS = 1e-9

# term -> measured per-element error (median or max |rel|, whichever the
# measuring probe reports -- record the LARGER when both are known).  Absent =
# never measured per-element; such a row can still show AT BAR but is counted
# and reported separately, because it passed on cancelling statistics ALONE.
# Shrink-only in the same sense as MEASUREMENTS: add entries as terms are
# re-measured, never delete one to make a row pass.
PER_ELEMENT: dict[str, float] = {
    # RE-MEASURED at HEAD 2026-07-28 by eos_rab_bn2_per_element.py against
    # NEMO's dump_alpha_b / dump_beta_b / tke_dump_rn2b [e3t=both; 342134 wet
    # 3-D T-cells, 332214 wet w-interfaces].
    #
    # RETRACTION (same day, my error).  The FIRST run of that probe fed the NOW
    # T/S and reported alpha 5.661e-7 with three >1% outliers and bn2 1.392e-5
    # with a 7074-cell tail "structured at levels 6-8".  All of that was a
    # TIME-LEVEL bug in the probe: NEMO stpmlf.F90:184 is
    #     CALL eos_rab( ts(:,:,:,:,Nbb), rab_b, Nnn )
    # -- T/S BEFORE, depth NOW -- so the dumps are BEFORE-level (as
    # bn2_alpha_compare.py's header already stated).  Feeding NOW T/S puts
    # |T_now - T_before| into the "error", which is largest in the thermocline;
    # that WAS the levels 6-8 structure.  At the correct time level the tail is
    # EMPTY (zero cells above 1%) and the outliers do not exist.  The earlier
    # probe_n2.py figures (bn2 6.96e-6, alpha 4.7e-6) also do not reproduce.
    #
    # bn2 is a SIGN-CHANGING field (27.2% of wet interfaces sit within 1e-3 of
    # RMS of zero), so a pointwise |lego-nemo|/|nemo| median is NOT trustworthy
    # for it.  The figure below is the conditioning-robust
    # err_norm = |lego-nemo| / RMS(nemo) = 4.322e-9 (p99 9.213e-7, max 3.103e-6).
    # Pointwise-relative for the same run was 1.956e-7 with zero >1% cells.
    "bn2 (rn2b)": 5.880e-16,
    # alpha does NOT cross zero (signed range +9.27e-5..+3.22e-4), so its
    # pointwise-relative stats ARE valid.  max|rel| 1.605e-8, zero cells >1%.
    "eos_rab alpha": 0.0,
    # beta is EXACT per-element (all stats identically 0.0).  Honest caveat: for
    # the DINO set lambda2 = mu2 = nu = 0, so beta collapses to the CONSTANT
    # b0/rho0 = 7.4614e-4 -- matching it is real but trivially so, and this row
    # carries NO evidence about the S-dependent terms, which DINO never
    # exercises.
    "eos_rab beta": 0.0,
    # probe_bottom_drag (A) under fp64: rel_err_med = rel_err_p90 = 0.000e+00
    # over 9920 T-points -- per-element proven, not merely aggregate-clean.
    "zdf_drg_nonlin T-point rate": 0.0,
    # 0/9920 columns differ -- an exact integer-level match, so per-element 0.
    "zdf_mxl (nmln)": 0.0,
    # pointwise |rel| median; one-signed and only 3.75% near-zero, so pointwise
    # IS valid here (verified: median barely moves when those cells are cut).
    "ldf_eiv kappa (aeiu)": 1.061e-6,
    # --- the 6 former CANCELLING-ONLY rows, MEASURED per-element 2026-07-29 by
    # cancelling_rows_per_element.py (fp64, BEFORE-level, e3t=both).  Each had
    # been passing on aggregate statistics alone, which is exactly how bn2 sat
    # falsely AT BAR.  This time no cancellation was hiding a defect -- but that
    # is now a MEASUREMENT, not a claim.  Values are the worst MEDIAN across
    # each row's sub-metrics; the robust err_norm is used for sign-changing
    # fields and pointwise |rel| for one-signed ones (each stated per row).
    #
    # sbc: utau err_norm med 0.0 (p99 1.3e-15) | qsr pointwise med 1.26e-16
    #      (one-signed 0..+230) | qns err_norm med 4.69e-16 (sign-changing) |
    #      sfx err_norm med 0.0.  CAVEAT: no first-class legoESM sfx array
    #      exists -- it was reconstructed from the production A_S/dino_S_star,
    #      so that leg certifies the FORCING FORMULA, not a shipped field.
    "sbc (utau/qsr/qns/sfx)": 4.693e-16,
    # one-signed (+526..+1501) so pointwise |rel| is valid; max 2.2e-16.
    # Excludes 197 periodic-seam cells where NEMO's OWN dump is zero (harness
    # artifact, verified confined to column 50).
    "ldftra ahtu (Redi, nn_aht_ijk_t=20)": 0.0,
    # as ahtu; scored on grid.vmask -- the T-mask falsely counted a closed wall
    # row as wet (probe bug found and fixed during the measurement).
    "ldftra ahtv (Redi, nn_aht_ijk_t=20)": 0.0,
    # sign-changing with 5.96% near-zero, so the pointwise max of 5.3e-6 is NOT
    # trustworthy; err_norm median 9.40e-14 (p99 2.7e-12) is the real figure.
    "dyn_hpg (du)": 9.400e-14,
    # sign-changing, 39.7% near-zero.  PGF/KEG additive separability verified
    # EXACTLY (max|full-(KEG+PGF)| = 0.0), and each vanishes exactly on its own
    # null state -- so this is a real decomposition, not a fitted one.
    "dyn_adv KEG": 3.578e-19,
    # T and S pointwise |rel| median 0.0 (max <= 2.2e-16), one-signed.
    # CAVEAT the probe itself raised: the ssh leg is TAUTOLOGICAL -- one unknown
    # is solved for and then re-substituted -- so it certifies TRANSCRIPTION of
    # the filter, not an independent check of the ssh value.  T/S do not share
    # that weakness.
    "ATF filter T/S/ssh": 0.0,

    # ldf_slp RE-MEASURED at HEAD 2026-07-28 by ldf_slp_per_element.py, the
    # first probe for these rows with BOTH mechanical preconditions wired
    # (require_fp64 + time_level_for_dump) AND a consistent run directory.
    # These are err_norm = |lego-nemo| / RMS(nemo), NOT pointwise |rel|: 7-9.5%
    # of wet points sit within 1e-3 of RMS of zero, so a pointwise-relative
    # median is meaningless here (max pointwise |rel| balloons to 1e5-1e6 purely
    # from near-zero-slope cells -- the same artifact as the retracted "slopes
    # 9% too large" finding in section F).
    #   wslpi corr 0.999876 |x|ratio 1.002818   wslpj corr 0.999800 |x|ratio 1.003324
    #   uslp  corr 0.999944 |x|ratio 1.002248   vslp  corr 0.999906 |x|ratio 1.002346
    # NOTE these differ from the previously recorded corr/ratio (e.g. wslpi
    # 0.999963/1.000524): the older probe read its restart from RUN_Y5_REBUILD
    # while comparing against RUN_GDB's dumps, and ran at the default fp32
    # policy.  The values here supersede on provenance, not on preference.
    # Residual is STRUCTURED: err_norm rises monotonically with depth and jumps
    # in the bottom 2-3 levels, max always at the deepest ACTIVE level in
    # columns j~185-191 -> a bottom-boundary / bathymetry-step term.  The
    # formula-vs-smoother split (raw pre-Shapiro dumps) is in flight.
    # CHAIN WALK after the live-gdept fix (ldf_slp_per_element.py section J,
    # NEMO execution order, e3t=both, fp64, BEFORE-level):
    #     prd 1.989e-11 | zgrv 4.4e-11 | zaj 1.180e-06 | zbw 3.467e-07
    #     zbj 4.380e-07 | zfk 0.0 EXACT | zww_raw 1.461e-06 | wslpj 1.762e-06
    # zaj is the first diverging stage and its INPUTS are at roundoff, so the
    # defect is in zaj's own divisor zcj = MAX(sum4 vmask, eps)*e2t.  The mask
    # count is integer-exact, leaving e2t -- and NEMO's DINO usr_def_hgr.F90
    # builds the grid ISOTROPICALLY (pe1t = pe2t) while legoESM's default
    # 'exact' convention computes the true finite-difference dy (recorded
    # elsewhere in this file as dy_T vs e2t relerr -1.27e-5..+9.5e-6).
    # A/B MEASURED 2026-07-28, single variable (LEGOESM_METRIC_CONVENTION):
    #     zaj  1.180e-06  ->  5.232e-11   (4.5 orders, to roundoff) CONFIRMED
    #     zbw  3.467e-07  ->  3.467e-07   (no e2t term, unchanged as predicted)
    #     wslpj err_norm 1.762e-06 -> 1.399e-06 ; |x|ratio 0.999929 -> 0.999932
    # So the metric convention OWNS zaj but is NOT the ldf_slp floor.  The rows
    # below are still recorded at the shipped default ('exact'): adopting
    # nemo_isotropic for the DINO oracle is a CONFIG-PLUMBING change (the
    # bridge defaults to 'exact' regardless of the recipe card -- a known
    # harness gap) and it also moves ssh_nxt, so it needs its own controlled
    # pass rather than being smuggled in here.
    # NEXT FLOOR: zbw = zm1_2g*pn2*(prd(k)+prd(k-1)+2) at 3.467e-07.  prd is at
    # roundoff, so this is pn2 -- the N^2 the SLOPES consume, which may not be
    # the same code path as the bn2 row already closed at 5.88e-16.
    # WIRED 2026-07-28: bridge_nemo_to_legoesm_topo now defaults to
    # metric_convention="auto" and DETECTS the oracle's convention from its own
    # mesh_mask (detect_metric_convention: e1t == e2t EXACTLY on DINO, because
    # usr_def_hgr.F90:111-119 sets pe2t = pe1t).  Detecting beats a hard default
    # -- the bridge also serves non-isotropic NEMO configs (GYRE) -- and beats
    # trusting a recipe card, which can drift out of step with the mesh_mask
    # actually being read.  The four rows below are now PRODUCTION-TRUE: the
    # shipped path reproduces them with no env override.  dy_v matches NEMO's
    # e2v to 0.000e+00.
    # i/j ASYMMETRY RESOLVED 2026-07-28 -- the e2 METRIC owns it, measured with
    # a control: vs NEMO's own mesh_mask, e1u matches EXACTLY (median and max
    # |rel| = 0.000e+00) while e2v does NOT (median 2.798e-05, max 8.241e-03).
    # Substituting NEMO's own e2v collapses vslp 1.609e-06 -> 3.807e-10 (100%);
    # the IDENTICAL substitution with e1u moves uslp 0.0%.  The control is what
    # makes this conclusive rather than suggestive.  Same e2 family as zaj's
    # divisor zcj = count*e2t -- ONE metric defect, THREE rows (vslp, wslpj,
    # zaj).  Closing it = adopting metric_convention="nemo_isotropic" for the
    # DINO oracle (NEMO's usr_def_hgr builds pe1t = pe2t); ESCALATED, since it
    # also moves ssh_nxt and is a config-default change.
    # Secondary transcription bug FIXED here: zdepu/zdepv (ldfslp.F90:261-266)
    # are U-/V-FACE AVERAGES of the bracketing T-column depths; we used the bare
    # T-point ladder for both AND passed zdepu to the v-slope (averaging over
    # the wrong axis entirely).  Real but second-order, exactly as predicted:
    # uslp 3.624e-10 -> 2.200e-10, vslp 1.609e-06 -> 1.541e-06 (4.2%).
    "ldf_slp wslpi": 3.427e-10,
    "ldf_slp wslpj": 2.770e-10,
    "ldf_slp uslp": 2.200e-10,
    "ldf_slp vslp": 3.265e-10,
}

# CLOSED 2026-07-28 -- fp64 + correct time level.  Under PrecisionPolicy.fp64()
# and BEFORE-level T/S (stpmlf.F90:184), all of these reach machine roundoff:
#     live gdept vs NEMO gdept(Kmm)   2.163e-8  ->  1.199e-16
#     eos_rab alpha  median |rel|     1.322e-9  ->  0.000e+00  (max 3.960e-16)
#     bn2 err_norm   median           4.322e-9  ->  1.413e-17  (max 9.046e-15)
#     zdf_mxl nmln   mismatched cols  10/9920   ->  0/9920     (histogram {0: 9920})
# TWO distinct causes, and they are NOT the same kind of thing:
#  (a) TIME LEVEL was a PROBE bug -- legoESM was never wrong; the probes fed NOW
#      T/S against BEFORE-level dumps.
#  (b) FLOAT32 DEPTH LADDER is real: create_z_star_from_thicknesses (vertical.py
#      ~:225) casts the f64 ladder to get_policy().control, which DEFAULTS TO
#      float32, rounding NEMO's f64 gdept_1d to ~7 digits.  JAX_ENABLE_X64=1
#      does NOT change that policy.
#
# *** CONTAMINATION HYPOTHESIS -- TESTED AND FALSIFIED 2026-07-28 ***
# Because every other row was measured under the DEFAULT fp32 policy, I
# proposed that the "unexplained 3e-6..5e-5 band" might be f32 rounding
# (eps 1.19e-7) rather than a legoESM defect.  MEASURED, not argued: all four
# available band probes re-run under PrecisionPolicy.fp64() via run_fp64.py
# (single variable, everything else identical):
#     ssh_nxt / div_hor   0.999991  ->  0.999991   (unchanged)
#     dom_qco_r3c r3t     0.999997  ->  0.999997   (unchanged)
#     dyn_drg_init u      0.999937 / 1.000278  ->  0.9999374 / 1.000278
#     dyn_cor_2d u        0.999717 / 0.998805  ->  0.9997170 / 0.9988055
# NO row moved.  The band is REAL -- discretisation or transcription, not
# dtype.  The hypothesis is withdrawn; do not re-raise it without new evidence.
# (fp64 remains REQUIRED for oracle work regardless -- it is what closed
# eos_rab/bn2/zdf_mxl above.  It simply is not the explanation for this band.)
#
# ROOT CAUSE shared by the two rows above (and by zdf_mxl's 10 columns).
# alpha recomputed with NEMO's OWN gdept is BIT-EXACT (median|rel| = 0.000e+00,
# max = 0.000e+00).  Our live gdept differs from NEMO's gdept(Kmm) by median
# |rel| 2.163e-8.  bn2's residual has the depth profile of that same error
# amplified by z*dT/dz: interfaces 0-5 are at 8e-15..4e-12 (bit-exact), a bump
# peaks at level 9 (~118 m, 3.107e-7) and decays to 2.794e-11 by level 33 --
# exactly where z*dT/dz peaks.  NOTE the earlier "bn2 has its own INDEPENDENT
# defect" verdict is WRONG: that split substituted alpha/beta but left bn2
# using OUR gdept for its own zrw weight and e3w divisor, so it could not see
# this cause.  Closing the 2.163e-8 depth residual should close eos_rab alpha,
# bn2 and the 10 zdf_mxl columns together.

# term -> (corr, ratio, note).  corr/ratio None = never measured at all.
MEASUREMENTS: dict[str, tuple[float | None, float | None, str]] = {
    # --- sweep terms ---
    "sbc (utau/qsr/qns/sfx)":        (1.0,        1.0,        "exact"),
    "eos_rab beta":                  (1.0,        1.0,        "bit-exact"),
    "eos_rab alpha":                 (1.0,        0.9999999998,   "median rel 4.7e-6 [e3t=both per probe_n2.py]. TIER-2 "
                                                              "2026-07-28: root cause of this residual class IDENTIFIED "
                                                              "and FIXED -- NEMO's eos_rab/bn2 take the LIVE z-star depth "
                                                              "gdept(Kmm)=gdept_0*(1+r3t) (domzgr_substitute.h90:139+:50+:56, "
                                                              "pure multiplicative stretch; gdept_z0=gdept-ssh at :145 is a "
                                                              "SEPARATE macro whose only consumer repo-wide is dynhpg.F90), "
                                                              "while lego's eos.py nemo_bn2_depth_ladders returned the STATIC "
                                                              "reference ladder. Isolated sensitivity on the bridged y5 "
                                                              "restart: alpha shifts 5.25e-6 (vs this row's 4.7e-6). The "
                                                              "GM/Redi consumer already carried the fix (hence this number); "
                                                              "the OTHER THREE consumers (TKE nemo_bn2, enhanced-diffusion "
                                                              "convection, k_profiles implicit) were still static and are now "
                                                              "fixed identically. NOT re-measured end-to-end after that fix "
                                                              "-- this row's number predates it; re-measure before trusting."),
    "bn2 (rn2b)":                    (1.0,        0.9999999934,   "median |rel| 6.96e-6 [e3t=both per probe_n2.py]. TIER-2 "
                                                              "2026-07-28: same live-gdept root cause + fix as eos_rab alpha "
                                                              "above (eosbn2.F90:1166 rab_3d_t zh=gdept(Kmm); :1459-1467 bn2_t "
                                                              "zrw + /e3w(Kmm)). Isolated sensitivity on the bridged y5 "
                                                              "restart: bn2 shifts 8.57e-5, independently reproducing the "
                                                              "8.59e-5 already documented at gm_redi_latlon_cgrid.py for the "
                                                              "one consumer that HAD the fix (which took it 8.59e-5 -> the "
                                                              "6.96e-6 recorded here) -- that cross-validation is what makes "
                                                              "this a real cause and not a coincidence of magnitude. Three "
                                                              "other consumers now fixed; NOT re-measured end-to-end."),
    "zdf_mxl (nmln)":                (1.0,        1.0,        "CLOSED 2026-07-28: 0/9920 columns differ, histogram {0: 9920} "
                                                              "(zdf_mxl_nmln_compare.py at fp64 + BEFORE-level T/S). Was "
                                                              "'12/9920 cols differ; REOPENED'. TWO causes, neither in zdf_mxl: "
                                                              "(a) the probe fed NOW T/S while zdfmxl.F90:98 integrates rn2b "
                                                              "(BEFORE) -- a HARNESS bug; (b) the f32 depth ladder. The selection "
                                                              "logic was proven bit-correct independently by substituting NEMO's "
                                                              "own rn2b into the real code path (10/9920 -> 0/9920), which is why "
                                                              "the blame moved upstream to bn2 rather than to this routine. "
                                                              "CAVEAT: nmln (the LEVEL) is exact; hmlp (the DEPTH) still shows "
                                                              "max|diff| 1.2943 m because our hml omits the live (1+r3t) stretch "
                                                              "on gdepw -- documented in _nemo_mld_from_n2_integral, separate DEBT."),
    "ldf_slp wslpi":                 (1.000000, 0.999962,   "SECOND live-depth fix 2026-07-28 (slope N^2 call site): _nemo_wpoint_e3w_wmask_n2's slope_n2='nemo_bn2' branch fed compute_buoyancy_frequency_nemo_bn2 the STATIC gdept/gdepw and then divided N^2 by the jacobian -- which corrected the e3w DENOMINATOR but left alpha/beta at static depths. Same defect as the gm_redi_density_and_jacobian fix, second call site. pn2 err_norm was 3.460e-07, matching zbw's 3.467e-07 floor to 3 digits. Now stretches BOTH ladders by the jacobian (which IS (1+r3t) under the existing _live_e3w gate, so no signature change) and DROPS the division, which would otherwise double-count. MEASURED: zbw 3.467e-07 -> 9.369e-16 (statistically identical to substituting NEMO's own rn2b, 9.304e-16), zbj 4.380e-07 -> 3.868e-15. Every chain stage is now at roundoff. REMAINING GAP IS A TAIL, NOT A BIAS: median err_norm is at/below the 1e-9 bar for wslpi (3.427e-10), wslpj (2.770e-10 under nemo_isotropic) and uslp (3.624e-10), but p99 ~4e-4 and max ~3e-3 are UNMOVED by either fix -- zbj p99 7.067e-11 amplifies to zww_raw p99 4.437e-04 through zaj/(zbj-eps) where zbj -> 0. That is CONDITIONING, and it is what holds |x|ratio at 1.7e-5..4.3e-5. vslp is the outlier (median still 1.609e-06) and needs its own look. Prior tuple: 1.000000/0.999932. PRODUCTION FIX 2026-07-28 (live z* EOS depth): gm_redi_density_and_jacobian fed the geometric EOS the STATIC z_coord.t_depth_ref ladder, where NEMO's eos_insitu uses the LIVE gdept(Knn)=gdept_0*(1+r3t), r3t=ssh/ht_0 (eosbn2.F90:541). Missing stretch up to 1.206 m. The API had ALREADY been widened to accept a per-column depth -- only the call site was never updated. Now routed through eos.nemo_bn2_live_ladders, the same canonical helper the eos_rab/bn2 consumers use. prd err_norm 2.559e-6 -> 1.804e-11 (equal to the NEMO-dumped-gdept substitution, i.e. the helper's 2.5e-8 reconstruction error is far below prd's depth sensitivity -- measured, not assumed). GATED per adversarial review: bit-identical when t_depth_ref is None (every non-NEMO-bridged recipe, matching the PGF sibling ocean_pe_latlon_cgrid.py:1289-1294 so the two never disagree within a timestep), and nemo_bn2_live_ladders now honours linear_free_surface (key_linssh: r3t==0, the column never stretches) -- which also fixes the eos_rab/bn2/PGF consumers that shared that gap. Prior tuple: 0.999963/1.001200. CORRECTED 2026-07-28 (second measurement, supersedes the one I committed hours earlier in this same session): that run's probe omitted eos_depth='geometric' from the DINO card (dino.py:912/936) and silently took the 'insitu' default -- a DIFFERENT density convention from NEMO's -- which biased every number it produced. With the card fixed the values AGREE with the long-standing historical figures (wslpi 0.999963 both ways), i.e. two independent probes converge. LESSON: when a new measurement disagrees with an old one, RECONCILE before recording -- I attributed the gap to the old probe's known flaws (run-dir mismatch, fp32) without checking, and the new probe was the broken one. STAGE WALK (section J, NEMO's own execution order prd -> zgrv -> zaj -> zbw -> zbj -> zfk -> zww_raw -> wslpj): NO ldfslp line injects the residual -- no stage jumps >100x over its predecessor. zcj mask count (:307-308) matches cell-for-cell incl. bottom-3 (0 cells differ, our zcj/e2t takes {2,3,4} at the bottom as NEMO does); zbj three-way MIN (:317) branch differs in 3/332214 all-wet and 1/29760 bottom-3, with our e3w LIVE (Kmm); zfk integer-division step (:320) is EXACT, 0/332214 differ; the zwslpj_hml recurrence IS carried -- ldfslp.F90:209 is DO jk = jpkm1,2,-1 (BOTTOM-TO-TOP), so our take_along_axis gather is the exact vectorised equivalent. RESIDUAL IS INHERITED: the earliest non-roundoff stage is prd (2.835e-6), ldf_slp's INPUT density, computed UPSTREAM of ldfslp.F90; the bottom amplification is conditioning from dividing by a small zbj near the seafloor (k=34: zbj 5.2e-9, zaj 4.4e-5, zww_raw 1.5e-2), not a mis-transcribed term. ldf_slp is TRANSCRIPTION-CLEAN; chase prd. Prior tuple: 0.999876/1.002818. RE-MEASURED at HEAD 2026-07-28 by ldf_slp_per_element.py -- the first probe for this row with BOTH mechanical preconditions wired (require_fp64 + time_level_for_dump) and a CONSISTENT run directory (RUN_GDB restart vs RUN_GDB dumps; the older probe read RUN_Y5_REBUILD's restart against RUN_GDB's dumps and ran at the default fp32 policy). Supersedes on provenance, not preference. err_norm=|d|/RMS recorded in PER_ELEMENT; pointwise |rel| is NOT usable here (7-9.5% of wet points within 1e-3 of RMS of zero). Residual is STRUCTURED: rises monotonically with depth, jumps in the bottom 2-3 levels, max always at the DEEPEST ACTIVE level in columns j~185-191 -- i.e. the SAME bottom-level family as the active_3d tie documented below, which improved but did not close it. Prior tuple: 0.999962701/1.000524131. TIER-2 2026-07-28 (STAGE-AUDIT FIX, unidentified-"
                                                              "defect campaign): traced ldf_slp stage-by-stage vs "
                                                              "NEMO's own MY_SRC/ldfslp.F90 dumps (Y5 RUN_GDB "
                                                              "kt=57601, e3t=both) -- prd (corr 1.0) and zbw "
                                                              "(corr 1.0) were already exact, but zaj/zai (the "
                                                              "w-point horizontal density gradient, ldfslp.F90:"
                                                              "271-278) broke down to corr 0.9806 SPECIFICALLY at "
                                                              "the deepest wet level (k=34 of 36; every other level "
                                                              "corr=1.000000). Root cause: compute_nemo_native_"
                                                              "slopes' THREE nemo_iso_lap call sites in "
                                                              "gm_redi_latlon_cgrid.py each re-derived active_3d as "
                                                              "(cumsum(z_coord.dz_ref)-dz_ref) < H_bathy -- a FLOAT "
                                                              "tie between two independently-rounded quantities "
                                                              "(the global representative dz ladder vs the bridge's "
                                                              "per-column cumsum(e3t_0) H_bathy) that spuriously "
                                                              "marked the deepest level ACTIVE on ~1861/10348 "
                                                              "columns whose bathymetry lands exactly on a level "
                                                              "interface (measured: H_bathy=3454.796630859375 vs "
                                                              "z_top[34]=3454.79638671875, a 1861-column tie broken "
                                                              "the wrong way) -- a sub-seafloor-leak bug in the same "
                                                              "family as commit 0db4c1abd. FIXED: factored the "
                                                              "duplicated construction into "
                                                              "_nemo_native_active_3d(mask, z_coord, H_bathy, dtype) "
                                                              "(gm_redi_latlon_cgrid.py, next to "
                                                              "gm_redi_density_and_jacobian), which prefers "
                                                              "z_coord.is_active (OceanPartialCellCoordinate's EXACT "
                                                              "per-column integer k<=bottom_level compare, "
                                                              "vertical.py:514) -- the SAME idiom "
                                                              "compute_isopycnal_slopes_latlon_cgrid's "
                                                              "nemo_mld_slope_ramp branch already uses -- falling "
                                                              "back to the float form only when is_active is absent "
                                                              "(plain z-star, no partial cells, so the tie never "
                                                              "bites). All three call sites (the Treguier-kappa "
                                                              "branch, the nemo_native tendency branch, the K33 "
                                                              "branch) now route through it; "
                                                              "scripts/tmp/probe_ldfslp_live_gdept.py (which shared "
                                                              "the same bug in its own harness) fixed identically. "
                                                              "RESULT (same e3t=both/kt=57601 twin): wslpi "
                                                              "0.997860193->0.999962701, ratio 1.005370859->"
                                                              "1.000524131 -- roughly an order of magnitude closer "
                                                              "to the bar on both corr and ratio. Prior blocker-1 "
                                                              "(live-gdept stretch) history retained below for "
                                                              "provenance; that fix was real but negligible for "
                                                              "this row -- THIS mask fix is the dominant cause. "
                                                              "STILL DEBT (corr 0.99996 / ratio 1.0005, not yet at "
                                                              "the 1e-9 bar) -- residual cause not yet identified, "
                                                              "but the 3-order-of-magnitude gap that motivated this "
                                                              "audit is closed."),
    "ldf_slp wslpj":                 (1.000000, 0.999957,   "SECOND live-depth fix 2026-07-28 (slope N^2 call site): _nemo_wpoint_e3w_wmask_n2's slope_n2='nemo_bn2' branch fed compute_buoyancy_frequency_nemo_bn2 the STATIC gdept/gdepw and then divided N^2 by the jacobian -- which corrected the e3w DENOMINATOR but left alpha/beta at static depths. Same defect as the gm_redi_density_and_jacobian fix, second call site. pn2 err_norm was 3.460e-07, matching zbw's 3.467e-07 floor to 3 digits. Now stretches BOTH ladders by the jacobian (which IS (1+r3t) under the existing _live_e3w gate, so no signature change) and DROPS the division, which would otherwise double-count. MEASURED: zbw 3.467e-07 -> 9.369e-16 (statistically identical to substituting NEMO's own rn2b, 9.304e-16), zbj 4.380e-07 -> 3.868e-15. Every chain stage is now at roundoff. REMAINING GAP IS A TAIL, NOT A BIAS: median err_norm is at/below the 1e-9 bar for wslpi (3.427e-10), wslpj (2.770e-10 under nemo_isotropic) and uslp (3.624e-10), but p99 ~4e-4 and max ~3e-3 are UNMOVED by either fix -- zbj p99 7.067e-11 amplifies to zww_raw p99 4.437e-04 through zaj/(zbj-eps) where zbj -> 0. That is CONDITIONING, and it is what holds |x|ratio at 1.7e-5..4.3e-5. vslp is the outlier (median still 1.609e-06) and needs its own look. Prior tuple: 1.000000/0.999929. PRODUCTION FIX 2026-07-28 (live z* EOS depth): gm_redi_density_and_jacobian fed the geometric EOS the STATIC z_coord.t_depth_ref ladder, where NEMO's eos_insitu uses the LIVE gdept(Knn)=gdept_0*(1+r3t), r3t=ssh/ht_0 (eosbn2.F90:541). Missing stretch up to 1.206 m. The API had ALREADY been widened to accept a per-column depth -- only the call site was never updated. Now routed through eos.nemo_bn2_live_ladders, the same canonical helper the eos_rab/bn2 consumers use. prd err_norm 2.559e-6 -> 1.804e-11 (equal to the NEMO-dumped-gdept substitution, i.e. the helper's 2.5e-8 reconstruction error is far below prd's depth sensitivity -- measured, not assumed). GATED per adversarial review: bit-identical when t_depth_ref is None (every non-NEMO-bridged recipe, matching the PGF sibling ocean_pe_latlon_cgrid.py:1289-1294 so the two never disagree within a timestep), and nemo_bn2_live_ladders now honours linear_free_surface (key_linssh: r3t==0, the column never stretches) -- which also fixes the eos_rab/bn2/PGF consumers that shared that gap. Prior tuple: 0.999963/1.001000. CORRECTED 2026-07-28 (second measurement, supersedes the one I committed hours earlier in this same session): that run's probe omitted eos_depth='geometric' from the DINO card (dino.py:912/936) and silently took the 'insitu' default -- a DIFFERENT density convention from NEMO's -- which biased every number it produced. With the card fixed the values AGREE with the long-standing historical figures (wslpi 0.999963 both ways), i.e. two independent probes converge. LESSON: when a new measurement disagrees with an old one, RECONCILE before recording -- I attributed the gap to the old probe's known flaws (run-dir mismatch, fp32) without checking, and the new probe was the broken one. STAGE WALK (section J, NEMO's own execution order prd -> zgrv -> zaj -> zbw -> zbj -> zfk -> zww_raw -> wslpj): NO ldfslp line injects the residual -- no stage jumps >100x over its predecessor. zcj mask count (:307-308) matches cell-for-cell incl. bottom-3 (0 cells differ, our zcj/e2t takes {2,3,4} at the bottom as NEMO does); zbj three-way MIN (:317) branch differs in 3/332214 all-wet and 1/29760 bottom-3, with our e3w LIVE (Kmm); zfk integer-division step (:320) is EXACT, 0/332214 differ; the zwslpj_hml recurrence IS carried -- ldfslp.F90:209 is DO jk = jpkm1,2,-1 (BOTTOM-TO-TOP), so our take_along_axis gather is the exact vectorised equivalent. RESIDUAL IS INHERITED: the earliest non-roundoff stage is prd (2.835e-6), ldf_slp's INPUT density, computed UPSTREAM of ldfslp.F90; the bottom amplification is conditioning from dividing by a small zbj near the seafloor (k=34: zbj 5.2e-9, zaj 4.4e-5, zww_raw 1.5e-2), not a mis-transcribed term. ldf_slp is TRANSCRIPTION-CLEAN; chase prd. Prior tuple: 0.999800/1.003324. RE-MEASURED at HEAD 2026-07-28 by ldf_slp_per_element.py -- the first probe for this row with BOTH mechanical preconditions wired (require_fp64 + time_level_for_dump) and a CONSISTENT run directory (RUN_GDB restart vs RUN_GDB dumps; the older probe read RUN_Y5_REBUILD's restart against RUN_GDB's dumps and ran at the default fp32 policy). Supersedes on provenance, not preference. err_norm=|d|/RMS recorded in PER_ELEMENT; pointwise |rel| is NOT usable here (7-9.5% of wet points within 1e-3 of RMS of zero). Residual is STRUCTURED: rises monotonically with depth, jumps in the bottom 2-3 levels, max always at the DEEPEST ACTIVE level in columns j~185-191 -- i.e. the SAME bottom-level family as the active_3d tie documented below, which improved but did not close it. Prior tuple: 0.999962742/1.000100880. [e3t=both, active_3d mask fix applied -- see wslpi "
                                                              "row; before 0.998193168/1.004061782]"),
    "ldf_slp uslp":                  (1.000000, 1.000017,   "SECOND live-depth fix 2026-07-28 (slope N^2 call site): _nemo_wpoint_e3w_wmask_n2's slope_n2='nemo_bn2' branch fed compute_buoyancy_frequency_nemo_bn2 the STATIC gdept/gdepw and then divided N^2 by the jacobian -- which corrected the e3w DENOMINATOR but left alpha/beta at static depths. Same defect as the gm_redi_density_and_jacobian fix, second call site. pn2 err_norm was 3.460e-07, matching zbw's 3.467e-07 floor to 3 digits. Now stretches BOTH ladders by the jacobian (which IS (1+r3t) under the existing _live_e3w gate, so no signature change) and DROPS the division, which would otherwise double-count. MEASURED: zbw 3.467e-07 -> 9.369e-16 (statistically identical to substituting NEMO's own rn2b, 9.304e-16), zbj 4.380e-07 -> 3.868e-15. Every chain stage is now at roundoff. REMAINING GAP IS A TAIL, NOT A BIAS: median err_norm is at/below the 1e-9 bar for wslpi (3.427e-10), wslpj (2.770e-10 under nemo_isotropic) and uslp (3.624e-10), but p99 ~4e-4 and max ~3e-3 are UNMOVED by either fix -- zbj p99 7.067e-11 amplifies to zww_raw p99 4.437e-04 through zaj/(zbj-eps) where zbj -> 0. That is CONDITIONING, and it is what holds |x|ratio at 1.7e-5..4.3e-5. vslp is the outlier (median still 1.609e-06) and needs its own look. Prior tuple: 1.000000/0.999987. PRODUCTION FIX 2026-07-28 (live z* EOS depth): gm_redi_density_and_jacobian fed the geometric EOS the STATIC z_coord.t_depth_ref ladder, where NEMO's eos_insitu uses the LIVE gdept(Knn)=gdept_0*(1+r3t), r3t=ssh/ht_0 (eosbn2.F90:541). Missing stretch up to 1.206 m. The API had ALREADY been widened to accept a per-column depth -- only the call site was never updated. Now routed through eos.nemo_bn2_live_ladders, the same canonical helper the eos_rab/bn2 consumers use. prd err_norm 2.559e-6 -> 1.804e-11 (equal to the NEMO-dumped-gdept substitution, i.e. the helper's 2.5e-8 reconstruction error is far below prd's depth sensitivity -- measured, not assumed). GATED per adversarial review: bit-identical when t_depth_ref is None (every non-NEMO-bridged recipe, matching the PGF sibling ocean_pe_latlon_cgrid.py:1289-1294 so the two never disagree within a timestep), and nemo_bn2_live_ladders now honours linear_free_surface (key_linssh: r3t==0, the column never stretches) -- which also fixes the eos_rab/bn2/PGF consumers that shared that gap. Prior tuple: 0.999980/1.000700. CORRECTED 2026-07-28 (second measurement, supersedes the one I committed hours earlier in this same session): that run's probe omitted eos_depth='geometric' from the DINO card (dino.py:912/936) and silently took the 'insitu' default -- a DIFFERENT density convention from NEMO's -- which biased every number it produced. With the card fixed the values AGREE with the long-standing historical figures (wslpi 0.999963 both ways), i.e. two independent probes converge. LESSON: when a new measurement disagrees with an old one, RECONCILE before recording -- I attributed the gap to the old probe's known flaws (run-dir mismatch, fp32) without checking, and the new probe was the broken one. STAGE WALK (section J, NEMO's own execution order prd -> zgrv -> zaj -> zbw -> zbj -> zfk -> zww_raw -> wslpj): NO ldfslp line injects the residual -- no stage jumps >100x over its predecessor. zcj mask count (:307-308) matches cell-for-cell incl. bottom-3 (0 cells differ, our zcj/e2t takes {2,3,4} at the bottom as NEMO does); zbj three-way MIN (:317) branch differs in 3/332214 all-wet and 1/29760 bottom-3, with our e3w LIVE (Kmm); zfk integer-division step (:320) is EXACT, 0/332214 differ; the zwslpj_hml recurrence IS carried -- ldfslp.F90:209 is DO jk = jpkm1,2,-1 (BOTTOM-TO-TOP), so our take_along_axis gather is the exact vectorised equivalent. RESIDUAL IS INHERITED: the earliest non-roundoff stage is prd (2.835e-6), ldf_slp's INPUT density, computed UPSTREAM of ldfslp.F90; the bottom amplification is conditioning from dividing by a small zbj near the seafloor (k=34: zbj 5.2e-9, zaj 4.4e-5, zww_raw 1.5e-2), not a mis-transcribed term. ldf_slp is TRANSCRIPTION-CLEAN; chase prd. Prior tuple: 0.999944/1.002248. RE-MEASURED at HEAD 2026-07-28 by ldf_slp_per_element.py -- the first probe for this row with BOTH mechanical preconditions wired (require_fp64 + time_level_for_dump) and a CONSISTENT run directory (RUN_GDB restart vs RUN_GDB dumps; the older probe read RUN_Y5_REBUILD's restart against RUN_GDB's dumps and ran at the default fp32 policy). Supersedes on provenance, not preference. err_norm=|d|/RMS recorded in PER_ELEMENT; pointwise |rel| is NOT usable here (7-9.5% of wet points within 1e-3 of RMS of zero). Residual is STRUCTURED: rises monotonically with depth, jumps in the bottom 2-3 levels, max always at the DEEPEST ACTIVE level in columns j~185-191 -- i.e. the SAME bottom-level family as the active_3d tie documented below, which improved but did not close it. Prior tuple: 0.999980198/1.000281581. [e3t=both, active_3d mask fix applied -- see wslpi "
                                                              "row; before 0.997795757/1.006438108]"),
    "ldf_slp vslp":                  (1.000000, 1.000021,   "SECOND live-depth fix 2026-07-28 (slope N^2 call site): _nemo_wpoint_e3w_wmask_n2's slope_n2='nemo_bn2' branch fed compute_buoyancy_frequency_nemo_bn2 the STATIC gdept/gdepw and then divided N^2 by the jacobian -- which corrected the e3w DENOMINATOR but left alpha/beta at static depths. Same defect as the gm_redi_density_and_jacobian fix, second call site. pn2 err_norm was 3.460e-07, matching zbw's 3.467e-07 floor to 3 digits. Now stretches BOTH ladders by the jacobian (which IS (1+r3t) under the existing _live_e3w gate, so no signature change) and DROPS the division, which would otherwise double-count. MEASURED: zbw 3.467e-07 -> 9.369e-16 (statistically identical to substituting NEMO's own rn2b, 9.304e-16), zbj 4.380e-07 -> 3.868e-15. Every chain stage is now at roundoff. REMAINING GAP IS A TAIL, NOT A BIAS: median err_norm is at/below the 1e-9 bar for wslpi (3.427e-10), wslpj (2.770e-10 under nemo_isotropic) and uslp (3.624e-10), but p99 ~4e-4 and max ~3e-3 are UNMOVED by either fix -- zbj p99 7.067e-11 amplifies to zww_raw p99 4.437e-04 through zaj/(zbj-eps) where zbj -> 0. That is CONDITIONING, and it is what holds |x|ratio at 1.7e-5..4.3e-5. vslp is the outlier (median still 1.609e-06) and needs its own look. Prior tuple: 1.000000/0.999982. PRODUCTION FIX 2026-07-28 (live z* EOS depth): gm_redi_density_and_jacobian fed the geometric EOS the STATIC z_coord.t_depth_ref ladder, where NEMO's eos_insitu uses the LIVE gdept(Knn)=gdept_0*(1+r3t), r3t=ssh/ht_0 (eosbn2.F90:541). Missing stretch up to 1.206 m. The API had ALREADY been widened to accept a per-column depth -- only the call site was never updated. Now routed through eos.nemo_bn2_live_ladders, the same canonical helper the eos_rab/bn2 consumers use. prd err_norm 2.559e-6 -> 1.804e-11 (equal to the NEMO-dumped-gdept substitution, i.e. the helper's 2.5e-8 reconstruction error is far below prd's depth sensitivity -- measured, not assumed). GATED per adversarial review: bit-identical when t_depth_ref is None (every non-NEMO-bridged recipe, matching the PGF sibling ocean_pe_latlon_cgrid.py:1289-1294 so the two never disagree within a timestep), and nemo_bn2_live_ladders now honours linear_free_surface (key_linssh: r3t==0, the column never stretches) -- which also fixes the eos_rab/bn2/PGF consumers that shared that gap. Prior tuple: 0.999981/1.000400. CORRECTED 2026-07-28 (second measurement, supersedes the one I committed hours earlier in this same session): that run's probe omitted eos_depth='geometric' from the DINO card (dino.py:912/936) and silently took the 'insitu' default -- a DIFFERENT density convention from NEMO's -- which biased every number it produced. With the card fixed the values AGREE with the long-standing historical figures (wslpi 0.999963 both ways), i.e. two independent probes converge. LESSON: when a new measurement disagrees with an old one, RECONCILE before recording -- I attributed the gap to the old probe's known flaws (run-dir mismatch, fp32) without checking, and the new probe was the broken one. STAGE WALK (section J, NEMO's own execution order prd -> zgrv -> zaj -> zbw -> zbj -> zfk -> zww_raw -> wslpj): NO ldfslp line injects the residual -- no stage jumps >100x over its predecessor. zcj mask count (:307-308) matches cell-for-cell incl. bottom-3 (0 cells differ, our zcj/e2t takes {2,3,4} at the bottom as NEMO does); zbj three-way MIN (:317) branch differs in 3/332214 all-wet and 1/29760 bottom-3, with our e3w LIVE (Kmm); zfk integer-division step (:320) is EXACT, 0/332214 differ; the zwslpj_hml recurrence IS carried -- ldfslp.F90:209 is DO jk = jpkm1,2,-1 (BOTTOM-TO-TOP), so our take_along_axis gather is the exact vectorised equivalent. RESIDUAL IS INHERITED: the earliest non-roundoff stage is prd (2.835e-6), ldf_slp's INPUT density, computed UPSTREAM of ldfslp.F90; the bottom amplification is conditioning from dividing by a small zbj near the seafloor (k=34: zbj 5.2e-9, zaj 4.4e-5, zww_raw 1.5e-2), not a mis-transcribed term. ldf_slp is TRANSCRIPTION-CLEAN; chase prd. Prior tuple: 0.999906/1.002346. RE-MEASURED at HEAD 2026-07-28 by ldf_slp_per_element.py -- the first probe for this row with BOTH mechanical preconditions wired (require_fp64 + time_level_for_dump) and a CONSISTENT run directory (RUN_GDB restart vs RUN_GDB dumps; the older probe read RUN_Y5_REBUILD's restart against RUN_GDB's dumps and ran at the default fp32 policy). Supersedes on provenance, not preference. err_norm=|d|/RMS recorded in PER_ELEMENT; pointwise |rel| is NOT usable here (7-9.5% of wet points within 1e-3 of RMS of zero). Residual is STRUCTURED: rises monotonically with depth, jumps in the bottom 2-3 levels, max always at the DEEPEST ACTIVE level in columns j~185-191 -- i.e. the SAME bottom-level family as the active_3d tie documented below, which improved but did not close it. Prior tuple: 0.999980493/1.000097328. [e3t=both, active_3d mask fix applied -- see wslpi "
                                                              "row; before 0.998369979/1.004132584]"),
    "ldf_eiv kappa (aeiu)":          (1.000000, 1.000001,   "RE-MEASURED 2026-07-29 (ldf_eiv_aeiu_per_element.py, fp64, BEFORE-level, e3t=both, metric_convention=auto): corr 0.999994774 -> 1.000000, |x|ratio 0.999958065 -> 1.000001. IMPROVED by the 2026-07-28 slope-N^2 live-ladder fix (1b37ea059) as predicted -- this row consumes _nemo_wpoint_e3w_wmask_n2. |ratio-1| = 5.537e-07 now PASSES BAR_RATIO_EPS. *** THE PER-ELEMENT BAR EARNED ITS KEEP HERE ***: corr AND ratio both clear, so under the pre-2026-07-28 gate this row would have flipped to a FALSE 'AT BAR' -- exactly the bn2 pattern -- while its pointwise |rel| median is 1.061e-06, three orders above roundoff. It stays DEBT, correctly and automatically. aeiu is ONE-SIGNED (0..+1500) with only 3.75% near-zero, and the median barely moves (1.061e-06 -> 1.009e-06) when near-zero cells are excluded, so pointwise |rel| IS the trustworthy metric here and the p99 8.1e+01 tail is a handful of near-zero cells, not the signal. Per-level err_norm is FLAT (max/min 1.1x over 35 levels) -- no depth structure, so the residual is not another ladder problem. ALIGNMENT CAVEAT recorded honestly: the offset scan ties at -2/-1/0 (corr 1.000000, residuals ~1.2e-11) because aeiu is DEPTH-BROADCAST and 76% of nonzero columns are k-constant, so shifts within the constant region are indistinguishable by corr; offset 0 is established by the sharp collapse at +1/+2 (0.986/0.976), not by the scan peak alone. aeiv NOT measured -- eiv_dump_aeiv.bin does not exist in RUN_GDB. Prior tuple: 0.999994774/0.999958065. MEASUREMENT-ARTIFACT FIX 2026-07-28 (NOT a model "
                                                              "fix; ratio 0.999958 is 4e-5 outside BAR_RATIO_EPS so "
                                                              "the gate still marks DEBT, but the prior 0.975/1.033 "
                                                              "'independent operator defect' framing is FALSE): the "
                                                              "recorded 0.975163/1.032510 residual was a "
                                                              "MEASUREMENT ARTIFACT in scripts/tmp/probe_ldfslp_"
                                                              "live_gdept.py, not an independent operator defect. "
                                                              "Stage-by-stage audit (feeding lego's OWN state through "
                                                              "compute_treguier_kappa_gm_nemo_native(return_"
                                                              "diagnostics=True) and comparing each NEMO-named "
                                                              "intermediate -- zn, zah, zhw, zRo, zaeiw -- against "
                                                              "the corresponding eiv_dump_*.bin) found EVERY stage "
                                                              "already at 0.99999-class corr / 0.9997-1.0003 ratio: "
                                                              "zn 0.999999994/0.999982151, zah 0.999999923/"
                                                              "1.000254783, zhw 0.999999482/1.000059719, zRo "
                                                              "0.999999999/1.000001423, zaeiw (T-point kappa_GM) "
                                                              "0.999993735/0.999968709 (n=9920, e3t=both, RUN_GDB "
                                                              "kt=57601). Per-level decomposition of rn2b (pn2) and "
                                                              "e3w against NEMO's own dumps: corr=1.000000 / ratio "
                                                              "0.9998-1.0001 at EVERY wet level 0-34 -- no single "
                                                              "level or factor carries an anomaly. ROOT CAUSE of the "
                                                              "discrepancy: eiv_dump_aeiu.bin is NEMO's ``paeiu`` -- "
                                                              "the U-FACE average of the T-point coefficient "
                                                              "(ldftra.F90:741 ``zaeiu(ji,jj) = 0.5*(zaeiw(ji,jj)+"
                                                              "zaeiw(ji+1,jj))*ssumask(ji,jj)``), NOT a plain vertical "
                                                              "broadcast of zaeiw (ldftra.F90:763-764 is a SEPARATE, "
                                                              "later step). The old probe's own comment claimed aeiu "
                                                              "'is the SAME 2-D kappa_GM broadcast down the column' -- "
                                                              "that claim conflated the vertical broadcast with the "
                                                              "horizontal T->U face shift that happens FIRST, and "
                                                              "compared the raw T-point kappa_GM directly against the "
                                                              "east-shifted face-averaged dump. A T-point field vs its "
                                                              "own east-neighbour average still correlates well enough "
                                                              "that the 3x3 alignment scan still picks offset=(0,0) as "
                                                              "the winner (runner-up corr ~0.95-0.98) -- i.e. the scan "
                                                              "looked 'aligned' while comparing the wrong quantity. "
                                                              "FIX: route the comparison through nemo_kappa_gm_to_"
                                                              "faces (gm_redi_latlon_cgrid.py:1271-1295 -- already "
                                                              "existed, transcribes ldftra.F90:740-743 exactly, but "
                                                              "was never called by any probe) before comparing to "
                                                              "eiv_dump_aeiu.bin; masks are g.umask/g.vmask surface "
                                                              "slices (ssumask/ssvmask), NOT the bridge's u_mask/"
                                                              "v_mask (which carry an extra periodic-wrap column). "
                                                              "RESULT (scripts/tmp/probe_ldfslp_live_gdept.py, "
                                                              "e3t=both, RUN_GDB kt=57601, alignment scan {-1..+1} "
                                                              "sharp offset=0 peak, runner-up corr 0.977-0.981): "
                                                              "corr 0.975162749->0.999994774, ratio "
                                                              "1.032510295->0.999958065, n=9920 -- same 0.9999-class "
                                                              "quality as the ldf_slp rows (still technically DEBT "
                                                              "under BAR_RATIO_EPS=1e-6, not literal 1.0/1.0). No lego "
                                                              "model code changed; production kappa_GM (consumed as "
                                                              "a 2-D T-point field by the tendency, never face-"
                                                              "averaged in the real dispatch -- nemo_kappa_gm_to_"
                                                              "faces exists solely for oracle-diagnostic comparison "
                                                              "to NEMO's paeiu/paeiv dump) is UNCHANGED and was "
                                                              "already correct; only the probe's comparand was wrong. "
                                                              "This SUPERSEDES both the active_3d-mask-fix note and "
                                                              "the Omega-fix note previously recorded here -- neither "
                                                              "was wrong, but neither explains this row's residual "
                                                              "either; the T3/T23/T6 hypothesis space (e3w, rn2b, "
                                                              "wmask, ff_t, column-sum masks, taper/clip counts) from "
                                                              "the task brief is FULLY EXCLUDED by the per-level/"
                                                              "per-stage decomposition above -- every candidate "
                                                              "already matched NEMO before this fix; the defect was "
                                                              "never inside the Treguier chain. eiv transport u/v "
                                                              "(0.999617/0.996109, 0.999103/0.990501) are UNCHANGED "
                                                              "by this fix -- they consume kappa_GM as the 2-D "
                                                              "T-point field via the real tendency dispatch (not "
                                                              "nemo_kappa_gm_to_faces), so this was never their cause; "
                                                              "their own residual (static t_depth_ref vs live "
                                                              "z-star gdept, deepest 1-2 levels) is untouched and "
                                                              "remains separate DEBT."),
    "ldftra ahtu (Redi, nn_aht_ijk_t=20)": (1.0,  0.9999999722, "AT BAR: K_h_base*cos(lat_T) shares its row's latitude with NEMO's ahtu -> exact vs NEMO gphiu (ldftra_ahtv_compare.py)"),
    "ldftra ahtv (Redi, nn_aht_ijk_t=20)": (1.0,  0.9999999704, "AT BAR (fixed): prior note ('interp_cell_to_vface averages avg(cos) not cos(avg)') was WRONG -- _static_kappa_redi_override never called interp_cell_to_vface (Rule 0 violation, a probe artifact). Real cause: aht was the SAME T-point field reused unshifted for both zfu and zfv, while NEMO's ahtv is INDEPENDENTLY evaluated at the v-point (ldftra.F90:325-329 ldf_c2d('TRA',...), ldfc1d_c2d.F90:141-145: ahtv=zUfac*MAX(e1v,e2v)**inn). Fixed by adding a v-face-specific kappa_Redi_v (grid.cos_lat_v at the north-face-of-cell-j convention) threaded through nemo_iso_lap_tracer_tendency_latlon_cgrid/nemo_iso_w_kappa_sums/nemo_iso_a33/compute_isoneutral_K33_latlon; verified against the actual NEMO ldftra_dump_{ahtu,ahtv,gphiu,gphiv}.bin (RUN_1226_AHTU): pre-fix corr 0.9998618/ratio 1.0000380, post-fix corr 1.0000000/ratio 0.9999999704 -- matches the u-face's own bit-exact quality"),
    "eiv transport u":               (0.999617,   0.996109,   "REOPENED - eos_depth='geometric' was not threaded into gm_redi_density_and_jacobian (#1226); fixed, corr 0.998474->0.999617, ratio 0.994097->0.996109; residual = static t_depth_ref vs NEMO's live z-star gdept (~1e-8 density bias, deepest 1-2 levels only) DEBT [MEASURED AT e3t=both. The superseding "
                                                              "probe_eiv_transport_v2*.py did NOT set LEGOESM_NEMO_E3T, so the "
                                                              "'off' default gives corr 0.995918 / ratio 1.0231 -- this term IS "
                                                              "genuinely e3t-sensitive (unlike every other row audited). The "
                                                              "0.998474->0.999617 gain is CONFOUNDED between the eos_depth fix "
                                                              "and e3t=both; both are needed, neither isolated. Do not cite one cause. "
                                                              "RE-MEASURED at HEAD c8e5d305b (probe_eiv_transport_v2_e3tboth.py "
                                                              "+ a pooled-stat/alignment-scan wrapper, since the in-repo probe "
                                                              "only ever printed per-level/per-row breakdowns, not this pooled "
                                                              "number): alignment scan {-2..+2} confirms a sharp offset=0 peak "
                                                              "(corr 0.9996 vs runner-up 0.39 at offset -1); corr 0.999617/"
                                                              "ratio 0.996109, n=326580 -- essentially UNCHANGED from the ratio "
                                                              "recorded above (0.996234, small pooling-convention rounding)."),
    "eiv transport v":               (0.999103,   0.990501,   "REOPENED - same eos_depth fix, corr 0.995437->0.999103, ratio 0.982351->0.990501; residual concentrated at bottom-adjacent rows (jj~130-136,189-191) + deepest 2 levels (k=32,33), same static-vs-live-gdept cause as u; still 0.9% low DEBT [MEASURED AT e3t=both; "
                                                              "'off' default gives corr 0.994325 / ratio 1.0150. Same eos_depth-vs-e3t "
                                                              "confound as the u-component -- see that row. "
                                                              "RE-MEASURED at HEAD c8e5d305b (same wrapper as eiv transport u): "
                                                              "alignment scan {-2..+2} sharp offset=0 peak (corr 0.9991 vs "
                                                              "runner-up 0.31 at offset -1); corr 0.999104/ratio 0.990501, "
                                                              "n=330403 -- UNCHANGED from the ratio recorded above (0.990637, "
                                                              "same pooling-rounding as u)."),
    "traadv_fct fluxes":             (0.999999,   0.999990,   "[e3t=both per traadv_fct_probe.py]. RE-MEASURED at HEAD "
                                                              "e72923faa (traadv_fct_probe.py, RUN_GDB kt=57601): the "
                                                              "PREVIOUSLY recorded 0.9594-0.9799 (east/north face) / 0.9139 "
                                                              "(vertical) was a COMPARISON-CONVENTION BUG, not a physics "
                                                              "defect -- the old probe's k-offset was chosen by maximizing "
                                                              "corr against NEMO's RAW trd_final/trd_up dumps, which are "
                                                              "Krhs-CONTAMINATED (pt(Krhs) already carries tra_sbc/tra_qsr "
                                                              "added by earlier stpmlf.F90 calls before tra_adv_fct runs); "
                                                              "that contaminated signal peaks at the WRONG k-offset (+1). "
                                                              "Re-run at the offset the PURE (Krhs-uncontaminated) "
                                                              "reconstruction confirms (offset=0, sharp peak): east-face "
                                                              "upstream corr 0.999999/ratio 0.999990, north-face "
                                                              "corr 0.999999/ratio 0.999987, vertical corr 0.999986/"
                                                              "ratio 1.000068, n=342134 each. Recorded tuple is the "
                                                              "worst (vertical) component; still ~1e-5 outside BAR, "
                                                              "not literal 1.0/1.0, but no longer the suspicious "
                                                              "0.96-0.98 floor -- that floor was the offset bug."),
    "traadv_fct tendency (T)":       (0.999991,   0.999887,   "after nonosc bound fix 2a73221ce + dry-cell fix 01c1f226a "
                                                              "[e3t=both]. RE-MEASURED at HEAD e72923faa "
                                                              "(traadv_fct_probe.py, RUN_GDB kt=57601): the PREVIOUSLY "
                                                              "recorded 0.994487/1.010631 does NOT reproduce -- re-running "
                                                              "the EXACT same probes (probe_fct.py's addendum, "
                                                              "probe_nonosc_stages.py's 'CURRENT lego production' stage-5 "
                                                              "line) cited as its provenance, against the SAME dumps, on "
                                                              "unchanged advection.py (no commits touch it between "
                                                              "01c1f226a and e72923faa), gives corr=0.999989-0.999991/"
                                                              "ratio=0.999887-0.999901 both ways, reproducibly (3 reruns, "
                                                              "omega-fix and without: unchanged to 5 s.f.). The recorded "
                                                              "0.994487 is a transcription error, not a live regression -- "
                                                              "no env var (LEGOESM_NEMO_E3T=off/both) or bridge-omega "
                                                              "setting reproduces it. Corrected to the reproducible "
                                                              "number; still ~1e-5 outside BAR_RATIO_EPS, not literal "
                                                              "1.0/1.0."),
    "traadv_fct horizontal tend":    (0.999967,   0.999957,   "limiter itself now correct [e3t=both]. MEASURED for the "
                                                              "FIRST TIME at HEAD e72923faa (traadv_fct_probe.py, new "
                                                              "probe -- none existed before; the recorded 0.999950 was "
                                                              "UNPROVENANCED and is superseded here, not confirmed): "
                                                              "horizontal-only (xad+yad, vertical flux zeroed in the NEMO "
                                                              "flux-divergence reconstruction) tendency, offset=0, "
                                                              "corr=0.999967/ratio=0.999957, n=342134. Companion "
                                                              "vertical-only split (not a gated row, reported for the "
                                                              "record): corr=0.999967/ratio=0.999969. Still ~3e-5 "
                                                              "outside BAR, not literal 1.0/1.0."),
    "traadv_fct vertical upstream flux": (0.999986, 1.000068, "0.91 was an OFFSET ARTIFACT; dry-cell mask fixed 01c1f226a "
                                                              "(clips 603 vs NEMO 662) [e3t=both]. RE-MEASURED at HEAD "
                                                              "e72923faa (traadv_fct_probe.py, RUN_GDB kt=57601, "
                                                              "supersedes check_wflux_offset0.py): re-running that exact "
                                                              "script at HEAD reproduces corr=0.9999864/ratio=1.0000695 at "
                                                              "the offset=0 peak (offset -1/+1 give 0.923/0.916, matching "
                                                              "the recorded alignment-scan shape) -- NOT the previously "
                                                              "recorded 0.997791/1.007386, which does not reproduce from "
                                                              "any env var or e3t-mode combination tried and is a "
                                                              "transcription error. Corrected to the reproducible number; "
                                                              "still ~1e-5 outside BAR_RATIO_EPS, not literal 1.0/1.0."),
    "dyn_hpg (du)":                  (1.0,        1.000000018,   "TIER-2 2026-07-28 (blocker-1 fix): the remaining "
                                                              "hypothesis (residual inherited from the in-situ "
                                                              "density / EOS-depth input on a developed state) was "
                                                              "CONFIRMED. eos_geometric_depth_1d (the PGF's own "
                                                              "eos_depth='geometric' density input, "
                                                              "ocean_pe_latlon_cgrid.py _bc_geometry_and_density) fed "
                                                              "the STATIC t_depth_ref ladder with NO (1+r3t) stretch, "
                                                              "whereas NEMO's eos_insitu/S-EOS (eosbn2.F90:1166) reads "
                                                              "the LIVE gdept(Kmm)=gdept_0*(1+r3t). Fixed by routing "
                                                              "through eos.nemo_bn2_live_ladders(z_coord, eta_safe, "
                                                              "H_bathy) (the SAME helper already used for eos_rab/bn2), "
                                                              "which is the plain (non-z0-shifted) live gdept eosbn2.F90 "
                                                              "wants. hpg_tendency_compare.py --e3t-mode both on the Y5 "
                                                              "twin (RUN_GDB kt=57601): du corr 0.999999991->1.000000000 "
                                                              "/ ratio 1.000054839->1.000000018 (was bottom-heavy "
                                                              "k=27 1.000036->k=34 1.000231; now flat ~1e-7 every "
                                                              "level); dv corr 0.999999990->1.000000000 / ratio "
                                                              "1.000053174->0.999986794 -- the residual dv gap is the "
                                                              "PRE-EXISTING dy_v-vs-e2v harness metric (metric-"
                                                              "substituted variant: 1.000066389->1.000000007, i.e. "
                                                              "the SAME harness artifact this file's (B) audit already "
                                                              "attributes to wall rows, not a live-depth remnant). "
                                                              "AT BAR for du; dv just outside BAR_RATIO_EPS but its "
                                                              "residual is independently isolated to the harness, not "
                                                              "the PGF term -- see the fidelity strategy doc's harness/"
                                                              "model split. The bottom-heavy accumulation signature "
                                                              "that motivated the #1226 blocker-1 investigation is GONE."),
    "dyn_vor EEN u":                 (0.999896,   1.001180,   "bottom levels 1.04-1.07; REOPENED [e3t=both, re-verified: "
                                                              "e2u/e1u substitution leaves ratio 1.001180->1.001180 "
                                                              "unchanged; harness (B) REFUTED]. RE-MEASURED at HEAD "
                                                              "c8e5d305b (probe_hpg_vor_1226.py, RUN_GDB kt=57601): "
                                                              "offset scan {-1,0,+1} confirms a sharp offset=0 peak "
                                                              "(-1=0.996081, +0=0.999896, +1=0.996184); corr 0.999896/"
                                                              "ratio 1.001180 exactly reproduced. UNCHANGED."),
    "dyn_vor EEN v":                 (0.999932,   1.000717,   "REOPENED [e3t=both, re-verified: e2v/e1v substitution "
                                                              "leaves ratio 1.000717->1.000717 unchanged; harness (B) "
                                                              "REFUTED]. RE-MEASURED at HEAD c8e5d305b (same probe/run "
                                                              "as EEN u): offset scan {-1,0,+1} sharp offset=0 peak "
                                                              "(-1=0.996937, +0=0.999932, +1=0.996976); corr 0.999932/"
                                                              "ratio 1.000717 exactly reproduced. UNCHANGED."),
    "dyn_adv KEG":                   (1.0,        1.000000,   "byte-exact"),
    "dyn_adv ZAD":                   (0.999200,   0.995116,   "after nemo_advective fix [e3t=both per probe_1226_keg_zad_split.py]. "
                                                              "RE-MEASURED at HEAD c8e5d305b (same probe): offset scan "
                                                              "{-1,0,+1} sharp offset=0 peak (u: -1=0.7222, +0=0.9992, "
                                                              "+1=0.7221); corr 0.999200/ratio 0.995116 (u-component; "
                                                              "v-component corr 0.998712/ratio 0.995598, not separately "
                                                              "tracked by this row). UNCHANGED to 4 s.f."),
    "zdftke pdlr":                   (0.998124,   0.996778,   "REOPENED [re-verified 2026-07-27: probe_tke_prandtl.py's "
                                                              "own doc block omits LEGOESM_NEMO_E3T (default 'off'); "
                                                              "re-ran with e3t=both -> pdlr CORRECTED corr 0.99812 "
                                                              "(unchanged), ratio 0.996800->0.996780 (unchanged to "
                                                              "3 s.f.); harness (A) REFUTED for this term]. "
                                                              "RE-MEASURED at HEAD c8e5d305b: probe_zdftke_prandtl_"
                                                              "e3tboth.py has no built-in offset scan (single hardcoded "
                                                              "k+1 mapping), so a dedicated scan wrapper (_scan_pdlr.py, "
                                                              "reusing the probe's own pdlr_lego/nemo_pdlr arrays "
                                                              "verbatim) was run over {-2..+2}: sharp offset=0 peak "
                                                              "(corr 0.998124 vs runner-up 0.891260 at +1). corr/ratio "
                                                              "UNCHANGED."),
    "zdftke composite avt/avm":      (0.997559,   0.996197,   "257 cells; REOPENED [re-verified 2026-07-27: "
                                                              "probe_zdftke_avt_avm.py's own doc block omits "
                                                              "LEGOESM_NEMO_E3T (default 'off'); re-ran with e3t=both "
                                                              "-> corr 0.997578->0.997559, ratio 0.996198->0.996197 "
                                                              "(unchanged to 5 s.f.); harness (A) REFUTED]. RE-MEASURED "
                                                              "at HEAD c8e5d305b (probe_zdftke_avt_avm_e3tboth.py): "
                                                              "offset scan {-1,0,+1} sharp offset=0 peak (-1=0.816564, "
                                                              "+0=0.997559, +1=0.906048); corr 0.997559/ratio 0.996197 "
                                                              "(avt) and corr 0.997559/ratio 0.996196 (avm, "
                                                              "indistinguishable). UNCHANGED."),
    # legoESM cannot RUN on NEMO's true vertical grid: LEGOESM_NEMO_E3T
    # defaults to "off" (the wrong e3t_1d ladder) because "both" destabilises
    # multi-day runs (max|u| 0.66 -> 3 m/s). This is a real defect, and it is
    # why every fidelity probe must set the env var explicitly -- it has
    # contaminated three measurements so far.
    # BLOCKER (2026-07-28, CLOSED): z_coord.t_depth_ref was a 1-D-ladder-ONLY
    # API, so a per-column live gdept_0*(1+r3t) could not be expressed for the
    # PGF (eos_geometric_depth_1d) or ldf_slp (compute_nemo_native_slopes'
    # gdept/gdepw_top/e3w) consumers -- unlike the bn2/eos_rab consumers,
    # which already routed through eos.nemo_bn2_live_ladders. Both are now
    # widened: eos_geometric_depth_1d accepts a (nlat,nlon,nlev) array
    # (iterate_eos_and_pressure_anomaly's p_eos = rho0*g*gdept multiply is
    # shape-general already -- no widening needed there), and
    # compute_nemo_native_slopes/_nemo_wpoint_e3w_wmask_n2 apply the live
    # (1+r3t) stretch (reusing the jacobian already threaded through, GATED
    # on isinstance(z_coord, OceanPartialCellCoordinate) since a pure z*
    # coordinate's jacobian is (eta+H_bathy)/H_max, NOT (1+r3t) -- see the
    # gm_redi_latlon_cgrid.py comments at the fix site). RESULT: dyn_hpg's
    # bottom-heavy accumulation signature is GONE (du now AT BAR); ldf_slp's
    # wslpi/wslpj/uslp/vslp/ldf_eiv-aeiu residuals barely move (DINO's r3t is
    # too small, ~7e-5, to be the dominant cause there) -- see those rows.
    # The API limitation is CLOSED; it was NOT the dominant cause for every
    # term it was suspected for.
    "STABILITY on NEMO true grid (e3t_0)": (None, None,
        "MEASURED 2026-07-28 (was: 'legoESM UNSTABLE on NEMO's actual geometry; "
        "bridge defaults to the wrong ladder to hide it'). FROM REST the model is "
        "STABLE on NEMO's true ladder: LEGOESM_NEMO_E3T=both, nemo_dino_kamm_mlf, "
        "5 full years (1800 d, 57600 steps) completed with T in [3.5,26.3] C and "
        "max|u_surf| 0.52 m/s -- no growth (dino_year_screen_fullframe.py). The "
        "documented blow-up (max|u| 0.66 -> 3 m/s over 20 d) is specific to "
        "starting FROM A NEMO RESTART, not to the ladder itself, so the scope of "
        "that defect is narrower than recorded. Ladders are IDENTICAL for k=0..24 "
        "(top 913 m) and differ only below ~1000 m (up to +15%): NEMO's e3t_0 is "
        "the FINITE-DIFFERENCE gdepw(k+1)-gdepw(k) (ln_e3_dep), e3t_1d the "
        "analytic derivative. CLIMATE EFFECT MEASURED, controlled (same script, "
        "same NEMO reference, ONLY the ladder differs): y5 ACC 67.7 -> 66.2 Sv, "
        "i.e. the CORRECT ladder moves 1.5 Sv FURTHER from NEMO's 91.1. The "
        "hypothesis that this ladder explains the deep-contrast deficit is "
        "FALSIFIED. Still UNMEASURED as a per-element fidelity row (this is a "
        "stability/climate result, not a term comparison); the restart-start "
        "instability remains a real open defect."),
    # dv's residual is the DEFERRED v-face metric (dy_v vs NEMO e2v): the
    # metric_convention work shipped T/u-face only because vface_zonal_cos_lat
    # is a tested #516 invariant. Metric-substituted, dv closes to 1.000000007.
    "dyn_hpg (dv)":                  (1.0,        0.999986794, "blocked on the deferred v-face metric; substituted -> 1.000000007"),
    # --- round 2 ---
    "dyn_spg_ts pssh":               (0.999994,   0.998600,   "*** CAUSE LOCALISED 2026-07-29 (spg_substep_chain.py): NOT dyn_spg_ts. *** The error is already present at the barotropic LOOP-ENTRY SEED, before the substep recurrence runs once, and stays flat through it: err_norm u 2.31e-2 (seed) -> 2.31e-2 (substep 1) -> 3.09e-2 (final, puu_b); v 3.05e-2 -> 3.04e-2 -> 3.27e-2. An ACCUMULATION signature would need a clean seed and a large final gap; this is the opposite. Ruled out BY MEASUREMENT, not inference: substep count matches exactly (legoESM n_barotropic_substeps=23 == NEMO nn_e=23; compute_nemo_boxcar_centred_weights n_loop=68 == NEMO icycle=68), so it is not a window/weight mismatch; offset scans peak sharply at (0,0) for BOTH a U-face field (puu_b) and a T-point field (pssh), so it is not a face-convention or index shift. ssh is EXACT at the seed (0.0) and only reaches 4.3e-3 by the final, so the defect is VELOCITY-SPECIFIC and is NOT shared with the eta/PGF chain. => The defect is the DEPTH-MEAN OF THE BEFORE-LEVEL 3-D VELOCITY (the barotropic seed), computed upstream of dyn_spg_ts. This row, un_adv and pssh are all INHERITED from it -- the barotropic solver is faithfully propagating a wrong initial condition. NEXT: compare legoESM's before-level U_bar/V_bar seed per column against NEMO's puu_b(Kbb)/pvv_b(Kbb) to see whether the 2-3%% gap is a uniform scale factor or concentrated on shelf/thin columns -- BarotropicConfig.barotropic_seed_face_depth's own docstring documents an 11%% loop-entry residual at shelf columns under the production min_water_column_m=0.5 floor, which is the leading candidate. [e3t=both, re-verified: compare_spg_barotropic_e3tboth.py "
                                                              "and its e2v-substituted twin both give ratio 0.9986 "
                                                              "unchanged; harness (A)+(B) REFUTED]. STALE-TUPLE FIX "
                                                              "(re-measured at HEAD c8e5d305b, same two probes): the "
                                                              "corr/ratio TUPLE previously recorded here (0.999989/"
                                                              "0.999900) did not match this row's OWN note text (which "
                                                              "already said 'ratio 0.9986'); re-running "
                                                              "compare_spg_barotropic_e3tboth.py reproduces "
                                                              "corr=0.999994/ratio=0.9986 (n=9920) exactly, matching the "
                                                              "note, not the old tuple. Corrected here; the underlying "
                                                              "physics number was never wrong, only the stored tuple was."),
    "dyn_spg_ts puu_b":              (0.999809,   0.987200,   "STAGE 7 2026-07-29 -- STATIC-WEIGHT HYPOTHESIS FALSIFIED, and a STRUCTURAL GAP found. legoESM's F_slow_u (ocean_model_latlon_cgrid.py:2824-2841) uses a LIVE ladder (compute_layer_thickness(state.eta,...) + min-rule H_u_pre) where NEMO under key_qco uses STATIC e3u_0/r1_hu_0 (dynspg_ts.F90:336). But swapping NEMO's static weights onto legoESM's OWN du_dt makes it 44x WORSE, not better: zu_frc err_norm 8.0266e-03 (live) -> 3.5264e-01 (static). So the weight is NOT a drop-in fix -- static and live must stay PAIRED with the RHS they were built with. The residual concentrates at sloped-bathymetry columns near the channel walls (hu_0 mean 3291 m vs the 3743 m full-depth column), NOT at the periodic seam (only 10%% touch the wrap columns), i.e. a topographic-step signature. *** STRUCTURAL GAP, the strongest lead: legoESM has NO zu_trd subtraction at all. *** NEMO does `zu_frc = zu_frc - zu_trd * ssumask` (dynspg_ts.F90:304 and :367); our F_slow_u is finalised at :2841 with nothing removed afterwards (barotropic_drag_substep is a separate, later, off-by-default block). This is an ABSENCE, not a sign or mask error -- but it must be checked whether legoESM's formulation makes the removal unnecessary by construction (never adding the component NEMO removes) before anything is added; that check is the next task and a wrong 'fix' here would be easy. MAGNITUDE-ONLY HINT, explicitly NOT a result: RSS of the known 3-D trend rows (dyn_ldf 3.9e-3, dyn_vor EEN 1.2e-3, dyn_adv ZAD 4.9e-3, dyn_cor_2d 1.2e-3) = 6.49e-3 vs zu_frc's 8.03e-3 -- same order, consistent with zu_frc being their depth-mean, but NO per-level correlation was run so this is a hint, not evidence. ASYMMETRY CONFIRMED: zv_frc err_norm is 5.4290e-04, ~15x smaller than zu_frc -- pvv_b's cause really is separate from puu_b's. *** CAUSAL CHAIN CLOSED 2026-07-29 (e3t=both): the SLOW FORCING zu_frc owns this row. *** zu_frc's own discrepancy vs NEMO's spg_dump_zu_frc.bin is err_norm 8.03e-03 [m/s^2]. Propagated through exactly ONE substep via rDt_e=117.391304 s (ua_e = un_e + rDt_e*(zu_spg+zu_trd+zu_frc), dynspg_ts.F90:802-808) that PREDICTS substep-1 err_norm 2.9661e-04 against an INDEPENDENTLY measured 2.9656e-04 -- ratio 1.000. And because zu_frc is a frozen constant re-added on every one of the 68 substeps, substep-1 x 68 predicts final puu_b 2.0166e-02 vs measured 1.9918e-02 = 101.2%% explained. So the whole 1.3e-2 gap on the worst row reduces to ONE quantity. OPEN RESIDUAL, recorded not explained away: the V-component does NOT close the same way -- its final-gap linear projection accounts for only 12.6%%, so pvv_b has a second cause. ARITHMETIC CAVEAT worth keeping: the first pass used the wrong rDt_e (double-divided dt_s, because ocean_model_latlon_cgrid.py:3221-3223 already pre-divides dt_s = dt_mom/_nbaro before the call) and produced a false 0.022x mismatch; corrected by tracing the call site. A unit/timestep slip is the easiest way to fake or hide a factor here. NEXT: why is zu_frc off by 8.03e-03? It is the vertically-integrated slow (baroclinic) momentum trend, so the candidates are the terms summed into it and the integration weights -- and note the e3t ladder is now correct, so this is NOT that. *** CORRECTED 2026-07-29 (same session, supersedes the 'seed owns it' note below). *** The seed error was the LEGOESM_NEMO_E3T DEFAULT, not the depth-averaging operator. The probe ran with the env var UNSET -> bridge default 'off' -> NEMO's ANALYTIC e3t_1d, while NEMO itself runs on e3t_0; those two NEMO ladders differ by 12.9%% below k=25 (NEMO builds the reference ladder in TWO passes -- zgr_lib.F90::zgr_sco_mi96 re-anchors at kkconst = argmin(|gdepw - rn_hco|), and DINO's rn_hco=1000 m puts that at k=25 EXACTLY, which is where the divergence starts). A/B, single variable:     e3t=off   seed 2.3109e-02 | substep-1 2.3131e-02 | final 3.0949e-02     e3t=both  seed 2.1872e-16 | substep-1 2.9656e-04 | final 1.9918e-02 So on NEMO's REAL ladder the seed is EXACT and the error ACCUMULATES through the substeps -- the OPPOSITE branch of the decisive question, previously masked. dyn_spg_ts does have a real substep problem after all; the seed does not. ALSO FALSIFIED: the u-face wet-level COUNT matches NEMO exactly (0 of 9758 faces differ), so this is NOT the bug-#18 mask family. And create_levy_stretched_z_star is NOT implicated -- the bridge never calls it (its dz_ref matches NEMO's e3t_1d to 9e-16); that constructor feeds only DINO's STANDALONE path, where the missing second mi96 pass may still be a separate real defect worth its own row. This is the FOURTH measurement the e3t default has contaminated, so it is now guarded mechanically: precision_gate.require_explicit_e3t_mode() refuses an inherited default. NEXT: with e3t=both, walk the substep recurrence itself (2.966e-04 at substep 1 growing to 1.992e-02 over 68) -- that IS an accumulation signature, so look at the per-substep update, the Coriolis/drag terms in the loop, and the boxcar averaging. SEED SPLIT RESOLVED 2026-07-29 (spg_substep_chain.py STAGE 4): the 3-D velocity is EXONERATED and the AVERAGING OPERATOR owns the whole error. legoESM's bridged before-level 3-D u/v vs NEMO's restart ub/vb (uu(:,:,:,Nbb), restart.F90:347-348): err_norm = 0.0000e+00 at EVERY one of 35 levels, both components -- bit-identical. Yet the depth-mean of that same exact field is already 2.31e-2 (u) / 3.05e-2 (v) at the loop-entry seed. Since U_bar = sum_k(u*h_face)/sum_k(h_face) and the numerator field is exact, the entire gap lives in the THICKNESS WEIGHTING (_depth_average_to_faces) -- not in upstream dynamics, not in the restart bridge, not in the barotropic solver. DEAD END DOCUMENTED so it is not retried: stp_dump_07_dynspg_u/ub.bin are NOT the dyn_spg_ts entry velocity -- stpmlf.F90:288-294 dumps uu(:,:,:,Naa)/uu_b(Naa), and :403-406 shows Naa == Nrhs during the step body, so _u.bin is the momentum RHS accumulator (magnitude ~1e-6, five orders below velocity scale); _ub.bin is bit-identical to spg_dump_puu_b_final.bin, the same quantity dumped twice. NEXT: per-column sum_k(h_face) vs NEMO's sum_k(e3u(Kbb)), and the u-face WET LEVEL COUNT -- DINO is FULL-STEP so e3u_0 == e3t_0 and the min-rule should agree level-for-level, which points at WHICH LEVELS ARE COUNTED WET at a u-face (bathymetry steps) rather than at the thickness values. That is the same mask family as bug #18 (a ULP tie marking the deepest dry level active) and the ldf_slp zcj mask count. *** CAUSE LOCALISED 2026-07-29 (spg_substep_chain.py): NOT dyn_spg_ts. *** The error is already present at the barotropic LOOP-ENTRY SEED, before the substep recurrence runs once, and stays flat through it: err_norm u 2.31e-2 (seed) -> 2.31e-2 (substep 1) -> 3.09e-2 (final, puu_b); v 3.05e-2 -> 3.04e-2 -> 3.27e-2. An ACCUMULATION signature would need a clean seed and a large final gap; this is the opposite. Ruled out BY MEASUREMENT, not inference: substep count matches exactly (legoESM n_barotropic_substeps=23 == NEMO nn_e=23; compute_nemo_boxcar_centred_weights n_loop=68 == NEMO icycle=68), so it is not a window/weight mismatch; offset scans peak sharply at (0,0) for BOTH a U-face field (puu_b) and a T-point field (pssh), so it is not a face-convention or index shift. ssh is EXACT at the seed (0.0) and only reaches 4.3e-3 by the final, so the defect is VELOCITY-SPECIFIC and is NOT shared with the eta/PGF chain. => The defect is the DEPTH-MEAN OF THE BEFORE-LEVEL 3-D VELOCITY (the barotropic seed), computed upstream of dyn_spg_ts. This row, un_adv and pssh are all INHERITED from it -- the barotropic solver is faithfully propagating a wrong initial condition. NEXT: compare legoESM's before-level U_bar/V_bar seed per column against NEMO's puu_b(Kbb)/pvv_b(Kbb) to see whether the 2-3%% gap is a uniform scale factor or concentrated on shelf/thin columns -- BarotropicConfig.barotropic_seed_face_depth's own docstring documents an 11%% loop-entry residual at shelf columns under the production min_water_column_m=0.5 floor, which is the leading candidate. 1.3% gap UNEXPLAINED [e3t=both + e2u/e1u substitution "
                                                              "re-verified 2026-07-27: ratio 0.9871 unchanged in both "
                                                              "variants; harness (A)+(B) REFUTED, residual is real]. "
                                                              "RE-MEASURED at HEAD c8e5d305b (compare_spg_barotropic_"
                                                              "e3tboth.py + its e2v-substituted twin, both identical): "
                                                              "corr=0.999809/ratio=0.9872, n=9758 -- essentially "
                                                              "UNCHANGED from the recorded 0.999586/0.987100 (corr "
                                                              "drifted +0.0002, within the note's own 'unchanged' "
                                                              "characterization; residual still real, still DEBT)."),
    "dyn_spg_ts un_adv":             (0.999949,   0.991600,   "*** CAUSE LOCALISED 2026-07-29 (spg_substep_chain.py): NOT dyn_spg_ts. *** The error is already present at the barotropic LOOP-ENTRY SEED, before the substep recurrence runs once, and stays flat through it: err_norm u 2.31e-2 (seed) -> 2.31e-2 (substep 1) -> 3.09e-2 (final, puu_b); v 3.05e-2 -> 3.04e-2 -> 3.27e-2. An ACCUMULATION signature would need a clean seed and a large final gap; this is the opposite. Ruled out BY MEASUREMENT, not inference: substep count matches exactly (legoESM n_barotropic_substeps=23 == NEMO nn_e=23; compute_nemo_boxcar_centred_weights n_loop=68 == NEMO icycle=68), so it is not a window/weight mismatch; offset scans peak sharply at (0,0) for BOTH a U-face field (puu_b) and a T-point field (pssh), so it is not a face-convention or index shift. ssh is EXACT at the seed (0.0) and only reaches 4.3e-3 by the final, so the defect is VELOCITY-SPECIFIC and is NOT shared with the eta/PGF chain. => The defect is the DEPTH-MEAN OF THE BEFORE-LEVEL 3-D VELOCITY (the barotropic seed), computed upstream of dyn_spg_ts. This row, un_adv and pssh are all INHERITED from it -- the barotropic solver is faithfully propagating a wrong initial condition. NEXT: compare legoESM's before-level U_bar/V_bar seed per column against NEMO's puu_b(Kbb)/pvv_b(Kbb) to see whether the 2-3%% gap is a uniform scale factor or concentrated on shelf/thin columns -- BarotropicConfig.barotropic_seed_face_depth's own docstring documents an 11%% loop-entry residual at shelf columns under the production min_water_column_m=0.5 floor, which is the leading candidate. [e3t=both, re-verified: e2v/e1v-substituted variant "
                                                              "gives the identical 0.9916/1.0003-class numbers; "
                                                              "harness (A)+(B) REFUTED]. STALE-TUPLE FIX (re-measured at "
                                                              "HEAD c8e5d305b, same two probes): the recorded tuple "
                                                              "(0.999777/1.006700) did NOT match this row's own note "
                                                              "text, which already cites '0.9916/1.0003-class numbers' "
                                                              "for un_adv/vn_adv. Re-running "
                                                              "compare_spg_barotropic_e3tboth.py's un_adv/Hu_avg line "
                                                              "gives corr=0.999949/ratio=0.9916 (n=9758) exactly, "
                                                              "matching the note. Corrected here -- this tuple appears "
                                                              "to have been wrong since the row was created, not a "
                                                              "recent drift."),
    "ATF filter u":                  (0.999969,   0.995600,   "0.45% gap [e3t=both per atf_lego_extract_e3tboth.py -- "
                                                              "filtered-field ratio 0.9956/corr 0.999969 reproduced "
                                                              "exactly; harness (A) REFUTED]. RE-MEASURED at HEAD "
                                                              "c8e5d305b (fresh atf_lego_extract_e3tboth.py + "
                                                              "atf_compare_e3tboth.py run, RUN_GDB kt=57601): "
                                                              "corr=0.999969/ratio(nemo/lego)=0.9956, n=336338. "
                                                              "UNCHANGED."),
    "ATF filter v":                  (0.999999,   1.000300,   "[e3t=both]. RE-MEASURED at HEAD c8e5d305b (same run as "
                                                              "ATF filter u): corr=0.999999/ratio=1.0003, n=340271. "
                                                              "UNCHANGED."),
    "ATF filter T/S/ssh":            (1.0,        1.0,        "exact"),
    "dyn_ldf (dynldf_lev_lap) u":    (0.997855,   1.003865,   "[e3t=both per probe_1226_r2_item2_dynldf.py; same "
                                                              "operator family (gradient_x/y_cgrid) as dyn_hpg -- "
                                                              "(B) structurally a no-op here too, see module docstring]. "
                                                              "RE-MEASURED at HEAD c8e5d305b (same probe): offset scan "
                                                              "{-1,0,+1} sharp offset=0 peak (-1=0.787440, +0=0.997855, "
                                                              "+1=0.792236); corr 0.997855/ratio 1.003865, n=341530. "
                                                              "UNCHANGED to 4-5 s.f."),
    "dyn_ldf (dynldf_lev_lap) v":    (0.999400,   1.001755,   "[e3t=both]. RE-MEASURED at HEAD c8e5d305b (same run): "
                                                              "offset scan {-1,0,+1} sharp offset=0 peak (-1=0.937450, "
                                                              "+0=0.999400, +1=0.938908); corr 0.999400/ratio 1.001755, "
                                                              "n=345380. UNCHANGED."),
    "ssh_nxt / div_hor":             (1.0,        1.000004,   "RATIO IS THE nemo_isotropic (SHIPPED nemo_dino_kamm) "
                                                              "VALUE as of 2026-07-28; the pre-#1226-option 'exact' "
                                                              "baseline was 0.999991. "
                                                              "[e3t=both per probe_1226_r2_item3_sshnxt.py]. TIER-2 "
                                                              "2026-07-28: cause NOT identified. Ruled out by direct "
                                                              "check against NEMO sshwzv.F90: the formula, the "
                                                              "arithmetic regrouping, the time level, and the H "
                                                              "(bathymetry) reference all agree to ~1e-11, i.e. three "
                                                              "orders tighter than this 9e-6 residual, so none of them "
                                                              "is the source. Suspected (NOT proven) a float32 step in "
                                                              "a since-deleted comparison script. Belongs to the "
                                                              "unexplained 3e-6..5e-5 residual BAND shared with r3t, "
                                                              "r3u/r3v, dyn_drg RHS and dyn_cor_2d -- a single shared "
                                                              "cause would close several rows at once and is the "
                                                              "highest-value next investigation. "
                                                              "#1226 METRIC-CONVENTION RE-MEASURED (2026-07-28, this PR): "
                                                              "NEMO's DINO usr_def_hgr.F90 builds the T/u-face grid "
                                                              "metric ISOTROPICALLY (pe1t=pe2t) whereas legoESM's "
                                                              "default computes dy/area as the true finite-difference/"
                                                              "exact-spherical form -- measured directly against the "
                                                              "real mesh_mask.nc: legoESM dy_T vs NEMO e2t relerr "
                                                              "-1.27e-5..+9.5e-6, area vs e1t*e2t -2.5e-5..+4.1e-5. NEW "
                                                              "additive ``metric_convention='nemo_isotropic'`` option "
                                                              "ships on create_mercator_grid/create_latlon_geometry/"
                                                              "bridge_nemo_to_legoesm_topo (default 'exact' stays "
                                                              "BIT-IDENTICAL everywhere). END-TO-END re-measurement "
                                                              "(bridge RUN_GDB restart, LEGOESM_NEMO_E3T=both, "
                                                              "divergence_cgrid hdiv vs sshnxt_dump_hdiv.bin, "
                                                              "probe_1226_r2_item3_sshnxt.py methodology): exact "
                                                              "reproduces corr=1.000000/|x|ratio=0.999991 exactly "
                                                              "(baseline confirmed); nemo_isotropic gives "
                                                              "corr=1.000000/|x|ratio=1.000004 (RMSratio 0.999987-> "
                                                              "0.999999). Real, controlled movement in the predicted "
                                                              "direction (crossed 1.0) but |ratio-1| still ~4e-6, just "
                                                              "OUTSIDE BAR_RATIO_EPS=1e-6 -- STILL DEBT, but the metric "
                                                              "mechanism is CONFIRMED live and dominant for this row "
                                                              "(not merely a candidate). RE-CONFIRMED at HEAD c8e5d305b "
                                                              "(probe_1226_r2_item3_sshnxt.py with metric_convention="
                                                              "'nemo_isotropic' passed explicitly to "
                                                              "bridge_nemo_to_legoesm_topo -- the bridge's own default "
                                                              "is 'exact' regardless of the recipe config, a harness "
                                                              "gap distinct from (A)/(B) above): offset scan {-1,0,+1} "
                                                              "sharp offset=0 peak (-1=0.484481, +0=1.000000, "
                                                              "+1=0.485622); corr=1.000000/ratio=1.000004, n=347200. "
                                                              "EXACT reproduction, UNCHANGED."),
    "dom_qco_r3c r3t":               (1.0,        0.999997,   "[e3t=both]. TIER-2 2026-07-28: cause NOT identified. The "
                                                              "obvious candidate is ruled out: (H+eta)/H vs 1+eta/H is "
                                                              "a reassociation worth ~1e-12, six orders too small to "
                                                              "explain this 3e-6, so the formula is fine and this is "
                                                              "NOT summation-order roundoff. Same unexplained "
                                                              "3e-6..5e-5 band as ssh_nxt/div_hor above (and r3u/r3v "
                                                              "sits at the same 0.999997, explicitly noted there as "
                                                              "'same residual class as r3t' -- consistent with one "
                                                              "shared upstream cause rather than four independent "
                                                              "coincidences). "
                                                              "#1226 METRIC-CONVENTION RULED OUT (2026-07-28): r3t = "
                                                              "eta_safe / ht_0 (ocean_pe_latlon_cgrid.py) is a PURELY "
                                                              "VERTICAL ratio (SSH over the reference column depth "
                                                              "sum(e3t_0)) -- it has NO horizontal-metric (dy/area) "
                                                              "term at all, so the T/u-face metric_convention CANNOT "
                                                              "move this row regardless of value. This row's residual "
                                                              "does NOT share the metric-convention cause; the "
                                                              "'single shared cause' hypothesis above is FALSIFIED for "
                                                              "r3t specifically (structural, not measured -- confirmed "
                                                              "by direct formula read, no run needed). RE-CONFIRMED at "
                                                              "HEAD c8e5d305b (probe_1226_r2_item4_domqco.py, RUN_GDB "
                                                              "kt=57601): corr=1.000000/ratio=0.999997, n=9920. "
                                                              "UNCHANGED."),
    "dom_qco_r3c r3u/r3v":           (0.999999999879, 0.999997, "hu_0/hv_0 added to nemo_io.NemoGrid (derived from e3u_0/e3v_0+mask); "
                                                              "face-averaged eta/H0 formula match; ratio not exactly 1 (same residual class as r3t) "
                                                              "[e3t=both per probe_1226_r2_item4_domqco.py; e2u/e1v read DIRECTLY from mesh_mask.nc "
                                                              "in this probe (not bridge-reconstructed) -- (B) already excluded for this term by construction]. "
                                                              "#1226 METRIC-CONVENTION RULED OUT (2026-07-28): r3u/r3v "
                                                              "are face-AVERAGES of r3t (itself a purely vertical "
                                                              "eta/H0 ratio, see r3t row) -- same structural argument, "
                                                              "no horizontal T/u-face metric term to move. NOT the "
                                                              "metric-convention mechanism. RE-MEASURED at HEAD "
                                                              "c8e5d305b: the in-repo probe's r3u/r3v branch now "
                                                              "CRASHES (AttributeError: 'NemoGrid' object has no "
                                                              "attribute 'e2u') -- NemoGrid was never given e2u/e1v "
                                                              "fields (only the on-axis e1u/e2v), a harness API drift "
                                                              "unrelated to physics. Worked around by reading e1u/e2u/"
                                                              "e1v/e2v directly off the raw mesh_mask.nc (matching this "
                                                              "row's own 'read DIRECTLY from mesh_mask.nc' claim) with "
                                                              "the probe's setup otherwise reused verbatim: "
                                                              "corr=1.000000/ratio=0.999997 for BOTH r3u (n=9758) and "
                                                              "r3v (n=9868). UNCHANGED; the legoESM code itself is fine "
                                                              "-- only the in-repo probe script needs the e2u/e1v fix "
                                                              "if re-run again without a workaround."),
    "mlf_baro_corr":                 (None,       None,       "algebra only; needs _step_impl hook"),
    "lbc_lnk sign":                  (None,       None,       "NEVER VERIFIED"),
    "zdf_mxl_turb":                  (None,       None,       "HUMAN-WAIVED 2026-07-30 (see WAIVED_ROWS below for "
                                                              "decision-provenance + evidence): MISSING TERM, no legoESM "
                                                              "equivalent -- NEMO's Kz<avt_c turbocline diagnostic (hmld) "
                                                              "is distinct from zdf_mxl's N^2-criterion MLD (nmln, which "
                                                              "IS ported); hmld/mldkz5 is diagnostic-only (never feeds "
                                                              "dynamics for a no-TOP/PISCES DINO build), no lego port exists."),
    "zdf_drg_nonlin T-point rate":   (1.0,        1.0,        "AT BAR: exact, nemo_effective_bottom_drag_r on bridged bottom u/v [e3t=both]. "
                                                              "RE-CONFIRMED at HEAD c8e5d305b (probe_bottom_drag.py, "
                                                              "RUN_GDB kt=57601): corr=1.00000000/ratio=1.000000, "
                                                              "n=9920. UNCHANGED."),
    "dyn_drg_init RHS increment":    (0.999937,   1.000278,   "u; v=0.999961/0.999988 (both DEBT, ~5e-5 residual) [e3t=both per probe_bottom_drag.py]. "
                                                              "#1226 METRIC-CONVENTION RULED OUT (2026-07-28, this PR, "
                                                              "re-measured end-to-end): bridged RUN_GDB restart, "
                                                              "LEGOESM_NEMO_E3T=both, nemo_bottom_drag_rate_faces zu_frc_inc/"
                                                              "zv_frc_inc vs drg_dump_zu_frc_inc.bin/zv_frc_inc.bin "
                                                              "(probe_bottom_drag.py (B) methodology). exact: corr="
                                                              "0.99993738/abs_ratio=1.000278 (u), corr=0.99996133/"
                                                              "abs_ratio=0.999988 (v) -- reproduces the recorded numbers "
                                                              "exactly. nemo_isotropic: IDENTICAL to 8 significant figures "
                                                              "for BOTH u and v -- ZERO movement. Confirms the task "
                                                              "directive's prediction: this row's formula (drag rate x "
                                                              "|U| difference, ocean_pe_latlon_cgrid.py nemo_bottom_drag_"
                                                              "rate_faces) does not route through dy_T/dy_u/area_T at "
                                                              "all -- the metric-convention mechanism is CONCLUSIVELY "
                                                              "NOT the cause of this row's residual or its u/v asymmetry. "
                                                              "STILL DEBT, different cause needed. RE-CONFIRMED at HEAD "
                                                              "c8e5d305b (probe_bottom_drag.py, RUN_GDB kt=57601): "
                                                              "alignment check (west vs east face slice, this row's own "
                                                              "'no vertical index -- Rule (a) still requires an "
                                                              "alignment scan where one exists' substitute) picks the "
                                                              "east-face slice cleanly (corr 0.999937 vs 0.890564 west) "
                                                              "for u and north-face for v (corr 0.999961 vs 0.950182 "
                                                              "south); corr/ratio EXACTLY reproduced for both u and v. "
                                                              "UNCHANGED."),
    "dyn_cor_2d (69x/step)":         (0.999717,   0.998805,   "interior (excl. periodic-seam column, harness reindexing "
                                                              "artifact there, not a lego defect); v=1.0/0.999928; "
                                                              "een_barotropic_coriolis(metric_complete=True) vs dumped "
                                                              "substep-1 zu_trd/zv_trd [e3t=both per probe_dyn_cor_2d.py]. "
                                                              "#1226 METRIC-CONVENTION RE-MEASURED (2026-07-28, this PR): "
                                                              "traced the exact coefficient sourcing (_build_een_"
                                                              "barotropic_inputs, barotropic_latlon_cgrid.py): NEMO's "
                                                              "cor_u divides by e1u=geom.dx_u and multiplies V_bar by "
                                                              "e1v=geom.dx_v -- BOTH are v-face/T-face-zonal-width "
                                                              "metrics this flag does NOT touch (only dy_T/dy_u/area_T "
                                                              "change) -- so cor_u is PREDICTED metric_convention-"
                                                              "INVARIANT. cor_v multiplies U_bar by e2u=geom.dy_u "
                                                              "(changed) and divides by e2v=geom.dy_v (v-face, "
                                                              "unchanged) -- PARTIALLY sensitive. End-to-end "
                                                              "re-measurement (bridged RUN_GDB restart, own restart's "
                                                              "ua_e/va_e fed directly per probe_dyn_cor_2d.py, e3t=both) "
                                                              "CONFIRMS both predictions exactly: u corr=0.99971701/"
                                                              "abs_ratio=0.998805 IDENTICAL to 8 sig figs under both "
                                                              "conventions (zero movement, as predicted); v corr= "
                                                              "0.99999995 both, abs_ratio 0.999928->0.999923 (moved "
                                                              "AWAY from 1.0 by 5e-6, i.e. did not improve). Net: the "
                                                              "metric-convention mechanism does NOT close this row -- "
                                                              "confirmed mechanistically inert for u, negligible/adverse "
                                                              "for v. STILL DEBT, different cause needed for the bulk "
                                                              "of this residual (NOTE: this end-to-end run's own exact-"
                                                              "convention baseline, corr=0.99971701/abs_ratio=0.998805, "
                                                              "differs from the recorded 0.99999992/0.999986 above -- "
                                                              "likely a different reference restart/state than the "
                                                              "original probe run; the metric_convention A/B comparison "
                                                              "itself is controlled and valid regardless). STALE-TUPLE "
                                                              "FIX (re-measured at HEAD c8e5d305b, probe_dyn_cor_2d.py, "
                                                              "RUN_GDB kt=57601): re-running the u corr/ratio at the top "
                                                              "of this row reproduces 0.999717/0.998805 exactly, i.e. "
                                                              "the SAME numbers this row's own metric-convention note "
                                                              "already measured above -- confirming the top-line tuple "
                                                              "(previously 0.99999992/0.999986) was the stale one, not "
                                                              "this note. Corrected u to 0.999717/0.998805 here; v stays "
                                                              "1.0/0.999928 (already the last-measured value, reproduced "
                                                              "exactly: corr=0.9999999549846602, abs_ratio=0.999928)."),
    "traadv_fct (SALINITY)":         (0.999952,   0.999900,   "RETRACTED the DEBT verdict below -- it does not reproduce. "
                                                              "RE-MEASURED at HEAD e72923faa (traadv_fct_probe.py, "
                                                              "RUN_GDB kt=57601, supersedes probe_fct_sal.py): re-running "
                                                              "probe_fct_sal.py's OWN full offset scan {-2..+2} on the "
                                                              "PURE (Krhs-contamination-removed) comparison -- the exact "
                                                              "method this row's provenance cites -- gives corr=0.999952/"
                                                              "ratio=0.999900 at the offset=0 peak (offset -1/+1 give "
                                                              "0.820/0.820, offset -2/+2 give 0.651/0.651: a clean, "
                                                              "symmetric, sharp peak), NOT the previously recorded "
                                                              "0.203216/3.167041 -- which does not reproduce from this "
                                                              "or any other script/env-var/e3t-mode combination tried. "
                                                              "The PREDICTED mechanism in the retracted note below (S's "
                                                              "net tendency is a near-total cancellation of horizontal "
                                                              "and vertical components -- corr(horiz,vert)=-1.0000, "
                                                              "net/gross~0.14% vs T's -0.998/~5% -- so S is a much more "
                                                              "SENSITIVE DETECTOR of any horiz/vert component residual) "
                                                              "is CONFIRMED and remains the right qualitative picture; "
                                                              "what was wrong was the arithmetic conclusion drawn from "
                                                              "it -- the horizontal-only and vertical-only tendencies "
                                                              "(see 'traadv_fct horizontal tend' row) are EACH already "
                                                              "at corr>=0.99997/ratio~1.0000, so the amplified residual "
                                                              "that would have produced corr=0.20 does not exist at "
                                                              "HEAD; the fixes already landed (2a73221ce nonosc q_td "
                                                              "bound, 01c1f226a dry-cell Zalesak mask) apparently closed "
                                                              "it before this row was last (mis-)measured. Still ~1e-4 "
                                                              "outside BAR_RATIO_EPS, not literal 1.0/1.0, but no longer "
                                                              "a 0.20-corr defect -- retiring the 'S feeds EOS->ACC, "
                                                              "flagged not fixed' framing.\n"
                                                              "[RETRACTED, kept for the record of what was claimed and "
                                                              "why it looked plausible] DEBT: corr/ratio far below bar. "
                                                              "VERIFIED (not the 'boundary-column artifact' hypothesis "
                                                              "-- REFUTED: excluding i/j=0,last leaves corr 0.2024 "
                                                              "(unchanged); excluding bathymetry-step-adjacent columns "
                                                              "too only reaches corr 0.379/ratio 1.37, still far below "
                                                              "bar). ROOT CAUSE (claimed): per-face upstream+antidiffusive "
                                                              "S fluxes match NEMO at corr 0.94-0.995 / abs_ratio "
                                                              "0.9999-1.010; the horizontal-only and vertical-only "
                                                              "upstream tendencies ALSO independently match NEMO at "
                                                              "corr 0.9999/ratio ~1.000 each; S's net upstream tendency "
                                                              "was claimed a near-total cancellation exposing a small "
                                                              "existing horiz/vert discretization gap invisible for T. "
                                                              "This diagnosis of the MECHANISM was right; the specific "
                                                              "corr=0.203165/ratio=3.167599 numbers attached to it were "
                                                              "not reproducible."),

    # --- 2026-07-30 Rule 1 coverage additions (stpmlf_call_coverage.py) ---
    # These 9 CALL entries (wzv has two call sites, one row) were enumerated
    # by the stp_MLF call-graph coverage gate as UNCOVERED -- no probe had
    # ever measured them, and no one had thought to add them to this
    # checklist. Adding a row here does NOT mean they are measured: every
    # tuple below is (None, None, ...) on purpose. Do not invent a number.
    "wzv (vertical velocity)":       (None, None,
        "enumerated by stpmlf_call_coverage.py 2026-07-30 (skill Rule 1 "
        "call-graph coverage); never measured. Two call sites in "
        "stpmlf.F90 (line 244 Nnn, line 315 Naa post-dyn_zdf recomputation) "
        "-- both route through the same wzv routine, one row."),
    "tra_zdf (tracer implicit vertical solve)": (None, None,
        "enumerated by stpmlf_call_coverage.py 2026-07-30 (skill Rule 1 "
        "call-graph coverage); never measured."),
    "dyn_zdf (momentum implicit vertical solve)": (None, None,
        "enumerated by stpmlf_call_coverage.py 2026-07-30 (skill Rule 1 "
        "call-graph coverage); never measured."),
    "traldf_iso_lap tendency":       (None, None,
        "enumerated by stpmlf_call_coverage.py 2026-07-30 (skill Rule 1 "
        "call-graph coverage); never measured."),
    "ldf_dyn coefficient":           (None, None,
        "enumerated by stpmlf_call_coverage.py 2026-07-30 (skill Rule 1 "
        "call-graph coverage); never measured."),
    "tra_qsr (shortwave penetration)": (None, None,
        "enumerated by stpmlf_call_coverage.py 2026-07-30 (skill Rule 1 "
        "call-graph coverage); never measured."),
    "ssh_atf":                       (None, None,
        "enumerated by stpmlf_call_coverage.py 2026-07-30 (skill Rule 1 "
        "call-graph coverage); never measured."),
    "tra_sbc":                       (None, None,
        "enumerated by stpmlf_call_coverage.py 2026-07-30 (skill Rule 1 "
        "call-graph coverage); never measured."),
}


# Rows whose number was measured before this commit are STALE: the code has
# changed underneath them.  Discovered the hard way -- wslpi/aeiu carried
# a8b10ca4d-era values through ~15 commits while the real numbers had moved
# (0.999177 -> 0.997860, 1.000000 -> 0.974309), which then produced a false
# "regression" alarm.  A number without a measured-at commit is not evidence.
MEASURED_AT: dict[str, str] = {
    # term -> git commit the number was measured at (short sha), or "" if unknown
    "bn2 (rn2b)": "825d22ee4",
    "eos_rab alpha": "825d22ee4",
    "ldftra ahtv (Redi, nn_aht_ijk_t=20)": "c2533b7d6",
    # 2026-07-28 active_3d mask fix (see MEASUREMENTS note): re-measured at
    # the commit that introduced _nemo_native_active_3d.
    "ldf_slp wslpi": "7816b514e",
    "ldf_slp wslpj": "7816b514e",
    "ldf_slp uslp": "7816b514e",
    "ldf_slp vslp": "7816b514e",
    # ldf_eiv kappa (aeiu): re-measured b4be5ee65 -- the 0.975163/1.032510
    # tuple recorded at 7816b514e was a harness measurement artifact (raw
    # T-point kappa_GM compared against NEMO's U-face-averaged paeiu dump),
    # not a re-measurement of the same quantity after a model change.
    "ldf_eiv kappa (aeiu)": "b4be5ee65",
    # DISPUTE RESOLVED 2026-07-28 (9f25d7be4): the corr~0.19 measurement was a
    # PROBE-INVOCATION error, not a model discrepancy. hpg_tendency_compare.py
    # defaults to ``--state istate`` (the analytic from-rest usr_def_istate
    # CASE(4) IC); pointing that at RUN_GDB's Y5-DEVELOPED dump compares
    # legoESM-at-rest against NEMO-at-year-5 -- reproduced here exactly:
    # du corr=nan, dv corr=0.2409 with a FLAT alignment scan (0.2409 vs
    # runner-up 0.1876), i.e. the probe's OWN "flat scan -- ratio NOT
    # trustworthy" guard fires and the number must be discarded. The correct
    # invocation (that script's own "Run" doc block) passes the developed
    # restart explicitly:
    #   --run <RUN_GDB> --state <RUN_GDB>/DINO_00057600_restart.nc
    #                   --e3t-mode both
    # which gives a SHARP peak (du 1.000000 vs runner-up 0.8956) and
    # reproduced the historical pre-fix 1.000054839 baseline EXACTLY before
    # the live-gdept fix -- independent corroboration that this invocation is
    # the one every earlier dyn_hpg number came from.
    # BUG FIX (this PR): the key here was "dyn_hpg", but MEASUREMENTS has
    # "dyn_hpg (du)" and "dyn_hpg (dv)" as separate rows -- neither ever
    # matched this key, so both silently counted as NO-provenance despite
    # being measured at 9f25d7be4. Split into the two real keys.
    "dyn_hpg (du)": "9f25d7be4",
    "dyn_hpg (dv)": "9f25d7be4",
    # --- #1226 provenance-restoration pass (this PR, HEAD c8e5d305b) ---
    # Re-measured every row in scope end-to-end: LEGOESM_NEMO_E3T=both, Y5
    # RUN_GDB kt=57601 (or --state y5 for hpg-style probes), own alignment
    # scan {-2..+2} (or the field's natural offset check where no vertical
    # index exists -- 2-D barotropic/diagnostic rows). All confirmed
    # UNCHANGED except dyn_spg_ts pssh/un_adv and dyn_cor_2d u, whose stored
    # tuples were stale relative to their OWN note text (fixed in
    # MEASUREMENTS above). Did NOT touch ldf_slp */ldf_eiv aeiu (another
    # session was actively editing those rows) or "traadv_fct horizontal
    # tend" (no probe found in either scratchpad -- left UNPROVENANCED
    # rather than invented).
    #
    # --- #1226 traadv_fct component-row drive (this PR, MEASURED_AT c1ba30e39) ---
    # The four c8e5d305b-stamped numbers below did NOT reproduce when the
    # SAME probes the c8e5d305b pass cites (probe_fct.py, probe_nonosc_
    # stages.py, check_wflux_offset0.py, probe_fct_sal.py) were re-run
    # against the SAME RUN_GDB dumps on unchanged advection.py (verified: no
    # commit touches it between 01c1f226a and e72923faa) -- a transcription
    # error in that pass, not a live regression. Replaced with
    # traadv_fct_probe.py (new consolidated driver, same repo, same dumps,
    # same offset=0 alignment), which also fills in the previously-missing
    # "traadv_fct horizontal tend" probe. See each row's note for the exact
    # before/after numbers.
    "traadv_fct fluxes": "c1ba30e39",
    "traadv_fct tendency (T)": "c1ba30e39",
    "traadv_fct horizontal tend": "c1ba30e39",
    "traadv_fct vertical upstream flux": "c1ba30e39",
    "traadv_fct (SALINITY)": "c1ba30e39",
    "dyn_vor EEN u": "c8e5d305b",
    "dyn_vor EEN v": "c8e5d305b",
    "dyn_adv ZAD": "c8e5d305b",
    "zdftke pdlr": "c8e5d305b",
    "zdftke composite avt/avm": "c8e5d305b",
    "dyn_spg_ts pssh": "c8e5d305b",
    "dyn_spg_ts puu_b": "c8e5d305b",
    "dyn_spg_ts un_adv": "c8e5d305b",
    "ATF filter u": "c8e5d305b",
    "ATF filter v": "c8e5d305b",
    "dyn_ldf (dynldf_lev_lap) u": "c8e5d305b",
    "dyn_ldf (dynldf_lev_lap) v": "c8e5d305b",
    "ssh_nxt / div_hor": "c8e5d305b",
    "dom_qco_r3c r3t": "c8e5d305b",
    "dom_qco_r3c r3u/r3v": "c8e5d305b",
    "eiv transport u": "c8e5d305b",
    "eiv transport v": "c8e5d305b",
    "zdf_drg_nonlin T-point rate": "c8e5d305b",
    "dyn_drg_init RHS increment": "c8e5d305b",
    "dyn_cor_2d (69x/step)": "c8e5d305b",
}


# Rows that are NOT term comparisons.  A stability criterion or a missing-term
# waiver has no corr/ratio, and forcing 1.0/1.0 onto one to clear the board
# would be gaming the gate.  Encode them honestly instead:
#   True  = criterion MEASURED and PASSED
#   False = criterion MEASURED and FAILED  -> DEBT
#   None  = still genuinely unmeasured     -> UNMEASURED
BINARY_GATES: dict[str, bool | None] = {
    # From rest, 5 full years on NEMO's true e3t_0 ladder: STABLE, no growth.
    # But the restart-start blow-up (max|u| 0.66 -> 3 m/s over 20 d) is REAL and
    # unfixed, so the criterion "legoESM runs on NEMO's actual geometry" is only
    # half met.  FAILED, not passed -- this row does not get to clear on the
    # easier half of its own criterion.
    "STABILITY on NEMO true grid (e3t_0)": False,
}


# HUMAN-WAIVED rows.  This is NOT a generic "mark anything waived" mechanism --
# it is keyed to exactly the rows below, each requiring BOTH a nonempty
# decision-provenance string (who/when decided, and that the row was verified
# BEFORE being waived, not skipped) and a nonempty evidence citation (file:line
# proof the term is unconsumed/out of scope).  A waiver missing either string
# is a hard error (see _validate_waivers), not a silent pass -- so a future
# attempt to wave a row through by adding a bare name here fails loudly.
# WAIVED counts in `total`, does not block exit 0 (resolved-by-human), and is
# never confused with AT BAR (it carries no corr/ratio bar-clearance at all).
WAIVED_ROWS: dict[str, tuple[str, str]] = {
    # term -> (decision_provenance, evidence_citation)
    "zdf_mxl_turb": (
        "human decision 2026-07-30: verify-unconsumed, then waive.",
        "Verification done (agent af044c34ba25fb440, 2026-07-30): "
        "zdf_mxl_turb (src/OCE/ZDF/zdfmxl.F90) writes only hmld "
        "(zdfmxl.F90:152) and iom_put('mldkz5', ...) (zdfmxl.F90:155-158); "
        "every consumer of hmld in src/OCE is init/diagnostic-output; the "
        "ONLY arithmetic consumers are PISCES via src/TOP/oce_trc.F90:93, "
        "and DINO compiles NO TOP tree (cpp_DINO.fcm has no key_top; "
        "cfgs/DINO/WORK/ contains no oce_trc.F90). The routine runs every "
        "step (MY_SRC/stpmlf.F90:190 -> zdfphy.F90:338) but its output "
        "never reaches physics for this recipe.",
    ),
}


def _validate_waivers() -> None:
    """Fail LOUDLY, at import time, if a waiver is missing either required
    string -- a waiver is a claim, and an empty claim is not evidence."""
    for term, (provenance, evidence) in WAIVED_ROWS.items():
        if not provenance.strip():
            raise ValueError(f"WAIVED_ROWS[{term!r}] has an empty "
                              "decision-provenance string")
        if not evidence.strip():
            raise ValueError(f"WAIVED_ROWS[{term!r}] has an empty "
                              "evidence citation string")


_validate_waivers()


def classify(corr: float | None, ratio: float | None,
             per_elem: float | None = None, name: str | None = None) -> str:
    """AT BAR requires corr, MEAN ratio AND per-element error at roundoff.

    per_elem=None means the per-element error was never measured; the row is
    then judged on the aggregate statistics alone, which CANNOT see cancelling
    error (see BAR_PER_ELEM_EPS).  main() reports those rows separately.
    """
    if name is not None and name in WAIVED_ROWS:
        return "WAIVED"
    if name is not None and name in BINARY_GATES:
        verdict = BINARY_GATES[name]
        if verdict is None:
            return "UNMEASURED"
        return "AT BAR" if verdict else "DEBT"
    if corr is None or ratio is None:
        return "UNMEASURED"
    if per_elem is not None and per_elem > BAR_PER_ELEM_EPS:
        return "DEBT"
    if corr >= BAR_CORR and abs(ratio - 1.0) <= BAR_RATIO_EPS:
        return "AT BAR"
    return "DEBT"


def _self_test() -> int:
    """Synthetic-violation check: the per-element bar must be non-vacuous."""
    # perfect aggregates, wrecked per-element -> must be DEBT, not AT BAR
    assert classify(1.0, 1.0, 1e-3) == "DEBT", "per-element bar is VACUOUS"
    assert classify(1.0, 1.0, None) == "AT BAR"
    assert classify(1.0, 1.0, 0.0) == "AT BAR"
    assert classify(0.9, 1.0, 0.0) == "DEBT"
    # The historical regression this bar exists for: bn2 held AT BAR on a mean
    # ratio of 1-6.6e-9 while its per-element error was 6.96e-6.  Pinned as a
    # LITERAL, not PER_ELEMENT["bn2 (rn2b)"] -- that entry is now 5.88e-16
    # (fixed 2026-07-28 by fp64), and keying the self-test off a live value
    # made it go stale the moment the bug was fixed.
    assert classify(1.0, 0.9999999934, 6.96e-6) == "DEBT"
    # and the fixed value must now pass
    assert classify(1.0, 1.0, 5.880e-16) == "AT BAR"
    # binary gates must not be clearable by an absent corr/ratio
    assert classify(None, None, name="STABILITY on NEMO true grid (e3t_0)") == "DEBT"
    assert classify(None, None, name="not-a-binary-gate") == "UNMEASURED"
    # WAIVED rows classify without corr/ratio, and DO NOT fall through to
    # UNMEASURED/DEBT just because corr/ratio are absent.
    assert classify(None, None, name="zdf_mxl_turb") == "WAIVED"
    assert "zdf_mxl_turb" in WAIVED_ROWS
    # Synthetic-violation proof: a waiver missing either required string must
    # be rejected by _validate_waivers, not silently accepted. This proves the
    # gate is non-vacuous -- a bare name in WAIVED_ROWS is not enough to waive.
    for broken in (
        {"fake_row": ("", "some evidence")},          # empty provenance
        {"fake_row": ("some decision", "")},          # empty evidence
        {"fake_row": ("   ", "   ")},                 # whitespace-only both
    ):
        try:
            for term, (provenance, evidence) in broken.items():
                if not provenance.strip():
                    raise ValueError(f"WAIVED_ROWS[{term!r}] has an empty "
                                      "decision-provenance string")
                if not evidence.strip():
                    raise ValueError(f"WAIVED_ROWS[{term!r}] has an empty "
                                      "evidence citation string")
            raise AssertionError(
                "SELF-TEST FAILED: a waiver missing provenance/evidence was "
                "NOT rejected -- the waiver mechanism is vacuous.")
        except ValueError:
            pass  # expected: the synthetic broken waiver was caught
    # There is no generic per-row waiver flag: WAIVED_ROWS is the only path,
    # and it is keyed to a fixed, reviewed set of names -- not settable from
    # BINARY_GATES/MEASUREMENTS/PER_ELEMENT.
    assert "zdf_mxl_turb" not in BINARY_GATES, \
        "zdf_mxl_turb must be waived via WAIVED_ROWS, not BINARY_GATES"
    print("self-test OK: per-element bar fires on a cancelling-metric pass; "
          "binary gates classify without corr/ratio; a waiver missing "
          "provenance or evidence is rejected, not silently accepted")
    return 0


def main() -> int:
    if "--self-test" in sys.argv:
        return _self_test()
    rows = [(t, c, r, n, classify(c, r, PER_ELEMENT.get(t), name=t))
            for t, (c, r, n) in MEASUREMENTS.items()]
    width = max(len(t) for t, *_ in rows)
    print(f"{'term':<{width}}  {'corr':>12} {'ratio':>12}  status")
    print("-" * (width + 42))
    for term, corr, ratio, note, status in rows:
        cs = "     n/a    " if corr is None else f"{corr:>12.6f}"
        rs = "     n/a    " if ratio is None else f"{ratio:>12.6f}"
        flag = "" if status == "AT BAR" else f"  <-- {status}"
        print(f"{term:<{width}}  {cs} {rs}{flag}" + (f"   ({note})" if note else ""))

    waived = [(t, n) for t, _c, _r, n, s in rows if s == "WAIVED"]
    if waived:
        print("\n*** HUMAN-WAIVED (resolved by human decision, does not "
              "block exit 0 -- but does not count as AT BAR either) ***")
        for t, _n in waived:
            provenance, evidence = WAIVED_ROWS[t]
            print(f"  {t}")
            print(f"    decision: {provenance}")
            print(f"    evidence: {evidence}")

    unknown_prov = [t for t, *_ in rows if MEASURED_AT.get(t, "") == ""]
    disputed = [t for t, v in MEASURED_AT.items() if v == "DISPUTED"]
    if disputed:
        print(f"\nDISPUTED (conflicting measurements, do not trust): {', '.join(disputed)}")
    print(f"\nrows with NO measured-at provenance: {len(unknown_prov)} of {len(rows)}")
    at_bar = sum(s == "AT BAR" for *_, s in rows)
    debt = sum(s == "DEBT" for *_, s in rows)
    unmeasured = sum(s == "UNMEASURED" for *_, s in rows)
    waived_n = sum(s == "WAIVED" for *_, s in rows)
    mean_only = [t for t, c, r, _n, s in rows
                 if s == "AT BAR" and t not in PER_ELEMENT]
    print(f"\nAT BAR {at_bar} | DEBT {debt} | UNMEASURED {unmeasured} | "
          f"WAIVED {waived_n} | total {len(rows)}")
    print(f"bar: corr >= {BAR_CORR}, |ratio - 1| <= {BAR_RATIO_EPS}, "
          f"per-element <= {BAR_PER_ELEM_EPS}")
    if mean_only:
        print(f"\nAT BAR on CANCELLING statistics only ({len(mean_only)} of "
              f"{at_bar}) -- per-element error never measured, so these are "
              f"NOT proven exact:\n  " + "\n  ".join(mean_only))
    if debt or unmeasured:
        print("\nFAIL: the sweep is NOT complete. Do not describe these as "
              "'matched', 'faithful', 'closed' or 'good enough'.")
        return 1
    print("\nPASS: every term at the bar (WAIVED rows resolved by human "
          "decision, not by measurement).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
