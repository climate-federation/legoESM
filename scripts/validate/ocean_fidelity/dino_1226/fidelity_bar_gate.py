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
    "zdf_mxl (nmln)":                (0.99859,    None,       "12/9920 cols differ; REOPENED [e3t=both per probe_nmln_kanc.py]"),
    "ldf_slp wslpi":                 (0.997860193, 1.005370859,   "TIER-2 2026-07-28 (blocker-1 fix): the SAME live-gdept "
                                                              "capability as dyn_hpg/eos_rab/bn2 above was extended "
                                                              "here (compute_nemo_native_slopes' gdept/gdepw_top now "
                                                              "prefer z_coord.t_depth_ref like _nemo_wpoint_e3w_ "
                                                              "wmask_n2 already did, and zhmlpt/zdepu/zck/e3w/n2_int "
                                                              "all now get the live (1+r3t) stretch on an "
                                                              "OceanPartialCellCoordinate, matching ldfslp.F90:143, "
                                                              ":226-229, :289 exactly). MEASURED "
                                                              "(scripts/tmp/probe_ldfslp_live_gdept.py, ephemeral, "
                                                              "e3t=both, Y5 RUN_GDB kt=57601): the live gdept/e3w "
                                                              "THEMSELVES now reproduce NEMO's own dumped "
                                                              "eiv_dump_{gdept,e3w}.bin to corr=1.000000000/"
                                                              "ratio~1.0 (was e3w ratio 1.000057796 pre-fix -- "
                                                              "confirms the live-stretch construction is CORRECT), "
                                                              "but wslpi itself barely moves: corr 0.997860208-> "
                                                              "0.997860193, ratio 1.005420645->1.005370859. "
                                                              "REFUTES the accumulation/live-depth hypothesis as the "
                                                              "DOMINANT cause for this row: DINO's r3t=ssh/ht_0 is "
                                                              "tiny (ssh rms 0.54 m vs ht_0 O(1e3-4e3) m; jacobian "
                                                              "rms over wet cols = 0.999931, i.e. ~7e-5 from 1), so "
                                                              "the live stretch is real and now correct but "
                                                              "numerically negligible against this row's ~0.5-1% "
                                                              "residual. Absolute level differs slightly from the "
                                                              "prior probe_wslpi.py number (different harness/mask "
                                                              "convention); the BEFORE/AFTER delta on the SAME "
                                                              "harness is the trustworthy comparison here. Still "
                                                              "DEBT, cause NOT identified -- the #1226 blocker-1 "
                                                              "1-D-ladder API limitation is CLOSED (a live per-column "
                                                              "depth CAN now be expressed), but that was not this "
                                                              "row's dominant residual."),
    "ldf_slp wslpj":                 (0.998193168, 1.004061782,   "[e3t=both, blocker-1 fix applied -- see wslpi row; "
                                                              "before 0.998193273/1.004125163, negligible move]"),
    "ldf_slp uslp":                  (0.997795757, 1.006438108,   "[e3t=both, blocker-1 fix applied -- see wslpi row; "
                                                              "before 0.997795738/1.006434620, negligible move]"),
    "ldf_slp vslp":                  (0.998369979, 1.004132584,   "[e3t=both, blocker-1 fix applied -- see wslpi row; "
                                                              "before 0.998369962/1.004128764, negligible move]"),
    "ldf_eiv kappa (aeiu)":          (0.974309327, 1.026019288,   "TIER-2 2026-07-28 (blocker-1 fix applied, see "
                                                              "wslpi row): aeiu barely moves either (before "
                                                              "corr=0.974309121/ratio=1.026017892), consistent with "
                                                              "the wslpi/wslpj inputs it inherits from barely moving. "
                                                              "Prior Omega-fix history retained below. Row's absolute "
                                                              "level differs from the previously-recorded 1.0/1.000608 "
                                                              "(different measurement harness -- this row's own "
                                                              "probe_aeiu_v3.py no longer exists to reproduce exactly; "
                                                              "scripts/tmp/probe_ldfslp_live_gdept.py measures a 2-D "
                                                              "k=0 aeiu slice against eiv_dump_aeiu.bin directly). "
                                                              "STILL DEBT, inherits from ldf_slp DEBT above. Prior "
                                                              "note: a REAL constant bug found and FIXED, but it "
                                                              "accounts for only ~1/20 of the original 1.000608 row -- "
                                                              "read both halves. (a) FIXED: NEMO phycst.F90:89 sets "
                                                              "omega=2*rpi/rsiday (full double precision; DINO has no "
                                                              "key_cice, so the rounded literal branch does NOT run), "
                                                              "while legoESM's canonical constants.Omega is only a "
                                                              "4-significant-figure value -- 1.5875e-5 low. Omega "
                                                              "enters zRo=0.4*zn/|f| linearly and is then SQUARED into "
                                                              "zaeiw, so the coefficient carries 3.175e-5. Proven "
                                                              "decisively by feeding NEMO's OWN dumped e3w/rn2b/wslpi/"
                                                              "wslpj through the Treguier chain: zn/zah/zhw reproduce "
                                                              "NEMO bit-exactly (relerr 0), zRo carries exactly "
                                                              "1.578e-5 and zaeiw exactly 3.156e-5, and BOTH collapse "
                                                              "to <=9e-16 when NEMO's omega is substituted -- i.e. "
                                                              "given matched inputs the operator is machine-exact. "
                                                              "Fixed by threading an omega param (default "
                                                              "constants.Omega, zero behaviour change) through "
                                                              "compute_treguier_kappa_gm{,_nemo_native} + "
                                                              "gm_redi_tracer_tendency_latlon, and wiring "
                                                              "DINOConfig.omega/LatLonCGridOceanConfig.omega = "
                                                              "NEMO_CONSTANTS_CONFIG.Omega on the nemo_dino_kamm_mlf "
                                                              "card (also feeds create_mercator_grid so grid.f "
                                                              "matches). The global constants.Omega was deliberately "
                                                              "NOT changed (125-file blast radius). (b) NOT CLOSED: "
                                                              "the remainder is INHERITED from this row's upstream "
                                                              "inputs, which are themselves DEBT rows (ldf_slp wslpi/"
                                                              "wslpj/uslp/vslp above). This row cannot reach bar "
                                                              "until ldf_slp does."),
    "ldftra ahtu (Redi, nn_aht_ijk_t=20)": (1.0,  0.9999999722, "AT BAR: K_h_base*cos(lat_T) shares its row's latitude with NEMO's ahtu -> exact vs NEMO gphiu (ldftra_ahtv_compare.py)"),
    "ldftra ahtv (Redi, nn_aht_ijk_t=20)": (1.0,  0.9999999704, "AT BAR (fixed): prior note ('interp_cell_to_vface averages avg(cos) not cos(avg)') was WRONG -- _static_kappa_redi_override never called interp_cell_to_vface (Rule 0 violation, a probe artifact). Real cause: aht was the SAME T-point field reused unshifted for both zfu and zfv, while NEMO's ahtv is INDEPENDENTLY evaluated at the v-point (ldftra.F90:325-329 ldf_c2d('TRA',...), ldfc1d_c2d.F90:141-145: ahtv=zUfac*MAX(e1v,e2v)**inn). Fixed by adding a v-face-specific kappa_Redi_v (grid.cos_lat_v at the north-face-of-cell-j convention) threaded through nemo_iso_lap_tracer_tendency_latlon_cgrid/nemo_iso_w_kappa_sums/nemo_iso_a33/compute_isoneutral_K33_latlon; verified against the actual NEMO ldftra_dump_{ahtu,ahtv,gphiu,gphiv}.bin (RUN_1226_AHTU): pre-fix corr 0.9998618/ratio 1.0000380, post-fix corr 1.0000000/ratio 0.9999999704 -- matches the u-face's own bit-exact quality"),
    "eiv transport u":               (0.999617,   0.996234,   "REOPENED - eos_depth='geometric' was not threaded into gm_redi_density_and_jacobian (#1226); fixed, corr 0.998474->0.999617, ratio 0.994097->0.996234; residual = static t_depth_ref vs NEMO's live z-star gdept (~1e-8 density bias, deepest 1-2 levels only) DEBT [MEASURED AT e3t=both. The superseding "
                                                              "probe_eiv_transport_v2*.py did NOT set LEGOESM_NEMO_E3T, so the "
                                                              "'off' default gives corr 0.995918 / ratio 1.0231 -- this term IS "
                                                              "genuinely e3t-sensitive (unlike every other row audited). The "
                                                              "0.998474->0.999617 gain is CONFOUNDED between the eos_depth fix "
                                                              "and e3t=both; both are needed, neither isolated. Do not cite one cause]"),
    "eiv transport v":               (0.999103,   0.990637,   "REOPENED - same eos_depth fix, corr 0.995437->0.999103, ratio 0.982351->0.990637; residual concentrated at bottom-adjacent rows (jj~130-136,189-191) + deepest 2 levels (k=32,33), same static-vs-live-gdept cause as u; still 0.9% low DEBT [MEASURED AT e3t=both; "
                                                              "'off' default gives corr 0.994325 / ratio 1.0150. Same eos_depth-vs-e3t "
                                                              "confound as the u-component -- see that row]"),
    "traadv_fct fluxes":             (0.99994,    1.000100,   "[e3t=both per probe_fct.py/probe_fct_pure.py]"),
    "traadv_fct tendency (T)":       (0.994500,   None,       "after nonosc bound fix 2a73221ce [e3t=both]"),
    "traadv_fct horizontal tend":    (0.999950,   None,       "limiter itself now correct [e3t=both]"),
    "traadv_fct vertical upstream flux": (0.997700, None,     "0.91 was an OFFSET ARTIFACT; dry-cell mask fixed 01c1f226a (clips 603 vs NEMO 662) [e3t=both]"),
    "dyn_hpg":                       (1.0,        0.999986794,   "TIER-2 2026-07-28 (blocker-1 fix): the remaining "
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
                                                              "unchanged; harness (B) REFUTED]"),
    "dyn_vor EEN v":                 (0.999932,   1.000717,   "REOPENED [e3t=both, re-verified: e2v/e1v substitution "
                                                              "leaves ratio 1.000717->1.000717 unchanged; harness (B) "
                                                              "REFUTED]"),
    "dyn_adv KEG":                   (1.0,        1.000000,   "byte-exact"),
    "dyn_adv ZAD":                   (0.999200,   0.995000,   "after nemo_advective fix [e3t=both per probe_1226_keg_zad_split.py]"),
    "zdftke pdlr":                   (0.998120,   0.996800,   "REOPENED [re-verified 2026-07-27: probe_tke_prandtl.py's "
                                                              "own doc block omits LEGOESM_NEMO_E3T (default 'off'); "
                                                              "re-ran with e3t=both -> pdlr CORRECTED corr 0.99812 "
                                                              "(unchanged), ratio 0.996800->0.996780 (unchanged to "
                                                              "3 s.f.); harness (A) REFUTED for this term]"),
    "zdftke composite avt/avm":      (0.997560,   0.996197,   "257 cells; REOPENED [re-verified 2026-07-27: "
                                                              "probe_zdftke_avt_avm.py's own doc block omits "
                                                              "LEGOESM_NEMO_E3T (default 'off'); re-ran with e3t=both "
                                                              "-> corr 0.997578->0.997559, ratio 0.996198->0.996197 "
                                                              "(unchanged to 5 s.f.); harness (A) REFUTED]"),
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
    "STABILITY on NEMO true grid (e3t_0)": (None, None, "legoESM UNSTABLE on NEMO's actual geometry; bridge defaults to the wrong ladder to hide it"),
    # --- round 2 ---
    "dyn_spg_ts pssh":               (0.999989,   0.999900,   "[e3t=both, re-verified: compare_spg_barotropic_e3tboth.py "
                                                              "and its e2v-substituted twin both give ratio 0.9986 "
                                                              "unchanged; harness (A)+(B) REFUTED]"),
    "dyn_spg_ts puu_b":              (0.999586,   0.987100,   "1.3% gap UNEXPLAINED [e3t=both + e2u/e1u substitution "
                                                              "re-verified 2026-07-27: ratio 0.9871 unchanged in both "
                                                              "variants; harness (A)+(B) REFUTED, residual is real]"),
    "dyn_spg_ts un_adv":             (0.999777,   1.006700,   "[e3t=both, re-verified: e2v/e1v-substituted variant "
                                                              "gives the identical 0.9916/1.0003-class numbers; "
                                                              "harness (A)+(B) REFUTED]"),
    "ATF filter u":                  (0.999969,   0.995500,   "0.45% gap [e3t=both per atf_lego_extract_e3tboth.py -- "
                                                              "filtered-field ratio 0.9956/corr 0.999969 reproduced "
                                                              "exactly; harness (A) REFUTED]"),
    "ATF filter v":                  (0.999999,   1.000400,   "[e3t=both]"),
    "ATF filter T/S/ssh":            (1.0,        1.0,        "exact"),
    "dyn_ldf (dynldf_lev_lap) u":    (0.997900,   1.003900,   "[e3t=both per probe_1226_r2_item2_dynldf.py; same "
                                                              "operator family (gradient_x/y_cgrid) as dyn_hpg -- "
                                                              "(B) structurally a no-op here too, see module docstring]"),
    "dyn_ldf (dynldf_lev_lap) v":    (0.999400,   1.001800,   "[e3t=both]"),
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
                                                              "(not merely a candidate)."),
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
                                                              "by direct formula read, no run needed)."),
    "dom_qco_r3c r3u/r3v":           (0.999999999879, 0.999997, "hu_0/hv_0 added to nemo_io.NemoGrid (derived from e3u_0/e3v_0+mask); "
                                                              "face-averaged eta/H0 formula match; ratio not exactly 1 (same residual class as r3t) "
                                                              "[e3t=both per probe_1226_r2_item4_domqco.py; e2u/e1v read DIRECTLY from mesh_mask.nc "
                                                              "in this probe (not bridge-reconstructed) -- (B) already excluded for this term by construction]. "
                                                              "#1226 METRIC-CONVENTION RULED OUT (2026-07-28): r3u/r3v "
                                                              "are face-AVERAGES of r3t (itself a purely vertical "
                                                              "eta/H0 ratio, see r3t row) -- same structural argument, "
                                                              "no horizontal T/u-face metric term to move. NOT the "
                                                              "metric-convention mechanism."),
    "mlf_baro_corr":                 (None,       None,       "algebra only; needs _step_impl hook"),
    "lbc_lnk sign":                  (None,       None,       "NEVER VERIFIED"),
    "zdf_mxl_turb":                  (None,       None,       "MISSING TERM: no legoESM equivalent -- NEMO's Kz<avt_c "
                                                              "turbocline diagnostic (hmld) is distinct from zdf_mxl's "
                                                              "N^2-criterion MLD (nmln, which IS ported); hmld/mldkz5 "
                                                              "is diagnostic-only (never feeds dynamics), no lego port exists"),
    "zdf_drg_nonlin T-point rate":   (1.0,        1.0,        "AT BAR: exact, nemo_effective_bottom_drag_r on bridged bottom u/v [e3t=both]"),
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
                                                              "STILL DEBT, different cause needed."),
    "dyn_cor_2d (69x/step)":         (0.99999992, 0.999986,   "interior (excl. periodic-seam column, harness reindexing "
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
                                                              "itself is controlled and valid regardless)."),
    "traadv_fct (SALINITY)":         (0.203165,   3.167599,   "DEBT: corr/ratio far below bar. VERIFIED (not the "
                                                              "'boundary-column artifact' hypothesis -- REFUTED: "
                                                              "excluding i/j=0,last leaves corr 0.2024 (unchanged); "
                                                              "excluding bathymetry-step-adjacent columns too only "
                                                              "reaches corr 0.379/ratio 1.37, still far below bar). "
                                                              "Method: measured via the SAME flux-reconstruction-from-"
                                                              "dumps technique as T (immune to the tra_sbc/tra_qsr "
                                                              "Krhs contamination); raw contaminated dump gives an "
                                                              "even worse corr=0.052, so contamination was not "
                                                              "hiding a good match either. ROOT CAUSE: per-face "
                                                              "upstream+antidiffusive S fluxes match NEMO at "
                                                              "corr 0.94-0.995 / abs_ratio 0.9999-1.010 (metric-"
                                                              "scaled: lego's mass_flux is pre-metric [m^2/s], "
                                                              "NEMO's ztFu/v/w are full volume-flux "
                                                              "[m^3/s] per traadv.F90:321-333 -- scaling lego's flux "
                                                              "by grid.dy*0.5 (zonal)/v-face metric (merid)/grid.area "
                                                              "(vert) before comparing is mandatory, a raw pre-metric "
                                                              "vs post-metric flux comparison gives a spurious "
                                                              "~1e-5..1e-10 ratio that is NOT a physics defect). The "
                                                              "horizontal-only and vertical-only upstream tendencies "
                                                              "ALSO independently match NEMO at corr 0.9999/ratio "
                                                              "~1.000 each. But S's net upstream tendency is a "
                                                              "near-total CANCELLATION of those two terms -- "
                                                              "corr(horiz,vert) = -0.9999, net/gross = 1.2% (T's is "
                                                              "-0.995 / 10.2% net/gross) -- so S's advective tendency "
                                                              "is ~8x more exposed to the SAME absolute-scale "
                                                              "horiz/vert discretization mismatch that is invisible "
                                                              "for T. This is a genuine interior defect (survives "
                                                              "excluding boundary+topo-step columns), NOT a masked-"
                                                              "cell/edge artifact, though its proximate cause is "
                                                              "catastrophic-cancellation amplification of an "
                                                              "existing (already-documented in advection.py's nonosc "
                                                              "box-tightening comment) small horiz/vert discretization "
                                                              "gap, not a new independent S-specific bug. S feeds the "
                                                              "EOS -> ACC: flagged, not yet fixed. [e3t=both per "
                                                              "probe_fct_sal.py -- harness-contamination audit REFUTES "
                                                              "both (A) and (B) for this term, corr=0.20/ratio=3.17 is real]"),
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
    "ldf_slp wslpi": "9f25d7be4",
    "ldf_slp wslpj": "9f25d7be4",
    "ldf_slp uslp": "9f25d7be4",
    "ldf_slp vslp": "9f25d7be4",
    "ldf_eiv kappa (aeiu)": "9f25d7be4",
    "dyn_hpg": "DISPUTED",   # two agents measured corr 0.19 vs 1.0 -- harness gap
}


def classify(corr: float | None, ratio: float | None) -> str:
    if corr is None or ratio is None:
        return "UNMEASURED"
    if corr >= BAR_CORR and abs(ratio - 1.0) <= BAR_RATIO_EPS:
        return "AT BAR"
    return "DEBT"


def main() -> int:
    rows = [(t, c, r, n, classify(c, r)) for t, (c, r, n) in MEASUREMENTS.items()]
    width = max(len(t) for t, *_ in rows)
    print(f"{'term':<{width}}  {'corr':>12} {'ratio':>12}  status")
    print("-" * (width + 42))
    for term, corr, ratio, note, status in rows:
        cs = "     n/a    " if corr is None else f"{corr:>12.6f}"
        rs = "     n/a    " if ratio is None else f"{ratio:>12.6f}"
        flag = "" if status == "AT BAR" else f"  <-- {status}"
        print(f"{term:<{width}}  {cs} {rs}{flag}" + (f"   ({note})" if note else ""))

    unknown_prov = [t for t, *_ in rows if MEASURED_AT.get(t, "") == ""]
    disputed = [t for t, v in MEASURED_AT.items() if v == "DISPUTED"]
    if disputed:
        print(f"\nDISPUTED (conflicting measurements, do not trust): {', '.join(disputed)}")
    print(f"rows with NO measured-at provenance: {len(unknown_prov)} of {len(rows)}")
    at_bar = sum(s == "AT BAR" for *_, s in rows)
    debt = sum(s == "DEBT" for *_, s in rows)
    unmeasured = sum(s == "UNMEASURED" for *_, s in rows)
    print(f"\nAT BAR {at_bar} | DEBT {debt} | UNMEASURED {unmeasured} | total {len(rows)}")
    print(f"bar: corr >= {BAR_CORR}, |ratio - 1| <= {BAR_RATIO_EPS}")
    if debt or unmeasured:
        print("\nFAIL: the sweep is NOT complete. Do not describe these as "
              "'matched', 'faithful', 'closed' or 'good enough'.")
        return 1
    print("\nPASS: every term at the bar.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
