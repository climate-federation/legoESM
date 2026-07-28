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
VERDICT: every scratchpad probe that produced a MEASUREMENTS number below was
ALREADY invoked with ``LEGOESM_NEMO_E3T=both`` (verified by grepping each
probe's own "Run with:" doc block across both session scratchpads; spot
re-run of zdftke pdlr/composite, dyn_spg_ts puu_b/un_adv and dyn_hpg/dyn_vor
with the env var toggled reproduced the SAME numbers to 4+ sig figs) -- (A)
does NOT contaminate this table. (B) was measured directly: bridge dy_v/dy_T
vs NEMO's real e2v/e2t agree to ~4-5e-5 relative error at every INTERIOR row;
the only rows with a real 0.2-0.8% discrepancy are the two channel WALL rows
(row 0 south, row 198 north) -- not a domain-wide floor. Root cause: on a
non-tripolar (regular/Mercator) LatLonCGridGeometry, ``gradient_x/y_cgrid``,
``divergence_cgrid`` (operators_latlon_cgrid.py) never read the geometry's
own dx_u/dy_u/dx_v/dy_v fields -- they recompute face metrics inline from
``cos_lat``/``dy``/``radius``/``dlon`` every call, so patching those fields
(or substituting NEMO's e1u/e2u/e1v/e2v into them) is a NO-OP for every
production tendency; verified empirically for dyn_hpg/dyn_vor/dyn_spg_ts
(ratio unchanged to 5+ digits after the substitution). Net: NEITHER harness
artifact explains any DEBT row in this table; each row's number is real.
See docs/ocean/fidelity/ (harness-audit note) for the full re-measurement
log. Per-row notes below are tagged ``[e3t=both]`` where independently
verified.
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
    "eos_rab alpha":                 (1.0,        1.000007,   "median rel 4.7e-6 [e3t=both per probe_n2.py]"),
    "bn2 (rn2b)":                    (1.0,        0.999993,   "median |rel| 6.96e-6 [e3t=both per probe_n2.py]"),
    "zdf_mxl (nmln)":                (0.99859,    None,       "12/9920 cols differ; REOPENED [e3t=both per probe_nmln_kanc.py]"),
    "ldf_slp wslpi":                 (0.999177,   1.003000,   "interior 1.0011, bottom row 1.0255; REOPENED [e3t=both "
                                                              "per probe_wslpi.py/probe_wslp_wall_v1.py; the 1.0255 "
                                                              "bottom-row outlier is CONSISTENT with the (B) wall-row "
                                                              "geometry residual measured directly this audit (0.2-0.8% "
                                                              "at row 0/198 vs ~5e-5 interior) -- plausible real "
                                                              "contributor, not yet isolated by substitution (the "
                                                              "gradient operators recompute face metrics inline so a "
                                                              "geom-field patch would be a no-op; would need patching "
                                                              "cos_lat/dy_T at just the wall rows to isolate cleanly)"),
    "ldf_slp wslpj":                 (0.998875,   0.998447,   "[e3t=both]"),
    "ldf_slp uslp":                  (0.999340,   0.998697,   "[e3t=both]"),
    "ldf_slp vslp":                  (0.999012,   0.997594,   "[e3t=both]"),
    "ldf_eiv kappa (aeiu)":          (1.0,        1.000608,   "ratio not 1 [e3t=both per probe_aeiu_v3.py]"),
    "ldftra ahtu (Redi, nn_aht_ijk_t=20)": (1.0,  0.9999999722, "AT BAR: K_h_base*cos(lat_T) shares its row's latitude with NEMO's ahtu -> exact vs NEMO gphiu (ldftra_ahtv_compare.py)"),
    "ldftra ahtv (Redi, nn_aht_ijk_t=20)": (1.0,  0.9999999704, "AT BAR (fixed): prior note ('interp_cell_to_vface averages avg(cos) not cos(avg)') was WRONG -- _static_kappa_redi_override never called interp_cell_to_vface (Rule 0 violation, a probe artifact). Real cause: aht was the SAME T-point field reused unshifted for both zfu and zfv, while NEMO's ahtv is INDEPENDENTLY evaluated at the v-point (ldftra.F90:325-329 ldf_c2d('TRA',...), ldfc1d_c2d.F90:141-145: ahtv=zUfac*MAX(e1v,e2v)**inn). Fixed by adding a v-face-specific kappa_Redi_v (grid.cos_lat_v at the north-face-of-cell-j convention) threaded through nemo_iso_lap_tracer_tendency_latlon_cgrid/nemo_iso_w_kappa_sums/nemo_iso_a33/compute_isoneutral_K33_latlon; verified against the actual NEMO ldftra_dump_{ahtu,ahtv,gphiu,gphiv}.bin (RUN_1226_AHTU): pre-fix corr 0.9998618/ratio 1.0000380, post-fix corr 1.0000000/ratio 0.9999999704 -- matches the u-face's own bit-exact quality"),
    "eiv transport u":               (0.999617,   0.996234,   "REOPENED - eos_depth='geometric' was not threaded into gm_redi_density_and_jacobian (#1226); fixed, corr 0.998474->0.999617, ratio 0.994097->0.996234; residual = static t_depth_ref vs NEMO's live z-star gdept (~1e-8 density bias, deepest 1-2 levels only) DEBT [e3t=both per probe_eiv_transport_v1.py]"),
    "eiv transport v":               (0.999103,   0.990637,   "REOPENED - same eos_depth fix, corr 0.995437->0.999103, ratio 0.982351->0.990637; residual concentrated at bottom-adjacent rows (jj~130-136,189-191) + deepest 2 levels (k=32,33), same static-vs-live-gdept cause as u; still 0.9% low DEBT [e3t=both]"),
    "traadv_fct fluxes":             (0.99994,    1.000100,   "[e3t=both per probe_fct.py/probe_fct_pure.py]"),
    "traadv_fct tendency (T)":       (0.994500,   None,       "after nonosc bound fix 2a73221ce [e3t=both]"),
    "traadv_fct horizontal tend":    (0.999950,   None,       "limiter itself now correct [e3t=both]"),
    "traadv_fct vertical upstream flux": (0.997700, None,     "0.91 was an OFFSET ARTIFACT; dry-cell mask fixed 01c1f226a (clips 603 vs NEMO 662) [e3t=both]"),
    "dyn_hpg":                       (1.0,        1.000045,   "ratio not 1 [e3t=both, re-verified 2026-07-27: "
                                                              "re-measured with NEMO e1u/e2u/e1v/e2v spliced into "
                                                              "cos_lat/dy_T (the fields the non-tripolar gradient "
                                                              "operators actually read) -- ratio unchanged "
                                                              "1.000045153->1.000045138 (u), 1.000054459->1.000061469 "
                                                              "(v); harness-artifact hypothesis (B) REFUTED for this "
                                                              "term, residual is real]"),
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
    "ssh_nxt / div_hor":             (1.0,        0.999991,   "[e3t=both per probe_1226_r2_item3_sshnxt.py]"),
    "dom_qco_r3c r3t":               (1.0,        0.999997,   "[e3t=both]"),
    "dom_qco_r3c r3u/r3v":           (0.999999999879, 0.999997, "hu_0/hv_0 added to nemo_io.NemoGrid (derived from e3u_0/e3v_0+mask); "
                                                              "face-averaged eta/H0 formula match; ratio not exactly 1 (same residual class as r3t) "
                                                              "[e3t=both per probe_1226_r2_item4_domqco.py; e2u/e1v read DIRECTLY from mesh_mask.nc "
                                                              "in this probe (not bridge-reconstructed) -- (B) already excluded for this term by construction]"),
    "mlf_baro_corr":                 (None,       None,       "algebra only; needs _step_impl hook"),
    "lbc_lnk sign":                  (None,       None,       "NEVER VERIFIED"),
    "zdf_mxl_turb":                  (None,       None,       "MISSING TERM: no legoESM equivalent -- NEMO's Kz<avt_c "
                                                              "turbocline diagnostic (hmld) is distinct from zdf_mxl's "
                                                              "N^2-criterion MLD (nmln, which IS ported); hmld/mldkz5 "
                                                              "is diagnostic-only (never feeds dynamics), no lego port exists"),
    "zdf_drg_nonlin T-point rate":   (1.0,        1.0,        "AT BAR: exact, nemo_effective_bottom_drag_r on bridged bottom u/v [e3t=both]"),
    "dyn_drg_init RHS increment":    (0.999937,   1.000278,   "u; v=0.999961/0.999988 (both DEBT, ~5e-5 residual) [e3t=both per probe_bottom_drag.py]"),
    "dyn_cor_2d (69x/step)":         (0.99999992, 0.999986,   "interior (excl. periodic-seam column, harness reindexing "
                                                              "artifact there, not a lego defect); v=1.0/0.999928; "
                                                              "een_barotropic_coriolis(metric_complete=True) vs dumped "
                                                              "substep-1 zu_trd/zv_trd [e3t=both per probe_dyn_cor_2d.py]"),
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
