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
