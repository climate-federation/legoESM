# Forcing audit: legoESM production AMIP vs CAM6 (CESM2.1 cam_cesm2_1_rel_60 hist_cam6) — claims for adversarial review

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
