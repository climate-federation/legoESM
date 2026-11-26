## Audit result

**Meta-caveat first:** I do not have access to the Levante/Derecho filesystem or this repository in this session. I can therefore not confirm any `file:line` citation (rrtmgp.py:504-508, radiation/config.py:258, surface_utils.py:489-491) or read the ICON/CMIP6 netCDF headers. What I can do is check each claim for internal consistency, physics, and numerics, and flag where a REFUTE would be warranted *if* the code reads as you describe. Treat this as an adversarial consistency review, not a code verification.

**C1 GHG — AGREE (internal consistency).** Global-annual-mean GHGs with linear interp is standard; the halogen bookkeeping point is correct and its ~0.02 W/m² magnitude is right order (minor halocarbon forcing in 1979 is small, and lumping CFC-11eq vs CFC-11 + fixed minors differs by well under 0.05 W/m²). One nit: "measured, unweighted lat mean" of LBC file values is fine for a globally uniform application. Not a cold-bias candidate — AGREE with your verdict. UNSURE only on the actual file contents (as you yourself flag).

**C2 Solar — AGREE, with one real gap.** Same SOLARIS-HEPPA 3.2 lineage ⇒ TSI equivalent. But your "RRTMG→RRTMGP one-band rotation" of the 14-band SSI deserves scrutiny: collapsing 14 bands can shift stratospheric SW heating, though effect on surface tas is <0.05 W/m² — your number is plausible. Orbit: if yours is present-day Berger and CAM6 is also fixed near-present, EQUIVALENT holds; if CAM6 uses `orb_iyear` = first sim year and yours uses a different epoch, the difference is still negligible for 1979 amip. AGREE.

**C3 Ozone — AGREE.** CCMI-1-0 vmro3 vs WACCM-derived prescribed ozone: genuinely different products, different temporal resolution (monthly vs 5-day), zonal-mean both. O(0.1) W/m² TOA and stratospheric T differences are the right order; correct that this is not the primary cold driver. The 16-ghost-levels extension is standard practice (as in CESM's `ozone_file` handling).

**C4 Volcanic — AGREE, and this is your best catch.** If SW extinction uses global gray SSA 0.93 / g 0.70 while the file carries ω_sun ≈ 0.99 for sulfate, then ~5–7% of volcanic SW extinction is spuriously absorbed in the stratosphere — your arithmetic is right. Correct that 1979 (quiescent, AOD ~0.005) is unaffected, so it cannot explain the 1979 bias; correct that El Chichón (1982) / Pinatubo (1991) years will show too little surface SW cooling + too much strat heating. This matters for trend/timing comparisons post-1982 — I would flag it as a **known issue independent of the cold bias**. Verify the `--volcanic-aerosol-lw` flag actually reads the LW profile rather than reusing the SW one (common failure mode — check whether ext_earth omega_earth is applied or SSA 0.93 reused).

**C5 Tropospheric aerosol — AGREE.** MACv2-SP over MACv1 background vs interactive MAM4: your directional argument is right — gray SSA 0.93 is more absorbing than realistic sulfate/OM mixtures, and a column-uniform (pressure-thickness) profile puts aerosol aloft vs MAM4's PBL-concentrated profile; both push toward warming/positive SW forcing relative to CAM6, so it cannot *directly* cause the NH cold/albedo bias. The CCN path (Andreae 2009 on total AOT) with tuned rsut is the plausible high-latitude pathway — CDNC over the Arctic/NH land affects low cloud phase/optics; but that is a model-response question, not a forcing-input difference you can sign a priori. Keep it PLAUSIBLE/secondary.

**C6 SST/SIC — AGREE.** PCMDI-AMIP-1-1-10 vs Hurrell HadOISST merge is version drift only; both linear-interp mid-month BC values. The T_freeze floor and [0,1] clip are cosmetic. Equivalent for bias purposes.

**C7 Land use — AGREE.** simyr2000 fixed PFTs vs CLM5 transient: global forcing <0.1 W/m², and for a 1979-onward AMIP run the crop expansion history is small in 1979. Not the driver.

**C8 Snow impurities — AGREE, and this is the strongest forcing-side candidate you have.** If SNICAR-adjacent code is absent (no black-carbon/dust-on-snow term), your snow is brighter than CLM5's everywhere, maximally in spring melt over Eurasia/Tibet/boreal NA — exactly the pattern of +0.10–0.15 TOA albedo / cold NH land. Literature magnitudes (+0.05 W/m² global, O(1–5) W/m² regional spring) are correct. Note honestly: this is a *model-physics* gap (missing process), not a "forcing file" mismatch, and BATS-vs-SNICAR grain aging adds in the same direction. Keep as leading forcing-side candidate, but also verify it against the surface-albedo parameterization itself (snow-covered-Fraction / albedo thresholds) before attributing.

**C9 — AGREE.** No N cycle ⇒ N-dep N/A.

## Forcing channels you MISSED

1. **Aerosol LW / dust longwave** — you say "no LW" for tropospheric aerosol (fine for MACv2-SP sulfate-dominated), but if any dust is present, dust LW is non-negligible over Sahara/Tibet. Probably absent in your scheme; worth stating explicitly.
2. **Stratospheric water vapor (CH₄-oxidation trend)** — CAM6/WACCM carries it; a fixed stratospheric H₂O profile misses a small (≈0.05 W/m²) but monotone greenhouse trend from 1979→present. Not a 1979 bias term.
3. **Surface datasets beyond land use:** vegetation phenology/initial conditions, soil texture/color, glacier/wetland extent, and **land-ocean mask / topography differences** between your grid and CAM6's — these are forcing-side inputs and a classic source of Tibet/Antarctica cold biases (elevation/smoothing). Recommend auditing `surfdata` topography before blaming any radiative channel.
4. **Sea-ice/snow-on-ice albedo treatment at the ice surface** — the SIC file is equivalent, but the *surface albedo of the ice/snow over ocean* at high latitudes is a forcing/boundary-condition channel directly relevant to your Arctic TOA albedo bias; C6 equivalence says nothing about it.
5. **Prescribed vs interactive stratospheric aerosol precursor climatology / background 1979 AOD** — you treat volcanic as episodic; check whether CAM6's WACCM file includes a non-zero background that your ext_sun(1979)≈0 misses. Likely small.
6. **Emissions-adjacent: DMS / ocean biogeochemistry** feeding MAM4 sulfate CCN — part of the indirect-effect difference in C5, but worth listing since it's Arctic-CDNC-relevant.

## Bottom line

- **No REFUTE** is possible from where I sit (no code access) — and none of the claims is internally inconsistent enough to reject on physics.
- AGREE with the overall verdict: GHG/solar/SST equivalent; volcanic (C4) and aerosol (C5) optics differ but in the wrong direction or wrong years to explain a 1979 cold bias; **C8 (missing snow impurities) plus the un-audited channels 3 and 4 (topography/land-surface data, snow-on-sea-ice albedo) are the credible forcing-side candidates for the NH-high-latitude cold/albedo bias.**
- Before publishing this audit, do the three reads you flagged: (i) confirm the cited `file:line`s, (ii) confirm the LW volcanic profile uses ω_earth not the gray SSA, (iii) grep the surfdata/topography comparison — I'd bet at least part of the Tibet/Antarctica cold bias is an elevation/mask issue, not radiation at all.
