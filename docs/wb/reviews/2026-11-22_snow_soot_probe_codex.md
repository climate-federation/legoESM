A. Instrument defects — HOLD pending correction

- **HIGH — History albedo has mismatched averaging support.** `probe_snow_albedo_snicar.py:94–107` divides snow-conditioned SNOFSR means by unconditional FSDS means. CESM `SurfaceRadiationMod.F90:951–970` sets SNOFSR to `spval` whenever `snl>=0`; FSNO≥0.9 does not guarantee continuous snow-layer existence. **Fix:** use incident fluxes with identical snow conditioning and spatial support, or explicitly label this ratio an unvalidated proxy. Tightening FSNO alone cannot establish validity.

- **HIGH — The prescribed temperatures differ between models.** `clm50_snicar_oracle.F90:301–313` treats schedule temperature as the snow-surface intercept, adding `gradient × depth` to each layer center. `ours_daily():196–200` uses the unadjusted schedule temperature. Moreover, CESM `SnowSnicarMod.F90:1116–1131` uses the top-layer center as its upper boundary, making the diagnosed top-layer gradient half the prescribed linear gradient. **Fix:** define surface and layer-center temperatures explicitly, construct the intended CLM diagnostic gradient, and report the actual temperature supplied to each aging scheme.

- **HIGH — Deposition fractions are not snowpack composition.** `probe_snow_albedo_snicar.py:119–134,217–224` imposes one MAM deposition split on both snow layers. CESM `SnowHydrologyMod.F90:72–79` scavenges hydrophilic/hydrophobic BC and dust bins differently, so retained fractions need not equal deposition fractions. The split also includes ocean and snow-free locations. **Fix:** prefer snowpack species masses; otherwise identify the split as assumed, restrict deposition weighting to the relevant land domain, and report composition sensitivities.

- **HIGH — The result cannot establish the stated regional production claim.** One synthetic, snowfall-free column with pooled loading does not measure regional spring albedo. The scalar aging law and its production overrides are correctly reproduced, but production also applies snow-cover blending, background-albedo floors, and potentially canopy masking (`multilayer_land.py:984–997`; `surface_albedo.py:374–381`). **Fix:** describe the verdict as conditional on this synthetic snow-only experiment; establish regional production bias using matched production states and forcing.

- **MEDIUM — Invalid inputs and NaNs can evade the fatal checks.** The history mask checks only reflected and incident component fluxes; other required fields can propagate NaNs. Neither positive layer masses nor nonnegative concentrations are enforced. NaN comparisons also pass both self-checks (`probe_snow_albedo_snicar.py:246–252`). **Fix:** validate every required input, positive layer masses, concentration bounds, complete output shape/timestamps, finite outputs, and albedo bounds before scoring; demonstrate rejection with planted violations.

- **MEDIUM — Requested zenith sensitivities are discarded.** The oracle computes cosz 0.3 and 0.6, but Python retains only 0.45 (`probe_snow_albedo_snicar.py:234–241`). **Fix:** preserve and report broadband scores at all three angles, keeping 0.45 as the registered primary.

- **MEDIUM — Provenance does not identify the executing instrument.** Both probe sources and the registration are currently untracked, while the artifact records only HEAD and external paths (`probe_snow_albedo_snicar.py:263–270`). **Fix:** commit the registration and instrument before measurement; stamp source/input hashes, compiler/version, and build flags.

- **LOW — Monthly means receive equal temporal weight.** `hist_stats():97–115` weights cell-months by area but omits interval duration. **Fix:** include `time_bounds` durations for a MAM time mean, or explicitly identify the statistic as an equal-month mean.

Source checks found no argument-layout, active-layer indexing, species-order, or stub-value defect for the prescribed two-layer case. Zero OC matches `SurfaceAlbedoMod.F90:469–474`; diffuse coszen=0.5 supplies the required positive daylight gate while direct source terms are zero. The days 61–105 slice and CONFIRMS/REFUTES inequalities match the registration exactly.

B. Detail-choice votes

1: KEEP 1979–1983 — reasonable five-year baseline; weight monthly means by duration.  
2: KEEP FSNO≥0.9 and landfrac≥0.5 — useful snow-dominant sampling, conditional on correcting the history averaging mismatch.  
3: ALT report all-phi/all-pho bracket alongside the deposition estimate — deposition does not determine retained snow composition; flag dust-bin uncertainty separately.  
4: KEEP two-layer mass-conserving construction — preserves surface enrichment; label pooled-column results as synthetic.  
5: KEEP 250 kg/m³ — acceptable prescribed density; it controls aging as well as thickness.  
6: ALT explicitly matched surface/layer-center temperatures and diagnosed gradients — correct the forcing mismatch; retain the schedule as a synthetic experiment.  
7: KEEP days 61–105 and cosz 0.45 — matches registration; actually report 0.3 and 0.6 sensitivities.  
8: KEEP common history SW weights — supports a controlled spectral comparison; the flat legoESM snow term is unchanged by normalized weights.  
9: KEEP Δalbedo×mean FSDS — label it an equivalent snow-surface SW difference, not diagnosed regional or atmospheric forcing.  
10: ALT retain both checks and add finite/range/input checks plus planted failures — current comparisons allow NaNs and do not validate composition or layout.

UNVERIFIED: Compilation, oracle execution, and numerical sensitivity magnitudes; this was a source review.
