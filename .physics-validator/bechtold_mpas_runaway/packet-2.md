# Adversarial review packet — round 2 (responses + your two new candidates verified)

You are the same independent adversarial physics reviewer. Round 1 is in
`review-1.md`. Below I (a) concede your corrections, (b) report that I
independently VERIFIED both of your new candidates in the code, (c) give a
revised interaction hypothesis, (d) adopt your factorial test. Attack what
remains; tell me if we have converged or what finding is still open.

## I concede (your round-1 corrections were right)
- **OLR mechanism backwards.** My "drier air collapses LW cooling → runaway" is
  wrong for clear sky: less water vapour → MORE OLR → MORE cooling, which
  OPPOSES a runaway. A radiative-sink FAILURE therefore needs a real forcing
  (clouds/aerosol), not drying. Retracted.
- **368 W/m² not secure.** It was derived from a naive model-level-mean T; I do
  not know the reported mean is area/pressure-mass-weighted, so the implied
  column energy input is unquantified. Retracted as a number.
- **F6 height mismatch.** My storm-track probe peaked at L26/902 hPa (lower
  trop), NOT the mid-trop L13 bulge; the probe does NOT support a
  detrainment-height attribution. A saturated M_u is vigorous transport, not
  energy creation. F6 downgraded to "amplifier/shaper, not a source".
- **F7 is a FALSE POSITIVE (you were right).** Verified `bechtold.py:2491`
  `plume = plume._replace(q_c_u=L_converted)` — the anvil detrainment at
  `:2782` uses the POST-conversion residual, not the unconverted condensate. No
  double book.
- **F10 over-claimed.** The "zero interceptions" is from a sparse-cadence log
  (`model_driver.py:6546-6553`), not a cumulative counter; it does not prove no
  interception. Needs instrumentation.
- **F8 magnitude.** Agreed — code alone can't bound Hines at "<368 W/m²";
  diagnose eps_gwd.
- **F1** — three test functions blocked (not two); and a collection error makes
  CI RED (loud), unless the file/job is excluded — either way the guard gives no
  protection; it is not "silently" disabled.

## Your two new candidates — I VERIFIED BOTH in the code

### A [CONFIRMED] — MPAS never sets bechtold_dx_m → ZTAURES=1 → coarse-grid over-vigor
- `bechtold_dx_m: float = 0.0` (`config.py:1059`); the resolver only COPIES it
  (`physics_pipeline.py:2985`), argparse default 0.0 (`run_amip.py:870`).
- I grepped every `dx_m`/cell-area site: the only dx computations
  (`component_factory.py:160-233`, `model_driver.py:9984`,
  `estimate_min_dx_cubed_sphere`) feed CUBED-SPHERE diffusion coefficients.
  **Nothing sets `bechtold_dx_m` from the MPAS SCVT cell area.** The config
  field comment claiming "the driver sets it from the grid" is FALSE.
- `_ifs_ztaures` (`bechtold.py:414-430`): `if dx_m <= 0.0: return 1.0`. So on
  ~2.2° (~250 km) cells the physical ZTAURES≈3 (3× longer turnover, 3× weaker
  M_b) is REPLACED by 1.0 → the deep CAPE closure runs ~3× too vigorous. This is
  the Bechtold-SPECIFIC coarse-grid defect. It AMPLIFIES the (conservative,
  R=0) convective pump; it is not itself a net energy source.

### B [CONFIRMED code defect; activation-conditional] — volcanic LW mis-applied
- The RRTMGP API docstring itself (`rrtmgp.py:599-605`) states the LW slot needs
  **per-layer ABSORPTION** optical depth, "NOT extinction … a caller holding
  extinction OD must scale by the absorption fraction 1−ω first."
- The forcing path loads `ext_earth` = **extinction** and collapses it to a
  single COLUMN AOD (`external.py:562`, band-mean); NO 1−ω scaling anywhere.
- `distribute_column_aod_to_layers` (`surface_utils.py`) spreads that column AOD
  by PRESSURE-MASS: `w = dp/Σdp; aod_layers = aod_col*w`. So ~90% of a
  physically STRATOSPHERIC volcanic AOD is deposited in the TROPOSPHERE
  (`model_driver.py:2415-2420` builds `_aerosol_lw_od` this way).
- Net: extinction-as-absorption (over-count) + stratosphere→troposphere
  misplacement ⇒ spurious TROPOSPHERIC LW greenhouse = a DIRECT column energy
  source (reduced OLR), exactly the radiative-sink failure the runaway needs.
- CAVEAT (magnitude): gated by `volcanic_aerosol_lw` (default False,
  `config.py:502`) AND a volcanic file present (`model_driver.py:2238`). I
  CANNOT confirm probe_bech40 enabled it (its experiment_config.json isn't in
  scope). And physical volcanic LW AOD is small except El Chichón(1982)/
  Pinatubo(1991); absent a units error the tropospheric forcing is O(1-5 W/m²),
  not obviously runaway-scale. **This is a real defect to fix regardless; its
  contribution to THIS runaway needs the flag status + a diagnosed AOD.**

## Revised hypothesis (PLAUSIBLE — the factorial decides)
Convection is conservative (probe R=0 across soundings/resolutions), so it is
NOT the net source. The coherent runaway is an **A×B interaction**:
- **A** (dx_m=0) makes Bechtold over-vigorous → over-draws surface LHF from the
  fixed SST and pumps it up (conservatively) into the mid/upper troposphere.
- **B** (volcanic-LW mis-distribution, IF active) traps upwelling LW in the
  troposphere → the OLR sink FAILS (supplying the failure that drying alone,
  per your correction, does NOT).
- Together: surface-fed heat can't be radiated away → warming; the mid-trop
  bulge = convective detrainment heating co-located with the spurious LW
  absorption. This explains why base (SBM + pinned deck) is healthy and why a
  SINGLE-delta arm may be insufficient (interaction, my F9).
Alternative if B was OFF: A over-vigor + the transient SW-aerosol/ozone deck or
a surface-flux term supplies the imbalance; ranking then needs the ledger.

## Adopted decisive test (your design)
An instrumented **2×2 factorial** SBM/Bechtold × pinned/transient-deck, all from
ONE identical pre-divergence checkpoint, logging per step the area-weighted
per-process energy ledger Q_i = ⟨∫(c_p·Ṫ_i + L_v·q̇_v,i) dp/g⟩ for convection,
turbulence, radiation (split SW/LW), GWD, microphysics, hard-sat drain; plus
TOA/surface net fluxes, column water, CAPE, and the fraction of columns at
M_u/M_b_max. Expected: Q_conv≈0 in all Bechtold arms (positive system energy
must show up in radiation/turbulence/GWD); a Bechtold main effect ⇒
convection-triggered; a deck main effect ⇒ forcing (isolate volcanic-LW by also
diagnosing `_aerosol_lw_od`·heating); a non-additive term ⇒ the A×B interaction.
Repeat the Bechtold-pinned cell with physical `dx_m≈√(cell area)` (or lower
M_b_max): if the bulge/plateau vanishes, that specifically convicts A.

## Ask
1. With A and B both verified as real defects, do you have any REMAINING finding,
   or have we converged on: (i) A = confirmed Bechtold coarse-grid wiring bug;
   (ii) B = confirmed volcanic-LW radiation bug (activation-conditional);
   (iii) convection itself conservative (not the source); (iv) the 2×2 factorial
   + `_aerosol_lw_od` diagnosis as the arbiter?
2. Is my A×B interaction framing sound, or is there a simpler single dominant
   cause you would bet on given the anatomy (mid-trop bulge, dry hot storm-track
   columns, |u| spike then fall)?
3. Any other confound in the factorial, or any code path I still have not
   checked that could be the direct energy source (surface layer, microphysics,
   the MPAS dycore's own energy budget)?
