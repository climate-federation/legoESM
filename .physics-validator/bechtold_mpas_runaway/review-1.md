## Verdict

Your leaf probe convincingly refutes a *finite, leaf-local* Bechtold “heat without compensating vapor sink” defect on the exercised implicit-flux path. It does **not** establish F6 as the root cause of the coupled MPAS runaway, and `den_bech` alone cannot discriminate F6 from radiation/surface-flux feedback.

Two stronger unaccounted candidates need immediate audit:

- Unless explicitly passed, MPAS runs use `bechtold_dx_m=0`, i.e. legacy ZTAURES=1 rather than the coarse-grid factor up to 3. The code claims the driver sets grid spacing, but the resolver only copies the user setting. `config.py:1453-1460`, `physics_pipeline.py:2983-2985`, `bechtold.py:414-430`.

- If `--volcanic-aerosol-lw` was actually enabled, the forcing path converts CMIP `ext_earth` **extinction** to a column AOD, then supplies it as pure LW **absorption** and distributes it through the whole pressure column. `external.py:493-500,562-578`, `model_driver.py:2415-2421`, `rrtmgp.py:599-605,842-849`, `surface_utils.py:237-257`. That violates the radiation API’s stated extinction-to-absorption requirement and destroys the stratospheric vertical structure. It is a serious deck-path bug, conditional on the explicit LW flag being true.

## Claims F1–F10

- **F1: confirmed, with two corrections.** `cape_sink_heating_ratio` at `test_bechtold_column_conservation.py:47` is absent from `BechtoldConfig`, so collection fails. All **three** tests in the file are blocked, not two. However, “silently disabled in CI” is not established: ordinary pytest collection would fail CI unless this file/job is excluded. Also this is not probe-exact: it uses `M_b_max=0.02`, `dt=300 s`, L20, whereas your run uses 0.05, 75 s, L30 (`:42-70`).

- **F2/F3: physically correct.** In-plume conversion replaces `plume.q_c_u` at `bechtold.py:2489-2492`; the constant `precip_efficiency` split is bypassed at `:2815-2868`. SBM does not read it (`physics_pipeline.py:2928-2933`). Call the two arms *physically identical*, not necessarily byte-identical at the config/JIT-cache level.

- **F4: SBM irrelevance confirmed; “serialization artifact” unproven.** `ExperimentConfig` uses `None`, but the legacy AMIP schema has a literal `0.0` default at `amip_config.py:108-110`. `config.py:2510` merely copies the value, and JSON serialization uses `_asdict()` without coercion (`config.py:2566-2589`). The 0.0 may be a legacy-schema default, not a None→0 serialization conversion. It remains harmless for SBM.

- **F5: confirmed.** The MPAS loop does persist the physics carry: `model_driver.py:6196-6198,6510-6517`. No reset-every-step bug found.

- **F6: plausible mechanism, not confirmed root cause.** A saturated `M_u` is evidence of vigorous transport, not energy creation. The cap deliberately leaves fully triggered columns unchanged (`bechtold.py:2703-2733`). Your storm-track example at L26/902 hPa is lower troposphere, not mid/upper troposphere; it does not support the stated detrainment-height attribution. The L13 bulge needs process-split heating profiles.

- **F7: the alleged “full anvil plus rain” double-book is misstated.** The anvil source uses the *post-conversion residual* `q_c_u`, not the original unconverted plume condensate (`bechtold.py:2489-2492,2781-2805`). Rain gets a separately paired vapor sink and latent heating (`:2824-2867`). The approximation may be poor, but the tested path is not the claimed double count.

- **F8: sign and local KE→heat closure are correct, but its magnitude is unknown.** `dT/dt=-(u\,du/dt+v\,dv/dt)/c_p` is correctly signed (`hines.py:299-308`). Still, the scheme explicitly has an unbudgeted launched-wave source and unclosed limiter/cap losses (`hines.py:65-84,127-136`). Code alone cannot justify “cannot be 368 W m⁻² globally”; diagnose `eps_gwd`.

- **F9: confirmed, and incomplete.** The deck may also change transient GHGs if `experiment` is a transient template (`model_driver.py:2430-2448`), besides ozone/aerosol/volcanic fields. Radiation cadence is another confound because heating may be held between solves (`model_driver.py:6501-6507`).

- **F10: likely inert in the dry columns, but zero log lines do not prove no interception.** The code logs only the current step at sparse cadence; it does not accumulate hits between logging times (`model_driver.py:6542-6553`). Instrument a cumulative count/energy drain before calling cap10 irrelevant.

## Leaf conservation and MPAS paths

I found no normal finite-value sign reversal in the requested paths:

- The implicit mass-flux sign is conservative and correctly oriented: `mass_flux.py:558-603`; cloud detrainment is paired with vapor removal at `:615-620`.
- The latent pairings at `bechtold.py:2781-2805` and `:2842-2867` have the correct signs.
- Radiation’s net-flux-divergence sign is presently correct at `two_stream.py:1019-1036`.

So R≈0 plus water closure is sufficient to reject the original mechanism **for normal finite leaf calls using this code path**. It is not a full MPAS energy proof. Independent `nan_to_num` sanitization of T, vapor, and cloud tendencies can break a paired budget once non-finite values occur (`bechtold.py:3089-3102`); monitor it.

Your omitted optional inputs do not expose a hidden MPAS conservation path: the standalone MPAS bridge does not pass `shf_w_m2`, `lhf_w_m2`, `dT_dt_rad`, `land_frac`, or `omega` at all (`integration.py:570-586`). Thus the leaf probe’s `None` values are representative. But this is itself a wiring problem:

- CAPDCYCL cannot run without surface fluxes and land fraction (`bechtold.py:2580-2617`).
- Shallow closure cannot run without SHF/LHF (`:2646-2666`).
- MPAS has no reconstructed `v`, so it passes `moisture_convergence=None` (`integration.py:424-463`). Contrary to that bridge comment, Bechtold then uses pure PBL-CAPE; there is no saturation-deficit proxy in the leaf (`bechtold.py:2237-2249`).
- `land_frac=None` makes the land RH-break inactive in subcloud evaporation (`bechtold.py:3004-3013`).

Therefore “all IFS flags on” does not mean all IFS closures are active on MPAS.

## Revised hypothesis

I agree with this narrowed statement: Bechtold may shape the dry, warm profile while surface turbulent fluxes and radiation provide net column energy. Fixed SST can indeed provide latent/sensible enthalpy; that needs a flux ledger, not inference from CAPE.

I do **not** agree that “drier air collapses LW cooling” is established; absent cloud/aerosol effects, less water vapour generally increases OLR and cooling. Nor is the inferred 368 W m⁻² secure unless the reported temperature is area- and pressure-mass-weighted and includes the moist-enthalpy tendency. A simple global mean of model-level T cannot be converted that way.

My ranking:

1. Verify whether LW volcanic aerosol was active; if yes, its absorption/extinction and vertical-placement errors are a major direct forcing candidate.
2. Verify `bechtold_dx_m`; if zero, the missing coarse-grid turnover weakening is the leading Bechtold-specific defect.
3. Treat `M_u` plateau as a credible amplifier/profile shaper, not yet the root source.
4. Measure surface turbulent flux, radiative convergence, and Hines dissipation before excluding either.

## Best discriminating test

`den_bech` is the best **first screen**, but it is not decisive:

- If it runs away, that proves Bechtold is sufficient to trigger the coupled feedback under the pinned deck—not that F6 supplies energy.
- If it stays healthy, it does not exonerate Bechtold because a Bechtold×deck interaction remains possible.

The decisive single test should be an **instrumented 2×2 restart experiment**, treated as one factorial test: SBM/Bechtold × pinned/transient deck, all from one identical pre-divergence checkpoint. On every step, save area-weighted column budgets for convection, turbulence, radiation, GWD, microphysics, moisture clipping, TOA/surface net fluxes, water, CAPE, and fraction of columns at `M_u/M_b_max`.

Use

\[
Q_i=\left\langle\int (c_p\,\dot T_i+L_v\,\dot q_{v,i})\,dp/g\right\rangle .
\]

Expected discriminator:

- `Q_conv≈0` in all Bechtold arms; positive system energy must appear in turbulent/radiative/GWD terms.
- A Bechtold main effect with pinned deck identifies convection-triggered feedback.
- A deck main effect identifies forcing.
- A non-additive term identifies the interaction.
- Repeat the Bechtold-pinned member with physical `dx_m≈sqrt(cell area)` or a lower `M_b_max`; removal of the plateau/bulge then specifically supports F6.
