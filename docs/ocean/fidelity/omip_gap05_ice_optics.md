# Gap 5: selectable state-dependent bulk ice optics in CORE-II

Codex author; Claude and GLM reviewers requested. Uncommitted capability
addition, not gap closure or an ORCA1 climate result. No default, production
card or simulation changes. No climate measurement; user prohibits commits.

## Inventory before edits

Searched `packages/ice`, shared surface albedo, CORE-II construction and
coupler consumers, and existing shortwave tests before building anything.

- `ice/shortwave.py:compute_ice_sw`: constant broadband albedo with configurable
  transmission; Maykut-Untersteiner temperature/thickness albedo with ZERO
  transmission and no snow dependence; `delta_eddington` is an empirical
  two-band Briegleb-Light surrogate, NOT a delta-Eddington transfer solver.
  It includes temperature, ice thickness, snow masking, pond area/depth and
  exponential bare-ice attenuation. These are existing `SeaIceConfig` choices.
- `ice/sea_ice.py:uses_new_physics` selects extended thermodynamics whenever
  shortwave is nonconstant (also for enabled brine, bulk snow, etc.). Its
  `_thermo_v2` consumes the shortwave result and returns transmitted SW weighted
  by the pre-thermodynamic category concentration. `_step_dynamic_v2` sums categories
  and subtracts transmission from ocean heat extraction. No new exchange
  channel is needed.
- `temp_dependent_albedo` / shared `surface_albedo.ice_albedo` is a separate
  legacy temperature-only broadband option (cold 0.65, warm 0.45 over
  5 K below ocean freezing). `_thermo_single` uses it in absorbed radiation
  and the legacy paths report it as response albedo; it has no transmission
  or snow dependence. It is not the extended shortwave energy partition. CORE-II enables brine, so this is not its active optics path.
- These optical kernels are column-local and accept arbitrary spatial array
  shapes: tripole/regular lat-lon, MPAS, cubed sphere, and a trailing category
  dimension. Grid restrictions are in the driver/dynamics, not radiation.
  CORE-II host prognostic ice supports tripole, lat-lon and MPAS; it rejects
  cubed sphere. FESOM has separate construction and rejects the new selector.
- The existing CORE-II host configuration resolves to constant albedo 0.65,
  incident-flux transmission 0.03 (CLI default), brine enabled, snow disabled,
  ponds disabled. Library transmission defaults to zero. Constant SW already
  debits transmission from ice absorption; the ocean blend's extra ice
  transmission is zero. The starting premise does not imply a budget leak.
- **Latent defect:** the surrogate's snow-band depth ramp interpolated from
  black, then its snow-cover ramp masked underlying ice. Thin snow could
  therefore darken ice. An existing form-pin test actually required this
  erroneous black endpoint. Repair the interpolation from underlying bare
  ice to snow, retaining all band coefficients and depth scales.

## Available oracle, source and branch

Read NEMO **5.0.1** at
`/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1`.
The configuration is `/burg-archive/glab/users/pg2328/nemo_orca1/ORCA1/EXPREF/`.
No 5.0.2 equivalence is claimed. These are source/namelist observations,
not certification of job 9701531's running executable.

`ORCA1/cpp_ORCA1.fcm:1` includes `key_si3`.
`src/ICE/icealb.F90:9` gates SI3 with `key_si3`; lines 155–169 implement
bare/melting ice, linear/log thin-ice albedo and exponential snow albedo;
lines 178–185 mix surface fractions and clear/overcast cloud effects.
`icevar.F90:1599–1600` implements the selected `nn_snwfra=2` snow fraction
`h_s/(h_s+0.02)` (reference namelist line 137). This is different from our
linear masking. `namelist_ice_cfg:151–152` disables ponds;
`icethd_pnd.F90:1416` then disables their albedo effect, so the relevant bare
ice branch is the no-pond arm, not the pond-enabled constant dry-ice arm.
`namelist_ice_ref:306–312` carries the dry/melting snow/ice and pond albedos;
the cfg namalb block has no overrides.

`namelist_ice_ref:150` selects `nn_qtrice=0`, not the state-dependent snow
extinction branch. `icethd_zdf_bl99.F90:151–157` therefore uses constant
snow extinction. Reference lines 170–173 set ice extinction 1/m, snow 10/m
(the unselected melting-snow value is 7/m). Lines 205–229 attenuate the
penetrating flux through snow and ice layers, assign layer absorption and
return bottom transmission. This is more than a replacement coefficient:
our bulk state has no interior layer absorption/enthalpy evolution.
The selected surface-scattering arm is `src/OCE/SBC/sbcblk.F90:1323–1338`:
it blocks penetration for **any** snow, and otherwise applies cloud-dependent
transmission with a thin-ice ramp to net ice shortwave. Thus nonzero
transmission through snow layers is not active under this local selection;
the snow-layer attenuation machinery still exists. The unselected
`nn_qtrice=1` arm at line 1345 uses a fixed fraction of net ice shortwave.
Full SI3 optics is a real build involving its albedo/cloud dependence,
scattering-layer partition, snow-cover law, layer heating and thermodynamics.
Issue #1455 read failed (network); no comment posted.

## Added versus reused

Add `--ice-shortwave constant|maykut_untersteiner|delta_eddington`, default
None (retain the current selection). Explicit selection requires prognostic
ice, including explicit `constant`. Unknown selections raise at parser and
resolver. State-dependent optics rejects explicitly supplied
`--ice-thermo-sw-trans`, even when set to the old default: it cannot affect
those schemes. Omitted transmission becomes the existing library zero for
those schemes; the optics kernel owns its computed transmission.

Reuse the existing optical dispatch, category handling, ice heat debit,
ocean heat credit and snow state. Repair only the surrogate's thin-snow
interpolation; do not port or label a partial formula as SI3.

Snow and ice albedos are distinct: constant and Maykut ignore snow optically;
`delta_eddington` uses carried snow depth in either snow mode. `off` preserves
initialized/deposited snow and existing ablation; `bulk` additionally enables
snowfall accumulation (before optics) and selected flooding. The snow
conductivity/flooding selectors retain gap 4 semantics. No snow option is
required merely to read snow already present. Pond physics remains disabled
in CORE-II; exposing an optical kernel does not enable pond evolution.

Example opt-in, not a production card or experiment:

```text
--prognostic-sea-ice --ice-shortwave delta_eddington --ice-snow bulk
```

## Flux contracts and verification plan (before tests)

Incident SW is nonnegative downward W/m2 per ice area. Reflected SW is
outgoing positive; absorption and transmission are incoming positive.
`reflected + absorbed + transmitted = incident`, with transmission bounded
by the non-reflected input. Beer-Lambert attenuation decreases transmission
with thickness. In this bulk surrogate all non-transmitted absorption heats
the surface skin; it does not resolve internal ice-layer heating.
`ocean_heat_extraction` is positive OUT of the ocean: transmitted SW must
enter it with a minus sign, weighted by the ice area present during optics.
The coupler converts extraction back to positive-into-ocean `q_net` exactly
once. CORE-II passes zero additional under-ice transmission. Consequently
transmitted light reaches the ocean through surface heat, not through the
open-water `sw_down` vertical radiation channel; under-ice vertical deposition
is outside this subset.

CPU tests will cover parser/resolver/main forwarding, inactive explicit flags,
snow brightening and transmission, one/multiple category production steps,
independently reconstructed SW closure, actual ocean blending, JIT and
nonzero gradients. Production-line reversions must make corresponding tests
fail. No integrated climate score or NEMO-faithfulness number is produced.

## Exact boundary: what this is NOT

Not SI3 optics, not full CICE delta-Eddington, not BL99 layer heating,
not NEMO cloud-dependent albedo, not NEMO snow scattering/extinction,
not pond evolution, not a vertical under-ice ocean radiation treatment,
not a global energy/climate certification. No NEMO 5.0.2 claim.
The existing surrogate uses incident-flux i0 and zero transmission under
fully snow-covered or pond-covered fractions. Its ramps and extinction
remain CICE-lineage approximations, different from SI3. `run_omip.py`, the
FESOM setup and production card are not upgraded. No new prognostic state.

## Files touched

- `packages/ice/legoesm/ice/shortwave.py`: repair snow-band interpolation;
  reuse bare-ice band albedos as the zero-depth endpoints.
- `scripts/run/run_omip_core2.py`: selector, resolver and host config forwarding.
- `tests/ice/unit/test_omip_ice_optics.py`: new selection, column budget,
  synthetic-tripole coupling, snow interaction, JIT and gradient tests.
- `tests/ice/unit/test_ice_shortwave_faithful.py`: replace the old black-endpoint
  form pin with the corrected ice-to-snow midpoint.
- `tests/ice/unit/test_omip_bulk_snow.py`: include the new actual main resolver
  in the shared AST construction fixture and forward explicit CLI flags.
- `tests/ice/unit/test_sw_transmittance_const.py`: replace the stale string
  forwarding assertion with execution of actual main construction.
- `tests/unit/test_run_omip_core2_ice_categories.py`: supply unchanged optical
  defaults to its category-only AST fixture.
- This scope record. Session logs/mutation scripts are under ignored
  `logs/gap05/`; unrelated pre-existing untracked files are untouched.

## Validation receipts (CPU)

All pytest runs use `JAX_PLATFORMS=cpu JAX_ENABLE_X64=1` and
`/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python`, through the
existing `logs/gap04/run_tests.py` checkout-import wrapper (no local venv).

- **31 new optics tests passed**, final `logs/gap05/new_final.log`. Real
  one/five-category steps use `create_synthetic_tripole`: regular metrics
  with an active fold, not native ORCA1 distorted geometry. Eager/JIT state
  and response agreement is checked on both category counts. Nonzero ice-
  thickness and snow-depth optical derivatives match centered differences.
- **105 existing tests passed**, `logs/gap05/initial.log`: bulk snow 37,
  shortwave form pins 34, constant transmission 5, CORE-II categories 29.
- **137 passed**, `logs/gap05/regression.log`: new optics 31, extended ice
  physics 68, pond albedo 3, ocean two-way coupling 11, dispatch guards 24.
  The new suite is counted only once: **242 unique functional tests**.
- **21 hygiene tests passed**, `logs/gap05/hygiene.log`: changed production
  file constants/coefficient checks and their detector/fingerprint controls
  (`-k 'shortwave.py or run_omip_core2.py or detector or fingerprint'`).
  This is a targeted check, not a claim that the full repository suite passed.
- **263 unique tests passed in total**; focused repetitions are not added.
- **26/26 production-line mutations detected**; consolidated receipt
  `logs/gap05/mutation_receipt.txt`, per-case logs under `logs/gap05/mutations/`.
  Covers parser/resolver/config forwarding, explicit/inactive selectors,
  FESOM refusal, both snow-band inputs and black-endpoint reversion,
  attenuation, carried snow, surface absorption, area weighting, single/multi-
  category ocean heat signs, ocean mapper sign and duplicate transmission in
  the actual main blend. Mutated functions compile in memory from temporary
  inspectable files; tracked production source is never overwritten.
  Two first-pass targets used a nonexistent `_step_v2` name, so no tests ran
  for those attempts. Corrected `_step_dynamic_v2` targets both produce actual
  pytest assertion failures (`mutations_followup.log`: 2/2 detected).

The skin budget reconstructs absorption from storage and upward basal
conduction independently of the SW kernel's residual. It closes reflected +
absorbed + transmitted against incident within 1e-8 W/m2 in synthetic x64
columns for every selection. Step tests separately compare the pre-step
area-weighted transmitted flux against negative ocean heat extraction, then
execute the actual main blend expression and verify the same gain in ocean
`q_net`. Post-step response albedo is checked against post-step category
optics; it is not substituted for the albedo that acted during the step.
This is a local SW/skin budget and forcing handoff, not whole-column enthalpy,
long-run coupled energy conservation or ocean vertical deposition validation.

The first new-suite run had one test failure: the FESOM error-message matcher
expected an underscore although production correctly printed the CLI hyphen.
Only the test matcher was corrected. No production guard was weakened.

## Defaults and review status

`logs/gap05/defaults.log` records the mechanical HEAD comparison using the
existing `logs/gap04/check_defaults.py`: **208/208 existing CLI defaults
unchanged**, new `ice_shortwave=None`; ice config and physical constants
sources byte-identical, **75/75 nested ice-config leaves unchanged**. No
production cards changed. This includes albedo 0.65, runner transmission
0.03, library transmission 0, constant optics, slab thermodynamics, brine/
pond/snow library switches, gap-4 snow defaults (`off`, None, None), snow
conductivity/flooding, category count, dynamics/transport, ridging, lead
freezing, optical coefficients and ramp lengths. The repaired opt-in
surrogate changes thin-snow behavior; preserving defaults does not mean
preserving that defect when callers explicitly select the surrogate.

**UNREVIEWED: neither independent reviewer returned approval.** Claude CLI
request timed out (`logs/gap05/claude_review.log`); its session-end hook also
reported read-only plugin storage. GLM's existing client failed DNS resolution
(`logs/gap05/glm_review.log`). No author self-review or substitute model is
presented as Claude/GLM approval. Requested independent review remains open.

Choices: ASKED — expose the largest correct reusable subset, repair the
latent snow-optics defect, preserve every default/card, CPU validation and
no commits. UNASKED production scientific selections: none. No staging.
UNVERIFIED — independent review approval, full SI3/CICE optics, native ORCA1
integration/climate effects, distributed execution and job runtime provenance.
