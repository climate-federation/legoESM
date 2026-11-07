# Gap 6: ocean-owned surface liquidus for prognostic OMIP ice

Codex author; Claude and GLM reviewers requested. Uncommitted capability
addition, not gap closure or a climate result. User explicitly requires all
defaults/cards unchanged and no commits. No integration or measurement lane
is run; the verification below is unit-level, with this scope recorded before
its tests. No claim about job 9701531's executable or NEMO 5.0.2.

## Inventory before edits

Searched ice config/state/thermodynamics/brine/transport/ITD, ocean EOS,
CORE-II construction and exchange consumers, existing freezing and salt-budget
tests, and shared thermo before adding anything. No automated gate exists for
the completeness of this search.

- `packages/ice/legoesm/ice/config.py:BrineConfig`: optional bulk salt tracking,
  default disabled; new lead/basal ice salinity is the configurable
  `constants.S_ice_bulk_default` (4), with numerical bounds 0..12 PSU.
- `state.py` already carries `S_ice` per category. `brine.py` evolves this bulk
  salinity by mixing new lead/basal ice, salty flooding, fresh refreeze,
  melt removal, and salt retention during sublimation. Bounds release excess
  salt through the same budget residual. This is NOT fixed salinity of all
  ice, but it is also NOT SI3 gravity drainage or melt-season flushing.
- `sea_ice.py:_thermo_v2` calls that salt budget; `_step_dynamic_v2` transports
  salt inventory and sums category exchanges. Tracer-aware Lipscomb ITD remaps
  salt. `BrineConfig.enabled` selects this path; there is no drainage/flushing
  selector. Existing `--prognostic-ice-salinity` sets new-ice salinity only.
- The ice thermodynamic paths use `SeaIceConfig.T_freeze_ocean`, a fixed
  Kelvin value by default (-1.8 C). Basal conduction, basal turbulent heat,
  lead gating/heat deficit and ice-free skin values use it. The separate
  freshwater `T_melt_surface` caps surface melt at 0 C. `_future/bitz_lipscomb.py`
  is not a selectable live multilayer thermodynamic path. It does already
  provide an ice-salinity-dependent melting temperature `-mu_ice_freeze*S`
  in Celsius, brine enthalpy, heat capacity and conductivity. These use ICE
  salinity, not the ocean surface liquidus, and remain unconnected.
- Ocean `eos.freezing_point` already exposes constant, MOM6 `linear_S`, and
  UNESCO/Millero `unesco`, returning Kelvin, with pressure in Pa for UNESCO.
  `nemo_eos_fzp` separately implements NEMO's TEOS-10 CT/SA polynomial, returning
  Celsius with optional depth in metres. It was not in that scheme dispatch.
  Its unguarded square root has a NaN AD endpoint at S=0 even though the full
  product has a finite derivative. The new endpoint test must detect reversion.
- `--freeze-scheme` currently feeds ocean freeze-floor and prescribed-ice
  relaxation surrogates, not prognostic ice. These remain separate controls.
- Current CORE-II host tripole prognostic construction enables brine, retains
  new-ice salinity 4 unless overridden, uses fixed liquidus, slab thermal
  physics, `nemo_qlead`, constant optics, runner transmission 0.03, snow/ponds
  off, one category unless overridden, free drift and fold-aware advection.
  Multi-category selection enables tracer-conserving ITD. No production card
  is changed; these are source-resolved defaults, not job-runtime provenance.
- Ice depends only on core, not ocean: the new boundary must be computed by
  the driver and passed into ice, preserving package independence.

## Available oracle, source and branch

Read NEMO **5.0.1** at
`/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1`, with configuration in
`/burg-archive/glab/users/pg2328/nemo_orca1/ORCA1/EXPREF`.
`ORCA1/cpp_ORCA1.fcm:1` enables `key_si3`.
`namelist_cfg:308` selects `ln_teos10=.true.`; the EOS-80 line above is commented.
`src/ICE/icestp.F90:134-135` evaluates `eos_fzp(sss_m)` at the surface and adds
`rt0` to obtain Kelvin. `src/OCE/TRA/eosbn2.F90:1674-1688` is the selected
TEOS-10 polynomial; the EOS-80 alternative starts at 1691. Our existing
`nemo_eos_fzp` reuses that polynomial. No salinity-scale conversion is implied:
TEOS-10 requires absolute salinity/CT; UNESCO expects practical salinity/PT.

**The cfg overrides the ref:** `namelist_ice_cfg:108` selects `nn_icesal=2`,
not the reference's `4`. Lines 113-124 enable drainage/flushing and set the
bulk restoring salinities/times. `src/ICE/icethd_sal.F90:202-246` is the live
CASE(2): warm surfaces flush toward 2, cold surfaces at/below the basal
liquidus drain toward 5, subject to thickness eligibility; bounds and the
salinity profile are then applied. CASE(4)'s permeability/Rayleigh machinery
is not this selection. Config `rn_sinew=0.75` at line 116 conflicts with its
own comment recommending 0.30 for option 2; we do not silently reinterpret
or copy that value. `src/ICE/icethd_do.F90:171-176` confirms that CASE(2,4)
forms new lead ice at `rn_sinew*sss_1d`, rather than the constant CASE(1).
`icethd_sal.F90:645-646` reads reference then cfg, verifying the override order.
These are source/namelist observations, not proof of a
particular running binary's namelist.

`src/ICE/icesbc.F90:364` uses the liquidus in the mixed-layer freezing energy;
line 371 uses it in basal heat exchange. Our scalar heat-transfer coefficient
is still a bulk approximation to SI3's friction-dependent transfer.
Issue #1455 could not be read (GitHub connection failed); no comment posted.

## Added versus reused

Add `--ice-freeze-scheme constant|linear_S|unesco|nemo_teos10`, default None.
Every explicit choice, including constant, requires prognostic ice. Unknown
choices raise in parser/resolver/evaluation. FESOM rejects the new selector;
cube/scan exclusions for prognostic ice remain. The main host call reads live
top-level ocean salinity and evaluates the existing ocean EOS in Kelvin.

Add optional `step_sea_ice(ocean_freezing_temperature_K=...)`, default None,
requiring the SST spatial shape without a category axis. It locally replaces
the basal boundary in the immutable config, so all thermal consumers see the
same field and every category shares its ocean cell's liquidus. The caller's
config stays unchanged; no new prognostic state or package dependency.
Reuse the existing thermal operators, bulk salt budget, freshwater channel,
heat response and ocean blend. Repair only the TEOS-10 zero-salinity adjoint
by guarding the square root's argument; preserve forward polynomial values.

Example opt-in (not a production card or experiment):
`--prognostic-sea-ice --woa-init --ice-freeze-scheme nemo_teos10`.
Input thermodynamic scales must already be appropriate; this option does not
convert practical salinity/potential temperature to SA/CT.

## Flux contracts and verification plan (before tests)

Let M_i = rho_i * sum_k(a_k h_k S_k) / 1000, kg salt per m2 of ocean cell.
Ice emits `salt_flux=(M_i_old-M_i_new)/dt`, positive INTO ocean. The ocean
mapper preserves that sign, without another concentration weight: the ice
exchange is already per-cell. Thus dt*F_salt + Delta M_i = 0.
Freezing salty ice has NEGATIVE salt flux (ice takes salt); melting has
POSITIVE salt flux. Freshwater is negative on freezing and positive on melt.
Net freezing salinification comes from water removal exceeding proportional
salt removal, not from reversing the salt channel or adding rejection twice.
At fixed ocean reference mass M_o the virtual tracer increment is
`(1000*F_salt - S_o*F_water)*dt/M_o`; the exact mass exchange interpretation is
`(M_o*S_o + 1000*F_salt*dt)/(M_o + F_water*dt)`.

`ocean_heat_extraction` is positive OUT of ocean; the mapper negates it to
positive-IN `q_net`. The liquidus changes basal conduction through
K*(Tf-T_skin) and basal heat through C*max(SST-Tf,0). Lead freezing uses
`zqfr=rho_o*cp*dz*(Tf-SST)` and `qlead=min(0,zqld-zqfr)`. Ice takes volume
`-qlead/(rho_i*L_f)` and returns latent heat to ocean. Tests must show that the
open-water cooling plus returned latent brings a supercooled column to its
local liquidus, and that hotter, saltier columns can remain ice-free.
This does not alter C, freshwater surface melting, or atmospheric forcing.

CPU tests will exercise actual main resolver and step expressions, all EOS
selections, nonuniform fresh/saline columns, one and five categories,
independent lead/basal ledgers, growth/melt salt closure through the actual
blend, JIT and nonzero gradients. Mutation controls must fail on the production
forwarding and numerical lines, not just compare a routine against itself.

## Exact boundary: what this is NOT

Not SI3 salt relaxation, new-ice entrainment law, vertical salinity profiles,
layer enthalpy/BL99, permeability drainage, flushing hydraulics, full SI3 ice,
or a global salt/energy/climate certification. The CASE(2) subset still needs
its eligibility/order/bounds and salinity-profile/enthalpy interaction ported
and reviewed before claiming oracle evolution. No arbitrary relaxation is
added to the existing conservative salt store. Its fixed new-ice salinity
and numerical cap remain debt. New-ice salinity is not capped by local ocean
salinity: below the configured new-ice salinity, that entrainment convention
can demand more salt than the freezing seawater contains. This subset does
not claim a physically suitable freshwater-ice salt law. Coupled growth/melt
tests use ocean salinities 5..40, above the default new-ice value; the EOS
itself is separately tested at zero salinity. No production card, default, simulation,
`run_omip.py`, FESOM construction or ocean EOS selection is changed. No 5.0.2
claim and no integrated NEMO-faithfulness score. No new climate numbers are
citable from this uncommitted work.

## Files touched and validation receipts

- `packages/ice/legoesm/ice/sea_ice.py`: optional Kelvin boundary and shape guard.
- `packages/ocean/legoesm/ocean/eos.py`: finite TEOS-10 derivative at S=0.
- `scripts/run/run_omip_core2.py`: selector, resolver, EOS reuse and actual
  host-loop surface-salinity forwarding.
- `tests/ice/unit/test_omip_ice_freezing.py`: new selection, independent
  thermodynamic/coupling budgets, legacy-path, JIT and gradient tests.
- This scope record. No existing test or fixture modified. Unrelated existing
  untracked files are untouched; no staging or commits.

All tests use `JAX_PLATFORMS=cpu JAX_ENABLE_X64=1` and
`/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python`, via the existing
`logs/gap04/run_tests.py` checkout-import wrapper. New tests explicitly set
and restore `PrecisionPolicy.fp64()`; environment x64 alone is not relied on.

- **28 new tests passed**, `logs/gap06/final.log`.
- **68 existing extended-ice tests passed**, `logs/gap06/expanded.log`
  (95 passed total there, including an earlier 27-test version of the new
  suite; those repetitions are not added to the unique count).
- **104 existing regression tests passed**, `logs/gap06/regression_actual.log`:
  ice salt 15, lead heat 7, gap-5 optics 31, two-way ocean coupling 11,
  ocean freezing 33, prescribed ice-shelf melt 7.
- **24 hygiene tests passed**, `logs/gap06/hygiene.log`, targeting changed
  production files and constant/coefficient detector/fingerprint controls.
  This is a targeted check, not a whole-repository test claim.
- **224 unique pytest tests passed** (28 new + 172 existing + 24 hygiene);
  repeated focused runs are not counted twice.
- **30/30 production mutations detected**: 29 in `logs/gap06/mutations.log`
  and the independent surface-melt-temperature control in
  `logs/gap06/mutations_surface.log`. Individual assertion failures live in
  `logs/gap06/mutations/`; the script compiles functions in memory and never
  overwrites tracked source. Covers parser, inactive/unknown/FESOM refusals,
  actual main resolver and surface-level forwarding, EOS dispatch and Kelvin
  conversion, ice-step boundary and shape guard, conduction, basal heat,
  both lead sources, skin temperature, freshwater surface-melt independence,
  new/basal salt inputs, salt residual, ocean salt mapper/consumer, ocean heat
  sign, and reversion of the NaN endpoint fix.

The lead ledger reconstructs latent energy and new ice mass from the local
surface liquidus, mixed-layer capacity and prescribed open-water cooling.
The basal ledger independently reconstructs implicit skin conduction and
heat-driven volume change. Salt storage is reconstructed from post-step ice
state, not from the reported salt flux. It closes against the actual main
blend's salt channel and the production ocean real-salt tendency helper
(`ocean_pe_latlon_cgrid.py:4134-4143` consumes this helper with actual cell
thickness). The heat balance tests the actual main blend's q_net. These are
column-local checks on a synthetic tripole grid with one/five categories;
no advective transport, distorted native ORCA geometry, full ocean timestep,
long integration or global enthalpy certification is inferred.

Test corrections are recorded rather than hidden: the first four basal
expectations omitted the existing minimum-snow-resistance floor. Correcting
the ledger, without changing production or tolerance, exposed a second
incorrect test assumption that melt always freshens. Melting 8-PSU ice into
5-PSU water salinifies; into more saline water it freshens. The final test
checks both, while salt flux is positive in both. These were errors in NEW
tests, not an existing test pinning a degenerate physical limit. The initial
regression command also named a nonexistent ice-test path and collected zero;
the corrected command above produced the reported pass count.

## Defaults and review status

`logs/gap06/defaults.log`, using the existing mechanical default audit:
**209/209 existing CLI defaults unchanged; 75/75 nested ice-config leaves
unchanged**. Ice config and physical constants source are byte-identical to
HEAD. No production card changed. New CLI default is `ice_freeze_scheme=None`;
new ice-step boundary defaults to None and preserves custom fixed-temperature
configs. This includes new-ice salinity 4, ice salinity bounds, ocean reference
salinity, fixed -1.8 C basal liquidus, freshwater 0 C surface melt, ocean heat
coefficient, all snow/optics/transmission/pond/category/ITD/dynamics/transport/
ridging defaults, lead source, ocean EOS and surrogate freeze selections.
The endpoint fix changes the derivative at zero salinity, not a selector or
forward physical coefficient.

**UNREVIEWED: no independent approval.** Claude CLI was terminated after its
180-second timeout without a verdict (`logs/gap06/claude_review.log`, exit
124); its end hook also reported read-only plugin storage. GLM's existing
client failed DNS (`logs/gap06/glm_review.log`). An earlier GLM attempt raced
prompt-file creation and failed to read it; it was retried after the file
existed and then hit DNS. These failures are not approvals, and no substitute
reviewer or author self-review is presented as Claude/GLM review.

Choices: ASKED — inventory-first, largest correct reusable subset, explicit
salt/heat budgets, default/card preservation, CPU tests and no commits.
UNASKED production scientific selections: none. No simulation launched.
UNVERIFIED — independent review approval, input SA/CT provenance in an actual
job, native ORCA integration/climate effects, distributed execution, SI3
salinity evolution and layer thermodynamics. No automatic gate certifies
those omissions or independent review.
