# Gap 4: selectable bulk snow on OMIP prognostic ice

Codex author; Claude and GLM reviewers. This is an uncommitted capability
addition requested by the user, not closure of gap 4 or SI3 equivalence.
No production card, default, or simulation is changed. No climate measurement.

## Inventory before edits

Searched `packages/ice` for BL99, bitz_lipscomb, snow, conductivity, layer
state, flooding and their call sites; searched CORE-II setup, forcing and
existing ice tests before adding code. The existing implementations are:

- `ice/_future/bitz_lipscomb.py:147`: grid-independent implicit multilayer
  **ice-only conduction** core, brine enthalpy and Untersteiner conductivity.
  No production caller. No growth/melt/remapping integration. Its tests are
  `tests/unit/test_bitz_lipscomb.py`; their existence is not integration.
- `ice/state.py`: a single surface temperature and bulk snow depth per
  category, no prognostic ice or snow layer enthalpies/temperatures.
- `ice/snow.py`: grid-independent snowfall accumulation, series snow/ice
  resistance, snow-first melt and sublimation, deposition, and Archimedes
  snow-ice flooding. `SnowConfig` already exposes density, conductivity,
  heat capacity, minimum depth, flooding and sublimation partition.
- `ice/sea_ice.py:_thermo_v2`: these snow kernels are already called by the
  extended thermodynamics. Works on arbitrary column shapes, including
  tripole, regular lat-lon, cubed sphere and MPAS. Grid-specific limitations
  concern dynamics/transport, not this local column calculation.
- Snow volume already participates in C-grid transport, category lift/remap,
  ridging and the CORE-II restart state. No new state or transport is needed
  for the bulk subset.
- CORE-II `main()` enables brine, selecting extended thermodynamics, but
  omits snow configuration: `enabled=False`, `k_snow=constants.k_snow` (0.31 W/m/K),
  `flooding=True` (gated by enabled). Its forcing adapter already forwards
  snowfall, using zero if the forcing cache lacks snow. Native ice init
  already loads optional `ht_s` into bulk snow depth.
- **Correction to the original framing:** disabled snow gates accumulation
  and flooding, not all snow effects. The brine-enabled extended path still
  uses initialized snow in resistance, heat capacity and ablation; deposition
  can also populate snow. The new `off` selector preserves that behavior.

## Available oracle, source and branch

Read NEMO **5.0.1**, under
`/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1`, not the unavailable
5.0.2 path in the house rules. ORCA1's local `EXPREF/namelist_ice_cfg`
(`.../nemo_orca1/ORCA1/EXPREF/`) sets `nlay_i=3`, `nlay_s=3` at lines 25–26,
ice thermodynamics on at line 32, conduction-flux boundary forcing off at
line 81, and `rn_cnd_s=0.5 W/m/K` at line 92.

The reference namelist selects BL99 and **Pringle (2007)** ice conductivity
at lines 165–167. `src/ICE/icethd_zdf.F90:59–66` dispatches the BL99 solver
with `np_cnd_OFF` for this boundary condition; lines 90–91 read reference
then configuration and line 112 forwards the snow conductivity.
`icethd_zdf_bl99.F90:261` is the Pringle arm, unlike our unused core's
Untersteiner formula. Lines 314–329 construct snow/ice interface conductance;
lines 338–347 include separate ice/snow heat capacities.
`icethd_dh.F90:447–485` performs flooding, seawater mass/salt/heat exchange,
and transfers snow-layer enthalpy to ice. Our bulk flooding does not reproduce
that enthalpy treatment. These are source/namelist observations; no running
binary or preprocessor configuration was certified. Issue #1455 read failed
(network); no comment posted.

## Added versus reused

Only selection/validation/forwarding is added to CORE-II:

- `--ice-snow off|bulk`, default `off`, forwards to `SnowConfig.enabled`.
- `--ice-snow-k`, default `None`, forwards an explicitly supplied finite,
  positive conductivity into existing `SnowConfig.k_snow`.
- `--ice-snow-flooding on|off`, default `None`, forwards to the existing
  flooding switch. Omitted overrides retain the existing config values.

Unknown selections raise; bulk requires prognostic ice, and overrides require
bulk. The FESOM runner has a separate setup and its existing allowlist rejects
these new options. The shared non-FESOM setup supports them on tripole and
its other supported prognostic-ice grids. All physics kernels are reused.

Example selection, not a changed card or launched experiment:

```text
--prognostic-sea-ice --ice-snow bulk --ice-snow-k 0.5
```

The empirical conductivity is supplied to an existing config field; no literal
coefficient is inserted in a function body or signature default. Existing
library conductivity and every other coefficient remain unchanged.

## Flux contracts and verification plan (before tests)

All surface fluxes are W/m2 per ice area unless explicitly area weighted:
absorbed shortwave and net longwave positive into the skin; sensible and latent
heat positive upward to the atmosphere. Conductive flux is positive upward
from base to skin: `K*(T_base-T_new)`. The surface balance is
`Q_sfc + F_cond = C_skin*(T_new-T_old)/dt + surface_melt_energy/dt`.
Melt energy first consumes snow, then ice; surplus after full ablation is
returned to the ocean. `ocean_heat_extraction` is positive **out of the ocean**;
surplus heat and transmitted shortwave are negative contributions to it.
Freshwater is positive into the ocean; snowfall increases the ice snow store,
sublimation removes it, and flooding withdraws seawater from the ocean.

Tests will exercise actual main construction statements, each new forwarding
hop and guard, single/multiple category tripole steps, snow mass accumulation,
conductivity response, surface sensible/melting/melt-out energy balances,
flooding mass exchange and switch, eager/JIT and gradients. Production-line
mutations must fail the corresponding tests. All runs use `JAX_PLATFORMS=cpu`.

## Exact boundary: what this is NOT

This is **not BL99 multilayer ice or three-layer snow**, not full SI3 flooding
enthalpy, and not a demonstrated improvement of the ORCA1 climate. Full BL99
needs prognostic layer state, surface and basal phase change, snow-layer
conduction, enthalpy transport, ITD/ridging remapping, restart I/O, and the
actual selected NEMO conductivity/solver details. That is a real build.
The existing bulk surface equation's closure is distinct from a multilayer
whole-column enthalpy budget: changing snow/ice mass and flooding are not
represented by prognostic layer enthalpy. No such total-energy equivalence is
asserted. The existing extreme cold surface-temperature clamp and inherited
transport/coastal limitations are not upgraded by this wiring.
No new snow-dependent albedo is selected: constant shortwave/albedo remains.
Missing snow in a forcing cache retains the existing zero-snowfall behavior;
this does not derive a snow/rain partition. `run_omip.py` and FESOM wiring are
outside this CORE-II tripole subset.

## Files touched

- `scripts/run/run_omip_core2.py`: resolver, CLI and real setup forwarding.
- `tests/ice/unit/test_omip_bulk_snow.py`: new wiring and physics tests.
- `tests/unit/test_run_omip_core2_ice_categories.py`: supply the unchanged snow
  default to the existing gap-3 AST construction fixture.
- This scope record. No file under `packages/ice` is changed.

## Validation receipts (CPU)

All pytest calls use `JAX_PLATFORMS=cpu JAX_ENABLE_X64=1` and
`/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python`. This checkout
has no `.venv/bin/python`; the runner in `logs/gap04/run_tests.py` prepends
this checkout's root, `src`, and package directories to `sys.path`.

- New bulk snow tests: **37 passed** (`logs/gap04/new_final.log`). They cover
  parser/real main/resolver/config forwarding, guard failures, independently
  refused FESOM selectors, one/five-category C-grid steps, actual CORE-II
  snowfall forwarding, JIT parity and a nonzero conductivity derivative
  checked against centered finite differences.
- Existing CORE-II category tests: **29 passed**; dispatch ratchet:
  **24 passed** (`logs/gap04/targeted_final.log`, combined with an earlier
  36-test version of the new suite: 89 passed). The script resolver is outside
  that ratchet's package-only discovery; its unknown-value guards are tested
  directly and mutation checked.
- Existing ice regressions: **137 passed** (`logs/gap04/regression.log`):
  extended physics 68, snow kernel 14, C-grid transport 9, convergence ridging
  16, transmitted shortwave 5, surface thermodynamics 25.
- **227 unique relevant tests passed.** Focused repetitions are not added.
- Broad constants/coefficient hygiene: **4,799 passed, 15 failed, 2 skipped**
  (`logs/gap04/hygiene.log`). All 15 failures reproduce against an untouched
  `git archive HEAD` snapshot (`logs/gap04/baseline_hygiene.log`: 15 failed).
  They concern existing FV3/ocean/probe sites; no allowances, constants,
  tolerances or skipped-test markers were changed.
  Final changed-file constants checks: **3 passed**
  (`logs/gap04/changed_constants.log`; repeated checks, not added above).
- **28/28 production mutations detected** (`logs/gap04/mutation_receipt.txt`,
  per-case logs under `logs/gap04/mutations/`). Temporary functions are compiled
  in memory with inspectable temporary source; production files are never
  replaced. This covers each new forwarding hop, parser and resolver guards,
  FESOM refusal, snow accumulation/precipitation mask, conductivity/flooding
  consumers, heat capacity, upward basal conduction, atmospheric heat-flux
  signs and the melt-surplus return. The first surface-only check did not
  detect a reversed basal conductive flux; the added basal latent-growth test
  now detects it. Two initial mutation targets were ambiguous (no test ran);
  their corrected targets both produce actual pytest failures.

Surface closure tests independently reconstruct the skin storage and boundary
flux balance for cooling, snow-only melt, complete snow/ice ablation with
surplus heat returned to the ocean, and fully supplied sublimation/deposition.
Their residual tolerance is 1e-8 W/m2 or tighter in float64. Basal conduction
has a separate ice-volume/latent-heat check; flooding has an independent
snow-to-ice/ocean mass balance and negative ocean-withdrawal sign test. These
are synthetic local process tests, not a global coupled energy certification.
Complete-melt closure isolates phase melting with zero sublimation; the
existing cold clamp and capped-sublimation re-solve are not certified here.

## Defaults and review status

`logs/gap04/check_defaults.py` compares the actual HEAD and working parser:
**all 205 existing CLI defaults are unchanged**. The new defaults are
`ice_snow="off"`, `ice_snow_k=None`, `ice_snow_flooding=None`.
The entire ice config source and constants source are byte-identical to HEAD:
**all 75 nested ice-config leaves are unchanged**. In particular, slab
thermodynamics, disabled snow, conductivity 0.31 W/m/K, the existing True
flooding switch gated by snow enablement, category count, dynamics/transport,
ridging, brine, shortwave/albedo, lead-freeze selection, physical coefficients
and numerical limits retain their existing defaults. The production card is
untouched; no experiment or integration was launched. No commits or staging.

**UNREVIEWED: Claude and GLM approval unavailable.** The Claude CLI diff-review
attempt reports `Not logged in`; GLM's existing review client fails DNS
resolution. Logs: `logs/gap04/claude_review.log` and
`logs/gap04/glm_review.log`. Neither is an approval. The author has not acted
as an independent reviewer or substituted a differently named model.

Choices: ASKED — expose a self-standing subset, reuse existing snow physics,
preserve every default and the production card, and do not commit.
UNASKED scientific selections: none.

UNVERIFIED: independent review approval, full SI3/BL99 layer thermodynamics,
whole-column enthalpy equivalence, real ORCA1 climate/distributed integration,
and the original job's runtime/binary provenance. Inventory statements are
source observations, not a new faithfulness score.
