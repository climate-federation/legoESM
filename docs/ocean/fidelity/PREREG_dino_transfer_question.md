# Preregistration: DINO transfer question

Date: 2026-08-30. Session: `01a053d4-8e9f-7212-bbdb-19ba2d64e140`.
Branch: `fidelity/dino-transfer-codex`. **PREREGISTERED, NOT RUN.** No GPU or
NEMO arm was launched while writing this document.

Frozen evidence basis: `dino_split_explicit_momentum_chain_round94_result.md`,
`PREREG_climate_rebattery_round94.md`,
`dino_climate_rebattery_round94_handoff.md`,
`dino_full_step_coverage_round94_result.md`, `dino_step_chain_coverage.md`,
`dino_campaign_synthesis.md`, `dino_outstanding_fidelity_debt.md`,
`dino_1226_state.md`, `PREREG_verdict360.md`, and
`PREREG_floor90_ensemble.md` at the preregistration parent commit, plus the
subsequently registered `PREREG_multi_year_climate_equivalence.md` held battery
on the sweep branch. T1 reuses that battery's statistics but not its bridge.

## Question, order, and scope

Round 94 certified one narrow proposition: a clean fp64
`nemo_dino_kamm_mlf` twin, initialized from NEMO's day-180 restart with the
BEFORE, T-point-stress, and TKE bridges, passes the registered re-battery. It
did not certify legoESM's own initialization, the forward-Euler sibling card,
or construction through the public recipe catalog. These experiments test
those three transfer boundaries in cheapest-instrument-reuse order:

1. **T3 RECIPE ASSEMBLY**: CPU config identity, then a five-day twin;
2. **T2 THE SIBLING CARD**: the round-94 battery with one high-level variable,
   the card name;
3. **T1 STANDALONE START**: six fresh 20-year members per model.

Passing all three supports three separate transfer statements. It does not
turn the round-94 one-year twin into a proof of equilibrium, all-card fidelity,
or all-construction-path fidelity. Round 94 closed `116/116` accounting rows
but only `34/37` strict active rows; `ldf_dyn`, `tra_sbc`, and `tra_qsr` remain
outside strict active comparison (with the registered S zero-freshwater Rule
1b treatment). Transfer does not promote those rows. No aggregate verdict may
hide a failed row.

## Common admission and fail-closed rules

- Pin one clean producer commit and record its SHA, the session ID, Python/JAX
  versions, device, precision policy, grid/mask hashes, configuration dump,
  forcing clock, initial-state provenance, output hashes, and instrument
  hashes. A tracked-dirty producer invalidates an arm.
- Use fp64 computation and fp64 scored storage. `JAX_ENABLE_X64=1` alone is not
  admission; the artifact must stamp the fp64 precision policy.
- Use the DINO `2700 s` time step, 360-day/12-month model year, current faithful
  defaults, and no selector override unless the arm definition says otherwise.
- Any missing stamp, non-finite member, incomplete window, mask/grid mismatch,
  post-hoc replacement of a bar, or failed planted violation yields `INVALID`.
- Config comparisons are recursive leaf comparisons. They include field name,
  type, dtype, shape, scalar value, array bytes, enum/string value, and missing
  or extra leaves. NaNs match only at the same indices with the same dtype.
- Each scorer must ship classifier boundary tests and a planted physical or
  configuration violation. The plant must be scored by the same executable and
  must fail for the expected row. Self-comparison is not a control.
- Report every registered row and its verdict. `TRANSFER_CONFIRMED` requires all
  registered rows for that experiment to pass; otherwise report the exact
  failing rows and `TRANSFER_REFUTED` or `UNRESOLVED` as defined below.

## T3 — RECIPE ASSEMBLY

### Hypothesis and current catalog admission

The faithful `nemo_dino_kamm_mlf` configuration can be requested through
`legoesm.ocean.recipes.get_recipe`, assembled through
`legoesm.ocean.recipes.assemble_ocean_config`, and resolves identically to the
oracle DINO factory path.

At the preregistration parent this hypothesis was **false by source
inspection**. `packages/ocean/legoesm/ocean/recipes.py` contained only
`nemo_dino_v1`, documented as the cruder Wright/KPP, forward-Euler
approximation; the faithful card was not catalog-reachable. That is the frozen
pre-change T3 finding. `nemo_dino_v1` remains the negative control, never an
alias or admissible substitute.

**ASKED catalog choice (2026-08-30):** add the distinct public name
`nemo_dino_kamm_mlf_v1`. The entry and committed
`recipe_transfer_identity.py` probe implement the already-registered positive
arm. The name denotes the Kamm DINO setup's campaign-certified MLF structural
identity; dimensionful setup objects retain their explicit setup ownership.
The source-level reachability finding does not pre-judge the post-change
identity gate: only its zero-row artifact may do that.

### Arms

- **O, oracle path**: `dino_config_for_recipe("nemo_dino_kamm_mlf")`, then the
  existing DINO setup/factory to its fully resolved ocean model config.
- **C, catalog path**: `get_recipe(<faithful-versioned-name>, "latlon")`, then
  `assemble_ocean_config(...)` with independently constructed DINO setup
  parameters. The setup parameter builder may reuse the established DINO grid,
  vertical coordinate, forcing, and dimensional constants; it may not obtain
  recipe-owned values by flattening O.
- **P, plant**: a copy of C with one nonzero, behaviorally live resolved field
  changed (`dt` by one representable fp64 value is the default plant). The
  identity gate must name and reject that exact row.
- **N, catalog negative control**: `nemo_dino_v1` on the same setup. It must
  produce at least one named diff and can never satisfy the positive gate.

The committed probe must print both source bundles before assembly and the
complete flattened O/C/N resolved tables after assembly. It must also prove
that C's scheme keys came from the catalog entry, not setup overrides. A key
present in both recipe and setup is an ownership collision and invalidates T3.

### Frozen config bar

`CONFIG_IDENTITY=PASS` if and only if O versus C has:

- zero unequal leaves;
- zero missing leaves;
- zero extra leaves;
- zero recipe/setup ownership collisions; and
- a passing plant plus a nonempty N diff.

One differing row is a hard failure; there is no tolerance and no percentage
identity verdict. The current absence of a faithful catalog entry is
`UNMEASURED/NOT_ASSEMBLABLE`, not a pass.

### Behavioral check

Only after config identity passes, run O and C for five days from the exact same
admitted round-94 day-180 bridge. Preserve the seasonal clock, BEFORE bridge,
T-point-stress bridge, TKE bridge, fp64 policy, ladder `both`, and every harness
argument. Save days 0 and 5 in fp64. The construction path is the sole variable.

`BEHAVIOR_IDENTITY=PASS` only if every prognostic pytree leaf (`T`, `S`, `eta`,
`u`, `v`, all BEFORE levels, TKE state, and any carried auxiliary state) is
bit-identical at both snapshots and both runs stamp identical masks and forcing
epochs. A missing carried leaf or one unequal bit fails. This is a behavioral
guard, not an independent statistical bar.

T3 is `TRANSFER_CONFIRMED` only when both identity gates pass. Config identity
with behavioral mismatch is `TRANSFER_REFUTED`. A config mismatch stops the
GPU check and is also `TRANSFER_REFUTED` once an assemblable candidate exists.

## T2 — THE SIBLING CARD

### Controlled comparison

Run the round-94 re-battery verbatim with `nemo_dino_kamm` in place of
`nemo_dino_kamm_mlf`. The high-level experimental variable is the named card.
The instrument, bridge, NEMO references, duration, storage, duplicate policy,
clock, precision, and command-line selectors are unchanged. The resolved-config
receipt must list every leaf that differs between the two named cards. Because
the card bundles multiple intentional resolved differences, the result is about
the **sibling card as a bundle**, not an attribution to the outer integrator
alone.

The pinned instruments are:

- `kamm_twin_90d.py`, SHA-256
  `4be31374f2858a477f13f90c688c278feb28861bb12e175fab7e68f2e56b1448`;
- `climate_rebattery_score.py`, SHA-256
  `cf85b65302ed65d2a5d8b7ea7a77fb6fa35abd55b5fde8d6e541db9f266996db`;
- `mld_climate_audit.py`, SHA-256
  `cf1bcffb4ee7994bd4eb433f6b3fa3ba5ac0f2610463b9bb3133ea0c1762dc14`;
- NEMO five-day wall comparator `/tmp/dino_eta_waves/nemo_5d_eta.npz`,
  SHA-256
  `52bc6c70697126f7522114dbe2fc5cda5b56ce566db28f488d6809b79997b47a`.

No scoring code may be edited for the sibling result. The existing scorer does
not require an MLF recipe stamp; its faithful selector table is shared by these
cards. If the unmodified scorer rejects an FE artifact, T2 is `INVALID` and the
rejection is the result—do not create an FE exception after seeing outputs.

### Arms and frozen bars

Produce independent duplicate pairs `fe_climate_a/b` (360 days, fp64 3-D at
days 0, 90, and 360) and `fe_wall_a/b` (five days, every-step fp64 SSH). Each
pair must be bit-identical on every scored field before science scoring.

The round-94 bars remain byte-for-byte unchanged:

- southern-basin MLD RMS: `CONFIRM <= 11.2397455 m`,
  `REFUTE >= 20.2775 m`, otherwise `PARTIAL/INDETERMINATE`;
- day-360 basin: historical gap `-0.9519122331848315 Sv`, floor
  `F=0.06173656216045926 Sv`, and the exact five-branch decision tree in
  `PREREG_climate_rebattery_round94.md`;
- wall: `CONFIRMED` only when first-eight-step all-domain ratio `<=1.25` and
  wall share `<=0.17`; `REFUTED` when ratio `>=2.30` and share `>=0.38`;
  otherwise `UNRESOLVED`.

T2 reports the unmodified scorer's verdict and every component. Only a full
`CONFIRMED` result transfers the bridged certification to the sibling card.
This never certifies FE from a standalone start.

**NEMO need:** none if the hash-pinned wall artifact and the round-94 restart,
stepdump, and day-90/day-360 comparators remain admissible. If any is missing or
hash/provenance-inadmissible, the user must regenerate the corresponding NEMO
reference before T2; no substituted epoch is allowed.

## T1 — STANDALONE START

### Start-mode rule and legal recorded material

T1 uses `nemo_dino_kamm_mlf` with its own DINO analytic/from-rest initializer,
not `RUN_TRAJ`, `RUN_STEPDUMP`, a NEMO restart, or any before-level bridge.
The artifact must stamp `twin_start_mode=standalone`, initial-state factory and
hash, no bridge paths, and the first-step policy selected by the card itself.
NEMO likewise uses its own DINO from-rest path. The two initial states are
field-diffed and reported, never made identical by copying one model into the
other.

The following recorded references have sharply limited legal use:

| recorded NEMO material | legal T1 use | illegal use |
|---|---|---|
| `RUN_20Y_REBUILD/DINO_1y_00060101_00201230_grid_U.nc` and `RUN_40Y_REBUILD/DINO_1y_00210101_00401230_grid_U.nc` | deterministic years 1–40 ACC/control trajectory and drift context | ensemble floor or equilibrium evidence |
| corresponding `grid_T.nc` files | only variables that a committed audit first proves numerically valid | un-audited T/S climate truth; the campaign records corrupt history T/S scaling |
| `RUN_ENS_M{1,2,3}/` plus `noise_floor_out.txt`, recorded as three `1e-14`-relative-T branches from the NEMO year-20 restart over model years 21–30 | reducer/perturbation controls after restoration | any T1 arm or floor: the start and horizon differ |
| `RUN_VERDICT360_M0..M3` and the four day-180-bridged one-year verdict products | bridge-continuity control only | standalone climate level or from-rest variability |
| the held `PREREG_multi_year_climate_equivalence.md` six-member products, if run | instrument/reducer control | any T1 arm or floor: those members start at the day-180 bridge |
| registered maxima `ACC 0.091 Sv`, upper contrast `1.1e-4`, deep contrast `4.5e-5`, sigma max/mean `9.5e-5 kg m-3` | historical acceptance-gate context only | every T1 verdict denominator |

At preregistration, `RUN_20Y_REBUILD` and `RUN_40Y_REBUILD` are present at the
oracle path. The recorded `RUN_ENS_M{1,2,3}` member directories and
`noise_floor_out.txt` are not present there. That absence does not block T1
because those members are ineligible anyway. **No recorded NEMO ensemble or
published floor is legal as a T1 science arm or denominator.** T1 requires a
fresh from-rest NEMO ensemble; this is the NEMO-side compute need.

### Twenty-year standalone ensemble and sampling

Reuse the registered 20-year held battery with exactly one conceptual variable:
each model constructs its own start instead of receiving the common day-180
bridge. Each side has six fresh members: an unperturbed control and now-level
temperature relative kicks of `1e-14` at seeds 1, 2, 3, 4, and 5, applied to
that model's own initialized temperature. Every non-temperature byte is
identical across members within a model. Seed labels align construction but do
not create paired samples.

For NEMO, run its analytic DINO initializer once to produce a pre-step `t=0`
restart, then derive members 1–5 from that NEMO-owned restart by changing only
now-level temperature. This is an own-start checkpoint, not a cross-model
bridge. The t=0 capture path, perturbation receipt, and proof that every other
restart variable is byte-identical are mandatory NEMO controls.

Each member runs 20 360-day model years. Save full fp64 T/S/SSH/U/V at annual
endpoints y1–y15 and monthly endpoints throughout y16–y20: 75 matched dates,
with full-state years y1, y5, y10, y15, and y20. Preserve fp64 live reductions
at all 75 dates, including all 197 row transports. Missing or inferred dates,
float32 storage, differing output calendars, or a bridge stamp invalidate T1.

The deterministic controls compare both initialized states, reproduce the
archived from-rest y20/y40 ACC reducer values where their valid U history
permits, and report the unperturbed years-1–20 trajectories. Those controls do
not enter a floor.

### Frozen climate-statistic families

Import, without retyping, the six families and reducers frozen in
`PREREG_multi_year_climate_equivalence.md`:

1. additive full-section and channel ACC statistics;
2. south/channel/north basin transport and every one of the 197 row transports,
   with the `1e-9 Sv` basin-to-full closure plant;
3. south/channel/north MLD seasonal cycles from `mld_climate_audit.py`;
4. the registered basin/depth T/S water-mass census and sigma0 thresholds;
5. upper/deep density-contrast statistics; and
6. the registered mean-state and variability statistics.

The final climate window is the 60 monthly samples in years 16–20. Preserve
the exact final-five-year means, y11–y20 annual-endpoint slopes, twelve-month
climatologies, seasonal amplitudes, deseasonalized monthly sample standard
deviations, interannual standard deviations, quantile convention, masks, and
depth classes. A family aggregate cannot replace its 197 row verdicts.

### Horizon-matched floor and frozen bars

For each registered scalar statistic `s`, compute its six member values on
each side and then exactly:

`gap_s = mean(lego_s) - mean(nemo_s)`

`L_s,N_s = sample standard deviations across the six members`

`floor_s = sqrt(L_s**2 + N_s**2)` and `R_s = abs(gap_s) / floor_s`.

This floor belongs only to the standalone 20-year distribution, statistic,
window, and storage epoch. It is not transported from any archive listed
above. A side-spread imbalance above `10x` is stamped `ONE_SIDED` and both
spreads remain visible.

Reuse 20,000 deterministic nonparametric bootstrap draws, seed `1455`,
independently resampling six members with replacement on each side. Let
`R_lo/R_hi` be the 2.5/97.5 percentiles:

- `CONFIRM` when `R_hi <= 2.0`;
- `REFUTE` when `R_lo > 2.0`;
- `UNRESOLVED` otherwise.

The existing quantization gate also transfers verbatim: fewer than four
distinct member values on either side, a floor below ten times the measured
numerical/storage quantum, more than 1% zero/nonfinite bootstrap floors, or a
missing control yields `UNRESOLVED_QUANTIZED`. Each family uses the bootstrap
maximum `R`; all six families must confirm for
`STANDALONE_STATISTICALLY_INDISTINGUISHABLE_AT_20Y`.

### What is claimable, equilibration, and cost

The 20-year result is a **horizon-specific from-rest climate-distribution
claim**, not an equilibrium claim. NEMO's separate control is still climbing
at year 40 (year 20 `142.810 Sv`, year 40 `176.124 Sv`, years 21–40 LSQ slope
`+1.255 Sv yr-1`). Because T1 scores the same elapsed dates, sample calendar,
and spin-up history on both sides—and includes slope and variability as
statistics—the years-16–20 windows are statistically comparable even when
they are not stationary. A pass licenses “indistinguishable at the 20-year
from-rest horizon,” never “equilibrated” or “same attractor.”

If an equilibrium claim is required, preregister synchronized ten-year
extensions of all twelve members. At each extension retain monthly sampling
over the terminal five years and y(last-9)–y(last) slopes. A statistic is
`LEVEL_STATIONARY` only when, on both sides, the absolute ensemble-mean
ten-year slope times five is at most twice that side's six-member terminal
level spread and its endpoint floor is not increasing across the last two
five-year blocks. Until every level family passes this admission, only the
horizon-specific distribution and trend statements are claimable. The known
single-control y40 ramp is advance evidence that 40 years may still be
insufficient, not permission to weaken the gate.

Minimum cost is 120 new legoESM model-years and 120 new NEMO model-years.
Recorded throughput for the analogous held bridge battery budgets roughly
12–15 hours for legoESM in three two-GPU waves and 15–18 serial hours for NEMO,
before standalone-instrument and snapshot overhead. Record measured wall time
in SLOTs. Every equilibrium extension costs another 60 model-years per model.

T1 emits stability, every scalar and family verdict, the horizon-specific
capstone verdict, and a separately admitted stationarity statement. No T2/T3
result fills a missing T1 family.

## Frozen outcome matrix

| experiment | confirms | refutes | unresolved/invalid |
|---|---|---|---|
| T3 | zero config diffs and bit-identical five-day behavior | any admitted config row or behavior bit differs | faithful catalog entry absent, failed provenance, or missing artifact |
| T2 | unmodified round-94 scorer returns full confirmed battery | scorer returns a registered refutation/debt | partial band or failed admission |
| T1 | all six standalone 20-year families confirm; equilibrium only after its separate stationarity gate | instability or any registered family refutes | bootstrap overlap, quantization, missing member/control, or nonstationary equilibrium gate |

The model-level claim is conjunctive and enumerated, never rhetorical: it may
name only the construction paths, cards, start modes, horizons, metrics, and
bars actually confirmed by these experiments.
