# NEMO testcase fidelity receipt — lane 3b, SI3 Phase 6 DH snow carrier

Tracker: `climate-federation/legoESM#1699`

Recovery git: `/tmp/codex-si3thd-localgit`

Implementation/evidence commits through this receipt:

- `5d89a05971e` — initial DH-owner preregistration.
- `59e8aac6131` — corrected two-operation interaction preregistration after the
  initial remap-only hypothesis was refuted by operands.
- `875a62168de` — exact-entry 1-D thickness-bridge correction preregistration.
- `ad51e2593a3` — snow-carrier operation-order preregistration.
- `6cdad8a8f7eaef41435262e78b66553b935d7dc1` — production transcription,
  fail-closed reader, private arms, plants, and unit tests.
- `df346f0a5f8747630f4a18137f69fe29bda803d0` — complete 8,760-step evidence
  artifact.
- `f27d0bc72d8` — reversed-record plant and snow-thickness gradient coverage
  requested by final gate review.

## Verdict

**CONFIRMED for the kt5734 two-operation DH owner; DEBT for the full column.**
At kt5734, NEMO's capped thickness-space sublimation retains a
`2.117582368135751e-22 m` fourth snow segment while the old legoESM mass-space
loop produces four exact zeros.  NEMO then passes that carrier through its
unconditional `snw_ent` remap.  A binary64 replay matches both source stages at
zero ulp, and the combined private arm reduces `POST_DH.e_s` from
`777833.3635432672 J m-3` (normalised `1.0`) to
`5.413312464952469e-8 J m-3` (normalised `6.959475793495391e-14`), an
improvement factor of `1.4368898314649512e13`.

The proposed single causal chain from kt4242 to kt5734 is **REFUTED**.  The
combined snow-carrier/remap arm leaves kt4242 `POST_DH.h_i` bit-identical, so
the kt4242 `2.8315499258551526e-11` normalised ice-thickness injection is an
independent unresolved debt.  The annual gate remains **DEBT**, and this
receipt makes no whole-column fidelity claim.

## Rung 3.5 closure — user decision, 2026-09-04

**CLOSED AS MIXED DEBT.**  The retained debt rows are kt4242
`POST_DH.h_i`, an independent exact-entry injection of
`2.8315499258551526e-11`; positive-subnormal-snow rows led by kt5860
`POST_ZDF.e_s`, normalised `0.02510099530281747`; and year-end continuous
errors `t_su=4.654045553508542e-6`, `e_i=1.1466756156615379e-4`, and
`h_i=1.045384260731163e-4`.  **Phenomenology: MATCHED by the user's closure
decision** on the six measured rows; the retained measurement labels below
remain AT-FLOOR for five rows and UNMEASURED-floor for growth onset.  These
debts remain documented below; no further C1D-column work is authorized in
this lane.

## Rule 0: active NEMO source and executing legoESM

The shipped source was read before assigning the owner:

- `icethd.F90:140-163` converts the selected category to 1-D, calls ZDF, DH,
  temperature, salinity, temperature, and then converts back.  Its snow energy
  conversion uses `h_s*a_i > epsi20` at `:429-435` and multiplies the resulting
  volumetric enthalpy back by `h_s*a_i/nlay_s` at `:438-450`.
- `icethd_dh.F90:139-145,166-177` creates four source-ordered snow segments:
  precipitation followed by three old layers.  `:179-202` computes one capped
  `zdeltah`, walks segments from 0 upward, updates `h_s_1d`, and deliberately
  leaves the commented-out enthalpy reset inactive.
- `icethd_dh.F90:204-225` consumes available surface energy from snow first;
  `:231-315` then hands the remaining `zq_top` to ice surface melt and ice
  sublimation.  Thus the no-more-snow transition is a retained zero-thickness
  enthalpy carrier plus continued ice ablation, not a snow-enthalpy reset.
- Basal growth/melt is `icethd_dh.F90:321-424`; complete ice loss clears snow
  at `:426-439`; and basal-up snow-ice flooding is `:441-485`.
- The unconditional snow remap and snow-temperature reset are
  `icethd_dh.F90:494-507`.  `snw_ent` at `:535-613` forms cumulative
  thickness/content in loop order, sets the final cumulative content exactly,
  and divides nonnegative increments by `MAX(zhnew,epsi20)`.  There is no
  namelist or preprocessor switch around the selected operations.
- The premise that `ice_var_zapsmall` applies `epsi20` to snow volume is
  **CORRECTED**: `icevar.F90:404-416` uses `v_s > epsi20` only to reconstruct
  or reset snow temperature.  `ice_var_zapsmall` at `:601-706` instead clears
  snow energy/volume only when `MIN(a_i,v_i,h_i) < epsi10` (`:664-706`).

The executing legoESM lines in commit `6cdad8a8f7e` are
`bitz_lipscomb.py:660-709` for the cumulative `snw_ent` transcription,
`:725-812` for the source-ordered carrier and capped sublimation,
`:814-840` for snow-first surface melt, `:842-963` for ice surface/basal
growth and melt, and `:976-1024` for flooding, remaps, and bulk salinity.  The
two discriminator hooks are underscore-private and default on; neither is in
`IceConfig` or the column card, so the ORCA1-resolved identity cannot construct
a mixed public combination.

## Copy-only WRITE-only oracle discriminator

No shipped NEMO file or shipped configuration was changed.  The instrumented
source is a copy at:

`/data/abyssal/dbalwada/nemo-testcases-l3/nemo502_si3thd_phase6_src`

The accepted additive run is:

`/data/abyssal/dbalwada/nemo-testcases-l3/c1d_omip_l3_sasice_phase6_dh_operands_writeonly`

The first probe revision had initialized three local operands solely for early
records.  Adversarial review correctly rejected that as not literally
WRITE-only.  Those assignments were removed.  At stages 0 and 1, the stream
now writes explicit zero sentinels for not-yet-defined `zdeltah` and
`zevap_rema` without assigning model state; the registry labels them
unavailable.  `zq_top` is already defined, and `dh_snowice` is initialized by
the caller at `icethd.F90:143-150`.  Rebuild plus a new run root produced
byte-identical thermodynamics frames, ZDF inputs, `ocean.output`, and final ice
restart relative to the rejected probe, proving the removed assignments were
inert for the measured thermodynamics.  The exchange stream was **not**
byte-identical (`6eefecd0...` before versus `f02d1edd...` after), so no
exchange-stream identity is claimed.  The isolated year gate consumes the
separately registered, byte-identical ZDF input stream.

| artifact | SHA-256 |
|---|---|
| final copied `MY_SRC/icethd_dh.F90` | `fb4f9f22aee0ee15a6accd2172bbb16b79f8fd958b07aa684cd8c770bcb99a2d` |
| final copied `nemo.exe` | `34680b4587f587ad744027b63f2aab6e9a44f3ad2ab3a53d6ec61a737f3ee80d` |
| `oracle_si3_dh_operands.bin` | `9efbcb9113c2748f9085c519092848596daf0f573ba8a4625d02ee3c08bc8b14` |
| `oracle_si3_dh_remap_operands.bin` | `8fbeb7df70b3c66b4e7acdd7ab0df3ed8fc01bfaa6150dbc47e3b444638b40a5` |
| `oracle_si3_thd_frames.bin` | `7fc9df2707a85581075e3c69b26784155151a55640a5693eb32c34fb710ea49b` |
| `oracle_si3_zdf_inputs.bin` | `cd1b15c821f19442a840e99c067c640e5146b754fc137a2c81e88856d6ea7efd` |
| `oracle_si3_exchange_frames.bin` | `f02d1edd3fb9bf6487527e58c1bf77321eeca9116695c8025323c48ee9a865d6` |
| `ocean.output` | `3e47d39f061ca1f9fa111a46b01bb3d1dbcfa10be73422fced0abc9e4af25430` |
| `C1D_SASICE_00008760_restart_ice.nc` | `b61cb8443e14b3f0868ef125f621c0b1748aff7bc827dfab0d4044fc3f773b4e` |

The new streams contain only kt4242 and kt5734.  The main evidence JSON retains
all eight oracle/legoesm boundaries and full fields for kt4241--4243 and
kt5733--5735 (plus the earlier diagnostic snapshots).

### DH operand frame registry

| stage | time level | active source |
|---:|---|---|
| 0 `INITIALIZED` | after segment initialization, before snowfall; sublimation locals are unavailable sentinels | `icethd_dh.F90:127-145` |
| 1 `POST_PRECIP` | after snowfall, before sublimation; same unavailable sentinels | `:166-177` |
| 2 `POST_SUBLIMATION` | after sequential sublimation/deposition, before snow melt | `:179-202` |
| 3 `POST_SNOW_MELT` | after snow-first surface melt, before ice surface processing | `:204-225` |
| 4 `POST_ICE_SURFACE` | after ice surface melt/sublimation, before basal processing | `:231-315` |
| 5 `POST_BASAL` | after basal growth/melt, before complete-ice-loss handling | `:321-424` |
| 6 `POST_NO_ICE` | after complete-ice-loss handling, before flooding | `:426-439` |
| 7 `POST_FLOOD` | after basal-up flooding, before remaps | `:441-485` |
| 8 `SNW_ENT` | cumulative arrays and result inside the completed remap | `:535-613` |
| 9 `POST_SNW_ENT` | after `snw_ent`, before the snow-temperature inverse | `:494-507` |

The reader rejects magic/version/size/stage/order errors, duplicate remap
steps, and truncated payloads.

## First divergence, replay, scaling, and arm

At kt4242, snowfall (`1.483619325881591e-5`) creates
`1.4048954949214863e-4 m` snow and negative evaporation
(`-8.54658622864243e-7`) deposits another `9.32354861306447e-6 m`.  Surface
heat then removes all snow and continues into ice.  The final `h_i` injection
is unchanged by the snow-carrier/remap arm and remains **UNOWNED**.

At kt5734, positive snowfall creates `8.808589674726812e-6 m` above three
`6.069552195555419e-7 m` old layers.  Positive evaporation enters NEMO's
source-order sublimation.  The first differing internal boundary is
`POST_SUBLIMATION`: NEMO has `[0,0,0,2.117582368135751e-22] m`, while the old
legoesm loop has four zeros.  Later snow melt and flooding are inactive for
this carrier; `snw_ent` turns the residual into
`[777833.363543267, 777833.363543267, 777833.3635432672] J m-3`.

The scalar sublimation replay and the cumulative remap replay each agree with
both kt4242 and kt5734 operand dumps at **0 ulp**.  Neither single arm can move
the final kt5734 row: source-ordered segments followed by the legacy remap and
legacy all-zero segments followed by `snw_ent` both finish at zero.  The
preregistered combined interaction is therefore the measured owner.

| residual scale | NEMO/remap maximum `e_s` (J m-3) | legacy | absolute gap |
|---:|---:|---:|---:|
| 1 | `777833.3635432672` | 0 | `777833.3635432672` |
| 1/2 | `388916.6817716336` | 0 | `388916.6817716336` |
| 1/4 | `194458.3408858168` | 0 | `194458.3408858168` |

This linear scaling was measured before assigning the combined owner.  The
production fix transcribes operations for which NEMO exposes no switch; the
legacy alternatives remain private validation hooks only.

## Full-year exact-entry result

The strict-CPU run used `JAX_PLATFORMS=cpu`, `CUDA_VISIBLE_DEVICES=''`, and
`set_policy(PrecisionPolicy.fp64())`.  The artifact prints backend `cpu`,
oracle/legoesm dtypes `float64`, 8,760 exact-entry steps, 70,080 boundary
frames, and 621,960 field rows.  Its gate status is **DEBT**, so the CLI exits
1 by design.  `stderr` is empty.

The exact-entry thickness bridge uses unchanged registered 1-D `h_i/h_s` only
for independent probes.  It is validated bit-for-bit against DH initialization
at kt4242/5734 and is not injected into the continuous trajectory.  The old
global `v/a` quotient differed by one ulp at kt5734; correcting this harness
defect was necessary to test the actual NEMO DH operand.

| exact-entry statistic | Phase 5 before | Phase 6 after |
|---|---:|---:|
| over-bar field rows | 36,852 | 35,282 |
| first over-bar | kt5 `POST_DO.e_s`, `1.011825579870701e-15` | kt5 `POST_DH.e_s`, `1.21389354074819e-15` |
| first above `1e-12` | kt4242 `POST_DH.h_i`, `2.8315499258551526e-11` | unchanged |
| first above `1e-3` | kt5734 `POST_DH.e_s`, `1.0` | kt5842 `POST_ZDF.e_s`, `0.012837520586235438` |
| largest genuine row | kt5734 `POST_DH.e_s`, `1.0` | kt5860 `POST_ZDF.e_s`, `0.02510099530281747` |

The largest remaining genuine row is absolute
`2832427.961354971 J m-3` over NEMO denominator
`112841260.96135496 J m-3`.  It occurs when NEMO carries positive subnormal
snow volume (`5e-324 m`) into `ZDF_SNOW_PRESENT`; it is not a
denominator-one/NEMO-zero normalization artifact.  The after histogram has
35,282 genuine-relative rows across 2,906 steps and **zero** denominator-one,
NEMO-exactly-zero rows.  The JSON retains the complete per-step trajectory and
histograms by step, sub-call, and variable.

## Continuous growth before/after

Metric: per-step normalised L-infinity, Phase-5 production before versus the
Phase-6 production trajectory after.  Continuous initialization remains the
global oracle ENTRY state; no subsequent oracle value is injected.

| step | `t_su` before / after | `e_i` before / after | `h_i` before / after | `h_s` before / after |
|---:|---:|---:|---:|---:|
| 1 | `4.4501e-16 / 4.4501e-16` | `3.5558e-16 / 3.5558e-16` | `0 / 0` | `5.5511e-17 / 2.7756e-17` |
| 10 | `2.2611e-16 / 1.3567e-15` | `7.6074e-15 / 7.6074e-15` | `6.6513e-16 / 6.6513e-16` | `3.8858e-16 / 1.3878e-16` |
| 100 | `1.8449e-15 / 1.8449e-15` | `3.9660e-14 / 3.9660e-14` | `3.5127e-15 / 3.5127e-15` | `2.8588e-15 / 1.4155e-15` |
| 1000 | `1.6713e-14 / 9.5669e-15` | `1.4516e-13 / 1.0596e-13` | `1.2740e-13 / 9.0038e-14` | `3.0309e-14 / 1.4100e-14` |
| 3000 | `1.8244e-14 / 1.1945e-14` | `3.0119e-13 / 2.5775e-13` | `2.7116e-13 / 2.4355e-13` | `1.1408e-13 / 6.4504e-14` |
| 5000 | `0 / 0` | `9.6606e-7 / 3.5095e-7` | `6.6909e-7 / 1.3598e-7` | `0 / 0` |
| 8760 | `6.6334e-6 / 4.6540e-6` | `1.6342e-4 / 1.1467e-4` | `1.4899e-4 / 1.0454e-4` | `6.3172e-14 / 3.0309e-14` |

## Six phenomenology rows

| quantity | NEMO | Phase 5 before | Phase 6 after | after distance | fp32-fp64 floor / status |
|---|---|---|---|---:|---|
| minimum thickness | `0.5550952376958682 m` | `0.5551788290706163 m` | `0.555144003761985 m` | `4.876606611681211e-5 m` | `0.11110316885826799 m`, AT-FLOOR |
| maximum thickness | `2.445688622638321 m` | `2.445688622637648 m` | `2.445688622637688 m` | `6.328271240363392e-13 m` | `1.0480229467058066e-4 m`, AT-FLOOR |
| minimum date | `2018-09-09T00:00Z` | same | same | `0 h` | `2304 h`, AT-FLOOR |
| maximum date | `2018-05-13T00:00Z` | same | same | `0 h` | `0 h`, AT-FLOOR |
| melt onset | `2018-05-14` (day 133) | same | same | `0 d` | `0 d`, AT-FLOOR |
| growth onset | `2018-09-09` (day 251) | same | same | `0 d` | fp32 has no onset, UNMEASURED |

The minimum-thickness distance moves toward NEMO from `8.35913747481e-5 m`
to `4.87660661168e-5 m`.  Labels compare only against legoESM fp32-fp64
spread; this rung does not provide a NEMO scheme spread.

## Terminal residual classification

**MIXED DEBT, not threshold-noise-only.**  The kt4239 and kt5495 continuous
jumps occur where the same-step exact-entry maximum is at most `2e-15`, so
those two examples are threshold amplification of floating-point noise.
However, kt4242 is already `2.83e-11`, kt4943 injects
`2.59e-8`, and positive-subnormal-snow ZDF rows exceed `1e-3` from kt5842.
Therefore the user's conditional terminal classification is not satisfied.

Going further requires two separate discriminators: a written-order DH
ice-thickness replay at kt4242, and a ZDF operand experiment for NEMO-positive
subnormal snow versus JAX/XLA arithmetic from kt5842.  Bit-exact reproduction
of NEMO's whole-step summation order would be needed to remove only the
threshold-amplified component.

## Tests and planted controls

- Focused Phase-2/Phase-2b plus BL99 tests: `37 passed in 62.81s`.
  This includes JIT and finite-gradient checks through the new remap at
  ordinary, `2e-22 m`, and zero snow.
- Post-review remap reader/thickness-gradient checks: `2 passed in 2.28s`.
- Constants ratchet on every touched production/gate/test file:
  `5 passed, 3383 deselected in 0.64s`.
- The repository-wide constants ratchet was also run: `3418 passed, 2 skipped,
  5 failed`; all five failures are pre-existing untouched atmosphere/grid/DINO
  files, and no touched file failed.
- `--plant-dh-owner` exits 1 at the 100-fold discriminator.
- `--plant-entry-bridge` exits 1 at the bit-exact thickness check.
- Unit plants reject bad DH magic, duplicate/out-of-order remap records, and a
  truncated remap payload.  Existing operator-trajectory, outlier,
  branch-census, scope-accounting, and stream-truncation plants remain red.
- Independent source and gate reviews initially returned HOLD on the stale
  JSON, non-WRITE-only probe initialization, continuous-bridge contamination,
  remap reader ordering, and missing gradient check.  Every item was corrected
  before the final year artifact.  Source re-review's last factual exchange-
  hash correction was also applied; final gate re-review returned **SHIP** and
  found the production source order consistent and the owner plant non-vacuous.

The committed evidence artifact is
`nemo_testcases_l3thd_phase6_year_gate.json`, SHA-256
`9571996d72875a3c312fb5b84170d5383bedc7d41fff8ebd9a75f38d3b2a0f9f`.
Evidence path: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/evidence/si3thd/nemo_testcases_l3thd_phase6_year_gate.json`, SHA-256 `9571996d72875a3c312fb5b84170d5383bedc7d41fff8ebd9a75f38d3b2a0f9f` (see `SHA256SUMS`).

## Forcing and coupled-rung debt

The unchanged official NEMO sette_inputs r5.0.0 archive URL is
`https://gws-access.jasmin.ac.uk/public/nemo-vol1/sette_inputs/r5.0.0/C1D_v5.0.0.tar.gz`.
The local archive MD5 is `9456e6a0a84d40630ad1804fd4061caf` and SHA-256 is
`54a2ceefd9126e180676e68eaa28ded85cc3b93ea3f0dda0fb964a035a4fc382`.
The ERA5 member actually consumed by the selected case,
`ERA5_NorthGreenland_surface_84N_-36E_1h_y2018.nc`, is SHA-256
`e5ec49445d2569019c45dec24255b9c7daf050079444b0e6e6d86a5b82317afe`.

**DEBT — coupled rung:** this isolated column consumes NEMO's dumped entry
`qns_ice` and `dqns_ice`.  NEMO's `sbcblk` bulk-flux computation is **NOT
CERTIFIED** by this rung.

## Choice register

- ASKED — read and quote the active NEMO melt-season snow, flooding, remap,
  conversion, and correction branches and the executing legoESM lines.
- ASKED — retain both-model full boundary state at kt4241--4243 and
  kt5733--5735; add copy-only, WRITE-only internal DH operands where needed.
- ASKED — measure scaling before ownership, preregister corrections, use
  underscore-private one-variable/interaction arms, and keep the no-switch
  source identity unbranched in production.
- ASKED — rerun the entire strict-CPU/fp64 year, report exact-entry counts,
  first/largest rows, the seven-step growth table, and all six phenomenology
  rows before/after.
- ASKED — commit complete per-step evidence with explicit pathspecs, use the
  recovery git, create the requested bundle, print hashes/session ID, and do
  not push.
- UNASKED — public thermodynamic selectors, alternative ice identities,
  forcing/timestep/bar changes, mid-trajectory oracle injection, GPU/MPI,
  coupled `sbcblk` certification, or deletion/modification of shipped NEMO,
  shipped configuration, or earlier run roots.

## FLAGGED FOR FUTURE DELETION

None.  The rejected additive probe root
`c1d_omip_l3_sasice_phase6_dh_operands` is retained and clearly non-canonical;
no prior run root or shipped artifact was deleted.
