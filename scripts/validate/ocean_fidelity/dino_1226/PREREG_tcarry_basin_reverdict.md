# PRE-REGISTRATION — corrected T carry and the southern-basin deficit

Written after the five-day Round-4 flicker arms and before either basin arm
runs. The five-day result is not used as a basin outcome. This registration
asks whether the now-confirmed bridge representation defect contributes
materially to the recorded southern-basin transport deficit.

## 2026-08-28 post-verdict default decision

The human selected reconstructed T carry as the harness default after this
measurement completed.  The frozen historical invocations below remain exact
records of what ran when T reconstruction was opt-in; they must not be read as
current-default examples.  To reproduce any legacy U-as-T arm now, add
`--bridge-before-stress-legacy-u-as-t`.  Explicit
`--bridge-before-stress-tpoint` remains accepted but is redundant on the new
default.  No metric, artifact interpretation, floor, or verdict changes.

## 2026-08-28 floor-seed binding amendment before score

All three registered seed arms completed, but no seed NPZ has been opened and
no member basin value, spread, RSS floor, corrected-T endpoint, or T-carry
classification has been computed or seen before this amendment.  Bind clean
producer `820e3500bf3317c1cc19ce61484497391e4b1f6b` and these receipts:

| member | artifact SHA-256 | log SHA-256 |
|---|---|---|
| seed 1 | `67f04b2e04a7e7d13f4e9e5cfa4f4ab128dcdf17ccd0eed27bb554d2efd694ee` | `11b96452254544fb9c882590526ec133a93121c26ba3ec62c70852fb5abf6138` |
| seed 2 | `7420d121f8a3f0e631f563ac869daff9d242e1a8ce14de62e4d35e41f182e595` | `f628c4b44cbaed686df1ccab2f619bcee359fd56a70d4318ace1e30d1b887982` |
| seed 3 | `c4688539ee33042a1dc41fbff9605fd43bd30b0a880e8327d6ff1e010d046fa9` | `47838be749fb9e572277c77307a9b6327386b3c25d11d73d9b3f142e41ce884c` |

Hashing all six files and reading only the logs before this commit confirmed
clean producer, fp64/both/bridged/15552000 s, default NEMO bridge Omega,
resolved NEMO EEN weighting, legacy U-as-T carry, 2880 steps, and
`STABLE=True` for every seed.

The earlier handoff text named the retained `d6dc89e91...` Stage-1 control as
the same-producer target, while the executor ran the exact registered commands
at the later instrument commit above.  Resolve that receipt difference without
transferring a response: use the already hash-bound unperturbed NEMO-Omega arm
`tcarry_omega90_nemo.npz` from producer `9e339ad1b...` as member zero.
`git diff --stat 9e339ad1b..820e3500b -- packages/ src/
scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py` is empty, so these
four artifacts share the same model and harness.  The exact legoESM member set
is therefore control plus seeds 1/2/3; no Stage-1 corrected artifact enters
the floor.

No metric, `ddof=1` convention, NEMO member set, RSS formula, baseline,
classification threshold, compensation rule, or decision order changes in
this binding amendment.  Commit the floor instrument and this binding before
opening a seed artifact.

### 2026-08-28 current floor measured and registered

The committed floor scorer ran at clean commit
`f873a7cc60a008723ddf16964831211985ec19ff`, scorer SHA-256
`8ebd99862285fd28abe1ed97f2146388b311504cdbb5584ebcc38821af24e684`.
It measured the exact legoESM absolute-transport member set

`[8.400753233279739, 8.400398404915238, 8.400493373055788,
8.400491317473405] Sv`

and sample standard deviation
`s_lego = 0.0001526669333485373 Sv` (`ddof=1`).  It reloaded the unchanged
NEMO members `RUN_VERDICT360_M0..M3` through
`basin_seasonal_decomp.nemo_state`, the same four-member side used by the old
floor, rather than copying or assuming that side.  Their absolute transports
are

`[8.839838743271773, 8.839838714018859, 8.83983874992388,
8.839838749808038] Sv`

and `s_nemo = 1.7109343555843327e-08 Sv` (`ddof=1`).  Therefore register

`F90_current = hypot(s_lego, s_nemo) = 0.00015266693430725714 Sv`.

This is the same two-sided convention as the old floor: each model's own
four-member sample standard deviation, RSS-combined.  The NEMO side is
unchanged and was directly re-derived; it contributes negligibly but is not
dropped or replaced by `sqrt(2)`.  All four legoESM members passed the
acceptance gate `PASS 5 | FAIL 0` at 5x.  Output
`/tmp/tcarry_basin_floor90.json` has SHA-256
`5d5ba993775b0db381b2d0afa7236ef349749d15ef2eac8427a1473381aedf7d`.

Before the paired score, bind the retained Stage-1 artifacts themselves:
legacy SHA-256
`ae114e7c71f530da253083e4f07f83e94bb66ed1d89c06c27909da9b36fc1e37`
and corrected-T SHA-256
`2f2e22fe3ca48bf9f923eccd412295a96b0d84b5781103f5f8fe71117532702e`.
No corrected-T basin endpoint or verdict was computed before this floor and
artifact-binding amendment.  The paired scorer now carries the new baseline,
new floor, floor-output hash, and both retained artifact hashes; its frozen
classifier and compensation rules are unchanged.

### 2026-08-28 final paired score — UNRESOLVED/FLOOR

The hash-bound paired scorer ran at clean commit
`e22db0c387bacd1a0ff7954bf0a69c7e3f88dc00`, scorer SHA-256
`4e1b7c3ae699b3d0ea43a21226be6f1af1dad50644d855db0edfc18fea8afa7e`.
The floor receipt, retained artifact hashes, producer/config/stagger receipts,
day-0 bit identity, independent basin reducers, and every planted control
passed.  The registered basin outputs are:

| quantity | legacy U-as-T | corrected T | corrected − legacy |
|---|---:|---:|---:|
| `Gbasin90` [Sv] | `-0.4390855099920348` | `-0.43905611603502415` | `+0.00002939395701062608` |
| `G4` rows 1..4 [Sv] | `-0.2971375104352858` | `-0.29719818378611107` | `-0.00006067335082526881` |

`R90 = Delta90 / abs(Glegacy90) = 0.00006694358238138922`
(`0.006694358238138922%`).  But the frozen classifier applies the floor branch
first: `abs(Delta90) = 0.00002939395701062608 Sv` is only
`0.09626824938878994` of
`2F90_current = 0.00030533386861451427 Sv`.  The mechanical verdict is
therefore **UNRESOLVED/FLOOR**.  The otherwise sub-2% ratio cannot be promoted
to REFUTED because the registered decision order makes that branch
unreachable inside `2F`.

The complete row battery is:

| row `j` | legacy [Sv] | corrected [Sv] | delta [Sv] |
|---:|---:|---:|---:|
| 0 | `0` | `0` | `0` |
| 1 | `-0.07088386596773866` | `-0.0708998081403559` | `-0.000015942172617244843` |
| 2 | `-0.08117908510875305` | `-0.08119610933637306` | `-0.000017024227620010546` |
| 3 | `-0.08810307286198249` | `-0.08811839383697104` | `-0.000015320974988547453` |
| 4 | `-0.05697148649681161` | `-0.0569838724724111` | `-0.000012385975599493726` |
| 5 | `-0.05266128077617971` | `-0.05267467673638504` | `-0.000013395960205331292` |
| 6 | `-0.03831368521133194` | `-0.038323682001471404` | `-0.000009996790139465972` |
| 7 | `-0.027969468239574802` | `-0.02796631418510942` | `+0.0000031540544653818614` |
| 8 | `-0.02057923932879091` | `-0.020570519785424013` | `+0.000008719543366897398` |
| 9 | `-0.012030726034930161` | `-0.012022903939897356` | `+0.00000782209503280562` |
| 10 | `-0.002109427890088389` | `-0.002097883789627164` | `+0.000011544100461224893` |
| 11 | `+0.002730282784913385` | `+0.0027432250296582916` | `+0.000012942244744906795` |
| 12 | `+0.004570590067720315` | `+0.004573887439333624` | `+0.000003297371613308897` |
| 13 | `+0.004414955071513571` | `+0.004480935720008183` | `+0.00006598064849461238` |

Every compensation control is safe:

| acceptance metric | corrected-minus-legacy absolute-gap change | allowed |
|---|---:|---:|
| ACC | `+0.0005189000472256566` | `0.091` |
| upper contrast | `-1.3781865026984974e-07` | `0.00011` |
| deep contrast | `-7.306341100361824e-08` | `0.000045` |
| surface sigma max | `-5.068818609288428e-07` | `0.000095` |
| surface sigma mean | `-3.194503950254557e-08` | `0.000095` |

Both arms are inside all five 5x acceptance gates.  Output
`/tmp/tcarry_basin_reverdict_current.json` has SHA-256
`a4ddaf7e3c5862ac6c661ca447e72476a085f929c16ade1ba5bedd07b0bb8f89`.
The corrected bridge carry owns most wall flicker, but its day-90 southern-
basin transport response is below the measured two-sided floor; this lane
does not establish basin-deficit ownership or non-ownership.

## 2026-08-28 baseline re-registration after epoch ownership

The hash-bound EEN-by-bridge-Omega fourth corner measured
`-0.42606954856856305 Sv`, within `0.0002846699870264757 Sv` of the frozen
historical endpoint and therefore inside the unchanged
`0.0002899800477248501 Sv` ownership band.  The registered combined-ownership
branch fired, with interaction
`I = -0.0840808093571308 Sv` (**NON_ADDITIVE**).  Scorer SHA-256 is
`c82a4e9d87ea3fe89f0f066952da624ef1811e7f6d7f846101e2ea453481303e`;
output SHA-256 is
`846706fadaf1ef12623f243049901a5d52bff65abde4b6845042b73c25fb80dc`.

Re-register `Glegacy90 = -0.43908550999203477 Sv`.  The scorer now carries
that exact baseline but sets day-90 `FLOOR` to non-finite/UNMEASURED, so it
fails closed before loading the corrected endpoint.  The historical
`0.00014499002386242506 Sv` floor is not transferred.

Exactly three GPU integrations remain before the T-carry basin verdict can be
scored: the already-registered seed 1, seed 2, and seed 3 arms below, at the
same `d6dc89e91...` producer as the retained unperturbed legacy control and
corrected-T arm.  No new control or corrected-T integration is required while
those retained hash-bound artifacts remain available.  After binding the
three seed receipts, one CPU floor reduction must compute `F90_current` from
the control-plus-seeds legoESM sample standard deviation RSS-combined with the
unchanged four-member NEMO sample standard deviation.  Then the existing
retained legacy/corrected pair must be re-scored once on CPU with the new
baseline and floor.  No corrected-arm basin number has yet been adopted.

## 2026-08-28 EEN-off artifact binding before score

The registered current-trajectory EEN-off arm has completed, but neither its
NPZ payload nor any basin endpoint has been loaded or scored before this
binding amendment.  Bind:

- artifact `results/dino_1455/tcarry_basin90_legacy_een_off.npz`, SHA-256
  `a7f3bf5555ec1f6a7ed58792ede8b9f10188c93c21ff81f0553b2fb3cc400dd4`;
- log `results/dino_1455/tcarry_basin90_legacy_een_off.log`, SHA-256
  `465c01c104dfb8e8f82dc5f6bd0ae70eb50d009ec125457ac259a5f0168e0d02`;
- clean producer
  `6c64f261aa33b43c372b81b30d4569481116766d`.

The scorer is
`scripts/validate/ocean_fidelity/dino_1226/tcarry_een_off_discriminator.py`.
It must hash both inputs before loading, require the registered EEN-off log
banner and legacy-carry/grid/precision/clock/stability receipts, and verify
that `git diff --stat d6dc89e91..6c64f261a -- packages/ src/` is empty.  It
then loads member-zero NEMO day 90 and computes exactly one number,
`Goff_current90`, with the existing `tcarry_basin_reverdict._reduce` basin
functional.  That reducer must print agreement between its row sum and
`acc_driver_decomp.rowset()["g_south"]` to `1e-12 Sv`.  The existing dry/wet
reducer plants and the five-metric 5x acceptance gate must pass.

The frozen decision is unchanged:

- **CONFIRM full EEN ownership** iff
  `|Goff_current90 - (-0.4257848785815366)| <= 0.0002899800477248501 Sv`;
- **REFUTE full EEN ownership** otherwise.

The classifier self-test must prove both states are reachable.  On CONFIRM,
re-register `Glegacy90 = -0.43908550999203477 Sv` with the EEN ownership
receipt, but do not transfer the historical floor: the already-registered
current-SHA control plus seeds 1/2/3 still must measure `F90_current`.  On
REFUTE, leave the STOP active and preregister the old/new bridge-Omega arm.
No threshold, floor, metric, or classifier changed in this binding amendment.

### 2026-08-28 scored result — REFUTED and STOP

The committed hash-bound scorer at clean commit `d49747097054946e50553862fbdf9d90b68a5cc6`
(SHA-256 `a9cb7ce96266ab73ee5ec2eea32c4484a5ea287338a5b4e86bb948e9b616df2a`)
measured:

`Goff_current90 = -0.3484425774226274 Sv`.

Its distance from the frozen historical baseline is
`0.07734230115890917 Sv`, 266.71 times the registered
`0.0002899800477248501 Sv` band.  Therefore full EEN ownership is
**REFUTED**.  The row-transport and `g_south` implementations agree within
`1.7763568394002505e-15 Sv` on each absolute endpoint, every reducer plant
passed, and the acceptance gate certified `PASS 5 | FAIL 0` at 5x.

Output `/tmp/tcarry_een_off_discriminator.json` has SHA-256
`35332f72823b76127972d76b039e039fd5c8de36db71ae42b64349cc89f7b1a1`.
This result does **not** re-register `BASELINE[90]` or `FLOOR[90]`; the STOP
remains active.  The seed floor arms below remain registered but must wait
until the baseline epoch is owned.  The next discriminator is preregistered
in `PREREG_tcarry_bridge_omega_discriminator.md`.

## 2026-08-27 Rule-1e reconciliation amendment after Stage-1 STOP #2

The amended scorer correctly stopped after measuring the retained legacy arm:
`G_basin90 = -0.43908550999203477 Sv`, a `-0.01330063141049817 Sv` miss from
the registered historical baseline, 45.87 times its `2F` reproduction band.
It did not load or score the corrected-T endpoint.  Therefore no T-carry basin
delta or verdict exists yet.

The historical JSON's analysis HEAD is `1d68fb289d6457e74ced8a1c71ab7eaceb8a29b1`.
Its underlying four legoESM artifacts were produced cleanly at
`a7b940f75c04d824b478de4e1728220e3a71989e`, not at the analysis HEAD.  The
resolved model protocol is recorded in
`docs/ocean/fidelity/dino_tcarry_baseline_reconciliation.md`.

The committed offline reconciliation probe, preregistered before it scored
the historical EEN artifacts, measured the day-90 epoch shift as:

`-0.013300631410499975 = -0.006565231376747249 (EEN) + -0.006735400033752725 (remainder) Sv`.

EEN is PARTIAL at 49.36% under the registered 80–120% dominant-owner band.
The remainder is 1002 times the direct rotation-rate scale and changes sign
over days 30/60/90, so direct linear Omega ownership is REFUTED and nonlinear
Omega ownership remains UNRESOLVED.  Output SHA-256:
`c9a8306d086648a1bdb82e303040462b15371dafcdf9a8752106f7c6830f9842`.

**No baseline or floor is re-registered in this amendment.**  `BASELINE[90]`
and `FLOOR[90]` in the scorer remain historical receipts that deliberately
keep the STOP active.  Adopting the new legacy number now would violate Rule
1e rather than resolve it.

### Registered next GPU arm — current-trajectory EEN discriminator

Run at the retained arms' exact clean producer
`d6dc89e91c9ae6b07d146991d2cb6c850f261bb0`.  The existing arm selector is
reused; no model edit or new selector is required.  It keeps the legacy
U-as-T carry and differs from the retained legacy artifact only by
`een_metric_weighting: nemo -> off`:

```sh
DINO_EEN_METRIC=off CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 \
  LEGOESM_NEMO_E3T=both \
  .venv/bin/python \
  scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py \
  nemo_dino_kamm_mlf results/dino_1455/tcarry_basin90_legacy_een_off.npz \
  --days 90 --bridge-before --save-3d --no-surface-stress-implicit
```

The log must stamp producer `d6dc89e91...`, dirty tracked count zero,
`ARM: een_metric_weighting=off`, `U_AS_T_LEGACY`, fp64, `both`, bridged,
15552000 s, and `STABLE=True`.  The acceptance gate must still certify 5/5.
Before scoring, amend the reconciliation probe with the returned artifact and
log SHA-256s; do not score an unbound file.

Let `Goff_current90` be its same-functional basin gap.

- **CONFIRM EEN owns the full epoch mismatch at the current trajectory** iff
  `|Goff_current90 - (-0.4257848785815366)| <= 0.0002899800477248501 Sv`.
  Equivalently, the current same-SHA EEN response must reproduce the observed
  `-0.01330063141049817 Sv` epoch shift within that same band.
- **REFUTE full ownership** iff it misses that historical baseline by more
  than the band.  The already-measured PARTIAL historical response still
  stands; the remaining discriminator is then a current-SHA old/new bridge
  Omega arm, which requires a fail-closed selector and a separate prereg before
  implementation.

Only after this arm owns the baseline epoch may `Glegacy90` be re-registered.
Threshold fractions (10% CONFIRM, 2% REFUTE), compensation rules, metric
definition, and classification order remain unchanged.

### Registered floor remeasurement — required after baseline ownership

The old four-member floor predates both live changes and is not plausibly
SHA-stable: its own same-functional spread grows by more than three orders
between days 30 and 90.  Retain the existing unperturbed current legacy arm and
produce seeds 1, 2, and 3 at the same `d6dc89e91...` producer, default EEN on,
legacy U-as-T carry:

```sh
CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both \
  .venv/bin/python scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py \
  nemo_dino_kamm_mlf results/dino_1455/tcarry_floor90_seed1.npz \
  --days 90 --bridge-before --save-3d --no-surface-stress-implicit --perturb-seed 1

CUDA_VISIBLE_DEVICES=1 JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both \
  .venv/bin/python scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py \
  nemo_dino_kamm_mlf results/dino_1455/tcarry_floor90_seed2.npz \
  --days 90 --bridge-before --save-3d --no-surface-stress-implicit --perturb-seed 2

CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both \
  .venv/bin/python scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py \
  nemo_dino_kamm_mlf results/dino_1455/tcarry_floor90_seed3.npz \
  --days 90 --bridge-before --save-3d --no-surface-stress-implicit --perturb-seed 3
```

After their hashes are bound, compute the sample standard deviation of the
same `G_basin90` functional across control+seeds and combine it by RSS with the
unchanged hash-bound four-member NEMO standard deviation.  That measured value
becomes `F90_current`; no transfer, `sqrt(2)` shortcut, or old legoESM spread
is permitted.  Until it exists, the corrected-T deficit classification stays
STOPPED even if the baseline epoch is owned.

## 2026-08-27 instrument-only amendment after Stage-1 STOP

The executor completed both registered 90-day arms cleanly and both separate
acceptance gates certified `PASS 5 | FAIL 0`. The paired basin scorer then
crashed in its **first planted receipt control**, before loading/scoring either
day-90 endpoint: its `_Swap` NPZ overlay implemented keyed access but not
membership, so `certifiable_grid_and_precision` triggered Python's integer
iteration fallback and the real archive raised `KeyError('0 is not a file in
the archive')`. **No `Gbasin90`, row profile, delta, ratio, or basin verdict was
produced or seen before this amendment.**

Amended scorer SHA-256:
`f09629792aed293bdea0b7b7004617f45c2390a9865cb393550031e51560a9d5`.
The overlay now supplies `.files`, `__contains__`, keyed access, iteration,
length, `.keys()`, and `.get()`. Its self-test opens two synthetic NPZ files on
disk and sends every receipt plant through the same overlay/consumer path.
The retained Stage-1 arms remain bound to their exact producing commit
`d6dc89e91c9ae6b07d146991d2cb6c850f261bb0`; the amended scorer prints its
own distinct HEAD and accepts only that registered producer for day 90. Future
conditional day-360 arms remain bound to the scorer HEAD that produces them.
**Diff scope:** mapping protocol, test coverage, and this exact retained-arm
producer receipt only. The basin metric, reducers, baselines, floors,
thresholds, classification order, acceptance safety rule, arm contents, and
GPU commands are unchanged.

## Reuse audit and exact metric

The search terms were `acc_driver_decomp`, `group_transport`, `g_south`,
`row_transport`, `Gbasin`, `southern basin`, `floor90`, and `RECORDED_FLOOR`.
The campaign already owns both required reductions:

1. `acc_driver_decomp.rowset()["g_south"]`: `group_transport` over
   `LAT_GROUPS[0] = slice(0,A.J0)`, reference `e3t_1d`, `A.e2u_col`, and the
   additive mean over longitudes `2:-2`.
2. `basin_seasonal_decomp.row_transport`, consumed without re-derivation by
   `wall_visc_ablation_gap.measure()["Gbasin"]`: per-T-row transport on the
   same `e3t_1d`/`A.e2u_col`/mean reduction, summed over rows `0..13`.

The primary metric is signed

`Gbasin90 = sum_j=0..13(row_transport(lego,j)-row_transport(NEMO,j)) [Sv]`.

The two implementations must agree within `1e-12 Sv`; otherwise STOP. The
complete rows `0..13` profile and `G4` (rows `1..4`) are mandatory companions,
not alternative verdicts. No median reduction may be substituted: the mean is
what makes the latitude partition additive.

The recorded uncorrected day-90 baseline is exactly
`Glegacy90 = -0.4257848785815366 Sv` (`basin_seasonal_decomp.py`'s retained
series `/tmp/dino_basin_seasonal_decomp.json`, SHA-256
`63d4e60dd68281bc6101a35f86cb3f4406cb2ddbc27848b74473876226e85849`;
the committed control is `RECORDED_GAP[90] = -0.42578`). The measured two-sided
day-90 floor for this same southern-basin functional is exactly
`F90 = 0.00014499002386242506 Sv`, the RSS of the four-member legoESM and NEMO
sample spreads from `/tmp/dino_verdict360/m{0..3}_*.npz` and
`RUN_VERDICT360_M{0..3}`. This is not the acceptance gate's transferred
`0.091 Sv` tolerance.

## Stage 1 — controlled 90-day pair

Both arms use the standard `nemo_dino_kamm_mlf` twin, the same day-180 NEMO
restart, bridged before level, fp64, NEMO ladders, explicit surface stress, and
3-D snapshots. The sole selected variable is the initial before-stress carry:
legacy U-as-T versus reconstructed analytic T. At execution time the legacy
arm omitted the then-opt-in selector and stamped `U_AS_T_LEGACY`; under the
current default it instead requires `--bridge-before-stress-legacy-u-as-t`.
The corrected arm explicitly selected T and stamped `T`, `15552000.0`, and the
registered content hash.

```sh
CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both \
  .venv/bin/python scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py \
  nemo_dino_kamm_mlf results/dino_1455/tcarry_basin90_legacy.npz \
  --days 90 --bridge-before --save-3d --no-surface-stress-implicit

CUDA_VISIBLE_DEVICES=1 JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both \
  .venv/bin/python scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py \
  nemo_dino_kamm_mlf results/dino_1455/tcarry_basin90_corrected.npz \
  --days 90 --bridge-before --bridge-before-stress-tpoint --save-3d \
  --no-surface-stress-implicit
```

Exact gate and paired-scorer invocations:

```sh
.venv/bin/python scripts/validate/ocean_fidelity/dino_1226/acceptance_gate_90d.py \
  results/dino_1455/tcarry_basin90_legacy.npz --level 5
.venv/bin/python scripts/validate/ocean_fidelity/dino_1226/acceptance_gate_90d.py \
  results/dino_1455/tcarry_basin90_corrected.npz --level 5

CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  .venv/bin/python \
  scripts/validate/ocean_fidelity/dino_1226/tcarry_basin_reverdict.py \
  results/dino_1455/tcarry_basin90_legacy.npz \
  results/dino_1455/tcarry_basin90_corrected.npz \
  --day 90 --out results/dino_1455/tcarry_basin90_reverdict.json
```

Before classification, both acceptance-gate artifacts must certify the claim's
grid/precision/clock, the legacy arm must reproduce `Glegacy90` within
`2*F90 = 0.0002899800477248501 Sv`, and the two arms must have bit-identical
day-0 prognostic fields and all stamps except the registered carry identity.
The scorer's dry/wet planted controls must pass.

Define `Delta90 = Gcorrected90 - Glegacy90` (positive reduces the negative
deficit) and `R90 = Delta90/abs(Glegacy90)`.

- **UNRESOLVED/FLOOR** has first priority iff `|Delta90| <= 2*F90`; it cannot
  also be REFUTE.
- Otherwise **CONFIRMS material deficit contribution** iff `R90 >= 0.10`
  (`Delta90 >= 0.04257848785815366 Sv`), `|Delta90| > 2*F90`, and no
  acceptance-gate metric's absolute NEMO gap degrades by more than its exact
  committed `1x acceptance_gate_90d.FLOORS` value relative to legacy.
- Otherwise **REFUTES material deficit contribution** iff `R90 <= 0.02`
  (`Delta90 <= 0.008515697571630731 Sv`). A negative material change is
  reported additionally as a material compensator, not silently folded into
  “no effect”.
- `0.02 < R90 < 0.10`, or a transport gain accompanied by a gate regression,
  is **UNRESOLVED**.

The acceptance result and all five metric deltas are printed in the same table
as `Gbasin90`, `G4`, and rows `0..13`; transport alone cannot earn a clean fix
verdict through compensation.

## Stage 2 — registered one-year extension, conditional

The campaign's prize is recorded at one year, so a 360-day pair is registered
now but is authorized **only if Stage 1 CONFIRMS**. REFUTE or UNRESOLVED at day
90 is a STOP. Both one-year arms retain every ten-day snapshot so the recorded
endpoint and final-90-day window can be scored without interpolation.

```sh
CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both \
  .venv/bin/python scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py \
  nemo_dino_kamm_mlf results/dino_1455/tcarry_basin360_legacy.npz \
  --days 360 --bridge-before --save-3d --no-surface-stress-implicit \
  --snap-days 0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350,360

CUDA_VISIBLE_DEVICES=1 JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both \
  .venv/bin/python scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py \
  nemo_dino_kamm_mlf results/dino_1455/tcarry_basin360_corrected.npz \
  --days 360 --bridge-before --bridge-before-stress-tpoint --save-3d \
  --no-surface-stress-implicit \
  --snap-days 0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350,360
```

The endpoint uses the same `g_south`/row metric at day 360. The exact recorded
legacy baseline is `-0.9519122331848315 Sv`; the directly measured two-sided
floor is `F360 = 0.06173656216045926 Sv`, from the same retained four-member
artifacts and series. The legacy endpoint must reproduce within `2*F360 =
0.12347312432091852 Sv` or STOP. Define `Delta360` and `R360` as above. The
exact 10% threshold is `0.09519122331848316 Sv`; the exact 2% threshold is
`0.01903824466369663 Sv`. Classification order is the same and mutually
exclusive: `|Delta360| <= 2*F360` is **UNRESOLVED/FLOOR**; otherwise CONFIRM
requires `R360 >= 0.10`; otherwise REFUTE requires `R360 <= 0.02`; everything
between is UNRESOLVED. At both endpoints a compensation safety failure means
UNRESOLVED: for any of the five `acceptance_gate_90d.metrics` quantities, the
corrected absolute NEMO gap exceeds the legacy absolute gap by more than that
metric's committed `acceptance_gate_90d.FLOORS` value. The final-90-day mean over
days `280..360` is descriptive unless a same-functional ensemble floor is
computed from the retained verdict members; the endpoint alone controls this
registered attribution verdict.

Exact conditional day-360 scorer invocation:

```sh
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  .venv/bin/python \
  scripts/validate/ocean_fidelity/dino_1226/tcarry_basin_reverdict.py \
  results/dino_1455/tcarry_basin360_legacy.npz \
  results/dino_1455/tcarry_basin360_corrected.npz \
  --day 360 --out results/dino_1455/tcarry_basin360_reverdict.json
```

The paired scorer is fail-closed. It reuses `acceptance_gate_90d.load_candidate`
and `.metrics`, `acc_driver_decomp.group_transport`, and
`basin_seasonal_decomp.row_transport`; checks their two reductions to
`1e-12 Sv`; prints full-precision rows, `G4`, `Gbasin`, metric deltas, hashes,
and stamps; enforces paired day-0 identity, selector identity, legacy baseline,
and floor controls; and runs non-vacuous dry/wet transport and selector-swap
plants on every invocation. `--self-test` exercises the mutually exclusive
classifier and controls without reading a GPU result.

No arm in either stage runs in this CPU-only preregistration round.
