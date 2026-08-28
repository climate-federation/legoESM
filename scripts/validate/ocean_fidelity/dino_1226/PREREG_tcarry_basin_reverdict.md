# PRE-REGISTRATION — corrected T carry and the southern-basin deficit

Written after the five-day Round-4 flicker arms and before either basin arm
runs. The five-day result is not used as a basin outcome. This registration
asks whether the now-confirmed bridge representation defect contributes
materially to the recorded southern-basin transport deficit.

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
legacy U-as-T versus reconstructed analytic T. The legacy arm omits the
opt-in selector and must stamp `U_AS_T_LEGACY`; the corrected arm adds it and
must stamp `T`, `15552000.0`, and the registered content hash.

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
