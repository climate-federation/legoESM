# PRE-REGISTRATION — corrected T carry and the southern-basin deficit

Written after the five-day Round-4 flicker arms and before either basin arm
runs. The five-day result is not used as a basin outcome. This registration
asks whether the now-confirmed bridge representation defect contributes
materially to the recorded southern-basin transport deficit.

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

Exact existing scorer invocations:

```sh
.venv/bin/python scripts/validate/ocean_fidelity/dino_1226/acceptance_gate_90d.py \
  results/dino_1455/tcarry_basin90_legacy.npz --level 5
.venv/bin/python scripts/validate/ocean_fidelity/dino_1226/acceptance_gate_90d.py \
  results/dino_1455/tcarry_basin90_corrected.npz --level 5

CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  .venv/bin/python -m \
  scripts.validate.ocean_fidelity.dino_1226.wall_visc_ablation_gap \
  results/dino_1455/tcarry_basin90_legacy.npz \
  results/dino_1455/tcarry_basin90_corrected.npz \
  --labels legacy-U-as-T,corrected-T
```

Before classification, both acceptance-gate artifacts must certify the claim's
grid/precision/clock, the legacy arm must reproduce `Glegacy90` within
`2*F90 = 0.0002899800477248501 Sv`, and the two arms must have bit-identical
day-0 prognostic fields and all stamps except the registered carry identity.
The scorer's dry/wet planted controls must pass.

Define `Delta90 = Gcorrected90 - Glegacy90` (positive reduces the negative
deficit) and `R90 = Delta90/abs(Glegacy90)`.

- **CONFIRMS material deficit contribution** iff `R90 >= 0.10`
  (`Delta90 >= 0.04257848785815366 Sv`), `|Delta90| > 2*F90`, and no
  acceptance-gate metric's absolute NEMO gap degrades by more than its own
  registered tolerance relative to legacy.
- **REFUTES material deficit contribution** iff `R90 <= 0.02`
  (`Delta90 <= 0.008515697571630733 Sv`). A negative material change is
  reported additionally as a material compensator, not silently folded into
  “no effect”.
- `0.02 < R90 < 0.10`, or a transport gain accompanied by a gate regression,
  is **UNRESOLVED**. Any `|Delta90| <= 2*F90` is explicitly floor-limited.

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
0.12347312432091852 Sv` or STOP. Define `Delta360` and `R360` as above:
CONFIRM requires `R360 >= 0.10` and `Delta360 > 2*F360`; REFUTE requires
`R360 <= 0.02`; everything between, any result inside `2*F360`, or a
compensating density/gate regression is UNRESOLVED. The final-90-day mean over
days `280..360` is descriptive unless a same-functional ensemble floor is
computed from the retained verdict members; the endpoint alone controls this
registered attribution verdict.

No arm in either stage runs in this CPU-only preregistration round.
