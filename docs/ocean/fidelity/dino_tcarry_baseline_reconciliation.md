# T-carry basin re-verdict — Stage-1 baseline reconciliation

**Status (2026-08-27): STOP remains active.**  The retained legacy arm's
`G_basin90 = -0.43908550999203477 Sv` is valid at its own clean producer, but
it is not a reproduction of the registered historical baseline.  No
corrected-T endpoint has been loaded or scored.

Probe: `scripts/validate/ocean_fidelity/dino_1226/tcarry_baseline_reconcile.py`
at commit `3031854233091d1ed164f4154816db2c43ffd523`, SHA-256
`092201d1f9114d3d3855b2dfecd874f25cd7e2d41e347fc59be5673b7bcbe951`.
Output: `/tmp/tcarry_baseline_reconciliation.json`, SHA-256
`c9a8306d086648a1bdb82e303040462b15371dafcdf9a8752106f7c6830f9842`.
All numbers below are **CONFIRMED measurements** from that committed probe
unless explicitly labelled otherwise.

## Which run produced `-0.4257848785815366 Sv`

The number is stored in `/tmp/dino_basin_seasonal_decomp.json` (SHA-256
`63d4e60dd68281bc6101a35f86cb3f4406cb2ddbc27848b74473876226e85849`).
Its `git_head = 1d68fb289d6457e74ced8a1c71ab7eaceb8a29b1` is the **analysis** commit,
not the model producer.  The underlying member-0 model artifact and all four
member logs name the clean producer
`a7b940f75c04d824b478de4e1728220e3a71989e`; `.launch_sha` agrees.

The resolved historical configuration, from `m0_control.log`, was:

| field | historical artifact |
|---|---|
| recipe | `nemo_dino_kamm_mlf`, no option flags |
| dtype / vertical ladder | fp64 / NEMO `both` |
| start / clock | bridged before level / 15552000 s restart clock |
| barotropic diffusion / face depth | `0.0` / `nemo_ssh_avg` |
| vertical drag composition | in matrix, baroclinic-only, barotropic substep |
| lateral slope / GM cap | `nemo_iso_lap` / 200 |
| analytic wind extrema | -0.2 / +0.1 Pa |
| producer tree | `a7b940f75...`, dirty tracked count 0 |

The current retained legacy artifact was produced cleanly at
`d6dc89e91c9ae6b07d146991d2cb6c850f261bb0`, with those same logged fields,
the same restart, bit-identical day-0 prognostic fields, and the historical
`U_AS_T_LEGACY` stress carry.

## Model-relevant diff to base `782b0d788`

The content diff, not merge chronology, leaves two live numerical changes on
this serial DINO path.  More than one live row means attribution requires the
measured decomposition below.

| resolved field/path | `a7b940f` artifact | base `782b0d788` | status for this comparison |
|---|---|---|---|
| 3-D EEN transport metrics | per-unit-width, option off | NEMO `e1v/e1u`, `e2u/e2v`, card on | **LIVE DIFF**, `ae08a6740`, merged with the southern-basin lane |
| bridge/grid rotation rate | rounded global `7.292e-5` | NEMO sidereal `7.292115083046e-5` | **LIVE DIFF**, `15653c392`, PR #1681 |
| config-side rotation pin | key-cice literal `7.292116e-5` | NEMO sidereal expression | **LIVE but 1.257e-7 relative**, same PR #1681 |
| vertex-Coriolis placement | implicit historical cell average | explicit `cell_average` default | MATCH; new selector is not selected |
| twin start default | default changed after the old run | bridged | MATCH in artifacts; both logs certify bridged starts |
| after-reconcile face thickness | minimum | arithmetic mean | MATCH on open DINO full-step faces; commit `19590b454` records bit identity there |
| SPMD/OMIP/ORCA additions | absent | present | INACTIVE in this serial DINO recipe |
| corrected T stress carry | absent | opt-in selector | MATCH for legacy arm; selector omitted and stamp is `U_AS_T_LEGACY` |

## Discriminating offline measurement

The retained one-variable EEN pair was produced cleanly at `a6a07a9e2`, with
the same clock/grid/start/dtype and `een_metric_weighting: off -> nemo` as its
only selected difference.  Scoring it with the same reducers gives:

| day | whole epoch shift [Sv] | isolated EEN shift [Sv] | remainder [Sv] | EEN / epoch | remainder / direct Omega scale |
|---:|---:|---:|---:|---:|---:|
| 30 | -0.0024574125 | -0.0028900973 | +0.0004326848 | 117.61% | 2558x |
| 60 | -0.0007490707 | +0.0000181234 | -0.0007671940 | -2.42% | 252x |
| 90 | **-0.0133006314** | **-0.0065652314** | **-0.0067354000** | **49.36%** | **1002x** |

At day 90, EEN explains 70.86% of the wall-row (rows 1..4) epoch shift but
only 49.36% of the whole basin shift.  Only 37.86% of the remainder lies in
rows 1..4; 62.14% lies elsewhere.  Therefore:

- **EEN is PARTIAL**, under the registered 80–120% dominant-owner band.
- **Direct linear rotation-rate ownership is REFUTED**: the EEN-subtracted
  remainder is 1002 times the coefficient-scaled prediction.
- The registered nonlinear-Omega premise is **not supported offline**: the
  remainder changes sign between day 30 and day 60, although its absolute
  size grows thereafter and its day-90 row support is broad.  Attribution of
  the remaining `-0.0067354 Sv` is **UNRESOLVED**.

In signed-prediction terms, the historical EEN pair predicts a negative
`-0.0065652 Sv` day-90 move.  Coefficient scaling alone does not determine the
sign of the direct Omega response, so its honest prediction is two-sided but
only about `+/-6.7e-6 Sv`; either sign is three orders too small.  The
registered nonlinear-Omega premise predicts the observed negative direction
already at days 30 and 60 and growing through day 90.  Instead its isolated
remainder is `+0.0004327`, `-0.0007672`, then `-0.0067354 Sv`.

The `-0.4257848786 Sv` baseline is therefore not replaced merely because the
current artifact has better provenance.  Rule 1e's STOP is doing its job.

## Next discriminating arm and floor

The cheapest next arm is the current producer with only the EEN card selection
restored to `off`.  It tests the EEN response at the exact current trajectory,
rather than transferring the older A/B response across later merges.  The
exact command and bars are frozen in
`PREREG_tcarry_basin_reverdict.md`'s dated reconciliation amendment.

The historical `F90 = 0.00014499002386242506 Sv` is **not plausibly
SHA-stable**.  The same old ensemble's floor grows from `6.145e-8 Sv` at day
30 to `3.443e-6 Sv` at day 60 and `1.450e-4 Sv` at day 90; it is a
trajectory-dependent chaotic spread, and the trajectory has measurably moved.
A current-SHA control plus seeds 1/2/3 must remeasure the legoESM side before
the corrected-T deficit verdict is classified.  The NEMO member artifacts are
unchanged and may supply the hash-bound NEMO side of the same RSS formula.

## 2026-08-28 current-trajectory EEN-off discriminator

The registered arm was bound before scoring to artifact SHA-256
`a7f3bf5555ec1f6a7ed58792ede8b9f10188c93c21ff81f0553b2fb3cc400dd4`
and log SHA-256
`465c01c104dfb8e8f82dc5f6bd0ae70eb50d009ec125457ac259a5f0168e0d02`.
Its clean producer is `6c64f261aa33b43c372b81b30d4569481116766d`.
The requested content audit,
`git diff --stat d6dc89e91..6c64f261a -- packages/ src/`, is empty: the
producer interval contains only docs, validation scripts, and tests, with no
model-package or `src/` change.

The committed discriminator measured:

| registered quantity | result |
|---|---:|
| `Goff_current90` | `-0.3484425774226274 Sv` |
| distance from historical baseline | `0.07734230115890917 Sv` |
| frozen ownership band | `0.0002899800477248501 Sv` |
| miss / band | `266.71` |
| row versus `g_south` absolute-endpoint disagreement | `1.7763568394002505e-15 Sv` |
| acceptance gate | `PASS 5 | FAIL 0` at 5x |

The frozen outcome is **REFUTED_FULL_EEN_OWNERSHIP**.  EEN's response is
strongly trajectory-dependent: at the current trajectory, selecting NEMO
weighting moves the basin gap by
`-0.43908550999203477 - (-0.3484425774226274) = -0.09064293256940737 Sv`,
not the historical pair's `-0.006565231376747249 Sv`.  This is descriptive;
it does not assign the epoch shift to EEN or Omega.

The output `/tmp/tcarry_een_off_discriminator.json` has SHA-256
`35332f72823b76127972d76b039e039fd5c8de36db71ae42b64349cc89f7b1a1`.
The current baseline is **not re-registered**, the floor is unchanged, the
corrected-T arm remains unread, and STOP remains active.  The paired old/new
bridge-Omega discriminator is frozen in
`PREREG_tcarry_bridge_omega_discriminator.md`; do not run the seed-floor arms
until that baseline reconciliation succeeds.
