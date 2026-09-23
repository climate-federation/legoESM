# PRE-REGISTRATION — Rule-1e reconciliation of the day-90 basin baseline

Written 2026-08-27 after the retained Stage-1 legacy arm stopped at
`G_basin90 = -0.43908550999203477 Sv`, and before any day-30/day-60 value,
row profile, or historical EEN-arm basin value was computed.  The paired
corrected-T arm remains unread and unscored until this disagreement is owned.

## 2026-08-27 instrument-only amendment after reconciliation STOP

The first invocation stopped before producing any metric.  The reused
`load_candidate` refused the three historical artifacts because they predate
the later `seasonal_t0_reference_seconds` stamp, even though their clean logs
and artifact `seasonal_t0_seconds` fields record the exact 15552000 s clock.
No day-30/day-60/day-90 gap, row, EEN basin value, or reconciliation verdict
printed before this amendment.  The probe now uses that loader's existing
`DINO_GATE_ALLOW_LEGACY_CLOCK=1` escape only while reading the three
hash-bound historical artifacts, after independently checking their clean
producer logs and 15552000 s artifact stamps.  The current retained artifact
is still read with the escape absent and must pass the current clock guard.
No input, metric, reducer, candidate prediction, or decision bar changed.

The amended invocation then stopped before printing any reconciliation metric
because the old day-30 gap recomputed through the current independent reducer
was `-0.010717232432998713 Sv`, while the JSON serialization is
`-0.010717232432999602 Sv`: an `8.9e-16 Sv` arithmetic-order difference.  The
registered bit-for-bit cross-implementation receipt was therefore
unsatisfiable.  It is replaced by the same `1e-12 Sv` agreement bar already
owned by `tcarry_basin_reverdict._reduce`; a planted `1e-6 Sv` receipt shift
must fail that check.  This changes only the receipt equality operator.  No
input, computed metric, candidate prediction, or scientific decision bar
changed, and no day result or verdict printed before this amendment.

## Inputs and provenance question

The published `-0.4257848785815366 Sv` is an analysis value in
`/tmp/dino_basin_seasonal_decomp.json` (SHA-256
`63d4e60dd68281bc6101a35f86cb3f4406cb2ddbc27848b74473876226e85849`).
That JSON stamps analysis HEAD
`1d68fb289d6457e74ced8a1c71ab7eaceb8a29b1`; it is not the model producer.
The underlying clean member-0 trajectory is
`/tmp/dino_verdict360/m0_control.npz` (SHA-256
`f2031397ec6230aa07c40408947821e5c67a2156c5099ead6a66182e3e3f4ec9`),
whose `.launch_sha` and log both name
`a7b940f75c04d824b478de4e1728220e3a71989e`, dirty tracked count zero.

That producer ran `nemo_dino_kamm_mlf`, fp64, NEMO `both` ladders, the
restart seasonal clock at 15552000 s, and an exact bridged before level.  Its
log pins `barotropic_diffusion_alpha=0`, `barotropic_face_depth=nemo_ssh_avg`,
the three NEMO vertical-drag selections, `nemo_iso_lap`, `kappa_GM_max=200`,
and the analytic -0.2/+0.1 Pa stress extrema.  The current retained legacy
artifact is `results/dino_1455/tcarry_basin90_legacy.npz` (SHA-256
`ae114e7c71f530da253083e4f07f83e94bb66ed1d89c06c27909da9b36fc1e37`),
produced cleanly at `d6dc89e91c9ae6b07d146991d2cb6c850f261bb0` with the same resolved
protocol and the historical `U_AS_T_LEGACY` carry.

## Reuse audit

Searched `G_basin`, `g_south`, `row_transport`, `seasonal_decomp`,
`een_metric`, and `DINO_EEN_METRIC`.  The probe will reuse, without changing
their conventions:

- `acceptance_gate_90d.load_candidate` for every legoESM snapshot;
- `basin_seasonal_decomp.nemo_state(0, day)` for the retained NEMO member;
- `tcarry_basin_reverdict._reduce`, which already cross-checks the row sum
  against `acc_driver_decomp.rowset()["g_south"]` to `1e-12 Sv` and owns the
  dry/wet planted controls.

The search also found the already-run one-variable EEN pair at clean HEAD
`a6a07a9e2b4c31201691bd829bceb4984f374d62`:
`/home/dbalwada/legoESM/results/dino_1455_een_metric/twin90_armA_off.npz`
(SHA-256 `7677ef28c8733673117576b73f000574e1737f008d62c7686fdc45997f86a6ac`)
and `twin90_armB_nemo.npz` (SHA-256
`5c164e41a8a1f8259a712263d281391dc7dff04bbabf72c5d6e2cc72e02830ac`).
Their logs certify the same clock/grid/start/dtype and exactly one selected
difference, `een_metric_weighting: off -> nemo`.  The existing gate result
records a +0.006142 Sv circumpolar move, but no southern-basin value has been
computed from these artifacts.

## Exact outputs

At days 30, 60, and 90 the probe prints, at full precision:

1. `G_old`, recomputed from the `a7b940f` member and the retained NEMO member;
2. `G_current`, from the retained `d6dc89e` legacy arm and the same NEMO member;
3. `S_epoch = G_current - G_old` and rows 0..13 of that shift;
4. `S_EEN = G_nemo_weighted - G_off` from the historical one-variable EEN
   pair, including its rows 0..13 and wall rows 1..4;
5. `S_remainder = S_epoch - S_EEN`, the signed fraction
   `S_EEN/S_epoch`, and the old same-day floor from the locked JSON;
6. the direct coefficient scale
   `|G_old| * |Omega_NEMO/Omega_rounded - 1|`, where the bridge changed from
   `7.292e-5` to NEMO's measured `7.292115083046e-5`.

The old JSON values at all three days must reproduce within `1e-12 Sv` from
the independent old-member reduction.  The day-0 prognostic arrays of old and current artifacts
must be bit-identical.  Artifact hashes, log producer SHAs/config receipts,
flags, current Git SHA, and dirty tracked count print before any metric.
Dry-face plants must leave the score unchanged and a wet-face plant must move
it, or the probe stops.

## Predictions and bars, frozen before the offline score

The observed endpoint epoch shift is already known from the executor:
`S_epoch90 = -0.01330063141049817 Sv`.

**C1 — the two-Earth bridge/rotation fix.**  The live geometry coefficient
increased by `1.578e-5` relative; the card-side pin changed only
`-1.257e-7` relative.  A direct linear response on the old day-90 gap is thus
order `6.7e-6 Sv`, about 2000 times smaller than the observed shift.  It also
predicts a diffuse Coriolis-sensitive row response, not a wall-only jump.
Direct linear ownership is REFUTED if the EEN-subtracted day-90 remainder is
more than 100 times that coefficient-scaled value.  It is PLAUSIBLE as a
nonlinear accumulated contributor only if the absolute remainder grows in
the same signed direction from day 30 through 60 to 90 and is spread beyond
rows 1..4.  No offline pattern alone can CONFIRM nonlinear rotation ownership;
that requires a same-SHA 90-day old-Omega/current-Omega arm.

**C2 — the EEN transport metric weighting.**  The one-variable pair predicts
the sign and size directly: `S_EEN90` from those retained arms.  It CONFIRMS
dominant ownership of the baseline disagreement if it has the observed
negative sign and explains 80–120% of `S_epoch90`.  It owns the disagreement
to the campaign floor if additionally
`|S_remainder90| <= 2*F_old90 = 0.0002899800477248501 Sv`.  It is REFUTED as
dominant owner if it has the opposite sign or explains less than 20%; values
between are PARTIAL.  These are epoch-reconciliation labels, not a new verdict
on the southern-basin physical deficit.

If neither candidate owns the remainder to the floor, STOP remains active.
The discriminating GPU design is then one 90-day replay of the legacy carry
at model producer `a7b940f75c04d824b478de4e1728220e3a71989e`, using the
current fail-closed harness/scorer backported without model changes, followed
by current-SHA one-variable old/new Omega only if the replay reproduces the
old baseline.  No such arm is run in this CPU-only round.
