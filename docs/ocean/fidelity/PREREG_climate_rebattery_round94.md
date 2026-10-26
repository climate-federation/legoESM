# Preregistration: current-default climate re-battery, round 94

Date: 2026-08-30. Session: `01a04e34-d1fb-73e0-b25a-177641f0a246`.
Frozen before any new GPU arm is run.

## Question and epoch rule

After the tracer-velocity write, coupled momentum/QCO path, Redi flux, and A33
fixes, what are the current faithful defaults' registered MLD, day-360
southern-basin transport, and five-day wall-flicker verdicts?

Old legoESM arms are not admissibility controls: they were produced at older
code/config epochs and twice caused invalid-control interpretations. The new
epoch control is an independent duplicate of the current faithful default.
Both members are launched from one pinned clean producer, the same bridged
BEFORE state and T-point stress reconstruction, fp64, and NEMO ladder `both`,
with no physics-selector overrides. A result is invalid unless the duplicate
members have identical run configs, day-0 state, masks, and bit-identical
fields at every scored snapshot. The scorer records member A's metrics as the
fresh epoch baseline only after that gate passes.

## Arms

| pair | length | storage | purpose |
|---|---:|---|---|
| `climate_a`, `climate_b` | 360 d | fp64 3-D at days 0, 90, 360 | day-90 MLD and day-360 basin |
| `wall_a`, `wall_b` | 5 d | every-step fp64 SSH, 160 samples | wall flicker |

The harness receipt must stamp every selector introduced by the ZDF and
momentum lanes, including the Redi horizontal/skew/A33/W-stage selectors,
Treguier reduction/sqrt, barotropic seed/EEN/PGF/continuity/transport,
coupled ZAD, and WZV-call-2 selectors. The two DINO cards' resolved current
values must all be the faithful values; a missing stamp or any CLI physics
override invalidates the battery.

## MLD bands

The scorer reuses the reviewed `mld_climate_audit.py` criterion and NEMO
day-90 reconstruction, SHA-256
`cf1bcffb4ee7994bd4eb433f6b3fa3ba5ac0f2610463b9bb3133ea0c1762dc14`.
The frozen southern-basin RMS bands are unchanged:

- `CONFIRM` at or below `11.2397455 m`;
- `REFUTE` at or above `20.2775 m`;
- `PARTIAL/INDETERMINATE` between them.

Channel/equator bias and RMS are reported without new post-hoc bars. The old
`22.479491 m` lego baseline is context only, never a control-admission target.

## Day-360 basin bands

Use the registered `tcarry_basin_reverdict.py` NEMO reducer and day-360
comparator. Preserve the historical registered gap
`-0.9519122331848315 Sv` and floor `F=0.06173656216045926 Sv` solely as the
response target. Let `delta=current_gap-historical_gap` (positive improves
the negative deficit):

1. `|delta| <= 2F`: `UNRESOLVED/FLOOR`;
2. current faithful fails any absolute 5x acceptance gate: `INVALID_CURRENT`;
3. `delta/|historical_gap| >= 0.10`: `CONFIRMED`;
4. `delta/|historical_gap| <= 0.02`: `REFUTED`;
5. otherwise: `UNRESOLVED`.

The fresh current gap and per-row vector are persisted as the new epoch
baseline. The historical gap is not required to reproduce.

## Wall-flicker bands

Use certified comparator `/tmp/dino_eta_waves/nemo_5d_eta.npz`, SHA-256
`52bc6c70697126f7522114dbe2fc5cda5b56ce566db28f488d6809b79997b47a`.
After duplicate identity and the existing land-poison/amplitude plants pass:

- `CONFIRMED` if first-eight-step all-domain ratio is at most `1.25` and wall
  share at most `0.17`;
- `REFUTED` if ratio is at least `2.30` and wall share at least `0.38`;
- otherwise `UNRESOLVED`.

No legacy-control range is applied. The measured ratio/share pair becomes the
fresh wall epoch baseline.

## Fail-closed controls

The scorer must reject dirty or differing producers, non-fp64 control or
storage, wrong seasonal clock/start/stress/ladder, missing snapshots, config
or mask differences, non-bit-identical duplicate fields, a wrong NEMO wall
hash, and any non-default faithful selector. Planted classifier values must
reach every MLD, basin, and wall verdict branch. A failure produces
`INVALID_REBATTERY`; no individual science verdict may be quoted.
