# PRE-REGISTRATION — missing EEN × bridge-Omega interaction corner

Written 2026-08-28 after bridge-Omega ownership was REFUTED and before the
missing fourth arm is run.  The corrected-T endpoint remains unread and
unscored.  This is a GPU handoff design; STOP after committing it.

## 2026-08-28 post-run binding amendment before score

The registered fourth-corner arm completed, but its NPZ has not been opened
and no `D`, `I`, endpoint miss, or ownership classification has been computed
or seen before this amendment.  Bind its clean producer
`b14a17dd6f14592594daacc4b64c6a1de2a0a004` and exact receipts:

- artifact `results/dino_1455/tcarry_omega90_legacy_een_off.npz`, SHA-256
  `678a6a914367561596a259cc27a50f9294d14e7c0ed32c68aefe9ee6fff2a6f8`;
- log `results/dino_1455/tcarry_omega90_legacy_een_off.log`, SHA-256
  `9145e6100b8b4eaea38741209190ba08bfa71578406c21ae8fcee39ce7f83639`.

Hashing both files and reading only the log before this commit confirmed the
registered legacy-rounded bridge-Omega and EEN-off banners, clean producer,
fp64/both/bridged/15552000 s legacy stress-carry receipts, 2880 steps,
`STABLE=True`, and the standalone `PASS 5 | FAIL 0` gate at 5x.  This amendment
changes no metric, locked corner, reducer, baseline, floor, validity bar,
ownership bar, interaction label, control, or decision branch.

## Why this is the next discriminator

Three corners of the current-harness 2x2 are now hash-bound and measured:

| bridge Omega | EEN weighting | day-90 basin gap [Sv] |
|---|---|---:|
| NEMO | NEMO | `A = -0.4390855099920348` |
| legacy-rounded | NEMO | `B = -0.4326316717808396` |
| NEMO | off | `C = -0.3484425774226274` |
| legacy-rounded | off | `D = UNMEASURED` |

The historical baseline used legacy-rounded bridge geometry and EEN off, so
`D` is the exact missing composition.  Transferring either historical EEN or
current Omega responses across the other axis is already contradicted by the
three corners.  Under additivity they predict

`Dadd = B + C - A = -0.34198873921143225 Sv`,

which is `-0.08379613937010433 Sv` away from the historical baseline
`-0.4257848785815366 Sv`.  The fourth corner decides whether that large
non-commuting interaction owns the epoch or whether another change remains.

## Exact arm and GPU invocation

No implementation change is required.  Use the committed fail-closed
selectors together, from one clean producer with no other DINO override:

```sh
DINO_EEN_METRIC=off CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 \
  LEGOESM_NEMO_E3T=both \
  .venv/bin/python scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py \
  nemo_dino_kamm_mlf results/dino_1455/tcarry_omega90_legacy_een_off.npz \
  --days 90 --bridge-before --save-3d --no-surface-stress-implicit \
  --bridge-omega legacy-rounded
```

The log must show clean producer and dirty count zero, fp64, `both`, bridged,
seasonal 15552000 s, `U_AS_T_LEGACY`, `ARM: bridge_omega=legacy-rounded`,
`ARM: een_metric_weighting=off`, resolved EEN `off`, 2880 steps, and
`STABLE=True`.  Its standalone acceptance gate must certify `PASS 5 | FAIL 0`
at 5x.  Commit its artifact/log SHA-256 binding before loading the NPZ.

## Exact outputs and controls

Extend the paired scorer into a four-corner scorer that reuses the same
`tcarry_basin_reverdict._reduce` functional.  It prints:

1. the three locked corner gaps `A`, `B`, and `C`, each required to reproduce
   its hash-bound receipt within `1e-12 Sv`;
2. `D = GoldOmega_EENoff90`, the new same-functional basin gap;
3. `I = A - B - C + D`, the signed EEN-by-Omega interaction;
4. `D - Dadd`, which is algebraically the same `I` and must agree within
   `1e-12 Sv` as a planted arithmetic cross-check.

Every corner's row-transport and `g_south` reducers must agree within
`1e-12 Sv`.  Day-0 T/S/eta/u/v and ordinary run-config leaves must be
bit-identical across the relevant pairs; only registered Omega/EEN receipts
may differ.  Existing Omega mode/rate/hash plants remain, and new plants must
swap the fourth corner's EEN receipt and replace `D` with `Dadd`; both must
fire.  The scorer prints git SHA, flags, all input hashes, and controls.

## Frozen validity and ownership bars

Keep the existing reproduction/ownership band
`2F = 0.0002899800477248501 Sv`; it is not a new floor.

- **INVALID/STOP before ownership scoring** if any locked corner misses its
  receipt by more than `1e-12 Sv`, the new acceptance gate is uncertified,
  any paired receipt differs off-axis, or independent reducers miss `1e-12`.
- **CONFIRM COMBINED EEN×OMEGA OWNERSHIP** iff valid and
  `|D - (-0.4257848785815366)| <= 2F`.
- **REFUTE COMBINED EEN×OMEGA OWNERSHIP** iff valid and that inequality misses.
- Descriptively label the axes **NON-ADDITIVE** iff `|I| > 2F`, otherwise
  **ADDITIVE/BELOW-BAND**.  This label cannot override the ownership result.

The planted classifier self-test must reach INVALID, CONFIRM, and REFUTE.  On
CONFIRM, the epoch is owned, `Glegacy90` may be re-registered at the current
baseline, and the current-SHA floor ensemble becomes the next gate.  On
REFUTE, STOP and isolate the remaining config-side Omega pin or another
model-relevant historical/current change before adopting any baseline.
