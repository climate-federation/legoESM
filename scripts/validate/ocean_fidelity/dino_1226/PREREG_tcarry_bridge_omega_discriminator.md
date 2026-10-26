# PRE-REGISTRATION — paired bridge-Omega baseline discriminator

Written 2026-08-28 after the hash-bound current-trajectory EEN-off arm
REFUTED full EEN ownership, and before an old/new bridge-Omega selector is
implemented or either arm is run.  The corrected-T endpoint remains unread
and unscored.  This is a GPU handoff design; STOP after committing it.

## 2026-08-28 post-verdict default decision

These completed historical arms intentionally used and stamped
`U_AS_T_LEGACY` when that carry was the default.  After the human default flip,
reproduction requires adding `--bridge-before-stress-legacy-u-as-t` to each
command below.  Their receipts, measurements, and decision bars are unchanged.

## 2026-08-28 instrument-only receipt amendment before GPU

Implementation inspection found that `LatLonCGridGeometry` stores three
Coriolis arrays, `f_T`, `f_u`, and `f_v`, plus its scalar `omega`.  The initial
text below incorrectly said the selector changes "the two bridge Coriolis
arrays" because it named only the T and v content hashes.  No arm has run and
no metric has been seen.  Correct the receipt scope to the geometry scalar and
all three deterministic Coriolis arrays; add `bridge_f_u_sha256`.  Every other
geometry/state/config leaf must remain bit-identical.  No scientific output,
baseline, floor, threshold, validity rule, or classifier changes.

## 2026-08-28 post-run binding amendment before score

Both registered arms completed, but neither NPZ has been opened and no
`GnewOmega90`, `GoldOmega90`, or `Domega90` value has been computed or seen
before this amendment.  Bind the clean common producer
`9e339ad1b2032bc47ec132fb2ad6f00ea071bbb9` and these exact files:

- NEMO-Omega artifact SHA-256
  `862debedf76b5e4ff686655f89ee9d33eec56c87a8a05c3f7cffff1502dc53d6`;
- NEMO-Omega log SHA-256
  `4081b4950d3e551937ee492541573f48b0fcc19ee8c4d3eee90f0e45254beaf4`;
- legacy-rounded artifact SHA-256
  `dc11a5f831f4dec573b3c04d1b2f3b73ea8e74bfd1a30f3c20029757c5928a94`;
- legacy-rounded log SHA-256
  `52045b52fee947dd8fdbd682773726d6e88ae6152f82f9a42446f99949d14b7d`.

Hashing the files and reading the logs before this commit confirmed the
registered arm banners, clean producer, fp64/both/bridged/15552000 s legacy
stress-carry receipts, resolved NEMO EEN weighting, 2880 steps, and
`STABLE=True`.  That log-only verification did not open either artifact.  No
metric, reducer, baseline, floor, validity bar, ownership bar, or classifier
changes in this binding amendment.

## 2026-08-28 scored result — REFUTED and STOP

The bound scorer ran at clean commit
`eee4b9cee2d8a37f8438206e97b84bee52b13aa2`, scorer SHA-256
`783f7c6c16bb5ddb273a433b47952ccaa3e9253b5b42b4e69d64e5cc2aec5741`.
All receipt plants fired, paired day-0 fields were bit-identical, both basin
implementations agreed at or below `1.7763568394002505e-15 Sv`, and both
acceptance gates certified `PASS 5 | FAIL 0` at 5x.

The frozen outputs are:

- `GnewOmega90 = -0.4390855099920348 Sv`, exact reproduction of the retained
  current baseline at printed precision;
- `GoldOmega90 = -0.4326316717808396 Sv`;
- `Domega90 = +0.006453838211195162 Sv`.

Validity therefore PASSES, but the old-Omega endpoint misses the historical
baseline by `0.006846793199303036 Sv`, and the paired delta misses its target
by `0.0068467931993030084 Sv`: both are 23.61 times the frozen band.  The
registered outcome is **REFUTED_BRIDGE_OMEGA_OWNERSHIP**.  Output
`/tmp/tcarry_bridge_omega_score.json` has SHA-256
`d8bdb1dcd87e9dcc2b051d163a2b7e0268b25bd8f479dbe5c0fdc06db7ba18b8`.

No baseline or floor is re-registered; STOP remains active.  The next
discriminator is the missing fourth corner of the current-SHA EEN-by-bridge-
Omega 2x2, preregistered in `PREREG_tcarry_een_omega_interaction.md`.

## Question and exact output

The historical legacy-carry baseline is
`Ghistorical90 = -0.4257848785815366 Sv`, produced when the NEMO bridge built
its Coriolis geometry with rounded legoESM Earth rotation.  The retained
current legacy-carry baseline is
`Gcurrent90 = -0.43908550999203477 Sv`, with the NEMO sidereal rate.  Their
signed old-minus-new target is:

`Dtarget = +0.01330063141049817 Sv`.

The paired scorer will produce exactly three scientific numbers with the
existing `tcarry_basin_reverdict._reduce` functional:

1. `GnewOmega90`, the NEMO-sidereal bridge arm minus retained NEMO member 0;
2. `GoldOmega90`, the rounded-legacy bridge arm minus the same NEMO member;
3. `Domega90 = GoldOmega90 - GnewOmega90`.

Both row-transport and `acc_driver_decomp.rowset()["g_south"]`
implementations must agree within `1e-12 Sv` for every endpoint.

## Selector to build, exactly

Add a fail-closed `kamm_twin_90d.py` CLI selector:

```text
--bridge-omega {nemo,legacy-rounded}
```

Its default is `nemo`, preserving every existing invocation bit-for-bit.
It changes only the `omega=` supplied to
`bridge_nemo_to_legoesm_topo`: `NEMO_CONSTANTS_CONFIG.Omega` for `nemo`, and
`legoesm.constants.Omega` for `legacy-rounded`.  It must not change
`DINOConfig.omega`, the model config's constants, forcing, restart fields,
EEN weighting, stress carry, or any prognostic value.  Thus the legacy arm
recreates only the historical CONFIG-versus-GEOMETRY two-Earth split; both
arms keep the current NEMO config-side rate.

Do not relax the bridge's NEMO-f guard.  Extend it with an opt-in
counterfactual reference mode used only by `legacy-rounded`: validate the
built `f_T` against NEMO `ff_t` scaled by
`legoesm.constants.Omega / NEMO_CONSTANTS_CONFIG.Omega` at the existing
precision-aware tolerance, while still printing the intentional unscaled
NEMO mismatch.  Default `nemo` validation is unchanged.  A planted mode/rate
swap must fail this guard.

Every artifact must stamp content, not only intent:

- `bridge_omega_mode` (`nemo` or `legacy-rounded`);
- `bridge_omega_rad_s`;
- `bridge_f_T_sha256`, `bridge_f_u_sha256`, and `bridge_f_v_sha256`;
- `config_omega_rad_s` (identical NEMO value in both arms);
- `een_metric_weighting` (identical `nemo` in both arms).

Unit tests must prove the default is bit-identical, the selector changes only
the geometry's registered `omega` scalar and `f_T/f_u/f_v` arrays, the
config-side rate remains identical, and a planted stamp/rate swap is rejected.
The paired scorer must compare day-0
T/S/eta/u/v bit-for-bit and require all ordinary run-config leaves identical;
only the registered bridge-Omega stamps and their Coriolis hashes may differ.

## Exact GPU invocations

Build and commit the selector and paired scorer first.  Run both arms from
that same clean producer, with no `DINO_EEN_METRIC` environment override:

```sh
CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both \
  .venv/bin/python scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py \
  nemo_dino_kamm_mlf results/dino_1455/tcarry_omega90_nemo.npz \
  --days 90 --bridge-before --save-3d --no-surface-stress-implicit \
  --bridge-omega nemo

CUDA_VISIBLE_DEVICES=1 JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both \
  .venv/bin/python scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py \
  nemo_dino_kamm_mlf results/dino_1455/tcarry_omega90_legacy_rounded.npz \
  --days 90 --bridge-before --save-3d --no-surface-stress-implicit \
  --bridge-omega legacy-rounded
```

Both logs must show clean identical producer SHAs, fp64, `both`, bridged,
seasonal 15552000 s, `U_AS_T_LEGACY`, default NEMO EEN weighting, 2880 steps,
and `STABLE=True`.  Both separate acceptance gates must certify `PASS 5 |
FAIL 0` at 5x.  Bind artifact and log SHA-256s in a committed amendment before
the scorer loads either NPZ.

## Frozen validity and decision bars

The reproduction/ownership band remains the already-registered historical
`2F = 0.0002899800477248501 Sv`; it is not a new floor estimate.

- **INVALID/STOP before ownership scoring** if
  `|GnewOmega90 - (-0.43908550999203477)| > 2F`, either acceptance gate is
  uncertified, any paired receipt differs off-axis, or either independent
  reducer misses `1e-12 Sv`.
- **CONFIRM bridge-Omega ownership** iff valid and both
  `|GoldOmega90 - (-0.4257848785815366)| <= 2F` and
  `|Domega90 - 0.01330063141049817| <= 2F`.
- **REFUTE bridge-Omega ownership** iff valid but either CONFIRM inequality
  misses its band.

Both scientific outcomes are reachable in the scorer self-test.  On CONFIRM,
the epoch is owned and `Glegacy90` may be re-registered at the current
baseline; the current-SHA floor ensemble must still run before the corrected-T
basin verdict.  On REFUTE, STOP and inventory other model-relevant changes or
non-commuting EEN/Omega interactions before any baseline adoption.
