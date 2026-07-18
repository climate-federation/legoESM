# AIMIP T106 checkpoint handoff (for evaluation)

Trained weights for the AIMIP scale campaign (T106, `config/aimip/scale/`).
**Weights are NOT in git** (`results/`/`*.eqx` are gitignored runtime artifacts);
this file is the pointer + provenance so a developer can evaluate them.

## How to evaluate a checkpoint

The eval drivers rebuild each variant's skeleton **exactly as trained** from
`--variant` + `--suite` (both in this repo), then load `--ckpt`:

```bash
# AIMIP AMIP-realism eval:
python scripts/run/run_aimip_amip_inference.py \
  --variant <variant> --suite config/aimip/scale/suite_<variant>.yaml \
  --ckpt <shared_path>/<variant>/params.eqx

# WB2 forecast-scorecard eval (bonus cross-check):
python scripts/validate/run_aimip_wb2_eval.py \
  --variant <variant> --suite config/aimip/scale/suite_<variant>.yaml \
  --ckpt <shared_path>/<variant>/params.eqx
```

**Check out the training commit first** (below) so the config the eval reconstructs
matches the weights. `sfno_full` additionally needs `norm_stats.npz` sitting next
to its `params.eqx` (the eval loads that sidecar).

## Checkpoints

Shared location (fill in when copied): `LEGOESM_CKPT_ROOT = <e.g. /glade/campaign/<proj>/aimip_t106_ckpts or a group-readable /glade/work path>`

| variant | trained-on commit | files | sha256 | notes |
|---|---|---|---|---|
| `classical`  | `<sha>` | `params.eqx` | `<fill>` | sweep-winner physics + RRTMGP |
| `column_nn`  | `<sha>` (earlier — record the checkout it ran on) | `params.eqx` | `<fill>` | column MLP 512/6 |
| `sfno_full`  | `62c857cfa` | `params.eqx`, **`norm_stats.npz`** | `<fill>` | 32/2, base curriculum, lr 5e-5 (fair-set) |

Protocol (all variants, identical — the fairness contract): T106, 1152 scattered
pairs (1979–2014 × 4 seasons), base curriculum 12→24→72→120 h (7 epochs). See
`docs/wb/aimip_and_wb_two_campaign_plan.md`.

## Copy + checksum (run on Derecho once a run's `params.eqx` exists)

```bash
# per variant — bundle the eval-required files to the shared location:
DST=$LEGOESM_CKPT_ROOT/<variant>; mkdir -p "$DST"
cp results/aimip_scale_t106/<variant>/<variant>/params.eqx "$DST/"
# sfno_full ONLY — the norm-stats sidecar is required for eval:
cp results/aimip_scale_t106/sfno_full/sfno_full/norm_stats.npz "$DST/"   # sfno_full only
sha256sum "$DST"/*                                                        # paste into the table above
```

(If off-cluster access is needed instead of a GLADE path, attach the same files
as GitHub Release assets — see the handoff note in the session.)
