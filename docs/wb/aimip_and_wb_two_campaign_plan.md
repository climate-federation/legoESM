# AIMIP and WeatherBench: two separate campaigns (Design B) — plan

**Status:** active plan (2026-07-18). Owner: Linnia.
**Direction:** Design **B** — AIMIP and WeatherBench are **independent campaigns**
with **different protocols**; each model family is **trained twice** (once per
campaign). Each campaign is **internally protocol-matched** (fair within itself);
the two are **not** cross-comparable (a family appears twice, trained
differently). This supersedes the earlier Design-A "one model, two rulers" scope.

**Why B (not A):** only the AIMIP pipeline can feed *both* evals
(`run_aimip_amip_inference.py` for AMIP realism **and** `run_aimip_wb2_eval.py`
for WB2), while the WB harness is WB2-only — so Design A forced everything through
AIMIP. We chose instead to keep WeatherBench as its own native campaign
(`run_wb_campaign.pbs`, T63) and accept training each family twice.

---

## Campaign 1 — AIMIP (yours; T106; AIMIP eval)

Pipeline `run_aimip.py`, config tree `config/aimip/scale/`, single protocol
**T106, 1152 scattered pairs (1979–2014 × 4 seasons), base curriculum
12→24→72→120 h (7 epochs)**. Internally fair: all families identical except the
model. Eval = `run_aimip_amip_inference.py` (prescribed-SST AMIP realism).

| family | status |
|---|---|
| `column_nn` | **trained** (days) ✓ |
| `classical` | **running** ✓ |
| `sfno_full` | **config ready** — `variant_sfno_full.yaml` now 32/2 + base curriculum + `aimip_lr 5e-5` (P1-confirmed to fit 40 GB). Pending a 5-min finite check, then run. |
| `sfno_physics` | optional 4th (not required). |

**Internal-fairness note:** `sfno_full` was the only AIMIP family with the
curriculum OFF; the fair-set edit turns it ON to match `column_nn`/`classical`.
This is needed for AIMIP's *own* comparison regardless of WeatherBench.

**Eval:** AIMIP AMIP-inference on all trained families. (`run_aimip_wb2_eval.py`
is now OPTIONAL — a bonus WB2 cross-check of AIMIP checkpoints, not the WB
deliverable.)

## Campaign 2 — WeatherBench (Pierre's harness; T63; WB2 eval)

Pipeline `scripts/run/run_weatherbench_campaign.py` via
`scripts/cluster/derecho/run_wb_campaign.pbs`, config `config/wb/campaign/spectral_t63.yaml`,
3 families **physics / neural_gcm / sfno**, WB2 scorecard vs real SOTA
(`fetch_wb2_sota.py` → `plot_wb_scorecard.py`). Internally fair by its own
docstring (one `--config`, only the model differs).

| family | status |
|---|---|
| `physics` | **runnable now** — 60-Jan is fine for physics (no NN to starve). |
| `neural_gcm` | **BLOCKED (#1160)** — 60-Jan starves the NN (#1047 null); needs scattered ~10²–10³ pairs + eval-split early stopping. |
| `sfno` | same data blocker; **stopgap** = the T63 scattered-data run (`train_sfno_full_scale.pbs`, 3200 pairs) — but that is the *AIMIP pipeline at T63*, not `train_weatherbench_scale`, so it is **not** protocol-matched to WB physics/neural_gcm. Resolving that is Pierre's call. |

**WB campaign open question (Pierre's):** for a fair WB scorecard all three
families must train through **one** driver at one protocol. Today physics/neural_gcm
go through `train_weatherbench_scale` while the only working sfno (scattered data)
goes through `run_aimip` at T63. Either (a) give `train_weatherbench_scale` the
scattered-data loader (#1160) and run all three natively, or (b) accept the
AIMIP-pipeline-T63 sfno as the WB sfno and document the driver difference.

---

## What to run now

**AIMIP (self-contained, ready):**
1. Finite check (~5 min): `run_aimip.py --suite config/aimip/scale/finitecheck_sfno_full.yaml`
   with `JAX_ENABLE_X64=1 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_ALLOCATOR=platform`.
   Epoch-0 finite → good; NaN → lower `aimip_lr`.
2. `qsub -v VARIANT=sfno_full,LEGOESM_REPO=<worktree> train_aimip_derecho.pbs`
   (single-GPU; self-chains per epoch).
3. AIMIP eval (`run_aimip_amip_inference.py --variant <f>`) on classical / column_nn / sfno_full.

**WeatherBench (coordinate w/ Pierre):**
4. Physics now: `qsub -v LEGOESM_REPO=<worktree>,WB_MODES=physics run_wb_campaign.pbs`.
5. neural_gcm / sfno: hold for the #1160 scattered-data loader; the T63 scattered
   sfno (`train_sfno_full_scale.pbs`) keeps running as the interim sfno result.

## P1 result (kept — still governs the AIMIP sfno_full)

**PASS (2026-07-18):** 32/2 T106 sfno_full ran the worst-case 120 h / 20-step
checkpointed backward + optimizer update within 40 GB (`grad_norm` returned, no
`RESOURCE_EXHAUSTED`). So the AIMIP fair-set sfno_full carries the curriculum.
Probe config: `config/aimip/probe_t106/`. The probe's epoch-0 NaN was an expected
artifact of jumping random weights straight to a 120 h rollout; the real
short-start curriculum (12 h) avoids it — hence the finite check + `aimip_lr 5e-5`.

## Watch items

- **sfno_full has no mid-epoch resume** (host-resident, per-epoch checkpoint) →
  each 12 h link must finish ≥1 epoch; the 120 h-phase epoch over 1152 pairs
  (~2–4 h est. at 32/2) is the one to watch (stops loudly if it exceeds ~11 h).
- WB2 eval numbers are **internal-only** (sample-mean climatology, tiny init
  sampling, t2m/u10 proxies) — honest floors are own persistence/climatology.

## Related

`#976` (WB2 wind round-trip, fixed) · `#1047` (60-sample null) · `#1160`
(WB scattered-data pre-flight) · `[[aimip-scale-training-setup]]` ·
`[[wb-scale-training-derecho]]`.
