# Unified AIMIP + WeatherBench-2 evaluation (Design A) — scope

**Status:** scoping (2026-07-18). No fair-set runs changed yet. Owner: Linnia.

## Goal

Evaluate the **same trained model families** under **two rulers** — the AIMIP
(interactive AMIP realism) evaluation *and* the WeatherBench-2 (forecast-skill
vs SOTA) evaluation — so that a model's AIMIP score and its WB2 score describe
**one object**, not two separately-trained ones. This is "Design A: train once,
eval twice," chosen over "Design B: two native campaigns" because it is the only
design under which *the training protocol is literally identical across the two
evaluations*, and because it does not train every family twice.

Families in scope: `classical`, `column_nn`, `sfno_full` (WB2-eval-supported
today). `sfno_physics` is a possible 4th (see Decisions).

## Fairness invariant (non-negotiable)

A scorecard is fair only if, **across families, everything is held byte-identical
except the model**: resolution · training data + sampling · epochs/steps · loss ·
rollout curriculum · eval leads · climatology baseline · metric. Two *different*
scorecards (AIMIP vs WB2) may differ **from each other**, but each must be
internally uniform. (This is the CLAUDE.md controlled-comparison rule.)

## Feasibility verdict: mostly already built

| piece | state |
|---|---|
| WB2 eval of an AIMIP checkpoint | **exists** — `scripts/validate/run_aimip_wb2_eval.py`, `VALID_VARIANTS = (classical, column_nn, sfno_full)`; rebuilds each variant's skeleton exactly as trained → `scorecard.json` |
| WB2-eval polar-wind NaN blocker (#976) | **fixed** — `f51ba9ded` (PR #1010), exact left-inverse for grid-winds→vor/div |
| AIMIP eval | **exists** — `scripts/run/run_aimip_amip_inference.py` + fleet figures |
| Training driver (all families) | **exists** — `scripts/run/run_aimip.py --suite` |

So Design A = train each family once via `run_aimip.py`, then score the same
`epoch_*.eqx` with (a) AIMIP metrics and (b) `run_aimip_wb2_eval.py --variant <f>`.

## The single protocol (anchor)

**AIMIP scale @ T106, 1152 scattered pairs** (`config/aimip/scale/`,
`base_t106_allyears.yaml`: 1979–2014 × 4 seasons × n_days=2). Rationale:

- `column_nn` (trained for days) and `classical` (running) are **already** on it.
- Its data is a **legitimate scattered regime** (36 yr × 4 seasons) — **not** the
  60-Jan starved set that produced the #1047 null. So keeping it does not
  re-introduce data starvation.

Everything else must match this. Only `sfno_full` is off-protocol today.

## Per-family work matrix

| family | now | Design-A action | WB2 eval |
|---|---|---|---|
| `column_nn` | T106 scale, trained (days) | **keep** ✓ | ✅ |
| `classical` | T106 scale, running | **keep** ✓ | ✅ |
| `sfno_full` | **T63 wbcompare, 3200, curriculum, 128/8** | **re-run at T106 scale** (see below) | ✅ |
| `sfno_physics` | — | out of WB scope unless bridged | ❌ not in `VALID_VARIANTS` |

## The one hard call — `sfno_full` at T106

The T106 scale `sfno_full` config exists (`config/aimip/scale/variant_sfno_full.yaml`)
but is **32/2, curriculum OFF, chunk OFF** — a *weak* emulator (that weakness is
why it was moved to the T63 wbcompare config). For a fair set it must be at T106
like the others. Options:

- **(a)** run the existing T106 32/2 sfno_full as-is — fair but weak.
- **(b)** enable **curriculum + chunking** on the T106 32/2 sfno_full (curriculum
  is now supported via `build_sfno_curriculum_epoch_plan`) — fairer *and*
  stronger, **if it fits 40 GB** → this is **P1**.

Note (b) also fixes a **latent within-AIMIP unfairness**: `column_nn`/`classical`
inherit the base curriculum + chunking, while `sfno_full` disables both.

**The running T63/3200 sfno_full (D4) continues as a SEPARATE standalone
"best-effort emulator" result — reported to Pierre in isolation, not a member of
the fair T106 set.**

## Current mismatches to close (sfno_full vs the anchor)

resolution T63→T106 · data 3200/n_days=5 → 1152/n_days=2 · curriculum on→(match) ·
size 128/8 → 32/2 · lr 5e-5 → (match base).

## Config consolidation

- **Source of truth = `config/aimip/scale/`** (T106) for all fair-set training.
- **`config/aimip/wbcompare/`** relabelled as the *standalone* T63 sfno experiment
  (D4), not a fair-set member.
- **WB-campaign NN families** (`config/wb/campaign/` neural_gcm/sfno) superseded by
  WB2-eval-of-AIMIP-checkpoints. Keep `run_wb_campaign.pbs` only if the team wants
  a *natively-WB-trained physics* baseline (a Design-B remnant — confirm w/ Pierre).
- **Two eval configs** over the same T106 checkpoints: AIMIP (amip inference) +
  WB2 (`run_aimip_wb2_eval.py`).

## Decisions & probes

- **D1** anchor = T106 (keeps column_nn's days). *Chosen: yes.*
- **D2** sfno_full option (a) weak-now vs (b) curriculum-at-T106. *Pending P1.*
- **D3** `sfno_physics` in/out (needs a `VALID_VARIANTS` addition to the WB2 bridge if in). *Open.*
- **D4** running T63 sfno_full → **keep as standalone**, report to Pierre in isolation. *Chosen.*
- **P1** memory probe: does 32/2 + curriculum + chunked fit T106/40 GB? **PASS (2026-07-18)** — the worst-case 120 h / 20-step checkpointed backward ran a full forward+backward+update within 40 GB (`grad_norm` returned; no `RESOURCE_EXHAUSTED`). Option **(b)** is memory-viable. (The probe's epoch-0 NaN is an expected artifact of jumping a random-init SFNO straight to a 120 h rollout — not memory, not instability; the real short-start curriculum avoids it.)
- **P2** WB2-eval numbers are internal-only (sample-mean climatology, tiny init
  sampling, t2m/u10 proxies) — confirm acceptable for the deliverable. *Open.*

## P1 — analytical prediction + empirical probe

**Analytical:** the memory note records **64/4 @ T106 ≈ 63 GB** (OOM). 32/2 halves
embed and blocks → ~¼ the activation ≈ **~16 GB** for one forward+backward. The
120 h curriculum uses a **checkpointed scan** (`jax.checkpoint nothing_saveable`,
recompute-in-backward), so a 20-step rollout costs **~1 step**, not ×20 — plus
~20 small state snapshots (~1–2 GB) + optimizer moments (small). **Predicted peak
~18–25 GB → fits 40 GB with headroom.** Empirical probe confirms:

- Config: `config/aimip/probe_t106/{suite,variant}_sfno_full.yaml` — T106, 32/2,
  curriculum `[[120, 1]]` (worst-case 120 h phase, 1 epoch), 2 tiny windows.
- Launch (Derecho, single 40 GB A100): see the probe command in the session /
  `train_sfno_full_scale.pbs` with `WB_SUITE=config/aimip/probe_t106/suite_sfno_full.yaml`.
- **Read:** if the first training step completes a forward+backward without OOM →
  option (b) viable (proceed with a curriculum T106 sfno_full for the fair set).
  If it OOMs → fall back to option (a) (weak-but-fair 32/2 no-curriculum).

**P1 RESULT (2026-07-18): PASS.** Ran on a 40 GB A100 interactively
(`JAX_ENABLE_X64=1 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_ALLOCATOR=platform`).
The 32/2 T106 sfno_full (23.7 M params) completed the full forward + 20-step
120 h checkpointed backward + optimizer update **without OOM**. So the fair-set
sfno_full carries the curriculum (option b). The probe's epoch-0 NaN is expected
(random init → 120 h rollout); the fair set uses the base short-start curriculum
(12→24→72→120 h, matching column_nn) and will likely need `aimip_lr: 5e-5`
(per-model, like the T63 run) + norm-stats to keep epoch 0 finite — confirm with a
short finite-check before the full run.

## Team alignment (gate before re-running)

Pierre pointed to the separate WB harness (Design B). Design A **retires the WB
campaign's NN training** in favour of eval-bridging AIMIP checkpoints — a change
from his ask. Confirm "same models, two scorecards" is the agreed deliverable
before re-running `sfno_full`.

## Sunk-cost summary

- **Kept:** `column_nn` (days), `classical` (running), the whole eval stack.
- **Redone:** `sfno_full` re-run once at T106 (the T63 run continues as a
  standalone D4 result).
- **New work:** relabel configs, run P1, optionally bridge `sfno_physics` into the
  WB2 eval, run the two evals per family.

## Related

`#976` (WB2 wind round-trip, fixed) · `#919` (WB2 eval driver) · `#1047` (60-sample
null) · `#1160` (WB-campaign scattered-data pre-flight) · `[[aimip-scale-training-setup]]`
· `[[wb-scale-training-derecho]]`.
