# LES-suite campaign — running jobs, output locations, and how to resume

This file tells a future session **where the campaign output lives on disk** and
**how to resume** the LES-truth-suite science campaign (LES_SUITE.md §7). The suite
infrastructure is complete (CHANGELOG.md); this is the compute campaign that fills in
the Q1/Q2/Q3 numbers. `results/` is **gitignored** — outputs live on the filesystem,
not in git; this doc is the committed pointer to them.

## Submitted job(s)
- **SLURM job 9157232** — `scripts/cluster/les_suite/les_suite_campaign.sbatch`
  (submitted 2026-07-22, branch `les-suite-optimization`; 1× V100S + 8 CPU, 2-day
  limit). Runs `scripts/cluster/les_suite/run_les_suite_campaign.sh`.
- Check status: `squeue -u ac5006` (look for `les_suit`). SLURM stdout:
  `results/les_suite/campaign/slurm-<jobid>.out`. Human progress log (per-step
  timestamps): `results/les_suite/campaign/progress.log`.

## What the campaign does (4 steps, idempotent)
1. **Emit the Q1 buoyancy-axis flux sweep** (GPU, ~7 min each): free-convective CBLs
   at surface fluxes Q0 ∈ {0.02, 0.04, 0.08, 0.12} K m/s (0.06 already emitted
   unlabeled ⇒ 5 flux points) → `results/les_suite/artifacts/cbl_nieuwstadt__lasd__q0_<Q0>.npz`.
2. **Q1a structural-ceiling sweep** (CPU, fast): counter-gradient diagnostic per flux
   → `results/les_suite/q1_counter_gradient_sweep.json` (list of {Q0, has_counter_gradient_layer,
   layer_base_m, layer_top_m, counter_gradient_fraction}, sorted by Q0). **This is the
   Q1a deliverable** — the flux at which a counter-gradient layer first appears is the
   structural ceiling on local closures.
3. **Q2 tuning campaign** (CPU, SLOW — ~1.5 h per (flux, closure) due to the
   recompile-per-candidate cost): derivative-free tier-1 tuning of 5 closures
   (smagorinsky, louis, holtslag_boville, ysu, mynn25) on every flux artifact →
   `results/les_suite/tuned/<artifact>__<scheme>__df.json` (each: default/best loss +
   best overrides).
4. **Scorecard** → `results/les_suite/scorecard.md` (Q2 per-regime ranking + Q3
   coefficient table). Regenerate anytime with
   `python scripts/validate/les_suite/build_les_scorecard.py`.

## Output map
| what | path |
|---|---|
| LES reference artifacts (self-describing .npz) | `results/les_suite/artifacts/*.npz` |
| Q1a structural sweep | `results/les_suite/q1_counter_gradient_sweep.json` |
| Per-(flux,closure) tuned results | `results/les_suite/tuned/*__df.json` |
| Q2/Q3 scorecard | `results/les_suite/scorecard.md` |
| Gate-0 (committed fixture) | `docs/atmosphere/les_suite/gate0_nieuwstadt_result.json` |
| Campaign progress log | `results/les_suite/campaign/progress.log` |

## How to RESUME (future session)
1. `squeue -u ac5006` — is job 9157232 (or a resubmit) still running?
2. `tail -50 results/les_suite/campaign/progress.log` — see the last completed step.
3. **The campaign is idempotent**: it skips any artifact / tuned JSON that already
   exists. To continue after a timeout/kill, just resubmit:
   `sbatch scripts/cluster/les_suite/les_suite_campaign.sbatch`.
4. Read the results: `cat results/les_suite/q1_counter_gradient_sweep.json` (Q1a),
   `cat results/les_suite/scorecard.md` (Q2/Q3). Record the numbers in CHANGELOG.md
   §"First real science outputs" and LES_SUITE.md §7.1.

## Known limits / next infrastructure work (see CHANGELOG.md)
- **Tuner recompiles per candidate** → the campaign uses a COARSE search (tier-1,
  n_random=3). The fix is the **AD path** (traced params via `apply_param_overrides`
  inside a jitted/AD loss — reuse the RCE trainer's `lax.scan` +
  `eqx.filter_value_and_grad` in `scripts/run/train_scm_rce_params.py`). That makes
  one compile serve all candidates (deeper tuning, minutes not hours) AND delivers
  D4's AD-vs-derivative-free comparison. Build this before scaling the tuning.
- **Only the dry-convective CBL regime is wired** in `run_les_suite.py`. The full
  Q1/Q2/Q3 answers need the stable (SBL), sheared-convective (add `--Ug`), and moist
  (BOMEX, DYCOMS) regime IC builders + emission, then re-run this campaign per regime.
  The Q3 inter-regime coefficient spread is 0 until ≥2 regimes are tuned.
- **σ_LES error bars (D7)** need the SGS-spread runs (each case × {lasd, smagorinsky,
  vreman}) + the 2×-resolution convergence runs — not yet in this campaign.
