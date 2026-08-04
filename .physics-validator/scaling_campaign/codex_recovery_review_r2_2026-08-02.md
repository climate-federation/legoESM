Reading additional input from stdin...
OpenAI Codex v0.145.0
--------
workdir: /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
model: gpt-5.6-terra
provider: openai
approval: never
sandbox: read-only
reasoning effort: xhigh
reasoning summaries: none
session id: 019fc226-2ada-7320-922e-df183bca9f5a
--------
user
Round-2 re-review after your FIX-FIRST (transcript: .physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md). Changes made since:
1. docs/performance/scaling/levante_campaign_2026-07-24.md — Phase-3 section rewritten: owned-vs-wet cells distinction (metis wet min/max 78000-102880 at np128), METIS claim scoped to 'this configuration on this lane', scale-out-not-rank-count wording, block:cyclic socket-distribution correction, UCX warn caveat, s9 executed padded cells 2621568, peak scoped to synthetic physics=none bench, floor 'consistent with' not confirmed, weak-pair claim RETRACTED loudly with the three confounds named. New sections: recovered atm128 receipt (LL2048@128 f32 5.58ms/39.11 GC/s vs @64 6.7324 job 26502539 same protocol, eff 0.60), hundreds-push job table.
2. scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch — v2: per-pid wait status collection, solo_pre AND solo_post brackets, steps 5000/warmup 100 (~60s stepping window vs launch skew), per-arm epoch echoes + end-of-job sacct step nodelist/Start/End table as overlap evidence, receipts-validity gate over all 6 files, confirm bar reworded to guaranteed >=3.64x solo.
3. scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch — s8 lloyd0 np8/16/32 matched to s9 protocol (steps 12/warmup 3, sfc, reorder-for 128) with falsifiability block.
4. scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch (LL2304@96/192 + LL2880@192 f32; divisibility rationale), atm_latlon2d_cpu_hundreds.sbatch (r512 2-D pencil np64-512 f64 moist, wall-pole labelled), prewarm_s10.sbatch (shared partition).
5. scripts/plot/plot_scaling_paper_figure.py — LL2048 series (64,6.73)/(128,5.58) + f64@128 scatter; mesh-family note on MPAS panel.
Verify each of your 12 items is addressed or explicitly deferred; check new scripts for bash/srun bugs (rc propagation, step resource flags, divisibility 2304/96, 2304/192, 2880/192, r512 pencil factorability at np512); check the remaining claims' arithmetic. End with 'VERDICT: SHIP' or 'VERDICT: FIX-FIRST' + numbered issues.
codex
I’ll trace the prior 12 findings against the revised artifacts, then independently audit the new job scripts and the remaining numerical claims.
exec
/usr/bin/bash -lc "pwd && rg --files -g 'AGENTS.md' -g 'codex_recovery_review_2026-08-02.md' -g 'levante_campaign_2026-07-24.md' -g 'mpas_s9_ensemble.sbatch' -g 'mpas_s8_lloyd0_ladder.sbatch' -g 'atm_latlon_hundreds.sbatch' -g 'atm_latlon2d_cpu_hundreds.sbatch' -g 'prewarm_s10.sbatch' -g 'plot_scaling_paper_figure.py' -g 'README*' -g 'pyproject.toml' -g 'pytest.ini' -g 'tox.ini' -g 'setup.cfg'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 377ms:
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
README.md
pyproject.toml
packages/ocean/README.md
packages/ocean/pyproject.toml
packages/core/README.md
packages/core/pyproject.toml
packages/atmosphere/pyproject.toml
packages/coupler/README.md
packages/coupler/pyproject.toml
packages/ice/README.md
packages/ice/pyproject.toml
packages/atmosphere/README.md
packages/tools/README.md
packages/tools/pyproject.toml
docs/dev-notes/README.md
scripts/cluster/scaling_levante/README.md
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch
scripts/cluster/scaling_levante/atm_latlon2d_cpu_hundreds.sbatch
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch
scripts/cluster/scaling_levante/prewarm_s10.sbatch
packages/ml/README.md
packages/ml/pyproject.toml
scripts/cluster/aimip_scale/README.md
docs/performance/scaling/levante_campaign_2026-07-24.md
packages/land/README.md
packages/land/pyproject.toml
scripts/cluster/scaling_derecho/README.md
scripts/cluster/derecho_lmip/README.md
scripts/README.md
config/4DVar_single/README.md
scripts/plot/plot_scaling_paper_figure.py
scripts/experiment/dino/README.md
config/templates/README.md
tests/validation/README_DYCORE_PROGRESSION.md
scripts/run/mpas_realistic_geometry/README.md
config/machines/README.md
scripts/validate/ocean_fidelity/dino_1226/README.md
config/aimip/wbcompare/classical_sweep/README.md

exec
/usr/bin/bash -lc "sed -n '1,260p' .physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md && printf '\\n--- DOCUMENT ---\\n' && sed -n '1,420p' docs/performance/scaling/levante_campaign_2026-07-24.md" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 246ms:
Reading additional input from stdin...
OpenAI Codex v0.145.0
--------
workdir: /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
model: gpt-5.6-terra
provider: openai
approval: never
sandbox: read-only
reasoning effort: xhigh
reasoning summaries: none
session id: 019fc214-68e7-7b63-b44a-89f0f7351ad0
--------
user
Adversarial review of a scaling-campaign iteration. Attack arithmetic, logic, confounds, and overclaim. Context: docs/performance/scaling/levante_campaign_2026-07-24.md, new final section 'Phase-3 receipts recovered after the 2026-07-31 session drop (2026-08-02)', plus the edit to scripts/plot/plot_scaling_paper_figure.py (git diff HEAD -- scripts/plot/plot_scaling_paper_figure.py) and the new scripts/tmp/mpas_s9_ensemble.sbatch.

Claims to attack:
1. METIS A/B (job 26600094, all arms 5120 cells/rank, f64, 32 ranks/node): A s7np32 geometric 189.82ms, B s8np128 geometric 308.96, C s7np32 metis 191.99, D s8np128 metis 333.39, E s8np128 metis block:block 537.91. My claims: rank-count term geometric 1.628 vs metis 1.736 -> partition-quality lever DEAD/refuted; block:cyclic mandatory (E/D=1.61); lever#2 closed refuted on this lane.
2. s9 GPU ladder (job 26600095, f32 sfc lloyd0, 2621442 cells L26): np32 12.47ms/5.47GC/s, np64 9.60/7.10, np128 11.48/5.94. Claims: new MPAS peak 7.10 GC/s = 2.2x s8 best 3.23 GC/s (s8 655362 cells, np64 5.27ms, 26 levels -> check GC/s arithmetic); 64->128 anti-scales at 20.5k cells/GPU consistent with ~30k floor; weak pairs s8->s9 at matched cells/GPU: 6.92->12.47 (0.55), 7.10->9.60 (0.74), 8.13->11.48 (0.71) -> ~1.4x matched-tile cost per 4x ranks = GPU rank-count term.
3. Ensemble job design (mpas_s9_ensemble.sbatch): 4 concurrent 32-GPU srun steps in one 32-node allocation, per-step SLURM_STEP_NODELIST coordinator autodetect, shared jobid-derived port claimed safe because hosts differ; solo control in-job; CONFIRM bar max(replica)<=1.10x solo. Attack: step isolation, GPU binding, srun flag errors, cache effects, whether solo-then-concurrent ordering biases, whether 12 steps is enough discrimination.
Check every ratio and GC/s conversion. Flag any claim that should be labelled PLAUSIBLE rather than CONFIRMED, any protocol mismatch making a comparison confounded, and any sbatch bug. End with 'VERDICT: SHIP' or 'VERDICT: FIX-FIRST' plus a numbered issue list.
codex
I’ll audit the receipts and scheduler script against the stated claims, recompute every metric, and look for comparison and isolation confounds.
exec
/usr/bin/bash -lc "pwd && rg --files -g 'levante_campaign_2026-07-24.md' -g 'plot_scaling_paper_figure.py' -g 'mpas_s9_ensemble.sbatch' -g 'AGENTS.md' -g 'CLAUDE.md' | sort && git status --short && git diff --check HEAD -- scripts/plot/plot_scaling_paper_figure.py && git diff HEAD -- scripts/plot/plot_scaling_paper_figure.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 676ms:
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
CLAUDE.md
docs/performance/scaling/levante_campaign_2026-07-24.md
scripts/plot/plot_scaling_paper_figure.py
 M docs/performance/scaling/levante_campaign_2026-07-24.md
 M scripts/plot/plot_scaling_paper_figure.py
?? .physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md
diff --git a/scripts/plot/plot_scaling_paper_figure.py b/scripts/plot/plot_scaling_paper_figure.py
index 93056ac19..a67ca53a8 100644
--- a/scripts/plot/plot_scaling_paper_figure.py
+++ b/scripts/plot/plot_scaling_paper_figure.py
@@ -33,7 +33,8 @@ from matplotlib.lines import Line2D
 SOURCES = {
     "atm_latlon": "26450848/26453240/26449147 (f32), 26494902 (f64)",
     "atm_cube": "26452894/26453782",
-    "atm_mpas": "26454476/26454618/26486288/26493638/26493734",
+    "atm_mpas": "26454476/26454618/26486288/26493638/26493734, "
+                "s8 np32-128 26549646/26538474, s9 26600095",
     "atm_ico_cpu": "26495083 (f32), 26495437 (f64) — both block:cyclic",
     "oc_latlon": "26460444-501/26460365/26493592",
     "oc_tripole": "26493837/26493648",
@@ -56,10 +57,13 @@ PANELS = [
         note="f64 pending",
     ),
     dict(
-        key="atm_mpas", title="MPAS icosahedral", sub="L8 28 km L26 · A100 NCCL",
-        series=[("float32", [(2, 19.90), (4, 14.12), (8, 6.92), (16, 7.10)]),
-                ("float64", [(2, 38.34), (4, 20.09), (8, 18.98)])],
-        note="incl. fusion fix",
+        key="atm_mpas", title="MPAS icosahedral", sub="subdiv-8/9 L26 · A100 NCCL",
+        series=[("float32 (subdiv-8)", [(2, 19.90), (4, 14.12), (8, 6.92),
+                                        (16, 7.10), (32, 8.13), (64, 5.27),
+                                        (128, 6.47)]),
+                ("float32 (subdiv-9)", [(32, 12.47), (64, 9.60), (128, 11.48)]),
+                ("float64 (subdiv-8)", [(2, 38.34), (4, 20.09), (8, 18.98)])],
+        note="s9 weak eff 0.55–0.74\nvs s8 at matched tile",
     ),
     dict(
         key="atm_ico_cpu", title="icosahedral", sub="subdiv-7 L26 · Milan CPU–MPI",
@@ -102,11 +106,13 @@ COLORS = {"float32": "#0072B2", "float64": "#D55E00",
           "float64 (packed)": "#E69F00",
           "mixed (f64 store)": "#009E73",
           "float32 (C768)": "#0072B2", "float32 (C384)": "#56B4E9",
+          "float32 (subdiv-8)": "#0072B2", "float32 (subdiv-9)": "#56B4E9",
           "float64 (subdiv-7)": "#D55E00", "float64 (subdiv-8)": "#E69F00"}
 MARKERS = {"float32": "o", "float64": "s", "mixed (f64 store)": "D",
            "f32 · LL1536/2048 @64": "*",
            "float64 (packed)": "s",
            "float32 (C768)": "o", "float32 (C384)": "^",
+           "float32 (subdiv-8)": "o", "float32 (subdiv-9)": "^",
            "float64 (subdiv-7)": "s", "float64 (subdiv-8)": "v"}
 
 

exec
/usr/bin/bash -lc "sed -n '1,240p' CLAUDE.md && printf '\\n--- campaign tail ---\\n' && tail -n 320 docs/performance/scaling/levante_campaign_2026-07-24.md && printf '\\n--- ensemble candidates ---\\n' && (rg --files --hidden --no-ignore | rg '("'^|/)mpas_s9_ensemble'"\\.sbatch"'$|mpas.*ensemble.*'"\\.sbatch"'$'"' || true) && printf '\\n--- plot remainder ---\\n' && sed -n '1,280p' scripts/plot/plot_scaling_paper_figure.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 256ms:
# legoESM Claude Memory

## Role
Senior JAX+ESM dev. Skeptical, verify-first. Optimize: correctness, physical consistency, differentiability, maintainability. Prefer `opusplan`/`opus` high effort for dycore/physics/parallel/debug. Fast mode off.

## Repo Facts
- Differentiable ESM in JAX: atm, ocean, land, sea ice, coupler, DA, ML.
- End-to-end `jax.grad` compat = goal. Never break autodiff/JIT/pytree.
- Mass conservation hard. Energy/momentum when scheme permits.
- Parallel entry: `ParallelRuntime.create()`.
- Grids: cubed-sphere, lat-lon, Gaussian/spectral, Voronoi/MPAS, icosahedral.

## Training (`src/legoesm/training/`)
- 3 modes: physics param tune, neural GCM, SFNO+dycore.
- All: `build_segment_fn(...).raw` (non-JIT, non-donating) inside `eqx.filter_value_and_grad`.
- `SegmentForcing` = explicit arg to `run_segment` (not closure) → prevents recompile.
- `TrainablePhysicsParams` wraps 8 params, Equinox module, sigmoid constraints.
- ERA5: `era5_to_state.py` lat-lon → grid, Zarr cache.
- Losses: `training/losses.py` imports `ml/loss.py`. No dup.
- **MPI AD**: `global_sum_mpi` (allreduce SUM) full VJP. MPI halo: `_sendrecv_vjp` custom_vjp. `fix_mass`/`zero_mean_tendency` flow grads via global reductions. `global_max_mpi`/`global_min_mpi` NOT diff — keep out of losses.

## Operating Mode
- Nontrivial task: short plan before edit. Read nearby impl+tests first. Ambiguous numerics/physics/API: ask.
- Minimal diffs. No unrelated refactor in bug fix.
- **Codex adversarial review MANDATORY after any major code implementation/change.** Trigger: new module/feature, dycore/physics/parallel/ocean/land/ice/coupler/training edit, >~50 LOC, multi-file, or anything touching numerics/AD/JIT/pytree/conservation. Run the **iterate-with-codex agent** loop below (`/codex:adversarial-review --wait` → fix flagged → `/codex:review --wait` → repeat until clean or 30 iter) BEFORE declaring done; report that review ran + verdict.
  **If the review SUBAGENT dies (spend limit, API error), that is NOT a review
  waiver — the codex CLI is a separate binary with separate credentials and is
  usually still reachable: `codex exec --sandbox read-only -C <repo> "<prompt>"`
  (`which codex`, `~/.codex/auth.json`). Try the CLI directly before ever
  proceeding unreviewed, and if BOTH are unavailable say "UNREVIEWED" in every
  status until one succeeds.** 2026-07-26: a subagent hit a monthly spend limit
  and many iterations ran unreviewed while the CLI worked fine the whole time. Exempt: trivial/mechanical edits (typo, comment, rename, doc/markdown/`.tex`-only, single config value).
- **Pre-impl search mandatory**: before new fn/helper/class/operator/diagnostic/init/load/loss/numerical routine, grep `src/legoesm/` for similar names/docstrings/formulas in `thermo.py`, `constants.py`, `eos.py`, `ml/loss.py`, `diagnostics/`, `core/`, `atmosphere/physics/_shared.py`. State searched+found. Similar exists → extend/factor.
- **Shared utilities — never re-derive** (prod, scripts, validators, plotters, tests, notebooks, probes):
  - Constants: `from legoesm import constants` → `T_freeze`, `R_d`, `c_pd`, `L_v`, `R_v`, `epsilon`, `g`, `p_ref`, `kappa`, `sigma_sb`, `T_freeze_ocean`. No literals `273.15`/`287.0`/`1004.64`/`2.501e6`/`461.51`/`0.622`/`9.80616`/`6.371e6`/`7.292e-5`.
  - Saturation: `from legoesm.thermo import saturation_vapor_pressure, saturation_mixing_ratio, saturation_mixing_ratio_ice`. No re-impl Tetens/Magnus/Clausius–Clapeyron (plotters incl). Why: re-derived `e_sat=611.2*exp(17.67*Tc/(Tc+243.5))` diverged from model → false supersat in CI.
  - Column integrals: `legoesm.diagnostics.column_integrals` (`column_water_vapor`). No inline `jnp.sum(q*p_s*dsigma)/g`.
  - Losses: `ml/loss.py` (`area_weighted_mse`, `spectral_loss`, `per_variable_mse`).
  - Optimizer: `ml/training.create_optimizer()` (warmup+cosine+clip).
  - SCM-RCE gradient tuning: reuse `scripts/run/run_scm_rce_campaign.py` for CRM
    reference extraction / SCM evaluation and `legoesm.training.scm_rce_metrics`
    for the normalized profile score. No duplicated RCE profile numerics.
  - SCM-RCE param training defaults to MUON via `ml.training.create_optimizer()`,
    initializes from `results/scm_rce_campaign/tuned_parameters.json`, writes a
    recommended trained JSON under `results/`, and never mutates production
    `*Config` defaults. Apply trainable overrides inside the loss so leaves are
    traced; static frozen leaves stay outside.
  - Every new `.py`, including `scripts/run/*.py` drivers, gets a direct test.
    Scheme/factory dispatch must raise on unknown selections.
  - Atm column (h, ρ, virtual T): `atmosphere.physics._shared`.
  - Ocean EOS/pressure: `ocean.eos` (`compute_ocean_rho`, `compute_ocean_rho_and_pressure`).
  - SFNO: `ml/sfno.py`. No new neural op archs in training.
  - Channel packing: `ml/channel_packing.py` (`PE3DChannelSpec`, `pack_pe_state`, `unpack_pe_output`).
  - Ocean baroclinic (#214): `ocean/dynamics/ocean_tendency_common.py` (`iterate_eos_and_pressure_anomaly`, `apply_sponge_tracer_relaxation`, `apply_freshwater_virtual_salt_top`, `implicit_bottom_drag_factor`) in new `ocean_pe_*.py`.
  - Ocean barotropic (#214): `ocean/dynamics/barotropic_common.py` (`compute_filter_weights`, `bebt_blend`, `maxvel_clip`) in new `barotropic_*.py`. `tests/ocean/unit/test_no_scheme_duplication.py` enforces.
  - Plotters NOT exempt. Use model helpers for q_sat, RH, ρ, virtual T, MSE.
- No duplicate numerics across dycores/physics/grids/tests. Indexing/naming-only copy-paste forbidden.
- **No laziness on hard/large code** (>100 LOC, multi-component, full operator chains): no `pass`/`NotImplementedError` stubs, no partial-called-done, no skip edge cells/boundary halos/corner stencils/non-duogrid/MPI-sharded/AD-VJP. No happy-path-only tests. Too big → say so, list remainder, quantify risk.

## Attribution Gates — MANDATORY, each from a real 2026-07 failure
Model is near operational. Every rule below is mechanical: satisfy it or state
explicitly that you did not. "I was careful" is not compliance.

- **PROVE THE PATH EXECUTES before blaming a line.** Naming a file:line as the
  cause requires showing that line runs in the configuration under test: print
  the ENCLOSING FUNCTION (`awk` the nearest `def` above it) and confirm the
  active lane/driver calls it. FAILURE: blamed the positivity clamps at
  `model_driver.py:10923` for the century's water source; they live in
  `_run_per_step` while the century runs `_run_mpas`, which contains no
  moisture clamp at all. A fix was nearly written for a lane the run never
  touches. Same class as reading an entry point instead of the full path.
- **REUSING A REFERENCE IMPL MEANS PORTING ITS EXCLUSIONS, not just its
  formula.** State which of the reference's guards/scope conditions you kept
  and which you dropped, with a reason for each. FAILURE: copied
  `spectral_les_moist.conserving_positive` but not its `n_water` split, so the
  column-conserving borrow was applied to number concentrations
  (`N_c`/`N_i`/`N_r`) — unphysical, and it fed M2005 deposition (~N_i^(2/3)),
  producing a fake "accelerating dry bias" that was reported before being
  caught.
- **A TEST THAT INSPECTS SOURCE MUST NAME THE SYMBOL THAT RUNS, and must be
  shown to FAIL when the feature is removed.** An `inspect.getsource(X)`
  assertion where X is a delegating wrapper passes while proving nothing.
  FAILURE: asserted against `MPASPrimitiveEquationModel.step`; the floors are
  in `_step_jit`.
- **TOOL STATUS IS NOT EVIDENCE — read the output tail.** An exit code without
  the tool's own success line (pytest's `N passed`, "COMPLETED in Xs") is
  UNVERIFIED; OOM kills and timeouts can surface as success. FAILURE: reported
  a regression suite green on exit-0 that was actually `Out Of Memory` mid-run.
  Quote the decisive line when claiming a suite passed.
- **EVERY BASELINE/ALLOW-LIST REASON STRING IS A CLAIM — verify it in code
  before writing it.** A plausible-sounding reason permanently hides a real
  defect. FAILURE: classified `convective_buoyancy_death_memory` as "carried in
  SegmentCarry" (it is not — the leaf `BechtoldConfig.buoyancy_death_memory`
  exists and nothing maps to it), asserted an `SBMConfig.precip_efficiency`
  leaf that does not exist, and credited `micro_substeps` to a consumer that
  reads `args.`, not the config field.
- **RATE / TENDENCY / SKILL COMPARISONS: identical windows on BOTH sides, and
  print the window next to the number.** Differing spans is a confound, not a
  result. FAILURE: TCW over days 190-530 vs CMOR year 1 gave "+38.7 kg/m2/yr";
  matched windows gave +13.4. Extends the existing controlled-comparison rule
  to derived rates.
- **A DIAGNOSTIC'S PRINTED PRECISION BOUNDS THE RATE YOU CAN CLAIM.** Log CWV
  at 0.1 kg/m2 over 8 days resolves only ~±4.6 kg/m2/yr — do not report a
  trend inside one quantum. Prefer fp64 from model state (checkpoints) over
  parsed log lines. Same class as the throughput-quantization error.
- **`JAX_ENABLE_X64=1` on any numerics/conservation test.** An fp32 mismatch is
  NOT a failure until re-run with x64; and a *new* failure is not yours until
  reproduced with your change stashed. Do both before reporting a regression.
- **RUN-TARGET PARAMS ARE ABSOLUTE (`TARGET_DAYS`), and "latest checkpoint"
  MOVES.** For a controlled pair, COPY the pinned checkpoint into each arm dir;
  never use a `PREV_CKPT_DIR`-style newest-wins pointer while another run is
  advancing. FAILURE (twice): arms exited instantly at "Already at/past
  target".
- **A LAUNCHER FLAG THAT SWITCHES ONE FORCING CHANNEL MUST SWITCH ALL OF
  THEM.** Verify the resolved paths in the run log, not the flag you passed.
  FAILURE: `CENTURY_DECK=1` set era-correct ozone+volcanic but left 1979-2016
  SST.

## JAX
- Pure pytree fns. `lax.scan` time integration. `vmap`/batched arrays over Python loops on array dims. `jnp.where`/`lax.cond`/`fori_loop`/`scan` not Python control flow on traced.
- **Feature gating exception** (`fix_mass`, `fix_moisture`): Python `if` on static bool in closure — NOT `jnp.where` (traces both branches). `jnp.where` only for data-dependent traced selection.
- Stable shapes. No retrace. Dtype: spectral=x64+complex128; finite-volume can float32.
- No host/device thrash, NumPy in traced code, hidden non-JAX side effects.
- **Buffer donation + `jax.grad`**: `donate_argnums` conflicts reverse-mode AD. JIT fn inside `jax.grad`/`eqx.filter_value_and_grad` → provide non-donating variant (`.raw`). See `build_segment_fn`.
- **Closures vs explicit args**: closure captures = compile-time consts. Per-iter changing val (SST, solar) → pass as traced arg. See `SegmentForcing`.

## Earth System
- Conservation, metric consistency, staggered-grid consistency, halo correctness = first-class.
- No silent clip/damp/coerce unless justified+validated.
- Preserve units, sign conventions, monotonicity/positivity, hydrostatic/nonhydrostatic.
- Cubed-sphere/curvilinear: assume edge+metric errors first.
- Physics coupling: column closure + consistent flux signs.
- DA/diff: preserve smoothness. No gratuitous nondiff.
- **Sign-convention check MANDATORY on every equation/flux/tendency edit.** Before declaring done on any code touching a PDE term, flux, source/sink, BC, or budget update: (1) state the coordinate convention in scope (z up/down, flux positive-up/down/into-body) as a comment at the term; (2) walk EACH term and confirm its sign matches that convention — gravity vs capillary/diffusion divergence, top vs bottom BC, source vs sink, the `±` in `state = state ± dt·tend`; (3) confirm the budget closes (`in − out − Δstorage = 0`) and exchanged fluxes carry the SAME sign at both ends of a coupling (land runoff `+into ocean` must arrive `+into ocean`). A comment label and the code must agree — a flux commented "upward" computed as downward is a defect, fix the label or the math. Mechanical gate where feasible: a sign/conservation unit test (analytic column, manufactured solution, or `assert` budget residual ≈ 0) — passing norms alone never certify a sign is right (a flipped flux can still be small). Common flips: z-axis direction, evap positive-up vs moistening, brine/salt vs freshwater dilution, stress atmospheric vs ocean convention (`-tau`), free-drainage vs gravity double-count.

## Parallel/HPC
- Correctness across serial/multi-device/MPI/hybrid.
- Sharded: reason about halo exchange, reductions, partition specs, global invariants.
- Validate single-rank → smallest distributed.
- Backends differ (Metal/GPU/CPU/spectral/MPI). Apple Silicon: spectral on CPU.
- **MPI**: `initialize_distributed(global_n=N)` → `scatter_to_local()` → rank-local step → `gather_to_global()` for I/O only. Never full global per rank.
- **4D halo**: `pad_halo_4d()`+`pad_halo_vector_4d()` all vert in one msg. All 3D ops in `operators_3d.py` use 4D. Never `vmap(pad_halo)`.
- **MPI halo AD**: all `sendrecv` via `_sendrecv_vjp` (`@jax.custom_vjp` in `halo_exchange.py`). Only `allreduce(SUM)` AD-safe; `MAX`/`MIN`/`allgather`/`bcast` = diagnostics only.
- **Device mesh under MPI**: per-rank count to `create_device_mesh()`, not total.

## Oracle-Recipe Fidelity (ocean) — see docs/ocean/fidelity/oracle_recipe_strategy.md
- ADDITIVE to Validation Rules: oracle work NEVER replaces unit tests, the ocean matrix, conservation checks, or visual verification. Truth tiers (conservation/equivariance/analytic) outrank oracle-matching.
- Recipe = pure config selecting shared canonical blocks (never a bespoke `veros_*` solver). Oracle-matching numerics go in the canonical module (`eos.py`, advection/limiter dispatch, `vertical_mixing/`, integrator dispatch) as selectable options.
- Mimicry-only glue (halo strip, axis transpose, time-level handling) lives in the fidelity harness, never the model. Test: "would a user with a different goal ever select this?" No → harness.
- Conventions handled only in the bridge, verified by equivariance tests (`physics(φ(x))=φ(physics(x))` to tol); a "convention" that changes the wet domain/answers is physics → config, not bridge.
- Constants are config (`ConstantsConfig`), not module-global monkey-patches (no `override_constants` in shippable paths); defaults reference `legoesm.constants`; base only, derived (κ,ε) recomputed.
- Oracle tendency-match (tier 3) trusted only for a block that also clears truth tiers (0–2). MACHINE-ENFORCED (#388 Ask#4): `ocean/fidelity/precedence.py::evaluate_precedence` LOCKS oracle tiers (≥3) on any truth-tier (0–2) failure; surfaced + exit-gated by `scripts/validate/ocean_fidelity/build_fidelity_scorecard.py` (the one generated scorecard).

## Recipe×Setup template adapters (#388)
- **One shared selector, never re-implemented.** Every component YAML experiment adapter exposing a `setup:` block (atmosphere `Config`, ocean `OceanExperimentConfig`, sea-ice `SeaIceExperimentConfig`, …) MUST route validation + command-building + signature through `legoesm.core.setup_selector` (`MatrixRunnerSpec` + `validate_setup`/`build_matrix_command`/`setup_signature`/`require_positive_finite`). Per-component CLI differences (case flag `--only`/`--test`, exact-match `exact_prefix`, which `--levels`/`--dt`/`--days`/`--resolution` flags exist) are a `MatrixRunnerSpec`, NOT a copy-pasted `parts=[...]` builder or private `_SETUP_KEYS`/`_validate_setup`. Component-specific case-name validation goes through the `known_names=` kwarg. A re-implemented selector is REJECTED in review. Factory naming: `_<component>_matrix_spec()`. FOLLOW-UP: dedup `OceanExperimentConfig`'s inline selector onto the shared helper once PR #465 + the shared-helper PR both land on main (the inline copy predates the extraction).
- **Every setup template's `(name, grid)` MUST be a real matrix case**, enforced by a catalog-backed test (`run_<comp>_test_matrix._build_test_matrix()` → assert the pair exists) so `--test/--only =<name> --grid <grid>` never selects nothing. New template without this gate → REJECTED.
- **An exact case selector matching nothing is a hard error** (`raise SystemExit`) in every matrix-runner mode — never a silent no-op (dispatch-hardening). MPI-safe: key the guard off the global pre-slice match count; empty-slice ranks fall through to the barrier.
- **New `experiment_registry` mode → same-PR `get_adapter` test** in `test_experiment_registry.test_get_adapter_resolves_classes` (a template test importing the class directly does NOT cover the dispatch).
- Land has no standalone idealized-case surface (no registry/matrix runner) → recipe×setup is N/A there; do not invent one to match the pattern. Truth-tier precedence stays ocean-scoped per `oracle_recipe_strategy.md`; use `TRUTH_TIERS`/`ORACLE_TIER_FLOOR` by name (no hardcoded `3`).

## Validation
- Narrowest test after edits. Numerical changes: analytical/benchmark > unit tests alone. `JAX_ENABLE_X64=1` unless float32/Metal task.
- Dycore: Williamson, Galewsky, Jablonowski-Williamson, DCMIP, Held-Suarez, ocean benchmarks.
- Conservation/reductions/coupler: mass+energy diagnostics.
- Parallel: unsharded vs sharded, single-rank vs MPI.
- Too expensive: say what ran/didn't, residual risk.
- **CRITICAL — Controlled comparison: change ONE variable, hold the eval protocol FIXED to the baseline.** To claim a change (resolution, params, scheme, days) improved/degraded/"is comparable" vs a prior result, keep EVERYTHING else byte-identical to that baseline: forcing data + its sampling (years/days/hours/climatology), evaluation grid, metric definition (bias vs RMSE), region masks, timestepping. A metric that moved because the protocol/sampling changed is a **CONFOUND, not a result** — NEVER compare a number computed on one sampling/grid/metric to a number from another and call the difference an effect. If a resource limit (network cap, compute, time) forces a lighter or different sampling, **re-run the BASELINE at that SAME sampling before comparing** — a fresh baseline is cheap insurance; a confounded claim is not. Before writing "improved"/"degraded"/"better"/"comparable" vs any earlier number, explicitly confirm the two configs differ ONLY in the variable under test; if you cannot, say so and claim NO direction. Report the full config (data+sampling, grid, days/steps, params, metric) next to every number so the reader knows exactly what is being compared. Assuming two runs are comparable when the setup drifted is the error that turns "we improved it" into "we degraded it."
- **CRITICAL — Precision gate for comparison/skill/causal claims (do NOT be sloppy — 2026-07-20 EC-site lesson).** (a) **Harness self-check FIRST**: before reporting a NEW scheme's skill vs a validated baseline, run the KNOWN baseline through your OWN harness and confirm it reproduces the baseline's established/published number (±small tol). If your harness scores a validated scheme wildly off — e.g. two-leaf H looked "broken" (NSE≈−1) when the paper figure tracks obs — the HARNESS is wrong; fix it BEFORE any new-scheme claim. (b) **Match the reference metric EXACTLY**: metric aggregation (daily-mean vs half-hourly point-wise NSE), window (multi-year JJA vs one summer), masks (valid-forcing) are ALL part of the protocol. Report the SAME metric the reference used, plus any alternative, WITH the sample N. A single-window point-wise metric is NOT a skill verdict, and never rank schemes by a metric that punishes one scheme's known artifact (point-wise scatter/spikes) while ignoring the dimension of interest (mean diurnal shape). (c) **Never let a QC mask flatter one side**: if you drop a scheme's unphysical spikes from scoring, report BOTH the failures-penalized primary score AND the separately-labeled plausible-only sensitivity score. (d) **Do NOT INFER causal origin — INSTRUMENT it**: "the spikes come from X" requires LOGGING X and the alternatives (raw vs intermediate vs final), not deduction from reading one code path; "correctly hooked up" requires reading the FULL path, not the entry point. (e) **Label every claim CONFIRMED (evidence shown) vs PLAUSIBLE (inferred)** — adversarial review WILL refute over-confident inferences; state uncertainty up front rather than presenting a verdict table that a five-minute check overturns.
- **CRITICAL — Visual verify spatial/grid artifacts**: passing tests+norms NECESSARY ≠ SUFFICIENT for cubed-sphere ops, halo exchange, diffusion coeffs, grid metrics. Edge artifacts/cube imprint/grid-scale noise only detected visually (v-wind W2, wind_speed W5). Run `--only sw --grid cubed_sphere --quick` + inspect PNGs vs baseline. Norms can improve while artifacts worsen. Never claim "tests pass, edge fixed" from pytest alone.
- **Diffusion sensitivity**: div damping + hyperdiff AMPLIFY halo errors at cubed-sphere face boundaries. Check W2 v-wind visually when touching `_hyperdiff_cube`, `_div_damp_cube`, diffusion params.
- **Visual-regression gate (cube imprint)**: `scripts/validate/visual_regression.py --check` numericises the W2 v-wind cube-imprint check (SSIM + per-panel perceptual hash + edge-artifact ratio vs tiny committed ref in `tests/visual_baselines/`). Deterministic metric math gated in CI (`tests/test_visual_regression_metrics.py`); full cube-SW `--check` runs as a NIGHTLY non-blocking CI job until tolerances are calibrated across CI hardware. Tiny numeric baselines (.npy+json) ARE tracked — the one carve-out to "no tracked visual baselines".
- **CRITICAL — VALIDATE THE INSTRUMENT BEFORE QUOTING ITS NUMBER (2026-07-25 duogrid lesson: 8 confident claims, all retracted).** A diagnostic script is UNTRUSTED CODE until it passes its own controls. Never state a finding — never write "measured", "confirmed", "proven", "VERDICT" — from a probe's first output. **Before quoting any diagnostic number, run these five checks and say in the message that you ran them:**
  1. **Right conserved/invariant quantity?** Budget what the SYSTEM conserves, not a convenient proxy. (Failed: reported "vertex creates energy" from **KE alone** — KE is NOT conserved in shallow water, it trades with PE. Total `E=∫area(½h|V|²+½gh²)` reversed the sign of the conclusion.)
  2. **Same transform / units / staggering on BOTH sides?** Two "A-grid winds" from different operators are DIFFERENT QUANTITIES. (Failed: ours `c2l_ord2` vs oracle `C2L_ORD=4` — the SAME raw state gave 1.96e-2 vs 5.79e-2, a 3× swing that WAS the reported effect. Also: never budget across a stage boundary where the state changes representation — mid-step FV3 winds are in circulation form, which produced ±5.6e10 garbage.)
  3. **Same time, resolution, config?** Index by MATCHED TIME, not frame number. (Failed: mapped day→frame as `round(day)-1` against an HOURLY file, comparing our day-1 to their hour-1; and quoted a **C12** wedge gain (~300×) as the mechanism for a **C48** instability, where it is ~124×.)
  4. **Is the metric measuring what its name says?** Prove it on a synthetic case with a KNOWN answer before use. (Failed: called `mean|f−4-neighbour-mean|` a "2Δx grid-scale" measure — it is a high-pass/curvature residual that a merely sharper SMOOTH feature reproduces. Failed: a "gain" probe that re-filled a FIXED source, which is trivially 1.0000 by construction.)
  5. **Can the reduction support the claim?** `max` over tiles/corners/components taken independently per run can peak at DIFFERENT physical locations; a max-of-per-tile-means is not a global mean. Keep argmax metadata and map to a common physical location before claiming "localized".
  Plus: **diff ARRAYS, never printed summaries** (claimed "bit-identical ⇒ deterministic, not chaos"; the arrays actually differed by 9e-6 — only the rounded printout matched). **Never let a probe print its own verdict** ("=> the growth is REAL") — the interpretation belongs in the analysis after the controls pass, not baked into the tool where it gets echoed back as evidence. **`nanmax`/`nanmean`/`nansum` hide failures** — make NaN and missing frames FATAL. **Record every effective flag, env var and git SHA in each artifact**; a default `--n 36` silently mis-slicing a C48 file runs fine and lies.

--- DOCUMENT ---
# Levante weak/strong scaling campaign — 2026-07-24 (first hardware receipts + fixes)

One-day campaign turning the authored-but-never-run Levante job set
(`scripts/cluster/scaling_levante/`) into measured curves for every grid,
fixing what broke, and moving the worst axis (ocean strong scaling) to a
measured 2× improvement. All receipts on the post-merge tree `d3ec1ccce`+
(campaign branch `worktree-scaling-campaign`); job IDs cited throughout are
Levante SLURM jobs from 2026-07-24. Codex adversarial review: 4 rounds
(transcripts under `.physics-validator/scaling_campaign/`); every
measurement claim below carries the round-3 corrections.

Machines: Levante `gpu` partition (4× A100-80 SXM NVLink/node, IB HDR200),
`compute` (2× AMD Milan 7763). All GPU multinode = route-B
(`jax.distributed` + NCCL over IB verbs — `NET/IB mlx5` confirmed in-log;
route-A CUDA-aware mpi4jax not exercised on Levante).

## Headline results (strong scaling, f32 unless noted)

| Axis | Ladder | Result | Job(s) |
|---|---|---|---|
| Atm lat-lon LL720×1440 L26 | 4→8→16 A100 (1→4 nodes) | 7.72→5.40→3.54 ms/step, monotone; np16 = 7.6 GC/s (477 Mc/s/GPU sustained) | 26450848/26453240/26449147 |
| Atm MPAS ico L8 (28 km) L26 | 6→16 A100 | 8.66→7.08 ms/step; np16 = 2.41 GC/s — 1.6x the Derecho 16-A100 aggregate reported in `derecho_levante_sota_review_2026-07.md` SS3b (route-A, eff ~0.38 @16); CROSS-MACHINE, different stack/date - indicative, not a controlled A/B | 26453240/26449147 |
| **Atm cube C768/L60 (same-path cs-spmd)** | 6→24 A100 | 58.35→14.09 ms/step = **4.14× = eff 1.04 (at ideal)**, 15.1 GC/s (629 Mc/s/GPU) | 26453782 |
| Atm cube C384/L60 (same-path cs-spmd) | 6→24 A100 | 15.44→8.81 ms/step = 1.75× (eff 0.44), 6.0 GC/s | 26452894 |
| Atm cube C192/L60 (same-path) | 6→24 | 6.20→6.80 ms — ANTI-scales (eff 0.23): 9.2k cols/GPU is below the ~30k-column floor | 26452979 |
| Ocean lat-lon LL576×1152 L20, production implicit | 4→8→16 | 15.6→17.2→17.4 ms — anti-scales across nodes | 26452743-45 |
| Ocean same, improved (wide-halo + vmix-f32) | 4→8→16 | 12.8→11.1→8.6 ms — monotone, **2.01× at 16 GPUs** | 26452804-06, 26453279 |
| Cube tiled >6-GPU lane (its own bench) | 24 A100, C384/L60 closed loop | 9.01 ms/step, 5.9 GC/s, 18.2 SYPD — first >6-GPU production-lane receipts | 26450938/26452632 |

Single-node GPU (job 26445836): cube C192 gray_sbm strong eff 0.84–1.07
(1→3 A100); C48/C96 latency-floored. Ocean single-node solver ladder: below.

**THE cube strong-scaling result** — read 1.04 as "at ideal", NOT "better
than ideal": efficiency slightly above 1 is expected when the BASE leg is
per-device disadvantaged (np6 holds 4x the working set per GPU of np24, so
part of the 4.14x is cache/occupancy recovery rather than parallel
efficiency). The claim is that comm does not degrade this ladder, not that
parallelism is free. (same code path, harness, config, IC;
only the tile size varies — jobs 26452979/26452894/26453782): 6→24 GPU
speedup 0.91× / 1.75× / 4.14× at 9.2k / 36.9k / 147k columns per GPU. The
"poor cube strong scaling" of the earlier receipts TRACKS TILE SIZE:
holding code path, harness, config and IC fixed and varying only the tile,
efficiency goes 0.23 -> 0.44 -> 1.04, so tile size is SUFFICIENT to recover
ideal scaling at 24 A100 across 6 nodes. (That shows comm does not degrade
the ladder at production tiles; it does not prove comm costs nothing at
small tiles — that needs a per-phase profile.) Production rule confirmed:
keep >~30k columns/GPU.

Tiled-lane size sweep at fixed 24 GPUs (own bench, closed loop, CFL-scaled
dt; jobs 26450938/26452632/26453645): 1.87 / 5.89 / 14.34 GCells/s at
C192/C384/C768 = 78 / 246 / 597 Mc/s per GPU — per-device throughput still
climbing at 8.85M cells/GPU, so an A100 is not saturated even there.

## Ocean strong-scaling: bottleneck → fix (the campaign's improvement arc)

Dose-response on the implicit-CN barotropic reduction count
(LL384×768 L20 f64, solver-matched via `--force-pcg`, 4×A100 NVLink,
jobs 26447957/26449622):

| arm | reductions/step | nd1/nd2/nd4 ms | eff4 |
|---|---|---|---|
| implicit fixed-PCG standard | 123 | 29.3/21.1/15.0 | 0.49 |
| implicit PCG single_reduce | 63 | 29.6/19.8/14.0 | 0.53 |
| explicit + wide-halo | 0 (solver) | 33.4/19.1/11.5 | 0.73 |

Monotone count→efficiency mapping at every size (LL192/384/768); at LL768
all arms converge to 0.71–0.77 (tiles amortize latency). Fused-halo == plain implicit (a NULL result: pad aggregation does not
move this step, consistent with reductions being the larger cost - the
arms differ in solver internals too, so this is consistency, not proof). NCCL_PROTO
default ≡ LL, LL128 harmful (job 26449812) — protocol lever closed.
`LEGOESM_VMIX_F32_SOLVE=1` (f32 tridiagonal vmix inside the f64 step):
icn +9.1%, wide +11.7% at nd4, conservation-gated at `--cons-rtol 1e-5`
on every arm (job 26452547). Combined best config (wide + vmix-f32):
10.32 ms at LL384 nd4 = 1.45× the production config, and the multinode
2.01× above. CLAIM SCOPE (codex round-3): the count→time slope is an
*effective time per eliminated reduction-batch in these executables*
(~14–18 µs/batch), NOT a measured allreduce latency — `single_reduce`
changes solver work/fusion too; a dependency-matched microbenchmark would
be needed for a latency claim. Both winning options are existing config
selections (`barotropic_solver="explicit_substep"` + wide-halo flags,
`LEGOESM_VMIX_F32_SOLVE`); production defaults unchanged — promotion needs
the wide-halo stability gates (`SCALING_STATUS_AUDIT` item 3) and a
science sign-off on the mixed-precision vmix.

## Atm ladders added late in the campaign

**Lat-lon WEAK at a production tile** (45 rows x 1440 lon x L26 = 64.8k
columns/GPU, f32, job 26454476): efficiency 1.00 / 0.45 / 0.41 / 0.45 /
0.44 at 1/2/4/8/16 GPUs — the cost is paid ONCE on the first cross-device
step and then FLAT to 16 GPUs across two node crossings (1.09 -> 7.66
GCells/s aggregate). Weak scaling on this grid is a fixed entry toll, not
a compounding one.

**MPAS icosahedral L8 (28 km) STRONG, identical padded mesh** (jobs
26454476 + 26454618): 19.90 / 17.09 / 6.92 / 7.10 ms at np 2/4/8/16.
Taking np2 as the base (it has the BEST per-device throughput, 430
Mc/s/GPU): 2->8 = 2.88x = eff 0.72, 2->16 = 2.80x = eff 0.35 (small-tile
floor).

OPEN ANOMALY, characterised not explained: per-GPU throughput dips at
np=4 (247 Mc/s/GPU vs 430 at np2 and 306 at np8), so np4 is barely faster
than np2 while np8 is 2.5x faster than np4. Evidence gathered:
- REPRODUCIBLE: two repeats per arm agree within 1 % (19.92/19.87,
  17.14/17.04, 6.87/6.97).
- PLACEMENT REFUTED: np4 packed on one node (17.09 ms) == np4 spread over
  two nodes (17.12 ms), so node crossing is irrelevant.
- HALO VOLUME REFUTED: ghost-cell census on the padded mesh gives
  1540/1587/1400/1136 ghost cells per device at np 2/4/8/16 — flat to
  falling, and under 3 % of owned cells at every count.
- COLLECTIVE COUNT REFUTED (HLO census, ico L7, CPU virtual devices —
  device count is a compile-time property so the HLO matches what the GPUs
  execute): collective-permutes per step are 3 / 9 / 21 at np 2/4/8, i.e.
  np8 issues 2.3x MORE collectives than np4 and still runs 2.5x faster.
  Collective COUNT therefore cannot explain the np4 dip (this assumes cost
rises with count; a per-message-size effect is not excluded). (Fusion count 136/173/240,
  bitcasts 526/582/694 — the np8 program is finer-grained.)
- PARTITION METHOD REFUTED (job 26455829): the dip is method-independent —
  np4/np8 = 17.22/7.00 ms (sfc), 17.20/6.97 (metis), 19.35/6.26
  (geometric). Every method shows the same 2.5-3.1x jump.
- XLA CODEGEN ENV KNOBS REFUTED (job 26455948): np4 is 17.12 ms base,
  17.06 autotune-level-4, 17.10 latency-hiding-off, 17.01
  command-buffers-off — every arm within 1 %, none recovers np4.
  (The multi-output-fusion arm errored on an unsupported flag name and is
  not counted.)
VERDICT: five hypotheses refuted by measurement (placement, halo volume,
collective count, partition method, codegen env knobs). The cheap levers known to this campaign are exhausted; the remaining suspect — per-device kernel efficiency
for this shape — needs a GPU op-level profile (nsys / XLA op profile of
np4 vs np8), which is a separate instrumented project, not another timing
run. Per-GPU throughput across the ladder is non-monotone in tile size
(430 / 249 / 305 / 150 Mc/s/GPU at 327688 / 163844 / 81922 / 40961
cells/device), which is itself the clue to hand the profiler.
PRACTICAL GUIDANCE MEANWHILE: run this grid at np>=8, where per-device
throughput is 304-340 Mc/s/GPU vs 220-248 at np4.

RESOLVED 2026-07-26 (nsys job 26479922 + HLO dump 26480096 + sqlite
timeline): the dip is an XLA CODEGEN pathology, localized to named
kernels. CONFIRMED: (1) the dip reproduces under nsys with matched
protocol (L8, padded-16 mesh: 21.01/17.78/7.17 ms at np2/4/8 vs campaign
19.90/17.09/6.92 — ~5% profiler overhead); (2) at np4 ONLY, giant
serialized "loop fusion" kernels appear — loop_add_fusion_1/2 at 3.6 ms
per launch (vs ~3 us for ordinary elementwise kernels) plus a THIRD
once-per-step group (the unsuffixed loop_add_fusion: 12 of its 44
instances are >1 ms at ~3.3 ms, the rest are the ordinary us-scale adds)
— and the sqlite timeline places all three groups' big instances at the
17.8 ms step cadence (stddev 78 us: deterministic compute, not comm
wait): 3.6 + 3.6 + 3.3 ~= 10.5-10.9 ms/step = the np4 excess; (3) in the optimized
step HLO these are mega-fusions ON THE HALO PATH: `%loop_add_fusion =
f32[491520,26]` (edge-tendency add chain, 22 operands incl. an
input_scatter_fusion) and `%loop_add_fusion.4 = f32[163844,26]` (cell
array), with the shard_map halo-pack concatenates taking the same adds +
parameter lists as operands. PLAUSIBLE (inferred from kInput fusion
semantics + operand lists, not separately timed): the emitter RECOMPUTES
the expensive scatter+add chain inside each consumer fusion, which is why
the cost multiplies. WHY np4: fusion cost-model decisions depend on the
shard shape; at np2/np8 the mega-fusion is not built. This also explains
why the earlier env-knob sweep missed it — autotune/latency-hiding flags
do not change fusion-pass decisions. Fusion-pass flag A/B at np4 ran
(job 26480162): flag route CLOSED — three of four candidate fusion flags
no longer exist in this XLA (upstream removals), the fourth is null, and
the GPU plugin does not list its flags via --help.

FIX ATTEMPTS, both measured (base 19.90 / 17.09 / 6.92 ms at np2/4/8):

| barrier placement | np2 | np4 | np8 | verdict |
|---|---|---|---|---|
| tendency INPUT side (job 26480261) | 21.40 | 16.58 | 7.01 | null at np4, -7.5% np2 — REVERTED |
| tendency OUTPUT side (job 26480310) | 22.03 | **14.09** | 7.04 | **-17.6% time np4** (1.21x), +10.7% time np2 — REVERTED |

(Single runs per arm; the campaign's np4 repeat spread (+-0.3%) supports
an informal ~+-1 pp error on these percentages, not a formal CI.)

The HLO frame table pinpointed the fusion: the 3.6 ms kernels resolve to
`pytree_ops.py:10` (`pytree_axpy.<locals>.<lambda>`) — the RK stage
combine mega-fused with the tendency graph's tail. An output-side
optimization_barrier recovers 3 ms of the ~10.9 at np4 but costs np2
10.7% (it also blocks fusion that HELPS there), so neither barrier ships
unconditionally. Parity + conservation smoke passed on both attempts.

STATUS: **FIXED, SHIPPED GATED** (codex rounds 11-12: strategy consult
BEFORE implementing, then post-review). `_FUSION_BARRIER_WORKLOADS` in
`sharded_dynamics.py` applies the tendency-output optimization_barrier
only at the measured workload signature (n_dev, edge rows, cell rows,
nlev) = (4, 1_966_080, 655_376, 26) — every operand trace-time static.
Verification ladder (job 26486288 vs same-day dead-gate 26486123):
np2 19.86 (campaign base 19.90 — at baseline), **np4 14.12 = -20.6%
same-day / -17.4% vs campaign base**, np8 6.99 (base 6.92). Parity +
conservation smoke green; 23 SPMD parity tests pass. Two instructive
misfires on the way, both caught by measurement: the first gate keyed
per-shard rows (never fired — the trace-time array is the GLOBAL view),
and edge-rows-only was over-broad (L8 edges are unpadded and divisible
several ways — codex round-12). The residual np4 gap to ideal (~14.1 vs
~9.9 from np2/2) is the un-barriered remainder of the fusion; further
recovery needs the integrator-level restructure (codex round-11 ranked
it last on blast radius) or an upstream XLA fix — both remain
follow-ups.

## Ocean strong scaling vs TILE SIZE (jobs 26456334/37 vs 26452804-06)

The same improved config (wide-halo + vmix-f32, multicontroller NCCL/IB,
f32 L20) run at two tile sizes, 4 -> 16 GPUs:

| grid | cells/GPU @16 | np4 / np8 / np16 ms | eff @8 | eff @16 | aggregate @16 |
|---|---|---|---|---|---|
| LL576x1152 (13.3M) | 0.83M | 12.81 / 11.05 / 8.63 | 0.58 | 0.37 | 1.53 GCells/s |
| LL1152x2304 (53.1M) | 3.3M | 43.47 / 27.79 / 17.24 | **0.78** | **0.63** | **3.08 GCells/s** |

Both ladders are monotone; the bigger tile is uniformly better at every
device count (jobs 26456334 / 26457693 / 26456337 vs 26452804-06).

So the ocean shows the SAME tile-size dependence the cube does: the 2.01x
multinode improvement measured at LL576 was partly a floor effect, and at
a production tile the identical code scales substantially better (0.37 ->
0.63). Per-device throughput also rises (259 -> 305 Mc/s/GPU at np4).
Config is byte-identical between the two rows; only the grid changes.

REFUTED EN ROUTE: the np8 leg timed out twice (>90 min still tracing) while
np4 — a LARGER per-device tile — finished in ~25 min, which looked like a
compile-time cliff at that device count. It is not: the third attempt ran
the identical configuration in **99 seconds** with a 21.6 s compile (job
26457693). The earlier hangs were transient/environmental, not
reproducible, and no compile-time defect is claimed.

## Weak scaling at production per-device size (job 26453523)

The earlier weak ladders used a 64-row base (0.17M cells/GPU — under the
latency floor). Re-run at PRODUCTION size (288 rows × 1152 lon × L20 =
6.6M cells/GPU, 1→4 A100, conservation gated): production implicit
1.00/0.72/0.70, improved wide-halo+vmix-f32 1.00/0.86/0.85.
PRECISION MATCHED (self-audit correction): BOTH ladders compared here are
**float32** — the production-tile run is f32, so it is compared against
the earlier ladder's f32 rows (eff 0.26/0.25 at nd 2/4), not its f64 rows (both from job 26445836).
An earlier revision of this file mislabelled the production-tile run f64
and cited the f64 small-base numbers; the direction and size of the effect
are unchanged, but the comparison is only valid precision-matched. So the earlier weak ladder measured a below-floor tile rather than a code
limit, and the same
config that fixes strong scaling also carries weak (+0.15 at nd4). Ideal is
flat; the improved arm holds 22.5→22.9 ms while production drifts
17.8→25.3 ms.

## "It used to be faster / did we regress?" — resolved, no regression

- Cross-machine anchor (matched bench/config/grid/physics/precision,
  job 26450081): Levante single A100-80 latlon-moist r720 f32 =
  **418.5 Mc/s** vs Derecho single-A100 ≈370. SCOPE: this establishes NO
  LARGE REGRESSION, not a precise machine ranking — the two campaigns
  differ in machine (A100-80 SXM vs A100-40), jax/tree version and date,
  so the ~13% gap is not attributable to any single factor.
- Cube "404 vs 141 Mc/s": the 404 is the single-GPU RTX-5090 Held-Suarez
  row in `SCALING_SUMMARY.md` SS1; ours is gray+SBM on A100 (job 26445836).
  That file's own tier table prices gray+SBM ~3x Held-Suarez, so ~135 is
  the expected equivalent vs 141 measured. PLAUSIBLE reconciliation from
  two published tables, NOT a matched A/B (GPU, physics and date differ).
- Ocean absolutes (A100 f64 165-201 Mc/s, f32 352, job 26445836; 5090 f64
  152 / f32 400 from `SCALING_SUMMARY.md` SS1) are of the same order -
  again a cross-machine sanity check, not a controlled comparison.
- Ginsburg "0.92 eff @2 GPU" reconciled: the old bench silently defaulted
  to `explicit_substep`; our explicit/wide arm reproduces that class
  (0.88 @2, LL384) — the production implicit config was never measured
  there. So the gap is explained by the solver the old bench selected;
  labelling it 'protocol, not regression' is an inference from that
  config difference, not an independent bisect.
- No merge regression: nd=1 stock-CG LL192 8.90 ms pre-merge (job 26445836)
  vs 8.94 ms post-merge (smoke on tree d3ec1ccce) - one sample each, so
  this bounds a large regression only.

## CPU-MPI (compute nodes)

Single-node ladders (job 26445986, f64): atm latlon strong eff
0.87–0.92@np2 → 0.11–0.23@np32–64; ocean implicit 0.90@2 → 0.24@32; weak
collapses ≤0.14@32. CAVEATS: np=1 ocean leg was stock-CG (solver-mismatched
— fixed via `--force-pcg` in the job scripts; np≥2 slopes valid), and
single-node ladders conflate Milan DRAM contention with comm. The first
4-node pair collided into one OUTDIR (same-second stamp) and was discarded.

4-node SPREAD ladder (job 26452578, f64, ranks round-robin, solver-matched):
atm latlon np2 eff ~1.00; spreading ranks over 4 nodes nearly doubles
efficiency at high rank counts (r128 np32: 0.38 spread vs 0.20 packed),
which is CONSISTENT with per-node memory-bandwidth contention in the
packed ladder (not isolated by a bandwidth counter), decaying to 0.06–0.16
at np64–128 — the 1-D band perimeter ceiling as designed (r256/np128 = 2
rows/rank). Ocean strong spread: np2 eff 1.32 (superlinear - typical of a base leg whose working set does
not fit cache; not instrumented here), 0.88@8,
0.35@32, wall at np64 (79 ms > np32's 74 ms). The job died in a high-rank
ocean case (one rank exit-3 → kill-on-bad-exit) before the weak tail —
np128 ocean + weak ladders and the rank-failure attribution remain open.

## Atm ICOSAHEDRAL CPU-MPI (job 26452579, 1 node, f64 moist, L26)

The last measurement gap, and the healthiest strong-scaling curve in the
campaign. Efficiency t1/(n*tn) by subdivision:

| subdiv | np2 | np4 | np8 | np16 | np32 | np64 |
|---|---|---|---|---|---|---|
| 4 | 0.90 | 0.76 | 1.00 | 0.53 | 0.26 | 0.15 |
| 5 | 1.02 | 0.88 | 1.15 | 0.56 | 0.58 | 0.30 |
| 6 | 1.02 | 0.88 | 1.07 | 0.51 | 0.54 | 0.57 |
| 7 | 1.03 | 0.92 | 1.02 | 0.51 | 0.51 | 0.52 |

**REVISION 2026-07-27 — the subdiv-7 row was PLACEMENT-LIMITED, not
comm-limited.** Re-running it with `--distribution=block:cyclic` (the
Milan fix, discovered after this sweep) gives f64 np64 **efficiency 0.71,
up from 0.52**, and the high-rank columns move most: placement alone is
worth 2.00x at np16, 1.75x at np32, 1.38x at np64 (job 26495437 vs
26452579). The f32 ladder at matched placement (job 26495083) reaches
**0.88**. So the "np16 dip" visible across every row of this table is
substantially the same NUMA effect found later in the packed CPU atm
ladder — one fix, two symptoms. Precision itself is worth a near-constant
~1.4x here; the naive cross-job comparison would have read 2.81x at np16
and attributed placement to precision.
| 8 | 1.04 | 0.93 | 1.18 | 0.61 | 0.49 | — |

THE TILE-SIZE PATTERN, THIRD LANE (cube and ico are both atmosphere:
two components, three decomposition lanes): the np64 column collapses
on coarse grids (0.15 at subdiv4, 0.30 at subdiv5) and holds on fine ones
(0.52-0.57 at subdiv6-7). Same pattern as the cube (0.23 -> 1.04) and the ocean (0.37 -> 0.63), now
on a third lane and a different transport (CPU-MPI, not NCCL). It is the
campaign's most reproducible ASSOCIATION — but changing C-resolution, LL
size or ico subdivision also changes the global problem, so tile size is
not causally isolated.

Shape: ~1.0 through np8, one step down, then FLAT 0.5 from np16 to np64 —
4x more ranks with no further loss. CONSISTENT with a fixed per-rank cost
rather than growing communication, but flat efficiency alone does not
identify which; that needs phase-level timing. (The >1 points at np8 are the same
base-leg-working-set effect noted for the cube; read as "at ideal".)

Against the 4-node SPREAD lat-lon ladder (job 26452578) at high rank
counts the contrast is large: ico holds 0.52 at np64 where lat-lon r128/r256
is at 0.12/0.16. That matches the documented expectation that a 2-D cell
partition beats a 1-D latitude band on perimeter/area. CAVEAT: ico ran
PACKED on one node and lat-lon SPREAD over four, so this compares
decomposition AND placement together, not decomposition alone.

The ico WEAK ladder from the same job is non-monotone (1.00 / 0.47 / 0.81 /
0.48 / 0.46 / 0.22 / 0.45 at np 1..64) — the per-rank problem size is not
held constant cleanly across that sweep's subdivision steps, so no weak
claim is made from it.

## DISTANCE TO THE THEORETICAL LIMIT, measured (job 26457977)

With all three ingredients measured on this machine — fabric constants
(17.82 us, 64.22 GB/s) and the per-tile single-device compute term — the
ocean LL576 f64 implicit ladder finally has a real roofline:

| nd | measured | calibrated bound | measured/bound | at % of floor |
|---|---|---|---|---|
| 2 | 42.71 ms | 34.77 ms | 1.228 | 81 % |
| 4 | 23.53 ms | 16.82 ms | 1.398 | 72 % |

Bound = per-device compute (32.58 / 14.63 ms) + modelled comm (2.21) +
modelled reduction (2.19). The unmodelled gap is **~5 ms/step and roughly
FLAT** with device count (7.9 ms at nd2, 6.7 at nd4), which is why the
ratio worsens as compute shrinks.

WHAT THE GAP IS NOT — the omitted-traffic explanation is REFUTED
(`scripts/tmp/probe_ocean_halo_bytes.py`, HLO byte census on CPU virtual
devices). The bench's `comm_scope_note` correctly warns that its census is
"barotropic implicit-CN PCG scope only … baroclinic 3-D pads NOT counted",
and the true volume IS much larger: **16.22 MB/step across 110
collective-permutes vs the censused 2.25 MB — a 7.2x undercount**. But
completing the census moves the bound by only **0.22 ms**, because the
comm term is LATENCY-dominated: at 122 messages x 17.82 us the latency part
is 2.174 ms while even 16 MB at 64.22 GB/s is just 0.253 ms.

So with the byte census completed the unexplained residual is still 5.5 ms
(nd2) and 4.3 ms (nd4).

SECOND CANDIDATE ALSO REFUTED (`scripts/tmp/probe_sharded_overhead.py`,
job 26458553): the sharded formulation does NOT do measurably more work.
Timing the SHARDED step on a 1-device mesh (all the padding, band-edge and
v-row-reconstruction machinery present, ppermutes self-to-self so no real
traffic) against the UNSHARDED step at the identical tile:

| tile | unsharded | sharded on 1 device | overhead |
|---|---|---|---|
| 288x1152x20 | 33.13 ms | 32.79 ms | **-0.34 ms (-1.0 %)** |
| 144x1152x20 | 15.88 ms | 15.92 ms | **+0.04 ms (+0.2 %)** |

Zero within noise at both tiles, so the bound's compute term is the RIGHT
reference and extra sharded work is not the gap.

WHERE THAT LEAVES IT (quantified, one candidate standing): the residual
divided by the message count is **83 us/message at nd2 and 73 us at nd4**,
versus **17.8 us** for the same collective measured in isolation — an in-
context cost 4-5x the best case. That is consistent with EXPOSED,
un-overlapped communication rather than raw wire time.

THIRD CANDIDATE REFUTED, AND IT IDENTIFIES THE MECHANISM (job 26458930).
If the residual were communication the scheduler is currently hiding work
behind, DISABLING XLA's latency-hiding scheduler would hurt. It does not:

| arm | nd2 | nd4 | vs default |
|---|---|---|---|
| default (LHS on) | 42.69 | 23.53 ms | — |
| `latency_hiding_scheduler=false` | 42.44 | 23.49 | **+0.6 % / +0.2 %** |
| `enable_pipelined_p2p=true` | 42.76 | 23.54 | -0.2 % / -0.1 % |
| CP combining @32 MiB | 42.55 | 23.59 | +0.3 % / -0.3 % |

Turning overlap OFF is free (marginally faster), and no scheduling flag
moves the step. The scheduler has nothing to hide the comm behind.

**MECHANISM (the three tested alternatives are not dominant): the
residual is best explained by EXPOSED, DEPENDENCY-SERIALIZED
SYNCHRONISATION.** These are eliminations of the TESTED implementations,
not of every possible communication explanation. Not bytes
(7.2x more = +0.22 ms), not sharded-formulation work (0 ms), not
hideable-by-scheduling (0 ms). It is the unavoidable cost of sync points
that sit on a dependent chain.

This retro-explains every earlier arm in the campaign, which is the check
that the mechanism is right rather than merely last-standing:
- fused-halo NULL — aggregation reduces message COUNT but not chain DEPTH;
- `single_reduce` HELPED (0.49 -> 0.53) — Chronopoulos-Gear restructures
  the recurrence into fewer DEPENDENT reduction batches;
- wide-halo HELPED MOST (-> 0.73) — it deletes the barotropic solver's sync
  points outright;
- the f64/f32 flip — more compute per sync point dilutes a fixed sync cost.

ACTIONABLE CONSEQUENCE: the lever for this lane is reducing the NUMBER OF
DEPENDENT SYNCHRONISATION POINTS, not message aggregation, byte

exec
/usr/bin/bash -lc "rg -n -C 3 'VERDICT:|"'^[0-9]+'"\\. |FIX-FIRST|ISSUE|issue|"'^[#]{1,4} '"' .physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md | tail -n 260" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 374ms:
11030:2791:[l20131.lvt.dkrz.de:271566] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11031:2792:[l20119.lvt.dkrz.de:1442699] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11032:2793:[l20119.lvt.dkrz.de:1442688] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11033:2794:[l20147.lvt.dkrz.de:3469650] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11034:2795:[l20131.lvt.dkrz.de:271567] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11035:2796:[l20130.lvt.dkrz.de:51419] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11036:2797:[l20119.lvt.dkrz.de:1442682] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11037:2798:[l20119.lvt.dkrz.de:1442687] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11038:2799:[l20119.lvt.dkrz.de:1442693] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11039:2800:[l20130.lvt.dkrz.de:51420] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11040:2801:[l20131.lvt.dkrz.de:271561] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11041:2802:[l20131.lvt.dkrz.de:271547] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11042:2803:[l20131.lvt.dkrz.de:271550] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11043:2804:[l20131.lvt.dkrz.de:271571] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11044:2805:[l20131.lvt.dkrz.de:271557] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11045:2806:[l20119.lvt.dkrz.de:1442686] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11046:2807:[l20130.lvt.dkrz.de:51418] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11047:2808:[l20130.lvt.dkrz.de:51401] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11048:2809:[l20119.lvt.dkrz.de:1442692] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11049:2810:[l20119.lvt.dkrz.de:1442690] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11050:2811:[l20147.lvt.dkrz.de:3469648] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11051:2812:[l20130.lvt.dkrz.de:51425] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11052:2813:[l20131.lvt.dkrz.de:271549] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11053:2814:[l20130.lvt.dkrz.de:51423] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11054:2815:[l20131.lvt.dkrz.de:271560] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11055:2816:[l20130.lvt.dkrz.de:51400] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11056:2817:[l20131.lvt.dkrz.de:271548] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11057:2818:[l20130.lvt.dkrz.de:51424] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11058:2819:[l20130.lvt.dkrz.de:51426] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11059:2820:[l20119.lvt.dkrz.de:1442680] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11060:2821:[l20130.lvt.dkrz.de:51422] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11061:2822:[l20130.lvt.dkrz.de:51408] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11062:2823:[l20130.lvt.dkrz.de:51407] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11063:2824:[l20130.lvt.dkrz.de:51397] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11064:2825:[l20130.lvt.dkrz.de:51416] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11065:2826:[l20130.lvt.dkrz.de:51409] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11066:2827:[l20130.lvt.dkrz.de:51421] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11067-2828-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/voronoi_mpi.py:632: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
11068-2829-  require_mpi_stack()
11069-2830-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/voronoi_mpi.py:632: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
--
11080-4492-[1785518454.769837] [l20131:271545:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
11081-4493-[1785518454.822726] [l20119:1442706:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
11082-4494:--- s8 np=128 partition=metis dist=block:block ---
11083:4495:[l20119.lvt.dkrz.de:1443884] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11084:4496:[l20131.lvt.dkrz.de:272741] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11085:4497:[l20131.lvt.dkrz.de:272728] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11086:4498:[l20131.lvt.dkrz.de:272733] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11087:4499:[l20131.lvt.dkrz.de:272743] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11088:4500:[l20147.lvt.dkrz.de:3470822] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11089:4501:[l20147.lvt.dkrz.de:3470820] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11090:4502:[l20131.lvt.dkrz.de:272756] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11091:4503:[l20119.lvt.dkrz.de:1443902] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11092:4504:[l20147.lvt.dkrz.de:3470848] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11093:4505:[l20147.lvt.dkrz.de:3470828] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11094:4506:[l20147.lvt.dkrz.de:3470821] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11095:4507:[l20119.lvt.dkrz.de:1443901] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11096:4508:[l20131.lvt.dkrz.de:272739] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11097:4509:[l20147.lvt.dkrz.de:3470845] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11098:4510:[l20147.lvt.dkrz.de:3470849] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11099:4511:[l20131.lvt.dkrz.de:272729] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11100:4512:[l20131.lvt.dkrz.de:272737] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11101:4513:[l20130.lvt.dkrz.de:52609] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11102:4514:[l20119.lvt.dkrz.de:1443905] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11103:4515:[l20147.lvt.dkrz.de:3470835] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11104:4516:[l20130.lvt.dkrz.de:52582] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11105:4517:[l20131.lvt.dkrz.de:272727] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11106:4518:[l20130.lvt.dkrz.de:52588] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11107:4519:[l20131.lvt.dkrz.de:272730] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11108:4520:[l20131.lvt.dkrz.de:272735] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11109:4521:[l20119.lvt.dkrz.de:1443900] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11110:4522:[l20119.lvt.dkrz.de:1443910] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11111:4523:[l20119.lvt.dkrz.de:1443897] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11112:4524:[l20131.lvt.dkrz.de:272732] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11113:4525:[l20119.lvt.dkrz.de:1443906] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11114:4526:[l20119.lvt.dkrz.de:1443903] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11115:4527:[l20130.lvt.dkrz.de:52597] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11116:4528:[l20119.lvt.dkrz.de:1443904] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11117:4529:[l20147.lvt.dkrz.de:3470824] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11118:4530:[l20131.lvt.dkrz.de:272736] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11119:4531:[l20119.lvt.dkrz.de:1443882] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11120:4532:[l20119.lvt.dkrz.de:1443893] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11121:4533:[l20119.lvt.dkrz.de:1443891] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11122:4534:[l20119.lvt.dkrz.de:1443898] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11123:4535:[l20130.lvt.dkrz.de:52586] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11124:4536:[l20130.lvt.dkrz.de:52604] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11125:4537:[l20119.lvt.dkrz.de:1443880] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11126:4538:[l20130.lvt.dkrz.de:52585] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11127:4539:[l20131.lvt.dkrz.de:272746] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11128:4540:[l20147.lvt.dkrz.de:3470833] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11129:4541:[l20130.lvt.dkrz.de:52583] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11130:4542:[l20119.lvt.dkrz.de:1443881] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11131:4543:[l20147.lvt.dkrz.de:3470840] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11132:4544:[l20147.lvt.dkrz.de:3470843] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11133:4545:[l20131.lvt.dkrz.de:272738] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11134:4546:[l20130.lvt.dkrz.de:52605] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11135:4547:[l20147.lvt.dkrz.de:3470819] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11136:4548:[l20147.lvt.dkrz.de:3470844] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11137:4549:[l20119.lvt.dkrz.de:1443908] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11138:4550:[l20147.lvt.dkrz.de:3470838] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11139:4551:[l20147.lvt.dkrz.de:3470818] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11140:4552:[l20130.lvt.dkrz.de:52581] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11141:4553:[l20131.lvt.dkrz.de:272758] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11142:4554:[l20119.lvt.dkrz.de:1443896] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11143:4555:[l20119.lvt.dkrz.de:1443899] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11144:4556:[l20130.lvt.dkrz.de:52594] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11145:4557:[l20130.lvt.dkrz.de:52598] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11146:4558:[l20130.lvt.dkrz.de:52599] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11147:4559:[l20147.lvt.dkrz.de:3470842] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11148:4560:[l20131.lvt.dkrz.de:272745] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11149:4561:[l20119.lvt.dkrz.de:1443911] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11150:4562:[l20147.lvt.dkrz.de:3470830] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11151:4563:[l20131.lvt.dkrz.de:272749] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11152:4564:[l20130.lvt.dkrz.de:52593] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11153:4565:[l20131.lvt.dkrz.de:272731] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11154:4566:[l20147.lvt.dkrz.de:3470834] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11155:4567:[l20119.lvt.dkrz.de:1443907] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11156:4568:[l20147.lvt.dkrz.de:3470841] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11157:4569:[l20119.lvt.dkrz.de:1443892] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11158:4570:[l20119.lvt.dkrz.de:1443889] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11159:4571:[l20130.lvt.dkrz.de:52596] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11160:4572:[l20119.lvt.dkrz.de:1443895] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11161:4573:[l20147.lvt.dkrz.de:3470839] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11162:4574:[l20130.lvt.dkrz.de:52587] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11163:4575:[l20147.lvt.dkrz.de:3470832] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11164:4576:[l20131.lvt.dkrz.de:272755] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11165:4577:[l20130.lvt.dkrz.de:52592] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11166:4578:[l20130.lvt.dkrz.de:52590] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11167:4579:[l20147.lvt.dkrz.de:3470831] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11168:4580:[l20119.lvt.dkrz.de:1443909] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11169:4581:[l20131.lvt.dkrz.de:272754] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11170:4582:[l20131.lvt.dkrz.de:272753] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11171:4583:[l20119.lvt.dkrz.de:1443885] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11172:4584:[l20119.lvt.dkrz.de:1443886] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11173:4585:[l20130.lvt.dkrz.de:52603] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11174:4586:[l20130.lvt.dkrz.de:52584] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11175:4587:[l20131.lvt.dkrz.de:272757] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11176:4588:[l20131.lvt.dkrz.de:272750] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11177:4589:[l20147.lvt.dkrz.de:3470823] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11178:4590:[l20130.lvt.dkrz.de:52579] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11179:4591:[l20130.lvt.dkrz.de:52595] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11180:4592:[l20130.lvt.dkrz.de:52580] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11181:4593:[l20147.lvt.dkrz.de:3470827] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11182:4594:[l20119.lvt.dkrz.de:1443883] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11183:4595:[l20130.lvt.dkrz.de:52591] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11184:4596:[l20130.lvt.dkrz.de:52606] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11185:4597:[l20147.lvt.dkrz.de:3470825] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11186:4598:[l20147.lvt.dkrz.de:3470836] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11187:4599:[l20147.lvt.dkrz.de:3470847] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11188:4600:[l20147.lvt.dkrz.de:3470846] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11189:4601:[l20131.lvt.dkrz.de:272734] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11190:4602:[l20131.lvt.dkrz.de:272747] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11191:4603:[l20119.lvt.dkrz.de:1443888] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11192:4604:[l20131.lvt.dkrz.de:272751] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11193:4605:[l20147.lvt.dkrz.de:3470829] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11194:4606:[l20130.lvt.dkrz.de:52607] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11195:4607:[l20130.lvt.dkrz.de:52608] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11196:4608:[l20130.lvt.dkrz.de:52610] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11197:4609:[l20131.lvt.dkrz.de:272740] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11198:4610:[l20131.lvt.dkrz.de:272744] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11199:4611:[l20131.lvt.dkrz.de:272752] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11200:4612:[l20119.lvt.dkrz.de:1443887] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11201:4613:[l20147.lvt.dkrz.de:3470837] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11202:4614:[l20119.lvt.dkrz.de:1443894] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11203:4615:[l20147.lvt.dkrz.de:3470826] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11204:4616:[l20130.lvt.dkrz.de:52601] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11205:4617:[l20131.lvt.dkrz.de:272748] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11206:4618:[l20130.lvt.dkrz.de:52602] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11207:4619:[l20119.lvt.dkrz.de:1443890] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11208:4620:[l20131.lvt.dkrz.de:272742] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11209:4621:[l20130.lvt.dkrz.de:52600] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11210:4622:[l20130.lvt.dkrz.de:52589] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11211-4623-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/voronoi_mpi.py:632: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
11212-4624-  require_mpi_stack()
11213-4625-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/voronoi_mpi.py:632: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
--
11267-451-[mpas-ocean np=32 L7 nCells=163842 nlev=20 solver=explicit_substep halo=in_step] compile=12255.1ms fused=189.8206ms/step (probe_latency=187.99ms) gate_loop_latency=187.13ms/step
11268-452-[1785518297.289277] [l20119:1439070:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
11269-453---- s8 np=128 partition=geometric dist=block:cyclic ---
11270:454-[l20131.lvt.dkrz.de:270332] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11271---
11272-2145-[1785518353.342904] [l20119:1440313:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
11273-2146-[1785518353.389333] [l20119:1440311:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
--
11309-4390-[1785518454.772658] [l20119:1442708:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
11310-4391-[1785518454.799331] [l20130:51417:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
11311---
11312:4573-[l20147.lvt.dkrz.de:3470839] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11313:4574-[l20130.lvt.dkrz.de:52587] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11314:4575-[l20147.lvt.dkrz.de:3470832] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11315:4576-[l20131.lvt.dkrz.de:272755] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11316:4577:[l20130.lvt.dkrz.de:52592] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11317:4578-[l20130.lvt.dkrz.de:52590] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11318:4579-[l20147.lvt.dkrz.de:3470831] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11319:4580-[l20119.lvt.dkrz.de:1443909] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11320:4581-[l20131.lvt.dkrz.de:272754] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11321---
11322-6229-[1785518522.187344] [l20131:272751:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
11323-6230-[1785518522.118272] [l20131:272755:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
--
12936-| s8 np64 GC/s | 3.233 using natural cells; 3.235 using executed padded cells |
12937-| s9/s8 peak ratio | **2.19×** → 2.2× |
12938-
12939:1. The METIS matrix does not hold every arm at exactly 5,120 cells/rank. The receipts show geometric 5,120–5,121, but METIS is 5,100–5,145 at 32 ranks and 5,093–5,144 at 128. The wording at [the Phase-3 table](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/docs/performance/scaling/levante_campaign_2026-07-24.md:1594) is false; call it a nominal 5,120-cell target and report min/max.
12940-
12941:2. “Partition-quality lever DEAD/refuted” overclaims. This is evidence that this METIS configuration loses to geometric at L8/128 despite lower cut/halo metrics; it does not eliminate partition quality or mapping generally. The 32→128 terms are weak-scale, 1→4-node scale-out terms, not isolated rank-count terms.
12942-
12943:3. The placement mechanism is misdescribed. Both `block:cyclic` and `block:block` use `block` for rank-to-node placement; the second field changes CPU allocation across sockets. E/D does support “`block:cyclic` is 1.61× faster in this receipt,” but “per-socket bandwidth beats communication locality” is **PLAUSIBLE**, not confirmed.
12944-
12945:4. The METIS receipt ran without parity or conservation gates and logs an unsupported MPI/JAX pairing warning. That does not erase the timing observation, but blocks a production-quality “lever closed” claim until the stack is supported and at least one gated arm per partition passes.
12946-
12947:5. The s8→s9 weak-pair claim is confounded. s9 explicitly used `lloyd=0`; the s8 source predates the `--lloyd` option and therefore used the generator’s default production relaxation. They are not the “same protocol.” The first two comparator points are also from the earlier np2–16 ladder, not the claimed np32–128 extension rows. Remove the causal GPU rank-count claim and the plot annotation until s8 is rerun with `lloyd=0` and matched reorder policy.
12948-
12949:6. “Consistent ~1.4×” is arithmetically false: the three costs are 1.80×, 1.35×, and 1.41×. The first is a large outlier. The proposed GPU rank-count term is therefore **PLAUSIBLE at best**, even after the Lloyd confound is fixed.
12950-
12951:7. The reported mesh counts are natural, not executed padded counts. The s9 receipt executed 2,621,568 cells, not 2,621,442; its np32 provenance SHA is also `unknown`. Throughput rounding remains correct, but report executed `n_cells` and resolve that provenance hole. “New MPAS-atmosphere peak” should be scoped to the synthetic, `physics=none` benchmark.
12952-
12953:8. “Floor confirmed” should be **PLAUSIBLE/consistent with the floor**, not CONFIRMED: one below-floor point anti-scales, with no replication and known shape/codegen variation.
12954-
12955:9. The ensemble script masks failures: each background `run_arm` changes `rc` only in its subshell, and `srun ... || { rc=1; }` returns success. Missing/corrupt JSON is silently skipped, so the job can print `DONE rc=0`. [The script](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/scripts/tmp/mpas_s9_ensemble.sbatch:44) must preserve each background PID/status and require all five valid receipts. It is also ignored by Git, so it is not reproducibly reviewable.
12956-
12957:10. GPU binding is otherwise sound: `--gpu-bind=none` leaves all GPUs visible and JAX’s Slurm auto-detection selects by local rank. Full-GPU `--exact` requests should force disjoint eight-node steps. But log each `SLURM_STEP_NODELIST` and coordinator host; that is the evidence the shared JAX job-ID port is safe.
12958-
12959:11. The ensemble does not synchronize measurement windows, pair each replica topology with a solo control, or counterbalance solo-first ordering. It can measure different node/fabric groups or non-overlapping windows rather than contention. Twelve steps give nine dependent steady samples per launch, not enough independent evidence for a 10% confirmation bar.
12960-
12961:12. Ensemble arithmetic needs tighter wording: if solo reproduces 5.47 GC/s, ideal aggregate is 21.88 GC/s. The 1.10 threshold guarantees only 19.89 GC/s = 3.636× the 32-GPU solo rate, or 3.35× the observed 128-GPU single trajectory—not 21.9 GC/s.
12962-
12963:VERDICT: FIX-FIRST
12964-tokens used
12965-232,430
12966-Arithmetic checks:
--
12979-| s8 np64 GC/s | 3.233 using natural cells; 3.235 using executed padded cells |
12980-| s9/s8 peak ratio | **2.19×** → 2.2× |
12981-
12982:1. The METIS matrix does not hold every arm at exactly 5,120 cells/rank. The receipts show geometric 5,120–5,121, but METIS is 5,100–5,145 at 32 ranks and 5,093–5,144 at 128. The wording at [the Phase-3 table](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/docs/performance/scaling/levante_campaign_2026-07-24.md:1594) is false; call it a nominal 5,120-cell target and report min/max.
12983-
12984:2. “Partition-quality lever DEAD/refuted” overclaims. This is evidence that this METIS configuration loses to geometric at L8/128 despite lower cut/halo metrics; it does not eliminate partition quality or mapping generally. The 32→128 terms are weak-scale, 1→4-node scale-out terms, not isolated rank-count terms.
12985-
12986:3. The placement mechanism is misdescribed. Both `block:cyclic` and `block:block` use `block` for rank-to-node placement; the second field changes CPU allocation across sockets. E/D does support “`block:cyclic` is 1.61× faster in this receipt,” but “per-socket bandwidth beats communication locality” is **PLAUSIBLE**, not confirmed.
12987-
12988:4. The METIS receipt ran without parity or conservation gates and logs an unsupported MPI/JAX pairing warning. That does not erase the timing observation, but blocks a production-quality “lever closed” claim until the stack is supported and at least one gated arm per partition passes.
12989-
12990:5. The s8→s9 weak-pair claim is confounded. s9 explicitly used `lloyd=0`; the s8 source predates the `--lloyd` option and therefore used the generator’s default production relaxation. They are not the “same protocol.” The first two comparator points are also from the earlier np2–16 ladder, not the claimed np32–128 extension rows. Remove the causal GPU rank-count claim and the plot annotation until s8 is rerun with `lloyd=0` and matched reorder policy.
12991-
12992:6. “Consistent ~1.4×” is arithmetically false: the three costs are 1.80×, 1.35×, and 1.41×. The first is a large outlier. The proposed GPU rank-count term is therefore **PLAUSIBLE at best**, even after the Lloyd confound is fixed.
12993-
12994:7. The reported mesh counts are natural, not executed padded counts. The s9 receipt executed 2,621,568 cells, not 2,621,442; its np32 provenance SHA is also `unknown`. Throughput rounding remains correct, but report executed `n_cells` and resolve that provenance hole. “New MPAS-atmosphere peak” should be scoped to the synthetic, `physics=none` benchmark.
12995-
12996:8. “Floor confirmed” should be **PLAUSIBLE/consistent with the floor**, not CONFIRMED: one below-floor point anti-scales, with no replication and known shape/codegen variation.
12997-
12998:9. The ensemble script masks failures: each background `run_arm` changes `rc` only in its subshell, and `srun ... || { rc=1; }` returns success. Missing/corrupt JSON is silently skipped, so the job can print `DONE rc=0`. [The script](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/scripts/tmp/mpas_s9_ensemble.sbatch:44) must preserve each background PID/status and require all five valid receipts. It is also ignored by Git, so it is not reproducibly reviewable.
12999-
13000:10. GPU binding is otherwise sound: `--gpu-bind=none` leaves all GPUs visible and JAX’s Slurm auto-detection selects by local rank. Full-GPU `--exact` requests should force disjoint eight-node steps. But log each `SLURM_STEP_NODELIST` and coordinator host; that is the evidence the shared JAX job-ID port is safe.
13001-
13002:11. The ensemble does not synchronize measurement windows, pair each replica topology with a solo control, or counterbalance solo-first ordering. It can measure different node/fabric groups or non-overlapping windows rather than contention. Twelve steps give nine dependent steady samples per launch, not enough independent evidence for a 10% confirmation bar.
13003-
13004:12. Ensemble arithmetic needs tighter wording: if solo reproduces 5.47 GC/s, ideal aggregate is 21.88 GC/s. The 1.10 threshold guarantees only 19.89 GC/s = 3.636× the 32-GPU solo rate, or 3.35× the observed 128-GPU single trajectory—not 21.9 GC/s.
13005-
13006:VERDICT: FIX-FIRST

exec
/usr/bin/bash -lc "rg -n -i -C 5 'Phase-3|recovered|s9|METIS|weak-pair|weak pair|2621568|2621568|LL2048|hundreds|2304|2880|512|floor|UCX|confirm|3\\.64|3\\.636|solo|ensemble|peak' docs/performance/scaling/levante_campaign_2026-07-24.md" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 311ms:
9-(transcripts under `.physics-validator/scaling_campaign/`); every
10-measurement claim below carries the round-3 corrections.
11-
12-Machines: Levante `gpu` partition (4× A100-80 SXM NVLink/node, IB HDR200),
13-`compute` (2× AMD Milan 7763). All GPU multinode = route-B
14:(`jax.distributed` + NCCL over IB verbs — `NET/IB mlx5` confirmed in-log;
15-route-A CUDA-aware mpi4jax not exercised on Levante).
16-
17-## Headline results (strong scaling, f32 unless noted)
18-
19-| Axis | Ladder | Result | Job(s) |
20-|---|---|---|---|
21-| Atm lat-lon LL720×1440 L26 | 4→8→16 A100 (1→4 nodes) | 7.72→5.40→3.54 ms/step, monotone; np16 = 7.6 GC/s (477 Mc/s/GPU sustained) | 26450848/26453240/26449147 |
22-| Atm MPAS ico L8 (28 km) L26 | 6→16 A100 | 8.66→7.08 ms/step; np16 = 2.41 GC/s — 1.6x the Derecho 16-A100 aggregate reported in `derecho_levante_sota_review_2026-07.md` SS3b (route-A, eff ~0.38 @16); CROSS-MACHINE, different stack/date - indicative, not a controlled A/B | 26453240/26449147 |
23-| **Atm cube C768/L60 (same-path cs-spmd)** | 6→24 A100 | 58.35→14.09 ms/step = **4.14× = eff 1.04 (at ideal)**, 15.1 GC/s (629 Mc/s/GPU) | 26453782 |
24-| Atm cube C384/L60 (same-path cs-spmd) | 6→24 A100 | 15.44→8.81 ms/step = 1.75× (eff 0.44), 6.0 GC/s | 26452894 |
25:| Atm cube C192/L60 (same-path) | 6→24 | 6.20→6.80 ms — ANTI-scales (eff 0.23): 9.2k cols/GPU is below the ~30k-column floor | 26452979 |
26-| Ocean lat-lon LL576×1152 L20, production implicit | 4→8→16 | 15.6→17.2→17.4 ms — anti-scales across nodes | 26452743-45 |
27-| Ocean same, improved (wide-halo + vmix-f32) | 4→8→16 | 12.8→11.1→8.6 ms — monotone, **2.01× at 16 GPUs** | 26452804-06, 26453279 |
28-| Cube tiled >6-GPU lane (its own bench) | 24 A100, C384/L60 closed loop | 9.01 ms/step, 5.9 GC/s, 18.2 SYPD — first >6-GPU production-lane receipts | 26450938/26452632 |
29-
30-Single-node GPU (job 26445836): cube C192 gray_sbm strong eff 0.84–1.07
31:(1→3 A100); C48/C96 latency-floored. Ocean single-node solver ladder: below.
32-
33-**THE cube strong-scaling result** — read 1.04 as "at ideal", NOT "better
34-than ideal": efficiency slightly above 1 is expected when the BASE leg is
35-per-device disadvantaged (np6 holds 4x the working set per GPU of np24, so
36-part of the 4.14x is cache/occupancy recovery rather than parallel
--
41-"poor cube strong scaling" of the earlier receipts TRACKS TILE SIZE:
42-holding code path, harness, config and IC fixed and varying only the tile,
43-efficiency goes 0.23 -> 0.44 -> 1.04, so tile size is SUFFICIENT to recover
44-ideal scaling at 24 A100 across 6 nodes. (That shows comm does not degrade
45-the ladder at production tiles; it does not prove comm costs nothing at
46:small tiles — that needs a per-phase profile.) Production rule confirmed:
47-keep >~30k columns/GPU.
48-
49-Tiled-lane size sweep at fixed 24 GPUs (own bench, closed loop, CFL-scaled
50-dt; jobs 26450938/26452632/26453645): 1.87 / 5.89 / 14.34 GCells/s at
51-C192/C384/C768 = 78 / 246 / 597 Mc/s per GPU — per-device throughput still
--
93-
94-**MPAS icosahedral L8 (28 km) STRONG, identical padded mesh** (jobs
95-26454476 + 26454618): 19.90 / 17.09 / 6.92 / 7.10 ms at np 2/4/8/16.
96-Taking np2 as the base (it has the BEST per-device throughput, 430
97-Mc/s/GPU): 2->8 = 2.88x = eff 0.72, 2->16 = 2.80x = eff 0.35 (small-tile
98:floor).
99-
100-OPEN ANOMALY, characterised not explained: per-GPU throughput dips at
101-np=4 (247 Mc/s/GPU vs 430 at np2 and 306 at np8), so np4 is barely faster
102-than np2 while np8 is 2.5x faster than np4. Evidence gathered:
103-- REPRODUCIBLE: two repeats per arm agree within 1 % (19.92/19.87,
--
113-  np8 issues 2.3x MORE collectives than np4 and still runs 2.5x faster.
114-  Collective COUNT therefore cannot explain the np4 dip (this assumes cost
115-rises with count; a per-message-size effect is not excluded). (Fusion count 136/173/240,
116-  bitcasts 526/582/694 — the np8 program is finer-grained.)
117-- PARTITION METHOD REFUTED (job 26455829): the dip is method-independent —
118:  np4/np8 = 17.22/7.00 ms (sfc), 17.20/6.97 (metis), 19.35/6.26
119-  (geometric). Every method shows the same 2.5-3.1x jump.
120-- XLA CODEGEN ENV KNOBS REFUTED (job 26455948): np4 is 17.12 ms base,
121-  17.06 autotune-level-4, 17.10 latency-hiding-off, 17.01
122-  command-buffers-off — every arm within 1 %, none recovers np4.
123-  (The multi-output-fusion arm errored on an unsupported flag name and is
--
132-PRACTICAL GUIDANCE MEANWHILE: run this grid at np>=8, where per-device
133-throughput is 304-340 Mc/s/GPU vs 220-248 at np4.
134-
135-RESOLVED 2026-07-26 (nsys job 26479922 + HLO dump 26480096 + sqlite
136-timeline): the dip is an XLA CODEGEN pathology, localized to named
137:kernels. CONFIRMED: (1) the dip reproduces under nsys with matched
138-protocol (L8, padded-16 mesh: 21.01/17.78/7.17 ms at np2/4/8 vs campaign
139-19.90/17.09/6.92 — ~5% profiler overhead); (2) at np4 ONLY, giant
140-serialized "loop fusion" kernels appear — loop_add_fusion_1/2 at 3.6 ms
141-per launch (vs ~3 us for ordinary elementwise kernels) plus a THIRD
142-once-per-step group (the unsuffixed loop_add_fusion: 12 of its 44
--
200-f32 L20) run at two tile sizes, 4 -> 16 GPUs:
201-
202-| grid | cells/GPU @16 | np4 / np8 / np16 ms | eff @8 | eff @16 | aggregate @16 |
203-|---|---|---|---|---|---|
204-| LL576x1152 (13.3M) | 0.83M | 12.81 / 11.05 / 8.63 | 0.58 | 0.37 | 1.53 GCells/s |
205:| LL1152x2304 (53.1M) | 3.3M | 43.47 / 27.79 / 17.24 | **0.78** | **0.63** | **3.08 GCells/s** |
206-
207-Both ladders are monotone; the bigger tile is uniformly better at every
208-device count (jobs 26456334 / 26457693 / 26456337 vs 26452804-06).
209-
210-So the ocean shows the SAME tile-size dependence the cube does: the 2.01x
211:multinode improvement measured at LL576 was partly a floor effect, and at
212-a production tile the identical code scales substantially better (0.37 ->
213-0.63). Per-device throughput also rises (259 -> 305 Mc/s/GPU at np4).
214-Config is byte-identical between the two rows; only the grid changes.
215-
216-REFUTED EN ROUTE: the np8 leg timed out twice (>90 min still tracing) while
--
221-reproducible, and no compile-time defect is claimed.
222-
223-## Weak scaling at production per-device size (job 26453523)
224-
225-The earlier weak ladders used a 64-row base (0.17M cells/GPU — under the
226:latency floor). Re-run at PRODUCTION size (288 rows × 1152 lon × L20 =
227-6.6M cells/GPU, 1→4 A100, conservation gated): production implicit
228-1.00/0.72/0.70, improved wide-halo+vmix-f32 1.00/0.86/0.85.
229-PRECISION MATCHED (self-audit correction): BOTH ladders compared here are
230-**float32** — the production-tile run is f32, so it is compared against
231-the earlier ladder's f32 rows (eff 0.26/0.25 at nd 2/4), not its f64 rows (both from job 26445836).
232-An earlier revision of this file mislabelled the production-tile run f64
233-and cited the f64 small-base numbers; the direction and size of the effect
234:are unchanged, but the comparison is only valid precision-matched. So the earlier weak ladder measured a below-floor tile rather than a code
235-limit, and the same
236-config that fixes strong scaling also carries weak (+0.15 at nd4). Ideal is
237-flat; the improved arm holds 22.5→22.9 ms while production drifts
238-17.8→25.3 ms.
239-
--
340-
341-With all three ingredients measured on this machine — fabric constants
342-(17.82 us, 64.22 GB/s) and the per-tile single-device compute term — the
343-ocean LL576 f64 implicit ladder finally has a real roofline:
344-
345:| nd | measured | calibrated bound | measured/bound | at % of floor |
346-|---|---|---|---|---|
347-| 2 | 42.71 ms | 34.77 ms | 1.228 | 81 % |
348-| 4 | 23.53 ms | 16.82 ms | 1.398 | 72 % |
349-
350-Bound = per-device compute (32.58 / 14.63 ms) + modelled comm (2.21) +
--
420-DEPENDENT SYNCHRONISATION POINTS, not message aggregation, byte
421-compression, or XLA scheduling flags — three families this campaign has
422-now measured to be null here.
423-
424-Honest answer to "how far from the theoretical limit are we": 72-81 % of a
425:now-calibrated floor, with the shortfall attributable to neither bandwidth
426-nor byte volume.
427-
428-## The mechanism's prediction, TESTED — and the lever it exposes (job 26459382)
429-
430-If exposed dependent sync is the cost, PCG iteration count is the most
--
640-
641-Per-step time is flat 100 -> 600 steps (12.81 -> 12.74 ms). Both arms' heat
642-drift grows ~linearly and is similar between them, consistent with the
643-shared baroclinic/tracer path dominating it.
644-
645:MECHANISM CONFIRMED FROM A THIRD ANGLE. The wide-halo arm records its own
646-message census: **120 standard barotropic messages/step -> 4** (n_loop=30
647-substeps, stencil reach 3, one fixed wide exchange per chunk). A 30x cut in
648-barotropic exchanges is precisely why it wins where sync dominates, and it
649-is the SAME quantity the single_reduce analysis isolated as the half it
650-could not touch (44.1 us/iter of matvec halo). Three independent
--
667-   `tests/parallel/test_latlon_ocean_spmd_wide_halo.py` (4-device SPMD,
668-   2e-4/1e-3). These are TOLERANCE parity, not bit identity — "the filter
669-   sees bit-identical inputs" is too strong; differences are XLA
670-   re-association at the serial tier and larger at the distributed tiers.
671-   ONE REAL BEHAVIOURAL DELTA to disclose: wide-halo requires LOCAL
672:   subcycle clamping, and with an active `eta_floor` the clamp/
673-   redistribution schedule differs from the standard path — a reviewer
674-   should check that config interaction, not filter stability in general.
675-
676-2. **The scheme choice — explicit_substep vs implicit_cn — is a choice
677-   between two EXISTING schemes, not validation of a new one.**
--
686-   config switch that field for scale-out runs".
687-
688-WHAT ACTUALLY REMAINS (scoped to that question): the OMIP override to
689-implicit_cn presumably encodes a preference (dt headroom / stiffness at
690-depth on that config). The remaining sign-off is experiment-level: run the
691:OMIP case with explicit_substep+wide at production dt and confirm the
692-3-seed conservation result (heat 7.1 SE, salt 15.2 SE lower than
693-implicit_cn at 600 steps unforced) holds under forcing. That is a science
694-review of ONE config field on ONE experiment, not a scheme-stability
695-program.
696-
--
709-| cells/rank | subdiv-7 | subdiv-8 |
710-|---|---|---|
711-| 10 200 | — | np64 = 960 ms |
712-| 5 100 | — | np128 = 526 |
713-| **2 600** | **np64 = 165** | **np256 = 356** |
714:| 1 300 | np128 = 92 | (np512 pending) |
715-| 600 | np256 = 56 | — |
716:| 300 | **np512 = 67 (REGRESSES)** | — |
717-
718-Both meshes still scale at 2 600 cells/rank; subdiv-7 keeps gaining down
719-to 600 and only ANTI-SCALES at 300 (56 -> 67 ms). **The turnover tracks
720:work per rank, not rank count** — the tile-floor hypothesis, confirmed
721-on a lane where the two can be separated. Practical consequence: rank
722-counts beyond the campaign's old 64 ceiling keep paying as long as
723-resolution rises with them; subdiv-8 reaches 356 ms at 256 ranks, a
724-count the campaign never previously tested.
725-
726-CAVEATS: single runs, no repeats. The np64 point here (165 ms) is FASTER
727-than the same configuration measured on 1 node earlier (220.6 ms, job
728-26495437) because this job spreads 64 ranks over 4 nodes — the
729-node-spreading effect the campaign already documented; ladders are
730-internally consistent but the two jobs are not interchangeable.
731:The subdiv-8 np512 arm OOM-killed at 128 ranks/node (every rank derives
732-the global mesh); rerun spread over 8 nodes as job 26497704.
733-
734-**GPU CEILING FOUND — the cubed-sphere cannot currently exceed 54 GPUs.**
735-The tiled cube path is bit-identity-validated only at kt=2 (24 devices)
736-and kt=3 (54) (`sharded_dynamics.py:754`); kt=4 (96) falls back to
737-REPLICATING the global state, which is what killed the 96-GPU attempt
738:(job 26495955: "byte size of input/output arguments (83247045120)
739-exceeds the base limit"), and very likely the f64 cube retry that hit
740-the walltime (26495388). This is a VALIDATION limit, not a hardware one,
741-and it is the single biggest blocker to atmospheric scale-out: 28 idle
742-GPU nodes were available and unusable by that lane. Matched triangle
743-resubmitted inside the validated counts (job 26497294): C768@24 (147.5k
744-cols/GPU anchor), C768@54 (65.5k), C1152@54 (147.5k — same tile as the
745-anchor at 2.25x the devices).
746-
747-The lat-lon band decomposition has no such ceiling; its matched pair
748-runs at 64 GPUs (job 26497323): LL720@16 and LL1440@64 both hold 64.8k
749:columns/GPU, with LL720@64 (16.2k) as the sub-floor control.
750-
751:**Cube tile-floor arm, measured (job 26497294):** C768 L60 from 24 to 54
752-GPUs = 19.33 -> 13.93 ms, **1.39x at 2.25x devices, efficiency 0.62** —
753-and the tile only falls to 65.5k cols/GPU, still well ABOVE the ~30k
754:floor. So unlike the CPU lane, the cube's loss here is NOT explained by
755:the tile floor alone; there is real device-count cost to quantify.
756-(Note the same-job C768@24 anchor reads 19.33 ms where the campaign's
757-figure carries 14.09 ms for C768@24 — different lane/protocol between
758-those jobs, so only the within-job 24-vs-54 contrast is used.)
759-
760-**THE CUBE PLATEAU, IDENTIFIED (job 26498347).** The fixed-tile contrast
761:finally ran inside working configs — C512@24 vs C768@54, both 65.5k
762-cols/GPU:
763-
764-| arm | tile | devices | ms/step |
765-|---|---|---|---|
766:| C512 kt=2 | 65.5k | 24 | 14.85 |
767-| C768 kt=3 | 65.5k | 54 | **13.95** |
768-
769-**2.25x the devices carrying 2.25x the problem costs nothing** (1.06x, in
770-the model's favour) — so communication does NOT grow with device count on
771-this lane, and the strong-scaling loss is not a comm wall. Cross-job
--
784-comes from a TWO-POINT fit and assumes the 11.27 ms term is constant as
785-tiles shrink and device count rises. A perimeter-like halo term would
786-FALL with tile size while collective latency could RISE with rank count;
787-the 2.25x fixed-tile contrast supports only "little growth over the
788-tested range", not universality. That single number explains
789:the cube's efficiency 0.62, the empirical tile floor, and the plateau in
790-the figure.
791-
792-**THE SAME STRUCTURE ON LAT-LON.** Two fixed-device (16 GPU) points —
793-LL720 at 64.8k cols/GPU = 4.19 ms and LL1024 at 131k = 5.73 ms — give
794-
--
807-cols/GPU on both sides:
808-
809-| arm | devices | cells | ms/step |
810-|---|---|---|---|
811-| LL1024x2048 | 16 | 4.2 M | 5.73 |
812:| **LL2048x4096** | **64** | **218.1 M** | **6.73** |
813-
814-4x the devices carrying 4x the problem costs **+17 %** — weak-scaling
815-efficiency **0.85**, sustaining **32.4 GCells/s (506 Mcells/s/GPU) on
816-218 million cells**, the campaign's largest atmospheric run by an order
817-of magnitude. So communication grows only weakly with device count here
818-too (the cube's equivalent contrast was free at 2.25x; lat-lon pays 17 %
819-for 4x). Neither lane is comm-limited at these counts — both are limited
820-by the per-step fixed cost above.
821-
822-Scale-out receipts on this lane: LL1536x3072 at 64 GPUs = 4.97 ms (job
823:26498266) and the LL2048 point above.
824-
825-**WHAT THE FIXED TERM IS — ATTRIBUTED (nsys job 26504836): the halo
826-exchange, scaling with tile PERIMETER.** Profiling both tiles at the
827-SAME 24 GPUs isolates it by subtraction:
828-
829-| tile | NCCL time / 12 steps | launches | share of GPU time |
830-|---|---|---|---|
831-| C768, 147.5k cols/GPU | 498.7 ms | 1452 | 66.5 % |
832:| C512, 65.5k cols/GPU | 280.7 ms | 1128 | 65.4 % |
833-
834-The comm term grows **1.78x for a 2.25x larger tile AREA** — close to the
835-sqrt(2.25) = 1.50x a PERIMETER law predicts, and nowhere near the 2.25x
836-an area law would give. That is the mechanism behind the fitted "fixed"
837-11.27 ms: halo cost tracks the tile EDGE while compute tracks the tile
--
854-* whether it explains the CLOSED-LOOP fixed term is NOT yet established.
855-A matching closed-loop profile is running (job 26507936).
856-
857-WHAT THE KERNEL NAMES SHOW (single-shot lane, both tiles, 24 GPUs):
858-
859:| kernel | C768 | C512 | per step |
860-|---|---|---|---|
861-| `ncclDevKernel_SendRecv` (halo) | 311.8 ms / 1440 | 186.7 / 1116 | ~120 launches |
862-| `AllReduce_Sum_f32_RING_LL` | 186.9 ms / **12** | 94.0 / **12** | **exactly 1** |
863-
864-The per-step all-reduce averages **15.6 ms at C768** and scales 1.99x
--
878-instance:
879-
880-    arrival skew = latest NCCL-kernel start - earliest NCCL-kernel start
881-
882-ms-scale skew with microsecond-scale service time on the last-arriving
883:rank confirms the barrier reading; aligned starts implicate the
884-collective or the scheduler instead.
885-
886-**CLOSED-LOOP PROFILE (job 26510470) — the production lane, isolated
887-per-step with a marker kernel** (`loop_add_fusion_3`, exactly one per
888-timed step; naive time-windowing was still catching setup and XLA
--
920-per step (the ocean's wide-halo trick, 120 -> 4 messages, applied to the
921-cube), (c) overlapping halo exchange with interior compute.
922-
923-**A THIRD cube limit — f64 is effectively unrunnable at 24 GPUs.** The
924-tiled lane (the CORRECT >6-GPU vehicle) failed to complete even C384 L60
925:in f64 within a 3 h wall (job 26512794), where the same arm in f32 runs
926-in ~9 ms/step. No output, no error — it never finished compiling. That
927-is consistent with #1370: if setup allocates global-sized buffers, f64
928-doubles them, and the compile/allocation path degrades accordingly. So
929-the cube's f64 GPU column stays EMPTY in the figure, and it is a
930-capability gap rather than a measurement I skipped.
--
942-The campaign's cure for every plateau is a larger tile via higher
943-resolution. Testing that at the largest rank counts failed on BOTH
944-transports, for two DIFFERENT reasons:
945-
946-**CPU (atm icosahedral): subdiv-9 is not supported by the generator.**
947:Jobs 26512349 and 26514768 did not time out in mesh construction as I
948-assumed — they raised immediately:
949-
950-    ValueError: subdivision_level=9 would create 2.62e+06 cells.
951-    Maximum supported level is 8 (655,362 cells). For higher
952-    resolutions, use load_mpas_mesh() with a pre-built mesh file.
--
958-or bench guard stated the answer before I ran the job (the etopo dt
959-warning and the n_lat divisibility check were the others). No pre-built
960-finer mesh is present in the tree, so testing beyond subdiv-8 requires
961-sourcing an MPAS mesh file first.
962-
963:**GPU: blocked by the global-allocation defect (#1370)** — LL2304 wanted
964-102 GB/device, C1152 97-106 GB.
965-
966-**CONSEQUENCE, and it is the campaign's sharpest practical finding:**
967:raising resolution is the ONLY measured cure for the tile-floor plateau,
968-and it is currently unavailable on both transports — capped at subdiv-8
969-on CPU by the mesh generator, and by per-device global allocation on GPU.
970-So the useful rank/device ceilings measured here are NOT hardware limits:
971-
972-| lane | useful ceiling | what caps it |
973-|---|---|---|
974:| atm ico CPU | ~512-1024 ranks at subdiv-8 | mesh generator caps at subdiv-8 |
975:| ocean MPAS CPU | 512 ranks at subdiv-8 | same generator cap + rank-count term |
976-| atm/ocean GPU | 64 GPUs at LL1536-2048 | #1370 global per-device allocation |
977-| cube GPU | 54 GPUs at C768 | kt validation (#1360) + #1370 |
978-
979-Unblocking #1370 and sourcing a subdiv-9+ mesh are therefore worth more
980-than any further tuning: both lanes have headroom that is currently
--
1002-  on this dycore — it alters RK2/RK3 boundary tendencies. Exact
1003-  communication avoidance would need a 3 x radius = 6-cell overlap with
1004-  cube-edge interpolation, and the tiled transport supports only halo
1005-  1/2. That is a new algorithm, not a port.
1006-
1007:**THE BOUND (one step, 86 SendRecv, latency floor 17.8 us measured):**
1008-
1009-| class | n | total |
1010-|---|---|---|
1011-| <=25 us (latency-bound) | 39 | **0.61 ms** |
1012-| 25-100 us | 33 | 1.20 ms |
--
1051-drift cannot accumulate, or overlap the resync exchanges with interior
1052-compute — a scoped dycore-scheduling follow-up, not a bench or config
1053-change. No further profiling is needed; the mechanism chain
1054-(launch-count -> bytes -> skew -> ordering) is now measured end to end.
1055-
1056:## Tripole (ORCA fold) past 4 GPUs — first receipts (job 26512798)
1057-
1058-The fold is the ocean's production topology but had only ever been
1059:measured to 4 GPUs. LL1152x2304 L20 f32, implicit_cn + forced PCG:
1060-
1061-| devices | tile | ms/step | GC/s |
1062-|---|---|---|---|
1063-| 8 | 331.8k cols/GPU | 33.97 | 1.56 |
1064-| 16 | 165.9k | **23.79** | 2.23 |
--
1075-same-job matched contrast (+1.2-3.7 %, job 26493837).
1076-
1077-## The MPAS mesh cap — lifted (subdiv-9 unblocked for 128 GPUs)
1078-
1079-The generator's hard subdiv-8 cap was the CPU-side resolution blocker
1080:and made 128-GPU MPAS floor-starved by construction (subdiv-8 at np128 =
1081-5.1k cells/GPU). Chain shipped 2026-07-30 (codex round-19 design,
1082-commit 70f3ce636):
1083-
1084-* **Cache-or-prewarm policy** for subdiv 9-10: a cache hit always loads;
1085-  a miss RAISES with prewarm instructions unless the process is the
--
1102-  parallelise there. Full f32 ladder np2->128 (np32 from job 26549646):
1103-  19.90 / 14.12 / 6.92 / 7.10 / **8.13** / 5.27 / 6.47 — NON-MONOTONE:
1104-  np32 is WORSE than np16 while np64 is the minimum, and f64 shows the
1105-  same pattern (np32 14.26 vs np64 8.96). This is the np4-dip signature
1106-  at another count — count-specific codegen/fusion behaviour layered on
1107:  the tile floor (the fusion pathology on this lane is already proven
1108-  shape-dependent). Recorded as observed; not chased further at subdiv-8
1109:  since the mesh is below the floor at all these counts anyway.
1110-* Payoff ladder submitted (job 26549775): subdiv-9 at 32/64/128 GPUs =
1111-  81.9k/41.0k/20.5k cells/GPU — the first MPAS many-GPU ladder whose
1112:  lower rungs sit ABOVE the ~30k floor.
1113-
1114-OPERATIONAL NOTE: a Lustre incident mid-implementation left the module
1115-with an undefined constant on disk for ~12 h; two queued jobs (mpas32
1116-26534061, and possibly mpas128's first attempt) died on that NameError
1117-window and were resubmitted post-fix. The git index inode went stale on
--
1121-
1122-## Ocean GPU scale-out to 64 devices — and a CROSS-LANE memory defect (#1370)
1123-
1124-| arm | devices | tile | ms/step |
1125-|---|---|---|---|
1126:| LL1152x2304 L20 | 16 | 165.9k cols/GPU | 19.83 |
1127:| LL1152x2304 L20 | 64 | 41.5k | **11.19** (4.75 GC/s) |
1128:| LL2304x4608 L20 | 64 | 165.9k | **OOM — 102 GB/device** |
1129-
1130-The strong arm reaches 64 GPUs (19.83 -> 11.19 ms, 1.77x for 4x devices,
1131:eff 0.44 — the tile falls to 41.5k, near the ~30k floor, so this is the
1132:floor behaving exactly as the atmosphere's does).
1133-
1134-**The fixed-tile arm could not run, and WHY it could not is the finding.**
1135:LL2304 at 64 GPUs asked for **102.04 GB per device**. The per-device
1136-SHARD is 3.3 M cells — about 0.2 GB for fifteen f32 fields. But the
1137-GLOBAL problem is 212 M cells = 12.7 GB per field-set, and ~8 such
1138-buffers is ~102 GB: an exact match. The allocation tracks the GLOBAL
1139-size, not the shard.
1140-
--
1144-global-sized buffers per device, so RESOLUTION is capped regardless of
1145-device count.** Both lanes shard correctly one size down (ocean LL1152
1146-@64, cube C768 @54), so the sharded step is sound — it is the
1147-setup/allocation path.
1148-
1149:**OCEAN FIXED-TILE CONTRAST, recovered by sizing under the wall (job
1150:26510472).** Instead of retrying LL2304, LL1632 @32 holds the anchor's
1151-tile (166.5k vs 165.9k cols/GPU) at HALF the global size:
1152-
1153-| arm | devices | tile | ms/step |
1154-|---|---|---|---|
1155:| LL1152x2304 | 16 | 165.9k | 19.83 |
1156-| LL1632x3264 | 32 | 166.5k | **19.55** (5.45 GC/s) |
1157-
1158-**2x the devices carrying 2x the problem costs NOTHING** (0.99x,
1159-weak-scaling efficiency 1.01). So the ocean GPU lane joins the
1160-atmosphere: comm does not grow with device count at constant tile
--
1172-| compiled temps | 0.22 | 0.21 | not remat pressure |
1173-| state leaves (per-device) | 0.070 sharded / 0 replicated | same | sharding correct |
1174-| **bytes_in_use** | **1.58 GB** | **3.11 GB** | **tracks GLOBAL size** |
1175-
1176-Per-device residency is a constant **~7.4 global-field equivalents** —
1177:and 7.4x the LL2304 field size is the observed 102 GB wall. So the
1178-defect is NOT step-entry replication (my original hypothesis — refuted
1179-in its specific form) and NOT remat: it is **SETUP-TIME global device
1180-arrays that stay alive after sharding** — the globally-built initial
1181-state, the vertex-mask cache primed FROM the global state, and the
1182-replicated geometry stacks. The cube's level-independent wall fits: its
--
1202-global size. All 13 SPMD gate suites (equivalence/tripole/wide-halo)
1203-pass. One diagnostic casualty, harmless to production: the probe's
1204-OUTER re-jit now refuses ("closing over a multi-process jax.Array"),
1205-because the wrapper closes over the now-sharded stacks — the production
1206-inner jit receives them as ARGUMENTS and is unaffected (the probe's own
1207:step invocation ran). Remaining acceptance: the LL2304@64 wall run.
1208-
1209-WHY THIS IS THE CAMPAIGN'S MOST IMPORTANT BLOCKER: the measured cure for
1210-every plateau is a LARGER TILE, i.e. raising resolution as devices are
1211-added. This defect makes that impossible — adding GPUs cannot buy
1212:resolution — so every GPU lane is pinned at the tile floor. Filed as
1213-**#1370** with the arithmetic; distinct from #1360 (the cube's kt
1214-validation ceiling), and validating kt=4 alone would NOT unblock C1152.
1215-
1216-## Ocean MPAS Voronoi scale-out — and a lane that does NOT obey the tile law
1217-
--
1257-  not have.
1258-* **Partition quality degrades WITH rank count:** sfc costs 1.22x at 32
1259-  ranks but **1.97x** at 128. So the two effects compound, and the
1260-  default `auto` is doing well to land near geometric.
1261-* Practical: pin `--partition-method geometric` (or auto) on this lane;
1262:  sfc is actively harmful at scale. pymetis is absent from `.venv-mpi`,
1263:  so the low-cut METIS arm codex wanted is still unmeasured.
1264-
1265:**512-RANK LADDER (job 26508063):** s7 63.83 ms, s8 194.80 ms.
1266:s8 keeps gaining 256->512 (254.41 -> 194.80 = **1.31x**, at 1 280
1267-cells/rank) while s7 goes flat (65.71 -> 63.83 = 1.03x, at 320
1268:cells/rank — below the 300-600 floor). Same floor as the atmosphere,
1269-reached at a different rank count because the mesh differs. So this lane
1270:DOES scale to 512 ranks when the mesh is large enough; it just pays the
1271-rank-count term on the way.
1272-
1273-UNEXPLAINED, flagged not resolved: s7's per-doubling speedup RISES
1274-(1.29 / 1.43 / 1.57) even in the controlled ladder. Efficiency improving
1275-as the tile shrinks has no physical mechanism I can name; the likeliest
--
1308-does not fire) while f64 np8 ANTI-scaled (20.10 -> 21.42). Adding the
1309-(8, ..., float64) signature entry — receipt: **np8 21.42 -> 18.98
1310-(observed -11.4 %, single run, no CI; the unchanged np4 control at 20.09
1311-supports specificity but does not quantify variance)**. The f64 ladder is now
1312-monotone 38.34/20.09/18.98, and the shape x dtype dependence of the
1313:fusion pathology is confirmed from a second angle (element size shifts
1314-the pathological rung). Gate now carries dtype in the signature; both
1315-entries have same-day receipts.
1316-
1317-**Grid coverage — first receipts for the two unmeasured ocean grids**
1318-(job 26493648, nd1/2/4, both precisions):
--
1328-  bench's own metadata says it: `n_ranks: 1, cells_per_rank_achieved:
1329-  163842` — every arm ran ONE rank on the FULL mesh, because this bench
1330-  decomposes by MPI RANK (its docstring states the SPMD multi-device
1331-  path does not exist by design) and my CUDA_VISIBLE_DEVICES invocation
1332-  never created ranks. The flat curves were the SAME single-device run
1333:  repeated, not a latency floor and not a defect — codex round-13's
1334-  "hypothesis, not verdict" was righter than it knew. What survives:
1335-  single-device timings (s6 ~7 ms f32/f64, s7 ~21.9 ms f32).
1336-
1337-  **THE REAL LADDER (job 26494036, CPU-MPI f64, np1-16, block:cyclic,
1338-  ranks=N verified in metadata): MPAS-ocean SCALES.** s6 (41k cells):
--
1474-## Closed levers (nulls with receipts — do not re-run)
1475-
1476-NCCL_PROTO forcing (default already optimal; LL128 −8–12%), fused-halo on
1477-the implicit arm AND on the wide arm (pad aggregation is not the residual),
1478-`xla_gpu_collective_permute_combine_threshold_bytes` alone,
1479:`--xla_gpu_enable_pipelined_p2p` alone, `LEGOESM_BAROCLINIC_F32` (~+0.5%, within run-to-run spread; job 26451282).
1480-PGLE arm invalid as measured here (the 33-step window catches its
1481-profile+recompile). The +8.5% figure is from the DERECHO lane-T campaign
1482-(see the SOTA review), not reproduced on Levante — rerun long-window if
1483-revisited.
1484-
--
1487-1. **OMIP config sign-off for explicit_substep+wide** (formerly "wide-halo
1488-   stability gates" — reframed 2026-07-26 after codex round-9). Wide-halo
1489-   has tolerance-parity coverage at all three transport tiers (serial
1490-   1e-12, MPI 1e-10, SPMD 2e-4/1e-3) and explicit_substep is an
1491-   established scheme (the model default; benches and OMIP explicitly
1492:   choose implicit_cn). Remaining: the eta_floor x local-clamp config
1493-   interaction, the OMIP case at production dt under forcing, and a
1494-   science sign-off on the f32 vmix solve. Worth 2.01x multinode.
1495-2. **MPAS ico np4 per-device dip** — five hypotheses refuted by
1496-   measurement (see above); needs a GPU op-level profile (nsys / XLA op
1497-   profile of np4 vs np8). A scoped instrumentation project, not a knob.
--
1582-(eff 1.04) and tiled closed loop (14.3 GCells/s); atm lat-lon weak at a
1583-production tile; ocean weak at a production tile; the ico CPU-MPI ladder;
1584-the CPU spread ladder; and the diagnosis tool's halo + overlap phases,
1585-which were found broken and fixed with a contract test.
1586-
1587:## Phase-3 receipts recovered after the 2026-07-31 session drop (2026-08-02)
1588-
1589-Both jobs the dropped session left behind COMPLETED; neither had been
1590-analysed. First read-out below, CORRECTED per codex round-20
1591-(`.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md`,
1592-VERDICT FIX-FIRST, 12 items — the round that caught a Lloyd-mesh confound
1593-in the first draft's weak-scaling claim).
1594-
1595:### 1. METIS / placement A-B (ocean MPAS CPU, job 26600094)
1596-
1597-Matrix at a matched OWNED-cell target (`cells_per_rank_achieved` = 5,120
1598-exactly in every arm; method pinned per arm, never `auto`, which flipped
1599:meaning when pymetis appeared in `.venv-mpi` on 2026-07-31), f64, nlev 20,
1600:32 ranks/node. WET load is NOT matched under metis — per-rank wet
1601:cell-levels min/max: geometric@32 100,740–102,420; metis@32
1602:96,800–102,720; metis@128 **78,000–102,880** (one rank 24 % under the
1603:mean) — METIS balances owned cells, not wet cells, on this bathymetry.
1604-
1605-| arm | config | ms/step |
1606-|---|---|---|
1607-| A | s7 np32 geometric, block:cyclic | 189.82 |
1608-| B | s8 np128 geometric, block:cyclic | 308.96 |
1609:| C | s7 np32 metis, block:cyclic | 191.99 |
1610:| D | s8 np128 metis, block:cyclic | 333.39 |
1611:| E | s8 np128 metis, block:block | 537.91 |
1612-
1613:* **This METIS configuration LOSES to geometric at s8/np128** (D/B =
1614-  +7.9 %), despite the better offline cut (partq s8@np128: edge_cut
1615-  1.89 % vs 2.01 %, halo mean 592 vs 629). Scale-out term (s7@32 ->
1616-  s8@128, which crosses 1 -> 4 nodes as well as 4x ranks — NOT a pure
1617:  rank-count isolate): geometric 1.628, metis 1.736. The offline-quality
1618:  -> step-time inference FAILS on this lane; part of metis's loss is
1619-  PLAUSIBLY its own wet-load imbalance (above). Scope: closes the
1620:  "swap in METIS as-is" lever on the CPU-MPI ocean lane; does NOT rule
1621-  out partition/mapping improvements generally (e.g. wet-cell-weighted
1622:  METIS was NOT tested).
1623-* **`block:cyclic` stays mandatory on packed CPU lanes** (E/D = 1.61x at
1624-  a byte-identical partition). NOTE the second `--distribution` field is
1625-  the INTRA-NODE (socket) distribution — both arms place ranks on nodes
1626-  identically; the swing is socket-level. Mechanism (per-socket
1627-  memory-bandwidth balance) PLAUSIBLE, consistent with the np16 Milan
1628-  2.13x receipt; never instrumented with bandwidth counters.
1629-* Caveats: timing-only receipt — no parity/conservation gate ran in
1630:  these arms, and the CPU nodes emit `UCX WARN transports
1631-  'cuda_copy','cuda_ipc','gdr_copy' are not available` (the _env.sh GPU
1632:  UCX_TLS list on a CPU node; UCX falls back to rc/sm — cosmetic for
1633-  timing, but a "production config" claim would need a gated arm).
1634-
1635-### 2. subdiv-9 payoff ladder (atm MPAS ico GPU, job 26600095)
1636-
1637-f32, sfc partition (padded-128 reorder), lloyd=0 LABELLED SYNTHETIC
--
1647-
1648-* **np64 = 7.10 GC/s is the best MPAS-atmosphere number on this
1649-  synthetic dynamics-only bench** (2.19x the subdiv-8 best, 3.23 GC/s at
1650-  its np64: 655,362 natural cells x 26 lev / 5.27 ms).
1651-* Strong 32->64 speedup 1.299 (eff 0.65); 64->128 speedup 0.836 —
1652:  ANTI-scales at 20.5k cells/GPU. CONSISTENT WITH the ~30k floor seen on
1653-  the other lanes (single unreplicated point on a lane with known
1654-  count-specific codegen variation — not by itself proof).
1655:* **RETRACTED (codex round-20): the first draft's s8->s9 "weak
1656-  efficiency 0.55–0.74" pairs and the "~1.4x per 4x ranks GPU rank-count
1657-  term".** Confounds: (a) every existing s8 receipt is the generator's
1658:  default PRODUCTION Lloyd mesh, while s9 is lloyd=0 — different mesh
1659-  family, not the same protocol; (b) two comparator points came from the
1660-  np2-16 ladder (jobs 26454476/26454618), not the np32-128 extension
1661-  rows; (c) the three ratios are 1.80/1.35/1.41 — not "consistent
1662-  ~1.4x". A matched s8 lloyd=0 np8/16/32 rerun is submitted (see below);
1663-  no weak-scaling direction is claimed until it lands.
1664-
1665-### Next receipts submitted 2026-08-02
1666-
1667:1. **Ensemble receipt** (`scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch`)
1668:   — codex lever #1: 4 concurrent 32-GPU s9 replicas on disjoint 8-node
1669:   sets vs SAME-JOB solo controls bracketing phase B (solo before AND
1670-   after, ordering counterbalanced). steps=5000 so the stepping window
1671-   (~60 s) dwarfs launch skew; per-arm `SLURM_STEP_NODELIST` +
1672:   wall-clock brackets logged as overlap evidence. CONFIRM bar:
1673:   max(replica) <= 1.10x mean(solo) => guaranteed aggregate >= 3.64x the
1674:   32-GPU solo rate (>= 19.9 GC/s if solo reproduces 5.47) = ~3.3x the
1675-   observed 128-GPU single-trajectory rate. REFUTE: replica slowdown
1676-   >10 % = the fabric-contention term, quantified per replica.
1677:2. **s8 lloyd=0 matched rerun** — de-confounds the weak pair: np8/16/32
1678-   (81.9k/41.0k/20.5k cells/GPU) on the SAME lloyd=0 family, same sfc +
1679:   `--reorder-for 128`, same steps/warmup as the s9 ladder. Weak pairs
1680-   recomputed only from these.
1681-
1682:### 3. Recovered phase-2 receipt: lat-lon atmosphere at 128 GPUs (job 26534060, ran 2026-07-30, unanalysed until now)
1683-
1684:LL2048x4096 L26, same bench + protocol (steps 12 / warmup 3) as the
1685-@64 row (job 26502539, f32 6.73 ms):
1686-
1687-| arm | ms/step | GC/s (col-levels) |
1688-|---|---|---|
1689-| f32 @128 (32.7k cols/GPU) | 5.58 | **39.11** |
1690-| f64 @128 | 9.60 | 22.72 |
1691-
1692-f32 strong 64->128: 1.21x for 2x devices (eff 0.60) with the tile at
1693:32.7k cols/GPU — at/near the floor, consistent with the tile law.
1694-39.1 GC/s is the highest measured throughput of ANY lane in the
1695-campaign. The companion oc128 (26534067) FAILED pre-#1370-fix with the
1696-109.5 GB resident-args signature; retry submitted post-fix (below).
1697-
1698:## Hundreds-of-devices push (user directive 2026-08-02)
1699-
1700:"Push the scaling to hundreds of CPUs and GPUs for lat-lon and MPAS on
1701-GPUs." Machine ceiling: 56 nodes x 4 = 224 a100_80 GPUs; compute
1702-partition effectively unbounded for our rank counts. Submitted set:
1703-
1704-| job | what | devices | why |
1705-|---|---|---|---|
1706:| 26628021 | s9 ensemble contention | 128 GPU (4x32) | lever #1 receipt |
1707:| 26628071 | oc LL2304 retry post-#1370 | 128 GPU | pre-fix failure was resident-args; predicted PASS at ~0.10 GB/dev residency |
1708:| 26628072 | atm LL2304 @96/@192 + LL2880 @192 | 96-192 GPU | LL2048 does not divide 192; LL2880@192 = 86.4k cols/GPU ABOVE floor |
1709:| 26628073 | atm lat-lon 2-D pencil r512 np64-512 | 512 CPU ranks | hundreds-of-CPUs lat-lon (wall-pole lane, labelled) |
1710:| 26628074 | subdiv-10 lloyd0 prewarm | 1 CPU | unlocks MPAS 128-224 GPUs ABOVE floor (81.9k-46.8k cells/GPU) |
1711:| 26628076 | s8 lloyd0 np8/16/32 | 32 GPU | weak-pair de-confound (codex r20 item 5) |
1712-
1713-s10 ladder (128/192/224 GPUs) submits once 26628074's cache lands.

exec
/usr/bin/bash -lc "sed -n '1587,1735p' docs/performance/scaling/levante_campaign_2026-07-24.md && printf '\\n--- ensemble ---\\n' && nl -ba scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch && printf '\\n--- s8 ladder ---\\n' && nl -ba scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch && printf '\\n--- latlon hundreds ---\\n' && nl -ba scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch && printf '\\n--- latlon2d hundreds ---\\n' && nl -ba scripts/cluster/scaling_levante/atm_latlon2d_cpu_hundreds.sbatch && printf '\\n--- prewarm ---\\n' && nl -ba scripts/cluster/scaling_levante/prewarm_s10.sbatch && printf '\\n--- plot ---\\n' && nl -ba scripts/plot/plot_scaling_paper_figure.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 374ms:
## Phase-3 receipts recovered after the 2026-07-31 session drop (2026-08-02)

Both jobs the dropped session left behind COMPLETED; neither had been
analysed. First read-out below, CORRECTED per codex round-20
(`.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md`,
VERDICT FIX-FIRST, 12 items — the round that caught a Lloyd-mesh confound
in the first draft's weak-scaling claim).

### 1. METIS / placement A-B (ocean MPAS CPU, job 26600094)

Matrix at a matched OWNED-cell target (`cells_per_rank_achieved` = 5,120
exactly in every arm; method pinned per arm, never `auto`, which flipped
meaning when pymetis appeared in `.venv-mpi` on 2026-07-31), f64, nlev 20,
32 ranks/node. WET load is NOT matched under metis — per-rank wet
cell-levels min/max: geometric@32 100,740–102,420; metis@32
96,800–102,720; metis@128 **78,000–102,880** (one rank 24 % under the
mean) — METIS balances owned cells, not wet cells, on this bathymetry.

| arm | config | ms/step |
|---|---|---|
| A | s7 np32 geometric, block:cyclic | 189.82 |
| B | s8 np128 geometric, block:cyclic | 308.96 |
| C | s7 np32 metis, block:cyclic | 191.99 |
| D | s8 np128 metis, block:cyclic | 333.39 |
| E | s8 np128 metis, block:block | 537.91 |

* **This METIS configuration LOSES to geometric at s8/np128** (D/B =
  +7.9 %), despite the better offline cut (partq s8@np128: edge_cut
  1.89 % vs 2.01 %, halo mean 592 vs 629). Scale-out term (s7@32 ->
  s8@128, which crosses 1 -> 4 nodes as well as 4x ranks — NOT a pure
  rank-count isolate): geometric 1.628, metis 1.736. The offline-quality
  -> step-time inference FAILS on this lane; part of metis's loss is
  PLAUSIBLY its own wet-load imbalance (above). Scope: closes the
  "swap in METIS as-is" lever on the CPU-MPI ocean lane; does NOT rule
  out partition/mapping improvements generally (e.g. wet-cell-weighted
  METIS was NOT tested).
* **`block:cyclic` stays mandatory on packed CPU lanes** (E/D = 1.61x at
  a byte-identical partition). NOTE the second `--distribution` field is
  the INTRA-NODE (socket) distribution — both arms place ranks on nodes
  identically; the swing is socket-level. Mechanism (per-socket
  memory-bandwidth balance) PLAUSIBLE, consistent with the np16 Milan
  2.13x receipt; never instrumented with bandwidth counters.
* Caveats: timing-only receipt — no parity/conservation gate ran in
  these arms, and the CPU nodes emit `UCX WARN transports
  'cuda_copy','cuda_ipc','gdr_copy' are not available` (the _env.sh GPU
  UCX_TLS list on a CPU node; UCX falls back to rc/sm — cosmetic for
  timing, but a "production config" claim would need a gated arm).

### 2. subdiv-9 payoff ladder (atm MPAS ico GPU, job 26600095)

f32, sfc partition (padded-128 reorder), lloyd=0 LABELLED SYNTHETIC
scaling mesh, executed padded n_cells = 2,621,568 (natural 2,621,442),
L26; steps 12 / warmup 3; physics=none dynamics-only bench; np32 row's
git-SHA field absent (provenance = job script + this doc):

| GPUs | cells/GPU | ms/step | GC/s (cell-levels) |
|---|---|---|---|
| 32 | 81.9k | 12.47 | 5.47 |
| 64 | 41.0k | 9.60 | 7.10 |
| 128 | 20.5k | 11.48 | 5.94 |

* **np64 = 7.10 GC/s is the best MPAS-atmosphere number on this
  synthetic dynamics-only bench** (2.19x the subdiv-8 best, 3.23 GC/s at
  its np64: 655,362 natural cells x 26 lev / 5.27 ms).
* Strong 32->64 speedup 1.299 (eff 0.65); 64->128 speedup 0.836 —
  ANTI-scales at 20.5k cells/GPU. CONSISTENT WITH the ~30k floor seen on
  the other lanes (single unreplicated point on a lane with known
  count-specific codegen variation — not by itself proof).
* **RETRACTED (codex round-20): the first draft's s8->s9 "weak
  efficiency 0.55–0.74" pairs and the "~1.4x per 4x ranks GPU rank-count
  term".** Confounds: (a) every existing s8 receipt is the generator's
  default PRODUCTION Lloyd mesh, while s9 is lloyd=0 — different mesh
  family, not the same protocol; (b) two comparator points came from the
  np2-16 ladder (jobs 26454476/26454618), not the np32-128 extension
  rows; (c) the three ratios are 1.80/1.35/1.41 — not "consistent
  ~1.4x". A matched s8 lloyd=0 np8/16/32 rerun is submitted (see below);
  no weak-scaling direction is claimed until it lands.

### Next receipts submitted 2026-08-02

1. **Ensemble receipt** (`scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch`)
   — codex lever #1: 4 concurrent 32-GPU s9 replicas on disjoint 8-node
   sets vs SAME-JOB solo controls bracketing phase B (solo before AND
   after, ordering counterbalanced). steps=5000 so the stepping window
   (~60 s) dwarfs launch skew; per-arm `SLURM_STEP_NODELIST` +
   wall-clock brackets logged as overlap evidence. CONFIRM bar:
   max(replica) <= 1.10x mean(solo) => guaranteed aggregate >= 3.64x the
   32-GPU solo rate (>= 19.9 GC/s if solo reproduces 5.47) = ~3.3x the
   observed 128-GPU single-trajectory rate. REFUTE: replica slowdown
   >10 % = the fabric-contention term, quantified per replica.
2. **s8 lloyd=0 matched rerun** — de-confounds the weak pair: np8/16/32
   (81.9k/41.0k/20.5k cells/GPU) on the SAME lloyd=0 family, same sfc +
   `--reorder-for 128`, same steps/warmup as the s9 ladder. Weak pairs
   recomputed only from these.

### 3. Recovered phase-2 receipt: lat-lon atmosphere at 128 GPUs (job 26534060, ran 2026-07-30, unanalysed until now)

LL2048x4096 L26, same bench + protocol (steps 12 / warmup 3) as the
@64 row (job 26502539, f32 6.73 ms):

| arm | ms/step | GC/s (col-levels) |
|---|---|---|
| f32 @128 (32.7k cols/GPU) | 5.58 | **39.11** |
| f64 @128 | 9.60 | 22.72 |

f32 strong 64->128: 1.21x for 2x devices (eff 0.60) with the tile at
32.7k cols/GPU — at/near the floor, consistent with the tile law.
39.1 GC/s is the highest measured throughput of ANY lane in the
campaign. The companion oc128 (26534067) FAILED pre-#1370-fix with the
109.5 GB resident-args signature; retry submitted post-fix (below).

## Hundreds-of-devices push (user directive 2026-08-02)

"Push the scaling to hundreds of CPUs and GPUs for lat-lon and MPAS on
GPUs." Machine ceiling: 56 nodes x 4 = 224 a100_80 GPUs; compute
partition effectively unbounded for our rank counts. Submitted set:

| job | what | devices | why |
|---|---|---|---|
| 26628021 | s9 ensemble contention | 128 GPU (4x32) | lever #1 receipt |
| 26628071 | oc LL2304 retry post-#1370 | 128 GPU | pre-fix failure was resident-args; predicted PASS at ~0.10 GB/dev residency |
| 26628072 | atm LL2304 @96/@192 + LL2880 @192 | 96-192 GPU | LL2048 does not divide 192; LL2880@192 = 86.4k cols/GPU ABOVE floor |
| 26628073 | atm lat-lon 2-D pencil r512 np64-512 | 512 CPU ranks | hundreds-of-CPUs lat-lon (wall-pole lane, labelled) |
| 26628074 | subdiv-10 lloyd0 prewarm | 1 CPU | unlocks MPAS 128-224 GPUs ABOVE floor (81.9k-46.8k cells/GPU) |
| 26628076 | s8 lloyd0 np8/16/32 | 32 GPU | weak-pair de-confound (codex r20 item 5) |

s10 ladder (128/192/224 GPUs) submits once 26628074's cache lands.

--- ensemble ---
     1	#!/bin/bash -l
     2	#SBATCH --job-name=mpas_s9_ens
     3	#SBATCH --account=bb1596_gpu
     4	#SBATCH --partition=gpu
     5	#SBATCH --constraint=a100_80
     6	#SBATCH --nodes=32
     7	#SBATCH --gpus-per-node=4
     8	#SBATCH --exclusive
     9	#SBATCH --mem=0
    10	#SBATCH --time=01:30:00
    11	#SBATCH --output=mpas_s9_ens.%j.log
    12	# ENSEMBLE-PARALLELISM RECEIPT (codex 2026-07-31 consult lever #1; design
    13	# fixed per codex round-20 items 9-12).  The tile floor caps ONE
    14	# trajectory's strong scaling; past it the honest use of 128 GPUs is K
    15	# independent replicas at the per-trajectory sweet spot.  This job
    16	# measures the only thing that can refute that: FABRIC CONTENTION between
    17	# replicas sharing the IB tree.
    18	#
    19	#   solo_pre  : ONE 32-GPU s9 run, other 24 nodes idle
    20	#   phase B   : FOUR concurrent 32-GPU s9 runs on disjoint 8-node sets
    21	#               (SLURM_STEP_NODELIST is per-step -> per-replica
    22	#               coordinator autodetect; jobid-derived port shared but
    23	#               hosts differ)
    24	#   solo_post : solo again AFTER phase B (brackets ordering/thermal
    25	#               drift; contrast uses mean of the two solos)
    26	#
    27	# Falsifiability, written BEFORE submit:
    28	#   numbers : 2 solo + 4 replica steady_median_ms
    29	#   CONFIRM : max(replica) <= 1.10 x mean(solo) -> guaranteed aggregate
    30	#             >= 4/1.10 = 3.64x the 32-GPU solo rate (~19.9 GC/s if solo
    31	#             reproduces 5.47) = ~3.3x the observed 128-GPU
    32	#             single-trajectory rate.  NOT "4x": 1.10 is the bar, the
    33	#             margin below it is the measured contention.
    34	#   REFUTE  : any replica > 1.10x solo -> contention term, quantified
    35	#             per replica.
    36	# Protocol: config identical to job 26600095 np32 rung (sfc, lloyd 0,
    37	# f32, padded-128 reorder) EXCEPT steps 5000 / warmup 100 so the stepping
    38	# window (~60 s at 12.5 ms/step) dwarfs launch skew between replicas --
    39	# overlap is EVIDENCED, not assumed, by the per-step Start/End + NodeList
    40	# table sacct prints at the end.  Absolute ms/step is therefore only
    41	# compared WITHIN this job (solo vs replicas), never against the
    42	# steps-12 ladder rows.
    43	set -uo pipefail
    44	SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
    45	export JAX_PLATFORMS=cuda,cpu
    46	export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
    47	export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
    48	export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
    49	source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
    50	cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
    51	OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/mpas_s9_ens_j${SLURM_JOB_ID}}"
    52	mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
    53	rc=0
    54	
    55	run_arm () { # tag  (one 32-GPU replica on 8 disjoint nodes)
    56	  echo "[$1] launch epoch=$(date +%s.%N)"
    57	  JAX_ENABLE_X64=0 srun --nodes=8 --ntasks=32 --ntasks-per-node=4 \
    58	      --gpus-per-node=4 --gpu-bind=none --exact --kill-on-bad-exit=1 \
    59	      --job-name="arm_$1" \
    60	    "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
    61	      --multicontroller --n-devices 32 \
    62	      --subdivision 9 --nlev 26 --steps 5000 --warmup 100 --lloyd 0 \
    63	      --partition-method sfc --reorder-for 128 \
    64	      --out "$OUTDIR/$1.jsonl"
    65	  s=$?
    66	  echo "[$1] exit=$s epoch=$(date +%s.%N)"
    67	  return $s
    68	}
    69	
    70	echo "=== solo_pre: 1x32 GPUs, 24 nodes idle ==="
    71	run_arm solo_pre || { echo "solo_pre FAILED"; rc=1; }
    72	
    73	echo "=== phase B: 4 concurrent 32-GPU replicas ==="
    74	pids=(); tags=(rep0 rep1 rep2 rep3)
    75	for T in "${tags[@]}"; do run_arm "$T" & pids+=($!); done
    76	for i in "${!pids[@]}"; do
    77	  wait "${pids[$i]}" || { echo "${tags[$i]} FAILED"; rc=1; }
    78	done
    79	
    80	echo "=== solo_post: 1x32 GPUs, 24 nodes idle ==="
    81	run_arm solo_post || { echo "solo_post FAILED"; rc=1; }
    82	
    83	echo "=== RESULTS ==="
    84	for T in solo_pre rep0 rep1 rep2 rep3 solo_post; do
    85	  F="$OUTDIR/$T.jsonl"
    86	  "$PY" -c "
    87	import json,sys
    88	try:
    89	    d=json.loads(open('$F').readline())
    90	    ms=d['steady_median_ms']
    91	except Exception as e:
    92	    print('$T: MISSING/UNPARSEABLE ->', e); sys.exit(1)
    93	print(f'$T: {ms:8.2f} ms  {d.get(\"mcells_per_s\",0)/1000:.2f} GC/s')" \
    94	    || { echo "$T receipt invalid"; rc=1; }
    95	done
    96	
    97	echo "=== overlap evidence: per-step nodelist + wall window ==="
    98	sacct -j "$SLURM_JOB_ID" \
    99	  --format=JobID%18,JobName%12,NodeList%45,Start,End,State -P || true
   100	echo "DONE rc=$rc"; exit $rc

--- s8 ladder ---
     1	#!/bin/bash -l
     2	#SBATCH --job-name=mpas_s8_l0
     3	#SBATCH --account=bb1596_gpu
     4	#SBATCH --partition=gpu
     5	#SBATCH --constraint=a100_80
     6	#SBATCH --nodes=8
     7	#SBATCH --gpus-per-node=4
     8	#SBATCH --exclusive
     9	#SBATCH --mem=0
    10	#SBATCH --time=01:30:00
    11	#SBATCH --output=mpas_s8_l0.%j.log
    12	# DE-CONFOUND RERUN (codex round-20 item 5): every prior subdiv-8 GPU
    13	# receipt is the generator's default PRODUCTION-Lloyd mesh, while the
    14	# subdiv-9 ladder (job 26600095) is the lloyd=0 synthetic family — so no
    15	# s8-vs-s9 weak-scaling pair was protocol-clean.  This ladder reruns s8
    16	# np8/16/32 on the SAME lloyd=0 family, same sfc + --reorder-for 128,
    17	# same steps/warmup as the s9 ladder.  Weak pairs at matched cells/GPU
    18	# (81.9k / 41.0k / 20.5k) are computed ONLY from these rows vs 26600095.
    19	#
    20	# Falsifiability, written BEFORE submit:
    21	#   numbers : s8-lloyd0 np8/16/32 steady_median_ms
    22	#   CONFIRM (a matched-tile scale-out term exists): s9/s8 ratios at
    23	#             matched cells/GPU stay well above 1 (prior draft saw
    24	#             1.80/1.35/1.41 on the CONFOUNDED pairs)
    25	#   REFUTE  : ratios collapse toward ~1.0 -> the draft's "term" was the
    26	#             Lloyd-mesh confound, and MPAS-GPU weak scaling is near
    27	#             ideal at matched tile.
    28	# Mesh: prewarmed into LEGOESM_MESH_CACHE_DIR (subdiv-8 is below the
    29	# big-mesh refuse threshold, so a cache miss falls back to in-process
    30	# builds — slower, still correct).
    31	set -uo pipefail
    32	SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
    33	export JAX_PLATFORMS=cuda,cpu
    34	export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
    35	export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
    36	export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
    37	source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
    38	cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
    39	OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/mpas_s8_l0_j${SLURM_JOB_ID}}"
    40	mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
    41	rc=0
    42	for NP in 8 16 32; do
    43	  NODES=$(( NP / 4 ))
    44	  echo "=== s8 lloyd0 np=$NP f32 (matched to s9 ladder protocol) ==="
    45	  JAX_ENABLE_X64=0 srun --nodes="$NODES" --ntasks="$NP" --ntasks-per-node=4 \
    46	      --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 \
    47	    "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
    48	      --multicontroller --n-devices "$NP" \
    49	      --subdivision 8 --nlev 26 --steps 12 --warmup 3 --lloyd 0 \
    50	      --partition-method sfc --reorder-for 128 \
    51	      --out "$OUTDIR/np${NP}.jsonl" || { echo "np$NP FAILED"; rc=1; }
    52	done
    53	echo "=== RESULTS (cells/GPU: 81.9k / 41.0k / 20.5k) ==="
    54	for NP in 8 16 32; do
    55	  F="$OUTDIR/np${NP}.jsonl"
    56	  "$PY" -c "
    57	import json,sys
    58	try:
    59	    d=json.loads(open('$F').readline())
    60	    print(f'np$NP: {d[\"steady_median_ms\"]:8.2f} ms')
    61	except Exception as e:
    62	    print('np$NP: MISSING/UNPARSEABLE ->', e); sys.exit(1)" \
    63	    || { echo "np$NP receipt invalid"; rc=1; }
    64	done
    65	echo "DONE rc=$rc"; exit $rc

--- latlon hundreds ---
     1	#!/bin/bash -l
     2	#SBATCH --job-name=atm_ll_192
     3	#SBATCH --account=bb1596_gpu
     4	#SBATCH --partition=gpu
     5	#SBATCH --constraint=a100_80
     6	#SBATCH --nodes=48
     7	#SBATCH --gpus-per-node=4
     8	#SBATCH --exclusive
     9	#SBATCH --mem=0
    10	#SBATCH --time=02:00:00
    11	#SBATCH --output=atm_ll_192.%j.log
    12	# HUNDREDS-OF-GPUS lat-lon atmosphere (user directive 2026-08-02).
    13	# LL2048 does not divide 192 ranks (2048/192 non-integer bands), so the
    14	# >128 strong pair moves to LL2304 (96 and 192 both divide 2304) and the
    15	# above-floor "hundreds" point is LL2880x5760 @192 = 86.4k cols/GPU.
    16	#   arm 1: LL2304x4608 @ 96  f32 (55.3k cols/GPU)
    17	#   arm 2: LL2304x4608 @192  f32 (27.6k cols/GPU — floor territory;
    18	#          predicted flat-to-anti per the tile law)
    19	#   arm 3: LL2880x5760 @192  f32 (86.4k cols/GPU — the honest 192-GPU
    20	#          working point; predicted to hold ~existing GC/s levels)
    21	# Context anchors (same bench, steps 12/warmup 3): LL2048@128 f32
    22	# 5.58 ms 39.11 GC/s (job 26534060).
    23	set -uo pipefail
    24	SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
    25	export JAX_PLATFORMS=cuda,cpu
    26	export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
    27	export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
    28	source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
    29	cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
    30	OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/atm_ll192_j${SLURM_JOB_ID}}"
    31	mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
    32	rc=0
    33	run_arm () { # nlat nlon np tag
    34	  echo "=== atm LL$1x$2 @$3 f32 ==="
    35	  JAX_ENABLE_X64=0 srun --ntasks="$3" --ntasks-per-node=4 \
    36	      --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 \
    37	    "$PY" scripts/bench/bench_atm_latlon_spmd_scaling.py \
    38	      --multicontroller --n-devices "$3" --mode strong \
    39	      --n-lat "$1" --n-lon "$2" --nlev 26 \
    40	      --steps 12 --warmup 3 \
    41	      --out "$OUTDIR/$4.jsonl" || { echo "$4 FAILED"; rc=1; }
    42	}
    43	run_arm 2304 4608  96 LL2304_f32_np96
    44	run_arm 2304 4608 192 LL2304_f32_np192
    45	run_arm 2880 5760 192 LL2880_f32_np192
    46	echo "=== RESULTS ==="
    47	for F in "$OUTDIR"/*.jsonl; do
    48	  "$PY" -c "
    49	import json,os
    50	try:
    51	    d=json.loads(open('$F').readline())
    52	    print(f\"{os.path.basename('$F'):24s} {d['steady_median_ms']:8.2f} ms {d.get('mcells_per_s',0)/1000:6.2f} GC/s\")
    53	except Exception as e:
    54	    print(os.path.basename('$F'), 'UNPARSEABLE', e)"
    55	done
    56	echo "DONE rc=$rc"; exit $rc

--- latlon2d hundreds ---
     1	#!/bin/bash -l
     2	#SBATCH --job-name=cpu_ll2d
     3	#SBATCH --account=bb1596
     4	#SBATCH --partition=compute
     5	#SBATCH --nodes=16
     6	#SBATCH --exclusive
     7	#SBATCH --time=04:00:00
     8	#SBATCH --output=cpu_ll2d.%j.log
     9	# HUNDREDS-OF-CPUS lat-lon atmosphere (user directive 2026-08-02): the
    10	# 2-D pencil path (wall poles, labelled throughput lane — NOT the pole
    11	# fold; pole-matched A/B remains open) at r512 (512x1024 = 524k cols) so
    12	# np512 still holds ~1k cols/rank.  Self-contained strong ladder
    13	# np64->512, f64 moist, 32 rpn, block:cyclic per the placement receipt.
    14	set -uo pipefail
    15	SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
    16	export JAX_PLATFORMS=cpu
    17	export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
    18	export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv-mpi/bin/python}"
    19	source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
    20	cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
    21	OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/cpu_ll2d_j${SLURM_JOB_ID}}"
    22	mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
    23	rc=0
    24	for NP in 64 128 256 512; do
    25	  NODES=$(( NP / 32 )); [ "$NODES" -lt 1 ] && NODES=1
    26	  echo "--- atm latlon 2-D pencil r512 f64 np=$NP ---"
    27	  JAX_ENABLE_X64=1 srun --nodes="$NODES" --ntasks="$NP" --ntasks-per-node=32 \
    28	      --distribution=block:cyclic --cpu-bind=cores --kill-on-bad-exit=1 \
    29	    "$PY" scripts/bench/run_cpu_mpi_scaling.py \
    30	      --grid latlon --latlon-2d --mode single --physics moist \
    31	      --precision float64 --n-levels 26 --resolution 512 \
    32	      --n-warmup 2 --n-timing 6 \
    33	      --output-dir "$OUTDIR/np${NP}" < /dev/null || { echo "np$NP FAILED"; rc=1; }
    34	done
    35	echo "=== RESULTS ==="
    36	for NP in 64 128 256 512; do
    37	  F=$(find "$OUTDIR/np${NP}" -name "*.json" 2>/dev/null | head -1)
    38	  [ -n "$F" ] && "$PY" -c "
    39	import json; d=json.load(open('$F')); r=d['results'][-1] if isinstance(d.get('results'),list) else d
    40	print(f'np$NP: {(r.get(\"ms_per_step\") or r.get(\"time_per_step_ms\")):9.2f} ms')" 2>/dev/null
    41	done
    42	echo "DONE rc=$rc"; exit $rc

--- prewarm ---
     1	#!/bin/bash -l
     2	#SBATCH --job-name=prewarm_s10
     3	#SBATCH --account=bb1596
     4	#SBATCH --partition=shared
     5	#SBATCH --ntasks=1
     6	#SBATCH --cpus-per-task=8
     7	#SBATCH --mem=120G
     8	#SBATCH --time=10:00:00
     9	#SBATCH --output=prewarm_s10.%j.log
    10	# Prewarm subdiv-10 lloyd=0 (10.5M cells, ~7 GB npz) into the shared
    11	# mesh cache — unlocks MPAS at 128-224 GPUs ABOVE the ~30k tile floor
    12	# (np128 = 81.9k, np224 = 46.8k cells/GPU).  s9 (2.62M) built in 67 min;
    13	# s10 estimated ~4-5 h.
    14	set -uo pipefail
    15	SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
    16	export JAX_PLATFORMS=cpu
    17	export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
    18	export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
    19	export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
    20	source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
    21	cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
    22	"$PY" scripts/data/prewarm_voronoi_mesh.py --level 10 --lloyd 0

--- plot ---
     1	"""Publication figure: strong scaling per grid, per precision, vs ideal.
     2	
     3	One panel per (component, grid). Each panel plots measured ms/step against
     4	device count on log-log axes, one line per precision, with a DASHED IDEAL
     5	line anchored at each series' own base point (t_base * n_base / n).
     6	
     7	Every number is a measured Levante receipt; the SOURCES table below carries
     8	the SLURM job id for each series so a reader can trace any point. Panels
     9	with only one precision measured say so in-panel rather than leaving the
    10	reader to guess — no interpolation, no fabricated series.
    11	
    12	Anchoring note: ideal lines are anchored at each series' FIRST measured
    13	point. Where that base leg is cache-disadvantaged (a single device holding
    14	the whole problem), the measured curve can sit BELOW ideal; that is a
    15	property of the base leg, not superlinear parallelism, and is flagged in
    16	the caption rather than hidden by re-anchoring.
    17	
    18	Usage
    19	-----
    20	    python scripts/plot/plot_scaling_paper_figure.py --out fig_scaling.pdf
    21	"""
    22	from __future__ import annotations
    23	
    24	import argparse
    25	
    26	import matplotlib
    27	matplotlib.use("Agg")
    28	import matplotlib.pyplot as plt
    29	from matplotlib.lines import Line2D
    30	
    31	# --- Measured data -------------------------------------------------------
    32	# (devices, ms/step). Job ids are the provenance for each series.
    33	SOURCES = {
    34	    "atm_latlon": "26450848/26453240/26449147 (f32), 26494902 (f64), "
    35	                  "LL2048@64 26502539, LL2048@128 26534060",
    36	    "atm_cube": "26452894/26453782",
    37	    "atm_mpas": "26454476/26454618/26486288/26493638/26493734, "
    38	                "s8 np32-128 26549646/26538474, s9 26600095",
    39	    "atm_ico_cpu": "26495083 (f32), 26495437 (f64) — both block:cyclic",
    40	    "oc_latlon": "26460444-501/26460365/26493592",
    41	    "oc_tripole": "26493837/26493648",
    42	    "oc_mpas": "26494036 (f64), 26494908 (f32)",
    43	}
    44	
    45	PANELS = [
    46	    dict(
    47	        key="atm_latlon", title="lat–lon", sub="720×1440 L26 · A100 NCCL",
    48	        series=[("float32", [(4, 7.72), (8, 5.40), (16, 3.54)]),
    49	                ("float64", [(4, 16.11), (8, 11.34), (16, 5.62)]),
    50	                ("float32 (LL2048)", [(64, 6.73), (128, 5.58)]),
    51	                ],
    52	        scatter=[("LL1536 @64", 64, 4.97), ("LL2048 f64 @128", 128, 9.60)],
    53	        note="LL2048@128 = 39.1 GC/s",
    54	    ),
    55	    dict(
    56	        key="atm_cube", title="cubed-sphere", sub="C384/C768 L60 · A100 NCCL",
    57	        series=[("float32 (C768)", [(6, 58.35), (24, 14.09)]),
    58	                ("float32 (C384)", [(6, 15.44), (24, 8.81)])],
    59	        note="f64 pending",
    60	    ),
    61	    dict(
    62	        key="atm_mpas", title="MPAS icosahedral", sub="subdiv-8/9 L26 · A100 NCCL",
    63	        series=[("float32 (subdiv-8)", [(2, 19.90), (4, 14.12), (8, 6.92),
    64	                                        (16, 7.10), (32, 8.13), (64, 5.27),
    65	                                        (128, 6.47)]),
    66	                ("float32 (subdiv-9)", [(32, 12.47), (64, 9.60), (128, 11.48)]),
    67	                ("float64 (subdiv-8)", [(2, 38.34), (4, 20.09), (8, 18.98)])],
    68	        note="s8 production mesh;\ns9 lloyd-0 synthetic",
    69	    ),
    70	    dict(
    71	        key="atm_ico_cpu", title="icosahedral", sub="subdiv-7 L26 · Milan CPU–MPI",
    72	        series=[("float32", [(1, 7399.72), (2, 3250.27), (4, 1673.07),
    73	                             (8, 827.06), (16, 444.58), (32, 230.35),
    74	                             (64, 131.58)]),
    75	                ("float64", [(1, 9987.81), (2, 4727.46), (4, 2349.06),
    76	                             (8, 1161.98), (16, 624.06), (32, 350.89),
    77	                             (64, 220.57), (128, 92.4), (256, 56.2),
    78	                             (512, 66.7)])],
    79	        note="to 512 ranks",
    80	    ),
    81	    dict(
    82	        key="oc_latlon", title="lat–lon", sub="576×1152 L20 · A100 NCCL",
    83	        series=[("float32", [(1, 36.45), (2, 22.48), (4, 12.84), (8, 11.04), (16, 8.71)]),
    84	                ("float64", [(1, 64.81), (2, 41.63), (4, 22.09)]),
    85	                ("mixed (f64 store)", [(1, 52.80), (4, 19.13)])],
    86	        note="best arm shown",
    87	    ),
    88	    dict(
    89	        key="oc_tripole", title="tripole (ORCA fold)", sub="576×1152 L20 · A100 NCCL",
    90	        series=[("float32", [(1, 35.52), (2, 25.03), (4, 15.91)]),
    91	                ("float64", [(1, 63.84), (2, 43.36), (4, 24.12)])],
    92	        note="fold +1.2–3.7 %",
    93	    ),
    94	    dict(
    95	        key="oc_mpas", title="MPAS Voronoi", sub="subdiv-7/8 · Milan CPU–MPI, 32 rpn",
    96	        series=[("float64 (subdiv-7)", [(32, 190.22), (64, 147.65),
    97	                                        (128, 102.93), (256, 65.71),
    98	                                        (512, 63.83)]),
    99	                ("float64 (subdiv-8)", [(32, 861.25), (64, 494.89),
   100	                                        (128, 309.05), (256, 254.41),
   101	                                        (512, 194.80)])],
   102	        note="32 ranks/node fixed",
   103	    ),
   104	]
   105	
   106	COLORS = {"float32": "#0072B2", "float64": "#D55E00",
   107	          "f32 · LL1536/2048 @64": "#009E73",
   108	          "float64 (packed)": "#E69F00",
   109	          "mixed (f64 store)": "#009E73",
   110	          "float32 (C768)": "#0072B2", "float32 (C384)": "#56B4E9",
   111	          "float32 (LL2048)": "#009E73",
   112	          "float32 (subdiv-8)": "#0072B2", "float32 (subdiv-9)": "#56B4E9",
   113	          "float64 (subdiv-7)": "#D55E00", "float64 (subdiv-8)": "#E69F00"}
   114	MARKERS = {"float32": "o", "float64": "s", "mixed (f64 store)": "D",
   115	           "f32 · LL1536/2048 @64": "*",
   116	           "float64 (packed)": "s",
   117	           "float32 (C768)": "o", "float32 (C384)": "^",
   118	           "float32 (LL2048)": "^",
   119	           "float32 (subdiv-8)": "o", "float32 (subdiv-9)": "^",
   120	           "float64 (subdiv-7)": "s", "float64 (subdiv-8)": "v"}
   121	
   122	
   123	def _style():
   124	    plt.rcParams.update({
   125	        "font.family": "sans-serif",
   126	        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
   127	        "font.size": 7,
   128	        "axes.labelsize": 7.5,
   129	        "axes.titlesize": 8,
   130	        "xtick.labelsize": 6.5,
   131	        "ytick.labelsize": 6.5,
   132	        "legend.fontsize": 6,
   133	        "axes.linewidth": 0.6,
   134	        "xtick.major.width": 0.6,
   135	        "ytick.major.width": 0.6,
   136	        "xtick.minor.width": 0.4,
   137	        "ytick.minor.width": 0.4,
   138	        "lines.linewidth": 1.1,
   139	        "lines.markersize": 3.4,
   140	        "figure.dpi": 300,
   141	        "savefig.dpi": 300,
   142	        "pdf.fonttype": 42,   # editable text in the PDF (journal requirement)
   143	        "ps.fonttype": 42,
   144	    })
   145	
   146	
   147	def main() -> int:
   148	    ap = argparse.ArgumentParser(description=__doc__)
   149	    ap.add_argument("--out", default="fig_scaling.pdf")
   150	    ap.add_argument("--png", default=None, help="also write a PNG here")
   151	    args = ap.parse_args()
   152	
   153	    _style()
   154	    ncol, nrow = 4, 2
   155	    # Nature double-column ~180 mm
   156	    fig, axs = plt.subplots(nrow, ncol, figsize=(180 / 25.4, 100 / 25.4))
   157	    axs = axs.ravel()
   158	
   159	    for i, spec in enumerate(PANELS):
   160	        ax = axs[i]
   161	        for label, pts in spec["series"]:
   162	            xs = [p[0] for p in pts]
   163	            ys = [p[1] for p in pts]
   164	            c = COLORS.get(label, "#444444")
   165	            ax.plot(xs, ys, MARKERS.get(label, "o") + "-", color=c,
   166	                    label=label, markerfacecolor="white",
   167	                    markeredgewidth=0.9, clip_on=False, zorder=3)
   168	            # ideal anchored at this series' own base point
   169	            n0, t0 = xs[0], ys[0]
   170	            ideal = [t0 * n0 / n for n in xs]
   171	            ax.plot(xs, ideal, "--", color=c, lw=0.7, alpha=0.55, zorder=2)
   172	
   173	        for lab, x, y in spec.get("scatter", []):
   174	            ax.plot([x], [y], "*", color="#009E73", markersize=7,
   175	                    markeredgewidth=0.8, clip_on=False, zorder=4)
   176	            ax.annotate(lab, (x, y), fontsize=5.2, color="#009E73",
   177	                        textcoords="offset points", xytext=(-4, 5), ha="right")
   178	
   179	        ax.set_xscale("log", base=2)
   180	        ax.set_yscale("log")
   181	        allx = sorted({p[0] for _, pts in spec["series"] for p in pts}
   182	                      | {x for _, x, _ in spec.get("scatter", [])})
   183	        # thin crowded tick sets to powers spanning the range
   184	        if len(allx) > 6:
   185	            allx = [x for i, x in enumerate(allx) if i % 3 == 0 or x == allx[-1]]
   186	        ax.set_xticks(allx)
   187	        ax.set_xticklabels([str(x) for x in allx])
   188	        ax.minorticks_off()
   189	        ax.set_title(spec["title"], pad=9, loc="left", fontweight="bold")
   190	        ax.text(0, 1.015, spec["sub"], transform=ax.transAxes, fontsize=5.5,
   191	                color="#555555", va="bottom")
   192	        if spec["note"]:
   193	            ax.text(0.98, 0.98, spec["note"], transform=ax.transAxes,
   194	                    fontsize=5.2, color="#888888", ha="right", va="top",
   195	                    style="italic")
   196	        ax.tick_params(direction="out", length=2.5)
   197	        for s in ("top", "right"):
   198	            ax.spines[s].set_visible(False)
   199	        ax.legend(frameon=False, loc="lower left", handlelength=1.6,
   200	                  borderpad=0.2, labelspacing=0.25)
   201	        if i % ncol == 0:
   202	            ax.set_ylabel("time per step (ms)")
   203	        if i >= ncol:
   204	            ax.set_xlabel("devices (GPUs or MPI ranks)")
   205	        # panel letter
   206	        ax.text(-0.30, 1.22, chr(ord("a") + i), transform=ax.transAxes,
   207	                fontsize=9, fontweight="bold", va="top")
   208	
   209	    # last cell: legend/provenance instead of an empty frame
   210	    ax = axs[len(PANELS)]
   211	    ax.axis("off")
   212	    handles = [
   213	        Line2D([], [], color="#0072B2", marker="o", markerfacecolor="white",
   214	               markeredgewidth=0.9, label="float32"),
   215	        Line2D([], [], color="#D55E00", marker="s", markerfacecolor="white",
   216	               markeredgewidth=0.9, label="float64"),
   217	        Line2D([], [], color="#009E73", marker="D", markerfacecolor="white",
   218	               markeredgewidth=0.9, label="mixed (f64 storage,\nf32 internals)"),
   219	        Line2D([], [], color="#009E73", marker="*", ls="none", markersize=7,
   220	               label="high-resolution point\n(not part of a ladder)"),
   221	        Line2D([], [], color="#666666", ls="--", lw=0.7,
   222	               label="ideal (anchored at\neach series' base)"),
   223	    ]
   224	    ax.legend(handles=handles, frameon=False, loc="upper left",
   225	              handlelength=1.8, labelspacing=0.55, borderpad=0)
   226	    ax.text(0, -0.06, "Levante: 4×A100-80 SXM/node (NVLink, IB HDR200);\n"
   227	                      "2×AMD Milan 7763 CPU nodes. Every point is a\n"
   228	                      "measured receipt; job ids in the source table.",
   229	            transform=ax.transAxes, fontsize=5.0, color="#777777", va="top")
   230	
   231	    # row band labels
   232	    for row, name in ((0, "ATMOSPHERE"), (1, "OCEAN")):
   233	        y = 0.945 if row == 0 else 0.475
   234	        fig.text(0.008, y, name, fontsize=7.5, fontweight="bold",
   235	                 rotation=90, va="top", ha="left", color="#222222")
   236	    fig.subplots_adjust(left=0.085, right=0.995, top=0.885, bottom=0.105,
   237	                        wspace=0.50, hspace=0.78)
   238	    fig.savefig(args.out, bbox_inches="tight")
   239	    if args.png:
   240	        fig.savefig(args.png, bbox_inches="tight")
   241	    print(args.out if not args.png else f"{args.out} {args.png}")
   242	    return 0
   243	
   244	
   245	if __name__ == "__main__":
   246	    raise SystemExit(main())

exec
/usr/bin/bash -lc "nl -ba scripts/cluster/scaling_levante/_env.sh && printf '\\n--- git diff stats/status ---\\n' && git status --short && git diff --check HEAD && git diff -- docs/performance/scaling/levante_campaign_2026-07-24.md scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch scripts/cluster/scaling_levante/atm_latlon2d_cpu_hundreds.sbatch scripts/cluster/scaling_levante/prewarm_s10.sbatch scripts/plot/plot_scaling_paper_figure.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 462ms:
     1	# Shared environment for DKRZ Levante GPU scaling jobs (sourced by the SLURM
     2	# scripts here).  SLURM/OpenMPI twin of scripts/cluster/scaling_derecho/_env.sh
     3	# (which is PBS/Cray-MPICH).
     4	# ---------------------------------------------------------------------------
     5	# EDIT the marked values (account / repo / conda env / module versions) before
     6	# the first submit.  Everything is overridable from the sbatch environment, e.g.
     7	#   sbatch --export=ALL,LEGOESM_CONDA_ENV=my-jax-env gpu_moist_scaling.slurm
     8	#
     9	# Levante GPU partition (partition `gpu`): 60 nodes, each 2x AMD EPYC 7763 +
    10	# 4x NVIDIA A100 (56 nodes 80GB, 4 nodes 40GB), InfiniBand HDR200.  MPI stack is
    11	# OpenMPI over UCX with CUDA-aware transports -- NOT Cray MPICH.
    12	# ---------------------------------------------------------------------------
    13	
    14	# --- (1) Project allocation (matches SBATCH --account in the job scripts) -----
    15	#     Levante GPU jobs bill a *_gpu sub-account (run_levante_gpu_scaling.sh uses
    16	#     bd1083_gpu, matching the SBATCH --account in the .slurm, which is the
    17	#     source of truth).  This default is only for interactive sourcing.
    18	export LEGOESM_SLURM_ACCOUNT="${LEGOESM_SLURM_ACCOUNT:-bd1083_gpu}"
    19	
    20	# --- (2) Repo location on Levante -- EDIT to where you cloned legoESM ---------
    21	REPO="${LEGOESM_REPO:-/work/bd1083/$USER/legoESM}"
    22	export REPO
    23	
    24	# --- (3) Conda env with a CUDA jaxlib AND a CUDA-aware mpi4jax (see README) ---
    25	CONDA_ENV="${LEGOESM_CONDA_ENV:-legoesm-gpu}"
    26	
    27	# --- Federation PYTHONPATH (belt-and-braces; `pip install -e .` makes it
    28	#     redundant but harmless) ------------------------------------------------
    29	PP="$REPO/src"
    30	for p in atmosphere core coupler ice land ml ocean tools; do
    31	  PP="$PP:$REPO/packages/$p"
    32	done
    33	export PYTHONPATH="$PP:${PYTHONPATH:-}"
    34	
    35	# --- Modules + conda -- EDIT the module versions to the Levante stack you built
    36	#     mpi4py / mpi4jax against (README Step 1); pinned versions matter because
    37	#     the runtime libmpi ABI must match the build ABI ------------------------
    38	module load python3 2>/dev/null || true      # EDIT: e.g. python3/2023.01-gcc-11.2.0
    39	module load openmpi 2>/dev/null || true       # EDIT: the CUDA-aware openmpi you built against
    40	module load cuda    2>/dev/null || true       # EDIT: matching cuda toolkit
    41	if command -v conda >/dev/null 2>&1; then
    42	  conda activate "$CONDA_ENV" 2>/dev/null || true
    43	fi
    44	PY="${LEGOESM_PYTHON:-$(command -v python)}"
    45	export PY
    46	
    47	# --- JAX / runtime knobs -----------------------------------------------------
    48	export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
    49	export MPI4JAX_NO_WARN_JAX_VERSION=1
    50	export MPLBACKEND="${MPLBACKEND:-Agg}"          # headless plotting
    51	# DKRZ scratch is /scratch/<first-letter-of-user>/<user>.
    52	export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/$USER}"
    53	# Persistent JIT cache reuses compiles across runs; set empty to force a cold
    54	# compile (true compile_time_s).  On SCRATCH so it survives between jobs.
    55	export LEGOESM_JIT_CACHE_DIR="${LEGOESM_JIT_CACHE_DIR:-$SCRATCH/legoesm_jit_cache}"
    56	
    57	# --- OpenMPI + UCX CUDA-aware fabric (GPU route-A) ---------------------------
    58	# Route-A hands the on-device sendrecv buffer straight to MPI (the whole point:
    59	# no device->host->device staging, which would erase multi-GPU scaling).  On
    60	# Levante that path is OpenMPI-over-UCX; the pml/osc + UCX transports below turn
    61	# on GPU-direct: cuda_copy + cuda_ipc intra-node, gdr_copy over InfiniBand HDR
    62	# inter-node.  Requires a CUDA-aware mpi4jax (README) + MPI4JAX_USE_CUDA_MPI=1
    63	# (set in the job script).  UCX_MEMTYPE_CACHE=n avoids a stale device/host
    64	# memtype-cache hang that CUDA-aware sendrecv is prone to.
    65	export OMPI_MCA_pml="${OMPI_MCA_pml:-ucx}"
    66	export OMPI_MCA_osc="${OMPI_MCA_osc:-ucx}"
    67	export UCX_TLS="${UCX_TLS:-rc,cuda_copy,cuda_ipc,gdr_copy,sm,self}"
    68	export UCX_MEMTYPE_CACHE="${UCX_MEMTYPE_CACHE:-n}"
    69	export UCX_RNDV_SCHEME="${UCX_RNDV_SCHEME:-put_zcopy}"
    70	
    71	# --- NCCL over InfiniBand (route-B: jax.distributed multi-node lanes) --------
    72	# NCCL (shard_map/ppermute collectives under jax.distributed) uses its own
    73	# IB-verbs stack — independent of the UCX/MPI settings above; the two configs
    74	# coexist. Bootstrap ring runs over IPoIB: verify the interface name once with
    75	# `ip addr` on a gpu node (a wrong NCCL_SOCKET_IFNAME is the #1 cause of
    76	# multi-node NCCL bootstrap timeouts on IB clusters).
    77	export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ib0}"
    78	export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
    79	# Prefix-match BOTH HCAs (mlx5_0/mlx5_1 — one per socket on Levante nodes).
    80	export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5}"
    81	# GPUDirect RDMA when NIC and GPU share a NUMA/PCIe root.
    82	export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-PHB}"
    83	export NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-1}"
    84	
    85	export TMPDIR="${TMPDIR:-$SCRATCH/tmp}"
    86	mkdir -p "$TMPDIR" 2>/dev/null || true

--- git diff stats/status ---
 M docs/performance/scaling/levante_campaign_2026-07-24.md
 M scripts/plot/plot_scaling_paper_figure.py
?? .physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md
?? .physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md
?? scripts/cluster/scaling_levante/atm_latlon2d_cpu_hundreds.sbatch
?? scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch
?? scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch
?? scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch
?? scripts/cluster/scaling_levante/prewarm_s10.sbatch
diff --git a/docs/performance/scaling/levante_campaign_2026-07-24.md b/docs/performance/scaling/levante_campaign_2026-07-24.md
index 24988a632..910d3555e 100644
--- a/docs/performance/scaling/levante_campaign_2026-07-24.md
+++ b/docs/performance/scaling/levante_campaign_2026-07-24.md
@@ -1583,3 +1583,131 @@ DONE during the campaign (were open at the start): C768 same-path ladder
 production tile; ocean weak at a production tile; the ico CPU-MPI ladder;
 the CPU spread ladder; and the diagnosis tool's halo + overlap phases,
 which were found broken and fixed with a contract test.
+
+## Phase-3 receipts recovered after the 2026-07-31 session drop (2026-08-02)
+
+Both jobs the dropped session left behind COMPLETED; neither had been
+analysed. First read-out below, CORRECTED per codex round-20
+(`.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md`,
+VERDICT FIX-FIRST, 12 items — the round that caught a Lloyd-mesh confound
+in the first draft's weak-scaling claim).
+
+### 1. METIS / placement A-B (ocean MPAS CPU, job 26600094)
+
+Matrix at a matched OWNED-cell target (`cells_per_rank_achieved` = 5,120
+exactly in every arm; method pinned per arm, never `auto`, which flipped
+meaning when pymetis appeared in `.venv-mpi` on 2026-07-31), f64, nlev 20,
+32 ranks/node. WET load is NOT matched under metis — per-rank wet
+cell-levels min/max: geometric@32 100,740–102,420; metis@32
+96,800–102,720; metis@128 **78,000–102,880** (one rank 24 % under the
+mean) — METIS balances owned cells, not wet cells, on this bathymetry.
+
+| arm | config | ms/step |
+|---|---|---|
+| A | s7 np32 geometric, block:cyclic | 189.82 |
+| B | s8 np128 geometric, block:cyclic | 308.96 |
+| C | s7 np32 metis, block:cyclic | 191.99 |
+| D | s8 np128 metis, block:cyclic | 333.39 |
+| E | s8 np128 metis, block:block | 537.91 |
+
+* **This METIS configuration LOSES to geometric at s8/np128** (D/B =
+  +7.9 %), despite the better offline cut (partq s8@np128: edge_cut
+  1.89 % vs 2.01 %, halo mean 592 vs 629). Scale-out term (s7@32 ->
+  s8@128, which crosses 1 -> 4 nodes as well as 4x ranks — NOT a pure
+  rank-count isolate): geometric 1.628, metis 1.736. The offline-quality
+  -> step-time inference FAILS on this lane; part of metis's loss is
+  PLAUSIBLY its own wet-load imbalance (above). Scope: closes the
+  "swap in METIS as-is" lever on the CPU-MPI ocean lane; does NOT rule
+  out partition/mapping improvements generally (e.g. wet-cell-weighted
+  METIS was NOT tested).
+* **`block:cyclic` stays mandatory on packed CPU lanes** (E/D = 1.61x at
+  a byte-identical partition). NOTE the second `--distribution` field is
+  the INTRA-NODE (socket) distribution — both arms place ranks on nodes
+  identically; the swing is socket-level. Mechanism (per-socket
+  memory-bandwidth balance) PLAUSIBLE, consistent with the np16 Milan
+  2.13x receipt; never instrumented with bandwidth counters.
+* Caveats: timing-only receipt — no parity/conservation gate ran in
+  these arms, and the CPU nodes emit `UCX WARN transports
+  'cuda_copy','cuda_ipc','gdr_copy' are not available` (the _env.sh GPU
+  UCX_TLS list on a CPU node; UCX falls back to rc/sm — cosmetic for
+  timing, but a "production config" claim would need a gated arm).
+
+### 2. subdiv-9 payoff ladder (atm MPAS ico GPU, job 26600095)
+
+f32, sfc partition (padded-128 reorder), lloyd=0 LABELLED SYNTHETIC
+scaling mesh, executed padded n_cells = 2,621,568 (natural 2,621,442),
+L26; steps 12 / warmup 3; physics=none dynamics-only bench; np32 row's
+git-SHA field absent (provenance = job script + this doc):
+
+| GPUs | cells/GPU | ms/step | GC/s (cell-levels) |
+|---|---|---|---|
+| 32 | 81.9k | 12.47 | 5.47 |
+| 64 | 41.0k | 9.60 | 7.10 |
+| 128 | 20.5k | 11.48 | 5.94 |
+
+* **np64 = 7.10 GC/s is the best MPAS-atmosphere number on this
+  synthetic dynamics-only bench** (2.19x the subdiv-8 best, 3.23 GC/s at
+  its np64: 655,362 natural cells x 26 lev / 5.27 ms).
+* Strong 32->64 speedup 1.299 (eff 0.65); 64->128 speedup 0.836 —
+  ANTI-scales at 20.5k cells/GPU. CONSISTENT WITH the ~30k floor seen on
+  the other lanes (single unreplicated point on a lane with known
+  count-specific codegen variation — not by itself proof).
+* **RETRACTED (codex round-20): the first draft's s8->s9 "weak
+  efficiency 0.55–0.74" pairs and the "~1.4x per 4x ranks GPU rank-count
+  term".** Confounds: (a) every existing s8 receipt is the generator's
+  default PRODUCTION Lloyd mesh, while s9 is lloyd=0 — different mesh
+  family, not the same protocol; (b) two comparator points came from the
+  np2-16 ladder (jobs 26454476/26454618), not the np32-128 extension
+  rows; (c) the three ratios are 1.80/1.35/1.41 — not "consistent
+  ~1.4x". A matched s8 lloyd=0 np8/16/32 rerun is submitted (see below);
+  no weak-scaling direction is claimed until it lands.
+
+### Next receipts submitted 2026-08-02
+
+1. **Ensemble receipt** (`scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch`)
+   — codex lever #1: 4 concurrent 32-GPU s9 replicas on disjoint 8-node
+   sets vs SAME-JOB solo controls bracketing phase B (solo before AND
+   after, ordering counterbalanced). steps=5000 so the stepping window
+   (~60 s) dwarfs launch skew; per-arm `SLURM_STEP_NODELIST` +
+   wall-clock brackets logged as overlap evidence. CONFIRM bar:
+   max(replica) <= 1.10x mean(solo) => guaranteed aggregate >= 3.64x the
+   32-GPU solo rate (>= 19.9 GC/s if solo reproduces 5.47) = ~3.3x the
+   observed 128-GPU single-trajectory rate. REFUTE: replica slowdown
+   >10 % = the fabric-contention term, quantified per replica.
+2. **s8 lloyd=0 matched rerun** — de-confounds the weak pair: np8/16/32
+   (81.9k/41.0k/20.5k cells/GPU) on the SAME lloyd=0 family, same sfc +
+   `--reorder-for 128`, same steps/warmup as the s9 ladder. Weak pairs
+   recomputed only from these.
+
+### 3. Recovered phase-2 receipt: lat-lon atmosphere at 128 GPUs (job 26534060, ran 2026-07-30, unanalysed until now)
+
+LL2048x4096 L26, same bench + protocol (steps 12 / warmup 3) as the
+@64 row (job 26502539, f32 6.73 ms):
+
+| arm | ms/step | GC/s (col-levels) |
+|---|---|---|
+| f32 @128 (32.7k cols/GPU) | 5.58 | **39.11** |
+| f64 @128 | 9.60 | 22.72 |
+
+f32 strong 64->128: 1.21x for 2x devices (eff 0.60) with the tile at
+32.7k cols/GPU — at/near the floor, consistent with the tile law.
+39.1 GC/s is the highest measured throughput of ANY lane in the
+campaign. The companion oc128 (26534067) FAILED pre-#1370-fix with the
+109.5 GB resident-args signature; retry submitted post-fix (below).
+
+## Hundreds-of-devices push (user directive 2026-08-02)
+
+"Push the scaling to hundreds of CPUs and GPUs for lat-lon and MPAS on
+GPUs." Machine ceiling: 56 nodes x 4 = 224 a100_80 GPUs; compute
+partition effectively unbounded for our rank counts. Submitted set:
+
+| job | what | devices | why |
+|---|---|---|---|
+| 26628021 | s9 ensemble contention | 128 GPU (4x32) | lever #1 receipt |
+| 26628071 | oc LL2304 retry post-#1370 | 128 GPU | pre-fix failure was resident-args; predicted PASS at ~0.10 GB/dev residency |
+| 26628072 | atm LL2304 @96/@192 + LL2880 @192 | 96-192 GPU | LL2048 does not divide 192; LL2880@192 = 86.4k cols/GPU ABOVE floor |
+| 26628073 | atm lat-lon 2-D pencil r512 np64-512 | 512 CPU ranks | hundreds-of-CPUs lat-lon (wall-pole lane, labelled) |
+| 26628074 | subdiv-10 lloyd0 prewarm | 1 CPU | unlocks MPAS 128-224 GPUs ABOVE floor (81.9k-46.8k cells/GPU) |
+| 26628076 | s8 lloyd0 np8/16/32 | 32 GPU | weak-pair de-confound (codex r20 item 5) |
+
+s10 ladder (128/192/224 GPUs) submits once 26628074's cache lands.
diff --git a/scripts/plot/plot_scaling_paper_figure.py b/scripts/plot/plot_scaling_paper_figure.py
index 93056ac19..5bea5cef8 100644
--- a/scripts/plot/plot_scaling_paper_figure.py
+++ b/scripts/plot/plot_scaling_paper_figure.py
@@ -31,9 +31,11 @@ from matplotlib.lines import Line2D
 # --- Measured data -------------------------------------------------------
 # (devices, ms/step). Job ids are the provenance for each series.
 SOURCES = {
-    "atm_latlon": "26450848/26453240/26449147 (f32), 26494902 (f64)",
+    "atm_latlon": "26450848/26453240/26449147 (f32), 26494902 (f64), "
+                  "LL2048@64 26502539, LL2048@128 26534060",
     "atm_cube": "26452894/26453782",
-    "atm_mpas": "26454476/26454618/26486288/26493638/26493734",
+    "atm_mpas": "26454476/26454618/26486288/26493638/26493734, "
+                "s8 np32-128 26549646/26538474, s9 26600095",
     "atm_ico_cpu": "26495083 (f32), 26495437 (f64) — both block:cyclic",
     "oc_latlon": "26460444-501/26460365/26493592",
     "oc_tripole": "26493837/26493648",
@@ -45,9 +47,10 @@ PANELS = [
         key="atm_latlon", title="lat–lon", sub="720×1440 L26 · A100 NCCL",
         series=[("float32", [(4, 7.72), (8, 5.40), (16, 3.54)]),
                 ("float64", [(4, 16.11), (8, 11.34), (16, 5.62)]),
+                ("float32 (LL2048)", [(64, 6.73), (128, 5.58)]),
                 ],
-        scatter=[("LL1536 @64", 64, 4.97), ("LL2048 @64", 64, 6.73)],
-        note="+ 64-GPU high-res points",
+        scatter=[("LL1536 @64", 64, 4.97), ("LL2048 f64 @128", 128, 9.60)],
+        note="LL2048@128 = 39.1 GC/s",
     ),
     dict(
         key="atm_cube", title="cubed-sphere", sub="C384/C768 L60 · A100 NCCL",
@@ -56,10 +59,13 @@ PANELS = [
         note="f64 pending",
     ),
     dict(
-        key="atm_mpas", title="MPAS icosahedral", sub="L8 28 km L26 · A100 NCCL",
-        series=[("float32", [(2, 19.90), (4, 14.12), (8, 6.92), (16, 7.10)]),
-                ("float64", [(2, 38.34), (4, 20.09), (8, 18.98)])],
-        note="incl. fusion fix",
+        key="atm_mpas", title="MPAS icosahedral", sub="subdiv-8/9 L26 · A100 NCCL",
+        series=[("float32 (subdiv-8)", [(2, 19.90), (4, 14.12), (8, 6.92),
+                                        (16, 7.10), (32, 8.13), (64, 5.27),
+                                        (128, 6.47)]),
+                ("float32 (subdiv-9)", [(32, 12.47), (64, 9.60), (128, 11.48)]),
+                ("float64 (subdiv-8)", [(2, 38.34), (4, 20.09), (8, 18.98)])],
+        note="s8 production mesh;\ns9 lloyd-0 synthetic",
     ),
     dict(
         key="atm_ico_cpu", title="icosahedral", sub="subdiv-7 L26 · Milan CPU–MPI",
@@ -102,11 +108,15 @@ COLORS = {"float32": "#0072B2", "float64": "#D55E00",
           "float64 (packed)": "#E69F00",
           "mixed (f64 store)": "#009E73",
           "float32 (C768)": "#0072B2", "float32 (C384)": "#56B4E9",
+          "float32 (LL2048)": "#009E73",
+          "float32 (subdiv-8)": "#0072B2", "float32 (subdiv-9)": "#56B4E9",
           "float64 (subdiv-7)": "#D55E00", "float64 (subdiv-8)": "#E69F00"}
 MARKERS = {"float32": "o", "float64": "s", "mixed (f64 store)": "D",
            "f32 · LL1536/2048 @64": "*",
            "float64 (packed)": "s",
            "float32 (C768)": "o", "float32 (C384)": "^",
+           "float32 (LL2048)": "^",
+           "float32 (subdiv-8)": "o", "float32 (subdiv-9)": "^",
            "float64 (subdiv-7)": "s", "float64 (subdiv-8)": "v"}
 
 

codex
I’ve confirmed the prior review is explicitly tracked in the new section. I’m now checking whether the scripts actually enforce the stated protocol—not just describe it—especially receipt validation and Slurm step allocation.
exec
/usr/bin/bash -lc "rg -n -C 3 -- '--n-lat|n_lat|n-lon|n_lon|latlon-2d|multicontroller|reorder-for|n-devices|SLURM_STEP_NODELIST|coordinator|steady_median_ms|mcells_per_s|cells_per_rank_achieved|n_cells' scripts/bench packages scripts/cluster/scaling_levante | head -n 1000" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 395ms:
scripts/cluster/scaling_levante/README.md-71-
scripts/cluster/scaling_levante/README.md-72-| File | What |
scripts/cluster/scaling_levante/README.md-73-|---|---|
scripts/cluster/scaling_levante/README.md:74:| `gpu_scaling.sbatch` | ATM (cube AMIP physics, single-process multi-GPU) + OCEAN (`bench_ocean_latlon_spmd_scaling.py`, 1/2/4 A100 with parity+conservation smoke) on one GPU node. Plain GPU env — no mpi4jax. |
scripts/cluster/scaling_levante/README.md:75:| `gpu_multinode_scaling.sbatch` | MULTI-NODE route-B lanes over NCCL/IB (SLURM auto-detected `jax.distributed`): A = cube `--cs-spmd` (6 GPU / 2 nodes), C = atm lat-lon `--multicontroller` (8 GPU), D = ocean `--multicontroller` (8 GPU). |
scripts/cluster/scaling_levante/README.md-76-| `cpu_scaling.sbatch` | ATM + OCEAN CPU-MPI rank ladders on `compute` nodes (`legoesm-mpi` env), with fail-fast smokes. |
scripts/cluster/scaling_levante/README.md-77-| `diagnosis.sbatch` | BOTTLENECK diagnosis via `scripts/bench/run_scaling_diagnosis.py` — per-phase halo bandwidth, reduction latency, roofline, compute/comm overlap, and the **static collective census** (collective-permute + all-reduce + all-gather per step) the throughput jobs do NOT capture. Climbs the cube face-shard `1 2 3` ladder so the message-count-vs-shard curve — the LATENCY-bound anti-scaling signal — is recorded. `sbatch --export=ALL,MODE=census` for counts only. Derecho twin: `scaling_derecho/diagnosis.pbs`. |
scripts/cluster/scaling_levante/README.md-78-
--
scripts/cluster/scaling_levante/README.md-87-SLURM directives cannot expand env vars.
scripts/cluster/scaling_levante/README.md-88-
scripts/cluster/scaling_levante/README.md-89-
scripts/cluster/scaling_levante/README.md:90:## 2026-07 lane E: icosahedral/MPAS multicontroller
scripts/cluster/scaling_levante/README.md-91-
scripts/cluster/scaling_levante/README.md-92-`gpu_multinode_scaling.sbatch` gained lane E (`RUN_MPAS=1`, default on):
scripts/cluster/scaling_levante/README.md-93-icosahedral MPAS PE over `jax.distributed` + NCCL via
--
scripts/cluster/scaling_levante/README.md-95-ppermute halos), 6 tasks = 2 nodes x 3 GPUs (`nCells = 10*4^L + 2` splits
scripts/cluster/scaling_levante/README.md-96-evenly for 1/2/3/6). Subdiv-4 parity+conservation smoke gates the timed
scripts/cluster/scaling_levante/README.md-97-`ICO_LEVEL` (default L7) case. Federation gate:
scripts/cluster/scaling_levante/README.md:98:`tests/parallel/test_mpas_spmd_multicontroller_selfspawn.py`.
scripts/cluster/scaling_levante/README.md-99-
scripts/cluster/scaling_levante/README.md-100-
scripts/cluster/scaling_levante/README.md-101-## 2026-07 lane T: comm-tuning A/B ladder (`RUN_TUNE=1`)
--
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-13-# receipt is the generator's default PRODUCTION-Lloyd mesh, while the
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-14-# subdiv-9 ladder (job 26600095) is the lloyd=0 synthetic family — so no
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-15-# s8-vs-s9 weak-scaling pair was protocol-clean.  This ladder reruns s8
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:16:# np8/16/32 on the SAME lloyd=0 family, same sfc + --reorder-for 128,
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-17-# same steps/warmup as the s9 ladder.  Weak pairs at matched cells/GPU
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-18-# (81.9k / 41.0k / 20.5k) are computed ONLY from these rows vs 26600095.
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-19-#
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-20-# Falsifiability, written BEFORE submit:
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:21:#   numbers : s8-lloyd0 np8/16/32 steady_median_ms
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-22-#   CONFIRM (a matched-tile scale-out term exists): s9/s8 ratios at
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-23-#             matched cells/GPU stay well above 1 (prior draft saw
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-24-#             1.80/1.35/1.41 on the CONFOUNDED pairs)
--
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-45-  JAX_ENABLE_X64=0 srun --nodes="$NODES" --ntasks="$NP" --ntasks-per-node=4 \
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-46-      --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 \
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-47-    "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:48:      --multicontroller --n-devices "$NP" \
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-49-      --subdivision 8 --nlev 26 --steps 12 --warmup 3 --lloyd 0 \
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:50:      --partition-method sfc --reorder-for 128 \
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-51-      --out "$OUTDIR/np${NP}.jsonl" || { echo "np$NP FAILED"; rc=1; }
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-52-done
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-53-echo "=== RESULTS (cells/GPU: 81.9k / 41.0k / 20.5k) ==="
--
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-57-import json,sys
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-58-try:
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-59-    d=json.loads(open('$F').readline())
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:60:    print(f'np$NP: {d[\"steady_median_ms\"]:8.2f} ms')
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-61-except Exception as e:
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-62-    print('np$NP: MISSING/UNPARSEABLE ->', e); sys.exit(1)" \
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-63-    || { echo "np$NP receipt invalid"; rc=1; }
--
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch-35-  JAX_ENABLE_X64=0 srun --ntasks="$3" --ntasks-per-node=4 \
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch-36-      --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 \
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch-37-    "$PY" scripts/bench/bench_atm_latlon_spmd_scaling.py \
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch:38:      --multicontroller --n-devices "$3" --mode strong \
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch:39:      --n-lat "$1" --n-lon "$2" --nlev 26 \
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch-40-      --steps 12 --warmup 3 \
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch-41-      --out "$OUTDIR/$4.jsonl" || { echo "$4 FAILED"; rc=1; }
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch-42-}
--
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch-49-import json,os
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch-50-try:
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch-51-    d=json.loads(open('$F').readline())
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch:52:    print(f\"{os.path.basename('$F'):24s} {d['steady_median_ms']:8.2f} ms {d.get('mcells_per_s',0)/1000:6.2f} GC/s\")
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch-53-except Exception as e:
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch-54-    print(os.path.basename('$F'), 'UNPARSEABLE', e)"
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch-55-done
--
scripts/cluster/scaling_levante/gpu_scaling.sbatch-90-
scripts/cluster/scaling_levante/gpu_scaling.sbatch-91-if [ "$RUN_OCEAN" = "1" ]; then
scripts/cluster/scaling_levante/gpu_scaling.sbatch-92-    echo "=== OCEAN SMOKE: parity + conservation gates (LL32/L6, nd=2) ==="
scripts/cluster/scaling_levante/gpu_scaling.sbatch:93:    "$PY" scripts/bench/bench_ocean_latlon_spmd_scaling.py \
scripts/cluster/scaling_levante/gpu_scaling.sbatch:94:        --n-devices 2 \
scripts/cluster/scaling_levante/gpu_scaling.sbatch:95:        --n-lat 32 --n-lon 64 --nlev 6 \
scripts/cluster/scaling_levante/gpu_scaling.sbatch-96-        --steps 4 --warmup 1 \
scripts/cluster/scaling_levante/gpu_scaling.sbatch-97-        --parity-gate --check-conservation --cons-rtol 1e-6 \
scripts/cluster/scaling_levante/gpu_scaling.sbatch-98-        --out "$OUTDIR/ocean_smoke/ocean_spmd_scaling.jsonl"
--
scripts/cluster/scaling_levante/gpu_scaling.sbatch-118-                    echo "--- ocean spmd: mode=$mode prec=$prec nd=$nd res=$res ---"
scripts/cluster/scaling_levante/gpu_scaling.sbatch-119-                    if [ "$prec" = "float64" ]; then _X64=1; else _X64=0; fi
scripts/cluster/scaling_levante/gpu_scaling.sbatch-120-                    if [ "$mode" = "strong" ]; then
scripts/cluster/scaling_levante/gpu_scaling.sbatch:121:                        _GEOM="--mode strong --n-lat $res --n-lon $((2 * res))"
scripts/cluster/scaling_levante/gpu_scaling.sbatch-122-                    else
scripts/cluster/scaling_levante/gpu_scaling.sbatch:123:                        _GEOM="--mode weak --nlat-per-dev $res --n-lon $((2 * res))"
scripts/cluster/scaling_levante/gpu_scaling.sbatch-124-                    fi
scripts/cluster/scaling_levante/gpu_scaling.sbatch-125-                    # --force-pcg: solver-matched ladder (nd=1 otherwise
scripts/cluster/scaling_levante/gpu_scaling.sbatch-126-                    # times adaptive stock CG vs fixed-M PCG at nd>1).
--
scripts/cluster/scaling_levante/gpu_scaling.sbatch-128-                    # ladder row must carry its own gate receipt, not
scripts/cluster/scaling_levante/gpu_scaling.sbatch-129-                    # inherit the smoke's (codex round-3 finding 6).
scripts/cluster/scaling_levante/gpu_scaling.sbatch-130-                    JAX_ENABLE_X64=$_X64 "$PY" \
scripts/cluster/scaling_levante/gpu_scaling.sbatch:131:                        scripts/bench/bench_ocean_latlon_spmd_scaling.py \
scripts/cluster/scaling_levante/gpu_scaling.sbatch:132:                        --n-devices "$nd" $_GEOM --nlev "$OC_NLEV" \
scripts/cluster/scaling_levante/gpu_scaling.sbatch-133-                        --force-pcg \
scripts/cluster/scaling_levante/gpu_scaling.sbatch-134-                        --check-conservation --cons-rtol "${OC_CONS_RTOL:-1e-5}" \
scripts/cluster/scaling_levante/gpu_scaling.sbatch-135-                        --steps $((N_WARMUP + N_TIMING)) --warmup "$N_WARMUP" \
--
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-18-#
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-19-#   solo_pre  : ONE 32-GPU s9 run, other 24 nodes idle
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-20-#   phase B   : FOUR concurrent 32-GPU s9 runs on disjoint 8-node sets
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch:21:#               (SLURM_STEP_NODELIST is per-step -> per-replica
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch:22:#               coordinator autodetect; jobid-derived port shared but
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-23-#               hosts differ)
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-24-#   solo_post : solo again AFTER phase B (brackets ordering/thermal
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-25-#               drift; contrast uses mean of the two solos)
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-26-#
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-27-# Falsifiability, written BEFORE submit:
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch:28:#   numbers : 2 solo + 4 replica steady_median_ms
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-29-#   CONFIRM : max(replica) <= 1.10 x mean(solo) -> guaranteed aggregate
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-30-#             >= 4/1.10 = 3.64x the 32-GPU solo rate (~19.9 GC/s if solo
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-31-#             reproduces 5.47) = ~3.3x the observed 128-GPU
--
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-58-      --gpus-per-node=4 --gpu-bind=none --exact --kill-on-bad-exit=1 \
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-59-      --job-name="arm_$1" \
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-60-    "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch:61:      --multicontroller --n-devices 32 \
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-62-      --subdivision 9 --nlev 26 --steps 5000 --warmup 100 --lloyd 0 \
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch:63:      --partition-method sfc --reorder-for 128 \
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-64-      --out "$OUTDIR/$1.jsonl"
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-65-  s=$?
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-66-  echo "[$1] exit=$s epoch=$(date +%s.%N)"
--
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-87-import json,sys
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-88-try:
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-89-    d=json.loads(open('$F').readline())
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch:90:    ms=d['steady_median_ms']
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-91-except Exception as e:
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-92-    print('$T: MISSING/UNPARSEABLE ->', e); sys.exit(1)
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch:93:print(f'$T: {ms:8.2f} ms  {d.get(\"mcells_per_s\",0)/1000:.2f} GC/s')" \
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-94-    || { echo "$T receipt invalid"; rc=1; }
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-95-done
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-96-
--
scripts/cluster/scaling_levante/atm_latlon2d_cpu_hundreds.sbatch-27-  JAX_ENABLE_X64=1 srun --nodes="$NODES" --ntasks="$NP" --ntasks-per-node=32 \
scripts/cluster/scaling_levante/atm_latlon2d_cpu_hundreds.sbatch-28-      --distribution=block:cyclic --cpu-bind=cores --kill-on-bad-exit=1 \
scripts/cluster/scaling_levante/atm_latlon2d_cpu_hundreds.sbatch-29-    "$PY" scripts/bench/run_cpu_mpi_scaling.py \
scripts/cluster/scaling_levante/atm_latlon2d_cpu_hundreds.sbatch:30:      --grid latlon --latlon-2d --mode single --physics moist \
scripts/cluster/scaling_levante/atm_latlon2d_cpu_hundreds.sbatch-31-      --precision float64 --n-levels 26 --resolution 512 \
scripts/cluster/scaling_levante/atm_latlon2d_cpu_hundreds.sbatch-32-      --n-warmup 2 --n-timing 6 \
scripts/cluster/scaling_levante/atm_latlon2d_cpu_hundreds.sbatch-33-      --output-dir "$OUTDIR/np${NP}" < /dev/null || { echo "np$NP FAILED"; rc=1; }
--
scripts/cluster/scaling_levante/cube_tiled_step.sbatch-10-# ===========================================================================
scripts/cluster/scaling_levante/cube_tiled_step.sbatch-11-# Sub-face TILED cube hydrostatic step on Levante: 24 GPUs (6 x 4 A100),
scripts/cluster/scaling_levante/cube_tiled_step.sbatch-12-# kt=2 -> (6,2,2) mesh — the >6-GPU strong-scaling lane (audit item 5).
scripts/cluster/scaling_levante/cube_tiled_step.sbatch:13:# Stage A: CPU-virtual parity gate; Stage B: multicontroller timed run.
scripts/cluster/scaling_levante/cube_tiled_step.sbatch-14-# GPU binding: all 4 node GPUs visible per task (--gpu-bind=none); bare
scripts/cluster/scaling_levante/cube_tiled_step.sbatch-15-# jax.distributed.initialize() under SLURM derives
scripts/cluster/scaling_levante/cube_tiled_step.sbatch-16-# local_device_ids=[SLURM_LOCALID], so each task binds the LOCALID-th GPU.
--
scripts/cluster/scaling_levante/cube_tiled_step.sbatch-68-    --kt "${KT}" --resolution 48 --nlev 10 --steps 4 --warmup 1 \
scripts/cluster/scaling_levante/cube_tiled_step.sbatch-69-    --closed-loop --parity-gate --out "${OUT%.jsonl}_loop_parity.jsonl"
scripts/cluster/scaling_levante/cube_tiled_step.sbatch-70-
scripts/cluster/scaling_levante/cube_tiled_step.sbatch:71:echo "== Stage B: 24-GPU multicontroller timed runs =="
scripts/cluster/scaling_levante/cube_tiled_step.sbatch-72-export JAX_PLATFORMS=cuda
scripts/cluster/scaling_levante/cube_tiled_step.sbatch-73-# #921 (same defect as the Derecho twin): the closed-loop lane adds the
scripts/cluster/scaling_levante/cube_tiled_step.sbatch-74-# dry-mass fixer's GLOBAL all-reduce (_tile_fix_ps_mass_delta psum over
--
scripts/cluster/scaling_levante/cube_tiled_step.sbatch-79-export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_nccl_comm_splitting=false --xla_gpu_enable_latency_hiding_scheduler=false"
scripts/cluster/scaling_levante/cube_tiled_step.sbatch-80-timeout 1800 srun "$PY" scripts/bench/bench_cube_tiled_step_scaling.py \
scripts/cluster/scaling_levante/cube_tiled_step.sbatch-81-    --kt "${KT}" --resolution "${RES}" --nlev "${NLEV}" --dt "${DT}" \
scripts/cluster/scaling_levante/cube_tiled_step.sbatch:82:    --steps 12 --warmup 2 --multicontroller --out "${OUT}"
scripts/cluster/scaling_levante/cube_tiled_step.sbatch-83-echo "== single-shot done"
scripts/cluster/scaling_levante/cube_tiled_step.sbatch-84-# Closed-loop production lane (state resident tile-sharded across steps).
scripts/cluster/scaling_levante/cube_tiled_step.sbatch-85-echo "== closed-loop start"
scripts/cluster/scaling_levante/cube_tiled_step.sbatch-86-timeout 1800 srun "$PY" scripts/bench/bench_cube_tiled_step_scaling.py \
scripts/cluster/scaling_levante/cube_tiled_step.sbatch-87-    --kt "${KT}" --resolution "${RES}" --nlev "${NLEV}" --dt "${DT}" \
scripts/cluster/scaling_levante/cube_tiled_step.sbatch:88:    --steps 12 --warmup 2 --multicontroller --closed-loop --out "${OUT}"
scripts/cluster/scaling_levante/cube_tiled_step.sbatch-89-
scripts/cluster/scaling_levante/cube_tiled_step.sbatch-90-echo "done: ${OUT}"
--
scripts/cluster/scaling_levante/gpu_moist_scaling.slurm-27-# "generalize route-A to multi-node" + "Levante equivalent").
scripts/cluster/scaling_levante/gpu_moist_scaling.slurm-28-#
scripts/cluster/scaling_levante/gpu_moist_scaling.slurm-29-# NOT for cubed-sphere (served by run_levante_gpu_scaling.py single-process
scripts/cluster/scaling_levante/gpu_moist_scaling.slurm:30:# sharding; Kessler-moist cube needs the --cs-spmd coordinator path) or spectral
scripts/cluster/scaling_levante/gpu_moist_scaling.slurm-31-# (single device, no MPI).
scripts/cluster/scaling_levante/gpu_moist_scaling.slurm-32-#
scripts/cluster/scaling_levante/gpu_moist_scaling.slurm-33-# PREREQS (read scripts/cluster/scaling_levante/README.md):
--
packages/ml/legoesm/ml/sfno.py-120-
packages/ml/legoesm/ml/sfno.py-121-        Parameters
packages/ml/legoesm/ml/sfno.py-122-        ----------
packages/ml/legoesm/ml/sfno.py:123:        x : array, shape (n_lat, n_lon, in_channels)
packages/ml/legoesm/ml/sfno.py-124-            Input field on the Gaussian grid.
packages/ml/legoesm/ml/sfno.py-125-        grid : GaussianGrid
packages/ml/legoesm/ml/sfno.py-126-            Grid for SH transforms.
packages/ml/legoesm/ml/sfno.py-127-
packages/ml/legoesm/ml/sfno.py-128-        Returns
packages/ml/legoesm/ml/sfno.py-129-        -------
packages/ml/legoesm/ml/sfno.py:130:        array, shape (n_lat, n_lon, out_channels)
packages/ml/legoesm/ml/sfno.py-131-            Predicted output field.
packages/ml/legoesm/ml/sfno.py-132-        """
packages/ml/legoesm/ml/sfno.py-133-        # Save input for big residual skip
packages/ml/legoesm/ml/sfno.py-134-        x_in = x
packages/ml/legoesm/ml/sfno.py-135-
packages/ml/legoesm/ml/sfno.py:136:        # Encoder: (n_lat, n_lon, in_channels) → (n_lat, n_lon, embed_dim)
packages/ml/legoesm/ml/sfno.py-137-        x = jax.vmap(jax.vmap(self.encoder))(x)
packages/ml/legoesm/ml/sfno.py-138-
packages/ml/legoesm/ml/sfno.py-139-        # Processor: N SFNO blocks.  When ``gradient_checkpoint`` is
--
packages/ml/legoesm/ml/sfno.py-151-            else:
packages/ml/legoesm/ml/sfno.py-152-                x = block(x, grid)
packages/ml/legoesm/ml/sfno.py-153-
packages/ml/legoesm/ml/sfno.py:154:        # Decoder: (n_lat, n_lon, embed_dim) → (n_lat, n_lon, out_channels)
packages/ml/legoesm/ml/sfno.py-155-        x = jax.vmap(jax.vmap(self.decoder))(x)
packages/ml/legoesm/ml/sfno.py-156-
packages/ml/legoesm/ml/sfno.py-157-        # Big residual skip (ACE-style)
--
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-20-#                         gap-#0 lane (native ppermute halos replace the
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-21-#                         route-A mpi4jax >=2-node ceiling)
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-22-#   Lane D (RUN_OCEAN=1)  ocean lat-band multihost, 8 procs
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:23:#   Lane E (RUN_MPAS=1)   icosahedral MPAS multicontroller, 6 procs
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-24-#   Lane T (RUN_TUNE=1, default 0) comm-tuning A/B ladder (fused-halo /
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-25-#                         XLA CP-combining + pipelined p2p / PGLE); outputs
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-26-#                         under _ab_tuning/, excluded from the curves
--
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-61-RUN_MPAS="${RUN_MPAS:-1}"
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-62-# Lat-band lanes C/D task count (8 = 2 nodes; 16 = sbatch --nodes=4).
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-63-NP_LL="${NP_LL:-8}"
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:64:# Lane E device count; >6 pads the mesh via --reorder-for (even split).
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-65-NP_MPAS="${NP_MPAS:-6}"
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-66-# Slurm job STEPS do not inherit the job's GPU allocation on all Slurm
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-67-# versions/configs — without an explicit step gres some tasks see zero
--
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-157-fi
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-158-
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-159-# Gap-#0 lane: native ppermute band halos over NCCL replace the route-A
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:160:# mpi4jax >=2-node ceiling. Gate: tests/parallel/test_atm_latlon_spmd_multicontroller_selfspawn.py.
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-161-if [ "$RUN_LATLON" = "1" ]; then
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-162-    echo "=== LANE C: atm latlon multihost np=$NP_LL (LL$LL_RES) ==="
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-163-    # shellcheck disable=SC2086
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-164-    srun --ntasks="$NP_LL" --ntasks-per-node=4 $STEP_GPU_OPTS --kill-on-bad-exit=1 bash -c "$PIN" _ \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-165-        "$PY" scripts/bench/bench_atm_latlon_spmd_scaling.py \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:166:        --multicontroller --n-devices "$NP_LL" \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:167:        --mode "$MODE" --n-lat "$LL_RES" --n-lon $((2 * LL_RES)) \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-168-        --nlev "$NLEV" \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-169-        --steps $((N_WARMUP + N_TIMING)) --warmup "$N_WARMUP" \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-170-        --out "$OUTDIR/latlon_multihost/spmd_scaling.jsonl"
--
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-172-    [ "$rc" -ne 0 ] && { echo "LANE C FAILED rc=$rc"; rc_all=$rc; }
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-173-fi
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-174-
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:175:# Gate: tests/parallel/test_latlon_ocean_spmd_multicontroller_selfspawn.py.
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-176-if [ "$RUN_OCEAN" = "1" ]; then
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-177-    echo "=== LANE D: ocean latlon multihost np=$NP_LL (LL$OC_RES) $OC_EXTRA ==="
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-178-    # OC_EXTRA: extra bench flags for improved-config ladders (e.g.
--
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-180-    # implicit_cn. Env like LEGOESM_VMIX_F32_SOLVE propagates via --export.
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-181-    # shellcheck disable=SC2086
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-182-    srun --ntasks="$NP_LL" --ntasks-per-node=4 $STEP_GPU_OPTS --kill-on-bad-exit=1 bash -c "$PIN" _ \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:183:        "$PY" scripts/bench/bench_ocean_latlon_spmd_scaling.py \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:184:        --multicontroller --n-devices "$NP_LL" \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:185:        --mode "$MODE" --n-lat "$OC_RES" --n-lon $((2 * OC_RES)) \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-186-        --nlev "$OC_NLEV" ${OC_EXTRA:-} \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-187-        --steps $((N_WARMUP + N_TIMING)) --warmup "$N_WARMUP" \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:188:        --out "$OUTDIR/ocean_multicontroller/ocean_spmd_scaling.jsonl"
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-189-    rc=$?
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-190-    [ "$rc" -ne 0 ] && { echo "LANE D FAILED rc=$rc"; rc_all=$rc; }
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-191-fi
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-192-
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:193:# Lane E: icosahedral/MPAS multicontroller (6 procs = 2 nodes x 3 GPUs) —
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-194-# cell-partition reorder + ppermute halo (make_voronoi_sharded_step) over
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-195-# jax.distributed. np=6: nCells = 10*4^L+2 admits 1/2/3/6 even splits at
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-196-# every level. Smoke gate (parity+conservation) first, then the timed case.
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:197:# Gate: tests/parallel/test_mpas_spmd_multicontroller_selfspawn.py.
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-198-if [ "$RUN_MPAS" = "1" ]; then
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:199:    echo "=== LANE E: icosahedral MPAS multicontroller np=$NP_MPAS (L$ICO_LEVEL) ==="
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-200-    # >6 devices: 10*4^L+2 = 2*odd has no even split beyond 6 — pad the mesh
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:201:    # to the target count via --reorder-for (identical padded mesh per ladder).
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-202-    E_TPN=3; [ $((NP_MPAS % 4)) -eq 0 ] && E_TPN=4
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-203-    E_REORDER=""
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:204:    [ "$NP_MPAS" -gt 6 ] && E_REORDER="--reorder-for $NP_MPAS"
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-205-    # shellcheck disable=SC2086
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-206-    srun --ntasks="$NP_MPAS" --ntasks-per-node="$E_TPN" $STEP_GPU_OPTS --kill-on-bad-exit=1 bash -c "$PIN" _ \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-207-        "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:208:        --multicontroller --n-devices "$NP_MPAS" \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-209-        --subdivision 4 --nlev 8 --steps 4 --warmup 1 \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-210-        --partition-method sfc $E_REORDER \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-211-        --parity-gate --check-conservation \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:212:        --out "$OUTDIR/mpas_multicontroller/smoke.jsonl" \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-213-    && srun --ntasks="$NP_MPAS" --ntasks-per-node="$E_TPN" $STEP_GPU_OPTS --kill-on-bad-exit=1 bash -c "$PIN" _ \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-214-        "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:215:        --multicontroller --n-devices "$NP_MPAS" \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-216-        --subdivision "$ICO_LEVEL" --nlev "$NLEV" \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-217-        --steps $((N_WARMUP + N_TIMING)) --warmup "$N_WARMUP" \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-218-        --partition-method sfc $E_REORDER \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:219:        --out "$OUTDIR/mpas_multicontroller/mpas_spmd_scaling.jsonl"
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-220-    rc=$?
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-221-    [ "$rc" -ne 0 ] && { echo "LANE E FAILED rc=$rc"; rc_all=$rc; }
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-222-fi
--
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-229-# pipelined p2p), pgle (profile-guided latency estimation; recompiles after
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-230-# the profiling runs — bench-jit-safe, not for AOT jobs). Outputs land under
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-231-# _ab_tuning/ which aggregate_bcw_scaling SKIPS (A/B receipts never join the
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:232:# scaling curves); compare arms via steady_median_ms / sypd per JSONL row.
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-233-RUN_TUNE="${RUN_TUNE:-0}"
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-234-if [ "$RUN_TUNE" = "1" ]; then
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-235-    echo "=== LANE T: comm-tuning A/B (latlon atm np=8; ocean np=8; cube np=6) ==="
--
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-244-            esac
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-245-            srun --ntasks=8 --ntasks-per-node=4 $STEP_GPU_OPTS --kill-on-bad-exit=1 bash -c "$PIN" _ \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-246-                "$PY" scripts/bench/bench_atm_latlon_spmd_scaling.py \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:247:                --multicontroller --n-devices 8 \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:248:                --mode strong --n-lat "$LL_RES" --n-lon $((2 * LL_RES)) \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-249-                --nlev "$NLEV" \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-250-                --steps $((N_WARMUP + N_TIMING)) --warmup "$N_WARMUP" \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-251-                --out "$OUTDIR/_ab_tuning/latlon_${ARM}.jsonl"
--
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-258-        (
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-259-            [ "$ARM" = fused ] && export LEGOESM_LATLON_SPMD_FUSED_HALO=1
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-260-            srun --ntasks=8 --ntasks-per-node=4 $STEP_GPU_OPTS --kill-on-bad-exit=1 bash -c "$PIN" _ \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:261:                "$PY" scripts/bench/bench_ocean_latlon_spmd_scaling.py \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:262:                --multicontroller --n-devices 8 \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:263:                --mode strong --n-lat "$OC_RES" --n-lon $((2 * OC_RES)) \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-264-                --nlev "$OC_NLEV" \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-265-                --steps $((N_WARMUP + N_TIMING)) --warmup "$N_WARMUP" \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-266-                --out "$OUTDIR/_ab_tuning/ocean_${ARM}.jsonl"
--
scripts/bench/bench_barotropic_mcut.py-15-
scripts/bench/bench_barotropic_mcut.py-16-Run::
scripts/bench/bench_barotropic_mcut.py-17-
scripts/bench/bench_barotropic_mcut.py:18:    mpirun -np 2 python scripts/bench/bench_barotropic_mcut.py --n-lat 96 --n-lon 192
scripts/bench/bench_barotropic_mcut.py-19-
scripts/bench/bench_barotropic_mcut.py-20-Writes ``docs/performance/scaling/barotropic_mcut.csv`` on rank 0 for the plotter.
scripts/bench/bench_barotropic_mcut.py-21-"""
--
scripts/bench/bench_barotropic_mcut.py-45-from legoesm.ocean.dynamics.barotropic_common import solve_helmholtz_implicit
scripts/bench/bench_barotropic_mcut.py-46-
scripts/bench/bench_barotropic_mcut.py-47-
scripts/bench/bench_barotropic_mcut.py:48:def _coastal_mask(n_lat, n_lon):
scripts/bench/bench_barotropic_mcut.py:49:    m = np.ones((n_lat, n_lon))
scripts/bench/bench_barotropic_mcut.py-50-    m[0, :] = 0.0
scripts/bench/bench_barotropic_mcut.py-51-    m[-1, :] = 0.0
scripts/bench/bench_barotropic_mcut.py:52:    m[:, n_lon // 8: n_lon // 8 + 4] = 0.0           # meridional coast
scripts/bench/bench_barotropic_mcut.py:53:    m[2 * n_lat // 5: 2 * n_lat // 5 + 6,
scripts/bench/bench_barotropic_mcut.py:54:      4 * n_lon // 9: 4 * n_lon // 9 + 15] = 0.0      # interior basin
scripts/bench/bench_barotropic_mcut.py-55-    return m
scripts/bench/bench_barotropic_mcut.py-56-
scripts/bench/bench_barotropic_mcut.py-57-
scripts/bench/bench_barotropic_mcut.py-58-def main():
scripts/bench/bench_barotropic_mcut.py-59-    p = argparse.ArgumentParser()
scripts/bench/bench_barotropic_mcut.py:60:    p.add_argument("--n-lat", type=int, default=96)
scripts/bench/bench_barotropic_mcut.py:61:    p.add_argument("--n-lon", type=int, default=192)
scripts/bench/bench_barotropic_mcut.py-62-    p.add_argument("--coeff", type=float, default=5.0e7)
scripts/bench/bench_barotropic_mcut.py-63-    p.add_argument("--m-sweep", type=int, nargs="+",
scripts/bench/bench_barotropic_mcut.py-64-                   default=[2, 4, 6, 8, 12, 20, 40, 60])
--
scripts/bench/bench_barotropic_mcut.py-75-
scripts/bench/bench_barotropic_mcut.py-76-    comm = MPI.COMM_WORLD
scripts/bench/bench_barotropic_mcut.py-77-    rank, n_ranks = comm.Get_rank(), comm.Get_size()
scripts/bench/bench_barotropic_mcut.py:78:    n_lat, n_lon = args.n_lat, args.n_lon
scripts/bench/bench_barotropic_mcut.py:79:    if n_lat % n_ranks != 0:
scripts/bench/bench_barotropic_mcut.py-80-        if rank == 0:
scripts/bench/bench_barotropic_mcut.py:81:            print(f"SKIP: n_lat {n_lat} not divisible by n_ranks {n_ranks}")
scripts/bench/bench_barotropic_mcut.py-82-        return
scripts/bench/bench_barotropic_mcut.py-83-
scripts/bench/bench_barotropic_mcut.py-84-    coeff = jnp.asarray(args.coeff)
scripts/bench/bench_barotropic_mcut.py:85:    grid_raw = create_latlon_grid(n_lat, n_lon)
scripts/bench/bench_barotropic_mcut.py:86:    mask = jnp.asarray(_coastal_mask(n_lat, n_lon))
scripts/bench/bench_barotropic_mcut.py-87-    rng = np.random.default_rng(13)
scripts/bench/bench_barotropic_mcut.py:88:    H_cell = jnp.asarray(1000.0 + 500.0 * rng.random((n_lat, n_lon))) * mask
scripts/bench/bench_barotropic_mcut.py:89:    rhs_g = jnp.asarray(rng.standard_normal((n_lat, n_lon))) * mask
scripts/bench/bench_barotropic_mcut.py-90-
scripts/bench/bench_barotropic_mcut.py:91:    layout = make_latlon_band_layout(rank, n_ranks, n_lat, n_lon)
scripts/bench/bench_barotropic_mcut.py-92-    s, e = layout.lat_start, layout.lat_end
scripts/bench/bench_barotropic_mcut.py-93-    set_halo_backend("mpi", layout)
scripts/bench/bench_barotropic_mcut.py-94-
--
scripts/bench/bench_barotropic_mcut.py-96-    Hc_l, m_l = H_cell[s:e], mask[s:e]
scripts/bench/bench_barotropic_mcut.py-97-    rhs_l = rhs_g[s:e]
scripts/bench/bench_barotropic_mcut.py-98-    H_u, H_v, u_mask, v_mask = _faces_from_cell_depth(Hc_l, m_l,
scripts/bench/bench_barotropic_mcut.py:99:                                                      layout.n_lat_local, n_lon)
scripts/bench/bench_barotropic_mcut.py-100-    A_op = _make_helmholtz(H_u, H_v, coeff, grid_l, m_l, u_mask, v_mask)
scripts/bench/bench_barotropic_mcut.py-101-    inv_diag = _helmholtz_inv_diag(H_u, H_v, coeff, grid_l, m_l)
scripts/bench/bench_barotropic_mcut.py-102-    w = grid_l.area * m_l
--
scripts/bench/bench_barotropic_mcut.py-168-                 for (lbl, mi, M, var, red) in configs]
scripts/bench/bench_barotropic_mcut.py-169-        if rank == 0:
scripts/bench/bench_barotropic_mcut.py-170-            base = timed[0][1]   # jacobi_std
scripts/bench/bench_barotropic_mcut.py:171:            print(f"\n=== WALL-TIME np{n_ranks} ({n_lat}x{n_lon}) ===")
scripts/bench/bench_barotropic_mcut.py-172-            for lbl, t, red, M in timed:
scripts/bench/bench_barotropic_mcut.py-173-                print(f"  {lbl:20s} M={M:3d}: {t:8.3f} ms/solve "
scripts/bench/bench_barotropic_mcut.py-174-                      f"({red} reductions/step)  speedup={base / t:.2f}x")
--
scripts/bench/bench_barotropic_mcut.py-176-            os.makedirs(os.path.dirname(wt_out), exist_ok=True)
scripts/bench/bench_barotropic_mcut.py-177-            with open(wt_out, "w") as f:
scripts/bench/bench_barotropic_mcut.py-178-                f.write("config,M,ms_per_solve,reductions_per_step,speedup,"
scripts/bench/bench_barotropic_mcut.py:179:                        "n_ranks,n_lat,n_lon\n")
scripts/bench/bench_barotropic_mcut.py-180-                for lbl, t, red, M in timed:
scripts/bench/bench_barotropic_mcut.py-181-                    f.write(f"{lbl},{M},{t:.4f},{red},{base / t:.4f},"
scripts/bench/bench_barotropic_mcut.py:182:                            f"{n_ranks},{n_lat},{n_lon}\n")
scripts/bench/bench_barotropic_mcut.py-183-            print(f"wrote {wt_out}")
scripts/bench/bench_barotropic_mcut.py-184-
scripts/bench/bench_barotropic_mcut.py-185-    if rank == 0:
--
scripts/bench/bench_barotropic_mcut.py-189-        mj = m_to_target("jacobi")
scripts/bench/bench_barotropic_mcut.py-190-        mg = m_to_target("multigrid")
scripts/bench/bench_barotropic_mcut.py-191-        print(f"\n=== M to reach rel_residual<={args.target:.0e} "
scripts/bench/bench_barotropic_mcut.py:192:              f"(grid {n_lat}x{n_lon}, np{n_ranks}) ===")
scripts/bench/bench_barotropic_mcut.py-193-        print(f"  jacobi:    M={mj}  reductions/step={2 * mj if mj else 'NA'}")
scripts/bench/bench_barotropic_mcut.py-194-        print(f"  multigrid: M={mg}  reductions/step={2 * mg if mg else 'NA'}")
scripts/bench/bench_barotropic_mcut.py-195-        if mj and mg:
--
scripts/bench/bench_barotropic_mcut.py-197-                  f"reduction-count cut {2 * mj} -> {2 * mg}")
scripts/bench/bench_barotropic_mcut.py-198-        os.makedirs(os.path.dirname(args.out), exist_ok=True)
scripts/bench/bench_barotropic_mcut.py-199-        with open(args.out, "w") as f:
scripts/bench/bench_barotropic_mcut.py:200:            f.write("preconditioner,M,rel_residual,reductions_per_step,n_ranks,n_lat,n_lon\n")
scripts/bench/bench_barotropic_mcut.py-201-            for (nm, M, r) in rows:
scripts/bench/bench_barotropic_mcut.py:202:                f.write(f"{nm},{M},{r:.6e},{2 * M},{n_ranks},{n_lat},{n_lon}\n")
scripts/bench/bench_barotropic_mcut.py-203-        print(f"\nwrote {args.out}")
scripts/bench/bench_barotropic_mcut.py-204-
scripts/bench/bench_barotropic_mcut.py-205-
--
packages/coupler/legoesm/coupler/coupled_latlon_band.py-95-    """One EXPLICIT sensible-heat coupling update (band-local, per cell).
packages/coupler/legoesm/coupler/coupled_latlon_band.py-96-
packages/coupler/legoesm/coupler/coupled_latlon_band.py-97-    ``atm_state`` is a C-grid ``CGridLatLonHydrostaticState`` (rank-local band);
packages/coupler/legoesm/coupler/coupled_latlon_band.py:98:    ``sst`` is the co-located slab SST ``(n_lat_local, n_lon)`` [K].  The atm
packages/coupler/legoesm/coupler/coupled_latlon_band.py-99-    surface-layer temperature is the lowest model level ``T[..., -1]``.  Returns
packages/coupler/legoesm/coupler/coupled_latlon_band.py-100-    ``(atm_state_new, sst_new, clamp_energy)``: the surface-air T and SST updated
packages/coupler/legoesm/coupler/coupled_latlon_band.py-101-    by the equal-and-opposite flux ``F = k_exchange·(SST − T_sfc_air)``, plus the
--
packages/coupler/legoesm/coupler/coupled_latlon_band.py-116-    never fires, and a dedicated test checks the clamp bookkeeping directly).
packages/coupler/legoesm/coupler/coupled_latlon_band.py-117-    """
packages/coupler/legoesm/coupler/coupled_latlon_band.py-118-    T = atm_state.T
packages/coupler/legoesm/coupler/coupled_latlon_band.py:119:    t_sfc_air = T[..., -1]                       # (n_lat_local, n_lon)
packages/coupler/legoesm/coupler/coupled_latlon_band.py-120-    flux = cfg.k_exchange * (sst - t_sfc_air)    # W/m² into the atmosphere
packages/coupler/legoesm/coupler/coupled_latlon_band.py-121-    t_sfc_air_new = t_sfc_air + dt_couple * flux / cfg.c_atm_area
packages/coupler/legoesm/coupler/coupled_latlon_band.py-122-    sst_free = sst - dt_couple * flux / cfg.c_ocean_area
--
packages/coupler/legoesm/coupler/coupled_latlon_band.py-135-    EXCHANGE conserves EXACTLY.  The freezing clamp is the one non-conservative
packages/coupler/legoesm/coupler/coupled_latlon_band.py-136-    term: it is a SOURCE to this diagnosed budget (see
packages/coupler/legoesm/coupler/coupled_latlon_band.py-137-    :func:`apply_surface_coupling`), so ``Δ(this) = Σ area·clamp_energy``.
packages/coupler/legoesm/coupler/coupled_latlon_band.py:138:    ``area`` is the per-cell area ``(n_lat_local, n_lon)`` [m²] (rank-local
packages/coupler/legoesm/coupler/coupled_latlon_band.py-139-    band); the caller allreduces across ranks for the global total."""
packages/coupler/legoesm/coupler/coupled_latlon_band.py-140-    t_sfc_air = atm_state.T[..., -1]
packages/coupler/legoesm/coupler/coupled_latlon_band.py-141-    e_col = cfg.c_atm_area * t_sfc_air + cfg.c_ocean_area * sst
--
packages/ocean/legoesm/ocean/rpe.py-108-        )
packages/ocean/legoesm/ocean/rpe.py-109-    else:
packages/ocean/legoesm/ocean/rpe.py-110-        # lat-lon C-grid / regional / channel / spectral all expose
packages/ocean/legoesm/ocean/rpe.py:111:        # ``grid.area`` of shape ``(n_lat, n_lon)``.
packages/ocean/legoesm/ocean/rpe.py-112-        area = np.asarray(grid.area, dtype=np.float64)
packages/ocean/legoesm/ocean/rpe.py-113-
packages/ocean/legoesm/ocean/rpe.py-114-    # Layer thicknesses (rest dz_ref * jacobian via compute_layer_thickness).
--
packages/ocean/legoesm/ocean/advection.py-88-
packages/ocean/legoesm/ocean/advection.py-89-    Parameters
packages/ocean/legoesm/ocean/advection.py-90-    ----------
packages/ocean/legoesm/ocean/advection.py:91:    f : array, shape (n_lat, n_lon, nlev)
packages/ocean/legoesm/ocean/advection.py-92-        Tracer at cell centers.
packages/ocean/legoesm/ocean/advection.py:93:    mass_flux_u : array, shape (n_lat, n_lon+1, nlev)
packages/ocean/legoesm/ocean/advection.py-94-        Thickness-weighted velocity (h*u) at u-faces.
packages/ocean/legoesm/ocean/advection.py:95:    h_u : array, shape (n_lat, n_lon+1, nlev)
packages/ocean/legoesm/ocean/advection.py-96-        Layer thickness interpolated to u-faces.
packages/ocean/legoesm/ocean/advection.py-97-    grid : LatLonGrid
packages/ocean/legoesm/ocean/advection.py-98-    dt : float
--
packages/ocean/legoesm/ocean/advection.py-100-
packages/ocean/legoesm/ocean/advection.py-101-    Returns
packages/ocean/legoesm/ocean/advection.py-102-    -------
packages/ocean/legoesm/ocean/advection.py:103:    f_u : array, shape (n_lat, n_lon+1, nlev)
packages/ocean/legoesm/ocean/advection.py-104-        Tracer value at u-faces (to be multiplied by mass_flux_u for flux).
packages/ocean/legoesm/ocean/advection.py-105-    """
packages/ocean/legoesm/ocean/advection.py-106-    eps = 1e-30
packages/ocean/legoesm/ocean/advection.py:107:    n_lon = f.shape[1]
packages/ocean/legoesm/ocean/advection.py-108-
packages/ocean/legoesm/ocean/advection.py-109-    # Cell-width at u-face latitudes
packages/ocean/legoesm/ocean/advection.py-110-    if is_tripolar(grid):
packages/ocean/legoesm/ocean/advection.py:111:        dx_3d = grid.dx_u[:, :, jnp.newaxis]  # (n_lat, n_lon+1, 1)
packages/ocean/legoesm/ocean/advection.py-112-    else:
packages/ocean/legoesm/ocean/advection.py:113:        dx = grid.radius * grid.dlon * grid.cos_lat  # (n_lat,)
packages/ocean/legoesm/ocean/advection.py-114-        dx_3d = dx[:, jnp.newaxis, jnp.newaxis]
packages/ocean/legoesm/ocean/advection.py-115-
packages/ocean/legoesm/ocean/advection.py:116:    # Velocity and CFL at interior faces (n_lat, n_lon, nlev)
packages/ocean/legoesm/ocean/advection.py:117:    # Face j sits between cell (j-1) mod n_lon and cell j.
packages/ocean/legoesm/ocean/advection.py:118:    # mass_flux interior: first n_lon faces
packages/ocean/legoesm/ocean/advection.py:119:    mf = mass_flux_u[:, :n_lon, :]
packages/ocean/legoesm/ocean/advection.py-120-    t_grad = ratio_grad_floor(f.dtype)
packages/ocean/legoesm/ocean/advection.py:121:    h_face = h_u[:, :n_lon, :]
packages/ocean/legoesm/ocean/advection.py-122-    vel = grad_safe_ratio(mf, jnp.maximum(h_face, eps), h_face > t_grad)
packages/ocean/legoesm/ocean/advection.py-123-    cfl = jnp.minimum(jnp.abs(vel) * dt / dx_3d, 1.0)
packages/ocean/legoesm/ocean/advection.py-124-
packages/ocean/legoesm/ocean/advection.py-125-    # 5-point stencil (periodic in longitude)
packages/ocean/legoesm/ocean/advection.py-126-    # For face j: donor for positive flow is cell j-1, receiver is cell j
packages/ocean/legoesm/ocean/advection.py:127:    f_jm2 = jnp.roll(f, 2, axis=1)   # f[:, (j-2) % n_lon]
packages/ocean/legoesm/ocean/advection.py:128:    f_jm1 = jnp.roll(f, 1, axis=1)   # f[:, (j-1) % n_lon] = donor for +flow
packages/ocean/legoesm/ocean/advection.py-129-    f_j = f                            # f[:, j] = receiver for +flow
packages/ocean/legoesm/ocean/advection.py:130:    f_jp1 = jnp.roll(f, -1, axis=1)  # f[:, (j+1) % n_lon]
packages/ocean/legoesm/ocean/advection.py-131-
packages/ocean/legoesm/ocean/advection.py-132-    # --- Positive flow (from cell j-1 to cell j) ---
packages/ocean/legoesm/ocean/advection.py-133-    # Donor = f_jm1, Downstream = f_j, Upwind-of-donor = f_jm2
--
packages/ocean/legoesm/ocean/advection.py-171-    # Select based on flow direction
packages/ocean/legoesm/ocean/advection.py-172-    f_face = jnp.where(mf > 0, f_face_pos, f_face_neg)
packages/ocean/legoesm/ocean/advection.py-173-
packages/ocean/legoesm/ocean/advection.py:174:    # Wrap: face n_lon equals face 0 (periodic)
packages/ocean/legoesm/ocean/advection.py-175-    return jnp.concatenate([f_face, f_face[:, 0:1, :]], axis=1)
packages/ocean/legoesm/ocean/advection.py-176-
packages/ocean/legoesm/ocean/advection.py-177-
--
packages/ocean/legoesm/ocean/advection.py-194-
packages/ocean/legoesm/ocean/advection.py-195-    Parameters
packages/ocean/legoesm/ocean/advection.py-196-    ----------
packages/ocean/legoesm/ocean/advection.py:197:    f : array, shape (n_lat, n_lon, nlev)
packages/ocean/legoesm/ocean/advection.py-198-        Tracer at cell centers.
packages/ocean/legoesm/ocean/advection.py:199:    mass_flux_v : array, shape (n_lat+1, n_lon, nlev)
packages/ocean/legoesm/ocean/advection.py-200-        Thickness-weighted velocity (h*v) at v-faces.
packages/ocean/legoesm/ocean/advection.py:201:    h_v : array, shape (n_lat+1, n_lon, nlev)
packages/ocean/legoesm/ocean/advection.py-202-        Layer thickness interpolated to v-faces.
packages/ocean/legoesm/ocean/advection.py-203-    grid : LatLonGrid
packages/ocean/legoesm/ocean/advection.py-204-    dt : float
--
packages/ocean/legoesm/ocean/advection.py-206-
packages/ocean/legoesm/ocean/advection.py-207-    Returns
packages/ocean/legoesm/ocean/advection.py-208-    -------
packages/ocean/legoesm/ocean/advection.py:209:    f_v : array, shape (n_lat+1, n_lon, nlev)
packages/ocean/legoesm/ocean/advection.py-210-        Tracer value at v-faces. Zero at pole boundaries.
packages/ocean/legoesm/ocean/advection.py-211-    """
packages/ocean/legoesm/ocean/advection.py-212-    eps = 1e-30
packages/ocean/legoesm/ocean/advection.py:213:    n_lat = f.shape[0]
packages/ocean/legoesm/ocean/advection.py-214-
packages/ocean/legoesm/ocean/advection.py-215-    # Face-to-face distance at interior v-faces.
packages/ocean/legoesm/ocean/advection.py-216-    if is_tripolar(grid):
packages/ocean/legoesm/ocean/advection.py-217-        # Tripolar: per-cell meridional spacing from 2D metrics.
packages/ocean/legoesm/ocean/advection.py:218:        dy_v_int = grid.dy_v[1:-1, 0]  # (n_lat-1,) from interior rows
packages/ocean/legoesm/ocean/advection.py-219-    else:
packages/ocean/legoesm/ocean/advection.py-220-        # Regular or Mercator: variable-dy safe.
packages/ocean/legoesm/ocean/advection.py:221:        dy_h = grid.dy * 0.5                                # (n_lat,)
packages/ocean/legoesm/ocean/advection.py:222:        dy_v_int = 0.5 * (dy_h[1:] + dy_h[:-1])              # (n_lat-1,)
packages/ocean/legoesm/ocean/advection.py-223-
packages/ocean/legoesm/ocean/advection.py:224:    # Interior v-faces: indices 1 to n_lat-1 (between cells 0..n_lat-2 and 1..n_lat-1)
packages/ocean/legoesm/ocean/advection.py-225-    # Face i sits between cell i-1 (south) and cell i (north).
packages/ocean/legoesm/ocean/advection.py:226:    mf_int = mass_flux_v[1:-1, :, :]   # (n_lat-1, n_lon, nlev)
packages/ocean/legoesm/ocean/advection.py-227-    h_v_int = h_v[1:-1, :, :]
packages/ocean/legoesm/ocean/advection.py-228-    t_grad = ratio_grad_floor(f.dtype)
packages/ocean/legoesm/ocean/advection.py-229-    vel_int = grad_safe_ratio(
--
packages/ocean/legoesm/ocean/advection.py-231-    cfl = jnp.minimum(jnp.abs(vel_int) * dt / dy_v_int[:, jnp.newaxis, jnp.newaxis], 1.0)
packages/ocean/legoesm/ocean/advection.py-232-
packages/ocean/legoesm/ocean/advection.py-233-    # Build stencil with ghost cells at boundaries (Neumann: copy boundary value)
packages/ocean/legoesm/ocean/advection.py:234:    # Ghost: f[-1] = f[0], f[-2] = f[0] at south; f[n_lat] = f[n_lat-1] at north
packages/ocean/legoesm/ocean/advection.py-235-    f_ext = jnp.concatenate([f[:1, :, :], f[:1, :, :], f, f[-1:, :, :], f[-1:, :, :]], axis=0)
packages/ocean/legoesm/ocean/advection.py:236:    # f_ext indices: 0,1 = south ghosts; 2..n_lat+1 = real; n_lat+2, n_lat+3 = north ghosts
packages/ocean/legoesm/ocean/advection.py-237-    # Interior face i (1-indexed in original) corresponds to between cell i-1 and cell i.
packages/ocean/legoesm/ocean/advection.py-238-    # In f_ext, cell i-1 = index i+1, cell i = index i+2.
packages/ocean/legoesm/ocean/advection.py-239-
packages/ocean/legoesm/ocean/advection.py:240:    # Vectorized over all interior faces i=1..n_lat-1 (1-indexed):
packages/ocean/legoesm/ocean/advection.py-241-    # Cell k in original lives at f_ext[k+2].
packages/ocean/legoesm/ocean/advection.py-242-    # Face i is between cell i-1 (south) and cell i (north):
packages/ocean/legoesm/ocean/advection.py-243-    #   south-of-south = cell i-2 → f_ext[i]
packages/ocean/legoesm/ocean/advection.py-244-    #   south          = cell i-1 → f_ext[i+1]
packages/ocean/legoesm/ocean/advection.py-245-    #   north          = cell i   → f_ext[i+2]
packages/ocean/legoesm/ocean/advection.py-246-    #   north-of-north = cell i+1 → f_ext[i+3]
packages/ocean/legoesm/ocean/advection.py:247:    # For i=1..n_lat-1 the slices are:
packages/ocean/legoesm/ocean/advection.py:248:    f_south2 = f_ext[1:n_lat, :, :]          # f_ext[1..n_lat-1]
packages/ocean/legoesm/ocean/advection.py:249:    f_south = f_ext[2:n_lat + 1, :, :]       # f_ext[2..n_lat]
packages/ocean/legoesm/ocean/advection.py:250:    f_north = f_ext[3:n_lat + 2, :, :]       # f_ext[3..n_lat+1]
packages/ocean/legoesm/ocean/advection.py:251:    f_north2 = f_ext[4:n_lat + 3, :, :]      # f_ext[4..n_lat+2]
packages/ocean/legoesm/ocean/advection.py-252-
packages/ocean/legoesm/ocean/advection.py-253-    # --- Positive flow (south to north): donor = f_south, downstream = f_north ---
packages/ocean/legoesm/ocean/advection.py-254-    delta_pos = f_north - f_south
--
packages/ocean/legoesm/ocean/advection.py-459-
packages/ocean/legoesm/ocean/advection.py-460-    Parameters
packages/ocean/legoesm/ocean/advection.py-461-    ----------
packages/ocean/legoesm/ocean/advection.py:462:    tracer : (n_lat, n_lon, nlev) tracer field.
packages/ocean/legoesm/ocean/advection.py:463:    mass_flux_u : (n_lat, n_lon+1, nlev) at u-faces.
packages/ocean/legoesm/ocean/advection.py:464:    mass_flux_v : (n_lat+1, n_lon, nlev) at v-faces.
packages/ocean/legoesm/ocean/advection.py-465-    w_half : (..., nlev+1) vertical velocity on half levels.
packages/ocean/legoesm/ocean/advection.py:466:    h_k : (n_lat, n_lon, nlev) layer thickness at cell centers.
packages/ocean/legoesm/ocean/advection.py:467:    h_u : (n_lat, n_lon+1, nlev) layer thickness at u-faces.
packages/ocean/legoesm/ocean/advection.py:468:    h_v : (n_lat+1, n_lon, nlev) layer thickness at v-faces.
packages/ocean/legoesm/ocean/advection.py-469-    grid : LatLonGrid
packages/ocean/legoesm/ocean/advection.py-470-    dt : float
packages/ocean/legoesm/ocean/advection.py-471-
packages/ocean/legoesm/ocean/advection.py-472-    Returns
packages/ocean/legoesm/ocean/advection.py-473-    -------
packages/ocean/legoesm/ocean/advection.py:474:    div_h_flux : (n_lat, n_lon, nlev)
packages/ocean/legoesm/ocean/advection.py-475-        Horizontal flux divergence div(mass_flux * T_face).
packages/ocean/legoesm/ocean/advection.py:476:    vert_flux_div : (n_lat, n_lon, nlev)
packages/ocean/legoesm/ocean/advection.py-477-        Vertical flux divergence.
packages/ocean/legoesm/ocean/advection.py-478-    """
packages/ocean/legoesm/ocean/advection.py-479-    from legoesm.ocean.dynamics.latlon_cgrid_operators import divergence_cgrid
--
packages/ocean/legoesm/ocean/advection.py-531-
packages/ocean/legoesm/ocean/advection.py-532-    Parameters
packages/ocean/legoesm/ocean/advection.py-533-    ----------
packages/ocean/legoesm/ocean/advection.py:534:    f : array, shape (n_lat, n_lon, nlev)
packages/ocean/legoesm/ocean/advection.py-535-        Tracer at cell centers.
packages/ocean/legoesm/ocean/advection.py:536:    mass_flux_u : array, shape (n_lat, n_lon+1, nlev)
packages/ocean/legoesm/ocean/advection.py-537-        Thickness-weighted velocity at u-faces (sign determines upwind).
packages/ocean/legoesm/ocean/advection.py-538-
packages/ocean/legoesm/ocean/advection.py-539-    Returns
packages/ocean/legoesm/ocean/advection.py-540-    -------
packages/ocean/legoesm/ocean/advection.py:541:    f_u : array, shape (n_lat, n_lon+1, nlev)
packages/ocean/legoesm/ocean/advection.py-542-        PPM face values at u-points.
packages/ocean/legoesm/ocean/advection.py-543-    """
packages/ocean/legoesm/ocean/advection.py-544-    from legoesm.core.operators_fv import ppm_edge_values, ppm_limit
packages/ocean/legoesm/ocean/advection.py-545-
packages/ocean/legoesm/ocean/advection.py:546:    n_lat, n_lon, nlev = f.shape
packages/ocean/legoesm/ocean/advection.py-547-
packages/ocean/legoesm/ocean/advection.py-548-    # Pad longitude with halo=2 (periodic)
packages/ocean/legoesm/ocean/advection.py-549-    f_pad = jnp.concatenate([f[:, -2:, :], f, f[:, :2, :]], axis=1)
packages/ocean/legoesm/ocean/advection.py:550:    # f_pad shape: (n_lat, n_lon+4, nlev)
packages/ocean/legoesm/ocean/advection.py-551-
packages/ocean/legoesm/ocean/advection.py-552-    # PPM edge values along longitude (axis=-2 must be the reconstruction dir)
packages/ocean/legoesm/ocean/advection.py:553:    # Rearrange to (n_lat, n_lon+4, nlev) → axis=-2 is already longitude ✓
packages/ocean/legoesm/ocean/advection.py-554-    # But ppm_edge_values operates on axis=-2 with shape (..., M, K)
packages/ocean/legoesm/ocean/advection.py:555:    # We need (nlev, n_lat, n_lon+4) then compute along axis=-2=n_lon+4... no.
packages/ocean/legoesm/ocean/advection.py:556:    # Actually ppm_edge_values needs shape (..., M, K) where M=n_lon+4 is the
packages/ocean/legoesm/ocean/advection.py:557:    # direction. Our shape is (n_lat, n_lon+4, nlev): axis=-2=n_lon+4 ✓!
packages/ocean/legoesm/ocean/advection.py:558:    q_hat = ppm_edge_values(f_pad)  # (n_lat, n_lon+3, nlev)
packages/ocean/legoesm/ocean/advection.py-559-
packages/ocean/legoesm/ocean/advection.py-560-    # Left/right edge values for each cell
packages/ocean/legoesm/ocean/advection.py:561:    a_L = q_hat[:, :-1, :]   # (n_lat, n_lon+2, nlev)
packages/ocean/legoesm/ocean/advection.py:562:    a_R = q_hat[:, 1:, :]    # (n_lat, n_lon+2, nlev)
packages/ocean/legoesm/ocean/advection.py:563:    q_c = f_pad[:, 1:-1, :]  # (n_lat, n_lon+2, nlev) — cell averages
packages/ocean/legoesm/ocean/advection.py-564-
packages/ocean/legoesm/ocean/advection.py-565-    # Colella-Woodward limiter
packages/ocean/legoesm/ocean/advection.py-566-    a_L, a_R = ppm_limit(q_c, a_L, a_R)
--
packages/ocean/legoesm/ocean/advection.py-568-    # At face j (between cell j-1 and cell j):
packages/ocean/legoesm/ocean/advection.py-569-    # - positive flow → use right edge of cell j-1 = a_R[j-1] (in padded coords: a_R[j])
packages/ocean/legoesm/ocean/advection.py-570-    # - negative flow → use left edge of cell j = a_L[j] (in padded coords: a_L[j+1])
packages/ocean/legoesm/ocean/advection.py:571:    # Interior faces in original coords: j=0..n_lon (n_lon+1 faces, wrapping)
packages/ocean/legoesm/ocean/advection.py:572:    # In padded+reconstructed coords: j=0..n_lon maps to a_R[1..n_lon+1], a_L[2..n_lon+2]
packages/ocean/legoesm/ocean/advection.py:573:    q_R_left = a_R[:, 1:n_lon + 2, :]   # (n_lat, n_lon+1, nlev)
packages/ocean/legoesm/ocean/advection.py:574:    q_L_right = a_L[:, 2:n_lon + 3, :]  # (n_lat, n_lon+1, nlev)
packages/ocean/legoesm/ocean/advection.py-575-
packages/ocean/legoesm/ocean/advection.py:576:    # But we have n_lon+2 elements in a_R/a_L. Let me recheck...
packages/ocean/legoesm/ocean/advection.py:577:    # a_R shape = (n_lat, n_lon+2, nlev). Indices 0..n_lon+1.
packages/ocean/legoesm/ocean/advection.py:578:    # For n_lon+1 faces (including periodic wrap):
packages/ocean/legoesm/ocean/advection.py:579:    # Face j=0..n_lon: use a_R[1:n_lon+2] and a_L[2:n_lon+3]...
packages/ocean/legoesm/ocean/advection.py:580:    # But a_L only has n_lon+2 elements (indices 0..n_lon+1), so a_L[2:n_lon+3]
packages/ocean/legoesm/ocean/advection.py:581:    # would exceed bounds for large n_lon. Let me fix:
packages/ocean/legoesm/ocean/advection.py:582:    # Face j in original (0-indexed, j=0..n_lon):
packages/ocean/legoesm/ocean/advection.py:583:    #   Left cell = (j-1) mod n_lon → in padded: index j+1 (halo offset)
packages/ocean/legoesm/ocean/advection.py-584-    #   Right cell = j → in padded: index j+2
packages/ocean/legoesm/ocean/advection.py-585-    # a_R for left cell: a_R[j+1-1] = a_R[j] (right edge of padded cell j+1, but...)
packages/ocean/legoesm/ocean/advection.py-586-
packages/ocean/legoesm/ocean/advection.py:587:    # Actually simpler: after PPM on (n_lat, n_lon+4, nlev), we get n_lon+3 edges.
packages/ocean/legoesm/ocean/advection.py:588:    # Remove the outermost edges (fully in halo): keep inner n_lon+1 edges.
packages/ocean/legoesm/ocean/advection.py:589:    # These correspond to faces 0..n_lon in the original grid.
packages/ocean/legoesm/ocean/advection.py:590:    q_hat[:, 1:-1, :]  # (n_lat, n_lon+1, nlev) — interior edges
packages/ocean/legoesm/ocean/advection.py-591-
packages/ocean/legoesm/ocean/advection.py-592-    # For each edge, the left cell's right-edge is the edge value approached from left,
packages/ocean/legoesm/ocean/advection.py-593-    # and the right cell's left-edge is approached from right.
--
packages/ocean/legoesm/ocean/advection.py-597-
packages/ocean/legoesm/ocean/advection.py-598-    # Re-derive from the limited a_L, a_R:
packages/ocean/legoesm/ocean/advection.py-599-    # a_L[i], a_R[i] are left/right edges of the i-th cell in the padded array.
packages/ocean/legoesm/ocean/advection.py:600:    # Padded cells: 0(halo), 1(halo), 2..n_lon+1(real), n_lon+2(halo), n_lon+3(halo)
packages/ocean/legoesm/ocean/advection.py:601:    # Real cells in padded: indices 2..n_lon+1
packages/ocean/legoesm/ocean/advection.py-602-    # Faces between real cells: face j (0-indexed) is between real cell j and j+1
packages/ocean/legoesm/ocean/advection.py-603-    #   = between padded cells j+2 and j+3
packages/ocean/legoesm/ocean/advection.py-604-    #   Left cell right edge = a_R[j+2-1] = a_R[j+1]... no, a_R[k] is right edge of
packages/ocean/legoesm/ocean/advection.py-605-    #   padded cell k. So right edge of padded cell j+2 = a_R[j+2].
packages/ocean/legoesm/ocean/advection.py:606:    #   But wait: a_R has shape (n_lon+2) — indices 0..n_lon+1.
packages/ocean/legoesm/ocean/advection.py:607:    #   Padded cells from which a_L/a_R are computed: cells 1..(n_lon+2) in f_pad
packages/ocean/legoesm/ocean/advection.py:608:    #   (from q_c = f_pad[:, 1:-1, :] which is cells 1..n_lon+2, i.e. n_lon+2 cells)
packages/ocean/legoesm/ocean/advection.py:609:    #   So a_L[k], a_R[k] for k=0..n_lon+1 correspond to padded cells 1..n_lon+2.
packages/ocean/legoesm/ocean/advection.py:610:    #   Real data cells in padded: 2..n_lon+1 → a_L/a_R indices 1..n_lon.
packages/ocean/legoesm/ocean/advection.py-611-    #   Face j between real cells j and j+1:
packages/ocean/legoesm/ocean/advection.py-612-    #     = between a_L/a_R indices j+1 and j+2
packages/ocean/legoesm/ocean/advection.py-613-    #     → left cell right edge = a_R[j+1]
packages/ocean/legoesm/ocean/advection.py-614-    #     → right cell left edge = a_L[j+2]
packages/ocean/legoesm/ocean/advection.py:615:    #   For j=0..n_lon-1: a_R[1..n_lon] and a_L[2..n_lon+1]
packages/ocean/legoesm/ocean/advection.py:616:    #   For periodic face j=n_lon (=face 0): same as face 0.
packages/ocean/legoesm/ocean/advection.py-617-
packages/ocean/legoesm/ocean/advection.py:618:    # Upwind selection for n_lon interior faces + 1 periodic wrap:
packages/ocean/legoesm/ocean/advection.py:619:    q_R_left = a_R[:, 1:n_lon + 1, :]    # (n_lat, n_lon, nlev) — right edge of left cell
packages/ocean/legoesm/ocean/advection.py:620:    q_L_right = a_L[:, 2:n_lon + 2, :]   # (n_lat, n_lon, nlev) — left edge of right cell
packages/ocean/legoesm/ocean/advection.py-621-
packages/ocean/legoesm/ocean/advection.py:622:    mf = mass_flux_u[:, :n_lon, :]  # interior n_lon faces
packages/ocean/legoesm/ocean/advection.py-623-    f_face = jnp.where(mf > 0, q_R_left, q_L_right)
packages/ocean/legoesm/ocean/advection.py-624-
packages/ocean/legoesm/ocean/advection.py-625-    # Monotonicity clamp (local bounds of adjacent cells)
--
packages/ocean/legoesm/ocean/advection.py-628-    f_face = jnp.clip(f_face, jnp.minimum(f_left, f_right),
packages/ocean/legoesm/ocean/advection.py-629-                       jnp.maximum(f_left, f_right))
packages/ocean/legoesm/ocean/advection.py-630-
packages/ocean/legoesm/ocean/advection.py:631:    # Periodic wrap: face n_lon = face 0
packages/ocean/legoesm/ocean/advection.py-632-    return jnp.concatenate([f_face, f_face[:, 0:1, :]], axis=1)
packages/ocean/legoesm/ocean/advection.py-633-
packages/ocean/legoesm/ocean/advection.py-634-
--
packages/ocean/legoesm/ocean/advection.py-642-
packages/ocean/legoesm/ocean/advection.py-643-    Parameters
packages/ocean/legoesm/ocean/advection.py-644-    ----------
packages/ocean/legoesm/ocean/advection.py:645:    f : array, shape (n_lat, n_lon, nlev)
packages/ocean/legoesm/ocean/advection.py-646-        Tracer at cell centers.
packages/ocean/legoesm/ocean/advection.py:647:    mass_flux_v : array, shape (n_lat+1, n_lon, nlev)
packages/ocean/legoesm/ocean/advection.py-648-        Thickness-weighted velocity at v-faces.
packages/ocean/legoesm/ocean/advection.py-649-
packages/ocean/legoesm/ocean/advection.py-650-    Returns
packages/ocean/legoesm/ocean/advection.py-651-    -------
packages/ocean/legoesm/ocean/advection.py:652:    f_v : array, shape (n_lat+1, n_lon, nlev)
packages/ocean/legoesm/ocean/advection.py-653-        PPM face values at v-points. Zero at pole boundaries.
packages/ocean/legoesm/ocean/advection.py-654-    """
packages/ocean/legoesm/ocean/advection.py-655-    from legoesm.core.operators_fv import ppm_limit
packages/ocean/legoesm/ocean/advection.py-656-
packages/ocean/legoesm/ocean/advection.py:657:    n_lat, n_lon, nlev = f.shape
packages/ocean/legoesm/ocean/advection.py-658-
packages/ocean/legoesm/ocean/advection.py-659-    # Extended field with ghost cells (Neumann BC: reflect boundary rows)
packages/ocean/legoesm/ocean/advection.py-660-    f_ext = jnp.concatenate(
packages/ocean/legoesm/ocean/advection.py-661-        [f[1::-1, :, :], f, f[-1:-3:-1, :, :]], axis=0)
packages/ocean/legoesm/ocean/advection.py:662:    # f_ext shape: (n_lat+4, n_lon, nlev)
packages/ocean/legoesm/ocean/advection.py:663:    # Indices: 0,1=south ghosts; 2..n_lat+1=real; n_lat+2,n_lat+3=north ghosts
packages/ocean/legoesm/ocean/advection.py-664-
packages/ocean/legoesm/ocean/advection.py:665:    # 4th-order edge values at interior faces i=1..n_lat-1:
packages/ocean/legoesm/ocean/advection.py-666-    # Face i is between cell i-1 and cell i in original.
packages/ocean/legoesm/ocean/advection.py-667-    # In f_ext: cell i-1=index i+1, cell i=index i+2.
packages/ocean/legoesm/ocean/advection.py-668-    # a_i = (7/12)*(f_ext[i+1]+f_ext[i+2]) - (1/12)*(f_ext[i]+f_ext[i+3])
packages/ocean/legoesm/ocean/advection.py:669:    s0 = f_ext[1:n_lat, :, :]      # f_ext[i] for i=1..n_lat-1
packages/ocean/legoesm/ocean/advection.py:670:    s1 = f_ext[2:n_lat + 1, :, :]  # cell i-1
packages/ocean/legoesm/ocean/advection.py:671:    s2 = f_ext[3:n_lat + 2, :, :]  # cell i
packages/ocean/legoesm/ocean/advection.py:672:    s3 = f_ext[4:n_lat + 3, :, :]  # f_ext[i+3]
packages/ocean/legoesm/ocean/advection.py-673-
packages/ocean/legoesm/ocean/advection.py-674-    a_int = (7.0 / 12.0) * (s1 + s2) - (1.0 / 12.0) * (s0 + s3)
packages/ocean/legoesm/ocean/advection.py-675-    # Monotone clamp between neighbors
packages/ocean/legoesm/ocean/advection.py-676-    a_int = jnp.clip(a_int, jnp.minimum(s1, s2), jnp.maximum(s1, s2))
packages/ocean/legoesm/ocean/advection.py:677:    # shape: (n_lat-1, n_lon, nlev)
packages/ocean/legoesm/ocean/advection.py-678-
packages/ocean/legoesm/ocean/advection.py:679:    # Build full edge array (n_lat+1 interfaces):
packages/ocean/legoesm/ocean/advection.py:680:    # Face 0=south wall, faces 1..n_lat-1=interior, face n_lat=north wall
packages/ocean/legoesm/ocean/advection.py-681-    a_full = jnp.concatenate([f[:1, :, :], a_int, f[-1:, :, :]], axis=0)
packages/ocean/legoesm/ocean/advection.py:682:    # shape: (n_lat+1, n_lon, nlev)
packages/ocean/legoesm/ocean/advection.py-683-
packages/ocean/legoesm/ocean/advection.py-684-    # Left/right edges per cell:
packages/ocean/legoesm/ocean/advection.py-685-    # Cell j: T_L=a_full[j] (south edge), T_R=a_full[j+1] (north edge)
packages/ocean/legoesm/ocean/advection.py:686:    T_L = a_full[:-1, :, :]  # (n_lat, n_lon, nlev)
packages/ocean/legoesm/ocean/advection.py-687-    T_R = a_full[1:, :, :]
packages/ocean/legoesm/ocean/advection.py-688-
packages/ocean/legoesm/ocean/advection.py-689-    # Colella-Woodward limiter
--
packages/ocean/legoesm/ocean/advection.py-692-    # Upwind face value at interior faces:
packages/ocean/legoesm/ocean/advection.py-693-    # Positive flow (south→north): donor=cell i-1, use T_R of cell i-1
packages/ocean/legoesm/ocean/advection.py-694-    # Negative flow (north→south): donor=cell i, use T_L of cell i
packages/ocean/legoesm/ocean/advection.py:695:    T_R_south = T_R[:-1, :, :]  # T_R of cells 0..n_lat-2
packages/ocean/legoesm/ocean/advection.py:696:    T_L_north = T_L[1:, :, :]   # T_L of cells 1..n_lat-1
packages/ocean/legoesm/ocean/advection.py-697-
packages/ocean/legoesm/ocean/advection.py-698-    mf_int = mass_flux_v[1:-1, :, :]
packages/ocean/legoesm/ocean/advection.py-699-    f_face = jnp.where(mf_int > 0, T_R_south, T_L_north)
--
packages/ocean/legoesm/ocean/advection.py-805-
packages/ocean/legoesm/ocean/advection.py-806-    NEMO ``traadv_fct`` high-order horizontal flux with ``nn_fct_h=2``
packages/ocean/legoesm/ocean/advection.py-807-    (``0.5*pU*(pt(ji)+pt(ji+1))``).  Periodic in longitude; returns
packages/ocean/legoesm/ocean/advection.py:808:    (n_lat, n_lon+1, nlev) with the wrap column appended.
packages/ocean/legoesm/ocean/advection.py-809-    """
packages/ocean/legoesm/ocean/advection.py-810-    f_face = 0.5 * (jnp.roll(f, 1, axis=1) + f)          # face j: cells j-1, j
packages/ocean/legoesm/ocean/advection.py-811-    return jnp.concatenate([f_face, f_face[:, 0:1, :]], axis=1)
--
packages/ocean/legoesm/ocean/advection.py-814-def centred2_to_v_points(f: jnp.ndarray) -> jnp.ndarray:
packages/ocean/legoesm/ocean/advection.py-815-    """2nd-order centred tracer at v-faces: 0.5·(T_south + T_north).
packages/ocean/legoesm/ocean/advection.py-816-
packages/ocean/legoesm/ocean/advection.py:817:    Wall faces (j=0, n_lat) copy the adjacent cell — their mass flux is
packages/ocean/legoesm/ocean/advection.py:818:    zero so the value only needs to be finite. Returns (n_lat+1, n_lon,
packages/ocean/legoesm/ocean/advection.py-819-    nlev).
packages/ocean/legoesm/ocean/advection.py-820-    """
packages/ocean/legoesm/ocean/advection.py:821:    f_int = 0.5 * (f[:-1, :, :] + f[1:, :, :])           # interior n_lat-1 faces
packages/ocean/legoesm/ocean/advection.py-822-    return jnp.concatenate([f[:1, :, :], f_int, f[-1:, :, :]], axis=0)
packages/ocean/legoesm/ocean/advection.py-823-
packages/ocean/legoesm/ocean/advection.py-824-
--
packages/ocean/legoesm/ocean/advection.py-850-
packages/ocean/legoesm/ocean/advection.py-851-    Parameters
packages/ocean/legoesm/ocean/advection.py-852-    ----------
packages/ocean/legoesm/ocean/advection.py:853:    tracer : (n_lat, n_lon, nlev)
packages/ocean/legoesm/ocean/advection.py:854:    mass_flux_u : (n_lat, n_lon+1, nlev)
packages/ocean/legoesm/ocean/advection.py:855:    mass_flux_v : (n_lat+1, n_lon, nlev)
packages/ocean/legoesm/ocean/advection.py:856:    w_half : (n_lat, n_lon, nlev+1)
packages/ocean/legoesm/ocean/advection.py:857:    h_k : (n_lat, n_lon, nlev) layer thickness
packages/ocean/legoesm/ocean/advection.py-858-    grid : LatLonGrid
packages/ocean/legoesm/ocean/advection.py-859-    dt : float
packages/ocean/legoesm/ocean/advection.py-860-    high_order : {"ppm", "centred2"}
--
packages/ocean/legoesm/ocean/advection.py-864-        DINO / ORCA1 namelist selection). The centred face value is NOT
packages/ocean/legoesm/ocean/advection.py-865-        pre-clamped to local bounds (NEMO doesn't); the Zalesak step
packages/ocean/legoesm/ocean/advection.py-866-        supplies all the monotonicity.
packages/ocean/legoesm/ocean/advection.py:867:    tracer_before : (n_lat, n_lon, nlev) or None
packages/ocean/legoesm/ocean/advection.py-868-        BEFORE-level tracer (Kbb) for the leapfrog outer step.  Under the
packages/ocean/legoesm/ocean/advection.py-869-        modified leap-frog the FCT-limited advective increment is applied to
packages/ocean/legoesm/ocean/advection.py-870-        the BEFORE state (``T(Naa) = T(Nbb) + 2dt·RHS``), so the Zalesak
--
packages/ocean/legoesm/ocean/advection.py-880-
packages/ocean/legoesm/ocean/advection.py-881-    Returns
packages/ocean/legoesm/ocean/advection.py-882-    -------
packages/ocean/legoesm/ocean/advection.py:883:    div_h_flux : (n_lat, n_lon, nlev) horizontal flux divergence
packages/ocean/legoesm/ocean/advection.py:884:    vert_flux_div : (n_lat, n_lon, nlev) vertical flux divergence
packages/ocean/legoesm/ocean/advection.py-885-    """
packages/ocean/legoesm/ocean/advection.py-886-    from legoesm.ocean.dynamics.latlon_cgrid_operators import divergence_cgrid
packages/ocean/legoesm/ocean/advection.py-887-    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
--
packages/ocean/legoesm/ocean/advection.py-972-
packages/ocean/legoesm/ocean/advection.py-973-    # --- Step 3: True sign-split Zalesak (1979) limiter (issue #212) ---
packages/ocean/legoesm/ocean/advection.py-974-    # Anti-diffusive face fluxes:
packages/ocean/legoesm/ocean/advection.py:975:    ad_flux_u = flux_u_hi - flux_u_low      # (n_lat, n_lon+1, nlev)
packages/ocean/legoesm/ocean/advection.py:976:    ad_flux_v = flux_v_hi - flux_v_low      # (n_lat+1, n_lon, nlev)
packages/ocean/legoesm/ocean/advection.py-977-    ad_vert_int = F_vert_hi_int - F_vert_low_int  # (..., nlev-1)
packages/ocean/legoesm/ocean/advection.py-978-
packages/ocean/legoesm/ocean/advection.py-979-    # Local min / max over the (cell + 6 neighbours) stencil.  For non-
--
packages/ocean/legoesm/ocean/advection.py-1037-
packages/ocean/legoesm/ocean/advection.py-1038-    Parameters
packages/ocean/legoesm/ocean/advection.py-1039-    ----------
packages/ocean/legoesm/ocean/advection.py:1040:    f : array, shape (n_lat, n_lon, nlev)
packages/ocean/legoesm/ocean/advection.py-1041-        Tracer at cell centers.
packages/ocean/legoesm/ocean/advection.py:1042:    mass_flux_u : array, shape (n_lat, n_lon+1, nlev)
packages/ocean/legoesm/ocean/advection.py-1043-        Thickness-weighted velocity at u-faces (sign determines upwind).
packages/ocean/legoesm/ocean/advection.py-1044-    order : {5, 7}
packages/ocean/legoesm/ocean/advection.py-1045-        WENO order.
packages/ocean/legoesm/ocean/advection.py-1046-
packages/ocean/legoesm/ocean/advection.py-1047-    Returns
packages/ocean/legoesm/ocean/advection.py-1048-    -------
packages/ocean/legoesm/ocean/advection.py:1049:    f_u : array, shape (n_lat, n_lon+1, nlev)
packages/ocean/legoesm/ocean/advection.py-1050-        WENO face values at u-points.
packages/ocean/legoesm/ocean/advection.py-1051-    """
packages/ocean/legoesm/ocean/advection.py-1052-    weno_fn = {5: weno5_z, 7: weno7_z}[order]
packages/ocean/legoesm/ocean/advection.py-1053-    hw = {5: 3, 7: 4}[order]
packages/ocean/legoesm/ocean/advection.py:1054:    n_lon = f.shape[1]
packages/ocean/legoesm/ocean/advection.py-1055-
packages/ocean/legoesm/ocean/advection.py-1056-    # Convert point values to cell averages. WENO reconstruction is
packages/ocean/legoesm/ocean/advection.py-1057-    # a finite-volume method expecting cell-average inputs; passing
--
packages/ocean/legoesm/ocean/advection.py-1068-
packages/ocean/legoesm/ocean/advection.py-1069-    f_plus, f_minus = weno_fn(stencil)
packages/ocean/legoesm/ocean/advection.py-1070-
packages/ocean/legoesm/ocean/advection.py:1071:    mf = mass_flux_u[:, :n_lon, :]
packages/ocean/legoesm/ocean/advection.py-1072-    f_face = weno_upwind(f_plus, f_minus, mf)
packages/ocean/legoesm/ocean/advection.py-1073-
packages/ocean/legoesm/ocean/advection.py:1074:    # Periodic wrap: face n_lon = face 0
packages/ocean/legoesm/ocean/advection.py-1075-    return jnp.concatenate([f_face, f_face[:, 0:1, :]], axis=1)
packages/ocean/legoesm/ocean/advection.py-1076-
packages/ocean/legoesm/ocean/advection.py-1077-
--
packages/ocean/legoesm/ocean/advection.py-1087-
packages/ocean/legoesm/ocean/advection.py-1088-    Parameters
packages/ocean/legoesm/ocean/advection.py-1089-    ----------
packages/ocean/legoesm/ocean/advection.py:1090:    f : array, shape (n_lat, n_lon, nlev)
packages/ocean/legoesm/ocean/advection.py-1091-        Tracer at cell centers.
packages/ocean/legoesm/ocean/advection.py:1092:    mass_flux_v : array, shape (n_lat+1, n_lon, nlev)
packages/ocean/legoesm/ocean/advection.py-1093-        Thickness-weighted velocity at v-faces.
packages/ocean/legoesm/ocean/advection.py-1094-    order : {5, 7}
packages/ocean/legoesm/ocean/advection.py-1095-        WENO order.
packages/ocean/legoesm/ocean/advection.py-1096-
packages/ocean/legoesm/ocean/advection.py-1097-    Returns
packages/ocean/legoesm/ocean/advection.py-1098-    -------
packages/ocean/legoesm/ocean/advection.py:1099:    f_v : array, shape (n_lat+1, n_lon, nlev)
packages/ocean/legoesm/ocean/advection.py-1100-        WENO face values at v-points. Zero at pole boundaries.
packages/ocean/legoesm/ocean/advection.py-1101-    """
packages/ocean/legoesm/ocean/advection.py-1102-    weno_fn = {5: weno5_z, 7: weno7_z}[order]
packages/ocean/legoesm/ocean/advection.py-1103-    hw = {5: 3, 7: 4}[order]
packages/ocean/legoesm/ocean/advection.py:1104:    n_lat = f.shape[0]
packages/ocean/legoesm/ocean/advection.py-1105-
packages/ocean/legoesm/ocean/advection.py-1106-    # Convert point values to cell averages (meridional, bounded).
packages/ocean/legoesm/ocean/advection.py-1107-    from legoesm.core.weno import point_to_cellavg_bounded
--
packages/ocean/legoesm/ocean/advection.py-1113-        [f_avg[:1, :, :]] * hw + [f_avg] + [f_avg[-1:, :, :]] * hw, axis=0
packages/ocean/legoesm/ocean/advection.py-1114-    )
packages/ocean/legoesm/ocean/advection.py-1115-
packages/ocean/legoesm/ocean/advection.py:1116:    # Stencil for interior faces i=1..n_lat-1.
packages/ocean/legoesm/ocean/advection.py-1117-    # Cell k in original = f_ext[k + hw].
packages/ocean/legoesm/ocean/advection.py-1118-    # Face i: WENO at I+1/2 where I = i-1. Need cells i-hw..i+(hw-1).
packages/ocean/legoesm/ocean/advection.py-1119-    # In f_ext: indices i..i+(2*hw-1).
packages/ocean/legoesm/ocean/advection.py:1120:    stencil = [f_ext[1 + j: n_lat + j, :, :] for j in range(2 * hw)]
packages/ocean/legoesm/ocean/advection.py-1121-
packages/ocean/legoesm/ocean/advection.py-1122-    f_plus, f_minus = weno_fn(stencil)
packages/ocean/legoesm/ocean/advection.py-1123-
--
packages/ocean/legoesm/ocean/advection.py-1314-
packages/ocean/legoesm/ocean/advection.py-1315-    Parameters
packages/ocean/legoesm/ocean/advection.py-1316-    ----------
packages/ocean/legoesm/ocean/advection.py:1317:    ad_flux_u : array (n_lat, n_lon+1, nlev) — anti-diffusive u-face flux
packages/ocean/legoesm/ocean/advection.py:1318:    ad_flux_v : array (n_lat+1, n_lon, nlev) — anti-diffusive v-face flux
packages/ocean/legoesm/ocean/advection.py:1319:    ad_vert_int : array (n_lat, n_lon, nlev-1) — anti-diffusive interior
packages/ocean/legoesm/ocean/advection.py-1320-        vertical interface flux (positive = upward).
packages/ocean/legoesm/ocean/advection.py:1321:    q_td : array (n_lat, n_lon, nlev) — provisional low-order update.
packages/ocean/legoesm/ocean/advection.py:1322:    q_min, q_max : array (n_lat, n_lon, nlev) — local stencil bounds.
packages/ocean/legoesm/ocean/advection.py:1323:    h_k : array (n_lat, n_lon, nlev) — layer thickness.
packages/ocean/legoesm/ocean/advection.py-1324-    dt : float — baroclinic time step.
packages/ocean/legoesm/ocean/advection.py-1325-    grid : LatLonGrid.
packages/ocean/legoesm/ocean/advection.py-1326-    eps : float — divide-by-zero guard for empty P+/P-.
packages/ocean/legoesm/ocean/advection.py-1327-
packages/ocean/legoesm/ocean/advection.py-1328-    Returns
packages/ocean/legoesm/ocean/advection.py-1329-    -------
packages/ocean/legoesm/ocean/advection.py:1330:    alpha_u_full : (n_lat, n_lon+1, nlev) — alpha for each u-face, with
packages/ocean/legoesm/ocean/advection.py:1331:        ``face[n_lon] == face[0]`` for periodic-x.
packages/ocean/legoesm/ocean/advection.py:1332:    alpha_v : (n_lat+1, n_lon, nlev) — alpha for each v-face; the two
packages/ocean/legoesm/ocean/advection.py-1333-        wall faces carry placeholder 1.0 (their face flux is zero by
packages/ocean/legoesm/ocean/advection.py-1334-        the wall mask, so any alpha is harmless).
packages/ocean/legoesm/ocean/advection.py:1335:    alpha_vert_face : (n_lat, n_lon, nlev-1) — alpha for each interior
packages/ocean/legoesm/ocean/advection.py-1336-        vertical interface.
packages/ocean/legoesm/ocean/advection.py-1337-    """
packages/ocean/legoesm/ocean/advection.py-1338-    # Local positive / negative parts of the anti-diffusive face fluxes.
--
packages/ocean/legoesm/ocean/advection.py-1347-    if is_tripolar(grid):
packages/ocean/legoesm/ocean/advection.py-1348-        # Tripolar: use full 2D metrics — column-0 extraction is NOT
packages/ocean/legoesm/ocean/advection.py-1349-        # valid on the bipolar cap where dy_u/dx_v vary in longitude.
packages/ocean/legoesm/ocean/advection.py:1350:        face_dy = grid.dy_u                               # (n_lat, n_lon+1)
packages/ocean/legoesm/ocean/advection.py:1351:        face_dx = grid.dx_v                               # (n_lat+1, n_lon)
packages/ocean/legoesm/ocean/advection.py-1352-        _is_2d_dy = True
packages/ocean/legoesm/ocean/advection.py-1353-        _is_2d_dx = True
packages/ocean/legoesm/ocean/advection.py-1354-    else:
packages/ocean/legoesm/ocean/advection.py-1355-        R_planet = grid.radius
packages/ocean/legoesm/ocean/advection.py-1356-        dlon = grid.dlon
packages/ocean/legoesm/ocean/advection.py-1357-        # face_dy at h-points: cell-row meridional extent (1D, Mercator-safe).
packages/ocean/legoesm/ocean/advection.py:1358:        face_dy = (grid.dy * 0.5)[:, jnp.newaxis, jnp.newaxis]  # (n_lat,1,1)
packages/ocean/legoesm/ocean/advection.py-1359-        # #516: single-source v-face zonal cos(lat_v) (interior
packages/ocean/legoesm/ocean/advection.py-1360-        # cos(0.5·(lat[j]+lat[j+1])), poles 0) — routes through the shared
packages/ocean/legoesm/ocean/advection.py-1361-        # backend-aware helper so an MPI lat-band cut keeps the neighbour-rank
packages/ocean/legoesm/ocean/advection.py-1362-        # metric instead of the old serial jnp.pad (which zeroed local band
packages/ocean/legoesm/ocean/advection.py-1363-        # edges).  Bit-identical on serial.
packages/ocean/legoesm/ocean/advection.py:1364:        face_dx = R_planet * dlon * vface_zonal_cos_lat(grid)  # (n_lat+1,)
packages/ocean/legoesm/ocean/advection.py-1365-        _is_2d_dy = False
packages/ocean/legoesm/ocean/advection.py-1366-        _is_2d_dx = False
packages/ocean/legoesm/ocean/advection.py:1367:    area = grid.area[..., jnp.newaxis]            # (n_lat, n_lon, 1)
packages/ocean/legoesm/ocean/advection.py-1368-
packages/ocean/legoesm/ocean/advection.py-1369-    # Per-cell magnitudes of incoming / outgoing horizontal flux.
packages/ocean/legoesm/ocean/advection.py-1370-    # u-face j is the WEST face of cell j and EAST face of cell j-1.
--
packages/ocean/legoesm/ocean/advection.py-1373-    #   outgoing  = F_u_neg at WEST face (westward out) + F_u_pos at EAST face (eastward out)
packages/ocean/legoesm/ocean/advection.py-1374-    if _is_2d_dy:
packages/ocean/legoesm/ocean/advection.py-1375-        # Per-face dy weighting: west face = face_dy[:, :-1], east = face_dy[:, 1:].
packages/ocean/legoesm/ocean/advection.py:1376:        dy_w = face_dy[:, :-1, jnp.newaxis]              # (n_lat, n_lon, 1)
packages/ocean/legoesm/ocean/advection.py:1377:        dy_e = face_dy[:, 1:, jnp.newaxis]               # (n_lat, n_lon, 1)
packages/ocean/legoesm/ocean/advection.py-1378-        in_u_w  = F_u_pos[:, :-1, :] * dy_w + F_u_neg[:, 1:, :] * dy_e
packages/ocean/legoesm/ocean/advection.py-1379-        out_u_w = F_u_neg[:, :-1, :] * dy_w + F_u_pos[:, 1:, :] * dy_e
packages/ocean/legoesm/ocean/advection.py-1380-    else:
--
packages/ocean/legoesm/ocean/advection.py-1383-
packages/ocean/legoesm/ocean/advection.py-1384-    # v-face j is the SOUTH face of cell j and NORTH face of cell j-1; weighted by face_dx[j].
packages/ocean/legoesm/ocean/advection.py-1385-    if _is_2d_dx:
packages/ocean/legoesm/ocean/advection.py:1386:        dx_s = face_dx[:-1, :, jnp.newaxis]              # (n_lat, n_lon, 1)
packages/ocean/legoesm/ocean/advection.py:1387:        dx_n = face_dx[1:, :, jnp.newaxis]               # (n_lat, n_lon, 1)
packages/ocean/legoesm/ocean/advection.py-1388-    else:
packages/ocean/legoesm/ocean/advection.py-1389-        dx_s = face_dx[:-1, jnp.newaxis, jnp.newaxis]
packages/ocean/legoesm/ocean/advection.py-1390-        dx_n = face_dx[1:, jnp.newaxis, jnp.newaxis]
--
packages/ocean/legoesm/ocean/advection.py-1400-        P_out_h = (out_u * face_dy + out_v_w) / area
packages/ocean/legoesm/ocean/advection.py-1401-
packages/ocean/legoesm/ocean/advection.py-1402-    # Vertical: pad with zeros at the top / bottom (rigid lid + floor) so
packages/ocean/legoesm/ocean/advection.py:1403:    # cell-c indexing is uniform.  ad_vert_int has shape (n_lat, n_lon,
packages/ocean/legoesm/ocean/advection.py-1404-    # nlev-1) for interfaces 0..nlev-2 between cell k (above) and k+1
packages/ocean/legoesm/ocean/advection.py-1405-    # (below); F > 0 = upward.  Pad (single HLO op) instead of
packages/ocean/legoesm/ocean/advection.py-1406-    # alloc-zeros + 3-array concatenate.
--
packages/ocean/legoesm/ocean/advection.py-1434-        Q_dn, jnp.maximum(inc_out, eps), inc_out > t_grad))
packages/ocean/legoesm/ocean/advection.py-1435-
packages/ocean/legoesm/ocean/advection.py-1436-    # ---- Per-face alpha selection ----
packages/ocean/legoesm/ocean/advection.py:1437:    # u-face j: cell L = (j-1)%n_lon (west), cell R = j (east).
packages/ocean/legoesm/ocean/advection.py-1438-    # F > 0  → flow east, into R, out of L  → α = min(R+_R, R-_L).
packages/ocean/legoesm/ocean/advection.py-1439-    # F < 0  → flow west, into L, out of R  → α = min(R+_L, R-_R).
packages/ocean/legoesm/ocean/advection.py:1440:    n_lon = ad_flux_u.shape[1] - 1
packages/ocean/legoesm/ocean/advection.py:1441:    R_in_R_u = R_in                                 # (n_lat, n_lon, nlev)
packages/ocean/legoesm/ocean/advection.py-1442-    R_in_L_u = jnp.roll(R_in, 1, axis=1)
packages/ocean/legoesm/ocean/advection.py-1443-    R_out_R_u = R_out
packages/ocean/legoesm/ocean/advection.py-1444-    R_out_L_u = jnp.roll(R_out, 1, axis=1)
packages/ocean/legoesm/ocean/advection.py:1445:    ad_face_u_int = ad_flux_u[:, :n_lon, :]
packages/ocean/legoesm/ocean/advection.py-1446-    alpha_u_pos = jnp.minimum(R_in_R_u, R_out_L_u)
packages/ocean/legoesm/ocean/advection.py-1447-    alpha_u_neg = jnp.minimum(R_in_L_u, R_out_R_u)
packages/ocean/legoesm/ocean/advection.py-1448-    alpha_u_int = jnp.where(
packages/ocean/legoesm/ocean/advection.py-1449-        ad_face_u_int > 0.0, alpha_u_pos,
packages/ocean/legoesm/ocean/advection.py-1450-        jnp.where(ad_face_u_int < 0.0, alpha_u_neg, 1.0),
packages/ocean/legoesm/ocean/advection.py-1451-    )
packages/ocean/legoesm/ocean/advection.py:1452:    # Periodic wrap: face n_lon == face 0.
packages/ocean/legoesm/ocean/advection.py-1453-    alpha_u_full = jnp.concatenate(
packages/ocean/legoesm/ocean/advection.py-1454-        [alpha_u_int, alpha_u_int[:, :1, :]], axis=1,
packages/ocean/legoesm/ocean/advection.py-1455-    )
packages/ocean/legoesm/ocean/advection.py-1456-
packages/ocean/legoesm/ocean/advection.py:1457:    # v-face j (interior, 1 ≤ j ≤ n_lat-1): cell S = j-1, cell N = j.
packages/ocean/legoesm/ocean/advection.py-1458-    R_in_N_v  = R_in[1:, :, :]
packages/ocean/legoesm/ocean/advection.py-1459-    R_in_S_v  = R_in[:-1, :, :]
packages/ocean/legoesm/ocean/advection.py-1460-    R_out_N_v = R_out[1:, :, :]
--
packages/ocean/legoesm/ocean/advection.py-1611-
packages/ocean/legoesm/ocean/advection.py-1612-    Parameters
packages/ocean/legoesm/ocean/advection.py-1613-    ----------
packages/ocean/legoesm/ocean/advection.py:1614:    u : (n_lat, n_lon+1, nlev), v : (n_lat+1, n_lon, nlev) — cell-centre-depth
packages/ocean/legoesm/ocean/advection.py-1615-        face velocities (the C-grid prognostic u/v).
packages/ocean/legoesm/ocean/advection.py-1616-    dz_ref : (nlev,) reference layer thicknesses.
packages/ocean/legoesm/ocean/advection.py:1617:    u_mask : (n_lat, n_lon+1), v_mask : (n_lat+1, n_lon) — face wet masks.
packages/ocean/legoesm/ocean/advection.py-1618-
packages/ocean/legoesm/ocean/advection.py-1619-    Returns
packages/ocean/legoesm/ocean/advection.py-1620-    -------
packages/ocean/legoesm/ocean/advection.py:1621:    u_w : (n_lat, n_lon+1, M), v_w : (n_lat+1, n_lon, M) with M = nlev-1.
packages/ocean/legoesm/ocean/advection.py-1622-    """
packages/ocean/legoesm/ocean/advection.py-1623-    dz = jnp.asarray(dz_ref, dtype=u.dtype)
packages/ocean/legoesm/ocean/advection.py-1624-    dzw = 0.5 * (dz[:-1] + dz[1:])                       # (M,) Veros dzw (interior)
--
packages/ocean/legoesm/ocean/advection.py-1633-    v_w = v_w.at[..., -1].add(v[..., -1] * bottom_extra)
packages/ocean/legoesm/ocean/advection.py-1634-    u_w = u_w * u_mask[:, :, jnp.newaxis]
packages/ocean/legoesm/ocean/advection.py-1635-    v_w = v_w * v_mask[:, :, jnp.newaxis]
packages/ocean/legoesm/ocean/advection.py:1636:    # Enforce the periodic-lon wrap column (face n_lon ≡ face 0) so the
packages/ocean/legoesm/ocean/advection.py-1637-    # superbee wrap flux and the continuity divergence see the SAME face
packages/ocean/legoesm/ocean/advection.py-1638-    # velocity even if the caller's u carries a stale wrap column — the
packages/ocean/legoesm/ocean/advection.py-1639-    # column-integral conservation of the advective tendency depends on it.

exec
/usr/bin/bash -lc "nl -ba scripts/bench/bench_mpas_spmd_scaling.py | sed -n '1,360p' && printf '\\n--- latlon bench ---\\n' && nl -ba scripts/bench/bench_atm_latlon_spmd_scaling.py | sed -n '1,400p' && printf '\\n--- cpu bench parser ---\\n' && rg -n -C 4 'add_argument|latlon-2d|n-ranks|MPI|resolution|output-dir' scripts/bench/run_cpu_mpi_scaling.py | head -n 400" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 298ms:
     1	"""Strong scaling bench for the device-sharded icosahedral/MPAS (TRiSK)
     2	hydrostatic atm step (``make_voronoi_sharded_step`` — cell-partition reorder +
     3	ppermute halo).
     4	
     5	The Voronoi twin of ``bench_atm_latlon_spmd_scaling.py`` (mirrored
     6	flag-for-flag where the grids allow): the global mesh is REORDERED with
     7	``reorder_voronoi_for_sharding`` (METIS/RCB/Hilbert-SFC cell partition, ghost-
     8	padded to an even device split) so each device's contiguous ``P("device")``
     9	shard is a spatially compact cell cluster, then the SSP-RK3 step exchanges
    10	only the partition-boundary halo per stage via ``jax.lax.ppermute``.
    11	
    12	  strong: fixed subdivision level, vary n_devices -> speedup = t(1)/t(n).
    13	  (Weak scaling rides the subdivision ladder: one level = 4x the cells, so
    14	  L at 4*n_dev matches L-1 at n_dev per-device load — there is no per-device
    15	  row knob like the lat-lon benches' --nlat-per-dev.)
    16	
    17	nlev caveat (#1113): the multi-node ceiling at THIN nlev is the ppermute ROUND
    18	count (``hlo_collective_permutes``, recorded per row) x the ~0.11 ms launch
    19	floor, NOT bandwidth — a fixed per-step overhead. At the ``--nlev 8`` default it
    20	dominates (~1.34 Gcells/s wall from N=4), so the default UNDERSTATES production
    21	scalability: at ``--nlev 26`` the per-cell compute grows ~3.25x, the flat wall
    22	dissolves (~2.07+ Gcells/s, ~1.9x higher at 16 GPUs), and a size-dependent term
    23	enters. Report the production curve at production thickness; nlev=8 is the
    24	overhead-mechanism receipt, not the campaign number.
    25	
    26	Device count is fixed at process start, so each n_devices runs as a SEPARATE
    27	process; this script benches ONE n_devices and appends a JSON line.
    28	JAX_PLATFORMS=cpu with --xla_force_host_platform_device_count gives virtual
    29	CPU devices (communication-overhead characterization, NOT a real speedup).
    30	
    31	Multi-controller (route-B, ``--multicontroller``): identical contract to the
    32	lat-lon benches — every process calls ``jax.distributed.initialize`` BEFORE
    33	any other JAX use, the ("device",) mesh is built over the GLOBAL
    34	``jax.devices()``, and the existing ``make_voronoi_sharded_step`` ppermute
    35	halo + the mass-fix psum run unchanged across processes (NCCL on GPU / gloo
    36	on CPU). NO mpi4jax is armed in this mode (the documented mixed-stack
    37	deadlock hazard). Every process computes the SAME reorder host-side; under
    38	``--multicontroller`` the partition checksum is asserted equal across
    39	processes (a rank-divergent partition — e.g. one rank resolving
    40	``--partition-method auto`` to METIS and another to RCB — would silently
    41	corrupt the halo schedule).
    42	
    43	Launch (cluster, one process per GPU):
    44	  srun -n 6 python bench_mpas_spmd_scaling.py --multicontroller \
    45	      --n-devices 6 ...            # SLURM: coordinator auto-detected
    46	  mpiexec -n 6 python ... --multicontroller --coordinator host0:9876
    47	CPU smoke (single process, virtual devices):
    48	  JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=2 \
    49	  JAX_ENABLE_X64=1 python scripts/bench/bench_mpas_spmd_scaling.py \
    50	      --subdivision 3 --nlev 4 --n-devices 2 --steps 4 --parity-gate
    51	"""
    52	from __future__ import annotations
    53	
    54	import argparse
    55	import json
    56	import os
    57	import sys
    58	import time
    59	import zlib
    60	from pathlib import Path
    61	
    62	import jax
    63	import numpy as np
    64	
    65	# Repo root on the path for tests.test_cases.baroclinic_wave (the same
    66	# baroclinic-wave IC the icosahedral lanes of run_levante_gpu_scaling use).
    67	sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    68	# Bench dir for the shared metadata module (sibling-script import pattern).
    69	sys.path.insert(0, str(Path(__file__).resolve().parent))
    70	
    71	# Shared self-describing scaling metadata (anti-fake-scaling audit): merged
    72	# under rec["metadata"] so a virtual-CPU-device proxy, a gloo/TCP fabric run,
    73	# or an f32 ablation is falsifiable from the JSONL row alone.  metadata.py
    74	# imports JAX lazily, so this is safe before jax.distributed.initialize.
    75	from metadata import (  # noqa: E402
    76	    annotate_incomplete, hlo_collective_census, scaling_metadata,
    77	    tidy_throughput_fields)
    78	
    79	# SPMD full-step parity tolerances — the FLOATING-POINT RE-ASSOCIATION floor
    80	# of the sharded step (ppermute halo + mass-fix psum reduction-order change),
    81	# NOT a bug margin; a real halo/partition regression shows up orders of
    82	# magnitude above these.  Values extend the 1-step envelope of
    83	# tests/parallel/test_voronoi_sharded_equivalence.py (u/T atol 1e-6, p_s
    84	# atol 1e-1) to the smoke window; the floor grows with steps, hence the cap.
    85	MPAS_PARITY_TOLS = {  # precision -> field -> (rtol, atol)
    86	    "float64": {"u": (1.0e-5, 1.0e-5), "T": (1.0e-6, 1.0e-5),
    87	                "p_s": (1.0e-5, 1.0)},
    88	    "float32": {"u": (1.0e-3, 1.0e-3), "T": (1.0e-4, 1.0e-3),
    89	                "p_s": (1.0e-3, 50.0)},
    90	}
    91	MPAS_PARITY_MAX_STEPS = 8
    92	
    93	# Conservation gate default: with fix_mass=True the step restores the global
    94	# dry mass to the pre-step value each step, so the drift over a smoke window
    95	# is the allreduce rounding floor, not scheme drift.
    96	MASS_RTOL_DEFAULTS = {"float64": 1.0e-11, "float32": 1.0e-5}
    97	
    98	
    99	def build_model_and_state(subdivision, nlev, reorder_target, run_nd, method,
   100	                          moist=False, lloyd_iterations=50):
   101	    """Reordered+padded global mesh, MPAS PE model, baroclinic-wave IC.
   102	
   103	    ``reorder_target`` sets the PARTITION (and ghost padding) so every run
   104	    of a strong-scaling ladder times the IDENTICAL mesh; ``run_nd`` is the
   105	    device count of THIS run's mesh/model (the two differ for the
   106	    single-device reference leg of a ladder, via ``--reorder-for``).
   107	    ``moist=True`` attaches the q_v/q_c/q_r tracers (moist baroclinic
   108	    wave) so the sharded step's packed tracer halo exchange + RK tracer
   109	    advection sit on the timed/gated path.
   110	    """
   111	    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
   112	        MPASPrimitiveEquationConfig,
   113	        MPASPrimitiveEquationModel,
   114	    )
   115	    from legoesm.grids.vertical import create_sigma_coordinate
   116	    from legoesm.grids.voronoi import create_voronoi_mesh
   117	    from legoesm.parallel.mesh import create_voronoi_device_mesh
   118	    from legoesm.parallel.voronoi_partition import reorder_voronoi_for_sharding
   119	
   120	    mesh = create_voronoi_mesh(subdivision_level=subdivision,
   121	                               lloyd_iterations=lloyd_iterations)
   122	    mesh = reorder_voronoi_for_sharding(mesh, reorder_target, method=method)
   123	    if run_nd > 1 and (mesh.nCells % run_nd or mesh.nEdges % run_nd):
   124	        # Padding only guarantees divisibility for reorder_target.
   125	        raise SystemExit(
   126	            f"padded mesh (nCells={mesh.nCells}, nEdges={mesh.nEdges}) not "
   127	            f"divisible by --n-devices {run_nd}; use a ladder where every "
   128	            f"count divides --reorder-for ({reorder_target}).")
   129	    sigma = create_sigma_coordinate(nlev)
   130	    # Same recipe as the icosahedral lane of run_levante_gpu_scaling /
   131	    # tests/parallel/test_voronoi_sharded_equivalence.py: del4 hyperdiffusion,
   132	    # energy-conserving PV flux, SSP-RK3, global mass fixer.
   133	    cfg = MPASPrimitiveEquationConfig(
   134	        nu_del4=1e16, nu_del4_ps=1e16, fix_mass=True,
   135	        pv_scheme="energy", time_integrator="ssp_rk3",
   136	    )
   137	    dev_config = create_voronoi_device_mesh(
   138	        nCells=mesh.nCells, nEdges=mesh.nEdges, nVertices=mesh.nVertices,
   139	        n_devices=run_nd,
   140	    )
   141	    if dev_config.n_devices > 1:
   142	        from legoesm.parallel.mesh import replicate_pytree
   143	        mesh_model = replicate_pytree(mesh, dev_config)
   144	    else:
   145	        mesh_model = mesh
   146	    model = MPASPrimitiveEquationModel(mesh_model, sigma, cfg)
   147	    # #1100 MPAS twin: the timed path never global-builds the state.
   148	    # build_sharded_baroclinic_wave_state_mpas creates every leaf via
   149	    # jax.make_array_from_callback (only THIS process's shard rows are
   150	    # ever materialised; value-identical (few-ULP contract, measured
   151	    # exact on the pinned CPU stack) to global-build + shard_pytree —
   152	    # tests/parallel/test_mpas_partitionlocal_build.py).  The GLOBAL
   153	    # state is built lazily in main() only for the parity/conservation
   154	    # gates (small smoke scales).  The mesh itself is still global per
   155	    # process — its SFC-partition-local construction is the open
   156	    # remainder of #1100.
   157	    from tests.test_cases.baroclinic_wave import (
   158	        build_sharded_baroclinic_wave_state_mpas,
   159	    )
   160	    state_sharded = build_sharded_baroclinic_wave_state_mpas(
   161	        mesh, sigma, dev_config, perturbed=True, moist=moist)
   162	    return mesh, model, state_sharded, dev_config
   163	
   164	
   165	def _block(state):
   166	    jax.block_until_ready([leaf for leaf in jax.tree.leaves(state)
   167	                           if leaf is not None])
   168	
   169	
   170	def _global_dry_mass(state, mesh):
   171	    """sum(p_s * areaCell) on host arrays — the quantity fix_mass pins."""
   172	    ps = np.asarray(state.p_s.data)
   173	    area = np.asarray(mesh.areaCell)
   174	    return float(np.sum(ps * area))
   175	
   176	
   177	def main() -> int:
   178	    p = argparse.ArgumentParser()
   179	    p.add_argument("--subdivision", type=int, default=5,
   180	                   help="icosahedral subdivision level L "
   181	                        "(nCells = 10*4^L + 2 before ghost padding)")
   182	    p.add_argument("--nlev", type=int, default=8)
   183	    p.add_argument("--lloyd", type=int, default=50,
   184	                   help="Lloyd relaxation iterations for the mesh. 50 = "
   185	                        "production SCVT; 0 = labelled synthetic scaling "
   186	                        "mesh (scaling receipts only, never physics — "
   187	                        "must match the prewarmed cache key at subdiv>=9).")
   188	    p.add_argument("--n-devices", type=int, required=True)
   189	    p.add_argument("--reorder-for", type=int, default=None,
   190	                   help="partition/reorder the mesh for THIS device count "
   191	                        "(default: --n-devices). Pin it to the ladder's "
   192	                        "max so single-device reference runs time the "
   193	                        "identical reordered mesh.")
   194	    p.add_argument("--partition-method",
   195	                   choices=["auto", "geometric", "metis", "sfc"],
   196	                   default="auto")
   197	    p.add_argument("--physics", choices=["none", "held_suarez", "kessler"],
   198	                   default="none",
   199	                   help="Operator-split physics on the timed path. "
   200	                        "'kessler' also attaches the q_v/q_c/q_r moist-"
   201	                        "baroclinic-wave tracers (packed tracer halo "
   202	                        "exchange + RK tracer advection on the gated "
   203	                        "path) and extends the parity gate to the "
   204	                        "tracer fields.")
   205	    p.add_argument("--halo-strategy",
   206	                   choices=["auto", "ppermute", "allgather"],
   207	                   default="auto",
   208	                   help="Halo strategy for make_voronoi_sharded_step. "
   209	                        "'auto' picks allgather below the per-device "
   210	                        "cell threshold — force 'ppermute' to exercise "
   211	                        "the neighbor-round schedule on small gate "
   212	                        "meshes (the multicontroller selfspawn tests "
   213	                        "do).  Recorded in the JSONL row.")
   214	    p.add_argument("--steps", type=int, default=12)
   215	    p.add_argument("--warmup", type=int, default=2)
   216	    p.add_argument("--dt", type=float, default=None,
   217	                   help="timestep [s]; default auto: 600 * 4**(4-L) "
   218	                        "(CFL: dx halves per level), min 30 s.")
   219	    p.add_argument("--out", type=str,
   220	                   default="results/a1/mpas_spmd_scaling.jsonl")
   221	    p.add_argument(
   222	        "--parity-gate", action="store_true",
   223	        help="Correctness gate: compare the gathered sharded trajectory "
   224	             "against the single-device model.step trajectory on the SAME "
   225	             "reordered mesh (smoke windows only; the re-association floor "
   226	             "grows with steps).")
   227	    p.add_argument(
   228	        "--check-conservation", action="store_true",
   229	        help="Gate global dry-mass drift sum(p_s*areaCell) over the run "
   230	             "(pre-shard state vs gathered final state; exits nonzero on "
   231	             "breach).")
   232	    p.add_argument("--mass-rtol", type=float, default=None,
   233	                   help="Conservation tolerance (default: 1e-11 f64 / "
   234	                        "1e-5 f32 — fix_mass pins the mass each step).")
   235	    p.add_argument("--multicontroller", action="store_true",
   236	                   help="Route-B multi-controller: jax.distributed.initialize "
   237	                        "per process, ('device',) mesh over the GLOBAL device "
   238	                        "set (one process per GPU / per CPU-device group). NO "
   239	                        "mpi4jax. --n-devices must equal the global device "
   240	                        "count.")
   241	    p.add_argument("--coordinator", type=str, default=None,
   242	                   help="host:port for jax.distributed when auto-detection "
   243	                        "(SLURM) is unavailable; process count/id then come "
   244	                        "from OMPI_COMM_WORLD_SIZE/RANK.")
   245	    args = p.parse_args()
   246	
   247	    # Validate the timing window BEFORE any model/device work (codex, ocean
   248	    # twin): an empty steady slice would only fail after the expensive run.
   249	    if args.steps < 1:
   250	        raise SystemExit(f"--steps must be >= 1, got {args.steps}")
   251	    if not (0 <= args.warmup < args.steps):
   252	        raise SystemExit(
   253	            f"--warmup must satisfy 0 <= warmup < steps "
   254	            f"(got warmup={args.warmup}, steps={args.steps})")
   255	    if args.parity_gate and args.steps > MPAS_PARITY_MAX_STEPS:
   256	        raise SystemExit(
   257	            f"--parity-gate is a smoke gate (re-association floor grows "
   258	            f"with steps); --steps {args.steps} > {MPAS_PARITY_MAX_STEPS} "
   259	            f"cap.")
   260	
   261	    if args.multicontroller:
   262	        # MUST run before any other JAX use (backend init). SLURM auto-detects;
   263	        # mpiexec needs the explicit coordinator + launcher env vars (OpenMPI
   264	        # OMPI_*, or Cray PALS PMI_* on Derecho).
   265	        if args.coordinator is not None:
   266	            n_procs = int(os.environ.get(
   267	                "OMPI_COMM_WORLD_SIZE", os.environ.get("PMI_SIZE", "0")))
   268	            proc_id = int(os.environ.get(
   269	                "OMPI_COMM_WORLD_RANK", os.environ.get("PMI_RANK", "-1")))
   270	            if n_procs < 1 or proc_id < 0:
   271	                raise SystemExit(
   272	                    "--coordinator given but no launcher rank env found "
   273	                    "(OMPI_COMM_WORLD_SIZE/RANK or PMI_SIZE/PMI_RANK).")
   274	            jax.distributed.initialize(
   275	                coordinator_address=args.coordinator,
   276	                num_processes=n_procs, process_id=proc_id)
   277	        else:
   278	            # Environment-routed: SLURM/OMPI -> bare auto-detect; PALS/PMI
   279	            # (Derecho mpiexec) -> mpi4py bootstrap. Real init failures
   280	            # re-raise loudly.
   281	            from legoesm.parallel.early_init import (
   282	                init_jax_distributed_with_fallback,
   283	            )
   284	            init_jax_distributed_with_fallback()
   285	
   286	    from legoesm.parallel.sharded_dynamics import (
   287	        gather_voronoi_state_spmd,
   288	        make_voronoi_sharded_step,
   289	    )
   290	
   291	    nd = args.n_devices
   292	    avail = len(jax.devices())
   293	    if avail < nd:
   294	        raise SystemExit(f"need {nd} devices, have {avail} "
   295	                         f"(set --xla_force_host_platform_device_count)")
   296	    if args.multicontroller and nd != avail:
   297	        # A mesh over a strict subset would leave some processes' devices out
   298	        # of the program (non-addressable participation hazard). Route-B uses
   299	        # ALL global devices: one shard per device across every process.
   300	        raise SystemExit(
   301	            f"--multicontroller: --n-devices ({nd}) must equal the GLOBAL "
   302	            f"device count ({avail} across {jax.process_count()} processes).")
   303	
   304	    dt = args.dt
   305	    if dt is None:
   306	        dt = max(600.0 * 4.0 ** (4 - args.subdivision), 30.0)
   307	
   308	    reorder_for = args.reorder_for if args.reorder_for is not None else nd
   309	    if reorder_for < nd:
   310	        raise SystemExit(
   311	            f"--reorder-for ({reorder_for}) must be >= --n-devices ({nd}): "
   312	            f"the ghost padding only guarantees divisibility for the "
   313	            f"partition target.")
   314	    mesh, model, s0, dev_config = build_model_and_state(
   315	        args.subdivision, args.nlev, reorder_for, nd, args.partition_method,
   316	        moist=(args.physics == "kessler"), lloyd_iterations=args.lloyd)
   317	
   318	    if args.multicontroller:
   319	        # Every process computed the reorder independently — assert the
   320	        # partitions agree before any collective uses the halo schedule.
   321	        # The checksum covers the entity ORDER (coordinates) and the
   322	        # connectivity the ppermute schedule + TRiSK stencils read; a
   323	        # rank-divergent partition (e.g. one rank resolving
   324	        # --partition-method auto to METIS, another to RCB) cannot slip
   325	        # through on cell positions alone.
   326	        from jax.experimental import multihost_utils
   327	        crc = 0
   328	        for arr, dtype in (
   329	            (mesh.latCell, np.float64), (mesh.latEdge, np.float64),
   330	            (mesh.cellsOnEdge, np.int64), (mesh.edgesOnCell, np.int64),
   331	            (mesh.cellsOnCell, np.int64), (mesh.areaCell, np.float64),
   332	        ):
   333	            crc = zlib.crc32(np.ascontiguousarray(
   334	                np.asarray(arr, dtype=dtype)).tobytes(), crc)
   335	        crc = zlib.crc32(
   336	            np.asarray([mesh.nCells, mesh.nEdges, mesh.nVertices],
   337	                       dtype=np.int64).tobytes(), crc)
   338	        multihost_utils.assert_equal(
   339	            np.uint32(crc),
   340	            fail_message="partition/reorder checksum differs across "
   341	                         "processes (rank-divergent --partition-method "
   342	                         "resolution?)")
   343	
   344	    physics_fn = None
   345	    if args.physics == "held_suarez":
   346	        from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_forcing_mpas
   347	        physics_fn = held_suarez_forcing_mpas
   348	    elif args.physics == "kessler":
   349	        # Warm-rain microphysics over the moist BCW tracers.  Kessler's
   350	        # saturation adjustment is a rate over the dt bound HERE, so it
   351	        # must match the stepping dt (make_kessler_forcing_mpas contract).
   352	        from legoesm.atmosphere.forcing.idealized.kessler_forcing import (
   353	            make_kessler_forcing_mpas,
   354	        )
   355	        physics_fn = make_kessler_forcing_mpas(dt)
   356	
   357	    # #1100: s0 from build_model_and_state is ALREADY partition-local-sharded
   358	    # when n_devices > 1 (no per-process global build on the timed path).
   359	    # The parity/conservation gates are the ONLY consumers of a global
   360	    # initial state — build it lazily here, at their smoke scales only

--- latlon bench ---
     1	"""Strong/weak scaling bench for the lat-band SPMD lat-lon C-grid hydrostatic
     2	atm step (make_sharded_atm_latlon_step / run_atm_latlon_spmd, the A1 work).
     3	
     4	Times the SHARDED step across an N-device ("lat",) mesh and reports per-step
     5	wall time + speedup vs 1 device.  The DEFAULT lane follows the M1 measurement
     6	contract (``metadata.timed_scan_blocks``): fused ``lax.scan`` blocks of
     7	``--steps`` steps with device sync only AROUND each block (``fused_step_ms``,
     8	slowest process across controllers) plus a SEPARATE individually-synced
     9	dispatch-latency probe (``step_latency_ms``) — never mixed.  The
    10	jit(shard_map) step is built once and cached by make_sharded_atm_latlon_step;
    11	re-tracing would show up as every block paying the scan-compile cost again.
    12	
    13	  strong: fixed (n_lat, n_lon, nlev), vary n_devices -> speedup = t(1)/t(n).
    14	  weak:   n_lat = nlat_per_dev * n_devices (fixed per-device rows) -> ideal flat.
    15	
    16	``--segment-steps N`` (M2b): times the COMPILED-SEGMENT lane instead — ONE
    17	jitted lax.scan of N sharded steps per block (make_sharded_atm_latlon_segment,
    18	band-SHARDED geometry, in-graph finite scalar), so --steps counts BLOCKS of N
    19	steps and the per-step numbers derive from whole-block wall times.  Unlike the
    20	default lane (which scans the bench-local step fn), the segment is the
    21	PRODUCTION artifact; the record carries ``segment_mode=true`` +
    22	``per_block_ms`` and the per-device geometry bytes (replicated vs
    23	band-sharded) computed from the real band-grid shapes.
    24	
    25	Receipt honesty: a segment whose in-graph finite scalar reports a non-finite
    26	state STOPS the timed loop and stamps the record ``finite_ok=false`` +
    27	``valid=false`` (with ``completed_blocks`` saying how far it got, a
    28	``diverged: ...`` entry under ``metadata._incomplete``, and nulled
    29	``sypd``/``mcells_per_s``) — a diverging trajectory is never serialized as
    30	valid scaling data.  The default fused lane has no in-graph finite check, so
    31	its rows carry ``finite_ok=null``.
    32	
    33	Device count is fixed at process start, so each n_devices runs as a SEPARATE
    34	process (one sbatch step per count); this script benches ONE n_devices and
    35	appends a JSON line. JAX_PLATFORMS=cpu with --xla_force_host_platform_device_count
    36	gives virtual CPU devices (communication-overhead characterization, NOT a real
    37	speedup); a real number needs one GPU per band.
    38	
    39	Multi-controller (route-B, ``--multicontroller``): the lat-lon analogue of the
    40	cubed-sphere ``run_cpu_mpi_scaling --cs-spmd`` A1 path. Every process calls
    41	``jax.distributed.initialize`` BEFORE any other JAX use, the ("lat",) mesh is
    42	built over the GLOBAL ``jax.devices()`` (all processes), and the existing
    43	``make_sharded_atm_latlon_step`` + band-ppermute halo runs unchanged — the
    44	ppermute/psum collectives cross processes via the distributed runtime (NCCL on
    45	GPU / gloo on CPU). NO mpi4jax is armed in this mode (mixing the mpi4jax halo
    46	machinery with jax.distributed collectives in one program is the documented
    47	mixed-stack deadlock hazard — see run_cpu_mpi_scaling._build_cubed_sphere_spmd).
    48	This is the halo path that keeps intra-node GPU traffic on NCCL and bypasses
    49	the host-staged / CXI-inject-broken cross-node GPU-direct MPI route (see
    50	docs/performance/multinode_gpu_direct_cxi.md).
    51	
    52	Launch (cluster, one process per GPU):
    53	  srun -n 8 python bench_atm_latlon_spmd_scaling.py --multicontroller \
    54	      --n-devices 8 ...            # SLURM: coordinator auto-detected
    55	  mpiexec -n 8 python ... --multicontroller --coordinator host0:9876
    56	"""
    57	from __future__ import annotations
    58	
    59	import argparse
    60	import json
    61	import os
    62	import time
    63	
    64	import sys
    65	from pathlib import Path
    66	
    67	import jax
    68	import jax.numpy as jnp
    69	import numpy as np
    70	
    71	# Bench dir for the shared metadata module (sibling-script import pattern —
    72	# needed when this file is loaded by path from tests, not run as a script).
    73	sys.path.insert(0, str(Path(__file__).resolve().parent))
    74	
    75	# Shared self-describing scaling metadata (anti-fake-scaling audit): merged
    76	# under rec["metadata"] so a virtual-CPU-device proxy, a gloo/TCP fabric run,
    77	# or an f32 ablation is falsifiable from the JSONL row alone.  metadata.py
    78	# imports JAX lazily, so this is safe before jax.distributed.initialize.
    79	from metadata import (  # noqa: E402
    80	    annotate_incomplete,
    81	    calibrated_bound,
    82	    comm_accounting,
    83	    scaling_metadata,
    84	    tidy_throughput_fields,
    85	)
    86	
    87	
    88	def _build_model(n_lat, n_lon, nlev):
    89	    from legoesm import constants
    90	    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
    91	        CGridLatLonPrimitiveEquationConfig, CGridLatLonPrimitiveEquationModel)
    92	    from legoesm.grids.latlon import create_latlon_grid
    93	    from legoesm.grids.vertical import create_sigma_coordinate
    94	
    95	    grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon, radius=constants.R_earth,
    96	                              omega=constants.Omega)
    97	    sigma = create_sigma_coordinate(n_levels=nlev)
    98	    cfg = CGridLatLonPrimitiveEquationConfig(
    99	        fix_mass=True, use_polar_filter=False, use_ppm_transport=True,
   100	        time_integrator="ssp_rk3")
   101	    return CGridLatLonPrimitiveEquationModel(grid, sigma, cfg)
   102	
   103	
   104	def _build(n_lat, n_lon, nlev):
   105	    # nd=1 lane + tests: global (unsharded) IC build, unchanged protocol.
   106	    from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init_latlon
   107	    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
   108	        hydrostatic_to_cgrid)
   109	
   110	    model = _build_model(n_lat, n_lon, nlev)
   111	    hs0 = held_suarez_init_latlon(model.grid, model.sigma_coord)
   112	    c0 = hydrostatic_to_cgrid(hs0, model.grid)
   113	    return model, c0
   114	
   115	
   116	def _block(state):
   117	    jax.block_until_ready(jax.tree.leaves(state))
   118	
   119	
   120	def main() -> int:
   121	    p = argparse.ArgumentParser()
   122	    p.add_argument("--n-lat", type=int, default=128)
   123	    p.add_argument("--n-lon", type=int, default=256)
   124	    p.add_argument("--nlev", type=int, default=30)
   125	    p.add_argument("--n-devices", type=int, required=True)
   126	    p.add_argument("--mode", choices=["strong", "weak"], default="strong")
   127	    p.add_argument("--nlat-per-dev", type=int, default=32,
   128	                   help="weak mode: lat rows per device")
   129	    p.add_argument("--steps", type=int, default=12,
   130	                   help="Default lane: steps per fused lax.scan timing "
   131	                        "block. Segment mode: number of timed BLOCKS of "
   132	                        "--segment-steps steps each.")
   133	    p.add_argument("--warmup", type=int, default=2,
   134	                   help="Segment mode: timed blocks dropped from steady "
   135	                        "stats (block 0 includes the scan compile). Default "
   136	                        "lane: retained for CLI compat (fused-block timing "
   137	                        "separates compile/probe/blocks explicitly).")
   138	    p.add_argument("--blocks", type=int, default=2,
   139	                   help="Default lane: timed fused blocks (per-block times "
   140	                        "expose drift).")
   141	    p.add_argument("--probe-steps", type=int, default=3,
   142	                   help="Default lane: individually-synced steps for the "
   143	                        "SEPARATE dispatch-latency probe (step_latency_ms).")
   144	    p.add_argument("--segment-steps", type=int, default=0,
   145	                   help="M2b: >0 compiles ONE lax.scan segment of this many "
   146	                        "steps (built once, reused; band-sharded geometry) "
   147	                        "and times BLOCKS of segment calls instead of "
   148	                        "per-step host dispatch. 0 = default fused lane.")
   149	    p.add_argument("--physics", choices=["none", "held_suarez"], default="none")
   150	    p.add_argument("--dt", type=float, default=60.0)
   151	    p.add_argument("--single-dev-fused-ms", type=float, default=None,
   152	                   help="fused_step_ms of the nd=1 row at the SAME per-device "
   153	                        "size (compute ingredient of the calibrated T_bound, "
   154	                        "audit item 8). Omitted at nd>1 -> bound emitted null "
   155	                        "+ flagged incomplete; nd=1 uses its own measurement.")
   156	    p.add_argument("--comm-latency-us", type=float, default=None,
   157	                   help="MEASURED per-message latency [us] of THIS machine's "
   158	                        "fabric. Default: MACHINE-CALIBRATED-REQUIRED "
   159	                        "placeholder in metadata.py -> bound_calibrated=false.")
   160	    p.add_argument("--comm-bandwidth-gbs", type=float, default=None,
   161	                   help="MEASURED link bandwidth [GB/s] of THIS machine's "
   162	                        "fabric. Default: MACHINE-CALIBRATED-REQUIRED "
   163	                        "placeholder in metadata.py -> bound_calibrated=false.")
   164	    p.add_argument("--out", type=str, default="results/a1/spmd_scaling.jsonl")
   165	    p.add_argument("--multicontroller", action="store_true",
   166	                   help="Route-B multi-controller: jax.distributed.initialize "
   167	                        "per process, ('lat',) mesh over the GLOBAL device set "
   168	                        "(one process per GPU / per CPU-device group). NO "
   169	                        "mpi4jax. --n-devices must equal the global device "
   170	                        "count.")
   171	    p.add_argument("--coordinator", type=str, default=None,
   172	                   help="host:port for jax.distributed when auto-detection "
   173	                        "(SLURM) is unavailable; process count/id then come "
   174	                        "from OMPI_COMM_WORLD_SIZE/RANK.")
   175	    args = p.parse_args()
   176	
   177	    # Validate the schedule BEFORE any model/device work: a zero/negative
   178	    # --steps would otherwise surface only as timed_scan_blocks' None
   179	    # headline (default lane) or an empty timed loop (segment mode) after
   180	    # the expensive build (the ocean twin's guard).
   181	    if args.steps < 1:
   182	        raise SystemExit(f"--steps must be >= 1, got {args.steps}")
   183	
   184	    if args.multicontroller:
   185	        # MUST run before any other JAX use (backend init).  The SHARED
   186	        # helper owns the launcher-env contract (SLURM/OMPI auto-detect,
   187	        # PALS mpi4py bootstrap, explicit-coordinator path) AND the
   188	        # hardening: post-init silent-fallback guard + NCCL net-plugin
   189	        # warning — an inline init here would bypass both (codex).
   190	        from legoesm.parallel.early_init import (
   191	            init_multicontroller_distributed,
   192	        )
   193	        init_multicontroller_distributed(args.coordinator)
   194	
   195	    from legoesm.atmosphere.dynamics.gcm.sharded_atm_latlon_step import (
   196	        atm_latlon_geometry_bytes,
   197	        build_sharded_held_suarez_state_atm_latlon,
   198	        make_sharded_atm_latlon_segment,
   199	        make_sharded_atm_latlon_step)
   200	    seg_n = int(args.segment_steps)
   201	    if seg_n < 0:
   202	        raise SystemExit(f"--segment-steps must be >= 0, got {seg_n}")
   203	    physics_fn = None
   204	    if args.physics == "held_suarez":
   205	        from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_forcing_latlon
   206	        physics_fn = held_suarez_forcing_latlon
   207	
   208	    nd = args.n_devices
   209	    avail = len(jax.devices())
   210	    if avail < nd:
   211	        raise SystemExit(f"need {nd} devices, have {avail} "
   212	                         f"(set --xla_force_host_platform_device_count)")
   213	    if args.multicontroller and nd != avail:
   214	        # A mesh over a strict subset would leave some processes' devices out
   215	        # of the program (non-addressable participation hazard). Route-B uses
   216	        # ALL global devices: one band per device across every process.
   217	        raise SystemExit(
   218	            f"--multicontroller: --n-devices ({nd}) must equal the GLOBAL "
   219	            f"device count ({avail} across {jax.process_count()} processes).")
   220	    n_lat = args.n_lat if args.mode == "strong" else args.nlat_per_dev * nd
   221	    if n_lat % nd != 0:
   222	        raise SystemExit(f"n_lat {n_lat} not divisible by n_devices {nd}")
   223	
   224	    if nd == 1:
   225	        model, c0 = _build(n_lat, args.n_lon, args.nlev)
   226	        mesh = None
   227	        c = c0
   228	    else:
   229	        # #1100: band-local IC construction. The nd>1 lanes never materialise
   230	        # the global (n_lat, n_lon, nlev) state per process — each leaf is
   231	        # created via make_array_from_callback for the rows this process's
   232	        # devices own (no global build, no device_put replication, no
   233	        # assert_equal all-gather). This is what lets full-node-packed CPU
   234	        # rungs (128 procs/node) survive at large n_lat.
   235	        model = _build_model(n_lat, args.n_lon, args.nlev)
   236	        mesh = jax.sharding.Mesh(np.array(jax.devices()[:nd]),
   237	                                 axis_names=("lat",))
   238	        c = build_sharded_held_suarez_state_atm_latlon(
   239	            model.grid, model.sigma_coord, mesh)
   240	    if seg_n > 0:
   241	        seg_fn = make_sharded_atm_latlon_segment(
   242	            model, mesh, seg_n, physics_fn=physics_fn)
   243	    else:
   244	        step = make_sharded_atm_latlon_step(model, mesh,
   245	                                            physics_fn=physics_fn)
   246	
   247	    per_block_ms = None
   248	    completed_blocks = None
   249	    finite_ok = None   # default fused lane: no in-graph finite check -> null
   250	    timing = None   # metadata.timed_scan_blocks metrics (default lane only)
   251	    if seg_n > 0:
   252	        # Multi-controller: align every process before the timed loop so
   253	        # block wall times aren't skewed by startup jitter (and once after,
   254	        # so no process exits while peers still hold collectives in flight).
   255	        # (The default lane's fences live inside timed_scan_blocks.)
   256	        if jax.process_count() > 1:
   257	            from jax.experimental import multihost_utils
   258	            multihost_utils.sync_global_devices("atm_latlon_spmd_bench_start")
   259	        # Segment mode: each timed BLOCK is one compiled lax.scan of seg_n
   260	        # steps; the host sync per block is the production pattern — read the
   261	        # in-graph finite SCALAR, then block on the state for honest timing.
   262	        per_block_ms = []
   263	        finite_ok = True
   264	        for i in range(args.steps):
   265	            t0 = time.perf_counter()
   266	            c, ok = seg_fn(c, args.dt)
   267	            ok_b = bool(ok)
   268	            _block(c)
   269	            per_block_ms.append((time.perf_counter() - t0) * 1e3)
   270	            if not ok_b:
   271	                # A non-finite state poisons every later block: stop timing
   272	                # and mark the whole record invalid — a warning alone let a
   273	                # diverging trajectory serialize as valid scaling data, and
   274	                # an early false was even forgotten by later true blocks
   275	                # (codex batch4).
   276	                finite_ok = False
   277	                print(f"[warn] segment finite scalar FALSE after block {i} "
   278	                      f"(step {(i + 1) * seg_n}) — stopping the timed loop; "
   279	                      "the record is marked INVALID (finite_ok=false, "
   280	                      "valid=false) and its throughput fields are nulled")
   281	                break
   282	        completed_blocks = len(per_block_ms)
   283	        per_step_ms = [b / seg_n for b in per_block_ms]
   284	        if jax.process_count() > 1:
   285	            from jax.experimental import multihost_utils
   286	            multihost_utils.sync_global_devices("atm_latlon_spmd_bench_end")
   287	        steady = per_step_ms[args.warmup:]
   288	        if not steady:
   289	            # Divergence stopped the run inside the warmup window — fall back
   290	            # to every completed unit (the record is already marked invalid;
   291	            # this only keeps the diagnostic median well-defined).
   292	            steady = per_step_ms
   293	        med = float(np.median(steady))
   294	    else:
   295	        # Measurement contract (scaling audit gaps #1/#2): fused ``lax.scan``
   296	        # blocks with sync only AROUND the block — the previous per-step
   297	        # host-synced loop measured dispatch+sync latency, not fused device
   298	        # throughput.  Dispatch latency stays measured SEPARATELY
   299	        # (``step_latency_ms``); multi-controller runs record the
   300	        # slowest-process block time + imbalance ratio.
   301	        from metadata import timed_scan_blocks
   302	        c, timing = timed_scan_blocks(
   303	            lambda st: step(st, args.dt), c,
   304	            block_steps=args.steps, n_blocks=args.blocks,
   305	            probe_steps=args.probe_steps,
   306	            sync_label="atm_latlon_spmd_bench")
   307	        # Headline = fused per-step time from the SLOWEST process; key name
   308	        # kept for the aggregators.
   309	        med = float(timing["fused_step_ms"])
   310	
   311	    # valid=false ONLY on an observed non-finite state; the default fused
   312	    # lane (finite_ok=None: unchecked) stays valid.
   313	    valid = finite_ok is not False
   314	    # A diverging segment run's med is not a measurement: feed the bound
   315	    # honest nulls (its flat throughput twins are nulled after assembly).
   316	    _measured_med = med if valid else None
   317	    # Honest per-device geometry residency (from the real band-grid shapes):
   318	    # the default lane replicates all-band stacks; the segment lane shards.
   319	    geom_bytes = (atm_latlon_geometry_bytes(model.grid, nd) if nd > 1
   320	                  else None)
   321	
   322	    # Communication accounting (audit item 4) + calibrated T_bound (item 8).
   323	    # nd=1: zero inter-device traffic is a FACT (recorded as 0), so the
   324	    # bound is complete and trivially equals the measured compute.  nd>1:
   325	    # there is no analytic halo-message census for the atm latlon step yet
   326	    # (the ocean twin derives one from its barotropic solver) — the comm
   327	    # ingredients are recorded null with this reason and the bound is
   328	    # emitted incomplete rather than fabricated.
   329	    if nd <= 1:
   330	        _msgs, _bytes_msg, _nred = 0, 0, 0
   331	        _bytes_lower = False   # zero traffic is exact, not an undercount
   332	        _comm_note = "single device: no inter-device halo/reduction traffic"
   333	    else:
   334	        _msgs, _bytes_msg, _nred = None, None, None
   335	        _bytes_lower = None
   336	        _comm_note = ("no analytic halo-message census for the atm latlon "
   337	                      "step yet (audit item 4 follow-up) — comm fields null, "
   338	                      "not fabricated")
   339	    comm_rec = comm_accounting(
   340	        halo_messages_per_step=_msgs,
   341	        bytes_per_message=_bytes_msg,
   342	        full_state_gathers_per_step=0,   # fused scan/segment: no per-step gather
   343	        scope_note=_comm_note,
   344	        bytes_are_lower_bound=_bytes_lower,
   345	    )
   346	    bound_rec = calibrated_bound(
   347	        measured_fused_step_ms=_measured_med,
   348	        # Invalid rows feed the bound NOTHING: even the CLI-provided nd=1
   349	        # baseline is withheld so bound_ingredients.compute_ms cannot dress
   350	        # a diverging row up as a modelled one (codex).
   351	        single_device_fused_step_ms=(
   352	            _measured_med if nd == 1
   353	            else (args.single_dev_fused_ms if valid else None)),
   354	        halo_messages_per_step=comm_rec["halo_messages_per_step"],
   355	        halo_bytes_per_step=comm_rec["halo_bytes_per_step"],
   356	        n_reductions_per_step=_nred,
   357	        # rank imbalance is measured by timed_scan_blocks (default lane);
   358	        # the segment lane records no cross-process block gather -> null.
   359	        rank_imbalance=(float(timing["rank_imbalance"])
   360	                        if timing is not None else None),
   361	        latency_us=args.comm_latency_us,
   362	        bandwidth_GBs=args.comm_bandwidth_gbs,
   363	    )
   364	
   365	    rec = dict(
   366	        mode=args.mode, n_devices=nd, n_lat=n_lat, n_lon=args.n_lon,
   367	        nlev=args.nlev, physics=args.physics, steps=args.steps,
   368	        platform=jax.default_backend(),
   369	        n_processes=jax.process_count(),
   370	        multicontroller=bool(args.multicontroller),
   371	        segment_mode=(seg_n > 0),
   372	        segment_steps=(seg_n if seg_n > 0 else None),
   373	        # Measurement validity (codex batch4): finite_ok is the ACCUMULATED
   374	        # in-graph finite verdict (null in the unchecked default fused lane);
   375	        # valid=false marks the row as NOT scaling data; completed_blocks
   376	        # says where a diverging segment run stopped.
   377	        finite_ok=finite_ok,
   378	        valid=valid,
   379	        completed_blocks=completed_blocks,
   380	        steady_median_ms=round(med, 4),
   381	        cells=n_lat * args.n_lon * args.nlev,
   382	    )
   383	    if seg_n > 0:
   384	        # Segment lane: unit 0 = the first BLOCK (includes the scan
   385	        # compile); per-step numbers derive from whole blocks.
   386	        rec.update(
   387	            compile_ms=round(per_block_ms[0], 1),
   388	            steady_min_ms=round(float(np.min(steady)), 2),
   389	            per_step_ms=[round(x, 2) for x in per_step_ms],
   390	            per_block_ms=[round(x, 2) for x in per_block_ms],
   391	        )
   392	    else:
   393	        # Default lane: the timed_scan_blocks metrics (fused_step_ms,
   394	        # step_latency_ms, block_ms, parallel_block_ms, rank_imbalance, ...
   395	        # — the M1 measurement contract).
   396	        rec.update(**timing)
   397	    # Flat aggregator-compatible identity + metric fields: without a
   398	    # top-level ``sypd``/``grid_type`` this lane's rows are invisible to
   399	    # aggregate_bcw_scaling.py → empty SYPD panels in the CPU-vs-GPU plots.
   400	    rec.update(

--- cpu bench parser ---
1-#!/usr/bin/env python
2:"""CPU MPI scaling benchmark for AMIP-like runs.
3-
4:Measures wall-clock time per step and SYPD across varying MPI rank counts
5:and resolutions for the supported grid+physics combinations.
6-
7:MPI-scalable grids (multi-rank weak/strong scaling, genuinely
8-domain-decomposed at the dycore level):
9-  icosahedral   -- MPAS Voronoi TRiSK PE dycore (cell partition)
10-  latlon        -- Lat-lon C-grid FV PE dycore (latitude-band
11-                   decomposition via ``make_latlon_mpi_step``;
12-                   needs >=2 lat rows per rank for the halo=2
13-                   PPM/biharmonic exchanges)
14-
15:Single-rank only (listed but their MPI paths are not domain-decomposed
16-at the dycore level — see iter 13/14 honest-sweep guards):
17-  cubed-sphere  -- C-D grid + FV3 PE dycore (replicated dynamics
18:                   under MPI; iter 3 added scattered halo support but
19-                   driver-side state scatter is not yet implemented)
20:  spectral      -- Gaussian + spectral PE dycore (no MPI path at all)
21-
22-Physics levels:
23-  held_suarez   -- Newtonian relaxation (cheapest, no I/O)
24-  gray_sbm      -- Gray radiation + SBM convection
25-  rrtmg_full    -- RRTMG radiation + Kessler microphysics + SBM
26-
27:Design: one benchmark case per MPI process invocation (see F1 in plan).
28-Use --sweep to generate all cases for a SLURM array job.
29-
30-Usage
31------
32:Multi-rank MPI scaling (icosahedral or latlon)::
33-
34-    mpirun -np 8 python scripts/run_cpu_mpi_scaling.py \\
35-        --grid icosahedral --mode strong --physics held_suarez
36-
--
39-
40-Single-rank case (any grid)::
41-
42-    python scripts/run_cpu_mpi_scaling.py \\
43:        --grid cubed-sphere --resolution 48 --physics held_suarez
44-
45-Sweep mode (generate case list, no execution)::
46-
47-    python scripts/run_cpu_mpi_scaling.py --sweep \\
--
90-    # cores).  ==1 (the default packing, one rank per core) => force
91-    # single-threaded Eigen so packed ranks never oversubscribe.  >1 (hybrid:
92-    # fewer ranks x more cores/rank) => let Eigen multi-thread so each rank uses
93-    # its allocated cores -- fewer ranks means fewer halo messages, the codex
94:    # MPI-improve lever, without idling cores.
95-    #
96-    # SLURM sets SLURM_CPUS_PER_TASK; PBS/PALS (Derecho route-B) does NOT — it
97-    # exports the per-rank thread count as OMP_NUM_THREADS (see
98-    # cube_scaling_cpu_routeb.sh).  Fall back to it (matching write_result_json's
--
109-        os.environ["XLA_FLAGS"] = xla_flags
110-
111-
112-def _configure_jax_gpu(precision: str) -> None:
113:    """Pin THIS MPI rank to one local GPU and run JAX on cuda (route-A).
114-
115-    Single-node multi-GPU via mpi4jax (the SAME mpi4jax halo machinery as the
116-    CPU path — make_latlon_mpi_step / cube — just on cuda devices over the
117-    PCIe pair).  Must run BEFORE any JAX import.  Local rank from the launcher
118:    env (OpenMPI / SLURM).  Mirrors the ocean harness ``_configure_jax_gpu``
119-    (bench_ocean_mpi_scaling.py) so the atm lat-lon dycore gets a 2-GPU
120-    number via the proven overlay-venv route-A (cuda jax + CUDA-built
121-    mpi4jax)."""
122-    # RESPECT an EXPLICIT CUDA_VISIBLE_DEVICES (set by the launcher's per-task
--
129-    # explicit CVD, so honoring an explicit CVD here is safe.
130-    # (JAX_PLATFORMS / x64 / prealloc below run either way.)
131-    _existing_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
132-    if not _existing_cvd:
133:        local = (os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK")
134-                 or os.environ.get("MV2_COMM_WORLD_LOCAL_RANK"))
135-        if local is None:
136-            # SLURM_LOCALID is exported even in a plain sbatch step (ntasks=1, no
137-            # srun); pinning on it THERE hides all but GPU 0 from a single-
--
151-    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
152-
153-
154-def _launcher_world_size() -> int:
155:    """World size the MPI/SLURM/PALS launcher env reports (1 = no launcher).
156-
157-    GLOBAL sizes only (codex 2026-07-24 round-3): PALS_LOCAL_SIZE is
158-    PER-NODE — using it as world size would accept a one-node partial
159-    federation as complete. PALS jobs expose no global size env here, so
160-    they fall through to 1 and rely on the mpi4py path. Prefer the STEP
161-    task count over the allocation's SLURM_NTASKS so an `srun -n1` inside
162-    a larger allocation is not mistaken for the allocation-wide count.
163-    """
164:    for var in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE",
165-                "SLURM_STEP_NUM_TASKS", "SLURM_NTASKS"):
166-        val = os.environ.get(var)
167-        if val and val.isdigit():
168-            return int(val)
169-    return 1
170-
171-
172-def _under_mpi_launcher() -> bool:
173:    """True when an MPI launcher started this process.
174-
175-    Size vars alone under-detect Cray PALS (PALS_LOCAL_SIZE is per-node;
176-    a job can expose only per-rank ids) — so the PRESENCE of a per-rank id
177-    counts as launcher evidence too (codex round-2).
--
179-    if _launcher_world_size() > 1:
180-        return True
181-    return any(
182-        v in os.environ
183:        for v in ("PALS_RANKID", "PMI_RANK", "OMPI_COMM_WORLD_RANK")
184-    )
185-
186-
187-def _init_mpi() -> tuple[int, int]:
188:    """Initialize MPI and return (rank, n_ranks)."""
189-    try:
190:        from mpi4py import MPI
191:        comm = MPI.COMM_WORLD
192-        return comm.Get_rank(), comm.Get_size()
193-    except (ImportError, RuntimeError) as e:
194-        # RuntimeError: mpi4py installed but no loadable libmpi (common in
195:        # a GPU-only venv). A single-process run must not require MPI —
196:        # BUT under a real MPI launcher a broken mpi4py must fail LOUDLY
197-        # here, or every rank silently runs duplicated serial work
198-        # reporting n_ranks=1 (codex finding).
199-        if _under_mpi_launcher():
200-            raise RuntimeError(
201:                f"MPI launcher detected (world size "
202-                f"{_launcher_world_size()}) but mpi4py is unusable: {e}"
203-            ) from e
204-        return 0, 1
205-
--
210-
211-@dataclass
212-class TimingResult:
213-    n_ranks: int
214:    resolution: int
215-    n_levels: int
216-    precision: str
217-    mode: str
218-    grid_type: str
--
255-            self.timestamp_utc = datetime.now(timezone.utc).isoformat()
256-
257-
258-# ===========================================================================
259:# Grid choices and resolution ladders
260-# ===========================================================================
261-
262-GRID_CHOICES = ("cubed-sphere", "latlon", "icosahedral", "spectral")
263-PHYSICS_CHOICES = ("none", "held_suarez", "gray_sbm", "rrtmg_full", "moist")
--
268-    # Iter 40 honest-sweep: ``_build_physics_fn`` only knows how to
269-    # construct the Held-Suarez forcing.  ``gray_sbm`` /
270-    # ``rrtmg_full`` are routed through the AMIP segment driver
271-    # (``run_levante_gpu_scaling.py`` ``_run_segment_benchmark``),
272:    # not through this CPU-MPI script.  Listing them as supported
273-    # here let users pass ``--physics gray_sbm`` and silently
274-    # benchmark dycore-only with the moist-physics label.
275-    # "moist" = moisture (q_v/q_c/q_r) + Kessler warm-rain condensation,
276-    # NO radiation: the moist baroclinic-wave case.  Wired for every grid whose
277-    # dycore advects tracers: the cubed-sphere FV3 C-D grid (advective form,
278:    # consistent with T — single-rank + full-state MPI; NOT the cs-spmd
279-    # sub-face path, which still rejects non-"none" physics), the lat-lon C-grid
280:    # (mass-weighted flux form, serial + band/2-D-pencil MPI), and
281-    # icosahedral/MPAS.  Kessler is column-local, so it adds NO horizontal halo
282-    # coupling beyond the dycore's tracer exchange (moist scales like dry).
283-    "cubed-sphere": {"none", "held_suarez", "moist"},
284-    "latlon": {"none", "held_suarez", "moist"},
285-    "icosahedral": {"none", "held_suarez", "moist"},
286-    # spectral "moist" = q_v/q_c/q_r + Kessler warm-rain (no radiation), the
287-    # moist baroclinic wave, via make_kessler_forcing_spectral.  Single-device
288:    # (spectral has no MPI path) — a physics-capability case, not a scaling one.
289-    "spectral": {"none", "held_suarez", "moist"},
290-}
291-
292-
--
320-WEAK_BASE_CS = 24
321-WEAK_BASE_LL = 64
322-WEAK_BASE_ICO = 4  # subdivision level
323-
324:# Strong scaling resolution sets
325-STRONG_RES_CS = [24, 48, 96]
326-STRONG_RES_LL = [64, 128, 256]
327-STRONG_RES_ICO = [4, 5, 6, 7, 8]  # levels 4-8 (L8 = 655,362 cells, ~25 km)
328-STRONG_RES_SP = [21, 42]
329-
330-
331:def _weak_resolution_cs(n_ranks: int, base_n: int = WEAK_BASE_CS) -> int:
332-    n_raw = base_n * math.sqrt(n_ranks)
333-    return max(4, 2 * round(n_raw / 2))
334-
335-
336:def _weak_resolution_ll(n_ranks: int, base_n: int = WEAK_BASE_LL) -> int:
337-    """Weak-scaling n_lat for the lat-lon band decomposition.
338-
339-    Constant *cells per rank* (the icosahedral analog): total cells
340-    scale as ``n_lat * n_lon = 2 * n_lat**2``, so ``n_lat ~
--
362-        n_rounded += 2
363-    return n_rounded
364-
365-
366:def _weak_resolution_ico(n_ranks: int, base_level: int = WEAK_BASE_ICO) -> int:
367-    base_cells = 10 * 4 ** base_level + 2
368-    best_level = base_level
369-    best_ratio = float("inf")
370-    for lev in range(base_level, 9):
--
380-def _valid_rank_counts(max_ranks: int, grid_type: str) -> list[int]:
381-    if grid_type == "spectral":
382-        return [1]
383-    if grid_type == "cubed-sphere":
384:        # Iter 13 honest-sweep guard: cubed-sphere MPI is not yet
385-        # domain-decomposed — every rank holds the full (6, n, n, ...)
386-        # state and runs the full dycore.  Multi-rank wall-clock
387-        # measurements are *not* real weak/strong scaling, just
388-        # rank-replicated computation plus halo overhead.  Until the
389-        # halo-side scattered indexing (iter 3) is plumbed through
390-        # ``model_driver.py`` and the scaling drivers actually scatter
391:        # per-rank state, restrict cubed-sphere MPI sweeps to rank 1
392-        # so the summary numbers reflect genuine single-rank
393-        # throughput rather than replicated-dynamics noise.
394-        return [1]
395-    # latlon and icosahedral: powers of 2 up to max
--
430-        dx_min = (math.pi / 2) * R / (n_grid * math.sqrt(3))
431-
432-    dt = cfl * dx_min / (u_max + c_grav)
433-    # Round down to a "nice" value; no floor — lat-lon pole cells can
434:    # require sub-second timesteps at very high resolution.
435-    if dt >= 30.0:
436-        return 30.0 * int(dt / 30.0)
437-    elif dt >= 5.0:
438-        return 5.0 * int(dt / 5.0)
--
449-
450-def _build_amip_step(
451-    *,
452-    grid_type: str,
453:    resolution: int,
454-    nlev: int,
455-    rank: int,
456-    n_ranks: int,
457-    precision: str,
--
474-    from legoesm.grids.vertical import create_sigma_coordinate
475-    sigma = create_sigma_coordinate(nlev)
476-
477-    if dt is None:
478:        dt = _auto_dt(resolution, grid_type)
479-
480-    def _cast(x):
481-        if isinstance(x, jnp.ndarray) and jnp.issubdtype(x.dtype, jnp.floating):
482-            return x.astype(dtype)
--
484-
485-    if grid_type == "cubed-sphere":
486-        if cs_spmd:
487-            out = _build_cubed_sphere_spmd(
488:                resolution, nlev, dt, dtype, physics_level, _cast)
489-        else:
490:            out = _build_cubedsphere(resolution, nlev, sigma, dt, dtype, rank,
491-                                     n_ranks, physics_level, _cast)
492-    elif grid_type == "latlon":
493:        out = _build_latlon(resolution, nlev, sigma, dt, dtype, rank, n_ranks,
494-                            physics_level, _cast, latlon_2d=latlon_2d)
495-    elif grid_type == "icosahedral":
496:        out = _build_icosahedral(resolution, nlev, sigma, dt, dtype, rank,
497-                                 n_ranks, physics_level, _cast)
498-    elif grid_type == "spectral":
499:        out = _build_spectral(resolution, nlev, sigma, dt, dtype,
500-                              physics_level, _cast)
501-    else:
502-        raise ValueError(f"Unsupported grid: {grid_type!r}")
503-    # Normalize to (step_fn, state, dt, total_cells, cells_per_rank,
--
505-    # metrics; the other builders return a 5-tuple, padded with None here.
506-    return out if len(out) == 6 else (*out, None)
507-
508-
509:def _build_cubedsphere(resolution, nlev, sigma, dt, dtype, rank, n_ranks,
510-                        physics_level, cast_fn):
511-    import jax
512-    import jax.numpy as jnp
513-
--
519-        hydrostatic_to_fv3,
520-    )
521-    from tests.test_cases.baroclinic_wave import baroclinic_wave_init
522-
523:    grid = create_cubed_sphere(resolution)
524-    cdgrid = create_cubed_sphere_cdgrid(grid)
525-    config = CDGridPrimitiveEquationConfig(
526-        hyperdiff_coeff=0.0,
527-        hyperdiff_ps_coeff=0.0,
--
535-    state_cc = baroclinic_wave_init(grid, sigma, perturbed=True, moist=_moist)
536-    state = hydrostatic_to_fv3(state_cc, cdgrid)
537-    state = jax.tree.map(cast_fn, state)
538-
539:    total_cells = 6 * resolution * resolution * nlev
540-
541:    # MPI: initialise distributed but keep full (6, n, n, ...) state
542-    # on every rank, matching the production driver.  pad_halo_mpi
543:    # requires the full shape; each rank steps all faces and MPI halo
544-    # exchange keeps owned faces correct.  The FV3 CD-grid dycore advects
545-    # q_v/q_c/q_r in advective form (consistent with T); its tracer halo
546-    # rides the same auto-dispatched pad_halo_4d (mpi/spmd/local).
547-    if n_ranks > 1:
548-        from legoesm.parallel.distributed import initialize_distributed
549:        initialize_distributed(global_n=resolution, grid_type="cubed_sphere")
550-
551-    if _moist:
552-        # Kessler warm-rain bound to this step's dt (the physics_fn convention
553-        # passes no timestep).  Column-local, so it adds NO horizontal halo
--
564-    cells_per_rank = total_cells // max(1, n_ranks)
565-    return step_fn, state, dt, total_cells, cells_per_rank
566-
567-
568:def _build_cubed_sphere_spmd(resolution, nlev, dt, dtype, physics_level,
569-                             cast_fn):
570-    """A1 path: TRUE cubed-sphere decomposition via jax.distributed.
571-
572-    Multi-controller SPMD: the global face mesh is built from
--
626-    # silently building a broken/replicated program.
627-    if n_global > 6:
628-        kt = int(round((n_global // 6) ** 0.5))
629-        return _build_cubed_sphere_tiled_loop(
630:            resolution, nlev, dt, physics_level, cast_fn, kt)
631-
632-    cfg_mesh = create_device_mesh(n_devices=n_global, devices=gdev)
633:    grid = create_cubed_sphere(resolution)
634-    sigma = create_sigma_coordinate(nlev)
635-    state = hydrostatic_to_fv3(
636-        baroclinic_wave_init(grid, sigma, perturbed=True, moist=_moist),
637-        create_cubed_sphere_cdgrid(grid))
--
650-        zero_mean_ps_tendency=True,
651-    )
652-    model = CDGridPrimitiveEquationModel(grid, sigma, config)
653-
654:    sharded_step = make_sharded_step(model, cfg_mesh, n=resolution, nlev=nlev)
655-    if _moist:
656-        # Kessler warm-rain bound to this step's dt; column-local, so it adds
657-        # NO horizontal halo coupling beyond the dycore's q_v/q_c/q_r tracer
658-        # exchange (which rides the SAME multiface-ppermute halo as T under the
--
665-    else:
666-        step_fn = sharded_step
667-    state = shard_pytree(state, cfg_mesh)
668-
669:    total_cells = 6 * resolution * resolution * nlev
670-    cells_per_rank = total_cells // n_global
671-    return step_fn, state, dt, total_cells, cells_per_rank
672-
673-
674:def _build_cubed_sphere_tiled_loop(resolution, nlev, dt, physics_level,
675-                                   cast_fn, kt):
676-    """np>6 cube lane: the BLOCKED persistent tiled step (6*kt^2 devices).
677-
678-    State stays TILE-SHARDED across steps (input layout == output layout,
--
707-            f"or 'moist' (the Kessler column bridge); got "
708-            f"{physics_level!r}. Run <=6 devices for other physics."
709-        )
710-    _moist = physics_level == "moist"
711:    if resolution % kt:
712-        raise ValueError(
713:            f"cs-spmd tiled lane: resolution {resolution} must divide by "
714-            f"kt={kt} (tile edge = n/kt).")
715-
716:    grid = create_cubed_sphere(resolution)
717-    cdgrid = create_cubed_sphere_cdgrid(grid)
718-    sigma = create_sigma_coordinate(nlev)
719-    # IDENTICAL config to the np<=6 cs-spmd lane (controlled comparison).
720-    # This literal is inside the tiled envelope by construction (every damp
--
737-
738-    n_global = 6 * kt * kt
739-    dev = np.array(jax.devices()[:n_global]).reshape(6, kt, kt)
740-    mesh = Mesh(dev, axis_names=("face", "tile_i", "tile_j"))
741:    nl = resolution // kt
742-
743-    # Moist: the Kessler COLUMN bridge — the SAME shared
744-    # kessler_column_tendencies core the np<=6 lane's
745-    # make_kessler_forcing_cube physics_fn runs, so the device ladder
--
752-        )
753-        column_physics_fn = make_kessler_column_physics_fn(sigma, dt)
754-

exec
/usr/bin/bash -lc "rg -n 'add_argument|def main|def _build_latlon|latlon_2d|latlon-2d|2-D|pencil|n_lat %|n_lon %|n_ranks' scripts/bench/bench_atm_latlon_spmd_scaling.py scripts/bench/run_cpu_mpi_scaling.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 339ms:
scripts/bench/bench_atm_latlon_spmd_scaling.py:120:def main() -> int:
scripts/bench/bench_atm_latlon_spmd_scaling.py:122:    p.add_argument("--n-lat", type=int, default=128)
scripts/bench/bench_atm_latlon_spmd_scaling.py:123:    p.add_argument("--n-lon", type=int, default=256)
scripts/bench/bench_atm_latlon_spmd_scaling.py:124:    p.add_argument("--nlev", type=int, default=30)
scripts/bench/bench_atm_latlon_spmd_scaling.py:125:    p.add_argument("--n-devices", type=int, required=True)
scripts/bench/bench_atm_latlon_spmd_scaling.py:126:    p.add_argument("--mode", choices=["strong", "weak"], default="strong")
scripts/bench/bench_atm_latlon_spmd_scaling.py:127:    p.add_argument("--nlat-per-dev", type=int, default=32,
scripts/bench/bench_atm_latlon_spmd_scaling.py:129:    p.add_argument("--steps", type=int, default=12,
scripts/bench/bench_atm_latlon_spmd_scaling.py:133:    p.add_argument("--warmup", type=int, default=2,
scripts/bench/bench_atm_latlon_spmd_scaling.py:138:    p.add_argument("--blocks", type=int, default=2,
scripts/bench/bench_atm_latlon_spmd_scaling.py:141:    p.add_argument("--probe-steps", type=int, default=3,
scripts/bench/bench_atm_latlon_spmd_scaling.py:144:    p.add_argument("--segment-steps", type=int, default=0,
scripts/bench/bench_atm_latlon_spmd_scaling.py:149:    p.add_argument("--physics", choices=["none", "held_suarez"], default="none")
scripts/bench/bench_atm_latlon_spmd_scaling.py:150:    p.add_argument("--dt", type=float, default=60.0)
scripts/bench/bench_atm_latlon_spmd_scaling.py:151:    p.add_argument("--single-dev-fused-ms", type=float, default=None,
scripts/bench/bench_atm_latlon_spmd_scaling.py:156:    p.add_argument("--comm-latency-us", type=float, default=None,
scripts/bench/bench_atm_latlon_spmd_scaling.py:160:    p.add_argument("--comm-bandwidth-gbs", type=float, default=None,
scripts/bench/bench_atm_latlon_spmd_scaling.py:164:    p.add_argument("--out", type=str, default="results/a1/spmd_scaling.jsonl")
scripts/bench/bench_atm_latlon_spmd_scaling.py:165:    p.add_argument("--multicontroller", action="store_true",
scripts/bench/bench_atm_latlon_spmd_scaling.py:171:    p.add_argument("--coordinator", type=str, default=None,
scripts/bench/bench_atm_latlon_spmd_scaling.py:221:    if n_lat % nd != 0:
scripts/bench/bench_atm_latlon_spmd_scaling.py:436:        # cells_per_rank is per PROCESS (n_ranks semantics); the per-device
scripts/bench/run_cpu_mpi_scaling.py:188:    """Initialize MPI and return (rank, n_ranks)."""
scripts/bench/run_cpu_mpi_scaling.py:198:        # reporting n_ranks=1 (codex finding).
scripts/bench/run_cpu_mpi_scaling.py:213:    n_ranks: int
scripts/bench/run_cpu_mpi_scaling.py:233:    # (proc_lat x proc_lon pencil).  Lets the collector/plotter separate the
scripts/bench/run_cpu_mpi_scaling.py:234:    # 2-D-pencil curve from the 1-D band laggard.  N/A for other grids ("band"
scripts/bench/run_cpu_mpi_scaling.py:280:    # (mass-weighted flux form, serial + band/2-D-pencil MPI), and
scripts/bench/run_cpu_mpi_scaling.py:331:def _weak_resolution_cs(n_ranks: int, base_n: int = WEAK_BASE_CS) -> int:
scripts/bench/run_cpu_mpi_scaling.py:332:    n_raw = base_n * math.sqrt(n_ranks)
scripts/bench/run_cpu_mpi_scaling.py:336:def _weak_resolution_ll(n_ranks: int, base_n: int = WEAK_BASE_LL) -> int:
scripts/bench/run_cpu_mpi_scaling.py:341:    sqrt(n_ranks)`` keeps cells/rank fixed.  Constraints layered on
scripts/bench/run_cpu_mpi_scaling.py:344:    * divisible by ``n_ranks`` (uniform bands → clean cells/rank),
scripts/bench/run_cpu_mpi_scaling.py:359:    n_raw = base_n * math.sqrt(n_ranks)
scripts/bench/run_cpu_mpi_scaling.py:361:    while n_rounded % n_ranks != 0 or n_rounded < 2 * n_ranks:
scripts/bench/run_cpu_mpi_scaling.py:366:def _weak_resolution_ico(n_ranks: int, base_level: int = WEAK_BASE_ICO) -> int:
scripts/bench/run_cpu_mpi_scaling.py:372:        cells_per_rank = cells / n_ranks
scripts/bench/run_cpu_mpi_scaling.py:456:    n_ranks: int,
scripts/bench/run_cpu_mpi_scaling.py:461:    latlon_2d: bool = False,
scripts/bench/run_cpu_mpi_scaling.py:491:                                     n_ranks, physics_level, _cast)
scripts/bench/run_cpu_mpi_scaling.py:493:        out = _build_latlon(resolution, nlev, sigma, dt, dtype, rank, n_ranks,
scripts/bench/run_cpu_mpi_scaling.py:494:                            physics_level, _cast, latlon_2d=latlon_2d)
scripts/bench/run_cpu_mpi_scaling.py:497:                                 n_ranks, physics_level, _cast)
scripts/bench/run_cpu_mpi_scaling.py:509:def _build_cubedsphere(resolution, nlev, sigma, dt, dtype, rank, n_ranks,
scripts/bench/run_cpu_mpi_scaling.py:547:    if n_ranks > 1:
scripts/bench/run_cpu_mpi_scaling.py:564:    cells_per_rank = total_cells // max(1, n_ranks)
scripts/bench/run_cpu_mpi_scaling.py:808:def _factor_2d_latlon(n_ranks, n_lat, n_lon, min_lat=2, min_lon=2):
scripts/bench/run_cpu_mpi_scaling.py:809:    """Factor ``n_ranks`` into ``(proc_lat, proc_lon)`` for the 2-D pencil,
scripts/bench/run_cpu_mpi_scaling.py:811:    (the whole point of 2-D vs the 1-D band).
scripts/bench/run_cpu_mpi_scaling.py:822:    for pl in range(1, n_ranks + 1):
scripts/bench/run_cpu_mpi_scaling.py:823:        if n_ranks % pl:
scripts/bench/run_cpu_mpi_scaling.py:825:        pc = n_ranks // pl
scripts/bench/run_cpu_mpi_scaling.py:836:            f"_factor_2d_latlon: no 2-D factorisation of n_ranks={n_ranks} "
scripts/bench/run_cpu_mpi_scaling.py:844:def _build_latlon(resolution, nlev, sigma, dt, dtype, rank, n_ranks,
scripts/bench/run_cpu_mpi_scaling.py:845:                   physics_level, cast_fn, latlon_2d=False):
scripts/bench/run_cpu_mpi_scaling.py:894:    if n_ranks > 1 and latlon_2d:
scripts/bench/run_cpu_mpi_scaling.py:895:        # 2-D pencil (proc_lat x proc_lon) decomposition — the SOTA fix for
scripts/bench/run_cpu_mpi_scaling.py:899:        # (serial) -> 2-D layout -> rank-local block model -> scatter ->
scripts/bench/run_cpu_mpi_scaling.py:900:        # make_latlon_2d_mpi_step (which arms the MPI halo backend + sets the
scripts/bench/run_cpu_mpi_scaling.py:903:            make_latlon_2d_layout,
scripts/bench/run_cpu_mpi_scaling.py:904:            make_latlon_2d_mpi_step,
scripts/bench/run_cpu_mpi_scaling.py:905:            scatter_state_latlon_2d,
scripts/bench/run_cpu_mpi_scaling.py:908:        proc_lat, proc_lon = _factor_2d_latlon(n_ranks, n_lat, n_lon)
scripts/bench/run_cpu_mpi_scaling.py:910:        layout2d = make_latlon_2d_layout(
scripts/bench/run_cpu_mpi_scaling.py:917:        state = scatter_state_latlon_2d(cgrid_global, layout2d)
scripts/bench/run_cpu_mpi_scaling.py:918:        step_fn = make_latlon_2d_mpi_step(
scripts/bench/run_cpu_mpi_scaling.py:924:    if n_ranks > 1:
scripts/bench/run_cpu_mpi_scaling.py:945:        if n_lat // n_ranks < 2:
scripts/bench/run_cpu_mpi_scaling.py:949:                f"{n_ranks} ranks ({n_lat // n_ranks} rows/rank)."
scripts/bench/run_cpu_mpi_scaling.py:973:        # ``total // n_ranks`` is only exact when n_lat divides evenly
scripts/bench/run_cpu_mpi_scaling.py:975:        # run does not — the first ``n_lat % n_ranks`` ranks then carry
scripts/bench/run_cpu_mpi_scaling.py:992:def _build_icosahedral(resolution, nlev, sigma, dt, dtype, rank, n_ranks,
scripts/bench/run_cpu_mpi_scaling.py:1010:    if n_ranks > 1:
scripts/bench/run_cpu_mpi_scaling.py:1049:    if n_ranks > 1:
scripts/bench/run_cpu_mpi_scaling.py:1066:        layout = make_voronoi_partition_layout(mesh, rank, n_ranks,
scripts/bench/run_cpu_mpi_scaling.py:1086:    cells_per_rank = total_cells // max(1, n_ranks)
scripts/bench/run_cpu_mpi_scaling.py:1141:    n_ranks: int,
scripts/bench/run_cpu_mpi_scaling.py:1150:    latlon_2d: bool = False,
scripts/bench/run_cpu_mpi_scaling.py:1164:        n_ranks=n_ranks,
scripts/bench/run_cpu_mpi_scaling.py:1169:        latlon_2d=latlon_2d,
scripts/bench/run_cpu_mpi_scaling.py:1171:    # "2d" only for a genuine multi-rank lat-lon pencil; everything else
scripts/bench/run_cpu_mpi_scaling.py:1174:        latlon_2d and grid_type == "latlon" and n_ranks > 1
scripts/bench/run_cpu_mpi_scaling.py:1188:            f"  [{precision}] {res_label}/L{nlev} on {n_ranks} rank(s) | "
scripts/bench/run_cpu_mpi_scaling.py:1282:    # but n_ranks is the PROCESS count (1 for a single-process multi-GPU run).
scripts/bench/run_cpu_mpi_scaling.py:1285:    # this path; the multi-controller cs-spmd path has n_ranks == device_count
scripts/bench/run_cpu_mpi_scaling.py:1287:    _record_ndev = jax.device_count() if cs_spmd else n_ranks
scripts/bench/run_cpu_mpi_scaling.py:1289:        n_ranks=_record_ndev,
scripts/bench/run_cpu_mpi_scaling.py:1320:    n_ranks: int
scripts/bench/run_cpu_mpi_scaling.py:1387:    ``result.n_ranks`` was rewritten to ``jax.device_count()`` for the
scripts/bench/run_cpu_mpi_scaling.py:1388:    filename / plot axis.  The metadata ``n_ranks`` (= process count) must NOT
scripts/bench/run_cpu_mpi_scaling.py:1394:    # Tag a non-default (2-D) decomposition into the filename so a 2-D-pencil
scripts/bench/run_cpu_mpi_scaling.py:1400:        f"r{result.resolution}_n{result.n_ranks}_{result.precision}.json"
scripts/bench/run_cpu_mpi_scaling.py:1422:    # a hybrid 8r x 4c run and a packed 32r x 1c run both report n_ranks but use
scripts/bench/run_cpu_mpi_scaling.py:1423:    # 32 vs 128 cores. cpus_per_task * n_ranks = the true resource count.
scripts/bench/run_cpu_mpi_scaling.py:1433:    payload["n_cores"] = result.n_ranks * _cpt
scripts/bench/run_cpu_mpi_scaling.py:1440:    # cs-spmd: result.n_ranks is the DEVICE count (rewritten upstream); leave
scripts/bench/run_cpu_mpi_scaling.py:1441:    # metadata n_ranks to auto process-count.  Non-cs-spmd (mpi4jax): jax is
scripts/bench/run_cpu_mpi_scaling.py:1443:    _md_n_ranks = None if cs_spmd else result.n_ranks
scripts/bench/run_cpu_mpi_scaling.py:1450:        n_ranks=_md_n_ranks,
scripts/bench/run_cpu_mpi_scaling.py:1455:        transport=("mpi4jax" if (not cs_spmd and result.n_ranks > 1)
scripts/bench/run_cpu_mpi_scaling.py:1457:        n_gpus=(result.n_ranks
scripts/bench/run_cpu_mpi_scaling.py:1460:        # cells_per_rank is per PROCESS (n_ranks semantics).  cs-spmd:
scripts/bench/run_cpu_mpi_scaling.py:1462:        # while metadata n_ranks defaults to jax.process_count() — divide
scripts/bench/run_cpu_mpi_scaling.py:1487:        f"{result.n_ranks:>5d} ranks | {result.resolution:<5d} | "
scripts/bench/run_cpu_mpi_scaling.py:1503:    p.add_argument(
scripts/bench/run_cpu_mpi_scaling.py:1507:    p.add_argument(
scripts/bench/run_cpu_mpi_scaling.py:1511:    p.add_argument(
scripts/bench/run_cpu_mpi_scaling.py:1515:    p.add_argument(
scripts/bench/run_cpu_mpi_scaling.py:1519:    p.add_argument(
scripts/bench/run_cpu_mpi_scaling.py:1523:    p.add_argument(
scripts/bench/run_cpu_mpi_scaling.py:1529:    p.add_argument("--n-levels", type=int, default=26)
scripts/bench/run_cpu_mpi_scaling.py:1530:    p.add_argument("--n-warmup", type=int, default=5)
scripts/bench/run_cpu_mpi_scaling.py:1531:    p.add_argument("--n-timing", type=int, default=50)
scripts/bench/run_cpu_mpi_scaling.py:1532:    p.add_argument(
scripts/bench/run_cpu_mpi_scaling.py:1543:    p.add_argument(
scripts/bench/run_cpu_mpi_scaling.py:1544:        "--latlon-2d", action="store_true",
scripts/bench/run_cpu_mpi_scaling.py:1545:        help="Lat-lon C-grid 2-D pencil decomposition (proc_lat x proc_lon "
scripts/bench/run_cpu_mpi_scaling.py:1551:             "tests/distributed/test_latlon_2d_mpi_step.py (mass<1e-12 + "
scripts/bench/run_cpu_mpi_scaling.py:1554:    p.add_argument(
scripts/bench/run_cpu_mpi_scaling.py:1558:    p.add_argument(
scripts/bench/run_cpu_mpi_scaling.py:1562:    p.add_argument(
scripts/bench/run_cpu_mpi_scaling.py:1566:    p.add_argument(
scripts/bench/run_cpu_mpi_scaling.py:1573:def main() -> int:
scripts/bench/run_cpu_mpi_scaling.py:1576:    # --latlon-2d does not propagate through the sweep/--case round-trip yet
scripts/bench/run_cpu_mpi_scaling.py:1577:    # (CaseSpec carries no decomposition field), so a ``--sweep --latlon-2d``
scripts/bench/run_cpu_mpi_scaling.py:1579:    # Fail loud — use a direct ``--resolution N --latlon-2d`` invocation (the
scripts/bench/run_cpu_mpi_scaling.py:1581:    if args.sweep and args.latlon_2d:
scripts/bench/run_cpu_mpi_scaling.py:1583:            "ERROR: --latlon-2d is not threaded through --sweep yet (CaseSpec "
scripts/bench/run_cpu_mpi_scaling.py:1586:            "--latlon-2d' run (one per rank count) instead.",
scripts/bench/run_cpu_mpi_scaling.py:1682:        # NCCL cube lane that needs no MPI at all). n_ranks keeps the
scripts/bench/run_cpu_mpi_scaling.py:1688:        n_ranks = _lw if _lw > 1 else _jax.process_count()
scripts/bench/run_cpu_mpi_scaling.py:1690:        rank, n_ranks = _init_mpi()
scripts/bench/run_cpu_mpi_scaling.py:1696:    # produce N independent serial runs labelled n_ranks=N.
scripts/bench/run_cpu_mpi_scaling.py:1699:        if n_ranks > 1 and _jax.process_count() != n_ranks:
scripts/bench/run_cpu_mpi_scaling.py:1702:                    f"ERROR: --cs-spmd launched with {n_ranks} MPI "
scripts/bench/run_cpu_mpi_scaling.py:1710:        if n_ranks == 1 and is_rank0:
scripts/bench/run_cpu_mpi_scaling.py:1738:            resolution = _weak_resolution_cs(n_ranks) if mode == "weak" else 48
scripts/bench/run_cpu_mpi_scaling.py:1740:            resolution = _weak_resolution_ll(n_ranks) if mode == "weak" else 64
scripts/bench/run_cpu_mpi_scaling.py:1742:            resolution = _weak_resolution_ico(n_ranks) if mode == "weak" else 5
scripts/bench/run_cpu_mpi_scaling.py:1747:    if grid_type == "spectral" and n_ranks > 1:
scripts/bench/run_cpu_mpi_scaling.py:1754:    # explicit ``--case`` with ``n_ranks>1`` or invoke the script under
scripts/bench/run_cpu_mpi_scaling.py:1758:    if grid_type == "cubed-sphere" and n_ranks > 1 and not args.cs_spmd:
scripts/bench/run_cpu_mpi_scaling.py:1772:    if args.latlon_2d and grid_type != "latlon":
scripts/bench/run_cpu_mpi_scaling.py:1774:            print("ERROR: --latlon-2d applies only to --grid latlon.",
scripts/bench/run_cpu_mpi_scaling.py:1778:    # lat).  The 2-D pencil splits lat over proc_lat (< n_ranks), so its own
scripts/bench/run_cpu_mpi_scaling.py:1780:    # guard for --latlon-2d.
scripts/bench/run_cpu_mpi_scaling.py:1781:    if (grid_type == "latlon" and not args.latlon_2d
scripts/bench/run_cpu_mpi_scaling.py:1782:            and n_ranks > 1 and resolution // n_ranks < 2):
scripts/bench/run_cpu_mpi_scaling.py:1787:                f"{n_ranks} ranks gives {resolution // n_ranks} "
scripts/bench/run_cpu_mpi_scaling.py:1800:        print(f"  Ranks:     {n_ranks}")
scripts/bench/run_cpu_mpi_scaling.py:1810:        n_ranks=n_ranks,
scripts/bench/run_cpu_mpi_scaling.py:1818:        latlon_2d=bool(args.latlon_2d),

exec
/usr/bin/bash -lc "sed -n '110,245p' scripts/bench/bench_atm_latlon_spmd_scaling.py && printf '\\n--- 2d factor and builder ---\\n' && sed -n '808,990p' scripts/bench/run_cpu_mpi_scaling.py && printf '\\n--- parser/main validation ---\\n' && sed -n '1490,1830p' scripts/bench/run_cpu_mpi_scaling.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 253ms:
    model = _build_model(n_lat, n_lon, nlev)
    hs0 = held_suarez_init_latlon(model.grid, model.sigma_coord)
    c0 = hydrostatic_to_cgrid(hs0, model.grid)
    return model, c0


def _block(state):
    jax.block_until_ready(jax.tree.leaves(state))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-lat", type=int, default=128)
    p.add_argument("--n-lon", type=int, default=256)
    p.add_argument("--nlev", type=int, default=30)
    p.add_argument("--n-devices", type=int, required=True)
    p.add_argument("--mode", choices=["strong", "weak"], default="strong")
    p.add_argument("--nlat-per-dev", type=int, default=32,
                   help="weak mode: lat rows per device")
    p.add_argument("--steps", type=int, default=12,
                   help="Default lane: steps per fused lax.scan timing "
                        "block. Segment mode: number of timed BLOCKS of "
                        "--segment-steps steps each.")
    p.add_argument("--warmup", type=int, default=2,
                   help="Segment mode: timed blocks dropped from steady "
                        "stats (block 0 includes the scan compile). Default "
                        "lane: retained for CLI compat (fused-block timing "
                        "separates compile/probe/blocks explicitly).")
    p.add_argument("--blocks", type=int, default=2,
                   help="Default lane: timed fused blocks (per-block times "
                        "expose drift).")
    p.add_argument("--probe-steps", type=int, default=3,
                   help="Default lane: individually-synced steps for the "
                        "SEPARATE dispatch-latency probe (step_latency_ms).")
    p.add_argument("--segment-steps", type=int, default=0,
                   help="M2b: >0 compiles ONE lax.scan segment of this many "
                        "steps (built once, reused; band-sharded geometry) "
                        "and times BLOCKS of segment calls instead of "
                        "per-step host dispatch. 0 = default fused lane.")
    p.add_argument("--physics", choices=["none", "held_suarez"], default="none")
    p.add_argument("--dt", type=float, default=60.0)
    p.add_argument("--single-dev-fused-ms", type=float, default=None,
                   help="fused_step_ms of the nd=1 row at the SAME per-device "
                        "size (compute ingredient of the calibrated T_bound, "
                        "audit item 8). Omitted at nd>1 -> bound emitted null "
                        "+ flagged incomplete; nd=1 uses its own measurement.")
    p.add_argument("--comm-latency-us", type=float, default=None,
                   help="MEASURED per-message latency [us] of THIS machine's "
                        "fabric. Default: MACHINE-CALIBRATED-REQUIRED "
                        "placeholder in metadata.py -> bound_calibrated=false.")
    p.add_argument("--comm-bandwidth-gbs", type=float, default=None,
                   help="MEASURED link bandwidth [GB/s] of THIS machine's "
                        "fabric. Default: MACHINE-CALIBRATED-REQUIRED "
                        "placeholder in metadata.py -> bound_calibrated=false.")
    p.add_argument("--out", type=str, default="results/a1/spmd_scaling.jsonl")
    p.add_argument("--multicontroller", action="store_true",
                   help="Route-B multi-controller: jax.distributed.initialize "
                        "per process, ('lat',) mesh over the GLOBAL device set "
                        "(one process per GPU / per CPU-device group). NO "
                        "mpi4jax. --n-devices must equal the global device "
                        "count.")
    p.add_argument("--coordinator", type=str, default=None,
                   help="host:port for jax.distributed when auto-detection "
                        "(SLURM) is unavailable; process count/id then come "
                        "from OMPI_COMM_WORLD_SIZE/RANK.")
    args = p.parse_args()

    # Validate the schedule BEFORE any model/device work: a zero/negative
    # --steps would otherwise surface only as timed_scan_blocks' None
    # headline (default lane) or an empty timed loop (segment mode) after
    # the expensive build (the ocean twin's guard).
    if args.steps < 1:
        raise SystemExit(f"--steps must be >= 1, got {args.steps}")

    if args.multicontroller:
        # MUST run before any other JAX use (backend init).  The SHARED
        # helper owns the launcher-env contract (SLURM/OMPI auto-detect,
        # PALS mpi4py bootstrap, explicit-coordinator path) AND the
        # hardening: post-init silent-fallback guard + NCCL net-plugin
        # warning — an inline init here would bypass both (codex).
        from legoesm.parallel.early_init import (
            init_multicontroller_distributed,
        )
        init_multicontroller_distributed(args.coordinator)

    from legoesm.atmosphere.dynamics.gcm.sharded_atm_latlon_step import (
        atm_latlon_geometry_bytes,
        build_sharded_held_suarez_state_atm_latlon,
        make_sharded_atm_latlon_segment,
        make_sharded_atm_latlon_step)
    seg_n = int(args.segment_steps)
    if seg_n < 0:
        raise SystemExit(f"--segment-steps must be >= 0, got {seg_n}")
    physics_fn = None
    if args.physics == "held_suarez":
        from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_forcing_latlon
        physics_fn = held_suarez_forcing_latlon

    nd = args.n_devices
    avail = len(jax.devices())
    if avail < nd:
        raise SystemExit(f"need {nd} devices, have {avail} "
                         f"(set --xla_force_host_platform_device_count)")
    if args.multicontroller and nd != avail:
        # A mesh over a strict subset would leave some processes' devices out
        # of the program (non-addressable participation hazard). Route-B uses
        # ALL global devices: one band per device across every process.
        raise SystemExit(
            f"--multicontroller: --n-devices ({nd}) must equal the GLOBAL "
            f"device count ({avail} across {jax.process_count()} processes).")
    n_lat = args.n_lat if args.mode == "strong" else args.nlat_per_dev * nd
    if n_lat % nd != 0:
        raise SystemExit(f"n_lat {n_lat} not divisible by n_devices {nd}")

    if nd == 1:
        model, c0 = _build(n_lat, args.n_lon, args.nlev)
        mesh = None
        c = c0
    else:
        # #1100: band-local IC construction. The nd>1 lanes never materialise
        # the global (n_lat, n_lon, nlev) state per process — each leaf is
        # created via make_array_from_callback for the rows this process's
        # devices own (no global build, no device_put replication, no
        # assert_equal all-gather). This is what lets full-node-packed CPU
        # rungs (128 procs/node) survive at large n_lat.
        model = _build_model(n_lat, args.n_lon, args.nlev)
        mesh = jax.sharding.Mesh(np.array(jax.devices()[:nd]),
                                 axis_names=("lat",))
        c = build_sharded_held_suarez_state_atm_latlon(
            model.grid, model.sigma_coord, mesh)
    if seg_n > 0:
        seg_fn = make_sharded_atm_latlon_segment(
            model, mesh, seg_n, physics_fn=physics_fn)
    else:
        step = make_sharded_atm_latlon_step(model, mesh,
                                            physics_fn=physics_fn)

--- 2d factor and builder ---
def _factor_2d_latlon(n_ranks, n_lat, n_lon, min_lat=2, min_lon=2):
    """Factor ``n_ranks`` into ``(proc_lat, proc_lon)`` for the 2-D pencil,
    MINIMISING the per-rank halo perimeter ``n_lat/proc_lat + n_lon/proc_lon``
    (the whole point of 2-D vs the 1-D band).

    Constraints: each block keeps ``>= min_lat`` latitude rows (the halo=2
    PPM/biharmonic exchange) and ``>= min_lon`` longitude columns.  Raises if
    no valid factorisation exists (e.g. too many ranks for the resolution),
    rather than silently building a degenerate block.

    Returns the min-perimeter pair; ties broken toward the more balanced
    block (smaller ``|n_lat/pl - n_lon/pc|``).
    """
    best = None  # (perimeter, imbalance, proc_lat, proc_lon)
    for pl in range(1, n_ranks + 1):
        if n_ranks % pl:
            continue
        pc = n_ranks // pl
        blat, blon = n_lat // pl, n_lon // pc
        if blat < min_lat or blon < min_lon:
            continue
        perim = n_lat / pl + n_lon / pc
        imbal = abs(n_lat / pl - n_lon / pc)
        key = (perim, imbal)
        if best is None or key < best[0]:
            best = (key, pl, pc)
    if best is None:
        raise ValueError(
            f"_factor_2d_latlon: no 2-D factorisation of n_ranks={n_ranks} "
            f"keeps >= {min_lat} lat rows AND >= {min_lon} lon cols per block "
            f"for n_lat={n_lat}, n_lon={n_lon}.  Reduce ranks or raise "
            f"resolution."
        )
    return best[1], best[2]


def _build_latlon(resolution, nlev, sigma, dt, dtype, rank, n_ranks,
                   physics_level, cast_fn, latlon_2d=False):
    import jax
    import jax.numpy as jnp

    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        CGridLatLonPrimitiveEquationModel,
        CGridLatLonPrimitiveEquationConfig,
        hydrostatic_to_cgrid,
    )

    from legoesm.core.cfl import pole_cell_dx, cfl_max_dt

    n_lat = resolution
    n_lon = 2 * resolution
    grid = create_latlon_grid(n_lat, n_lon)

    # Clamp dt to pole-cell CFL limit (computed on the GLOBAL grid so
    # every rank derives the identical dt — only the boundary ranks
    # own the actual pole rows under band MPI).
    dx_pole = pole_cell_dx(grid)
    dt = min(dt, cfl_max_dt(dx_pole, 300.0, cfl_number=0.8, ndim=1))

    config = CGridLatLonPrimitiveEquationConfig(
        A_h=0.0,
        fix_mass=True,
        use_polar_filter=False,
    )

    # Baroclinic wave init for lat-lon (cell-centred HydrostaticState).
    from tests.test_cases.baroclinic_wave import (
        baroclinic_wave_init_latlon,
    )
    _moist = physics_level == "moist"
    state = baroclinic_wave_init_latlon(grid, sigma, perturbed=True, moist=_moist)
    state = jax.tree.map(cast_fn, state)

    total_cells = n_lat * n_lon * nlev

    if _moist:
        # Kessler warm-rain bound to this step's dt (the physics_fn convention
        # passes no timestep).  Column-local, so it adds NO horizontal halo
        # coupling beyond the dycore's existing mass-consistent tracer
        # exchange — moist scales on the same ladder as dry.
        from legoesm.atmosphere.forcing.idealized.kessler_forcing import make_kessler_forcing_latlon
        physics_fn = make_kessler_forcing_latlon(dt)
    else:
        physics_fn = _build_physics_fn(physics_level, "latlon")

    if n_ranks > 1 and latlon_2d:
        # 2-D pencil (proc_lat x proc_lon) decomposition — the SOTA fix for
        # the 1-D band's high-rank halo-perimeter starvation.  Wall poles
        # (regular grid; use_polar_filter already False above).  Same
        # build shape as the band: global cell-centred -> global C-grid
        # (serial) -> 2-D layout -> rank-local block model -> scatter ->
        # make_latlon_2d_mpi_step (which arms the MPI halo backend + sets the
        # rank-aware pole_v_bc + allreduced total_area itself).
        from legoesm.parallel.latlon_mpi import (
            make_latlon_2d_layout,
            make_latlon_2d_mpi_step,
            scatter_state_latlon_2d,
            slice_latlon_grid_to_block_2d,
        )
        proc_lat, proc_lon = _factor_2d_latlon(n_ranks, n_lat, n_lon)
        cgrid_global = hydrostatic_to_cgrid(state, grid)
        layout2d = make_latlon_2d_layout(
            rank, proc_lat, proc_lon, n_lat, n_lon,
        )
        block_grid = slice_latlon_grid_to_block_2d(grid, layout2d)
        local_model = CGridLatLonPrimitiveEquationModel(
            block_grid, sigma, config, dt=dt,
        )
        state = scatter_state_latlon_2d(cgrid_global, layout2d)
        step_fn = make_latlon_2d_mpi_step(
            local_model, layout2d, physics_fn=physics_fn,
        )
        cells_per_rank = layout2d.n_lat_local * layout2d.n_lon_local * nlev
        return step_fn, state, dt, total_cells, cells_per_rank

    if n_ranks > 1:
        # Latitude-band MPI (mirrors the multi-rank icosahedral path):
        # 1. convert the global state to raw C-grid arrays *before*
        #    arming the MPI halo backend (the conversion's pole pads
        #    must run on the global array with the local backend),
        # 2. arm the band layout + MPI halo backend,
        # 3. slice the global grid to this rank's band, build the
        #    rank-local model on it,
        # 4. scatter the global state to the band,
        # 5. wrap the step;  ``make_latlon_mpi_step`` forwards
        #    ``physics_fn`` per RK stage exactly like the serial
        #    ``model.step(state, dt, physics_fn=...)`` path
        #    (Held-Suarez is column-local, so it adds no halo
        #    coupling beyond the dycore's own exchanges).
        from legoesm.parallel.distributed import initialize_distributed_latlon
        from legoesm.parallel.latlon_mpi import (
            make_latlon_mpi_step,
            scatter_state_latlon,
            slice_latlon_grid_to_band,
        )

        if n_lat // n_ranks < 2:
            raise ValueError(
                f"lat-lon band MPI needs >=2 lat rows per rank for the "
                f"halo=2 PPM/biharmonic exchange; got n_lat={n_lat} on "
                f"{n_ranks} ranks ({n_lat // n_ranks} rows/rank)."
            )

        # (1) global cell-centred -> global C-grid, still serial.
        cgrid_global = hydrostatic_to_cgrid(state, grid)

        # (2) band layout + MPI halo backend.
        layout = initialize_distributed_latlon(
            global_n_lat=n_lat, global_n_lon=n_lon,
        )

        # (3) rank-local band model (the wrapper re-instantiates it
        # with rank-aware pole_v_bc + allreduced total_area itself).
        band_grid = slice_latlon_grid_to_band(grid, layout)
        local_model = CGridLatLonPrimitiveEquationModel(
            band_grid, sigma, config, dt=dt,
        )

        # (4) + (5)
        state = scatter_state_latlon(cgrid_global, layout)
        step_fn = make_latlon_mpi_step(
            local_model, layout, physics_fn=physics_fn,
        )
        # ACTUAL rank-local cell count (Codex P1-fix review, MINOR):
        # ``total // n_ranks`` is only exact when n_lat divides evenly
        # (the sweep generator enforces that, a direct ``--resolution``
        # run does not — the first ``n_lat % n_ranks`` ranks then carry
        # one extra row).  Rank 0 is in that first group, so its count
        # is the bottleneck-rank load — the right weak-scaling
        # normalizer for the rank-0-written result JSON.
        cells_per_rank = layout.n_lat_local * n_lon * nlev
    else:
        model = CGridLatLonPrimitiveEquationModel(grid, sigma, config, dt=dt)
        if physics_fn is not None:
            _phys = physics_fn
            step_fn = lambda state, dt: model.step(state, dt, physics_fn=_phys)
        else:
            step_fn = model.step
        cells_per_rank = total_cells

    return step_fn, state, dt, total_cells, cells_per_rank


--- parser/main validation ---
        f"SYPD={result.sypd:>8.3f} | {result.mcells_per_s:>9.1f} Mcells/s"
    )


# ===========================================================================
# CLI
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CPU MPI scaling benchmark for legoESM AMIP-like runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--grid", choices=list(GRID_CHOICES), default="latlon",
        help="Grid / dycore type.",
    )
    p.add_argument(
        "--resolution", type=int, default=0,
        help="Grid resolution (0 = auto from mode/ranks).",
    )
    p.add_argument(
        "--physics", choices=list(PHYSICS_CHOICES), default="held_suarez",
        help="Physics complexity level.",
    )
    p.add_argument(
        "--mode", choices=["weak", "strong", "both", "single"], default="single",
        help="Scaling mode: weak, strong, both, or single (one case).",
    )
    p.add_argument(
        "--precision", choices=["float32", "float64"], default="float64",
        help="Floating-point precision.",
    )
    p.add_argument(
        "--device", choices=["cpu", "gpu"], default="cpu",
        help="cpu (default, CPU-MPI) or gpu (route-A: pin each rank to one "
             "local GPU, run the SAME mpi4jax dycore on the PCIe pair). "
             "Needs the overlay venv (cuda jax + CUDA-built mpi4jax).",
    )
    p.add_argument("--n-levels", type=int, default=26)
    p.add_argument("--n-warmup", type=int, default=5)
    p.add_argument("--n-timing", type=int, default=50)
    p.add_argument(
        "--cs-spmd", action="store_true",
        help="Cubed-sphere TRUE domain decomposition via jax.distributed "
             "multi-controller SPMD (global face mesh + multiface "
             "ppermute; the A1 path).  Replaces the replicated-dynamics "
             "refusal: launch with srun -n {2,3,6} (must divide 6).  "
             "Uses jax.distributed ONLY — the mpi4jax halo backend is "
             "never armed in this mode (mixed stacks deadlock).  "
             "Parity receipt: scripts/tmp/_probe_spmd_cube_parity.py "
             "(shard-local vs serial = 6.7e-10 @5 steps, job 8462928).",
    )
    p.add_argument(
        "--latlon-2d", action="store_true",
        help="Lat-lon C-grid 2-D pencil decomposition (proc_lat x proc_lon "
             "factored from the rank count to minimise the per-rank halo "
             "perimeter) instead of the 1-D latitude band.  WALL POLES only "
             "(regular grid; use_polar_filter off) — a labeled throughput "
             "benchmark, NOT the atmosphere's 180-deg pole fold.  Targets the "
             "band's high-rank starvation (weak-E ~0.05).  Validated by "
             "tests/distributed/test_latlon_2d_mpi_step.py (mass<1e-12 + "
             "2x2==1x4).  --grid latlon only.",
    )
    p.add_argument(
        "--output-dir", type=str, default="results/cpu_scaling",
        help="Output directory for results.",
    )
    p.add_argument(
        "--sweep", action="store_true",
        help="Generate case list (JSON lines) and exit. No execution.",
    )
    p.add_argument(
        "--max-ranks", type=int, default=64,
        help="Maximum rank count for sweep mode.",
    )
    p.add_argument(
        "--case", type=str, default=None,
        help="JSON case spec (from sweep output).",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()

    # --latlon-2d does not propagate through the sweep/--case round-trip yet
    # (CaseSpec carries no decomposition field), so a ``--sweep --latlon-2d``
    # would silently emit band cases and the launcher would run the BAND path.
    # Fail loud — use a direct ``--resolution N --latlon-2d`` invocation (the
    # measurement sbatch does) until the sweep threads the flag (codex).
    if args.sweep and args.latlon_2d:
        print(
            "ERROR: --latlon-2d is not threaded through --sweep yet (CaseSpec "
            "has no decomposition field), so the swept cases would silently "
            "run the 1-D band.  Use a direct '--grid latlon --resolution N "
            "--latlon-2d' run (one per rank count) instead.",
            flush=True,
        )
        return 2

    # --- Sweep mode: just print cases and exit ---
    if args.sweep:
        # Iter 41: validate the grid+physics combo *before* the
        # sweep generator runs, so unsupported tiers (e.g. moist
        # physics on cubed-sphere/lat-lon CPU MPI, see iter 40)
        # error out immediately with a clear message instead of
        # producing a JSON-lines list that subsequently fails at
        # runtime under the SLURM array.
        _validate_physics(args.grid, args.physics)
        cases = generate_sweep_cases(
            args.grid, args.mode if args.mode != "single" else "both",
            args.physics, args.max_ranks,
            args.n_levels, args.precision,
        )
        for c in cases:
            print(json.dumps(asdict(c)))
        return 0

    # --- Configure JAX for the target device (ENV only; no backend init) ---
    # The GPU "is the backend really CUDA?" assertion is DEFERRED to after the
    # --cs-spmd jax.distributed.initialize() below: that init must precede ANY
    # call that brings up the XLA backend, and jax.default_backend() does.
    if getattr(args, "device", "cpu") == "gpu":
        _configure_jax_gpu(args.precision)
    else:
        _configure_jax_cpu(args.precision)

    # --- A1 SPMD mode: federate processes into ONE multi-controller JAX
    # program BEFORE any other JAX use.  jax.distributed only — the
    # mpi4jax halo backend is never armed on this path (mixed stacks
    # deadlock; see --cs-spmd help).  Single-process launches skip the
    # init (jax.distributed requires a real multi-process environment).
    if args.cs_spmd:
        # Resolve the REQUESTED grid before touching jax.distributed:
        # every non-cubed-sphere builder arms the mpi4jax halo backend,
        # and mpi4jax + jax.distributed collectives in one program is
        # the documented mixed-stack deadlock.  (--case is parsed again
        # below; this early peek only needs the grid field.)
        _grid_req = (
            json.loads(args.case).get("grid", args.grid)
            if args.case else args.grid
        )
        if _grid_req != "cubed-sphere":
            print(
                f"ERROR: --cs-spmd supports only --grid cubed-sphere "
                f"(got {_grid_req!r}); lat-lon/icosahedral paths arm "
                "mpi4jax, which must never coexist with "
                "jax.distributed in one program.",
                flush=True,
            )
            return 2
        import jax as _jax
        import os as _os
        # Launcher-agnostic process count via the canonical helper, which
        # covers OpenMPI / PMI / Cray PALS / SLURM (#764: the prior inline
        # subset omitted PALS_LOCAL_SIZE, so a Derecho mpiexec route-B
        # launch skipped initialize() and each rank ran an independent np1
        # mesh — caught loudly by the consistency gate below, but the sweep
        # never federated).
        _nproc = _launcher_world_size()
        if _nproc > 1:
            # PBS/PALS has no bare-initialize auto-detection; the helper
            # falls back to the mpi4py bootstrap (plain MPI, mpi4jax
            # never armed on the --cs-spmd path).
            from legoesm.parallel.early_init import (
                init_jax_distributed_with_fallback,
            )
            init_jax_distributed_with_fallback()

    # --- GPU backend assertion (DEFERRED past --cs-spmd init) ---
    # Now safe to touch the backend: jax.distributed.initialize() (if any) has
    # run.  Fail LOUD on CUDA fallback so a GPU job that silently ran on CPU
    # (CUDA init failed / not bound) can never record CPU numbers labeled GPU
    # (the original g1..g16-ran-on-CPU bug).  Applies to single-GPU and cs-spmd.
    if getattr(args, "device", "cpu") == "gpu":
        import jax as _jax_bk
        _bk = _jax_bk.default_backend()
        if _bk != "gpu":
            raise SystemExit(
                f"--device gpu requested but JAX default backend is {_bk!r} "
                f"(CUDA unavailable / not bound). Refusing to record "
                f"CPU-fallback numbers as GPU. Check CUDA_VISIBLE_DEVICES / "
                f"the cuda jax plugin on this node."
            )

    # --- MPI init ---
    if args.cs_spmd:
        # Route-B: jax.distributed is already federated (initialized above,
        # BEFORE any JAX use) and mpi4jax is never armed — rank identity
        # comes from the runtime, so a CUDA venv without a loadable libmpi
        # is VALID here (job 26449146: the mpi4py loud-guard killed the
        # NCCL cube lane that needs no MPI at all). n_ranks keeps the
        # LAUNCHER world size so the partial-federation gate below still
        # compares jax.process_count() against what was launched.
        import jax as _jax
        _lw = _launcher_world_size()
        rank = _jax.process_index()
        n_ranks = _lw if _lw > 1 else _jax.process_count()
    else:
        rank, n_ranks = _init_mpi()
    is_rank0 = (rank == 0)

    # cs-spmd consistency gate: every launched process must have joined
    # ONE multi-controller program.  A mismatch means initialize() was
    # skipped (unknown launcher) or partially failed — measuring would
    # produce N independent serial runs labelled n_ranks=N.
    if args.cs_spmd:
        import jax as _jax
        if n_ranks > 1 and _jax.process_count() != n_ranks:
            if is_rank0:
                print(
                    f"ERROR: --cs-spmd launched with {n_ranks} MPI "
                    f"processes but jax.process_count()="
                    f"{_jax.process_count()} — jax.distributed did not "
                    "federate them (unsupported launcher?).  Refusing "
                    "to record replicated-serial numbers.",
                    flush=True,
                )
            return 3
        if n_ranks == 1 and is_rank0:
            print(
                "NOTE: --cs-spmd with a single process = sharded-on-1-"
                "device, NOT multi-controller SPMD; use the no-flag "
                "serial path for baselines.",
                flush=True,
            )

    # --- Parse case spec if provided ---
    if args.case:
        case = json.loads(args.case)
        grid_type = case["grid"]
        resolution = case["resolution"]
        physics_level = case["physics"]
        mode = case["mode"]
        nlev = case.get("nlev", args.n_levels)
        precision = case.get("precision", args.precision)
    else:
        grid_type = args.grid
        resolution = args.resolution
        physics_level = args.physics
        mode = args.mode
        nlev = args.n_levels
        precision = args.precision

    # Auto-resolve resolution
    if resolution == 0:
        if grid_type == "cubed-sphere":
            resolution = _weak_resolution_cs(n_ranks) if mode == "weak" else 48
        elif grid_type == "latlon":
            resolution = _weak_resolution_ll(n_ranks) if mode == "weak" else 64
        elif grid_type == "icosahedral":
            resolution = _weak_resolution_ico(n_ranks) if mode == "weak" else 5
        else:
            resolution = 42

    # Non-domain-decomposed grids only support 1 rank in this driver.
    if grid_type == "spectral" and n_ranks > 1:
        if is_rank0:
            print("ERROR: Spectral grid does not support MPI. Use 1 rank.")
        return 1
    # Iter 14 follow-up: defensive runtime guard for cubed-sphere MPI.
    # The iter 13 ``_valid_rank_counts`` guard prevents the sweep from
    # *generating* multi-rank cases, but a user could still pass an
    # explicit ``--case`` with ``n_ranks>1`` or invoke the script under
    # ``mpirun -np N`` with ``--mode single``.  Refuse the
    # configuration up-front instead of silently capturing replicated-
    # dynamics numbers.
    if grid_type == "cubed-sphere" and n_ranks > 1 and not args.cs_spmd:
        if is_rank0:
            print(
                "ERROR: cubed-sphere MPI is currently replicated-"
                "dynamics-only (every rank holds full state).  Use 1 "
                "rank or --grid icosahedral for genuine MPI scaling.",
                flush=True,
            )
        return 2
    # Lat-lon band MPI is real (latitude-band decomposition), but a
    # band must hold at least 2 lat rows for the halo=2 PPM /
    # biharmonic exchange (``pad_halo_latlon_mpi`` raises when
    # ``halo > n_lat_local``).  Refuse undersized configurations
    # up-front with a clear message instead of a mid-build traceback.
    if args.latlon_2d and grid_type != "latlon":
        if is_rank0:
            print("ERROR: --latlon-2d applies only to --grid latlon.",
                  flush=True)
        return 2
    # The >=2-lat-rows-per-rank guard is for the 1-D BAND (all ranks split
    # lat).  The 2-D pencil splits lat over proc_lat (< n_ranks), so its own
    # _factor_2d_latlon validates the per-block rows/cols — skip the band
    # guard for --latlon-2d.
    if (grid_type == "latlon" and not args.latlon_2d
            and n_ranks > 1 and resolution // n_ranks < 2):
        if is_rank0:
            print(
                f"ERROR: lat-lon band MPI needs >=2 lat rows per rank "
                f"(halo=2 exchange); resolution={resolution} on "
                f"{n_ranks} ranks gives {resolution // n_ranks} "
                f"rows/rank.  Increase --resolution or reduce ranks.",
                flush=True,
            )
        return 2

    if is_rank0:
        print("=" * 72)
        print("  legoESM CPU MPI Scaling Benchmark")
        print("=" * 72)
        print(f"  Grid:      {grid_type}")
        print(f"  Physics:   {physics_level}")
        print(f"  Precision: {precision}")
        print(f"  Ranks:     {n_ranks}")
        print(f"  Resolution:{resolution}")
        print(f"  Levels:    {nlev}")
        print(f"  Mode:      {mode}")
        print("=" * 72)

    result = run_single_benchmark(
        grid_type=grid_type,
        resolution=resolution,
        nlev=nlev,
        n_ranks=n_ranks,
        rank=rank,
        precision=precision,
        mode=mode,
        physics_level=physics_level,
        n_warmup=args.n_warmup,
        n_timing=args.n_timing,
        cs_spmd=bool(args.cs_spmd),
        latlon_2d=bool(args.latlon_2d),
    )

    if is_rank0:
        print_summary(result)
        output_dir = Path(args.output_dir) / f"{grid_type}_{physics_level}_{mode}"
        write_result_json(result, output_dir, cs_spmd=bool(args.cs_spmd))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

exec
/usr/bin/bash -lc "rg -n -C 5 'def make_latlon_2d_layout|class.*2D|n_lat.*proc_lat|n_lon.*proc_lon|divisible|LatLon2D' packages/core packages/atmosphere | head -n 600 && printf '\\n--- srun examples resource flags ---\\n' && rg -n --glob '*.sbatch' -- '--nodes=.*--ntasks|--exact|--gpus-per-node|--ntasks-per-node' scripts/cluster/scaling_levante | head -n 600" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 410ms:
packages/core/legoesm/grids/tripole.py-558-                           n_pad: int) -> LatLonCGridGeometry:
packages/core/legoesm/grids/tripole.py-559-    """Prepend ``n_pad`` LAND rows to the SOUTH of a (tri)polar C-grid geometry.
packages/core/legoesm/grids/tripole.py-560-
packages/core/legoesm/grids/tripole.py-561-    The lat-band SPMD ocean step requires ``n_lat % n_devices == 0`` (one uniform
packages/core/legoesm/grids/tripole.py-562-    ``shard_map`` program per band).  A real mesh (eORCA025 ``n_lat=1207``) is not
packages/core/legoesm/grids/tripole.py:563:    divisible by an arbitrary device count, so the driver appends land rows.  They
packages/core/legoesm/grids/tripole.py-564-    go at the SOUTH because the bipolar fold is NORTH-relative
packages/core/legoesm/grids/tripole.py-565-    (``fold_j = n_lat-1``, ``cap_j`` measured down from the north): padding the
packages/core/legoesm/grids/tripole.py-566-    south leaves the fold seam at the (new) northernmost row and the bipolar cap
packages/core/legoesm/grids/tripole.py-567-    band physically unchanged — only its ROW INDEX shifts by ``+n_pad``.
packages/core/legoesm/grids/tripole.py-568-
--
packages/core/legoesm/grids/polar_filter.py-26-# The polar filter is a per-latitude-row longitude rFFT, so it needs the WHOLE
packages/core/legoesm/grids/polar_filter.py-27-# longitude circle on one rank.  Under a longitude split each rank owns only a
packages/core/legoesm/grids/polar_filter.py-28-# lon SLICE, so the filter must first gather the full lon axis, rFFT/mask/irFFT
packages/core/legoesm/grids/polar_filter.py-29-# on it, then scatter its block back.  The 2-D MPI step
packages/core/legoesm/grids/polar_filter.py-30-# (``make_latlon_2d_mpi_step``) injects an AD-safe gather/scatter pair here
packages/core/legoesm/grids/polar_filter.py:31:# (bound to its ``LatLon2DLayout`` + longitude row sub-communicator) BEFORE it
packages/core/legoesm/grids/polar_filter.py-32-# builds the jitted step; the filter reads this process-global at trace time,
packages/core/legoesm/grids/polar_filter.py-33-# exactly like the halo backend (``grids.halo.get_halo_backend``).  This keeps
packages/core/legoesm/grids/polar_filter.py-34-# ``polar_filter`` free of a ``parallel.latlon_mpi`` import (no cycle) and the
packages/core/legoesm/grids/polar_filter.py-35-# dycore call sites (``fourier_filter(field, grid, mask)``) unchanged.  ``None``
packages/core/legoesm/grids/polar_filter.py-36-# (serial / lat-band proc_lon==1 / SPMD) → the local rFFT path below, unchanged.
--
packages/core/legoesm/parallel/latlon_mpi.py-316-        north_rank=rank + 1 if rank < n_ranks - 1 else None,
packages/core/legoesm/parallel/latlon_mpi.py-317-        fold=fold,
packages/core/legoesm/parallel/latlon_mpi.py-318-    )
packages/core/legoesm/parallel/latlon_mpi.py-319-
packages/core/legoesm/parallel/latlon_mpi.py-320-
packages/core/legoesm/parallel/latlon_mpi.py:321:class LatLon2DLayout(NamedTuple):
packages/core/legoesm/parallel/latlon_mpi.py-322-    """2-D pencil (lat × lon) decomposition layout for MPI.
packages/core/legoesm/parallel/latlon_mpi.py-323-
packages/core/legoesm/parallel/latlon_mpi.py-324-    Increment 2 of the lat-lon 2-D decomposition
packages/core/legoesm/parallel/latlon_mpi.py-325-    (``docs/performance/scaling/latlon_2d_decomposition_design.md``).  Generalises
packages/core/legoesm/parallel/latlon_mpi.py-326-    :class:`LatLonBandLayout` from a 1-D latitude band to a 2-D
--
packages/core/legoesm/parallel/latlon_mpi.py-363-    if idx < rem:
packages/core/legoesm/parallel/latlon_mpi.py-364-        return idx * (base + 1), base + 1
packages/core/legoesm/parallel/latlon_mpi.py-365-    return rem * (base + 1) + (idx - rem) * base, base
packages/core/legoesm/parallel/latlon_mpi.py-366-
packages/core/legoesm/parallel/latlon_mpi.py-367-
packages/core/legoesm/parallel/latlon_mpi.py:368:def make_latlon_2d_layout(
packages/core/legoesm/parallel/latlon_mpi.py-369-    rank: int,
packages/core/legoesm/parallel/latlon_mpi.py-370-    proc_lat: int,
packages/core/legoesm/parallel/latlon_mpi.py-371-    proc_lon: int,
packages/core/legoesm/parallel/latlon_mpi.py-372-    n_lat: int,
packages/core/legoesm/parallel/latlon_mpi.py-373-    n_lon: int,
packages/core/legoesm/parallel/latlon_mpi.py-374-    fold: "FoldDescriptor | None" = None,
packages/core/legoesm/parallel/latlon_mpi.py:375:) -> LatLon2DLayout:
packages/core/legoesm/parallel/latlon_mpi.py-376-    """Build a 2-D pencil decomposition layout for ``rank``.
packages/core/legoesm/parallel/latlon_mpi.py-377-
packages/core/legoesm/parallel/latlon_mpi.py-378-    ``rank = proc_row * proc_lon + proc_col`` (row-major).  Latitude is
packages/core/legoesm/parallel/latlon_mpi.py-379-    split over ``proc_lat`` (line, pole-terminated), longitude over
packages/core/legoesm/parallel/latlon_mpi.py-380-    ``proc_lon`` (periodic ring).  ``proc_lat * proc_lon`` must equal the
--
packages/core/legoesm/parallel/latlon_mpi.py-385-            f"proc_lat and proc_lon must be >=1, got "
packages/core/legoesm/parallel/latlon_mpi.py-386-            f"proc_lat={proc_lat}, proc_lon={proc_lon}")
packages/core/legoesm/parallel/latlon_mpi.py-387-    n_ranks = proc_lat * proc_lon
packages/core/legoesm/parallel/latlon_mpi.py-388-    if rank < 0 or rank >= n_ranks:
packages/core/legoesm/parallel/latlon_mpi.py-389-        raise ValueError(f"rank {rank} out of range [0, {n_ranks})")
packages/core/legoesm/parallel/latlon_mpi.py:390:    if n_lat < proc_lat:
packages/core/legoesm/parallel/latlon_mpi.py-391-        raise ValueError(
packages/core/legoesm/parallel/latlon_mpi.py:392:            f"cannot split {n_lat} lat rows over proc_lat={proc_lat} "
packages/core/legoesm/parallel/latlon_mpi.py-393-            "(>=1 row/block required for a 1-cell halo)."
packages/core/legoesm/parallel/latlon_mpi.py-394-        )
packages/core/legoesm/parallel/latlon_mpi.py:395:    if n_lon < proc_lon:
packages/core/legoesm/parallel/latlon_mpi.py-396-        raise ValueError(
packages/core/legoesm/parallel/latlon_mpi.py:397:            f"cannot split {n_lon} lon cols over proc_lon={proc_lon} "
packages/core/legoesm/parallel/latlon_mpi.py-398-            "(>=1 col/block required for a 1-cell halo)."
packages/core/legoesm/parallel/latlon_mpi.py-399-        )
packages/core/legoesm/parallel/latlon_mpi.py-400-
packages/core/legoesm/parallel/latlon_mpi.py-401-    proc_row, proc_col = divmod(rank, proc_lon)
packages/core/legoesm/parallel/latlon_mpi.py:402:    lat_start, n_lat_local = _even_split(n_lat, proc_lat, proc_row)
packages/core/legoesm/parallel/latlon_mpi.py:403:    lon_start, n_lon_local = _even_split(n_lon, proc_lon, proc_col)
packages/core/legoesm/parallel/latlon_mpi.py-404-
packages/core/legoesm/parallel/latlon_mpi.py-405-    def _rank_at(r, c):
packages/core/legoesm/parallel/latlon_mpi.py-406-        return r * proc_lon + c
packages/core/legoesm/parallel/latlon_mpi.py-407-
packages/core/legoesm/parallel/latlon_mpi.py-408-    # Latitude = pole-terminated line: None at the grid's top/bottom row.
--
packages/core/legoesm/parallel/latlon_mpi.py-411-                  if proc_row < proc_lat - 1 else None)
packages/core/legoesm/parallel/latlon_mpi.py-412-    # Longitude = periodic ring: always defined (wrap modulo proc_lon).
packages/core/legoesm/parallel/latlon_mpi.py-413-    west_rank = _rank_at(proc_row, (proc_col - 1) % proc_lon)
packages/core/legoesm/parallel/latlon_mpi.py-414-    east_rank = _rank_at(proc_row, (proc_col + 1) % proc_lon)
packages/core/legoesm/parallel/latlon_mpi.py-415-
packages/core/legoesm/parallel/latlon_mpi.py:416:    return LatLon2DLayout(
packages/core/legoesm/parallel/latlon_mpi.py-417-        rank=rank, n_ranks=n_ranks, proc_lat=proc_lat, proc_lon=proc_lon,
packages/core/legoesm/parallel/latlon_mpi.py-418-        proc_row=proc_row, proc_col=proc_col,
packages/core/legoesm/parallel/latlon_mpi.py-419-        n_lat_global=n_lat, n_lon_global=n_lon,
packages/core/legoesm/parallel/latlon_mpi.py-420-        n_lat_local=n_lat_local, n_lon_local=n_lon_local,
packages/core/legoesm/parallel/latlon_mpi.py-421-        lat_start=lat_start, lat_end=lat_start + n_lat_local,
--
packages/core/legoesm/parallel/latlon_mpi.py-424-        west_rank=west_rank, east_rank=east_rank, fold=fold,
packages/core/legoesm/parallel/latlon_mpi.py-425-    )
packages/core/legoesm/parallel/latlon_mpi.py-426-
packages/core/legoesm/parallel/latlon_mpi.py-427-
packages/core/legoesm/parallel/latlon_mpi.py-428-def scatter_field_latlon_2d(
packages/core/legoesm/parallel/latlon_mpi.py:429:    global_field: jax.Array, layout: LatLon2DLayout,
packages/core/legoesm/parallel/latlon_mpi.py-430-) -> jax.Array:
packages/core/legoesm/parallel/latlon_mpi.py-431-    """Slice the rank's 2-D block ``[lat_start:lat_end, lon_start:lon_end]``
packages/core/legoesm/parallel/latlon_mpi.py-432-    from a global (n_lat, n_lon[, ...]) field.  Deterministic slice (every
packages/core/legoesm/parallel/latlon_mpi.py-433-    rank derives the identical global field) — no MPI."""
packages/core/legoesm/parallel/latlon_mpi.py-434-    return global_field[
--
packages/core/legoesm/parallel/latlon_mpi.py-436-        layout.lon_start:layout.lon_end,
packages/core/legoesm/parallel/latlon_mpi.py-437-    ]
packages/core/legoesm/parallel/latlon_mpi.py-438-
packages/core/legoesm/parallel/latlon_mpi.py-439-
packages/core/legoesm/parallel/latlon_mpi.py-440-def gather_field_latlon_2d(
packages/core/legoesm/parallel/latlon_mpi.py:441:    local_field, layout: LatLon2DLayout,
packages/core/legoesm/parallel/latlon_mpi.py-442-    *, is_u_face: bool = False, is_v_face: bool = False,
packages/core/legoesm/parallel/latlon_mpi.py-443-):
packages/core/legoesm/parallel/latlon_mpi.py-444-    """Reassemble the global field from every rank's 2-D block (I/O only).
packages/core/legoesm/parallel/latlon_mpi.py-445-
packages/core/legoesm/parallel/latlon_mpi.py-446-    mpi4py ``allgather`` of host blocks, placed by (proc_row, proc_col).
--
packages/core/legoesm/parallel/latlon_mpi.py-497-    for ls, los, blk in blocks:
packages/core/legoesm/parallel/latlon_mpi.py-498-        out[ls:ls + blk.shape[0], los:los + blk.shape[1]] = blk
packages/core/legoesm/parallel/latlon_mpi.py-499-    return out
packages/core/legoesm/parallel/latlon_mpi.py-500-
packages/core/legoesm/parallel/latlon_mpi.py-501-
packages/core/legoesm/parallel/latlon_mpi.py:502:def make_lon_row_comm(layout: LatLon2DLayout):
packages/core/legoesm/parallel/latlon_mpi.py-503-    """Create the longitude-ring sub-communicator for this rank's
packages/core/legoesm/parallel/latlon_mpi.py-504-    ``proc_row`` (the ``proc_lon`` ranks that share a latitude band).
packages/core/legoesm/parallel/latlon_mpi.py-505-
packages/core/legoesm/parallel/latlon_mpi.py-506-    COLLECTIVE over ``COMM_WORLD`` (every rank must call it).  ``key=
packages/core/legoesm/parallel/latlon_mpi.py-507-    proc_col`` orders the sub-comm ranks by longitude block, so an
--
packages/core/legoesm/parallel/latlon_mpi.py-516-
packages/core/legoesm/parallel/latlon_mpi.py-517-# custom_vjp on SCALAR lon metadata only (NOT the whole layout): the
packages/core/legoesm/parallel/latlon_mpi.py-518-# layout's ``fold`` field can carry jax.Array permutations, which must
packages/core/legoesm/parallel/latlon_mpi.py-519-# not become custom-VJP static args (codex MAJOR 2026-06-13).  The public
packages/core/legoesm/parallel/latlon_mpi.py-520-# wrapper below extracts the scalar fields.  nondiff args = lon_start,
packages/core/legoesm/parallel/latlon_mpi.py:521:# lon_end, n_lon_global, proc_lon, row_comm (indices 1..5).
packages/core/legoesm/parallel/latlon_mpi.py-522-@functools.partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3, 4, 5))
packages/core/legoesm/parallel/latlon_mpi.py-523-def _lon_gather_full_p(local_block, lon_start, lon_end, n_lon_global,
packages/core/legoesm/parallel/latlon_mpi.py-524-                       proc_lon, row_comm):
packages/core/legoesm/parallel/latlon_mpi.py-525-    from legoesm.parallel.reductions import mpi4jax_array_result, require_mpi_stack
packages/core/legoesm/parallel/latlon_mpi.py-526-
packages/core/legoesm/parallel/latlon_mpi.py-527-    # Checked accessor (not a bare ``import mpi4jax``): runs the GPU-transport
packages/core/legoesm/parallel/latlon_mpi.py-528-    # preflight so this device-array allgather can't slip a GPU-direct
packages/core/legoesm/parallel/latlon_mpi.py-529-    # misconfiguration past the fail-closed check.
packages/core/legoesm/parallel/latlon_mpi.py-530-    mpi4jax, _ = require_mpi_stack()
packages/core/legoesm/parallel/latlon_mpi.py-531-
packages/core/legoesm/parallel/latlon_mpi.py:532:    if n_lon_global % proc_lon != 0:
packages/core/legoesm/parallel/latlon_mpi.py-533-        raise ValueError(
packages/core/legoesm/parallel/latlon_mpi.py-534-            f"lon_gather_full needs an equal lon split (n_lon="
packages/core/legoesm/parallel/latlon_mpi.py:535:            f"{n_lon_global} % proc_lon={proc_lon} != 0) for the "
packages/core/legoesm/parallel/latlon_mpi.py-536-            "allgather; non-uniform splits need alltoallv (not yet "
packages/core/legoesm/parallel/latlon_mpi.py-537-            "implemented)."
packages/core/legoesm/parallel/latlon_mpi.py-538-        )
packages/core/legoesm/parallel/latlon_mpi.py-539-    # (proc_lon, n_lat_local, n_lon_local[, nlev]); key=proc_col ordered
packages/core/legoesm/parallel/latlon_mpi.py-540-    # the sub-comm ranks west->east, so axis 0 IS lon-block order.
--
packages/core/legoesm/parallel/latlon_mpi.py-545-
packages/core/legoesm/parallel/latlon_mpi.py-546-def _lon_gather_full_p_fwd(local_block, lon_start, lon_end, n_lon_global,
packages/core/legoesm/parallel/latlon_mpi.py-547-                           proc_lon, row_comm):
packages/core/legoesm/parallel/latlon_mpi.py-548-    # No residual: the adjoint depends only on the (static) scalar args.
packages/core/legoesm/parallel/latlon_mpi.py-549-    return _lon_gather_full_p(
packages/core/legoesm/parallel/latlon_mpi.py:550:        local_block, lon_start, lon_end, n_lon_global, proc_lon, row_comm), None
packages/core/legoesm/parallel/latlon_mpi.py-551-
packages/core/legoesm/parallel/latlon_mpi.py-552-
packages/core/legoesm/parallel/latlon_mpi.py:553:def _lon_gather_full_p_bwd(lon_start, lon_end, n_lon_global, proc_lon,
packages/core/legoesm/parallel/latlon_mpi.py-554-                           row_comm, _res, g_full):
packages/core/legoesm/parallel/latlon_mpi.py-555-    """Adjoint of the lat-pencil gather.
packages/core/legoesm/parallel/latlon_mpi.py-556-
packages/core/legoesm/parallel/latlon_mpi.py-557-    Forward ``G`` maps each rank's block ``x_s`` into the full-lon array
packages/core/legoesm/parallel/latlon_mpi.py-558-    that is REPLICATED across the whole row ring (``y_r = concat_s x_s``
--
packages/core/legoesm/parallel/latlon_mpi.py-574-
packages/core/legoesm/parallel/latlon_mpi.py-575-
packages/core/legoesm/parallel/latlon_mpi.py-576-_lon_gather_full_p.defvjp(_lon_gather_full_p_fwd, _lon_gather_full_p_bwd)
packages/core/legoesm/parallel/latlon_mpi.py-577-
packages/core/legoesm/parallel/latlon_mpi.py-578-
packages/core/legoesm/parallel/latlon_mpi.py:579:def lon_gather_full(local_block: jax.Array, layout: LatLon2DLayout,
packages/core/legoesm/parallel/latlon_mpi.py-580-                    row_comm) -> jax.Array:
packages/core/legoesm/parallel/latlon_mpi.py-581-    """In-trace lat-pencil transpose: assemble the FULL longitude axis
packages/core/legoesm/parallel/latlon_mpi.py-582-    for this rank's latitude band from the ``proc_lon`` lon-ring blocks.
packages/core/legoesm/parallel/latlon_mpi.py-583-
packages/core/legoesm/parallel/latlon_mpi.py-584-    The crux primitive for any operation that needs all longitudes on a
--
packages/core/legoesm/parallel/latlon_mpi.py-592-    AD-SAFE (custom VJP on :func:`_lon_gather_full_p`): the forward is
packages/core/legoesm/parallel/latlon_mpi.py-593-    ``allgather`` (no native VJP), but the gather is a linear map whose
packages/core/legoesm/parallel/latlon_mpi.py-594-    adjoint is exact — a local block contributes to the full-lon array
packages/core/legoesm/parallel/latlon_mpi.py-595-    on EVERY rank of the row ring, so the cotangent's adjoint is
packages/core/legoesm/parallel/latlon_mpi.py-596-    ``allreduce(SUM)`` over the row ring (AD-safe) then a slice of this
packages/core/legoesm/parallel/latlon_mpi.py:597:    rank's lon block.  Requires an EQUAL lon split (``n_lon % proc_lon
packages/core/legoesm/parallel/latlon_mpi.py-598-    == 0``) so the allgather blocks share a shape; non-uniform splits
packages/core/legoesm/parallel/latlon_mpi.py-599-    need an ``alltoallv`` (future).
packages/core/legoesm/parallel/latlon_mpi.py-600-
packages/core/legoesm/parallel/latlon_mpi.py-601-    COLLECTIVE PRECONDITION (deadlock safety): both the forward gather
packages/core/legoesm/parallel/latlon_mpi.py-602-    AND its reverse-mode ``allreduce`` are collectives over ``row_comm``,
--
packages/core/legoesm/parallel/latlon_mpi.py-607-    rank-selective use, keep the call in the traced graph on all ranks
packages/core/legoesm/parallel/latlon_mpi.py-608-    and gate with arithmetic masks (zero cotangents where inactive).
packages/core/legoesm/parallel/latlon_mpi.py-609-    """
packages/core/legoesm/parallel/latlon_mpi.py-610-    return _lon_gather_full_p(
packages/core/legoesm/parallel/latlon_mpi.py-611-        local_block, layout.lon_start, layout.lon_end,
packages/core/legoesm/parallel/latlon_mpi.py:612:        layout.n_lon_global, layout.proc_lon, row_comm)
packages/core/legoesm/parallel/latlon_mpi.py-613-
packages/core/legoesm/parallel/latlon_mpi.py-614-
packages/core/legoesm/parallel/latlon_mpi.py-615-def lon_scatter_full(full_field: jax.Array,
packages/core/legoesm/parallel/latlon_mpi.py:616:                     layout: LatLon2DLayout) -> jax.Array:
packages/core/legoesm/parallel/latlon_mpi.py-617-    """Slice this rank's longitude block ``[lon_start:lon_end]`` from a
packages/core/legoesm/parallel/latlon_mpi.py-618-    full-longitude field — inverse of :func:`lon_gather_full`."""
packages/core/legoesm/parallel/latlon_mpi.py-619-    return full_field[:, layout.lon_start:layout.lon_end]
packages/core/legoesm/parallel/latlon_mpi.py-620-
packages/core/legoesm/parallel/latlon_mpi.py-621-
--
packages/core/legoesm/parallel/latlon_mpi.py-982-    — there are no pole/wall ends: every rank in the longitude ring
packages/core/legoesm/parallel/latlon_mpi.py-983-    sends its west edge west and its east edge east, and the wrap is
packages/core/legoesm/parallel/latlon_mpi.py-984-    just the ring topology (the rank owning the last lon block has the
packages/core/legoesm/parallel/latlon_mpi.py-985-    rank owning the first as its east neighbour).  This is a STANDALONE
packages/core/legoesm/parallel/latlon_mpi.py-986-    primitive (takes plain neighbour ranks, not a layout) so the future
packages/core/legoesm/parallel/latlon_mpi.py:987:    ``LatLon2DLayout`` can call it with ``layout.west_rank`` etc.; it is
packages/core/legoesm/parallel/latlon_mpi.py-988-    NOT yet wired into any step.
packages/core/legoesm/parallel/latlon_mpi.py-989-
packages/core/legoesm/parallel/latlon_mpi.py-990-    When the longitude ring has a single member (``west_rank == east_rank
packages/core/legoesm/parallel/latlon_mpi.py-991-    == rank`` — i.e. ``proc_lon == 1``, the current 1-D latitude-band
packages/core/legoesm/parallel/latlon_mpi.py-992-    case) the wrap is performed LOCALLY from the rank's own columns,
--
packages/core/legoesm/parallel/latlon_mpi.py-1013-
packages/core/legoesm/parallel/latlon_mpi.py-1014-    n_lon_local = field.shape[1]
packages/core/legoesm/parallel/latlon_mpi.py-1015-    if halo > n_lon_local:
packages/core/legoesm/parallel/latlon_mpi.py-1016-        raise ValueError(
packages/core/legoesm/parallel/latlon_mpi.py-1017-            f"exchange_halo_lon: halo={halo} exceeds n_lon_local="
packages/core/legoesm/parallel/latlon_mpi.py:1018:            f"{n_lon_local} on rank {rank}.  Reduce proc_lon or "
packages/core/legoesm/parallel/latlon_mpi.py-1019-            "increase longitude resolution."
packages/core/legoesm/parallel/latlon_mpi.py-1020-        )
packages/core/legoesm/parallel/latlon_mpi.py-1021-
packages/core/legoesm/parallel/latlon_mpi.py-1022-    # Single-member lon ring (proc_lon == 1): local periodic wrap — the
packages/core/legoesm/parallel/latlon_mpi.py-1023-    # west ghost is the rank's own EAST edge, the east ghost its WEST
--
packages/core/legoesm/parallel/latlon_mpi.py-1096-    return jnp.concatenate([west_halo, field, east_halo], axis=1)
packages/core/legoesm/parallel/latlon_mpi.py-1097-
packages/core/legoesm/parallel/latlon_mpi.py-1098-
packages/core/legoesm/parallel/latlon_mpi.py-1099-def _pad_lat_wall_2d(
packages/core/legoesm/parallel/latlon_mpi.py-1100-    field: jax.Array,
packages/core/legoesm/parallel/latlon_mpi.py:1101:    layout: LatLon2DLayout,
packages/core/legoesm/parallel/latlon_mpi.py-1102-    halo: int,
packages/core/legoesm/parallel/latlon_mpi.py-1103-    south_value: float,
packages/core/legoesm/parallel/latlon_mpi.py-1104-    north_value: float,
packages/core/legoesm/parallel/latlon_mpi.py-1105-) -> jax.Array:
packages/core/legoesm/parallel/latlon_mpi.py-1106-    """Lat-axis (N/S) wall pad for a 2-D pencil row — the shared core of
--
packages/core/legoesm/parallel/latlon_mpi.py-1118-    """
packages/core/legoesm/parallel/latlon_mpi.py-1119-    # halo must fit the SMALLEST local lat block: a neighbour owning fewer
packages/core/legoesm/parallel/latlon_mpi.py-1120-    # than `halo` rows would send/recv a mismatched slab and the exchange
packages/core/legoesm/parallel/latlon_mpi.py-1121-    # would truncate/hang (same guard rationale as pad_halo_latlon_2d's
packages/core/legoesm/parallel/latlon_mpi.py-1122-    # lon side).  _even_split gives the first blocks one extra row, so the
packages/core/legoesm/parallel/latlon_mpi.py:1123:    # smallest block is floor(n_lat_global / proc_lat).
packages/core/legoesm/parallel/latlon_mpi.py:1124:    min_lat_block = layout.n_lat_global // layout.proc_lat
packages/core/legoesm/parallel/latlon_mpi.py-1125-    if halo > min_lat_block:
packages/core/legoesm/parallel/latlon_mpi.py-1126-        raise ValueError(
packages/core/legoesm/parallel/latlon_mpi.py-1127-            f"_pad_lat_wall_2d: halo={halo} exceeds the smallest local lat "
packages/core/legoesm/parallel/latlon_mpi.py-1128-            f"block ({min_lat_block}=n_lat_global {layout.n_lat_global}//"
packages/core/legoesm/parallel/latlon_mpi.py-1129-            f"proc_lat {layout.proc_lat}); a neighbour would send/recv a "
--
packages/core/legoesm/parallel/latlon_mpi.py-1169-    return jnp.concatenate([south, field, north], axis=0)
packages/core/legoesm/parallel/latlon_mpi.py-1170-
packages/core/legoesm/parallel/latlon_mpi.py-1171-
packages/core/legoesm/parallel/latlon_mpi.py-1172-def pad_with_pole_bc_lat_2d(
packages/core/legoesm/parallel/latlon_mpi.py-1173-    interior: jax.Array,
packages/core/legoesm/parallel/latlon_mpi.py:1174:    layout: LatLon2DLayout,
packages/core/legoesm/parallel/latlon_mpi.py-1175-    halo: int = 1,
packages/core/legoesm/parallel/latlon_mpi.py-1176-    south_value: float = 0.0,
packages/core/legoesm/parallel/latlon_mpi.py-1177-    north_value: float = 0.0,
packages/core/legoesm/parallel/latlon_mpi.py-1178-) -> jax.Array:
packages/core/legoesm/parallel/latlon_mpi.py-1179-    """Lat-axis-ONLY wall pad for a 2-D pencil layout (NO longitude halo).
--
packages/core/legoesm/parallel/latlon_mpi.py-1183-    only the lat axis (interior cut sendrecv + pole wall constant), leaving
packages/core/legoesm/parallel/latlon_mpi.py-1184-    longitude untouched.  The band path never split longitude, so its
packages/core/legoesm/parallel/latlon_mpi.py-1185-    wall-BC pad is lat-only; the 2-D pencil keeps that contract and the
packages/core/legoesm/parallel/latlon_mpi.py-1186-    operator adds lon ghosts through its own dispatched lon halo.  Routed
packages/core/legoesm/parallel/latlon_mpi.py-1187-    here from ``halo_latlon.pad_with_pole_bc_lat`` when the active topology
packages/core/legoesm/parallel/latlon_mpi.py:1188:    is a :class:`LatLon2DLayout` (wall poles only; the tripolar north fold /
packages/core/legoesm/parallel/latlon_mpi.py-1189-    vector-u seam is excluded upstream).  AD-safe via the shared sendrecv
packages/core/legoesm/parallel/latlon_mpi.py-1190-    VJP.
packages/core/legoesm/parallel/latlon_mpi.py-1191-    """
packages/core/legoesm/parallel/latlon_mpi.py-1192-    if halo <= 0:
packages/core/legoesm/parallel/latlon_mpi.py-1193-        return interior
packages/core/legoesm/parallel/latlon_mpi.py-1194-    return _pad_lat_wall_2d(interior, layout, halo, south_value, north_value)
packages/core/legoesm/parallel/latlon_mpi.py-1195-
packages/core/legoesm/parallel/latlon_mpi.py-1196-
packages/core/legoesm/parallel/latlon_mpi.py-1197-def pad_halo_latlon_2d(
packages/core/legoesm/parallel/latlon_mpi.py-1198-    field: jax.Array,
packages/core/legoesm/parallel/latlon_mpi.py:1199:    layout: LatLon2DLayout,
packages/core/legoesm/parallel/latlon_mpi.py-1200-    halo: int = 1,
packages/core/legoesm/parallel/latlon_mpi.py-1201-    pole_bc: str = "wall",
packages/core/legoesm/parallel/latlon_mpi.py-1202-    south_value: float = 0.0,
packages/core/legoesm/parallel/latlon_mpi.py-1203-    north_value: float = 0.0,
packages/core/legoesm/parallel/latlon_mpi.py-1204-) -> jax.Array:
--
packages/core/legoesm/parallel/latlon_mpi.py-1248-    # its send/recv would be shorter than this rank expects and the MPI
packages/core/legoesm/parallel/latlon_mpi.py-1249-    # exchange truncates/aborts/hangs before any reshape (codex MAJOR
packages/core/legoesm/parallel/latlon_mpi.py-1250-    # 2026-06-13). Smallest block on an even-ish split = floor(n/proc).
packages/core/legoesm/parallel/latlon_mpi.py-1251-    # _even_split gives the first (n % parts) blocks one extra row/col, so
packages/core/legoesm/parallel/latlon_mpi.py-1252-    # the SMALLEST block is exactly floor(n_global / parts).
packages/core/legoesm/parallel/latlon_mpi.py:1253:    min_lat_block = layout.n_lat_global // layout.proc_lat
packages/core/legoesm/parallel/latlon_mpi.py:1254:    min_lon_block = layout.n_lon_global // layout.proc_lon
packages/core/legoesm/parallel/latlon_mpi.py-1255-    if halo > min_lat_block or halo > min_lon_block:
packages/core/legoesm/parallel/latlon_mpi.py-1256-        raise ValueError(
packages/core/legoesm/parallel/latlon_mpi.py-1257-            f"pad_halo_latlon_2d: halo={halo} exceeds the smallest local "
packages/core/legoesm/parallel/latlon_mpi.py-1258-            f"block (lat {min_lat_block}=n_lat_global "
packages/core/legoesm/parallel/latlon_mpi.py:1259:            f"{layout.n_lat_global}//proc_lat {layout.proc_lat}, lon "
packages/core/legoesm/parallel/latlon_mpi.py:1260:            f"{min_lon_block}=n_lon_global {layout.n_lon_global}//proc_lon "
packages/core/legoesm/parallel/latlon_mpi.py-1261-            f"{layout.proc_lon}); a neighbour would send/recv a mismatched "
packages/core/legoesm/parallel/latlon_mpi.py-1262-            f"halo and the MPI exchange would abort/hang.")
packages/core/legoesm/parallel/latlon_mpi.py-1263-
packages/core/legoesm/parallel/latlon_mpi.py-1264-    # N/S — lat-axis wall pad (interior cut sendrecv at the AD-safe
packages/core/legoesm/parallel/latlon_mpi.py-1265-    # rank-as-tag LINE pattern, pole side wall); shared verbatim with
--
packages/core/legoesm/parallel/latlon_mpi.py-1895-        phis=phis_local,
packages/core/legoesm/parallel/latlon_mpi.py-1896-        tracers=tracers_local if tracers_local else state.tracers,
packages/core/legoesm/parallel/latlon_mpi.py-1897-    )
packages/core/legoesm/parallel/latlon_mpi.py-1898-
packages/core/legoesm/parallel/latlon_mpi.py-1899-
packages/core/legoesm/parallel/latlon_mpi.py:1900:def scatter_state_latlon_2d(state, layout: LatLon2DLayout):
packages/core/legoesm/parallel/latlon_mpi.py-1901-    """Extract the rank-local 2-D block from a global C-grid lat-lon state.
packages/core/legoesm/parallel/latlon_mpi.py-1902-
packages/core/legoesm/parallel/latlon_mpi.py-1903-    The 2-D analog of :func:`scatter_state_latlon`: slice BOTH the latitude
packages/core/legoesm/parallel/latlon_mpi.py-1904-    band ``[lat_start, lat_end)`` and the longitude pencil
packages/core/legoesm/parallel/latlon_mpi.py-1905-    ``[lon_start, lon_end)``.  Pure indexing — no MPI (each rank slices its own
--
packages/core/legoesm/parallel/latlon_mpi.py-2455-        total_area=global_total,
packages/core/legoesm/parallel/latlon_mpi.py-2456-    )
packages/core/legoesm/parallel/latlon_mpi.py-2457-
packages/core/legoesm/parallel/latlon_mpi.py-2458-
packages/core/legoesm/parallel/latlon_mpi.py-2459-def slice_latlon_grid_to_block_2d(
packages/core/legoesm/parallel/latlon_mpi.py:2460:    grid, layout: LatLon2DLayout, *, skip_total_area_reduce: bool = False,
packages/core/legoesm/parallel/latlon_mpi.py-2461-):
packages/core/legoesm/parallel/latlon_mpi.py-2462-    """Slice a global ``LatLonGrid`` to this rank's 2-D lat x lon block.
packages/core/legoesm/parallel/latlon_mpi.py-2463-
packages/core/legoesm/parallel/latlon_mpi.py-2464-    The 2-D analog of :func:`slice_latlon_grid_to_band`: every lat-dependent
packages/core/legoesm/parallel/latlon_mpi.py-2465-    array is sliced ``[lat_start:lat_end]`` AND every lon-dependent array
--
packages/core/legoesm/parallel/latlon_mpi.py-2892-    return step_fn
packages/core/legoesm/parallel/latlon_mpi.py-2893-
packages/core/legoesm/parallel/latlon_mpi.py-2894-
packages/core/legoesm/parallel/latlon_mpi.py-2895-def make_latlon_2d_mpi_step(
packages/core/legoesm/parallel/latlon_mpi.py-2896-    model,
packages/core/legoesm/parallel/latlon_mpi.py:2897:    layout: LatLon2DLayout,
packages/core/legoesm/parallel/latlon_mpi.py-2898-    *,
packages/core/legoesm/parallel/latlon_mpi.py-2899-    physics_fn: Callable | None = None,
packages/core/legoesm/parallel/latlon_mpi.py-2900-) -> Callable:
packages/core/legoesm/parallel/latlon_mpi.py-2901-    """Build an MPI step for the lat-lon C-grid dycore on a 2-D pencil.
packages/core/legoesm/parallel/latlon_mpi.py-2902-
packages/core/legoesm/parallel/latlon_mpi.py-2903-    The 2-D ``(proc_lat × proc_lon)`` analogue of :func:`make_latlon_mpi_step`.
packages/core/legoesm/parallel/latlon_mpi.py-2904-    Identical architecture — activate the MPI halo backend once with the
packages/core/legoesm/parallel/latlon_mpi.py:2905:    :class:`LatLon2DLayout`, build a rank-local model with the global
packages/core/legoesm/parallel/latlon_mpi.py-2906-    sphere area (mass-fixer divisor) and ``pole_v_bc`` tracking which lat
packages/core/legoesm/parallel/latlon_mpi.py-2907-    ends touch a physical pole, then delegate each step to
packages/core/legoesm/parallel/latlon_mpi.py-2908-    ``model._step_cgrid`` whose backend-aware operators fetch halo data via
packages/core/legoesm/parallel/latlon_mpi.py-2909-    the 2-D dispatch (``pad_halo_latlon`` → :func:`pad_halo_latlon_2d`,
packages/core/legoesm/parallel/latlon_mpi.py-2910-    ``pad_with_pole_bc_lat`` → :func:`pad_with_pole_bc_lat_2d`,
--
packages/core/legoesm/parallel/latlon_mpi.py-2972-        getattr(model.config, "use_polar_filter", False)))
packages/core/legoesm/parallel/latlon_mpi.py-2973-    # Validate the EQUAL-split precondition of the polar-filter lon-gather
packages/core/legoesm/parallel/latlon_mpi.py-2974-    # allgather HERE — before any mutation (set_halo_backend / the global
packages/core/legoesm/parallel/latlon_mpi.py-2975-    # area allreduce / the model build) — so a rejected uneven split leaves
packages/core/legoesm/parallel/latlon_mpi.py-2976-    # the process UNCONFIGURED (codex C7 P2), not half-set-up.
packages/core/legoesm/parallel/latlon_mpi.py:2977:    if _polar_lon_split and layout.n_lon_global % layout.proc_lon != 0:
packages/core/legoesm/parallel/latlon_mpi.py-2978-        raise NotImplementedError(
packages/core/legoesm/parallel/latlon_mpi.py-2979-            "make_latlon_2d_mpi_step: use_polar_filter=True under a longitude "
packages/core/legoesm/parallel/latlon_mpi.py-2980-            f"split needs an EQUAL split (n_lon={layout.n_lon_global} % "
packages/core/legoesm/parallel/latlon_mpi.py-2981-            f"proc_lon={layout.proc_lon} != 0) — the polar-filter lon-gather "
packages/core/legoesm/parallel/latlon_mpi.py-2982-            "allgather requires equal blocks.  Use a proc_lon that divides "
--
packages/atmosphere/legoesm/atmosphere/physics/turbulence/override_sharding.py-26-
packages/atmosphere/legoesm/atmosphere/physics/turbulence/override_sharding.py-27-def localize_turbulence_override(override: Any, layout: Any) -> Any:
packages/atmosphere/legoesm/atmosphere/physics/turbulence/override_sharding.py-28-    """Slice a GLOBAL per-column ``clubb_lite`` override to ``layout``'s rank tile.
packages/atmosphere/legoesm/atmosphere/physics/turbulence/override_sharding.py-29-
packages/atmosphere/legoesm/atmosphere/physics/turbulence/override_sharding.py-30-    ``layout`` is the active MPI layout (``LatLonBandLayout`` — a latitude band, or
packages/atmosphere/legoesm/atmosphere/physics/turbulence/override_sharding.py:31:    ``LatLon2DLayout`` — a lat×lon pencil).  Returns ``override`` UNCHANGED when no
packages/atmosphere/legoesm/atmosphere/physics/turbulence/override_sharding.py-32-    slicing is needed (non-clubb scheme, all-scalar fields, or per-column fields not
packages/atmosphere/legoesm/atmosphere/physics/turbulence/override_sharding.py-33-    at the global column count — i.e. already rank-local).  Supports lat-lon
packages/atmosphere/legoesm/atmosphere/physics/turbulence/override_sharding.py:34:    (``LatLonBandLayout`` / ``LatLon2DLayout``), cubed-sphere (``DistributedLayout``
packages/atmosphere/legoesm/atmosphere/physics/turbulence/override_sharding.py-35-    face-only / tiled, via the model's ``scatter``), AND MPAS/Voronoi
packages/atmosphere/legoesm/atmosphere/physics/turbulence/override_sharding.py-36-    (``VoronoiPartitionLayout``, gathered at the rank's ``local_cells``).  An
packages/atmosphere/legoesm/atmosphere/physics/turbulence/override_sharding.py-37-    UNRECOGNIZED decomposition passes the override THROUGH unchanged (honoring an
packages/atmosphere/legoesm/atmosphere/physics/turbulence/override_sharding.py-38-    advanced user who pre-sliced per rank via
packages/atmosphere/legoesm/atmosphere/physics/turbulence/override_sharding.py-39-    :func:`legoesm.training.deploy_correction.slice_override_columns`); a GLOBAL
--
packages/atmosphere/legoesm/atmosphere/physics/turbulence/override_sharding.py-118-    owned+halo ``local_cells``), so the local override lands on EXACTLY the columns
packages/atmosphere/legoesm/atmosphere/physics/turbulence/override_sharding.py-119-    the rank's physics consumes.  ``None`` ⇒ an unrecognized decomposition.
packages/atmosphere/legoesm/atmosphere/physics/turbulence/override_sharding.py-120-    """
packages/atmosphere/legoesm/atmosphere/physics/turbulence/override_sharding.py-121-    import jax.numpy as jnp
packages/atmosphere/legoesm/atmosphere/physics/turbulence/override_sharding.py-122-    from legoesm.parallel.comm import CommTopology
packages/atmosphere/legoesm/atmosphere/physics/turbulence/override_sharding.py:123:    from legoesm.parallel.latlon_mpi import LatLon2DLayout, LatLonBandLayout
packages/atmosphere/legoesm/atmosphere/physics/turbulence/override_sharding.py-124-    from legoesm.parallel.layout import (
packages/atmosphere/legoesm/atmosphere/physics/turbulence/override_sharding.py-125-        DistributedLayout,
packages/atmosphere/legoesm/atmosphere/physics/turbulence/override_sharding.py-126-        SingleRankLayout,
packages/atmosphere/legoesm/atmosphere/physics/turbulence/override_sharding.py-127-        scatter,
packages/atmosphere/legoesm/atmosphere/physics/turbulence/override_sharding.py-128-    )
packages/atmosphere/legoesm/atmosphere/physics/turbulence/override_sharding.py-129-    from legoesm.parallel.voronoi_mpi import VoronoiPartitionLayout
packages/atmosphere/legoesm/atmosphere/physics/turbulence/override_sharding.py-130-
packages/atmosphere/legoesm/atmosphere/physics/turbulence/override_sharding.py:131:    if isinstance(layout, LatLon2DLayout):
packages/atmosphere/legoesm/atmosphere/physics/turbulence/override_sharding.py-132-        return ((layout.n_lat_global, layout.n_lon_global),
packages/atmosphere/legoesm/atmosphere/physics/turbulence/override_sharding.py-133-                lambda f: f[layout.lat_start:layout.lat_end,
packages/atmosphere/legoesm/atmosphere/physics/turbulence/override_sharding.py-134-                            layout.lon_start:layout.lon_end])
packages/atmosphere/legoesm/atmosphere/physics/turbulence/override_sharding.py-135-    if isinstance(layout, LatLonBandLayout):
packages/atmosphere/legoesm/atmosphere/physics/turbulence/override_sharding.py-136-        return ((layout.n_lat_global, layout.n_lon_global),
--
packages/core/legoesm/parallel/mesh.py-820-                    sharding = tiled_face_only or config.face_sharding
packages/core/legoesm/parallel/mesh.py-821-                    return multiprocess_safe_device_put(leaf, sharding)
packages/core/legoesm/parallel/mesh.py-822-                if config.tiling != (1, 1) and leaf.ndim >= 3:
packages/core/legoesm/parallel/mesh.py-823-                    # STAGGERED face-plane leaves — D-grid winds
packages/core/legoesm/parallel/mesh.py-824-                    # (6, n+1, n, ...) / (6, n, n+1, ...) — cannot
packages/core/legoesm/parallel/mesh.py:825:                    # shard over the tile axes (IndivisibleError, P4
packages/core/legoesm/parallel/mesh.py-826-                    # discovery probe job 8464703).  Phase-1: face-only
packages/core/legoesm/parallel/mesh.py-827-                    # sharding (spatial replicated across a face's kt^2
packages/core/legoesm/parallel/mesh.py-828-                    # tile devices); the tiled shard_map stage slices
packages/core/legoesm/parallel/mesh.py-829-                    # its LOCAL duplicated-shared-row block body-side
packages/core/legoesm/parallel/mesh.py-830-                    # (Pace layout) via staggered_face_to_tile_blocks.
--
packages/core/legoesm/parallel/mesh.py-1154-    import jax.numpy as jnp
packages/core/legoesm/parallel/mesh.py-1155-
packages/core/legoesm/parallel/mesh.py-1156-    n = cdgrid.base.n
packages/core/legoesm/parallel/mesh.py-1157-    if n % kt != 0:
packages/core/legoesm/parallel/mesh.py-1158-        raise ValueError(
packages/core/legoesm/parallel/mesh.py:1159:            f"face resolution n={n} not divisible by kt={kt}")
packages/core/legoesm/parallel/mesh.py-1160-    nl = n // kt
packages/core/legoesm/parallel/mesh.py-1161-
packages/core/legoesm/parallel/mesh.py-1162-    def _walk(obj, prefix=""):
packages/core/legoesm/parallel/mesh.py-1163-        fields = getattr(obj, "_fields", None)
packages/core/legoesm/parallel/mesh.py-1164-        if fields is None:
--
packages/core/legoesm/parallel/voronoi_partition.py-902-# ============================================================================
packages/core/legoesm/parallel/voronoi_partition.py-903-# Mesh padding for even sharding
packages/core/legoesm/parallel/voronoi_partition.py-904-# ============================================================================
packages/core/legoesm/parallel/voronoi_partition.py-905-
packages/core/legoesm/parallel/voronoi_partition.py-906-def _pad_voronoi_for_sharding(mesh: VoronoiMesh, n_devices: int) -> VoronoiMesh:
packages/core/legoesm/parallel/voronoi_partition.py:907:    """Pad cell/edge arrays so their sizes are divisible by *n_devices*.
packages/core/legoesm/parallel/voronoi_partition.py-908-
packages/core/legoesm/parallel/voronoi_partition.py-909-    Adds ghost cells/edges that are inert in physics:
packages/core/legoesm/parallel/voronoi_partition.py-910-    - Ghost cells: ``areaCell=1`` (avoids 0/0 NaN in divergence), all
packages/core/legoesm/parallel/voronoi_partition.py-911-      connectivity = -1 (masked by operators), signs/weights = 0.
packages/core/legoesm/parallel/voronoi_partition.py-912-    - Ghost edges: ``dvEdge=0`` (zero flux contribution), ``dcEdge=1``
--
packages/core/legoesm/parallel/voronoi_partition.py-1148-        edgeSignOnCell=reorder_col(mesh.edgeSignOnCell, cell_perm),
packages/core/legoesm/parallel/voronoi_partition.py-1149-        edgeSignOnVertex=reorder_col(mesh.edgeSignOnVertex, vert_perm),
packages/core/legoesm/parallel/voronoi_partition.py-1150-        meshDensity=reorder_1d(mesh.meshDensity, cell_perm),
packages/core/legoesm/parallel/voronoi_partition.py-1151-    )
packages/core/legoesm/parallel/voronoi_partition.py-1152-
packages/core/legoesm/parallel/voronoi_partition.py:1153:    # --- Pad so that nCells and nEdges are divisible by n_devices ---
packages/core/legoesm/parallel/voronoi_partition.py-1154-    return _pad_voronoi_for_sharding(reordered, n_devices)
--
packages/core/legoesm/grids/operators_latlon_cgrid.py-274-    Adds ``halo`` ghost columns on each lon side so a compact zonal stencil
packages/core/legoesm/grids/operators_latlon_cgrid.py-275-    spans longitude partition cuts.  Backend dispatch:
packages/core/legoesm/grids/operators_latlon_cgrid.py-276-
packages/core/legoesm/grids/operators_latlon_cgrid.py-277-    * local / band / SPMD-lat / non-2-D MPI — every rank owns the full
packages/core/legoesm/grids/operators_latlon_cgrid.py-278-      longitude circle, so the wrap is LOCAL: ``jnp.pad(mode="wrap")``.
packages/core/legoesm/grids/operators_latlon_cgrid.py:279:    * 2-D pencil (``LatLon2DLayout``) — longitude is split, so the wrap
packages/core/legoesm/grids/operators_latlon_cgrid.py-280-      becomes an MPI ring exchange with the W/E neighbour
packages/core/legoesm/grids/operators_latlon_cgrid.py-281-      (:func:`legoesm.parallel.latlon_mpi.exchange_halo_lon`).
packages/core/legoesm/grids/operators_latlon_cgrid.py-282-    * 2-D SPMD ``("lat", "lon")`` mesh (M3a) — the wrap becomes the cyclic
packages/core/legoesm/grids/operators_latlon_cgrid.py-283-      ring ``ppermute`` over the ``"lon"`` mesh axis
packages/core/legoesm/grids/operators_latlon_cgrid.py-284-      (:func:`legoesm.parallel.latlon_spmd.lon_ring_ghosts_spmd`); a
--
packages/core/legoesm/grids/operators_latlon_cgrid.py-301-    )
packages/core/legoesm/grids/operators_latlon_cgrid.py-302-    backend = get_halo_backend()
packages/core/legoesm/grids/operators_latlon_cgrid.py-303-    if backend == "mpi":
packages/core/legoesm/grids/operators_latlon_cgrid.py-304-        topology = get_mpi_topology()
packages/core/legoesm/grids/operators_latlon_cgrid.py-305-        from legoesm.parallel.latlon_mpi import (
packages/core/legoesm/grids/operators_latlon_cgrid.py:306:            LatLon2DLayout, exchange_halo_lon,
packages/core/legoesm/grids/operators_latlon_cgrid.py-307-        )
packages/core/legoesm/grids/operators_latlon_cgrid.py:308:        if isinstance(topology, LatLon2DLayout):
packages/core/legoesm/grids/operators_latlon_cgrid.py-309-            return exchange_halo_lon(
packages/core/legoesm/grids/operators_latlon_cgrid.py-310-                f, topology.west_rank, topology.east_rank,
packages/core/legoesm/grids/operators_latlon_cgrid.py-311-                topology.rank, halo=halo,
packages/core/legoesm/grids/operators_latlon_cgrid.py-312-            )
packages/core/legoesm/grids/operators_latlon_cgrid.py-313-    elif backend == "spmd":
--
packages/core/legoesm/grids/operators_latlon_cgrid.py-397-    from legoesm.grids.halo import get_halo_backend, get_mpi_topology
packages/core/legoesm/grids/operators_latlon_cgrid.py-398-    if get_halo_backend() != "mpi":
packages/core/legoesm/grids/operators_latlon_cgrid.py-399-        return None
packages/core/legoesm/grids/operators_latlon_cgrid.py-400-    topology = get_mpi_topology()
packages/core/legoesm/grids/operators_latlon_cgrid.py-401-    from legoesm.parallel.latlon_mpi import (
packages/core/legoesm/grids/operators_latlon_cgrid.py:402:        LatLon2DLayout, LatLonBandLayout,
packages/core/legoesm/grids/operators_latlon_cgrid.py-403-    )
packages/core/legoesm/grids/operators_latlon_cgrid.py-404-    # Band AND 2-D pencil expose the same pole-terminated lat-LINE
packages/core/legoesm/grids/operators_latlon_cgrid.py-405-    # semantics (``south_rank``/``north_rank is None`` at the physical
packages/core/legoesm/grids/operators_latlon_cgrid.py-406-    # poles), so the pole-touch test is identical.  Recognising the 2-D
packages/core/legoesm/grids/operators_latlon_cgrid.py-407-    # layout here is what makes ``lat_ends_are_poles`` /
packages/core/legoesm/grids/operators_latlon_cgrid.py-408-    # ``interp_cell_to_vface_halo`` correct on a ``proc_lat>1`` pencil rank
packages/core/legoesm/grids/operators_latlon_cgrid.py-409-    # — without it an interior lat-cut rank would clamp its band edge to a
packages/core/legoesm/grids/operators_latlon_cgrid.py-410-    # physical pole (e.g. curl_vertex's sin clamp), corrupting metrics.
packages/core/legoesm/grids/operators_latlon_cgrid.py:411:    if isinstance(topology, (LatLonBandLayout, LatLon2DLayout)) and (
packages/core/legoesm/grids/operators_latlon_cgrid.py-412-        topology.south_rank is not None
packages/core/legoesm/grids/operators_latlon_cgrid.py-413-        or topology.north_rank is not None
packages/core/legoesm/grids/operators_latlon_cgrid.py-414-    ):
packages/core/legoesm/grids/operators_latlon_cgrid.py-415-        return topology
packages/core/legoesm/grids/operators_latlon_cgrid.py-416-    return None
--
packages/core/legoesm/parallel/ensemble.py-444-    handles ``n_members / n_devices`` ensemble members.
packages/core/legoesm/parallel/ensemble.py-445-
packages/core/legoesm/parallel/ensemble.py-446-    Parameters
packages/core/legoesm/parallel/ensemble.py-447-    ----------
packages/core/legoesm/parallel/ensemble.py-448-    n_members : int
packages/core/legoesm/parallel/ensemble.py:449:        Total number of ensemble members.  Must be divisible by the
packages/core/legoesm/parallel/ensemble.py-450-        number of devices.
packages/core/legoesm/parallel/ensemble.py-451-    devices : sequence of jax.Device, optional
packages/core/legoesm/parallel/ensemble.py-452-        Devices to use.  Default: all available.
packages/core/legoesm/parallel/ensemble.py-453-
packages/core/legoesm/parallel/ensemble.py-454-    Returns
--
packages/core/legoesm/parallel/ensemble.py-460-        devices = jax.devices()
packages/core/legoesm/parallel/ensemble.py-461-    n_devices = len(devices)
packages/core/legoesm/parallel/ensemble.py-462-
packages/core/legoesm/parallel/ensemble.py-463-    if n_members % n_devices != 0:
packages/core/legoesm/parallel/ensemble.py-464-        raise ValueError(
packages/core/legoesm/parallel/ensemble.py:465:            f"n_members ({n_members}) must be divisible by "
packages/core/legoesm/parallel/ensemble.py-466-            f"n_devices ({n_devices})"
packages/core/legoesm/parallel/ensemble.py-467-        )
packages/core/legoesm/parallel/ensemble.py-468-
packages/core/legoesm/parallel/ensemble.py-469-    return Mesh(np.array(devices), axis_names=('ensemble',))
packages/core/legoesm/parallel/ensemble.py-470-
--
packages/core/legoesm/parallel/distributed_fft.py-221-
packages/core/legoesm/parallel/distributed_fft.py-222-
packages/core/legoesm/parallel/distributed_fft.py-223-def _y_itranspose(spec_full_ky, ny_out_global, n_ranks, comm, B):
packages/core/legoesm/parallel/distributed_fft.py-224-    """Inverse of :func:`_y_transpose_fft` at an arbitrary output y-length:
packages/core/legoesm/parallel/distributed_fft.py-225-    complex ``(ny_out_global, B_loc, nz)`` spectrum → real ``(ny_out_local, B,
packages/core/legoesm/parallel/distributed_fft.py:226:    nz)`` (y decomposed). ``ny_out_global`` must be divisible by ``n_ranks``."""
packages/core/legoesm/parallel/distributed_fft.py-227-    P = n_ranks
packages/core/legoesm/parallel/distributed_fft.py-228-    ny_out_local = ny_out_global // P
packages/core/legoesm/parallel/distributed_fft.py-229-    B_loc, nz = spec_full_ky.shape[1], spec_full_ky.shape[2]
packages/core/legoesm/parallel/distributed_fft.py-230-    ycoarse = jnp.fft.ifft(spec_full_ky, axis=0).real    # (ny_out_global, B_loc, nz)
packages/core/legoesm/parallel/distributed_fft.py-231-    blk = ycoarse.reshape(P, ny_out_local, B_loc, nz)
--
packages/core/legoesm/parallel/latlon_spmd.py-117-    ``v_lower`` representation, INSIDE a ``shard_map`` over ``axis``.
packages/core/legoesm/parallel/latlon_spmd.py-118-
packages/core/legoesm/parallel/latlon_spmd.py-119-    The staggered meridional velocity ``v`` has a leading dim ``n_lat+1`` (faces
packages/core/legoesm/parallel/latlon_spmd.py-120-    at latitude interfaces), coprime with ``n_lat`` for ``N>1`` so it cannot be
packages/core/legoesm/parallel/latlon_spmd.py-121-    sharded directly; it is carried as ``v_lower = v[:n_lat]`` (``n_lat`` rows,
packages/core/legoesm/parallel/latlon_spmd.py:122:    divisible by ``N``). Each band's NORTH boundary face is the next band's
packages/core/legoesm/parallel/latlon_spmd.py-123-    ``v_lower[0]`` (= the shared global interface row), lifted down via
packages/core/legoesm/parallel/latlon_spmd.py-124-    ``ppermute(..., perm_north)``; the north-most band has no neighbour there and
packages/core/legoesm/parallel/latlon_spmd.py-125-    receives the pole-wall zero (the ppermute non-target). Pure array core (no
packages/core/legoesm/parallel/latlon_spmd.py-126-    Field/state coupling) shared by the ocean and atmosphere lat-band SPMD steps
packages/core/legoesm/parallel/latlon_spmd.py-127-    so the v-stagger numerics are written ONCE (factored from the ocean step's
--
packages/core/legoesm/parallel/latlon_spmd.py-223-    LAST column is the periodic seam (``u[:, n_lon] == u[:, 0]`` by
packages/core/legoesm/parallel/latlon_spmd.py-224-    construction — ``interp_cell_to_uface`` builds it from the wrap, and every
packages/core/legoesm/parallel/latlon_spmd.py-225-    C-grid tendency preserves the identity because face ``n_lon`` and face
packages/core/legoesm/parallel/latlon_spmd.py-226-    ``0`` difference the same wrapped operands).  Under a 2-D lon split, ``u``
packages/core/legoesm/parallel/latlon_spmd.py-227-    is therefore carried as ``u_left = u[:, :n_lon]`` (``n_lon`` columns,
packages/core/legoesm/parallel/latlon_spmd.py:228:    divisible by ``p_lon``); each tile's EAST boundary face is its east
packages/core/legoesm/parallel/latlon_spmd.py-229-    neighbour's ``u_left[:, 0]`` (the shared global interface column), lifted
packages/core/legoesm/parallel/latlon_spmd.py-230-    via the cyclic ring permutation.  Unlike the v/pole case, longitude is
packages/core/legoesm/parallel/latlon_spmd.py-231-    periodic so EVERY tile is a ppermute target — the wrap pair
packages/core/legoesm/parallel/latlon_spmd.py-232-    ``(0, p_lon-1)`` delivers tile 0's first column as the LAST tile's seam,
packages/core/legoesm/parallel/latlon_spmd.py-233-    which equals the global ``u[:, n_lon]`` by the seam identity above.
--
packages/core/legoesm/grids/halo_latlon.py-180-        return None
packages/core/legoesm/grids/halo_latlon.py-181-    return mesh
packages/core/legoesm/grids/halo_latlon.py-182-
packages/core/legoesm/grids/halo_latlon.py-183-
packages/core/legoesm/grids/halo_latlon.py-184-def _dispatch_latlon_2d_fold(data, topology, halo, *, is_vector_v):
packages/core/legoesm/grids/halo_latlon.py:185:    """Fold-family ``pad_halo_latlon*`` dispatch for a ``LatLon2DLayout``.
packages/core/legoesm/grids/halo_latlon.py-186-
packages/core/legoesm/grids/halo_latlon.py-187-    ``proc_lon == 1`` is a pure latitude band (every rank owns the full lon
packages/core/legoesm/grids/halo_latlon.py-188-    circle), so the 180-deg pole fold is LOCAL — reuse the validated band
packages/core/legoesm/grids/halo_latlon.py-189-    fold (:func:`legoesm.parallel.latlon_mpi.pad_halo_latlon_mpi` on the
packages/core/legoesm/grids/halo_latlon.py-190-    equivalent :class:`~legoesm.parallel.latlon_mpi.LatLonBandLayout`),
--
packages/core/legoesm/grids/halo_latlon.py-255-        # an MPI run was activated for a different grid type (e.g.
packages/core/legoesm/grids/halo_latlon.py-256-        # cubed-sphere) and a lat-lon op was called by mistake; fall
packages/core/legoesm/grids/halo_latlon.py-257-        # back to the local serial path rather than crashing in the
packages/core/legoesm/grids/halo_latlon.py-258-        # MPI dispatch with an opaque error.
packages/core/legoesm/grids/halo_latlon.py-259-        from legoesm.parallel.latlon_mpi import (
packages/core/legoesm/grids/halo_latlon.py:260:            LatLon2DLayout, LatLonBandLayout, pad_halo_latlon_mpi,
packages/core/legoesm/grids/halo_latlon.py-261-        )
packages/core/legoesm/grids/halo_latlon.py-262-        if isinstance(topology, LatLonBandLayout):
packages/core/legoesm/grids/halo_latlon.py-263-            return pad_halo_latlon_mpi(
packages/core/legoesm/grids/halo_latlon.py-264-                data, topology, halo=halo, is_vector_v=False,
packages/core/legoesm/grids/halo_latlon.py-265-            )
packages/core/legoesm/grids/halo_latlon.py:266:        if isinstance(topology, LatLon2DLayout):
packages/core/legoesm/grids/halo_latlon.py-267-            return _dispatch_latlon_2d_fold(
packages/core/legoesm/grids/halo_latlon.py-268-                data, topology, halo, is_vector_v=False,
packages/core/legoesm/grids/halo_latlon.py-269-            )
packages/core/legoesm/grids/halo_latlon.py-270-    _spmd = _try_spmd_latlon_pad(data, halo, negate=False)
packages/core/legoesm/grids/halo_latlon.py-271-    if _spmd is not None:
--
packages/core/legoesm/grids/halo_latlon.py-309-    """
packages/core/legoesm/grids/halo_latlon.py-310-    from legoesm.grids.halo import get_halo_backend, get_mpi_topology
packages/core/legoesm/grids/halo_latlon.py-311-    if get_halo_backend() == "mpi":
packages/core/legoesm/grids/halo_latlon.py-312-        topology = get_mpi_topology()
packages/core/legoesm/grids/halo_latlon.py-313-        from legoesm.parallel.latlon_mpi import (
packages/core/legoesm/grids/halo_latlon.py:314:            LatLon2DLayout, LatLonBandLayout, pad_halo_latlon_mpi,
packages/core/legoesm/grids/halo_latlon.py-315-        )
packages/core/legoesm/grids/halo_latlon.py-316-        if isinstance(topology, LatLonBandLayout):
packages/core/legoesm/grids/halo_latlon.py-317-            return pad_halo_latlon_mpi(
packages/core/legoesm/grids/halo_latlon.py-318-                data, topology, halo=halo, is_vector_v=True,
packages/core/legoesm/grids/halo_latlon.py-319-            )
packages/core/legoesm/grids/halo_latlon.py:320:        if isinstance(topology, LatLon2DLayout):
packages/core/legoesm/grids/halo_latlon.py-321-            return _dispatch_latlon_2d_fold(
packages/core/legoesm/grids/halo_latlon.py-322-                data, topology, halo, is_vector_v=True,
packages/core/legoesm/grids/halo_latlon.py-323-            )
packages/core/legoesm/grids/halo_latlon.py-324-    _spmd = _try_spmd_latlon_pad(data, halo, negate=True)

--- srun examples resource flags ---
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:7:#SBATCH --gpus-per-node=4
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:45:  JAX_ENABLE_X64=0 srun --nodes="$NODES" --ntasks="$NP" --ntasks-per-node=4 \
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:46:      --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 \
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch:7:#SBATCH --gpus-per-node=4
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch:57:  JAX_ENABLE_X64=0 srun --nodes=8 --ntasks=32 --ntasks-per-node=4 \
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch:58:      --gpus-per-node=4 --gpu-bind=none --exact --kill-on-bad-exit=1 \
scripts/cluster/scaling_levante/gpu_scaling.sbatch:6:#SBATCH --ntasks-per-node=1
scripts/cluster/scaling_levante/gpu_scaling.sbatch:7:#SBATCH --gpus-per-node=4
scripts/cluster/scaling_levante/cpu_scaling.sbatch:5:#SBATCH --ntasks-per-node=64
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:6:#SBATCH --ntasks-per-node=4
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:7:#SBATCH --gpus-per-node=4
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:70:STEP_GPU_OPTS="${STEP_GPU_OPTS:---gpus-per-node=4 --gpu-bind=none}"
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:140:srun --ntasks="$PRE_NP" --ntasks-per-node=4 --time=2 $STEP_GPU_OPTS bash -c "$PIN" _ \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:147:    srun --ntasks=6 --ntasks-per-node=3 $STEP_GPU_OPTS --kill-on-bad-exit=1 bash -c "$PIN" _ \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:164:    srun --ntasks="$NP_LL" --ntasks-per-node=4 $STEP_GPU_OPTS --kill-on-bad-exit=1 bash -c "$PIN" _ \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:182:    srun --ntasks="$NP_LL" --ntasks-per-node=4 $STEP_GPU_OPTS --kill-on-bad-exit=1 bash -c "$PIN" _ \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:206:    srun --ntasks="$NP_MPAS" --ntasks-per-node="$E_TPN" $STEP_GPU_OPTS --kill-on-bad-exit=1 bash -c "$PIN" _ \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:213:    && srun --ntasks="$NP_MPAS" --ntasks-per-node="$E_TPN" $STEP_GPU_OPTS --kill-on-bad-exit=1 bash -c "$PIN" _ \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:245:            srun --ntasks=8 --ntasks-per-node=4 $STEP_GPU_OPTS --kill-on-bad-exit=1 bash -c "$PIN" _ \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:260:            srun --ntasks=8 --ntasks-per-node=4 $STEP_GPU_OPTS --kill-on-bad-exit=1 bash -c "$PIN" _ \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:276:            srun --ntasks=6 --ntasks-per-node=3 $STEP_GPU_OPTS --kill-on-bad-exit=1 bash -c "$PIN" _ \
scripts/cluster/scaling_levante/cube_tiled_step.sbatch:5:#SBATCH --ntasks-per-node=4
scripts/cluster/scaling_levante/cube_tiled_step.sbatch:6:#SBATCH --gpus-per-node=4
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch:7:#SBATCH --gpus-per-node=4
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch:35:  JAX_ENABLE_X64=0 srun --ntasks="$3" --ntasks-per-node=4 \
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch:36:      --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 \
scripts/cluster/scaling_levante/atm_latlon2d_cpu_hundreds.sbatch:27:  JAX_ENABLE_X64=1 srun --nodes="$NODES" --ntasks="$NP" --ntasks-per-node=32 \
scripts/cluster/scaling_levante/diagnosis.sbatch:6:#SBATCH --ntasks-per-node=1
scripts/cluster/scaling_levante/diagnosis.sbatch:7:#SBATCH --gpus-per-node=4

exec
/usr/bin/bash -lc "rg -n -C 4 'cells_per_rank_achieved|wet|owned|partition_quality|edge_cut|halo mean|s8 np=128 partition=metis|s8 np=128 partition=geometric|s7 np=32 partition=metis|s7 np=32 partition=geometric|26600094|333\\.39|308\\.96|191\\.99|189\\.82' .physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md | tail -n 1000" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 220ms:
13-user
14-Adversarial review of a scaling-campaign iteration. Attack arithmetic, logic, confounds, and overclaim. Context: docs/performance/scaling/levante_campaign_2026-07-24.md, new final section 'Phase-3 receipts recovered after the 2026-07-31 session drop (2026-08-02)', plus the edit to scripts/plot/plot_scaling_paper_figure.py (git diff HEAD -- scripts/plot/plot_scaling_paper_figure.py) and the new scripts/tmp/mpas_s9_ensemble.sbatch.
15-
16-Claims to attack:
17:1. METIS A/B (job 26600094, all arms 5120 cells/rank, f64, 32 ranks/node): A s7np32 geometric 189.82ms, B s8np128 geometric 308.96, C s7np32 metis 191.99, D s8np128 metis 333.39, E s8np128 metis block:block 537.91. My claims: rank-count term geometric 1.628 vs metis 1.736 -> partition-quality lever DEAD/refuted; block:cyclic mandatory (E/D=1.61); lever#2 closed refuted on this lane.
18-2. s9 GPU ladder (job 26600095, f32 sfc lloyd0, 2621442 cells L26): np32 12.47ms/5.47GC/s, np64 9.60/7.10, np128 11.48/5.94. Claims: new MPAS peak 7.10 GC/s = 2.2x s8 best 3.23 GC/s (s8 655362 cells, np64 5.27ms, 26 levels -> check GC/s arithmetic); 64->128 anti-scales at 20.5k cells/GPU consistent with ~30k floor; weak pairs s8->s9 at matched cells/GPU: 6.92->12.47 (0.55), 7.10->9.60 (0.74), 8.13->11.48 (0.71) -> ~1.4x matched-tile cost per 4x ranks = GPU rank-count term.
19-3. Ensemble job design (mpas_s9_ensemble.sbatch): 4 concurrent 32-GPU srun steps in one 32-node allocation, per-step SLURM_STEP_NODELIST coordinator autodetect, shared jobid-derived port claimed safe because hosts differ; solo control in-job; CONFIRM bar max(replica)<=1.10x solo. Attack: step isolation, GPU binding, srun flag errors, cache effects, whether solo-then-concurrent ordering biases, whether 12 steps is enough discrimination.
20-Check every ratio and GC/s conversion. Flag any claim that should be labelled PLAUSIBLE rather than CONFIRMED, any protocol mismatch making a comparison confounded, and any sbatch bug. End with 'VERDICT: SHIP' or 'VERDICT: FIX-FIRST' plus a numbered issue list.
21-codex
--
228-## Oracle-Recipe Fidelity (ocean) — see docs/ocean/fidelity/oracle_recipe_strategy.md
229-- ADDITIVE to Validation Rules: oracle work NEVER replaces unit tests, the ocean matrix, conservation checks, or visual verification. Truth tiers (conservation/equivariance/analytic) outrank oracle-matching.
230-- Recipe = pure config selecting shared canonical blocks (never a bespoke `veros_*` solver). Oracle-matching numerics go in the canonical module (`eos.py`, advection/limiter dispatch, `vertical_mixing/`, integrator dispatch) as selectable options.
231-- Mimicry-only glue (halo strip, axis transpose, time-level handling) lives in the fidelity harness, never the model. Test: "would a user with a different goal ever select this?" No → harness.
232:- Conventions handled only in the bridge, verified by equivariance tests (`physics(φ(x))=φ(physics(x))` to tol); a "convention" that changes the wet domain/answers is physics → config, not bridge.
233-- Constants are config (`ConstantsConfig`), not module-global monkey-patches (no `override_constants` in shippable paths); defaults reference `legoesm.constants`; base only, derived (κ,ε) recomputed.
234-- Oracle tendency-match (tier 3) trusted only for a block that also clears truth tiers (0–2). MACHINE-ENFORCED (#388 Ask#4): `ocean/fidelity/precedence.py::evaluate_precedence` LOCKS oracle tiers (≥3) on any truth-tier (0–2) failure; surfaced + exit-gated by `scripts/validate/ocean_fidelity/build_fidelity_scorecard.py` (the one generated scorecard).
235-
236-## Recipe×Setup template adapters (#388)
--
483-3. ~~Calibrated theoretical-limit lines~~ **DONE later in this campaign**:
484-   `bench_ppermute_microbench.py` measured the fabric constants (NVLink
485-   17.8 us / 64.2 GB/s, IB 26.3 us / 23.5 GB/s) and the roofline sections
486-   above use them. Kept here only so the list's numbering stays stable.
487:4. **Ocean wet-cell compaction + wet-balanced partitions** — SPLIT
488-   2026-07-26 into a cheap half and an expensive half:
489-
490:   *Cheap half — wet-BALANCED bands* (no indirection, uneven band heights
491-   equalizing OCEAN cells per rank): already implemented on the MPI lane
492:   (`bench_ocean_mpi_scaling.py --wet-balance`, ETOPO continents); A/B at
493-   np16/np32 running (job 26480448, r128/r256 at CFL-scaled dt 100/50s).
494-   OPERATIONAL TRAIL kept for honesty: four earlier submissions failed —
495-   wrong venv (26479815), then non-finite at dt 600 and 300 (26479904/
496-   26480203/26480298), briefly mis-read as a lane defect until the
--
498-   need dt<=150 at LL96, which I had not read; the isolated-basin
499-   hypothesis tested along the way was refuted (`fill_isolated_basins`
500-   made no difference, consistent with dt being the real cause).
501-   MEASURED (job 26480448, r128 ETOPO
502:   CFL dt=100s, f64, np16/np32, single runs): wet-balancing LOSES —
503:   equal-rows 37.89 / 30.50 ms vs wet-balanced **47.39 / 44.20 ms**
504-   (25-45 % SLOWER). The mechanism is coherent with the gather
505-   microbench's finding: this lane computes DENSE arrays (a land cell
506:   costs the same as a wet one), so per-rank cost tracks TOTAL ROWS, and
507-   equalizing WET cells makes total rows uneven — it balances the wrong
508-   quantity. Wet-balancing could only pay on an implementation whose cost
509:   tracks wet cells (i.e. compacted), and the expensive-half measurement
510:   below shows compaction itself does not pay at real wet fractions.
511-
512-   VERDICT on the audit's item 4 as a whole: BOTH halves measured, BOTH
513-   negative on this codebase — the "~2x on ~40%-land grids" projection is
514-   refuted twice over (gather penalty eats the compaction saving at 0.71
515:   wet; wet-balanced bands worsen dense-compute balance). The item is
516-   CLOSED as not-worth-building, with receipts. (Also moot for the SPMD
517-   lane: jax equal-shard sharding would need padding to the max band,
518-   returning exactly the imbalance removed.)
519-
520-   *Expensive half — gather/scatter compaction: MEASURED, and the audit's
521-   "~2x" is REFUTED* (`bench_gather_vs_slice_stencil.py`, job 26479884,
522-   A100 f32, correctness self-checked). Per-cell gather penalty for a
523-   5-point Laplacian vs the dense sliced version: **1.40-1.77x**, so
524:   compaction wins only when wet_fraction < 0.56-0.72 (size-dependent).
525:   At the REAL global-ocean wet fraction (~0.71), packed-gather is a net
526-   LOSS on the full LL576 grid (ratio 1.16) and a wash at the nd4 tile
527-   (1.03). SCOPE (codex round-10): the two numbers are END-MEMBER ESTIMATES, not
528-   a bound — 0.86x is the pure-stencil member (measured), 1.41x the
529-   pure-column ideal; a real step also pays packing/scattering at the
530:   interface, sees real wet topology (not banded), richer stencils, and
531-   communication, none of which the microbench prices. What survives
532-   regardless: the audit's 2x assumed zero indirection cost and is
533:   refuted; at ~0.71 wet the measured stencil member is a net LOSS. VERDICT: do not build compaction for the global latlon ocean;
534:   revisit only for a configuration that is genuinely <~55% wet.
535-5. **2-D lat-lon decomposition** at >=64 ranks: the 1-D band's perimeter
536-   ceiling is now measured (0.12-0.16 at np64 spread, vs ico's 0.52), which
537-   quantifies the prize.
538-6. ~~Milan np16 anomaly~~ **RESOLVED 2026-07-26 (job 26479904): rank
--
573-
574-Both jobs the dropped session left behind COMPLETED; neither had been
575-analysed. Numbers below are their first read-out.
576-
577:### 1. METIS / placement A-B (ocean MPAS CPU, job 26600094) — both halves of codex lever #2 are DEAD
578-
579-Matrix at IDENTICAL 5,120 cells/rank (anti-masquerade fields verified:
580:`cells_per_rank_achieved=5120`, `n_ranks` 32/128, method pinned per arm —
581-never `auto`, which flipped meaning when pymetis appeared in `.venv-mpi`
582-on 2026-07-31), f64, nlev 20, 32 ranks/node, `--distribution` as listed:
583-
584-| arm | config | ms/step |
585-|---|---|---|
586:| A | s7 np32 geometric, block:cyclic | 189.82 |
587:| B | s8 np128 geometric, block:cyclic | 308.96 |
588:| C | s7 np32 metis, block:cyclic | 191.99 |
589:| D | s8 np128 metis, block:cyclic | 333.39 |
590-| E | s8 np128 metis, block:block | 537.91 |
591-
592-* **Partition quality does NOT buy back the rank-count term — METIS is
593-  WORSE at scale.** Rank-count term geometric B/A = 1.628; metis
594-  D/C = 1.736 (D is +7.9 % over B), despite METIS holding the better
595:  offline cut on this exact mesh (partq s8@np128: edge_cut 1.89 % vs
596:  2.01 %, halo mean 592 vs 629). The offline-quality -> step-time
597-  inference is REFUTED on this lane; the full-step measurement fails
598-  codex's own ">=10 % full-step win" bar in the wrong direction.
599-* **Placement: block:cyclic stays mandatory** (E/D = 1.61x at a
600-  byte-identical partition) — same direction as the np16 Milan receipt
--
890-if __name__ == "__main__":
891-    raise SystemExit(main())
892-
893-exec
894:/usr/bin/bash -lc "printf '%s\\n' '--- sbatch ---'; nl -ba scripts/tmp/mpas_s9_ensemble.sbatch; printf '%s\\n' '--- phase exact with locations ---'; nl -ba docs/performance/scaling/levante_campaign_2026-07-24.md | sed -n '/Phase-3 receipts recovered/,"'$p'"'; printf '%s\\n' '--- relevant scripts / bench references ---'; rg -n -i -C 3 '2660009[45]|26549646|26538474|subdiv-9|s9|sfc|lloyd|cells_per_rank_achieved|partq|ensemble' docs scripts results 2>/dev/null | head -n 1000" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
895- succeeded in 641ms:
896---- sbatch ---
897-     1	#!/bin/bash -l
898-     2	#SBATCH --job-name=mpas_s9_ens
--
967-  1588	
968-  1589	Both jobs the dropped session left behind COMPLETED; neither had been
969-  1590	analysed. Numbers below are their first read-out.
970-  1591	
971:  1592	### 1. METIS / placement A-B (ocean MPAS CPU, job 26600094) — both halves of codex lever #2 are DEAD
972-  1593	
973-  1594	Matrix at IDENTICAL 5,120 cells/rank (anti-masquerade fields verified:
974:  1595	`cells_per_rank_achieved=5120`, `n_ranks` 32/128, method pinned per arm —
975-  1596	never `auto`, which flipped meaning when pymetis appeared in `.venv-mpi`
976-  1597	on 2026-07-31), f64, nlev 20, 32 ranks/node, `--distribution` as listed:
977-  1598	
978-  1599	| arm | config | ms/step |
979-  1600	|---|---|---|
980:  1601	| A | s7 np32 geometric, block:cyclic | 189.82 |
981:  1602	| B | s8 np128 geometric, block:cyclic | 308.96 |
982:  1603	| C | s7 np32 metis, block:cyclic | 191.99 |
983:  1604	| D | s8 np128 metis, block:cyclic | 333.39 |
984-  1605	| E | s8 np128 metis, block:block | 537.91 |
985-  1606	
986-  1607	* **Partition quality does NOT buy back the rank-count term — METIS is
987-  1608	  WORSE at scale.** Rank-count term geometric B/A = 1.628; metis
988-  1609	  D/C = 1.736 (D is +7.9 % over B), despite METIS holding the better
989:  1610	  offline cut on this exact mesh (partq s8@np128: edge_cut 1.89 % vs
990:  1611	  2.01 %, halo mean 592 vs 629). The offline-quality -> step-time
991-  1612	  inference is REFUTED on this lane; the full-step measurement fails
992-  1613	  codex's own ">=10 % full-step win" bar in the wrong direction.
993-  1614	* **Placement: block:cyclic stays mandatory** (E/D = 1.61x at a
994-  1615	  byte-identical partition) — same direction as the np16 Milan receipt
--
1840---
1841-scripts/bench/bench_ocean_mpas_scaling.py-748-        # HORIZONTAL cells/rank — the same unit as --cells-per-rank, so a
1842-scripts/bench/bench_ocean_mpas_scaling.py-749-        # weak-mode row is comparable to its target (codex: the 3-D count
1843-scripts/bench/bench_ocean_mpas_scaling.py-750-        # made rows look nlev-x larger).
1844:scripts/bench/bench_ocean_mpas_scaling.py:751:        cells_per_rank_achieved=int(mesh.nCells) // n_ranks,
1845-scripts/bench/bench_ocean_mpas_scaling.py-752-        # Fix 4 honesty flag: at np=1 the parity reference pre-runs the
1846-scripts/bench/bench_ocean_mpas_scaling.py-753-        # SAME shape before the timed loop, so per_step_ms[0] may not
1847-scripts/bench/bench_ocean_mpas_scaling.py-754-        # contain the real JIT compile.
1848---
--
6461-162-      --gpus-per-task=n       number of GPUs required per spawned task
6462-163-      --mem-per-gpu=n         real memory required per allocated GPU
6463-
6464-exec
6465:/usr/bin/bash -lc "python -c 'from decimal import Decimal as D; A,B,C,DD,E=map(D,[\"189.82\",\"308.96\",\"191.99\",\"333.39\",\"537.91\"]); print(\"B/A\",B/A,\"D/C\",DD/C,\"D/B\",DD/B,\"E/D\",E/DD,\"E/B\",E/B); s8=[D(\"6.92\"),D(\"7.10\"),D(\"8.13\")]; s9=[D(\"12.47\"),D(\"9.60\"),D(\"11.48\")]; print(\"weak efficiency\",*[x/y for x,y in zip(s8,s9)]); print(\"weak time ratio\",*[y/x for x,y in zip(s8,s9)]); print(\"32->64 speedup\",s9[0]/s9[1],\"eff\",s9[0]/s9[1]/2,\"64->128 speedup\",s9[1]/s9[2],\"eff\",s9[1]/s9[2]/2); cells=D(2621442); lev=D(26); print(\"s9 GC/s\",*[cells*lev/(t/D(1000))/D(10)**9 for t in s9]); print(\"s8 strict cells GC/s\",D(655362)*lev/(D(\"5.27\")/D(1000))/D(10)**9); print(\"s8 padded cells GC/s\",D(655376)*lev/(D(\"5.27\")/D(1000))/D(10)**9); solo=D(\"5.47\"); single128=D(\"5.94\"); print(\"all healthy 4xsolo\",4*solo,\"threshold\",4*solo/D(\"1.10\"),\"threshold vs s9-128\",4*solo/D(\"1.10\")/single128,\"all vs 128\",4*solo/single128)' && printf '%s\\n' '--- stored MPAS receipts ---' && rg -n -C 3 '\"n_cells\"\\s*:\\s*655|\"nCells\"\\s*:\\s*655|655376|655362|\"steady_median_ms\"\\s*:\\s*5\\.27|\"steady_median_ms\"\\s*:\\s*9\\.60|26600095|26600094' . --glob '*.jsonl' --glob '*.log' --glob '*.out' --glob '*.md' --glob '*.csv' --glob '*.txt' 2>/dev/null | head -n 1200 && printf '%s\\n' '--- campaign figure handling of cell count / plot source ---' && rg -n -C 4 '655_?3(62|76)|GC/s|mcells_per_s|gcell|throughput' docs/performance/scaling scripts/plot/plot_scaling_paper_figure.py | head -n 1000" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
6466- exited 127 in 359ms:
6467-/usr/bin/bash: python: command not found
6468-
6469-exec
6470:/usr/bin/bash -lc "/usr/bin/python3 -c 'from decimal import Decimal as D; A,B,C,DD,E=map(D,[\"189.82\",\"308.96\",\"191.99\",\"333.39\",\"537.91\"]); print(\"B/A\",B/A,\"D/C\",DD/C,\"D/B\",DD/B,\"E/D\",E/DD,\"E/B\",E/B); s8=[D(\"6.92\"),D(\"7.10\"),D(\"8.13\")]; s9=[D(\"12.47\"),D(\"9.60\"),D(\"11.48\")]; print(\"weak efficiency\",*[x/y for x,y in zip(s8,s9)]); print(\"weak time ratio\",*[y/x for x,y in zip(s8,s9)]); print(\"32->64 speedup\",s9[0]/s9[1],\"eff\",s9[0]/s9[1]/2,\"64->128 speedup\",s9[1]/s9[2],\"eff\",s9[1]/s9[2]/2); cells=D(2621442); lev=D(26); print(\"s9 GC/s\",*[cells*lev/(t/D(1000))/D(10)**9 for t in s9]); print(\"s8 strict cells GC/s\",D(655362)*lev/(D(\"5.27\")/D(1000))/D(10)**9); print(\"s8 padded cells GC/s\",D(655376)*lev/(D(\"5.27\")/D(1000))/D(10)**9); solo=D(\"5.47\"); single128=D(\"5.94\"); print(\"all healthy 4xsolo\",4*solo,\"threshold\",4*solo/D(\"1.10\"),\"threshold vs s9-128\",4*solo/D(\"1.10\")/single128,\"all vs 128\",4*solo/single128)' ; printf '%s\\n' '--- stored MPAS receipts ---'; rg -n -C 3 '\"n_cells\"\\s*:\\s*655|\"nCells\"\\s*:\\s*655|655376|655362|\"steady_median_ms\"\\s*:\\s*5\\.27|\"steady_median_ms\"\\s*:\\s*9\\.60|26600095|26600094' . --glob '*.jsonl' --glob '*.log' --glob '*.out' --glob '*.md' --glob '*.csv' --glob '*.txt' 2>/dev/null | head -n 1200; printf '%s\\n' '--- campaign figure handling of cell count / plot source ---'; rg -n -C 4 '655_?3(62|76)|GC/s|mcells_per_s|gcell|throughput' docs/performance/scaling scripts/plot/plot_scaling_paper_figure.py | head -n 1000" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
6471- succeeded in 1801ms:
6472-B/A 1.627647244758191971341270677 D/C 1.736496692536069586957654044 D/B 1.079071724495080269290523045 E/D 1.613455712528870092084345661 E/B 1.741034438114966338684619368
6473-weak efficiency 0.5549318364073777064955894146 0.7395833333333333333333333333 0.7081881533101045296167247387
6474-weak time ratio 1.802023121387283236994219653 1.352112676056338028169014085 1.412054120541205412054120541
--
6507---
6508-./mpasoc_2x2b.26508336.log-2124-[1785187889.686434] [l40121:994955:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6509-./mpasoc_2x2b.26508336.log-2125-[1785187889.824931] [l40121:994954:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6510-./mpasoc_2x2b.26508336.log-2126-[1785187889.718408] [l40121:994939:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6511:./mpasoc_2x2b.26508336.log:2127:{"component": "ocean", "grid": "voronoi", "mode": "strong", "subdivision": 8, "n_ranks": 128, "n_cells": 655362, "n_edges": 1966080, "nlev": 20, "partition_method": "geometric", "steps": 12, "dt": 60.0, "platform": "cpu", "compile_ms": 12748.4, "steady_median_ms": 313.165, "step_latency_gate_loop_ms": 312.67, "step_latency_gate_loop_min_ms": 310.21, "per_step_ms": [12748.4, 317.7, 317.5, 314.4, 315.0, 311.9, 314.7, 312.8, 312.1, 312.6, 311.5, 310.2], "cells": 13107240, "cells_per_rank_achieved": 5120, "compile_prewarmed_by_parity_ref": false, "barotropic_solver": "explicit_substep", "halo_refresh": "in_step", "stage_halo_correct": true, "stage_halo_note": "per-step packed entry refresh + in-step stage-frontier refreshes (make_mpas_ocean_halo_refresh) \u2014 stage-correct; see docs/performance/scaling/mpas_ocean_distributed_stage_audit.md", "fused": {"compile_ms": 310.6, "scan_compile_ms": 14782.3, "step_latency_ms": 308.818, "block_ms": [2502.83, 2503.98], "parallel_block_ms": [2504.19, 2506.45], "fused_step_ms": 313.165, "rank_imbalance": 1.0009, "rank_imbalance_per_block": [1.0009, 1.001], "block_steps": 8, "n_blocks": 2, "probe_steps": 3}, "wet_cell": {"wet_cell_levels": 13057600, "wet_cell_levels_per_device": 102012.5, "wet_fraction": 0.9962127801123654, "wet_equals_total": false, "wet_cell_levels_per_device_min": 96040, "wet_cell_levels_per_device_max": 102420}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-27T21:32:22.829346+00:00", "grid": "voronoi", "component": "ocean", "resolution": "L8", "n_levels": 20, "precision": "float64", "decomposition": "cell_partition", "solver_variant": "mpas_ocean_explicit_substep", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 128, "n_gpus": 0, "device_count": 1, "process_count": 1, "devices_per_rank": 1, "cells_per_rank": 102400, "backend": "cpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "mpi4jax", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l40121.lvt.dkrz.de", "slurm_job_id": "26508336", "git_sha": "346a77f20", "gpu_direct_requested": false, "mpi4jax_cuda_support": false, "gpu_direct_active": false, "host_staged_halo": false, "partition_metrics": {"n_owned_cells": 5120, "n_halo_cells": 658, "owned_halo_ratio": 0.128515625, "n_neighbor_ranks": 7, "messages_per_exchange": 7, "halo_recv_cells": 658, "owned_send_cells": 661, "cells_per_rank_min": 5120, "cells_per_rank_max": 5121, "edge_cut_total": 80510, "max_neighbor_ranks": 8}, "extra": {"partition_method": "geometric", "steps": 12, "warmup": 2, "parity_gate": false, "check_conservation": false, "halo_refresh": "in_step", "barotropic_solver": "explicit_substep", "block_steps": 8, "blocks": 2, "probe_steps": 3}}}
6512-./mpasoc_2x2b.26508336.log:2128:[mpas-ocean np=128 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=12748.4ms fused=313.165ms/step (probe_latency=308.818ms) gate_loop_latency=312.67ms/step
6513-./mpasoc_2x2b.26508336.log-2129-[1785187889.706319] [l40121:994925:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6514-./mpasoc_2x2b.26508336.log-2130-[1785187889.776770] [l40121:994948:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6515-./mpasoc_2x2b.26508336.log-2131-[1785187889.765943] [l40121:994956:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6516---
6517-./mpasoc_2x2b.26508336.log-4488-[1785187991.417987] [l40140:52155:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6518-./mpasoc_2x2b.26508336.log-4489-[1785187991.397635] [l40140:52180:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6519-./mpasoc_2x2b.26508336.log-4490-[1785187991.371800] [l40135:2569978:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6520:./mpasoc_2x2b.26508336.log:4491:{"component": "ocean", "grid": "voronoi", "mode": "strong", "subdivision": 8, "n_ranks": 128, "n_cells": 655362, "n_edges": 1966080, "nlev": 20, "partition_method": "sfc", "steps": 12, "dt": 60.0, "platform": "cpu", "compile_ms": 13919.2, "steady_median_ms": 617.6562, "step_latency_gate_loop_ms": 589.96, "step_latency_gate_loop_min_ms": 586.94, "per_step_ms": [13919.2, 599.2, 594.6, 590.9, 590.9, 587.6, 589.9, 590.0, 596.5, 588.8, 586.9, 588.1], "cells": 13107240, "cells_per_rank_achieved": 5120, "compile_prewarmed_by_parity_ref": false, "barotropic_solver": "explicit_substep", "halo_refresh": "in_step", "stage_halo_correct": true, "stage_halo_note": "per-step packed entry refresh + in-step stage-frontier refreshes (make_mpas_ocean_halo_refresh) \u2014 stage-correct; see docs/performance/scaling/mpas_ocean_distributed_stage_audit.md", "fused": {"compile_ms": 579.9, "scan_compile_ms": 18007.2, "step_latency_ms": 585.002, "block_ms": [4942.05, 4937.34], "parallel_block_ms": [4943.21, 4939.29], "fused_step_ms": 617.6562, "rank_imbalance": 1.0004, "rank_imbalance_per_block": [1.0004, 1.0004], "block_steps": 8, "n_blocks": 2, "probe_steps": 3}, "wet_cell": {"wet_cell_levels": 13057600, "wet_cell_levels_per_device": 102012.5, "wet_fraction": 0.9962127801123654, "wet_equals_total": false, "wet_cell_levels_per_device_min": 96060, "wet_cell_levels_per_device_max": 102420}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-27T21:34:11.238101+00:00", "grid": "voronoi", "component": "ocean", "resolution": "L8", "n_levels": 20, "precision": "float64", "decomposition": "cell_partition", "solver_variant": "mpas_ocean_explicit_substep", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 128, "n_gpus": 0, "device_count": 1, "process_count": 1, "devices_per_rank": 1, "cells_per_rank": 102400, "backend": "cpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "mpi4jax", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l40121.lvt.dkrz.de", "slurm_job_id": "26508336", "git_sha": "346a77f20", "gpu_direct_requested": false, "mpi4jax_cuda_support": false, "gpu_direct_active": false, "host_staged_halo": false, "partition_metrics": {"n_owned_cells": 5121, "n_halo_cells": 850, "owned_halo_ratio": 0.16598320640499903, "n_neighbor_ranks": 7, "messages_per_exchange": 7, "halo_recv_cells": 850, "owned_send_cells": 839, "cells_per_rank_min": 5120, "cells_per_rank_max": 5121, "edge_cut_total": 105825, "max_neighbor_ranks": 9}, "extra": {"partition_method": "sfc", "steps": 12, "warmup": 2, "parity_gate": false, "check_conservation": false, "halo_refresh": "in_step", "barotropic_solver": "explicit_substep", "block_steps": 8, "blocks": 2, "probe_steps": 3}}}
6521-./mpasoc_2x2b.26508336.log:4492:[mpas-ocean np=128 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=13919.2ms fused=617.6562ms/step (probe_latency=585.002ms) gate_loop_latency=589.96ms/step
6522-./mpasoc_2x2b.26508336.log-4493-[1785187991.094702] [l40121:997326:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6523-./mpasoc_2x2b.26508336.log-4494-=== 2x2 RESULTS (both cells hold 5120 cells/rank) ===
6524:./mpasoc_2x2b.26508336.log-4495-s7 np32 geometric:   190.07 ms  owned_max/min=102420/100740
6525---
6526-./mpas_nsys.26479922.log-5-  return lax_numpy.astype(self, dtype, copy=copy, device=device)
6527-./mpas_nsys.26479922.log-6-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/lib/python3.14/site-packages/jax/_src/numpy/array_methods.py:125: UserWarning: Explicitly requested dtype float64 requested in astype is not available, and will be truncated to dtype float32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/jax-ml/jax#current-gotchas for more.
6528-./mpas_nsys.26479922.log-7-  return lax_numpy.astype(self, dtype, copy=copy, device=device)
--
6552---
6553-./mpasoc_rpn.26505286.log-7159-[1785176811.624146] [l40609:286074:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6554-./mpasoc_rpn.26505286.log-7160-[1785176811.585797] [l40609:286091:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6555-./mpasoc_rpn.26505286.log-7161-[1785176811.628239] [l40609:286069:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6556:./mpasoc_rpn.26505286.log:7162:{"component": "ocean", "grid": "voronoi", "mode": "strong", "subdivision": 8, "n_ranks": 32, "n_cells": 655362, "n_edges": 1966080, "nlev": 20, "partition_method": "auto", "steps": 12, "dt": 60.0, "platform": "cpu", "compile_ms": 15717.1, "steady_median_ms": 861.2525, "step_latency_gate_loop_ms": 866.56, "step_latency_gate_loop_min_ms": 859.71, "per_step_ms": [15717.1, 873.7, 859.7, 865.2, 864.7, 874.3, 872.9, 868.0, 865.7, 861.0, 867.4, 880.3], "cells": 13107240, "cells_per_rank_achieved": 20480, "compile_prewarmed_by_parity_ref": false, "barotropic_solver": "explicit_substep", "halo_refresh": "in_step", "stage_halo_correct": true, "stage_halo_note": "per-step packed entry refresh + in-step stage-frontier refreshes (make_mpas_ocean_halo_refresh) \u2014 stage-correct; see docs/performance/scaling/mpas_ocean_distributed_stage_audit.md", "fused": {"compile_ms": 868.0, "scan_compile_ms": 21466.5, "step_latency_ms": 861.989, "block_ms": [6876.48, 6889.25], "parallel_block_ms": [6888.23, 6891.81], "fused_step_ms": 861.2525, "rank_imbalance": 1.0006, "rank_imbalance_per_block": [1.0009, 1.0004], "block_steps": 8, "n_blocks": 2, "probe_steps": 3}, "wet_cell": {"wet_cell_levels": 13057600, "wet_cell_levels_per_device": 408050.0, "wet_fraction": 0.9962127801123654, "wet_equals_total": false, "wet_cell_levels_per_device_min": 403240, "wet_cell_levels_per_device_max": 409620}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-27T18:28:07.406910+00:00", "grid": "voronoi", "component": "ocean", "resolution": "L8", "n_levels": 20, "precision": "float64", "decomposition": "cell_partition", "solver_variant": "mpas_ocean_explicit_substep", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 32, "n_gpus": 0, "device_count": 1, "process_count": 1, "devices_per_rank": 1, "cells_per_rank": 409601, "backend": "cpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "mpi4jax", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l40609.lvt.dkrz.de", "slurm_job_id": "26505286", "git_sha": "aabce0876", "gpu_direct_requested": false, "mpi4jax_cuda_support": false, "gpu_direct_active": false, "host_staged_halo": false, "partition_metrics": {"n_owned_cells": 20480, "n_halo_cells": 1259, "owned_halo_ratio": 0.061474609375, "n_neighbor_ranks": 6, "messages_per_exchange": 6, "halo_recv_cells": 1259, "owned_send_cells": 1258, "cells_per_rank_min": 20480, "cells_per_rank_max": 20481, "edge_cut_total": 40803, "max_neighbor_ranks": 8}, "extra": {"partition_method": "auto", "steps": 12, "warmup": 2, "parity_gate": false, "check_conservation": false, "halo_refresh": "in_step", "barotropic_solver": "explicit_substep", "block_steps": 8, "blocks": 2, "probe_steps": 3}}}
6557-./mpasoc_rpn.26505286.log:7163:[mpas-ocean np=32 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=15717.1ms fused=861.2525ms/step (probe_latency=861.989ms) gate_loop_latency=866.56ms/step
6558-./mpasoc_rpn.26505286.log-7164-[1785176811.630160] [l40609:286064:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6559-./mpasoc_rpn.26505286.log-7165-[1785176811.640647] [l40609:286088:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6560-./mpasoc_rpn.26505286.log-7166-[1785176811.641111] [l40609:286085:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6561---
6562-./mpasoc_rpn.26505286.log-8065-[1785176890.748010] [l40611:224859:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6563-./mpasoc_rpn.26505286.log-8066-[1785176890.767903] [l40611:224856:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6564-./mpasoc_rpn.26505286.log-8067-[1785176890.692244] [l40610:234172:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6565:./mpasoc_rpn.26505286.log:8068:{"component": "ocean", "grid": "voronoi", "mode": "strong", "subdivision": 8, "n_ranks": 64, "n_cells": 655362, "n_edges": 1966080, "nlev": 20, "partition_method": "auto", "steps": 12, "dt": 60.0, "platform": "cpu", "compile_ms": 13606.9, "steady_median_ms": 494.8875, "step_latency_gate_loop_ms": 467.45, "step_latency_gate_loop_min_ms": 462.35, "per_step_ms": [13606.9, 465.0, 468.1, 464.5, 465.9, 472.4, 466.8, 477.2, 462.4, 464.8, 471.0, 471.8], "cells": 13107240, "cells_per_rank_achieved": 10240, "compile_prewarmed_by_parity_ref": false, "barotropic_solver": "explicit_substep", "halo_refresh": "in_step", "stage_halo_correct": true, "stage_halo_note": "per-step packed entry refresh + in-step stage-frontier refreshes (make_mpas_ocean_halo_refresh) \u2014 stage-correct; see docs/performance/scaling/mpas_ocean_distributed_stage_audit.md", "fused": {"compile_ms": 461.6, "scan_compile_ms": 16858.2, "step_latency_ms": 460.002, "block_ms": [3948.95, 3962.98], "parallel_block_ms": [3952.5, 3965.7], "fused_step_ms": 494.8875, "rank_imbalance": 1.0008, "rank_imbalance_per_block": [1.0009, 1.0008], "block_steps": 8, "n_blocks": 2, "probe_steps": 3}, "wet_cell": {"wet_cell_levels": 13057600, "wet_cell_levels_per_device": 204025.0, "wet_fraction": 0.9962127801123654, "wet_equals_total": false, "wet_cell_levels_per_device_min": 198440, "wet_cell_levels_per_device_max": 204820}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-27T18:29:07.513939+00:00", "grid": "voronoi", "component": "ocean", "resolution": "L8", "n_levels": 20, "precision": "float64", "decomposition": "cell_partition", "solver_variant": "mpas_ocean_explicit_substep", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 64, "n_gpus": 0, "device_count": 1, "process_count": 1, "devices_per_rank": 1, "cells_per_rank": 204800, "backend": "cpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "mpi4jax", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l40609.lvt.dkrz.de", "slurm_job_id": "26505286", "git_sha": "aabce0876", "gpu_direct_requested": false, "mpi4jax_cuda_support": false, "gpu_direct_active": false, "host_staged_halo": false, "partition_metrics": {"n_owned_cells": 10240, "n_halo_cells": 904, "owned_halo_ratio": 0.08828125, "n_neighbor_ranks": 7, "messages_per_exchange": 7, "halo_recv_cells": 904, "owned_send_cells": 906, "cells_per_rank_min": 10240, "cells_per_rank_max": 10241, "edge_cut_total": 56765, "max_neighbor_ranks": 8}, "extra": {"partition_method": "auto", "steps": 12, "warmup": 2, "parity_gate": false, "check_conservation": false, "halo_refresh": "in_step", "barotropic_solver": "explicit_substep", "block_steps": 8, "blocks": 2, "probe_steps": 3}}}
6566-./mpasoc_rpn.26505286.log:8069:[mpas-ocean np=64 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=13606.9ms fused=494.8875ms/step (probe_latency=460.002ms) gate_loop_latency=467.45ms/step
6567-./mpasoc_rpn.26505286.log-8070-[1785176890.662728] [l40610:234164:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6568-./mpasoc_rpn.26505286.log-8071-[1785176890.660276] [l40610:234190:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6569-./mpasoc_rpn.26505286.log-8072-[1785176890.742754] [l40611:224847:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6570---
6571-./mpasoc_rpn.26505286.log-9850-[1785176950.894198] [l40615:270231:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6572-./mpasoc_rpn.26505286.log-9851-[1785176950.891008] [l40615:270248:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6573-./mpasoc_rpn.26505286.log-9852-[1785176950.871792] [l40615:270250:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6574:./mpasoc_rpn.26505286.log:9853:{"component": "ocean", "grid": "voronoi", "mode": "strong", "subdivision": 8, "n_ranks": 128, "n_cells": 655362, "n_edges": 1966080, "nlev": 20, "partition_method": "auto", "steps": 12, "dt": 60.0, "platform": "cpu", "compile_ms": 12817.6, "steady_median_ms": 309.0494, "step_latency_gate_loop_ms": 312.49, "step_latency_gate_loop_min_ms": 309.97, "per_step_ms": [12817.6, 312.2, 312.5, 312.5, 312.2, 310.2, 314.4, 313.2, 313.5, 310.0, 313.7, 310.7], "cells": 13107240, "cells_per_rank_achieved": 5120, "compile_prewarmed_by_parity_ref": false, "barotropic_solver": "explicit_substep", "halo_refresh": "in_step", "stage_halo_correct": true, "stage_halo_note": "per-step packed entry refresh + in-step stage-frontier refreshes (make_mpas_ocean_halo_refresh) \u2014 stage-correct; see docs/performance/scaling/mpas_ocean_distributed_stage_audit.md", "fused": {"compile_ms": 309.8, "scan_compile_ms": 14638.2, "step_latency_ms": 309.777, "block_ms": [2471.95, 2468.48], "parallel_block_ms": [2474.15, 2470.64], "fused_step_ms": 309.0494, "rank_imbalance": 1.0008, "rank_imbalance_per_block": [1.0007, 1.0009], "block_steps": 8, "n_blocks": 2, "probe_steps": 3}, "wet_cell": {"wet_cell_levels": 13057600, "wet_cell_levels_per_device": 102012.5, "wet_fraction": 0.9962127801123654, "wet_equals_total": false, "wet_cell_levels_per_device_min": 96040, "wet_cell_levels_per_device_max": 102420}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-27T18:29:59.635219+00:00", "grid": "voronoi", "component": "ocean", "resolution": "L8", "n_levels": 20, "precision": "float64", "decomposition": "cell_partition", "solver_variant": "mpas_ocean_explicit_substep", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 128, "n_gpus": 0, "device_count": 1, "process_count": 1, "devices_per_rank": 1, "cells_per_rank": 102400, "backend": "cpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "mpi4jax", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l40609.lvt.dkrz.de", "slurm_job_id": "26505286", "git_sha": "aabce0876", "gpu_direct_requested": false, "mpi4jax_cuda_support": false, "gpu_direct_active": false, "host_staged_halo": false, "partition_metrics": {"n_owned_cells": 5120, "n_halo_cells": 658, "owned_halo_ratio": 0.128515625, "n_neighbor_ranks": 7, "messages_per_exchange": 7, "halo_recv_cells": 658, "owned_send_cells": 661, "cells_per_rank_min": 5120, "cells_per_rank_max": 5121, "edge_cut_total": 80510, "max_neighbor_ranks": 8}, "extra": {"partition_method": "auto", "steps": 12, "warmup": 2, "parity_gate": false, "check_conservation": false, "halo_refresh": "in_step", "barotropic_solver": "explicit_substep", "block_steps": 8, "blocks": 2, "probe_steps": 3}}}
6575-./mpasoc_rpn.26505286.log:9854:[mpas-ocean np=128 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=12817.6ms fused=309.0494ms/step (probe_latency=309.777ms) gate_loop_latency=312.49ms/step
6576-./mpasoc_rpn.26505286.log-9855-[1785176950.832636] [l40612:241276:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6577-./mpasoc_rpn.26505286.log-9856-[1785176950.888182] [l40615:270244:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6578-./mpasoc_rpn.26505286.log-9857-[1785176950.895603] [l40615:270219:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6579---
6580-./mpasoc_rpn.26505286.log-13461-[1785177005.820720] [l40609:287386:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6581-./mpasoc_rpn.26505286.log-13462-[1785177005.824238] [l40609:287376:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6582-./mpasoc_rpn.26505286.log-13463-[1785177005.834299] [l40609:287398:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6583:./mpasoc_rpn.26505286.log:13464:{"component": "ocean", "grid": "voronoi", "mode": "strong", "subdivision": 8, "n_ranks": 256, "n_cells": 655362, "n_edges": 1966080, "nlev": 20, "partition_method": "auto", "steps": 12, "dt": 60.0, "platform": "cpu", "compile_ms": 12760.0, "steady_median_ms": 254.4062, "step_latency_gate_loop_ms": 237.42, "step_latency_gate_loop_min_ms": 235.0, "per_step_ms": [12760.0, 240.4, 237.5, 241.6, 237.9, 238.4, 237.3, 235.0, 236.5, 235.0, 243.1, 235.1], "cells": 13107240, "cells_per_rank_achieved": 2560, "compile_prewarmed_by_parity_ref": false, "barotropic_solver": "explicit_substep", "halo_refresh": "in_step", "stage_halo_correct": true, "stage_halo_note": "per-step packed entry refresh + in-step stage-frontier refreshes (make_mpas_ocean_halo_refresh) \u2014 stage-correct; see docs/performance/scaling/mpas_ocean_distributed_stage_audit.md", "fused": {"compile_ms": 232.9, "scan_compile_ms": 14248.9, "step_latency_ms": 233.377, "block_ms": [2034.69, 2035.53], "parallel_block_ms": [2034.82, 2035.68], "fused_step_ms": 254.4062, "rank_imbalance": 1.0002, "rank_imbalance_per_block": [1.0004, 1.0001], "block_steps": 8, "n_blocks": 2, "probe_steps": 3}, "wet_cell": {"wet_cell_levels": 13057600, "wet_cell_levels_per_device": 51006.25, "wet_fraction": 0.9962127801123654, "wet_equals_total": false, "wet_cell_levels_per_device_min": 44840, "wet_cell_levels_per_device_max": 51220}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-27T18:30:53.404989+00:00", "grid": "voronoi", "component": "ocean", "resolution": "L8", "n_levels": 20, "precision": "float64", "decomposition": "cell_partition", "solver_variant": "mpas_ocean_explicit_substep", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 256, "n_gpus": 0, "device_count": 1, "process_count": 1, "devices_per_rank": 1, "cells_per_rank": 51200, "backend": "cpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "mpi4jax", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l40609.lvt.dkrz.de", "slurm_job_id": "26505286", "git_sha": "aabce0876", "gpu_direct_requested": false, "mpi4jax_cuda_support": false, "gpu_direct_active": false, "host_staged_halo": false, "partition_metrics": {"n_owned_cells": 2560, "n_halo_cells": 478, "owned_halo_ratio": 0.18671875, "n_neighbor_ranks": 6, "messages_per_exchange": 6, "halo_recv_cells": 478, "owned_send_cells": 477, "cells_per_rank_min": 2560, "cells_per_rank_max": 2561, "edge_cut_total": 114563, "max_neighbor_ranks": 8}, "extra": {"partition_method": "auto", "steps": 12, "warmup": 2, "parity_gate": false, "check_conservation": false, "halo_refresh": "in_step", "barotropic_solver": "explicit_substep", "block_steps": 8, "blocks": 2, "probe_steps": 3}}}
6584-./mpasoc_rpn.26505286.log:13465:[mpas-ocean np=256 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=12760.0ms fused=254.4062ms/step (probe_latency=233.377ms) gate_loop_latency=237.42ms/step
6585-./mpasoc_rpn.26505286.log-13466-[1785177005.816587] [l40609:287369:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6586-./mpasoc_rpn.26505286.log-13467-=== RESULTS (32 ranks/node throughout) ===
6587-./mpasoc_rpn.26505286.log-13468-s7 np32:    190.22 ms
--
6680---
6681-./mpasoc_512.26508063.log-14180-[1785187449.129135] [l40112:76380:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6682-./mpasoc_512.26508063.log-14181-[1785187448.195394] [l40111:48882:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6683-./mpasoc_512.26508063.log-14182-[1785187449.303553] [l40107:171225:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6684:./mpasoc_512.26508063.log:14183:{"component": "ocean", "grid": "voronoi", "mode": "strong", "subdivision": 8, "n_ranks": 512, "n_cells": 655362, "n_edges": 1966080, "nlev": 20, "partition_method": "auto", "steps": 12, "dt": 60.0, "platform": "cpu", "compile_ms": 14009.9, "steady_median_ms": 194.8, "step_latency_gate_loop_ms": 179.54, "step_latency_gate_loop_min_ms": 178.51, "per_step_ms": [14009.9, 186.5, 180.7, 179.4, 181.3, 183.2, 179.6, 178.7, 181.2, 179.2, 178.9, 178.5], "cells": 13107240, "cells_per_rank_achieved": 1280, "compile_prewarmed_by_parity_ref": false, "barotropic_solver": "explicit_substep", "halo_refresh": "in_step", "stage_halo_correct": true, "stage_halo_note": "per-step packed entry refresh + in-step stage-frontier refreshes (make_mpas_ocean_halo_refresh) \u2014 stage-correct; see docs/performance/scaling/mpas_ocean_distributed_stage_audit.md", "fused": {"compile_ms": 174.3, "scan_compile_ms": 15350.1, "step_latency_ms": 175.611, "block_ms": [1554.4, 1561.75], "parallel_block_ms": [1554.82, 1561.98], "fused_step_ms": 194.8, "rank_imbalance": 1.0002, "rank_imbalance_per_block": [1.0002, 1.0002], "block_steps": 8, "n_blocks": 2, "probe_steps": 3}, "wet_cell": {"wet_cell_levels": 13057600, "wet_cell_levels_per_device": 25503.125, "wet_fraction": 0.9962127801123654, "wet_equals_total": false, "wet_cell_levels_per_device_min": 19240, "wet_cell_levels_per_device_max": 25620}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-27T21:24:56.660152+00:00", "grid": "voronoi", "component": "ocean", "resolution": "L8", "n_levels": 20, "precision": "float64", "decomposition": "cell_partition", "solver_variant": "mpas_ocean_explicit_substep", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 512, "n_gpus": 0, "device_count": 1, "process_count": 1, "devices_per_rank": 1, "cells_per_rank": 25600, "backend": "cpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "mpi4jax", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l40107.lvt.dkrz.de", "slurm_job_id": "26508063", "git_sha": "346a77f20", "gpu_direct_requested": false, "mpi4jax_cuda_support": false, "gpu_direct_active": false, "host_staged_halo": false, "partition_metrics": {"n_owned_cells": 1280, "n_halo_cells": 346, "owned_halo_ratio": 0.2703125, "n_neighbor_ranks": 5, "messages_per_exchange": 5, "halo_recv_cells": 346, "owned_send_cells": 336, "cells_per_rank_min": 1280, "cells_per_rank_max": 1281, "edge_cut_total": 162499, "max_neighbor_ranks": 9}, "extra": {"partition_method": "auto", "steps": 12, "warmup": 2, "parity_gate": false, "check_conservation": false, "halo_refresh": "in_step", "barotropic_solver": "explicit_substep", "block_steps": 8, "blocks": 2, "probe_steps": 3}}}
6685-./mpasoc_512.26508063.log:14184:[mpas-ocean np=512 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=14009.9ms fused=194.8ms/step (probe_latency=175.611ms) gate_loop_latency=179.54ms/step
6686-./mpasoc_512.26508063.log-14185-[1785187449.321319] [l40107:171211:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6687-./mpasoc_512.26508063.log-14186-[1785187448.963288] [l40107:171214:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6688-./mpasoc_512.26508063.log-14187-[1785187448.256137] [l40126:572777:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6689---
6690-./cpu_f32.26534068.log-19846-[1785310651.666866] [l10501:731285:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6691-./cpu_f32.26534068.log-19847-[1785310651.689656] [l10501:731309:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6692-./cpu_f32.26534068.log-19848-[1785310651.713175] [l10501:731305:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6693:./cpu_f32.26534068.log:19849:{"component": "ocean", "grid": "voronoi", "mode": "strong", "subdivision": 8, "n_ranks": 64, "n_cells": 655362, "n_edges": 1966080, "nlev": 20, "partition_method": "geometric", "steps": 10, "dt": 60.0, "platform": "cpu", "compile_ms": 13967.3, "steady_median_ms": 491.1337, "step_latency_gate_loop_ms": 460.91, "step_latency_gate_loop_min_ms": 458.56, "per_step_ms": [13967.3, 463.8, 464.1, 464.7, 459.3, 460.2, 460.9, 460.9, 461.0, 458.6], "cells": 13107240, "cells_per_rank_achieved": 10240, "compile_prewarmed_by_parity_ref": false, "barotropic_solver": "explicit_substep", "halo_refresh": "in_step", "stage_halo_correct": true, "stage_halo_note": "per-step packed entry refresh + in-step stage-frontier refreshes (make_mpas_ocean_halo_refresh) \u2014 stage-correct; see docs/performance/scaling/mpas_ocean_distributed_stage_audit.md", "fused": {"compile_ms": 457.3, "scan_compile_ms": 17036.3, "step_latency_ms": 457.995, "block_ms": [3932.62, 3917.69], "parallel_block_ms": [3936.94, 3921.2], "fused_step_ms": 491.1337, "rank_imbalance": 1.0009, "rank_imbalance_per_block": [1.0009, 1.0009], "block_steps": 8, "n_blocks": 2, "probe_steps": 3}, "wet_cell": {"wet_cell_levels": 13057600, "wet_cell_levels_per_device": 204025.0, "wet_fraction": 0.9962127801123654, "wet_equals_total": false, "wet_cell_levels_per_device_min": 198440, "wet_cell_levels_per_device_max": 204820}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-29T07:38:38.349085+00:00", "grid": "voronoi", "component": "ocean", "resolution": "L8", "n_levels": 20, "precision": "float32", "decomposition": "cell_partition", "solver_variant": "mpas_ocean_explicit_substep", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 64, "n_gpus": 0, "device_count": 1, "process_count": 1, "devices_per_rank": 1, "cells_per_rank": 204800, "backend": "cpu", "precision_knobs": {"JAX_ENABLE_X64": "0", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "mpi4jax", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l10501.lvt.dkrz.de", "slurm_job_id": "26534068", "git_sha": "4c44e6507", "gpu_direct_requested": false, "mpi4jax_cuda_support": false, "gpu_direct_active": false, "host_staged_halo": false, "partition_metrics": {"n_owned_cells": 10240, "n_halo_cells": 904, "owned_halo_ratio": 0.08828125, "n_neighbor_ranks": 7, "messages_per_exchange": 7, "halo_recv_cells": 904, "owned_send_cells": 906, "cells_per_rank_min": 10240, "cells_per_rank_max": 10241, "edge_cut_total": 56765, "max_neighbor_ranks": 8}, "extra": {"partition_method": "geometric", "steps": 10, "warmup": 2, "parity_gate": false, "check_conservation": false, "halo_refresh": "in_step", "barotropic_solver": "explicit_substep", "block_steps": 8, "blocks": 2, "probe_steps": 3}}}
6694-./cpu_f32.26534068.log:19850:[mpas-ocean np=64 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=13967.3ms fused=491.1337ms/step (probe_latency=457.995ms) gate_loop_latency=460.91ms/step
6695-./cpu_f32.26534068.log-19851-[1785310651.705998] [l10501:731278:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6696-./cpu_f32.26534068.log-19852-[1785310651.617794] [l10501:731304:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6697-./cpu_f32.26534068.log-19853-[1785310651.709972] [l10501:731279:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6698---
6699-./cpu_f32.26534068.log-23765-[1785310722.408745] [l10503:446185:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6700-./cpu_f32.26534068.log-23766-[1785310722.377428] [l10503:446183:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6701-./cpu_f32.26534068.log-23767-[1785310722.408870] [l10503:446172:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6702:./cpu_f32.26534068.log:23768:{"component": "ocean", "grid": "voronoi", "mode": "strong", "subdivision": 8, "n_ranks": 256, "n_cells": 655362, "n_edges": 1966080, "nlev": 20, "partition_method": "geometric", "steps": 10, "dt": 60.0, "platform": "cpu", "compile_ms": 12882.5, "steady_median_ms": 258.8344, "step_latency_gate_loop_ms": 238.02, "step_latency_gate_loop_min_ms": 236.57, "per_step_ms": [12882.5, 243.9, 238.5, 243.9, 238.8, 237.1, 238.3, 237.4, 237.7, 236.6], "cells": 13107240, "cells_per_rank_achieved": 2560, "compile_prewarmed_by_parity_ref": false, "barotropic_solver": "explicit_substep", "halo_refresh": "in_step", "stage_halo_correct": true, "stage_halo_note": "per-step packed entry refresh + in-step stage-frontier refreshes (make_mpas_ocean_halo_refresh) \u2014 stage-correct; see docs/performance/scaling/mpas_ocean_distributed_stage_audit.md", "fused": {"compile_ms": 233.7, "scan_compile_ms": 14646.6, "step_latency_ms": 233.395, "block_ms": [2070.33, 2068.75], "parallel_block_ms": [2071.87, 2069.48], "fused_step_ms": 258.8344, "rank_imbalance": 1.0005, "rank_imbalance_per_block": [1.0006, 1.0004], "block_steps": 8, "n_blocks": 2, "probe_steps": 3}, "wet_cell": {"wet_cell_levels": 13057600, "wet_cell_levels_per_device": 51006.25, "wet_fraction": 0.9962127801123654, "wet_equals_total": false, "wet_cell_levels_per_device_min": 44840, "wet_cell_levels_per_device_max": 51220}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-29T07:39:26.728045+00:00", "grid": "voronoi", "component": "ocean", "resolution": "L8", "n_levels": 20, "precision": "float32", "decomposition": "cell_partition", "solver_variant": "mpas_ocean_explicit_substep", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 256, "n_gpus": 0, "device_count": 1, "process_count": 1, "devices_per_rank": 1, "cells_per_rank": 51200, "backend": "cpu", "precision_knobs": {"JAX_ENABLE_X64": "0", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "mpi4jax", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l10501.lvt.dkrz.de", "slurm_job_id": "26534068", "git_sha": "4c44e6507", "gpu_direct_requested": false, "mpi4jax_cuda_support": false, "gpu_direct_active": false, "host_staged_halo": false, "partition_metrics": {"n_owned_cells": 2560, "n_halo_cells": 478, "owned_halo_ratio": 0.18671875, "n_neighbor_ranks": 6, "messages_per_exchange": 6, "halo_recv_cells": 478, "owned_send_cells": 477, "cells_per_rank_min": 2560, "cells_per_rank_max": 2561, "edge_cut_total": 114563, "max_neighbor_ranks": 8}, "extra": {"partition_method": "geometric", "steps": 10, "warmup": 2, "parity_gate": false, "check_conservation": false, "halo_refresh": "in_step", "barotropic_solver": "explicit_substep", "block_steps": 8, "blocks": 2, "probe_steps": 3}}}
6703-./cpu_f32.26534068.log:23769:[mpas-ocean np=256 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=12882.5ms fused=258.8344ms/step (probe_latency=233.395ms) gate_loop_latency=238.02ms/step
6704-./cpu_f32.26534068.log-23770-[1785310722.380564] [l10503:446164:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6705-./cpu_f32.26534068.log-23771-[1785310721.985782] [l10511:485943:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6706-./cpu_f32.26534068.log-23772-[1785310721.912111] [l10511:485939:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6707---
6708-./cpu_f32.26534068.log-31935-[1785310771.358683] [l10525:2627697:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6709-./cpu_f32.26534068.log-31936-[1785310771.293733] [l10525:2627688:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6710-./cpu_f32.26534068.log-31937-[1785310771.578716] [l10514:3182138:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6711:./cpu_f32.26534068.log:31938:{"component": "ocean", "grid": "voronoi", "mode": "strong", "subdivision": 8, "n_ranks": 512, "n_cells": 655362, "n_edges": 1966080, "nlev": 20, "partition_method": "geometric", "steps": 10, "dt": 60.0, "platform": "cpu", "compile_ms": 14299.2, "steady_median_ms": 195.6313, "step_latency_gate_loop_ms": 179.19, "step_latency_gate_loop_min_ms": 178.33, "per_step_ms": [14299.2, 182.2, 179.8, 179.2, 179.2, 181.8, 179.1, 178.3, 181.1, 178.9], "cells": 13107240, "cells_per_rank_achieved": 1280, "compile_prewarmed_by_parity_ref": false, "barotropic_solver": "explicit_substep", "halo_refresh": "in_step", "stage_halo_correct": true, "stage_halo_note": "per-step packed entry refresh + in-step stage-frontier refreshes (make_mpas_ocean_halo_refresh) \u2014 stage-correct; see docs/performance/scaling/mpas_ocean_distributed_stage_audit.md", "fused": {"compile_ms": 173.8, "scan_compile_ms": 15371.2, "step_latency_ms": 174.221, "block_ms": [1566.45, 1562.97], "parallel_block_ms": [1566.63, 1563.47], "fused_step_ms": 195.6313, "rank_imbalance": 1.0002, "rank_imbalance_per_block": [1.0001, 1.0002], "block_steps": 8, "n_blocks": 2, "probe_steps": 3}, "wet_cell": {"wet_cell_levels": 13057600, "wet_cell_levels_per_device": 25503.125, "wet_fraction": 0.9962127801123654, "wet_equals_total": false, "wet_cell_levels_per_device_min": 19240, "wet_cell_levels_per_device_max": 25620}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-29T07:40:17.641961+00:00", "grid": "voronoi", "component": "ocean", "resolution": "L8", "n_levels": 20, "precision": "float32", "decomposition": "cell_partition", "solver_variant": "mpas_ocean_explicit_substep", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 512, "n_gpus": 0, "device_count": 1, "process_count": 1, "devices_per_rank": 1, "cells_per_rank": 25600, "backend": "cpu", "precision_knobs": {"JAX_ENABLE_X64": "0", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "mpi4jax", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l10501.lvt.dkrz.de", "slurm_job_id": "26534068", "git_sha": "4c44e6507", "gpu_direct_requested": false, "mpi4jax_cuda_support": false, "gpu_direct_active": false, "host_staged_halo": false, "partition_metrics": {"n_owned_cells": 1280, "n_halo_cells": 346, "owned_halo_ratio": 0.2703125, "n_neighbor_ranks": 5, "messages_per_exchange": 5, "halo_recv_cells": 346, "owned_send_cells": 336, "cells_per_rank_min": 1280, "cells_per_rank_max": 1281, "edge_cut_total": 162499, "max_neighbor_ranks": 9}, "extra": {"partition_method": "geometric", "steps": 10, "warmup": 2, "parity_gate": false, "check_conservation": false, "halo_refresh": "in_step", "barotropic_solver": "explicit_substep", "block_steps": 8, "blocks": 2, "probe_steps": 3}}}
6712-./cpu_f32.26534068.log:31939:[mpas-ocean np=512 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=14299.2ms fused=195.6313ms/step (probe_latency=174.221ms) gate_loop_latency=179.19ms/step
6713-./cpu_f32.26534068.log-31940-[1785310771.701387] [l10512:3170550:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6714-./cpu_f32.26534068.log-31941-[1785310771.787357] [l10512:3170560:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6715-./cpu_f32.26534068.log-31942-[1785310771.842842] [l10512:3170561:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
--
6767-./mpas_hlo.26480096.log-12-W0726 12:49:02.811921  727949 pjrt_client.cc:1604] WatchTasksAsync failed for task 0: UNAVAILABLE: failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.34:64992: Failed to connect to remote host: Connection refused
6768-./mpas_hlo.26480096.log-13-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
6769-./mpas_hlo.26480096.log-14-:UNKNOWN:Error received from peer  {grpc_message:"failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.34:64992: Failed to connect to remote host: Connection refused", grpc_status:14}
6770---
6771:./mpasoc_metis.26600094.log:1:outdir=/scratch/b/b381103/legoesm_scaling/mpasoc_metis_j26600094
6772:./mpasoc_metis.26600094.log-2---- s7 np=32 partition=geometric dist=block:cyclic ---
6773:./mpasoc_metis.26600094.log-3-[l20119.lvt.dkrz.de:1439071] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
6774:./mpasoc_metis.26600094.log-4-[l20119.lvt.dkrz.de:1439096] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
6775---
6776:./mpasoc_metis.26600094.log-447-[1785518297.281831] [l20119:1439072:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6777:./mpasoc_metis.26600094.log-448-[1785518297.249801] [l20119:1439095:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6778:./mpasoc_metis.26600094.log-449-[1785518297.254104] [l20119:1439079:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6779:./mpasoc_metis.26600094.log:450:{"component": "ocean", "grid": "voronoi", "mode": "strong", "subdivision": 7, "n_ranks": 32, "n_cells": 163842, "n_edges": 491520, "nlev": 20, "partition_method": "geometric", "steps": 12, "dt": 60.0, "platform": "cpu", "compile_ms": 12255.1, "steady_median_ms": 189.8206, "step_latency_gate_loop_ms": 187.13, "step_latency_gate_loop_min_ms": 185.51, "per_step_ms": [12255.1, 188.4, 187.4, 187.6, 187.4, 186.3, 186.9, 186.0, 188.5, 186.6, 185.5, 190.5], "cells": 3276840, "cells_per_rank_achieved": 5120, "compile_prewarmed_by_parity_ref": false, "barotropic_solver": "explicit_substep", "halo_refresh": "in_step", "stage_halo_correct": true, "stage_halo_note": "per-step packed entry refresh + in-step stage-frontier refreshes (make_mpas_ocean_halo_refresh) \u2014 stage-correct; see docs/performance/scaling/mpas_ocean_distributed_stage_audit.md", "fused": {"compile_ms": 183.7, "scan_compile_ms": 13431.3, "step_latency_ms": 187.99, "block_ms": [1516.95, 1516.98], "parallel_block_ms": [1518.81, 1518.32], "fused_step_ms": 189.8206, "rank_imbalance": 1.0011, "rank_imbalance_per_block": [1.0012, 1.0011], "block_steps": 8, "n_blocks": 2, "probe_steps": 3}, "wet_cell": {"wet_cell_levels": 3264640, "wet_cell_levels_per_device": 102020.0, "wet_fraction": 0.9962769009167368, "wet_equals_total": false, "wet_cell_levels_per_device_min": 100740, "wet_cell_levels_per_device_max": 102420}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-31T17:19:09.693551+00:00", "grid": "voronoi", "component": "ocean", "resolution": "L7", "n_levels": 20, "precision": "float64", "decomposition": "cell_partition", "solver_variant": "mpas_ocean_explicit_substep", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 32, "n_gpus": 0, "device_count": 1, "process_count": 1, "devices_per_rank": 1, "cells_per_rank": 102401, "backend": "cpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "mpi4jax", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l20119.lvt.dkrz.de", "slurm_job_id": "26600094", "git_sha": "7151d12a1", "gpu_direct_requested": false, "mpi4jax_cuda_support": false, "gpu_direct_active": false, "host_staged_halo": false, "partition_metrics": {"n_owned_cells": 5120, "n_halo_cells": 628, "owned_halo_ratio": 0.12265625, "n_neighbor_ranks": 6, "messages_per_exchange": 6, "halo_recv_cells": 628, "owned_send_cells": 626, "cells_per_rank_min": 5120, "cells_per_rank_max": 5121, "edge_cut_total": 20654, "max_neighbor_ranks": 8}, "extra": {"partition_method": "geometric", "steps": 12, "warmup": 2, "parity_gate": false, "check_conservation": false, "halo_refresh": "in_step", "barotropic_solver": "explicit_substep", "block_steps": 8, "blocks": 2, "probe_steps": 3}}}
6780:./mpasoc_metis.26600094.log-451-[mpas-ocean np=32 L7 nCells=163842 nlev=20 solver=explicit_substep halo=in_step] compile=12255.1ms fused=189.8206ms/step (probe_latency=187.99ms) gate_loop_latency=187.13ms/step
6781:./mpasoc_metis.26600094.log-452-[1785518297.289277] [l20119:1439070:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6782:./mpasoc_metis.26600094.log-453---- s8 np=128 partition=geometric dist=block:cyclic ---
6783---
6784:./mpasoc_metis.26600094.log-2146-[1785518353.389333] [l20119:1440311:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6785:./mpasoc_metis.26600094.log-2147-[1785518353.342035] [l20119:1440310:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6786:./mpasoc_metis.26600094.log-2148-[1785518353.342454] [l20119:1440297:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6787:./mpasoc_metis.26600094.log:2149:{"component": "ocean", "grid": "voronoi", "mode": "strong", "subdivision": 8, "n_ranks": 128, "n_cells": 655362, "n_edges": 1966080, "nlev": 20, "partition_method": "geometric", "steps": 12, "dt": 60.0, "platform": "cpu", "compile_ms": 12628.0, "steady_median_ms": 308.9581, "step_latency_gate_loop_ms": 311.3, "step_latency_gate_loop_min_ms": 308.75, "per_step_ms": [12628.0, 312.0, 311.7, 310.9, 309.9, 310.0, 312.0, 311.9, 312.1, 308.7, 309.7, 311.7], "cells": 13107240, "cells_per_rank_achieved": 5120, "compile_prewarmed_by_parity_ref": false, "barotropic_solver": "explicit_substep", "halo_refresh": "in_step", "stage_halo_correct": true, "stage_halo_note": "per-step packed entry refresh + in-step stage-frontier refreshes (make_mpas_ocean_halo_refresh) \u2014 stage-correct; see docs/performance/scaling/mpas_ocean_distributed_stage_audit.md", "fused": {"compile_ms": 308.8, "scan_compile_ms": 14719.4, "step_latency_ms": 311.988, "block_ms": [2471.77, 2467.17], "parallel_block_ms": [2472.65, 2470.68], "fused_step_ms": 308.9581, "rank_imbalance": 1.0008, "rank_imbalance_per_block": [1.0008, 1.0008], "block_steps": 8, "n_blocks": 2, "probe_steps": 3}, "wet_cell": {"wet_cell_levels": 13057600, "wet_cell_levels_per_device": 102012.5, "wet_fraction": 0.9962127801123654, "wet_equals_total": false, "wet_cell_levels_per_device_min": 96040, "wet_cell_levels_per_device_max": 102420}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-31T17:20:07.513962+00:00", "grid": "voronoi", "component": "ocean", "resolution": "L8", "n_levels": 20, "precision": "float64", "decomposition": "cell_partition", "solver_variant": "mpas_ocean_explicit_substep", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 128, "n_gpus": 0, "device_count": 1, "process_count": 1, "devices_per_rank": 1, "cells_per_rank": 102400, "backend": "cpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "mpi4jax", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l20119.lvt.dkrz.de", "slurm_job_id": "26600094", "git_sha": "7151d12a1", "gpu_direct_requested": false, "mpi4jax_cuda_support": false, "gpu_direct_active": false, "host_staged_halo": false, "partition_metrics": {"n_owned_cells": 5120, "n_halo_cells": 658, "owned_halo_ratio": 0.128515625, "n_neighbor_ranks": 7, "messages_per_exchange": 7, "halo_recv_cells": 658, "owned_send_cells": 661, "cells_per_rank_min": 5120, "cells_per_rank_max": 5121, "edge_cut_total": 80510, "max_neighbor_ranks": 8}, "extra": {"partition_method": "geometric", "steps": 12, "warmup": 2, "parity_gate": false, "check_conservation": false, "halo_refresh": "in_step", "barotropic_solver": "explicit_substep", "block_steps": 8, "blocks": 2, "probe_steps": 3}}}
6788:./mpasoc_metis.26600094.log:2150:[mpas-ocean np=128 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=12628.0ms fused=308.9581ms/step (probe_latency=311.988ms) gate_loop_latency=311.30ms/step
6789:./mpasoc_metis.26600094.log-2151-[1785518353.372656] [l20119:1440282:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6790:./mpasoc_metis.26600094.log-2152-[1785518357.149452] [l20130:50192:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6791:./mpasoc_metis.26600094.log-2153-[1785518357.157028] [l20130:50198:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6792---
6793:./mpasoc_metis.26600094.log-2684-[1785518411.087402] [l20119:1441479:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6794:./mpasoc_metis.26600094.log-2685-[1785518411.080442] [l20119:1441486:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6795:./mpasoc_metis.26600094.log-2686-[1785518411.064649] [l20119:1441490:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6796:./mpasoc_metis.26600094.log:2687:{"component": "ocean", "grid": "voronoi", "mode": "strong", "subdivision": 7, "n_ranks": 32, "n_cells": 163842, "n_edges": 491520, "nlev": 20, "partition_method": "metis", "steps": 12, "dt": 60.0, "platform": "cpu", "compile_ms": 12579.2, "steady_median_ms": 191.9856, "step_latency_gate_loop_ms": 190.29, "step_latency_gate_loop_min_ms": 188.92, "per_step_ms": [12579.2, 193.6, 191.0, 190.2, 188.9, 191.2, 189.7, 189.8, 188.9, 192.5, 191.4, 190.4], "cells": 3276840, "cells_per_rank_achieved": 5120, "compile_prewarmed_by_parity_ref": false, "barotropic_solver": "explicit_substep", "halo_refresh": "in_step", "stage_halo_correct": true, "stage_halo_note": "per-step packed entry refresh + in-step stage-frontier refreshes (make_mpas_ocean_halo_refresh) \u2014 stage-correct; see docs/performance/scaling/mpas_ocean_distributed_stage_audit.md", "fused": {"compile_ms": 186.8, "scan_compile_ms": 14192.4, "step_latency_ms": 188.993, "block_ms": [1534.07, 1532.56], "parallel_block_ms": [1535.85, 1535.92], "fused_step_ms": 191.9856, "rank_imbalance": 1.0012, "rank_imbalance_per_block": [1.0012, 1.0011], "block_steps": 8, "n_blocks": 2, "probe_steps": 3}, "wet_cell": {"wet_cell_levels": 3264640, "wet_cell_levels_per_device": 102020.0, "wet_fraction": 0.9962769009167368, "wet_equals_total": false, "wet_cell_levels_per_device_min": 96800, "wet_cell_levels_per_device_max": 102720}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-31T17:20:51.591156+00:00", "grid": "voronoi", "component": "ocean", "resolution": "L7", "n_levels": 20, "precision": "float64", "decomposition": "cell_partition", "solver_variant": "mpas_ocean_explicit_substep", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 32, "n_gpus": 0, "device_count": 1, "process_count": 1, "devices_per_rank": 1, "cells_per_rank": 102401, "backend": "cpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "mpi4jax", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l20119.lvt.dkrz.de", "slurm_job_id": "26600094", "git_sha": "7151d12a1", "gpu_direct_requested": false, "mpi4jax_cuda_support": false, "gpu_direct_active": false, "host_staged_halo": false, "partition_metrics": {"n_owned_cells": 5133, "n_halo_cells": 549, "owned_halo_ratio": 0.10695499707773232, "n_neighbor_ranks": 6, "messages_per_exchange": 6, "halo_recv_cells": 549, "owned_send_cells": 556, "cells_per_rank_min": 5100, "cells_per_rank_max": 5145, "edge_cut_total": 18784, "max_neighbor_ranks": 7}, "extra": {"partition_method": "metis", "steps": 12, "warmup": 2, "parity_gate": false, "check_conservation": false, "halo_refresh": "in_step", "barotropic_solver": "explicit_substep", "block_steps": 8, "blocks": 2, "probe_steps": 3}}}
6797:./mpasoc_metis.26600094.log-2688-[mpas-ocean np=32 L7 nCells=163842 nlev=20 solver=explicit_substep halo=in_step] compile=12579.2ms fused=191.9856ms/step (probe_latency=188.993ms) gate_loop_latency=190.29ms/step
6798:./mpasoc_metis.26600094.log-2689-[1785518411.064180] [l20119:1441477:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6799:./mpasoc_metis.26600094.log-2690-[1785518411.073813] [l20119:1441504:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6800---
6801:./mpasoc_metis.26600094.log-4384-[1785518454.816638] [l20119:1442698:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6802:./mpasoc_metis.26600094.log-4385-[1785518454.766060] [l20119:1442707:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6803:./mpasoc_metis.26600094.log-4386-[1785518454.774753] [l20119:1442679:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6804:./mpasoc_metis.26600094.log:4387:{"component": "ocean", "grid": "voronoi", "mode": "strong", "subdivision": 8, "n_ranks": 128, "n_cells": 655362, "n_edges": 1966080, "nlev": 20, "partition_method": "metis", "steps": 12, "dt": 60.0, "platform": "cpu", "compile_ms": 13554.5, "steady_median_ms": 333.39, "step_latency_gate_loop_ms": 348.11, "step_latency_gate_loop_min_ms": 345.44, "per_step_ms": [13554.5, 350.4, 352.4, 348.9, 348.3, 349.4, 348.0, 347.5, 347.1, 348.5, 345.4, 347.6], "cells": 13107240, "cells_per_rank_achieved": 5120, "compile_prewarmed_by_parity_ref": false, "barotropic_solver": "explicit_substep", "halo_refresh": "in_step", "stage_halo_correct": true, "stage_halo_note": "per-step packed entry refresh + in-step stage-frontier refreshes (make_mpas_ocean_halo_refresh) \u2014 stage-correct; see docs/performance/scaling/mpas_ocean_distributed_stage_audit.md", "fused": {"compile_ms": 343.0, "scan_compile_ms": 15677.5, "step_latency_ms": 343.721, "block_ms": [2663.79, 2664.88], "parallel_block_ms": [2666.74, 2667.5], "fused_step_ms": 333.39, "rank_imbalance": 1.0007, "rank_imbalance_per_block": [1.0007, 1.0008], "block_steps": 8, "n_blocks": 2, "probe_steps": 3}, "wet_cell": {"wet_cell_levels": 13057600, "wet_cell_levels_per_device": 102012.5, "wet_fraction": 0.9962127801123654, "wet_equals_total": false, "wet_cell_levels_per_device_min": 78000, "wet_cell_levels_per_device_max": 102880}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-31T17:21:47.977228+00:00", "grid": "voronoi", "component": "ocean", "resolution": "L8", "n_levels": 20, "precision": "float64", "decomposition": "cell_partition", "solver_variant": "mpas_ocean_explicit_substep", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 128, "n_gpus": 0, "device_count": 1, "process_count": 1, "devices_per_rank": 1, "cells_per_rank": 102400, "backend": "cpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "mpi4jax", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l20119.lvt.dkrz.de", "slurm_job_id": "26600094", "git_sha": "7151d12a1", "gpu_direct_requested": false, "mpi4jax_cuda_support": false, "gpu_direct_active": false, "host_staged_halo": false, "partition_metrics": {"n_owned_cells": 5113, "n_halo_cells": 562, "owned_halo_ratio": 0.10991590064541365, "n_neighbor_ranks": 4, "messages_per_exchange": 4, "halo_recv_cells": 562, "owned_send_cells": 553, "cells_per_rank_min": 5093, "cells_per_rank_max": 5144, "edge_cut_total": 75748, "max_neighbor_ranks": 8}, "extra": {"partition_method": "metis", "steps": 12, "warmup": 2, "parity_gate": false, "check_conservation": false, "halo_refresh": "in_step", "barotropic_solver": "explicit_substep", "block_steps": 8, "blocks": 2, "probe_steps": 3}}}
6805:./mpasoc_metis.26600094.log:4388:[mpas-ocean np=128 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=13554.5ms fused=333.39ms/step (probe_latency=343.721ms) gate_loop_latency=348.11ms/step
6806:./mpasoc_metis.26600094.log-4389-[1785518454.797213] [l20119:1442677:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6807:./mpasoc_metis.26600094.log-4390-[1785518454.772658] [l20119:1442708:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6808:./mpasoc_metis.26600094.log-4391-[1785518454.799331] [l20130:51417:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6809---
6810:./mpasoc_metis.26600094.log-6252-[1785518521.847542] [l20119:1443892:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6811:./mpasoc_metis.26600094.log-6253-[1785518521.996105] [l20130:52583:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6812:./mpasoc_metis.26600094.log-6254-[1785518521.884325] [l20119:1443901:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6813:./mpasoc_metis.26600094.log:6255:{"component": "ocean", "grid": "voronoi", "mode": "strong", "subdivision": 8, "n_ranks": 128, "n_cells": 655362, "n_edges": 1966080, "nlev": 20, "partition_method": "metis", "steps": 12, "dt": 60.0, "platform": "cpu", "compile_ms": 14329.9, "steady_median_ms": 537.905, "step_latency_gate_loop_ms": 557.24, "step_latency_gate_loop_min_ms": 553.39, "per_step_ms": [14329.9, 558.5, 561.4, 554.0, 558.4, 555.1, 560.6, 569.8, 559.2, 554.6, 556.1, 553.4], "cells": 13107240, "cells_per_rank_achieved": 5120, "compile_prewarmed_by_parity_ref": false, "barotropic_solver": "explicit_substep", "halo_refresh": "in_step", "stage_halo_correct": true, "stage_halo_note": "per-step packed entry refresh + in-step stage-frontier refreshes (make_mpas_ocean_halo_refresh) \u2014 stage-correct; see docs/performance/scaling/mpas_ocean_distributed_stage_audit.md", "fused": {"compile_ms": 547.5, "scan_compile_ms": 17807.2, "step_latency_ms": 547.997, "block_ms": [4295.08, 4293.81], "parallel_block_ms": [4303.79, 4302.69], "fused_step_ms": 537.905, "rank_imbalance": 1.0021, "rank_imbalance_per_block": [1.002, 1.0023], "block_steps": 8, "n_blocks": 2, "probe_steps": 3}, "wet_cell": {"wet_cell_levels": 13057600, "wet_cell_levels_per_device": 102012.5, "wet_fraction": 0.9962127801123654, "wet_equals_total": false, "wet_cell_levels_per_device_min": 78000, "wet_cell_levels_per_device_max": 102880}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-31T17:23:14.881483+00:00", "grid": "voronoi", "component": "ocean", "resolution": "L8", "n_levels": 20, "precision": "float64", "decomposition": "cell_partition", "solver_variant": "mpas_ocean_explicit_substep", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 128, "n_gpus": 0, "device_count": 1, "process_count": 1, "devices_per_rank": 1, "cells_per_rank": 102400, "backend": "cpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "mpi4jax", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l20119.lvt.dkrz.de", "slurm_job_id": "26600094", "git_sha": "7151d12a1", "gpu_direct_requested": false, "mpi4jax_cuda_support": false, "gpu_direct_active": false, "host_staged_halo": false, "partition_metrics": {"n_owned_cells": 5113, "n_halo_cells": 562, "owned_halo_ratio": 0.10991590064541365, "n_neighbor_ranks": 4, "messages_per_exchange": 4, "halo_recv_cells": 562, "owned_send_cells": 553, "cells_per_rank_min": 5093, "cells_per_rank_max": 5144, "edge_cut_total": 75748, "max_neighbor_ranks": 8}, "extra": {"partition_method": "metis", "steps": 12, "warmup": 2, "parity_gate": false, "check_conservation": false, "halo_refresh": "in_step", "barotropic_solver": "explicit_substep", "block_steps": 8, "blocks": 2, "probe_steps": 3}}}
6814:./mpasoc_metis.26600094.log:6256:[mpas-ocean np=128 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=14329.9ms fused=537.905ms/step (probe_latency=547.997ms) gate_loop_latency=557.24ms/step
6815:./mpasoc_metis.26600094.log-6257-[1785518521.623596] [l20119:1443880:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6816:./mpasoc_metis.26600094.log-6258-[1785518522.112727] [l20130:52580:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6817:./mpasoc_metis.26600094.log-6259-[1785518521.530617] [l20119:1443887:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6818---
6819:./mpasoc_metis.26600094.log-6304-*                                                                              *
6820:./mpasoc_metis.26600094.log-6305-*                       We hope you enjoyed the DKRZ supercomputer LEVANTE ... *
6821:./mpasoc_metis.26600094.log-6306-*
6822:./mpasoc_metis.26600094.log:6307:* JobID            : 26600094
6823:./mpasoc_metis.26600094.log-6308-* JobName          : mpasoc_metis                                      
6824:./mpasoc_metis.26600094.log-6309-* Account          : bb1596
6825:./mpasoc_metis.26600094.log-6310-* User             : b381103 (200166), bd1083 (1468)                   
6826---
6827-./mpas_f64.26493638.log-1-outdir=/scratch/b/b381103/legoesm_scaling/mpas_f64_j26493638
6828-./mpas_f64.26493638.log-2-=== np=2 (L8, padded-16, f64) ===
6829-./mpas_f64.26493638.log:3:{"component": "mpas_atm", "subdivision": 8, "n_devices": 2, "n_cells": 655376, "n_edges": 1966080, "nlev": 26, "partition_method": "sfc", "physics": "none", "halo_strategy_requested": "auto", "halo_strategy_effective": "ppermute", "steps": 12, "dt": 30.0, "platform": "gpu", "n_processes": 2, "multicontroller": true, "compile_ms": 7096.0, "steady_median_ms": 38.34, "steady_min_ms": 36.96, "per_step_ms": [7096.0, 38.6, 38.5, 37.0, 38.4, 38.3, 38.4, 38.3, 38.4, 38.5, 38.3, 38.3], "cells": 17039776, "hlo_collective_permutes": null, "hlo_collectives": null, "grid_type": "icosahedral", "resolution": 8, "n_levels": 26, "mode": "strong", "precision": "float64", "physics_level": "none", "backend": "gpu", "dt_seconds": 30.0, "time_per_step_ms": 38.343999999597145, "total_cells": 17039776, "sypd": 2.142069779230805, "mcells_per_s": 444.39223868607934, "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-27T08:05:23.017848+00:00", "grid": "icosahedral", "component": "atmosphere", "resolution": "L8", "n_levels": 26, "precision": "float64", "decomposition": "cell_partition", "solver_variant": "n/a", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 2, "n_gpus": 2, "device_count": 2, "process_count": 2, "devices_per_rank": 1, "cells_per_rank": 8519888, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "nccl", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50109.lvt.dkrz.de", "slurm_job_id": "26493638", "git_sha": "unknown", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"partition_method": "sfc", "physics": "none", "steps": 12, "multicontroller": true, "cells_per_device": 8519888}}}
--
6852---
6853-./mpasoc_out.26504842.log-7179-[1785175224.355565] [l10237:120586:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6854-./mpasoc_out.26504842.log-7180-[1785175224.358010] [l10242:1595934:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6855-./mpasoc_out.26504842.log-7181-[1785175224.373837] [l10237:120587:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6856:./mpasoc_out.26504842.log:7182:{"component": "ocean", "grid": "voronoi", "mode": "strong", "subdivision": 8, "n_ranks": 32, "n_cells": 655362, "n_edges": 1966080, "nlev": 20, "partition_method": "auto", "steps": 12, "dt": 60.0, "platform": "cpu", "compile_ms": 15274.6, "steady_median_ms": 563.7044, "step_latency_gate_loop_ms": 586.85, "step_latency_gate_loop_min_ms": 579.8, "per_step_ms": [15274.6, 593.8, 585.3, 586.3, 591.9, 601.5, 587.4, 581.9, 579.8, 612.5, 585.6, 594.9], "cells": 13107240, "cells_per_rank_achieved": 20480, "compile_prewarmed_by_parity_ref": false, "barotropic_solver": "explicit_substep", "halo_refresh": "in_step", "stage_halo_correct": true, "stage_halo_note": "per-step packed entry refresh + in-step stage-frontier refreshes (make_mpas_ocean_halo_refresh) \u2014 stage-correct; see docs/performance/scaling/mpas_ocean_distributed_stage_audit.md", "fused": {"compile_ms": 582.7, "scan_compile_ms": 18790.8, "step_latency_ms": 591.262, "block_ms": [4503.8, 4505.3], "parallel_block_ms": [4509.28, 4509.99], "fused_step_ms": 563.7044, "rank_imbalance": 1.0011, "rank_imbalance_per_block": [1.0011, 1.0011], "block_steps": 8, "n_blocks": 2, "probe_steps": 3}, "wet_cell": {"wet_cell_levels": 13057600, "wet_cell_levels_per_device": 408050.0, "wet_fraction": 0.9962127801123654, "wet_equals_total": false, "wet_cell_levels_per_device_min": 403240, "wet_cell_levels_per_device_max": 409620}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-27T18:01:25.686454+00:00", "grid": "voronoi", "component": "ocean", "resolution": "L8", "n_levels": 20, "precision": "float64", "decomposition": "cell_partition", "solver_variant": "mpas_ocean_explicit_substep", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 32, "n_gpus": 0, "device_count": 1, "process_count": 1, "devices_per_rank": 1, "cells_per_rank": 409601, "backend": "cpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "mpi4jax", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l10218.lvt.dkrz.de", "slurm_job_id": "26504842", "git_sha": "aa0b2eafb", "gpu_direct_requested": false, "mpi4jax_cuda_support": false, "gpu_direct_active": false, "host_staged_halo": false, "partition_metrics": {"n_owned_cells": 20480, "n_halo_cells": 1259, "owned_halo_ratio": 0.061474609375, "n_neighbor_ranks": 6, "messages_per_exchange": 6, "halo_recv_cells": 1259, "owned_send_cells": 1258, "cells_per_rank_min": 20480, "cells_per_rank_max": 20481, "edge_cut_total": 40803, "max_neighbor_ranks": 8}, "extra": {"partition_method": "auto", "steps": 12, "warmup": 2, "parity_gate": false, "check_conservation": false, "halo_refresh": "in_step", "barotropic_solver": "explicit_substep", "block_steps": 8, "blocks": 2, "probe_steps": 3}}}
6857-./mpasoc_out.26504842.log:7183:[mpas-ocean np=32 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=15274.6ms fused=563.7044ms/step (probe_latency=591.262ms) gate_loop_latency=586.85ms/step
6858-./mpasoc_out.26504842.log-7184-[1785175224.431721] [l10218:70140:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6859-./mpasoc_out.26504842.log-7185---- mpas-ocean s8 np=64 (f64, cyclic) ---
6860-./mpasoc_out.26504842.log-7186-[l10242.lvt.dkrz.de:1596254] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
6861---
6862-./mpasoc_out.26504842.log-8030-[1785175288.207821] [l10242:1596265:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6863-./mpasoc_out.26504842.log-8031-[1785175288.132399] [l10218:70481:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6864-./mpasoc_out.26504842.log-8032-[1785175288.207836] [l10242:1596263:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6865:./mpasoc_out.26504842.log:8033:{"component": "ocean", "grid": "voronoi", "mode": "strong", "subdivision": 8, "n_ranks": 64, "n_cells": 655362, "n_edges": 1966080, "nlev": 20, "partition_method": "auto", "steps": 12, "dt": 60.0, "platform": "cpu", "compile_ms": 13659.2, "steady_median_ms": 404.4325, "step_latency_gate_loop_ms": 383.33, "step_latency_gate_loop_min_ms": 382.56, "per_step_ms": [13659.2, 384.8, 384.1, 382.9, 383.8, 383.2, 384.2, 383.2, 383.4, 384.0, 382.6, 383.0], "cells": 13107240, "cells_per_rank_achieved": 10240, "compile_prewarmed_by_parity_ref": false, "barotropic_solver": "explicit_substep", "halo_refresh": "in_step", "stage_halo_correct": true, "stage_halo_note": "per-step packed entry refresh + in-step stage-frontier refreshes (make_mpas_ocean_halo_refresh) \u2014 stage-correct; see docs/performance/scaling/mpas_ocean_distributed_stage_audit.md", "fused": {"compile_ms": 378.2, "scan_compile_ms": 16175.9, "step_latency_ms": 382.988, "block_ms": [3229.96, 3237.65], "parallel_block_ms": [3232.32, 3238.6], "fused_step_ms": 404.4325, "rank_imbalance": 1.0009, "rank_imbalance_per_block": [1.0009, 1.0009], "block_steps": 8, "n_blocks": 2, "probe_steps": 3}, "wet_cell": {"wet_cell_levels": 13057600, "wet_cell_levels_per_device": 204025.0, "wet_fraction": 0.9962127801123654, "wet_equals_total": false, "wet_cell_levels_per_device_min": 198440, "wet_cell_levels_per_device_max": 204820}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-27T18:02:18.975548+00:00", "grid": "voronoi", "component": "ocean", "resolution": "L8", "n_levels": 20, "precision": "float64", "decomposition": "cell_partition", "solver_variant": "mpas_ocean_explicit_substep", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 64, "n_gpus": 0, "device_count": 1, "process_count": 1, "devices_per_rank": 1, "cells_per_rank": 204800, "backend": "cpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "mpi4jax", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l10218.lvt.dkrz.de", "slurm_job_id": "26504842", "git_sha": "aa0b2eafb", "gpu_direct_requested": false, "mpi4jax_cuda_support": false, "gpu_direct_active": false, "host_staged_halo": false, "partition_metrics": {"n_owned_cells": 10240, "n_halo_cells": 904, "owned_halo_ratio": 0.08828125, "n_neighbor_ranks": 7, "messages_per_exchange": 7, "halo_recv_cells": 904, "owned_send_cells": 906, "cells_per_rank_min": 10240, "cells_per_rank_max": 10241, "edge_cut_total": 56765, "max_neighbor_ranks": 8}, "extra": {"partition_method": "auto", "steps": 12, "warmup": 2, "parity_gate": false, "check_conservation": false, "halo_refresh": "in_step", "barotropic_solver": "explicit_substep", "block_steps": 8, "blocks": 2, "probe_steps": 3}}}
6866-./mpasoc_out.26504842.log:8034:[mpas-ocean np=64 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=13659.2ms fused=404.4325ms/step (probe_latency=382.988ms) gate_loop_latency=383.33ms/step
6867-./mpasoc_out.26504842.log-8035-[1785175288.142310] [l10218:70472:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6868-./mpasoc_out.26504842.log-8036-[1785175288.126163] [l10244:790895:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6869-./mpasoc_out.26504842.log-8037-[1785175288.126476] [l10244:790903:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6870---
6871-./mpasoc_out.26504842.log-9789-[1785175341.750400] [l10218:71113:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6872-./mpasoc_out.26504842.log-9790-[1785175341.746254] [l10244:791504:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6873-./mpasoc_out.26504842.log-9791-[1785175341.766658] [l10218:71112:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6874:./mpasoc_out.26504842.log:9792:{"component": "ocean", "grid": "voronoi", "mode": "strong", "subdivision": 8, "n_ranks": 128, "n_cells": 655362, "n_edges": 1966080, "nlev": 20, "partition_method": "auto", "steps": 12, "dt": 60.0, "platform": "cpu", "compile_ms": 12670.2, "steady_median_ms": 309.9656, "step_latency_gate_loop_ms": 312.57, "step_latency_gate_loop_min_ms": 308.79, "per_step_ms": [12670.2, 314.0, 313.6, 314.3, 312.6, 319.6, 314.8, 309.8, 311.9, 308.8, 312.6, 311.1], "cells": 13107240, "cells_per_rank_achieved": 5120, "compile_prewarmed_by_parity_ref": false, "barotropic_solver": "explicit_substep", "halo_refresh": "in_step", "stage_halo_correct": true, "stage_halo_note": "per-step packed entry refresh + in-step stage-frontier refreshes (make_mpas_ocean_halo_refresh) \u2014 stage-correct; see docs/performance/scaling/mpas_ocean_distributed_stage_audit.md", "fused": {"compile_ms": 309.2, "scan_compile_ms": 14923.8, "step_latency_ms": 308.874, "block_ms": [2476.18, 2479.08], "parallel_block_ms": [2477.45, 2482.0], "fused_step_ms": 309.9656, "rank_imbalance": 1.0008, "rank_imbalance_per_block": [1.0008, 1.0008], "block_steps": 8, "n_blocks": 2, "probe_steps": 3}, "wet_cell": {"wet_cell_levels": 13057600, "wet_cell_levels_per_device": 102012.5, "wet_fraction": 0.9962127801123654, "wet_equals_total": false, "wet_cell_levels_per_device_min": 96040, "wet_cell_levels_per_device_max": 102420}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-27T18:03:09.005391+00:00", "grid": "voronoi", "component": "ocean", "resolution": "L8", "n_levels": 20, "precision": "float64", "decomposition": "cell_partition", "solver_variant": "mpas_ocean_explicit_substep", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 128, "n_gpus": 0, "device_count": 1, "process_count": 1, "devices_per_rank": 1, "cells_per_rank": 102400, "backend": "cpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "mpi4jax", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l10218.lvt.dkrz.de", "slurm_job_id": "26504842", "git_sha": "aa0b2eafb", "gpu_direct_requested": false, "mpi4jax_cuda_support": false, "gpu_direct_active": false, "host_staged_halo": false, "partition_metrics": {"n_owned_cells": 5120, "n_halo_cells": 658, "owned_halo_ratio": 0.128515625, "n_neighbor_ranks": 7, "messages_per_exchange": 7, "halo_recv_cells": 658, "owned_send_cells": 661, "cells_per_rank_min": 5120, "cells_per_rank_max": 5121, "edge_cut_total": 80510, "max_neighbor_ranks": 8}, "extra": {"partition_method": "auto", "steps": 12, "warmup": 2, "parity_gate": false, "check_conservation": false, "halo_refresh": "in_step", "barotropic_solver": "explicit_substep", "block_steps": 8, "blocks": 2, "probe_steps": 3}}}
6875-./mpasoc_out.26504842.log:9793:[mpas-ocean np=128 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=12670.2ms fused=309.9656ms/step (probe_latency=308.874ms) gate_loop_latency=312.57ms/step
6876-./mpasoc_out.26504842.log-9794-[1785175341.786438] [l10218:71091:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6877-./mpasoc_out.26504842.log-9795-[1785175341.750476] [l10218:71097:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6878-./mpasoc_out.26504842.log-9796-[1785175341.728337] [l10244:791516:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6879---
6880-./mpasoc_out.26504842.log-13256-[1785175393.007207] [l10237:122730:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6881-./mpasoc_out.26504842.log-13257-[1785175393.092947] [l10218:72312:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6882-./mpasoc_out.26504842.log-13258-[1785175393.142807] [l10218:72343:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6883:./mpasoc_out.26504842.log:13259:{"component": "ocean", "grid": "voronoi", "mode": "strong", "subdivision": 8, "n_ranks": 256, "n_cells": 655362, "n_edges": 1966080, "nlev": 20, "partition_method": "auto", "steps": 12, "dt": 60.0, "platform": "cpu", "compile_ms": 12914.2, "steady_median_ms": 298.2331, "step_latency_gate_loop_ms": 277.99, "step_latency_gate_loop_min_ms": 275.05, "per_step_ms": [12914.2, 280.0, 279.4, 276.7, 278.1, 287.7, 283.1, 278.4, 276.2, 275.5, 275.1, 277.9], "cells": 13107240, "cells_per_rank_achieved": 2560, "compile_prewarmed_by_parity_ref": false, "barotropic_solver": "explicit_substep", "halo_refresh": "in_step", "stage_halo_correct": true, "stage_halo_note": "per-step packed entry refresh + in-step stage-frontier refreshes (make_mpas_ocean_halo_refresh) \u2014 stage-correct; see docs/performance/scaling/mpas_ocean_distributed_stage_audit.md", "fused": {"compile_ms": 272.7, "scan_compile_ms": 14871.2, "step_latency_ms": 269.868, "block_ms": [2372.86, 2397.08], "parallel_block_ms": [2374.3, 2397.43], "fused_step_ms": 298.2331, "rank_imbalance": 1.0003, "rank_imbalance_per_block": [1.0004, 1.0001], "block_steps": 8, "n_blocks": 2, "probe_steps": 3}, "wet_cell": {"wet_cell_levels": 13057600, "wet_cell_levels_per_device": 51006.25, "wet_fraction": 0.9962127801123654, "wet_equals_total": false, "wet_cell_levels_per_device_min": 44840, "wet_cell_levels_per_device_max": 51220}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-27T18:04:02.394298+00:00", "grid": "voronoi", "component": "ocean", "resolution": "L8", "n_levels": 20, "precision": "float64", "decomposition": "cell_partition", "solver_variant": "mpas_ocean_explicit_substep", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 256, "n_gpus": 0, "device_count": 1, "process_count": 1, "devices_per_rank": 1, "cells_per_rank": 51200, "backend": "cpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "mpi4jax", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l10218.lvt.dkrz.de", "slurm_job_id": "26504842", "git_sha": "aa0b2eafb", "gpu_direct_requested": false, "mpi4jax_cuda_support": false, "gpu_direct_active": false, "host_staged_halo": false, "partition_metrics": {"n_owned_cells": 2560, "n_halo_cells": 478, "owned_halo_ratio": 0.18671875, "n_neighbor_ranks": 6, "messages_per_exchange": 6, "halo_recv_cells": 478, "owned_send_cells": 477, "cells_per_rank_min": 2560, "cells_per_rank_max": 2561, "edge_cut_total": 114563, "max_neighbor_ranks": 8}, "extra": {"partition_method": "auto", "steps": 12, "warmup": 2, "parity_gate": false, "check_conservation": false, "halo_refresh": "in_step", "barotropic_solver": "explicit_substep", "block_steps": 8, "blocks": 2, "probe_steps": 3}}}
6884-./mpasoc_out.26504842.log:13260:[mpas-ocean np=256 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=12914.2ms fused=298.2331ms/step (probe_latency=269.868ms) gate_loop_latency=277.99ms/step
6885-./mpasoc_out.26504842.log-13261-[1785175393.116389] [l10218:72289:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6886-./mpasoc_out.26504842.log-13262-[1785175392.630692] [l10218:72314:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6887-./mpasoc_out.26504842.log-13263-[1785175393.093092] [l10218:72352:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
--
7014---
7015-./mpasoc_2x2.26508258.log-3987-[1785187558.010152] [l30556:517578:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
7016-./mpasoc_2x2.26508258.log-3988-[1785187558.010284] [l30556:517570:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
7017-./mpasoc_2x2.26508258.log-3989-[1785187557.974631] [l30556:517573:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
7018:./mpasoc_2x2.26508258.log:3990:{"component": "ocean", "grid": "voronoi", "mode": "strong", "subdivision": 8, "n_ranks": 128, "n_cells": 655362, "n_edges": 1966080, "nlev": 20, "partition_method": "sfc", "steps": 12, "dt": 60.0, "platform": "cpu", "compile_ms": 13799.9, "steady_median_ms": 610.4875, "step_latency_gate_loop_ms": 584.34, "step_latency_gate_loop_min_ms": 582.19, "per_step_ms": [13799.9, 592.1, 588.0, 587.1, 587.2, 587.9, 582.2, 583.7, 584.4, 583.7, 584.3, 583.6], "cells": 13107240, "cells_per_rank_achieved": 5120, "compile_prewarmed_by_parity_ref": false, "barotropic_solver": "explicit_substep", "halo_refresh": "in_step", "stage_halo_correct": true, "stage_halo_note": "per-step packed entry refresh + in-step stage-frontier refreshes (make_mpas_ocean_halo_refresh) \u2014 stage-correct; see docs/performance/scaling/mpas_ocean_distributed_stage_audit.md", "fused": {"compile_ms": 574.2, "scan_compile_ms": 17946.3, "step_latency_ms": 580.0, "block_ms": [4880.17, 4882.77], "parallel_block_ms": [4882.64, 4885.16], "fused_step_ms": 610.4875, "rank_imbalance": 1.0004, "rank_imbalance_per_block": [1.0004, 1.0004], "block_steps": 8, "n_blocks": 2, "probe_steps": 3}, "wet_cell": {"wet_cell_levels": 13057600, "wet_cell_levels_per_device": 102012.5, "wet_fraction": 0.9962127801123654, "wet_equals_total": false, "wet_cell_levels_per_device_min": 96060, "wet_cell_levels_per_device_max": 102420}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-27T21:26:57.758599+00:00", "grid": "voronoi", "component": "ocean", "resolution": "L8", "n_levels": 20, "precision": "float64", "decomposition": "cell_partition", "solver_variant": "mpas_ocean_explicit_substep", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 128, "n_gpus": 0, "device_count": 1, "process_count": 1, "devices_per_rank": 1, "cells_per_rank": 102400, "backend": "cpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "mpi4jax", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l30556.lvt.dkrz.de", "slurm_job_id": "26508258", "git_sha": "346a77f20", "gpu_direct_requested": false, "mpi4jax_cuda_support": false, "gpu_direct_active": false, "host_staged_halo": false, "partition_metrics": {"n_owned_cells": 5121, "n_halo_cells": 850, "owned_halo_ratio": 0.16598320640499903, "n_neighbor_ranks": 7, "messages_per_exchange": 7, "halo_recv_cells": 850, "owned_send_cells": 839, "cells_per_rank_min": 5120, "cells_per_rank_max": 5121, "edge_cut_total": 105825, "max_neighbor_ranks": 9}, "extra": {"partition_method": "sfc", "steps": 12, "warmup": 2, "parity_gate": false, "check_conservation": false, "halo_refresh": "in_step", "barotropic_solver": "explicit_substep", "block_steps": 8, "blocks": 2, "probe_steps": 3}}}
7019-./mpasoc_2x2.26508258.log:3991:[mpas-ocean np=128 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=13799.9ms fused=610.4875ms/step (probe_latency=580.0ms) gate_loop_latency=584.34ms/step
7020-./mpasoc_2x2.26508258.log-3992-[1785187557.973953] [l30556:517564:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
7021-./mpasoc_2x2.26508258.log-3993-[1785187558.067581] [l30574:586591:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
7022-./mpasoc_2x2.26508258.log-3994-[1785187558.216350] [l30586:490353:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
--
7212---
7213-./docs/performance/scaling/levante_campaign_2026-07-24.md-1589-Both jobs the dropped session left behind COMPLETED; neither had been
7214-./docs/performance/scaling/levante_campaign_2026-07-24.md-1590-analysed. Numbers below are their first read-out.
7215-./docs/performance/scaling/levante_campaign_2026-07-24.md-1591-
7216:./docs/performance/scaling/levante_campaign_2026-07-24.md:1592:### 1. METIS / placement A-B (ocean MPAS CPU, job 26600094) — both halves of codex lever #2 are DEAD
7217-./docs/performance/scaling/levante_campaign_2026-07-24.md-1593-
7218-./docs/performance/scaling/levante_campaign_2026-07-24.md-1594-Matrix at IDENTICAL 5,120 cells/rank (anti-masquerade fields verified:
7219:./docs/performance/scaling/levante_campaign_2026-07-24.md-1595-`cells_per_rank_achieved=5120`, `n_ranks` 32/128, method pinned per arm —
7220---
7221-./docs/performance/scaling/levante_campaign_2026-07-24.md-1620-  lane. The 1.63x rank-count term at matched tile stands UNATTRIBUTED
7222-./docs/performance/scaling/levante_campaign_2026-07-24.md-1621-  mechanistically (not partition, not placement).
7223-./docs/performance/scaling/levante_campaign_2026-07-24.md-1622-
--
7391---
7392-docs/performance/scaling/SCALING_STATUS_AUDIT.md-32-
7393-docs/performance/scaling/SCALING_STATUS_AUDIT.md-33-| grid | CPU-MPI | GPU / SPMD | true weak/strong evidence (a) | known blockers | next measurement |
7394-docs/performance/scaling/SCALING_STATUS_AUDIT.md-34-|------|---------|------------|-------------------------------|----------------|------------------|
7395:docs/performance/scaling/SCALING_STATUS_AUDIT.md-35-| **lat-lon C-grid** | (a) latitude-band MPI, `bench_ocean_mpi_scaling.py` (parity + conservation gates; `--wet-balance`; distributed fixed-M PCG / single_reduce / preconditioner options) | (a) full-step SPMD `bench_ocean_latlon_spmd_scaling.py` (single-process multi-device AND `--multicontroller` NCCL; parity + conservation gates); jit-once sharded step | 2-GPU full step: strong 0.92 eff at production size, weak 0.97 @ ~590k cells/rank (Ginsburg). CPU-MPI strong np16→32 eff 0.55 (implicit-CN reduction wall; split-explicit + local clamp opt-in 1.15–1.65× at ≥2 nodes) | implicit-CN allreduce wall at high ranks; land-cell load imbalance | Derecho/Levante A100 ladders (jobs exist, unrun); wide-halo split-explicit A/B at ≥16 ranks; wet-cell-balanced partitions at scale |
7396-docs/performance/scaling/SCALING_STATUS_AUDIT.md:36:| **tripole / eORCA** | (c) fold-aware band-MPI validated at operator level AND full-model (#883: full-model tripole MPI validation + tripole bench lanes on `bench_ocean_mpi_scaling.py`) | (c) tripole fold wired into the SPMD step (#883: `make_sharded_ocean_step` fold support; `bench_ocean_latlon_spmd_scaling.py --tripole`); single-GPU throughput (b: 313 Mc/s fp32, `SCALING_SUMMARY.md`); no multi-device scale receipts | none | scale receipts only | tripole rank/device ladders on Derecho/Levante (lanes exist) |
7397:docs/performance/scaling/SCALING_STATUS_AUDIT.md:37:| **MPAS / Voronoi** | (c) bench lane + STAGE-CORRECT distributed step (PR #1162: `MPASOceanModel.step(halo_refresh=...)` re-arms the halo at every audited dependency frontier — T1/T3 tendencies incl. the vertex channel, R1–R3 step, B0–B2 explicit substeps, I0–I1 implicit predictor; `bench_ocean_mpas_scaling.py --halo-refresh in_step`, auto at n_ranks>1, earns `stage_halo_correct=true`; np2 owned-cell parity < 5e-11 with a non-vacuity tripwire). Serial `halo_refresh=None` byte-identical. Remaining deferred: freshwater owned-mask threading, wide-halo explicit substeps (`mpas_ocean_distributed_stage_audit.md`) | (b) single-GPU throughput only (333 Mc/s fp32 impl_cn) | none | remaining stage-audit deferred items | np≥2 CPU-MPI scaling receipts on the stage-correct lane |
7398-docs/performance/scaling/SCALING_STATUS_AUDIT.md-38-| **cubed-sphere** | (c) supported in `run_omip.py`; generic distributed layout, MPI conservation tested; no dedicated scaling lane | (c) same | none | no lane | only if a science driver demands it |
7399-docs/performance/scaling/SCALING_STATUS_AUDIT.md-39-
7400-docs/performance/scaling/SCALING_STATUS_AUDIT.md-40-## CRM / LES (plane dycore) — see `crm_les_scaling.md`
7401-docs/performance/scaling/SCALING_STATUS_AUDIT.md-41-
--
7855-docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-87-     full `hlo_collectives` dict alongside the legacy scalar.
7856-docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-88-
7857-docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md-89-## 4. What to run on Derecho / Levante now
7858---
7859:docs/performance/scaling/mpas_ocean_distributed_stage_audit.md-199-* thread `owned_mask` into the freshwater normalization means to retire the
7860-docs/performance/scaling/mpas_ocean_distributed_stage_audit.md-200-  multi-rank refusal (`ocean_pe_mpas.py:947`; helper support already exists —
7861-docs/performance/scaling/mpas_ocean_distributed_stage_audit.md-201-  `freshwater.py:181-261`).
7862-docs/performance/scaling/mpas_ocean_distributed_stage_audit.md-202-
7863-docs/performance/scaling/mpas_ocean_distributed_stage_audit.md:203:Until then, multi-rank MPAS-ocean full-step rows are throughput/parity-bounded
--
8554-docs/performance/scaling/levante_campaign_2026-07-24.md-1343-  **s7 2->16 = 0.65** — the tile-size pattern reproduces on a FOURTH
8555-docs/performance/scaling/levante_campaign_2026-07-24.md-1344-  lane (bigger mesh holds efficiency deeper), and the np8->16 flattening
8556-docs/performance/scaling/levante_campaign_2026-07-24.md-1345-  sits exactly where per-rank cells fall to 2.5k (s6) vs 10k (s7).
8557---
8558:docs/performance/scaling/levante_campaign_2026-07-24.md-1548-   refuted; at ~0.71 wet the measured stencil member is a net LOSS. VERDICT: do not build compaction for the global latlon ocean;
8559:docs/performance/scaling/levante_campaign_2026-07-24.md-1549-   revisit only for a configuration that is genuinely <~55% wet.
8560-docs/performance/scaling/levante_campaign_2026-07-24.md-1550-5. **2-D lat-lon decomposition** at >=64 ranks: the 1-D band's perimeter
8561-docs/performance/scaling/levante_campaign_2026-07-24.md-1551-   ceiling is now measured (0.12-0.16 at np64 spread, vs ico's 0.52), which
8562-docs/performance/scaling/levante_campaign_2026-07-24.md-1552-   quantifies the prize.
8563-docs/performance/scaling/levante_campaign_2026-07-24.md-1553-6. ~~Milan np16 anomaly~~ **RESOLVED 2026-07-26 (job 26479904): rank
--
8582-docs/performance/scaling/levante_campaign_2026-07-24.md-1572-   with the decomposition. Report as a regular-vs-wall-pole path
8583---
8584-docs/performance/scaling/levante_campaign_2026-07-24.md-1593-
8585-docs/performance/scaling/levante_campaign_2026-07-24.md-1594-Matrix at IDENTICAL 5,120 cells/rank (anti-masquerade fields verified:
8586:docs/performance/scaling/levante_campaign_2026-07-24.md-1595-`cells_per_rank_achieved=5120`, `n_ranks` 32/128, method pinned per arm —
8587-docs/performance/scaling/levante_campaign_2026-07-24.md-1596-never `auto`, which flipped meaning when pymetis appeared in `.venv-mpi`
8588-docs/performance/scaling/levante_campaign_2026-07-24.md-1597-on 2026-07-31), f64, nlev 20, 32 ranks/node, `--distribution` as listed:
8589-docs/performance/scaling/levante_campaign_2026-07-24.md-1598-
8590-docs/performance/scaling/levante_campaign_2026-07-24.md-1599-| arm | config | ms/step |
8591-docs/performance/scaling/levante_campaign_2026-07-24.md-1600-|---|---|---|
8592:docs/performance/scaling/levante_campaign_2026-07-24.md:1601:| A | s7 np32 geometric, block:cyclic | 189.82 |
8593:docs/performance/scaling/levante_campaign_2026-07-24.md:1602:| B | s8 np128 geometric, block:cyclic | 308.96 |
8594:docs/performance/scaling/levante_campaign_2026-07-24.md:1603:| C | s7 np32 metis, block:cyclic | 191.99 |
8595:docs/performance/scaling/levante_campaign_2026-07-24.md:1604:| D | s8 np128 metis, block:cyclic | 333.39 |
8596-docs/performance/scaling/levante_campaign_2026-07-24.md:1605:| E | s8 np128 metis, block:block | 537.91 |
8597-docs/performance/scaling/levante_campaign_2026-07-24.md-1606-
8598-docs/performance/scaling/levante_campaign_2026-07-24.md-1607-* **Partition quality does NOT buy back the rank-count term — METIS is
8599-docs/performance/scaling/levante_campaign_2026-07-24.md-1608-  WORSE at scale.** Rank-count term geometric B/A = 1.628; metis
8600-docs/performance/scaling/levante_campaign_2026-07-24.md-1609-  D/C = 1.736 (D is +7.9 % over B), despite METIS holding the better
8601:docs/performance/scaling/levante_campaign_2026-07-24.md-1610-  offline cut on this exact mesh (partq s8@np128: edge_cut 1.89 % vs
8602:docs/performance/scaling/levante_campaign_2026-07-24.md-1611-  2.01 %, halo mean 592 vs 629). The offline-quality -> step-time
8603-docs/performance/scaling/levante_campaign_2026-07-24.md-1612-  inference is REFUTED on this lane; the full-step measurement fails
8604-docs/performance/scaling/levante_campaign_2026-07-24.md-1613-  codex's own ">=10 % full-step win" bar in the wrong direction.
8605-docs/performance/scaling/levante_campaign_2026-07-24.md:1614:* **Placement: block:cyclic stays mandatory** (E/D = 1.61x at a
8606-docs/performance/scaling/levante_campaign_2026-07-24.md-1615-  byte-identical partition) — same direction as the np16 Milan receipt
--
8652-   666	    # in-loop residual is not exposed; single-rank runs the stock adaptive
8653-   667	    # CG (count not exposed; tol/maxiter recorded).  The probe below is ONE
8654-   668	    # standalone free-surface solve at the FINAL state with zero slow
8655-   669	    # forcing, OUTSIDE any timed loop, via return_residual=True — the
8656:   670	    # owned-masked global relative Helmholtz residual of the returned eta.
8657-   671	    # Solver-HEALTH evidence, not the benchmarked solve's residual
8658-   672	    # (bench_ocean_latlon_spmd_scaling convention).  Collective at
8659-   673	    # n_ranks > 1: every rank calls it.
8660-   674	    zero_forcing_probe_residual = None
--
8733-   747	        cells=int(mesh.nCells) * args.nlev,
8734-   748	        # HORIZONTAL cells/rank — the same unit as --cells-per-rank, so a
8735-   749	        # weak-mode row is comparable to its target (codex: the 3-D count
8736-   750	        # made rows look nlev-x larger).
8737:   751	        cells_per_rank_achieved=int(mesh.nCells) // n_ranks,
8738-   752	        # Fix 4 honesty flag: at np=1 the parity reference pre-runs the
8739-   753	        # SAME shape before the timed loop, so per_step_ms[0] may not
8740-   754	        # contain the real JIT compile.
8741-   755	        compile_prewarmed_by_parity_ref=bool(
--
8750-   764	        # Single-rank rows have no partition, hence trivially true.
8751-   765	        stage_halo_correct=bool(n_ranks == 1 or halo_refresh == "in_step"),
8752-   766	        stage_halo_note=stage_halo_note_for(n_ranks, halo_refresh),
8753-   767	        fused=fused,
8754:   768	        wet_cell=wet_rec,
8755-   769	        solver_iters=solver_iters,
8756-   770	        solver_iters_mode=solver_iters_mode,
8757-   771	        zero_forcing_probe_residual=zero_forcing_probe_residual,
8758-   772	        zero_forcing_probe_measured=zero_forcing_probe_measured,
--
9163-packages/core/legoesm/parallel/voronoi_mpi.py-193-    differentiable).  Degrades to the per-rank values on a single rank.  Call
9164-packages/core/legoesm/parallel/voronoi_mpi.py-194-    OUTSIDE the timed loop (each reduction is a collective).
9165-packages/core/legoesm/parallel/voronoi_mpi.py-195-
9166-packages/core/legoesm/parallel/voronoi_mpi.py:196:    Adds: ``cells_per_rank_min`` / ``cells_per_rank_max`` (load balance),
9167:packages/core/legoesm/parallel/voronoi_mpi.py-197-    ``edge_cut_total`` (sum of ghost cells over ranks), ``max_neighbor_ranks``.
9168-packages/core/legoesm/parallel/voronoi_mpi.py-198-    """
9169:packages/core/legoesm/parallel/voronoi_mpi.py-199-    n_owned = jnp.asarray(float(local["n_owned_cells"]))
9170-packages/core/legoesm/parallel/voronoi_mpi.py-200-    halo_recv = jnp.asarray(float(local["halo_recv_cells"]))
9171-packages/core/legoesm/parallel/voronoi_mpi.py-201-    neighbors = jnp.asarray(float(local["n_neighbor_ranks"]))
9172-packages/core/legoesm/parallel/voronoi_mpi.py-202-    out = dict(local)
9173:packages/core/legoesm/parallel/voronoi_mpi.py:203:    out["cells_per_rank_min"] = int(global_min_mpi(n_owned))
9174:packages/core/legoesm/parallel/voronoi_mpi.py:204:    out["cells_per_rank_max"] = int(global_max_mpi(n_owned))
9175:packages/core/legoesm/parallel/voronoi_mpi.py-205-    out["edge_cut_total"] = int(global_sum_mpi(halo_recv))
9176-packages/core/legoesm/parallel/voronoi_mpi.py-206-    out["max_neighbor_ranks"] = int(global_max_mpi(neighbors))
9177-packages/core/legoesm/parallel/voronoi_mpi.py-207-    return out
9178-packages/core/legoesm/parallel/voronoi_mpi.py-208-
9179-packages/core/legoesm/parallel/voronoi_mpi.py-209-
--
9184-packages/core/legoesm/parallel/voronoi_mpi.py-1151-                # measured -> N_i overflow NaN; per-volume N_c/N_r keep the
9185-packages/core/legoesm/parallel/voronoi_mpi.py-1152-                # plain clip pending density-aware repair).  GLOBAL residual
9186-packages/core/legoesm/parallel/voronoi_mpi.py:1153:                # redistribution over OWNED cells via allreduce-SUM (the one
9187-packages/core/legoesm/parallel/voronoi_mpi.py-1154-                # AD-safe collective) so the factor is decomposition-
9188:packages/core/legoesm/parallel/voronoi_mpi.py-1155-                # independent; owned-mask weighting keeps halo cells out of
9189-packages/core/legoesm/parallel/voronoi_mpi.py-1156-                # the budget exactly like the mass fixer.
9190-packages/core/legoesm/parallel/voronoi_mpi.py-1157-                # TRUE layer-mass dp weight (post-mass-fix p_s): identical
9191-packages/core/legoesm/parallel/voronoi_mpi.py-1158-                # rescale on pure sigma (per-column p_s cancels), correct on
9192---
--
9520-packages/ocean/legoesm/ocean/forcing/isf_spe.py-4-shipped with the NEMO ORCA1 reference configuration
9521-packages/ocean/legoesm/ocean/forcing/isf_spe.py-5-(``runoff-icb_DaiTrenberth_Depoorter.nc``: eORCA1 curvilinear, monthly
9522-packages/ocean/legoesm/ocean/forcing/isf_spe.py-6-``sornfisf`` [kg/m²/s, >= 0 into the ocean] plus the injection depth
9523-packages/ocean/legoesm/ocean/forcing/isf_spe.py-7-band ``sodepmin_isf``/``sodepmax_isf`` [m]) and regrids it to the model
9524:packages/ocean/legoesm/ocean/forcing/isf_spe.py-8-tracer grid — nearest-wet-neighbour with the monthly melt totals
9525---
9526-packages/ocean/legoesm/ocean/init_woa.py-141-
9527-packages/ocean/legoesm/ocean/init_woa.py-142-
9528-packages/ocean/legoesm/ocean/init_woa.py-143-def _open_woa_dataset(path: str | Path):
--
9595-packages/ocean/legoesm/ocean/physics/convection/plume.py-219-    # 0``.  The plume's ``active`` mask is also zero there, so
9596-packages/ocean/legoesm/ocean/physics/convection/plume.py-220-    # ``column_dT`` and ``column_dS`` are zero and the correction
9597---
9598-packages/ocean/legoesm/ocean/fidelity/nemo_recipe.py-855-
9599:packages/ocean/legoesm/ocean/fidelity/nemo_recipe.py-856-    # Consistent 30-wet-level z* (H_max == sum(e3t) == H_bathy == 4300.71 m).
9600:packages/ocean/legoesm/ocean/fidelity/nemo_recipe.py-857-    e3t_wet, gdept_wet = _nemo_gyre_vertical_ladder()
9601-packages/ocean/legoesm/ocean/fidelity/nemo_recipe.py-858-    # NEMO GYRE_BARE is built with key_linssh: LINEAR free surface, layer
9602-packages/ocean/legoesm/ocean/fidelity/nemo_recipe.py-859-    # thicknesses FIXED at the eta=0 reference. Under z-star the sigma
9603-packages/ocean/legoesm/ocean/fidelity/nemo_recipe.py:860:    # redistribution of deta/dt manufactured a spurious bottom-intensified
9604-packages/ocean/legoesm/ocean/fidelity/nemo_recipe.py-861-    # abyssal circulation (surf/deep rms(u) 0.20 vs NEMO 8.2); linssh flips it
9605-packages/ocean/legoesm/ocean/fidelity/nemo_recipe.py-862-    # to NEMO's surface-intensified structure (5.91) and collapses the abyssal
9606-packages/ocean/legoesm/ocean/fidelity/nemo_recipe.py-863-    # density drift to NEMO's level. See nemo_gyre_fidelity_plan.md item A.
9607:packages/ocean/legoesm/ocean/fidelity/nemo_recipe.py-864-    z_coord = create_z_star_from_thicknesses(e3t_wet, gdept_wet)._replace(
9608-packages/ocean/legoesm/ocean/fidelity/nemo_recipe.py-865-        linear_free_surface=True)
9609---
9610-packages/ocean/legoesm/ocean/fidelity/veros_global_1deg_recipe.py-82-     ``FluxFeedbackConfig.penetrative_shortwave=True`` with
9611-packages/ocean/legoesm/ocean/fidelity/veros_global_1deg_recipe.py-83-     ``shortwave_water_type="I"`` (≡ the setup literals R=0.58, ζ1=0.35,
--
9631-packages/ocean/legoesm/ocean/fidelity/veros_global_4deg_recipe.py-203-    c_eps=0.5,                    # Veros eke_c_eps
9632-packages/ocean/legoesm/ocean/fidelity/veros_global_4deg_recipe.py-204-    l_min=100.0,                  # Veros eke_lmin
9633---
9634-packages/ocean/legoesm/ocean/freshwater.py-380-    depth contributes only its above-depth fraction), so
9635:packages/ocean/legoesm/ocean/freshwater.py-381-    ``h_rnf = min(runoff_spread_m, wet column depth)`` exactly on any vertical
9636-packages/ocean/legoesm/ocean/freshwater.py-382-    grid (NEMO ``h_rnf = min(rn_dep_max, depth)``).  The COLUMN-INTEGRAL salt
9637-packages/ocean/legoesm/ocean/freshwater.py-383-    tendency is exactly ``-S_ref*(F_top + R)/rho_0`` — bit-identical
9638-packages/ocean/legoesm/ocean/freshwater.py-384-    conservation to the legacy top-cell closure, only the vertical
9639-packages/ocean/legoesm/ocean/freshwater.py:385:    distribution changes.
--
9715-   618	        rtol, atol = MPAS_OCEAN_PARITY_TOLS[args.precision]
9716-   619	        ok = True
9717-   620	        for nm in PARITY_FIELDS:
9718-   621	            if n_ranks > 1:
9719:   622	                got = gather_owned_cells(
9720-   623	                    getattr(state, nm).data, part, comm, int(mesh.nCells))
9721-   624	            else:
9722-   625	                got = np.asarray(getattr(state, nm).data)
9723-   626	            if is_rank0:
--
9763-   666	    # in-loop residual is not exposed; single-rank runs the stock adaptive
9764-   667	    # CG (count not exposed; tol/maxiter recorded).  The probe below is ONE
9765-   668	    # standalone free-surface solve at the FINAL state with zero slow
9766-   669	    # forcing, OUTSIDE any timed loop, via return_residual=True — the
9767:   670	    # owned-masked global relative Helmholtz residual of the returned eta.
9768-   671	    # Solver-HEALTH evidence, not the benchmarked solve's residual
9769-   672	    # (bench_ocean_latlon_spmd_scaling convention).  Collective at
9770-   673	    # n_ranks > 1: every rank calls it.
9771-   674	    zero_forcing_probe_residual = None
--
9844-   747	        cells=int(mesh.nCells) * args.nlev,
9845-   748	        # HORIZONTAL cells/rank — the same unit as --cells-per-rank, so a
9846-   749	        # weak-mode row is comparable to its target (codex: the 3-D count
9847-   750	        # made rows look nlev-x larger).
9848:   751	        cells_per_rank_achieved=int(mesh.nCells) // n_ranks,
9849-   752	        # Fix 4 honesty flag: at np=1 the parity reference pre-runs the
9850-   753	        # SAME shape before the timed loop, so per_step_ms[0] may not
9851-   754	        # contain the real JIT compile.
9852-   755	        compile_prewarmed_by_parity_ref=bool(
--
9861-   764	        # Single-rank rows have no partition, hence trivially true.
9862-   765	        stage_halo_correct=bool(n_ranks == 1 or halo_refresh == "in_step"),
9863-   766	        stage_halo_note=stage_halo_note_for(n_ranks, halo_refresh),
9864-   767	        fused=fused,
9865:   768	        wet_cell=wet_rec,
9866-   769	        solver_iters=solver_iters,
9867-   770	        solver_iters_mode=solver_iters_mode,
9868-   771	        zero_forcing_probe_residual=zero_forcing_probe_residual,
9869-   772	        zero_forcing_probe_measured=zero_forcing_probe_measured,
--
10144-747-        cells=int(mesh.nCells) * args.nlev,
10145-748-        # HORIZONTAL cells/rank — the same unit as --cells-per-rank, so a
10146-749-        # weak-mode row is comparable to its target (codex: the 3-D count
10147-750-        # made rows look nlev-x larger).
10148:751-        cells_per_rank_achieved=int(mesh.nCells) // n_ranks,
10149-752-        # Fix 4 honesty flag: at np=1 the parity reference pre-runs the
10150-753:        # SAME shape before the timed loop, so per_step_ms[0] may not
10151-754-        # contain the real JIT compile.
10152-755-        compile_prewarmed_by_parity_ref=bool(
--
10175-/usr/bin/bash -lc "printf '%s\\n' '--- job output locations ---'; find . -maxdepth 3 -type f -iname '*2660009[45]*' -o -iname '*26627810*' | sort; printf '%s\\n' '--- exact subdiv-8 rows used in weak pairs ---'; rg -l '\"component\": \"mpas_atm\"' . --glob '*.log' --glob '*.jsonl' 2>/dev/null | while IFS= read -r f; do rg -n '\"n_devices\": (8|16|32).*\"steady_median_ms\": (6\\.92|7\\.10|8\\.13)|\"steady_median_ms\": (6\\.92|7\\.10|8\\.13).*\"n_devices\": (8|16|32)' \""'$f" && printf '"'%s\\n' \"file="'$f"; done | head -n 800; printf '"'%s\\n' '--- current / receipt commit history ---'; git log --all --decorate --oneline -n 40; printf '%s\\n' '--- reachable receipt commits ---'; git log --all --oneline --ancestry-path 70f3ce636..HEAD 2>/dev/null | head -n 300; git show -s --format='%H%n%ad%n%s' --date=iso-strict 70f3ce636 10533f1f9 2>/dev/null; printf '%s\\n' '--- local branch status / selected diffs ---'; git diff --stat HEAD; git diff --check HEAD -- docs/performance/scaling/levante_campaign_2026-07-24.md scripts/plot/plot_scaling_paper_figure.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
10176- succeeded in 9181ms:
10177---- job output locations ---
10178-./mpas_s9.26600095.log
10179:./mpasoc_metis.26600094.log
10180---- exact subdiv-8 rows used in weak pairs ---
10181-67:{"component": "mpas_atm", "subdivision": 8, "n_devices": 32, "n_cells": 655392, "n_edges": 1966080, "nlev": 26, "partition_method": "sfc", "physics": "none", "halo_strategy_requested": "auto", "halo_strategy_effective": "ppermute", "steps": 12, "dt": 30.0, "platform": "gpu", "n_processes": 32, "multicontroller": true, "compile_ms": 6060.7, "steady_median_ms": 8.13, "steady_min_ms": 8.01, "per_step_ms": [6060.7, 11.4, 8.6, 8.3, 8.3, 8.2, 8.1, 8.1, 8.0, 8.0, 8.8, 8.0], "cells": 17040192, "hlo_collective_permutes": null, "hlo_collectives": null, "grid_type": "icosahedral", "resolution": 8, "n_levels": 26, "mode": "strong", "precision": "float32", "physics_level": "none", "backend": "gpu", "dt_seconds": 30.0, "time_per_step_ms": 8.125972002744675, "total_cells": 17040192, "sypd": 10.107778317008782, "mcells_per_s": 2097.0035331458694, "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-30T00:29:33.554535+00:00", "grid": "icosahedral", "component": "atmosphere", "resolution": "L8", "n_levels": 26, "precision": "float32", "decomposition": "cell_partition", "solver_variant": "n/a", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 32, "n_gpus": 32, "device_count": 32, "process_count": 32, "devices_per_rank": 1, "cells_per_rank": 532506, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "0", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "nccl", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50112.lvt.dkrz.de", "slurm_job_id": "26549646", "git_sha": "10533f1f9", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"partition_method": "sfc", "physics": "none", "steps": 12, "multicontroller": true, "cells_per_device": 532506}}}
10182-file=./mpas32.26549646.log
10183---- current / receipt commit history ---
--
10237- scripts/plot/plot_scaling_paper_figure.py          | 16 +++--
10238- 2 files changed, 85 insertions(+), 5 deletions(-)
10239-
10240-exec
10241:/usr/bin/bash -lc "printf '%s\\n' '--- receipt files and relevant rows ---'; ls -1 mpas*.log | sort; for F in mpas*.log; do rg -n '\"component\": \"mpas_atm\".*\"n_devices\": (8|16|32)|\"n_devices\": (8|16|32).*\"component\": \"mpas_atm\"' \""'$F" | head -n 20 && echo "FILE=$F"; done; printf '"'%s\\n' '--- lloyd option history and job chronology ---'; git show -s --format='%H%n%ad%n%s%n' --date=iso-strict 7151d12a1 00624f397 10533f1f9 9745e4da6 70f3ce636; git show --stat --oneline 7151d12a1; git show 10533f1f9:scripts/bench/bench_mpas_spmd_scaling.py | rg -n -C 4 'lloyd|create_voronoi_mesh|rec = dict|n_cells'; printf '%s\\n' '--- current s9 exact receipt (all JSON rows) ---'; rg -n '"'^'"\\{\"component\": \"mpas_atm\"' mpas_s9.26600095.log; printf '%s\\n' '--- current metis exact receipt (all JSON rows) ---'; rg -n '"'^'"\\{\"component\": \"ocean\"' mpasoc_metis.26600094.log" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
10242- succeeded in 773ms:
10243---- receipt files and relevant rows ---
10244-mpas128.26538474.log
10245-mpas32.26534061.log
--
10265-mpasoc_32.26494908.log
10266-mpasoc_512.26508063.log
10267-mpasoc_cpu.26494036.log
10268-mpasoc_metis.26600076.log
10269:mpasoc_metis.26600094.log
10270-mpasoc_out.26504842.log
10271-mpasoc_rpn.26505286.log
10272-FILE=mpas128.26538474.log
10273-FILE=mpas32.26534061.log
--
10310-FILE=mpasoc_32.26494908.log
10311-FILE=mpasoc_512.26508063.log
10312-FILE=mpasoc_cpu.26494036.log
10313-FILE=mpasoc_metis.26600076.log
10314:FILE=mpasoc_metis.26600094.log
10315-FILE=mpasoc_out.26504842.log
10316-FILE=mpasoc_rpn.26505286.log
10317---- lloyd option history and job chronology ---
10318-7151d12a184cf7324495b264e8bfa087d6ba4737
--
10370-67:{"component": "mpas_atm", "subdivision": 9, "n_devices": 32, "n_cells": 2621568, "n_edges": 7864320, "nlev": 26, "partition_method": "sfc", "physics": "none", "lloyd_iterations": 0, "halo_strategy_requested": "auto", "halo_strategy_effective": "ppermute", "steps": 12, "dt": 30.0, "platform": "gpu", "n_processes": 32, "multicontroller": true, "compile_ms": 7179.2, "steady_median_ms": 12.47, "steady_min_ms": 12.22, "per_step_ms": [7179.2, 12.9, 12.6, 12.5, 12.5, 12.5, 12.5, 12.4, 12.5, 12.5, 12.2, 12.2], "cells": 68160768, "hlo_collective_permutes": null, "hlo_collectives": null, "grid_type": "icosahedral", "resolution": 9, "n_levels": 26, "mode": "strong", "precision": "float32", "physics_level": "none", "backend": "gpu", "dt_seconds": 30.0, "time_per_step_ms": 12.468885004636832, "total_cells": 68160768, "sypd": 6.587238841598036, "mcells_per_s": 5466.468571540511, "metadata": {"schema_version": 2, "timestamp_utc": "2026-08-01T03:24:40.404674+00:00", "grid": "icosahedral", "component": "atmosphere", "resolution": "L9", "n_levels": 26, "precision": "float32", "decomposition": "cell_partition", "solver_variant": "n/a", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 32, "n_gpus": 32, "device_count": 32, "process_count": 32, "devices_per_rank": 1, "cells_per_rank": 2130024, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "0", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "nccl", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50000.lvt.dkrz.de", "slurm_job_id": "26600095", "git_sha": "unknown", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"partition_method": "sfc", "physics": "none", "steps": 12, "multicontroller": true, "cells_per_device": 2130024}}}
10371-294:{"component": "mpas_atm", "subdivision": 9, "n_devices": 64, "n_cells": 2621568, "n_edges": 7864320, "nlev": 26, "partition_method": "sfc", "physics": "none", "lloyd_iterations": 0, "halo_strategy_requested": "auto", "halo_strategy_effective": "ppermute", "steps": 12, "dt": 30.0, "platform": "gpu", "n_processes": 64, "multicontroller": true, "compile_ms": 6474.6, "steady_median_ms": 9.6, "steady_min_ms": 9.55, "per_step_ms": [6474.6, 13.0, 10.1, 10.0, 9.6, 9.8, 9.6, 9.6, 9.6, 9.6, 9.5, 9.6], "cells": 68160768, "hlo_collective_permutes": null, "hlo_collectives": null, "grid_type": "icosahedral", "resolution": 9, "n_levels": 26, "mode": "strong", "precision": "float32", "physics_level": "none", "backend": "gpu", "dt_seconds": 30.0, "time_per_step_ms": 9.601722995284945, "total_cells": 68160768, "sypd": 8.55424840461413, "mcells_per_s": 7098.805915716509, "metadata": {"schema_version": 2, "timestamp_utc": "2026-08-01T03:31:30.728837+00:00", "grid": "icosahedral", "component": "atmosphere", "resolution": "L9", "n_levels": 26, "precision": "float32", "decomposition": "cell_partition", "solver_variant": "n/a", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 64, "n_gpus": 64, "device_count": 64, "process_count": 64, "devices_per_rank": 1, "cells_per_rank": 1065012, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "0", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "nccl", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50000.lvt.dkrz.de", "slurm_job_id": "26600095", "git_sha": "7151d12a1", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"partition_method": "sfc", "physics": "none", "steps": 12, "multicontroller": true, "cells_per_device": 1065012}}}
10372-929:{"component": "mpas_atm", "subdivision": 9, "n_devices": 128, "n_cells": 2621568, "n_edges": 7864320, "nlev": 26, "partition_method": "sfc", "physics": "none", "lloyd_iterations": 0, "halo_strategy_requested": "auto", "halo_strategy_effective": "ppermute", "steps": 12, "dt": 30.0, "platform": "gpu", "n_processes": 128, "multicontroller": true, "compile_ms": 7532.3, "steady_median_ms": 11.48, "steady_min_ms": 11.37, "per_step_ms": [7532.3, 15.4, 12.4, 11.7, 11.5, 11.6, 11.4, 11.4, 11.5, 11.4, 11.5, 11.4], "cells": 68160768, "hlo_collective_permutes": null, "hlo_collectives": null, "grid_type": "icosahedral", "resolution": 9, "n_levels": 26, "mode": "strong", "precision": "float32", "physics_level": "none", "backend": "gpu", "dt_seconds": 30.0, "time_per_step_ms": 11.482901012641378, "total_cells": 68160768, "sypd": 7.152854798934616, "mcells_per_s": 5935.8491312398055, "metadata": {"schema_version": 2, "timestamp_utc": "2026-08-01T03:45:45.637722+00:00", "grid": "icosahedral", "component": "atmosphere", "resolution": "L9", "n_levels": 26, "precision": "float32", "decomposition": "cell_partition", "solver_variant": "n/a", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 128, "n_gpus": 128, "device_count": 128, "process_count": 128, "devices_per_rank": 1, "cells_per_rank": 532506, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "0", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "nccl", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50000.lvt.dkrz.de", "slurm_job_id": "26600095", "git_sha": "7151d12a1", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"partition_method": "sfc", "physics": "none", "steps": 12, "multicontroller": true, "cells_per_device": 532506}}}
10373---- current metis exact receipt (all JSON rows) ---
10374:450:{"component": "ocean", "grid": "voronoi", "mode": "strong", "subdivision": 7, "n_ranks": 32, "n_cells": 163842, "n_edges": 491520, "nlev": 20, "partition_method": "geometric", "steps": 12, "dt": 60.0, "platform": "cpu", "compile_ms": 12255.1, "steady_median_ms": 189.8206, "step_latency_gate_loop_ms": 187.13, "step_latency_gate_loop_min_ms": 185.51, "per_step_ms": [12255.1, 188.4, 187.4, 187.6, 187.4, 186.3, 186.9, 186.0, 188.5, 186.6, 185.5, 190.5], "cells": 3276840, "cells_per_rank_achieved": 5120, "compile_prewarmed_by_parity_ref": false, "barotropic_solver": "explicit_substep", "halo_refresh": "in_step", "stage_halo_correct": true, "stage_halo_note": "per-step packed entry refresh + in-step stage-frontier refreshes (make_mpas_ocean_halo_refresh) \u2014 stage-correct; see docs/performance/scaling/mpas_ocean_distributed_stage_audit.md", "fused": {"compile_ms": 183.7, "scan_compile_ms": 13431.3, "step_latency_ms": 187.99, "block_ms": [1516.95, 1516.98], "parallel_block_ms": [1518.81, 1518.32], "fused_step_ms": 189.8206, "rank_imbalance": 1.0011, "rank_imbalance_per_block": [1.0012, 1.0011], "block_steps": 8, "n_blocks": 2, "probe_steps": 3}, "wet_cell": {"wet_cell_levels": 3264640, "wet_cell_levels_per_device": 102020.0, "wet_fraction": 0.9962769009167368, "wet_equals_total": false, "wet_cell_levels_per_device_min": 100740, "wet_cell_levels_per_device_max": 102420}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-31T17:19:09.693551+00:00", "grid": "voronoi", "component": "ocean", "resolution": "L7", "n_levels": 20, "precision": "float64", "decomposition": "cell_partition", "solver_variant": "mpas_ocean_explicit_substep", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 32, "n_gpus": 0, "device_count": 1, "process_count": 1, "devices_per_rank": 1, "cells_per_rank": 102401, "backend": "cpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "mpi4jax", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l20119.lvt.dkrz.de", "slurm_job_id": "26600094", "git_sha": "7151d12a1", "gpu_direct_requested": false, "mpi4jax_cuda_support": false, "gpu_direct_active": false, "host_staged_halo": false, "partition_metrics": {"n_owned_cells": 5120, "n_halo_cells": 628, "owned_halo_ratio": 0.12265625, "n_neighbor_ranks": 6, "messages_per_exchange": 6, "halo_recv_cells": 628, "owned_send_cells": 626, "cells_per_rank_min": 5120, "cells_per_rank_max": 5121, "edge_cut_total": 20654, "max_neighbor_ranks": 8}, "extra": {"partition_method": "geometric", "steps": 12, "warmup": 2, "parity_gate": false, "check_conservation": false, "halo_refresh": "in_step", "barotropic_solver": "explicit_substep", "block_steps": 8, "blocks": 2, "probe_steps": 3}}}
10375:2149:{"component": "ocean", "grid": "voronoi", "mode": "strong", "subdivision": 8, "n_ranks": 128, "n_cells": 655362, "n_edges": 1966080, "nlev": 20, "partition_method": "geometric", "steps": 12, "dt": 60.0, "platform": "cpu", "compile_ms": 12628.0, "steady_median_ms": 308.9581, "step_latency_gate_loop_ms": 311.3, "step_latency_gate_loop_min_ms": 308.75, "per_step_ms": [12628.0, 312.0, 311.7, 310.9, 309.9, 310.0, 312.0, 311.9, 312.1, 308.7, 309.7, 311.7], "cells": 13107240, "cells_per_rank_achieved": 5120, "compile_prewarmed_by_parity_ref": false, "barotropic_solver": "explicit_substep", "halo_refresh": "in_step", "stage_halo_correct": true, "stage_halo_note": "per-step packed entry refresh + in-step stage-frontier refreshes (make_mpas_ocean_halo_refresh) \u2014 stage-correct; see docs/performance/scaling/mpas_ocean_distributed_stage_audit.md", "fused": {"compile_ms": 308.8, "scan_compile_ms": 14719.4, "step_latency_ms": 311.988, "block_ms": [2471.77, 2467.17], "parallel_block_ms": [2472.65, 2470.68], "fused_step_ms": 308.9581, "rank_imbalance": 1.0008, "rank_imbalance_per_block": [1.0008, 1.0008], "block_steps": 8, "n_blocks": 2, "probe_steps": 3}, "wet_cell": {"wet_cell_levels": 13057600, "wet_cell_levels_per_device": 102012.5, "wet_fraction": 0.9962127801123654, "wet_equals_total": false, "wet_cell_levels_per_device_min": 96040, "wet_cell_levels_per_device_max": 102420}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-31T17:20:07.513962+00:00", "grid": "voronoi", "component": "ocean", "resolution": "L8", "n_levels": 20, "precision": "float64", "decomposition": "cell_partition", "solver_variant": "mpas_ocean_explicit_substep", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 128, "n_gpus": 0, "device_count": 1, "process_count": 1, "devices_per_rank": 1, "cells_per_rank": 102400, "backend": "cpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "mpi4jax", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l20119.lvt.dkrz.de", "slurm_job_id": "26600094", "git_sha": "7151d12a1", "gpu_direct_requested": false, "mpi4jax_cuda_support": false, "gpu_direct_active": false, "host_staged_halo": false, "partition_metrics": {"n_owned_cells": 5120, "n_halo_cells": 658, "owned_halo_ratio": 0.128515625, "n_neighbor_ranks": 7, "messages_per_exchange": 7, "halo_recv_cells": 658, "owned_send_cells": 661, "cells_per_rank_min": 5120, "cells_per_rank_max": 5121, "edge_cut_total": 80510, "max_neighbor_ranks": 8}, "extra": {"partition_method": "geometric", "steps": 12, "warmup": 2, "parity_gate": false, "check_conservation": false, "halo_refresh": "in_step", "barotropic_solver": "explicit_substep", "block_steps": 8, "blocks": 2, "probe_steps": 3}}}
10376:2687:{"component": "ocean", "grid": "voronoi", "mode": "strong", "subdivision": 7, "n_ranks": 32, "n_cells": 163842, "n_edges": 491520, "nlev": 20, "partition_method": "metis", "steps": 12, "dt": 60.0, "platform": "cpu", "compile_ms": 12579.2, "steady_median_ms": 191.9856, "step_latency_gate_loop_ms": 190.29, "step_latency_gate_loop_min_ms": 188.92, "per_step_ms": [12579.2, 193.6, 191.0, 190.2, 188.9, 191.2, 189.7, 189.8, 188.9, 192.5, 191.4, 190.4], "cells": 3276840, "cells_per_rank_achieved": 5120, "compile_prewarmed_by_parity_ref": false, "barotropic_solver": "explicit_substep", "halo_refresh": "in_step", "stage_halo_correct": true, "stage_halo_note": "per-step packed entry refresh + in-step stage-frontier refreshes (make_mpas_ocean_halo_refresh) \u2014 stage-correct; see docs/performance/scaling/mpas_ocean_distributed_stage_audit.md", "fused": {"compile_ms": 186.8, "scan_compile_ms": 14192.4, "step_latency_ms": 188.993, "block_ms": [1534.07, 1532.56], "parallel_block_ms": [1535.85, 1535.92], "fused_step_ms": 191.9856, "rank_imbalance": 1.0012, "rank_imbalance_per_block": [1.0012, 1.0011], "block_steps": 8, "n_blocks": 2, "probe_steps": 3}, "wet_cell": {"wet_cell_levels": 3264640, "wet_cell_levels_per_device": 102020.0, "wet_fraction": 0.9962769009167368, "wet_equals_total": false, "wet_cell_levels_per_device_min": 96800, "wet_cell_levels_per_device_max": 102720}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-31T17:20:51.591156+00:00", "grid": "voronoi", "component": "ocean", "resolution": "L7", "n_levels": 20, "precision": "float64", "decomposition": "cell_partition", "solver_variant": "mpas_ocean_explicit_substep", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 32, "n_gpus": 0, "device_count": 1, "process_count": 1, "devices_per_rank": 1, "cells_per_rank": 102401, "backend": "cpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "mpi4jax", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l20119.lvt.dkrz.de", "slurm_job_id": "26600094", "git_sha": "7151d12a1", "gpu_direct_requested": false, "mpi4jax_cuda_support": false, "gpu_direct_active": false, "host_staged_halo": false, "partition_metrics": {"n_owned_cells": 5133, "n_halo_cells": 549, "owned_halo_ratio": 0.10695499707773232, "n_neighbor_ranks": 6, "messages_per_exchange": 6, "halo_recv_cells": 549, "owned_send_cells": 556, "cells_per_rank_min": 5100, "cells_per_rank_max": 5145, "edge_cut_total": 18784, "max_neighbor_ranks": 7}, "extra": {"partition_method": "metis", "steps": 12, "warmup": 2, "parity_gate": false, "check_conservation": false, "halo_refresh": "in_step", "barotropic_solver": "explicit_substep", "block_steps": 8, "blocks": 2, "probe_steps": 3}}}
10377:4387:{"component": "ocean", "grid": "voronoi", "mode": "strong", "subdivision": 8, "n_ranks": 128, "n_cells": 655362, "n_edges": 1966080, "nlev": 20, "partition_method": "metis", "steps": 12, "dt": 60.0, "platform": "cpu", "compile_ms": 13554.5, "steady_median_ms": 333.39, "step_latency_gate_loop_ms": 348.11, "step_latency_gate_loop_min_ms": 345.44, "per_step_ms": [13554.5, 350.4, 352.4, 348.9, 348.3, 349.4, 348.0, 347.5, 347.1, 348.5, 345.4, 347.6], "cells": 13107240, "cells_per_rank_achieved": 5120, "compile_prewarmed_by_parity_ref": false, "barotropic_solver": "explicit_substep", "halo_refresh": "in_step", "stage_halo_correct": true, "stage_halo_note": "per-step packed entry refresh + in-step stage-frontier refreshes (make_mpas_ocean_halo_refresh) \u2014 stage-correct; see docs/performance/scaling/mpas_ocean_distributed_stage_audit.md", "fused": {"compile_ms": 343.0, "scan_compile_ms": 15677.5, "step_latency_ms": 343.721, "block_ms": [2663.79, 2664.88], "parallel_block_ms": [2666.74, 2667.5], "fused_step_ms": 333.39, "rank_imbalance": 1.0007, "rank_imbalance_per_block": [1.0007, 1.0008], "block_steps": 8, "n_blocks": 2, "probe_steps": 3}, "wet_cell": {"wet_cell_levels": 13057600, "wet_cell_levels_per_device": 102012.5, "wet_fraction": 0.9962127801123654, "wet_equals_total": false, "wet_cell_levels_per_device_min": 78000, "wet_cell_levels_per_device_max": 102880}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-31T17:21:47.977228+00:00", "grid": "voronoi", "component": "ocean", "resolution": "L8", "n_levels": 20, "precision": "float64", "decomposition": "cell_partition", "solver_variant": "mpas_ocean_explicit_substep", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 128, "n_gpus": 0, "device_count": 1, "process_count": 1, "devices_per_rank": 1, "cells_per_rank": 102400, "backend": "cpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "mpi4jax", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l20119.lvt.dkrz.de", "slurm_job_id": "26600094", "git_sha": "7151d12a1", "gpu_direct_requested": false, "mpi4jax_cuda_support": false, "gpu_direct_active": false, "host_staged_halo": false, "partition_metrics": {"n_owned_cells": 5113, "n_halo_cells": 562, "owned_halo_ratio": 0.10991590064541365, "n_neighbor_ranks": 4, "messages_per_exchange": 4, "halo_recv_cells": 562, "owned_send_cells": 553, "cells_per_rank_min": 5093, "cells_per_rank_max": 5144, "edge_cut_total": 75748, "max_neighbor_ranks": 8}, "extra": {"partition_method": "metis", "steps": 12, "warmup": 2, "parity_gate": false, "check_conservation": false, "halo_refresh": "in_step", "barotropic_solver": "explicit_substep", "block_steps": 8, "blocks": 2, "probe_steps": 3}}}
10378:6255:{"component": "ocean", "grid": "voronoi", "mode": "strong", "subdivision": 8, "n_ranks": 128, "n_cells": 655362, "n_edges": 1966080, "nlev": 20, "partition_method": "metis", "steps": 12, "dt": 60.0, "platform": "cpu", "compile_ms": 14329.9, "steady_median_ms": 537.905, "step_latency_gate_loop_ms": 557.24, "step_latency_gate_loop_min_ms": 553.39, "per_step_ms": [14329.9, 558.5, 561.4, 554.0, 558.4, 555.1, 560.6, 569.8, 559.2, 554.6, 556.1, 553.4], "cells": 13107240, "cells_per_rank_achieved": 5120, "compile_prewarmed_by_parity_ref": false, "barotropic_solver": "explicit_substep", "halo_refresh": "in_step", "stage_halo_correct": true, "stage_halo_note": "per-step packed entry refresh + in-step stage-frontier refreshes (make_mpas_ocean_halo_refresh) \u2014 stage-correct; see docs/performance/scaling/mpas_ocean_distributed_stage_audit.md", "fused": {"compile_ms": 547.5, "scan_compile_ms": 17807.2, "step_latency_ms": 547.997, "block_ms": [4295.08, 4293.81], "parallel_block_ms": [4303.79, 4302.69], "fused_step_ms": 537.905, "rank_imbalance": 1.0021, "rank_imbalance_per_block": [1.002, 1.0023], "block_steps": 8, "n_blocks": 2, "probe_steps": 3}, "wet_cell": {"wet_cell_levels": 13057600, "wet_cell_levels_per_device": 102012.5, "wet_fraction": 0.9962127801123654, "wet_equals_total": false, "wet_cell_levels_per_device_min": 78000, "wet_cell_levels_per_device_max": 102880}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-31T17:23:14.881483+00:00", "grid": "voronoi", "component": "ocean", "resolution": "L8", "n_levels": 20, "precision": "float64", "decomposition": "cell_partition", "solver_variant": "mpas_ocean_explicit_substep", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 128, "n_gpus": 0, "device_count": 1, "process_count": 1, "devices_per_rank": 1, "cells_per_rank": 102400, "backend": "cpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "mpi4jax", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l20119.lvt.dkrz.de", "slurm_job_id": "26600094", "git_sha": "7151d12a1", "gpu_direct_requested": false, "mpi4jax_cuda_support": false, "gpu_direct_active": false, "host_staged_halo": false, "partition_metrics": {"n_owned_cells": 5113, "n_halo_cells": 562, "owned_halo_ratio": 0.10991590064541365, "n_neighbor_ranks": 4, "messages_per_exchange": 4, "halo_recv_cells": 562, "owned_send_cells": 553, "cells_per_rank_min": 5093, "cells_per_rank_max": 5144, "edge_cut_total": 75748, "max_neighbor_ranks": 8}, "extra": {"partition_method": "metis", "steps": 12, "warmup": 2, "parity_gate": false, "check_conservation": false, "halo_refresh": "in_step", "barotropic_solver": "explicit_substep", "block_steps": 8, "blocks": 2, "probe_steps": 3}}}
10379-
10380-exec
10381:/usr/bin/bash -lc "printf '%s\\n' '--- s9 32 full exact record ---'; rg -n '"'^'"\\{\"component\": \"mpas_atm\"' mpas_s9.26600095.log | head -n 1; printf '%s\\n' '--- s8 weak-pair receipt candidates ---'; rg -n -l 'steady_median_ms\": (6\\.92|7\\.10|8\\.13)' --glob 'mpas*.log' . | sort; rg -n -C 0 'steady_median_ms\": (6\\.92|7\\.10|8\\.13)' --glob 'mpas*.log' . | head -n 100; printf '%s\\n' '--- lloyd state in old benchmark commit ---'; git show 10533f1f9:scripts/bench/bench_mpas_spmd_scaling.py | sed -n '90,220p'; printf '%s\\n' '--- phase job log start and launch flags ---'; sed -n '1,180p' mpas_s9.26600095.log; printf '%s\\n' '--- METIS job command echo / distribution provenance ---'; rg -n -C 3 -i 'distribution|block:|srun|"'^===|command|geometric|metis'"' mpasoc_metis.26600094.log | head -n 1000" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
10382- succeeded in 328ms:
10383---- s9 32 full exact record ---
10384-67:{"component": "mpas_atm", "subdivision": 9, "n_devices": 32, "n_cells": 2621568, "n_edges": 7864320, "nlev": 26, "partition_method": "sfc", "physics": "none", "lloyd_iterations": 0, "halo_strategy_requested": "auto", "halo_strategy_effective": "ppermute", "steps": 12, "dt": 30.0, "platform": "gpu", "n_processes": 32, "multicontroller": true, "compile_ms": 7179.2, "steady_median_ms": 12.47, "steady_min_ms": 12.22, "per_step_ms": [7179.2, 12.9, 12.6, 12.5, 12.5, 12.5, 12.5, 12.4, 12.5, 12.5, 12.2, 12.2], "cells": 68160768, "hlo_collective_permutes": null, "hlo_collectives": null, "grid_type": "icosahedral", "resolution": 9, "n_levels": 26, "mode": "strong", "precision": "float32", "physics_level": "none", "backend": "gpu", "dt_seconds": 30.0, "time_per_step_ms": 12.468885004636832, "total_cells": 68160768, "sypd": 6.587238841598036, "mcells_per_s": 5466.468571540511, "metadata": {"schema_version": 2, "timestamp_utc": "2026-08-01T03:24:40.404674+00:00", "grid": "icosahedral", "component": "atmosphere", "resolution": "L9", "n_levels": 26, "precision": "float32", "decomposition": "cell_partition", "solver_variant": "n/a", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 32, "n_gpus": 32, "device_count": 32, "process_count": 32, "devices_per_rank": 1, "cells_per_rank": 2130024, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "0", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "nccl", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50000.lvt.dkrz.de", "slurm_job_id": "26600095", "git_sha": "unknown", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"partition_method": "sfc", "physics": "none", "steps": 12, "multicontroller": true, "cells_per_device": 2130024}}}
10385---- s8 weak-pair receipt candidates ---
--
10698-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/lib/python3.14/site-packages/jax/_src/numpy/array_methods.py:125: UserWarning: Explicitly requested dtype float64 requested in astype is not available, and will be truncated to dtype float32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/jax-ml/jax#current-gotchas for more.
10699-  return lax_numpy.astype(self, dtype, copy=copy, device=device)
10700-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/lib/python3.14/site-packages/jax/_src/numpy/array_methods.py:125: UserWarning: Explicitly requested dtype float64 requested in astype is not available, and will be truncated to dtype float32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/jax-ml/jax#current-gotchas for more.
10701---- METIS job command echo / distribution provenance ---
10702:1:outdir=/scratch/b/b381103/legoesm_scaling/mpasoc_metis_j26600094
10703:2:--- s7 np=32 partition=geometric dist=block:cyclic ---
10704-3:[l20119.lvt.dkrz.de:1439071] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
10705-4:[l20119.lvt.dkrz.de:1439096] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
10706-5:[l20119.lvt.dkrz.de:1439079] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
10707-6:[l20119.lvt.dkrz.de:1439074] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
--
10739---
10740-447-[1785518297.281831] [l20119:1439072:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
10741-448-[1785518297.249801] [l20119:1439095:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
10742-449-[1785518297.254104] [l20119:1439079:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
10743:450:{"component": "ocean", "grid": "voronoi", "mode": "strong", "subdivision": 7, "n_ranks": 32, "n_cells": 163842, "n_edges": 491520, "nlev": 20, "partition_method": "geometric", "steps": 12, "dt": 60.0, "platform": "cpu", "compile_ms": 12255.1, "steady_median_ms": 189.8206, "step_latency_gate_loop_ms": 187.13, "step_latency_gate_loop_min_ms": 185.51, "per_step_ms": [12255.1, 188.4, 187.4, 187.6, 187.4, 186.3, 186.9, 186.0, 188.5, 186.6, 185.5, 190.5], "cells": 3276840, "cells_per_rank_achieved": 5120, "compile_prewarmed_by_parity_ref": false, "barotropic_solver": "explicit_substep", "halo_refresh": "in_step", "stage_halo_correct": true, "stage_halo_note": "per-step packed entry refresh + in-step stage-frontier refreshes (make_mpas_ocean_halo_refresh) \u2014 stage-correct; see docs/performance/scaling/mpas_ocean_distributed_stage_audit.md", "fused": {"compile_ms": 183.7, "scan_compile_ms": 13431.3, "step_latency_ms": 187.99, "block_ms": [1516.95, 1516.98], "parallel_block_ms": [1518.81, 1518.32], "fused_step_ms": 189.8206, "rank_imbalance": 1.0011, "rank_imbalance_per_block": [1.0012, 1.0011], "block_steps": 8, "n_blocks": 2, "probe_steps": 3}, "wet_cell": {"wet_cell_levels": 3264640, "wet_cell_levels_per_device": 102020.0, "wet_fraction": 0.9962769009167368, "wet_equals_total": false, "wet_cell_levels_per_device_min": 100740, "wet_cell_levels_per_device_max": 102420}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-31T17:19:09.693551+00:00", "grid": "voronoi", "component": "ocean", "resolution": "L7", "n_levels": 20, "precision": "float64", "decomposition": "cell_partition", "solver_variant": "mpas_ocean_explicit_substep", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 32, "n_gpus": 0, "device_count": 1, "process_count": 1, "devices_per_rank": 1, "cells_per_rank": 102401, "backend": "cpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "mpi4jax", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l20119.lvt.dkrz.de", "slurm_job_id": "26600094", "git_sha": "7151d12a1", "gpu_direct_requested": false, "mpi4jax_cuda_support": false, "gpu_direct_active": false, "host_staged_halo": false, "partition_metrics": {"n_owned_cells": 5120, "n_halo_cells": 628, "owned_halo_ratio": 0.12265625, "n_neighbor_ranks": 6, "messages_per_exchange": 6, "halo_recv_cells": 628, "owned_send_cells": 626, "cells_per_rank_min": 5120, "cells_per_rank_max": 5121, "edge_cut_total": 20654, "max_neighbor_ranks": 8}, "extra": {"partition_method": "geometric", "steps": 12, "warmup": 2, "parity_gate": false, "check_conservation": false, "halo_refresh": "in_step", "barotropic_solver": "explicit_substep", "block_steps": 8, "blocks": 2, "probe_steps": 3}}}
10744:451-[mpas-ocean np=32 L7 nCells=163842 nlev=20 solver=explicit_substep halo=in_step] compile=12255.1ms fused=189.8206ms/step (probe_latency=187.99ms) gate_loop_latency=187.13ms/step
10745-452-[1785518297.289277] [l20119:1439070:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
10746:453:--- s8 np=128 partition=geometric dist=block:cyclic ---
10747-454:[l20131.lvt.dkrz.de:270332] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
10748-455:[l20119.lvt.dkrz.de:1440310] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
10749-456:[l20119.lvt.dkrz.de:1440300] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
10750-457:[l20119.lvt.dkrz.de:1440302] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
--
10878---
10879-2146-[1785518353.389333] [l20119:1440311:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
10880-2147-[1785518353.342035] [l20119:1440310:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
10881-2148-[1785518353.342454] [l20119:1440297:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
10882:2149:{"component": "ocean", "grid": "voronoi", "mode": "strong", "subdivision": 8, "n_ranks": 128, "n_cells": 655362, "n_edges": 1966080, "nlev": 20, "partition_method": "geometric", "steps": 12, "dt": 60.0, "platform": "cpu", "compile_ms": 12628.0, "steady_median_ms": 308.9581, "step_latency_gate_loop_ms": 311.3, "step_latency_gate_loop_min_ms": 308.75, "per_step_ms": [12628.0, 312.0, 311.7, 310.9, 309.9, 310.0, 312.0, 311.9, 312.1, 308.7, 309.7, 311.7], "cells": 13107240, "cells_per_rank_achieved": 5120, "compile_prewarmed_by_parity_ref": false, "barotropic_solver": "explicit_substep", "halo_refresh": "in_step", "stage_halo_correct": true, "stage_halo_note": "per-step packed entry refresh + in-step stage-frontier refreshes (make_mpas_ocean_halo_refresh) \u2014 stage-correct; see docs/performance/scaling/mpas_ocean_distributed_stage_audit.md", "fused": {"compile_ms": 308.8, "scan_compile_ms": 14719.4, "step_latency_ms": 311.988, "block_ms": [2471.77, 2467.17], "parallel_block_ms": [2472.65, 2470.68], "fused_step_ms": 308.9581, "rank_imbalance": 1.0008, "rank_imbalance_per_block": [1.0008, 1.0008], "block_steps": 8, "n_blocks": 2, "probe_steps": 3}, "wet_cell": {"wet_cell_levels": 13057600, "wet_cell_levels_per_device": 102012.5, "wet_fraction": 0.9962127801123654, "wet_equals_total": false, "wet_cell_levels_per_device_min": 96040, "wet_cell_levels_per_device_max": 102420}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-31T17:20:07.513962+00:00", "grid": "voronoi", "component": "ocean", "resolution": "L8", "n_levels": 20, "precision": "float64", "decomposition": "cell_partition", "solver_variant": "mpas_ocean_explicit_substep", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 128, "n_gpus": 0, "device_count": 1, "process_count": 1, "devices_per_rank": 1, "cells_per_rank": 102400, "backend": "cpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "mpi4jax", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l20119.lvt.dkrz.de", "slurm_job_id": "26600094", "git_sha": "7151d12a1", "gpu_direct_requested": false, "mpi4jax_cuda_support": false, "gpu_direct_active": false, "host_staged_halo": false, "partition_metrics": {"n_owned_cells": 5120, "n_halo_cells": 658, "owned_halo_ratio": 0.128515625, "n_neighbor_ranks": 7, "messages_per_exchange": 7, "halo_recv_cells": 658, "owned_send_cells": 661, "cells_per_rank_min": 5120, "cells_per_rank_max": 5121, "edge_cut_total": 80510, "max_neighbor_ranks": 8}, "extra": {"partition_method": "geometric", "steps": 12, "warmup": 2, "parity_gate": false, "check_conservation": false, "halo_refresh": "in_step", "barotropic_solver": "explicit_substep", "block_steps": 8, "blocks": 2, "probe_steps": 3}}}
10883-2150-[mpas-ocean np=128 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=12628.0ms fused=308.9581ms/step (probe_latency=311.988ms) gate_loop_latency=311.30ms/step
10884-2151-[1785518353.372656] [l20119:1440282:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
10885-2152-[1785518357.149452] [l20130:50192:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
10886---
10887-2245-[1785518357.102350] [l20131:270336:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
10888-2246-[1785518357.111728] [l20147:3468423:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
10889-2247-[1785518357.104151] [l20131:270358:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
10890:2248:--- s7 np=32 partition=metis dist=block:cyclic ---
10891-2249:[l20119.lvt.dkrz.de:1441497] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
10892-2250:[l20119.lvt.dkrz.de:1441493] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
10893-2251:[l20119.lvt.dkrz.de:1441506] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
10894-2252:[l20119.lvt.dkrz.de:1441507] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
--
10926---
10927-2684-[1785518411.087402] [l20119:1441479:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
10928-2685-[1785518411.080442] [l20119:1441486:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
10929-2686-[1785518411.064649] [l20119:1441490:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
10930:2687:{"component": "ocean", "grid": "voronoi", "mode": "strong", "subdivision": 7, "n_ranks": 32, "n_cells": 163842, "n_edges": 491520, "nlev": 20, "partition_method": "metis", "steps": 12, "dt": 60.0, "platform": "cpu", "compile_ms": 12579.2, "steady_median_ms": 191.9856, "step_latency_gate_loop_ms": 190.29, "step_latency_gate_loop_min_ms": 188.92, "per_step_ms": [12579.2, 193.6, 191.0, 190.2, 188.9, 191.2, 189.7, 189.8, 188.9, 192.5, 191.4, 190.4], "cells": 3276840, "cells_per_rank_achieved": 5120, "compile_prewarmed_by_parity_ref": false, "barotropic_solver": "explicit_substep", "halo_refresh": "in_step", "stage_halo_correct": true, "stage_halo_note": "per-step packed entry refresh + in-step stage-frontier refreshes (make_mpas_ocean_halo_refresh) \u2014 stage-correct; see docs/performance/scaling/mpas_ocean_distributed_stage_audit.md", "fused": {"compile_ms": 186.8, "scan_compile_ms": 14192.4, "step_latency_ms": 188.993, "block_ms": [1534.07, 1532.56], "parallel_block_ms": [1535.85, 1535.92], "fused_step_ms": 191.9856, "rank_imbalance": 1.0012, "rank_imbalance_per_block": [1.0012, 1.0011], "block_steps": 8, "n_blocks": 2, "probe_steps": 3}, "wet_cell": {"wet_cell_levels": 3264640, "wet_cell_levels_per_device": 102020.0, "wet_fraction": 0.9962769009167368, "wet_equals_total": false, "wet_cell_levels_per_device_min": 96800, "wet_cell_levels_per_device_max": 102720}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-31T17:20:51.591156+00:00", "grid": "voronoi", "component": "ocean", "resolution": "L7", "n_levels": 20, "precision": "float64", "decomposition": "cell_partition", "solver_variant": "mpas_ocean_explicit_substep", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 32, "n_gpus": 0, "device_count": 1, "process_count": 1, "devices_per_rank": 1, "cells_per_rank": 102401, "backend": "cpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "mpi4jax", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l20119.lvt.dkrz.de", "slurm_job_id": "26600094", "git_sha": "7151d12a1", "gpu_direct_requested": false, "mpi4jax_cuda_support": false, "gpu_direct_active": false, "host_staged_halo": false, "partition_metrics": {"n_owned_cells": 5133, "n_halo_cells": 549, "owned_halo_ratio": 0.10695499707773232, "n_neighbor_ranks": 6, "messages_per_exchange": 6, "halo_recv_cells": 549, "owned_send_cells": 556, "cells_per_rank_min": 5100, "cells_per_rank_max": 5145, "edge_cut_total": 18784, "max_neighbor_ranks": 7}, "extra": {"partition_method": "metis", "steps": 12, "warmup": 2, "parity_gate": false, "check_conservation": false, "halo_refresh": "in_step", "barotropic_solver": "explicit_substep", "block_steps": 8, "blocks": 2, "probe_steps": 3}}}
10931-2688-[mpas-ocean np=32 L7 nCells=163842 nlev=20 solver=explicit_substep halo=in_step] compile=12579.2ms fused=191.9856ms/step (probe_latency=188.993ms) gate_loop_latency=190.29ms/step
10932-2689-[1785518411.064180] [l20119:1441477:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
10933-2690-[1785518411.073813] [l20119:1441504:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
10934---
10935-2696-[1785518411.073814] [l20119:1441480:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
10936-2697-[1785518411.093296] [l20119:1441496:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
10937-2698-[1785518411.089328] [l20119:1441508:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
10938:2699:--- s8 np=128 partition=metis dist=block:cyclic ---
10939-2700:[l20147.lvt.dkrz.de:3469645] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
10940-2701:[l20147.lvt.dkrz.de:3469662] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
10941-2702:[l20147.lvt.dkrz.de:3469644] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
10942-2703:[l20147.lvt.dkrz.de:3469637] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
--
11070---
11071-4384-[1785518454.816638] [l20119:1442698:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
11072-4385-[1785518454.766060] [l20119:1442707:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
11073-4386-[1785518454.774753] [l20119:1442679:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
11074:4387:{"component": "ocean", "grid": "voronoi", "mode": "strong", "subdivision": 8, "n_ranks": 128, "n_cells": 655362, "n_edges": 1966080, "nlev": 20, "partition_method": "metis", "steps": 12, "dt": 60.0, "platform": "cpu", "compile_ms": 13554.5, "steady_median_ms": 333.39, "step_latency_gate_loop_ms": 348.11, "step_latency_gate_loop_min_ms": 345.44, "per_step_ms": [13554.5, 350.4, 352.4, 348.9, 348.3, 349.4, 348.0, 347.5, 347.1, 348.5, 345.4, 347.6], "cells": 13107240, "cells_per_rank_achieved": 5120, "compile_prewarmed_by_parity_ref": false, "barotropic_solver": "explicit_substep", "halo_refresh": "in_step", "stage_halo_correct": true, "stage_halo_note": "per-step packed entry refresh + in-step stage-frontier refreshes (make_mpas_ocean_halo_refresh) \u2014 stage-correct; see docs/performance/scaling/mpas_ocean_distributed_stage_audit.md", "fused": {"compile_ms": 343.0, "scan_compile_ms": 15677.5, "step_latency_ms": 343.721, "block_ms": [2663.79, 2664.88], "parallel_block_ms": [2666.74, 2667.5], "fused_step_ms": 333.39, "rank_imbalance": 1.0007, "rank_imbalance_per_block": [1.0007, 1.0008], "block_steps": 8, "n_blocks": 2, "probe_steps": 3}, "wet_cell": {"wet_cell_levels": 13057600, "wet_cell_levels_per_device": 102012.5, "wet_fraction": 0.9962127801123654, "wet_equals_total": false, "wet_cell_levels_per_device_min": 78000, "wet_cell_levels_per_device_max": 102880}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-31T17:21:47.977228+00:00", "grid": "voronoi", "component": "ocean", "resolution": "L8", "n_levels": 20, "precision": "float64", "decomposition": "cell_partition", "solver_variant": "mpas_ocean_explicit_substep", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 128, "n_gpus": 0, "device_count": 1, "process_count": 1, "devices_per_rank": 1, "cells_per_rank": 102400, "backend": "cpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "mpi4jax", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l20119.lvt.dkrz.de", "slurm_job_id": "26600094", "git_sha": "7151d12a1", "gpu_direct_requested": false, "mpi4jax_cuda_support": false, "gpu_direct_active": false, "host_staged_halo": false, "partition_metrics": {"n_owned_cells": 5113, "n_halo_cells": 562, "owned_halo_ratio": 0.10991590064541365, "n_neighbor_ranks": 4, "messages_per_exchange": 4, "halo_recv_cells": 562, "owned_send_cells": 553, "cells_per_rank_min": 5093, "cells_per_rank_max": 5144, "edge_cut_total": 75748, "max_neighbor_ranks": 8}, "extra": {"partition_method": "metis", "steps": 12, "warmup": 2, "parity_gate": false, "check_conservation": false, "halo_refresh": "in_step", "barotropic_solver": "explicit_substep", "block_steps": 8, "blocks": 2, "probe_steps": 3}}}
11075:4388-[mpas-ocean np=128 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=13554.5ms fused=333.39ms/step (probe_latency=343.721ms) gate_loop_latency=348.11ms/step
11076-4389-[1785518454.797213] [l20119:1442677:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
11077-4390-[1785518454.772658] [l20119:1442708:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
11078---
11079-4491-[1785518454.799469] [l20130:51421:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
11080-4492-[1785518454.769837] [l20131:271545:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
11081-4493-[1785518454.822726] [l20119:1442706:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
11082:4494:--- s8 np=128 partition=metis dist=block:block ---
11083-4495:[l20119.lvt.dkrz.de:1443884] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11084-4496:[l20131.lvt.dkrz.de:272741] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11085-4497:[l20131.lvt.dkrz.de:272728] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11086-4498:[l20131.lvt.dkrz.de:272733] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
--
11214---
11215-6252-[1785518521.847542] [l20119:1443892:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
11216-6253-[1785518521.996105] [l20130:52583:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
11217-6254-[1785518521.884325] [l20119:1443901:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
11218:6255:{"component": "ocean", "grid": "voronoi", "mode": "strong", "subdivision": 8, "n_ranks": 128, "n_cells": 655362, "n_edges": 1966080, "nlev": 20, "partition_method": "metis", "steps": 12, "dt": 60.0, "platform": "cpu", "compile_ms": 14329.9, "steady_median_ms": 537.905, "step_latency_gate_loop_ms": 557.24, "step_latency_gate_loop_min_ms": 553.39, "per_step_ms": [14329.9, 558.5, 561.4, 554.0, 558.4, 555.1, 560.6, 569.8, 559.2, 554.6, 556.1, 553.4], "cells": 13107240, "cells_per_rank_achieved": 5120, "compile_prewarmed_by_parity_ref": false, "barotropic_solver": "explicit_substep", "halo_refresh": "in_step", "stage_halo_correct": true, "stage_halo_note": "per-step packed entry refresh + in-step stage-frontier refreshes (make_mpas_ocean_halo_refresh) \u2014 stage-correct; see docs/performance/scaling/mpas_ocean_distributed_stage_audit.md", "fused": {"compile_ms": 547.5, "scan_compile_ms": 17807.2, "step_latency_ms": 547.997, "block_ms": [4295.08, 4293.81], "parallel_block_ms": [4303.79, 4302.69], "fused_step_ms": 537.905, "rank_imbalance": 1.0021, "rank_imbalance_per_block": [1.002, 1.0023], "block_steps": 8, "n_blocks": 2, "probe_steps": 3}, "wet_cell": {"wet_cell_levels": 13057600, "wet_cell_levels_per_device": 102012.5, "wet_fraction": 0.9962127801123654, "wet_equals_total": false, "wet_cell_levels_per_device_min": 78000, "wet_cell_levels_per_device_max": 102880}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-31T17:23:14.881483+00:00", "grid": "voronoi", "component": "ocean", "resolution": "L8", "n_levels": 20, "precision": "float64", "decomposition": "cell_partition", "solver_variant": "mpas_ocean_explicit_substep", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 128, "n_gpus": 0, "device_count": 1, "process_count": 1, "devices_per_rank": 1, "cells_per_rank": 102400, "backend": "cpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "mpi4jax", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l20119.lvt.dkrz.de", "slurm_job_id": "26600094", "git_sha": "7151d12a1", "gpu_direct_requested": false, "mpi4jax_cuda_support": false, "gpu_direct_active": false, "host_staged_halo": false, "partition_metrics": {"n_owned_cells": 5113, "n_halo_cells": 562, "owned_halo_ratio": 0.10991590064541365, "n_neighbor_ranks": 4, "messages_per_exchange": 4, "halo_recv_cells": 562, "owned_send_cells": 553, "cells_per_rank_min": 5093, "cells_per_rank_max": 5144, "edge_cut_total": 75748, "max_neighbor_ranks": 8}, "extra": {"partition_method": "metis", "steps": 12, "warmup": 2, "parity_gate": false, "check_conservation": false, "halo_refresh": "in_step", "barotropic_solver": "explicit_substep", "block_steps": 8, "blocks": 2, "probe_steps": 3}}}
11219-6256-[mpas-ocean np=128 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=14329.9ms fused=537.905ms/step (probe_latency=547.997ms) gate_loop_latency=557.24ms/step
11220-6257-[1785518521.623596] [l20119:1443880:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
11221-6258-[1785518522.112727] [l20130:52580:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
11222---
11223-6286-[1785518521.787274] [l20119:1443905:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
11224-6287-[1785518521.549753] [l20119:1443903:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
11225-6288-[1785518521.792657] [l20119:1443909:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
11226-6289:=== RESULTS (A-D all hold 5120 cells/rank) ===
11227:6290:A_s7np32_geometric:   189.82 ms  method=geometric ranks=32
11228:6291:B_s8np128_geometric:   308.96 ms  method=geometric ranks=128
11229:6292:C_s7np32_metis:   191.99 ms  method=metis ranks=32
11230:6293:D_s8np128_metis:   333.39 ms  method=metis ranks=128
11231-6294:E_s8np128_metis_blockblock:   537.90 ms  method=metis ranks=128
11232-6295-DONE rc=0
11233-6296-
11234-6297-********************************************************************************
11235---
11236-6305-*                       We hope you enjoyed the DKRZ supercomputer LEVANTE ... *
11237-6306-*
11238:6307-* JobID            : 26600094
11239-6308:* JobName          : mpasoc_metis                                      
11240-6309-* Account          : bb1596
11241-6310-* User             : b381103 (200166), bd1083 (1468)                   
11242-6311-* Partition        : compute
--
11252-
11253-codex
11254-The receipts expose two substantive confounds beyond wording: METIS does not hold every rank at 5,120 cells (it spans 5,093–5,144 at 128 ranks), and the s8 comparison rows predate the `--lloyd` control and therefore use the default relaxed mesh while s9 explicitly uses `lloyd=0`. Those invalidate the strong causal versions of the lever claims.
11255-exec
11256:/usr/bin/bash -lc "printf '%s\\n' '--- literal weak-pair rows, with provenance ---'; rg -n -F -e '\"steady_median_ms\": 6.92' -e '\"steady_median_ms\": 7.1' -e '\"steady_median_ms\": 8.13' mpas*.log; printf '%s\\n' '--- METIS quality statements / exact values ---'; rg -n -i -C 4 'partq|edge_cut|halo mean|halo_mean|1\\.89|2\\.01|592|629|cells_per_rank_(min|max)' mpasoc_metis.26600094.log | head -n 1400; printf '%s\\n' '--- test/warning and result tail ---'; rg -n -C 2 'RuntimeWarning: Detected versions outside|outside legoESM|stage_halo_correct|parity_gate|check_conservation|DONE rc|RESULTS' mpasoc_metis.26600094.log | tail -n 1000; printf '%s\\n' '--- exact current PLOT source table rendering / parser static check ---'; /usr/bin/python3 -c \"import ast; ast.parse(open('scripts/plot/plot_scaling_paper_figure.py').read()); print('AST OK')\"; tail -n 85 scripts/plot/plot_scaling_paper_figure.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
11257- succeeded in 418ms:
11258---- literal weak-pair rows, with provenance ---
11259-mpas32.26549646.log:67:{"component": "mpas_atm", "subdivision": 8, "n_devices": 32, "n_cells": 655392, "n_edges": 1966080, "nlev": 26, "partition_method": "sfc", "physics": "none", "halo_strategy_requested": "auto", "halo_strategy_effective": "ppermute", "steps": 12, "dt": 30.0, "platform": "gpu", "n_processes": 32, "multicontroller": true, "compile_ms": 6060.7, "steady_median_ms": 8.13, "steady_min_ms": 8.01, "per_step_ms": [6060.7, 11.4, 8.6, 8.3, 8.3, 8.2, 8.1, 8.1, 8.0, 8.0, 8.8, 8.0], "cells": 17040192, "hlo_collective_permutes": null, "hlo_collectives": null, "grid_type": "icosahedral", "resolution": 8, "n_levels": 26, "mode": "strong", "precision": "float32", "physics_level": "none", "backend": "gpu", "dt_seconds": 30.0, "time_per_step_ms": 8.125972002744675, "total_cells": 17040192, "sypd": 10.107778317008782, "mcells_per_s": 2097.0035331458694, "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-30T00:29:33.554535+00:00", "grid": "icosahedral", "component": "atmosphere", "resolution": "L8", "n_levels": 26, "precision": "float32", "decomposition": "cell_partition", "solver_variant": "n/a", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 32, "n_gpus": 32, "device_count": 32, "process_count": 32, "devices_per_rank": 1, "cells_per_rank": 532506, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "0", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "nccl", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50112.lvt.dkrz.de", "slurm_job_id": "26549646", "git_sha": "10533f1f9", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"partition_method": "sfc", "physics": "none", "steps": 12, "multicontroller": true, "cells_per_device": 532506}}}
11260-mpas_nsys.26479922.log:174:{"component": "mpas_atm", "subdivision": 8, "n_devices": 8, "n_cells": 655376, "n_edges": 1966080, "nlev": 26, "partition_method": "sfc", "physics": "none", "halo_strategy_requested": "auto", "halo_strategy_effective": "ppermute", "steps": 12, "dt": 30.0, "platform": "gpu", "n_processes": 8, "multicontroller": true, "compile_ms": 6621.3, "steady_median_ms": 7.17, "steady_min_ms": 7.06, "per_step_ms": [6621.3, 8.9, 8.4, 7.2, 7.2, 7.1, 7.1, 7.1, 11.7, 9.0, 10.9, 9.7], "cells": 17039776, "hlo_collective_permutes": null, "hlo_collectives": null, "grid_type": "icosahedral", "resolution": 8, "n_levels": 26, "mode": "strong", "precision": "float32", "physics_level": "none", "backend": "gpu", "dt_seconds": 30.0, "time_per_step_ms": 7.166521972976625, "total_cells": 17039776, "sypd": 11.461002132370208, "mcells_per_s": 2377.6911679407726, "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-26T10:40:28.596045+00:00", "grid": "icosahedral", "component": "atmosphere", "resolution": "L8", "n_levels": 26, "precision": "float32", "decomposition": "cell_partition", "solver_variant": "n/a", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 8, "n_gpus": 8, "device_count": 8, "process_count": 8, "devices_per_rank": 1, "cells_per_rank": 2129972, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "0", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "nccl", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50115.lvt.dkrz.de", "slurm_job_id": "26479922", "git_sha": "5fcfcb691", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"partition_method": "sfc", "physics": "none", "steps": 12, "multicontroller": true, "cells_per_device": 2129972}}}
--
11262-446-[1785518297.254555] [l20119:1439087:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
11263-447-[1785518297.281831] [l20119:1439072:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
11264-448-[1785518297.249801] [l20119:1439095:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
11265-449-[1785518297.254104] [l20119:1439079:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
11266:450:{"component": "ocean", "grid": "voronoi", "mode": "strong", "subdivision": 7, "n_ranks": 32, "n_cells": 163842, "n_edges": 491520, "nlev": 20, "partition_method": "geometric", "steps": 12, "dt": 60.0, "platform": "cpu", "compile_ms": 12255.1, "steady_median_ms": 189.8206, "step_latency_gate_loop_ms": 187.13, "step_latency_gate_loop_min_ms": 185.51, "per_step_ms": [12255.1, 188.4, 187.4, 187.6, 187.4, 186.3, 186.9, 186.0, 188.5, 186.6, 185.5, 190.5], "cells": 3276840, "cells_per_rank_achieved": 5120, "compile_prewarmed_by_parity_ref": false, "barotropic_solver": "explicit_substep", "halo_refresh": "in_step", "stage_halo_correct": true, "stage_halo_note": "per-step packed entry refresh + in-step stage-frontier refreshes (make_mpas_ocean_halo_refresh) \u2014 stage-correct; see docs/performance/scaling/mpas_ocean_distributed_stage_audit.md", "fused": {"compile_ms": 183.7, "scan_compile_ms": 13431.3, "step_latency_ms": 187.99, "block_ms": [1516.95, 1516.98], "parallel_block_ms": [1518.81, 1518.32], "fused_step_ms": 189.8206, "rank_imbalance": 1.0011, "rank_imbalance_per_block": [1.0012, 1.0011], "block_steps": 8, "n_blocks": 2, "probe_steps": 3}, "wet_cell": {"wet_cell_levels": 3264640, "wet_cell_levels_per_device": 102020.0, "wet_fraction": 0.9962769009167368, "wet_equals_total": false, "wet_cell_levels_per_device_min": 100740, "wet_cell_levels_per_device_max": 102420}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-31T17:19:09.693551+00:00", "grid": "voronoi", "component": "ocean", "resolution": "L7", "n_levels": 20, "precision": "float64", "decomposition": "cell_partition", "solver_variant": "mpas_ocean_explicit_substep", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 32, "n_gpus": 0, "device_count": 1, "process_count": 1, "devices_per_rank": 1, "cells_per_rank": 102401, "backend": "cpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "mpi4jax", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l20119.lvt.dkrz.de", "slurm_job_id": "26600094", "git_sha": "7151d12a1", "gpu_direct_requested": false, "mpi4jax_cuda_support": false, "gpu_direct_active": false, "host_staged_halo": false, "partition_metrics": {"n_owned_cells": 5120, "n_halo_cells": 628, "owned_halo_ratio": 0.12265625, "n_neighbor_ranks": 6, "messages_per_exchange": 6, "halo_recv_cells": 628, "owned_send_cells": 626, "cells_per_rank_min": 5120, "cells_per_rank_max": 5121, "edge_cut_total": 20654, "max_neighbor_ranks": 8}, "extra": {"partition_method": "geometric", "steps": 12, "warmup": 2, "parity_gate": false, "check_conservation": false, "halo_refresh": "in_step", "barotropic_solver": "explicit_substep", "block_steps": 8, "blocks": 2, "probe_steps": 3}}}
11267:451-[mpas-ocean np=32 L7 nCells=163842 nlev=20 solver=explicit_substep halo=in_step] compile=12255.1ms fused=189.8206ms/step (probe_latency=187.99ms) gate_loop_latency=187.13ms/step
11268-452-[1785518297.289277] [l20119:1439070:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
11269:453---- s8 np=128 partition=geometric dist=block:cyclic ---
11270-454-[l20131.lvt.dkrz.de:270332] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
11271---
11272-2145-[1785518353.342904] [l20119:1440313:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
11273-2146-[1785518353.389333] [l20119:1440311:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
11274-2147-[1785518353.342035] [l20119:1440310:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
11275-2148-[1785518353.342454] [l20119:1440297:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
11276:2149:{"component": "ocean", "grid": "voronoi", "mode": "strong", "subdivision": 8, "n_ranks": 128, "n_cells": 655362, "n_edges": 1966080, "nlev": 20, "partition_method": "geometric", "steps": 12, "dt": 60.0, "platform": "cpu", "compile_ms": 12628.0, "steady_median_ms": 308.9581, "step_latency_gate_loop_ms": 311.3, "step_latency_gate_loop_min_ms": 308.75, "per_step_ms": [12628.0, 312.0, 311.7, 310.9, 309.9, 310.0, 312.0, 311.9, 312.1, 308.7, 309.7, 311.7], "cells": 13107240, "cells_per_rank_achieved": 5120, "compile_prewarmed_by_parity_ref": false, "barotropic_solver": "explicit_substep", "halo_refresh": "in_step", "stage_halo_correct": true, "stage_halo_note": "per-step packed entry refresh + in-step stage-frontier refreshes (make_mpas_ocean_halo_refresh) \u2014 stage-correct; see docs/performance/scaling/mpas_ocean_distributed_stage_audit.md", "fused": {"compile_ms": 308.8, "scan_compile_ms": 14719.4, "step_latency_ms": 311.988, "block_ms": [2471.77, 2467.17], "parallel_block_ms": [2472.65, 2470.68], "fused_step_ms": 308.9581, "rank_imbalance": 1.0008, "rank_imbalance_per_block": [1.0008, 1.0008], "block_steps": 8, "n_blocks": 2, "probe_steps": 3}, "wet_cell": {"wet_cell_levels": 13057600, "wet_cell_levels_per_device": 102012.5, "wet_fraction": 0.9962127801123654, "wet_equals_total": false, "wet_cell_levels_per_device_min": 96040, "wet_cell_levels_per_device_max": 102420}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-31T17:20:07.513962+00:00", "grid": "voronoi", "component": "ocean", "resolution": "L8", "n_levels": 20, "precision": "float64", "decomposition": "cell_partition", "solver_variant": "mpas_ocean_explicit_substep", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 128, "n_gpus": 0, "device_count": 1, "process_count": 1, "devices_per_rank": 1, "cells_per_rank": 102400, "backend": "cpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "mpi4jax", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l20119.lvt.dkrz.de", "slurm_job_id": "26600094", "git_sha": "7151d12a1", "gpu_direct_requested": false, "mpi4jax_cuda_support": false, "gpu_direct_active": false, "host_staged_halo": false, "partition_metrics": {"n_owned_cells": 5120, "n_halo_cells": 658, "owned_halo_ratio": 0.128515625, "n_neighbor_ranks": 7, "messages_per_exchange": 7, "halo_recv_cells": 658, "owned_send_cells": 661, "cells_per_rank_min": 5120, "cells_per_rank_max": 5121, "edge_cut_total": 80510, "max_neighbor_ranks": 8}, "extra": {"partition_method": "geometric", "steps": 12, "warmup": 2, "parity_gate": false, "check_conservation": false, "halo_refresh": "in_step", "barotropic_solver": "explicit_substep", "block_steps": 8, "blocks": 2, "probe_steps": 3}}}
11277-2150-[mpas-ocean np=128 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=12628.0ms fused=308.9581ms/step (probe_latency=311.988ms) gate_loop_latency=311.30ms/step
11278-2151-[1785518353.372656] [l20119:1440282:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
11279-2152-[1785518357.149452] [l20130:50192:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
11280-2153-[1785518357.157028] [l20130:50198:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
--
11292-2683-[1785518411.094262] [l20119:1441493:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
11293-2684-[1785518411.087402] [l20119:1441479:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
11294-2685-[1785518411.080442] [l20119:1441486:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
11295-2686-[1785518411.064649] [l20119:1441490:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
11296:2687:{"component": "ocean", "grid": "voronoi", "mode": "strong", "subdivision": 7, "n_ranks": 32, "n_cells": 163842, "n_edges": 491520, "nlev": 20, "partition_method": "metis", "steps": 12, "dt": 60.0, "platform": "cpu", "compile_ms": 12579.2, "steady_median_ms": 191.9856, "step_latency_gate_loop_ms": 190.29, "step_latency_gate_loop_min_ms": 188.92, "per_step_ms": [12579.2, 193.6, 191.0, 190.2, 188.9, 191.2, 189.7, 189.8, 188.9, 192.5, 191.4, 190.4], "cells": 3276840, "cells_per_rank_achieved": 5120, "compile_prewarmed_by_parity_ref": false, "barotropic_solver": "explicit_substep", "halo_refresh": "in_step", "stage_halo_correct": true, "stage_halo_note": "per-step packed entry refresh + in-step stage-frontier refreshes (make_mpas_ocean_halo_refresh) \u2014 stage-correct; see docs/performance/scaling/mpas_ocean_distributed_stage_audit.md", "fused": {"compile_ms": 186.8, "scan_compile_ms": 14192.4, "step_latency_ms": 188.993, "block_ms": [1534.07, 1532.56], "parallel_block_ms": [1535.85, 1535.92], "fused_step_ms": 191.9856, "rank_imbalance": 1.0012, "rank_imbalance_per_block": [1.0012, 1.0011], "block_steps": 8, "n_blocks": 2, "probe_steps": 3}, "wet_cell": {"wet_cell_levels": 3264640, "wet_cell_levels_per_device": 102020.0, "wet_fraction": 0.9962769009167368, "wet_equals_total": false, "wet_cell_levels_per_device_min": 96800, "wet_cell_levels_per_device_max": 102720}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-31T17:20:51.591156+00:00", "grid": "voronoi", "component": "ocean", "resolution": "L7", "n_levels": 20, "precision": "float64", "decomposition": "cell_partition", "solver_variant": "mpas_ocean_explicit_substep", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 32, "n_gpus": 0, "device_count": 1, "process_count": 1, "devices_per_rank": 1, "cells_per_rank": 102401, "backend": "cpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "mpi4jax", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l20119.lvt.dkrz.de", "slurm_job_id": "26600094", "git_sha": "7151d12a1", "gpu_direct_requested": false, "mpi4jax_cuda_support": false, "gpu_direct_active": false, "host_staged_halo": false, "partition_metrics": {"n_owned_cells": 5133, "n_halo_cells": 549, "owned_halo_ratio": 0.10695499707773232, "n_neighbor_ranks": 6, "messages_per_exchange": 6, "halo_recv_cells": 549, "owned_send_cells": 556, "cells_per_rank_min": 5100, "cells_per_rank_max": 5145, "edge_cut_total": 18784, "max_neighbor_ranks": 7}, "extra": {"partition_method": "metis", "steps": 12, "warmup": 2, "parity_gate": false, "check_conservation": false, "halo_refresh": "in_step", "barotropic_solver": "explicit_substep", "block_steps": 8, "blocks": 2, "probe_steps": 3}}}
11297-2688-[mpas-ocean np=32 L7 nCells=163842 nlev=20 solver=explicit_substep halo=in_step] compile=12579.2ms fused=191.9856ms/step (probe_latency=188.993ms) gate_loop_latency=190.29ms/step
11298-2689-[1785518411.064180] [l20119:1441477:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
11299-2690-[1785518411.073813] [l20119:1441504:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
11300-2691-[1785518411.088174] [l20119:1441503:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
--
11302-4383-[1785518454.796638] [l20119:1442701:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
11303-4384-[1785518454.816638] [l20119:1442698:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
11304-4385-[1785518454.766060] [l20119:1442707:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
11305-4386-[1785518454.774753] [l20119:1442679:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
11306:4387:{"component": "ocean", "grid": "voronoi", "mode": "strong", "subdivision": 8, "n_ranks": 128, "n_cells": 655362, "n_edges": 1966080, "nlev": 20, "partition_method": "metis", "steps": 12, "dt": 60.0, "platform": "cpu", "compile_ms": 13554.5, "steady_median_ms": 333.39, "step_latency_gate_loop_ms": 348.11, "step_latency_gate_loop_min_ms": 345.44, "per_step_ms": [13554.5, 350.4, 352.4, 348.9, 348.3, 349.4, 348.0, 347.5, 347.1, 348.5, 345.4, 347.6], "cells": 13107240, "cells_per_rank_achieved": 5120, "compile_prewarmed_by_parity_ref": false, "barotropic_solver": "explicit_substep", "halo_refresh": "in_step", "stage_halo_correct": true, "stage_halo_note": "per-step packed entry refresh + in-step stage-frontier refreshes (make_mpas_ocean_halo_refresh) \u2014 stage-correct; see docs/performance/scaling/mpas_ocean_distributed_stage_audit.md", "fused": {"compile_ms": 343.0, "scan_compile_ms": 15677.5, "step_latency_ms": 343.721, "block_ms": [2663.79, 2664.88], "parallel_block_ms": [2666.74, 2667.5], "fused_step_ms": 333.39, "rank_imbalance": 1.0007, "rank_imbalance_per_block": [1.0007, 1.0008], "block_steps": 8, "n_blocks": 2, "probe_steps": 3}, "wet_cell": {"wet_cell_levels": 13057600, "wet_cell_levels_per_device": 102012.5, "wet_fraction": 0.9962127801123654, "wet_equals_total": false, "wet_cell_levels_per_device_min": 78000, "wet_cell_levels_per_device_max": 102880}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-31T17:21:47.977228+00:00", "grid": "voronoi", "component": "ocean", "resolution": "L8", "n_levels": 20, "precision": "float64", "decomposition": "cell_partition", "solver_variant": "mpas_ocean_explicit_substep", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 128, "n_gpus": 0, "device_count": 1, "process_count": 1, "devices_per_rank": 1, "cells_per_rank": 102400, "backend": "cpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "mpi4jax", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l20119.lvt.dkrz.de", "slurm_job_id": "26600094", "git_sha": "7151d12a1", "gpu_direct_requested": false, "mpi4jax_cuda_support": false, "gpu_direct_active": false, "host_staged_halo": false, "partition_metrics": {"n_owned_cells": 5113, "n_halo_cells": 562, "owned_halo_ratio": 0.10991590064541365, "n_neighbor_ranks": 4, "messages_per_exchange": 4, "halo_recv_cells": 562, "owned_send_cells": 553, "cells_per_rank_min": 5093, "cells_per_rank_max": 5144, "edge_cut_total": 75748, "max_neighbor_ranks": 8}, "extra": {"partition_method": "metis", "steps": 12, "warmup": 2, "parity_gate": false, "check_conservation": false, "halo_refresh": "in_step", "barotropic_solver": "explicit_substep", "block_steps": 8, "blocks": 2, "probe_steps": 3}}}
11307:4388-[mpas-ocean np=128 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=13554.5ms fused=333.39ms/step (probe_latency=343.721ms) gate_loop_latency=348.11ms/step
11308-4389-[1785518454.797213] [l20119:1442677:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
11309-4390-[1785518454.772658] [l20119:1442708:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
11310-4391-[1785518454.799331] [l20130:51417:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
11311---
--
11337-6251-[1785518522.085039] [l20130:52608:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
11338-6252-[1785518521.847542] [l20119:1443892:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
11339-6253-[1785518521.996105] [l20130:52583:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
11340-6254-[1785518521.884325] [l20119:1443901:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
11341:6255:{"component": "ocean", "grid": "voronoi", "mode": "strong", "subdivision": 8, "n_ranks": 128, "n_cells": 655362, "n_edges": 1966080, "nlev": 20, "partition_method": "metis", "steps": 12, "dt": 60.0, "platform": "cpu", "compile_ms": 14329.9, "steady_median_ms": 537.905, "step_latency_gate_loop_ms": 557.24, "step_latency_gate_loop_min_ms": 553.39, "per_step_ms": [14329.9, 558.5, 561.4, 554.0, 558.4, 555.1, 560.6, 569.8, 559.2, 554.6, 556.1, 553.4], "cells": 13107240, "cells_per_rank_achieved": 5120, "compile_prewarmed_by_parity_ref": false, "barotropic_solver": "explicit_substep", "halo_refresh": "in_step", "stage_halo_correct": true, "stage_halo_note": "per-step packed entry refresh + in-step stage-frontier refreshes (make_mpas_ocean_halo_refresh) \u2014 stage-correct; see docs/performance/scaling/mpas_ocean_distributed_stage_audit.md", "fused": {"compile_ms": 547.5, "scan_compile_ms": 17807.2, "step_latency_ms": 547.997, "block_ms": [4295.08, 4293.81], "parallel_block_ms": [4303.79, 4302.69], "fused_step_ms": 537.905, "rank_imbalance": 1.0021, "rank_imbalance_per_block": [1.002, 1.0023], "block_steps": 8, "n_blocks": 2, "probe_steps": 3}, "wet_cell": {"wet_cell_levels": 13057600, "wet_cell_levels_per_device": 102012.5, "wet_fraction": 0.9962127801123654, "wet_equals_total": false, "wet_cell_levels_per_device_min": 78000, "wet_cell_levels_per_device_max": 102880}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-31T17:23:14.881483+00:00", "grid": "voronoi", "component": "ocean", "resolution": "L8", "n_levels": 20, "precision": "float64", "decomposition": "cell_partition", "solver_variant": "mpas_ocean_explicit_substep", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 128, "n_gpus": 0, "device_count": 1, "process_count": 1, "devices_per_rank": 1, "cells_per_rank": 102400, "backend": "cpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "mpi4jax", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l20119.lvt.dkrz.de", "slurm_job_id": "26600094", "git_sha": "7151d12a1", "gpu_direct_requested": false, "mpi4jax_cuda_support": false, "gpu_direct_active": false, "host_staged_halo": false, "partition_metrics": {"n_owned_cells": 5113, "n_halo_cells": 562, "owned_halo_ratio": 0.10991590064541365, "n_neighbor_ranks": 4, "messages_per_exchange": 4, "halo_recv_cells": 562, "owned_send_cells": 553, "cells_per_rank_min": 5093, "cells_per_rank_max": 5144, "edge_cut_total": 75748, "max_neighbor_ranks": 8}, "extra": {"partition_method": "metis", "steps": 12, "warmup": 2, "parity_gate": false, "check_conservation": false, "halo_refresh": "in_step", "barotropic_solver": "explicit_substep", "block_steps": 8, "blocks": 2, "probe_steps": 3}}}
11342-6256-[mpas-ocean np=128 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=14329.9ms fused=537.905ms/step (probe_latency=547.997ms) gate_loop_latency=557.24ms/step
11343-6257-[1785518521.623596] [l20119:1443880:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
11344-6258-[1785518522.112727] [l20130:52580:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
11345-6259-[1785518521.530617] [l20119:1443887:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
--
12345-6159-[1785518522.148533] [l20131:272729:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
12346---
12347-6253-[1785518521.996105] [l20130:52583:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
12348-6254-[1785518521.884325] [l20119:1443901:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
12349:6255:{"component": "ocean", "grid": "voronoi", "mode": "strong", "subdivision": 8, "n_ranks": 128, "n_cells": 655362, "n_edges": 1966080, "nlev": 20, "partition_method": "metis", "steps": 12, "dt": 60.0, "platform": "cpu", "compile_ms": 14329.9, "steady_median_ms": 537.905, "step_latency_gate_loop_ms": 557.24, "step_latency_gate_loop_min_ms": 553.39, "per_step_ms": [14329.9, 558.5, 561.4, 554.0, 558.4, 555.1, 560.6, 569.8, 559.2, 554.6, 556.1, 553.4], "cells": 13107240, "cells_per_rank_achieved": 5120, "compile_prewarmed_by_parity_ref": false, "barotropic_solver": "explicit_substep", "halo_refresh": "in_step", "stage_halo_correct": true, "stage_halo_note": "per-step packed entry refresh + in-step stage-frontier refreshes (make_mpas_ocean_halo_refresh) \u2014 stage-correct; see docs/performance/scaling/mpas_ocean_distributed_stage_audit.md", "fused": {"compile_ms": 547.5, "scan_compile_ms": 17807.2, "step_latency_ms": 547.997, "block_ms": [4295.08, 4293.81], "parallel_block_ms": [4303.79, 4302.69], "fused_step_ms": 537.905, "rank_imbalance": 1.0021, "rank_imbalance_per_block": [1.002, 1.0023], "block_steps": 8, "n_blocks": 2, "probe_steps": 3}, "wet_cell": {"wet_cell_levels": 13057600, "wet_cell_levels_per_device": 102012.5, "wet_fraction": 0.9962127801123654, "wet_equals_total": false, "wet_cell_levels_per_device_min": 78000, "wet_cell_levels_per_device_max": 102880}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-31T17:23:14.881483+00:00", "grid": "voronoi", "component": "ocean", "resolution": "L8", "n_levels": 20, "precision": "float64", "decomposition": "cell_partition", "solver_variant": "mpas_ocean_explicit_substep", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 128, "n_gpus": 0, "device_count": 1, "process_count": 1, "devices_per_rank": 1, "cells_per_rank": 102400, "backend": "cpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "mpi4jax", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l20119.lvt.dkrz.de", "slurm_job_id": "26600094", "git_sha": "7151d12a1", "gpu_direct_requested": false, "mpi4jax_cuda_support": false, "gpu_direct_active": false, "host_staged_halo": false, "partition_metrics": {"n_owned_cells": 5113, "n_halo_cells": 562, "owned_halo_ratio": 0.10991590064541365, "n_neighbor_ranks": 4, "messages_per_exchange": 4, "halo_recv_cells": 562, "owned_send_cells": 553, "cells_per_rank_min": 5093, "cells_per_rank_max": 5144, "edge_cut_total": 75748, "max_neighbor_ranks": 8}, "extra": {"partition_method": "metis", "steps": 12, "warmup": 2, "parity_gate": false, "check_conservation": false, "halo_refresh": "in_step", "barotropic_solver": "explicit_substep", "block_steps": 8, "blocks": 2, "probe_steps": 3}}}
12350-6256-[mpas-ocean np=128 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=14329.9ms fused=537.905ms/step (probe_latency=547.997ms) gate_loop_latency=557.24ms/step
12351-6257-[1785518521.623596] [l20119:1443880:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
12352---
12353-6287-[1785518521.549753] [l20119:1443903:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
12354-6288-[1785518521.792657] [l20119:1443909:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
12355-6289:=== RESULTS (A-D all hold 5120 cells/rank) ===
12356:6290-A_s7np32_geometric:   189.82 ms  method=geometric ranks=32
12357:6291-B_s8np128_geometric:   308.96 ms  method=geometric ranks=128
12358---
12359:6293-D_s8np128_metis:   333.39 ms  method=metis ranks=128
12360-6294-E_s8np128_metis_blockblock:   537.90 ms  method=metis ranks=128
12361-6295:DONE rc=0
12362-6296-
12363-6297-********************************************************************************

exec
/usr/bin/bash -lc "rg -o '\"cells_per_rank_min\": [0-9]+, \"cells_per_rank_max\": [0-9]+' .physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md | tail -n 30 && printf '\\n--- all job 26600094 JSON identities / partition minmax / wet minmax ---\\n' && /usr/bin/python3 - <<'PY'
import re
p='.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md'
s=open(p, errors='replace').read()
for m in re.finditer(r'\\{\"component\": \"ocean\".*?\"slurm_job_id\": \"26600094\".*?\\}\\}\\}', s):
    rec=m.group()
    print('method=',re.search(r'\"partition_method\": \"(["'^"]+)"'"',rec).group(1),
          'ranks=',re.search(r'\"n_ranks\": (\\d+)',rec).group(1),
          'ms=',re.search(r'\"steady_median_ms\": ([0-9.]+)',rec).group(1),
          'owned=',re.search(r'\"cells_per_rank_min\": (\\d+), \"cells_per_rank_max\": (\\d+)',rec).groups(),
          'wet=',re.search(r'\"wet_cell_levels_per_device_min\": (\\d+), \"wet_cell_levels_per_device_max\": (\\d+)',rec).groups())
PY" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 exited 1 in 231ms:
"cells_per_rank_min": 1280, "cells_per_rank_max": 1281
"cells_per_rank_min": 10240, "cells_per_rank_max": 10241
"cells_per_rank_min": 2560, "cells_per_rank_max": 2561
"cells_per_rank_min": 1280, "cells_per_rank_max": 1281
"cells_per_rank_min": 5120, "cells_per_rank_max": 5121
"cells_per_rank_min": 5120, "cells_per_rank_max": 5121
"cells_per_rank_min": 5100, "cells_per_rank_max": 5145
"cells_per_rank_min": 5093, "cells_per_rank_max": 5144
"cells_per_rank_min": 5093, "cells_per_rank_max": 5144
"cells_per_rank_min": 20480, "cells_per_rank_max": 20481
"cells_per_rank_min": 10240, "cells_per_rank_max": 10241
"cells_per_rank_min": 5120, "cells_per_rank_max": 5121
"cells_per_rank_min": 2560, "cells_per_rank_max": 2561
"cells_per_rank_min": 5120, "cells_per_rank_max": 5121
"cells_per_rank_min": 5120, "cells_per_rank_max": 5121
"cells_per_rank_min": 5120, "cells_per_rank_max": 5121
"cells_per_rank_min": 5100, "cells_per_rank_max": 5145
"cells_per_rank_min": 5093, "cells_per_rank_max": 5144
"cells_per_rank_min": 5093, "cells_per_rank_max": 5144
"cells_per_rank_min": 5120, "cells_per_rank_max": 5121
"cells_per_rank_min": 5120, "cells_per_rank_max": 5121
"cells_per_rank_min": 5100, "cells_per_rank_max": 5145
"cells_per_rank_min": 5093, "cells_per_rank_max": 5144
"cells_per_rank_min": 5093, "cells_per_rank_max": 5144
"cells_per_rank_min": 5120, "cells_per_rank_max": 5121
"cells_per_rank_min": 5120, "cells_per_rank_max": 5121
"cells_per_rank_min": 5100, "cells_per_rank_max": 5145
"cells_per_rank_min": 5093, "cells_per_rank_max": 5144
"cells_per_rank_min": 5093, "cells_per_rank_max": 5144
"cells_per_rank_min": 5093, "cells_per_rank_max": 5144

--- all job 26600094 JSON identities / partition minmax / wet minmax ---
/usr/bin/bash: cannot create temp file for here-document: Read-only file system

exec
/usr/bin/bash -lc "sed -n '344,640p' scripts/bench/bench_mpas_spmd_scaling.py && printf '\\n--- ocean cells field provenance ---\\n' && sed -n '700,770p' scripts/bench/bench_ocean_mpas_scaling.py && printf '\\n--- MPAS output validity fields ---\\n' && rg -n -C 4 'valid=|\"valid\"|steady_median_ms|mcells_per_s|out_path|args.out|Path\\(args.out' scripts/bench/bench_mpas_spmd_scaling.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 234ms:
    physics_fn = None
    if args.physics == "held_suarez":
        from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_forcing_mpas
        physics_fn = held_suarez_forcing_mpas
    elif args.physics == "kessler":
        # Warm-rain microphysics over the moist BCW tracers.  Kessler's
        # saturation adjustment is a rate over the dt bound HERE, so it
        # must match the stepping dt (make_kessler_forcing_mpas contract).
        from legoesm.atmosphere.forcing.idealized.kessler_forcing import (
            make_kessler_forcing_mpas,
        )
        physics_fn = make_kessler_forcing_mpas(dt)

    # #1100: s0 from build_model_and_state is ALREADY partition-local-sharded
    # when n_devices > 1 (no per-process global build on the timed path).
    # The parity/conservation gates are the ONLY consumers of a global
    # initial state — build it lazily here, at their smoke scales only
    # (deterministic identical build on every process; bit-identical to the
    # sharded s0 per tests/parallel/test_mpas_partitionlocal_build.py).
    s0_global = None
    if args.parity_gate or args.check_conservation:
        from tests.test_cases.baroclinic_wave import baroclinic_wave_init_mpas
        s0_global = (s0 if dev_config.n_devices <= 1
                     else baroclinic_wave_init_mpas(
                         mesh, model.sigma_coord, perturbed=True,
                         moist=(args.physics == "kessler")))

    # Parity reference: the plain single-device trajectory on the SAME
    # reordered mesh.  model.step's signature is call-compatible.
    serial_final = None
    if args.parity_gate:
        from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
            MPASPrimitiveEquationModel,
        )
        ref_model = (model if dev_config.n_devices <= 1
                     else MPASPrimitiveEquationModel(
                         mesh, model.sigma_coord, model.config))
        _s = s0_global
        for _ in range(args.steps):
            _s = ref_model.step(_s, dt, physics_fn=physics_fn)
        _block(_s)
        serial_final = _s

    mass_before = None
    if args.check_conservation:
        mass_before = _global_dry_mass(s0_global, mesh)

    step = make_voronoi_sharded_step(
        model, dev_config, halo_strategy=args.halo_strategy)
    # Already in the sharded layout (partition-local build) for nd > 1;
    # single-device s0 is the plain global state.
    s = s0

    # Multi-controller: align every process around the timed loop.
    if jax.process_count() > 1:
        from jax.experimental import multihost_utils
        multihost_utils.sync_global_devices("mpas_spmd_bench_start")

    # Per-step timing: step 0 includes compile; record each step so re-trace
    # (every step slow) is visible vs steady-state (steps 1.. fast).
    per_step_ms = []
    for _ in range(args.steps):
        t0 = time.perf_counter()
        if physics_fn is not None:
            s = step(s, dt, physics_fn=physics_fn)
        else:
            s = step(s, dt)
        _block(s)
        per_step_ms.append((time.perf_counter() - t0) * 1e3)

    if jax.process_count() > 1:
        from jax.experimental import multihost_utils
        multihost_utils.sync_global_devices("mpas_spmd_bench_end")

    # HLO collective-permute census (#1113 ask 2): a STATIC compile property of
    # the sharded step — the ppermute ROUND count that decomposes multi-node
    # overhead (overhead ~= CPs/step * ~0.11 ms launch floor). The cube benches
    # record this; the MPAS row did not, forcing an out-of-band census. Counted
    # AFTER the timed loop so the census compile can't perturb per_step_ms[0]'s
    # compile timing (the executable is already cached — this re-lower/compile
    # is a cache hit; the count is data-independent, static in the partition).
    # Best-effort (None if compilation is unsupported); the serial n=1 leg has
    # no ppermute halo -> 0.
    if physics_fn is not None:
        _census_fn = lambda st: step(st, dt, physics_fn=physics_fn)  # noqa: E731
    else:
        _census_fn = lambda st: step(st, dt)  # noqa: E731
    # ONE compile → full per-family census; the CP scalar (the #1113 round-count
    # wall) is the collective_permute member, so no second compile for it.
    hlo_census = hlo_collective_census(_census_fn, s)
    hlo_cp = hlo_census["collective_permute"] if hlo_census else None

    # --- Correctness gates (before any timing is reported) -----------------
    if args.parity_gate or args.check_conservation:
        final_global = (gather_voronoi_state_spmd(s, dev_config)
                        if dev_config.n_devices > 1 else s)
        prec = "float64" if jax.config.jax_enable_x64 else "float32"
        rank0 = jax.process_index() == 0
        if args.check_conservation:
            mass_after = _global_dry_mass(final_global, mesh)
            tol = (args.mass_rtol if args.mass_rtol is not None
                   else MASS_RTOL_DEFAULTS[prec])
            rel = abs(mass_after - mass_before) / abs(mass_before)
            if rank0:
                print(f"    conservation dry-mass: rel drift={rel:.3e} "
                      f"(tol {tol:.1e}) over {args.steps} steps", flush=True)
            if rel > tol:
                if rank0:
                    print("ERROR: conservation gate BREACHED.", flush=True)
                return 4
        if args.parity_gate:
            tols = MPAS_PARITY_TOLS[prec]
            ok = True
            checks = [
                (name, getattr(serial_final, name).data,
                 getattr(final_global, name).data, rtol, atol)
                for name, (rtol, atol) in tols.items()
            ]
            if serial_final.tracers is not None:
                # Moist run: the tracer fields ride the packed exchange +
                # RK advection — gate them too (q re-association floor is
                # far below the q_v scale; reuse the T tolerances).
                q_rtol, q_atol = tols["T"]
                if set(final_global.tracers or {}) != set(
                        serial_final.tracers):
                    if rank0:
                        print("ERROR: sharded run dropped tracer fields.",
                              flush=True)
                    return 5
                checks += [
                    (k, serial_final.tracers[k].data,
                     final_global.tracers[k].data, q_rtol, q_atol * 1e-3)
                    for k in sorted(serial_final.tracers)
                ]
            for name, want, got, rtol, atol in checks:
                want = np.asarray(want)
                got = np.asarray(got)
                field_ok = bool(np.allclose(got, want, rtol=rtol, atol=atol))
                ok &= field_ok
                if rank0:
                    mx = (float(np.max(np.abs(got - want)))
                          if want.size else 0.0)
                    print(f"    parity {name:>4s}: max|diff|={mx:.3e} "
                          f"{'OK' if field_ok else 'MISMATCH'}", flush=True)
            if not ok:
                if rank0:
                    print("ERROR: SPMD parity gate MISMATCH vs the "
                          "single-device reference.", flush=True)
                return 5

    steady = per_step_ms[args.warmup:]
    med = float(np.median(steady))
    rec = dict(
        component="mpas_atm",
        subdivision=args.subdivision, n_devices=nd,
        n_cells=int(mesh.nCells), n_edges=int(mesh.nEdges), nlev=args.nlev,
        partition_method=args.partition_method, physics=args.physics,
        # lloyd=0 is the LABELLED synthetic scaling mesh — anti-masquerade:
        # a row without this field could pass as a production-SCVT receipt.
        lloyd_iterations=args.lloyd,
        # Requested vs EFFECTIVE (post-"auto") strategy — a JSONL row
        # saying "auto" would not reveal whether ppermute or allgather
        # was actually measured (codex M3c-2 MINOR).
        halo_strategy_requested=args.halo_strategy,
        halo_strategy_effective=getattr(
            step, "_halo_strategy_effective", "serial"),
        steps=args.steps, dt=dt,
        platform=jax.default_backend(),
        n_processes=jax.process_count(),
        multicontroller=bool(args.multicontroller),
        compile_ms=round(per_step_ms[0], 1),
        steady_median_ms=round(med, 2),
        steady_min_ms=round(float(np.min(steady)), 2),
        per_step_ms=[round(x, 1) for x in per_step_ms],
        cells=int(mesh.nCells) * args.nlev,
        # ppermute round count/step (static compile property; #1113) — the
        # multi-node ceiling is this count x the ~0.11 ms launch floor, so it
        # belongs on every row like the cube benches.
        hlo_collective_permutes=hlo_cp,
        # full per-family census (permute + all-reduce + all-gather + ...) on
        # the SAME compile: exposes any reduction the ico step introduces.
        hlo_collectives=hlo_census,
    )
    # Flat aggregator-compatible identity + metric fields (see the latlon
    # twin): resolution = subdivision level, matching run_cpu_mpi_scaling's
    # icosahedral convention so both lanes land on the same plot curves.
    rec.update(
        grid_type="icosahedral",
        resolution=args.subdivision,
        n_levels=args.nlev,
        mode="strong",  # this bench fixes the mesh and sweeps devices
        precision="float64" if jax.config.jax_enable_x64 else "float32",
        physics_level=args.physics,
        backend=jax.default_backend(),
        **tidy_throughput_fields(
            dt_seconds=dt, time_per_step_ms=med,
            total_cells=int(mesh.nCells) * args.nlev),
    )
    rec["metadata"] = annotate_incomplete(scaling_metadata(
        grid="icosahedral",
        component="atmosphere",
        resolution=f"L{args.subdivision}",
        n_levels=args.nlev,
        precision="float64" if jax.config.jax_enable_x64 else "float32",
        n_gpus=(nd if jax.default_backend() in ("gpu", "cuda", "rocm")
                else 0),
        decomposition="cell_partition" if nd > 1 else "none",
        # cells_per_rank is per PROCESS (n_ranks semantics); the per-device
        # share lives in extra.cells_per_device — a single-process 4-device
        # SPMD run has 1 rank owning ALL cells (codex finding 3).
        cells_per_rank=int(mesh.nCells) * args.nlev
        // max(jax.process_count(), 1),
        scaling_kind="strong",  # this bench fixes the mesh and sweeps devices
        extra={
            "partition_method": args.partition_method,
            "physics": args.physics,
            "steps": args.steps,
            "multicontroller": bool(args.multicontroller),
            "cells_per_device": int(mesh.nCells) // nd * args.nlev,
        },
    ))
    # Multi-controller: every process times the same program; process 0 owns
    # the JSONL + stdout (others would duplicate/corrupt the append).
    if jax.process_index() == 0:
        _outdir = os.path.dirname(args.out)
        if _outdir:  # a bare basename --out needs no mkdir
            os.makedirs(_outdir, exist_ok=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec))
        print(f"[mpas nd={nd} L{args.subdivision} nCells={mesh.nCells} "
              f"nlev={args.nlev}] compile={rec['compile_ms']}ms "
              f"steady_median={med:.2f}ms/step "
              f"(per-step: {rec['per_step_ms']})")
        if rec["metadata"]["virtual_cpu_devices"]:
            print("[virtual-cpu] forced host-platform CPU devices: this row "
                  "is a communication-overhead / correctness proxy, NOT "
                  "hardware scaling — do not report it as a speedup.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

--- ocean cells field provenance ---
            # BEFORE its in-loop cell exchange — a residual assembled
            # from rotten halos is not solver-health evidence.  One
            # packed refresh here (collective; OUTSIDE every timed loop)
            # so the probe measures a cleanly-assembled system (codex
            # finding 2).  Under per_step the last advance() already
            # ended with this exact exchange, so the probe input is
            # fresh on every path.
            from legoesm.parallel.voronoi_mpi import (
                exchange_state_mpas_ocean,
            )
            state = exchange_state_mpas_ocean(state, _layout)
        _probe_out = barotropic_implicit_mpas(
            state, model.mesh, z_coord, config, args.dt,
            return_residual=True)
        zero_forcing_probe_residual = float(
            jax.block_until_ready(_probe_out[3]))
        zero_forcing_probe_measured = True
    else:
        solver_iters = None
        solver_iters_mode = "explicit_substep (no iterative solve)"
        residual_reason = ("explicit_substep barotropic has no iterative "
                           "solve — no solver residual exists to measure")

    steady = per_step_ms[args.warmup:]
    gate_loop_med = float(np.median(steady))
    # M1 headline contract (mirrors bench_ocean_latlon_spmd_scaling): the
    # aggregator-facing ``steady_median_ms`` carries the FUSED per-step
    # number (cross-rank MAX-reduced per block); the host-synced gate-loop
    # median measures dispatch+sync latency and is demoted to the
    # explicitly-named ``step_latency_gate_loop_ms``.  Honest null when
    # the fused measurement is disabled (--block-steps 0) — a latency
    # number must never masquerade as fused throughput (codex finding 1).
    fused_step_ms = fused.get("fused_step_ms") if fused is not None else None
    rec = dict(
        component="ocean",
        grid="voronoi",
        mode=args.mode, subdivision=subdivision, n_ranks=n_ranks,
        n_cells=int(mesh.nCells), n_edges=int(mesh.nEdges), nlev=args.nlev,
        partition_method=args.partition_method,
        steps=args.steps, dt=args.dt,
        platform=jax.default_backend(),
        compile_ms=round(per_step_ms[0], 1),
        steady_median_ms=(round(fused_step_ms, 4)
                          if fused_step_ms is not None else None),
        step_latency_gate_loop_ms=round(gate_loop_med, 2),
        step_latency_gate_loop_min_ms=round(float(np.min(steady)), 2),
        per_step_ms=[round(x, 1) for x in per_step_ms],
        cells=int(mesh.nCells) * args.nlev,
        # HORIZONTAL cells/rank — the same unit as --cells-per-rank, so a
        # weak-mode row is comparable to its target (codex: the 3-D count
        # made rows look nlev-x larger).
        cells_per_rank_achieved=int(mesh.nCells) // n_ranks,
        # Fix 4 honesty flag: at np=1 the parity reference pre-runs the
        # SAME shape before the timed loop, so per_step_ms[0] may not
        # contain the real JIT compile.
        compile_prewarmed_by_parity_ref=bool(
            args.parity_gate and n_ranks == 1),
        # --- M1 lane fields (scaling-M3d increment-1) ---
        barotropic_solver=args.barotropic_solver,
        halo_refresh=halo_refresh,
        # 'in_step' rows run the stage-frontier refreshes inside the step
        # (audited insertion points R1-R3/T1-T2/B0-B2/I0-I1) on top of the
        # per-step entry exchange -> stage-correct.  'per_step'/'none'
        # multi-rank rows keep within-step staleness and stay FALSE.
        # Single-rank rows have no partition, hence trivially true.
        stage_halo_correct=bool(n_ranks == 1 or halo_refresh == "in_step"),
        stage_halo_note=stage_halo_note_for(n_ranks, halo_refresh),
        fused=fused,
        wet_cell=wet_rec,
        solver_iters=solver_iters,
        solver_iters_mode=solver_iters_mode,

--- MPAS output validity fields ---
511-        platform=jax.default_backend(),
512-        n_processes=jax.process_count(),
513-        multicontroller=bool(args.multicontroller),
514-        compile_ms=round(per_step_ms[0], 1),
515:        steady_median_ms=round(med, 2),
516-        steady_min_ms=round(float(np.min(steady)), 2),
517-        per_step_ms=[round(x, 1) for x in per_step_ms],
518-        cells=int(mesh.nCells) * args.nlev,
519-        # ppermute round count/step (static compile property; #1113) — the
--
564-    ))
565-    # Multi-controller: every process times the same program; process 0 owns
566-    # the JSONL + stdout (others would duplicate/corrupt the append).
567-    if jax.process_index() == 0:
568:        _outdir = os.path.dirname(args.out)
569-        if _outdir:  # a bare basename --out needs no mkdir
570-            os.makedirs(_outdir, exist_ok=True)
571:        with open(args.out, "a") as f:
572-            f.write(json.dumps(rec) + "\n")
573-        print(json.dumps(rec))
574-        print(f"[mpas nd={nd} L{args.subdivision} nCells={mesh.nCells} "
575-              f"nlev={args.nlev}] compile={rec['compile_ms']}ms "

exec
/usr/bin/bash -lc "rg -n -C 5 'SLURM_STEP_NODELIST|coordinator|SLURM_JOB_ID|port|initialize' packages/core/legoesm/parallel/early_init.py && printf '\\n--- srun docs local available ---\\n' && (srun --help 2>&1 | rg -n -C 2 -- '--exact|exclusive|gpus-per-node|gpu-bind' || true) && printf '\\n--- shell static syntax ---\\n' && bash -n scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch scripts/cluster/scaling_levante/atm_latlon2d_cpu_hundreds.sbatch scripts/cluster/scaling_levante/prewarm_s10.sbatch && echo bash-syntax-ok && printf '\\n--- py compile ---\\n' && /usr/bin/python3 -m py_compile scripts/plot/plot_scaling_paper_figure.py && echo pycompile-ok" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 exited 1 in 513ms:
1-"""Early JAX distributed initialization.
2-
3:Must be called before any legoESM or jax.numpy import that triggers XLA
4:backend discovery.  Importing this module is cheap: it only touches stdlib
5-and optionally mpi4py + jax (base package), never jax.numpy or legoESM.
6-
7:Usage (at the very top of an entry-point script, before all other imports)::
8-
9:    from legoesm.parallel.early_init import maybe_init_jax_distributed
10-    maybe_init_jax_distributed()
11-
12:    # Safe to import legoESM / jax.numpy from here onward.
13-"""
14-
15:from __future__ import annotations
16-
17:import os
18:import socket
19:import zlib
20-
21-# Idempotency flag — must NOT be a jax.process_count() probe: that call
22:# initialises the XLA backend, after which jax.distributed.initialize() raises
23-# "must be called before any JAX calls that might initialise the XLA backend"
24-# (issue #693: every multi-node run died here).
25-_INITIALIZED = False
26-
27:# Legacy fixed coordinator port, kept as the last-resort fallback when no
28-# scheduler job id is present (matches the historical hardcoded value so
29-# launcher-less local runs keep working unchanged).
30-_LEGACY_COORDINATOR_PORT = 1234
31-
32-
33:def resolve_coordinator_port(default: int = _LEGACY_COORDINATOR_PORT) -> int:
34:    """Deterministic ``jax.distributed`` coordinator port for THIS job.
35-
36-    Precedence:
37-
38-    1. ``LEGOESM_COORDINATOR_PORT`` (explicit override);
39:    2. derived from the scheduler job id (``SLURM_JOB_ID`` / ``PBS_JOBID``)
40:       via crc32 into the dynamic-port range — every rank of one job
41:       computes the SAME port with no communication, while two different
42:       jobs sharing a node get different ports (a fixed port means the
43:       second job's coordinator dies with EADDRINUSE on shared clusters);
44:    3. ``default`` (the legacy fixed port) when no scheduler id exists.
45-
46-    crc32 (not ``hash()``) because Python string hashing is randomized
47-    per process — ranks would disagree.
48-    """
49-    env = os.environ.get("LEGOESM_COORDINATOR_PORT")
50-    if env:
51-        return int(env)
52:    jobid = os.environ.get("SLURM_JOB_ID") or os.environ.get("PBS_JOBID")
53-    if jobid:
54-        # 20000 + [0, 40000): stays inside the unprivileged range and clear
55:        # of the ephemeral-port ceiling on common Linux configs.
56-        return 20000 + zlib.crc32(jobid.encode()) % 40000
57-    return default
58-
59-
60-def launcher_world_size() -> int:
61-    """World size DECLARED by the job launcher's environment (0 = none).
62-
63:    Reads the union of the launcher families every entry point supports:
64-    SLURM step (``SLURM_STEP_NUM_TASKS``), Open MPI
65-    (``OMPI_COMM_WORLD_SIZE``), PMI/PALS (``PMI_SIZE``), then the
66-    allocation-wide ``SLURM_NTASKS`` last.  Used by the post-init fallback
67-    guard — the launcher's declaration is the ground truth a federated
68-    runtime must match.
69-    """
70-    # Precedence = closeness to THIS process's launcher: the srun STEP
71-    # size, then the MPI launcher's own world (mpiexec inside a SLURM
72:    # allocation exports OMPI/PMI sizes — the truth), and only then the
73-    # allocation-wide SLURM_NTASKS (weakest: it describes the allocation,
74-    # not necessarily this launch; codex).
75-    for var in ("SLURM_STEP_NUM_TASKS", "OMPI_COMM_WORLD_SIZE",
76-                "PMI_SIZE", "SLURM_NTASKS"):
77-        v = os.environ.get(var)
--
82-
83-def check_no_silent_process_fallback() -> None:
84-    """Fail LOUDLY when the federated process count disagrees with the
85-    launcher's declared world size.
86-
87:    The route-B hazard this guards: ``jax.distributed.initialize`` (or an
88-    auto-detect miss) silently federates FEWER processes than the launcher
89-    started — N un-federated copies then run the same program, clobber each
90-    other's output, and a bench records a fake single-process row as an
91-    N-rank result.  Mirror of the route-A ``check_no_silent_mpi_fallback``
92-    (runtime.py); same override env for emergencies:
93-    ``LEGOESM_ALLOW_SINGLE_PROCESS_UNDER_MPI=1``.
94-
95:    Call AFTER ``jax.distributed.initialize`` — ``jax.process_count()`` is
96-    then safe (the backend is already federated).
97-    """
98-    if os.environ.get("LEGOESM_ALLOW_SINGLE_PROCESS_UNDER_MPI") == "1":
99-        return
100-    declared = launcher_world_size()
101-    if declared <= 1:
102-        return
103:    import jax
104-
105-    actual = int(jax.process_count())
106-    if actual != declared:
107-        raise RuntimeError(
108-            f"jax.distributed federated {actual} process(es) but the "
109-            f"launcher declared {declared} (SLURM_NTASKS / "
110-            f"OMPI_COMM_WORLD_SIZE / PMI_SIZE): a silent fallback would run "
111-            f"{declared} un-federated copies and record fake scaling rows. "
112:            f"Fix the launch (coordinator/port/env) or set "
113-            f"LEGOESM_ALLOW_SINGLE_PROCESS_UNDER_MPI=1 to override.")
114-
115-
116-def _launcher_local_rank() -> tuple[str, str] | None:
117-    """(env var, value) of the launcher's NODE-LOCAL rank, or ``None``.
118-
119-    Union of the launcher families the init paths serve: Cray PALS,
120-    Open MPI, MVAPICH, and SLURM (SLURM_LOCALID only on a genuine
121:    multi-task launch — on a single-task sbatch step it is exported too
122-    and pinning on it would hide all but GPU 0 from a single-process
123-    multi-GPU run, the documented silent eff=0.5 bug).
124-    """
125-    for var in ("PALS_LOCAL_RANKID", "OMPI_COMM_WORLD_LOCAL_RANK",
126-                "MV2_COMM_WORLD_LOCAL_RANK"):
--
165-                f"(one rank per GPU).")
166-        return [idx]
167-    return [0]
168-
169-
170:def nccl_transport_report() -> dict:
171:    """Best-effort NCCL transport facts for run metadata (route-B analog of
172-    the mpi4jax GPU-direct preflight).
173-
174:    NCCL has no Python-queryable transport API; what IS knowable up front:
175-    the fabric env knobs and whether an OFI/net plugin library
176-    (``libnccl-net*``) is discoverable on ``LD_LIBRARY_PATH``/``LD_PRELOAD``
177-    / ``NCCL_NET_PLUGIN``.  ``missing_net_plugin_multi_node`` records the
178-    FACT of a multi-node launch with no net plugin visible — on OFI fabrics
179-    (Derecho Slingshot) that means NCCL silently runs correct-but-slow TCP
--
181-    socket-bound'); native-IB fabrics run fine without a plugin, which is
182-    why the field states the fact, not the inference.  Advisory (the
183-    definitive check stays ``NCCL_DEBUG=INFO`` in the job log); recorded so
184-    a socket-bound row is falsifiable from the record.
185-    """
186:    import glob
187-
188-    plugin_hit = None
189-    if os.environ.get("NCCL_NET_PLUGIN"):
190-        plugin_hit = os.environ["NCCL_NET_PLUGIN"]
191-    else:
--
227-            multi_node and plugin_hit is None),
228-    }
229-
230-
231-def init_jax_distributed_with_fallback() -> None:
232:    """``jax.distributed.initialize()`` with a PBS/PALS-safe fallback.
233-
234:    Bare ``initialize()`` auto-detects SLURM / Open MPI (mpirun) / Cloud
235-    TPU only.  Under PBS + Cray PALS (Derecho ``mpiexec``) nothing is
236:    detected and it raises — so the transport is chosen by ENVIRONMENT,
237-    not by parsing exception text (real failures can mention
238-    "cluster"/"detect" too):
239-
240:    - SLURM / Open MPI env present: bare ``initialize()`` (its cluster
241-      auto-detection also derives ``local_device_ids`` from the launcher's
242-      local-rank variable); any failure re-raises loudly.
243-    - PALS/PMI-only env (Derecho ``mpiexec``): the mpi4py bootstrap
244-      (``cluster_detection_method="mpi4py"``, the documented ALCF Cray-EX
245-      recipe) with ``local_device_ids=[0]`` — the repo's PALS job shims pin
246-      ``CUDA_VISIBLE_DEVICES`` to ONE device per rank, so local index 0 is
247-      the pinned GPU (the #693 device-binding convention).  Plain-MPI
248-      bootstrap only; mpi4jax is never armed here, so the
249-      jax.distributed-vs-mpi4jax mixed-stack hazard does not apply.
250-
251:    No-op if a previous call already initialized the runtime (shared
252-    ``_INITIALIZED`` flag with :func:`maybe_init_jax_distributed`; the
253:    "already initialized" RuntimeError from an out-of-band init is also
254-    treated as a no-op).
255-    """
256-    global _INITIALIZED
257-    if _INITIALIZED:
258-        return
259-
260:    import importlib.util
261-
262:    import jax
263-
264-    # Cross-path idempotency: an OUTER bootstrap
265:    # (initialize_jax_distributed_multiprocess / a launcher script) may have
266-    # federated the processes without setting THIS module's flag.
267:    # is_initialized() is the supported check (#749).  Still verify the
268-    # OUTER federation against the launcher's declared world size — a
269:    # pre-initialized 1-process runtime under N launcher ranks is the same
270-    # silent-fallback hazard (codex).
271:    if jax.distributed.is_initialized():
272-        _INITIALIZED = True
273-        check_no_silent_process_fallback()
274-        return
275-
276-    auto_detectable = any(
277:        v in os.environ for v in ("SLURM_JOB_ID", "OMPI_COMM_WORLD_SIZE")
278-    )
279-    pals_only = not auto_detectable and (
280-        "PALS_RANKID" in os.environ or "PMI_RANK" in os.environ
281-    )
282:    if pals_only and importlib.util.find_spec("mpi4py") is not None:
283:        jax.distributed.initialize(
284-            cluster_detection_method="mpi4py",
285-            # Shim-pinned CUDA_VISIBLE_DEVICES -> [0]; unpinned bare
286-            # mpiexec -> index by PALS_LOCAL_RANKID so node-sharing ranks
287-            # bind DIFFERENT GPUs (contended-GPU-0 hazard).
288-            local_device_ids=_pals_local_device_ids(),
289-        )
290-        _INITIALIZED = True
291-        check_no_silent_process_fallback()
292-        return
293-    try:
294:        jax.distributed.initialize()
295-    except RuntimeError as e:
296-        if "already" in str(e).lower():
297-            _INITIALIZED = True
298-            check_no_silent_process_fallback()
299-            return
--
305-def _warn_missing_nccl_plugin(rank: int | None = None) -> None:
306-    """Rank-0 warning when a multi-node launch has no NCCL net plugin
307-    visible — cross-node collectives then likely run correct-but-slow TCP
308-    sockets (the documented Derecho shape).  ``rank=None`` derives the rank
309-    from the federated runtime (safe post-init)."""
310:    report = nccl_transport_report()
311:    if not report["missing_net_plugin_multi_node"]:
312-        return
313-    if rank is None:
314-        try:
315:            import jax
316-
317-            rank = int(jax.process_index())
318-        except Exception:
319-            rank = 0
320-    if rank == 0:
--
325-            "slow — the documented Derecho socket-bound shape). Verify "
326-            "with NCCL_DEBUG=INFO; build/load the aws-ofi-nccl plugin "
327-            "for fabric speed.", flush=True)
328-
329-
330:def init_multicontroller_distributed(coordinator: str | None = None) -> None:
331-    """Initialize ``jax.distributed`` for a route-B multicontroller launch.
332-
333-    Shared by every ``--multicontroller`` entry point (the ocean/atm SPMD
334-    benches and the ``run_omip`` route-B driver) so the launcher-env contract
335-    lives in ONE place.  MUST run before any other JAX use (backend init).
336-
337:    - Explicit ``coordinator`` (``host:port``): read the launcher rank/size
338-      from Open MPI ``OMPI_COMM_WORLD_SIZE``/``RANK`` or Cray PALS
339:      ``PMI_SIZE``/``PMI_RANK`` and call ``jax.distributed.initialize``
340-      directly (the mpiexec path; also how the self-spawn tests inject rank).
341:    - No ``coordinator``: delegate to :func:`init_jax_distributed_with_fallback`
342-      (SLURM/OMPI auto-detect or the PALS mpi4py bootstrap).
343-
344-    Real init failures re-raise loudly — a missing launcher rank env is a
345-    hard ``SystemExit``, never a silent single-process fallback (that would
346-    run N identical un-federated copies clobbering each other's output).
347-
348-    Idempotent: a no-op if the federation is already up (another entry point —
349:    e.g. run_amip's import-time ``maybe_init_jax_distributed`` on a real MPI
350:    launch — may have initialized first; a second ``jax.distributed.initialize``
351:    would raise "already initialized").
352-    """
353-    global _INITIALIZED
354-    if _INITIALIZED:
355-        return
356:    if coordinator is None:
357-        init_jax_distributed_with_fallback()
358-        _warn_missing_nccl_plugin()
359-        return
360-
361:    import jax
362-
363:    # Cross-path idempotency on the explicit-coordinator path too: an
364-    # out-of-band bootstrap may have federated already — a second
365:    # initialize() raises (codex).  Still verify the federation size.
366:    if jax.distributed.is_initialized():
367-        _INITIALIZED = True
368-        check_no_silent_process_fallback()
369-        return
370-
371-    n_procs = int(os.environ.get(
372-        "OMPI_COMM_WORLD_SIZE", os.environ.get("PMI_SIZE", "0")))
373-    proc_id = int(os.environ.get(
374-        "OMPI_COMM_WORLD_RANK", os.environ.get("PMI_RANK", "-1")))
375-    if n_procs < 1 or proc_id < 0:
376-        raise SystemExit(
377:            "--coordinator given but no launcher rank env found "
378-            "(OMPI_COMM_WORLD_SIZE/RANK or PMI_SIZE/PMI_RANK).")
379:    jax.distributed.initialize(
380:        coordinator_address=coordinator,
381-        num_processes=n_procs, process_id=proc_id,
382-        # Same per-rank device binding as the PALS bootstrap path: a bare
383-        # multi-device CUDA_VISIBLE_DEVICES must not bind every local rank
384-        # to GPU 0 (codex).
385-        local_device_ids=_pals_local_device_ids())
386-    _INITIALIZED = True
387-    check_no_silent_process_fallback()
388-    _warn_missing_nccl_plugin(rank=proc_id)
389-
390-
391:def maybe_init_jax_distributed(coordinator_port: int | None = None) -> bool:
392-    """Initialize ``jax.distributed`` if running under multi-node MPI.
393-
394-    Detects the MPI world size from the environment (SLURM_NTASKS /
395-    PMI_SIZE / OMPI_COMM_WORLD_SIZE).  When >1 rank and the ranks span
396:    more than one hostname, calls ``jax.distributed.initialize()`` with
397:    rank-0's hostname as coordinator.  On single-node MPI or serial runs,
398-    does nothing.
399-
400:    Returns True if ``jax.distributed.initialize()`` was called, False
401-    otherwise.  Safe to call multiple times — idempotency is tracked via a
402-    module-level flag (NOT a ``jax.process_count()`` probe, which would
403:    initialise the XLA backend and then make ``initialize()`` raise; #693).
404-
405:    ``coordinator_port=None`` (default) resolves the port via
406:    :func:`resolve_coordinator_port` (env override / job-id-derived /
407-    legacy 1234).
408-    """
409-    global _INITIALIZED
410-    if _INITIALIZED:
411-        return False
--
415-    # (codex).
416-    ntasks = launcher_world_size()
417-    if ntasks <= 1:
418-        return False
419-
420:    from mpi4py import MPI
421-    comm = MPI.COMM_WORLD
422-    rank = comm.Get_rank()
423-    size = comm.Get_size()
424-
425-    hosts = comm.allgather(socket.gethostname())
426-    if len(set(hosts)) <= 1:
427-        # Single-node MPI: JAX distributed not needed.
428-        return False
429-
430:    import jax
431:    if coordinator_port is None:
432:        coordinator_port = resolve_coordinator_port()
433:    coordinator = f"{hosts[0]}:{coordinator_port}"
434-    # Per-rank device binding via the shared launcher-family helper: a
435-    # shim-pinned CUDA_VISIBLE_DEVICES (the SLURM --gpu-bind=single:1
436-    # standard, #693) resolves to [0] exactly as before; an UNPINNED or
437-    # multi-device visible list indexes by the launcher's node-local rank
438-    # (guarded SLURM_LOCALID / OMPI / PALS) instead of piling every local
439-    # rank onto GPU 0 (codex).
440:    jax.distributed.initialize(
441:        coordinator_address=coordinator,
442-        num_processes=size,
443-        process_id=rank,
444-        local_device_ids=_pals_local_device_ids(),
445-    )
446-    _INITIALIZED = True

--- srun docs local available ---
117-
118-Consumable resources related options:
119:      --exact                 use only the resources requested for the step
120-                              (by default, all non-gres resources on each node
121-                              in the allocation will be used in the step)
122:      --exclusive[=user]      for job allocation, this allocates nodes in
123:                              in exclusive mode
124:                              for job steps, this is equivalent to --exact
125:      --exclusive[=mcs]       allocate nodes in exclusive mode when
126-                              cpu consumable resource is enabled
127:                              and mcs plugin is enabled (--exact implied)
128-                              or don't share CPUs for job steps
129-      --mem-per-cpu=MB        maximum amount of real memory per allocated
--
156-      --cpus-per-gpu=n        number of CPUs required per allocated GPU
157-  -G, --gpus=n                count of GPUs required for the job
158:      --gpu-bind=...          task to gpu binding options
159-      --gpu-freq=...          frequency and voltage of GPUs
160:      --gpus-per-node=n       number of GPUs required per allocated node
161-      --gpus-per-socket=n     number of GPUs required per allocated socket
162-      --gpus-per-task=n       number of GPUs required per spawned task

--- shell static syntax ---
bash-syntax-ok

--- py compile ---
  File "scripts/plot/plot_scaling_paper_figure.py", line 22
    from __future__ import annotations
                                     ^
SyntaxError: future feature annotations is not defined


exec
/usr/bin/bash -lc 'for P in .venv/bin/python .venv-mpi/bin/python /usr/bin/python3; do if [ -x "$P" ]; then "$P" -V; fi; done
rg -n -C 4 '"'SLURM_STEP_NODELIST|SLURM_NODELIST|slurm|coordinator_port|coordinator_address' .venv/lib .venv-mpi/lib 2>/dev/null | head -n 1000 || true
printf '\\n--- intended helper call sites ---\\n'
rg -n -C 3 'resolve_coordinator_port|maybe_init_jax_distributed\\(' packages scripts | head -n 500" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 1024ms:
Python 3.6.8

--- intended helper call sites ---
scripts/bench/run_levante_gpu_scaling.py-202-                if len(set(hostnames)) > 1:
scripts/bench/run_levante_gpu_scaling.py-203-                    import jax
scripts/bench/run_levante_gpu_scaling.py-204-                    from legoesm.parallel.early_init import (
scripts/bench/run_levante_gpu_scaling.py:205:                        resolve_coordinator_port,
scripts/bench/run_levante_gpu_scaling.py-206-                    )
scripts/bench/run_levante_gpu_scaling.py-207-                    jax.distributed.initialize(
scripts/bench/run_levante_gpu_scaling.py-208-                        coordinator_address=(
scripts/bench/run_levante_gpu_scaling.py:209:                            f"{hostnames[0]}:{resolve_coordinator_port()}"
scripts/bench/run_levante_gpu_scaling.py-210-                        ),
scripts/bench/run_levante_gpu_scaling.py-211-                        num_processes=n_procs,
scripts/bench/run_levante_gpu_scaling.py-212-                        process_id=rank,
--
scripts/bench/run_scaling_diagnosis.py-153-                hostnames = comm.allgather(socket.gethostname())
scripts/bench/run_scaling_diagnosis.py-154-                if len(set(hostnames)) > 1:
scripts/bench/run_scaling_diagnosis.py-155-                    from legoesm.parallel.early_init import (
scripts/bench/run_scaling_diagnosis.py:156:                        resolve_coordinator_port,
scripts/bench/run_scaling_diagnosis.py-157-                    )
scripts/bench/run_scaling_diagnosis.py:158:                    _port = resolve_coordinator_port()
scripts/bench/run_scaling_diagnosis.py-159-                    jax.distributed.initialize(
scripts/bench/run_scaling_diagnosis.py-160-                        coordinator_address=f"{hostnames[0]}:{_port}",
scripts/bench/run_scaling_diagnosis.py-161-                        num_processes=n_procs,
--
packages/core/legoesm/parallel/distributed.py-167-        # Env override / crc32(job-id)-derived / legacy fixed port: two jobs
packages/core/legoesm/parallel/distributed.py-168-        # sharing a node must not collide on the rendezvous socket
packages/core/legoesm/parallel/distributed.py-169-        # (EADDRINUSE on the second job's rank 0).
packages/core/legoesm/parallel/distributed.py:170:        from legoesm.parallel.early_init import resolve_coordinator_port
packages/core/legoesm/parallel/distributed.py-171-
packages/core/legoesm/parallel/distributed.py:172:        coordinator_port = resolve_coordinator_port(
packages/core/legoesm/parallel/distributed.py-173-            default=_JAX_DIST_COORDINATOR_PORT)
packages/core/legoesm/parallel/distributed.py-174-    coordinator_bind = f"{coordinator_address}:{coordinator_port}"
packages/core/legoesm/parallel/distributed.py-175-
--
packages/core/legoesm/parallel/early_init.py-7-Usage (at the very top of an entry-point script, before all other imports)::
packages/core/legoesm/parallel/early_init.py-8-
packages/core/legoesm/parallel/early_init.py-9-    from legoesm.parallel.early_init import maybe_init_jax_distributed
packages/core/legoesm/parallel/early_init.py:10:    maybe_init_jax_distributed()
packages/core/legoesm/parallel/early_init.py-11-
packages/core/legoesm/parallel/early_init.py-12-    # Safe to import legoESM / jax.numpy from here onward.
packages/core/legoesm/parallel/early_init.py-13-"""
--
packages/core/legoesm/parallel/early_init.py-30-_LEGACY_COORDINATOR_PORT = 1234
packages/core/legoesm/parallel/early_init.py-31-
packages/core/legoesm/parallel/early_init.py-32-
packages/core/legoesm/parallel/early_init.py:33:def resolve_coordinator_port(default: int = _LEGACY_COORDINATOR_PORT) -> int:
packages/core/legoesm/parallel/early_init.py-34-    """Deterministic ``jax.distributed`` coordinator port for THIS job.
packages/core/legoesm/parallel/early_init.py-35-
packages/core/legoesm/parallel/early_init.py-36-    Precedence:
--
packages/core/legoesm/parallel/early_init.py-388-    _warn_missing_nccl_plugin(rank=proc_id)
packages/core/legoesm/parallel/early_init.py-389-
packages/core/legoesm/parallel/early_init.py-390-
packages/core/legoesm/parallel/early_init.py:391:def maybe_init_jax_distributed(coordinator_port: int | None = None) -> bool:
packages/core/legoesm/parallel/early_init.py-392-    """Initialize ``jax.distributed`` if running under multi-node MPI.
packages/core/legoesm/parallel/early_init.py-393-
packages/core/legoesm/parallel/early_init.py-394-    Detects the MPI world size from the environment (SLURM_NTASKS /
--
packages/core/legoesm/parallel/early_init.py-403-    initialise the XLA backend and then make ``initialize()`` raise; #693).
packages/core/legoesm/parallel/early_init.py-404-
packages/core/legoesm/parallel/early_init.py-405-    ``coordinator_port=None`` (default) resolves the port via
packages/core/legoesm/parallel/early_init.py:406:    :func:`resolve_coordinator_port` (env override / job-id-derived /
packages/core/legoesm/parallel/early_init.py-407-    legacy 1234).
packages/core/legoesm/parallel/early_init.py-408-    """
packages/core/legoesm/parallel/early_init.py-409-    global _INITIALIZED
--
packages/core/legoesm/parallel/early_init.py-429-
packages/core/legoesm/parallel/early_init.py-430-    import jax
packages/core/legoesm/parallel/early_init.py-431-    if coordinator_port is None:
packages/core/legoesm/parallel/early_init.py:432:        coordinator_port = resolve_coordinator_port()
packages/core/legoesm/parallel/early_init.py-433-    coordinator = f"{hosts[0]}:{coordinator_port}"
packages/core/legoesm/parallel/early_init.py-434-    # Per-rank device binding via the shared launcher-family helper: a
packages/core/legoesm/parallel/early_init.py-435-    # shim-pinned CUDA_VISIBLE_DEVICES (the SLURM --gpu-bind=single:1
--
scripts/run/run_amip.py-33-from legoesm.parallel.early_init import maybe_init_jax_distributed
scripts/run/run_amip.py-34-
scripts/run/run_amip.py-35-if "--multicontroller" not in sys.argv:
scripts/run/run_amip.py:36:    maybe_init_jax_distributed()
scripts/run/run_amip.py-37-
scripts/run/run_amip.py-38-from legoesm.driver.config import (
scripts/run/run_amip.py-39-    VALID_RADIATION,

codex
Two arithmetic defects have surfaced in the revised prose: the recovered LL2048@128 point has 65.5k—not 32.7k—columns/GPU, and the LL2304 planned-arm tile counts are each off by 2×. I’m finishing the evidence check and exact issue mapping now.
exec
/usr/bin/bash -lc "/usr/bin/python3 -c \"from decimal import Decimal as D
A,B,C,DD,E=map(D,('189.82','308.96','191.99','333.39','537.91'))
s9n=D(2621568); lev=D(26); s8n=D(655362)
print('metis B/A,D/C,D/B,E/D:', B/A,DD/C,DD/B,E/DD)
print('s9 GC/s:',*[s9n*lev/(t/D(1000))/D(10)**9 for t in map(D,('12.47','9.60','11.48'))])
print('s8 best GC/s, peak ratio:',s8n*lev/(D('5.27')/D(1000))/D(10)**9,(s9n*lev/(D('9.60')/D(1000))/D(10)**9)/(s8n*lev/(D('5.27')/D(1000))/D(10)**9))
print('s9 speedups:',D('12.47')/D('9.60'),D('12.47')/D('9.60')/2,D('9.60')/D('11.48'))
ll=D(2048)*D(4096); work=ll*lev
print('LL2048 cols,collevels,tile128,GCs@5.58,GCs@9.60,eff64to128:',ll,work,ll/D(128),work/(D('5.58')/D(1000))/D(10)**9,work/(D('9.60')/D(1000))/D(10)**9,(D('6.7324')/D('5.58'))/2)
print('LL2304 tiles@96,@192; LL2880@192:',D(2304)*D(4608)/D(96),D(2304)*D(4608)/D(192),D(2880)*D(5760)/D(192))
print('ensemble:',D(4)/D('1.10'),D(4)*D('5.47')/D('1.10'),(D(4)*D('5.47')/D('1.10'))/D('5.94'))
print('s10 tiles:',D(10)*D(4)**10+D(2),(D(10)*D(4)**10+D(2))/D(128),(D(10)*D(4)**10+D(2))/D(224))
print('wet shortfall:',1-D(78000)/D('102012.5'))\" && printf '\\n--- required code references ---\\n' && nl -ba scripts/bench/bench_ocean_mpas_scaling.py | sed -n '742,755p' && nl -ba scripts/bench/bench_mpas_spmd_scaling.py | sed -n '401,427p;492,578p' && nl -ba packages/core/legoesm/parallel/early_init.py | sed -n '330,359p' && nl -ba packages/core/legoesm/parallel/early_init.py | sed -n '391,446p'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 389ms:
metis B/A,D/C,D/B,E/D: 1.627647244758191971341270677 1.736496692536069586957654044 1.079071724495080269290523045 1.613455712528870092084345661
s9 GC/s: 5.465979791499599037690457097 7.10008 5.937349128919860627177700348
s8 best GC/s, peak ratio: 3.233285009487666034155597723 2.195933850299529115206557597
s9 speedups: 1.298958333333333333333333333 0.6494791666666666666666666665 0.8362369337979094076655052265
LL2048 cols,collevels,tile128,GCs@5.58,GCs@9.60,eff64to128: 8388608 218103808 65536 39.08670394265232974910394265 22.71914666666666666666666667 0.6032616487455197132616487455
LL2304 tiles@96,@192; LL2880@192: 110592 55296 86400
ensemble: 3.636363636363636363636363636 19.89090909090909090909090909 3.348637894092439546985001530
s10 tiles: 10485762 81920.015625 46811.4375
wet shortfall: 0.2353878201200833231221664012

--- required code references ---
   742	        steady_median_ms=(round(fused_step_ms, 4)
   743	                          if fused_step_ms is not None else None),
   744	        step_latency_gate_loop_ms=round(gate_loop_med, 2),
   745	        step_latency_gate_loop_min_ms=round(float(np.min(steady)), 2),
   746	        per_step_ms=[round(x, 1) for x in per_step_ms],
   747	        cells=int(mesh.nCells) * args.nlev,
   748	        # HORIZONTAL cells/rank — the same unit as --cells-per-rank, so a
   749	        # weak-mode row is comparable to its target (codex: the 3-D count
   750	        # made rows look nlev-x larger).
   751	        cells_per_rank_achieved=int(mesh.nCells) // n_ranks,
   752	        # Fix 4 honesty flag: at np=1 the parity reference pre-runs the
   753	        # SAME shape before the timed loop, so per_step_ms[0] may not
   754	        # contain the real JIT compile.
   755	        compile_prewarmed_by_parity_ref=bool(
   401	
   402	    # Per-step timing: step 0 includes compile; record each step so re-trace
   403	    # (every step slow) is visible vs steady-state (steps 1.. fast).
   404	    per_step_ms = []
   405	    for _ in range(args.steps):
   406	        t0 = time.perf_counter()
   407	        if physics_fn is not None:
   408	            s = step(s, dt, physics_fn=physics_fn)
   409	        else:
   410	            s = step(s, dt)
   411	        _block(s)
   412	        per_step_ms.append((time.perf_counter() - t0) * 1e3)
   413	
   414	    if jax.process_count() > 1:
   415	        from jax.experimental import multihost_utils
   416	        multihost_utils.sync_global_devices("mpas_spmd_bench_end")
   417	
   418	    # HLO collective-permute census (#1113 ask 2): a STATIC compile property of
   419	    # the sharded step — the ppermute ROUND count that decomposes multi-node
   420	    # overhead (overhead ~= CPs/step * ~0.11 ms launch floor). The cube benches
   421	    # record this; the MPAS row did not, forcing an out-of-band census. Counted
   422	    # AFTER the timed loop so the census compile can't perturb per_step_ms[0]'s
   423	    # compile timing (the executable is already cached — this re-lower/compile
   424	    # is a cache hit; the count is data-independent, static in the partition).
   425	    # Best-effort (None if compilation is unsupported); the serial n=1 leg has
   426	    # no ppermute halo -> 0.
   427	    if physics_fn is not None:
   492	                return 5
   493	
   494	    steady = per_step_ms[args.warmup:]
   495	    med = float(np.median(steady))
   496	    rec = dict(
   497	        component="mpas_atm",
   498	        subdivision=args.subdivision, n_devices=nd,
   499	        n_cells=int(mesh.nCells), n_edges=int(mesh.nEdges), nlev=args.nlev,
   500	        partition_method=args.partition_method, physics=args.physics,
   501	        # lloyd=0 is the LABELLED synthetic scaling mesh — anti-masquerade:
   502	        # a row without this field could pass as a production-SCVT receipt.
   503	        lloyd_iterations=args.lloyd,
   504	        # Requested vs EFFECTIVE (post-"auto") strategy — a JSONL row
   505	        # saying "auto" would not reveal whether ppermute or allgather
   506	        # was actually measured (codex M3c-2 MINOR).
   507	        halo_strategy_requested=args.halo_strategy,
   508	        halo_strategy_effective=getattr(
   509	            step, "_halo_strategy_effective", "serial"),
   510	        steps=args.steps, dt=dt,
   511	        platform=jax.default_backend(),
   512	        n_processes=jax.process_count(),
   513	        multicontroller=bool(args.multicontroller),
   514	        compile_ms=round(per_step_ms[0], 1),
   515	        steady_median_ms=round(med, 2),
   516	        steady_min_ms=round(float(np.min(steady)), 2),
   517	        per_step_ms=[round(x, 1) for x in per_step_ms],
   518	        cells=int(mesh.nCells) * args.nlev,
   519	        # ppermute round count/step (static compile property; #1113) — the
   520	        # multi-node ceiling is this count x the ~0.11 ms launch floor, so it
   521	        # belongs on every row like the cube benches.
   522	        hlo_collective_permutes=hlo_cp,
   523	        # full per-family census (permute + all-reduce + all-gather + ...) on
   524	        # the SAME compile: exposes any reduction the ico step introduces.
   525	        hlo_collectives=hlo_census,
   526	    )
   527	    # Flat aggregator-compatible identity + metric fields (see the latlon
   528	    # twin): resolution = subdivision level, matching run_cpu_mpi_scaling's
   529	    # icosahedral convention so both lanes land on the same plot curves.
   530	    rec.update(
   531	        grid_type="icosahedral",
   532	        resolution=args.subdivision,
   533	        n_levels=args.nlev,
   534	        mode="strong",  # this bench fixes the mesh and sweeps devices
   535	        precision="float64" if jax.config.jax_enable_x64 else "float32",
   536	        physics_level=args.physics,
   537	        backend=jax.default_backend(),
   538	        **tidy_throughput_fields(
   539	            dt_seconds=dt, time_per_step_ms=med,
   540	            total_cells=int(mesh.nCells) * args.nlev),
   541	    )
   542	    rec["metadata"] = annotate_incomplete(scaling_metadata(
   543	        grid="icosahedral",
   544	        component="atmosphere",
   545	        resolution=f"L{args.subdivision}",
   546	        n_levels=args.nlev,
   547	        precision="float64" if jax.config.jax_enable_x64 else "float32",
   548	        n_gpus=(nd if jax.default_backend() in ("gpu", "cuda", "rocm")
   549	                else 0),
   550	        decomposition="cell_partition" if nd > 1 else "none",
   551	        # cells_per_rank is per PROCESS (n_ranks semantics); the per-device
   552	        # share lives in extra.cells_per_device — a single-process 4-device
   553	        # SPMD run has 1 rank owning ALL cells (codex finding 3).
   554	        cells_per_rank=int(mesh.nCells) * args.nlev
   555	        // max(jax.process_count(), 1),
   556	        scaling_kind="strong",  # this bench fixes the mesh and sweeps devices
   557	        extra={
   558	            "partition_method": args.partition_method,
   559	            "physics": args.physics,
   560	            "steps": args.steps,
   561	            "multicontroller": bool(args.multicontroller),
   562	            "cells_per_device": int(mesh.nCells) // nd * args.nlev,
   563	        },
   564	    ))
   565	    # Multi-controller: every process times the same program; process 0 owns
   566	    # the JSONL + stdout (others would duplicate/corrupt the append).
   567	    if jax.process_index() == 0:
   568	        _outdir = os.path.dirname(args.out)
   569	        if _outdir:  # a bare basename --out needs no mkdir
   570	            os.makedirs(_outdir, exist_ok=True)
   571	        with open(args.out, "a") as f:
   572	            f.write(json.dumps(rec) + "\n")
   573	        print(json.dumps(rec))
   574	        print(f"[mpas nd={nd} L{args.subdivision} nCells={mesh.nCells} "
   575	              f"nlev={args.nlev}] compile={rec['compile_ms']}ms "
   576	              f"steady_median={med:.2f}ms/step "
   577	              f"(per-step: {rec['per_step_ms']})")
   578	        if rec["metadata"]["virtual_cpu_devices"]:
   330	def init_multicontroller_distributed(coordinator: str | None = None) -> None:
   331	    """Initialize ``jax.distributed`` for a route-B multicontroller launch.
   332	
   333	    Shared by every ``--multicontroller`` entry point (the ocean/atm SPMD
   334	    benches and the ``run_omip`` route-B driver) so the launcher-env contract
   335	    lives in ONE place.  MUST run before any other JAX use (backend init).
   336	
   337	    - Explicit ``coordinator`` (``host:port``): read the launcher rank/size
   338	      from Open MPI ``OMPI_COMM_WORLD_SIZE``/``RANK`` or Cray PALS
   339	      ``PMI_SIZE``/``PMI_RANK`` and call ``jax.distributed.initialize``
   340	      directly (the mpiexec path; also how the self-spawn tests inject rank).
   341	    - No ``coordinator``: delegate to :func:`init_jax_distributed_with_fallback`
   342	      (SLURM/OMPI auto-detect or the PALS mpi4py bootstrap).
   343	
   344	    Real init failures re-raise loudly — a missing launcher rank env is a
   345	    hard ``SystemExit``, never a silent single-process fallback (that would
   346	    run N identical un-federated copies clobbering each other's output).
   347	
   348	    Idempotent: a no-op if the federation is already up (another entry point —
   349	    e.g. run_amip's import-time ``maybe_init_jax_distributed`` on a real MPI
   350	    launch — may have initialized first; a second ``jax.distributed.initialize``
   351	    would raise "already initialized").
   352	    """
   353	    global _INITIALIZED
   354	    if _INITIALIZED:
   355	        return
   356	    if coordinator is None:
   357	        init_jax_distributed_with_fallback()
   358	        _warn_missing_nccl_plugin()
   359	        return
   391	def maybe_init_jax_distributed(coordinator_port: int | None = None) -> bool:
   392	    """Initialize ``jax.distributed`` if running under multi-node MPI.
   393	
   394	    Detects the MPI world size from the environment (SLURM_NTASKS /
   395	    PMI_SIZE / OMPI_COMM_WORLD_SIZE).  When >1 rank and the ranks span
   396	    more than one hostname, calls ``jax.distributed.initialize()`` with
   397	    rank-0's hostname as coordinator.  On single-node MPI or serial runs,
   398	    does nothing.
   399	
   400	    Returns True if ``jax.distributed.initialize()`` was called, False
   401	    otherwise.  Safe to call multiple times — idempotency is tracked via a
   402	    module-level flag (NOT a ``jax.process_count()`` probe, which would
   403	    initialise the XLA backend and then make ``initialize()`` raise; #693).
   404	
   405	    ``coordinator_port=None`` (default) resolves the port via
   406	    :func:`resolve_coordinator_port` (env override / job-id-derived /
   407	    legacy 1234).
   408	    """
   409	    global _INITIALIZED
   410	    if _INITIALIZED:
   411	        return False
   412	
   413	    # Shared precedence (step > OMPI/PMI > allocation-wide NTASKS): an
   414	    # `srun -n1` inside a larger allocation must NOT enter the MPI path
   415	    # (codex).
   416	    ntasks = launcher_world_size()
   417	    if ntasks <= 1:
   418	        return False
   419	
   420	    from mpi4py import MPI
   421	    comm = MPI.COMM_WORLD
   422	    rank = comm.Get_rank()
   423	    size = comm.Get_size()
   424	
   425	    hosts = comm.allgather(socket.gethostname())
   426	    if len(set(hosts)) <= 1:
   427	        # Single-node MPI: JAX distributed not needed.
   428	        return False
   429	
   430	    import jax
   431	    if coordinator_port is None:
   432	        coordinator_port = resolve_coordinator_port()
   433	    coordinator = f"{hosts[0]}:{coordinator_port}"
   434	    # Per-rank device binding via the shared launcher-family helper: a
   435	    # shim-pinned CUDA_VISIBLE_DEVICES (the SLURM --gpu-bind=single:1
   436	    # standard, #693) resolves to [0] exactly as before; an UNPINNED or
   437	    # multi-device visible list indexes by the launcher's node-local rank
   438	    # (guarded SLURM_LOCALID / OMPI / PALS) instead of piling every local
   439	    # rank onto GPU 0 (codex).
   440	    jax.distributed.initialize(
   441	        coordinator_address=coordinator,
   442	        num_processes=size,
   443	        process_id=rank,
   444	        local_device_ids=_pals_local_device_ids(),
   445	    )
   446	    _INITIALIZED = True

codex
VERDICT: FIX-FIRST

Prior 12 disposition: 1 open; 2–3 fixed; 4 explicitly deferred; 5–6 fixed; 7 partially deferred (SHA still absent); 8 fixed; 9 fixed for exit-status/readability; 10 partial; 11 open; 12 fixed.

1. The METIS section still falsely equates `cells_per_rank_achieved=5120` with equal owned-cell load. That field is integer floor division, not partition balance ([source](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_ocean_mpas_scaling.py:751)). The recovered receipt’s actual owned ranges remain geometric 5120–5121; METIS 5100–5145 @32 and 5093–5144 @128. Revise [the claim](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/docs/performance/scaling/levante_campaign_2026-07-24.md:1597) to “nominal mean target,” alongside both owned and wet ranges.

2. LL2048@128 has 65,536 columns/GPU, not 32.7k: `2048×4096/128`. Thus its 0.603 efficiency is correct, but attributing it to being at/near the 30k floor is not ([campaign](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/docs/performance/scaling/levante_campaign_2026-07-24.md:1692)). The displayed 5.58 ms converts to 39.09 GC/s; 39.11 is only consistent if the underlying time was more precise than the table.

3. The LL2304 planned-arm tile counts are both 2× too small: @96 is 110.6k and @192 is 55.3k columns/GPU, not 55.3k and 27.6k. LL2880@192 = 86.4k is correct. Therefore the supposed LL2304@192 floor control is not below-floor ([script](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch:12)).

4. The ensemble is improved, but it does not counterbalance ordering: it always runs `solo_pre → all replicas → solo_post` ([script](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch:70)). Nor does it enforce overlap: `sacct` is best-effort (`|| true`), and no per-step node list/coordinator/port is actually printed. It can screen a 10% co-execution penalty, but cannot confirm that any penalty is specifically fabric contention rather than placement/topology/drift.

5. “Receipt validity” presently means JSON parsing plus presence of `steady_median_ms`; it accepts `NaN` and does not validate finite state, expected topology, or metadata. This matters for the 5,000-step run: the MPAS bench records timings but has no finite-state gate ([bench](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_mpas_spmd_scaling.py:402)). The two hundreds scripts also print missing/unparseable result files without converting that into nonzero `rc` ([GPU](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch:47), [CPU](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/atm_latlon2d_cpu_hundreds.sbatch:36)).

6. The np32 s9 SHA is visibly deferred, not recovered. Keep the result explicitly non-reproduction-grade until the job log or executed revision is recovered ([campaign](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/docs/performance/scaling/levante_campaign_2026-07-24.md:1637)).

The other arithmetic checks pass: METIS ratios, s9 GC/s and speedups, 3.64× ensemble threshold, and s10 tile counts. Slurm sizing is valid: the ensemble’s four 8-node full-GPU `--exact` steps fit disjointly; 2304 divides 96/192, 2880 divides 192; and r512@512 factors as 16×32 ranks, giving 32×32 columns/rank. Shell syntax and diff whitespace are clean.
tokens used
187,126
VERDICT: FIX-FIRST

Prior 12 disposition: 1 open; 2–3 fixed; 4 explicitly deferred; 5–6 fixed; 7 partially deferred (SHA still absent); 8 fixed; 9 fixed for exit-status/readability; 10 partial; 11 open; 12 fixed.

1. The METIS section still falsely equates `cells_per_rank_achieved=5120` with equal owned-cell load. That field is integer floor division, not partition balance ([source](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_ocean_mpas_scaling.py:751)). The recovered receipt’s actual owned ranges remain geometric 5120–5121; METIS 5100–5145 @32 and 5093–5144 @128. Revise [the claim](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/docs/performance/scaling/levante_campaign_2026-07-24.md:1597) to “nominal mean target,” alongside both owned and wet ranges.

2. LL2048@128 has 65,536 columns/GPU, not 32.7k: `2048×4096/128`. Thus its 0.603 efficiency is correct, but attributing it to being at/near the 30k floor is not ([campaign](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/docs/performance/scaling/levante_campaign_2026-07-24.md:1692)). The displayed 5.58 ms converts to 39.09 GC/s; 39.11 is only consistent if the underlying time was more precise than the table.

3. The LL2304 planned-arm tile counts are both 2× too small: @96 is 110.6k and @192 is 55.3k columns/GPU, not 55.3k and 27.6k. LL2880@192 = 86.4k is correct. Therefore the supposed LL2304@192 floor control is not below-floor ([script](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch:12)).

4. The ensemble is improved, but it does not counterbalance ordering: it always runs `solo_pre → all replicas → solo_post` ([script](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch:70)). Nor does it enforce overlap: `sacct` is best-effort (`|| true`), and no per-step node list/coordinator/port is actually printed. It can screen a 10% co-execution penalty, but cannot confirm that any penalty is specifically fabric contention rather than placement/topology/drift.

5. “Receipt validity” presently means JSON parsing plus presence of `steady_median_ms`; it accepts `NaN` and does not validate finite state, expected topology, or metadata. This matters for the 5,000-step run: the MPAS bench records timings but has no finite-state gate ([bench](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_mpas_spmd_scaling.py:402)). The two hundreds scripts also print missing/unparseable result files without converting that into nonzero `rc` ([GPU](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch:47), [CPU](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/atm_latlon2d_cpu_hundreds.sbatch:36)).

6. The np32 s9 SHA is visibly deferred, not recovered. Keep the result explicitly non-reproduction-grade until the job log or executed revision is recovered ([campaign](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/docs/performance/scaling/levante_campaign_2026-07-24.md:1637)).

The other arithmetic checks pass: METIS ratios, s9 GC/s and speedups, 3.64× ensemble threshold, and s10 tile counts. Slurm sizing is valid: the ensemble’s four 8-node full-GPU `--exact` steps fit disjointly; 2304 divides 96/192, 2880 divides 192; and r512@512 factors as 16×32 ranks, giving 32×32 columns/rank. Shell syntax and diff whitespace are clean.
