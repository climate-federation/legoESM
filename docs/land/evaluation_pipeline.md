# legoESM land-surface evaluation pipeline

A shared, config-driven harness for evaluating legoESM land/canopy runs
against Fortran references, sibling schemes, and flux-tower observations.
It exists so a collaborator can **score their own land run, add a site, a
variable, a metric, or a reference without copying numerics into a new
script** — every stage has one home and one tested implementation.

> **Status.** The scoring core, recipe driver, references, CLI and tests are
> in place and green. The observation-scoring branch (FLUXNET/CHATS obs) and
> the runner/plot dedup are staged follow-ups — see
> [What is not here yet](#what-is-not-here-yet).

---

## 1. Why this shape (grounded in the field)

The design follows how established land/ESM benchmarking systems separate
concerns, so the pipeline is legible to anyone who has used them:

| System | What we borrowed |
| --- | --- |
| **ILAMB** (Collier et al. 2018, *JAMES*, doi:10.1029/2018MS001354) | Component **scores on [0, 1]** via `exp(-relative_error)`; bias normalised by reference variability; **centralised** RMSE (bias scored separately); Taylor-diagram distribution score; **variable → group → overall** aggregation. |
| **PLUMBER2 / PALS** (Ukkola et al. 2022, *ESSD* 14:449; Abramowitz et al. 2024, *BG* 21:5517) | Flux-tower metrics: NME, correlation, 5th/95th-percentile error, **PDF overlap**; scoring a model against **sibling benchmarks** (here: the two-leaf / SimpleSEB schemes). |
| **ESMValTool** (Righi et al. 2020, *GMD* 13:3383) | The **recipe**: a declarative YAML describing datasets + diagnostics, run by one engine — not a bespoke script per experiment. |
| **CTSM–NEON** (Wieder et al. 2023, *GMD* 16:5979) | Per-site bundle (forcing + params + obs) and a one-command site run. |

The pipeline is **five stages**; each maps to one module/location:

```
(1) data        docs/land/evaluation_data_manifest.md  +  scripts/data/fetch_land_eval_data.py
(2) forcing     packages/land/legoesm/land/boundary_data/   (chats_obs, fluxnet_forcing, ...)
(3) run         scripts/run/run_chats7_offline.py, run_fluxnet_offline.py   -> results/<run>/*.out
(4) score       packages/land/legoesm/land/evaluation/      (metrics, fluxio, scorecard, references, recipe)
(5) present     scripts/validate/eval_land.py (tables + scorecard.json), scripts/plot/plot_*  (figures)
```

---

## 2. The evaluation package

`packages/land/legoesm/land/evaluation/` — import as
`legoesm.land.evaluation`:

| Module | Responsibility |
| --- | --- |
| `metrics.py` | Pure-numpy comparison metrics + ILAMB unit-interval scores. **The single source of truth** — no script re-implements a metric. `METRIC_REGISTRY` maps names → callables; `get_metric` raises on an unknown name. |
| `fluxio.py` | The CLM-ML `.out` column **schema** (`FLUX_VARS`/`FSUN_VARS`/`AUX_VARS`) + `load_out` / `load_out_named`. |
| `scorecard.py` | `score_variable` (metrics + overall score) and `Scorecard` (variable → group → overall aggregation) → JSON. |
| `references.py` | Resolve a reference id to a directory following the in-tree → shared-bundle → error policy. Unknown id raises. |
| `recipe.py` | Parse/validate a recipe YAML (`load_recipe`), and run it (`run_recipe`). Unknown metric/reference/variable raises **before** scoring. |

CLI entry point: `scripts/validate/eval_land.py`.

---

## 3. Quickstart

Everything for the CHATS7 parity case (experiment E1) is in-tree — no data
pull. From the repo root, in your legoESM environment:

```bash
# (3) run the adapter offline at CHATS7, May 2007, Fortran-parity settings
JAX_ENABLE_X64=1 python scripts/run/run_chats7_offline.py \
    --output-dir results/chats7_adapter_2007-05 \
    --met-type 3 --runge-kutta-type 41 --num-ml-steps 30

# (4/5) score it against the Fortran + JAX references and write a scorecard
JAX_ENABLE_X64=1 python scripts/validate/eval_land.py \
    --recipe config/land_eval/chats7_parity.yaml
```

This prints a per-variable metric table per reference and writes
`validation_output/scorecard_chats7_parity.json`. List the available
metrics with `eval_land.py --list-metrics`.

---

## 4. Recipes

A recipe is a YAML file (in `config/land_eval/`) describing one or more
**cases**. A case = one model run scored against one or more references over
a set of variables and metrics.

```yaml
name: chats7_parity
output_dir: validation_output
cases:
  - name: chats7_adapter_parity
    model:
      label: legoESM adapter
      path: results/chats7_adapter_2007-05   # dir of <tag>_<schema>.out
      tag: CHATS7_2007-05
    references: [fortran_v2, jax_standalone]   # ids from references.py
    variables: [flux:shflx, flux:lhflx, flux:gpp, fsun:tl_sun, aux:btran]
    metrics: [bias, rmse, nrmse, corr, bias_score, rmse_score, taylor_score]
```

- **`variables`** are `schema:key` (e.g. `flux:shflx`); a bare `key` works
  when unambiguous. Valid keys are the `fluxio` schema (`--list` them by
  reading `FLUX_VARS`/`FSUN_VARS`/`AUX_VARS`).
- **`metrics`** are names from `METRIC_REGISTRY`. A typo raises at load.
- **`references`** are ids from `references.KNOWN_REFERENCES`, or a mapping
  `{id: ..., path: ..., label: ...}` to override the location.
- **`score_weights`** (optional, per case) override the ILAMB default
  component weights `{bias_score: 2, rmse_score: 2, taylor_score: 1}`.

The overall variable score is the weighted mean of the score components
(always computed from the defaults, independent of which raw metrics you
display), variables roll up to per-`group` (= schema tag) scores, and groups
roll up to one overall score per `(model, reference)` pair.

---

## 5. How to extend

**Add a metric** — add a pure `f(ref, mod) -> float` to `metrics.py`, register
it in `METRIC_REGISTRY` (and `SCORE_METRICS` if it is a [0, 1] higher-better
score), add a unit test in `tests/land/unit/test_eval_metrics.py`. It is then
usable by name in any recipe.

**Add a variable** — extend the relevant schema list in `fluxio.py`
(`FLUX_VARS`/`FSUN_VARS`/`AUX_VARS`) *and* the matching writer in
`scripts/run/run_chats7_offline.py` so column order stays in lockstep. The
recipe can then score `schema:new_key`.

**Add a reference** — add a `ReferenceSpec` to `KNOWN_REFERENCES` in
`references.py` (id, label, `fmt`, in-tree default and/or bundle relpath),
and document its acquisition in the data manifest. `fmt="clm_ml_out"`
references are scored by the recipe engine today.

**Add a site** — add a `FluxnetSite` to
`boundary_data/fluxnet_forcing.py::SITES`, document the data source in the
manifest, and add a fetch entry. Then `run_fluxnet_offline.py --site <code>`
produces the `.out` + `obs_targets.csv`.

**Add a recipe** — drop a YAML in `config/land_eval/`. Nothing else.

Every new `.py` gets a direct unit test (CLAUDE.md); dispatch (metric /
reference / scheme selection) must raise on an unknown value.

---

## 6. Data access

Data directories are gitignored — a fresh clone has no data. The policy is
**per-dataset** (see `docs/land/evaluation_data_manifest.md` for the full
list, DOIs, sizes and checksums):

1. **In-tree** — CHATS7 forcing + Fortran/JAX reference outputs ship in the
   repo (`clm-ml-jax/…`, `docs/output_files_clm_ml-v2/…`). Nothing to fetch.
2. **Fetch scripts** — FLUXNET (AmeriFlux/ICOS) and MODIS are downloaded and
   prepared via `scripts/data/fetch_land_eval_data.py` (reproducible; nothing
   large enters git).
3. **Shared bundle** — for staged/large data, set `$LEGO_LAND_EVAL_DATA` to a
   bundle root; `references.py` looks there before failing. When a reference
   cannot be found, the error names exactly what to fetch and where.

---

## 7. Metric reference

Lower-is-better **errors** (model units unless noted):
`bias`, `mae`, `rmse`, `centered_rmse` (bias removed), `nrmse` (RMSE/σ_ref),
`nme` (PLUMBER2 normalised mean error), `p5_error`/`p95_error`,
`std_ratio` (σ_mod/σ_ref → 1 is ideal).

Higher-is-better **[0, 1] scores** (used in the overall score):
- `bias_score` = `exp(-|bias|/σ_ref)`
- `rmse_score` = `exp(-crmse/σ_ref)` (centralised)
- `taylor_score` = `2(1+R)/(σ_ratio + 1/σ_ratio)²`
- `pdf_overlap` = distribution overlap coefficient (timing-insensitive)

Also: `corr`/`pearson_r`, and `phase_score(ref_cycle, mod_cycle)` for the
phase agreement of a diurnal/seasonal composite. `diurnal_cycle(t, v)`
builds the 48-bin composite shared by every diurnal figure.

---

## 8. Reusing the pattern in another component

This package is land-scoped, but the shape is component-neutral. To evaluate
ocean or atmosphere runs the same way, copy the pattern: a component
`evaluation/` package with a `metrics` module (reuse these where the maths is
identical), a format/`io` module for that component's outputs, `scorecard`
(this one is already generic — it only needs `VariableResult`s), a
`references` registry, and component recipes under `config/<comp>_eval/`.
Keep the metric definitions shared rather than re-derived.

---

## What is not here yet

Honest scope boundary (staged follow-ups, each a small increment on this
foundation):

- **Observation scoring in the recipe engine.** `run_case` scores
  `clm_ml_out` references (Fortran/JAX, row-aligned to the model grid). The
  `obs_csv`/`chats_obs` formats need per-site time-alignment; today the
  obs comparison lives in the dedicated plotters
  (`scripts/plot/plot_obs_vs_adapter_4sites.py`,
  `plot_chats7_profile_e2.py`), which already compute r/RMSE/bias against
  obs. Pointing a recipe case at an obs reference raises with that guidance
  rather than mis-scoring. Extending `run_case` with an `obs_csv` branch
  (align on the `time` column) is the next increment.
- **`.out` writer dedup.** The row writers still live in
  `run_chats7_offline.py` (they read the live CLM-ML `mlcanopy_type`);
  `run_fluxnet_offline.py` imports them via `importlib`. Moving them into
  `fluxio.py` (with lazy CLM imports) removes that hack.
- **Plot stage on the scorecard.** The plotters currently recompute their
  own summary stats; pointing them at `scorecard.json` closes the loop.
