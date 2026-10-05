# legoESM testing & experiment strategy

One shared, complexity-tiered strategy across every component (atmosphere, ocean,
land, sea-ice, coupler), with pass/fail energy & mass conservation gates, machine
profiles, and full reproducibility. This is the contract a new component plugs
into; it is designed to expand without re-deriving plumbing.

## The tier ladder (the cross-component contract)

One taxonomy, used by **both** the pytest markers and the test-matrix framework,
derived from the existing physical-complexity axis
(`legoesm.components.complexity.ModelComplexity`) and grid-extent axis
(`legoesm.grids.capability.EXTENTS`):

| Tier | Name | Complexity / extent | Examples | Data | Conservation gate |
|---|---|---|---|---|---|
| **0** | unit | kernels/operators | ops, halo, PPM, EOS, thermo | none | numerical invariants |
| **1** | research | idealized; column / shallow-water | dry dycore, SCM, Williamson, Galewsky | none | mass + energy + AAM, analytic-benchmark error |
| **2** | intermediate | hydrostatic 3D + slab | Held-Suarez, baroclinic wave, aquaplanet, RCEMIP | synthetic | mass + energy + moisture |
| **3** | operational | full complexity + real forcing | AMIP, OMIP, ERA5 ingest | required | full budget closure + reproducibility |

Research → operational. tier ↔ template category: tier1 ↔ `1d/`,`2d/`; tier1–2 ↔
`3d_idealized/`; tier3 ↔ `global_ocean/`,`coupled/`.

## 1. pytest tiers

Markers `tier0`..`tier3` (registered in `pyproject.toml`) are **opt-in
selectors**. The default `addopts` stays `-m 'not slow'` (it does *not* tier-gate,
so untagged tests still run). Run a rung explicitly:

```bash
.venv/bin/python -m pytest -m "tier1 and not slow"          # fast research ladder
JAX_ENABLE_X64=1 .venv/bin/python -m pytest -m tier2        # intermediate (slow)
```

Tag a test by module: `pytestmark = pytest.mark.tier1`. The shared conservation
gates are exposed to any pytest test via the `conservation_gate` fixture, so a
test asserts PASS/FAIL with the *same* gates the matrix runners use:

```python
def test_mass_conserved(conservation_gate):
    ok, notes = conservation_gate.mass_gate(True, "", mass_series, component="atmosphere")
    assert ok, notes
```

## 2. The test-matrix framework — `legoesm.experiments.matrix`

The single, component-agnostic home for the tier taxonomy, case/result types,
conservation gates, summary reporting, and the `MatrixRunner` base. It imports
only stdlib/numpy/`legoesm.diagnostics` — never a component — so it stays on the
`legoesm-tools` side of the federation DAG. The per-component runners in
`scripts/matrix/` import their component and subclass `MatrixRunner`.

**Plug in a component** — declare cases, run one case, chain gates:

```python
from legoesm.experiments.matrix import MatrixRunner, MatrixCase, RunStatus, mass_gate, energy_gate

class OceanMatrix(MatrixRunner):
    component = "ocean"
    def build_cases(self):
        return [MatrixCase("ocean", "rest_state", "cubed_sphere", tier=1,
                           complexity="full_3d", resolution="C24", duration_days=5)]
    def run_case(self, case, *, quick, output_dir):
        # integrate; compute conservation series via legoesm.diagnostics
        ok, notes = True, ""
        ok, notes = mass_gate(ok, notes, volume_series, component="ocean")
        ok, notes = energy_gate(ok, notes, heat_series, component="ocean")  # routes to heat_rel_drift
        return (RunStatus.PASS if ok else RunStatus.FAIL), notes, {}

if __name__ == "__main__":
    raise SystemExit(OceanMatrix().main())   # --tier/--grid/--only/--quick/--list/--allow-empty
```

`main()` filters, runs, records PASS/FAIL/ERROR/SKIP, writes
`results/<component>/{summary.json,summary.txt,summary.md}`, detects PASS→FAIL
regressions vs the prior summary, and **exits non-zero** on any FAIL/ERROR/empty
selection (so CI gates on it directly).

**Conservation gates** (`legoesm.experiments.matrix.gates`) all wrap the single
centralized drift convention in `legoesm.diagnostics.conservation_drift`
(`compute_relative_drift` + `apply_drift_tolerance`), with per-component
tolerances in one `CONS_THRESH`:

- `mass_gate` (ocean/sea-ice → volume), `energy_gate` (ocean → heat content),
  `heat_gate`, `salt_gate`
- `benchmark_error_gate` (analytic L2/Linf), `finite_gate` (NaN/Inf crash check)

Feed them the *scalar timeseries* you compute from
`diagnostics.compute_total_energy_{nh,pe}` / `column_water_vapor` /
`compute_atmospheric_angular_momentum`.

## 3. The dycore regression suite

`tests/atmosphere/dycore/regression/` holds 57 curated FV3/cubed-sphere
regression sentinels (one per locked numerical invariant — total energy, mass,
AAM, div-damp, cube-imprint, PPM limiter, halo, A2B, DCMIP, Williamson
production), tier1, four marked `slow`. See its `MANIFEST.md` for the kept→invariant
map and the coverage delta. The non-curated remainder is archived (recoverable)
under `scripts/tmp/dycore_iter_archive/`.

```bash
JAX_ENABLE_X64=1 .venv/bin/python -m pytest tests/atmosphere/dycore/regression/ -m tier1 -o addopts=""
```

## 4. The experiment harness — templates, machines, reproducibility

Layered on legoESM's **existing** reproducibility core (do not reinvent it):
`legoesm run <config.yaml>` writes `run_manifest.json` (resolved config +
`state_digest` + `git_hash` + jax/numpy versions + platform), and `legoesm
reproduce <manifest> --check` does a **bit-identical** re-run. That manifest is a
superset of the lesommer `experiment.tag`.

The harness adds the *intent/ergonomics* layer:

- **`config/templates/<category>/<name>.yaml`** — versioned `legoesm run` configs
  with an `experiment:` block (tier / complexity / extent / maturity / description
  / data / conservation_gates). `config/templates/README.md` lists them; the
  canonical run-status table is `project_status.md`.
- **`config/machines/*.yaml`** — per-host profiles (scheduler, `jax_platforms`,
  precision, venv, data_root, slurm). `lego_detect_machine.py` resolves by hostname.
- **`scripts/experiment/init_experiment.py <category/name> --name N --output-dir D
  [-o dot.key=val ...]`** — resolves template + overrides + machine → a
  self-contained dir (`config.yaml` runnable, `run.sh` launcher, `run.yaml`
  intent). Strict-validates *before* writing (bad override fails fast).
- **`scripts/experiment/validate_templates.py [--write-status]`** — loads every
  template through the same `Config.from_yaml(...).to_experiment_config().validate_strict()`
  path `legoesm run` uses; regenerates `project_status.md`.
- **`scripts/experiment/fetch_data.py check|fetch <template>`** — resolves a
  template's `experiment.data` against `config/data_catalog.yaml` + the machine
  `data_root`; idealized templates need nothing.

**Workflow:**

```bash
# idealized (no data)
python scripts/experiment/fetch_data.py check 2d/williamson2_sw
python scripts/experiment/init_experiment.py 2d/williamson2_sw --name w2 --output-dir ./runs/w2
cd ./runs/w2 && bash run.sh

# higher resolution + longer
python scripts/experiment/init_experiment.py 2d/williamson2_sw --name w2hi --output-dir ./runs/w2hi \
    -o grid.resolution=96 -o time.duration_hours=240

# data-gated template (the one template with a data list; the run itself is
# idealized — for AMIP use config/amip/amip_production.yaml via run_amip.py)
python scripts/experiment/fetch_data.py fetch 3d_idealized/hydrostatic_gray_1yr
python scripts/experiment/init_experiment.py 3d_idealized/hydrostatic_gray_1yr --name hs1 --output-dir ./runs/hs1

# reproduce any past run from its manifest
legoesm reproduce ./runs/w2/<output>/run_manifest.json --check
```

## 5. Source-guardrail harness (AI-assisted-dev tripwires)

A layered set of *static* guards that verify the existing source obeys the
project rules (constants/saturation/dispatch/units, the federation DAG, and
spec-first physics contracts). These run as ordinary pytest — fast, no GPU — and
are the cheapest "verify the existing code" pass. See
[`docs/architecture/ai_guardrails/domain_architect_vs_syntax_engine.md`](../architecture/ai_guardrails/domain_architect_vs_syntax_engine.md)
for the design (Domain Architect dictates logic; Syntax Engine fills the body).

```bash
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu .venv/bin/python -m pytest \
  tests/test_no_hardcoded_constants.py \
  tests/test_no_saturation_reimpl.py \
  tests/test_dispatch_hardening.py \
  tests/test_physics_contracts.py \
  tests/test_import_boundaries.py \
  tests/test_federation_plan.py \
  tests/test_install_federation.py -q
```

- **Ratchet audits** — `tests/_ratchet_audit.py` (shared engine, import-only) drives
  `test_no_hardcoded_constants` (no `9.80616`/`273.15`/… literals; use
  `legoesm.constants`) and `test_no_saturation_reimpl` (no re-derived Tetens/Magnus;
  use `legoesm.thermo`). Keys on `(file, value, count)` so swapping one banned
  literal for another still goes red; budgets ratchet *down* only.
- **Dispatch hardening** — `test_dispatch_hardening` proves every `scheme="…"`
  factory raises `ValueError` on an unknown name (no silent default).
- **Physics contracts** — `test_physics_contracts` checks every physics module
  declares a machine-checked `__physics_contract__` (units / sign / conserves /
  differentiable / reference / acceptance criterion).
- **Federation boundaries** — `test_import_boundaries` (zero-ignore import-linter
  contracts), `test_federation_plan` (every subpackage + loose module assigned to
  exactly one member), `test_install_federation` (the pip-install DAG resolver).
- **LIVE editor hooks** — `.claude/hooks/check_banned_literals.py` (blocks a banned
  constant *before* the edit lands) and `require_review_artifact.py` (review
  reminder) are wired in `.claude/settings.json` and fire automatically during
  Claude Code edits — no manual run needed.
- **Scientific validators** — `scripts/validate/*.py` exercise real numerics
  (each is `python scripts/validate/<name>.py`), e.g.
  `validate_convection_physics.py`, `validate_baro_solver.py`,
  `validate_ocean_scm.py`, `validate_federation_packaging.py` (per-member wheels
  build + root-absent import), and `visual_regression.py` (cube-imprint / edge
  artifacts — **inspect the PNGs**; norms alone are not sufficient).
- **Conservation sweep** — `python scripts/matrix/check_conservation_all.py`.

## 6. Adversarial-review agents (codex + subagents)

Beyond the static guards, two kinds of AI agents review the existing code. They
are **user-triggered** (and, for codex, billed); the result feeds the
iterate-to-clean loop in `CLAUDE.md`.

- **Codex** (slash commands) — adversarial review of the current diff/branch:

  ```text
  /codex:adversarial-review --wait      # find bugs; then fix flagged
  /codex:review --wait                  # re-review; repeat until clean (≤30 iter)
  ```

  One-time setup: `/codex:setup` (needs the Codex CLI; model config in
  `~/.codex/config.toml`).

- **Specialized subagents** (`.claude/agents/*.md`) — ask Claude Code to launch one
  on a target module/area:
  - `physics-validator` — units, sign conventions, conservation, differentiability,
    idealized-test fidelity, checked against the module's `__physics_contract__`
    (drives Codex internally, iterates to convergence). Use when adding/refactoring
    any physics scheme.
  - `lego-modularity-tester` — swaps parameterizations / NNs / dycores / grids /
    integrators / complexity levels and asserts every valid config still runs.
  - `dycore-tester`, `test-differentiability`, `test-scalability`,
    `test-land-ice`, `validate-matrix`, `slopbuster` (also `/slopbuster audit all`).

  Example: *"run physics-validator on `atmosphere/physics/convection/zhang_mcfarlane.py`"*
  or *"run lego-modularity-tester over the atmosphere dycores"*.

CI runs the static layers automatically (`.github/workflows/{ci,mpi-distributed,
mpi-nightly,claude-code-review}.yml`).

## Adding a component / case / template

1. **New matrix cases** → add `MatrixCase`s to your `scripts/matrix/<component>` runner
   and chain the relevant gates in `run_case`. No framework changes.
2. **New pytest tier** → tag with `pytestmark = pytest.mark.tierN`; use the
   `conservation_gate` fixture for pass/fail.
3. **New template** → drop a YAML under `config/templates/<category>/` with an
   `experiment:` block; run `validate_templates.py --write-status`.
4. **New machine** → add `config/machines/<name>.yaml` with `hostnames` patterns.
5. **New dataset** → add it to `config/data_catalog.yaml`.

## Conventions

- CPU only on Apple Silicon (`JAX_PLATFORMS=cpu`; the Metal backend is broken).
- `JAX_ENABLE_X64=1` for scientific/conservation tests unless explicitly fp32/Metal.
- The CI **unit tier runs fp32-by-default** (no job-wide `JAX_ENABLE_X64`); unit
  tests that need x64 self-enable it at module scope. The x64 CI jobs are
  `top-level-fidelity-tests` and `visual-regression`.
- Bit-reproducibility caveats (GPU reduction order, JIT cache, fp32-vs-x64) are
  catalogued in
  [`docs/architecture/portability_gpu_mpi_precision.md`](../architecture/portability_gpu_mpi_precision.md)
  ("Nondeterminism sources").
- Passing norms are necessary but **not sufficient** for cubed-sphere/halo/diffusion
  changes — visually inspect W2 v-wind / W5 wind-speed PNGs (see CLAUDE.md).
