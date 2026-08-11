# Contributing to legoESM

legoESM is a fully differentiable Earth System Model in JAX. Contributions are
welcome — new parameterizations, dycores, grids, diagnostics, bug fixes, docs.
This guide encodes the project's working rules so a PR respects the architecture
by construction. (The same rules live in `CLAUDE.md` for AI-assisted work.)

## Development setup

```bash
git clone https://github.com/climate-federation/legoESM
cd legoESM
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Run the test suite:

```bash
.venv/bin/python -m pytest tests/
```

**Scientific tests must enable 64-bit floats** (spectral cores and conservation
checks rely on it):

```bash
JAX_ENABLE_X64=1 .venv/bin/python -m pytest <target>
```

On Apple Silicon the Metal JAX backend is unreliable — run on CPU
(`JAX_PLATFORMS=cpu`).

## The non-negotiables

These are the invariants every change must preserve.

1. **Differentiability.** End-to-end `jax.grad` through the coupled model is the
   project's defining feature. Never break autodiff, JIT, or pytree structure.
   Use `jnp.where`/`lax.cond`/`lax.scan`/`fori_loop` — not Python control flow —
   on traced values. Implicit solves (surface energy balance, implicit coupling)
   are differentiated via the implicit-function theorem
   (`optimistix`/`lax.custom_root`), never by backprop through an unbounded loop.
   No hard regime branches in flux/closure code — blend smoothly so VJPs stay
   continuous.
2. **Conservation.** Mass conservation is a hard constraint; energy and momentum
   are conserved wherever the scheme permits. Flux-form coupling conserves by
   construction. New exchanged quantities get a global-budget test.
3. **Every new `.py` gets a direct unit test in the same PR.** For a physics
   scheme, test the leaf tendency function — not just an integration smoke test
   through a factory. Adding a config dispatch (`Literal` + factory) requires a
   test through the public config in the same PR.
4. **No duplicate numerics.** The refactor *moves* code, it does not copy it.
   Before writing a new helper/constant/formula, grep `src/legoesm/` for an
   existing one and extend it. No thin re-export wrappers
   (`X_utils.py` re-exporting `X.py`).
5. **AI-guardrail harness (Domain Architect vs Syntax Engine).** See
   `docs/architecture/ai_guardrails/domain_architect_vs_syntax_engine.md`. The human pins the
   *logic* (units/signs/conserved-qty/valid-sets/references); mechanical gates
   make any violation fail loudly. These are ratchets — extend, never weaken;
   `*_TODO`/budget lists only shrink:
   - Every physics scheme module declares a `__physics_contract__`
     (`tests/test_physics_contracts.py`); a new one must ship a contract or be
     classified. Author the contract + an acceptance test *before* the body.
   - The constant/saturation/dispatch/validate-strict ratchets
     (`tests/test_no_hardcoded_constants.py`, `test_no_saturation_reimpl.py`,
     `test_dispatch_hardening.py`, `test_validate_strict_coverage.py`) must stay
     green; annotate a genuine non-physical literal with `# const-ok: <reason>`
     rather than weakening a gate. Each gate ships a non-vacuity self-test —
     never make a gate that can pass on empty input.
   - The local hooks in `.claude/hooks/` (banned-literal block + review reminder)
     are a convenience; **CI is authoritative**.

## Shared utilities — never re-derive

- **Constants** live in `src/legoesm/constants.py`. Import them
  (`from legoesm import constants`). No literal `273.15`, `287.0`, `1004.64`,
  `9.80616`, `6.371e6`, `7.292e-5`, etc., anywhere — including tests, scripts,
  plotters, and notebooks. A new physical constant is added there *before* use.
- **Saturation/thermo:** `legoesm.thermo` (`saturation_vapor_pressure`,
  `saturation_mixing_ratio`). Never re-implement Tetens/Magnus/Clausius–Clapeyron.
- **Column integrals:** `legoesm.diagnostics.column_integrals`.
- **Losses:** `legoesm.ml.loss`. **Optimizer:** `legoesm.ml.training.create_optimizer`.
- **Ocean EOS/pressure:** `legoesm.ocean.eos`.

Tunable scheme parameters (timescales, drag coefficients, sigmoid sharpness)
belong in a scheme `*Config` NamedTuple, not as magic numbers in function bodies.
Carry a unit suffix (`_C`, `_K`, `_s`, `_days`, `_m`) on any quantity whose unit
is otherwise ambiguous.

## Import-boundary discipline

- `core/` must not import from `runtime/`, `parallel/`, `driver/`, `training/`,
  or `experiments/` at module top level (use function-scope deferred imports if
  truly needed). Components never import each other or the coupler — only `core`
  is shared.
- No importing private (`_`-prefixed) symbols across modules. Promote to a public
  name or factor a public wrapper.

## Dispatch discipline

Every `scheme="..."` factory must `raise ValueError` on an unknown value — never
silently fall through to a default (that masks typos and dead branches). Validate
on the static Python value at function entry, not inside a traced
`lax.cond`/`fori_loop` body. Add membership-set assertions to
`ExperimentConfig.validate_strict` for new scheme literals.

## Reproducibility

The repo's safety net is a bit-repro anchor: a mechanical/refactor change must
reproduce bit-identically (`legoesm reproduce --check <manifest>`). An *intended*
numerics change regenerates golden digests in a dedicated PR whose sole diff is
the digest update — so a numerics change can never sneak in under a refactor.

## Validating numerical changes

Passing unit tests and norms are **necessary but not sufficient** for
grid/dycore/halo changes. For any cubed-sphere or grid-touching change, also
inspect the W2 v-wind / W5 wind-speed PNGs against a baseline — edge artifacts
and grid-scale noise are only visible visually. Prefer analytical/benchmark
validation (Williamson, Galewsky, Jablonowski–Williamson, Held–Suarez, ocean
benchmarks) over unit tests alone for numerics.

## How to add a component (stub)

A component is an interchangeable "lego" brick implementing `AbstractComponent`:
`prognostic_variables`, `required_forcing`, `provided_fluxes`, and a pure
`tendency(grid, state, forcing, params) -> dstate`. Every component **must run
standalone on a single-column grid** (its cheapest gradient-check harness) with
partners supplied by `PrescribedX` bricks. Register it for discovery, add it to
`validate_strict`, and add the standalone single-column test in the same PR.
*(Full how-to expands as Stage B of the refactor lands the contracts.)*

## Pull requests

- Keep diffs minimal; no unrelated refactor inside a bug fix.
- One logical change per PR; add a `CHANGELOG.md` entry.
- Stage explicit paths — never `git add .`/`-A`. Never commit `docs/references/`
  (local research PDFs); cite by filename/DOI instead.
- CI must be green: tests, `ruff`, version single-source check, and (from Stage B)
  import-linter.

### Merging: one at a time, and verify it landed

**`gh pr merge` reporting success is not evidence that the change is on
`main`.** Squash merges that land within a few seconds of each other clobber
one another: whichever merge updates the ref last wins, the others' squash
commits are orphaned, and GitHub still shows those PRs `MERGED` with a
`mergeCommit` OID that is **unfetchable** afterwards.

Measured on 2026-08-10: seven PRs were lost this way — #1540 and #1541 in one
pair at 05:00, and #1550/#1551/#1552/#1554/#1555 from a loop that merged the
queue back to back. All were recovered (#1557), but only because the head
branches still existed.

So:

1. **Never loop `gh pr merge`.** Merge one PR, then verify, then the next.
2. **Always merge with `--delete-branch=false`.** Once the squash commit is
   orphaned the head branch is the *only* recovery path.
3. **Verify both ancestry and content** before starting the next merge:

   ```bash
   git fetch -q origin main
   git merge-base --is-ancestor "$MERGE_COMMIT" origin/main   # ancestry
   git show origin/main:path/to/changed_file | grep -c some_new_symbol  # content
   ```

If a merge did go missing, recover it by taking the PR's files from its own
head branch, and only for files where `main` has not diverged since that PR's
merge base:

```bash
base=$(git merge-base origin/main origin/<pr-branch>)
git diff --quiet "$base" origin/main -- "$f" && git checkout origin/<pr-branch> -- "$f"
```

Do **not** merge the head branch wholesale (PR branches here are often stacked,
so the merge drags in another PR's pre-squash history and can revert work that
is already on `main`), and do **not** `git apply` the raw `gh pr diff` (it
applies textually onto a moved `main` and leaves a tree that imports but fails).
