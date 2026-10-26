# Federation — the uv-workspace carve (Stage D)

legoESM ships today as one installable package (`legoesm`), but its internal
layering is already a clean, acyclic dependency DAG: the three import-linter
contracts in `pyproject.toml` are **zero-ignore** (`tests/test_import_boundaries.py`
gates them), which *proves* the boundaries below hold — including deferred /
function-scope imports (grimp sees those too). So the carve into a
[uv workspace](https://docs.astral.sh/uv/concepts/workspaces/) of independently
installable members is **mechanical, not architectural**: no import has to move,
only files.

## Member layout

Each member depends only on the members *below* it. The arrows are exactly the
proven import-linter contracts.

**Directory convention.** A member's folder drops the `legoesm-` prefix (we are
already under `legoesm`) and there is no `src/` level: the source lives at
`packages/<member>/legoesm/<subpkg>/` (e.g. `packages/core/legoesm/core/`,
`packages/ml/legoesm/tuning.py`). Each member's `pyproject.toml` ships it with
`[tool.hatch.build.targets.wheel] packages = ["legoesm"]` — a PEP-420 portion of
the shared `legoesm` namespace (NO `legoesm/__init__.py`), so everything still
imports as `legoesm.<subpkg>`. The **distribution names keep the `legoesm-`
prefix** (`legoesm-core`, `legoesm-atmosphere`, …) — that is the PyPI identity and
what `[tool.uv.sources]` resolves; only the on-disk folder is shortened. (We do
*not* use a `sources` prefix-remap from `src` to `legoesm`: hatchling/uv refuse
editable installs for prefix-rewriting remaps — `editables` #20 — which would
break `uv sync`.) The root `legoesm` meta-member is the exception: its loose
modules stay under `src/legoesm/` so an editable install does not put the repo
root on `sys.path`.

| Member | Bundles (`legoesm/…`) | Depends on |
|--------|---------------------------|------------|
| **legoesm-core** (the substrate) | `core`, `grids`, `runtime`, `parallel`, `io`, `timestepping`, `components` + loose modules `constants`, `thermo`, `registry`, `surface_albedo`, `_version` | — (imports nothing above; contract #1/#4) |
| **legoesm-atmosphere** | `atmosphere` | core |
| **legoesm-ocean** | `ocean` | core |
| **legoesm-land** | `land` | core |
| **legoesm-ice** | `ice` | core |
| **legoesm-coupler** | `coupler`, `driver` | core + the four components + ml, tools |
| **legoesm-ml** | `ml`, `training`, `da` + loose module `tuning` | core, atmosphere, ocean, coupler, tools |
| **legoesm-tools** | `forcing`, `diagnostics`, `experiments`, `visualization` | core, atmosphere, ocean, coupler |
| **legoesm** (meta) | loose modules `cli`, `config`, `dycore_factory`, `experiment_registry`, `supported_matrix`; `legoesm[all]` | every member |

The orchestration cluster — **legoesm-coupler**, **legoesm-ml**, **legoesm-tools** —
is *not* a clean DAG: the driver coordinates the components and pulls forcing
(tools) + a deferred ERA5 path (ml); training (ml) drives the driver (coupler)
and uses forcing (tools); tools' experiment configs go through the driver. These
mutual edges (a uv-workspace cycle, which uv resolves) are why the three install
*together* — unlike the four independent components. The exact member-level
edges are AST-verified (top-level + deferred) and pinned in each member's
`pyproject.toml`. The loose top-level modules (not subpackages) are placed by the
import DAG, not by where the file sits today: `surface_albedo` ships with the
substrate because land/ice/coupler import it at top level (a standalone
`pip install legoesm-land` must get it from core); `tuning` ships with ml (its
only importer); the rest are the meta layer. `tests/test_federation_plan.py` pins
**both** the subpackage map and the loose-module map, so a new top-level
subpackage *or* module fails the test until it is assigned to a member.

Contract #2 (the four Earth-system components are mutually independent) is what
lets `legoesm-atmosphere` / `-ocean` / `-land` / `-ice` be **separate** members
rather than one blob — each can be `pip install`ed and run standalone (a single
column is its cheapest gradient-check harness; see `legoesm.components`).
Contract #3 (components must not import coupler/driver/training) is what keeps the
component members *below* the coupler member, so the DAG stays acyclic.

## Carve procedure (per member — DONE; recorded for adding future members)

1. `packages/<member>/pyproject.toml` — `[project]` (dist name `legoesm-<member>`)
   with the right inter-member deps + `[tool.uv.sources]` (`workspace = true`);
   `[tool.hatch.build.targets.wheel] packages = ["legoesm"]` (namespace package, so
   all members still import as `legoesm.<subpkg>`).
2. `git mv` the subpackage to `packages/<member>/legoesm/<subpkg>` (byte-identical
   relocation — numerics provably unchanged).
3. Root `pyproject.toml`: `[tool.uv.workspace] members = ["packages/*"]`; the root
   `legoesm` meta-member depends on the new member + lists it in `[tool.uv.sources]`.
4. `pip install --no-deps -e packages/<member>` (or `uv sync`) for the dev wiring;
   re-run `lint-imports` (4 kept / 0 broken), `tests/test_federation_plan.py`, and
   `scripts/validate_federation_packaging.py` before committing.

## Status: DONE

All eight members are carved (`packages/{core,atmosphere,ocean,land,ice,coupler,ml,tools}`)
plus the root `legoesm` meta. The four import-linter contracts are zero-/baseline-
locked (`tests/test_import_boundaries.py`), and `scripts/validate_federation_packaging.py`
builds every member wheel and proves namespace isolation, the dependency DAG,
root-absent installs (core+ocean, core+land via `surface_albedo`), the SFNO `[ml]`
extra gating, and entry-point dycore sharing. Dev wiring is proper editable installs
(no manual `.pth` hack); `uv.lock` pins the full workspace.

## Already done toward federation

- The three import-linter contracts (zero-ignore) + the `test_import_boundaries`
  gate — the separability proof.
- The import inversions that made it true: `coupling_fields`, `bulk_flux`,
  `surface_energy`, `PhysicsOutput`, `grid_adapters` → `core`; the SW state →
  `core`, the SW dycore → the `SW_BAROTROPIC_REGISTRY` + entry-point bootstrap;
  `parallel` no longer reaches up into atmosphere (`type(model)(...)`).
- Per-layer `[project.optional-dependencies]` feature extras (`ml`, `mesh`, `viz`,
  `data`, `docs`, `mpi`) already split optional weight off the core install.

## Breaking changes (the carve is intentionally not backward-compatible)

Independent shipping is incompatible with the old monolithic `legoesm.*` surface:
you cannot have both a package that re-exports everything AND members that install
separately. These breaks were taken early (pre-publication, zero internal callers —
verified by grep) and are *gated by `tests/test_public_surface.py`* so they cannot
regress silently. **Do not "fix" them with shims** — a `legoesm/__init__.py` shim
re-breaks PEP-420, and a `legoesm/io/restart.py` shim re-adds the `io → driver`
edge the independence contract forbids.

| Removed | Replacement |
|---------|-------------|
| `legoesm.__version__` | `importlib.metadata.version("legoesm")` (or `legoesm._version.__version__`) |
| `from legoesm import Field` | `from legoesm.core.field import Field` |
| `legoesm.hours/minutes/days/years` | inline the factor, or a local helper (these were unused conveniences) |
| `import legoesm.io.restart` / `from legoesm.io import save_restart` | `from legoesm.driver.restart import save_restart` |
| `legoesm.io.checkpoint` | `legoesm.driver.checkpoint` |
| `legoesm.io.distributed_checkpoint` | `legoesm.driver.distributed_checkpoint` |

`legoesm.io` now exposes only the substrate I/O — CMOR output, the generic
`state_checkpoint`, and the pure `state_digest` helpers. The config-aware
checkpoint/restart (run manifest, reproducibility spine, config serialization)
lives in the driver layer because it depends on the experiment-config schema,
which sits above the substrate.
