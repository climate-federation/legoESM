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
three proven contracts.

| Member | Bundles (`src/legoesm/…`) | Depends on |
|--------|---------------------------|------------|
| **legoesm-core** (the substrate) | `core`, `grids`, `runtime`, `parallel`, `io`, `timestepping`, `components` | — (imports nothing above; contract #1) |
| **legoesm-atmosphere** | `atmosphere` | core |
| **legoesm-ocean** | `ocean` | core |
| **legoesm-land** | `land` | core |
| **legoesm-ice** | `ice` | core |
| **legoesm-coupler** | `coupler`, `driver` | core + the four components |
| **legoesm-ml** | `ml`, `training`, `da` | core (+ components for training) |
| **legoesm-tools** | `forcing`, `diagnostics`, `experiments`, `visualization` | core + components |
| **legoesm** (meta) | re-exports; `legoesm[all]` | every member |

Contract #2 (the four Earth-system components are mutually independent) is what
lets `legoesm-atmosphere` / `-ocean` / `-land` / `-ice` be **separate** members
rather than one blob — each can be `pip install`ed and run standalone (a single
column is its cheapest gradient-check harness; see `legoesm.components`).
Contract #3 (components must not import coupler/driver/training) is what keeps the
component members *below* the coupler member, so the DAG stays acyclic.

## Carve procedure (per member, deferred to a coordinated window)

1. `packages/<member>/pyproject.toml` — `[project]` with the right inter-member
   deps; `[tool.hatch.build] packages = ["src/legoesm/<subpkg>"]` (namespace
   package, so all members still import as `legoesm.<subpkg>`).
2. `git mv src/legoesm/<subpkg> packages/<member>/src/legoesm/<subpkg>` (byte-identical
   relocation — numerics provably unchanged).
3. Root `pyproject.toml`: `[tool.uv.workspace] members = ["packages/*"]`; the root
   `legoesm` becomes the meta-member whose extras pull the others.
4. Re-run `lint-imports` (must stay 3 kept / 0 broken — now *enforced across member
   boundaries*) and the full test matrix before committing the member.

## Why it is deferred, not done here

The relocation touches ocean/ice/atmosphere file paths that a **concurrent session**
is actively editing, and it needs the full CI matrix (CPU + MPI + the W2/W5 visual
suite) green per member to prove no path/packaging regression. Doing it blind would
risk clobbering in-flight work and silent import breakage. The architecture is
*ready* (the contracts prove it); the physical move is a mechanical follow-up to run
when the tree is quiet.

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
