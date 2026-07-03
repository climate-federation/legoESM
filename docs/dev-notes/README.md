# `md_files/` — internal development notes

Working notes, plans, audits, review trackers, and development logs that are
**not** part of the published documentation site (`docs/index.md` toctree).
They are kept under version control for history and cross-reference, but live
here so the `docs/` root holds only the curated, rendered pages.

Code and curated docs reference these by **basename** (e.g. `see fv3_faithful.md`)
so comments survive the move into this folder.

## Cubed-sphere / FV3 fidelity & grids

- `fv3_faithful.md` — FV3-faithfulness state of the cubed-sphere dycore.
- `fv3_fortran_fidelity_review.md`, `fv3_fortran_fidelity_review_20260414.md` — line-by-line Fortran-vs-JAX fidelity reviews.
- `faithful_latlon_FV.md` — lat-lon finite-volume faithfulness notes.
- `cubed_sphere_cgrid_ocean_plan.md` — cubed-sphere C-grid ocean migration plan.
- `cubed_sphere_edge_artifacts.md` — cube-edge artifact diagnosis (W2 v-wind).
- `CROSS_GRID_COMPARISON_REPORT.md`, `cross_grid_comparison_plots_plan.md` — cross-grid comparison report + plot plan.
- `LATLON_CGRID_MIGRATION.md` — lat-lon C-grid migration log.
- `mercator_grid_plan.md` — Mercator grid plan.
- `ocean_grid_staggering.md` — ocean grid staggering reference.

## Ocean development & fidelity

- `OCEAN_DEVELOPMENT_LOG.md` — running ocean development log.
- `ocean_boundary_conditions_analysis.md` — ocean BC analysis.
- `ocean_experiments_reference.md` — ocean experiments reference.
- `ocean_faithfulness_nemo.md` — NEMO-faithfulness notes.
- `ocean_test_experiments_audit.md` — ocean test/experiment audit.
- `ocean_test_matrix_changelog.md` — ocean test-matrix changelog.
- `ocean_validation_improvement_plan.md` — ocean validation improvement plan.
- `OMIP_faithful.md` — OMIP/CORE2 faithfulness tracker.

## Physics / CRM / LES

- `CRM_faithful_SAM.md` — SAM-faithful CRM notes.
- `les_plane_turbulence_notes.md` — plane-LES turbulence notes.
- `parameterization_checks.md` — physics parameterization checks.
- `slab_s2s_documentation.md` — slab S2S documentation.

## Status & summaries

- `implementation_summary.md` — comprehensive implementation/test summary.
- `SIMULATION_FULL_CHECK_PROGRESS.md` — full-simulation check progress.
- `GPU_SCALING_BRANCH.md` — GPU-scaling branch notes.
