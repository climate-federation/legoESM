"""Audit inline ``from legoesm`` / ``import legoesm`` count per
hot-path dycore file.

Iters 71..187 hoisted ~80 inline imports out of indented blocks
(function bodies / methods) so they fire once at module-load time
rather than once per call (or once per JIT trace).  Iters 225..233
caught and re-applied PE + SW + NH + ocean dycores that had drifted
back through ``/clear`` cycles.

This test pins the *post-iter-229* state by counting the ``^[ \t]+(from
legoesm|import legoesm)`` occurrences (i.e. inline, indented) in each
hot-path dycore module and asserting it is ≤ the known intentional
budget.  Drift above the budget = a future ``/clear`` cycle re-
introduced an inline import; the test failure tells the iteration
exactly which file to clean up, without requiring the human to
manually grep the codebase.

Intentional inline-import budget per file:

* Hydrostatic PE: ``spectral_pe.py``, ``primitive_eq_mpas.py``,
  ``primitive_eq_latlon_cgrid.py`` = 0 each.
* ``primitive_eq_cdgrid.py`` = 4 — three mutable singletons need late
  binding (``_halo_backend`` and ``_mpi_topology`` from
  ``grids.halo``, ``_spmd_mesh`` from
  ``parallel.cubesphere_exchange``); ``_mpi_topology`` is referenced
  in two MPI branches so it appears twice (4 total).  See scaling.md
  §13.f for the rationale.
* Shallow water + NH dycores get the same audit.  Pure-function
  budgets are 0; raise only when a mutable-singleton late-binding is
  documented (so far none on these paths).  Iter-229 starts the
  sweep with ``shallow_water_fv3_cdgrid.py`` cleaned up; remaining
  files are tracked and will be hoisted in subsequent iterations.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


_SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "legoesm"
_ATMOS_DYN = _SRC_ROOT / "atmosphere" / "dynamics"
_OCEAN_DYN = _SRC_ROOT / "ocean" / "dynamics"
_ATMOS_PHYS = _SRC_ROOT / "atmosphere" / "physics"
_COUPLER = _SRC_ROOT / "coupler"
_ICE = _SRC_ROOT / "ice"
_LAND = _SRC_ROOT / "land"
_CORE = _SRC_ROOT / "core"
_PARALLEL = _SRC_ROOT / "parallel"


# Module path → max inline ``from legoesm`` / ``import legoesm`` count.
# Bump only when adding a *new* mutable-singleton inline that has been
# shown to require late binding.  Pure-function imports must always
# move to module level; their budget stays 0.
INLINE_IMPORT_BUDGET = {
    # Hydrostatic PE dycores (iter-225..228 sweep).
    _ATMOS_DYN / "spectral_pe.py":               0,
    _ATMOS_DYN / "primitive_eq_mpas.py":         0,
    _ATMOS_DYN / "primitive_eq_latlon_cgrid.py": 0,
    _ATMOS_DYN / "primitive_eq_cdgrid.py":       4,
    # Shallow-water + NH dycores (iter-229..232 sweep, complete).
    _ATMOS_DYN / "shallow_water_fv3_cdgrid.py":  0,
    _ATMOS_DYN / "shallow_water_mpas.py":        0,
    _ATMOS_DYN / "shallow_water_latlon_cgrid.py": 0,
    _ATMOS_DYN / "spectral_sw.py":               0,
    _ATMOS_DYN / "compressible_euler_cdgrid.py": 3,
    _ATMOS_DYN / "compressible_euler_mpas.py":   0,
    _ATMOS_DYN / "spectral_nh.py":               0,
    # Ocean dycores (iter-233 sweep, in progress).  Budgets reflect
    # measured counts as of iter-233 — fails-on-drift now, ratchet
    # down to 0 in subsequent iterations as each file is cleaned up.
    _OCEAN_DYN / "ocean_pe_latlon_cgrid.py":      0,
    _OCEAN_DYN / "ocean_pe_cdgrid.py":            0,
    _OCEAN_DYN / "ocean_pe_mpas.py":              0,
    _OCEAN_DYN / "ocean_pe_fc.py":                0,
    _OCEAN_DYN / "spectral_ocean_pe.py":          0,
    _OCEAN_DYN / "ocean_model.py":                0,
    _OCEAN_DYN / "ocean_model_latlon_cgrid.py":   0,
    _OCEAN_DYN / "ocean_model_mpas.py":           0,
    _OCEAN_DYN / "eta_floor.py":                  0,   # iter-233: cleaned
    _OCEAN_DYN / "barotropic_mpas.py":            0,
    # Physics integration files (iter-240 sweep, in progress).  Budgets
    # reflect *measured* counts — fails-on-drift now, ratchet down to 0
    # in subsequent iterations as each is hoisted.
    _ATMOS_PHYS / "turbulence" / "integration.py":         0,  # iter-240
    _ATMOS_PHYS / "gravity_wave_drag" / "integration.py":  0,  # iter-240
    _ATMOS_PHYS / "microphysics" / "integration.py":       0,  # iter-241
    # Radiation keeps 3 intentional inlines: RRTMGP lazy loads (×2)
    # avoid importing the heavy optics submodule chain when users use
    # gray radiation only.  Same lazy-load pattern as iter-216 SW FV3
    # CONNECTIVITY constants (which were safe to hoist) — but RRTMGP
    # imports trigger ~50 MB of optics constant-loading at import time.
    _ATMOS_PHYS / "radiation" / "integration.py":          3,  # iter-241
    _ATMOS_PHYS / "convection" / "integration.py":         0,  # iter-242
    # Coupler files (iter-243 sweep, in progress).  Budgets reflect
    # measured counts — fails-on-drift now, ratchet down in subsequent
    # iterations.
    _COUPLER / "accumulator.py":          0,  # iter-243: cleaned
    _COUPLER / "surface_exchange.py":     0,  # iter-243: cleaned
    _COUPLER / "coupler.py":              0,  # iter-244
    _COUPLER / "mpas_adapter.py":         0,  # iter-244
    # Ice + land surface modules (iter-245 sweep, in progress).
    _ICE / "state.py":                    0,  # iter-245: cleaned
    _ICE / "sea_ice.py":                  0,  # iter-246: cleaned
    _LAND / "slab_land.py":               0,  # iter-245: cleaned
    _LAND / "multilayer_land.py":         0,  # iter-245: cleaned
    # Core modules (iter-247 sweep, in progress).  precision.py keeps
    # 2 intentional inlines: importing runtime.backend at module level
    # creates a circular import chain (runtime/precision.py and
    # runtime/config.py both import core.precision; if core.precision
    # then top-imports runtime.backend, the runtime package's
    # __init__-time loading of runtime.precision deadlocks).
    _CORE / "precision.py":               2,  # iter-247: cycle-avoidance
    _CORE / "field.py":                   0,  # iter-247: cleaned
    _CORE / "tracers.py":                 0,  # iter-247: cleaned
    _CORE / "fv_tp_2d.py":                0,  # iter-247: cleaned
    _CORE / "operators_fv_latlon.py":     0,  # iter-248: cleaned
    # core/hardware.py is the legacy wrapper for runtime.backend; hoisting
    # its 4 inline imports recreates the same precision.py cycle (it's in
    # the runtime-package import chain).  Pinned at measured count.
    _CORE / "hardware.py":                4,  # iter-248: cycle-avoidance (legacy wrapper)
    _CORE / "operators.py":               3,  # iter-249: get_halo_backend hoisted; 3 remaining are cycle-avoidance (conservation._accumulation_dtype, parallel.reductions.global_sum_mpi, parallel.mesh.get_active_config — all reachable from core.field via dependency chain)
    _CORE / "operators_3d.py":            0,  # iter-249: cleaned
    _CORE / "operators_cdgrid.py":        1,  # iter-250: kept fv3_sw_core._d2a2c_vect inline (fv3_sw_core top-imports operators_cdgrid)
    _CORE / "conservation.py":            0,  # iter-251: hoisted all 12 (precision/runtime.backend/parallel.reductions/grids.vertical/operators_voronoi pure-fn imports)
    _CORE / "fv3_sw_core.py":             0,  # iter-252: hoisted all 14 (duogrid.ext_vector_dgrid×3, halo.{pad_halo,pad_halo_vector,synchronize_*}, operators_cdgrid.{a2b_ord4,fv3_d2cc,fv3_cc2c}, fv_tp_2d.{_pert_ppm,compute_transport_quantities,fv_tp_2d,transport_step})
    # Parallel package (iter-253 sweep, in progress).  Budgets reflect
    # measured counts — fails-on-drift now, ratchet down in subsequent
    # iterations as each file is hoisted.  Note: ``voronoi_partition.py``
    # and ``ensemble.py`` each show 1 match from a Sphinx ``::`` literal
    # block in the module docstring (a code example showing import
    # usage), not a real runtime inline import — budget pinned at 1 to
    # tolerate the false positive while still catching genuine drift.
    _PARALLEL / "runtime.py":             16,
    _PARALLEL / "sharded_dynamics.py":    11,
    _PARALLEL / "async_halo.py":          11,
    _PARALLEL / "cubesphere_exchange.py":  7,
    _PARALLEL / "reductions.py":           5,
    _PARALLEL / "halo_exchange.py":        5,
    _PARALLEL / "scaling_diagnostics.py":  3,
    _PARALLEL / "distributed.py":          3,
    _PARALLEL / "voronoi_mpi.py":          0,  # iter-253: hoisted partition_cells_{geometric,metis}
    _PARALLEL / "profiling.py":            2,  # both lines are docstring `::` examples
    _PARALLEL / "layout.py":               2,
    _PARALLEL / "latlon_mpi.py":           1,  # iter-253: hoisted _get_sendrecv_vjp; line 11 is docstring `::` example
    _PARALLEL / "device_config.py":        2,
    _PARALLEL / "voronoi_partition.py":    1,  # docstring `::` example
    _PARALLEL / "halo_exchange_voronoi.py": 1,
    _PARALLEL / "ensemble.py":             1,  # docstring `::` example
}


_INLINE_IMPORT_RE = re.compile(
    r"^[ \t]+(from legoesm|import legoesm)\b",
    re.MULTILINE,
)


@pytest.mark.parametrize(
    "path,budget",
    sorted(INLINE_IMPORT_BUDGET.items(), key=lambda kv: kv[0].name),
    ids=lambda v: v.name if isinstance(v, Path) else str(v),
)
def test_inline_legoesm_import_budget(path: Path, budget: int):
    assert path.exists(), f"missing dycore file: {path}"
    src = path.read_text()
    matches = _INLINE_IMPORT_RE.findall(src)
    count = len(matches)
    assert count <= budget, (
        f"{path.name} has {count} inline ``from legoesm`` / "
        f"``import legoesm`` (budget {budget}).  Hoist any "
        f"pure-function imports to module level — see scaling.md "
        f"§13.e/§13.f/§13.g for the iter-225..232 atmos sweep and "
        f"§13.l/§13.m for the iter-233+ ocean sweep that "
        f"established the current budgets.  Mutable singletons "
        f"(``_halo_backend``, ``_mpi_topology``, ``_spmd_mesh``) "
        f"are the *only* legitimate reason to keep an inline import "
        f"in these files."
    )
