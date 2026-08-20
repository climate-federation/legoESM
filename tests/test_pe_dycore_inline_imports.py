"""Audit inline ``from legoesm`` / ``import legoesm`` count per
hot-path dycore file.

Iters 71..187 hoisted ~80 inline imports out of indented blocks
(function bodies / methods) so they fire once at module-load time
rather than once per call (or once per JIT trace).  Iters 225..233
caught and re-applied PE + SW + NH + ocean dycores that had drifted
back through ``/clear`` cycles.

This test counts the ``^[ \t]+(from legoesm|import legoesm)`` occurrences
(i.e. inline, indented) in each hot-path dycore module and asserts it is ≤ a
recorded budget.  Drift above the budget tells the iteration exactly which file
re-introduced an inline import, without requiring the human to grep the codebase.

**Post-federation re-baseline.** The federation carve (FEDERATION.md) split the
package into uv-workspace members and broke several cross-package import cycles by
moving imports to function scope (the deferred-import pattern in CLAUDE.md), which
legitimately *raised* the inline counts above the pre-carve iter-225..253 budgets.
This test had been silently red since the carve (it anchored to the now-moved
``src/legoesm`` paths, so every ``path.exists()`` short-circuited before counting).
Resolving each module through ``legoesm_source_path`` (namespace-aware) restored the
audit, and the budgets below were re-measured against the post-carve tree.  They are
a *drift baseline*, not a target: ratcheting them back toward 0 (hoisting the
genuinely pure-function imports, keeping only mutable-singleton / cycle-avoidance
late binding) is the inline-import sweep's follow-up.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath

import pytest

from tests.legoesm_paths import legoesm_source_path

# Keys below are RELATIVE paths under the ``legoesm`` namespace (e.g.
# ``atmosphere/dynamics/gcm/spectral_pe.py``).  The federation carve moved these
# subpackages out to ``packages/<member>/legoesm/<subpkg>``, so the concrete
# on-disk file is resolved at test time via ``legoesm_source_path`` (namespace-
# aware, spans every ``legoesm.__path__`` root) instead of a hardcoded
# ``src/legoesm`` anchor.
# a16e1493a moved every GCM dycore under the ``gcm`` bucket
# (dynamics/{gcm,les,crm,shared,neural}); the pre-reorg anchor made all 11
# entries below raise FileNotFoundError instead of counting anything.
_ATMOS_DYN = PurePosixPath("atmosphere/dynamics/gcm")
_OCEAN_DYN = PurePosixPath("ocean/dynamics")
_ATMOS_PHYS = PurePosixPath("atmosphere/physics")
_COUPLER = PurePosixPath("coupler")
_ICE = PurePosixPath("ice")
_LAND = PurePosixPath("land")
_CORE = PurePosixPath("core")
_PARALLEL = PurePosixPath("parallel")


# Module path → max inline ``from legoesm`` / ``import legoesm`` count.
# Bump only when adding a *new* mutable-singleton inline that has been
# shown to require late binding.  Pure-function imports must always
# move to module level; their budget stays 0.
INLINE_IMPORT_BUDGET = {
    # Hydrostatic PE dycores (iter-225..228 sweep).
    _ATMOS_DYN / "spectral_pe.py": 1,
    _ATMOS_DYN / "primitive_eq_mpas.py": 1,
    _ATMOS_DYN / "primitive_eq_latlon_cgrid.py": 3,
    _ATMOS_DYN / "primitive_eq_cdgrid.py": 20,
    # Shallow-water + NH dycores (iter-229..232 sweep, complete).
    _ATMOS_DYN / "shallow_water_fv3_cdgrid.py": 5,
    _ATMOS_DYN / "shallow_water_mpas.py": 0,
    _ATMOS_DYN / "shallow_water_latlon_cgrid.py": 3,
    _ATMOS_DYN / "spectral_sw.py": 1,
    _ATMOS_DYN / "compressible_euler_cdgrid.py": 23,
    _ATMOS_DYN / "compressible_euler_mpas.py": 0,
    _ATMOS_DYN / "spectral_nh.py": 3,
    # Ocean dycores (iter-233 sweep, in progress).  Budgets reflect
    # measured counts as of iter-233 — fails-on-drift now, ratchet
    # down to 0 in subsequent iterations as each file is cleaned up.
    _OCEAN_DYN / "ocean_pe_latlon_cgrid.py": 25,
    _OCEAN_DYN / "ocean_pe_cdgrid.py": 0,
    _OCEAN_DYN / "ocean_pe_mpas.py": 4,
    _OCEAN_DYN / "spectral_ocean_pe.py": 0,
    _OCEAN_DYN / "ocean_model.py": 1,
    _OCEAN_DYN / "ocean_model_latlon_cgrid.py": 55,
    _OCEAN_DYN / "ocean_model_mpas.py": 11,
    _OCEAN_DYN / "eta_floor.py": 1,
    _OCEAN_DYN / "barotropic_mpas.py": 1,
    # Physics integration files (iter-240 sweep, in progress).  Budgets
    # reflect *measured* counts — fails-on-drift now, ratchet down to 0
    # in subsequent iterations as each is hoisted.
    _ATMOS_PHYS / "turbulence" / "integration.py": 4,
    _ATMOS_PHYS / "gravity_wave_drag" / "integration.py": 1,
    _ATMOS_PHYS / "microphysics" / "integration.py": 1,
    # Radiation keeps 3 intentional inlines: RRTMGP lazy loads (×2)
    # avoid importing the heavy optics submodule chain when users use
    # gray radiation only.  Same lazy-load pattern as iter-216 SW FV3
    # CONNECTIVITY constants (which were safe to hoist) — but RRTMGP
    # imports trigger ~50 MB of optics constant-loading at import time.
    _ATMOS_PHYS / "radiation" / "integration.py": 13,
    _ATMOS_PHYS / "convection" / "integration.py": 0,
    # Coupler files (iter-243 sweep, in progress).  Budgets reflect
    # measured counts — fails-on-drift now, ratchet down in subsequent
    # iterations.
    _COUPLER / "accumulator.py": 0,
    _COUPLER / "surface_exchange.py": 0,
    _COUPLER / "coupler.py": 0,
    _COUPLER / "mpas_adapter.py": 0,
    # Ice + land surface modules (iter-245 sweep, in progress).
    _ICE / "state.py": 0,
    # Hoist sweep tranche 1: all 9 inline imports moved to module level (six
    # grid-type imports used only for isinstance checks, plus ice.state's
    # slab_to_dynamic and ice.rheology's strain_rates -- neither module imports
    # sea_ice back, and sea_ice already imports both at module scope).
    _ICE / "sea_ice.py": 0,
    _LAND / "slab_land.py": 0,
    _LAND / "multilayer_land.py": 0,
    # Core modules (iter-247 sweep, in progress).  precision.py keeps
    # 2 intentional inlines: importing runtime.backend at module level
    # creates a circular import chain (runtime/precision.py and
    # runtime/config.py both import core.precision; if core.precision
    # then top-imports runtime.backend, the runtime package's
    # __init__-time loading of runtime.precision deadlocks).
    _CORE / "precision.py": 1,
    _CORE / "field.py": 0,
    _CORE / "tracers.py": 0,
    # get_mpi_topology is the ACCESSOR, not the mutable singleton: it reads
    # the module global on every call, so a module-level binding still observes
    # set_halo_backend().  Importing the accessor is what CLAUDE.md prescribes.
    _CORE / "fv_tp_2d.py": 0,
    _CORE / "operators_fv_latlon.py": 0,
    # core/hardware.py is the legacy wrapper for runtime.backend; hoisting
    # its 4 inline imports recreates the same precision.py cycle (it's in
    # the runtime-package import chain).  Pinned at measured count.
    _CORE / "hardware.py": 4,
    _CORE / "operators.py": 2,
    _CORE / "operators_3d.py": 0,
    _CORE / "operators_cdgrid.py": 1,
    _CORE / "conservation.py": 12,
    _CORE / "fv3_sw_core.py": 0,
    # Parallel package (iter-253 sweep, in progress).  Budgets reflect
    # measured counts — fails-on-drift now, ratchet down in subsequent
    # iterations as each file is hoisted.  Note: ``voronoi_partition.py``
    # and ``ensemble.py`` each show 1 match from a Sphinx ``::`` literal
    # block in the module docstring (a code example showing import
    # usage), not a real runtime inline import — budget pinned at 1 to
    # tolerate the false positive while still catching genuine drift.
    _PARALLEL / "runtime.py": 7,
    _PARALLEL / "sharded_dynamics.py": 9,
    _PARALLEL / "async_halo.py": 4,
    _PARALLEL / "cubesphere_exchange.py": 8,
    _PARALLEL / "reductions.py": 2,
    _PARALLEL / "halo_exchange.py": 3,
    _PARALLEL / "scaling_diagnostics.py": 3,
    _PARALLEL / "distributed.py": 11,
    _PARALLEL / "voronoi_mpi.py": 1,
    _PARALLEL / "profiling.py": 2,  # both lines are docstring `::` examples
    _PARALLEL / "layout.py": 2,
    _PARALLEL / "latlon_mpi.py": 13,
    _PARALLEL / "device_config.py": 5,
    _PARALLEL / "voronoi_partition.py": 1,  # docstring `::` example
    _PARALLEL / "halo_exchange_voronoi.py": 4,
    _PARALLEL / "ensemble.py": 1,  # docstring `::` example
}


_INLINE_IMPORT_RE = re.compile(
    r"^[ \t]+(from legoesm|import legoesm)\b",
    re.MULTILINE,
)


@pytest.mark.parametrize(
    "path,budget",
    sorted(INLINE_IMPORT_BUDGET.items(), key=lambda kv: kv[0].name),
    ids=lambda v: v.name if isinstance(v, PurePosixPath) else str(v),
)
def test_inline_legoesm_import_budget(path: PurePosixPath, budget: int):
    # Resolve the relative namespace path to the concrete file across every
    # legoesm root (raises FileNotFoundError if the audited module went missing).
    resolved = legoesm_source_path(path)
    src = resolved.read_text()
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


# --------------------------------------------------------------------------- #
# Cold-import gate for the hoisted edges.
#
# Hoisting an inline import to module level is only safe while the target does
# not import its way back to the host.  A back-edge added later would surface as
# a partially-initialised module, and ONLY on a cold interpreter whose first
# ``legoesm`` import is the affected module — a session that already imported
# the package (which every other test does) will not reproduce it.  So each
# module below is imported in a FRESH subprocess, first, on its own.
#
# Grow this list whenever a further hoist-sweep tranche lands.
# --------------------------------------------------------------------------- #
COLD_IMPORT_MODULES = (
    "legoesm.core.fv3_sw_core",
    "legoesm.core.fv_tp_2d",
    "legoesm.grids.halo",
    "legoesm.ice.sea_ice",
    "legoesm.ice.state",
    "legoesm.ice.rheology",
    "legoesm.ice.dynamics",
)


@pytest.mark.parametrize("module", COLD_IMPORT_MODULES)
def test_module_imports_cold(module: str):
    """Importing *module* first, in a fresh interpreter, must not cycle."""
    import subprocess
    import sys

    r = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True, text=True, timeout=300,
    )
    assert r.returncode == 0, (
        f"cold import of {module} failed — a module-level import hoisted out of "
        f"a function body has (re)introduced a circular import. Move the "
        f"offending import back into the function that uses it and raise that "
        f"file's INLINE_IMPORT_BUDGET with the cycle named in a comment.\n"
        f"{r.stderr[-2000:]}"
    )
