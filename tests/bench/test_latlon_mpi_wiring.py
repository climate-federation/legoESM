"""Guard the lat-lon MPI wiring in ``run_levante_gpu_scaling.py`` (issue #641).

The harness routes a multi-rank lat-lon run through
``make_latlon_mpi_step`` via several helpers that it imports *lazily*
inside ``run_benchmark`` (so JAX init order stays intact).  A lazy import
means a rename in ``legoesm.parallel.latlon_mpi`` /
``primitive_eq_latlon_cgrid`` would only break at runtime under
``mpirun`` — never on a normal CPU CI run.  This test pins the contract
cheaply (no JAX state, no MPI):

* lat-lon is in the validated MPI set, so the honest-sweep guard lets a
  multi-rank lat-lon run through instead of aborting;
* every symbol the harness's lat-lon MPI branch imports still exists;
* single-process lat-lon is still capped at 1 device (route-B SPMD is
  deliberately not wired — MPI is the only multi-rank lat-lon path).

The actual MPI == serial numerics are covered by
``tests/distributed/test_latlon_mpi_step.py``; this is the wiring tripwire.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

_HARNESS = (
    Path(__file__).resolve().parents[2]
    / "scripts" / "bench" / "run_levante_gpu_scaling.py"
)


def _load_harness():
    spec = importlib.util.spec_from_file_location(
        "run_levante_gpu_scaling", _HARNESS,
    )
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: the harness defines @dataclass types, and
    # dataclass processing resolves ``cls.__module__`` via sys.modules.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_latlon_in_mpi_supported_grids():
    mod = _load_harness()
    assert "latlon" in mod._MPI_SUPPORTED_GRIDS
    # icosahedral must remain — the lat-lon add is additive, not a swap.
    assert "icosahedral" in mod._MPI_SUPPORTED_GRIDS


def test_latlon_mpi_helpers_exist():
    """The symbols the harness lat-lon MPI branch imports lazily."""
    latlon_mpi = importlib.import_module("legoesm.parallel.latlon_mpi")
    for name in (
        "make_latlon_band_layout",
        "slice_latlon_grid_to_band",
        "scatter_state_latlon",
        "make_latlon_mpi_step",
    ):
        assert hasattr(latlon_mpi, name), f"latlon_mpi.{name} missing"

    cgrid = importlib.import_module(
        "legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid"
    )
    # The IC is a cell-centered HydrostaticState; the harness converts it
    # to the raw-array C-grid state make_latlon_mpi_step needs before scatter.
    assert hasattr(cgrid, "hydrostatic_to_cgrid")


def test_single_process_latlon_capped_at_one_device():
    """Route-B SPMD is not wired: single-process lat-lon stays at 1 device.

    (Multi-rank lat-lon scaling goes through MPI, where ``fixed_gpu_count``
    bypasses ``_valid_gpu_counts`` entirely.)
    """
    mod = _load_harness()
    assert mod._valid_gpu_counts(4, "latlon") == [1]


def test_baroclinic_ic_converts_and_scatters_to_cgrid():
    """The exact convert->scatter sequence the harness MPI branch relies on.

    ``baroclinic_wave_init_latlon`` yields a Field-wrapped, cell-centered
    ``HydrostaticState``; ``scatter_state_latlon`` raw-slices ``state.T[s:e]``
    and therefore REQUIRES a raw-array C-grid ``CGridLatLonHydrostaticState``.
    The harness bridges that gap with ``hydrostatic_to_cgrid`` BEFORE scatter
    (issue #641).  This guards that regression directly: drop the conversion
    (or scatter the Field state first) and the slice raises
    "'Field' object is not subscriptable" — exactly the bug this wiring hit.
    No MPI needed: a 1-rank band layout is a pure slice.
    """
    import jax.numpy as jnp
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        CGridLatLonHydrostaticState,
        hydrostatic_to_cgrid,
    )
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.parallel.latlon_mpi import (
        make_latlon_band_layout,
        scatter_state_latlon,
    )

    from legoesm import constants
    from tests.test_cases.baroclinic_wave import baroclinic_wave_init_latlon

    grid = create_latlon_grid(
        n_lat=8, radius=constants.R_earth, omega=constants.Omega,
    )
    sigma = create_sigma_coordinate(n_levels=3)
    state = baroclinic_wave_init_latlon(grid, sigma, perturbed=True, moist=False)

    # Convert on the GLOBAL grid, then slice — the harness's order.
    cgrid_state = hydrostatic_to_cgrid(state, grid)
    layout = make_latlon_band_layout(
        rank=0, n_ranks=1, n_lat=grid.n_lat, n_lon=grid.n_lon,
    )
    local = scatter_state_latlon(cgrid_state, layout)

    # The scattered state must be the raw-array C-grid type _step_cgrid wants.
    assert isinstance(local, CGridLatLonHydrostaticState)
    for field in ("u", "v", "T", "p_s"):
        leaf = getattr(local, field)
        assert isinstance(leaf, jnp.ndarray), f"{field} is not a raw array"
    # C-grid v is at lat interfaces: one extra row vs cell-centered T.
    assert local.v.shape[0] == local.T.shape[0] + 1


if __name__ == "__main__":
    test_latlon_in_mpi_supported_grids()
    test_latlon_mpi_helpers_exist()
    test_single_process_latlon_capped_at_one_device()
    test_baroclinic_ic_converts_and_scatters_to_cgrid()
    print("ok")
