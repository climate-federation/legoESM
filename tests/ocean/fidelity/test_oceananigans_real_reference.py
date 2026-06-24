"""Real-data validation of the Oceananigans harness against a GENERATED oracle.

Unlike the synthetic round-trip tests, this loads an ACTUAL Oceananigans NetCDF
reference (produced by ``scripts/data/generate_oceananigans_barotropic_gyre_reference.jl``
run in a working Julia/Oceananigans env) and pushes it through the real io reader,
offline runner, and state bridge — proving the harness handles real Oceananigans
output (staggered coords, bottom-up z, NetCDF axis order).

Skipped (not failed) when no reference archive is present, so CI stays green
without the Julia toolchain; it activates automatically once the reference is
generated under ``$LEGOESM_OCEAN_FIDELITY_OCEANANIGANS_REF`` (or the cache).
"""

from __future__ import annotations

import numpy as np
import pytest

from legoesm.ocean.fidelity.oceananigans_runner import (
    load_oceananigans_reference,
    reference_root,
)

_CASE = "barotropic_gyre"


def _have_reference() -> bool:
    case_dir = reference_root() / _CASE
    return case_dir.is_dir() and any(case_dir.glob("*.nc"))


pytestmark = pytest.mark.skipif(
    not _have_reference(),
    reason=f"no Oceananigans {_CASE} reference archive (generate with "
           f"scripts/data/generate_oceananigans_{_CASE}_reference.jl)",
)


def test_real_reference_loads_through_runner():
    res = load_oceananigans_reference(_CASE)
    # The prognostic fields are present and finite.
    for f in ("u", "v"):
        assert f in res.variables, f"missing {f} in real reference"
        assert np.all(np.isfinite(res.variables[f])), f"{f} has non-finite values"
    # Time series is monotonic non-decreasing.
    t = np.asarray(res.times_s)
    assert t.ndim == 1 and t.size >= 1
    assert np.all(np.diff(t) >= 0)
    # Grid metadata is populated.
    assert res.grid_metadata["nx"] and res.grid_metadata["ny"]


def test_real_reference_bridges_into_legoesm_state():
    """The real snapshot bridges into a shape-matching legoESM state without
    error (validates the z-reverse + adaptive C-grid faces on REAL data). A
    GLOBAL grid of the oracle's (nx, ny) is used purely for shape-matching —
    the physical regional-geometry match is a separate recipe-card concern."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.fidelity.oceananigans_state_bridge import (
        oceananigans_snapshot_to_legoesm_state,
    )
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.vertical import create_ocean_z_star

    res = load_oceananigans_reference(_CASE)
    nx, ny = res.grid_metadata["nx"], res.grid_metadata["ny"]
    nz = max(1, res.grid_metadata["nz"])
    grid = create_latlon_grid(n_lat=ny, n_lon=nx)
    z_coord = create_ocean_z_star(nz, H_max=4000.0)
    base = rest_state_latlon_cgrid_ocean(grid, z_coord, land_lat_threshold=90.0)

    out = oceananigans_snapshot_to_legoesm_state(res, base)
    assert "u" in out.info["placed_fields"]
    assert "v" in out.info["placed_fields"]
    assert bool(np.all(np.isfinite(np.asarray(out.state.u.data))))
    assert bool(np.all(np.isfinite(np.asarray(out.state.v.data))))
