"""End-to-end MITgcm oracle integration test against a REAL generated reference.

Opt-in: runs only when ``$LEGOESM_OCEAN_FIDELITY_MITGCM_REF`` points at a
directory containing a generated ``barotropic_gyre/`` reference (produced by
``scripts/data/generate_mitgcm_barotropic_gyre_reference.py`` on a machine with a
Fortran toolchain). Skipped otherwise, so CI without MITgcm stays green while a
developer who has built the reference gets a genuine reader+runner+bridge+monitor
check against real MITgcm bytes.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

_REF_ROOT = os.environ.get("LEGOESM_OCEAN_FIDELITY_MITGCM_REF")
_CASE_DIR = Path(_REF_ROOT) / "barotropic_gyre" if _REF_ROOT else None

pytestmark = pytest.mark.skipif(
    _CASE_DIR is None or not _CASE_DIR.is_dir(),
    reason="no generated MITgcm barotropic_gyre reference "
    "($LEGOESM_OCEAN_FIDELITY_MITGCM_REF unset or empty)",
)


def test_real_reference_loads_and_bridges():
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.fidelity import mitgcm_runner
    from legoesm.ocean.fidelity.mitgcm_state_bridge import (
        mitgcm_snapshot_to_legoesm_state,
    )
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.vertical import create_ocean_z_star

    res = mitgcm_runner.load_mitgcm_reference("barotropic_gyre")
    assert set(res.variables) >= {"U", "V", "Eta"}
    nx, ny = res.grid_metadata["nx"], res.grid_metadata["ny"]

    grid = create_latlon_grid(n_lat=ny, n_lon=nx)
    z = create_ocean_z_star(max(1, res.grid_metadata["nz"]), H_max=5000.0)
    base = rest_state_latlon_cgrid_ocean(grid, z, land_lat_threshold=90.0)
    out = mitgcm_snapshot_to_legoesm_state(res, base, cyclic_x=False)
    # West-face velocities land 1:1; legoESM carries the extra east-wall face.
    assert out.state.u.data.shape[1] == nx + 1
    assert {"u", "v", "eta"} <= set(out.info["placed_fields"])


def test_field_dump_agrees_with_monitor_diagnostic():
    """The strongest correctness check: the global max of the Eta FIELD I read
    must equal MITgcm's own monitor dynstat_eta_max (computed independently)."""
    from legoesm.ocean.fidelity import mitgcm_monitor, mitgcm_runner

    res = mitgcm_runner.load_mitgcm_reference("barotropic_gyre")
    mon = mitgcm_monitor.parse_monitor_file(_CASE_DIR / "output.txt")
    field_eta_max = float(np.max(res.variables["Eta"]))
    # float32 field dump vs MITgcm's internal monitor reduction -> ~1e-7 rtol.
    np.testing.assert_allclose(
        field_eta_max, mon.final("dynstat_eta_max"), rtol=1e-5
    )
    field_uvel_max = float(np.max(res.variables["U"]))
    np.testing.assert_allclose(
        field_uvel_max, mon.final("dynstat_uvel_max"), rtol=1e-5
    )
