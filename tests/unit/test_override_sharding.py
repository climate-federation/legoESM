"""Localize a GLOBAL per-column turbulence override to a rank's MPI tile.

Unit-covers ``legoesm.atmosphere.physics.turbulence.override_sharding`` — the
driver-path (lenient) slicer the dycore build calls so a deployed GLOBAL
per-column ``clubb_lite`` override reaches each rank's LOCAL columns.  The real
2-rank distributed validation lives in
``tests/distributed/test_latlon_mpi_override.py``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.atmosphere.physics.turbulence.config import (
    CLUBBLiteConfig,
    TurbulenceConfig,
)
from legoesm.atmosphere.physics.turbulence.override_sharding import (
    localize_turbulence_override,
)
from legoesm.parallel.latlon_mpi import (
    make_latlon_2d_layout,
    make_latlon_band_layout,
)

N_LAT, N_LON = 4, 4
NCOL = N_LAT * N_LON


def _global_override(ncol=NCOL):
    return TurbulenceConfig(scheme="clubb_lite", clubb_lite=CLUBBLiteConfig(
        C_K=jnp.arange(ncol, dtype=jnp.float64) * 0.01 + 0.3))


def _band(rank, n_ranks):
    return make_latlon_band_layout(rank=rank, n_ranks=n_ranks, n_lat=N_LAT, n_lon=N_LON)


def test_band_single_rank_identity():
    ov = _global_override()
    local = localize_turbulence_override(ov, _band(0, 1))
    np.testing.assert_allclose(
        np.asarray(local.clubb_lite.C_K), np.asarray(ov.clubb_lite.C_K))


def test_band_multirank_reassembles():
    ov = _global_override()
    parts = [np.asarray(localize_turbulence_override(ov, _band(r, 2)).clubb_lite.C_K)
             for r in range(2)]
    np.testing.assert_allclose(np.concatenate(parts), np.asarray(ov.clubb_lite.C_K))


def test_2d_pencil_slices():
    ov = _global_override()
    lay = make_latlon_2d_layout(3, 2, 2, N_LAT, N_LON)  # rank 3 = SE block
    local = np.asarray(localize_turbulence_override(ov, lay).clubb_lite.C_K)
    gidx = (np.arange(NCOL).reshape(N_LAT, N_LON)
            [lay.lat_start:lay.lat_end, lay.lon_start:lay.lon_end].reshape(-1))
    np.testing.assert_allclose(local, np.asarray(ov.clubb_lite.C_K)[gidx])


# --------------------------------------------------------------------------- #
# Lenient pass-through: anything that does NOT need slicing is returned verbatim.
# --------------------------------------------------------------------------- #
def test_scalar_override_passthrough():
    ov = TurbulenceConfig(scheme="clubb_lite", clubb_lite=CLUBBLiteConfig(C_K=0.7))
    assert localize_turbulence_override(ov, _band(0, 2)) is ov


def test_non_clubb_override_passthrough():
    ov = TurbulenceConfig(scheme="mynn25")
    assert localize_turbulence_override(ov, _band(0, 2)) is ov


def test_already_local_override_passthrough():
    # A per-column field NOT at the global ncol (e.g. already a rank's 8 columns)
    # is left as-is — the localizer is idempotent / safe to call repeatedly.
    ov = _global_override(ncol=8)            # 8 != 16 (the layout's global ncol)
    assert localize_turbulence_override(ov, _band(0, 2)) is ov


def test_none_passthrough():
    assert localize_turbulence_override(None, _band(0, 2)) is None


# --------------------------------------------------------------------------- #
# Loud failures.
# --------------------------------------------------------------------------- #
class _CubedLayout:
    """Stand-in for a non-lat-lon (cubed-sphere) distributed layout."""


def test_cubed_sphere_override_passes_through():
    # Auto-localization is lat-lon-only: a non-lat-lon layout passes the override
    # THROUGH (honoring a pre-sliced escape-hatch override); a global one fails
    # loudly downstream in broadcast_column_param, not here.
    ov = _global_override()
    assert localize_turbulence_override(ov, _CubedLayout()) is ov


def test_cubed_sphere_scalar_override_passthrough():
    # A scalar override under a cubed layout also passes through (nothing to slice).
    ov = TurbulenceConfig(scheme="clubb_lite", clubb_lite=CLUBBLiteConfig(C_K=0.7))
    assert localize_turbulence_override(ov, _CubedLayout()) is ov


def test_inconsistent_lengths_raise():
    ov = TurbulenceConfig(scheme="clubb_lite", clubb_lite=CLUBBLiteConfig(
        C_K=jnp.ones(NCOL), Pr_t=jnp.ones(8)))   # one global, one not
    with pytest.raises(ValueError, match="inconsistent column counts"):
        localize_turbulence_override(ov, _band(0, 2))
