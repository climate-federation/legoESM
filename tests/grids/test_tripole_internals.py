"""Direct unit tests for tripole internals.

Addresses PR #268 slopbuster REJECT #2: ``_detect_fold``,
``_compute_rotation_angles``, ``_read_nemo_mesh_mask`` were reachable
only via the ``run_global_overturning_tripole.py`` script.  These tests
exercise each one against a synthetic NEMO-style mesh.
"""

from __future__ import annotations

import os
import tempfile

import jax.numpy as jnp
import numpy as np
import pytest


# -------------------------------------------------------------------------
# _detect_fold
# -------------------------------------------------------------------------


class TestDetectFold:
    def test_ideal_symmetric_fold(self):
        from legoesm.grids.tripole import _detect_fold

        n_lat, n_lon = 16, 32
        lon = jnp.linspace(0.0, 360.0, n_lon, endpoint=False)
        lat = jnp.linspace(-80.0, 60.0, n_lat)
        glamt = jnp.broadcast_to(lon[None, :], (n_lat, n_lon))

        gphit = jnp.broadcast_to(lat[:, None], (n_lat, n_lon))
        # Force the fold row to be perfectly symmetric (constant in lon)
        gphit = gphit.at[-1].set(60.0)

        fold = _detect_fold(glamt, gphit, n_lat, n_lon)

        assert fold.is_active is True
        assert fold.fold_j == n_lat - 1
        # perm_T[i] = n_lon - 1 - i
        assert jnp.all(fold.perm_T == jnp.arange(n_lon - 1, -1, -1))
        assert jnp.all(fold.perm_v == fold.perm_T)
        assert fold.vector_sign_u == -1.0
        assert fold.vector_sign_v == -1.0

    def test_asymmetric_fold_raises(self):
        from legoesm.grids.tripole import _detect_fold

        n_lat, n_lon = 16, 32
        lon = jnp.linspace(0.0, 360.0, n_lon, endpoint=False)
        lat = jnp.linspace(-80.0, 60.0, n_lat)
        glamt = jnp.broadcast_to(lon[None, :], (n_lat, n_lon))
        gphit = jnp.broadcast_to(lat[:, None], (n_lat, n_lon))
        # Break the fold-row symmetry: monotonically rising in i
        gphit = gphit.at[-1].set(jnp.linspace(50.0, 70.0, n_lon))

        with pytest.raises(ValueError, match="Fold symmetry check failed"):
            _detect_fold(glamt, gphit, n_lat, n_lon)

    def test_asymmetry_threshold_kwarg_loosens_check(self):
        from legoesm.grids.tripole import _detect_fold

        n_lat, n_lon = 8, 16
        glamt = jnp.zeros((n_lat, n_lon))
        gphit = jnp.broadcast_to(
            jnp.linspace(-70.0, 60.0, n_lat)[:, None], (n_lat, n_lon),
        )
        # Inject a 0.5 deg lat bump that the default 0.1 tolerance rejects
        gphit = gphit.at[-1].set(60.0)
        gphit = gphit.at[-1, 0].set(60.5)

        with pytest.raises(ValueError):
            _detect_fold(glamt, gphit, n_lat, n_lon)
        # Loosening the threshold lets it through.
        fold = _detect_fold(
            glamt, gphit, n_lat, n_lon, max_fold_asym_deg=1.0,
        )
        assert fold.is_active is True


# -------------------------------------------------------------------------
# _compute_rotation_angles
# -------------------------------------------------------------------------


class TestComputeRotationAngles:
    def test_below_cap_is_identity(self):
        from legoesm.grids.tripole import _compute_rotation_angles

        n_lat, n_lon = 12, 24
        # Regular lat-lon coordinates → no rotation expected
        lon = jnp.linspace(0.0, 360.0, n_lon, endpoint=False)
        lat = jnp.linspace(-80.0, 80.0, n_lat)
        glam = jnp.broadcast_to(lon[None, :], (n_lat, n_lon))
        gphi = jnp.broadcast_to(lat[:, None], (n_lat, n_lon))

        # cap_j past the end → all rows below cap
        cos_au, sin_au, cos_av, sin_av = _compute_rotation_angles(
            glam, gphi, glam, gphi, cap_j=n_lat,
        )
        assert jnp.allclose(cos_au, 1.0)
        assert jnp.allclose(sin_au, 0.0)
        assert jnp.allclose(cos_av, 1.0)
        assert jnp.allclose(sin_av, 0.0)

    def test_above_cap_uses_local_grid_orientation(self):
        from legoesm.grids.tripole import _compute_rotation_angles

        n_lat, n_lon = 12, 24
        lon = jnp.linspace(0.0, 360.0, n_lon, endpoint=False)
        lat = jnp.linspace(-80.0, 60.0, n_lat)
        glam = jnp.broadcast_to(lon[None, :], (n_lat, n_lon))
        gphi = jnp.broadcast_to(lat[:, None], (n_lat, n_lon))

        cos_au, sin_au, _, _ = _compute_rotation_angles(
            glam, gphi, glam, gphi, cap_j=n_lat // 2,
        )
        # In the no-cap (regular lat-lon) limit the local i-axis is east,
        # so cos≈1, sin≈0 even above the cap row.
        assert jnp.allclose(cos_au, 1.0, atol=1e-5)
        assert jnp.allclose(sin_au, 0.0, atol=1e-5)


# -------------------------------------------------------------------------
# _read_nemo_mesh_mask  (skip without netCDF4)
# -------------------------------------------------------------------------


netcdf4 = pytest.importorskip("netCDF4")


def _write_synthetic_mesh_mask(path: str, n_lat: int = 8, n_lon: int = 16) -> None:
    """Write a minimal NEMO-style mesh_mask.nc file."""
    ds = netcdf4.Dataset(path, "w")
    ds.createDimension("y", n_lat)
    ds.createDimension("x", n_lon)

    glamt = np.broadcast_to(
        np.linspace(0.0, 360.0, n_lon, endpoint=False)[None, :],
        (n_lat, n_lon),
    ).astype(np.float64)
    gphit = np.broadcast_to(
        np.linspace(-80.0, 80.0, n_lat)[:, None], (n_lat, n_lon),
    ).astype(np.float64)
    ones = np.ones((n_lat, n_lon), dtype=np.float64)

    for name, arr in (
        ("glamt", glamt), ("gphit", gphit),
        ("glamu", glamt), ("gphiu", gphit),
        ("glamv", glamt), ("gphiv", gphit),
        ("e1t", ones * 1.0e4), ("e2t", ones * 1.0e4),
        ("e1u", ones * 1.0e4), ("e2u", ones * 1.0e4),
        ("e1v", ones * 1.0e4), ("e2v", ones * 1.0e4),
        ("tmask", ones), ("umask", ones), ("vmask", ones),
    ):
        v = ds.createVariable(name, "f8", ("y", "x"))
        v[:] = arr
    ds.close()


class TestReadNemoMeshMask:
    def test_reads_2d_vars(self):
        from legoesm.grids.tripole import _read_nemo_mesh_mask

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "mesh.nc")
            _write_synthetic_mesh_mask(path, n_lat=8, n_lon=16)
            raw = _read_nemo_mesh_mask(path)

        for name in ("glamt", "gphit", "e1t", "e2t",
                     "e1u", "e2u", "e1v", "e2v",
                     "tmask", "umask", "vmask"):
            assert name in raw, f"missing {name}"
            assert raw[name].shape == (8, 16)

    def test_squeezes_extra_singleton_axes(self):
        """NEMO mesh files often carry singleton time/depth axes; ensure
        they are squeezed to plain 2D."""
        from legoesm.grids.tripole import _read_nemo_mesh_mask

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "mesh.nc")
            ds = netcdf4.Dataset(path, "w")
            ds.createDimension("t", 1)
            ds.createDimension("z", 1)
            ds.createDimension("y", 4)
            ds.createDimension("x", 8)
            v = ds.createVariable("glamt", "f8", ("t", "z", "y", "x"))
            v[:] = np.zeros((1, 1, 4, 8))
            v = ds.createVariable("gphit", "f8", ("t", "z", "y", "x"))
            v[:] = np.zeros((1, 1, 4, 8))
            for name in ("e1t", "e2t", "e1u", "e2u", "e1v", "e2v"):
                v = ds.createVariable(name, "f8", ("y", "x"))
                v[:] = np.ones((4, 8))
            ds.close()

            raw = _read_nemo_mesh_mask(path)
            assert raw["glamt"].shape == (4, 8)
            assert raw["gphit"].shape == (4, 8)
