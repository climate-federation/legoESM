"""#1441: the streamfunction diagnostics must use the MODEL's face rules.

Two defects, both in the incumbent ``diagnostics_streamfunction``:

1. ``_v_face_geometry`` built the v-face thickness with an ARITHMETIC MEAN and
   the face mask as a 2-D land product, while the model advects with the MIN
   rule (``min_cell_to_vface``) and per-LEVEL face masks. On eORCA1 the two
   disagreed by up to 1.9 Sv at a single zonal section — the size of the signal.
2. ``barotropic_streamfunction`` weighted every u-face of a row by the 1-D
   ``grid.dy``, which is wrong on a tripole where the bipolar cap makes dy vary
   strongly with longitude.

Each test below is written to FAIL against the old rule, not merely to pass
against the new one.
"""

from __future__ import annotations

import numpy as np

from legoesm import constants
from legoesm.ocean.diagnostics_streamfunction import (
    _v_face_geometry,
    barotropic_streamfunction,
    moc_streamfunction,
)


class _FakeGrid:
    def __init__(self, radius=constants.R_earth):
        self.radius = radius


def _step_bathymetry(n_lat=6, n_lon=4, nlev=3):
    """Adjacent rows with DIFFERENT thickness — where min != mean."""
    h = np.full((n_lat, n_lon, nlev), 100.0)
    h[n_lat // 2:, :, -1] = 10.0     # a shallower shelf to the north
    mask = np.ones((n_lat, n_lon))
    return h, mask


def test_v_face_thickness_is_the_min_rule_not_the_mean():
    h, mask = _step_bathymetry()
    n_lat, n_lon, nlev = h.shape
    v = np.zeros((n_lat + 1, n_lon, nlev))
    _dx, h_v, _m, _lat = _v_face_geometry(v, h, mask, _FakeGrid())

    j = n_lat // 2                    # the face straddling the step
    got = h_v[j, :, -1]
    expect_min = np.minimum(h[j - 1, :, -1], h[j, :, -1])
    expect_mean = 0.5 * (h[j - 1, :, -1] + h[j, :, -1])
    np.testing.assert_allclose(got, expect_min, rtol=1e-12)
    # Non-vacuity: the two rules genuinely differ at this face.
    assert not np.allclose(expect_min, expect_mean), "test bathymetry is flat"


def test_v_face_mask_is_per_level_not_a_2d_land_product():
    """Columns with different bottom_level: the face must be dry below the
    shallower seafloor, which a 2-D mask[j-1]*mask[j] cannot express."""
    n_lat, n_lon, nlev = 6, 4, 3
    h = np.full((n_lat, n_lon, nlev), 100.0)
    h[n_lat // 2:, :, -1] = 0.0       # north columns end one level higher
    mask = np.ones((n_lat, n_lon))
    v = np.zeros((n_lat + 1, n_lon, nlev))
    _dx, _h_v, v_mask, _lat = _v_face_geometry(v, h, mask, _FakeGrid())

    assert v_mask.ndim == 3 and v_mask.shape[-1] == nlev, (
        f"expected a per-level face mask, got shape {v_mask.shape}")
    j = n_lat // 2
    assert float(v_mask[j, 0, -1]) == 0.0, (
        "the bottom face at the step must be DRY; a 2-D land product would "
        "leave it wet and let MOC integrate through the seafloor")
    assert float(v_mask[j, 0, 0]) > 0.0, "the surface face must stay wet"


def test_moc_transport_matches_the_models_min_rule_flux():
    """psi at the bottom == the min-rule zonal-section transport."""
    h, mask = _step_bathymetry()
    n_lat, n_lon, nlev = h.shape
    rng = np.random.default_rng(0)
    v = rng.normal(scale=0.01, size=(n_lat + 1, n_lon, nlev))
    grid = _FakeGrid()

    psi = moc_streamfunction(v, h, np.zeros((n_lat, n_lon)),
                             np.full((n_lat, n_lon), 500.0), mask, grid)

    # Reference built INDEPENDENTLY from the model's operators, not from
    # _v_face_geometry — otherwise both sides could move together to the same
    # wrong non-mean rule and the test would still pass (codex P3).
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        compute_face_masks_3d, min_cell_to_vface,
    )
    h_v_ref = np.asarray(min_cell_to_vface(jnp.asarray(h), grid))
    active = jnp.asarray(((mask[:, :, None] > 0) & (h > 0)), dtype=h_v_ref.dtype)
    _um, v_mask_ref = compute_face_masks_3d(active, grid)
    v_mask_ref = np.asarray(v_mask_ref)
    dx_v, h_v, v_mask, _lat = _v_face_geometry(v, h, mask, grid)
    np.testing.assert_allclose(h_v, h_v_ref, rtol=1e-12)
    np.testing.assert_allclose(v_mask, v_mask_ref, rtol=1e-12)
    ref = -(v * h_v_ref * v_mask_ref
            * dx_v[:, :, None]).sum(axis=1).cumsum(axis=1) / 1e6
    np.testing.assert_allclose(psi, ref, rtol=1e-12)

    # And the mean rule would give a DIFFERENT answer at the step row.
    h_v_mean = np.zeros_like(v)
    h_v_mean[1:-1] = 0.5 * (h[:-1] + h[1:])
    ref_mean = -(v * h_v_mean * v_mask
                 * dx_v[:, :, None]).sum(axis=1).cumsum(axis=1) / 1e6
    assert not np.allclose(psi[n_lat // 2], ref_mean[n_lat // 2]), (
        "min and mean coincide here — the test would not detect a regression")


class _TripoleProxy:
    """Minimal stand-in exposing the 2-D dy_u a tripole carries."""

    def __init__(self, n_lat, n_lon):
        self.radius = constants.R_earth
        self.dlon = 0.0                       # tripole sentinel
        self.dy = np.full((n_lat,), 1.0e5)
        # dy varies strongly with LONGITUDE, as it does on the bipolar cap.
        self.dy_u = np.linspace(0.5e5, 1.5e5, n_lon + 1)[None, :] * np.ones(
            (n_lat, 1))
        self.dx_v = np.full((n_lat + 1, n_lon), 1.0e5)
        self.fold = None


def test_bsf_uses_the_2d_dy_on_a_tripole(monkeypatch):
    """SCOPE (codex): this pins the BRANCH and the absence of a spurious 0.5.
    It monkeypatches ``_is_tripolar``, so it cannot catch a regression in
    tripole RECOGNITION or in real FoldDescriptor handling — that needs a built
    tripole grid, which this unit test deliberately does not construct.
    """
    import legoesm.ocean.diagnostics_streamfunction as ds

    # The proxy must NOT be recognised as a tripole on its own, or the
    # monkeypatch below would be a no-op and the comparison vacuous.
    assert ds._is_tripolar(_TripoleProxy(4, 4)) is False
    monkeypatch.setattr(ds, "_is_tripolar", lambda g: True)

    n_lat, n_lon, nlev = 5, 6, 2
    grid = _TripoleProxy(n_lat, n_lon)
    h = np.full((n_lat, n_lon, nlev), 50.0)
    mask = np.ones((n_lat, n_lon))
    u = np.full((n_lat, n_lon + 1, nlev), 0.1)

    psi_2d = barotropic_streamfunction(u, h, mask, grid)

    monkeypatch.setattr(ds, "_is_tripolar", lambda g: False)
    psi_1d = barotropic_streamfunction(u, h, mask, grid)

    assert not np.allclose(psi_2d, psi_1d), (
        "the 2-D dy_u made no difference — either it is not being read or the "
        "proxy's dy_u does not vary in longitude, and #1441's second finding "
        "would be undetectable")
    # The 2-D result must equal the explicit per-face weighting.
    U_dz = np.sum(u * np.concatenate(
        [np.minimum(h, np.roll(h, 1, axis=1)),
         np.minimum(h, np.roll(h, 1, axis=1))[:, 0:1, :]], axis=1), axis=-1)
    expect = -np.cumsum(U_dz * grid.dy_u, axis=0)[:, :-1] / 1e6
    np.testing.assert_allclose(psi_2d, expect, rtol=1e-12)
