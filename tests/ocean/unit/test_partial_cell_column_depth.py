"""The column depth returned by ``make_partial_cell_latlon`` equals the sum of its layer thicknesses.

The barotropic step uses H_bathy while the tracers and the baroclinic step use the layer thicknesses, so the two
must agree, as MITgcm does by recomputing the bottom from the lopped cells (``R_low = rF(1) - sum drF hFacC``,
ini_masks_etc.F) and MOM6 by clipping the depth to MAXIMUM_DEPTH (limit_topography).  Two cases broke it: a
column deeper than the reference grid (the bottom cell is capped at its reference thickness, the depth was not)
and a column within 1e-5 of a reference interface (the near-full snap thickens the bottom cell, the depth was
not).  A column of any other depth keeps its depth exactly.
"""
import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from legoesm.ocean.init_latlon_cgrid import make_partial_cell_latlon  # noqa: E402
from legoesm.ocean.vertical import compute_layer_thickness, create_z_star_from_thicknesses  # noqa: E402

DZ = np.array([10.0, 20.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1420.0])    # bottom 3000 m


def _check(H_in):
    zc = create_z_star_from_thicknesses(DZ)
    H_in = np.asarray(H_in, dtype=float)
    ocean = (H_in > 0).astype(float)
    zc_pc, H, lm = make_partial_cell_latlon(zc, H_in, ocean, smoothing_passes=0, min_levels=2)
    wet = lm > 0.5
    s = np.asarray(compute_layer_thickness(jnp.zeros_like(jnp.asarray(H)), jnp.asarray(H), zc_pc)).sum(-1)
    return H, s, wet


def test_column_deeper_than_the_reference_grid_is_cut_to_its_layers():
    H, s, wet = _check([[3000.0 + 250.0, 3400.0, 2500.0]])
    assert wet.all()
    assert np.max(np.abs(s - H)[wet] / H[wet]) < 1e-12, (H, s)
    assert np.allclose(H[0, :2], 3000.0)                  # cut at the reference bottom


def test_column_near_a_reference_interface_keeps_its_depth_equal_to_the_layers():
    near = 180.0 + 200.0 * (1 - 4e-6)                     # inside the near-full snap of the 180 to 380 m level
    H, s, wet = _check([[near, 1000.0 - 1e-3, 1234.5]])
    assert np.max(np.abs(s - H)[wet] / H[wet]) < 1e-12, (H, s)


def test_other_columns_keep_their_depth_exactly():
    H_in = np.array([[123.4, 777.0, 2999.0, 1500.25]])
    H, s, wet = _check(H_in)
    assert np.array_equal(H[wet], H_in[wet])              # bit for bit
    assert np.max(np.abs(s - H)[wet] / H[wet]) < 1e-12
