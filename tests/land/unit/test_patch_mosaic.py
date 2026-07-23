"""Unit tests for the N-patch canopy mosaic config + area-weighted aggregation.

Pure (no canopy solve): exercises ``PatchMosaicConfig.validate`` dispatch
hardening, the ``savanna_two_patch`` factory, and ``_area_weight`` reduction math.
The mosaic-vs-canopy degenerate-identity check (needs a real canopy solve) lives
in ``tests/land/integration/test_ec_site_run.py``.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.land.surface_scheme.base import SurfaceFluxOutput
from legoesm.land.surface_scheme.patch_mosaic import (
    PatchMosaicConfig,
    PatchSpec,
    _area_weight,
    savanna_two_patch,
)


def test_validate_accepts_unit_and_two_patch():
    PatchMosaicConfig(patches=(PatchSpec(frac=1.0),)).validate()
    PatchMosaicConfig(patches=(PatchSpec(frac=0.4, fc4=0.0),
                               PatchSpec(frac=0.6, fc4=1.0))).validate()


@pytest.mark.parametrize("patches", [
    (),                                              # no patches
    (PatchSpec(frac=0.4), PatchSpec(frac=0.4)),      # sum 0.8 != 1
    (PatchSpec(frac=0.0), PatchSpec(frac=1.0)),      # zero fraction
    (PatchSpec(frac=-0.1), PatchSpec(frac=1.1)),     # negative fraction
    (PatchSpec(frac=1.0, fc4=1.5),),                 # fc4 out of [0,1]
    (PatchSpec(frac=1.0, vcmax_c4_scale=-1.0),),     # negative scale
    (PatchSpec(frac=float("nan")),),                 # non-finite fraction
])
def test_validate_raises_on_bad_mosaic(patches):
    with pytest.raises(ValueError):
        PatchMosaicConfig(patches=patches).validate()


def test_savanna_two_patch_shapes_and_split():
    m = savanna_two_patch(tree_frac=0.4)
    assert len(m.patches) == 2
    tree, grass = m.patches
    assert tree.frac == pytest.approx(0.4) and grass.frac == pytest.approx(0.6)
    assert tree.fc4 == 0.0 and grass.fc4 == 1.0        # C3 trees, C4 grass
    for bad in (0.0, 1.0, -0.2):
        with pytest.raises(ValueError):
            savanna_two_patch(tree_frac=bad)


def _fake_output(shflx, gpp, T_surface=300.0, emissivity=0.97, n_iters=None):
    """A minimal SurfaceFluxOutput over a patch axis; None diagnostic fields."""
    n = len(shflx)
    z = jnp.zeros(n)
    br = lambda v: jnp.asarray(v, dtype=float) * jnp.ones(n) \
        if np.ndim(v) == 0 else jnp.asarray(v, dtype=float)
    return SurfaceFluxOutput(
        shflx=jnp.asarray(shflx, dtype=float), lhflx=z, tau_x=z, tau_y=z,
        sw_net=z, lw_net=z, lw_up=z, G_soil=z,
        T_surface=br(T_surface), q_surface=z, albedo=jnp.full(n, 0.2),
        emissivity=br(emissivity), z0=jnp.full(n, 0.1),
        gpp=jnp.asarray(gpp, dtype=float),
        n_iters=None if n_iters is None else jnp.asarray(n_iters))


def test_area_weight_single_patch_is_identity():
    out = _fake_output([12.0], [5.0])
    agg = _area_weight(out, jnp.asarray([1.0]))
    assert float(agg.shflx[0]) == pytest.approx(12.0)
    assert float(agg.gpp[0]) == pytest.approx(5.0)
    assert agg.shflx.shape == (1,)


def test_area_weight_sums_fluxes_and_averages_state():
    out = _fake_output([10.0, 30.0], [4.0, 8.0])
    agg = _area_weight(out, jnp.asarray([0.25, 0.75]))
    # per-area flux: area-weighted mean = 0.25*10 + 0.75*30 = 25
    assert float(agg.shflx[0]) == pytest.approx(25.0)
    assert float(agg.gpp[0]) == pytest.approx(0.25 * 4 + 0.75 * 8)
    # intensive state held equal across patches -> unchanged
    assert float(agg.T_surface[0]) == pytest.approx(300.0)
    assert float(agg.albedo[0]) == pytest.approx(0.2)
    # None diagnostic fields pass through
    assert agg.gs_Sun is None


def test_area_weight_equal_patches_equal_single():
    """Two identical patches (any split) == one patch with those values."""
    two = _area_weight(_fake_output([20.0, 20.0], [6.0, 6.0]),
                       jnp.asarray([0.3, 0.7]))
    one = _area_weight(_fake_output([20.0], [6.0]), jnp.asarray([1.0]))
    assert float(two.shflx[0]) == pytest.approx(float(one.shflx[0]))
    assert float(two.gpp[0]) == pytest.approx(float(one.gpp[0]))


def test_area_weight_T_surface_is_radiometric():
    """T_surface reduces so eps*sigma*T^4 stays consistent with the summed lw_up,
    NOT a linear mean; single patch is exact identity."""
    fr = jnp.asarray([0.4, 0.6])
    out = _fake_output([0.0, 0.0], [0.0, 0.0],
                       T_surface=[280.0, 320.0], emissivity=[0.95, 0.99])
    agg = _area_weight(out, fr)
    fw = np.asarray(fr)
    eps = np.asarray([0.95, 0.99]); T = np.asarray([280.0, 320.0])
    T_expect = (np.sum(fw * eps * T ** 4) / np.sum(fw * eps)) ** 0.25
    assert float(agg.T_surface[0]) == pytest.approx(T_expect)
    assert float(agg.T_surface[0]) != pytest.approx(np.sum(fw * T))  # not linear
    # single patch: radiometric reconstruction is the identity
    one = _area_weight(_fake_output([0.0], [0.0], T_surface=[295.0]),
                       jnp.asarray([1.0]))
    assert float(one.T_surface[0]) == pytest.approx(295.0)


def test_area_weight_n_iters_is_max_not_weighted():
    """n_iters is a solver count -> reduced by max (stays an integer count)."""
    agg = _area_weight(_fake_output([0.0, 0.0], [0.0, 0.0], n_iters=[3, 7]),
                       jnp.asarray([0.5, 0.5]))
    assert int(agg.n_iters[0]) == 7
