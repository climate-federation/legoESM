"""Direct tests for the coupled land calibrator's pure parameter machinery
(scripts/run/train_coupled_land_era5.py).  The end-to-end coupled gradient is
covered by tests/unit/test_multilayer_land_driver.py::
test_build_training_segment_land_gradient; here we test the leaf helpers."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

import scripts.run.train_coupled_land_era5 as C


def test_train_keys_are_the_four_coupled_params():
    assert C._TRAIN_KEYS == ("pft_z0", "pft_vcmax", "pft_g1", "pft_lcma")


def test_init_params4_has_only_the_four_groups():
    p = C.init_params4()
    assert set(p) == set(C._TRAIN_KEYS)
    for k in C._TRAIN_KEYS:
        assert np.asarray(p[k]).shape == (17,)        # per-PFT (17 CLM5 PFTs)


def test_constrain4_maps_into_physical_bounds():
    # raw values spanning a wide range must land inside each group's bounds
    raw = {k: jnp.linspace(-8.0, 8.0, 17) for k in C._TRAIN_KEYS}
    cp = C.constrain4(raw)
    for k in C._TRAIN_KEYS:
        lo, hi = C.ML.BOUNDS_EXT[k]
        v = np.asarray(cp[k])
        assert np.all(v >= lo - 1e-6) and np.all(v <= hi + 1e-6), k
    # roughness bound sanity (forests up to ~2-3 m, never negative)
    assert C.ML.BOUNDS_EXT["pft_z0"][0] > 0.0


def test_build_land_params_overrides_only_the_four_fields():
    from legoesm.land.surface_params import LandSurfaceParams
    ncol = 5
    o = jnp.ones(ncol)
    # a distinctive sentinel base so we can detect which fields change
    base = LandSurfaceParams(
        albedo_veg=0.11 * o, emissivity=0.97 * o, z0=0.5 * o, W_max=150.0 * o,
        C_soil=2.0e6 * o, d_soil=1.0 * o, root_depth=1.0 * o, theta_wp=0.12 * o,
        theta_fc=0.30 * o, Vc_max25=50.0 * o, LCMA=50.0 * o, g1=9.0 * o)
    pft = jnp.eye(17)[jnp.zeros(ncol, dtype=int)]      # all bare-soil PFT (ncol,17)
    lp = C.build_land_params(C.init_params4(), base, pft)

    changed = {"z0", "Vc_max25", "g1", "LCMA"}
    for f in base._fields:
        if getattr(base, f) is None:
            continue  # optional prescribed field (e.g. LAI) not set on this base
        same = np.allclose(np.asarray(getattr(lp, f)), np.asarray(getattr(base, f)))
        if f in changed:
            # overridden from the per-PFT trainable set (may coincide, but shape holds)
            assert np.asarray(getattr(lp, f)).shape == (ncol,)
        else:
            assert same, f"{f} must be untouched by the coupled calibrator"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
