"""Unit test for ERA5 radiation-flux target packing (``_era5_held_fluxes``).

Covers the zeros fallback (fluxes not loaded → byte-identical legacy) and
the regrid + return-order contract (rsut, OLR, sfc_net_sw, sfc_net_lw).
The ERA5→W/m² derivation arithmetic in ``load_era5_slice`` is verified on
the real WB2 store by the T21 smoke (needs GCS); this keeps the pure logic
covered offline.  Runnable directly or under pytest.
"""

from __future__ import annotations

import numpy as np

from legoesm.training.era5_to_state import ERA5Slice, _era5_held_fluxes

SHAPE_2D = (4, 8)


def _slice(with_fluxes: bool) -> ERA5Slice:
    z3 = np.zeros((4, 8, 3), dtype=np.float32)
    z2 = np.zeros((4, 8), dtype=np.float32)
    lat = np.linspace(-1.5, 1.5, 4)
    lon = np.linspace(0.0, 6.0, 8)
    kw = {}
    if with_fluxes:
        kw = dict(
            rsut=np.full((4, 8), 100.0, np.float32),
            olr=np.full((4, 8), 240.0, np.float32),
            sfc_net_sw=np.full((4, 8), 165.0, np.float32),
            sfc_net_lw=np.full((4, 8), -55.0, np.float32),
        )
    return ERA5Slice(
        T=z3, u=z3, v=z3, q=z3, p_s=z2, sst=z2, phis=z2,
        lat=lat, lon=lon, plev_Pa=np.array([300.0, 500.0, 850.0]), **kw
    )


def test_no_fluxes_returns_zeros():
    out = _era5_held_fluxes(_slice(False), lambda f: f, SHAPE_2D)
    assert len(out) == 4
    for a in out:
        assert a.shape == SHAPE_2D
        assert float(np.abs(np.asarray(a)).max()) == 0.0


def test_fluxes_regrid_and_order():
    """Identity regrid → helper returns (rsut, OLR, sfc_net_sw, sfc_net_lw)."""
    rsut, olr, snsw, snlw = _era5_held_fluxes(
        _slice(True), lambda f: f, SHAPE_2D
    )
    assert np.allclose(np.asarray(rsut), 100.0)
    assert np.allclose(np.asarray(olr), 240.0)
    assert np.allclose(np.asarray(snsw), 165.0)
    assert np.allclose(np.asarray(snlw), -55.0)


def test_regrid_callback_applied():
    """The regrid callback is actually invoked (scale by 2)."""
    rsut, *_ = _era5_held_fluxes(_slice(True), lambda f: f * 2.0, SHAPE_2D)
    assert np.allclose(np.asarray(rsut), 200.0)


def _run_all():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
    print("era5_flux_targets: all self-checks passed")


if __name__ == "__main__":
    _run_all()
