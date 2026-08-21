"""The NEMO bridge must build its vertical grid from e3t_0, not e3t_1d.

NEMO integrates with the 3-D scale factors ``e3t_0`` (key_vco_3d). ``e3t_1d`` is
a different, UNSTRETCHED reference ladder. For DINO they agree in the upper
ocean and part company below the ~1000 m re-anchor (first level differing by
more than 1% is k=25, top face 982.4 m), by up to 70.4 m at the deepest wet
level -- 12.9% of e3t_0, 14.8% of e3t_1d. Both sum to the same 4000.000 m over
the wet levels, so this is a REDISTRIBUTION of thickness, not a change of
domain depth.

Using e3t_1d put legoESM's abyssal layers 0.9-14.8% off (of e3t_1d; 0.9-12.9%
of e3t_0) over k=25..34, and left the bottom of
its PARTIAL-DEPTH columns misplaced: identical in the 75% of wet columns that
reach all 35 levels, but 70.4-104.2 m too deep in the other 25%, a 21.9 m mean
over all wet columns -- and 4.70e-03 relative on total wet volume. That is
precisely the depth range where #1226's ACC deficit is sourced (80% of the
missing thermal wind below 2000 m). Thermal wind integrates density x
THICKNESS, so this corrupted the quantity the deficit is measured in.
(Numbers re-measured 2026-08-21 from RUN_TRAJ/mesh_mask.nc, #1455.)

The bug was invisible for GYRE (key_linssh, where e3t_0 == e3t_1d), which is
why it survived so long. These tests are synthetic -- they do not need the NEMO
build -- so they run in CI.
"""
import numpy as np
import pytest

from legoesm.ocean.fidelity.nemo_io import NemoGrid


def _synthetic_grid(n_lat=6, n_lon=5, nlev=8, with_e3t_0=True):
    """A full-step-z NemoGrid whose e3t_0 deliberately DIFFERS from e3t_1d."""
    e3t_1d = np.linspace(10.0, 100.0, nlev)
    # e3t_0 is stretched: same shape of ladder, but scaled so it is materially
    # different at depth -- mirroring the real DINO divergence.
    e3t_0_1d = e3t_1d * np.linspace(1.0, 1.30, nlev)
    gdept_1d = np.cumsum(e3t_1d) - 0.5 * e3t_1d
    gdept_0_1d = np.cumsum(e3t_0_1d) - 0.5 * e3t_0_1d

    # Staircase bathymetry: column (j,i) is wet for the first k_bot(j,i) levels.
    k_bot = np.full((n_lat, n_lon), nlev, dtype=int)
    k_bot[0, :] = nlev - 3
    k_bot[-1, :] = nlev - 1
    kk = np.arange(nlev)[None, None, :]
    tmask = (kk < k_bot[:, :, None]).astype(np.float64)

    ones2 = np.ones((n_lat, n_lon))
    # Coriolis must be CONSISTENT with gphit -- the bridge validates
    # ff_t == 2*Omega*sin(gphit) and rejects a mismatch.
    from legoesm import constants
    lat_t = ones2 * 10.0
    lat_v = lat_t + 0.5
    f_of = lambda lat: 2.0 * constants.Omega * np.sin(np.deg2rad(lat))
    kw = dict(
        glamt=ones2, gphit=lat_t, e1t=ones2 * 1000.0, e2t=ones2 * 1000.0,
        e1u=ones2 * 1000.0, e2v=ones2 * 1000.0,
        ff_t=f_of(lat_t), ff_f=f_of(lat_v),
        e3t_1d=e3t_1d, gdept_1d=gdept_1d, gdepw_1d=np.cumsum(e3t_1d) - e3t_1d,
        tmask=tmask, umask=tmask.copy(), vmask=tmask.copy(),
        # V-point latitudes: the topo bridge needs exact meridional cell faces.
        gphiv=lat_v,
    )
    if with_e3t_0:
        kw["e3t_0"] = np.broadcast_to(
            e3t_0_1d, (n_lat, n_lon, nlev)).copy()
        kw["gdept_0"] = np.broadcast_to(
            gdept_0_1d, (n_lat, n_lon, nlev)).copy()
    return NemoGrid(**kw), e3t_1d, e3t_0_1d, k_bot


def test_nemogrid_carries_e3t_0_optionally():
    """e3t_0/gdept_0 are optional so GYRE-era constructors stay valid."""
    g_with, _, _, _ = _synthetic_grid(with_e3t_0=True)
    g_without, _, _, _ = _synthetic_grid(with_e3t_0=False)
    assert g_with.e3t_0 is not None and g_with.gdept_0 is not None
    assert g_without.e3t_0 is None and g_without.gdept_0 is None


def test_prefers_e3t_0_over_the_1d_ladder():
    """With mode="both" the helper returns NEMO's ACTUAL scale factors.

    NOTE the shipped DEFAULT is mode="off" (the 1-D ladder): adopting e3t_0 is
    geometrically correct but currently DESTABILISES the model (see the
    docstring in nemo_state_bridge). These tests pin the correct behaviour so it
    is ready the moment the instability is fixed.
    """
    from legoesm.ocean.fidelity.nemo_state_bridge import (
        effective_vertical_scale_factors,
    )
    g, e3t_1d, e3t_0_1d, _ = _synthetic_grid()
    tmask = np.asarray(g.tmask) > 0.5
    e3t, t_depth, src = effective_vertical_scale_factors(g, tmask, mode="both")
    assert src == "e3t_0"
    # The two ladders must actually differ, or the test proves nothing.
    assert np.abs(e3t_0_1d - e3t_1d).max() > 1.0
    np.testing.assert_allclose(e3t[tmask.any(axis=(0, 1))],
                               e3t_0_1d[tmask.any(axis=(0, 1))], rtol=1e-12)


def test_falls_back_to_e3t_1d_for_gyre_era_grids():
    """Without e3t_0 (GYRE, key_linssh) the 1-D ladder is still correct."""
    from legoesm.ocean.fidelity.nemo_state_bridge import (
        effective_vertical_scale_factors,
    )
    g, e3t_1d, _, _ = _synthetic_grid(with_e3t_0=False)
    tmask = np.asarray(g.tmask) > 0.5
    e3t, _t, src = effective_vertical_scale_factors(g, tmask, mode="both")
    assert src == "e3t_1d"
    np.testing.assert_allclose(e3t, e3t_1d, rtol=1e-12)


def test_partial_cell_grid_is_rejected_not_averaged():
    """Horizontally-varying e3t_0 means ln_zps; it must RAISE.

    Silently averaging a thinned partial bottom cell into a full one would
    produce a plausible-looking but wrong bathymetry.
    """
    from legoesm.ocean.fidelity.nemo_state_bridge import (
        effective_vertical_scale_factors,
    )
    g, _, _, _ = _synthetic_grid()
    e3t_0 = np.array(g.e3t_0, copy=True)
    # Perturb a WET cell: column 0 is only wet to nlev-3, so thinning its
    # LAST level would change nothing (the guard only inspects wet cells).
    assert g.tmask[0, 0, 0] > 0.5
    e3t_0[0, 0, 0] *= 0.5               # one thinned cell = partial cell
    g = g._replace(e3t_0=e3t_0)
    tmask = np.asarray(g.tmask) > 0.5
    with pytest.raises(ValueError, match="PARTIAL-CELL"):
        effective_vertical_scale_factors(g, tmask, mode="both")


def test_bathymetry_from_e3t_0_gives_the_true_column_depth():
    """cumsum(e3t_0)[k_bot-1] is the bathymetry the bridge must produce."""
    from legoesm.ocean.fidelity.nemo_state_bridge import (
        effective_vertical_scale_factors,
    )
    g, e3t_1d, e3t_0_1d, k_bot = _synthetic_grid()
    tmask = np.asarray(g.tmask) > 0.5
    e3t, _t, _s = effective_vertical_scale_factors(g, tmask, mode="both")
    H = np.cumsum(e3t)[k_bot - 1]
    np.testing.assert_allclose(H, np.cumsum(e3t_0_1d)[k_bot - 1], rtol=1e-12)
    # and it is materially different from the e3t_1d answer
    assert np.abs(H - np.cumsum(e3t_1d)[k_bot - 1]).max() > 1.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
