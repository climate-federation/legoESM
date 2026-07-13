"""--ice-init: NEMO SI3 ice IC -> the prognostic sea-ice state.

Gates for ``legoesm.ocean.forcing.nemo_native_fields.load_nemo_ice_init``
and the ``scripts/run/run_omip_core2.py`` wiring:

  * the loader reads the SI3 layout (at_i/ht_i[/ht_s/sm_i/tmsu], 2-D
    ``y``/``x`` coords zeroed on land) and regrids nearest-WET — a land
    source cell (coord-invalid) is never sampled;
  * clamps: concentration in [0, 1], thicknesses >= 0, coherence
    (conc > 0 <=> h_ice > 0), target land zeroed, T_su capped at the
    melting point and NaN off-ice;
  * missing OPTIONAL vars (ht_s/sm_i/tmsu) -> None; missing REQUIRED
    (at_i/ht_i) -> ValueError;
  * the native eORCA-interior embed path (same convention as
    ``load_nemo_monthly_init_ts``) is taken when the valid source coords
    match the model interior — exact, no interpolation;
  * 1-D point-list targets (MPAS cell centres) keep their shape (the
    shared ``NearestWetRegridder`` would meshgrid two 1-D axes);
  * ``run_omip_core2._apply_ice_init`` maps only the fields present,
    leaves dynamics fields zero, respects the Field ``.replace`` API,
    and an all-zero IC reproduces the zero-ice cold-start state
    (zero-path invariance);
  * ``--ice-init`` without ``--prognostic-sea-ice`` raises at arg
    validation (before any file/grid work).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

netCDF4 = pytest.importorskip("netCDF4")

from legoesm import constants
from legoesm.ocean.forcing.curvilinear_regrid import coords_match
from legoesm.ocean.forcing.nemo_native_fields import load_nemo_ice_init


def _runner():
    from scripts.run import run_omip_core2 as r
    return r


# ---------------------------------------------------------------------------
# Synthetic SI3-layout file
# ---------------------------------------------------------------------------

# 3 x 4 source grid: column j=0 is LAND (SI3 zeroes coords + fields there),
# the rest ocean.  Lats 70/72/74 N, lons 10/20/30 E (land col gets 0,0).
_SRC_LAT_1D = np.array([70.0, 72.0, 74.0])
_SRC_LON_1D = np.array([0.0, 10.0, 20.0, 30.0])  # slot 0 overwritten by land


def _write_si3_file(path, *, drop=(), at_i=None, ht_i=None, coord_names=None):
    """Minimal Ice_initialization.nc clone: (time_counter, y, x) vars +
    2-D ``y``/``x`` coords with (0, 0) on the land column j=0."""
    lat2 = np.broadcast_to(_SRC_LAT_1D[:, None], (3, 4)).copy()
    lon2 = np.broadcast_to(_SRC_LON_1D[None, :], (3, 4)).copy()
    lat2[:, 0] = 0.0
    lon2[:, 0] = 0.0
    default_at = np.zeros((3, 4))
    default_at[:, 1] = 0.4          # ice along the j=1 ocean column
    default_ht = np.zeros((3, 4))
    default_ht[:, 1] = 1.5
    fields = {
        "at_i": default_at if at_i is None else np.asarray(at_i, float),
        "ht_i": default_ht if ht_i is None else np.asarray(ht_i, float),
        "ht_s": np.where(default_at > 0, 0.2, 0.0),
        "sm_i": np.where(default_at > 0, 6.0, 0.0),
        "tmsu": np.where(default_at > 0, 260.0, 0.0),
    }
    lat_name, lon_name = coord_names or ("y", "x")
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("time_counter", 1)
        ds.createDimension("y", 3)
        ds.createDimension("x", 4)
        for var, arr in fields.items():
            if var in drop:
                continue
            v = ds.createVariable(var, "f8", ("time_counter", "y", "x"))
            v[0] = arr
        vy = ds.createVariable(lat_name, "f4", ("y", "x"))
        vy[:] = lat2
        vx = ds.createVariable(lon_name, "f4", ("y", "x"))
        vx[:] = lon2
    return lat2, lon2, fields


# ---------------------------------------------------------------------------
# Loader: nearest-wet, clamps, coherence, land masking
# ---------------------------------------------------------------------------


def test_loader_nearest_wet_never_samples_land(tmp_path):
    """A target point NEXT to the SI3 land column must take the nearest
    OCEAN source value — land cells (coord-invalid, garbage payload) are
    excluded from the KDTree entirely."""
    p = str(tmp_path / "ice.nc")
    at = np.zeros((3, 4))
    at[:, 1] = 0.4
    at[:, 0] = 7.7        # garbage on LAND (would clamp to 1.0 if sampled)
    ht = np.zeros((3, 4))
    ht[:, 1] = 1.5
    ht[:, 0] = 9.9
    _write_si3_file(p, at_i=at, ht_i=ht)
    # Target: 2x2 grid sitting just WEST of the ice column (closer to the
    # land column's true location than to j=2), all ocean.
    tlat = np.array([[70.0, 72.0], [70.0, 72.0]]).T
    tlon = np.full((2, 2), 8.0)
    wet = np.ones((2, 2))
    ic = load_nemo_ice_init(p, tlat, tlon, wet)
    np.testing.assert_allclose(ic.concentration, 0.4)   # not 1.0 (=clamped 7.7)
    np.testing.assert_allclose(ic.h_ice, 1.5)           # not 9.9
    np.testing.assert_allclose(ic.h_snow, 0.2)
    np.testing.assert_allclose(ic.S_ice, 6.0)
    np.testing.assert_allclose(ic.T_su, 260.0)


def test_loader_clamps_and_coherence(tmp_path):
    """conc > 1 clamps to 1; negative thickness floors to 0 and zeroes the
    pair (no ghost ice: conc > 0 <=> h_ice > 0); target land zeroed."""
    p = str(tmp_path / "ice.nc")
    at = np.zeros((3, 4))
    ht = np.zeros((3, 4))
    at[0, 1], ht[0, 1] = 1.4, 2.0     # over-unity conc, real ice
    at[1, 1], ht[1, 1] = 0.5, -1.0    # negative thickness -> both zeroed
    at[2, 1], ht[2, 1] = 0.9, 1.0     # real ice, later landed by wet_mask
    _write_si3_file(p, at_i=at, ht_i=ht)
    # Targets exactly on the three ice-column points.
    tlat = _SRC_LAT_1D.reshape(3, 1).copy()
    tlon = np.full((3, 1), 10.0)
    wet = np.ones((3, 1))
    wet[2, 0] = 0.0                   # model land at the third point
    ic = load_nemo_ice_init(p, tlat, tlon, wet)
    assert ic.concentration[0, 0] == 1.0          # clamped
    assert ic.h_ice[0, 0] == 2.0
    assert ic.concentration[1, 0] == 0.0          # coherence zeroed
    assert ic.h_ice[1, 0] == 0.0
    assert ic.h_snow[1, 0] == 0.0                 # follows presence
    assert ic.S_ice[1, 0] == 0.0
    assert np.isnan(ic.T_su[1, 0])                # off-ice -> NaN
    assert ic.concentration[2, 0] == 0.0          # target land masked
    assert ic.h_ice[2, 0] == 0.0


def test_loader_tsu_capped_at_freeze(tmp_path):
    """T_su above the melting point caps at constants.T_freeze."""
    p = str(tmp_path / "ice.nc")
    _write_si3_file(p)
    with netCDF4.Dataset(p, "a") as ds:
        t = ds.variables["tmsu"][0]
        t[:, 1] = 280.0               # unphysical: above melting
        ds.variables["tmsu"][0] = t
    tlat = np.array([[72.0]])
    tlon = np.array([[10.0]])
    ic = load_nemo_ice_init(p, tlat, tlon, np.ones((1, 1)))
    assert ic.T_su[0, 0] == pytest.approx(constants.T_freeze)


def test_loader_missing_optional_none_required_raises(tmp_path):
    p1 = str(tmp_path / "no_opt.nc")
    _write_si3_file(p1, drop=("ht_s", "sm_i", "tmsu"))
    tlat = np.array([[72.0]])
    tlon = np.array([[10.0]])
    ic = load_nemo_ice_init(p1, tlat, tlon, np.ones((1, 1)))
    assert ic.h_snow is None and ic.S_ice is None and ic.T_su is None
    assert ic.concentration[0, 0] == pytest.approx(0.4)

    p2 = str(tmp_path / "no_req.nc")
    _write_si3_file(p2, drop=("at_i",))
    with pytest.raises(ValueError, match="at_i"):
        load_nemo_ice_init(p2, tlat, tlon, np.ones((1, 1)))


def test_loader_native_embed_exact(tmp_path):
    """Target = source grid + ORCA halos (north row + 2 cyclic columns):
    the interior embeds EXACTLY (same convention as the T/S monthly init)
    even though the SI3 land cells carry (0, 0) coords — the same-mesh
    check runs on the coordinate-VALID cells only."""
    p = str(tmp_path / "ice.nc")
    src_lat, src_lon, fields = _write_si3_file(p)
    n_lat, n_lon = 3 + 1, 4 + 2
    # Model coords: interior = TRUE source locations (real values on the
    # land column too — the model grid knows its land coordinates).
    int_lat = np.broadcast_to(_SRC_LAT_1D[:, None], (3, 4))
    int_lon = np.broadcast_to(_SRC_LON_1D[None, :], (3, 4))
    tlat = np.zeros((n_lat, n_lon))
    tlon = np.zeros((n_lat, n_lon))
    tlat[:-1, 1:-1] = int_lat
    tlon[:-1, 1:-1] = int_lon
    tlat[:-1, 0], tlon[:-1, 0] = int_lat[:, -1], int_lon[:, -1]
    tlat[:-1, -1], tlon[:-1, -1] = int_lat[:, 0], int_lon[:, 0]
    tlat[-1], tlon[-1] = tlat[-2], tlon[-2]
    wet = np.ones((n_lat, n_lon))
    ic = load_nemo_ice_init(p, tlat, tlon, wet)
    np.testing.assert_array_equal(ic.concentration[:-1, 1:-1],
                                  fields["at_i"])
    np.testing.assert_array_equal(ic.h_ice[:-1, 1:-1], fields["ht_i"])
    # Cyclic overlap: col 0 <- interior col[-1] (ice-free), col -1 <- col 0.
    np.testing.assert_array_equal(ic.concentration[:-1, 0],
                                  fields["at_i"][:, -1])
    np.testing.assert_array_equal(ic.concentration[:-1, -1],
                                  fields["at_i"][:, 0])


def test_loader_1d_point_list_target(tmp_path):
    """MPAS-style 1-D cell-centre lists keep their (nCells,) shape (no
    accidental meshgrid) and take nearest-wet values."""
    p = str(tmp_path / "ice.nc")
    _write_si3_file(p)
    tlat = np.array([70.0, 72.0, 74.0, 70.0])
    tlon = np.array([10.0, 10.0, 10.0, 30.0])
    wet = np.ones(4)
    ic = load_nemo_ice_init(p, tlat, tlon, wet)
    assert ic.concentration.shape == (4,)
    np.testing.assert_allclose(ic.concentration[:3], 0.4)
    assert ic.concentration[3] == 0.0             # ice-free ocean column


def test_coords_match_valid_mask():
    """The valid= restriction: zeroed land coords don't break the same-mesh
    check, and an all-False mask can never claim a match."""
    lat = np.array([[10.0, 20.0], [0.0, 30.0]])
    lon = np.array([[5.0, 6.0], [0.0, 7.0]])
    tgt_lat = lat.copy()
    tgt_lon = lon.copy()
    tgt_lat[1, 0], tgt_lon[1, 0] = -45.0, 120.0   # model's real land coords
    valid = np.array([[True, True], [False, True]])
    assert not coords_match(lat, lon, tgt_lat, tgt_lon)          # legacy: fails
    assert coords_match(lat, lon, tgt_lat, tgt_lon, valid=valid)  # masked: ok
    assert not coords_match(lat, lon, tgt_lat, tgt_lon,
                            valid=np.zeros((2, 2), dtype=bool))


# ---------------------------------------------------------------------------
# run_omip_core2 wiring
# ---------------------------------------------------------------------------


def _zero_state(shape=(4, 5)):
    from legoesm.ice import init_dynamic_ice_state
    import jax.numpy as jnp
    st = init_dynamic_ice_state(shape, S_ice_init=0.0)
    return st._replace(concentration=st.concentration.replace(
        data=jnp.zeros_like(st.concentration.data)))


def test_apply_ice_init_maps_fields():
    from legoesm.ocean.forcing.nemo_native_fields import NemoIceInit
    r = _runner()
    st = _zero_state((2, 3))
    conc = np.array([[0.9, 0.0, 0.5], [0.0, 0.0, 0.0]])
    h = np.array([[1.5, 0.0, 0.7], [0.0, 0.0, 0.0]])
    hs = np.array([[0.2, 0.0, 0.1], [0.0, 0.0, 0.0]])
    si = np.array([[6.0, 0.0, 4.0], [0.0, 0.0, 0.0]])
    tsu = np.where(conc > 0, 255.0, np.nan)
    out = r._apply_ice_init(st, NemoIceInit(conc, h, hs, si, tsu))
    np.testing.assert_allclose(np.asarray(out.concentration.data), conc)
    np.testing.assert_allclose(np.asarray(out.h_ice.data), h)
    np.testing.assert_allclose(np.asarray(out.h_snow.data), hs)
    np.testing.assert_allclose(np.asarray(out.S_ice.data), si)
    T = np.asarray(out.T_ice.data)
    assert np.all(T[conc > 0] == 255.0)
    # Off-ice cells keep the state's own default (260 K) — no NaN leaks.
    assert np.all(T[conc == 0] == np.asarray(st.T_ice.data)[conc == 0])
    assert np.isfinite(T).all()
    # Dynamics/pond fields have no SI3 counterpart: still zero.
    for name in ("u_ice", "v_ice", "sigma_11", "sigma_22", "sigma_12",
                 "pond_area", "pond_depth"):
        np.testing.assert_array_equal(
            np.asarray(getattr(out, name).data), 0.0)


def test_apply_ice_init_optional_none_and_zero_path():
    """Optional fields None -> untouched; an all-zero IC reproduces the
    zero-ice cold-start state exactly (zero-path invariance)."""
    from legoesm.ocean.forcing.nemo_native_fields import NemoIceInit
    r = _runner()
    st = _zero_state((2, 2))
    z = np.zeros((2, 2))
    out = r._apply_ice_init(st, NemoIceInit(z, z, None, None, None))
    for name in st._fields:
        np.testing.assert_array_equal(
            np.asarray(getattr(out, name).data),
            np.asarray(getattr(st, name).data), err_msg=name)


def test_apply_ice_init_shape_mismatch_raises():
    from legoesm.ocean.forcing.nemo_native_fields import NemoIceInit
    r = _runner()
    st = _zero_state((2, 3))
    z = np.zeros((3, 2))
    with pytest.raises(ValueError, match="shape"):
        r._apply_ice_init(st, NemoIceInit(z, z, None, None, None))
    z3 = np.zeros((2, 3, 5))          # phantom category axis
    with pytest.raises(ValueError, match="rank|categor"):
        r._apply_ice_init(st, NemoIceInit(z3, z3, None, None, None))
    ok = np.zeros((2, 3))             # conc fine, h_ice mismatched (codex r2)
    with pytest.raises(ValueError, match="h_ice"):
        r._apply_ice_init(st, NemoIceInit(ok, z, None, None, None))


def test_ice_init_requires_prognostic_flag(monkeypatch, tmp_path):
    """--ice-init without --prognostic-sea-ice raises at arg validation,
    before any forcing/grid work touches the filesystem."""
    r = _runner()
    monkeypatch.setattr(
        "sys.argv",
        ["run_omip_core2.py", "--grid", "tripole",
         "--ice-init", str(tmp_path / "ice.nc")])
    with pytest.raises(ValueError, match="ice-init.*prognostic-sea-ice"):
        r.main()


def test_ice_init_cli_default_is_none(monkeypatch):
    """No --ice-init -> args.ice_init None: the setup block that replaces
    the zero-ice cold start is gated on `is not None`, so the legacy path
    is structurally byte-identical."""
    import argparse
    r = _runner()
    seen = {}
    orig = argparse.ArgumentParser.parse_args

    def _spy(self, *a, **k):
        ns = orig(self, *a, **k)
        seen["ns"] = ns
        raise SystemExit(0)           # stop main() right after parsing

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", _spy)
    monkeypatch.setattr("sys.argv", ["run_omip_core2.py", "--grid", "tripole"])
    with pytest.raises(SystemExit):
        r.main()
    assert seen["ns"].ice_init is None


# ---------------------------------------------------------------------------
# Snapshot + CSV additions
# ---------------------------------------------------------------------------


class _FakeGrid:
    def __init__(self, area):
        self.area = area


def test_ice_global_stats():
    from legoesm.ocean.forcing.nemo_native_fields import NemoIceInit
    r = _runner()
    st = _zero_state((2, 2))
    conc = np.array([[0.5, 0.0], [1.0, 0.8]])
    h = np.array([[1.0, 0.0], [2.0, 3.0]])
    st = r._apply_ice_init(st, NemoIceInit(conc, h, None, None, None))
    mask = np.array([[1.0, 1.0], [1.0, 0.0]])     # land the (1,1) cell
    area = np.full((2, 2), 10.0)
    a, c, hmax, cm = r._ice_global_stats(st, _FakeGrid(area), mask)
    assert a == pytest.approx((0.5 + 1.0) * 10.0)  # land cell excluded
    assert c == pytest.approx((0.5 + 1.0) / 2.0)
    assert hmax == 2.0                             # 3.0 sits on land
    np.testing.assert_allclose(cm, np.where(mask > 0.5, conc, 0.0))


def test_snapshot_writes_ice_fields(tmp_path):
    """_save_snapshot(ice_state=...) adds ocean-masked ice_concentration +
    ice_thickness; without ice_state the npz keys are unchanged (old
    loaders keep working — compare_omip_nemo reads explicit keys)."""
    from legoesm.core.field import Field
    from legoesm.ocean.forcing.nemo_native_fields import NemoIceInit
    import jax.numpy as jnp
    from types import SimpleNamespace
    r = _runner()
    n = 3
    mask = np.ones((n, n))
    mask[0, 0] = 0.0

    def F(a, name):
        return Field(data=jnp.asarray(a), name=name,
                     dims=("x", "y"), units="1")

    state = SimpleNamespace(
        T=F(np.zeros((n, n, 2)), "T"), S=F(np.zeros((n, n, 2)), "S"),
        u=F(np.zeros((n, n, 2)), "u"), v=None, eta=None, H_bathy=None,
        land_mask=F(mask, "land_mask"),
    )
    lat = np.zeros((n, n))
    conc = np.full((n, n), 0.5)
    h = np.full((n, n), 1.5)
    ice = r._apply_ice_init(_zero_state((n, n)),
                            NemoIceInit(conc, h, None, None, None))
    r._save_snapshot(tmp_path, "t1", state, lat, lat, ice_state=ice)
    with np.load(tmp_path / "snapshot_t1.npz") as z:
        assert set(["ice_concentration", "ice_thickness"]) <= set(z.files)
        np.testing.assert_allclose(z["ice_concentration"], conc * mask)
        np.testing.assert_allclose(z["ice_thickness"], h * mask)
    r._save_snapshot(tmp_path, "t0", state, lat, lat)
    with np.load(tmp_path / "snapshot_t0.npz") as z:
        assert "ice_concentration" not in z.files
        assert "ice_thickness" not in z.files
        assert {"T", "S", "u", "land_mask", "lat_T", "lon_T"} <= set(z.files)
