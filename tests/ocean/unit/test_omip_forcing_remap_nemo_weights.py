"""``forcing_remap="nemo_weights"``: NEMO's per-channel split (bicubic winds,
bilinear rest) on every point grid, and the SAME choice on the heat/momentum,
freshwater and sea-ice samplers (the winds set evaporation in one and latent
heat in the other, so a mismatch breaks E = -lh/L_v)."""
from __future__ import annotations

import os
import pathlib
import types

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from legoesm.ocean.coupler import omip2_applicator as A  # noqa: E402

RUNNER = (pathlib.Path(__file__).resolve().parents[3]
          / "scripts" / "run" / "run_omip_core2.py")
CHANNELS = ("u10", "v10", "T_air", "q_air", "sw_down", "lw_down", "precip")


def _forcing():
    lat = np.linspace(-88.5, 88.5, 60)
    lon = np.arange(0.0, 360.0, 3.0)
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    wavy = np.cos(np.radians(LO) * 7.0) * np.cos(np.radians(LA) * 5.0)
    f = types.SimpleNamespace(lat=lat, lon=lon, snow=None, slp=None)
    for k, n in enumerate(CHANNELS):
        setattr(f, n, (1.0 + k + wavy)[None])
    return f


def _pts():
    rng = np.random.default_rng(1)
    return rng.uniform(-60, 60, 300), rng.uniform(0, 360, 300)


def _grid(grid_type, lat, lon):
    if grid_type == "mpas":
        return types.SimpleNamespace(latCell=np.radians(lat),
                                     lonCell=np.radians(lon))
    return types.SimpleNamespace(lat=np.radians(lat), lon=np.radians(lon))


@pytest.mark.parametrize("grid_type", ["mpas", "fesom"])
def test_point_grids_get_bicubic_winds_and_bilinear_rest(grid_type):
    f = _forcing()
    lat, lon = _pts()
    g = _grid(grid_type, lat, lon)
    nw = A._sample_omip2_forcing(f, 0, g, grid_type,
                                 forcing_remap="nemo_weights")
    bl = A._sample_omip2_forcing(f, 0, g, grid_type, forcing_remap="bilinear")
    ref = A._sample_forcing_points(f, 0, lat, lon, method="nemo_weights")
    for n in CHANNELS:
        np.testing.assert_allclose(nw[n], ref[n], rtol=0, atol=1e-12)
    for n in ("u10", "v10"):
        assert np.max(np.abs(nw[n] - bl[n])) > 1e-3, n
    for n in CHANNELS[2:]:
        np.testing.assert_allclose(nw[n], bl[n], rtol=0, atol=1e-12)


def test_latlon_refuses_a_point_method():
    g = types.SimpleNamespace(lat=np.radians(np.linspace(-80, 80, 10)),
                              lon=np.radians(np.linspace(0, 350, 20)))
    with pytest.raises(ValueError, match="point sampler"):
        A._sample_omip2_forcing(_forcing(), 0, g, "latlon",
                                forcing_remap="nemo_weights")


class _Seen(Exception):
    pass


@pytest.fixture
def spy(monkeypatch):
    def fake(forcing, idx_t, grid, grid_type, forcing_remap="bilinear"):
        raise _Seen(forcing_remap)
    monkeypatch.setattr(A, "_sample_omip2_forcing", fake)


def test_freshwater_forwards_the_remap(spy):
    with pytest.raises(_Seen, match="nemo_weights"):
        A.compute_omip2_freshwater_forcing(
            None, forcing=None, idx_t=0, grid=None, grid_type="mpas",
            forcing_remap="nemo_weights")


def test_ice_sampler_forwards_the_remap(spy):
    with pytest.raises(_Seen, match="nemo_weights"):
        A.sample_omip2_forcing(None, 0, None, "fesom",
                               forcing_remap="nemo_weights")


def test_every_driver_sampler_call_passes_the_remap():
    """AST, per call: a NEW sampler call added without the kwarg fails too."""
    import ast
    tree = ast.parse(RUNNER.read_text())
    names = {"compute_omip2_surface_forcing", "compute_omip2_freshwater_forcing",
             "sample_omip2_forcing"}
    calls = [c for c in ast.walk(tree) if isinstance(c, ast.Call)
             and getattr(c.func, "id", getattr(c.func, "attr", None)) in names]
    assert len(calls) == 7, len(calls)
    for c in calls:
        kw = {k.arg: ast.unparse(k.value) for k in c.keywords}
        assert kw.get("forcing_remap") == "args.forcing_remap", (
            c.lineno, ast.unparse(c.func))


_ORACLE = pathlib.Path(
    "/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS")
_MESH = pathlib.Path("/burg-archive/glab/users/pg2328/legoESM/data/grids/"
                     "eORCA1.2_mesh_mask.nc")
_NYF = pathlib.Path("/burg-archive/home/pg2328/.cache/legoesm/ocean_fidelity/"
                    "forcing/core2_nyf_mod/nyf.zarr")


@pytest.mark.skipif(not (_ORACLE.exists() and _MESH.exists() and _NYF.exists()),
                    reason="oracle weights / mesh / CORE-II cache absent")
def test_point_bicubic_reproduces_NEMOs_own_wind_weights_globally():
    """Instrument check for the whole ocean, not just the nino3 box.

    Measured 2026-10-13 (records 0 and 60): over every ocean point south of
    85.7N the point form equals ORCA1's bicubic SCRIP remap to <=1.2e-4 m/s.
    North of that (~150 points on the tripole fold rows) SCRIP differs by up
    to 8 m/s from BOTH point methods alike, so that is the weight files at the
    fold, not the bicubic -- excluded here, recorded, not fixed."""
    import xarray as xr
    with xr.open_dataset(_MESH, decode_times=False) as m:
        lat = np.asarray(m["gphit"]).squeeze()[0:331, 1:361]
        lon = np.asarray(m["glamt"]).squeeze()[0:331, 1:361]
        ocean = np.asarray(m["tmask"]).squeeze()[0][0:331, 1:361] > 0
    s = xr.open_zarr(_NYF, consolidated=False)
    f = types.SimpleNamespace(lat=np.asarray(s["lat"]), lon=np.asarray(s["lon"]))
    for nm in CHANNELS:
        setattr(f, nm, np.asarray(s[nm][0:1], dtype=np.float64))
    f.snow = f.slp = None
    with xr.open_dataset(_MESH, decode_times=False) as m:
        g = types.SimpleNamespace(
            lat_T=np.radians(np.asarray(m["gphit"]).squeeze()),
            lon_T=np.radians(np.asarray(m["glamt"]).squeeze()))
    out = {r: A.sample_omip2_forcing(f, 0, g, "tripole", forcing_remap=r)
           for r in ("nemo_weights", "nemo_scrip", "bilinear")}
    pt, scrip, bl = (out[r]["u10"][0:331, 1:361]
                     for r in ("nemo_weights", "nemo_scrip", "bilinear"))
    for nm in CHANNELS[2:]:      # scalars: both bilinear, oracle vs ours
        d = np.abs(out["nemo_weights"][nm] - out["nemo_scrip"][nm])[0:331, 1:361]
        assert np.median(d[ocean & (lat < 85.0)]) < 1e-5 * max(
            1.0, float(np.abs(out["nemo_scrip"][nm]).max())), nm
    sel = ocean & (lat < 85.0)
    err = np.abs(pt - scrip)[sel]
    assert np.median(err) < 1e-5 and err.max() < 5e-4, (np.median(err), err.max())
    # non-vacuity: bilinear is NOT the oracle at the same points
    assert np.abs(bl - scrip)[sel].max() > 0.5


def test_fesom_lane_gate_lets_the_remap_through():
    """The FESOM stage gate refuses any non-default flag it does not list, so
    without the registration the card would die at start-up."""
    from scripts.run.run_omip_core2 import (_build_arg_parser,
                                            validate_fesom_stage)
    p = _build_arg_parser()
    args = p.parse_args(["--grid", "fesom", "--fesom-mesh-dir", "/m",
                         "--output", "/o", "--no-emp",
                         "--forcing-remap", "nemo_weights"])
    validate_fesom_stage(args, p)
