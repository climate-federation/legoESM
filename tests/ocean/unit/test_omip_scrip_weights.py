"""Gap 13, exact reproduction: remap through NEMO's OWN SCRIP weight files.

Every convention asserted here was MEASURED before the code was written
(probe jobs in scripts/cluster/omip_nemo/_scrip_*.sbatch), not inferred:

  * src indices are 1-based flat C-order into the 94x192 CORE-II grid, stored
    as float;
  * the source latitude axis is used ascending, as-is -- proven with an
    ANTISYMMETRIC field, because a symmetric one such as cos(lat) cannot
    detect a latitude flip and my first probe missed it for exactly that
    reason;
  * the destination is python [0:331, 1:361] of our (332, 362) mesh, which is
    stated in the weight file's own history attribute;
  * over the 113761 full 4-point stencils a linear field reproduces to a
    median 2.18e-08.

The tests that need the real files skip when they are absent, so this suite
still runs on a machine without the oracle.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

from legoesm.ocean.coupler import omip2_applicator as A

ORACLE = pathlib.Path(
    "/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS")
BILIN = ORACLE / "weights_coreII_2_eORCA1.4.2_bilinear.nc"
BICUB = ORACLE / "weights_coreII_2_eORCA1.4.2_bicubic.nc"
MESH = pathlib.Path("/burg-archive/glab/users/pg2328/legoESM/data/grids/"
                    "eORCA1.2_mesh_mask.nc")
CORE_SHAPE = (94, 192)

needs_oracle = pytest.mark.skipif(
    not BILIN.exists(), reason="oracle weight files not present")


# --- the padding, which is testable without the oracle ----------------------

def test_padding_matches_NEMOs_asymmetric_latitude_treatment():
    """The latitude halo is NOT symmetric in NEMO and getting it wrong is easy.

    fldread.F90:1495-1499: the SOUTH edge is replicated, the NORTH edge is
    LINEARLY EXTRAPOLATED as 2*f[-1] - f[-2]. Replicating both -- the first
    version of this function -- flattens the gradient in the northernmost
    source row and perturbs every Arctic derivative stencil.
    """
    f = np.arange(12.0).reshape(3, 4)
    p = A._pad_source(f)
    assert p.shape == (5, 6)
    # longitude wraps (CORE-II is a cyclic global grid)
    assert p[1, 0] == f[0, -1]
    assert p[1, -1] == f[0, 0]
    # south: replicate
    assert np.array_equal(p[0], p[1])
    # north: extrapolate, NOT replicate
    assert np.array_equal(p[-1], 2.0 * p[-2] - p[-3])
    assert not np.array_equal(p[-1], p[-2])


def test_north_extrapolation_is_exact_on_a_linear_field():
    """On a field linear in latitude the extrapolated ghost row must continue
    the line exactly -- that is the whole point of using 2f-f over replicate."""
    lat = np.arange(5.0).reshape(-1, 1)
    f = np.repeat(lat, 3, axis=1)
    p = A._pad_source(f)
    assert np.allclose(p[-1, 1:-1], 5.0)


# --- the reader -------------------------------------------------------------

@needs_oracle
def test_bilinear_file_has_four_triples_and_sums_to_one():
    src0, wgt, n = A.load_scrip_weights(BILIN)
    assert n == 4
    assert src0.min() >= 0
    assert src0.max() < CORE_SHAPE[0] * CORE_SHAPE[1]
    s = wgt.sum(axis=0)
    assert np.allclose(s, 1.0, atol=1e-9), (s.min(), s.max())


@needs_oracle
def test_bicubic_file_has_sixteen_triples_and_does_NOT_sum_to_one():
    """The measurement that stopped a wrong implementation: if these summed to
    one they would be value weights, and a naive weighted sum would be right.
    They do not, because twelve of them multiply DERIVATIVES."""
    src0, wgt, n = A.load_scrip_weights(BICUB)
    assert n == 16
    s = wgt.sum(axis=0)
    assert s.min() < 0.9 and s.max() > 1.1, (s.min(), s.max())


@needs_oracle
def test_a_non_weights_file_is_rejected():
    with pytest.raises(ValueError, match="weight triples"):
        A.load_scrip_weights(MESH)


# --- the application --------------------------------------------------------

def _analytic(lat, lon):
    """ANTISYMMETRIC in latitude on purpose -- a symmetric field cannot detect
    a latitude flip, which is the failure this whole suite exists to exclude.
    Linear, so bilinear reproduces it exactly."""
    return lat / 90.0 + 0.0 * lon


def _dst_latlon():
    import xarray as xr
    with xr.open_dataset(MESH, decode_times=False) as m:
        glamt = np.asarray(m["glamt"]).squeeze()
        gphit = np.asarray(m["gphit"]).squeeze()
    return gphit[0:331, 1:361], glamt[0:331, 1:361]


def _src_axes():
    import xarray as xr
    z = ("/burg-archive/home/pg2328/.cache/legoesm/ocean_fidelity/forcing/"
         "core2_nyf_mod/nyf.zarr")
    s = xr.open_zarr(z, consolidated=False)
    return np.asarray(s["lat"]), np.asarray(s["lon"])


@needs_oracle
def test_bilinear_reproduces_a_linear_field_on_full_stencils():
    """THE pre-registered number: median error 2.18e-08 over full stencils."""
    if not MESH.exists():
        pytest.skip("mesh absent")
    src0, wgt, n = A.load_scrip_weights(BILIN)
    slat, slon = _src_axes()
    field = _analytic(slat[:, None], slon[None, :])
    out = A.apply_scrip_weights(field, src0, wgt, n, (slat.size, slon.size))
    dlat, dlon = _dst_latlon()
    err = np.abs(out - _analytic(dlat, dlon))
    full = (np.abs(wgt) > 0).sum(axis=0) == 4
    assert np.median(err[full]) < 1e-6, np.median(err[full])


@needs_oracle
def test_bicubic_beats_treating_its_weights_as_value_weights():
    """NON-VACUITY for the derivative terms. A naive sum(f[src]*wgt) is the
    implementation a careless reading produces; it must be MEASURABLY worse.
    If both scored the same, the twelve derivative weights would be doing
    nothing and the bicubic path would be decorative."""
    if not MESH.exists():
        pytest.skip("mesh absent")
    src0, wgt, n = A.load_scrip_weights(BICUB)
    slat, slon = _src_axes()
    field = _analytic(slat[:, None], slon[None, :])
    proper = A.apply_scrip_weights(field, src0, wgt, n,
                                   (slat.size, slon.size))
    naive = (field.ravel(order="C")[src0] * wgt).sum(axis=0)
    dlat, dlon = _dst_latlon()
    truth = _analytic(dlat, dlon)
    full = (np.abs(wgt) > 0).sum(axis=0) == 16
    e_proper = np.median(np.abs(proper - truth)[full])
    e_naive = np.median(np.abs(naive - truth)[full])
    assert e_proper < e_naive / 100.0, (e_proper, e_naive)
    assert e_proper < 1e-6, e_proper


@needs_oracle
def test_every_derivative_GROUP_actually_contributes():
    """codex MAJOR: my first fields were LONGITUDE-INDEPENDENT, so the
    i-gradient and cross-derivative weights multiplied zero and both
    application tests passed with those terms deleted. Twelve of the sixteen
    weights were untested.

    Use a field varying in BOTH directions, then zero each weight group in
    turn: every group must change the answer, or it is dead code.
    """
    if not MESH.exists():
        pytest.skip("mesh absent")
    src0, wgt, n = A.load_scrip_weights(BICUB)
    slat, slon = _src_axes()
    la, lo = slat[:, None], slon[None, :]
    field = (la / 90.0) * np.cos(np.deg2rad(lo)) + np.sin(np.deg2rad(lo))
    shape = (slat.size, slon.size)
    full = A.apply_scrip_weights(field, src0, wgt, n, shape)
    for lo_i, hi_i, label in ((4, 8, "d/di"), (8, 12, "d/dj"),
                              (12, 16, "cross")):
        w = wgt.copy()
        w[lo_i:hi_i] = 0.0
        without = A.apply_scrip_weights(field, src0, w, n, shape)
        moved = np.abs(full - without)
        assert moved.max() > 1e-6, f"{label} weights contribute nothing"


@needs_oracle
def test_the_corner_repeat_guard_actually_fires():
    """codex MINOR: the real file passes the guard, so nothing proved it
    works. Plant a mismatched corner and require the raise."""
    src0, wgt, n = A.load_scrip_weights(BICUB)
    bad = src0.copy()
    bad[7, 0, 0] += 1            # break the repeat in one triple
    with pytest.raises(ValueError, match="same four source corners"):
        A.apply_scrip_weights(np.zeros(CORE_SHAPE), bad, wgt, n, CORE_SHAPE)


@needs_oracle
def test_a_constant_field_survives_both_files():
    """Bilinear weights sum to 1 so a constant is exact. For bicubic the
    derivative terms vanish on a constant, so it must ALSO be exact -- that is
    a real check on the derivative stencils, not a tautology."""
    if not MESH.exists():
        pytest.skip("mesh absent")
    slat, slon = _src_axes()
    field = np.full((slat.size, slon.size), 3.75)
    for path, want_n in ((BILIN, 4), (BICUB, 16)):
        src0, wgt, n = A.load_scrip_weights(path)
        assert n == want_n
        out = A.apply_scrip_weights(field, src0, wgt, n,
                                    (slat.size, slon.size))
        full = (np.abs(wgt) > 0).sum(axis=0) == want_n
        assert np.allclose(out[full], 3.75, atol=1e-9), (
            path.name, np.abs(out[full] - 3.75).max())


@needs_oracle
def test_the_reader_is_cached_by_path():
    A._SCRIP_CACHE.clear()
    A.load_scrip_weights(BILIN)
    A.load_scrip_weights(BILIN)
    assert len(A._SCRIP_CACHE) == 1


# --- the grid-level wiring --------------------------------------------------

def test_halo_helper_rejects_a_mismatched_interior():
    """A weights file for a different grid must not be silently reshaped in."""
    with pytest.raises(ValueError, match="does not fit"):
        A.scrip_interior_to_full_tripole(np.zeros((100, 200)))


def test_halo_helper_fills_everything_and_mirrors_the_overlap():
    interior = np.random.default_rng(3).normal(size=(331, 360)) + 10.0
    full = A.scrip_interior_to_full_tripole(interior)
    assert full.shape == (332, 362)
    assert np.all(full != 0.0), "an entry was left as a silent zero"
    assert np.array_equal(full[0:331, 1:361], interior)
    assert np.array_equal(full[0:331, 0], full[0:331, 360])
    assert np.array_equal(full[0:331, 361], full[0:331, 1])
    assert np.array_equal(full[331], full[330])


def test_the_runner_helper_is_an_alias_not_a_second_copy():
    """One halo convention, one implementation. Two copies would drift."""
    # tests/ocean/unit/ -> parents[3] is the repo root, not parents[2].
    src = (pathlib.Path(__file__).resolve().parents[3] / "scripts" / "run"
           / "run_omip_core2.py").read_text()
    assert "scrip_interior_to_full_tripole" in src
    assert "out[331, :] = out[330, :]" not in src, "duplicate halo logic"


class _Grid:
    def __init__(self, ny=332, nx=362):
        import numpy as _np
        self.lat_T = _np.deg2rad(_np.linspace(-80, 80, ny))[:, None] \
            * _np.ones((1, nx))
        self.lon_T = _np.deg2rad(_np.linspace(0, 359, nx))[None, :] \
            * _np.ones((ny, 1))


class _F:
    lat = None
    lon = None

    def __init__(self, ny=94, nx=192):
        type(self).lat = np.linspace(-88.5, 88.5, ny)
        type(self).lon = np.linspace(0, 358.125, nx)
        f = np.ones((1, ny, nx))
        for n in ("u10", "v10", "T_air", "q_air", "sw_down", "lw_down",
                  "precip"):
            setattr(self, n, f.copy())
        self.snow = None
        self.slp = None


def test_unknown_forcing_remap_raises():
    with pytest.raises(ValueError, match="unknown forcing_remap"):
        A._sample_omip2_forcing(_F(), 0, _Grid(), "tripole",
                                forcing_remap="cubic")


def test_scrip_forcing_is_refused_on_non_tripole_grids():
    with pytest.raises(ValueError, match="tripole only"):
        A._sample_omip2_forcing(_F(), 0, _Grid(), "mpas",
                                forcing_remap="nemo_scrip")


def test_scrip_forcing_is_refused_on_a_wrongly_sized_tripole():
    with pytest.raises(ValueError, match=r"\(332, 362\)"):
        A._sample_omip2_forcing(_F(), 0, _Grid(100, 200), "tripole",
                                forcing_remap="nemo_scrip")


def test_the_default_is_still_bilinear_so_nothing_moved():
    import inspect
    sig = inspect.signature(A._sample_omip2_forcing)
    assert sig.parameters["forcing_remap"].default == "bilinear"


@needs_oracle
def test_scrip_forcing_returns_full_fields_and_routes_winds_bicubically():
    """The per-channel split is the point: winds through the BICUBIC file,
    everything else through the BILINEAR one."""
    f = _F()
    out = A.sample_forcing_tripole_scrip(f, 0)
    for name, field in out.items():
        assert field.shape == (332, 362), (name, field.shape)
    # A constant source must come back constant through BOTH files -- which
    # also proves the bicubic derivative terms vanish correctly.
    for name, field in out.items():
        assert np.allclose(field, 1.0, atol=1e-6), (
            name, float(np.abs(field - 1.0).max()))
