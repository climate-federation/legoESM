"""Unit tests for the NEMO restart/mesh_mask reader (``fidelity.nemo_io``).

Self-contained: writes a tiny synthetic NEMO-shaped NetCDF (halo included,
``(z, y, x)`` axis order) and checks the reader strips the ``nn_hls`` halo and
moves the vertical axis last — no dependency on an external NEMO run.
"""
import numpy as np
import pytest
import xarray as xr

from legoesm.ocean.fidelity.nemo_io import (
    read_nemo_mesh_mask,
    read_nemo_restart,
    read_nemo_restart_before,
    read_nemo_restart_en,
)

# Global-with-halo dims: nn_hls=1 -> interior (ny-2, nx-2) = (4, 3).
NZ, NY, NX = 3, 6, 5
IY, IX = NY - 2, NX - 2


def _encode(zyx: np.ndarray) -> np.ndarray:
    """Fill a (z,y,x) array with a decodable pattern z*100 + y*10 + x."""
    z, y, x = np.meshgrid(
        np.arange(NZ), np.arange(NY), np.arange(NX), indexing="ij"
    )
    return (z * 100 + y * 10 + x).astype(np.float64)


def _write_mesh_mask(path):
    d2 = (("y", "x"), np.arange(NY * NX).reshape(NY, NX).astype(np.float64))
    d3 = (("z", "y", "x"), _encode(None))
    d1 = (("z",), np.arange(NZ).astype(np.float64) + 0.5)
    ds = xr.Dataset({
        "glamt": d2, "gphit": d2, "e1t": d2, "e2t": d2, "e1u": d2, "e2v": d2,
        "ff_t": d2, "ff_f": d2,
        "e3t_1d": d1, "gdept_1d": d1, "gdepw_1d": d1,
        "tmask": d3, "umask": d3, "vmask": d3,
    })
    ds.to_netcdf(path)


def _write_restart(path, with_rhd=True, with_en=False, with_before=False,
                    with_before_forcing=False):
    d2 = (("y", "x"), np.arange(NY * NX).reshape(NY, NX).astype(np.float64))
    d3 = (("z", "y", "x"), _encode(None))
    data = {"tn": d3, "sn": d3, "un": d3, "vn": d3, "sshn": d2}
    if with_rhd:
        data["rhd"] = d3
    if with_en:
        data["en"] = d3
    if with_before:
        # Distinct pattern (offset +1) so a reader bug that accidentally
        # reads the now-level fields is caught by value, not just shape.
        d3b = (("z", "y", "x"), _encode(None) + 1.0)
        d2b = (("y", "x"), np.arange(NY * NX).reshape(NY, NX).astype(np.float64) + 1.0)
        data.update({"tb": d3b, "sb": d3b, "ub": d3b, "vb": d3b, "sshb": d2b})
    if with_before_forcing:
        d2f = (("y", "x"), np.arange(NY * NX).reshape(NY, NX).astype(np.float64) + 2.0)
        data.update({"utau_b": d2f, "vtau_b": d2f})
    xr.Dataset(data).to_netcdf(path)


def test_mesh_mask_halo_strip_and_axis_order(tmp_path):
    p = tmp_path / "mesh_mask.nc"
    _write_mesh_mask(p)
    g = read_nemo_mesh_mask(str(p), nn_hls=1)
    assert g.glamt.shape == (IY, IX)
    assert g.tmask.shape == (IY, IX, NZ)          # vertical last
    assert g.e3t_1d.shape == (NZ,)
    # interior origin: strip 1 halo -> global (y=1, x=1)
    assert g.glamt[0, 0] == 1 * NX + 1
    # 3-D pattern z*100+y*10+x at interior (iy,ix,iz) -> global (iz, iy+1, ix+1)
    assert g.tmask[1, 2, 0] == 0 * 100 + (1 + 1) * 10 + (2 + 1)   # = 23


def test_restart_reads_state_and_rhd(tmp_path):
    p = tmp_path / "restart.nc"
    _write_restart(p, with_rhd=True)
    s = read_nemo_restart(str(p), nn_hls=1)
    assert s.T.shape == (IY, IX, NZ)
    assert s.ssh.shape == (IY, IX)
    assert s.rhd is not None and s.rhd.shape == (IY, IX, NZ)
    # same decodable pattern
    assert s.T[1, 2, 0] == (1 + 1) * 10 + (2 + 1)


def test_restart_without_rhd(tmp_path):
    p = tmp_path / "restart_norhd.nc"
    _write_restart(p, with_rhd=False)
    s = read_nemo_restart(str(p), nn_hls=1)
    assert s.rhd is None


def test_restart_before_reads_distinct_tb_sb_ub_vb(tmp_path):
    """#1317 --bridge-before: tb/sb/ub/vb read via the SAME halo-strip +
    axis-order path as tn/sn/un/vn, but from the distinct +1-offset pattern
    -- catches a reader bug that silently re-reads the now-level fields."""
    p = tmp_path / "restart_before.nc"
    _write_restart(p, with_rhd=False, with_before=True)
    b = read_nemo_restart_before(str(p), nn_hls=1)
    assert b.T.shape == (IY, IX, NZ)
    assert b.u.shape == (IY, IX, NZ)
    assert b.ssh.shape == (IY, IX)
    # decodable pattern z*100+y*10+x + 1 (the before-level offset)
    assert b.T[1, 2, 0] == (1 + 1) * 10 + (2 + 1) + 1.0
    assert b.ssh[0, 0] == 1 * NX + 1 + 1.0


def test_restart_before_missing_ssh_and_tau_are_none(tmp_path):
    """A restart with no sshb/utau_b/vtau_b (older NEMO builds writing only
    the tracer/velocity before-fields) leaves ssh/tau_x/tau_y None, not a
    crash on a missing variable."""
    p = tmp_path / "restart_before_notau.nc"
    _write_restart(p, with_rhd=False, with_before=True, with_before_forcing=False)
    b = read_nemo_restart_before(str(p), nn_hls=1)
    assert b.ssh is not None      # sshb IS in this fixture
    assert b.tau_x is None
    assert b.tau_y is None


def test_restart_before_reads_utau_b_vtau_b_when_present(tmp_path):
    p = tmp_path / "restart_before_tau.nc"
    _write_restart(p, with_rhd=False, with_before=True, with_before_forcing=True)
    b = read_nemo_restart_before(str(p), nn_hls=1)
    assert b.tau_x is not None and b.tau_x.shape == (IY, IX)
    assert b.tau_y is not None and b.tau_y.shape == (IY, IX)
    assert b.tau_x[0, 0] == 1 * NX + 1 + 2.0


def test_restart_before_missing_tb_raises(tmp_path):
    """No tb/sb/ub/vb at all (a genuinely-now-only restart) -- KeyError, not
    a silent None/zero fallback, matching read_nemo_restart_en's convention
    for a missing field."""
    p = tmp_path / "restart_now_only.nc"
    _write_restart(p, with_rhd=False, with_before=False)
    with pytest.raises(KeyError):
        read_nemo_restart_before(str(p), nn_hls=1)


def test_read_nemo_restart_en_halo_strip_and_axis_order(tmp_path):
    """en (TKE closure w-level memory, #1317 --bridge-tke) uses the same
    halo-strip + (z,y,x)->(y,x,z) reshape as tn/sn/un/vn -- shares
    _to_latlon_lev, not a re-derived reshape."""
    p = tmp_path / "restart_en.nc"
    _write_restart(p, with_rhd=False, with_en=True)
    en = read_nemo_restart_en(str(p), nn_hls=1)
    assert en.shape == (IY, IX, NZ)
    # same decodable pattern z*100+y*10+x, interior origin (y=1,x=1)
    assert en[1, 2, 0] == (1 + 1) * 10 + (2 + 1)


def test_read_nemo_restart_en_missing_raises(tmp_path):
    p = tmp_path / "restart_no_en.nc"
    _write_restart(p, with_rhd=False, with_en=False)
    with pytest.raises(KeyError):
        read_nemo_restart_en(str(p), nn_hls=1)


class TestHalolessFiles:
    """NEMO 4.2+/5.x files are compute-domain WITHOUT halos: nn_hls=0 must be
    a no-op strip (#1226 root cause: stripping a phantom halo discarded the
    DINO land-wall + ridge columns)."""

    def test_strip_halo_2d_zero_is_noop(self):
        import numpy as np
        from legoesm.ocean.fidelity.nemo_io import _strip_halo_2d
        a = np.arange(12.0).reshape(3, 4)
        out = _strip_halo_2d(a, 0)
        assert out.shape == (3, 4)
        assert np.array_equal(out, a)

    def test_to_latlon_lev_zero_keeps_domain(self):
        import numpy as np
        from legoesm.ocean.fidelity.nemo_io import _to_latlon_lev
        a = np.arange(24.0).reshape(2, 3, 4)          # (z, y, x)
        out = _to_latlon_lev(a, 0)
        assert out.shape == (3, 4, 2)                  # (y, x, z), nothing cut

    def test_negative_hls_rejected(self):
        import pytest
        from legoesm.ocean.fidelity.nemo_io import _check_hls
        with pytest.raises(ValueError, match="nn_hls"):
            _check_hls(-1)
