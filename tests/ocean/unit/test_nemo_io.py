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
    read_nemo_restart_tke_coefficients,
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


def _write_mesh_mask(path, with_e3uv_0=False, with_e3w_0=False):
    d2 = (("y", "x"), np.arange(NY * NX).reshape(NY, NX).astype(np.float64))
    d3 = (("z", "y", "x"), _encode(None))
    d1 = (("z",), np.arange(NZ).astype(np.float64) + 0.5)
    data = {
        "glamt": d2, "gphit": d2, "e1t": d2, "e2t": d2, "e1u": d2, "e2v": d2,
        "ff_t": d2, "ff_f": d2,
        "e3t_1d": d1, "gdept_1d": d1, "gdepw_1d": d1,
        "tmask": d3, "umask": d3, "vmask": d3,
    }
    if with_e3uv_0:
        # distinct constant thickness per level so hu_0/hv_0 = nlev * 1.0 *
        # (mask sum) is a hand-checkable reduction, independent of _encode.
        d3_const = (("z", "y", "x"), np.ones((NZ, NY, NX)))
        data["e3u_0"] = d3_const
        data["e3v_0"] = d3_const
    if with_e3w_0:
        data["e3w_0"] = d3
        data["gdepw_0"] = d3
    ds = xr.Dataset(data)
    ds.to_netcdf(path)


def _write_restart(path, with_rhd=True, with_en=False, with_tke_coeffs=False,
                    with_before=False,
                    with_before_forcing=False, with_barotropic_velocity=False):
    d2 = (("y", "x"), np.arange(NY * NX).reshape(NY, NX).astype(np.float64))
    d3 = (("z", "y", "x"), _encode(None))
    data = {"tn": d3, "sn": d3, "un": d3, "vn": d3, "sshn": d2}
    if with_rhd:
        data["rhd"] = d3
    if with_barotropic_velocity:
        data["uu_n"] = d2
        data["vv_n"] = (("y", "x"), d2[1] + 1000.0)
    if with_en:
        data["en"] = d3
    if with_tke_coeffs:
        data["avm_k"] = (("z", "y", "x"), _encode(None) + 1000.0)
        data["avt_k"] = (("z", "y", "x"), _encode(None) + 2000.0)
        data["dissl"] = (("z", "y", "x"), _encode(None) + 3000.0)
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


def test_mesh_mask_e3w0_halo_strip_and_axis_order(tmp_path):
    p = tmp_path / "mesh_mask_e3w0.nc"
    _write_mesh_mask(p, with_e3w_0=True)
    g = read_nemo_mesh_mask(str(p), nn_hls=1)
    assert g.e3w_0 is not None
    assert g.e3w_0.shape == (IY, IX, NZ)
    assert g.e3w_0[1, 2, 2] == 2 * 100 + (1 + 1) * 10 + (2 + 1)
    assert g.gdepw_0 is not None
    assert g.gdepw_0.shape == (IY, IX, NZ)
    assert g.gdepw_0[1, 2, 2] == 2 * 100 + (1 + 1) * 10 + (2 + 1)


def test_mesh_mask_hu0_hv0_derived_from_e3u0_e3v0(tmp_path):
    """#1226 item 2: hu_0/hv_0 = sum_k(e3u_0*umask) / sum_k(e3v_0*vmask)
    (domain.F90:140-146), derived by the reader when e3u_0/e3v_0 are
    present -- the r3u/r3v comparison needs these and mesh_mask.nc never
    carries hu_0/hv_0 directly."""
    p = tmp_path / "mesh_mask_e3uv0.nc"
    _write_mesh_mask(p, with_e3uv_0=True)
    g = read_nemo_mesh_mask(str(p), nn_hls=1)
    assert g.e3u_0 is not None and g.e3u_0.shape == (IY, IX, NZ)
    assert g.e3v_0 is not None and g.e3v_0.shape == (IY, IX, NZ)
    assert g.hu_0 is not None and g.hu_0.shape == (IY, IX)
    assert g.hv_0 is not None and g.hv_0.shape == (IY, IX)
    # e3u_0 == 1.0 everywhere -> hu_0 = sum_k(umask) at that column.
    umask_sum = g.umask.sum(axis=-1)
    vmask_sum = g.vmask.sum(axis=-1)
    np.testing.assert_allclose(g.hu_0, umask_sum)
    np.testing.assert_allclose(g.hv_0, vmask_sum)


def test_mesh_mask_hu0_hv0_none_when_e3uv0_absent(tmp_path):
    """No silent zero/garbage fallback: absent e3u_0/e3v_0 -> None, matching
    the existing e3t_0/gdept_0 optional-field convention."""
    p = tmp_path / "mesh_mask_no_e3uv0.nc"
    _write_mesh_mask(p, with_e3uv_0=False)
    g = read_nemo_mesh_mask(str(p), nn_hls=1)
    assert g.e3u_0 is None
    assert g.e3v_0 is None
    assert g.hu_0 is None
    assert g.hv_0 is None


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


def test_restart_reads_prognostic_depth_mean_pair(tmp_path):
    """restart.F90:175-182 / :304-314 uu_n/vv_n stay separate from un/vn."""
    p = tmp_path / "restart_baro.nc"
    _write_restart(p, with_rhd=False, with_barotropic_velocity=True)
    s = read_nemo_restart(str(p), nn_hls=1)
    assert s.uu_b.shape == (IY, IX)
    assert s.vv_b.shape == (IY, IX)
    assert s.uu_b[1, 2] == (1 + 1) * NX + (2 + 1)
    assert s.vv_b[1, 2] == 1000.0 + (1 + 1) * NX + (2 + 1)


def test_restart_rejects_half_depth_mean_pair(tmp_path):
    p = tmp_path / "restart_half_baro.nc"
    _write_restart(p, with_rhd=False, with_barotropic_velocity=True)
    with xr.open_dataset(p, decode_times=False) as ds:
        altered = ds.drop_vars("vv_n").load()
    altered.to_netcdf(p, mode="w")
    with pytest.raises(ValueError, match="both uu_n and vv_n"):
        read_nemo_restart(str(p), nn_hls=1)


def test_restart_malformed_depth_mean_does_not_use_missing_field_fallback(
        tmp_path):
    """restart.F90:311-323 falls back only when uu_n is absent, not unreadable."""
    p = tmp_path / "restart_bad_baro.nc"
    _write_restart(p, with_rhd=False, with_barotropic_velocity=True)
    with xr.open_dataset(p, decode_times=False) as ds:
        altered = ds.load()
    altered["uu_n"] = (("y", "x"), np.full((NY, NX), "not-a-real"))
    altered.to_netcdf(p, mode="w")
    with pytest.raises((TypeError, ValueError)):
        read_nemo_restart(str(p), nn_hls=1)


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


def test_read_nemo_restart_tke_coefficients_axis_order_and_missing(tmp_path):
    p = tmp_path / "restart_coeff.nc"
    _write_restart(p, with_rhd=False, with_tke_coeffs=True)
    avm, avt, dissl = read_nemo_restart_tke_coefficients(str(p), nn_hls=1)
    assert avm.shape == avt.shape == dissl.shape == (IY, IX, NZ)
    assert avm[1, 2, 0] == 1023.0
    assert avt[1, 2, 0] == 2023.0
    assert dissl[1, 2, 0] == 3023.0
    q = tmp_path / "restart_no_coeff.nc"
    _write_restart(q, with_rhd=False)
    with pytest.raises(ValueError, match="avm_k"):
        read_nemo_restart_tke_coefficients(str(q), nn_hls=1)


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
