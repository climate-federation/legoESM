"""``--probe-k-transplant``: an oracle's per-step avt/avm replaces the applied
interface coefficients at fixed columns (probe only)."""
from __future__ import annotations

import importlib.util
import os
import pathlib
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

_spec = importlib.util.spec_from_file_location(
    "_trd_fixture", pathlib.Path(__file__).with_name("test_trd_zdf_accumulate.py"))
_fx = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_fx)
_fp64, _setup, _DT, _core2 = _fx._fp64, _fx._setup, _fx._DT, _fx._core2


def _run(m, st, tke0, src):
    got = []
    out = jax.jit(lambda s: m._apply_implicit_vertical_mixing(
        s, _DT, None, tke_old=tke0, return_tke=True, surface_tracer_forcing=src,
        col_callback=lambda *a: got.append([np.asarray(x) for x in a]))[0])(st)
    jax.effects_barrier()
    return out, got[0][4], got[0][5]          # state, applied K, applied A


@pytest.mark.parametrize("mode", ["both", "avt", "avm"])
def test_transplant_replaces_only_the_selected_coefficients(mode):
    m, st, tke0, src, rate, lm, dz = _setup()
    nz = st.u.data.shape[-1]                   # sheared flow: momentum solve consumes A
    st = st._replace(u=st.u.replace(data=st.u.data + 0.1 * jnp.linspace(1.0, 0.0, nz)))
    out0, K0, A0 = _run(m, st, tke0, src)
    nk = K0.shape[-1]
    j, i = np.array([3, 4]), np.array([5, 9])
    avt = np.full((1, 2, nk), 0.37); avm = np.full((1, 2, nk), 0.11)
    avt[0, 1, -1] = np.nan                     # oracle gap -> keep ours
    m._k_transplant = _core2()._KTransplant(j, i, avt, avm, mode)
    out1, K1, A1 = _run(m, st, tke0, src)
    heat, mom = mode in ("both", "avt"), mode in ("both", "avm")
    if heat:
        np.testing.assert_array_equal(K1[3, 5], 0.37)
        np.testing.assert_array_equal(K1[4, 9, :-1], 0.37)
        np.testing.assert_array_equal(K1[4, 9, -1], K0[4, 9, -1])
    else:
        np.testing.assert_array_equal(K1, K0)
    if mom:
        np.testing.assert_array_equal(A1[j, i], 0.11)
    else:
        np.testing.assert_array_equal(A1, A0)
    keep = np.ones(K0.shape[:2], bool); keep[j, i] = False
    np.testing.assert_array_equal(K1[keep], K0[keep])
    np.testing.assert_array_equal(A1[keep], A0[keep])
    # non-vacuous: the solve consumed the transplant
    dT = np.abs(np.asarray(out1.T.data) - np.asarray(out0.T.data)).max()
    du = np.abs(np.asarray(out1.u.data) - np.asarray(out0.u.data)).max()
    assert (dT > 1e-9) == heat and (du > 1e-9) == mom
    assert m._k_transplant.step == 1


def test_counter_advances_per_step_and_refuses_overrun():
    r = np.arange(2.0)[:, None, None] * np.ones((2, 1, 3))
    x = _core2()._KTransplant([0], [0], r, r + 10.0, "both")
    np.testing.assert_array_equal(x.fetch(), [[[0.0] * 3], [[10.0] * 3]])
    np.testing.assert_array_equal(x.fetch(), [[[1.0] * 3], [[11.0] * 3]])
    x.check_consumed(2)
    with pytest.raises(RuntimeError, match="oracle has 2 steps"):
        x.fetch()
    with pytest.raises(RuntimeError, match="served 2 oracle rows for 3"):
        x.check_consumed(3)                    # planted under-run
    with pytest.raises(ValueError, match="unknown"):
        _core2()._KTransplant([0], [0], np.zeros((1, 1, 3)), np.zeros((1, 1, 3)), "K")


def test_loader_maps_columns_levels_and_fill(tmp_path):
    import netCDF4 as nc
    p = tmp_path / "w.nc"
    d = nc.Dataset(p, "w")
    for n, s in (("t", 2), ("z", 5), ("y", 3), ("x", 4)):
        d.createDimension(n, s)
    a = np.arange(2 * 5 * 3 * 4, dtype=np.float32).reshape(2, 5, 3, 4)
    a[:, 4, 1, 2] = 1e20
    for v, sc in (("avt", 1.0), ("avm", 2.0), ("avt_k", 0.5)):
        d.createVariable(v, "f4", ("t", "z", "y", "x"))[:] = a * sc
    d.close()
    lat = np.zeros((6, 8)); lon = np.full((6, 8), 230.0); lm = np.ones((6, 8))
    jj, ii, avt, avm, avt_k = _core2().load_nemo_strip_transplant(
        str(p), lat, lon, lm, n_iface=4, j0=2, i0=3)
    assert len(jj) == 3 * 4 and avt.shape == (2, 12, 4)
    c = int(np.nonzero((jj == 3) & (ii == 5))[0][0])        # strip (1, 2)
    np.testing.assert_array_equal(avt[:, c, :3], a[:, 1:4, 1, 2])   # ours k = w-level k+1
    assert np.isnan(avt[:, c, 3]).all()                          # fill -> NaN
    np.testing.assert_array_equal(avm[:, c, :3], 2 * a[:, 1:4, 1, 2])
    np.testing.assert_array_equal(avt_k[:, c, :3], 0.5 * a[:, 1:4, 1, 2])
    x = _core2()._KTransplant(jj, ii, avt, avm, "noncl", avt_k=avt_k)
    np.testing.assert_array_equal(x.fetch()[0], avt[0] - avt_k[0])
    with pytest.raises(ValueError, match="needs the oracle avt_k"):
        _core2()._KTransplant(jj, ii, avt, avm, "avt_k")


def test_non_tripole_grid_is_refused(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_omip_core2.py", "--grid", "mpas",
                                      "--probe-k-transplant", "x.nc", "both"])
    with pytest.raises(SystemExit, match="tripole only"):
        _core2().main()


def test_split_solve_is_refused():
    m, st, tke0, src, *_ = _setup()
    m._k_transplant = _core2()._KTransplant([3], [5], np.zeros((1, 1, 5)), np.zeros((1, 1, 5)), "both")
    with pytest.raises(NotImplementedError, match="combined"):
        m._apply_implicit_vertical_mixing(st, _DT, None, tke_old=tke0, return_tke=True,
                                          surface_tracer_forcing=src, do_momentum=False)


@pytest.mark.parametrize("extra", [["--scan-block", "8"], ["--restart-from", "r.npz"]])
def test_scan_block_and_resume_are_refused(monkeypatch, extra):
    monkeypatch.setattr(sys, "argv", ["run_omip_core2.py", "--grid", "tripole",
                                      "--probe-k-transplant", "x.nc", "both"] + extra)
    with pytest.raises(SystemExit, match="fresh run"):
        _core2().main()


def _iwm_model(partial, iwm_on):
    """The trd fixture's model with IWM forcing, on z* or on partial cells
    with shallow columns (so the sub-seafloor extrapolation matters)."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.physics.vertical_mixing.internal_wave_mixing import (
        IWMConfig, uniform_iwm_forcing)
    from legoesm.ocean.vertical import create_partial_cell_coordinate
    m, st, tke0, src, *_ = _setup()
    iwm = IWMConfig(enabled=iwm_on, power_nsq_wm2=1.0e-3, power_bot_wm2=1.0e-3)
    phys = m.config.physics._replace(vertical_mixing=m.config.physics.vertical_mixing._replace(iwm=iwm))
    z = m.z_coord
    if partial:
        H = np.full(st.H_bathy.data.shape, 600.0); H[:, ::3] = 350.0
        st = st._replace(H_bathy=st.H_bathy.replace(data=jnp.asarray(H)))
        z = create_partial_cell_coordinate(z, jnp.asarray(H))
    m2 = LatLonCGridOceanModel(m.grid, z, m.config._replace(physics=phys),
                               iwm_forcing=uniform_iwm_forcing(iwm, st.T.data.shape[:2]) if iwm_on else None)
    m2._ensure_rigid_lid_data(st)
    return m2, st, tke0, src


@pytest.mark.parametrize("partial", [False, True])
def test_split_modes_recover_our_closure_and_our_iwm_exactly(partial):
    m_on, st, tke0, src = _iwm_model(partial, True)
    m_off, *_ = _iwm_model(partial, False)
    _, K_on, _ = _run(m_on, st, tke0, src)
    _, K_off, _ = _run(m_off, st, tke0, src)
    j, i = np.array([3, 4, 5]), np.array([0, 9, 3])          # incl. shallow columns
    nk = K_on.shape[-1]
    assert np.abs(K_on[j, i] - K_off[j, i]).max() > 1e-6     # IWM is not a no-op here
    zero = np.zeros((1, 3, nk))
    m_on._k_transplant = _core2()._KTransplant(j, i, zero, zero, "noncl", avt_k=zero)
    _, K_cl, _ = _run(m_on, st, tke0, src)                   # our closure + 0
    np.testing.assert_allclose(K_cl[j, i], K_off[j, i], rtol=0, atol=1e-15)
    m_on._k_transplant = _core2()._KTransplant(j, i, zero, zero, "avt_k", avt_k=zero)
    _, K_iw, _ = _run(m_on, st, tke0, src)                   # 0 + our IWM
    np.testing.assert_allclose(K_iw[j, i], K_on[j, i] - K_off[j, i], rtol=0, atol=1e-15)


def _run_e(m, st, tke0, src):
    got = []
    jax.jit(lambda s: m._apply_implicit_vertical_mixing(
        s, _DT, None, tke_old=tke0, return_tke=True, surface_tracer_forcing=src,
        col_callback=lambda *a: got.append([np.asarray(x) for x in a]))[0])(st)
    jax.effects_barrier()
    return got[0][5], got[0][6]                 # applied A, post-step e


def test_tke_transplant_class_and_loader(tmp_path):
    import netCDF4 as nc
    en = np.full((3, 1, 2), 4e-4); dl = np.full((3, 1, 2), 0.02); dl[1, 0, 1] = 0.0
    x = _core2()._TKETransplant([0], [0], en, dl, "l")
    r0, r1 = x.fetch(), x.fetch()
    np.testing.assert_allclose(r0[1], 1.0)                        # l = sqrt(4e-4)/0.02
    assert np.isnan(r0[2]).all() and np.isnan(r1[1, 0, 1])        # l(kt-1) at step 1; dissl 0
    np.testing.assert_allclose(r1[2], r0[1])                      # l(kt-1) = previous l(kt)
    with pytest.raises(ValueError, match="unknown"):
        _core2()._TKETransplant([0], [0], en, dl, "K")
    p = tmp_path / "en.nc"
    d = nc.Dataset(p, "w")
    for n, s in (("t", 2), ("z", 5), ("y", 3), ("x", 4)):
        d.createDimension(n, s)
    for v, val in (("en_k", 1e-4), ("dissl_k", 1e-2)):
        d.createVariable(v, "f4", ("t", "z", "y", "x"))[:] = val
    d.close()
    out = _core2().load_nemo_strip_transplant(
        str(p), np.zeros((6, 8)), np.full((6, 8), 230.0), np.ones((6, 8)), 4,
        j0=2, i0=3, names=("en_k", "dissl_k"))
    assert len(out) == 4 and out[2].shape == (2, 12, 4)


def test_tke_transplant_energy_mode_sits_after_etau_and_feeds_K():
    m, st, tke0, src, *_ = _setup()
    A0, e0 = _run_e(m, st, tke0, src)
    j, i = np.array([3, 4]), np.array([5, 9])
    nk = e0.shape[-1]
    ident = e0[j, i][None]                                         # our own post-step e
    m._tke_transplant = _core2()._TKETransplant(j, i, ident, np.ones((1, 2, nk)), "e")
    A1, e1 = _run_e(m, st, tke0, src)
    np.testing.assert_array_equal(e1, e0)                          # identity: bitwise neutral
    np.testing.assert_array_equal(A1, A0)
    m._tke_transplant = _core2()._TKETransplant(j, i, 4.0 * ident, np.ones((1, 2, nk)), "e")
    A2, e2 = _run_e(m, st, tke0, src)
    np.testing.assert_array_equal(e2[j, i], 4.0 * e0[j, i])        # replaced e is what is carried
    assert np.abs(A2[j, i] - A0[j, i]).max() > 1e-6                # and K is built from it
    keep = np.ones(e0.shape[:2], bool); keep[j, i] = False
    np.testing.assert_array_equal(e2[keep], e0[keep])
    assert m._tke_transplant.step == 1


def test_tke_transplant_length_mode_sets_K_from_oracle_length():
    m, st, tke0, src, *_ = _setup()
    A0, e0 = _run_e(m, st, tke0, src)
    j, i = np.array([3, 4]), np.array([5, 9])
    nk = e0.shape[-1]
    L = 2.0
    en = np.full((1, 2, nk), 1e-4); dl = np.sqrt(en) / L          # oracle l = L
    m._tke_transplant = _core2()._TKETransplant(j, i, en, dl, "l")
    A1, e1 = _run_e(m, st, tke0, src)
    cfg = m.config.physics.vertical_mixing.tke
    want = m.config.A_v + cfg.c_k * L * np.sqrt(2.0 * np.maximum(e1[j, i], cfg.tke_background))
    np.testing.assert_allclose(A1[j, i], want, rtol=1e-12)
    assert np.abs(A1[j, i] - A0[j, i]).max() > 1e-6


def test_tke_transplant_refusals(monkeypatch):
    for extra, msg in ((["--grid", "mpas"], "needs --grid tripole"),
                       (["--grid", "tripole", "--tripole-vmix", "tke", "--tke-prognostic",
                         "--scan-block", "8"], "fresh run")):
        monkeypatch.setattr(sys, "argv", ["run_omip_core2.py"] + extra
                            + ["--probe-tke-transplant", "x.nc", "e"])
        with pytest.raises(SystemExit, match=msg):
            _core2().main()


@pytest.mark.parametrize("mode,e_changed,entry_l", [("lf", False, False), ("l", True, True), ("el", True, True)])
def test_tke_transplant_split_length_modes(mode, e_changed, entry_l):
    """lf touches only the final length (our post-step e is bitwise ours); l and
    el also set the step-entry length (so e moves); el also replaces e."""
    m, st, tke0, src, *_ = _setup()
    A0, e0 = _run_e(m, st, tke0, src)
    j, i = np.array([3, 4]), np.array([5, 9])
    nk = e0.shape[-1]
    en = np.full((2, 2, nk), 3e-4); dl = np.sqrt(en) / 2.0
    m._tke_transplant = _core2()._TKETransplant(j, i, en, dl, mode)
    m._tke_transplant.fetch()                       # step 2: l(kt-1) is defined
    A1, e1 = _run_e(m, st, tke0, src)
    if mode == "el":
        np.testing.assert_array_equal(e1[j, i], 3e-4)
    elif mode == "lf":
        np.testing.assert_array_equal(e1, e0)
    else:
        assert np.abs(e1[j, i] - e0[j, i]).max() > 1e-12
    cfg = m.config.physics.vertical_mixing.tke
    want = m.config.A_v + cfg.c_k * 2.0 * np.sqrt(2.0 * np.maximum(e1[j, i], cfg.tke_background))
    np.testing.assert_allclose(A1[j, i], want, rtol=1e-12)


def test_tke_energy_transplant_sits_after_etau():
    """With the sub-mixed-layer injection ON, an identity transplant of our own
    post-step e is bitwise neutral only if the overwrite comes after etau."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    m, st, tke0, src, *_ = _setup()
    vm = m.config.physics.vertical_mixing
    phys = m.config.physics._replace(vertical_mixing=vm._replace(tke=vm.tke._replace(etau_mode="below_ml")))
    m = LatLonCGridOceanModel(m.grid, m.z_coord, m.config._replace(physics=phys))
    m._ensure_rigid_lid_data(st)
    A0, e0 = _run_e(m, st, tke0, src)
    m_off, *_ = _setup()
    _, e_noetau = _run_e(m_off, st, tke0, src)
    j, i = np.array([3, 4]), np.array([5, 9])
    assert np.abs(e0[j, i] - e_noetau[j, i]).max() > 1e-9          # etau is live here
    m._tke_transplant = _core2()._TKETransplant(j, i, e0[j, i][None], np.ones((1, 2, e0.shape[-1])), "e")
    A1, e1 = _run_e(m, st, tke0, src)
    np.testing.assert_array_equal(e1, e0)
    np.testing.assert_array_equal(A1, A0)
