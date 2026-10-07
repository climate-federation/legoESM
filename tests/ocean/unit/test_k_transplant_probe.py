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
    for v, sc in (("avt", 1.0), ("avm", 2.0)):
        d.createVariable(v, "f4", ("t", "z", "y", "x"))[:] = a * sc
    d.close()
    lat = np.zeros((6, 8)); lon = np.full((6, 8), 230.0); lm = np.ones((6, 8))
    jj, ii, avt, avm = _core2().load_nemo_strip_transplant(
        str(p), lat, lon, lm, n_iface=4, j0=2, i0=3)
    assert len(jj) == 3 * 4 and avt.shape == (2, 12, 4)
    c = int(np.nonzero((jj == 3) & (ii == 5))[0][0])        # strip (1, 2)
    np.testing.assert_array_equal(avt[:, c, :3], a[:, 1:4, 1, 2])   # ours k = w-level k+1
    assert np.isnan(avt[:, c, 3]).all()                          # fill -> NaN
    np.testing.assert_array_equal(avm[:, c, :3], 2 * a[:, 1:4, 1, 2])


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
