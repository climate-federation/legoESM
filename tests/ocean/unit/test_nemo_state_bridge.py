"""Unit tests for the NEMO -> legoESM state bridge (``fidelity.nemo_state_bridge``).

Self-contained: builds a small synthetic beta-plane ``NemoGrid``/``NemoState``
(no external NEMO run) and checks the geometry (Coriolis reconstruction), the
velocity staggering (NEMO east/north face -> legoESM u/v faces), and the state
placement.
"""
import numpy as np

from legoesm.ocean.fidelity.nemo_io import NemoGrid, NemoState
from legoesm.ocean.fidelity.nemo_state_bridge import bridge_nemo_to_legoesm

NY, NX, NZ = 4, 5, 3
DXY = 1.0e5
F0_TRUE, BETA_TRUE = 1.0e-4, 1.0e-5   # per-cell f increment over dy


def _synthetic():
    ones2 = np.ones((NY, NX))
    # ff_t linear in y: f(y_c[j]) with y_c[j]=(j+0.5)*dy  -> f0 + beta*(j+0.5)*dy
    yc = (np.arange(NY) + 0.5) * DXY
    ff_t = (F0_TRUE + BETA_TRUE * yc)[:, None] * np.ones((1, NX))
    # ff_f lives at the F-point (NE corner) = y_g[j] = j*dy, a half-cell south of
    # y_c[j]; same beta slope (what the bridge checks), physically offset value.
    yg = np.arange(NY) * DXY
    ff_f = (F0_TRUE + BETA_TRUE * yg)[:, None] * np.ones((1, NX))
    grid = NemoGrid(
        glamt=ones2 * np.arange(NX)[None, :], gphit=ones2 * np.arange(NY)[:, None],
        e1t=ones2 * DXY, e2t=ones2 * DXY, e1u=ones2 * DXY, e2v=ones2 * DXY,
        ff_t=ff_t, ff_f=ff_f,
        e3t_1d=np.array([10.0, 20.0, 30.0]),
        gdept_1d=np.array([5.0, 20.0, 45.0]),
        gdepw_1d=np.array([0.0, 10.0, 30.0]),
        tmask=np.ones((NY, NX, NZ)), umask=np.ones((NY, NX, NZ)),
        vmask=np.ones((NY, NX, NZ)),
    )
    rng = np.random.default_rng(0)
    state = NemoState(
        T=15.0 + rng.random((NY, NX, NZ)), S=35.0 + rng.random((NY, NX, NZ)),
        u=rng.random((NY, NX, NZ)), v=rng.random((NY, NX, NZ)),
        ssh=0.01 * rng.random((NY, NX)), rhd=None,
    )
    return grid, state


def test_bridge_geometry_and_staggering():
    grid, state = _synthetic()
    out = bridge_nemo_to_legoesm(grid, state)

    # Coriolis reconstructed to ~roundoff (ff_t is exactly the beta-plane f).
    assert out.f_match_max_abs < 1e-12, out.f_match_max_abs

    st = out.state
    assert st.T.data.shape == (NY, NX, NZ)
    assert st.u.data.shape == (NY, NX + 1, NZ)      # west-face array, n_lon+1
    assert st.v.data.shape == (NY + 1, NX, NZ)      # south-face array, n_lat+1
    assert st.eta.data.shape == (NY, NX)

    # Staggering. legoESM u[j,i] is the WEST face of T-cell (j,i)
    # (latlon_cgrid_operators.py:3623); NEMO u(i) is the EAST face = west face of
    # cell i+1. So NEMO u[k-1] must land at u_face[:,k] (west face of cell k),
    # i.e. u_face[:,1:]==nemo_u with a west-wall column at index 0. A prepend/
    # append flip of this convention is caught here; the end-to-end validation is
    # the momentum-tendency match against NEMO.
    u = np.asarray(st.u.data)
    assert np.allclose(u[:, 0, :], 0.0)             # west wall
    assert np.allclose(u[:, 1:, :], state.u)        # interior + east wall = NEMO u
    # NEMO north-face v -> v_face[1:], south wall zero.
    v = np.asarray(st.v.data)
    assert np.allclose(v[0, :, :], 0.0)             # south wall
    assert np.allclose(v[1:, :, :], state.v)
    # Divergence-identity guard: under legoESM's west-face stencil the bridged
    # zonal convergence at an interior cell equals NEMO's east-face difference
    # (fails if the prepend flips to an append). Use a NONLINEAR u so a shift
    # changes the interior value.
    nemo_u_nl = (np.arange(NX) ** 2).astype(float)[None, :, None] * np.ones_like(state.u)
    out_nl = bridge_nemo_to_legoesm(grid, state._replace(u=nemo_u_nl))
    uf = np.asarray(out_nl.state.u.data)
    for k in range(1, NX):                          # interior west faces
        lego_diff = uf[:, k + 1, :] - uf[:, k, :]   # west-face stencil
        nemo_diff = nemo_u_nl[:, k, :] - nemo_u_nl[:, k - 1, :]
        assert np.allclose(lego_diff, nemo_diff), k

    # T/S placed directly at T-points.
    assert np.allclose(np.asarray(st.T.data), state.T)
    assert np.allclose(np.asarray(st.S.data), state.S)


def test_bridge_rejects_bad_coriolis():
    import pytest
    grid, state = _synthetic()
    # Corrupt ff_t so it is no longer the linear beta-plane the fit assumes.
    bad = grid.ff_t.copy()
    bad[2, 1] += 5e-5
    grid = grid._replace(ff_t=bad)
    with pytest.raises(ValueError, match="Coriolis mismatch"):
        bridge_nemo_to_legoesm(grid, state)
