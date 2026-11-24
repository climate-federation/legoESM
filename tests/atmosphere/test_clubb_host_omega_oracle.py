"""CLUBB's wm from the host omega vs CAM6's clubb_intr (oracle fixture).

Fixture ``oracle_fixtures/clubb_host_omega_cam.npz`` was produced by RUNNING
the verbatim clubb_intr.F90 lines of ESCOMP/CAM cam_cesm2_1_rel_60 (:1626-1674
grid and rho_in = pdel/(g dz), :1677-1680 wm_zt = -omega/(rho_in g), :1778
wm_zm = zt2zm_api(wm_zt)) with CLUBB's grid_type 3 grid and
linear_interpolated_azm (clubb_release 324635f52, the ghost-level grid of the
r8099 era), compiled with gfortran in fp64 (legoESM constants in the stubs).
Generator: scripts/validate/oracle/clubb_host_omega/ (build.sh +
make_fixture.py).  Columns: 12 from an A2 C1 smoke dump (SE Pacific, warm
pool, high terrain) with the dumped T, q_v, p_s and wap; pressures and
heights are the MPAS bridge's own.

Metric: max |ours - oracle| / |oracle| point by point, bound 1e-10 in fp64.
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics._shared import compute_rho  # noqa: E402
from legoesm.atmosphere.physics.turbulence import clubb as C  # noqa: E402, N812

_FIX = Path(__file__).parent / "oracle_fixtures" / "clubb_host_omega_cam.npz"
_TOL = 1.0e-10


def _legoesm_wm(monkeypatch, fx):
    got = {}
    real_core = C.advance_clubb_core

    def core(*a, **k):
        got.update(wm_zt=np.asarray(k["wm_zt"]), wm_zm=np.asarray(k["wm_zm"]))
        return real_core(*a, **k)

    monkeypatch.setattr(C, "advance_clubb_core", core)
    T = jnp.asarray(fx["T"])
    q_v = jnp.asarray(fx["q_v"])
    ncol, nlev = T.shape
    cfg = C.CLUBBConfig(trop_cloud_top_press=0.0, prognostic=True)
    zeros = jnp.zeros_like(T)
    C.clubb_step(zeros + 5.0, zeros - 3.0, T, q_v,
                 C.init_clubb_moments(ncol, nlev, cfg, dtype=jnp.float64),
                 jnp.asarray(fx["p_full"]), jnp.asarray(fx["p_half"]),
                 jnp.asarray(fx["z_full"]), jnp.asarray(fx["z_half"]),
                 T[:, -1] + 1.0, q_v[:, -1],
                 compute_rho(T, jnp.asarray(fx["p_full"]), q_v=q_v), 300.0, cfg,
                 omega=jnp.asarray(fx["omega"]), omega_cam_layer_density=True)
    return got


def _rel(ours, ref):
    nz = ref != 0.0
    assert nz.mean() > 0.9
    return float(np.max(np.abs(ours[nz] - ref[nz]) / np.abs(ref[nz]))), ours[~nz]


def test_wm_matches_cam6(monkeypatch):
    fx = np.load(_FIX)
    got = _legoesm_wm(monkeypatch, fx)
    # CAM's level 1 is the ghost below the surface (wm_zt = 0); legoESM's grid
    # starts at CAM's level 2.  Momentum levels coincide one to one.
    assert np.all(fx["wm_zt_cam"][:, 0] == 0.0)
    rel_t, zero_t = _rel(got["wm_zt"], fx["wm_zt_cam"][:, 1:])
    rel_m, zero_m = _rel(got["wm_zm"], fx["wm_zm_cam"])
    assert rel_t < _TOL, rel_t
    assert rel_m < _TOL, rel_m
    assert not np.any(zero_t) and not np.any(zero_m)


def test_the_fixture_tells_the_density_and_surface_apart(monkeypatch):
    """Non-vacuous: the bridge's ideal-gas rho at its full-level pressures
    (7e-8 off CAM's layer density on this grid) and a copied surface value
    would both miss the oracle by far more than the bound."""
    fx = np.load(_FIX)
    got = _legoesm_wm(monkeypatch, fx)
    T, q_v = jnp.asarray(fx["T"]), jnp.asarray(fx["q_v"])
    rho = np.asarray(compute_rho(T, jnp.asarray(fx["p_full"]), q_v=q_v))[:, ::-1]
    from legoesm import constants
    wm_rho = -np.asarray(fx["omega"])[:, ::-1] / (rho * constants.g)
    assert _rel(wm_rho, fx["wm_zt_cam"][:, 1:])[0] > 1e2 * _TOL
    copied = got["wm_zt"][:, 0]
    assert _rel(copied, fx["wm_zm_cam"][:, 0])[0] > 0.4
