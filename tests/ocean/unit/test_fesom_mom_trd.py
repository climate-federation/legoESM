"""FESOM box momentum budget (``--mom-trd-accumulate``, ``fesom_jax.mom_trd``).

* Read-only: with the budget on, every prognostic leaf after 4 forced steps is
  BITWISE identical to the run with it off.
* Closure: per step, ``total == sum(COMPOSED_TERMS)``; the measured residual is
  printed (``-s``) next to the largest term.  A term left out must break it.
* ``vvisc`` (read off the recomputed RHS) equals the divergence of the implicit
  solve's own interior fluxes; a wrong recomputed viscosity must break it.
* CLI: the flag is FESOM-only, forced-loop-only, and passes the FESOM gate.

Flat-bottom pi mesh, z-star, NEMO-style lateral viscosity (uniform ahm), NEMO
vertical momentum advection, a sheared flow and a lateral temperature contrast
so that every term is non-zero.  Run with ``JAX_PLATFORMS=cpu JAX_ENABLE_X64=1``.
"""
from __future__ import annotations

import dataclasses
import sys
from types import SimpleNamespace

import numpy as np
import pytest

fesom_jax = pytest.importorskip("fesom_jax")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from legoesm.ocean.state import OceanSurfaceForcing  # noqa: E402
from legoesm.ocean.dynamics.ocean_model_fesom import (  # noqa: E402
    FesomOceanConfig, FesomOceanModel, build_flat_bottom_mesh, create_rest_state,
    element_centroid_lat_lon)

DT = 900.0
NLEV = 6
H_MAX = 600.0
BOX = dict(lat_half_deg=40.0, lon0_deg=0.0, lon1_deg=360.0, z_bot_m=250.0)   # partial layer 3


@pytest.fixture(scope="module")
def mesh():
    try:
        from fesom_jax.mesh import DEFAULT_PI_MESH_DIR, load_mesh
        pi = load_mesh(mesh_dir=DEFAULT_PI_MESH_DIR)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"packaged pi mesh unavailable: {exc}")
    return build_flat_bottom_mesh(pi, H_max=H_MAX, nlev=NLEV)


def _model(mesh):
    z = SimpleNamespace(z_full_ref=np.asarray(mesh.Z, dtype=np.float64), n_levels=NLEV)
    ahm = jnp.where(jnp.asarray(mesh.node_layer_mask), 2.0e4, 0.0)
    m = FesomOceanModel(mesh, z, FesomOceanConfig(
        dt=DT, vertical_coordinate="zstar", constants="fesom",
        vertical_momentum_scheme="nemo_advective"),
        visc_nemo=(ahm, jnp.asarray(2.0)))
    return m, z


def _run(mesh, on, n=4):
    m, z = _model(mesh)
    if on:
        m.enable_mom_trd(**BOX)
    st = create_rest_state(mesh, z, stratified=True, vertical_coordinate="zstar")
    lat, lon = (np.asarray(x) for x in element_centroid_lat_lon(mesh))
    zc = -np.asarray(mesh.Z)[None, :]
    inner = st.inner
    uv = np.zeros(np.asarray(inner.uv).shape)
    uv[:, :NLEV, 0] = 0.3 * np.cos(3 * lon)[:, None] * np.exp(-zc / 200.0)
    uv[:, :NLEV, 1] = 0.1 * np.sin(2 * lat)[:, None] * np.exp(-zc / 300.0)
    uv = np.where(np.asarray(mesh.elem_layer_mask)[..., None], uv, 0.0)
    geo = np.asarray(mesh.geo_coord_nod2D)
    T = np.asarray(inner.T) + np.where(np.asarray(mesh.node_layer_mask),
                                       2.0 * np.cos(geo[:, 0])[:, None], 0.0)
    st = dataclasses.replace(st, inner=dataclasses.replace(
        inner, uv=jnp.asarray(uv), T=jnp.asarray(T), T_old=jnp.asarray(T)))
    nn = int(mesh.nod2D)
    sf = OceanSurfaceForcing(tau_x=jnp.asarray(0.1 * np.cos(geo[:, 1])),
                             tau_y=jnp.asarray(0.03 * np.sin(geo[:, 0])),
                             q_net=jnp.zeros((nn,)))
    out = []
    for _ in range(n):
        st = m.step(st, DT, surface_forcing=sf)
        out.append(m.last_mom_trd)
    return st, out


@pytest.fixture(scope="module")
def runs(mesh):
    return _run(mesh, False), _run(mesh, True)


def test_budget_leaves_prognostics_bitwise_unchanged(runs):
    (off, _), (on, _) = runs
    lo, lon_ = jax.tree.leaves(off.inner), jax.tree.leaves(on.inner)
    assert len(lo) == len(lon_)
    for a, b in zip(lo, lon_):
        a, b = np.asarray(a), np.asarray(b)
        assert np.all(np.isfinite(a)) if a.dtype.kind == "f" else True
        assert np.array_equal(a, b)
    print(f"\n[mom-trd] {len(lo)} prognostic leaves bitwise identical after 4 steps")


def test_closure_and_terms_nonzero(runs):
    from fesom_jax.mom_trd import COMPOSED_TERMS
    _, (_, terms) = runs
    for i, t in enumerate(terms):
        big = max(abs(t[k]) for k in COMPOSED_TERMS + ("total",))
        res = t["total"] - sum(t[k] for k in COMPOSED_TERMS)
        print(f"\n[mom-trd] step {i + 1}: residual {res:.3e}, largest term {big:.3e}, "
              f"vvisc {t['vvisc']:.6e} vs solve-flux check {t['vvisc_flux']:.6e}; "
              + ", ".join(f"{k}={t[k]:.2e}" for k in COMPOSED_TERMS))
        assert abs(res) <= 1e-10 * big
        assert t["residual"] == pytest.approx(res, abs=1e-12 * big)
        # independent: vvisc (solve change minus forcing) vs the solve's own fluxes
        assert abs(t["vvisc"] - t["vvisc_flux"]) <= 1e-10 * big
    last = terms[-1]
    for k in ("coriolis", "adv_h", "adv_v", "pgf", "hvisc", "stress", "vvisc",
              "ssh_expl", "ssh_impl"):
        assert last[k] != 0.0, k
    assert abs(terms[1]["ab_mismatch"]) < 1e-3 * max(abs(terms[1]["coriolis"]), abs(terms[1]["adv_h"]))


def _parser():
    from scripts.run.run_omip_core2 import _build_arg_parser
    return _build_arg_parser()


def test_cli_flag_reaches_fesom_forced_lane_only(monkeypatch):
    from scripts.run.run_omip_core2 import main, validate_fesom_stage
    p = _parser()
    base = ["--grid", "fesom", "--fesom-mesh-dir", "/m", "--output", "/o", "--no-emp"]
    assert p.parse_args(base).mom_trd_accumulate is False
    a = p.parse_args(base + ["--mom-trd-accumulate"])
    assert a.mom_trd_accumulate is True
    validate_fesom_stage(a, p)
    with pytest.raises(SystemExit):
        validate_fesom_stage(p.parse_args(base + ["--fesom-unforced", "--mom-trd-accumulate"]), p)
    monkeypatch.setattr(sys, "argv", ["x", "--grid", "tripole", "--mom-trd-accumulate"])
    with pytest.raises(SystemExit, match="mom-trd-accumulate is wired on --grid fesom only"):
        main()


def test_uniform_eastward_wind_projects_to_full_eastward_stress(mesh):
    """The pi mesh frame is rotated, so a wrong east projection shows here: a
    uniform geographic tau_x (atmospheric convention, s = -1 on the ocean) must
    give stress = -tau/rho0 in the box, and a pure tau_y nothing."""
    from fesom_jax.config import DENSITY_0
    tau = 0.1
    nn = int(mesh.nod2D)
    for tx, ty, want in ((tau, 0.0, -tau / DENSITY_0), (0.0, tau, 0.0)):
        m, z = _model(mesh)
        m.enable_mom_trd(**BOX)
        st = create_rest_state(mesh, z, stratified=True, vertical_coordinate="zstar")
        m.step(st, DT, surface_forcing=OceanSurfaceForcing(
            tau_x=jnp.full((nn,), tx), tau_y=jnp.full((nn,), ty), q_net=jnp.zeros((nn,))))
        got = m.last_mom_trd["stress"]
        print(f"\n[mom-trd] uniform tau=({tx}, {ty}): stress {got:.6e}, expected {want:.6e}")
        assert got == pytest.approx(want, abs=2e-3 * tau / DENSITY_0)
