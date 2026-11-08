"""ORCA1 oracle: the MLE potential density comes from the run's EOS.

NEMO 5.0.1 ORCA1 (``ln_teos10=.true.``) feeds ``tra_mle_trp`` the run's
``rhop`` (``eos_insitu_pot``, surface-referenced potential density).  The
combined-physics MLE bridge used the Wright default whatever EOS the run
declared (tier-3 audit finding M1: ML level wrong in 4.3% of columns).

Fixture ``fixtures/orca1_t3/mle_rhop.npz`` (+ ``.json`` provenance): three
6x6 column patches from the tier-3 MLE record (kt 17281, day-30 restart,
32-rank run) -- two around Wright-mismatch columns, one seeded random --
holding NEMO's Kmm T/S/ssh, geometry, ``rhop`` and ``inml_mle``.

The test runs the production chain ``make_ocean_physics(..., eos_fn=E)`` with
E = the run's EOS (``nemo_teos10``) and checks the sub-quantity (campaign vote
C6) at the campaign tolerance (C2): ``rho_pot`` vs NEMO ``rhop`` at rel
<= 1e-12 (measured 8.9e-16 over all 32 ranks), and the ML level == NEMO's
``inml_mle`` exactly (NEMO 20-seed spread on ``inml_mle``: 0).
"""
from __future__ import annotations

import pathlib
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

FIX = pathlib.Path(__file__).parent / "fixtures" / "orca1_t3" / "mle_rhop.npz"
TOL = 1e-12


@pytest.fixture(autouse=True)
def _fp64_policy():
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    saved = get_policy()
    set_policy(PrecisionPolicy.fp64())
    yield
    set_policy(saved)


class _State(types.SimpleNamespace):
    def _replace(self, **kw):
        d = dict(self.__dict__)
        d.update(kw)
        return _State(**d)


def _F(x):
    return types.SimpleNamespace(data=x)


def _patch(z, n):
    from legoesm.grids.tripole import create_synthetic_tripole
    from legoesm.ocean.vertical import (
        create_partial_cell_coordinate, create_z_star_from_thicknesses)
    tm = z[f"p{n}_tmask"]
    e3t3 = z[f"p{n}_e3t_3d"]
    mbkt = z[f"p{n}_mbkt"]
    nlat, nlon, _ = tm.shape
    zs = create_z_star_from_thicknesses(
        jnp.asarray(z["e3t_1d"]), jnp.asarray(z["gdept_1d"]),
        nemo_e3w_source="depth_difference")
    H = np.sum(e3t3 * tm, axis=-1)
    tmpl = create_partial_cell_coordinate(zs, jnp.asarray(H))
    f64 = lambda x: jnp.asarray(np.asarray(x, np.float64))  # noqa: E731
    # Production --nemo-vertical passes no gdept ladder (t_depth_ref None).
    zc = tmpl._replace(
        z_full_ref=f64(zs.z_full_ref), z_half_ref=f64(zs.z_half_ref),
        dz_ref=f64(zs.dz_ref), dz_half_ref=f64(zs.dz_half_ref),
        t_depth_ref=None, h_partial=jnp.asarray(e3t3 * tm),
        bottom_level=jnp.asarray(np.where(tm[..., 0] > 0, mbkt - 1, -1),
                                 dtype=tmpl.bottom_level.dtype),
        is_active=jnp.asarray(tm > 0, dtype=tmpl.is_active.dtype))
    T = jnp.asarray(z[f"p{n}_T"])
    st = _State(T=_F(T), S=_F(jnp.asarray(z[f"p{n}_S"])),
                u=_F(jnp.zeros_like(T)), v=_F(jnp.zeros_like(T)),
                eta=_F(jnp.asarray(z[f"p{n}_ssh"])), H_bathy=_F(jnp.asarray(H)),
                land_mask=_F(jnp.asarray(tm[..., 0])))
    grid = create_synthetic_tripole(n_lat=nlat, n_lon=nlon, dtype=jnp.float64)
    return st, grid, zc, tm


def _run_production_chain(st, grid, zc, eos_fn, monkeypatch):
    """make_ocean_physics (MLE only) -> capture rho_pot and the ML mask the
    production MLE kernel builds from it."""
    import legoesm.ocean.physics.lateral_mixing.mle_latlon_cgrid as M
    from legoesm.ocean.physics.combined import (
        OceanPhysicsConfig, make_ocean_physics)
    from legoesm.ocean.physics.lateral_mixing.mle import MLEConfig
    cap = {}
    real = M.mle_mld_and_buoyancy

    def spy(rho_pot, *a, **k):
        out = real(rho_pot, *a, **k)
        cap["rho_pot"], cap["in_ml"] = rho_pot, out[2]
        return out

    monkeypatch.setattr(M, "mle_mld_and_buoyancy", spy)
    b = OceanPhysicsConfig()
    cfg = b._replace(
        vertical_mixing=b.vertical_mixing._replace(scheme="none"),
        lateral_mixing=b.lateral_mixing._replace(scheme="none"),
        surface_forcing=b.surface_forcing._replace(scheme="none"),
        bottom_drag=b.bottom_drag._replace(scheme="none"),
        convection=b.convection._replace(scheme="none"),
        shortwave_penetration=None, mle=MLEConfig(ce=0.06))
    make_ocean_physics(cfg, eos_fn=eos_fn)(st, grid, zc)
    return (np.asarray(cap["rho_pot"]),
            np.sum(np.asarray(cap["in_ml"]), axis=-1) + 1.0)


def _rel(a, b):
    d = np.maximum(np.abs(a), np.abs(b))
    return np.where(d > 0, np.abs(a - b) / np.where(d > 0, d, 1.0), 0.0)


@pytest.mark.parametrize("n", [0, 1, 2])
def test_mle_rhop_and_ml_level_match_nemo_orca1(n, monkeypatch):
    from legoesm.ocean.eos import make_eos_fn
    z = np.load(FIX)
    st, grid, zc, tm = _patch(z, n)
    wet3 = tm > 0
    wet2 = tm[..., 0] > 0
    rho, inml = _run_production_chain(
        st, grid, zc, make_eos_fn("nemo_teos10"), monkeypatch)
    worst = float(np.max(_rel(rho[wet3], z[f"p{n}_rhop"][wet3])))
    assert worst <= TOL, f"patch {n}: rho_pot vs NEMO rhop rel {worst:.3e}"
    bad = int(np.sum(inml[wet2] != z[f"p{n}_inml_mle"][wet2]))
    assert bad == 0, f"patch {n}: {bad}/{int(wet2.sum())} ML levels != NEMO"


def test_fixture_discriminates_the_eos():
    """Fixture sanity: with Wright the mismatch patches DO get a wrong ML
    level (otherwise the oracle test above could not fail)."""
    from legoesm.ocean.eos import make_eos_fn
    z = np.load(FIX)
    mp = pytest.MonkeyPatch()
    try:
        bad = 0
        for n in (0, 1):
            st, grid, zc, tm = _patch(z, n)
            _, inml = _run_production_chain(
                st, grid, zc, make_eos_fn("wright"), mp)
            w = tm[..., 0] > 0
            bad += int(np.sum(inml[w] != z[f"p{n}_inml_mle"][w]))
    finally:
        mp.undo()
    assert bad > 0


def test_wright_run_unchanged_by_eos_forwarding(monkeypatch):
    """The production card runs eos='wright', eos_depth='insitu': its
    physics eos_fn is make_eos_fn(eos='wright', eos_linear=None,
    eos_nemo_seos=None), so forwarding it is bit-identical to the old
    make_eos_fn() default."""
    from legoesm.ocean.eos import make_eos_fn
    z = np.load(FIX)
    st, grid, zc, _ = _patch(z, 2)
    card = make_eos_fn(eos="wright", eos_linear=None, eos_nemo_seos=None)
    a, ia = _run_production_chain(st, grid, zc, card, monkeypatch)
    b, ib = _run_production_chain(st, grid, zc, None, monkeypatch)
    ref = np.asarray(make_eos_fn()(st.T.data, st.S.data,
                                   jnp.zeros_like(st.T.data)))
    assert np.array_equal(a, b) and np.array_equal(ia, ib)
    assert np.array_equal(b, ref)
