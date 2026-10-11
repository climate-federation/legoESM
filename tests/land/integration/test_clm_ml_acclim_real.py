"""Knauer acclimation (acclim_type 2) on the real CLM-ML backend: a warm step under Medlyn with the growth temperature
prescribed runs, is finite, differs from the Kattge & Knorr step, and refuses to run without the growth temperature."""
from __future__ import annotations

import pytest

pytest.importorskip("legoesm.land.canopy.clm_ml_backend.multilayer_canopy")

import jax.numpy as jnp  # noqa: E402

from tests.land.integration.test_clm_ml_jit_forward import NCOL, _forcing, pytestmark  # noqa: E402,F401


def _fixtures(acclimation, t_home=None):
    from legoesm.land.canopy.config import CLMMLCanopyConfig
    from legoesm.land.config import MultiLayerLandConfig
    Ts = jnp.full((NCOL, 8), 290.0); psi = jnp.full((NCOL, 8), -0.5); th = jnp.full((NCOL, 8), 0.25); lat = jnp.zeros(NCOL)
    cfg = CLMMLCanopyConfig(stomatal_model="medlyn", acclimation=acclimation, t_home_C=t_home).validate()
    return dict(Ts=Ts, psi=psi, th=th, lat=lat, cfg=cfg, lc=MultiLayerLandConfig(surface_scheme=cfg))


def _step(Tl, cstate, tacclim, *, Ts, psi, th, lat, cfg, lc):
    from legoesm.land.canopy.clm_ml_interface import compute_clm_ml_canopy_fluxes
    return compute_clm_ml_canopy_fluxes(T_soil_top=Ts[:, 0], forcing=_forcing(Tl), canopy_config=cfg, land_config=lc, land_params=None,
                                        canopy_state=cstate, dt=1800.0, T_soil=Ts, psi_soil=psi, theta_soil=th, lat=lat, doy=180.0, tacclim_K=tacclim)


def test_knauer_step_runs_and_differs_from_kk():
    import legoesm.land.canopy.clm_ml_interface as ifc
    Tl = jnp.full(NCOL, 296.0)
    kw = _fixtures("kk"); ifc._last_topology_key = None
    out_kk, st = _step(Tl, None, None, **kw)
    out_kk2, _ = _step(Tl, st, None, **kw)
    kw2 = _fixtures("knauer", t_home=20.0); ifc._last_topology_key = None
    out_kn, st2 = _step(Tl, None, jnp.full(NCOL, 288.15), **kw2)
    out_kn2, _ = _step(Tl, st2, jnp.full(NCOL, 288.15), **kw2)
    for o in (out_kn, out_kn2):
        assert jnp.isfinite(o.gpp).all() and jnp.isfinite(o.shflx).all() and jnp.isfinite(o.lhflx).all()
    assert float(out_kn2.gpp[0]) != float(out_kk2.gpp[0])
    # the prescribed growth temperature is what the state carries (no running mean under the override; review P1)
    assert jnp.allclose(st2.t_a10_arr, 288.15)
    # a warmer growth temperature under Knauer changes the step again (the activation energy and entropy move)
    out_kn3, _ = _step(Tl, st2, jnp.full(NCOL, 303.15), **kw2)
    assert float(out_kn3.gpp[0]) != float(out_kn2.gpp[0])
    # the home temperature is runtime data: a different T_home in the same process changes the step (review P2: it was a
    # trace-time constant of the jitted nitrogen profile)
    kw3 = _fixtures("knauer", t_home=30.0); ifc._last_topology_key = None
    out_th, st3 = _step(Tl, None, jnp.full(NCOL, 288.15), **kw3)
    out_th2, _ = _step(Tl, st3, jnp.full(NCOL, 288.15), **kw3)
    assert float(out_th2.gpp[0]) != float(out_kn2.gpp[0])


def test_knauer_without_growth_temperature_is_refused():
    import legoesm.land.canopy.clm_ml_interface as ifc
    kw = _fixtures("knauer", t_home=20.0); ifc._last_topology_key = None
    with pytest.raises(ValueError, match="tacclim_K"):
        _step(jnp.full(NCOL, 296.0), None, None, **kw)
