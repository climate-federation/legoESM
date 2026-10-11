"""CLM-ML foliage clumping override (DifferBESS v10, 2026-10-11): ``clump_fac_jax`` threads a per-PFT clumping array from
``compute_clm_ml_canopy_fluxes`` through ``MLCanopyFluxes`` to ``SolarRadiation`` and ``CanopyNitrogenProfile`` in place of
the module table ``MLpftcon.clump_fac`` (1.0 for every PFT), so the DifferBESS runner can pass each site's driver clumping
index per time step. Contract: the table passed explicitly reproduces the default path exactly; a smaller clumping index
makes the canopy more transparent (more beam reaches the ground, less absorbed) and changes the fluxes; the override works
on the host path and under jax.jit on the device path alike.
"""
from __future__ import annotations

import pytest

pytest.importorskip("legoesm.land.canopy.clm_ml_backend.multilayer_canopy")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from tests.land.integration.test_clm_ml_jit_forward import NCOL, _forcing, _warm_start, pytestmark  # noqa: E402,F401


def _step_cf(Tl, cstate, gi, cf, *, Ts, psi, th, lat, cfg, lc):
    from legoesm.land.canopy.clm_ml_interface import compute_clm_ml_canopy_fluxes
    return compute_clm_ml_canopy_fluxes(
        T_soil_top=Ts[:, 0], forcing=_forcing(Tl), canopy_config=cfg,
        land_config=lc, land_params=None, canopy_state=cstate, dt=1800.0,
        T_soil=Ts, psi_soil=psi, theta_soil=th, lat=lat, doy=180.0, grid_info=gi, clump_fac_jax=cf)


def _mlcanopy(cstate):
    """The backend canopy container carried in the interface's CanopyState (absorbed SW per band, beam transmittance)."""
    for name in ("mlcanopy", "mlcanopy_inst", "ml"):
        m = getattr(cstate, name, None)
        if m is not None and hasattr(m, "swveg_canopy"):
            return m
    return None


def _table():
    from legoesm.land.canopy.clm_ml_backend.multilayer_canopy import MLpftconMod as pm
    return pm.MLpftcon.clump_fac


def test_explicit_table_reproduces_default_path():
    import legoesm.land.canopy.clm_ml_interface as ifc
    st0, gi, kw = _warm_start()
    Tl = jnp.full(NCOL, 296.0)
    ifc._last_topology_key = None
    out_ref, _ = _step_cf(Tl, st0, None, None, **kw)
    ifc._last_topology_key = None
    out_tab, _ = _step_cf(Tl, st0, None, _table(), **kw)
    for name in ("shflx", "lhflx", "gpp", "sw_net", "lw_net"):
        assert jnp.array_equal(getattr(out_ref, name), getattr(out_tab, name)), name


def test_smaller_clumping_changes_radiation_and_fluxes():
    import legoesm.land.canopy.clm_ml_interface as ifc
    st0, gi, kw = _warm_start()
    pft = int(kw["cfg"].pft_clm)
    Tl = jnp.full(NCOL, 296.0)
    ifc._last_topology_key = None
    out_1, st_1 = _step_cf(Tl, st0, None, _table(), **kw)
    ifc._last_topology_key = None
    out_c, st_c = _step_cf(Tl, st0, None, _table().at[pft].set(0.6), **kw)
    # a clumped canopy (Omega 0.6) intercepts less: the vegetation's absorbed shortwave falls and more beam reaches the
    # ground; the column total (out.sw_net = canopy + soil) and the fluxes move; GPP changes but its sign is not implied
    # (the coupled leaf response can rise with a brighter floor; review 2026-10-11)
    ml1, mlc = _mlcanopy(st_1), _mlcanopy(st_c)
    if ml1 is not None and mlc is not None:
        assert float(mlc.swveg_canopy[1, 1] + mlc.swveg_canopy[1, 2]) < float(ml1.swveg_canopy[1, 1] + ml1.swveg_canopy[1, 2])
        assert float(mlc.tbi_profile[1, 0]) > float(ml1.tbi_profile[1, 0])
    assert float(out_c.sw_net[0]) != float(out_1.sw_net[0])
    assert jnp.isfinite(out_c.gpp).all() and float(out_c.gpp[0]) != float(out_1.gpp[0])
    assert float(out_c.shflx[0]) != float(out_1.shflx[0]) or float(out_c.lhflx[0]) != float(out_1.lhflx[0])


def test_override_under_jit_matches_host():
    import legoesm.land.canopy.clm_ml_interface as ifc
    st0, gi, kw = _warm_start()
    pft = int(kw["cfg"].pft_clm)
    cf = _table().at[pft].set(0.7)
    Tl = jnp.full(NCOL, 296.0)
    ifc._last_topology_key = None
    out_host, _ = _step_cf(Tl, st0, None, cf, **kw)
    out_dev = jax.jit(lambda t, c: _step_cf(t, st0, gi, c, **kw)[0])(Tl, cf)
    for name in ("shflx", "lhflx", "gpp", "sw_net", "lw_net"):
        assert jnp.allclose(getattr(out_host, name), getattr(out_dev, name), atol=1e-6, rtol=1e-7), name
