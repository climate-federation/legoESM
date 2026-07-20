"""CLM-ML multilayer canopy: jax.jit-traceable FORWARD step (S1 de-host).

The default forward path marshals forcing to the host (``np.array(forcing.*)``)
and mutates CLM process-global orbital/topology state every step, so it cannot run
inside ``jax.jit`` (Q2 of the de-host spike: ``TracerArrayConversionError``).  S1
adds a *traceable* warm-step path: a warm-started step whose caller threads a
concrete ``grid_info`` runs fully on-device (jnp forcing, ``grid=``, device solar
zenith via ``cos_zenith_device=``), so a ``jax.jit`` scan-over-time keeps the whole
segment on device — the prerequisite for global-offline and coupled-AMIP CLM-ML.

These tests require a clm-ml-jax build whose ``MLCanopyFluxes`` accepts
``cos_zenith_device=`` (device solar zenith).  Until that revision is released they
skip, exactly like the other real-backend CLM-ML integration tests.
"""
from __future__ import annotations

import pytest

pytest.importorskip("multilayer_canopy")

import inspect  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from multilayer_canopy import MLCanopyFluxesMod as _mlmod  # noqa: E402

# Gate on the device-solar-zenith capability (the S1 backend change).
_HAS_DEVICE_ZENITH = (
    "cos_zenith_device" in inspect.signature(_mlmod.MLCanopyFluxes).parameters
)
pytestmark = pytest.mark.skipif(
    not _HAS_DEVICE_ZENITH,
    reason="clm-ml-jax build lacks MLCanopyFluxes(cos_zenith_device=) (S1 de-host)",
)

NCOL = 1


def _forcing(Tl):
    from legoesm.core.coupling_fields import AtmToSurface
    return AtmToSurface(
        sw_down=jnp.full(NCOL, 400.0), lw_down=jnp.full(NCOL, 350.0),
        precip_total=jnp.zeros(NCOL), precip_snow=jnp.zeros(NCOL),
        T_lowest=Tl, q_lowest=jnp.full(NCOL, 0.010),
        u_lowest=jnp.full(NCOL, 4.0), v_lowest=jnp.full(NCOL, 1.5),
        p_lowest=jnp.full(NCOL, 95000.0), p_surface=jnp.full(NCOL, 100000.0),
        rho_lowest=jnp.full(NCOL, 1.2), cos_zenith=jnp.full(NCOL, 0.7),
        co2_ppmv=jnp.full(NCOL, 400.0), has_radiation=jnp.ones(NCOL),
        has_precipitation=jnp.ones(NCOL))


def _fixtures():
    from legoesm.land.canopy.config import CLMMLCanopyConfig
    from legoesm.land.config import MultiLayerLandConfig
    Ts = jnp.full((NCOL, 8), 290.0)
    psi = jnp.full((NCOL, 8), -0.5)
    th = jnp.full((NCOL, 8), 0.25)
    lat = jnp.zeros(NCOL)
    cfg = CLMMLCanopyConfig()
    lc = MultiLayerLandConfig(surface_scheme=cfg)
    return Ts, psi, th, lat, cfg, lc


def _step(Tl, cstate, gi, *, Ts, psi, th, lat, cfg, lc):
    from legoesm.land.canopy.clm_ml_interface import compute_clm_ml_canopy_fluxes
    return compute_clm_ml_canopy_fluxes(
        T_soil_top=Ts[:, 0], forcing=_forcing(Tl), canopy_config=cfg,
        land_config=lc, land_params=None, canopy_state=cstate, dt=1800.0,
        T_soil=Ts, psi_soil=psi, theta_soil=th, lat=lat, doy=180.0, grid_info=gi)


def _warm_start():
    import legoesm.land.canopy.clm_ml_interface as ifc
    from legoesm.land.canopy.clm_ml_interface import extract_clm_ml_grid_info
    Ts, psi, th, lat, cfg, lc = _fixtures()
    kw = dict(Ts=Ts, psi=psi, th=th, lat=lat, cfg=cfg, lc=lc)
    # Reset the process-global topology cache so the cold start builds this
    # config's structure, not a prior test's leftover (CLM-ML uses process globals).
    ifc._last_topology_key = None
    out0, st0 = _step(jnp.full(NCOL, 295.0), None, None, **kw)
    gi = extract_clm_ml_grid_info(st0)
    return st0, gi, kw


def test_forward_step_runs_under_jit():
    """A warm-started forward step traces + executes under jax.jit (ncol=1)."""
    st0, gi, kw = _warm_start()
    jstep = jax.jit(lambda Tl: _step(Tl, st0, gi, **kw)[0])
    out = jstep(jnp.full(NCOL, 296.0))
    assert jnp.isfinite(out.shflx).all()
    assert jnp.isfinite(out.lhflx).all()


def test_device_path_matches_host_path():
    """DEVICE traceable path == the original HOST path, bit-close.

    The precision-gate self-check: the SAME warm step two ways on the SAME state —
    the original host path (grid_info=None -> _traceable False -> host solar +
    cached topology) versus the device path under jax.jit (grid_info=gi ->
    _traceable True -> cos_zenith_device + re-installed topology).  A real
    divergence would mean the de-host changed the physics, not just where it runs
    (it once did: the scan/grid= path dropped the ML sub-step flux averaging, so
    the canopy-air storage term reappeared in sensible heat — a ~40% shflx error).

    CLM-ML keeps topology in PROCESS-GLOBAL state that the eager path reads through
    a cache (``_last_topology_key``): a prior test with a different config leaves
    stale globals, so we reset the cache before the host reference to force it to
    re-install THIS config's topology (the device path always re-installs).  In a
    truly fresh process both paths are bit-identical.
    """
    import legoesm.land.canopy.clm_ml_interface as ifc
    st0, gi, kw = _warm_start()
    Tl = jnp.full(NCOL, 296.0)
    ifc._last_topology_key = None                              # force host re-install
    out_host, _ = _step(Tl, st0, None, **kw)                   # grid_info=None: host path
    out_dev = jax.jit(lambda t: _step(t, st0, gi, **kw)[0])(Tl)  # device jit path
    for name in ("shflx", "lhflx", "gpp", "sw_net", "lw_net", "stflx_air"):
        h = getattr(out_host, name)
        d = getattr(out_dev, name)
        assert jnp.allclose(h, d, atol=1e-6, rtol=1e-7), (
            f"{name}: host {h} vs device {d}")


def test_traceable_rejects_nonzero_met_type():
    """met_type != 0 on the traceable path is a hard error (curr_calday reliance)."""
    from legoesm.land.canopy.config import CLMMLCanopyConfig
    from legoesm.land.config import MultiLayerLandConfig
    from legoesm.land.canopy.clm_ml_interface import compute_clm_ml_canopy_fluxes
    st0, gi, kw = _warm_start()
    cfg3 = CLMMLCanopyConfig(met_type=3)
    lc3 = MultiLayerLandConfig(surface_scheme=cfg3)
    Ts, psi, th, lat = kw["Ts"], kw["psi"], kw["th"], kw["lat"]
    with pytest.raises(ValueError, match="met_type==0"):
        compute_clm_ml_canopy_fluxes(
            T_soil_top=Ts[:, 0], forcing=_forcing(jnp.full(NCOL, 296.0)),
            canopy_config=cfg3, land_config=lc3, land_params=None,
            canopy_state=st0, dt=1800.0, T_soil=Ts, psi_soil=psi, theta_soil=th,
            lat=lat, doy=180.0, grid_info=gi)


def test_traceable_rejects_supplied_lon():
    """lon != None on the traceable path is a hard error (solar-semantics divergence)."""
    from legoesm.land.canopy.clm_ml_interface import compute_clm_ml_canopy_fluxes
    st0, gi, kw = _warm_start()
    Ts, psi, th, lat, cfg, lc = (kw["Ts"], kw["psi"], kw["th"], kw["lat"],
                                 kw["cfg"], kw["lc"])
    with pytest.raises(ValueError, match="requires lon=None"):
        compute_clm_ml_canopy_fluxes(
            T_soil_top=Ts[:, 0], forcing=_forcing(jnp.full(NCOL, 296.0)),
            canopy_config=cfg, land_config=lc, land_params=None,
            canopy_state=st0, dt=1800.0, T_soil=Ts, psi_soil=psi, theta_soil=th,
            lat=lat, lon=jnp.zeros(NCOL), doy=180.0, grid_info=gi)


def test_grad_flows_through_jit_forward():
    """jax.grad of a jit-compiled forward step w.r.t. T_lowest is finite/non-zero."""
    st0, gi, kw = _warm_start()

    @jax.jit
    def loss(Tl):
        out, _ = _step(Tl, st0, gi, **kw)
        return jnp.sum(out.lhflx)

    g = jax.grad(loss)(jnp.full(NCOL, 296.0))
    assert jnp.isfinite(g).all()
    assert jnp.any(g != 0.0)


def test_trace_asserts_turbulence_scheme_applied():
    """A traceable step whose scheme was never eagerly applied fails LOUDLY.

    Guards the process-global psihat contract: the host float() verification cannot
    run inside a trace, so the traceable path asserts against the last eagerly
    applied scheme instead of silently running whatever tables happen to be set.
    """
    import legoesm.land.canopy.clm_ml_interface as ifc
    saved_applied = ifc._APPLIED_TURBULENCE_SCHEME
    saved_lock = ifc._DIFF_TURBULENCE_SCHEME
    try:
        ifc._APPLIED_TURBULENCE_SCHEME = None  # simulate "no eager apply yet"
        ifc._DIFF_TURBULENCE_SCHEME = None
        with pytest.raises(RuntimeError, match="applied \\+ verified by an eager"):
            ifc._assert_turbulence_scheme_for_trace("rsl_bonan")
    finally:
        ifc._APPLIED_TURBULENCE_SCHEME = saved_applied
        ifc._DIFF_TURBULENCE_SCHEME = saved_lock
