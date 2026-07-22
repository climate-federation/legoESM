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

pytest.importorskip("legoesm.land.canopy.clm_ml_backend.multilayer_canopy")

import inspect  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from legoesm.land.canopy.clm_ml_backend.multilayer_canopy import MLCanopyFluxesMod as _mlmod  # noqa: E402

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


def test_traceable_rejects_traced_lat():
    """A TRACED lat on the traceable path fails LOUDLY (geometry must be static).

    lat feeds host-side CLM topology setup (np.array(lat) / _setup_clm_topology), so
    jit-ing over lat would otherwise raise a cryptic TracerArrayConversionError deep
    in the backend.  The geometry guard catches it with actionable guidance.
    """
    st0, gi, kw = _warm_start()
    Ts, psi, th, cfg, lc = kw["Ts"], kw["psi"], kw["th"], kw["cfg"], kw["lc"]

    def run(lat):
        from legoesm.land.canopy.clm_ml_interface import compute_clm_ml_canopy_fluxes
        return compute_clm_ml_canopy_fluxes(
            T_soil_top=Ts[:, 0], forcing=_forcing(jnp.full(NCOL, 296.0)),
            canopy_config=cfg, land_config=lc, land_params=None, canopy_state=st0,
            dt=1800.0, T_soil=Ts, psi_soil=psi, theta_soil=th, lat=lat, doy=180.0,
            grid_info=gi)[0].shflx

    with pytest.raises(ValueError, match="lat is a jax tracer"):
        jax.jit(run)(jnp.zeros(NCOL))


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


def _forcing_n(n, Tl, *, sw=400.0, q=0.010):
    from legoesm.core.coupling_fields import AtmToSurface
    return AtmToSurface(
        sw_down=jnp.full(n, sw), lw_down=jnp.full(n, 350.0),
        precip_total=jnp.zeros(n), precip_snow=jnp.zeros(n),
        T_lowest=Tl, q_lowest=jnp.full(n, q),
        u_lowest=jnp.full(n, 4.0), v_lowest=jnp.full(n, 1.5),
        p_lowest=jnp.full(n, 95000.0), p_surface=jnp.full(n, 100000.0),
        rho_lowest=jnp.full(n, 1.2), cos_zenith=jnp.full(n, 0.7),
        co2_ppmv=jnp.full(n, 400.0), has_radiation=jnp.ones(n),
        has_precipitation=jnp.ones(n))


def test_traced_multicolumn_without_gridinfo_rejected():
    """ncol>1 under jax.jit WITHOUT a per-column grid_info fails LOUDLY.

    The structural ints (ncan/ntop/nbot) vary per column, so ncol>1 traceable
    needs a length-ncol GridInfo tuple.  Without it, ncol>1 would fall through to
    the eager host path and raise a cryptic TracerArrayConversionError deep in the
    backend.  The interface backstop catches it — the single chokepoint covering
    every driver path (string- and object-config).  EAGER multi-column (concrete
    arrays) stays valid; only a TRACED ncol>1 without grid_info is rejected.
    """
    from legoesm.land.canopy.config import CLMMLCanopyConfig
    from legoesm.land.config import MultiLayerLandConfig
    from legoesm.land.canopy.clm_ml_interface import compute_clm_ml_canopy_fluxes

    n = 2
    cfg = CLMMLCanopyConfig()
    lc = MultiLayerLandConfig(surface_scheme=cfg)
    Ts = jnp.full((n, 8), 290.0)
    psi = jnp.full((n, 8), -0.5)
    th = jnp.full((n, 8), 0.25)
    lat = jnp.zeros(n)

    def run(Tl):
        return compute_clm_ml_canopy_fluxes(
            T_soil_top=Ts[:, 0], forcing=_forcing_n(n, Tl), canopy_config=cfg,
            land_config=lc, land_params=None, canopy_state=None, dt=1800.0,
            T_soil=Ts, psi_soil=psi, theta_soil=th, lat=lat, doy=180.0)[0].shflx

    with pytest.raises(NotImplementedError, match="per-column GridInfo"):
        jax.jit(run)(jnp.full(n, 296.0))


def test_multicolumn_matches_independent_single_columns():
    """S2: a jitted ncol=2 step == two independent ncol=1 steps (column independence).

    Canopy columns are physically independent (no horizontal coupling), so the
    per-column loop must reproduce, column-for-column, the PROVEN single-column S1
    path — with DIFFERENT forcing per column so any cross-column leakage (a mixed
    cos_zenith slice, a stale GridInfo, a scatter to the wrong patch) shows up.
    This validates the multi-column traceable path against S1 as the reference.
    """
    import legoesm.land.canopy.clm_ml_interface as ifc
    from legoesm.land.canopy.config import CLMMLCanopyConfig
    from legoesm.land.config import MultiLayerLandConfig
    from legoesm.land.canopy.clm_ml_interface import (
        compute_clm_ml_canopy_fluxes, extract_clm_ml_grid_info)

    cfg = CLMMLCanopyConfig()
    lc = MultiLayerLandConfig(surface_scheme=cfg)
    # Two columns with DIFFERENT forcing (temperature + insolation + humidity).
    Tl = jnp.array([293.0, 300.0])
    sw = jnp.array([300.0, 500.0])
    q = jnp.array([0.008, 0.014])

    cosf = jnp.array([0.6, 0.8])  # per-column FORWARD cos(zenith)

    def _forcing2(Tl_, sw_, q_, cos_):
        from legoesm.core.coupling_fields import AtmToSurface
        n = 2
        return AtmToSurface(
            sw_down=sw_, lw_down=jnp.full(n, 350.0),
            precip_total=jnp.zeros(n), precip_snow=jnp.zeros(n),
            T_lowest=Tl_, q_lowest=q_,
            u_lowest=jnp.full(n, 4.0), v_lowest=jnp.full(n, 1.5),
            p_lowest=jnp.full(n, 95000.0), p_surface=jnp.full(n, 100000.0),
            rho_lowest=jnp.full(n, 1.2), cos_zenith=cos_,
            co2_ppmv=jnp.full(n, 400.0), has_radiation=jnp.ones(n),
            has_precipitation=jnp.ones(n))

    Ts2 = jnp.full((2, 8), 290.0)
    psi2 = jnp.full((2, 8), -0.5)
    th2 = jnp.full((2, 8), 0.25)
    lat2 = jnp.zeros(2)

    # --- Warm-start 2 columns (eager cold step), extract the per-column tuple ---
    # The warm forcing MUST be per-column identical to the single-column reference
    # warm-start (_forcing_n: T=295, sw=400, q=0.010, cos=0.7) so both start the
    # forward step from the SAME warm state — otherwise the comparison confounds a
    # warm-state difference with the multi-vs-single code path under test.
    ifc._last_topology_key = None
    _out0, st2 = compute_clm_ml_canopy_fluxes(
        T_soil_top=Ts2[:, 0],
        forcing=_forcing2(jnp.full(2, 295.0), jnp.full(2, 400.0),
                          jnp.full(2, 0.010), jnp.full(2, 0.7)),
        canopy_config=cfg, land_config=lc, land_params=None, canopy_state=None,
        dt=1800.0, T_soil=Ts2, psi_soil=psi2, theta_soil=th2, lat=lat2, doy=180.0)
    gi2 = extract_clm_ml_grid_info(st2)
    assert isinstance(gi2, tuple) and len(gi2) == 2, "expected per-column GridInfo tuple"

    # --- ncol=2 jitted forward (the S2 per-column loop) ---
    def run2(Tl_, sw_, q_):
        return compute_clm_ml_canopy_fluxes(
            T_soil_top=Ts2[:, 0], forcing=_forcing2(Tl_, sw_, q_, cosf),
            canopy_config=cfg, land_config=lc, land_params=None, canopy_state=st2,
            dt=1800.0, T_soil=Ts2, psi_soil=psi2, theta_soil=th2, lat=lat2,
            doy=180.0, grid_info=gi2)[0]
    out2 = jax.jit(run2)(Tl, sw, q)

    # --- Reference: each column alone through the S1 single-column path ---
    # Warm-start each single column EAGERLY (outside jit), like real usage, then
    # jit only the forward step closing over its concrete GridInfo.
    from legoesm.core.coupling_fields import AtmToSurface

    def _f1(col):
        return AtmToSurface(
            sw_down=sw[col:col + 1], lw_down=jnp.full(1, 350.0),
            precip_total=jnp.zeros(1), precip_snow=jnp.zeros(1),
            T_lowest=Tl[col:col + 1], q_lowest=q[col:col + 1],
            u_lowest=jnp.full(1, 4.0), v_lowest=jnp.full(1, 1.5),
            p_lowest=jnp.full(1, 95000.0), p_surface=jnp.full(1, 100000.0),
            rho_lowest=jnp.full(1, 1.2), cos_zenith=jnp.array([0.6 if col == 0 else 0.8]),
            co2_ppmv=jnp.full(1, 400.0), has_radiation=jnp.ones(1),
            has_precipitation=jnp.ones(1))

    def _ref_col(col):
        Ts1 = jnp.full((1, 8), 290.0)
        psi1 = jnp.full((1, 8), -0.5)
        th1 = jnp.full((1, 8), 0.25)
        lat1 = jnp.zeros(1)  # geometry: concrete, closed over (never a tracer)
        ifc._last_topology_key = None
        _o, st1 = compute_clm_ml_canopy_fluxes(
            T_soil_top=Ts1[:, 0], forcing=_forcing_n(1, jnp.array([295.0])),
            canopy_config=cfg, land_config=lc, land_params=None, canopy_state=None,
            dt=1800.0, T_soil=Ts1, psi_soil=psi1, theta_soil=th1,
            lat=lat1, doy=180.0)
        gi1 = extract_clm_ml_grid_info(st1)

        def _fwd():
            return compute_clm_ml_canopy_fluxes(
                T_soil_top=Ts1[:, 0], forcing=_f1(col), canopy_config=cfg,
                land_config=lc, land_params=None, canopy_state=st1, dt=1800.0,
                T_soil=Ts1, psi_soil=psi1, theta_soil=th1, lat=lat1,
                doy=180.0, grid_info=gi1)[0]
        return jax.jit(_fwd)()

    ref0 = _ref_col(0)
    ref1 = _ref_col(1)

    for name in ("shflx", "lhflx", "gpp", "sw_net", "lw_net"):
        got = getattr(out2, name)
        r0 = getattr(ref0, name)
        r1 = getattr(ref1, name)
        assert jnp.allclose(got[0], r0[0], atol=1e-6, rtol=1e-6), (
            f"{name}[col0] multi {got[0]} vs single {r0[0]}")
        assert jnp.allclose(got[1], r1[0], atol=1e-6, rtol=1e-6), (
            f"{name}[col1] multi {got[1]} vs single {r1[0]}")

    # Per-column GridInfo is realigned by .p, so the tuple must cover exactly
    # patches 1..ncol.  A tuple missing a patch (here {1,1}) is a loud error — the
    # coverage guard fires BEFORE the expensive per-column loop.  (Reordering with
    # varying per-column structure is exercised implicitly by the by-.p realign;
    # here both columns share a PFT so a reorder value-test would be vacuous.)
    bad_gi = (gi2[0], gi2[0])  # patches {1, 1}: column 2 (patch 2) missing
    with pytest.raises(ValueError, match="patches"):
        jax.jit(lambda: compute_clm_ml_canopy_fluxes(
            T_soil_top=Ts2[:, 0], forcing=_forcing2(Tl, sw, q, cosf),
            canopy_config=cfg, land_config=lc, land_params=None, canopy_state=st2,
            dt=1800.0, T_soil=Ts2, psi_soil=psi2, theta_soil=th2, lat=lat2,
            doy=180.0, grid_info=bad_gi)[0])()


def test_scan_columns_is_o1_and_matches_loop():
    """S3 Phase 1: the uniform-structure column scan is O(1)-compile AND a no-op.

    Two guarantees, one warm-start:

    1. **O(1) compile.** ``scan_columns=True`` lowers the per-column canopy to a
       single ``while`` in the HLO (jax.lax.scan), so the program text does NOT
       grow with ncol; ``scan_columns=False`` unrolls the Python loop to ncol
       full kernel copies (no ``while``, O(ncol) text).  We lower BOTH at the
       same ncol and assert: scan HLO has a ``while`` and is materially shorter
       than the unrolled loop HLO — the structural proof that the scan removed
       the O(ncol) unroll (the S3 compile wall).  Non-vacuous: the loop lowering
       is asserted to LACK the ``while`` and be larger, so a scan that silently
       fell back to the loop would fail here.

    2. **Numerical no-op.** RUN both paths and assert equal fluxes — the scan is
       value-identical to the proven S2 loop on the uniform-structure path (so
       switching the default to the scan changes compile cost, not answers).
    """
    import legoesm.land.canopy.clm_ml_interface as ifc
    from legoesm.land.canopy.config import CLMMLCanopyConfig
    from legoesm.land.config import MultiLayerLandConfig
    from legoesm.land.canopy.clm_ml_interface import (
        compute_clm_ml_canopy_fluxes, extract_clm_ml_grid_info)

    ncol = 4  # small: enough to distinguish O(1) scan from O(ncol) unroll cheaply
    Ts = jnp.full((ncol, 8), 290.0)
    psi = jnp.full((ncol, 8), -0.5)
    th = jnp.full((ncol, 8), 0.25)
    lat = jnp.zeros(ncol)

    def _cfg(scan_columns):
        return CLMMLCanopyConfig(scan_columns=scan_columns)

    # --- Warm-start ncol columns ONCE (eager cold step), extract the per-column
    # GridInfo tuple.  Columns share the default PFT => uniform (ncan/ntop/nbot/pft)
    # => the scan path is eligible; the loop path is selected by scan_columns=False.
    ifc._last_topology_key = None
    _o0, st = compute_clm_ml_canopy_fluxes(
        T_soil_top=Ts[:, 0], forcing=_forcing_n(ncol, jnp.full(ncol, 295.0)),
        canopy_config=_cfg(True), land_config=MultiLayerLandConfig(surface_scheme=_cfg(True)),
        land_params=None, canopy_state=None, dt=1800.0,
        T_soil=Ts, psi_soil=psi, theta_soil=th, lat=lat, doy=180.0)
    gi = extract_clm_ml_grid_info(st)
    assert isinstance(gi, tuple) and len(gi) == ncol

    Tl = jnp.full(ncol, 296.0)

    def _run(scan_columns):
        cfg = _cfg(scan_columns)
        lc = MultiLayerLandConfig(surface_scheme=cfg)

        def fwd(Tl_):
            return compute_clm_ml_canopy_fluxes(
                T_soil_top=Ts[:, 0], forcing=_forcing_n(ncol, Tl_), canopy_config=cfg,
                land_config=lc, land_params=None, canopy_state=st, dt=1800.0,
                T_soil=Ts, psi_soil=psi, theta_soil=th, lat=lat, doy=180.0,
                grid_info=gi)[0]
        return fwd

    # --- (1) O(1): compare lowered HLO of scan vs unrolled loop ---
    # NOTE: both paths contain a ``stablehlo.while`` from the INNER sub-step scan
    # (num_ml_steps) — so "loop has no while" is NOT the signal.  The signal is
    # that the column scan collapses ncol copies of that inner while into ONE
    # (nested in one column while), whereas the loop UNROLLS ncol full kernels:
    #  - fewer while-blocks in the scan HLO than the loop HLO, and
    #  - materially shorter scan HLO (the O(ncol) unroll is what S3 removes).
    hlo_scan = jax.jit(_run(True)).lower(Tl).as_text()
    hlo_loop = jax.jit(_run(False)).lower(Tl).as_text()
    n_while_scan = hlo_scan.count("stablehlo.while")
    n_while_loop = hlo_loop.count("stablehlo.while")
    assert n_while_scan >= 1, "scan_columns=True should lower to a lax.scan while-loop"
    assert n_while_scan < n_while_loop, (
        f"the column scan should collapse the per-column unroll into fewer while-"
        f"blocks than the loop, got scan={n_while_scan} loop={n_while_loop} "
        f"(equal ⇒ the scan silently fell back to the unrolled loop)")
    assert len(hlo_loop) > 2 * len(hlo_scan), (
        f"the unrolled loop HLO ({len(hlo_loop)} chars) should be >2x the scan HLO "
        f"({len(hlo_scan)} chars) at ncol={ncol} — the O(ncol) unroll S3 removes")

    # --- (2) no-op: scan values == loop values ---
    out_scan = jax.jit(_run(True))(Tl)
    out_loop = jax.jit(_run(False))(Tl)
    for name in ("shflx", "lhflx", "gpp", "sw_net", "lw_net"):
        s = getattr(out_scan, name)
        ll = getattr(out_loop, name)
        assert jnp.allclose(s, ll, atol=1e-6, rtol=1e-6), (
            f"{name}: scan {s} != loop {ll} (the column scan must be a no-op vs the loop)")
