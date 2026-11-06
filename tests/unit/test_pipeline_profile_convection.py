"""Unified-pipeline wiring tests for the profile-prognostic convection
schemes (tiedtke / zhang_mcfarlane / kain_fritsch / emanuel / bechtold),
Kuo's moisture-convergence plumbing, and the p3 / ml_emulator
microphysics (all wired 2026-06-10; previously bridge-factory-only).

The load-bearing check is BRIDGE EQUIVALENCE: for an identical state the
pipeline's convection increment must match
``make_convection_physics("hydrostatic")`` to roundoff — the two paths
share the scheme traits and input plumbing, and this test keeps them
from drifting.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.core.field import Field
from legoesm.core.state import HydrostaticState
from legoesm.driver.config import DycoreConfig, ExperimentConfig, GridConfig
from legoesm.driver.physics_pipeline import build_physics_pipeline
from legoesm.atmosphere.physics.convection.config import ConvectionConfig
from legoesm.atmosphere.physics.convection.integration import (
    make_convection_physics,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.vertical import make_hybrid_levels

NLEV = 8
DT = 600.0
# tiedtke (WRF port) has its own contract (pure-sigma omega, dynamics and
# lagged turbulence inputs, cloud ice): test_tiedtke_pipeline_matches_bridge.
PROFILE_SCHEMES = (
    "zhang_mcfarlane", "kain_fritsch", "emanuel", "bechtold",
)


@pytest.fixture(scope="module")
def setup():
    return _make_setup(NLEV)


def _make_setup(nlev):
    grid = create_cubed_sphere(4)
    sigma = make_hybrid_levels(nlev)
    s2 = grid.grid_lat.shape
    s3 = (*s2, nlev)
    rng = np.random.default_rng(0)
    # Moist, conditionally unstable tropical-ish column (index -1 = surface)
    Tprof = 300.0 - 60.0 * np.linspace(0, 1, nlev)[::-1]
    T = jnp.asarray(np.broadcast_to(Tprof, s3).copy() + rng.normal(0, 0.3, s3))
    qprof = 18e-3 * np.exp(-3 * np.linspace(0, 1, nlev)[::-1])
    q_v = jnp.asarray(np.broadcast_to(qprof, s3) * (1 + 0.05 * rng.normal(0, 1, s3)))
    u = jnp.asarray(8.0 * np.sin(2 * np.asarray(grid.grid_lon))[..., None]
                    * np.ones(s3))
    v = jnp.asarray(8.0 * np.cos(2 * np.asarray(grid.grid_lat))[..., None]
                    * np.ones(s3))
    fields = dict(
        T=T, p_s=jnp.full(s2, 1.0e5), q_v=q_v,
        q_c=jnp.zeros(s3), q_r=jnp.zeros(s3),
        u=u, v=v,
        sst=jnp.full(s2, 302.0), sic=jnp.zeros(s2),
        lat=jnp.asarray(grid.grid_lat),
        z2=jnp.zeros(s2), z3=jnp.zeros(s3),
    )
    d3 = ("face", "x", "y", "level")
    d2 = ("face", "x", "y")
    state = HydrostaticState(
        u=Field(u, name="u", dims=d3, units="m/s"),
        v=Field(v, name="v", dims=d3, units="m/s"),
        T=Field(T, name="T", dims=d3, units="K"),
        p_s=Field(fields["p_s"], name="p_s", dims=d2, units="Pa"),
        phis=Field(jnp.zeros(s2), name="phis", dims=d2, units="m2/s2"),
        tracers={"q_v": Field(q_v, name="q_v", dims=d3, units="kg/kg")},
    )
    return grid, sigma, fields, state


def _pipeline_step(grid, sigma, f, scheme, conv_prog=None, **step_kw):
    config = ExperimentConfig(
        grid=GridConfig(grid_type="cubed_sphere", resolution=4, nlev=NLEV),
        dycore=DycoreConfig(model_type="hydrostatic", discretization="cdgrid"),
        convection=scheme, radiation="none", microphysics="none",
    )
    config.validate_strict()
    pipe = build_physics_pipeline(grid, sigma, config)
    return pipe.physics_step_no_rad(
        f["T"], f["p_s"], f["q_v"], f["q_c"], f["q_r"], conv_prog,
        f["u"], f["v"], f["sst"], f["sic"], f["lat"], DT,
        f["z3"], f["z2"], f["z2"], f["z2"], f["z2"], f["z2"], **step_kw,
    )


def test_zm_cloud_fraction_carry_reaches_rain_evaporation(setup):
    """CAM6 ZM evaporates convective rain at ``ke*(1-cldfrc)*(1-RH)*sqrt(flx)``
    with the previous step's total cloud fraction.  The pipeline threads the
    ``cloud_fraction`` carry into ``physics_step_no_rad`` (codex round 1);
    a half-cloudy carry must halve the evaporation relative to ``None``."""
    grid, sigma, f, _ = setup
    ncol = 6 * 4 * 4
    base = _pipeline_step(grid, sigma, f, "zhang_mcfarlane")
    cloudy = _pipeline_step(grid, sigma, f, "zhang_mcfarlane",
                            cloud_fraction=jnp.full((ncol, NLEV), 0.5))
    assert float(jnp.abs(base.precip).max()) > 0.0, "ZM must fire on this column"
    d = float(jnp.abs(cloudy.dq_v_dt - base.dq_v_dt).max())
    assert d > 0.0, "the cloud-fraction carry did not reach the rain evaporation"
    # less evaporation aloft -> more rain reaches the surface
    assert float((cloudy.precip - base.precip).sum()) > 0.0


class TestBridgeEquivalence:

    @pytest.mark.parametrize("scheme", PROFILE_SCHEMES + ("kuo",))
    def test_pipeline_matches_bridge(self, setup, scheme):
        grid, sigma, f, state = setup
        base = _pipeline_step(grid, sigma, f, "none")
        out = _pipeline_step(grid, sigma, f, scheme)
        pipe_dT = out.dT_dt - base.dT_dt
        pipe_dq = out.dq_v_dt - base.dq_v_dt

        bridge_fn = make_convection_physics(
            ConvectionConfig(scheme=scheme), model_type="hydrostatic", dt=DT,
        )
        ncol = int(np.prod(f["p_s"].shape))
        if scheme == "kuo":
            res = bridge_fn(state, grid, sigma)
        else:
            phys_state = SimpleNamespace(
                conv_prog_profile=jnp.zeros((ncol, NLEV)),
                conv_stoch_state=jnp.zeros((ncol,)),
                prng_key=jax.random.PRNGKey(0),
            )
            res = bridge_fn(state, grid, sigma, phys_state)
        tend = res[0] if isinstance(res, tuple) else res
        bridge_dT = tend.dT_dt.data
        bridge_dq = tend.tracer_tendencies["q_v"].data if hasattr(
            tend, "tracer_tendencies") and tend.tracer_tendencies else None

        scale = float(jnp.max(jnp.abs(bridge_dT))) + 1e-30
        assert float(jnp.max(jnp.abs(pipe_dT - bridge_dT))) <= 1e-12 + 1e-9 * scale
        if bridge_dq is not None:
            qscale = float(jnp.max(jnp.abs(bridge_dq))) + 1e-30
            assert (float(jnp.max(jnp.abs(pipe_dq - bridge_dq)))
                    <= 1e-12 + 1e-9 * qscale)


def test_kf_wrf_variant_pipeline_matches_bridge(setup):
    """The opt-in WRF KF-Eta variant (PR #2249) gets the real winds on both
    paths: pipeline (ExperimentConfig.kain_fritsch_variant) and bridge
    (KainFritschConfig.variant) agree to roundoff."""
    from legoesm.atmosphere.physics.convection.config import KainFritschConfig
    grid, sigma, f, state = setup

    def step(scheme):
        config = ExperimentConfig(
            grid=GridConfig(grid_type="cubed_sphere", resolution=4, nlev=NLEV),
            dycore=DycoreConfig(model_type="hydrostatic", discretization="cdgrid"),
            convection=scheme, radiation="none", microphysics="none",
            kain_fritsch_variant="wrf_kfeta",
        )
        config.validate_strict()
        pipe = build_physics_pipeline(grid, sigma, config)
        return pipe.physics_step_no_rad(
            f["T"], f["p_s"], f["q_v"], f["q_c"], f["q_r"], None,
            f["u"], f["v"], f["sst"], f["sic"], f["lat"], DT,
            f["z3"], f["z2"], f["z2"], f["z2"], f["z2"], f["z2"])

    pipe_dT = step("kain_fritsch").dT_dt - step("none").dT_dt
    bridge_fn = make_convection_physics(
        ConvectionConfig(scheme="kain_fritsch",
                         kain_fritsch=KainFritschConfig(variant="wrf_kfeta")),
        model_type="hydrostatic", dt=DT)
    ncol = int(np.prod(f["p_s"].shape))
    phys_state = SimpleNamespace(conv_prog_profile=jnp.zeros((ncol, NLEV)),
                                 conv_stoch_state=jnp.zeros((ncol,)),
                                 prng_key=jax.random.PRNGKey(0))
    res = bridge_fn(state, grid, sigma, phys_state)
    bridge_dT = (res[0] if isinstance(res, tuple) else res).dT_dt.data
    assert bool(jnp.all(jnp.isfinite(pipe_dT)))
    scale = float(jnp.max(jnp.abs(bridge_dT))) + 1e-30
    assert float(jnp.max(jnp.abs(pipe_dT - bridge_dT))) <= 1e-12 + 1e-9 * scale


class TestCarryAndFiring:

    @pytest.mark.parametrize("scheme", PROFILE_SCHEMES)
    def test_profile_carry_shape_stable(self, setup, scheme):
        grid, sigma, f, _ = setup
        ncol = int(np.prod(f["p_s"].shape))
        out1 = _pipeline_step(grid, sigma, f, scheme)
        assert out1.conv_prog.shape == (ncol, NLEV)
        out2 = _pipeline_step(grid, sigma, f, scheme, conv_prog=out1.conv_prog)
        assert out2.conv_prog.shape == (ncol, NLEV)
        assert bool(jnp.all(jnp.isfinite(out2.conv_prog)))

    @pytest.mark.parametrize(
        "scheme", ["zhang_mcfarlane", "kain_fritsch", "bechtold"],
    )
    def test_scheme_fires_on_unstable_column(self, setup, scheme):
        grid, sigma, f, _ = setup
        base = _pipeline_step(grid, sigma, f, "none")
        out = _pipeline_step(grid, sigma, f, scheme)
        assert float(jnp.max(jnp.abs(out.dT_dt - base.dT_dt))) > 1e-12

    def test_kuo_receives_moisture_convergence(self, setup):
        """Regression: the pipeline used to call Kuo without MC, leaving
        it permanently quiescent on resolved grids (fixed 2026-06-10)."""
        grid, sigma, f, _ = setup
        base = _pipeline_step(grid, sigma, f, "none")
        out = _pipeline_step(grid, sigma, f, "kuo")
        assert float(jnp.max(jnp.abs(out.dT_dt - base.dT_dt))) > 1e-12

    def test_bechtold_stochastic_rejected_at_build(self, setup):
        from legoesm.driver import physics_pipeline as pp

        config = ExperimentConfig(
            grid=GridConfig(grid_type="cubed_sphere", resolution=4, nlev=NLEV),
            dycore=DycoreConfig(
                model_type="hydrostatic", discretization="cdgrid"),
            convection="bechtold", radiation="none",
        )
        # Deterministic default resolves fine.
        fn, conv_cfg = pp._resolve_convection(config)
        assert callable(fn)
        assert conv_cfg.enable_stochastic is False
        # The build-time guard rejects the stochastic mode (PRNG-key
        # carry not threaded by the driver).
        with pytest.raises(NotImplementedError, match="PRNG"):
            pp._check_pipeline_convection_supported(
                "bechtold", conv_cfg._replace(enable_stochastic=True),
            )


class TestMicrophysicsWiring:

    @pytest.mark.parametrize("scheme", ["p3", "ml_emulator"])
    def test_micro_steps_finite(self, setup, scheme):
        grid, sigma, f, _ = setup
        config = ExperimentConfig(
            grid=GridConfig(grid_type="cubed_sphere", resolution=4, nlev=NLEV),
            dycore=DycoreConfig(
                model_type="hydrostatic", discretization="cdgrid"),
            convection="none", radiation="none", microphysics=scheme,
        )
        config.validate_strict()
        pipe = build_physics_pipeline(grid, sigma, config)
        z3 = f["z3"]
        out = pipe.physics_step_no_rad(
            f["T"], f["p_s"], f["q_v"], f["q_c"] + 1e-4, f["q_r"], None,
            f["u"], f["v"], f["sst"], f["sic"], f["lat"], DT,
            z3, f["z2"], f["z2"], f["z2"], f["z2"], f["z2"],
            q_i=z3 + 1e-5, q_s=z3, q_g=z3,
            N_c=z3 + 5e7, N_r=z3, N_i=z3 + 1e3,
        )
        for arr in (out.dT_dt, out.dq_v_dt, out.dq_c_dt, out.dq_i_dt,
                    out.dq_s_dt, out.dq_g_dt):
            assert bool(jnp.all(jnp.isfinite(arr)))
        # The scheme must actually act on this cloudy state.
        assert float(jnp.max(jnp.abs(out.dT_dt))) > 0.0


def test_tiedtke_pipeline_matches_bridge():
    """WRF Tiedtke through the driver pipeline and through the hydrostatic
    bridge, on the same columns and inputs (pure-sigma coordinate: the port's
    omega is diagnosed by sigma continuity): heating, drying and the cumulus
    friction the pipeline now applies to the resolved winds agree to
    round-off, and the scheme fires (heating and CMT non-zero)."""
    from legoesm.grids.vertical import create_sigma_coordinate
    # 30 levels: the 8-level module column never triggers the WRF scheme
    # (its lowest layer is ~1 km deep; probed, ldcum=0 for every forcing)
    NLEV = 30
    grid, _, f, state = _make_setup(NLEV)
    sigma = create_sigma_coordinate(NLEV, sigma_top=0.02)
    s3 = f["T"].shape
    ncol = int(np.prod(f["p_s"].shape))
    lev = jnp.arange(NLEV) >= NLEV - 4               # low-level convergence
    dqdyn = jnp.broadcast_to(jnp.where(lev, 3.0e-8, 0.0), s3)
    turb = jnp.zeros(s3)
    evap = jnp.full(f["p_s"].shape, 3.0e-5)
    z3 = f["z3"]

    def step(scheme):
        config = ExperimentConfig(
            grid=GridConfig(grid_type="cubed_sphere", resolution=4, nlev=NLEV),
            dycore=DycoreConfig(model_type="hydrostatic", discretization="cdgrid"),
            convection=scheme, radiation="none", microphysics="none",
            turbulence="louis",   # tiedtke reads its lagged qvpblten / qfx
        )
        config.validate_strict()
        pipe = build_physics_pipeline(grid, sigma, config)
        return pipe.physics_step_no_rad(
            f["T"], f["p_s"], f["q_v"], f["q_c"], f["q_r"], None,
            f["u"], f["v"], f["sst"], f["sic"], f["lat"], DT,
            z3, f["z2"], f["z2"], f["z2"], f["z2"], f["z2"], q_i=z3,
            dq_v_dt_dyn=dqdyn, turb_dq_v_prev=turb, evap_sfc_prev=evap)

    base, out = step("none"), step("tiedtke")
    bridge_fn = make_convection_physics(
        ConvectionConfig(scheme="tiedtke"), model_type="hydrostatic", dt=DT)
    st = state._replace(tracers={**state.tracers,
                                 "q_i": Field(z3, name="q_i", dims=state.T.dims)})
    ps = SimpleNamespace(conv_prog_profile=jnp.zeros((ncol, NLEV)),
                         dyn_tendency_qv=dqdyn, turb_tendency_qv=turb,
                         evap_sfc=evap)
    res = bridge_fn(st, grid, sigma, ps)
    tend = res[0] if isinstance(res, tuple) else res
    for pipe_x, bridge_x in ((out.dT_dt - base.dT_dt, tend.dT_dt.data),
                             (out.du_dt - base.du_dt, tend.du_dt.data),
                             (out.dv_dt - base.dv_dt, tend.dv_dt.data)):
        scale = float(jnp.max(jnp.abs(bridge_x)))
        assert scale > 0.0, "tiedtke did not fire / no cumulus friction"
        assert float(jnp.max(jnp.abs(pipe_x - bridge_x))) <= 1e-12 + 1e-9 * scale


def _bechtold_step_pe(grid, sigma, f, precip_efficiency):
    """One Bechtold pipeline step at a given convective rain-split efficiency
    (microphysics='none', so surface precip == the in-updraft convective rain).
    ``precip_efficiency=None`` leaves the ExperimentConfig knob unset -> Bechtold
    keeps its OWN 0.7 default (#929)."""
    config = ExperimentConfig(
        grid=GridConfig(grid_type="cubed_sphere", resolution=4, nlev=NLEV),
        dycore=DycoreConfig(model_type="hydrostatic", discretization="cdgrid"),
        convection="bechtold", radiation="none", microphysics="none",
        convective_precip_efficiency=precip_efficiency,
        # PE-split ISOLATION (the 2026-07-17 default flips turned on the
        # snow/melt march + downdraft + capdcycl + land-RHEBC, which emit
        # surface precip independent of the PE rain species and would
        # break the PE=0 -> zero-precip premise):
        bechtold_use_ifs_snow_melt=False,
        bechtold_use_ifs_downdraft=False,
        bechtold_use_ifs_capdcycl=False,
        bechtold_use_ifs_land_rhebc=False,
    )
    config.validate_strict()
    pipe = build_physics_pipeline(grid, sigma, config)
    return pipe.physics_step_no_rad(
        f["T"], f["p_s"], f["q_v"], f["q_c"], f["q_r"], None,
        f["u"], f["v"], f["sst"], f["sic"], f["lat"], DT,
        f["z3"], f["z2"], f["z2"], f["z2"], f["z2"], f["z2"])


def test_bechtold_precip_efficiency_threads_from_experiment_config_929(setup):
    """#929: the shared ExperimentConfig.convective_precip_efficiency knob must
    reach Bechtold's precip_efficiency in _resolve_convection (it was previously
    threaded to Tiedtke ONLY, so Bechtold ignored the CLI and PE 0/0.2/1 all
    resolved to 0.7).  ``None`` (the default sentinel) keeps the scheme default
    0.7 (ON, the #929 anvil-drain fix); an EXPLICIT value overrides it (0.0 =
    legacy detrain-all)."""
    from legoesm.driver import physics_pipeline as pp

    def _pe_of(knob):
        cfg = ExperimentConfig(
            grid=GridConfig(grid_type="cubed_sphere", resolution=4, nlev=NLEV),
            dycore=DycoreConfig(
                model_type="hydrostatic", discretization="cdgrid"),
            convection="bechtold", radiation="none",
            convective_precip_efficiency=knob,
        )
        _, conv_cfg = pp._resolve_convection(cfg)
        return conv_cfg.precip_efficiency

    assert _pe_of(None) == 0.7   # unset -> Bechtold scheme default (ON)
    assert _pe_of(0.0) == 0.0    # explicit 0 -> legacy detrain-all (dq_r None)
    assert _pe_of(0.5) == 0.5    # explicit override reaches the scheme
    assert _pe_of(1.0) == 1.0


def test_bechtold_convective_rain_reaches_surface_precip_929(setup):
    """#929 end-to-end: with the rain split ON (PE=0.7, the default) Bechtold's
    in-updraft rain (dq_r_conv_dt) reaches surface precip; with it EXPLICITLY
    OFF (PE=0.0) no separate rain species is produced (dq_r None) so — under
    microphysics='none' — convective surface precip is zero (the legacy path);
    and UNSET (None) behaves like the ON default, NOT like 0.0.  Pins all three
    so the test is non-vacuous and the sentinel is verified end-to-end."""
    grid, sigma, f, _state = setup

    out_on = _bechtold_step_pe(grid, sigma, f, 0.7)
    precip_on = np.asarray(out_on.precip)
    assert np.all(np.isfinite(precip_on)), "convective-rain precip non-finite"
    assert float(np.sum(np.maximum(precip_on, 0.0))) > 0.0, (
        "Bechtold in-updraft rain (dq_r_conv_dt) did not reach surface precip "
        "at precip_efficiency=0.7 -- the #929 rain-split routing regressed"
    )
    assert float(np.max(precip_on)) < 1.0e-3, (
        f"convective-rain precip unphysically large: {float(np.max(precip_on)):.3e}"
    )

    # PE=0 explicitly threaded -> legacy detrain-all, no rain species -> zero
    # convective surface precip under microphysics='none'.
    out_off = _bechtold_step_pe(grid, sigma, f, 0.0)
    assert float(np.sum(np.maximum(np.asarray(out_off.precip), 0.0))) == 0.0, (
        "with convective_precip_efficiency=0 Bechtold must emit no rain species "
        "(dq_r None), so surface precip stays zero under microphysics='none'"
    )

    # Unset (None) must keep the ON default (0.7), NOT resolve to 0.0.
    out_default = _bechtold_step_pe(grid, sigma, f, None)
    assert float(np.sum(np.maximum(np.asarray(out_default.precip), 0.0))) > 0.0, (
        "unset knob must keep Bechtold's 0.7 default ON (#929), not disable it"
    )


def test_pipeline_stateless_gwd_composite_runs_and_sums(setup):
    """#834 (codex adversarial finding 1): a stateless '+'-composite
    (``hines+mcfarlane``) through the coupler ``PhysicsPipeline`` must set
    ``_gwd_composite`` (NOT ``_gwd_prognostic``), unpack the combined executor's
    ``(GWDOutput, None)`` tuple in the composite branch, and sum both sources'
    momentum tendencies.  Regresses the ``TypeError: _combined_gwd() missing 1
    required positional argument: 'spectrum_in'`` that the plain single-return
    else-path raised before the fix.
    """
    grid, sigma, f, state = setup

    def _gwd_step(scheme):
        config = ExperimentConfig(
            grid=GridConfig(grid_type="cubed_sphere", resolution=4, nlev=NLEV),
            dycore=DycoreConfig(model_type="hydrostatic", discretization="cdgrid"),
            convection="none", radiation="none", microphysics="none",
            gravity_wave_drag=scheme,
        )
        config.validate_strict()
        pipe = build_physics_pipeline(grid, sigma, config)
        # all land: CAM6 gw_oro needs the land fraction (lf = 1 is its
        # unscaled drag)
        pipe.f_land = jnp.ones_like(f["sst"])
        out = pipe.physics_step_no_rad(
            f["T"], f["p_s"], f["q_v"], f["q_c"], f["q_r"], None,
            f["u"], f["v"], f["sst"], f["sic"], f["lat"], DT,
            f["z3"], f["z2"], f["z2"], f["z2"], f["z2"], f["z2"],
        )
        return pipe, out

    pipe_c, out_c = _gwd_step("hines+mcfarlane")
    # The flag wiring that routes to the composite (tuple-unpacking) branch.
    assert pipe_c._gwd_composite is True
    assert pipe_c._gwd_prognostic is False
    # Runs without the pre-fix TypeError and stays finite.
    assert bool(jnp.all(jnp.isfinite(out_c.du_dt)))
    assert bool(jnp.all(jnp.isfinite(out_c.dv_dt)))

    # Driver-path sum contract: composite GWD tendency == hines + mcfarlane,
    # each measured against the no-GWD baseline (exact arithmetic regardless of
    # how strongly either source is active on this column).
    _, out_h = _gwd_step("hines")
    _, out_m = _gwd_step("mcfarlane")
    _, out_n = _gwd_step("none")
    du_composite = out_c.du_dt - out_n.du_dt
    du_sum = (out_h.du_dt - out_n.du_dt) + (out_m.du_dt - out_n.du_dt)
    assert jnp.allclose(du_composite, du_sum, atol=1e-10)


def test_thompson_pipeline_numbers_match_bridge():
    """WRF Thompson through the driver pipeline and the hydrostatic bridge on
    the same columns: the pipeline converts its per-mass N_r carry to the
    scheme's per-volume number and back, as the bridge does (vote Y1b), so
    the rain-number tendency agrees and is non-zero; a carry without the
    full registry refuses (vote Y2a)."""
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig
    from legoesm.atmosphere.physics.microphysics.integration import (
        make_microphysics_physics,
    )
    NLEV = 30
    grid, _, f, state = _make_setup(NLEV)
    sigma = create_sigma_coordinate(NLEV, sigma_top=0.02)
    z3 = f["z3"]
    low = jnp.arange(NLEV) >= NLEV - 6
    q_r = jnp.broadcast_to(jnp.where(low, 2.0e-4, 0.0), z3.shape)
    n_r = jnp.broadcast_to(jnp.where(low, 5.0e4, 0.0), z3.shape)   # per kg
    config = ExperimentConfig(
        grid=GridConfig(grid_type="cubed_sphere", resolution=4, nlev=NLEV),
        dycore=DycoreConfig(model_type="hydrostatic", discretization="cdgrid"),
        convection="none", radiation="none", microphysics="thompson",
    )
    config.validate_strict()
    pipe = build_physics_pipeline(grid, sigma, config)

    def step(**kw):
        return pipe.physics_step_no_rad(
            f["T"], f["p_s"], f["q_v"], f["q_c"], q_r, None,
            f["u"], f["v"], f["sst"], f["sic"], f["lat"], DT,
            z3, f["z2"], f["z2"], f["z2"], f["z2"], f["z2"], **kw)

    full = dict(q_i=z3, q_s=z3, q_g=z3, N_c=z3, N_r=n_r, N_i=z3)
    out = step(**full)
    bridge = make_microphysics_physics(
        MicrophysicsConfig(scheme="thompson"), model_type="hydrostatic", dt=DT)
    d3 = state.T.dims
    tr = {k: Field(x, name=k, dims=d3) for k, x in (
        ("q_v", f["q_v"]), ("q_c", f["q_c"]), ("q_r", q_r), ("q_i", z3),
        ("q_s", z3), ("q_g", z3), ("N_c", z3), ("N_r", n_r), ("N_i", z3))}
    tend = bridge(state._replace(tracers=tr), grid, sigma)
    ref = tend.tracer_tendencies["N_r"].data
    scale = float(jnp.max(jnp.abs(ref)))
    assert scale > 0.0, "no rain-number tendency"
    assert float(jnp.max(jnp.abs(out.dN_r_dt - ref))) <= 1e-9 * scale
    with pytest.raises(ValueError, match="N_i"):
        step(**{**full, "N_i": None})
