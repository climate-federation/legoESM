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
PROFILE_SCHEMES = (
    "tiedtke", "zhang_mcfarlane", "kain_fritsch", "emanuel", "bechtold",
)


@pytest.fixture(scope="module")
def setup():
    grid = create_cubed_sphere(4)
    sigma = make_hybrid_levels(NLEV)
    s2 = grid.grid_lat.shape
    s3 = (*s2, NLEV)
    rng = np.random.default_rng(0)
    # Moist, conditionally unstable tropical-ish column (index -1 = surface)
    Tprof = 300.0 - 60.0 * np.linspace(0, 1, NLEV)[::-1]
    T = jnp.asarray(np.broadcast_to(Tprof, s3).copy() + rng.normal(0, 0.3, s3))
    qprof = 18e-3 * np.exp(-3 * np.linspace(0, 1, NLEV)[::-1])
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


def _pipeline_step(grid, sigma, f, scheme, conv_prog=None):
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
        f["z3"], f["z2"], f["z2"], f["z2"], f["z2"], f["z2"],
    )


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
        "scheme", ["tiedtke", "zhang_mcfarlane", "kain_fritsch", "bechtold"],
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


def _tiedtke_step_pe(grid, sigma, f, precip_efficiency):
    """One tiedtke pipeline step with a given convective rain-split efficiency
    (microphysics='none', so surface precip == the in-updraft convective rain)."""
    config = ExperimentConfig(
        grid=GridConfig(grid_type="cubed_sphere", resolution=4, nlev=NLEV),
        dycore=DycoreConfig(model_type="hydrostatic", discretization="cdgrid"),
        convection="tiedtke", radiation="none", microphysics="none",
        convective_precip_efficiency=precip_efficiency,
    )
    config.validate_strict()
    pipe = build_physics_pipeline(grid, sigma, config)
    return pipe.physics_step_no_rad(
        f["T"], f["p_s"], f["q_v"], f["q_c"], f["q_r"], None,
        f["u"], f["v"], f["sst"], f["sic"], f["lat"], DT,
        f["z3"], f["z2"], f["z2"], f["z2"], f["z2"], f["z2"])


def test_convective_rain_reaches_surface_precip_832(setup):
    """#832: Tiedtke SPLITS detrained condensate via ``precip_efficiency`` into
    anvil cloud (``dq_c_conv_dt`` -> q_c) and in-updraft RAIN (``dq_r_conv_dt``).
    The pipeline previously consumed ONLY ``dq_c_conv_dt``, so the rain fraction
    vanished from the water budget (surface precip near zero) while its
    condensation latent heat stayed in ``dT_dt`` -> the upper-tropospheric warm
    drift #832 reports.  The fix (a) wires ``ExperimentConfig.
    convective_precip_efficiency`` into ``TiedtkeConfig.precip_efficiency`` (the
    field was dead) and (b) precipitates the in-updraft rain in the pipeline.

    ``microphysics='none'``, so the ONLY possible surface precip is that
    convective rain.  With the split ON it must be positive; with it OFF (the
    legacy path) it is identically zero (the rain is trapped in q_c) — pinning
    BOTH directions so the test is non-vacuous.
    """
    grid, sigma, f, _state = setup

    out = _tiedtke_step_pe(grid, sigma, f, 0.7)
    precip = np.asarray(out.precip)
    assert np.all(np.isfinite(precip)), "convective-rain precip has non-finite cells"
    total = float(np.sum(np.maximum(precip, 0.0)))
    assert total > 0.0, (
        "tiedtke in-updraft rain (dq_r_conv_dt) did not reach surface precip with "
        "precip_efficiency=0.7 + microphysics='none' — the #832 dropped tendency "
        "(or the dead convective_precip_efficiency wiring) is back"
    )
    # Physical rate, not a blow-up: a single convecting column << ~86 mm/day.
    assert float(np.max(precip)) < 1.0e-3, (
        f"convective-rain precip unphysically large: max={float(np.max(precip)):.3e}"
    )

    # Split OFF (precip_efficiency=0): no rain species is produced, so with
    # microphysics='none' the convective condensate is trapped in q_c and surface
    # precip is identically zero — the legacy behaviour, byte-preserved.
    out0 = _tiedtke_step_pe(grid, sigma, f, 0.0)
    assert float(np.sum(np.maximum(np.asarray(out0.precip), 0.0))) == 0.0, (
        "with precip_efficiency=0 the convective rain split must be OFF (no "
        "dq_r_conv_dt), so surface precip stays zero under microphysics='none'"
    )


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
