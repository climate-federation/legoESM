"""CAM6 clubb_intr moist host mapping: CLUBB owns cloud liquid.

CAM6 feeds CLUBB total water ``rt = q + ql`` and ``thl = (T - L_v/c_p ql)/exner``
and hands its PDF liquid ``rcm`` back as cloud liquid (clubb_intr.F90).  With
``liquid_handoff`` the port does the same, and Morrison's own liquid
condensation is switched off so the two do not both act.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants  # noqa: E402
from legoesm.atmosphere.physics._shared import (  # noqa: E402
    exner_function,
    virtual_temperature,
)
from legoesm.atmosphere.physics.turbulence.clubb import (  # noqa: E402
    CLUBBConfig,
    clubb_turbulence_prognostic,
    integrate_clubb_column,
    pack_clubb_moments,
)
from tests.unit.test_clubb_scheme import _scm_column  # noqa: E402


@pytest.fixture(scope="module")
def spun_up():
    """A moist column with spun-up moments, carrying CLUBB liquid (dry mapping)."""
    kw = _scm_column(ncol=2, nlev=24, dtheta_dz=4e-3)
    cfg = CLUBBConfig(prognostic=True, clubb_dt=300.0)
    u, v, T, q, m, _ = integrate_clubb_column(**kw, dt=150.0, nsteps=40, config=cfg)
    tv = jnp.maximum(virtual_temperature(T, q), cfg.T0 * 0.5)
    rho = kw["p_full"] / (constants.R_d * tv)
    dz = np.abs(np.asarray(kw["z_half"])[:, :-1] - np.asarray(kw["z_half"])[:, 1:])
    return dict(kw=kw, u=u, v=v, T=T, q=q, m=pack_clubb_moments(m), rho=rho,
                mass=np.asarray(rho) * dz)


def _step(s, q_c, dt=150.0, handoff=True):
    cfg = CLUBBConfig(prognostic=True, clubb_dt=300.0, liquid_handoff=handoff)
    kw = s["kw"]
    # Zero surface fluxes: T_sfc / q_sfc track the near-surface state.
    return clubb_turbulence_prognostic(
        s["u"], s["v"], s["T"], s["q"], s["m"], kw["p_full"], kw["p_half"],
        kw["z_full"], kw["z_half"], s["T"][:, -1], s["q"][:, -1], s["rho"], dt,
        cfg, q_c=q_c if handoff else None)


@pytest.mark.parametrize("dt, tol, seed", [
    (150.0, 1e-12, 0.0),   # n_sub = 1, zero sfc flux, no host liquid
    (150.0, 1e-12, 0.2),   # host liquid present: moved, not stacked on top
    (900.0, 1e-8, 0.0),    # sub-cycle: sfc flux budgeted
])
def test_handoff_conserves_total_water_and_thl_and_hands_liquid_back(
        spun_up, dt, tol, seed):
    s = spun_up
    # Same total water in every case; ``seed`` of it is declared cloud liquid.
    q_c = seed * jnp.asarray(s["q"])
    s = dict(s, q=s["q"] - q_c)
    out, _ = _step(s, q_c, dt=dt)
    if dt == 150.0:
        assert np.all(np.asarray(out.shflx) == 0.0)
        assert np.all(np.asarray(out.lhflx) == 0.0)
    dqc = np.asarray(out.dq_c_dt)
    dqv = np.asarray(out.dq_v_dt)
    # Liquid actually moves (the spun-up column carries rcm ~ 1e-3).
    assert float(np.max(np.abs(dqc) * dt)) > 1e-5
    assert np.all(np.isfinite(dqc))
    # Neither phase goes negative after the step.
    assert np.all(np.asarray(q_c) + dt * dqc >= -1e-15)
    assert np.all(np.asarray(s["q"]) + dt * dqv >= -1e-15)
    mass = s["mass"]
    # Column total water changes only by the surface moisture flux.
    col_rt = np.sum(mass * (np.asarray(s["q"]) + np.asarray(q_c)), axis=1)
    resid_rt = (np.sum(mass * (dqv + dqc), axis=1)
                - np.asarray(out.lhflx) / constants.L_v)
    assert np.all(np.abs(resid_rt) * dt / col_rt < tol)
    # Column liquid potential temperature changes only by the surface heat flux.
    exner = np.asarray(exner_function(s["kw"]["p_full"]))
    dthl = (np.asarray(out.dT_dt) - constants.L_v / constants.c_pd * dqc) / exner
    col_thl = np.sum(mass * np.asarray(s["T"]) / exner, axis=1)
    resid_thl = (np.sum(mass * dthl, axis=1)
                 - np.asarray(out.shflx) / (constants.c_pd * exner[:, -1]))
    assert np.all(np.abs(resid_thl) * dt / col_thl < tol)


def test_flag_and_q_c_must_agree(spun_up):
    s = spun_up
    with pytest.raises(ValueError, match="liquid_handoff"):
        _step(s, None, handoff=True)
    cfg = CLUBBConfig(prognostic=True, clubb_dt=300.0)
    kw = s["kw"]
    with pytest.raises(ValueError, match="liquid_handoff"):
        clubb_turbulence_prognostic(
            s["u"], s["v"], s["T"], s["q"], s["m"], kw["p_full"], kw["p_half"],
            kw["z_full"], kw["z_half"], s["T"][:, -1], s["q"][:, -1], s["rho"],
            150.0, cfg, q_c=jnp.zeros_like(s["q"]))
    off, _ = _step(s, None, handoff=False)
    assert off.dq_c_dt is None


def test_handoff_is_jit_and_grad_clean(spun_up):
    s = spun_up

    def loss(T):
        out, _ = _step(dict(s, T=T), jnp.zeros_like(s["q"]))
        return jnp.sum(out.dq_c_dt ** 2) + jnp.sum(out.dT_dt ** 2)

    val_e = loss(s["T"])
    val_j = jax.jit(loss)(s["T"])
    assert np.isclose(float(val_e), float(val_j), rtol=1e-10)
    g = jax.grad(loss)(s["T"])
    assert np.all(np.isfinite(np.asarray(g))) and float(jnp.max(jnp.abs(g))) > 0.0


def test_morrison_liquid_condensation_switch():
    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
    from legoesm.atmosphere.physics.microphysics.morrison import (
        morrison_microphysics,
    )
    from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
    from legoesm.thermo import saturation_mixing_ratio
    n, k = 3, 4
    T = jnp.full((n, k), 285.0)
    p_full = jnp.broadcast_to(jnp.linspace(9.0e4, 7.0e4, k), (n, k))
    q_v = 1.05 * saturation_mixing_ratio(T, p_full)       # supersaturated
    z = jnp.zeros((n, k))
    hyd = HydrometeorState(q_c=jnp.full((n, k), 1e-4), q_r=z, q_i=z, q_s=z,
                           q_g=z, N_c=jnp.full((n, k), 1e8), N_r=z, N_i=z)
    args = (T, q_v, hyd, p_full,
            jnp.broadcast_to(jnp.linspace(9.5e4, 6.5e4, k + 1), (n, k + 1)),
            jnp.full((n, k), 1.0), jnp.full((n, k), 500.0), 600.0)
    on = morrison_microphysics(*args, MorrisonConfig(publish_qc_budget=True))
    off = morrison_microphysics(*args, MorrisonConfig(
        publish_qc_budget=True, liquid_condensation=False))
    assert float(jnp.min(on.qc_budget["condensation"])) > 0.0
    assert float(jnp.max(jnp.abs(off.qc_budget["condensation"]))) == 0.0
    # The latent heating goes with it: the whole temperature-tendency
    # difference is L_v/c_pd times the condensation that was removed.
    dT_gap = np.asarray(on.dT_dt) - np.asarray(off.dT_dt)
    np.testing.assert_allclose(
        dT_gap, constants.L_v / constants.c_pd
        * np.asarray(on.qc_budget["condensation"]), rtol=1e-6, atol=1e-12)


def test_driver_wiring_and_refusals():
    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
    from legoesm.atmosphere.physics.turbulence.integration import (
        _make_hydrostatic_turbulence,
        _make_mpas_turbulence,
    )
    from legoesm.driver.config import ExperimentConfig
    from legoesm.driver.physics_pipeline import (
        thread_morrison_scalars,
        turbulence_config_for,
    )
    from legoesm.forcing.amip_config import AMIPExperimentConfig
    assert (ExperimentConfig._field_defaults["clubb_liquid_handoff"]
            is AMIPExperimentConfig().clubb_liquid_handoff is False)
    from legoesm.driver.config import DycoreConfig, GridConfig
    cfg = ExperimentConfig(grid=GridConfig(grid_type="voronoi"),
                           dycore=DycoreConfig(discretization="mpas"),
                           turbulence="clubb", clubb_prognostic=True,
                           microphysics="morrison", cld_macmic_num_steps=3,
                           clubb_liquid_handoff=True)
    cfg.validate_strict()
    assert turbulence_config_for(cfg).clubb.liquid_handoff is True
    assert thread_morrison_scalars(
        cfg, "morrison", MorrisonConfig()).liquid_condensation is False
    default = cfg._replace(clubb_liquid_handoff=False)
    base = MorrisonConfig()
    assert thread_morrison_scalars(default, "morrison", base) is base
    assert not getattr(turbulence_config_for(default).clubb, "liquid_handoff", False)
    assert ExperimentConfig.from_amip_config(
        cfg.to_amip_config()).clubb_liquid_handoff is True
    for bad in (dict(turbulence="louis"), dict(clubb_prognostic=False),
                dict(microphysics="thompson"), dict(cld_macmic_num_steps=1)):
        with pytest.raises(ValueError, match="clubb_liquid_handoff"):
            cfg._replace(**bad).validate_strict()
    tc = TurbulenceConfig(scheme="clubb", clubb=CLUBBConfig(
        prognostic=True, liquid_handoff=True))
    # an explicit override must agree with the flat flag in both directions
    with pytest.raises(ValueError, match="turbulence_override"):
        turbulence_config_for(cfg._replace(turbulence_override=TurbulenceConfig(
            scheme="clubb", clubb=CLUBBConfig(prognostic=True))))
    with pytest.raises(ValueError, match="turbulence_override"):
        turbulence_config_for(default._replace(turbulence_override=tc))
    assert turbulence_config_for(
        cfg._replace(turbulence_override=tc)).clubb.liquid_handoff is True
    _make_mpas_turbulence(tc, 600.0)                       # the wired lane
    with pytest.raises(ValueError, match="not wired"):
        _make_hydrostatic_turbulence(tc, 600.0)
    with pytest.raises(ValueError, match="prognostic"):
        _make_mpas_turbulence(TurbulenceConfig(scheme="clubb", clubb=CLUBBConfig(
            liquid_handoff=True)), 600.0)


def test_cli_and_deck():
    from scripts.run.run_amip import (
        _postprocess_args,
        build_arg_parser,
        build_config_from_args,
    )
    from legoesm.driver.run_config_yaml import load_yaml_config
    import pathlib
    parser = build_arg_parser()
    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--turbulence", "clubb", "--clubb-prognostic",
        "--microphysics", "morrison", "--clubb-liquid-handoff"]), parser))
    assert cfg.clubb_liquid_handoff is True
    deck = (pathlib.Path(__file__).resolve().parents[2]
            / "config" / "amip" / "amip_production.yaml")
    keys = load_yaml_config(deck, build_arg_parser())
    # production is the CAM6 mapping, with no post-CLUBB saturation drain
    # (CAM6 clubb_do_liqsupersat = .false.) -- user 2026-09-24
    assert keys["clubb_liquid_handoff"] is True
    assert keys["hard_saturation_adjustment"] is False


def _raw_step(s, q_c, cfg):
    from legoesm.atmosphere.physics.turbulence.clubb import (
        clubb_step,
        unpack_clubb_moments,
    )
    kw = s["kw"]
    return clubb_step(s["u"], s["v"], s["T"], s["q"], unpack_clubb_moments(s["m"]),
                      kw["p_full"], kw["p_half"], kw["z_full"], kw["z_half"],
                      s["T"][:, -1], s["q"][:, -1], s["rho"], 150.0, cfg, q_c=q_c)


def test_handed_liquid_is_the_grid_mean_not_the_in_layer_liquid(spun_up):
    """CAM hands back rcm_inout (grid mean); rcm_in_layer is the in-cloud
    value behind the cloud-cover diagnostic and would over-fill cloud edges."""
    s = spun_up
    q_c = jnp.zeros_like(s["q"])
    cfg = CLUBBConfig(prognostic=True, clubb_dt=300.0, liquid_handoff=True)
    *_, diags = _raw_step(s, q_c, cfg)
    handed = np.asarray(q_c) + 150.0 * np.asarray(diags["dq_c_dt"])
    mean = np.asarray(diags["rcm_mean"])[:, ::-1]
    layer = np.asarray(diags["rcm"])[:, ::-1]
    assert np.max(np.abs(layer - mean)) > 1e-6          # the two genuinely differ
    np.testing.assert_allclose(handed, np.clip(mean, 0.0, None), atol=1e-15)


def test_host_liquid_is_kept_above_the_clubb_top(spun_up):
    """Above the CLUBB top the host liquid is blended back in with the same
    log-pressure taper as the cloud fraction (CAM keeps cldliq there)."""
    s = spun_up
    q_c = jnp.full_like(s["q"], 1.0e-5)
    p_top = 6.0e4
    p = np.asarray(s["kw"]["p_full"])

    def handed(cfg):
        *_, diags = _raw_step(s, q_c, cfg)
        return (np.asarray(q_c) + 150.0 * np.asarray(diags["dq_c_dt"]),
                np.asarray(diags["rcm_mean"])[:, ::-1])

    cfg = CLUBBConfig(prognostic=True, clubb_dt=300.0, liquid_handoff=True,
                      trop_cloud_top_press=p_top)
    got, mean = handed(cfg)
    w = 1.0 / (1.0 + np.exp(-(np.log(p) - np.log(p_top))
                            / cfg.trop_cloud_taper_lnp_width))
    expect = np.clip(w * mean + (1.0 - w) * 1.0e-5, 0.0, None)
    np.testing.assert_allclose(got, expect, rtol=1e-10, atol=1e-18)
    # Control: without the top, the liquid far above is CLUBB's, not the host's.
    above = p < 0.5 * p_top
    assert above.any()
    no_top, mean0 = handed(cfg._replace(trop_cloud_top_press=0.0))
    assert np.max(np.abs(no_top[above] - got[above])) > 1e-7
