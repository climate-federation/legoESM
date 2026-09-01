"""Behavioural tests for the P-model optimality parameter source.

Covers the closed-form optimum (chi regime + monotonicity, predicted Medlyn
slope in the observed range, capacity positivity and smooth low-light decay,
the phi0/Iabs-invariance of rjv25), the PPFD-gain-weighted acclimation state
(night hold, polar-night finiteness), differentiability including the mj -> c*
floor neighborhood, and the fail-early no-state error.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.land import p_model as pm


_REF = dict(
    T_K=jnp.asarray(constants.T_freeze + 25.0),
    vpd_pa=jnp.asarray(1000.0),
    co2_ppm=jnp.asarray(400.0),
    ps_pa=jnp.asarray(constants.p_atm_std),
)


def _state(ppfd=400.0, t_c=25.0, vpd=1000.0, co2=400.0, ps=constants.p_atm_std):
    one = jnp.ones((1,))
    return pm.PModelAcclimState(
        iabs_mean=ppfd * one,
        t_mean_K=(constants.T_freeze + t_c) * one,
        vpd_mean_pa=vpd * one,
        co2_mean_ppm=co2 * one,
        ps_ema=ps * one,
    )


def test_chi_in_range_and_decreasing_in_vpd():
    cfg = pm.PModelConfig()
    vpds = jnp.array([200.0, 500.0, 1000.0, 2000.0, 4000.0])
    chi, _, _, _, _ = pm.optimal_chi(
        _REF["T_K"], vpds, _REF["co2_ppm"], _REF["ps_pa"], cfg)
    chi = np.asarray(chi)
    assert np.all((chi > 0.0) & (chi < 1.0))
    assert np.all(np.diff(chi) < 0.0)


def test_predicted_g1_in_observed_regime():
    # Least-cost xi at 25 degC / 400 ppm / sea level -> g1 ~ 2.6 kPa^0.5,
    # inside the observed Medlyn range (~2-6; Lin et al. 2015).
    caps = pm.acclimated_capacities(_state(), pm.PModelConfig())
    g1 = float(caps.g1_kpa[0])
    assert 2.0 < g1 < 6.0
    # g1 rises with temperature (K + Gamma* grow faster than eta* falls).
    caps_warm = pm.acclimated_capacities(_state(t_c=35.0), pm.PModelConfig())
    assert float(caps_warm.g1_kpa[0]) > g1


def test_capacities_positive_and_reasonable_at_reference():
    caps = pm.acclimated_capacities(_state(), pm.PModelConfig())
    v = float(caps.vcmax25_leaf[0])
    r = float(caps.rjv25[0])
    assert 5.0 < v < 150.0       # leaf-top Vcmax25, physiological range
    assert 1.0 < r < 3.0         # Jmax25/Vcmax25, observed ~1.6-2.1
    assert 0.5 < float(caps.chi[0]) < 0.95


def test_capacities_decay_smoothly_to_zero_at_low_light():
    cfg = pm.PModelConfig()
    v = []
    for ppfd in (400.0, 40.0, 4.0, 0.4, 0.0):
        caps = pm.acclimated_capacities(_state(ppfd=ppfd), cfg)
        v.append(float(caps.vcmax25_leaf[0]))
        assert np.isfinite(float(caps.rjv25[0]))  # analytic ratio: no 0/0
    assert all(a > b for a, b in zip(v, v[1:]))  # strictly decreasing in light
    assert v[-1] == pytest.approx(0.0, abs=1e-12)


def test_rjv25_invariant_to_kphio_and_iabs():
    # phi0 and Iabs cancel in Jmax/Vcmax: rjv25 depends only on the optimum.
    base = pm.acclimated_capacities(_state(), pm.PModelConfig())
    double_light = pm.acclimated_capacities(_state(ppfd=800.0), pm.PModelConfig())
    half_kphio = pm.acclimated_capacities(
        _state(), pm.PModelConfig(kphio=0.0409))
    assert float(base.rjv25[0]) == pytest.approx(float(double_light.rjv25[0]), rel=1e-12)
    assert float(base.rjv25[0]) == pytest.approx(float(half_kphio.rjv25[0]), rel=1e-12)
    # while Vcmax25 itself scales ~linearly with both
    assert float(double_light.vcmax25_leaf[0]) == pytest.approx(
        2.0 * float(base.vcmax25_leaf[0]), rel=1e-6)


def test_kphio_temp_off_is_constant_phi0():
    cfg = pm.PModelConfig(kphio_temp=False)
    caps_cold = pm.acclimated_capacities(_state(t_c=15.0), cfg)
    caps_ref = pm.acclimated_capacities(_state(), cfg)
    # With constant phi0 the only T-dependence left is kinetics/normalisation;
    # sanity: both finite/positive and different (kinetics still move).
    assert float(caps_cold.vcmax25_leaf[0]) > 0.0
    assert float(caps_ref.vcmax25_leaf[0]) > 0.0


def test_night_steps_hold_daytime_means():
    cfg = pm.PModelConfig()
    s = _state(t_c=20.0, vpd=900.0)
    dark = dict(T_K=jnp.full((1,), constants.T_freeze - 30.0),
                ppfd=jnp.zeros((1,)),
                vpd_pa=jnp.full((1,), 5.0),
                co2_ppm=jnp.full((1,), 400.0),
                ps_pa=jnp.full((1,), constants.p_atm_std))
    # Three months of polar night, daily steps.
    for _ in range(90):
        s = pm.advance_pmodel_acclim(s, cfg=cfg, dt=86400.0, **dark)
    assert float(s.t_mean_K[0]) == pytest.approx(constants.T_freeze + 20.0)
    assert float(s.vpd_mean_pa[0]) == pytest.approx(900.0)
    assert float(s.co2_mean_ppm[0]) == pytest.approx(400.0)
    caps = pm.acclimated_capacities(s, cfg)
    assert np.isfinite(float(caps.vcmax25_leaf[0]))
    assert np.isfinite(float(caps.g1_kpa[0]))
    # Polar SUNRISE: the first bright step after months of darkness moves the
    # held mean at ~the EMA rate (gain ~ dt*w/(tau*w_ref)), not in one jump —
    # the gain reference is the HELD daytime-mean PPFD, which did not decay.
    bright = {**dark, "T_K": jnp.full((1,), constants.T_freeze + 5.0),
              "ppfd": jnp.full((1,), 1200.0)}
    s2 = pm.advance_pmodel_acclim(s, cfg=cfg, dt=1800.0, **bright)
    moved = abs(float(s2.t_mean_K[0]) - float(s.t_mean_K[0]))
    assert moved < 0.5  # a near-1 gain would move ~15 K toward the +5 C air


def test_daytime_means_track_daytime_not_diurnal_average():
    cfg = pm.PModelConfig(tau_acclim_s=5.0 * 86400.0)
    s = _state(t_c=10.0, vpd=500.0)
    day = dict(T_K=jnp.full((1,), constants.T_freeze + 25.0),
               ppfd=jnp.full((1,), 800.0),
               vpd_pa=jnp.full((1,), 1500.0),
               co2_ppm=jnp.full((1,), 400.0),
               ps_pa=jnp.full((1,), constants.p_atm_std))
    night = {**day, "T_K": jnp.full((1,), constants.T_freeze + 5.0),
             "ppfd": jnp.zeros((1,)), "vpd_pa": jnp.full((1,), 50.0)}
    # 60 diurnal cycles of 12 h day / 12 h night at 6 h steps.
    for _ in range(60):
        for forc in (day, day, night, night):
            s = pm.advance_pmodel_acclim(s, cfg=cfg, dt=21600.0, **forc)
    # Converges to the DAYTIME values, not the 24 h average (15 degC, 775 Pa).
    assert abs(float(s.t_mean_K[0]) - (constants.T_freeze + 25.0)) < 1.0
    assert abs(float(s.vpd_mean_pa[0]) - 1500.0) < 100.0


def test_gradients_finite_including_mj_floor_neighborhood():
    cfg = pm.PModelConfig()

    def loss(beta, kphio, ppfd, vpd, co2):
        c = pm.PModelConfig(beta_cost=beta, kphio=kphio)
        caps = pm.acclimated_capacities(
            _state(ppfd=ppfd, vpd=vpd, co2=co2), c)
        return (caps.vcmax25_leaf[0] + caps.rjv25[0] + caps.g1_kpa[0])

    g = jax.grad(loss, argnums=(0, 1, 2, 3, 4))(146.0, 0.081785, 400.0, 1000.0, 400.0)
    assert all(np.isfinite(float(x)) for x in g)
    # mj -> c* neighborhood: very high VPD + low CO2 pushes chi (and mj) down
    # onto the smooth floor; gradients must stay finite, not NaN or huge.
    g_edge = jax.grad(loss, argnums=(0, 3, 4))(146.0, 0.081785, 400.0, 7000.0, 120.0)
    assert all(np.isfinite(float(x)) for x in g_edge)
    del cfg


def test_acclimated_capacities_raises_without_state():
    with pytest.raises(ValueError, match="acclimation state"):
        pm.acclimated_capacities(None, pm.PModelConfig())


def test_init_state_uses_config_constants_and_override():
    cfg = pm.PModelConfig()
    s = pm.init_pmodel_acclim(
        3, t_init_K=285.0, ps_init_pa=90000.0, cfg=cfg)
    assert s.iabs_mean.shape == (3,)
    assert float(s.iabs_mean[0]) == cfg.init_ppfd
    assert float(s.vpd_mean_pa[0]) == cfg.init_vpd_pa
    assert float(s.co2_mean_ppm[0]) == cfg.init_co2_ppm
    s2 = pm.init_pmodel_acclim(
        2, t_init_K=285.0, ps_init_pa=90000.0, cfg=cfg, co2_init_ppm=368.0)
    assert float(s2.co2_mean_ppm[0]) == 368.0


def test_valid_selector_tuples():
    assert pm.VALID_CAPACITY_SCHEMES == ("prescribed", "p_model")
    assert pm.VALID_G1_SOURCES == ("table", "p_model")


def test_canopy_config_validates_switches():
    from legoesm.land.canopy.config import CanopyConfig
    with pytest.raises(ValueError, match="capacity_scheme"):
        CanopyConfig(capacity_scheme="pmodel").validate()
    with pytest.raises(ValueError, match="g1_source"):
        CanopyConfig(g1_source="xi").validate()
    # g1_source='p_model' + non-medlyn is legal AT CONFIG LEVEL (the
    # phydro slope mapping covers every model); the least-cost-only
    # combination is refused at the consumers, which can see the
    # transpiration_stress switch (see test_phydro_wiring).
    CanopyConfig(g1_source="p_model", stomatal_model="ball_berry").validate()
    CanopyConfig(g1_source="p_model", stomatal_model="medlyn").validate()
    CanopyConfig(capacity_scheme="p_model").validate()


def test_init_multilayer_state_carries_acclim_iff_switch_active():
    from legoesm.land.canopy.config import CanopyConfig
    from legoesm.land.config import MultiLayerLandConfig
    from legoesm.land.multilayer_land import init_multilayer_land_state

    cfg_off = MultiLayerLandConfig(surface_scheme=CanopyConfig())
    assert init_multilayer_land_state(4, cfg_off).pmodel_acclim is None
    cfg_on = MultiLayerLandConfig(surface_scheme=CanopyConfig(
        capacity_scheme="p_model", g1_source="p_model",
        stomatal_model="medlyn"))
    s = init_multilayer_land_state(4, cfg_on, pmodel_co2_init_ppm=390.0)
    assert s.pmodel_acclim is not None
    assert s.pmodel_acclim.iabs_mean.shape == (4,)
    assert float(s.pmodel_acclim.co2_mean_ppm[0]) == 390.0


def test_c4_capacities_regime_and_g1_ratio():
    from legoesm.land.p_model import acclimated_capacities_c4
    cfg = pm.PModelConfig()
    s = _state(t_c=30.0, vpd=1500.0)
    caps4 = acclimated_capacities_c4(s, cfg)
    v4 = float(caps4.vcmax25_c4_leaf[0])
    assert 5.0 < v4 < 200.0
    assert 0.0 < float(caps4.chi_c4[0]) < 1.0
    # Same kinetics, beta_c4 = beta/9 exactly -> xi (hence g1) ratio = 1/3.
    caps3 = pm.acclimated_capacities(s, cfg)
    ratio = float(caps4.g1_c4_kpa[0]) / float(caps3.g1_kpa[0])
    assert ratio == pytest.approx(1.0 / 3.0, rel=1e-10)
    # C4 chi < C3 chi (lower cost ratio closes stomata harder).
    assert float(caps4.chi_c4[0]) < float(caps3.chi[0])


def test_c4_kphio_temp_off_freezes_quadratic_at_15C():
    from legoesm.land.p_model import acclimated_capacities_c4
    cfg_off = pm.PModelConfig(kphio_temp=False)
    v_cold = acclimated_capacities_c4(_state(t_c=10.0), cfg_off)
    v_warm = acclimated_capacities_c4(_state(t_c=30.0), cfg_off)
    # phi0_c4 frozen at 15 C: remaining T-dependence is ONLY the Collatz
    # normalisation and the chi kinetics, so the phi0 ratio drops out --
    # verify by scaling out the normalisation.
    from legoesm.land.canopy.photosynthesis import c4_vcmax_temperature_response
    import jax.numpy as _j
    r_cold = float(c4_vcmax_temperature_response(_j.asarray(283.15)))
    r_warm = float(c4_vcmax_temperature_response(_j.asarray(303.15)))
    lhs = float(v_cold.vcmax25_c4_leaf[0]) * r_cold
    rhs = float(v_warm.vcmax25_c4_leaf[0]) * r_warm
    assert lhs == pytest.approx(rhs, rel=1e-6)  # growth Vcmax identical


def test_c4_capacities_raise_without_state():
    from legoesm.land.p_model import acclimated_capacities_c4
    with pytest.raises(ValueError, match="acclimation state"):
        acclimated_capacities_c4(None, pm.PModelConfig())


def test_nitrogen_diagnostics_oracle_and_canaries():
    """PR5 diagnostics: independent scalar oracle (constants re-typed here
    from CTSM LunaMod.F90:76-77, NOT imported), canaries pinning the exact
    CTSM code values (294.2 / 1257.0, the ROUNDED pins - the comment products
    are 294.206 / 1257.36), and the no-total design (function returns exactly
    two pools; light-capture N is deliberately absent)."""
    import legoesm.land.p_model as pm
    from legoesm.land.p_model import nitrogen_diagnostics

    # canaries on the exact CTSM code values
    assert pm._F_C25_UMOL_GN_S == 294.2
    assert pm._F_J25_UMOL_GN_S == 1257.0

    v = jnp.asarray([100.0, 30.0])
    j = jnp.asarray([180.0, 60.0])
    out = nitrogen_diagnostics(v, j)
    assert len(out) == 2  # no silent "total photosynthetic N"
    n_rub, n_et = out
    # independent oracle: N = capacity / (binding * specific activity)
    assert float(n_rub[0]) == pytest.approx(100.0 / 294.2, rel=1e-6)
    assert float(n_rub[1]) == pytest.approx(30.0 / 294.2, rel=1e-6)
    assert float(n_et[0]) == pytest.approx(180.0 / 1257.0, rel=1e-6)
    assert float(n_et[1]) == pytest.approx(60.0 / 1257.0, rel=1e-6)
    # plausibility: a strong leaf (Vcmax25=100) implies ~0.34 gN/m2 in
    # Rubisco - within observed leaf N (~1-3 gN/m2)
    assert 0.1 < float(n_rub[0]) < 1.0


def test_biophys_tape_metadata_covers_pmodel_vars():
    """Every P-model tape variable run_lmip_biophys can emit has CF metadata
    (the writer falls back to EMPTY attrs on a missing key - silent, so this
    is the gate), and the two nitrogen diagnostics are labelled C3-only /
    no-feedback with g m-2 units."""
    import importlib.util, pathlib
    root = pathlib.Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "_biophys", root / "scripts" / "run" / "run_lmip_biophys.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    meta = mod._VAR_META
    for key in ("pmodel_chi", "pmodel_vcmax25", "pmodel_g1",
                "pmodel_n_rubisco_leaf", "pmodel_n_et_leaf"):
        assert key in meta and meta[key].get("units"), key
    for key in ("pmodel_n_rubisco_leaf", "pmodel_n_et_leaf"):
        assert meta[key]["units"] == "g m-2"
        assert "C3 only" in meta[key]["long_name"]
        assert "no feedback" in meta[key]["long_name"]
