"""Cloud-fraction saturation-curve selector (``CloudConfig.saturation_scheme``).

The RH the diagnostic cloud-fraction schemes threshold against was measured
against LIQUID (Tetens) saturation at every temperature, so genuinely
ICE-saturated cold air (TTL / tropical anvil, ~205-245 K, where the liquid
curve sits up to ~60% above the ice curve) read RH ~0.55-0.75 < rh_crit and
the schemes diagnosed NO cloud where the model carried substantial detrained
ice (#1521: production day-365 anvil, Sundqvist cf = 0.000 at every level
while 77% of anvil cells were super-saturated over the mixed-phase curve).

``saturation_scheme="mixed_phase"`` measures RH against the liquid/ice curve
blended by the scheme's OWN condensate ice-fraction ramp (T_freeze ->
T_ice_only; IFS alpha(T) convention, Tiedtke 1993), so the RH criterion and
the diagnosed condensate phase agree.  ``"liquid"`` stays the byte-identical
legacy default.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics.clouds.cloud_fraction import (
    compute_cloud_properties,
)
from legoesm.atmosphere.physics.clouds.config import CloudConfig, build_cloud_config
from legoesm.thermo import saturation_mixing_ratio, saturation_mixing_ratio_ice

jax.config.update("jax_enable_x64", True)


def _cold_ice_saturated_column(nlev: int = 12):
    """Column whose cold upper levels are exactly ICE-saturated.

    Radiation column convention: index 0 = model TOP (cold), last = surface.
    q_v is set to the ICE saturation value at the cold levels (< T_ice_only)
    and to 0.4 * liquid saturation at the warm levels (sub-saturated, no
    cloud either way).
    """
    T = jnp.linspace(210.0, 295.0, nlev)[None, :]
    p_full = jnp.linspace(2.0e4, 9.8e4, nlev)[None, :]
    p_half = jnp.linspace(1.8e4, 1.0e5, nlev + 1)[None, :]
    dp = p_half[:, 1:] - p_half[:, :-1]
    q_sat_liq = saturation_mixing_ratio(T, p_full)
    q_sat_ice = saturation_mixing_ratio_ice(T, p_full)
    cold = T < 233.15                       # below default T_ice_only
    q_v = jnp.where(cold, q_sat_ice, 0.4 * q_sat_liq)
    return T, p_full, dp, q_v, np.asarray(cold)[0]


def test_liquid_default_is_field_default():
    assert CloudConfig().saturation_scheme == "liquid"


def test_ice_saturated_anvil_liquid_zero_mixed_positive():
    """THE defect: ice-saturated cold levels give cf=0 on the liquid curve and
    cf>0 (indeed ~1 at exact ice saturation) on the mixed-phase curve."""
    T, p_full, dp, q_v, cold = _cold_ice_saturated_column()
    base = CloudConfig(scheme="sundqvist", rh_crit=0.85)
    cf_liq = np.asarray(compute_cloud_properties(
        T, p_full, q_v, dp, base).cloud_fraction)
    cf_mix = np.asarray(compute_cloud_properties(
        T, p_full, q_v, dp,
        base._replace(saturation_scheme="mixed_phase")).cloud_fraction)
    # Liquid curve: RH_liq = q_sat_ice/q_sat_liq < rh_crit at cold T => no cloud.
    assert float(cf_liq[0, cold].max()) == 0.0
    # Mixed curve: RH -> 1 at the ice-saturated levels => full cloud.
    assert float(cf_mix[0, cold].min()) > 0.95
    # Warm sub-saturated levels identical (weight w(T)=1 above T_freeze;
    # in-ramp warm levels are sub-saturated on BOTH curves at RH 0.4).
    warm = ~cold
    np.testing.assert_allclose(cf_mix[0, warm], cf_liq[0, warm], atol=1e-12)


def test_warm_column_bit_identical():
    """T >= T_freeze everywhere => the blend weight is exactly 0 ice, so the
    mixed_phase path returns the SAME q_sat and identical cloud fields."""
    nlev = 8
    T = jnp.linspace(275.0, 300.0, nlev)[None, :]
    p_full = jnp.linspace(5.0e4, 9.8e4, nlev)[None, :]
    p_half = jnp.linspace(4.8e4, 1.0e5, nlev + 1)[None, :]
    dp = p_half[:, 1:] - p_half[:, :-1]
    q_v = 0.9 * saturation_mixing_ratio(T, p_full)
    base = CloudConfig(scheme="sundqvist", rh_crit=0.8)
    out_liq = compute_cloud_properties(T, p_full, q_v, dp, base)
    out_mix = compute_cloud_properties(
        T, p_full, q_v, dp, base._replace(saturation_scheme="mixed_phase"))
    np.testing.assert_array_equal(np.asarray(out_liq.cloud_fraction),
                                  np.asarray(out_mix.cloud_fraction))
    np.testing.assert_array_equal(np.asarray(out_liq.lwp),
                                  np.asarray(out_mix.lwp))


def test_mixed_qsat_bounded_by_pure_curves():
    """Below freezing the blended q_sat lies between the ice and liquid curves
    (=> RH_mixed >= RH_liq, cloud onset can only move EARLIER, never later)."""
    T = jnp.linspace(200.0, 272.0, 30)[None, :]
    p_full = jnp.full((1, 30), 3.0e4)
    q_sat_liq = np.asarray(saturation_mixing_ratio(T, p_full))
    q_sat_ice = np.asarray(saturation_mixing_ratio_ice(T, p_full))
    # Reproduce the scheme's blend via its public output: feed q_v = blended
    # q_sat by construction and check cf == cf(RH=1) exactly is overkill;
    # instead check the curve ordering that the blend inherits.
    assert (q_sat_ice <= q_sat_liq + 1e-30).all()
    cfg = CloudConfig(scheme="sundqvist", saturation_scheme="mixed_phase")
    from legoesm.atmosphere.physics.clouds.cloud_fraction import _ice_fraction
    w_ice = np.asarray(_ice_fraction(T, cfg))
    q_sat_mix = (1.0 - w_ice) * q_sat_liq + w_ice * q_sat_ice
    assert (q_sat_mix <= q_sat_liq + 1e-30).all()
    assert (q_sat_mix >= q_sat_ice - 1e-30).all()


def test_xu_randall_also_gains_cold_cloud():
    """The selector feeds BOTH RH schemes (xu_randall's RH and its q_sat
    denominator), not just sundqvist."""
    T, p_full, dp, q_v, cold = _cold_ice_saturated_column()
    q_i = jnp.where(jnp.asarray(cold)[None, :], 2.0e-5, 0.0)
    q_c = jnp.zeros_like(T)
    base = CloudConfig(scheme="xu_randall")
    cf_liq = np.asarray(compute_cloud_properties(
        T, p_full, q_v, dp, base, q_cloud=q_c, q_ice=q_i).cloud_fraction)
    cf_mix = np.asarray(compute_cloud_properties(
        T, p_full, q_v, dp, base._replace(saturation_scheme="mixed_phase"),
        q_cloud=q_c, q_ice=q_i).cloud_fraction)
    assert float(cf_mix[0, cold].min()) > float(cf_liq[0, cold].max())


def test_unknown_saturation_scheme_raises():
    T, p_full, dp, q_v, _ = _cold_ice_saturated_column()
    cfg = CloudConfig(scheme="sundqvist", saturation_scheme="tetens_only")
    with pytest.raises(ValueError, match="saturation_scheme"):
        compute_cloud_properties(T, p_full, q_v, dp, cfg)


def test_jit_parity_and_finite_grad():
    """Eager vs jit identical; gradient of mean cf w.r.t. q_v finite (the
    blend is smooth — no new AD hazard)."""
    T, p_full, dp, q_v, _ = _cold_ice_saturated_column()
    cfg = CloudConfig(scheme="sundqvist", saturation_scheme="mixed_phase")

    def f(qv):
        return jnp.mean(compute_cloud_properties(
            T, p_full, qv, dp, cfg).cloud_fraction)

    eager = f(q_v)
    jitted = jax.jit(f)(q_v)
    np.testing.assert_allclose(float(eager), float(jitted), rtol=1e-12)
    g = jax.grad(f)(q_v)
    assert np.isfinite(np.asarray(g)).all()


def test_build_cloud_config_round_trip():
    cc = build_cloud_config("sundqvist", saturation_scheme="mixed_phase")
    assert cc.saturation_scheme == "mixed_phase"
    # None => CloudConfig default (legacy liquid, byte-identical).
    assert build_cloud_config("sundqvist").saturation_scheme == "liquid"


def test_experiment_config_validate_strict_membership():
    from legoesm.driver.config import ExperimentConfig
    with pytest.raises(ValueError, match="cloud_saturation_scheme"):
        ExperimentConfig(cloud_saturation_scheme="bogus").validate_strict()
    ExperimentConfig(cloud_saturation_scheme="mixed_phase").validate_strict()


def test_standalone_cloud_config_forwards_saturation_scheme():
    """The MPAS lane's builder must forward the curve, or the flag is accepted
    by the CLI and then SILENTLY DROPPED on the production lane."""
    from types import SimpleNamespace

    from legoesm.driver.model_driver import _standalone_cloud_config

    cfg = SimpleNamespace(convective_cloud=False,
                          cloud_saturation_scheme="mixed_phase")
    assert _standalone_cloud_config(
        cfg, "sundqvist").saturation_scheme == "mixed_phase"
    # Absent attribute => None => CloudConfig default (legacy).
    assert _standalone_cloud_config(
        SimpleNamespace(convective_cloud=False),
        "sundqvist").saturation_scheme == "liquid"


def test_clt_diagnostic_builder_threads_saturation_scheme():
    """``clt`` is recomputed by the DiagnosticCollector from its OWN
    CloudConfig, built in ``ModelDriver._create_diagnostics`` (the symbol that
    runs: ``ModelDriver.setup`` calls it).  The RH saturation CURVE sets the
    cloud fraction itself, so if that builder does not thread it a
    ``mixed_phase`` run publishes ``clt`` computed on the LIQUID curve while
    radiation integrated the mixed-phase field — the published clt would miss
    exactly the cold cirrus the switch adds (same class as the p_xr/alpha_xr
    gap already fixed there).
    """
    import inspect
    import re

    from legoesm.driver.model_driver import ModelDriver

    src = inspect.getsource(ModelDriver._create_diagnostics)
    # Strip comment lines FIRST: a mention in a comment must not satisfy the
    # gate (a test that cannot fail proves nothing).
    code = "\n".join(ln for ln in src.splitlines()
                     if not ln.lstrip().startswith("#"))
    # The kwarg must live inside the build_cloud_config(...) call, not merely
    # somewhere in the function.
    call = re.search(r"build_cloud_config\((.*?)\n            \)", code,
                     re.DOTALL)
    assert call is not None, "build_cloud_config call not found"
    assert "saturation_scheme=" in call.group(1)
    assert "cloud_saturation_scheme" in call.group(1)
    # And the inspected symbol is live: setup() invokes it.
    assert "_create_diagnostics()" in inspect.getsource(ModelDriver.setup)
