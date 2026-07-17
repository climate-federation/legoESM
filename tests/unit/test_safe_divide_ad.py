"""Issue #249 regression: AD-safe ``clip + divide`` replacements.

Each test below constructs a state in which the offending denominator
(column-condensation total, ``q_i`` / ``x_c``, ``cos_lat * cos_delta``,
``N``, ``tau_sat``) collapses to zero — exactly the configuration that
makes the legacy ``a / jnp.clip(b, eps, None)`` pattern emit
``-a / b**2`` cotangents under ``jax.value_and_grad``.  After the
``safe_divide`` fix the gradients must be finite.

A passing pre-fix smoke run (``a / jnp.clip(b, ...)``) would yield
``+inf``/``NaN`` gradients here; the assertions catch the regression.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.atmosphere.physics._shared import safe_divide
from legoesm.atmosphere.physics.convection.sbm import sbm_convection
from legoesm.atmosphere.physics.convection.dca import dca_convection
from legoesm.atmosphere.physics.convection.kuo import kuo_convection
from legoesm.atmosphere.physics.convection.config import (
    SBMConfig, DCAConfig, KuoConfig,
)
from legoesm.atmosphere.physics.microphysics.morrison import morrison_microphysics
from legoesm.atmosphere.physics.microphysics.thompson import thompson_microphysics
from legoesm.atmosphere.physics.microphysics.seifert_beheng import (
    seifert_beheng_microphysics,
)
from legoesm.atmosphere.physics.microphysics.config import (
    MorrisonConfig, ThompsonConfig, SeifertBehengConfig,
)
from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
from legoesm.atmosphere.physics.radiation.solar import (
    daily_mean_insolation, daylight_fraction,
)
from legoesm.atmosphere.physics.gravity_wave_drag.lindzen import lindzen_gwd
from legoesm.atmosphere.physics.gravity_wave_drag.prognostic_spectral import (
    prognostic_spectral_gwd,
)
from legoesm.atmosphere.physics.gravity_wave_drag.config import (
    LindzenConfig, PrognosticSpectralConfig,
)


# ---------------------------------------------------------------------------
# safe_divide direct unit tests
# ---------------------------------------------------------------------------

class TestSafeDivide:
    """Direct exercises of the helper itself."""

    def test_forward_matches_plain_divide_above_eps(self):
        a = jnp.array([1.0, 2.5, -3.0, 7.0])
        b = jnp.array([2.0, 5.0, -1.5, 1e-3])  # all |b| > 1e-6
        eps = 1e-6
        out = safe_divide(a, b, eps=eps)
        ref = a / b
        assert jnp.allclose(out, ref, atol=0.0, rtol=0.0)

    def test_forward_returns_fill_at_zero_denominator(self):
        a = jnp.array([1.0, 2.0, -3.0])
        b = jnp.array([0.0, 1e-30, -1e-25])
        out = safe_divide(a, b, eps=1e-20, fill=0.0)
        assert jnp.all(out == 0.0)

    def test_forward_returns_custom_fill(self):
        a = jnp.array([1.0])
        b = jnp.array([0.0])
        out = safe_divide(a, b, eps=1e-20, fill=42.0)
        assert float(out[0]) == 42.0

    def test_grad_finite_at_zero_denominator(self):
        # The whole point of the helper.  ``jax.grad`` of plain
        # ``a / jnp.clip(b, eps)`` would emit ``-a / eps**2`` here.
        def f(b):
            a = jnp.ones_like(b)
            return jnp.sum(safe_divide(a, b, eps=1e-20))
        b0 = jnp.array([0.0, 0.0, 0.0])
        g = jax.grad(f)(b0)
        assert jnp.all(jnp.isfinite(g))
        assert jnp.all(g == 0.0)  # cotangent path masked out

    def test_grad_matches_plain_divide_above_eps(self):
        # In the unmasked branch the two implementations must agree on
        # the gradient as well, not just the forward.
        def f_safe(a, b):
            return jnp.sum(safe_divide(a, b, eps=1e-20))
        def f_plain(a, b):
            return jnp.sum(a / b)
        a = jnp.array([1.0, 2.0, 3.0])
        b = jnp.array([0.5, 1.5, -2.0])
        g_safe_a, g_safe_b = jax.grad(f_safe, argnums=(0, 1))(a, b)
        g_plain_a, g_plain_b = jax.grad(f_plain, argnums=(0, 1))(a, b)
        assert jnp.allclose(g_safe_a, g_plain_a, atol=0.0, rtol=0.0)
        assert jnp.allclose(g_safe_b, g_plain_b, atol=0.0, rtol=0.0)

    def test_grad_handles_signed_denominator_at_zero(self):
        # ``|denominator| > eps`` mask catches both polarities.
        def f(b):
            a = jnp.full_like(b, 2.0)
            return jnp.sum(safe_divide(a, b, eps=1e-10))
        b = jnp.array([1e-20, -1e-20, 0.0])
        g = jax.grad(f)(b)
        assert jnp.all(jnp.isfinite(g))


# ---------------------------------------------------------------------------
# Helper: build a "no-convection" column where ``col_local_cond`` collapses
# ---------------------------------------------------------------------------

def _stable_dry_column(nlev=12, ncol=2):
    """Strongly stable, very dry column — convection trigger off, column
    integrals of any ``max(-dq_v, 0)`` style quantity collapse to ~0.
    """
    p_s = 1.0e5
    sigma_half = jnp.linspace(0.0, 1.0, nlev + 1)
    sigma_full = 0.5 * (sigma_half[:-1] + sigma_half[1:])
    p_half = jnp.broadcast_to((sigma_half * p_s)[None, :], (ncol, nlev + 1))
    p_full = jnp.broadcast_to((sigma_full * p_s)[None, :], (ncol, nlev))
    T = jnp.full((ncol, nlev), 250.0)  # cold and uniform
    q_v = jnp.full((ncol, nlev), 1e-12)  # essentially dry
    return T, q_v, p_full, p_half


# ---------------------------------------------------------------------------
# Convection schemes — column-rescaling singularity (sbm.py / dca.py / kuo.py)
# ---------------------------------------------------------------------------

class TestConvectionADSafety:
    """``dq_c_conv_dt = local_cond * (col_net_drying / col_local_cond)``
    with ``col_local_cond → 0`` in a non-convecting column."""

    def _scheme_loss(self, scheme_fn, config, T, q_v, p_full, p_half):
        out = scheme_fn(T, q_v, p_full, p_half, 300.0, config=config)
        # Sum every output field that flows from the divide so a single
        # NaN anywhere in the chain trips the assertion.
        return (
            jnp.sum(out.dT_dt)
            + jnp.sum(out.dq_v_dt)
            + jnp.sum(out.dq_c_conv_dt)
            + jnp.sum(out.cape)
        )

    @pytest.mark.parametrize("scheme_fn,config", [
        (sbm_convection, SBMConfig()),
        (dca_convection, DCAConfig()),
        (kuo_convection, KuoConfig()),
    ], ids=["sbm", "dca", "kuo"])
    def test_grad_finite_in_non_convecting_column(self, scheme_fn, config):
        T, q_v, p_full, p_half = _stable_dry_column()

        def loss(qv):
            return self._scheme_loss(scheme_fn, config, T, qv, p_full, p_half)

        val, grad = jax.value_and_grad(loss)(q_v)
        assert jnp.isfinite(val), f"forward NaN/inf for {scheme_fn.__name__}"
        assert jnp.all(jnp.isfinite(grad)), (
            f"non-finite gradient through {scheme_fn.__name__} "
            f"in non-convecting column — issue #249 regression"
        )


# ---------------------------------------------------------------------------
# Microphysics — ``dN_*_dt`` divides with vanishing cloud field
# ---------------------------------------------------------------------------

def _cloud_free_state(nlev=8, ncol=2):
    """Cloud-free column: ``q_c = q_i = q_s = q_g = 0`` so ``x_c`` and
    ``q_i`` are exactly zero — the legacy clip floor is active."""
    z = jnp.zeros((ncol, nlev))
    return HydrometeorState(
        q_c=z, q_r=z, q_i=z, q_s=z, q_g=z,
        N_c=z, N_r=z, N_i=z,
    )


def _moist_column(nlev=8, ncol=2):
    p_s = 1.0e5
    sigma_half = jnp.linspace(0.0, 1.0, nlev + 1)
    sigma_full = 0.5 * (sigma_half[:-1] + sigma_half[1:])
    p_half = jnp.broadcast_to((sigma_half * p_s)[None, :], (ncol, nlev + 1))
    p_full = jnp.broadcast_to((sigma_full * p_s)[None, :], (ncol, nlev))
    T = jnp.full((ncol, nlev), 280.0)
    q_v = jnp.full((ncol, nlev), 5e-3)  # moist but not extreme
    rho = p_full / (constants.R_d * T)
    dz = jnp.full((ncol, nlev), 500.0)
    return T, q_v, p_full, p_half, rho, dz


class TestMicrophysicsADSafety:
    """``dN_c_dt = -dq_c_au * rho / x_c`` and similar — with ``x_c = 0``
    in a cloud-free column."""

    def _scheme_loss(self, scheme_fn, config, q_v, T, hydro, p_full, p_half, rho, dz):
        out = scheme_fn(T, q_v, hydro, p_full, p_half, rho, dz, 300.0, config=config)
        # Touch every number-tendency emerging from the safe_divide.  Schemes
        # that don't carry ice (Seifert-Beheng) leave dN_i_dt at the default
        # zero, so a guarded sum keeps the loss scheme-agnostic.
        loss = (
            jnp.sum(out.dT_dt)
            + jnp.sum(out.dq_v_dt)
            + jnp.sum(out.dq_c_dt)
            + jnp.sum(out.dq_r_dt)
            + jnp.sum(out.dN_c_dt)
            + jnp.sum(out.dN_r_dt)
        )
        if getattr(out, "dN_i_dt", None) is not None:
            loss = loss + jnp.sum(out.dN_i_dt)
        return loss

    @pytest.mark.parametrize("scheme_fn,config", [
        (seifert_beheng_microphysics, SeifertBehengConfig()),
        (morrison_microphysics, MorrisonConfig()),
        (thompson_microphysics, ThompsonConfig()),
    ], ids=["seifert_beheng", "morrison", "thompson"])
    def test_grad_finite_in_cloud_free_column(self, scheme_fn, config):
        T, q_v, p_full, p_half, rho, dz = _moist_column()
        hydro = _cloud_free_state()

        def loss(qv):
            return self._scheme_loss(
                scheme_fn, config, qv, T, hydro, p_full, p_half, rho, dz,
            )

        val, grad = jax.value_and_grad(loss)(q_v)
        assert jnp.isfinite(val), f"forward NaN/inf for {scheme_fn.__name__}"
        assert jnp.all(jnp.isfinite(grad)), (
            f"non-finite gradient through {scheme_fn.__name__} "
            f"with empty cloud field — issue #249 regression"
        )

    @pytest.mark.parametrize("scheme_fn,config", [
        (morrison_microphysics, MorrisonConfig()),
        (thompson_microphysics, ThompsonConfig()),
    ], ids=["morrison", "thompson"])
    def test_dN_i_dt_trace_positive_q_i_matches_legacy_limit(self, scheme_fn, config):
        """Forward regression for the ``aggregation * N_i / q_i`` term
        in the trace-positive-ice regime ``q_i ∈ (1e-15, 1e-12]`` —
        the production code keeps the legacy ``clip(q_i, 1e-15) +
        divide`` form here because the floor is high enough to keep
        the cotangent bounded.  An intermediate codex iteration
        replaced this with a ``safe_divide(eps=1e-12)`` form that
        silently zeroed the aggregation sink for any
        ``|q_i| ≤ 1e-12``, and a follow-up rewrite as a per-mass rate
        was non-equivalent in the sub-floor regime — both regressions
        Codex flagged in rounds 3-4.

        For ``q_i = 5e-13 kg/kg`` (above the legacy clip floor) and
        ``N_i = 1e3 /kg``, the legacy / production formula gives
        ``-aggregation * N_i / q_i = -agg_coeff * f_ice * qi_scale *
        N_i``.  At ``T = 240 K < cooper_T_act``, ``f_ice ≈ 1`` and
        ``qi_scale ≈ 1`` for these tiny sinks, so we expect
        ``dN_i_dt ≈ -config.agg_coeff * N_i ≈ -1.0`` /kg/s.
        """
        T, q_v, p_full, p_half, rho, dz = _moist_column()
        # Force a cold column so the ice fraction sigmoid sits at ≈1.
        T = jnp.full_like(T, 240.0)
        # Pin the LEGACY heuristic dN_i_autoconv (aggregation·N_i/clip(q_i,
        # 1e-15)).  The default schemes use the AD-safe DCS-number-removal form
        # (morrison mg_ferrier via the "mg" flavor; thompson "capacitance"), so
        # select each scheme's heuristic clip path and disable Cooper
        # nucleation (N_i0=0) to isolate the aggregation sink — see
        # test_dN_i_dt_sub_floor_q_i_uses_clip for the full rationale.
        config = config._replace(N_i0=0.0)
        if "morrison_flavor" in config._fields:
            config = config._replace(morrison_flavor="sam")
        if "ice_to_snow_scheme" in config._fields:
            config = config._replace(ice_to_snow_scheme="heuristic")
        if "ice_growth_scheme" in config._fields:
            config = config._replace(ice_growth_scheme="heuristic")
        ncol, nlev = T.shape
        q_i_trace = 5e-13
        N_i_value = 1e3
        hydro = HydrometeorState(
            q_c=jnp.zeros((ncol, nlev)), q_r=jnp.zeros((ncol, nlev)),
            q_i=jnp.full((ncol, nlev), q_i_trace),
            q_s=jnp.zeros((ncol, nlev)), q_g=jnp.zeros((ncol, nlev)),
            N_c=jnp.zeros((ncol, nlev)), N_r=jnp.zeros((ncol, nlev)),
            N_i=jnp.full((ncol, nlev), N_i_value),
        )
        out = scheme_fn(
            T, q_v, hydro, p_full, p_half, rho, dz, 300.0, config=config,
        )
        # Aggregation contribution to dN_i_dt = -agg_coeff * f_ice *
        # qi_scale * N_i.  At trace q_i and dt=300 s, qi_sink_total*dt
        # = (agg_coeff * q_i + melt_ice + ...)*dt ≪ q_i, so qi_scale=1.
        # Cooper nucleation at T=240 K and rho≈1.4 contributes a
        # positive dN_i_nuc; subtract it from the observed
        # dN_i_dt to isolate the aggregation term.
        cooper_term = config.N_i0 * jnp.exp(
            config.cooper_a * jnp.maximum(constants.T_freeze - T, 0.0)
        ) / jnp.clip(rho, 0.1)
        dN_i_nuc_expected = jnp.clip(cooper_term - N_i_value, 0.0) / 300.0
        agg_contribution = out.dN_i_dt - dN_i_nuc_expected
        expected_agg = -config.agg_coeff * N_i_value  # f_ice ≈ 1, qi_scale ≈ 1
        # Allow ~5 % tolerance for the f_ice sigmoid not being exactly 1
        # at T=240 K (sigmoid(5*(265-240))=sigmoid(125) is essentially 1
        # but Cooper nucleation also contributes a tiny dN_i_nuc).
        assert jnp.allclose(
            agg_contribution, expected_agg, atol=0.0, rtol=0.05,
        ), (
            f"{scheme_fn.__name__}: dN_i_dt aggregation contribution "
            f"{float(agg_contribution.reshape(-1)[0]):.6e} does not match the "
            f"legacy q_i→0+ limit {float(expected_agg):.6e} — "
            f"safe_divide form would have returned 0 here (issue #249 "
            f"codex round 3 regression)"
        )

    @pytest.mark.parametrize("scheme_fn,config", [
        (morrison_microphysics, MorrisonConfig()),
        (thompson_microphysics, ThompsonConfig()),
    ], ids=["morrison", "thompson"])
    def test_dN_i_dt_sub_floor_q_i_uses_clip(self, scheme_fn, config):
        """Forward regression in the ``q_i < 1e-15`` sub-clip regime.
        The legacy / production form ``aggregation * N_i / clip(q_i,
        1e-15)`` linearly scales with ``q_i`` here (since
        ``aggregation ∝ q_i`` and the clip-floored denominator is
        constant).  Both the safe_divide(eps=1e-12) and per-mass-rate
        rewrites would deviate: the former zeroes this branch
        entirely; the latter scales like the q_i ≥ 1e-15 limit
        instead of the clipped-denominator linear-in-q_i form.
        """
        T, q_v, p_full, p_half, rho, dz = _moist_column()
        T = jnp.full_like(T, 240.0)
        # Select each scheme's legacy clip-divide dN_i_autoconv form (the one
        # this test pins) and disable Cooper nucleation so dN_i_dt isolates the
        # aggregation sink cleanly:
        #  - morrison: the heuristic ice→snow path needs morrison_flavor="sam";
        #    the default "mg" flavor forces mg_ferrier (DCS-number removal,
        #    which gives ~0 here — the AD-safe form, not a regression).
        #  - thompson: the clip form is ice_growth_scheme="heuristic" (the
        #    default "capacitance" uses the same DCS-number removal).
        #  - N_i0=0 removes Cooper nucleation, whose clamp would otherwise
        #    dominate dN_i_dt and confound the aggregation-sink isolation
        #    (thompson's unclamped Cooper demand is O(1e6) here).
        config = config._replace(N_i0=0.0)
        if "morrison_flavor" in config._fields:
            config = config._replace(morrison_flavor="sam")
        if "ice_to_snow_scheme" in config._fields:
            config = config._replace(ice_to_snow_scheme="heuristic")
        if "ice_growth_scheme" in config._fields:
            config = config._replace(ice_growth_scheme="heuristic")
        ncol, nlev = T.shape
        q_i_subfloor = 1e-16  # 1 decade below the 1e-15 clip floor
        N_i_value = 1e3
        hydro = HydrometeorState(
            q_c=jnp.zeros((ncol, nlev)), q_r=jnp.zeros((ncol, nlev)),
            q_i=jnp.full((ncol, nlev), q_i_subfloor),
            q_s=jnp.zeros((ncol, nlev)), q_g=jnp.zeros((ncol, nlev)),
            N_c=jnp.zeros((ncol, nlev)), N_r=jnp.zeros((ncol, nlev)),
            N_i=jnp.full((ncol, nlev), N_i_value),
        )
        out = scheme_fn(
            T, q_v, hydro, p_full, p_half, rho, dz, 300.0, config=config,
        )
        # Legacy / production: aggregation = agg_coeff * q_i * f_ice * qi_scale.
        # Clip(q_i, 1e-15) = 1e-15 (active).  At trace q_i and dt=300 s the
        # donor-clamp scale qi_scale = min(1, q_i / divisor_floor) where
        # divisor_floor=1e-15 → qi_scale = q_i / 1e-15 = 0.1.
        # So aggregation = agg_coeff * q_i * f_ice * (q_i / 1e-15)
        #               = agg_coeff * f_ice * q_i² / 1e-15.
        # And aggregation * N_i / 1e-15 = agg_coeff * f_ice * N_i * q_i² / 1e-30.
        f_ice = 1.0  # T=240 K, cooper_T_act=265 → sigmoid argument well past +
        qi_scale = q_i_subfloor / 1e-15
        agg_clipped_demand = config.agg_coeff * q_i_subfloor * f_ice  # pre-clamp
        aggregation = agg_clipped_demand * qi_scale
        expected_agg_term = -aggregation * N_i_value / 1e-15
        # Cooper nucleation contributes a positive dN_i_nuc; isolate it.
        cooper_term = config.N_i0 * jnp.exp(
            config.cooper_a * jnp.maximum(constants.T_freeze - T, 0.0)
        ) / jnp.clip(rho, 0.1)
        dN_i_nuc_expected = jnp.clip(cooper_term - N_i_value, 0.0) / 300.0
        agg_contribution = out.dN_i_dt - dN_i_nuc_expected
        assert jnp.allclose(
            agg_contribution, expected_agg_term, atol=1e-12, rtol=0.05,
        ), (
            f"{scheme_fn.__name__}: dN_i_dt at sub-floor q_i={q_i_subfloor:.1e} "
            f"gave {float(agg_contribution.reshape(-1)[0]):.6e}; legacy clip+divide "
            f"yields {float(expected_agg_term):.6e}.  A safe_divide(eps=1e-12) "
            f"or per-mass-rate rewrite would give 0 or ~1e-3 respectively "
            f"(issue #249 codex round 4 regression)."
        )

    @pytest.mark.parametrize("scheme_fn,config", [
        (morrison_microphysics, MorrisonConfig()),
        (thompson_microphysics, ThompsonConfig()),
    ], ids=["morrison", "thompson"])
    def test_dN_i_dt_grad_finite_through_q_i(self, scheme_fn, config):
        """``dN_i_dt`` must stay finite (forward AND VJP) when
        differentiating *through ``q_i``* at exactly zero — the singular
        ice-number autoconversion configuration #249 hardened.

        On the DEFAULT configs reviewed here (morrison mg_ferrier,
        thompson capacitance) the ice→snow number sink is the
        DCS-number-removal form ``min(aggregation / cons22, N_i/dt)`` —
        no ``q_i`` denominator at all, so the cotangent is trivially
        finite.  The selectable heuristic branch instead divides by
        ``clip(q_i, 1e-15)``; the floor keeps that form bounded too.
        Seeds ``N_i > 0`` so the aggregation numerator and the ``q_i``
        path both flow into the tendency (issue #249 codex round 2).
        """
        T, q_v, p_full, p_half, rho, dz = _moist_column()
        ncol, nlev = T.shape

        def loss(q_i):
            hydro = HydrometeorState(
                q_c=jnp.zeros((ncol, nlev)), q_r=jnp.zeros((ncol, nlev)),
                q_i=q_i,  # the singular denominator we're tracing through
                q_s=jnp.zeros((ncol, nlev)), q_g=jnp.zeros((ncol, nlev)),
                N_c=jnp.zeros((ncol, nlev)), N_r=jnp.zeros((ncol, nlev)),
                N_i=jnp.full((ncol, nlev), 1e3),
            )
            out = scheme_fn(
                T, q_v, hydro, p_full, p_half, rho, dz, 300.0, config=config,
            )
            return jnp.sum(out.dN_i_dt)

        q_i0 = jnp.zeros((ncol, nlev))
        val, grad = jax.value_and_grad(loss)(q_i0)
        assert jnp.isfinite(val), (
            f"forward NaN/inf for dN_i_dt in {scheme_fn.__name__}"
        )
        assert jnp.all(jnp.isfinite(grad)), (
            f"non-finite gradient through dN_i_dt wrt q_i in "
            f"{scheme_fn.__name__} at q_i=0 — issue #249 regression"
        )


# ---------------------------------------------------------------------------
# Solar: polar singularity ``cos_lat * cos_delta = 0`` at lat = ±π/2
# ---------------------------------------------------------------------------

class TestSolarADSafety:
    """``cos_hs = -sin_lat sin_delta / (cos_lat cos_delta)`` blows up at
    the poles.  After the fix grad-w.r.t.-lat is finite there."""

    def test_daily_mean_insolation_grad_finite_at_pole(self):
        # Include both poles plus a few mid-latitudes to make sure the
        # mid-latitude branch is not collateral damage.
        lat = jnp.array([-jnp.pi / 2, -0.5, 0.0, 0.5, jnp.pi / 2])

        def loss(la):
            return jnp.sum(daily_mean_insolation(la, day_of_year=80.0))

        val, grad = jax.value_and_grad(loss)(lat)
        assert jnp.isfinite(val)
        assert jnp.all(jnp.isfinite(grad)), (
            "non-finite gradient of daily_mean_insolation at the pole "
            "— issue #249 regression"
        )

    def test_daylight_fraction_grad_finite_at_pole(self):
        lat = jnp.array([-jnp.pi / 2, 0.0, jnp.pi / 2])

        def loss(la):
            return jnp.sum(daylight_fraction(la, day_of_year=80.0))

        val, grad = jax.value_and_grad(loss)(lat)
        assert jnp.isfinite(val)
        assert jnp.all(jnp.isfinite(grad))

    def test_grad_finite_at_pole_fp32_solstice(self):
        """AD regression in fp32 at solstice: ``sin_delta ≈ 0.4`` so the
        numerator does not vanish at the pole.  The legacy
        ``-sin_lat sin_delta / clip(cos_lat cos_delta, _TINY)`` form
        produced ``NaN`` reverse-mode gradients here in fp32 — codex
        round 5 reproduced ``[nan, nan]`` for both functions.  The
        production where-before-divide form keeps gradients finite.
        """
        # day_of_year=172 → boreal summer solstice (delta ≈ +23.45°)
        lat_fp32 = jnp.array([-jnp.pi / 2, jnp.pi / 2], dtype=jnp.float32)

        def loss_insolation(la):
            return jnp.sum(daily_mean_insolation(la, day_of_year=172.0))

        def loss_daylight(la):
            return jnp.sum(daylight_fraction(la, day_of_year=172.0))

        for name, loss in [
            ("daily_mean_insolation", loss_insolation),
            ("daylight_fraction", loss_daylight),
        ]:
            val, grad = jax.value_and_grad(loss)(lat_fp32)
            assert jnp.isfinite(val), f"forward NaN/inf for {name} at fp32 pole"
            assert jnp.all(jnp.isfinite(grad)), (
                f"non-finite gradient through {name} at fp32 pole on "
                f"solstice — issue #249 codex round 5 regression"
            )

    def test_polar_day_night_branches_in_float32(self):
        """Forward regression: in float32, ``jnp.cos(jnp.pi/2) ≈ -4.4e-8``
        carries a negative roundoff sign.  A previous AD-fix iteration
        replaced the ``jnp.clip(cos_lat * cos_delta, _TINY, None)``
        positive floor with ``safe_divide``, which preserved that
        negative roundoff and flipped polar day vs polar night at the
        solstice in fp32 (codex round 2 on issue #249).  Production
        keeps the legacy positive ``_TINY`` floor for this expression;
        this test pins that behaviour.
        """
        # Force fp32: the input dtype propagates through the function.
        lat_fp32 = jnp.array(
            [-jnp.pi / 2, jnp.pi / 2], dtype=jnp.float32,
        )
        # Day 172 = boreal summer solstice (delta ≈ +23.45°).
        # Boreal summer: north pole has 24h sun, south pole has 0h sun.
        Q_solstice = daily_mean_insolation(lat_fp32, day_of_year=172.0)
        df_solstice = daylight_fraction(lat_fp32, day_of_year=172.0)
        # South pole (idx 0): polar night
        assert float(Q_solstice[0]) == pytest.approx(0.0, abs=1.0), (
            "south pole at boreal summer solstice should be polar night"
        )
        assert float(df_solstice[0]) == pytest.approx(0.0, abs=1e-3), (
            "south pole at boreal summer solstice daylight fraction "
            "should be ~0"
        )
        # North pole (idx 1): polar day — daylight fraction ~1
        assert float(df_solstice[1]) == pytest.approx(1.0, abs=1e-3), (
            "north pole at boreal summer solstice daylight fraction "
            "should be ~1"
        )


# ---------------------------------------------------------------------------
# Gravity-wave drag — divisions by ``N`` and ``tau_sat``
# ---------------------------------------------------------------------------

def _isothermal_column(ncol=2, nlev=10):
    """Isothermal hydrostatic column.  ``dtheta/dz`` is small so ``N``
    floors at ``1e-4`` (sqrt of the ``N2_half`` clip), exercising the
    AD-safe ``1/N`` divide in both GWD backends."""
    p_s = 1.0e5
    sigma_half = jnp.linspace(0.0, 1.0, nlev + 1)
    sigma_full = 0.5 * (sigma_half[:-1] + sigma_half[1:])
    p_half = jnp.broadcast_to((sigma_half * p_s)[None, :], (ncol, nlev + 1))
    p_full = jnp.broadcast_to((sigma_full * p_s)[None, :], (ncol, nlev))
    T = jnp.full((ncol, nlev), 250.0)  # isothermal -> N near floor
    z_half = jnp.broadcast_to(
        jnp.linspace(20000.0, 0.0, nlev + 1)[None, :], (ncol, nlev + 1),
    )
    z_full = 0.5 * (z_half[:, :-1] + z_half[:, 1:])
    rho = p_full / (constants.R_d * T)
    u = jnp.full((ncol, nlev), 5.0)  # mild westerly
    v = jnp.zeros((ncol, nlev))
    lat = jnp.zeros(ncol)
    return u, v, T, p_full, p_half, z_full, z_half, rho, lat


class TestGWDADSafety:

    def test_lindzen_tau_sat_floor_preserves_breaking(self):
        """Forward-physics regression: when ``tau_sat`` saturates at its
        ``1e-10`` pre-clip floor and ``tau_carry`` is well above the
        floor, the smooth-breaking sigmoid must still fire.  The first
        AD-fix iteration used ``safe_divide(eps=1e-10)`` with the
        default ``fill=0.0``, which silently zeroed ``excess`` in
        floor-clipped cells and disabled breaking — a forward-physics
        change Codex flagged in adversarial review of issue #249.  The
        production fix tightens ``eps`` to ``1e-12`` (strictly below
        the pre-clip floor) so the divide always proceeds.
        """
        ncol, nlev = 2, 6
        u = jnp.full((ncol, nlev), 0.05)  # below the 0.1 m/s |U_proj| floor
        v = jnp.zeros((ncol, nlev))
        T = jnp.full((ncol, nlev), 250.0)
        p_s = 1.0e5
        sigma_half = jnp.linspace(0.0, 1.0, nlev + 1)
        sigma_full = 0.5 * (sigma_half[:-1] + sigma_half[1:])
        p_half = jnp.broadcast_to((sigma_half * p_s)[None, :], (ncol, nlev + 1))
        p_full = jnp.broadcast_to((sigma_full * p_s)[None, :], (ncol, nlev))
        z_half = jnp.broadcast_to(
            jnp.linspace(20000.0, 0.0, nlev + 1)[None, :], (ncol, nlev + 1),
        )
        z_full = 0.5 * (z_half[:, :-1] + z_half[:, 1:])
        rho = p_full / (constants.R_d * T)
        lat = jnp.zeros(ncol)
        cfg = LindzenConfig(h_topo=2000.0)
        out = lindzen_gwd(
            u, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, cfg,
        )
        # If the safe_divide fill kicks in at the floor, ``excess =
        # - 1`` (the pinned threshold) and the breaking sigmoid clamps to ≈0, so
        # column dissipation collapses to zero.  Production behaviour
        # (and the legacy clip+divide) launches a finite tau_0 that
        # saturates against the tau_sat floor and dissipates.
        assert jnp.all(jnp.isfinite(out.eps_gwd))
        assert jnp.any(jnp.abs(out.eps_gwd) > 0.0), (
            "Lindzen breaking did not fire at the tau_sat floor — "
            "safe_divide eps choice is masking out the floor branch "
            "(issue #249 codex follow-up)"
        )

    def test_lindzen_grad_finite_in_neutral_column(self):
        u, v, T, p_full, p_half, z_full, z_half, rho, lat = _isothermal_column()
        cfg = LindzenConfig()

        def loss(uu):
            out = lindzen_gwd(
                uu, v, T, p_full, p_half, z_full, z_half, rho, lat,
                300.0, cfg,
            )
            return jnp.sum(out.du_dt) + jnp.sum(out.dT_dt)

        val, grad = jax.value_and_grad(loss)(u)
        assert jnp.isfinite(val)
        assert jnp.all(jnp.isfinite(grad)), (
            "non-finite gradient through lindzen_gwd in nearly neutral "
            "stratification — issue #249 regression"
        )

    def test_prognostic_spectral_grad_finite_in_neutral_column(self):
        u, v, T, p_full, p_half, z_full, z_half, rho, lat = _isothermal_column()
        cfg = PrognosticSpectralConfig()
        ncol, nlev = u.shape
        spectrum_in = jnp.full(
            (ncol, cfg.n_azimuths, cfg.n_wavenumbers), cfg.launch_flux,
        )

        def loss(uu):
            out, _ = prognostic_spectral_gwd(
                uu, v, T, p_full, p_half, z_full, z_half, rho, lat,
                300.0, cfg, spectrum_in,
            )
            return jnp.sum(out.du_dt) + jnp.sum(out.dT_dt)

        val, grad = jax.value_and_grad(loss)(u)
        assert jnp.isfinite(val)
        assert jnp.all(jnp.isfinite(grad)), (
            "non-finite gradient through prognostic_spectral_gwd in "
            "nearly neutral stratification — issue #249 regression"
        )


# ---------------------------------------------------------------------------
# mixing_length (Blackadar 1962) — shared by 6 turbulence closures
# ---------------------------------------------------------------------------

class TestMixingLength:
    """``_shared.mixing_length`` factors the asymptotic master length
    ``l = κz / (1 + κz/l_∞)`` that Louis / TKE / CLUBB-lite /
    Holtslag-Boville / EDMF / Smagorinsky-Lilly all share."""

    def test_asymptotic_limits(self):
        from legoesm.atmosphere.physics._shared import mixing_length
        l_inf = 100.0
        # Near surface (κz ≪ l_∞): l → κz.
        z_small = jnp.array([2.0, 5.0])
        l_small = mixing_length(z_small, l_inf)
        assert jnp.allclose(l_small, constants.kappa_vk * z_small, rtol=0.1)
        # Far aloft (κz ≫ l_∞): l → l_∞.
        l_high = mixing_length(jnp.array([1.0e5]), l_inf)
        assert float(l_high[0]) > 0.9 * l_inf
        assert float(l_high[0]) < l_inf

    def test_matches_closed_form_and_monotonic(self):
        from legoesm.atmosphere.physics._shared import mixing_length
        l_inf = 80.0
        z = jnp.array([10.0, 50.0, 200.0, 1000.0])
        k = constants.kappa_vk
        expected = k * z / (1.0 + k * z / l_inf)
        assert jnp.allclose(mixing_length(z, l_inf), expected, rtol=1e-12)
        # Strictly increasing with height.
        assert jnp.all(jnp.diff(mixing_length(z, l_inf)) > 0)

    def test_floor_and_ad_safe_at_zero(self):
        from legoesm.atmosphere.physics._shared import mixing_length
        # z = 0 is clipped to z_floor so l stays finite and differentiable.
        assert jnp.isfinite(mixing_length(jnp.array([0.0]), 100.0)[0])
        g = jax.grad(lambda z: jnp.sum(mixing_length(z, 100.0)))(
            jnp.array([0.0, 1.0, 50.0])
        )
        assert jnp.all(jnp.isfinite(g))
