"""Unit tests for the optional prognostic aerosol-number tracer.

Covers :mod:`legoesm.atmosphere.physics.microphysics.prognostic_aerosol`:

  * SOURCES increase number (surface emission; SO2->sulfate oxidation);
  * SINKS decrease number (surface dry deposition; precipitation wet
    scavenging), with the correct tendency sign;
  * number stays >= 0 and the implicit sink cannot drive it negative in one
    step, at ANY step size;
  * ``enabled=False`` leaves the state byte-identical (proxy path untouched);
  * the prognostic number feeds ARG activation (Part-2 -> Part-1 hand-off);
  * the step is differentiable.
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.microphysics.prognostic_aerosol import (  # noqa: E402
    PrognosticAerosolConfig,
    aerosol_number_tendency,
    step_prognostic_aerosol,
)

_NCOL, _NLEV = 4, 6
_DZ = jnp.full((_NCOL, _NLEV), 500.0)          # 500 m layers
_NO_PRECIP = jnp.zeros((_NCOL,))
_CFG = PrognosticAerosolConfig(enabled=True)   # defaults, switched on


def test_emission_increases_number_at_surface_only():
    n0 = jnp.zeros((_NCOL, _NLEV))
    n1 = step_prognostic_aerosol(n0, 100.0, _DZ, _NO_PRECIP, _CFG)
    # Surface (bottom) layer gains aerosol; interior stays zero (emission is a
    # surface source and there is no oxidation precursor here).
    assert jnp.all(n1[:, -1] > 0.0)
    assert jnp.allclose(n1[:, :-1], 0.0)
    # Number is non-negative everywhere.
    assert jnp.all(n1 >= 0.0)


def test_oxidation_precursor_is_a_volumetric_source():
    n0 = jnp.zeros((_NCOL, _NLEV))
    so2 = jnp.full((_NCOL, _NLEV), 1.0e8)
    n_no_ox = step_prognostic_aerosol(n0, 100.0, _DZ, _NO_PRECIP, _CFG)
    n_ox = step_prognostic_aerosol(
        n0, 100.0, _DZ, _NO_PRECIP, _CFG, so2_precursor=so2)
    # Oxidation adds particles at every level; interior levels rise above the
    # emission-only (zero-interior) baseline.
    assert jnp.all(n_ox[:, :-1] > n_no_ox[:, :-1])
    assert jnp.all(n_ox >= 0.0)


def test_wet_scavenging_decreases_number():
    n0 = jnp.full((_NCOL, _NLEV), 2.0e8)
    precip = jnp.full((_NCOL,), 1.0e-3)            # kg/m^2/s
    # Isolate the sink: zero the surface emission so only scavenging/dep act.
    n1 = step_prognostic_aerosol(
        n0, 100.0, _DZ, precip, _CFG,
        emission_flux=jnp.zeros((_NCOL,)))
    assert jnp.all(n1 < n0)                        # scavenging removes aerosol
    assert jnp.all(n1 >= 0.0)


def test_dry_deposition_acts_at_surface():
    n0 = jnp.full((_NCOL, _NLEV), 2.0e8)
    # No emission, no precip: only surface dry deposition removes aerosol.
    n1 = step_prognostic_aerosol(
        n0, 100.0, _DZ, _NO_PRECIP, _CFG,
        emission_flux=jnp.zeros((_NCOL,)))
    assert jnp.all(n1[:, -1] < n0[:, -1])          # surface loses aerosol
    assert jnp.allclose(n1[:, :-1], n0[:, :-1])    # interior unchanged (no sink)


def test_tendency_signs():
    # Emission-only, N=0: surface tendency is a pure positive source.
    dndt_src = aerosol_number_tendency(
        jnp.zeros((_NCOL, _NLEV)), _DZ, _NO_PRECIP, _CFG)
    assert jnp.all(dndt_src[:, -1] > 0.0)
    # Sink-only (no emission, N>0, precip>0): tendency is negative.
    n0 = jnp.full((_NCOL, _NLEV), 1.0e8)
    dndt_sink = aerosol_number_tendency(
        n0, _DZ, jnp.full((_NCOL,), 1.0e-3), _CFG,
        emission_flux=jnp.zeros((_NCOL,)))
    assert jnp.all(dndt_sink < 0.0)


def test_positivity_under_huge_step_and_precip():
    # Even with an enormous timestep and torrential precip, the implicit sink
    # can never drive the number negative.
    n0 = jnp.full((_NCOL, _NLEV), 5.0e7)
    n1 = step_prognostic_aerosol(
        n0, 1.0e6, _DZ, jnp.full((_NCOL,), 1.0), _CFG,
        emission_flux=jnp.zeros((_NCOL,)))
    assert jnp.all(n1 >= 0.0)
    assert jnp.all(jnp.isfinite(n1))
    # And with a negative-going explicit tendency the state still cannot cross
    # zero after the implicit update.
    n2 = step_prognostic_aerosol(
        jnp.zeros((_NCOL, _NLEV)), 1.0e6, _DZ,
        jnp.full((_NCOL,), 1.0), _CFG, emission_flux=jnp.zeros((_NCOL,)))
    assert jnp.all(n2 >= 0.0)


def test_positivity_holds_for_garbage_config():
    # Robustness: even a physically invalid config (negative emission and
    # negative sink coefficients) cannot break positivity — the source and
    # sink rate are clamped >= 0 internally, so 1 + dt*L never vanishes and N
    # never goes negative.
    n0 = jnp.full((_NCOL, _NLEV), 1.0e8)
    cfg = PrognosticAerosolConfig(
        enabled=True,
        emission_number_flux_m2_s=-1.0e9,
        dry_dep_velocity_m_s=-1.0,
        wet_scavenging_coeff_m2_kg=-5.0)
    n1 = step_prognostic_aerosol(
        n0, 1.0e5, _DZ, jnp.full((_NCOL,), 1.0), cfg)
    assert jnp.all(n1 >= 0.0)
    assert jnp.all(jnp.isfinite(n1))
    # Negative "sources"/"sinks" clamp to zero -> the state is simply conserved.
    assert jnp.allclose(n1, n0)


def test_zero_oxidation_timescale_stays_finite():
    # A zero/garbage oxidation timescale must not inject NaN/inf (tau_ox is
    # floored before the 1/tau_ox division).
    cfg = PrognosticAerosolConfig(enabled=True, so2_oxidation_timescale_s=0.0)
    so2 = jnp.full((_NCOL, _NLEV), 1.0e8)
    n1 = step_prognostic_aerosol(
        jnp.zeros((_NCOL, _NLEV)), 100.0, _DZ, _NO_PRECIP, cfg,
        so2_precursor=so2)
    assert jnp.all(jnp.isfinite(n1))
    assert jnp.all(n1 >= 0.0)


def test_disabled_is_byte_identical():
    n0 = jnp.full((_NCOL, _NLEV), 1.234e8)
    cfg_off = PrognosticAerosolConfig(enabled=False)
    n1 = step_prognostic_aerosol(
        n0, 100.0, _DZ, jnp.full((_NCOL,), 1.0e-3), cfg_off)
    assert jnp.array_equal(n1, n0)                 # state untouched


def test_prognostic_number_feeds_arg_activation():
    # Part-2 -> Part-1 hand-off: the prognostic aerosol number drives ARG.
    from legoesm.atmosphere.physics.microphysics.arg_activation import (
        ActivationConfig,
        arg_cdnc_from_config,
    )
    T = jnp.full((_NCOL, _NLEV), 285.0)
    p = jnp.full((_NCOL, _NLEV), 9.5e4)
    cfg = ActivationConfig(scheme="arg", w_char_m_s=0.3)
    n_low = jnp.full((_NCOL, _NLEV), 50.0e6)       # 50 cm^-3
    n_high = jnp.full((_NCOL, _NLEV), 500.0e6)     # 500 cm^-3
    cdnc_low = arg_cdnc_from_config(cfg, T, p, aerosol_number=n_low)
    cdnc_high = arg_cdnc_from_config(cfg, T, p, aerosol_number=n_high)
    assert cdnc_low.shape == (_NCOL, _NLEV)
    assert jnp.all(cdnc_high > cdnc_low)           # more aerosol -> more CDNC
    # Activated number never exceeds the available aerosol.
    assert jnp.all(cdnc_low <= n_low)
    assert jnp.all(cdnc_high <= n_high)


def test_step_is_differentiable():
    n0 = jnp.full((_NCOL, _NLEV), 1.0e8)
    precip = jnp.full((_NCOL,), 1.0e-3)

    def mean_number(emission_scale):
        cfg = _CFG._replace(
            emission_number_flux_m2_s=1.0e8 * emission_scale)
        return jnp.mean(
            step_prognostic_aerosol(n0, 100.0, _DZ, precip, cfg))

    g = jax.grad(mean_number)(1.0)
    assert jnp.isfinite(g) and g > 0.0             # more emission -> more N
