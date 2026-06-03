"""SAM M2005 Cooper ice nucleation tests (iter-9 M6).

gSAM ``MICRO_M2005`` (INUC=0, ``module_mp_graupel.f90:3387-3398``) fires
deposition-nucleation only where
    (RH_liq ≥ 0.999 AND T ≤ 265.15 K)  OR  RH_ice ≥ 1.08
with target number ``kc2 = min(0.005·exp(0.304·(T_f−T))·1000, 5e5)/ρ``
(canonical Cooper 0.005 /L = 5 /m³ base, capped 500 /L) and an initial
crystal MASS source ``MNUCCD = NNUCCD·MI0``, ``MI0 = 4/3·π·ρ_ci·(10µm)³``.

The legacy legoESM form nucleated in ANY cold air (no supersaturation
gate), used a 1000×-too-high base number, and added no mass.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
from legoesm.atmosphere.physics.microphysics.morrison import (
    morrison_microphysics,
)
from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
from legoesm.thermo import (
    saturation_mixing_ratio,
    saturation_mixing_ratio_ice,
)


jax.config.update("jax_enable_x64", True)


def _nuc(q_v, T=243.0, p=4.0e4, rho=0.6, N_i=0.0, config=None):
    """Return (dN_i_dt, dq_i_dt) for a single ice-free-ish cold cell."""
    z = jnp.zeros((1, 1))
    hm = HydrometeorState(
        q_c=z, q_r=z, q_i=z, q_s=z, q_g=z,
        N_c=z, N_r=z, N_i=jnp.full((1, 1), N_i),
    )
    out = morrison_microphysics(
        jnp.full((1, 1), T), jnp.full((1, 1), q_v), hm,
        jnp.full((1, 1), p), jnp.full((1, 2), p),
        jnp.full((1, 1), rho), jnp.full((1, 1), 300.0), 20.0,
        config if config is not None else MorrisonConfig(),
    )
    return float(out.dN_i_dt[0, 0]), float(out.dq_i_dt[0, 0])


def _qsi(T, p):
    return float(saturation_mixing_ratio_ice(jnp.asarray(T), jnp.asarray(p)))


def test_no_nucleation_in_subsaturated_cold_air():
    """The M6 fix: a cold (T=243 K) but ice-SUBsaturated cell must NOT
    nucleate (RH_ice<1.08, RH_liq<0.999) — the legacy bare-f_ice gate
    produced spurious ice here."""
    qsi = _qsi(243.0, 4.0e4)
    dN, dq = _nuc(0.9 * qsi)               # 90 % ice RH, cold
    assert dN < 1.0                        # ≈ 0 crystals/kg/s
    assert dq < 1.0e-12                    # negligible ice mass source


def test_nucleation_fires_when_ice_supersaturated():
    """RH_ice ≥ 1.08 ⇒ nucleation fires (number AND mass source)."""
    qsi = _qsi(243.0, 4.0e4)
    dN, dq = _nuc(1.15 * qsi)
    assert dN > 0.0                        # crystals form
    assert dq > 0.0                        # MI0 mass source


def test_nucleation_fires_via_liquid_saturated_cold_path():
    """(RH_liq ≥ 0.999 AND T ≤ 265.15) also triggers nucleation even if
    RH_ice < 1.08 is not the trigger — here liquid-saturated at 260 K."""
    T, p = 260.0, 5.0e4
    qsl = float(saturation_mixing_ratio(jnp.asarray(T), jnp.asarray(p)))
    dN, _ = _nuc(1.0 * qsl, T=T, p=p)      # exactly liquid-saturated
    assert dN > 0.0


def test_nucleation_off_above_activation_temperature():
    """Warm (T=275 K) subsaturated air: both gate branches off."""
    qsi = _qsi(275.0, 6.0e4)
    dN, _ = _nuc(0.95 * qsi, T=275.0, p=6.0e4)
    assert dN < 1.0


def test_cooper_number_is_canonical_not_1000x():
    """Base number is the canonical Cooper 5 /m³ (= 0.005 /L), NOT the
    old 5e3 /m³. Target N ≈ min(5·exp(0.304·ΔT), 5e5)/ρ."""
    import math
    T, p, rho = 243.0, 4.0e4, 0.6
    qsi = _qsi(T, p)
    dN, _ = _nuc(1.2 * qsi, T=T, p=p, rho=rho, N_i=0.0)
    # Over dt=20 s starting from N_i=0 the number rate ≈ target/dt.
    target = min(5.0 * math.exp(0.304 * (273.15 - T)), 5.0e5) / rho
    assert dN == pytest.approx(target / 20.0, rel=0.05)
    # The old 5e3 base would give ~1000× this.
    assert dN < target / 20.0 * 2.0


def test_nucleation_number_capped():
    """At very cold T the Cooper number saturates at the 500 /L cap."""
    import math
    T, p, rho = 220.0, 2.0e4, 0.3        # very cold ⇒ uncapped formula huge
    qsi = _qsi(T, p)
    dN, _ = _nuc(1.2 * qsi, T=T, p=p, rho=rho)
    cap = 5.0e5 / rho                      # capped target [1/kg]
    assert dN <= cap / 20.0 * 1.001
    # Formula without cap would exceed the cap here.
    assert 5.0 * math.exp(0.304 * (273.15 - T)) > 5.0e5


def test_nucleation_mass_source_matches_mi0():
    """dq_i from nucleation ≈ dN·MI0 with MI0 = 4/3·π·ρ_ci·(10µm)³."""
    qsi = _qsi(243.0, 4.0e4)
    cfg = MorrisonConfig()
    # Use a config with deposition off-ish by setting q_i=0 so the only
    # ice mass source is nucleation; compare dq_i to dN·MI0.
    dN, dq = _nuc(1.15 * qsi, config=cfg)
    mi0 = 4.0 / 3.0 * jnp.pi * cfg.rho_cloud_ice * cfg.ice_nuc_radius ** 3
    # dq_i_dt ≈ dN·MI0 + small deposition on the floor q_i; within a factor.
    assert dq == pytest.approx(dN * float(mi0), rel=0.2)


def test_nucleation_ad_safe():
    """jax.grad through the smooth gate stays finite across the threshold."""
    def loss(q_v):
        return _nuc_jax(q_v)

    def _nuc_jax(q_v):
        z = jnp.zeros((1, 1))
        hm = HydrometeorState(
            q_c=z, q_r=z, q_i=z, q_s=z, q_g=z,
            N_c=z, N_r=z, N_i=z,
        )
        out = morrison_microphysics(
            jnp.full((1, 1), 243.0), q_v.reshape(1, 1), hm,
            jnp.full((1, 1), 4.0e4), jnp.full((1, 2), 4.0e4),
            jnp.full((1, 1), 0.6), jnp.full((1, 1), 300.0), 20.0,
            MorrisonConfig(),
        )
        return jnp.sum(out.dN_i_dt)

    qsi = _qsi(243.0, 4.0e4)
    for frac in (0.9, 1.05, 1.08, 1.2):
        g = jax.grad(loss)(jnp.asarray(frac * qsi))
        assert bool(jnp.isfinite(g))
