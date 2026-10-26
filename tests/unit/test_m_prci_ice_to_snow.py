"""SAM PRCI ice→snow autoconversion tests (iter-20).

gSAM ``MICRO_M2005`` (module_mp_graupel.f90:3322-3326) converts cloud ice to
snow by the DEPOSITIONAL growth of the ice PSD across the snow-size threshold
DCS::

    LAMI = (ρ_ci·π·N_i/q_i)^⅓,  N0I = N_i·LAMI
    NPRCI = (4/(DCS·ρ_ci))·(q_v−q_sat_i)·ρ·N0I·exp(−LAMI·DCS)·DV/ABI
    PRCI  = (π·ρ_ci·DCS³/6)·NPRCI = (2π·DCS²/3)·ρ·N0I·exp(−LAMI·DCS)·DV·(q_v−q_sat_i)₊/ABI

This replaces legoESM's crude constant-rate ``agg_coeff·q_i·f_ice``. It is
deposition-driven (only positive ice supersaturation grows ice across DCS),
self-gates on N_i, and needs no prognostic snow number (NPRCI is dropped —
single-moment snow).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
from legoesm.atmosphere.physics.microphysics.morrison import (
    morrison_microphysics,
)
from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
from legoesm.thermo import saturation_specific_humidity_ice


jax.config.update("jax_enable_x64", True)


def _run(T=250.0, p=4.0e4, q_i=1.0e-4, N_i=1.0e5, ssat=0.1,
         scheme="m2005_autoconv"):
    """Return (dq_s_dt, dq_i_dt) for a single ice cell at ice supersat ``ssat``
    (q_v = (1+ssat)·q_sat_i). No q_c/q_r/q_s, so the only ice→snow path is
    autoconversion."""
    qsi = float(saturation_specific_humidity_ice(jnp.asarray(T), jnp.asarray(p)))
    q_v = (1.0 + ssat) * qsi
    rho = p / (constants.R_d * T)
    z = jnp.zeros((1, 1))
    hm = HydrometeorState(
        q_c=z, q_r=z, q_i=jnp.full((1, 1), q_i), q_s=z, q_g=z,
        N_c=z, N_r=z, N_i=jnp.full((1, 1), N_i),
    )
    out = morrison_microphysics(
        jnp.full((1, 1), T), jnp.full((1, 1), q_v), hm,
        jnp.full((1, 1), p), jnp.full((1, 2), p),
        jnp.full((1, 1), rho), jnp.full((1, 1), 300.0), 20.0,
        # morrison_flavor="sam" leaves ice_to_snow_scheme untouched; the
        # default "mg" flavor would clobber it to mg_ferrier in
        # resolve_morrison_flavor and these tests pin the SAM PRCI / heuristic
        # paths and the unknown-scheme guard directly.
        MorrisonConfig(
            ice_to_snow_scheme=scheme, N_i0=0.0, morrison_flavor="sam"),
    )
    return float(out.dq_s_dt[0, 0]), float(out.dq_i_dt[0, 0])


def _prci_formula(T=250.0, p=4.0e4, q_i=1.0e-4, N_i=1.0e5, ssat=0.1):
    qsi = float(saturation_specific_humidity_ice(jnp.asarray(T), jnp.asarray(p)))
    rho = p / (constants.R_d * T)
    cons12 = (500.0 * math.pi) ** (1.0 / 3.0)   # ρ_ci=500 (cloud-ice density)
    lami = cons12 * (N_i / max(q_i, 1.0e-9)) ** (1.0 / 3.0)
    n0i = N_i * lami
    dv = 8.794e-5 * T ** 1.81 / p
    dqsidt = constants.L_s * qsi / (constants.R_v * T ** 2)
    abi = 1.0 + dqsidt * constants.L_s / constants.c_pd
    dcs = 250.0e-6
    return ((2 * math.pi / 3) * dcs ** 2 * rho * n0i
            * math.exp(-lami * dcs) * dv * (ssat * qsi) / abi)


def test_prci_matches_sam_formula():
    """With no other snow source, dq_s = PRCI exactly."""
    dq_s, _ = _run()
    assert dq_s == pytest.approx(_prci_formula(), rel=1e-6)


def test_prci_zero_when_subsaturated():
    """Ice-subsaturated ⇒ ice shrinks (sublimates), does NOT grow across DCS
    into snow ⇒ no autoconversion."""
    dq_s, _ = _run(ssat=-0.1)
    assert dq_s == 0.0


def test_prci_self_gates_on_ice_number():
    """N_i=0 ⇒ N0I=0 ⇒ PRCI=0 (no ice to autoconvert)."""
    dq_s, _ = _run(q_i=0.0, N_i=0.0)
    assert dq_s == 0.0


def test_prci_increases_with_supersaturation():
    """More ice supersaturation ⇒ faster depositional growth across DCS ⇒
    more snow autoconversion (PRCI ∝ (q_v−q_sat_i))."""
    lo, _ = _run(ssat=0.05)
    hi, _ = _run(ssat=0.2)
    assert hi > lo > 0.0


def test_prci_conserves_ice_plus_snow():
    """Autoconversion moves mass q_i → q_s: switching schemes changes dq_s
    and dq_i by equal-and-opposite amounts (the only differing term)."""
    ds_m, di_m = _run(scheme="m2005_autoconv")
    ds_h, di_h = _run(scheme="heuristic")
    # Δdq_s = +Δautoconv, Δdq_i = −Δautoconv ⇒ Δdq_s = −Δdq_i.
    assert (ds_m - ds_h) == pytest.approx(-(di_m - di_h), abs=1e-12)


def test_heuristic_scheme_differs_and_available():
    """The legacy constant-rate scheme remains selectable and gives a
    different (here larger) rate than the deposition-driven PRCI."""
    ds_m, _ = _run(scheme="m2005_autoconv")
    ds_h, _ = _run(scheme="heuristic")
    assert ds_h != pytest.approx(ds_m, rel=1e-3)
    assert ds_h > 0.0


def test_prci_ad_safe():
    """jax.grad through PRCI wrt q_i is finite (safe_pow on N_i/q_i; exp
    smooth)."""
    qsi = float(saturation_specific_humidity_ice(jnp.asarray(250.0), jnp.asarray(4.0e4)))

    def loss(qi):
        z = jnp.zeros((1, 1))
        hm = HydrometeorState(
            q_c=z, q_r=z, q_i=qi.reshape(1, 1), q_s=z, q_g=z,
            N_c=z, N_r=z, N_i=jnp.full((1, 1), 1.0e5),
        )
        out = morrison_microphysics(
            jnp.full((1, 1), 250.0), jnp.full((1, 1), 1.1 * qsi), hm,
            jnp.full((1, 1), 4.0e4), jnp.full((1, 2), 4.0e4),
            jnp.full((1, 1), 0.5), jnp.full((1, 1), 300.0), 20.0,
            MorrisonConfig(N_i0=0.0),
        )
        return jnp.sum(out.dq_s_dt)

    for qi0 in (0.0, 1.0e-4):
        g = jax.grad(loss)(jnp.asarray(qi0))
        assert bool(jnp.isfinite(g))


def test_unknown_ice_to_snow_scheme_raises():
    with pytest.raises(ValueError, match="Unknown ice_to_snow_scheme"):
        _run(scheme="aggregate_typo")
