"""Homogeneous freezing of cloud water tests (iter-25).

gSAM ``MICRO_M2005`` (module_mp_graupel.f90:4661-4669): below 233.15 K (−40 °C)
all supercooled cloud water freezes instantly to cloud ice (``QHOMOC=q_c/dt``),
the droplet number becomes ice number (``NHOMOC=N_c/dt``), and the latent heat
of fusion is released. legoESM previously had NO cloud-water freezing, so
supercooled cloud water persisted below −40 °C. legoESM uses a smooth sigmoid
threshold (AD-safe) rather than SAM's hard switch.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
from legoesm.atmosphere.physics.microphysics.morrison import (
    morrison_microphysics,
)
from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
from legoesm.thermo import saturation_mixing_ratio


jax.config.update("jax_enable_x64", True)


def _run(T, do_homo=True, q_c=5.0e-4, p=2.0e4):
    qsl = float(saturation_mixing_ratio(jnp.asarray(T), jnp.asarray(p)))
    rho = p / (constants.R_d * T)
    z = jnp.zeros((1, 1))
    hm = HydrometeorState(
        q_c=jnp.full((1, 1), q_c), q_r=z, q_i=z, q_s=z, q_g=z,
        N_c=z, N_r=z, N_i=z,
    )
    out = morrison_microphysics(
        jnp.full((1, 1), T), jnp.full((1, 1), qsl), hm,
        jnp.full((1, 1), p), jnp.full((1, 2), p),
        jnp.full((1, 1), rho), jnp.full((1, 1), 300.0), 20.0,
        MorrisonConfig(do_homogeneous_freezing=do_homo),
    )
    return (float(out.dq_c_dt[0, 0]), float(out.dq_i_dt[0, 0]),
            float(out.dT_dt[0, 0]), float(out.dN_i_dt[0, 0]))


def test_cloud_water_freezes_below_minus40():
    """At −45 °C cloud water glaciates: q_c sink, q_i source, warming, ice
    number injected."""
    dc, di, dT, dNi = _run(228.0)
    assert dc < 0.0 and di > 0.0
    assert dT > 0.0          # L_f released
    assert dNi > 0.0         # droplet number → ice number


def test_no_homogeneous_freezing_above_minus40():
    """The homogeneous-freezing contribution (on−off) is ~0 at −30 °C and
    large at −45 °C — the −40 °C threshold."""
    on30, off30 = _run(243.0)[0], _run(243.0, do_homo=False)[0]
    on45, off45 = _run(228.0)[0], _run(228.0, do_homo=False)[0]
    contrib30 = abs(on30 - off30)
    contrib45 = abs(on45 - off45)
    assert contrib45 > 1000.0 * (contrib30 + 1e-30)


def test_conserves_water_and_releases_latent_heat():
    """on−off difference isolates the homogeneous freezing: q_c → q_i (Δdq_c =
    −Δdq_i) with the fusion latent heat warming (ΔdT > 0)."""
    dc_on, di_on, dT_on, dNi_on = _run(228.0)
    dc_off, di_off, dT_off, dNi_off = _run(228.0, do_homo=False)
    assert (dc_on - dc_off) == pytest.approx(-(di_on - di_off), abs=1e-12)
    assert (dc_on - dc_off) < 0.0          # cloud water frozen
    assert dT_on > dT_off                  # warmer with freezing
    assert dNi_on > dNi_off                # ice number injected


def test_disabled_flag_removes_freezing():
    """do_homogeneous_freezing=False ⇒ no cloud-water freezing at −45 °C."""
    dc_off, di_off, _, _ = _run(228.0, do_homo=False)
    # Without homogeneous freezing the cloud water is nearly inert at −45 °C
    # (only slow WBF/deposition), so |dq_c| is tiny vs the q_c/dt freeze rate.
    assert abs(dc_off) < 5.0e-4 / 20.0 * 0.5


def test_homogeneous_freezing_ad_safe():
    qsl = float(saturation_mixing_ratio(jnp.asarray(228.0), jnp.asarray(2.0e4)))

    def loss(q_c0):
        rho = 2.0e4 / (constants.R_d * 228.0)
        z = jnp.zeros((1, 1))
        hm = HydrometeorState(
            q_c=q_c0.reshape(1, 1), q_r=z, q_i=z, q_s=z, q_g=z,
            N_c=z, N_r=z, N_i=z,
        )
        out = morrison_microphysics(
            jnp.full((1, 1), 228.0), jnp.full((1, 1), qsl), hm,
            jnp.full((1, 1), 2.0e4), jnp.full((1, 2), 2.0e4),
            jnp.full((1, 1), rho), jnp.full((1, 1), 300.0), 20.0,
            MorrisonConfig(),
        )
        return jnp.sum(out.dq_i_dt)

    for q0 in (0.0, 5.0e-4):
        g = jax.grad(loss)(jnp.asarray(q0))
        assert bool(jnp.isfinite(g))
