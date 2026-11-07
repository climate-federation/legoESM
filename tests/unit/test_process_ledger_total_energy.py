"""The TOTAL-energy column, and why the dry-enthalpy one cannot attribute a leak.

#1354. The ledger's ``energy`` column is dry enthalpy, ``c_pd * int T dp/g``.
That is the right quantity for attributing HEATING and the wrong one for
attributing a LEAK: adiabatic dynamics converts enthalpy into geopotential and
kinetic energy continuously, every joule of that conversion lands in the
``dynamics`` row (which is a store delta and cannot tell conversion from
leakage), and on one level-5 artifact that row reads 33 W/m^2 with no leak
needed to explain it.

These tests are the non-vacuity argument for the new column. Each one
constructs a transformation with a KNOWN answer and checks that dry enthalpy
gets it wrong in a specific, named way while total energy gets it right.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.diagnostics.process_ledger import (
    column_store_snapshot_column,
    column_total_energy,
    total_energy_entry_column,
)

NLEV = 12
NCOL = 5


@pytest.fixture
def column():
    """A plausible, NON-uniform hydrostatic column set."""
    rng = np.random.default_rng(1354)
    dsigma = jnp.full((NLEV,), 1.0 / NLEV)
    p_s = jnp.asarray(1.0e5 + 3.0e3 * rng.standard_normal(NCOL))
    sigma = (jnp.cumsum(dsigma) - 0.5 * dsigma)
    T = jnp.asarray(200.0 + 90.0 * sigma)[None, :] * jnp.ones((NCOL, 1))
    u = jnp.asarray(5.0 + 30.0 * rng.random((NCOL, NLEV)))
    v = jnp.asarray(-10.0 + 20.0 * rng.random((NCOL, NLEV)))
    q_v = jnp.asarray(1.0e-3 + 8.0e-3 * rng.random((NCOL, NLEV)))
    phis = jnp.asarray(9.81 * 1500.0 * rng.random(NCOL))
    return dict(dsigma=dsigma, p_s=p_s, T=T, u=u, v=v, q_v=q_v, phis=phis)


def test_enthalpy_to_kinetic_conversion_is_invisible_to_the_enthalpy_column(column):
    """THE POINT OF THE WHOLE COLUMN.

    Take a conversion that a conserving adiabatic core performs: cool the
    column and put exactly the lost enthalpy into the wind. Dry enthalpy sees
    a large sink; total energy must see nothing.
    """
    d = column
    dt = 1.0
    # Cool by a level-dependent amount, then accelerate the wind so that the
    # kinetic gain equals the enthalpy loss layer by layer.
    dT = -0.05 * jnp.linspace(0.5, 1.5, NLEV)[None, :] * jnp.ones((NCOL, 1))
    # c_p dT + u du = 0  =>  du = -c_p dT / u
    du = -constants.c_pd * dT / d["u"]

    enth_rate = (constants.c_pd
                 * jnp.sum(dT / dt * d["p_s"][:, None] * d["dsigma"], axis=-1)
                 / constants.g)
    te_rate = total_energy_entry_column(
        d["p_s"], d["dsigma"], dT_dt=dT / dt, du_dt=du / dt, u=d["u"])

    # Dry enthalpy reports a real sink...
    assert float(jnp.max(jnp.abs(enth_rate))) > 10.0, (
        "the constructed conversion is too small for this test to be "
        "meaningful; it must be large in the enthalpy column")
    # ...and total energy reports essentially nothing.
    assert float(jnp.max(jnp.abs(te_rate))) < 1e-9 * float(
        jnp.max(jnp.abs(enth_rate))), (
        f"total-energy rate {float(jnp.max(jnp.abs(te_rate))):.3e} W/m^2 is "
        f"not negligible against the enthalpy rate "
        f"{float(jnp.max(jnp.abs(enth_rate))):.3e} — the conversion is "
        f"leaking into the total-energy column, which is exactly what this "
        f"column exists not to do")


def test_condensation_is_invisible_to_total_energy_and_visible_to_enthalpy(column):
    """Latent heating is a conversion too, and the same argument applies."""
    d = column
    dt = 1.0
    dq_v = -1.0e-6 * jnp.ones((NCOL, NLEV))          # condense vapour
    dT = -constants.L_v * dq_v / constants.c_pd       # exactly its latent heat

    enth_rate = (constants.c_pd
                 * jnp.sum(dT / dt * d["p_s"][:, None] * d["dsigma"], axis=-1)
                 / constants.g)
    te_rate = total_energy_entry_column(
        d["p_s"], d["dsigma"], dT_dt=dT / dt, dq_v_dt=dq_v / dt)

    assert float(jnp.max(jnp.abs(enth_rate))) > 1.0
    assert float(jnp.max(jnp.abs(te_rate))) < 1e-9 * float(
        jnp.max(jnp.abs(enth_rate)))


def test_a_real_heat_source_is_visible_in_both(column):
    """Non-vacuity in the other direction: the column is not identically zero.

    A genuine radiative heating with no compensating term must show up at its
    full size in BOTH columns, or the total-energy column is just returning
    zero and the two tests above prove nothing.
    """
    d = column
    dT = 1.0e-5 * jnp.ones((NCOL, NLEV))   # ~0.86 K/day, no conversion partner
    enth_rate = (constants.c_pd
                 * jnp.sum(dT * d["p_s"][:, None] * d["dsigma"], axis=-1)
                 / constants.g)
    te_rate = total_energy_entry_column(d["p_s"], d["dsigma"], dT_dt=dT)
    np.testing.assert_allclose(np.asarray(te_rate), np.asarray(enth_rate),
                               rtol=1e-12)
    assert float(jnp.min(jnp.abs(te_rate))) > 1.0


def test_a_mass_fix_is_measured_by_SNAPSHOTS_and_the_rate_helper_refuses(column):
    """The mass fixer moves p_s, and the rate helper must not pretend to cover it.

    Review finding: a process that moves ``p_s`` changes BOTH the surface term
    and every layer's mass, so a rate helper that carries the first without
    the second is a mixed convention and is wrong even over flat ground. The
    helper now refuses, and the snapshot pair is what measures it.
    """
    d = column
    dps = 5.0            # +5 Pa, the scale of a mass-fix correction
    with pytest.raises(ValueError, match="FIXED-LAYER-MASS"):
        total_energy_entry_column(d["p_s"], d["dsigma"],
                                  dp_s_dt=jnp.full((NCOL,), dps),
                                  phis=d["phis"])

    # The snapshot pair carries both terms. Bump p_s and re-snapshot: the
    # change contains the surface term AND the layer-mass term, and it is not
    # negligible over 1.5 km of terrain.
    before = column_total_energy(d["p_s"], d["dsigma"], d["T"], u=d["u"],
                                 v=d["v"], q_v=d["q_v"], phis=d["phis"])
    after = column_total_energy(d["p_s"] + dps, d["dsigma"], d["T"], u=d["u"],
                                v=d["v"], q_v=d["q_v"], phis=d["phis"])
    delta = after - before
    surface_only = d["phis"] * dps / constants.g
    assert float(jnp.min(jnp.abs(delta))) > 1.0e3, (
        "a 5 Pa mass fix should move the column total energy by more than a "
        "kJ/m^2; if it does not, this test is not exercising the surface term")
    # The layer-mass half is NOT small next to the surface half -- which is
    # exactly why including only one of them was refused above.
    assert float(jnp.min(jnp.abs(delta - surface_only))) > 0.1 * float(
        jnp.max(jnp.abs(surface_only))), (
        "the layer-mass contribution is negligible against the surface term "
        "here, so this test cannot justify the refusal it is documenting")


def test_a_momentum_tendency_without_its_wind_raises(column):
    """Silently dropping a momentum tendency would under-report a process."""
    d = column
    with pytest.raises(ValueError, match="without"):
        total_energy_entry_column(d["p_s"], d["dsigma"],
                                  du_dt=jnp.zeros((NCOL, NLEV)))


def test_a_band_may_not_carry_the_whole_column_surface_term(column):
    """The surface term belongs to the column, not to a band inside it."""
    d = column
    w = jnp.concatenate([jnp.ones(NLEV // 2), jnp.zeros(NLEV - NLEV // 2)])
    with pytest.raises(ValueError, match="whole"):
        column_total_energy(d["p_s"], d["dsigma"], d["T"], phis=d["phis"],
                            level_weight=w)


def test_store_agrees_with_the_enthalpy_store_when_only_T_is_supplied(column):
    """``column_total_energy`` must reduce to the existing enthalpy store.

    With no winds, no water and no surface geopotential it is the same
    quantity the ledger already snapshots, so the new helper cannot be
    disagreeing with the old one about the part they share.
    """
    d = column
    te = column_total_energy(d["p_s"], d["dsigma"], d["T"])
    old = column_store_snapshot_column(d["p_s"], d["dsigma"], d["T"])[..., 1]
    np.testing.assert_allclose(np.asarray(te), np.asarray(old), rtol=1e-12)


def test_kinetic_and_latent_terms_are_the_size_the_atmosphere_has(column):
    """Sanity range: the terms must be physically plausible, not just finite."""
    d = column
    enth = column_total_energy(d["p_s"], d["dsigma"], d["T"])
    full = column_total_energy(d["p_s"], d["dsigma"], d["T"], u=d["u"],
                               v=d["v"], q_v=d["q_v"], phis=d["phis"])
    ke = column_total_energy(d["p_s"], d["dsigma"], jnp.zeros_like(d["T"]),
                             u=d["u"], v=d["v"])
    lat = column_total_energy(d["p_s"], d["dsigma"], jnp.zeros_like(d["T"]),
                              q_v=d["q_v"])
    # Column enthalpy ~ 2.5e9 J/m^2, column KE ~ 1e6-1e7, latent ~ 1e8.
    assert 1e9 < float(jnp.mean(enth)) < 1e10, float(jnp.mean(enth))
    assert 1e5 < float(jnp.mean(ke)) < 1e8, float(jnp.mean(ke))
    assert 1e7 < float(jnp.mean(lat)) < 1e9, float(jnp.mean(lat))
    assert float(jnp.mean(full)) > float(jnp.mean(enth))
