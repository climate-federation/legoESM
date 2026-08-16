"""Every convection scheme must account for the water it removes.

The test a reviewer asked for after Emanuel was found destroying ~88 % of its
convective rain.  Stated generally: whatever water a scheme takes out of the
vapour reservoir must reappear in something it DECLARES — the cloud-water
source ``dq_c_conv_dt`` that this model routes to microphysics, or the
in-updraught rain source ``dq_r_conv_dt`` that the physics pipeline
column-integrates into surface precipitation.  So

    ∫(dq_v + dq_c + dq_r) dp/g == 0

for a scheme that neither creates nor destroys water.

A non-zero residual is water created or destroyed, and it is INVISIBLE in the
temperature field — the latent heat is released during ascent whether or not
the water is accounted for — which is exactly why the Emanuel leak survived so
long and why a temperature-based check would never have caught it.

``KNOWN_LEAKING`` is SHRINK-ONLY: a scheme may only be added with a measured
residual and a reason; removing one requires the residual to be at tolerance.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants

#: Column residual tolerated, in mm/day.  Far below the 2.13 mm/day Emanuel
#: leak and far above float64 round-off on a 40-level column.
CLOSURE_TOL_MM_DAY = 1.0e-3

#: Schemes with a MEASURED water-conservation defect.  SHRINK-ONLY.
#: Empty since the Emanuel EP*CLW leak was fixed by emitting it as
#: ``dq_r_conv_dt`` (the channel bechtold/tiedtke/mass_flux already use).
KNOWN_LEAKING: dict[str, str] = {}

SCHEMES = (
    "sbm", "dca", "kuo", "mass_flux", "edmf",
    "zhang_mcfarlane", "kain_fritsch", "emanuel", "tiedtke", "bechtold",
)


def _column(n=40):
    """A conditionally unstable deep-tropical sounding, TOP-TO-BOTTOM.

    This repo's profiles run top first with the surface at index -1; getting
    that backwards is a documented way to produce a plausible wrong number.
    """
    p_half = np.linspace(5_000.0, 100_000.0, n + 1)
    p_full = 0.5 * (p_half[:-1] + p_half[1:])
    sigma = p_full / p_full[-1]
    T = 200.0 + 100.0 * sigma ** 0.5
    q_v = 0.018 * sigma ** 3
    return p_full, p_half, T, q_v


def _residual_mm_day(scheme: str) -> float:
    """``∫(dq_v + dq_c + dq_r) dp/g`` for one convection call, in mm/day."""
    from legoesm.atmosphere.physics.convection.integration import (
        _get_convection_fn,
    )
    from scripts.run import run_scm_rce_campaign as camp

    p_full, p_half, T, q_v = _column()
    cfg = camp.make_physics_config(convection=scheme)
    _name, conv_fn, scheme_cfg = _get_convection_fn(cfg.convection)
    if conv_fn is None:
        pytest.skip(f"{scheme}: no backend function")

    kwargs = dict(
        T=jnp.asarray(T)[None, :],
        q_v=jnp.asarray(q_v)[None, :],
        p_full=jnp.asarray(p_full)[None, :],
        p_half=jnp.asarray(p_half)[None, :],
        config=scheme_cfg,
    )
    try:
        out = conv_fn(**kwargs)
    except TypeError as exc:
        pytest.skip(f"{scheme}: leaf signature needs more inputs ({exc})")

    dp = jnp.asarray(np.diff(p_half))[None, :]
    water = out.dq_v_dt + out.dq_c_conv_dt
    if getattr(out, "dq_r_conv_dt", None) is not None:
        water = water + out.dq_r_conv_dt
    net = jnp.sum(water * dp, axis=-1) / constants.g      # kg/m^2/s
    return float(jnp.abs(net)[0]) * 86_400.0              # -> mm/day


@pytest.mark.parametrize("scheme", SCHEMES)
def test_convection_neither_creates_nor_destroys_water(scheme):
    residual = _residual_mm_day(scheme)
    if scheme in KNOWN_LEAKING:
        assert residual > CLOSURE_TOL_MM_DAY, (
            f"{scheme} is in KNOWN_LEAKING ({KNOWN_LEAKING[scheme]}) but its "
            f"residual is now {residual:.3e} mm/day. If the leak is fixed, "
            "REMOVE it from KNOWN_LEAKING in the same commit.")
        return
    assert residual <= CLOSURE_TOL_MM_DAY, (
        f"{scheme} creates or destroys {residual:.3e} mm/day of water. Every "
        "kg the scheme removes from vapour must be declared as cloud water "
        "(dq_c_conv_dt) or as convective rain (dq_r_conv_dt). This defect is "
        "invisible in the temperature field, so no thermal check will find it.")


def test_emanuel_declares_its_precipitating_condensate():
    """The specific regression: Emanuel's EP*CLW must leave via the rain
    channel, not vanish.  MEASURED before the fix: 2.13 mm/day destroyed,
    precipitation 0.28 against the CRM's 2.4."""
    from legoesm.atmosphere.physics.convection.integration import (
        _get_convection_fn,
    )
    from scripts.run import run_scm_rce_campaign as camp

    p_full, p_half, T, q_v = _column()
    cfg = camp.make_physics_config(convection="emanuel")
    _n, conv_fn, scheme_cfg = _get_convection_fn(cfg.convection)
    assert scheme_cfg.use_genuine_mixing, (
        "this regression is about the genuine-mixing path")
    try:
        out = conv_fn(T=jnp.asarray(T)[None, :], q_v=jnp.asarray(q_v)[None, :],
                      p_full=jnp.asarray(p_full)[None, :],
                      p_half=jnp.asarray(p_half)[None, :], config=scheme_cfg)
    except TypeError as exc:
        pytest.skip(f"emanuel leaf signature needs more inputs ({exc})")
    assert out.dq_r_conv_dt is not None, (
        "emanuel returned no convective rain source; EP*CLW is being dropped "
        "again (see emanuel.py, the deficit/dq_r block)")
    assert float(jnp.min(out.dq_r_conv_dt)) >= 0.0, (
        "convective rain is a SOURCE; a negative value is a sign error")


def test_known_leaking_is_shrink_only():
    """A guard on the guard: the allow-list must never grow."""
    assert KNOWN_LEAKING == {}, (
        "a scheme was added to KNOWN_LEAKING. This list may only SHRINK; a new "
        f"water leak is a defect to fix, not to register. Got: {KNOWN_LEAKING}")
