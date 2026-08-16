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


#: Schemes whose leaf cannot be called with a bare column here (they need a
#: carry whose shape this test cannot synthesise).  SHRINK-ONLY, and asserted
#: below — a silent `pytest.skip` on TypeError made an earlier version of this
#: file report "1 passed, 11 skipped", i.e. it did not test the very bug it was
#: written for.
NOT_DIRECTLY_CALLABLE: dict[str, str] = {}


def _leaf_kwargs(conv_fn, scheme_cfg, p_full, p_half, T, q_v):
    """Build the leaf's arguments from its ACTUAL signature.

    Read, not guessed: the leaves differ (emanuel takes a prognostic profile,
    bechtold also takes winds, a stochastic carry and a PRNG key), and several
    return a tuple rather than a bare ConvectionOutput.
    """
    import inspect

    import jax

    ncol, nlev = 1, p_full.shape[0]
    zeros_2d = jnp.zeros((ncol, nlev))
    supply = {
        "T": jnp.asarray(T)[None, :],
        "q_v": jnp.asarray(q_v)[None, :],
        "p_full": jnp.asarray(p_full)[None, :],
        "p_half": jnp.asarray(p_half)[None, :],
        "dt": 600.0,
        "config": scheme_cfg,
        "u": zeros_2d,
        "v": zeros_2d,
        "conv_prog_profile": zeros_2d,
        "conv_stoch_state": zeros_2d,
        "prng_key": jax.random.PRNGKey(0),
    }
    sig = inspect.signature(conv_fn)
    kwargs, missing = {}, []
    for name, param in sig.parameters.items():
        if name in supply:
            kwargs[name] = supply[name]
        elif param.default is inspect.Parameter.empty:
            missing.append(name)
    return kwargs, missing


def _call_leaf(scheme: str):
    """Return the ConvectionOutput for one scheme, or (None, reason)."""
    from legoesm.atmosphere.physics.convection.integration import (
        _get_convection_fn,
    )
    from scripts.run import run_scm_rce_campaign as camp

    p_full, p_half, T, q_v = _column()
    cfg = camp.make_physics_config(convection=scheme)
    _name, conv_fn, scheme_cfg = _get_convection_fn(cfg.convection)
    if conv_fn is None:
        return None, "no backend function"
    kwargs, missing = _leaf_kwargs(conv_fn, scheme_cfg, p_full, p_half, T, q_v)
    if missing:
        return None, f"cannot synthesise required args {missing}"
    out = conv_fn(**kwargs)
    if isinstance(out, tuple):        # emanuel returns (output, carry)
        out = out[0]
    return out, None


def _residual_mm_day(scheme: str) -> float:
    """``∫(dq_v + dq_c + dq_r) dp/g`` for one convection call, in mm/day."""
    out, reason = _call_leaf(scheme)
    assert out is not None, (
        f"{scheme}: {reason}. Add it to NOT_DIRECTLY_CALLABLE with a reason "
        "rather than letting the test silently pass.")
    _p_full, p_half, _T, _q = _column()
    dp = jnp.asarray(np.diff(p_half))[None, :]
    water = out.dq_v_dt + out.dq_c_conv_dt
    if getattr(out, "dq_r_conv_dt", None) is not None:
        water = water + out.dq_r_conv_dt
    net = jnp.sum(water * dp, axis=-1) / constants.g      # kg/m^2/s
    return float(jnp.abs(net)[0]) * 86_400.0              # -> mm/day


@pytest.mark.parametrize("scheme", SCHEMES)
def test_convection_neither_creates_nor_destroys_water(scheme):
    if scheme in NOT_DIRECTLY_CALLABLE:
        pytest.skip(f"{scheme}: {NOT_DIRECTLY_CALLABLE[scheme]}")
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

    cfg = camp.make_physics_config(convection="emanuel")
    _n, _fn, scheme_cfg = _get_convection_fn(cfg.convection)
    assert scheme_cfg.use_genuine_mixing, (
        "this regression is about the genuine-mixing path")
    out, reason = _call_leaf("emanuel")
    assert out is not None, f"emanuel could not be called: {reason}"
    assert out.dq_r_conv_dt is not None, (
        "emanuel returned no convective rain source; EP*CLW is being dropped "
        "again (see emanuel.py, the deficit/dq_r block)")
    assert float(jnp.min(out.dq_r_conv_dt)) >= 0.0, (
        "convective rain is a SOURCE; a negative value is a sign error")


def test_every_scheme_is_actually_exercised():
    """The guard that an earlier version of this file needed and lacked.

    It skipped on TypeError and reported "1 passed, 11 skipped" — nine schemes,
    including the one the file exists for, were never tested at all. A skip is
    only acceptable when it is DECLARED.
    """
    assert NOT_DIRECTLY_CALLABLE == {}, (
        "a scheme became uncallable by this harness. Fix the harness or record "
        f"the reason; this list may only SHRINK. Got: {NOT_DIRECTLY_CALLABLE}")
    covered = [s for s in SCHEMES if s not in NOT_DIRECTLY_CALLABLE]
    assert "emanuel" in covered, "emanuel is the regression this file exists for"
    assert len(covered) == len(SCHEMES)


def test_known_leaking_is_shrink_only():
    """A guard on the guard: the allow-list must never grow."""
    assert KNOWN_LEAKING == {}, (
        "a scheme was added to KNOWN_LEAKING. This list may only SHRINK; a new "
        f"water leak is a defect to fix, not to register. Got: {KNOWN_LEAKING}")
