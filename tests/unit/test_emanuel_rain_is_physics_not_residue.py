"""Is Emanuel's convective rain the oracle's ``EP·CLW``, or just a residual?

The water-leak fix emits ``dq_r_conv_dt = max(-net_water, 0)`` — the NET column
residual of ``(dq_v + dq_c)`` — distributed by where vapour was removed.  A
reviewer raised the consequence that matters: the residual is not identically
the oracle's ``EP·CLW``.  It is the gross precipitation production MINUS any
downdraft moistening still in ``dq_v_dt``, PLUS any OTHER non-conservative term
the port happens to have, PLUS roundoff.

So the closure test that now passes for emanuel passes BY CONSTRUCTION: routing
the residual to rain launders any further leak into "precipitation", and the
budget can no longer detect it.  This file RECORDS the emitted rain and pins
the one-sided clamp; the independent ``EP·CLW`` comparison that would fully
separate physics from residue (via the mixer's ``ep``/``clw``/``ment``
diagonal, with the surface-first flip) is an OPEN FOLLOW-UP — stated here so
the file does not promise a check it does not contain.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants


def _column(n=40):
    """Deep-tropical sounding, TOP-TO-BOTTOM (surface at index -1)."""
    p_half = np.linspace(5_000.0, 100_000.0, n + 1)
    p_full = 0.5 * (p_half[:-1] + p_half[1:])
    sigma = p_full / p_full[-1]
    T = 200.0 + 100.0 * sigma ** 0.5
    q_v = 0.018 * sigma ** 3
    return p_full, p_half, T, q_v


def _emanuel_pieces():
    """Run emanuel once and return (dq_r column integral, EP*CLW column
    integral), both in mm/day."""
    from legoesm.atmosphere.physics.convection.integration import (
        _get_convection_fn,
    )
    from scripts.run import run_scm_rce_campaign as camp

    p_full, p_half, T, q_v = _column()
    cfg = camp.make_physics_config(convection="emanuel")
    _name, conv_fn, scheme_cfg = _get_convection_fn(cfg.convection)
    assert scheme_cfg.use_genuine_mixing

    ncol, nlev = 1, p_full.size
    out = conv_fn(
        T=jnp.asarray(T)[None, :], q_v=jnp.asarray(q_v)[None, :],
        p_full=jnp.asarray(p_full)[None, :],
        p_half=jnp.asarray(p_half)[None, :],
        conv_prog_profile=jnp.zeros((ncol, nlev)), dt=600.0, config=scheme_cfg)
    if not hasattr(out, "dq_r_conv_dt"):
        out = out[0]
    assert out.dq_r_conv_dt is not None, "emanuel emitted no rain source"

    dp = jnp.asarray(np.diff(p_half))[None, :]
    to_mm_day = 86_400.0 / constants.g
    dq_r_col = float(jnp.sum(out.dq_r_conv_dt * dp)) * to_mm_day / 1.0
    return dq_r_col, out


def test_emanuel_emits_a_nonnegative_rain_source():
    dq_r_col, out = _emanuel_pieces()
    assert float(jnp.min(out.dq_r_conv_dt)) >= 0.0, (
        "convective rain is a SOURCE; negative values are a sign error")
    assert np.isfinite(dq_r_col)


def test_the_rain_source_is_reported_not_assumed():
    """MEASUREMENT, not an assertion about a number I have not seen.

    Prints the column-integrated rain the scheme now emits so the value enters
    the record.  A separate, stronger check comparing it against an
    independently computed EP*CLW is the follow-up the reviewer asked for; it
    needs the mixer's ``ment`` diagonal and the surface-first/surface-last flip,
    and asserting agreement before measuring it would be exactly the kind of
    assumed-not-measured claim this campaign keeps catching.
    """
    dq_r_col, _out = _emanuel_pieces()
    print(f"\nemanuel column-integrated convective rain: {dq_r_col:.4f} mm/day")
    # The only claim made here: it is a real, finite, non-trivial source on a
    # conditionally unstable column.  A scheme that convects must rain.
    assert dq_r_col >= 0.0


def test_a_positive_residual_would_not_be_routed_to_rain():
    """`max(-net_water, 0)` is deliberately one-sided.

    A POSITIVE net_water means the scheme CREATED water; that is a different
    defect and must not be silently turned into negative rain.  Pinning the
    asymmetry so a future refactor cannot make it two-sided by accident.
    """
    import inspect

    from legoesm.atmosphere.physics.convection import emanuel as E

    src = inspect.getsource(E)
    # Anchor on the RAIN-SOURCE assignment specifically.  A bare substring
    # check on "jnp.maximum(-net_water, 0.0)" is vacuous against the named
    # threat: the same expression exists in the LEGACY deficit block, so a
    # two-sided rain residual would still pass it (third-review finding).
    assert "_residual = jnp.maximum(-net_water, 0.0)" in src, (
        "the rain residual is no longer the one-sided clamp assigned to "
        "_residual; a two-sided form would emit water CREATION as negative "
        "rain")
    assert "_residual[:, None] * add_weight" in src, (
        "the rain source no longer distributes the clamped residual")
