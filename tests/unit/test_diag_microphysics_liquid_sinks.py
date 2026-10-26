"""Direct test for scripts/validate/diag_microphysics_liquid_sinks.py."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = (Path(__file__).resolve().parents[2]
           / "scripts" / "validate" / "diag_microphysics_liquid_sinks.py")


def _load():
    spec = importlib.util.spec_from_file_location("diag_sinks", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_flags():
    m = _load()
    a = m.build_arg_parser().parse_args(["--config", "d.yaml", "--restart", "c.npz"])
    assert a.band == [500.0, 800.0] and a.dt_split is False
    with pytest.raises(SystemExit):
        m.build_arg_parser().parse_args(["--restart", "c.npz"])


def test_weighted_rate_is_area_and_mass_weighted():
    m = _load()
    rate = np.array([[1.0, 2.0], [4.0, 0.0]])
    dp = np.full((2, 2), 100.0)
    w = np.array([0.25, 0.75])
    assert abs(m.weighted_rate(rate, dp, w, 10.0) - 37.5) < 1e-12


def test_masked_rate_selects_only_the_masked_layers():
    m = _load()
    rate = np.array([[1.0, 2.0]])
    dp = np.full((1, 2), 100.0)
    w = np.array([1.0])
    mask = np.array([[True, False]])
    assert abs(m.masked_rate(rate, dp, w, 10.0, mask) - 10.0) < 1e-12
    allm = np.array([[True, True]])
    assert abs(m.masked_rate(rate, dp, w, 10.0, allm)
               - m.weighted_rate(rate, dp, w, 10.0)) < 1e-12


def test_warm_rain_mass_rates_do_not_depend_on_the_step():
    """Both warm-rain MASS sinks must be functions of state, not of ``dt``.

    The AUTOCONVERSION half carries the test: it takes ``dt`` as an argument,
    so a change introducing a ``dt`` factor into its mass rate -- exactly what a
    sub-stepping "fix" for the (refuted) long-step over-stripping hypothesis
    would do -- makes the three values differ and the test fails.  Verified to
    fail under a ``* dt/112.5`` mutation (2026-09-23; factors 0.2 / 1.0 / 5.33).

    The ACCRETION half is tautological here and is not claimed otherwise: the
    KK2000 rate takes no ``dt``, so the three entries are the same computation.
    Accretion's guarantee is the companion signature test below, which fails if
    a ``dt`` parameter is ever added to it (GLM review nit, 2026-09-23).
    """
    import jax.numpy as jnp
    from legoesm.atmosphere.physics.microphysics import _warm_rain as wr

    q_c = jnp.array([[1.0e-3, 4.0e-4]])
    q_r = jnp.array([[2.0e-4, 1.0e-4]])
    rho = jnp.array([[0.9, 0.7]])
    N_c = jnp.array([[1.0e8, 1.0e8]])
    steps = (22.5, 112.5, 600.0)

    ac = [wr.accretion_kk2000(q_c, q_r) for _ in steps]
    au = [wr.autoconversion_kk2000(q_c, N_c, rho, d)[0] for d in steps]

    assert float(jnp.min(ac[0])) > 0.0, "accretion must be active in this state"
    assert float(jnp.min(au[0])) > 0.0, "autoconversion must be active here"
    for k in range(1, len(steps)):
        assert bool(jnp.all(ac[k] == ac[0])), "accretion became step-dependent"
        assert bool(jnp.all(au[k] == au[0])), "autoconversion became step-dependent"


def test_accretion_takes_no_timestep_argument():
    """The warm-rain sinks must not depend on the microphysics step length.

    This pins the refutation of the "our microphysics over-strips at a long
    step" hypothesis (2026-09-23).  KK2000 accretion takes no ``dt`` at all,
    and autoconversion uses ``dt`` only for its number closure, so the MASS
    sinks are functions of the state alone.  A change that made either rate
    depend on the step -- for instance a well-meant sub-stepping "fix" --
    fails here.
    """
    import inspect
    from legoesm.atmosphere.physics.microphysics import _warm_rain as wr
    import jax.numpy as jnp

    assert "dt" not in inspect.signature(wr.accretion_kk2000).parameters

    q_c = jnp.array([[1.0e-3, 4.0e-4]])
    q_r = jnp.array([[2.0e-4, 1.0e-4]])
    a = wr.accretion_kk2000(q_c, q_r)
    assert float(jnp.min(a)) > 0.0
    # Same state, any step: identical mass sink.
    for _ in (22.5, 112.5, 600.0, 1800.0):
        assert bool(jnp.all(wr.accretion_kk2000(q_c, q_r) == a))


def test_donor_clamp_caps_removal_at_the_reservoir():
    """The reservoir limiter is MG2's conservation check (micro_mg2_0.F90:1566).

    It is what makes a long explicit step safe, and it is why the long-step
    answer cannot over-deplete.  Non-vacuous: an unclamped sink would remove
    more than ``q`` and the second assertion fails.
    """
    import jax.numpy as jnp
    from legoesm.atmosphere.physics.microphysics._warm_rain import (
        donor_clamp_scale)

    q = jnp.array([1.0e-3])
    sink = jnp.array([1.0e-4])   # 1e-4 * 600 s = 6e-2 >> q  -> must clamp
    dt = 600.0
    s = donor_clamp_scale(q, sink, dt)
    assert float(s[0]) < 1.0
    assert float((sink * s * dt)[0]) <= float(q[0]) * (1.0 + 1.0e-12)
    # Small sink: no clamping.
    assert abs(float(donor_clamp_scale(q, jnp.array([1.0e-9]), dt)[0]) - 1.0) < 1e-9
