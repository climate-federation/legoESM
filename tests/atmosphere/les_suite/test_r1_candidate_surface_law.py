"""Round-1 candidates use the stress a column with a surface law passes in (CloudBench,
PI 2026-11-11), and their own surface law when it passes NaN (LES runs)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.atmosphere.les_suite import candidate as cmod
from legoesm.atmosphere.physics._shared import exner_function

from legoesm import constants

jax.config.update("jax_enable_x64", True)

_R1 = (Path(__file__).resolve().parents[3] / "scripts" / "validate" / "les_suite"
       / "cosci_candidates" / "round1")
_UW, _VW = -0.09, 0.02
N = 16


def _column(uw_s, vw_s, z0_m=1e-3, sfc_drag_most=1.0):
    z_half = jnp.linspace(0.0, 1600.0, N + 1)
    z_full = 0.5 * (z_half[1:] + z_half[:-1])
    rho_full = 1.2 * jnp.exp(-z_full / 8000.0)
    p_half = 1.0e5 * jnp.exp(-z_half / 8000.0)
    p_full = 1.0e5 * jnp.exp(-z_full / 8000.0)
    return cmod.column_inputs(
        z_full=z_full, z_half=z_half, rho_full=rho_full,
        rho_half=cmod.half_level_average(rho_full, rho_full[0]),
        mass=(p_half[:-1] - p_half[1:]) / constants.g, exner=exner_function(p_full),
        exner_half=exner_function(p_half), p_full=p_full, p_half=p_half,
        theta_l=300.0 + 0.004 * jnp.maximum(z_full - 600.0, 0.0),
        q_t=0.01 - 2e-6 * z_full, u=5.0 + 0.002 * z_full, v=jnp.sin(z_full / 300.0),
        w_theta_s=0.05, w_qt_s=1e-4, uw_s=uw_s, vw_s=vw_s, dt=600.0,
        z0_m=z0_m, z0h_m=z0_m, sfc_drag_most=sfc_drag_most)


def _run(name, inp, params=None):
    """Fluxes and new aux of one call, from a fresh load of the candidate file."""
    cand = cmod.load_candidate(_R1 / f"{name}.py")
    return cand.fluxes(cand.default_params() if params is None else params, inp,
                       cand.init_aux(inp))


def _leaves(tree):
    return [np.asarray(x) for x in jax.tree_util.tree_leaves(tree)]


EXPLICIT = ["r1_c1_wave_kernel", "r1_c1_wave_kernel_mem", "r1_c2_third_moment",
            "r1_l2_pc_edmf", "r1_c3_transilient", "r1_codex_pressure_turning_trans",
            "r1_c4_turning_plumes"]
#: what enters each candidate ONLY through its own surface law: roughness, except for
#: the codex candidate, whose mixing lengths also read z0 (its stability function then)
_OWN_LAW_KNOB = {"r1_codex_pressure_turning_trans": "psi_m"}


@pytest.mark.parametrize("name", EXPLICIT)
def test_candidate_uses_the_passed_stress(name):
    _fa, (uw, vw), _aux = _run(name, _column(_UW, _VW))
    np.testing.assert_allclose([float(uw[0]), float(vw[0])], [_UW, _VW], rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("name", EXPLICIT)
def test_law_column_ignores_the_candidates_own_surface_layer(name, monkeypatch):
    """On a law column every flux and the new aux must not depend on what enters only
    the candidate's own surface law: any own u*, stress or drag still used downstream
    (plume launch, zeta, TKE production, cold start, implicit drag) would show here."""
    a = _leaves(_run(name, _column(_UW, _VW, z0_m=1e-3)))
    if _OWN_LAW_KNOB.get(name) == "psi_m":
        import legoesm.core.bulk_flux as bf
        monkeypatch.setattr(bf, "psi_m", lambda zeta, *a, **k: 0.0 * zeta + 0.3)
        b = _leaves(_run(name, _column(_UW, _VW, z0_m=1e-3)))
    else:
        b = _leaves(_run(name, _column(_UW, _VW, z0_m=0.1)))
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("name", EXPLICIT)
def test_nan_stress_keeps_the_own_law_bit_identically(name, monkeypatch):
    """With NaN passed, the result equals that of the candidate whose stress
    selection is replaced by its own values (the round-1 behaviour)."""
    inp = _column(jnp.nan, jnp.nan)
    got = _leaves(_run(name, inp))
    monkeypatch.setattr(cmod, "surface_stress", lambda _x, uw, vw, us: (uw, vw, us))
    ref = _leaves(_run(name, inp))
    for x, y in zip(got, ref, strict=True):
        np.testing.assert_array_equal(x, y)
    _fa, (uw, _vw), _aux = _run(name, inp)
    assert abs(float(uw[0]) - _UW) > 1e-3 * abs(_UW)
    # control: the patch bites (with it, a passed stress is no longer used)
    _fa, (uw, _vw), _aux = _run(name, _column(_UW, _VW))
    assert abs(float(uw[0]) - _UW) > 1e-3 * abs(_UW)


@pytest.mark.parametrize("name", EXPLICIT)
def test_gradient_through_a_law_column_is_finite(name):
    cand = cmod.load_candidate(_R1 / f"{name}.py")
    inp = _column(_UW, _VW, sfc_drag_most=jnp.nan)   # as CloudBench: no wall-drag record
    p0 = cand.default_params()

    def loss(p):
        (wt, wq), (uw, vw), _ = cand.fluxes(p, inp, cand.init_aux(inp))
        return jnp.sum(wt ** 2) + jnp.sum(uw[1:] ** 2 + vw[1:] ** 2)
    g = jax.grad(loss)(p0)
    assert all(np.isfinite(float(v)) for v in g.values())


def test_l2_internal_u_star_is_that_of_the_passed_stress():
    spec = importlib.util.spec_from_file_location("_r1_l2", _R1 / "r1_l2_pc_edmf.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cand = cmod.load_candidate(_R1 / "r1_l2_pc_edmf.py")
    inp = _column(_UW, _VW)
    d = mod.diagnostics(cand.default_params(), inp, cand.init_aux(inp))
    np.testing.assert_allclose(float(d["ustar"]), (_UW ** 2 + _VW ** 2) ** 0.25, rtol=1e-12)


def test_surface_stress_helper_selects_and_has_finite_gradient_at_zero():
    inp = _column(_UW, _VW)
    uw, vw, us = cmod.surface_stress(inp, 1.0, 2.0, 3.0)
    assert (float(uw), float(vw)) == (_UW, _VW)
    np.testing.assert_allclose(float(us), (_UW ** 2 + _VW ** 2) ** 0.25, rtol=1e-14)
    assert tuple(float(x) for x in cmod.surface_stress(_column(jnp.nan, jnp.nan),
                                                       1.0, 2.0, 3.0)) == (1.0, 2.0, 3.0)

    def u_star(s):
        return cmod.surface_stress(_column(s, 0.0 * s), 1.0, 2.0, 3.0)[2]
    assert float(u_star(0.0)) == 0.0
    assert np.isfinite(float(jax.grad(u_star)(0.0)))
    # u* = |tau|^(1/2): d u*/d tau = -0.5 |tau|^(-1/2) for tau < 0
    np.testing.assert_allclose(float(jax.grad(u_star)(_UW)), -0.5 * abs(_UW) ** -0.5,
                               rtol=1e-12)
    # NaN passed: the own u* carries its gradient through the selection
    own = jax.grad(lambda o: cmod.surface_stress(_column(jnp.nan, jnp.nan), o, o, o)[2])(3.0)
    assert float(own) == 1.0
