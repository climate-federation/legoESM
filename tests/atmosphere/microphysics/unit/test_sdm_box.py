"""Unit tests for the composed persistent SDM box driver (run_box / box_step)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from legoesm import constants
from legoesm.thermo import saturation_vapor_pressure
from legoesm.atmosphere.physics.microphysics.sdm import (
    BoxState,
    SDMConfig,
    SuperDropletState,
    box_step,
    box_water,
    run_box,
)


def _box(n_sd, radius, multiplicity, T=283.0, p=9.0e4, q_v=0.0, seed=0):
    o = jnp.ones((n_sd,))
    droplets = SuperDropletState(
        multiplicity=o * multiplicity, radius=o * radius,
        solute_mass=o * 0.0, active=o)
    return BoxState(droplets=droplets, T=jnp.asarray(T), p=jnp.asarray(p),
                    q_v=jnp.asarray(q_v), key=random.PRNGKey(seed))


def test_run_box_collision_conserves_mass_and_reduces_number():
    cfg = SDMConfig(collision_kernel="golovin", golovin_b=1.5e3)
    V = 1.0
    box0 = _box(1024, radius=1.4e-5, multiplicity=1.0e9 * V / 1024)
    q_l0, _, N0 = (float(v) for v in box_water(box0, V))   # TRUE initial baseline
    final, hist = run_box(box0, V, dt=1.0, n_steps=50, cfg=cfg,
                          do_condensation=False, do_coalescence=True)
    q_l = np.asarray(hist["q_l"])
    N_hist = np.asarray(hist["N"])
    # collision conserves Σξm (hence q_l, M_air fixed) from the TRUE initial; the
    # first stored step must already match it (catches a first-step bug).
    assert abs(q_l[0] - q_l0) / q_l0 < 1e-9
    assert np.max(np.abs(q_l - q_l0)) / q_l0 < 1e-9
    assert N_hist[-1] < N0
    assert np.all(np.diff(N_hist) <= 1e-3 * N0)   # monotone non-increasing
    assert np.all(hist["finite"] > 0.5)           # finite over the FULL trajectory


def test_run_box_condensation_only_conserves_total_water_and_grows():
    cfg = SDMConfig(include_curvature=False, include_solute=False)
    V = 1.0
    T0, p0 = 283.0, 9.0e4
    e_s = float(saturation_vapor_pressure(jnp.asarray(T0)))
    e = 1.02 * e_s                                  # S0 = 1.02 supersaturated
    q_v0 = constants.epsilon * e / (p0 - e)
    box0 = _box(256, radius=5.0e-6, multiplicity=1.0e8 * V / 256,
                T=T0, p=p0, q_v=q_v0)
    r0 = float(box0.droplets.radius[0])
    q_t0 = float(box0.q_v) + float(box_water(box0, V)[0])   # TRUE initial q_t
    final, hist = run_box(box0, V, dt=0.5, n_steps=200, cfg=cfg,
                          do_condensation=True, do_coalescence=False)
    q_t = np.asarray(hist["q_t"])
    # total water conserved from the TRUE initial (fixed dry-air M_air); droplets
    # grew; vapor fell; T rose; finite over the full trajectory.
    assert abs(q_t[0] - q_t0) / q_t0 < 1e-9
    assert np.max(np.abs(q_t - q_t0)) / q_t0 < 1e-9
    assert float(final.droplets.radius[0]) > r0
    assert float(final.q_v) < q_v0
    assert float(final.T) > T0                      # latent heating
    assert np.all(hist["finite"] > 0.5)


def test_box_step_deterministic_and_jit():
    cfg = SDMConfig(collision_kernel="golovin")
    V = 1.0
    box0 = _box(128, 1.4e-5, 5.0e5, seed=3)
    M_air = float(box0.p) / (constants.R_d * float(box0.T)) * V
    a = box_step(box0, V, M_air, 1.0, cfg)
    b = box_step(box0, V, M_air, 1.0, cfg)
    assert jnp.array_equal(a.droplets.multiplicity, b.droplets.multiplicity)
    run = jax.jit(lambda bx: run_box(bx, V, 1.0, 10, cfg)[0])
    fin = run(box0)
    assert jnp.isfinite(fin.T) and jnp.all(jnp.isfinite(fin.droplets.radius))
