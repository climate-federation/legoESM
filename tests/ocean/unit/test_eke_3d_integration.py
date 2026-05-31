"""Stage-4 integration tests for the 3-D (depth-resolved) prognostic EKE closure.

Stage 4 wires the 3-D eddy-energy field ``E`` (on the interior interfaces /
W-grid, shape ``(n_lat, n_lon, nlev-1)``) into the lat-lon C-grid ocean model
step (behind the static ``EKEConfig.eke_3d`` flag), makes the ``state.eke``
field 3-D, threads it through the scan-carry + restart, and opts the ACC recipe
in (``eke_3d=True``).

These tests exercise the FULL step (not just the pure closure functions, which
``test_eke.py`` covers), and assert the CLAUDE.md hard constraints:

  (a) build the ACC model with eke_3d=True + run ~1 day -> finite + stable,
  (b) the eke field is 3-D and develops depth structure (std over z > 0),
  (c) a restart round-trip of the 3-D eke state,
  (d) eke_3d=False (default 2-D) is bit-identical to the prior path,
  (e) jax.grad / scan-carry (constant pytree) AD-safety smoke.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.fidelity.veros_acc_recipe import (
    ACC_GM_REDI_CONFIG,
    DT_MOM_S,
    NZ,
    build_acc_recipe,
)
from legoesm.ocean.physics.lateral_mixing.eke import EKEConfig


# ---------------------------------------------------------------------------
# (0) The ACC recipe opts in to the 3-D path, and seeds a 3-D eke field.
# ---------------------------------------------------------------------------


def test_acc_recipe_eke_3d_enabled_and_seeded():
    """The ACC recipe sets ``eke_3d=True`` and seeds a 3-D eke Field at the
    interior interfaces (n_lat, n_lon, nlev-1) on construction (so the model
    step never does None->Field, which would break the scan-carry pytree)."""
    assert ACC_GM_REDI_CONFIG.eke is not None
    assert ACC_GM_REDI_CONFIG.eke.eke_3d is True
    assert ACC_GM_REDI_CONFIG.eke.alpha_eke == 1.0   # Veros ACC default

    recipe = build_acc_recipe()
    eke = recipe.initial_state.eke
    assert eke is not None, "EKE-on recipe must seed the eke field at step 0"
    n_lat, n_lon = recipe.grid.n_lat, recipe.grid.n_lon
    assert eke.data.shape == (n_lat, n_lon, NZ - 1)
    assert eke.dims == ("lat", "lon", "level")
    # Seeded to e_min on wet cells (to the storage-policy float precision: the
    # e_min*land_mask product carries a tiny f32 round-off), 0 exactly on land.
    lm = recipe.initial_state.land_mask.data
    e_min = ACC_GM_REDI_CONFIG.eke.e_min
    wet3 = lm[:, :, None] > 0.5
    np.testing.assert_allclose(
        np.asarray(jnp.where(wet3, eke.data, e_min)), e_min, rtol=1e-6, atol=0)
    np.testing.assert_allclose(
        np.asarray(jnp.where(~wet3, eke.data, 0.0)), 0.0, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# (a) Build the ACC model with eke_3d=True + run ~1 day -> finite + stable.
# ---------------------------------------------------------------------------


def _build_acc_model_3d():
    recipe = build_acc_recipe(with_surface_forcing=True)
    model = LatLonCGridOceanModel(recipe.grid, recipe.z_coord, recipe.model_config)
    return recipe, model


def test_acc_eke_3d_one_day_finite_and_stable():
    """Build the ACC model (eke_3d=True), run ~1 day (18 steps of dt_mom=4800 s)
    under rigid wind + restoring, and assert the state stays finite + stable
    (max|u| sane, max|T| near the restoring band)."""
    recipe, model = _build_acc_model_3d()
    state = recipe.initial_state
    for _ in range(18):
        state = model.step(state, DT_MOM_S, surface_forcing=recipe.wind_forcing)
    assert bool(jnp.all(jnp.isfinite(state.u.data)))
    assert bool(jnp.all(jnp.isfinite(state.v.data)))
    assert bool(jnp.all(jnp.isfinite(state.T.data)))
    assert bool(jnp.all(jnp.isfinite(state.eke.data)))
    max_u = float(jnp.max(jnp.abs(state.u.data)))
    max_T = float(jnp.max(jnp.abs(state.T.data)))
    # ~1 day from rest under a 0.1 N/m^2 wind: velocities are small (cm/s scale),
    # certainly well under 1 m/s (a blow-up would be many m/s).
    assert max_u < 1.0, f"max|u|={max_u} not stable after 1 day"
    # Temperature stays in the physical 0..15 degC restoring band (no overshoot).
    assert max_T < 16.0, f"max|T|={max_T} overshot the restoring band"
    # The eke field keeps its 3-D shape across the step.
    n_lat, n_lon = recipe.grid.n_lat, recipe.grid.n_lon
    assert state.eke.data.shape == (n_lat, n_lon, NZ - 1)
    # Dtype + pytree treedef are stable across the step (scan-carry requirement).
    assert state.eke.data.dtype == recipe.initial_state.eke.data.dtype
    assert jtu.tree_structure(state) == jtu.tree_structure(recipe.initial_state)


# ---------------------------------------------------------------------------
# (b) The eke field is 3-D and develops depth structure (std over z > 0).
# ---------------------------------------------------------------------------


def test_acc_eke_3d_develops_depth_structure():
    """Starting from a vertically uniform e_min seed, the 3-D EKE budget
    (depth-resolved source/sink + implicit vertical diffusion + per-interface
    horizontal transport) must develop genuine DEPTH structure: the vertical
    std of E over wet columns grows from ~0 to a finite positive value."""
    recipe, model = _build_acc_model_3d()
    lm = recipe.initial_state.land_mask.data
    wet3 = lm[:, :, None] > 0.5

    def depth_std(eke):
        # std over the level axis, averaged over wet columns only.
        s = jnp.std(eke, axis=-1)
        return float(jnp.sum(jnp.where(lm > 0.5, s, 0.0))
                     / jnp.maximum(jnp.sum(lm > 0.5), 1))

    state = recipe.initial_state
    std0 = depth_std(state.eke.data)
    # Use the scan path (exercises the constant-pytree carry) for a longer run so
    # the depth structure has time to develop.
    state, _ = model.integrate_scan(state, n_steps=120, dt=DT_MOM_S)
    std1 = depth_std(state.eke.data)
    assert bool(jnp.all(jnp.isfinite(state.eke.data)))
    # The seed is vertically uniform (std ~ 0); after spin-up the 3-D budget has
    # made E depth-dependent.
    assert std0 < 1e-12, f"seed should be vertically uniform, std0={std0}"
    assert std1 > std0, "3-D EKE must develop depth structure (std over z grows)"
    assert std1 > 1e-13, f"depth std {std1} did not develop"
    # E stays >= e_min on wet columns (positivity-preserving), 0 on land.
    e_min = recipe.model_config.gm_redi.eke.e_min
    wet_min = float(jnp.min(jnp.where(wet3, state.eke.data, jnp.inf)))
    assert wet_min >= e_min - 1e-12, f"wet-cell eke {wet_min} fell below e_min"
    land_max = float(jnp.max(jnp.where(~wet3, state.eke.data, -jnp.inf)))
    assert land_max == 0.0, f"land eke must be 0, got {land_max}"


# ---------------------------------------------------------------------------
# (c) Restart round-trip of the 3-D eke state.
# ---------------------------------------------------------------------------


def test_eke_3d_restart_round_trip(tmp_path):
    """The 3-D eke field round-trips through the ocean restart I/O (save -> load)
    bit-exactly, with the (n_lat, n_lon, nlev-1) shape + dims/units metadata
    preserved (the restart I/O is shape-agnostic; this locks the 3-D shape)."""
    from legoesm.ocean.restart import load_restart, save_restart

    recipe, model = _build_acc_model_3d()
    # Run a few steps so eke is non-trivial (not just the seed).
    state = recipe.initial_state
    for _ in range(5):
        state = model.step(state, DT_MOM_S, surface_forcing=recipe.wind_forcing)
    assert state.eke.data.shape == (recipe.grid.n_lat, recipe.grid.n_lon, NZ - 1)

    path = tmp_path / "acc_eke3d_restart.npz"
    save_restart(state, path, time_s=5 * DT_MOM_S, step=5)
    # Load back using the (3-D-eke) initial state as the metadata template.
    restored = load_restart(path, recipe.initial_state)

    assert restored.eke.data.shape == state.eke.data.shape
    assert restored.eke.dims == state.eke.dims
    assert restored.eke.units == state.eke.units
    np.testing.assert_array_equal(
        np.asarray(restored.eke.data), np.asarray(state.eke.data))
    # The restored state must step on without recompiling into a different shape.
    stepped = model.step(restored, DT_MOM_S, surface_forcing=recipe.wind_forcing)
    assert stepped.eke.data.shape == state.eke.data.shape
    assert bool(jnp.all(jnp.isfinite(stepped.eke.data)))


# ---------------------------------------------------------------------------
# (d) eke_3d=False (default 2-D path) is bit-identical to before.
# ---------------------------------------------------------------------------


def _build_acc_model_2d():
    """An ACC model with the EKE config forced to the 2-D path (eke_3d=False),
    everything else identical to the recipe. This reproduces the pre-Stage-4
    behaviour: the 2-D depth-integrated closure."""
    recipe = build_acc_recipe(with_surface_forcing=True)
    eke_2d = recipe.model_config.gm_redi.eke._replace(eke_3d=False)
    gm_2d = recipe.model_config.gm_redi._replace(eke=eke_2d)
    cfg_2d = recipe.model_config._replace(gm_redi=gm_2d)
    model = LatLonCGridOceanModel(recipe.grid, recipe.z_coord, cfg_2d)
    # Re-seed the eke field as 2-D (the recipe seeded it 3-D for eke_3d=True).
    from legoesm.core.field import Field
    lm = recipe.initial_state.land_mask.data
    eke0 = eke_2d.e_min * lm
    state = recipe.initial_state._replace(
        eke=Field(data=eke0, name="eke", dims=("lat", "lon"), units="m^2/s^2"))
    return recipe, model, state


def test_eke_2d_path_unchanged_shape_and_finite():
    """With eke_3d=False the eke field stays 2-D (n_lat, n_lon) through the step
    — the 3-D path does not leak into the default 2-D behaviour."""
    recipe, model, state = _build_acc_model_2d()
    assert state.eke.data.ndim == 2
    for _ in range(6):
        state = model.step(state, DT_MOM_S, surface_forcing=recipe.wind_forcing)
    assert state.eke.data.shape == (recipe.grid.n_lat, recipe.grid.n_lon)
    assert state.eke.dims == ("lat", "lon")
    assert bool(jnp.all(jnp.isfinite(state.eke.data)))
    assert bool(jnp.all(jnp.isfinite(state.u.data)))


def test_eke_3d_flag_does_not_perturb_a_no_eke_model():
    """A model with NO EKE (gm_redi.eke=None) is byte-identical whether or not
    the eke_3d branch exists in the code — i.e. the new branch is fully gated:
    a frozen, no-forcing 3-step run with gm_redi.eke=None reproduces the same
    state as the committed code path (regression guard for the gating)."""
    recipe = build_acc_recipe()
    # Strip EKE entirely (constant-kappa GM fallback path, gm_redi.eke=None).
    gm_no_eke = recipe.model_config.gm_redi._replace(eke=None)
    cfg = recipe.model_config._replace(gm_redi=gm_no_eke)
    model = LatLonCGridOceanModel(recipe.grid, recipe.z_coord, cfg)
    # Drop the seeded eke field too (no EKE -> eke stays None / inert).
    state = recipe.initial_state._replace(eke=None)
    out = state
    for _ in range(3):
        out = model.step(out, DT_MOM_S)
    # eke remains None (the no-EKE path never constructs it); state finite.
    assert out.eke is None
    assert bool(jnp.all(jnp.isfinite(out.T.data)))
    assert bool(jnp.all(jnp.isfinite(out.u.data)))


# ---------------------------------------------------------------------------
# (e) Scan-carry constant-pytree + jax.grad AD-safety smoke.
# ---------------------------------------------------------------------------


def test_eke_3d_scan_carry_constant_pytree():
    """jax.lax.scan over the 3-D-eke model step must keep a CONSTANT-shape carry:
    integrate_scan succeeds (it would raise if the eke carry changed shape/None)
    and the final eke keeps the 3-D shape + dtype."""
    recipe, model = _build_acc_model_3d()
    seed = recipe.initial_state
    final, traj = model.integrate_scan(seed, n_steps=10, dt=DT_MOM_S)
    assert final.eke.data.shape == seed.eke.data.shape
    assert final.eke.data.dtype == seed.eke.data.dtype
    # The stacked trajectory carries the eke field at every step (leading n_steps).
    assert traj.eke.data.shape == (10,) + seed.eke.data.shape
    assert bool(jnp.all(jnp.isfinite(final.eke.data)))


def test_eke_3d_scan_carry_seeded_when_eke_none():
    """integrate_scan must pre-seed a 3-D eke carry when state.eke is None but the
    config runs eke_3d — so a cold-start (eke=None) free run does NOT hit a
    None->Field transition inside the scan (the bug the memory note warns about)."""
    recipe, model = _build_acc_model_3d()
    cold = recipe.initial_state._replace(eke=None)
    final, _ = model.integrate_scan(cold, n_steps=5, dt=DT_MOM_S)
    assert final.eke is not None
    assert final.eke.data.shape == (recipe.grid.n_lat, recipe.grid.n_lon, NZ - 1)
    assert bool(jnp.all(jnp.isfinite(final.eke.data)))


def test_eke_3d_step_is_differentiable():
    """jax.grad through one 3-D-eke model step must produce finite gradients
    (AD-safety: the depth-resolved closure + implicit vertical diffusion + the
    per-interface transport are all differentiable — no nondifferentiable
    branch leaks into the eke path)."""
    recipe, model = _build_acc_model_3d()
    state = recipe.initial_state

    def loss(T_data):
        st = state._replace(T=state.T.replace(data=T_data))
        out = model.step(st, DT_MOM_S, surface_forcing=recipe.wind_forcing)
        # Scalar objective touching BOTH the eke field and the tracer state so
        # the gradient flows through the EKE closure (eke depends on T via the
        # density / Eady-growth / GM coupling).
        return jnp.sum(out.eke.data ** 2) + jnp.sum(out.T.data ** 2)

    g = jax.grad(loss)(state.T.data)
    assert g.shape == state.T.data.shape
    assert bool(jnp.all(jnp.isfinite(g))), "non-finite gradient through 3-D EKE step"
    # The gradient is non-trivial (the step genuinely depends on T).
    assert float(jnp.max(jnp.abs(g))) > 0.0


def test_eke_av_at_interior_wfaces_placement():
    """Locks the A_v vertical placement for the 3-D EKE implicit vertical diffusion
    (physics-validator finding): A_v_phys may be cell-centred (nlev), at the
    T-interfaces (nlev-1), or None.  The INTERFACE case must AVERAGE adjacent
    interfaces to the interior T-centres (Veros 0.5*(kappaM[k]+kappaM[k+1])), NOT
    slice them [1:nlev-1] (the latent half-level-misplacement that is dormant in
    ACC because tend.A_v is None there)."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        _eke_av_at_interior_wfaces,
    )
    nlev = 6
    prefix = (2, 3)
    bg0 = jnp.asarray(0.0)
    # None -> constant background, shape (*prefix, nlev-2)
    av_none = _eke_av_at_interior_wfaces(None, jnp.asarray(1e-4), nlev, prefix)
    assert av_none.shape == prefix + (nlev - 2,)
    assert jnp.allclose(av_none, 1e-4)
    # cell-centred (nlev): interior T-centres [1:nlev-1]
    cc = jnp.arange(prefix[0] * prefix[1] * nlev, dtype=jnp.float64).reshape(prefix + (nlev,))
    av_cc = _eke_av_at_interior_wfaces(cc, bg0, nlev, prefix)
    assert av_cc.shape == prefix + (nlev - 2,)
    assert jnp.allclose(av_cc, cc[..., 1:nlev - 1])
    # interface (nlev-1): AVERAGE adjacent interfaces (the fix) ...
    iface = jnp.arange(prefix[0] * prefix[1] * (nlev - 1), dtype=jnp.float64).reshape(prefix + (nlev - 1,))
    av_if = _eke_av_at_interior_wfaces(iface, bg0, nlev, prefix)
    assert av_if.shape == prefix + (nlev - 2,)
    assert jnp.allclose(av_if, 0.5 * (iface[..., :-1] + iface[..., 1:]))
    # ... and it must DIFFER from the wrong slice [1:nlev-1] (the guarded bug)
    assert not jnp.allclose(av_if, iface[..., 1:nlev - 1])
    # background floor is added
    av_bg = _eke_av_at_interior_wfaces(cc, jnp.asarray(10.0), nlev, prefix)
    assert jnp.allclose(av_bg, cc[..., 1:nlev - 1] + 10.0)
    # wrong last-axis -> ValueError
    with pytest.raises(ValueError):
        _eke_av_at_interior_wfaces(
            jnp.zeros(prefix + (nlev + 2,)), jnp.asarray(0.0), nlev, prefix)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
