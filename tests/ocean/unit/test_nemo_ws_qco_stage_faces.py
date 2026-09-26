"""NEMO qco stage geometry inside the WS-RK3 identity.

Two source facts are pinned here, each with the reverted behaviour shown to
break the assertion:

* ``dom_qco_r3c_RK3`` (NEMO 5.0.2 ``src/OCE/DOM/domqco.F90:219-222``) builds
  the U-face free-surface ratio as an ``e1e2t``-weighted mean of **ssh**
  divided by ``hu_0``, and ``domzgr_substitute.h90:127`` makes the stage face
  thickness ``e3u(Kmm) = e3u_0*(1 + r3u(Kmm)*umask)``.  Taking the ``min`` of
  the two stretched T thicknesses instead is a different number whenever the
  free surface is tilted.
* ``stprk3_stg.F90:373-378`` (and ``dynzdf.F90``'s ``key_qco`` branch at
  stage 3) weight every stage velocity update by ``(1+r3u(Kbb))`` /
  ``(1+r3u(Kmm))`` / ``(1+r3u(Kaa))``.
"""

import jax.numpy as jnp
import numpy as np

import legoesm.ocean.dynamics.ocean_model_latlon_cgrid as model_module
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.grids.latlon import ensure_geometry
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    compute_face_masks_3d,
    min_cell_to_uface,
)
from legoesm.ocean.vertical import compute_layer_thickness

from tests.ocean.unit.test_nemo_ws_tracer_rk3 import (
    _lock_model,
    _tilted_entry_after_one_step,
)


def _tilted_geometry():
    """LOCK card + the tilted post-step entry state (non-degenerate eta)."""
    set_policy(PrecisionPolicy.fp64())
    card, entry = _tilted_entry_after_one_step()
    z_coord = card.recipe.z_coord
    cfg = card.recipe.model_config
    h_ref = compute_layer_thickness(
        jnp.zeros_like(entry.eta.data), entry.H_bathy.data, z_coord,
        min_water_column_m=cfg.min_water_column_m)
    u_mask3, v_mask3 = compute_face_masks_3d(z_coord.is_active, card.recipe.grid)
    return (card, entry, h_ref,
            u_mask3.astype(h_ref.dtype), v_mask3.astype(h_ref.dtype))


def _nemo_r3u(eta, hu_0, grid):
    """domqco.F90:219-222 on NEMO's native east faces, computed here."""
    geom = ensure_geometry(grid)
    area_t = np.asarray(geom.area_T, dtype=np.float64)
    area_u = np.asarray((geom.dx_u * geom.dy_u)[:, 1:], dtype=np.float64)
    weighted = area_t * np.asarray(eta, dtype=np.float64)
    numerator = 0.5 * (weighted + np.roll(weighted, -1, axis=1))
    hu_0 = np.asarray(hu_0, dtype=np.float64)
    return np.where(hu_0 > 0.0, numerator / np.where(hu_0 > 0.0, hu_0, 1.0)
                    / area_u, 0.0)


def test_ws_stage_face_thickness_is_nemo_e3u_0_times_one_plus_r3u():
    """The stage face thickness is e3u_0*(1+r3u), NOT min of stretched cells."""
    card, entry, h_ref, u_mask3, v_mask3 = _tilted_geometry()
    eta = entry.eta.data
    assert float(np.abs(np.asarray(eta)).max()) > 1.0e-3, "degenerate eta"

    e3u_0 = np.asarray(min_cell_to_uface(h_ref))[:, 1:, :]
    hu_0 = np.sum(e3u_0 * np.asarray(u_mask3)[:, 1:, :], axis=-1)
    r3u = _nemo_r3u(eta, hu_0, card.recipe.grid)
    expected = e3u_0 * (1.0 + r3u[..., None] * np.asarray(u_mask3)[:, 1:, :])

    h_u, _, one_plus_r3u, _ = model_module._nemo_ws_qco_stage_faces(
        eta, h_ref, u_mask3, v_mask3, card.recipe.grid)
    native = np.asarray(h_u)[:, 1:, :]
    assert native.dtype == np.float64
    np.testing.assert_allclose(native, expected, rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(
        np.asarray(one_plus_r3u)[:, 1:], 1.0 + r3u, rtol=1e-14, atol=0.0)

    # Non-vacuity: the reverted (min-of-stretched) rule is a DIFFERENT number.
    h_stage = compute_layer_thickness(
        eta, entry.H_bathy.data, card.recipe.z_coord,
        min_water_column_m=card.recipe.model_config.min_water_column_m)
    reverted = np.asarray(min_cell_to_uface(h_stage))[:, 1:, :]
    wet = np.asarray(u_mask3)[:, 1:, :] > 0.5
    assert float(np.abs(reverted - expected)[wet].max()) > 1.0e-6


def test_ws_stage_velocity_carries_the_qco_factor():
    """Stage 1 obeys u(Kaa) = (1+r3u(Kbb))/(1+r3u(Kaa)) * (unweighted update).

    Stage 1 has Kmm = Kbb, so the whole update is scaled by one depth-uniform
    ratio.  The stage barotropic correction (stprk3_stg.F90:433-446) then
    removes the depth mean, which commutes with a depth-uniform factor, so
    ``faithful - omitted == (ratio - 1) * baroclinic(omitted)`` EXACTLY.
    """
    card, entry, h_ref, u_mask3, v_mask3 = _tilted_geometry()
    hooks = model_module._NEMOWSRK3TestHooks
    faithful = _lock_model(hooks(expose_momentum_stage=1)).step(
        entry, dt=card.dt_s)
    omitted = _lock_model(hooks(
        expose_momentum_stage=1, omit_stage_qco_factor=True)).step(
        entry, dt=card.dt_s)

    eta_bb = np.asarray(entry.eta.data)
    eta_aa = np.asarray(_lock_model().step(entry, dt=card.dt_s).eta.data)
    eta_13 = eta_bb + (eta_aa - eta_bb) / 3.0
    e3u_0 = np.asarray(min_cell_to_uface(h_ref))[:, 1:, :]
    umask_n = np.asarray(u_mask3)[:, 1:, :]
    hu_0 = np.sum(e3u_0 * umask_n, axis=-1)
    ratio = ((1.0 + _nemo_r3u(eta_bb, hu_0, card.recipe.grid))
             / (1.0 + _nemo_r3u(eta_13, hu_0, card.recipe.grid)))

    u_f = np.asarray(faithful.u.data)[:, 1:, :]
    u_o = np.asarray(omitted.u.data)[:, 1:, :]
    weights = e3u_0 * umask_n
    depth_mean = (np.sum(u_o * weights, axis=-1)
                  / np.maximum(np.sum(weights, axis=-1), np.finfo(float).tiny))
    predicted = (ratio - 1.0)[..., None] * (u_o - depth_mean[..., None])

    wet = umask_n > 0.5
    moved = float(np.abs(u_f - u_o)[wet].max())
    assert moved > 0.0, "the qco stage factor is inert: fix reverted?"
    assert float(np.abs((u_f - u_o) - predicted)[wet].max()) < 0.02 * moved


def test_ws_program_selects_the_up3_branch_by_the_advected_velocity_pair():
    """The WS-RK3 stage program passes NEMO's UP3 selector
    (dynadv_up3.F90:166-170) to every stage RHS; the private
    ``legacy_up3_transport_sign_selector`` hook restores the transport sign.
    On the tilted LOCK entry the barotropic ``zub`` flips the pair sign at
    the front, so the two must differ -- a hook that no longer reaches the
    executing path (or a program that stopped passing the rule) makes them
    bit-identical and fails this test.
    """
    card, entry, h_ref, u_mask3, v_mask3 = _tilted_geometry()
    hooks = model_module._NEMOWSRK3TestHooks
    faithful = _lock_model().step(entry, dt=card.dt_s)
    legacy = _lock_model(hooks(legacy_up3_transport_sign_selector=True)).step(
        entry, dt=card.dt_s)
    wet = np.asarray(u_mask3)[:, 1:, :] > 0.5
    u_f = np.asarray(faithful.u.data)[:, 1:, :]
    u_l = np.asarray(legacy.u.data)[:, 1:, :]
    assert float(np.abs(u_f - u_l)[wet].max()) > 0.0, (
        "the UP3 selector is inert in the WS-RK3 program: fix reverted?")
