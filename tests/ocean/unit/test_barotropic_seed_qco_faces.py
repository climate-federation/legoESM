"""The barotropic loop-entry seed is NEMO's e3u_0-weighted column mean.

NEMO seeds its external loop with ``puu_b(Kmm)`` (``dynspg_ts.F90:487``),
which ``stprk3_stg.F90:439-446`` has imposed on ``uu`` with the ``e3u_0/hu_0``
weights, and whose live face thickness is ``e3u(Kmm) = e3u_0*(1+r3u(Kmm))``
(``domzgr_substitute.h90``, ``domqco.F90:219-222``).  On the certified
OVERFLOW-zps oracle ``uu_b(Kbb)`` equals that mean of its own ``uu(Kbb)`` to
``2e-17`` at kt=2..4.  The previous card-mesh seed took the per-level MIN of
the two STRETCHED T-cell thicknesses rescaled by a column-depth ratio, which
mis-normalises every face whose two columns have different reference depths
(the shelf break): a uniform velocity came back scaled by
``(1+r3t_e)/(1+r3t_w)``.  Both facts are pinned here with the reverted rule
shown to break the assertion.
"""

import jax.numpy as jnp
import numpy as np

from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean.dynamics import barotropic_latlon_cgrid as barotropic
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    compute_face_masks_3d,
    min_cell_to_uface,
)
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.fidelity.nemo_testcase_recipe import build_overflow_zps_card
from legoesm.ocean.vertical import compute_layer_thickness

_ENTRY = {}


def _overflow_entry_after_one_step():
    """OVERFLOW-zps kt=2 entry: a moving state over stepped bathymetry."""
    if not _ENTRY:
        set_policy(PrecisionPolicy.fp64())
        card = build_overflow_zps_card()
        model = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
        _ENTRY["card"] = card
        _ENTRY["entry"] = model.step(card.recipe.initial_state, dt=card.dt_s)
    return _ENTRY["card"], _ENTRY["entry"]


def _seed(card, entry, **kw):
    cfg = card.recipe.model_config
    eta = jnp.asarray(entry.eta.data)
    h_k = compute_layer_thickness(
        eta, entry.H_bathy.data, card.recipe.z_coord,
        min_water_column_m=cfg.min_water_column_m)
    U_bar, _ = barotropic._depth_average_to_faces(
        jnp.asarray(entry.u.data), jnp.asarray(entry.v.data), h_k,
        jnp.asarray(cfg.min_water_column_m), jnp.asarray(entry.land_mask.data),
        jnp.asarray(entry.u_mask.data), jnp.asarray(entry.v_mask.data),
        card.recipe.grid,
        seed_face_depth=cfg.barotropic.barotropic_seed_face_depth,
        seed_evaluation=cfg.barotropic.barotropic_seed_evaluation,
        eta_dyn=eta, H_bathy=jnp.asarray(entry.H_bathy.data),
        area=jnp.asarray(card.recipe.grid.area), z_coord=card.recipe.z_coord,
        **kw)
    return np.asarray(U_bar)[:, 1:]


def _nemo_reference_mean(card, entry):
    """stprk3_stg.F90:440: SUM(e3u_0*uu)*r1_hu_0 on NEMO's native east faces."""
    cfg = card.recipe.model_config
    h_ref = compute_layer_thickness(
        jnp.zeros_like(jnp.asarray(entry.eta.data)), entry.H_bathy.data,
        card.recipe.z_coord, min_water_column_m=cfg.min_water_column_m)
    u_mask3, _ = compute_face_masks_3d(card.recipe.z_coord.is_active,
                                       card.recipe.grid)
    e3u_0 = np.asarray(min_cell_to_uface(h_ref))[:, 1:, :]
    umask = np.asarray(u_mask3, dtype=np.float64)[:, 1:, :]
    hu_0 = np.sum(e3u_0 * umask, axis=-1)
    u = np.asarray(entry.u.data)[:, 1:, :]
    wet = hu_0 > 0.0
    return np.where(wet, np.sum(e3u_0 * u * umask, axis=-1)
                    / np.where(wet, hu_0, 1.0), 0.0), wet


def test_card_seed_is_the_nemo_e3u_0_column_mean():
    card, entry = _overflow_entry_after_one_step()
    assert card.recipe.model_config.barotropic.barotropic_seed_evaluation == "nemo_literal"
    seed = _seed(card, entry)
    reference, wet = _nemo_reference_mean(card, entry)
    assert seed.dtype == np.float64
    scale = float(np.max(np.abs(reference[wet])))
    assert scale > 1.0e-6, "degenerate entry"
    assert float(np.max(np.abs(seed - reference)[wet])) <= 1.0e-15 * scale


def test_reverted_min_rule_rescale_mis_normalises_the_shelf_break():
    """Non-vacuity: the min-of-stretched-cells rescale is a DIFFERENT number
    wherever the two columns' reference depths differ, and only there."""
    card, entry = _overflow_entry_after_one_step()
    seed = _seed(card, entry)
    legacy = _seed(card, entry, legacy_seed_min_rule_faces=True)
    reference, wet = _nemo_reference_mean(card, entry)
    scale = float(np.max(np.abs(reference[wet])))
    diff = np.abs(legacy - seed)
    assert float(np.max(diff[wet])) > 1.0e-9 * scale
    h_ref = np.asarray(compute_layer_thickness(
        np.zeros_like(np.asarray(entry.eta.data)), entry.H_bathy.data,
        card.recipe.z_coord,
        min_water_column_m=card.recipe.model_config.min_water_column_m))
    column_depth = np.sum(h_ref, axis=-1)
    step_face = (np.abs(column_depth - np.roll(column_depth, -1, axis=1))
                 > 1.0e-9)[:, :]
    flat = wet & ~step_face
    assert float(np.max(diff[flat])) <= 1.0e-15 * scale
