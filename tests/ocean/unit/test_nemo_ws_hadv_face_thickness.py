"""The WS-RK3 flux-form momentum advection consumes NEMO's ``e3u(Kmm)``.

``dynadv_up3.F90:160,205-207`` builds ``zFu = e2u*e3u(Kmm)*uu`` and divides
the flux divergence by ``e3u(Kmm) = e3u_0*(1+r3u(Kmm))``
(``domzgr_substitute.h90:127``, ``domqco.F90:219-220``); under WS-RK3 the
stage transport at ``stprk3_stg.F90:273`` carries the same thickness.
legoESM's ``tendencies()`` used to rebuild its own ``h_u`` as the min of the
two STRETCHED T thicknesses, which differs from NEMO's rule by
``0.5*|ssh_W - ssh_E|/hu_0`` -- the measured owner of the OVERFLOW-zps kt=2
stage-3 ``u`` remainder and of the ``slow_u`` debt.  The stage program now
hands the shared kernel's pair to every ``tendencies()`` call it makes; the
private ``legacy_hadv_min_face_thickness`` hook is the one-variable control
(NEMO has no such switch).
"""

import jax.numpy as jnp
import numpy as np

import legoesm.ocean.dynamics.ocean_model_latlon_cgrid as model_module
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    compute_face_masks_3d,
    min_cell_to_uface,
    min_cell_to_vface,
)
from legoesm.ocean.vertical import compute_layer_thickness

from tests.ocean.unit.test_nemo_ws_tracer_rk3 import (
    _lock_model,
    _tilted_entry_after_one_step,
)


def _step(entry, dt, **hooks):
    return _lock_model(model_module._NEMOWSRK3TestHooks(**hooks)).step(entry, dt=dt)


def test_stage_hadv_thickness_is_live_on_a_tilted_entry_and_inert_from_rest():
    set_policy(PrecisionPolicy.fp64())
    card, entry = _tilted_entry_after_one_step()
    faithful = _step(entry, card.dt_s)
    legacy = _step(entry, card.dt_s, legacy_hadv_min_face_thickness=True)
    du = np.asarray(faithful.u.data) - np.asarray(legacy.u.data)
    assert du.dtype == np.float64
    scale = float(np.max(np.abs(np.asarray(faithful.u.data))))
    # Live where the entry ssh differs across a face (reverting the fix makes
    # both arms identical).
    assert float(np.max(np.abs(du))) > 1.0e-12 * scale
    # From the card's rest state (eta = 0) the min of the reference
    # thicknesses IS e3u_0 = e3u(Kbb): the two arms are bit-identical.
    rest = card.recipe.initial_state
    a = _step(rest, card.dt_s)
    b = _step(rest, card.dt_s, legacy_hadv_min_face_thickness=True)
    np.testing.assert_array_equal(np.asarray(a.u.data), np.asarray(b.u.data))
    np.testing.assert_array_equal(np.asarray(a.T.data), np.asarray(b.T.data))


def test_tendencies_face_thickness_kwarg_defaults_to_the_min_rule_bit_for_bit():
    """``momentum_flux_face_thickness=None`` is the historical operator; the
    model's own min-rule pair passed explicitly reproduces it bit for bit,
    and NEMO's ``e3u(Kmm)`` pair moves it on a tilted ssh."""
    set_policy(PrecisionPolicy.fp64())
    card, entry = _tilted_entry_after_one_step()
    model = _lock_model()
    cfg, z, grid = card.recipe.model_config, card.recipe.z_coord, card.recipe.grid
    h_k = compute_layer_thickness(entry.eta.data, entry.H_bathy.data, z,
                                  min_water_column_m=cfg.min_water_column_m)
    own = (min_cell_to_uface(h_k), min_cell_to_vface(h_k, grid))
    base = model.tendencies(entry, dt=card.dt_s, momentum_only=True)
    same = model.tendencies(entry, dt=card.dt_s, momentum_only=True,
                            momentum_flux_face_thickness=own)
    np.testing.assert_array_equal(np.asarray(same.du_dt.data), np.asarray(base.du_dt.data))
    h_ref = compute_layer_thickness(jnp.zeros_like(entry.eta.data), entry.H_bathy.data, z,
                                    min_water_column_m=cfg.min_water_column_m)
    u_mask3, v_mask3 = compute_face_masks_3d(z.is_active, grid)
    nemo_pair = model_module._nemo_ws_qco_stage_faces(
        entry.eta.data, h_ref, u_mask3.astype(h_ref.dtype), v_mask3.astype(h_ref.dtype), grid)[:2]
    nemo = model.tendencies(entry, dt=card.dt_s, momentum_only=True,
                            momentum_flux_face_thickness=nemo_pair)
    assert float(np.max(np.abs(np.asarray(nemo.du_dt.data) - np.asarray(base.du_dt.data)))) > 0.0
