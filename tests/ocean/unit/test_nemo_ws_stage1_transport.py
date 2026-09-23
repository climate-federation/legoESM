"""Stage 1 of the WS-RK3 momentum ladder advects with NEMO's Kmm transport.

``stprk3_stg.F90:265-275`` builds ONE ``zFu = e2u*e3u(Kmm)*(uu(Kmm) + zub)``
per stage with ``zub = un_adv/hu(Kmm) - uu_b(Kmm)`` and hands it to stage 1's
``dyn_adv`` at ``:315`` exactly as to stages 2-3 at ``:333`` (isomorphism row
S-21).  legoESM's stage-1 RHS is the step-entry ``tendencies()`` result, which
ran before the external solve produced ``un_adv``; the landed fix adds the
transport's advection as the difference of the stage helper with and without
it.  The private ``momentum_transport_reconcile`` hook is the one-variable
control (NEMO has no such switch).
"""

import numpy as np

import legoesm.ocean.dynamics.ocean_model_latlon_cgrid as model_module
from legoesm.core.precision import PrecisionPolicy, set_policy

from tests.ocean.unit.test_nemo_ws_tracer_rk3 import (
    _lock_model,
    _tilted_entry_after_one_step,
)


def _stage1(entry, dt, **hooks):
    kw = dict(expose_momentum_stage=1)
    kw.update(hooks)
    return _lock_model(model_module._NEMOWSRK3TestHooks(**kw)).step(entry, dt=dt)


def test_stage1_momentum_carries_the_zub_transport_and_is_baroclinic():
    set_policy(PrecisionPolicy.fp64())
    card, entry = _tilted_entry_after_one_step()
    with_zub = _stage1(entry, card.dt_s)
    without = _stage1(entry, card.dt_s, momentum_transport_reconcile=False)
    du = np.asarray(with_zub.u.data) - np.asarray(without.u.data)
    assert du.dtype == np.float64
    scale = float(np.max(np.abs(np.asarray(with_zub.u.data))))
    # Live from a moving entry (reverting the fix makes both arms identical).
    assert float(np.max(np.abs(du))) > 1.0e-9 * scale
    # Both arms impose the same stage depth mean (stprk3_stg.F90:439-446), so
    # the transport's effect is purely baroclinic: the e3u_0-weighted column
    # mean of the difference vanishes to roundoff.
    from legoesm.ocean.dynamics.latlon_cgrid_operators import min_cell_to_uface
    from legoesm.ocean.vertical import compute_layer_thickness
    h_ref = compute_layer_thickness(
        np.zeros_like(np.asarray(entry.eta.data)), entry.H_bathy.data,
        card.recipe.z_coord,
        min_water_column_m=card.recipe.model_config.min_water_column_m)
    e3u_0 = np.asarray(min_cell_to_uface(h_ref))
    column = np.sum(e3u_0 * du, axis=-1) / np.maximum(np.sum(e3u_0, axis=-1), 1.0)
    assert float(np.max(np.abs(column))) < 1.0e-14 * scale


def test_stage1_transport_is_inert_from_rest():
    """From rest the transport multiplies a zero advected velocity."""
    set_policy(PrecisionPolicy.fp64())
    card, _ = _tilted_entry_after_one_step()
    rest = card.recipe.initial_state
    a = _stage1(rest, card.dt_s)
    b = _stage1(rest, card.dt_s, momentum_transport_reconcile=False)
    np.testing.assert_array_equal(np.asarray(a.u.data), np.asarray(b.u.data))
