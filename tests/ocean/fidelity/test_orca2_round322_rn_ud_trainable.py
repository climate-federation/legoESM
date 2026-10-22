"""Round 322 (DECISION 149): rn_Ud is trainable only when a run names it with a
start value; kappa_Redi is frozen out of rung-10 training.

The rung-10 gradient test needs the round-5 acquisition record and compiles
rung 10's step and its pullback (minutes, tens of GB); it skips without them.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean.fidelity.nemo_testcase_recipe import ORCA2_RUNG10_FROZEN_PARAMS
from legoesm.training.param_collector import (
    build_orca2_rung10_trainable_params,
    build_registry,
    build_trainable_params,
)

pytestmark = pytest.mark.skipif(not jax.config.jax_enable_x64, reason="needs JAX_ENABLE_X64=1")

GM = "ocean.lat.gm_redi"
RN_UD = f"{GM}.kappa_redi_diffusive_velocity"
KAPPA_REDI = f"{GM}.kappa_Redi"
ORCA2_RN_UD = 0.018  # ORCA2_OMIP_L4/EXP00/namelist_cfg:306
RUNG10_DLDLN_RN_UD_REF = -4.5e-5  # PREREG_nemo_testcases_l4_orca2_round322.md P4


def test_rn_ud_spec_bounds_are_kappa_redi_bounds_through_mode_20_aht0():
    reg = {m.qualified_name: m for m in build_registry()}
    ud, kr = reg[RN_UD], reg[KAPPA_REDI]
    assert ud.explicit_only and ud.default is None and not kr.explicit_only
    rn_ld = 200.0e3  # ORCA2 rn_Ld
    # aht0 = 1/2 rn_Ud rn_Ld  ->  rn_Ud = aht0 / (1/2 rn_Ld)
    np.testing.assert_allclose(ud.bounds, [k / (0.5 * rn_ld) for k in kr.bounds], rtol=1e-12)
    assert ud.bounds == (0.0033, 0.03)


@pytest.mark.parametrize("tier", ["core", "extended", "aggressive"])
def test_tier_collector_never_returns_rn_ud(tier):
    names = {c.name for c in build_trainable_params(tier=tier, dtype=jnp.float64).constraints}
    assert RN_UD not in names
    if tier != "core":
        assert KAPPA_REDI in names  # the tier path itself is live


def test_explicit_request_with_start_value_returns_rn_ud():
    p = build_trainable_params(active_scheme_keys={GM}, tier=0, include=(RN_UD,),
                               init_values={RN_UD: ORCA2_RN_UD}, dtype=jnp.float64)
    assert [c.name for c in p.constraints] == [RN_UD]
    np.testing.assert_allclose(float(p.as_dict()[RN_UD]), ORCA2_RN_UD, rtol=1e-12)


def test_explicit_request_without_start_value_is_refused():
    with pytest.raises(ValueError, match="explicit_only"):
        build_trainable_params(active_scheme_keys={GM}, tier="extended", include=(RN_UD,),
                               dtype=jnp.float64)
    with pytest.raises(ValueError, match="explicit_only"):
        build_trainable_params(active_scheme_keys={GM}, tier="extended", include=(RN_UD,),
                               init_values={KAPPA_REDI: 1000.0}, dtype=jnp.float64)


def test_explicit_request_that_cannot_be_reached_is_refused():
    with pytest.raises(ValueError, match="explicit_only"):  # no start value, scheme inactive
        build_trainable_params(active_scheme_keys={"ocean.lat.visbeck"}, tier=0,
                               include=(RN_UD,), dtype=jnp.float64)
    with pytest.raises(ValueError, match="not reachable"):  # scheme inactive
        build_trainable_params(active_scheme_keys={"ocean.lat.visbeck"}, tier=0,
                               include=(RN_UD,), init_values={RN_UD: ORCA2_RN_UD},
                               dtype=jnp.float64)
    with pytest.raises(ValueError, match="not reachable"):  # also excluded
        build_trainable_params(active_scheme_keys={GM}, tier=0, include=(RN_UD,),
                               exclude=(RN_UD,), init_values={RN_UD: ORCA2_RN_UD},
                               dtype=jnp.float64)


@pytest.mark.parametrize("tier", ["core", "extended", "aggressive"])
def test_rung10_entry_never_returns_kappa_redi(tier):
    assert set(ORCA2_RUNG10_FROZEN_PARAMS) == {KAPPA_REDI}
    plain = {c.name for c in build_orca2_rung10_trainable_params(
        tier=tier, dtype=jnp.float64).constraints}
    assert KAPPA_REDI not in plain and RN_UD not in plain
    p = build_orca2_rung10_trainable_params(active_scheme_keys={GM}, tier=tier,
                                            include=(RN_UD,),
                                            init_values={RN_UD: ORCA2_RN_UD},
                                            dtype=jnp.float64)
    names = {c.name for c in p.constraints}
    assert KAPPA_REDI not in names and RN_UD in names


def test_rung10_entry_refuses_a_frozen_parameter_by_name():
    for kw in ({"include": (KAPPA_REDI,)}, {"init_values": {KAPPA_REDI: 1000.0}}):
        with pytest.raises(ValueError, match="frozen on ORCA2 rung 10"):
            build_orca2_rung10_trainable_params(active_scheme_keys={GM}, tier="extended",
                                                dtype=jnp.float64, **kw)


def _rung10_record_exists():
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round315_grad_probe as probe,
    )
    return probe.ROOTS["rung10"].exists()


@pytest.mark.skipif(not _rung10_record_exists(), reason="no rung-10 record")
def test_rung10_explicit_rn_ud_gradient_at_n1_matches_preregistered_value():
    import copy

    from legoesm.core.param_overrides import apply_param_overrides
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.training.trainable_params import TrainablePhysicsParams
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round315_grad_probe as probe,
    )

    card, state0, forcing = probe.build("rung10")
    r = card.recipe
    model = LatLonCGridOceanModel(r.grid, r.z_coord, r.model_config, iwm_forcing=r.iwm_forcing)
    base = model.config
    assert float(base.gm_redi.kappa_redi_diffusive_velocity) == ORCA2_RN_UD
    params = build_orca2_rung10_trainable_params(
        active_scheme_keys={GM}, tier=0, include=(RN_UD,),
        init_values={RN_UD: ORCA2_RN_UD}, dtype=jnp.float64)
    raw_step = type(model).__dict__["_step_jitted"].__wrapped__

    def step(state, raw, dt, fw, sf):
        tp = TrainablePhysicsParams(raw_values=raw, constraints=params.constraints)
        m = copy.copy(model)
        m.config = base._replace(gm_redi=apply_param_overrides(
            base.gm_redi, tp.to_overrides()[GM]))
        return raw_step(m, m._seed_tke_preclosure_carry(state), dt, fw, sf)

    losses, _ = probe.make_losses(model, card.recipe.z_coord)
    rev = probe.chain_reverse(jax.jit(step), state0, dict(params.raw_values), forcing, (1,),
                              {"T": losses["T"]}, card.dt_s, log=lambda *_: None)
    g_raw = rev[("T", 1)]["dscales"][RN_UD]
    # per unit relative change of rn_Ud, the unit round 320/321 reported (-4.5e-5)
    dud_draw = float(jax.grad(lambda r: TrainablePhysicsParams(
        raw_values={RN_UD: r}, constraints=params.constraints).as_dict()[RN_UD])(
            params.raw_values[RN_UD]))
    g_rel = g_raw / dud_draw * ORCA2_RN_UD
    print(f"rung10 N=1 T loss: dL/draw = {g_raw:.6e}, dL/dln(rn_Ud) = {g_rel:.6e}")
    assert np.isfinite(g_raw) and g_raw != 0.0
    # PREREG P4: the round-320/321 value, negative, within 1 %
    assert abs(g_rel / RUNG10_DLDLN_RN_UD_REF - 1.0) <= 0.01, g_rel
