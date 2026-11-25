"""Round 319: rung-10 trainable scalars are traced inputs and move the state.

The rung-10 tests need the round-5 acquisition record and compile rung 10's
step (minutes, tens of GB); they skip when the record is absent.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round315_grad_probe as probe,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round319_traced_scalars as ts,
)

pytestmark = pytest.mark.skipif(not jax.config.jax_enable_x64, reason="needs JAX_ENABLE_X64=1")

# Scalars that move no state on rung 10, each with the reason a trainer must
# freeze it out (measured, round 319). Every other scalar must move the state.
_NP_LIN = "rung 10 runs nemo_linear: rate = cd0 * uc0 (zdfdrg.F90 np_lin) never reads it"
RUNG10_INERT = {"cd_max": _NP_LIN, "z0": _NP_LIN, "ke0": _NP_LIN,
                # Round 320: rung 10 runs L2's mode 20, 0.5*rn_Ud*max(e1,e2)
                # per face; kappa_Redi only records the nominal aht0.
                "kappa_redi": "inert since round 320 (mode-20 coefficient from rn_Ud)"}
# Scalars that move the state when raised and not when lowered: the loss
# gradient is exactly 0, so a gradient trainer cannot move them. Each one is
# pending the user's decision; it is not a live trainable.
RUNG10_ONE_SIDED = {
    "kappaH_min": "one-sided, gradient exactly 0, frozen out of training (DECISION 143): the floor "
                  "sits 1 ulp below the TKE branch's minimum (0.1 * 1e-9) in 7,634 cells",
}


def _toy_config():
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
    from legoesm.ocean.physics.vertical_mixing.config import TKEConfig, VerticalMixingConfig
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.vertical import create_full_step_coordinate, create_ocean_z_star

    grid = create_latlon_grid(n_lat=6, n_lon=4)
    z = create_ocean_z_star(n_levels=3, H_max=3000.0)
    ones = jnp.ones((grid.n_lat, grid.n_lon, z.n_levels))
    z = create_full_step_coordinate(z, bottom_level=np.full(ones.shape[:2], z.n_levels))
    z = z._replace(nemo_ldf_ahmt=ones, nemo_ldf_ahmf=2.0 * ones)
    ph = OceanPhysicsConfig(vertical_mixing=VerticalMixingConfig(scheme="tke", tke=TKEConfig()))
    cfg = LatLonCGridOceanConfig.from_flat(
        bottom_drag_scheme="nemo_linear", bottom_drag_cd0=1e-3, bottom_drag_cdmax=0.1,
        bottom_drag_z0=3e-3, bottom_drag_ke0=2.5e-3, bottom_drag_uc0=0.4,
        gm_redi=GMRediConfig(kappa_GM=0.0, kappa_Redi=1.0e3), physics=ph)
    return cfg, z


def _fields(cfg, z):
    tke = cfg.physics.vertical_mixing.tke
    return {**{k: getattr(cfg.bottom_drag, f) for k, f in ts.DRAG.items()},
            "kappa_redi": cfg.gm_redi.kappa_Redi,
            "visc_scale": (np.asarray(z.nemo_ldf_ahmt), np.asarray(z.nemo_ldf_ahmf)),
            **{k: getattr(tke, f) for k, f in ts.TKE.items()}}


def _flat(v):
    return np.concatenate([np.ravel(x) for x in v]) if isinstance(v, tuple) else np.asarray(v)


def test_scaled_is_identity_at_one_and_each_scale_touches_only_its_field():
    cfg, z = _toy_config()
    base = _fields(cfg, z)
    ones = {k: 1.0 for k in ts.SCALARS}
    for k, got in [(None, _fields(*ts.scaled(cfg, z, ones)))] + [
            (k, _fields(*ts.scaled(cfg, z, {**ones, k: 2.0}))) for k in ts.SCALARS]:
        for j in ts.SCALARS:
            want = 2.0 * _flat(base[j]) if j == k else _flat(base[j])
            np.testing.assert_array_equal(_flat(got[j]), want, err_msg=f"{k} -> {j}")
    with pytest.raises(ValueError, match="unknown scalars"):
        ts.scaled(cfg, z, {"cd00": 1.0})


def test_moved_counts_sees_a_planted_value_signed_zero_and_ignores_nan_nan():
    a = {"x": np.array([1.0, np.nan, 0.0, 3.0]), "n": np.array([1, 2])}
    assert ts.moved_counts(a, a) == {}
    b = {"x": np.array([1.0, np.nan, -0.0, 3.5]), "n": np.array([1, 3])}
    assert ts.moved_counts(a, b) == {"['x']": 2, "['n']": 1}


_REC = probe.ROOTS["rung10"]
needs_record = pytest.mark.skipif(not _REC.exists(), reason=f"no rung-10 record at {_REC}")


@needs_record
def test_cd0_traces_through_rung10_step_under_jit_grad():
    """jit(grad) of a rung-10 one-step loss w.r.t. traced cd0 traces; with
    the TKE bottom-BC float() casts restored it raises ConcretizationTypeError."""
    assert len(str(ts.trace_cd0("rung10")).splitlines()) > 0


@pytest.fixture(scope="module")
def rung10_scan():
    # two steps: the lateral viscosity scale first moves the state at kt 2
    return ts.scan("rung10", rel=1e-3, steps=2, log=lambda *_: None)


@needs_record
def test_rung10_traced_step_at_one_is_the_production_step(rung10_scan):
    assert rung10_scan["parity_kt1_vs_production"] == {}


@needs_record
@pytest.mark.parametrize("name", ts.SCALARS)
def test_rung10_scalar_moves_both_ways_or_is_declared(rung10_scan, name):
    """A live trainable needs a nonzero loss gradient: it must move the state
    under both +1e-3 and -1e-3. Moving one way only is a kink with gradient 0."""
    row = rung10_scan["scalars"][name]
    up, down = (sum(row[k]["moved_total_by_kt"]) for k in ("up", "down"))
    if name in RUNG10_INERT:
        assert (up, down) == (0, 0), f"{name} now moves ({up}, {down}): drop it from RUNG10_INERT"
    elif name in RUNG10_ONE_SIDED:
        assert up > 0 and down == 0, f"{name} is no longer one-sided ({up}, {down})"
    else:
        assert up > 0 and down > 0, f"{name}: moves {up} values raised, {down} lowered"
