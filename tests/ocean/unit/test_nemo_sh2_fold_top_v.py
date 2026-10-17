"""zdf_sh2 on the T-pivot row's north V face (V1-selected).

zdfsh2.f90:98-113 evaluates zsh2v on the pivot row's north face too: NEMO's
lbc_nfd (T-pivot, lbcnfd.f90:584-720) makes that face wet, its v the mirrored
image, its north avm_k and ssh the images of the row below the pivot. With
symmetric pivot-row inputs NEMO's pivot-row p_sh2 is exactly mirror-symmetric
(measured on the V1 record); the card's wall form is not.
"""
from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.field import Field
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.physics.vertical_mixing.config import TKEConfig

pytestmark = pytest.mark.skipif(not jax.config.x64_enabled,
                                reason="binary64 receipt")

NY, NX, NZ = 3, 6, 3
PERM = np.asarray([(NX - i) % NX for i in range(NX)])      # card fold_convention


class _Z:
    def __init__(self, base, vmask):
        self._base, self.nemo_een_barotropic = base, SimpleNamespace(vmask=vmask)

    def __getattr__(self, name):
        return getattr(self._base, name)


def _sym(a):
    """Make the pivot (top) row of a T/W field mirror-symmetric."""
    a = np.array(a)
    a[-1] = 0.5 * (a[-1] + a[-1][PERM])
    return a


def _case(flag, fold=True, stage="step_entry"):
    from legoesm.ocean.vertical import (
        create_partial_cell_coordinate, create_z_star_from_thicknesses,
    )

    rng = np.random.default_rng(7)
    one = np.ones((NY, NX))
    raw = create_z_star_from_thicknesses(
        [1.0, 2.0, 3.0],
        nemo_gdept_0_m=np.broadcast_to([0.5, 2.0, 4.5], (NY, NX, NZ)).copy(),
        nemo_gdepw_0_m=np.broadcast_to([0.0, 1.0, 3.0], (NY, NX, NZ)).copy(),
        nemo_e3t_0_m=np.broadcast_to([1.0, 2.0, 3.0], (NY, NX, NZ)).copy(),
        nemo_e3w_0_m=np.broadcast_to([1.0, 1.5, 2.5], (NY, NX, NZ)).copy(),
        nemo_hu_0_m=6.0 * one, nemo_hv_0_m=6.0 * one, nemo_e1e2t_m=one,
        nemo_e1e2u_m=one, nemo_e1e2v_m=one)
    z = _Z(create_partial_cell_coordinate(raw, jnp.full((NY, NX), raw.H_max)),
           jnp.ones((NY, NX, NZ)))
    eta = _sym(rng.uniform(-0.2, 0.2, (NY, NX)))
    avm = _sym(rng.uniform(0.1, 0.9, (NY, NX, NZ - 1)))
    u = rng.uniform(-1, 1, (NY, NX + 1, NZ))
    # pivot-row U faces pair k <-> NX+1-k (east face of i = west face of
    # PERM[i]) with the U fold sign -1; face NX is the seam copy of face 0
    upair = (NX + 1 - np.arange(NX)) % NX
    u[-1, :NX] = 0.5 * (u[-1, :NX] - u[-1, :NX][upair])
    u[-1, NX] = u[-1, 0]
    v = rng.uniform(-1, 1, (NY + 1, NX, NZ))
    v[-1] = -v[-2][PERM]                                   # stored fold image row
    d3, d2 = ("lat", "lon", "level"), ("lat", "lon")
    tke = TKEConfig(tke_shear_evaluation_stage=stage,
                    tke_shear_production="nemo_face_native_nbb2",
                    tke_shear_avm_weighting="nemo_face",
                    tke_shear_metric_source="nemo_qco_live_face",
                    nemo_sh2_fold_top_v=flag)
    cfg = SimpleNamespace(physics=SimpleNamespace(
        vertical_mixing=SimpleNamespace(scheme="tke", tke=tke)))
    grid = SimpleNamespace(fold=SimpleNamespace(
        is_active=True, perm_T=jnp.asarray(PERM)) if fold else None)
    model = SimpleNamespace(config=cfg, z_coord=z, grid=grid)

    def p_sh2(eta, u, v, avm):
        state = SimpleNamespace(
            T=Field(jnp.zeros((NY, NX, NZ)), "T", d3, "degC"),
            H_bathy=Field(jnp.full((NY, NX), raw.H_max), "H_bathy", d2, "m"),
            eta=Field(eta, "eta", d2, "m"), u=Field(u, "u", d3, "m/s"),
            v=Field(v, "v", d3, "m/s"), tke_avm=Field(avm, "tke_avm", d3, "m2/s"))
        return LatLonCGridOceanModel._tke_step_entry_p_sh2(model, state)

    args = tuple(jnp.asarray(a) for a in (eta, u, v, avm))
    traced = np.asarray(jax.jit(p_sh2)(*args))
    np.testing.assert_array_equal(np.asarray(p_sh2(*args)), traced)  # eager == jit
    return traced


def test_selected_pivot_row_shear_is_mirror_symmetric():
    top = _case(True)[-1]
    np.testing.assert_array_equal(top, top[PERM])


def test_default_keeps_the_wall_face_and_its_asymmetry():
    on, off = _case(True), _case(False)
    np.testing.assert_array_equal(on[:-1], off[:-1])        # only the pivot row moves
    assert not np.array_equal(off[-1], off[-1][PERM])


def test_flag_refuses_a_grid_without_a_fold():
    with pytest.raises(ValueError, match="nemo_sh2_fold_top_v"):
        _case(True, fold=False)


def test_flag_refuses_the_in_solve_shear_that_would_ignore_it():
    with pytest.raises(ValueError, match="read only by the step-entry"):
        _case(True, stage="implicit_solve_state")
