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


def _bottom_bc_production_case(flag: bool, fold: bool = True):
    """Production vertical-mixing call exercising ``_tke_bottom_dirichlet`` on a
    partial-cell north-fold staircase (``mbkt = 2`` < ``nlev - 1 = 3``) with
    zero vertical shear ``dv == 0`` so ``_tke_bottom_dirichlet`` is isolated
    from ``_tke_step_entry_p_sh2`` inside ``_apply_implicit_vertical_mixing``.
    """
    from legoesm import constants
    from legoesm.core.field import Field
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.dynamics.ocean_tendency_common import (
        nemo_effective_bottom_drag_r,
    )
    from legoesm.ocean.physics.vertical_mixing.config import TKEConfig
    from legoesm.ocean.physics.vertical_mixing.tke import (
        nemo_bottom_tke_dirichlet,
    )
    from legoesm.ocean.vertical import (
        create_partial_cell_coordinate, create_z_star_from_thicknesses,
    )

    rng = np.random.default_rng(19)
    nz_bot = 4
    one = np.ones((NY, NX))
    raw = create_z_star_from_thicknesses(
        [1.0, 2.0, 3.0, 4.0],
        nemo_gdept_0_m=np.broadcast_to([0.5, 2.0, 4.5, 8.0], (NY, NX, nz_bot)).copy(),
        nemo_gdepw_0_m=np.broadcast_to([0.0, 1.0, 3.0, 6.0], (NY, NX, nz_bot)).copy(),
        nemo_e3t_0_m=np.broadcast_to([1.0, 2.0, 3.0, 4.0], (NY, NX, nz_bot)).copy(),
        nemo_e3w_0_m=np.broadcast_to([1.0, 1.5, 2.5, 3.5], (NY, NX, nz_bot)).copy(),
        nemo_hu_0_m=6.0 * one, nemo_hv_0_m=6.0 * one, nemo_e1e2t_m=one,
        nemo_e1e2u_m=one, nemo_e1e2v_m=one)
    # Depth 6.0 m makes levels k=0,1,2 wet (bottom_level=2) and k=3 dry, so the
    # in-solve seafloor Dirichlet slot k=2 (NEMO jk=mbkt+1=4) couples into the
    # wet interface k=1 (NEMO jk=3) during the literal Thomas reverse sweep.
    # compute_face_masks_3d leaves north-fold face vm3[-1]=0; the card's
    # lbc_lnk-processed NEMO vmask (dommsk.F90:146-172) has vmask[-1, :, :3]=1.
    base_z = create_partial_cell_coordinate(raw, jnp.full((NY, NX), 6.0))
    raw_vmask = np.zeros((NY, NX, nz_bot), dtype=np.float64)
    raw_vmask[..., :3] = 1.0
    z = base_z._replace(
        nemo_een_barotropic=SimpleNamespace(vmask=jnp.asarray(raw_vmask)))

    eta = _sym(rng.uniform(-0.05, 0.05, (NY, NX)))
    avm = _sym(rng.uniform(0.05, 0.2, (NY, NX, nz_bot - 1)))
    tke0 = _sym(rng.uniform(1.0e-4, 5.0e-4, (NY, NX, nz_bot - 1)))
    u = np.zeros((NY, NX + 1, nz_bot), dtype=np.float64)
    v = np.zeros((NY + 1, NX, nz_bot), dtype=np.float64)
    # Depth-uniform south-face velocity on the pivot row (dv == 0 everywhere);
    # north face is its T-pivot fold image -v[-2][PERM].
    v_row = np.array([0.8, -0.3, 0.5, -0.1, 0.4, -0.6], dtype=np.float64)
    v[-2, :, :] = v_row[:, None]
    v[-1, :, :] = -v_row[PERM, None]

    d3, d2 = ("lat", "lon", "level"), ("lat", "lon")
    tke_cfg = TKEConfig(
        prognostic=True,
        bottom_tke_bc=True,
        tke_background=1.0e-14,
        mxl_min=1.0e-3,
        kappaM_min=1.0e-12,
        kappaH_min=1.0e-12,
        tke_preclosure_coeff_source="carried_previous_step",
        tke_shear_evaluation_stage="step_entry",
        tke_shear_production="nemo_face_native_nbb2",
        tke_shear_avm_weighting="nemo_face",
        tke_shear_metric_source="nemo_qco_live_face",
        nemo_sh2_fold_top_v=flag,
    )
    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.convection.config import OceanConvectionConfig
    from legoesm.ocean.physics.vertical_mixing.config import (
        VerticalMixingConfig,
    )
    physics = OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(scheme="tke", tke=tke_cfg),
        convection=OceanConvectionConfig(scheme="none"),
    )
    bottom_drag = SimpleNamespace(
        bottom_drag_scheme="nemo_linear",
        bottom_drag_cd0=1.0e-3,
        bottom_drag_cdmax=0.1,
        bottom_drag_z0=3.0e-3,
        bottom_drag_ke0=2.5e-3,
        bottom_drag_uc0=0.4,
    )
    cfg = SimpleNamespace(
        physics=physics,
        bottom_drag=bottom_drag,
        eos="linear",
        eos_linear=None,
        eos_nemo_seos=None,
        implicit_ice_ocean_top_drag=False,
        A_v=1.0e-4,
        K_v=1.0e-5,
    )
    grid = SimpleNamespace(
        lat_T=jnp.zeros((NY, NX), dtype=jnp.float64),
        fold=SimpleNamespace(is_active=True, perm_T=jnp.asarray(PERM))
        if fold else None,
    )
    model = SimpleNamespace(
        config=cfg,
        z_coord=z,
        grid=grid,
        _iwm_forcing=None,
        _nemo_ws_test_hooks=SimpleNamespace(
            zdf_momentum_observer=None,
            tke_rhs_materialization="",
            tke_rhs_intermediate="",
        ),
        _tke_post_mixing_active=lambda: False,
        _lat_dependent_constant_vmix=lambda: False,
    )
    from types import MethodType
    for name in ("_tke_bottom_dirichlet", "_tke_bottom_level",
                 "_tke_step_entry_p_sh2"):
        setattr(model, name, MethodType(getattr(LatLonCGridOceanModel, name), model))

    class _StateNS(SimpleNamespace):
        def _replace(self, **kwargs):
            return _StateNS(**{**self.__dict__, **kwargs})

    def run_vmix(eta_in, u_in, v_in, avm_in, tke_in):
        T_arr = jnp.broadcast_to(
            jnp.asarray([15.0, 10.0, 6.0, 2.0], dtype=jnp.float64), (NY, NX, nz_bot))
        S_arr = jnp.full((NY, NX, nz_bot), 35.0, dtype=jnp.float64)
        state = _StateNS(
            T=Field(T_arr, "T", d3, "degC"),
            S=Field(S_arr, "S", d3, "psu"),
            H_bathy=Field(jnp.full((NY, NX), 6.0, dtype=jnp.float64), "H_bathy", d2, "m"),
            land_mask=Field(jnp.ones((NY, NX), dtype=jnp.float64), "land_mask", d2, "1"),
            eta=Field(eta_in, "eta", d2, "m"),
            u=Field(u_in, "u", d3, "m/s"),
            v=Field(v_in, "v", d3, "m/s"),
            tke=Field(tke_in, "tke", d3, "m2/s2"),
            tke_avm=Field(avm_in, "tke_avm", d3, "m2/s"),
            tke_avt=Field(avm_in, "tke_avt", d3, "m2/s"),
            tke_dissl=None,
        )
        bot = model._tke_bottom_dirichlet(state, u_now=u_in, v_now=v_in)
        kv, av = LatLonCGridOceanModel._apply_implicit_vertical_mixing(
            model, state, 10800.0, None, tke_old=tke_in,
            u_now=u_in, v_now=v_in,
            eta_now=eta_in, eta_step_entry=eta_in, return_K_profiles=True)
        return bot, kv, av

    args = tuple(jnp.asarray(a, dtype=jnp.float64) for a in (eta, u, v, avm, tke0))
    with jax.disable_jit():
        eager_bot, eager_kv, eager_av = [np.asarray(x) for x in run_vmix(*args)]
    jit_bot, jit_kv, jit_av = [np.asarray(x) for x in jax.jit(run_vmix)(*args)]
    np.testing.assert_array_equal(eager_bot, jit_bot)
    np.testing.assert_allclose(eager_kv, jit_kv, rtol=1e-13, atol=0.0)
    np.testing.assert_allclose(eager_av, jit_av, rtol=1e-13, atol=0.0)

    # Direct NEMO zdftke.f90:303-309 reference on the pivot row (mbkt=2, e3t_b=3.0):
    # both south and north V faces are wet in NEMO's lbc_lnk vmask, so zmskv=1.
    v_cc_top = 0.5 * (v[-2, :, 2] + v[-1, :, 2])
    r_t_top = nemo_effective_bottom_drag_r(
        jnp.zeros(NX, dtype=jnp.float64),
        jnp.asarray(v_cc_top, dtype=jnp.float64),
        jnp.full(NX, 3.0, dtype=jnp.float64),
        scheme="nemo_linear",
        cd0=1.0e-3, cd_max=0.1, z0=3.0e-3, ke0=2.5e-3, uc0=0.4,
        von_karman=constants.kappa_von_karman,
    )
    v_sum_top = jnp.asarray(v[-2, :, 2] + v[-1, :, 2], dtype=jnp.float64)
    nemo_ref_top = np.asarray(nemo_bottom_tke_dirichlet(
        r_t_top, jnp.zeros(NX, dtype=jnp.float64), v_sum_top, tke_cfg))
    return jit_bot, jit_kv, nemo_ref_top


def test_production_bottom_dirichlet_restores_north_fold_vmask_and_matches_nemo():
    bot_on, kv_on, nemo_ref_top = _bottom_bc_production_case(True)
    bot_off, kv_off, _ = _bottom_bc_production_case(False)
    # Interior rows below the pivot row are untouched.
    np.testing.assert_array_equal(bot_on[:-1], bot_off[:-1])
    np.testing.assert_array_equal(kv_on[:-1], kv_off[:-1])
    # Selected branch matches NEMO's lbc_lnk vmask formula and is fold-symmetric
    # both at _tke_bottom_dirichlet and in the post-Thomas-solve K_v profile at
    # wet interface k=1 (coupled to bottom Dirichlet slot k=2).
    np.testing.assert_allclose(bot_on[-1], nemo_ref_top, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(bot_on[-1], bot_on[-1][PERM])
    np.testing.assert_array_equal(kv_on[-1, :, 1], kv_on[-1, PERM, 1])
    # Reverting/disabling the north-fold V-face mask breaks both the bottom BC
    # and the coupled wet K_v profile on the pivot row.
    assert not np.array_equal(bot_off[-1], nemo_ref_top)
    assert not np.array_equal(bot_off[-1], bot_off[-1][PERM])
    assert not np.array_equal(kv_off[-1, :, 1], kv_on[-1, :, 1])
    assert not np.array_equal(kv_off[-1, :, 1], kv_off[-1, PERM, 1])
    with pytest.raises(ValueError, match="nemo_sh2_fold_top_v"):
        _bottom_bc_production_case(True, fold=False)


