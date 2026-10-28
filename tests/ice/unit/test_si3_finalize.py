"""Independent scalar and transformation tests for production SI3 O15/O16."""

from __future__ import annotations

import ast
import inspect

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.ice.si3_finalize import (
    SI3Fluxes,
    SI3StressInputs,
    SI3UpdateFluxForcing,
    si3_ice_update_tau,
    si3_o15_step,
    si3_o16_step,
)
from legoesm.ice.si3_orca2_state import SI3ORCA2StateConfig, SI3State
from legoesm.ice.si3_surface import SI3SurfaceConfig


def _config(*, nn_hls: int = 0) -> SI3ORCA2StateConfig:
    return SI3ORCA2StateConfig.for_lane3_c1d(
        jpl=1,
        nlay_i=1,
        nlay_s=1,
        nn_hls=nn_hls,
        nn_icesal=4,
        ln_pnd=True,
        ln_pnd_LEV=True,
        ln_pnd_lids=True,
        ln_pnd_alb=True,
        ln_icethd=True,
        rn_simin=0.1,
        rn_sinew=4.0,
        rDt_ice=10.0,
    )


def _surface_config(*, nn_hls: int = 0) -> SI3SurfaceConfig:
    return SI3SurfaceConfig(
        rn_Cd_ia=1.0, rn_Ce_ia=1.0, rn_Ch_ia=1.0, rn_pfac=1.0,
        rn_efac=1.0, rn_snwblow=0.0, nn_snwfra=2,
        rn_alb_sdry=0.85, rn_alb_smlt=0.75, rn_alb_idry=0.60,
        rn_alb_imlt=0.50, rn_alb_dpnd=0.27, rn_alb_lpnd=0.07,
        rn_alb_hpiv=0.05, ld_pnd_alb=True, nn_qtrice=0,
        ln_icedyn=False, rn_Cd_io=0.005, rDt_ice=10.0, nn_hls=nn_hls,
        ln_blk=True, ln_mixcpl=False, ln_Cx_ice_cst=True,
        ln_Cx_ice_frm=False, ln_Cx_ice_AN05=False, ln_Cx_ice_LU12=False,
        ln_Cx_ice_LG15=False, nn_flxdist=-1, ln_cndflx=False,
        ln_leadhfx=True, ln_icedO=True, ln_icedH=True,
    )


def _fluxes(shape: tuple[int, int], value: float = 0.0) -> SI3Fluxes:
    return SI3Fluxes(**{
        name: jnp.full(shape, value, dtype=jnp.float64) for name in SI3Fluxes._fields
    })


def _o15_state(a: jax.Array, v: jax.Array, salt: jax.Array) -> SI3State:
    cfg = _config()
    state = SI3State.zeros(1, 1, cfg)
    one3 = jnp.ones((1, 1, 1), dtype=jnp.float64)
    one4 = jnp.ones((1, 1, 1, 1), dtype=jnp.float64)
    eqv = state.equivalent._replace(
        h_i=v / a, h_s=0.02 * one3, h_ip=0.005 * one3,
        h_il=0.001 * one3, a_ip_frac=0.2 * one3,
        a_ip_eff=0.1 * one3, sz_i=2.0 * one4,
        t_i=270.0 * one4, t_s=269.0 * one4, a_i=a, v_s=0.01 * one3,
    )
    agg = state.aggregated._replace(at_i=a[..., 0], vt_i=v[..., 0])
    return state._replace(
        a_i=a, v_i=v, v_s=0.01 * one3, t_su=271.0 * one3,
        a_ip=0.1 * one3, v_ip=0.001 * one3, v_il=0.0001 * one3,
        e_s=2.0 * one4, e_i=3.0 * one4, szv_i=salt,
        equivalent=eqv, aggregated=agg,
    )


def test_o15_scalar_oracle_eager_jit_and_central_gradient() -> None:
    """The oracle is plain NumPy and shares no model functions or constants."""
    cfg = _config()
    a = jnp.asarray([[[0.5]]], dtype=jnp.float64)
    v = jnp.asarray([[[0.025]]], dtype=jnp.float64)
    salt = jnp.asarray([[[[1.0e-4]]]], dtype=jnp.float64)
    flux = _fluxes((1, 1))

    def run(state_arg: SI3State):
        return si3_o15_step(
            state_arg, flux, sss_m=jnp.asarray([[1.0]]),
            sst_m=jnp.asarray([[-1.8]]), rn_amax_2d=jnp.asarray([[0.95]]),
            rn_himin=0.1, config=cfg,
        )

    with jax.disable_jit():
        eager = run(_o15_state(a, v, salt))
    compiled = jax.jit(run)(_o15_state(a, v, salt))
    np.testing.assert_array_equal(eager.state.a_i, compiled.state.a_i)
    np.testing.assert_array_equal(eager.state.szv_i, compiled.state.szv_i)

    # Scalar oracle: h=0.05 thins area to a*h/hmin=0.25.  The salt lower
    # bound is 0.1*v and the upper bound is 4*sss*v, hence salt -> 0.0025.
    h_oracle = np.float64(0.025) / np.float64(0.5)
    a_oracle = np.float64(0.5) * h_oracle / np.float64(0.1)
    pond_oracle = np.float64(0.1) * h_oracle / np.float64(0.1)
    salt_oracle = np.float64(0.1) * np.float64(0.025)
    np.testing.assert_array_equal(compiled.state.a_i, [[[a_oracle]]])
    np.testing.assert_array_equal(compiled.state.a_ip, [[[pond_oracle]]])
    np.testing.assert_array_equal(compiled.state.szv_i, [[[[salt_oracle]]]])
    expected_sfx = -(salt_oracle - 1.0e-4) * 917.0 / 10.0
    np.testing.assert_allclose(compiled.fluxes.sfx_res, expected_sfx, rtol=0, atol=2e-17)

    def objective(x: jax.Array) -> jax.Array:
        return jnp.sum(run(_o15_state(x, v, salt)).state.a_i)

    grad = jax.grad(objective)(a)
    eps = 1.0e-6
    central = (float(objective(a + eps)) - float(objective(a - eps))) / (2.0 * eps)
    assert np.all(np.isfinite(np.asarray(grad)))
    np.testing.assert_allclose(float(grad[0, 0, 0]), central, rtol=2e-7, atol=2e-9)


def _forcing(shape2: tuple[int, int], shape3: tuple[int, int, int]) -> SI3UpdateFluxForcing:
    z2 = jnp.zeros(shape2, dtype=jnp.float64)
    z3 = jnp.zeros(shape3, dtype=jnp.float64)
    values = {name: z2 for name in SI3UpdateFluxForcing._fields}
    for name in (
        "a_i_b", "qsr_ice", "qns_ice", "qtr_ice_bot", "qevap_ice",
        "qcn_ice", "qcn_ice_top", "qcn_ice_bot", "qml_ice",
        "qtr_ice_top", "alb_ice",
    ):
        values[name] = z3
    values.update(
        qns_tot=jnp.full(shape2, 10.0), qsr_tot=jnp.full(shape2, 20.0),
        qsr_oce=jnp.full(shape2, 5.0), emp_oce=jnp.full(shape2, 7.0),
        smask0=jnp.ones(shape2), tmask=jnp.ones(shape2),
        snwice_mass=jnp.full(shape2, 4.0), snwice_mass_b=jnp.full(shape2, 3.0),
    )
    return SI3UpdateFluxForcing(**values)


def test_o16_scalar_flux_oracle_eager_jit_and_zero_ice() -> None:
    cfg = _config()
    state = SI3State.zeros(1, 1, cfg)
    flux = _fluxes((1, 1))
    # Independent scalar inputs make each left-associated sum observable.
    flux = flux._replace(
        hfx_sum=jnp.asarray([[1.0]]), hfx_bom=jnp.asarray([[2.0]]),
        hfx_bog=jnp.asarray([[3.0]]), hfx_dif=jnp.asarray([[4.0]]),
        hfx_opw=jnp.asarray([[5.0]]), hfx_snw=jnp.asarray([[6.0]]),
        hfx_thd=jnp.asarray([[7.0]]), hfx_dyn=jnp.asarray([[8.0]]),
        hfx_res=jnp.asarray([[9.0]]), hfx_sub=jnp.asarray([[10.0]]),
        hfx_spr=jnp.asarray([[11.0]]), wfx_bog=jnp.asarray([[1.0]]),
        wfx_bom=jnp.asarray([[2.0]]), wfx_sum=jnp.asarray([[3.0]]),
        wfx_sni=jnp.asarray([[4.0]]), wfx_opw=jnp.asarray([[5.0]]),
        wfx_dyn=jnp.asarray([[6.0]]), wfx_res=jnp.asarray([[7.0]]),
        wfx_lam=jnp.asarray([[8.0]]), wfx_pnd=jnp.asarray([[9.0]]),
    )
    forcing = _forcing((1, 1), (1, 1, 1))

    def run(s: SI3State):
        return si3_o16_step(
            s, flux, forcing, sss_m=jnp.asarray([[34.0]]), config=cfg,
            surface_config=_surface_config(),
        )

    with jax.disable_jit():
        eager = run(state)
    compiled = jax.jit(run)(state)
    np.testing.assert_array_equal(eager.outputs.qns, compiled.outputs.qns)
    np.testing.assert_array_equal(eager.outputs.emp, compiled.outputs.emp)
    # qt=30-1-2-3-4-5-6+7+8+9+10+11=54; no ice => qsr=20.
    assert float(compiled.outputs.qsr[0, 0]) == 20.0
    assert float(compiled.outputs.qns[0, 0]) == 34.0
    # ice mass flux=36 and no snow/puddle error beyond wfx_pnd=9.
    assert float(compiled.outputs.fwfice[0, 0]) == 45.0
    assert float(compiled.outputs.emp[0, 0]) == -38.0
    for leaf in jax.tree_util.tree_leaves(compiled):
        assert np.all(np.isfinite(np.asarray(leaf)))


def test_o16_tau_gradient_matches_central_difference() -> None:
    cfg = _config(nn_hls=1)
    shape = (4, 4)
    z = jnp.zeros(shape, dtype=jnp.float64)

    def objective(speed: jax.Array) -> jax.Array:
        inputs = SI3StressInputs(
            uu_kbb=z, vv_kbb=z, umask_k1=jnp.ones(shape), vmask_k1=jnp.ones(shape),
            u_ice=jnp.ones(shape) * speed, v_ice=jnp.ones(shape) * 0.5,
            u_oce=z, v_oce=z, drag_io=jnp.ones(shape) * 0.005,
            at_i=jnp.ones(shape) * 0.5, utau=z, vtau=z, tmod_io=z,
            taum=jnp.ones((2, 2)), utau_oce=z[1:-1, 1:-1], vtau_oce=z[1:-1, 1:-1],
        )
        return jnp.sum(si3_ice_update_tau(inputs, config=cfg, ln_drgice_imp=True).utau)

    x = jnp.asarray(1.0, dtype=jnp.float64)
    grad = float(jax.grad(objective)(x))
    eps = 1.0e-5
    central = (float(objective(x + eps)) - float(objective(x - eps))) / (2 * eps)
    assert np.isfinite(grad)
    np.testing.assert_allclose(grad, central, rtol=2e-7, atol=2e-7)


def test_production_call_sites_contain_owned_stages() -> None:
    for function, required in ((si3_o15_step, "si3_ice_cor"), (si3_o16_step, "si3_ice_update_flx")):
        tree = ast.parse(inspect.getsource(function))
        calls = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        assert required in calls
