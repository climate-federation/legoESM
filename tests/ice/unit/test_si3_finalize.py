"""Independent scalar and transformation tests for production SI3 O15/O16."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.ice import sea_ice as sea_ice_module
from legoesm.ice.config import build_si3_orca2_o2_config
from legoesm.ice.si3_finalize import (
    SI3CorrectionResult,
    SI3FinalizeForcing,
    SI3FinalizeResponse,
    SI3Fluxes,
    SI3StressInputs,
    SI3UpdateFluxForcing,
    si3_ice_update_tau,
    si3_o15_step,
    si3_o16_step,
)
from legoesm.ice.si3_orca2_state import SI3ORCA2StateConfig, SI3State
from legoesm.ice.si3_surface import SI3SurfaceConfig


def _config(
    *, nn_hls: int = 0, jpl: int = 1, nlay_i: int = 1, nlay_s: int = 1
) -> SI3ORCA2StateConfig:
    return SI3ORCA2StateConfig.for_lane3_c1d(
        jpl=jpl,
        nlay_i=nlay_i,
        nlay_s=nlay_s,
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


def test_o15_five_category_rebin_oracle_jit_and_gradient() -> None:
    """Exercise ``_ice_itd_reb``/``_shift_boundary`` with a NumPy oracle."""
    cfg = _config(jpl=5)
    a0 = np.asarray([0.20, 0.10, 0.10, 0.10, 0.10], dtype=np.float64)
    h0 = np.asarray([1.00, 1.00, 2.00, 3.00, 5.00], dtype=np.float64)
    v0 = a0 * h0

    def make_state(scale: jax.Array) -> SI3State:
        state = SI3State.zeros(1, 1, cfg)
        a = jnp.asarray(a0).reshape(1, 1, 5)
        v = jnp.asarray(v0).reshape(1, 1, 5).at[..., 0].multiply(scale)
        eqv = state.equivalent._replace(h_i=v / a, a_i=a)
        agg = state.aggregated._replace(at_i=jnp.sum(a, axis=-1), vt_i=jnp.sum(v, axis=-1))
        return state._replace(a_i=a, v_i=v, equivalent=eqv, aggregated=agg)

    def run(scale: jax.Array, *, skip: bool = False):
        return si3_o15_step(
            make_state(scale),
            _fluxes((1, 1)),
            sss_m=jnp.asarray([[34.0]]),
            sst_m=jnp.asarray([[-1.8]]),
            rn_amax_2d=jnp.asarray([[0.95]]),
            rn_himin=0.01,
            config=cfg,
            _plant_skip_rebin=skip,
        )

    eager = run(jnp.asarray(1.0))
    compiled = jax.jit(lambda x: run(x))(jnp.asarray(1.0))
    np.testing.assert_array_equal(eager.state.a_i, compiled.state.a_i)
    np.testing.assert_array_equal(eager.state.v_i, compiled.state.v_i)
    assert not np.array_equal(
        np.asarray(compiled.state.a_i), np.asarray(run(jnp.asarray(1.0), skip=True).state.a_i)
    )

    # Independent scalar-category oracle for the two NEMO rebin sweeps.
    a_ref, v_ref = a0.copy(), v0.copy()
    hi_max = np.asarray(compiled.state.bounds.hi_max)
    hi_mean = np.asarray(compiled.state.bounds.hi_mean)
    for donor in range(4):
        if a_ref[donor] > 0.0 and v_ref[donor] > a_ref[donor] * hi_max[donor + 1]:
            da = np.float64(0.5) * a_ref[donor]
            dv = v_ref[donor] - (np.float64(0.5) * a_ref[donor]) * hi_mean[donor]
            a_ref[donor] -= da
            a_ref[donor + 1] += da
            v_ref[donor] -= dv
            v_ref[donor + 1] += dv
    for receiver in range(3, -1, -1):
        donor = receiver + 1
        if a_ref[donor] > 0.0 and v_ref[donor] <= a_ref[donor] * hi_max[receiver + 1]:
            a_ref[receiver] += a_ref[donor]
            v_ref[receiver] += v_ref[donor]
            a_ref[donor] = 0.0
            v_ref[donor] = 0.0
    np.testing.assert_array_equal(compiled.state.a_i[0, 0], a_ref)
    np.testing.assert_array_equal(compiled.state.v_i[0, 0], v_ref)

    def objective(x: jax.Array) -> jax.Array:
        weights = jnp.arange(1.0, 6.0, dtype=jnp.float64)
        return jnp.sum(run(x).state.v_i * weights)

    x = jnp.asarray(1.0, dtype=jnp.float64)
    grad = float(jax.grad(objective)(x))
    eps = 1.0e-6
    central = (float(objective(x + eps)) - float(objective(x - eps))) / (2.0 * eps)
    assert np.isfinite(grad)
    np.testing.assert_allclose(grad, central, rtol=2e-7, atol=2e-9)


def test_o15_min_tie_keeps_bounded_low_signed_zero() -> None:
    """Object-code ``minsd src=bounded_low,dst=upper`` keeps +0 on a tie."""
    cfg = _config()._replace(rn_simin=0.0)
    a = jnp.asarray([[[0.5]]], dtype=jnp.float64)
    v = jnp.asarray([[[0.25]]], dtype=jnp.float64)
    state = _o15_state(a, v, jnp.asarray([[[[0.0]]]], dtype=jnp.float64))
    out = si3_o15_step(
        state,
        _fluxes((1, 1)),
        sss_m=jnp.asarray([[-0.0]], dtype=jnp.float64),
        sst_m=jnp.asarray([[-1.8]], dtype=jnp.float64),
        rn_amax_2d=jnp.asarray([[0.95]], dtype=jnp.float64),
        rn_himin=0.1,
        config=cfg,
    )
    assert int(np.asarray(out.state.szv_i).view(np.uint64)[0, 0, 0, 0]) == 0


def test_finalize_rejects_state_and_config_shape_disagreement() -> None:
    state = SI3State.zeros(1, 1, _config())
    with pytest.raises(ValueError, match="state/config shape mismatch"):
        si3_o15_step(
            state,
            _fluxes((1, 1)),
            sss_m=jnp.ones((1, 1)),
            sst_m=jnp.zeros((1, 1)),
            rn_amax_2d=jnp.ones((1, 1)),
            rn_himin=0.1,
            config=_config(jpl=5),
        )


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

    def objective(hfx_res: jax.Array) -> jax.Array:
        changed = flux._replace(hfx_res=jnp.reshape(hfx_res, (1, 1)))
        return si3_o16_step(
            state,
            changed,
            forcing,
            sss_m=jnp.asarray([[34.0]]),
            config=cfg,
            surface_config=_surface_config(),
        ).outputs.qns[0, 0]

    assert float(jax.grad(objective)(jnp.asarray(9.0))) == 1.0


def test_o16_multi_category_aggregation_and_gradient() -> None:
    cfg = _config(jpl=5)
    state0 = SI3State.zeros(1, 1, cfg)
    a0 = jnp.asarray([[[0.05, 0.10, 0.15, 0.20, 0.25]]], dtype=jnp.float64)
    v0 = a0 * jnp.asarray([[[0.2, 0.8, 1.5, 2.5, 4.0]]])
    forcing = _forcing((1, 1), (1, 1, 5))

    def run(scale: jax.Array):
        a = a0 * scale
        state = state0._replace(a_i=a, v_i=v0)
        return si3_o16_step(
            state,
            _fluxes((1, 1)),
            forcing,
            sss_m=jnp.asarray([[34.0]]),
            config=cfg,
            surface_config=_surface_config(),
        )

    eager = run(jnp.asarray(1.0))
    compiled = jax.jit(run)(jnp.asarray(1.0))
    np.testing.assert_array_equal(eager.outputs.fr_i, compiled.outputs.fr_i)
    np.testing.assert_array_equal(compiled.outputs.fr_i, jnp.sum(a0, axis=-1))
    grad = jax.grad(lambda x: jnp.sum(run(x).outputs.fr_i))(jnp.asarray(1.0))
    np.testing.assert_allclose(float(grad), float(jnp.sum(a0)), rtol=0, atol=0)


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


def test_step_sea_ice_executes_finalize_stages_and_o15_removal_changes_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _config(nn_hls=1)
    shape, owned = (4, 4), (2, 2)
    state0 = SI3State.zeros(*shape, cfg)
    a = jnp.full((*shape, 1), 0.5)
    v = jnp.full((*shape, 1), 0.025)
    state0 = state0._replace(
        a_i=a,
        v_i=v,
        u_ice=jnp.ones(shape),
        v_ice=jnp.ones(shape) * 0.5,
        szv_i=jnp.full((*shape, 1, 1), 1.0e-4),
        equivalent=state0.equivalent._replace(h_i=v / a, a_i=a),
        aggregated=state0.aggregated._replace(at_i=a[..., 0], vt_i=v[..., 0]),
    )
    z = jnp.zeros(shape)
    stress = SI3StressInputs(
        uu_kbb=z,
        vv_kbb=z,
        umask_k1=jnp.ones(shape),
        vmask_k1=jnp.ones(shape),
        u_ice=jnp.ones(shape),
        v_ice=jnp.ones(shape) * 0.5,
        u_oce=z,
        v_oce=z,
        drag_io=jnp.ones(shape) * 0.005,
        at_i=a[..., 0],
        utau=z,
        vtau=z,
        tmod_io=z,
        taum=jnp.ones(owned),
        utau_oce=jnp.zeros(owned),
        vtau_oce=jnp.zeros(owned),
    )
    update = _forcing(owned, (*owned, 1))._replace(
        tmask=jnp.ones(shape),
        snwice_mass=jnp.full(shape, 4.0),
        snwice_mass_b=jnp.full(shape, 3.0),
    )
    forcing = SI3FinalizeForcing(
        fluxes=_fluxes(shape),
        update_state=state0,
        update=update,
        stress=stress,
        sss_m=jnp.full(shape, 34.0),
        sst_m=jnp.full(shape, -1.8),
        rn_amax_2d=jnp.full(shape, 0.95),
        rn_himin=0.1,
        surface_config=_surface_config(nn_hls=1),
        ln_drgice_imp=False,
    )
    sea_cfg = build_si3_orca2_o2_config(cfg)

    def step():
        return sea_ice_module.step_sea_ice(
            state0, forcing, jnp.zeros(owned), jnp.zeros(owned), jnp.zeros(owned),
            sea_cfg, 0.0, cfg.rDt_ice,
        )

    _, response = step()
    assert isinstance(response, SI3FinalizeResponse)
    assert np.any(np.asarray(response.correction.state.a_i[1:-1, 1:-1]) != 0.5)
    assert np.any(np.asarray(response.stress.utau[1:-1, 1:-1]) != 0.0)

    monkeypatch.setattr(
        sea_ice_module,
        "si3_o15_step",
        lambda state, fluxes, **_: SI3CorrectionResult(state=state, fluxes=fluxes),
    )
    _, bypassed = step()
    assert not np.array_equal(
        np.asarray(response.correction.state.a_i),
        np.asarray(bypassed.correction.state.a_i),
    )
