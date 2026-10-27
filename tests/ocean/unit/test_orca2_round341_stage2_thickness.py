"""Round 341 production-call-site, Stage-1/Stage-3/legacy bit-identity, and
eager==jit controls for Stage-2 ``_nemo_ws_stage_transport`` stored ``e3t(Kmm)``
thickness (``ocean_model_latlon_cgrid.py:2293-2295`` and ``:9570-9572``).
"""
from __future__ import annotations

from pathlib import Path
import sys
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
for _pkg in (REPO, REPO / "packages/core", REPO / "packages/ocean"):
    if str(_pkg) not in sys.path:
        sys.path.insert(0, str(_pkg))

from legoesm.core.field import Field  # noqa: E402
from legoesm.core.precision import PrecisionPolicy, set_policy  # noqa: E402
from legoesm.ocean.dynamics import ocean_model_latlon_cgrid as omlc  # noqa: E402
from legoesm.ocean.eos import (  # noqa: E402
    nemo_r3t_rk3_stage1_stretch,
    nemo_r3t_stretch,
)
from legoesm.ocean.fidelity.nemo_testcase_recipe import (  # noqa: E402
    build_gyre_zco_card,
)


def _from_hex64(hex_str: str) -> np.float64:
    return np.asarray(np.uint64(int(hex_str, 16))).view(np.float64)


def _hex64(val: float | np.floating) -> str:
    arr = np.asarray(val, dtype=np.float64)
    return f"0x{int(arr.view(np.uint64)):016x}"


def _build_stage2_fixture(*, stored_flag: bool = True):
    """Build a compact C-grid ocean model and state whose interior wet cell
    ``(j=1, i=1, k=3)`` carries the exact Round-341 ``kt=10`` binary64 geometry
    (``r1_ht_0 = 0x3f9642c8590b2164``, ``e3t_0[k=3] = 0x402402c768e8a600``,
    ``ssh_b = 0xc004dac971407002``, ``ssha = 0x3fd4c94d4afbb824``) where NEMO's
    stored Stage-1 ``r3t(Kmm)`` thickness ``e3t_0 * (1 + r3t(Kmm))``
    (``0x40234d4bd6440687``) differs by 1 ULP from
    ``nemo_qco_live_t_thickness(eta_13)`` (``0x40234d4bd6440686``).
    """
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    card = build_gyre_zco_card()
    grid = card.recipe.grid
    z0 = card.recipe.z_coord
    e3t_0 = np.asarray(z0.nemo_e3t_0, dtype=np.float64).copy()
    e3t_0[1, 1, 3] = float(_from_hex64("0x402402c768e8a600"))
    e3t_0_jnp = jnp.asarray(e3t_0, dtype=jnp.float64)
    z_coord = z0._replace(nemo_e3t_0=e3t_0_jnp, h_partial=e3t_0_jnp)

    base_cfg = card.recipe.model_config
    cfg = base_cfg._replace(
        barotropic=base_cfg.barotropic._replace(n_barotropic_substeps=2),
        physics=None,
        gm_redi=None,
        implicit_vertical_mixing=False,
        zdf_drag_in_matrix=False,
        zdf_baroclinic_only=False,
        barotropic_drag_substep=False,
        rk3_stage_r3t_nemo_literal=True,
        rk3_stage2_tracer_nemo_stored_thickness=stored_flag,
    )
    model = omlc.LatLonCGridOceanModel(grid, z_coord, cfg)

    st0 = card.recipe.initial_state
    H_bathy = st0.H_bathy.data.at[1, 1].set(46.0)
    eta_b = st0.eta.data.at[1, 1].set(float(_from_hex64("0xc004dac971407002")))
    eta_a = st0.eta.data.at[1, 1].set(float(_from_hex64("0x3fd4c94d4afbb824")))
    u_arr = st0.u.data + 0.02 * st0.u_mask.data[..., None]
    v_arr = st0.v.data - 0.015 * st0.v_mask.data[..., None]

    state = st0._replace(
        H_bathy=Field(H_bathy, st0.H_bathy.name, st0.H_bathy.dims, st0.H_bathy.units),
        eta=Field(eta_b, st0.eta.name, st0.eta.dims, st0.eta.units),
        eta_before=Field(eta_b, st0.eta.name, st0.eta.dims, st0.eta.units),
        u=Field(u_arr, st0.u.name, st0.u.dims, st0.u.units),
        v=Field(v_arr, st0.v.name, st0.v.dims, st0.v.units),
        u_before=Field(u_arr, st0.u.name, st0.u.dims, st0.u.units),
        v_before=Field(v_arr, st0.v.name, st0.v.dims, st0.v.units),
    )
    return model, state, eta_b, eta_a


_STEP_SPY_CACHE: dict[str, object] | None = None


def _run_production_step_spies() -> dict[str, object]:
    global _STEP_SPY_CACHE
    if _STEP_SPY_CACHE is not None:
        return _STEP_SPY_CACHE

    model_on, state, eta_b, eta_a = _build_stage2_fixture(stored_flag=True)
    model_off, _, _, _ = _build_stage2_fixture(stored_flag=False)
    zc = model_on.z_coord
    grid = model_on.grid
    dt = 10800.0
    Hu_avg = jnp.sum(state.u.data, axis=-1)
    Hv_avg = jnp.sum(state.v.data, axis=-1)

    calls_on: dict[int, tuple] = {}
    calls_off: dict[int, tuple] = {}
    real_transport = omlc._nemo_ws_stage_transport

    bt_override = (
        eta_a,
        state.uu_b.data,
        state.vv_b.data,
        Hu_avg,
        Hv_avg,
    )

    def make_spy(target_dict):
        def _spy(stage_velocity, h_stage, stage_index, **kwargs):
            out = real_transport(stage_velocity, h_stage, stage_index, **kwargs)
            if stage_index in (0, 1) and kwargs.get("eta_before") is not None:
                def _store(vals, idx=int(stage_index)):
                    target_dict[idx] = tuple(
                        None if v is None else np.asarray(v, dtype=np.float64)
                        for v in vals
                    )

                jax.debug.callback(
                    _store,
                    tuple(0.0 if v is None else v for v in out),
                    ordered=True,
                )
            return out

        return _spy

    try:
        omlc._nemo_ws_stage_transport = make_spy(calls_on)
        m_on = omlc.LatLonCGridOceanModel(
            grid,
            zc,
            model_on.config,
            _nemo_ws_test_hooks=omlc._NEMOWSRK3TestHooks(
                stage_barotropic_output_override=bt_override
            ),
        )
        _ = m_on.step(state, dt)
        omlc._nemo_ws_stage_transport = make_spy(calls_off)
        m_off = omlc.LatLonCGridOceanModel(
            grid,
            zc,
            model_off.config,
            _nemo_ws_test_hooks=omlc._NEMOWSRK3TestHooks(
                stage_barotropic_output_override=bt_override
            ),
        )
        _ = m_off.step(state, dt)
    finally:
        omlc._nemo_ws_stage_transport = real_transport

    _STEP_SPY_CACHE = {"on": calls_on, "off": calls_off}
    return _STEP_SPY_CACHE


def test_stage2_transport_uses_stored_h_s2_kmm_and_fails_on_revert() -> None:
    """Exercise ``_nemo_ws_stage_transport`` and the production ``_step_impl``
    Stage-2 call site (``ocean_model_latlon_cgrid.py:2293-2295`` and ``:9570-9572``).
    Reverting either ``stored_h_stage`` in ``_nemo_ws_stage_transport`` or
    ``_h_s2_kmm`` / ``stored_h_stage`` at the Stage-2 call site in ``_step_impl``
    makes this test fail.
    """
    model_on, state, eta_b, eta_a = _build_stage2_fixture(stored_flag=True)

    # 1. Verify _nemo_ws_stage_transport Stage-2 call directly with the exact kt=10 cell
    zc = model_on.z_coord
    grid = model_on.grid
    cfg = model_on.config
    dt = 10800.0
    h_ref = jnp.asarray(zc.nemo_e3t_0, dtype=jnp.float64)
    umask = state.u_mask.data[..., None] * jnp.ones_like(state.u.data)
    vmask = state.v_mask.data[..., None] * jnp.ones_like(state.v.data)

    qt_13 = nemo_r3t_rk3_stage1_stretch(zc, eta_b, eta_a, state.H_bathy.data)
    h_s2_kmm = omlc.nemo_source_round(h_ref * qt_13[..., None])
    r2_3 = np.float64(2.0) / np.float64(3.0)
    r1_3 = np.float64(1.0) / np.float64(3.0)
    eta_13 = omlc.nemo_source_round(r2_3 * eta_b + r1_3 * eta_a)
    h_live_13 = omlc.nemo_qco_live_t_thickness(
        eta_13, state.H_bathy.data, zc, jnp.float64, e3t_0=h_ref
    )

    assert _hex64(np.asarray(h_s2_kmm)[1, 1, 3]) == "0x40234d4bd6440687"
    assert _hex64(np.asarray(h_live_13)[1, 1, 3]) == "0x40234d4bd6440686"

    Hu_avg = jnp.sum(state.u.data, axis=-1)
    Hv_avg = jnp.sum(state.v.data, axis=-1)

    g1_stored = omlc._nemo_ws_stage_transport(
        (state.u.data, state.v.data),
        h_s2_kmm,
        1,
        eta_stage=eta_13,
        stored_h_stage=True,
        h_ref=h_ref,
        Hu_avg=Hu_avg,
        Hv_avg=Hv_avg,
        u_mask_3d=umask,
        v_mask_3d=vmask,
        grid=grid,
        z_coord=zc,
        H_bathy=state.H_bathy.data,
        config=cfg,
        dt=dt,
        eta_before=eta_b,
        eta_after=eta_a,
    )
    g1_rebuilt = omlc._nemo_ws_stage_transport(
        (state.u.data, state.v.data),
        h_s2_kmm,
        1,
        eta_stage=eta_13,
        stored_h_stage=False,
        h_ref=h_ref,
        Hu_avg=Hu_avg,
        Hv_avg=Hv_avg,
        u_mask_3d=umask,
        v_mask_3d=vmask,
        grid=grid,
        z_coord=zc,
        H_bathy=state.H_bathy.data,
        config=cfg,
        dt=dt,
        eta_before=eta_b,
        eta_after=eta_a,
    )
    e3t_st = np.asarray(g1_stored[3], dtype=np.float64)[1, 1, 3]
    e3t_rb = np.asarray(g1_rebuilt[3], dtype=np.float64)[1, 1, 3]
    assert _hex64(e3t_st) == "0x40234d4bd6440687"
    assert _hex64(e3t_rb) == "0x40234d4bd6440686"

    # Feed into the exact kt=10 Stage-2 tracer division
    h_scaled = _from_hex64("0x3f15e5f2c773b397")
    v_scaled = _from_hex64("0xbebdbf795f7c1fc3")
    term1 = _from_hex64("0xbff520edea197e7a")
    qt_s1_val = np.asarray(qt_13, dtype=np.float64)[1, 1]
    qt_s2_val = np.asarray(
        omlc._nemo_ws_stage_r3t_stretch(
            cfg, 3, lambda: (eta_b, eta_a), zc, state.H_bathy.data
        ),
        dtype=np.float64,
    )[1, 1]
    rDt = np.float64(5400.0)
    rhs_st = -(h_scaled / e3t_st) - (v_scaled / e3t_st)
    rhs_rb = -(h_scaled / e3t_rb) - (v_scaled / e3t_rb)
    T_s2_st = (term1 + (rDt * qt_s1_val) * rhs_st) / qt_s2_val
    T_s2_rb = (term1 + (rDt * qt_s1_val) * rhs_rb) / qt_s2_val
    assert _hex64(T_s2_st) == "0xbff663da73f853c6"
    assert _hex64(T_s2_rb) == "0xbff663da73f853c7"

    # 2. Exercise the production _step_impl Stage-2 call site (:9570-9572)
    spies = _run_production_step_spies()
    h_prod_on = spies["on"][1][3]
    h_prod_off = spies["off"][1][3]
    assert _hex64(h_prod_on[1, 1, 3]) == "0x40234d4bd6440687"
    assert _hex64(h_prod_off[1, 1, 3]) == "0x40234d4bd6440686"


def test_stage1_stage3_and_legacy_caller_stay_bit_identical() -> None:
    """Prove that Stage 1 (``:9341``, ``stage_index=0``), Stage 3 (``:9641``,
    ``stage_index=2``), and the legacy caller (``:10145``, across
    ``stage_index=0, 1, 2``) remain bit-identical before vs after the Stage-2
    fix, and that ``stored_h_stage=True`` is refused for ``stage_index != 1``.
    """
    model_on, state, eta_b, eta_a = _build_stage2_fixture(stored_flag=True)
    model_off, _, _, _ = _build_stage2_fixture(stored_flag=False)
    zc = model_on.z_coord
    grid = model_on.grid
    dt = 10800.0
    h_ref = jnp.asarray(zc.nemo_e3t_0, dtype=jnp.float64)
    umask = state.u_mask.data[..., None] * jnp.ones_like(state.u.data)
    vmask = state.v_mask.data[..., None] * jnp.ones_like(state.v.data)
    Hu_avg = jnp.sum(state.u.data, axis=-1)
    Hv_avg = jnp.sum(state.v.data, axis=-1)

    # Guard refuses stored_h_stage=True for Stage 1 (0) and Stage 3 (2)
    for bad_stage in (0, 2):
        with pytest.raises(ValueError, match="Stage-2"):
            omlc._nemo_ws_stage_transport(
                (state.u.data, state.v.data),
                h_ref,
                bad_stage,
                eta_stage=eta_b,
                stored_h_stage=True,
                h_ref=h_ref,
                Hu_avg=Hu_avg,
                Hv_avg=Hv_avg,
                u_mask_3d=umask,
                v_mask_3d=vmask,
                grid=grid,
                z_coord=zc,
                H_bathy=state.H_bathy.data,
                config=model_on.config,
                dt=dt,
                eta_before=eta_b,
                eta_after=eta_a,
            )

    # Stage 1 (:9341, stage_index=0) in production _step_impl is bit-identical across all 12 tuple slots
    spies = _run_production_step_spies()
    for slot_idx, (a, b) in enumerate(
        zip(spies["on"][0], spies["off"][0], strict=True)
    ):
        if a is None or b is None:
            assert a is None and b is None
        else:
            arr_a = np.asarray(a, dtype=np.float64)
            arr_b = np.asarray(b, dtype=np.float64)
            assert np.array_equal(arr_a.view(np.uint64), arr_b.view(np.uint64)), (
                f"Stage 1 slot {slot_idx} moved"
            )

    # Stage 3 (stage_index=2) when given identical Stage-3 inputs is bit-identical across all 12 slots
    r1_2 = np.float64(0.5)
    eta_12 = omlc.nemo_source_round(r1_2 * (eta_b + eta_a))
    g2_a = omlc._nemo_ws_stage_transport(
        (state.u.data, state.v.data),
        h_ref,
        2,
        eta_stage=eta_12,
        h_ref=h_ref,
        Hu_avg=Hu_avg,
        Hv_avg=Hv_avg,
        u_mask_3d=umask,
        v_mask_3d=vmask,
        grid=grid,
        z_coord=zc,
        H_bathy=state.H_bathy.data,
        config=model_on.config,
        dt=dt,
        eta_before=eta_b,
        eta_after=eta_a,
    )
    g2_b = omlc._nemo_ws_stage_transport(
        (state.u.data, state.v.data),
        h_ref * 1.125,  # perturbed h_stage argument is ignored on Stage 3
        2,
        eta_stage=eta_12,
        h_ref=h_ref,
        Hu_avg=Hu_avg,
        Hv_avg=Hv_avg,
        u_mask_3d=umask,
        v_mask_3d=vmask,
        grid=grid,
        z_coord=zc,
        H_bathy=state.H_bathy.data,
        config=model_off.config,
        dt=dt,
        eta_before=eta_b,
        eta_after=eta_a,
    )
    for slot_idx, (a, b) in enumerate(zip(g2_a, g2_b, strict=True)):
        if a is None or b is None:
            assert a is None and b is None
        else:
            arr_a = np.asarray(a, dtype=np.float64)
            arr_b = np.asarray(b, dtype=np.float64)
            assert np.array_equal(arr_a.view(np.uint64), arr_b.view(np.uint64)), (
                f"Stage 3 slot {slot_idx} moved"
            )

    # Legacy caller (:10144-10162) is bit-identical across all 3 stages
    src_lines = Path(omlc.__file__).read_text(encoding="utf-8").splitlines()
    legacy_block = textwrap.dedent("\n".join(src_lines[10130:10162]))
    assert "_nemo_ws_stage_transport(" in legacy_block

    def _eval_legacy_caller(m_obj):
        ns = {
            **omlc.__dict__,
            "h_k_old": h_ref,
            "h_k_new": h_ref * 1.05,
            "state": state,
            "state_new": state._replace(
                eta=Field(eta_a, state.eta.name, state.eta.dims, state.eta.units)
            ),
            "_zc": zc,
            "_cfg_b": m_obj.config,
            "Hu_avg": Hu_avg,
            "Hv_avg": Hv_avg,
            "u_mask_3d_tracer": umask,
            "v_mask_3d_tracer": vmask,
            "_grid": grid,
            "dt": dt,
            "self": m_obj,
            "_nemo_ws_velocity_stages": (
                (state.u.data, state.v.data),
                (state.u.data * 0.9, state.v.data * 0.9),
                (state.u.data * 0.8, state.v.data * 0.8),
            ),
        }
        exec(compile(legacy_block, "<legacy_caller_10145>", "exec"), ns)
        return ns["_nemo_ws_stage_transport_geometry"]

    leg_calls_on = _eval_legacy_caller(model_on)
    leg_calls_off = _eval_legacy_caller(model_off)
    assert len(leg_calls_on) == len(leg_calls_off) == 3
    for stg_idx, (tup_on, tup_off) in enumerate(
        zip(leg_calls_on, leg_calls_off, strict=True)
    ):
        for slot_idx, (a, b) in enumerate(zip(tup_on, tup_off, strict=True)):
            if a is None or b is None:
                assert a is None and b is None
            else:
                arr_a = np.asarray(a, dtype=np.float64)
                arr_b = np.asarray(b, dtype=np.float64)
                assert np.array_equal(
                    arr_a.view(np.uint64), arr_b.view(np.uint64)
                ), f"Legacy caller stage {stg_idx} slot {slot_idx} moved"


def test_stage2_stored_thickness_eager_equals_jit_with_dynamic_args() -> None:
    """Checklist M4: ``eager == jit`` under ``with jax.disable_jit():`` on the
    entire eager call and all dynamic inputs passed as JIT arguments, plus a
    reverted/unbarriered control that fails ``pytest.raises(AssertionError)``.
    """
    model, state, eta_b, eta_a = _build_stage2_fixture(stored_flag=True)
    zc = model.z_coord
    grid = model.grid
    cfg = model.config
    dt = 10800.0
    h_ref = jnp.asarray(zc.nemo_e3t_0, dtype=jnp.float64)
    tmask = jnp.where(zc.is_active > 0, 1.0, 0.0)
    umask = state.u_mask.data[..., None] * jnp.ones_like(state.u.data)
    vmask = state.v_mask.data[..., None] * jnp.ones_like(state.v.data)
    Hu_avg = jnp.sum(state.u.data, axis=-1)
    Hv_avg = jnp.sum(state.v.data, axis=-1)

    def _stage2_kernel(
        u_in, v_in, uu_b_in, vv_b_in, eta_b_in, eta_a_in, h_ref_in, Hu_in, Hv_in, T_b_in, T_s1_in
    ):
        qt_b = nemo_r3t_stretch(
            zc, eta_b_in, state.H_bathy.data, evaluation="nemo_reciprocal"
        )
        qt_13 = nemo_r3t_rk3_stage1_stretch(
            zc, eta_b_in, eta_a_in, state.H_bathy.data
        )
        qt_12 = omlc._nemo_ws_stage_r3t_stretch(
            cfg, 3, lambda: (eta_b_in, eta_a_in), zc, state.H_bathy.data
        )
        h_s2 = omlc.nemo_source_round(h_ref_in * qt_13[..., None])
        r2_3 = np.float64(2.0) / np.float64(3.0)
        r1_3 = np.float64(1.0) / np.float64(3.0)
        eta_13 = omlc.nemo_source_round(
            omlc.nemo_source_round(r2_3 * eta_b_in)
            + omlc.nemo_source_round(r1_3 * eta_a_in)
        )
        g1 = omlc._nemo_ws_stage_transport(
            (u_in, v_in),
            h_s2,
            1,
            eta_stage=eta_13,
            stored_h_stage=True,
            h_ref=h_ref_in,
            Hu_avg=Hu_in,
            Hv_avg=Hv_in,
            u_mask_3d=umask,
            v_mask_3d=vmask,
            grid=grid,
            z_coord=zc,
            H_bathy=state.H_bathy.data,
            config=cfg,
            dt=dt,
            eta_before=eta_b_in,
            eta_after=eta_a_in,
            literal_wzv=True,
            barotropic_velocity=(uu_b_in, vv_b_in),
            nemo_stage_pair=(1, eta_b_in, eta_a_in),
        )
        rhs = omlc._nemo_cen2_tracer_rhs(
            T_s1_in, g1[7], g1[8], g1[2], g1[3], tmask, grid
        )
        r = omlc.nemo_source_round
        q_after = jnp.where(tmask > 0.5, qt_12[..., None], 1.0)
        T_s2 = jnp.where(
            tmask > 0.5,
            r(
                r(
                    r(qt_b[..., None] * T_b_in)
                    + r(r((dt / 2.0) * qt_13[..., None]) * rhs)
                )
                / q_after
            ),
            T_b_in,
        )
        return g1[3], rhs, T_s2

    T_s1 = state.T.data + 0.015
    args = (
        state.u.data,
        state.v.data,
        state.uu_b.data,
        state.vv_b.data,
        eta_b,
        eta_a,
        h_ref,
        Hu_avg,
        Hv_avg,
        state.T.data,
        T_s1,
    )
    with jax.disable_jit():
        h_eager, rhs_eager, T_eager = _stage2_kernel(*args)
    h_jit, rhs_jit, T_jit = jax.jit(_stage2_kernel)(*args)

    for a, b in ((h_eager, h_jit), (rhs_eager, rhs_jit), (T_eager, T_jit)):
        arr_a = np.asarray(a, dtype=np.float64)
        arr_b = np.asarray(b, dtype=np.float64)
        assert np.array_equal(arr_a.view(np.uint64), arr_b.view(np.uint64))

    # Reverted control (stored_h_stage=False) must fail bit-identity against the stored-thickness kernel
    def _reverted_kernel(
        u_in, v_in, uu_b_in, vv_b_in, eta_b_in, eta_a_in, h_ref_in, Hu_in, Hv_in, T_b_in, T_s1_in
    ):
        qt_b = nemo_r3t_stretch(
            zc, eta_b_in, state.H_bathy.data, evaluation="nemo_reciprocal"
        )
        qt_13 = nemo_r3t_rk3_stage1_stretch(
            zc, eta_b_in, eta_a_in, state.H_bathy.data
        )
        qt_12 = omlc._nemo_ws_stage_r3t_stretch(
            cfg, 3, lambda: (eta_b_in, eta_a_in), zc, state.H_bathy.data
        )
        h_s2 = omlc.nemo_source_round(h_ref_in * qt_13[..., None])
        r2_3 = np.float64(2.0) / np.float64(3.0)
        r1_3 = np.float64(1.0) / np.float64(3.0)
        eta_13 = omlc.nemo_source_round(
            omlc.nemo_source_round(r2_3 * eta_b_in)
            + omlc.nemo_source_round(r1_3 * eta_a_in)
        )
        g1 = omlc._nemo_ws_stage_transport(
            (u_in, v_in),
            h_s2,
            1,
            eta_stage=eta_13,
            stored_h_stage=False,
            h_ref=h_ref_in,
            Hu_avg=Hu_in,
            Hv_avg=Hv_in,
            u_mask_3d=umask,
            v_mask_3d=vmask,
            grid=grid,
            z_coord=zc,
            H_bathy=state.H_bathy.data,
            config=cfg,
            dt=dt,
            eta_before=eta_b_in,
            eta_after=eta_a_in,
            literal_wzv=True,
            barotropic_velocity=(uu_b_in, vv_b_in),
            nemo_stage_pair=(1, eta_b_in, eta_a_in),
        )
        rhs = omlc._nemo_cen2_tracer_rhs(
            T_s1_in, g1[7], g1[8], g1[2], g1[3], tmask, grid
        )
        r = omlc.nemo_source_round
        q_after = jnp.where(tmask > 0.5, qt_12[..., None], 1.0)
        T_s2 = jnp.where(
            tmask > 0.5,
            r(
                r(
                    r(qt_b[..., None] * T_b_in)
                    + r(r((dt / 2.0) * qt_13[..., None]) * rhs)
                )
                / q_after
            ),
            T_b_in,
        )
        return g1[3], rhs, T_s2

    h_rev, _, _ = jax.jit(_reverted_kernel)(*args)
    with pytest.raises(AssertionError):
        assert np.array_equal(
            np.asarray(h_jit, dtype=np.float64).view(np.uint64),
            np.asarray(h_rev, dtype=np.float64).view(np.uint64),
        )
