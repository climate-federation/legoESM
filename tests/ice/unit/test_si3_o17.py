"""Structural and fail-closed tests for the new O17 production entry."""

from pathlib import Path
import struct

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.ice import sea_ice
from legoesm.ice import si3_o17 as o17_mod
from legoesm.ice.config import SeaIceConfig, build_si3_orca2_o17_config
from legoesm.ice.si3_o17 import SI3O17Card, SI3O17LiveForcing, SI3O17StepState, si3_o17_step
from legoesm.ice.si3_orca2_state import SI3ORCA2StateConfig

from scripts.validate.ocean_fidelity.testcases import (
    nemo_si3_o17_ad_probe,
    nemo_si3_o17_composition_scorer,
)
from scripts.validate.ocean_fidelity.testcases.nemo_si3_o17_ad_probe import (
    DEFAULT_MASTER,
    _case,
)
from scripts.validate.ocean_fidelity.testcases.nemo_si3_o17_acquisition.preflight import (
    ORDER,
    preflight,
)

REPO = Path(__file__).resolve().parents[3]


def test_o17_typed_pytree_boundary_and_explicit_card() -> None:
    assert SI3O17LiveForcing._fields[:4] == ("kt", "sss_m", "sst_m", "rn_amax_2d")
    assert not (
        {"redistribution_state", "correction_state", "thermo", "fluxes"}
        & set(SI3O17LiveForcing._fields)
    )
    assert SI3O17StepState._fields == ("ice", "aevp")
    assert SI3O17Card._fields[-1] == "core2_precipitation_closure"
    assert callable(si3_o17_step)


def test_o17_card_refuses_core2_closure_before_any_inference() -> None:
    card = SI3O17Card(*([None] * (len(SI3O17Card._fields) - 1)), True)
    with pytest.raises(ValueError, match="core2_precipitation_closure=False"):
        build_si3_orca2_o17_config(card)


def test_o17_call_order_preflight_and_swap_drop_plants() -> None:
    report = preflight(REPO)
    assert report["call_order"] == list(ORDER)
    assert report["steps"] == 100
    assert any("o4_orca2_prather_gate" in name for name in report["imported_component_tools"])
    for plant in ("swap-rheology-advection", "drop-redistribution"):
        with pytest.raises(SystemExit, match="STATUS PLANT-FIRED"):
            preflight(REPO, plant)


def test_o17_public_production_dispatch_calls_composition(monkeypatch) -> None:
    state_cfg = SI3ORCA2StateConfig.for_orca2_sas_ice(ln_icedyn=True, ln_icethd=True)
    card = object()
    config = SeaIceConfig(
        dynamics="none",
        transport="none",
        thermo_scheme="si3_orca2_o2",
        ice_constants=state_cfg.ice_constants,
        si3_orca2=state_cfg,
        core2_precipitation_closure=False,
        si3_o17=card,
    )
    entry = SI3O17StepState(None, None)
    forcing = SI3O17LiveForcing(1, *([None] * (len(SI3O17LiveForcing._fields) - 1)))
    expected = (entry, object())
    calls = []

    def planted(step_entry, live_forcing, actual_card):
        calls.append((step_entry, live_forcing, actual_card))
        return expected

    monkeypatch.setattr(sea_ice, "si3_o17_step", planted)
    actual = sea_ice.step_sea_ice(entry, forcing, None, None, None, config, 0.0, state_cfg.rDt_ice)
    assert actual == expected
    assert calls == [(entry, forcing, card)]


def test_o17_record_reader_is_self_describing_and_rejects_truncation(tmp_path) -> None:
    assert nemo_si3_o17_composition_scorer.VALID_LABELS == ("INDEPENDENT", "GIVEN-NEMO-ENTRY")
    record = tmp_path / "o17_kind22_step0001_rank0000.bin"
    field = b"v_i".ljust(32, b" ")
    end = b"__END__".ljust(32, b" ")
    payload = (
        b"O17S0001" + struct.pack("=i8i", 1, 22, 1, 0, 6, 5, 5, 10, 5)
        + field + struct.pack("=i4i", 2, 2, 3, 0, 0)
        + np.arange(6, dtype=np.float64).reshape(2, 3, order="F").tobytes(order="F")
        + end + struct.pack("=i4i", -1, 0, 0, 0, 0)
    )
    record.write_bytes(payload)
    parsed = nemo_si3_o17_composition_scorer.read_record(record)
    assert parsed["meta"]["kind"] == 22
    np.testing.assert_array_equal(parsed["fields"]["v_i"], np.arange(6).reshape(2, 3, order="F"))
    record.write_bytes(payload[:143])  # one byte short of v_i's six float64 values
    with pytest.raises(ValueError, match="truncated"):
        nemo_si3_o17_composition_scorer.read_record(record)


def test_o17_unified_acquisition_routes_real_scorer_only_after_one_master_run() -> None:
    run = (REPO / "scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_acquisition/run.sh").read_text()
    scorer = (REPO / "scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_composition_scorer.py").read_text()
    assert run.count('"$master" --run "$evidence"') == 1
    assert "nemo_si3_o4_orca2_prather_acquisition/run.sh" not in run
    assert "ACQUISITION_NEEDED" not in scorer
    assert "si3_o17_step(entry, forcing, card)" in scorer
    assert 'touch "$EVIDENCE/RUN_COMPLETE"' in (
        REPO / "scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o16_master_acquisition/run.sh"
    ).read_text()


def test_o17_step_executes_active_ice_eager_jit_and_numerical_plants() -> None:
    """Run the real composition and prove order/fold/forcing features alter numbers."""
    entry, forcing, card = _case(DEFAULT_MASTER)
    card = card._replace(aevp=card.aevp._replace(n_subcycles=1))

    def run(step_entry, dynamic_forcing, **plants):
        live = SI3O17LiveForcing(1, *dynamic_forcing)
        # O6 redistribution has its own admitted eager/JIT numerical suite;
        # omit that already-covered kernel so this composition gate can reach
        # and compile the O7 forcing boundary on the shared test host.
        plants.setdefault("_plant_drop_redistribution", True)
        return si3_o17_step(
            step_entry, live, card, _test_return_surface_probe=True, **plants
        )

    dynamic = forcing[1:]
    with jax.disable_jit():
        eager = run(entry, dynamic)

    compiled = jax.jit(run)(entry, dynamic)
    for got, want in zip(compiled, eager, strict=True):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=0.0, atol=1.0e-12, equal_nan=True)
    assert np.count_nonzero(np.asarray(eager[2])) > 0

    with jax.disable_jit():
        swapped = run(entry, dynamic, _plant_swap_rheology_advection=True)
        no_fold = run(entry, dynamic, _plant_drop_deformation_fold=True)
    assert any(
        not np.array_equal(np.asarray(got), np.asarray(want))
        for got, want in zip(swapped[:2], eager[:2], strict=True)
    )
    assert any(
        not np.array_equal(np.asarray(got), np.asarray(want))
        for got, want in zip(no_fold[4:6], eager[4:6], strict=True)
    )
    half_frq = forcing._replace(surface=forcing.surface._replace(frq_m=forcing.surface.frq_m * 0.5))
    with jax.disable_jit():
        half_frq_fluxes = run(entry, half_frq[1:])[6:8]
    assert any(
        not np.array_equal(np.asarray(got), np.asarray(want))
        for got, want in zip(half_frq_fluxes, eager[6:8], strict=True)
    )

    with jax.disable_jit():
        tail_eager = o17_mod._thermodynamic_tail(entry.ice, card)
    tail_compiled = jax.jit(lambda state: o17_mod._thermodynamic_tail(state, card))(entry.ice)
    np.testing.assert_allclose(
        np.asarray(tail_compiled.oa_i), np.asarray(tail_eager.oa_i), rtol=0.0, atol=1.0e-12
    )
    np.testing.assert_array_equal(np.asarray(tail_compiled.u_ice), np.asarray(tail_eager.u_ice))
    np.testing.assert_array_equal(np.asarray(tail_compiled.v_ice), np.asarray(tail_eager.v_ice))

    with pytest.raises(ValueError, match="ln_ice_embd=F"):
        si3_o17_step(entry, forcing, card._replace(ln_ice_embd=True))


def test_o17_inactive_category_open_water_snowfall_nan_fix_and_plant(monkeypatch) -> None:
    """Prove O17 open-water snowfall has 0 NaNs now and fails under the 96bb44f0aeee unguarded plant."""
    entry, forcing, card = _case(DEFAULT_MASTER)
    card_fast = card._replace(aevp=card.aevp._replace(n_subcycles=1))
    z = jnp.zeros_like(entry.ice.u_ice)
    entry_still = SI3O17StepState(
        ice=entry.ice._replace(u_ice=z, v_ice=z),
        aevp=entry.aevp._replace(u_ice_u=z, v_ice_v=z),
    )
    forcing_still = forcing._replace(
        aevp_forcing=forcing.aevp_forcing._replace(
            air_stress_u_t=z, air_stress_v_t=z, ocean_u_u=z, ocean_v_v=z
        ),
        surface=forcing.surface._replace(
            utau=jnp.zeros_like(forcing.surface.utau),
            vtau=jnp.zeros_like(forcing.surface.vtau),
            ssu_m=z,
            ssv_m=z,
        ),
    )

    step_fixed, trace_fixed = si3_o17_step(
        entry_still, forcing_still, card_fast, _plant_drop_redistribution=True
    )
    state_fixed = step_fixed.ice
    fluxes_fixed = trace_fixed.finalize.final.fluxes
    out_fixed = trace_fixed.finalize.final.outputs
    stress_fixed = trace_fixed.finalize.stress
    h = card_fast.state.nn_hls
    owned = (slice(h, -h), slice(h, -h))

    for field in state_fixed._fields:
        val = getattr(state_fixed, field)
        if hasattr(val, "shape") and val.ndim >= 2:
            assert int(np.isnan(np.asarray(val[owned])).sum()) == 0, f"NaN in owned state.{field}"
    for field in fluxes_fixed._fields:
        assert int(np.isnan(np.asarray(getattr(fluxes_fixed, field))).sum()) == 0, f"NaN in fluxes.{field}"
    for field in out_fixed._fields:
        assert int(np.isnan(np.asarray(getattr(out_fixed, field))).sum()) == 0, f"NaN in outputs.{field}"
    for field in stress_fixed._fields:
        assert int(np.isnan(np.asarray(getattr(stress_fixed, field))).sum()) == 0, f"NaN in stress.{field}"

    nine_fields = (
        "hfx_spr",
        "hfx_sub",
        "wfx_err_sub",
        "wfx_snw_sub",
        "wfx_spr",
        "wfx_sub",
        "emp",
        "fwfice",
        "qns",
    )
    for name in ("hfx_spr", "hfx_sub", "wfx_err_sub", "wfx_snw_sub", "wfx_spr", "wfx_sub", "fwfice"):
        arr = getattr(out_fixed, name) if hasattr(out_fixed, name) else getattr(fluxes_fixed, name)
        assert float(arr[0, 0]) == 0.0, f"Expected 0.0 at open-water (0,0) for {name}"
    assert float(out_fixed.emp[0, 0]) == float(trace_fixed.surface_stress.blk_ice_2.emp_oce[0, 0])
    assert float(out_fixed.emp[0, 0]) == pytest.approx(2.0e-8, rel=1.0e-15, abs=0.0)
    assert np.isfinite(float(out_fixed.qns[0, 0]))
    np.testing.assert_array_equal(
        np.asarray(state_fixed.t_si[2, 2, :]),
        np.full(card_fast.state.jpl, card_fast.constants.T0, dtype=np.float64),
    )

    orig_thermo_entries = o17_mod._thermo_entries

    def planted_unguarded_thermo_entries(*args, **kwargs):
        entries = orig_thermo_entries(*args, **kwargs)
        return tuple(
            dict(e, active=jnp.ones_like(e["active"], dtype=jnp.bool_))
            for e in entries
        )

    monkeypatch.setattr(o17_mod, "_thermo_entries", planted_unguarded_thermo_entries)
    step_plant, trace_plant = si3_o17_step(
        entry_still, forcing_still, card_fast, _plant_drop_redistribution=True
    )
    state_plant = step_plant.ice
    fluxes_plant = trace_plant.finalize.final.fluxes
    out_plant = trace_plant.finalize.final.outputs

    for name in nine_fields:
        arr_plant = getattr(out_plant, name) if hasattr(out_plant, name) else getattr(fluxes_plant, name)
        assert np.isnan(float(arr_plant[0, 0])), f"Planted unguarded path must produce NaN in {name}[0,0]"
        arr_fixed = getattr(out_fixed, name) if hasattr(out_fixed, name) else getattr(fluxes_fixed, name)
        np.testing.assert_array_equal(
            np.asarray(arr_fixed[1, 0], dtype=np.float64).view(np.uint64),
            np.asarray(arr_plant[1, 0], dtype=np.float64).view(np.uint64),
        )
    assert not np.allclose(np.asarray(state_plant.t_si[2, 2, :]), card_fast.constants.T0)

    for field in ("a_i", "v_i", "v_s", "sv_i", "oa_i", "t_su", "t_si", "e_i", "e_s", "szv_i"):
        np.testing.assert_array_equal(
            np.asarray(getattr(state_fixed, field)[3, 2], dtype=np.float64).view(np.uint64),
            np.asarray(getattr(state_plant, field)[3, 2], dtype=np.float64).view(np.uint64),
        )


def test_o17_newly_guarded_outputs_reverse_mode_matches_central_differences_and_plants() -> None:
    """Prove per-entry-relative central FD on all newly guarded outputs and fail under plants."""
    from legoesm import constants
    from legoesm.ice.si3_thd_pnd import si3_ice_thd_o12_stages

    entry, forcing, card = _case(DEFAULT_MASTER)
    s0 = entry.ice
    h = card.state.nn_hls
    owned = (slice(h, -h), slice(h, -h))
    sf = forcing.surface

    v_s0 = s0.v_s.at[3, 2, 0].set(0.0)
    e_s0 = s0.e_s.at[3, 2, :, 0].set(0.0)
    t_su0 = s0.t_su.at[3, 2, 1].set(card.constants.T0)
    t_si0 = s0.t_si.at[owned].set(card.constants.T0)
    eqv0 = o17_mod.si3_var_glo2eqv(
        v_i=s0.v_i,
        v_s=v_s0,
        sv_i=s0.sv_i,
        a_i=s0.a_i,
        oa_i=s0.oa_i,
        a_ip=s0.a_ip,
        v_ip=s0.v_ip,
        v_il=s0.v_il,
        e_s=e_s0,
        e_i=s0.e_i,
        szv_i=s0.szv_i,
        kn=2,
        hi_max_top=99.0,
        sss_m=forcing.sss_m,
        config=card.state,
    )
    agg0 = o17_mod.si3_var_agg(
        v_i=s0.v_i,
        v_s=eqv0.v_s,
        sv_i=s0.sv_i,
        a_i=eqv0.a_i,
        oa_i=s0.oa_i,
        a_ip=s0.a_ip,
        v_ip=s0.v_ip,
        v_il=s0.v_il,
        t_su=t_su0,
        t_si=t_si0,
        e_s=e_s0,
        e_i=s0.e_i,
        szv_i=s0.szv_i,
        a_ip_eff=eqv0.a_ip_eff,
        t_i=eqv0.t_i,
        t_s=eqv0.t_s,
        kn=2,
        config=card.state,
    )
    s0 = s0._replace(
        v_s=eqv0.v_s, e_s=e_s0, t_su=t_su0, t_si=t_si0, equivalent=eqv0, aggregated=agg0
    )

    t_bo_celsius = o17_mod.si3_eos_fzp(forcing.sss_m, kbnd=0, nn_hls=h)
    t_bo = (t_bo_celsius[owned] + card.constants.T0) * sf.smask0 + card.constants.T0 * (
        1.0 - sf.smask0
    )
    b1 = o17_mod.si3_blk_ice_1(
        wndi=sf.wndi,
        wndj=sf.wndj,
        theta_air_zt=sf.theta_air_zt,
        q_air_zt=sf.q_air_zt,
        slp=sf.slp,
        tm_su=s0.aggregated.tm_su[owned],
        rhoa=sf.rhoa,
        config=card.surface,
    )
    utau_ice, vtau_ice = o17_mod.si3_ice_sbc_tau_lbc(b1.utau_ice, b1.vtau_ice, card.fold)
    alb = o17_mod.si3_ice_alb(
        t_su=s0.t_su[owned],
        h_i=s0.equivalent.h_i[owned],
        h_s=s0.equivalent.h_s[owned],
        a_ip_eff=s0.equivalent.a_ip_eff[owned],
        h_ip=s0.equivalent.h_ip[owned],
        h_il=s0.equivalent.h_il[owned],
        cloud_fra=sf.cloud_fra,
        smask0=sf.smask0,
        config=card.surface,
    )
    a_i_b = s0.a_i[owned]
    at_i_b = o17_mod.fortran_sum_categories(a_i_b, axis=-1)
    b2 = o17_mod.si3_blk_ice_2(
        t_su=s0.t_su[owned],
        h_s=s0.equivalent.h_s[owned],
        h_i=s0.equivalent.h_i[owned],
        alb_ice=alb,
        theta_air_zt=sf.theta_air_zt,
        q_air_zt=sf.q_air_zt,
        slp=sf.slp,
        qlw=sf.qlw,
        precip=sf.precip,
        snow=sf.snow,
        qsr=sf.qsr,
        rhoa=sf.rhoa,
        wndm_ice=b1.wndm_ice,
        Ch_ice=b1.Ch_ice,
        Ce_ice=b1.Ce_ice,
        theta_zu_i=b1.theta_zu_i,
        q_zu_i=b1.q_zu_i,
        smask0=sf.smask0,
        sst_m=forcing.sst_m[owned],
        emp=sf.emp,
        at_i_b=at_i_b,
        a_i_b=a_i_b,
        qns_oce=sf.qns_oce,
        qsr_oce=sf.qsr_oce,
        cloud_fra=sf.cloud_fra,
        config=card.surface,
    )
    b2 = b2._replace(qns_ice=b2.qns_ice.at[1, 0, 1].set(120.0))
    other = o17_mod.si3_ice_flx_other(
        **o17_mod.orca2_flx_other_neighbours(
            u_ice=s0.u_ice, v_ice=s0.v_ice, ssu_m=sf.ssu_m, ssv_m=sf.ssv_m, nn_hls=h
        ),
        utau=sf.utau,
        vtau=sf.vtau,
        smask0=sf.smask0,
        at_i=s0.aggregated.at_i[owned],
        at_i_b=at_i_b,
        vt_i=s0.aggregated.vt_i[owned],
        qsr_oce=sf.qsr_oce,
        qns_oce=sf.qns_oce,
        qemp_oce=b2.qemp_oce,
        frq_m=sf.frq_m,
        e3t_m=sf.e3t_m,
        sst_m=forcing.sst_m[owned],
        t_bo=t_bo,
        rn_amax_2d=forcing.rn_amax_2d[owned],
        config=card.surface,
    )
    surface = o17_mod.SI3SurfaceOutputs(b1, utau_ice, vtau_ice, alb, b2, other)
    fluxes0 = o17_mod.SI3Fluxes(*(jnp.zeros_like(forcing.sss_m) for _ in o17_mod.SI3Fluxes._fields))
    dh_fluxes0 = o17_mod.SI3DHFluxes(
        **{name: getattr(fluxes0, name) for name in o17_mod.SI3DHFluxes._fields}
    )

    def run_newly_guarded(
        a_i,
        v_i,
        v_s,
        e_i,
        e_s,
        szv_i,
        t_su,
        t_si_in,
        qns_ice,
        qsr_ice,
        dqns_ice,
        qtr_ice_top,
        qsb_ice_bot,
        fhld,
        sprecip,
    ):
        eqv = o17_mod.si3_var_glo2eqv(
            v_i=v_i,
            v_s=v_s,
            sv_i=s0.sv_i,
            a_i=a_i,
            oa_i=s0.oa_i,
            a_ip=s0.a_ip,
            v_ip=s0.v_ip,
            v_il=s0.v_il,
            e_s=e_s,
            e_i=e_i,
            szv_i=szv_i,
            kn=2,
            hi_max_top=99.0,
            sss_m=forcing.sss_m,
            config=card.state,
        )
        agg = o17_mod.si3_var_agg(
            v_i=v_i,
            v_s=eqv.v_s,
            sv_i=s0.sv_i,
            a_i=eqv.a_i,
            oa_i=s0.oa_i,
            a_ip=s0.a_ip,
            v_ip=s0.v_ip,
            v_il=s0.v_il,
            t_su=t_su,
            t_si=t_si_in,
            e_s=e_s,
            e_i=e_i,
            szv_i=szv_i,
            a_ip_eff=eqv.a_ip_eff,
            t_i=eqv.t_i,
            t_s=eqv.t_s,
            kn=2,
            config=card.state,
        )
        st = s0._replace(
            a_i=eqv.a_i,
            v_i=v_i,
            v_s=eqv.v_s,
            e_i=e_i,
            e_s=e_s,
            szv_i=szv_i,
            t_su=t_su,
            t_si=t_si_in,
            equivalent=eqv,
            aggregated=agg,
        )
        surf = surface._replace(
            blk_ice_2=surface.blk_ice_2._replace(
                qns_ice=qns_ice,
                qsr_ice=qsr_ice,
                dqns_ice=dqns_ice,
                qtr_ice_top=qtr_ice_top,
                sprecip=sprecip,
            ),
            flx_other=surface.flx_other._replace(qsb_ice_bot=qsb_ice_bot, fhld=fhld),
        )
        thermo_entries = o17_mod._thermo_entries(
            st, forcing, surf, fluxes0, t_bo, card.state.nn_hls, card.constants.epsi10
        )
        stages = si3_ice_thd_o12_stages(
            category_entries=thermo_entries,
            category_fluxes=dh_fluxes0,
            sfx_bri=fluxes0.sfx_bri,
            sfx_lam=fluxes0.sfx_lam,
            hfx_thd=fluxes0.hfx_thd,
            wfx_lam=fluxes0.wfx_lam,
            a_ip=st.a_ip,
            v_ip=st.v_ip,
            h_ip=st.equivalent.h_ip,
            v_il=st.v_il,
            h_il=st.equivalent.h_il,
            a_ip_frac=st.equivalent.a_ip_frac,
            wfx_pnd=fluxes0.wfx_pnd,
            namelist=card.thermo_namelist,
            rDt_ice=card.state.rDt_ice,
            r1_Dt_ice=1.0 / card.state.rDt_ice,
            r1_nlay_i=1.0 / card.state.nlay_i,
            r1_nlay_s=1.0 / card.state.nlay_s,
            rho_i=card.constants.rho_ice,
            rho_s=card.constants.rho_snow,
            rho_w=constants.rho_water,
            gravity=constants.g_nemo,
            nn_hls=card.state.nn_hls,
            constants=card.constants,
        )
        t_si = jnp.stack(
            [
                jnp.where(e["active"], c.o10.o9.zdf.t_si, st.t_si[..., jl])
                for jl, (e, c) in enumerate(zip(thermo_entries, stages.categories))
            ],
            axis=-1,
        )[owned]
        thermo_active_owned = jnp.stack([e["active"][owned] for e in thermo_entries], axis=-1)
        qtr_bot = jnp.where(
            thermo_active_owned,
            jnp.stack([c.o10.o9.zdf.qtr_ice_bot for c in stages.categories], axis=-1)[owned],
            0.0,
        )
        qcn_top = jnp.where(
            thermo_active_owned,
            jnp.stack([c.o10.o9.zdf.qcn_ice_top for c in stages.categories], axis=-1)[owned],
            0.0,
        )
        qcn_bot = jnp.where(
            thermo_active_owned,
            jnp.stack([c.o10.o9.zdf.qcn_ice_bot for c in stages.categories], axis=-1)[owned],
            0.0,
        )
        qml_ice_out = jnp.where(
            thermo_active_owned,
            jnp.stack([c.o10.o9.dh.qml_ice for c in stages.categories], axis=-1)[owned],
            0.0,
        )
        carried = stages.fluxes.category
        return {
            "a_i": o17_mod._pack_categories(stages, "a_i")[owned],
            "v_i": o17_mod._pack_categories(stages, "v_i")[owned],
            "v_s": o17_mod._pack_categories(stages, "v_s")[owned],
            "e_i": o17_mod._pack_categories(stages, "e_i")[owned],
            "e_s": o17_mod._pack_categories(stages, "e_s")[owned],
            "szv_i": o17_mod._pack_categories(stages, "szv_i")[owned],
            "t_su": o17_mod._pack_categories(stages, "t_su")[owned],
            "t_si": t_si,
            "qtr_bot": qtr_bot,
            "qcn_top": qcn_top,
            "qcn_bot": qcn_bot,
            "qml_ice": qml_ice_out,
            "sfx_bri": stages.fluxes.sfx_bri[owned],
            "hfx_thd": stages.fluxes.hfx_thd[owned],
            "hfx_dif": stages.fluxes.hfx_dif[owned],
            "hfx_sub": carried.hfx_sub[owned],
            "wfx_snw_sub": carried.wfx_snw_sub[owned],
            "wfx_ice_sub": carried.wfx_ice_sub[owned],
            "hfx_spr": carried.hfx_spr[owned],
            "wfx_spr": carried.wfx_spr[owned],
        }

    args = (
        s0.a_i,
        s0.v_i,
        s0.v_s,
        s0.e_i,
        s0.e_s,
        s0.szv_i,
        s0.t_su,
        s0.t_si,
        b2.qns_ice,
        b2.qsr_ice,
        b2.dqns_ice,
        b2.qtr_ice_top,
        other.qsb_ice_bot,
        other.fhld,
        b2.sprecip,
    )
    f_jit = jax.jit(run_newly_guarded)
    out, vjp_fn = jax.vjp(f_jit, *args)
    for k, v in out.items():
        assert int(np.isnan(np.asarray(v)).sum()) == 0, f"NaN in forward output {k}"
    np.testing.assert_array_equal(np.asarray(out["qtr_bot"][0, 0]), np.zeros(5))
    np.testing.assert_array_equal(np.asarray(out["qcn_top"][0, 0]), np.zeros(5))
    np.testing.assert_array_equal(np.asarray(out["qcn_bot"][0, 0]), np.zeros(5))
    np.testing.assert_array_equal(np.asarray(out["qml_ice"][0, 0]), np.zeros(5))
    np.testing.assert_array_equal(
        np.asarray(out["t_si"][0, 0]), np.full(5, card.constants.T0, dtype=np.float64)
    )

    ones_cot = {k: jnp.ones_like(v) for k, v in out.items()}
    all_grads = vjp_fn(ones_cot)
    for idx, g in enumerate(all_grads):
        assert int(np.isnan(np.asarray(g)).sum()) == 0, f"NaN in reverse gradient arg[{idx}]"

    checks = (
        ("t_si", (1, 0, 2), 8, (1, 0, 2), 1.0e-4),
        ("t_si", (0, 0, 0), 7, (2, 2, 0), 1.0e-4),
        ("qtr_bot", (1, 0, 0), 11, (1, 0, 0), 1.0e-5),
        ("qcn_top", (1, 0, 2), 8, (1, 0, 2), 1.0e-4),
        ("qcn_bot", (1, 0, 2), 3, (3, 2, 9, 2), 10.0),
        ("qml_ice", (1, 0, 1), 8, (1, 0, 1), 1.0e-4),
        ("t_su", (1, 0, 2), 8, (1, 0, 2), 1.0e-4),
        ("a_i", (1, 0, 2), 0, (3, 2, 2), 1.0e-6),
        ("v_i", (1, 0, 2), 1, (3, 2, 2), 1.0e-6),
        ("v_s", (1, 0, 2), 2, (3, 2, 2), 1.0e-6),
        ("e_i", (1, 0, 0, 2), 3, (3, 2, 0, 2), 10.0),
        ("e_s", (1, 0, 0, 2), 2, (3, 2, 2), 1.0e-6),
        ("szv_i", (1, 0, 0, 2), 5, (3, 2, 0, 2), 1.0e-5),
        ("sfx_bri", (1, 0), 5, (3, 2, 0, 2), 1.0e-5),
        ("hfx_thd", (1, 0), 12, (1, 0), 1.0e-4),
        ("hfx_dif", (1, 0), 3, (3, 2, 0, 2), 10.0),
        ("hfx_sub", (1, 0), 8, (1, 0, 2), 1.0e-4),
        ("wfx_snw_sub", (1, 0), 0, (3, 2, 2), 1.0e-6),
        ("wfx_ice_sub", (1, 0), 0, (3, 2, 0), 1.0e-6),
        ("hfx_spr", (1, 0), 14, (1, 0), 1.0e-11),
        ("wfx_spr", (1, 0), 14, (1, 0), 1.0e-11),
    )

    def _assert_per_entry_relative_fd(vjp_callable, rtol: float = 1.0e-6) -> None:
        for out_key, out_idx, arg_pos, in_idx, eps in checks:
            cot = {k: jnp.zeros_like(v) for k, v in out.items()}
            cot[out_key] = cot[out_key].at[out_idx].set(1.0)
            grads = vjp_callable(cot)
            ad = float(grads[arg_pos][in_idx])
            assert ad != 0.0, f"Zero AD derivative for {out_key}{out_idx} wrt arg[{arg_pos}]{in_idx}"
            args_p = list(args)
            args_m = list(args)
            args_p[arg_pos] = args[arg_pos].at[in_idx].add(eps)
            args_m[arg_pos] = args[arg_pos].at[in_idx].add(-eps)
            fp = float(f_jit(*args_p)[out_key][out_idx])
            fm = float(f_jit(*args_m)[out_key][out_idx])
            fd = (fp - fm) / (2.0 * eps)
            rel = abs(ad - fd) / max(abs(ad), abs(fd))
            assert rel <= rtol, (
                f"Per-entry-relative FD mismatch for {out_key}{out_idx} wrt arg[{arg_pos}]{in_idx}: "
                f"ad={ad:+.8e} fd={fd:+.8e} rel={rel:.3e} > {rtol:.3e}"
            )

    _assert_per_entry_relative_fd(vjp_fn)

    # Plant 1: stop_gradient on newly guarded outputs must fail the check
    _, vjp_stop = jax.vjp(
        lambda *xs: jax.tree_util.tree_map(jax.lax.stop_gradient, f_jit(*xs)),
        *args,
    )
    with pytest.raises(AssertionError, match="Zero AD derivative"):
        _assert_per_entry_relative_fd(vjp_stop)

    # Plant 2: 1 + 1e-5 tangent scaling on newly guarded outputs must fail the 1e-6 check
    def vjp_scaled(cot):
        scaled_cot = jax.tree_util.tree_map(lambda c: c * (1.0 + 1.0e-5), cot)
        return vjp_fn(scaled_cot)

    with pytest.raises(AssertionError, match="Per-entry-relative FD mismatch"):
        _assert_per_entry_relative_fd(vjp_scaled)

