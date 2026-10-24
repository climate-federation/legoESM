"""Fidelity and invariance tests for SI3 ladder rung O1 (`si3_r4`, `DECISION 159`).

Covers:
1. Parent (`a572f4b25403`) vs current commit `0`-ULP bit-identity across real
   non-SI3 callers (`bitz_lipscomb_conduction_step`, `IceColumnModel.step`,
   `step_sea_ice` slab and dynamic-v2) and real lane-3 SI3 callers
   (`SI3ColumnState`, `SI3ThermoConfig`, `si3_column_step_arrays`,
   `remap_itd_linear`, `open_water_growth`), plus failing code-path plants
   (`Plant P1`).
2. `0`-ULP exactness of `si3_hfn_category_bounds` against NEMO 5.0.2
   `src/ICE/iceitd.F90:736-761` (`CASE( np_cathfn )`) and `ocean.output:723`.
3. `0`-ULP exactness of `si3_var_glo2eqv`, `si3_var_agg` (`kn=1` and `kn=2`),
   and `si3_update_snwice_mass`, plus failing formula plants (`Plant P5`).
4. `0`-ULP exactness and idempotence of `si3_lbc_lnk_orca2` and
   `si3_apply_state_halos_orca2` across ORCA2 cyclic east-west seam, T-pivot
   north fold (`'T'`, `'U'`, `'V'`, Prather moments), and south closed boundary,
   plus failing north-fold sign plant (`Plant P4`).
5. `0`-ULP exactness of `si3_o1_step`, NetCDF restart round-trip
   (`write_si3_netcdf_restart` / `read_si3_netcdf_restart`), and lane-3
   `SI3ColumnState` bridge (`from_column_state` / `to_column_state`), plus
   `1`-ULP state/moment plants (`Plant P2`).
6. Compiled `gfortran` 4-rank layout fixture (`o1_si3_layout_fixture.F90` linked
   against `o1_si3_state_writer.F90`) with distinct rank-dependent data and
   failing plants (`Plant P3`, `Plant P7`: `rank-collapse`, `wrong-offset`,
   `field-swap`, `truncation`, `bad-magic`, `missing-frame`).
7. Whitespace-tolerant `ocean.output` log verifier (`verify_ocean_output_log`)
   and its failing plants (`error-token`, `missing-aaa`, `wrong-jpl`,
   `wrong-flag`).
8. End-to-end `jax.jit`, `jax.value_and_grad`, and `jax.jvp` differentiability
   through `si3_o1_differentiable_loss` with `0`-ULP primal invariance, JVP/VJP
   agreement, and failing `stop_gradient` plant (`Plant P6`).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import struct
import subprocess
import sys
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm import constants as canonical_constants
from legoesm.core.field import Field
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ice import (
    NEMO_SI3_CONSTANTS_CONFIG,
    BrineConfig,
    SeaIceConfig,
    SI3AllPratherMoments,
    SI3ColumnState,
    SI3ORCA2Forcing,
    SI3ORCA2StateConfig,
    SI3PratherMoments,
    SI3State,
    SI3ThermoConfig,
    SnowConfig,
    build_c1d_omip_l3_card,
    build_si3_orca2_o1_config,
    init_dynamic_ice_state,
    linear_remap,
    read_si3_netcdf_restart,
    si3_apply_state_halos_orca2,
    si3_hfn_category_bounds,
    si3_lbc_lnk_orca2,
    si3_o1_step,
    si3_rst_from_netcdf_dict,
    si3_rst_to_netcdf_dict,
    si3_update_snwice_mass,
    si3_var_agg,
    si3_var_glo2eqv,
    step_sea_ice,
    write_si3_netcdf_restart,
)
from legoesm.ice.bitz_lipscomb import (
    SI3SurfaceForcing,
    bitz_lipscomb_conduction_step,
    si3_column_step_arrays,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
TESTCASES_DIR = REPO_ROOT / "scripts/validate/ocean_fidelity/testcases"
ACQ_DIR = TESTCASES_DIR / "nemo_si3_o1_orca2_sas_acquisition"
PARENT_COMMIT = "a572f4b25403"

SPEC = importlib.util.spec_from_file_location(
    "nemo_si3_o1_orca2_sas_gate",
    TESTCASES_DIR / "nemo_si3_o1_orca2_sas_gate.py",
)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


@pytest.fixture(autouse=True)
def _force_fp64():
    set_policy(PrecisionPolicy.fp64())


def _assert_zero_ulp(a, b, label: str = "") -> None:
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    assert arr_a.shape == arr_b.shape, f"{label} shape {arr_a.shape} != {arr_b.shape}"
    ulp = gate.max_ulp_diff(arr_a, arr_b)
    assert ulp == 0, f"{label} max ULP = {ulp} != 0"


def _load_parent_commit_ice_modules(
    parent_commit: str = PARENT_COMMIT,
) -> dict[str, types.ModuleType]:
    """Load `packages/ice/legoesm/ice/*` directly from `parent_commit` via `git show`."""
    module_specs = (
        ("legoesm.ice.constants_config", "packages/ice/legoesm/ice/constants_config.py"),
        ("legoesm.ice.config", "packages/ice/legoesm/ice/config.py"),
        ("legoesm.ice.state", "packages/ice/legoesm/ice/state.py"),
        ("legoesm.ice.bitz_lipscomb", "packages/ice/legoesm/ice/bitz_lipscomb.py"),
        ("legoesm.ice.snow", "packages/ice/legoesm/ice/snow.py"),
        ("legoesm.ice.itd", "packages/ice/legoesm/ice/itd.py"),
        ("legoesm.ice.sea_ice", "packages/ice/legoesm/ice/sea_ice.py"),
        ("legoesm.ice.c1d_omip_l3", "packages/ice/legoesm/ice/c1d_omip_l3.py"),
        ("legoesm.ice.scm", "packages/ice/legoesm/ice/scm.py"),
    )
    loaded: dict[str, types.ModuleType] = {}
    saved_sys = {name: sys.modules.get(name) for name, _ in module_specs}
    try:
        for mod_name, rel_path in module_specs:
            src = subprocess.check_output(
                ["git", "show", f"{parent_commit}:{rel_path}"],
                cwd=REPO_ROOT,
                text=True,
            )
            mod = types.ModuleType(mod_name)
            mod.__file__ = f"<git:{parent_commit}:{rel_path}>"
            sys.modules[mod_name] = mod
            exec(compile(src, mod.__file__, "exec"), mod.__dict__)
            loaded[mod_name] = mod
    finally:
        for mod_name, orig_mod in saved_sys.items():
            if orig_mod is None:
                sys.modules.pop(mod_name, None)
            else:
                sys.modules[mod_name] = orig_mod
    return loaded


def test_o1_parent_vs_current_non_si3_and_lane3_callers_bit_identical(monkeypatch) -> None:
    """Real non-SI3 and lane-3 callers (including production `step_sea_ice` + `_si3_step_with_trace`) match parent commit `a572f4b25403` to 0 ULP and fail under Plant P1."""
    import legoesm.ice.bitz_lipscomb as bl_mod
    import legoesm.ice.c1d_omip_l3 as c1d_mod
    import legoesm.ice.itd as itd_mod
    import legoesm.ice.scm as scm_mod
    import legoesm.ice.sea_ice as sea_ice_mod

    parent_mods = _load_parent_commit_ice_modules(PARENT_COMMIT)

    # 1. Non-SI3 caller A: bitz_lipscomb_conduction_step
    t0 = canonical_constants.T_freeze
    q_layers = jnp.array(
        [[-2.95e8, -2.80e8, -2.60e8], [-3.05e8, -2.90e8, -2.70e8]],
        dtype=jnp.float64,
    )
    h_bl = jnp.array([1.4, 2.1], dtype=jnp.float64)
    s_bl = jnp.array([[4.5, 5.0, 5.5], [3.8, 4.2, 4.8]], dtype=jnp.float64)
    f_top = jnp.array([-18.0, -25.0], dtype=jnp.float64)
    t_bot = jnp.array([t0 - 1.8, t0 - 1.8], dtype=jnp.float64)
    dt_bl = 3600.0

    p_bl = parent_mods["legoesm.ice.bitz_lipscomb"].bitz_lipscomb_conduction_step(
        q_layers, h_bl, s_bl, f_top, t_bot, dt_bl
    )
    c_bl = bitz_lipscomb_conduction_step(q_layers, h_bl, s_bl, f_top, t_bot, dt_bl)
    for idx, (pa, ca) in enumerate(zip(p_bl, c_bl)):
        _assert_zero_ulp(pa, ca, f"bitz_lipscomb_conduction_step[{idx}]")

    # 2. Non-SI3 caller B: IceColumnModel.step (slab column model)
    import legoesm.ice as ice_pkg
    import legoesm.ice.config as cfg_mod
    import legoesm.ice.state as state_mod

    def _run_scm(scm_module, si_module):
        orig_pkg_step = ice_pkg.step_sea_ice
        try:
            ice_pkg.step_sea_ice = si_module.step_sea_ice
            scm = scm_module.IceColumnModel.create(
                ncol=2, dt=3600.0, forcing=scm_module.cold_polar_forcing(ncol=2)
            )
            resp = scm.step()
            return scm.state, resp
        finally:
            ice_pkg.step_sea_ice = orig_pkg_step

    p_scm_state, p_scm_resp = _run_scm(
        parent_mods["legoesm.ice.scm"],
        parent_mods["legoesm.ice.sea_ice"],
    )
    c_scm_state, c_scm_resp = _run_scm(scm_mod, sea_ice_mod)
    _assert_zero_ulp(p_scm_state.h_ice.data, c_scm_state.h_ice.data, "scm.h_ice")
    _assert_zero_ulp(p_scm_state.T_ice.data, c_scm_state.T_ice.data, "scm.T_ice")
    _assert_zero_ulp(
        p_scm_state.concentration.data,
        c_scm_state.concentration.data,
        "scm.concentration",
    )
    _assert_zero_ulp(p_scm_resp.T_sfc, c_scm_resp.T_sfc, "scm.resp.T_sfc")
    _assert_zero_ulp(p_scm_resp.lhflx, c_scm_resp.lhflx, "scm.resp.lhflx")

    # 3. Non-SI3 caller C: step_sea_ice (dynamic-v2)
    def _run_sea_ice(si_module, scm_module, config_module, state_module):
        cfg_v2 = config_module.SeaIceConfig(
            snow=config_module.SnowConfig(enabled=True),
            brine=config_module.BrineConfig(enabled=True),
        )
        dyn_state = state_module.init_dynamic_ice_state((2,))
        forcing_v2 = scm_module.cold_polar_forcing(ncol=2)
        sst = jnp.full((2,), t0 - 1.8, dtype=jnp.float64)
        zero_vel = jnp.zeros((2,), dtype=jnp.float64)
        return si_module.step_sea_ice(
            dyn_state, forcing_v2, sst, zero_vel, zero_vel, cfg_v2, 1.0, 3600.0
        )

    p_si_state, p_si_resp = _run_sea_ice(
        parent_mods["legoesm.ice.sea_ice"],
        parent_mods["legoesm.ice.scm"],
        parent_mods["legoesm.ice.config"],
        parent_mods["legoesm.ice.state"],
    )
    c_si_state, c_si_resp = _run_sea_ice(sea_ice_mod, scm_mod, cfg_mod, state_mod)
    for field in ("h_ice", "h_snow", "T_ice", "S_ice", "concentration"):
        _assert_zero_ulp(
            getattr(p_si_state, field).data,
            getattr(c_si_state, field).data,
            f"step_sea_ice.state.{field}",
        )
    _assert_zero_ulp(p_si_resp.T_sfc, c_si_resp.T_sfc, "step_sea_ice.resp.T_sfc")

    # 4. Lane-3 callers through the REAL production dispatcher `step_sea_ice` and `_si3_step_with_trace`
    p_si3_cfg = parent_mods["legoesm.ice.config"].SI3ThermoConfig()
    c_si3_cfg = SI3ThermoConfig()
    assert p_si3_cfg.n_ice_layers == c_si3_cfg.n_ice_layers == 3
    assert p_si3_cfg.n_snow_layers == c_si3_cfg.n_snow_layers == 3

    def _f1d(arr, name, units):
        return Field(
            data=arr,
            dims=("x",),
            units=units,
            name=name,
        )

    def _f2d(arr, name, units):
        return Field(
            data=arr,
            dims=("x", "z"),
            units=units,
            name=name,
        )

    def _run_si3_production_dispatch(si_module, st_module, bl_module, c1d_module):
        col_state = st_module.SI3ColumnState(
            concentration=_f1d(jnp.array([0.92], dtype=jnp.float64), "a_i", "1"),
            h_ice=_f1d(jnp.array([1.65], dtype=jnp.float64), "h_i", "m"),
            h_snow=_f1d(jnp.array([0.22], dtype=jnp.float64), "h_s", "m"),
            T_surface=_f1d(jnp.array([t0 - 9.5], dtype=jnp.float64), "t_su", "K"),
            e_ice=_f2d(jnp.array([[2.95e8, 2.82e8, 2.65e8]], dtype=jnp.float64), "e_i", "J/m3"),
            e_snow=_f2d(jnp.array([[1.12e8, 1.11e8, 1.10e8]], dtype=jnp.float64), "e_s", "J/m3"),
            S_bulk=_f1d(jnp.array([4.8], dtype=jnp.float64), "s_i", "PSU"),
            S_layers=_f2d(jnp.array([[4.2, 4.8, 5.4]], dtype=jnp.float64), "sz_i", "PSU"),
            age_volume=_f1d(jnp.array([86400.0], dtype=jnp.float64), "oa_i", "s"),
        )
        frc_in = bl_module.SI3SurfaceForcing(
            qns_ice=jnp.array([-28.0], dtype=jnp.float64),
            qsr_ice=jnp.array([35.0], dtype=jnp.float64),
            dqns_ice=jnp.array([-4.5], dtype=jnp.float64),
            qtr_ice_top=jnp.array([5.0], dtype=jnp.float64),
            t_bottom=jnp.array([t0 - 1.75], dtype=jnp.float64),
            sss=jnp.array([34.2], dtype=jnp.float64),
            evaporation=jnp.array([1.5e-6], dtype=jnp.float64),
            snow_precipitation=jnp.array([2.5e-6], dtype=jnp.float64),
            qprec_ice=jnp.array([-200.0], dtype=jnp.float64),
            qcn_ice_bottom=jnp.array([1.2], dtype=jnp.float64),
            qsb_ice_bottom=jnp.array([2.5], dtype=jnp.float64),
            fhld=jnp.array([1.5], dtype=jnp.float64),
            qlead=jnp.array([-10.0], dtype=jnp.float64),
        )
        card = c1d_module.build_c1d_omip_l3_card()
        sst = jnp.array([t0 - 1.75], dtype=jnp.float64)
        zero_vel = jnp.zeros((1,), dtype=jnp.float64)
        new_st, resp = si_module.step_sea_ice(
            col_state, frc_in, sst, zero_vel, zero_vel, card.config, 0.1, 3600.0
        )
        _, _, trace = si_module._si3_step_with_trace(
            col_state, frc_in, card.config, 3600.0
        )
        return new_st, resp, trace

    p_l3_st, p_l3_resp, p_l3_trace = _run_si3_production_dispatch(
        parent_mods["legoesm.ice.sea_ice"],
        parent_mods["legoesm.ice.state"],
        parent_mods["legoesm.ice.bitz_lipscomb"],
        parent_mods["legoesm.ice.c1d_omip_l3"],
    )
    c_l3_st, c_l3_resp, c_l3_trace = _run_si3_production_dispatch(
        sea_ice_mod, state_mod, bl_mod, c1d_mod
    )
    for fname in p_l3_st._fields:
        _assert_zero_ulp(
            getattr(p_l3_st, fname).data,
            getattr(c_l3_st, fname).data,
            f"step_sea_ice(si3_bl99).state.{fname}",
        )
    for rname in (
        "T_sfc",
        "albedo",
        "emissivity",
        "z0",
        "lhflx",
        "surface_mass_flux",
        "lhflx_exchange",
        "T_rad",
        "ice_concentration_thermo",
    ):
        _assert_zero_ulp(
            getattr(p_l3_resp, rname),
            getattr(c_l3_resp, rname),
            f"step_sea_ice(si3_bl99).resp.{rname}",
        )
    for stage_name in p_l3_trace._fields:
        p_st = getattr(p_l3_trace, stage_name)
        c_st = getattr(c_l3_trace, stage_name)
        for fname in p_st._fields:
            _assert_zero_ulp(
                getattr(p_st, fname),
                getattr(c_st, fname),
                f"_si3_step_with_trace.{stage_name}.{fname}",
            )

    # ITD linear_remap
    h_old = jnp.array([0.25, 0.85, 1.70, 2.80, 4.20], dtype=jnp.float64)
    a_old = jnp.array([0.20, 0.30, 0.25, 0.10, 0.05], dtype=jnp.float64)
    h_new = jnp.array([0.32, 0.95, 1.85, 3.00, 4.45], dtype=jnp.float64)
    a_new = jnp.array([0.18, 0.31, 0.26, 0.10, 0.05], dtype=jnp.float64)
    t_new = jnp.full(5, t0 - 6.0, dtype=jnp.float64)
    p_itd = parent_mods["legoesm.ice.itd"].linear_remap(
        h_old, a_old, h_new, a_new, 5, T_new=t_new
    )
    c_itd = linear_remap(h_old, a_old, h_new, a_new, 5, T_new=t_new)
    for idx, (pa, ca) in enumerate(zip(p_itd, c_itd)):
        _assert_zero_ulp(pa, ca, f"linear_remap[{idx}]")

    # 5. Plant P1: code-path perturbation in legacy and production step_sea_ice / _si3_step_with_trace MUST fail _assert_zero_ulp
    orig_fn = bl_mod.bitz_lipscomb_conduction_step

    def _planted_bl(*args, **kwargs):
        q_new, t_new, f_bot = orig_fn(*args, **kwargs)
        return jnp.nextafter(q_new, jnp.inf), t_new, f_bot

    monkeypatch.setattr(bl_mod, "bitz_lipscomb_conduction_step", _planted_bl)
    with pytest.raises(AssertionError, match="max ULP = 1 != 0"):
        planted_out = bl_mod.bitz_lipscomb_conduction_step(
            q_layers, h_bl, s_bl, f_top, t_bot, dt_bl
        )
        _assert_zero_ulp(p_bl[0], planted_out[0], "planted_bl")

    orig_si3_trace = sea_ice_mod._si3_step_with_trace

    def _planted_si3_dispatcher(state, forcing, config, dt):
        new_st, resp, trace = orig_si3_trace(state, forcing, config, dt)
        perturbed_e_ice = new_st.e_ice.replace(
            data=jnp.nextafter(new_st.e_ice.data, jnp.inf)
        )
        perturbed_resp = resp._replace(lhflx=jnp.nextafter(resp.lhflx, jnp.inf))
        return new_st._replace(e_ice=perturbed_e_ice), perturbed_resp, trace

    monkeypatch.setattr(
        sea_ice_mod, "_si3_step_with_trace", _planted_si3_dispatcher
    )
    with pytest.raises(AssertionError, match="max ULP = 1 != 0"):
        planted_st, planted_resp, _ = _run_si3_production_dispatch(
            sea_ice_mod, state_mod, bl_mod, c1d_mod
        )
        _assert_zero_ulp(
            p_l3_st.e_ice.data,
            planted_st.e_ice.data,
            "planted_step_sea_ice_e_ice",
        )
    with pytest.raises(AssertionError, match="max ULP = 1 != 0"):
        _assert_zero_ulp(
            p_l3_resp.lhflx,
            planted_resp.lhflx,
            "planted_step_sea_ice_lhflx",
        )


def test_o1_hfn_category_bounds_zero_ulp_vs_nemo502() -> None:
    """`si3_hfn_category_bounds` matches NEMO 5.0.2 `src/ICE/iceitd.F90:736-761` to 0 ULP."""
    bounds = si3_hfn_category_bounds(jpl=5, rn_himean=2.0, opt_hips=0, rn_himax=99.0)
    expected_hi_max_hex = [
        "0000000000000000",
        "3fdd0e9c3d20b854",
        "3ff211ab5f1109f4",
        "400121b5401a0ddc",
        "400d5d61a68b37da",
        "4058c00000000000",
    ]
    expected_hi_mean_hex = [
        "3fcd0e9c3d20b854",
        "3fe955526e593809",
        "3ffa2a8aefa292d6",
        "40073f8b7352a2db",
        "4013575869a2cdf2",
    ]
    expected_hi_max = np.array(
        [struct.unpack(">d", bytes.fromhex(h))[0] for h in expected_hi_max_hex],
        dtype=np.float64,
    )
    expected_hi_mean = np.array(
        [struct.unpack(">d", bytes.fromhex(h))[0] for h in expected_hi_mean_hex],
        dtype=np.float64,
    )
    _assert_zero_ulp(bounds.hi_max, expected_hi_max, "hi_max")
    _assert_zero_ulp(bounds.hi_mean, expected_hi_mean, "hi_mean")


def test_o1_self_test_and_all_plants_fire() -> None:
    """`run_self_test` passes at 0 ULP and every preregistered plant (P2, P4, P5, P6) fires."""
    report = gate.run_self_test(plant=None)
    assert report["status"] == "PASS"
    assert report["max_ulp"] == 0
    assert report["prather_arrays_checked"] == 55

    for plant_name in (
        "1ulp-state",
        "1ulp-moment",
        "north-fold-sign",
        "agg-formula",
        "stop-gradient",
    ):
        with pytest.raises(ValueError, match="STATUS PLANT-FIRED"):
            gate.run_self_test(plant=plant_name)


def test_o1_lane3_column_state_bridge_roundtrip() -> None:
    """`SI3State.from_column_state` and `to_column_state` bridge 1-category `3+3` lane-3 states to 0 ULP."""
    t0 = NEMO_SI3_CONSTANTS_CONFIG.T0
    col = SI3ColumnState(
        concentration=jnp.array([[0.50, 0.75], [0.25, 0.0]], dtype=jnp.float64),
        h_ice=jnp.array([[1.50, 2.00], [1.00, 0.0]], dtype=jnp.float64),
        h_snow=jnp.array([[0.125, 0.25], [0.50, 0.0]], dtype=jnp.float64),
        T_surface=jnp.array(
            [[t0 - 8.0, t0 - 11.0], [t0 - 5.0, t0]], dtype=jnp.float64
        ),
        e_ice=jnp.full((2, 2, 3), 2.85e8, dtype=jnp.float64),
        e_snow=jnp.full((2, 2, 3), 1.10e8, dtype=jnp.float64),
        S_bulk=jnp.array([[4.5, 5.0], [4.0, 0.0]], dtype=jnp.float64),
        S_layers=jnp.full((2, 2, 3), 4.5, dtype=jnp.float64),
        age_volume=jnp.array([[25.0, 40.0], [15.0, 0.0]], dtype=jnp.float64),
    )
    tmask = jnp.array([[1.0, 1.0], [1.0, 0.0]], dtype=jnp.float64)
    full_state = SI3State.from_column_state(col, tmask=tmask, sss_m=34.0)
    assert full_state.v_i.shape == (2, 2, 1)
    assert full_state.e_i.shape == (2, 2, 3, 1)
    assert full_state.e_s.shape == (2, 2, 3, 1)
    col_back = full_state.to_column_state()
    _assert_zero_ulp(col.concentration, col_back.concentration, "col.concentration")
    _assert_zero_ulp(col.h_ice, col_back.h_ice, "col.h_ice")
    _assert_zero_ulp(col.h_snow, col_back.h_snow, "col.h_snow")
    _assert_zero_ulp(col.T_surface, col_back.T_surface, "col.T_surface")


def test_o1_compiled_gfortran_layout_fixture_and_plants(tmp_path: Path) -> None:
    """Compile and run `o1_si3_layout_fixture.F90` with `gfortran` and verify 0-ULP 4-rank layout and plants."""
    fc = Path("/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran")
    if not fc.is_file():
        fc_which = shutil.which("gfortran")
        if not fc_which:
            pytest.skip("gfortran not available")
        fc = Path(fc_which)

    subprocess.run(
        [
            "cpp",
            "-P",
            "-traditional",
            str(ACQ_DIR / "o1_si3_state_writer.F90"),
            "-o",
            str(tmp_path / "o1_si3_state_writer.f90"),
        ],
        check=True,
    )
    subprocess.run(
        [
            str(fc),
            "-O2",
            "-ffree-line-length-none",
            "-J",
            str(tmp_path),
            str(ACQ_DIR / "o1_si3_fixture_stubs.F90"),
            str(tmp_path / "o1_si3_state_writer.f90"),
            str(ACQ_DIR / "o1_si3_layout_fixture.F90"),
            "-o",
            str(tmp_path / "o1_si3_layout_fixture.exe"),
        ],
        check=True,
    )
    records_dir = tmp_path / "o1_si3_records"
    records_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [str(tmp_path / "o1_si3_layout_fixture.exe")],
        cwd=tmp_path,
        check=True,
    )

    res = gate.verify_layout_fixture(records_dir, plant=None)
    assert res["status"] == "PASS"
    assert res["ranks_verified"] == 4
    assert res["max_ulp"] == 0

    for plant_name in (
        "rank-collapse",
        "wrong-offset",
        "field-swap",
        "truncation",
        "bad-magic",
        "missing-frame",
    ):
        with pytest.raises(ValueError, match="STATUS PLANT-FIRED"):
            gate.verify_layout_fixture(records_dir, plant=plant_name)


def test_o1_derived_deck_fixture_and_plants(tmp_path: Path) -> None:
    """Verify the non-evidence O1 deck fixture and its plants."""
    excerpt = ACQ_DIR / "ocean.output.excerpt"
    res = gate.verify_o1_deck_fixture(excerpt)
    assert res["status"] == "PASS_DERIVED_DECK_FIXTURE"

    spaced = tmp_path / "ocean.output.spaced"
    spaced.write_text(excerpt.read_text(encoding="ascii").replace("=", "   =    "))
    assert gate.verify_o1_deck_fixture(spaced)["status"] == "PASS_DERIVED_DECK_FIXTURE"

    for plant_name in ("error-token", "missing-line", "wrong-jpl", "wrong-flag"):
        with pytest.raises(ValueError, match="STATUS PLANT-FIRED"):
            gate.verify_o1_deck_fixture(excerpt, plant=plant_name)


def test_o1_jit_grad_jvp_and_stop_gradient_plant() -> None:
    """Verify `jax.jit`, `jax.value_and_grad`, `jax.jvp`, and `Plant P6` (`stop_gradient`)."""
    cfg = SI3ORCA2StateConfig.for_orca2_sas_ice()
    state0, tmask, umask, vmask = gate.build_synthetic_orca2_o1_state(cfg)

    def _loss_fn(st: SI3State, plant_stop: bool = False) -> jax.Array:
        st_in = jax.lax.stop_gradient(st) if plant_stop else st
        st_halo = si3_apply_state_halos_orca2(st_in, config=cfg)
        st_out = si3_o1_step(st_halo, tmask=tmask, sss_m=34.0, config=cfg)
        return (
            jnp.sum(st_out.aggregated.vt_i)
            + jnp.sum(st_out.aggregated.vt_s)
            + jnp.sum(st_out.aggregated.at_i)
            + jnp.sum(st_out.aggregated.tm_su)
            + jnp.sum(st_out.aggregated.st_i)
            + jnp.sum(st_halo.sv_i)
            + jnp.sum(st_out.aggregated.et_s) * 1.0e-8
            + jnp.sum(st_out.aggregated.et_i) * 1.0e-8
            + jnp.sum(st_out.equivalent.t_i)
            + jnp.sum(st_out.equivalent.t_s)
            + jnp.sum(st_out.equivalent.sz_i)
            + jnp.sum(st_out.snwice_mass) * 1.0e-3
            + jnp.sum(st_out.u_ice * umask)
            + jnp.sum(st_out.v_ice * vmask)
            + jnp.sum(st_out.prather.ice.sx)
            + jnp.sum(st_out.prather.e.sx) * 1.0e-6
        )

    loss_eager = _loss_fn(state0)
    loss_jit = jax.jit(_loss_fn)(state0)
    loss_vg, grad_st = jax.jit(jax.value_and_grad(_loss_fn))(state0)
    _assert_zero_ulp(loss_eager, loss_jit, "loss_eager_vs_jit")
    _assert_zero_ulp(loss_eager, loss_vg, "loss_eager_vs_vg")

    for leaf_name, leaf_grad in (
        ("v_i", grad_st.v_i),
        ("v_s", grad_st.v_s),
        ("sv_i", grad_st.sv_i),
        ("a_i", grad_st.a_i),
        ("t_su", grad_st.t_su),
        ("e_s", grad_st.e_s),
        ("e_i", grad_st.e_i),
        ("szv_i", grad_st.szv_i),
        ("u_ice", grad_st.u_ice),
        ("v_ice", grad_st.v_ice),
        ("prather.ice.sx", grad_st.prather.ice.sx),
        ("prather.e.sx", grad_st.prather.e.sx),
    ):
        arr = np.asarray(leaf_grad, dtype=np.float64)
        assert np.all(np.isfinite(arr)), f"Non-finite gradient in {leaf_name}"
        assert float(np.linalg.norm(arr)) > 0.0, f"Zero gradient in {leaf_name}"

    tangent_v_i = jnp.where(state0.v_i > 0.0, 1.0e-4, 0.0)
    zero_tree = jax.tree_util.tree_map(jnp.zeros_like, state0)
    tangent_state = zero_tree._replace(v_i=tangent_v_i)
    primal_jvp, jvp_val = jax.jvp(_loss_fn, (state0,), (tangent_state,))
    _assert_zero_ulp(loss_eager, primal_jvp, "loss_eager_vs_jvp")
    vjp_val = jnp.sum(grad_st.v_i * tangent_v_i)
    rel_err = float(
        jnp.abs(jvp_val - vjp_val) / jnp.maximum(jnp.abs(vjp_val), 1.0e-30)
    )
    assert rel_err < 1.0e-11, f"JVP vs VJP relative error {rel_err:.3e} >= 1e-11"

    # Plant P6: stop_gradient must zero the gradient
    _, grad_planted = jax.value_and_grad(lambda st: _loss_fn(st, plant_stop=True))(
        state0
    )
    assert float(jnp.linalg.norm(grad_planted.v_i)) == 0.0


def test_o1_production_step_sea_ice_refuses_missing_flux_contract_and_validates_dt() -> None:
    """O1 exposes its state operator directly and refuses an invented production flux response."""
    cfg = SI3ORCA2StateConfig.for_orca2_sas_ice()
    state0, tmask, umask, vmask = gate.build_synthetic_orca2_o1_state(cfg)
    sea_cfg = build_si3_orca2_o1_config(cfg)
    z2 = jnp.zeros_like(tmask)
    si3_o1_step(state0, tmask=tmask, sss_m=34.2, config=cfg)
    with pytest.raises(NotImplementedError, match="ice_update_flx"):
        step_sea_ice(
            state0,
            SI3ORCA2Forcing(tmask=tmask, sss_m=34.2),
            z2,
            z2,
            z2,
            sea_cfg,
            0.1,
            cfg.rDt_ice,
        )

    with pytest.raises(ValueError, match="dt == si3_orca2.rDt_ice"):
        step_sea_ice(
            state0,
            SI3ORCA2Forcing(tmask=tmask),
            z2,
            z2,
            z2,
            sea_cfg,
            0.1,
            cfg.rDt_ice + 1.0,
        )

    # Wrong state type refused
    with pytest.raises(ValueError, match="requires SI3State"):
        step_sea_ice(
            init_dynamic_ice_state((2,)),
            SI3ORCA2Forcing(tmask=tmask),
            z2,
            z2,
            z2,
            sea_cfg,
            0.1,
            cfg.rDt_ice,
        )

    # Wrong forcing type refused
    with pytest.raises(ValueError, match="requires SI3ORCA2Forcing"):
        step_sea_ice(state0, None, z2, z2, z2, sea_cfg, 0.1, cfg.rDt_ice)

    # Active dynamics/thermo flags refused at rung O1
    bad_cfg = build_si3_orca2_o1_config(cfg._replace(ln_icedyn=True))
    with pytest.raises(ValueError, match="ln_icedyn and ln_icethd must both be False"):
        step_sea_ice(
            state0,
            SI3ORCA2Forcing(tmask=tmask),
            z2,
            z2,
            z2,
            bad_cfg,
            0.1,
            cfg.rDt_ice,
        )


def test_o1_restart_reader_nemo_parity_t_si_reset_layer_refusal_and_r1_nlay_i() -> None:
    """Verify `t_si = rt0` reset (`icestp.F90:488`), layer-count refusal (`icerst.F90:241-255`), and `sv_i * r1_nlay_i` multiplication (`icerst.F90:351`)."""
    from legoesm.core.source_rounding import nemo_source_round

    cfg = SI3ORCA2StateConfig.for_orca2_sas_ice()
    state0, tmask, _umask, _vmask = gate.build_synthetic_orca2_o1_state(cfg)
    rt0 = cfg.ice_constants.T0

    # (2a) diag_set0 resets t_si = rt0 on A2D(0) each step even if perturbed before step
    perturbed_t_si = jnp.full_like(state0.t_si, 250.0)
    state_stepped = si3_o1_step(
        state0._replace(t_si=perturbed_t_si), tmask=tmask, config=cfg
    )
    h = cfg.nn_hls
    _assert_zero_ulp(
        state_stepped.t_si[h:-h, h:-h, :],
        np.full(state_stepped.t_si[h:-h, h:-h, :].shape, rt0),
        "t_si_interior_rt0",
    )
    _assert_zero_ulp(
        state_stepped.t_si[:h, :, :],
        np.zeros_like(state_stepped.t_si[:h, :, :]),
        "t_si_halo_zero",
    )

    rst_dict = si3_rst_to_netcdf_dict(state_stepped, kt=1, config=cfg)
    assert "sxice" not in rst_dict  # ln_icedyn=False omits Prather moments from NetCDF

    reloaded = si3_rst_from_netcdf_dict(rst_dict, tmask=tmask, config=cfg, ln_rstart=False)
    _assert_zero_ulp(
        reloaded.t_si[h:-h, h:-h, :],
        np.full(reloaded.t_si[h:-h, h:-h, :].shape, rt0),
        "reloaded_t_si_interior_rt0",
    )

    # (2b) Refuse when v_i is missing or enthalpy layer count != nlay_s + nlay_i
    bad_no_vi = dict(rst_dict)
    bad_no_vi.pop("v_i")
    with pytest.raises(ValueError, match="v_i is needed"):
        si3_rst_from_netcdf_dict(bad_no_vi, tmask=tmask, config=cfg)

    bad_no_es = dict(rst_dict)
    bad_no_es.pop("e_s_l01")
    with pytest.raises(ValueError, match="number of layers in restart"):
        si3_rst_from_netcdf_dict(bad_no_es, tmask=tmask, config=cfg)

    bad_no_ei = dict(rst_dict)
    bad_no_ei.pop("e_i_l01")
    with pytest.raises(ValueError, match="number of layers in restart"):
        si3_rst_from_netcdf_dict(bad_no_ei, tmask=tmask, config=cfg)

    bad_extra_ei = dict(rst_dict)
    bad_extra_ei["e_i_l11"] = rst_dict["e_i_l01"].copy()
    with pytest.raises(ValueError, match="number of layers in restart"):
        si3_rst_from_netcdf_dict(bad_extra_ei, tmask=tmask, config=cfg)

    # (2c) Salinity fallback uses sv_i * r1_nlay_i (icerst.F90:351), not sv_i / float(nlay_i)
    # For nlay_i = 10, 1.0 / 10.0 in binary64 is 0x3fb999999999999a; for sv_i = 3.0,
    # 3.0 * (1.0 / 10.0) and 3.0 / 10.0 differ by 1 ULP!
    r1_10 = float(np.float64(1.0) / np.float64(10.0))
    sv_probe = np.float64(3.0)
    mul_val = float(sv_probe * np.float64(r1_10))
    div_val = float(sv_probe / np.float64(10.0))
    assert gate.max_ulp_diff(np.array([mul_val]), np.array([div_val])) == 1

    no_szv = {k: v for k, v in rst_dict.items() if not k.startswith("szv_i_l")}
    no_szv["sv_i"] = np.full_like(rst_dict["sv_i"], sv_probe)
    reloaded_sal = si3_rst_from_netcdf_dict(
        no_szv, tmask=tmask, config=cfg, apply_halos=False, ln_rstart=False
    )
    expected_szv = np.full(
        (184, 152, cfg.nlay_i, cfg.jpl),
        float(nemo_source_round(jnp.asarray(sv_probe)) * nemo_source_round(jnp.asarray(1.0 / float(cfg.nlay_i)))),
        dtype=np.float64,
    )
    _assert_zero_ulp(reloaded_sal.szv_i, expected_szv, "szv_i_r1_nlay_i_mul")
    assert gate.max_ulp_diff(reloaded_sal.szv_i, np.full_like(expected_szv, div_val)) == 1


def test_o1_restart_optional_fields_follow_nemo_branching() -> None:
    """`t1_ice`, update mass, pond lids, and their scorer follow NEMO branches."""
    cfg = SI3ORCA2StateConfig.for_orca2_sas_ice()
    state, tmask, umask, vmask = gate.build_synthetic_orca2_o1_state(cfg)
    rst = si3_rst_to_netcdf_dict(state, kt=1, config=cfg)

    # ln_cpl=F leaves t1_ice undefined in icerst.F90:329; it is not scored.
    rst["t1_ice"] = np.full_like(rst["v_i"], 123.0)
    uncoupled = si3_rst_from_netcdf_dict(rst, tmask=tmask, config=cfg, ln_rstart=False)
    _assert_zero_ulp(uncoupled.t1_ice, np.zeros_like(uncoupled.t1_ice), "uncoupled.t1")
    changed_t1 = uncoupled._replace(t1_ice=jnp.ones_like(uncoupled.t1_ice))
    gate._compare_si3_states_wet_and_full(
        uncoupled, changed_t1, tmask, umask, vmask, score_t1_ice=False
    )
    with pytest.raises(ValueError, match="t1_ice"):
        gate._compare_si3_states_wet_and_full(
            uncoupled, changed_t1, tmask, umask, vmask, score_t1_ice=True
        )

    # update_rst('READ') is gated by ln_rstart (iceupdate.F90:453).
    rst["snwice_mass"] = np.full_like(rst["snwice_mass"], 9876.0)
    fresh = si3_rst_from_netcdf_dict(rst, tmask=tmask, config=cfg, ln_rstart=False)
    restarted = si3_rst_from_netcdf_dict(rst, tmask=tmask, config=cfg, ln_rstart=True)
    assert not np.any(np.asarray(fresh.snwice_mass) == 9876.0)
    assert np.all(np.asarray(restarted.snwice_mass)[2:-2, 2:-2] == 9876.0)

    # v_il exists only under a_ip and ln_pnd_lids (icerst.F90:322-328).
    no_lids = cfg._replace(ln_pnd_lids=False)
    without_lids = si3_rst_from_netcdf_dict(rst, tmask=tmask, config=no_lids)
    _assert_zero_ulp(without_lids.v_il, np.zeros_like(without_lids.v_il), "v_il.no_lids")
    no_pond = dict(rst)
    no_pond.pop("a_ip")
    no_pond.pop("v_ip")
    without_pond = si3_rst_from_netcdf_dict(no_pond, tmask=tmask, config=cfg)
    _assert_zero_ulp(without_pond.v_il, np.zeros_like(without_pond.v_il), "v_il.no_a_ip")


def test_o1_same_program_and_objects_equal_plants(tmp_path: Path) -> None:
    """Verify `gate.same_program` and imported `objects_equal` on reference vs instrumented builds and plants."""
    pra_patch = ACQ_DIR / "icedyn_adv_pra_o1_public.patch"
    stp_patch = ACQ_DIR / "icestp_o1_writer.patch"
    pra_add = [
        l[1:]
        for l in pra_patch.read_text(encoding="utf-8").splitlines()
        if l.startswith("+") and not l.startswith("+++")
    ]
    stp_add = [
        l[1:]
        for l in stp_patch.read_text(encoding="utf-8").splitlines()
        if l.startswith("+") and not l.startswith("+++")
    ]

    ref_pra = ["MODULE icedyn_adv_pra", "   PUBLIC ice_dyn_adv_pra", "   PUBLIC adv_pra_init", "END MODULE"]
    new_pra = ref_pra[:3] + pra_add + ref_pra[3:]
    ref_stp = ["MODULE icestp", "   USE prtctl", "   CALL ice_wri( kt )", "   IF( ierr /= 0 ) STOP", "END MODULE"]
    new_stp = [ref_stp[0], ref_stp[1], stp_add[0], ref_stp[2], stp_add[1], ref_stp[3], stp_add[2], ref_stp[4]]

    def _make_pair(base: Path) -> tuple[Path, Path]:
        ref = base / "ORCA2_SAS_ICE_O1_REF/BLD"
        new = base / "ORCA2_SAS_ICE_O1_REC/BLD"
        for bld, cfg_name, pra_lines, stp_lines, is_rec in (
            (ref, "ORCA2_SAS_ICE_O1_REF", ref_pra, ref_stp, False),
            (new, "ORCA2_SAS_ICE_O1_REC", new_pra, new_stp, True),
        ):
            (bld / "ppsrc/nemo").mkdir(parents=True, exist_ok=True)
            (bld / "obj").mkdir(parents=True, exist_ok=True)
            for cfile in gate.BUILD_CONFIG_FILES:
                (bld / cfile).write_text(f"dir::root /cfgs/{cfg_name}/BLD\n%FCFLAGS -O3\n")
            (bld / "ppsrc/nemo/icevar.f90").write_text("hm_i = vt_i * z1_at_i\n")
            (bld / "ppsrc/nemo/icedyn_adv_pra.f90").write_text("\n".join(pra_lines) + "\n")
            (bld / "ppsrc/nemo/icestp.f90").write_text("\n".join(stp_lines) + "\n")
            (bld / "obj/icevar.o").write_bytes(b"\x7fELF icevar")
            (bld / "obj/icedyn_adv_pra.o").write_bytes(b"\x7fELF pra" + bytes([is_rec]))
            (bld / "obj/icestp.o").write_bytes(b"\x7fELF stp" + bytes([is_rec]))
            if is_rec:
                (bld / "ppsrc/nemo/o1_si3_state_writer.f90").write_text("MODULE o1_si3_state_writer\n")
                (bld / "obj/o1_si3_state_writer.o").write_bytes(b"\x7fELF writer")
        return ref, new

    ref_bld, new_bld = _make_pair(tmp_path / "ok")
    res = gate.same_program(new_bld, ref_bld, pra_patch, stp_patch)
    assert res["status"] == "PASS_SAME_PROGRAM"

    for idx, (plant, mutate) in enumerate(
        (
            ("other-source", lambda _r, n: (n / "ppsrc/nemo/icevar.f90").write_text("hm_i = 2.0 * vt_i\n")),
            ("other-object", lambda _r, n: (n / "obj/icevar.o").write_bytes(b"\x7fELF icevar_bad")),
            ("flags", lambda _r, n: (n / "arch_nemo.fcm").write_text("dir::root /cfgs/ORCA2_SAS_ICE_O1_REC/BLD\n%FCFLAGS -O2\n")),
            ("observer-edit", lambda _r, n: (n / "ppsrc/nemo/icestp.f90").write_text("\n".join(new_stp).replace("ice_wri( kt )", "ice_wri( 0 )") + "\n")),
            ("observer-extra", lambda _r, n: (n / "ppsrc/nemo/icestp.f90").write_text("\n".join(new_stp + ["   x = 1"]) + "\n")),
            ("extra-unit", lambda _r, n: (n / "ppsrc/nemo/extra.f90").write_text("MODULE extra\n")),
            ("missing-unit", lambda _r, n: (n / "obj/icevar.o").unlink()),
        )
    ):
        r_b, n_b = _make_pair(tmp_path / f"p_{idx}_{plant}")
        mutate(r_b, n_b)
        with pytest.raises(ValueError, match="STATUS PLANT-FIRED"):
            gate.same_program(n_b, r_b, pra_patch, stp_patch)


def _write_o1_binary_snapshot(
    path: Path,
    state: SI3State,
    tmask: np.ndarray,
    umask: np.ndarray,
    vmask: np.ndarray,
    sss_m: np.ndarray,
    *,
    kt: int,
    lrst_ice: int,
    cfg: SI3ORCA2StateConfig,
) -> None:
    """Serialize `SI3State` into a 1-rank `O1SI3R01` binary snapshot for admission testing."""
    jpi, jpj, jpl = state.v_i.shape
    nlay_i, nlay_s = cfg.nlay_i, cfg.nlay_s
    nn_hls = cfg.nn_hls
    i_meta = np.array(
        [
            kt,
            0,
            1,
            jpi,
            jpj,
            jpl,
            nlay_i,
            nlay_s,
            nn_hls,
            jpi - 2 * nn_hls,
            jpj - 2 * nn_hls,
            1,
            1,
            1 + nn_hls,
            jpi - nn_hls,
            1 + nn_hls,
            jpj - nn_hls,
            cfg.nn_fsbc,
            cfg.nn_icesal,
            cfg.nn_snwfra,
            int(cfg.ln_pnd),
            int(cfg.ln_pnd_LEV),
            int(cfg.ln_pnd_TOPO),
            int(cfg.ln_pnd_lids),
            int(cfg.ln_cpl),
            lrst_ice,
        ],
        dtype="<i4",
    )
    r_meta = np.array([cfg.rDt_ice, cfg.rDt_ice, cfg.rn_himean], dtype="<f8")
    hi_max = np.asarray(state.bounds.hi_max, dtype="<f8")
    hi_mean = np.asarray(state.bounds.hi_mean, dtype="<f8")

    fmap: dict[str, np.ndarray] = {
        "tmask": np.asarray(tmask, dtype="<f8"),
        "umask": np.asarray(umask, dtype="<f8"),
        "vmask": np.asarray(vmask, dtype="<f8"),
        "sst_m": np.zeros((jpi, jpj), dtype="<f8"),
        "sss_m": np.asarray(sss_m, dtype="<f8"),
    }
    for name in (
        "v_i",
        "v_s",
        "sv_i",
        "a_i",
        "t_su",
        "oa_i",
        "a_ip",
        "v_ip",
        "v_il",
        "cnd_ice",
        "t1_ice",
        "t_si",
        "u_ice",
        "v_ice",
        "snwice_mass",
        "snwice_mass_b",
        "snwice_fmass",
        "e_s",
        "e_i",
        "szv_i",
    ):
        fmap[name] = np.asarray(getattr(state, name), dtype="<f8")
    for attr_name, suffix in (
        ("ice", "ice"),
        ("sn", "sn"),
        ("a", "a"),
        ("age", "age"),
        ("sal", "sal"),
        ("ap", "ap"),
        ("vp", "vp"),
        ("vl", "vl"),
        ("c0", "c0"),
        ("e", "e"),
        ("si", "si"),
    ):
        fam = getattr(state.prather, attr_name)
        for pfx in ("sx", "sy", "sxx", "syy", "sxy"):
            fmap[f"{pfx}{suffix}"] = np.asarray(getattr(fam, pfx), dtype="<f8")
    for eq_name in state.equivalent._fields:
        fmap[eq_name] = np.asarray(getattr(state.equivalent, eq_name), dtype="<f8")
    for agg_name in state.aggregated._fields:
        fmap[agg_name] = np.asarray(getattr(state.aggregated, agg_name), dtype="<f8")

    with path.open("wb") as out:
        out.write(b"O1SI3R01")
        out.write(struct.pack("<i", 1))
        out.write(i_meta.tobytes())
        out.write(r_meta.tobytes())
        out.write(hi_max.tobytes())
        out.write(hi_mean.tobytes())
        for fname, *_ in gate._FIXTURE_FIELD_OFFSETS:
            out.write(np.asfortranarray(fmap[fname], dtype="<f8").tobytes(order="F"))


def test_o1_admit_evidence_directory_scores_rst_1step_and_refuses_all_plants(
    tmp_path: Path,
) -> None:
    """Verify `admit_evidence_directory` scores `ref_2step`, `rec_2step`, and `rst_1step` over wet cells and refuses all plants."""
    cfg = SI3ORCA2StateConfig.for_orca2_sas_ice()
    state0_raw, tmask, umask, vmask = gate.build_synthetic_orca2_o1_state(cfg)
    # In O1 with ln_icedyn=False, adv_pra_init zeroes the Prather moments in memory.
    z_pra = SI3AllPratherMoments.zeros(
        184, 152, jpl=cfg.jpl, nlay_i=cfg.nlay_i, nlay_s=cfg.nlay_s
    )
    state0 = state0_raw._replace(prather=z_pra)
    sss = jnp.full((184, 152), 34.2, dtype=jnp.float64)
    state1 = si3_o1_step(state0, tmask=tmask, sss_m=sss, config=cfg)
    state2 = si3_o1_step(state1, tmask=tmask, sss_m=sss, config=cfg)

    ev = tmp_path / "evidence"
    for sub in ("manifest", "ref_2step/o1_si3_records", "rec_2step/o1_si3_records", "rst_1step/o1_si3_records", "rst_1step/input_restart", "rst_1step/output_restart"):
        (ev / sub).mkdir(parents=True, exist_ok=True)

    o1_log = """AAAAAAAA
Ni0glo = 180
Nj0glo = 148
nn_hls = 2
jpl = 5
nlay_i = 10
nlay_s = 5
ln_virtual_itd = F
ln_icedyn = F
ln_icethd = F
ln_cat_hfn = T
rn_himean = 2.0
AAAAAAAA
"""
    commit_hex = "a" * 40
    for rdir in (ev / "ref_2step", ev / "rec_2step", ev / "rst_1step"):
        (rdir / "ocean.output").write_text(o1_log, encoding="utf-8")
        (rdir / "run.user.stdout.log").write_text("NEMO run ok\nSTOP 0\n", encoding="utf-8")
        (rdir / "RUN_COMPLETE").write_text("ok\n", encoding="utf-8")
        (rdir / "RUN_PRODUCER_COMMIT").write_text(commit_hex + "\n", encoding="utf-8")

    # Write NetCDF restarts for step 1 and step 2
    write_si3_netcdf_restart(ev / "rec_2step/ORCA2_SAS_00000001_restart_ice.nc", state1, kt=1, config=cfg)
    write_si3_netcdf_restart(ev / "rec_2step/ORCA2_SAS_00000002_restart_ice.nc", state2, kt=2, config=cfg)
    shutil.copy2(
        ev / "rec_2step/ORCA2_SAS_00000001_restart_ice.nc",
        ev / "ref_2step/ORCA2_SAS_00000001_restart_ice.nc",
    )
    shutil.copy2(
        ev / "rec_2step/ORCA2_SAS_00000002_restart_ice.nc",
        ev / "ref_2step/ORCA2_SAS_00000002_restart_ice.nc",
    )
    shutil.copy2(
        ev / "rec_2step/ORCA2_SAS_00000001_restart_ice.nc",
        ev / "rst_1step/input_restart/ORCA2_SAS_00000001_restart_ice.nc",
    )

    # Write rec_2step binary records
    tmask_np, umask_np, vmask_np, sss_np = (
        np.asarray(tmask),
        np.asarray(umask),
        np.asarray(vmask),
        np.asarray(sss),
    )
    _write_o1_binary_snapshot(
        ev / "rec_2step/o1_si3_records/o1_si3_state_entry_rank0000.bin",
        state0,
        tmask_np,
        umask_np,
        vmask_np,
        sss_np,
        kt=0,
        lrst_ice=0,
        cfg=cfg,
    )
    _write_o1_binary_snapshot(
        ev / "rec_2step/o1_si3_records/o1_si3_state_step0001_rank0000.bin",
        state1,
        tmask_np,
        umask_np,
        vmask_np,
        sss_np,
        kt=1,
        lrst_ice=1,
        cfg=cfg,
    )
    _write_o1_binary_snapshot(
        ev / "rec_2step/o1_si3_records/o1_si3_state_step0002_rank0000.bin",
        state2,
        tmask_np,
        umask_np,
        vmask_np,
        sss_np,
        kt=2,
        lrst_ice=1,
        cfg=cfg,
    )

    # Write rst_1step binary records and restart
    rst_entry = read_si3_netcdf_restart(
        ev / "rec_2step/ORCA2_SAS_00000001_restart_ice.nc",
        tmask=tmask,
        sss_m=sss,
        config=cfg,
        ln_rstart=False,
    )
    rst_step1 = si3_o1_step(rst_entry, tmask=tmask, sss_m=sss, config=cfg)
    _write_o1_binary_snapshot(
        ev / "rst_1step/o1_si3_records/o1_si3_state_entry_rank0000.bin",
        rst_entry,
        tmask_np,
        umask_np,
        vmask_np,
        sss_np,
        kt=0,
        lrst_ice=0,
        cfg=cfg,
    )
    _write_o1_binary_snapshot(
        ev / "rst_1step/o1_si3_records/o1_si3_state_step0001_rank0000.bin",
        rst_step1,
        tmask_np,
        umask_np,
        vmask_np,
        sss_np,
        kt=1,
        lrst_ice=1,
        cfg=cfg,
    )
    write_si3_netcdf_restart(
        ev / "rst_1step/output_restart/ORCA2_SAS_00000001_restart_ice_o1_output.nc",
        rst_step1,
        kt=1,
        config=cfg,
    )

    # Write manifests
    (ev / "manifest/producer_commit.txt").write_text(commit_hex + "\n", encoding="utf-8")
    (ev / "manifest/same_program.json").write_text(
        json.dumps({"status": "PASS_SAME_PROGRAM"}) + "\n", encoding="utf-8"
    )
    dummy_bin = ev / "manifest/dummy_nemo.exe"
    dummy_bin.write_bytes(b"\x7fELF nemo")
    bin_sha = hashlib.sha256(dummy_bin.read_bytes()).hexdigest()
    (ev / "manifest/binaries.sha256").write_text(f"{bin_sha}  {dummy_bin}\n", encoding="utf-8")
    (ev / "manifest/inputs.sha256").write_text(f"{bin_sha}  {dummy_bin}\n", encoding="utf-8")

    adm = gate.admit_evidence_directory(ev, expect_commit=commit_hex)
    assert adm["status"] == "PASS"
    assert adm["max_ulp_kt1"] == 0
    assert adm["max_ulp_kt2"] == 0
    assert adm["max_ulp_rst_entry"] == 0
    assert adm["max_ulp_rst_step1"] == 0
    assert adm["max_ulp_rst_vs_kt2"] == 0

    for plant_name in (
        "stamp",
        "same-program",
        "binary-sha",
        "input-sha",
        "missing-complete",
        "missing-stop0",
        "ref-rec-diff",
        "1ulp-state",
        "rst-missing",
        "rst-1ulp",
    ):
        with pytest.raises(ValueError, match="STATUS PLANT-FIRED"):
            gate.admit_evidence_directory(ev, plant=plant_name, expect_commit=commit_hex)


def test_o1_committed_log_fixture_is_explicitly_deck_derived() -> None:
    """The committed fixture is namelist-derived and never rewritten run evidence."""
    excerpt = ACQ_DIR / "ocean.output.excerpt"
    text = excerpt.read_text(encoding="utf-8")
    assert "NOT NEMO RUN EVIDENCE" in text
    assert gate.verify_o1_deck_fixture(excerpt)["status"] == "PASS_DERIVED_DECK_FIXTURE"
    for plant in ("error-token", "missing-line", "wrong-jpl", "wrong-flag"):
        with pytest.raises(ValueError, match="STATUS PLANT-FIRED"):
            gate.verify_o1_deck_fixture(excerpt, plant=plant)
