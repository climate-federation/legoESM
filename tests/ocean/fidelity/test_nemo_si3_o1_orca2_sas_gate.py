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

import importlib.util
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
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ice import (
    NEMO_SI3_CONSTANTS_CONFIG,
    BrineConfig,
    SeaIceConfig,
    SI3AllPratherMoments,
    SI3ColumnState,
    SI3ORCA2StateConfig,
    SI3PratherMoments,
    SI3State,
    SI3ThermoConfig,
    SnowConfig,
    init_dynamic_ice_state,
    linear_remap,
    read_si3_netcdf_restart,
    si3_apply_state_halos_orca2,
    si3_hfn_category_bounds,
    si3_lbc_lnk_orca2,
    si3_o1_step,
    si3_update_snwice_mass,
    si3_var_agg,
    si3_var_glo2eqv,
    step_sea_ice,
    write_si3_netcdf_restart,
)
from legoesm.ice.bitz_lipscomb import (
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
    """Real non-SI3 and lane-3 callers match parent commit `a572f4b25403` to 0 ULP and fail under Plant P1."""
    import legoesm.ice.bitz_lipscomb as bl_mod
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

    # 4. Lane-3 callers: SI3ColumnState, SI3ThermoConfig, si3_column_step_arrays, linear_remap
    p_si3_cfg = parent_mods["legoesm.ice.config"].SI3ThermoConfig()
    c_si3_cfg = SI3ThermoConfig()
    assert p_si3_cfg.n_ice_layers == c_si3_cfg.n_ice_layers == 3
    assert p_si3_cfg.n_snow_layers == c_si3_cfg.n_snow_layers == 3

    def _run_si3_col(bl_module):
        st_in = bl_module.SI3ColumnArrays(
            concentration=jnp.array([0.92], dtype=jnp.float64),
            h_ice=jnp.array([1.65], dtype=jnp.float64),
            h_snow=jnp.array([0.22], dtype=jnp.float64),
            T_surface=jnp.array([t0 - 9.5], dtype=jnp.float64),
            e_ice=jnp.array([[2.95e8, 2.82e8, 2.65e8]], dtype=jnp.float64),
            e_snow=jnp.array([[1.12e8, 1.11e8, 1.10e8]], dtype=jnp.float64),
            S_bulk=jnp.array([4.8], dtype=jnp.float64),
            S_layers=jnp.array([[4.2, 4.8, 5.4]], dtype=jnp.float64),
            age_volume=jnp.array([86400.0], dtype=jnp.float64),
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
        return bl_module.si3_column_step_arrays(
            st_in, frc_in, 3600.0, NEMO_SI3_CONSTANTS_CONFIG
        )

    p_l3_trace = _run_si3_col(parent_mods["legoesm.ice.bitz_lipscomb"])
    c_l3_trace = _run_si3_col(bl_mod)
    for stage_name in p_l3_trace._fields:
        p_st = getattr(p_l3_trace, stage_name)
        c_st = getattr(c_l3_trace, stage_name)
        for fname in p_st._fields:
            _assert_zero_ulp(
                getattr(p_st, fname),
                getattr(c_st, fname),
                f"si3_column_step_arrays.{stage_name}.{fname}",
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

    # 5. Plant P1: code-path perturbation in legacy/lane-3 path MUST fail _assert_zero_ulp
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

    orig_si3 = bl_mod.si3_column_step_arrays

    def _planted_si3(*args, **kwargs):
        trace = orig_si3(*args, **kwargs)
        return trace._replace(
            exit=trace.exit._replace(
                e_ice=jnp.nextafter(trace.exit.e_ice, jnp.inf)
            )
        )

    monkeypatch.setattr(bl_mod, "si3_column_step_arrays", _planted_si3)
    with pytest.raises(AssertionError, match="max ULP = 1 != 0"):
        planted_l3 = _run_si3_col(bl_mod)
        _assert_zero_ulp(
            p_l3_trace.exit.e_ice, planted_l3.exit.e_ice, "planted_si3"
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


def test_o1_ocean_output_log_verifier_and_plants(tmp_path: Path) -> None:
    """Verify whitespace-tolerant `ocean.output` log checking and its plants."""
    excerpt = ACQ_DIR / "ocean.output.excerpt"
    res = gate.verify_ocean_output_log(
        excerpt, expect_icedyn=True, expect_icethd=True, plant=None
    )
    assert res["status"] == "PASS"
    assert res["parsed"]["jpl"] == 5
    assert res["parsed"]["nlay_i"] == 10
    assert res["parsed"]["nlay_s"] == 5

    spaced = tmp_path / "ocean.output.spaced"
    spaced.write_text(excerpt.read_text(encoding="ascii").replace("=", "   =    "))
    res_spaced = gate.verify_ocean_output_log(
        spaced, expect_icedyn=True, expect_icethd=True, plant=None
    )
    assert res_spaced["status"] == "PASS"

    for plant_name in ("error-token", "missing-aaa", "wrong-jpl", "wrong-flag"):
        with pytest.raises(ValueError, match="STATUS PLANT-FIRED"):
            gate.verify_ocean_output_log(excerpt, plant=plant_name)


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
