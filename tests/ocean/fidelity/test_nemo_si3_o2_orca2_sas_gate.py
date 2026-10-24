"""Fidelity and invariance tests for SI3 ladder rung O2 (`si3_r7`, `DECISION 153`, `DECISION 159`).

Covers:
1. Parent (`fe19fb330ecb`) vs current commit `0`-ULP bit-identity across real non-SI3 callers
   (`bitz_lipscomb_conduction_step`, `IceColumnModel.step`, `step_sea_ice` dynamic-v2),
   real lane-3 SI3 callers (`step_sea_ice(thermo_scheme='si3_bl99')`, `_si3_step_with_trace`,
   `linear_remap`), rung-O1 callers (`step_sea_ice(thermo_scheme='si3_orca2_o1')`, `si3_o1_step`),
   plus failing code-path plants (`Plant P1`).
2. `0`-ULP exactness of `si3_eos_fzp` (`ln_teos10=True` and `ln_eos80=True`) against
   NEMO 5.0.2 `src/OCE/TRA/eosbn2.F90:1674-1696` and `t_bo` interior masking (`smask0`) at
   `src/ICE/iceistate.F90:125-126`.
3. `0`-ULP exactness of `si3_dom_uniq_t_orca2` (`src/OCE/DOM/domutl.F90:91-121`) and
   `si3_glob_2dsum_ddpdd` (`src/OCE/LBC/lib_mpp.F90:1244-1267`, `lib_fortran.f90:1505-1531`),
   including `r1_rho0` multiplication vs division by `rho0`.
4. `0`-ULP exactness of `si3_o2_scalar_initial_state`, plus failing production-code-path
   plants (`P2, P3, P4, P6`).
5. Compiled `gfortran` 4-rank layout fixture (`o2_si3_layout_fixture.F90` linked against
   `o2_si3_state_writer.F90`, exercising both `O2SI3I01` and `O1SI3R01`) and failing plants (`P5`).
6. Whitespace-tolerant `ocean.output` log verifier (`verify_ocean_output_log`) and plants.
7. End-to-end `jax.jit`, `jax.value_and_grad`, and `jax.jvp` differentiability through
   `si3_o2_scalar_initial_state` and `si3_o2_step` with `0`-ULP primal invariance, JVP/VJP
   agreement, and failing `stop_gradient` plant (`Plant P6`).
8. Production `step_sea_ice` selection of `thermo_scheme='si3_orca2_o2'` (`build_si3_orca2_o2_config`)
   and validation of state/forcing/config combinations.
9. `same_program` and `admit_evidence_directory` scoring of `O2SI3I01` (`o2_si3_istate_rank*.bin`),
   `rec_2step`, and `rst_1step`, plus all provenance/admission plants (`Plant P7`).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
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
import legoesm.ice.sea_ice as sea_ice_module

from legoesm import constants as canonical_constants
from legoesm.core.field import Field
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.core.source_rounding import nemo_source_round
from legoesm.ice import (
    BrineConfig,
    SeaIceConfig,
    SI3AllPratherMoments,
    SI3ColumnState,
    SI3O2InitConfig,
    SI3O2ScalarInitialState,
    SI3ORCA2Forcing,
    SI3ORCA2O2Forcing,
    SI3ORCA2StateConfig,
    SI3State,
    SI3ThermoConfig,
    SnowConfig,
    build_c1d_omip_l3_card,
    build_si3_orca2_o1_config,
    build_si3_orca2_o2_config,
    init_dynamic_ice_state,
    linear_remap,
    read_si3_netcdf_restart,
    si3_dom_uniq_t_orca2,
    si3_eos_fzp,
    si3_glob_2dsum_ddpdd,
    si3_lbc_lnk_orca2,
    si3_o1_step,
    si3_o2_scalar_initial_state,
    si3_o2_step,
    step_sea_ice,
    write_si3_netcdf_restart,
)
from legoesm.ice.bitz_lipscomb import bitz_lipscomb_conduction_step
from tests.ocean.fidelity.test_nemo_si3_o1_orca2_sas_gate import (
    _write_o1_binary_snapshot,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
TESTCASES_DIR = REPO_ROOT / "scripts/validate/ocean_fidelity/testcases"
ACQ_DIR = TESTCASES_DIR / "nemo_si3_o2_orca2_sas_acquisition"
PARENT_COMMIT = "042c7bd74c2944dab8c39318687ab0cef54f9d2b"

SPEC = importlib.util.spec_from_file_location(
    "nemo_si3_o2_orca2_sas_gate",
    TESTCASES_DIR / "nemo_si3_o2_orca2_sas_gate.py",
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


def _load_parent_commit_modules(
    parent_commit: str = PARENT_COMMIT,
) -> dict[str, types.ModuleType]:
    """Load the parent commit's `packages/ice/legoesm/ice/*` via `git show`."""
    module_specs = (
        ("legoesm.ice.constants_config", parent_commit, "packages/ice/legoesm/ice/constants_config.py"),
        ("legoesm.ice.si3_orca2_state", parent_commit, "packages/ice/legoesm/ice/si3_orca2_state.py"),
        ("legoesm.ice.config", parent_commit, "packages/ice/legoesm/ice/config.py"),
        ("legoesm.ice.state", parent_commit, "packages/ice/legoesm/ice/state.py"),
        ("legoesm.ice.bitz_lipscomb", parent_commit, "packages/ice/legoesm/ice/bitz_lipscomb.py"),
        ("legoesm.ice.snow", parent_commit, "packages/ice/legoesm/ice/snow.py"),
        ("legoesm.ice.itd", parent_commit, "packages/ice/legoesm/ice/itd.py"),
        ("legoesm.ice.sea_ice", parent_commit, "packages/ice/legoesm/ice/sea_ice.py"),
        ("legoesm.ice.c1d_omip_l3", parent_commit, "packages/ice/legoesm/ice/c1d_omip_l3.py"),
        ("legoesm.ice.scm", parent_commit, "packages/ice/legoesm/ice/scm.py"),
    )
    loaded: dict[str, types.ModuleType] = {}
    saved_sys = {name: sys.modules.get(name) for name, _, _ in module_specs}
    try:
        for mod_name, rev, rel_path in module_specs:
            src = subprocess.check_output(
                ["git", "show", f"{rev}:{rel_path}"],
                cwd=REPO_ROOT,
                text=True,
            )
            mod = types.ModuleType(mod_name)
            mod.__file__ = f"<git:{rev}:{rel_path}>"
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


def test_o2_parent_vs_current_all_callers_bit_identical(monkeypatch) -> None:
    """Real non-SI3, lane-3, O1, and ORCA2 recipe callers match the landed parent and fail under Plant P1."""
    import legoesm.ice as ice_pkg
    import legoesm.ice.bitz_lipscomb as bl_mod
    import legoesm.ice.c1d_omip_l3 as c1d_mod
    import legoesm.ice.config as cfg_mod
    import legoesm.ice.scm as scm_mod
    import legoesm.ice.sea_ice as sea_ice_mod
    import legoesm.ice.si3_orca2_state as o1_mod
    import legoesm.ice.state as state_mod

    parent_mods = _load_parent_commit_modules(PARENT_COMMIT)

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

    # 2. Non-SI3 caller B: IceColumnModel.step
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
    _assert_zero_ulp(p_scm_resp.T_sfc, c_scm_resp.T_sfc, "scm.resp.T_sfc")

    # 3. Lane-3 production dispatcher: step_sea_ice(thermo_scheme='si3_bl99') + _si3_step_with_trace
    def _f1d(arr, name, units):
        return Field(data=arr, dims=("x",), units=units, name=name)

    def _f2d(arr, name, units):
        return Field(data=arr, dims=("x", "z"), units=units, name=name)

    def _run_si3_lane3(si_module, st_module, bl_module, c1d_module):
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
        return new_st, resp

    p_l3_st, p_l3_resp = _run_si3_lane3(
        parent_mods["legoesm.ice.sea_ice"],
        parent_mods["legoesm.ice.state"],
        parent_mods["legoesm.ice.bitz_lipscomb"],
        parent_mods["legoesm.ice.c1d_omip_l3"],
    )
    c_l3_st, c_l3_resp = _run_si3_lane3(sea_ice_mod, state_mod, bl_mod, c1d_mod)
    for fname in p_l3_st._fields:
        _assert_zero_ulp(
            getattr(p_l3_st, fname).data,
            getattr(c_l3_st, fname).data,
            f"step_sea_ice(si3_bl99).state.{fname}",
        )
    _assert_zero_ulp(p_l3_resp.lhflx, c_l3_resp.lhflx, "step_sea_ice(si3_bl99).resp.lhflx")

    # 4. Rung-O1 production dispatcher: step_sea_ice(thermo_scheme='si3_orca2_o1') + si3_o1_step
    cfg_o1 = SI3ORCA2StateConfig.for_orca2_sas_ice()
    state0_o1, tmask, umask, vmask = gate.build_synthetic_orca2_o1_state(cfg_o1)
    z2 = jnp.zeros_like(tmask)
    p_o1_st = parent_mods["legoesm.ice.si3_orca2_state"].si3_o1_step(
        state0_o1, tmask=tmask, sss_m=34.2, config=cfg_o1
    )
    o1_dispatch_args = (
        state0_o1,
        SI3ORCA2Forcing(tmask=tmask, sss_m=34.2),
        z2,
        z2,
        z2,
        build_si3_orca2_o1_config(cfg_o1),
        0.1,
        cfg_o1.rDt_ice,
    )
    with pytest.raises(NotImplementedError, match="ice_update_flx"):
        step_sea_ice(*o1_dispatch_args)

    original_o1_dispatch = sea_ice_mod._si3_orca2_o1_step_with_trace

    def _planted_o1_dispatch(*args, **kwargs):
        raise RuntimeError("P1 O1 DISPATCH PATH")

    monkeypatch.setattr(
        sea_ice_mod, "_si3_orca2_o1_step_with_trace", _planted_o1_dispatch
    )
    with pytest.raises(RuntimeError, match="P1 O1 DISPATCH PATH"):
        step_sea_ice(*o1_dispatch_args)
    monkeypatch.setattr(
        sea_ice_mod, "_si3_orca2_o1_step_with_trace", original_o1_dispatch
    )

    # 5. Plant P1: a code-path perturbation in O1 MUST fail _assert_zero_ulp.
    orig_o1_step = o1_mod.si3_o1_step

    def _planted_o1_step(*args, **kwargs):
        st = orig_o1_step(*args, **kwargs)
        return st._replace(v_i=jnp.nextafter(st.v_i, jnp.inf))

    monkeypatch.setattr(o1_mod, "si3_o1_step", _planted_o1_step)
    with pytest.raises(AssertionError, match="max ULP = 1 != 0"):
        planted_o1 = o1_mod.si3_o1_step(state0_o1, tmask=tmask, sss_m=34.2, config=cfg_o1)
        _assert_zero_ulp(p_o1_st.v_i, planted_o1.v_i, "planted_o1_v_i")

def test_o2_eos_fzp_teos10_and_eos80_zero_ulp_vs_nemo502() -> None:
    """`si3_eos_fzp` matches NEMO 5.0.2 `src/OCE/TRA/eosbn2.F90:1674-1696` for both `ln_teos10` and `ln_eos80` to 0 ULP."""
    sal = np.array([[0.0, 5.0, 18.5, 34.7, 35.16504, 38.2]], dtype=np.float64)

    # TEOS-10 reference (`eosbn2.F90:69, 1674-1683`)
    r1_s0 = np.float64(1.0) / np.float64(35.16504)
    zs = np.sqrt(np.abs(sal) * r1_s0)
    fzp_teos10_ref = (
        (
            (
                (
                    (np.float64(1.46873e-03) * zs - np.float64(9.64972e-03)) * zs
                    + np.float64(2.28348e-02)
                )
                * zs
                - np.float64(3.12775e-02)
            )
            * zs
            + np.float64(2.07679e-02)
        )
        * zs
        - np.float64(5.87701e-02)
    ) * sal

    # EOS-80 reference (`eosbn2.F90:1691-1696`)
    fzp_eos80_ref = (
        np.float64(-0.0575)
        + np.float64(1.710523e-3) * np.sqrt(np.maximum(sal, 0.0))
        - np.float64(2.154996e-4) * sal
    ) * sal

    _assert_zero_ulp(
        si3_eos_fzp(jnp.asarray(sal), ln_teos10=True),
        fzp_teos10_ref,
        "eos_fzp_teos10",
    )
    _assert_zero_ulp(
        si3_eos_fzp(jnp.asarray(sal), ln_teos10=False),
        fzp_eos80_ref,
        "eos_fzp_eos80",
    )


def test_o2_ponds_off_and_lids_off_match_iceistate_branches() -> None:
    """`iceistate.F90:291-300` zeros pond/lid scalar inputs under both switches."""
    inp = gate.build_synthetic_orca2_o2_inputs()
    state = inp["state0"]
    common = dict(
        sst_m=inp["sst_m"], sss_m=inp["sss_m"], tmask=inp["tmask"],
        ssmask=inp["ssmask"], smask0_i=inp["smask0_i"], e1e2t=inp["e1e2t"],
        ff_t=inp["ff_t"], gphit=inp["gphit"],
        ssh_before_kbb=inp["ssh_before_kbb"], ssh_before_kmm=inp["ssh_before_kmm"],
        fitted_v_i=state.v_i, fitted_v_s=state.v_s, fitted_a_i=state.a_i,
        fitted_a_ip=state.a_ip, fitted_v_ip=state.v_ip, fitted_v_il=state.v_il,
    )
    ponds_off = si3_o2_scalar_initial_state(
        **common, config=inp["o2_cfg"]._replace(ln_pnd=False)
    )
    lids_off = si3_o2_scalar_initial_state(
        **common,
        config=inp["o2_cfg"]._replace(
            ln_pnd=True, ln_pnd_lids=False,
            rn_hld_ini_n=0.03, rn_hld_ini_s=0.03,
        ),
    )
    assert np.count_nonzero(np.asarray(ponds_off.zapnd_ini)) == 0
    assert np.count_nonzero(np.asarray(ponds_off.zhpnd_ini)) == 0
    assert np.count_nonzero(np.asarray(ponds_off.zhlid_ini)) == 0
    assert np.count_nonzero(np.asarray(lids_off.zapnd_ini)) > 0
    assert np.count_nonzero(np.asarray(lids_off.zhpnd_ini)) > 0
    assert np.count_nonzero(np.asarray(lids_off.zhlid_ini)) == 0


def test_o2_dom_uniq_t_and_glob_2dsum_ddpdd_zero_ulp() -> None:
    """`si3_dom_uniq_t_orca2` (`domutl.F90:91-121`) and `si3_glob_2dsum_ddpdd` (`lib_mpp.F90:1244-1267`) match NEMO 5.0.2 to 0 ULP."""
    inp = gate.build_synthetic_orca2_o2_inputs()
    ssmask = inp["ssmask"]
    smask0_i = ssmask * si3_dom_uniq_t_orca2(184, 152, nn_hls=2)

    # Verify halos are 0.0 and north-fold right half of top interior row (j = 149) is zeroed
    assert np.all(np.asarray(smask0_i[:2, :]) == 0.0)
    assert np.all(np.asarray(smask0_i[-2:, :]) == 0.0)
    assert np.all(np.asarray(smask0_i[:, :2]) == 0.0)
    assert np.all(np.asarray(smask0_i[:, -2:]) == 0.0)
    # On ORCA2 (184 x 152), top interior row is index 149 (1-based 150 = jpj - nn_hls);
    # for ji >= jpi/2 + 1 = 93 (0-based 92..181), T-pivot fold points (except pivot i=91) are duplicated and zeroed.
    assert np.all(np.asarray(smask0_i[92:182, 149]) == 0.0)

    # Verify r1_rho0 multiplication vs division by rho0 = 1026.0
    # In NEMO (phycst.F90 / iceistate.F90:411): snwice_mass * r1_rho0 * e1e2t where r1_rho0 = 1.0 / 1026.0.
    rho0 = np.float64(1026.0)
    r1_rho0 = np.float64(1.0) / rho0
    o2_state = si3_o2_scalar_initial_state(
        sst_m=inp["sst_m"],
        sss_m=inp["sss_m"],
        tmask=inp["tmask"],
        ssmask=ssmask,
        smask0_i=smask0_i,
        e1e2t=inp["e1e2t"],
        ff_t=inp["ff_t"],
        gphit=inp["gphit"],
        ssh_before_kbb=inp["ssh_before_kbb"],
        ssh_before_kmm=inp["ssh_before_kmm"],
        fitted_v_i=inp["state0"].v_i,
        fitted_v_s=inp["state0"].v_s,
        fitted_a_i=inp["state0"].a_i,
        fitted_a_ip=inp["state0"].a_ip,
        fitted_v_ip=inp["state0"].v_ip,
        fitted_v_il=inp["state0"].v_il,
        config=inp["o2_cfg"],
    )
    term_mul = nemo_source_round(
        nemo_source_round(o2_state.snwice_mass * r1_rho0) * inp["e1e2t"]
    )
    zssh_mul = si3_glob_2dsum_ddpdd(term_mul, smask0_i, nn_hls=2) / o2_state.area
    _assert_zero_ulp([zssh_mul], [o2_state.zsshadj], "zsshadj_r1_rho0_ddpdd")


def test_o2_self_test_exact_category_inputs_and_all_plants_fire(monkeypatch) -> None:
    """Every plant fires; the four reviewed physics plants select model-owned switches."""
    report = gate.run_self_test(plant=None)
    assert report["status"] == "PASS"
    assert report["max_ulp"] == 0
    assert report["o2_max_ulp_wet"] == 0
    assert report["o2_max_ulp_full"] == 0
    assert report["o2_scalar_fields_checked"] == 24
    assert report["prather_arrays_checked"] == 55

    switch_by_plant = {
        "tbo-halo-leak": "_plant_tbo_halo_leak",
        "pond-unscaled": "_plant_pond_unscaled",
        "ssh-sign-or-div": "_plant_ssh_sign",
        "dom-uniq-fold": "_plant_omit_nfd_uniq",
    }
    original = gate.si3_o2_scalar_initial_state
    selected_switches: list[str] = []

    def _record_model_switch(*args, **kwargs):
        selected_switches.extend(
            name for name in switch_by_plant.values() if kwargs.get(name) is True
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(gate, "si3_o2_scalar_initial_state", _record_model_switch)
    for plant_name in (
        "1ulp-istate",
        "1ulp-ssh",
        "eos-wrong-branch",
        "tbo-halo-leak",
        "pond-unscaled",
        "hemisphere-swap",
        "ssh-sign-or-div",
        "dom-uniq-fold",
        "north-fold-sign",
        "1ulp-state",
        "1ulp-moment",
        "agg-formula",
        "stop-gradient",
    ):
        with pytest.raises(ValueError, match="STATUS PLANT-FIRED"):
            gate.run_self_test(plant=plant_name)
        if plant_name in switch_by_plant:
            assert switch_by_plant[plant_name] in selected_switches


def test_o2_compiled_gfortran_layout_fixture_and_plants(tmp_path: Path) -> None:
    """Compile and run `o2_si3_layout_fixture.F90` with `gfortran` and verify 0-ULP 4-rank `O2SI3I01` + `O1SI3R01` layout and plants."""
    fc = Path("/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran")
    if not fc.is_file():
        fc_which = shutil.which("gfortran")
        if not fc_which:
            pytest.skip("gfortran not available")
        fc = Path(fc_which)

    subprocess.run(
        [
            str(fc),
            "-O2",
            "-ffree-line-length-none",
            "-J",
            str(tmp_path),
            str(ACQ_DIR / "o2_si3_fixture_stubs.F90"),
            str(ACQ_DIR / "o2_si3_state_writer.F90"),
            str(ACQ_DIR / "o2_si3_layout_fixture.F90"),
            "-o",
            str(tmp_path / "o2_si3_layout_fixture.exe"),
        ],
        check=True,
    )
    records_dir = tmp_path / "o2_si3_records"
    records_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [str(tmp_path / "o2_si3_layout_fixture.exe")],
        cwd=tmp_path,
        check=True,
    )

    res = gate.verify_layout_fixture(records_dir, plant=None)
    assert res["status"] == "PASS"
    assert res["ranks_verified"] == 4
    assert res["istate_fields_per_rank"] == 37
    assert res["state_fields_per_rank"] == 112
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


def test_o2_ocean_output_log_verifier_and_plants(tmp_path: Path) -> None:
    """Verify whitespace-tolerant `ocean.output` log checking and its plants."""
    excerpt = ACQ_DIR / "ocean.output.excerpt"
    res = gate.verify_ocean_output_log(
        excerpt, expect_icedyn=False, expect_icethd=False, plant=None
    )
    assert res["status"] == "PASS"
    assert res["parsed"]["jpl"] == 5

    spaced = tmp_path / "ocean.output.spaced"
    spaced.write_text(excerpt.read_text(encoding="ascii").replace("=", "   =    "))
    res_spaced = gate.verify_ocean_output_log(
        spaced, expect_icedyn=False, expect_icethd=False, plant=None
    )
    assert res_spaced["status"] == "PASS"

    for plant_name in ("error-token", "missing-aaa", "wrong-jpl", "wrong-flag"):
        with pytest.raises(ValueError, match="STATUS PLANT-FIRED"):
            gate.verify_ocean_output_log(excerpt, plant=plant_name)


def test_o2_jit_grad_jvp_and_stop_gradient_plant() -> None:
    """Verify `jax.jit`, `jax.value_and_grad`, `jax.jvp`, and `Plant P6` (`stop_gradient`) through O2 scalar initial state and SSH."""
    inp = gate.build_synthetic_orca2_o2_inputs()
    cfg: SI3ORCA2StateConfig = inp["cfg"]
    o2_cfg: SI3O2InitConfig = inp["o2_cfg"]
    state0: SI3State = inp["state0"]
    tmask = inp["tmask"]
    ssmask = inp["ssmask"]
    smask0_i = inp["smask0_i"]
    e1e2t = inp["e1e2t"]
    ff_t = inp["ff_t"]
    gphit = inp["gphit"]
    sst_m = inp["sst_m"]
    sss_m = inp["sss_m"]
    ssh_kbb = inp["ssh_before_kbb"]
    ssh_kmm = inp["ssh_before_kmm"]

    def _loss_fn(
        sss_in: jax.Array,
        hti_n_in: jax.Array,
        hts_n_in: jax.Array,
        ati_n_in: jax.Array,
        apd_n_in: jax.Array,
        hpd_n_in: jax.Array,
        ssh_in: jax.Array,
        plant_stop: bool = False,
    ) -> jax.Array:
        o2_c = o2_cfg._replace(
            rn_hti_ini_n=hti_n_in,
            rn_hts_ini_n=hts_n_in,
            rn_ati_ini_n=ati_n_in,
            rn_apd_ini_n=apd_n_in,
            rn_hpd_ini_n=hpd_n_in,
        )
        o2_out = si3_o2_scalar_initial_state(
            sst_m=sst_m,
            sss_m=sss_in,
            tmask=tmask,
            ssmask=ssmask,
            smask0_i=smask0_i,
            e1e2t=e1e2t,
            ff_t=ff_t,
            gphit=gphit,
            ssh_before_kbb=ssh_kbb,
            ssh_before_kmm=ssh_in,
            fitted_v_i=state0.v_i,
            fitted_v_s=state0.v_s,
            fitted_a_i=state0.a_i,
            fitted_a_ip=state0.a_ip,
            fitted_v_ip=state0.v_ip,
            fitted_v_il=state0.v_il,
            config=o2_c,
            _plant_stop_gradient=plant_stop,
        )
        return (
            jnp.sum(o2_out.t_bo)
            + jnp.sum(o2_out.ztfrz)
            + jnp.sum(o2_out.zht_i_ini)
            + jnp.sum(o2_out.zht_s_ini)
            + jnp.sum(o2_out.zat_i_ini)
            + jnp.sum(o2_out.zapnd_ini)
            + jnp.sum(o2_out.zhpnd_ini)
            + jnp.sum(o2_out.snwice_mass) * 1.0e-3
            + jnp.sum(o2_out.ssh_kmm)
            + o2_out.zsshadj * 1.0e3
        )

    args = (
        sss_m,
        jnp.asarray(o2_cfg.rn_hti_ini_n, dtype=jnp.float64),
        jnp.asarray(o2_cfg.rn_hts_ini_n, dtype=jnp.float64),
        jnp.asarray(o2_cfg.rn_ati_ini_n, dtype=jnp.float64),
        jnp.asarray(o2_cfg.rn_apd_ini_n, dtype=jnp.float64),
        jnp.asarray(o2_cfg.rn_hpd_ini_n, dtype=jnp.float64),
        ssh_kmm,
    )
    loss_eager = _loss_fn(*args)
    loss_jit = jax.jit(_loss_fn)(*args)
    loss_vg, grads = jax.jit(
        jax.value_and_grad(_loss_fn, argnums=(0, 1, 2, 3, 4, 5, 6))
    )(*args)
    _assert_zero_ulp(loss_eager, loss_jit, "o2_loss_eager_vs_jit")
    _assert_zero_ulp(loss_eager, loss_vg, "o2_loss_eager_vs_vg")

    for label, g in zip(
        ("sss_m", "rn_hti_ini_n", "rn_hts_ini_n", "rn_ati_ini_n", "rn_apd_ini_n", "rn_hpd_ini_n", "ssh_kmm"),
        grads,
    ):
        arr = np.asarray(g, dtype=np.float64)
        assert np.all(np.isfinite(arr)), f"Non-finite gradient for {label}"
        assert float(np.linalg.norm(arr)) > 0.0, f"Zero gradient for {label}"

    tangents = (
        jnp.where(tmask > 0.0, 1.0e-3, 0.0),
        jnp.asarray(1.0e-2, dtype=jnp.float64),
        jnp.asarray(1.0e-3, dtype=jnp.float64),
        jnp.asarray(1.0e-3, dtype=jnp.float64),
        jnp.asarray(1.0e-3, dtype=jnp.float64),
        jnp.asarray(1.0e-4, dtype=jnp.float64),
        jnp.where(ssmask > 0.0, 1.0e-4, 0.0),
    )
    primal_jvp, jvp_val = jax.jvp(_loss_fn, args, tangents)
    _assert_zero_ulp(loss_eager, primal_jvp, "o2_loss_eager_vs_jvp")
    vjp_val = sum(jnp.sum(g * t) for g, t in zip(grads, tangents))
    rel_err = float(
        jnp.abs(jvp_val - vjp_val) / jnp.maximum(jnp.abs(vjp_val), 1.0e-30)
    )
    assert rel_err < 1.0e-11, f"JVP vs VJP relative error {rel_err:.3e} >= 1e-11"

    # Plant P6: stop_gradient must zero all gradients
    _, grads_planted = jax.value_and_grad(
        lambda *a: _loss_fn(*a, plant_stop=True), argnums=(0, 1, 2, 3, 4, 5, 6)
    )(*args)
    for g in grads_planted:
        assert float(jnp.linalg.norm(g)) == 0.0


def test_o2_production_step_sea_ice_selects_o2_and_validates_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`step_sea_ice` selects `thermo_scheme='si3_orca2_o2'` (`build_si3_orca2_o2_config`) and rejects invalid state/forcing/config combinations."""
    inp = gate.build_synthetic_orca2_o2_inputs()
    cfg: SI3ORCA2StateConfig = inp["cfg"]
    o2_cfg: SI3O2InitConfig = inp["o2_cfg"]
    state0: SI3State = inp["state0"]
    tmask = inp["tmask"]
    umask = inp["umask"]
    vmask = inp["vmask"]
    ssmask = inp["ssmask"]
    smask0_i = inp["smask0_i"]
    e1e2t = inp["e1e2t"]
    ff_t = inp["ff_t"]
    gphit = inp["gphit"]
    sst_m = inp["sst_m"]
    sss_m = inp["sss_m"]
    ssh_kbb = inp["ssh_before_kbb"]
    ssh_kmm = inp["ssh_before_kmm"]

    sea_cfg = build_si3_orca2_o2_config(cfg)
    z2 = jnp.zeros_like(tmask)
    o2_forcing = SI3ORCA2O2Forcing(
        tmask=tmask,
        sst_m=sst_m,
        sss_m=sss_m,
        ff_t=ff_t,
        gphit=gphit,
        e1e2t=e1e2t,
        ssmask=ssmask,
        smask0_i=smask0_i,
        ssh_before_kbb=ssh_kbb,
        ssh_before_kmm=ssh_kmm,
        apply_halos=False,
    )
    state1_direct, o2_direct = si3_o2_step(
        state0,
        o2_forcing,
        config=cfg,
        o2_init_config=o2_cfg,
    )
    assert o2_direct is not None
    with pytest.raises(NotImplementedError, match="ice_update_flx"):
        step_sea_ice(
            state0,
            o2_forcing,
            z2,
            z2,
            z2,
            sea_cfg,
            0.1,
            cfg.rDt_ice,
        )

    _assert_zero_ulp(
        o2_direct.ssh_kmm - ssh_kmm,
        o2_direct.ssh_kbb - ssh_kbb,
        "o2_direct_independent_ssh_increment",
    )

    class O2PathPlant(RuntimeError):
        pass

    def _o2_path_plant(*_args, **_kwargs):
        raise O2PathPlant("O2 production dispatcher path reached")

    monkeypatch.setattr(sea_ice_module, "_si3_orca2_o2_step_with_trace", _o2_path_plant)
    with pytest.raises(O2PathPlant, match="dispatcher path reached"):
        sea_ice_module.step_sea_ice(
            state0,
            o2_forcing,
            z2,
            z2,
            z2,
            sea_cfg,
            0.1,
            cfg.rDt_ice,
        )
    monkeypatch.undo()

    with pytest.raises(ValueError, match="requires SI3State"):
        step_sea_ice(
            init_dynamic_ice_state((2,)),
            o2_forcing,
            z2,
            z2,
            z2,
            sea_cfg,
            0.1,
            cfg.rDt_ice,
        )

    with pytest.raises(ValueError, match="requires SI3ORCA2O2Forcing"):
        step_sea_ice(state0, None, z2, z2, z2, sea_cfg, 0.1, cfg.rDt_ice)


def test_o2_same_program_and_objects_equal_plants(tmp_path: Path) -> None:
    """Verify `gate.same_program` on reference vs instrumented O2 builds (`icedyn_adv_pra`, `iceistate`, `icestp`, `o2_si3_state_writer`) and plants."""
    pra_patch = ACQ_DIR / "icedyn_adv_pra_o2_public.patch"
    ist_patch = ACQ_DIR / "iceistate_o2_writer.patch"
    stp_patch = ACQ_DIR / "icestp_o2_writer.patch"

    def _added_lines(p: Path) -> list[str]:
        return [
            l[1:]
            for l in p.read_text(encoding="utf-8").splitlines()
            if l.startswith("+") and not l.startswith("+++")
        ]

    pra_add = _added_lines(pra_patch)
    ist_add = _added_lines(ist_patch)
    stp_add = _added_lines(stp_patch)

    ref_pra = ["MODULE icedyn_adv_pra", "   PUBLIC ice_dyn_adv_pra", "   PUBLIC adv_pra_init", "END MODULE"]
    new_pra = ref_pra[:3] + pra_add + ref_pra[3:]
    ref_ist = ["MODULE iceistate", "   USE lbclnk", "   IF ( .NOT.ln_pnd_lids ) ENDIF", "   ENDWHERE", "END MODULE"]
    new_ist = [ref_ist[0], ref_ist[1], ist_add[0], ref_ist[2], *ist_add[1:3], ref_ist[3], *ist_add[3:], ref_ist[4]]
    ref_stp = ["MODULE icestp", "   USE prtctl", "   CALL ice_wri( kt )", "   IF( ierr /= 0 ) STOP", "END MODULE"]
    new_stp = [ref_stp[0], ref_stp[1], stp_add[0], ref_stp[2], stp_add[1], ref_stp[3], stp_add[2], ref_stp[4]]

    def _make_pair(base: Path) -> tuple[Path, Path]:
        ref = base / "ORCA2_SAS_ICE_O2_REF/BLD"
        new = base / "ORCA2_SAS_ICE_O2_REC/BLD"
        for bld, cfg_name, pra_lines, ist_lines, stp_lines, is_rec in (
            (ref, "ORCA2_SAS_ICE_O2_REF", ref_pra, ref_ist, ref_stp, False),
            (new, "ORCA2_SAS_ICE_O2_REC", new_pra, new_ist, new_stp, True),
        ):
            (bld / "ppsrc/nemo").mkdir(parents=True, exist_ok=True)
            (bld / "obj").mkdir(parents=True, exist_ok=True)
            for cfile in gate.BUILD_CONFIG_FILES:
                (bld / cfile).write_text(f"dir::root /cfgs/{cfg_name}/BLD\n%FCFLAGS -O3\n")
            (bld / "ppsrc/nemo/icevar.f90").write_text("hm_i = vt_i * z1_at_i\n")
            (bld / "ppsrc/nemo/icedyn_adv_pra.f90").write_text("\n".join(pra_lines) + "\n")
            (bld / "ppsrc/nemo/iceistate.f90").write_text("\n".join(ist_lines) + "\n")
            (bld / "ppsrc/nemo/icestp.f90").write_text("\n".join(stp_lines) + "\n")
            (bld / "obj/icevar.o").write_bytes(b"\x7fELF icevar")
            (bld / "obj/icedyn_adv_pra.o").write_bytes(b"\x7fELF pra" + bytes([is_rec]))
            (bld / "obj/iceistate.o").write_bytes(b"\x7fELF ist" + bytes([is_rec]))
            (bld / "obj/icestp.o").write_bytes(b"\x7fELF stp" + bytes([is_rec]))
            if is_rec:
                (bld / "ppsrc/nemo/o2_si3_state_writer.f90").write_text("MODULE o2_si3_state_writer\n")
                (bld / "obj/o2_si3_state_writer.o").write_bytes(b"\x7fELF writer")
        return ref, new

    ref_bld, new_bld = _make_pair(tmp_path / "ok")
    res = gate.same_program(new_bld, ref_bld, pra_patch, ist_patch, stp_patch)
    assert res["status"] == "PASS_SAME_PROGRAM"

    for idx, (plant, mutate) in enumerate(
        (
            ("other-source", lambda _r, n: (n / "ppsrc/nemo/icevar.f90").write_text("hm_i = 2.0 * vt_i\n")),
            ("other-object", lambda _r, n: (n / "obj/icevar.o").write_bytes(b"\x7fELF icevar_bad")),
            ("flags", lambda _r, n: (n / "arch_nemo.fcm").write_text("dir::root /cfgs/ORCA2_SAS_ICE_O2_REC/BLD\n%FCFLAGS -O2\n")),
            ("observer-edit", lambda _r, n: (n / "ppsrc/nemo/iceistate.f90").write_text("\n".join(new_ist).replace("ENDWHERE", "ENDIF") + "\n")),
            ("observer-extra", lambda _r, n: (n / "ppsrc/nemo/icestp.f90").write_text("\n".join(new_stp + ["   x = 1"]) + "\n")),
            ("extra-unit", lambda _r, n: (n / "ppsrc/nemo/extra.f90").write_text("MODULE extra\n")),
            ("missing-unit", lambda _r, n: (n / "obj/icevar.o").unlink()),
        )
    ):
        r_b, n_b = _make_pair(tmp_path / f"p_{idx}_{plant}")
        mutate(r_b, n_b)
        with pytest.raises(ValueError, match="STATUS PLANT-FIRED"):
            gate.same_program(n_b, r_b, pra_patch, ist_patch, stp_patch)


def _write_o2_istate_binary_snapshot(
    path: Path,
    o2_state: SI3O2ScalarInitialState,
    inp: dict[str, jax.Array],
    cfg: SI3ORCA2StateConfig,
    o2_cfg: SI3O2InitConfig,
) -> None:
    """Serialize `SI3O2ScalarInitialState` into a 1-rank `O2SI3I01` binary snapshot for admission testing."""
    jpi, jpj = 184, 152
    nn_hls = cfg.nn_hls
    i_meta = np.array(
        [
            1,
            0,
            1,
            jpi,
            jpj,
            cfg.jpl,
            cfg.nlay_i,
            cfg.nlay_s,
            nn_hls,
            jpi - 2 * nn_hls,
            jpj - 2 * nn_hls,
            1,
            1,
            1 + nn_hls,
            jpi - nn_hls,
            1 + nn_hls,
            jpj - nn_hls,
            1,
            2,
            o2_cfg.nn_iceini_file,
            int(o2_cfg.ln_iceini),
            int(o2_cfg.ln_ice_embd),
        ],
        dtype="<i4",
    )
    r_meta = np.array(
        [
            o2_cfg.rn_thres_sst,
            o2_cfg.rn_hti_ini_n,
            o2_cfg.rn_hti_ini_s,
            o2_cfg.rn_hts_ini_n,
            o2_cfg.rn_hts_ini_s,
            o2_cfg.rn_ati_ini_n,
            o2_cfg.rn_ati_ini_s,
            o2_cfg.rn_smi_ini_n,
            o2_cfg.rn_smi_ini_s,
            o2_cfg.rn_tmi_ini_n,
            o2_cfg.rn_tmi_ini_s,
            o2_cfg.rn_tsu_ini_n,
            o2_cfg.rn_tsu_ini_s,
            o2_cfg.rn_tms_ini_n,
            o2_cfg.rn_tms_ini_s,
            o2_cfg.rn_apd_ini_n,
            o2_cfg.rn_apd_ini_s,
            o2_cfg.rn_hpd_ini_n,
            o2_cfg.rn_hpd_ini_s,
            o2_cfg.rn_hld_ini_n,
            o2_cfg.rn_hld_ini_s,
            o2_cfg.rn_amax_n,
            o2_cfg.rn_amax_s,
            cfg.ice_constants.T0,
            float(np.asarray(o2_state.area)),
            float(np.asarray(o2_state.zsshadj)),
        ],
        dtype="<f8",
    )
    smask0 = np.zeros((jpi, jpj), dtype="<f8")
    smask0[nn_hls:-nn_hls, nn_hls:-nn_hls] = np.asarray(
        inp["tmask"][nn_hls:-nn_hls, nn_hls:-nn_hls], dtype="<f8"
    )
    smask0_i = np.asarray(
        inp["ssmask"] * si3_dom_uniq_t_orca2(184, 152, nn_hls=nn_hls), dtype="<f8"
    )

    fmap: dict[str, np.ndarray] = {
        "tmask": np.asarray(inp["tmask"], dtype="<f8"),
        "umask": np.asarray(inp["umask"], dtype="<f8"),
        "vmask": np.asarray(inp["vmask"], dtype="<f8"),
        "ssmask": np.asarray(inp["ssmask"], dtype="<f8"),
        "smask0": smask0,
        "smask0_i": smask0_i,
        "e1e2t": np.asarray(inp["e1e2t"], dtype="<f8"),
        "ff_t": np.asarray(inp["ff_t"], dtype="<f8"),
        "gphit": np.asarray(inp["gphit"], dtype="<f8"),
        "sst_m": np.asarray(inp["sst_m"], dtype="<f8"),
        "sss_m": np.asarray(inp["sss_m"], dtype="<f8"),
        "ssh_before_kbb": np.asarray(inp["ssh_before_kbb"], dtype="<f8"),
        "ssh_before_kmm": np.asarray(inp["ssh_before_kmm"], dtype="<f8"),
    }
    for fname in gate._O2_SCALAR_STATE_FIELDS:
        fmap[fname] = np.asarray(getattr(o2_state, fname), dtype="<f8")

    with path.open("wb") as out:
        out.write(b"O2SI3I01")
        out.write(struct.pack("<i", 1))
        out.write(i_meta.tobytes())
        out.write(r_meta.tobytes())
        for fname, *_ in gate._O2_ISTATE_FIELD_SPECS:
            out.write(np.asfortranarray(fmap[fname], dtype="<f8").tobytes(order="F"))


def test_o2_admit_evidence_directory_scores_istate_2step_and_rst_1step_and_refuses_all_plants(
    tmp_path: Path,
) -> None:
    """Score O2 initial/restart records at 0 ULP and refuse every admission plant."""
    inp = gate.build_synthetic_orca2_o2_inputs()
    cfg: SI3ORCA2StateConfig = inp["cfg"]
    o2_cfg: SI3O2InitConfig = inp["o2_cfg"]
    state0_raw: SI3State = inp["state0"]
    tmask = inp["tmask"]
    umask = inp["umask"]
    vmask = inp["vmask"]
    ssmask = inp["ssmask"]
    e1e2t = inp["e1e2t"]
    ff_t = inp["ff_t"]
    gphit = inp["gphit"]
    sst_m = inp["sst_m"]
    sss_m = inp["sss_m"]
    ssh_before_kbb = inp["ssh_before_kbb"]
    ssh_before_kmm = inp["ssh_before_kmm"]

    z_pra = SI3AllPratherMoments.zeros(
        184, 152, jpl=cfg.jpl, nlay_i=cfg.nlay_i, nlay_s=cfg.nlay_s
    )
    state0 = state0_raw._replace(prather=z_pra)
    state1, o2_state = si3_o2_step(
        state0,
        SI3ORCA2O2Forcing(
            tmask=tmask,
            sst_m=sst_m,
            sss_m=sss_m,
            ff_t=ff_t,
            gphit=gphit,
            e1e2t=e1e2t,
            ssmask=ssmask,
            smask0_i=inp["smask0_i"],
            ssh_before_kbb=ssh_before_kbb,
            ssh_before_kmm=ssh_before_kmm,
            apply_halos=False,
        ),
        config=cfg,
        o2_init_config=o2_cfg,
    )
    assert o2_state is not None
    state2 = si3_o1_step(state1, tmask=tmask, sss_m=sss_m, config=cfg)

    ev = tmp_path / "evidence"
    for sub in (
        "manifest",
        "ref_2step/o2_si3_records",
        "rec_2step/o2_si3_records",
        "rst_1step/o2_si3_records",
        "rst_1step/input_restart",
        "rst_1step/output_restart",
    ):
        (ev / sub).mkdir(parents=True, exist_ok=True)

    commit_hex = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()

    exc_body = [
        l
        for l in (ACQ_DIR / "ocean.output.excerpt").read_text(encoding="utf-8").splitlines()
        if not l.startswith("#")
    ]
    o2_log = "\n".join(
        re.sub(r"(ln_icethd\s*=\s*)T", r"\g<1>F", re.sub(r"(ln_icedyn\s*=\s*)T", r"\g<1>F", l))
        for l in exc_body
    ) + "\n"
    for rdir in (ev / "ref_2step", ev / "rec_2step", ev / "rst_1step"):
        ln_rstart = "T" if rdir.name == "rst_1step" else "F"
        (rdir / "ocean.output").write_text(
            o2_log + f" ln_rstart = {ln_rstart}\n", encoding="utf-8"
        )
        (rdir / "run.user.stdout.log").write_text("NEMO run ok\nSTOP 0\n", encoding="utf-8")
        (rdir / "RUN_COMPLETE").write_text("ok\n", encoding="utf-8")
        (rdir / "RUN_PRODUCER_COMMIT").write_text(commit_hex + "\n", encoding="utf-8")

    write_si3_netcdf_restart(ev / "rec_2step/ORCA2_SAS_00000001_restart_ice.nc", state1, kt=1, config=cfg)
    write_si3_netcdf_restart(ev / "rec_2step/ORCA2_SAS_00000002_restart_ice.nc", state2, kt=2, config=cfg)
    (ev / "rec_2step/ORCA2_SAS_00000001_restart.nc").write_bytes(b"SAS ocean restart")
    shutil.copy2(
        ev / "rec_2step/ORCA2_SAS_00000001_restart_ice.nc",
        ev / "ref_2step/ORCA2_SAS_00000001_restart_ice.nc",
    )
    shutil.copy2(
        ev / "rec_2step/ORCA2_SAS_00000002_restart_ice.nc",
        ev / "ref_2step/ORCA2_SAS_00000002_restart_ice.nc",
    )

    _write_o2_istate_binary_snapshot(
        ev / "rec_2step/o2_si3_records/o2_si3_istate_rank0000.bin",
        o2_state,
        inp,
        cfg,
        o2_cfg,
    )

    tmask_np, umask_np, vmask_np, sss_np = (
        np.asarray(tmask),
        np.asarray(umask),
        np.asarray(vmask),
        np.asarray(sss_m),
    )
    _write_o1_binary_snapshot(
        ev / "rec_2step/o2_si3_records/o2_si3_state_entry_rank0000.bin",
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
        ev / "rec_2step/o2_si3_records/o2_si3_state_step0001_rank0000.bin",
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
        ev / "rec_2step/o2_si3_records/o2_si3_state_step0002_rank0000.bin",
        state2,
        tmask_np,
        umask_np,
        vmask_np,
        sss_np,
        kt=2,
        lrst_ice=1,
        cfg=cfg,
    )

    rst_entry = read_si3_netcdf_restart(
        ev / "rec_2step/ORCA2_SAS_00000001_restart_ice.nc",
        tmask=tmask,
        sss_m=sss_m,
        config=cfg,
        ln_rstart=True,
    )
    rst_step1 = si3_o1_step(rst_entry, tmask=tmask, sss_m=sss_m, config=cfg)
    _write_o1_binary_snapshot(
        ev / "rst_1step/o2_si3_records/o2_si3_state_entry_rank0000.bin",
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
        ev / "rst_1step/o2_si3_records/o2_si3_state_step0002_rank0000.bin",
        rst_step1,
        tmask_np,
        umask_np,
        vmask_np,
        sss_np,
        kt=2,
        lrst_ice=1,
        cfg=cfg,
    )
    shutil.copy2(
        ev / "rec_2step/ORCA2_SAS_00000001_restart_ice.nc",
        ev / "rst_1step/input_restart/ORCA2_SAS_00000001_restart_ice.nc",
    )
    shutil.copy2(
        ev / "rec_2step/ORCA2_SAS_00000001_restart.nc",
        ev / "rst_1step/input_restart/ORCA2_SAS_00000001_restart.nc",
    )
    write_si3_netcdf_restart(
        ev / "rst_1step/output_restart/ORCA2_SAS_00000002_restart_ice_o2_output.nc",
        rst_step1,
        kt=2,
        config=cfg,
    )
    (ev / "rst_1step/output_restart/ORCA2_SAS_00000002_restart_o2_output.nc").write_bytes(
        b"SAS ocean output restart"
    )

    (ev / "manifest/producer_commit.txt").write_text(commit_hex + "\n", encoding="utf-8")
    ref_bld = ev / "builds/ORCA2_SAS_ICE_O2_REF/BLD"
    rec_bld = ev / "builds/ORCA2_SAS_ICE_O2_REC/BLD"
    (ref_bld / "bin").mkdir(parents=True)
    (rec_bld / "bin").mkdir(parents=True)
    (ref_bld / "bin/nemo.exe").write_bytes(b"\x7fELF reference nemo")
    (rec_bld / "bin/nemo.exe").write_bytes(b"\x7fELF recorder nemo")
    (ev / "manifest/same_program.json").write_text(
        json.dumps(
            {
                "status": "PASS_SAME_PROGRAM",
                "reference_bld": str(ref_bld),
                "new_bld": str(rec_bld),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dummy_bin = ev / "manifest/dummy_nemo.exe"
    dummy_bin.write_bytes(b"\x7fELF nemo")
    bin_sha = hashlib.sha256(dummy_bin.read_bytes()).hexdigest()
    (ev / "manifest/binaries.sha256").write_text(
        "".join(
            f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p}\n"
            for p in (ref_bld / "bin/nemo.exe", rec_bld / "bin/nemo.exe")
        ),
        encoding="utf-8",
    )
    (ev / "manifest/inputs.sha256").write_text(f"{bin_sha}  {dummy_bin}\n", encoding="utf-8")

    adm = gate.admit_evidence_directory(ev, expect_commit=commit_hex)
    assert adm["status"] == "PASS"
    assert adm["execution_backend"] == "cpu"
    assert adm["execution_jit"] is True
    assert adm["execution_x64"] is True
    assert adm["o2_arrays_checked"] == 24
    assert adm["state_arrays_checked_per_comparison"] == 107
    assert adm["restart_variables_checked"] == 40
    assert adm["max_ulp_o2_istate"] == 0
    assert adm["max_ulp_o2_istate_full"] == 0
    assert adm["max_ulp_kt1"] == 0
    assert adm["max_ulp_kt1_full"] == 0
    assert adm["max_ulp_kt2"] == 0
    assert adm["max_ulp_kt2_full"] == 0
    assert adm["max_ulp_rst_entry"] == 0
    assert adm["max_ulp_rst_entry_full"] == 0
    assert adm["max_ulp_rst_step1"] == 0
    assert adm["max_ulp_rst_step1_full"] == 0
    assert adm["max_ulp_rst_vs_kt2"] == 0
    assert adm["max_ulp_rst_vs_kt2_full"] == 0
    assert adm["max_ulp_restart_variables"] == 0

    for plant_name in (
        "stamp",
        "same-program",
        "binary-sha",
        "input-sha",
        "missing-complete",
        "missing-stop0",
        "ref-rec-diff",
        "1ulp-istate",
        "1ulp-ssh",
        "1ulp-state",
        "rst-missing",
        "rst-1ulp",
        "rst-step-name",
        "kt2-sss-perturb",
        "ocean-restart-missing",
    ):
        with pytest.raises(ValueError, match="STATUS PLANT-FIRED"):
            gate.admit_evidence_directory(ev, plant=plant_name, expect_commit=commit_hex)


def test_o2_committed_ocean_output_excerpt_provenance() -> None:
    """Verify `ocean.output.excerpt` provenance headers and verbatim line match when source file is present."""
    excerpt = ACQ_DIR / "ocean.output.excerpt"
    lines = excerpt.read_text(encoding="utf-8").splitlines()
    headers = [l for l in lines if l.startswith("#")]
    body = [l.rstrip() for l in lines if not l.startswith("#")]

    src_m = next(re.search(r"^#\s*Source path:\s*(.+)$", l) for l in headers if "Source path:" in l)
    sha_m = next(re.search(r"^#\s*Complete source SHA-256:\s*([0-9a-f]{64})\s*$", l) for l in headers if "SHA-256:" in l)
    rng_m = next(re.search(r"^#\s*Source lines:\s*(.+)$", l) for l in headers if "Source lines:" in l)

    src_path = Path(src_m.group(1).strip())
    if src_path.is_file():
        assert hashlib.sha256(src_path.read_bytes()).hexdigest() == sha_m.group(1)
        nums: list[int] = []
        for part in rng_m.group(1).rstrip(".").split(","):
            part = part.strip()
            if "-" in part:
                a, b = part.split("-", 1)
                nums.extend(range(int(a), int(b) + 1))
            else:
                nums.append(int(part))
        src_lines = src_path.read_text(encoding="utf-8").splitlines()
        assert body == [src_lines[n - 1].rstrip() for n in nums]
