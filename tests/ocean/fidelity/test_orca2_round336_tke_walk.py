"""Focused controls for the round-336 production TKE boundary walk."""

from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
for _pkg in (REPO, REPO / "packages/core", REPO / "packages/ocean"):
    if str(_pkg) not in sys.path:
        sys.path.insert(0, str(_pkg))
for _mod in list(sys.modules):
    if _mod == "legoesm" or _mod.startswith("legoesm."):
        sys.modules.pop(_mod, None)

import jax.numpy as jnp
import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round336_tke_walk as walk,
)
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks_3d
from legoesm.ocean.physics.vertical_mixing.tke import nemo_bottom_tke_dirichlet


def test_score_is_wet_bitwise_and_reports_maximum() -> None:
    expected = np.array([0.0, 1.0, 2.0])
    actual = expected.copy()
    actual[0] = -0.0
    actual[1] = np.nextafter(actual[1], np.inf)
    row = walk._score(actual, expected, np.array([False, True, True]))
    assert row["wet_words"] == 2
    assert row["unequal_wet_words"] == 1
    assert row["max_abs"] == row["max_abs_unequal"]


def test_walk_uses_two_actual_card_steps_and_orders_post_solve_first() -> None:
    source = Path(walk.__file__).read_text(encoding="utf-8")
    assert "for kt in (1, 2):" in source
    assert "model.step(" in source
    assert source.index('"solve_back", "after_solve"') < source.index(
        '"after_solve": (trace.en_post_sweep')
    assert '"final_avt"' in source


def test_compiled_solver_observer_matches_production_and_fires_plants() -> None:
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    # Synthetic 1x2x31 column: 3 wet levels (k=0,1,2), seafloor BC at k=3.
    a = np.zeros((1, 2, 31), dtype=np.float64)
    b = np.ones((1, 2, 31), dtype=np.float64)
    c = np.zeros((1, 2, 31), dtype=np.float64)
    rhs = np.full((1, 2, 31), 1.0e-6, dtype=np.float64)
    a[..., 1:3] = -0.2
    b[..., 1:3] = 1.5
    c[..., 1:3] = -0.25
    rhs[..., 0] = 2.0e-4
    rhs[..., 1] = 1.5e-4
    rhs[..., 2] = 8.0e-5
    rhs[..., 3] = 5.0e-5
    a[..., 0] = 1.0
    b[..., 0] = 1.0 / rhs[..., 0]
    diag, work, bot, back = walk._solver_boundaries({
        "matrix_lower": a,
        "matrix_diag": b,
        "matrix_upper": c,
        "matrix_rhs_en": rhs,
    })
    wet30 = np.zeros((1, 2, 30), dtype=bool)
    wet30[..., :3] = True
    bot_mask30 = np.zeros((1, 2, 30), dtype=bool)
    bot_mask30[..., 3] = True
    active30 = wet30 | bot_mask30
    all30 = np.ones((1, 2, 30), dtype=bool)
    w_active = jnp.asarray(wet30[..., 1:].astype(np.float64))
    floor = 1.0e-10
    after_solve = np.concatenate(
        [rhs[..., :1], np.maximum(back[..., 1:], floor) * np.asarray(w_active)],
        axis=-1,
    )
    local = {
        "matrix_rhs_en": rhs,
        "solve_diag": np.concatenate([diag, np.zeros((1, 2, 1))], axis=-1),
        "solve_rhs": np.concatenate([work, np.zeros((1, 2, 1))], axis=-1),
        "solve_bottom": np.concatenate([bot, np.zeros((1, 2, 1))], axis=-1),
        "solve_back": np.concatenate([back, np.zeros((1, 2, 1))], axis=-1),
        "after_solve_en": np.concatenate([after_solve, np.zeros((1, 2, 1))], axis=-1),
    }
    a_in = jnp.asarray(a[..., :-1])
    b_in = jnp.asarray(b[..., :-1])
    c_in = jnp.concatenate([jnp.zeros((1, 2, 1)), jnp.asarray(c[..., 1:-1]).at[..., -1].set(0.0)], axis=-1)
    rhs_in = jnp.asarray(rhs[..., :-1])
    sfc_in = jnp.asarray(rhs[..., 0])

    res = walk._evaluate_compiled_solver(
        a_in, b_in, c_in, rhs_in, sfc_in, w_active, floor, True,
        local, wet30, bot_mask30, active30, all30,
    )
    assert res["bit_identical_to_uninstrumented_production"] is True
    assert res["first_non_bit_wet30_solver_stage"] is None
    assert res["first_non_bit_active_solve30_stage"] is None

    for stage in ("matrix_rhs", "solve_diag", "solve_rhs", "solve_bottom", "solve_back", "after_solve"):
        with pytest.raises(walk.WalkError, match=f"compiled solver {stage} plant fired"):
            walk._evaluate_compiled_solver(
                a_in, b_in, c_in, rhs_in, sfc_in, w_active, floor, True,
                local, wet30, bot_mask30, active30, all30, plant_stage=stage,
            )

    # Perturbing the dry seafloor BC slot k=3 leaves wet30 solve_diag/rhs/bottom
    # bit-identical and fires solve_back as first_non_bit_wet30_solver_stage.
    rhs_perturbed = rhs_in.at[0, 0, 3].set(rhs_in[0, 0, 3] + 1.0e-7)
    res_pert = walk._evaluate_compiled_solver(
        a_in, b_in, c_in, rhs_perturbed, sfc_in, w_active, floor, True,
        local, wet30, bot_mask30, active30, all30,
    )
    assert res_pert["first_non_bit_wet30_solver_stage"] == "solve_back"
    assert res_pert["first_non_bit_active_solve30_stage"] == "matrix_rhs"
    assert res_pert["wet30_rows"]["solve_diag"]["unequal_wet_words"] == 0
    assert res_pert["wet30_rows"]["solve_rhs"]["unequal_wet_words"] == 0
    assert res_pert["wet30_rows"]["solve_bottom"]["unequal_wet_words"] == 0
    assert res_pert["wet30_rows"]["solve_back"]["unequal_wet_words"] == 2
    assert res_pert["bottom_bc30_rows"]["matrix_rhs"]["unequal_wet_words"] == 1


def test_admitted_record_compiled_solver_and_bottom_dirichlet_fix() -> None:
    record_dir = Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/"
        "round336/acquisition/orca1ice_tke_internal_kt2_np2"
    )
    if not record_dir.exists():
        pytest.skip("admitted round336 record not present")
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    records = [walk.admission.read_record(record_dir / walk.admission.record_name(r))
               for r in (0, 1)]
    local = {name: walk._assemble(records, name)
             for name in walk.admission.FIELDS if name not in walk.admission.ONE_D}
    full = {name: walk._assemble(records, name, full=True)
            for name in walk.admission.FULL_3D | walk.admission.FULL_2D}
    wet30 = full["tke_wmask"][..., :-1] != 0.0
    mbkt = full["entry_mbkt"].astype(np.int64)
    ssmask = full["entry_ssmask"] != 0.0
    valid_bot = (mbkt >= 1) & (mbkt <= 29) & ssmask
    bottom_bc_mask30 = np.zeros_like(wet30, dtype=bool)
    jj, ii = np.meshgrid(np.arange(wet30.shape[0]), np.arange(wet30.shape[1]), indexing="ij")
    bottom_bc_mask30[jj[valid_bot], ii[valid_bot], mbkt[valid_bot]] = True
    active_solve30 = wet30 | bottom_bc_mask30
    all30 = np.ones_like(wet30, dtype=bool)
    w_active = jnp.asarray(full["tke_wmask"][..., 1:-1], dtype=jnp.float64)

    z0 = jnp.zeros_like(jnp.asarray(local["matrix_rhs_en"][..., :1]))
    o0 = jnp.ones_like(z0)
    a_rec = jnp.concatenate([z0, jnp.asarray(local["matrix_lower"][..., 1:-1])], axis=-1)
    b_rec = jnp.concatenate([o0, jnp.asarray(local["matrix_diag"][..., 1:-1])], axis=-1)
    c_rec = jnp.concatenate(
        [z0, jnp.asarray(local["matrix_upper"][..., 1:-1]).at[..., -1].set(0.0)],
        axis=-1)
    rhs_rec = jnp.asarray(local["matrix_rhs_en"][..., :-1])
    sfc_rec = jnp.asarray(local["matrix_rhs_en"][..., 0])

    rec_res = walk._evaluate_compiled_solver(
        a_rec, b_rec, c_rec, rhs_rec, sfc_rec, w_active, 1.0e-10, True,
        local, wet30, bottom_bc_mask30, active_solve30, all30,
    )
    assert rec_res["first_non_bit_wet30_solver_stage"] is None
    assert rec_res["first_non_bit_active_solve30_stage"] is None
    for stage in ("matrix_rhs", "solve_diag", "solve_rhs", "solve_bottom", "solve_back", "after_solve"):
        assert rec_res["wet30_rows"][stage]["unequal_wet_words"] == 0
        assert rec_res["active_solve30_rows"][stage]["unequal_wet_words"] == 0

    # Verify the 51 north-fold bottom_dirichlet departures and their exact removal.
    from legoesm.ocean.physics.vertical_mixing.tke import TKEConfig
    cfg = TKEConfig(bottom_tke_bc=True, tke_background=1.0e-10)
    tmask = jnp.asarray(full["entry_tmask"][..., :-1], dtype=jnp.float64)
    uu_east = jnp.asarray(full["entry_uu_kbb"][..., :-1], dtype=jnp.float64)
    u_f = jnp.concatenate([uu_east[:, -1:, :], uu_east], axis=1)
    vv_north = jnp.asarray(full["entry_vv_kbb"][..., :-1], dtype=jnp.float64)
    v_f = jnp.concatenate([jnp.zeros_like(vv_north[:1]), vv_north], axis=0)
    um3, vm3_unfixed = compute_face_masks_3d(tmask)
    um3 = um3.astype(jnp.float64)
    vm3_unfixed = vm3_unfixed.astype(jnp.float64)
    vm3_fixed = vm3_unfixed.at[-1].set(
        jnp.asarray(full["entry_vmask"][-1, :, :-1], dtype=jnp.float64))
    bl_idx = jnp.maximum(jnp.asarray(mbkt - 1), 0)[..., None]
    _at_bottom = lambda arr: jnp.take_along_axis(arr, bl_idx, axis=-1)[..., 0]
    ssmask_f = _at_bottom(tmask)
    r_t = jnp.where(ssmask_f > 0.0, 0.0004, 0.0)

    def _eval_bottom_bc(vm_use):
        m_uw, m_ue = _at_bottom(um3[:, :-1, :]), _at_bottom(um3[:, 1:, :])
        m_vs, m_vn = _at_bottom(vm_use[:-1, :, :]), _at_bottom(vm_use[1:, :, :])
        u_sum = (2.0 - m_uw * m_ue) * (
            _at_bottom(u_f[:, :-1, :]) * m_uw + _at_bottom(u_f[:, 1:, :]) * m_ue)
        v_sum = (2.0 - m_vs * m_vn) * (
            _at_bottom(v_f[:-1, :, :]) * m_vs + _at_bottom(v_f[1:, :, :]) * m_vn)
        return np.asarray(ssmask_f * nemo_bottom_tke_dirichlet(r_t, u_sum, v_sum, cfg))

    nemo_bot_2d = np.zeros_like(ssmask, dtype=np.float64)
    nemo_bot_2d[valid_bot] = local["matrix_rhs_en"][jj[valid_bot], ii[valid_bot], mbkt[valid_bot]]
    unfixed_score = walk._score_with_ranks(_eval_bottom_bc(vm3_unfixed), nemo_bot_2d, valid_bot)
    fixed_score = walk._score_with_ranks(_eval_bottom_bc(vm3_fixed), nemo_bot_2d, valid_bot)
    assert unfixed_score["wet_words"] == 11676
    assert unfixed_score["unequal_wet_words"] == 51
    assert unfixed_score["rank0_unequal_wet_words"] == 26
    assert unfixed_score["rank1_unequal_wet_words"] == 25
    assert fixed_score["unequal_wet_words"] == 0
    assert fixed_score["max_abs"] == 0.0


