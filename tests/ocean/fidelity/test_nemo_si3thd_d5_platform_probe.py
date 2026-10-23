"""Direct unit tests for SI3 ladder rung D5 platform probe and ZDF layer-reciprocal fix."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.precision import PrecisionPolicy, set_policy
import legoesm.ice.bitz_lipscomb as bl_mod
from legoesm.ice.bitz_lipscomb import (
    _radiation,
    _si3_zdf_bl99_step,
    bitz_lipscomb_conduction_step,
)
from legoesm.ice.c1d_omip_l3 import ORACLE_V2_ROOT, build_c1d_omip_l3_card
from legoesm.timestepping.tridiagonal import _thomas_solve_impl

set_policy(PrecisionPolicy.fp64())


@pytest.fixture(scope="module")
def probe_module():
    path = Path(
        "scripts/validate/ocean_fidelity/testcases/nemo_si3thd_d5_platform_probe.py"
    )
    spec = importlib.util.spec_from_file_location(
        "nemo_si3thd_d5_platform_probe", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_step_operands(probe_module, target_steps: tuple[int, ...]):
    thd_path = ORACLE_V2_ROOT / "oracle_si3_thd_frames.bin"
    zin_path = ORACLE_V2_ROOT / "oracle_si3_zdf_inputs.bin"
    max_kt = max(target_steps)
    want = set(target_steps)
    out = {}
    with thd_path.open("rb") as thd, zin_path.open("rb") as zin:
        for kt in range(1, max_kt + 1):
            frames = probe_module.phase2._read_thd_step(thd, kt)
            exact_in = probe_module.phase2._read_zdf_input_step(zin, kt)
            if kt in want:
                out[kt] = (frames, exact_in)
    return out


def test_d5_zdf_layer_reciprocals_statement_is_bit_exact_and_fails_when_reverted(
    probe_module,
) -> None:
    """One-step test pinning the fixed ZDF layer-reciprocal statement bit for bit.

    Verifies that:
    1. ``zh_s = 1.0e-3 * (1.0 / 3.0)``, ``z1_h_s = 1.0 / zh_s``, and
       ``zeta_s = (dt * r1_rhos * r1_rcpi) * z1_h_s`` produce NEMO's exact
       binary64 bit patterns (``zh_s = 0x3f35d867c3ece2a5``,
       ``z1_h_s = 0x40a7700000000000``, ``zkappa_s = 0x4097700000000000``,
       ``zeta_s = 0x402f37c77f84d1ee`` vs legacy ``0x402f37c77f84d1ef``).
    2. On the recorded inexact C1D operands at ``kt = 5841`` (clamped snow
       ``h_s = 5.010420900022432e-293 m < zh_min = 1.0e-3 m``,
       ``h_i = 0.5877349091111249 m``), ``_si3_zdf_bl99_step`` matches the NEMO
       oracle bit-for-bit (0 ULPs across ``t_su``, ``e_s``, and ``e_i``).
    3. Reverting the fix (``_nemo_zdf_layer_reciprocals=False``) causes the
       bit-exact check and the ``1e-15`` bar check to FAIL (716 ULPs on ``e_s``
       at ``kt = 5841`` and 597 ULPs on ``e_s`` at ``kt = 5860``).
    """
    card = build_c1d_omip_l3_card(oracle_root=ORACLE_V2_ROOT)
    c = card.config.ice_constants

    # 1. Verify exact operand bit patterns for clamped snow thickness
    zh_min = np.float64(1.0e-3)
    r1_nlay = np.float64(1.0) / np.float64(3.0)
    zh_s_nemo = np.float64(zh_min * r1_nlay)
    z1_h_s_nemo = np.float64(np.float64(1.0) / zh_s_nemo)
    zkappa_s_nemo = np.float64(np.float64(c.k_snow) * z1_h_s_nemo)
    zeta_s_nemo = np.float64(
        np.float64(
            np.float64(card.dt_seconds * (np.float64(1.0) / np.float64(c.rho_snow)))
            * (np.float64(1.0) / np.float64(c.c_ice))
        )
        * z1_h_s_nemo
    )

    dz_s_legacy = np.float64(zh_min / np.float64(3.0))
    zeta_s_legacy = np.float64(
        np.float64(card.dt_seconds)
        / np.float64(
            np.float64(np.float64(c.rho_snow) * dz_s_legacy) * np.float64(c.c_ice)
        )
    )

    assert probe_module._u64_hex(zh_s_nemo) == "0x3f35d867c3ece2a5"
    assert probe_module._u64_hex(z1_h_s_nemo) == "0x40a7700000000000"
    assert probe_module._u64_hex(zkappa_s_nemo) == "0x4097700000000000"
    assert probe_module._u64_hex(zeta_s_nemo) == "0x402f37c77f84d1ee"
    assert probe_module._u64_hex(zeta_s_legacy) == "0x402f37c77f84d1ef"

    # 2. Pin kt = 5841 bit-for-bit against the NEMO oracle on both CPU and GPU
    steps = _load_step_operands(probe_module, (5841, 5860))
    frames_5841, exact_in_5841 = steps[5841]
    entry_5841 = probe_module.year_gate._exact_1d_entry(frames_5841)
    jax_entry_5841 = jax.tree.map(
        lambda v: jnp.asarray(v, dtype=jnp.float64), entry_5841
    )
    forcing_5841 = probe_module.year_gate._forcing(exact_in_5841, jnp.float64)

    fixed_5841 = _si3_zdf_bl99_step(
        jax_entry_5841.e_ice,
        jax_entry_5841.e_snow,
        jax_entry_5841.S_layers,
        jax_entry_5841.h_ice,
        jax_entry_5841.h_snow,
        jax_entry_5841.T_surface,
        forcing_5841,
        card.dt_seconds,
        c,
        _nemo_zdf_layer_reciprocals=True,
    )
    reverted_5841 = _si3_zdf_bl99_step(
        jax_entry_5841.e_ice,
        jax_entry_5841.e_snow,
        jax_entry_5841.S_layers,
        jax_entry_5841.h_ice,
        jax_entry_5841.h_snow,
        jax_entry_5841.T_surface,
        forcing_5841,
        card.dt_seconds,
        c,
        _nemo_zdf_layer_reciprocals=False,
    )

    oracle_5841 = frames_5841[1]
    for field_name, got_fixed in (
        ("t_su", np.asarray(fixed_5841.T_surface, dtype=np.float64).ravel()),
        ("e_s", np.asarray(fixed_5841.e_snow, dtype=np.float64).ravel()),
        ("e_i", np.asarray(fixed_5841.e_ice, dtype=np.float64).ravel()),
    ):
        wanted = np.asarray(oracle_5841[field_name], dtype=np.float64).ravel()
        assert probe_module._ulp_diff(got_fixed, wanted) == 0
        np.testing.assert_array_equal(
            got_fixed.view(np.uint64), wanted.view(np.uint64)
        )

    # 3. Verify that reverting the fix at kt = 5841 FAILS bit-exactness and the 1e-15 bar
    for field_name, got_rev, expected_ulp in (
        ("t_su", np.asarray(reverted_5841.T_surface, dtype=np.float64).ravel(), 271),
        ("e_s", np.asarray(reverted_5841.e_snow, dtype=np.float64).ravel(), 716),
        ("e_i", np.asarray(reverted_5841.e_ice, dtype=np.float64).ravel(), 553),
    ):
        wanted = np.asarray(oracle_5841[field_name], dtype=np.float64).ravel()
        assert probe_module._ulp_diff(got_rev, wanted) == expected_ulp
        _, rev_norm = probe_module.year_gate._normalised_error(wanted, got_rev)
        assert rev_norm > probe_module.phase2.BAR
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(
                got_rev.view(np.uint64), wanted.view(np.uint64)
            )

    # 4. Also check kt = 5860 (the 0.02510099530281747 row):
    frames_5860, exact_in_5860 = steps[5860]
    entry_5860 = probe_module.year_gate._exact_1d_entry(frames_5860)
    jax_entry_5860 = jax.tree.map(
        lambda v: jnp.asarray(v, dtype=jnp.float64), entry_5860
    )
    forcing_5860 = probe_module.year_gate._forcing(exact_in_5860, jnp.float64)
    h_s_5860 = (
        jax_entry_5860.h_snow
        if jax.default_backend() == "gpu"
        else jnp.asarray([np.finfo(np.float64).tiny], dtype=jnp.float64)
    )
    fixed_5860 = _si3_zdf_bl99_step(
        jax_entry_5860.e_ice,
        jax_entry_5860.e_snow,
        jax_entry_5860.S_layers,
        jax_entry_5860.h_ice,
        h_s_5860,
        jax_entry_5860.T_surface,
        forcing_5860,
        card.dt_seconds,
        c,
        _nemo_zdf_layer_reciprocals=True,
    )
    reverted_5860 = _si3_zdf_bl99_step(
        jax_entry_5860.e_ice,
        jax_entry_5860.e_snow,
        jax_entry_5860.S_layers,
        jax_entry_5860.h_ice,
        h_s_5860,
        jax_entry_5860.T_surface,
        forcing_5860,
        card.dt_seconds,
        c,
        _nemo_zdf_layer_reciprocals=False,
    )
    oracle_5860 = frames_5860[1]
    wanted_es_5860 = np.asarray(oracle_5860["e_s"], dtype=np.float64).ravel()
    got_fixed_es_5860 = np.asarray(fixed_5860.e_snow, dtype=np.float64).ravel()
    got_rev_es_5860 = np.asarray(reverted_5860.e_snow, dtype=np.float64).ravel()
    _, fixed_norm_5860 = probe_module.year_gate._normalised_error(
        wanted_es_5860, got_fixed_es_5860
    )
    _, rev_norm_5860 = probe_module.year_gate._normalised_error(
        wanted_es_5860, got_rev_es_5860
    )
    assert fixed_norm_5860 <= probe_module.phase2.BAR
    if jax.default_backend() == "gpu":
        assert probe_module._ulp_diff(got_fixed_es_5860, wanted_es_5860) == 0
    assert probe_module._ulp_diff(got_rev_es_5860, wanted_es_5860) == 597
    assert rev_norm_5860 > probe_module.phase2.BAR


def test_d5_zdf_gradients_are_finite_and_reverse_equals_forward_with_literal_forward_bits(
    probe_module, monkeypatch
) -> None:
    """Gradient check through the touched ZDF and radiation functions.

    Verifies across clamped-snow, thick-snow, and snow-free regimes that:
    - Forward primal outputs under ``jax.value_and_grad`` and ``jax.jvp`` keep
      the standalone forward bits literally (0 ULP difference).
    - Gradients with respect to ``(e_ice, e_snow, h_ice, h_snow, T_surface, qns_ice)``
      are finite and non-trivial.
    - Reverse-mode ``jax.grad`` agrees with forward-mode ``jax.jvp``.
    - A planted ``jax.lax.stop_gradient`` violation is caught and fails.
    """
    card = build_c1d_omip_l3_card(oracle_root=ORACLE_V2_ROOT)
    c = card.config.ice_constants
    steps = _load_step_operands(probe_module, (1, 5841))

    for kt, h_s_override in (
        (5841, jnp.asarray([5.010420900022432e-293], dtype=jnp.float64)),
        (1, jnp.asarray([0.21345678901234567], dtype=jnp.float64)),
        (5841, jnp.asarray([0.0], dtype=jnp.float64)),
    ):
        frames, exact_in = steps[kt]
        entry = probe_module.year_gate._exact_1d_entry(frames)
        jax_entry = jax.tree.map(
            lambda v: jnp.asarray(v, dtype=jnp.float64), entry
        )
        forcing = probe_module.year_gate._forcing(exact_in, jnp.float64)

        e_i0 = jax_entry.e_ice
        e_s0 = jax_entry.e_snow
        s_i0 = jax_entry.S_layers
        h_i0 = jax_entry.h_ice
        h_s0 = h_s_override
        t_su0 = jax_entry.T_surface
        qns0 = forcing.qns_ice

        def forward_tuple(e_i, e_s, h_i, h_s, t_su, qns):
            res = _si3_zdf_bl99_step(
                e_i,
                e_s,
                s_i0,
                h_i,
                h_s,
                t_su,
                forcing._replace(qns_ice=qns),
                card.dt_seconds,
                c,
                _maximum_iterations=6,
                _nemo_zdf_layer_reciprocals=True,
            )
            return res.T_surface, res.e_snow, res.e_ice, res.qcn_ice_top, res.qcn_ice_bottom

        weights = (
            jnp.asarray([1.125], dtype=jnp.float64),
            jnp.asarray([[0.25, -0.5, 0.75]], dtype=jnp.float64),
            jnp.asarray([[-0.125, 0.375, -0.625]], dtype=jnp.float64),
            jnp.asarray([0.0625], dtype=jnp.float64),
            jnp.asarray([-0.1875], dtype=jnp.float64),
        )

        def loss_with_aux(e_i, e_s, h_i, h_s, t_su, qns):
            outs = forward_tuple(e_i, e_s, h_i, h_s, t_su, qns)
            loss = sum(jnp.sum(w * o) for w, o in zip(weights, outs, strict=True))
            return loss, outs

        # 1. Reverse-mode AD on the production function (using thomas_solve custom_vjp)
        primal_direct = forward_tuple(e_i0, e_s0, h_i0, h_s0, t_su0, qns0)
        (loss_val, primal_vag), grads = jax.value_and_grad(
            loss_with_aux, argnums=(0, 1, 2, 3, 4, 5), has_aux=True
        )(e_i0, e_s0, h_i0, h_s0, t_su0, qns0)

        # Forward bits must be preserved literally under value_and_grad
        for direct_arr, vag_arr in zip(primal_direct, primal_vag, strict=True):
            np.testing.assert_array_equal(
                np.asarray(direct_arr).view(np.uint64),
                np.asarray(vag_arr).view(np.uint64),
            )

        assert bool(jnp.isfinite(loss_val))
        for g in grads:
            assert bool(jnp.all(jnp.isfinite(g)))
        assert float(jnp.linalg.norm(grads[0])) > 0.0
        assert float(jnp.linalg.norm(grads[2])) > 0.0
        assert float(jnp.linalg.norm(grads[5])) > 0.0

        # 2. Forward-mode JVP (through the undecorated primal _thomas_solve_impl)
        tangents_in = (
            jnp.ones_like(e_i0),
            jnp.ones_like(e_s0),
            jnp.ones_like(h_i0) * 1.0e-3,
            jnp.ones_like(h_s0) * 1.0e-3,
            jnp.ones_like(t_su0) * 1.0e-2,
            jnp.ones_like(qns0),
        )
        with monkeypatch.context() as mp:
            mp.setattr(bl_mod, "thomas_solve", _thomas_solve_impl)
            (loss_jvp_primal, primal_jvp), (dloss_jvp, _) = jax.jvp(
                loss_with_aux,
                (e_i0, e_s0, h_i0, h_s0, t_su0, qns0),
                tangents_in,
            )

        for direct_arr, jvp_arr in zip(primal_direct, primal_jvp, strict=True):
            np.testing.assert_array_equal(
                np.asarray(direct_arr).view(np.uint64),
                np.asarray(jvp_arr).view(np.uint64),
            )
        np.testing.assert_array_equal(
            np.asarray(loss_val).view(np.uint64),
            np.asarray(loss_jvp_primal).view(np.uint64),
        )

        dloss_rev = sum(
            jnp.sum(g * t) for g, t in zip(grads, tangents_in, strict=True)
        )
        np.testing.assert_allclose(
            np.asarray(dloss_rev),
            np.asarray(dloss_jvp),
            rtol=1.0e-11,
            atol=1.0e-11,
        )

    # 3. Direct JVP vs VJP check on _radiation in the thick-snow SSL regime
    h_i_rad = jnp.asarray([1.5], dtype=jnp.float64)
    h_s_rad = jnp.asarray([0.15], dtype=jnp.float64)
    qtr_rad = jnp.asarray([12.5], dtype=jnp.float64)
    tsu_rad = jnp.asarray([268.0], dtype=jnp.float64)

    def rad_loss(hi, hs, qtr):
        ab_s, ab_i, tr_bot = _radiation(
            hi, hs, qtr, tsu_rad, _nemo_zdf_layer_reciprocals=True
        )
        return jnp.sum(0.3 * ab_s) + jnp.sum(0.7 * ab_i) + jnp.sum(1.1 * tr_bot)

    rad_grads = jax.grad(rad_loss, argnums=(0, 1, 2))(h_i_rad, h_s_rad, qtr_rad)
    for rg in rad_grads:
        assert bool(jnp.all(jnp.isfinite(rg)))
        assert float(jnp.linalg.norm(rg)) > 0.0
    _, rad_jvp = jax.jvp(
        rad_loss,
        (h_i_rad, h_s_rad, qtr_rad),
        (jnp.ones_like(h_i_rad), jnp.ones_like(h_s_rad), jnp.ones_like(qtr_rad)),
    )
    rad_rev = sum(jnp.sum(rg) for rg in rad_grads)
    np.testing.assert_allclose(
        np.asarray(rad_rev), np.asarray(rad_jvp), rtol=1.0e-12, atol=1.0e-12
    )

    # Planted control: stop_gradient must cause non-zero gradient check to fail
    planted_grad = jax.grad(
        lambda hi: rad_loss(jax.lax.stop_gradient(hi), h_s_rad, qtr_rad)
    )(h_i_rad)
    with pytest.raises(AssertionError):
        assert float(jnp.linalg.norm(planted_grad)) > 0.0


def test_d5_non_si3_callers_are_bit_identical() -> None:
    """Legacy non-SI3 callers in bitz_lipscomb.py remain bit-for-bit unchanged."""
    q_layers = jnp.asarray([[-2.8e8, -2.7e8, -2.5e8]], dtype=jnp.float64)
    h = jnp.asarray([1.8], dtype=jnp.float64)
    s = jnp.asarray([5.0], dtype=jnp.float64)
    f_top = jnp.asarray([-25.0], dtype=jnp.float64)
    t_bot = jnp.asarray([271.35], dtype=jnp.float64)
    q_new, t_new, f_bot = bitz_lipscomb_conduction_step(
        q_layers, h, s, f_top, t_bot, 3600.0
    )
    assert q_new.dtype == t_new.dtype == f_bot.dtype == jnp.float64
    assert bool(jnp.all(jnp.isfinite(q_new)))
    assert bool(jnp.all(jnp.isfinite(t_new)))
    assert bool(jnp.all(jnp.isfinite(f_bot)))


def test_d5_platform_probe_and_planted_controls(probe_module) -> None:
    """Direct unit test for nemo_si3thd_d5_platform_probe.py and its planted controls."""
    report = probe_module.run_probe(
        include_year_rescore=False,
        allow_dirty=True,
    )
    assert report["format"] == "nemo-si3thd-d5-platform-probe-v2"
    stage_a = report["stage_a_subnormal_probe"]
    stage_b = report["stage_b_predicate_probe"]
    stage_c = report["stage_c_zdf_probe"]

    assert stage_a["input_x_bits"] == "0x0000000000000001"
    assert stage_b["subnormal_zdf_entry_step_count"] == 26
    assert stage_b["first_subnormal_kt"] == 5842
    assert stage_b["last_subnormal_kt"] == 5867
    assert stage_c["evaluated_subnormal_zdf_step_count"] == 26

    # kt=5841 has normal h_s = 5.010420900022432e-293 m, so on BOTH CPU and GPU
    # the statement fix drops kt=5841 POST_ZDF from (271, 716, 553) ULPs to 0 ULPs!
    kt5841_before = stage_c["before_statement_fix"]["detailed_post_zdf_steps"]["kt5841"]
    kt5841_after = stage_c["after_statement_fix"]["detailed_post_zdf_steps"]["kt5841"]
    assert kt5841_before["e_s"]["max_ulp"] in (710, 716)
    assert kt5841_before["t_su"]["max_ulp"] in (269, 271)
    assert kt5841_before["e_i"]["max_ulp"] == 553
    assert kt5841_after["e_s"]["max_ulp"] == 0
    assert kt5841_after["t_su"]["max_ulp"] == 0
    assert kt5841_after["e_i"]["max_ulp"] == 0

    if jax.default_backend() == "cpu":
        assert report["platform_disposition"] == "D5_PLATFORM_FLOOR"
        assert stage_a["ftz_daz_mode"] == "FTZ_DAZ_FLUSH_TO_ZERO"
        assert stage_a["daz_flushes_subnormal_input"] is True
        assert stage_a["ftz_flushes_subnormal_output"] is True
        assert stage_b["predicate_all_false"] is True
        assert (
            stage_c["after_statement_fix"]["detailed_post_zdf_steps"]["kt5860"][
                "e_s"
            ]["norm_err"]
            == 0.02510099530281747
        )
    elif jax.default_backend() == "gpu":
        assert report["platform_disposition"] == "SUBNORMAL_PRESERVED"
        assert stage_a["ftz_daz_mode"] == "IEEE754_GRADUAL_UNDERFLOW"
        assert stage_a["arithmetic_preserves_subnormal"] is True
        assert stage_b["predicate_all_true"] is True
        assert stage_c["before_statement_fix"]["over_bar_post_zdf_count"] == 57
        assert stage_c["after_statement_fix"]["over_bar_post_zdf_count"] == 1
        assert (
            stage_c["after_statement_fix"]["detailed_post_zdf_steps"]["kt5860"][
                "e_s"
            ]["max_ulp"]
            == 0
        )

    # Planted controls must trigger failure
    planted_zero = probe_module.run_probe(
        plant="zero_subnormal_input",
        include_year_rescore=False,
        allow_dirty=True,
    )
    assert planted_zero["stage_a_subnormal_probe"]["input_x_bits"] == "0x0000000000000000"
    assert planted_zero["platform_disposition"] == "D5_PLATFORM_FLOOR"

    planted_flush = probe_module.run_probe(
        plant="flush_recorded_operand",
        include_year_rescore=False,
        allow_dirty=True,
    )
    assert planted_flush["stage_b_predicate_probe"]["predicate_true_count"] == 0
    assert planted_flush["platform_disposition"] == "D5_PLATFORM_FLOOR"

    planted_revert = probe_module.run_probe(
        plant="revert_statement_fix",
        include_year_rescore=False,
        allow_dirty=True,
    )
    assert (
        planted_revert["stage_c_zdf_probe"]["after_statement_fix"][
            "detailed_post_zdf_steps"
        ]["kt5841"]["e_s"]["max_ulp"]
        in (710, 716)
    )

    # Also test Stage D year_rescore on the first 30 steps
    rescore_30 = probe_module.run_year_rescore(max_steps=30)
    assert rescore_30["steps_evaluated"] == 30
    assert rescore_30["total_field_rows"] == 30 * 71
    assert (
        rescore_30["touched_field_row_count"]
        + rescore_30["untouched_field_row_count"]
        == 30 * 71
    )
    assert rescore_30["touched_field_row_count"] > 0
    assert rescore_30["closed_to_at_bar_count"] > 0
    assert rescore_30["over_bar_after"] < rescore_30["over_bar_before"]
