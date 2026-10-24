"""Direct unit tests for SI3 ladder rung D5 platform probe and ZDF layer-reciprocal fix."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import types

import jax

REPO_ROOT = Path(__file__).resolve().parents[3]
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

    # 2. Pin kt = 5841 bit-for-bit against the NEMO oracle on both CPU and GPU,
    #    and kt = 4242 post-solve conductive flux (iteration-frozen zkappa,
    #    icethd_zdf_bl99.f90:782-784) against oracle_si3_dh_operands.bin and POST_DH.
    steps = _load_step_operands(probe_module, (4242, 5841, 5860))
    dh_ops = probe_module.year_gate._read_dh_operands(
        ORACLE_V2_ROOT / "oracle_si3_dh_operands.bin"
    )
    frames_4242, exact_in_4242 = steps[4242]
    entry_4242 = probe_module.year_gate._exact_1d_entry(
        frames_4242, sss_m=exact_in_4242["sss"]
    )
    jax_entry_4242 = jax.tree.map(
        lambda v: jnp.asarray(v, dtype=jnp.float64), entry_4242
    )
    forcing_4242 = probe_module.year_gate._forcing(exact_in_4242, jnp.float64)
    trace_fixed_4242 = bl_mod.si3_column_step_arrays(
        jax_entry_4242,
        forcing_4242,
        card.dt_seconds,
        c,
        _nemo_zdf_layer_reciprocals=True,
    )
    trace_rev_4242 = bl_mod.si3_column_step_arrays(
        jax_entry_4242,
        forcing_4242,
        card.dt_seconds,
        c,
        _nemo_zdf_layer_reciprocals=False,
    )
    zdf_fixed_4242 = _si3_zdf_bl99_step(
        jax_entry_4242.e_ice,
        jax_entry_4242.e_snow,
        jax_entry_4242.S_layers,
        jax_entry_4242.h_ice,
        jax_entry_4242.h_snow,
        jax_entry_4242.T_surface,
        forcing_4242,
        card.dt_seconds,
        c,
        _nemo_zdf_layer_reciprocals=True,
    )
    zdf_rev_4242 = _si3_zdf_bl99_step(
        jax_entry_4242.e_ice,
        jax_entry_4242.e_snow,
        jax_entry_4242.S_layers,
        jax_entry_4242.h_ice,
        jax_entry_4242.h_snow,
        jax_entry_4242.T_surface,
        forcing_4242,
        card.dt_seconds,
        c,
        _nemo_zdf_layer_reciprocals=False,
    )
    zq_oracle_4242 = np.asarray([dh_ops[4242][0]["zq_top"]], dtype=np.float64)
    zq_fixed_4242 = np.asarray(
        (
            zdf_fixed_4242.qns_ice
            + forcing_4242.qsr_ice
            - forcing_4242.qtr_ice_top
            - zdf_fixed_4242.qcn_ice_top
        )
        * card.dt_seconds,
        dtype=np.float64,
    ).ravel()
    zq_rev_4242 = np.asarray(
        (
            zdf_rev_4242.qns_ice
            + forcing_4242.qsr_ice
            - forcing_4242.qtr_ice_top
            - zdf_rev_4242.qcn_ice_top
        )
        * card.dt_seconds,
        dtype=np.float64,
    ).ravel()
    assert probe_module._ulp_diff(zq_fixed_4242, zq_oracle_4242) == 0
    assert probe_module._ulp_diff(zq_rev_4242, zq_oracle_4242) > 100_000_000
    hi_oracle_4242 = np.asarray(frames_4242[2]["h_i"], dtype=np.float64).ravel()
    hi_fixed_4242 = np.asarray(
        trace_fixed_4242.post_dh.h_ice, dtype=np.float64
    ).ravel()
    hi_rev_4242 = np.asarray(
        trace_rev_4242.post_dh.h_ice, dtype=np.float64
    ).ravel()
    assert probe_module._ulp_diff(hi_fixed_4242, hi_oracle_4242) == 0
    assert probe_module._ulp_diff(hi_rev_4242, hi_oracle_4242) in (140889, 140890)
    _, hi_rev_norm_4242 = probe_module.year_gate._normalised_error(
        hi_oracle_4242, hi_rev_4242
    )
    assert hi_rev_norm_4242 > probe_module.phase2.BAR
    frames_5841, exact_in_5841 = steps[5841]
    entry_5841 = probe_module.year_gate._exact_1d_entry(
        frames_5841, sss_m=exact_in_5841["sss"]
    )
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
    for field_name, got_rev, expected_ulps in (
        ("t_su", np.asarray(reverted_5841.T_surface, dtype=np.float64).ravel(), (269, 271)),
        ("e_s", np.asarray(reverted_5841.e_snow, dtype=np.float64).ravel(), (710, 716)),
        ("e_i", np.asarray(reverted_5841.e_ice, dtype=np.float64).ravel(), (553,)),
    ):
        wanted = np.asarray(oracle_5841[field_name], dtype=np.float64).ravel()
        assert probe_module._ulp_diff(got_rev, wanted) in expected_ulps
        _, rev_norm = probe_module.year_gate._normalised_error(wanted, got_rev)
        assert rev_norm > probe_module.phase2.BAR
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(
                got_rev.view(np.uint64), wanted.view(np.uint64)
            )

    # 4. Also check kt = 5860 (the 0.02510099530281747 row):
    frames_5860, exact_in_5860 = steps[5860]
    entry_5860 = probe_module.year_gate._exact_1d_entry(
        frames_5860, sss_m=exact_in_5860["sss"]
    )
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
        entry = probe_module.year_gate._exact_1d_entry(
            frames, sss_m=exact_in["sss"]
        )
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


def _parent_commit_bitz_lipscomb_conduction_step(
    q_layers, h, S, F_top, T_bottom_K, dt, *, plant_reciprocal_dz: bool = False
):
    """Exact parent-commit (a572f4b254) implementation of bitz_lipscomb_conduction_step."""
    n = q_layers.shape[-1]
    S_b = jnp.broadcast_to(
        S[..., None] if S.ndim == h.ndim else S, q_layers.shape
    )
    if plant_reciprocal_dz:
        dz = jnp.maximum(h, bl_mod._H_MIN) * (1.0 / n)
    else:
        dz = jnp.maximum(h, bl_mod._H_MIN) / n
    dz_e = dz[..., None]
    T_old = bl_mod.ice_temperature_from_enthalpy(q_layers, S_b)
    c_i = bl_mod.ice_specific_heat(T_old, S_b)
    k_c = bl_mod.ice_thermal_conductivity(T_old, S_b)
    k_iface = (
        2.0
        * k_c[..., :-1]
        * k_c[..., 1:]
        / (k_c[..., :-1] + k_c[..., 1:])
    )
    heat_cap = bl_mod.canonical_constants.rho_ice * c_i * dz_e / dt
    if plant_reciprocal_dz:
        g_int = k_iface * (1.0 / dz_e)
        g_bot = 2.0 * k_c[..., -1] * (1.0 / dz)
    else:
        g_int = k_iface / dz_e
        g_bot = 2.0 * k_c[..., -1] / dz
    zero = jnp.zeros_like(heat_cap[..., :1])
    a = jnp.concatenate([zero, -g_int], axis=-1)
    c = jnp.concatenate([-g_int, zero], axis=-1)
    g_up = jnp.concatenate([zero, g_int], axis=-1)
    g_dn = jnp.concatenate([g_int, zero], axis=-1)
    b = heat_cap + g_up + g_dn
    b = b.at[..., -1].add(g_bot)
    d = heat_cap * T_old
    d = d.at[..., 0].add(F_top)
    d = d.at[..., -1].add(g_bot * T_bottom_K)
    T_new = bl_mod.thomas_solve(a, b, c, d)
    q_new = q_layers + bl_mod.canonical_constants.rho_ice * c_i * (
        T_new - T_old
    )
    F_bottom = g_bot * (T_new[..., -1] - T_bottom_K)
    return q_new, T_new, F_bottom


R5_PARENT_COMMIT = "a572f4b2540351bf49328d1533211427f9c9d772"
R6_PARENT_COMMIT = "071dfa8443699a44e68904caaf3d02e17417e757"


def _load_parent_commit_non_si3_modules(
    parent_commit: str = R6_PARENT_COMMIT,
) -> dict[str, types.ModuleType]:
    """Load the ice and tridiagonal modules directly from `parent_commit` via `git show`."""
    module_specs = (
        (
            "legoesm.timestepping.tridiagonal",
            "packages/core/legoesm/timestepping/tridiagonal.py",
        ),
        ("legoesm.ice.bitz_lipscomb", "packages/ice/legoesm/ice/bitz_lipscomb.py"),
        ("legoesm.ice.snow", "packages/ice/legoesm/ice/snow.py"),
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


def test_d5_non_si3_callers_are_bit_identical(monkeypatch, probe_module) -> None:
    """Compare Round 6 parent commit (`071dfa844369`) vs current across non-SI3 callers and the production lane-3 SI3 dispatcher (`sea_ice.py:287-299,497-514`).

    Expected to change on the SI3 lane-3 production dispatcher (`step_sea_ice`
    with `thermo_scheme='si3_bl99'` under `jax.jit` vs Round 6 parent `071dfa844369`):
      - At `kt = 5241` (Statement Family 1): ONLY `new_state.e_ice.data` (moving
        from parent's `pts_old = sr(thickness * enthalpy)` to `0` ULP vs C1D
        oracle `POST_DH.e_i`);
      - At `kt = 3996` (Statement Family 1): ONLY `new_state.h_ice.data` and
        `new_state.e_ice.data` (both moving to `0` ULP vs C1D oracle `POST_DH.h_i`
        and `POST_DH.e_i`);
      - At `kt = 5460` (Statement Family 2): ONLY `new_state.h_snow.data` and
        `new_state.e_snow.data` (both moving from parent's `10`/`12`-ULP error to
        `0` ULP vs C1D oracle `POST_DH.h_s` and `POST_DH.e_s`).
    Expected to stay bit-for-bit (`uint64`) identical between parent (`071dfa844369`)
    and current:
      - All non-SI3 callers (`bitz_lipscomb_conduction_step`, `thomas_solve(..., 'normalised')`,
        `IceColumnModel.step()` slab, and `step_sea_ice` dynamic-v2), and
      - On the SI3 lane-3 production dispatcher: all other `SI3ColumnState` fields
        and all 22 `TileResponse` fields on `kt = 5241`, `kt = 3996`, and `kt = 5460`
        (plus all 9 `SI3ColumnState` fields and all 22 `TileResponse` fields at `kt = 43`).
    """
    from legoesm.core.field import Field
    import legoesm.ice.scm as scm_mod
    import legoesm.ice.sea_ice as sea_ice_mod
    from legoesm.ice.state import SI3ColumnState
    import legoesm.timestepping.tridiagonal as tridiag_mod
    from legoesm.ice import (
        BrineConfig,
        SeaIceConfig,
        SnowConfig,
        init_dynamic_ice_state,
    )

    parent_mods = _load_parent_commit_non_si3_modules(R6_PARENT_COMMIT)
    r5_parent_mods = _load_parent_commit_non_si3_modules(R5_PARENT_COMMIT)
    current_mods = {
        "legoesm.timestepping.tridiagonal": tridiag_mod,
        "legoesm.ice.bitz_lipscomb": bl_mod,
        "legoesm.ice.sea_ice": sea_ice_mod,
        "legoesm.ice.scm": scm_mod,
    }

    q_layers = jnp.asarray([[-2.8e8, -2.7e8, -2.5e8]], dtype=jnp.float64)
    s = jnp.asarray([5.0], dtype=jnp.float64)
    f_top = jnp.asarray([-25.0], dtype=jnp.float64)
    t_bot = jnp.asarray([271.35], dtype=jnp.float64)

    rng = np.random.default_rng(20261024)
    a_t = jnp.asarray(rng.uniform(-0.3, -0.05, (2, 5)), dtype=jnp.float64).at[
        ..., 0
    ].set(0.0)
    b_t = jnp.asarray(rng.uniform(2.0, 3.0, (2, 5)), dtype=jnp.float64)
    c_t = jnp.asarray(rng.uniform(-0.3, -0.05, (2, 5)), dtype=jnp.float64).at[
        ..., -1
    ].set(0.0)
    d_t = jnp.asarray(rng.uniform(250.0, 272.0, (2, 5)), dtype=jnp.float64)

    def _run_non_si3_paths(mods: dict[str, types.ModuleType]) -> tuple[np.ndarray, ...]:
        m_tridiag = mods["legoesm.timestepping.tridiagonal"]
        m_bl = mods["legoesm.ice.bitz_lipscomb"]
        m_sea_ice = mods["legoesm.ice.sea_ice"]
        m_scm = mods["legoesm.ice.scm"]

        out_bits: list[np.ndarray] = []

        # (a) Non-SI3 `bitz_lipscomb_conduction_step` across thick, kt=5841, and thin ice
        for h_val in (1.8, 0.5877349091111249, 1.0e-3):
            h = jnp.asarray([h_val], dtype=jnp.float64)
            q_new, t_new, fb = m_bl.bitz_lipscomb_conduction_step(
                q_layers, h, s, f_top, t_bot, 3600.0
            )
            for arr in (q_new, t_new, fb):
                assert arr.dtype == jnp.float64
                assert bool(jnp.all(jnp.isfinite(arr)))
                out_bits.append(np.asarray(arr, dtype=np.float64).view(np.uint64))

        # (b) Default `thomas_solve(..., "normalised")`
        sol_norm = m_tridiag.thomas_solve(a_t, b_t, c_t, d_t)
        assert sol_norm.dtype == jnp.float64
        out_bits.append(np.asarray(sol_norm, dtype=np.float64).view(np.uint64))

        # (c) Non-SI3 `IceColumnModel` slab step (bind `legoesm.ice.step_sea_ice`
        #     so `scm.step()`'s `from legoesm.ice import step_sea_ice` calls `m_sea_ice.step_sea_ice`)
        import legoesm.ice as ice_pkg

        orig_pkg_step = ice_pkg.step_sea_ice
        try:
            ice_pkg.step_sea_ice = m_sea_ice.step_sea_ice
            scm = m_scm.IceColumnModel.create(
                ncol=2, dt=3600.0, forcing=m_scm.cold_polar_forcing(ncol=2)
            )
            resp_slab = scm.step()
        finally:
            ice_pkg.step_sea_ice = orig_pkg_step
        out_bits.extend(
            (
                np.asarray(scm.state.h_ice.data, dtype=np.float64).view(np.uint64),
                np.asarray(scm.state.T_ice.data, dtype=np.float64).view(np.uint64),
                np.asarray(scm.state.concentration.data, dtype=np.float64).view(
                    np.uint64
                ),
                np.asarray(resp_slab.T_sfc, dtype=np.float64).view(np.uint64),
                np.asarray(resp_slab.lhflx, dtype=np.float64).view(np.uint64),
            )
        )

        # (d) Non-SI3 `step_sea_ice` dynamic-v2 step
        cfg_v2 = SeaIceConfig(
            snow=SnowConfig(enabled=True),
            brine=BrineConfig(enabled=True),
        )
        dyn_state = init_dynamic_ice_state((2,))
        forcing_v2 = m_scm.cold_polar_forcing(ncol=2)
        sst = jnp.full((2,), 271.35, dtype=jnp.float64)
        zero_vel = jnp.zeros((2,), dtype=jnp.float64)
        new_dyn, resp_v2 = m_sea_ice.step_sea_ice(
            dyn_state, forcing_v2, sst, zero_vel, zero_vel, cfg_v2, 1.0, 3600.0
        )
        out_bits.extend(
            (
                np.asarray(new_dyn.h_ice.data, dtype=np.float64).view(np.uint64),
                np.asarray(new_dyn.h_snow.data, dtype=np.float64).view(np.uint64),
                np.asarray(new_dyn.T_ice.data, dtype=np.float64).view(np.uint64),
                np.asarray(new_dyn.S_ice.data, dtype=np.float64).view(np.uint64),
                np.asarray(resp_v2.T_sfc, dtype=np.float64).view(np.uint64),
            )
        )
        return tuple(out_bits)

    parent_bits = _run_non_si3_paths(parent_mods)
    current_bits = _run_non_si3_paths(current_mods)
    for arr_cur, arr_par in zip(current_bits, parent_bits, strict=True):
        np.testing.assert_array_equal(arr_cur, arr_par)

    def _assert_differs_from_parent(cand_bits: tuple[np.ndarray, ...]) -> None:
        with pytest.raises(AssertionError):
            for arr_c, arr_p in zip(cand_bits, parent_bits, strict=True):
                np.testing.assert_array_equal(arr_c, arr_p)

    # Planted control 1 (in `bitz_lipscomb_conduction_step` code path):
    # planting the SI3 reciprocal `dz = h * (1/n)` inside `bitz_lipscomb_conduction_step`
    # changes the executed output at `h = 0.5877349091111249 m` and fails against parent.
    with monkeypatch.context() as mp:
        mp.setattr(
            bl_mod,
            "bitz_lipscomb_conduction_step",
            lambda q, h, S, ft, tb, dt: _parent_commit_bitz_lipscomb_conduction_step(
                q, h, S, ft, tb, dt, plant_reciprocal_dz=True
            ),
        )
        _assert_differs_from_parent(_run_non_si3_paths(current_mods))

    # Planted control 2 (in `thomas_solve` code path):
    # routing the default `"normalised"` solver to `"nemo_unnormalised"` inside
    # `legoesm.timestepping.tridiagonal` changes the executed output and fails against parent.
    with monkeypatch.context() as mp:
        mp.setattr(
            tridiag_mod,
            "thomas_solve",
            lambda a, b, c_band, d, forward_elimination="normalised": _thomas_solve_impl(
                a, b, c_band, d, "nemo_unnormalised"
            ),
        )
        _assert_differs_from_parent(_run_non_si3_paths(current_mods))

    # Planted control 3 (in `step_sea_ice` slab code path):
    # perturbing `simple_bulk_fluxes` in `legoesm.ice.sea_ice` by 1 ULP on `lhflx`
    # changes the executed `IceColumnModel.step()` output and fails against parent.
    orig_bulk = sea_ice_mod.simple_bulk_fluxes

    def _planted_bulk(*a, **kw):
        tau_x, tau_y, shflx, lhflx = orig_bulk(*a, **kw)
        return tau_x, tau_y, shflx, jnp.nextafter(lhflx, jnp.inf)

    with monkeypatch.context() as mp:
        mp.setattr(sea_ice_mod, "simple_bulk_fluxes", _planted_bulk)
        _assert_differs_from_parent(_run_non_si3_paths(current_mods))

    # Planted control 4 (in `step_sea_ice` dynamic-v2 code path):
    # perturbing `update_salinity_and_salt_flux` in `legoesm.ice.sea_ice` by 1 ULP
    # on `S_new` changes the executed `step_sea_ice` dynamic-v2 output and fails against parent.
    orig_brine = sea_ice_mod.update_salinity_and_salt_flux

    def _planted_brine(*a, **kw):
        res = orig_brine(*a, **kw)
        return res._replace(
            **{res._fields[0]: jnp.nextafter(res[0], jnp.inf)}
        )

    with monkeypatch.context() as mp:
        mp.setattr(
            sea_ice_mod, "update_salinity_and_salt_flux", _planted_brine
        )
        _assert_differs_from_parent(_run_non_si3_paths(current_mods))

    # (e) Production lane-3 SI3 dispatcher path (`sea_ice.py:287-299,497-514`):
    #     Compare Round 6 parent (`071dfa844369`) vs current under `jax.jit` on
    #     C1D steps `kt = 43`, `kt = 5241` (Statement Family 1 `e_ice`),
    #     `kt = 3996` (Statement Family 1 `h_ice`, `e_ice`), and `kt = 5460`
    #     (Statement Family 2 `h_snow`, `e_snow`), plus `kt = 43` vs Round 5 parent.
    r5_parent_bits = _run_non_si3_paths(r5_parent_mods)
    for arr_cur, arr_r5p in zip(current_bits, r5_parent_bits, strict=True):
        np.testing.assert_array_equal(arr_cur, arr_r5p)

    card = build_c1d_omip_l3_card(oracle_root=ORACLE_V2_ROOT)
    si3_steps = _load_step_operands(probe_module, (43, 3996, 5241, 5460))
    dims = ("column",)
    sst_1d = jnp.full((1,), 271.35, dtype=jnp.float64)
    zvel_1d = jnp.zeros((1,), dtype=jnp.float64)

    def _build_si3_state_and_forcing(
        kt_val: int, m_bl_mod: types.ModuleType
    ) -> tuple[SI3ColumnState, object, dict]:
        fr_kt, ex_kt = si3_steps[kt_val]
        ent_kt = probe_module.year_gate._exact_1d_entry(
            fr_kt, sss_m=ex_kt["sss"], _nemo_entry_conversion=True
        )
        st_kt = SI3ColumnState(
            concentration=Field(
                jnp.asarray(ent_kt.concentration, dtype=jnp.float64),
                name="ice_concentration",
                dims=dims,
                units="1",
            ),
            h_ice=Field(
                jnp.asarray(ent_kt.h_ice, dtype=jnp.float64),
                name="h_ice",
                dims=dims,
                units="m",
            ),
            h_snow=Field(
                jnp.asarray(ent_kt.h_snow, dtype=jnp.float64),
                name="h_snow",
                dims=dims,
                units="m",
            ),
            T_surface=Field(
                jnp.asarray(ent_kt.T_surface, dtype=jnp.float64),
                name="ice_surface_temperature",
                dims=dims,
                units="K",
            ),
            e_ice=Field(
                jnp.asarray(ent_kt.e_ice, dtype=jnp.float64),
                name="ice_layer_enthalpy",
                dims=dims + ("ice_layer",),
                units="J/m3",
            ),
            e_snow=Field(
                jnp.asarray(ent_kt.e_snow, dtype=jnp.float64),
                name="snow_layer_enthalpy",
                dims=dims + ("snow_layer",),
                units="J/m3",
            ),
            S_bulk=Field(
                jnp.asarray(ent_kt.S_bulk, dtype=jnp.float64),
                name="bulk_ice_salinity",
                dims=dims,
                units="PSU",
            ),
            S_layers=Field(
                jnp.asarray(ent_kt.S_layers, dtype=jnp.float64),
                name="ice_layer_salinity",
                dims=dims + ("ice_layer",),
                units="PSU",
            ),
            age_volume=Field(
                jnp.asarray(ent_kt.age_volume, dtype=jnp.float64),
                name="ice_age_volume",
                dims=dims,
                units="s",
            ),
        )
        forc_kt = m_bl_mod.SI3SurfaceForcing(
            *(
                jnp.asarray(ex_kt[n], dtype=jnp.float64)
                for n in probe_module.phase2.ZDF_INPUT_NAMES
            )
        )
        return st_kt, forc_kt, fr_kt

    def _run_si3_dispatcher(
        mods: dict[str, types.ModuleType], kt_val: int
    ) -> tuple[SI3ColumnState, object, dict]:
        m_bl_mod = mods["legoesm.ice.bitz_lipscomb"]
        m_sea_ice_mod = mods["legoesm.ice.sea_ice"]
        st_in, forc_in, fr_kt = _build_si3_state_and_forcing(kt_val, m_bl_mod)
        fn = jax.jit(
            lambda s, f: m_sea_ice_mod.step_sea_ice(
                s,
                f,
                sst_1d,
                zvel_1d,
                zvel_1d,
                card.config,
                1.0,
                card.dt_seconds,
            )
        )
        st_out, resp_out = fn(st_in, forc_in)
        return st_out, resp_out, fr_kt

    def _verify_si3_dispatcher_vs_parent(
        cand_mods: dict[str, types.ModuleType],
    ) -> None:
        # 1. Against Round 6 parent (`071dfa844369`) at kt = 43: all 9 state
        #    fields and all 22 response fields are bit-for-bit identical, while
        #    against Round 5 parent (`a572f4b254`) ONLY `e_ice` changes to 0 ULP.
        par_st43, par_rp43, fr43 = _run_si3_dispatcher(parent_mods, 43)
        r5p_st43, _, _ = _run_si3_dispatcher(r5_parent_mods, 43)
        cur_st43, cur_rp43, _ = _run_si3_dispatcher(cand_mods, 43)
        oracle_ei43 = (
            np.asarray(fr43[2]["e_i"], dtype=np.float64).ravel().view(np.uint64)
        )
        r5p_ei43 = (
            np.asarray(r5p_st43.e_ice.data, dtype=np.float64)
            .ravel()
            .view(np.uint64)
        )
        cur_ei43 = (
            np.asarray(cur_st43.e_ice.data, dtype=np.float64)
            .ravel()
            .view(np.uint64)
        )
        assert not np.array_equal(cur_ei43, r5p_ei43)
        np.testing.assert_array_equal(cur_ei43, oracle_ei43)
        for fname in SI3ColumnState._fields:
            np.testing.assert_array_equal(
                np.asarray(getattr(cur_st43, fname).data, dtype=np.float64).view(
                    np.uint64
                ),
                np.asarray(getattr(par_st43, fname).data, dtype=np.float64).view(
                    np.uint64
                ),
            )
        for rname in cur_rp43._fields:
            np.testing.assert_array_equal(
                np.asarray(getattr(cur_rp43, rname), dtype=np.float64).view(
                    np.uint64
                ),
                np.asarray(getattr(par_rp43, rname), dtype=np.float64).view(
                    np.uint64
                ),
            )

        # 2. Against Round 6 parent (`071dfa844369`) on touched steps:
        #    - kt = 5241: ONLY `e_ice` changes (to 0 ULP vs oracle `POST_DH.e_i`);
        #    - kt = 3996: ONLY `h_ice` and `e_ice` change (to 0 ULP vs oracle `POST_DH.h_i`, `POST_DH.e_i`);
        #    - kt = 5460: ONLY `h_snow` and `e_snow` change (to 0 ULP vs oracle `POST_DH.h_s`, `POST_DH.e_s`).
        expected_changes = (
            (5241, ("e_ice",), {"e_ice": "e_i"}),
            (3996, ("h_ice", "e_ice"), {"h_ice": "h_i", "e_ice": "e_i"}),
            (5460, ("h_snow", "e_snow"), {"h_snow": "h_s", "e_snow": "e_s"}),
        )
        for kt_val, changed_fields, oracle_map in expected_changes:
            par_st, par_rp, fr_kt = _run_si3_dispatcher(parent_mods, kt_val)
            cur_st, cur_rp, _ = _run_si3_dispatcher(cand_mods, kt_val)
            for fname in changed_fields:
                par_bits = (
                    np.asarray(getattr(par_st, fname).data, dtype=np.float64)
                    .ravel()
                    .view(np.uint64)
                )
                cur_bits = (
                    np.asarray(getattr(cur_st, fname).data, dtype=np.float64)
                    .ravel()
                    .view(np.uint64)
                )
                ora_bits = (
                    np.asarray(fr_kt[2][oracle_map[fname]], dtype=np.float64)
                    .ravel()
                    .view(np.uint64)
                )
                assert not np.array_equal(cur_bits, par_bits)
                np.testing.assert_array_equal(cur_bits, ora_bits)
            for fname in SI3ColumnState._fields:
                if fname in changed_fields:
                    continue
                np.testing.assert_array_equal(
                    np.asarray(
                        getattr(cur_st, fname).data, dtype=np.float64
                    ).view(np.uint64),
                    np.asarray(
                        getattr(par_st, fname).data, dtype=np.float64
                    ).view(np.uint64),
                )
            for rname in cur_rp._fields:
                np.testing.assert_array_equal(
                    np.asarray(getattr(cur_rp, rname), dtype=np.float64).view(
                        np.uint64
                    ),
                    np.asarray(getattr(par_rp, rname), dtype=np.float64).view(
                        np.uint64
                    ),
                )

    _verify_si3_dispatcher_vs_parent(current_mods)

    # Planted control 5a (in `step_sea_ice` -> `_si3_step_with_trace` dispatcher
    # at `sea_ice.py:287-299,497-514`): perturbing an unchanged output (`concentration`)
    # by 1 ULP inside `_si3_step_with_trace` must make the dispatcher control fail.
    orig_si3_step = sea_ice_mod._si3_step_with_trace

    def _planted_si3_step_perturb_a(*a, **kw):
        new_st, resp, tr = orig_si3_step(*a, **kw)
        perturbed_a = new_st.concentration.replace(
            data=jnp.nextafter(new_st.concentration.data, jnp.inf)
        )
        return new_st._replace(concentration=perturbed_a), resp, tr

    with monkeypatch.context() as mp:
        mp.setattr(
            sea_ice_mod, "_si3_step_with_trace", _planted_si3_step_perturb_a
        )
        with pytest.raises(AssertionError):
            _verify_si3_dispatcher_vs_parent(current_mods)

    # Planted control 5b (in `_si3_step_with_trace` -> `si3_column_step_arrays`
    # call at `sea_ice.py:514`): reverting `_nemo_dh_ice_content_accumulation=False`
    # inside the dispatcher must make the Statement Family 1 check (`kt = 5241, 3996`) fail.
    orig_si3_arrays = sea_ice_mod.si3_column_step_arrays

    def _planted_si3_arrays_no_r6_step1(arrays, forcing, dt, ice_constants):
        return orig_si3_arrays(
            arrays,
            forcing,
            dt,
            ice_constants,
            _nemo_dh_ice_content_accumulation=False,
        )

    with monkeypatch.context() as mp:
        mp.setattr(
            sea_ice_mod,
            "si3_column_step_arrays",
            _planted_si3_arrays_no_r6_step1,
        )
        with pytest.raises(AssertionError):
            _verify_si3_dispatcher_vs_parent(current_mods)

    # Planted control 5c (in `_si3_step_with_trace` -> `si3_column_step_arrays`
    # call at `sea_ice.py:514`): reverting `_nemo_dh_snow_update=False` inside
    # the dispatcher must make the Statement Family 2 check (`kt = 5460`) fail.
    def _planted_si3_arrays_no_r6_step2(arrays, forcing, dt, ice_constants):
        return orig_si3_arrays(
            arrays,
            forcing,
            dt,
            ice_constants,
            _nemo_dh_snow_update=False,
        )

    with monkeypatch.context() as mp:
        mp.setattr(
            sea_ice_mod,
            "si3_column_step_arrays",
            _planted_si3_arrays_no_r6_step2,
        )
        with pytest.raises(AssertionError):
            _verify_si3_dispatcher_vs_parent(current_mods)


def test_d5_ftz_daz_classification_distinguishes_ftz_only_from_daz(
    probe_module,
) -> None:
    """Verify DAZ is classified ONLY from the Stage B `h_s > 0` predicate, not `x*1.0`/`x+0.0`."""
    # Synthetic FTZ-only platform: `x*1.0`, `x+0.0`, and `a*b` flush subnormal
    # outputs to 0, but `h_s > 0.0` (normal boolean output) is True because
    # subnormal inputs are NOT flushed on input (DAZ=0).
    ftz_only_stage_a = {
        "arithmetic_preserves_subnormal": False,
        "ftz_flushes_subnormal_output": True,
        "jit_mul_one_out_bits": "0x0000000000000000",
        "jit_add_zero_out_bits": "0x0000000000000000",
        "jit_produce_subnormal_out_bits": "0x0000000000000000",
    }
    ftz_only_stage_b = {
        "inputs_are_positive_subnormal": True,
        "predicate_all_true": True,
        "predicate_all_false": False,
    }
    classified_ftz_only = probe_module.classify_ftz_daz(
        ftz_only_stage_a, ftz_only_stage_b
    )
    assert classified_ftz_only["ftz_flushes_subnormal_output"] is True
    assert classified_ftz_only["daz_flushes_subnormal_input"] is False
    assert classified_ftz_only["ftz_daz_mode"] == "FTZ_ONLY_FLUSH_TO_ZERO"

    # Planted control: the old invalid DAZ formula (from `x*1.0` and `x+0.0`)
    # falsely reports `daz_flushes_subnormal_input=True` under FTZ-only and
    # MUST fail the FTZ-only check.
    planted_invalid = probe_module.classify_ftz_daz(
        ftz_only_stage_a,
        ftz_only_stage_b,
        plant_invalid_daz_from_mul_add=True,
    )
    with pytest.raises(AssertionError):
        assert planted_invalid["daz_flushes_subnormal_input"] is False

    # When `h_s > 0.0` flushes positive subnormal inputs to False, DAZ is True:
    ftz_daz_stage_b = {
        "inputs_are_positive_subnormal": True,
        "predicate_all_true": False,
        "predicate_all_false": True,
    }
    classified_ftz_daz = probe_module.classify_ftz_daz(
        ftz_only_stage_a, ftz_daz_stage_b
    )
    assert classified_ftz_daz["ftz_flushes_subnormal_output"] is True
    assert classified_ftz_daz["daz_flushes_subnormal_input"] is True
    assert classified_ftz_daz["ftz_daz_mode"] == "FTZ_DAZ_FLUSH_TO_ZERO"


def test_d5_platform_probe_and_planted_controls(probe_module) -> None:
    """Direct unit test for nemo_si3thd_d5_platform_probe.py and its planted controls."""
    report = probe_module.run_probe(
        include_year_rescore=False,
        year_max_steps=30,
        allow_dirty=True,
    )
    assert report["format"] == "nemo-si3thd-d5-platform-probe-v2"
    stage_a = report["stage_a_subnormal_probe"]
    stage_b = report["stage_b_predicate_probe"]
    stage_c = report["stage_c_zdf_probe"]
    step1 = report["step1_reviewer_reproduction"]

    assert (
        step1["zdf_entry_snow_temperature"]["step2_plus_step3_nonzero_ulp_steps"]
        == 0
    )
    assert (
        step1["normal_steps_post_zdf_26202_rows"][
            "nemo_exact_inputs_source_rounded"
        ]["zero_ulp_rows"]
        == 30 * 3
    )

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
        assert stage_b["daz_flushes_subnormal_input"] is True
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
        assert stage_a["daz_flushes_subnormal_input"] is False
        assert stage_b["daz_flushes_subnormal_input"] is False
        assert stage_b["predicate_all_true"] is True
        assert stage_c["before_statement_fix"]["over_bar_post_zdf_count"] == 57
        assert stage_c["after_statement_fix"]["over_bar_post_zdf_count"] == 0
        assert stage_c["after_statement_fix"]["nonzero_ulp_post_zdf_count"] == 0
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
        include_step1_reproduction=False,
        allow_dirty=True,
    )
    assert planted_zero["stage_a_subnormal_probe"]["input_x_bits"] == "0x0000000000000000"
    assert planted_zero["stage_a_subnormal_probe"]["ftz_flushes_subnormal_output"] is False
    assert planted_zero["platform_disposition"] == "D5_PLATFORM_FLOOR"

    planted_flush = probe_module.run_probe(
        plant="flush_recorded_operand",
        include_year_rescore=False,
        include_step1_reproduction=False,
        allow_dirty=True,
    )
    assert planted_flush["stage_b_predicate_probe"]["predicate_true_count"] == 0
    assert planted_flush["stage_b_predicate_probe"]["inputs_are_positive_subnormal"] is False
    assert planted_flush["stage_b_predicate_probe"]["daz_flushes_subnormal_input"] is False
    assert planted_flush["platform_disposition"] == "D5_PLATFORM_FLOOR"

    planted_revert = probe_module.run_probe(
        plant="revert_statement_fix",
        include_year_rescore=False,
        include_step1_reproduction=False,
        allow_dirty=True,
    )
    assert (
        planted_revert["stage_c_zdf_probe"]["after_statement_fix"][
            "detailed_post_zdf_steps"
        ]["kt5841"]["e_s"]["max_ulp"]
        in (710, 716)
    )

    planted_snow_temp = probe_module.run_probe(
        plant="revert_snow_temp_statement",
        include_year_rescore=False,
        include_step1_reproduction=True,
        year_max_steps=30,
        allow_dirty=True,
    )
    assert (
        planted_snow_temp["step1_reviewer_reproduction"][
            "zdf_entry_snow_temperature"
        ]["step2_plus_step3_nonzero_ulp_steps"]
        > 0
    )

    # Also test Stage D year_rescore on the first 30 steps
    rescore_30 = probe_module.run_year_rescore(max_steps=30)
    assert rescore_30["execution_mode"] == "jax.jit"
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
    assert (
        rescore_30["arms"]["step2_plus_step3_plus_step4"][
            "normal_post_zdf_3field"
        ]["zero_ulp_rows"]
        == 30 * 3
    )
    assert (
        rescore_30["arms"]["step2_plus_step3_plus_step4"][
            "normal_post_zdf_3field"
        ]["over_bar_rows"]
        == 0
    )
    s3_s4 = rescore_30["step3_vs_step4_bitwise_comparison"]
    assert s3_s4["total_field_rows"] == 30 * 71
    assert (
        s3_s4["changed_field_rows"] + s3_s4["unchanged_field_rows"] == 30 * 71
    )
    assert len(s3_s4["step3_full_year_output_sha256"]) == 64
    assert len(s3_s4["step4_full_year_output_sha256"]) == 64
    if jax.default_backend() == "gpu":
        assert s3_s4["changed_field_rows"] == 0
        assert s3_s4["max_ulp_diff"] == 0
        assert s3_s4["identical_full_year_output_sha256"] is True


def test_r3_step2_harness_entry_conversion_is_bit_exact_and_fails_when_reverted(
    probe_module,
) -> None:
    """Pin Step 2 harness entry conversion (`icevar.F90:368,374,390,412,488-506`) bit-for-bit."""
    card = build_c1d_omip_l3_card(oracle_root=ORACLE_V2_ROOT)
    c = card.config.ice_constants
    steps = _load_step_operands(probe_module, (1, 2, 15))

    # At kt = 2, `(e_s / v_s) * 3.0` vs `(e_s * 3.0) / v_s`, `(e_i / v_i) * 3.0`
    # vs `(e_i * 3.0) / v_i`, and `_nemo_salprof(s_i)` vs `szv_i * 3.0 / v_i`
    # differ on inexact recorded C1D operands.
    frames_2, exact_2 = steps[2]
    entry_nemo = probe_module.year_gate._exact_1d_entry(
        frames_2, sss_m=exact_2["sss"], _nemo_entry_conversion=True
    )
    entry_rev = probe_module.year_gate._exact_1d_entry(
        frames_2, sss_m=exact_2["sss"], _nemo_entry_conversion=False
    )

    oracle_szi_2 = np.asarray(frames_2[1]["sz_i"], dtype=np.float64).ravel()
    assert (
        probe_module._ulp_diff(
            np.asarray(entry_nemo.S_layers, dtype=np.float64).ravel(),
            oracle_szi_2,
        )
        == 0
    )
    assert (
        probe_module._ulp_diff(
            np.asarray(entry_rev.S_layers, dtype=np.float64).ravel(),
            oracle_szi_2,
        )
        == 1
    )
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(
            np.asarray(entry_rev.S_layers, dtype=np.float64).view(np.uint64),
            oracle_szi_2.view(np.uint64),
        )

    # Verify entry `e_s` at kt = 15 (`(e_s / v_s) * 3.0` vs `(e_s * 3.0) / v_s`)
    frames_15, exact_15 = steps[15]
    entry_nemo_15 = probe_module.year_gate._exact_1d_entry(
        frames_15, sss_m=exact_15["sss"], _nemo_entry_conversion=True
    )
    entry_rev_15 = probe_module.year_gate._exact_1d_entry(
        frames_15, sss_m=exact_15["sss"], _nemo_entry_conversion=False
    )
    assert (
        probe_module._ulp_diff(
            np.asarray(entry_nemo_15.e_snow, dtype=np.float64).ravel(),
            np.asarray(entry_rev_15.e_snow, dtype=np.float64).ravel(),
        )
        == 1
    )
    ts_nemo_15 = np.asarray(
        bl_mod.snow_temperature_from_enthalpy(
            jnp.asarray(entry_nemo_15.e_snow, dtype=jnp.float64),
            c,
            _nemo_glo2eqv_snow_temp=True,
            _nemo_source_round=True,
        ),
        dtype=np.float64,
    ).ravel()
    ts_rev_15 = np.asarray(
        bl_mod.snow_temperature_from_enthalpy(
            jnp.asarray(entry_rev_15.e_snow, dtype=np.float64),
            c,
            _nemo_glo2eqv_snow_temp=True,
            _nemo_source_round=True,
        ),
        dtype=np.float64,
    ).ravel()
    oracle_ts_15 = np.asarray(exact_15["t_s"], dtype=np.float64).ravel()
    assert probe_module._ulp_diff(ts_nemo_15, oracle_ts_15) == 0
    assert probe_module._ulp_diff(ts_rev_15, oracle_ts_15) == 1
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(
            ts_rev_15.view(np.uint64), oracle_ts_15.view(np.uint64)
        )

    # Verify both SSS-dependent guards in `_nemo_salprof` (`icevar.F90:495-496`):
    # (1) Low-SSS guard: `s_i >= 0.5 * rn_sinew * sss_m` forces `zalpha = 0`
    #     (constant profile `[s_i, s_i, s_i]`), whereas omitting the guard gives
    #     a sloped profile (`zalpha = 1` for `s_i = 2.0 <= 3.5`).
    s_low_sss = np.asarray([2.0], dtype=np.float64)
    sss_low = np.asarray([4.0], dtype=np.float64)  # 0.5 * 0.75 * 4.0 = 1.5 <= 2.0
    got_low_sss = probe_module.phase2._nemo_salprof(s_low_sss, sss_m=sss_low)
    ref_low_sss = np.asarray(
        bl_mod.option2_salinity_profile(
            jnp.asarray(s_low_sss), jnp.asarray(sss_low)
        ),
        dtype=np.float64,
    )
    np.testing.assert_array_equal(
        got_low_sss.view(np.uint64), ref_low_sss.view(np.uint64)
    )
    np.testing.assert_array_equal(
        got_low_sss, np.asarray([[2.0, 2.0, 2.0]], dtype=np.float64)
    )
    # Without the low-SSS guard (using default sss_m=34.0), `s_i=2.0` has `zalpha=1.0`:
    unclamped_low_sss = probe_module.phase2._nemo_salprof(
        s_low_sss, sss_m=34.0
    )
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(
            unclamped_low_sss.view(np.uint64), ref_low_sss.view(np.uint64)
        )

    # (2) Minimum-salinity SSS-dependent guard:
    #     `s_i <= nlay_i * MIN(rn_simin, rn_sinew * sss_m)` forces `zalpha = 0`.
    #     When `rn_sinew * sss_m = 0.75 * 0.08 = 0.06 < rn_simin = 0.1`, the
    #     threshold is `3 * 0.06 = 0.18` (not hardcoded `3 * 0.1 = 0.3`), so
    #     `s_i = 0.25` has `s_i > 0.18` (and `0.5 * 0.06 = 0.03 <= 0.25` when
    #     `rn_sinew` is tested with high `rn_simin`). Test `rn_simin=0.5`,
    #     `sss_m=20.0`, `s_i=1.2`: `3 * min(0.5, 15.0) = 1.5 >= 1.2` -> `zalpha = 0`.
    got_min_guard = probe_module.phase2._nemo_salprof(
        np.asarray([1.2], dtype=np.float64),
        sss_m=20.0,
        rn_simin=0.5,
    )
    np.testing.assert_array_equal(
        got_min_guard, np.asarray([[1.2, 1.2, 1.2]], dtype=np.float64)
    )


def test_r3_step3_snow_temperature_from_enthalpy_is_bit_exact_and_fails_when_reverted(
    probe_module,
) -> None:
    """Pin Step 3 `snow_temperature_from_enthalpy` (`icevar.F90:412-413`) bit-for-bit."""
    card = build_c1d_omip_l3_card(oracle_root=ORACLE_V2_ROOT)
    c = card.config.ice_constants
    steps = _load_step_operands(probe_module, (2, 15, 1334))

    for kt in (2, 15, 1334):
        frames, exact_in = steps[kt]
        entry = probe_module.year_gate._exact_1d_entry(
            frames, sss_m=exact_in["sss"], _nemo_entry_conversion=True
        )
        e_s_jax = jnp.asarray(entry.e_snow, dtype=jnp.float64)
        oracle_ts = np.asarray(exact_in["t_s"], dtype=np.float64).ravel()

        got_fixed = np.asarray(
            jax.jit(
                lambda e: bl_mod.snow_temperature_from_enthalpy(
                    e,
                    c,
                    _nemo_order=True,
                    _nemo_glo2eqv_snow_temp=True,
                    _nemo_source_round=True,
                )
            )(e_s_jax),
            dtype=np.float64,
        ).ravel()
        got_reverted = np.asarray(
            jax.jit(
                lambda e: bl_mod.snow_temperature_from_enthalpy(
                    e,
                    c,
                    _nemo_order=True,
                    _nemo_glo2eqv_snow_temp=False,
                    _nemo_source_round=True,
                )
            )(e_s_jax),
            dtype=np.float64,
        ).ravel()

        assert probe_module._ulp_diff(got_fixed, oracle_ts) == 0
        np.testing.assert_array_equal(
            got_fixed.view(np.uint64), oracle_ts.view(np.uint64)
        )
        assert probe_module._ulp_diff(got_reverted, oracle_ts) >= 1
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(
                got_reverted.view(np.uint64), oracle_ts.view(np.uint64)
            )

    # At kt = 1334, reverting Step 3 alone pushes POST_ZDF.e_s over the 1e-15 bar!
    frames_1334, exact_in_1334 = steps[1334]
    entry_1334 = probe_module.year_gate._exact_1d_entry(
        frames_1334, sss_m=exact_in_1334["sss"], _nemo_entry_conversion=True
    )
    jax_entry_1334 = jax.tree.map(
        lambda v: jnp.asarray(v, dtype=jnp.float64), entry_1334
    )
    forcing_1334 = probe_module.year_gate._forcing(exact_in_1334, jnp.float64)
    zdf_fixed_1334 = jax.jit(
        lambda ent, f: _si3_zdf_bl99_step(
            ent.e_ice,
            ent.e_snow,
            ent.S_layers,
            ent.h_ice,
            ent.h_snow,
            ent.T_surface,
            f,
            card.dt_seconds,
            c,
            _nemo_glo2eqv_snow_temp=True,
            _nemo_source_round=True,
        )
    )(jax_entry_1334, forcing_1334)
    zdf_rev_1334 = jax.jit(
        lambda ent, f: _si3_zdf_bl99_step(
            ent.e_ice,
            ent.e_snow,
            ent.S_layers,
            ent.h_ice,
            ent.h_snow,
            ent.T_surface,
            f,
            card.dt_seconds,
            c,
            _nemo_glo2eqv_snow_temp=False,
            _nemo_source_round=True,
        )
    )(jax_entry_1334, forcing_1334)
    oracle_es_1334 = np.asarray(frames_1334[1]["e_s"], dtype=np.float64).ravel()
    got_es_fixed_1334 = np.asarray(zdf_fixed_1334.e_snow, dtype=np.float64).ravel()
    got_es_rev_1334 = np.asarray(zdf_rev_1334.e_snow, dtype=np.float64).ravel()
    assert probe_module._ulp_diff(got_es_fixed_1334, oracle_es_1334) == 0
    _, norm_rev_1334 = probe_module.year_gate._normalised_error(
        oracle_es_1334, got_es_rev_1334
    )
    assert norm_rev_1334 > probe_module.phase2.BAR


def test_r3_step4_cpu_fusion_prevention_is_bit_exact_and_fails_when_reverted(
    probe_module,
) -> None:
    """Pin Step 4 `nemo_source_round` fusion prevention (DECISION 158) across POST_ZDF and POST_DH..EXIT bit-for-bit."""
    card = build_c1d_omip_l3_card(oracle_root=ORACLE_V2_ROOT)
    c = card.config.ice_constants
    steps = _load_step_operands(
        probe_module,
        (1, 2, 3, 2899, 2906, 3961, 3963, 4237, 4242, 4738, 5841),
    )

    for kt in (1, 3, 2906, 4242, 5841):
        frames, exact_in = steps[kt]
        entry = probe_module.year_gate._exact_1d_entry(
            frames, sss_m=exact_in["sss"], _nemo_entry_conversion=True
        )
        jax_entry = jax.tree.map(
            lambda v: jnp.asarray(v, dtype=jnp.float64), entry
        )
        forcing = probe_module.year_gate._forcing(exact_in, jnp.float64)
        zdf_fixed = jax.jit(
            lambda ent, f: _si3_zdf_bl99_step(
                ent.e_ice,
                ent.e_snow,
                ent.S_layers,
                ent.h_ice,
                ent.h_snow,
                ent.T_surface,
                f,
                card.dt_seconds,
                c,
                _nemo_glo2eqv_snow_temp=True,
                _nemo_source_round=True,
            )
        )(jax_entry, forcing)
        oracle_f1 = frames[1]
        for fname, cand_val in (
            ("t_su", zdf_fixed.T_surface),
            ("e_i", zdf_fixed.e_ice),
            ("e_s", zdf_fixed.e_snow),
        ):
            wanted = np.asarray(oracle_f1[fname], dtype=np.float64).ravel()
            got = np.asarray(cand_val, dtype=np.float64).ravel()
            assert probe_module._ulp_diff(got, wanted) == 0
            np.testing.assert_array_equal(
                got.view(np.uint64), wanted.view(np.uint64)
            )

    # On CPU default JIT (where FMA is enabled unless prevented by nemo_source_round),
    # reverting `_nemo_source_round=False` at kt = 2906 FAILS bit-exactness
    # and exceeds the 1e-15 bar on POST_ZDF.e_s, and reverting `_nemo_source_round=False`
    # in POST_DH..EXIT changes bits at kt = 3963 (option2_salinity_profile),
    # kt = 2899 (_nemo_ice_enthalpy_remap), kt = 3961 (_nemo_snow_enthalpy_remap),
    # kt = 4237 (_dh_step snow melt), and kt = 4738 (_dh_step ice melt).
    if jax.default_backend() == "cpu" and "--xla_cpu_max_isa=AVX" not in os.environ.get("XLA_FLAGS", ""):
        frames_2906, exact_in_2906 = steps[2906]
        entry_2906 = probe_module.year_gate._exact_1d_entry(
            frames_2906, sss_m=exact_in_2906["sss"], _nemo_entry_conversion=True
        )
        jax_entry_2906 = jax.tree.map(
            lambda v: jnp.asarray(v, dtype=jnp.float64), entry_2906
        )
        forcing_2906 = probe_module.year_gate._forcing(exact_in_2906, jnp.float64)
        zdf_unrounded_2906 = jax.jit(
            lambda ent, f: _si3_zdf_bl99_step(
                ent.e_ice,
                ent.e_snow,
                ent.S_layers,
                ent.h_ice,
                ent.h_snow,
                ent.T_surface,
                f,
                card.dt_seconds,
                c,
                _nemo_glo2eqv_snow_temp=True,
                _nemo_source_round=False,
            )
        )(jax_entry_2906, forcing_2906)
        oracle_es_2906 = np.asarray(frames_2906[1]["e_s"], dtype=np.float64).ravel()
        got_es_unrounded_2906 = np.asarray(
            zdf_unrounded_2906.e_snow, dtype=np.float64
        ).ravel()
        assert probe_module._ulp_diff(got_es_unrounded_2906, oracle_es_2906) >= 4
        _, norm_unrounded_2906 = probe_module.year_gate._normalised_error(
            oracle_es_2906, got_es_unrounded_2906
        )
        assert norm_unrounded_2906 > probe_module.phase2.BAR
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(
                got_es_unrounded_2906.view(np.uint64),
                oracle_es_2906.view(np.uint64),
            )

        step_rounded = jax.jit(
            lambda ent, f: bl_mod.si3_column_step_arrays(
                ent,
                f,
                card.dt_seconds,
                c,
                _nemo_dh_ice_content_accumulation=False,
                _nemo_dh_snow_update=False,
                _nemo_source_round=True,
            )
        )
        step_unrounded = jax.jit(
            lambda ent, f: bl_mod.si3_column_step_arrays(
                ent,
                f,
                card.dt_seconds,
                c,
                _nemo_dh_ice_content_accumulation=False,
                _nemo_dh_snow_update=False,
                _nemo_source_round=False,
            )
        )
        for kt_check, stage_idx, fname in (
            (3963, 4, "sz_i"),
            (2899, 2, "e_i"),
            (3961, 2, "e_s"),
            (4237, 2, "h_s"),
            (4738, 2, "h_i"),
        ):
            fr_k, ex_k = steps[kt_check]
            ent_k = jax.tree.map(
                lambda v: jnp.asarray(v, dtype=jnp.float64),
                probe_module.year_gate._exact_1d_entry(
                    fr_k, sss_m=ex_k["sss"], _nemo_entry_conversion=True
                ),
            )
            forc_k = probe_module.year_gate._forcing(ex_k, jnp.float64)
            maps_r = probe_module.year_gate._stage_maps(
                step_rounded(ent_k, forc_k)
            )
            maps_u = probe_module.year_gate._stage_maps(
                step_unrounded(ent_k, forc_k)
            )
            arr_r = np.asarray(maps_r[stage_idx][fname], dtype=np.float64).ravel()
            arr_u = np.asarray(maps_u[stage_idx][fname], dtype=np.float64).ravel()
            assert probe_module._ulp_diff(arr_r, arr_u) >= 1


def test_r3_all_touched_functions_gradients_and_forward_bits() -> None:
    """Verify finite non-zero gradients, reverse == forward, and literal forward bits on all Round 3 touched functions."""
    card = build_c1d_omip_l3_card(oracle_root=ORACLE_V2_ROOT)
    c = card.config.ice_constants

    e_s0 = jnp.asarray([[1.12e8, 1.09e8, 1.05e8]], dtype=jnp.float64)
    e_i0 = jnp.asarray([[2.95e8, 2.75e8, 2.55e8]], dtype=jnp.float64)
    s_i0 = jnp.asarray([[3.15, 6.30, 9.45]], dtype=jnp.float64)
    t_i0 = jnp.asarray([[258.15, 263.25, 268.35]], dtype=jnp.float64)
    t_s0 = jnp.asarray([[254.15, 259.25, 265.35]], dtype=jnp.float64)
    s_bulk0 = jnp.asarray([4.1], dtype=jnp.float64)
    sss0 = jnp.asarray([34.0], dtype=jnp.float64)
    h_seg0 = jnp.asarray([[0.05, 0.12, 0.11, 0.09]], dtype=jnp.float64)
    e_seg0 = jnp.asarray([[1.15e8, 1.12e8, 1.09e8, 1.05e8]], dtype=jnp.float64)
    h_tot0 = jnp.asarray([0.37], dtype=jnp.float64)

    for fn, args in (
        (
            lambda es: bl_mod.snow_temperature_from_enthalpy(es, c),
            (e_s0,),
        ),
        (
            lambda ei, si: bl_mod.ice_temperature_from_enthalpy(ei, si, c),
            (e_i0, s_i0),
        ),
        (
            lambda ti, si: bl_mod.ice_enthalpy_from_temperature(ti, si, c),
            (t_i0, s_i0),
        ),
        (
            lambda ts: bl_mod.snow_enthalpy_from_temperature(ts, c),
            (t_s0,),
        ),
        (
            lambda ti, si: bl_mod.p07_conductivity(ti, si, c),
            (t_i0, s_i0),
        ),
        (
            lambda sb: bl_mod.option2_salinity_profile(sb, sss0),
            (s_bulk0,),
        ),
        (
            lambda h_seg, e_seg, h_tot: bl_mod._piecewise_remap(
                h_seg, e_seg, h_tot
            ),
            (h_seg0, e_seg0, h_tot0),
        ),
        (
            lambda h_seg, e_seg: bl_mod._nemo_snow_enthalpy_remap(h_seg, e_seg),
            (h_seg0, e_seg0),
        ),
    ):
        primal_direct = fn(*args)

        def loss_with_aux(*xs):
            out = fn(*xs)
            return jnp.sum(out * jnp.asarray([[0.3, -0.7, 1.1]], dtype=jnp.float64)), out

        argnums = tuple(range(len(args)))
        (loss_vag, out_vag), grads = jax.value_and_grad(
            loss_with_aux, argnums=argnums, has_aux=True
        )(*args)
        tangents = tuple(jnp.ones_like(a) for a in args)
        (loss_jvp, out_jvp), (dloss_jvp, _) = jax.jvp(
            loss_with_aux, args, tangents
        )

        np.testing.assert_array_equal(
            np.asarray(primal_direct).view(np.uint64),
            np.asarray(out_vag).view(np.uint64),
        )
        np.testing.assert_array_equal(
            np.asarray(primal_direct).view(np.uint64),
            np.asarray(out_jvp).view(np.uint64),
        )
        np.testing.assert_array_equal(
            np.asarray(loss_vag).view(np.uint64),
            np.asarray(loss_jvp).view(np.uint64),
        )
        for g in grads:
            assert bool(jnp.all(jnp.isfinite(g)))
            assert float(jnp.linalg.norm(g)) > 0.0
        dloss_rev = sum(
            jnp.sum(g * t) for g, t in zip(grads, tangents, strict=True)
        )
        np.testing.assert_allclose(
            np.asarray(dloss_rev),
            np.asarray(dloss_jvp),
            rtol=1.0e-11,
            atol=1.0e-11,
        )

    # Also test `thomas_solve(..., "nemo_unnormalised")` directly
    rng = np.random.default_rng(20261023)
    a_t = jnp.asarray(rng.uniform(-0.2, -0.01, (2, 7)), dtype=jnp.float64).at[..., 0].set(0.0)
    b_t = jnp.asarray(rng.uniform(2.0, 3.0, (2, 7)), dtype=jnp.float64)
    c_t = jnp.asarray(rng.uniform(-0.2, -0.01, (2, 7)), dtype=jnp.float64).at[..., -1].set(0.0)
    d_t = jnp.asarray(rng.uniform(250.0, 272.0, (2, 7)), dtype=jnp.float64)

    primal_ts = bl_mod.thomas_solve(a_t, b_t, c_t, d_t, "nemo_unnormalised")

    def thomas_loss(a, b, c_band, d):
        sol = bl_mod.thomas_solve(a, b, c_band, d, "nemo_unnormalised")
        return jnp.sum(sol), sol

    def thomas_impl_loss(a, b, c_band, d):
        sol = _thomas_solve_impl(a, b, c_band, d, "nemo_unnormalised")
        return jnp.sum(sol), sol

    (l_vag, s_vag), g_ts = jax.value_and_grad(
        thomas_loss, argnums=(0, 1, 2, 3), has_aux=True
    )(a_t, b_t, c_t, d_t)
    t_in = (
        jnp.ones_like(a_t).at[..., 0].set(0.0),
        jnp.ones_like(b_t),
        jnp.ones_like(c_t).at[..., -1].set(0.0),
        jnp.ones_like(d_t),
    )
    (l_jvp, s_jvp), (dl_jvp, _) = jax.jvp(
        thomas_impl_loss, (a_t, b_t, c_t, d_t), t_in
    )
    np.testing.assert_array_equal(
        np.asarray(primal_ts).view(np.uint64),
        np.asarray(s_vag).view(np.uint64),
    )
    np.testing.assert_array_equal(
        np.asarray(primal_ts).view(np.uint64),
        np.asarray(s_jvp).view(np.uint64),
    )
    for g in g_ts:
        assert bool(jnp.all(jnp.isfinite(g)))
        assert float(jnp.linalg.norm(g)) > 0.0
    dl_rev = sum(jnp.sum(g * t) for g, t in zip(g_ts, t_in, strict=True))
    np.testing.assert_allclose(
        np.asarray(dl_rev), np.asarray(dl_jvp), rtol=1.0e-13, atol=1.0e-13
    )


def test_r5_step1_nemo_ice_enthalpy_remap_is_bit_exact_and_fails_when_reverted(
    probe_module,
) -> None:
    """Pin Statement 1 (`_nemo_ice_enthalpy_remap`, `icevar.F90:957-1034`) bit-for-bit on inexact C1D steps and fail when reverted."""
    import inspect

    sig = inspect.signature(bl_mod._piecewise_remap)
    assert "_nemo_source_round" not in sig.parameters

    card = build_c1d_omip_l3_card(oracle_root=ORACLE_V2_ROOT)
    c = card.config.ice_constants
    steps = _load_step_operands(probe_module, (43, 704, 3898, 4230, 4922))

    step_nemo = jax.jit(
        lambda ent, f: bl_mod.si3_column_step_arrays(
            ent,
            f,
            card.dt_seconds,
            c,
            _nemo_ice_remap=True,
            _nemo_source_round=True,
        )
    )
    step_rev = jax.jit(
        lambda ent, f: bl_mod.si3_column_step_arrays(
            ent,
            f,
            card.dt_seconds,
            c,
            _nemo_ice_remap=False,
            _nemo_source_round=True,
        )
    )

    for kt in (43, 704, 3898, 4230, 4922):
        fr, ex = steps[kt]
        ent = jax.tree.map(
            lambda v: jnp.asarray(v, dtype=jnp.float64),
            probe_module.year_gate._exact_1d_entry(
                fr, sss_m=ex["sss"], _nemo_entry_conversion=True
            ),
        )
        forc = probe_module.year_gate._forcing(ex, jnp.float64)
        tr_nemo = step_nemo(ent, forc)
        tr_rev = step_rev(ent, forc)

        oracle_ei = np.asarray(fr[2]["e_i"], dtype=np.float64).ravel()
        got_nemo_ei = np.asarray(
            tr_nemo.post_dh.e_ice, dtype=np.float64
        ).ravel()
        got_rev_ei = np.asarray(tr_rev.post_dh.e_ice, dtype=np.float64).ravel()

        assert probe_module._ulp_diff(got_nemo_ei, oracle_ei) == 0
        np.testing.assert_array_equal(
            got_nemo_ei.view(np.uint64), oracle_ei.view(np.uint64)
        )
        assert probe_module._ulp_diff(got_rev_ei, oracle_ei) >= 5
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(
                got_rev_ei.view(np.uint64), oracle_ei.view(np.uint64)
            )
        _, norm_nemo = probe_module.year_gate._normalised_error(
            oracle_ei, got_nemo_ei
        )
        _, norm_rev = probe_module.year_gate._normalised_error(
            oracle_ei, got_rev_ei
        )
        assert norm_nemo == 0.0
        assert norm_rev > probe_module.phase2.BAR


def test_r5_step2_ice_density_reciprocal_in_dh_step_is_bit_exact_and_fails_when_reverted(
    probe_module,
) -> None:
    """Pin Statement 2 (`-e_i * r1_rhoi`, `icethd_dh.F90:254,391`) bit-for-bit at kt=4893 and fail when reverted."""
    card = build_c1d_omip_l3_card(oracle_root=ORACLE_V2_ROOT)
    c = card.config.ice_constants
    steps = _load_step_operands(probe_module, (4893,))
    fr, ex = steps[4893]
    ent = jax.tree.map(
        lambda v: jnp.asarray(v, dtype=jnp.float64),
        probe_module.year_gate._exact_1d_entry(
            fr, sss_m=ex["sss"], _nemo_entry_conversion=True
        ),
    )
    forc = probe_module.year_gate._forcing(ex, jnp.float64)

    # Pin the exact zEi = -e_i * r1_rhoi and zdE = zEi - zEw bits at kt = 4893 (layer 1, k=0)
    ei_1 = float(np.asarray(fr[1]["e_i"], dtype=np.float64).ravel()[0])
    si_1 = float(np.asarray(fr[1]["sz_i"], dtype=np.float64).ravel()[0])
    r1_rhoi = np.float64(1.0) / np.float64(c.rho_ice)
    zei_recip = np.float64(-ei_1) * r1_rhoi
    zei_fdiv = np.float64(-ei_1) / np.float64(c.rho_ice)
    zew = np.float64(c.c_ocean) * (
        np.float64(-c.liquidus_slope) * np.float64(si_1)
    )
    zde_recip = zei_recip - zew
    zde_fdiv = zei_fdiv - zew

    assert probe_module._u64_hex(zei_recip) == "0xc10ee367890ddc03"
    assert probe_module._u64_hex(zei_fdiv) == "0xc10ee367890ddc02"
    assert probe_module._u64_hex(zde_recip) == "0xc10edecdb63bb14a"
    assert probe_module._u64_hex(zde_fdiv) == "0xc10edecdb63bb149"

    step_recip = jax.jit(
        lambda e, f: bl_mod.si3_column_step_arrays(
            e,
            f,
            card.dt_seconds,
            c,
            _nemo_ice_remap=True,
            _nemo_dh_ice_density_reciprocal=True,
            _nemo_dh_ice_content_accumulation=False,
            _nemo_dh_snow_update=False,
            _nemo_salprof_factoring=True,
        )
    )
    step_fdiv = jax.jit(
        lambda e, f: bl_mod.si3_column_step_arrays(
            e,
            f,
            card.dt_seconds,
            c,
            _nemo_ice_remap=True,
            _nemo_dh_ice_density_reciprocal=False,
            _nemo_dh_ice_content_accumulation=False,
            _nemo_dh_snow_update=False,
            _nemo_salprof_factoring=True,
        )
    )
    maps_recip = probe_module.year_gate._stage_maps(step_recip(ent, forc))
    maps_fdiv = probe_module.year_gate._stage_maps(step_fdiv(ent, forc))

    oracle_post_dh_ei = np.asarray(fr[2]["e_i"], dtype=np.float64).ravel()
    got_recip_post_dh_ei = np.asarray(
        maps_recip[2]["e_i"], dtype=np.float64
    ).ravel()
    got_fdiv_post_dh_ei = np.asarray(
        maps_fdiv[2]["e_i"], dtype=np.float64
    ).ravel()

    assert (
        probe_module._ulp_diff(got_recip_post_dh_ei, oracle_post_dh_ei)
        < probe_module._ulp_diff(got_fdiv_post_dh_ei, oracle_post_dh_ei)
    )
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(
            got_fdiv_post_dh_ei.view(np.uint64),
            got_recip_post_dh_ei.view(np.uint64),
        )

    # All 6 downstream e_i stages (POST_DH, POST_TEMP1, POST_SAL, POST_TEMP2, POST_DO, EXIT)
    # differ when reverted to true division at kt = 4893:
    for stage_idx in (2, 3, 4, 5, 6, 7):
        arr_r = np.asarray(
            maps_recip[stage_idx]["e_i"], dtype=np.float64
        ).ravel()
        arr_d = np.asarray(
            maps_fdiv[stage_idx]["e_i"], dtype=np.float64
        ).ravel()
        assert probe_module._ulp_diff(arr_r, arr_d) >= 1


def test_r5_step3_option2_salinity_profile_factoring_is_bit_exact_and_fails_when_reverted(
    probe_module,
) -> None:
    """Pin Statement 3 (`option2_salinity_profile`, `icevar.F90:504-505,575-576`) bit-for-bit and fail when reverted."""
    card = build_c1d_omip_l3_card(oracle_root=ORACLE_V2_ROOT)
    c = card.config.ice_constants
    steps = _load_step_operands(probe_module, (3961, 3963, 4413))

    step_nemo = jax.jit(
        lambda e, f: bl_mod.si3_column_step_arrays(
            e,
            f,
            card.dt_seconds,
            c,
            _nemo_ice_remap=True,
            _nemo_dh_ice_density_reciprocal=True,
            _nemo_salprof_factoring=True,
        )
    )
    step_rev = jax.jit(
        lambda e, f: bl_mod.si3_column_step_arrays(
            e,
            f,
            card.dt_seconds,
            c,
            _nemo_ice_remap=True,
            _nemo_dh_ice_density_reciprocal=True,
            _nemo_salprof_factoring=False,
        )
    )

    for kt in (3961, 3963, 4413):
        fr, ex = steps[kt]
        ent = jax.tree.map(
            lambda v: jnp.asarray(v, dtype=jnp.float64),
            probe_module.year_gate._exact_1d_entry(
                fr, sss_m=ex["sss"], _nemo_entry_conversion=True
            ),
        )
        forc = probe_module.year_gate._forcing(ex, jnp.float64)
        maps_nemo = probe_module.year_gate._stage_maps(step_nemo(ent, forc))
        maps_rev = probe_module.year_gate._stage_maps(step_rev(ent, forc))

        for stage_idx in (4, 5):
            oracle_szi = np.asarray(
                fr[stage_idx]["sz_i"], dtype=np.float64
            ).ravel()
            got_nemo = np.asarray(
                maps_nemo[stage_idx]["sz_i"], dtype=np.float64
            ).ravel()
            got_rev = np.asarray(
                maps_rev[stage_idx]["sz_i"], dtype=np.float64
            ).ravel()
            assert probe_module._ulp_diff(got_nemo, oracle_szi) == 0
            np.testing.assert_array_equal(
                got_nemo.view(np.uint64), oracle_szi.view(np.uint64)
            )
            assert probe_module._ulp_diff(got_rev, oracle_szi) == 1
            with pytest.raises(AssertionError):
                np.testing.assert_array_equal(
                    got_rev.view(np.uint64), oracle_szi.view(np.uint64)
                )


def test_r5_step4_harness_salprof_zsi1_zsi0_and_missing_sss_refusal(
    probe_module,
) -> None:
    """Pin Statement 4 (`zsi1`/`zsi0` parameter names in `_nemo_salprof` and refusal when `sss_m`/`sss` is absent)."""
    import inspect

    sig = inspect.signature(probe_module.phase2._nemo_salprof)
    params = sig.parameters
    assert "zsi1" in params and params["zsi1"].default == 4.5
    assert "zsi0" in params and params["zsi0"].default == 3.5
    assert "rn_sal_gd" not in params
    assert "rn_sal_fl" not in params
    assert params["sss_m"].default is None

    # Verify custom zsi1/zsi0 transition behavior and agreement with option2_salinity_profile
    s_mid = np.asarray([4.1], dtype=np.float64)
    sss_val = np.asarray([34.0], dtype=np.float64)
    got_default = probe_module.phase2._nemo_salprof(s_mid, sss_m=sss_val)
    ref_default = np.asarray(
        bl_mod.option2_salinity_profile(
            jnp.asarray(s_mid), jnp.asarray(sss_val)
        ),
        dtype=np.float64,
    )
    np.testing.assert_array_equal(
        got_default.view(np.uint64), ref_default.view(np.uint64)
    )

    # Planted namelist values (5.0, 2.0) differ from NEMO's fixed (4.5, 3.5) at s_i = 4.1:
    got_planted_namelist = probe_module.phase2._nemo_salprof(
        s_mid, sss_m=sss_val, zsi1=5.0, zsi0=2.0
    )
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(
            got_planted_namelist.view(np.uint64),
            got_default.view(np.uint64),
        )

    # Refusal when sss_m / sss is absent or non-finite across all harness entry points:
    steps = _load_step_operands(probe_module, (1,))
    fr_1, ex_1 = steps[1]
    with pytest.raises(
        probe_module.phase2.GateError, match="missing sea-surface salinity"
    ):
        probe_module.phase2._nemo_salprof(s_mid)
    with pytest.raises(
        probe_module.phase2.GateError, match="missing sea-surface salinity"
    ):
        probe_module.phase2._nemo_salprof(s_mid, sss_m=None)
    with pytest.raises(
        probe_module.phase2.GateError,
        match="invalid or non-finite sea-surface salinity",
    ):
        probe_module.phase2._nemo_salprof(s_mid, sss_m=np.nan)
    with pytest.raises(
        probe_module.phase2.GateError, match="missing sea-surface salinity"
    ):
        probe_module.phase2._entry_arrays(fr_1[0])
    with pytest.raises(
        probe_module.phase2.GateError, match="missing sea-surface salinity"
    ):
        probe_module.year_gate._exact_1d_entry(fr_1)

    ex_missing_sss = {k: v for k, v in ex_1.items() if k != "sss"}
    with pytest.raises(
        probe_module.phase2.GateError, match="missing sea-surface salinity"
    ):
        probe_module.phase2._forcing(ex_missing_sss)
    with pytest.raises(
        probe_module.phase2.GateError, match="missing sea-surface salinity"
    ):
        probe_module.year_gate._forcing(ex_missing_sss, jnp.float64)
    with pytest.raises(
        probe_module.phase2.GateError, match="missing sea-surface salinity"
    ):
        probe_module.year_gate._exact_1d_entry(
            fr_1, sss_m=ex_missing_sss.get("sss")
        )


def test_r5_all_touched_functions_gradients_and_forward_bits(
    probe_module, monkeypatch
) -> None:
    """Verify finite nonzero gradients, reverse == forward, and literal 0-ULP forward bits through all Round 5 touched functions."""
    card = build_c1d_omip_l3_card(oracle_root=ORACLE_V2_ROOT)
    c = card.config.ice_constants

    # 1. _nemo_ice_enthalpy_remap and option2_salinity_profile
    h_seg0 = jnp.asarray(
        [[0.02, 0.58, 0.62, 0.59, 0.04]], dtype=jnp.float64
    )
    e_seg0 = jnp.asarray(
        [[2.98e8, 2.91e8, 2.74e8, 2.53e8, 2.31e8]], dtype=jnp.float64
    )
    h_new0 = jnp.sum(h_seg0, axis=-1)
    s_bulk0 = jnp.asarray([4.123456789012345], dtype=jnp.float64)
    sss0 = jnp.asarray([34.0], dtype=jnp.float64)

    for fn, args in (
        (
            lambda hs, es: bl_mod._nemo_ice_enthalpy_remap(
                hs, es, _nemo_source_round=True
            ),
            (h_seg0, e_seg0),
        ),
        (
            lambda sb, sss: bl_mod.option2_salinity_profile(
                sb, sss, _nemo_source_round=True, _nemo_salprof_factoring=True
            ),
            (s_bulk0, sss0),
        ),
    ):
        primal_direct = fn(*args)

        def loss_with_aux(*xs):
            out = fn(*xs)
            w = jnp.asarray([[0.35, -0.65, 1.15]], dtype=jnp.float64)
            return jnp.sum(out * w), out

        argnums = tuple(range(len(args)))
        (loss_vag, out_vag), grads = jax.value_and_grad(
            loss_with_aux, argnums=argnums, has_aux=True
        )(*args)
        tangents = tuple(jnp.ones_like(a) * 1.0e-3 for a in args)
        (loss_jvp, out_jvp), (dloss_jvp, _) = jax.jvp(
            loss_with_aux, args, tangents
        )

        np.testing.assert_array_equal(
            np.asarray(primal_direct).view(np.uint64),
            np.asarray(out_vag).view(np.uint64),
        )
        np.testing.assert_array_equal(
            np.asarray(primal_direct).view(np.uint64),
            np.asarray(out_jvp).view(np.uint64),
        )
        np.testing.assert_array_equal(
            np.asarray(loss_vag).view(np.uint64),
            np.asarray(loss_jvp).view(np.uint64),
        )
        for g in grads:
            assert bool(jnp.all(jnp.isfinite(g)))
        assert float(jnp.linalg.norm(grads[0])) > 0.0
        dloss_rev = sum(
            jnp.sum(g * t) for g, t in zip(grads, tangents, strict=True)
        )
        np.testing.assert_allclose(
            np.asarray(dloss_rev),
            np.asarray(dloss_jvp),
            rtol=1.0e-11,
            atol=1.0e-11,
        )

    # 2. Full _dh_step and si3_column_step_arrays at an active melt/remap step (kt = 4413)
    steps = _load_step_operands(probe_module, (4413,))
    fr, ex = steps[4413]
    ent = jax.tree.map(
        lambda v: jnp.asarray(v, dtype=jnp.float64),
        probe_module.year_gate._exact_1d_entry(
            fr, sss_m=ex["sss"], _nemo_entry_conversion=True
        ),
    )
    forc = probe_module.year_gate._forcing(ex, jnp.float64)

    def col_tuple(h_i, e_i, s_bulk, qns):
        st = ent._replace(h_ice=h_i, e_ice=e_i, S_bulk=s_bulk)
        fc = forc._replace(qns_ice=qns)
        tr = bl_mod.si3_column_step_arrays(st, fc, card.dt_seconds, c)
        return (
            tr.post_dh.h_ice,
            tr.post_dh.e_ice,
            tr.post_sal.S_layers,
            tr.exit.h_ice,
            tr.exit.e_ice,
        )

    weights = (
        jnp.asarray([1.25], dtype=jnp.float64),
        jnp.asarray([[0.25, -0.5, 0.75]], dtype=jnp.float64),
        jnp.asarray([[-0.15, 0.35, -0.55]], dtype=jnp.float64),
        jnp.asarray([0.85], dtype=jnp.float64),
        jnp.asarray([[0.125, -0.25, 0.375]], dtype=jnp.float64),
    )

    def col_loss_with_aux(h_i, e_i, s_bulk, qns):
        outs = col_tuple(h_i, e_i, s_bulk, qns)
        loss = sum(
            jnp.sum(w * o) for w, o in zip(weights, outs, strict=True)
        )
        return loss, outs

    primal_col = col_tuple(ent.h_ice, ent.e_ice, ent.S_bulk, forc.qns_ice)
    (loss_col_vag, outs_col_vag), grads_col = jax.value_and_grad(
        col_loss_with_aux, argnums=(0, 1, 2, 3), has_aux=True
    )(ent.h_ice, ent.e_ice, ent.S_bulk, forc.qns_ice)

    for p_arr, v_arr in zip(primal_col, outs_col_vag, strict=True):
        np.testing.assert_array_equal(
            np.asarray(p_arr).view(np.uint64),
            np.asarray(v_arr).view(np.uint64),
        )
    for g in grads_col:
        assert bool(jnp.all(jnp.isfinite(g)))
        assert float(jnp.linalg.norm(g)) > 0.0

    tangents_col = (
        jnp.ones_like(ent.h_ice) * 1.0e-3,
        jnp.ones_like(ent.e_ice),
        jnp.ones_like(ent.S_bulk) * 1.0e-3,
        jnp.ones_like(forc.qns_ice),
    )
    with monkeypatch.context() as mp:
        mp.setattr(bl_mod, "thomas_solve", _thomas_solve_impl)
        (loss_col_jvp, outs_col_jvp), (dloss_col_jvp, _) = jax.jvp(
            col_loss_with_aux,
            (ent.h_ice, ent.e_ice, ent.S_bulk, forc.qns_ice),
            tangents_col,
        )
    for p_arr, j_arr in zip(primal_col, outs_col_jvp, strict=True):
        np.testing.assert_array_equal(
            np.asarray(p_arr).view(np.uint64),
            np.asarray(j_arr).view(np.uint64),
        )
    dloss_col_rev = sum(
        jnp.sum(g * t) for g, t in zip(grads_col, tangents_col, strict=True)
    )
    np.testing.assert_allclose(
        np.asarray(dloss_col_rev),
        np.asarray(dloss_col_jvp),
        rtol=1.0e-10,
        atol=1.0e-10,
    )

    # Planted control: stop_gradient must cause nonzero gradient check to fail
    planted_g = jax.grad(
        lambda es: jnp.sum(
            bl_mod._nemo_ice_enthalpy_remap(
                h_seg0, jax.lax.stop_gradient(es)
            )
        )
    )(e_seg0)
    with pytest.raises(AssertionError):
        assert float(jnp.linalg.norm(planted_g)) > 0.0


def test_r6_step1_dh_ice_content_accumulation_is_bit_exact_and_fails_when_reverted(
    probe_module,
) -> None:
    """Pin Round 6 Statement Family 1 (`_nemo_dh_ice_content_accumulation`, `icethd_dh.f90:138-147,243-325,340-374,378-434,480-495,521`) bit-for-bit on inexact C1D steps and snow-ice flooding, and fail when reverted."""
    card = build_c1d_omip_l3_card(oracle_root=ORACLE_V2_ROOT)
    c = card.config.ice_constants
    over_bar_kts = (3996, 4014, 4092, 4400, 4413, 5241)
    steps = _load_step_operands(probe_module, over_bar_kts + (4359,))

    step_nemo = jax.jit(
        lambda ent, f: bl_mod.si3_column_step_arrays(
            ent,
            f,
            card.dt_seconds,
            c,
            _nemo_dh_ice_content_accumulation=True,
            _nemo_dh_snow_update=False,
        )
    )
    step_rev = jax.jit(
        lambda ent, f: bl_mod.si3_column_step_arrays(
            ent,
            f,
            card.dt_seconds,
            c,
            _nemo_dh_ice_content_accumulation=False,
            _nemo_dh_snow_update=False,
        )
    )

    for kt in over_bar_kts:
        fr, ex = steps[kt]
        ent = jax.tree.map(
            lambda v: jnp.asarray(v, dtype=jnp.float64),
            probe_module.year_gate._exact_1d_entry(
                fr, sss_m=ex["sss"], _nemo_entry_conversion=True
            ),
        )
        forc = probe_module.year_gate._forcing(ex, jnp.float64)
        maps_nemo = probe_module.year_gate._stage_maps(step_nemo(ent, forc))
        maps_rev = probe_module.year_gate._stage_maps(step_rev(ent, forc))

        # 1-D stages (POST_DH..POST_TEMP2): e_i and h_i are 0 ULP vs oracle,
        # and reverting _nemo_dh_ice_content_accumulation=False fails bit-exactness:
        oracle_dh_ei = np.asarray(fr[2]["e_i"], dtype=np.float64).ravel()
        got_nemo_dh_ei = np.asarray(
            maps_nemo[2]["e_i"], dtype=np.float64
        ).ravel()
        got_rev_dh_ei = np.asarray(
            maps_rev[2]["e_i"], dtype=np.float64
        ).ravel()
        assert probe_module._ulp_diff(got_nemo_dh_ei, oracle_dh_ei) == 0
        np.testing.assert_array_equal(
            got_nemo_dh_ei.view(np.uint64), oracle_dh_ei.view(np.uint64)
        )
        assert probe_module._ulp_diff(got_rev_dh_ei, oracle_dh_ei) >= 3
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(
                got_rev_dh_ei.view(np.uint64), oracle_dh_ei.view(np.uint64)
            )

        # Global stages (POST_DO and EXIT): norm_err <= BAR with Statement 1,
        # and reverting _nemo_dh_ice_content_accumulation=False crosses BAR (> 1e-15):
        for stage_idx in (6, 7):
            oracle_glo_ei = np.asarray(
                probe_module.phase2._center_global(fr[stage_idx])["e_i"],
                dtype=np.float64,
            ).ravel()
            got_nemo_glo = np.asarray(
                maps_nemo[stage_idx]["e_i"], dtype=np.float64
            ).ravel()
            got_rev_glo = np.asarray(
                maps_rev[stage_idx]["e_i"], dtype=np.float64
            ).ravel()
            _, norm_nemo = probe_module.year_gate._normalised_error(
                oracle_glo_ei, got_nemo_glo
            )
            _, norm_rev = probe_module.year_gate._normalised_error(
                oracle_glo_ei, got_rev_glo
            )
            assert norm_nemo <= probe_module.phase2.BAR
            assert norm_rev > probe_module.phase2.BAR

    # Also pin POST_DH.h_i at kt = 4359 (worst h_i ULP step in Round 5, 3 ULP -> 0 ULP):
    fr_4359, ex_4359 = steps[4359]
    ent_4359 = jax.tree.map(
        lambda v: jnp.asarray(v, dtype=jnp.float64),
        probe_module.year_gate._exact_1d_entry(
            fr_4359, sss_m=ex_4359["sss"], _nemo_entry_conversion=True
        ),
    )
    forc_4359 = probe_module.year_gate._forcing(ex_4359, jnp.float64)
    maps_nemo_4359 = probe_module.year_gate._stage_maps(
        step_nemo(ent_4359, forc_4359)
    )
    maps_rev_4359 = probe_module.year_gate._stage_maps(
        step_rev(ent_4359, forc_4359)
    )
    oracle_hi_4359 = np.asarray(fr_4359[2]["h_i"], dtype=np.float64).ravel()
    got_nemo_hi_4359 = np.asarray(
        maps_nemo_4359[2]["h_i"], dtype=np.float64
    ).ravel()
    got_rev_hi_4359 = np.asarray(
        maps_rev_4359[2]["h_i"], dtype=np.float64
    ).ravel()
    assert probe_module._ulp_diff(got_nemo_hi_4359, oracle_hi_4359) == 0
    np.testing.assert_array_equal(
        got_nemo_hi_4359.view(np.uint64), oracle_hi_4359.view(np.uint64)
    )
    assert probe_module._ulp_diff(got_rev_hi_4359, oracle_hi_4359) == 2

    # Also pin the snow-ice flooding branch (`flood > 0`, `icethd_dh.f90:480-495`)
    # on an inexact column where snow mass floods basal snow into snow-ice:
    fr_sni, ex_sni = steps[5241]
    ent_sni_base = probe_module.year_gate._exact_1d_entry(
        fr_sni, sss_m=ex_sni["sss"], _nemo_entry_conversion=True
    )
    ent_sni = jax.tree.map(
        lambda v: jnp.asarray(v, dtype=jnp.float64),
        ent_sni_base._replace(
            h_ice=np.asarray([0.3712345678901234], dtype=np.float64),
            h_snow=np.asarray([0.1987654321098765], dtype=np.float64),
            e_snow=np.asarray(
                [[1.081234567890123e8, 1.092345678901234e8, 1.103456789012345e8]],
                dtype=np.float64,
            ),
        ),
    )
    forc_sni = probe_module.year_gate._forcing(ex_sni, jnp.float64)
    tr_sni_nemo = step_nemo(ent_sni, forc_sni)
    tr_sni_rev = step_rev(ent_sni, forc_sni)
    assert float(np.asarray(tr_sni_nemo.post_dh.h_ice).ravel()[0]) > float(
        np.asarray(ent_sni.h_ice).ravel()[0]
    )
    assert (
        probe_module._ulp_diff(
            np.asarray(tr_sni_nemo.post_dh.e_ice, dtype=np.float64).ravel(),
            np.asarray(tr_sni_rev.post_dh.e_ice, dtype=np.float64).ravel(),
        )
        >= 1
    )


def test_r6_step2_dh_snow_update_is_bit_exact_and_fails_when_reverted(
    probe_module,
) -> None:
    """Pin Round 6 Statement Family 2 (`_nemo_dh_snow_update`, `icethd_dh.f90:178-187,194-195,231`) bit-for-bit on all 7 snow over-bar steps and cancelling-pair steps, and fail when reverted."""
    card = build_c1d_omip_l3_card(oracle_root=ORACLE_V2_ROOT)
    c = card.config.ice_constants
    snow_over_bar_kts = (4943, 4945, 5460, 5644, 5822, 5960, 5962)
    cancelling_pair_kts = (5733, 5932)
    steps = _load_step_operands(
        probe_module, snow_over_bar_kts + cancelling_pair_kts
    )

    step_nemo = jax.jit(
        lambda ent, f: bl_mod.si3_column_step_arrays(
            ent,
            f,
            card.dt_seconds,
            c,
            _nemo_dh_ice_content_accumulation=True,
            _nemo_dh_snow_update=True,
        )
    )
    step_rev = jax.jit(
        lambda ent, f: bl_mod.si3_column_step_arrays(
            ent,
            f,
            card.dt_seconds,
            c,
            _nemo_dh_ice_content_accumulation=True,
            _nemo_dh_snow_update=False,
        )
    )

    for kt in snow_over_bar_kts:
        fr, ex = steps[kt]
        ent = jax.tree.map(
            lambda v: jnp.asarray(v, dtype=jnp.float64),
            probe_module.year_gate._exact_1d_entry(
                fr, sss_m=ex["sss"], _nemo_entry_conversion=True
            ),
        )
        forc = probe_module.year_gate._forcing(ex, jnp.float64)
        maps_nemo = probe_module.year_gate._stage_maps(step_nemo(ent, forc))
        maps_rev = probe_module.year_gate._stage_maps(step_rev(ent, forc))

        # 1. All 1-D snow stages (POST_DH..POST_TEMP2): e_s and h_s are 0 ULP vs oracle,
        #    and reverting _nemo_dh_snow_update=False fails bit-exactness:
        for stage_idx in (2, 3, 4, 5):
            oracle_es = np.asarray(fr[stage_idx]["e_s"], dtype=np.float64).ravel()
            oracle_hs = np.asarray(fr[stage_idx]["h_s"], dtype=np.float64).ravel()
            got_nemo_es = np.asarray(
                maps_nemo[stage_idx]["e_s"], dtype=np.float64
            ).ravel()
            got_rev_es = np.asarray(
                maps_rev[stage_idx]["e_s"], dtype=np.float64
            ).ravel()
            got_nemo_hs = np.asarray(
                maps_nemo[stage_idx]["h_s"], dtype=np.float64
            ).ravel()
            got_rev_hs = np.asarray(
                maps_rev[stage_idx]["h_s"], dtype=np.float64
            ).ravel()

            assert probe_module._ulp_diff(got_nemo_es, oracle_es) == 0
            assert probe_module._ulp_diff(got_nemo_hs, oracle_hs) == 0
            np.testing.assert_array_equal(
                got_nemo_es.view(np.uint64), oracle_es.view(np.uint64)
            )
            np.testing.assert_array_equal(
                got_nemo_hs.view(np.uint64), oracle_hs.view(np.uint64)
            )
            assert probe_module._ulp_diff(got_rev_es, oracle_es) >= 1
            if kt in (5460, 5960, 5962):
                assert probe_module._ulp_diff(got_rev_hs, oracle_hs) >= 1
            if kt in (5460, 5644):
                _, norm_rev_1d = probe_module.year_gate._normalised_error(
                    oracle_es, got_rev_es
                )
                assert norm_rev_1d > probe_module.phase2.BAR

        # 2. Global stages (POST_DO and EXIT): e_s and v_s are 0 ULP vs oracle,
        #    and reverting _nemo_dh_snow_update=False crosses BAR (> 1e-15) on all 7 steps:
        for stage_idx in (6, 7):
            oracle_glo = probe_module.phase2._center_global(fr[stage_idx])
            oracle_glo_es = np.asarray(oracle_glo["e_s"], dtype=np.float64).ravel()
            oracle_glo_vs = np.asarray(oracle_glo["v_s"], dtype=np.float64).ravel()
            got_nemo_glo_es = np.asarray(
                maps_nemo[stage_idx]["e_s"], dtype=np.float64
            ).ravel()
            got_rev_glo_es = np.asarray(
                maps_rev[stage_idx]["e_s"], dtype=np.float64
            ).ravel()
            got_nemo_glo_vs = np.asarray(
                maps_nemo[stage_idx]["v_s"], dtype=np.float64
            ).ravel()
            assert probe_module._ulp_diff(got_nemo_glo_es, oracle_glo_es) == 0
            assert probe_module._ulp_diff(got_nemo_glo_vs, oracle_glo_vs) == 0
            np.testing.assert_array_equal(
                got_nemo_glo_es.view(np.uint64), oracle_glo_es.view(np.uint64)
            )
            np.testing.assert_array_equal(
                got_nemo_glo_vs.view(np.uint64), oracle_glo_vs.view(np.uint64)
            )
            _, norm_rev_glo = probe_module.year_gate._normalised_error(
                oracle_glo_es, got_rev_glo_es
            )
            assert norm_rev_glo > probe_module.phase2.BAR

    # 3. Cancelling-pair verification at kt = 5733 and kt = 5932:
    #    In `zh_s(0)` (`icethd_dh.f90:179`) and `zdeltah` (`icethd_dh.f90:194`),
    #    XLA constant-folding of `dt * r1_rhos` gives a +1 ULP error in `zh_s(0)`
    #    and a -1 ULP error in `zdeltah` that cancelled each other in Round 5;
    #    fixingboth in `_nemo_dh_snow_update=True` preserves 0 ULP on both operands
    #    and on `POST_DH.h_s` / `POST_DH.e_s`.
    for kt in cancelling_pair_kts:
        fr, ex = steps[kt]
        ent = jax.tree.map(
            lambda v: jnp.asarray(v, dtype=jnp.float64),
            probe_module.year_gate._exact_1d_entry(
                fr, sss_m=ex["sss"], _nemo_entry_conversion=True
            ),
        )
        forc = probe_module.year_gate._forcing(ex, jnp.float64)
        maps_nemo = probe_module.year_gate._stage_maps(step_nemo(ent, forc))
        oracle_hs = np.asarray(fr[2]["h_s"], dtype=np.float64).ravel()
        oracle_es = np.asarray(fr[2]["e_s"], dtype=np.float64).ravel()
        got_hs = np.asarray(maps_nemo[2]["h_s"], dtype=np.float64).ravel()
        got_es = np.asarray(maps_nemo[2]["e_s"], dtype=np.float64).ravel()
        assert probe_module._ulp_diff(got_hs, oracle_hs) == 0
        assert probe_module._ulp_diff(got_es, oracle_es) == 0


def test_r6_all_touched_functions_gradients_and_forward_bits(
    probe_module, monkeypatch
) -> None:
    """Verify finite nonzero gradients, reverse == forward, and literal 0-ULP forward bits through all Round 6 touched functions."""
    card = build_c1d_omip_l3_card(oracle_root=ORACLE_V2_ROOT)
    c = card.config.ice_constants

    # 1. `_nemo_ice_enthalpy_remap` with direct `content=ze_i_old` input (`icevar.f90:971-1048`)
    h_seg0 = jnp.asarray(
        [[0.02, 0.58, 0.62, 0.59, 0.04]], dtype=jnp.float64
    )
    ze_old0 = jnp.asarray(
        [[5.96e6, 1.6878e8, 1.6988e8, 1.4927e8, 9.24e6]], dtype=jnp.float64
    )

    def remap_content_fn(hs, ze_c):
        return bl_mod._nemo_ice_enthalpy_remap(
            hs, content=ze_c, _nemo_source_round=True
        )

    primal_rc = remap_content_fn(h_seg0, ze_old0)

    def rc_loss_with_aux(hs, ze_c):
        out = remap_content_fn(hs, ze_c)
        w = jnp.asarray([[0.35, -0.65, 1.15]], dtype=jnp.float64)
        return jnp.sum(out * w), out

    (loss_rc_vag, out_rc_vag), grads_rc = jax.value_and_grad(
        rc_loss_with_aux, argnums=(0, 1), has_aux=True
    )(h_seg0, ze_old0)
    tangents_rc = (jnp.ones_like(h_seg0) * 1.0e-3, jnp.ones_like(ze_old0))
    (loss_rc_jvp, out_rc_jvp), (dloss_rc_jvp, _) = jax.jvp(
        rc_loss_with_aux, (h_seg0, ze_old0), tangents_rc
    )
    np.testing.assert_array_equal(
        np.asarray(primal_rc).view(np.uint64),
        np.asarray(out_rc_vag).view(np.uint64),
    )
    np.testing.assert_array_equal(
        np.asarray(primal_rc).view(np.uint64),
        np.asarray(out_rc_jvp).view(np.uint64),
    )
    for g in grads_rc:
        assert bool(jnp.all(jnp.isfinite(g)))
        assert float(jnp.linalg.norm(g)) > 0.0
    dloss_rc_rev = sum(
        jnp.sum(g * t) for g, t in zip(grads_rc, tangents_rc, strict=True)
    )
    np.testing.assert_allclose(
        np.asarray(dloss_rc_rev),
        np.asarray(dloss_rc_jvp),
        rtol=1.0e-11,
        atol=1.0e-11,
    )

    # 2. Full `si3_column_step_arrays` and `_dh_step` across both an active
    #    ice-melt step (kt = 5241) and an active snow-precip/sublimation/melt step (kt = 5460)
    steps = _load_step_operands(probe_module, (5241, 5460))
    for kt in (5241, 5460):
        fr, ex = steps[kt]
        ent = jax.tree.map(
            lambda v: jnp.asarray(v, dtype=jnp.float64),
            probe_module.year_gate._exact_1d_entry(
                fr, sss_m=ex["sss"], _nemo_entry_conversion=True
            ),
        )
        forc = probe_module.year_gate._forcing(ex, jnp.float64)

        def col_r6_tuple(h_i, h_s, e_i, e_s, sprecip, evap):
            st = ent._replace(h_ice=h_i, h_snow=h_s, e_ice=e_i, e_snow=e_s)
            fc = forc._replace(snow_precipitation=sprecip, evaporation=evap)
            tr = bl_mod.si3_column_step_arrays(
                st,
                fc,
                card.dt_seconds,
                c,
                _nemo_dh_ice_content_accumulation=True,
                _nemo_dh_snow_update=True,
            )
            return (
                tr.post_dh.h_ice,
                tr.post_dh.h_snow,
                tr.post_dh.e_ice,
                tr.post_dh.e_snow,
                tr.exit.h_ice,
                tr.exit.e_ice,
            )

        weights_r6 = (
            jnp.asarray([1.25], dtype=jnp.float64),
            jnp.asarray([0.95], dtype=jnp.float64),
            jnp.asarray([[0.25, -0.5, 0.75]], dtype=jnp.float64),
            jnp.asarray([[-0.15, 0.35, -0.55]], dtype=jnp.float64),
            jnp.asarray([0.85], dtype=jnp.float64),
            jnp.asarray([[0.125, -0.25, 0.375]], dtype=jnp.float64),
        )

        def col_r6_loss(h_i, h_s, e_i, e_s, sprecip, evap):
            outs = col_r6_tuple(h_i, h_s, e_i, e_s, sprecip, evap)
            loss = sum(
                jnp.sum(w * o) for w, o in zip(weights_r6, outs, strict=True)
            )
            return loss, outs

        args_r6 = (
            ent.h_ice,
            ent.h_snow,
            ent.e_ice,
            ent.e_snow,
            forc.snow_precipitation,
            forc.evaporation,
        )
        primal_r6 = col_r6_tuple(*args_r6)
        (loss_r6_vag, outs_r6_vag), grads_r6 = jax.value_and_grad(
            col_r6_loss, argnums=(0, 1, 2, 3, 4, 5), has_aux=True
        )(*args_r6)

        for p_arr, v_arr in zip(primal_r6, outs_r6_vag, strict=True):
            np.testing.assert_array_equal(
                np.asarray(p_arr).view(np.uint64),
                np.asarray(v_arr).view(np.uint64),
            )
        for g in grads_r6:
            assert bool(jnp.all(jnp.isfinite(g)))
        assert float(jnp.linalg.norm(grads_r6[0])) > 0.0
        assert float(jnp.linalg.norm(grads_r6[2])) > 0.0
        if kt == 5460:
            assert float(jnp.linalg.norm(grads_r6[1])) > 0.0
            assert float(jnp.linalg.norm(grads_r6[3])) > 0.0
            assert float(jnp.linalg.norm(grads_r6[4])) > 0.0
            assert float(jnp.linalg.norm(grads_r6[5])) > 0.0

        tangents_r6 = (
            jnp.ones_like(ent.h_ice) * 1.0e-3,
            jnp.ones_like(ent.h_snow) * 1.0e-3,
            jnp.ones_like(ent.e_ice),
            jnp.ones_like(ent.e_snow),
            jnp.ones_like(forc.snow_precipitation) * 1.0e-7,
            jnp.ones_like(forc.evaporation) * 1.0e-7,
        )
        with monkeypatch.context() as mp:
            mp.setattr(bl_mod, "thomas_solve", _thomas_solve_impl)
            (loss_r6_jvp, outs_r6_jvp), (dloss_r6_jvp, _) = jax.jvp(
                col_r6_loss, args_r6, tangents_r6
            )
        for p_arr, j_arr in zip(primal_r6, outs_r6_jvp, strict=True):
            np.testing.assert_array_equal(
                np.asarray(p_arr).view(np.uint64),
                np.asarray(j_arr).view(np.uint64),
            )
        dloss_r6_rev = sum(
            jnp.sum(g * t) for g, t in zip(grads_r6, tangents_r6, strict=True)
        )
        np.testing.assert_allclose(
            np.asarray(dloss_r6_rev),
            np.asarray(dloss_r6_jvp),
            rtol=1.0e-10,
            atol=1.0e-10,
        )

    # Planted control: stop_gradient on `content` must cause nonzero gradient check to fail
    planted_g_rc = jax.grad(
        lambda zc: jnp.sum(
            bl_mod._nemo_ice_enthalpy_remap(
                h_seg0, content=jax.lax.stop_gradient(zc)
            )
        )
    )(ze_old0)
    with pytest.raises(AssertionError):
        assert float(jnp.linalg.norm(planted_g_rc)) > 0.0
