"""Round 344: native-transcendental fast-literal K0 configuration."""

from __future__ import annotations

import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean import bulk_flux_omip as bulk
from legoesm.ocean.fidelity.nemo_testcase_recipe import build_orca2_zps_card
from legoesm.ocean.physics.vertical_mixing.config import TKEConfig
from legoesm.ocean.physics.vertical_mixing.double_diffusion import DoubleDiffusionConfig
from legoesm.ocean.physics.vertical_mixing.internal_wave_mixing import IWMConfig
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round343_speed_benchmark as bench,
)


DECK = bench.DEFAULT_DECK_ROOT


@pytest.mark.skipif(not DECK.is_dir(), reason="ORCA2 K0 deck is unavailable")
def test_fast_literal_card_changes_only_transcendentals_and_plant() -> None:
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    base = bulk.build_nemo_orca2_k0_ocean_card(DECK)
    fast = bench.build_fast_literal_mode_orca2_k0_card(base)
    assert bench.verify_fast_literal_card(base, fast) == {
        "changed_fields": ["ocean.transcendentals"],
        "retained_inventory_entries": 44,
    }
    assert fast.ocean.recipe is base.ocean.recipe
    vm = fast.ocean.recipe.model_config.physics.vertical_mixing
    assert (vm.ddm.enabled, vm.ddm.form) == (True, "nemo_literal")
    assert (vm.iwm.form, vm.iwm.tsdiff) == ("nemo_literal", True)
    assert vm.tke.tke_htau_evaluation == "nemo_scalar_libm"
    assert vm.tke.tke_etau_exponential_evaluation == "nemo_scalar_libm"

    bad_tke = vm.tke._replace(tke_htau_evaluation="jax_expression")
    bad_mc = fast.ocean.recipe.model_config._replace(
        physics=fast.ocean.recipe.model_config.physics._replace(
            vertical_mixing=vm._replace(tke=bad_tke)
        )
    )
    bad_recipe = fast.ocean.recipe._replace(
        model_config=bad_mc, physics_config=bad_mc.physics
    )
    bad = fast._replace(ocean=fast.ocean._replace(recipe=bad_recipe))
    with pytest.raises(bench.GateError, match="tke_htau_evaluation"):
        bench.verify_fast_literal_card(base, bad)


@pytest.mark.skipif(not DECK.is_dir(), reason="ORCA2 K0 deck is unavailable")
def test_rung10_s4_k0_and_all_defaults_remain_exact() -> None:
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    rung10 = build_orca2_zps_card(DECK, initial_state="nemo_k0_istate")
    s4 = bulk.build_nemo_orca2_k0_ocean_card(DECK)
    assert s4.ocean.recipe.model_config == rung10.recipe.model_config
    assert s4.ocean.recipe.physics_config == rung10.recipe.physics_config
    assert s4.ocean.transcendentals == rung10.transcendentals == "libm"
    vm = rung10.recipe.model_config.physics.vertical_mixing
    assert (vm.ddm.enabled, vm.ddm.form) == (True, "nemo_literal")
    assert (vm.iwm.form, vm.iwm.tsdiff) == ("nemo_literal", True)
    assert vm.tke.tke_htau_evaluation == "nemo_scalar_libm"
    assert vm.tke.tke_etau_exponential_evaluation == "nemo_scalar_libm"
    assert TKEConfig().tke_htau_evaluation == "jax_expression"
    assert TKEConfig().tke_etau_exponential_evaluation == "jax_expression"
    assert (DoubleDiffusionConfig().enabled, DoubleDiffusionConfig().form) == (
        False, "hybrid"
    )
    assert (IWMConfig().form, IWMConfig().tsdiff) == ("generic", False)


def _synthetic_arm(arm_id: str, value: float) -> dict:
    timing = bench.compute_step_timing_summary([10.0, 9.0] + [0.5] * 62)
    daily = []
    for kt in bench.DAILY_CHECK_STEPS_8D:
        row = {"kt": kt}
        row.update({metric: value for metric in bench._ACCURACY_METRICS})
        daily.append(row)
    return {
        "arm_id": arm_id,
        "variant": "fast_literal" if arm_id.startswith("fast") else "cat_a_plus_b",
        "dtype": "float64",
        "transcendentals": "native",
        "timing": timing,
        "daily_accuracy": daily,
    }


def test_fast_report_direction_gate_and_plant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = {
        "arms": {
            "fast_literal_cpu_fp64": _synthetic_arm("fast_literal_cpu_fp64", 1.0e-8),
            "fast_literal_gpu_fp64": _synthetic_arm("fast_literal_gpu_fp64", 2.0e-8),
        },
        "round343_normal_fp64": {
            "normal_cpu_fp64": _synthetic_arm("normal_cpu_fp64", 1.0e-3),
            "normal_gpu_fp64": _synthetic_arm("normal_gpu_fp64", 1.0e-3),
        },
    }
    bench.validate_fast_literal_benchmark_report(report)
    report["arms"]["fast_literal_cpu_fp64"]["daily_accuracy"][0][
        "sst_rms_wet_c"
    ] = 2.0e-3
    with pytest.raises(bench.GateError, match="is not closer"):
        bench.validate_fast_literal_benchmark_report(report)

    # Assembly consumes the saved arm/baseline schema, not hidden live state.
    report["arms"]["fast_literal_cpu_fp64"]["daily_accuracy"][0][
        "sst_rms_wet_c"
    ] = 1.0e-8
    for arm_id, arm in report["arms"].items():
        (tmp_path / f"arm_{arm_id}.json").write_text(json.dumps(arm))
    old_path = tmp_path / "round343.json"
    old_path.write_text(json.dumps({"arms": report["round343_normal_fp64"]}))
    monkeypatch.setattr(bench, "worktree_stamp", lambda: "test-clean-stamp")
    assembled = bench.assemble_fast_literal_benchmark_report(
        output_dir=tmp_path, round343_report=old_path
    )
    assert assembled["round"] == "orca2_r344"


@pytest.mark.skipif(
    os.environ.get("ORCA2_R344_REAL_GRADIENT") != "1",
    reason="operator-only real fast-literal K0 CPU gradient",
)
def test_real_fast_literal_k0_n1_reverse_equals_forward() -> None:
    set_policy(PrecisionPolicy.fp64(transcendentals="native"))
    base = bulk.build_nemo_orca2_k0_ocean_card(DECK)
    card = bench.build_fast_literal_mode_orca2_k0_card(base)
    controls = bench.precompute_step_controls(
        card,
        DECK,
        bench.DEFAULT_CORE_MANIFEST,
        bench.DEFAULT_ICE_COVER_PATH,
        1,
    )
    control = controls[0]
    state = card.ocean.recipe.initial_state

    def objective(delta):
        atmosphere = control["s0f"].before
        moved = atmosphere._replace(u_wind=atmosphere.u_wind + delta)
        s0f = control["s0f"]._replace(before=moved, after=moved)
        result = bulk.nemo_orca2_k0_ocean_step(
            card,
            state,
            s0f=s0f,
            ice_if=control["ice_if"],
            s0=control["s0_base"],
            s2=control["s2_base"],
            chl=control["chl"],
            geothermal_heat_flux_mwm2=control["geothermal_heat_flux_mwm2"],
            sponge=control["sponge"],
            nemo_kt=1,
        )
        ocean = (
            result.ocean.state_after
            if hasattr(result.ocean, "state_after")
            else result.ocean
        )
        return jnp.sum(ocean.T.data)

    zero = jnp.asarray(0.0, dtype=jnp.float64)
    direction = jnp.asarray(1.0, dtype=jnp.float64)
    primal, forward = jax.jvp(objective, (zero,), (direction,))
    reverse = jax.grad(objective)(zero) * direction
    assert np.isfinite(np.asarray(primal))
    assert np.isfinite(np.asarray(forward))
    assert np.isfinite(np.asarray(reverse))
    assert float(np.abs(np.asarray(forward))) > 0.0
    np.testing.assert_allclose(reverse, forward, rtol=2.0e-12, atol=2.0e-12)
