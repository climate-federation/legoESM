"""Bitwise regression pin of the SI3 ORCA2 lane functions on ``main``.

THIS IS A SELF-REFERENCED REGRESSION PIN, NOT AN ORACLE.  The committed
outputs in ``tests/ice/fixtures/si3_orca2_regression/`` are what the lane
computed on deterministic synthetic ORCA2-shaped inputs when the fixture was
generated (``provenance.json``); they say nothing about agreement with NEMO.
The test recomputes every rung from the stored inputs and requires every
output leaf to be bit-identical (floats compared through same-width unsigned views,
so ``-0.0``/``+0.0`` and NaN payloads also count).  A change that moves any
bit fails here; if it was intended, regenerate with ``make_fixture.py`` and
say so in the PR.

Pinned: ``si3_lbc_lnk_orca2`` (T/U/V/F, both signs, both ``ldfull``, 2-D and
4-D), ``si3_apply_state_halos_orca2``, ``si3_o4_step`` (icycle 1, 2 and a
delayed-CFL icycle-3 case), ``si3_orca2_ice_strength`` and
``si3_orca2_aevp_solver``/``_trace`` (both fold programs, both
``differentiable`` arms), ``si3_dom_uniq_t_orca2``, ``si3_o15_step``,
``si3_o16_step``, ``sea_ice._si3_orca2_o2_step_with_trace``, and the full O17
body's live qns/qsr wiring and JIT-traced Prather path.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.ice import si3_o17 as si3_o17_module  # noqa: E402
from legoesm.ice import si3_orca2_state as si3_orca2_state_module  # noqa: E402
from legoesm.ice.config import SI3_HARD_SWITCHES  # noqa: E402
from legoesm.ice.si3_o17 import (  # noqa: E402
    si3_o17_step,
    si3_o17_through_update_flx,
)
from legoesm.ice.si3_orca2_state import si3_dyn_orca2  # noqa: E402

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "si3_orca2_regression"


def _load_generator():
    spec = importlib.util.spec_from_file_location("si3_orca2_make_fixture", FIXTURE / "make_fixture.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GEN = _load_generator()


PROVENANCE = (FIXTURE / "provenance.json").read_text()


_bits = GEN._bits

_MAIN_EQUIVALENT_O17_HARD_DIGEST = (
    "d5f53f53e308321652d08e135bf8b0334b6fdcd3daaed7614e19fd69a6e4089f"
)


def _assert_bitwise(expected: dict, actual: dict) -> None:
    where = f"\nfixture provenance: {PROVENANCE}"
    assert sorted(actual) == sorted(expected), "output leaf set changed" + where
    for key in expected:
        exp, got = np.asarray(expected[key]), np.asarray(actual[key])
        assert got.dtype == exp.dtype and got.shape == exp.shape, key + where
        np.testing.assert_array_equal(_bits(got), _bits(exp), err_msg=key + where)


def _assert_tree_bitwise(expected, actual) -> None:
    assert jax.tree_util.tree_structure(actual) == jax.tree_util.tree_structure(expected)
    for expected_leaf, actual_leaf in zip(
        jax.tree_util.tree_leaves(expected),
        jax.tree_util.tree_leaves(actual),
        strict=True,
    ):
        np.testing.assert_array_equal(_bits(actual_leaf), _bits(expected_leaf))


def _tree_digest(tree) -> str:
    digest = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(tree):
        digest.update(np.asarray(leaf).tobytes())
    return digest.hexdigest()


def _load(rung: str) -> tuple[dict, dict]:
    with np.load(FIXTURE / f"{rung}.npz") as z:
        inputs = {k[3:]: z[k] for k in z.files if k.startswith("in/")}
        outputs = {k[4:]: z[k] for k in z.files if k.startswith("out/")}
    return inputs, outputs


def test_generator_blob_provenance_is_current() -> None:
    payload = (FIXTURE / "make_fixture.py").read_bytes()
    git_blob = hashlib.sha1(
        f"blob {len(payload)}\0".encode() + payload,
        usedforsecurity=False,
    ).hexdigest()
    assert json.loads(PROVENANCE)["generator_blob_sha"] == git_blob


@pytest.fixture
def pin_policy():
    if jax.default_backend() != "cpu":
        pytest.skip(
            f"bitwise pin runs under the CPU-only libm policy; backend={jax.default_backend()}"
        )
    with GEN.set_pin_policy():
        yield


@pytest.mark.parametrize("rung", sorted(GEN.RUNGS))
def test_rung_is_bitwise_identical_to_committed_snapshot(rung: str, pin_policy) -> None:
    inputs, expected = _load(rung)
    _assert_bitwise(expected, GEN.RUNGS[rung](inputs))


@pytest.mark.parametrize("rung", sorted(GEN.RUNGS))
def test_one_ulp_change_in_a_stored_output_trips_the_pin(rung: str) -> None:
    """Non-vacuity: the comparison itself can fail on a single-ulp change."""
    _, expected = _load(rung)
    key = next(k for k in sorted(expected) if np.asarray(expected[k]).dtype == np.float64)
    planted = dict(expected)
    arr = np.array(planted[key], copy=True)
    flat = arr.reshape(-1)
    flat[0] = np.nextafter(flat[0], np.inf)
    planted[key] = arr
    with pytest.raises(AssertionError):
        _assert_bitwise(expected, planted)


def test_o17_numerical_body_pins_live_qns_ice_and_qsr(pin_policy) -> None:
    entry, forcing, card = GEN.build_o17_case()
    core = si3_o17_through_update_flx(entry, forcing, card)

    np.testing.assert_array_equal(
        _bits(core.update_forcing.qsr), _bits(forcing.surface.qsr)
    )
    raw_qns = np.asarray(core.surface_stress.blk_ice_2.qns_ice)
    thd_qns = np.asarray(jnp.stack(
        [category.o10.o9.zdf.qns_ice for category in core.thermodynamics.categories],
        axis=-1,
    ))[GEN.NN_HLS:-GEN.NN_HLS, GEN.NN_HLS:-GEN.NN_HLS]
    actual_qns = np.asarray(core.update_forcing.qns_ice)
    from_raw = _bits(actual_qns) == _bits(raw_qns)
    from_thermodynamics = _bits(actual_qns) == _bits(thd_qns)
    assert np.all(from_raw | from_thermodynamics)
    assert np.any(from_thermodynamics & ~from_raw)


def test_o17_step_executes_under_jit_and_matches_eager_main_pin(pin_policy) -> None:
    entry, forcing, card = GEN.build_o17_case()
    card = card._replace(surface=card.surface._replace(switches=SI3_HARD_SWITCHES))
    kt = jnp.asarray(1, dtype=jnp.int64)
    forcing = forcing._replace(kt=kt)
    eager = si3_o17_step(entry, forcing, card)
    compiled = jax.jit(
        lambda traced_kt: si3_o17_step(
            entry, forcing._replace(kt=traced_kt), card
        )
    )(kt)

    assert len(jax.tree_util.tree_leaves(compiled)) == 1209
    _assert_tree_bitwise(eager, compiled)
    assert _tree_digest(compiled) == _MAIN_EQUIVALENT_O17_HARD_DIGEST


def test_prather_tracer_dispatch_explicitly_includes_kt_parity() -> None:
    source = "".join(
        inspect.getsource(
            si3_orca2_state_module._si3_pra_subcycle_cond_dispatch
        ).split()
    )
    assert "tree_leaves((init_ops,icycle_arr,kt_parity))" in source


def test_o17_coupled_boundary_accepts_nn_fsbc_2(pin_policy) -> None:
    entry, forcing, card = GEN.build_o17_case()
    result = si3_o17_through_update_flx(
        entry,
        forcing,
        card._replace(state=card.state._replace(nn_fsbc=2)),
    )
    assert result.state.ice.a_i.shape == entry.ice.a_i.shape


def test_traced_prather_kt_preserves_wide_parity_and_refuses_zero(
    pin_policy, monkeypatch
) -> None:
    entry, forcing, card = GEN.build_o17_case()
    card = card._replace(state=card.state._replace(nn_fsbc=3))
    captured = {}
    original = si3_o17_module.si3_dyn_orca2

    def capture(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return original(*args, **kwargs)

    monkeypatch.setattr(si3_o17_module, "si3_dyn_orca2", capture)
    si3_o17_through_update_flx(entry, forcing, card)
    monkeypatch.setattr(si3_o17_module, "si3_dyn_orca2", original)

    def run(kt):
        kwargs = dict(captured["kwargs"])
        kwargs["kt"] = kt
        return si3_dyn_orca2(*captured["args"], **kwargs)

    compiled = jax.jit(run)
    reference = compiled(jnp.asarray(1, dtype=jnp.int64))
    wide = compiled(jnp.asarray(2**31 + 1, dtype=jnp.int64))
    _assert_tree_bitwise(reference, wide)
    with pytest.raises(
        (ValueError, jax.errors.JaxRuntimeError), match="requires kt >= 1"
    ):
        compiled(jnp.asarray(0, dtype=jnp.int64))


@pytest.mark.parametrize("kt", (jnp.asarray(True), jnp.asarray(1.5)))
def test_o17_prather_trace_refuses_non_integer_kt(kt, pin_policy) -> None:
    entry, forcing, card = GEN.build_o17_case()
    with pytest.raises(ValueError, match="kt >= 1"):
        jax.eval_shape(
            lambda e, f: si3_o17_step(e, f, card),
            entry,
            forcing._replace(kt=kt),
        )


def test_o17_prather_refuses_concrete_zero_kt(pin_policy) -> None:
    entry, forcing, card = GEN.build_o17_case(kt=0)
    with pytest.raises(ValueError, match="kt >= 1"):
        si3_o17_step(entry, forcing, card)
