"""SSS CLI defaults, legacy config parity, and both live forwarding paths.

Like test_run_omip_core2_preclosure_coeff_cli, inspect named forwarding hops
without loading an ocean mesh. Execute the actual constructor calls as well:
builder-only tests cannot detect a parsed flag discarded by either caller.
"""
from __future__ import annotations

import ast
import copy
import inspect
from types import SimpleNamespace

import pytest


def _core2():
    import scripts.run.run_omip_core2 as core2
    return core2


@pytest.fixture(scope="module")
def tree():
    return ast.parse(inspect.getsource(_core2()))


def _calls(tree, caller, callee):
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == caller)
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == callee]
    assert calls, f"{caller} never calls {callee}"
    return calls


def _eval(node, **namespace):
    return eval(compile(ast.Expression(node), "<SSS caller>", "eval"), namespace)


def test_parser_and_config_defaults_are_unchanged():
    from legoesm.ocean.forcing.sss_restoring import SSSRestoringConfig

    args = _core2()._build_arg_parser().parse_args([])
    assert args.sss_restore is False
    assert args.sss_restore_tau_days == 365.0
    assert args.sss_restore_regions is None
    assert args.sss_restore_normalization is None
    assert args.sss_restore_bound_mmday is None
    assert args.sss_restore_channel is None
    assert args.sss_restore_file is None
    assert args.sss_ice_gate_nemo is False
    assert args.river_mouth_restoring_gate is False
    assert _core2().build_sss_restoring_config(
        sss_restore_tau_days=args.sss_restore_tau_days
    ) == SSSRestoringConfig(enabled=True)


@pytest.mark.parametrize("regions", ["uniform", "omip2"])
def test_region_choice_reaches_config(regions):
    from legoesm.ocean.forcing.sss_restoring import DEFAULT_OMIP2_REGIONS

    args = _core2()._build_arg_parser().parse_args(
        ["--sss-restore-regions", regions])
    cfg = _core2().build_sss_restoring_config(
        sss_restore_tau_days=45.5, sss_restore_regions=args.sss_restore_regions)
    assert DEFAULT_OMIP2_REGIONS  # The uniform-vs-regional check must discriminate.
    assert cfg.regions == (() if regions == "uniform" else DEFAULT_OMIP2_REGIONS)
    assert cfg.tau_restore_days_default == 45.5


@pytest.mark.parametrize("flag", ["sss_restore_regions", "sss_restore_normalization"])
def test_unknown_builder_choice_raises(flag):
    with pytest.raises(ValueError, match=flag.replace("_", "-")):
        _core2().build_sss_restoring_config(
            sss_restore_tau_days=365.0, **{flag: "typo"})


@pytest.mark.parametrize("flag", ["--sss-restore-regions", "--sss-restore-normalization"])
def test_unknown_parser_choice_raises(flag):
    with pytest.raises(SystemExit) as exc:
        _core2()._build_arg_parser().parse_args([flag, "typo"])
    assert exc.value.code == 2


@pytest.mark.parametrize("flag", ["sss_restore_tau_days", "sss_restore_bound_mmday"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 0.0, -1.0])
def test_numeric_guards_exit_cleanly(flag, value):
    kwargs = {"sss_restore_tau_days": 365.0, flag: value}
    with pytest.raises(SystemExit, match=flag.replace("_", "-") + " must be finite and > 0"):
        _core2().build_sss_restoring_config(**kwargs)


# Frozen pre-refactor contract: at e22b975e4 both constructors passed enabled
# and tau, optionally normalization, nemo_linear, and the converted flux bound.
# main used 86400.0; run_fesom_forced_loop used _SEC_PER_DAY. Compare the entire
# config, not just the newly selectable regions. Do not call the shared builder
# to construct the expected result.
@pytest.mark.parametrize("caller", ["main", "run_fesom_forced_loop"])
@pytest.mark.parametrize("tau", [365.0, 45.5])
@pytest.mark.parametrize("normalization", [None, "s_target", "live_s"])
@pytest.mark.parametrize("bound", [None, 4.0, 17.25])
@pytest.mark.parametrize("ice", [False, True])
@pytest.mark.parametrize("regions", [None, "omip2"])
def test_actual_call_matches_pre_refactor_config(
    tree, caller, tau, normalization, bound, ice, regions,
):
    from legoesm import constants
    from legoesm.ocean.forcing.sss_restoring import SSSRestoringConfig

    legacy_kwargs = {}
    if ice:
        legacy_kwargs["ice_gate_mode"] = "nemo_linear"
    if normalization is not None:
        legacy_kwargs["normalization"] = normalization
    if bound is not None:
        seconds = 86400.0 if caller == "main" else _core2()._SEC_PER_DAY
        legacy_kwargs["max_flux_kg_m2_s"] = (
            float(bound) * 1.0e-3 / seconds * float(constants.rho_water))
    expected = SSSRestoringConfig(
        enabled=True, tau_restore_days_default=float(tau), **legacy_kwargs)
    args = SimpleNamespace(
        sss_restore_tau_days=tau, sss_restore_normalization=normalization,
        sss_restore_bound_mmday=bound, sss_ice_gate_nemo=ice,
        sss_restore_regions=regions)
    for call in _calls(tree, caller, "build_sss_restoring_config"):
        actual = _eval(call, args=args,
                       build_sss_restoring_config=_core2().build_sss_restoring_config)
        assert actual == expected


_BUILDER_FLAGS = (
    "sss_restore_tau_days", "sss_restore_normalization", "sss_restore_bound_mmday",
    "sss_ice_gate_nemo", "sss_restore_regions",
)


def _assert_forwarded(call, flag):
    kws = [k for k in call.keywords if k.arg == flag]
    assert len(kws) == 1, f"missing named forwarding for {flag}"
    assert ast.dump(kws[0].value) == ast.dump(ast.parse(f"args.{flag}", mode="eval").body)


@pytest.mark.parametrize("caller", ["main", "run_fesom_forced_loop"])
@pytest.mark.parametrize("flag", _BUILDER_FLAGS)
def test_each_caller_forwards_each_parsed_flag_by_name(tree, caller, flag):
    for call in _calls(tree, caller, "build_sss_restoring_config"):
        _assert_forwarded(call, flag)


@pytest.mark.parametrize("caller", ["main", "run_fesom_forced_loop"])
def test_uniform_config_from_each_actual_call(tree, caller):
    args = _core2()._build_arg_parser().parse_args([
        "--sss-restore", "--sss-restore-regions", "uniform",
        "--sss-restore-tau-days", "45.5", "--sss-restore-normalization", "live_s",
        "--sss-restore-bound-mmday", "4.0", "--sss-ice-gate-nemo",
    ])
    from legoesm import constants
    from legoesm.ocean.forcing.sss_restoring import SSSRestoringConfig

    expected = SSSRestoringConfig(
        enabled=True, tau_restore_days_default=45.5, regions=(),
        normalization="live_s", ice_gate_mode="nemo_linear",
        max_flux_kg_m2_s=4.0 * 1.0e-3 / 86400.0 * float(constants.rho_water))
    for call in _calls(tree, caller, "build_sss_restoring_config"):
        assert _eval(call, args=args,
                     build_sss_restoring_config=_core2().build_sss_restoring_config) == expected


@pytest.mark.parametrize("mutation", ["drop", "hardcode", "wrong_attribute"])
@pytest.mark.parametrize("flag", _BUILDER_FLAGS)
def test_forwarding_check_rejects_planted_discard(tree, mutation, flag):
    call = copy.deepcopy(_calls(tree, "main", "build_sss_restoring_config")[0])
    kw = next(k for k in call.keywords if k.arg == flag)
    if mutation == "drop":
        call.keywords.remove(kw)
    else:
        kw.value = (ast.Constant(None) if mutation == "hardcode"
                    else ast.parse("args.unrelated", mode="eval").body)
    with pytest.raises(AssertionError):
        _assert_forwarded(call, flag)


@pytest.mark.parametrize("argv,enabled", [
    ([], False), (["--river-mouth-restoring-gate"], True),
    (["--no-river-mouth-restoring-gate"], False),
    (["--river-mouth-restoring-gate", "--no-river-mouth-restoring-gate"], False),
    (["--no-river-mouth-restoring-gate", "--river-mouth-restoring-gate"], True),
])
@pytest.mark.parametrize("caller,callee", [
    ("run_fesom_forced_loop", "compute_sss_restoring_flux"),
    ("run_fesom_forced_loop", "apply_sss_restoring_step_fesom"),
    ("main", "_sss_flux_fn"),
    ("main", "apply_sss_restoring_step"),
    ("main", "apply_sss_restoring_step_mpas"),
])
def test_mouth_flag_reaches_every_consumer_by_name(tree, argv, enabled, caller, callee):
    import numpy as np

    args = _core2()._build_arg_parser().parse_args(argv)
    assert args.river_mouth_restoring_gate is enabled
    runoff = np.array([2.0, 3.0])  # Nonzero sentinel: a discarded gate must fail.
    namespace = dict(args=args, _R=runoff, _S_now=runoff, jnp=np)
    # Every lane that hoists the gate into a local `_R_gate` must do it with
    # exactly ONE conditional assignment, and that assignment is what the call
    # sites below are evaluated against.  Resolving it per CALLER (rather than
    # only for `main`) is what makes the fesom loop's hoisted form testable:
    # a second, divergent `_R_gate` assignment in the same function would fail
    # the count, and a call site that stopped forwarding it would fail below.
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == caller)
    assignments = [n for n in ast.walk(fn) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "_R_gate"
                           for t in n.targets)
                   and isinstance(n.value, ast.IfExp)]
    if assignments:
        assert len(assignments) == 1
        namespace["_R_gate"] = _eval(assignments[0].value, **namespace)
    for call in _calls(tree, caller, callee):
        kws = [k for k in call.keywords if k.arg == "river_runoff"]
        assert len(kws) == 1, f"{caller} -> {callee} discarded river_runoff"
        value = _eval(kws[0].value, **namespace)
        if enabled:
            np.testing.assert_array_equal(value, runoff)
        else:
            assert value is None


def test_help_declares_mouth_opt_out():
    assert "--no-river-mouth-restoring-gate" in _core2()._build_arg_parser().format_help()


def test_regions_without_restoring_is_rejected_before_model_build(monkeypatch):
    import sys

    monkeypatch.setattr(sys, "argv", ["run_omip_core2.py", "--sss-restore-regions", "uniform"])
    with pytest.raises(ValueError, match="--sss-restore-regions requires --sss-restore"):
        _core2().main()
