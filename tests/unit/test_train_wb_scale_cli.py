"""Arg-parse + config tests for the WB scale-training entry (JAX-free, login-safe)."""
import importlib.util
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[2]
_ENTRY = _ROOT / "scripts" / "run" / "train_weatherbench_scale.py"
_spec = importlib.util.spec_from_file_location("train_wb_scale", _ENTRY)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)          # top-level must NOT import jax


def test_argparse_roundtrip():
    cfg = mod.build_scale_config_from_args(
        ["--mode", "neural_gcm", "--resolution", "0.7", "--epochs", "20",
         "--multi-step-hours", "6,12", "--eval-wb2"])
    assert cfg.mode == "neural_gcm"
    assert cfg.resolution_deg == 0.7
    assert cfg.n_epochs == 20
    assert cfg.multi_step_hours == (6, 12)
    assert cfg.eval_wb2 is True


def test_argparse_defaults_and_all_modes():
    cfg = mod.build_scale_config_from_args([])
    # --resolution defaults to None (#817 papercut fix): the grid comes from
    # the YAML; the value is derived for logging and an explicit mismatch is a
    # hard error (see test_resolution_yaml_check below).
    assert cfg.mode == "neural_gcm" and cfg.resolution_deg is None and cfg.grad_accum == 1
    assert cfg.training_core == "latlon"   # default core: byte-unchanged path
    for m in ("physics", "neural_gcm", "sfno"):
        assert mod.build_scale_config_from_args(["--mode", m]).mode == m


def test_argparse_rejects_bad_mode():
    with pytest.raises(SystemExit):
        mod.build_scale_config_from_args(["--mode", "bogus"])


def test_argparse_training_core_roundtrip_and_rejects_bad():
    """#817: --training-core selects the spectral (semi-implicit) training
    core; unknown values are rejected by argparse choices."""
    for core in ("latlon", "spectral"):
        cfg = mod.build_scale_config_from_args(["--training-core", core])
        assert cfg.training_core == core
    with pytest.raises(SystemExit):
        mod.build_scale_config_from_args(["--training-core", "bogus"])


def test_resolution_yaml_check():
    """#817 papercut: --resolution must MATCH the YAML grid or hard-error —
    the old flag silently logged one resolution while training at another."""
    yml = {"n_lat": 256, "n_lon": 512}
    # None (default) -> derived from the YAML.
    cfg = mod.build_scale_config_from_args([])
    assert abs(mod._check_resolution_matches_yaml(cfg, yml) - 180.0 / 256) < 1e-9
    # Matching explicit value passes.
    cfg = mod.build_scale_config_from_args(["--resolution", "0.703125"])
    assert mod._check_resolution_matches_yaml(cfg, yml) == 0.703125
    # Mismatch (2.8 deg vs a 0.7 deg YAML) -> SystemExit, not a silent no-op.
    cfg = mod.build_scale_config_from_args(["--resolution", "2.8"])
    with pytest.raises(SystemExit, match="does not match the YAML"):
        mod._check_resolution_matches_yaml(cfg, yml)


def test_config_yaml_loads():
    y = yaml.safe_load(open(_ROOT / "config" / "wb" / "scale" / "train_07deg.yaml"))
    assert y["grid"] == "latlon" and y["dt"] > 0
    assert {"w_T", "multi_step_hours"} <= set(y["loss"])
    assert y["loss"]["w_spec_crps_T"] == 0.0            # spectral-CRPS off at scale
    assert y["train_years"] and y["eval_years"] == [2020]


def test_n_days_cli_roundtrip_and_reject():
    """#1047 ask a: --n-days sets the training-window length; default None."""
    assert mod.build_scale_config_from_args([]).n_days is None
    assert mod.build_scale_config_from_args(["--n-days", "60"]).n_days == 60
    with pytest.raises(SystemExit, match="--n-days must be >= 1"):
        mod.build_scale_config_from_args(["--n-days", "0"])


def test_resolve_n_days_precedence():
    """CLI cfg.n_days > YAML n_training_days > 3; --smoke forces 1 (#1047)."""
    from legoesm.training.scale_build import _resolve_n_days

    base = mod.build_scale_config_from_args([])          # n_days=None, smoke=False
    assert _resolve_n_days(base, {}) == 3                # historical default
    assert _resolve_n_days(base, {"n_training_days": 60}) == 60   # YAML wins over 3
    cli = mod.build_scale_config_from_args(["--n-days", "10"])
    assert _resolve_n_days(cli, {"n_training_days": 60}) == 10    # CLI wins over YAML
    assert _resolve_n_days(base._replace(smoke=True), {"n_training_days": 60}) == 1
    with pytest.raises(ValueError, match="must be >= 1"):
        _resolve_n_days(base, {"n_training_days": 0})


def test_rollout_hours_matches_first_lead():
    """The training rollout horizon = the FIRST multi_step_hours lead — the same
    lead load_era5_samples uses to pick the target, so pred and target stay at
    the same forecast time (the old hardwired single_day_rollout scored a 24 h
    forecast against a 6 h target)."""
    from legoesm.training.scale_build import rollout_hours

    cfg = mod.build_scale_config_from_args(["--multi-step-hours", "12,24"])
    assert rollout_hours(cfg, {}) == 12.0
    # CLI default (6,12) -> 6 h
    cfg = mod.build_scale_config_from_args([])
    assert rollout_hours(cfg, {}) == 6.0
    # no CLI leads -> YAML loss.multi_step_hours wins; nothing at all -> 6 h
    cfg = cfg._replace(multi_step_hours=())
    assert rollout_hours(cfg, {"loss": {"multi_step_hours": [12, 24]}}) == 12.0
    assert rollout_hours(cfg, {}) == 6.0


def test_rrtmgp_cache_is_warmed_before_the_traced_loss():
    """The classical arm builds RRTMGP inside ``make_run_seg``, and ``loss_fn``
    calls that under ``eqx.filter_value_and_grad``.  With a cold optics cache
    the NetCDF gas-optics load then runs against Equinox tracers and the job
    dies with TracerArrayConversionError (job 26905933, six minutes of ERA5
    loading wasted first).  The entry point must therefore build the segment
    once with CONCRETE params, before the training loop.

    Inspects ``_main``, which HOLDS the body; the public ``main`` is a
    three-line wrapper that only applies the MPI-abort guard.
    """
    import ast

    tree = ast.parse(_ENTRY.read_text())
    main = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "_main")

    # It must be a bare statement in main's OWN body: a call nested in an inner
    # def is the traced one this guards against, and one wrapped in ``if
    # cfg.mode == ...`` or a swallowing ``try`` leaves the classical arm exactly
    # as broken as before (codex).
    warm = [stmt.value for stmt in main.body
            if isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Call)
            and isinstance(stmt.value.func, ast.Name)
            and stmt.value.func.id == "make_run_seg"]
    assert warm, ("_main() must call make_run_seg(...) as an unconditional "
                  "top-level statement — it warms the RRTMGP optics-table "
                  "cache outside the trace")
    assert any(isinstance(a, ast.Name) and a.id == "params"
               for call in warm for a in call.args), (
        "the warm-up must pass the concrete params pytree, not a placeholder")

    # ... and before the ERA5 load, so a broken physics config fails in seconds
    # rather than after minutes of data loading.
    era5 = [n.lineno for n in ast.walk(main)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id == "load_era5_samples"]
    assert era5, "ERA5 loader call not found — did main() get restructured?"
    assert min(c.lineno for c in warm) < min(era5), (
        "the warm-up must run BEFORE the ERA5 load, not after it")


def _guard():
    """The #1464 surface-stress guard, from the module it now lives in.

    It moved out of this driver into ``legoesm.training.scale_build`` because
    guarding one driver left the EVALUATION driver free to build the same
    unequalised arm and write a scorecard from it: both go through
    ``build_mode_components``, so that is where the check belongs.
    """
    from legoesm.training.scale_build import check_surface_drag_confound
    return check_surface_drag_confound


def _confounded_yaml(**extra):
    neural = {"surface_drag_confounded": "core_does_not_read_the_key"}
    neural.update(extra)
    return {"neural_gcm": neural, "classical": {"turbulence": "louis"}}


def test_the_spectral_core_refuses_a_config_that_declares_the_key_unread():
    """#1464: that declaration is a property of the CORE, not of the file.

    The lat-lon core never reads ``neural_gcm.surface_drag``; the spectral one
    does. Running a config that declares the key unread on the spectral core
    would hand the learned arm no surface stress while the classical arm it is
    scored against carries Louis -- the confound, wearing the label that says
    it is not there."""
    with pytest.raises(SystemExit) as e:
        _guard()(_confounded_yaml(), "neural_gcm", "spectral")
    assert "surface_drag" in str(e.value)


def test_the_latlon_core_accepts_the_same_config():
    """On the core the declaration is about, it is simply true."""
    assert _guard()(_confounded_yaml(), "neural_gcm", "latlon") is None


def test_asking_for_the_drag_clears_the_refusal():
    """A config that enables the drag is equalised, whatever it declares."""
    assert _guard()(
        _confounded_yaml(surface_drag=True, surface_drag_scheme="louis"),
        "neural_gcm", "spectral") is None


def test_the_guard_is_reached_through_the_builder_not_just_the_trainer():
    """The trainer is not the only door.

    ``run_weatherbench_eval`` builds the same components and writes a
    scorecard; a guard installed in the training entry point alone is walked
    straight past by it.  Both call ``build_mode_components``, so assert the
    check happens THERE."""
    import ast
    import inspect

    from legoesm.training import scale_build

    src = inspect.getsource(scale_build.build_mode_components)
    called = {n.func.id for n in ast.walk(ast.parse(src.lstrip()))
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "check_surface_drag_confound" in called, (
        "build_mode_components does not run the surface-stress guard, so any "
        "entry point that is not the trainer builds an unequalised learned "
        "arm without a word (#1464)")


def test_a_campaign_the_builder_cannot_equalise_is_named_not_waved_through():
    """The second declaration had no runtime consequence at all.

    ``builder_refuses_classical_scheme`` is legitimate -- the drag builder
    genuinely cannot reproduce a prognostic scheme -- but the run still
    produces a table whose arms differ by a momentum sink.  Silence there
    reads as an equalised comparison."""
    yml = {"neural_gcm": {
               "surface_drag_confounded": "builder_refuses_classical_scheme"},
           "classical": {"turbulence": "clubb"}}
    note = _guard()(yml, "neural_gcm", "spectral")
    assert note and "clubb" in note and "surface stress" in note, note
    # and it is silent once the arms ARE equalised
    yml["neural_gcm"]["surface_drag"] = True
    assert _guard()(yml, "neural_gcm", "spectral") is None


def test_the_scorecard_records_the_confound_beside_the_numbers():
    """A log line is not a record; the file the plots read has to carry it."""
    import ast
    import pathlib

    src = (pathlib.Path(__file__).resolve().parents[2] / "scripts" /
           "validate" / "run_weatherbench_eval.py").read_text()
    keys = {n.value for n in ast.walk(ast.parse(src))
            if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert "surface_drag_confound" in keys, (
        "the scorecard meta block does not record whether the learned arm "
        "carried a surface stress, so a confounded table is indistinguishable "
        "from an equalised one once the log scrolls away")


def _entry_ast():
    import ast
    return ast.parse(_ENTRY.read_text())


def test_the_entrypoint_aborts_the_whole_job_when_one_rank_dies():
    """Under MPI a rank that raises anywhere -- data load, the reachability
    probe, the training loop -- must MPI_Abort, or its peers block forever in
    the next collective.  ``main`` exists only to apply that guard."""
    import ast

    tree = _entry_ast()
    main = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    def _is_guard(node):
        return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "mpi_abort_on_uncaught"
                and any(isinstance(a, ast.Name) and a.id == "_main"
                        for a in node.args))

    # The guarded wrapper must be CALLED, not merely built: an unused
    # ``mpi_abort_on_uncaught(_main)`` guards nothing.
    invoked = [n for n in ast.walk(main)
               if isinstance(n, ast.Call) and _is_guard(n.func)]
    assert invoked, "main() must CALL mpi_abort_on_uncaught(_main)(...)"


def test_the_freeze_is_reached_only_through_the_mode_gate():
    """A neural model must never be frozen on a zero gradient (its decoder is
    zero-initialized, so every upstream weight is dead at step 0).  The gate is
    only worth anything if the call site actually sits behind it."""
    import ast

    tree = _entry_ast()
    main = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "_main")
    calls = [n for n in ast.walk(main)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "freeze_unreachable"]
    assert calls, "_main() must call freeze_unreachable"
    # Only the TRUE branch counts: a call in the `else` would run for exactly
    # the neural modes the gate exists to protect.
    in_true_branch = set()
    for stmt in ast.walk(main):
        if not isinstance(stmt, ast.If):
            continue
        if not any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                   and c.func.id == "_uses_reachability_freeze"
                   for c in ast.walk(stmt.test)):
            continue
        for body_stmt in stmt.body:
            in_true_branch.update(id(n) for n in ast.walk(body_stmt))
    assert all(id(c) in in_true_branch for c in calls), (
        "every freeze_unreachable(...) call must sit in the TRUE branch of an "
        "`if _uses_reachability_freeze(...)`")


def test_the_epoch_checkpoint_carries_the_frozen_parameters():
    """A frozen leaf lives in ``static``; a checkpoint written from the
    trainable half alone would silently drop 100+ parameters."""
    import ast

    tree = _entry_ast()
    main = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "_main")
    writes = [n for n in ast.walk(main)
              if isinstance(n, ast.Call)
              and isinstance(n.func, ast.Attribute)
              and n.func.attr == "tree_serialise_leaves"]
    assert writes, "_main() must write an epoch checkpoint"
    # Both halves must be named: the CURRENT trainable leaves the loop handed
    # back (serialising the stale outer `arr` would write epoch-0 values) and
    # the frozen half. The call sits inside the atomic-write lambda, so look
    # through the whole write expression rather than at its direct arguments.
    combines = [c for w in writes for c in ast.walk(w)
                if isinstance(c, ast.Call)
                and isinstance(c.func, ast.Attribute)
                and c.func.attr == "combine"
                and [x.id for x in c.args
                     if isinstance(x, ast.Name)] == ["cur_arr", "static"]]
    assert combines, ("the checkpoint must serialise "
                      "eqx.combine(cur_arr, static)")


def test_the_frozen_parameters_are_re_measured_after_training():
    """The freeze is decided on the untrained model; the run must say which
    frozen parameters became reachable once the live ones moved."""
    import ast

    tree = _entry_ast()
    main = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "_main")
    loop = [n.lineno for n in ast.walk(main)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id == "mpi_data_parallel_training_loop"]
    remeasure = [n.lineno for n in ast.walk(main)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id == "measure_leaf_reachability"]
    assert loop and remeasure, "_main() must train and then re-measure"
    assert max(remeasure) > max(loop), (
        "the re-measurement must run AFTER the training loop -- on the trained "
        "parameters, not the initial ones")


def test_resume_restores_the_optimizer_state_not_just_the_parameters():
    """AdamW's moments and the warmup-cosine step live in the optimizer state.
    Restoring parameters alone is a cold restart wearing a resumed run's name,
    and nothing in the log would say so."""
    import ast

    tree = _entry_ast()
    main = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "_main")
    reads = [n for n in ast.walk(main)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "tree_deserialise_leaves"]
    assert len(reads) >= 2, (
        "resume must deserialise BOTH the parameters and the optimizer state; "
        f"found {len(reads)} deserialise call(s)")
    # One of them must be fed the optimizer-state path (index 1 of the triple).
    opt = [r for r in reads
           if any(isinstance(sub, ast.Subscript)
                  and isinstance(sub.slice, ast.Constant) and sub.slice.value == 1
                  for a in r.args for sub in ast.walk(a))]
    assert opt, "no deserialise call reads the .opt.eqx path"


def test_every_epoch_writes_all_three_checkpoint_files():
    """Parameters, optimizer state and the frozen-leaf list. Any one missing
    makes the other two unresumable, and the resume path refuses rather than
    silently restarting cold."""
    import ast

    tree = _entry_ast()
    main = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "_main")
    on_epoch = next(n for n in ast.walk(main)
                    if isinstance(n, ast.FunctionDef) and n.name == "on_epoch")
    writes = [n for n in ast.walk(on_epoch)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
              and n.func.id == "_atomic_write"]
    assert len(writes) == 3, (
        f"on_epoch must write parameters, optimizer state and the frozen list "
        f"atomically; found {len(writes)} atomic write(s)")


def test_yaml_training_keys_resolved():
    """2026-08-23: YAML lr/optimizer were silently inert — the v2 campaign
    asked for lr 1.5e-3 and trained at the CLI default 3e-4 (manifest proof).
    CLI wins when given; YAML wins over the historical defaults."""
    yml = {"lr": 1.5e-3, "optimizer": "muon",
           "weight_decay": 2.0e-5, "grad_clip_norm": 0.5}
    cfg = mod.build_scale_config_from_args([])
    assert cfg.lr is None and cfg.optimizer is None   # CLI silent
    cfg2, wd, clip = mod._resolve_training_keys(cfg, yml)
    assert cfg2.lr == 1.5e-3 and cfg2.optimizer == "muon"
    assert wd == 2.0e-5 and clip == 0.5
    # CLI overrides YAML.
    cfg = mod.build_scale_config_from_args(["--lr", "7e-4", "--optimizer", "adam"])
    cfg2, _, _ = mod._resolve_training_keys(cfg, yml)
    assert cfg2.lr == 7e-4 and cfg2.optimizer == "adam"
    # Neither given -> historical defaults.
    cfg2, wd, clip = mod._resolve_training_keys(
        mod.build_scale_config_from_args([]), {})
    assert cfg2.lr == 3.0e-4 and cfg2.optimizer == "adamw"
    assert wd is None and clip is None                # caller uses TrainingConfig


def test_yaml_grad_accum_is_a_hard_error():
    """grad_accum has no implementation here; a value other than 1 must
    refuse to run rather than be silently ignored."""
    cfg = mod.build_scale_config_from_args([])
    with pytest.raises(SystemExit, match="grad.accum"):
        mod._resolve_training_keys(cfg, {"grad_accum": 4})
    with pytest.raises(SystemExit, match="grad.accum"):
        mod._resolve_training_keys(
            mod.build_scale_config_from_args(["--grad-accum", "2"]), {})


def test_yaml_null_training_keys_fall_back():
    """`lr:` with no value parses as None; it must fall back, not crash or
    become the string "None" (GLM diff-review finding)."""
    cfg = mod.build_scale_config_from_args([])
    cfg2, wd, clip = mod._resolve_training_keys(
        cfg, {"lr": None, "optimizer": None, "n_epochs": None,
              "grad_accum": None, "weight_decay": None,
              "grad_clip_norm": None})
    assert cfg2.lr == mod._DEFAULT_LR
    assert cfg2.optimizer == mod._DEFAULT_OPTIMIZER
    assert cfg2.n_epochs == mod._DEFAULT_N_EPOCHS
    assert wd is None and clip is None


def test_cli_grad_accum_hard_errors_even_with_yaml_1():
    """codex P1: --grad-accum 2 with YAML grad_accum: 1 must NOT silently
    proceed as 1 — both sources hard-error on != 1."""
    with pytest.raises(SystemExit, match="grad.accum"):
        mod._resolve_training_keys(
            mod.build_scale_config_from_args(["--grad-accum", "2"]),
            {"grad_accum": 1})


def test_yaml_non_integral_epochs_rejected():
    """codex P2: n_epochs: 1.9 must not silently truncate to one epoch."""
    cfg = mod.build_scale_config_from_args([])
    with pytest.raises(SystemExit, match="not an integer"):
        mod._resolve_training_keys(cfg, {"n_epochs": 1.9})
    with pytest.raises(SystemExit, match="not an integer"):
        mod._resolve_training_keys(cfg, {"n_epochs": True})
    cfg2, _, _ = mod._resolve_training_keys(cfg, {"n_epochs": 12.0})
    assert cfg2.n_epochs == 12


def test_warmup_steps_cli_roundtrip_and_deck_default():
    """--warmup-steps lets one campaign arm depart from a shared deck.

    The flux A/B trained both arms at a warmup the deck did not carry; editing
    the deck instead would have moved the column-MLP and SFNO arms, which read
    the same file.
    """
    assert mod.build_scale_config_from_args([]).warmup_steps is None
    assert mod.build_scale_config_from_args(
        ["--warmup-steps", "200"]).warmup_steps == 200


def test_a_link_that_changes_the_warmup_refuses_the_previous_state():
    """Two links of one chain that disagree must not share optimizer state.

    The resolved warmup is part of the resume fingerprint, so an arm whose
    launcher passes a different --warmup-steps cannot silently continue on
    Adam moments built for another schedule.  This is the invariant the
    launcher's "every link passes the same values" note depends on.
    """
    cfg = mod.build_scale_config_from_args(
        ["--mode", "physics", "--training-core", "spectral", "--epochs", "4",
         "--lr", "1.5e-3", "--optimizer", "adamw"])
    yml = {"n_lat": 96, "n_lon": 192, "nlev": 32, "warmup_steps": 500}
    deck = mod._run_fingerprint(cfg, yml, 500, 12, 240, 1)
    override = mod._run_fingerprint(cfg, yml, 200, 12, 240, 1)
    assert deck["warmup_steps"] == 500 and override["warmup_steps"] == 200
    assert deck != override
    # ...and nothing else moved: the deck content is identical, so a chain
    # that keeps its values keeps its state.
    assert deck["config_sha256"] == override["config_sha256"]
    assert mod._run_fingerprint(cfg, yml, 500, 12, 240, 1) == deck


def test_only_the_first_argument_selects_the_signature_query():
    """A value that merely EQUALS the query token must not skip training.

    The pre-parser scanned every argument, so a wrapper whose CONFIG or OUT
    held this token printed a signature and exited 0 having trained nothing —
    and an afterany chain then repeated that no-op for every link.
    """
    tok = "--print-latest-complete"
    with pytest.raises(SystemExit) as e:
        # The token sits where CONFIG's value goes, with a following argument
        # for the old code to mistake for a directory. Reaching argparse's
        # rejection of the bad mode proves the pre-parser declined to claim
        # this invocation; the old scan printed a signature and returned 0.
        mod.main(["--config", tok, "--mode", "bogus"])
    assert e.value.code != 0
