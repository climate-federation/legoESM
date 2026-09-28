#!/usr/bin/env python
"""WeatherBench scale training (Derecho / Levante), data-parallel across ranks.

Multi-node data-parallel: one MPI rank per GPU, each rank trains the SAME
replicated model on a distinct ERA5 shard; gradients are averaged across ranks
each step via ``legoesm.training.data_parallel`` (mpi4jax ``global_sum_mpi``).
Supports all three modes (``--mode physics|neural_gcm|sfno``) at 0.7 deg lat-lon.

Launch: ``scripts/cluster/{derecho/train_wb.pbs, levante/train_wb.slurm}``.
Single-process (no MPI) also works (num_processes == 1 -> the serial path), for a
single-GPU smoke.

The arg-parse / config layer (``build_scale_config_from_args``) is import-light
and JAX-free (unit-tested on the login node); all heavy imports live inside
``main`` so ``--help`` and the CLI test do not pull in JAX.
"""
from __future__ import annotations

import argparse
import os as _os
from typing import NamedTuple

VALID_MODES = ("physics", "neural_gcm", "sfno")
# Historical CLI defaults, now the LAST fallback behind the campaign YAML
# (single source: parser help, resolver and tests all read these).
_DEFAULT_LR = 3.0e-4
_DEFAULT_OPTIMIZER = "adamw"
_DEFAULT_N_EPOCHS = 40
VALID_TRAINING_CORES = ("latlon", "spectral")


class ScaleConfig(NamedTuple):
    mode: str
    config_path: str
    resolution_deg: float
    n_epochs: int
    multi_step_hours: tuple
    lr: float
    optimizer: str
    grad_accum: int
    out_dir: str
    resume: bool
    eval_wb2: bool
    smoke: bool
    # #817 blocker 1: 'latlon' = the explicit C-grid production core;
    # 'spectral' = the Gaussian semi-implicit training core whose implicit
    # gravity-wave treatment keeps the training adjoint bounded (and whose
    # grid has no pole-cell CFL clamp).  Appended AFTER the original fields
    # so positional construction in existing tests stays valid.
    training_core: str = "latlon"
    # #1047 ask a: training-window length [days] per train year. None = fall
    # back to YAML ``n_training_days`` then 3 (the historical hardcoded value
    # that data-starved the column MLP). --smoke still forces 1. Appended last
    # so positional construction in existing tests stays valid.
    n_days: int | None = None
    # None = take the deck's ``warmup_steps``. An explicit value lets one
    # campaign arm depart from a deck its sibling arms share. Appended last so
    # positional construction in existing tests stays valid.
    warmup_steps: int | None = None
    # Held-out windows scored after every epoch (0 = no validation pass).
    n_val_windows: int = 0


# Scenes used for the POST-TRAINING reachability report.  A subsample, not the
# whole shard, because this one only REPORTS -- the freeze decision itself
# reads every sample (see the block in ``_main``), so no sampling question
# touches what the run trains.  Spread evenly rather than taken from the front,
# because consecutive WeatherBench samples are six hours apart and are nearly
# the same weather.  How much of the year that spread covers is the config's
# business, not this constant's: the campaign deck lists 60 one-day windows at
# twelve month-starts, while config/wb/scale/train_07deg.yaml leaves its
# train_windows commented out and gets three days of one January.
_N_REACHABILITY_REPORT = 24


def _checkpoint_paths(out_dir, epoch):
    """The three files one epoch's checkpoint is made of.

    Parameters alone are not a resumable state.  AdamW carries a first and
    second moment per leaf and the step count that drives the warmup-cosine
    schedule, so a run restarted from parameters only restarts the optimizer
    COLD and the learning rate back at warmup — it looks like a resume and
    silently is not.  The frozen-leaf list is here for the same reason: the
    trainable set is decided by a measurement on the initial parameters, and
    re-deriving it on resumed parameters could pick a DIFFERENT set, which the
    saved optimizer state would no longer match.
    """
    import os

    base = os.path.join(out_dir, f"epoch_{epoch:04d}")
    return base + ".eqx", base + ".opt.eqx", base + ".frozen.json"


def _manifest_is_valid(obj, expect_keys=None):
    """Whether a checkpoint manifest is one this trainer would resume from.

    Structure only unless ``expect_keys`` is given: ``schema`` an int that is
    not a bool (``True == 1`` in Python), ``frozen`` a list of names, and a
    fingerprint mapping.  With ``expect_keys`` the fingerprint's key set must
    match exactly, which is the check the resume path adds once it knows what
    the current run looks like.
    """
    if not isinstance(obj, dict):
        return False
    schema = obj.get("schema")
    if isinstance(schema, bool) or not isinstance(schema, int):
        return False
    if schema != _MANIFEST_SCHEMA:
        return False
    frozen = obj.get("frozen")
    if not isinstance(frozen, list) or not all(isinstance(x, str) for x in frozen):
        return False
    fp = obj.get("fingerprint")
    if not isinstance(fp, dict):
        return False
    return expect_keys is None or set(fp) == set(expect_keys)


def _read_manifest(path):
    """The parsed manifest, or ``None`` if it is absent, truncated or invalid."""
    import json as _json

    try:
        with open(path) as fh:
            obj = _json.load(fh)
    except (OSError, ValueError):
        return None
    return obj if _manifest_is_valid(obj) else None


def _latest_complete_checkpoint(out_dir):
    """The newest epoch whose parameters, optimizer state AND frozen list all
    exist, or ``None``.

    Walking DOWN rather than refusing on the newest is what keeps a chained run
    alive.  A job killed between the three writes leaves the newest epoch
    incomplete, and older runs wrote parameters only — refusing outright would
    make every resumed link fail while the self-chaining wrapper resubmits it,
    forever.  Stepping back costs one epoch of recompute and always terminates.
    """
    import os
    import re

    if not os.path.isdir(out_dir):
        return None
    try:
        names = os.listdir(out_dir)
    except OSError:
        return None
    epochs = sorted(
        (int(m.group(1)) for m in
         (re.fullmatch(r"epoch_(\d+)\.eqx", f) for f in names)
         if m is not None),
        reverse=True)
    for ep in epochs:
        pp, op, fp = _checkpoint_paths(out_dir, ep)
        if (os.path.exists(pp) and os.path.exists(op)
                and _read_manifest(fp) is not None):
            return ep
    return None


def print_latest_complete_signature(root):
    """Print a change signature for the newest COMPLETE epoch under ``root``.

    The self-chaining job wrapper calls this instead of re-implementing what a
    complete epoch is.  It used to grep the manifest for a substring, which
    counted a truncated file as progress and missed a differently formatted
    one — so the wrapper would chain a link the trainer then refused, and the
    pair looped while reporting a chained run.

    ``root`` and its immediate subdirectories are searched, because a campaign
    gives each family its own directory under one output root.  Prints nothing
    when there is no complete epoch, so an empty signature means "no progress".
    """
    import os

    # Never raises: the caller is a shell that treats an empty answer as "no
    # progress", so an unreadable directory must look like nothing rather than
    # like a crash it would then have to interpret.
    try:
        children = sorted(os.listdir(root)) if os.path.isdir(root) else []
    except OSError:
        return
    best = None
    for d in [root] + [os.path.join(root, x) for x in children]:
        try:
            if not os.path.isdir(d):
                continue
            ep = _latest_complete_checkpoint(d)
            if ep is None:
                continue
            st = os.stat(_checkpoint_paths(d, ep)[2])
        except OSError:
            continue
        path = _checkpoint_paths(d, ep)[2]
        cand = (st.st_mtime_ns, f"{path}:{st.st_mtime_ns}:{st.st_size}")
        if best is None or cand[0] > best[0]:
            best = cand
    if best is None:
        # A persisted reachability-probe result IS resumable progress: the
        # next link restores it and skips the ~9 h probe (see
        # _read_probe_freeze). Without this, a link that finished only the
        # probe would trip the wrapper's epoch-0 restart-loop guard and the
        # chain would stop exactly where resuming has the most to save.
        for d in [root] + [os.path.join(root, x) for x in children]:
            try:
                st = os.stat(_probe_freeze_path(d))
            except OSError:
                continue
            cand = (st.st_mtime_ns,
                    f"{_probe_freeze_path(d)}:{st.st_mtime_ns}:{st.st_size}")
            if best is None or cand[0] > best[0]:
                best = cand
    if best is not None:
        print(best[1])


# Schema 2, 2026-08-26: the terrain-reconciliation fix changed the CONTENT of
# every ingested sample while leaving the config fingerprint unchanged (the
# ingest transform is code, not YAML) — a schema-1 checkpoint or probe result
# was measured on pre-fix data and must not resume against post-fix samples.
_MANIFEST_SCHEMA = 2


def _run_fingerprint(cfg, yml, warmup, roll_steps, n_global_samples, nproc):
    """What the restored optimizer state was built for.

    The Equinox templates only check the SHAPE of the tree, so a resume with a
    different learning rate, optimizer, epoch count, sample set or rank count
    loads happily and then runs the restored Adam moments and step count under
    a schedule they were never part of — a continuation that looks clean and is
    a different experiment.

    The experiment is identified by the CONTENT of the resolved YAML, not by
    its path: editing a value in the same file changes the run while leaving a
    pathname identical, and ``config/x.yaml`` versus ``./config/x.yaml`` is the
    same run under two names.  The resolved warmup and rollout length are
    carried explicitly because they reach the schedule and the loss without
    necessarily changing anything else here.
    """
    import hashlib
    import json as _json

    from legoesm.ml.training import TrainingConfig

    def _canon(o):
        # A bare ``str`` fallback is not canonical: a YAML ``!!set`` renders in
        # iteration order, which moves with PYTHONHASHSEED, so the same config
        # would hash differently between two jobs.
        if isinstance(o, (set, frozenset)):
            return sorted(map(str, o))
        return str(o)

    digest = hashlib.sha256(
        _json.dumps(yml, sort_keys=True, default=_canon).encode()).hexdigest()
    # Exactly the defaults ``create_optimizer`` reads, and no more. They are
    # CODE rather than config, and a supported optimizer can hang its whole
    # behaviour on one of them (the muon variants scale learning rate and
    # weight decay by their own fields), so a changed default must not match an
    # old checkpoint. Hashing unrelated fields instead would churn the
    # fingerprint and refuse resumes that are perfectly valid.
    _t = TrainingConfig()
    _defaults = {k: getattr(_t, k) for k in
                 ("weight_decay", "grad_clip_norm", "adamw_lr_scale",
                  "adamw_weight_decay_scale", "muon_lr_scale",
                  "muon_weight_decay_scale")}
    # The muon routing threshold is a module constant rather than a config
    # field, and it decides which parameters muon touches at all.
    try:
        from legoesm.ml import training as _tr

        _defaults["muon_min_dim"] = getattr(_tr, "_MUON_MIN_DIM_DEFAULT", None)
    except Exception:                                   # pragma: no cover
        _defaults["muon_min_dim"] = None
    return {
        "optimizer_defaults": hashlib.sha256(
            _json.dumps(_defaults, sort_keys=True).encode()).hexdigest()[:16],
        "config_sha256": digest[:32], "mode": cfg.mode,
        "training_core": cfg.training_core, "n_epochs": int(cfg.n_epochs),
        "lr": float(cfg.lr), "optimizer": str(cfg.optimizer),
        "smoke": bool(cfg.smoke), "warmup_steps": int(warmup),
        "rollout_steps": int(roll_steps),
        "n_global_samples": int(n_global_samples), "nproc": int(nproc),
    }


def _val_yaml(yml: dict, n_windows: int) -> dict:
    """The deck, with its training windows replaced by HELD-OUT ones.

    The windows are taken from ``eval_years`` -- the same years the scorecard
    scores -- spread evenly over the calendar so the validation number is not
    one season's weather. Everything else (grid, cadence, lead time, loss) is
    the deck's own, because a validation loss computed under a different
    protocol than the training loss is not comparable to it.
    """
    years = list(yml.get("eval_years") or [])
    if not years:
        raise SystemExit(
            "--val-windows needs `eval_years` in the campaign YAML: there is "
            "no held-out year to validate on.")
    # Month starts, evenly spread, cycling the eval years.
    starts = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    windows = []
    for i in range(int(n_windows)):
        year = int(years[i % len(years)])
        windows.append([year, starts[(i * 5) % len(starts)], 1])
    out = dict(yml)
    out["train_windows"] = windows
    out["train_years"] = years
    return out


def _device_sample(sample):
    """One host-resident sample, moved to the device for a no-gradient score.

    The training loop does exactly this per step (#1286 fix A keeps the shard
    on the host); the validation pass must not diverge from it, or the two
    losses would not be the same measurement.
    """
    import jax

    return jax.device_put(sample)


def _physical_params(params) -> dict:
    """Every trained parameter in PHYSICAL units, by qualified name.

    Read from the live pytree through each bundle's own constraint transforms,
    so the recorded numbers are the ones the physics saw -- not raw sigmoid
    pre-images that need the bounds to interpret.
    """
    out = {}
    for _attr in ("classical", "schemes"):
        _part = getattr(params, _attr, None)
        _as_dict = getattr(_part, "as_dict", None)
        if _as_dict is None:
            continue
        for _k, _v in _as_dict().items():
            _arr = np.asarray(_v)
            out[f"{_attr}.{_k}"] = (float(_arr) if _arr.ndim == 0
                                    else _arr.reshape(-1).tolist())
    if not out:
        _as_dict = getattr(params, "as_dict", None)
        if _as_dict is not None:
            for _k, _v in _as_dict().items():
                _arr = np.asarray(_v)
                out[_k] = (float(_arr) if _arr.ndim == 0
                           else _arr.reshape(-1).tolist())
    return out


def _probe_freeze_path(out_dir):
    import os

    return os.path.join(out_dir, "probe_freeze.json")


def _read_probe_freeze(out_dir, expect_fp):
    """The persisted reachability-probe result, or None.

    Honoured only when its fingerprint matches the CURRENT run exactly — the
    frozen set is a measurement on the initial parameters, shard and config,
    so any drift invalidates it the same way it invalidates an optimizer
    state.
    """
    import json as _json

    try:
        with open(_probe_freeze_path(out_dir)) as fh:
            obj = _json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("frozen"), list):
        return None
    if obj.get("schema") != _MANIFEST_SCHEMA:
        return None          # unknown schema fails toward re-probing (GLM)
    if obj.get("fingerprint") != expect_fp:
        return None
    if not all(isinstance(x, str) for x in obj["frozen"]):
        return None
    return obj["frozen"]


def _atomic_write(path, write_fn):
    """Write via a temporary file and rename, so a job killed mid-write leaves
    the PREVIOUS checkpoint intact rather than a truncated one that resume
    would happily load."""
    import os
    import socket

    # PID alone collides across nodes: two ranks or two jobs on different hosts
    # can share one, and they see the same shared filesystem.
    tmp = f"{path}.tmp.{socket.gethostname()}.{os.getpid()}"
    write_fn(tmp)
    os.replace(tmp, path)


def _uses_reachability_freeze(mode: str) -> bool:
    """Whether ``mode``'s parameters may be frozen on a zero-gradient probe.

    ONLY the scheme-parameter mode.  There, a zero gradient over the whole
    shard is strong evidence that the selected schemes do not read that knob at
    all — usually because it belongs to a scheme or a mode this run does not
    select — and that is a property of the configuration, which training does
    not change.  It is EVIDENCE, not proof: a knob behind a physical gate that
    no scene opens looks identical, which is what the post-training
    re-measurement exists to catch.

    A NEURAL model must never be filtered this way.  ``sfno`` is the sharp
    case: the registry ZEROES its decoder weight and bias by policy, so the
    untrained model emits exactly-zero tendencies and epoch 0 integrates the
    pure dycore — and by the chain rule EVERY upstream weight then has an
    exactly-zero gradient at step 0.  Freezing on that evidence would leave the
    decoder as the only trainable thing in the network, permanently.
    ``neural_gcm`` is not zero-initialized (it scales its output by 0.01
    instead), but the argument is the same in kind: a neural weight's zero
    gradient is a property of the CURRENT weights, which the next update
    changes, not of the configuration.  Only a scheme knob can be unreachable
    in a way that training will never fix.
    """
    return mode == "physics"


def _parse_hours(s):
    if not s:
        return ()
    return tuple(int(x) for x in str(s).split(",") if x != "")


def _clamped_warmup(total_steps: int, desired_warmup: int) -> int:
    """Warmup step count safe for ``optax.warmup_cosine_decay_schedule``.

    The schedule builds its cosine phase with ``decay_steps - warmup_steps``
    (``decay_steps`` == ``total_steps`` here), which optax requires to be
    strictly positive.  A tiny run (``--smoke`` gives ``total_steps`` of a few)
    otherwise trips ``cosine_decay_schedule requires positive decay_steps``.

    Honor the configured warmup verbatim whenever it is valid, only capping at
    ``total_steps - 1`` so ``decay_steps >= 1`` (this is the *sole* constraint
    optax imposes — no silent hyperparameter coercion beyond it).  The cap binds
    only when ``desired >= total_steps``, e.g. the degenerate ``total_steps == 1``
    (one epoch over a single local sample) -> warmup 0.  A YAML
    ``warmup_steps: <negative>`` is floored to 0 (no warmup), never passed
    through as a negative to optax.
    """
    return min(max(0, int(desired_warmup)), max(0, total_steps - 1))


def _rank_aware_warmup(desired_warmup: int, nproc: int) -> int:
    """Rank-invariant per-rank warmup from a GLOBAL-schedule ``warmup_steps``.

    ``total_steps`` fed to the cosine schedule is PER-RANK
    (``n_epochs * len(local)`` with ``len(local) == n_global_samples / nproc``),
    but ``warmup_steps`` in the YAML is sized against the GLOBAL schedule
    (``n_epochs * n_global_samples``).  Passed through verbatim it is therefore
    ``nproc`` times too large as a FRACTION of the per-rank schedule — 17% of
    the schedule at 1 rank becomes 69% at 4 ranks and ~100% at 16, where the
    learning rate never leaves warmup and never enters cosine decay (the #1464
    T106 defect, and the same latent bug in the other multi-rank WB configs).

    Dividing by ``nproc`` restores the configured warmup FRACTION at any rank
    count: ``warmup/(n_epochs*len(local)) ≈ warmup_steps/(n_epochs*n_global)``,
    to the NEAREST INTEGER warmup step (``round`` ties-to-even) — e.g. T106 at
    16 ranks resolves ``round(1000/16)=62`` for ``62/360=17.2%`` against an
    intended ``17.36%``, the closest a whole-step warmup can sit.
    Because the shards are balanced (``_sharded_indices`` drops the remainder,
    so every rank has the SAME ``len(local)`` and therefore the same
    ``total_steps``), that fraction is exact and identical on every rank — no
    per-rank schedule divergence.  At a single rank the division is the
    identity, so serial runs are unchanged; only the multi-rank schedule —
    where the bug lives — moves.  The result still passes through
    :func:`_clamped_warmup` for the optax positivity constraint.

    CONVENTION: this holds the warmup FRACTION of the schedule invariant, which
    is the right target under this trainer's FIXED peak LR (it is equivalent to
    holding the number of SAMPLES seen during warmup constant across rank
    count).  It would be the WRONG target under a linear-LR-scaling rule
    (LR ∝ nproc), where warmup should instead be held constant in steps — this
    trainer does not scale LR with rank count.  Note the schedule LENGTH in
    optimizer steps is still ``1/nproc`` (fewer, larger-batch updates over the
    same data): that is the standard large-batch trade, by design, not a bug
    this fixes — runs at different rank counts are not step-for-step equivalent.

    A YAML that intends a literal per-rank warmup can set ``nproc``-scaled
    values, but the global-fraction reading matches every committed WB config
    and the trainer's own ``n_global_samples`` fingerprint.

    ``desired <= 0`` stays 0 (no warmup, as configured).  A positive warmup
    floors at 1 rather than rounding away to 0 at very large rank counts: the
    bug being fixed is warmup SWALLOWING the run, and silently deleting it
    entirely at high ``nproc`` is the same class of surprise in the other
    direction.  The upper bound (``< total_steps``) is left to
    :func:`_clamped_warmup`.
    """
    d = max(0, int(desired_warmup))
    if d == 0:
        return 0
    return max(1, round(d / max(1, int(nproc))))


def _resolve_warmup(desired_warmup: int, total_steps: int, nproc: int) -> int:
    """The warmup the optimizer actually gets: rank-aware, then optax-clamped.

    The single resolution the training loop uses — kept as one named seam so a
    test covers the PRODUCTION path (not just the pieces): reverting the call
    site to the raw YAML ``warmup_steps`` moves this function's output at
    ``nproc > 1`` and fails ``test_resolve_warmup_is_rank_aware_at_the_call_site``.
    """
    return _clamped_warmup(total_steps, _rank_aware_warmup(desired_warmup, nproc))


def _apply_smoke_overrides(cfg: ScaleConfig, yml: dict) -> ScaleConfig:
    """Shrink ``yml`` (in place) + ``cfg`` to a fast single-GPU wiring check.

    A ~minutes pipeline check, per the runbook: a tiny 32x64x8 grid, one epoch,
    and **gray radiation** in place of rrtmgp.  The full rrtmgp k-distribution
    graph's JIT compile alone exceeds a 40-min single-GPU walltime inside the
    differentiable checkpointed rollout (issue #797 item 6); gray flows through
    the identical wiring under test (``compute_radiation_core``'s lat/lon
    flatten, the q_v smoothing gate) but compiles in seconds, so the smoke
    actually completes.  Opt back into the heavy path (to smoke the rrtmgp
    compile itself, with a bumped walltime) via ``smoke_radiation: rrtmgp``.
    """
    yml["n_lat"], yml["n_lon"], yml["nlev"] = 32, 64, 8
    # The spectral core takes its grid from ``spectral.n_max`` and only falls
    # back to n_lat; drop it so the smoke runs the same small grid (T21).
    if "spectral" in yml:
        yml["spectral"] = {k: v for k, v in yml["spectral"].items()
                           if k != "n_max"}
    yml["radiation"] = str(yml.get("smoke_radiation", "gray"))
    return cfg._replace(n_epochs=1)


def build_scale_config_from_args(argv=None) -> ScaleConfig:
    """Parse CLI into a ScaleConfig. Import-light + JAX-free (login-node testable)."""
    p = argparse.ArgumentParser(description="WeatherBench data-parallel scale training")
    p.add_argument("--mode", choices=VALID_MODES, default="neural_gcm")
    p.add_argument("--config", default="config/wb/scale/train_07deg.yaml")
    # The model grid comes from the YAML (n_lat/n_lon/nlev or spectral.n_max),
    # NOT from this flag — a --resolution that silently disagreed with the YAML
    # was the #817 papercut ("--resolution is a no-op on the model grid").
    # Default None = derive from the YAML for logging; an explicit value that
    # mismatches the YAML grid is now a hard error in main() instead of a
    # silent no-op.
    p.add_argument("--resolution", type=float, default=None, dest="resolution_deg",
                   help="Informational check only: must match the YAML grid "
                        "(180/n_lat). To change resolution, edit the YAML.")
    p.add_argument("--training-core", choices=VALID_TRAINING_CORES,
                   default="latlon", dest="training_core",
                   help="latlon = explicit C-grid production core; spectral = "
                        "Gaussian semi-implicit training core (#817 blocker 1: "
                        "bounded adjoint, no pole-cell dt clamp).")
    # None = YAML ``n_epochs``, else the historical 40 (same contract as --lr).
    p.add_argument("--epochs", type=int, default=None, dest="n_epochs")
    p.add_argument("--n-days", type=int, default=None, dest="n_days",
                   help="Training-window length [days] per train year (#1047). "
                        "Default None = YAML n_training_days, else 3. More days = "
                        "more samples (the WB arm was data-starved at 3). --smoke "
                        "forces 1.")
    p.add_argument("--multi-step-hours", default="6,12", dest="multi_step_hours")
    # None = take the campaign YAML's ``lr`` / ``optimizer`` (falling back to
    # the historical 3e-4 / adamw).  The CLI default used to be 3e-4 outright,
    # which silently overrode the YAML: the wb_classical_v2 campaign asked for
    # lr 1.5e-3 in its deck and trained at 3e-4 (checkpoint manifest proof).
    p.add_argument("--lr", type=float, default=None,
                   help=f"Peak LR. Default: YAML `lr`, else {_DEFAULT_LR}.")
    p.add_argument("--optimizer", default=None,
                   help="adamw|adam|muon|muon_partitioned. Default: YAML "
                        f"`optimizer`, else {_DEFAULT_OPTIMIZER}.")
    p.add_argument("--val-windows", type=int, default=0, dest="n_val_windows",
                   help="Held-out windows (from the deck's eval_years) scored "
                        "WITHOUT gradient after every epoch, so the run has a "
                        "generalisation curve rather than only a training "
                        "one. 0 (default) keeps the historical behaviour: no "
                        "validation pass.")
    p.add_argument("--warmup-steps", type=int, default=None,
                   dest="warmup_steps",
                   help="Global-schedule warmup steps, overriding the deck's "
                        "`warmup_steps`. Lets one campaign arm depart from a "
                        "deck other arms share. Every link of a chained run "
                        "must pass the same value (the resume fingerprint "
                        "covers the schedule).")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--out", default="results/wb_scale", dest="out_dir")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--eval-wb2", action="store_true")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args(argv)
    # --mode is already restricted by argparse ``choices=VALID_MODES`` (exits 2
    # on an unknown value), so no manual membership guard is needed here.
    if a.n_epochs is not None and a.n_epochs < 1:
        # total_steps = n_epochs * n_local_samples feeds optax's cosine
        # decay_steps, which must be positive; a zero/negative epoch count
        # would make it non-positive.
        raise SystemExit(f"--epochs must be >= 1, got {a.n_epochs}")
    if a.n_days is not None and a.n_days < 1:
        raise SystemExit(f"--n-days must be >= 1, got {a.n_days}")
    return ScaleConfig(
        mode=a.mode, config_path=a.config, resolution_deg=a.resolution_deg,
        n_epochs=a.n_epochs, multi_step_hours=_parse_hours(a.multi_step_hours),
        lr=a.lr, optimizer=a.optimizer, grad_accum=a.grad_accum,
        warmup_steps=a.warmup_steps, n_val_windows=a.n_val_windows,
        out_dir=a.out_dir, resume=a.resume, eval_wb2=a.eval_wb2, smoke=a.smoke,
        training_core=a.training_core, n_days=a.n_days,
    )


def _check_resolution_matches_yaml(cfg: ScaleConfig, yml: dict) -> float:
    """Resolve the run's resolution [deg] from the YAML grid; reject a
    mismatched explicit ``--resolution`` (the #817 no-op papercut).

    The model grid is built from ``yml["n_lat"]/["n_lon"]`` (lat-lon core) or
    ``yml["spectral"]["n_max"]`` (spectral core); ``--resolution`` never fed it.
    ``None`` (the default) derives 180/n_lat for logging.  An explicit value
    that disagrees by more than 5% is a HARD error — the old behaviour trained
    at the YAML grid while logging the flag's value.
    """
    derived = 180.0 / float(yml["n_lat"])
    if cfg.resolution_deg is None:
        return derived
    if abs(cfg.resolution_deg - derived) > 0.05 * derived:
        raise SystemExit(
            f"--resolution {cfg.resolution_deg:g} deg does not match the YAML "
            f"grid ({yml['n_lat']}x{yml['n_lon']} = {derived:.3g} deg). The "
            f"grid comes from the YAML; edit n_lat/n_lon (and nlev) there "
            f"instead of passing --resolution.")
    return cfg.resolution_deg


def _mpi_rank_size():
    """(rank, num_processes) — shared impl in legoesm.training.data_parallel
    (one copy for both this driver and the AIMIP chunked DP loop)."""
    from legoesm.training.data_parallel import mpi_rank_size
    return mpi_rank_size()



def main(argv=None):
    """Entry point, wrapped so that under a MULTI-rank job a rank which dies
    anywhere (data load, the reachability probe, the training loop) MPI_Aborts
    the whole job instead of leaving its peers blocked forever in the next
    collective (#985).  Transparent at a single rank."""
    import sys

    # The job wrapper asks THIS script what a complete epoch is, rather than
    # re-implementing the answer in shell. Handled before anything heavy is
    # imported: it runs once per link, on a login-class shell.
    args = list(sys.argv[1:] if argv is None else argv)
    # FIRST argument only: scanning the whole list let any OTHER option's VALUE
    # select the query mode, so a job wrapper whose CONFIG/OUT happened to hold
    # this token printed a signature and exited 0 having trained nothing.
    if args[:1] == ["--print-latest-complete"]:
        if len(args) < 2:
            print("--print-latest-complete needs a directory", file=sys.stderr)
            raise SystemExit(2)
        print_latest_complete_signature(args[1])
        return 0

    from legoesm.training.data_parallel import mpi_abort_on_uncaught
    return mpi_abort_on_uncaught(_main)(argv)


def _yml_int(yml, key, default):
    """Integer YAML value; bools and non-integral numerics are hard errors
    rather than silent truncation (``n_epochs: 1.9`` used to train ONE epoch).
    """
    v = _yml_or(yml, key, default)
    if isinstance(v, bool) or (isinstance(v, float) and not v.is_integer()) \
            or not isinstance(v, (int, float)):
        raise SystemExit(f"YAML {key}={v!r} is not an integer")
    return int(v)


def _yml_or(yml, key, default):
    """YAML value unless the key is absent OR present-but-null.  An explicit
    ``lr:`` with no value parses as None, and ``float(None)``/``str(None)``
    would crash or produce the string "None" (GLM diff review)."""
    v = yml.get(key)
    return default if v is None else v


def _resolve_training_keys(cfg, yml):
    """Resolve lr/optimizer from CLI-else-YAML and read the optimizer keys the
    YAML owns.  Returns ``(cfg, weight_decay, grad_clip_norm)`` where the two
    floats are ``None`` when the YAML is silent (caller substitutes
    ``TrainingConfig`` defaults, which live behind a JAX import).

    Every campaign YAML carries ``lr``/``optimizer``/``weight_decay``/
    ``grad_clip_norm``/``grad_accum``; until 2026-08-23 only ``warmup_steps``
    was actually read, so a deck could change its schedule and the run would
    silently keep the CLI defaults.  ``grad_accum`` has no implementation in
    this trainer, so any resolved value other than 1 is a hard error rather
    than a silently-ignored knob.
    """
    n_epochs = cfg.n_epochs if cfg.n_epochs is not None \
        else _yml_int(yml, "n_epochs", _DEFAULT_N_EPOCHS)
    if n_epochs < 1:
        raise SystemExit(f"n_epochs must be >= 1, got {n_epochs}")
    cfg = cfg._replace(
        lr=float(_yml_or(yml, "lr", _DEFAULT_LR)) if cfg.lr is None else cfg.lr,
        optimizer=(str(_yml_or(yml, "optimizer", _DEFAULT_OPTIMIZER))
                   if cfg.optimizer is None else cfg.optimizer),
        n_epochs=n_epochs,
    )
    # BOTH sources hard-error on != 1: gradient accumulation has no
    # implementation here, and it used to be a silent no-op from either side
    # (every prior run with the flag had a smaller effective batch than its
    # owner believed).
    for src, accum in (("--grad-accum", int(cfg.grad_accum)),
                       ("YAML grad_accum", _yml_int(yml, "grad_accum", 1))):
        if accum != 1:
            raise SystemExit(
                f"{src}={accum} requested but gradient accumulation is not "
                "implemented in train_weatherbench_scale; remove it or "
                "implement it (a silently-ignored optimizer knob is how the "
                "v2 campaign trained at the wrong learning rate).")
    wd = _yml_or(yml, "weight_decay", None)
    clip = _yml_or(yml, "grad_clip_norm", None)
    return (cfg, None if wd is None else float(wd),
            None if clip is None else float(clip))


def _main(argv=None):
    cfg = build_scale_config_from_args(argv)

    # Heavy imports here so --help / the CLI test stay JAX-free.
    import logging
    import pathlib

    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import yaml
    from legoesm.ml.training import TrainingConfig, create_optimizer
    from legoesm.training.data_parallel import (
        mpi_data_parallel_training_loop,
    )

    rank, nproc = _mpi_rank_size()
    logging.basicConfig(level=logging.INFO if rank == 0 else logging.WARNING)
    log = logging.getLogger("wb_scale")

    yml = yaml.safe_load(open(cfg.config_path))
    from legoesm.training.scale_build import validate_wb_campaign_yaml
    validate_wb_campaign_yaml(yml)
    cfg, _wd, _clip = _resolve_training_keys(cfg, yml)
    _t_defaults = TrainingConfig()
    weight_decay = _t_defaults.weight_decay if _wd is None else _wd
    grad_clip_norm = _t_defaults.grad_clip_norm if _clip is None else _clip
    if cfg.smoke:  # tiny, gray-radiation single-GPU wiring check (see helper)
        cfg = _apply_smoke_overrides(cfg, yml)
    # Logged AFTER the smoke override so n_epochs is what the run executes.
    if _wd is not None and cfg.optimizer in ("adam", "muon"):
        # TrainingConfig documents decay as ignored for adam; plain muon
        # applies it only through its (default-zero) scale fields.
        log.warning("YAML weight_decay=%g with optimizer=%s is partially or "
                    "fully inert (see TrainingConfig)", _wd, cfg.optimizer)
    log.info("training keys resolved: lr=%g optimizer=%s n_epochs=%d "
             "weight_decay=%g grad_clip_norm=%g "
             "(CLI overrides YAML; YAML overrides defaults)",
             cfg.lr, cfg.optimizer, cfg.n_epochs, weight_decay, grad_clip_norm)
    # Resolution comes FROM the YAML grid; an explicit mismatched --resolution
    # is a hard error, not a silent no-op (#817 papercut). Under --smoke the
    # grid is forced to 32x64, so the derived value is used verbatim.
    res_deg = (180.0 / float(yml["n_lat"]) if cfg.smoke
               else _check_resolution_matches_yaml(cfg, yml))
    log.info("WB scale train: mode=%s core=%s res=%.2fdeg ranks=%d",
             cfg.mode, cfg.training_core, res_deg, nproc)

    # --- build grid / sigma / physics pipeline / model / mode loss_fn ---
    #  (reuses the same package builders run_aimip_latlon uses; the mode differs
    #   only in the make_run_seg factory, per training_driver._build_train_step)
    from legoesm.training.scale_build import build_mode_components  # thin adapter (below)
    model, grid, sigma, params, make_run_seg, loss_config, dt = build_mode_components(cfg, yml)

    # Warm the RRTMGP optics-table cache with a CONCRETE build, before anything
    # traced runs.  RRTMGP reads its NetCDF gas-optics tables on first use and
    # stashes them in a module-level cache keyed on static file paths and gas
    # concentrations.  The classical arm constructs that stack inside
    # ``make_run_seg``, which ``loss_fn`` below reaches under
    # ``eqx.filter_jit(jax.value_and_grad(...))`` (data_parallel.py) — and
    # inside a trace ``jnp.array(<numpy table>)`` is a tracer, so the loader's
    # own ``np.asarray`` on it dies with TracerArrayConversionError (job
    # 26905933, after the 6-minute ERA5 load; reproduced standalone on JAX
    # 0.10.0).  One concrete call here fills the cache — its key holds only
    # static paths/flags, never a trained leaf, so the in-trace build hits it
    # and skips the load.  It is also where a broken physics config now fails,
    # seconds in rather than minutes.  ``neural_gcm``/``sfno`` build a cheap
    # closure here and are unaffected.  Mirrors the warm-up the AIMIP trainer
    # runs before its train step (``neural_gcm_spectral.main``).
    # Single-rank by construction in this campaign (ranks=1); at >1 rank every
    # rank reads its own copy of the tables, because the cache is per-process.
    # ``RRTMGP.preload_mpi`` (rank 0 reads + broadcasts) would avoid that, but
    # it needs the arm's RRTMGPConfig, which is built inside the physics factory
    # and not exposed here — wire it through if WB ever trains multi-rank.
    make_run_seg(params)

    # --- ERA5 IC/target/forcing samples, sharded across ranks ---
    # #1286: the loader builds ONLY this rank's contiguous shard (fix B — never
    # the full global list) and keeps it HOST-resident (fix A — the training
    # loop device_puts one sample at a time).  The rank's slice is byte-for-byte
    # the old ``shard_samples(build_all(), rank, nproc)`` partition, so gradient
    # semantics are unchanged; we no longer materialize the global GPU-resident
    # set that OOM'd at T106.
    from legoesm.training.scale_build import load_era5_samples
    local = load_era5_samples(cfg, yml, grid, sigma,
                              rank=rank, nproc=nproc, host_resident=True)
    if nproc > 1 and len(local) == 0:
        raise RuntimeError(
            f"rank {rank}: empty local shard (ranks={nproc} > global samples); "
            "reduce ranks or add training data")
    log.info("ERA5 samples: %d local/rank (host-resident, sharded-at-build)",
             len(local))

    # --- data-parallel loss over Equinox array-leaves ---
    from legoesm.training.losses import combined_loss
    from legoesm.training.scale_build import rollout_hours
    sigma_full = jnp.asarray(sigma.sigma_full)

    # The rollout horizon MUST match the target's lead time: load_era5_samples
    # pairs each IC with the state rollout_hours later (the first
    # multi_step_hours lead, default 6 h).  A fixed 24 h single_day_rollout here
    # would score a 24 h forecast against a 6 h target.
    roll_steps = int(rollout_hours(cfg, yml) * 3600.0 / dt)
    # MEMORY PROBE ONLY: override the scan trip count so the compiled gradient
    # can be measured at several rollout lengths with EVERY other thing --
    # shapes, physics, dt, loss -- held identical. The only difference between
    # arms is how many times the remat'd step is scanned, which is exactly the
    # variable that separates "the unrolled rollout holds the memory" from
    # "one step's physics does". Never set in a training run; the forecast it
    # produces would no longer match the target's lead.
    if _os.environ.get("LEGOESM_WB_ROLL_STEPS"):
        roll_steps = int(_os.environ["LEGOESM_WB_ROLL_STEPS"])
        print(f"MEMPROBE roll_steps OVERRIDDEN to {roll_steps}", flush=True)

    def _loss(trainable, sample):
        ic, target, forcing = sample
        pred = make_run_seg(trainable).raw(ic, roll_steps, forcing)
        return combined_loss(pred, target, sigma_full, grid=grid, config=loss_config)

    # Sized here rather than next to the optimizer because the resume check
    # below fingerprints the schedule: a run that changed its warmup or its
    # step count must not silently continue on restored Adam moments.
    total_steps = cfg.n_epochs * max(len(local), 1)
    # Make the YAML warmup RANK-AWARE before clamping: warmup_steps is sized
    # against the global schedule but total_steps is per-rank, so a literal
    # count balloons to ~nproc x its intended fraction (see
    # ``_rank_aware_warmup``; identity at nproc==1).  Then clamp safely below
    # total_steps (``_clamped_warmup``): the cosine schedule needs
    # decay_steps > 0, which the tiny --smoke run otherwise violates.
    warmup = _resolve_warmup(
        cfg.warmup_steps if cfg.warmup_steps is not None
        else yml.get("warmup_steps", TrainingConfig().warmup_steps),
        total_steps, nproc)

    # --- resume from the last completed epoch --------------------------------
    # Restored BEFORE the freeze, and the frozen set is restored WITH the
    # parameters rather than re-measured, so that a resume changes WALLTIME and
    # nothing else. Re-probing would make the trainable set a function of where
    # the scheduler happened to kill the job: a chained run would then diverge
    # from an uninterrupted one, and two chains killed at different points
    # would diverge from each other. (It also keeps the set matched to the
    # optimizer state that was built for it, and skips the probe's cost.)
    import json

    start_epoch, resumed_frozen = 0, None
    if cfg.resume:
        _ep = _latest_complete_checkpoint(cfg.out_dir)
        if _ep is None:
            # Either nothing has been written yet, or every epoch present is
            # missing its optimizer state (an older run, or a job killed
            # between writes). Both start from zero: a parameters-only restore
            # would restart AdamW cold with the learning rate back at warmup
            # while reporting a resumed run.
            # The reachability probe result IS restorable on its own: at T63
            # with si_substeps=3 the probe alone costs ~9 h, so a link killed
            # between probe and first-epoch write would otherwise re-measure
            # forever inside a 12 h walltime (job 27194893). The probe is a
            # pure function of the initial parameters + shard + config, all
            # covered by the fingerprint, so restoring it changes WALLTIME
            # and nothing else.
            _now_fp = _run_fingerprint(cfg, yml, warmup, roll_steps,
                                       len(local) * nproc, nproc)
            # Single-process only: under MPI an uncoordinated restore (one
            # rank reads the file, another hits an OSError and re-probes)
            # would leave ranks in different collectives and hang (codex).
            # Multi-rank WB training re-probes until a root-read + broadcast
            # exists. NOTE (codex, deferred): the fingerprint identifies the
            # shard by sample COUNT, not content — CLI --n-days is not in it;
            # the campaign path never passes --n-days, and adding the key
            # would orphan payloads already written under the current key
            # set. Add it at the next fingerprint schema change.
            _pf = (_read_probe_freeze(cfg.out_dir, _now_fp)
                   if nproc == 1 else None)
            if _pf is not None:
                resumed_frozen = set(_pf)
                log.info("resume: no complete epoch, but a fingerprint-"
                         "matched probe result exists (%d frozen leaves) -- "
                         "skipping the reachability probe, training from "
                         "epoch 0", len(resumed_frozen))
            else:
                log.info("resume: no COMPLETE epoch checkpoint under %s -- "
                         "starting from epoch 0", cfg.out_dir)
        else:
            _pp, _op, _fp = _checkpoint_paths(cfg.out_dir, _ep)
            params = eqx.tree_deserialise_leaves(_pp, params)
            _manifest = json.loads(open(_fp).read())
            _saved_fp = (_manifest.get("fingerprint")
                         if isinstance(_manifest, dict) else None)
            _now_fp = _run_fingerprint(cfg, yml, warmup, roll_steps,
                                       len(local) * nproc, nproc)
            # An absent or partial fingerprint is NOT a match. Accepting one
            # would restore parameters and optimizer state with no
            # compatibility check at all, which is the failure this exists to
            # prevent -- and it is exactly what a checkpoint written before
            # this manifest existed looks like.
            if not _manifest_is_valid(_manifest, expect_keys=_now_fp):
                raise RuntimeError(
                    f"resume: {_fp} is not a schema-{_MANIFEST_SCHEMA} "
                    "manifest with a complete run fingerprint (it predates "
                    "this format, or was truncated). Its optimizer state "
                    "cannot be checked for compatibility, so continuing from "
                    "it would silently risk a different experiment. Start a "
                    "fresh output directory.")
            _drift = {k: (v, _now_fp.get(k)) for k, v in _saved_fp.items()
                      if _now_fp.get(k) != v}
            if _drift:
                raise RuntimeError(
                    f"resume: {_pp} was written by a run configured "
                    f"differently -- {_drift} (saved, now). The restored Adam "
                    "moments and step count belong to that schedule, so "
                    "continuing would silently run a different experiment. "
                    "Start a fresh output directory, or re-run with the "
                    "original settings.")
            resumed_frozen = set(_manifest["frozen"])
            start_epoch = _ep + 1
            log.info("resume: epoch %d restored from %s (%d frozen leaves); "
                     "training epochs %d..%d",
                     _ep, _pp, len(resumed_frozen), start_epoch,
                     cfg.n_epochs - 1)
            if start_epoch >= cfg.n_epochs:
                log.info("resume: already at the requested %d epochs -- "
                         "nothing to do", cfg.n_epochs)

    # --- NO INERT PARAMETERS: freeze what this configuration cannot reach ---
    # The trainable bundle carries a knob for every scheme family and for the
    # switched-off modes of the families that ARE selected, so a large part of
    # it has no gradient path in any one run: measured on the classical arm,
    # 110 of 157 leaves had an exactly-zero gradient on all 48 probe scenes.
    # Left in the optimized set they land in the checkpoint and the tuned
    # report next to the parameters the loss actually moved, with nothing to
    # tell them apart — the run ships "tuned" values for parameters it never
    # trained.  (They also drift under AdamW's decoupled weight decay, but only
    # by ~1e-5 relative over the campaign's 12 epochs: real, and not the
    # reason.)  They are frozen here instead: still applied to the physics,
    # never updated, and named in the log.
    #
    # Reachability is MEASURED, because a hand-kept list of scheme names and
    # mode flags rots the moment a scheme gains a switch.  It is a GLOBAL
    # property under MPI, or the ranks would partition differently and the
    # per-step gradient allreduce would mismatch.  Cost: one extra compile of
    # the probe program, plus one forward+adjoint per probe scene before the
    # epoch loop and again after it.
    if _uses_reachability_freeze(cfg.mode):
        from legoesm.training.inert_params import (
            freeze_unreachable,
            measure_leaf_reachability,
            mpi_max_reduce,
            probe_indices,
        )
        _probe_static = eqx.partition(params, eqx.is_inexact_array)[1]
        # Kept alive past the freeze: the same program re-measures on the
        # TRAINED parameters at the end of the run (no second compile).
        _probe_vg = eqx.filter_jit(jax.value_and_grad(
            lambda a, s: _loss(eqx.combine(a, _probe_static), s)))
        # n_probe=None => every sample in the shard.  That costs one extra
        # forward+adjoint pass over the data (one epoch-equivalent of the
        # twelve a link trains) and removes the SAMPLING risk entirely: a leaf
        # is frozen iff it is zero on every scene in the shard.  It does NOT
        # make the answer exact for the whole run -- the gradients are those of
        # the INITIAL parameters, and a gate that opens as the live parameters
        # move is exactly the case the post-training report below exists to
        # catch.  The loss carries no parameter regularizer and the probe
        # applies no gradient clipping, so an exactly-zero gradient here is a
        # real disconnection on these scenes, not a value clipped to zero.
        # POLICY freeze on top of the MEASURED one: with ERA5 surface fluxes
        # prescribed the classical surface leaves (bulk exchange coefficients,
        # roughness, albedo, emissivity, the spatial surface field) are inert
        # by construction, and a low-sun column can still leak a tiny albedo
        # gradient that the zero-gradient probe would not freeze.  Applied on
        # BOTH branches (idempotent on resume) and BEFORE the persist below,
        # so the saved list is the effective trainable set.
        def _policy_freeze(arr_, static_, frozen_):
            if not bool(yml.get("era5_surface_fluxes", False)):
                return arr_, static_, frozen_
            from legoesm.training.inert_params import (
                apply_forced_freeze,
                prescribed_surface_frozen_names,
            )
            forced = prescribed_surface_frozen_names(params)
            if rank == 0 and forced:
                log.info("FROZEN BY POLICY (ERA5 surface fluxes prescribed): "
                         "%s", ", ".join(forced))
            return apply_forced_freeze(params, frozen_, forced)

        if resumed_frozen is not None:
            # Same trainable set as the job that wrote the optimizer state.
            from legoesm.training.inert_params import trainable_filter_spec

            _arr_all = eqx.partition(params, eqx.is_inexact_array)[0]
            _names = [jax.tree_util.keystr(pth)
                      for pth, _ in jax.tree.leaves_with_path(_arr_all)]
            _absmax = [0.0 if nm in resumed_frozen else 1.0 for nm in _names]
            arr, static = eqx.partition(
                params, trainable_filter_spec(params, _absmax, _names))
            frozen_names = [nm for nm in _names if nm in resumed_frozen]
            if len(frozen_names) != len(resumed_frozen):
                raise RuntimeError(
                    f"resume: {len(resumed_frozen)} frozen leaf names were "
                    f"saved but only {len(frozen_names)} match this run's "
                    "parameter tree; the configuration has changed since that "
                    "checkpoint and its optimizer state no longer applies.")
            n_probe_used = None          # nothing was probed; it was restored
            arr, static, frozen_names = _policy_freeze(arr, static, frozen_names)
        else:
            if _os.environ.get("LEGOESM_WB_MEM_PROBE") == "1":
                import sys as _sys
                _sys.path.insert(0, "scripts/tmp")
                import _probe_wb_mem_total as _pm
                _pm.run(_probe_vg, eqx.partition(params, eqx.is_inexact_array)[0],
                        local[0], label=_os.environ.get("LEGOESM_WB_MEM_ARM", "?"))
            arr, static, frozen_names, n_probe_used = freeze_unreachable(
                params, _probe_vg, local, n_probe=None, num_processes=nproc)
            arr, static, frozen_names = _policy_freeze(arr, static, frozen_names)
            # Persist the measurement immediately: at T63/si_substeps=3 the
            # probe costs ~9 h, and a link killed between here and the first
            # epoch write would otherwise re-measure on every chained resume
            # (job 27194893 died exactly there). Fingerprint-gated on read.
            if rank == 0:
                _os.makedirs(cfg.out_dir, exist_ok=True)
                _pf_payload = {
                    "schema": _MANIFEST_SCHEMA,
                    "fingerprint": _run_fingerprint(
                        cfg, yml, warmup, roll_steps, len(local) * nproc,
                        nproc),
                    "frozen": sorted(frozen_names),
                }
                import pathlib as _pathlib

                _atomic_write(
                    _probe_freeze_path(cfg.out_dir),
                    lambda t: _pathlib.Path(t).write_text(
                        json.dumps(_pf_payload)))
        n_live = len(jax.tree.leaves(arr))
        if n_probe_used is None:
            log.info("trainable leaves: %d live, %d frozen (RESTORED from the "
                     "checkpoint, not re-measured -- a resume must not change "
                     "the trainable set)", n_live, len(frozen_names))
        else:
            log.info("trainable leaves: %d live, %d frozen (unreachable on "
                     "every one of %d usable probe scenes)",
                     n_live, len(frozen_names), n_probe_used)
        if frozen_names:
            log.warning("FROZEN, no gradient on any scene in this shard at "
                        "the initial parameters: %s",
                        ", ".join(frozen_names))
        if n_live == 0:
            raise RuntimeError(
                "every trainable leaf is unreachable in this configuration -- "
                "the run would optimize nothing. Check the selected schemes "
                "and the rollout length before relaunching.")
    else:
        arr, static = eqx.partition(params, eqx.is_inexact_array)
        frozen_names, _probe_vg = [], None

    def loss_fn(arr_leaves, sample):
        return _loss(eqx.combine(arr_leaves, static), sample)

    optimizer = create_optimizer(TrainingConfig(
        lr=cfg.lr, optimizer=cfg.optimizer,
        warmup_steps=warmup, total_steps=total_steps,
        weight_decay=weight_decay, grad_clip_norm=grad_clip_norm,
    ))
    _fingerprint = _run_fingerprint(cfg, yml, warmup, roll_steps,
                                    len(local) * nproc, nproc)
    opt_state = optimizer.init(arr)
    if start_epoch > 0:
        # The moments AND the schedule step live here; restoring parameters
        # without this is a cold restart wearing a resumed run's name.
        opt_state = eqx.tree_deserialise_leaves(
            _checkpoint_paths(cfg.out_dir, start_epoch - 1)[1], opt_state)

    import os
    os.makedirs(cfg.out_dir, exist_ok=True)

    # --- held-out samples, scored without gradient after each epoch --------
    # A training curve alone cannot say whether the parameters are learning
    # the physics or the training scenes: both fall. These come from the
    # deck's EVAL years, the same ones the scorecard uses, so the two agree
    # about what "held out" means.
    val_samples = []
    if int(getattr(cfg, "n_val_windows", 0)) > 0:
        # The reduction below is unconditional, so a rank with an empty shard is
        # safe; this only keeps the WARNING visible rather than silent.
        if nproc > int(cfg.n_val_windows):
            log.warning(
                "--val-windows %d with %d ranks: some ranks hold no validation "
                "samples; the held-out loss is still a global mean over the "
                "samples that exist.", int(cfg.n_val_windows), nproc)
        val_samples = load_era5_samples(
            cfg, _val_yaml(yml, int(cfg.n_val_windows)), grid, sigma,
            rank=rank, nproc=nproc, host_resident=True)
        log.info("validation samples: %d local/rank from eval years %s",
                 len(val_samples), yml.get("eval_years"))

    def _validation_loss(cur_arr):
        # EVERY rank reaches the reduction, including one whose validation
        # shard came out empty: returning early on that rank left the others
        # blocked in the allreduce for the rest of the walltime (GLM). The
        # per-rank sum and count are reduced together, so the mean is over the
        # samples that exist rather than over the ranks that have any.
        if not val_samples and nproc <= 1:
            return None
        total = 0.0
        for _s in val_samples:
            total += float(loss_fn(cur_arr, _device_sample(_s)))
        if nproc <= 1:
            return total / len(val_samples)
        from mpi4py import MPI
        g_total = float(MPI.COMM_WORLD.allreduce(total, op=MPI.SUM))
        g_count = int(MPI.COMM_WORLD.allreduce(len(val_samples), op=MPI.SUM))
        if g_count == 0:
            return None
        return g_total / g_count

    _history_path = os.path.join(cfg.out_dir, "training_history.jsonl")

    def on_epoch(epoch, mean_loss, cur_arr, cur_opt_state):
        # Every rank runs the validation pass: it ends in a collective.
        val_loss = _validation_loss(cur_arr)
        if rank == 0:
            if val_loss is None:
                log.info("Epoch %4d: loss=%.6f", epoch, mean_loss)
            else:
                log.info("Epoch %4d: loss=%.6f val=%.6f",
                         epoch, mean_loss, val_loss)
            # One line per epoch: the two losses and every trained parameter
            # in PHYSICAL units, so the run's parameter trajectory can be read
            # without loading a checkpoint and inverting the transforms.
            _row = {"epoch": int(epoch), "train_loss": float(mean_loss),
                    "val_loss": None if val_loss is None else float(val_loss),
                    "params": _physical_params(eqx.combine(cur_arr, static))}
            with open(_history_path, "a") as _fh:
                _fh.write(json.dumps(_row) + "\n")
            ppath, opath, fpath = _checkpoint_paths(cfg.out_dir, epoch)
            # Parameters, optimizer state and the trainable set together --
            # any one of them missing makes the other two unresumable.
            _atomic_write(ppath, lambda t: eqx.tree_serialise_leaves(
                t, eqx.combine(cur_arr, static)))       # CURRENT params
            _atomic_write(opath, lambda t: eqx.tree_serialise_leaves(
                t, cur_opt_state))
            # Written LAST: it is the marker that says this epoch is
            # complete, and the chain wrapper keys its progress check off it.
            _atomic_write(fpath, lambda t: pathlib.Path(t).write_text(
                json.dumps({"schema": _MANIFEST_SCHEMA,
                            "frozen": frozen_names,
                            "fingerprint": _fingerprint})))

    arr, opt_state, history = mpi_data_parallel_training_loop(
        loss_fn, arr, opt_state, optimizer, local, cfg.n_epochs, nproc,
        on_epoch=on_epoch, start_epoch=start_epoch)
    params = eqx.combine(arr, static)
    log.info("training done: final loss=%.6f", history[-1] if history else float("nan"))

    # The freeze was decided on the UNTRAINED model.  A leaf can become
    # reachable once the live parameters move and open a gate no probe scene
    # opened — the clear case here is the microphysics, which is dead only
    # because every sample starts with exactly zero cloud condensate.  Nothing
    # unfreezes mid-run (that would rebuild the optimizer state and recompile),
    # so measure it now and SAY so, rather than let the next run inherit the
    # same blind spot silently.  Same probe program as before, hence no new
    # compile; every rank runs it because it ends in a collective.
    if frozen_names:
        _after, _names_after, _n_checked = measure_leaf_reachability(
            _probe_vg, params,
            [local[i] for i in probe_indices(len(local),
                                             _N_REACHABILITY_REPORT)],
            n_probe=_N_REACHABILITY_REPORT, reduce=mpi_max_reduce(nproc))
        _frozen = set(frozen_names)
        # Report the MAGNITUDE and every NAME, not a yes/no: a gradient of
        # 1e-30 is numerical dust, and the only non-arbitrary scale for "big
        # enough to matter" comes from the parameters this run actually
        # trained.
        # Finite positives only: a leaf can carry the probe's "connected but
        # the scene was poisoned" infinity, and that is a sentinel, not a
        # magnitude — it must not enter the scale.
        _live = [m for nm, m in zip(_names_after, _after)
                 if nm not in _frozen and 0.0 < m < float("inf")]
        # A low QUANTILE rather than the minimum: one trained parameter sitting
        # at 1e-20 of numerical dust would otherwise collapse the scale and
        # make every wakeup look material (GLM).  Nearest-rank 5th percentile,
        # which IS the minimum when fewer than 20 parameters are live -- there
        # is no lower order statistic to take.
        _floor = (sorted(_live)[max(0, -(-len(_live) // 20) - 1)]
                  if _live else 0.0)
        # The infinity is the probe's "connected, but that scene's gradient
        # was non-finite" marker, NOT a magnitude — reported separately rather
        # than printed as an enormous gradient.
        woke = sorted(((m, nm) for nm, m in zip(_names_after, _after)
                       if nm in _frozen and 0.0 < m < float("inf")),
                      reverse=True)
        woke_poisoned = sorted(nm for nm, m in zip(_names_after, _after)
                               if nm in _frozen and m == float("inf"))
        if rank == 0:
            if woke:
                log.warning(
                    "%d of the %d frozen parameters are REACHABLE on the "
                    "trained model and were NOT trained; %d of them carry a "
                    "gradient at least as large as the fifth-percentile "
                    "gradient among the parameters this run did train (%.3g). "
                    "The magnitudes are in the raw "
                    "pre-constraint space, so read each against that scale, "
                    "not against each other. Relaunch with these in the "
                    "trainable set to use them: %s",
                    len(woke), len(frozen_names),
                    sum(1 for m, _ in woke if m >= _floor), _floor,
                    ", ".join(f"{nm} ({m:.3g})" for m, nm in woke))
            if woke_poisoned:
                log.warning(
                    "%d further frozen parameters were reached by a NON-FINITE "
                    "gradient on the trained model: connected, but carrying no "
                    "usable magnitude: %s",
                    len(woke_poisoned), ", ".join(woke_poisoned))
            if not woke and not woke_poisoned:
                log.info("all %d frozen parameters are still unreachable on "
                         "the trained model, over the %d scene(s) this check "
                         "measured (a subsample of the shard: a leaf that "
                         "wakes only on one of the others would not show up "
                         "here)", len(frozen_names), _n_checked)

    if cfg.eval_wb2 and rank == 0:
        log.info("running WB2 scorecard on the held-out window ...")
        from legoesm.training.scale_build import evaluate_wb2
        evaluate_wb2(cfg, yml, model, params, make_run_seg, grid, sigma)


if __name__ == "__main__":
    main()
