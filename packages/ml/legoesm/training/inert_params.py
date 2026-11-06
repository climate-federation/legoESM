"""No inert parameters: freeze what the loss cannot move, on measured evidence.

STRICT RULE (user directive 2026-08-17): every leaf of a trainable pytree must
carry loss gradient.  A parameter the forward never consumes pretends to be
calibrated: the run writes it into the checkpoint and the tuned-parameter
report alongside the ones the loss actually moved, and nothing in the output
distinguishes them.  (It also drifts under AdamW's decoupled weight decay, but
only just: at the campaign's lr 3e-4, decay 1e-5 and ~2900 steps that is a
relative change of order 1e-5 — real, and far too small to be the reason.  The
reason is that the number is reported as calibrated.)

The land trainer states the rule for a flat dict of leaves
(``scripts/run/train_land_params_era5.py``, ``assert_no_inert``).  The
scheme-parameter trainers need one piece it does not: a REACHABILITY split.  A
classical model carries one parameterization per family, while the trainable
bundle carries knobs for every family plus knobs belonging to a switched-off
mode of a selected scheme.  Those are not defects — they are unreachable BY
CONFIGURATION — and the honest treatment is to freeze them OUT of the
optimized set, keeping their value in the forward, rather than to train them
against a zero gradient.

Reachability is decided by MEASUREMENT rather than by a hand-kept list of
scheme names and mode flags, which rots the moment a scheme gains a switch: a
leaf is TREATED as unreachable when its gradient is exactly zero on EVERY
probe sample.  That is evidence of a configuration-level disconnection, not
proof of one — a knob behind a physical gate that no probe scene opens is
indistinguishable from a knob nothing reads, which is why the caller is
expected to re-measure once the live parameters have moved.
The probe reads several samples SPREAD OVER THE RECORD because a leaf can be
legitimately zero on one scene (a convective trigger that does not fire there)
and live on the next; consecutive WeatherBench samples are six hours apart and
are nearly the same weather, so a contiguous probe is weak evidence.

Measured on the WeatherBench classical arm, 2026-08-18 (48 scenes, T21/L32):
110 of 157 leaves were identically zero on every one of them — 44 CLUBB
closure constants belonging to the prognostic path (the arm runs the diagnostic
one), 35 knobs for schemes this arm does not select, 13 microphysics rate
coefficients (nothing creates condensate inside a 6-hour window that starts
cloud-free), 11 convection knobs behind switched-off options, 7 cloud knobs for
convective cloud, which is off.
"""
from __future__ import annotations

import logging

import equinox as eqx
import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)

__all__ = ["assert_no_inert", "assert_no_inert_over", "measure_leaf_reachability",
           "trainable_filter_spec",
           "mpi_max_reduce", "probe_indices", "freeze_unreachable",
           "gradient_access_gate", "GATE_K_MAX"]


def assert_no_inert(g: dict) -> None:
    """STRICT RULE (user directive 2026-08-17): no inert parameters, ever.

    Every leaf in the trainable pytree must carry loss gradient; a parameter the
    forward never consumes silently pretends to be calibrated.  Called on the
    FULL-data init gradient of every training run (never a mini-batch — a
    seasonally/spatially gated param absent from one batch is not inert).

    Raises on (a) an identically-zero leaf (inert) and (b) an all-non-finite
    leaf — the reviewers' case: an all-NaN gradient would be zeroed by the
    downstream sanitiser every step, i.e. silently inert while looking live.
    A PARTIALLY zero per-PFT vector (some components live) passes with a printed
    warning: components for PFTs absent from the sample are legitimately zero
    and stay pinned at their prior."""
    dead = [k for k, v in g.items() if bool(jnp.all(v == 0.0))]
    poisoned = [k for k, v in g.items() if not bool(jnp.any(jnp.isfinite(v)))]
    if dead or poisoned:
        raise ValueError(
            "no-inert-parameters gate: "
            + (f"identically-zero gradient: {sorted(dead)}; " if dead else "")
            + (f"all-non-finite gradient (sanitiser would zero it every step): "
               f"{sorted(poisoned)}; " if poisoned else "")
            + "remove the key from the trainable set, enable the physics path "
              "that consumes it, or fix the init state (run the pre-filter)")
    for k, v in g.items():
        nz = int(jnp.sum(v == 0.0)); tot = int(jnp.asarray(v).size)
        if 0 < nz and tot > 1 and nz > tot // 2:
            logger.warning("no-inert gate: %s has %d/%d zero-gradient components "
                           "(PFTs absent from the sample stay at the prior)",
                           k, nz, tot)


def assert_no_inert_over(grads) -> None:
    """:func:`assert_no_inert` on the elementwise max |grad| over per-sample
    gradient dicts: a leaf gated off on one sample is not inert.  ``fmax``
    ignores a NaN sample, so one bad sample cannot poison a leaf."""
    absmax = None
    for g in grads:
        g = jax.tree.map(jnp.abs, g)
        absmax = g if absmax is None else jax.tree.map(jnp.fmax, absmax, g)
    if absmax is None:
        raise ValueError("no-inert-parameters gate: no samples to judge")
    assert_no_inert(absmax)


def _leaf_names(tree):
    return [jax.tree_util.keystr(p)
            for p, _ in jax.tree.leaves_with_path(tree)]


# Leaf-name markers of the classical SURFACE parameters.  With ERA5 surface
# fluxes prescribed as the lower boundary condition (WB ``era5_surface_fluxes``)
# these get no gradient — the bulk formula's output is replaced and the
# radiative surface is pinned — so they are frozen by POLICY, not only by
# measurement (no-inert-parameters rule).
PRESCRIBED_SURFACE_LEAF_MARKERS = (
    "surface_Cd_neutral",
    "surface_Ch_neutral",
    "surface_z0",
    "surface_z0h_z0_ratio",
    "surface_most_unstable_gamma",
    "surface_most_stable_beta",
    "rrtmgp_sfc_albedo",
    "rrtmgp_sfc_emissivity",
    "gray_sfc_albedo",
    "gray_sfc_emissivity",
    "spatial_surface",
    # Lat-lon classical model's TrainablePhysicsParams.raw_values leaves:
    # the bulk-transfer coefficients / surface albedos parameterize the same
    # prescribed air-sea coupling.  Brackets included so e.g. C_E cannot
    # match an unrelated key.
    "['C_H']",
    "['C_E']",
    "['albedo_ice']",
    "['albedo_ocean']",
)


# The subset of those markers that only shape the turbulent surface EXCHANGE
# (bulk coefficients, roughness, stability functions and their spatial fields):
# inert when the training surface setting is "none".  Albedo and emissivity are
# absent because radiation still reads them.
NO_SURFACE_FLUX_LEAF_MARKERS = (
    "surface_Cd_neutral",
    "surface_Ch_neutral",
    "surface_z0",
    "surface_z0h_z0_ratio",
    "surface_most_unstable_gamma",
    "surface_most_stable_beta",
    # spatial exchange fields only; the spatial albedo/emissivity stay live
    "fields['Cd_neutral']",
    "fields['Ch_neutral']",
    "fields['z0']",
    "['C_H']",
    "['C_E']",
)


def no_surface_flux_frozen_names(params) -> list:
    """Sorted keystr names of the surface-exchange leaves in ``params``."""
    arrays = eqx.partition(params, eqx.is_inexact_array)[0]
    return sorted(
        nm for nm in _leaf_names(arrays)
        if any(m in nm for m in NO_SURFACE_FLUX_LEAF_MARKERS))


def prescribed_surface_frozen_names(params) -> list:
    """Sorted keystr names of the classical-surface leaves in ``params``.

    Every inexact-array leaf whose path contains one of
    :data:`PRESCRIBED_SURFACE_LEAF_MARKERS`; empty when none match (a neural
    pytree, or a classical one without those knobs).
    """
    arrays = eqx.partition(params, eqx.is_inexact_array)[0]
    return sorted(
        nm for nm in _leaf_names(arrays)
        if any(m in nm for m in PRESCRIBED_SURFACE_LEAF_MARKERS))


def surface_policy_frozen_names(yml, params):
    """``(names, reason)`` of the surface leaves a WB deck freezes by policy:
    every classical surface leaf when ERA5 fluxes are prescribed, the
    exchange leaves only when ``classical.surface_bulk`` is "none", else
    ``([], None)``."""
    if bool(yml.get("era5_surface_fluxes", False)):
        return (prescribed_surface_frozen_names(params),
                "ERA5 surface fluxes prescribed")
    if str((yml.get("classical") or {}).get(
            "surface_bulk", "constant")) == "none":
        return (no_surface_flux_frozen_names(params),
                "classical.surface_bulk='none'")
    return [], None


def apply_forced_freeze(params, frozen_names, forced_names):
    """Re-partition ``params`` so the union of both name lists is frozen.

    Returns ``(arr, static, frozen_names)`` with ``frozen_names`` the sorted,
    de-duplicated union — the same triple ``freeze_unreachable`` hands back,
    so a caller can substitute it in place.
    """
    merged = sorted(set(frozen_names) | set(forced_names))
    arrays = eqx.partition(params, eqx.is_inexact_array)[0]
    names = _leaf_names(arrays)
    absmax = [0.0 if nm in merged else 1.0 for nm in names]
    arr, static = eqx.partition(
        params, trainable_filter_spec(params, absmax, names))
    return arr, static, merged


def measure_leaf_reachability(value_and_grad_fn, params, samples, *,
                              n_probe=4, reduce=None, device_put=True,
                              resume_state=None, on_scene=None):
    """Largest |gradient| each leaf reaches over ``n_probe`` samples.

    ``value_and_grad_fn(arr, sample) -> (loss, grad)`` for one sample, with
    ``arr`` the array partition of ``params``.  A sample counts as USABLE only
    if BOTH its loss and its whole gradient are finite — the same bar the
    training step applies, and the reason a NaN loss with a finite (typically
    zero) gradient is the worst kind of evidence: it looks clean.  ``n_used``
    counts only those.  Every sample nonetheless contributes what it can prove
    about CONNECTIVITY (see the loop below), which can only keep a parameter
    trainable, never freeze one.

    ``device_put`` may also be a callable ``sample -> placed sample`` (a
    sharded placement); ``True`` means ``jax.device_put``.

    With ``device_put`` each sample is moved to the device just before its
    evaluation and released after, exactly as the training loop stages them —
    the samples are host-resident on purpose and pre-staging the whole probe
    set would put several full model states on the card at once.

    ``reduce(absmax, n_used) -> (absmax, n_used)`` combines the per-process
    measurements before any of them is judged (see :func:`mpi_max_reduce`).  It
    is called UNCONDITIONALLY, on every process, before the all-poisoned check
    below — a collective that some ranks skip on their way to a rank-local
    raise hangs the job instead of failing it.

    ``resume_state=(n_done, absmax, n_used)`` skips the first ``n_done``
    samples and continues from their saved per-process measurement;
    ``on_scene(n_done, absmax, n_used)`` is called after every sample so a
    caller can persist that state (a probe longer than one job link).

    Returns ``(absmax, names, n_used)``.
    """
    arr, _static = eqx.partition(params, eqx.is_inexact_array)
    names = _leaf_names(arr)
    absmax = [0.0] * len(names)
    n_used = 0
    n_done = 0
    if resume_state is not None:
        n_done, absmax, n_used = resume_state
        absmax = [float(a) for a in absmax]
        if len(absmax) != len(names):
            raise ValueError(
                f"reachability probe: saved progress has {len(absmax)} leaves, "
                f"this parameter tree has {len(names)}")
    place = (device_put if callable(device_put)
             else jax.device_put if device_put else (lambda s: s))
    import logging
    import time
    for i_s, sample in enumerate(samples[:n_probe]):
        if i_s < n_done:
            continue
        _t0 = time.perf_counter()
        loss, grad = value_and_grad_fn(arr, place(sample))
        jax.block_until_ready(grad)
        logging.getLogger(__name__).info(
            "reachability probe scene %d/%d %.1f s", i_s + 1,
            len(samples[:n_probe]), time.perf_counter() - _t0)
        leaves = jax.tree.leaves(grad)
        finite = [bool(jnp.all(jnp.isfinite(g))) for g in leaves]
        usable = bool(jnp.isfinite(loss)) and all(finite)
        # EVERY scene contributes CONNECTIVITY evidence, whether or not it is
        # usable, because that evidence can only ever KEEP a parameter
        # trainable.  A finite leaf contributes its magnitude; a leaf whose
        # gradient came back inf/NaN is manifestly connected to the loss, so it
        # gets a positive sentinel (the value is only ever compared to zero).
        # Throwing a poisoned scene away wholesale is how a live parameter gets
        # frozen: its one non-zero gradient may be on exactly that scene.
        #
        # The error this ADMITS, deliberately, is the other direction: a leaf
        # can pick up a NaN cotangent from a `jnp.where` branch that does not
        # affect the output, so a genuinely inert parameter can be kept
        # trainable.  It then receives no update (its gradient is zero on every
        # usable scene), which is a strictly better failure than freezing a
        # parameter the loss can actually move.
        for k, g in enumerate(leaves):
            absmax[k] = max(absmax[k],
                            float(jnp.max(jnp.abs(g))) if finite[k]
                            else float("inf"))
        if not usable:
            logger.warning(
                "reachability probe: a sample had a non-finite loss or "
                "gradient and does not count as a usable scene (it is evidence "
                "about the scene, not the parameters); %d leaf/leaves were "
                "non-finite on it and are kept as REACHABLE",
                sum(1 for f in finite if not f))
        else:
            n_used += 1
        if on_scene is not None:
            on_scene(i_s + 1, list(absmax), n_used)
    if reduce is not None:
        absmax, n_used = reduce(absmax, n_used)
    if n_used == 0:
        raise ValueError(
            "reachability probe: every probe sample had a non-finite loss or "
            "gradient, so nothing can be concluded about which parameters are "
            "reachable. Fix the scenes before trusting the trainable set.")
    return absmax, names, n_used


def trainable_filter_spec(params, absmax, names=None):
    """Equinox filter spec: True for reachable array leaves, False otherwise.

    Pass it where ``eqx.is_inexact_array`` would go::

        spec = trainable_filter_spec(params, absmax, names)
        arr, static = eqx.partition(params, spec)

    A frozen leaf keeps its value — it is still applied to the physics, it
    simply receives no optimizer update (and no weight decay), which is the
    honest treatment of a parameter no probe scene could move.

    ``absmax`` is positional over ``params``'s array leaves in flatten order.
    ALWAYS pass the ``names`` that :func:`measure_leaf_reachability` returned
    with it: the LEAF PATHS are then compared, so a vector measured on a tree
    with different or reordered paths is rejected instead of silently freezing
    the wrong parameter.  Only paths — a tree with the same paths and different
    leaf SHAPES still passes, which is harmless here (the decision is one
    boolean per leaf, not per element).  Without ``names`` only the length is
    checked, which any same-sized foreign tree would pass.
    """
    arr, _static = eqx.partition(params, eqx.is_inexact_array)
    here = _leaf_names(arr)
    if len(here) != len(absmax):
        raise ValueError(
            f"reachability vector has {len(absmax)} entries but the pytree has "
            f"{len(here)} array leaves")
    if names is not None and list(names) != here:
        names = list(names)
        first = next((f"{a!r} vs {b!r}" for a, b in zip(names, here) if a != b),
                     f"{len(names)} names for {len(here)} leaves")
        raise ValueError(
            "reachability was measured on a different pytree: leaf paths "
            f"differ (first mismatch {first})")
    reach = dict(zip(here, absmax))
    return jax.tree_util.tree_map_with_path(
        lambda path, leaf: bool(eqx.is_inexact_array(leaf)
                                and reach[jax.tree_util.keystr(path)] > 0.0),
        params)


def mpi_max_reduce(num_processes, comm=None):
    """``reduce`` hook making reachability a GLOBAL property under MPI.

    Each rank probes its own shard, so without this every rank would freeze a
    different set of leaves — the ranks would then hold pytrees of different
    structure and the per-step gradient allreduce would mismatch.  A leaf is
    kept if ANY rank can move it (MAX), and the count of usable probe samples
    is summed so the "every sample poisoned" abort is unanimous.

    Returns ``None`` for a single process, which the caller passes straight
    through as "no reduction".
    """
    if num_processes <= 1:
        return None

    def _reduce(absmax, n_used):
        import numpy as np
        from mpi4py import MPI

        c = MPI.COMM_WORLD if comm is None else comm
        local = np.asarray(absmax, dtype=np.float64)
        out = np.empty_like(local)
        c.Allreduce(local, out, op=MPI.MAX)
        return out.tolist(), int(c.allreduce(int(n_used), op=MPI.SUM))

    return _reduce


def probe_indices(n_samples, n_probe):
    """Which samples to probe.  ``n_probe=None`` means EVERY sample.

    Probing everything removes the SAMPLING risk from the decision: a leaf is
    then frozen iff it is zero on every scene in the shard, for the cost of one
    extra forward+adjoint pass over it (one epoch-equivalent out of the twelve
    a campaign link trains).  Use it for the decision.  It does not make the
    decision exact for the whole RUN — see :func:`freeze_unreachable` on why a
    gate can still open later.

    A count instead gives an evenly spaced subset INCLUDING both ends, for the
    cheaper REPORTING pass.  Spanning the record matters more than the count —
    consecutive WeatherBench samples are six hours apart and are nearly the
    same weather, and a plain ``samples[::stride]`` never reaches the last one.
    ``n_probe=1`` is a single sample and therefore no span at all.
    """
    if n_samples < 1:
        raise ValueError("reachability probe: no samples")
    if n_probe is None:
        return list(range(n_samples))
    if n_probe < 1:
        raise ValueError(f"reachability probe: n_probe={n_probe} < 1")
    n = min(n_probe, n_samples)
    if n == 1:
        return [0]
    return sorted({round(i * (n_samples - 1) / (n - 1)) for i in range(n)})


def freeze_unreachable(params, value_and_grad_fn, samples, *, n_probe=None,
                       num_processes=1, comm=None, device_put=True,
                       resume_state=None, on_scene=None):
    """Split ``params`` into (trainable, frozen) by measured reachability.

    The probe samples are spread across ``samples`` (see :func:`probe_indices`)
    rather than taken from the front.  ``device_put`` stages each one on the
    device for its own evaluation, exactly as the training loop does — a
    host-resident sample carrying a bare PYTHON scalar (a numpy scalar already
    counts as an array to equinox) would otherwise be frozen as a static
    argument and retrace the probe program on every call.

    ``n_probe=None`` (the default) probes EVERY sample, so the frozen set is
    not a sampling estimate: it is exactly the set that is dead on the whole
    shard AT THE PARAMETERS PASSED IN.

    THE DECISION IS ONE-SHOT and taken on the UNTRAINED parameters.  A leaf can
    in principle become reachable later, once the live parameters move and open
    a gate that no probe scene opened (the clear case here: nothing creates
    cloud condensate at t=0, so the microphysics rates are dead until the
    convection and turbulence knobs change that).  Nothing unfreezes it inside
    the run; re-measure on the trained parameters afterwards and report what
    woke up, so the next run can be launched with a wider trainable set instead
    of inheriting the same blind spot.

    Returns ``(trainable, frozen, frozen_names, n_used)``, where ``trainable``
    and ``frozen`` recombine to ``params`` under ``eqx.combine``.
    """
    probe = [samples[i] for i in probe_indices(len(samples), n_probe)]
    absmax, names, n_used = measure_leaf_reachability(
        value_and_grad_fn, params, probe, n_probe=len(probe),
        reduce=mpi_max_reduce(num_processes, comm), device_put=device_put,
        resume_state=resume_state, on_scene=on_scene)
    spec = trainable_filter_spec(params, absmax, names)
    trainable, frozen = eqx.partition(params, spec)
    frozen_names = [nm for nm, m in zip(names, absmax) if not m > 0.0]
    return trainable, frozen, frozen_names, n_used


# ---------------------------------------------------------------------------
# Gradient-access gate (user rule 2026-11-05: "ensure we have access to the
# parameters that need to be trained. This should always be checked before any
# training.").  Design reviewed to GO by codex and GLM (round 7).
# ---------------------------------------------------------------------------

#: Longest rehearsal the gate runs before refusing (VOTED 3-0: Claude, codex,
#: GLM).  Real schedules need 4 steps fresh (lr = 0 only at step 0) and 3 on
#: resume; a schedule needing more is an unreviewed shape and fails closed.
GATE_K_MAX = 8

#: Rehearse until this many steps with lr > 0 have run (VOTED 3-0, round 2:
#: Claude, codex, GLM).  A zero-initialised final layer delays the layers
#: upstream of it by one update, and a zero-initialised learned embedding
#: feeding a first layer delays that layer's embedding columns by one more:
#: the real v9 NeuralGCM's embedding columns first get a gradient on the
#: THIRD lr > 0 step (4 steps on a fresh run, lr = 0 at step 0).  A deeper
#: zero-init chain aborts, named.
_GATE_N_POSITIVE = 3

#: Optimizers whose update with ZERO parameters is exactly the gradient-driven
#: update: AdamW's decoupled decay term is ``weight_decay * params`` and its
#: moments depend on the gradient only.  Any other optimizer refuses the run.
_GATE_OPTIMIZERS = ("adamw",)


class GradientAccessError(SystemExit):
    """The gate's verdict: raised identically on every rank."""


class _JaxControl:
    """Control channel for a JAX multi-process job without MPI (the
    column-sharded trainer: one logical trainer over many processes, joined
    by ``jax.distributed``): integer gathers over every process."""

    def gather(self, v: int):
        import numpy as np
        from jax.experimental import multihost_utils
        return np.asarray(multihost_utils.process_allgather(
            np.asarray(int(v), np.int64))).reshape(-1)

    def Free(self):
        pass


def _any_rank(flag: int, comm) -> int:
    """MAX of an integer over every process (identity without ``comm``)."""
    if comm is None:
        return int(flag)
    if isinstance(comm, _JaxControl):
        return int(comm.gather(flag).max())
    from mpi4py import MPI
    return int(comm.allreduce(int(flag), op=MPI.MAX))


def _ranks_agree(value: int, comm) -> bool:
    if comm is None:
        return True
    if isinstance(comm, _JaxControl):
        g = comm.gather(value)
        return bool(g.min() == g.max())
    from mpi4py import MPI
    return (comm.allreduce(int(value), op=MPI.MAX)
            == comm.allreduce(int(value), op=MPI.MIN))


def _schedule_count(opt_state) -> int:
    """The step count the learning-rate schedule reads next."""
    import optax
    st = [x for x in jax.tree_util.tree_leaves(
        opt_state, is_leaf=lambda x: isinstance(x, optax.ScaleByScheduleState))
        if isinstance(x, optax.ScaleByScheduleState)]
    if len(st) != 1:
        raise ValueError(f"expected one learning-rate schedule in the "
                         f"optimizer state, found {len(st)}")
    return int(st[0].count)


def _adamw_chain_problem(opt_state):
    """None if ``opt_state`` has exactly the structure of
    ``create_optimizer``'s clip + AdamW chain, else what differs.  Checked on
    the STATE, not a name: a mislabelled or wrapped (masked, partitioned)
    optimizer changes the state's structure."""
    import optax
    st = opt_state
    ok = (isinstance(st, tuple) and len(st) == 2
          and isinstance(st[0], optax.EmptyState)
          and isinstance(st[1], tuple) and len(st[1]) == 3
          and isinstance(st[1][0], optax.ScaleByAdamState)
          and isinstance(st[1][1], optax.EmptyState)
          and isinstance(st[1][2], optax.ScaleByScheduleState))
    if ok:
        return None
    return ("optimizer state is not the clip_by_global_norm + adamw chain "
            f"create_optimizer builds: {jax.tree_util.tree_structure(st)}")


def gradient_access_gate(steps, start, optimizer, schedule, *,
                         optimizer_name, num_processes=1, place=None,
                         excluded=None, k_max=GATE_K_MAX, log=None):
    """Rehearse the run's next real steps and abort unless every trainable
    element is reachable by the loss AND by the update that trains it.

    ``steps``: iterator of ``(value_and_grad_fn, sample)`` for the units the
    run will take next, in order (the run's own objective per curriculum stage
    and its own sample order).  ``start``: the LIST ``[params, opt_state]``
    (trainable array tree and optimizer state the run starts from).  The gate
    takes the two out and CLEARS the list, so a caller that keeps no other
    reference lets the starting copy be freed after the first update; the
    trees themselves are never modified (functional updates, discarded).  ``schedule``:
    the learning rate at an optimizer step count (``make_lr_schedule``).

    Each rehearsal step mirrors ``mpi_data_parallel_train_step``: local value
    and gradient, cross-rank MEAN, ``optimizer.update``, the shared
    ``apply_param_updates``.  With ``g`` the post-mean gradient,
    ``du = optimizer.update(g, s, zeros)`` (decay-free update) and
    ``ulp = |spacing(p)|``, an element passes when it ever has ``g != 0`` AND
    ever ``du != 0`` AND, on every step where ``du != 0`` and ``|du| >= ulp``,
    ``apply(p, du) != p``.  A leaf none of whose elements had a step with
    ``|du| >= ulp`` aborts as "apply path unverified" (VOTED 3-0).

    The rehearsal runs until the ``_GATE_N_POSITIVE``-th step with lr > 0
    (zero-initialised layers and embeddings make what they gate live one or
    two updates later), at most ``k_max`` steps; a run that ends sooner has
    all its remaining steps rehearsed, of which at least one must have lr > 0.  ``excluded``: optional tree of bool masks (MEASURED exclusions,
    elements zero by construction); an excluded element must have value 0 and
    gradient 0 on every rehearsal step.

    Every per-rank failure (exception incl. out-of-memory, non-finite loss or
    gradient, tree mismatch) is turned into a flag combined by allreduce MAX at
    two points per step -- before the gradient allreduce and after the update
    -- so all ranks raise together.  A HANG (not an exception) between those
    points is bounded only by the collective timeout and the MPI abort wrapper.

    Returns a report dict (per-leaf counts, ``k``, the rehearsal's final
    params/opt_state for tests).  Raises ``GradientAccessError`` (exit code 1)
    on any failure, on every rank.
    """
    log = logger if log is None else log
    place = (lambda x: x) if place is None else place
    # Control flags go to EVERY process (a column-sharded run has one logical
    # trainer, num_processes=1, over many processes) on a private duplicate
    # communicator, so they can never interleave with the gradient traffic.
    # MPI data parallel (num_processes > 1): a private Dup of COMM_WORLD.
    # A JAX multi-process job without MPI data parallel (column sharding):
    # JAX's own process gathers, since such a job need not load MPI at all.
    # Otherwise an MPI job whose JAX is not distributed-initialised still
    # gets the MPI channel; no MPI library at all = a single process.
    comm = None
    if int(num_processes) > 1:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD.Dup()
    elif jax.process_count() > 1:
        comm = _JaxControl()
    else:
        try:
            from mpi4py import MPI
            if MPI.COMM_WORLD.Get_size() > 1:
                comm = MPI.COMM_WORLD.Dup()
        except (ImportError, RuntimeError):
            pass
    try:
        # ``start`` itself is handed on (not unpacked here), so no frame of
        # this wrapper keeps the starting trees alive during the rehearsal.
        return _gate_body(steps, start, optimizer, schedule,
                          optimizer_name, num_processes, place, excluded,
                          k_max, log, comm)
    finally:
        if comm is not None:
            comm.Free()


def _gate_body(steps, start, optimizer, schedule, optimizer_name,
               num_processes, place, excluded, k_max, log, comm):
    import zlib

    import numpy as np
    from legoesm.training.data_parallel import (
        all_reduce_grad_mean,
        apply_param_updates,
    )

    def fail(msg):
        raise GradientAccessError(f"GRADIENT-ACCESS GATE FAILED: {msg}")

    def sync(err, what):
        # Every process reaches this; any local failure fails all of them.
        if _any_rank(int(err is not None), comm):
            fail(f"{what} (local: {err or 'ok'})")

    host = np.asarray   # verdict arrays live on the HOST: no device memory
    params, opt_state = start
    start.clear()
    err, sig = None, 0
    try:
        if optimizer_name not in _GATE_OPTIMIZERS:
            raise ValueError(f"optimizer {optimizer_name!r} is not one the gate "
                             f"can verify ({', '.join(_GATE_OPTIMIZERS)})")
        problem = _adamw_chain_problem(opt_state)
        if problem:
            raise ValueError(problem)
        leaves, treedef = jax.tree_util.tree_flatten(params)
        names = [jax.tree_util.keystr(pth)
                 for pth, _ in jax.tree_util.tree_leaves_with_path(params)]
        if not leaves:
            raise ValueError("the trainable tree is empty")
        bad_dt = [n for n, x in zip(names, leaves)
                  if jnp.dtype(x.dtype) not in (jnp.float32, jnp.float64)]
        if bad_dt:
            raise ValueError(f"trainable leaves must be float32/float64: {bad_dt}")
        empty = [n for n, x in zip(names, leaves) if x.size == 0]
        if empty:
            raise ValueError(f"empty trainable leaves: {empty}")
        shapes = [tuple(x.shape) for x in leaves]
        sizes = [int(x.size) for x in leaves]
        ex = ([np.zeros(sh, bool) for sh in shapes] if excluded is None
              else [host(m).astype(bool) for m in treedef.flatten_up_to(excluded)])
        if [m.shape for m in ex] != shapes:
            raise ValueError("exclusion masks do not match the leaf shapes")
        sig = zlib.crc32(repr((str(treedef), [(n, sh, str(x.dtype)) for n, sh, x
                                              in zip(names, shapes, leaves)])).encode())
        bad_ex = [n for n, x, m in zip(names, leaves, ex)
                  if m.any() and bool(np.any(host(x)[m] != 0))]
        if bad_ex:
            raise ValueError("excluded elements must be exactly 0 (weight decay "
                             f"would move them otherwise): {bad_ex}")
        count0 = _schedule_count(opt_state)
        # Ranks must agree on the optimizer state's structure and schedule
        # position too, or they would rehearse different step counts.
        sig = zlib.crc32(repr((sig, str(jax.tree_util.tree_structure(opt_state)),
                               count0)).encode())
        acc = {k: [np.zeros(sh, bool) for sh in shapes]
               for k in ("g", "du", "mvfail", "checked", "exgrad")}
    except Exception as e:   # noqa: BLE001 -- flagged, raised on all processes
        err = f"{type(e).__name__}: {e}"
    sync(err, "gate setup failed")
    if not _ranks_agree(sig, comm):
        fail("the trainable tree differs between ranks")

    # Only the rehearsal's own copies stay referenced from here on: a caller
    # that hands over its only reference (the trainer moves its own copy to
    # the host first) frees the starting tree after the first update.
    p, s = params, opt_state
    del params, opt_state, leaves
    n_pos, k, lrs = 0, 0, []
    it = iter(steps)
    while n_pos < _GATE_N_POSITIVE:
        if k >= k_max:
            fail(f"no step number {_GATE_N_POSITIVE} with lr > 0 within {k_max} rehearsal steps "
                 f"(lr = {lrs}); the schedule keeps lr at 0 too long")
        # --- sync (a): local step, BEFORE the gradient allreduce ---------
        err, exhausted, gsig = None, False, 0
        try:
            try:
                vg, sample = next(it)
            except StopIteration:
                exhausted = True
            if not exhausted:
                loss, g = vg(p, place(sample))
                del sample
                g_l = treedef.flatten_up_to(g)
                # What the gradient allreduce will pack, per rank: must agree.
                gsig = zlib.crc32(repr([(tuple(x.shape), str(x.dtype))
                                        for x in g_l]).encode())
                if not (bool(jnp.isfinite(loss)) and all(
                        bool(jnp.all(jnp.isfinite(x))) for x in g_l)):
                    err = f"non-finite local loss/gradient (loss={float(loss)!r})"
                del g_l
        except Exception as e:   # noqa: BLE001 -- flagged, raised on all ranks
            err = f"{type(e).__name__}: {e}"
        if _any_rank(int(exhausted), comm):
            # Every rank must have run out together, and a rank that did run
            # a step must not have failed it.
            sync(err, f"rehearsal step {k + 1} failed")
            if _any_rank(int(not exhausted), comm):
                fail("ranks ran out of rehearsal steps at different points")
            # The run ends before the third lr > 0 step: every remaining step
            # was rehearsed (vote 2026-11-06, 3-0); at least one must update.
            if n_pos == 0:
                fail(f"no lr > 0 step left in the run (lr = {lrs}); nothing "
                     "left whose gradient access could be verified")
            log.warning("gradient-access gate: the run has only %d lr > 0 "
                        "step(s) left; ALL remaining steps rehearsed (lr = %s)",
                        n_pos, lrs)
            break
        sync(err, f"rehearsal step {k + 1} failed")
        if not _ranks_agree(gsig, comm):
            fail(f"rehearsal step {k + 1}: gradient packing differs between ranks")
        err = None
        try:
            g = all_reduce_grad_mean(g, num_processes)
            # Complete the (asynchronously dispatched) gradient collective
            # before any host control collective can run.
            g = jax.block_until_ready(g)
        except Exception as e:   # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
        sync(err, f"rehearsal step {k + 1} gradient mean failed")
        # --- sync (b): update and verdicts, AFTER the allreduce ----------
        # Order keeps at most one extra parameter-sized tree alive beyond
        # what the training step itself holds: the decay-free update ``du``
        # is formed (from a zeros tree) and judged leaf by leaf, then dropped,
        # before the run's own update is formed.
        err = None
        try:
            lr = float(schedule(count0 + k))
            du = optimizer.update(
                g, s, jax.tree_util.tree_map(jnp.zeros_like, p))[0]
            # The movement probe goes through the SAME tree-level application
            # the run uses (one transient tree), judged leaf by leaf.
            moved = treedef.flatten_up_to(apply_param_updates(p, du))
            for i, (gi, di, pi) in enumerate(zip(
                    treedef.flatten_up_to(g), treedef.flatten_up_to(du),
                    treedef.flatten_up_to(p))):
                if not (bool(jnp.all(jnp.isfinite(gi)))
                        and bool(jnp.all(jnp.isfinite(di)))):
                    err = "non-finite mean gradient or update"
                # du != 0 too: spacing(0) is 0 where denormals flush to zero,
                # which would make a zero update "checkable" against itself.
                chk = (di != 0) & (jnp.abs(di) >= jnp.abs(jnp.spacing(pi)))
                mi = moved[i]
                moved[i] = None
                acc["g"][i] |= host(gi != 0)
                acc["du"][i] |= host(di != 0)
                acc["checked"][i] |= host(chk)
                acc["mvfail"][i] |= host(chk & (mi == pi))
                acc["exgrad"][i] |= ex[i] & host(gi != 0)
                del mi, chk
            del du, moved
            upd, s = optimizer.update(g, s, p)
            del g
            p_new = apply_param_updates(p, upd)
            # The run's own application must write the FULL update: p + upd
            # exactly (what optax.apply_updates computes), leaf by leaf.
            for i, (x, pi, ui) in enumerate(zip(treedef.flatten_up_to(p_new),
                                                treedef.flatten_up_to(p),
                                                treedef.flatten_up_to(upd))):
                if not bool(jnp.all(x == (pi + ui).astype(pi.dtype))):
                    err = f"the update was not applied to {names[i]}"
            del upd
            p = p_new
            del p_new
            for i, x in enumerate(treedef.flatten_up_to(p)):
                if not bool(jnp.all(jnp.isfinite(x))):
                    err = "non-finite parameters after the update"
                if ex[i].any() and bool(np.any(host(x)[ex[i]] != 0)):
                    err = f"excluded elements of {names[i]} moved off 0"
        except Exception as e:   # noqa: BLE001 -- flagged, raised on all ranks
            err = f"{type(e).__name__}: {e}"
        sync(err, f"rehearsal step {k + 1} update failed")
        lrs.append(lr)
        n_pos += lr > 0.0
        k += 1

    def _dead_lines(dead):
        # Whole dead rows / columns of a matrix leaf: names the input feature
        # or output channel the loss cannot reach.
        if dead.ndim != 2 or not dead.any():
            return ""
        rows, cols = np.nonzero(dead.all(1))[0], np.nonzero(dead.all(0))[0]
        out = ""
        if len(cols):
            out += f" dead_cols={cols.tolist()[:80]}"
        if len(rows):
            out += f" dead_rows={rows.tolist()[:80]}"
        return out

    rows, failed = [], []
    for i, n in enumerate(names):
        live = ~ex[i]
        dead = live & ~acc["g"][i]
        row = {"leaf": n, "size": sizes[i], "excluded": int(ex[i].sum()),
               "no_grad": int(dead.sum()),
               "no_update": int((live & ~acc["du"][i]).sum()),
               "not_moved": int((live & acc["mvfail"][i]).sum()),
               "excluded_with_grad": int(acc["exgrad"][i].sum()),
               "checked": int((live & acc["checked"][i]).sum()),
               "where": _dead_lines(dead)}
        row["apply_unverified"] = int(row["checked"] == 0
                                      and row["excluded"] < row["size"])
        rows.append(row)
        if any(row[c] for c in ("no_grad", "no_update", "not_moved",
                                "excluded_with_grad", "apply_unverified")):
            failed.append(row)
    # Verdicts are rank-identical (post-mean gradient, replicated params);
    # the MAX makes the decision identical even if they were not.
    if _any_rank(int(bool(failed)), comm):
        lines = "; ".join(
            f"{r['leaf']} ({r['size']} el: no_grad={r['no_grad']} "
            f"no_update={r['no_update']} not_moved={r['not_moved']} "
            f"excluded_with_grad={r['excluded_with_grad']} "
            f"apply_unverified={r['apply_unverified']}{r['where']})"
            for r in failed)
        fail(f"{len(failed)} of {len(rows)} trainable leaves lack access over "
             f"{k} rehearsal steps (lr = {lrs}): {lines}")
    n_el = sum(r["size"] - r["excluded"] for r in rows)
    n_chk = sum(r["checked"] for r in rows)
    log.info("gradient-access gate PASSED: %d leaves, %d elements, %d "
             "rehearsal steps (lr = %s); %d excluded elements; %d elements "
             "(%.1f%%) movement-checked, the rest updated below float "
             "precision on every step", len(rows), n_el, k, lrs,
             sum(r["excluded"] for r in rows), n_chk,
             100.0 * n_chk / max(n_el, 1))
    return {"k": k, "lrs": lrs, "rows": rows, "params": p, "opt_state": s}
