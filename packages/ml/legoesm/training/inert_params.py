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

__all__ = ["assert_no_inert", "measure_leaf_reachability", "trainable_filter_spec",
           "mpi_max_reduce", "probe_indices", "freeze_unreachable"]


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
                              n_probe=4, reduce=None, device_put=True):
    """Largest |gradient| each leaf reaches over ``n_probe`` samples.

    ``value_and_grad_fn(arr, sample) -> (loss, grad)`` for one sample, with
    ``arr`` the array partition of ``params``.  A sample counts as USABLE only
    if BOTH its loss and its whole gradient are finite — the same bar the
    training step applies, and the reason a NaN loss with a finite (typically
    zero) gradient is the worst kind of evidence: it looks clean.  ``n_used``
    counts only those.  Every sample nonetheless contributes what it can prove
    about CONNECTIVITY (see the loop below), which can only keep a parameter
    trainable, never freeze one.

    With ``device_put`` each sample is moved to the device just before its
    evaluation and released after, exactly as the training loop stages them —
    the samples are host-resident on purpose and pre-staging the whole probe
    set would put several full model states on the card at once.

    ``reduce(absmax, n_used) -> (absmax, n_used)`` combines the per-process
    measurements before any of them is judged (see :func:`mpi_max_reduce`).  It
    is called UNCONDITIONALLY, on every process, before the all-poisoned check
    below — a collective that some ranks skip on their way to a rank-local
    raise hangs the job instead of failing it.

    Returns ``(absmax, names, n_used)``.
    """
    arr, _static = eqx.partition(params, eqx.is_inexact_array)
    names = _leaf_names(arr)
    absmax = [0.0] * len(names)
    n_used = 0
    for sample in samples[:n_probe]:
        loss, grad = value_and_grad_fn(
            arr, jax.device_put(sample) if device_put else sample)
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
            continue
        n_used += 1
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
                       num_processes=1, comm=None, device_put=True):
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
        reduce=mpi_max_reduce(num_processes, comm), device_put=device_put)
    spec = trainable_filter_spec(params, absmax, names)
    trainable, frozen = eqx.partition(params, spec)
    frozen_names = [nm for nm, m in zip(names, absmax) if not m > 0.0]
    return trainable, frozen, frozen_names, n_used
