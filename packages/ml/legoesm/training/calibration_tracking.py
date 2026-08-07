"""Live tracking for a multi-day climate-model calibration campaign.

One source of truth, three renderings
-------------------------------------
A calibration campaign (:mod:`legoesm.training.etki`) evaluates an ensemble of
~20 members, each a full simulated year (~19 GPU-h), for ~6 iterations.  The
whole campaign is on the order of **120 data points over several days** — not a
training curve.  Everything here is sized for that: the record is the unit, not
the step, and the reader is a human deciding whether to keep spending.

* **``members.jsonl``** — the permanent record and the SOURCE OF TRUTH.  One
  line per member per iteration, appended and fsync'd immediately, so a SLURM
  kill can lose at most the line being written and never corrupts history.  It
  is plain greppable JSON: useful with no tooling at all.
* **``tb/``** — TensorBoard event files (:mod:`legoesm.training.tfevent_writer`),
  plain files on disk.  No server: Levante sits behind a login node and a tunnel
  is a nuisance, so the user points TensorBoard at the directory when they want
  it.
* **``index.html``** — a self-contained static page (inline CSS/JS, no CDN),
  regenerated after each iteration.  The at-a-glance view.

The event files and the page are **derived by re-reading the JSONL**, never from
in-memory state.  That is the whole point of the layout: the three outputs
cannot disagree, and a dashboard can be rebuilt from the log alone long after
the job that wrote it died (:func:`render_dashboard`).

What the page is FOR
--------------------
Four questions, in order of how much they should change your behaviour:

1. **Is the ensemble converging?**  Mean, best and SPREAD per iteration.  A
   narrowing spread is convergence; a spread that stays wide or grows means the
   optimiser is thrashing and the campaign should stop rather than keep
   spending.
2. **Is a parameter jammed at a bound?**  Every parameter is drawn as its
   position WITHIN its registered ``__param_spec__`` interval.  A parameter
   walking to its limit and sticking there is what happens when the optimiser
   chases a bias no parameter can close — the exact failure the observation-error
   floor (:mod:`legoesm.training.observation_error`) exists to prevent.  It is
   rendered as an alarm, not as a number in a table someone has to notice.
3. **Which measurement is driving the fit?**  Per-statistic contribution.  If
   one statistic dominates, the "calibration" is a single-variable fit.
4. **Are members blowing up?**  ``etki_update`` correctly DROPS non-finite
   members from its statistics rather than crashing; a rising blow-up count
   means the optimiser is walking into an unstable region, and a silent drop
   would hide that.

Decoupled from the optimiser
----------------------------
Nothing here imports ETKI.  The tracker consumes :class:`MemberRecord` (or a
plain dict of the same shape), so swapping the optimiser — which the ETKI
docstring says is intended — changes nothing on this side.  It also does not
compute the loss or its decomposition: the caller passes what the scorer
produced, so the page can never disagree with the objective that was actually
optimised.

Tracking never kills a run
--------------------------
Every public entry point on :class:`CalibrationTracker` is wrapped so an
exception is logged (stderr + ``tracking_errors.log``) and swallowed.  Losing a
plot is acceptable; losing 19 GPU-h is not.  This deliberately covers the JSONL
write too — see :meth:`CalibrationTracker.log_member`.

This module itself imports no JAX and no NumPy — only the stdlib — so rendering
a page is cheap and needs no device.  :func:`resolve_param_bounds` is the one
exception and imports the registry (and therefore JAX) lazily, inside the
function.  Note that reaching any of this through ``legoesm.training`` triggers
that package's own eager JAX imports; that is a property of the package, not of
this module.
"""

from __future__ import annotations

import contextlib
import html
import json
import math
import os
import statistics
import sys
import tempfile
import time
import traceback
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, NamedTuple

from legoesm.training.tfevent_writer import ScalarPoint, write_scalar_event_file

# --- layout of a run directory ---------------------------------------------
MEMBERS_FILENAME = "members.jsonl"
DASHBOARD_FILENAME = "index.html"
TENSORBOARD_DIRNAME = "tb"
ERROR_LOG_FILENAME = "tracking_errors.log"

# Schema tag on every JSONL line. Bump only for an incompatible change; the
# reader refuses a version it does not know rather than mis-parsing it.
SCHEMA_VERSION = 1

# A parameter whose ensemble mean sits within this fraction of the bound-to-bound
# interval of either end is flagged PINNED. 3% of the interval is deliberately
# tighter than "at the bound": by the time an ensemble mean is exactly on a
# clamp the campaign has already wasted iterations, so the alarm has to fire
# while the parameter is still only APPROACHING it.
#
# Related but different: ``campaign_summary._BOUND_ATOL/_BOUND_RTOL`` detect a
# per-column coefficient FIELD sitting exactly on its clamp. That is an
# after-the-fact clamp count; this is a proximity warning on a scalar ensemble
# mean. Do not merge them — the two questions have different answers.
DEFAULT_PIN_THRESHOLD = 0.03

# A statistic contributing more than this share of the total fitted loss makes
# the campaign a single-variable fit in all but name; the page says so.
DOMINANCE_SHARE = 0.5


# ---------------------------------------------------------------------------
# The record: what a caller hands over
# ---------------------------------------------------------------------------


class StatisticRecord(NamedTuple):
    """One fitted (or diagnostic) statistic for one member.

    ``contribution`` is that statistic's share of THIS member's loss, in the
    caller's own units — for an ETKI misfit it is the squared normalised
    residual ``((observed - model) / sigma)**2``. The tracker never recomputes
    it: it renders what the scorer produced.

    ``fitted=False`` marks a statistic that is REPORTED but not part of the
    fitted vector — ``net_toa``, whose exclusion
    (:data:`legoesm.training.observation_error.OBSERVATION_FLOORS`) is
    deliberate. Diagnostic statistics are shown separately and are excluded from
    the contribution shares, so they can never look like they are driving a fit
    they take no part in.
    """

    name: str
    model: float
    observed: float
    contribution: float
    fitted: bool = True


class MemberRecord(NamedTuple):
    """One ensemble member at one iteration — the unit of the whole system.

    ``loss`` may be non-finite for a member whose simulation blew up; set
    ``blew_up`` explicitly for a member that produced no numbers at all. A
    non-finite loss implies ``blew_up`` whether or not the caller said so.
    """

    iteration: int
    member: int
    parameters: Mapping[str, float]
    loss: float
    statistics: Sequence[StatisticRecord] = ()
    blew_up: bool = False
    timestamp: float | None = None
    note: str = ""

    @property
    def failed(self) -> bool:
        """True when this member must be excluded from ensemble statistics."""
        return bool(self.blew_up) or not _finite(self.loss)


def _finite(value: Any) -> bool:
    """True for a real, finite number (rejects None, NaN, inf, non-numerics)."""
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(as_float)


def _json_number(value: Any) -> float | None:
    """A JSON-safe number: non-finite and non-numeric both become ``null``.

    ``json.dumps`` emits bare ``NaN``/``Infinity`` by default, which is not
    valid JSON and breaks every non-Python reader of the log. The permanent
    record must be parseable by anything, so non-finite values are written as
    ``null`` and the ``blew_up`` flag carries the meaning.
    """
    return float(value) if _finite(value) else None


def coerce_record(record: MemberRecord | Mapping[str, Any]) -> MemberRecord:
    """Accept a :class:`MemberRecord` or a plain dict of the same shape.

    The dict form keeps the tracker usable from a driver that has no reason to
    import this module's types (the decoupling requirement). ``statistics`` may
    be a sequence of :class:`StatisticRecord`, a sequence of dicts, or a mapping
    of ``name -> {...}``; a mapping preserves insertion order, which is the
    order the page draws them in.
    """
    if isinstance(record, MemberRecord):
        stats = tuple(_coerce_statistic(s) for s in record.statistics)
        return record._replace(parameters=dict(record.parameters),
                               statistics=stats)
    if not isinstance(record, Mapping):
        raise TypeError(
            f"expected a MemberRecord or a mapping, got {type(record).__name__}")
    for key in ("iteration", "member"):
        if key not in record:
            raise ValueError(f"member record is missing required key {key!r}")
    raw_stats = record.get("statistics", ())
    if isinstance(raw_stats, Mapping):
        stats = tuple(_coerce_statistic(v, name=k) for k, v in raw_stats.items())
    else:
        stats = tuple(_coerce_statistic(s) for s in raw_stats)
    params = record.get("parameters", {})
    if not isinstance(params, Mapping):
        raise TypeError("member record 'parameters' must be a mapping of "
                        f"name -> value, got {type(params).__name__}")
    iteration, member = int(record["iteration"]), int(record["member"])
    # Rejected here rather than deep in the event encoder, where a negative
    # step raises and takes the whole render with it.
    if iteration < 0 or member < 0:
        raise ValueError(
            f"iteration and member must be >= 0, got iteration={iteration}, "
            f"member={member}")
    return MemberRecord(
        iteration=iteration,
        member=member,
        parameters={str(k): _to_float(v) for k, v in params.items()},
        loss=_to_float(record.get("loss")),
        statistics=stats,
        blew_up=bool(record.get("blew_up", False)),
        timestamp=(float(record["timestamp"])
                   if record.get("timestamp") is not None else None),
        note=str(record.get("note", "")),
    )


def _to_float(value: Any) -> float:
    """Numbers pass through; ``None`` and unparseable values become NaN."""
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _coerce_statistic(stat: Any, name: str | None = None) -> StatisticRecord:
    if isinstance(stat, StatisticRecord):
        return stat
    if not isinstance(stat, Mapping):
        raise TypeError(
            f"expected a StatisticRecord or mapping, got {type(stat).__name__}")
    stat_name = name if name is not None else stat.get("name")
    if not stat_name:
        raise ValueError("statistic record has no 'name'")
    return StatisticRecord(
        name=str(stat_name),
        model=_to_float(stat.get("model")),
        observed=_to_float(stat.get("observed")),
        contribution=_to_float(stat.get("contribution")),
        fitted=bool(stat.get("fitted", True)),
    )


def record_to_json(record: MemberRecord) -> str:
    """Serialise one record to a single JSON line (no newline)."""
    payload: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "iteration": int(record.iteration),
        "member": int(record.member),
        "timestamp": (float(record.timestamp)
                      if record.timestamp is not None else time.time()),
        "blew_up": bool(record.failed),
        "loss": _json_number(record.loss),
        "parameters": {str(k): _json_number(v)
                       for k, v in record.parameters.items()},
        "statistics": {
            s.name: {"model": _json_number(s.model),
                     "observed": _json_number(s.observed),
                     "contribution": _json_number(s.contribution),
                     "fitted": bool(s.fitted)}
            for s in record.statistics},
    }
    # The JSONL keys statistics by name, so a duplicate name would be silently
    # dropped on write and the log would disagree with the record it came from
    # — the one thing this whole layout exists to prevent. Refuse instead.
    if len(payload["statistics"]) != len(record.statistics):
        counts: dict[str, int] = {}
        for stat in record.statistics:
            counts[stat.name] = counts.get(stat.name, 0) + 1
        duplicates = sorted(n for n, c in counts.items() if c > 1)
        raise ValueError(
            f"member ({record.iteration}, {record.member}) reports statistic "
            f"{duplicates} more than once; the log keys statistics by name, so "
            "one of them would be lost")
    if record.note:
        payload["note"] = str(record.note)
    # allow_nan=False makes a non-finite leak a LOUD error here rather than an
    # unparseable permanent record; _json_number should have caught them all.
    return json.dumps(payload, allow_nan=False, sort_keys=False)


# ---------------------------------------------------------------------------
# The append-only log
# ---------------------------------------------------------------------------


def append_member_record(path: str | os.PathLike[str],
                         record: MemberRecord | Mapping[str, Any]) -> None:
    """Append one record to the JSONL log, durably.

    Opens, writes, flushes, ``fsync``s and closes on every call. Holding a file
    handle open across a multi-hour member would risk losing buffered lines to a
    kill, and 120 opens over several days costs nothing.
    """
    line = record_to_json(coerce_record(record))
    path = os.fspath(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_member_records(path: str | os.PathLike[str]) -> list[MemberRecord]:
    """Read every well-formed record back, in file order.

    Tolerant by design: a line that is truncated or unparseable (a kill during a
    write) is SKIPPED with a warning rather than raising, because the whole
    point of an append-only log is that the surviving history stays readable.
    A line whose ``schema`` is from the future raises — mis-parsing a record is
    worse than refusing it.
    """
    path = os.fspath(path)
    if not os.path.exists(path):
        return []
    records: list[MemberRecord] = []
    with open(path, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                print(f"[calibration_tracking] skipping unparseable line "
                      f"{lineno} of {path} (truncated write?)", file=sys.stderr)
                continue
            if not isinstance(payload, dict):
                print(f"[calibration_tracking] skipping line {lineno} of "
                      f"{path}: a record must be a JSON object, got "
                      f"{type(payload).__name__}", file=sys.stderr)
                continue
            # A FUTURE schema raises; a CORRUPT one is just another malformed
            # line. Both checks sit inside the guard, because the whole promise
            # of an append-only log is that one bad line cannot cost the rest.
            try:
                schema = int(payload.get("schema", SCHEMA_VERSION))
            except (TypeError, ValueError):
                print(f"[calibration_tracking] skipping line {lineno} of "
                      f"{path}: unreadable schema "
                      f"{payload.get('schema')!r}", file=sys.stderr)
                continue
            if schema > SCHEMA_VERSION:
                raise ValueError(
                    f"{path}:{lineno} has schema {schema}, newer than this "
                    f"reader's {SCHEMA_VERSION}; upgrade rather than "
                    "mis-parsing the record")
            try:
                records.append(coerce_record(payload))
            except (TypeError, ValueError) as exc:
                print(f"[calibration_tracking] skipping malformed line "
                      f"{lineno} of {path}: {exc}", file=sys.stderr)
    return records


# ---------------------------------------------------------------------------
# Aggregation — the one place ensemble statistics are defined
# ---------------------------------------------------------------------------


class ParameterSummary(NamedTuple):
    """One parameter's ensemble position at one iteration.

    ``bound_fraction`` is ``(value - lo) / (hi - lo)``: 0 at the lower bound, 1
    at the upper. The interval is treated LINEARLY because that is how the
    ``__param_spec__`` sigmoid transform maps the unconstrained parameter onto
    it — a log-scaled fraction would not be the quantity the optimiser is
    actually moving through.

    ``None`` bounds mean no ``__param_spec__`` entry was supplied for this
    parameter: it is still logged and plotted by value, but it cannot be checked
    for pinning, and the page says so rather than implying it is safe.
    """

    name: str
    lo: float | None
    hi: float | None
    mean: float
    minimum: float
    maximum: float
    bound_fraction: float | None
    fraction_min: float | None
    fraction_max: float | None
    pinned: str          # "" | "lower" | "upper"
    n_reported: int = 0  # surviving members that actually carried this param


class StatisticSummary(NamedTuple):
    """One statistic's ensemble-mean contribution at one iteration.

    ``share`` is of the total DECOMPOSED contribution across fitted statistics,
    which is not necessarily the loss: the caller may report a loss carrying
    terms these statistics do not decompose. :attr:`IterationSummary` exposes
    both so the page can name the difference instead of implying there is none.

    ``observed`` is TAKEN from the members, not averaged — it is a reference
    value, identical for every member by construction.
    """

    name: str
    fitted: bool
    model_mean: float
    observed: float
    contribution_mean: float
    share: float         # of the total decomposed contribution; 0.0 if not fitted
    n_reported: int = 0  # surviving members behind model_mean


class IterationSummary(NamedTuple):
    """Everything the renderings need about one iteration.

    Loss statistics are over the VALID members only — an iteration is scored on
    what survived. ``None`` when nothing survived, which the renderers must
    handle rather than assume away.

    "Valid" here means ``blew_up`` is unset AND the loss is finite
    (:attr:`MemberRecord.failed`). That is the same SPIRIT as
    ``etki_update``'s dropping of members with non-finite forward evaluations,
    but not the same TEST: ETKI checks every entry of the forward vector, so a
    member with one NaN statistic and a finite loss is dropped there and kept
    here. Flag such a member ``blew_up`` explicitly if you want the two to
    agree exactly; the tracker deliberately does not re-derive the optimiser's
    validity rule, because then a change on either side would silently
    desynchronise them.
    """

    iteration: int
    n_members: int
    n_valid: int
    blown_up: tuple[int, ...]
    loss_mean: float | None
    loss_best: float | None
    loss_worst: float | None
    loss_std: float | None
    parameters: tuple[ParameterSummary, ...]
    statistics: tuple[StatisticSummary, ...]
    wall_time: float

    @property
    def loss_spread(self) -> float | None:
        """max - min across surviving members: the convergence signal."""
        if self.loss_best is None or self.loss_worst is None:
            return None
        return self.loss_worst - self.loss_best

    @property
    def decomposed_total(self) -> float:
        """Sum of the fitted statistics' mean contributions.

        The denominator behind every ``share``. Compare it with
        ``loss_mean``: if they differ, the statistics do not fully account for
        the loss and a share is a share of the DECOMPOSITION, not of the loss.
        """
        return sum(s.contribution_mean for s in self.statistics
                   if s.fitted and _finite(s.contribution_mean)
                   and s.contribution_mean > 0.0)

    @property
    def undecomposed_loss(self) -> float | None:
        """Loss not accounted for by the per-statistic contributions."""
        if self.loss_mean is None:
            return None
        return self.loss_mean - self.decomposed_total


def _bound_fraction(value: float, lo: float, hi: float) -> float | None:
    """Position in ``[lo, hi]`` as a fraction; ``None`` for a degenerate range.

    NOT clipped: a value outside its own registered bounds is a real condition
    (a mis-registered bound, or an optimiser step applied before the constraint
    transform) and clipping it to 1.0 would disguise it as merely pinned.
    """
    if not (_finite(value) and _finite(lo) and _finite(hi)):
        return None
    span = float(hi) - float(lo)
    if span <= 0.0:
        return None
    return (float(value) - float(lo)) / span


def summarize_iteration(
    records: Sequence[MemberRecord],
    param_bounds: Mapping[str, tuple[float, float]] | None = None,
    *,
    pin_threshold: float = DEFAULT_PIN_THRESHOLD,
) -> IterationSummary:
    """Reduce one iteration's member records to its summary.

    A member index appearing more than once is collapsed to its LAST record.
    That is not defensive noise: the log is append-only and the campaign is
    restartable, so a member re-run after a crash legitimately appears twice,
    and the later attempt supersedes the earlier one. Averaging both would
    weight that member twice in the ensemble mean — a silently wrong number,
    which is worse than a crash. The JSONL itself keeps every attempt; only
    this ensemble view de-duplicates.
    """
    if not records:
        raise ValueError("cannot summarize an iteration with no records")
    if not 0.0 <= pin_threshold < 0.5:
        raise ValueError(
            f"pin_threshold must be in [0, 0.5), got {pin_threshold!r}")
    param_bounds = param_bounds or {}
    iteration = records[0].iteration
    # De-duplication keys on the member index, so records from more than one
    # iteration would silently erase each other. Group with `summarize` first.
    mixed = {r.iteration for r in records}
    if len(mixed) != 1:
        raise ValueError(
            f"summarize_iteration takes the records of ONE iteration, got "
            f"{sorted(mixed)}; use summarize() to group them")
    by_member: dict[int, MemberRecord] = {}
    for record in records:
        by_member[int(record.member)] = record          # last write wins
    records = [by_member[m] for m in sorted(by_member)]
    valid = [r for r in records if not r.failed]
    blown = tuple(sorted(r.member for r in records if r.failed))

    losses = [float(r.loss) for r in valid]
    loss_mean = statistics.fmean(losses) if losses else None
    loss_best = min(losses) if losses else None
    loss_worst = max(losses) if losses else None
    loss_std = statistics.pstdev(losses) if len(losses) > 1 else (
        0.0 if losses else None)

    # Parameters: ordered by first appearance so the page is stable across
    # iterations. Blown-up members are excluded here too — their parameters were
    # evaluated but produced no usable score, and ETKI resets them to the
    # ensemble mean, so including them would smear the position being tracked.
    names: list[str] = []
    for record in valid:
        for name in record.parameters:
            if name not in names:
                names.append(name)
    parameters = []
    for name in names:
        values = [float(r.parameters[name]) for r in valid
                  if name in r.parameters and _finite(r.parameters[name])]
        if not values:
            continue
        bounds = param_bounds.get(name)
        lo, hi = (float(bounds[0]), float(bounds[1])) if bounds else (None, None)
        mean = statistics.fmean(values)
        low, high = min(values), max(values)
        if lo is None or hi is None:
            frac = frac_lo = frac_hi = None
            pinned = ""
        else:
            frac = _bound_fraction(mean, lo, hi)
            frac_lo = _bound_fraction(low, lo, hi)
            frac_hi = _bound_fraction(high, lo, hi)
            pinned = ""
            # QUORUM: pinning is the page's headline alarm, so it is only
            # raised when a MAJORITY of the surviving members actually reported
            # this parameter. Without it, one member out of twenty carrying a
            # stray value fires the same red banner as a whole ensemble jammed
            # against its bound, and the reader cannot tell them apart.
            quorum = len(values) * 2 > len(valid)
            if frac is not None and quorum:
                if frac <= pin_threshold:
                    pinned = "lower"
                elif frac >= 1.0 - pin_threshold:
                    pinned = "upper"
        parameters.append(ParameterSummary(
            name=name, lo=lo, hi=hi, mean=mean, minimum=low, maximum=high,
            bound_fraction=frac, fraction_min=frac_lo, fraction_max=frac_hi,
            pinned=pinned, n_reported=len(values)))

    # Statistics: same first-appearance ordering, so the stacked bars keep one
    # colour per statistic across the whole campaign.
    stat_names: list[str] = []
    fitted_flags: dict[str, bool] = {}
    for record in valid:
        for stat in record.statistics:
            if stat.name not in stat_names:
                stat_names.append(stat.name)
            # A statistic is diagnostic-only if ANY member reported it so; the
            # exclusion is a property of the objective, not of a member.
            fitted_flags[stat.name] = (
                fitted_flags.get(stat.name, True) and bool(stat.fitted))
    raw: list[tuple[str, bool, float, float, float, int]] = []
    for name in stat_names:
        # ONE mask for model and observed: the page subtracts them to show the
        # bias, and a difference of two means taken over different subsets of
        # members is not a bias of anything. The contribution carries its own
        # mask because a DIAGNOSTIC statistic has no contribution by design
        # (net_toa), and dropping it from the reported values on that account
        # would hide the number the exclusion exists to keep visible.
        models: list[float] = []
        observations: list[float] = []
        contributions: list[float] = []
        for record in valid:
            for stat in record.statistics:
                if stat.name != name:
                    continue
                if _finite(stat.model) and _finite(stat.observed):
                    models.append(float(stat.model))
                    observations.append(float(stat.observed))
                if _finite(stat.contribution):
                    contributions.append(float(stat.contribution))
        # The observed value is a REFERENCE, identical for every member by
        # construction. Averaging it would quietly paper over a scorer bug, so
        # take the first and complain loudly if the members disagree.
        observed = observations[0] if observations else float("nan")
        if observations and max(observations) != min(observations):
            print(f"[calibration_tracking] members report DIFFERENT observed "
                  f"values for {name!r} ({min(observations)!r} … "
                  f"{max(observations)!r}); using {observed!r}. A reference "
                  "value must not vary between members — this is a scorer bug",
                  file=sys.stderr)
        raw.append((
            name, fitted_flags.get(name, True),
            statistics.fmean(models) if models else float("nan"),
            observed,
            statistics.fmean(contributions) if contributions else float("nan"),
            len(models)))
    # Shares are of the TOTAL DECOMPOSED CONTRIBUTION, which is not necessarily
    # the loss (the caller may report a loss carrying terms the statistics do
    # not decompose). `decomposed_total` is surfaced so the page can say so
    # rather than implying the two are the same. Negative contributions make a
    # share meaningless, so the denominator is the sum of POSITIVE ones and a
    # negative contribution reports a zero share instead of an absurd one.
    fitted_total = sum(c for _, fitted, _, _, c, _n in raw
                       if fitted and _finite(c) and c > 0.0)
    stat_summaries = tuple(
        StatisticSummary(
            name=name, fitted=fitted, model_mean=model, observed=observed,
            contribution_mean=contribution, n_reported=n_reported,
            share=((contribution / fitted_total)
                   if (fitted and fitted_total > 0.0 and _finite(contribution)
                       and contribution > 0.0)
                   else 0.0))
        for name, fitted, model, observed, contribution, n_reported in raw)

    stamps = [r.timestamp for r in records if r.timestamp is not None]
    return IterationSummary(
        iteration=iteration,
        n_members=len(records),
        n_valid=len(valid),
        blown_up=blown,
        loss_mean=loss_mean,
        loss_best=loss_best,
        loss_worst=loss_worst,
        loss_std=loss_std,
        parameters=tuple(parameters),
        statistics=stat_summaries,
        wall_time=float(max(stamps)) if stamps else 0.0,
    )


def summarize(
    records: Iterable[MemberRecord],
    param_bounds: Mapping[str, tuple[float, float]] | None = None,
    *,
    pin_threshold: float = DEFAULT_PIN_THRESHOLD,
) -> tuple[IterationSummary, ...]:
    """Group records by iteration and summarise each, in iteration order."""
    grouped: dict[int, list[MemberRecord]] = {}
    for record in records:
        grouped.setdefault(int(record.iteration), []).append(record)
    return tuple(
        summarize_iteration(grouped[i], param_bounds,
                            pin_threshold=pin_threshold)
        for i in sorted(grouped))


def pinned_parameters(
    summaries: Sequence[IterationSummary]) -> tuple[ParameterSummary, ...]:
    """Parameters pinned at a bound in the LATEST iteration.

    Latest only, deliberately: a parameter that touched a bound early and moved
    off it is not the failure mode — one that is sitting there NOW is.
    """
    if not summaries:
        return ()
    return tuple(p for p in summaries[-1].parameters if p.pinned)


# ---------------------------------------------------------------------------
# Rendering 2: TensorBoard scalars
# ---------------------------------------------------------------------------


def tensorboard_points(
    summaries: Sequence[IterationSummary]) -> list[ScalarPoint]:
    """The scalar stream for the event file, derived only from ``summaries``."""
    points: list[ScalarPoint] = []

    def add(tag: str, step: int, value: float | None, wall: float) -> None:
        if value is None:
            return
        points.append(ScalarPoint(tag=tag, step=step, value=float(value),
                                  wall_time=wall))

    for summary in summaries:
        step, wall = summary.iteration, summary.wall_time
        add("loss/ensemble_mean", step, summary.loss_mean, wall)
        add("loss/best_member", step, summary.loss_best, wall)
        add("loss/worst_member", step, summary.loss_worst, wall)
        add("loss/spread_std", step, summary.loss_std, wall)
        add("loss/spread_range", step, summary.loss_spread, wall)
        add("ensemble/blown_up", step, float(len(summary.blown_up)), wall)
        add("ensemble/valid_members", step, float(summary.n_valid), wall)
        for param in summary.parameters:
            add(f"parameters/{param.name}/mean", step, param.mean, wall)
            add(f"parameters/{param.name}/bound_fraction", step,
                param.bound_fraction, wall)
        for stat in summary.statistics:
            add(f"statistics/{stat.name}/contribution_mean", step,
                stat.contribution_mean, wall)
            add(f"statistics/{stat.name}/model_mean", step, stat.model_mean,
                wall)
    return points


def write_tensorboard(directory: str | os.PathLike[str],
                      summaries: Sequence[IterationSummary]) -> str:
    """Write the whole event file for ``summaries``; returns its path."""
    return str(write_scalar_event_file(directory,
                                       tensorboard_points(summaries)))


# ---------------------------------------------------------------------------
# Rendering 3: the static page
# ---------------------------------------------------------------------------

# Categorical slots 1-5 (validated: dataviz validate_palette.js, light and dark,
# ALL CHECKS PASS; worst adjacent CVD dE 9.1 light / 8.4 dark). The light-mode
# contrast WARN on slots 3-5 is relieved as the method requires -- every chart
# below ships a table view and a labelled legend, so no value is colour-only.
_SERIES_LIGHT = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                 "#008300", "#4a3aa7", "#e34948")
_SERIES_DARK = ("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181",
                "#008300", "#9085e9", "#e66767")

# Ordinal ramp for the iteration index on the parameter tracks: one hue,
# light -> dark. Window 250-600 of the blue sequential ramp, which satisfies
# BOTH ordinal floors (light: no lighter than 250; dark: no darker than 600),
# so one ramp serves both modes.
_ITER_RAMP = ("#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
              "#1c5cab", "#184f95")

_PAGE_TITLE = "Calibration campaign"

# A literal character, not the ``&mdash;`` entity: these strings are also used
# inside SVG attribute values (tooltips), where an HTML entity is not XML-clean
# and would show up literally when read back with ``textContent``.
_EM_DASH = "—"


def _tip(*lines: str) -> str:
    """Assemble tooltip text.

    Plain text joined by newlines and rendered with ``textContent`` (CSS
    ``white-space: pre-line``) rather than markup through ``innerHTML``: a
    parameter name arrives from the caller, and there is no reason for the page
    to be able to execute anything it contains.
    """
    return "\n".join(lines)


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _fmt(value: float | None, digits: int = 3) -> str:
    """Human number; an absent value is an em dash, never a misleading 0."""
    if value is None or not _finite(value):
        return _EM_DASH
    magnitude = abs(float(value))
    if magnitude and (magnitude < 1e-3 or magnitude >= 1e5):
        return f"{value:.{digits}e}"
    return f"{value:.{digits}f}"


def _fmt_pct(fraction: float | None) -> str:
    if fraction is None or not _finite(fraction):
        return _EM_DASH
    return f"{100.0 * fraction:.1f}%"


def _series_color(index: int) -> tuple[str, str]:
    """Categorical slot ``index`` (light, dark); never cycles past the 8 slots."""
    if index >= len(_SERIES_LIGHT):
        # The 9th series is not a generated hue: it folds into a neutral
        # "other" tone and the table view carries its identity.
        return ("#898781", "#898781")
    return (_SERIES_LIGHT[index], _SERIES_DARK[index])


def _ramp_color(index: int, total: int) -> str:
    if total <= 1:
        return _ITER_RAMP[-1]
    position = index / (total - 1)
    return _ITER_RAMP[round(position * (len(_ITER_RAMP) - 1))]


def _nice_axis(low: float, high: float) -> tuple[float, float]:
    """A padded [min, max] that is never degenerate (the single-point case)."""
    if not (_finite(low) and _finite(high)):
        return (0.0, 1.0)
    if high - low < 1e-12:
        pad = max(abs(high) * 0.1, 1.0)
        return (low - pad, high + pad)
    pad = (high - low) * 0.08
    return (low - pad, high + pad)


_CSS = """
:root {
  color-scheme: light;
  --surface-1: #fcfcfb; --page: #f9f9f7;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  --good: #0ca30c; --warning: #fab219; --serious: #ec835a; --critical: #d03b3b;
  --danger-wash: rgba(208,59,59,0.13);
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100; --s5: #e87ba4;
  --s6: #008300; --s7: #4a3aa7; --s8: #e34948;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --surface-1: #1a1a19; --page: #0d0d0d;
    --text-primary: #ffffff; --text-secondary: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --danger-wash: rgba(208,59,59,0.22);
    --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181;
    --s6: #008300; --s7: #9085e9; --s8: #e66767;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface-1: #1a1a19; --page: #0d0d0d;
  --text-primary: #ffffff; --text-secondary: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
  --danger-wash: rgba(208,59,59,0.22);
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181;
  --s6: #008300; --s7: #9085e9; --s8: #e66767;
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 28px 22px 60px;
  background: var(--page); color: var(--text-primary);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  font-size: 14px; line-height: 1.5;
}
.wrap { max-width: 1080px; margin: 0 auto; }
h1 { font-size: 22px; margin: 0 0 4px; font-weight: 650; }
h2 { font-size: 15px; margin: 0 0 2px; font-weight: 620; }
.sub { color: var(--text-secondary); font-size: 13px; margin: 0; }
.meta { color: var(--muted); font-size: 12px; margin: 6px 0 0; }
.card {
  background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 10px; padding: 18px 20px; margin: 18px 0;
}
.card > .hint { color: var(--text-secondary); font-size: 12.5px; margin: 2px 0 14px; }
.tiles { display: flex; flex-wrap: wrap; gap: 12px; margin: 18px 0; }
.tile {
  background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 10px; padding: 12px 16px; min-width: 156px; flex: 1 1 156px;
}
.tile .label { color: var(--text-secondary); font-size: 12px; }
.tile .value { font-size: 27px; font-weight: 600; margin-top: 2px; }
.tile .delta { font-size: 12px; color: var(--text-secondary); margin-top: 1px; }
.banner {
  border-radius: 10px; padding: 13px 16px; margin: 18px 0;
  border: 1px solid var(--border); display: flex; gap: 11px; align-items: flex-start;
}
.banner.alarm { background: var(--danger-wash); border-color: var(--critical); }
.banner.ok { background: transparent; }
.banner .icon { font-size: 17px; line-height: 1.3; }
.banner .body { flex: 1; }
.banner .title { font-weight: 640; }
.banner .title.alarm { color: var(--critical); }
.banner .title.ok { color: var(--good); }
.banner ul { margin: 6px 0 0; padding-left: 19px; }
svg { display: block; width: 100%; height: auto; overflow: visible; }
.legend { display: flex; flex-wrap: wrap; gap: 14px; margin: 10px 0 2px;
          font-size: 12.5px; color: var(--text-secondary); }
.legend .item { display: flex; align-items: center; gap: 6px; }
.swatch { width: 11px; height: 11px; border-radius: 3px; flex: none; }
.swatch.line { height: 3px; width: 15px; border-radius: 2px; }
.prow { display: flex; align-items: center; gap: 14px; padding: 9px 10px;
        border-radius: 8px; border-left: 3px solid transparent; }
.prow + .prow { margin-top: 2px; }
.prow.pinned { background: var(--danger-wash); border-left-color: var(--critical); }
.prow .pname { width: 210px; flex: none; font-size: 12.5px; word-break: break-word; }
.prow .pname .bounds { color: var(--muted); font-size: 11px; display: block;
                       font-variant-numeric: tabular-nums; }
.prow .ptrack { flex: 1; min-width: 180px; }
.prow .pval { width: 122px; flex: none; text-align: right;
              font-variant-numeric: tabular-nums; font-size: 12.5px; }
.prow .pval .frac { color: var(--text-secondary); font-size: 11.5px; display: block; }
.badge { display: inline-flex; align-items: center; gap: 4px; border-radius: 5px;
         padding: 1px 7px; font-size: 11px; font-weight: 640; white-space: nowrap; }
.badge.crit { background: var(--critical); color: #fff; }
.badge.warn { background: var(--warning); color: #2a2200; }
.badge.none { background: transparent; color: var(--muted); }
.chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 4px; }
.chip { border: 1px solid var(--border); border-radius: 8px; padding: 7px 11px;
        font-size: 12.5px; }
.chip.bad { border-color: var(--critical); background: var(--danger-wash); }
.chip .n { font-weight: 640; }
.chip .who { color: var(--text-secondary); font-size: 11.5px; display: block; }
details { margin-top: 14px; }
summary { cursor: pointer; color: var(--text-secondary); font-size: 12.5px;
          padding: 3px 0; }
table { border-collapse: collapse; width: 100%; margin-top: 8px; font-size: 12.5px;
        font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: 5px 9px; border-bottom: 1px solid var(--grid); }
th:first-child, td:first-child { text-align: left; font-variant-numeric: normal; }
th { color: var(--text-secondary); font-weight: 570; }
.empty { color: var(--text-secondary); padding: 26px 0; text-align: center; }
.footer { color: var(--muted); font-size: 11.5px; margin-top: 26px;
          border-top: 1px solid var(--border); padding-top: 12px; }
.footer code { font-size: 11.5px; }
#tt { position: fixed; pointer-events: none; opacity: 0; transition: opacity .09s;
      white-space: pre-line;
      background: var(--surface-1); color: var(--text-primary); font-size: 12px;
      border: 1px solid var(--border); border-radius: 7px; padding: 6px 9px;
      box-shadow: 0 3px 12px rgba(0,0,0,.17); z-index: 20; max-width: 280px; }
#theme { position: absolute; top: 26px; right: 22px; background: var(--surface-1);
         color: var(--text-secondary); border: 1px solid var(--border);
         border-radius: 7px; padding: 5px 11px; font-size: 12px; cursor: pointer; }
"""

_JS = """
(function () {
  var tip = document.getElementById('tt');
  function show(e) {
    var t = e.target.closest('[data-tip]');
    if (!t) return;
    tip.textContent = t.getAttribute('data-tip');
    tip.style.opacity = '1';
    move(e);
  }
  function move(e) {
    var x = e.clientX + 14, y = e.clientY + 16;
    var r = tip.getBoundingClientRect();
    if (x + r.width > window.innerWidth - 8) x = e.clientX - r.width - 12;
    if (y + r.height > window.innerHeight - 8) y = e.clientY - r.height - 12;
    tip.style.left = x + 'px'; tip.style.top = y + 'px';
  }
  document.addEventListener('mouseover', show);
  document.addEventListener('mousemove', function (e) {
    if (tip.style.opacity === '1') move(e);
  });
  document.addEventListener('mouseout', function (e) {
    if (e.target.closest('[data-tip]')) tip.style.opacity = '0';
  });
  // Keyboard parity: focusing a mark shows the same content as hover.
  document.addEventListener('focusin', function (e) {
    var t = e.target.closest('[data-tip]');
    if (!t) return;
    var b = t.getBoundingClientRect();
    tip.textContent = t.getAttribute('data-tip');
    tip.style.opacity = '1';
    tip.style.left = b.right + 10 + 'px';
    tip.style.top = b.bottom + 8 + 'px';
  });
  document.addEventListener('focusout', function () { tip.style.opacity = '0'; });
  var btn = document.getElementById('theme');
  btn.addEventListener('click', function () {
    var root = document.documentElement;
    var dark = getComputedStyle(root).colorScheme.indexOf('dark') >= 0;
    root.setAttribute('data-theme', dark ? 'light' : 'dark');
  });
})();
"""


# --- SVG chart builders -----------------------------------------------------
# Geometry is fixed in a viewBox and scaled by CSS, so the page is responsive
# without any layout maths at render time. Every container is sized to INCLUDE
# its axis band (no nested scrollbars).

_LOSS_W, _LOSS_H = 760, 320
_LOSS_ML, _LOSS_MR, _LOSS_MT, _LOSS_MB = 66, 104, 16, 44
_STACK_W, _STACK_H = 760, 300
_STACK_ML, _STACK_MR, _STACK_MT, _STACK_MB = 66, 24, 16, 44
_TRACK_W, _TRACK_H = 620, 44
_TRACK_X0, _TRACK_X1 = 6, 614


def _x_positions(count: int, left: float, width: float) -> list[float]:
    """Evenly spaced x for a line chart; a single point sits in the middle."""
    if count <= 0:
        return []
    if count == 1:
        return [left + width / 2.0]
    return [left + i * width / (count - 1) for i in range(count)]


def _y_scale(value: float, low: float, high: float,
             top: float, height: float) -> float:
    return top + height * (1.0 - (value - low) / (high - low))


def _segments(points: Sequence[tuple[float, float] | None]
              ) -> list[list[tuple[float, float]]]:
    """Split a point list on ``None`` gaps so a dead iteration breaks the line
    rather than being interpolated across (which would invent a value)."""
    runs: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    for point in points:
        if point is None:
            if current:
                runs.append(current)
            current = []
        else:
            current.append(point)
    if current:
        runs.append(current)
    return runs


def _gridlines(low: float, high: float, left: float, width: float,
               top: float, height: float, ticks: int = 4) -> str:
    out = []
    for i in range(ticks + 1):
        value = low + (high - low) * i / ticks
        y = _y_scale(value, low, high, top, height)
        out.append(
            f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{left + width:.1f}" '
            f'y2="{y:.1f}" stroke="var(--grid)" stroke-width="1"/>'
            f'<text x="{left - 9:.1f}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="var(--muted)">{_fmt(value, 2)}</text>')
    return "".join(out)


def _iteration_axis(xs: Sequence[float], iterations: Sequence[int],
                    baseline_y: float, left: float, width: float) -> str:
    out = [f'<line x1="{left:.1f}" y1="{baseline_y:.1f}" '
           f'x2="{left + width:.1f}" y2="{baseline_y:.1f}" '
           f'stroke="var(--axis)" stroke-width="1"/>']
    for x, iteration in zip(xs, iterations):
        out.append(
            f'<text x="{x:.1f}" y="{baseline_y + 18:.1f}" text-anchor="middle" '
            f'font-size="11" fill="var(--muted)">{iteration}</text>')
    out.append(
        f'<text x="{left + width / 2:.1f}" y="{baseline_y + 36:.1f}" '
        f'text-anchor="middle" font-size="11.5" fill="var(--text-secondary)">'
        f'iteration</text>')
    return "".join(out)


def _loss_chart(summaries: Sequence[IterationSummary]) -> str:
    """Ensemble mean, best member and the min-max spread band per iteration.

    The band is the convergence signal the reader is here for, so it is drawn
    first and largest; the two lines sit on top of it.
    """
    plot_w = _LOSS_W - _LOSS_ML - _LOSS_MR
    plot_h = _LOSS_H - _LOSS_MT - _LOSS_MB
    xs = _x_positions(len(summaries), _LOSS_ML, plot_w)
    values = [v for s in summaries
              for v in (s.loss_mean, s.loss_best, s.loss_worst)
              if v is not None and _finite(v)]
    if not values:
        return ('<p class="empty">No surviving members yet &mdash; every '
                'member logged so far blew up.</p>')
    low, high = _nice_axis(min(values), max(values))
    baseline_y = _LOSS_MT + plot_h
    parts = [f'<svg viewBox="0 0 {_LOSS_W} {_LOSS_H}" role="img" '
             f'aria-label="Loss per iteration: ensemble mean, best member and '
             f'member spread">']
    parts.append(_gridlines(low, high, _LOSS_ML, plot_w, _LOSS_MT, plot_h))

    def y_of(value: float | None) -> float | None:
        if value is None or not _finite(value):
            return None
        return _y_scale(float(value), low, high, _LOSS_MT, plot_h)

    # Spread band (min-max across surviving members). Drawn as one polygon per
    # CONTIGUOUS run of live iterations, exactly like the lines: an iteration
    # in which every member blew up leaves a real gap, and a band stretched
    # across it would draw a spread the campaign never measured. A single-point
    # run has no polygon and becomes a whisker (below) instead.
    band_pts = [
        (x, y_of(s.loss_best), y_of(s.loss_worst))
        for x, s in zip(xs, summaries)]
    band_run: list[tuple[float, float, float]] = []
    for point in [*band_pts, (0.0, None, None)]:      # sentinel flushes the last
        x, lo, hi = point
        if lo is not None and hi is not None:
            band_run.append((x, lo, hi))
            continue
        if len(band_run) >= 2:
            top_edge = " ".join(f"{px:.1f},{plo:.1f}" for px, plo, _ in band_run)
            bottom_edge = " ".join(
                f"{px:.1f},{phi:.1f}" for px, _, phi in reversed(band_run))
            parts.append(f'<polygon points="{top_edge} {bottom_edge}" '
                         f'fill="var(--s1)" fill-opacity="0.16"/>')
        band_run = []
    for x, lo, hi in band_pts:
        if lo is None or hi is None or abs(hi - lo) < 0.5:
            continue
        parts.append(f'<line x1="{x:.1f}" y1="{lo:.1f}" x2="{x:.1f}" '
                     f'y2="{hi:.1f}" stroke="var(--s1)" stroke-opacity="0.42" '
                     f'stroke-width="2"/>')

    for key, css_var, label in (("loss_mean", "--s1", "ensemble mean"),
                                ("loss_best", "--s2", "best member")):
        series = [
            (x, y_of(getattr(s, key)))
            for x, s in zip(xs, summaries)]
        runs = _segments([(x, y) if y is not None else None for x, y in series])
        for run in runs:
            if len(run) >= 2:
                path = " ".join(f"{x:.1f},{y:.1f}" for x, y in run)
                parts.append(
                    f'<polyline points="{path}" fill="none" '
                    f'stroke="var({css_var})" stroke-width="2" '
                    f'stroke-linejoin="round" stroke-linecap="round"/>')
        for (x, y), summary in zip(series, summaries):
            if y is None:
                continue
            value = getattr(summary, key)
            tip = _tip(f"iteration {summary.iteration}",
                       f"{label}: {_fmt(value, 4)}",
                       f"{summary.n_valid} of {summary.n_members} "
                       f"members valid")
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" '
                f'fill="var({css_var})" stroke="var(--surface-1)" '
                f'stroke-width="2" tabindex="0" data-tip="{tip}"/>')
        # Direct endpoint label (selective labelling: the endpoint only).
        last = next(((x, y) for x, y in reversed(series) if y is not None),
                    None)
        if last is not None:
            last_x, last_y = last
            parts.append(
                f'<text x="{last_x + 11:.1f}" y="{last_y + 4:.1f}" '
                f'font-size="11.5" fill="var(--text-secondary)">'
                f'{_esc(label)}</text>')

    parts.append(_iteration_axis(xs, [s.iteration for s in summaries],
                                 baseline_y, _LOSS_ML, plot_w))
    parts.append("</svg>")
    legend = _legend([("var(--s1)", "ensemble mean", True),
                      ("var(--s2)", "best member", True),
                      ("var(--s1)", "member spread (min–max)", False)])
    return legend + "".join(parts)


def _legend(items: Sequence[tuple[str, str, bool]]) -> str:
    """``(color, label, is_line)`` -> a legend row. Always present for >= 2."""
    cells = []
    for color, label, is_line in items:
        cls = "swatch line" if is_line else "swatch"
        opacity = "" if is_line else ' style="opacity:.3"'
        cells.append(
            f'<span class="item"><span class="{cls}" '
            f'style="background:{color}"{opacity}></span>{_esc(label)}</span>')
    return f'<div class="legend">{"".join(cells)}</div>'


def _parameter_track(name: str, history: Sequence[
        tuple[int, ParameterSummary | None]], total_iterations: int,
        pin_threshold: float) -> str:
    """One parameter's row: its position within bounds across all iterations."""
    latest = next((p for _, p in reversed(history) if p is not None), None)
    if latest is None:
        return ""
    span = _TRACK_X1 - _TRACK_X0
    mid = _TRACK_H / 2.0
    parts = [f'<svg viewBox="0 0 {_TRACK_W} {_TRACK_H}" role="img" '
             f'aria-label="{_esc(name)} position within its allowed range">']
    if latest.lo is None or latest.hi is None:
        parts.append(
            f'<text x="{_TRACK_X0}" y="{mid + 4:.1f}" font-size="11.5" '
            f'fill="var(--muted)">no registered bounds &mdash; '
            f'cannot check for pinning</text></svg>')
        return "".join(parts)

    # Track + the two danger zones. The zones are drawn as part of the scale so
    # a marker entering one is unmistakable without reading a number.
    parts.append(
        f'<rect x="{_TRACK_X0}" y="{mid - 7:.1f}" width="{span}" height="14" '
        f'rx="7" fill="var(--grid)"/>')
    zone = span * pin_threshold
    for zone_x in (_TRACK_X0, _TRACK_X1 - zone):
        parts.append(
            f'<rect x="{zone_x:.1f}" y="{mid - 7:.1f}" width="{zone:.1f}" '
            f'height="14" fill="var(--critical)" fill-opacity="0.30"/>')

    def x_of(fraction: float | None) -> float | None:
        if fraction is None or not _finite(fraction):
            return None
        # Clamped for DRAWING only; the table view carries the raw fraction, so
        # an out-of-bounds value is never silently redrawn as merely pinned.
        return _TRACK_X0 + span * min(max(fraction, 0.0), 1.0)

    # Ensemble min-max whisker at the latest iteration.
    x_min, x_max = x_of(latest.fraction_min), x_of(latest.fraction_max)
    if x_min is not None and x_max is not None and x_max - x_min > 1.0:
        parts.append(
            f'<rect x="{x_min:.1f}" y="{mid - 4:.1f}" '
            f'width="{x_max - x_min:.1f}" height="8" rx="4" '
            f'fill="var(--s1)" fill-opacity="0.30"/>')

    trajectory = [(x_of(p.bound_fraction), i, p)
                  for i, p in history if p is not None]
    drawn = [(x, i, p) for x, i, p in trajectory if x is not None]
    if len(drawn) >= 2:
        path = " ".join(f"{x:.1f},{mid:.1f}" for x, _, _ in drawn)
        parts.append(f'<polyline points="{path}" fill="none" '
                     f'stroke="var(--text-secondary)" stroke-opacity="0.45" '
                     f'stroke-width="2"/>')
    for x, iteration, param in drawn:
        color = _ramp_color(iteration, max(total_iterations, 1))
        out_of_range = (param.bound_fraction is not None
                        and not 0.0 <= param.bound_fraction <= 1.0)
        tip = _tip(
            _esc(name), f"iteration {iteration}",
            f"mean {_fmt(param.mean, 5)}",
            f"position {_fmt_pct(param.bound_fraction)} of range",
            *(("OUTSIDE ITS REGISTERED BOUNDS",) if out_of_range else ()))
        parts.append(
            f'<circle cx="{x:.1f}" cy="{mid:.1f}" r="5" fill="{color}" '
            f'stroke="var(--surface-1)" stroke-width="2" tabindex="0" '
            f'data-tip="{tip}"/>')
    if drawn:
        x, iteration, _ = drawn[-1]
        parts.append(
            f'<text x="{x:.1f}" y="{mid - 12:.1f}" text-anchor="middle" '
            f'font-size="10.5" fill="var(--text-secondary)">it {iteration}'
            f'</text>')
    parts.append("</svg>")
    return "".join(parts)


def _stat_stack(summaries: Sequence[IterationSummary]) -> str:
    """Stacked per-statistic contribution to the ensemble-mean loss."""
    names: list[str] = []
    for summary in summaries:
        for stat in summary.statistics:
            if stat.fitted and stat.name not in names:
                names.append(stat.name)
    if not names:
        return ('<p class="empty">No fitted statistics logged yet.</p>')
    plot_w = _STACK_W - _STACK_ML - _STACK_MR
    plot_h = _STACK_H - _STACK_MT - _STACK_MB
    # Only POSITIVE contributions are drawn, so only positive contributions set
    # the axis; including a negative one would leave the stack taller than its
    # own maximum and (with overflow visible) painting outside the card.
    totals = []
    for summary in summaries:
        by_name = {s.name: s for s in summary.statistics}
        totals.append(sum(
            by_name[n].contribution_mean for n in names
            if n in by_name and _finite(by_name[n].contribution_mean)
            and by_name[n].contribution_mean > 0.0))
    top = max(totals) if totals else 0.0
    if top <= 0.0:
        return ('<p class="empty">Every fitted contribution logged so far is '
                'zero or missing.</p>')
    low, high = 0.0, top * 1.08
    baseline_y = _STACK_MT + plot_h
    band = plot_w / max(len(summaries), 1)
    bar_w = min(48.0, band * 0.58)
    parts = [f'<svg viewBox="0 0 {_STACK_W} {_STACK_H}" role="img" '
             f'aria-label="Per-statistic contribution to the ensemble-mean '
             f'loss, by iteration">']
    parts.append(_gridlines(low, high, _STACK_ML, plot_w, _STACK_MT, plot_h))
    centers = []
    for index, summary in enumerate(summaries):
        center = _STACK_ML + (index + 0.5) * band
        centers.append(center)
        by_name = {s.name: s for s in summary.statistics}
        cursor = 0.0
        for slot, name in enumerate(names):
            entry = by_name.get(name)
            if entry is None or not _finite(entry.contribution_mean):
                continue
            value = float(entry.contribution_mean)
            if value <= 0.0:
                continue
            y_top = _y_scale(cursor + value, low, high, _STACK_MT, plot_h)
            y_bottom = _y_scale(cursor, low, high, _STACK_MT, plot_h)
            # 2px surface gap between stacked segments (never a border).
            height = max(y_bottom - y_top - 2.0, 0.8)
            color = f"var(--s{slot + 1})" if slot < 8 else "var(--muted)"
            tip = _tip(f"iteration {summary.iteration}", _esc(name),
                       f"contribution {_fmt(value, 4)}",
                       f"{_fmt_pct(entry.share)} of the decomposed loss",
                       f"model {_fmt(entry.model_mean, 4)} vs observed "
                       f"{_fmt(entry.observed, 4)}")
            parts.append(
                f'<rect x="{center - bar_w / 2:.1f}" y="{y_top:.1f}" '
                f'width="{bar_w:.1f}" height="{height:.1f}" rx="2" '
                f'fill="{color}" tabindex="0" data-tip="{tip}"/>')
            cursor += value
    parts.append(_iteration_axis(centers, [s.iteration for s in summaries],
                                 baseline_y, _STACK_ML, plot_w))
    parts.append("</svg>")
    legend = _legend([
        (f"var(--s{i + 1})" if i < 8 else "var(--muted)", name, False)
        for i, name in enumerate(names)])
    return legend + "".join(parts)


# --- page assembly ----------------------------------------------------------


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]],
           caption: str = "Table view") -> str:
    """The WCAG-clean twin of every chart: no value is colour-only.

    Cells are escaped HERE, symmetrically with the headers, so callers pass
    plain text and the next row added cannot become an injection point by
    forgetting to escape. That means cells must NOT be pre-escaped and must not
    carry HTML entities — use ``_EM_DASH``, not ``&mdash;``.
    """
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_esc(cell)}</td>" for cell in row) + "</tr>"
        for row in rows)
    return (f"<details><summary>{_esc(caption)}</summary><table><thead><tr>"
            f"{head}</tr></thead><tbody>{body}</tbody></table></details>")


def _tile(label: str, value: str, delta: str = "") -> str:
    delta_html = f'<div class="delta">{delta}</div>' if delta else ""
    return (f'<div class="tile"><div class="label">{_esc(label)}</div>'
            f'<div class="value">{value}</div>{delta_html}</div>')


def _delta_text(current: float | None, previous: float | None,
                lower_is_better: bool = True) -> str:
    if current is None or previous is None:
        return ""
    change = current - previous
    if abs(change) < 1e-12:
        return "unchanged since last iteration"
    good = (change < 0) if lower_is_better else (change > 0)
    arrow = "&#9660;" if change < 0 else "&#9650;"
    word = "better" if good else "worse"
    return f"{arrow} {_fmt(abs(change), 3)} {word} than the previous iteration"


def _banner(summaries: Sequence[IterationSummary],
            pin_threshold: float) -> str:
    """The one thing on the page a reader must not be able to miss."""
    if not summaries:
        return ('<div class="banner ok"><div class="icon">&#9675;</div>'
                '<div class="body"><div class="title">Awaiting the first '
                'iteration</div><div>No members have been logged yet. This '
                'page refreshes itself after every iteration.</div>'
                '</div></div>')
    pinned = pinned_parameters(summaries)
    latest = summaries[-1]
    if pinned:
        items = "".join(
            f"<li><b>{_esc(p.name)}</b> is at its <b>{p.pinned}</b> bound "
            f"({_fmt(p.mean, 5)}, {_fmt_pct(p.bound_fraction)} of the "
            f"{_fmt(p.lo, 4)}&ndash;{_fmt(p.hi, 4)} range)</li>"
            for p in pinned)
        return (
            '<div class="banner alarm"><div class="icon">&#9888;</div>'
            '<div class="body"><div class="title alarm">'
            f'{len(pinned)} parameter(s) pinned at a bound at iteration '
            f'{latest.iteration}</div>'
            f'<ul>{items}</ul>'
            '<div>A parameter jammed against its registered range means the '
            'optimiser is chasing a bias no parameter can close. Check the '
            'observation-error floor for the dominant statistic before '
            'spending another iteration.</div></div></div>')
    return ('<div class="banner ok"><div class="icon">&#10003;</div>'
            '<div class="body"><div class="title ok">No parameter is at a '
            f'bound</div><div>Every parameter sits more than '
            f'{100 * pin_threshold:.0f}% of its allowed range away from both '
            'limits at the latest iteration.</div></div></div>')


def _loss_section(summaries: Sequence[IterationSummary]) -> str:
    if not summaries:
        return ""
    rows = [[str(s.iteration), _fmt(s.loss_mean, 4), _fmt(s.loss_best, 4),
             _fmt(s.loss_worst, 4), _fmt(s.loss_spread, 4),
             _fmt(s.loss_std, 4), f"{s.n_valid}/{s.n_members}"]
            for s in summaries]
    notes = []
    # The spread is compared against its own MINIMUM, not against the first
    # iteration: the prior spread is always large, so "smaller than iteration
    # 0" is true even for an ensemble that has started re-expanding. What
    # matters is whether it is growing NOW.
    spreads = [s.loss_spread for s in summaries
               if s.loss_spread is not None and _finite(s.loss_spread)]
    if len(spreads) >= 3:
        floor = min(spreads)
        if floor > 0.0 and spreads[-1] > 1.2 * floor:
            notes.append(
                '<p class="hint"><b>The spread has re-widened to '
                f'{spreads[-1] / floor:.1f}&times; its smallest value.</b> The '
                'ensemble stopped collapsing on a consensus and is spreading '
                'out again &mdash; the optimiser is thrashing, and further '
                'iterations may not be worth their GPU-hours.</p>')
        elif spreads[-1] < 0.5 * spreads[0]:
            notes.append('<p class="hint">The spread is still contracting '
                         '&mdash; the ensemble is converging.</p>')
    means = [(s.iteration, s.loss_mean) for s in summaries
             if s.loss_mean is not None and _finite(s.loss_mean)]
    if len(means) >= 2 and means[-1][1] > means[-2][1]:
        best_it, best_val = min(means, key=lambda pair: pair[1])
        notes.append(
            '<p class="hint"><b>The ensemble-mean loss went UP at iteration '
            f'{means[-1][0]}</b> ({_fmt(means[-2][1], 4)} &rarr; '
            f'{_fmt(means[-1][1], 4)}). The best iteration so far is '
            f'{best_it} at {_fmt(best_val, 4)}.</p>')
    spread_note = "".join(notes)
    return (
        '<section class="card"><h2>Loss</h2>'
        '<p class="hint">Ensemble mean, best member, and the min&ndash;max '
        'spread across surviving members. A narrowing band is convergence; a '
        'band that stays wide or grows is the signal to stop spending.</p>'
        + _loss_chart(summaries) + spread_note
        + _table(["iteration", "mean", "best", "worst", "spread (max-min)",
                  "std", "valid members"], rows)
        + "</section>")


def _parameter_section(summaries: Sequence[IterationSummary],
                       pin_threshold: float) -> str:
    if not summaries:
        return ""
    names: list[str] = []
    for summary in summaries:
        for param in summary.parameters:
            if param.name not in names:
                names.append(param.name)
    if not names:
        return ""
    total_iterations = max(s.iteration for s in summaries) + 1
    rows_html = []
    for name in names:
        history = [(s.iteration,
                    next((p for p in s.parameters if p.name == name), None))
                   for s in summaries]
        latest = next((p for _, p in reversed(history) if p is not None), None)
        if latest is None:
            continue
        if latest.pinned:
            badge = (f'<span class="badge crit">&#9888; PINNED AT '
                     f'{latest.pinned.upper()} BOUND</span>')
        elif latest.lo is None:
            badge = '<span class="badge warn">? no bounds</span>'
        else:
            badge = ""
        bounds = ("no registered bounds" if latest.lo is None
                  else f"{_fmt(latest.lo, 4)} &ndash; {_fmt(latest.hi, 4)}")
        rows_html.append(
            f'<div class="prow{" pinned" if latest.pinned else ""}">'
            f'<div class="pname">{_esc(name)}'
            f'<span class="bounds">{bounds}</span></div>'
            f'<div class="ptrack">'
            f'{_parameter_track(name, history, total_iterations, pin_threshold)}'
            f'</div>'
            f'<div class="pval">{_fmt(latest.mean, 5)}'
            f'<span class="frac">{_fmt_pct(latest.bound_fraction)} of range'
            f'</span>{badge}</div></div>')
    table_rows = []
    for summary in summaries:
        for param in summary.parameters:
            table_rows.append([
                str(summary.iteration), param.name, _fmt(param.mean, 6),
                _fmt(param.minimum, 6), _fmt(param.maximum, 6),
                _fmt(param.lo, 4), _fmt(param.hi, 4),
                _fmt_pct(param.bound_fraction),
                (param.pinned.upper() if param.pinned else _EM_DASH)])
    ramp = "".join(
        f'<span class="item"><span class="swatch" style="background:'
        f'{_ramp_color(s.iteration, total_iterations)}"></span>'
        f'{s.iteration}</span>' for s in summaries)
    return (
        '<section class="card"><h2>Parameters within their allowed range</h2>'
        '<p class="hint">Each track spans the parameter\'s registered '
        '<code>__param_spec__</code> bounds, left = lower, right = upper. The '
        'red zones at each end are the '
        f'{100 * pin_threshold:.0f}% pinning margin; the dots are the ensemble '
        'mean at each iteration and the pale bar is the current ensemble '
        'min&ndash;max. A dot walking into a red zone is the failure this '
        'campaign is watching for.</p>'
        f'<div class="legend"><span class="item">iteration:</span>{ramp}'
        f'<span class="item"><span class="swatch" style="background:var(--s1);'
        f'opacity:.3"></span>ensemble min&ndash;max (latest)</span></div>'
        + "".join(rows_html)
        + _table(["iteration", "parameter", "ensemble mean", "min", "max",
                  "lower bound", "upper bound", "position in range",
                  "pinned"], table_rows)
        + "</section>")


def _statistic_section(summaries: Sequence[IterationSummary]) -> str:
    if not summaries:
        return ""
    latest = summaries[-1]
    fitted = [s for s in latest.statistics if s.fitted]
    diagnostics = [s for s in latest.statistics if not s.fitted]
    if not fitted and not diagnostics:
        return ""
    note = ""
    if fitted:
        dominant = max(fitted, key=lambda s: s.share)
        if dominant.share > DOMINANCE_SHARE:
            note = (
                f'<p class="hint"><b>{_esc(dominant.name)} is '
                f'{_fmt_pct(dominant.share)} of the decomposed loss.</b> '
                'With one statistic this dominant the campaign is a '
                'single-variable fit '

                'in all but name &mdash; the other statistics are along for '
                'the ride, not constraining the parameters.</p>')
    residual = latest.undecomposed_loss
    if (residual is not None and latest.loss_mean
            and abs(residual) > 0.01 * abs(latest.loss_mean)):
        note += (
            f'<p class="hint">The fitted statistics account for '
            f'{_fmt(latest.decomposed_total, 4)} of the {_fmt(latest.loss_mean, 4)} '
            f'ensemble-mean loss; {_fmt(residual, 4)} is NOT decomposed by any '
            'of them. Percentages below are shares of the decomposition, not '
            'of the loss.</p>')
    rows = []
    for summary in summaries:
        for stat in summary.statistics:
            rows.append([
                str(summary.iteration), stat.name,
                ("fitted" if stat.fitted else "diagnostic only"),
                _fmt(stat.model_mean, 4), _fmt(stat.observed, 4),
                _fmt(stat.model_mean - stat.observed, 4),
                _fmt(stat.contribution_mean, 4),
                (_fmt_pct(stat.share) if stat.fitted else _EM_DASH)])
    diag_html = ""
    if diagnostics:
        chips = "".join(
            f'<div class="chip"><span class="n">{_esc(s.name)}</span> '
            f'{_fmt(s.model_mean, 3)} vs {_fmt(s.observed, 3)} observed'
            f'<span class="who">bias '
            f'{_fmt(s.model_mean - s.observed, 3)} &mdash; reported, NOT '
            f'fitted</span></div>' for s in diagnostics)
        diag_html = (
            '<p class="hint" style="margin-top:16px">Diagnostic statistics '
            '&mdash; deliberately excluded from the fitted vector, shown so '
            'the headline number is never lost:</p>'
            f'<div class="chips">{chips}</div>')
    return (
        '<section class="card"><h2>What is driving the fit</h2>'
        '<p class="hint">Each statistic\'s contribution to the ensemble-mean '
        'loss, stacked per iteration. Segment height is how much of the loss '
        'that measurement owns.</p>'
        + _stat_stack(summaries) + note + diag_html
        + _table(["iteration", "statistic", "role", "model (ens. mean)",
                  "observed", "bias", "contribution", "share of decomposed loss"],
                 rows)
        + "</section>")


def _blowup_section(summaries: Sequence[IterationSummary]) -> str:
    if not summaries:
        return ""
    chips = []
    for summary in summaries:
        count = len(summary.blown_up)
        who = (", ".join(str(m) for m in summary.blown_up)
               if summary.blown_up else "none")
        chips.append(
            f'<div class="chip{" bad" if count else ""}">'
            f'<span class="n">iteration {summary.iteration}: {count}/'
            f'{summary.n_members}</span>'
            f'<span class="who">members: {_esc(who)}</span></div>')
    counts = [len(s.blown_up) for s in summaries]
    if all(count == 0 for count in counts):
        note = '<p class="hint">No member has blown up.</p>'
    elif len(counts) >= 2 and counts[-1] > 0 and counts[-1] >= max(counts[:-1]):
        # Rising OR at its worst right now: either way the ensemble is being
        # pushed further into the unstable region every iteration.
        note = ('<p class="hint"><b>The blow-up count is at its highest in the '
                'latest iteration.</b> The optimiser is walking the ensemble '
                'into an unstable region of parameter space.</p>')
    else:
        worst = max(range(len(counts)), key=lambda i: counts[i])
        note = ('<p class="hint"><b>'
                f'{counts[worst]} of {summaries[worst].n_members} members blew '
                f'up at iteration {summaries[worst].iteration}.</b> Blown-up '
                'members are dropped from the ensemble update, so the '
                'effective ensemble there was smaller than it looks &mdash; '
                + (f'the update was computed from {summaries[worst].n_valid} '
                   'members.</p>'
                   if summaries[worst].n_valid >= 2 else
                   f'only {summaries[worst].n_valid} survived, which is too '
                   'few for an ensemble update at all (ETKI raises below '
                   'two).</p>'))
    return ('<section class="card"><h2>Blown-up members</h2>'
            '<p class="hint">Members whose simulation produced a non-finite '
            'score. They are dropped from the ensemble update rather than '
            'crashing the campaign &mdash; correct, but silent, so they are '
            'counted here.</p>'
            f'<div class="chips">{"".join(chips)}</div>{note}</section>')


def render_html(
    summaries: Sequence[IterationSummary],
    *,
    title: str = _PAGE_TITLE,
    run_label: str = "",
    source_path: str = "",
    pin_threshold: float = DEFAULT_PIN_THRESHOLD,
    generated_at: float | None = None,
) -> str:
    """Build the whole self-contained page.

    Renders correctly for ZERO summaries (an "awaiting the first iteration"
    page) and for ONE (charts degrade to single markers, never a divide by
    ``n - 1``). Both cases are covered by tests, because a campaign is most
    likely to be looked at when it has just started.
    """
    stamp = time.strftime("%Y-%m-%d %H:%M:%S %Z",
                          time.localtime(generated_at if generated_at
                                         is not None else time.time()))
    total_members = sum(s.n_members for s in summaries)
    total_blown = sum(len(s.blown_up) for s in summaries)
    meta = (f"{len(summaries)} iteration(s) &middot; {total_members} member "
            f"evaluation(s) &middot; {total_blown} blown up &middot; "
            f"generated {_esc(stamp)}")

    tiles = ""
    if summaries:
        latest = summaries[-1]
        previous = summaries[-2] if len(summaries) >= 2 else None
        best_ever = min((s.loss_best for s in summaries
                         if s.loss_best is not None), default=None)
        tiles = (
            '<div class="tiles">'
            + _tile("ensemble-mean loss", _fmt(latest.loss_mean, 4),
                    _delta_text(latest.loss_mean,
                                previous.loss_mean if previous else None))
            + _tile("best member (this iteration)", _fmt(latest.loss_best, 4),
                    f"best ever {_fmt(best_ever, 4)}")
            # A literal minus sign, not "&minus;": _tile escapes its label, so
            # an entity here would render as the text "&minus;".
            + _tile("spread (max − min)", _fmt(latest.loss_spread, 4),
                    _delta_text(latest.loss_spread,
                                previous.loss_spread if previous else None))
            + _tile("valid members",
                    f"{latest.n_valid}/{latest.n_members}",
                    (f"{len(latest.blown_up)} blew up"
                     if latest.blown_up else "all members finite"))
            + "</div>")

    body = (tiles
            + _banner(summaries, pin_threshold)
            + _parameter_section(summaries, pin_threshold)
            + _loss_section(summaries)
            + _statistic_section(summaries)
            + _blowup_section(summaries))
    if not summaries:
        body += ('<section class="card"><p class="empty">Nothing logged yet. '
                 'Charts appear as soon as the first iteration completes.</p>'
                 '</section>')
    source = (f'Source of truth: <code>{_esc(source_path)}</code>. '
              if source_path else "")
    return (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
        "<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, "
        "initial-scale=1\">\n"
        f"<title>{_esc(title)}</title>\n<style>{_CSS}</style>\n</head>\n<body>\n"
        '<button id="theme" type="button">light / dark</button>\n'
        '<div class="wrap">\n'
        f'<h1>{_esc(title)}</h1>\n'
        f'<p class="sub">{_esc(run_label) if run_label else ""}</p>\n'
        f'<p class="meta">{meta}</p>\n'
        f"{body}\n"
        f'<p class="footer">{source}This page and the TensorBoard event files '
        'are both rendered FROM that log, so they cannot disagree with it; '
        'regenerate either at any time with '
        '<code>scripts/plot/render_calibration_dashboard.py</code>.</p>\n'
        "</div>\n"
        '<div id="tt" role="status" aria-live="polite"></div>\n'
        f"<script>{_JS}</script>\n</body>\n</html>\n")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def resolve_param_bounds(
    names: Iterable[str] | None = None) -> dict[str, tuple[float, float]]:
    """Registered ``__param_spec__`` bounds, keyed by ``scheme_key.field``.

    Imported lazily because :mod:`legoesm.training.param_collector` pulls in
    JAX; this module must stay importable on a login node with no device. Array
    -shaped parameters are skipped: a per-PFT vector has no single position in
    a scalar range, and inventing one would be a lie on the page.

    ``names`` restricts the result (and reports anything unregistered, so a
    typo'd parameter name is loud rather than silently unbounded).
    """
    from legoesm.training.param_collector import build_registry

    bounds: dict[str, tuple[float, float]] = {}
    for meta in build_registry():
        low, high = meta.bounds
        if isinstance(low, (list, tuple)) or isinstance(high, (list, tuple)):
            continue
        bounds[meta.qualified_name] = (float(low), float(high))
    if names is None:
        return bounds
    wanted = list(names)
    missing = [n for n in wanted if n not in bounds]
    if missing:
        print(f"[calibration_tracking] no registered bounds for {missing}; "
              "these parameters cannot be checked for pinning",
              file=sys.stderr)
    return {n: bounds[n] for n in wanted if n in bounds}


def render_dashboard(
    run_dir: str | os.PathLike[str],
    param_bounds: Mapping[str, tuple[float, float]] | None = None,
    *,
    title: str = _PAGE_TITLE,
    run_label: str = "",
    pin_threshold: float = DEFAULT_PIN_THRESHOLD,
    write_events: bool = True,
) -> str:
    """Re-derive the page (and event files) from the JSONL; returns the path.

    This is the ONLY way either rendering is produced — during a live campaign
    and from the CLI afterwards — which is what keeps them from disagreeing
    with the log or with each other.

    The PAGE is written first and the event files second, deliberately. The
    page is the output most people look at, so a failure in the secondary
    rendering must not be able to cost it; the reverse ordering meant one
    unrepresentable scalar took both, leaving the dashboard frozen for the rest
    of a multi-day campaign. For the same reason an event-file failure is
    reported and swallowed here rather than propagated.
    """
    run_dir = os.fspath(run_dir)
    os.makedirs(run_dir, exist_ok=True)
    jsonl = os.path.join(run_dir, MEMBERS_FILENAME)
    records = read_member_records(jsonl)
    summaries = summarize(records, param_bounds, pin_threshold=pin_threshold)

    page = render_html(summaries, title=title, run_label=run_label,
                       source_path=jsonl, pin_threshold=pin_threshold)
    path = os.path.join(run_dir, DASHBOARD_FILENAME)
    # A UNIQUE temp name: --watch may be re-rendering the same directory
    # concurrently with the live tracker, and a shared temp path lets the two
    # interleave into a published file matching neither.
    handle_fd, tmp = tempfile.mkstemp(dir=run_dir,
                                      prefix=f".{DASHBOARD_FILENAME}.",
                                      suffix=".tmp")
    try:
        with os.fdopen(handle_fd, "w", encoding="utf-8") as handle:
            handle.write(page)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, 0o644)       # mkstemp is 0600; the page is meant to be read
        os.replace(tmp, path)      # never serve a half-written page
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise

    if write_events:
        try:
            write_tensorboard(os.path.join(run_dir, TENSORBOARD_DIRNAME),
                              summaries)
        except Exception as exc:                    # noqa: BLE001
            print(f"[calibration_tracking] TensorBoard event file FAILED "
                  f"({type(exc).__name__}: {exc}) — the page at {path} is "
                  "unaffected", file=sys.stderr)
    return path


class CalibrationTracker:
    """The one object a campaign driver holds.

    Usage from inside the ensemble loop::

        tracker = CalibrationTracker(run_dir, param_bounds=bounds)
        for iteration in range(n_iterations):
            for member in range(n_members):
                ...                                   # 19 GPU-h happens here
                tracker.log_member(MemberRecord(...))  # durable immediately
            tracker.end_iteration()                    # redraw page + events

    Every method swallows its own exceptions. A tracking bug, a full disk or a
    read-only filesystem must never take down a campaign that has already spent
    hundreds of GPU-hours; the failure is printed and appended to
    ``tracking_errors.log`` instead. That includes the JSONL write: the log is
    the source of truth, but a lost line is still cheaper than a lost run.
    """

    def __init__(self, run_dir: str | os.PathLike[str],
                 param_bounds: Mapping[str, tuple[float, float]] | None = None,
                 *, title: str = _PAGE_TITLE, run_label: str = "",
                 pin_threshold: float = DEFAULT_PIN_THRESHOLD,
                 write_events: bool = True) -> None:
        self.run_dir = os.fspath(run_dir)
        self.param_bounds = dict(param_bounds or {})
        self.title = title
        self.run_label = run_label
        self.pin_threshold = pin_threshold
        self.write_events = write_events
        self.errors = 0
        # No state that a kill could lose: paths only. Everything else is
        # re-read from disk.
        self._guard("create run directory", os.makedirs, self.run_dir,
                    exist_ok=True)

    @property
    def members_path(self) -> str:
        return os.path.join(self.run_dir, MEMBERS_FILENAME)

    @property
    def dashboard_path(self) -> str:
        return os.path.join(self.run_dir, DASHBOARD_FILENAME)

    @property
    def tensorboard_dir(self) -> str:
        return os.path.join(self.run_dir, TENSORBOARD_DIRNAME)

    def _guard(self, what: str, function, *args, **kwargs) -> Any:
        """Run ``function``; on ANY exception log it and carry on."""
        try:
            return function(*args, **kwargs)
        except BaseException as exc:            # noqa: BLE001 - deliberate
            # BaseException, not Exception: a MemoryError or a recursion limit
            # inside a plotting helper must not propagate either. KeyboardInterrupt
            # and SystemExit are re-raised so a user's Ctrl-C still works.
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            self.errors += 1
            message = (f"[calibration_tracking] {what} FAILED "
                       f"({type(exc).__name__}: {exc}) - continuing; the "
                       "calibration run is unaffected")
            print(message, file=sys.stderr)
            try:
                with open(os.path.join(self.run_dir, ERROR_LOG_FILENAME), "a",
                          encoding="utf-8") as handle:
                    handle.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} "
                                 f"{what}\n")
                    handle.write(traceback.format_exc())
                    handle.write("\n")
            except Exception:                   # noqa: BLE001
                pass    # the error log is best-effort; a Ctrl-C landing here
                        # still propagates rather than being eaten
            return None

    def log_member(self, record: MemberRecord | Mapping[str, Any]) -> None:
        """Append one member's result to the JSONL. Durable on return."""
        self._guard("append member record", append_member_record,
                    self.members_path, record)

    def end_iteration(self) -> str | None:
        """Regenerate the page and the event files from the log."""
        path = self._guard(
            "render dashboard", render_dashboard, self.run_dir,
            self.param_bounds, title=self.title, run_label=self.run_label,
            pin_threshold=self.pin_threshold, write_events=self.write_events)
        return str(path) if path is not None else None

    # Alias: the same operation, named for out-of-band use.
    refresh = end_iteration

    def summaries(self) -> tuple[IterationSummary, ...]:
        """Current per-iteration summaries, re-read from the log."""
        result = self._guard(
            "summarize records", lambda: summarize(
                read_member_records(self.members_path), self.param_bounds,
                pin_threshold=self.pin_threshold))
        return result if result is not None else ()
