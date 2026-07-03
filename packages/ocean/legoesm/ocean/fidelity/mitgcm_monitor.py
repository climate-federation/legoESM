"""Parser for MITgcm ``%MON`` monitor output (``pkg/monitor`` / ``STDOUT``).

MITgcm prints a block of scalar diagnostics every ``monitorFreq`` — kinetic
energy (``ke_mean``/``ke_max``), free-surface and velocity extrema
(``dynstat_eta_max``, ``dynstat_uvel_max`` …), advective CFL, and the time
stamps (``time_tsnumber``, ``time_secondsf``). Lines look like::

    (PID.TID 0000.0001) %MON ke_mean                      =   5.9208303599508E-09

This is a **coarse oracle reference that needs no field dumps and no MITgcm
build**: the monitor log ships with every verification experiment
(``verification/<case>/results/output.txt``), so the early-evolution scalar
time series is available for comparison even where the Fortran model cannot be
rebuilt. It complements the field-level :mod:`mitgcm_runner` reference (which
*does* require a run), giving a build-verification-tier check (does legoESM
reproduce MITgcm's KE growth / extrema over the first N steps?).

The parser is format-only and carries no MITgcm dependency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# `(PID.TID p.t) %MON <key> = <value>`. Capture the optional PID.TID (to filter
# multi-tile logs to the master tile), the key, and the raw numeric token. The
# value may be a Fortran float (E or D exponent), a bare integer (time_tsnumber),
# or — on a blow-up — NaN/Inf, which MUST match so divergence is not silently
# read as a missing key.
_MON_LINE = re.compile(
    r"(?:\(PID\.TID\s+(\S+)\)\s*)?"
    r"%MON\s+(\S+)\s*=\s*"
    r"([-+]?(?:[0-9][-+0-9.eEdD]*|NaN|Inf(?:inity)?))"
)

# Time-stamp keys used to populate ``steps`` / ``times_s``.
_KEY_TSNUMBER = "time_tsnumber"
_KEY_SECONDS = "time_secondsf"


@dataclass(frozen=True)
class MitgcmMonitor:
    """Parsed MITgcm monitor time series.

    Attributes
    ----------
    steps : np.ndarray
        ``time_tsnumber`` per monitor record (int64 when every record reported
        it; float64 with NaN for a truncated record). Empty if the log carried
        no timestep numbers.
    times_s : np.ndarray
        ``time_secondsf`` per record (float seconds).
    series : dict[str, np.ndarray]
        Every monitored key -> its value per record (float; ``nan`` where a
        record lacked that key). Records are in file order.
    """

    steps: np.ndarray
    times_s: np.ndarray
    series: dict[str, np.ndarray]

    @property
    def n_records(self) -> int:
        return int(len(self.times_s))

    def final(self, key: str, *, require_finite: bool = True) -> float:
        """The last value of ``key`` (e.g. ``ke_mean`` at the end of the log).

        Raises ``KeyError`` if the key was never monitored and (with
        ``require_finite=True``, the default) ``ValueError`` if the final value
        is non-finite — a blown-up or truncated reference must fail LOUDLY
        rather than hand back a silent ``nan`` to a comparison.
        """
        if key not in self.series:
            raise KeyError(f"no monitor key {key!r}; have {sorted(self.series)}")
        value = float(self.series[key][-1])
        if require_finite and not np.isfinite(value):
            raise ValueError(
                f"final {key!r} is non-finite ({value}); the MITgcm reference "
                f"blew up or the log is truncated. Pass require_finite=False to "
                f"read it anyway."
            )
        return value


def _to_float(token: str) -> float:
    # MITgcm/Fortran may print the exponent as D (e.g. 1.2D+04); Python wants E.
    # NaN/Inf pass through float() directly.
    return float(token.replace("D", "E").replace("d", "e"))


def parse_monitor_text(text: str) -> MitgcmMonitor:
    """Parse ``%MON`` lines from MITgcm stdout/output text.

    Records are segmented by key repetition: a monitor block prints each key
    once, so re-seeing a key starts a new record. This is robust to the key
    ORDER within a block (``time_tsnumber`` appears mid-block, not first).

    Multi-tile / multi-process logs are filtered to the MASTER tile (the first
    ``PID.TID`` seen): without this, each tile's repeated keys would spuriously
    split a step into several records. The one-off startup grid block (``XC_max``
    …, printed before any ``time_tsnumber``) has no repeating keys, so it merges
    into the first record — harmless, but it means record 0 also carries grid
    keys.
    """
    records: list[dict[str, float]] = []
    current: dict[str, float] = {}
    master_pid: str | None = None
    for pid, key, token in _MON_LINE.findall(text):
        if pid:
            if master_pid is None:
                master_pid = pid
            elif pid != master_pid:
                continue  # skip non-master tile lines (would corrupt segmenting)
        if key in current:
            records.append(current)
            current = {}
        current[key] = _to_float(token)
    if current:
        records.append(current)

    all_keys: set[str] = set()
    for rec in records:
        all_keys.update(rec)

    series = {
        key: np.array([rec.get(key, np.nan) for rec in records], dtype=np.float64)
        for key in sorted(all_keys)
    }
    if _KEY_TSNUMBER not in series:
        steps = np.empty(0, dtype=np.int64)
    elif np.all(np.isfinite(series[_KEY_TSNUMBER])):
        steps = series[_KEY_TSNUMBER].astype(np.int64)
    else:
        # A truncated log left a record without its timestep stamp; keep float
        # so the missing entry stays NaN instead of casting to a garbage int.
        steps = series[_KEY_TSNUMBER]
    times_s = (
        series[_KEY_SECONDS]
        if _KEY_SECONDS in series
        else np.full(len(records), np.nan)
    )
    return MitgcmMonitor(steps=steps, times_s=times_s, series=series)


def parse_monitor_file(path: str | Path) -> MitgcmMonitor:
    """Parse a MITgcm monitor/stdout file (e.g. ``results/output.txt``)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MITgcm monitor log not found: {path}")
    return parse_monitor_text(path.read_text(errors="replace"))
