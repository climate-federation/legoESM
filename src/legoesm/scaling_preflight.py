"""Parse-time preflight for scale-out benchmarks (#1361).

Every check here is decidable from the ARGUMENTS ALONE -- no grid build, no
compile, no allocation -- so a misconfigured multi-node job fails at submit time
instead of after the scheduler has handed out nodes.  Three scale-out attempts
on 2026-07-27 each burned a multi-node allocation on a constraint that was
already known at argparse time (#1361):

    * ``--n-lat 720 --n-devices 64``  -> died after the 16-GPU arm had run
    * cube ``cs-spmd`` at 96 GPUs     -> silently replicated, then an 83 GB arg
    * cube ``C1152 L60 kt=3``         -> 105.7 GB/device vs 80 GB HBM, found
                                         during compile, two arms in

This is the CLAUDE.md dispatch-hardening rule applied to scaling configuration:
a selector that cannot possibly work is a HARD ERROR, never a silent fallback.

The validators raise :class:`ValueError`; ``preflight_or_exit`` converts that to
``SystemExit`` so a bench dies with a clean message rather than a traceback.
"""

from __future__ import annotations

# --- shard-ability of each cubed-sphere parallel path -----------------------
# cs-spmd shards the 6 faces across devices and only divides evenly at these
# counts; anything else silently replicated the global state before #1361.
CS_SPMD_DEVICE_COUNTS: tuple[int, ...] = (1, 2, 3, 6)
# The tiled path uses a (6, kt, kt) mesh -> n_devices = 6*kt^2 (6, 24, 54, 96).
_TILED_MAX_KT = 16

# Device HBM [bytes] for the accelerators these benches target. Used only for
# the memory preflight message; pass ``device_hbm_bytes`` to override.
DEVICE_HBM_BYTES: dict[str, int] = {
    "a100-40": 40 * 1024**3,
    "a100-80": 80 * 1024**3,
    "h100": 80 * 1024**3,
    "v100": 32 * 1024**3,
    "rtx8000": 48 * 1024**3,
}


def tiled_device_counts(max_kt: int = _TILED_MAX_KT) -> tuple[int, ...]:
    """Valid ``n_devices`` for the tiled cube path: ``6*kt^2``."""
    return tuple(6 * kt * kt for kt in range(1, max_kt + 1))


def nearest_divisible(n: int, divisor: int, *, n_candidates: int = 3) -> list[int]:
    """Values nearest ``n`` that ARE divisible by ``divisor`` (never empty).

    Used to turn "720 is not divisible by 64" into an actionable
    "use 704 or 768" instead of making the user do arithmetic at submit time.
    """
    if divisor <= 0:
        raise ValueError(f"divisor must be positive, got {divisor}")
    if n_candidates <= 0:
        raise ValueError(
            f"n_candidates must be positive, got {n_candidates} "
            f"(the docstring promises a non-empty list)")
    # Rank a window of multiples by ACTUAL distance to n, not by side. The
    # side-alternating version returned [64, 128, 192] for (190, 64) and
    # omitted 256, which is closer than 64 (codex, PR #1376).
    k = max(1, n // divisor)
    span = n_candidates + 2
    cands = {m * divisor for m in range(max(1, k - span), k + span + 1)}
    return sorted(cands, key=lambda v: (abs(v - n), v))[:n_candidates]


def validate_divisibility(extent: int, n_devices: int, *,
                          axis: str = "n_lat") -> None:
    """``extent`` must split evenly across ``n_devices`` along ``axis``."""
    if n_devices <= 0:
        raise ValueError(f"n_devices must be positive, got {n_devices}")
    if extent <= 0:
        raise ValueError(f"{axis} must be positive, got {extent}")
    if extent % n_devices:
        near = nearest_divisible(extent, n_devices)
        raise ValueError(
            f"{axis}={extent} is not divisible by n_devices={n_devices} "
            f"({extent / n_devices:.2f} rows/device). This is fatal AFTER the "
            f"allocation is granted, so it is rejected here. Nearest valid "
            f"{axis}: {near}.")


#: Thinnest lat-band height receipted to initialize NCCL cleanly on the
#: multi-node GPU lane. 12-row bands (LL2304 @ 192) deadlock NCCL channel
#: setup on the FIRST call and burn the whole walltime -- seven env
#: candidates refuted, while 15-row (LL2880 @ 192) and 16-row (@144) run
#: clean, and the same 12-row program executes fine on CPU virtual
#: devices (jobs 26979367/26996572/27007255/27013729/27014326/27015385/
#: 27016462 hung; 27036060 clean). Empirical floor, not a derived bound:
#: 13/14-row bands are untested.
MIN_GPU_BAND_ROWS = 15


def validate_band_rows_gpu(extent: int, n_devices: int, *,
                           axis: str = "n_lat") -> None:
    """Refuse lat-band heights below the receipted NCCL-init floor.

    Only meaningful for the multi-node GPU (NCCL) lane -- the caller
    decides when to apply it. Escape hatch for a deliberate retest:
    ``LEGOESM_ALLOW_THIN_BANDS=1``.
    """
    import os
    rows = extent // max(1, n_devices)
    if rows >= MIN_GPU_BAND_ROWS:
        return
    if os.environ.get("LEGOESM_ALLOW_THIN_BANDS", "") == "1":
        return
    raise ValueError(
        f"{axis}={extent} over {n_devices} devices gives {rows}-row bands; "
        f"bands thinner than {MIN_GPU_BAND_ROWS} rows deadlock NCCL "
        f"communicator setup on the multi-node GPU lane (receipt: 12-row "
        f"LL2304@192 hung seven independent walltimes; 15-row LL2880@192 "
        f"ran 6.74 ms/step). Use a larger {axis} or fewer devices, or set "
        f"LEGOESM_ALLOW_THIN_BANDS=1 for a deliberate retest.")


def validate_device_count(path: str, n_devices: int) -> None:
    """``n_devices`` must be shardable by the selected cube parallel ``path``.

    Before #1361 an unsupported count silently fell back to REPLICATING the
    global state, which then died with an opaque multi-GB argument-size error.
    """
    if n_devices <= 0:
        raise ValueError(f"n_devices must be positive, got {n_devices}")
    if path == "cs-spmd":
        if n_devices not in CS_SPMD_DEVICE_COUNTS:
            raise ValueError(
                f"cs-spmd face-shards only at n_devices in "
                f"{list(CS_SPMD_DEVICE_COUNTS)}, got {n_devices}. Larger counts "
                f"do NOT shard -- they replicate the global state and fail late "
                f"with an argument-size error. Use the tiled path "
                f"(n_devices = 6*kt^2) for higher device counts.")
    elif path == "tiled":
        valid = tiled_device_counts()
        if n_devices not in valid:
            raise ValueError(
                f"the tiled cube path needs n_devices = 6*kt^2, got "
                f"{n_devices}. Valid counts: {list(valid[:8])}...")
    else:
        # dispatch hardening: an unknown path must never validate silently
        raise ValueError(
            f"unknown cube parallel path {path!r}; expected 'cs-spmd' or "
            f"'tiled'.")


def estimate_bytes_per_device(*, n_columns: int, nlev: int, n_devices: int,
                              bytes_per_value: int = 8,
                              n_fields: int = 12,
                              working_set_factor: float = 3.0,
                              sharded: bool = True) -> float:
    """Rough per-device bytes for the model state.

    ``n_columns`` is the GLOBAL horizontal column count (``n_lat*n_lon``, or
    ``6*res^2`` for a cube). ``n_fields`` covers the prognostic set plus the
    usual diagnostics; ``working_set_factor`` accounts for the integrator's live
    temporaries (RK stages / checkpoints).

    ``sharded`` -- whether the selected path actually DIVIDES the state across
    devices. This is not a formality: per #1370 the cube/ocean SPMD lanes still
    allocate GLOBAL-sized buffers on every device, so dividing by ``n_devices``
    there under-estimates by exactly ``n_devices`` and would wave through the
    very configuration this preflight exists to reject. Callers on an
    unsharded path MUST pass ``sharded=False``; when #1370 lands they flip.

    CALIBRATION: the one measured point is the cube ``C1152 L60 kt=3`` arm,
    which needed 105.7 GB/device. Global per-field bytes there are
    ``6*1152^2*60*8`` = 3.56 GB, so the observed footprint is ~29.7x one field.
    The defaults (12 fields x 3.0 working set = 36x) give ~128 GB -- a ~21%
    OVER-estimate, deliberately on the safe side: a false "fits" costs a
    multi-node allocation, a false "does not fit" costs one CLI flag.
    """
    # Every term multiplies the estimate, so a zero/negative one silently
    # produces a "fits" verdict (codex Medium, PR #1376). n_columns and nlev
    # reach here from CLI flags that argparse only type-checks.
    for _name, _val in (("n_devices", n_devices), ("n_columns", n_columns),
                        ("nlev", nlev), ("bytes_per_value", bytes_per_value),
                        ("n_fields", n_fields),
                        ("working_set_factor", working_set_factor)):
        if _val <= 0:
            raise ValueError(f"{_name} must be positive, got {_val}")
    per_field = float(n_columns) * float(nlev) * float(bytes_per_value)
    total = per_field * n_fields * working_set_factor
    return total / float(n_devices) if sharded else total


def validate_memory(*, n_columns: int, nlev: int, n_devices: int,
                    device: str | None = None,
                    device_hbm_bytes: int | None = None,
                    bytes_per_value: int = 8,
                    n_fields: int = 12,
                    working_set_factor: float = 3.0,
                    sharded: bool = True) -> float:
    """Refuse a configuration whose estimated per-device bytes exceed HBM.

    Returns the estimate [bytes] when it fits (callers log it). No device
    given -> estimate only, no gate (we do not guess the hardware).
    """
    est = estimate_bytes_per_device(
        n_columns=n_columns, nlev=nlev, n_devices=n_devices,
        bytes_per_value=bytes_per_value, n_fields=n_fields,
        working_set_factor=working_set_factor, sharded=sharded)
    if device_hbm_bytes is None and device is not None:
        if device not in DEVICE_HBM_BYTES:
            raise ValueError(
                f"unknown device {device!r}; known: "
                f"{sorted(DEVICE_HBM_BYTES)} (or pass device_hbm_bytes).")
        device_hbm_bytes = DEVICE_HBM_BYTES[device]
    if device_hbm_bytes is not None and est > device_hbm_bytes:
        raise ValueError(
            f"estimated {est / 1024**3:.1f} GB/device exceeds the "
            f"{device_hbm_bytes / 1024**3:.0f} GB HBM of "
            f"{device or 'the target device'} "
            f"(global columns={n_columns}, nlev={nlev}, n_devices={n_devices}, "
            f"{bytes_per_value * 8}-bit). Raise n_devices, cut the resolution, "
            f"or drop to 32-bit.")
    return est


def preflight_or_exit(fn, *args, **kwargs):
    """Run a validator, converting ValueError into a clean ``SystemExit``.

    Benches call this from ``main()`` right after parsing so the failure is a
    one-line message at submit time, not a traceback mid-allocation.
    """
    try:
        return fn(*args, **kwargs)
    except ValueError as exc:
        raise SystemExit(f"[preflight] {exc}") from None
