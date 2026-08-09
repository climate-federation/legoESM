"""Compare two AMIP checkpoint directories bitwise, with non-vacuity controls.

The durable form of the byte-identity evidence for an opt-in dycore option:
run the driver on the BASE commit and on the branch with the flag OFF, then

    python scripts/validate/check_default_path_byte_identity.py \\
        --base   <base_output_dir> \\
        --same   <flag_off_output_dir> \\
        --differ <flag_on_output_dir> \\
        --days 1 2

asserts that ``--same`` is bit-identical to ``--base`` and that ``--differ``
is NOT (so a comparison that could not have failed is caught).  "Bit-identical"
means shape + exact dtype + raw bytes, not ``np.array_equal`` (which accepts
``+0.0`` vs ``-0.0`` and cross-dtype equality).  Why a script
and not a unit test: the run itself is a multi-minute subprocess driver
invocation, too heavy for the PR suite — but the comparison logic and the
non-vacuity controls belong in the repo, not in a scratch directory
(codex 2026-08-06 round 2).

Every key is compared bytewise (numeric AND string/metadata payloads); the
numeric subset drives the vacuity statistics and the max-delta report.

NON-VACUITY CONTROLS, all fatal:
  * a minimum number of numeric arrays and elements actually compared;
  * a minimum number of arrays that are NOT constant (an all-zero diagnostic
    field compares equal no matter what the model did — a prior agent on this
    campaign got a false pass exactly that way);
  * the named prognostic fields must be non-empty, finite and non-constant
    (a constant prognostic compares equal whatever the model did);
  * ``--differ`` must differ, or the flag reached nothing.

Exit code 0 = identical where required and different where required.
"""
from __future__ import annotations

import argparse
import hashlib
import sys

import numpy as np

# A comparison smaller than this cannot support a byte-identity claim.
MIN_ARRAYS = 10
MIN_ELEMENTS = 10_000
MIN_NONCONSTANT = 5
# Prognostic fields that must be present and non-degenerate.
REQUIRED_FIELDS = ("T", "u", "p_s")


def _load(directory: str, day: int):
    return np.load(f"{directory}/checkpoint_day_{day:04d}.npz")


def _numeric_keys(npz) -> list[str]:
    return sorted(k for k in npz.files
                  if np.asarray(npz[k]).dtype.kind in "fiub")


def _byte_signature(a: np.ndarray) -> tuple:
    """Shape + exact dtype + C-order raw bytes.

    NOT ``np.array_equal``: that accepts ``+0.0`` vs ``-0.0`` and equal values
    stored in different dtypes, so it cannot support a BYTE-identity claim
    (codex round 3).  Layout is normalised to C order by ``tobytes()`` — npz
    payloads are stored C-contiguous, so this is a no-op here and the claim is
    precisely "same shape, same dtype, same C-order bytes" (codex round 4).
    """
    return (a.shape, a.dtype.str, a.tobytes())


def _same_bytes(x, y) -> bool:
    return _byte_signature(x) == _byte_signature(y)


def compare_day(base: str, same: str, differ: str | None, day: int) -> None:
    b, s = _load(base, day), _load(same, day)
    d = _load(differ, day) if differ else None
    # ALL keys are compared, not just numeric ones: a real MPAS checkpoint
    # carries string payloads (``tracer_names``, ``physstate_meta_*``) and a
    # changed scheme name there is a changed run (codex round 4).  The numeric
    # subset is used only for the vacuity statistics and the delta report.
    all_keys = sorted(b.files)
    if all_keys != sorted(s.files):
        raise SystemExit(f"day {day}: key sets differ base vs --same")
    if d is not None and all_keys != sorted(d.files):
        raise SystemExit(f"day {day}: key sets differ base vs --differ")
    keys = _numeric_keys(b)

    n_elem = sum(np.asarray(b[k]).size for k in keys)
    n_nonconst = sum(
        1 for k in keys
        if np.asarray(b[k]).size > 1
        and np.ptp(np.asarray(b[k], np.float64)) > 0
    )
    print(f"\n### day {day}: {len(all_keys)} keys ({len(keys)} numeric), "
          f"{n_elem} elements, {n_nonconst} non-constant")
    if len(keys) < MIN_ARRAYS or n_elem < MIN_ELEMENTS:
        raise SystemExit(
            f"day {day}: VACUOUS comparison ({len(keys)} arrays, {n_elem} "
            f"elements; need >= {MIN_ARRAYS} / {MIN_ELEMENTS})")
    if n_nonconst < MIN_NONCONSTANT:
        raise SystemExit(
            f"day {day}: VACUOUS comparison — only {n_nonconst} non-constant "
            f"arrays (need >= {MIN_NONCONSTANT}); all-zero diagnostic arrays "
            f"compare equal no matter what the model did")

    for name in REQUIRED_FIELDS:
        if name not in keys:
            raise SystemExit(f"day {day}: required field {name!r} missing")
        a = np.asarray(b[name], np.float64)
        if a.size == 0:
            raise SystemExit(f"day {day}: required field {name!r} is empty")
        if not np.all(np.isfinite(a)):
            raise SystemExit(f"day {day}: required field {name!r} is non-finite")
        if np.ptp(a) == 0:
            raise SystemExit(f"day {day}: required field {name!r} is constant "
                             f"— it cannot distinguish the two runs")
        print(f"   {name}: shape {a.shape} range [{a.min():.4f}, {a.max():.4f}]"
              f" sha1 {hashlib.sha1(a.tobytes()).hexdigest()[:12]}")

    same_diffs = [k for k in all_keys
                  if not _same_bytes(np.asarray(b[k]), np.asarray(s[k]))]
    print(f"   base vs --same  : {len(same_diffs)} differing arrays -> "
          f"{'BIT-IDENTICAL' if not same_diffs else same_diffs}")
    if same_diffs:
        raise SystemExit(f"day {day}: DEFAULT PATH PERTURBED: {same_diffs}")

    if d is not None:
        diffs = [(k, (float(np.max(np.abs(np.asarray(b[k], np.float64)
                                          - np.asarray(d[k], np.float64))))
                      if k in keys else float("nan")))
                 for k in all_keys
                 if not _same_bytes(np.asarray(b[k]), np.asarray(d[k]))]
        print(f"   base vs --differ: {len(diffs)} differing arrays "
              + ", ".join(f"{k}={v:.3e}" for k, v in diffs[:6]))
        if not diffs:
            raise SystemExit(
                f"day {day}: NON-VACUITY FAILED — the --differ run is "
                f"identical too, so the comparison proves nothing")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True, help="reference output directory")
    p.add_argument("--same", required=True,
                   help="directory that must be BIT-IDENTICAL to --base")
    p.add_argument("--differ", default=None,
                   help="directory that must DIFFER (non-vacuity control)")
    p.add_argument("--days", type=int, nargs="+", default=[1],
                   help="checkpoint days to compare")
    a = p.parse_args(argv)
    for day in a.days:
        compare_day(a.base, a.same, a.differ, day)
    print("\nPASS: --same is bit-identical to --base"
          + ("; --differ does differ." if a.differ else "."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
