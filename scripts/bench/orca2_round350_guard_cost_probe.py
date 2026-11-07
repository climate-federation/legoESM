"""Time the round-349 fast-literal child with the NEMO source-rounding guards on or off.

One variable: ``--guards off`` replaces ``legoesm.core.source_rounding.nemo_source_round``
with a pass-through before any model module is imported; everything else is the
unchanged round-349 child (same arguments, pinning, preflight and timing window).
Timing only: a guards-off run does not reproduce NEMO's bits.  Scope is that one shared
helper; the smaller guard helpers (implicit-solver multiply rounding, barotropic IEEE add,
TKE binary64 rounding, nemo_source_barrier, raw optimization barriers) stay on in both arms.

Usage: orca2_round350_guard_cost_probe.py --guards on|off --report R.json -- <child args>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

BENCH = (
    Path(__file__).resolve().parents[1]
    / "validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round343_speed_benchmark.py"
)


def install_passthrough_guard():
    """Swap the guard for a pass-through; return (original, {"eager": n, "traced": n})."""
    import jax
    from legoesm.core import source_rounding

    original = source_rounding.nemo_source_round
    calls = {"eager": 0, "traced": 0}

    def passthrough(value):
        calls["traced" if isinstance(value, jax.core.Tracer) else "eager"] += 1
        return value

    source_rounding.nemo_source_round = passthrough
    return original, calls


def stale_bindings(original) -> list[str]:
    """Modules that still hold the original guard (bound before the swap)."""
    return sorted(
        name
        for name, module in list(sys.modules.items())
        if any(value is original for value in getattr(module, "__dict__", {}).values())
    )


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    split = argv.index("--")
    parser = argparse.ArgumentParser()
    parser.add_argument("--guards", choices=("on", "off"), required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv[:split])

    import legoesm.core.source_rounding  # noqa: F401  -- same import order in both arms

    original = calls = None
    if args.guards == "off":
        original, calls = install_passthrough_guard()
    spec = importlib.util.spec_from_file_location("round349_bench", BENCH)
    bench = importlib.util.module_from_spec(spec)
    sys.modules["round349_bench"] = bench
    spec.loader.exec_module(bench)
    child_rc = bench.main(argv[split + 1 :])

    report = {"guards": args.guards, "child_rc": child_rc}
    ok = child_rc == 0
    if args.guards == "off":
        report["passthrough_calls"] = calls
        report["stale_bindings"] = stale_bindings(original)
        ok = ok and calls["traced"] > 0 and not report["stale_bindings"]
    report["valid"] = ok
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"GUARD_PROBE guards={args.guards} valid={ok} {json.dumps(report)}")
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
