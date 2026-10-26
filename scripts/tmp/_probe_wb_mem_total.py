"""Compile the WB training gradient and print ONLY XLA's own memory totals.

Deliberately NOT the HLO-buffer ranking used earlier: that sums every named
value (parameters, tuple extractions, fusion internals) and is not a peak live
set, so it can rank components in the wrong order -- codex's finding, and the
reason the first diagnosis blamed the biggest TENSOR instead of the biggest
CONSUMER. ``memory_analysis()`` is XLA's estimate for the executable.
"""
import sys

import jax


def run(vg, arr, sample, label=""):
    compiled = vg.lower(arr, jax.device_put(sample)).compile()
    compiled = getattr(compiled, "compiled", compiled)  # equinox wraps it
    m = compiled.memory_analysis()
    g = 2 ** 30
    print(f"MEMRESULT arm={label} "
          f"temp={m.temp_size_in_bytes / g:.2f} "
          f"arg={m.argument_size_in_bytes / g:.2f} "
          f"out={m.output_size_in_bytes / g:.2f} "
          f"alias={m.alias_size_in_bytes / g:.2f}", flush=True)
    sys.exit(0)
