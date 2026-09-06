#!/usr/bin/env python
"""Is there a GLOBAL XLA:CPU flag that disables the fused multiply-add?

Round 37 found that XLA on CPU evaluates NEMO's ``zrhs - zwi/zwt*pt``
(compiled ``trazdf.f90:532``) and ``(pt - zws*pt)/zwt`` (``:546``) as a FUSED
multiply-add, where gfortran rounds the multiply and the subtraction
separately, and it fixed that PER SITE -- ``_round_the_multiply`` in
``implicit_solver.py`` turns the subtrahend into an add, which an FMA cannot
absorb.  A per-site fix is a knob on every future site.

This probe asks whether the installed toolchain offers a GLOBAL switch
instead.  It enumerates candidate flag settings, runs each in its OWN
subprocess (``XLA_FLAGS`` is read when the backend initialises, so a flag set
after import does nothing), and reports for each whether ``c - a*b`` under
``jit`` reproduces the SEPARATELY ROUNDED result or the FUSED one.

The two reference answers are computed on the host, in the same fp64:

* separate -- ``numpy``'s ``c - a*b``, two roundings, which is what gfortran
  emits for NEMO's statement;
* fused -- ``math.fma(-a, b, c)``, one rounding, which is what round 37
  measured legoESM performing.

The triples are drawn so that the two answers DIFFER on a large fraction of
them: without that the probe would report "matches separate" for a flag that
does nothing, which is the vacuous-control failure this campaign has hit
before.  The count of differing triples is printed, and a run in which it is
zero is a hard failure rather than a pass.

This changes NO default.  Whether any flag it finds becomes the default is a
user decision and is recorded in the round's ASKED table.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

# Each candidate is (label, XLA_FLAGS value).  The empty string is the
# BASELINE arm and must be present: a probe with no baseline cannot tell a
# flag that works from a toolchain that never fused in the first place.
# FIVE CANDIDATES THIS BUILD REFUSES OUTRIGHT are NOT listed, because an arm
# that cannot execute is not a measurement:
# --xla_backend_extra_options=--fp-contract=off / =on (LLVM's -fp-contract is
# not a registered cl::opt here: "Unknown command line argument"),
# --xla_cpu_disable_platform_dependent_math,
# --xla_cpu_disable_new_fusion_emitters (both "Unknown flag in XLA_FLAGS"),
# and the two combined.  Their refusals are recorded in the round-38 receipt.
CANDIDATES = (
    ("baseline", ""),
    ("allow_excess_precision_false", "--xla_allow_excess_precision=false"),
    ("cpu_fast_math_false", "--xla_cpu_enable_fast_math=false"),
    ("opt_level_zero", "--xla_backend_optimization_level=0"),
    # THE ARM A FLAG-NAME SEARCH MISSES.  A fused multiply-add is an
    # AVX2/FMA3 instruction, so CAPPING THE ISA below AVX2 removes it while
    # leaving optimisation at full strength.  Nothing in the flag's NAME says
    # "fma", which is exactly why the first version of this probe -- which
    # searched 404 flag names for fma / contraction / fp-contract -- reported
    # that no such flag exists.  An independent diff review found it.
    ("max_isa_avx", "--xla_cpu_max_isa=AVX"),
    ("max_isa_sse4_2", "--xla_cpu_max_isa=SSE4_2"),
    ("max_isa_avx2", "--xla_cpu_max_isa=AVX2"),
    ("max_isa_avx512", "--xla_cpu_max_isa=AVX512"),
)
N = 4096
SEED = 20260906


def _triples():
    """Triples on which the fused and separate answers differ often.

    ``c`` is drawn close to ``a*b`` so the subtraction cancels most of the
    product's leading bits and the single-rounding difference survives into
    the result.
    """
    import numpy as np
    rng = np.random.default_rng(SEED)
    a = rng.uniform(-4.0, 4.0, N)
    b = rng.uniform(-4.0, 4.0, N)
    c = a * b * (1.0 + rng.uniform(-1e-3, 1e-3, N))
    return a, b, c


def _child() -> int:
    """One arm: report what ``c - a*b`` under jit reproduces."""
    import numpy as np
    import jax
    import jax.numpy as jnp

    a, b, c = _triples()
    separate = c - a * b                       # numpy: two roundings
    fused = np.array([math.fma(-ai, bi, ci)    # one rounding
                      for ai, bi, ci in zip(a, b, c)])
    differ = int(np.count_nonzero(
        separate.view(np.uint64) != fused.view(np.uint64)))

    @jax.jit
    def under_jit(aa, bb, cc):
        return cc - aa * bb

    got = np.asarray(under_jit(jnp.asarray(a), jnp.asarray(b), jnp.asarray(c)))
    print(json.dumps({
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "backend": jax.default_backend(),
        "n": N,
        "reference_pair_differ": differ,
        "matches_separate": int(np.count_nonzero(
            got.view(np.uint64) == separate.view(np.uint64))),
        "matches_fused": int(np.count_nonzero(
            got.view(np.uint64) == fused.view(np.uint64))),
    }))
    # A run whose two references agree everywhere proves nothing at all.
    return 0 if differ > N // 8 else 3


# The 2x2 that decides whether a global flag could REPLACE the per-site fix.
SWEEP_ARMS = (
    ("rounding_on__flag_off", True, ""),
    ("rounding_off_flag_off", False, ""),
    ("rounding_on__opt0", True, "--xla_backend_optimization_level=0"),
    ("rounding_off_opt0", False, "--xla_backend_optimization_level=0"),
    ("rounding_on__isa_avx", True, "--xla_cpu_max_isa=AVX"),
    ("rounding_off_isa_avx", False, "--xla_cpu_max_isa=AVX"),
)


def _sweep_child(record: Path, *, rounding: bool) -> int:
    """One 2x2 cell: legoESM's ordered solve on NEMO's own dumped matrix.

    ``rounding`` False replaces ``_round_the_multiply`` -- the round-37
    per-site anti-fusion helper -- by the identity, so this arm measures what
    the flag alone does.  The patch is a module attribute, which is how
    ``_nemo_ordered_solve`` looks the helper up at call time.
    """
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import nemo_testcase_l2_gyre_round35_trazdf_matrix as gate
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.physics.vertical_mixing import implicit_solver

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    if not rounding:
        implicit_solver._round_the_multiply = lambda product: product
    rec = gate.read_trazdf_matrix(record)
    got = gate.lego_sweep(rec)
    jpkm1 = rec["header"]["jpkm1"]
    out = {"xla_flags": os.environ.get("XLA_FLAGS", ""), "rounding": rounding}
    for tag in ("T", "S"):
        oracle = gate._box(rec, f"sol_{tag}_pre_clamp", jpkm1)
        row = gate.bit_row(f"sweep.{tag}", oracle, got[tag])
        out[tag] = {"bit_unequal": row["bit_unequal"], "n": row["n"],
                    "absolute_max": row["absolute_max"]}
    print(json.dumps(out))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true",
                        help="internal: run ONE arm in this process")
    parser.add_argument("--sweep-child", type=Path, default=None,
                        help="internal: run ONE 2x2 cell in this process")
    parser.add_argument("--rounding", choices=("on", "off"), default="on")
    parser.add_argument("--sweep-record", type=Path, default=None,
                        help="the round-37 trazdf record; runs the 2x2")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    if args.child:
        return _child()
    if args.sweep_child is not None:
        return _sweep_child(args.sweep_child, rounding=args.rounding == "on")
    if args.sweep_record is not None:
        rows = []
        for label, rounding, flags in SWEEP_ARMS:
            env = dict(os.environ)
            env["JAX_PLATFORMS"] = "cpu"
            env["JAX_ENABLE_X64"] = "1"
            if flags:
                env["XLA_FLAGS"] = flags
            else:
                env.pop("XLA_FLAGS", None)
            proc = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()),
                 "--sweep-child", str(args.sweep_record),
                 "--rounding", "on" if rounding else "off"],
                env=env, capture_output=True, text=True)
            tail = proc.stdout.strip().splitlines()
            try:
                row = json.loads(tail[-1]) if tail else {}
            except json.JSONDecodeError:
                row = {}
            row.update({"label": label, "flags": flags, "exit": proc.returncode})
            if proc.returncode != 0:
                row["stderr_tail"] = proc.stderr.strip().splitlines()[-3:]
            rows.append(row)
        text = json.dumps({"sweep_2x2": rows}, indent=1, sort_keys=True)
        if args.json:
            args.json.write_text(text + "\n")
        print(text)
        for row in rows:
            t, s = row.get("T", {}), row.get("S", {})
            print(f"{row['label']:<24} T {t.get('bit_unequal')}/{t.get('n')} "
                  f"S {s.get('bit_unequal')}/{s.get('n')}")
        return 0

    rows = []
    for label, flags in CANDIDATES:
        env = dict(os.environ)
        env["JAX_PLATFORMS"] = "cpu"
        env["JAX_ENABLE_X64"] = "1"
        if flags:
            env["XLA_FLAGS"] = flags
        else:
            env.pop("XLA_FLAGS", None)
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--child"],
            env=env, capture_output=True, text=True)
        tail = proc.stdout.strip().splitlines()
        try:
            row = json.loads(tail[-1]) if tail else {}
        except json.JSONDecodeError:
            row = {}
        row.update({"label": label, "flags": flags, "exit": proc.returncode,
                    "accepted": proc.returncode in (0, 3)})
        if not row.get("accepted"):
            row["stderr_tail"] = proc.stderr.strip().splitlines()[-3:]
        rows.append(row)

    baseline = next(r for r in rows if r["label"] == "baseline")
    if baseline.get("exit") == 3:
        print("FAIL: the two reference answers agree; this probe is vacuous")
        return 3
    # AN ARM THAT DID NOT RUN IS NOT AN ARM THAT FOUND NOTHING.  Five of the
    # first version's nine candidates aborted with "Unknown command line
    # argument" and the summary still printed a one-line verdict as though the
    # survey had been nine flags deep.  A rejected flag is now reported as
    # REJECTED and the probe exits non-zero, so a survey cannot silently
    # shrink.
    rejected = [r["label"] for r in rows if not r.get("accepted")]
    fuses = baseline.get("matches_fused") == N
    disablers = [r["label"] for r in rows
                 if r.get("matches_separate") == N and r["label"] != "baseline"]
    report = {"baseline_fuses": fuses, "n": N,
              "rejected_arms": rejected,
              "arms_attempted": len(rows),
              "arms_that_ran": len(rows) - len(rejected),
              "reference_pair_differ": baseline.get("reference_pair_differ"),
              "flag_that_disables_contraction": disablers, "rows": rows}
    text = json.dumps(report, indent=1, sort_keys=True)
    if args.json:
        args.json.write_text(text + "\n")
    print(text)
    for row in rows:
        print(f"{row['label']:<38} accepted={str(row['accepted']):<5} "
              f"separate {row.get('matches_separate')}/{N} "
              f"fused {row.get('matches_fused')}/{N}")
    print(f"BASELINE-FUSES {fuses}")
    print(f"DISABLERS {disablers or 'none'}")
    print(f"ARMS {len(rows) - len(rejected)}/{len(rows)} ran; "
          f"REJECTED {rejected or 'none'}")
    if rejected:
        print("FAIL: this survey is shallower than it looks; the rejected "
              "arms above never executed")
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
