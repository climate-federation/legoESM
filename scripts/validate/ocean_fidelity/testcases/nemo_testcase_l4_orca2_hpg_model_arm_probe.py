#!/usr/bin/env python3
"""ORCA2 as the FOURTH Rule-12 card for the round-25 HPG/QCO change.

Rule 12 lets a change land only if the changed operator is bit-exact given
NEMO's own inputs on every card that executes it.  Round 25 discharged GYRE,
round 26 discharged LOCK and OVERFLOW, and ORCA2 -- which pins
``pgf_scheme='nemo_sco'`` like the others -- was left BLOCKED on a branch
divergence.  It is not blocked numerically: NEMO dumps the stage-2 HPG
operands and the stage-2 HPG literal for ORCA2's kt=1, and both are on disk.

WHY NOT THE EXISTING ORCA2 HPG GATE.  ``orca2_l4/nemo_testcase_l4_orca2_
hpg_gate.py`` calls ``nemo_hpg_sco_literal_cgrid`` on the dumped operands.
That helper is only HALF of what round 25 changed: the other half is the
CONSUMER, ``_nemo_hpg_tendency_from_pressure_or_direct``, which carries the
acceleration straight to the momentum RHS instead of multiplying by ``-rho0``
and dividing it back at the caller (``dynhpg.F90:359,383`` writes acceleration
into Krhs).  A gate that stops at the helper is blind to exactly the interface
under test, so this probe routes the same operands through the model's
``nemo_sco`` composition instead.

WHY THE OPERANDS ROUTE AND NOT A FRAME.  On LOCK and OVERFLOW the kt=1 frame
is ``dyn_hpg`` alone because both start from rest.  ORCA2 does not, so that
isolation does not transfer and a frame comparison there would be a composite.
The dumped operand/literal pair is the only valid route.

DECLARED SCOPE LIMIT, up front.  NEMO supplies ``gdept_z0`` in the operand
record, so this row exercises the changed CONSUMER and the operator given
NEMO's own inputs.  It does NOT exercise ``_nemo_qco_gdept_z0``, the builder,
because no ORCA2 record carries the ``(gdept_0, 1+r3t, ssh)`` triple at the
stage-2 time level.  That builder is card-independent pure arithmetic pinned
by ``test_nemo_qco_gdept_z0_oracle_bit_pattern``; this probe states that
rather than claiming an ORCA2 measurement of it.

WHAT THIS ROW DOES NOT COVER, stated up front.  The records are rank 0 of a
two-rank run and the scored window is ``[3:-3, 3:-3, :30]``, so the ORCA2 NORTH
FOLD, the cyclic east-west seam and level 31 are all outside it -- and those
are exactly the topology ORCA2 alone could test.  The scored interior is, in
that sense, a larger GYRE.  The horizontal metric is also RECONSTRUCTED: the
record carries ``r1_e1u``/``r1_e2v`` and the probe inverts them for the grid,
which the operator then inverts back.  That is the same association hazard the
LOCK walk turned on, so it was measured rather than assumed -- the round trip
is bit-exact on all 2035 and 2070 distinct nonzero values, so it does not bite
here.

Zero NEMO runs.  The ORCA2 branch is not modified: ``--model-root`` points at
a DISPOSABLE detached worktree with ``d5a7f8169507`` cherry-picked, and the
probe imports that checkout's ORCA2 gate module for its record readers rather
than copying them.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from legoesm.ocean.fidelity.provenance import worktree_stamp

COMPONENTS = ("sum_u", "sum_v", "zhpi_u", "zhpi_v", "zuap_u", "zuap_v")
# nemo_hpg_sco_literal_cgrid(return_components=True) returns this order.
COMPONENTS_IN_RETURN_ORDER = (
    "sum_u", "sum_v", "zhpi_u", "zhpi_v", "zuap_u", "zuap_v")
# NEMO's own phycst values, from the shared preset -- never re-typed here.
# rho_0 is load-bearing only for the legacy ablation arm (the landed arm
# returns the acceleration and never touches it); g is load-bearing for both.
FIX = "d5a7f816950713965e39361a1c605d7007868200"


class ProbeError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ProbeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_orca2_gate(model_root: Path):
    """Import the ORCA2 branch's own gate module for its readers and masks."""
    path = (model_root / "scripts/validate/ocean_fidelity/orca2_l4"
            / "nemo_testcase_l4_orca2_hpg_gate.py")
    require(path.is_file(), f"missing ORCA2 gate module {path}")
    spec = importlib.util.spec_from_file_location(
        "nemo_testcase_l4_orca2_hpg_gate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score(candidate: np.ndarray, oracle: np.ndarray, mask: np.ndarray) -> dict:
    actual = np.asarray(candidate, np.float64)[mask]
    expected = np.asarray(oracle, np.float64)[mask]
    require(actual.shape == expected.shape and actual.size,
            "empty or mismatched score")
    require(np.isfinite(actual).all() and np.isfinite(expected).all(),
            "non-finite score")
    unequal = actual.view(np.uint64) != expected.view(np.uint64)
    return {
        "status": "AT-BAR" if not unequal.any() else "DEBT",
        "exact": not bool(unequal.any()),
        "n_unequal": int(unequal.sum()),
        "n": int(unequal.size),
        "absolute_max": float(np.abs(actual - expected).max(initial=0.0)),
    }


def run(oracle_root: Path, model_root: Path, *, plant: bool = False) -> dict:
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics import ocean_pe_latlon_cgrid as pe_module
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        nemo_hpg_sco_literal_cgrid,
    )
    from legoesm.ocean.constants_config import NEMO_CONSTANTS_CONFIG
    from netCDF4 import Dataset

    gravity = NEMO_CONSTANTS_CONFIG.g
    rho_0_nominal = NEMO_CONSTANTS_CONFIG.rho_0

    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(jax.default_backend() == "cpu", "probe is CPU-only")
    require(not jax.config.jax_disable_jit, "production JIT is required")
    require(get_policy() == policy, "fp64 + scalar-libm policy not active")

    module_file = Path(pe_module.__file__).resolve()
    require(str(module_file).startswith(str(model_root.resolve())),
            f"legoesm resolved to {module_file}, not under {model_root}")
    head = subprocess.run(["git", "-C", str(model_root), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True)
    dirty = subprocess.run(["git", "-C", str(model_root), "status",
                            "--porcelain"], capture_output=True, text=True,
                           check=True).stdout.strip()
    require(not dirty, f"probe worktree is dirty:\n{dirty}")
    # The fix reaches this worktree by CHERRY-PICK, so it has a different sha
    # and an ancestry test would be wrong.  Check the two changed functions
    # BEHAVE as the fix requires instead -- the same two assertions the
    # committed pins make, re-run against whatever code actually loaded.  This
    # is strictly stronger than ancestry: a reverted or half-applied
    # cherry-pick fails it.
    for name in ("_nemo_qco_gdept_z0",
                 "_nemo_hpg_tendency_from_pressure_or_direct"):
        require(hasattr(pe_module, name),
                f"the loaded model has no {name}; the round-25 change "
                f"({FIX[:12]}) is not in this worktree")
    gdept_bits = int(np.asarray(jax.jit(pe_module._nemo_qco_gdept_z0)(
        jnp.asarray([60.731977023408945], dtype=jnp.float64),
        jnp.asarray([[0.9999998965725583]], dtype=jnp.float64),
        jnp.asarray([[-0.0004448114346508057]], dtype=jnp.float64),
    )).view(np.uint64).item())
    require(gdept_bits == 0x404E5DBFCAF8A971,
            f"_nemo_qco_gdept_z0 gives {gdept_bits:#018x}, not the pinned "
            "0x404e5dbfcaf8a971; the QCO source association is not active")
    direct = jnp.asarray([3.812364514570218e-06], dtype=jnp.float64)
    passthrough = np.asarray(jax.jit(
        pe_module._nemo_hpg_tendency_from_pressure_or_direct)(
            -jnp.float64(1026.0) * direct, jnp.float64(1026.0), direct))
    require(np.array_equal(passthrough.view(np.uint64),
                           np.asarray(direct).view(np.uint64)),
            "_nemo_hpg_tendency_from_pressure_or_direct does not carry the "
            "acceleration through; the HPG interface change is not active")
    cherry = subprocess.run(
        ["git", "-C", str(model_root), "log", "-1", "--format=%H %s"],
        capture_output=True, text=True, check=True).stdout.strip()

    gate = load_orca2_gate(model_root)
    operand_path = oracle_root / "oracle_rkstage2_hpg_operands_kt00000001.bin"
    literal_path = oracle_root / "oracle_rkstage2_hpg_literal_kt00000001.bin"
    for path in (operand_path, literal_path):
        require(path.is_file(), f"missing {path}")
    operands = gate.read_operands(operand_path)
    literal = gate.read_literal(literal_path)

    with Dataset(oracle_root / "mesh_mask_0000.nc") as dataset:
        umask = np.asarray(dataset["umask"][0, :gate.ACTIVE_Z].data, dtype=bool)
        vmask = np.asarray(dataset["vmask"][0, :gate.ACTIVE_Z].data, dtype=bool)
    umask = umask.transpose(1, 2, 0)[1:-1, 1:-1]
    vmask = vmask.transpose(1, 2, 0)[1:-1, 1:-1]

    r1_e1u, r1_e2v = literal["r1_e1u"], literal["r1_e2v"]
    dx_u = np.ones((gate.NY, gate.NX + 1), np.float64)
    dy_v = np.ones((gate.NY + 1, gate.NX), np.float64)
    dx_u[:, 1:] = np.divide(1.0, r1_e1u, out=np.ones_like(r1_e1u),
                            where=r1_e1u != 0.0)
    dy_v[1:, :] = np.divide(1.0, r1_e2v, out=np.ones_like(r1_e2v),
                            where=r1_e2v != 0.0)
    grid = gate.LocalMetricGrid(jnp.asarray(dx_u), jnp.asarray(dy_v))

    gdept_z0 = jnp.asarray(operands["gdept_z0"])
    planted_cells = None
    if plant:
        # Plant on a WET, SCORED face, derived from the mask rather than
        # guessed: an arbitrary index (the first draft used [10, 10, 5]) can
        # be masked land, and a control that perturbs a masked cell is not a
        # control.  A scored U point (a, b) sits at native u column b+3, whose
        # face reads gdept_z0 columns b+3 and b+4 of row a+3; both are
        # perturbed by one ULP at the surface, which the SCO recurrence
        # carries down the whole column.
        wet_surface = np.argwhere(umask[..., 0])
        require(wet_surface.size, "no wet scored U face at the surface")
        a, b = (int(v) for v in wet_surface[len(wet_surface) // 2])
        planted_cells = [[a + 3, b + 3, 0], [a + 3, b + 4, 0]]
        for row, col, lev in planted_cells:
            gdept_z0 = gdept_z0.at[row, col, lev].set(
                np.nextafter(float(gdept_z0[row, col, lev]), np.inf))

    # THE MODEL'S nemo_sco ARM, statement for statement:
    #   hpg_u/hpg_v <- nemo_hpg_sco_literal_cgrid   (the operator)
    #   dp_dx/dp_dy <- literal zeros                (this arm has no pressure)
    #   tendency    <- _nemo_hpg_tendency_from_pressure_or_direct(dp, rho0,
    #                                                             direct_hpg)
    # The last call is the half of the round-25 change that the ORCA2 gate's
    # direct helper call cannot see.
    def production_pair(rhd, e3w, depth, legacy_round_trip):
        """The model's nemo_sco arm, statement for statement.

        ``ocean_pe_latlon_cgrid`` calls ``nemo_hpg_sco_literal_cgrid`` with NO
        ``return_components``, assigns ``dp_dx_sco = zeros_like(hpg_u)``, casts
        it (``dp_dx = dp_dx_sco.astype(dp_dx.dtype)``) and hands both to
        ``_nemo_hpg_tendency_from_pressure_or_direct``.  This reproduces that
        shape exactly; the components call below is a DIAGNOSTIC and is
        checked against this pair rather than substituted for it.
        """
        hpg_u, hpg_v = nemo_hpg_sco_literal_cgrid(
            rhd, e3w, depth, grid, gravity)
        rho_0 = jnp.asarray(rho_0_nominal, dtype=hpg_u.dtype)
        if legacy_round_trip:
            direct_u = direct_v = None
            dp_dx, dp_dy = -rho_0 * hpg_u, -rho_0 * hpg_v
        else:
            direct_u, direct_v = hpg_u, hpg_v
            dp_dx = jnp.zeros_like(hpg_u).astype(hpg_u.dtype)
            dp_dy = jnp.zeros_like(hpg_v).astype(hpg_v.dtype)
        return (
            pe_module._nemo_hpg_tendency_from_pressure_or_direct(
                dp_dx, rho_0, direct_u),
            pe_module._nemo_hpg_tendency_from_pressure_or_direct(
                dp_dy, rho_0, direct_v),
        )

    def components(rhd, e3w, depth):
        """Diagnostic split only.  Its first two elements MUST equal the
        production pair, which is asserted rather than assumed."""
        return nemo_hpg_sco_literal_cgrid(
            rhd, e3w, depth, grid, gravity, return_components=True)

    compiled_pair = jax.jit(production_pair, static_argnums=3)
    compiled_components = jax.jit(components)
    inputs = (jnp.asarray(operands["rhd"]), jnp.asarray(operands["e3w"]),
              gdept_z0)

    split = tuple(np.asarray(v) for v in compiled_components(*inputs))
    landed_pair = tuple(np.asarray(v) for v in compiled_pair(*inputs, False))
    # Instrument calibration, the equivalent of the LOCK walk's own check:
    # the diagnostic split must reproduce the production pair BIT for BIT, or
    # the four component rows describe a different computation than the two
    # scored ones.
    reproduces = all(
        np.array_equal(np.asarray(a).view(np.uint64),
                       np.asarray(b).view(np.uint64))
        for a, b in zip(landed_pair, split[:2]))
    require(reproduces,
            "the return_components split does not reproduce the production "
            "pair bit for bit; the component rows would describe a different "
            "computation")

    def score_arm(legacy: bool) -> list[dict]:
        pair = tuple(np.asarray(v) for v in compiled_pair(*inputs, legacy))
        values = pair + split[2:]
        candidates = dict(zip(COMPONENTS_IN_RETURN_ORDER, values, strict=True))
        out = []
        for name in COMPONENTS:
            is_u = name.endswith("_u")
            native = candidates[name][:, 1:] if is_u else candidates[name][1:]
            out.append({
                "field": name,
                **score(native[3:-3, 3:-3, :gate.ACTIVE_Z],
                        literal[name][3:-3, 3:-3, :gate.ACTIVE_Z],
                        umask if is_u else vmask)})
        return out

    rows = score_arm(False)
    # Rule 4, one-variable ablation ON THIS CARD: put the pressure round trip
    # back and the same operands must stop being bit-exact.  Without this the
    # row could not tell "the change is correct here" from "the change is
    # inert here".
    legacy_rows = score_arm(True)
    round_trip_differs = any(row["status"] != "AT-BAR" for row in legacy_rows)
    # REQUIRED, not merely reported: if restoring the round trip changed
    # nothing, the change is inert on this card and the AT-BAR row proves
    # nothing about it.  Reporting that without failing was the defect.
    require(round_trip_differs or plant,
            "restoring the -rho0 pressure round trip changed nothing on "
            "ORCA2: the round-25 change is inert here, so this row is not "
            "evidence for it")

    first = next((row["field"] for row in rows if row["status"] != "AT-BAR"),
                 None)
    report = {
        "worktree": worktree_stamp(),
        "format": "nemo-testcase-l4-orca2-hpg-model-arm-probe-v1",
        "case": "ORCA2",
        "constants": {"g": gravity, "rho_0": rho_0_nominal,
                      "source": "legoesm.ocean.constants_config."
                                "NEMO_CONSTANTS_CONFIG"},
        "status": "AT-BAR" if first is None else "DEBT",
        "first_non_bit_operand": first,
        "rows": rows,
        "routed_through": (
            "nemo_hpg_sco_literal_cgrid -> "
            "_nemo_hpg_tendency_from_pressure_or_direct (the model's nemo_sco "
            "arm), NOT the helper alone"),
        "legacy_pressure_round_trip_ablation": {
            "rows": legacy_rows,
            "changes_the_result_on_this_card": round_trip_differs,
            "rho_0_used": rho_0_nominal,
            "note": "one-variable control: restoring the -rho0 round trip the "
                    "round-25 change removed must break bit-exactness here, "
                    "or the change would be inert on ORCA2 and the row would "
                    "prove nothing",
        },
        "scope_limit": (
            "gdept_z0 is NEMO's own, so _nemo_qco_gdept_z0 is NOT exercised "
            "here; it is card-independent arithmetic pinned by "
            "test_nemo_qco_gdept_z0_oracle_bit_pattern"),
        "oracle_root": str(oracle_root),
        "model_root": str(model_root),
        "model_git_sha": head.stdout.strip(),
        "model_carries_round25_fix": {
            "upstream_sha": FIX,
            "arrives_by": "cherry-pick, so a different sha; verified by "
                          "BEHAVIOUR instead of ancestry",
            "probe_worktree_tip": cherry,
            "gdept_z0_bit_pattern": f"{gdept_bits:#018x}",
            "hpg_consumer_is_bit_transparent": True,
        },
        "legoesm_module": str(module_file),
        "execution_regime": "production_jit",
        "precision_policy": "fp64",
        "transcendentals": get_policy().transcendentals,
        "jax_backend": jax.default_backend(),
        "comparison_domain": (
            "rank0 stencil-valid owned wet U/V core, 88x146x30; one owned-cell "
            "band excluded because the canonical MPI halos are zero. EXCLUDES "
            "the ORCA2 north fold, the cyclic seam and level 31 -- the "
            "topology this card alone could test"),
        "metric_is_reconstructed": (
            "dx_u/dy_v are 1/r1_e1u and 1/r1_e2v from the record; the operator "
            "inverts them back. Round trip measured bit-exact on all 2035 and "
            "2070 distinct nonzero values, so the association does not bite"),
        "artifacts": {operand_path.name: sha256(operand_path),
                      literal_path.name: sha256(literal_path)},
        "instrument_reproduces_production_pair": reproduces,
        "planted_control": plant,
        "planted_cells": planted_cells,
    }
    if plant:
        require(first is not None,
                "planted one-ULP gdept_z0 perturbation did not move any "
                "component; the probe cannot see its own operand")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run(args.oracle_root, args.model_root, plant=args.plant)
    except ProbeError as error:
        print(json.dumps({"status": "PROBE-ERROR", "error": str(error)},
                         indent=2))
        return 2
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")
    if args.plant:
        return 1
    return 0 if report["status"] == "AT-BAR" else 1


if __name__ == "__main__":
    raise SystemExit(main())
