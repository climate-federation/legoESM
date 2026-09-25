"""Direct non-vacuity checks for the round-50 literal LDF replay."""

from __future__ import annotations

import importlib.util
import hashlib
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).parents[3]
TESTCASES = ROOT / "scripts/validate/ocean_fidelity/testcases"
SPEC = importlib.util.spec_from_file_location(
    "nemo_testcase_l2_gyre_round50_ldf_association_gate",
    TESTCASES / "nemo_testcase_l2_gyre_round50_ldf_association_gate.py",
)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(TESTCASES))
SPEC.loader.exec_module(gate)


def _operands():
    y, x, z = 4, 5, 2
    rng = np.random.default_rng(50)
    tmask = np.ones((y, x, z))
    umask = np.ones((y, x + 1, z))
    vmask = np.ones((y + 1, x, z))
    fmask = np.ones((y + 1, x + 1, z))
    vmask[[0, -1]] = 0
    fmask[[0, -1]] = 0
    return {
        "u": rng.normal(size=umask.shape),
        "v": rng.normal(size=vmask.shape),
        "e3t_kbb": rng.uniform(1, 2, tmask.shape),
        "e3u_kbb": rng.uniform(1, 2, umask.shape),
        "e3v_kbb": rng.uniform(1, 2, vmask.shape),
        "e3f_live": rng.uniform(1, 2, fmask.shape),
        "e3u_kmm": rng.uniform(1, 2, umask.shape),
        "e3v_kmm": rng.uniform(1, 2, vmask.shape),
        "e2u": rng.uniform(2, 3, (y, x + 1)),
        "e1v": rng.uniform(2, 3, (y + 1, x)),
        "e2v": rng.uniform(2, 3, (y + 1, x)),
        "e1u": rng.uniform(2, 3, (y, x + 1)),
        "r1_e1e2t": rng.uniform(0.1, 0.2, (y, x)),
        "r1_e1e2f": rng.uniform(0.1, 0.2, (y + 1, x + 1)),
        "r1_e1u": rng.uniform(0.1, 0.2, (y, x + 1)),
        "r1_e2v": rng.uniform(0.1, 0.2, (y + 1, x)),
        "r1_e2u": rng.uniform(0.1, 0.2, (y, x + 1)),
        "r1_e1v": rng.uniform(0.1, 0.2, (y + 1, x)),
        "ahmt": np.ones((y, x, z)),
        "ahmf": fmask.copy(),
        "tmask": tmask,
        "umask": umask,
        "vmask": vmask,
        "fmask": fmask,
    }


def test_literal_replay_shapes_and_live_plant():
    operands = _operands()
    baseline = gate.literal_ldf_numpy(**operands)
    planted = gate.literal_ldf_numpy(**operands, plant=True)
    assert baseline["visc_u"].shape == operands["u"].shape
    assert baseline["visc_v"].shape == operands["v"].shape
    assert np.count_nonzero(baseline["zcur"] != planted["zcur"]) == 1
    assert np.any(baseline["visc_u"] != planted["visc_u"])


def test_association_ladder_is_closed_and_nonidentical():
    import jax.numpy as jnp

    values = gate._scale_variants(
        jnp, jnp.asarray([1.1]), jnp.asarray([2.2]), jnp.asarray([3.3]))
    assert set(values) == {
        "ab_div_c", "a_mul_bdivc", "b_mul_adivc", "ab_mul_recipc",
        "a_div_cdivb", "b_div_cdiva",
    }
    assert all(np.isfinite(np.asarray(value)).all() for value in values.values())


def test_shared_ldf_intermediate_seam_is_default_inert():
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        nemo_ldf_lap_viscosity_e3_cgrid,
    )

    set_policy(PrecisionPolicy.fp64())
    operands = _operands()
    grid = SimpleNamespace(
        dy_u=jnp.asarray(operands["e2u"]),
        dx_v=jnp.asarray(operands["e1v"]),
        dx_u=jnp.asarray(1.0 / operands["r1_e1u"]),
        dy_v=jnp.asarray(1.0 / operands["r1_e2v"]),
        area_T=jnp.asarray(1.0 / operands["r1_e1e2t"]),
        area_q=jnp.asarray(1.0 / operands["r1_e1e2f"]),
    )
    args = (
        jnp.asarray(operands["u"]), jnp.asarray(operands["v"]), grid,
        jnp.ones(4), jnp.ones(5), jnp.asarray(operands["e3t_kbb"]),
    )
    kwargs = {
        "mask": jnp.asarray(operands["tmask"]),
        "u_mask": jnp.asarray(operands["umask"]),
        "v_mask": jnp.asarray(operands["vmask"]),
        "vertex_mask": jnp.asarray(operands["fmask"]),
        "thickness_operands": tuple(jnp.asarray(operands[name]) for name in (
            "e3t_kbb", "e3u_kbb", "e3v_kbb", "e3f_live",
            "e3u_kmm", "e3v_kmm")),
    }
    ordinary = nemo_ldf_lap_viscosity_e3_cgrid(*args, **kwargs)
    diagnostic = nemo_ldf_lap_viscosity_e3_cgrid(
        *args, **kwargs, return_intermediates=True)
    compiled = jax.jit(lambda uu, vv: nemo_ldf_lap_viscosity_e3_cgrid(
        uu, vv, grid, args[3], args[4], args[5], **kwargs))(
            args[0], args[1])
    assert len(ordinary) == 2 and len(diagnostic) == 3
    np.testing.assert_array_equal(ordinary[0], diagnostic[0])
    np.testing.assert_array_equal(ordinary[1], diagnostic[1])
    np.testing.assert_allclose(ordinary[0], compiled[0], rtol=1.0e-15, atol=0.0)
    np.testing.assert_allclose(ordinary[1], compiled[1], rtol=1.0e-15, atol=0.0)
    assert set(diagnostic[2]) >= {"zdiv", "zcur", "visc_u", "visc_v"}

    grad = jax.grad(lambda uu: jnp.sum(
        nemo_ldf_lap_viscosity_e3_cgrid(
            uu, args[1], grid, args[3], args[4], args[5], **kwargs)[0]
    ))(args[0])
    assert np.isfinite(np.asarray(grad)).all()


def _developed_payload() -> bytes:
    chunks = [gate.DEVELOPED_MAGIC.ljust(16).encode("ascii")]
    chunks.append(struct.pack(
        "=9i", 1, 1081, 1, 1, 3, gate.DEVELOPED_NX,
        gate.DEVELOPED_NY, gate.DEVELOPED_NZ, 64))
    sizes = tuple(
        gate.DEVELOPED_COUNT if kind == "3d" else gate.DEVELOPED_SURFACE_COUNT
        for _, kind in gate.DEVELOPED_FIELDS)
    chunks.append(struct.pack(f"={len(sizes)}i", *sizes))
    for name, kind in gate.DEVELOPED_FIELDS:
        count = (gate.DEVELOPED_COUNT if kind == "3d"
                 else gate.DEVELOPED_SURFACE_COUNT)
        fill = 0.0 if name in ("zcur", "zdiv") else 1.0
        chunks.append(np.full(count, fill, dtype=np.float64).tobytes())
    return b"".join(chunks)


def _family_payload() -> bytes:
    nx, ny, nz = 36, 26, 31
    count = nx * ny * nz
    fields = 10
    chunks = [b"NEMO_L2_R146FAM".ljust(16), struct.pack(
        "=8i", 1, 1081, 1, 3, nx, ny, nz, 64),
        struct.pack(f"={fields}i", *((count,) * fields))]
    chunks.extend(np.ones(count, dtype=np.float64).tobytes()
                  for _ in range(fields))
    return b"".join(chunks)


def test_developed_record_admission_and_plants(tmp_path):
    payload = _developed_payload()
    assert len(payload) == gate.DEVELOPED_EXPECTED_SIZE
    record_path = tmp_path / gate.DEVELOPED_RECORD
    family_path = tmp_path / "oracle_developed_rhs_families_kt00001081.bin"
    stamp_path = tmp_path / f"{gate.DEVELOPED_RECORD}.stamp"
    record_path.write_bytes(payload)
    family_path.write_bytes(_family_payload())
    commit = "a" * 40
    stamp_path.write_text(
        f"{hashlib.sha256(payload).hexdigest()} {commit} {record_path.name}\n")

    report = gate.admit_developed_ldf_record(
        record_path, family_path, stamp_path, expect_commit=commit)
    assert report["status"] == "PASS"
    assert len(report["field_registry"]) == len(gate.DEVELOPED_FIELDS)
    assert report["owned_intermediate_cells"] == {
        "zcur": 35 * 25 * 30,
        "zdiv": 35 * 25 * 30,
    }
    for plant in ("header", "truncation", "missing-field", "zcur-ulp",
                  "post-ulp"):
        with pytest.raises(Exception):
            gate.admit_developed_ldf_record(
                record_path, family_path, stamp_path,
                expect_commit=commit, plant=plant)
