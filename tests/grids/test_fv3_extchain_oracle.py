"""Composed ext-chain oracle certificate tests.

Consumes ``tests/grids/fixtures/fv3_extchain_oracle_c12.npz`` produced by
``scripts/validate/fv3_native/compare_extchain_oracle.py`` from the
per-rank dumps of ``fv3_extchain_oracle_driver.F90`` (real pinned-tree
FV3 duo init + real FMS mpp exchange + real composed
``ext_scalar``/``ext_vector``; see
``scripts/cluster/fv3_native/extchain_oracle.sbatch``).

Tiers:
- verbatim-ness: the extract file's byte-verbatim blocks must match the
  pinned ``fv_duogrid.F90`` line ranges with the documented deviation
  lines stripped (skips off-cluster);
- fixture integrity: input sha recomputed from the stored arrays;
- oracle chain certification: driver CERT values (max|staged-composed|)
  are exactly 0.0;
- interior identity: neither side's exchange may touch compute-domain
  slots (bitwise vs input);
- constant-field exactness: level k=1 is constant, and both the oracle
  and the port preserve it through the full composed chain (Lagrange
  partition of unity) — port final vs oracle composed final at k=1 must
  agree to 1e-12.

The QUANTITATIVE per-stage/per-region divergence table is the compare
script's output (and the summary npz), not an assertion here — the
diagnosis, not the test, owns the verdict on the 1e-6 hypothesis.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "fv3_extchain_oracle_c12.npz"
PINNED = Path("/burg-archive/glab/users/pg2328/fv3_oracle_pinned/"
              "atmos_cubed_sphere-symmetryclean/tools/fv_duogrid.F90")
EXTRACT = (Path(__file__).parents[2] / "scripts" / "validate"
           / "fv3_native" / "fv3_extchain_extract.F90")

# byte-verbatim blocks: (pinned line range, extract subroutine name)
VERBATIM_RANGES = [
    ((977, 1137), "cube_rmp"),
    ((1177, 1420), "fill_corners_domain_decomp_2d"),
    ((1422, 1705), "fill_corners_domain_decomp_3d"),
    ((2590, 2674), "cubed_a2c_halo"),
    ((2676, 2763), "cubed_a2d_halo"),
    ((2765, 2844), "c2l_ord2_cgrid"),
]
# staged copies: verbatim modulo the documented deviation lines
STAGED_RANGES = [((505, 569), "ext_scalar_3d"), ((626, 975), "ext_vector")]


def _load():
    if not FIXTURE.exists():
        pytest.skip("extchain oracle fixture not generated "
                    "(scripts/cluster/fv3_native/extchain_oracle.sbatch)")
    return np.load(FIXTURE, allow_pickle=False)


# ---------------------------------------------------------------------------
# verbatim-ness (machine-enforced extract certificate)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not PINNED.exists(),
                    reason="pinned oracle tree not on this host")
def test_extract_blocks_are_verbatim():
    pinned = PINNED.read_text().splitlines()
    ext = EXTRACT.read_text()

    for (a, b), name in VERBATIM_RANGES:
        block = "\n".join(pinned[a - 1:b])
        assert block in ext, f"{name}: pinned {a}-{b} not byte-verbatim"

    # staged copies: strip documented deviation lines, undo the renames,
    # then require byte equality with the pinned range
    kept = []
    for line in ext.splitlines():
        if "EXTCHAIN-HOOK" in line:
            continue
        line = re.sub(r"\s*! EXTCHAIN renamed \(deviation D2\)$", "", line)
        line = line.replace(
            "ext_scalar_3d_staged(var, dg, bd, domain, istag, jstag, tag)",
            "ext_scalar_3d(var, dg, bd, domain, istag, jstag)")
        line = line.replace("end subroutine ext_scalar_3d_staged",
                            "end subroutine ext_scalar_3d")
        line = line.replace(
            "ext_vector_staged(u_in, v_in, dg, bd, domain, gridstruct, "
            "flagstruct, ieu_stag, jeu_stag, iev_stag, jev_stag, tag)",
            "ext_vector(u_in, v_in, dg, bd, domain, gridstruct, "
            "flagstruct, ieu_stag, jeu_stag, iev_stag, jev_stag)")
        line = line.replace("end subroutine ext_vector_staged",
                            "end subroutine ext_vector")
        kept.append(line)
    normalized = "\n".join(kept)
    for (a, b), name in STAGED_RANGES:
        block = "\n".join(pinned[a - 1:b])
        assert block in normalized, \
            f"{name}: staged copy deviates beyond the documented lines"


def test_extract_hook_lines_all_marked():
    """Every dump call inside the extract bodies carries the marker —
    an unmarked insertion would evade the verbatim check."""
    body = EXTRACT.read_text()
    in_extract = body.split("module extchain_extract_mod", 1)[1]
    for ln in in_extract.splitlines():
        if "extchain_dump3" in ln and "use extchain_dump_mod" not in ln \
                and "only:" not in ln:
            assert "EXTCHAIN-HOOK" in ln, ln


# ---------------------------------------------------------------------------
# fixture integrity + oracle certification
# ---------------------------------------------------------------------------

def test_fixture_input_sha_enforced():
    d = _load()
    h = hashlib.sha256()
    for t in range(1, 7):
        names = sorted(k[len(f"t{t}_"):] for k in d.files
                       if k.startswith(f"t{t}_"))
        for nm in names:
            if nm.startswith("IN_") and nm.endswith("_v1"):
                h.update(np.ascontiguousarray(d[f"t{t}_{nm}"]).tobytes())
    assert h.hexdigest() == str(d["input_sha256"])


def test_oracle_chain_certification_bitwise():
    d = _load()
    meta = json.loads(str(d["meta"]))
    certs = [ln for ln in meta["notes"] if ln.startswith("CERT")]
    assert len(certs) >= 6 * 4 * 2      # 4 families x 2 variants x 6 tiles
    for ln in certs:
        vals = [float(x) for x in ln.split()[2:]]
        assert all(v == 0.0 for v in vals), ln
    assert meta["cert_ok"] is True


def test_interior_untouched_by_oracle_chain():
    """The exchange must never write compute-domain slots: oracle
    composed final == input, bitwise, on the interior."""
    d = _load()
    n, ng = int(d["n"]), int(d["ng"])
    sl = slice(ng, ng + n)
    for t in range(1, 7):
        for fam in ("A", "DU", "DV", "CU", "CV"):
            fin = d[f"t{t}_CFIN_{fam}_v1"]
            inp = d[f"t{t}_IN_{fam}_v1"]
            # interior = strictly-compute slots common to all staggers
            df = np.abs(fin[sl, sl] - inp[sl, sl]).max()
            assert df == 0.0, (t, fam, df)


def test_constant_field_preserved_both_sides():
    """Level k=1 is constant 7.25: Lagrange partition of unity means the
    composed chain must return it everywhere, oracle AND port."""
    from legoesm.grids.fv3_native_ext_vector import (
        build_ext_context,
        ext_scalar_sixface,
    )
    from legoesm.grids.fv3_native_gridstruct import (
        build_fv3_native_gridstruct,
    )

    d = _load()
    n, ng = int(d["n"]), int(d["ng"])
    for t in range(1, 7):
        w = np.abs(d[f"t{t}_CFIN_A_v1"][:, :, 0] - 7.25).max()
        assert w < 1e-12, (t, w)

    gs6 = [build_fv3_native_gridstruct(n, ng, tile=t)
           for t in range(1, 7)]
    ectx = build_ext_context(n, ng, gs6)
    f6 = [np.array(d[f"t{t}_IN_A_v1"][:, :, 0], copy=True)
          for t in range(1, 7)]
    ext_scalar_sixface(f6, "A", ectx)
    for t in range(6):
        assert np.abs(f6[t] - 7.25).max() < 1e-12, t


def test_face_map_is_identity():
    """Instrument control C1, re-checked from the committed fixture."""
    from legoesm.grids.fv3_native_gridstruct import (
        build_fv3_native_gridstruct,
    )

    d = _load()
    n, ng = int(d["n"]), int(d["ng"])
    worst = 0.0
    for t in range(1, 7):
        gs = build_fv3_native_gridstruct(n, ng, tile=t)
        dlon = np.abs(np.mod(gs["agrid_lon"] - d[f"t{t}_AG_LON"]
                             + np.pi, 2 * np.pi) - np.pi).max()
        dlat = np.abs(gs["agrid_lat"] - d[f"t{t}_AG_LAT"]).max()
        worst = max(worst, float(dlon), float(dlat))
    assert worst < 1e-12, worst
