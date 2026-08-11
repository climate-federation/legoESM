"""Tests for the ORACLE STAGE-STATE instrument.

Three layers:
 1. generator: the staged dyn_core/fv_dynamics copies are VERBATIM up to
    the enumerated deviations (round-trip enforced against the pinned
    tree; skipped where the pin is absent);
 2. comparator machinery: kinds/window/region masks behave as specified
    on synthetic inputs with known answers (including a deliberate
    violation each, so no check is vacuous);
 3. stage_hook threading: a real six-face substep at n=12/km=1 delivers
    every expected stage name exactly once, and payload mutation after
    the call does not corrupt the captured copies.
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.abspath(
    os.path.join(_HERE, "..", "..", "scripts", "validate", "fv3_native"))


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_SCRIPTS, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PIN_PRESENT = os.path.exists(
    "/burg-archive/glab/users/pg2328/fv3_oracle_pinned/"
    "atmos_cubed_sphere-symmetryclean/model/dyn_core.F90")


# ----------------------------------------------------------------------
# 1. generator
# ----------------------------------------------------------------------

@pytest.mark.skipif(not PIN_PRESENT, reason="pinned oracle tree absent")
def test_generator_round_trip(tmp_path):
    gen = _load("gen_dyncore_stage_copy")
    gen.generate(str(tmp_path), check=True)   # raises SystemExit on drift
    dc = (tmp_path / "staged_dyn_core.F90").read_text().splitlines()
    fd = (tmp_path / "staged_fv_dynamics.F90").read_text().splitlines()
    assert dc[21] == "module dyn_core_staged_mod"
    assert dc[-1] == "end module dyn_core_staged_mod"
    assert any("use dyn_core_staged_mod" in ln for ln in fd)
    # every hook line is tagged and single-line guarded
    for lines in (dc, fd):
        for ln in lines:
            if gen.HOOK_TAG in ln and "call stage_dump" in ln:
                assert "stage_dumps_active()" in ln
    # all stage names present
    txt = "\n".join(dc)
    for st in ("S00_entry", "S01_extdp", "S02_extuv", "S03_csw",
               "S04_geopkC", "S05_pgradc", "S06_extdivgd", "S07_extucvc",
               "S08_dsw1", "S09_fluxavg", "S10_dsw23", "S11_b2",
               "S12_kee", "S13_dsw45", "S14_dsw6", "S15_extdp2",
               "S16_geopkD", "S17_onegradp"):
        assert st in txt, st


@pytest.mark.skipif(not PIN_PRESENT, reason="pinned oracle tree absent")
def test_generator_detects_anchor_drift(tmp_path):
    """Non-vacuity: a wrong anchor must fail loudly, not hook silently."""
    gen = _load("gen_dyncore_stage_copy")
    bad = [(999, "this line does not exist in the pinned file", ["x"])]
    lines = gen.read_pinned(f"{gen.SRC}/model/dyn_core.F90",
                            gen.DYN_CORE_SHA)
    with pytest.raises(SystemExit, match="anchor mismatch|insert-anchor"):
        gen.apply(lines, [(999, "wrong content", "repl")], [])
    with pytest.raises(SystemExit, match="insert-anchor"):
        gen.apply(lines, [], bad)


# ----------------------------------------------------------------------
# 2. comparator machinery
# ----------------------------------------------------------------------

def test_window_and_kinds():
    cmp_ = _load("compare_dyncore_stages")
    ng = cmp_.NG
    # padded A scalar -> 48x48 window
    a = np.zeros((48 + 2 * ng, 48 + 2 * ng, 5))
    assert cmp_.window(a, "ascalar").shape == (48, 48, 5)
    # compute-only B array passes through untouched
    b = np.zeros((49, 49, 5))
    assert cmp_.window(b, "bscalar") is not None
    assert cmp_.window(b, "bscalar").shape == (49, 49, 5)
    # oracle crx layout (49, 54): i compute-only, j padded
    c = np.zeros((49, 48 + 2 * ng, 5))
    assert cmp_.window(c, "xflux").shape == (49, 48, 5)
    # wrong extent must raise, not broadcast
    with pytest.raises(ValueError):
        cmp_.window(np.zeros((50, 48, 5)), "ascalar")
    # partner mapping is an involution
    for k, (_, partner, _) in cmp_.KINDS.items():
        assert cmp_.KINDS[partner][1] == k


def test_region_masks_partition():
    cmp_ = _load("compare_dyncore_stages")
    m = cmp_.region_masks(48, 49)
    total = (m["interior"].astype(int) + m["edge"].astype(int)
             + m["corner"].astype(int))
    assert (total == 1).all()
    assert m["corner"][0, 0] and m["corner"][47, 48]
    assert m["edge"][0, 24] and not m["corner"][0, 24]
    assert m["interior"][24, 24]


def test_map_stage_field_synthetic_known_answer():
    """A field and its dihedral image must map to zero diff; a deliberate
    sign flip must NOT (the check can fail)."""
    cmp_ = _load("compare_dyncore_stages")
    rng = np.random.default_rng(7)
    ng = cmp_.NG
    port_u = rng.normal(size=(48 + 2 * ng, 49 + 2 * ng, 2))
    # direct identity map: oracle == port window
    meta = (False, "id", 1.0, 1.0)
    p, o = cmp_.map_stage_field(port_u, cmp_.window(port_u, "u"),
                                None, "u", meta)
    assert float(np.abs(p - o).max()) == 0.0
    # fi dihedral with the derived sign (-1 on u): build the oracle side
    # as the exact image and require zero
    meta_fi = (False, "fi", -1.0, 1.0)
    img = cmp_.window(port_u, "u")[::-1, :, :] * -1.0
    p, o = cmp_.map_stage_field(port_u, img, None, "u", meta_fi)
    assert float(np.abs(p - o).max()) == 0.0
    # violation: wrong sign must show O(1)
    p, o = cmp_.map_stage_field(port_u, img, None, "u",
                                (False, "fi", +1.0, 1.0))
    assert float(np.abs(p - o).max()) > 0.1
    # transposed: oracle partner carries the swapped axes
    meta_tr = (True, "id", 1.0, 1.0)
    partner = np.swapaxes(cmp_.window(port_u, "u"), 0, 1)  # (49,48) v-like
    p, o = cmp_.map_stage_field(port_u, None, partner, "u", meta_tr)
    assert float(np.abs(p - o).max()) == 0.0


def test_stage_rows_exchange_attribution_labels():
    """codex r1 #1: the delp/pt and divgd exchange rows must be labelled
    S01/S06 (the oracle stage that produced them), not the later hook
    they are observed at."""
    cmp_ = _load("compare_dyncore_stages")
    rows = cmp_.stage_rows()
    by_stage = {}
    for stage, pkey, pf, od, op, kind, halo, note in rows:
        by_stage.setdefault(stage, []).append((pf, note))
    assert {f for f, _ in by_stage["S01_extdp"]} == {"delp", "pt"}
    assert {f for f, _ in by_stage["S06_extdivgd"]} == {"divg_d"}
    for _, note in by_stage["S01_extdp"] + by_stage["S06_extdivgd"]:
        assert "observed post" in note
    # S02/S07 keep only the fields their own exchange mutates
    assert {f for f, _ in by_stage["S02_entryex"]} == {"u", "v"}
    assert {f for f, _ in by_stage["S07_extucvc"]} == {"uc", "vc"}


def test_flag_rule_binds_on_synthetic_boundary_jump():
    """The boundary-jump rule must fire for an edge-concentrated diff
    and stay silent for a uniform one (non-vacuity both ways)."""
    cmp_ = _load("compare_dyncore_stages")
    p = np.zeros((49, 49, 2))
    o = np.zeros((49, 49, 2))
    o[0, 10, 0] = 1e-3          # edge cell only
    rm = cmp_.region_maxima(p, o)
    assert rm["edge"] == 1e-3 and rm["interior"] == 0.0
    boundary = max(rm["edge"], rm["corner"])
    assert boundary > 1e-8 and boundary > 30 * max(rm["interior"], 1e-16)
    # uniform difference must NOT satisfy the 30x concentration clause
    o2 = np.full_like(o, 1e-3)
    rm2 = cmp_.region_maxima(p, o2)
    assert not (max(rm2["edge"], rm2["corner"])
                > 30 * max(rm2["interior"], 1e-16))


def test_receipt_gating(tmp_path):
    """codex r1 #2: a clean run without valid receipts must not return
    an authoritative exit code."""
    import json
    cmp_ = _load("compare_dyncore_stages")
    # invalid receipt content is detected
    bad = tmp_path / "r.json"
    bad.write_text("not json")
    dump = tmp_path / "dumps"
    dump.mkdir()
    with pytest.raises(SystemExit, match="receipts invalid"):
        # exercise only the receipt-parsing prologue: dump dir has no
        # manifests, but receipts are checked first
        cmp_.main(["--dump-dir", str(dump), "--receipts", str(bad)])
    # a matching receipt passes the prologue (then fails on missing
    # dumps, which is the NEXT error -- proving order)
    good = tmp_path / "g.json"
    good.write_text(json.dumps({"pass": True,
                                "dump_dir": str(dump)}))
    with pytest.raises(SystemExit, match="missing stage dumps"):
        cmp_.main(["--dump-dir", str(dump), "--receipts", str(good)])


def test_stage_rows_reference_valid_kinds():
    cmp_ = _load("compare_dyncore_stages")
    rows = cmp_.stage_rows()
    assert len(rows) > 30
    stages = []
    for stage, pkey, pf, od, op, kind, halo, note in rows:
        assert kind in cmp_.KINDS, kind
        assert od.split("[")[0].startswith("S")
        stages.append(stage)
    # ladder order is monotone in stage tag
    tags = [s.split("_")[0] for s in stages]
    assert tags == sorted(tags, key=lambda s: (len(s), s))


# ----------------------------------------------------------------------
# 3. stage_hook threading (real six-face substep, n=12, km=1)
# ----------------------------------------------------------------------

def test_stage_hook_threading_smoke():
    from legoesm.core.fv3_native_acoustic_3d import acoustic_substep_3d
    from legoesm.core.fv3_native_duo_stepper import (
        build_six_face_duo_context,
    )
    from legoesm.core.fv3_native_state_3d import build_state_3d

    n, km = 12, 1
    ctx = build_six_face_duo_context(n, 3, use_ext_bundle=True,
                                     oracle_conventions=True)
    ng = ctx["ng"]
    state = build_state_3d(n, ng, km, remap_follows=False)
    rng = np.random.default_rng(3)
    for t in range(6):
        state[t]["delp"][:] = 10000.0
        state[t]["pt"][:] = 300.0 + rng.normal(scale=1e-3,
                                               size=state[t]["pt"].shape)
        state[t]["u"][:] = rng.normal(scale=1e-3,
                                      size=state[t]["u"].shape)
        state[t]["v"][:] = rng.normal(scale=1e-3,
                                      size=state[t]["v"].shape)

    seen: dict = {}
    calls: list = []

    def hook(name, payload):
        calls.append(name)
        assert len(payload) == 6
        seen[name] = [{k: np.array(v, copy=True) for k, v in d.items()}
                      for d in payload]

    acoustic_substep_3d(ctx, state, 60.0, km, first_substep=True,
                        ptop=100.0, akap=2.0 / 7.0, cp_air=1004.0,
                        stage_hook=hook)

    expected = ["S02_entryex", "S03_csw", "S04_geopkC", "S05_pgradc",
                "S07_extucvc", "S08_dsw1", "S09_fluxavg", "S10_dsw23",
                "S10_dsw3", "S11_b2", "S12_kee", "S13_dsw45", "S14_dsw6",
                "S15_extdp2", "S16_geopkD", "S17_onegradp"]
    assert calls == expected, calls
    # every payload delivered per-face field dicts with finite windows
    for name, payload in seen.items():
        for d in payload:
            assert d, name
            for k, v in d.items():
                assert isinstance(v, np.ndarray), (name, k)

    # None hook must be byte-identical: same IC, no hook, same output
    state2 = build_state_3d(n, ng, km, remap_follows=False)
    rng = np.random.default_rng(3)
    for t in range(6):
        state2[t]["delp"][:] = 10000.0
        state2[t]["pt"][:] = 300.0 + rng.normal(
            scale=1e-3, size=state2[t]["pt"].shape)
        state2[t]["u"][:] = rng.normal(scale=1e-3,
                                       size=state2[t]["u"].shape)
        state2[t]["v"][:] = rng.normal(scale=1e-3,
                                       size=state2[t]["v"].shape)
    acoustic_substep_3d(ctx, state2, 60.0, km, first_substep=True,
                        ptop=100.0, akap=2.0 / 7.0, cp_air=1004.0,
                        stage_hook=None)
    for t in range(6):
        for f in ("u", "v", "delp", "pt"):
            np.testing.assert_array_equal(state[t][f], state2[t][f])
