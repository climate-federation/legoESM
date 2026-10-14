"""The shipped one-degree GYRE card may not move farther from NEMO.

Round 9 of the quarter-degree lane changed shared stage-3 tracer code that the
shipped one-degree card also runs.  The digest test pins the card's
configuration, not its distance from NEMO, so a shared-kernel change could
move the card away from NEMO and still pass (round-8 review, N2).

This guard runs the shipped card's ten-step trajectory gate
(``nemo_testcase_l2_gyre_phase3_gate.py --trajectory-only --max-step 10``,
NEMO's scalar-math records) and holds every scored row to the absolute
residual measured after round 9.  The ceilings are explicit and have no
slack: a change that moves a row closer may lower its ceiling; a change that
moves one farther must raise it in the same commit and say why in its
receipt.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

GATE_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts/validate/ocean_fidelity/testcases"
    / "nemo_testcase_l2_gyre_phase3_gate.py"
)

# |legoESM - NEMO| absolute maxima per scored row, measured after round 9.
# Provenance: docs/ocean/fidelity/testcases/
# nemo_testcases_l2_gyre_hires_round9_receipt.md, "Shipped one-degree card".
CEILINGS: dict[str, float] = {
    'GYRE-zco.kt1.after.uu_b': 0.0,
    'GYRE-zco.kt1.after.vv_b': 0.0,
    'GYRE-zco.kt1.before.S': 0.0,
    'GYRE-zco.kt1.before.T': 0.0,
    'GYRE-zco.kt1.before.ssh': 0.0,
    'GYRE-zco.kt1.before.u': 0.0,
    'GYRE-zco.kt1.before.v': 0.0,
    'GYRE-zco.kt2.after.uu_b': 9.911383888096259e-12,
    'GYRE-zco.kt2.after.vv_b': 6.886906518156766e-12,
    'GYRE-zco.kt2.before.S': 2.1316282072803006e-14,
    'GYRE-zco.kt2.before.T': 1.4210854715202004e-14,
    'GYRE-zco.kt2.before.ssh': 0.0,
    'GYRE-zco.kt2.before.u': 8.326672684688674e-17,
    'GYRE-zco.kt2.before.v': 9.71445146547012e-17,
    'GYRE-zco.kt3.after.uu_b': 2.2601994061066666e-11,
    'GYRE-zco.kt3.after.vv_b': 1.5039306822464682e-11,
    'GYRE-zco.kt3.before.S': 5.984546191939444e-10,
    'GYRE-zco.kt3.before.T': 1.1306994451842911e-08,
    'GYRE-zco.kt3.before.ssh': 6.695688014451973e-11,
    'GYRE-zco.kt3.before.u': 4.4348118233283884e-10,
    'GYRE-zco.kt3.before.v': 5.758083176932161e-10,
    'GYRE-zco.kt4.after.uu_b': 4.15925262980954e-11,
    'GYRE-zco.kt4.after.vv_b': 2.5939927506338223e-11,
    'GYRE-zco.kt4.before.S': 4.3936410065725795e-10,
    'GYRE-zco.kt4.before.T': 8.301572762547949e-09,
    'GYRE-zco.kt4.before.ssh': 1.0420936769756084e-10,
    'GYRE-zco.kt4.before.u': 7.390304063159547e-10,
    'GYRE-zco.kt4.before.v': 8.552787367221981e-10,
    'GYRE-zco.kt5.after.uu_b': 7.270799804240391e-11,
    'GYRE-zco.kt5.after.vv_b': 4.481195687959516e-11,
    'GYRE-zco.kt5.before.S': 6.745537461938511e-10,
    'GYRE-zco.kt5.before.T': 9.896020003452577e-09,
    'GYRE-zco.kt5.before.ssh': 1.263750722992063e-10,
    'GYRE-zco.kt5.before.u': 4.966081990076532e-10,
    'GYRE-zco.kt5.before.v': 1.2071744021069364e-09,
    'GYRE-zco.kt6.after.uu_b': 1.0779279378814177e-10,
    'GYRE-zco.kt6.after.vv_b': 6.843136213935397e-11,
    'GYRE-zco.kt6.before.S': 7.593428108521039e-10,
    'GYRE-zco.kt6.before.T': 1.153840401002526e-08,
    'GYRE-zco.kt6.before.ssh': 2.226802665872274e-10,
    'GYRE-zco.kt6.before.u': 6.384814413284978e-10,
    'GYRE-zco.kt6.before.v': 1.6474821368239745e-09,
    'GYRE-zco.kt7.after.uu_b': 1.5212615802479013e-10,
    'GYRE-zco.kt7.after.vv_b': 9.812017127616701e-11,
    'GYRE-zco.kt7.before.S': 8.785079330664303e-10,
    'GYRE-zco.kt7.before.T': 1.2780251523736297e-08,
    'GYRE-zco.kt7.before.ssh': 2.6502913579029563e-10,
    'GYRE-zco.kt7.before.u': 6.15639848128291e-10,
    'GYRE-zco.kt7.before.v': 8.267342123391419e-10,
    'GYRE-zco.kt8.after.uu_b': 1.9874314347023692e-10,
    'GYRE-zco.kt8.after.vv_b': 1.33696666151234e-10,
    'GYRE-zco.kt8.before.S': 1.1402434552110208e-09,
    'GYRE-zco.kt8.before.T': 1.1085905526897477e-08,
    'GYRE-zco.kt8.before.ssh': 3.642763955064021e-10,
    'GYRE-zco.kt8.before.u': 2.8881571960404084e-10,
    'GYRE-zco.kt8.before.v': 6.073268570992371e-10,
    'GYRE-zco.kt9.after.uu_b': 2.538492714643237e-10,
    'GYRE-zco.kt9.after.vv_b': 1.775335998002725e-10,
    'GYRE-zco.kt9.before.S': 9.98504390281596e-10,
    'GYRE-zco.kt9.before.T': 9.547562740408466e-09,
    'GYRE-zco.kt9.before.ssh': 3.6949572481542736e-10,
    'GYRE-zco.kt9.before.u': 4.83790150301866e-10,
    'GYRE-zco.kt9.before.v': 1.3218470415465866e-09,
    'GYRE-zco.kt10.after.uu_b': 3.129318635816669e-10,
    'GYRE-zco.kt10.after.vv_b': 2.2271635837650683e-10,
    'GYRE-zco.kt10.before.S': 4.464453695618431e-09,
    'GYRE-zco.kt10.before.T': 8.730108191912223e-09,
    'GYRE-zco.kt10.before.ssh': 5.234091528250051e-10,
    'GYRE-zco.kt10.before.u': 2.552889666156133e-10,
    'GYRE-zco.kt10.before.v': 8.789710487233648e-10,
}


def _load_gate():
    spec = importlib.util.spec_from_file_location(
        "gyre_phase3_gate_ceiling", GATE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scored_rows(report: dict) -> dict[str, float]:
    """Every certified trajectory row of a phase-3 report, by name."""
    rows = {}
    for section in ("steps", "barotropic_state_steps"):
        for step in report.get(section) or ():
            for row in step["rows"]:
                rows[row["name"]] = float(row["absolute_max"])
    return rows


def rows_over_ceiling(report: dict, ceilings: dict[str, float]) -> list[str]:
    """Rows above their ceiling, plus any row the table does not cover."""
    rows = scored_rows(report)
    problems = [f"unscored ceiling {name}" for name in sorted(
        set(ceilings) - set(rows))]
    problems += [f"row without ceiling {name}" for name in sorted(
        set(rows) - set(ceilings))]
    problems += [
        f"{name}: {rows[name]!r} > ceiling {ceilings[name]!r}"
        for name in sorted(set(rows) & set(ceilings))
        if rows[name] > ceilings[name]]
    return problems


def _report_at_ceilings(scale: float = 1.0) -> dict:
    return {"steps": [{"rows": [
        {"name": name, "absolute_max": value * scale}
        for name, value in CEILINGS.items()]}]}


def test_ceiling_table_covers_the_seventy_certified_rows():
    assert len(CEILINGS) == 70
    assert all(value >= 0.0 for value in CEILINGS.values())


def test_ceiling_comparator_fires_on_one_worsened_row():
    assert rows_over_ceiling(_report_at_ceilings(), CEILINGS) == []
    worst = max(CEILINGS, key=CEILINGS.get)
    report = _report_at_ceilings()
    for row in report["steps"][0]["rows"]:
        if row["name"] == worst:
            row["absolute_max"] = CEILINGS[worst] * (1.0 + 2.0 ** -40)
    assert rows_over_ceiling(report, CEILINGS) == [
        f"{worst}: {CEILINGS[worst] * (1.0 + 2.0 ** -40)!r} > ceiling "
        f"{CEILINGS[worst]!r}"]
    missing = _report_at_ceilings()
    missing["steps"][0]["rows"].pop()
    assert rows_over_ceiling(missing, CEILINGS)[0].startswith(
        "unscored ceiling")


def test_shipped_card_stays_within_its_round9_distance_from_nemo(monkeypatch):
    gate = _load_gate()
    if not Path(gate.ROOT).is_dir():
        pytest.skip(f"NEMO one-degree scalar-math records absent: {gate.ROOT}")
    # The numbers are the guard; provenance is the receipt's job, so the
    # guard also runs on an uncommitted tree or a scratch copy.
    monkeypatch.setattr(
        gate, "worktree_stamp", lambda *args, **kwargs: {"guard": "ceiling"})
    report = gate.run(gate.ROOT, max_step=10, trajectory_only=True)
    assert rows_over_ceiling(report, CEILINGS) == []
