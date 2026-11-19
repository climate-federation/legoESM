"""Keep shared GYRE kernels from moving the shipped card farther from NEMO.

The card digest pins configuration, not oracle distance.  These no-slack
ceilings are the seventy scalar-math ten-step residuals measured after round
9.  A future improvement may lower them; worsening one requires an explicit
ceiling change and receipt disposition.
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

# |legoESM - NEMO| absolute maxima, scalar-libm oracle, round 9.
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
    'GYRE-zco.kt3.after.uu_b': 2.2605800911734697e-11,
    'GYRE-zco.kt3.after.vv_b': 1.5035459639475834e-11,
    'GYRE-zco.kt3.before.S': 1.0542322570472606e-10,
    'GYRE-zco.kt3.before.T': 6.610889613511972e-11,
    'GYRE-zco.kt3.before.ssh': 6.695688014451973e-11,
    'GYRE-zco.kt3.before.u': 1.0046085057585641e-11,
    'GYRE-zco.kt3.before.v': 7.112879729653798e-12,
    'GYRE-zco.kt4.after.uu_b': 4.159267570115477e-11,
    'GYRE-zco.kt4.after.vv_b': 2.593889599639132e-11,
    'GYRE-zco.kt4.before.S': 1.1883116712851916e-10,
    'GYRE-zco.kt4.before.T': 7.358380571531598e-11,
    'GYRE-zco.kt4.before.ssh': 1.0434915128157418e-10,
    'GYRE-zco.kt4.before.u': 2.2876856225345632e-11,
    'GYRE-zco.kt4.before.v': 1.7007912439526152e-11,
    'GYRE-zco.kt5.after.uu_b': 7.270737744508038e-11,
    'GYRE-zco.kt5.after.vv_b': 4.481261976080342e-11,
    'GYRE-zco.kt5.before.S': 1.5083401194715407e-10,
    'GYRE-zco.kt5.before.T': 2.0171952996861364e-10,
    'GYRE-zco.kt5.before.ssh': 1.2581730319052853e-10,
    'GYRE-zco.kt5.before.u': 4.189619528041644e-11,
    'GYRE-zco.kt5.before.v': 2.8117369543778636e-11,
    'GYRE-zco.kt6.after.uu_b': 1.0778717154935613e-10,
    'GYRE-zco.kt6.after.vv_b': 6.842757567168678e-11,
    'GYRE-zco.kt6.before.S': 1.7497825410828227e-10,
    'GYRE-zco.kt6.before.T': 1.27631238910908e-10,
    'GYRE-zco.kt6.before.ssh': 2.2301183816297865e-10,
    'GYRE-zco.kt6.before.u': 7.313652960160011e-11,
    'GYRE-zco.kt6.before.v': 4.747487125644767e-11,
    'GYRE-zco.kt7.after.uu_b': 1.5212944575945797e-10,
    'GYRE-zco.kt7.after.vv_b': 9.812501679251628e-11,
    'GYRE-zco.kt7.before.S': 2.0970958303223597e-10,
    'GYRE-zco.kt7.before.T': 1.6996537510749476e-10,
    'GYRE-zco.kt7.before.ssh': 2.6463523820213775e-10,
    'GYRE-zco.kt7.before.u': 1.0809721672465356e-10,
    'GYRE-zco.kt7.before.v': 7.358581799454811e-11,
    'GYRE-zco.kt8.after.uu_b': 1.98736204576333e-10,
    'GYRE-zco.kt8.after.vv_b': 1.336954266913104e-10,
    'GYRE-zco.kt8.before.S': 2.4163426814993727e-10,
    'GYRE-zco.kt8.before.T': 1.7686474507172534e-10,
    'GYRE-zco.kt8.before.ssh': 3.655319102957577e-10,
    'GYRE-zco.kt8.before.u': 1.5283083826256316e-10,
    'GYRE-zco.kt8.before.v': 9.929844446698866e-11,
    'GYRE-zco.kt9.after.uu_b': 2.5385125512061846e-10,
    'GYRE-zco.kt9.after.vv_b': 1.7753286297647608e-10,
    'GYRE-zco.kt9.before.S': 2.531592713239661e-10,
    'GYRE-zco.kt9.before.T': 2.2794566234551894e-10,
    'GYRE-zco.kt9.before.ssh': 3.715257910347214e-10,
    'GYRE-zco.kt9.before.u': 1.9967004482104866e-10,
    'GYRE-zco.kt9.before.v': 1.3427844426650015e-10,
    'GYRE-zco.kt10.after.uu_b': 3.1291732920102344e-10,
    'GYRE-zco.kt10.after.vv_b': 2.2271547713698103e-10,
    'GYRE-zco.kt10.before.S': 2.8397550977388164e-10,
    'GYRE-zco.kt10.before.T': 3.006057625043468e-10,
    'GYRE-zco.kt10.before.ssh': 5.238766694753982e-10,
    'GYRE-zco.kt10.before.u': 2.554526182599326e-10,
    'GYRE-zco.kt10.before.v': 1.807891145816365e-10,
}


def _load_gate():
    spec = importlib.util.spec_from_file_location(
        "gyre_phase3_gate_ceiling", GATE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scored_rows(report: dict) -> dict[str, float]:
    rows = {}
    for section in ("steps", "barotropic_state_steps"):
        for step in report.get(section) or ():
            for row in step["rows"]:
                rows[row["name"]] = float(row["absolute_max"])
    return rows


def rows_over_ceiling(report: dict, ceilings: dict[str, float]) -> list[str]:
    rows = scored_rows(report)
    problems = [f"unscored ceiling {name}" for name in sorted(
        set(ceilings) - set(rows))]
    problems += [f"row without ceiling {name}" for name in sorted(
        set(rows) - set(ceilings))]
    problems += [
        f"{name}: {rows[name]!r} > ceiling {ceilings[name]!r}"
        for name in sorted(set(rows) & set(ceilings))
        if rows[name] > ceilings[name]
    ]
    return problems


def _report_at_ceilings() -> dict:
    return {"steps": [{"rows": [
        {"name": name, "absolute_max": value}
        for name, value in CEILINGS.items()]}]}


def test_ceiling_comparator_covers_all_rows_and_fires_on_one_worsening():
    assert len(CEILINGS) == 70
    report = _report_at_ceilings()
    assert rows_over_ceiling(report, CEILINGS) == []
    worst = max(CEILINGS, key=CEILINGS.get)
    for row in report["steps"][0]["rows"]:
        if row["name"] == worst:
            row["absolute_max"] = CEILINGS[worst] * (1.0 + 2.0 ** -40)
    assert rows_over_ceiling(report, CEILINGS)[0].startswith(worst)


def test_shipped_card_stays_within_round9_nemo_distance(monkeypatch):
    gate = _load_gate()
    if not Path(gate.ROOT).is_dir():
        pytest.skip(f"one-degree scalar-math records absent: {gate.ROOT}")
    monkeypatch.setattr(
        gate, "worktree_stamp", lambda *args, **kwargs: {"ceiling": True})
    report = gate.run(gate.ROOT, max_step=10, trajectory_only=True)
    assert rows_over_ceiling(report, CEILINGS) == []
