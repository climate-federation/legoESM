"""Validate every committed tier-JSON fixture against the schema validator.

Acts as a CI guard: a typo in any fixture under
``tests/ocean/fidelity/fixtures/`` fails this test long before the Layer A
metric tests try to consume it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from legoesm.ocean.fidelity import tolerances

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"

EXPECTED_FIXTURES = {
    0: "tier0_invariants.json",
    1: "tier1_linear_waves.json",
    2: "tier2_geostrophic_thermalwind.json",
    3: "tier3_process_benchmarks.json",
    4: "tier4_wind_driven_gyres.json",
    5: "tier5_baroclinic_instability.json",
    6: "tier6_channel_circulation.json",
    7: "tier7_dino.json",
    8: "tier8_global_realistic.json",
}


def test_all_nine_tier_fixtures_present():
    for tier, name in EXPECTED_FIXTURES.items():
        path = FIXTURE_DIR / name
        assert path.exists(), f"missing tier-{tier} fixture: {path}"


@pytest.mark.parametrize(
    "tier,filename",
    sorted(EXPECTED_FIXTURES.items()),
    ids=[f"tier{tier}" for tier in sorted(EXPECTED_FIXTURES)],
)
def test_fixture_validates(tier, filename):
    parsed = tolerances.load_tier_file(FIXTURE_DIR / filename)
    assert parsed.tier == tier
    assert parsed.cases, f"tier {tier} fixture has empty 'cases'"
    for case_name, case in parsed.cases.items():
        lo, hi = case.tolerance_window
        assert lo <= hi, f"{filename}:{case_name}: tolerance lo > hi"
        assert case.ci_marker in ("fast", "nightly", "manual_only")
