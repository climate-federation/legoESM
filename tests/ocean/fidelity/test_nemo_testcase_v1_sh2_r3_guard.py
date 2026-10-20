"""V1 kt 1-2 bit-for-bit with zdf_sh2's r3u/r3v stored before the "1 +".

NEMO stores pr3u/pr3v (domqco.f90:266-269) before zdfsh2.f90:104-108 adds 1.
In the compiled V1 step XLA rewrites the r3v division by the constant face
area into a reciprocal multiply and fuses it with the "1 +" into one FMA; the
kt-2 diffusivity then misses NEMO at (116,45) and T/S inherit it. Removing the
guard (TKEConfig.nemo_sh2_r3_source_rounded) turns this test red.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import jax
import pytest

REPO = Path(__file__).resolve().parents[3]
CASES = REPO / "scripts/validate/ocean_fidelity/testcases"
RECORD = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
              "gridladder_vmix_rounds/round1/oracle_v1_r1")
TWINS = [RECORD / t for t in ("twin_a", "twin_b", "twin_c")]

pytestmark = [
    pytest.mark.skipif(not jax.config.x64_enabled, reason="binary64 receipt"),
    pytest.mark.skipif(not all(t.is_dir() for t in TWINS),
                       reason="the operator's V1 record is not on this machine"),
]


def test_v1_independent_kt1_2_every_field_bit_for_bit():
    sys.path.insert(0, str(CASES))
    spec = importlib.util.spec_from_file_location(
        "v1_r3_ladder", CASES / "nemo_testcase_g2_orca2bathy_ladder.py")
    ladder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ladder)
    out = ladder.run(TWINS[0], arm="independent", kt_from=1, kt_to=2,
                     restart_roots=TWINS, allow_dirty=True, case="ORCA2V1-tke")
    assert [k["kt"] for k in out["per_kt"]] == [1, 2]
    assert out["per_kt"][1]["all_levels"], "kt-2 level census missing"
    for k in out["per_kt"]:
        bad = [r["name"] for r in k["rows"] if r["n_unequal"]]
        for name, lv in (k["all_levels"] or {}).items():
            s = lv.get("summary")
            if (lv["n_unequal"] if s is None else
                    s["full"]["n_unequal"] or s["partial"]["n_unequal"]
                    or s["dry_n_unequal"]):
                bad.append(name)
        assert not bad, f"kt {k['kt']} unequal: {bad}"
