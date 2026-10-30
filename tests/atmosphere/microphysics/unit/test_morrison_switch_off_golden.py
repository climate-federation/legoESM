"""prognostic_droplet_number OFF leaves Morrison bit-identical (user decision 4a, 2026-10-19).

Golden ``oracle_fixtures/morrison_switch_off_golden.npz`` = every output leaf of
``morrison_microphysics`` for four switch-off configurations on the MG2 oracle
columns, a cold copy and a frozen copy (homogeneous freezing active), written by
``scripts/validate/oracle/morrison_mg2_warm_rain/make_switch_off_golden.py`` at the
commit recorded in its meta.  Compared bitwise.  Regenerated 2026-10-21 on the
rebase onto main (bit-identical to main) and 2026-10-22 for the smooth Bigg -4 C
step (user decision 3a: cold column set only, max relative change 1.8e-3).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[4]
GOLDEN = ROOT / "tests/atmosphere/microphysics/oracle_fixtures/morrison_switch_off_golden.npz"
GEN = ROOT / "scripts/validate/oracle/morrison_mg2_warm_rain/make_switch_off_golden.py"


def _gen():
    spec = importlib.util.spec_from_file_location("make_switch_off_golden", GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("config", sorted(_gen().CONFIGS))
def test_switch_off_is_bit_identical(config):
    gen = _gen()
    gold = dict(np.load(GOLDEN))
    meta = json.loads(str(gold.pop("meta")))
    env = gen.environment()
    hint = (f"golden recorded at legoESM {meta['legoesm_sha']} with jax {meta['jax']} / jaxlib "
            f"{meta['jaxlib']} on {meta['cpu']}; here jax {env['jax']} / jaxlib {env['jaxlib']} on "
            f"{env['cpu']}. A mismatch on a different jaxlib/CPU needs the golden regenerated AT "
            f"{meta['legoesm_sha']} (a reviewed change), never a looser comparison.")
    sets, dt = gen.column_sets()
    n = 0
    for sname, g in sets.items():
        got = gen.leaves(gen.run(gen.CONFIGS[config], g, dt))
        want = {k[len(f"{sname}/{config}/"):]: v for k, v in gold.items()
                if k.startswith(f"{sname}/{config}/")}
        assert set(got) == set(want), (sname, set(got) ^ set(want))
        for k, v in want.items():
            assert np.array_equal(got[k], v), f"{sname}/{config}/{k} differs. {hint}"
            n += 1
    assert n > 20
    # the frozen set must exercise homogeneous freezing, the process whose ice
    # number reads the droplet number (otherwise its bit identity is vacuous)
    assert np.abs(gold[f"frozen/{config}/['qc_budget']['homogeneous_freezing']"]).max() > 1e-12


def test_switch_defaults_off():
    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
    assert MorrisonConfig._field_defaults["prognostic_droplet_number"] is False
    # appended (positional ABI), then the two RULE 5 widths of 2026-10-22
    assert MorrisonConfig._fields[-3:] == (
        "prognostic_droplet_number", "mg2_presence_ramp_top", "mg2_frozen_rain_ice_ramp_frac")
