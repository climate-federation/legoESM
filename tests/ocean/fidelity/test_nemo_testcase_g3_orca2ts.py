"""Grid ladder rung G3 (DECISION 117): the certified G2 card and deck with ONE
switch, the initial state (ORCA2's T/S climatology read by &namtsd).

Deck and acquisition-script contracts need no data.  The card tests read
ORCA2's shipped domain and climatology files and skip when absent.  Nothing
here compares with a NEMO record: none exists yet.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe

REPO = Path(__file__).resolve().parents[3]
CASES = REPO / "scripts/validate/ocean_fidelity/testcases"
G3_DIR = CASES / "nemo_testcase_g3_orca2ts"
G2_DIR = CASES / "nemo_testcase_g2_orca2bathy"
INPUTS = Path("/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0")
FILES = {
    "ORCA_R2_zps_domcfg.nc": "7125f7a54e8693ff8f1327b878d2258d258878bc7302ac123ac671f2339d2839",
    "data_1m_potential_temperature_nomask.nc":
        "ca00905c27078305e80130ea6dd07fc575de2ae3257fad38003b329458cf7f7a",
    "data_1m_salinity_nomask.nc":
        "ad648d972f0631bde7e1b598d98470a979b7f2649271fc9cf5ccfca158e31d0c",
}
ORCA2_DECK = Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/ORCA2_OMIP_L4/EXP00/"
                  "namelist_cfg")
needs_inputs = pytest.mark.skipif(
    not all((INPUTS / name).is_file() for name in FILES), reason="no ORCA2 input files")
TSD = ("namtsd.ln_tsd_init", "namtsd.cn_dir", "namtsd.sn_tem", "namtsd.sn_sal")


@pytest.fixture(autouse=True)
def _fp64():
    before = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    yield
    set_policy(before)


def _deck(tmp_path, text, name="deck"):
    root = tmp_path / name
    root.mkdir()
    (root / "namelist_cfg").write_text(text)
    (root / "domain_cfg.nc").symlink_to(INPUTS / "ORCA_R2_zps_domcfg.nc")
    for data in list(FILES)[1:]:
        (root / data).symlink_to(INPUTS / data)
    return root


def test_g3_deck_is_g2_plus_orca2s_namtsd_lines():
    g2 = recipe._orca2g1_namelist_values(G2_DIR / "namelist_cfg")
    g3 = recipe._orca2g1_namelist_values(G3_DIR / "namelist_cfg")
    assert {k: v for k, v in g3.items() if k not in TSD} == {
        k: v for k, v in g2.items() if k not in TSD}
    assert (g2["namtsd.ln_tsd_init"], g3["namtsd.ln_tsd_init"]) == (False, True)
    assert g3["namtsd.ln_tsd_dmp"] is False and g3["namtra_dmp.ln_tradmp"] is False
    assert sorted(set(g3) - set(g2)) == ["namtsd.cn_dir", "namtsd.sn_sal", "namtsd.sn_tem"]
    assert recipe.read_orca2g1_deck(G3_DIR) == recipe.read_orca2g1_deck(G2_DIR)
    if ORCA2_DECK.is_file():        # the data lines are ORCA2's own
        orca2 = recipe._orca2g1_namelist_values(ORCA2_DECK)
        for name in TSD[1:]:
            assert "".join(str(g3[name]).split()) == "".join(str(orca2[name]).split())


def test_run_sh_reuses_g2s_binaries_and_pins_the_climatology():
    text = (G3_DIR / "run.sh").read_text()
    assert subprocess.run(["bash", "-n", str(G3_DIR / "run.sh")]).returncode == 0
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    assert not re.search(r"makenemo|maketools|\bmake\b", code)      # no build
    g2 = (G2_DIR / "run.sh").read_text()
    for pin in ("REF_SHA", "RUN_SHA", "SHIPPED_DOMCFG_SHA"):
        want = re.search(rf"readonly {pin}=([0-9a-f]{{64}})", g2).group(1)
        assert re.search(rf"readonly {pin}=([0-9a-f]{{64}})", text).group(1) == want
    assert FILES["data_1m_potential_temperature_nomask.nc"] in text
    assert FILES["data_1m_salinity_nomask.nc"] in text
    assert "'3,5,7,9,0,0,0,0,0,0'" in code and "'2,4,6,8,10,0,0,0,0,0'" in code
    # after its last entry nitrst is 0: twin c's last restart is step 9, not 10
    assert 'run_one "$RUN_BIN" "$TWIN_C" 10 "$L_ODD" 9' in code
    # admission renders each run's whole deck, restart list included
    assert 'render_deck "$dry/admit_deck" "$steps" "$list"' in code
    assert "gridladder_rounds/round14/oracle_g3_r14" in code
    assert "does not state the T/S read with damping off" in code
    for name, sha in FILES.items():
        if (INPUTS / name).is_file():
            assert hashlib.sha256((INPUTS / name).read_bytes()).hexdigest() == sha


@pytest.fixture(scope="module")
def g3_root(tmp_path_factory):
    if not all((INPUTS / name).is_file() for name in FILES):
        pytest.skip("no ORCA2 input files")
    return _deck(tmp_path_factory.mktemp("g3"), (G3_DIR / "namelist_cfg").read_text())


@pytest.fixture(scope="module")
def g3_card(g3_root):
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    return recipe.build_nemo_testcase_card("ORCA2G3-ts", deck_root=g3_root)


@needs_inputs
def test_g3_card_is_the_g2_card_started_from_orca2s_climatology(g3_card, g3_root, tmp_path):
    g2_root = _deck(tmp_path, (G2_DIR / "namelist_cfg").read_text(), "g2")
    g2 = recipe.build_nemo_testcase_card("ORCA2G2-zps", deck_root=g2_root)
    a, b = g3_card.recipe.model_config, g2.recipe.model_config
    assert [f for f in a._fields if repr(getattr(a, f)) != repr(getattr(b, f))] == []
    assert recipe._orca2g1_selector_failures(g3_card) == []
    assert g3_card.unmeasured_features == recipe.ORCA2G3_UNMEASURED
    with pytest.raises(ValueError, match="not execution-ready"):
        recipe.validate_orca2g1_flat_card_for_execution(g3_card, g3_root)
    s3, s2 = g3_card.recipe.initial_state, g2.recipe.initial_state
    assert np.array_equal(np.asarray(s3.eta.data), np.asarray(s2.eta.data))
    assert np.array_equal(np.asarray(g3_card.recipe.land_mask), np.asarray(g2.recipe.land_mask))
    wet = np.asarray(g3_card.recipe.z_coord.is_active).astype(bool)
    t3 = np.asarray(s3.T.data)
    assert np.all(t3[~wet] == 0.0) and not np.any(t3[wet] == 20.0)
    # NEMO's dtatsd.f90:219 skips the ORCA_R2 hand alterations with ln_tsd_dmp
    # off: the card is the unaltered climatology, which differs from the
    # altered one on exactly the ORCA2 lane's 1,283 T and 720 S cells.
    t_alt, s_alt = recipe.build_orca2_initial_ts(
        INPUTS / "data_1m_potential_temperature_nomask.nc",
        INPUTS / "data_1m_salinity_nomask.nc", wet)
    assert int((t3 != t_alt).sum()) == 1283
    assert int((np.asarray(s3.S.data) != s_alt).sum()) == 720


@needs_inputs
def test_g3_refuses_planted_disagreements(g3_card, g3_root, tmp_path):
    state = g3_card.recipe.initial_state
    deep = np.asarray(state.T.data).copy()
    deep[73, 50, 20] += 1.0e-12
    with pytest.raises(ValueError, match="initial_T"):
        recipe.validate_orca2g1_flat_card(g3_card._replace(recipe=g3_card.recipe._replace(
            initial_state=state._replace(T=state.T.replace(data=deep)))), g3_root)
    text = (G3_DIR / "namelist_cfg").read_text()
    plants = {
        "dmp_on": (text.replace("ln_tsd_dmp  = .false.", "ln_tsd_dmp  = .true."), "ln_tsd_dmp"),
        "file": (text.replace("'data_1m_salinity_nomask'", "'sali_ref_clim_monthly'"), "sn_sal"),
        "dt": (text.replace("rn_Dt       =  10800.", "rn_Dt       =   5400."), "rn_dt"),
        "date": (text.replace("nn_date0    =  010101", "nn_date0    =  010115"), "nn_date0"),
        "init_off": (text.replace("ln_tsd_init = .true.", "ln_tsd_init = .false."),
                     "ln_tsd_init"),
    }
    for name, (planted, match) in plants.items():
        assert planted != text, name
        with pytest.raises(ValueError, match=match):
            recipe.orca2g_initial_ts(_deck(tmp_path, planted, name), "ORCA2G3-ts",
                                     np.ones((148, 180, 30), dtype=bool))
    # the G2 rung refuses a deck that reads T/S, and names no data files
    with pytest.raises(ValueError, match="ln_tsd_init"):
        recipe.orca2g_initial_ts(g3_root, "ORCA2G2-zps", np.ones((148, 180, 30), dtype=bool))
    with pytest.raises(ValueError, match="not a grid-ladder rung"):
        recipe.orca2g_initial_ts(g3_root, "ORCA2-zps", np.ones((148, 180, 30), dtype=bool))
