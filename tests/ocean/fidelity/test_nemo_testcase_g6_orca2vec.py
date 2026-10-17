"""Grid-ladder G6: certified G5 plus ORCA2's vector-form momentum advection."""

from __future__ import annotations

import hashlib
import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe

REPO = Path(__file__).resolve().parents[3]
CASES = REPO / "scripts/validate/ocean_fidelity/testcases"
G5 = CASES / "nemo_testcase_g5_orca2ice"
G6 = CASES / "nemo_testcase_g6_orca2vec"
INPUTS = Path("/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0")
ORCA2 = Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/ORCA2_OMIP_L4/EXP00")
CARD_FILES = ("ORCA_R2_zps_domcfg.nc", "data_1m_potential_temperature_nomask.nc",
              "data_1m_salinity_nomask.nc")
needs_inputs = pytest.mark.skipif(
    not all((INPUTS / name).is_file() for name in CARD_FILES), reason="no ORCA2 inputs")
needs_nemo = pytest.mark.skipif(not ORCA2.is_dir(), reason="no NEMO tree")
SWITCH = {"namdyn_adv.ln_dynadv_off": (True, False),
          "namdyn_adv.ln_dynadv_vec": (False, True),
          "namdyn_adv.nn_dynkeg": (None, 0)}


@pytest.fixture(autouse=True)
def _fp64():
    before = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    yield
    set_policy(before)


def _values(path):
    return recipe._orca2g1_namelist_values(path)


def _root(tmp_path, rung):
    root = tmp_path / rung.name
    root.mkdir()
    for name in ("namelist_cfg", "namelist_ice_cfg"):
        shutil.copy(rung / name, root / name)
    (root / "domain_cfg.nc").symlink_to(INPUTS / CARD_FILES[0])
    for name in CARD_FILES[1:]:
        (root / name).symlink_to(INPUTS / name)
    return root


def test_g6_deck_is_g5_plus_namdyn_adv_only():
    g5, g6 = _values(G5 / "namelist_cfg"), _values(G6 / "namelist_cfg")
    changed = {key: (g5.get(key), g6.get(key)) for key in set(g5) | set(g6)
               if g5.get(key) != g6.get(key)}
    assert changed == SWITCH
    assert (G6 / "namelist_ice_cfg").read_bytes() == (G5 / "namelist_ice_cfg").read_bytes()
    pin = re.search(r"G6_DECK_SHA=(\w+)", (G6 / "run.sh").read_text()).group(1)
    assert hashlib.sha256((G6 / "namelist_cfg").read_bytes()).hexdigest() == pin


@needs_nemo
def test_g6_switch_lines_are_orca2s():
    orca2 = _values(ORCA2 / "namelist_cfg")
    g6 = _values(G6 / "namelist_cfg")
    for key in ("namdyn_adv.ln_dynadv_vec", "namdyn_adv.nn_dynkeg"):
        assert g6[key] == orca2[key]


def test_g6_reader_refuses_planted_switch(tmp_path):
    root = tmp_path / "deck"
    root.mkdir()
    text = (G6 / "namelist_cfg").read_text()
    (root / "namelist_cfg").write_text(text.replace("nn_dynkeg   = 0", "nn_dynkeg   = 1"))
    with pytest.raises(ValueError, match="nn_dynkeg"):
        recipe.read_orca2g1_deck(root, "ORCA2G6-vec")


@needs_inputs
def test_g6_card_is_g5_plus_g1b_vector_selectors(tmp_path):
    g5_root = _root(tmp_path, G5)
    g5 = recipe.build_orca2g5_ice_card(g5_root)
    g6_root = _root(tmp_path, G6)
    g6 = recipe.build_nemo_testcase_card("ORCA2G6-vec", deck_root=g6_root)
    c5 = g5.recipe.model_config
    assert g6.recipe.model_config == c5._replace(
        **dict(zip(recipe._ORCA2G_VEC_SELECTORS, recipe._ORCA2G_VEC_ARM)),
        rk3_stage1_rhs_nemo_literal=False,
        een_nemo_relative_vorticity=True,
        rk3_vector_rhs_nemo_order=True,
        nemo_hpg_sco_fold_geometry=True,
        rk3_vector_raw_vmask=True,
        nemo_first_wzv_after_ssh="rk3_extrapolated_carried",   # Decision 132
        nemo_wzv_stored_r3_ratios=True,          # round 7: wzv reads stored r3t
        nemo_zad_written_association=True,       # round 7: dyn_zad association
        nemo_stage2_momentum_wzv_half_step=True,  # round 8: stage-2 wzv clock pair
        barotropic=c5.barotropic._replace(
            barotropic_flux_update_nemo_literal=False,
            barotropic_entry_depth_nemo_literal=False,
            barotropic_literal_fold_unit_vector_form=True))
    # G5's own statements stay selected on G6.
    assert c5.implicit_ice_ocean_top_drag and c5.rk3_stage3_salt_nemo_written
    for field in ("T", "S", "u", "v", "uu_b", "vv_b", "eta"):
        assert np.array_equal(np.asarray(getattr(g6.recipe.initial_state, field).data),
                              np.asarray(getattr(g5.recipe.initial_state, field).data))
    assert (g6.surface_boundary_condition, g6.surface_input_operator) == (
        g5.surface_boundary_condition, g5.surface_input_operator)
    assert g6.unmeasured_features == recipe.ORCA2G6_UNMEASURED
    assert recipe._orca2g1_selector_failures(g6) == []
    with pytest.raises(ValueError, match="not execution-ready"):
        recipe.validate_orca2g1_flat_card_for_execution(g6, g6_root)
    # Each rung refuses the other's deck.
    with pytest.raises(ValueError, match="ln_dynadv_vec"):
        recipe.build_orca2g5_ice_card(g6_root)
    with pytest.raises(ValueError, match="ln_dynadv_vec"):
        recipe.build_orca2g6_vec_card(g5_root)


def test_run_script_reuses_g5_binaries_smoke_first_passive_no_step1_restart():
    run = G6 / "run.sh"
    assert subprocess.run(["bash", "-n", str(run)]).returncode == 0
    text = run.read_text()
    assert "makenemo" not in text.split("set -Eeuo pipefail", 1)[1]
    assert "build_targets" not in text
    executions = re.findall(r'^run_one "([^\"]+)" "\$(\w+)"', text, re.M)
    assert executions == [
        ("$TARGET/BLD/bin/nemo.exe", "SMOKE"),
        ("$REF_TARGET/BLD/bin/nemo.exe", "REF"),
        ("$TARGET/BLD/bin/nemo.exe", "TWIN_A"),
        ("$TARGET/BLD/bin/nemo.exe", "TWIN_B"),
        ("$TARGET/BLD/bin/nemo.exe", "TWIN_C"),
    ]
    assert 'for step in $(seq 2 "$steps")' in text
    assert "list_c() { (($1 >= 3)) && pad10 $(seq 3 2" in text
    assert "vector form dynamics : total vorticity = Coriolis + relative vorticity" in text
    assert "g5_records.py\" admit" in text and '--steps "$steps"' in text
    assert "REF_BIN_SHA=e8ebf4ae" in text and "REC_BIN_SHA=41cd9a63" in text
    assert "round36/oracle_g6_r36" in text


def test_g2_family_ladder_routes_g6_through_the_ice_record_path():
    sys.path.insert(0, str(CASES))
    path = CASES / "nemo_testcase_g2_orca2bathy_ladder.py"
    spec = importlib.util.spec_from_file_location("g6_g2_ladder", path)
    ladder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ladder)
    assert "ORCA2G6-vec" in ladder.CASES
    assert "ORCA2G6-vec" in ladder.ICE_CASES
    assert "ORCA2G6-vec" in ladder.FORCED
    assert "operators" in ladder.ARMS


G6_RECORD = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/"
                 "round36/oracle_g6_r36/twin_a")


@pytest.mark.skipif(not (G6_RECORD / "oracle_rkstage2_terms_kt00000001.bin").is_file(),
                    reason="no G6 record")
def test_g6_stage2_term_reader_admits_the_record_schema():
    sys.path.insert(0, str(CASES))
    path = CASES / "nemo_testcase_g2_orca2bathy_ladder.py"
    spec = importlib.util.spec_from_file_location("g6_operator_ladder", path)
    ladder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ladder)
    terms = ladder.read_stage2_terms(
        G6_RECORD / "oracle_rkstage2_terms_kt00000001.bin")
    assert tuple(terms) == (
        "before_u", "before_v", "after_hpg_u", "after_hpg_v",
        "after_vorticity_u", "after_vorticity_v",
        "after_advection_u", "after_advection_v",
    )
    assert all(value.shape == (152, 184, 31) for value in terms.values())
    assert not np.any(terms["before_u"])
    assert not np.any(terms["before_v"])
    assert np.any(terms["after_advection_u"])
    assert np.any(terms["after_advection_v"])


G5_RECORD = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/"
                 "round31/oracle_g5_r31")


@pytest.mark.skipif(not (G5_RECORD / "twin_a").is_dir(), reason="no G5 record")
def test_admission_takes_the_record_length_and_names_empty_comparisons():
    """G6's admission is G5's with the record length.  On G5's 10-step
    record: passivity compares step 10, twin C (odd steps only) is reported
    as having no common step rather than as identical, a length with no
    twin-A restart is refused by name, and the passivity plant fires."""
    import sys
    sys.path.insert(0, str(G5))
    import g5_records

    res = g5_records.admit(G5_RECORD, steps=10)
    assert res["passivity"]["step"] == 10
    assert res["reference_vs_twin_b_c"]["twin_c"] == (
        "no restart step in common with the reference")
    assert len(res["reference_vs_twin_b_c"]["twin_b"]) == 5      # steps 2..10 even
    with pytest.raises(g5_records.Refuse, match="no restart step 2..8 common"):
        g5_records.admit(G5_RECORD, steps=8)     # twin A wrote only step 10
    with pytest.raises(g5_records.Refuse, match="PASSIVITY FAILED"):
        g5_records.admit(G5_RECORD, "passivity", steps=10)


R42 = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/"
           "round42/oracle_g6_r42/twin_s")
needs_r42 = pytest.mark.skipif(
    not (R42 / "oracle_stp2d_terms_kt00000003.bin").is_file(),
    reason="the operator's G6 r42 stp_2D record is not on this machine")


@needs_r42
def test_first_wzv_is_nemos_under_jit_only_with_stored_ratios():
    """Traced jit, kt 3 from NEMO's restart 2: the first wzv's ww on levels
    2..30 equals NEMO's recorded ww (sshwzv.f90:297-298) bit for bit only when
    r3t(Kaa)/r3t(Kbb) are materialised (stp2d.f90:152, domqco.f90:257)."""
    import jax
    import jax.numpy as jnp
    import netCDF4
    from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks_3d
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import nemo_qco_wzv_operands
    sys.path.insert(0, str(CASES))
    import nemo_testcase_g1_orca2grid_ladder as g1
    import nemo_testcase_g2_orca2bathy_ladder as g2
    card = recipe.build_nemo_testcase_card("ORCA2G6-vec", deck_root=R42)
    cfg, z, grid = card.recipe.model_config, card.recipe.z_coord, card.recipe.grid
    nk = int(z.n_levels)
    nemo = g2.read_stp2d_terms(R42 / "oracle_stp2d_terms_kt00000003.bin", 3, nk)["ww"]
    with netCDF4.Dataset(R42 / "ORCA2G1_00000002_restart.nc") as h:
        lev = lambda n: g2.levels_last(h.variables[n][0], nk)  # noqa: E731
        u, v = jnp.asarray(g1.pad_u(lev("un"))), jnp.asarray(g1.pad_v(lev("vn")))
        ssh = {n: jnp.asarray(np.asarray(h.variables[n][0], np.float64))
               for n in ("sshn", "ssha")}
    um3, vm3 = compute_face_masks_3d(z.is_active, grid)
    tmask = jnp.asarray(z.is_active, jnp.float64)
    wet = np.asarray(tmask) > 0
    wet[0] = wet[-1] = False                 # rows reading the halo
    wet[..., 0] = False                      # ww(1) carries NEMO's runoff term

    def unequal(stored):
        f = jax.jit(lambda e, ea, uu, vv: nemo_qco_wzv_operands(
            e, e, uu, vv, grid, z, um3, vm3, tmask, card.dt_s,
            after_ssh_form="rk3_extrapolated_carried", eta_after_carried=ea,
            fold_exchange_ratios=bool(cfg.rk3_vector_raw_vmask),
            stored_r3_ratios=stored)[0])
        ww = np.asarray(f(ssh["sshn"], ssh["ssha"], u, v))[..., :nk]
        return int(np.count_nonzero((ww != nemo) & wet))

    assert cfg.nemo_wzv_stored_r3_ratios is True
    assert unequal(True) == 0
    assert unequal(False) > 1000             # round 7 measured 90,611


@needs_r42
def test_slow_forcing_reader_takes_the_records_time_levels():
    sys.path.insert(0, str(CASES))
    from nemo_testcase_l2_gyre_round16_slow_forcing import read_slow_forcing
    path = R42 / "oracle_slow_forcing_kt00000002.bin"
    rec = read_slow_forcing(path, dims=(184, 152, 31), levels=(2, 3, 1))
    assert (rec["header"]["kt"], rec["header"]["Kbb"], rec["header"]["Krhs"]) == (2, 3, 1)
    with pytest.raises(Exception, match="bad header"):
        read_slow_forcing(path, dims=(184, 152, 31))      # default (1, 1, 3)


def test_wzv_stretch_product_is_rounded_under_traced_jit():
    """sshwzv.f90:297-298 rounds r1_Dt*e3t_3d*(r3t(Kaa)-r3t(Kbb)) before the
    add; with every input traced XLA contracts it unless it is materialised."""
    import jax
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import nemo_qco_wzv_recurrence
    rng = np.random.default_rng(3)
    shape = (148, 180, 30)
    div = rng.normal(size=shape) * 1e-4
    e3 = rng.uniform(10.0, 500.0, size=shape)
    rb = rng.normal(size=shape[:2]) * 1e-3
    ra = rb + rng.normal(size=shape[:2]) * 1e-5
    tm = (rng.uniform(size=shape) > 0.1).astype(np.float64)
    dt = 10800.0
    ref = np.zeros(shape[:2] + (shape[2] + 1,))
    for jk in range(shape[2] - 1, -1, -1):
        ref[..., jk] = ref[..., jk + 1] - (
            div[..., jk] + (1.0 / dt) * e3[..., jk] * (ra - rb)) * tm[..., jk]

    def unequal(stored):
        f = jax.jit(lambda *a: nemo_qco_wzv_recurrence(*a, stored_r3_ratios=stored))
        got = np.asarray(f(*(jnp.asarray(x) for x in (div, e3, rb, ra, tm, dt))))
        return int(np.count_nonzero(got != ref))

    assert unequal(True) == 0
    assert unequal(False) > 0                # round 7 resume measured 1,425


R36 = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/"
           "round36/oracle_g6_r36")
needs_r36 = pytest.mark.skipif(
    not (R36 / "ref_100step/ORCA2G1_00000002_restart.nc").is_file(),
    reason="the operator's G6 r36 record is not on this machine")


@pytest.fixture(scope="module")
def kt3_rows():
    """kt 3 from NEMO's restart 2 through the ordinary jitted step (traced
    state): the stp_2D walk and the stage-2/3 rows on the r42 record, the
    step-end rows on r36.  Every assertion below is BIT-EXACT against NEMO's
    record (0 differing wet cells), never a bound on a measured count."""
    sys.path.insert(0, str(CASES))
    spec = importlib.util.spec_from_file_location(
        "g2_ladder_g6", CASES / "nemo_testcase_g2_orca2bathy_ladder.py")
    ladder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ladder)
    walk = ladder.run(R42, arm="stp2d_terms", kt_from=3, kt_to=3,
                      allow_dirty=True, case="ORCA2G6-vec")
    stages = ladder.run(R42, arm="stages", kt_from=3, exposures=("stage2", "stage3"),
                        allow_dirty=True, case="ORCA2G6-vec")
    given = ladder.run(R36 / "twin_a", arm="given_entry", kt_from=3, kt_to=3,
                       entries=(3,), restart_roots=(R36 / "ref_100step",),
                       allow_dirty=True, case="ORCA2G6-vec")
    rows = {r["name"]: r["n_unequal"] for r in walk["per_kt"][0]["stp2d_rows"]}
    for p in stages["per_kt"]:
        rows.update({r["name"]: r["n_unequal"] for r in p["rows"]})
    rows.update({r["name"]: r["n_unequal"] for r in given["per_kt"][0]["rows"]})
    return rows


# The comment on each line is that row with the ONE piece removed (round-7
# resume, round 8 for the stage-2 clock pair).
@needs_r42
@needs_r36
def test_kt3_step_entry_wzv_reads_stored_ratios(kt3_rows):
    assert kt3_rows["spg_ts_exit.uu_b"] == 0             # without: 1,337
    assert kt3_rows["spg_ts_exit.vv_b"] == 0             # without: 1,334


@needs_r42
@needs_r36
def test_kt3_stage2_momentum_wzv_is_nemos(kt3_rows):
    # the stage-2 momentum wzv (stprk3_stg.f90:327) on rDt = rn_Dt/2 with the
    # stored r3t(Kaa) = r1_2*(r3t(Kbb) + r3ta) (:188-189, :221); the half-step
    # pair raises without the call's stored ratios, so it also guards those
    assert kt3_rows["stage2.kt3.v"] == 0                 # without the pair: 54
    assert kt3_rows["stage3.kt3.u"] == 0                 # without the pair: 38


@needs_r42
@needs_r36
def test_kt3_zad_rounds_e1e2t_ww(kt3_rows):
    assert kt3_rows["given_entry.kt3.ssh"] == 0          # without: 568


@needs_r42
@needs_r36
def test_kt3_zad_written_association_on_u(kt3_rows):
    assert kt3_rows["after_zad.u"] == 0                  # without: 2,399
    assert kt3_rows["given_entry.kt3.uu_b"] == 0         # without: 216
