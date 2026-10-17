"""Grid-ladder G8: certified G5 plus ORCA2's lateral momentum diffusion."""

from __future__ import annotations

import hashlib
import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe

REPO = Path(__file__).resolve().parents[3]
CASES = REPO / "scripts/validate/ocean_fidelity/testcases"
G5 = CASES / "nemo_testcase_g5_orca2ice"
G7 = CASES / "nemo_testcase_g7_orca2drg"
G8 = CASES / "nemo_testcase_g8_orca2ldf"
INPUTS = Path("/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0")
ORCA2 = Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/ORCA2_OMIP_L4/EXP00")
CARD_FILES = ("ORCA_R2_zps_domcfg.nc", "data_1m_potential_temperature_nomask.nc",
              "data_1m_salinity_nomask.nc", "eddy_viscosity_3D.nc")
needs_inputs = pytest.mark.skipif(
    not all((INPUTS / name).is_file() for name in CARD_FILES), reason="no ORCA2 inputs")
needs_nemo = pytest.mark.skipif(not ORCA2.is_dir(), reason="no NEMO tree")
SWITCH = {"namdyn_ldf.ln_dynldf_off": (True, False),
          "namdyn_ldf.ln_dynldf_lap": (False, True),
          "namdyn_ldf.ln_dynldf_lev": (None, True),
          "namdyn_ldf.nn_ahm_ijk_t": (None, -30)}
G8_LDF_STATEMENTS = ("nemo_ldf_stored_f_area", "nemo_ldf_fold_rows",
                     "rk3_stage1_ldf_source_order", "nemo_ldf_stage_r3f")


@pytest.fixture(autouse=True, scope="module")
def _fp64():
    before = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    yield
    set_policy(before)


def _same(a, b):
    la, ta = jax.tree_util.tree_flatten(a)
    lb, tb = jax.tree_util.tree_flatten(b)
    return ta == tb and all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(la, lb))


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


def test_g8_deck_is_g5_plus_namdyn_ldf_only():
    g5, g8 = _values(G5 / "namelist_cfg"), _values(G8 / "namelist_cfg")
    changed = {key: (g5.get(key), g8.get(key)) for key in set(g5) | set(g8)
               if g5.get(key) != g8.get(key)}
    assert changed == SWITCH
    assert (G8 / "namelist_ice_cfg").read_bytes() == (G5 / "namelist_ice_cfg").read_bytes()
    pin = re.search(r"G8_DECK_SHA=(\w+)", (G8 / "run.sh").read_text()).group(1)
    assert hashlib.sha256((G8 / "namelist_cfg").read_bytes()).hexdigest() == pin


@needs_nemo
def test_g8_ldf_lines_resolve_to_orca2s():
    orca2, g8 = _values(ORCA2 / "namelist_cfg"), _values(G8 / "namelist_cfg")
    ref = _values(ORCA2 / "namelist_ref")
    for key in [k for k in ref if k.startswith("namdyn_ldf.")]:
        assert g8.get(key, ref[key]) == orca2.get(key, ref[key]), key
    assert not [k for k in orca2 if k.startswith(recipe._ORCA2G_LDF_UNSTATED)]
    assert ref["namdyn_ldf.nn_dynldf_typ"] == 0      # div-rot: the shared operator's form


@pytest.mark.parametrize("old,new,match", [
    ("   nn_ahm_ijk_t  = -30 ", "   nn_ahm_ijk_t  = 20  ", "nn_ahm_ijk_t"),
    ("   ln_dynldf_lev = .true. ", "   ln_dynldf_hor = .true. ", "ln_dynldf_hor"),
    ("   ln_dynldf_lev = .true. ", "   ln_dynldf_lev = .false.", "ln_dynldf_lev"),
    ("   ln_dynldf_blp = .false.", "   ln_dynldf_blp = .false.\n   nn_dynldf_typ = 1",
     "nn_dynldf_typ"),
    ("   ln_dynldf_lap = .true. ", "   ln_dynldf_lap = .false.", "without"),
])
def test_g8_reader_refuses_planted_ldf_line(tmp_path, old, new, match):
    root = tmp_path / "deck"
    root.mkdir()
    text = (G8 / "namelist_cfg").read_text()
    assert text.count(old) == 1
    (root / "namelist_cfg").write_text(text.replace(old, new))
    with pytest.raises(ValueError, match=match):
        recipe.read_orca2g1_deck(root, "ORCA2G8-ldf")


@needs_inputs
def test_g8_card_is_g5_plus_the_read_coefficient_laplacian(tmp_path):
    g5_root, g8_root = _root(tmp_path, G5), _root(tmp_path, G8)
    g5 = recipe.build_orca2g5_ice_card(g5_root)
    g8 = recipe.build_nemo_testcase_card("ORCA2G8-ldf", deck_root=g8_root)
    c5, c8 = g5.recipe.model_config, g8.recipe.model_config
    # A_h (OMT-3's edge) plus the four dyn_ldf statements of round 1
    assert [f for f in c5._fields if getattr(c5, f) != getattr(c8, f)] == [
        "lateral_viscosity", *G8_LDF_STATEMENTS]
    assert all(getattr(c8, f) is True and getattr(c5, f) is False for f in G8_LDF_STATEMENTS)
    assert c8.lateral_viscosity == c5.lateral_viscosity._replace(A_h=1.0e5)
    assert c5.lateral_viscosity.A_h == 0.0
    assert (c8.lateral_viscosity_operator, c8.lateral_viscosity_e3_weighting,
            c8.lateral_viscosity_coefficient_source, c8.lateral_side_bc) == (
        "nemo_div_curl", "nemo_e3", "nemo_ahm_3d_file", "free_slip")
    # the vertical coordinate gains only NEMO's read coefficient
    z5, z8 = g5.recipe.z_coord, g8.recipe.z_coord
    moved = [f for f in z5._fields if not _same(getattr(z5, f), getattr(z8, f))]
    assert moved == ["nemo_ldf_ahmt", "nemo_ldf_ahmf"]
    assert z5.nemo_ldf_ahmt is None and z5.nemo_ldf_ahmf is None
    tmask = np.asarray(z8.is_active, dtype=np.float64)
    ahmt, ahmf = np.asarray(z8.nemo_ldf_ahmt), np.asarray(z8.nemo_ldf_ahmf)
    assert ahmt.shape == tmask.shape and ahmf.shape == (tmask.shape[0] + 1, tmask.shape[1] + 1, 30)
    assert np.array_equal(ahmt == 0.0, tmask == 0.0)
    # ldfdyn.f90:387-390: ahmf carries fmask's VALUE, so rn_shlat = 2 doubles
    # the file's coefficient on coastal F points -- the boundary condition.
    assert ahmf.max() == 2.0 * 40000.0
    for field in ("T", "S", "u", "v", "uu_b", "vv_b", "eta"):
        assert np.array_equal(np.asarray(getattr(g8.recipe.initial_state, field).data),
                              np.asarray(getattr(g5.recipe.initial_state, field).data))
    assert g8.unmeasured_features == recipe.ORCA2G8_UNMEASURED
    assert recipe._orca2g1_selector_failures(g8) == []
    with pytest.raises(ValueError, match="not execution-ready"):
        recipe.validate_orca2g1_flat_card_for_execution(g8, g8_root)
    with pytest.raises(ValueError, match="ln_dynldf_lap"):
        recipe.build_orca2g5_ice_card(g8_root)
    with pytest.raises(ValueError, match="ln_dynldf_lap"):
        recipe.build_orca2g8_ldf_card(g5_root)
    # the selector check is not vacuous: G8's card with G5's viscosity fails it
    planted = g8._replace(recipe=g8.recipe._replace(model_config=c8._replace(
        lateral_viscosity=c5.lateral_viscosity)))
    assert recipe._orca2g1_selector_failures(planted) == ["ln_dynldf_lap"]
    for field, row in zip(G8_LDF_STATEMENTS, ("ldf stored F area", "ldf fold rows",
                                              "stage-1 ldf source order",
                                              "stage-3 ldf r3f")):
        planted = g8._replace(recipe=g8.recipe._replace(model_config=c8._replace(**{field: False})))
        assert recipe._orca2g1_selector_failures(planted) == [row]


@needs_inputs
def test_g8_requires_the_viscosity_file(tmp_path):
    root = _root(tmp_path, G8)
    (root / "eddy_viscosity_3D.nc").unlink()
    with pytest.raises(FileNotFoundError, match="eddy viscosity"):
        recipe.build_orca2g8_ldf_card(root)


def test_only_g8_selects_lateral_momentum_diffusion():
    assert [c for c, on in recipe._ORCA2G_LDF.items() if on] == ["ORCA2G8-ldf"]
    assert set(recipe._ORCA2G_LDF) == set(recipe._ORCA2G_JPK)
    # G8 inherits G5's rung tables and leaves drag and advection off
    for table in (recipe._ORCA2G_ZDF_FACES, recipe._ORCA2G_TRA_ZDF_ROUNDED,
                  recipe._ORCA2G_UE_RHS_ASCENDING, recipe._ORCA2G_ZDF_STAGE_R3U,
                  recipe._ORCA2G_WIND_STORED_DEPTH, recipe._ORCA2G_TRA_STAGE_ORDER,
                  recipe._ORCA2G_TSD_INIT, recipe._ORCA2G_TSD_DMP, recipe._ORCA2G_SBC,
                  recipe._ORCA2G_DYNADV_VEC, recipe._ORCA2G_DRG):
        assert table["ORCA2G8-ldf"] == table["ORCA2G5-ice"]
    assert "ORCA2G8-ldf" in recipe._ORCA2G_ICE and recipe._ORCA2G_JPK["ORCA2G8-ldf"] == 31
    # every other rung's deck states the OFF switch set the reader requires
    for deck in sorted(CASES.glob("nemo_testcase_g*_orca2*/namelist_cfg")):
        values = _values(deck)
        want = recipe._orca2g_ldf_switch_set(values)
        assert all(values[k] == v for k, v in want.items()), deck
        assert values["namdyn_ldf.ln_dynldf_lap"] is (deck.parent == G8)


def test_run_script_reuses_g5_binaries_copies_the_pinned_file_and_checks_the_log():
    run = G8 / "run.sh"
    assert subprocess.run(["bash", "-n", str(run)]).returncode == 0
    text = run.read_text()
    assert "makenemo" not in text.split("set -Eeuo pipefail", 1)[1]
    executions = re.findall(r'^run_one "([^\"]+)" "\$(\w+)"', text, re.M)
    assert executions == [
        ("$TARGET/BLD/bin/nemo.exe", "SMOKE"),
        ("$REF_TARGET/BLD/bin/nemo.exe", "REF"),
        ("$TARGET/BLD/bin/nemo.exe", "TWIN_A"),
        ("$TARGET/BLD/bin/nemo.exe", "TWIN_B"),
        ("$TARGET/BLD/bin/nemo.exe", "TWIN_C"),
    ]
    assert 'cp "$VISC_SRC" "$dir/eddy_viscosity_3D.nc"' in text
    assert "VISC_SHA=fc142ce094a0f68d255ff79b04f8fe5d6634b33b0bc1c07b61870cede39499cb" in text
    assert "read in eddy_viscosity_3D.nc file" in text and "nn_ahm_ijk_t += +-30$" in text
    assert "ln_drg_OFF += +T$" in text
    assert "g5_records.py\" admit" in text
    assert "REF_BIN_SHA=e8ebf4ae" in text and "REC_BIN_SHA=41cd9a63" in text
    assert "round45/oracle_g8_r45" in text and "G7" not in text
    # the rest of the script is G7's, line for line
    g7 = (G7 / "run.sh").read_text().splitlines()
    extra = [line for line in text.splitlines() if line not in g7]
    assert len(extra) < 40, extra


def test_g2_family_ladder_routes_g8_through_the_ice_record_path():
    sys.path.insert(0, str(CASES))
    spec = importlib.util.spec_from_file_location(
        "g8_g2_ladder", CASES / "nemo_testcase_g2_orca2bathy_ladder.py")
    ladder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ladder)
    assert "ORCA2G8-ldf" in ladder.CASES
    assert "ORCA2G8-ldf" in ladder.ICE_CASES and "ORCA2G8-ldf" in ladder.FORCED


def _ldf_literal(raw, e3t, tm, ahmt_n, ahmf_n, u, v):
    """dynldf_lev.f90:121-140 at Kbb = Kmm with r3 = 0, on arrays extended by
    one T-pivot ghost row (lbcnfd T-fold images; U/V velocity sign -1)."""
    n = u.shape[1]
    pt, pf = (n - np.arange(n)) % n, (n - np.arange(n) - 1) % n
    gT = lambda a: np.concatenate([a, a[-2:-1][:, pt]], 0)          # noqa: E731
    gU = lambda a, s=1: np.concatenate([a, s * a[-2:-1][:, pf]], 0)  # noqa: E731
    gV = lambda a, s=1: np.concatenate([a, s * a[-3:-2][:, pt]], 0)  # noqa: E731
    gF = lambda a: np.concatenate([a, a[-3:-2][:, pf]], 0)          # noqa: E731
    X = lambda a: np.asarray(a, np.float64)[..., None]               # noqa: E731
    um, vm = gU(np.asarray(raw.umask)), gV(np.asarray(raw.vmask))
    e3u, e3v, e3f = gU(np.asarray(raw.e3u_0)), gV(np.asarray(raw.e3v_0)), gF(np.asarray(raw.e3f_0))
    e1u, e2u = gU(X(raw.e1u)), gU(X(raw.e2u))
    e1v, e2v = gV(X(raw.e1v)), gV(X(raw.e2v))
    e1f, e2f, e1t, e2t = gF(X(raw.e1f)), gF(X(raw.e2f)), gT(X(raw.e1t)), gT(X(raw.e2t))
    e3t, ahmt, ahmf = gT(e3t), gT(ahmt_n), gF(ahmf_n)
    u, v = gU(u, -1), gV(v, -1)
    E = lambda a: np.roll(a, -1, 1)   # noqa: E731
    W = lambda a: np.roll(a, 1, 1)    # noqa: E731
    N = lambda a: np.roll(a, -1, 0)   # noqa: E731
    S = lambda a: np.roll(a, 1, 0)    # noqa: E731
    safe = lambda h: np.where(h > 0, h, 1.0)  # noqa: E731
    zwf = ahmf * e3f / (e1f * e2f) * ((E(e2v * v) - e2v * v) - (N(e1u * u) - e1u * u))
    fu, fv = e2u * e3u * u, e1v * e3v * v
    zwt = ahmt / (e1t * e2t) / safe(e3t) * ((fu - W(fu)) + (fv - S(fv)))
    lu = um * (-(zwf - S(zwf)) / e2u / safe(e3u) + (E(zwt) - zwt) / e1u)
    lv = vm * ((zwf - W(zwf)) / e1v / safe(e3v) + (N(zwt) - zwt) / e2v)
    return lu[:-1], lv[:-1]


@needs_inputs
def test_g8_dyn_ldf_lev_matches_nemo_statement_with_stored_f_area_and_fold_rows(tmp_path):
    """The card's operator against dynldf_lev.f90:121-140 on the T-pivot fold.

    Fails with the grid's own F area (its area_q holds the next F point's
    area on the vertex layout, ~13% here) and without the fold rows (the
    zeroed V fold row and missing F fold row, ~1-4% on rows 146/147)."""
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        nemo_ldf_lap_viscosity_e3_cgrid as op)
    from legoesm.ocean.vertical import nemo_qco_r1_area_f
    card = recipe.build_nemo_testcase_card("ORCA2G8-ldf", deck_root=_root(tmp_path, G8))
    grid, z = card.recipe.grid, card.recipe.z_coord
    raw = z.nemo_een_barotropic
    tm = np.asarray(z.is_active, np.float64)
    e3t = np.asarray(z.nemo_e3t_0, np.float64) * tm
    um, vm = np.asarray(raw.umask), np.asarray(raw.vmask)
    rng = np.random.default_rng(8)
    n = tm.shape[1]
    u = rng.normal(0.0, 0.1, um.shape) * um
    v = rng.normal(0.0, 0.1, vm.shape) * vm
    v[-1] = -v[-2][(n - np.arange(n)) % n]          # the lbc_lnk 'V', -1 fold row
    pf = (n - np.arange(n) - 1) % n                 # and the self-imaged U pivot row
    u[-1] = 0.5 * (u[-1] - u[-1][pf]) * um[-1] * um[-1][pf]
    pu = lambda a: np.concatenate([a[:, -1:], a], 1)                       # noqa: E731
    pv = lambda a: np.concatenate([np.zeros_like(a[:1]), a], 0)            # noqa: E731
    def pfv(a):
        s = np.concatenate([a[:1], a], 0)
        return np.concatenate([s[:, -1:], s], 1)
    th = tuple(jnp.asarray(x) for x in (e3t, pu(np.asarray(raw.e3u_0)), pv(np.asarray(raw.e3v_0)),
                                        pfv(np.asarray(raw.e3f_0)), pu(np.asarray(raw.e3u_0)),
                                        pv(np.asarray(raw.e3v_0))))
    r1f = pfv(np.asarray(nemo_qco_r1_area_f(raw, grid, jnp.float64)))
    vmask_compact = np.asarray(raw.vmask).copy()
    vmask_compact[-1] = 0.0                          # legoESM's compact V mask: fold row dry

    def run(stored, fold):
        f = jax.jit(lambda uu, vv: op(
            uu, vv, grid, z.nemo_ldf_ahmt, z.nemo_ldf_ahmf, th[0], mask=jnp.asarray(tm[..., 0]),
            u_mask=jnp.asarray(pu(um)), v_mask=jnp.asarray(pv(vmask_compact)),
            thickness_operands=th, coefficient_fmask_already_applied=True,
            stored_r1_e1e2f=jnp.asarray(r1f) if stored else None, nemo_fold_rows=fold))
        a, b = f(jnp.asarray(pu(u)), jnp.asarray(pv(v)))
        return np.asarray(a)[:, 1:], np.asarray(b)[1:]

    ahmf_n = np.asarray(z.nemo_ldf_ahmf)[1:, 1:]
    lu, lv = _ldf_literal(raw, e3t, tm, np.asarray(z.nemo_ldf_ahmt), ahmf_n, u, v)
    def err(stored, fold):
        cu, cv = run(stored, fold)
        # row 0 is Antarctica; V row 147 is replaced by the stage's fold exchange
        return (np.abs(cu - lu)[1:].max() / np.abs(lu).max(),
                np.abs(cv - lv)[1:-1].max() / np.abs(lv).max())
    eu, ev = err(True, True)
    assert eu < 1e-14 and ev < 1e-14, (eu, ev)
    assert min(err(False, True)) > 1e-3           # grid area_q: the next F point's area
    eu, ev = err(True, False)
    assert eu > 1e-3 and ev > 1e-3                # fold rows: u row 147, v row 146


R45 = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round45/oracle_g8_r45")


@pytest.mark.skipif(not (R45 / "twin_a/oracle_step_entry_kt00000003.bin").is_file(),
                    reason="no round-45 G8 record")
def test_g8_kt3_step_from_nemo_entry_is_bit_identical_on_every_level():
    """kt 3 and kt 4 from NEMO's restart of kt 2, through the card's jitted
    step (state traced, x64), against NEMO's after-step restarts.

    Each of the four G8 wirings moves bits here and fails this test alone:
    stage-1 hpg->ldf->vor order (stp2d.f90:141-147), the stage-3 r3f read
    (stprk3_stg.f90:245, the kt-3 V cell (125, 98, 9) one ulp off without
    it), the stored F area (domhgr.f90:156) and the fold rows
    (dynldf_lev.f90:121-140)."""
    sys.path.insert(0, str(CASES))
    import nemo_testcase_g2_orca2bathy_ladder as ladder
    out = ladder.run(R45 / "twin_a", arm="given_entry", entries=(3,), case="ORCA2G8-ldf",
                     restart_roots=(R45 / "twin_b", R45 / "twin_c"), allow_dirty=True)
    (kt,) = out["per_kt"]
    assert kt["two_step_kt"] == 4
    for rows, levels in ((kt["rows"], kt["all_levels"]),
                         (kt["two_step_rows"], kt["two_step_all_levels"])):
        assert len(rows) == 7 and len(levels) == 4
        bad = [r["name"] for r in rows if not r["bit_identical"]]
        for f, field in levels.items():
            for lev in field["levels"]:
                if lev["dry_n_unequal"] or lev["full"]["n_unequal"] or lev["partial"]["n_unequal"]:
                    bad.append(f"{f} level {lev['level']}")
        assert not bad, bad
