"""The TSUNAMI card (round 2, RK3 build): transcription, explicit switches,
the key_RK3 deviation, j-periodicity as card data, the one-level rest step.

The oracle comparison waits for the acquisition record
(scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_tsunami/run_rk3.sh).
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.constants_config import NEMO_CONSTANTS_CONFIG as CONSTANTS
from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe
from legoesm.ocean.fidelity.nemo_testcase_recipe import (
    TSUNAMI_DEVIATIONS,
    TSUNAMI_NAMELIST,
    TsunamiResolvedNamelist,
    TSUNAMI_UNMEASURED,
    build_tsunami_zco_card,
    tsunami_horizontal_coordinates,
    validate_nemo_testcase_card,
    validate_nemo_testcase_card_for_execution,
)
from legoesm.ocean.vertical import create_z_star_from_thicknesses


@pytest.fixture(scope="module", autouse=True)
def _fp64():
    previous = get_policy()
    set_policy(PrecisionPolicy.fp64())
    yield
    set_policy(previous)


@pytest.fixture(scope="module")
def card():
    return build_tsunami_zco_card()


def test_construction_and_shapes(card):
    s = card.recipe.initial_state
    assert card.case == "TSUNAMI-zco"
    assert (card.dt_s, card.n_steps, card.dummy_bottom_records) == (1000.0, 100, 1)
    # usrdef_nam.F90:113-117: NINT(2000/10)+1 = 201 each way, kpk = 2.
    assert s.eta.data.shape == (201, 201)
    assert s.T.data.shape == (201, 201, 1)
    assert s.u.data.shape == (201, 202, 1) and s.v.data.shape == (202, 201, 1)
    assert s.uu_b.data.shape == (201, 202) and s.vv_b.data.shape == (202, 201)
    assert np.all(np.asarray(s.T.data) == 20.0)
    assert np.all(np.asarray(s.S.data) == 30.0)
    for f in (s.u, s.v, s.uu_b, s.vv_b):
        assert not np.any(np.asarray(f.data))
    assert np.all(np.asarray(card.recipe.land_mask) == 1.0)


_NEMO = Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2")
_CASE = _NEMO / "tests/TSUNAMI"
_STATS = (Path(__file__).resolve().parents[3] / "scripts/validate/ocean_fidelity/"
          "testcases/nemo_testcase_full_statistics.py")
_UNIT_SUFFIXES = ("_m2_s", "_deg", "_km", "_m", "_s")
_ALIASES = {"ln_usr_sbc": "namsbc.ln_usr"}   # card label -> NEMO key
_NOT_IN_NAMELIST = {"key_RK3", "key_qco", "key_vco_1d", "nn_e_resolved",
                    "n_baro_upd"}


def _nemo_value(text):
    t = text.strip().lower()
    if t in (".true.", ".false."):
        return t == ".true."
    return float(t.replace("d", "e"))


def _resolved_nemo_namelist():
    spec = importlib.util.spec_from_file_location("tsunami_stats", _STATS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    resolved = mod.parse_namelist_values(_CASE / "EXPREF/namelist_ref")
    resolved.update(mod.parse_namelist_values(_CASE / "EXPREF/namelist_cfg"))
    return resolved


def _lookup(resolved, field):
    if field in _ALIASES:
        return _nemo_value(resolved[_ALIASES[field]])
    names = [field.lower()] + [field[: -len(x)].lower()
                               for x in _UNIT_SUFFIXES if field.endswith(x)]
    for name in names:
        vals = {v for k, v in resolved.items() if k.split(".", 1)[1] == name}
        if vals:
            assert len(vals) == 1, (field, vals)
            return _nemo_value(vals.pop())
    raise AssertionError(f"{field} not in NEMO's TSUNAMI namelists")


@pytest.mark.skipif(not _CASE.is_dir(), reason="NEMO 5.0.2 tree not present")
def test_explicit_switch_values_match_nemo_namelists():
    resolved = _resolved_nemo_namelist()
    for field in TsunamiResolvedNamelist._fields:
        if field in _NOT_IN_NAMELIST:
            continue
        assert getattr(TSUNAMI_NAMELIST, field) == _lookup(resolved, field), field
    keys = (_CASE / "cpp_TSUNAMI.fcm").read_text().split()
    # every built key equals the shipped one EXCEPT the recorded deviations
    differ = {k for k in ("key_RK3", "key_qco", "key_vco_1d")
              if getattr(TSUNAMI_NAMELIST, k) != (k in keys)}
    assert differ == {d[0] for d in TSUNAMI_DEVIATIONS}
    for field, shipped, built, _ in TSUNAMI_DEVIATIONS:
        assert (field in keys, getattr(TSUNAMI_NAMELIST, field)) == (shipped, built)


@pytest.mark.skipif(not _CASE.is_dir(), reason="NEMO 5.0.2 tree not present")
def test_rk3_stage_selector_is_the_module_constant():
    src = (_NEMO / "src/OCE/stprk3_stg.F90").read_text().splitlines()
    assert "n_baro_upd =  np_HYB" in src[43]
    assert TSUNAMI_NAMELIST.n_baro_upd == "np_HYB"


def test_switch_lookup_refuses_a_planted_drift():
    resolved = {"namdyn_spg.nn_e": "6", "namdom.rn_dt": "1000."}
    assert _lookup(resolved, "rn_Dt_s") == 1000.0
    with pytest.raises(AssertionError):
        _lookup(resolved, "rn_atfp")


@pytest.mark.parametrize("switch", ["ln_tide", "ln_traqsr", "ln_zdfnpc"])
def test_validator_refuses_an_spg_term_switched_on(card, monkeypatch, switch):
    monkeypatch.setattr(recipe, "TSUNAMI_NAMELIST",
                        TSUNAMI_NAMELIST._replace(**{switch: True}))
    with pytest.raises(ValueError, match=switch):
        validate_nemo_testcase_card(card)


def test_explicit_switch_values(card):
    cfg = card.recipe.model_config
    assert cfg.barotropic.barotropic_time_filter == "nemo_boxcar1_ab3"
    assert cfg.barotropic.n_barotropic_substeps == 6
    assert cfg.barotropic.barotropic_coriolis == "een_metric"
    assert (cfg.A_v, cfg.K_v, cfg.K_h) == (1.2e-4, 1.2e-5, 0.0)
    assert cfg.lateral_viscosity.A_h == 0.0
    assert cfg.eos == "nemo_seos" and cfg.eos_nemo_seos.a0 == 1.6550e-1
    # dynspg_ts.F90:1223-1240 by hand: zcmax = sqrt(grav*H*(2/dx^2)).
    zcmax = math.sqrt(CONSTANTS.g * 100.0 * 2.0 / 1.0e8)
    assert math.ceil(1000.0 / 0.8 * zcmax) == 6


def test_f_plane(card):
    f0 = 2.0 * CONSTANTS.Omega * math.sin(math.pi / 180.0 * 38.5)
    src = tsunami_horizontal_coordinates()
    assert src["f0"] == f0
    assert np.all(np.asarray(card.recipe.grid.f_T) == src["f0"])
    assert (src["ii0"], src["ij0"]) == (40, 80)


def test_initial_ssh_against_formula_at_three_cells(card):
    eta = np.asarray(card.recipe.initial_state.eta.data)
    # Interior extremes of zti, ztj are 161 and 121 cells (i = j = 201).
    zmax = math.sqrt((10.0 * 161) ** 2 + (10.0 * 121) ** 2) / 20.0
    cells = {(79, 39): 0.0, (79, 45): 60.0, (85, 43): math.hypot(40.0, 60.0)}
    for (j, i), dist in cells.items():
        want = 0.1 * math.cos(dist / zmax * math.pi * 0.5)
        assert eta[j, i] == pytest.approx(want, rel=1e-14, abs=0.0)
    assert eta[79, 39] == 0.1
    assert eta[0, 0] == 0.0
    assert np.count_nonzero(eta) == 325


def test_execution_gate_admits_the_measured_card(card):
    assert TSUNAMI_UNMEASURED == ()
    validate_nemo_testcase_card_for_execution(card)
    with pytest.raises(ValueError, match="TSUNAMI_UNMEASURED"):
        validate_nemo_testcase_card_for_execution(
            card._replace(unmeasured_features=("B4j:planted",)))


def test_j_periodicity_is_card_data(card):
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        NEMOTestcaseCard, build_nemo_testcase_card)
    assert card.j_periodic is True and TSUNAMI_NAMELIST.ln_Jperio is True
    assert NEMOTestcaseCard._field_defaults["j_periodic"] is False
    assert build_nemo_testcase_card("VORTEX-zco").j_periodic is False
    with pytest.raises(ValueError, match="j_periodic"):
        validate_nemo_testcase_card(card._replace(j_periodic=False))


def test_meridional_periodicity_scope_feeds_the_existing_y_wrap(card):
    from legoesm.grids import halo_latlon as hl
    import jax.numpy as jnp
    v = jnp.ones((5, 4))
    assert hl.get_meridionally_periodic() is False
    assert np.asarray(hl.zero_polar_lat_ends(v))[[0, -1]].sum() == 0.0
    with hl.meridional_periodicity(card.j_periodic):
        assert hl.get_meridionally_periodic() is True
        assert np.all(np.asarray(hl.zero_polar_lat_ends(v)) == 1.0)
    assert hl.get_meridionally_periodic() is False
    with pytest.raises(RuntimeError):
        with hl.meridional_periodicity(True):
            raise RuntimeError("restore on error")
    assert hl.get_meridionally_periodic() is False


def _step_fields(card, eta, *, caller_scope=None, model=None):
    """One model step of the card from (eta, the card's rest u/v/T/S).

    No y-wrap scope by default: the card's model config carries it.
    ``caller_scope`` sets a contrary caller scope the model must ignore.
    """
    import contextlib
    import jax.numpy as jnp
    from legoesm.grids.halo_latlon import meridional_periodicity
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel)
    r = card.recipe
    s0 = r.initial_state._replace(
        eta=r.initial_state.eta.replace(data=jnp.asarray(eta)))
    m = model or LatLonCGridOceanModel(r.grid, r.z_coord, r.model_config)
    scope = (contextlib.nullcontext() if caller_scope is None
             else meridional_periodicity(caller_scope))
    with scope:
        s1 = m.step(s0, card.dt_s)
    return {f: (np.asarray(getattr(s0, f).data), np.asarray(getattr(s1, f).data))
            for f in ("u", "v", "T", "S", "eta", "w", "uu_b", "vv_b")}


def _walled(card):
    """Plant: the same card with the step's y-wrap switched off."""
    cfg = card.recipe.model_config._replace(meridionally_periodic=False)
    return card._replace(recipe=card.recipe._replace(model_config=cfg))


def test_one_wet_level_rest_state_does_not_move_bit_for_bit(card):
    """B5: one wet level, zero forcing, flat ssh -> nothing moves, bitwise.

    Non-vacuity: the SAME step from the card's own ssh bump moves eta, so the
    comparison can fail.
    """
    assert card.recipe.z_coord.n_levels == 1
    eta0 = np.asarray(card.recipe.initial_state.eta.data)
    rest = _step_fields(card, np.zeros_like(eta0))
    for name, (a, b) in rest.items():
        assert a.shape == b.shape and a.tobytes() == b.tobytes(), name
    moved = _step_fields(card, eta0)
    assert moved["eta"][0].tobytes() != moved["eta"][1].tobytes()


def test_validator_refuses_planted_drift(card):
    cfg = card.recipe.model_config
    bad_cfg = cfg._replace(barotropic=cfg.barotropic._replace(
        n_barotropic_substeps=30))
    bad = card._replace(recipe=card.recipe._replace(model_config=bad_cfg))
    with pytest.raises(ValueError, match="filter/substeps"):
        validate_nemo_testcase_card(bad)
    with pytest.raises(ValueError, match="TSUNAMI_UNMEASURED"):
        validate_nemo_testcase_card(card._replace(unmeasured_features=("x",)))


def test_single_level_is_opt_in_only():
    with pytest.raises(ValueError, match=">= 2 thicknesses"):
        create_z_star_from_thicknesses(np.array([100.0]))
    z = create_z_star_from_thicknesses(np.array([100.0]), allow_single_level=True)
    assert z.n_levels == 1


def _roll_state(fields, sj, si):
    out = {}
    for name, f in fields.items():
        if name in ("uu_b", "u"):      # ni+1 faces: roll the distinct ni, re-close
            g = np.roll(f[:, :-1], (sj, si), axis=(0, 1))
            out[name] = np.concatenate([g, g[:, :1]], axis=1)
        elif name in ("vv_b", "v"):
            g = np.roll(f[:-1], (sj, si), axis=(0, 1))
            out[name] = np.concatenate([g, g[:1]], axis=0)
        else:
            out[name] = np.roll(f, (sj, si), axis=(0, 1))
    return out


def _seam_equivariance_unequal(card, seam_shift):
    """Cells where step(roll(bump on seam)) != roll(step(bump on seam))."""
    eta0 = np.asarray(card.recipe.initial_state.eta.data)  # centre (79, 39)
    sj, si = seam_shift
    on_seam = {k: b for k, (_, b) in _step_fields(
        card, np.roll(eta0, (sj, si), axis=(0, 1))).items()}
    moved = {k: b for k, (_, b) in _step_fields(
        card, np.roll(eta0, (sj + 37, si + 61), axis=(0, 1))).items()}
    want = _roll_state(on_seam, 37, 61)
    return {k: int(np.count_nonzero(moved[k] != want[k]))
            for k in ("eta", "uu_b", "vv_b")}


def test_i_seam_is_translation_equivariant_bit_for_bit(card):
    """B4, i-seam: the bump straddling the periodic i-seam steps exactly as
    the same bump moved 61 cells into the interior."""
    assert _seam_equivariance_unequal(card, (0, -39)) == {
        "eta": 0, "uu_b": 0, "vv_b": 0}


def test_j_seam_is_translation_equivariant_bit_for_bit(card):
    """B4j: NEMO's j-seam is a periodic copy (lbclnk.f90:2028-2034)."""
    assert _seam_equivariance_unequal(card, (-79, 0)) == {
        "eta": 0, "uu_b": 0, "vv_b": 0}


def test_j_seam_plant_walled_step_fires(card):
    """Plant: the same step with the j-wrap off walls the seam."""
    bad = _seam_equivariance_unequal(_walled(card), (-79, 0))
    assert min(bad.values()) > 0, bad


def _f_plane_off(card):
    """The card with f = 0 everywhere f enters (grid and EEN operands)."""
    import jax.numpy as jnp
    r = card.recipe
    g = r.grid._replace(**{k: jnp.zeros_like(getattr(r.grid, k))
                           for k in ("f_T", "f_u", "f_v", "ff_f")})
    raw = r.z_coord.nemo_een_barotropic
    z = r.z_coord._replace(
        nemo_een_barotropic=raw._replace(ff_f=np.zeros_like(raw.ff_f)))
    return card._replace(recipe=r._replace(grid=g, z_coord=z))


def test_symmetric_bump_on_both_seams_steps_transpose_symmetric(card):
    """A bump symmetric under i<->j, centred on the seam corner, steps to
    eta = eta.T and u(j, i) = v(i, j) bit for bit (f = 0: the reflection
    reverses the rotation sense, so it is a symmetry only without f)."""
    flat = _f_plane_off(card)
    n = card.recipe.grid.n_lat if hasattr(card.recipe.grid, "n_lat") else 201
    d = np.minimum(np.arange(n), n - np.arange(n)).astype(float)
    a = np.exp(-(d / 8.0) ** 2)
    eta0 = 0.1 * np.outer(a, a)
    assert eta0.tobytes() == eta0.T.copy().tobytes()
    out = {k: b for k, (_, b) in _step_fields(flat, eta0).items()}
    assert np.abs(out["eta"] - eta0).max() > 1e-6       # it moved
    assert out["eta"].tobytes() == out["eta"].T.copy().tobytes()
    for u, v in (("uu_b", "vv_b"), ("u", "v")):
        uu = out[u][:, :-1].reshape(n, n, -1)[..., 0]
        vv = out[v][:-1].reshape(n, n, -1)[..., 0]
        assert np.abs(uu).max() > 0
        assert uu.tobytes() == vv.T.copy().tobytes(), u
    # the walled step is not symmetric: the plant fires
    walled = {k: b for k, (_, b) in _step_fields(_walled(flat), eta0).items()}
    assert walled["eta"].tobytes() != walled["eta"].T.copy().tobytes()
    # with f on, the reflection is not a symmetry: the f = 0 switch is real
    rot = {k: b for k, (_, b) in _step_fields(card, eta0).items()}
    assert rot["uu_b"][:, :-1].tobytes() != rot["vv_b"][:-1].T.copy().tobytes()


def test_the_step_takes_its_topology_from_the_config_not_the_caller(card):
    """One model instance, stepped under either global state, gives the
    same bits (no stale walled trace); the validator refuses a card whose
    config disagrees with j_periodic."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel)
    r = card.recipe
    assert r.model_config.meridionally_periodic is True
    eta0 = np.roll(np.asarray(r.initial_state.eta.data), -79, axis=0)
    m = LatLonCGridOceanModel(r.grid, r.z_coord, r.model_config)
    off = _step_fields(card, eta0, caller_scope=False, model=m)
    on = _step_fields(card, eta0, caller_scope=True, model=m)
    for f in off:
        assert off[f][1].tobytes() == on[f][1].tobytes(), f
    walled = _step_fields(_walled(card), eta0)
    assert walled["eta"][1].tobytes() != on["eta"][1].tobytes()
    # False is honoured too: a walled config stays walled under a True scope.
    walled_on = _step_fields(_walled(card), eta0, caller_scope=True)
    for f in walled:
        assert walled_on[f][1].tobytes() == walled[f][1].tobytes(), f
    with pytest.raises(ValueError, match="meridionally_periodic"):
        validate_nemo_testcase_card(_walled(card))


def test_public_tendencies_take_their_topology_from_the_config(card):
    """tendencies() and tendencies_with_diagnostics() honour the config's
    y-wrap without a caller scope, and ignore a contrary one."""
    import contextlib
    import jax
    import jax.numpy as jnp
    from legoesm.grids.halo_latlon import meridional_periodicity
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel)
    r = card.recipe
    eta0 = np.roll(np.asarray(r.initial_state.eta.data), -79, axis=0)
    s0 = r.initial_state._replace(
        eta=r.initial_state.eta.replace(data=jnp.asarray(eta0)))

    def bits(model_config, method, scope=None):
        m = LatLonCGridOceanModel(r.grid, r.z_coord, model_config)
        with (contextlib.nullcontext() if scope is None
              else meridional_periodicity(scope)):
            out = getattr(m, method)(s0, dt=card.dt_s)
        return [np.asarray(x).tobytes() for x in jax.tree_util.tree_leaves(out)]

    periodic = r.model_config
    walled = _walled(card).recipe.model_config
    for method in ("tendencies", "tendencies_with_diagnostics"):
        on = bits(periodic, method)
        assert on == bits(periodic, method, scope=False), method
        off = bits(walled, method)
        assert off == bits(walled, method, scope=True), method
        assert on != off, method


def test_j_neighbour_helpers_index_map():
    """Native north-face row j is legoESM face j+1; face 0 is the seam:
    the last native row under the y-wrap, zero (or the given row) walled."""
    import jax.numpy as jnp
    from legoesm.grids import halo_latlon as hl
    a = jnp.arange(1.0, 5.0)[:, None] * jnp.ones((1, 2))   # rows 1, 2, 3, 4
    col = lambda x: np.asarray(x)[:, 0].tolist()            # noqa: E731
    assert col(hl.lat_north(a)) == [2, 3, 4, 0]
    assert col(hl.lat_south(a)) == [0, 1, 2, 3]
    assert col(hl.lat_faces_from_north(a)) == [0, 1, 2, 3, 4]
    assert col(hl.lat_faces_from_north(a, south=a[:1])) == [1, 1, 2, 3, 4]
    with hl.meridional_periodicity(True):
        assert col(hl.lat_north(a)) == [2, 3, 4, 1]
        assert col(hl.lat_south(a)) == [4, 1, 2, 3]
        assert col(hl.lat_faces_from_north(a)) == [4, 1, 2, 3, 4]
        assert col(hl.lat_faces_from_north(a, south=a[:1])) == [4, 1, 2, 3, 4]


def test_mask_rebuilds_and_runtime_check_follow_the_config(card):
    """step_checked's face-mask check and replace_land_mask see the y-wrap
    the config selects, not the caller's scope."""
    from legoesm.grids.halo_latlon import meridional_periodicity
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel)
    from legoesm.ocean.init_latlon_cgrid import replace_land_mask
    r = card.recipe
    s0 = r.initial_state
    m = LatLonCGridOceanModel(r.grid, r.z_coord, r.model_config)
    with meridional_periodicity(False):
        m._assert_runtime_invariants(s0)
    walled = LatLonCGridOceanModel(r.grid, r.z_coord,
                                   _walled(card).recipe.model_config)
    with pytest.raises(ValueError, match="u_mask/v_mask"):
        walled._assert_runtime_invariants(s0)
    v0 = np.asarray(s0.v_mask.data).tobytes()
    mask = s0.land_mask.data
    assert np.asarray(replace_land_mask(
        s0, mask, meridionally_periodic=True).v_mask.data).tobytes() == v0
    assert np.asarray(replace_land_mask(s0, mask).v_mask.data).tobytes() != v0
