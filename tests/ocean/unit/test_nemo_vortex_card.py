"""The VORTEX card: what it resolves, what it refuses, and the oracle gate.

Two layers, the same split the DINO and tank cards use:

* FILE-FREE tests check what is knowable without NEMO's record -- which
  namelist branch is transcribed, that the analytic initial state carries the
  source's anticyclonic sign convention and its ssh-stretched depth, and that
  a broken composition is REFUSED rather than silently rebuilt.
* The ORACLE test is the bit-for-bit comparison against NEMO's own kt=1
  step-entry record.  It is SKIPPED (never silently passed) while that record
  is not on this machine; round 1 writes the acquisition and stops.
"""
from __future__ import annotations

import glob
import math
import struct

import numpy as np
import pytest

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.constants_config import NEMO_CONSTANTS_CONFIG as CONSTANTS
from legoesm.ocean.fidelity.nemo_testcase_recipe import (
    VORTEX_UNMEASURED,
    build_nemo_testcase_card,
    build_vortex_zco_card,
    validate_nemo_testcase_card,
    validate_nemo_testcase_card_for_execution,
    vortex_horizontal_coordinates,
)

# The acquisition script writes here (round 1 of the VORTEX card).
_ORACLE = ("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex/round1/"
           "oracle_step_entry_kt00000001.bin")


@pytest.fixture(scope="module", autouse=True)
def _fp64():
    """Oracle cards are fp64 models; pin the policy instead of inheriting it.

    The certified-card digests below are global-state dependent: under the
    library default (fp32 storage) the same card hashes differently, so a
    digest measured in one session and checked in another would disagree for
    a reason that has nothing to do with the card.  A long mixed battery
    caught exactly that.  Set the policy the cards actually run at, and put
    it back afterwards.
    """
    previous = get_policy()
    set_policy(PrecisionPolicy.fp64())
    yield
    set_policy(previous)


@pytest.fixture(scope="module")
def card():
    return build_vortex_zco_card()


def test_dispatch_resolves_the_card_by_name():
    assert build_nemo_testcase_card("VORTEX-zco").case == "VORTEX-zco"
    with pytest.raises(ValueError, match="unknown NEMO testcase"):
        build_nemo_testcase_card("VORTEX")


def test_card_resolves_the_shipped_namelist(card):
    """namelist_cfg:19-24,32,41,204-216,252-260 and usrdef_nam.F90:96-121."""
    grid = card.recipe.grid
    assert np.asarray(grid.dx_T).shape == (63, 63)
    assert np.unique(np.asarray(grid.dx_T)).tolist() == [30000.0]
    assert np.unique(np.asarray(grid.dy_T)).tolist() == [30000.0]
    assert np.asarray(card.recipe.z_coord.is_active).shape == (63, 63, 10)
    # 61 x 61 wet columns x 10 levels: the outer ring is NEMO's closed
    # boundary (usrdef_zgr.F90:187-193).
    assert int(np.asarray(card.recipe.z_coord.is_active).sum()) == 61 * 61 * 10
    assert (card.dt_s, card.n_steps) == (2880.0, 3000)
    cfg = card.recipe.model_config
    assert cfg.barotropic.n_barotropic_substeps == 48       # nn_e, ln_bt_auto=F
    assert cfg.barotropic.barotropic_time_filter == "nemo_ab3am4"
    assert cfg.eos == "nemo_teos10"          # decision 64 deviation, note BF
    assert cfg.momentum_advection == "flux_form"
    assert cfg.momentum_flux_scheme == "nemo_up3"
    assert cfg.tracer_advection == "fct2"
    assert cfg.pgf_scheme == "nemo_sco"
    assert (cfg.A_v, cfg.K_v) == (1.0e-4, 0.0)              # rn_avm0, rn_avt0
    # namzdf does not set ln_zad_Aimp, so it stays .false. (namelist_ref:1177).
    # The identity this card inherits from resolves it True for OVERFLOW, so
    # the row is here rather than left to inheritance.
    assert cfg.adaptive_implicit_vertadv is False
    assert card.surface_boundary_condition == "none"        # usrdef_sbc zeros


def test_the_beta_plane_is_live_and_centred_on_the_reference_latitude(card):
    source = vortex_horizontal_coordinates()
    # usrdef_hgr.F90:83-84,108,130 -- 1800 km box centred on a T point.
    assert source["glamt"][0, 0] == -930.0
    assert source["glamt"][0, -1] == 930.0
    assert source["glamt"][0, 31] == 0.0 and source["gphit"][31, 0] == 0.0
    f0 = 2.0 * float(CONSTANTS.Omega) * math.sin(math.pi / 180.0 * 38.5)
    assert float(np.asarray(card.recipe.grid.f_T)[31, 31]) == pytest.approx(
        f0, rel=1e-12)
    # A live rotation operator: f must vary across the box, not be a constant.
    assert float(np.ptp(np.asarray(card.recipe.grid.ff_f))) > 3.0e-5


def test_the_card_fails_closed_on_the_coriolis_it_cannot_express(card):
    """namelist_cfg:182,193 -- EEN vorticity under FLUX-FORM momentum.

    NEMO's dyn_vor takes its ln_dynadv_vec=.false. arm and calls vor_een on
    the PLANETARY vorticity, so on this case EEN *is* the Coriolis operator.
    legoESM binds its EEN arm to vector-invariant momentum and gives the
    flux-form branch a 4-point C-grid average instead.  The card must say so
    and be refused for execution, NOT quietly run the average.
    """
    assert card.unmeasured_features == VORTEX_UNMEASURED
    assert len(card.unmeasured_features) == 2
    with pytest.raises(ValueError, match="not execution-ready"):
        validate_nemo_testcase_card_for_execution(card)
    # And the substitution the gap exists to prevent must be unreachable.
    assert not card.recipe.model_config.vorticity_scheme.endswith("_total")


def test_initial_state_is_the_anticyclonic_source_vortex(card):
    """usrdef_istate.F90:70,105,127 -- ``Here Anticyclonic``.

    A sign flip in either velocity component (the single most likely
    transcription defect) turns the eddy cyclonic and is caught here.
    """
    state = card.recipe.initial_state
    ssh = np.asarray(state.eta.data)
    u = np.asarray(state.u.data)[:, 1:, :]      # model[:, 1:] == NEMO's U
    v = np.asarray(state.v.data)[1:, :, :]
    assert ssh[31, 31] == ssh.max() > 0.9       # a surface HIGH at the centre
    # Clockwise about that high: eastward on its northern flank, southward on
    # its eastern flank.
    assert u[35, 31, 0] > 0.0 and u[27, 31, 0] < 0.0
    assert v[31, 35, 0] < 0.0 and v[31, 27, 0] > 0.0
    # usrdef_istate.F90:102,113-115 -- zH = 2500 m, so the four levels whose
    # T depth is below it carry no velocity at all.
    assert np.count_nonzero(u[:, :, 5:]) == 0
    assert np.count_nonzero(v[:, :, 5:]) == 0
    assert np.count_nonzero(u[:, :, :5]) > 0


def test_temperature_uses_the_ssh_stretched_depth_not_the_reference_ladder(card):
    """istate.F90:127-130 hands ``gdept(:,:,:,Kbb)``, the LIVE depth.

    ``rst_read_ssh`` sets ssh before ``dom_qco_zgr`` builds r3t, so the T
    profile is evaluated on ``gdept_0*(1+r3t)``.  Using the 1-D ladder instead
    is a silent 1.8e-4 relative depth error at the vortex centre; this test
    fails if the card ever drops the stretch.
    """
    state = card.recipe.initial_state
    temperature = np.asarray(state.T.data)
    ssh = np.asarray(state.eta.data)
    grav = float(CONSTANTS.g)
    rho0 = float(CONSTANTS.rho_0)
    n2 = 3.0e-3 ** 2
    height = 2500.0
    # Level 9 (4750 m reference) is below zH, so only the stratification term
    # survives and the whole value is a function of the depth operand alone.
    r3t = ssh[31, 31] / 5000.0
    reference = 250.0 + 500.0 * 9
    live = reference * (1.0 + r3t)
    def temp(depth):
        return 20.0 + (rho0 - rho0 * (1.0 + n2 * depth / grav)) / 0.28
    assert temperature[31, 31, 9] == pytest.approx(temp(live), rel=0, abs=1e-12)
    assert temperature[31, 31, 9] != pytest.approx(
        temp(reference), rel=0, abs=1e-12)


def test_salinity_and_land_follow_the_source_mask(card):
    state = card.recipe.initial_state
    salinity = np.asarray(state.S.data)
    active = np.asarray(card.recipe.z_coord.is_active)
    assert np.all(salinity[active] == 35.0)         # usrdef_istate.F90:93
    assert np.all(salinity[~active] == 0.0)
    assert np.all(np.asarray(state.T.data)[~active] == 0.0)
    assert np.all(np.asarray(state.eta.data)[0] == 0.0)
    assert np.all(np.asarray(state.eta.data)[:, -1] == 0.0)


def test_validator_refuses_a_broken_vortex_composition(card):
    """A card that loses its EEN rotation must be REFUSED, not rebuilt."""
    cfg = card.recipe.model_config
    broken = card._replace(unmeasured_features=())
    with pytest.raises(ValueError, match="EEN/flux-form Coriolis gap"):
        validate_nemo_testcase_card(broken)
    broken = card._replace(
        recipe=card.recipe._replace(
            model_config=cfg._replace(vorticity_scheme="een_total")))
    with pytest.raises(ValueError, match="_total vorticity scheme"):
        validate_nemo_testcase_card(broken)
    broken = card._replace(
        recipe=card.recipe._replace(
            model_config=cfg._replace(
                barotropic=cfg.barotropic._replace(
                    n_barotropic_substeps=50))))
    with pytest.raises(ValueError, match="does not match the executed oracle"):
        validate_nemo_testcase_card(broken)
    broken = card._replace(surface_boundary_condition="gyre_usrdef_sbc")
    with pytest.raises(ValueError, match="no surface forcing"):
        validate_nemo_testcase_card(broken)
    broken = card._replace(
        recipe=card.recipe._replace(
            model_config=cfg._replace(adaptive_implicit_vertadv=True)))
    with pytest.raises(ValueError, match="ln_zad_Aimp"):
        validate_nemo_testcase_card(broken)


def _read_step_entry(path, n_lat, n_lon):
    """Parse the record's OWN header; never predict its sizes (note BD)."""
    with open(path, "rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        version, step, nbb, nx, ny, nz, ntr, bits = struct.unpack(
            "=8i", handle.read(32))
        data = np.fromfile(handle, dtype=np.float64)
    assert magic == "NEMO_L1_ENTRY_1", magic
    assert (version, bits) == (1, 64)
    count = nx * ny * nz
    assert data.size == ntr * count + count * 2 + nx * ny, (
        f"payload {data.size} disagrees with the header {(nx, ny, nz, ntr)}")
    halo_x, halo_y = (nx - n_lon) // 2, (ny - n_lat) // 2
    assert nx - 2 * halo_x == n_lon and ny - 2 * halo_y == n_lat

    def xyz(values):
        return values.reshape((nx, ny, nz), order="F")[
            halo_x:nx - halo_x, halo_y:ny - halo_y].transpose(1, 0, 2)

    return {
        "step": step, "Nbb": nbb, "nz": nz,
        "T": xyz(data[:count]),
        "S": xyz(data[count:2 * count]),
        "u": xyz(data[2 * count:3 * count]),
        "v": xyz(data[3 * count:4 * count]),
        "ssh": data[4 * count:].reshape((nx, ny), order="F")[
            halo_x:nx - halo_x, halo_y:ny - halo_y].T,
    }


def _bits_equal(left, right, mask):
    """Compare RAW BITS, so +0.0 and -0.0 are not silently equal.

    ``==`` on float64 calls those two values identical; NEMO's signed halo
    exchange can produce a negative zero where the card produces a positive
    one, and that is a real difference in the record even though no arithmetic
    would notice it.  The count this returns is the one the prereg predicts.
    """
    use = np.asarray(mask, dtype=bool)
    lhs = np.ascontiguousarray(
        np.asarray(left, dtype=np.float64)[use]).view(np.int64)
    rhs = np.ascontiguousarray(
        np.asarray(right, dtype=np.float64)[use]).view(np.int64)
    return int(np.count_nonzero(lhs != rhs))


@pytest.mark.skipif(
    not glob.glob(_ORACLE),
    reason="VORTEX kt=1 step-entry record not acquired on this machine")
def test_initial_state_is_bit_exact_against_the_nemo_record(card):
    oracle = _read_step_entry(glob.glob(_ORACLE)[0], 63, 63)
    assert oracle["step"] == 1
    state = card.recipe.initial_state
    n_lev = int(card.recipe.z_coord.n_levels)
    # NEMO's record carries jpk levels; the card executes jpkm1 of them and
    # record jpk is the permanently dry dummy bottom (card.dummy_bottom_records).
    assert oracle["nz"] == n_lev + card.dummy_bottom_records
    for name in ("T", "S", "u", "v"):
        oracle[name] = oracle[name][..., :n_lev]
    active = np.asarray(card.recipe.z_coord.is_active)
    wet = active[:, :, 0]
    u_face = active & np.roll(active, -1, axis=1)
    u_face[:, -1] = False
    v_face = active & np.roll(active, -1, axis=0)
    v_face[-1] = False
    pairs = {
        "T": (np.asarray(state.T.data), active),
        "S": (np.asarray(state.S.data), active),
        "u": (np.asarray(state.u.data)[:, 1:, :], u_face),
        "v": (np.asarray(state.v.data)[1:, :, :], v_face),
        "ssh": (np.asarray(state.eta.data), wet),
    }
    unequal = {
        name: _bits_equal(oracle[name], value, mask)
        for name, (value, mask) in pairs.items()
    }
    assert unequal == {name: 0 for name in pairs}, unequal


# Measured at the lane tip c09a9e111 (before the VORTEX card existed) and
# re-measured with it present: identical.  The VORTEX card is additive, and
# this row is what keeps it additive -- a later edit to the shared identity
# that leaks into a certified card turns this red instead of moving a
# certified number quietly.
_CERTIFIED_CARD_DIGESTS = {
    "GYRE-zco": "f227194da309e66b",
    "LOCK_EXCHANGE-zco": "42d13c75ea8cbcc6",
    "OVERFLOW-zps": "c2bca636ac2f14ef",
}


def _card_digest(card):
    """Everything a card carries that could change an executed number.

    The first draft hashed only the config, the tracer/velocity/ssh state and
    the T-point Coriolis; a reviewer pointed out that a change to the vertical
    coordinate, to the masks, to the FACE Coriolis fields or to the barotropic
    pair would have slipped through, so all of those are in it now.
    """
    import hashlib
    handle = hashlib.sha256()
    handle.update(repr(card.recipe.model_config).encode())
    state = card.recipe.initial_state
    for name in ("T", "S", "u", "v", "eta", "uu_b", "vv_b", "land_mask",
                 "u_mask", "v_mask"):
        field = getattr(state, name, None)
        handle.update(
            b"none" if field is None
            else np.asarray(field.data, dtype=np.float64).tobytes())
    grid = card.recipe.grid
    for name in ("f_T", "f_u", "f_v", "ff_f", "dx_T", "dy_T"):
        value = getattr(grid, name, None)
        handle.update(
            b"none" if value is None
            else np.asarray(value, dtype=np.float64).tobytes())
    z = card.recipe.z_coord
    for name in ("dz_ref", "z_half_ref", "h_partial", "is_active",
                 "bottom_level", "nemo_e3t_0"):
        value = getattr(z, name, None)
        handle.update(
            b"none" if value is None
            else np.asarray(value, dtype=np.float64).tobytes())
    handle.update(
        np.asarray(card.recipe.land_mask, dtype=np.float64).tobytes())
    handle.update(repr((
        card.dt_s, card.n_steps, card.dummy_bottom_records,
        card.bbl_adv_option, card.bbl_gamma_s,
        card.bbl_diffusive_option, card.bbl_aht_m2_s,
        card.surface_boundary_condition, card.surface_input_operator,
        card.transcendentals, card.precision_policy,
        card.unmeasured_features, card.icebergs_enabled,
        card.iceberg_inputs)).encode())
    return handle.hexdigest()[:16]


@pytest.mark.parametrize("case", sorted(_CERTIFIED_CARD_DIGESTS))
def test_the_certified_cards_are_untouched_by_the_vortex_card(case):
    assert _card_digest(build_nemo_testcase_card(case)) == \
        _CERTIFIED_CARD_DIGESTS[case]
