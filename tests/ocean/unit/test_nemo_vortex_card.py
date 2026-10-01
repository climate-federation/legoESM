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

import contextlib
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

# The acquisition script writes one directory per round.  BOTH are compared,
# whichever exist: round 1's was produced on a deck that selected a different
# equation of state, and the preregistration's claim is that this makes NO
# difference to the initial state, because nothing in it reads the equation of
# state.  So a round-2 record that disagrees with round 1's on these five rows
# is the finding, and testing only the newest one would hide it.
_ORACLE_DIRS = ("round1", "round2")
_ORACLE = ("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex/*/"
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


def test_vector_card_states_nemos_two_wzv_calls_explicitly():
    """stprk3_stg:289-300: vector momentum and tracers use separate solves."""
    cfg = build_nemo_testcase_card("VORTEX_VEC-zco").recipe.model_config
    assert cfg.wzv_call2_evaluation == "nemo_literal"
    assert cfg.nemo_stage_momentum_wzv_split is True
    flux_cfg = build_nemo_testcase_card("VORTEX-zco").recipe.model_config
    assert flux_cfg.wzv_call2_evaluation == "generic"
    assert flux_cfg.nemo_stage_momentum_wzv_split is None


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
    # namelist_cfg:130-138 (&nameos), decision 69: the one card on NEMO's
    # simplified equation of state, with its OWN coefficients -- the shared
    # defaults are DINO's and every one of them differs.
    assert cfg.eos == "nemo_seos"
    assert cfg.eos_depth == "insitu"            # rn_mu1 = rn_mu2 = 0
    assert (cfg.eos_nemo_seos.a0, cfg.eos_nemo_seos.b0) == (0.28, 0.0)
    assert (cfg.eos_nemo_seos.lambda1, cfg.eos_nemo_seos.lambda2) == (0.0, 0.0)
    assert (cfg.eos_nemo_seos.mu1, cfg.eos_nemo_seos.mu2) == (0.0, 0.0)
    assert cfg.eos_nemo_seos.nu == 0.0
    assert (cfg.eos_nemo_seos.T0, cfg.eos_nemo_seos.S0) == (10.0, 35.0)
    assert cfg.eos_nemo_seos.rho0 == float(CONSTANTS.rho_0)
    # namelist_cfg:182,193 -- EEN vorticity on flux-form momentum.
    assert cfg.vorticity_scheme == "een_planetary"
    assert cfg.coriolis_scheme == "explicit_ab2"
    assert cfg.een_e3f_scheme == "nemo_avg4"
    assert cfg.een_metric_weighting == "nemo"
    assert cfg.een_q_boundary == "nemo_live"
    assert cfg.barotropic.barotropic_coriolis == "een_metric"
    assert cfg.barotropic_coriolis_split == "live"
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


def test_the_card_runs_nemos_own_coriolis_operator(card):
    """namelist_cfg:182,193 -- EEN vorticity under FLUX-FORM momentum.

    Round 1 declared this pair as a gap and fixed the card closed.  Round 2
    transcribed it: dynvor.F90:874 routes np_EEN, dyn_vor_init:891-893 hands
    the flux-form arm ntot = np_CME, and np_CME's metric term is built from
    first differences of the scale factors (dynvor.F90:905-908), which are
    bitwise zero on this Cartesian mesh -- so the energy-and-enstrophy triad
    on the planetary vorticity IS the whole operator here.  The card must run
    that, never legoESM's 4-point C-grid average.
    """
    assert card.unmeasured_features == VORTEX_UNMEASURED == ()
    validate_nemo_testcase_card_for_execution(card)     # no longer refused
    cfg = card.recipe.model_config
    assert cfg.vorticity_scheme == "een_planetary"
    # f rides the triad, so the separate rotation must be off or it enters twice.
    assert cfg.coriolis_scheme == "explicit_ab2"


def test_the_transcribed_coriolis_refuses_every_untranscribed_pairing():
    """The new arm is NEMO's flux-form one ONLY; nothing else may select it."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        assert_een_planetary_metric_term_vanishes,
    )
    base = build_nemo_testcase_card("VORTEX-zco").recipe
    def build(**kw):
        LatLonCGridOceanModel(
            base.grid, base.z_coord, base.model_config._replace(**kw))
    # Vector-invariant momentum routes NEMO to np_CRV, where the triad also
    # carries the RELATIVE vorticity -- a different operator.
    # (the NEMO UP3 vertical arm is part of the same flux-form program, so it
    # is moved off too; otherwise its own guard fires first and this row would
    # prove nothing about the vorticity guard)
    with pytest.raises(ValueError, match="FLUX-FORM vorticity arm"):
        build(momentum_advection="vector_invariant",
              vertical_momentum_scheme="upwind_perturbation")
    with pytest.raises(ValueError, match="explicit_ab2"):
        build(coriolis_scheme="matsuno_split")
    with pytest.raises(ValueError, match="nemo_avg4"):
        build(een_e3f_scheme="min")
    with pytest.raises(ValueError, match="een_metric_weighting"):
        build(een_metric_weighting="off")
    with pytest.raises(ValueError, match="barotropic arm"):
        build(barotropic_coriolis_split="frozen")
    # And the dropped np_CME metric term is only zero on a Cartesian mesh.
    assert_een_planetary_metric_term_vanishes(base.grid)
    class _Stretched:
        dx_T = np.array([[30000.0, 30001.0]])
    with pytest.raises(ValueError, match="np_CME"):
        assert_een_planetary_metric_term_vanishes(_Stretched())


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
    broken = card._replace(unmeasured_features=("a gap nobody proved",))
    with pytest.raises(ValueError, match="VORTEX_UNMEASURED"):
        validate_nemo_testcase_card(broken)
    broken = card._replace(
        recipe=card.recipe._replace(
            model_config=cfg._replace(vorticity_scheme="al81")))
    with pytest.raises(ValueError, match="een_planetary"):
        validate_nemo_testcase_card(broken)
    broken = card._replace(
        recipe=card.recipe._replace(
            model_config=cfg._replace(eos="nemo_teos10")))
    with pytest.raises(ValueError, match="simplified equation of state"):
        validate_nemo_testcase_card(broken)
    broken = card._replace(
        recipe=card.recipe._replace(
            model_config=cfg._replace(
                eos_nemo_seos=cfg.eos_nemo_seos._replace(a0=0.165))))
    with pytest.raises(ValueError, match="simplified equation of state"):
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


def _vortex_legacy_config(cfg):
    """The composition this round replaced: 4-point C-grid average Coriolis."""
    return cfg._replace(
        vorticity_scheme="al81", coriolis_scheme="explicit_ab2",
        een_e3f_scheme="min", een_metric_weighting="off",
        een_q_boundary="neumann_fill",
        barotropic=cfg.barotropic._replace(barotropic_coriolis="avg"),
        barotropic_coriolis_split="frozen")


def _vortex_momentum_tendency(recipe, cfg):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    model = LatLonCGridOceanModel(recipe.grid, recipe.z_coord, cfg)
    out = model.tendencies(recipe.initial_state)
    return np.asarray(out.du_dt.data), np.asarray(out.dv_dt.data)


def test_the_transcribed_coriolis_is_a_rotation_and_not_a_missing_one():
    """NON-VACUITY, and it must fail BOTH ways the feature can break.

    A first draft of this row only asked that the transcribed arm differ from
    the 4-point average it replaced.  A reviewer pointed out that this passes
    for the worst possible regression: delete the new branch and the card has
    NO rotation at all (the separate face-Coriolis add is gated off for this
    scheme), which also differs from the average -- by far more.  So the row
    now pins a BAND, against the natural scale f*|u|.

    Measured on this card: two stencils of the SAME rotation differ by 1.2e-4
    of f*|u|, while a card with the rotation missing differs by 0.76 of it --
    four orders of magnitude apart, so the band is not delicate.
    """
    recipe = build_nemo_testcase_card("VORTEX-zco").recipe
    cfg = recipe.model_config
    f_max = float(np.max(np.abs(np.asarray(recipe.grid.ff_f))))
    u_max = max(float(np.max(np.abs(np.asarray(recipe.initial_state.u.data)))),
                float(np.max(np.abs(np.asarray(recipe.initial_state.v.data)))))
    scale = f_max * u_max
    assert scale > 0.0
    new_u, new_v = _vortex_momentum_tendency(recipe, cfg)
    old_u, old_v = _vortex_momentum_tendency(recipe, _vortex_legacy_config(cfg))
    moved = max(float(np.max(np.abs(new_u - old_u))),
                float(np.max(np.abs(new_v - old_v))))
    # It is a DIFFERENT operator: a dead branch, or the same stencil under a
    # new name, turns this red.
    assert moved > 1e-12, moved
    # It is still a ROTATION: a dropped f x u turns this red.
    assert moved < 1e-2 * scale, (moved, scale)
    # And the rotation is actually there, at the right order of magnitude.
    assert float(np.max(np.abs(new_u))) > 0.1 * scale


def test_the_matching_barotropic_arm_is_executed_not_only_declared():
    """NON-VACUITY for the SECOND arm, which a tendency call never reaches.

    The baroclinic trend above is one of the two places NEMO runs this
    operator; the other is inside the split-explicit substeps.  Only a STEP
    executes those, so this row takes one and requires the result to depend on
    which barotropic Coriolis was selected.
    """
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    card = build_nemo_testcase_card("VORTEX-zco")
    recipe, cfg = card.recipe, card.recipe.model_config
    states = []
    for which in (cfg, _vortex_legacy_config(cfg)):
        model = LatLonCGridOceanModel(recipe.grid, recipe.z_coord, which)
        states.append(np.asarray(
            model.step(recipe.initial_state, card.dt_s).u.data))
    assert states[0].shape == states[1].shape
    assert float(np.max(np.abs(states[0] - states[1]))) > 0.0


def test_the_nemo_seos_the_card_selects_reaches_the_model():
    """NON-VACUITY for the equation of state, through the MODEL.

    A first draft called the equation-of-state factory directly with the
    card's coefficients.  A reviewer pointed out that this proves nothing
    about the three production call sites this round threaded: revert all
    three and the row still passes, because the test supplies the
    coefficients itself.  So it goes through the model instead, and the
    control is a card carrying the SHARED defaults -- which are another
    experiment's.  If the coefficients stop reaching the model, the two
    tendencies coincide and this turns red.
    """
    from legoesm.ocean.eos import NemoSEOSConfig
    recipe = build_nemo_testcase_card("VORTEX-zco").recipe
    cfg = recipe.model_config
    assert cfg.eos_nemo_seos is not None
    mine_T, mine_S = _vortex_momentum_tendency(recipe, cfg)
    defaults = cfg._replace(eos_nemo_seos=NemoSEOSConfig())
    theirs_T, theirs_S = _vortex_momentum_tendency(recipe, defaults)
    # The pressure-gradient force is built from density, so a different fluid
    # is a different momentum tendency.
    assert float(np.max(np.abs(mine_T - theirs_T))) > 0.0


def test_the_nemo_seos_the_card_selects_is_the_shipped_law():
    """And the law itself is namelist_cfg's, not DINO's.

    eosbn2.F90:295-302 with rn_b0 = rn_lambda* = rn_mu* = rn_nu = 0 reduces to
    rho = rho0 - 0.28*(T - 10): linear in temperature, and blind to BOTH
    salinity and depth.  Those three properties are what a wrong coefficient
    set would break, so they are asserted rather than the coefficients alone.
    """
    from legoesm.ocean.eos import make_eos_fn
    cfg = build_nemo_testcase_card("VORTEX-zco").recipe.model_config
    eos = make_eos_fn(cfg.eos, eos_nemo_seos=cfg.eos_nemo_seos)
    rho0 = float(cfg.eos_nemo_seos.rho0)
    T = np.array([10.0, 20.0, 4.0])
    S = np.array([35.0, 35.0, 35.0])
    p = np.zeros(3)
    assert np.allclose(np.asarray(eos(T, S, p)), rho0 - 0.28 * (T - 10.0),
                       rtol=0, atol=1e-12)
    # salinity-blind (rn_b0 = rn_nu = 0)
    assert np.allclose(np.asarray(eos(T, S + 5.0, p)), np.asarray(eos(T, S, p)),
                       rtol=0, atol=0)
    # depth-blind (rn_mu1 = rn_mu2 = 0)
    assert np.allclose(np.asarray(eos(T, S, p + 5.0e7)),
                       np.asarray(eos(T, S, p)), rtol=0, atol=0)
    # and the shared defaults are a DIFFERENT fluid
    assert not np.allclose(np.asarray(make_eos_fn("nemo_seos")(T, S, p)),
                           np.asarray(eos(T, S, p)))


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


def _ulp_distance(left, right):
    """Representable floats between two arrays, correct ACROSS ZERO.

    Raw IEEE-754 bit patterns are not monotonic across the sign boundary:
    subtracting them calls a pair straddling zero about 2**63 apart.  This
    card's unequal cells are all one-signed, so the naive form happened to be
    right here -- a reviewer flagged it as an instrument that would mislead the
    next card rather than a defect in this one.  Mapping negatives onto the
    two's-complement continuation fixes it for every case.
    """
    def key(x):
        bits = np.ascontiguousarray(
            np.asarray(x, dtype=np.float64)).view(np.int64).copy()
        neg = bits < 0
        bits[neg] = np.int64(np.uint64(0x8000000000000000)) - bits[neg]
        return bits
    return np.abs(key(left).astype(object) - key(right).astype(object))


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
@pytest.mark.parametrize("record", sorted(glob.glob(_ORACLE)))
def test_initial_state_is_bit_exact_against_the_nemo_record(card, record):
    oracle = _read_step_entry(record, 63, 63)
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
    # MEASURED against round 1's record, and this REFUTES round 1's
    # preregistration, which predicted a bit-exact initial state on all five
    # fields.  Temperature and salinity are bit-exact.  The other three are
    # not, and the residual is the COMPILED TRANSCENDENTAL FLOOR, not a
    # transcription defect: every unequal cell is 1 or 2 ULP (u: 692 at 1 ULP
    # + 96 at 2; ssh: 96 at 1 + 8 at 2), and they sit in the far tail of the
    # eddy's Gaussian, where the unequal ssh values run down to 4.5e-92 while
    # the O(0.9 m) centre is exact.  libm's exp and the compiled exp disagree
    # in the last bit there; the association is NEMO's own.  The card already
    # declares transcendentals="libm" for exactly this reason.
    assert unequal == {"T": 0, "S": 0, "u": 788, "v": 788, "ssh": 104}, unequal
    for name in ("u", "v", "ssh"):
        value, mask = pairs[name]
        ulp = _ulp_distance(oracle[name][mask], np.asarray(value)[mask])
        assert int(ulp.max()) <= 2, (name, int(ulp.max()))


# Round 1 measured these at lane tip c09a9e111.  They are RE-PINNED here, for
# two reasons that are kept separate on purpose:
#
#  (1) The lane advanced 35 commits between that tip and 5301122fe2ef, and
#      those commits legitimately moved all three cards.  The drift is the
#      lane's, not this card's: the round-2 values below were measured on the
#      forward-ported tip with every VORTEX round-2 edit STASHED, and they
#      already differed from round 1's.
#  (2) Round 2 added ONE field to the shared model config (eos_nemo_seos), and
#      this digest hashes repr(model_config), so a purely additive default
#      moves it.  Nothing executed changed.  That is not an argument, it is
#      measured: with the round-2 edits stashed and unstashed, a field-by-field
#      diff of all three cards' resolved configs reports
#      added=['eos_nemo_seos'] removed=[] changed=[] on each, and the added
#      field is None on all three -- which the companion test below asserts.
_CERTIFIED_CARD_DIGESTS_ROUND1 = {          # lane tip c09a9e111, superseded
    "GYRE-zco": "f227194da309e66b",
    "LOCK_EXCHANGE-zco": "42d13c75ea8cbcc6",
    "OVERFLOW-zps": "c2bca636ac2f14ef",
}
_CERTIFIED_CARD_DIGESTS_TIP = {             # 5301122fe2ef, round-2 edits STASHED
    "GYRE-zco": "af1f9f5e99d9a30f",
    "LOCK_EXCHANGE-zco": "af88a6e54a70263f",
    "OVERFLOW-zps": "375ec3040cc4ce53",
}
_CERTIFIED_CARD_DIGESTS_ROUND2 = {          # 5bbac73f6, superseded by round 4
    "GYRE-zco": "eaef11b4c2e37a31",
    "LOCK_EXCHANGE-zco": "159ca3d07db0a3f5",
    "OVERFLOW-zps": "73174751388503aa",
}
# Round 4 (DECISION 75).  GYRE's digest moved between `5bbac73f6` and this
# round, and the move was measured field by field on the resolved config
# rather than assumed: FIVE rows, and nothing else on any card.
#
#   value changed   physics.lateral_mixing.biharmonic.enforce_cfl  False -> True
#   field removed   physics.lateral_mixing.biharmonic.cfl_dt_estimate  (was 3600.0)
#   field removed   physics.bottom_drag.linear.r                       (was 0.0011)
#   field removed   physics.bottom_drag.quadratic.C_d                  (was 0.0025)
#   field removed   physics.vertical_mixing.kpp.c_b                    (was 0.599)
#
# All five sit on blocks this card does not select (no lateral mixing, no
# bottom drag, TKE rather than KPP), so nothing executed changed.  The ONE
# value is now stated by the card itself (see the default-independence test
# below); the four removals are fields that no longer exist, which no card
# can state, so they are recorded here and the pin moves.  LOCK_EXCHANGE and
# OVERFLOW carry no physics block at all and their pins are unchanged.
# ROUND 5 moves all three again, for one reason and only one: the model
# configuration gained a field, ``nemo_first_wzv_after_ssh``, which names
# which NEMO time-stepping program's after-SSH slot the first wzv call reads.
# The digest hashes the NamedTuple's printed form and that prints every
# field, so adding one moves every card's digest whether or not the card
# selects anything with it.  The tanks state "" (they never reach the
# branch) and nothing they execute changed -- their ladders are byte-equal
# in this round's evidence.  GYRE states "rk3_extrapolated", which is the
# value it already resolved to before the field existed, so the digest moved
# and its numbers did not: the certified kt=1..10 rows are re-measured in the
# round-5 receipt.
# Round 191 / Decision 78 changes GYRE's explicit value from the uncarried to
# the carried form after round 6 measured the latter.  The tank values and
# digests stay fixed; GYRE's new digest is pinned with its switched trajectory.
_CERTIFIED_CARD_DIGESTS = {                 # round 191 / Decision 78
    "GYRE-zco": "308af4c536cbfd9a",
    "LOCK_EXCHANGE-zco": "d794c4c5cb3dd880",
    "OVERFLOW-zps": "2bb9d9be75fd924d",
}


@pytest.mark.parametrize("case", sorted(_CERTIFIED_CARD_DIGESTS))
def test_the_only_config_change_to_a_certified_card_is_the_added_eos_field(case):
    """The digest moved; prove WHAT moved, rather than re-pinning blind.

    ``eos_nemo_seos`` is the one field round 2 added to the shared config.  On
    every certified card it must be ``None`` -- i.e. those cards keep reading
    the defaults they always read, and the VORTEX coefficients reach VORTEX
    alone.  A future edit that gives one of them a value turns this red.
    """
    cfg = build_nemo_testcase_card(case).recipe.model_config
    assert "eos_nemo_seos" in cfg._fields
    assert cfg.eos_nemo_seos is None
    assert cfg.eos == "nemo_teos10"


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


# --------------------------------------------------------------------------
# The two VORTEX decks (decision 73).  The vector-EEN card is the flux card
# with ONE thing changed: the momentum scheme set.  That claim is what makes
# the pair a one-variable comparison, so it is asserted from the two committed
# patches rather than described in prose.
_DECKS = (
    "scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_vortex")
_SHIPPED_DECK = (
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/tests/VORTEX/EXPREF/"
    "namelist_cfg")


def _resolved_deck(patch_name, tmp_path):
    """Apply one committed deck patch to the shipped namelist and return it."""
    import shutil
    import subprocess
    from pathlib import Path

    tmp_path.mkdir(parents=True, exist_ok=True)
    target = tmp_path / "namelist_cfg"
    shutil.copy(_SHIPPED_DECK, target)
    patch = Path(__file__).resolve().parents[3] / _DECKS / patch_name
    subprocess.run(["patch", "-s", str(target)], check=True,
                   stdin=patch.open("rb"))
    return target.read_text().splitlines()


@pytest.mark.skipif(
    not glob.glob(_SHIPPED_DECK),
    reason="NEMO's shipped VORTEX namelist is not on this machine")
def test_the_vector_deck_differs_from_the_flux_deck_only_in_the_momentum_set(
        tmp_path):
    """dynadv.F90:184-190 counts the forms; dynvor.f90:855-868 reads the count.

    So these two lines, and the experiment name that keeps the two records
    apart, are the WHOLE difference between the two VORTEX cards.  A third
    differing line would mean the pair is not a controlled comparison.
    """
    flux = _resolved_deck("namelist_cfg_omip_l1.patch", tmp_path / "a")
    vec = _resolved_deck("namelist_cfg_vec_een.patch", tmp_path / "b")
    assert len(flux) == len(vec)
    moved = [(a, b) for a, b in zip(flux, vec) if a != b]
    keys = sorted(line.split("=")[0].strip() for line, _ in moved)
    assert keys == ["cn_exp", "ln_dynadv_up3", "ln_dynadv_vec"], moved
    resolved = {key: value for key, value in
                (line.split("=", 1) for line in vec if "=" in line)}
    assert ".true." in resolved["   ln_dynadv_vec "]
    assert ".false." in resolved["   ln_dynadv_up3 "]
    # Both cards run the SAME vorticity scheme; only what it is handed differs.
    assert any(line.strip().startswith("ln_dynvor_een") and ".true." in line
               for line in vec)
    assert sum(1 for line in vec
               if line.strip().startswith(("ln_dynadv_vec", "ln_dynadv_cen2",
                                           "ln_dynadv_up3"))
               and ".true." in line) == 1


# --------------------------------------------------------------------------
# DECISION 75 (operator note BL addendum): a card's resolved configuration
# must not depend on a library default.  The GYRE digest moved in the first
# place because main flipped one, and the fix is for the card to state the
# field -- never to re-pin the digest alone.  This is the gate for that.
_ALL_NEMO_TESTCASE_CARDS = ("GYRE-zco", "LOCK_EXCHANGE-zco", "OVERFLOW-zps",
                            "VORTEX-zco", "VORTEX_VEC-zco")
# Every DINO recipe, not only the two NEMO-literal ones: decision 75 is
# about a library default moving under ANY card, and the two remaining
# recipes build their lateral mixing through the same two builders.
_DINO_NEMO_RECIPES = ("nemo_dino_kamm", "nemo_dino_kamm_mlf",
                      "legoesm_default", "nemo_paper")


@contextlib.contextmanager
def _lateral_mixing_defaults_flipped():
    """Flip every ``enforce_cfl`` a card could inherit, then put them back.

    Two layers have to move, because a NamedTuple's default sub-config is a
    frozen INSTANCE built when its class was defined: flipping
    ``BiharmonicConfig``'s own default does not reach the instance sitting in
    ``LateralMixingConfig``'s defaults, which is the one the cards inherit.
    """
    from legoesm.ocean.physics.lateral_mixing import config as lmc

    def _flip(cls, **replacements):
        before = cls.__new__.__defaults__
        offset = len(cls._fields) - len(before)
        values = list(before)
        for name, value in replacements.items():
            values[cls._fields.index(name) - offset] = value
        cls.__new__.__defaults__ = tuple(values)
        return before

    restore = [
        (lmc.HarmonicConfig, _flip(
            lmc.HarmonicConfig,
            enforce_cfl=not lmc.HarmonicConfig().enforce_cfl)),
        (lmc.BiharmonicConfig, _flip(
            lmc.BiharmonicConfig,
            enforce_cfl=not lmc.BiharmonicConfig().enforce_cfl)),
    ]
    restore.append((lmc.LateralMixingConfig, _flip(
        lmc.LateralMixingConfig,
        harmonic=lmc.HarmonicConfig(),
        biharmonic=lmc.BiharmonicConfig())))
    try:
        yield
    finally:
        for cls, before in restore:
            cls.__new__.__defaults__ = before


def test_the_flip_is_a_real_flip():
    """The instrument moves something, or the tests below prove nothing.

    This is the non-vacuity plant for the whole section: it builds the
    lateral-mixing block the way the NEMO cards used to build it -- naming the
    scheme and inheriting the rest -- and shows that the flip DOES move it.
    """
    from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig

    before = LateralMixingConfig(scheme="none")
    with _lateral_mixing_defaults_flipped():
        during = LateralMixingConfig(scheme="none")
        assert during.biharmonic.enforce_cfl is not before.biharmonic.enforce_cfl
        assert during.harmonic.enforce_cfl is not before.harmonic.enforce_cfl
    assert LateralMixingConfig(scheme="none") == before


@pytest.mark.parametrize("case", _ALL_NEMO_TESTCASE_CARDS)
def test_no_nemo_card_inherits_the_explicit_cfl_cap_from_the_library(case):
    """Build the card twice, with the library default flipped in between."""
    clean = _card_digest(build_nemo_testcase_card(case))
    with _lateral_mixing_defaults_flipped():
        flipped = _card_digest(build_nemo_testcase_card(case))
    assert flipped == clean, (
        f"{case}'s resolved configuration follows a lateral-mixing library "
        "default; state the field on the card instead of re-pinning")


@pytest.mark.parametrize("case", _ALL_NEMO_TESTCASE_CARDS)
def test_every_nemo_card_states_the_explicit_cfl_cap_or_carries_no_block(case):
    """And say WHICH value each card states, so the digest is attributable."""
    physics = build_nemo_testcase_card(case).recipe.model_config.physics
    if physics is None:            # the tanks and both VORTEX cards
        return
    mixing = physics.lateral_mixing
    assert mixing.scheme == "none"
    assert mixing.harmonic.enforce_cfl is False
    assert mixing.biharmonic.enforce_cfl is True


@pytest.mark.parametrize("recipe_name", _DINO_NEMO_RECIPES)
def test_no_dino_nemo_card_inherits_the_explicit_cfl_cap(recipe_name):
    from legoesm.ocean.experiments.dino import (
        dino_config_for_recipe, dino_lat_lon_grid, dino_lat_lon_model_config,
    )

    def resolved():
        cfg = dino_config_for_recipe(recipe_name)
        # n_lon is a pure setup knob; the card is the scheme selection.
        grid = dino_lat_lon_grid(cfg, n_lon=12)
        _model, physics = dino_lat_lon_model_config(grid, cfg)
        return physics.lateral_mixing

    clean = resolved()
    with _lateral_mixing_defaults_flipped():
        flipped = resolved()
    assert flipped.harmonic == clean.harmonic
    assert flipped.biharmonic == clean.biharmonic
    assert clean.harmonic.enforce_cfl is False
    assert clean.biharmonic.enforce_cfl is True


# --- VORTEX round 5: the first wzv call's after-SSH is STATED, not inferred ---
# Decision 75 bans a scientific choice that lives in a default, and round 163
# rejected keying one selector off an unrelated one.  The form the first wzv
# call reads is therefore a field of its own, and a card that reaches the
# branch without stating it raises.

# Read from the model, never re-listed here: a census computed from a
# re-derived condition is how a gate came to disagree with the code it
# gated (operator note AR finding 2).  Round 6 added a third form; Decision 78
# selects it on GYRE and VORTEX-vector while ORCA2 waits for its own ladder.
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (  # noqa: E402
    NEMO_FIRST_WZV_AFTER_SSH_FORMS as _AFTER_SSH_FORMS,
)


@pytest.mark.parametrize("case", _ALL_NEMO_TESTCASE_CARDS)
def test_every_card_that_reaches_the_first_wzv_branch_states_its_after_ssh(case):
    config = build_nemo_testcase_card(case).recipe.model_config
    stated = config.nemo_first_wzv_after_ssh
    if config.zad_qco_evaluation == "nemo_literal":
        assert stated in _AFTER_SSH_FORMS, (
            f"{case} resolves NEMO's own first wzv call and must STATE which "
            "program's after-SSH slot it reads")
    else:
        assert stated == "", (
            f"{case} never reaches that branch, so it must not carry a value "
            "that looks like a selection")


@pytest.mark.parametrize("recipe_name", _DINO_NEMO_RECIPES)
def test_every_dino_nemo_recipe_states_its_after_ssh(recipe_name):
    from legoesm.ocean.experiments.dino import dino_config_for_recipe

    cfg = dino_config_for_recipe(recipe_name)
    if cfg.zad_qco_evaluation == "nemo_literal":
        assert cfg.nemo_first_wzv_after_ssh in _AFTER_SSH_FORMS


def test_the_after_ssh_form_does_not_follow_the_time_integrator():
    """The non-vacuity plant for the whole idea: the two cards that state the
    two DIFFERENT forms must not be separable by the integrator field, or the
    old inference would still reproduce every card and nothing was fixed."""
    from legoesm.ocean.experiments.dino import dino_config_for_recipe

    vortex = build_nemo_testcase_card("VORTEX_VEC-zco").recipe.model_config
    dino = dino_config_for_recipe("nemo_dino_kamm")
    assert vortex.nemo_first_wzv_after_ssh == "rk3_extrapolated_carried"
    assert dino.nemo_first_wzv_after_ssh == "leapfrog_continuity"
    # The inference this replaced read momentum_time_integrator, and on the
    # DINO card that field is not even set to the value the inference keyed
    # on -- which is exactly how a hidden coupling gets a card wrong without
    # anyone seeing it.
    assert getattr(dino, "momentum_time_integrator", None) != "rk3_ws"
