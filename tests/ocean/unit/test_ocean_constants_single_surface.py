"""The ocean physical constants have exactly ONE storage surface.

``LatLonCGridOceanConfig`` used to carry ``g`` / ``rho_0`` / ``omega`` as
NamedTuple fields IN ADDITION to ``constants: ConstantsConfig``.  The two
surfaces diverged silently: the NEMO/DINO oracle cards set the flat ``g`` and it
reached the momentum path, while every consumer of ``config.constants.g``
(``compute_N2``, the GM/Redi slope + ldf_eiv kappa builders, KPP/TKE) kept the
``legoesm.constants`` default -- a 5.0e-5 relative gap that was the entire
production N^2 residual against NEMO's own dumped ``rn2b`` (#1226).

The fix is structural, not a validator: within ``LatLonCGridOceanConfig`` (and
the physics pipeline it propagates into) ``constants`` is the only storage,
``g``/``rho_0``/``omega`` are read-only properties over it, and ``from_flat`` /
``replace_flat`` ROUTE the flat kwargs into it (and propagate the resolved set
into the physics pipeline).

SCOPE (the docstring used to overstate this): "exactly ONE storage" is about the
dynamics / vertical-mixing read path.  ``FluxFeedbackConfig``
(``physics/surface_forcing/config.py``) DELIBERATELY carries its own
``c_sw``/``rho_0`` -- the Veros surface-forcing block owns ``cp_0`` there -- and
nothing propagates ``constants`` into it.  That is by design, not a leak.

These tests pin that a divergence CANNOT be recreated:

  * re-adding a flat ``g``/``rho_0``/``omega`` NamedTuple field makes
    ``test_constants_have_a_single_storage_surface`` FAIL (the
    synthetic-violation check for the whole file);
  * a flat scalar and a ``constants=`` that disagree RAISE;
  * a physics pipeline pinned to a different set RAISES, both at ``from_flat``
    and at model-config validation (the raw-constructor bypass);
  * the ``nemo_dino_kamm_mlf`` oracle card's ``g`` reaches BOTH surfaces, and
    so does its ``c_p`` -> ``c_sw`` (the ALIAS case: the names differ, so a
    field-name-matching check would never have caught it);
  * ``dino_lat_lon_model_config``'s SECOND return value is the routed physics
    block, not the pre-routing local (``create_forcings()["physics_config"]``
    hands that object straight to callers);
  * an OLD run-manifest with top-level ``g``/``rho_0``/``omega`` still loads;
  * ``convective_K_A_flag`` (which took no ``g`` at all) honours the one it is
    given.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.ocean.constants_config import (
    ConstantsConfig, NEMO_CONSTANTS_CONFIG, VEROS_CONSTANTS_CONFIG,
)
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.physics.combined import OceanPhysicsConfig
from legoesm.ocean.physics.convection.config import EnhancedDiffusionConfig
from legoesm.ocean.physics.convection.enhanced_diffusion import (
    convective_K_A_flag,
)
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.vertical import create_ocean_z_star


@pytest.fixture(autouse=True)
def _x64():
    orig = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


# --------------------------------------------------------------------------
# The structural guarantee (synthetic-violation check for this whole file)
# --------------------------------------------------------------------------

def test_constants_have_a_single_storage_surface():
    """``g``/``rho_0``/``omega`` are PROPERTIES, never a second field.

    SYNTHETIC VIOLATION: re-declare ``g: float = constants.g`` on
    ``LatLonCGridOceanConfig`` and this test goes red -- which is the whole
    point, because that second field is what silently diverged.
    """
    fields = set(LatLonCGridOceanConfig._fields)
    assert "constants" in fields
    for name in ("g", "rho_0", "omega", "c_sw", "R_earth"):
        assert name not in fields, (
            f"{name!r} is a NamedTuple field again -- that recreates the second "
            "constant surface. It must route into `constants` via from_flat."
        )
    for name in ("g", "rho_0", "omega"):
        assert isinstance(getattr(LatLonCGridOceanConfig, name), property)


def test_flat_reads_track_the_single_storage():
    cfg = LatLonCGridOceanConfig.from_flat(constants=NEMO_CONSTANTS_CONFIG)
    assert cfg.g == NEMO_CONSTANTS_CONFIG.g
    assert cfg.rho_0 == NEMO_CONSTANTS_CONFIG.rho_0
    assert cfg.omega == NEMO_CONSTANTS_CONFIG.Omega
    # ...and the read-only property cannot be made to disagree: the old
    # ``_replace(g=...)`` escape hatch is now a LOUD TypeError instead of a
    # half-applied second value (use ``replace_flat``).
    with pytest.raises(TypeError, match="unexpected field names"):
        cfg._replace(g=constants.g)


# --------------------------------------------------------------------------
# from_flat / replace_flat routing
# --------------------------------------------------------------------------

def test_from_flat_routes_every_flat_constant_into_constants():
    cfg = LatLonCGridOceanConfig.from_flat(
        g=NEMO_CONSTANTS_CONFIG.g,
        rho_0=NEMO_CONSTANTS_CONFIG.rho_0,
        omega=NEMO_CONSTANTS_CONFIG.Omega,
        c_sw=NEMO_CONSTANTS_CONFIG.c_sw,
        R_earth=NEMO_CONSTANTS_CONFIG.R_earth,
    )
    assert cfg.constants == NEMO_CONSTANTS_CONFIG
    assert cfg.g == NEMO_CONSTANTS_CONFIG.g
    assert cfg.constants.g == cfg.g            # the divergence that existed


def test_from_flat_defaults_reference_legoesm_constants():
    cfg = LatLonCGridOceanConfig.from_flat()
    assert cfg.constants == ConstantsConfig()
    assert cfg.g == constants.g
    assert cfg.rho_0 == constants.rho_ocean
    assert cfg.omega == constants.Omega


def test_from_flat_partial_pin_keeps_other_constants_at_default():
    cfg = LatLonCGridOceanConfig.from_flat(g=NEMO_CONSTANTS_CONFIG.g)
    assert cfg.constants.g == NEMO_CONSTANTS_CONFIG.g
    assert cfg.constants.rho_0 == ConstantsConfig().rho_0


def test_from_flat_conflicting_flat_and_constants_raises():
    with pytest.raises(ValueError, match="conflicting ocean constants"):
        LatLonCGridOceanConfig.from_flat(
            g=VEROS_CONSTANTS_CONFIG.g, constants=NEMO_CONSTANTS_CONFIG)


def test_from_flat_agreeing_flat_and_constants_is_accepted():
    cfg = LatLonCGridOceanConfig.from_flat(
        g=NEMO_CONSTANTS_CONFIG.g, rho_0=NEMO_CONSTANTS_CONFIG.rho_0,
        constants=NEMO_CONSTANTS_CONFIG)
    assert cfg.constants == NEMO_CONSTANTS_CONFIG


def test_replace_flat_routes_and_cannot_diverge():
    cfg = LatLonCGridOceanConfig.from_flat(physics=OceanPhysicsConfig())
    out = cfg.replace_flat(g=NEMO_CONSTANTS_CONFIG.g)
    assert out.g == NEMO_CONSTANTS_CONFIG.g
    assert out.constants.g == NEMO_CONSTANTS_CONFIG.g
    assert out.physics.constants.g == NEMO_CONSTANTS_CONFIG.g


def test_replace_flat_rejects_disagreeing_spellings():
    cfg = LatLonCGridOceanConfig.from_flat()
    with pytest.raises(ValueError, match="conflicting ocean constants"):
        cfg.replace_flat(g=VEROS_CONSTANTS_CONFIG.g,
                         constants=NEMO_CONSTANTS_CONFIG)


def test_replace_flat_accepts_agreeing_spellings_like_from_flat():
    """ONE contract, not two.

    ``replace_flat`` used to raise on the mere PRESENCE of both spellings while
    ``from_flat`` raised only on a DISAGREEMENT -- same routing, two rules.
    Redundant is not a bug; contradictory is.
    """
    cfg = LatLonCGridOceanConfig.from_flat()
    out = cfg.replace_flat(g=NEMO_CONSTANTS_CONFIG.g,
                           constants=NEMO_CONSTANTS_CONFIG)
    assert out.constants is NEMO_CONSTANTS_CONFIG      # identity self-pin kept
    assert out.g == NEMO_CONSTANTS_CONFIG.g


def test_from_flat_adopts_a_physics_only_constants_pin():
    """Passing the constants ONCE, on ``physics``, is passing them once.

    It used to RAISE "pass them once" against a ``ConstantsConfig()`` default
    nobody asked for -- asymmetric with ``from_flat(constants=X,
    physics=OceanPhysicsConfig())``, which propagates X downward.  Leaving the
    model on the default instead would recreate the two-surface divergence.
    """
    cfg = LatLonCGridOceanConfig.from_flat(
        physics=OceanPhysicsConfig(constants=VEROS_CONSTANTS_CONFIG))
    assert cfg.constants is VEROS_CONSTANTS_CONFIG
    assert cfg.g == VEROS_CONSTANTS_CONFIG.g
    assert cfg.physics.constants == cfg.constants
    # a genuine two-system contradiction still raises
    with pytest.raises(ValueError, match="conflicting ocean constants"):
        LatLonCGridOceanConfig.from_flat(
            constants=NEMO_CONSTANTS_CONFIG,
            physics=OceanPhysicsConfig(constants=VEROS_CONSTANTS_CONFIG))


def test_constants_routing_stays_jit_and_grad_safe():
    """``from_flat``/``replace_flat`` must survive a TRACED constant.

    ``constants_config`` advertises ``g``/``rho_0``/``c_sw`` as differentiable
    pytree leaves and CLAUDE.md's trainable-override doctrine applies overrides
    INSIDE the loss, so a pinned constant can be a tracer.  Both entry points
    were tracer-safe before the single-surface refactor (they did no config
    comparison at all); the conflict checks must not take that away.

    SYNTHETIC VIOLATION: replace ``constants_equal(...) is False`` with a plain
    ``!=`` in ``_physics_with_constants`` / ``from_flat`` / ``replace_flat`` and
    the ``jax.jit`` calls below raise ``TracerBoolConversionError``.
    """
    base = LatLonCGridOceanConfig.from_flat(physics=OceanPhysicsConfig())

    def via_from_flat(g):
        return LatLonCGridOceanConfig.from_flat(
            g=g, physics=OceanPhysicsConfig()).physics.constants.g

    def via_replace_flat(g):
        return base.replace_flat(g=g).physics.constants.g

    for fn in (via_from_flat, via_replace_flat):
        assert float(jax.jit(fn)(VEROS_CONSTANTS_CONFIG.g)) == pytest.approx(
            VEROS_CONSTANTS_CONFIG.g)
        assert float(jax.grad(fn)(VEROS_CONSTANTS_CONFIG.g)) == pytest.approx(1.0)


def test_a_traced_physics_pin_is_never_silently_replaced():
    """The UNDECIDABLE-comparison case that a wrong ``None`` convention breaks.

    Passing the constants once, on ``physics``, with a TRACED value.  Eager is
    fine (concrete compare), so only a jit trace exposes it.  An earlier
    revision used ``is not False`` in ``_physics_with_constants`` ("not provably
    pinned -> overwrite") together with ``is False`` in ``from_flat``'s adopt
    ("not provably non-default -> don't adopt"); the pair threw the traced pin
    away and returned the legoESM DEFAULT g under jit -- no error, dead
    gradient.  The rule (see :func:`constants_equal`) is: an undecidable
    comparison NEVER discards a value the caller supplied.

    SYNTHETIC VIOLATION: flip the adopt guard in ``from_flat`` back to
    ``is False``, or ``_physics_with_constants``'s default test back to
    ``is not False``, and the jit assertions below return ``constants.g``.
    """
    def build(gval):
        phys = OceanPhysicsConfig(constants=ConstantsConfig(g=gval))
        cfg = LatLonCGridOceanConfig.from_flat(physics=phys)
        return cfg.constants.g, cfg.physics.constants.g

    pin = VEROS_CONSTANTS_CONFIG.g
    assert build(pin) == (pin, pin)                                   # eager
    for i in (0, 1):                        # model surface AND physics surface
        fn = lambda g, _i=i: jnp.asarray(build(g)[_i])
        assert float(jax.jit(fn)(pin)) == pytest.approx(pin), (
            "a traced physics-only constants pin was silently replaced")
        assert float(jax.grad(fn)(pin)) == pytest.approx(1.0)

    # ...and replace_flat must agree with from_flat on the physics-only pin
    # (they used to disagree: from_flat adopted, replace_flat raised).
    out = LatLonCGridOceanConfig.from_flat().replace_flat(
        physics=OceanPhysicsConfig(constants=VEROS_CONSTANTS_CONFIG))
    assert out.constants is VEROS_CONSTANTS_CONFIG
    assert out.physics.constants is VEROS_CONSTANTS_CONFIG

    # The propagation helper itself, on the path where from_flat's adopt CANNOT
    # rescue it: the model constants are already pinned to something else, so
    # the undecidable compare lands directly on the "is it on the defaults?"
    # test.  Concretely this raises (a real contradiction); traced it must LEAVE
    # THE CALLER'S PIN ALONE rather than substitute the model value.
    from legoesm.ocean.state import _physics_with_constants  # white-box

    def helper(gv):
        phys = OceanPhysicsConfig(constants=ConstantsConfig(g=gv))
        return jnp.asarray(
            _physics_with_constants(phys, NEMO_CONSTANTS_CONFIG).constants.g)

    with pytest.raises(ValueError, match="conflicting ocean constants"):
        helper(pin)                                             # concrete
    assert float(jax.jit(helper)(pin)) == pytest.approx(pin), (
        "undecidable compare overwrote an explicitly pinned traced constant")


def test_replace_flat_tolerates_an_explicit_constants_none():
    """``constants=None`` used to die with ``_replace() got multiple values``."""
    cfg = LatLonCGridOceanConfig.from_flat()
    out = cfg.replace_flat(g=VEROS_CONSTANTS_CONFIG.g, constants=None)
    assert out.constants.g == VEROS_CONSTANTS_CONFIG.g


def test_flat_fields_and_flat_get_cover_the_constants():
    cfg = LatLonCGridOceanConfig.from_flat(constants=VEROS_CONSTANTS_CONFIG)
    flat = LatLonCGridOceanConfig.flat_fields()
    for name in ("g", "rho_0", "omega", "c_sw", "R_earth"):
        assert name in flat
    assert cfg.flat_get("g") == VEROS_CONSTANTS_CONFIG.g
    assert cfg.flat_get("omega") == VEROS_CONSTANTS_CONFIG.Omega
    assert cfg.flat_get("c_sw") == VEROS_CONSTANTS_CONFIG.c_sw


# --------------------------------------------------------------------------
# Propagation into the physics pipeline (the surface that carried the bug)
# --------------------------------------------------------------------------

def test_from_flat_propagates_constants_into_physics():
    cfg = LatLonCGridOceanConfig.from_flat(
        g=NEMO_CONSTANTS_CONFIG.g, physics=OceanPhysicsConfig())
    assert cfg.physics.constants.g == NEMO_CONSTANTS_CONFIG.g
    assert cfg.physics.constants == cfg.constants


def test_from_flat_contradictory_physics_constants_raises():
    with pytest.raises(ValueError, match="conflicting ocean constants"):
        LatLonCGridOceanConfig.from_flat(
            constants=NEMO_CONSTANTS_CONFIG,
            physics=OceanPhysicsConfig(constants=VEROS_CONSTANTS_CONFIG))


def test_model_validation_rejects_a_raw_constructor_divergence():
    """The raw NamedTuple constructor bypasses from_flat -- the model catches it."""
    from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig

    def _phys(cc):
        # lateral mixing is a dynamics-level concern on the lat-lon C-grid
        return OceanPhysicsConfig(
            lateral_mixing=LateralMixingConfig(scheme="none"), constants=cc)

    bad = LatLonCGridOceanConfig(
        constants=NEMO_CONSTANTS_CONFIG, physics=_phys(VEROS_CONSTANTS_CONFIG))
    with pytest.raises(ValueError, match="physical constants disagree"):
        LatLonCGridOceanModel._validate_config(bad)
    ok = LatLonCGridOceanConfig(
        constants=NEMO_CONSTANTS_CONFIG, physics=_phys(NEMO_CONSTANTS_CONFIG))
    LatLonCGridOceanModel._validate_config(ok)          # no raise

    # ...and the guard uses ``constants_equal(...) is False``, not a bare
    # ``!=``: a TRACED constant must not turn this domain check into a
    # TracerBoolConversionError.  SYNTHETIC VIOLATION: restore the ``!=`` and
    # this raises TypeError instead of passing.
    def _validate_traced(g):
        # DISTINCT tracer objects on the two surfaces (``g`` vs ``1.0 * g``):
        # with the same object, tuple comparison identity-shortcuts and never
        # calls __eq__, so the bare ``!=`` would pass and prove nothing.
        LatLonCGridOceanModel._validate_config(LatLonCGridOceanConfig(
            constants=ConstantsConfig(g=g),
            physics=_phys(ConstantsConfig(g=1.0 * g))))
        return g
    assert float(jax.jit(_validate_traced)(constants.g)) == pytest.approx(
        constants.g)


# --------------------------------------------------------------------------
# The reported defect: the DINO/NEMO oracle card's g must reach BOTH surfaces
# --------------------------------------------------------------------------

def test_run_manifest_round_trip():
    """The #376 manifest codec: a post-fix config round-trips exactly."""
    from legoesm.ocean.config import ocean_config_from_dict, ocean_config_to_dict

    cfg = LatLonCGridOceanConfig.from_flat(
        g=NEMO_CONSTANTS_CONFIG.g, rho_0=NEMO_CONSTANTS_CONFIG.rho_0,
        omega=NEMO_CONSTANTS_CONFIG.Omega)
    encoded = ocean_config_to_dict(cfg)
    assert "g" not in encoded and encoded["constants"]["g"] == cfg.g
    assert ocean_config_from_dict(encoded) == cfg


# The ``(g, rho_0, Omega)`` triple the PRE-fix ``LatLonCGridOceanConfig`` left
# on each surface when a card did not pin it.  Pre-fix the flat FIELD defaults
# and the ``ConstantsConfig`` defaults were the same three ``legoesm.constants``
# values, which is exactly why "which surface is pinned" is decidable.
_DEFAULT_TRIPLE = (constants.g, constants.rho_ocean, constants.Omega)
_NEMO_TRIPLE = (NEMO_CONSTANTS_CONFIG.g, NEMO_CONSTANTS_CONFIG.rho_0,
                NEMO_CONSTANTS_CONFIG.Omega)
_VEROS_TRIPLE = (VEROS_CONSTANTS_CONFIG.g, VEROS_CONSTANTS_CONFIG.rho_0,
                 VEROS_CONSTANTS_CONFIG.Omega)

# GENUINE pre-fix manifest constants, per card.  NOT hand-written: these are the
# ``constants`` sub-block and the top-level flat triple that the PRE-refactor
# ``_encode_config`` emitted, read out of a real encoder run at commit
# 832a1b0c3 (a detached worktree at HEAD placed ahead of the editable install on
# ``PYTHONPATH``, then ``ocean_config_to_dict`` on each card's model config).
#
# The decisive property the earlier version of this test missed: a real pre-fix
# manifest carries BOTH spellings, because ``g``/``rho_0``/``omega`` were
# NamedTuple fields AND ``constants`` was a field.  Popping ``constants``
# produced a shape the pre-fix encoder NEVER emitted, so the test passed while
# every real artifact raised.
#
# ``expected`` is the triple the POST-fix card produces -- i.e. what a
# reconciled reload must reproduce.
_PRE_FIX_MANIFEST_CONSTANTS = {
    # card: (flat triple, constants triple, expected triple)
    # DINO cards pinned rho_0 on the FLAT surface only.
    "dino:legoesm_default": (
        (constants.g, NEMO_CONSTANTS_CONFIG.rho_0, constants.Omega),
        _DEFAULT_TRIPLE,
        (constants.g, NEMO_CONSTANTS_CONFIG.rho_0, constants.Omega)),
    "dino:veros": (
        (constants.g, NEMO_CONSTANTS_CONFIG.rho_0, constants.Omega),
        _DEFAULT_TRIPLE,
        (constants.g, NEMO_CONSTANTS_CONFIG.rho_0, constants.Omega)),
    # ...and this one pinned all THREE on the flat surface.
    "dino:nemo_dino_kamm_mlf": (_NEMO_TRIPLE, _DEFAULT_TRIPLE, _NEMO_TRIPLE),
    # The OTHER direction: the card pinned ``constants=VEROS_CONSTANTS_CONFIG``
    # and left the flat ``omega`` on the LatLonCGridOceanConfig DEFAULT.  A
    # blanket "flat wins" rule would silently downgrade Veros' Omega here --
    # which is why the rule is "the non-default value wins", per constant.
    "veros_acc": (
        (VEROS_CONSTANTS_CONFIG.g, VEROS_CONSTANTS_CONFIG.rho_0, constants.Omega),
        _VEROS_TRIPLE, _VEROS_TRIPLE),
}


def _pre_fix_encoding(flat, sub, phys_cc=None):
    """A manifest dict shaped exactly like the pre-fix encoder's output: the
    ``constants`` sub-dict AND the top-level flat spellings, both present.

    ``phys_cc`` is what the manifest recorded under ``physics.constants``.  The
    default is ``ConstantsConfig()`` because that is what all four genuine
    pre-fix encodings actually carry -- MEASURED, and it is the #1226 bug
    itself: the pre-fix cards never propagated their pin into the physics
    pipeline, which is the whole reason the refactor exists.  It is a parameter
    so the non-default case (a post-fix-shaped or hand-edited manifest) can be
    exercised too.
    """
    from legoesm.ocean.config import ocean_config_to_dict

    g, rho_0, omega = sub
    enc = ocean_config_to_dict(LatLonCGridOceanConfig(
        constants=ConstantsConfig(g=g, rho_0=rho_0, Omega=omega),
        physics=OceanPhysicsConfig(constants=phys_cc or ConstantsConfig())))
    assert "constants" in enc                      # the shape that was missing
    enc["g"], enc["rho_0"], enc["omega"] = flat
    return enc


@pytest.mark.parametrize("card", sorted(_PRE_FIX_MANIFEST_CONSTANTS))
def test_pre_fix_manifest_reloads_with_the_pinned_constants(card):
    """Every pre-fix ocean run manifest must still load (the B1 blocker).

    ``restart.py`` -> ``_deserialize_config`` -> ``ocean_config_from_dict``
    raises BEFORE ``config_hash_matches`` runs, so a raise here kills
    ``legoesm reproduce``, manifest validation and every restart-chain link on
    an existing ocean run directory.
    """
    import warnings

    from legoesm.ocean.config import ocean_config_from_dict

    from legoesm.ocean.config import LegacyOceanConstantsWarning

    flat, sub, expected = _PRE_FIX_MANIFEST_CONSTANTS[card]
    encoded = _pre_fix_encoding(flat, sub)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        restored = ocean_config_from_dict(encoded)
    assert (restored.constants.g, restored.constants.rho_0,
            restored.constants.Omega) == expected
    # the resolved set reaches the physics pipeline too (nothing left behind)
    assert restored.physics.constants == restored.constants
    assert any(issubclass(w.category, LegacyOceanConstantsWarning)
               for w in caught), "the reconciliation must announce itself"


def test_pre_fix_manifest_decodes_but_its_recorded_hash_no_longer_matches():
    """B1's real end state -- pinned so nobody over-claims it as closed.

    The decode is fixed (above), but ``validate_run_manifest``
    (``driver/restart.py``) then re-hashes the REBUILT config, and the refactor
    REMOVED three top-level keys from the encoding.  ``config_hash_matches``
    tolerates schema GROWTH only, so a pre-fix ocean manifest still fails
    validation -- now with the honest "config_hash does not match" provenance
    error instead of a bogus "conflicting ocean constants".  Closing that needs
    a shrink case in ``config_hash_matches`` or a one-shot manifest migration,
    NOT a codec change.
    """
    import hashlib
    import json
    import warnings

    from legoesm.driver.restart import config_hash_matches
    from legoesm.ocean.config import ocean_config_from_dict

    flat, sub, _ = _PRE_FIX_MANIFEST_CONSTANTS["dino:nemo_dino_kamm_mlf"]
    encoded = _pre_fix_encoding(flat, sub)
    stored_hash = hashlib.sha256(
        json.dumps(encoded, sort_keys=True).encode()).hexdigest()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rebuilt = ocean_config_from_dict(encoded)          # decode SUCCEEDS
    assert rebuilt.constants.g == flat[0]
    assert config_hash_matches(stored_hash, encoded, rebuilt, "ocean") is False


def test_pre_fix_manifest_with_two_pinned_values_is_ambiguous_and_raises():
    """Non-default on BOTH surfaces and disagreeing: no defensible winner."""
    from legoesm.ocean.config import ocean_config_from_dict

    encoded = _pre_fix_encoding(
        (VEROS_CONSTANTS_CONFIG.g, constants.rho_ocean, constants.Omega),
        (NEMO_CONSTANTS_CONFIG.g, constants.rho_ocean, constants.Omega))
    with pytest.raises(ValueError, match="two DIFFERENT pinned values"):
        ocean_config_from_dict(encoded)


def test_reconciliation_against_a_recorded_non_default_physics_constants():
    """A manifest whose ``physics.constants`` is itself pinned.

    All four GENUINE pre-fix encodings record ``physics.constants`` on the
    DEFAULTS -- measured; that IS the #1226 bug (the cards' pins never reached
    the pipeline).  But a hand-edited or post-fix-shaped dict can carry a pin,
    and the reconciliation feeds its result through ``_physics_with_constants``,
    so both outcomes must be deliberate rather than incidental.
    """
    import warnings

    from legoesm.ocean.config import ocean_config_from_dict

    # (a) recorded physics pin AGREES with what the reconciliation resolves ->
    #     loads, and the resolved set is kept.
    agreeing = _pre_fix_encoding(_VEROS_TRIPLE, _DEFAULT_TRIPLE,
                                 phys_cc=ConstantsConfig(*_VEROS_TRIPLE[:2],
                                                         constants.c_sw,
                                                         _VEROS_TRIPLE[2],
                                                         constants.R_earth))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = ocean_config_from_dict(agreeing)
    assert cfg.constants.g == VEROS_CONSTANTS_CONFIG.g
    assert cfg.physics.constants == cfg.constants

    # (b) recorded physics pin CONTRADICTS it -> loud, never a silent pick.
    conflicting = _pre_fix_encoding(_NEMO_TRIPLE, _DEFAULT_TRIPLE,
                                    phys_cc=VEROS_CONSTANTS_CONFIG)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="conflicting ocean constants"):
            ocean_config_from_dict(conflicting)


def test_manifest_without_a_constants_block_still_routes_the_flat_names():
    """Manifests older than ``ConstantsConfig`` itself (flat spellings only)."""
    from legoesm.ocean.config import ocean_config_from_dict, ocean_config_to_dict

    encoded = ocean_config_to_dict(LatLonCGridOceanConfig.from_flat())
    encoded.pop("constants")
    encoded.update(g=NEMO_CONSTANTS_CONFIG.g, rho_0=NEMO_CONSTANTS_CONFIG.rho_0,
                   omega=NEMO_CONSTANTS_CONFIG.Omega)
    restored = ocean_config_from_dict(encoded)
    assert restored.constants.g == NEMO_CONSTANTS_CONFIG.g
    assert restored.constants.rho_0 == NEMO_CONSTANTS_CONFIG.rho_0
    assert restored.constants.Omega == NEMO_CONSTANTS_CONFIG.Omega


def test_every_constants_field_is_reachable_from_a_card():
    """Alias-keyed coverage: a ConstantsConfig field with NO flat spelling is
    unroutable, so a card can pin it only by hand-building a ConstantsConfig --
    which is exactly how ``c_p`` -> ``c_sw`` was missed (the names differ, so
    a field-name-matching check sees nothing wrong).
    """
    from legoesm.ocean.state import CONSTANTS_FLAT_ALIASES

    assert set(CONSTANTS_FLAT_ALIASES.values()) == set(ConstantsConfig._fields)
    pinned = {v: i + 1.0 for i, v in enumerate(ConstantsConfig._fields)}
    cfg = LatLonCGridOceanConfig.from_flat(
        physics=OceanPhysicsConfig(),
        **{flat: pinned[field]
           for flat, field in CONSTANTS_FLAT_ALIASES.items()})
    assert cfg.constants == ConstantsConfig(**pinned)
    assert cfg.physics.constants == ConstantsConfig(**pinned)


def test_dino_returned_physics_config_is_the_routed_one():
    """The SECOND return value must not be the pre-routing local.

    ``create_forcings()["physics_config"]`` (dino.py) and
    ``neverworld2_lite.create_forcings`` hand that object straight to callers;
    returning the un-propagated local gave them a pipeline still on the
    ConstantsConfig DEFAULTS while the model ran on the card's values.
    """
    from legoesm.grids.latlon import create_mercator_grid
    from legoesm.ocean.experiments.dino import (
        DINO_RECIPES, DINOConfig, dino_lat_lon_model_config,
    )
    grid = create_mercator_grid(n_lon=16, lat_max_deg=70.0,
                                lon_west_deg=0.0, lon_east_deg=50.0)
    cfg = DINOConfig(**DINO_RECIPES["nemo_dino_kamm_mlf"])
    model_cfg, returned_physics = dino_lat_lon_model_config(grid, cfg)
    assert returned_physics is model_cfg.physics
    assert returned_physics.constants == model_cfg.constants
    assert returned_physics.constants.g == NEMO_CONSTANTS_CONFIG.g


def test_dino_card_specific_heat_reaches_the_constants():
    """``DINOConfig.c_p`` -> ``ConstantsConfig.c_sw`` across the name change."""
    from legoesm.grids.latlon import create_mercator_grid
    from legoesm.ocean.experiments.dino import (
        DINO_RECIPES, DINOConfig, dino_lat_lon_model_config,
    )
    grid = create_mercator_grid(n_lon=16, lat_max_deg=70.0,
                                lon_west_deg=0.0, lon_east_deg=50.0)
    cfg = DINOConfig(**DINO_RECIPES["nemo_dino_kamm_mlf"])
    assert cfg.c_p == NEMO_CONSTANTS_CONFIG.c_sw     # the card's own pin
    model_cfg, physics_cfg = dino_lat_lon_model_config(grid, cfg)
    assert model_cfg.constants.c_sw == cfg.c_p
    assert physics_cfg.constants.c_sw == cfg.c_p


def test_nemo_dino_kamm_card_pins_g_on_every_surface():
    from legoesm.ocean.experiments.dino import (
        DINO_RECIPES, DINOConfig, dino_lat_lon_model_config,
    )
    from legoesm.grids.latlon import create_mercator_grid
    grid = create_mercator_grid(n_lon=16, lat_max_deg=70.0,
                                lon_west_deg=0.0, lon_east_deg=50.0)
    cfg = DINOConfig(**DINO_RECIPES["nemo_dino_kamm_mlf"])
    model_cfg, phys_cfg = dino_lat_lon_model_config(grid, cfg)
    assert phys_cfg.constants == model_cfg.constants
    assert model_cfg.g == NEMO_CONSTANTS_CONFIG.g
    assert model_cfg.constants.g == NEMO_CONSTANTS_CONFIG.g
    assert model_cfg.physics.constants.g == NEMO_CONSTANTS_CONFIG.g
    # rho_0 / omega ride the same routing.
    assert model_cfg.constants.rho_0 == cfg.rho_0
    assert model_cfg.constants.Omega == cfg.omega
    assert model_cfg.physics.constants.Omega == cfg.omega


# --------------------------------------------------------------------------
# enhanced_diffusion: the N^2 trigger's g was unpinnable by any config
# --------------------------------------------------------------------------

def _unstable_column():
    z = create_ocean_z_star(n_levels=6, H_max=4000.0)
    J = jnp.ones((2, 2), dtype=jnp.float64)
    T = jnp.broadcast_to(
        jnp.linspace(18.0, 2.0, z.n_levels), (2, 2, z.n_levels))
    T = T.at[..., 0].set(0.0)                     # cold dense cap -> N^2 < 0
    rho = constants.rho_ocean - 0.2 * (T - 4.0)
    return z, J, rho


def test_convective_K_A_flag_honours_the_g_it_is_given():
    """N^2 = -(g/rho_ref) drho/dz, so the smooth trigger must move with g."""
    z, J, rho = _unstable_column()
    cfg = EnhancedDiffusionConfig(
        K_conv=1.0, K_bg=1e-5, nu_conv=0.0, nu_bg=0.0,
        smooth_transition=True, sigmoid_sharpness=1e3,
    )
    K_ref, _, _ = convective_K_A_flag(rho, z.dz_ref, J, cfg,
                                      g=constants.g)
    K_nemo, _, _ = convective_K_A_flag(rho, z.dz_ref, J, cfg,
                                       g=NEMO_CONSTANTS_CONFIG.g)
    K_half, _, _ = convective_K_A_flag(rho, z.dz_ref, J, cfg,
                                       g=0.5 * constants.g)
    # NEMO's g differs from legoESM's by 5.0e-5 relative -> a tiny but nonzero
    # response; halving g is a large one. Both prove g is actually consumed
    # (before this fix the kernel took no g at all and all three were equal).
    assert not bool(jnp.allclose(K_ref, K_nemo, rtol=0.0, atol=0.0))
    assert float(jnp.max(jnp.abs(K_half - K_ref))) > 1e-6


def test_convective_K_A_flag_honours_rho_ref():
    z, J, rho = _unstable_column()
    cfg = EnhancedDiffusionConfig(
        K_conv=1.0, K_bg=1e-5, nu_conv=0.0, nu_bg=0.0,
        smooth_transition=True, sigmoid_sharpness=1e3,
    )
    K_a, _, _ = convective_K_A_flag(rho, z.dz_ref, J, cfg, rho_ref=1025.0)
    K_b, _, _ = convective_K_A_flag(rho, z.dz_ref, J, cfg, rho_ref=512.5)
    assert float(jnp.max(jnp.abs(K_a - K_b))) > 1e-6


def test_convection_factory_threads_the_recipe_constants():
    """``make_convection_physics`` must pass ``constants_config`` to the kernel."""
    from legoesm.ocean.physics.convection.config import OceanConvectionConfig
    from legoesm.ocean.physics.convection.integration import (
        make_convection_physics,
    )
    z, J, rho = _unstable_column()
    conv = OceanConvectionConfig(
        scheme="enhanced_diffusion",
        enhanced_diffusion=EnhancedDiffusionConfig(
            K_conv=1.0, K_bg=1e-5, nu_conv=0.0, nu_bg=0.0,
            smooth_transition=True, sigmoid_sharpness=1e3,
        ),
    )
    fns = [make_convection_physics(conv, apply_diffusion=False,
                                   constants_config=cc)
           for cc in (ConstantsConfig(), ConstantsConfig(g=0.5 * constants.g))]
    from legoesm.core.field import Field
    from legoesm.ocean.state import OceanState
    shape = (2, 2, z.n_levels)
    T = jnp.broadcast_to(jnp.linspace(18.0, 2.0, z.n_levels), shape)
    T = T.at[..., 0].set(0.0)
    S = jnp.full(shape, 35.0)
    zero3 = jnp.zeros(shape)
    zero2 = jnp.zeros((2, 2))
    state = OceanState(
        u=Field(zero3), v=Field(zero3), T=Field(T), S=Field(S),
        eta=Field(zero2), H_bathy=Field(jnp.full((2, 2), 4000.0)),
        land_mask=Field(jnp.ones((2, 2))),
    )
    K = [fn(state, None, z).K_v for fn in fns]
    assert K[0] is not None and K[1] is not None
    assert float(jnp.max(jnp.abs(K[0] - K[1]))) > 1e-6
