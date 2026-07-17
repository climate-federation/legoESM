"""The atmosphere surface layer and the coupler OCEAN tile parameterize the
SAME air-sea interface, so their configs must agree.

Two configs describe one interface:

  * the ATMOSPHERE surface layer -- the ``SurfaceLayerConfig`` nested inside the
    active ``TurbulenceConfig`` sub-config, reached through the production
    resolver (``turbulence_config_for`` + ``apply_surface_flux_config``);
  * the coupler OCEAN tile -- ``CouplerConfig``, consumed by
    ``coupler.ocean_tile_response``.

When they disagree, the latent heat the ocean loses is not the moisture flux the
atmosphere gains: nothing fails, the interface just quietly stops conserving.
That is precisely the failure mode a norm check cannot see, so it is gated here
instead.

Validated on STATIC Python config values at driver construction, never inside a
traced body -- these are all `str`/`float` leaves of a config NamedTuple, so the
comparison is a compile-time Python `!=`, not a `jnp.where`.

The LAND, SLAB and LAKE tiles legitimately run their own scheme
(``--land-bulk-scheme`` / ``--slab-bulk-scheme``) against their own surface, and
are deliberately out of scope: this guards the air-sea interface only.

Why "reject" and not "derive"
-----------------------------
Silently deriving the tile config from the atmosphere would mean guessing at
physics across fields whose defaults intentionally differ (see KNOWN-UNGUARDED
below) -- exactly the kind of quiet coupling this module exists to forbid. Fail
loud, per the repo's dispatch-hardening doctrine.

What this guard does NOT prove
-----------------------------
Passing it means the two sides SELECT the same scheme and feed it the same
solver settings.  It does NOT prove the two sides compute an identical flux, and
must not be read that way (codex): the coupler tile always applies a 0.98 saline
factor via ``ocean_surface_q_sat`` while the atmosphere builds a FRESH-water
q_sat, and under ``aerobulk`` MOST the tile switches to Goff where the
atmosphere does not.  Those predate this module and are genuine physics
questions (seawater q_sat really is ~0.98x fresh), not config drift -- which is
exactly why they are out of scope for a config-consistency gate.  Scheme
equality is a NECESSARY condition for a conserving interface, not a sufficient
one.

KNOWN-UNGUARDED (verified, deliberately NOT gated here)
-------------------------------------------------------
These split the same interface but cannot be fixed without an answer-changing
physics decision, so they are documented rather than silently absorbed:

  * the constant-closure coefficients: atmosphere ``Cd_neutral``/``Ch_neutral``
    (both 1.5e-3) vs coupler ``Cd_ocean``/``Ch_ocean`` (both 1.5e-3) match by
    default but are differently NAMED, and the light-wind floors differ by
    design (atmosphere 0.01 m/s numerical floor vs coupler ``gustiness = 1.0``
    m/s Wing-2018 sub-grid floor, whose docstring explains the 1.0 default is a
    tuned slab-drift compromise). So matched ``"constant"`` schemes still do not
    imply identical light-wind fluxes.
"""
from __future__ import annotations

# Selects WHICH algorithm runs on each side.  Always compared: it is the one
# axis both closures read.
_SCHEME_AXIS = "bulk_scheme"

# Read ONLY by the MOST solver, on BOTH sides (same field name, same default on
# SurfaceLayerConfig and CouplerConfig).  Compared only once the interface is
# actually on a MOST scheme: the constant closure reads none of them
# (surface_layer.py passes them exclusively inside its `bulk_scheme in
# ("coare3","large_yeager")` branch and its constant branch reads only
# Cd_neutral/Ch_neutral; coupler.py gates them behind `_is_most` and
# ocean_surface_q_sat deliberately keeps the constant closure on Tetens).  So
# comparing them under "constant" would REJECT a config that runs identically on
# both sides -- e.g. `run_coupled --bulk-thermo-convention aerobulk`, which is
# inert under the default constant closure.  Verified: an earlier draft of this
# module compared them unconditionally and did exactly that.
_MOST_ONLY_AXES = ("thermo_convention", "stability_scheme", "z_ref",
                   "bulk_n_iter")

# The two sides do NOT agree on what counts as MOST.  The ATMOSPHERE dispatches
# MOST on ("coare3","large_yeager") only (surface_layer.py), while the coupler
# ocean tile also accepts "most" (coupler.py `_is_most`).  Since bulk_scheme must
# match, "most" is the single value where the predicates diverge -- and there it
# means the atmosphere silently falls through to the CONSTANT branch while the
# tile runs the MOST solver.  That is a split even though the strings are equal,
# so it is rejected outright (see _reject_most below).  ExperimentConfig.
# validate_strict already rejects surface_bulk_scheme="most" for this reason;
# a turbulence_override can still smuggle it into the nested surface config.
_ATM_MOST_SCHEMES = ("coare3", "large_yeager")


def resolve_effective_atm_surface(atm_config):
    """The ``SurfaceLayerConfig`` the ATMOSPHERE will ACTUALLY run.

    Resolved through the SAME production path the physics pipeline uses, so the
    guard cannot drift from the model.  Returns ``None`` ONLY for
    ``turbulence="none"``, which genuinely has no surface layer to compare
    against; every other scheme -- clubb included, once its default sub-config is
    materialized -- carries a real ``SurfaceLayerConfig``.

    Reading the DECLARED ``ExperimentConfig.surface_bulk_scheme`` instead would
    miss a real split: ``apply_surface_flux_config`` returns the
    ``turbulence_override`` UNCHANGED when every experiment-level surface field
    is default, and ``validate_strict`` only requires the override to share the
    active turbulence *scheme* -- never its nested surface settings.  So

        ExperimentConfig(surface_bulk_scheme="constant",   # declared
                         turbulence_override=...surface.bulk_scheme="coare3")

    declares "constant" and runs COARE3.  Comparing the declared field would
    wave that straight through, which is the same declared-vs-materialized bug
    this guard was written to catch, one layer up (codex).
    """
    # Deferred: physics_pipeline is heavy and imports across the driver package.
    from legoesm.atmosphere.physics.turbulence.integration import (
        materialize_sub_config,
    )
    from legoesm.driver.physics_pipeline import turbulence_config_for

    # `turbulence_config_for` ALREADY ends in `apply_surface_flux_config`, so
    # calling that again here would double-apply it -- idempotent, but then this
    # would not literally be the pipeline's path, which is the entire point of
    # resolving through production code (codex).
    tc = turbulence_config_for(atm_config)
    # `None` sub-config does not mean "no surface layer": dispatch substitutes a
    # default (CLUBBConfig() for clubb), and that default carries a real
    # constant surface config it runs fluxes with. Reading the None straight
    # would skip validation on a config that DOES have a surface -- e.g.
    # turbulence="clubb" + CouplerConfig(bulk_scheme="coare3") is a genuine
    # split (atmosphere constant vs tile COARE3) the guard must catch (codex).
    tc = materialize_sub_config(tc)
    sub = getattr(tc, getattr(tc, "scheme", ""), None)
    if sub is None:
        return None  # scheme="none": genuinely no surface layer
    return getattr(sub, "surface", None)


def validate_air_sea_consistency(atm_config, coupler_config) -> None:
    """Raise if the atmosphere surface layer and the coupler ocean tile disagree.

    ``coupler_config=None`` is NOT skipped.  Both drivers materialize a bare
    ``CouplerConfig()`` when they get ``None``
    (``coupler_cfg = self._coupler_config or CouplerConfig()``), and its
    ``bulk_scheme`` is "constant" REGARDLESS of the atmosphere -- so

        CoupledESMDriver(ExperimentConfig(surface_bulk_scheme="coare3"))

    produced exactly the split this guard exists to catch.  An earlier version
    returned early on ``None``, on the claim that the driver's defaults are
    "self-consistent by construction"; they are not.  Validate against the
    config the driver will ACTUALLY build.
    """
    from legoesm.coupler.config import CouplerConfig

    surf = resolve_effective_atm_surface(atm_config)
    if surf is None:
        return
    effective = coupler_config if coupler_config is not None else CouplerConfig()

    atm_scheme = getattr(surf, _SCHEME_AXIS, None)
    if atm_scheme == "most":
        raise ValueError(
            "air-sea bulk-flux scheme mismatch -- the atmosphere surface layer "
            "has bulk_scheme='most', which it does NOT implement: "
            "surface_layer.compute_surface_fluxes dispatches MOST on "
            f"{_ATM_MOST_SCHEMES} only, so 'most' silently falls through to the "
            "constant-coefficient branch while the coupler ocean tile DOES run "
            "the MOST solver for it. The two sides would run different physics "
            "on one interface despite the matching name. Use 'coare3' or "
            "'large_yeager' (ExperimentConfig.validate_strict rejects "
            "surface_bulk_scheme='most' for the same reason; a "
            "turbulence_override can still reach the nested surface config)."
        )

    axes = [_SCHEME_AXIS]
    if atm_scheme in _ATM_MOST_SCHEMES:
        axes += list(_MOST_ONLY_AXES)

    split = []
    for axis in axes:
        atm_v = getattr(surf, axis, None)
        ocn_v = getattr(effective, axis, None)
        if atm_v is None or ocn_v is None:
            continue  # a side that does not carry this axis cannot split on it
        if atm_v != ocn_v:
            split.append((axis, atm_v, ocn_v))

    # The T/q measurement heights are NOT same-named across the interface: the
    # coupler passes z_t_atm/z_q_atm explicitly, while the atmosphere's MOST call
    # omits z_t/z_q entirely, so compute_most_fluxes defaults them to its z_ref.
    # The atmosphere therefore has no field to compare -- its effective value IS
    # z_ref -- and a name-matching loop silently misses the split (codex):
    # CouplerConfig(z_ref=10, z_t_atm=2, z_q_atm=2, bulk_scheme="coare3") passes
    # while the two sides use different reference-height conventions.
    if atm_scheme in _ATM_MOST_SCHEMES:
        atm_z_ref = getattr(surf, "z_ref", None)
        for axis in ("z_t_atm", "z_q_atm"):
            ocn_v = getattr(effective, axis, None)
            if atm_z_ref is None or ocn_v is None:
                continue
            if ocn_v != atm_z_ref:
                split.append((f"{axis} (vs the atmosphere's z_ref, which its "
                              "MOST call defaults z_t/z_q to)", atm_z_ref, ocn_v))

    # Gustiness: both sides are nullable with None = scheme-native, so the
    # generic loop's skip-on-None would wave through atm None (-> 600 m for
    # coare3) vs tile 0.0 (off) — the exact default-path split that was
    # KNOWN-UNGUARDED before CouplerConfig.gustiness_w_zi became nullable.
    # Compare the EFFECTIVE z_i through the model's own resolver
    # (bulk_flux.resolve_gustiness_w_zi, the same call compute_most_fluxes
    # makes), never the raw fields.
    if atm_scheme in _ATM_MOST_SCHEMES:
        from legoesm.core.bulk_flux import resolve_gustiness_w_zi
        atm_zi = resolve_gustiness_w_zi(
            getattr(surf, "gustiness_w_zi", None), atm_scheme)
        ocn_zi = resolve_gustiness_w_zi(
            getattr(effective, "gustiness_w_zi", None), atm_scheme)
        if atm_zi != ocn_zi:
            split.append(("gustiness_w_zi (effective z_i [m]; None resolved "
                          "scheme-natively on both sides)", atm_zi, ocn_zi))
    if not split:
        return

    detail = "; ".join(
        f"{axis}: atmosphere {atm_v!r} vs ocean tile {ocn_v!r}"
        for axis, atm_v, ocn_v in split
    )
    raise ValueError(
        f"air-sea bulk-flux scheme mismatch -- {detail}. The atmosphere surface "
        "layer (the SurfaceLayerConfig of the active turbulence scheme, after "
        "ExperimentConfig.surface_* injection) and the coupler ocean tile "
        "(CouplerConfig, via coupler.ocean_tile_response) parameterize the SAME "
        "air-sea interface; a split means the ocean's latent-heat loss is not "
        "the atmosphere's moisture gain, and nothing will fail -- the interface "
        "just stops conserving. Set the CouplerConfig field(s) to match the "
        "atmosphere (run_coupled threads --surface-bulk-scheme / "
        "--bulk-thermo-convention / --surface-stability-scheme into both sides "
        "via build_coupler_config)."
    )
