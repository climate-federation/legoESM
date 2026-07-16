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

KNOWN-UNGUARDED (verified, deliberately NOT gated here)
-------------------------------------------------------
These split the same interface but cannot be fixed without an answer-changing
physics decision, so they are documented rather than silently absorbed:

  * ``gustiness_w_zi``: ``CouplerConfig.gustiness_w_zi`` is ``float = 0.0`` and
    so CANNOT express "scheme-native", while the atmosphere's
    ``SurfaceLayerConfig.gustiness_w_zi`` is ``float | None = None``, which
    ``bulk_flux`` resolves to ``_COARE_GUSTINESS_ZI = 600.0`` for coare3. So
    ``--surface-bulk-scheme coare3`` WITHOUT ``--gustiness-zi`` still splits:
    atmosphere 600 m vs tile 0.0 (off). This is a DEFAULT-PATH failure, not a
    hypothetical. Fixing it means making the coupler field nullable, which flips
    the tile default from off to on (coupled answers change) and drops the field
    out of the ``:float``-eligible ``__param_spec__`` set. Pinned by a strict
    xfail in tests/unit/test_air_sea_scheme_consistency.py.
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
    guard cannot drift from the model.  Returns ``None`` when the active
    turbulence scheme carries no surface sub-config (``turbulence="none"``, or a
    scheme such as clubb whose sub-config has no ``surface``) -- there is then no
    atmosphere-side surface layer to compare against.

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
    from legoesm.driver.physics_pipeline import (
        apply_surface_flux_config,
        turbulence_config_for,
    )

    tc = apply_surface_flux_config(turbulence_config_for(atm_config), atm_config)
    sub = getattr(tc, getattr(tc, "scheme", ""), None)
    if sub is None:
        return None
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
